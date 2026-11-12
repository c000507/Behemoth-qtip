"""TP-sharded Qwen3.5 MoE NEFFs (qwen35_graphs.py output) on the NeuronCores against the CPU mirror of the same graph
(qwen35_graphs.sublayer with the ranks as threads), one launch each, through cpp/neff_io.cpp, with RANDOM weights of the
real per-rank shapes (the checkpoint is not needed: the graph numerics are isolated per block kind).

  NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 \
    python neuron/tests/test_qwen35_device.py CONFIG.json NEFF_DIR NEFF_IO_BINARY [bucket names...]
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glm_repack as RP  # noqa: E402
import qwen35_graphs as Q  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
import qwen35_prepare as P  # noqa: E402
from glm_runner import ThreadAllReduce  # noqa: E402


def layer_weights(cfg, kind, seed):
    """Random HF-layout weights of ONE layer of `kind` (vocab shrunk: embeddings / head are not part of layer graphs)."""
    c = json.loads(json.dumps(cfg))
    t = c.get('text_config', c)
    t['num_hidden_layers'] = 1
    t['layer_types'] = ['linear_attention' if kind == 'gdn' else 'full_attention']
    t['vocab_size'] = 64
    return c, R.random_weights(c, seed=seed, scale=0.02)


def main():
    cfg_path, neffs, neff_io = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    only = set(sys.argv[4:])
    cfg = json.loads(Path(cfg_path).read_text())
    worst = 0.0
    results = {}
    for d in sorted(neffs.iterdir()):
        if not (d / 'meta.json').exists() or (only and d.name not in only):
            continue
        meta = json.loads((d / 'meta.json').read_text())
        kind = meta['kind'][len('qwen35_'):]
        if kind == 'head':
            continue
        g = Q.Geo(cfg, meta['geometry']['tp'])
        parts = kind.split('_')[1:] if kind.startswith('layer_') else [kind]
        attn_kind = parts[0] if parts[0] in ('gdn', 'gqa') else 'gdn'
        c1, w = layer_weights(cfg, attn_kind, seed=7)
        get = P.Getter(w)
        g1 = Q.Geo(c1, g.TP)   # same geometry, layer 0 of the needed kind
        shards = [P.shard_layer(get, 0, r, g1, meta['fp8']) for r in range(g.TP)]
        rel = check(d, meta, kind, parts, g, shards, neff_io)
        results[d.name] = rel
        worst = max(worst, rel)
    print(json.dumps({'worst': worst, 'blocks': results}))


def check(d, meta, kind, parts, g, shards, neff_io):
    R_, Q_, P_ = meta['R'], meta['Q'], meta['P']
    T = R_ * Q_
    torch.manual_seed(1)
    x = Q.bf(torch.randn(T, g.H) * 0.5)
    start = 40
    pos = torch.tensor([start + (i % Q_) if R_ == 1 else start + i for i in range(T)], dtype=torch.int64)
    table = torch.stack([torch.arange(P_, dtype=torch.int64) + r * P_ for r in range(R_)])
    table = table.clamp(max=meta['pages'] - 2)
    wslot = torch.stack([table[i // Q_, int(pos[i]) // 16] * 16 + pos[i] % 16 for i in range(T)])
    step = {'x': x, 'pos': pos, 'wslot': wslot, 'table': table, 'slot': torch.arange(R_, dtype=torch.int64),
            'cos_sin': Q.cos_sin_rows(pos, g.rot, g.theta)}
    states = {}
    for n, (shape, dt) in Q.state_shapes(parts[0], g, meta['pages'], meta['slots']).items():
        if dt == torch.float32:
            states[n] = torch.randn(shape) * (0.05 if n == 'kstate' else 0.3)
        else:
            states[n] = (torch.randn(shape) * 0.5).to(dt)
    per_rank = []
    for r in range(g.TP):
        if len(parts) == 1:
            sub = 'attn' if kind in ('gdn', 'gqa') else 'mlp'
            w = {k[len(sub) + 1:]: v for k, v in shards[r].items() if k.startswith(sub + '.')}
        else:
            w = {('a.' if k.startswith('attn.') else 'm.') + k.split('.', 1)[1]: v for k, v in shards[r].items()}
        per_rank.append({**step, **w, **{k: v.clone() for k, v in states.items()}})
    ar, ref = ThreadAllReduce(g.TP), [None] * g.TP
    moe_mode = meta.get('moe', 'all')

    def run(r):
        t = dict(per_rank[r])
        if len(parts) == 1:
            ref[r] = Q.sublayer(kind, t, g, R_, Q_, P_, ar(r), moe_mode)[0]
        else:
            sub = lambda pre: {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
            common = {k: v for k, v in t.items() if '.' not in k}
            xa, _ = Q.sublayer(parts[0], {**common, **sub('a.')}, g, R_, Q_, P_, ar(r), moe_mode)
            ref[r] = Q.sublayer(parts[1], {**common, **sub('m.'), 'x': xa}, g, R_, Q_, P_, ar(r), moe_mode)[0]
    th = [threading.Thread(target=run, args=(r,)) for r in range(g.TP)]
    [t.start() for t in th]
    [t.join() for t in th]
    names = meta['inputs'] + meta['states']
    with tempfile.TemporaryDirectory(dir=os.environ.get('IO_TMP')) as io:
        for r in range(g.TP):
            for i, n in enumerate(names):
                Path(io, f'in{i}.r{r}').write_bytes(RP.tensor_bytes(per_rank[r][n]))
        env = dict(os.environ, NEFF_IO_STATES=str(len(meta['states'])))
        t0 = time.time()
        subprocess.run([neff_io, str(d / 'graph.neff'), io, str(g.TP)], check=True, stdout=subprocess.DEVNULL, env=env)
        dt = time.time() - t0
        out = np.frombuffer(Path(io, 'out0.r0').read_bytes(), np.float32).reshape(T, -1)
        y_dev = torch.from_numpy(out.copy()) - x
        y_ref = ref[0] - x
        rel = float(torch.linalg.norm(y_dev - y_ref) / torch.linalg.norm(y_ref))
        same = max([float(np.abs(np.frombuffer(Path(io, f'out0.r{r}').read_bytes(), np.float32).reshape(T, -1) - out).max()) for r in range(1, g.TP)] or [0.0])
    print(f'{d.name:22s} {kind:14s} R={R_} Q={Q_} P={P_} ({"fp8" if meta["fp8"] else "bf16"}, moe={moe_mode}): block output rel L2 vs CPU mirror '
          f'{rel:.3e}; ranks differ by {same:.1e}; neff_io {dt:.1f}s', flush=True)
    return rel


if __name__ == '__main__':
    main()
