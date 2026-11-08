"""TP-sharded GLM sub-block NEFFs (glm_build_buckets.sh output, FP8 or bf16 weights) on the NeuronCores against the CPU
mirror of the same graph (glm_graphs.sublayer with the ranks as threads), one launch each, through cpp/neff_io.cpp.
Isolates the device numerics (FP8 tensor-engine matmuls, collectives) of every block kind from end-to-end amplification.

  NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 \
    python neuron/tests/test_glm_fp8_device.py CHECKPOINT NEFF_DIR NEFF_IO_BINARY [layers...]
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glm_graphs as G
import glm_repack as RP
from glm_runner import ThreadAllReduce


def main():
    model, neffs, neff_io = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    cfg = json.loads((Path(model) / 'config.json').read_text())
    worst = 0.0
    torch.manual_seed(0)
    for d in sorted(neffs.iterdir()):
        if not (d / 'meta.json').exists():
            continue
        meta = json.loads((d / 'meta.json').read_text())
        kind = meta['kind'][4:]
        if kind == 'head':
            continue
        geo = meta['geometry']
        g = G.Geo(cfg, geo['tp'], geo['ep'])
        fp8 = meta['fp8']
        pk = RP.Packer(RP.Checkpoint(model), g, fp8)
        R, Q, P = meta['R'], meta['Q'], meta['P']
        T = R * Q
        layer = next(l for l in range(g.L) if (('kda' if g.attn[l] == 'linear_attention' else 'mla') == kind or g.mlp[l] == kind))
        sub = 'attn' if kind in ('kda', 'mla') else 'mlp'
        for start in ((16, 0) if Q > 1 else (40, 8)) if kind == 'mla' else ((16,) if Q > 1 else (40,)):
            worst = max(worst, check_block(d, meta, kind, g, pk, R, Q, P, T, layer, sub, start, neff_io))
    print('worst', worst)


def check_block(d, meta, kind, g, pk, R, Q, P, T, layer, sub, start, neff_io):
    """One launch at positions start.. (MLA: start 40 / 16 selects pools, 8 / 0 sees every key: no selection effect)."""
    fp8 = meta['fp8']
    x = G.bf(torch.randn(T, g.S * g.H) * 0.5)
    pos = torch.tensor([start + (i % Q) if R == 1 else start for i in range(T)], dtype=torch.int64)
    table = torch.stack([torch.arange(P, dtype=torch.int64) + r * P for r in range(R)])
    wslot = torch.stack([table[i // Q, int(pos[i]) // 16] * 16 + pos[i] % 16 for i in range(T)])
    step = {'x': x, 'pos': pos, 'wslot': wslot, 'table': table, 'slot': torch.arange(R, dtype=torch.int64)}
    kv8 = meta.get('kv') == 'fp8'
    states = {}
    for n, (shape, dt) in G.state_shapes(kind, g, meta['pages'], meta['slots'], kv8).items():
        if dt == torch.float32 and n != 'lsc':
            states[n] = torch.randn(shape) * (0.05 if n == 'kstate' else 0.3)
        elif n == 'lsc':
            states[n] = torch.rand(shape) * 0.01 + 1e-3
        else:
            states[n] = (torch.randn(shape) * (0.5 if dt == torch.bfloat16 else 50)).clamp(-200, 200).to(dt)
    per_rank = []
    for r in range(g.TP):
        w = {k[len(sub) + 1:]: v for k, v in pk.layer(layer, r).items() if k.startswith(sub + '.')}
        per_rank.append({**step, **w, **{k: v.clone() for k, v in states.items()}})
    # CPU mirror, ranks as threads
    ar, ref = ThreadAllReduce(g.TP), [None] * g.TP

    def run(r):
        ref[r] = G.sublayer(kind, dict(per_rank[r]), g, R, Q, P, ar(r), kv8, meta.get('moe', 'all'))
    th = [threading.Thread(target=run, args=(r,)) for r in range(g.TP)]
    [t.start() for t in th]
    [t.join() for t in th]
    names = meta['inputs'] + meta['states']
    with tempfile.TemporaryDirectory() as io:
        for r in range(g.TP):
            for i, n in enumerate(names):
                Path(io, f'in{i}.r{r}').write_bytes(RP.tensor_bytes(per_rank[r][n]))
        env = dict(os.environ, NEFF_IO_STATES=str(len(meta['states'])))  # states aliased in place, as in the runtime
        subprocess.run([neff_io, str(d / 'graph.neff'), io, str(g.TP)], check=True, stdout=subprocess.DEVNULL, env=env)
        out = np.frombuffer(Path(io, 'out0.r0').read_bytes(), np.float32).reshape(T, -1)
        y_dev = torch.from_numpy(out.copy()) - x
        y_ref = ref[0][0] - x
        rel = float(torch.linalg.norm(y_dev - y_ref) / torch.linalg.norm(y_ref))
        other = np.frombuffer(Path(io, 'out0.r1').read_bytes(), np.float32).reshape(T, -1) if g.TP > 1 else out
        same = float(np.abs(other - out).max())
    print(f'{d.name:16s} layer {layer} ({"fp8" if fp8 else "bf16"}) pos {start}: block output rel L2 vs CPU mirror {rel:.3e}; '
          f'ranks differ by {same:.1e}', flush=True)
    return rel


if __name__ == '__main__':
    main()
