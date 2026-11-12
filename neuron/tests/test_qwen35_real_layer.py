"""Real-weight check of ONE Qwen3.5 MoE layer: the f32 oracle block (qwen35_moe_ref, HF-validated) vs the fp8 / bf16 CPU
mirror of the sharded graphs vs the fused layer NEFF on the NeuronCores, same random activations (a layer of the real
checkpoint, read straight from model.safetensors; no end-to-end run, so no embeddings needed).

  NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 \
    python neuron/tests/test_qwen35_real_layer.py MODEL_DIR NEFF_DIR NEFF_IO layer [layer...]
MODEL_DIR holds config.json + model.safetensors (the HF layout); NEFF_DIR the layer_gdn_moe_R_Q_P / layer_gqa_moe_R_Q_P
buckets (R = 1, Q = 1 are used). Prints rel L2: device vs mirror (device numerics), mirror vs oracle (fp8 loss), device vs oracle.
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


def layer_names(g, l):
    p = f'layers.{l}.'
    names = [p + 'input_layernorm.weight', p + 'post_attention_layernorm.weight']
    if g.kind(l) == 'gdn':
        names += [p + 'linear_attn.' + n for n in ('A_log', 'conv1d.weight', 'dt_bias', 'in_proj_a.weight', 'in_proj_b.weight',
                                                   'in_proj_qkv.weight', 'in_proj_z.weight', 'norm.weight', 'out_proj.weight')]
    else:
        names += [p + 'self_attn.' + n for n in ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight', 'q_norm.weight', 'k_norm.weight')]
    names += [p + 'mlp.' + n for n in ('gate.weight', 'experts.gate_up_proj', 'experts.down_proj', 'shared_expert.gate_proj.weight',
                                       'shared_expert.up_proj.weight', 'shared_expert.down_proj.weight', 'shared_expert_gate.weight')]
    return names


def rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def main():
    model, neffs, neff_io = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    layers = [int(a) for a in sys.argv[4:]] or [0, 3]
    cfg = json.loads((model / 'config.json').read_text())
    ck = RP.Checkpoint(model)
    results = {}
    for l in layers:
        for d in sorted(neffs.iterdir()):
            if not (d / 'meta.json').exists():
                continue
            meta = json.loads((d / 'meta.json').read_text())
            g = Q.Geo(cfg, meta['geometry']['tp'])
            if meta['kind'] != f'qwen35_layer_{g.kind(l)}_moe' or meta['R'] != 1 or meta['Q'] != 1:
                continue
            t0 = time.time()
            w = {ck.prefix + n: ck.tensor(ck.prefix + n) for n in layer_names(g, l)}
            ref = R.Ref(cfg, w)
            get = P.Getter(w)
            shards = [P.shard_layer(get, l, r, g, meta['fp8']) for r in range(g.TP)]
            print(f'layer {l} ({g.kind(l)}) weights read + sharded in {time.time() - t0:.0f}s', flush=True)
            results[f'L{l}/{d.name}'] = check(d, meta, l, g, ref, shards, neff_io)
    print(json.dumps(results))


def check(d, meta, l, g, ref, shards, neff_io):
    P_ = meta['P']
    torch.manual_seed(1)
    x = Q.bf(torch.randn(1, g.H) * 1.0)
    pos = torch.tensor([40], dtype=torch.int64)
    table = torch.arange(P_, dtype=torch.int64).reshape(1, P_).clamp(max=meta['pages'] - 2)
    wslot = torch.tensor([int(table[0, 40 // 16]) * 16 + 40 % 16])
    step = {'x': x, 'pos': pos, 'wslot': wslot, 'table': table, 'slot': torch.tensor([0]), 'cos_sin': Q.cos_sin_rows(pos, g.rot, g.theta)}
    # oracle (f32, exact): a state after 40 tokens does not exist here, so both sides start from the SAME random state
    st_or = {}
    states = {}
    for n, (shape, dt) in Q.state_shapes(g.kind(l), g, meta['pages'], meta['slots']).items():
        states[n] = (torch.randn(shape) * (0.05 if n == 'kstate' else 0.3)) if dt == torch.float32 else (torch.randn(shape) * 0.5).to(dt)
    if g.kind(l) == 'gdn':
        kd = g.Hk * g.Dk
        S = torch.zeros(g.Hv, g.Dk, g.Dv)
        conv = torch.zeros(2 * kd + g.Hv * g.Dv, g.K - 1)
        for r in range(g.TP):   # rank r's heads / channels of the shared random slot 0
            S[r * g.hl:(r + 1) * g.hl] = states['kstate'][0]
            rows = torch.cat((r * g.hk * g.Dk + torch.arange(g.hk * g.Dk), kd + r * g.hk * g.Dk + torch.arange(g.hk * g.Dk), 2 * kd + r * g.Wv + torch.arange(g.Wv)))
            conv[rows] = states['cstate'][0]
        st_or = {'S': S, 'conv': conv}
    else:
        # the oracle attends over the cache's keys 0..40 of pages table[0, :]: rebuild them as its past k / v (f32 of the stored bf16)
        kv = states['kv'][table[0]].float()                                              # [P, 2, kvl, 16, D]
        keys = kv.permute(1, 0, 3, 2, 4).reshape(2, P_ * 16, g.kvl, g.D)[:, :40]
        K = torch.zeros(40, g.kvh, g.D); V = torch.zeros(40, g.kvh, g.D)
        for r in range(g.TP):
            K[:, r * g.kvl:(r + 1) * g.kvl] = keys[0]; V[:, r * g.kvl:(r + 1) * g.kvl] = keys[1]
        st_or = {'k': K, 'v': V}
    n = R.rms(x, ref.t(f'layers.{l}.input_layernorm.weight'), g.eps)
    y = ref.linear_attn(l, n, st_or) if g.kind(l) == 'gdn' else ref.full_attn(l, n, pos, st_or)
    x1 = x + y
    n2 = R.rms(x1, ref.t(f'layers.{l}.post_attention_layernorm.weight'), g.eps)
    y_or = x1 + ref.moe(l, n2) - x
    # the mirror and the device
    per_rank = []
    for r in range(g.TP):
        w = {('a.' if k.startswith('attn.') else 'm.') + k.split('.', 1)[1]: v for k, v in shards[r].items()}
        per_rank.append({**step, **w, **{k: v.clone() for k, v in states.items()}})
    ar, out = ThreadAllReduce(g.TP), [None] * g.TP

    def run(r):
        t = dict(per_rank[r])
        sub = lambda pre: {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
        common = {k: v for k, v in t.items() if '.' not in k}
        xa, _ = Q.sublayer(g.kind(l), {**common, **sub('a.')}, g, 1, 1, P_, ar(r), meta.get('moe', 'all'))
        out[r] = Q.sublayer('moe', {**common, **sub('m.'), 'x': xa}, g, 1, 1, P_, ar(r), meta.get('moe', 'all'))[0]
    th = [threading.Thread(target=run, args=(r,)) for r in range(g.TP)]
    [t.start() for t in th]
    [t.join() for t in th]
    y_mir = out[0] - x
    names = meta['inputs'] + meta['states']
    with tempfile.TemporaryDirectory(dir=os.environ.get('IO_TMP')) as io:
        for r in range(g.TP):
            for i, nm in enumerate(names):
                Path(io, f'in{i}.r{r}').write_bytes(RP.tensor_bytes(per_rank[r][nm]))
        env = dict(os.environ, NEFF_IO_STATES=str(len(meta['states'])))
        subprocess.run([neff_io, str(d / 'graph.neff'), io, str(g.TP)], check=True, stdout=subprocess.DEVNULL, env=env)
        y_dev = torch.from_numpy(np.frombuffer(Path(io, 'out0.r0').read_bytes(), np.float32).reshape(1, -1).copy()) - x
    res = {'device_vs_mirror': rel(y_dev, y_mir), 'mirror_vs_oracle': rel(y_mir, y_or), 'device_vs_oracle': rel(y_dev, y_or),
           'layer_out_norm': float(torch.linalg.norm(y_or))}
    print(f'layer {l} {d.name} ({"fp8" if meta["fp8"] else "bf16"}): ' + ', '.join(f'{k} {v:.3e}' for k, v in res.items()), flush=True)
    return res


if __name__ == '__main__':
    main()
