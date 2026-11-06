"""Tensor-parallel (TP=2) FP8 Llama layer PREFILL for T tokens, compiled to one collectives NEFF.

Same sharding as trace_layer_tp.py (16/32 query heads, 4/8 KV heads, half of the MLP channels per
rank, all-reduce after o_proj and down_proj), but all T prompt tokens go through one graph: causal
attention over the T x T scores per head, K/V of all T positions returned for the cache. Timing
harness: inputs are synthetic (a causal mask input replaces in-graph constants).

usage: trace_prefill_tp.py MODEL WORKDIR [T] [layer]
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_layer import DIM, HEADS, HIDDEN, KVHEADS, Fp8Weight, build_layer_fp8, cos_sin_tables, lin_fp8
from trace_layer_tp import ORDER as _DECODE_ORDER, TP, shard

ORDER = ['x', 'cos_sin', 'mask', 'norm1', 'norm2', 'qw', 'ow', 'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']


def prefill_layer(x, cos_sin, mask, norm1, norm2, qw, ow, kf, ks, vf, vs, gf, gs, uf, us, df, ds):
    import torch_xla.core.xla_model as xm
    allreduce = lambda t: xm.all_reduce(xm.REDUCE_SUM, t, groups=[list(range(TP))])
    heads, kvheads, T = HEADS // TP, KVHEADS // TP, x.shape[0]
    half, scale = DIM // 2, DIM ** -0.5
    rms = lambda t, w: (t * torch.rsqrt(torch.mean(t * t, dim=-1, keepdim=True) + 1e-5) * w).to(torch.bfloat16)
    h = rms(x, norm1)
    q = lin_fp8(h, qw).float().reshape(T, heads, DIM)
    k = lin_fp8(h, Fp8Weight(kf, ks)).float().reshape(T, kvheads, DIM)
    v = lin_fp8(h, Fp8Weight(vf, vs)).float().reshape(T, kvheads, DIM)
    cos, sin = cos_sin[:, 0].reshape(T, 1, half), cos_sin[:, 1].reshape(T, 1, half)
    def rope(t):
        a, b = t[..., :half], t[..., half:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    q, k = rope(q), rope(k)
    outs = []
    for head in range(heads):
        kh = head // (heads // kvheads)
        scores = (q[:, head].to(torch.bfloat16) @ k[:, kh].to(torch.bfloat16).T).float() * scale + mask
        outs.append(torch.softmax(scores, dim=-1).to(torch.bfloat16) @ v[:, kh].to(torch.bfloat16))
    attn = torch.stack(outs, dim=1).reshape(T, heads * DIM)
    x = x + allreduce(lin_fp8(attn, ow).float())
    h = rms(x, norm2)
    act = torch.nn.functional.silu(lin_fp8(h, Fp8Weight(gf, gs))).float() * lin_fp8(h, Fp8Weight(uf, us)).float()
    x = x + allreduce(lin_fp8(act.to(torch.bfloat16), Fp8Weight(df, ds)).float())
    return x, k, v


def main():
    model, workdir = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    layer = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    import torch_neuronx
    weights = shard(build_layer_fp8(model, layer, 'fp8'), 0)
    sample = {
        'x': torch.randn(T, HIDDEN), 'cos_sin': cos_sin_tables(T),
        'mask': torch.triu(torch.full((T, T), -1e9), diagonal=1),
    }
    inputs = tuple(sample[k] if k in sample else weights[k] for k in ORDER)
    try:
        torch_neuronx.trace(prefill_layer, inputs, compiler_workdir=workdir)
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
    m = Path(workdir) / 'model'
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(m / 'graph.hlo'), str(m / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'],
                   cwd=workdir, check=True)
    print(json.dumps({'tp': TP, 'tokens': T, 'neff': str(Path(workdir) / 'graph.neff')}))


if __name__ == '__main__':
    main()
