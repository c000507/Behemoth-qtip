"""Micro NEFFs that isolate the fixed cost of a launch and of an in-graph all-reduce over TP ranks (layer_bench runs them):
  python bench_allreduce_graph.py OUT TP T K [rms]   ->  OUT/ar_T_K[_rms]/graph.neff
x [T, 4096] f32 -> K times: (x * 1.0001 [+ RMSNorm]) -> all_reduce(sum) / TP. K = 0: no collective at all.
"""
import os, subprocess, sys
from pathlib import Path
import torch

out, tp, T, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rms = len(sys.argv) > 5
import torch_neuronx
import torch_xla.core.xla_model as xm
os.environ.setdefault('XLA_ALWAYS_ALLREDUCE', '1')
groups = [list(range(tp))]


def fn(x, w):
    for _ in range(K):
        y = x * 1.0001
        if rms:
            y = y * torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + 1e-6) * w
        x = xm.all_reduce(xm.REDUCE_SUM, y, groups=groups) * (1.0 / tp)
    return x + w if K == 0 else x + 0.0 * w


d = Path(out) / f'ar_{T}_{K}{"_rms" if rms else ""}'
d.mkdir(parents=True, exist_ok=True)
torch_neuronx.trace(fn, (torch.zeros(T, 4096), torch.ones(4096)), compiler_workdir=str(d))
subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'], cwd=d, check=True, stdout=subprocess.DEVNULL)
print('built', d)
