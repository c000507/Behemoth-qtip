"""Trace the fused NVFP4 NKI projection into one NEFF for the C++ timing harness (cpp/layer_bench.cpp).

usage: bench_nki_nvfp4.py WORKDIR [K] [N] [B]   (single core: run layer_bench with ranks=1)
"""
from pathlib import Path
import sys

import ml_dtypes
import numpy as np
import torch
import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nki_linear import nvfp4_linear, nvfp4_pack

workdir = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
B = int(sys.argv[4]) if len(sys.argv) > 4 else 1
rng = np.random.default_rng(0)
packed = rng.integers(0, 256, (N, K // 2), dtype=np.uint8)
scales = rng.integers(0x28, 0x50, (N, K // 16), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
wp, sce, sco, ex = nvfp4_pack(packed, scales, 0.02)
t = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.uint16 if a.dtype == ml_dtypes.bfloat16 else a.dtype)).view(
    torch.bfloat16 if a.dtype == ml_dtypes.bfloat16 else torch.uint8)


def f(xT, wp, sce, sco, ex):
    return nvfp4_linear(xT, wp, sce, sco, ex)


inputs = (torch.randn(K, B).to(torch.bfloat16), t(wp), t(sce), t(sco), t(ex))
torch_neuronx.trace(f, inputs, compiler_workdir=workdir)
print('NEFF', workdir + '/graph.neff', 'weights_MB', round((wp.nbytes + sce.nbytes + sco.nbytes) / 1e6, 2))
