"""Trace the fused QTIP NKI kernel (or its decode-only variant) into one NEFF for cpp/layer_bench.cpp.

usage: bench_qtip_nki.py WORKDIR OUT_LOCAL IN_LOCAL T [linear|dense|ablate_gather_only|ablate_intops_only|ablate_load_only]      (single core: layer_bench ... ranks=1)
e.g. the 70B gate shard at TP=2: OUT_LOCAL=14336 IN_LOCAL=8192 (117 M weights, 29 MB of trellis words).
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qtip_nki
from qtip_nki import IN_GROUP, qtip_decode_dense, qtip_linear

workdir, out_local, in_local, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
variant = sys.argv[5] if len(sys.argv) > 5 else 'linear'
G = (in_local + IN_GROUP - 1) // IN_GROUP
rng = np.random.default_rng(0)
words = torch.from_numpy(rng.integers(0, 65536, (G, 128, out_local // 16 * 32)).astype(np.int32).astype(np.uint16).view(np.int16)).view(torch.uint16)
lut = torch.from_numpy(rng.integers(0, 2 ** 32, (128, 1024), dtype=np.uint64).astype(np.uint32).view(np.int32)).view(torch.uint32)
xT = torch.randn(128, G, 16, T).to(torch.bfloat16)
if variant == 'dense':
    fn = lambda words, lut: qtip_decode_dense(words, lut)
    inputs = (words, lut)
elif variant.startswith('ablate_'):
    kern = getattr(qtip_nki, 'qtip_' + variant)
    fn = lambda words, lut: kern(words, lut)
    inputs = (words, lut)
else:
    fn = lambda words, lut, xT: qtip_linear(words, lut, xT)
    inputs = (words, lut, xT)
torch_neuronx.trace(fn, inputs, compiler_workdir=workdir)
print('NEFF', workdir + '/graph.neff', 'weights_M', out_local * in_local / 1e6, 'words_MB', words.numel() * 2 / 1e6)
