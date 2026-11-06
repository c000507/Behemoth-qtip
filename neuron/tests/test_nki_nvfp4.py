"""NVFP4 NKI kernel vs a numpy reference, on the NKI CPU simulator (no device needed)."""
from pathlib import Path
import sys

import ml_dtypes
import numpy as np
import neuronxcc.nki as nki

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nki_linear import nvfp4_linear, nvfp4_pack

rng = np.random.default_rng(7)
K, N, B = 256, 512, 4
packed = rng.integers(0, 256, (N, K // 2), dtype=np.uint8)
scales = rng.integers(0x28, 0x50, (N, K // 16), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
gscale = 0.02
x = rng.normal(0, 1, (B, K)).astype(ml_dtypes.bfloat16)

pos = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
table = np.concatenate((pos, -pos))
codes = np.empty((N, K), np.uint8)
codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
dense = table[codes] * np.repeat(scales.astype(np.float32), 16, axis=1) * gscale
expected = x.astype(np.float32) @ dense.astype(ml_dtypes.bfloat16).astype(np.float32).T

wp, sce, sco, ex = nvfp4_pack(packed, scales, gscale)
y = nki.simulate_kernel(nvfp4_linear, np.ascontiguousarray(x.T), wp, sce, sco, ex)
rel = np.linalg.norm(y - expected) / np.linalg.norm(expected)
print(f'relative_l2={rel:.3e}')
assert rel < 5e-3, 'NVFP4 NKI kernel does not match the reference'
print('NVFP4 NKI kernel matches (simulator)')
