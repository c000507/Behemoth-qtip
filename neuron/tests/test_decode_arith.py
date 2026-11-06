"""CPU check: the gather-free E4M3 / E2M1 decoders in trace_layer.py match ml_dtypes / the NVFP4 table."""
from pathlib import Path
import sys

import ml_dtypes
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_layer import dec_nvfp4, e2m1_to_f32, e4m3_to_f32

codes = np.arange(256, dtype=np.uint8)
ref = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
finite = np.isfinite(ref)
got = e4m3_to_f32(torch.from_numpy(codes)).numpy()
assert np.array_equal(got[finite], ref[finite]), 'E4M3 decode mismatch'

positive = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
table = np.concatenate((positive, -positive))
assert np.array_equal(e2m1_to_f32(torch.arange(16)).numpy(), table), 'E2M1 decode mismatch'

rng = np.random.default_rng(1)
out, width = 8, 64
packed = rng.integers(0, 256, (out, width // 2), dtype=np.uint8)
scales = rng.integers(0, 0x78, (out, width // 16), dtype=np.uint8)  # finite E4M3 scales
glob = np.float32(0.0123)
lo, hi = packed & 15, packed >> 4
nib = np.empty((out, width), np.uint8)
nib[:, 0::2], nib[:, 1::2] = lo, hi
expect = (table[nib] * np.repeat(scales.view(ml_dtypes.float8_e4m3fn).astype(np.float32), 16, axis=1) * glob)
expect = expect.astype(ml_dtypes.bfloat16).astype(np.float32)
actual = dec_nvfp4(torch.from_numpy(packed), torch.from_numpy(scales), torch.tensor([glob])).float().numpy()
assert np.array_equal(actual, expect), 'NVFP4 decode mismatch'
print('decode arithmetic tests passed')
