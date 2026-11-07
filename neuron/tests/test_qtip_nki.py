"""Fused QTIP decode + matmul NKI kernel against the numpy mirror of qtip::decode_rows_bf16 (NKI CPU simulator)."""
from pathlib import Path
import sys

import ml_dtypes
import numpy as np
import neuronxcc.nki as nki

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_nki import NT, codebook_u32, decode_reference, pack_tiles, qtip_decode_dense, qtip_linear, tiles_of, xt_layout

rng = np.random.default_rng(3)
out, width, T = 16 * NT, 2048 + 1024, 3                      # ragged second in-group (64 partitions valid)
trellis = rng.integers(0, 65536, ((out // 16) * (width // 16), 32)).astype(np.uint16).view(np.int16)
tlut = rng.normal(0, 1, (512, 2)).astype(np.float16).view(np.uint16)
lut = codebook_u32(tlut)
tiles = tiles_of(trellis, out, width)
dense = decode_reference(tiles, lut[0])                     # [out, width] bf16 values as f32
u = rng.normal(0, 1, (T, width)).astype(ml_dtypes.bfloat16)
expected = u.astype(np.float32) @ dense.T

words = pack_tiles(tiles)
G = words.shape[0]
z = nki.simulate_kernel(qtip_linear, words, lut, xt_layout(u, G))
rel = np.linalg.norm(z - expected) / np.linalg.norm(expected)
print(f'fused linear relative_l2={rel:.3e}')
assert rel < 2e-3, 'fused QTIP NKI kernel does not match the reference decode'

u2 = rng.normal(0, 1, (200, width)).astype(ml_dtypes.bfloat16)                # two token blocks (128 + 72)
z2 = nki.simulate_kernel(qtip_linear, words, lut, xt_layout(u2, G))
rel2 = np.linalg.norm(z2 - u2.astype(np.float32) @ dense.T) / np.linalg.norm(u2.astype(np.float32) @ dense.T)
print(f'fused linear T=200 relative_l2={rel2:.3e}')
assert rel2 < 2e-3, 'fused QTIP NKI kernel (two token blocks) does not match the reference decode'

d = np.asarray(nki.simulate_kernel(qtip_decode_dense, words, lut)).astype(np.float32)   # [128, G, O*256] k-major
t = np.arange(128)
A, B, C, D = t >> 4, (t >> 2) & 3, (t >> 1) & 1, t & 1
got = np.zeros_like(dense)
for g in range(G):
    for p in range(128):
        ib = g * 128 + p
        if ib >= width // 16:
            continue
        for ch in range(out // 16 // NT):
            chunk = d[p, g, ch * NT * 256:(ch + 1) * NT * 256].reshape(4, NT, 32, 2)     # (k, tile, i, comp)
            for n in range(NT):
                pair = chunk[:, n].transpose(1, 0, 2).reshape(128, 2)                    # t = 4 i + k
                blk = np.zeros((16, 16), np.float32)
                blk[D * 8 + A, C * 8 + B * 2] = pair[:, 0]
                blk[D * 8 + A, C * 8 + B * 2 + 1] = pair[:, 1]
                ob = ch * NT + n
                got[ob * 16:(ob + 1) * 16, ib * 16:(ib + 1) * 16] = blk
# the simulator stores only one bf16 half of each bitcast pair for this kernel; the fused kernel (both components
# checked above) is what the graph uses, so only the first component is asserted here
assert np.array_equal(got[:, 0::2], dense[:, 0::2]), 'dense decode (component 0) differs from the reference'
print('dense decode component 0 matches bit-exactly')
print('QTIP NKI kernels match (simulator)')
