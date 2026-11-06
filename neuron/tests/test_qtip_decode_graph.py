"""CPU check: dec_qtip() (vectorized torch) equals a direct transcription of decode_tile() in runtime/src/qtip.cpp."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_layer import dec_qtip


def reference(trellis, tlut, out, width):
    i0n, j0n = out // 32, width // 32
    result = np.zeros((out, width), np.float32)
    for tr in range(out // 16):
        for tc in range(width // 16):
            i0, i1, j0, j1 = tr // 2, tr % 2, tc // 2, tc % 2
            w = [int(trellis[i0, j0, c, j1, i1]) & 0xffff for c in range(32)]
            for t in range(128):
                i, s = t >> 2, 4 * (t & 3)
                u = (w[i] << 16 | w[(i + 1) & 31]) & 0xffffffff
                state = (u >> (16 - s)) & 0xffff
                e = ((state * (state + 1)) >> 6) & 1023
                c0 = tlut[e & 511, 0] * (-1.0 if e >> 9 else 1.0)
                c1 = tlut[e & 511, 1]
                A, B, C, D = t >> 4, (t >> 2) & 3, (t >> 1) & 1, t & 1
                r, col = D * 8 + A, C * 8 + B * 2
                result[tr * 16 + r, tc * 16 + col] = c0
                result[tr * 16 + r, tc * 16 + col + 1] = c1
    return result


rng = np.random.default_rng(3)
out, width = 64, 96
trellis = rng.integers(-32768, 32768, (out // 32, width // 32, 32, 2, 2), dtype=np.int16)
tlut = rng.normal(0, 1, (512, 2)).astype(np.float16).astype(np.float32)
expected = reference(trellis, tlut, out, width)
got = dec_qtip(torch.from_numpy(trellis), torch.from_numpy(tlut), out, width).float().numpy()
want = torch.from_numpy(expected).to(torch.bfloat16).float().numpy()
assert np.array_equal(got, want), f'max diff {np.abs(got - want).max()}'
print('QTIP graph decode matches decode_tile()')

# the einsum matvec (no weight de-swizzle) must agree with the dense decode
from trace_layer import qtip_matvec
u = torch.from_numpy(rng.normal(0, 1, (3, width)).astype(np.float32)).to(torch.bfloat16)
dense = dec_qtip(torch.from_numpy(trellis), torch.from_numpy(tlut), out, width).float()
want_z = (u.float() @ dense.T).numpy()
got_z = qtip_matvec(u, torch.from_numpy(trellis), torch.from_numpy(tlut), out, width).numpy()
assert np.abs(got_z - want_z).max() < 1e-2 * np.abs(want_z).max(), 'qtip_matvec differs from the dense decode'
print('qtip_matvec matches the dense decode')
