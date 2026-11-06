from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import KernelBackend


def main():
    rng = np.random.default_rng(827)
    q = rng.normal(size=(4, 32)).astype(np.float32)
    k = rng.normal(size=(137, 2, 32)).astype(np.float32)
    v = rng.normal(size=(137, 2, 32)).astype(np.float32)
    got = KernelBackend().attention(q, k, v)
    for h in range(4):
        scores = k[:, h // 2] @ q[h] / np.sqrt(32)
        weights = np.exp(scores - scores.max())
        expected = weights @ v[:, h // 2] / weights.sum()
        np.testing.assert_allclose(got[h], expected, rtol=3e-5, atol=2e-6)
    print('NKI tiled GQA decode and online softmax match NumPy across a tile boundary')

if __name__ == '__main__': main()
