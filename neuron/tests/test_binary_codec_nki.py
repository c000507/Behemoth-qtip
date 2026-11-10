"""Fused binary decoder/matmul vs independently unpacked NumPy weights.

Simulator verifies multi-tile indexing, byte order, accumulation and BF16
rounding; it cannot establish device speed or hardware numerical correctness.
"""
from pathlib import Path
import sys
import ml_dtypes
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binary_codec_nki import binary_linear, nki


def main():
    rng = np.random.default_rng(321)
    for out,width,tokens in [(128,64,1),(256,192,8),(128,128,128)]:
        packed = rng.integers(0,256,(width//64,out,16),dtype=np.uint8)
        packed[:,0,0] = 0
        packed[:,0,1] = 255
        d = (rng.normal(size=(128,64))/8).astype(ml_dtypes.bfloat16)
        x = rng.normal(size=(width,tokens)).astype(ml_dtypes.bfloat16)
        weights = np.empty((out,width),np.float32)
        # Independent scalar bit enumeration, not the codec's unpack helper.
        for g in range(width//64):
            b = np.stack([((packed[g,:,j//8].astype(np.int32) >> (j%8)) & 1)*2-1
                          for j in range(128)],axis=1).astype(np.float32)
            weights[:,g*64:(g+1)*64] = (b @ d.astype(np.float32)).astype(ml_dtypes.bfloat16).astype(np.float32)
        expected = x.astype(np.float32).T @ weights.T
        if hasattr(nki,'simulate_kernel'):
            got = nki.simulate_kernel(binary_linear,packed,d,x)
        else:
            got = nki.simulate(binary_linear)(packed,d,x)
        rel = np.linalg.norm(got-expected)/np.linalg.norm(expected)
        print(f'{out=} {width=} {tokens=} relative_l2={rel:.3e}')
        assert rel < 1e-5, 'Binary NKI kernel differs from independently decoded BF16 weights'
    print('Binary NKI simulator: PASS')


if __name__ == '__main__':
    main()
