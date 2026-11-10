"""Format, optimization and precision checks; CPU only."""
from pathlib import Path
import sys
import tempfile
import unittest

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binary_codec import BinaryProjection, bf16, encode, fit, pack_codes, quantize, refine_codes, unpack_codes


class BinaryTests(unittest.TestCase):
    def test_byte_order_and_roundtrip(self):
        codes = np.ones((3, 2, 128), np.float32)
        codes[0] = -1
        codes[1, :, 1::2] = -1
        p = pack_codes(codes)
        self.assertEqual(p.shape, (2, 3, 16))
        np.testing.assert_array_equal(p[0, :, 0], [0, 0x55, 0xff])
        np.testing.assert_array_equal(unpack_codes(p), codes)

    def test_single_bit_descent(self):
        rng = np.random.default_rng(3)
        d = rng.normal(size=(128,64)).astype(np.float32) / 8
        x = rng.normal(size=(256,64)).astype(np.float32)
        b = rng.choice([-1.,1.],size=(256,128)).astype(np.float32)
        before = np.square(x-b@d).sum(axis=1)
        refine_codes(x,d,b,3)
        after = np.square(x-b@d).sum(axis=1)
        self.assertTrue(np.all(after <= before + 1e-5))
        self.assertTrue(np.isin(b,[-1,1]).all())

    def test_exact_representable_codes(self):
        # Distinct coefficients 1 and 2 per coordinate give {-3,-1,1,3}.
        d = np.concatenate([np.eye(64),2*np.eye(64)]).astype(np.float32)
        rng = np.random.default_rng(4)
        b = rng.choice([-1.,1.],size=(256,128)).astype(np.float32)
        x = b@d
        got = encode(x,d,restarts=4,sweeps=8)
        np.testing.assert_array_equal(got@d,x)

    def test_fit_storage_precision_and_reproducibility(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(512,64)).astype(np.float32)
        d,h = fit(x,iterations=4)
        self.assertTrue(h[-1] <= h[0])
        np.testing.assert_array_equal(d,bf16(d))
        w = rng.normal(size=(16,128)).astype(np.float32)
        q = quantize(w,d,restarts=2)
        q2 = quantize(w,d,restarts=2)
        np.testing.assert_array_equal(q.packed,q2.packed)
        self.assertEqual(q.storage_bytes,w.size//4+16384)
        b = unpack_codes(q.packed).reshape(-1,128)
        expected = bf16(b@q.decoder.astype(np.float32)).reshape(w.shape)
        np.testing.assert_array_equal(q.dense(),expected)
        with tempfile.TemporaryDirectory() as t:
            file = Path(t)/'projection.npz'
            q.save(file)
            loaded = BinaryProjection.load(file)
            np.testing.assert_array_equal(loaded.packed,q.packed)
            np.testing.assert_array_equal(loaded.dense(),q.dense())

    def test_validation(self):
        with self.assertRaises(ValueError):
            fit(np.full((128,64),np.nan))
        with self.assertRaises(ValueError):
            pack_codes(np.zeros((2,1,128)))
        with self.assertRaises(ValueError):
            encode(np.ones((2,64)),np.ones((128,64)),restarts=0)


if __name__ == '__main__':
    unittest.main()
