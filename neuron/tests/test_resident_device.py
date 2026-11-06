"""Device-only check that resident weight tensors track their source arrays.

Two distinct weight arrays sharing one kernel signature must each keep their
own device tensor, and repeated calls must follow the numpy contents even
after other arrays used the same signature in between.
"""
from pathlib import Path
import sys
import numpy as np
import ml_dtypes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nki_cache import KernelCache
from ops import fp8fn_project_loop

rng = np.random.default_rng(97)
lut = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(1, 256)
x = rng.normal(size=(1, 64)).astype(np.float32)
scale = np.ones((1, 32), np.float32)
def codes(seed):
    r = np.random.default_rng(seed)
    return r.choice(np.array([0, 1, 63, 119, 120, 121, 126, 128, 248, 254], np.uint8),
                    size=(1, 32 * 64))
w1, w2 = codes(11), codes(12)
cache = KernelCache()
first = cache.run(fp8fn_project_loop, w1, scale, x, lut, 64, 32, 1, resident=(0, 1, 3))
second = cache.run(fp8fn_project_loop, w2, scale, x, lut, 64, 32, 1, resident=(0, 1, 3))
np.testing.assert_allclose(first, fp8fn_project_loop(w1, scale, x, lut, 64, 32, 1), rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(second, fp8fn_project_loop(w2, scale, x, lut, 64, 32, 1), rtol=1e-6, atol=1e-6)
again = cache.run(fp8fn_project_loop, w1, scale, x, lut, 64, 32, 1, resident=(0, 1, 3))
np.testing.assert_allclose(again, first, rtol=1e-6, atol=1e-6)
# A different array object with equal content is a distinct resident tensor and
# must produce the same values. (In-place mutation of an already-resident array
# is not tracked: resident arrays must be treated as immutable.)
copy = w1.copy()
same = cache.run(fp8fn_project_loop, copy, scale, x, lut, 64, 32, 1, resident=(0, 1, 3))
np.testing.assert_allclose(same, first, rtol=1e-6, atol=1e-6)
assert len(cache._entries) == 1, 'same signature must reuse the compiled kernel'
state = next(iter(cache._entries.values()))
assert len(state.resident) == 5, 'w1, w2, the copy, the shared scale and the lut get one resident tensor each'
# Persisted NEFF cache: a fresh cache over the same directory must load the
# manifest + NEFF without recompiling and reproduce the same values.
import tempfile
import nki_cache
with tempfile.TemporaryDirectory() as directory:
    persisted = KernelCache(cache_dir=directory)
    np.testing.assert_allclose(persisted.run(fp8fn_project_loop, w1, scale, x, lut, 64, 32, 1, resident=(0, 1, 3)),
                               first, rtol=1e-6, atol=1e-6)
    reloaded = KernelCache(cache_dir=directory)
    result = reloaded.run(fp8fn_project_loop, w1, scale, x, lut, 64, 32, 1, resident=(0, 1, 3))
    assert isinstance(reloaded._entries[tuple(reloaded._entries)[0]].compiled,
                      nki_cache._LoadedKernel), 'second cache must load the persisted NEFF'
    np.testing.assert_allclose(result, first, rtol=1e-6, atol=1e-6)
print('resident tensors track per-array weights correctly')
