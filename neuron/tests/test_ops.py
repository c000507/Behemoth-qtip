"""CPU simulation checks against independent NumPy expressions."""
from pathlib import Path
import sys
import numpy as np
import nki

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops import rms_norm, swiglu, apply_rope, attention_chunk, fp8_project_loop, nvfp4_project_loop
import ml_dtypes


def main():
    rng = np.random.default_rng(791)
    x = rng.normal(size=(3, 64)).astype(np.float32)
    w = rng.normal(size=(1, 64)).astype(np.float32)
    norm = nki.simulate(rms_norm)(x, w, 1e-5)
    expected = x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(norm, expected, rtol=2e-5, atol=1e-6)
    up = rng.normal(size=x.shape).astype(np.float32)
    act = nki.simulate(swiglu)(x, up)
    np.testing.assert_allclose(act, x / (1 + np.exp(-x)) * up, rtol=2e-5, atol=1e-6)
    angles = rng.normal(size=(3, 32)).astype(np.float32)
    c, s = np.cos(angles), np.sin(angles)
    rotary = nki.simulate(apply_rope)(x, c, s)
    exp_rotary = np.concatenate((x[:, :32]*c-x[:, 32:]*s, x[:, 32:]*c+x[:, :32]*s), axis=1)
    np.testing.assert_allclose(rotary, exp_rotary, rtol=2e-5, atol=1e-6)
    q = rng.normal(size=(1, 32)).astype(np.float32)
    k = rng.normal(size=(32, 32)).astype(np.float32)
    v = rng.normal(size=(32, 32)).astype(np.float32)
    mask = np.zeros((1, 32), np.float32); mask[:, 19:] = -1e30
    stats = nki.simulate(attention_chunk)(q, k, v, mask, 32**-0.5)
    logits = q @ k.T * 32**-0.5 + mask
    exp = np.exp(logits - np.max(logits))
    expected = exp @ v / np.sum(exp)
    np.testing.assert_allclose(stats[:, :32] / stats[:, 33:34], expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(stats[:, 32:33], np.max(logits, axis=1, keepdims=True), rtol=1e-5, atol=1e-6)
    fw = rng.normal(size=(32, 64)).astype(ml_dtypes.float8_e4m3fn)
    fs = rng.uniform(.1, 1.2, size=(1, 32)).astype(np.float32)
    fx = rng.normal(size=(2, 64)).astype(ml_dtypes.bfloat16).astype(np.float32)
    got = nki.simulate(fp8_project_loop)(fw.reshape(1, -1), fs, fx.reshape(1, -1), 64, 32, 2).reshape(2, 32)
    expected = (fx @ fw.astype(np.float32).T) * fs
    np.testing.assert_allclose(got, expected, rtol=3e-5, atol=2e-6)
    codes = rng.integers(0, 16, size=(16,64), dtype=np.uint8)
    packed = (codes[:,::2] | (codes[:,1::2] << 4)).reshape(1,-1)
    block_scales = rng.uniform(.1,1.2,size=(16,4)).astype(ml_dtypes.float8_e4m3fn)
    ix = (np.arange(16,dtype=np.uint32)//2).reshape(1,16)
    shifts = ((np.arange(16,dtype=np.uint32)%2)*4).reshape(1,16)
    pos = np.array([0,.5,1,1.5,2,3,4,6],np.float32)
    lut = np.concatenate((pos,-pos)).reshape(1,16)
    global_scale = .17
    got = nki.simulate(nvfp4_project_loop)(packed, block_scales.reshape(1,-1), fx.reshape(1,-1),
                                            ix, shifts, lut, global_scale,64,16,2).reshape(2,16)
    dense = lut.reshape(-1)[codes] * np.repeat(block_scales.astype(np.float32),16,axis=1) * global_scale
    dense = dense.astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_allclose(got, fx @ dense.T, rtol=3e-5, atol=2e-6)
    print("NKI RMSNorm, SwiGLU, RoPE, attention, FP8 and NVFP4 projection match NumPy")


if __name__ == '__main__':
    main()
