"""Compare the NKI simulator with the repository's independent C++ reference."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import ml_dtypes
import numpy as np
import nki

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_kernel import decode_block, project_block, project_rotated, project_rotated_loop, transform_factored, transform_pow2
from test_checkpoint import write_checkpoint
from checkpoint import Checkpoint
from backend import KernelBackend
from qtip_layout import factored_hadamard_indices, hadamard_indices, indices, projection_indices, reconstruct



def hadamard(x):
    result = np.asarray(x, dtype=np.float64).copy()
    n = result.shape[-1]
    step = 1
    while step < n:
        view = result.reshape(-1, n // (step * 2), step * 2)
        a, b = view[:, :, :step].copy(), view[:, :, step:].copy()
        view[:, :, :step], view[:, :, step:] = a + b, a - b
        step *= 2
    return result / np.sqrt(n)


def main():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as directory:
        oracle = os.environ.get("QTIP_ORACLE")
        if oracle is None:
            oracle = str(Path(directory) / "qtip_oracle")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++20", "-O2",
                            "-ffunction-sections", "-fdata-sections",
                            "-I" + str(root / "runtime/include"), "-I" + str(root / "third_party"),
                            str(root / "neuron/tests/qtip_oracle.cpp"),
                            str(root / "runtime/src/qtip.cpp"), "-Wl,--gc-sections",
                            "-o", oracle], check=True)
        subprocess.run([oracle, directory], check=True)
        p = Path(directory)
        read = lambda name, dtype: np.fromfile(p / (name + ".bin"), dtype=dtype)
        trellis = read("trellis", np.uint16).reshape(2, 2, 128)
        lut = read("lut", np.uint32).reshape(1, 1024)
        expected = read("dense", np.float32).reshape(64, 64)
        got = np.empty_like(expected)
        wi, ni, sh = indices()
        for row in range(2):
            for col in range(2):
                block = trellis[row, col].reshape(1, 128)
                pairs = nki.simulate(decode_block)(block, lut, wi, ni, sh)
                got[row*32:(row+1)*32, col*32:(col+1)*32] = reconstruct(pairs)
        np.testing.assert_array_equal(got, expected)
        x = read("x", np.float32).reshape(3, 64)
        su, sv = read("su", np.float32), read("sv", np.float32)
        hi, hs = hadamard_indices(64)
        rotated = nki.simulate(transform_pow2)(x, su.reshape(1, 64), hi, hs, True, True).astype(np.float64)
        expected_rotated = hadamard(x * su).astype(ml_dtypes.bfloat16).astype(np.float64)
        np.testing.assert_allclose(rotated, expected_rotated, rtol=1e-5, atol=1e-6)
        projected = np.zeros((3, 64), dtype=np.float64)
        pi = projection_indices()
        replicated_lut = np.repeat(lut, 32, axis=0)
        for token in range(3):
            for row in range(2):
                for col in range(2):
                    block = trellis[row, col].reshape(1, 128)
                    xin = rotated[token, col*32:(col+1)*32].astype(np.float32).reshape(1, 32)
                    partial = nki.simulate(project_block)(block, replicated_lut, *pi, xin).reshape(32)
                    expected_partial = got[row*32:(row+1)*32, col*32:(col+1)*32] @ xin.reshape(32)
                    np.testing.assert_allclose(partial, expected_partial, rtol=1e-5, atol=1e-6)
                    projected[token, row*32:(row+1)*32] += partial.astype(np.float64)
        full = nki.simulate(project_rotated)(
            trellis.reshape(2, 256), replicated_lut, *pi, rotated.astype(np.float32))
        np.testing.assert_allclose(full, projected, rtol=1e-5, atol=1e-6)
        looped = nki.simulate(project_rotated_loop)(
            trellis.reshape(1, 512), replicated_lut, *pi, rotated.astype(np.float32).reshape(1, 192), 64, 2, 3).reshape(3, 64)
        np.testing.assert_allclose(looped, projected, rtol=1e-5, atol=1e-6)
        y = nki.simulate(transform_pow2)(full, sv.reshape(1, 64), hi, hs, False, False)
        np.testing.assert_allclose(y, read("y", np.float32).reshape(3, 64), rtol=1e-5, atol=1e-6)
        for k in (12, 28):
            n = k * 32
            vec = read(f"had_input_{k}", np.float32).reshape(1, n)
            matrix = read(f"had_matrix_{k}", np.int8).reshape(k, k)
            for transpose in (False, True):
                tables = factored_hadamard_indices(n, matrix, transpose)
                result = nki.simulate(transform_factored)(vec, np.ones((1, n), np.float32),
                                                          *tables, False, False)
                expected = read(f"had_{k}_{'t' if transpose else 'n'}", np.float32).reshape(1, n)
                np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)
                chunked = KernelBackend().transform(vec, np.ones((1,n), np.float32), transpose, False, False)
                np.testing.assert_allclose(chunked, expected, rtol=2e-5, atol=2e-6)
                scale = np.where(np.arange(n) % 2, -1., 1.).astype(np.float32).reshape(1, n)
                staged = nki.simulate(transform_factored)(vec, scale, *tables, True, True)
                chunked_input = KernelBackend().transform(vec, scale, transpose, True, True)
                np.testing.assert_allclose(chunked_input, staged, rtol=.008, atol=2e-6)
        prefix = 'model.layers.0.self_attn.q_proj.'
        fixture = {prefix+'trellis': ('I16', trellis.reshape(16, 32).view(np.int16)),
                   prefix+'tlut': ('F16', read('tlut', np.uint16).view(np.float16).reshape(512, 2)),
                   prefix+'SU': ('F32', su), prefix+'SV': ('F32', sv),
                   prefix+'rcp': ('I64', np.array(0, np.int64)),
                   prefix+'tp_rank': ('I64', np.array(0, np.int64))}
        write_checkpoint(p, fixture)
        loaded = Checkpoint(p).projection(0, 'self_attn.q_proj', 64, 64)
        actual = KernelBackend().linear(loaded, x)
        np.testing.assert_allclose(actual, read('y', np.float32).reshape(3,64), rtol=1e-5, atol=1e-6)
        actual_z = KernelBackend().linear(loaded, x, round_z_bf16=True)
        np.testing.assert_allclose(actual_z, read('yz', np.float32).reshape(3,64), rtol=1e-5, atol=1e-6)
        print("NKI simulated checkpoint QTIP projection and transforms match the C++ reference")


if __name__ == "__main__":
    main()
