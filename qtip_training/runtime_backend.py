"""Real behemoth QTIP kernels over PyTorch-managed tensors; no code/LUT uploads.

Forward reuses runtime WMMA/GEMV + slice reduction. The adjoint uses the runtime
BF16 row-tile decoder followed by FP32 BLAS, without rounding upstream gradients.
It is bounded-memory, but is not yet a fused transposed trellis GEMM.
"""
import ctypes as C
import os
from pathlib import Path

import torch

from .kernels import hadamard
from .data import sha256_file


class RuntimeBackend:
    name = "runtime"
    device_rounding = True

    def __init__(self, library=None, weight_rows=128):
        if not torch.version.hip:
            raise RuntimeError("runtime backend requires ROCm PyTorch and gfx1201; use runtime-reference on CPU")
        if weight_rows < 32 or weight_rows % 32:
            raise ValueError("weight_rows must be a positive multiple of 32")
        self.weight_rows = weight_rows
        path = library or os.environ.get("QTIP_TRAINING_LIBRARY") or (
            Path(__file__).resolve().parents[1] / "build-training/libqtip_training_hip.so")
        try:
            self.lib = C.CDLL(str(path))
        except OSError as error:
            raise RuntimeError(f"build the HIP bridge first (docs/FINE_TUNING.md): {path}") from error
        ptr, integer = C.c_void_p, C.c_int
        self.lib.qtip_training_error.restype = C.c_char_p
        self.lib.qtip_training_abi.restype = integer
        if self.lib.qtip_training_abi() != 1:
            raise RuntimeError("unsupported runtime bridge ABI")
        self.identity = sha256_file(path)
        self.lib.qtip_train_slices.argtypes = [integer] * 3
        self.lib.qtip_train_matmul.argtypes = [ptr] * 3 + [integer] * 4 + [ptr] * 3
        self.lib.qtip_train_hadamard.argtypes = [ptr, integer, integer, integer, ptr, ptr]
        self.lib.qtip_train_decode.argtypes = [ptr, ptr, integer, integer, ptr, ptr]
        self.luts = {}

    def checked(self, code):
        if code < 0:
            raise RuntimeError(self.lib.qtip_training_error().decode())
        return code

    @staticmethod
    def stream(x):
        if x.device.type != "cuda" or x.dtype != torch.float32 or not x.is_contiguous():
            raise ValueError("native kernels require contiguous HIP FP32 input")
        arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        if arch != "gfx1201":
            raise ValueError(f"runtime kernels require gfx1201, found {arch}")
        return torch.cuda.current_stream(x.device).cuda_stream

    def lut(self, tlut):
        # Double-buffered tensors reuse pointers with new contents. Version must
        # participate in invalidation; pointer-only caching uses the wrong layer.
        key = (tlut.device, tlut.data_ptr())
        cached = self.luts.get(key)
        if cached is None or cached[0] != tlut._version:
            base = tlut.to(torch.bfloat16)
            flipped = torch.stack((-base[:, 0], base[:, 1]), -1)
            packed = torch.cat((base, flipped)).contiguous().view(torch.int32).reshape(1024)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(tlut.device))
            self.luts[key] = (tlut._version, packed, ready)
        _, packed, ready = self.luts[key]
        stream = torch.cuda.current_stream(tlut.device)
        stream.wait_event(ready)
        packed.record_stream(stream)
        return packed

    def hadamard(self, x, transpose=False):
        # The runtime specializes the three Behemoth dimensions; small gradient
        # verification models use the same generated tables via the CPU oracle.
        x = x.contiguous()
        stream = self.stream(x)
        if x.shape[-1] not in (1024, 12288, 28672):
            return hadamard(x, transpose)
        result = torch.empty_like(x)
        with torch.cuda.device(x.device):
            self.checked(self.lib.qtip_train_hadamard(x.data_ptr(), x.numel() // x.shape[-1],
                         x.shape[-1], int(transpose), result.data_ptr(), stream))
        return result

    def matmul(self, x, trellis, tlut, m, n, transpose=False):
        x = x.contiguous()
        stream = self.stream(x)
        if (trellis.device != x.device or tlut.device != x.device or trellis.dtype != torch.int16
                or tlut.dtype != torch.float16 or not trellis.is_contiguous() or not tlut.is_contiguous()
                or m % 32 or n % 32 or trellis.numel() != m * n // 8 or tlut.shape != (512, 2)
                or x.shape[-1] != (m if transpose else n)):
            raise ValueError("invalid native QTIP storage, device or shape")
        with torch.cuda.device(x.device):
            lut = self.lut(tlut)
            rows = x.numel() // x.shape[-1]
            out = torch.zeros((*x.shape[:-1], n if transpose else m), dtype=x.dtype, device=x.device)
            if not transpose:
                rounded = x.to(torch.bfloat16)
                slices = self.checked(self.lib.qtip_train_slices(rows, m, n))
                partial = torch.empty((slices, rows, m), device=x.device, dtype=torch.float32)
                self.checked(self.lib.qtip_train_matmul(trellis.data_ptr(), lut.data_ptr(), rounded.data_ptr(),
                             rows, m, n, slices, partial.data_ptr(), out.data_ptr(), stream))
            else:
                # Each decoded row block is contiguous in the kernel swizzle.
                packed = trellis.reshape(m // 32, -1)
                for start in range(0, m, self.weight_rows):
                    end = min(start + self.weight_rows, m)
                    w = torch.empty((end - start, n), device=x.device, dtype=torch.bfloat16)
                    codes = packed[start // 32:end // 32]
                    self.checked(self.lib.qtip_train_decode(codes.data_ptr(), lut.data_ptr(), end - start, n,
                                 w.data_ptr(), stream))
                    out.add_(x[..., start:end] @ w.float())
            return out


def create_backend():
    return RuntimeBackend()
