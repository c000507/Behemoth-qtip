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
        if self.lib.qtip_training_abi() != 3:
            raise RuntimeError("unsupported runtime bridge ABI")
        self.identity = sha256_file(path)
        self.lib.qtip_train_slices.argtypes = [integer] * 3
        self.lib.qtip_train_matmul.argtypes = [ptr] * 3 + [integer] * 4 + [ptr] * 3
        self.lib.qtip_train_hadamard.argtypes = [ptr, integer, integer, integer, ptr, ptr]
        self.lib.qtip_train_decode.argtypes = [ptr, ptr, integer, integer, ptr, ptr]
        long = C.c_long
        self.lib.qtip_train_had.argtypes = [ptr, integer, long, integer, integer, integer, ptr, integer,
                                            ptr, long, ptr, long, ptr, ptr]
        f32 = C.c_float
        self.lib.qtip_train_rms_had.argtypes = [ptr, long, integer, integer, ptr, f32, ptr, ptr, long, ptr, ptr,
                                                long, ptr]
        self.lib.qtip_train_gateup.argtypes = [ptr, long, integer, integer, ptr, ptr, ptr, ptr, ptr, long, ptr,
                                               ptr, long, ptr]
        self.lib.qtip_train_colsum.argtypes = [ptr, long, ptr, integer, long, ptr, integer, integer, ptr, ptr]
        self.lib.qtip_train_swiglu_bwd.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, integer, integer, ptr]
        self.lib.qtip_train_rms_bwd.argtypes = [ptr, long, ptr, ptr, ptr, ptr, long, ptr, long, integer, integer,
                                                ptr]
        self.lib.qtip_train_rope.argtypes = [ptr, long, integer, integer, integer, ptr, ptr, ptr, integer, ptr]
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


    # ---------------------------------------------------------------- fast engine
    fast_dims = (1024, 12288, 28672)

    @staticmethod
    def _rows(t, n, dtype=None):
        if t.dim() != 2 or t.shape[1] != n or t.stride(1) != 1 or (dtype is not None and t.dtype != dtype):
            raise ValueError("expected a 2-D row-strided tensor with contiguous rows")
        return t.stride(0)

    def had(self, x, n, transpose=False, pre=None, round_in=False, out32=None, out16=None, post=None):
        """out32 = H(x'), out16 = bf16(post * H(x')), x' = pre * [bf16](x); H = hadU or hadUt / sqrt(n).

        x: fp32/bf16 [rows, n] view (row stride allowed, e.g. a column slice); outputs likewise.
        Runs on the current stream of x's device; no allocation, no host sync.
        """
        if x.dtype not in (torch.float32, torch.bfloat16) or x.device.type != "cuda":
            raise ValueError("had input must be HIP fp32/bf16")
        rows = x.shape[0]
        ld = self._rows(x, n)
        ld32 = self._rows(out32, n, torch.float32) if out32 is not None else 0
        ld16 = self._rows(out16, n, torch.bfloat16) if out16 is not None else 0
        for t in (out32, out16):
            if t is not None and t.shape[0] != rows:
                raise ValueError("had output rows differ")
        for v in (pre, post):
            if v is not None and (v.dtype != torch.float32 or v.shape != (n,) or not v.is_contiguous()):
                raise ValueError("had scale vectors must be contiguous fp32 [n]")
        stream = torch.cuda.current_stream(x.device).cuda_stream
        pointer = lambda t: None if t is None else t.data_ptr()
        self.checked(self.lib.qtip_train_had(x.data_ptr(), int(x.dtype == torch.bfloat16), ld, rows, n,
                     int(transpose), pointer(pre), int(round_in), pointer(out32), ld32, pointer(out16), ld16,
                     pointer(post), stream))

    @staticmethod
    def _vec(v, n):
        if v.dtype != torch.float32 or v.shape != (n,) or not v.is_contiguous():
            raise ValueError("expected contiguous fp32 vector")
        return v.data_ptr()

    @staticmethod
    def _cur(t):
        return torch.cuda.current_stream(t.device).cuda_stream

    def rms_had(self, x, w, eps, su, u, xn=None, r=None):
        """u = bf16(hadUt(su * xn)), xn = bf16(x * rsqrt(mean(x^2) + eps) * w); optional xn, r [rows]."""
        rows, n = x.shape
        ld = self._rows(x, n, torch.bfloat16)
        ldu = self._rows(u, n, torch.bfloat16)
        ldxn = self._rows(xn, n, torch.bfloat16) if xn is not None else 0
        if r is not None and (r.dtype != torch.float32 or r.numel() != rows or not r.is_contiguous()):
            raise ValueError("r must be contiguous fp32 [rows]")
        self.checked(self.lib.qtip_train_rms_had(x.data_ptr(), ld, rows, n, self._vec(w, n), eps, self._vec(su, n),
                     None if xn is None else xn.data_ptr(), ldxn, None if r is None else r.data_ptr(),
                     u.data_ptr(), ldu, self._cur(x)))

    def gateup(self, z, svs_g, svs_u, gate, up, product, cg=None, cu=None):
        rows, inter = gate.shape
        ldz = self._rows(z, 2 * inter, torch.float32)
        ldo = self._rows(gate, inter, torch.bfloat16)
        if self._rows(up, inter, torch.bfloat16) != ldo or self._rows(product, inter, torch.bfloat16) != ldo:
            raise ValueError("gate/up/product must share a layout")
        ldc = 0
        if cg is not None:
            ldc = self._rows(cg, inter, torch.float32)
            if cu is None or self._rows(cu, inter, torch.float32) != ldc:
                raise ValueError("cg and cu must be given together with one layout")
        self.checked(self.lib.qtip_train_gateup(z.data_ptr(), ldz, rows, inter, self._vec(svs_g, inter),
                     self._vec(svs_u, inter), gate.data_ptr(), up.data_ptr(), product.data_ptr(), ldo,
                     None if cg is None else cg.data_ptr(), None if cu is None else cu.data_ptr(), ldc,
                     self._cur(z)))

    def colsum(self, a, b, acc, rs=None):
        """acc += sum over rows of a * b (* rs[row]); a fp32, b fp32/bf16 [rows, n]."""
        rows, n = a.shape
        lda = self._rows(a, n, torch.float32)
        ldb = self._rows(b, n)
        if b.shape[0] != rows or b.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("colsum operands differ")
        if rs is not None and (rs.dtype != torch.float32 or rs.numel() != rows or not rs.is_contiguous()):
            raise ValueError("row scale must be contiguous fp32 [rows]")
        self.checked(self.lib.qtip_train_colsum(a.data_ptr(), lda, b.data_ptr(), int(b.dtype == torch.bfloat16), ldb,
                     None if rs is None else rs.data_ptr(), rows, n, self._vec(acc, n), self._cur(a)))

    def swiglu_bwd(self, dxsu, su, gate, up, dg, du):
        rows, cols = dxsu.shape
        for t, dtype in ((dxsu, torch.float32), (gate, torch.bfloat16), (up, torch.bfloat16), (dg, torch.float32),
                         (du, torch.float32)):
            if not t.is_contiguous() or t.shape != (rows, cols) or t.dtype != dtype:
                raise ValueError("swiglu_bwd operands must be contiguous [rows, cols]")
        self.checked(self.lib.qtip_train_swiglu_bwd(dxsu.data_ptr(), self._vec(su, cols), gate.data_ptr(), up.data_ptr(),
                     dg.data_ptr(), du.data_ptr(), rows, cols, self._cur(dxsu)))

    def rms_bwd(self, x, r, w, su, dxsu, dh):
        rows, n = x.shape
        ldx = self._rows(x, n, torch.bfloat16)
        ldd = self._rows(dxsu, n, torch.float32)
        ldh = self._rows(dh, n, torch.float32)
        if r.dtype != torch.float32 or r.numel() != rows or not r.is_contiguous():
            raise ValueError("r must be contiguous fp32 [rows]")
        self.checked(self.lib.qtip_train_rms_bwd(x.data_ptr(), ldx, r.data_ptr(), self._vec(w, n), self._vec(su, n),
                     dxsu.data_ptr(), ldd, dh.data_ptr(), ldh, rows, n, self._cur(x)))

    def rope(self, x, heads, cos, sin, out, backward=False):
        """x bf16 [rows, heads * 128] (row stride allowed) -> out [rows, heads, 128] (bf16, or fp32 adjoint)."""
        rows = x.shape[0]
        ld = self._rows(x, heads * 128, torch.bfloat16)
        if (out.shape != (rows, heads, 128) or not out.is_contiguous()
                or out.dtype != (torch.float32 if backward else torch.bfloat16)):
            raise ValueError("invalid rope output")
        for t in (cos, sin):
            if t.dtype != torch.float32 or t.shape[0] != rows or t.numel() != rows * 128 or not t.is_contiguous():
                raise ValueError("rope tables must be contiguous fp32 [rows, 128]")
        self.checked(self.lib.qtip_train_rope(x.data_ptr(), ld, rows, heads, 128, cos.data_ptr(), sin.data_ptr(),
                     out.data_ptr(), int(backward), self._cur(x)))

    def decode_into(self, trellis, lut, m, n, out):
        """out[m, n] bf16 (contiguous) = runtime-decoded rotated weight; lut from lut()."""
        if out.dtype != torch.bfloat16 or out.shape != (m, n) or not out.is_contiguous() or trellis.numel() != m * n // 8:
            raise ValueError("invalid decode destination")
        stream = torch.cuda.current_stream(out.device).cuda_stream
        self.checked(self.lib.qtip_train_decode(trellis.data_ptr(), lut.data_ptr(), m, n, out.data_ptr(), stream))


def create_backend():
    return RuntimeBackend()
