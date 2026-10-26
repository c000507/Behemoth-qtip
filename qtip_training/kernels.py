"""Bounded-memory reference kernels and an interface for runtime QTIP kernels.

No dense model weight or T x T attention matrix is constructed. All custom
backwards are analytic; the trellis and LUT are always frozen.
"""
import functools
import importlib
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F


@functools.lru_cache(None)
def _had_table(n):
    source = (Path(__file__).resolve().parents[1] / "src/qtip_constants.hpp").read_text()
    order = [int(v) for v in re.search(r"kHadOrder\[12\] = \{([^}]+)", source)[1].split(",")]
    k = next((k for k in order if n % k == 0), 1)
    p = n // k
    if n < 1 or p & (p - 1):
        raise ValueError(f"unsupported QTIP Hadamard dimension: {n}")
    if k == 1:
        return k, None
    values = re.search(rf"kHad{k}\[\d+\] = \{{([^}}]+)", source)[1]
    return k, torch.tensor([int(v) for v in values.split(",")]).reshape(k, k)


@functools.lru_cache(64)
def _device_had(n, device, dtype):
    k, table = _had_table(n)
    return k, None if table is None else table.to(device=device, dtype=dtype)


def hadamard(x, transpose=False):
    """Rows times T^T (hadU), or T (hadUt), using the quantizer's exact tables."""
    n = x.shape[-1]
    k, table = _device_had(n, x.device, x.dtype)
    y = x.reshape(*x.shape[:-1], k, n // k)
    h = 1
    while h < n // k:
        z = y.reshape(*x.shape[:-1], k, -1, 2, h)
        a, b = z.unbind(-2)
        y = torch.stack((a + b, a - b), -2).reshape_as(y)
        h *= 2
    if table is not None:
        y = (table.T if transpose else table) @ y
    return y.reshape_as(x) / math.sqrt(n)


def decode_rows(trellis, tlut, m, n, start=0, stop=None, dtype=torch.float32):
    """Decode aligned output rows of the on-disk kernel swizzle, including wrap bits.

    This independent port follows src/qtip_core.cpp, without the upstream CUDA
    extension. At most (stop-start) x n decoded weights exist at once.
    """
    stop = m if stop is None else stop
    if m % 32 or n % 32 or start % 32 or stop % 32 or not 0 <= start < stop <= m:
        raise ValueError("kernel trellis dimensions and row range must be multiples of 32")
    words = trellis.reshape(m // 32, n // 32, 32, 2, 2)[start // 32:stop // 32]
    words = words.permute(0, 4, 1, 3, 2).reshape(-1, 32).to(torch.int64) & 65535
    steps = torch.arange(128, device=words.device)
    # A state is the MSB-first circular 16-bit window at bit 4*step.
    shift = (4 * steps) % 16
    a = words[:, steps // 4]
    b = words[:, (steps // 4 + 1) % 32]
    state = ((a << shift) | (b >> (16 - shift))) & 65535
    prod = state * (state + 1)
    pairs = tlut[(prod >> 6) & 511].to(dtype)
    pairs[..., 0] *= torch.where((prod & 32768) != 0, -1, 1)
    seq = pairs.reshape(-1, 256)
    perm = torch.arange(256, device=words.device).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
    tiles = seq[:, torch.argsort(perm)].reshape((stop - start) // 16, n // 16, 16, 16)
    return tiles.permute(0, 2, 1, 3).reshape(stop - start, n)


class ReferenceBackend:
    """Portable correctness backend. Runtime backends implement the same two methods."""
    name = "reference"
    device_rounding = False

    def __init__(self, weight_rows=128):
        if weight_rows < 32 or weight_rows % 32:
            raise ValueError("weight_rows must be a positive multiple of 32")
        self.weight_rows = weight_rows

    hadamard = staticmethod(hadamard)

    def matmul(self, x, trellis, tlut, m, n, transpose=False):
        # Forward: x @ W^T. Transpose: x @ W. Both keep codes compressed
        # outside a single row tile. Runtime implementations fuse decode+GEMM.
        shape = (*x.shape[:-1], n if transpose else m)
        y = torch.zeros(shape, dtype=x.dtype, device=x.device)
        for start in range(0, m, self.weight_rows):
            end = min(start + self.weight_rows, m)
            lut = tlut.to(torch.bfloat16).float() if self.device_rounding else tlut
            w = decode_rows(trellis, lut, m, n, start, end, x.dtype)
            if transpose:
                y.add_(x[..., start:end] @ w)
            else:
                y[..., start:end] = x @ w.T
        return y


class RuntimeReferenceBackend(ReferenceBackend):
    """Runtime projection precision with straight-through rounding derivatives.

    Not a full server oracle: paged FP8 attention and TP reduction order differ.
    """
    name = "runtime-reference"
    device_rounding = True

    def matmul(self, x, trellis, tlut, m, n, transpose=False):
        if not transpose:
            x = x.to(torch.bfloat16).to(x.dtype)
        return super().matmul(x, trellis, tlut, m, n, transpose)


def load_backend(name, weight_rows=128):
    if name == "reference":
        return ReferenceBackend(weight_rows)
    if name == "runtime-reference":
        return RuntimeReferenceBackend(weight_rows)
    if name == "runtime":
        from .runtime_backend import RuntimeBackend
        return RuntimeBackend(weight_rows=weight_rows)
    backend = importlib.import_module(name).create_backend()
    if not all(callable(getattr(backend, method, None)) for method in ("matmul", "hadamard")):
        raise TypeError("runtime backend must implement matmul and hadamard")
    if not isinstance(getattr(backend, "device_rounding", None), bool):
        raise TypeError("backend must declare device_rounding (bool)")
    return backend


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, su, sv, scale, trellis, tlut, backend, round_z):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        su_eff = su.to(torch.bfloat16).to(dtype) if backend.device_rounding else su.to(dtype)
        a = backend.hadamard(x.to(dtype) * su_eff, transpose=True)
        b = backend.matmul(a, trellis, tlut, sv.numel(), su.numel())
        if backend.device_rounding and round_z:
            b = b.to(torch.bfloat16).to(dtype)
        c = backend.hadamard(b)
        ctx.save_for_backward(x, su, sv, scale, trellis, tlut)
        ctx.backend = backend
        ctx.round_z = round_z
        return (c * (sv * scale)).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, su, sv, scale, trellis, tlut = ctx.saved_tensors
        backend = ctx.backend
        input_dtype = x.dtype
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x, dy = x.to(dtype), dy.to(dtype)
        su_eff = su.to(torch.bfloat16).to(dtype) if backend.device_rounding else su
        a = backend.hadamard(x * su_eff, transpose=True)
        b = backend.matmul(a, trellis, tlut, sv.numel(), su.numel())
        if backend.device_rounding and ctx.round_z:
            b = b.to(torch.bfloat16).to(dtype)
        c = backend.hadamard(b)
        dims = tuple(range(x.ndim - 1))
        dsv = (dy * c * scale).sum(dims)
        dscale = (dy * c * sv).sum().reshape_as(scale)
        db = backend.hadamard(dy * (sv * scale), transpose=True)
        da = backend.matmul(db, trellis, tlut, sv.numel(), su.numel(), transpose=True)
        dxsu = backend.hadamard(da)
        dsu = (dxsu * x).sum(dims)
        return (dxsu * su_eff).to(input_dtype), dsu, dsv, dscale, None, None, None, None


def qtip_linear(x, su, sv, scale, trellis, tlut, backend, round_z=False):
    return _Linear.apply(x, su, sv, scale, trellis, tlut, backend, round_z)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, segments, block):
        # [heads, tokens, dim], GQA without materializing repeated K/V.
        h, t, d = q.shape
        if h % k.shape[0] or q.shape != (h, t, d) or k.shape != v.shape or k.shape[1:] != (t, d):
            raise ValueError("invalid grouped-query attention shapes")
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        # Keep the unrounded output for delta in the softmax backward. Saving
        # bf16 here introduces an avoidable error in dQ/dK at long context.
        out = torch.empty(q.shape, device=q.device, dtype=dtype)
        lse = torch.empty((h, t), device=q.device, dtype=dtype)
        group = h // k.shape[0]
        ki = torch.arange(h, device=q.device) // group
        pos = torch.arange(t, device=q.device)
        for i in range(0, t, block):
            ie = min(i + block, t)
            qi = q[:, i:ie].to(dtype)
            top = torch.full((h, ie - i), -torch.inf, device=q.device, dtype=dtype)
            denom = torch.zeros_like(top)
            acc = torch.zeros_like(qi)
            for j in range(0, ie, block):
                je = min(j + block, t)
                kj, vj = k[ki, j:je].to(dtype), v[ki, j:je].to(dtype)
                score = qi @ kj.transpose(-1, -2) / math.sqrt(d)
                mask = (pos[i:ie, None] >= pos[None, j:je]) & (segments[i:ie, None] == segments[None, j:je])
                score.masked_fill_(~mask, -torch.inf)
                new_top = torch.maximum(top, score.amax(-1))
                # A tile can contain no keys for a query. Avoid -inf - -inf.
                safe_top = torch.where(torch.isfinite(new_top), new_top, 0)
                alpha = torch.exp(top - safe_top)
                p = torch.exp(score - safe_top.unsqueeze(-1))
                acc = acc * alpha.unsqueeze(-1) + p @ vj
                denom = denom * alpha + p.sum(-1)
                top = new_top
            out[:, i:ie] = acc / denom.unsqueeze(-1)
            lse[:, i:ie] = top + denom.log()
        ctx.save_for_backward(q, k, v, out, lse, segments)
        ctx.block = block
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, segments = ctx.saved_tensors
        h, t, d = q.shape
        dtype = lse.dtype
        dq, dk, dv = (torch.zeros_like(a, dtype=dtype) for a in (q, k, v))
        ki = torch.arange(h, device=q.device) // (h // k.shape[0])
        pos = torch.arange(t, device=q.device)
        for i in range(0, t, ctx.block):
            ie = min(i + ctx.block, t)
            qi, doi = q[:, i:ie].to(dtype), dout[:, i:ie].to(dtype)
            delta = (doi * out[:, i:ie].to(dtype)).sum(-1, keepdim=True)
            for j in range(0, ie, ctx.block):
                je = min(j + ctx.block, t)
                kj, vj = k[ki, j:je].to(dtype), v[ki, j:je].to(dtype)
                score = qi @ kj.transpose(-1, -2) / math.sqrt(d)
                mask = (pos[i:ie, None] >= pos[None, j:je]) & (segments[i:ie, None] == segments[None, j:je])
                score.masked_fill_(~mask, -torch.inf)
                p = torch.exp(score - lse[:, i:ie, None])
                ds = p * (doi @ vj.transpose(-1, -2) - delta) / math.sqrt(d)
                dq[:, i:ie].add_(ds @ kj)
                dk[:, j:je].index_add_(0, ki, ds.transpose(-1, -2) @ qi)
                dv[:, j:je].index_add_(0, ki, p.transpose(-1, -2) @ doi)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def attention(q, k, v, segments, block=256):
    if block < 1:
        raise ValueError("attention block must be positive")
    return _Attention.apply(q, k, v, segments, block)


def rmsnorm(x, weight, eps):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    z = x.to(dtype)
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def norm_weight(weight, backend):
    if backend.device_rounding:
        return weight + (weight.to(torch.bfloat16).to(weight.dtype) - weight).detach()
    return weight


def rope(x, positions, theta):
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE head dimension must be even")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=dtype) / d)
    angle = positions.to(dtype)[:, None] * inv[None, :]
    c, s = angle.cos().repeat(1, 2), angle.sin().repeat(1, 2)
    half = torch.cat((-x[..., d // 2:], x[..., :d // 2]), -1)
    return (x * c + half * s).to(x.dtype)


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, head, labels, chunk):
        # Labels are already shifted; -100 excludes user/system/padding/boundaries.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        total = torch.zeros((), device=x.device, dtype=dtype)
        head_calc = head.to(dtype)
        for i in range(0, x.shape[0], chunk):
            y = labels[i:i + chunk]
            valid = y != -100
            if valid.any():
                logits = x[i:i + chunk][valid].to(dtype) @ head_calc.T
                total += F.cross_entropy(logits, y[valid], reduction="sum")
        ctx.save_for_backward(x, head, labels)
        ctx.chunk = chunk
        return total

    @staticmethod
    def backward(ctx, grad):
        x, head, labels = ctx.saved_tensors
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        dx = torch.zeros_like(x)
        head_calc = head.to(dtype)
        for i in range(0, x.shape[0], ctx.chunk):
            y = labels[i:i + ctx.chunk]
            valid = y != -100
            if valid.any():
                logits = x[i:i + ctx.chunk][valid].to(dtype) @ head_calc.T
                p = logits.softmax(-1)
                p[torch.arange(y[valid].numel(), device=y.device), y[valid]] -= 1
                dx[i:i + ctx.chunk][valid] = (grad * (p @ head_calc)).to(x.dtype)
        return dx, None, None, None


def cross_entropy(x, head, labels, chunk=128):
    if chunk < 1:
        raise ValueError("loss chunk must be positive")
    return _CrossEntropy.apply(x, head, labels, chunk)
