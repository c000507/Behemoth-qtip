"""Triton flash attention (causal, packed documents, GQA) for the fast engine on gfx1201.

Layout: q [T, H, D], k/v [T, KV, D] (bf16, last dim contiguous), one packed row of
documents described by doc_start[t] / doc_end[t] (first / one-past-last token of t's
document). Forward returns o [T, H, D] bf16 and lse2 [H, T] fp32 (log2-domain
logsumexp of the scaled scores). Backward is FA2-style without atomics: one kernel
accumulates dK/dV for a KV head over all of its GROUP query heads (so GQA needs no
reduction pass) and one kernel computes dQ. Fully visible tiles skip mask arithmetic.
"""
import math
import threading

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
# Triton's JIT/launcher is not thread-safe; replicas launch from worker threads. Launches
# are asynchronous, so the lock only serializes host-side launch overhead.
_LAUNCH = threading.Lock()


@triton.jit
def _fwd_tile(acc, l_i, m_i, q, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T, qk_scale,
              MASK: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    offs_n = start_n + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    kmask = offs_n < T
    k = tl.load(K + offs_n[None, :] * stride_kt + kvh * stride_kh + offs_d[:, None], mask=kmask[None, :], other=0.0)
    s = tl.dot(q, k) * qk_scale
    if MASK:
        valid = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] >= ds[:, None]) & kmask[None, :]
        s = tl.where(valid, s, float("-inf"))
        # A row can see no key of a masked tile (document boundary): avoid -inf - -inf.
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
    else:
        m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    v = tl.load(V + offs_n[:, None] * stride_kt + kvh * stride_kh + offs_d[None, :], mask=kmask[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, l_i, m_new


@triton.jit
def _fwd_kernel(Q, K, V, O, L, DS, qk_scale, T, stride_qt, stride_qh, stride_kt, stride_kh, stride_ot, stride_oh,
                GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // GROUP
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mmask = offs_m < T
    q = tl.load(Q + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=mmask[:, None], other=0.0)
    ds = tl.load(DS + offs_m, mask=mmask, other=0)
    lo = (tl.min(tl.where(mmask, ds, T), 0) // BN) * BN
    full_lo = ((tl.max(ds, 0) + BN - 1) // BN) * BN
    full_hi = pid_m * BM
    hi = tl.minimum(pid_m * BM + BM, T)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    mid_lo = tl.minimum(tl.maximum(full_lo, lo), full_hi)
    for start_n in range(lo, mid_lo, BN):
        acc, l_i, m_i = _fwd_tile(acc, l_i, m_i, q, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T,
                                  qk_scale, True, BN, D)
    for start_n in range(mid_lo, full_hi, BN):
        acc, l_i, m_i = _fwd_tile(acc, l_i, m_i, q, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T,
                                  qk_scale, False, BN, D)
    for start_n in range(tl.maximum(full_hi, mid_lo), hi, BN):
        acc, l_i, m_i = _fwd_tile(acc, l_i, m_i, q, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T,
                                  qk_scale, True, BN, D)
    o = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * stride_ot + h * stride_oh + offs_d[None, :], o.to(O.dtype.element_ty),
             mask=mmask[:, None])
    tl.store(L + h * T + offs_m, m_i + tl.math.log2(l_i), mask=mmask)


@triton.jit
def _dkdv_tile(dk, dv, k, v, Q, DO, L, Dl, DS, h, stride_qt, stride_qh, offs_n, start_m, T, qk_scale,
               MASK: tl.constexpr, BM: tl.constexpr, D: tl.constexpr, TLOAD: tl.constexpr):
    offs_m = start_m + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mmask = offs_m < T
    q = tl.load(Q + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=mmask[:, None], other=0.0)
    do = tl.load(DO + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=mmask[:, None], other=0.0)
    lse = tl.load(L + h * T + offs_m, mask=mmask, other=0.0)
    delta = tl.load(Dl + h * T + offs_m, mask=mmask, other=0.0)
    if TLOAD:
        qt = tl.load(Q + offs_m[None, :] * stride_qt + h * stride_qh + offs_d[:, None], mask=mmask[None, :], other=0.0)
        dot_ = tl.load(DO + offs_m[None, :] * stride_qt + h * stride_qh + offs_d[:, None], mask=mmask[None, :],
                       other=0.0)
    else:
        qt = tl.trans(q)
        dot_ = tl.trans(do)
    st = tl.dot(k, qt) * qk_scale
    pt = tl.math.exp2(st - lse[None, :])
    if MASK:
        dq_start = tl.load(DS + offs_m, mask=mmask, other=T)
        valid = (offs_m[None, :] >= offs_n[:, None]) & (offs_n[:, None] >= dq_start[None, :]) & mmask[None, :]
        pt = tl.where(valid, pt, 0.0)
    dv += tl.dot(pt.to(do.dtype), do)
    dpt = tl.dot(v, dot_)
    dst = pt * (dpt - delta[None, :])
    dk += tl.dot(dst.to(q.dtype), q)
    return dk, dv


@triton.jit
def _dkdv_kernel(Q, K, V, DO, L, Dl, DS, DE, DK, DV, qk_scale, sm_scale, T, stride_qt, stride_qh, stride_kt,
                 stride_kh, GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                 TLOAD: tl.constexpr):
    pid_n = tl.program_id(0)
    kvh = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    nmask = offs_n < T
    k = tl.load(K + offs_n[:, None] * stride_kt + kvh * stride_kh + offs_d[None, :], mask=nmask[:, None], other=0.0)
    v = tl.load(V + offs_n[:, None] * stride_kt + kvh * stride_kh + offs_d[None, :], mask=nmask[:, None], other=0.0)
    de = tl.load(DE + offs_n, mask=nmask, other=0)
    lo = pid_n * BN
    hi = tl.minimum(tl.max(de, 0), T)
    # Rows >= diag_hi see every key of this block (causal); rows < full_hi share all keys' documents.
    diag_hi = tl.minimum(((lo + BN + BM - 1) // BM) * BM, hi)
    full_hi = tl.maximum((tl.min(tl.where(nmask, de, T), 0) // BM) * BM, diag_hi)
    full_hi = tl.minimum(full_hi, hi)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    for g in range(GROUP):
        h = kvh * GROUP + g
        for start_m in range(lo, diag_hi, BM):
            dk, dv = _dkdv_tile(dk, dv, k, v, Q, DO, L, Dl, DS, h, stride_qt, stride_qh, offs_n, start_m, T,
                                qk_scale, True, BM, D, TLOAD)
        for start_m in range(diag_hi, full_hi, BM):
            dk, dv = _dkdv_tile(dk, dv, k, v, Q, DO, L, Dl, DS, h, stride_qt, stride_qh, offs_n, start_m, T,
                                qk_scale, False, BM, D, TLOAD)
        for start_m in range(full_hi, hi, BM):
            dk, dv = _dkdv_tile(dk, dv, k, v, Q, DO, L, Dl, DS, h, stride_qt, stride_qh, offs_n, start_m, T,
                                qk_scale, True, BM, D, TLOAD)
    dk *= sm_scale
    tl.store(DK + offs_n[:, None] * stride_kt + kvh * stride_kh + offs_d[None, :], dk.to(DK.dtype.element_ty),
             mask=nmask[:, None])
    tl.store(DV + offs_n[:, None] * stride_kt + kvh * stride_kh + offs_d[None, :], dv.to(DV.dtype.element_ty),
             mask=nmask[:, None])


@triton.jit
def _dq_tile(dq, q, do, lse, delta, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T, qk_scale,
             MASK: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    offs_n = start_n + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    kmask = offs_n < T
    kt = tl.load(K + offs_n[None, :] * stride_kt + kvh * stride_kh + offs_d[:, None], mask=kmask[None, :], other=0.0)
    vt = tl.load(V + offs_n[None, :] * stride_kt + kvh * stride_kh + offs_d[:, None], mask=kmask[None, :], other=0.0)
    s = tl.dot(q, kt) * qk_scale
    p = tl.math.exp2(s - lse[:, None])
    if MASK:
        valid = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] >= ds[:, None]) & kmask[None, :]
        p = tl.where(valid, p, 0.0)
    dp = tl.dot(do, vt)
    dsc = p * (dp - delta[:, None])
    dq += tl.dot(dsc.to(q.dtype), tl.trans(kt))
    return dq


@triton.jit
def _dq_kernel(Q, K, V, DO, L, Dl, DS, DQ, qk_scale, sm_scale, T, stride_qt, stride_qh, stride_kt, stride_kh,
               GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // GROUP
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mmask = offs_m < T
    q = tl.load(Q + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=mmask[:, None], other=0.0)
    do = tl.load(DO + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=mmask[:, None], other=0.0)
    lse = tl.load(L + h * T + offs_m, mask=mmask, other=0.0)
    delta = tl.load(Dl + h * T + offs_m, mask=mmask, other=0.0)
    ds = tl.load(DS + offs_m, mask=mmask, other=0)
    lo = (tl.min(tl.where(mmask, ds, T), 0) // BN) * BN
    full_lo = ((tl.max(ds, 0) + BN - 1) // BN) * BN
    full_hi = pid_m * BM
    hi = tl.minimum(pid_m * BM + BM, T)
    mid_lo = tl.minimum(tl.maximum(full_lo, lo), full_hi)
    dq = tl.zeros([BM, D], tl.float32)
    for start_n in range(lo, mid_lo, BN):
        dq = _dq_tile(dq, q, do, lse, delta, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T, qk_scale,
                      True, BN, D)
    for start_n in range(mid_lo, full_hi, BN):
        dq = _dq_tile(dq, q, do, lse, delta, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T, qk_scale,
                      False, BN, D)
    for start_n in range(tl.maximum(full_hi, mid_lo), hi, BN):
        dq = _dq_tile(dq, q, do, lse, delta, K, V, kvh, stride_kt, stride_kh, offs_m, ds, start_n, T, qk_scale,
                      True, BN, D)
    dq *= sm_scale
    tl.store(DQ + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dq.to(DQ.dtype.element_ty),
             mask=mmask[:, None])


# Tuned on gfx1201 at T = 16-32k, H = 96, KV = 8, D = 128 (docs/FINETUNE_PERF_REPORT.md).
CONFIG = dict(fwd=dict(BM=128, BN=64, num_warps=8, num_stages=1),
              dkdv=dict(BM=32, BN=128, num_warps=8, num_stages=2),
              dq=dict(BM=64, BN=32, num_warps=4, num_stages=2))


def doc_bounds(runs, t, device):
    start = torch.empty(t, dtype=torch.int32)
    end = torch.empty(t, dtype=torch.int32)
    for a, b in runs:
        start[a:b] = a
        end[a:b] = b
    return start.to(device), end.to(device)


def _check(q, k, v):
    t, h, d = q.shape
    if k.shape != v.shape or k.shape[0] != t or k.shape[2] != d or h % k.shape[1] or d not in (64, 128):
        raise ValueError("invalid attention shapes")
    for x in (q, k, v):
        if x.dtype != torch.bfloat16 or x.stride(2) != 1 or x.stride(1) != d:
            raise ValueError("attention tensors must be bf16 [T, H, D] with contiguous heads")
    if k.stride() != v.stride():
        raise ValueError("k and v must have the same strides")


def forward(q, k, v, doc_start, config=None):
    _check(q, k, v)
    c = dict(CONFIG["fwd"], **(config or {}))
    t, h, d = q.shape
    o = torch.empty_like(q)
    lse = torch.empty((h, t), dtype=torch.float32, device=q.device)
    scale = 1 / math.sqrt(d)
    with _LAUNCH:
        _fwd_kernel[(triton.cdiv(t, c["BM"]), h)](
            q, k, v, o, lse, doc_start, scale * LOG2E, t, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            o.stride(0), o.stride(1), GROUP=h // k.shape[1], BM=c["BM"], BN=c["BN"], D=d,
            num_warps=c["num_warps"], num_stages=c["num_stages"])
    return o, lse


def backward(do, q, k, v, o, lse, doc_start, doc_end, config=None, parts=("dkdv", "dq")):
    """Returns (dq, dk, dv); parts selects kernels (benchmarks)."""
    _check(q, k, v)
    t, h, d = q.shape
    if do.stride() != q.stride():
        do = do.contiguous()
        if do.stride() != q.stride():
            raise ValueError("do must share q's layout")
    delta = (do.float() * o.float()).sum(-1).t().contiguous()
    scale = 1 / math.sqrt(d)
    dq = torch.empty_like(q)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    group = h // k.shape[1]
    config = config or {}
    if "dkdv" in parts:
        c = dict(CONFIG["dkdv"], **config.get("dkdv", {}))
        with _LAUNCH:
            _dkdv_kernel[(triton.cdiv(t, c["BN"]), k.shape[1])](
                q, k, v, do, lse, delta, doc_start, doc_end, dk, dv, scale * LOG2E, scale, t, q.stride(0),
                q.stride(1), k.stride(0), k.stride(1), GROUP=group, BM=c["BM"], BN=c["BN"], D=d,
                TLOAD=c.get("TLOAD", False), num_warps=c["num_warps"], num_stages=c["num_stages"])
    if "dq" in parts:
        c = dict(CONFIG["dq"], **config.get("dq", {}))
        with _LAUNCH:
            _dq_kernel[(triton.cdiv(t, c["BM"]), h)](
                q, k, v, do, lse, delta, doc_start, dq, scale * LOG2E, scale, t, q.stride(0), q.stride(1),
                k.stride(0), k.stride(1), GROUP=group, BM=c["BM"], BN=c["BN"], D=d, num_warps=c["num_warps"],
                num_stages=c["num_stages"])
    return dq, dk, dv
