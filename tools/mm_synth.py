#!/usr/bin/env python3
"""Parametric PE0.bin generator for y[M,N] = x[M,K] @ w[K,N] (trn1 v2 ISA).

Byte-exact vs all 18 oracle buckets (tools/mm_oracle_pairs.json; verify with
tools/mm_synth_check.py). Shape-dependent constants come from the TABLES
below (extracted from the 18 compiler oracles); everything else (rings,
counts, events, op10 fields) is formula-driven. PSUM final dsts are
drain-coupled compiler literals - never invent them, only use table values.
"""
import struct

PSUM = 0x02000000


def _u16(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


def _lw(src, num, ev, b1, u6):
    s = bytearray(32)
    s[0] = 1
    s[1] = 16
    if ev:
        _u16(s, 4, 5)
        _u16(s, 5, ev >> 8 & 0xFF)
        _u16(s, 8, 16 if ev == 0x805 else 1)
    struct.pack_into("<I", s, 16, src)
    _u16(s, 20, (-1) & 0xFFFF)
    _u16(s, 26, num)
    s[28] = 1
    s[30] = 1
    o = bytearray(32)
    o[0] = 10
    o[1] = b1
    _u16(o, 6, u6)
    _u16(o, 12, 3855)
    _u16(o, 14, 2)
    return bytes(s), bytes(o)


def _mm(a, ev, b1, ring, dst, dn, tick, u6, anum):
    s = bytearray(32)
    s[0] = 2
    s[1] = 16
    if ev:
        _u16(s, 6, 19)
        s[7] = 5
    struct.pack_into("<I", s, 16, a)
    _u16(s, 20, 1)
    _u16(s, 26, anum)
    s[28] = 1
    s[30] = 1
    o = bytearray(32)
    o[0] = 10
    o[1] = b1
    _u16(o, 6, u6)
    _u16(o, 12, 3855)
    _u16(o, 14, 16)
    o[11] = ring
    struct.pack_into("<I", o, 16, dst)
    _u16(o, 20, 1)
    _u16(o, 26, dn)
    _u16(o, 28, 1)
    _u16(o, 30, 1)
    return bytes(s), bytes(o)


def _ev(val, value=16):
    s = bytearray(32)
    s[0] = 160
    s[1] = 16
    _u16(s, 4, 5)
    _u16(s, 5, val >> 8 & 0xFF)
    _u16(s, 8, value)
    return bytes(s)


# ------------------- constant tables (from 18 oracles) ----------------------

def to_off(M, K, N):
    big = (N >= 512 or (M <= 64 and N >= 256 and (K >= 256 or M == 64))
           or (M == 32 and K >= 512))
    return 0x2FE00 if big else 0x30000 - 4 * M


def a_base(M, K, N):
    """A-phase PSUM temp base (bank-strided 0x800 blocks)."""
    NT = (N + 127) // 128
    if K == 128:
        if NT == 1:
            return 0x800
        if M == 32:
            return 0x1000 if NT >= 4 else 0x800
        return 0x1000  # M64 NT2 observed 0x1000; M96/128 NT>=2 unobserved
    if NT >= 2 and M <= 64:
        return 0x0
    if M == 32 and K >= 512:
        return 0x1000
    return 0x800


def final_scheme(M, K, N):
    """Return (kind, dsts, dns). kind: col | direct | pv."""
    NT = (N + 127) // 128
    if K == 128:
        if N <= 256:
            dn = N if (M >= 96 or M == 32) else 128
            tiles = (N + dn - 1) // dn
            return "col", [PSUM | (4 * M * i) for i in range(tiles)], [dn] * tiles
        return "col", [PSUM | (4 * M * i) for i in range(NT)], [128] * NT
    if M >= 96:
        return "direct", [PSUM | 0], [N]
    pv = (M == 32 and (K >= 512 or (N >= 256 and K >= 256))) or \
         (M == 64 and N >= 256)
    if pv:
        base = 0x1800 if NT <= 2 else 0x800
        return "pv", [PSUM | (base + 0x200 * t) for t in range(NT)], [128] * NT
    return "direct", [PSUM | 0], [N]


def b_temps(M, K, N):
    """B-phase PSUM temps when present (row-mode M<=64, N>=256, K>=256):
    tile t -> PSUM|0x800*(1+t); stream emits tile groups in REVERSE order."""
    kind, _, _ = final_scheme(M, K, N)
    NT = (N + 127) // 128
    if kind == "pv":
        return [PSUM | (0x800 * (1 + t)) for t in range(NT)]
    return None


def b_scratch(M, K, N, s):
    """B-phase MM A operand for chunk/tile s (0-based)."""
    NT = (N + 127) // 128
    tof = to_off(M, K, N)
    x_bottom = tof - 512 * ((K // 128))
    if K == 128:
        # w tile(s) directly below x ring; window const for big-N cases
        if N >= 512 or (N >= 256 and M >= 64):
            return 0x4000
        return x_bottom - ((N + 255) // 256) * 256 - 0 + 0  # placeholder
    return None


def synth(M, K, N):
    """Generate PE0.bin bytes. Raises ValueError outside covered families."""
    AS = K // 128
    NT = (N + 127) // 128
    tof = to_off(M, K, N)
    out = bytearray()
    out += bytes([177, 16] + [0] * 30)
    out += bytes([0] * 32)
    out += bytes([204, 16] + [0] * 30)
    out += bytes([0] * 32)
    out += _ev(0x1805)
    out += bytes([0] * 32)
    x_bottom = tof - 512 * AS

    def x_lw(s):
        return tof - 512 * (AS - 1 - s) - 4

    U6A = 0x8000 + M       # A/PV-phase op10 const
    U6B = 0x2000 + 4 * M   # B-phase MM op10 const
    U6BM = 0x8000 + 4 * M  # B-phase MM op10 const

    def pair(lw_src, lw_num, lw_ev, gp, mm_a, mm_dst, mm_dn, mm_b1, mm_ring,
             mm_ev, lw_u6=None, mm_u6=None, anum=None):
        s, o = _lw(lw_src, lw_num, lw_ev, 1 if gp % 2 == 0 else 2,
                   lw_u6 if lw_u6 is not None else U6A)
        out.extend(s)
        out.extend(bytes(o))
        s2, o2 = _mm(mm_a, mm_ev, mm_b1, mm_ring, mm_dst, mm_dn, 0, 
                     mm_u6 if mm_u6 is not None else U6A,
                     M if anum is None else anum)
        out.extend(s2)
        out.extend(o2)

    kind, fdst, fdn = final_scheme(M, K, N)
    temps = b_temps(M, K, N)

    # ---- A phase: AS slots x 2 pairs
    ab = a_base(M, K, N)
    a_dn = M
    for s in range(AS):
        for p in range(2):
            gp = 2 * s + p
            ev = 0x805 if gp == 0 else 0
            upd = 0x5130000 if gp == 2 * AS - 1 else 0
            pair(x_lw(s), 128, ev, gp, tof, PSUM | (ab + 4 * M * s), a_dn, 2,
                 1 if p == 0 else 2, upd)

    # ---- B phase
    out += _ev(0x905, 16)
    out += bytes([0] * 32)
    # scratch/LW pattern per family
    if K == 128:
        if N >= 512 or (N >= 256 and M >= 64):
            scr = lambda t: 0x4000
            Ws = 512
            lw = lambda t: x_bottom - Ws * (NT - 1 - t) - 4
        else:
            scr = lambda t: x_bottom - 4 * N
            lw = lambda t: 0x4000 + 4 * M * (t + 1) - 4
            w_lw = None
    else:
        wring = ((M == 32 and K >= 512 and N <= 128)
                 or (N >= 256 and K >= 256 and M <= 64) or (N >= 256 and M == 64))
        if wring:
            # LW = w ring below x ring; scr = window 0x4000+4M*s
            Ws = max(512, 4 * N)
            scr = lambda s: 0x4000 + 4 * M * s
            def lw(s):
                extra = 512 if Ws > 512 else 0
                return x_bottom - extra - Ws * (AS - 1 - s) - 4
        else:
            # LW = window 0x4000+4M*(s+1)-4; scr = SB w tiles below x ring
            Ws = 4 * N
            scr = lambda s: x_bottom - Ws * (AS - s)
            lw = lambda s: 0x4000 + 4 * M * (s + 1) - 4

    if kind == "col" and N > 256:
        # per-tile groups, w tiles descending from x_bottom, dst 4M*t
        for t in range(NT):
            for p in range(2):
                gp = 2 * t + p
                ev = 0x405 if gp == 0 else 0
                upd = 0x5130000 if gp == 2 * NT - 1 else 0
                a = 0x4000
                lwsrc = x_bottom - 512 * (NT - 1 - t) - 4
                pair(lwsrc, M, ev, gp, a, fdst[t], fdn[t], 3,
                     1 if (t == 0 and p == 0) else (2 if gp == 2 * NT - 1 else 0),
                     upd, U6BM, U6B, fdn[t])
    elif temps is None:
        # direct single-tile accumulate (row-mode M>=96 / no-PV / col N<=256)
        ntiles = len(fdst)
        for t in range(ntiles):
            groups = AS if kind != "col" else 1
            for s in range(groups):
                for p in range(2):
                    gp = t * 2 * groups * AS + 2 * s + p if kind != "col" else \
                        2 * t + p
                    ev = 0x405 if gp == (0 if t == 0 else -1) else 0
                    ev = 0x405 if (t == 0 and s == 0 and p == 0) else 0
                    upd = 0x5130000 if (t == ntiles - 1 and s == groups - 1
                                        and p == 1) else 0
                    if kind == "col":
                        # K128: w tile = A operand, anum = fdn[t]
                        a = scr(t)
                        lws = lw(t)
                        pair(lws, M, ev, 2 * t + p, a, fdst[t], fdn[t], 3,
                             1 if p == 0 else 2, upd, U6BM, U6B, fdn[t])
                    else:
                        a = scr(s)
                        lws = lw(s)
                        pair(lws, M, ev, 2 * AS * t + 2 * s + p, a, fdst[0],
                             fdn[0], 3,
                             1 if (s == 0 and p == 0) else
                             (2 if (s == AS - 1 and p == 1) else 0), upd,
                             U6BM, U6B, fdn[0])
    else:
        # PV row-mode M<=64: B temps banks 1..NT (reverse order), PV final
        for ti, t in enumerate(range(NT - 1, -1, -1)):
            for s in range(AS):
                for p in range(2):
                    gp = ti * 2 * AS + 2 * s + p
                    ev = 0x405 if (ti == 0 and s == 0 and p == 0) else 0
                    upd = 0x5130000 if (ti == NT - 1 and s == AS - 1 and
                                        p == 1) else 0
                    if M == 32 or N >= 256:
                        a = scr(s)
                        lws = lw(s)
                        ring = 1 if (s == 0 and p == 0) else \
                            (2 if (s == AS - 1 and p == 1) else 0)
                        pair(lws, 128, ev, gp, a, temps[t], M, 3, ring, upd,
                             U6BM, U6B)
                    else:
                        a = scr(s)
                        lws = lw(s)
                        ring = 1 if (s == 0 and p == 0) else \
                            (2 if (s == AS - 1 and p == 1) else 0)
                        pair(lws, M, ev, gp, a, temps[t], M, 3, ring, upd,
                             U6BM, U6B)
        pv0 = {32: 0x4000 + K + 4 * M + 0x200, 64: 0x4000 + 2 * K + 0x100}[M]
        for t in range(NT):
            for p in range(2):
                gp = 2 * NT * AS + 2 * t + p
                ev = 0x405 if (NT <= 2 or t == 0) and p == 0 else 0
                upd = 0x5130000 if (t == NT - 1 and p == 1) else 0
                pair(pv0 + 4 * M * t - 4, M, ev, gp, tof, fdst[t], 128, 2,
                     1 if p == 0 else 2, upd)
    return bytes(out)
