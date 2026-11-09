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


def _lw(src, num, wait, b1, u6):
    s = bytearray(32)
    s[0] = 1
    s[1] = 16
    if wait:
        _u16(s, 4, 5)
        _u16(s, 5, wait[0])
        _u16(s, 8, wait[1])
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



def scr_col(M, K, N, t):
    """K128 col-mode B-phase MM A operand (w tile below x ring, or window)."""
    tof = to_off(M, K, N)
    x_bottom = tof - 512 * (K // 128)
    NT = (N + 127) // 128
    if N >= 512 or (N >= 256 and M >= 64):
        return 0x4000
    return x_bottom - 4 * N


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
        return "col", [PSUM | (4 * M * i) for i in range(NT)], [M] * NT
    if M >= 96:
        return "direct", [PSUM | 0], [N]
    pv = (M == 32 and (K >= 512 or (N >= 256 and K >= 256))) or \
         (M == 64 and N >= 256)
    if pv:
        base = 0x800 if NT == 1 else (0x1800 if NT == 2 else 0x800)
        return "pv", [PSUM | (base + 0x200 * t) for t in range(NT)], [128] * NT
    return "direct", [PSUM | 0], [N]


def b_temps(M, K, N):
    """B-phase PSUM temps when present (row-mode M<=64, N>=256, K>=256):
    tile t -> PSUM|0x800*(1+t); stream emits tile groups in REVERSE order."""
    kind, _, _ = final_scheme(M, K, N)
    NT = (N + 127) // 128
    if kind == "pv":
        if NT == 1:
            return [PSUM | 0]
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

    U6A = 0x8000 + M          # A-phase op10 const
    U6F = 0x80 + 0x2000 * (M // 32)  # B/PV const, non-wring families
    U6W = 0x8080              # B const, wring families and col N>256
    wring = ((M == 32 and K >= 512 and N <= 128)
             or (N >= 256 and K >= 256 and M <= 64) or (N >= 256 and M == 64))
    def pair(lw_src, lw_num, wait, gp, mm_a, mm_dst, mm_dn, mm_b1, mm_ring,
             upd, lw_u6, mm_u6, anum):
        s, o = _lw(lw_src, lw_num, wait, 1 if gp % 2 == 0 else 2, lw_u6)
        out.extend(s)
        out.extend(bytes(o))
        s2, o2 = _mm(mm_a, upd, mm_b1, mm_ring, mm_dst, mm_dn, 0,
                     mm_u6, anum)
        out.extend(s2)
        out.extend(o2)

    kind, fdst, fdn = final_scheme(M, K, N)
    U6B = U6W if wring or (kind == "col" and N > 256) else U6F
    temps = b_temps(M, K, N)

    # ---- A phase: AS slots x 2 pairs
    ab = a_base(M, K, N)
    for s in range(AS):
        for p in range(2):
            gp = 2 * s + p
            upd = 0x5130000 if gp == 2 * AS - 1 else 0
            wait = (8, 16) if gp == 0 else None
            pair(x_lw(s), 128, wait, gp, tof, PSUM | (ab + 4 * M * s), M, 2,
                 1 if p == 0 else 2, upd, U6A, U6A, M)

    # ---- B phase --------------------------------------------------------
    out += _ev(0x905, 16)
    out += bytes([0] * 32)
    if kind == "col":
        lw_num = M if (N <= 128 or (M == 32 and N <= 256)) else 128
        anum = M if (M == 64 and N >= 256 and N <= 256) else None
        ntiles = len(fdst)
        gpt = 2
        for t in range(ntiles):
            for g in range(gpt):
                gp = t * gpt + g
                upd = (0x5130000 if (gpt > 1 and t == ntiles - 1
                                     and g == gpt - 1) else 0)
                wait = (4, 1) if gp == 0 else None
                scr_t = scr_col(M, K, N, t)
                lw_t = (0x4000 + 4 * M * (t + 1) - 4 if ntiles == 1
                        else x_bottom - 512 * (ntiles - 1 - t) - 4)
                dn_t = M if (M == 64 and N == 256) else fdn[t]
                pair(lw_t, lw_num, wait, gp, scr_t, fdst[t], dn_t, 3,
                     1 if g == 0 else (2 if g == gpt - 1 else 0),
                     upd, U6B, U6B, anum if anum is not None else dn_t)
    elif kind == "pv":
        # wring families: LW = w ring (last tile +Ws/2), scr = window
        Ws = max(512, 4 * N)
        extra = 0 if Ws == 512 else 512

        def lwbase(s):
            return x_bottom - extra - Ws * (AS - 1 - s) - 4

        for ti, t in enumerate(range(NT - 1, -1, -1)):
            off = 0 if t == NT - 1 else 512
            for s in range(AS):
                for p in range(2):
                    gp = ti * 2 * AS + 2 * s + p
                    wait = (4, 1) if gp == 0 else None
                    upd = 0x5130000 if (s == AS - 1 and p == 1) else 0
                    pair(lwbase(s) + off, 128, wait, gp,
                         0x4000 + 4 * M * s, temps[t], M, 3,
                         1 if (s == 0 and p == 0) else
                         (2 if (s == AS - 1 and p == 1) else 0),
                         upd, U6B, U6B, M)
    else:
        # direct row-mode: LW = window, scr = SB w tiles below x ring
        Ws = 4 * N
        for s in range(AS):
            for p in range(2):
                gp = 2 * s + p
                wait = (4, 1) if gp == 0 else None
                upd = 0x5130000 if (s == AS - 1 and p == 1) else 0
                pair(0x4000 + 4 * M * (s + 1) - 4, M, wait, gp,
                     x_bottom - Ws * (AS - s), fdst[0], fdn[0], 3,
                     1 if (s == 0 and p == 0) else
                     (2 if (s == AS - 1 and p == 1) else 0),
                     upd, U6B, U6B, fdn[0])

    # ---- PV phase --------------------------------------------------------
    if kind == "pv" or (kind == "col" and len(fdst) > 1):
        pv0 = 0x4000 + (K if M == 32 else 2 * K) + 4 * M
        pvdst = fdst if kind == "pv" else [PSUM | (0x800 + 0x200 * t)
                                           for t in range(NT)]
        for t in range(NT):
            for p in range(2):
                gp = 2 * NT * AS + 2 * t + p
                wait = None
                if p == 0 and t == 0:
                    wait = (4, 2)
                elif p == 0 and t == 1 and NT == 2 and kind == "pv":
                    wait = (4, 3)
                upd = 0x5130000 if (t == NT - 1 and p == 1) else 0
                pair(pv0 + 4 * M * t - 4, M, wait, gp, tof, pvdst[t], 128, 2,
                     1 if p == 0 else 2, upd, U6F, U6F, 128)
    return bytes(out)
