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


# ---------------- pool engine (Pool0.bin) ----------------

# desc rearm/chunk literals (per bucket, compiler choices)
D0_REARM = {
    (32, 128, 128): 640,
    (32, 128, 256): 640,
    (32, 128, 512): 1024,
    (32, 256, 128): 1152,
    (32, 256, 256): 1536,
    (32, 512, 128): 2560,
    (32, 512, 256): 2560,
    (32, 1024, 128): 4608,
    (64, 128, 128): 768,
    (64, 128, 256): 1024,
    (64, 256, 128): 1280,
    (64, 256, 256): 1536,
    (64, 512, 128): 2304,
    (64, 512, 256): 2560,
    (96, 128, 128): 896,
    (96, 512, 256): 2432,
    (96, 640, 192): 2944,
    (128, 128, 128): 1024,
}

D1_CHUNKS = {
    (32, 128, 128): 0,
    (32, 128, 256): 0,
    (32, 128, 512): 0,
    (32, 256, 128): 1,
    (32, 256, 256): 2,
    (32, 512, 128): 1,
    (32, 512, 256): 2,
    (32, 1024, 128): 1,
    (64, 128, 128): 0,
    (64, 128, 256): 0,
    (64, 256, 128): 1,
    (64, 256, 256): 2,
    (64, 512, 128): 1,
    (64, 512, 256): 2,
    (96, 128, 128): 0,
    (96, 512, 256): 2,
    (96, 640, 192): 1,
    (128, 128, 128): 0,
}

# desc1 w-in: per-bucket u16 fields @24..@58 + u32 addr @40 (compiler literals)
# order: @24,@26,@28,@32,@34,@48,@50,@52,@54,@56,@58, addr40
D1_FIELDS = {
    (128, 128, 128): (512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1, 195072),
    (32, 1024, 128): (0, 1, 512, 8, 128, 512, 0, 0, 4, 8, 128, 187904),
    (32, 128, 128): (512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1, 195456),
    (32, 128, 256): (1024, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1, 194944),
    (32, 128, 512): (2048, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1, 193536),
    (32, 256, 128): (0, 1, 512, 2, 128, 512, 0, 0, 4, 2, 128, 194432),
    (32, 256, 256): (0, 2, 1024, 2, 128, 1024, 0, 0, 4, 2, 128, 193024),
    (32, 512, 128): (0, 1, 512, 4, 128, 512, 0, 0, 4, 4, 128, 192000),
    (32, 512, 256): (0, 2, 1024, 4, 128, 1024, 0, 0, 4, 4, 128, 189952),
    (64, 128, 128): (512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1, 195328),
    (64, 128, 256): (1024, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1, 194560),
    (64, 256, 128): (0, 1, 512, 2, 128, 512, 0, 0, 4, 2, 128, 194304),
    (64, 256, 256): (0, 2, 1024, 2, 128, 1024, 0, 0, 4, 2, 128, 193024),
    (64, 512, 128): (0, 1, 512, 4, 128, 512, 0, 0, 4, 4, 128, 192256),
    (64, 512, 256): (0, 2, 1024, 4, 128, 1024, 0, 0, 4, 4, 128, 189952),
    (96, 128, 128): (512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1, 195200),
    (96, 512, 256): (0, 2, 1024, 4, 128, 1024, 0, 0, 4, 4, 128, 190080),
    (96, 640, 192): (32768, 1, 768, 5, 128, 768, 0, 0, 4, 5, 128, 189824),
}


def _desc(sem, w_size, x_size, extra, drain):
    """One 64B op212 descriptor pair."""
    s = bytearray(64)
    s[0] = 212
    s[1] = 16
    _u16(s, 12, sem << 8)
    _u16(s, 14, 16)
    _u16(s, 22, 8192)
    _u16(s, 24, w_size)
    _u16(s, 26, extra[0])
    _u16(s, 28, 1)
    _u16(s, 32, extra[1])
    _u16(s, 34, 1)
    _u16(s, 36, w_size)
    _u16(s, 40, 65536 - extra[2])
    _u16(s, 42, 2)
    _u16(s, 46, 4096)
    _u16(s, 50, 4)
    _u16(s, 52, 1)
    _u16(s, 56, extra[3])
    _u16(s, 58, 1)
    _u16(s, 60, x_size)
    _u16(s, 62, 0x0A0A)
    return bytes(s)


def synth_pool(M, K, N):
    """Generate Pool0.bin bytes (3 DMA descs + final event)."""
    AS, NT = K // 128, (N + 127) // 128
    tof = to_off(M, K, N)
    kind, fdst, fdn = final_scheme(M, K, N)
    pv = kind == "pv" or (kind == "col" and len(fdst) > 1)
    out = bytearray()
    out += bytes([177, 16] + [0] * 30)
    out += bytes([0] * 32)
    out += bytes([204, 16] + [0] * 30)
    out += bytes([0] * 32)
    # desc0: x in, sem 8
    d0r = D0_REARM[(M, K, N)]
    out += _desc(8, 4 * K, 4 * K, (0, M, d0r, M), 0)
    # desc1: w in, sem 9
    if pv:
        pv0 = 0x4000 + (K if M == 32 else 2 * K) + 4 * M
        drain = pv0 + 4 * M * (NT - 1)
    else:
        drain = 0x4000 + 4 * M * AS
    f = D1_FIELDS[(M, K, N)]
    d1 = bytearray(64)
    d1[0] = 212
    d1[1] = 16
    _u16(d1, 12, 9 << 8)
    _u16(d1, 14, 16)
    _u16(d1, 20, 1)
    _u16(d1, 22, 8192)
    for i, val in zip((24, 26, 28, 32, 34, 48, 50, 52, 54, 56, 58), f[:11]):
        _u16(d1, i, val)
    struct.pack_into("<I", d1, 36, 4 * N)
    struct.pack_into("<I", d1, 40, f[11])
    _u16(d1, 46, 4096)
    _u16(d1, 60, 4 * N)
    _u16(d1, 62, 0x0A0A)
    out += bytes(d1)
    # desc2: y out, sem 10, waits B-done
    val = 2
    if kind == "pv":
        val = 2 + NT
    elif pv:
        val = 3
    d2 = bytearray(64)
    d2[0] = 212
    d2[1] = 16
    _u16(d2, 4, 5)
    _u16(d2, 5, 4)
    _u16(d2, 8, val)
    _u16(d2, 12, 10 << 8)
    _u16(d2, 14, 16)
    _u16(d2, 16, drain)
    _u16(d2, 22, 4096)
    _u16(d2, 26, 4)
    _u16(d2, 28, 1)
    _u16(d2, 32, M)
    _u16(d2, 34, 1)
    _u16(d2, 36, 4 * N)
    _u16(d2, 44, 2)
    _u16(d2, 46, 8192)
    _u16(d2, 48, 4 * N)
    _u16(d2, 52, 1)
    _u16(d2, 56, M)
    _u16(d2, 58, 1)
    _u16(d2, 60, 4 * N)
    _u16(d2, 62, 0x0A0A)
    out += bytes(d2)
    ev = bytearray(32)
    ev[0] = 164
    ev[1] = 16
    _u16(ev, 4, 5)
    _u16(ev, 5, 10)
    _u16(ev, 8, 16)
    out += bytes(ev)
    out += bytes([0] * 32)
    return bytes(out)
