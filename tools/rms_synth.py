#!/usr/bin/env python3
"""Own synthesis of the RMSNorm engine streams (DVE reduce + ACT chain).

y = x * rsqrt(mean(x^2, -1) + eps), x [M, K] f32. NO PE engine.

Structure (decoded from rms128 [128,128] + rms256 [256,128] oracles):

DVE0 (256B):
  slot192 = op66 TENSOR_REDUCE_ARITH_OP: wait (5,4) = pool-in sem4,
    u16@210 = 1 if M <= 128 else K, u16@216 = K, u16@218 = P (M/128),
    u16@236 = 0x4000 + 4M + 8 (PSUM reduce-result base, in op10 @224),
    u16@244 = P.
Activation0: prologue [177][204][160][35] + pairs:
  setup (op33+op122): u16@272 = -4M, u16@282 = M, u32@296 = K*K,
    u16@316 = 0x4008, f32@292 = 1.0;
  sqrt/div (op33+op170): u16@330 = P, u16@336 = 0x4000+4M+8,
    f32@356 = 1/K, f32@360 = eps, u16@368 = 0x4000+4M+0x10, @374 = P;
  y-passes j = 0..P-1 (see _y_pair; P==1 -> rms128 form exactly).
Pool0: [op212 d0 sem8][op128][op67+op10][op212 d1 sem9][op128][op164]:
  u16@152/164/188 = 2P, @168 = -4M, @218/250 = P, @272 = 0x4000+4M+8,
  @292/304/316 = 2P.

Verified byte-exact vs rms128 (M=128) and rms256 (M=256) oracles.
Constants = rms128 oracle bytes (M=128, K=128, eps=1e-6 base).
"""
import struct
from pathlib import Path


def _p(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


_BASE = Path(__file__).parent / "rms_oracles" / "rms128"
_DVE = (_BASE / "DVE0.bin").read_bytes()
_POOL = (_BASE / "Pool0.bin").read_bytes()
_ACT_BASE = (_BASE / "Activation0.bin").read_bytes()


def synth_dve(M: int, K: int) -> bytes:
    P = M // 128
    d = bytearray(_DVE)
    _p(d, 210, 1 if M <= 128 else K)
    _p(d, 218, P)
    _p(d, 236, 0x4000 + 4 * M + 8)
    _p(d, 248, P)
    return bytes(d)


def _y_pair(j, P, M, K):
    """(op33, op170) 64B for y-pass j (0-based), cloned from rms128 y0."""
    last = (j == P - 1) and P > 1
    first = (j == 0)
    a = bytearray(_ACT_BASE[384:448])     # rms128 y0 pair = clone base
    if last:
        struct.pack_into("<H", a, 4, 0)                      # wait (0,0)
        struct.pack_into("<H", a, 8, 0)                      # phase 0
    elif not (first and P == 1):
        struct.pack_into("<H", a, 6, 0)                      # const (13,04)->0
    _p(a, 16, (-4 * M) if first else (-4 * K))
    _p(a, 26, K)
    _p(a, 32 + 4, 0x4000 + 4 * M + 0x18 + 4 * j)             # op170@4
    _p(a, 32 + 16, 0x4000 + 4 * M + 0x20 + 4 * K * j)        # op170@16
    _p(a, 32 + 26, K)                                        # op170@26
    return bytes(a)


def synth_act(M: int, K: int, eps: float) -> bytes:
    P = M // 128
    a = bytearray(_ACT_BASE)          # 448B = prologue + setup + sqrt/div + y0
    _p(a, 272, -4 * M)                # pair1 op33@16
    _p(a, 282, M)                     # pair1 op33@26
    struct.pack_into("<I", a, 296, K * K)
    _p(a, 346, P)                     # pair2 op33@26
    _p(a, 336, 0x4000 + 4 * M + 8)    # pair2 op33@16
    _p(a, 314, M)                     # op122@26 (per-pass rows)
    _p(a, 378, P)                     # pair2 op170@26
    struct.pack_into("<f", a, 356, 1.0 / K)
    struct.pack_into("<f", a, 360, eps)
    _p(a, 368, 0x4000 + 4 * M + 0x10)  # pair2 op170@16
    head = bytes(a[:384])
    out = bytearray(head)
    for j in range(P):
        out += _y_pair(j, P, M, K)
    return bytes(out)


def synth_pool(M: int, K: int) -> bytes:
    P = M // 128
    p = bytearray(_POOL)
    _p(p, 152, 4 * M)
    _p(p, 164, 4 * M)
    _p(p, 168, -4 * M)
    _p(p, 188, 4 * M)
    _p(p, 218, P)
    _p(p, 250, P)
    _p(p, 208, 0x4000 + 4 * M + 0x10)
    _p(p, 240, 0x4000 + 4 * M + 0x18)
    _p(p, 272, 0x4000 + 4 * M + 0x20)
    _p(p, 292, 4 * M)
    _p(p, 304, 4 * M)
    _p(p, 316, 4 * M)
    return bytes(p)


def synth_streams(M: int, K: int, eps: float):
    return {"DVE0.bin": synth_dve(M, K), "Activation0.bin": synth_act(M, K, eps),
            "Pool0.bin": synth_pool(M, K)}


PROLOGUE = bytes.fromhex("".join([
    "b110000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "cc10000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000"]))
