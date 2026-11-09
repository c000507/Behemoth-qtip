#!/usr/bin/env python3
"""Own synthesis of the softmax engine streams (max/sum DVE + ACT exp).

y = softmax(x, -1), x [M, K] f32. NO PE.

Structure (decoded from softmax128 [128,128] + sf M96 K384 oracles):
  DVE = max-reduce (op66@128), row-prepare (op67@192 + op10@224),
  sum-reduce (op66@256 + op10@288), row-div (op72@320 + op10@352).
  ACT = setup pair (op33 + op170 w/ f32 1.0 + u32 0x408 payload) + exp pair.
  Shape fields (verified on both oracles):
    DVE op66#1: u16@140 = -4K, u16@152 = K
    DVE op10#1: u16@162 = M, @172 = 0x4000
    DVE op10#2: u16@226 = M
    DVE op66#2: u16@280 = K
    DVE op10#3: u16@290 = M, u16@300 = 0x4000 + 4K + 0x10
    DVE op72:   u16@332 = 0x4000 + 4K + 0x10
    DVE op10#4: u16@354 = M, u16@364 = 0x4000 + 4K + 0x18
    ACT pair1:  u16@208 = -4K, u16@218 = K,
                u16@226 = 0x780 - 0x20*max(0, (128-M)//32), u16@250 = K
    ACT pair2:  u16@272 = 0x4010, u16@290 = 0x8000+M, u16@282 = K, u16@292 = 0x4000+4K+0x18,
                u16@304 = 0x4000+4K+0x20, u16@314 = K
    POOL @152/164/188/228/240/252 = 4K; @160/184/224/248 = M;
         @168 = -4K; @208 = 0x4000 + 4K + 0x20
  Remaining bytes = constants (f32 -inf / 1.0 seeds, LUT family).
  NOTE: all PSUM/rearm windows scale with K (not M) for softmax.
"""
import struct
from pathlib import Path

_BASE = Path(__file__).parent / "oracle_sg00" / "softmax128"
_DVE = (_BASE / "DVE0.bin").read_bytes()
_ACT = (_BASE / "Activation0.bin").read_bytes()
_POOL = (_BASE / "Pool0.bin").read_bytes()


def _p(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


def synth_dve(M: int, K: int) -> bytes:
    d = bytearray(_DVE)
    _p(d, 140, -4 * K)                   # op66#1 @128
    _p(d, 152, K)
    _p(d, 162, M)                        # op10#1 @160
    _p(d, 226, M)                        # op10#2 @224
    _p(d, 280, K)                        # op66#2 @256
    _p(d, 290, M)                        # op10#3 @288
    _p(d, 300, 0x4000 + 4 * K + 0x10)
    _p(d, 332, 0x4000 + 4 * K + 0x10)    # op72 @320
    _p(d, 354, M)                        # op10#4 @352
    _p(d, 364, 0x4000 + 4 * K + 0x18)
    return bytes(d)


def synth_act(M: int, K: int) -> bytes:
    a = bytearray(_ACT)
    _p(a, 208, -4 * K)                   # pair1 op33
    _p(a, 218, K)
    _p(a, 226, 0x780 - 0x20 * max(0, (128 - M) // 32))
    _p(a, 250, K)
    _p(a, 290, 0x8000 + M)               # pair2 op170 b2..3
    _p(a, 282, K)                        # pair2 op33
    _p(a, 292, 0x4000 + 4 * K + 0x18)    # pair2 op170
    _p(a, 304, 0x4000 + 4 * K + 0x20)
    _p(a, 314, K)
    return bytes(a)


def synth_pool(M: int, K: int) -> bytes:
    p = bytearray(_POOL)
    _p(p, 152, 4 * K)
    _p(p, 160, M)
    _p(p, 164, 4 * K)
    _p(p, 168, -4 * K)
    _p(p, 184, M)
    _p(p, 188, 4 * K)
    _p(p, 208, 0x4000 + 4 * K + 0x20)
    _p(p, 224, M)
    _p(p, 228, 4 * K)
    _p(p, 240, 4 * K)
    _p(p, 248, M)
    _p(p, 252, 4 * K)
    return bytes(p)


def synth_streams(M: int, K: int):
    return {"DVE0.bin": synth_dve(M, K), "Activation0.bin": synth_act(M, K),
            "Pool0.bin": synth_pool(M, K)}


# device-verified shapes (AGENTS.md item 30)
VALIDATED = {(128, 128)}
