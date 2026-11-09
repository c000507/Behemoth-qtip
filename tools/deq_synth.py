#!/usr/bin/env python3
"""Own synthesis of the dequant engine streams (PE cast + DVE s-broadcast).

y = q.cast(f32) * s[None, :], q uint8 [M, K], s f32 [K].

Cloned from dequant128 [128,128] oracle, K-parameters from
dq128x256 [128,256] oracle diff (M128 family; ACT invariant):
  DVE: u16@272 = -K, u16@282 = K, u16@298 = K, u16@314 = K
  POOL: @164 = @188 = @292 = @304 = @316 = 4K; @168 = -5K;
        @216 = @228 = @252 = K; @248 = M; @232 = -K
  PE0: u16@272 = -5K, u16@282 = K, u16@314 = K (PE0 differs from DVE!)
  SP0: no shape fields.
Byte-exact vs dequant128 (128,128) and dq128x256 (128,256) oracles.
Device test pending.
VALIDATED = M = 128, K in {128, 256} (interpolation K192 unverified).
M64 family (deq64x256 oracle) has a different stream structure
(PE 1280B, ACT 512B, own def.json queue layout) - NOT covered here.
"""
import struct
from pathlib import Path

_BASE = Path(__file__).parent / "oracle_sg00" / "dequant128"
_DVE = (_BASE / "DVE0.bin").read_bytes()
_ACT = (_BASE / "Activation0.bin").read_bytes()
_POOL = (_BASE / "Pool0.bin").read_bytes()
_PE = (_BASE / "PE0.bin").read_bytes()
_SP = (_BASE / "SP0.bin").read_bytes()


def _p(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


def synth_streams(M: int, K: int):
    assert M == 128, "dequant synth M128 family only"
    d = bytearray(_DVE)
    p = bytearray(_POOL)
    e = bytearray(_PE)
    _p(d, 272, -K)
    _p(d, 282, K)
    _p(d, 298, K)
    _p(d, 314, K)
    for off in (164, 188, 292, 304, 316):
        _p(p, off, 4 * K)
    _p(p, 168, -5 * K)
    _p(p, 216, K)
    _p(p, 228, K)
    _p(p, 232, -K)
    _p(p, 248, M)
    _p(p, 252, K)
    _p(e, 272, -5 * K)
    _p(e, 282, K)
    _p(e, 314, K)
    return {"DVE0.bin": bytes(d), "Activation0.bin": bytes(_ACT),
            "Pool0.bin": bytes(p), "PE0.bin": bytes(e),
            "SP0.bin": bytes(_SP)}


VALIDATED = {(128, 128), (128, 256)}
