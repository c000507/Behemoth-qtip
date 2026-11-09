#!/usr/bin/env python3
"""Synthesize Activation0.bin for matmul oracles from (M, K, N).

Stream layout (verified against 29 oracle buckets in shape_templates.json):
  32B slots: prologue [op177][pad][op204][pad][op35][pad]  (byte-identical
  across all buckets), then one [op33 ACTIVATE][op170 TENSOR_LOAD] pair per
  "pass".  A pass drains a chunk of PSUM state through the SB window at
  0x4000 (op170@16 = 0x4000 + 4*cumulative_elems).

Pass structure (counts in f32 elems):
  pass 1        = M*AS  (A-phase x-temp drain, AS = K//128)
  middles       = final-tile gather, only when the PE stream has a PV/tile
                  phase (see _middles)
  last pass     = N     (final y drain)

Rules derived from the 29-bucket corpus:
  middles:
    - K==128 col mode (tiles at PSUM|4M*t, contiguous): one pass of
      tiles*M elems when tiles >= 2, none otherwise.  M64 N>256 K128 is
      direct (single PSUM|0 dst) -> no middle.
    - pv mode (B temps in strided banks, M==32 & (K>=512 or (N>=256 &
      K>=256)), or M==64 & N>=256): NT passes of M each when NT <= 2,
      none when NT >= 3 (matches degenerate 64_512_384 oracle: np=2).
    - M==64 & K>=1024 & N<256 (64_1024_128 swapped-roles family):
      one pass of M.
    - direct (everything else, incl. M >= 96): no middles.
  np = 2 + len(middles) is always in {2,3,4} under these rules; the
  op33 semaphore byte (byte17, via u16@16 = sem<<8) depends only on np:
      np=2 -> [8, 0]   np=3 -> [16, 0, 8]   np=4 -> [0, 16, 8, 24]
  op33:  @8 = pass index (1-based), @26 = pass elems, other u16 constant.
  op170: @2 = 0x8080 for all passes except the last, which is 0x8000+M
         (byte2 = M, byte3 = 128); @16 = 0x4000 + 4*cum_prev; @26 = elems.
"""
import json
from pathlib import Path

_T = json.load(open(Path(__file__).parent / "shape_templates.json"))
PROLOGUE = bytes.fromhex(_T["32_128_128"]["act_hex"])[:6 * 32]

_SEM_BY_NP = {2: [8, 0], 3: [16, 0, 8], 4: [0, 16, 8, 24]}


def _u16s(vals):
    out = bytearray()
    for v in vals:
        out += int(v).to_bytes(2, "little")
    return bytes(out)


def _op33(phase, sem, elems):
    return (bytes([33, 16, 0, 0])
            + _u16s([1285, 1043, phase, 0, 1, 256, sem << 8, 512,
                     1, 1, 1, elems, 1, 1]))


def _op170(src, off, elems):
    return (bytes([170, 10]) + (src & 0xFFFF).to_bytes(2, "little")
            + _u16s([0, 0x3F80, 0, 0, 0, 0, off, 0, 1, 1, 1, elems, 1, 1]))


def _middles(M, K, N):
    """Final-tile gather passes between the A-drain and y-drain."""
    if K == 128:
        if N > 256:
            if M == 64:
                return []  # direct single-bank final
            tiles = (N + 127) // 128  # col, dn = M
        else:
            dn = N if (M >= 96 or M == 32) else 128
            tiles = (N + dn - 1) // dn
        return [tiles * M] if tiles >= 2 else []
    if M >= 96:
        return []
    NT = (N + 127) // 128
    pv = (M == 32 and (K >= 512 or (N >= 256 and K >= 256))) or \
         (M == 64 and N >= 256)
    if pv:
        return [M] * NT if NT <= 2 else []
    if M == 64 and K >= 1024 and N < 256:
        return [M]  # swapped-roles K1024 family
    return []


def synth_act(M, K, N):
    """Activation0.bin bytes for the M x K @ K x N matmul oracle."""
    AS = K // 128
    counts = [M * AS] + _middles(M, K, N) + [N]
    np_ = len(counts)
    sems = _SEM_BY_NP[np_]
    out = bytearray(PROLOGUE)
    cum = 0
    for p, elems in enumerate(counts):
        out += _op33(p + 1, sems[p], elems)
        off = 0x4000 + 4 * cum
        src = 0x8000 + M if p == np_ - 1 else 0x8080
        out += _op170(src, off, elems)
        cum += elems
    return bytes(out)


if __name__ == "__main__":
    keys = sorted(_T, key=lambda x: tuple(map(int, x.split("_"))))
    ok = 0
    fails = []
    for k in keys:
        M, K, N = map(int, k.split("_"))
        want = bytes.fromhex(_T[k]["act_hex"])
        got = synth_act(M, K, N)
        if got == want:
            ok += 1
        else:
            fails.append(k)
    print(f"{ok}/{len(keys)}")
    if fails:
        print("FAIL:", ", ".join(fails))
