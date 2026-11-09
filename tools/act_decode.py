#!/usr/bin/env python3
"""Decode Activation0.bin for matmul oracles.

Stream layout (verified): 32B slots. Prologue = [op177][pad][op204][pad]
[op35][pad] then N x ([op33][op170]) pairs. Some buckets have a tail slot.
Prints per-bucket pair table + field correlations.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

T = json.load(open(Path(__file__).parent / "shape_templates.json"))
PAD = bytes(32)


def slots(buf):
    return [buf[i:i + 32] for i in range(0, len(buf), 32)]


def u16(s, off):
    return int.from_bytes(s[off:off + 2], "little")


def decode(buf):
    sl = slots(buf)
    ops = []
    for i, s in enumerate(sl):
        ops.append((i, s[0], s[1], s))
    return sl, ops


def main():
    keys = sorted(T, key=lambda x: tuple(map(int, x.split("_"))))
    print(f"{'bucket':>14} len  prologue(177/204/35 first-diff-vs-32_128_128)"
          f"  pairs  tail")
    for k in keys:
        M, K, N = map(int, k.split("_"))
        buf = bytes.fromhex(T[k]["act_hex"])
        sl = slots(buf)
        opcodes = [s[0] for s in sl]
        # prologue: slots 0..5 = 177,pad,204,pad,35,pad
        pro = [s for s in sl[:6]]
        n = (len(sl) - 6) // 2
        tail = None
        if len(sl) % 2 == 1 and len(sl) > 6:
            tail = sl[-1]
        print(f"{k:>14} {len(sl):3}  {pro[0][0]}/{pro[2][0]}/{pro[4][0]}"
              f"  pairs={n}  tail_op={tail[0] if tail else '-'}")


if __name__ == "__main__":
    main()
