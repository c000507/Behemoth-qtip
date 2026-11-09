#!/usr/bin/env python3
"""Detailed per-pair field dump + prologue byte-diff across all buckets."""
import json
import sys
from pathlib import Path

T = json.load(open(Path(__file__).parent / "shape_templates.json"))


def slots(buf):
    return [buf[i:i + 32] for i in range(0, len(buf), 32)]


def u16(s, off):
    return int.from_bytes(s[off:off + 2], "little")


keys = sorted(T, key=lambda x: tuple(map(int, x.split("_"))))

base = slots(bytes.fromhex(T["32_128_128"]["act_hex"]))[:6]
print("=== prologue slots 0..5 (bucket diffs vs 32_128_128) ===")
for k in keys:
    buf = bytes.fromhex(T[k]["act_hex"])
    sl = slots(buf)
    diffs = []
    for i in range(6):
        if sl[i] != base[i]:
            fb = [f"@{j}: {base[i][j]:#x}->{sl[i][j]:#x}"
                  for j in range(32) if base[i][j] != sl[i][j]]
            diffs.append(f"s{i}[{fb}]")
    print(f"{k:>14}: {'IDENTICAL' if not diffs else '; '.join(diffs)}")

print()
print("=== full prologue hex (32_128_128) ===")
for i, s in enumerate(base):
    print(f"slot{i} op={s[0]:3} wl={s[1]:#x}: {s.hex()}")

print()
hdr = (f"{'bucket':>14} {'p':>2} {'@8':>4} {'@16':>4} {'@26':>5} "
       f"{'170@2':>6} {'170@16':>7} {'170@26':>7} {'other diffs'}")
print(hdr)
for k in keys:
    M, K, N = map(int, k.split("_"))
    buf = bytes.fromhex(T[k]["act_hex"])
    sl = slots(buf)[6:]
    for p in range(len(sl) // 2):
        a = sl[2 * p]      # op33
        b = sl[2 * p + 1]  # op170
        # any u16 not in the standard list?
        std_a = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}
        extra = []
        for off in range(2, 32, 2):
            pass
        print(f"{k:>14} {p:>2} {u16(a,8):>4} {u16(a,16):>4} {u16(a,26):>5} "
              f"{u16(b,2):>6} {u16(b,16):>7} {u16(b,26):>7}  "
              f"A:{[u16(a,o) for o in (4,6,10,12,14,18,20,22,24,28,30)]} "
              f"B:{[u16(b,o) for o in (0,4,6,8,10,12,14,18,20,22,24,28,30)]}")
