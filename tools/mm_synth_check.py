#!/usr/bin/env python3
"""Byte-exact check: synth() vs all 18 oracle PE streams."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import mm_synth

t = json.load(open(Path(__file__).parent / "shape_templates.json"))
ok = 0
for k in sorted(t, key=lambda x: tuple(map(int, x.split("_")))):
    M, K, N = map(int, k.split("_"))
    want = bytes.fromhex(t[k]["pe_hex"])
    try:
        got = mm_synth.synth(M, K, N)
    except Exception as e:
        print(f"{k:>14}: EXC {e}")
        continue
    if got == want:
        print(f"{k:>14}: OK")
        ok += 1
        continue
    # locate first divergence in 32B slots
    n = min(len(got), len(want))
    slot = None
    for off in range(0, n, 32):
        if got[off:off+32] != want[off:off+32]:
            slot = off // 32
            break
    if slot is None and len(got) != len(want):
        print(f"{k:>14}: DIFF len got={len(got)} want={len(want)}")
        continue
    if slot is None:
        print(f"{k:>14}: OK?!")
        continue
    fields = []
    for i in range(0, 32, 2):
        w = int.from_bytes(want[slot*32+i:slot*32+i+2], "little")
        g = int.from_bytes(got[slot*32+i:got.__len__() and slot*32+i+2], "little")
        if w != g:
            fields.append(f"@{i}: want {w:#06x} got {g:#06x}")
    print(f"{k:>14}: DIFF slot {slot}: " + "; ".join(fields))
print(f"{ok}/18 byte-exact")
