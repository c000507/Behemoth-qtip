#!/usr/bin/env python3
"""DMA-only bandwidth probe containers (no DVE/PE/ACT compute).

Base = the validated 1-input shr container (bvw_shr): pool desc0 = HBM
x-in (sem 8, no wait), desc1 = SB y-out (waits DVE notify sem 6).

DMA-only patch: DVE0 = prologue only; desc1 wait idx 6 -> 8 (wait on
desc0's completion semaphore, GE value kept); ACT/PE/SP = idle
prologues already. Numerics: out0 == in0 (pure DMA passthrough).

Sizes: def.json var sizes = bytes; neff.json shapes = [n_e] int32.
Optional: patch qPoolDynamic.num_queues (queue/parallelism sweep).

Usage: dma_probe.py OUTDIR N_ELEMS [--queues Q]
"""
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

BASE = Path("/tmp/opencode/bvw_shr")  # validated wrapped container


def pu16(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


def build(outdir: Path, n_e: int, queues: int = None):
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(BASE, out)
    sg = out / "sg00"

    # strip the DVE work: prologue only
    dve = open(sg / "DVE0.bin", "rb").read()[:128]
    (sg / "DVE0.bin").write_bytes(dve)

    # Chain desc0 (x-in) -> desc1 (y-out) via desc0's AUTO completion
    # increment: an assigned sem (u32@12 = sem<<8) gets +16 when the
    # transfer completes (DVE waits GE 16 on sem 8 in the validated
    # container). desc1 = wait@6 (idx 8, mode 5) + GE value u32@8 = 16
    # (the stock value 0x00010000 = 65536 = one full DVE INC_COMPLETE
    # n*16 - replace with 16 = one desc completion).
    pool = bytearray(open(sg / "Pool0.bin", "rb").read())
    pu16(pool, 192 + 4, (8 << 8) | 5)   # desc1 wait: sem 8 GE (wait = u16@4)
    pu16(pool, 192 + 8, 16)             # desc1 GE value = 16 (one desc completion)
    # 4P fields = BYTES PER CHANNEL (n*4B / 128 lanes) - scale for n != 4096
    P = n_e // 128
    assert n_e % 128 == 0 and P > 0, "n_e must be a positive multiple of 128"
    for rel in (24, 36, 60):
        pu16(pool, 128 + rel, 4 * P)    # desc0 (x-in)
    pu16(pool, 128 + 40, (1 << 16) - 4 * P)  # desc0 rel40 = -4P
    for rel in (36, 48, 60):
        pu16(pool, 192 + rel, 4 * P)    # desc1 (y-out)
    (sg / "Pool0.bin").write_bytes(bytes(pool))

    # sizes: input0/output0 = n_e int32 bytes
    dj = json.loads((sg / "def.json").read_text())
    dj["var"]["input0"]["size"] = n_e * 4
    dj["var"]["output0"]["size"] = n_e * 4
    if queues is not None:
        dj["dma_queue"]["qPoolDynamic"]["num_queues"] = queues
    (sg / "def.json").write_text(json.dumps(dj))
    nj = json.loads((out / "neff.json").read_text())
    nj["attrs"]["shape"] = ["list_shape", [[n_e], [n_e]]]
    nj["metadata"]["signatures"]["default"]["inputs"]["input0"]["shape"] = [n_e]
    (out / "neff.json").write_text(json.dumps(nj))

    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out.parent / f"{out.name}.neff"), "--name", out.name],
                   check=True, capture_output=True)
    return out.parent / f"{out.name}.neff"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    n_e = int(sys.argv[2])
    q = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--queues" else None
    print(build(out, n_e, q))
