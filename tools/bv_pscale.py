#!/usr/bin/env python3
"""P-scaling probe builder for DVE BITVEC TS chains.

Builds probe containers from the validated 1-input shr oracle
(/tmp/opencode/oracle_shr/sg00, n=4096, P=32) by patching:

  DVE op83 head : rel16 = -4P, rel26 = P        (op8 carrier: rel26 = P)
  Pool desc0    : rel{22,34,54,58} = 4P, rel38 = -4P
  Pool desc1    : rel{32,36,48,56,60} = 4P

Field map derived from byte-diff of fresh shr oracles n=2048/4096/8192
(P = n/128; pool otherwise n-independent - tensor sizes come from the
DGE/shape side). Probes decouple P from n at fixed n=4096.

Usage: bv_pscale.py OUTDIR P R [--rel48 V]
"""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bv_synth import ts_bitvec, UPD_LAST

ORACLE = Path("/tmp/opencode/bvw_shr")  # WRAPPED container (kelf/neff.json + sg00)


def u16(buf, off):
    return struct.unpack_from("<H", buf, off)[0]


def pu16(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


def patch_dve_P(dve: bytes, P: int) -> bytes:
    """Scale P fields in every op83/op8 64B pair (prologue 6x32B untouched).

    TS law (from shr oracles n=2048/4096/8192): op83 head rel16 = -4P,
    rel26 = P; op8 carrier rel26 = P (carrier rel16 = dst, untouched).
    op81 TT uses a different rel16 law (0xFFF0 @P=32) - not patched here.
    """
    d = bytearray(dve)
    i = 0
    while i + 64 <= len(d):
        if d[i] in (83, 81) and d[i + 1] == 16:
            if d[i] == 83:
                pu16(d, i + 16, -4 * P)
                pu16(d, i + 26, P)
                pu16(d, i + 32 + 26, P)   # op8 carrier P
            else:
                pu16(d, i + 26, P)        # op81: rel26 only (rel16 law unknown)
            i += 64
        else:
            i += 32
    return bytes(d)


def patch_pool_P(pool: bytes, P: int, rel48_override=None) -> bytes:
    """2-desc pool (shr container): scale the 4P fields."""
    p = bytearray(pool)
    for rel in (24, 36, 60):          # desc0: 4P
        pu16(p, 128 + rel, 4 * P)
    pu16(p, 128 + 40, -4 * P)         # desc0: -4P
    for rel in (36, 48, 60):          # desc1: 4P
        pu16(p, 192 + rel, 4 * P)
    if rel48_override is not None:
        pu16(p, 192 + 48, rel48_override)
    return bytes(p)


def build(outdir: Path, P: int, R: int, rel48_override=None):
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "sg00").mkdir(parents=True)
    for f in ORACLE.iterdir():
        if f.is_file():
            shutil.copy(f, out / f.name)
    for f in (ORACLE / "sg00").iterdir():
        shutil.copy(f, out / "sg00" / f.name)
    sg = out / "sg00"

    dve = open(sg / "DVE0.bin", "rb").read()
    assert len(dve) == 192 and dve[128] == 83 and dve[160] == 8

    chain = []
    for _ in range(R):
        chain.append(("shr", 0, 1, 0x4000))
        chain.append(("shl", 0x4000, 1, 0x4000))
    ops = bytearray()
    for i, (op, src, imm, dst) in enumerate(chain):
        ops += ts_bitvec(op, src, imm, dst, P)
        if i == len(chain) - 1:
            pu16(ops, len(ops) - 64 + 6, UPD_LAST)  # final op head upd
    dve_new = patch_dve_P(dve[:4 * 32] + bytes(ops), P)
    (sg / "DVE0.bin").write_bytes(dve_new)

    pool = patch_pool_P(open(sg / "Pool0.bin", "rb").read(), P, rel48_override)
    (sg / "Pool0.bin").write_bytes(pool)

    out_neff = out.parent / f"{out.name}.neff"
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out_neff), "--name", out.name], check=True)
    return out_neff


if __name__ == "__main__":
    outdir = Path(sys.argv[1])
    P = int(sys.argv[2])
    R = int(sys.argv[3])
    rel48 = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == "--rel48" else None
    print(build(outdir, P, R, rel48))
