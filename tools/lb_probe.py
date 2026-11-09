#!/usr/bin/env python3
"""Low-bit weight probes, full-container patches (jsons + npy + streams).

Decoded laws (compiler oracles mm32_f32/mm32_bf16/mm32_mixed, 32x128x128):
- in_dtype byte @+32 of LW(op1)/MM(op2): FP32=10 BFLOAT16=6 FP8_EXP4=14.
- Compiler bf16-vs-f32 B-pair diffs: B-LW in_dtype + head@16 = 0x4000 +
  (M-1)*elem (0x407c -> 0x403e); B-MM = in_dtype ONLY. A-pair (x-side)
  also changes when x is 16-bit.
- Pool desc: rel{24,36,60} = 0x0200(f32)/0x0100(bf16) stride fields;
  rel62 = dtype marker pair 0x0a0a/0x0606.
- def.json var sizes = BYTES (input1 65536->32768); identity npy =
  eye(128).reshape(128,128,1,1) in elem dtype (f32 64KB / bf16 32KB,
  descr <f4 / <V2); tensor_map.json dtypes; neff.json dltype.
- Device: both-bf16 compiler container = BIT-EXACT vs torch bf16;
  compiler mixed (x f32, w bf16) = NaN.

Probes (32x128x128):
  wbf16 : w-only bf16 (x/y f32) on the f32 container.
  fp8   : both-fp8 from the compiler bf16 container (in_dtype 14,
          /2-rescaled fields, identity = fp8 eye, y f32).
"""
import io
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path("/tmp/opencode/lowbit/mm32_f32")
BF16C = Path("/tmp/opencode/lowbit/mm32_bf16")


def pu16(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


def eye_npy(item_bytes: int) -> bytes:
    """eye(128).reshape(128,128,1,1) with 1.0 in the given elem width."""
    if item_bytes == 4:
        a = np.eye(128, dtype=np.float32)
    elif item_bytes == 2:
        e = (np.eye(128, dtype=np.uint16) * 0x3F80)
        a = e.reshape(128, 128, 1, 1).view(np.dtype("V2"))
    elif item_bytes == 1:
        e = (np.eye(128, dtype=np.uint8) * 0x38)   # e4m3 1.0 = 0x38
        a = e.reshape(128, 128, 1, 1).view(np.dtype("V1"))
    else:
        raise ValueError(item_bytes)
    b = io.BytesIO()
    np.save(b, a)
    return b.getvalue()


def patch_jsons(out: Path, in0: int, in1: int, out0: int, dts, id_item: int,
                id_name: str):
    dj = json.loads((out / "sg00" / "def.json").read_text())
    dj["var"]["input0"]["size"] = in0
    dj["var"]["input1"]["size"] = in1
    dj["var"]["output0"]["size"] = out0
    # identity: edit IN PLACE (order-sensitive parser breaks on re-add;
    # key = identity_<NN>_sg0000 with whatever suffix the base used)
    old_id = [k for k in dj["var"] if k.startswith("identity_")]
    assert len(old_id) == 1, old_id
    iv = dj["var"][old_id[0]]
    iv["size"] = 128 * 128 * id_item
    iv["alignment"] = 128 * 128 * id_item
    iv["file_name"] = f"value_sg0000_{old_id[0]}_CRSM.npy"
    (out / "sg00" / "def.json").write_text(json.dumps(dj))
    tm = json.loads((out / "sg00" / "tensor_map.json").read_text())
    tm["input0"]["dtype"] = dts[0]
    tm["input1"]["dtype"] = dts[1]
    (out / "sg00" / "tensor_map.json").write_text(json.dumps(tm))
    nj = json.loads((out / "neff.json").read_text())
    nj["attrs"]["dltype"] = ["list_str", list(dts) + ["float32"]]
    sig = nj["metadata"]["signatures"]["default"]["inputs"]
    if "input0" in sig:
        sig["input0"]["dtype"] = dts[0]
    if "input1" in sig:
        sig["input1"]["dtype"] = dts[1]
    (out / "neff.json").write_text(json.dumps(nj))
    (out / "sg00" / iv["file_name"]).write_bytes(eye_npy(id_item))


def emit(srcdir: Path, out: Path):
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(srcdir, out)
    return tar(out)


def tar(out: Path):
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out.parent / f"{out.name}.neff"), "--name", out.name],
                   check=True, capture_output=True)
    return out.parent / f"{out.name}.neff"


def main():
    pd = Path("/tmp/opencode/lbprobe")
    pd.mkdir(exist_ok=True)

    # V1: w-only bf16 on f32 container
    out = pd / "wbf16"
    emit(BASE, out)
    pe = bytearray(open(out / "sg00" / "PE0.bin", "rb").read())
    pe[512 + 32] = 6
    pu16(pe, 512 + 16, 0x4000 + 31 * 2)
    pe[576 + 32] = 6
    (out / "sg00" / "PE0.bin").write_bytes(bytes(pe))
    pool = bytearray(open(out / "sg00" / "Pool0.bin", "rb").read())
    for rel in (24, 36, 60):
        pu16(pool, 192 + rel, 0x0100)
    pu16(pool, 192 + 62, 0x0606)
    (out / "sg00" / "Pool0.bin").write_bytes(bytes(pool))
    patch_jsons(out, 16384, 32768, 16384,
                ("float32", "bfloat16"), 4, "identity_37")
    print(tar(out))

    # V3: both-fp8 from compiler bf16 container
    out = pd / "fp8"
    emit(BF16C, out)
    pe = bytearray(open(out / "sg00" / "PE0.bin", "rb").read())
    for off in (192 + 32, 256 + 32, 384 + 32, 448 + 32):
        pe[off] = 14
    pu16(pe, 192 + 16, 0x10000 - 33)          # A-LW step -33
    pu16(pe, 256 + 16, 0x10000 - 32)          # A-MM @16 -32
    pu16(pe, 384 + 16, 0x4000 + 31)           # B-LW addr
    (out / "sg00" / "PE0.bin").write_bytes(bytes(pe))
    pool = bytearray(open(out / "sg00" / "Pool0.bin", "rb").read())
    for d in (128, 192, 256):
        for rel in (24, 36, 60):
            pu16(pool, d + rel, 0x0080)
        pu16(pool, d + 62, 0x0e0e)
    (out / "sg00" / "Pool0.bin").write_bytes(bytes(pool))
    patch_jsons(out, 4096, 16384, 16384,
                ("float8_e4m3fn", "float8_e4m3fn"), 1, "identity_39")
    print(tar(out))


if __name__ == "__main__":
    main()
