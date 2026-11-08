#!/usr/bin/env python3
"""Pack a NEFF: 1024-byte header + gzip(tar archive).

Usage: neff_emit.py <dir-with-tar-contents> <out.neff> [--name FILENAME]
The directory must contain what the tar should hold (info.json, neff.json, sg00/...).
Header fields beyond the verified ones are zero-filled; digest A = md5(gzip stream).
TODO: digest B + tail fields (validator currently does not require them).
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
from pathlib import Path


def build_tar(src: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(src)), recursive=False)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    src = Path(args.src)
    name = (args.name or src.name)[:63].encode()

    tar = build_tar(src)
    gz = gzip.compress(tar, mtime=0)

    hdr = bytearray(1024)
    struct.pack_into("<I", hdr, 0x00, 2)              # version
    struct.pack_into("<Q", hdr, 0x08, 0x400)          # payload offset
    struct.pack_into("<Q", hdr, 0x10, len(gz))        # payload size
    struct.pack_into("<Q", hdr, 0x18, 2)
    struct.pack_into("<I", hdr, 0xA8, 1)              # digest count
    md5 = hashlib.md5(gz).digest()
    swapped = b"".join(struct.pack("<I", *struct.unpack(">I", md5[i:i + 4])) for i in range(0, 16, 4))
    hdr[0xAC:0xBC] = swapped
    hdr[0xDC:0xDC + len(name)] = name

    Path(args.out).write_bytes(bytes(hdr) + gz)
    print(f"wrote {args.out}: hdr=1024B tar={len(tar)}B gz={len(gz)}B total={1024 + len(gz)}B")


if __name__ == "__main__":
    main()
