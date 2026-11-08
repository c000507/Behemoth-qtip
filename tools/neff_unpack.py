#!/usr/bin/env python3
"""Unpack a NEFF (Neuron Executable File Format) container.

NEFF v2 = 1024-byte header + gzip-compressed tar archive.
Usage: neff_unpack.py graph.neff [OUTDIR]
Prints header summary and extracts the archive to OUTDIR (default <name>_x).
"""
import gzip
import json
import struct
import sys
import tarfile
from pathlib import Path


def parse_header(data: bytes):
    if len(data) < 1024:
        raise ValueError(f"too small: {len(data)} bytes")
    ver, zero = struct.unpack_from("<II", data, 0)
    payload_off, payload_len, field18 = struct.unpack_from("<QQQ", data, 8)
    md5_count, = struct.unpack_from("<I", data, 0xA8)
    name = data[0xDC:0xDC + 64].split(b"\0")[0].decode("ascii", "replace")
    if data[payload_off:payload_off + 3] != b"\x1f\x8b\x08":
        raise ValueError(f"no gzip at 0x{payload_off:x}")
    return {
        "version": ver,
        "reserved_0x04": zero,
        "payload_offset": payload_off,
        "payload_size": payload_len,
        "file_size": len(data),
        "field_0x18": field18,
        "md5_count": md5_count,
        "name": name,
        "digest_a_md5": data[0xAC:0xBC].hex(),
        "digest_b_md5": data[0xCC:0xDC].hex(),
    }


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix("")
    data = src.read_bytes()
    hdr = parse_header(data)
    print(json.dumps(hdr, indent=2))
    payload = gzip.decompress(data[hdr["payload_offset"]:hdr["payload_offset"] + hdr["payload_size"]])
    with tarfile.open(fileobj=__import__("io").BytesIO(payload)) as tf:
        tf.extractall(out, filter="data")
        members = tf.getnames()
    print(f"extracted {len(members)} members to {out}")
    for p in sorted(out.rglob("*.json")):
        try:
            json.loads(p.read_text())
        except Exception as e:
            print(f"  WARN bad json {p}: {e}")


if __name__ == "__main__":
    main()
