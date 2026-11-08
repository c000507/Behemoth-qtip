import re
#!/usr/bin/env python3
"""Neuron v2 (trn1) instruction encoder/decoder driven by isa-fields.json.

Field map format: {f"{ver}.{struct}.{field}": {off,size,shr,mask}}
Each instruction = 64 bytes; fields written by (struct,field) key.
"""
import json
import struct
from pathlib import Path

FIELDS_PATH = Path(__file__).with_name("neff-isa-fields.json") if (Path(__file__).parent / "neff-isa-fields.json").exists() else Path("/tmp/opencode/isa-fields.json")
_FIELDS = json.loads(FIELDS_PATH.read_text())

V2 = {}
for k, v in _FIELDS.items():
    if not k.startswith("v2."):
        continue
    rest = k.split(".", 1)[1]
    if "_STRUCT." not in rest:
        continue
    s, f = rest.split("_STRUCT.", 1)
    if v.get("off") is None or v["off"] < 0:
        continue
    V2.setdefault(s, {})[f] = v

SIZES = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}



# MEM_PATTERN expansion: per-struct {src,dst} patterns, dims 0..N
# element layout: start_addr u32 @base+16i, step_elem u16 @base+4+16i, num_elem u16 @base+10+16i
_MEM_PAT = {
    "NEURON_ISA_TPB_S3D3_MM": {"src": 16, "dst": 48},
    "NEURON_ISA_TPB_S3D3_AC": {"src": 16, "dst": 48},
}
def _expand_mem_pattern(struct_name, values):
    base = _MEM_PAT.get(struct_name)
    if not base:
        return values
    out = {}
    for k, v in values.items():
        if isinstance(k, tuple):
            out[k] = v
            continue
        for side, boff in base.items():
            m = re.match(rf"{side}_mem_pattern_(start_addr|step_elem|num_elem)(?:(\d+))?$", k)
            if m:
                fld, dim = m.group(1), int(m.group(2) or 0)
                offs = {"start_addr": 0, "step_elem": 4, "num_elem": 10}
                out[(boff + 16 * dim + offs[fld], 4 if fld == "start_addr" else 2)] = v
                break
        else:
            out[k] = v
    return out

def encode(struct_name: str, values: dict, opcode: int, word_len: int = 16,
           debug_cmd: int = 0, debug_hint: int = 0) -> bytes:
    buf = bytearray(64)
    buf[0] = opcode
    buf[1] = word_len
    buf[2] = debug_cmd
    buf[3] = debug_hint
    spec = V2[struct_name]
    vals = _expand_mem_pattern(struct_name, values)
    for f, val in vals.items():
        if isinstance(f, tuple):
            off, size = f
            cur = int.from_bytes(buf[off:off + size], "little")
            buf[off:off + size] = (int(val) & ((1 << (size * 8)) - 1)).to_bytes(size, "little")
            continue
        if f.startswith("header_"):
            continue
        spec_f = spec.get(f)
        if spec_f is None:
            raise KeyError(f"{struct_name}.{f}")
        off, size, shr, mask = spec_f["off"], spec_f["size"], spec_f["shr"], spec_f["mask"]
        v = int(val)
        cur = int.from_bytes(buf[off:off + size], "little")
        nv = v
        if shr:
            nv = v << shr
            bits = mask if mask else ((1 << (size * 8)) - 1)
            keep = (1 << (size * 8)) - 1
            keep ^= ((1 << (size * 8 - shr)) - 1) << shr if mask is None else 0
            cur &= ~(mask if mask else (((1 << 32) - 1) ^ (((1 << (size * 8 - shr)) - 1) << shr)) & ((1 << (size * 8)) - 1))
        elif mask is not None:
            width = mask.bit_length()
            cur &= ~(mask << 0) & ((1 << (size * 8)) - 1)
            nv = (v & mask)
        else:
            cur = 0  # plain field overwrite
        cur |= nv & ((1 << (size * 8)) - 1)
        buf[off:off + size] = (cur & ((1 << (size * 8)) - 1)).to_bytes(size, "little")
    return bytes(buf)


def decode(buf: bytes, struct_name: str) -> dict:
    spec = V2[struct_name]
    out = {}
    for f, spec_f in spec.items():
        off, size, shr, mask = spec_f["off"], spec_f["size"], spec_f["shr"], spec_f["mask"]
        v = int.from_bytes(buf[off:off + size], "little")
        if shr and mask is not None:
            v = (v >> shr) & mask
        elif mask is not None:
            v = v & mask
        out[f] = v
    return out


def decode_stream(data: bytes, struct_for) -> list:
    """struct_for(opcode, lenbyte) -> struct name or None; 64B instrs."""
    out = []
    for i in range(0, len(data), 64):
        s = struct_for(data[i], data[i + 1])
        out.append((data[i], data[i + 1], decode(data[i:i + 64], s) if s else None))
    return out
