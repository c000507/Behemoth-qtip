#!/usr/bin/env python3
"""Pass 2: full DIE index over readelf dump; resolve NEURON_ISA typedef/struct layouts.

Usage: dwarf_resolve.py NDDBG.dwarf OUT.json OUT.md
"""
import json
import re
import sys

DIE_RE = re.compile(r"^\s*<(\d+)><([0-9a-f]+)>: Abbrev Number: (\d+) \((DW_TAG_\w+)\)")
ATTR_RE = re.compile(r"^\s*<([0-9a-f]+)>\s+(DW_AT_\w+)\s*:\s*(.*)$")
NAME_INDIR = re.compile(r"indirect string, offset: 0x[0-9a-f]+\):\s*(.*)$")
KEEP = re.compile(r"NEURON_ISA|TONGA_ISA")
TYPE_TAGS = {"DW_TAG_structure_type", "DW_TAG_union_type", "DW_TAG_enumeration_type",
             "DW_TAG_typedef", "DW_TAG_base_type", "DW_TAG_pointer_type", "DW_TAG_array_type",
             "DW_TAG_subroutine_type", "DW_TAG_volatile_type", "DW_TAG_const_type"}

def attr_val(name, raw):
    if name == "DW_AT_name":
        m = NAME_INDIR.search(raw)
        return m.group(1).strip() if m else raw.strip()
    if name in ("DW_AT_byte_size", "DW_AT_bit_size", "DW_AT_data_member_location",
                "DW_AT_data_bit_offset", "DW_AT_bit_offset", "DW_AT_upper_bound",
                "DW_AT_count", "DW_AT_const_value"):
        m = re.match(r"\s*(\d+)", raw)
        if m:
            return int(m.group(1))
    m = re.match(r"<0x([0-9a-f]+)>", raw)
    if m:
        return int(m.group(1), 16)
    return raw.strip()[:80]

def main():
    _, src, out_json, out_md = sys.argv
    dies = {}          # off -> {tag, attrs, members:[off]}
    order = []
    stack = []
    with open(src, errors="replace") as f:
        for line in f:
            m = DIE_RE.match(line)
            if m:
                lvl, off, abbrev, tag = int(m.group(1)), int(m.group(2), 16), int(m.group(3)), m.group(4)
                while stack and stack[-1][0] >= lvl:
                    plvl, poff = stack.pop()
                    pass
                if stack:
                    dies[stack[-1][1]].setdefault("children", []).append(off)
                dies[off] = {"tag": tag, "attrs": {}}
                order.append(off)
                stack.append((lvl, off))
                continue
            m = ATTR_RE.match(line)
            if m and stack:
                d = dies[stack[-1][1]]
                d["attrs"][m.group(2)] = attr_val(m.group(2), m.group(3))

    # resolve named types + emit all NEURON-related typedef expansions
    def typechain(off, depth=0):
        if off is None or depth > 12:
            return "?"
        d = dies.get(off)
        if d is None:
            return f"@{off:x}?"
        tag = d["tag"]
        a = d["attrs"]
        if tag == "DW_TAG_base_type":
            return a.get("DW_AT_name", "?")
        if tag == "DW_TAG_pointer_type":
            return typechain(a.get("DW_AT_type"), depth + 1) + "*"
        if tag == "DW_TAG_typedef":
            return a.get("DW_AT_name", typechain(a.get("DW_AT_type"), depth + 1))
        if tag == "DW_TAG_array_type":
            n = a.get("DW_AT_upper_bound", a.get("DW_AT_count"))
            n = (n + 1) if isinstance(n, int) else "[]"
            return typechain(a.get("DW_AT_type"), depth + 1) + f"[{n}]"
        if tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
            nm = a.get("DW_AT_name")
            return nm if nm else f"anon_{tag.split('_')[1]}@{off:x}<{d.get('byte_size','?')}>"
        return a.get("DW_AT_name", tag)

    def emit_struct(off, out, indent=0, depth=0, seen=None):
        if seen is None:
            seen = set()
        if off in seen or depth > 8:
            return
        seen.add(off)
        d = dies.get(off)
        if d is None:
            return
        tag = d["tag"]
        a = d["attrs"]
        kw = "struct" if tag == "DW_TAG_structure_type" else ("union" if tag == "DW_TAG_union_type" else "enum")
        pad = "  " * indent
        nm = a.get("DW_AT_name", f"anon@{off:x}")
        out.append(f"{pad}{kw} {nm} {{ // size={a.get('DW_AT_byte_size','?')}")
        for ch in d.get("children", []):
            c = dies.get(ch)
            if c is None:
                continue
            ca = c["attrs"]
            cn = ca.get("DW_AT_name", "")
            if c["tag"] in ("DW_TAG_member",):
                t = typechain(ca.get("DW_AT_type"))
                loc = ca.get("DW_AT_data_member_location", "")
                bits = ca.get("DW_AT_bit_size")
                dbits = ca.get("DW_AT_data_bit_offset", ca.get("DW_AT_bit_offset"))
                bit = f" : {bits}" if bits else ""
                ann = f" @bit{dbits}" if dbits is not None else ""
                arr = ca.get("DW_AT_upper_bound")
                out.append(f"{pad}  {t} {cn}{bit}; // loc={loc}{ann}")
                # expand nested named structs
                toff = ca.get("DW_AT_type")
                if toff and dies.get(toff, {}).get("tag") in ("DW_TAG_structure_type", "DW_TAG_union_type") and depth < 4:
                    emit_struct(toff, out, indent + 1, depth + 1, seen)
            elif c["tag"] == "DW_TAG_enumerator":
                out.append(f"{pad}  {cn} = {ca.get('DW_AT_const_value','?')},")
            elif c["tag"] in TYPE_TAGS and not cn:
                emit_struct(ch, out, indent + 1, depth + 1, seen)
        out.append(f"{pad}}};")

    named = {}
    for off in order:
        d = dies[off]
        nm = d["attrs"].get("DW_AT_name")
        if isinstance(nm, str) and KEEP.search(nm):
            named.setdefault(nm, off)

    out = ["# Neuron ISA (TPB) type layouts (from neuron-dbg/libndbg DWARF)", ""]
    for nm in sorted(named):
        off = named[nm]
        d = dies[off]
        out.append(f"## {nm} @{off:x}")
        buf = []
        if d["tag"] == "DW_TAG_typedef":
            tgt = d["attrs"].get("DW_AT_type")
            out.append(f"typedef {typechain(tgt)} {nm};")
            if tgt and dies.get(tgt, {}).get("tag") in ("DW_TAG_structure_type", "DW_TAG_union_type", "DW_TAG_enumeration_type"):
                emit_struct(tgt, buf)
                out += buf
        elif d["tag"] == "DW_TAG_variable":
            out.append(f"const auto {nm} = {d['attrs'].get('DW_AT_const_value','?')};")
        else:
            emit_struct(off, buf)
            out += buf
        out.append("")
    open(out_md, "w").write("\n".join(out))
    json.dump({"n_dies": len(dies), "named": {k: v for k, v in named.items()}}, open(out_json, "w"), indent=1)
    print(f"dies={len(dies)} named={len(named)} -> {out_md}")

if __name__ == "__main__":
    main()
