#!/usr/bin/env python3
"""Extract NEURON_ISA / TONGA_ISA definitions from readelf --debug-dump=info output.

Usage: dwarf_extract.py NDDBG.dwarf OUT.json OUT.md
Emits: const variables, enums, typedefs, structs/unions (with members incl bitfields).
"""
import json
import re
import sys

DIE_RE = re.compile(r"^\s*<(\d+)><([0-9a-f]+)>: Abbrev Number: \d+ \((DW_TAG_\w+)\)")
ATTR_RE = re.compile(r"^\s*<([0-9a-f]+)>\s+(DW_AT_\w+)\s*:\s*(.*)$")
NAME_INDIR = re.compile(r"indirect string, offset: 0x[0-9a-f]+\):\s*(.*)$")
TYPES = {"DW_TAG_structure_type", "DW_TAG_union_type", "DW_TAG_enumeration_type",
         "DW_TAG_typedef", "DW_TAG_base_type", "DW_TAG_pointer_type",
         "DW_TAG_array_type", "DW_TAG_enumerator", "DW_TAG_member",
         "DW_TAG_enumeration_type"}
KEEP = re.compile(r"NEURON_ISA|TONGA_ISA")

def attr_val(name, raw):
    if name == "DW_AT_name":
        m = NAME_INDIR.search(raw)
        return m.group(1).strip() if m else raw.strip()
    if name in ("DW_AT_byte_size", "DW_AT_bit_size", "DW_AT_data_member_location",
                "DW_AT_data_bit_offset", "DW_AT_const_value", "DW_AT_bit_offset",
                "DW_AT_upper_bound", "DW_AT_count"):
        m = re.match(r"\s*(\d+)", raw)
        if m:
            return int(m.group(1))
    return raw.strip() if len(raw) < 120 else raw.strip()[:120]

def main():
    _, src, out_json, out_md = sys.argv
    defs = {}
    consts = {}
    stack = []   # (level, dict)
    with open(src, errors="replace") as f:
        for line in f:
            m = DIE_RE.match(line)
            if m:
                lvl, off, tag = int(m.group(1)), int(m.group(2), 16), m.group(3)
                while stack and stack[-1][0] >= lvl:
                    done = stack.pop()
                    d = done[1]
                    nm = d["attrs"].get("DW_AT_name", "")
                    if isinstance(nm, str) and KEEP.search(nm):
                        defs[nm] = d
                    elif d["tag"] in ("DW_TAG_member", "DW_TAG_enumerator") and stack:
                        stack[-1][1].setdefault(d["tag"] + "s", []).append(d)
                stack.append((lvl, {"off": off, "tag": tag, "attrs": {}}))
                continue
            m = ATTR_RE.match(line)
            if m and stack:
                aname, raw = m.group(2), m.group(3)
                d = stack[-1][1]
                d["attrs"][aname] = attr_val(aname, raw)
    for nm, d in defs.items():
        if d["tag"] == "DW_TAG_variable" and "DW_AT_const_value" in d["attrs"]:
            consts[nm] = d["attrs"]["DW_AT_const_value"]
    json.dump({"consts": consts, "defs": defs}, open(out_json, "w"), indent=1)
    with open(out_md, "w") as f:
        f.write("# Neuron ISA (TPB) definitions extracted from neuron-dbg DWARF\n\n")
        f.write(f"## Constants ({len(consts)})\n\n| name | value |\n|---|---|\n")
        for k in sorted(consts):
            f.write(f"| {k} | {consts[k]} |\n")
        f.write(f"\n## Type definitions ({len(defs)})\n\n")
        for nm in sorted(defs):
            d = defs[nm]
            f.write(f"### {nm} ({d['tag'].replace('DW_TAG_','').replace('_type','')}"
                    f", size={d['attrs'].get('DW_AT_byte_size','?')})\n\n")
            for mem in d.get("DW_TAG_members", []) + d.get("DW_TAG_enumerators", []):
                a = mem["attrs"]
                bits = ""
                if "DW_AT_bit_size" in a:
                    bits = f" bits[{a.get('DW_AT_data_bit_offset', a.get('DW_AT_bit_offset','?'))}+{a['DW_AT_bit_size']}]"
                loc = a.get("DW_AT_data_member_location", "")
                val = a.get("DW_AT_const_value", "")
                f.write(f"- `{a.get('DW_AT_name','?')}` type@{a.get('DW_AT_type','-')}"
                        f" loc={loc}{bits} val={val}\n")
            f.write("\n")
    print(f"defs={len(defs)} consts={len(consts)} -> {out_json}, {out_md}")

if __name__ == "__main__":
    main()
