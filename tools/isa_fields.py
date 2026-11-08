#!/usr/bin/env python3
"""Fast ISA field extraction from neuron-dbg accessor functions (v2).

One objdump pass over .text; index lines by address; parse each accessor:
  mov %rdi,-0x18(%rbp); mov -0x18(%rbp),%rax; movzbl 0x1(%rax),%eax [shr/and] ; ret
"""
import re
import subprocess
import json
import sys

BIN = sys.argv[1]
OUT = sys.argv[2]

print("disassembling...", file=sys.stderr)
dis = subprocess.run(["objdump", "-d", BIN], capture_output=True, text=True).stdout
print("indexed", len(dis), "bytes of text", file=sys.stderr)

lines_by_addr = {}
addr_of = []
cur = None
for line in dis.splitlines():
    m = re.match(r"^\s+([0-9a-f]+):\t(.*)$", line)
    if m:
        a = int(m.group(1), 16)
        parts = m.group(2).split("\t")
        lines_by_addr[a] = (parts[-1] if len(parts) > 1 else parts[0]).strip()
        addr_of.append(a)
addr_of.sort()

nm = subprocess.run(["nm", BIN], capture_output=True, text=True).stdout
syms = []
for line in nm.splitlines():
    m = re.match(r"([0-9a-f]+) [Tt] get_(v\d|common|tonga)_(NEURON_ISA_\S+)$", line)
    if m:
        syms.append((int(m.group(1), 16), m.group(2), m.group(3)))
syms.sort()
import bisect

fields = {}
for addr, ver, rest in syms:
    m = re.match(r"(.+?_STRUCT)_(.+)$", rest)
    if not m:
        continue
    sname, fname = m.group(1), m.group(2)
    i = bisect.bisect_left(addr_of, addr)
    off = size = None
    shr = mask = None
    seen_ptr = False
    for a in addr_of[i:i + 24]:
        o = lines_by_addr[a]
        if o == "ret" or o.startswith("ret"):
            break
        if not seen_ptr and (re.match(r"mov\s+%rdi,", o) or re.match(r"mov\s+-0x18\(%rbp\),%r", o)):
            seen_ptr = True
            continue
        if not seen_ptr:
            continue
        mm = re.match(r"movzbl\s+(-?0x[0-9a-f]+)?\(%r[a-z0-9]+\)", o)
        if mm:
            size = 1
            off = int(mm.group(1), 16) if mm.group(1) else 0
            continue
        mm = re.match(r"movzwl\s+(-?0x[0-9a-f]+)?\(%r[a-z0-9]+\)", o)
        if mm:
            size = 2
            off = int(mm.group(1), 16) if mm.group(1) else 0
            continue
        mm = re.match(r"movslq\s+(-?0x[0-9a-f]+)?\(%r[a-z0-9]+\)", o)
        if mm:
            size = 4
            off = int(mm.group(1), 16) if mm.group(1) else 0
            continue
        mm = re.match(r"mov\s+(-?0x[0-9a-f]+)?\(%r[a-z0-9]+\),%e[a-z]x$", o)
        if mm:
            size = 4
            off = int(mm.group(1), 16) if mm.group(1) else 0
            continue
        mm = re.match(r"mov\s+(-?0x[0-9a-f]+)?\(%r[a-z0-9]+\),%r[a-z0-9]+$", o)
        if mm:
            size = 8
            off = int(mm.group(1), 16) if mm.group(1) else 0
            continue
        if off is not None:
            ms = re.match(r"shr\s+\$0x([0-9a-f]+)", o)
            if ms:
                shr = int(ms.group(1), 16)
                continue
            mm = re.match(r"and\s+\$0x([0-9a-f]+)", o)
            if mm:
                mask = int(mm.group(1), 16)
                continue
            if o.startswith("mov %eax,%") or o.startswith("mov %rax,%"):
                break
    fields[f"{ver}.{sname}.{fname}"] = {
        "off": off, "size": size, "shr": shr, "mask": mask, "addr": hex(addr)}

json.dump(fields, open(OUT, "w"), indent=1, sort_keys=True)
bad = [k for k, v in fields.items() if v["off"] is None]
print("fields:", len(fields), "unresolved:", len(bad), file=sys.stderr)
if bad[:10]:
    print("examples:", bad[:10], file=sys.stderr)
