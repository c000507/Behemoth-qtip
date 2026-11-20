#!/usr/bin/env python3
"""cls.py WINDOW.json [names]: per sub-block (between all-reduce ends) instruction classes: engine/opcode/dtype/shape -> count, busy us (sum of issue gaps capped by duration)."""
import json, sys, collections, re
j = json.load(open(sys.argv[1])); names = sys.argv[2].split(',') if len(sys.argv) > 2 else []
ins = sorted(j['instruction'], key=lambda i: i['timestamp'])
ars = sorted([c for c in j.get('cc_ops', []) if c.get('operation') == 'AllReduce'], key=lambda c: c['timestamp'])
lo = j['window'][0] if 'window' in j else min(i['timestamp'] for i in ins); edges = [lo] + [c['timestamp'] + c['duration'] for c in ars if c['timestamp'] + c['duration'] > lo]
def cls(i):
    op = i.get('opcode', ''); o = i.get('operands', '')
    if op in ('MATMUL', 'LDWEIGHTS'):
        m = re.search(r'src=(\w+)@\S*\[([^\]]*)\]\[([^\]]*)\]', o); dt = m.group(1) if m else '?'
        sz = re.search(r'(\d+)\*(\d+)', o); return f"{op} {dt} {sz.group(0) if sz else ''} mov{m.group(3) if m else ''}" if op == 'MATMUL' else f"{op} {dt} {sz.group(0) if sz else ''}"
    if op == 'ACTIVATE':
        fn = i.get('activation_fn', '').split('(')[0]
        m = re.search(r'src=(\w+)@\S*\[[^\]]*\]\[([^\]]*)\] dst=(\w+)@\S*\[[^\]]*\]\[([^\]]*)\]', o)
        ch = re.search(r'channels=(\d+)', o)
        if m:
            n = 1
            for x in m.group(2).split(','): n *= int(x)
            return f"ACT {fn} {m.group(1)}->{m.group(3)} free{'<=16' if n <= 16 else '<=512' if n <= 512 else '>512'} ch{ch.group(1) if ch else ''} psum{'Y' if '0x200' in o.split('src=')[1][:12] else 'N'}"
        return f'ACT {fn}'
    return f"{i.get('subgroup','')} {op}"
for k in range(len(edges) - 1):
    a, b = edges[k], edges[k + 1]
    sub = [i for i in ins if a <= i['timestamp'] < b]
    by = collections.defaultdict(lambda: [0, 0.0])
    eng_last = {}
    for i in sub:
        c = cls(i); e = i.get('subgroup')
        by[(e, c)][0] += 1; by[(e, c)][1] += i['duration']
    tot = collections.defaultdict(float)
    for (e, c), v in by.items(): tot[e] += v[1]
    print(f"== {names[k] if k < len(names) else k}: {(b - a)/1e3:.1f} us, {len(sub)} instr; summed durations by engine (us): " + ', '.join(f'{e} {v/1e3:.1f}' for e, v in sorted(tot.items(), key=lambda kv: -kv[1])))
    for (e, c), v in sorted(by.items(), key=lambda kv: -kv[1][1])[:16]:
        print(f'   {e:8s} {c:70s} {v[0]:6d} {v[1]/1e3:8.1f}  avg {v[1]/v[0]:.0f} ns')
