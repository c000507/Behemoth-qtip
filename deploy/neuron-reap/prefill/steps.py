#!/usr/bin/env python3
"""steps.py server.log: prefill / decode summary of the engine step log (per chunk: wall, exec, per-class nrt_execute of rank 0)."""
import re, sys, json, collections
pre = collections.defaultdict(list); dec = collections.defaultdict(list)
for l in open(sys.argv[1], errors='replace'):
    if not l.startswith('step rows='): continue
    m = re.match(r'step rows=(\d+) drafts=(\d+) accepted=(\d+) tokens=(\d+) wall_ms=([0-9.]+) (\{.*\})', l)
    if not m: continue
    rows, drafts, acc, toks, wall = int(m[1]), int(m[2]), int(m[3]), int(m[4]), float(m[5]); p = json.loads(m[6])
    bt = p.get('tokens', toks)
    (pre[(rows, bt)] if bt > rows * (1 + drafts) + 8 else dec[(rows, drafts)]).append((wall, p))
def med(v): v = sorted(v); return v[len(v) // 2]
for k in sorted(pre):
    v = pre[k]; r = lambda f: med([f(x[1]) for x in v])
    print(f"prefill rows={k[0]} tokens={k[1]} n={len(v)} wall med {med([x[0] for x in v]):.1f} ms ({1000 * k[1] / med([x[0] for x in v]):.0f} tok/s) launches {r(lambda p: p['launches'])} forward {r(lambda p: p['forward_ms']):.1f} exec {r(lambda p: p['exec_ms']):.1f} plan {r(lambda p: p['plan_ms']):.1f} | rank0 set {r(lambda p: p['r0']['set']):.1f} gdn {r(lambda p: p['r0']['gdn']):.1f} gqa {r(lambda p: p['r0']['gqa']):.1f} moe {r(lambda p: p['r0']['moe']):.1f} put {r(lambda p: p['r0']['put']):.1f} head {r(lambda p: p['r0']['head']):.1f} read {r(lambda p: p['r0']['read']):.1f}")
for k in sorted(dec):
    v = dec[k]
    if len(v) < 3: continue
    print(f"decode rows={k[0]} drafts={k[1]} n={len(v)} wall med {med([x[0] for x in v]):.1f} ms mean {sum(x[0] for x in v) / len(v):.1f}")
