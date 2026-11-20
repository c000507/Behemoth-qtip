#!/usr/bin/env python3
"""loads.py server.log: distribution of the grouped MoE's cold-tier load (BEHEMOTH_NEURON_QWEN35_MOE_LOAD_LOG=1 lines)."""
import re, sys, collections
by = collections.defaultdict(list); fb = 0; n = 0
for l in open(sys.argv[1], errors='replace'):
    m = re.match(r'moe_load layer (\d+) pos (-?\d+) load ([0-9.e+]+)( FALLBACK)?', l)
    if not m: continue
    by[int(m[1])].append(float(m[3])); fb += bool(m[4]); n += 1
al = sorted(v for x in by.values() for v in x)
if not al: print('no moe_load lines'); sys.exit()
q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
print(f'{n} grouped MoE launches, {fb} fallbacks ({100.0 * fb / n:.2f} %); load p50 {q(al, .5):.0f} p90 {q(al, .9):.0f} p99 {q(al, .99):.0f} p99.9 {q(al, .999):.0f} max {al[-1]:.0f}')
for c in (12, 16, 20, 24, 28, 32, 40, 48, 64): print(f'  load > {c}: {100.0 * sum(v > c for v in al) / n:.2f} %')
print('per layer max:', ' '.join(f'{l}:{max(v):.0f}' for l, v in sorted(by.items())))
print('per layer p90:', ' '.join(f'{l}:{q(sorted(v), .9):.0f}' for l, v in sorted(by.items())))

# full sorted loads (MOE_GSTAT graphs): fallback rate of every (MOE_HOT S, MOE_CAP C) = P(load of the (S+1)-th most loaded expert > C)
tops = []
for l in open(sys.argv[1], errors='replace'):
    if ' top ' in l and l.startswith('moe_load'):
        tops.append([int(x) for x in l.split(' top ')[1].split()])
if tops:
    print(f'{len(tops)} launches with sorted loads; fallback % by (hot S, cap C):')
    caps = (16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64)
    print('  S\\C ' + ' '.join(f'{c:6d}' for c in caps))
    for S in (0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48):
        if S < len(tops[0]):
            print(f'  {S:3d} ' + ' '.join(f'{100.0 * sum(t[S] > c for t in tops) / len(tops):6.2f}' for c in caps))
    print('  max load by rank: ' + ' '.join(f'{k}:{max(t[k] for t in tops)}' for k in (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 62) if k < len(tops[0])))
