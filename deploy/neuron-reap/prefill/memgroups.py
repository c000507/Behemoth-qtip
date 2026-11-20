#!/usr/bin/env python3
"""memgroups.py MEMTABLE SERVER.LOG: HBM (model code + DMA-ring spill, MiB, core 0) per NEFF group of a set, from the runtime's memory
table (/tmp/neuron_mem_table_device_0_nc_0.log, written when an allocation fails) and the server's 'Qwen35 NEFF' load lines."""
import re, sys, collections
def mb(s):
    m = re.match(r'([0-9.]+)(GB|MB|KB|B)', s.strip().rstrip('*'))
    return float(m[1]) * {'GB': 1024, 'MB': 1, 'KB': 1 / 1024, 'B': 1 / 1048576}[m[2]] if m else 0
rows = []
for l in open(sys.argv[1]):
    if l.startswith('\\_NEFF'):
        c = l.split('|'); rows.append((int(c[0].split()[1]), mb(c[1]), mb(c[2]), mb(c[5]), mb(c[9])))
rows.sort()
names = [re.match(r'Qwen35 NEFF (\S+) R=(\d+) Q=(\d+) P=(\d+) T=(\d+) S=(\d+)', l) for l in open(sys.argv[2], errors='replace') if l.startswith('Qwen35 NEFF')]
names = [m for m in names if m]
print(len(rows), 'table rows,', len(names), 'NEFF lines')
g = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for r, m in zip(rows, names):
    kind, R, Q = m[1], int(m[2]), int(m[3])
    key = ('head' if kind == 'head' else 'mtp / verify' if kind.startswith('mtp') or 'segv' in kind or kind.startswith('seg') and Q > 1 and Q < 8 else f'prefill Q={Q}' if Q > 1 else f'decode R={R}')
    if kind == 'head': key = 'heads'
    a = g[key]; a[0] += 1; a[1] += r[1]; a[2] += r[2]; a[3] = max(a[3], r[3]); a[4] += r[4]
    if Q >= 1024: print(f'  {kind} R={R} Q={Q} P={m[4]}: total {r[1]:.0f} = code {r[2]:.0f} + spill {r[4]:.0f} MiB, scratch {r[3]:.0f}')
for k, a in sorted(g.items()): print(f'{k:16s} {a[0]:3d} NEFFs  total {a[1]:6.0f} MiB (code {a[2]:.0f}, ring spill {a[4]:.0f}), largest scratch {a[3]:.0f}')
