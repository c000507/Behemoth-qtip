#!/usr/bin/env python3
"""ana.py PROFILE.json [--bins US] [--seg KINDS]: per-engine / DMA / collective breakdown of a neuron-profile json export.
Intervals = between consecutive all-reduce completions (each sub-block ends with its all-reduce), labelled with --seg
(comma list of sub-block kinds in order, e.g. gdn,moe,gdn,moe,...). Times in us."""
import json, sys, collections, argparse
ap = argparse.ArgumentParser(); ap.add_argument('f'); ap.add_argument('--bins', type=float, default=0); ap.add_argument('--seg', default='')
ap.add_argument('--top', type=int, default=25)
A = ap.parse_args()
j = json.load(open(A.f))
ENG = ['tensor', 'vector', 'scalar', 'gpsimd', 'sync', 'cc_instruction']
iv = collections.defaultdict(list)
for a in j['active_time']:
    iv[a['engine']].append((a['start_ts'], a['end_ts']))
dma = [d for d in j['dma']]
dmaiv = [(d['timestamp'], d['timestamp'] + d['duration'], d['transfer_size'], d['queue_type'], d['subgroup'].split('-')[0], d['is_cc_dma']) for d in dma]
ins = j['instruction']
t0 = min(i['timestamp'] for i in ins); t1 = max(i['timestamp'] + i['duration'] for i in ins)
cc = sorted(j.get('cc_ops', []), key=lambda c: c['timestamp'])
def union(ivs, lo, hi):
    s = sorted((max(a, lo), min(b, hi)) for a, b in ivs if b > lo and a < hi)
    tot, cur = 0, None
    for a, b in s:
        if cur is None or a > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [a, b]
        else: cur[1] = max(cur[1], b)
    if cur: tot += cur[1] - cur[0]
    return tot
def any_busy(lo, hi, with_dma=True):
    allv = [x for e in ENG if e != 'cc_instruction' for x in iv[e]] + ([(a, b) for a, b, *_ in dmaiv] if with_dma else [])
    return union(allv, lo, hi)
S = j['summary'][0]
print(f"file {A.f}\n total_time {S['total_time']*1e6:.1f} us  instr span {(t1-t0)/1e3:.1f} us  hbm_read {S['hbm_read_bytes']/1e6:.2f} MB  hbm_write {S['hbm_write_bytes']/1e6:.2f} MB"
      f"  -> {S['hbm_read_bytes']/max(1,(t1-t0))/1:.1f} GB/s over the span; dma_active {S['dma_active_time']*1e6:.1f} us")
print(' engine active (us, union):', {e: round(union(iv[e], t0, t1) / 1e3, 1) for e in ENG}, ' instr counts:', {k: S.get(k + '_instruction_count') for k in ('tensor_engine', 'vector_engine', 'scalar_engine', 'gpsimd_engine', 'sync_engine')})
print(f" any engine or DMA busy {any_busy(t0, t1)/1e3:.1f} us; compute engines only {any_busy(t0, t1, False)/1e3:.1f}; DMA only {union([(a,b) for a,b,*_ in dmaiv], t0, t1)/1e3:.1f}")
q = collections.defaultdict(lambda: [0, 0, []])
for a, b, n, qt, sg, isc in dmaiv:
    k = (qt, sg); q[k][0] += 1; q[k][1] += n; q[k][2].append((a, b))
print(' DMA queues (packets, MB, active us, GB/s while active):')
for k, (c, n, ivs) in sorted(q.items(), key=lambda kv: -kv[1][1])[:12]:
    act = union(ivs, t0, t1)
    print(f'   {k[0]:18s} {k[1]:22s} {c:7d} {n/1e6:9.3f} {act/1e3:8.1f} {n/max(act,1):7.1f}')
ccs = [c for c in cc]
print(f" cc ops {len(ccs)}: total {sum(c['duration'] for c in ccs)/1e3:.1f} us; " + ', '.join(f"{c['operation'] if c['dtype']!='BARRIER' else 'BARRIER'}@{(c['timestamp']-t0)/1e3:.1f}+{c['duration']/1e3:.1f}" for c in ccs[:6]) + (' ...' if len(ccs) > 6 else ''))
ars = [c for c in ccs if c.get('operation') == 'AllReduce']
if ars:
    d = sorted(c['duration'] for c in ars); print(f"   all-reduce durations: n {len(d)} min {d[0]/1e3:.1f} med {d[len(d)//2]/1e3:.1f} max {d[-1]/1e3:.1f} sum {sum(d)/1e3:.1f} us; trigger delay med {sorted(c.get('cc_trigger_start_delay',0) for c in ars)[len(ars)//2]/1e3:.2f}")
# opcode summary
print(' top opcodes by active time (engine opcode count active_us instr_us):')
for o in sorted(j['opcode_summary'], key=lambda o: -o['active_time_ns'])[:A.top]:
    print(f"   {o['engine']:8s} {o['opcode']:28s} {o['instruction_count']:7d} {o['active_time_ns']/1e3:8.1f} {o['instruction_time_ns']/1e3:8.1f}")
# hlo op groups (layer_summary): by name stem
grp = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
for l in j['layer_summary']:
    stem = l['name'].split('/')[-1].lstrip('_').split('.')[0]
    g = grp[stem]; g[0] += 1; g[1] += l['duration']; g[2] += l.get('tensor_engine_active_time', 0); g[3] += l.get('vector_engine_active_time', 0); g[4] += l.get('scalar_engine_active_time', 0); g[5] += l.get('gpsimd_engine_active_time', 0)
print(' HLO op stems by summed duration (n, dur_us, PE, DVE, ACT, POOL us):')
for k, g in sorted(grp.items(), key=lambda kv: -kv[1][1])[:A.top]:
    print(f'   {k:28s} {g[0]:5d} {g[1]/1e3:8.1f} {g[2]/1e3:7.1f} {g[3]/1e3:7.1f} {g[4]/1e3:7.1f} {g[5]/1e3:7.1f}')
# intervals between all-reduce ends
if ars:
    kinds = A.seg.split(',') if A.seg else []
    edges = [t0] + [c['timestamp'] + c['duration'] for c in ars]
    if edges[-1] < t1: edges.append(t1)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        ar = ars[i] if i < len(ars) else None
        nb = sum(n for a, b, n, qt, sg, isc in dmaiv if lo <= a < hi and isc == 'no')
        dact = union([(a, b) for a, b, *_ in dmaiv], lo, hi)
        busy = any_busy(lo, hi)
        rows.append((kinds[i] if i < len(kinds) else '?', (hi - lo) / 1e3, (ar['duration'] / 1e3 if ar else 0), nb / 1e6, dact / 1e3,
                     union(iv['tensor'], lo, hi) / 1e3, union(iv['vector'], lo, hi) / 1e3, union(iv['scalar'], lo, hi) / 1e3, union(iv['gpsimd'], lo, hi) / 1e3, (hi - lo - busy) / 1e3))
    print(' per interval (ends at each all-reduce end): kind wall_us AR_us DMA_MB DMA_us PE DVE ACT POOL idle_us')
    for r in rows:
        print('   %-4s %7.1f %6.1f %7.2f %7.1f %6.1f %6.1f %6.1f %6.1f %6.1f' % r)
    agg = collections.defaultdict(lambda: [0] * 10)
    for r in rows:
        a = agg[r[0]]; a[0] += 1
        for k in range(1, 10): a[k] += r[k]
    print(' per kind (mean): n wall AR DMA_MB DMA_us PE DVE ACT POOL idle')
    for k, a in agg.items():
        print('   %-4s %3d ' % (k, a[0]) + ' '.join('%7.2f' % (a[i] / a[0]) for i in range(1, 10)))
if A.bins:
    B = A.bins * 1e3; n = int((t1 - t0) / B) + 1
    print(f' timeline bins of {A.bins} us: t PE DVE ACT POOL DMA(MB) cc')
    for b in range(n):
        lo, hi = t0 + b * B, t0 + (b + 1) * B
        nb = sum(x[2] for x in dmaiv if lo <= x[0] < hi)
        ccb = union([(c['timestamp'], c['timestamp'] + c['duration']) for c in ccs], lo, hi)
        print('   %7.1f %5.0f%% %5.0f%% %5.0f%% %5.0f%% %7.3f %5.0f%%' % (b * A.bins, *(100 * union(iv[e], lo, hi) / B for e in ('tensor', 'vector', 'scalar', 'gpsimd')), nb / 1e6, 100 * ccb / B))
