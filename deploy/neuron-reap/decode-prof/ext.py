#!/usr/bin/env python3
"""ext.py PROFILE.json OUT.json A B: keep the instructions / dma / active_time / cc_ops between the end of all-reduce A-1 and the end of all-reduce B
(0-based AR indices; A = 0 starts at the first instruction) -> a small json with the same keys (+ 'window')."""
import json, sys
j = json.load(open(sys.argv[1])); A, B = int(sys.argv[3]), int(sys.argv[4])
ars = sorted([c for c in j.get('cc_ops', []) if c.get('operation') == 'AllReduce'], key=lambda c: c['timestamp'])
t0 = min(i['timestamp'] for i in j['instruction'])
lo = t0 if A == 0 else ars[A - 1]['timestamp'] + ars[A - 1]['duration']; hi = ars[B]['timestamp'] + ars[B]['duration']
o = {'window': [lo, hi], 'summary': j['summary'],
     'instruction': [i for i in j['instruction'] if lo <= i['timestamp'] < hi],
     'dma': [d for d in j['dma'] if lo <= d['timestamp'] < hi],
     'active_time': [a for a in j['active_time'] if a['end_ts'] > lo and a['start_ts'] < hi],
     'cc_ops': [c for c in j.get('cc_ops', []) if lo - 1 <= c['timestamp'] < hi], 'opcode_summary': [], 'layer_summary': [l for l in j['layer_summary'] if l['start'] < hi and l['end'] > lo]}
json.dump(o, open(sys.argv[2], 'w'))
print(len(o['instruction']), len(o['dma']))
