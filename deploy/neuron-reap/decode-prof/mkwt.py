#!/usr/bin/env python3
"""mkwt.py SRC DST: a WT=1 copy of a qwen35 prepared dir (qwen35_graphs.to_wt layout): the dense projections, router and head
[in, out], the experts' gate|up [E, H, 2 EIl]; everything else unchanged (other files symlinked). Experiment only."""
import json, os, sys, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
PROJ = {'qkv', 'z', 'b', 'a', 'o', 'q', 'k', 'v', 'sg', 'su', 'sd'}
j = json.load(open(src / 'prepared.json'))
for p in src.iterdir():
    if p.name not in ('prepared.json', 'r0', 'r1') and not (dst / p.name).exists():
        os.symlink(p, dst / p.name)
DT = {'f8e4m3': np.uint8, 'f32': np.float32, 'bf16': np.uint16, 'i64': np.int64}
for f, ents in j['files'].items():
    if not (f.startswith('r0/') or f.startswith('r1/')):
        continue
    (dst / f).parent.mkdir(parents=True, exist_ok=True)
    raw = np.fromfile(src / f, dtype=np.uint8)
    out = np.empty_like(raw)
    out[:] = raw
    for e in ents:
        n = e['name'].split('.')[-1]; sh = e['shape']; o, b = e['offset'], e['bytes']
        a = raw[o:o + b].view(DT[e['dtype']]).reshape(sh)
        if (n in PROJ or n == 'rw' or (f.endswith('head.bin') and n == 'w')) and len(sh) == 2:
            t = np.ascontiguousarray(a.T); e['shape'] = [sh[1], sh[0]]
        elif n == 'egu' and len(sh) == 3:
            t = np.ascontiguousarray(a.transpose(0, 2, 1)); e['shape'] = [sh[0], sh[2], sh[1]]
        else:
            continue
        out[o:o + b] = t.reshape(-1).view(np.uint8)
    out.tofile(dst / f)
    print(f, flush=True)
j['wt'] = True
json.dump(j, open(dst / 'prepared.json', 'w'))
print('MKWT_DONE')
