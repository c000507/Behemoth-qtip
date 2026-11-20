#!/usr/bin/env python3
"""loadsw.py server.log [W...]: expert loads of wide chunks from a Q=256 MOE_GSTAT=-1 run (BEHEMOTH_NEURON_QWEN35_MOE_LOAD_LOG=1; prompts run
one at a time). The per-expert load vectors of consecutive 256-token chunks are summed into windows of W tokens (default 256 1024 2048 4096;
only complete windows) -> the sorted load profile (max / p99 / median by rank, as a fraction of W) and the fallback rate of tier plans."""
import re, sys, collections
import numpy as np
rows = collections.defaultdict(list); seq = collections.defaultdict(int)
for l in open(sys.argv[1], errors='replace'):
    m = re.match(r'moe_load layer (\d+) pos (-?\d+) load \S+( FALLBACK)? top (.*)', l)
    if not m: continue
    layer, pos = int(m[1]), int(m[2]); v = np.array([int(x) for x in m[4].split()], dtype=np.int32)
    if pos == 0: seq[layer] += 1
    rows[(layer, seq[layer])].append((pos, v))
Ws = [int(x) for x in sys.argv[2:]] or [256, 1024, 2048, 4096]
for W in Ws:
    wins = []
    for key, ch in rows.items():
        ch = sorted(ch, key=lambda x: x[0]); acc = {}
        for pos, v in ch:
            if v.sum() != 8 * 256: continue        # complete 256-token chunks only
            acc.setdefault(pos // W, []).append(v)
        for w, vs in acc.items():
            if len(vs) == W // 256: wins.append(np.sort(np.sum(vs, axis=0))[::-1])
    if not wins: print(f'W={W}: no complete windows'); continue
    S = np.array(wins); E = S.shape[1]
    print(f'W={W}: {len(S)} windows (layers x chunks); load of the r-th most loaded expert as % of W: rank: max / p99 / median')
    print('  ' + '  '.join(f'{r}: {100 * S[:, r].max() / W:.1f}/{100 * np.percentile(S[:, r], 99) / W:.1f}/{100 * np.median(S[:, r]) / W:.1f}' for r in (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 204) if r < E))
    np.save(f'{sys.argv[1]}.loads{W}.npy', S)
