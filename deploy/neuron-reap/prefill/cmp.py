#!/usr/bin/env python3
"""cmp.py BASE.bin NEW.bin [steps=8]: per request, the prefill-end logits (record 0) and the following greedy steps of NEW vs BASE:
rel L2, top-1 agreement, BASE's top-1 margin, KL(base || new). Decode records are compared while the greedy tokens agree."""
import sys, numpy as np
V = 248320; n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
a = np.memmap(sys.argv[1], np.float32).reshape(-1, V); b = np.memmap(sys.argv[2], np.float32).reshape(-1, V)
print(f'records {a.shape[0]} / {b.shape[0]}; non-finite base {int((~np.isfinite(a)).sum())} new {int((~np.isfinite(b)).sum())}')
def lsm(x): x = x.astype(np.float64); m = x.max(); return x - m - np.log(np.exp(x - m).sum())
tot = agree = 0; rels = []
for q in range(min(a.shape[0], b.shape[0]) // n):
    line = []; same = True
    for i in range(n):
        x, y = a[q * n + i], b[q * n + i]
        if not same: break
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(x)); ta, tb = int(x.argmax()), int(y.argmax())
        s = np.sort(x)[-2:]; la, lb = lsm(x), lsm(y); kl = float((np.exp(la) * (la - lb)).sum())
        tot += 1; agree += ta == tb; rels.append(rel)
        line.append(f"{rel:.3f}{'' if ta == tb else '!(m=%.2f)' % (s[1] - s[0])}")
        if i == 0: head = f'prefill-end rel L2 {rel:.4f} KL {kl:.4f} top1 {"same" if ta == tb else "DIFF"} (base margin {s[1] - s[0]:.2f})'
        same = ta == tb
    print(f'req {q}: {head}; steps rel L2: {" ".join(line)}')
print(f'top-1 agreement {agree}/{tot} compared steps; rel L2 mean {np.mean(rels):.4f} max {np.max(rels):.4f}')
