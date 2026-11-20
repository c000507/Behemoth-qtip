#!/usr/bin/env python3
"""cmpref.py REFDIR TAG...: prefill-end logits of each run (run/TAG/logits.bin, 8 records per request) vs the f32 prepared-weights
reference (REFDIR/ref<N>.npy, N = 1-based request index): rel L2, KL(ref || device), top-1, the reference's top-1 margin."""
import sys, glob, os, numpy as np
V = 248320
def lsm(x): x = x.astype(np.float64); m = x.max(); return x - m - np.log(np.exp(x - m).sum())
refs = sorted(glob.glob(sys.argv[1] + '/ref*.npy'), key=lambda f: int(os.path.basename(f)[3:-4]))
for tag in sys.argv[2:]:
    a = np.memmap(f'/mnt/scratch/run/{tag}/logits.bin', np.float32).reshape(-1, V); out = []
    for f in refs:
        n = int(os.path.basename(f)[3:-4]); r = np.load(f)[-1]; d = a[(n - 1) * 8]
        s = np.sort(r)[-2:]; lr, ld = lsm(r), lsm(d)
        out.append(f"req{n}: rel {np.linalg.norm(d - r) / np.linalg.norm(r):.4f} KL {float((np.exp(lr) * (lr - ld)).sum()):.4f} top1 {'same' if r.argmax() == d.argmax() else 'DIFF'} (ref margin {s[1] - s[0]:.2f})")
    print(f'{tag}: ' + ' | '.join(out))
