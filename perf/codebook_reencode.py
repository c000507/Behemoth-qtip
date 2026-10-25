#!/usr/bin/env python3
"""Feasibility: re-encode a 1x16 AQLM codebook (65536 x 8) as a sum of M small
codebooks (K entries each, K=256 -> 2 KiB fp8 per book) WITHOUT calibration
data. Every original code c maps to (a_1(c), ..., a_M(c)); the code tensor is
rewritten offline. Reports usage-weighted relative reconstruction error of the
dequantized weights, i.e. ||W_1x16 - W_Mx8|| / ||W_1x16||.

Usage: reencode.py LAYER PROJ [M...]
"""
import json, struct, sys, time
import numpy as np

root = "/mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output/"
idx = json.load(open(root + "model.safetensors.index.json"))["weight_map"]


def tensor(name):
    fn = root + idx[name]
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        buf = f.read(b - a)
    dt = {"I16": np.int16, "F16": np.float16, "F32": np.float32, "BF16": np.uint16}[meta["dtype"]]
    return np.frombuffer(buf, dtype=dt).reshape(meta["shape"])


def fp8_e4m3_round(x):
    """Round to nearest OCP E4M3 (saturating), vectorised."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    a = np.abs(x)
    a = np.minimum(a, 448.0)
    # exponent of the fp8 value
    e = np.floor(np.log2(np.maximum(a, 2.0**-9)))
    e = np.clip(e, -6, 8)
    step = 2.0 ** (e - 3)  # 3 mantissa bits
    q = np.round(a / step) * step
    return (sign * q).astype(np.float32)


def assign(res, book):
    """Nearest entry (squared L2) for each row of res. res: (N,8), book: (K,8)."""
    # ||r - b||^2 = ||r||^2 - 2 r.b + ||b||^2 ; argmin over b
    d = -2.0 * res @ book.T + (book * book).sum(1)[None, :]
    return np.argmin(d, axis=1)


def weighted_kmeans(x, w, K, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ style init on a weighted sample
    p = w / w.sum()
    centers = x[rng.choice(len(x), size=K, replace=False, p=p)].copy()
    for _ in range(iters):
        a = assign(x, centers)
        for k in range(K):
            m = a == k
            if m.any():
                centers[k] = (x[m] * w[m, None]).sum(0) / w[m].sum()
            else:
                centers[k] = x[rng.integers(len(x))]
    return centers


def fit_additive(book, w, M, K=256, rounds=8, fp8=True, seed=0):
    """Residual-VQ init, then alternating: (1) per-codebook code refinement
    (coordinate descent, exact), (2) weighted least-squares codebook update."""
    N, D = book.shape
    codes = np.zeros((M, N), dtype=np.int64)
    books = np.zeros((M, K, D), dtype=np.float32)
    res = book.copy()
    for m in range(M):
        books[m] = weighted_kmeans(res, w, K, seed=seed + m)
        codes[m] = assign(res, books[m])
        res = res - books[m][codes[m]]

    def recon():
        r = np.zeros_like(book)
        for m in range(M):
            r += books[m][codes[m]]
        return r

    for it in range(rounds):
        # code refinement, one codebook at a time
        for m in range(M):
            partial = recon() - books[m][codes[m]]
            codes[m] = assign(book - partial, books[m])
        # codebook update by weighted coordinate descent (closed form per entry)
        for m in range(M):
            target = book - (recon() - books[m][codes[m]])
            for k in range(K):
                sel = codes[m] == k
                if sel.any():
                    books[m][k] = (target[sel] * w[sel, None]).sum(0) / max(w[sel].sum(), 1e-12)
        if fp8:
            # codebooks must be storable as fp8 with one shared scale (like the original)
            scale = np.abs(books).max() / 448.0
            books = fp8_e4m3_round(books / scale) * scale
    return books, codes, recon()


def main():
    L = int(sys.argv[1])
    proj = sys.argv[2]
    # each spec is M or MxK (e.g. 2x1024)
    specs = []
    for x in (sys.argv[3:] or ["2", "3", "4"]):
        m, _, k = x.partition("x")
        specs.append((int(m), int(k) if k else 256))
    name = f"model.layers.{L}.{proj}"
    codes = tensor(name + ".codes").astype(np.uint16).ravel()
    book = tensor(name + ".codebooks").astype(np.float32).reshape(65536, 8)
    w = np.bincount(codes, minlength=65536).astype(np.float64)
    # The runtime stores the book as fp8 with one scale: measure that error too.
    scale = np.abs(book).max() / 448.0
    book8 = fp8_e4m3_round(book / scale) * scale
    def rel(recon):
        num = (w * ((book - recon) ** 2).sum(1)).sum()
        den = (w * (book ** 2).sum(1)).sum()
        return np.sqrt(num / den)
    print(f"{name}: used={int((w>0).sum())} fp8(1x16 runtime) rel err={rel(book8):.4f}")
    for M, K in specs:
        t = time.time()
        books, acodes, recon = fit_additive(book, w, M, K=K)
        # unweighted error too (what a uniform-usage layer would see)
        unw = np.sqrt(((book - recon) ** 2).sum() / (book ** 2).sum())
        bits = M * int(np.log2(K))
        print(f"  M={M} K={K} table={M*K*8/1024:.0f} KiB ({bits} bits/8 weights = {bits/8:.2f} bpw): weighted rel err={rel(recon):.4f} "
              f"unweighted={unw:.4f}  fit {time.time()-t:.0f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
