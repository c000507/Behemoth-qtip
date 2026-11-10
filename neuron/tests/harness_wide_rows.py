"""Numerics check of the wide-row NKI decode buckets (n16_P / n32_P: R rows of R DISTINCT requests through nki_decode.decode_attention_fp8_2p)
against (a) the b18 torch-op-loop bucket of the same (R, P) and (b) the b18 dec8_P NKI bucket run on 8-row slices of the same rows
(same kernel code, R = 8: expected bit-identical per row). All through neff_io (NEFF_IO_STATES=2: cache + cscale aliased state), real
layer-0 weights, real embedding rows as residual input, per-request context K/V produced on the CPU with the real k/v projections + RoPE and
quantised per (token, K|V, head) to E4M3 like the graph. Every row = its own request: own random pages, own context length, own position.

usage: python harness_wide.py WORK R P SEED NEW_DIR LOOP_DIR DEC8_DIR [ROWS_REAL]
Prints one JSON line per row and a SUMMARY line (rel L2 of the layer delta out - x: new vs loop, new vs dec8-slices, loop vs dec8-slices)."""
import json
import os
import struct
import subprocess
import sys
import time

import ml_dtypes
import numpy as np

WORK, R, P, SEED, NEW, LOOP, DEC8 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
ROWS = int(sys.argv[8]) if len(sys.argv) > 8 else R         # real rows; the rest are engine pad rows (x = 0, pos 0, scratch wslot / table)
B18 = '/home/ubuntu/work/b18'
NEFF_IO = '/home/ubuntu/work/vk/bin/neff_io'
H, D, KVL, TP, PAGE = 5120, 128, 4, 2, 16
PAGES, SCRATCH = 5561, 5560
THETA, EPS = 1e9, 1e-5
E4 = ml_dtypes.float8_e4m3fn
rng = np.random.default_rng(SEED)
os.makedirs(WORK, exist_ok=True)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


weights = {}
with open(f'{B18}/prepared/model-layer-00000.bin', 'rb') as f:
    assert f.read(8) == b'BSHD0001'
    n = struct.unpack('<I', f.read(4))[0]
    entries = [struct.unpack('<IIQ', f.read(16)) for _ in range(n)]
    for rank, slot, nbytes in entries:
        weights[(rank, slot)] = f.read(nbytes)
for (rank, slot), b in weights.items():
    idx = 5 if slot == 100 else 6 if slot == 101 else 7 + slot
    with open(f'{WORK}/in{idx}.r{rank}', 'wb') as f:
        f.write(b)


def dequant(rank, slot_w, slot_s, out):
    w = np.frombuffer(weights[(rank, slot_w)], dtype=np.uint8).view(E4).astype(np.float32).reshape(H, out)
    s = np.frombuffer(weights[(rank, slot_s)], dtype=np.float32)
    return w * s[None, :]


with open(f'{B18}/model/model-embed.safetensors', 'rb') as f:
    hn = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hn))
off = hdr['model.embed_tokens.weight']['data_offsets'][0]
emb = np.memmap(f'{B18}/model/model-embed.safetensors', dtype=np.uint16, mode='r', offset=8 + hn + off, shape=(131072, H))


def embed(tokens):
    return (np.asarray(emb[tokens], dtype=np.uint32) << 16).view(np.float32)


def rope_rows(pos):
    inv = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]
    return np.stack((np.cos(ang), np.sin(ang)), axis=1).astype(np.float32)


def rope(z, cs):
    cos, sin = cs[:, 0][:, None, :], cs[:, 1][:, None, :]
    a, b = z[..., :D // 2], z[..., D // 2:]
    return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)


def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


# ---- scenario: ROWS distinct requests ------------------------------------------------------------------------------------------------------
ctx = rng.integers(40, P * PAGE - 1, size=ROWS)                   # committed context length per request (its row is at position ctx)
ctx[0] = P * PAGE - 1                                              # one request fills the bucket exactly
free_pages = rng.permutation(np.arange(0, 5500))
table = np.full((R, P), SCRATCH, np.int64)
pos = np.zeros(R, np.int64)
wslot = np.full(R, SCRATCH * PAGE, np.int64)
x = np.zeros((R, H), np.float32)
cache = np.zeros((TP, PAGES, 2, KVL, PAGE, D), np.uint8)
cscale = np.ones((TP, PAGES, 2, KVL, PAGE), np.float32)
norm1 = [np.frombuffer(weights[(rank, 100)], dtype=np.float32) for rank in range(TP)]
WK = [dequant(rank, 4, 5, KVL * D) for rank in range(TP)]
WV = [dequant(rank, 6, 7, KVL * D) for rank in range(TP)]
used = 0
for i in range(ROWS):
    c = int(ctx[i])
    needed = -(-(c + 1) // PAGE)
    pages = free_pages[used:used + needed].astype(np.int64); used += needed
    table[i, :needed] = pages
    pos[i] = c
    wslot[i] = pages[c // PAGE] * PAGE + c % PAGE
    toks = rng.integers(1000, 120000, size=c + 1)
    xa = embed(toks)
    x[i] = xa[c]
    cs = rope_rows(np.arange(c))
    for rank in range(TP):
        h = rmsnorm(xa[:c], norm1[rank]).astype(np.float32)
        k = rope((h @ WK[rank]).reshape(c, KVL, D), cs)
        v = (h @ WV[rank]).reshape(c, KVL, D)
        kv = np.stack((k, v), axis=1)
        sc = np.maximum(np.abs(kv).max(axis=-1), 1e-20) / 224.0
        codes = (kv / sc[..., None]).astype(E4).view(np.uint8)
        pg, of = pages[np.arange(c) // PAGE], np.arange(c) % PAGE
        cache[rank, pg, :, :, of, :] = codes
        cscale[rank, pg, :, :, of] = sc
cos_sin = rope_rows(pos)
log('ctx per row', ctx.tolist(), 'pages used', used)


def write(name, arr):
    arr.tofile(f'{WORK}/{name}')


def run(neff_dir, tag, xin, cs, p, ws, tab, cache_in, cscale_in):
    for r in range(TP):
        write(f'in21.r{r}', cache_in[r]); write(f'in22.r{r}', cscale_in[r])
    write('in0', xin.astype(np.float32)); write('in1', cs.astype(np.float32)); write('in2', p.astype(np.int64))
    write('in3', ws.astype(np.int64)); write('in4', tab.astype(np.int64))
    t0 = time.time()
    env = dict(os.environ, NEFF_IO_STATES='2', NEURON_RT_VISIBLE_CORES='0-1', NEURON_RT_ROOT_COMM_ID='127.0.0.1:62128')
    res = subprocess.run([NEFF_IO, f'{neff_dir}/graph.neff', WORK, str(TP)], env=env, capture_output=True, text=True)
    if res.returncode:
        log(res.stdout[-2000:], res.stderr[-2000:])
        raise SystemExit(f'neff_io failed for {neff_dir}')
    out = np.fromfile(f'{WORK}/out0.r0', dtype=np.float32).reshape(xin.shape)
    out_r1 = np.fromfile(f'{WORK}/out0.r1', dtype=np.float32).reshape(xin.shape)
    c = np.stack([np.fromfile(f'{WORK}/out1.r{r}', dtype=np.uint8).reshape(cache_in.shape[1:]) for r in range(TP)])
    s = np.stack([np.fromfile(f'{WORK}/out2.r{r}', dtype=np.float32).reshape(cscale_in.shape[1:]) for r in range(TP)])
    log(f'{tag}: {time.time() - t0:.1f} s, ranks agree on x: {np.abs(out - out_r1).max():.3e}')
    return out, c, s


out_new, c_new, s_new = run(NEW, f'new R={R}', x, cos_sin, pos, wslot, table, cache, cscale)
out_loop, c_loop, s_loop = run(LOOP, f'loop R={R}', x, cos_sin, pos, wslot, table, cache, cscale)
# dec8 slices: rows 8k..8k+7 (each request writes only its own pages, so the slices are independent of each other)
out_8 = np.zeros_like(x)
c_8 = cache.copy(); s_8 = cscale.copy()
for k in range(R // 8):
    sl = slice(8 * k, 8 * k + 8)
    o, c8, s8 = run(DEC8, f'dec8 rows {8 * k}-{8 * k + 7}', x[sl], cos_sin[sl], pos[sl], wslot[sl], table[sl], cache, cscale)
    out_8[sl] = o
    wr = (c8 != cache); c_8[wr] = c8[wr]
    ws_ = (s8 != cscale); s_8[ws_] = s8[ws_]


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


rows = []
for i in range(ROWS):
    d_new, d_loop, d_8 = out_new[i] - x[i], out_loop[i] - x[i], out_8[i] - x[i]
    row = {'row': i, 'pos': int(pos[i]), 'pages': int(-(-(pos[i] + 1) // PAGE)), 'new_vs_loop': rel(d_new, d_loop), 'new_vs_dec8': rel(d_new, d_8),
           'loop_vs_dec8': rel(d_loop, d_8), 'max_abs_new_dec8': float(np.abs(out_new[i] - out_8[i]).max()), 'finite': bool(np.isfinite(out_new[i]).all())}
    rows.append(row)
    print(json.dumps(row))
summary = {'R': R, 'P': P, 'rows': ROWS, 'seed': SEED,
           'new_vs_loop_max': max(r['new_vs_loop'] for r in rows), 'new_vs_loop_mean': float(np.mean([r['new_vs_loop'] for r in rows])),
           'new_vs_dec8_max': max(r['new_vs_dec8'] for r in rows), 'new_vs_dec8_mean': float(np.mean([r['new_vs_dec8'] for r in rows])),
           'loop_vs_dec8_max': max(r['loop_vs_dec8'] for r in rows), 'loop_vs_dec8_mean': float(np.mean([r['loop_vs_dec8'] for r in rows])),
           'kv_bytes_differ_new_loop': int((c_new != c_loop).sum() + (s_new != s_loop).sum()),
           'kv_bytes_differ_new_dec8': int((c_new != c_8).sum() + (s_new != s_8).sum()),
           'pad_rows_finite': bool(np.isfinite(out_new[ROWS:]).all())}
print('SUMMARY', json.dumps(summary))
