"""Inkling NEFFs (inkling_graphs.py output) on the NeuronCores against the CPU mirror of the same graph (inkling_graphs.visit /
segment / head, the ranks as threads), one launch each through cpp/neff_io, RANDOM weights of the real shapes (one random
sliding and one random global layer, with message gates; the real visits' ctl vectors), random states.

  NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 \
    python neuron/tests/test_inkling_device.py CONFIG.json NEFF_DIR NEFF_IO_BINARY [bucket names...]
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glm_repack as RP  # noqa: E402
import inkling_graphs as IG  # noqa: E402
import inkling_prepare as PR  # noqa: E402
import inkling_ref as R  # noqa: E402
from glm_runner import ThreadAllReduce  # noqa: E402

REP = {'s': 1, 'S': 7, 'g': 5, 'G': 11}   # a representative layer per visit kind (message gates on 7 / 11)


def shards(cfg, g, fp8):
    """per kind letter -> [rank] weight dicts of one random layer at the real shapes."""
    c = R.Cfg(cfg)
    ck = R.random_checkpoint(c, seed=3, layers=sorted(set(REP.values())), experts=False, vocab=64)
    get = PR.Source(ck)
    gen = torch.Generator().manual_seed(5)
    out = {}
    for kind, l in REP.items():
        gu = torch.randn(c.E, 2 * c.I, c.H, generator=gen) * 0.03
        dn = torch.randn(c.E, c.H, c.I, generator=gen) * 0.03
        ex = PR.experts_from_dense(gu, dn, g, fp8)
        del gu, dn
        out[kind] = [PR.shard_layer(get, l, r, g, fp8, experts=ex[r]) for r in range(g.TP)]
        for w in out[kind]:   # router logits of a realistic spread (std ~0.5): saturated sigmoids tie at 1.0 and flip the top-8
            w['rw'] = w['rw'] * 0.1
    return out


def visit_range(g, kinds):
    for v in range(len(g.visits) - len(kinds) + 1):
        if ''.join(g.kind(v + i) for i in range(len(kinds))) == ''.join(kinds):
            return v
    raise SystemExit(f'no visit range of kinds {kinds}')


def main():
    cfg_path, neffs, neff_io = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    only = set(sys.argv[4:])
    cfg = json.loads(Path(cfg_path).read_text())
    cache = {}
    for d in sorted(neffs.iterdir()):
        if not (d / 'meta.json').exists() or (only and d.name not in only):
            continue
        meta = json.loads((d / 'meta.json').read_text())
        if not meta['kind'].startswith('inkling_'):
            continue
        if meta.get('wt'):
            assert IG.WT, 'a WT NEFF needs WT=1 in the environment'
        g = IG.Geo(cfg, meta['geometry']['tp'])
        if meta['kind'] == 'inkling_head':
            check_head(d, meta, g, cfg, neff_io)
            continue
        key = meta['fp8']
        if key not in cache:
            cache[key] = shards(cfg, g, meta['fp8'])
        check(d, meta, g, cache[key], neff_io)


def check(d, meta, g, sh, neff_io):
    R_, Q_, P_ = meta['R'], meta['Q'], meta['P']
    kinds = meta['kinds']
    T = R_ * Q_
    torch.manual_seed(1)
    x = torch.randn(T, IG.NS, g.H) * 0.5
    x[:, 5:] *= 0.8
    x = x.reshape(T, -1)
    start = int(os.environ.get('DEV_START', min(1500, P_ * 16 - Q_ - 37 * R_)))   # every token's page inside the table (no duplicate writes)
    pos = torch.tensor([start + (i % Q_) if R_ == 1 else start + 37 * i for i in range(T)], dtype=torch.int64)
    pages = meta['pages']
    table = torch.stack([(torch.arange(P_, dtype=torch.int64) + r * P_) % (pages - 1) for r in range(R_)])
    wslot = torch.stack([table[i // Q_, min(int(pos[i]) // 16, P_ - 1)] * 16 + pos[i] % 16 for i in range(T)])
    step = {'x': x, 'pos': pos, 'wslot': wslot, 'table': table}
    v0 = visit_range(g, kinds)
    names = meta['inputs'] + meta['states']
    states = {}
    for i, k in enumerate(kinds):
        pre = f'l{i}.' if len(kinds) > 1 else ''
        for n, (shape, dt) in IG.state_shapes(k, g, R_, pages).items():
            if n.startswith('buf'):
                v = torch.randn(shape) * 0.3
                v[:, 0] = R.l2clip(v[:, 0] * 3, g.c.qk_clip)
                states[pre + n] = v.to(dt)
            elif n.startswith('conv'):
                states[pre + n] = torch.randn(shape) * 0.3
            else:
                states[pre + n] = (torch.randn(shape) * 0.5).to(dt)
    per_rank = []
    for r in range(g.TP):
        t = dict(step)
        for i, k in enumerate(kinds):
            pre = f'l{i}.' if len(kinds) > 1 else ''
            w = sh[k][r]
            for n in IG.weight_shapes(k, g, meta['fp8']):
                if n == 'ctl':
                    t[pre + n] = torch.tensor(g.ctl(v0 + i), dtype=torch.float32)
                else:
                    t[pre + n] = IG.to_wt(n, w[n]) if meta.get('wt') else w[n]
        t.update({k: v.clone() for k, v in states.items()})
        per_rank.append(t)
    ar, ref, rst = ThreadAllReduce(g.TP), [None] * g.TP, [None] * g.TP
    moe_mode = meta.get('moe', 'all')

    def run(r):
        t = dict(per_rank[r])
        if len(kinds) > 1:
            xo, st = IG.segment(kinds, t, g, R_, Q_, P_, ar(r), moe_mode)
        else:
            xo, st = IG.visit(kinds[0], t, g, R_, Q_, P_, ar(r), moe_mode, meta.get('part', 'full'))
        ref[r], rst[r] = xo.reshape(T, -1), st
    th = [threading.Thread(target=run, args=(r,)) for r in range(g.TP)]
    [t_.start() for t_ in th]
    [t_.join() for t_ in th]
    with tempfile.TemporaryDirectory(dir=os.environ.get('IO_TMP')) as io:
        for r in range(g.TP):
            for i, n in enumerate(names):
                Path(io, f'in{i}.r{r}').write_bytes(RP.tensor_bytes(per_rank[r][n].contiguous()))
        env = dict(os.environ, NEFF_IO_STATES=str(len(meta['states'])))
        t0 = time.time()
        subprocess.run([neff_io, str(d / 'graph.neff'), io, str(g.TP)], check=True, stdout=subprocess.DEVNULL, env=env)
        dt = time.time() - t0
        out = np.frombuffer(Path(io, 'out0.r0').read_bytes(), np.float32).reshape(T, -1)
        y_dev = torch.from_numpy(out.copy()) - x
        y_ref = ref[0] - x
        rel = float(torch.linalg.norm(y_dev - y_ref) / torch.linalg.norm(y_ref))
        fin = bool(np.isfinite(out).all())
        if os.environ.get('DEV_DETAIL'):
            yd, yr = y_dev.reshape(T, IG.NS, -1), y_ref.reshape(T, IG.NS, -1)
            pt = torch.linalg.norm(yd[:, 4] - yr[:, 4], dim=-1) / torch.linalg.norm(yr[:, 4], dim=-1)
            print('  slot 4 per-token rel: median', f'{float(pt.median()):.2e}', 'max', f'{float(pt.max()):.2e}', 'at', int(pt.argmax()),
                  'worst tokens', [int(i) for i in torch.topk(pt, min(8, T)).indices])
            print('  slot rel:', ' '.join(f'{j}:{float(torch.linalg.norm(yd[:, j] - yr[:, j]) / (torch.linalg.norm(yr[:, j]) + 1e-30)):.2e}' for j in range(IG.NS)))
            for si, n in enumerate(meta['states']):
                ref_s = rst[0].get(n)
                if ref_s is None:
                    continue
                raw = Path(io, f'out{si + 1}.r0').read_bytes()
                if ref_s.dtype == torch.bfloat16:
                    dev_s = torch.from_numpy(np.frombuffer(raw, np.int16).copy()).view(torch.bfloat16)
                else:
                    dev_s = torch.from_numpy(np.frombuffer(raw, np.float32).copy())
                a, b_ = dev_s.float().reshape(-1), ref_s.float().reshape(-1)
                d0 = per_rank[0][n].float().reshape(-1)
                if n.split('.')[-1].startswith('conv'):
                    C2 = 2 * g.CK
                    parts = {'kv': slice(0, C2 * (g.K - 1)), 'a': slice(C2 * (g.K - 1), (C2 + g.H) * (g.K - 1)), 'm': slice((C2 + g.H) * (g.K - 1), None)}
                    print('   conv parts rel:', ' '.join(f'{k}:{float(torch.linalg.norm(a[sl] - b_[sl]) / torch.linalg.norm(b_[sl])):.2e}' for k, sl in parts.items()))
                print(f'  state {n}: rel {float(torch.linalg.norm(a - b_) / torch.linalg.norm(b_)):.2e}; changed part rel {float(torch.linalg.norm((a - d0) - (b_ - d0)) / (torch.linalg.norm(b_ - d0) + 1e-30)):.2e}')
        same = max([float(np.abs(np.frombuffer(Path(io, f'out0.r{r}').read_bytes(), np.float32).reshape(T, -1) - out).max()) for r in range(1, g.TP)] or [0.0])
    print(f'{d.name:24s} kinds={"".join(kinds)} R={R_} Q={Q_} P={P_} ({"fp8" if meta["fp8"] else "bf16"}, moe={moe_mode}, wt={meta.get("wt")}): '
          f'rel L2 vs CPU mirror {rel:.3e}; finite {fin}; ranks differ by {same:.1e}; neff_io {dt:.1f}s', flush=True)


def check_head(d, meta, g, cfg, neff_io):
    T, S = meta['T'], meta['S']
    c = R.Cfg(cfg)
    gen = torch.Generator().manual_seed(9)
    ck = {f'{R.P}unembed.weight': (torch.randn(c.V, c.H, generator=gen) * 0.02).bfloat16(),
          f'{R.P}attn_res.output_query': torch.randn(c.H, generator=gen) * 0.5,
          f'{R.P}norm.weight': (1 + 0.1 * torch.randn(c.H, generator=gen)).bfloat16()}
    get = PR.Source(ck)
    heads = [PR.shard_head(get, r, g, meta['fp8']) for r in range(g.TP)]
    if meta.get('wt'):
        heads = [{k: IG.to_wt(k, v) for k, v in h.items()} for h in heads]
    x = torch.randn(T, IG.NS * g.H) * 0.5
    rows = torch.arange(S, dtype=torch.int64) % T
    with tempfile.TemporaryDirectory(dir=os.environ.get('IO_TMP')) as io:
        ref = []
        for r in range(g.TP):
            t = {**heads[r], 'x': x, 'rows': rows}
            ref.append(IG.head(t, g, rows))
            for i, n in enumerate(meta['inputs']):
                Path(io, f'in{i}.r{r}').write_bytes(RP.tensor_bytes(t[n].contiguous()))
        subprocess.run([neff_io, str(d / 'graph.neff'), io, str(g.TP)], check=True, stdout=subprocess.DEVNULL)
        rels = []
        for r in range(g.TP):
            out = torch.from_numpy(np.frombuffer(Path(io, f'out0.r{r}').read_bytes(), np.float32).copy()).reshape(S, -1)
            rels.append(float(torch.linalg.norm(out - ref[r]) / torch.linalg.norm(ref[r])))
    print(f'{d.name:24s} head T={T} S={S}: rel L2 vs CPU mirror per rank {" ".join(f"{x:.3e}" for x in rels)}', flush=True)


if __name__ == '__main__':
    main()
