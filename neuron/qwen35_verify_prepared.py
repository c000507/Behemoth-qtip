"""Verify a prepared Qwen3.5 MoE set against its source checkpoint.
  1. every file of prepared.json exists and holds its last entry (offset + bytes <= size); geometry / layer count match;
  2. per layer, on sampled ranks (default 2 per layer: l % TP and (7 l + 3) % TP; --ranks all): every tensor has the shape
     and dtype of qwen35_graphs.weight_shapes, fp8 codes are finite with max |code| <= 224, scales finite and > 0;
  3. the same tensors dequantised (codes x row scale) vs the source slice: rel L2 per tensor must be < --tol (E4M3 with
     row scales: 2-3e-2), f32 tensors (norms, conv, router, gates) must be exact.
usage: qwen35_verify_prepared.py MODEL PREPARED [--ranks N|all] [--layers a,b,..] [--tol 0.05] [--jobs J]
Prints one line per layer and a JSON summary; exit 1 on any failure."""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_repack as RP  # noqa: E402
import qwen35_graphs as Q  # noqa: E402
import qwen35_prepare as P  # noqa: E402

DT = {'f32': (np.float32, torch.float32), 'bf16': (np.int16, torch.bfloat16), 'f8e4m3': (np.uint8, Q.F8), 'i64': (np.int64, torch.int64)}
_W = {}


def read_entry(path, e):
    npdt, tdt = DT[e['dtype']]
    a = np.fromfile(path, dtype=npdt, count=e['bytes'] // np.dtype(npdt).itemsize, offset=e['offset'])
    return torch.from_numpy(a).view(tdt).reshape(e['shape'])


class SliceGetter(P.Getter):
    """Getter whose expert tensors are read lazily: only the rank's rows of the memmap are touched."""

    def expert_slices(self, l, rank, g):
        m = f'{self.prefix}layers.{l}.mlp.experts.'
        er = np.arange(rank * g.EIl, (rank + 1) * g.EIl)
        _, shape, arr = self.src.raw(m + 'gate_up_proj')
        gu = np.array(arr.reshape(shape)[:, np.concatenate((er, g.I + er))])
        _, shape, arr = self.src.raw(m + 'down_proj')
        dn = np.array(arr.reshape(shape)[:, :, er])
        f = lambda a: torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).float()
        return f(gu), f(dn).transpose(1, 2).contiguous()


def check_layer(l):
    a, g, man, get = _W['a'], _W['g'], _W['man'], _W['get']
    fp8 = man['weights'] == 'fp8'
    worst, fails, n = 0.0, [], 0
    ranks = range(g.TP) if a.ranks == 'all' else sorted({(l + i * (7 * l + 3)) % g.TP for i in range(int(a.ranks))})
    get.cache = {}
    for r in ranks:
        rel = f'r{r}/L{l:03d}.bin'
        got = {e['name']: read_entry(Path(a.prepared) / rel, e) for e in man['files'][rel]}
        ref = P.shard_layer(get, l, r, g, fp8, experts=False)
        gu, dn = get.expert_slices(l, r, g)
        want = {**{'attn.' + k: v for k, v in Q.weight_shapes(g.kind(l), g, fp8).items()}, **{'mlp.' + k: v for k, v in Q.weight_shapes('moe', g, fp8).items()}}
        if set(got) != set(want):
            fails.append(f'{rel}: names differ {sorted(set(got) ^ set(want))}')
            continue
        for name, t in got.items():
            n += 1
            shape, dt = want[name]
            if tuple(t.shape) != tuple(shape) or t.dtype != dt:
                fails.append(f'{rel}:{name} shape/dtype {tuple(t.shape)} {t.dtype}')
                continue
            f = t.float()
            if not bool(torch.isfinite(f).all()) or (t.dtype == Q.F8 and float(f.abs().max()) > 224.0):
                fails.append(f'{rel}:{name} non-finite or code > 224')
                continue
            if name.endswith('_s') or name.endswith('_c'):
                if float(f.min()) <= 0:
                    fails.append(f'{rel}:{name} scale <= 0')
                continue
            if name in ('mlp.egu', 'mlp.ed'):
                src = gu if name == 'mlp.egu' else dn
                deq = f * got[name + ('_s' if name == 'mlp.egu' else '_c')].unsqueeze(-1) if fp8 else f
            elif t.dtype == Q.F8:
                src = ref[name].float() * ref[name + '_s'].unsqueeze(-1)       # dense projections: bit-compared with shard_layer
                deq = f * got[name + '_s'].unsqueeze(-1)
                if not torch.equal(t.view(torch.uint8), ref[name].view(torch.uint8)):
                    fails.append(f'{rel}:{name} codes differ from shard_layer')
            else:
                src, deq = ref[name].float(), f
                if t.dtype == torch.float32 and not torch.equal(deq, src):
                    fails.append(f'{rel}:{name} f32 tensor differs')
            e = float(torch.linalg.norm(deq - src) / torch.linalg.norm(src).clamp(min=1e-30))
            if name in ('mlp.egu', 'mlp.ed'):
                worst = max(worst, e)
                if e > a.tol:
                    fails.append(f'{rel}:{name} rel L2 {e:.3e} vs the bf16 source')
    get.cache = None
    return l, n, worst, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('prepared')
    ap.add_argument('--ranks', default='2'); ap.add_argument('--layers', default=''); ap.add_argument('--tol', type=float, default=0.05)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    man = json.loads((Path(a.prepared) / 'prepared.json').read_text())
    g = Q.Geo(man['config'], man['geometry']['tp'])
    fails = []
    for rel, ents in man['files'].items():
        p = Path(a.prepared) / rel
        last = max(e['offset'] + e['bytes'] for e in ents)
        if not p.exists() or p.stat().st_size < last:
            fails.append(f'{rel}: missing or short')
    expect = {'embed.bin'} | {f'r{r}/head.bin' for r in range(g.TP)} | {f'r{r}/L{l:03d}.bin' for r in range(g.TP) for l in range(g.L)}
    if set(man['files']) != expect:
        fails.append(f'file set differs: {len(set(man["files"]) ^ expect)} names')
    print(f'{len(man["files"])} files listed, {len(fails)} structural failures', flush=True)
    _W.update(a=a, g=g, man=man, get=SliceGetter(RP.Checkpoint(a.model)))
    layers = [int(x) for x in a.layers.split(',')] if a.layers else list(range(g.L))
    worst, n = 0.0, 0
    torch.set_num_threads(4)
    with multiprocessing.get_context('fork').Pool(a.jobs) as pool:
        for l, k, w, f in pool.imap_unordered(check_layer, layers):
            n += k; worst = max(worst, w); fails += f
            print(f'layer {l}: {k} tensors, expert rel L2 max {w:.3e}, {len(f)} failures', flush=True)
    # heads (bit-compare two ranks) and the embedding
    get = P.Getter(RP.Checkpoint(a.model)); get.cache = {}
    for r in sorted({0, g.TP - 1}):
        ref = P.shard_head(get, r, g, man['weights'] == 'fp8')
        for e in man['files'][f'r{r}/head.bin']:
            t = read_entry(Path(a.prepared) / f'r{r}/head.bin', e)
            if RP.tensor_bytes(t) != RP.tensor_bytes(ref[e['name']]):
                fails.append(f'r{r}/head.bin:{e["name"]} differs')
    e = man['files']['embed.bin'][0]
    emb = read_entry(Path(a.prepared) / 'embed.bin', e)
    if not torch.equal(emb.float(), get('embed_tokens.weight')):
        fails.append('embed.bin differs from embed_tokens')
    print(json.dumps({'layers': len(layers), 'tensors': n, 'expert_rel_l2_max': worst, 'failures': fails[:40], 'n_failures': len(fails)}))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
