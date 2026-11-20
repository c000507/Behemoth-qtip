"""f32 reference of a prefill straight from a PREPARED set (no HF checkpoint needed): the graph functions of qwen35_graphs
in EXACT mode (f32 activations, no bf16 / fp8 activation rounding, f32 K/V, the token-loop recurrence, every expert
on every token) over the whole prompt as ONE chunk, with the prepared fp8 weights dequantised (code x row / channel scale).
What it isolates: everything the device graphs add on top of the stored weights - fp8 activation quantisation, bf16
rounding, the 8-bit KV cache, the chunk partition, and the FORM of a block (token loop vs chunked delta rule, MOE=all vs
MOE=group). Two device builds are equivalent when their logits are equally far from this reference.

usage: EXACT=1 python neuron/tests/qwen35_prepared_ref.py PREPARED_DIR tokens.json OUT.npy [LAST_N]
  tokens.json: a JSON array of prompt ids, or the server's BEHEMOTH_DIAGNOSTIC_DIR record ({"tokens": prompt + generated, "prompt_tokens": n})
  OUT.npy: f32 logits [LAST_N, vocab] of the last LAST_N prompt positions (default 1: the prefill-end logits)
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ['EXACT'] = '1'
os.environ.pop('KV8', None)
os.environ.pop('GDN_SLOT', None)
os.environ.pop('GDN_CHUNK', None)

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import qwen35_graphs as Q  # noqa: E402
from glm_runner import ThreadAllReduce  # noqa: E402

DT = {'f32': (np.float32, torch.float32), 'bf16': (np.int16, torch.bfloat16), 'f8e4m3': (np.uint8, Q.F8), 'i64': (np.int64, torch.int64)}


def read_file(prepared, man, rel, prefix=''):
    """{name: tensor} of one prepared file, fp8 tensors dequantised to f32 with their scales folded in (the *_s / *_c
    entries are dropped, so EXACT-mode graph code sees plain f32 weights)."""
    raw = {}
    for e in man['files'][rel]:
        npdt, tdt = DT[e['dtype']]
        a = np.fromfile(Path(prepared) / rel, dtype=npdt, count=e['bytes'] // np.dtype(npdt).itemsize, offset=e['offset'])
        raw[e['name']] = torch.from_numpy(a).view(tdt).reshape(e['shape'])
    out = {}
    for n, v in raw.items():
        if n.endswith('_s') or n.endswith('_c'):
            continue
        if n + '_s' in raw:                       # row scales: one per output row ([..., out, in] x [..., out])
            v = v.float() * raw[n + '_s'].unsqueeze(-1)
        elif n + '_c' in raw:                     # expert down_proj [E, I, H]: one scale per input channel [E, I]
            v = v.float() * raw[n + '_c'].unsqueeze(-1)
        elif v.dtype in (Q.F8, torch.bfloat16):
            v = v.float()
        out[n] = v
    return {k[len(prefix):]: v for k, v in out.items() if k.startswith(prefix)} if prefix else out


def main():
    prepared, tok_path, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    last_n = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    man = json.loads((prepared / 'prepared.json').read_text())
    g = Q.Geo(man['config'], man['geometry']['tp'])
    toks = json.loads(Path(tok_path).read_text())
    tokens = torch.tensor(toks['tokens'][:toks.get('prompt_tokens', len(toks['tokens']))] if isinstance(toks, dict) else toks, dtype=torch.int64)
    T = len(tokens)
    Pg = (T + 15) // 16
    pages, slots = Pg + 1, 2
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // g.TP))
    e = man['files']['embed.bin'][0]
    emb = np.memmap(prepared / 'embed.bin', dtype=np.int16, mode='r', offset=e['offset'], shape=tuple(e['shape']))
    x = torch.from_numpy(np.ascontiguousarray(emb[tokens.numpy()])).view(torch.bfloat16).float()
    pos = torch.arange(T, dtype=torch.int64)
    step = {'pos': pos, 'slot': torch.tensor([0]), 'table': torch.arange(Pg, dtype=torch.int64).reshape(1, Pg), 'wslot': pos.clone(),
            'cos_sin': Q.cos_sin_rows(pos, g.rot, g.theta)}
    t_all = time.time()
    for l in range(g.L):
        t0 = time.time()
        kind = g.kind(l)
        shards = [read_file(prepared, man, f'r{r}/L{l:03d}.bin') for r in range(g.TP)]
        states = {k: Q.zeros(*v) for k, v in Q.state_shapes(kind, g, pages, slots).items()}
        states = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in states.items()}   # f32 K/V
        ar, out = ThreadAllReduce(g.TP), [None] * g.TP

        def run_rank(r):
            w = shards[r]
            ta = {**step, 'x': x, **{k[5:]: v for k, v in w.items() if k.startswith('attn.')}, **{k: v.clone() for k, v in states.items()}}
            xa, _ = Q.sublayer(kind, ta, g, 1, T, Pg, ar(r))
            tm = {**step, 'x': xa, **{k[4:]: v for k, v in w.items() if k.startswith('mlp.')}}
            out[r] = Q.sublayer('moe', tm, g, 1, T, Pg, ar(r), 'all')[0]
        th = [threading.Thread(target=run_rank, args=(r,)) for r in range(g.TP)]
        [t.start() for t in th]
        [t.join() for t in th]
        x = out[0]
        print(f'layer {l} {kind}: max |x| {float(x.abs().max()):.2f} finite {bool(torch.isfinite(x).all())} {time.time() - t0:.1f}s', flush=True)
    heads = [read_file(prepared, man, f'r{r}/head.bin') for r in range(g.TP)]
    rows = torch.arange(T - last_n, T)
    logits = torch.cat([Q.head({'x': x, **heads[r]}, g, rows)[:, :g.VP // g.TP] for r in range(g.TP)], dim=1)[:, :g.V]
    np.save(out_path, logits.numpy().astype(np.float32))
    print(f'{T} tokens, {time.time() - t_all:.0f}s; last position argmax {int(logits[-1].argmax())}')


if __name__ == '__main__':
    main()
