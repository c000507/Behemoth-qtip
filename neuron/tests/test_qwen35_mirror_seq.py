"""Streaming CPU mirror of the fp8 TP graphs over a whole token sequence (teacher forced), layer by layer with the real
checkpoint: sharded like the prepared set (qwen35_prepare.shard_layer, fp8), the graph functions (qwen35_graphs.sublayer,
2 ranks as threads) in prefill form (R = 1, Q = T), the fp8 head. Reports per layer whether the residual is finite and its
max |x| (to locate an overflow the device shows), and the argmax agreement with the device's generated tokens.

usage: python neuron/tests/test_qwen35_mirror_seq.py MODEL_DIR run.json [out.json]   (run.json as test_qwen35_oracle.py)
"""
import json
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glm_repack as RP  # noqa: E402
import qwen35_graphs as Q  # noqa: E402
import qwen35_prepare as P  # noqa: E402
from glm_runner import ThreadAllReduce  # noqa: E402


def main():
    model, run = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    cfg = json.loads((model / 'config.json').read_text())
    g = Q.Geo(cfg, 2)
    ck = RP.Checkpoint(model)
    get = P.Getter(ck)
    prompt, gen = run['prompt'], run['generated']
    tokens = torch.tensor(prompt + gen[:-1], dtype=torch.int64)
    T = len(tokens)
    Pg = (T + 15) // 16
    pages, slots = Pg + 1, 2
    torch.set_num_threads(8)
    x = get('embed_tokens.weight')[tokens].float()
    pos = torch.arange(T, dtype=torch.int64)
    step = {'pos': pos, 'slot': torch.tensor([0]), 'table': torch.arange(Pg, dtype=torch.int64).reshape(1, Pg), 'wslot': pos.clone(),
            'cos_sin': Q.cos_sin_rows(pos, g.rot, g.theta)}
    report = []
    for l in range(g.L):
        t0 = time.time()
        kind = g.kind(l)
        shards = [P.shard_layer(get, l, r, g, True) for r in range(g.TP)]
        states = {k: Q.zeros(*v) for k, v in Q.state_shapes(kind, g, pages, slots).items()}
        ar, out = ThreadAllReduce(g.TP), [None] * g.TP

        def run_rank(r):
            w = shards[r]
            ta = {**step, 'x': x, **{k[5:]: v for k, v in w.items() if k.startswith('attn.')}, **{k: v.clone() for k, v in states.items()}}
            xa, _ = Q.sublayer(kind, ta, g, 1, T, Pg, ar(r))
            tm = {**step, 'x': xa, **{k[4:]: v for k, v in w.items() if k.startswith('mlp.')}}
            out[r] = (xa, Q.sublayer('moe', tm, g, 1, T, Pg, ar(r), 'all')[0])
        th = [threading.Thread(target=run_rank, args=(r,)) for r in range(g.TP)]
        [t.start() for t in th]
        [t.join() for t in th]
        xa, x = out[0]
        fa, fm = bool(torch.isfinite(xa).all()), bool(torch.isfinite(x).all())
        amax_pos = int(torch.nan_to_num(x.abs(), nan=float('inf')).amax(dim=1).argmax())
        rec = {'layer': l, 'kind': kind, 'attn_finite': fa, 'finite': fm, 'max_abs': float(torch.nan_to_num(x.abs(), nan=0.0).max()),
               'max_abs_pos': amax_pos, 'seconds': round(time.time() - t0, 1)}
        report.append(rec)
        print(json.dumps(rec), flush=True)
        if not fm:
            bad = (~torch.isfinite(x)).any(dim=1).nonzero().flatten().tolist()
            print('  non-finite rows (positions):', bad[:20], flush=True)
    heads = [P.shard_head(get, r, g, True) for r in range(g.TP)]
    rows = torch.arange(T)
    logits = torch.cat([Q.head({'x': x, **heads[r]}, g, rows)[:, :g.VP // g.TP] for r in range(g.TP)], dim=1)[:, :g.V]
    n0 = len(prompt) - 1
    agree = sum(int(logits[n0 + i].argmax()) == tok for i, tok in enumerate(gen))
    first_diff = next((i for i, tok in enumerate(gen) if int(logits[n0 + i].argmax()) != tok), None)
    print(f'fp8 mirror teacher-forced top-1 agreement with the device: {agree}/{len(gen)} (first difference at {first_diff})')
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps({'layers': report, 'agree': agree, 'n': len(gen), 'first_diff': first_diff}))


if __name__ == '__main__':
    main()
