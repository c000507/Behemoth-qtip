"""CPU checks of the Neuron GLM graphs (glm_graphs.py) and their sharding (glm_repack.py) against the glm53 oracle.

The graph functions run on the CPU through glm_runner.CpuRunner (TP ranks as threads, all-reduce = sum), on tiny random
dense and MoE checkpoints whose index_topk (16) makes the sparse pooled top-k selection engage after 16 keys.

  GLM53_BIN=build-glm53/glm53 python neuron/tests/test_glm_cpu.py

Tolerances are per position (see check()). Checks: bf16 weights at TP=1 vs the oracle (bf16 arithmetic), TP=2 with EP=2 and EP=1 (expert TP), chunked prefill vs
token steps, MOE=gather vs all, two interleaved sequences (state slots, page tables), FP8 weights and the FP8 latent cache.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import glm_graphs as G
import glm_repack as RP
from glm_runner import CpuRunner
from glm_tiny import make_checkpoint, oracle_logits, tiny_config

torch.set_num_threads(8)


def stats(a, b):
    """-> (median per-position relative L2, fraction of positions within 3x tolerance is computed by check, top-1, KL)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    rel = np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)
    top = float(np.mean(a.argmax(-1) == b.argmax(-1)))
    pa = np.exp(a - a.max(-1, keepdims=True)); pa /= pa.sum(-1, keepdims=True)
    pb = np.exp(b - b.max(-1, keepdims=True)); pb /= pb.sum(-1, keepdims=True)
    kl = float(np.mean(np.sum(pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30)), -1)))
    return rel, top, kl


def runner(model, cfg, tp, ep=None, fp8=False, kv8=False, moe='all'):
    g = G.Geo(cfg, tp, ep)
    pk = RP.Packer(RP.Checkpoint(model), g, fp8)
    layers = [[pk.layer(l, r) for l in range(g.L)] for r in range(tp)]
    heads = [pk.head(r) for r in range(tp)]
    return CpuRunner(g, layers, heads, pk.embed(), pages=65, slots=9, kv8=kv8, moe_mode=moe, max_pages_per_seq=4)


def run_tokens(r, tokens, chunk=1, seq=0):
    out = []
    for s in range(0, len(tokens), chunk):
        toks = tokens[s:s + chunk]
        out.append(r.step([(seq, toks, s)]).numpy())
    return np.concatenate(out)


def fp8_blocks(model, cfg, check):
    """Each sub-block once with FP8 weights against the same block with bf16 weights (TP=2, 8-token chunk, random
    residual): the FP8 mechanics (scales, expert layouts, the FP8 latent cache) without end-to-end amplification."""
    import torch
    from glm_runner import ThreadAllReduce
    import threading
    g = G.Geo(cfg, 2)
    packs = {f: RP.Packer(RP.Checkpoint(model), g, f) for f in (False, True)}
    torch.manual_seed(0)
    R, Q, P = 1, 8, 4
    x = G.bf(torch.randn(R * Q, g.S * g.H))
    step = {'x': x, 'pos': torch.arange(Q), 'wslot': torch.arange(Q), 'table': torch.arange(P).reshape(1, P), 'slot': torch.zeros(1, dtype=torch.int64)}
    for l in range(g.L):
        for sub in ('attn', 'mlp'):
            kind = ('kda' if g.attn[l] == 'linear_attention' else 'mla') if sub == 'attn' else g.mlp[l]
            outs = {}
            for f, kv8 in ((False, False), (True, False), (True, True)):
                if kv8 and kind != 'mla':
                    continue
                ar, res = ThreadAllReduce(2), [None, None]

                def run(rank):
                    t = {k[len(sub) + 1:]: v for k, v in packs[f].layer(l, rank).items() if k.startswith(sub + '.')}
                    t.update(step)
                    t.update({k: G.zeros(*v) for k, v in G.state_shapes(kind, g, 8, 2, kv8).items()})
                    res[rank] = G.sublayer(kind, t, g, R, Q, P, ar(rank), kv8)[0] - x
                th = [threading.Thread(target=run, args=(r,)) for r in range(2)]
                [h.start() for h in th]; [h.join() for h in th]
                outs[(f, kv8)] = res[0].numpy()
            check(f'layer {l} {kind} FP8 block vs bf16 block', outs[(True, False)], outs[(False, False)], 0.06, 0.0, 1.0)
            if (True, True) in outs:
                check(f'layer {l} {kind} FP8 block + FP8 latent cache', outs[(True, True)], outs[(False, False)], 0.06, 0.0, 1.0)


def main():
    binary = os.environ.get('GLM53_BIN', 'build-glm53/glm53')
    rng = np.random.default_rng(3)
    tokens = [int(t) for t in rng.integers(2, 290, 44)]
    positions = list(range(len(tokens)))
    failures = []

    def check(name, got, ref, rel_max, top_min, min_frac=1.0):
        # The sparse top-k pool selection and the expert routing are discrete: a near-tie decided differently by a
        # different (but equally exact) summation order changes a position's logits by tens of percent, and the KDA
        # state carries that into later positions. Configurations that select (index_topk 16) therefore bound the
        # median position and a fraction of positions (3x the bound); without selection every position is bounded.
        rel, top, kl = stats(got, ref)
        med, frac = float(np.median(rel)), float(np.mean(rel <= 3 * rel_max))
        ok = med <= rel_max and frac >= min_frac and top >= top_min
        print(f'{"ok  " if ok else "FAIL"} {name}: rel L2 median {med:.2e} max {rel.max():.2e} (<= {3 * rel_max:.0e}: {frac:.2f}), '
              f'top-1 {top:.3f}, KL {kl:.2e}', flush=True)
        if not ok:
            failures.append(name)
    with tempfile.TemporaryDirectory() as d:
        for moe in (True, False):
            for topk in (512, 16):
                cfg = tiny_config(moe=moe, index_topk=topk)
                tag = ('moe' if moe else 'dense') + f' topk={topk}'
                model = make_checkpoint(Path(d) / tag.replace(' ', '_'), cfg)
                ref = oracle_logits(binary, model, tokens, positions, 'bf16')
                frac, top = (1.0, 0.95) if topk == 512 else (0.75, 0.88)
                check(f'{tag} TP=1 bf16 vs oracle', run_tokens(runner(model, cfg, 1), tokens), ref, 2e-2, 0.95)
                check(f'{tag} TP=2 EP=2 bf16 vs oracle', run_tokens(runner(model, cfg, 2), tokens), ref, 2e-2, top, frac)
                if moe:
                    check(f'{tag} TP=2 EP=1 (expert TP)', run_tokens(runner(model, cfg, 2, 1), tokens), ref, 2e-2, top, frac)
                    check(f'{tag} MOE=gather', run_tokens(runner(model, cfg, 2, 2, moe='gather'), tokens), ref, 2e-2, top, frac)
                check(f'{tag} prefill chunks of 4', run_tokens(runner(model, cfg, 2), tokens, chunk=4), ref, 2e-2, top, frac)
                r = runner(model, cfg, 2)                         # two sequences as rows of one step: slots, page tables
                other = tokens[::-1]
                a, b = [], []
                for i in range(len(tokens)):
                    y = r.step([(0, [tokens[i]], i), (1, [other[i]], i)]).numpy()
                    a.append(y[:1]); b.append(y[1:])
                check(f'{tag} 2 interleaved rows (seq 0)', np.concatenate(a), ref, 2e-2, top, frac)
                check(f'{tag} 2 interleaved rows (seq 1)', np.concatenate(b), oracle_logits(binary, model, other, positions, 'bf16'), 2e-2, top, frac)
                if topk == 512:
                    fp8_blocks(model, cfg, check)
                    # end to end on a random tiny network FP8 errors compound (no trained robustness): reported, garbage-guarded
                    check(f'{tag} FP8 weights e2e (report)', run_tokens(runner(model, cfg, 2, fp8=True), tokens), ref, 0.5, 0.3, 0.5)
                    check(f'{tag} FP8 weights + FP8 latent e2e (report)', run_tokens(runner(model, cfg, 2, fp8=True, kv8=True), tokens), ref, 0.5, 0.3, 0.5)
    if failures:
        raise SystemExit('FAILED: ' + ', '.join(failures))
    print('all GLM CPU graph checks passed')


if __name__ == '__main__':
    main()
