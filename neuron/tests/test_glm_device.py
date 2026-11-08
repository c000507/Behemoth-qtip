"""Each GLM sub-block graph (glm_graphs.sublayer) compiled for one NeuronCore (TP=1, bf16 weights, so torch_neuronx
traces and runs it directly) against the same function on the CPU, on a tiny random checkpoint with the sparse top-k
selection active. Buckets: two 16-token prefill chunks (positions 0-15, 16-31) and a 2-row decode step (32, 33).

  NEURON_RT_VISIBLE_CORES=0 python neuron/tests/test_glm_device.py [moe|dense] [all|gather]
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import glm_graphs as G
import glm_repack as RP
from glm_tiny import make_checkpoint, tiny_config


def main():
    import torch_neuronx
    moe = (sys.argv[1] if len(sys.argv) > 1 else 'moe') == 'moe'
    mode = sys.argv[2] if len(sys.argv) > 2 else 'all'
    cfg = tiny_config(moe=moe)
    d = tempfile.mkdtemp()
    model = make_checkpoint(Path(d) / 'm', cfg)
    g = G.Geo(cfg, 1)
    pk = RP.Packer(RP.Checkpoint(model), g, False)
    P, pages, slots = 4, 9, 3
    rng = np.random.default_rng(5)
    torch.manual_seed(1)
    worst = 0.0
    compiled = {}
    for l in range(g.L):
        lw = pk.layer(l, 0)
        for sub in ('attn', 'mlp'):
            kind = ('kda' if g.attn[l] == 'linear_attention' else 'mla') if sub == 'attn' else g.mlp[l]
            w = {k[len(sub) + 1:]: v for k, v in lw.items() if k.startswith(sub + '.')}
            states = {k: G.zeros(*v) for k, v in G.state_shapes(kind, g, pages, slots).items()}
            snames = list(states)
            for R, Q, start in ((1, 16, 0), (1, 16, 16), (2, 1, 32)):
                T = R * Q
                x = G.bf(torch.randn(T, g.S * g.H) * 0.5)
                pos = torch.tensor([start + (i % Q) for i in range(T)], dtype=torch.int64) if R == 1 else torch.tensor([start, start + 1])
                table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]][:R], dtype=torch.int64)
                wslot = torch.stack([table[i // Q, int(pos[i]) // 16] * 16 + pos[i] % 16 for i in range(T)])
                slot = torch.arange(R, dtype=torch.int64)
                step = {'x': x, 'pos': pos, 'wslot': wslot, 'table': table, 'slot': slot}
                names = [n for n in G.step_shapes(kind, R, Q, P, g)] + list(w) + snames
                args = [{**step, **w, **states}[n] for n in names]

                def fn(*a, R=R, Q=Q, kind=kind, names=names):
                    t = dict(zip(names, a))
                    y, st = G.sublayer(kind, t, g, R, Q, P, lambda v: v, moe_mode=mode)
                    return (y,) + tuple(st[n] for n in snames)
                ref = fn(*args)
                key = (kind, R, Q)
                if key not in compiled:
                    t0 = time.time()
                    compiled[key] = torch_neuronx.trace(fn, tuple(args))
                    print(f'compiled {kind} R={R} Q={Q} in {time.time() - t0:.0f} s', flush=True)
                out = compiled[key](*args)
                for name, a, b in zip(['x'] + snames, out, ref):
                    a, b = a.float().cpu(), b.float()
                    if name == 'x':
                        a, b = a - x, b - x          # the block's contribution
                    rel = float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-30))
                    worst = max(worst, rel)
                    print(f'layer {l} {kind:5s} R={R} Q={Q} pos {start}: {name:7s} rel L2 {rel:.2e}', flush=True)
                states = {n: v for n, v in zip(snames, ref[1:])}   # continue from the CPU state
    print('worst', worst)
    if worst > 3e-2:
        raise SystemExit('device mismatch')
    print('GLM sub-block graphs on the device match the CPU mirror')


if __name__ == '__main__':
    main()
