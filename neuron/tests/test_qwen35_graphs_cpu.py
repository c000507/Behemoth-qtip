"""CPU mirror of the Qwen3.5 MoE graphs (qwen35_graphs.sublayer per rank, ranks as threads, qwen35_prepare.shard_layer
sharding) against the f32 oracle qwen35_moe_ref.Ref on a tiny random config: a prefill chunk (R = 1, Q tokens, with
padding), then decode steps (R rows: the same sequence plus a second one), states carried in the graphs' slot / page layout.

  EXACT=1 python neuron/tests/test_qwen35_graphs_cpu.py            # f32 everywhere: sharding / state logic must match to ~1e-5
  WEIGHTS=bf16 python neuron/tests/test_qwen35_graphs_cpu.py       # bf16 rounding as the device graphs
  WEIGHTS=fp8 python neuron/tests/test_qwen35_graphs_cpu.py        # fp8 weights + activations (exact E4M3 mirror)
  MOE=gather ...                                                   # decode steps through the gathered-expert path
"""
import os
import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_graphs as Q  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
import qwen35_prepare as P  # noqa: E402
from glm_runner import ThreadAllReduce  # noqa: E402

FP8 = os.environ.get('WEIGHTS', 'bf16') == 'fp8'
MOE_DEC = os.environ.get('MOE', 'all')


class Mirror:
    """The graph functions on the CPU for TP ranks, one shared page pool / slot set, states per rank and layer."""

    def __init__(self, cfg, weights, tp=2, pages=9, slots=4):
        self.g = g = Q.Geo(cfg, tp)
        get = P.Getter(weights)
        self.layers = [[P.shard_layer(get, l, r, g, FP8) for l in range(g.L)] for r in range(tp)]
        self.heads = [P.shard_head(get, r, g, FP8) for r in range(tp)]
        self.embed = get('embed_tokens.weight')
        self.pages, self.slots = pages, slots
        self.state = [[{k: Q.zeros(*v) for k, v in Q.state_shapes(g.kind(l), g, pages, slots).items()} for l in range(g.L)] for r in range(tp)]

    def launch(self, tokens, pos, slot, table, moe_mode):
        """tokens / pos [R, Q] (pos -1 = padding), slot [R], table [R, P] -> logits of every row's last token [R, V]."""
        g = self.g
        R_, Q_ = tokens.shape
        T = R_ * Q_
        x = self.embed[tokens.clamp(min=0).reshape(T)].float()
        posf = pos.reshape(T)
        wslot = torch.stack([table[i // Q_, int(p) // 16] * 16 + p % 16 if p >= 0 else torch.tensor((self.pages - 1) * 16) for i, p in enumerate(posf)])
        step = {'x': x, 'pos': posf, 'slot': slot, 'table': table, 'wslot': wslot, 'cos_sin': Q.cos_sin_rows(posf.clamp(min=0), g.rot, g.theta)}
        ar = ThreadAllReduce(g.TP)
        out = [None] * g.TP

        def run(r):
            xr = x
            for l in range(g.L):
                kind = g.kind(l)
                t = {**step, 'x': xr, **{k[5:]: v for k, v in self.layers[r][l].items() if k.startswith('attn.')}, **self.state[r][l]}
                xr, st = Q.sublayer(kind, t, g, R_, Q_, table.shape[1], ar(r))
                self.state[r][l].update(st)
                t = {**step, 'x': xr, **{k[4:]: v for k, v in self.layers[r][l].items() if k.startswith('mlp.')}}
                xr, _ = Q.sublayer('moe', t, g, R_, Q_, table.shape[1], ar(r), moe_mode)
            rows = torch.arange(R_) * Q_ + (pos >= 0).long().sum(dim=1) - 1
            out[r] = (xr, Q.head({'x': xr, **self.heads[r]}, g, rows))
        th = [threading.Thread(target=run, args=(r,)) for r in range(g.TP)]
        [t.start() for t in th]
        [t.join() for t in th]
        hidden = out[0][0]
        logits = torch.cat([out[r][1][:, :g.VP // g.TP] for r in range(g.TP)], dim=1)[:, :g.V]
        return hidden, logits


def rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def main():
    torch.manual_seed(0)
    cfg = dict(R.TINY)
    w = R.random_weights(cfg, seed=3)
    ref = R.Ref(cfg, w)
    mir = Mirror(cfg, w)
    V = cfg['vocab_size']
    worst = 0.0
    # sequence A: prefill 6 real tokens in a Q = 8 chunk (2 padding rows), pages 0-1, slot 0
    toksA = torch.randint(0, V, (6,))
    stA = ref.init_state()
    lg_ref, h_ref = ref.forward(toksA, stA, return_hidden=True)
    tokens = torch.cat((toksA, torch.zeros(2, dtype=torch.int64))).reshape(1, 8)
    pos = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]])
    h, lg = mir.launch(tokens, pos, torch.tensor([0]), torch.tensor([[0, 1]]), 'all')
    hn = Q.rms(h[:6], mir.heads[0]['norm'], mir.g.eps)
    e1, e2 = rel(hn, h_ref), rel(lg[0], lg_ref[-1])
    print(f'prefill chunk (6 of 8): final hidden rel L2 {e1:.3e}, last logits {e2:.3e}, top-1 {"same" if lg[0].argmax() == lg_ref[-1].argmax() else "DIFFERENT"}')
    worst = max(worst, e1, e2)
    # sequence B starts with a decode row at position 0 (state zeroed by the graph), pages 2-3, slot 1
    toksB = torch.randint(0, V, (4,))
    stB = ref.init_state()
    # decode steps: rows = [A continues, B from scratch]
    for i in range(4):
        tA = torch.randint(0, V, (1,))
        lA = ref.forward(tA, stA)
        lB = ref.forward(toksB[i:i + 1], stB)
        tokens = torch.stack((tA, toksB[i:i + 1]))
        pos = torch.tensor([[6 + i], [i]])
        h, lg = mir.launch(tokens, pos, torch.tensor([0, 1]), torch.tensor([[0, 1], [2, 3]]), MOE_DEC)
        eA, eB = rel(lg[0], lA[0]), rel(lg[1], lB[0])
        print(f'decode step {i}: logits rel L2 A {eA:.3e} B {eB:.3e}; top-1 A {"same" if lg[0].argmax() == lA[0].argmax() else "DIFF"} B {"same" if lg[1].argmax() == lB[0].argmax() else "DIFF"}')
        worst = max(worst, eA, eB)
    # states: the mirror's slot 0 vs the oracle's sequence A state (rank 0's heads)
    g = mir.g
    for l in range(g.L):
        if g.kind(l) == 'gdn':
            kd = g.Hk * g.Dk
            rows = torch.cat((torch.arange(g.hk * g.Dk), kd + torch.arange(g.hk * g.Dk), 2 * kd + torch.arange(g.Wv)))   # rank 0's channels
            dS = (mir.state[0][l]['kstate'][0] - stA[l]['S'][:g.hl]).abs().max().item()
            dC = (mir.state[0][l]['cstate'][0] - stA[l]['conv'][rows]).abs().max().item()
            print(f'layer {l} gdn state slot 0 vs oracle: recurrent {dS:.2e} conv {dC:.2e}')
            if os.environ.get('EXACT') == '1':
                assert dS < 1e-4 and dC < 1e-4
    mode = 'EXACT' if os.environ.get('EXACT') == '1' else ('fp8' if FP8 else 'bf16')
    print(f'[{mode}, decode MOE={MOE_DEC}] worst rel L2 {worst:.3e}')
    if os.environ.get('EXACT') == '1':
        assert worst < 1e-4, worst


if __name__ == '__main__':
    main()
