"""CPU mirror of the Inkling graphs (inkling_graphs.visit / segment / head per rank, ranks as threads, inkling_prepare
sharding of a tiny random ORIGINAL-format checkpoint) against the f32 oracle inkling_ref.Ref:
  sequence A prefilled in chunks of Q tokens (the last one padded), sequence B in one padded chunk, then decode steps
  with R = 2 rows (A and B), the bound per-row buf / conv states and the shared page pool carried as the engine would.

  EXACT=1 python neuron/tests/test_inkling_graphs_cpu.py          # f32 everywhere: must match the oracle to ~1e-5
  WEIGHTS=bf16 / WEIGHTS=fp8 ...                                   # the device numerics (bf16 rounding / fp8)
  MOE=gather (decode experts gathered), WT=1 (transposed layout), SEG=1 (decode through 6-visit segments)
"""
import os
import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inkling_graphs as IG  # noqa: E402
import inkling_prepare as PR  # noqa: E402
import inkling_ref as R  # noqa: E402
from glm_runner import ThreadAllReduce  # noqa: E402
from test_inkling_ref import TINY  # noqa: E402

FP8 = os.environ.get('WEIGHTS', 'bf16') == 'fp8'
MOE_DEC = os.environ.get('MOE', 'all')
SEG = os.environ.get('SEG', '0') == '1'
SPLIT = os.environ.get('SPLIT', '')   # visit kinds run as attention + MoE graphs (e.g. gG)
PAGE = IG.PAGE
PREFILL_MOE = os.environ.get('PREFILL_MOE', 'all')   # group: the split visits' MoE part of prefill chunks grouped (MOE_CAP / MOE_HOT)
STAT = {'grouped': 0, 'fallback': 0}
UPPER = os.environ.get('UPPER', '0') == '1'   # prefill visits through the loop-pass graph kinds (S / G), zero gates on first-pass visits


class Mirror:
    def __init__(self, cfg, ck, tp=2, pages=9, slots=3):
        self.g = g = IG.Geo(cfg, tp)
        get = PR.Source(ck)
        self.layers = []
        for r in range(tp):
            self.layers.append([])
        for l in range(g.L):
            ex = PR.expert_tensors(get, l, g, FP8)
            for r in range(tp):
                self.layers[r].append(PR.shard_layer(get, l, r, g, FP8, experts=ex[r]))
        self.heads = [PR.shard_head(get, r, g, FP8) for r in range(tp)]
        if IG.WT:
            self.layers = [[{k: IG.to_wt(k, v) for k, v in lr.items()} for lr in rl] for rl in self.layers]
            self.heads = [{k: IG.to_wt(k, v) for k, v in h.items()} for h in self.heads]
        self.embed = PR.normed_embedding(get, g).float()
        self.pages = pages
        nv = len(g.visits)
        self.wsh = {k: IG.weight_shapes(k, g, FP8) for k in 'sSgG'}
        # bound per-sequence state (slot = sequence id), per rank and visit; the page pool per rank and global visit
        self.bound = [[[{k: IG.zeros(*v) for k, v in IG.state_shapes(g.kind(v_), g, 1, pages).items() if k != 'kv'}
                        for v_ in range(nv)] for _ in range(slots)] for _ in range(tp)]
        self.pool = [[IG.zeros(*IG.state_shapes(g.kind(v_), g, 1, pages)['kv']) if g.kind(v_) in 'gG' else None
                      for v_ in range(nv)] for _ in range(tp)]

    def inputs(self, r, v, rows, as_kind=None):
        """as_kind: the graph kind run for the visit (the engine runs first-pass visits of prefill buckets through the
        loop-pass graphs S / G with ZERO message gates)."""
        g = self.g
        kind = as_kind or g.kind(v)
        l = g.visits[v][0]
        gate = lambda k: k == 'mg' or k[:3] in ('gd_', 'gf_', 'gg_')
        w = {k: (IG.zeros(*self.wsh[kind][k]) if gate(k) and kind != g.kind(v) else self.layers[r][l][k]) for k in self.wsh[kind] if k != 'ctl'}
        w['ctl'] = torch.tensor(g.ctl(v), dtype=torch.float32)
        st = {}
        R_ = len(rows)
        for i, seq in enumerate(rows):
            for k, t in self.bound[r][seq][v].items():
                st[k if R_ == 1 else f'{k}_{i}'] = t
        if kind in 'gG':
            st['kv'] = self.pool[r][v]
        return w, st

    def store(self, r, v, rows, new):
        R_ = len(rows)
        for i, seq in enumerate(rows):
            for k in ('buf', 'conv'):
                self.bound[r][seq][v][k] = new[k if R_ == 1 else f'{k}_{i}']
        if 'kv' in new:
            self.pool[r][v] = new['kv']

    def launch(self, tokens, pos, rows, table, moe_mode):
        """tokens / pos [R, Q] (pos -1 = padding), rows = sequence ids [R], table [R, P] -> logits of the last real
        token of each row [R, Vu]."""
        g = self.g
        R_, Q_ = tokens.shape
        T = R_ * Q_
        e = self.embed[tokens.clamp(min=0).reshape(T)]
        x = torch.zeros(T, IG.NS, g.H)
        x[:, 0], x[:, 4] = e, e
        x = x.reshape(T, IG.NS * g.H)
        posf = pos.reshape(T)
        P = table.shape[1]
        wslot = torch.stack([table[i // Q_, int(p) // PAGE] * PAGE + p % PAGE if p >= 0 else torch.tensor((self.pages - 1) * PAGE)
                             for i, p in enumerate(posf)])
        step = {'pos': posf, 'wslot': wslot, 'table': table}
        last = torch.tensor([i * Q_ + int((pos[i] >= 0).sum()) - 1 for i in range(R_)])
        ar = ThreadAllReduce(g.TP)
        out = [None] * g.TP
        errs = []

        def run(r):
            try:
                xr = x
                nv = len(g.visits)
                v = 0
                while v < nv:
                    if SEG and Q_ == 1 and v % 6 == 0 and v + 6 <= nv:
                        kinds = [g.kind(v + i) for i in range(6)]
                        t = dict(step, x=xr)
                        for i in range(6):
                            w, st = self.inputs(r, v + i, rows)
                            t.update({f'l{i}.{k}': val for k, val in {**w, **st}.items()})
                        xr, new = IG.segment(kinds, t, g, R_, Q_, P, ar(r), moe_mode)
                        xr = xr.reshape(T, -1)
                        for i in range(6):
                            pre = f'l{i}.'
                            self.store(r, v + i, rows, {k[len(pre):]: val for k, val in new.items() if k.startswith(pre)})
                        v += 6
                        continue
                    k = g.kind(v).upper() if UPPER and Q_ > 1 else g.kind(v)
                    w, st = self.inputs(r, v, rows, k)
                    if SPLIT and g.kind(v) in SPLIT:   # attention NEFF then MoE NEFF (only their own inputs)
                        wa = {n: w[n] for n in IG.part_names(k, 'attn', list(w))}
                        xr, new = IG.visit(k, {**step, **wa, **st, 'x': xr}, g, R_, Q_, P, ar(r), moe_mode, 'attn')
                        self.store(r, v, rows, new)
                        w, st = self.inputs(r, v, rows, k)
                        wm = {n: w[n] for n in IG.part_names(k, 'moe', list(w))}
                        stm = {n: x_ for n, x_ in st.items() if n.startswith('conv')}
                        tm = {'pos': step['pos'], **wm, **stm, 'x': xr.reshape(T, -1)}
                        mode = 'group' if PREFILL_MOE == 'group' and Q_ > 1 else moe_mode
                        xo, new = IG.visit(k, tm, g, R_, Q_, P, ar(r), mode, 'moe')
                        if 'ovf' in new:   # the engine's rule: an overflowed grouped launch is rerun with MOE=all
                            over = float(new.pop('ovf')[0]) > IG.Q.MOE_CAP
                            if r == 0:
                                STAT['grouped'] += 1
                                STAT['fallback'] += int(over)
                            if over:
                                assert all(torch.equal(new[n], stm[n]) for n in stm), 'overflowed launch changed the state'
                                xo, new = IG.visit(k, tm, g, R_, Q_, P, ar(r), 'all', 'moe')
                        xr = xo
                        new = {**{n: x_ for n, x_ in st.items() if not n.startswith('conv')}, **new}
                    else:
                        xr, new = IG.visit(k, {**step, **w, **st, 'x': xr}, g, R_, Q_, P, ar(r), moe_mode)
                    xr = xr.reshape(T, -1)
                    self.store(r, v, rows, new)
                    v += 1
                out[r] = IG.head({**self.heads[r], 'x': xr}, g, last)
            except Exception as ex:  # noqa: BLE001
                errs.append(ex)
                raise
        th = [threading.Thread(target=run, args=(r,)) for r in range(g.TP)]
        [t_.start() for t_ in th]
        [t_.join() for t_ in th]
        if errs:
            raise errs[0]
        return torch.cat(out, dim=1)[:, :g.Vu]


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    torch.manual_seed(0)
    cfg = R.Cfg(TINY)
    ck = R.random_checkpoint(cfg, seed=1, mxfp8_layers=[3, 20])
    ref = R.from_tensors(cfg, ck)
    m = Mirror(TINY, ck)
    Qc = 8
    A = torch.randint(2, cfg.Vu, (21,))
    B = torch.randint(2, cfg.Vu, (6,))
    tabA, tabB = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])
    stA, stB = {}, {}
    worst, top = 0.0, []
    # prefill A in chunks of Qc (the last padded), B in one padded chunk
    for c0 in range(0, len(A), Qc):
        ids = A[c0:c0 + Qc]
        n = len(ids)
        tok = torch.full((1, Qc), 0, dtype=torch.long)
        pos = torch.full((1, Qc), -1, dtype=torch.long)
        tok[0, :n], pos[0, :n] = ids, torch.arange(c0, c0 + n)
        lm = m.launch(tok, pos, [0], tabA, 'all')
        lr = ref.forward(ids, stA)[-1:]
        e = rel(lm, lr)
        worst = max(worst, e)
        top.append(int(lm.argmax(-1) == lr.argmax(-1)))
        print(f'prefill A chunk @{c0} ({n} tokens): rel {e:.3e}')
    tok = torch.zeros((1, Qc), dtype=torch.long)
    pos = torch.full((1, Qc), -1, dtype=torch.long)
    tok[0, :len(B)], pos[0, :len(B)] = B, torch.arange(len(B))
    lm = m.launch(tok, pos, [1], tabB, 'all')
    lr = ref.forward(B, stB)[-1:]
    e = rel(lm, lr)
    worst = max(worst, e)
    print(f'prefill B ({len(B)} tokens): rel {e:.3e}')
    na, nb = len(A), len(B)
    ta, tb = int(lm.argmax(-1)), 0
    ta = int(ref.forward(A[-1:], {k: (dict(v) if isinstance(v, dict) else v) for k, v in stA.items()}).argmax(-1)[-1]) if False else int(A[-1])
    nxtA, nxtB = int(torch.randint(2, cfg.Vu, (1,))), int(torch.randint(2, cfg.Vu, (1,)))
    for s in range(6):
        tok = torch.tensor([[nxtA], [nxtB]])
        pos = torch.tensor([[na], [nb]])
        table = torch.cat([tabA, tabB], 0)
        lm = m.launch(tok, pos, [0, 1], table, MOE_DEC)
        la = ref.forward(torch.tensor([nxtA]), stA)[-1]
        lb = ref.forward(torch.tensor([nxtB]), stB)[-1]
        e = max(rel(lm[0], la), rel(lm[1], lb))
        worst = max(worst, e)
        top += [int(lm[0].argmax() == la.argmax()), int(lm[1].argmax() == lb.argmax())]
        print(f'decode step {s} (pos {na} / {nb}): rel {e:.3e}')
        na, nb = na + 1, nb + 1
        nxtA, nxtB = int(la.argmax()), int(lb.argmax())
    # a new sequence C reusing A's slot (state zeroed by the graph at position 0)
    C = torch.randint(2, cfg.Vu, (5,))
    tok = torch.zeros((1, Qc), dtype=torch.long)
    pos = torch.full((1, Qc), -1, dtype=torch.long)
    tok[0, :len(C)], pos[0, :len(C)] = C, torch.arange(len(C))
    lm = m.launch(tok, pos, [0], tabA, 'all')
    lr = ref.forward(C, {})[-1:]
    e = rel(lm, lr)
    worst = max(worst, e)
    print(f'prefill C in A\'s slot ({len(C)} tokens): rel {e:.3e}')
    print(f'WORST rel {worst:.3e}  top1 {sum(top)}/{len(top)}' + (f'  grouped MoE launches {STAT["grouped"]}, fallbacks {STAT["fallback"]}' if STAT['grouped'] else ''))
    if IG.EXACT:
        assert worst < 1e-4, worst


if __name__ == '__main__':
    main()
