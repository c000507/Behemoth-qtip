"""Host-side driver of the GLM graphs: the runtime's step logic (embedding, page tables, KDA state slots, sub-block
sequence, head) in Python, executing the graph functions of glm_graphs.py on the CPU with TP ranks emulated by threads
(the all-reduce is a barrier + sum). It is the reference for cpp/glm_model.cpp and the CPU check of the sharding.

    r = CpuRunner(g, packer_or_prepared, pages=..., slots=...)
    logits = r.step([(seq, tokens, start)])     # each entry: tokens of one sequence starting at absolute position start
"""
import threading

import torch

import glm_graphs as G


class ThreadAllReduce:
    def __init__(self, n):
        self.n, self.barrier, self.parts, self.lock = n, threading.Barrier(n), {}, threading.Lock()

    def __call__(self, rank):
        def reduce(v):
            with self.lock:
                self.parts[rank] = v
            self.barrier.wait()
            total = sum(self.parts[r] for r in range(self.n))
            self.barrier.wait()
            return total
        return reduce


class CpuRunner:
    def __init__(self, g, layers, heads, embed, pages=65, slots=9, kv8=False, moe_mode='all', max_pages_per_seq=8):
        """layers[rank][l]: {'attn.*', 'mlp.*'} tensors (glm_repack.Packer.layer); heads[rank]: head tensors."""
        self.g, self.layers, self.heads, self.embed = g, layers, heads, embed
        self.pages, self.slots, self.kv8, self.moe_mode, self.mpp = pages, slots, kv8, moe_mode, max_pages_per_seq
        self.state = []
        for r in range(g.TP):
            per = []
            for l in range(g.L):
                kind = 'kda' if g.attn[l] == 'linear_attention' else 'mla'
                per.append({k: G.zeros(*v) for k, v in G.state_shapes(kind, g, pages, slots, kv8).items()})
            self.state.append(per)

    def table(self, seq):
        return [seq * self.mpp + i for i in range(self.mpp)]

    def step(self, entries, P=None, Q=None):
        """entries: [(seq, tokens, start)] -> logits [len(entries) * Q, vocab] (every row). Q (default: the longest entry)
        pads shorter entries with position -1 tokens, as the runtime pads a prefill chunk to its bucket."""
        g = self.g
        R, Q = len(entries), Q or max(len(e[1]) for e in entries)
        P = P or self.mpp
        T = R * Q
        x = torch.zeros(T, g.S * g.H)
        pos = torch.zeros(T, dtype=torch.int64)
        wslot = torch.zeros(T, dtype=torch.int64)
        table = torch.zeros(R, P, dtype=torch.int64)
        slot = torch.zeros(R, dtype=torch.int64)
        pos -= 1
        wslot += (self.pages - 1) * G.PAGE
        for r, (seq, toks, start) in enumerate(entries):
            tab = self.table(seq)
            table[r] = torch.tensor(tab[:P])
            slot[r] = seq
            for q, tok in enumerate(toks):
                i = r * Q + q
                e = G.bf(self.embed[tok].float())
                x[i] = e.repeat(g.S)
                pos[i] = start + q
                wslot[i] = tab[(start + q) // G.PAGE] * G.PAGE + (start + q) % G.PAGE
        ar = ThreadAllReduce(g.TP)
        out = [None] * g.TP

        def rank_fn(rank):
            xr = x.clone()
            red = ar(rank)
            for l in range(g.L):
                lw = self.layers[rank][l]
                for sub in ('attn', 'mlp'):
                    kind = ('kda' if g.attn[l] == 'linear_attention' else 'mla') if sub == 'attn' else g.mlp[l]
                    t = {k[len(sub) + 1:]: v for k, v in lw.items() if k.startswith(sub + '.')}
                    t.update(x=xr, pos=pos, wslot=wslot, table=table, slot=slot)
                    if sub == 'attn':
                        t.update(self.state[rank][l])
                    xr, st = G.sublayer(kind, t, g, R, Q, P, red, self.kv8, self.moe_mode)
                    if sub == 'attn':
                        self.state[rank][l].update(st)
            h = dict(self.heads[rank], x=xr)
            out[rank] = G.head(h, g, torch.arange(T))

        errors = []

        def guarded(r):
            try:
                rank_fn(r)
            except BaseException as e:  # noqa: BLE001 - re-raised below
                errors.append(e)
                ar.barrier.abort()
        threads = [threading.Thread(target=guarded, args=(r,)) for r in range(g.TP)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if errors:
            raise errors[0]
        return torch.cat(out, dim=1)[:, :g.V]
