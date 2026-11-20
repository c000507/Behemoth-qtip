"""CPU checks of the speculative-decoding graphs (EXACT f32, tiny random configs):
 1. verify bucket (VERIFY=1 segment, R = 1, Q tokens of one request): outputs == the same tokens as Q single decode steps,
    and the gdn state written to slot vslot[i] == the state after decode step i (kstate and conv state, every layer),
    with trailing padding tokens and with vslot[0] = the input slot (in place).
 2. mtp_block per rank (TP = 2, ranks as threads, qwen35_prepare.pack_mtp shards) == qwen35_moe_ref.Mtp: a first pass
    over several (hidden, next token) entries with a windowed table (lo > 0), then chained passes on its own output.
  python neuron/tests/test_qwen35_mtp_graph.py"""
import os
import sys
import threading
from pathlib import Path

os.environ['EXACT'] = '1'
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_graphs as Q  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
import qwen35_prepare as P  # noqa: E402
from glm_runner import ThreadAllReduce  # noqa: E402

ident = lambda v: v
PAGES, SLOTS = 9, 8


def seg_inputs(g, sh, states):
    t = {}
    for l in range(g.L):
        t.update({f'l{l}.a.' + k[5:]: v for k, v in sh[l].items() if k.startswith('attn.')})
        t.update({f'l{l}.m.' + k[4:]: v for k, v in sh[l].items() if k.startswith('mlp.')})
        t.update({f'l{l}.{k}': v for k, v in states[l].items()})
    return t


def test_verify():
    cfg = dict(R.TINY)
    g = Q.Geo(cfg, 1)
    w = R.random_weights(cfg, seed=2)
    get = P.Getter(w)
    seg = [g.kind(l) for l in range(g.L)]
    sh = [P.shard_layer(get, l, 0, g, False) for l in range(g.L)]
    torch.manual_seed(0)
    for Qv, n, start in ((4, 4, 21), (5, 3, 14), (2, 2, 0)):          # bucket Q, real tokens, first position (0: fresh sequence)
        st0 = [{k: torch.randn(v[0]) * 0.1 for k, v in Q.state_shapes(seg[l], g, PAGES, SLOTS).items()} for l in range(g.L)]
        x = torch.randn(Qv, g.H)
        table = torch.tensor([[3, 5, 1]])
        slot = torch.tensor([2])
        # reference: n single decode steps on a copy of the states, snapshot of the slot after each
        st = [{k: v.clone() for k, v in s.items()} for s in st0]
        xs, snaps = [], []
        for i in range(n):
            pos = torch.tensor([start + i])
            step = {'x': x[i:i + 1], 'pos': pos, 'slot': slot, 'table': table, 'wslot': table[0, pos // 16] * 16 + pos % 16,
                    'cos_sin': Q.cos_sin_rows(pos, g.rot, g.theta)}
            xo, new = Q.segment(seg, {**step, **seg_inputs(g, sh, st)}, g, 1, 1, 3, ident, 'gather')
            for k, v in new.items():
                l, name = int(k[1:k.index('.')]), k[k.index('.') + 1:]
                st[l][name] = v
            xs.append(xo)
            snaps.append([{k: v[2].clone() for k, v in st[l].items() if k != 'kv'} for l in range(g.L)])
        # verify chunk: positions start.., padding after n; state after token i -> vslot[i]
        Q.VERIFY = True
        pos = torch.tensor([start + i if i < n else -1 for i in range(Qv)])
        vslot = torch.tensor([2, 5, 0, 6, 4][:n] + [SLOTS - 1] * (Qv - n))
        wslot = torch.stack([table[0, p // 16] * 16 + p % 16 if p >= 0 else torch.tensor((PAGES - 1) * 16) for p in pos])
        step = {'x': x, 'pos': pos, 'slot': slot, 'vslot': vslot, 'table': table, 'wslot': wslot, 'cos_sin': Q.cos_sin_rows(pos.clamp(min=0), g.rot, g.theta)}
        stv = [{k: v.clone() for k, v in s.items()} for s in st0]
        xo, new = Q.segment(seg, {**step, **seg_inputs(g, sh, stv)}, g, 1, Qv, 3, ident, 'gather')
        Q.VERIFY = False
        dx = max(float((xo[i] - xs[i][0]).abs().max()) for i in range(n))
        ds = 0.0
        for l in range(g.L):
            if seg[l] != 'gdn':
                continue
            for i in range(n):
                for name in ('kstate', 'cstate'):
                    ds = max(ds, float((new[f'l{l}.{name}'][vslot[i]] - snaps[i][l][name]).abs().max()))
            # untouched slots stay as they were
            for sl in range(SLOTS - 1):
                if sl not in vslot.tolist():
                    assert torch.equal(new[f'l{l}.kstate'][sl], st0[l]['kstate'][sl])
        scale = float(torch.stack(xs).abs().max())
        print(f'verify Q={Qv} ({n} tokens from position {start}): outputs max |diff| {dx:.2e} (scale {scale:.2f}), per-position states max |diff| {ds:.2e}')
        assert dx < 2e-5 * max(1.0, scale) and ds < 2e-5, (dx, ds)


def test_mtp():
    cfg = dict(R.TINY)
    tp = 2
    g = Q.Geo(cfg, tp)
    w = R.random_weights(cfg, seed=1, prefix='model.')
    wm = R.random_mtp_weights(cfg, seed=3, experts=12)
    gm, ranks = P.pack_mtp(P.MtpGetter(wm), g, False)
    heads = [P.shard_head(P.Getter(w), r, g, False) for r in range(tp)]
    ref = R.Mtp(cfg, wm)
    embed, lm = w['model.embed_tokens.weight'], w['lm_head.weight']
    fnorm = 1.0 + w['model.norm.weight']
    torch.manual_seed(1)
    kv = [Q.zeros(*Q.state_shapes('gqa', gm, PAGES, SLOTS)['kv']) + 7.0 for _ in range(tp)]     # stale garbage everywhere: lo must hide it
    table_all = [4, 2, 6, 0]                                                                      # the request's pages
    first, P_ = 1, 2                                                                              # window: table pages 1.. (keys from position 16)
    lo_abs = 21                                                                                   # first MTP entry of the sequence

    def launch(x, nxt, pos, hnorm_of, sel):
        Qm = x.shape[0]
        n = len(pos)
        posr = torch.tensor([p - 16 * first for p in pos] + [-1] * (Qm - n))
        wslot = torch.tensor([table_all[p // 16] * 16 + p % 16 for p in pos] + [(PAGES - 1) * 16] * (Qm - n))
        e = torch.zeros(Qm, g.H)
        e[:n] = embed[torch.tensor(nxt)]
        step = {'x': x, 'e': e, 'pos': posr, 'cos_sin': Q.cos_sin_rows(torch.tensor(list(pos) + [0] * (Qm - n)), g.rot, g.theta), 'wslot': wslot,
                'table': torch.tensor([table_all[first:first + P_]]), 'lo': torch.tensor([lo_abs - 16 * first]), 'sel': torch.tensor([sel])}
        ar = ThreadAllReduce(tp)
        out = [None] * tp

        def run(r):
            t = dict(step)
            t.update({'a.' + k[5:]: v for k, v in ranks[r].items() if k.startswith('attn.')})
            t.update({'m.' + k[4:]: v for k, v in ranks[r].items() if k.startswith('mlp.')})
            t.update({k: ranks[r][k] for k in ('en', 'hn', 'fcols', 'fc')})
            t.update(hnorm=hnorm_of(r), mnorm=ranks[r]['norm'], w=heads[r]['w'], kv=kv[r])
            z, logits, st = Q.mtp_block(t, g, gm, Qm, P_, ar(r))
            kv[r] = st['kv']
            out[r] = (z, logits)
        th = [threading.Thread(target=run, args=(r,)) for r in range(tp)]
        [t.start() for t in th]
        [t.join() for t in th]
        assert torch.equal(out[0][0], out[1][0])
        return out[0][0], torch.cat([o[1] for o in out], dim=-1)[:, :g.V]

    def launch_chain(kv0, x, nxt, pos, sel, D):
        """mtp_chain_block (D drafts in one launch) from the cache state kv0 -> (drafts, the caches after it)."""
        Qm, n = x.shape[0], len(pos)
        cp = [pos[-1] + j for j in range(1, D)]
        VS = g.VP // tp
        e = torch.zeros(Qm, g.H)
        e[:n] = embed[torch.tensor(nxt)]
        step = {'x': x, 'e': e, 'pos': torch.tensor([p - 16 * first for p in pos] + [-1] * (Qm - n)),
                'cos_sin': Q.cos_sin_rows(torch.tensor(list(pos) + [0] * (Qm - n)), g.rot, g.theta),
                'wslot': torch.tensor([table_all[p // 16] * 16 + p % 16 for p in pos] + [(PAGES - 1) * 16] * (Qm - n)),
                'table': torch.tensor([table_all[first:first + P_]]), 'lo': torch.tensor([lo_abs - 16 * first]), 'sel': torch.tensor([sel]),
                'cpos': torch.tensor([p - 16 * first for p in cp]), 'ccos': Q.cos_sin_rows(torch.tensor(cp), g.rot, g.theta),
                'cwslot': torch.tensor([table_all[p // 16] * 16 + p % 16 for p in cp])}
        ar = ThreadAllReduce(tp)
        out, kvs = [None] * tp, [None] * tp

        def run(r):
            t = dict(step)
            t.update({'a.' + k[5:]: v for k, v in ranks[r].items() if k.startswith('attn.')})
            t.update({'m.' + k[4:]: v for k, v in ranks[r].items() if k.startswith('mlp.')})
            t.update({k: ranks[r][k] for k in ('en', 'hn', 'fcols', 'fc')})
            emb = torch.zeros(VS, g.H)
            rows = embed[r * VS:(r + 1) * VS]
            emb[:rows.shape[0]] = rows
            vbias = torch.zeros(VS)
            vbias[max(0, g.V - r * VS):] = -3e38
            t.update(hnorm=fnorm, mnorm=ranks[r]['norm'], w=heads[r]['w'], kv=kv0[r], emb=emb, vbias=vbias,
                     rhot=torch.nn.functional.one_hot(torch.tensor(r), tp).float(), rbase=torch.tensor([float(r * VS)]))
            out[r], st_ = Q.mtp_chain_block(t, g, gm, Qm, P_, D, ar(r))
            kvs[r] = st_['kv']
        th = [threading.Thread(target=run, args=(r,)) for r in range(tp)]
        [t.start() for t in th]
        [t.join() for t in th]
        assert torch.equal(out[0], out[1])
        return [int(v) for v in out[0][0]], kvs

    st = ref.init_state()
    worst = 0.0

    def check(a, b, what):
        nonlocal worst
        rel = float((a - b).abs().max() / b.abs().max())
        worst = max(worst, rel)
        print(f'  {what}: rel max |diff| {rel:.2e}')
        assert rel < 5e-5, (what, rel)
    toks = torch.randint(0, cfg['vocab_size'], (40,))
    p = lo_abs
    for n_entries, Qm in ((3, 5), (1, 1), (2, 5), (4, 5)):                          # accepted + 1 entries per round, bucket Q
        x = torch.randn(Qm, g.H)                                                     # the main model's final residual rows
        pos = list(range(p, p + n_entries))
        nxt = toks[p + 1:p + 1 + n_entries].tolist()
        h = R.rms(x[:n_entries], w['model.norm.weight'], g.eps)
        o = ref.step(h, embed[torch.tensor(nxt)], [q - lo_abs for q in pos], st)     # the reference numbers its entries from 0: RoPE is relative
        chains = []
        for mode in ('max', 'eq'):
            Q.MTP_ARGMAX = mode
            chain, chain_kv = launch_chain([c.clone() for c in kv], x, nxt, pos, n_entries - 1, 4)
            chains.append(chain)
        assert chains[0] == chains[1]
        z, logits = launch(x, nxt, pos, lambda r: fnorm, n_entries - 1)
        host = [int(logits.argmax(-1))]
        check(logits[0], o[-1] @ lm.T, f'first pass, {n_entries} entries at {pos[0]}.. (bucket Q={Qm}): logits')
        # chain 3 drafts on copies of the reference state; the graph writes the chained entries into the cache (overwritten later)
        s2 = {k: v.clone() for k, v in st.items()}
        oc = o[-1:]
        for j in range(1, 4):
            d = int(logits.argmax(-1))
            assert d == int((oc @ lm.T).argmax(-1))
            oc = ref.step(oc, embed[d].reshape(1, -1), [pos[-1] + j - lo_abs], s2)
            z, logits = launch(z, [d], [pos[-1] + j], lambda r: ranks[r]['norm'], 0)
            check(logits[0], oc[0] @ lm.T, f'  chain {j} at {pos[-1] + j}')
            host.append(int(logits.argmax(-1)))
        assert chain == host, (chain, host)                                         # the one-launch chain drafts what the host-chained launches draft
        assert all(torch.equal(a, b) for a, b in zip(chain_kv, kv))                 # and leaves the same cache
        print(f'  one-launch chain (D=4): drafts {chain} == host-chained')
        p += n_entries
    print(f'mtp_block (TP={tp}, window from page {first}, lo {lo_abs}) == qwen35_moe_ref.Mtp, worst rel {worst:.2e}')


if __name__ == '__main__':
    test_verify()
    test_mtp()
    print('OK')
