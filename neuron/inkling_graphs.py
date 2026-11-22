"""Inkling 2 Mini (inkling2_loop, text) graphs for NeuronCores, TP ranks, the experts split along their intermediate axis.

Numerical oracle: inkling_ref.Ref (f32, validated against HF transformers). Projections are fp8 (row-scaled E4M3 x
E4M3 activations, the stack's recipe via qwen35_graphs.lin) or bf16; EXACT=1 keeps everything f32 (sharding / state
logic tests). WT=1: dense projections, router and head stored transposed [in, out] (decode lever, no on-chip weight
transposes; docs/neuron-reap-decode-profile.md), experts' gate|up [E, H, 2 EIl].

The model is run as 48 layer VISITS per token (layers 6..17 three times, shared weights). One graph = one visit
(attention + MoE fused) or a SEGMENT of consecutive visits; the runtime binds the layer's weights and the visit's state.
Visit kinds: s (sliding-window layer), g (global layer), S / G the same in loop passes 2 and 3 (message gates).
The per-visit control that differs between visits of one kind (which depth-attention blocks are visible, block boundary,
loop restart, first layer) is a small f32 input `ctl` [CTL] (Geo.ctl), so one NEFF per (kind, bucket) serves every visit.

Residual state x [T, NS = 7, H] f32 per token (replicated on every rank):
  0 embedding block, 1..3 completed blocks B1..B3, 4 running partial sum, 5 / 6 previous loop pass's B2 / final partial
  (the message sources). The host writes the normed embedding into slots 0 and 4 (others 0); the head graph reads all.

State per visit (bound per row: the engine binds the request's own one-slot tensors, no runtime-indexed state DMA;
docs/neuron-reap-nonfinite.md):
  buf  [NB, 2, kvl, D] bf16: K|V of the request's last NB tokens in LOGICAL order (NB = sliding window for s, rel_extent
       for g), shifted by the graph (static relative-position bias, no ring arithmetic). Entry i of a row whose first new
       token is at position p0 holds position p0 - NB + i.
  conv [CS, K-1] f32: the last K-1 inputs of the k | v | post-attention | post-MoE short convs (CS = 2 kvl D + 2 H).
  g / G also: kv [pages, 2, kvl, 16, D] (the server's 16-token pages): keys farther than rel_extent come from there
       (no position bias there; the near keys come from buf, so no runtime-offset bias is needed).
Step inputs: x, pos [T] (-1 = padding), g: wslot [T] (page * 16 + offset, scratch page for padding), table [R, P].

usage: inkling_graphs.py WORKDIR {s|S|g|G|seg_<kinds>} R Q P | head T S
  env: MODEL (dir with config.json) or CONFIG, TP, PAGES (incl. one scratch page), WEIGHTS=fp8|bf16, MOE=all|gather, WT,
       NCC_FLAGS
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_graphs as G  # noqa: E402
import inkling_ref as R  # noqa: E402
import qwen35_graphs as Q  # noqa: E402  (lin / lin_partial / quant8 / egu_mm / bf honour EXACT and WT)

F8 = G.F8
PAGE = G.PAGE
EXACT, WT = Q.EXACT, Q.WT
NS = 7
CTL = 17
ROUTE = os.environ.get('ROUTE', 'topk')
MOE_SLOTS = os.environ.get('MOE_SLOTS', 'topk')
MOE_TSUB = Q.MOE_TSUB
MOE_EG = int(os.environ.get('MOE_EG', '0'))       # MOE=all in groups of this many experts (0: one pass)   # gathered decode MoE slot selection: topk | onehot     # topk | pad64 (membership from a 64-row padded topk; see qwen35_graphs.route)
NEG = -3e38
ATT_BLOCK = int(os.environ.get('ATT_BLOCK', '0'))     # global visits: far keys in blocks of this many pages, online softmax (0: one block)
ATT_QBLOCK = int(os.environ.get('ATT_QBLOCK', '0'))   # wide prefill chunks: queries in blocks of this many tokens
SKEW = os.environ.get('SKEW', 'bank')                 # relative-bias skew: bank (per-query shifted profile bank, matmul) | banke | reshape | bits
FARK = os.environ.get('FARK', 'dk')                   # far keys materialised as [kvl, D, keys] (dk) or [kvl, keys, D] (kd)
ATT_BF16 = os.environ.get('ATT_BF16', '1') == '1'     # attention scores / values contracted in bf16 on the device (f32 accumulate)
PROF = set(os.environ.get('PROF', '').split(','))   # timing probes only (WRONG output): nofar, nodyn, nomsg, nomoe, nonear


class Geo:
    def __init__(self, cfg, tp=2):
        self.c = c = R.Cfg(cfg)
        self.raw = c.raw
        self.H, self.L, self.V, self.Vu = c.H, c.L, c.V, c.Vu
        self.heads, self.kvh, self.D = c.heads, c.kvh, c.D
        self.W, self.X = c.W, c.rel_extent
        self.d_rel, self.K, self.eps = c.d_rel, c.K, c.eps
        self.E, self.topk, self.I = c.E, c.topk, c.I
        self.visits = c.visits
        self.TP = tp
        bad = [n for n, v in (('heads', self.heads), ('kv heads', self.kvh), ('expert intermediate', self.I)) if v % tp]
        if bad:
            raise ValueError(f'geometry not shardable over TP={tp}: ' + ', '.join(bad))
        self.hq, self.kvl, self.EIl = self.heads // tp, self.kvh // tp, self.I // tp
        self.CK = self.kvl * self.D
        self.CS = 2 * self.CK + 2 * self.H
        self.VS = -(-self.Vu // tp)                    # head rows per rank (the last rank's tail zero-padded)
        eos = c.raw.get('eos_token_id', 0)
        self.eos = eos if isinstance(eos, list) else [eos]
        self.max_context = c.raw.get('model_max_length', c.raw.get('max_position_embeddings', 1 << 20))

    def nb(self, kind):
        return self.W if kind in 'sS' else self.X

    def kind(self, v):
        l, p = self.visits[v]
        k = 's' if self.c.swa[l] else 'g'
        return k.upper() if p > 0 else k

    def ctl(self, v):
        """[CTL] floats: attn-input block mask (5), mlp-input block mask (5), boundary flag, boundary slot one-hot (4),
        first-layer flag, loop-restart flag. Masks: 0 visible, NEG hidden; slot 4 (the partial) always visible."""
        c = self.c
        l, p = self.visits[v]
        nb = c.nblocks[l] if l > 0 else 1
        bnd = l % c.blk == 0
        nm = nb + (1 if bnd and l > 0 else 0)
        ma = [0.0 if j < nb else NEG for j in range(4)] + [0.0]
        mm = [0.0 if j < nm else NEG for j in range(4)] + [0.0]
        oh = [0.0] * 4
        if bnd:
            oh[l // c.blk] = 1.0
        return ma + mm + [1.0 if bnd else 0.0] + oh + [1.0 if l == 0 else 0.0, 1.0 if (l == c.ls and p > 0) else 0.0]

    def meta(self):
        c = self.c
        return {'arch': 'inkling2_loop', 'hidden': self.H, 'vocab': self.Vu, 'vocab_padded': self.V, 'layers': self.L,
                'visits': len(self.visits), 'heads': self.heads, 'kv_heads': self.kvh, 'head_dim': self.D, 'window': self.W,
                'rel_extent': self.X, 'experts': self.E, 'top_k': self.topk, 'expert_intermediate': self.I, 'tp': self.TP,
                'page': PAGE, 'ns': NS, 'ctl': CTL, 'loop': [c.ls, c.le, c.passes]}


def geo_from_env():
    tp = int(os.environ.get('TP', '2'))
    path = os.environ.get('CONFIG') or str(Path(os.environ['MODEL']) / 'config.json')
    return Geo(json.loads(Path(path).read_text()), tp)


bf = Q.bf
lin = Q.lin
lin_partial = Q.lin_partial


def urms(x, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)


def l2clip(x, thr):
    n = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True))
    return x * torch.clamp(thr / (n + 1e-6), max=1.0)


def depth(stack, q, mask):
    """stack [T, n, H], q [nq, H], mask [n] additive -> [T, nq, H]."""
    s = urms(stack)
    lg = torch.einsum('qh,tnh->tqn', q, s) * (s.shape[-1] ** -0.5) + mask.reshape(1, 1, -1)
    return torch.einsum('tqn,tnh->tqh', torch.softmax(lg, dim=-1), s)


def msg_gate(h, m, t, x):
    """(1 - tanh(f)) h + tanh(g) m with z = down([urms h, urms m]) (rank 64); weights replicated, bf16 (EXACT: f32)."""
    z = torch.cat([urms(h), urms(m)], dim=-1)
    z = mm(z, t[f'gd_{x}'])
    f = torch.tanh(mm(z, t[f'gf_{x}']))
    gg = torch.tanh(mm(z, t[f'gg_{x}']))
    return (1 - f) * h + gg * m


def mm(h, w):
    """h [T, in] x w (WT: [in, out]; else [out, in]) in bf16 with f32 accumulate (EXACT: f32)."""
    wm = w if WT else w.T
    if EXACT:
        return torch.matmul(h, wm.float())
    return torch.matmul(h.to(torch.bfloat16), wm.to(torch.bfloat16)).float()


def dotm(a, b):
    """Batched attention contraction a [h, M, X] x b [h, X, Y] of bf16-valued operands. ATT_BF16: bf16 operands on the
    tensor engine; measured on trn1 (neuronx-cc 2.26): the result is ROUNDED to bf16 even when .float() follows directly
    (unlike fp8 x fp8), so the CPU mirror rounds too. ATT_BF16=0 / EXACT: f32."""
    if EXACT or not ATT_BF16:
        return torch.matmul(a.float(), b.float())
    if a.device.type == 'cpu':
        return bf(torch.matmul(a.float(), b.float()))
    return torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16)).float()


def tau(pos, g):
    c = g.c
    return 1.0 + c.alpha * torch.log(torch.clamp((pos + 1).float() / c.n_floor, min=1.0))


def select_rows(ctx, n, k):
    """ctx [L, C] -> ctx[n : n + k] for a runtime count n (0-d int64) by a one-hot matmul (exact: one nonzero per row)."""
    L = ctx.shape[0]
    sel = (torch.arange(L, device=ctx.device).reshape(1, L) == (torch.arange(k, device=ctx.device).reshape(k, 1) + n)).float()
    if EXACT or ctx.dtype == torch.float32:
        return torch.matmul(sel, ctx.float())
    return torch.matmul(sel.to(ctx.dtype), ctx).float()


def conv_step(x, w, st, n, dyn=None):
    """Depthwise causal conv over one row's Q tokens. x [Q, C], w [C, K] (w[:, K-1] = current token), st [C, K-1]
    oldest first, n = valid tokens (None: all). dyn [Q, C, K]: per-token taps added to w (global k / v).
    -> ((conv + x) / sqrt(K + 1 | 2K + 1), new state)."""
    K = w.shape[1]
    Qn = x.shape[0]
    ctx = torch.cat([st.t(), x], dim=0)                                                # [Q + K - 1, C]
    if dyn is None:
        y = sum(w[:, j].reshape(1, -1) * ctx[j:j + Qn] for j in range(K))
        y = (y + x) / math.sqrt(K + 1)
    else:
        y = sum((w[:, j].reshape(1, -1) + dyn[:, :, j]) * ctx[j:j + Qn] for j in range(K))
        y = (y + x) / math.sqrt(2 * K + 1)
    if n is None:
        new = ctx[Qn:Qn + K - 1]
    else:
        new = select_rows(ctx, n, K - 1)
    return y, new.t()


def skew(X):
    """X [h, Q, L] -> Y [h, Q, L + Q - 1] with Y[:, t, c] = X[:, t, c - t] (0 outside). SKEW=reshape: static pad / flatten /
    reshape (rows end up misaligned with the tiles: many small copies); bits: log2(Q) shift-and-select passes."""
    h, Qn, L = X.shape
    if Qn == 1:
        return X
    if SKEW == 'bits':
        Y = torch.cat([X, torch.zeros(h, Qn, Qn - 1, device=X.device, dtype=X.dtype)], dim=-1)
        tt = torch.arange(Qn, device=X.device).reshape(1, Qn, 1)
        sh = 1
        while sh < Qn:
            moved = torch.cat([torch.zeros(h, Qn, sh, device=X.device, dtype=X.dtype), Y[..., :-sh]], dim=-1)
            Y = torch.where(torch.bitwise_and(tt, sh) > 0, moved, Y)
            sh *= 2
        return Y
    M = L + Qn
    Z = torch.cat([X, torch.zeros(h, Qn, Qn, device=X.device, dtype=X.dtype)], dim=-1).reshape(h, Qn * M)
    return Z[:, :Qn * (M - 1)].reshape(h, Qn, M - 1)


# --------------------------------------------------------------------------------------------------------- blocks
def attention(t, g, kind, R_, Q_, P, qh, kh, vh, allreduce):
    """One visit's attention for R_ rows x Q_ tokens. qh / kh / vh [T, H] normed inputs. -> (o [T, H] reduced,
    {state: new})."""
    T, hq, kvl, D = R_ * Q_, g.hq, g.kvl, g.D
    glb = kind in 'gG'
    NB = g.nb(kind)
    rep = hq // kvl
    pos = t['pos']
    q = lin(qh, t, 'q').reshape(T, hq, D) * t['qs'].reshape(1, hq, D)
    q = l2clip(q, g.c.qk_clip)
    r = lin(qh, t, 'r').reshape(T, hq, g.d_rel) * t['rs'].reshape(1, hq, g.d_rel)
    r = l2clip(r, g.c.rel_clip)
    if glb:
        tq = tau(pos.clamp(min=0), g).reshape(T, 1, 1)
        q, r = q * tq, r * tq
    rpf = torch.flip(t['rp'], dims=[-1])                                              # [d_rel, NB]: column j = distance NB - 1 - j
    rl_dev = r.device
    sink = mm(qh, t['sk']).reshape(T, hq)
    gate = torch.sigmoid(lin(qh, t, 'gt'))
    kx = lin(kh, t, 'k')
    vx = lin(vh, t, 'v')
    if glb:
        dk = mm(kh, t['kw']).reshape(T, kvl, 1, g.K).expand(T, kvl, D, g.K).reshape(T, kvl * D, g.K)
        dv = mm(vh, t['vw']).reshape(T, kvl, 1, g.K).expand(T, kvl, D, g.K).reshape(T, kvl * D, g.K)
    outs, states, news = [], {}, []
    # scores = q . k / sqrt(D): the factor rides on the operands (cq on q before its bf16 rounding, the power of two ck on
    # the keys: exact in bf16), so no pass over the score tensor is spent on it. The multiply on the gathered far keys /
    # values also MATERIALISES them as one contiguous tensor: contracted straight out of the page gather, neuronx-cc
    # emits one matmul (and weight load) per 16-key page.
    sck = 2.0 ** -math.floor(math.log2(math.sqrt(D)))
    cq = 1.0 / (math.sqrt(D) * sck)
    scv = 0.125
    q = bf(q * cq)
    far = glb and 'nofar' not in PROF
    qb = ATT_QBLOCK if ATT_QBLOCK and Q_ > ATT_QBLOCK and Q_ % ATT_QBLOCK == 0 else Q_      # queries per block
    blk = ATT_BLOCK if ATT_BLOCK and P > ATT_BLOCK and P % ATT_BLOCK == 0 else P            # pages per far-key block
    nq = Q_ // qb
    bank3 = None
    if SKEW.startswith('bank') and qb > 1:
        # the bias as a matmul per query with ITS shifted copy of the profile bank (bank3[t, :, c] = profile of distance
        # t + NB - c; the same for every query block and row): the shifts are along the free dimension of [d_rel, .]
        # tiles, where the skew of the [queries, keys] bias itself is ~140 small DMA packets per row
        bank3 = torch.stack([torch.nn.functional.pad(rpf, (t_ + 1, qb - 1 - t_)) for t_ in range(qb)], dim=0)   # [qb, d_rel, NB + qb]
    for i in range(R_):
        sfx = f'_{i}' if R_ > 1 else ''
        rows = slice(i * Q_, (i + 1) * Q_)
        pr = pos[rows]
        n = None if Q_ == 1 else (pr >= 0).sum()
        cst = t['conv' + sfx]
        ck, cv = cst[:g.CK], cst[g.CK:2 * g.CK]
        dyn = glb and 'nodyn' not in PROF
        k, nk = conv_step(kx[rows], t['kc'], ck, n, dk[rows] if dyn else None)
        v, nv = conv_step(vx[rows], t['vc'], cv, n, dv[rows] if dyn else None)
        k = l2clip(k.reshape(Q_, kvl, D), g.c.qk_clip)
        v = urms(v.reshape(Q_, kvl, D), g.eps)
        k, v = bf(k), bf(v)
        buf = t['buf' + sfx]                                                          # [NB, 2, kvl, D]
        new = torch.stack((k, v), dim=1).to(buf.dtype)                                # [Q, 2, kvl, D]
        news.append(new)
        allkv = torch.cat([buf, new], dim=0)                                           # [NB + Q, 2, kvl, D]
        # head-major layout [kvl, rep, queries, keys]: the score and value contractions are batched matmuls over the kv
        # heads with nothing to transpose in between (a [q, h, g, k] layout made neuronx-cc reload the whole softmax
        # output from HBM once per head and key tile: 2.3 GB of DMA in a Q = 256 sliding visit)
        qh = q[rows].reshape(Q_, kvl, rep, D).permute(1, 2, 0, 3)                      # [kvl, rep, Q, D]
        rh = r[rows].permute(1, 0, 2)                                                  # [hq, Q, d_rel]
        skh = sink[rows].t().reshape(kvl, rep, Q_)
        # near keys (buffer + chunk) per query block of qb tokens: block j (first token b) sees the keys b .. b + NB + qb
        # of [buf; chunk] only (older ones are >= NB away: outside the window / far keys; later ones are in the future),
        # so a wide chunk costs Q / qb cells of the Q = qb size. Key kk of query tt (both block-local) is at distance
        # tt + NB - kk; its bias is rl[tt, :, dist] for 0 <= dist < NB: a static skew of the query's profile.
        m, l, acc = [None] * nq, [None] * nq, [None] * nq
        for j in range(nq):
            b = j * qb
            qs = slice(b, b + qb)
            nk_ = NB + qb
            span = allkv[b:b + nk_]
            qj = qh[:, :, qs].reshape(kvl, rep * qb, D)
            s = dotm(qj, span[:, 0].permute(1, 2, 0).float() * sck).reshape(kvl, rep, qb, nk_)
            if bank3 is not None:
                if SKEW == 'bank':
                    B = torch.matmul(r[rows][qs], bank3).permute(1, 0, 2).reshape(kvl, rep, qb, nk_)
                else:
                    B = torch.einsum('htd,tdc->htc', rh[:, qs], bank3).reshape(kvl, rep, qb, nk_)
            else:
                Z0 = torch.cat([torch.zeros(hq, qb, 1, device=rl_dev), torch.matmul(rh[:, qs], rpf)], dim=-1)   # [hq, qb, NB + 1]
                B = skew(Z0).reshape(kvl, rep, qb, nk_)
            tt = torch.arange(qb, device=pos.device).reshape(qb, 1)
            kidx = torch.arange(nk_, device=pos.device).reshape(1, nk_)
            dist = tt + NB - kidx
            kpos = pr[:1].reshape(1, 1) + (b - NB) + kidx                              # positions of the keys
            ok = (dist >= 0) & (dist < NB) & (kpos >= 0) & (pr[qs].reshape(qb, 1) >= 0)
            if kind in 'sS':
                ok = ok & (dist < g.W)
            s = torch.where(ok.reshape(1, 1, qb, nk_), s + B, torch.full_like(s, NEG))
            sk = skh[:, :, qs]
            vj = span[:, 1].permute(1, 0, 2)                                           # [kvl, keys, D]
            # online softmax over [near, sink, far blocks]: running max m, sum l, weighted values acc (the sink logit is
            # finite on every row, so m is finite from the start)
            m[j] = torch.maximum(torch.amax(s, dim=-1), sk)
            p = torch.exp(s - m[j].unsqueeze(-1))
            l[j] = p.sum(dim=-1) + torch.exp(sk - m[j])
            acc[j] = dotm(bf(p).reshape(kvl, rep * qb, nk_), vj)
        if far:
            # far keys (distance >= rel_extent, no bias) from the request's pages, in blocks of blk pages: the gathered
            # keys and the score / exp / value passes of one cell stay block-sized whatever the context (P)
            cache = t['kv']
            table = t['table'][i]
            for b0 in range(0, P, blk):
                if 'noread' in PROF:   # probe: the far keys from a static slice of the pool (no page gather)
                    kvp = cache[b0:b0 + blk] + (table.float().sum() * 0).to(cache.dtype)
                else:
                    kvp = cache[table[b0:b0 + blk]]                                    # [blk, 2, kvl, 16, D]
                nk_ = blk * PAGE
                if FARK == 'dk':
                    fk = kvp[:, 0].permute(1, 3, 0, 2).reshape(kvl, D, nk_).float() * sck
                else:
                    fk = (kvp[:, 0].permute(1, 0, 2, 3).reshape(kvl, nk_, D).float() * sck).transpose(1, 2)
                fv = kvp[:, 1].permute(1, 0, 2, 3).reshape(kvl, nk_, D).float() * scv
                fpos = torch.arange(b0 * PAGE, (b0 + blk) * PAGE, device=pos.device).reshape(1, nk_)
                for j in range(nq):
                    qs = slice(j * qb, (j + 1) * qb)
                    pq = pr[qs].reshape(qb, 1)
                    sf = dotm(qh[:, :, qs].reshape(kvl, rep * qb, D), fk).reshape(kvl, rep, qb, nk_)
                    sf = torch.where(((fpos <= pq - g.X) & (pq >= 0)).reshape(1, 1, qb, nk_), sf, torch.full_like(sf, NEG))
                    mn = torch.maximum(m[j], torch.amax(sf, dim=-1))
                    c = torch.exp(m[j] - mn)
                    p = torch.exp(sf - mn.unsqueeze(-1))
                    l[j] = l[j] * c + p.sum(dim=-1)
                    acc[j] = acc[j] * c.reshape(kvl, rep * qb, 1) + dotm(bf(p).reshape(kvl, rep * qb, nk_), fv) * (1.0 / scv)
                    m[j] = mn
        # (no softmax op and no [scores, sink] concatenation: the sink enters through m and l)
        o = [acc[j] / l[j].reshape(kvl, rep * qb, 1) for j in range(nq)]
        o = [x_.reshape(kvl, rep, qb, D) for x_ in o]
        o = (o[0] if nq == 1 else torch.cat(o, dim=2)).permute(2, 0, 1, 3)             # [Q, kvl, rep, D]
        outs.append(o.reshape(Q_, hq * D))
        # state: the last NB valid keys, the conv inputs
        if Q_ == 1:
            nbuf = allkv[1:]
        else:
            nbuf = select_rows(allkv.reshape(NB + Q_, -1), n, NB).reshape(NB, 2, kvl, D).to(buf.dtype)
        states['buf' + sfx] = nbuf
        states['conv' + sfx] = (nk, nv)
    if glb and 'nofar' in PROF:
        states['kv'] = t['kv']
    if glb and 'nofar' not in PROF:
        cache = t['kv']
        page = torch.bitwise_right_shift(t['wslot'], 4)
        off = torch.bitwise_and(t['wslot'], PAGE - 1)
        at = (page[:, None, None], torch.arange(2, dtype=torch.int64, device=pos.device)[None, :, None],
              torch.arange(kvl, dtype=torch.int64, device=pos.device)[None, None, :], off[:, None, None])
        if 'nowrite' in PROF:   # probe: the pool passes through (the chunk's keys are not written)
            states['kv'] = cache + (torch.cat(news, dim=0).float().sum() * 0 + t['wslot'].float().sum() * 0).to(cache.dtype)
        else:
            states['kv'] = cache.index_put(at, torch.cat(news, dim=0).to(cache.dtype))
    o = torch.cat(outs, dim=0) * gate
    return allreduce(lin_partial(bf(o), t, 'o')), states


def route(n, t, g):
    """sigmoid router + selection bias, top-k, renormalised, x route_scale -> combine weights [T, E] (0 off the top-k)."""
    logits = torch.matmul(n.float(), t['rw'] if WT else t['rw'].T)
    s = torch.sigmoid(logits)
    sel = s + t['rb'].reshape(1, -1)
    T = n.shape[0]
    experts = torch.arange(g.E, device=n.device)
    if ROUTE.startswith('pad') and T < int(ROUTE[3:]):
        N = int(ROUTE[3:])
        padded = torch.cat((sel, torch.zeros(N - T, g.E, device=n.device) + sel[:1] * 0.0), dim=0)
        top = torch.topk(padded, g.topk, dim=-1)[1][:T]
    else:
        top = torch.topk(sel, g.topk, dim=-1)[1]
    mask = (top.unsqueeze(-1) == experts.reshape(1, 1, g.E)).float().sum(dim=1)        # [T, E]
    w = s * mask
    return w / (w.sum(dim=-1, keepdim=True) + 1e-20) * g.c.route_scale


def route_top(n, t, g):
    """-> (top [T, K] expert indices, w [T, K] their combine weights): route() in index form (the grouped prefill MoE)."""
    logits = torch.matmul(n.float(), t['rw'] if WT else t['rw'].T)
    s = torch.sigmoid(logits)
    top = torch.topk(s + t['rb'].reshape(1, -1), g.topk, dim=-1)[1]                    # [T, K]
    onehot = (top.unsqueeze(-1) == torch.arange(g.E, device=n.device).reshape(1, 1, g.E)).float()
    w = torch.einsum('tke,te->tk', onehot, s)
    return top, w / (w.sum(dim=-1, keepdim=True) + 1e-20) * g.c.route_scale


def act2(gate, up, g):
    c = g.c
    return torch.tanh(gate / c.gate_cap) * c.gate_cap * torch.sigmoid(gate) * torch.tanh(up / c.up_cap) * c.up_cap


def act(gu, g):
    gate, up = gu[..., 0, :], gu[..., 1, :]
    c = g.c
    return torch.tanh(gate / c.gate_cap) * c.gate_cap * torch.sigmoid(gate) * torch.tanh(up / c.up_cap) * c.up_cap


def moe(n, t, g, mode, allreduce):
    """This rank's slices of the routed experts (no shared expert), one all-reduce."""
    T, E, EIl = n.shape[0], g.E, g.EIl
    if mode == 'group':
        # routed-only prefill MoE: qwen35_graphs.moe_grouped_idx (hot experts on every token, every other expert on its own
        # MOE_CAP token slots) with this model's activation. -> (sum, ovf [1]): exact iff ovf <= MOE_CAP, otherwise the
        # engine reruns the visit's MoE part with the MOE=all NEFF (the state update below is held back in that case).
        top, w = route_top(n, t, g)
        Q.silu_mul = lambda gate, up: act2(gate, up, g)
        routed, ovf = Q.moe_grouped_idx(n, top, w, t, g)
        return allreduce(routed), ovf
    comb = route(n, t, g)
    gu, gus, dw, dc = t['egu'], t.get('egu_s'), t['ed'], t.get('ed_c')
    if mode == 'gather' and g.topk * T < E:
        slots = g.topk * T
        used = (comb > 0).float().amax(dim=0, keepdim=True)
        ar = torch.arange(E, device=n.device).float().reshape(1, E)
        if MOE_SLOTS == 'onehot':
            # slot s <- the s-th used expert (ascending), empty slots <- expert 0 with weight 0; no top-k over the experts
            # (qwen35_graphs MOE_SLOTS=onehot: the device's wide top-k of the multi-row buckets is not reliable on trn1)
            before = (ar.reshape(E, 1) >= ar.reshape(1, E)).float()
            order = (used * before).sum(dim=-1).reshape(1, E) - 1.0
            sel = ((order == torch.arange(slots, device=n.device).float().reshape(slots, 1)) & (used > 0)).float()   # [slots, E]
            idx = (sel * ar).sum(dim=-1).round().long()
            comb = (comb.unsqueeze(1) * sel.unsqueeze(0)).sum(dim=-1)
        else:
            idx = torch.topk(used - ar * (0.5 / E), slots, dim=-1)[1].reshape(slots)
            comb = torch.index_select(comb, 1, idx)
        gu, dw = torch.index_select(gu, 0, idx), torch.index_select(dw, 0, idx)
        gus = torch.index_select(gus, 0, idx) if gus is not None else None
        dc = torch.index_select(dc, 0, idx) if dc is not None else None
    E_ = gu.shape[0]
    if mode == 'all' and MOE_EG and E_ > MOE_EG and E_ % MOE_EG == 0:
        # every expert on every token in expert groups: the [T, EG 2 EIl] intermediates stay small (the whole [T, E 2 EIl]
        # f32 tensor is 151 MB at T = 256 and spills); each group's activations are quantised on their own
        routed = 0
        for e0 in range(0, E_, MOE_EG):
            sl = slice(e0, e0 + MOE_EG)
            routed = routed + moe_experts(n, comb[:, sl], gu[sl], None if gus is None else gus[sl], dw[sl], None if dc is None else dc[sl], g)
        return allreduce(routed), None
    if mode == 'all' and MOE_TSUB and T > MOE_TSUB and T % MOE_TSUB == 0:
        # a wide chunk in token sub-blocks: the [sub, E 2 EIl] intermediates stay those of the Q = MOE_TSUB graph
        return allreduce(torch.cat([moe_experts(n[b:b + MOE_TSUB], comb[b:b + MOE_TSUB], gu, gus, dw, dc, g) for b in range(0, T, MOE_TSUB)], dim=0)), None
    return allreduce(moe_experts(n, comb, gu, gus, dw, dc, g)), None


def moe_experts(n, comb, gu, gus, dw, dc, g):
    T, EIl, E_ = n.shape[0], g.EIl, gu.shape[0]
    hgu = Q.egu_mm(n, gu, gus, EIl).reshape(T, E_, 2, EIl)
    a = bf(act(hgu, g)) * comb.unsqueeze(-1)
    if dc is not None:
        a = a * dc.reshape(1, E_, EIl)
    a2 = a.reshape(T, E_ * EIl)
    wd = dw.reshape(E_ * EIl, g.H)
    if dc is not None:
        s = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        routed = G.mm8((a2 / s).to(F8), wd).float() * s
    elif EXACT:
        routed = torch.matmul(a2, wd.float())
    else:
        routed = torch.matmul(a2.to(torch.bfloat16), wd.to(torch.bfloat16)).float()
    return routed


def visit(kind, t, g, R_, Q_, P, allreduce, moe_mode='all', part='full'):
    """One layer visit -> (x_new [T, NS, H], {state name: new tensor}). part: full (attention + MoE), attn (up to the
    residual after attention: slot 4 holds it, the blocks / restart are applied; conv state's MoE part passed through) or
    moe (the rest: from that x and the conv state the attn part wrote)."""
    T, H = R_ * Q_, g.H
    x = t['x'].reshape(T, NS, H)
    t = dict(t)
    if part != 'moe':
        for i in range(R_):   # a row starting at position 0 (new / re-prefilled sequence) starts from zero conv / buf state
            sfx = f'_{i}' if R_ > 1 else ''
            keep = (t['pos'][i * Q_] > 0).float()
            t['conv' + sfx] = t['conv' + sfx] * keep
            t['buf' + sfx] = t['buf' + sfx] * keep.to(t['buf' + sfx].dtype)
    c = t['ctl']
    ma, mmask, bnd, oh, first, restart = c[0:5], c[5:10], c[10], c[11:15], c[15], c[16]
    sl = [x[:, j] for j in range(NS)]
    msgs = kind in 'SG'
    if msgs and part != 'moe':   # loop restart (first loop layer of passes >= 2): message sources <- B2 / final partial, partial <- B1
        sl[5] = restart * sl[2] + (1 - restart) * sl[5]
        sl[6] = restart * sl[4] + (1 - restart) * sl[6]
        sl[4] = restart * sl[1] + (1 - restart) * sl[4]
    if msgs:
        mes = depth(torch.stack([sl[5], sl[6]], dim=1), t['mg'], torch.zeros(2, device=x.device))   # [T, 4, H]
    st = {}
    conv_parts = []
    if part == 'moe':
        res = sl[4]
        for i in range(R_):
            sfx = f'_{i}' if R_ > 1 else ''
            conv_parts.append((None if Q_ == 1 else (t['pos'][i * Q_:(i + 1) * Q_] >= 0).sum(), None))
    else:
        prt = sl[4]
        stack = torch.stack(sl[0:5], dim=1)                                            # [T, 5, H]
        if kind in 'sS':
            hin = depth(stack, t['aq'], ma)[:, 0]
            hin = first * prt + (1 - first) * hin
            if msgs:
                qh = Q.rms(msg_gate(hin, mes[:, 0], t, 'q'), t['qn'], g.eps)
                kh = Q.rms(msg_gate(hin, mes[:, 1], t, 'k'), t['qn'], g.eps)
                vh = Q.rms(msg_gate(hin, mes[:, 2], t, 'v'), t['qn'], g.eps)
            else:
                qh = kh = vh = Q.rms(hin, t['qn'], g.eps)
        else:
            qkv = depth(stack, t['aq'], ma)
            qi, ki, vi = [first * prt + (1 - first) * qkv[:, j] for j in range(3)]
            if msgs:
                qi, ki, vi = msg_gate(qi, mes[:, 0], t, 'q'), msg_gate(ki, mes[:, 1], t, 'k'), msg_gate(vi, mes[:, 2], t, 'v')
            qh, kh, vh = Q.rms(qi, t['qn'], g.eps), Q.rms(ki, t['kn'], g.eps), Q.rms(vi, t['vn'], g.eps)
        qh, kh, vh = bf(qh), bf(kh), bf(vh)
        a, st = attention(t, g, kind, R_, Q_, P, qh, kh, vh, allreduce)
        outs_a = []
        for i in range(R_):   # post-attention conv (replicated, on the reduced output)
            sfx = f'_{i}' if R_ > 1 else ''
            rows = slice(i * Q_, (i + 1) * Q_)
            n = None if Q_ == 1 else (t['pos'][rows] >= 0).sum()
            ya, na = conv_step(a[rows], t['ac'], t['conv' + sfx][2 * g.CK:2 * g.CK + H], n)
            outs_a.append(ya)
            conv_parts.append((n, na))
        a = torch.cat(outs_a, dim=0)
        for j in range(4):
            sl[j] = sl[j] + bnd * oh[j] * (prt - sl[j])
        res = a + (1 - bnd) * prt
        if part == 'attn':
            sl[4] = res
            for i in range(R_):
                sfx = f'_{i}' if R_ > 1 else ''
                nk, nv = st['conv' + sfx]
                st['conv' + sfx] = torch.cat([nk, nv, conv_parts[i][1], t['conv' + sfx][2 * g.CK + H:]], dim=0)
            return torch.stack(sl, dim=1), st
    hm = depth(torch.stack(sl[0:4] + [res], dim=1), t['mq'], mmask)[:, 0]
    if msgs:
        hm = msg_gate(hm, mes[:, 3], t, 'm')
    if 'nomoe' in PROF:
        ovf = None
        mo = allreduce(hm * t['ed'][0, 0, :1].float() + t['egu'].reshape(-1)[:1].float() + t['rw'].reshape(-1)[:1] + t['rb'][:1] + t.get('egu_s', t['rb'])[..., :1].reshape(-1)[:1] + t.get('ed_c', t['rb'])[..., :1].reshape(-1)[:1])
    else:
        mo, ovf = moe(bf(Q.rms(hm, t['mn'], g.eps)), t, g, moe_mode, allreduce)
    if ovf is not None:
        assert part == 'moe', 'MOE=group needs PART=moe (the rerun restarts from the MoE input)'
        st['ovf'] = ovf
    outs_m = []
    for i in range(R_):
        sfx = f'_{i}' if R_ > 1 else ''
        rows = slice(i * Q_, (i + 1) * Q_)
        n, na = conv_parts[i]
        cst = t['conv' + sfx]
        ym, nm = conv_step(mo[rows], t['mc'], cst[2 * g.CK + H:], n)
        if ovf is not None:   # an overflowed grouped launch leaves the state as it was (the engine reruns the part)
            nm = torch.where(ovf[0] <= float(Q.MOE_CAP), nm, cst[2 * g.CK + H:])
        outs_m.append(ym)
        if part == 'moe':
            st['conv' + sfx] = torch.cat([cst[:2 * g.CK + H], nm], dim=0)
        else:
            nk, nv = st['conv' + sfx]
            st['conv' + sfx] = torch.cat([nk, nv, na, nm], dim=0)
    sl[4] = res + torch.cat(outs_m, dim=0)
    return torch.stack(sl, dim=1), st


ATT_W = {'q', 'gt', 'k', 'v', 'o', 'r', 'sk', 'qs', 'rs', 'rp', 'kc', 'vc', 'ac', 'qn', 'kn', 'vn', 'aq', 'kw', 'vw'}


def part_names(kind, part, names):
    """the weight / ctl names a split part takes (mg and the message gates of q / k / v vs m)."""
    if part == 'full':
        return names
    out = []
    for n in names:
        base = n.split('_s')[0] if n.endswith('_s') else n
        att = base in ATT_W or n[:5] in ('gd_q', 'gf_q', 'gg_q', 'gd_k', 'gf_k', 'gg_k', 'gd_v', 'gf_v', 'gg_v')
        if n in ('ctl', 'mg') or (att if part == 'attn' else not att):
            out.append(n)
    return out


def head(t, g, rows):
    x = torch.index_select(t['x'], 0, rows).reshape(-1, NS, g.H)
    h = depth(x[:, 0:5], t['oq'], torch.zeros(5, device=x.device))[:, 0]
    h = bf(Q.rms(h, t['norm'], g.eps))
    if 'w_s' in t:
        a = torch.amax(torch.abs(h), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        lg = G.mm8((h / a).to(F8), t['w'] if WT else t['w'].T).float() * a * t['w_s'].reshape(1, -1)
    else:
        lg = mm(h, t['w'])
    cap = g.c.final_cap
    return torch.tanh(lg / cap) * cap


def segment(kinds, t, g, R_, Q_, P, allreduce, moe_mode='all'):
    """Consecutive visits in one graph: inputs l<i>.<name> are visit i's weights / ctl / states; step inputs shared."""
    common = {k: v for k, v in t.items() if '.' not in k}
    x, states = t['x'], {}
    for i, kind in enumerate(kinds):
        pre = f'l{i}.'
        own = {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
        x, st = visit(kind, {**common, **own, 'x': x}, g, R_, Q_, P, allreduce, moe_mode)
        states.update({pre + k: v for k, v in st.items()})
    return x, states


# ---------------------------------------------------------------------------------------------------------- shapes
def weight_shapes(kind, g, fp8=True):
    """name -> (shape, dtype) of one visit's per-rank inputs (attention + MoE + ctl); the prepare step writes the
    weights (ctl comes from Geo.ctl)."""
    H, f32, b16 = g.H, torch.float32, torch.bfloat16
    wt = F8 if fp8 else b16
    s = {'ctl': ((CTL,), f32)}

    def proj(name, out, inp, q8=fp8):
        s[name] = ((inp, out) if WT else (out, inp), (wt if q8 else (f32 if EXACT else b16)))
        if q8:
            s[name + '_s'] = ((out,), f32)
    hd = g.hq * g.D
    proj('q', hd, H); proj('gt', hd, H); proj('k', g.CK, H); proj('v', g.CK, H); proj('o', H, hd); proj('r', g.hq * g.d_rel, H)
    proj('sk', g.hq, H, q8=False)
    s['qs'] = ((g.hq * g.D,), f32); s['rs'] = ((g.hq * g.d_rel,), f32); s['rp'] = ((g.d_rel, g.nb(kind)), f32)
    s['kc'] = ((g.CK, g.K), f32); s['vc'] = ((g.CK, g.K), f32); s['ac'] = ((H, g.K), f32); s['mc'] = ((H, g.K), f32)
    s['qn'] = ((H,), f32); s['mn'] = ((H,), f32); s['mq'] = ((1, H), f32)
    if kind in 'sS':
        s['aq'] = ((1, H), f32)
    else:
        s['aq'] = ((3, H), f32); s['kn'] = ((H,), f32); s['vn'] = ((H,), f32)
        proj('kw', g.kvl * g.K, H, q8=False); proj('vw', g.kvl * g.K, H, q8=False)
    if kind in 'SG':
        s['mg'] = ((4, H), f32)
        for x in 'qkvm':
            r = g.c.rank
            s[f'gd_{x}'] = (((2 * H, r) if WT else (r, 2 * H)), f32 if EXACT else b16)
            s[f'gf_{x}'] = (((r, H) if WT else (H, r)), f32 if EXACT else b16)
            s[f'gg_{x}'] = (((r, H) if WT else (H, r)), f32 if EXACT else b16)
    s['rw'] = ((H, g.E) if WT else (g.E, H), f32); s['rb'] = ((g.E,), f32)
    s['egu'] = ((g.E, H, 2 * g.EIl) if WT else (g.E, 2 * g.EIl, H), wt)
    s['ed'] = ((g.E, g.EIl, H), wt)
    if fp8:
        s['egu_s'] = ((g.E, 2 * g.EIl), f32); s['ed_c'] = ((g.E, g.EIl), f32)
    return s


def state_shapes(kind, g, R_, pages, part='full'):
    """bound per-row buf / conv (suffix _i for R > 1); g / G: the paged pool kv. part moe: conv only."""
    kd = torch.float32 if EXACT else torch.bfloat16
    one = {'buf': ((g.nb(kind), 2, g.kvl, g.D), kd), 'conv': ((g.CS, g.K - 1), torch.float32)}
    if part == 'moe':
        one.pop('buf')
    s = dict(one) if R_ == 1 else {f'{k}_{i}': v for i in range(R_) for k, v in one.items()}
    if kind in 'gG' and part != 'moe':
        s['kv'] = ((pages, 2, g.kvl, PAGE, g.D), kd)
    return s


def step_shapes(kinds, R_, Q_, P, g, part='full'):
    T = R_ * Q_
    s = {'x': ((T, NS * g.H), torch.float32), 'pos': ((T,), torch.int64)}
    if any(k in 'gG' for k in kinds) and part != 'moe':
        s['wslot'] = ((T,), torch.int64); s['table'] = ((R_, P), torch.int64)
    return s


def head_shapes(g, T, S, fp8=True):
    s = {'x': ((T, NS * g.H), torch.float32), 'rows': ((S,), torch.int64), 'oq': ((1, g.H), torch.float32),
         'norm': ((g.H,), torch.float32)}
    s['w'] = (((g.H, g.VS) if WT else (g.VS, g.H)), F8 if fp8 else (torch.float32 if EXACT else torch.bfloat16))
    if fp8:
        s['w_s'] = ((g.VS,), torch.float32)
    return s


WT_NAMES = {'q', 'gt', 'k', 'v', 'o', 'r', 'sk', 'kw', 'vw', 'rw', 'w'}


def to_wt(name, w):
    """prepared [out, in] -> WT [in, out] (projections, router, message gates, head); experts' gate|up [E, H, 2 EIl]."""
    base = name.split('.')[-1]
    if base in WT_NAMES or base[:3] in ('gd_', 'gf_', 'gg_'):
        return w.t().contiguous()
    if base == 'egu':
        return w.permute(0, 2, 1).contiguous()
    return w


zeros = G.zeros


# ----------------------------------------------------------------------------------------------------------- trace
def main():
    workdir, kind = sys.argv[1], sys.argv[2]
    g = geo_from_env()
    fp8 = os.environ.get('WEIGHTS', 'fp8') == 'fp8'
    moe_mode = os.environ.get('MOE', 'all')
    pages = int(os.environ.get('PAGES', '65'))
    import torch_neuronx
    import torch_xla.core.xla_model as xm
    os.environ.setdefault('XLA_ALWAYS_ALLREDUCE', '1')
    groups = [list(range(g.TP))]
    allreduce = (lambda v: xm.all_reduce(xm.REDUCE_SUM, v, groups=groups)) if g.TP > 1 else (lambda v: v)
    if kind == 'head':
        T, S = int(sys.argv[3]), int(sys.argv[4])
        shp = head_shapes(g, T, S, fp8)
        names = list(shp)
        inputs = [torch.ones(*shp[n][0]) if n == 'w_s' else zeros(*shp[n]) for n in names]
        fn = lambda *a: head(dict(zip(names, a)), g, a[1])
        meta = {'kind': 'inkling_head', 'T': T, 'S': S, 'inputs': names, 'states': [], 'fp8': fp8, 'wt': WT}
        kwargs = {}
    else:
        R_, Q_, P = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        kinds = list(kind.split('_')[1]) if kind.startswith('seg_') else [kind]
        assert all(k in 'sSgG' for k in kinds), kind
        part = os.environ.get('PART', 'full')
        assert part == 'full' or len(kinds) == 1, 'split parts are single-visit graphs'
        step = step_shapes(kinds, R_, Q_, P, g, part)
        wsh, ssh = {}, {}
        for i, k_ in enumerate(kinds):
            pre = f'l{i}.' if len(kinds) > 1 else ''
            ws = weight_shapes(k_, g, fp8)
            wsh.update({pre + n: ws[n] for n in part_names(k_, part, list(ws))})
            ssh.update({pre + n: v for n, v in state_shapes(k_, g, R_, pages, part).items()})
        names, snames = list(step) + list(wsh), list(ssh)
        inputs = [zeros(*step[n]) for n in step] + [torch.ones(*wsh[n][0]) if n.endswith('_s') or n.endswith('_c') else zeros(*wsh[n]) for n in wsh]

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for n_ in snames:
                    self.register_buffer(n_.replace('.', '_'), zeros(*ssh[n_]))

            def forward(self, *a):
                t = dict(zip(names, a))
                for n_ in snames:
                    t[n_] = getattr(self, n_.replace('.', '_'))
                if len(kinds) > 1:
                    x, st = segment(kinds, t, g, R_, Q_, P, allreduce, moe_mode)
                else:
                    x, st = visit(kinds[0], t, g, R_, Q_, P, allreduce, moe_mode, part)
                for n_ in snames:
                    setattr(self, n_.replace('.', '_'), st[n_])
                if moe_mode == 'group':
                    return x.reshape(R_ * Q_, NS * g.H), st['ovf']
                return x.reshape(R_ * Q_, NS * g.H)
        fn = Block()
        kwargs = {'enable_aliasing': True}
        meta = {'kind': 'inkling_' + kind, 'kinds': kinds, 'R': R_, 'Q': Q_, 'P': P, 'pages': pages, 'inputs': names,
                'states': snames, 'fp8': fp8, 'moe': moe_mode, 'wt': WT, 'route': ROUTE, 'moe_slots': MOE_SLOTS, 'moe_eg': MOE_EG, 'part': part, 'extra_outputs': ['ovf'] if moe_mode == 'group' else [], 'moe_cap': Q.MOE_CAP if moe_mode == 'group' else 0,
                'moe_hot': Q.MOE_HOT, 'moe_ge': Q.MOE_GE, 'moe_tsub': MOE_TSUB, 'wt_moe': Q.WT_MOE, 'wt_grp': Q.WT_GRP, 'att_block': ATT_BLOCK, 'att_qblock': ATT_QBLOCK, 'att_bf16': ATT_BF16, 'skew': SKEW, 'fark': FARK, 'att_layout': 'hm', 'prof': sorted(PROF - {''})}
    meta['geometry'] = g.meta()
    try:
        torch_neuronx.trace(fn, tuple(inputs), compiler_workdir=workdir, **kwargs)
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
        print('trace-time compile error (expected with FN fp8):', str(error)[-700:])
    m = Path(workdir) / 'model'
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(m / 'graph.hlo'), str(m / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'] +
                   os.environ.get('NCC_FLAGS', '').split(), cwd=workdir, check=True)
    meta['neff'] = str(Path(workdir) / 'graph.neff')
    n_params = G.count_hlo_parameters(m / 'graph.hlo')
    if n_params is not None and n_params != len(meta['inputs']) + len(meta['states']):
        raise SystemExit(f'the traced graph has {n_params} parameters, meta lists {len(meta["inputs"]) + len(meta["states"])}: '
                         'an input is unused (dropped by the tracer) or the order is off')
    Path(workdir, 'meta.json').write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
