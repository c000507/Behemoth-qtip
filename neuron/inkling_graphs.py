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
ROUTE = os.environ.get('ROUTE', 'topk')     # topk | pad64 (membership from a 64-row padded topk; see qwen35_graphs.route)
NEG = -3e38


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
    """X [Q, h, L] -> Y [Q, h, L + Q - 1] with Y[t, :, c] = X[t, :, c - t] (0 outside): static pad / flatten / reshape."""
    Qn, h, L = X.shape
    if Qn == 1:
        return X
    M = L + Qn
    Z = torch.cat([X, torch.zeros(Qn, h, Qn, device=X.device, dtype=X.dtype)], dim=-1).permute(1, 0, 2).reshape(h, Qn * M)
    return Z[:, :Qn * (M - 1)].reshape(h, Qn, M - 1).permute(1, 0, 2)


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
    rl = torch.matmul(r, t['rp'])                                                     # [T, hq, NB]
    sink = mm(qh, t['sk']).reshape(T, hq)
    gate = torch.sigmoid(lin(qh, t, 'gt'))
    kx = lin(kh, t, 'k')
    vx = lin(vh, t, 'v')
    if glb:
        dk = mm(kh, t['kw']).reshape(T, kvl, 1, g.K).expand(T, kvl, D, g.K).reshape(T, kvl * D, g.K)
        dv = mm(vh, t['vw']).reshape(T, kvl, 1, g.K).expand(T, kvl, D, g.K).reshape(T, kvl * D, g.K)
    outs, states, news = [], {}, []
    q = bf(q)
    for i in range(R_):
        sfx = f'_{i}' if R_ > 1 else ''
        rows = slice(i * Q_, (i + 1) * Q_)
        pr = pos[rows]
        n = None if Q_ == 1 else (pr >= 0).sum()
        cst = t['conv' + sfx]
        ck, cv = cst[:g.CK], cst[g.CK:2 * g.CK]
        k, nk = conv_step(kx[rows], t['kc'], ck, n, dk[rows] if glb else None)
        v, nv = conv_step(vx[rows], t['vc'], cv, n, dv[rows] if glb else None)
        k = l2clip(k.reshape(Q_, kvl, D), g.c.qk_clip)
        v = urms(v.reshape(Q_, kvl, D), g.eps)
        k, v = bf(k), bf(v)
        buf = t['buf' + sfx]                                                          # [NB, 2, kvl, D]
        new = torch.stack((k, v), dim=1).to(buf.dtype)                                # [Q, 2, kvl, D]
        news.append(new)
        allkv = torch.cat([buf, new], dim=0)                                           # [NB + Q, 2, kvl, D]
        kk, vv = allkv[:, 0].float(), allkv[:, 1].float()
        qi = q[rows].reshape(Q_, kvl, rep, D)
        s = torch.einsum('qhgd,khd->qhgk', qi, kk) / math.sqrt(D)                     # [Q, kvl, rep, NB + Q]
        # relative bias: key kk of query tt sits at distance tt + NB - kk; bias = rl[tt, :, dist] for 0 <= dist < NB
        Z0 = torch.cat([torch.zeros(Q_, hq, 1, device=rl.device), torch.flip(rl[rows], dims=[-1])], dim=-1)
        B = skew(Z0).reshape(Q_, kvl, rep, NB + Q_)
        tt = torch.arange(Q_, device=pos.device).reshape(Q_, 1)
        kidx = torch.arange(NB + Q_, device=pos.device).reshape(1, NB + Q_)
        dist = tt + NB - kidx
        kpos = pr[:1].reshape(1, 1) - NB + kidx                                        # positions of the keys (p0 - NB + k)
        ok = (dist >= 0) & (dist < NB) & (kpos >= 0) & (pr.reshape(Q_, 1) >= 0)
        if kind in 'sS':
            ok = ok & (dist < g.W)
        s = torch.where(ok.reshape(Q_, 1, 1, NB + Q_), s + B, torch.full_like(s, NEG))
        parts = [s]
        if glb:
            cache = t['kv']
            kvp = cache[t['table'][i]].float()                                         # [P, 2, kvl, 16, D]
            fk = kvp[:, 0].permute(1, 0, 2, 3).reshape(kvl, P * PAGE, D)
            fv = kvp[:, 1].permute(1, 0, 2, 3).reshape(kvl, P * PAGE, D)
            sf = torch.einsum('qhgd,hkd->qhgk', qi, fk) / math.sqrt(D)
            fpos = torch.arange(P * PAGE, device=pos.device).reshape(1, 1, 1, P * PAGE)
            vis = (fpos <= (pr.reshape(Q_, 1, 1, 1) - g.X)) & (pr.reshape(Q_, 1, 1, 1) >= 0)
            parts.append(torch.where(vis, sf, torch.full_like(sf, NEG)))
        parts.append(sink[rows].reshape(Q_, kvl, rep, 1))
        p = torch.softmax(torch.cat(parts, dim=-1), dim=-1)
        o = torch.einsum('qhgk,khd->qhgd', bf(p[..., :NB + Q_]), vv)
        if glb:
            o = o + torch.einsum('qhgk,hkd->qhgd', bf(p[..., NB + Q_:NB + Q_ + P * PAGE]), fv)
        outs.append(o.reshape(Q_, hq * D))
        # state: the last NB valid keys, the conv inputs
        if Q_ == 1:
            nbuf = allkv[1:]
        else:
            nbuf = select_rows(allkv.reshape(NB + Q_, -1), n, NB).reshape(NB, 2, kvl, D).to(buf.dtype)
        states['buf' + sfx] = nbuf
        states['conv' + sfx] = (nk, nv)
    if glb:
        cache = t['kv']
        page = torch.bitwise_right_shift(t['wslot'], 4)
        off = torch.bitwise_and(t['wslot'], PAGE - 1)
        at = (page[:, None, None], torch.arange(2, dtype=torch.int64, device=pos.device)[None, :, None],
              torch.arange(kvl, dtype=torch.int64, device=pos.device)[None, None, :], off[:, None, None])
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


def act(gu, g):
    gate, up = gu[..., 0, :], gu[..., 1, :]
    c = g.c
    return torch.tanh(gate / c.gate_cap) * c.gate_cap * torch.sigmoid(gate) * torch.tanh(up / c.up_cap) * c.up_cap


def moe(n, t, g, mode, allreduce):
    """This rank's slices of the routed experts (no shared expert), one all-reduce."""
    T, E, EIl = n.shape[0], g.E, g.EIl
    comb = route(n, t, g)
    gu, gus, dw, dc = t['egu'], t.get('egu_s'), t['ed'], t.get('ed_c')
    if mode == 'gather' and g.topk * T < E:
        slots = g.topk * T
        used = (comb > 0).float().amax(dim=0, keepdim=True)
        ar = torch.arange(E, device=n.device).float().reshape(1, E)
        idx = torch.topk(used - ar * (0.5 / E), slots, dim=-1)[1].reshape(slots)
        comb = torch.index_select(comb, 1, idx)
        gu, dw = torch.index_select(gu, 0, idx), torch.index_select(dw, 0, idx)
        gus = torch.index_select(gus, 0, idx) if gus is not None else None
        dc = torch.index_select(dc, 0, idx) if dc is not None else None
    E_ = gu.shape[0]
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
    return allreduce(routed)


def visit(kind, t, g, R_, Q_, P, allreduce, moe_mode='all'):
    """One layer visit (attention + MoE) -> (x_new [T, NS, H], {state name: new tensor})."""
    T, H = R_ * Q_, g.H
    x = t['x'].reshape(T, NS, H)
    c = t['ctl']
    ma, mmask, bnd, oh, first, restart = c[0:5], c[5:10], c[10], c[11:15], c[15], c[16]
    sl = [x[:, j] for j in range(NS)]
    msgs = kind in 'SG'
    if msgs:   # loop restart (first loop layer of passes >= 2): message sources <- B2 / final partial, partial <- B1
        sl[5] = restart * sl[2] + (1 - restart) * sl[5]
        sl[6] = restart * sl[4] + (1 - restart) * sl[6]
        sl[4] = restart * sl[1] + (1 - restart) * sl[4]
    part = sl[4]
    stack = torch.stack(sl[0:5], dim=1)                                                # [T, 5, H]
    if msgs:
        mes = depth(torch.stack([sl[5], sl[6]], dim=1), t['mg'], torch.zeros(2, device=x.device))   # [T, 4, H]
    if kind in 'sS':
        hin = depth(stack, t['aq'], ma)[:, 0]
        hin = first * part + (1 - first) * hin
        if msgs:
            qh = Q.rms(msg_gate(hin, mes[:, 0], t, 'q'), t['qn'], g.eps)
            kh = Q.rms(msg_gate(hin, mes[:, 1], t, 'k'), t['qn'], g.eps)
            vh = Q.rms(msg_gate(hin, mes[:, 2], t, 'v'), t['qn'], g.eps)
        else:
            qh = kh = vh = Q.rms(hin, t['qn'], g.eps)
    else:
        qkv = depth(stack, t['aq'], ma)
        qi, ki, vi = [first * part + (1 - first) * qkv[:, j] for j in range(3)]
        if msgs:
            qi, ki, vi = msg_gate(qi, mes[:, 0], t, 'q'), msg_gate(ki, mes[:, 1], t, 'k'), msg_gate(vi, mes[:, 2], t, 'v')
        qh, kh, vh = Q.rms(qi, t['qn'], g.eps), Q.rms(ki, t['kn'], g.eps), Q.rms(vi, t['vn'], g.eps)
    qh, kh, vh = bf(qh), bf(kh), bf(vh)
    a, st = attention(t, g, kind, R_, Q_, P, qh, kh, vh, allreduce)
    # post-attention conv (replicated, on the reduced output)
    outs_a, outs_m, conv_parts = [], [], []
    for i in range(R_):
        sfx = f'_{i}' if R_ > 1 else ''
        rows = slice(i * Q_, (i + 1) * Q_)
        n = None if Q_ == 1 else (t['pos'][rows] >= 0).sum()
        cst = t['conv' + sfx]
        ya, na = conv_step(a[rows], t['ac'], cst[2 * g.CK:2 * g.CK + H], n)
        outs_a.append(ya)
        conv_parts.append((n, na))
    a = torch.cat(outs_a, dim=0)
    for j in range(4):
        sl[j] = sl[j] + bnd * oh[j] * (part - sl[j])
    res = a + (1 - bnd) * part
    hm = depth(torch.stack(sl[0:4] + [res], dim=1), t['mq'], mmask)[:, 0]
    if msgs:
        hm = msg_gate(hm, mes[:, 3], t, 'm')
    mo = moe(bf(Q.rms(hm, t['mn'], g.eps)), t, g, moe_mode, allreduce)
    for i in range(R_):
        sfx = f'_{i}' if R_ > 1 else ''
        rows = slice(i * Q_, (i + 1) * Q_)
        n, na = conv_parts[i]
        cst = t['conv' + sfx]
        ym, nm = conv_step(mo[rows], t['mc'], cst[2 * g.CK + H:], n)
        outs_m.append(ym)
        nk, nv = st['conv' + sfx]
        st['conv' + sfx] = torch.cat([nk, nv, na, nm], dim=0)
    sl[4] = res + torch.cat(outs_m, dim=0)
    return torch.stack(sl, dim=1), st


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


def state_shapes(kind, g, R_, pages):
    """bound per-row buf / conv (suffix _i for R > 1); g / G: the paged pool kv."""
    kd = torch.float32 if EXACT else torch.bfloat16
    one = {'buf': ((g.nb(kind), 2, g.kvl, g.D), kd), 'conv': ((g.CS, g.K - 1), torch.float32)}
    s = dict(one) if R_ == 1 else {f'{k}_{i}': v for i in range(R_) for k, v in one.items()}
    if kind in 'gG':
        s['kv'] = ((pages, 2, g.kvl, PAGE, g.D), kd)
    return s


def step_shapes(kinds, R_, Q_, P, g):
    T = R_ * Q_
    s = {'x': ((T, NS * g.H), torch.float32), 'pos': ((T,), torch.int64)}
    if any(k in 'gG' for k in kinds):
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
        step = step_shapes(kinds, R_, Q_, P, g)
        wsh, ssh = {}, {}
        for i, k_ in enumerate(kinds):
            pre = f'l{i}.' if len(kinds) > 1 else ''
            wsh.update({pre + n: v for n, v in weight_shapes(k_, g, fp8).items()})
            ssh.update({pre + n: v for n, v in state_shapes(k_, g, R_, pages).items()})
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
                    x, st = visit(kinds[0], t, g, R_, Q_, P, allreduce, moe_mode)
                for n_ in snames:
                    setattr(self, n_.replace('.', '_'), st[n_])
                return x.reshape(R_ * Q_, NS * g.H)
        fn = Block()
        kwargs = {'enable_aliasing': True}
        meta = {'kind': 'inkling_' + kind, 'kinds': kinds, 'R': R_, 'Q': Q_, 'P': P, 'pages': pages, 'inputs': names,
                'states': snames, 'fp8': fp8, 'moe': moe_mode, 'wt': WT, 'route': ROUTE}
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
