"""Qwen3.5 MoE (qwen3_5_moe) graphs for NeuronCores: Gated DeltaNet linear attention, GQA paged attention with output gate,
softmax-router MoE with a shared expert; tensor parallel (TP) with the experts split along their intermediate axis (ETP = TP).

Numerical oracle: qwen35_moe_ref.Ref (f32, validated against HF transformers). Every block mirrors one of its functions;
projection outputs are rounded to bf16 like glm_graphs (EXACT=1 keeps everything f32 for sharding / correctness tests).

A Qwen3.5 layer is  x -> RMSNorm -> {gdn | gqa} -> +x -> RMSNorm -> moe -> +x   (x [T, H] f32, no hyper-connections).
One NEFF per (block kind, bucket) serves every layer of that kind; the runtime binds the layer's weights and state by name.
Buckets: decode R rows x Q = 1, prefill R = 1 x Q tokens of one request. kind layer_<gdn|gqa>_moe fuses both sub-blocks.

Sharding over TP ranks (rank-dependent data are inputs, never graph constants; see docs/neuron-qwen35-moe.md section 3):
  gdn  qkv / conv / z / a / b / A_log / dt_bias by head (key heads Hk/TP, value heads Hv/TP: the query/key -> value head
       repeat is rank-local), out_proj row-parallel + all-reduce. State per layer: kstate f32 [slots, Hv/TP, Dk, Dv],
       cstate f32 [slots, C/TP, K-1] (the last K-1 qkv columns), indexed by a per-row slot, aliased in place.
  gqa  q|gate heads / TP, KV heads / TP (kv_heads must divide by TP: 2 heads, TP = 2), o_proj row-parallel + all-reduce.
       KV cache [pages, 2 (K|V), kvh/TP, 16, D] bf16 per layer, the server's 16-token pages. cos/sin rows from the host.
  moe  router f32 replicated, softmax top-k on the device (one-hot masks). Every expert's gate / up rows and down input
       channels split over the ranks (ETP = TP, EP = 1): MOE=all runs every expert on every token (prefill), MOE=gather
       gathers only the min(top_k * T, E) selected experts' slices (decode). Shared expert split likewise, its sigmoid gate
       replicated. One all-reduce sums shared + routed partials.
Weights: WEIGHTS=fp8 (default) row-scaled E4M3 x E4M3-activation matmuls as glm_graphs.lin (expert down_proj scaled per
input channel so the gathered experts fold into one contraction), WEIGHTS=bf16 for exactness checks. Norm weights are
stored as (1 + w) by the prepare step (Qwen's zero-centred RMSNorm), so the graphs multiply by them directly.

usage: qwen35_graphs.py WORKDIR {gdn|gqa|moe|layer_gdn_moe|layer_gqa_moe} R Q P | head T S
  env: MODEL (dir with config.json) or CONFIG, TP, PAGES (incl. one scratch page), SLOTS (incl. one scratch slot),
       WEIGHTS=fp8|bf16, MOE=all|gather, NCC_FLAGS
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_graphs as G  # noqa: E402  (bf, lin, lin_partial, mm8, F8, PAGE, zeros, count_hlo_parameters)

PAGE = G.PAGE
F8 = G.F8
MOE_DEBUG = os.environ.get('MOE_DEBUG', '')         # device bisection only: the MoE block returns an intermediate (comb|hgu|act|routed|shared)
MOE_SLOTS = os.environ.get('MOE_SLOTS', 'topk')  # gathered-expert slot selection: topk (original) | onehot (no top-k over experts)
EXACT = os.environ.get('EXACT', '0') == '1'   # CPU tests: f32 matmuls, no bf16 rounding (sharding correctness vs the oracle)


class Geo:
    """Text geometry of a qwen3_5_moe config plus the TP layout."""

    def __init__(self, cfg, tp=2):
        c = cfg.get('text_config', cfg)
        self.raw = c
        self.H, self.V, self.L = c['hidden_size'], c['vocab_size'], c['num_hidden_layers']
        self.eps = float(c['rms_norm_eps'])
        self.Hk, self.Dk = c['linear_num_key_heads'], c['linear_key_head_dim']
        self.Hv, self.Dv = c['linear_num_value_heads'], c['linear_value_head_dim']
        self.K = c['linear_conv_kernel_dim']
        self.heads, self.kvh = c['num_attention_heads'], c['num_key_value_heads']
        self.D = c.get('head_dim') or self.H // self.heads
        rp = c.get('rope_parameters') or {}
        self.theta = float(rp.get('rope_theta', c.get('rope_theta', 10000.0)))
        self.rot = int(self.D * float(rp.get('partial_rotary_factor', c.get('partial_rotary_factor', 1.0))))
        self.gate = bool(c.get('attn_output_gate', True))
        self.E, self.topk, self.I = c['num_experts'], c['num_experts_per_tok'], c['moe_intermediate_size']
        self.SI = c.get('shared_expert_intermediate_size', 0) or 0
        self.attn = list(c['layer_types'])
        eos = c.get('eos_token_id', 0)
        self.eos = eos if isinstance(eos, list) else [eos]
        self.max_context = c['max_position_embeddings']
        self.set_parallel(tp)

    def set_parallel(self, tp):
        """TP layout. Heads that are fewer than the ranks are REPLICATED (docs/neuron-qwen397.md): with kv_heads < TP every
        rank holds the one KV head its query heads attend to (kvl = 1, head kv_of(rank)); with linear key heads < TP every
        rank holds the one key head its value heads repeat (hk = 1, head key_of(rank)). The graphs are unchanged: they
        only see per-rank counts hq / kvl / hk / hl."""
        self.TP = tp
        bad = [n for n, v in (('linear value heads', self.Hv), ('attention heads', self.heads),
                              ('expert intermediate', self.I), ('shared intermediate', self.SI or tp)) if v % tp]
        if self.kvh % tp and (tp % self.kvh or self.heads % self.kvh):
            bad.append('kv heads')
        if self.Hk % tp and (tp % self.Hk or self.Hv % self.Hk):
            bad.append('linear key heads')
        if bad:
            raise ValueError(f'geometry not shardable over TP={tp}: ' + ', '.join(bad))
        self.hl = self.Hv // tp                                             # value heads per rank
        self.hk = max(1, self.Hk // tp)                                     # key heads per rank (1 = replicated when Hk < TP)
        self.hq = self.heads // tp
        self.kvl = max(1, self.kvh // tp)                                   # kv heads per rank (1 = replicated when kvh < TP)
        if self.hl % self.hk or self.hq % self.kvl:
            raise ValueError(f'geometry not shardable over TP={tp}: value heads per key head / q heads per kv head')
        self.W3 = 2 * self.hk * self.Dk + self.hl * self.Dv                # conv channels per rank (q | k | v)
        self.Wv = self.hl * self.Dv
        self.EIl, self.SIl = self.I // tp, self.SI // tp
        self.VP = (self.V + tp - 1) // tp * tp

    def key_of(self, rank):
        """first linear key head held by this rank (hk heads from there)."""
        return rank * self.hl // (self.Hv // self.Hk)

    def kv_of(self, rank):
        """first KV head held by this rank (kvl heads from there)."""
        return rank * self.hq // (self.heads // self.kvh)

    def kind(self, l):
        return 'gdn' if self.attn[l] == 'linear_attention' else 'gqa'

    def meta(self):
        return {'arch': 'qwen3_5_moe', 'hidden': self.H, 'vocab': self.V, 'layers': self.L, 'heads': self.heads, 'kv_heads': self.kvh,
                'head_dim': self.D, 'rot': self.rot, 'linear_key_heads': self.Hk, 'linear_value_heads': self.Hv, 'linear_dim': self.Dk,
                'conv': self.K, 'experts': self.E, 'top_k': self.topk, 'expert_intermediate': self.I, 'shared_intermediate': self.SI,
                'tp': self.TP, 'ep': 1, 'page': PAGE}


def geo_from_env():
    tp = int(os.environ.get('TP', '2'))
    path = os.environ.get('CONFIG') or str(Path(os.environ['MODEL']) / 'config.json')
    return Geo(json.loads(Path(path).read_text()), tp)


# ---------------------------------------------------------------------------------------------------------------- math
def bf(x):
    return x if EXACT else G.bf(x)


def lin(h, t, name):
    if EXACT:
        return torch.matmul(h.reshape(-1, h.shape[-1]), t[name].float().T).reshape(*h.shape[:-1], t[name].shape[0])
    return G.lin(h, t, name)


def lin_partial(h, t, name):
    if EXACT:
        return torch.matmul(h, t[name].float().T)
    return G.lin_partial(h, t, name)


def rms(x, w, eps):
    """w already holds (1 + weight)."""
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * w


def cos_sin_rows(pos, rot, theta):
    """Host side: f64 cos | sin rows [T, 2 * rot] for absolute positions (HF rotate_half layout, emb = cat(f, f))."""
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    f = pos.to(torch.float64).reshape(-1, 1) * inv.reshape(1, -1)
    emb = torch.cat((f, f), dim=-1)
    return torch.cat((emb.cos(), emb.sin()), dim=-1).float()


# ---------------------------------------------------------------------------------------------------------------- blocks
def gdn_block(n, t, g, R, Q, allreduce):
    """Gated DeltaNet on this rank's heads. n [T, H] normalised input (T = R * Q, one request per row r, its Q tokens in
    order). State slot per row t['slot'] [R]; a row whose first token is at position 0 starts from zero state. Padding
    tokens of a short prefill chunk have position -1 and leave the state untouched (same contract as glm_graphs.kda_block)."""
    T, hl, hk, Dk, Dv, W, K = R * Q, g.hl, g.hk, g.Dk, g.Dv, g.W3, g.K
    slot = t['slot']
    keep = (t['pos'].reshape(R, Q)[:, 0] != 0).float()
    proj = lin(n, t, 'qkv').reshape(R, Q, W)                                           # [R, Q, W]
    cs = t['cstate'][slot] * keep.reshape(R, 1, 1)                                     # [R, W, K-1], oldest first
    seq = torch.cat((cs.permute(0, 2, 1), proj), dim=1)                                # [R, K-1+Q, W]
    cw = t['conv']                                                                     # [W, K] f32
    conv = sum(seq[:, j:j + Q] * cw[:, j].reshape(1, 1, W) for j in range(K))          # [R, Q, W]
    valid = t['pos'].reshape(R, Q) >= 0
    if K > 1:                                                                          # state = the K-1 inputs before the first padding token
        nvalid = valid.long().sum(dim=1)
        idx = (nvalid.reshape(R, 1) + torch.arange(K - 1, device=n.device).reshape(1, K - 1)).reshape(R, K - 1, 1).expand(R, K - 1, W)
        cstate = t['cstate'].index_put((slot,), torch.gather(seq, 1, idx).permute(0, 2, 1))
    else:
        cstate = t['cstate']
    act = bf(conv * torch.sigmoid(conv))
    kd = hk * Dk
    q = act[..., :kd].reshape(R, Q, hk, Dk)
    k = act[..., kd:2 * kd].reshape(R, Q, hk, Dk)
    v = act[..., 2 * kd:].reshape(R, Q, hl, Dv)
    rep = hl // hk
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    qn = q * (torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6) / math.sqrt(Dk))
    kn = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(lin(n, t, 'b')).reshape(R, Q, hl)
    a = lin(n, t, 'a').reshape(R, Q, hl)
    gg = -torch.exp(t['alog']).reshape(1, 1, hl) * torch.nn.functional.softplus(a + t['dt'].reshape(1, 1, hl))
    decay = torch.exp(gg)                                                              # [R, Q, hl]
    st = t['kstate'][slot] * keep.reshape(R, 1, 1, 1)                                 # [R, hl, Dk, Dv]
    outs = []
    for i in range(Q):                                                                 # the recurrence, token by token
        nst = st * decay[:, i].reshape(R, hl, 1, 1)
        mem = torch.einsum('rhkd,rhk->rhd', nst, kn[:, i])
        delta = (v[:, i] - mem) * beta[:, i].unsqueeze(-1)
        nst = nst + kn[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append(torch.einsum('rhkd,rhk->rhd', nst, qn[:, i]))
        st = torch.where(valid[:, i].reshape(R, 1, 1, 1), nst, st) if Q > 1 else nst
    kstate = t['kstate'].index_put((slot,), st)
    y = torch.stack(outs, dim=1)                                                       # [R, Q, hl, Dv]
    z = lin(n, t, 'z').reshape(R, Q, hl, Dv)
    y = y * torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + g.eps) * t['onorm'] * (z * torch.sigmoid(z))
    out = allreduce(lin_partial(bf(y).reshape(T, g.Wv), t, 'o'))
    return out, {'kstate': kstate, 'cstate': cstate}


def gqa_block(n, t, g, R, Q, P, allreduce):
    """GQA attention with per-head q/k RMSNorm, partial RoPE and the sigmoid output gate over the request's paged keys.
    t['cos_sin'] [T, 2 rot] host rows; t['wslot'] [T] flat write slots (page * 16 + offset); t['table'] [R, P]."""
    T, hq, kvl, D, rot = R * Q, g.hq, g.kvl, g.D, g.rot
    qg = lin(n, t, 'q')
    if g.gate:
        qg = qg.reshape(T, hq, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
    else:
        q, gate = qg.reshape(T, hq, D), None
    k = lin(n, t, 'k').reshape(T, kvl, D)
    v = lin(n, t, 'v').reshape(T, kvl, D)
    q = rms(q, t['qn'], g.eps)
    k = rms(k, t['kn'], g.eps)
    cos, sin = t['cos_sin'][:, :rot].unsqueeze(1), t['cos_sin'][:, rot:].unsqueeze(1)
    h = rot // 2

    def rope(x):
        xr, xp = x[..., :rot], x[..., rot:]
        rh = torch.cat((-xr[..., h:], xr[..., :h]), dim=-1)
        return torch.cat((xr * cos + rh * sin, xp), dim=-1)
    q, k = bf(rope(q)), bf(rope(k))
    cache = t['kv']                                                                    # [pages, 2, kvl, 16, D]
    # scatter by (page, K|V, head, offset) index tensors, as trace_model.py: no permute / reshape of the aliased cache
    page = torch.bitwise_right_shift(t['wslot'], 4)
    off = torch.bitwise_and(t['wslot'], PAGE - 1)
    at = (page[:, None, None], torch.arange(2, dtype=torch.int64, device=n.device)[None, :, None],
          torch.arange(kvl, dtype=torch.int64, device=n.device)[None, None, :], off[:, None, None])
    cache = cache.index_put(at, torch.stack((k, bf(v)), dim=1).to(cache.dtype))         # [T, 2, kvl, D]
    table = t['table']                                                                 # [R, P]
    keys = P * PAGE
    kv = cache[table].float()                                                          # [R, P, 2, kvl, 16, D]
    kk = kv[:, :, 0].permute(0, 2, 1, 3, 4).reshape(R, kvl, keys, D)
    vv = kv[:, :, 1].permute(0, 2, 1, 3, 4).reshape(R, kvl, keys, D)
    rep = hq // kvl
    qh = q.reshape(R, Q, kvl, rep, D)
    s = torch.einsum('rqhgd,rhkd->rqhgk', qh, kk) / math.sqrt(D)
    kpos = torch.arange(keys, dtype=torch.int64, device=n.device).reshape(1, 1, 1, 1, keys)
    visible = kpos <= t['pos'].reshape(R, Q, 1, 1, 1)
    s = torch.where(visible, s, torch.full_like(s, -3e38))
    p = bf(torch.softmax(s, dim=-1))
    o = torch.einsum('rqhgk,rhkd->rqhgd', p, vv).reshape(T, hq, D)
    if gate is not None:
        o = o * torch.sigmoid(gate)
    out = allreduce(lin_partial(bf(o).reshape(T, hq * D), t, 'o'))
    return out, {'kv': cache}


def route(n, t, g):
    """Softmax over all experts, top-k, renormalised -> combine weights [T, E] (0 for unselected experts)."""
    probs = torch.softmax(torch.matmul(n.float(), t['rw'].T), dim=-1)                  # [T, E] f32
    T = n.shape[0]
    experts = torch.arange(g.E, device=n.device)
    top = torch.topk(probs, g.topk, dim=-1)[1]                                         # [T, K]
    onehot = (top.unsqueeze(-1) == experts.reshape(1, 1, g.E)).float()                 # [T, K, E]  (one-hot: see glm_graphs.route)
    w = torch.einsum('tke,te->tk', onehot, probs)
    w = w / w.sum(dim=-1, keepdim=True)
    return torch.einsum('tke,tk->te', onehot, w)


def moe_block(n, t, g, allreduce, mode='all'):
    """Shared expert (TP-split, sigmoid-gated) + this rank's slices of the selected experts (glm_graphs.moe_block layout)."""
    T, E, EIl = n.shape[0], g.E, g.EIl
    sgate = torch.sigmoid(torch.matmul(n.float(), t['sgate'].reshape(1, -1).T))        # [T, 1] replicated
    shared = lin_partial(bf(silu_mul(lin(n, t, 'sg'), lin(n, t, 'su'))), t, 'sd') * sgate
    comb = route(n, t, g)                                                              # [T, E]
    gu, gus, dw, dc = t['egu'], t.get('egu_s'), t['ed'], t.get('ed_c')                 # [E, 2EIl, H], [E, EIl, H]
    if mode == 'gather' and g.topk * T < E:
        slots = g.topk * T
        used = (comb > 0).float().amax(dim=0, keepdim=True)                            # [1, E]
        ar = torch.arange(E, device=n.device).float().reshape(1, E)
        if MOE_SLOTS == 'onehot':
            # slot s <- the s-th used expert (ascending), empty slots <- expert 0 with combine weight 0. No top-k over the
            # experts: the device's topk(k = 40 of 512) of the 4-row bucket returned non-finite output on trn1 (397B doc).
            # (written with masks and sums only: the matmul forms hit NCC_ITCT901 in neuronx-cc 2.26)
            before = (ar.reshape(E, 1) >= ar.reshape(1, E)).float()                     # [e, e'] = 1 when e' <= e
            order = (used * before).sum(dim=-1).reshape(1, E) - 1.0                     # rank of expert e among the used ones
            sel = ((order == torch.arange(slots, device=n.device).float().reshape(slots, 1)) & (used > 0)).float()   # [slots, E]
            idx = (sel * ar).sum(dim=-1).round().long()                                 # [slots]
            comb = (comb.unsqueeze(1) * sel.unsqueeze(0)).sum(dim=-1)                   # [T, slots], 0 for empty slots
        else:
            rank = used - ar * (0.5 / E)
            idx = torch.topk(rank, slots, dim=-1)[1].reshape(slots)                     # (2-D topk: a 1-D one crashed the XLA trace)
            comb = torch.index_select(comb, 1, idx)
        gu, dw = torch.index_select(gu, 0, idx), torch.index_select(dw, 0, idx)
        gus = torch.index_select(gus, 0, idx) if gus is not None else None
        dc = torch.index_select(dc, 0, idx) if dc is not None else None
    E_ = gu.shape[0]

    def dbg(v):   # an intermediate as the block output: first H columns, zero-padded
        v = v.reshape(T, -1).float()[:, :g.H]
        return allreduce(torch.nn.functional.pad(v, (0, g.H - v.shape[1]))), {}
    if MOE_DEBUG == 'comb':
        return dbg(comb)
    if MOE_DEBUG == 'shared':
        return dbg(shared)
    tt = {'w': gu.reshape(E_ * 2 * EIl, g.H)}
    if gus is not None:
        tt['w_s'] = gus.reshape(-1)
    hgu = lin(n, tt, 'w').reshape(T, E_, 2, EIl)
    if MOE_DEBUG == 'hgu':
        return dbg(hgu)
    act = bf(silu_mul(hgu[:, :, 0], hgu[:, :, 1])) * comb.unsqueeze(-1)
    if dc is not None:
        act = act * dc.reshape(1, E_, EIl)
    a2 = act.reshape(T, E_ * EIl)
    if MOE_DEBUG == 'act':
        return dbg(a2)
    wd = dw.reshape(E_ * EIl, g.H)
    if dc is not None:
        a = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        routed = G.mm8((a2 / a).to(F8), wd).float() * a
    elif EXACT:
        routed = torch.matmul(a2, wd.float())
    else:
        routed = torch.matmul(a2.to(torch.bfloat16), wd.to(torch.bfloat16)).float()
    if MOE_DEBUG == 'routed':
        return dbg(routed)
    return allreduce(shared + routed), {}


def silu_mul(gate, up):
    return gate * torch.sigmoid(gate) * up


def sublayer(kind, t, g, R, Q, P, allreduce, moe_mode='all'):
    """One sub-block with its pre-norm and residual -> (x_new, updated states)."""
    x = t['x']
    n = bf(rms(x, t['norm'], g.eps))
    if kind == 'gdn':
        y, st = gdn_block(n, t, g, R, Q, allreduce)
    elif kind == 'gqa':
        y, st = gqa_block(n, t, g, R, Q, P, allreduce)
    elif kind == 'moe':
        y, st = moe_block(n, t, g, allreduce, moe_mode)
    else:
        raise ValueError(kind)
    return x + bf(y), st


def head(t, g, rows):
    x = torch.index_select(t['x'], 0, rows)
    h = bf(rms(x, t['norm'], g.eps))
    if 'w_s' in t:
        a = torch.amax(torch.abs(h), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        return G.mm8((h / a).to(F8), t['w'].T).float() * a * t['w_s'].reshape(1, -1)
    if EXACT:
        return torch.matmul(h, t['w'].float().T)
    return torch.matmul(h.to(torch.bfloat16), t['w'].T).float()


# ---------------------------------------------------------------------------------------------------------------- shapes
def weight_shapes(kind, g, fp8=True):
    """name -> (shape, dtype) of a sub-block's per-layer inputs for one rank (the prepare step writes exactly these)."""
    H, f32, b16 = g.H, torch.float32, torch.bfloat16
    wt = F8 if fp8 else b16
    s = {'norm': ((H,), f32)}

    def proj(name, out, inp):
        s[name] = ((out, inp), wt)
        if fp8:
            s[name + '_s'] = ((out,), f32)
    if kind == 'gdn':
        proj('qkv', g.W3, H); proj('z', g.Wv, H); proj('b', g.hl, H); proj('a', g.hl, H); proj('o', H, g.Wv)
        s['conv'] = ((g.W3, g.K), f32); s['alog'] = ((g.hl,), f32); s['dt'] = ((g.hl,), f32); s['onorm'] = ((g.Dv,), f32)
    elif kind == 'gqa':
        proj('q', g.hq * g.D * (2 if g.gate else 1), H); proj('k', g.kvl * g.D, H); proj('v', g.kvl * g.D, H); proj('o', H, g.hq * g.D)
        s['qn'] = ((g.D,), f32); s['kn'] = ((g.D,), f32)
    elif kind == 'moe':
        proj('sg', g.SIl, H); proj('su', g.SIl, H); proj('sd', H, g.SIl)
        s['sgate'] = ((H,), f32); s['rw'] = ((g.E, H), f32)
        s['egu'] = ((g.E, 2 * g.EIl, H), wt); s['ed'] = ((g.E, g.EIl, H), wt)
        if fp8:
            s['egu_s'] = ((g.E, 2 * g.EIl), f32); s['ed_c'] = ((g.E, g.EIl), f32)
    return s


def state_shapes(kind, g, pages, slots):
    if kind == 'gdn':
        return {'kstate': ((slots, g.hl, g.Dk, g.Dv), torch.float32), 'cstate': ((slots, g.W3, g.K - 1), torch.float32)}
    if kind == 'gqa':
        return {'kv': ((pages, 2, g.kvl, PAGE, g.D), torch.float32 if EXACT else torch.bfloat16)}
    return {}


def step_shapes(kind, R, Q, P, g):
    T = R * Q
    s = {'x': ((T, g.H), torch.float32)}
    if kind in ('gdn', 'gqa'):
        s['pos'] = ((T,), torch.int64)
    if kind == 'gdn':
        s['slot'] = ((R,), torch.int64)
    if kind == 'gqa':
        s['cos_sin'] = ((T, 2 * g.rot), torch.float32); s['wslot'] = ((T,), torch.int64); s['table'] = ((R, P), torch.int64)
    return s


zeros = G.zeros


# ---------------------------------------------------------------------------------------------------------------- trace
def main():
    workdir, kind = sys.argv[1], sys.argv[2]
    g = geo_from_env()
    fp8 = os.environ.get('WEIGHTS', 'fp8') == 'fp8'
    moe_mode = os.environ.get('MOE', 'all')
    pages, slots = int(os.environ.get('PAGES', '65')), int(os.environ.get('SLOTS', '9'))
    import torch_neuronx
    import torch_xla.core.xla_model as xm
    os.environ.setdefault('XLA_ALWAYS_ALLREDUCE', '1')
    groups = [list(range(g.TP))]
    allreduce = (lambda v: xm.all_reduce(xm.REDUCE_SUM, v, groups=groups)) if g.TP > 1 else (lambda v: v)
    if kind == 'head':
        T, S = int(sys.argv[3]), int(sys.argv[4])
        names = ['x', 'rows', 'norm', 'w'] + (['w_s'] if fp8 else [])
        shp = {'x': ((T, g.H), torch.float32), 'rows': ((S,), torch.int64), 'norm': ((g.H,), torch.float32),
               'w': ((g.VP // g.TP, g.H), F8 if fp8 else torch.bfloat16), 'w_s': ((g.VP // g.TP,), torch.float32)}
        inputs = [zeros(*shp[n]) for n in names]
        fn = lambda *a: head(dict(zip(names, a)), g, a[1])
        meta = {'kind': 'qwen35_head', 'T': T, 'S': S, 'inputs': names, 'states': [], 'fp8': fp8}
        kwargs = {}
    else:
        R, Q, P = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        parts = kind.split('_')[1:] if kind.startswith('layer_') else [kind]
        step, wsh = {}, {}
        for i, part in enumerate(parts):
            step.update(step_shapes(part, R, Q, P, g))
            pre = ('a.', 'm.')[i] if len(parts) > 1 else ''
            wsh.update({pre + k: v for k, v in weight_shapes(part, g, fp8).items()})
        ssh = state_shapes(parts[0], g, pages, slots)
        names = list(step) + list(wsh)
        snames = list(ssh)
        inputs = [zeros(*step[n]) for n in step] + [zeros(*wsh[n]) for n in wsh]
        for n_ in names:   # scales of 1 keep the fp8 maths finite during the trace
            if n_.endswith('_s') or n_.endswith('_c'):
                inputs[names.index(n_)] = torch.ones(*wsh[n_][0])

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for n_ in snames:
                    self.register_buffer(n_, zeros(*ssh[n_]))

            def forward(self, *a):
                t = dict(zip(names, a))
                for n_ in snames:
                    t[n_] = getattr(self, n_)
                if len(parts) == 1:
                    x, st = sublayer(kind, t, g, R, Q, P, allreduce, moe_mode)
                else:
                    sub = lambda pre: {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
                    common = {k: v for k, v in t.items() if '.' not in k}
                    x, st = sublayer(parts[0], {**common, **sub('a.')}, g, R, Q, P, allreduce, moe_mode)
                    x, _ = sublayer(parts[1], {**common, **sub('m.'), 'x': x}, g, R, Q, P, allreduce, moe_mode)
                for n_ in snames:
                    setattr(self, n_, st[n_])   # mutated buffers: aliased input -> output (in place on the device)
                return x
        fn = Block()
        kwargs = {'enable_aliasing': True} if snames else {}
        meta = {'kind': 'qwen35_' + kind, 'R': R, 'Q': Q, 'P': P, 'pages': pages, 'slots': slots, 'inputs': names,
                'states': snames, 'fp8': fp8, 'moe': moe_mode, 'moe_slots': MOE_SLOTS}
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
