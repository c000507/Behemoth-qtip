"""GLM-5.3-Flash (glm5_next) graphs for NeuronCores: KDA linear attention, sparse MLA with pooled index keys, mHC,
dense SwiGLU and MoE routing + expert matmuls, tensor parallel (TP) with expert parallelism (EP) for the routed experts.

The numerical oracle is glm53/src/model.cpp (the CPU reference); every block below mirrors one of its functions and
rounds to bf16 at the same points where that is cheap (projection outputs, the residual after each mHC post).

A GLM layer is two sub-blocks, each wrapped in its own mHC pre/post (hyper-connections, `hc_mult` = S residual streams):
    x [T, S*H] f32 (replicated) -> hyper_pre -> RMSNorm -> {kda | mla} -> hyper_post -> x
    x                           -> hyper_pre -> RMSNorm -> {dense | moe} -> hyper_post -> x
One NEFF per (block kind, bucket) serves every layer of that kind (the runtime binds the layer's weights and state by
name, glm_runtime / cpp/glm_model.cpp). Buckets: decode R rows x Q=1, prefill R=1 x Q tokens of one request.

Sharding over TP ranks (every rank runs the same NEFF; rank-dependent data are inputs, never graph constants):
  KDA   q/k/v, conv, f_b, g_b, b, A_log, dt_bias by linear head (column-parallel), f_a / g_a replicated (128 rows),
        o_proj row-parallel + all-reduce. Recurrent state [slots, heads/TP, D, D] f32 and conv state
        [slots, 3, width/TP, K-1] f32 per layer, indexed by a per-row state slot, aliased in place.
  MLA   q_a, kv_a, the whole indexer replicated (small; every rank computes the same top-k selection), q_b and kv_b by
        head, o_proj row-parallel + all-reduce. Latent cache [pages, 16, kv_rank] (bf16, or E4M3 + one f32 scale per
        token with KV=fp8) and index cache [pages, 16, 2 * index_dim] bf16 (pooling keys | pooling gates) replicated,
        paged exactly like the server's 16-token pages.
  dense gate/up column-parallel, down row-parallel + all-reduce.
  MoE   router (f32) replicated, top-k on the device. Routed experts: EP groups of TP/ETP ranks hold E/EP experts each,
        every expert's intermediate split over the ETP ranks of its group (EP x ETP = TP). The shared expert is split
        over all TP ranks. One all-reduce sums shared + routed partials.
        MOE=all    every local expert runs on every token (static, best for wide batches / prefill),
        MOE=gather only the selected local experts' weights are gathered (slots = min(E/EP, top_k * T)), exact,
                   best for decode at small EP (bytes read per core: slots * expert / ETP).
Weights: WEIGHTS=fp8 (default) row-scaled E4M3 x E4M3-activation matmuls like trace_model.py (routed expert down_proj
is scaled per input channel so the experts' contraction folds into one 2D matmul), WEIGHTS=bf16 for exactness checks.

usage: glm_graphs.py WORKDIR {kda|mla|dense|moe|layer_<kda|mla>_<dense|moe>} R Q P | head T S
  (layer_*: a whole layer, attention + MLP sub-blocks, in one NEFF: half the launches; weights named a.* / m.*)
  env: MODEL (checkpoint dir with config.json) or CONFIG, TP, EP, PAGES (incl. one scratch page), SLOTS (KDA state
       slots incl. one scratch slot), WEIGHTS=fp8|bf16, KV=bf16|fp8, MOE=all|gather, NCC_FLAGS
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

PAGE = 16
F8 = torch.float8_e4m3fn


class Geo:
    """Text geometry of a glm5_next config plus the parallel layout."""

    def __init__(self, cfg, tp=1, ep=None):
        c = cfg.get('text_config', cfg)
        self.raw = c
        self.H, self.S, self.FF, self.V, self.L = c['hidden_size'], c['hc_mult'], c['intermediate_size'], c['vocab_size'], c['num_hidden_layers']
        self.heads, self.qrank, self.kvrank = c['num_attention_heads'], c['q_lora_rank'], c['kv_lora_rank']
        self.qdim, self.vdim = c['qk_nope_head_dim'], c['v_head_dim']
        la = c['linear_attn_config']
        self.lheads, self.ldim, self.conv = la['num_heads'], la['head_dim'], la['short_conv_kernel_size']
        self.glow = float(la.get('gate_lower_bound', -5.0))
        self.sinkhorn, self.hc_eps, self.eps = c['hc_sinkhorn_iters'], float(c['hc_eps']), float(c['rms_norm_eps'])
        self.iheads, self.idim, self.pool, self.topk = c['index_n_heads'], c['index_head_dim'], c['index_kpool'], c['index_topk']
        self.E, self.K = c.get('n_routed_experts', 0) or 0, c.get('num_experts_per_tok', 0) or 0
        self.EI, self.nshared = c.get('moe_intermediate_size', 0) or 0, c.get('n_shared_experts', 0) or 0
        self.groups, self.top_groups = c.get('n_group', 1) or 1, c.get('topk_group', 1) or 1
        self.norm_topk, self.rscale = bool(c.get('norm_topk_prob', True)), float(c.get('routed_scaling_factor', 1.0))
        self.limit = float(c.get('swiglu_limit', 10.0))
        self.attn = list(c['layer_types'])
        mlp = c.get('mlp_layer_types') or ['dense' if l < c.get('first_k_dense_replace', self.L) else 'sparse' for l in range(self.L)]
        self.mlp = ['moe' if m == 'sparse' else 'dense' for m in mlp]
        self.indexers = c.get('indexer_types') or ['full'] * self.L
        if any(i != 'full' for i in self.indexers):
            raise ValueError('shared indexer layers are not implemented on Neuron (both GLM-5.3 checkpoints use full indexers)')
        if c.get('qk_rope_head_dim', 0) or not c.get('mla_use_nope', True):
            raise ValueError('only NoPE MLA is supported')
        eos = c['eos_token_id']
        self.eos = eos if isinstance(eos, list) else [eos]
        self.max_context = c['max_position_embeddings']
        self.set_parallel(tp, ep)

    def set_parallel(self, tp, ep=None):
        self.TP = tp
        self.EP = ep or tp
        if tp % self.EP:
            raise ValueError(f'EP={self.EP} must divide TP={tp}')
        self.ETP = tp // self.EP
        bad = [n for n, v in (('linear heads', self.lheads), ('MLA heads', self.heads), ('intermediate', self.FF)) if v % tp]
        if self.E:
            if self.E % self.EP: bad.append('routed experts / EP')
            if self.EI % self.ETP: bad.append('expert intermediate / ETP')
            if (self.EI * self.nshared) % tp: bad.append('shared expert intermediate / TP')
        if bad:
            raise ValueError(f'geometry not shardable over TP={tp} EP={self.EP}: ' + ', '.join(bad))
        self.hl, self.wl = self.lheads // tp, self.lheads * self.ldim // tp    # KDA heads / width per rank
        self.ml = self.heads // tp                                             # MLA heads per rank
        self.ffl = self.FF // tp
        self.El = self.E // self.EP if self.E else 0                           # routed experts per EP group
        self.EIl = self.EI // self.ETP if self.E else 0                        # expert intermediate per rank
        self.SIl = self.EI * self.nshared // tp if self.E else 0               # shared expert intermediate per rank
        self.VP = (self.V + tp - 1) // tp * tp
        self.NC = self.S * (self.S + 2)

    def expert_ids(self, rank):
        g = rank // self.ETP
        return list(range(g * self.El, (g + 1) * self.El))

    def meta(self):
        return {'arch': 'glm5_next', 'hidden': self.H, 'streams': self.S, 'intermediate': self.FF, 'vocab': self.V,
                'layers': self.L, 'heads': self.heads, 'linear_heads': self.lheads, 'linear_dim': self.ldim,
                'kv_rank': self.kvrank, 'q_rank': self.qrank, 'experts': self.E, 'expert_intermediate': self.EI,
                'tp': self.TP, 'ep': self.EP, 'page': PAGE}


def geo_from_env():
    tp = int(os.environ.get('TP', '2'))
    ep = int(os.environ['EP']) if os.environ.get('EP') else None
    path = os.environ.get('CONFIG') or str(Path(os.environ['MODEL']) / 'config.json')
    return Geo(json.loads(Path(path).read_text()), tp, ep)


# ---------------------------------------------------------------------------------------------------------------- math
def bf(x):
    """Round to bf16, keep f32 (the oracle's round())."""
    return x.to(torch.bfloat16).float()


def rms(x, w, eps):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * w


def mm8(a8, b8):
    """fp8 x fp8 matmul. On the device torch.matmul runs it on the tensor engine; on the CPU (the mirror used by the
    tests) torch would round the result to fp8, so the codes are widened exactly and accumulated in f32."""
    if a8.device.type == 'cpu':
        return torch.matmul(a8.float(), b8.float())
    return torch.matmul(a8, b8)


def lin(h, t, name):
    """h [..., in] -> [..., out] f32 rounded to bf16. FP8 weights (t[name+'_s'] present): the activation row is scaled
    into E4M3 (amax / 224), the fp8 x fp8 matmul accumulates in f32, then both scales are applied. Otherwise bf16."""
    w = t[name]
    shape = h.shape
    h2 = h.reshape(-1, shape[-1])
    if name + '_s' in t:
        a = torch.amax(torch.abs(h2.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        y = mm8((h2.float() / a).to(F8), w.T).float() * a * t[name + '_s'].reshape(1, -1)
    else:
        y = torch.matmul(h2.to(torch.bfloat16), w.to(torch.bfloat16).T).float()
    return bf(y).reshape(*shape[:-1], w.shape[0])


def silu_limited(gate, up, limit):
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, -limit, limit)
    return gate * torch.sigmoid(gate) * up


def hyper_pre(x, t, g):
    """mHC pre (glm53 hyper_pre): x [T, S*H] -> collapsed [T, H], post [T, S], mixing [T, S, S] (bf16-rounded like the oracle)."""
    n, S = g.H, g.S
    T = x.shape[0]
    normalized = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + g.eps)
    coeff = torch.matmul(normalized, t['hc_fn'].T)                                     # [T, S(S+2)] f32
    sc, base = t['hc_scale'], t['hc_base']
    pre = torch.sigmoid(coeff[:, :S] * sc[0] + base[:S]) + g.hc_eps                   # [T, S]
    post = 2 * torch.sigmoid(coeff[:, S:2 * S] * sc[1] + base[S:2 * S])
    collapsed = torch.einsum('ts,tsh->th', pre, x.reshape(T, S, n))
    logits = (coeff[:, 2 * S:] * sc[2] + base[2 * S:]).reshape(T, S, S)
    mix = torch.softmax(logits, dim=-1) + g.hc_eps
    for it in range(g.sinkhorn):
        if it:
            mix = mix / (mix.sum(dim=-1, keepdim=True) + g.hc_eps)                    # rows
        mix = mix / (mix.sum(dim=-2, keepdim=True) + g.hc_eps)                        # columns
    return collapsed, bf(post), bf(mix)


def hyper_post(x, y, post, mix, g):
    """out_i = post_i * y + sum_j mix[j, i] * x_j, rounded to bf16 (the oracle's residual)."""
    T = x.shape[0]
    xs = x.reshape(T, g.S, g.H)
    out = post.unsqueeze(-1) * y.unsqueeze(1) + torch.einsum('tji,tjh->tih', mix, xs)
    return bf(out.reshape(T, g.S * g.H))


# ---------------------------------------------------------------------------------------------------------------- blocks
def kda_block(n, t, g, R, Q, allreduce):
    """KDA (glm53 Model::attention, linear_attention) on this rank's heads. n [T, H] normalized input (T = R * Q, one
    request per row r, its Q tokens in order). State slot per row: t['slot'] [R]; a row whose first token is at position
    0 starts from zero state (new or re-prefilled request), so the host never clears state. A prefill chunk shorter than
    the bucket ends with padding tokens at position -1, which leave the state untouched."""
    T, hl, D, W, K = R * Q, g.hl, g.ldim, g.wl, g.conv
    slot = t['slot']
    keep = (t['pos'].reshape(R, Q)[:, 0] != 0).float()                                # 0: start from zero state
    proj = torch.stack([lin(n, t, 'q'), lin(n, t, 'k'), lin(n, t, 'v')], dim=1).reshape(R, Q, 3, W)
    cs = t['cstate'][slot] * keep.reshape(R, 1, 1, 1)                                 # [R, 3, W, K-1], oldest first
    seq = torch.cat((cs.permute(0, 3, 1, 2), proj), dim=1)                             # [R, K-1+Q, 3, W]
    w = t['conv']                                                                      # [3, W, K]
    conv = sum(seq[:, j:j + Q] * w[:, :, j].reshape(1, 1, 3, W) for j in range(K))     # [R, Q, 3, W]
    valid = t['pos'].reshape(R, Q) >= 0                                                # padding tokens (pos -1) end a chunk
    if K > 1:                                                                          # state = the K-1 inputs before the first padding token
        nvalid = valid.long().sum(dim=1)
        idx = (nvalid.reshape(R, 1) + torch.arange(K - 1, device=n.device).reshape(1, K - 1)).reshape(R, K - 1, 1, 1).expand(R, K - 1, 3, W)
        cstate = t['cstate'].index_put((slot,), torch.gather(seq, 1, idx).permute(0, 2, 3, 1))
    else:
        cstate = t['cstate']
    act = bf(conv * torch.sigmoid(conv)).reshape(R, Q, 3, hl, D)
    q, k, v = act[:, :, 0], act[:, :, 1], act[:, :, 2]                                 # [R, Q, hl, D]
    gg = lin(lin(n, t, 'fa'), t, 'fb').reshape(R, Q, hl, D)
    gg = g.glow * torch.sigmoid(torch.exp(t['alog']).reshape(1, 1, hl, 1) * (gg + t['dt'].reshape(1, 1, hl, D)))
    decay = torch.exp(gg)
    beta = bf(torch.sigmoid(lin(n, t, 'b'))).reshape(R, Q, hl)
    qn = q * (torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6) / math.sqrt(D))
    kn = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    st = t['kstate'][slot] * keep.reshape(R, 1, 1, 1)                                 # [R, hl, D(key i), D(value d)]
    outs = []
    for i in range(Q):                                                                 # the recurrence, token by token
        nst = st * decay[:, i].unsqueeze(-1)
        mem = torch.einsum('rhid,rhi->rhd', nst, kn[:, i])
        delta = (v[:, i] - mem) * beta[:, i].unsqueeze(-1)
        nst = nst + kn[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append(torch.einsum('rhid,rhi->rhd', nst, qn[:, i]))
        st = torch.where(valid[:, i].reshape(R, 1, 1, 1), nst, st) if Q > 1 else nst
    kstate = t['kstate'].index_put((slot,), st)
    y = bf(torch.stack(outs, dim=1))                                                   # [R, Q, hl, D]
    gate = lin(lin(n, t, 'ga'), t, 'gb').reshape(R, Q, hl, D)
    y = y * torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + g.eps) * t['onorm'] * torch.sigmoid(gate)
    out = allreduce(lin_partial(bf(y).reshape(T, W), t, 'o'))
    return out, {'kstate': kstate, 'cstate': cstate}


def lin_partial(h, t, name):
    """Row-parallel projection: f32 partial sum (not rounded) to be all-reduced."""
    w = t[name]
    if name + '_s' in t:
        a = torch.amax(torch.abs(h.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        return mm8((h.float() / a).to(F8), w.T).float() * a * t[name + '_s'].reshape(1, -1)
    return torch.matmul(h.to(torch.bfloat16), w.to(torch.bfloat16).T).float()


def mla_block(n, t, g, R, Q, P, allreduce, kv8=False):
    """Sparse MLA with the pooled-key indexer (glm53 Model::attention, deepseek_sparse_attention) on this rank's heads.
    Keys are the request's paged positions; key j of the table is visible to the query at position p iff j <= p and
    (its pool is among the top-k pools of the query's indexer scores, or j is in the incomplete tail pool). When the
    bucket has at most index_topk keys every complete pool is selected, so the mask is plain causal (static)."""
    T, hl, KR, ID, pool = R * Q, g.ml, g.kvrank, g.idim, g.pool
    pos = t['pos']
    qr = bf(rms(lin(n, t, 'qa'), t['qa_norm'], g.eps))
    q = lin(qr, t, 'qb').reshape(R, Q, hl, g.qdim)
    lat = bf(rms(lin(n, t, 'kva'), t['kva_norm'], g.eps))                             # [T, KR]
    ik = lin(n, t, 'ik')
    mu = ik.mean(dim=-1, keepdim=True)
    var = ((ik - mu) ** 2).mean(dim=-1, keepdim=True)
    ik = bf((ik - mu) * torch.rsqrt(var + 1e-6) * t['ikn_w'] + t['ikn_b'])
    gates = lin(n, t, 'igate')
    page, off = torch.bitwise_right_shift(t['wslot'], 4), torch.bitwise_and(t['wslot'], PAGE - 1)
    states = {}
    if kv8:
        sc = torch.amax(torch.abs(lat), dim=-1).clamp(min=1e-20) / 224.0
        states['lat'] = t['lat'].index_put((page, off), (lat / sc.unsqueeze(-1)).to(F8))
        states['lsc'] = t['lsc'].index_put((page, off), sc)
    else:
        states['lat'] = t['lat'].index_put((page, off), lat.to(torch.bfloat16))
    states['idx'] = t['idx'].index_put((page, off), torch.cat((ik, gates), dim=-1).to(torch.bfloat16))
    table = t['table']                                                                  # [R, P]
    keys = P * PAGE
    if kv8:
        latk = (states['lat'][table].float() * states['lsc'][table].unsqueeze(-1)).reshape(R, keys, KR)
    else:
        latk = states['lat'][table].float().reshape(R, keys, KR)
    kpos = torch.arange(keys, dtype=torch.int64, device=n.device)
    qpos = pos.reshape(R, Q, 1)
    visible = kpos.reshape(1, 1, keys) <= qpos                                          # [R, Q, keys]
    if keys > g.topk:
        iq = lin(qr, t, 'iq').reshape(R, Q, g.iheads, ID)
        wts = lin(n, t, 'iw').reshape(R, Q, g.iheads)
        NP = keys // pool
        ix = states['idx'][table].float().reshape(R, NP, pool, 2 * ID)
        prob = bf(torch.softmax(ix[..., ID:] + t['ape'].reshape(1, 1, pool, ID), dim=2))
        pooled = bf(torch.sum(bf(prob * ix[..., :ID]), dim=2))                          # [R, NP, ID]
        dots = torch.einsum('rqhd,rpd->rqhp', iq, pooled) / math.sqrt(ID)
        score = torch.einsum('rqhp,rqh->rqp', torch.relu(dots), wts) / math.sqrt(g.iheads)
        complete = torch.div(qpos + 1, pool, rounding_mode='floor')                     # [R, Q, 1] complete pools
        pidx = torch.arange(NP, dtype=torch.int64, device=n.device).reshape(1, 1, NP)
        valid = pidx < complete
        score = score - pidx.float() * 1e-30                                            # ties (relu zeros) -> lower pool first, as the oracle
        score = torch.where(valid, score, torch.full_like(score, -3e38))
        kth = torch.topk(score, g.topk // pool, dim=-1)[0][..., -1:]                    # k-th best pool score
        chosen = (score >= kth) & valid                                                 # (scores are distinct: tie-break above)
        tail = kpos.reshape(1, 1, keys) >= complete * pool
        visible = visible & (chosen.unsqueeze(-1).expand(R, Q, NP, pool).reshape(R, Q, keys) | tail)
    kvb = t['kvb'].reshape(hl, g.qdim + g.vdim, KR).float()
    qabs = torch.einsum('rqhd,hdj->rqhj', q, kvb[:, :g.qdim])                          # absorbed query [R, Q, hl, KR]
    s = torch.einsum('rqhj,rkj->rqhk', qabs, latk) / math.sqrt(g.qdim)
    s = torch.where(visible.unsqueeze(2), s, torch.full_like(s, -3e38))
    p = bf(torch.softmax(s, dim=-1))
    wl = torch.einsum('rqhk,rkj->rqhj', p, latk)
    o = torch.einsum('rqhj,hdj->rqhd', wl, kvb[:, g.qdim:])                            # [R, Q, hl, vdim]
    out = allreduce(lin_partial(bf(o).reshape(T, hl * g.vdim), t, 'o'))
    return out, states


def dense_block(n, t, g, allreduce):
    act = bf(silu_limited(lin(n, t, 'g'), lin(n, t, 'u'), g.limit))
    return allreduce(lin_partial(act, t, 'd')), {}


def route(n, t, g):
    """noaux_tc routing (glm53 Model::mlp): sigmoid scores, bias-corrected choice, group limit, top-k, normalised
    weights times routed_scaling_factor -> combine weights [T, E] (0 for unselected experts)."""
    scores = torch.sigmoid(torch.matmul(n.float(), t['rw'].T))                         # [T, E] f32 router
    choice = scores + t['rb']
    T = n.shape[0]
    experts = torch.arange(g.E, device=n.device)
    # one-hot masks instead of scatter / gather: torch scatter into [T, E] gave wrong rows on trn1 for T = 16
    # (tests/test_glm_device.py), the comparisons are exact and cheap at E <= 288
    if g.groups > 1:
        per = g.E // g.groups
        top2 = torch.topk(choice.reshape(T, g.groups, per), 2, dim=-1)[0].sum(-1)      # [T, groups]
        gsel = torch.topk(top2, g.top_groups, dim=-1)[1]                               # [T, top_groups]
        allowed = (gsel.unsqueeze(-1) == (experts // per).reshape(1, 1, g.E)).any(dim=1)
        choice = torch.where(allowed, choice, torch.full_like(choice, -3e38))
    top = torch.topk(choice, g.K, dim=-1)[1]                                           # [T, K]
    onehot = (top.unsqueeze(-1) == experts.reshape(1, 1, g.E)).float()                 # [T, K, E]
    w = torch.einsum('tke,te->tk', onehot, scores)
    if g.norm_topk:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return torch.einsum('tke,tk->te', onehot, w * g.rscale)


def moe_block(n, t, g, allreduce, mode='all'):
    """Shared expert (TP-split) + this rank's routed experts (its EP group's experts, its ETP slice of each)."""
    T = n.shape[0]
    shared = lin_partial(bf(silu_limited(lin(n, t, 'sg'), lin(n, t, 'su'), g.limit)), t, 'sd')
    comb = torch.index_select(route(n, t, g), 1, t['eids'])                            # [T, El] local combine weights
    gu, gus, dw, dc = t['egu'], t.get('egu_s'), t['ed'], t.get('ed_c')                 # [El, 2EIl, H], [El, EIl, H]
    if mode == 'gather' and g.K * T < g.El:                                            # else every local expert may be needed
        slots = g.K * T
        used = (comb > 0).float().amax(dim=0)                                          # [El]
        rank = used - torch.arange(g.El, device=n.device).float() * (0.5 / g.El)        # used first, then low ids
        idx = torch.topk(rank, slots)[1]
        comb = torch.index_select(comb, 1, idx)
        gu, dw = torch.index_select(gu, 0, idx), torch.index_select(dw, 0, idx)
        gus = torch.index_select(gus, 0, idx) if gus is not None else None
        dc = torch.index_select(dc, 0, idx) if dc is not None else None
    E_, EIl = gu.shape[0], g.EIl
    flat = gu.reshape(E_ * 2 * EIl, g.H)
    tt = {'w': flat}
    if gus is not None:
        tt['w_s'] = gus.reshape(-1)
    hgu = lin(n, tt, 'w')
    hgu = hgu.reshape(T, E_, 2, EIl)
    act = bf(silu_limited(hgu[:, :, 0], hgu[:, :, 1], g.limit)) * comb.unsqueeze(-1)    # combine weights folded in
    if dc is not None:                                                                  # per-input-channel down scales
        act = act * dc.reshape(1, E_, EIl)
    a2 = act.reshape(T, E_ * EIl)
    wd = dw.reshape(E_ * EIl, g.H)
    if dc is not None:
        a = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        routed = mm8((a2 / a).to(F8), wd).float() * a
    else:
        routed = torch.matmul(a2.to(torch.bfloat16), wd.to(torch.bfloat16)).float()
    return allreduce(shared + routed), {}


def sublayer(kind, t, g, R, Q, P, allreduce, kv8=False, moe_mode='all'):
    """One mHC-wrapped sub-block -> (x_new, updated state dict)."""
    x = t['x']
    collapsed, post, mix = hyper_pre(x, t, g)
    n = bf(rms(bf(collapsed), t['norm'], g.eps))
    if kind == 'kda':
        y, st = kda_block(n, t, g, R, Q, allreduce)
    elif kind == 'mla':
        y, st = mla_block(n, t, g, R, Q, P, allreduce, kv8)
    elif kind == 'dense':
        y, st = dense_block(n, t, g, allreduce)
    elif kind == 'moe':
        y, st = moe_block(n, t, g, allreduce, moe_mode)
    else:
        raise ValueError(kind)
    return hyper_post(x, bf(y), post, mix, g), st


def head(t, g, rows):
    """Final: mean of the streams, RMSNorm, this rank's vocab slice of lm_head (bf16, or E4M3 + row scale)."""
    x = torch.index_select(t['x'], 0, rows)
    hidden = bf(x.reshape(-1, g.S, g.H).sum(dim=1) / g.S)
    h = bf(rms(hidden, t['norm'], g.eps))
    if 'w_s' in t:
        a = torch.amax(torch.abs(h), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        return mm8((h / a).to(F8), t['w'].T).float() * a * t['w_s'].reshape(1, -1)
    return torch.matmul(h.to(torch.bfloat16), t['w'].T).float()


# ---------------------------------------------------------------------------------------------------------------- shapes
def weight_shapes(kind, g, fp8=True, kv8=False):
    """name -> (shape, dtype) of a sub-block's per-layer inputs for this rank (the repacker writes exactly these)."""
    H, f32, b16 = g.H, torch.float32, torch.bfloat16
    wt = F8 if fp8 else b16
    s = {'hc_fn': ((g.NC, g.S * H), f32), 'hc_base': ((g.NC,), f32), 'hc_scale': ((3,), f32), 'norm': ((H,), f32)}

    def proj(name, out, inp):
        s[name] = ((out, inp), wt)
        if fp8:
            s[name + '_s'] = ((out,), f32)
    if kind == 'kda':
        W, D = g.wl, g.ldim
        for n_ in 'qkv':
            proj(n_, W, H)
        s['conv'] = ((3, W, g.conv), f32)
        proj('fa', D, H); proj('fb', W, D); proj('b', g.hl, H); proj('ga', D, H); proj('gb', W, D); proj('o', H, W)
        s['alog'] = ((g.hl,), f32); s['dt'] = ((W,), f32); s['onorm'] = ((D,), f32)
    elif kind == 'mla':
        proj('qa', g.qrank, H); s['qa_norm'] = ((g.qrank,), f32)
        proj('qb', g.ml * g.qdim, g.qrank)
        proj('kva', g.kvrank, H); s['kva_norm'] = ((g.kvrank,), f32)
        s['kvb'] = ((g.ml * (g.qdim + g.vdim), g.kvrank), b16)
        proj('o', H, g.ml * g.vdim)
        proj('iq', g.iheads * g.idim, g.qrank); proj('ik', g.idim, H); proj('igate', g.idim, H); proj('iw', g.iheads, H)
        s['ikn_w'] = ((g.idim,), f32); s['ikn_b'] = ((g.idim,), f32); s['ape'] = ((g.pool, g.idim), f32)
    elif kind == 'dense':
        proj('g', g.ffl, H); proj('u', g.ffl, H); proj('d', H, g.ffl)
    elif kind == 'moe':
        proj('sg', g.SIl, H); proj('su', g.SIl, H); proj('sd', H, g.SIl)
        s['rw'] = ((g.E, H), f32); s['rb'] = ((g.E,), f32); s['eids'] = ((g.El,), torch.int64)
        s['egu'] = ((g.El, 2 * g.EIl, H), wt); s['ed'] = ((g.El, g.EIl, H), wt)
        if fp8:
            s['egu_s'] = ((g.El, 2 * g.EIl), f32); s['ed_c'] = ((g.El, g.EIl), f32)
    return s


# MLA weights only the top-k selection reads (absent from graphs whose bucket has at most index_topk keys)
INDEXER_QUERY = ('iq', 'iq_s', 'iw', 'iw_s', 'ape')


def state_shapes(kind, g, pages, slots, kv8=False):
    if kind == 'kda':
        return {'kstate': ((slots, g.hl, g.ldim, g.ldim), torch.float32), 'cstate': ((slots, 3, g.wl, g.conv - 1), torch.float32)}
    if kind == 'mla':
        s = {'lat': ((pages, PAGE, g.kvrank), F8 if kv8 else torch.bfloat16)}
        if kv8:
            s['lsc'] = ((pages, PAGE), torch.float32)
        s['idx'] = ((pages, PAGE, 2 * g.idim), torch.bfloat16)
        return s
    return {}


def step_shapes(kind, R, Q, P, g):
    """Per-step inputs of a sub-block graph. Only inputs the graph reads: torch_neuronx drops unused ones from the
    NEFF, which would shift the runtime's name -> input index binding."""
    T = R * Q
    s = {'x': ((T, g.S * g.H), torch.float32)}
    if kind in ('kda', 'mla'):
        s['pos'] = ((T,), torch.int64)
    if kind == 'kda':
        s['slot'] = ((R,), torch.int64)
    if kind == 'mla':
        s['wslot'] = ((T,), torch.int64); s['table'] = ((R, P), torch.int64)
    return s


def zeros(shape, dtype):
    return torch.zeros(shape, dtype=torch.float32).to(dtype) if dtype == F8 else torch.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------------------------------------------- trace
def count_hlo_parameters(path):
    """Entry computation parameter count of a serialized HloModuleProto (None when the proto module is unavailable)."""
    try:
        from libneuronxla.proto import hlo_pb2
    except ImportError:
        return None
    module = hlo_pb2.HloModuleProto()
    module.ParseFromString(Path(path).read_bytes())
    entry = next(c for c in module.computations if c.id == module.entry_computation_id)
    return sum(1 for i in entry.instructions if i.opcode == 'parameter')


def main():
    workdir, kind = sys.argv[1], sys.argv[2]
    g = geo_from_env()
    fp8 = os.environ.get('WEIGHTS', 'fp8') == 'fp8'
    kv8 = os.environ.get('KV', 'bf16') == 'fp8'
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
        shp = {'x': ((T, g.S * g.H), torch.float32), 'rows': ((S,), torch.int64), 'norm': ((g.H,), torch.float32),
               'w': ((g.VP // g.TP, g.H), F8 if fp8 else torch.bfloat16), 'w_s': ((g.VP // g.TP,), torch.float32)}
        inputs = [zeros(*shp[n]) for n in names]
        fn = lambda *a: head(dict(zip(names, a)), g, a[1])
        meta = {'kind': 'glm_head', 'T': T, 'S': S, 'inputs': names, 'states': [], 'fp8': fp8}
        kwargs = {}
    else:
        R, Q, P = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        # layer_<attn>_<mlp>: both sub-blocks of a layer in one NEFF (half the launches); weights 'a.*' / 'm.*'
        parts = kind.split('_')[1:] if kind.startswith('layer_') else [kind]
        step, wsh = {}, {}
        for i, part in enumerate(parts):
            step.update(step_shapes(part, R, Q, P, g))
            pre = ('a.', 'm.')[i] if len(parts) > 1 else ''
            w = weight_shapes(part, g, fp8, kv8)
            if part == 'mla' and P * PAGE <= g.topk:   # every key visible: the indexer's query side is not in the graph
                w = {k: v for k, v in w.items() if k not in INDEXER_QUERY}
            wsh.update({pre + k: v for k, v in w.items()})
        ssh = state_shapes(parts[0], g, pages, slots, kv8)
        names = list(step) + list(wsh)
        snames = list(ssh)
        inputs = [zeros(*step[n]) for n in step] + [zeros(*wsh[n]) for n in wsh]
        for n_ in names:
            if n_.endswith('eids'):
                inputs[names.index(n_)] = torch.arange(g.El, dtype=torch.int64)

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
                    x, st = sublayer(kind, t, g, R, Q, P, allreduce, kv8, moe_mode)
                else:
                    sub = lambda pre: {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
                    common = {k: v for k, v in t.items() if '.' not in k}
                    x, st = sublayer(parts[0], {**common, **sub('a.')}, g, R, Q, P, allreduce, kv8, moe_mode)
                    x, _ = sublayer(parts[1], {**common, **sub('m.'), 'x': x}, g, R, Q, P, allreduce, kv8, moe_mode)
                for n_ in snames:
                    setattr(self, n_, st[n_])   # mutated buffers: aliased input -> output (in place on the device)
                return x
        fn = Block()
        kwargs = {'enable_aliasing': True} if snames else {}
        meta = {'kind': 'glm_' + kind, 'R': R, 'Q': Q, 'P': P, 'pages': pages, 'slots': slots, 'inputs': names,
                'states': snames, 'fp8': fp8, 'kv': 'fp8' if kv8 else 'bf16', 'moe': moe_mode}
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
    n_params = count_hlo_parameters(m / 'graph.hlo')
    if n_params is not None and n_params != len(meta['inputs']) + len(meta['states']):
        raise SystemExit(f'the traced graph has {n_params} parameters, meta lists {len(meta["inputs"]) + len(meta["states"])}: '
                         'an input is unused (dropped by the tracer) or the order is off')
    Path(workdir, 'meta.json').write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
