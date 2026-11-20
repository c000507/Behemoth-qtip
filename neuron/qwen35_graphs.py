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

usage: qwen35_graphs.py WORKDIR {gdn|gqa|moe|layer_gdn_moe|layer_gqa_moe|seg_<kinds>|mtp} R Q P | head T S
       VERIFY=1 seg_... 1 Q P: speculative verify bucket (per-position gdn states); mtp 1 Q P: the MTP head (MTP_EXPERTS)
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
ATT_BF16 = os.environ.get('ATT_BF16', '0') == '1'   # gqa: scores / values contracted in bf16 straight from the paged cache (long-context decode lever)
KV8 = os.environ.get('KV8', '0') == '1'      # gqa KV cache as fp8 e4m3 codes + per-token f32 scales (long-context pool: half the bytes)
KV_CODE = os.environ.get('KV_CODE', 'int8')        # KV8 codes: int8 (round(x / amax * 127): 8 bits uniform, ~10x less error than e4m3) | e4m3
ATT_BLOCK = int(os.environ.get('ATT_BLOCK', '0'))  # KV8 attention: key blocks of this many pages, online softmax (0: one block)
ATT_GATHER = os.environ.get('ATT_GATHER', 'index')   # gqa page gather: index (cache[table]) | flat (index_select over the flattened table)
GDN_GATHER = os.environ.get('GDN_GATHER', 'index')   # gdn state read: index (state[slot]) | flat (index_select)
GDN_EINSUM = os.environ.get('GDN_EINSUM', 'mm')       # decode recurrence contractions: mm (einsum -> batched matmul) | elem (multiply + sum)
GDN_CHUNK = int(os.environ.get('GDN_CHUNK', '0'))     # prefill: chunk size (power of two) of the matmul-form recurrence; 0 = token loop
GDN_INV = os.environ.get('GDN_INV', 'rec')            # GDN_CHUNK inverse: rec (per-block recursion, unit_lower_inverse) | diag (block-diagonal
                                                      # doubling with full [C, C] matmuls, every chunk in one batch: gdn_chunks2, docs/neuron-qwen397-prefill.md)
GDN_LOOP = os.environ.get('GDN_LOOP', '')             # prefill token loop: fused = one state contraction with (k | q) per token, no where() (padding: beta = 0, decay = 1)
RLOGIT = os.environ.get('RLOGIT', 'f32')              # router logits matmul: f32 | bf16 (explicit bf16 operands, f32 accumulate)
AR_BF16 = os.environ.get('AR_BF16', '0')           # 1: all-reduce the partial sums in bf16 (halves the collective bytes; prefill lever), moe: MoE only
ROUTE = os.environ.get('ROUTE', 'topk')          # router top-k: topk (torch.topk + one-hot) | rank | rank2 | tour (no topk op: topk_mask)
TOUR_GROUPS = int(os.environ.get('TOUR_GROUPS', '8'))
MOE_SPLIT = int(os.environ.get('MOE_SPLIT', '0'))      # gathered decode MoE: rows in groups of this many, one slot union per group (0: one union)
MOE_SLOTS = os.environ.get('MOE_SLOTS', 'topk')  # gathered-expert slot selection: topk (union of the rows' experts, original) | onehot (no top-k over
#                                                  experts) | row (slot r * K + j = the j-th expert of row r straight from the router's topk: no union)
MOE_BMM = os.environ.get('MOE_BMM', '0') == '1'  # with MOE_SLOTS=row: every row contracts only its own K slots (batched matmul) instead of all rows x all slots
ROUTER = os.environ.get('ROUTER', 'full')        # full: replicated f32 router [E, H] on every rank (8.4 MB read per MoE block) | shard: rank r holds the
#                                                  router columns of its hidden slice ('rws' [E, H / TP], 'rcols' [H / TP]) and the logits are all-reduced
GDN_DEC = os.environ.get('GDN_DEC', '0') == '1'    # decode (Q = 1): conv state = the last K-1 inputs by a static slice (no gather by the valid count)
GDN_FORM = os.environ.get('GDN_FORM', '')          # fused: decode recurrence with one state contraction (see gdn_block)
GDN_SLOT = os.environ.get('GDN_SLOT', '')          # bound: the gdn graph takes each row's state as its own [1, ...] tensor (R = 1: kstate / cstate,
#                                                  R > 1: kstate_<r> / cstate_<r>; no slot / vslot input) and the engine binds the slot's slice
#                                                  of the layer state, so the graph has no dynamic (runtime-indexed) state DMA. Verify buckets
#                                                  (R = 1) also return the states after positions 1..Q-1 as outputs kstate@<i> / cstate@<i>,
#                                                  which the engine binds to the slices of vslot[i] (docs/neuron-reap-nonfinite.md: the
#                                                  dynamic state gather / scatter races on trn1)
ROW_FORM = os.environ.get('ROW_FORM', '')         # probes of MOE_SLOTS=row: full | sort | launder | topk (see moe_routed)
LIN_CSE = os.environ.get('LIN_CSE', '0') == '1'  # share the fp8 activation scaling (amax, divide, cast) between the projections of one input
MOE_KERNEL = os.environ.get('MOE_KERNEL', '')     # dyn: gathered decode MoE through nki_moe_dyn.moe_dyn9 (loop over the DISTINCT experts)
if MOE_KERNEL:
    os.environ.setdefault('UNSAFE_FP8FNCAST', '1')   # torch float8_e4m3fn -> NKI float8_e4m3 (the HLO is retargeted the same way)
EGU_T_FORM = os.environ.get('EGU_T_FORM', 'permute')   # prefill (MOE=all) with EGU_T: permute (back to [E, 64, H]) | einsum
MOE_KM = int(os.environ.get('MOE_KM', '0'))          # kernel slot capacity override (default 10 T)
MOE_KPRE = os.environ.get('MOE_KPRE', '0') == '1'   # prefill MoE (MOE=all) through the kernel too (row chunks of 128)
MOE_KCH = int(os.environ.get('MOE_KCH', '32'))      # prefill row chunk (the kernel is validated on the device to 32 rows)
MOE_KUP = int(os.environ.get('MOE_KUP', '4'))       # super-iteration size for chunks wider than 32 rows (SBUF)
MOE_KU, MOE_KV = int(os.environ.get('MOE_KU', '8')), int(os.environ.get('MOE_KV', '2'))
MOE_KMEM = int(os.environ.get('MOE_KMEM', '160'))   # row kernel: up to this many slots the activations stay in SBUF between the phases
MOE_KSKIP = os.environ.get('MOE_KSKIP', '0') == '1'   # row kernel: padding slots (10 T not a multiple of 8) are not streamed
MOE_KFAST = int(os.environ.get('MOE_KFAST', '0'))   # row kernel: 1 = fewer vector instructions per group (one gate x up multiply, one amax per call)
MOE_KDENSE = int(os.environ.get('MOE_KDENSE', '0'))  # prefill with MOE_KERNEL=row MOE_KPRE=1: 1 = nki_moe_dyn.moe_dense9 (ONE pass over the experts for all the rows, per-super-iteration activation scale); 2: the same with two instructions for the scaled accumulate
MOE_KYD = int(os.environ.get('MOE_KYD', '0'))   # row kernel: y = accd * a2 straight from PSUM (one phase-3 call), no memset of the accumulator
MOE_KSTAT = int(os.environ.get('MOE_KSTAT', '0'))   # rows <= MOE_KSTAT: the kernel with static trip counts (all 10 T slots streamed, no device loop)
EGU_T = os.environ.get('EGU_T', '1' if MOE_KERNEL else '0') == '1'   # egu stored as egu_t [E * 128, 2048] (kernel layout, nki_moe.relayout_egu)
MTP_ARGMAX = os.environ.get('MTP_ARGMAX', 'eq')   # mtpc (draft chain in one launch): eq (no argmax op) | max (torch.max indices: aborts in the XLA trace)
MTP_CHAIN_DBG = os.environ.get('MTP_CHAIN_DBG', '0') == '1'
MTP_WIN = os.environ.get('MTP_WIN', 'arith')     # mtpc winner rank: arith (clamp arithmetic) | cmp (>= and a count)
GDN_VPUT = os.environ.get('GDN_VPUT', 'stack')   # verify state writes: stack (one scatter of R * Q states) | seq (one scatter of R rows per token:
#                                                  probe for the R > 1, Q = 2 buckets, whose gdn block costs 6.2 ms instead of 0.36 on the device)
VERIFY = os.environ.get('VERIFY', '0') == '1'   # speculative verify buckets (R = 1, Q = 1 + drafts tokens of one request): the gdn state after
#                                                  EVERY position goes to its own slot (step input 'vslot' [Q]); docs/neuron-qwen397-mtp.md
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


_Q8 = {}   # LIN_CSE: id(input) -> (input, scale a, fp8 codes): the projections of one tensor share one quantisation


def quant8(h2):
    """h2 [T, in] -> (a [T, 1], fp8 codes of h2 / a), as G.lin computes them."""
    hit = _Q8.get(id(h2)) if LIN_CSE else None
    if hit is not None and hit[0] is h2:
        return hit[1], hit[2]
    a = torch.amax(torch.abs(h2.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    h8 = (h2.float() / a).to(F8)
    if LIN_CSE:
        _Q8[id(h2)] = (h2, a, h8)
    return a, h8


def lin(h, t, name):
    if EXACT:
        return torch.matmul(h.reshape(-1, h.shape[-1]), t[name].float().T).reshape(*h.shape[:-1], t[name].shape[0])
    if LIN_CSE and name + '_s' in t and h.dim() == 2:
        a, h8 = quant8(h)
        return G.bf(G.mm8(h8, t[name].T).float() * a * t[name + '_s'].reshape(1, -1))
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
    slot = t.get('slot')
    keep = (t['pos'].reshape(R, Q)[:, 0] != 0).float()
    proj = lin(n, t, 'qkv').reshape(R, Q, W)                                           # [R, Q, W]
    bound = GDN_SLOT == 'bound'
    verify = VERIFY and Q > 1 if bound else 'vslot' in t
    if bound:
        assert R == 1 or not verify, 'GDN_SLOT=bound verify buckets: one row'
        rowst = lambda n: t[n] if R == 1 else torch.cat([t[f'{n}_{r}'] for r in range(R)], dim=0)
    cs = (rowst('cstate') if bound else torch.index_select(t['cstate'], 0, slot) if GDN_GATHER == 'flat' else t['cstate'][slot]) * keep.reshape(R, 1, 1)                                     # [R, W, K-1], oldest first
    seq = torch.cat((cs.permute(0, 2, 1), proj), dim=1)                                # [R, K-1+Q, W]
    cw = t['conv']                                                                     # [W, K] f32
    conv = sum(seq[:, j:j + Q] * cw[:, j].reshape(1, 1, W) for j in range(K))          # [R, Q, W]
    valid = t['pos'].reshape(R, Q) >= 0
    vs = t['vslot'].reshape(R, Q) if verify and not bound else None                    # the state after token i of row r -> slot vslot[r * Q + i]
    if bound and verify:                                                               # per-position conv states (states dict below)
        cstate = [seq[:, i + 1:i + K].permute(0, 2, 1) for i in range(Q)] if K > 1 else [t['cstate']] * Q
    elif verify and K > 1 and GDN_VPUT == 'seq':                                         # one scatter of R rows per token (see GDN_VPUT)
        cstate = t['cstate']
        for i in range(Q):
            cstate = cstate.index_put((vs[:, i],), seq[:, i + 1:i + K].permute(0, 2, 1))
    elif verify and K > 1:                                                             # (padding only at a row's end, its vslot = the scratch slot)
        cstate = t['cstate'].index_put((t['vslot'],), torch.stack([seq[:, i + 1:i + K] for i in range(Q)], dim=1).reshape(R * Q, K - 1, W).permute(0, 2, 1))
    elif K > 1 and Q == 1 and GDN_DEC:                                                 # decode rows are never padding: the new state is the last K-1 inputs
        cstate = seq[:, 1:].permute(0, 2, 1) if bound else t['cstate'].index_put((slot,), seq[:, 1:].permute(0, 2, 1))
    elif K > 1:                                                                        # state = the K-1 inputs before the first padding token
        nvalid = valid.long().sum(dim=1)
        idx = (nvalid.reshape(R, 1) + torch.arange(K - 1, device=n.device).reshape(1, K - 1)).reshape(R, K - 1, 1).expand(R, K - 1, W)
        cstate = torch.gather(seq, 1, idx).permute(0, 2, 1)
        if not bound:
            cstate = t['cstate'].index_put((slot,), cstate)
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
    st = (rowst('kstate') if bound else torch.index_select(t['kstate'], 0, slot) if GDN_GATHER == 'flat' else t['kstate'][slot]) * keep.reshape(R, 1, 1, 1)                                 # [R, hl, Dk, Dv]
    C = GDN_CHUNK
    if Q == 1 and GDN_FORM == 'fused':
        # one contraction of the stored state with (k | q) instead of two contractions of two materialised states:
        # mem = d (S k), out = (d S + k (x) delta) q = d (S q) + (k . q) delta. Equal to the loop form up to f32 rounding.
        d = decay[:, 0].reshape(R, hl, 1)
        c = torch.einsum('rhkd,rhkc->rhcd', st, torch.stack((kn[:, 0], qn[:, 0]), dim=-1))     # [R, hl, 2, Dv]
        delta = (v[:, 0] - d * c[:, :, 0]) * beta[:, 0].unsqueeze(-1)
        y = (d * c[:, :, 1] + (kn[:, 0] * qn[:, 0]).sum(dim=-1, keepdim=True) * delta).unsqueeze(1)
        st = st * d.unsqueeze(-1) + kn[:, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    elif Q > 1 and C > 1 and Q % C == 0 and not verify:
        st, y = (gdn_chunks2 if GDN_INV == 'diag' else gdn_chunks)(st, qn, kn, v, beta, gg, valid, C)
    elif Q > 1 and GDN_LOOP == 'fused' and not verify:
        # the Q = 1 fused form per token; padding tokens (only at a row's end) get beta = 0 and decay = 1: the state stays
        vf = valid.reshape(R, Q, 1).float()
        bm, dm = beta * vf, torch.exp(gg * vf)
        kq = (kn * qn).sum(dim=-1, keepdim=True)                                       # [R, Q, hl, 1]
        kqs = torch.stack((kn, qn), dim=-1)                                            # [R, Q, hl, Dk, 2]
        outs = []
        for i in range(Q):
            d = dm[:, i].reshape(R, hl, 1)
            c = torch.einsum('rhkd,rhkc->rhcd', st, kqs[:, i])                         # [R, hl, 2, Dv]
            delta = (v[:, i] - d * c[:, :, 0]) * bm[:, i].unsqueeze(-1)
            outs.append(d * c[:, :, 1] + kq[:, i] * delta)
            st = st * d.unsqueeze(-1) + kn[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        y = torch.stack(outs, dim=1)
    else:
        outs, sts = [], []
        elem = GDN_EINSUM == 'elem' and Q == 1
        contract = (lambda m, x: (m * x.unsqueeze(-1)).sum(dim=-2)) if elem else (lambda m, x: torch.einsum('rhkd,rhk->rhd', m, x))
        for i in range(Q):                                                             # the recurrence, token by token
            nst = st * decay[:, i].reshape(R, hl, 1, 1)
            mem = contract(nst, kn[:, i])
            delta = (v[:, i] - mem) * beta[:, i].unsqueeze(-1)
            nst = nst + kn[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append(contract(nst, qn[:, i]))
            st = torch.where(valid[:, i].reshape(R, 1, 1, 1), nst, st) if Q > 1 and not verify else nst
            sts.append(nst)
        y = torch.stack(outs, dim=1)                                                   # [R, Q, hl, Dv]
    if bound and verify:
        kstate = sts
    elif verify and GDN_VPUT == 'seq':
        kstate = t['kstate']
        for i in range(Q):
            kstate = kstate.index_put((vs[:, i],), sts[i])
    else:
        kstate = t['kstate'].index_put((t['vslot'],), torch.stack(sts, dim=1).reshape(R * Q, hl, Dk, Dv)) if verify else st if bound else t['kstate'].index_put((slot,), st)
    z = lin(n, t, 'z').reshape(R, Q, hl, Dv)
    y = y * torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + g.eps) * t['onorm'] * (z * torch.sigmoid(z))
    out = allreduce(lin_partial(bf(y).reshape(T, g.Wv), t, 'o'))
    if bound and verify:      # position 0 -> the row's own slot (aliased state), positions i >= 1 -> extra outputs bound to vslot[i]
        sd = {'kstate': kstate[0], 'cstate': cstate[0]}
        for i in range(1, Q):
            sd[f'kstate@{i}'], sd[f'cstate@{i}'] = kstate[i], cstate[i]
        return out, sd
    if bound and R > 1:
        return out, {f'{n}_{r}': v[r:r + 1] for r in range(R) for n, v in (('kstate', kstate), ('cstate', cstate))}
    return out, {'kstate': kstate, 'cstate': cstate}


def unit_lower_inverse(m):
    """(I + m)^-1 for strictly lower-triangular m [..., C, C], C a power of two: exact block recursion
    [[A, 0], [L, D]]^-1 = [[A^-1, 0], [-D^-1 L A^-1, D^-1]] bottom-up, batched over the diagonal blocks (matmuls only)."""
    C = m.shape[-1]
    lead = m.shape[:-2]
    inv = torch.ones(*lead, C, 1, 1, dtype=m.dtype, device=m.device)                  # C blocks of size 1
    s = 1
    while s < C:
        nb = C // (2 * s)
        blocks = torch.diagonal(m.reshape(*lead, nb, 2 * s, nb, 2 * s), dim1=-4, dim2=-2).movedim(-1, -3)   # [..., nb, 2s, 2s]
        low = blocks[..., s:, :s]                                                      # [..., nb, s, s]
        pair = inv.reshape(*lead, nb, 2, s, s)
        a_inv, d_inv = pair[..., 0, :, :], pair[..., 1, :, :]
        ll = -torch.matmul(torch.matmul(d_inv, low), a_inv)
        top = torch.cat((a_inv, torch.zeros_like(a_inv)), dim=-1)
        bot = torch.cat((ll, d_inv), dim=-1)
        inv = torch.cat((top, bot), dim=-2)                                            # [..., nb, 2s, 2s]
        s *= 2
    return inv.reshape(*lead, C, C)


def gdn_chunks(st, qn, kn, v, beta, gg, valid, C):
    """The gated delta rule over Q tokens in chunks of C (matmul form of the token loop; docs/neuron-qwen397.md 8.9):
    with G = cumsum(g) inside the chunk, U = (I + M)^-1 (beta V - diag(beta e^G) K S0), M[t, j<t] = beta_t e^(G_t - G_j) k_t.k_j,
    O = diag(e^G) Q S0 + N U (N[t, j<=t] = e^(G_t - G_j) q_t.k_j), S_C = e^(G_C) S0 + (diag(e^(G_C - G)) K)^T U.
    Padding tokens (valid False) get beta = 0 and g = 0, which leaves the state untouched.
    st [R, hl, Dk, Dv]; qn / kn [R, Q, hl, Dk]; v [R, Q, hl, Dv]; beta / gg [R, Q, hl] -> (state, y [R, Q, hl, Dv])."""
    R, Q, hl, Dk = kn.shape
    vf = valid.reshape(R, Q, 1).float()
    beta, gg = beta * vf, gg * vf
    ys = []
    eye_low = torch.tril(torch.ones(C, C, device=st.device), diagonal=-1)
    incl = torch.tril(torch.ones(C, C, device=st.device))
    for c0 in range(0, Q, C):
        K = kn[:, c0:c0 + C].permute(0, 2, 1, 3)                                       # [R, hl, C, Dk]
        Qm = qn[:, c0:c0 + C].permute(0, 2, 1, 3)
        V = v[:, c0:c0 + C].permute(0, 2, 1, 3)                                        # [R, hl, C, Dv]
        b = beta[:, c0:c0 + C].permute(0, 2, 1)                                        # [R, hl, C]
        G = (gg[:, c0:c0 + C].permute(0, 2, 1).unsqueeze(-2) * incl).sum(dim=-1)       # cumulative log decay [R, hl, C] (<= 0), mask-sum form
        ratio = torch.exp((G.unsqueeze(-1) - G.unsqueeze(-2)) * incl)                  # e^(G_t - G_j) for j <= t (1 elsewhere, masked below)
        M = b.unsqueeze(-1) * ratio * torch.matmul(K, K.transpose(-1, -2)) * eye_low
        N = ratio * torch.matmul(Qm, K.transpose(-1, -2)) * incl
        eG = torch.exp(G).unsqueeze(-1)                                                # [R, hl, C, 1]
        rhs = b.unsqueeze(-1) * (V - eG * torch.matmul(K, st))                         # [R, hl, C, Dv]
        U = torch.matmul(unit_lower_inverse(M), rhs)
        ys.append(eG * torch.matmul(Qm, st) + torch.matmul(N, U))
        last = G[..., -1:]                                                             # [R, hl, 1]
        st = torch.exp(last).unsqueeze(-1) * st + torch.matmul((torch.exp(last - G).unsqueeze(-1) * K).transpose(-1, -2), U)
    return st, torch.cat(ys, dim=2).permute(0, 2, 1, 3)


def gdn_chunks2(st, qn, kn, v, beta, gg, valid, C):
    """gdn_chunks (same algebra, same contract) arranged for the tensor engine: every matrix is [C, C] / [C, Dk] / [C, Dv] with
    C = 128 = one PE tile, everything that does not depend on the carried state (Gram matrices, the decay ratios, the inverse)
    is computed for ALL chunks in one batch [R, hl, Q / C], and (I + M)^-1 is built by block-diagonal doubling with full
    matmuls: with X_s = the block-diagonal matrix of the inverses of the s-blocks of I + M and L_s = M restricted to the
    (odd s-block, preceding even s-block) cells, X_2s = X_s - X_s L_s X_s (the exact 2 x 2 block inverse; log2(C) levels,
    2 batched matmuls each, no per-block slicing). The sequential part is 5 matmuls per chunk."""
    R, Q, hl, Dk = kn.shape
    Dv, nc, dev = v.shape[-1], Q // C, st.device
    vf = valid.reshape(R, Q, 1).float()
    beta, gg = beta * vf, gg * vf
    K = kn.permute(0, 2, 1, 3).reshape(R, hl, nc, C, Dk)
    Qm = qn.permute(0, 2, 1, 3).reshape(R, hl, nc, C, Dk)
    V = v.permute(0, 2, 1, 3).reshape(R, hl, nc, C, Dv)
    b = beta.permute(0, 2, 1).reshape(R, hl, nc, C, 1)
    gl = gg.permute(0, 2, 1).reshape(R, hl, nc, 1, C)
    incl = torch.tril(torch.ones(C, C, device=dev))
    low = torch.tril(torch.ones(C, C, device=dev), diagonal=-1)
    Gc = (gl * incl).sum(dim=-1)                                                       # cumulative log decay [R, hl, nc, C] (<= 0)
    ratio = torch.exp((Gc.unsqueeze(-1) - Gc.unsqueeze(-2)) * incl)                    # e^(G_t - G_j), j <= t (masked below)
    KT = K.transpose(-1, -2)
    M = b * ratio * torch.matmul(K, KT) * low
    N = ratio * torch.matmul(Qm, KT) * incl
    blk = torch.arange(C, device=dev)
    inv = None
    s = 1
    while s < C:
        bi = torch.div(blk, s, rounding_mode='floor')
        cell = ((bi.reshape(C, 1) % 2 == 1) & (bi.reshape(1, C) == bi.reshape(C, 1) - 1)).float()
        L = M * cell
        inv = (torch.eye(C, device=dev) - L) if inv is None else inv - torch.matmul(torch.matmul(inv, L), inv)
        s *= 2
    eG = torch.exp(Gc).unsqueeze(-1)                                                   # [R, hl, nc, C, 1]
    last = Gc[..., -1:]                                                                # [R, hl, nc, 1]
    eL = torch.exp(last).unsqueeze(-1)                                                 # [R, hl, nc, 1, 1]
    Kd = (torch.exp(last - Gc).unsqueeze(-1) * K).transpose(-1, -2)                    # [R, hl, nc, Dk, C]
    bV, beK, eQ = b * V, b * eG * K, eG * Qm
    ys = []
    for c in range(nc):
        U = torch.matmul(inv[:, :, c], bV[:, :, c] - torch.matmul(beK[:, :, c], st))   # [R, hl, C, Dv]
        ys.append(torch.matmul(eQ[:, :, c], st) + torch.matmul(N[:, :, c], U))
        st = eL[:, :, c] * st + torch.matmul(Kd[:, :, c], U)
    return st, torch.cat(ys, dim=2).permute(0, 2, 1, 3)


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
    if 'kvs' in t:                                                                     # KV8: fp8 codes + one f32 scale per (token, K|V, head)
        return gqa_fp8(n, t, g, R, Q, P, allreduce, q, gate, k, bf(v), cache, at)
    cache = cache.index_put(at, torch.stack((k, bf(v)), dim=1).to(cache.dtype))         # [T, 2, kvl, D]
    table = t['table']                                                                 # [R, P]
    keys = P * PAGE
    lowp = ATT_BF16 and not EXACT                                                      # contract the gathered K | V in bf16 (no f32 copy of the keys)
    if ATT_GATHER == 'flat':
        kv = torch.index_select(cache, 0, table.reshape(R * P)).reshape(R, P, 2, kvl, PAGE, D)
    else:
        kv = cache[table]
    kv = kv if lowp else kv.float()                                                    # [R, P, 2, kvl, 16, D]
    kk = kv[:, :, 0].permute(0, 2, 1, 3, 4).reshape(R, kvl, keys, D)
    vv = kv[:, :, 1].permute(0, 2, 1, 3, 4).reshape(R, kvl, keys, D)
    rep = hq // kvl
    qh = q.reshape(R, Q, kvl, rep, D)
    s = torch.einsum('rqhgd,rhkd->rqhgk', qh.to(kk.dtype), kk).float() / math.sqrt(D)
    kpos = torch.arange(keys, dtype=torch.int64, device=n.device).reshape(1, 1, 1, 1, keys)
    visible = kpos <= t['pos'].reshape(R, Q, 1, 1, 1)
    if 'lo' in t:                                                                      # MTP cache: entries exist from key t['lo'] [R] of the table on
        visible = visible & (kpos >= t['lo'].reshape(R, 1, 1, 1, 1))
    s = torch.where(visible, s, torch.full_like(s, -3e38))
    p = bf(torch.softmax(s, dim=-1))
    o = torch.einsum('rqhgk,rhkd->rqhgd', p.to(vv.dtype), vv).float().reshape(T, hq, D)
    if gate is not None:
        o = o * torch.sigmoid(gate)
    out = allreduce(lin_partial(bf(o).reshape(T, hq * D), t, 'o'))
    return out, {'kv': cache}


def gqa_fp8(n, t, g, R, Q, P, allreduce, q, gate, k, v, cache, at):
    """KV8 (long context): the cache holds fp8 e4m3 codes [pages, 2, kvl, 16, D] (half the bf16 bytes) and t['kvs'] one f32 scale
    per (page, K|V, head, offset) = amax over D / 224, the activation recipe of quant8. The scales are folded into the scores (keys)
    and the probabilities (values), so no dequantised copy of the keys exists. ATT_BLOCK = pages per key block: the keys are
    visited in blocks with a running max / sum (online softmax), which bounds the scratch of the 8192 / 16384-page buckets."""
    T, hq, kvl, D = R * Q, g.hq, g.kvl, g.D
    kv = torch.stack((k, v), dim=1).float()                                            # [T, 2, kvl, D]
    if cache.dtype == torch.int8:
        a = torch.amax(torch.abs(kv), dim=-1).clamp(min=1e-20) / 127.0                # [T, 2, kvl]
        cache = cache.index_put(at, torch.round(kv / a.unsqueeze(-1)).to(torch.int8))
    else:
        a = torch.amax(torch.abs(kv), dim=-1).clamp(min=1e-20) / 224.0
        cache = cache.index_put(at, (kv / a.unsqueeze(-1)).to(cache.dtype))
    scales = t['kvs'].index_put(at, a)                                 # [pages, 2, kvl, 16]
    table = t['table']
    rep = hq // kvl
    qh = q.reshape(R, Q, kvl, rep, D)
    dt = torch.float32 if EXACT else torch.bfloat16                                    # int8 / e4m3 codes are exact in bf16
    qd = qh.to(dt)
    pos = t['pos'].reshape(R, Q, 1, 1, 1)
    blk = ATT_BLOCK if ATT_BLOCK and P > ATT_BLOCK and P % ATT_BLOCK == 0 else P
    m = l = acc = None
    for b0 in range(0, P, blk):
        tb = table[:, b0:b0 + blk] if blk < P else table
        if ATT_GATHER == 'flat':
            kb = torch.index_select(cache, 0, tb.reshape(R * blk)).reshape(R, blk, 2, kvl, PAGE, D)
        else:
            kb = cache[tb]
        sb = scales[tb]                                                                # [R, blk, 2, kvl, 16]
        keys = blk * PAGE
        kk = kb[:, :, 0].permute(0, 2, 1, 3, 4).reshape(R, kvl, keys, D).to(dt)
        vv = kb[:, :, 1].permute(0, 2, 1, 3, 4).reshape(R, kvl, keys, D).to(dt)
        ks = sb[:, :, 0].permute(0, 2, 1, 3).reshape(R, 1, kvl, 1, keys)
        vs = sb[:, :, 1].permute(0, 2, 1, 3).reshape(R, 1, kvl, 1, keys)
        s = torch.einsum('rqhgd,rhkd->rqhgk', qd, kk).float() * (ks / math.sqrt(D))
        kpos = torch.arange(b0 * PAGE, b0 * PAGE + keys, dtype=torch.int64, device=n.device).reshape(1, 1, 1, 1, keys)
        s = torch.where(kpos <= pos, s, torch.full_like(s, -3e38))
        mb = torch.amax(s, dim=-1)
        if m is None:   # block 0 holds key 0, visible to every real row: m is finite from here on
            m = mb
            p = torch.exp(s - m.unsqueeze(-1))
            l = p.sum(-1)
            acc = torch.einsum('rqhgk,rhkd->rqhgd', bf(p * vs).to(dt), vv).float()
        else:
            mn = torch.maximum(m, mb)
            c = torch.exp(m - mn)
            p = torch.exp(s - mn.unsqueeze(-1))
            l = l * c + p.sum(-1)
            acc = acc * c.unsqueeze(-1) + torch.einsum('rqhgk,rhkd->rqhgd', bf(p * vs).to(dt), vv).float()
            m = mn
    o = (acc / l.unsqueeze(-1)).reshape(T, hq, D)
    if gate is not None:
        o = o * torch.sigmoid(gate)
    out = allreduce(lin_partial(bf(o).reshape(T, hq * D), t, 'o'))
    return out, {'kv': cache, 'kvs': scales}


def topk_mask(probs, k, form):
    """Exact top-k membership mask [T, E] (f32 0 / 1) of probs [T, E] under torch.topk's order (value descending, ties: the
    lower index wins), without a topk op (on trn1 / neuronx-cc 2.26 topk over [4, 512] / [8, 512] with k = 10 is non-finite,
    docs/neuron-qwen397.md 8.4). Forms (all select the same set, so the combine weights are bit-identical):
      rank  expert e is selected iff fewer than k experts beat it: one [T, E, E] comparison (6 passes over T * E * E elements;
            at E = 512 that is 2 M elements per 8 rows and was 20 % of the 8-row decode step, docs/neuron-qwen397-batch.md)
      rank2 the same matrix in 3 passes (where(e' < e, p' >= p, p' > p))
      tour  tournament: rank inside G groups of E / G experts ([T, G, S, S]), keep each group's k best as candidates
            (an expert of the global top-k is in its group's top-k), rank the G * k candidates ([T, Gk, Gk]), map back.
            8x fewer elements than rank at E = 512, G = 8."""
    T, E = probs.shape
    dev = probs.device
    if form in ('rank', 'rank2'):
        ex = torch.arange(E, device=dev)
        a, b = probs.unsqueeze(2), probs.unsqueeze(1)                                  # [T, E, 1] (e), [T, 1, E] (e')
        lower = (ex.reshape(1, 1, E) < ex.reshape(1, E, 1))                            # e' < e
        if form == 'rank':
            beats = ((b > a) | ((b == a) & lower)).float().sum(dim=-1)                 # [T, E]
        else:
            beats = torch.where(lower, b >= a, b > a).float().sum(dim=-1)
        return (beats < k).float()
    G = TOUR_GROUPS
    S = E // G
    if E % G or S < k:
        raise ValueError(f'tournament routing: {E} experts do not split into {G} groups of >= {k}')
    p = probs.reshape(T, G, S)
    ex = torch.arange(S, device=dev)
    lower = (ex.reshape(1, S) < ex.reshape(S, 1))                                      # [e, e']: e' < e
    lrank = torch.where(lower, p.unsqueeze(2) >= p.unsqueeze(3), p.unsqueeze(2) > p.unsqueeze(3)).float().sum(dim=-1)   # [T, G, S]
    m = (lrank.unsqueeze(2) == torch.arange(k, device=dev).float().reshape(1, 1, k, 1)).float()     # [T, G, k, S]: the s-th best of group g
    cand = (m * p.unsqueeze(2)).sum(dim=-1).reshape(T, G * k)                          # candidate values, index order = (group, local rank)
    C = G * k
    cx = torch.arange(C, device=dev)
    clower = (cx.reshape(1, C) < cx.reshape(C, 1))                                     # candidate c' before c: lower expert index on ties
    crank = torch.where(clower, cand.unsqueeze(1) >= cand.unsqueeze(2), cand.unsqueeze(1) > cand.unsqueeze(2)).float().sum(dim=-1)   # [T, C]
    csel = (crank < k).float().reshape(T, G, k, 1)
    return (m * csel).sum(dim=2).reshape(T, E)


def route(n, t, g, allreduce=None, want_top=False):
    """Softmax over all experts, top-k, renormalised -> combine weights [T, E] (0 for unselected experts).
    want_top (MOE_SLOTS=row): -> (expert indices [T, K], their renormalised weights [T, K]) instead.
    ROUTER=shard: this rank's partial logits over its hidden slice, summed by an f32 all-reduce (every rank receives the
    same bits, so every rank selects the same experts)."""
    if 'rws' in t:
        logits = allreduce(torch.matmul(torch.index_select(n, 1, t['rcols']).float(), t['rws'].T))
    elif RLOGIT == 'bf16' and not EXACT:
        logits = torch.matmul(n.to(torch.bfloat16), t['rw'].to(torch.bfloat16).T).float()
    else:
        logits = torch.matmul(n.float(), t['rw'].T)
    probs = torch.softmax(logits, dim=-1)                                              # [T, E] f32
    T = n.shape[0]
    if ROUTE == 'thr' and not want_top:
        # top-k by threshold: the k-th largest probability of the row (torch.topk VALUES; no index tensor, no one-hot, no
        # einsum). Equal to ROUTE=topk unless two probabilities tie exactly at the k-th place (then both are kept).
        thr = torch.topk(probs, g.topk, dim=-1)[0][:, g.topk - 1:g.topk]
        w = probs * (probs >= thr).float()
        return w / w.sum(dim=-1, keepdim=True)
    if want_top:
        N = int(ROUTE[3:]) if ROUTE.startswith('pad') else T
        padded = torch.cat((probs, torch.zeros(N - T, g.E, device=n.device) + probs[:1] * 0.0), dim=0) if T < N else probs
        tk = torch.topk(padded, g.topk, dim=-1)
        v = tk[0][:T]
        return tk[1][:T], v / v.sum(dim=-1, keepdim=True)
    experts = torch.arange(g.E, device=n.device)
    if ROUTE in ('rank', 'rank2', 'tour'):
        w = probs * topk_mask(probs, g.topk, ROUTE)
        return w / w.sum(dim=-1, keepdim=True)
    if ROUTE.startswith('pad'):
        # torch.topk on the rows padded to ROUTE=pad<N> rows (N = 64: the prefill shape, where the device's topk is correct;
        # 4 and 8 rows are not, docs/neuron-qwen397.md 8.4), used for the MEMBERSHIP only: the weights are then computed
        # exactly as in the rank forms, so the block output is bit-identical to ROUTE=rank whenever the mask is.
        # 0.09 ms per MoE block cheaper than rank at 8 rows (docs/neuron-qwen397-batch.md).
        N = int(ROUTE[3:])
        padded = torch.cat((probs, torch.zeros(N - T, g.E, device=n.device) + probs[:1] * 0.0), dim=0) if T < N else probs
        top = torch.topk(padded, g.topk, dim=-1)[1][:T]                                # [T, K]
        mask = (top.unsqueeze(-1) == experts.reshape(1, 1, g.E)).float().sum(dim=1)    # [T, E]
        w = probs * mask
        return w / w.sum(dim=-1, keepdim=True)
    top = torch.topk(probs, g.topk, dim=-1)[1]                                         # [T, K]
    onehot = (top.unsqueeze(-1) == experts.reshape(1, 1, g.E)).float()                 # [T, K, E]  (one-hot: see glm_graphs.route)
    w = torch.einsum('tke,te->tk', onehot, probs)
    w = w / w.sum(dim=-1, keepdim=True)
    return torch.einsum('tke,tk->te', onehot, w)


def moe_block(n, t, g, allreduce, mode='all', ar_route=None):
    """Shared expert (TP-split, sigmoid-gated) + this rank's slices of the selected experts (glm_graphs.moe_block layout)."""
    T, E, EIl = n.shape[0], g.E, g.EIl
    sgate = torch.sigmoid(torch.matmul(n.float(), t['sgate'].reshape(1, -1).T))        # [T, 1] replicated
    shared = lin_partial(bf(silu_mul(lin(n, t, 'sg'), lin(n, t, 'su'))), t, 'sd') * sgate
    rowslots = MOE_SLOTS == 'row' and mode == 'gather'
    comb = route(n, t, g, ar_route or allreduce, want_top=rowslots)                    # [T, E] | ([T, K], [T, K])
    if rowslots and MOE_DEBUG == 'route':
        top, w = comb
        comb = (w.unsqueeze(-1) * (top.unsqueeze(-1) == torch.arange(g.E, device=n.device).reshape(1, 1, g.E)).float()).sum(dim=1)
    if MOE_DEBUG in ('route', 'probs'):
        v = (comb if MOE_DEBUG == 'route' else torch.softmax(torch.matmul(n.float(), t['rw'].T), dim=-1)).reshape(T, -1).float()[:, :g.H]
        alive = sum(w.reshape(-1)[:1].float().sum() for k, w in t.items() if k not in ('x', 'norm', 'rcols')) * 0.0
        return allreduce(torch.nn.functional.pad(v, (0, g.H - v.shape[1])) + alive), {}
    groups = MOE_SPLIT if (mode == 'gather' and MOE_SPLIT and T > MOE_SPLIT and T % MOE_SPLIT == 0 and not MOE_DEBUG) else T
    if groups != T:
        # rows in groups of MOE_SPLIT, each with its own slot union: the gathered slices of one group stay small
        # (docs/neuron-qwen397-batch.md: the 160-slot gather of 16 rows falls off a cliff, 2 x 80 slots does not)
        routed = torch.cat([moe_routed(n[i:i + groups], comb[i:i + groups], t, g, mode)[0] for i in range(0, T, groups)], dim=0)
        return allreduce(shared + routed), {}
    routed, stage = moe_routed(n, comb, t, g, mode)
    if MOE_DEBUG:
        v = shared if MOE_DEBUG == 'shared' else stage
        v = v.reshape(T, -1).float()[:, :g.H]
        alive = sum(w.reshape(-1)[:1].float().sum() for k, w in t.items() if k not in ('x', 'norm', 'rcols')) * 0.0   # every input stays a graph parameter
        return allreduce(torch.nn.functional.pad(v, (0, g.H - v.shape[1])) + alive), {}
    return allreduce(shared + routed), {}


def moe_routed(n, comb, t, g, mode):
    """This rank's partial of the routed experts for rows n [T, H] with combine weights comb [T, E] -> (routed [T, H] f32,
    the MOE_DEBUG stage tensor or None)."""
    T, E, EIl = n.shape[0], g.E, g.EIl
    gu, gus, dw, dc = t['egu'], t.get('egu_s'), t['ed'], t.get('ed_c')                 # [E, 2EIl, H], [E, EIl, H]
    if MOE_KERNEL in ('row', 'dyn') and isinstance(comb, tuple) and n.device.type != 'cpu' and not MOE_DEBUG:
        # MOE_SLOTS=row with a kernel: the static row-slot kernel (10 T slots, no selection logic; nki_moe_dyn.moe_row9)
        import nki_moe_dyn as D
        top, w = comb
        n8, a = D.graph_inputs9(n)
        idx, idxf, cs = D.row_inputs9(top, w, T, g.topk)
        routed = D.moe_row9(n8, a, idx, idxf, cs, gus, dc, gu if gu.dim() == 2 else gu.reshape(E * 128, -1), dw.reshape(E * EIl, g.H),
                            U=min(MOE_KU, idx.shape[0] // 8), mem=1 if idx.shape[0] <= MOE_KMEM else 0,
                            NR=T * g.topk if MOE_KSKIP else 1000000, fast=MOE_KFAST, yd=MOE_KYD)
        return routed, routed
    if MOE_KERNEL == 'row' and mode == 'all' and MOE_KPRE and not isinstance(comb, tuple) and n.device.type != 'cpu' and not MOE_DEBUG:
        # prefill through the static kernel, DENSE: every expert is a slot (idx = arange(E)), rows in chunks of MOE_KCH (<= 128:
        # the stationary operand); every chunk streams all the experts once
        import nki_moe_dyn as D
        if MOE_KDENSE and T % min(MOE_KCH, T) == 0:
            tc = min(MOE_KCH, T)
            NH = T // tc
            n8, a = D.graph_inputs9(n)
            outs = []
            for h0 in range(0, NH, 4):   # at most 4 row blocks per kernel call (SBUF: x^T, x and y of every block stay resident)
                nh = min(4, NH - h0)
                r0, r1 = h0 * tc, (h0 + nh) * tc
                cs = comb[r0:r1].reshape(nh, tc, E).permute(0, 2, 1).reshape(nh * E, tc).contiguous()
                outs.append(D.moe_dense9(n8[r0:r1], a[r0:r1], cs, gus, dc, gu if gu.dim() == 2 else gu.reshape(E * 128, -1), dw.reshape(E * EIl, g.H),
                                         NH=nh, U=MOE_KUP, stt=1 if MOE_KDENSE == 1 else 0))
            routed = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
            return routed, routed
        idxf = torch.arange(E, device=n.device).float().reshape(E, 1)
        idx = idxf.to(torch.int32)
        outs = []
        for r0 in range(0, T, MOE_KCH):
            tc = min(MOE_KCH, T - r0)
            n8, a = D.graph_inputs9(n[r0:r0 + tc])
            outs.append(D.moe_row9(n8, a, idx, idxf, comb[r0:r0 + tc].T.contiguous(), gus, dc, gu if gu.dim() == 2 else gu.reshape(E * 128, -1),
                                   dw.reshape(E * EIl, g.H), U=MOE_KUP if tc > 32 else MOE_KU, mem=0, fast=MOE_KFAST))
        routed = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        return routed, routed
    if MOE_KERNEL == 'dyn' and (mode == 'gather' or MOE_KPRE) and not isinstance(comb, tuple) and n.device.type != 'cpu' and not MOE_DEBUG:
        # the distinct-expert kernel; prefill (mode all, MOE_KPRE=1): row chunks of <= 128 (every chunk streams the experts its rows use)
        import nki_moe_dyn as D
        outs = []
        for r0 in range(0, T, MOE_KCH):
            tc = min(MOE_KCH, T - r0)
            km = min(E, max(-(-g.topk * tc // 8) * 8, MOE_KM))
            n8, a = D.graph_inputs9(n[r0:r0 + tc])
            u = MOE_KU if tc <= 32 else MOE_KUP
            outs.append(D.moe_dyn9(n8, a, comb[r0:r0 + tc], gus, dc, gu if gu.dim() == 2 else gu.reshape(E * 128, -1), dw.reshape(E * EIl, g.H),
                                   KM=km, U=min(u, km // 8) if tc <= MOE_KSTAT else u, V=MOE_KV, stat=1 if tc <= MOE_KSTAT else 0))
        routed = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        return routed, routed
    if gu.dim() == 2 and mode == 'all' and EGU_T_FORM == 'bmm' and gus is not None and not EXACT and not isinstance(comb, tuple):
        # EGU_T layout, every expert, no relayout of the weights: egu_t viewed as E matrices [4096 (p, kt), 64] (contraction-major,
        # like ed) and a batched matmul over the experts with the activations permuted to the (p, kt) order
        a, n8 = quant8(n)
        KTn = g.H // 128
        n8p = n8.reshape(T, KTn, 128).transpose(1, 2).reshape(1, T, g.H)
        w3 = gu.reshape(E, g.H, 2 * EIl)
        if n.device.type == 'cpu':
            hgu = torch.matmul(n8p.float(), w3.float())                                # [E, T, 2 EIl]
        else:
            hgu = torch.matmul(n8p.expand(E, T, g.H), w3)
        hgu = G.bf(hgu.float() * a.reshape(1, T, 1) * gus.reshape(E, 1, 2 * EIl)).reshape(E, T, 2, EIl)
        act = bf(silu_mul(hgu[:, :, 0], hgu[:, :, 1])) * comb.T.unsqueeze(-1)           # [E, T, EIl]
        if dc is not None:
            act = act * dc.reshape(E, 1, EIl)
        a2 = act.permute(1, 0, 2).reshape(T, E * EIl)
        a = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        routed = G.mm8((a2 / a).to(F8), dw.reshape(E * EIl, g.H)).float() * a
        return routed, routed
    if gu.dim() == 2 and mode == 'all' and EGU_T_FORM == 'einsum' and gus is not None and not EXACT and not isinstance(comb, tuple):
        # EGU_T layout, every expert: hgu[t, e, o] = sum_{kt, p} n8[t, kt, p] egu_t[e, p, kt, o] (no relayout of the weights)
        a, n8 = quant8(n)
        w4 = gu.reshape(E, 128, g.H // 128, 2 * EIl)
        if n.device.type == 'cpu':
            hgu = torch.einsum('tkp,epko->teo', n8.float().reshape(T, g.H // 128, 128), w4.float())
        else:
            hgu = torch.einsum('tkp,epko->teo', n8.reshape(T, g.H // 128, 128), w4)
        hgu = G.bf(hgu.float() * a.reshape(T, 1, 1) * gus.reshape(1, E, 2 * EIl)).reshape(T, E, 2, EIl)
        act = bf(silu_mul(hgu[:, :, 0], hgu[:, :, 1])) * comb.unsqueeze(-1)
        if dc is not None:
            act = act * dc.reshape(1, E, EIl)
        a2 = act.reshape(T, E * EIl)
        a = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        routed = G.mm8((a2 / a).to(F8), dw.reshape(E * EIl, g.H)).float() * a
        return routed, routed
    egut = gu.dim() == 2
    if egut:   # EGU_T layout [E * 128, 2048] (egu_t[e * 128 + p, kt * 64 + o] = egu[e, o, kt * 128 + p])
        gu = gu.reshape(E, 128, g.H // 128 * 2 * EIl)
        if mode == 'all' or not (isinstance(comb, tuple) or g.topk * T < E):   # every expert: back to [E, 2EIl, H] (a full relayout)
            if EGU_T_FORM == 'u8' and gu.device.type != 'cpu':   # transpose the raw bytes (int8), bitcast back to fp8
                gu = gu.view(torch.int8).reshape(E, 128, g.H // 128, 2 * EIl).permute(0, 3, 2, 1).reshape(E, 2 * EIl, g.H).view(F8)
            else:
                gu = gu.reshape(E, 128, g.H // 128, 2 * EIl).permute(0, 3, 2, 1).reshape(E, 2 * EIl, g.H)
            egut = False
    untile = lambda w: w.reshape(-1, 128, g.H // 128, 2 * EIl).permute(0, 3, 2, 1).reshape(-1, 2 * EIl, g.H) if egut else w   # gathered slices only
    if isinstance(comb, tuple):
        # MOE_SLOTS=row: slot r * K + j holds the j-th expert of row r (the router's topk indices, no union over the rows, no
        # top-k over the experts, no [T, E] combine matrix). An expert selected by two rows is read twice; the union form
        # pads its slots with unused experts instead, so the bytes are the same 10 R slices.
        top, w = comb
        K = g.topk
        idx = top.reshape(T * K)
        perm = None
        if ROW_FORM == 'sort':      # probe: slots in ascending expert order (the union form's order), weights permuted to match
            idx, perm = torch.sort(idx)
        elif ROW_FORM == 'launder':  # probe: the indices through f32 and back
            idx = (idx.float() + 0.0).round().long()
        elif ROW_FORM == 'topk':    # probe: the indices re-derived by a [1, T K] topk (ascending expert order, duplicates kept)
            order = torch.topk(-(idx.float() + torch.arange(T * K, device=n.device).float() / (2 * T * K)).reshape(1, T * K), T * K, dim=-1)[1].reshape(T * K)
            idx, perm = torch.index_select(idx, 0, order), order
        gu, dw = untile(torch.index_select(gu, 0, idx)), torch.index_select(dw, 0, idx)
        gus = torch.index_select(gus, 0, idx) if gus is not None else None
        dc = torch.index_select(dc, 0, idx) if dc is not None else None
        if perm is not None:
            comb = (w.unsqueeze(1) * torch.eye(T, device=n.device).reshape(T, T, 1)).reshape(T, T * K)
            comb = torch.index_select(comb, 1, perm)
        elif MOE_BMM and not MOE_DEBUG and gus is not None and dc is not None and not EXACT:
            a, n8 = quant8(n)                                                          # [T, 1], [T, H] fp8
            hgu = bmm8(n8, gu.reshape(T, K * 2 * EIl, g.H)).float() * a * gus.reshape(T, K * 2 * EIl)
            hgu = G.bf(hgu).reshape(T, K, 2, EIl)
            act = bf(silu_mul(hgu[:, :, 0], hgu[:, :, 1])) * w.unsqueeze(-1) * dc.reshape(T, K, EIl)
            a2 = act.reshape(T, K * EIl)
            a = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
            routed = bmm8t((a2 / a).to(F8), dw.reshape(T, K * EIl, g.H)).float() * a
            return routed, routed
        if perm is None:
            comb = (w.unsqueeze(1) * torch.eye(T, device=n.device).reshape(T, T, 1)).reshape(T, T * K)   # row r uses its own K slots only
        if ROW_FORM == 'full':      # probe: the combine weights through the [T, E] matrix and a gather, as the union form does
            full = (w.unsqueeze(-1) * (top.unsqueeze(-1) == torch.arange(E, device=n.device).reshape(1, 1, E)).float()).sum(dim=1)
            comb = torch.index_select(full, 1, idx) * torch.eye(T, device=n.device).repeat_interleave(K, dim=1)
    elif mode == 'gather' and g.topk * T < E:
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
        gu, dw = untile(torch.index_select(gu, 0, idx)), torch.index_select(dw, 0, idx)
        gus = torch.index_select(gus, 0, idx) if gus is not None else None
        dc = torch.index_select(dc, 0, idx) if dc is not None else None
    E_ = gu.shape[0]
    if MOE_DEBUG == 'comb':
        return None, comb
    if MOE_DEBUG == 'shared':
        return None, None
    tt = {'w': gu.reshape(E_ * 2 * EIl, g.H)}
    if gus is not None:
        tt['w_s'] = gus.reshape(-1)
    hgu = lin(n, tt, 'w').reshape(T, E_, 2, EIl)
    if MOE_DEBUG == 'hgu':
        return None, hgu
    act = bf(silu_mul(hgu[:, :, 0], hgu[:, :, 1])) * comb.unsqueeze(-1)
    if dc is not None:
        act = act * dc.reshape(1, E_, EIl)
    a2 = act.reshape(T, E_ * EIl)
    if MOE_DEBUG == 'act':
        return None, a2
    wd = dw.reshape(E_ * EIl, g.H)
    if dc is not None:
        a = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        routed = G.mm8((a2 / a).to(F8), wd).float() * a
    elif EXACT:
        routed = torch.matmul(a2, wd.float())
    else:
        routed = torch.matmul(a2.to(torch.bfloat16), wd.to(torch.bfloat16)).float()
    return routed, routed


def bmm8(a8, w8):
    """per-row fp8 contraction: a8 [T, H], w8 [T, O, H] -> [T, O] (f32 accumulate on the device; widened on the CPU mirror)."""
    if a8.device.type == 'cpu':
        return torch.einsum('th,toh->to', a8.float(), w8.float())
    return torch.matmul(a8.unsqueeze(1), w8.transpose(1, 2)).squeeze(1)


def bmm8t(a8, w8):
    """per-row fp8 contraction over the first weight axis: a8 [T, I], w8 [T, I, H] -> [T, H]."""
    if a8.device.type == 'cpu':
        return torch.einsum('ti,tih->th', a8.float(), w8.float())
    return torch.matmul(a8.unsqueeze(1), w8).squeeze(1)


def silu_mul(gate, up):
    return gate * torch.sigmoid(gate) * up


def sublayer(kind, t, g, R, Q, P, allreduce, moe_mode='all'):
    """One sub-block with its pre-norm and residual -> (x_new, updated states)."""
    x = t['x']
    n = bf(rms(x, t['norm'], g.eps))
    full = allreduce
    if AR_BF16 == '1' or (AR_BF16 == 'moe' and kind == 'moe'):
        allreduce = lambda v: full(v.to(torch.bfloat16)).float()   # each rank's partial rounded to bf16, summed in bf16
    if kind == 'gdn':
        y, st = gdn_block(n, t, g, R, Q, allreduce)
    elif kind == 'gqa':
        y, st = gqa_block(n, t, g, R, Q, P, allreduce)
    elif kind == 'moe':
        y, st = moe_block(n, t, g, allreduce, moe_mode, ar_route=full)                # the router logits are always reduced in f32
    else:
        raise ValueError(kind)
    return x + bf(y), st


def segment(seg, t, g, R, Q, P, allreduce, moe_mode='all'):
    """Consecutive whole layers (attention kinds `seg`, each followed by its MoE) in one graph. Inputs l<i>.a.* / l<i>.m.* are
    layer i's attention / MoE weights, l<i>.<state> its state; the step inputs are shared. -> (x, {l<i>.<state>: new})."""
    common = {k: v for k, v in t.items() if '.' not in k}
    x, states = t['x'], {}
    for i, a_kind in enumerate(seg):
        pre = f'l{i}.'
        own = {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
        att = {k[2:]: v for k, v in own.items() if k.startswith('a.')}
        mlp = {k[2:]: v for k, v in own.items() if k.startswith('m.')}
        st_in = {k: v for k, v in own.items() if '.' not in k}
        x, st = sublayer(a_kind, {**common, **att, **st_in, 'x': x}, g, R, Q, P, allreduce, moe_mode)
        states.update({pre + k: v for k, v in st.items()})
        x, _ = sublayer('moe', {**common, **mlp, 'x': x}, g, R, Q, P, allreduce, moe_mode)
    return x, states


def sequence(kinds, t, g, R, Q, P, allreduce, moe_mode='all'):
    """Bench graphs (kind seq_<k>_<k>...): consecutive SUB-blocks (gdn | gqa | moe), each with its own weights l<i>.a.* (attention)
    or l<i>.m.* (moe) and states l<i>.<state>. layer_bench times them; the slope over the repeat count is the in-launch cost
    of one sub-block (docs/neuron-qwen397-batch.md)."""
    common = {k: v for k, v in t.items() if '.' not in k}
    x, states = t['x'], {}
    for i, kind in enumerate(kinds):
        pre = f'l{i}.'
        own = {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
        w = {k[2:]: v for k, v in own.items() if k[:2] in ('a.', 'm.')}
        st_in = {k: v for k, v in own.items() if '.' not in k}
        x, st = sublayer(kind, {**common, **w, **st_in, 'x': x}, g, R, Q, P, allreduce, moe_mode)
        states.update({pre + k: v for k, v in st.items()})
    return x, states


def mtp_block(t, g, gm, Q, P, allreduce):
    """The MTP head on Q entries of one request (qwen35_moe_ref.Mtp): entry i pairs the hidden state of position p_i with the
    embedding t['e'][i] of the NEXT token. t['x'] [Q, H] is a residual BEFORE its norm t['hnorm']: the main model's final
    residual with the final norm (the first pass after a decode / verify step; the engine binds that step's buffer), or
    the previous pass's own output row with mtp.norm (chained drafts): one NEFF serves both by the tensor bound to hnorm.
    fc is column-parallel (this rank's 2H / TP columns of cat(embedding, hidden), t['fcols']) + all-reduce; the layer is
    a gqa + moe pair with weights a.* / m.* of geometry gm (the MTP layer's own expert count) and its own paged KV cache
    t['kv']; t['pos'] / t['lo'] are relative to the first key of t['table'] (a window of the request's pages; RoPE rows
    use the absolute position). -> (z[sel] [1, H] pre-norm, this rank's logits shard of row sel [1, V / TP], {'kv'})."""
    h = rms(t['x'], t['hnorm'], g.eps)
    cat = torch.cat((rms(t['e'], t['en'], g.eps), rms(h, t['hn'], g.eps)), dim=-1)
    z = allreduce(lin_partial(bf(torch.index_select(cat, 1, t['fcols'])), t, 'fc'))
    common = {k: t[k] for k in ('pos', 'cos_sin', 'wslot', 'table', 'lo')}
    sub = lambda pre: {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
    z, st = sublayer('gqa', {**common, **sub('a.'), 'kv': t['kv'], 'x': z}, gm, 1, Q, P, allreduce)
    z, _ = sublayer('moe', {**sub('m.'), 'x': z}, gm, 1, Q, P, allreduce, 'gather')
    logits = head({'x': z, 'norm': t['mnorm'], 'w': t['w'], **({'w_s': t['w_s']} if 'w_s' in t else {})}, g, t['sel'])
    return torch.index_select(z, 0, t['sel']), logits, st


def cross_rank_argmax(logits, t, g, allreduce):
    """In-graph greedy pick over the vocabulary-sharded logits (the draft chain in one launch): this rank's shard row
    logits [1, V / TP] -> (embedding row [1, H] f32 of the global argmax, its token id [1, 1] f32) on EVERY rank, by two
    small all-reduces: the ranks' local maxima ([TP], one-hot by rank t['rhot']), then the winner's (embedding row | id)
    ([H + 1]; the other ranks add zeros). t['vbias'] [V / TP] masks the padded vocabulary rows, t['emb'] [V / TP, H] is
    this rank's slice of the embedding table, t['rbase'] [1] its first token id. An exact tie between two ranks adds two
    rows (an invalid id, the host drops such a draft): these are drafts, the verify step decides."""
    l = torch.where(t['vbias'].reshape(1, -1) < 0, t['vbias'].reshape(1, -1), logits)   # not logits + bias: a padded row may hold anything
    if MTP_ARGMAX == 'eq':                                                             # no argmax op: the last index reaching the maximum
        m = torch.amax(l, dim=-1)
        a = torch.amax(torch.where(l >= m.reshape(1, 1), torch.arange(l.shape[1], dtype=torch.float32, device=l.device).reshape(1, -1),
                                   torch.zeros_like(l)), dim=-1).long()
    else:
        m, a = torch.max(l, dim=-1)
    M = allreduce(t['rhot'] * m)
    # this rank wins when its maximum reaches every rank's. On the device `M >= amax(M)` came out true on EVERY rank (the
    # drafts were the sum of all ranks' picks), so no full reduce-max and, by default, no comparison: clamp arithmetic
    if MTP_WIN == 'cmp':
        win = (torch.sum((m.reshape(1) >= M).float()) >= g.TP - 0.5).float().reshape(1, 1)
    else:
        win = torch.clamp(torch.sum(torch.clamp((m.reshape(1) - M) * 1e30 + 1.0, 0.0, 1.0)) - (g.TP - 1), 0.0, 1.0).reshape(1, 1)
    row = torch.index_select(t['emb'], 0, a).float()
    out = allreduce(torch.cat((row, (a.float() + t['rbase']).reshape(1, 1)), dim=-1) * win)
    if MTP_CHAIN_DBG:                                                                  # device bisection: the ranks' maxima and this rank's (win, m, a)
        return out[:, :g.H], torch.cat((out[:, g.H:], M.reshape(1, -1), win, m.reshape(1, 1), a.float().reshape(1, 1)), dim=-1)
    return out[:, :g.H], out[:, g.H:]


def mtp_chain_block(t, g, gm, Q, P, D, allreduce):
    """D drafts in ONE launch: the first pass of mtp_block over Q entries, then D - 1 chained passes of the same layer on its
    own output (x <- z[sel] with mtp.norm, e <- the embedding of the previous draft picked in the graph), each writing its
    entry t['cpos'|'ccos'|'cwslot'][j] into the head's KV cache. -> (drafts [1, D] f32 token ids, {'kv'})."""
    z, logits, st = mtp_block(t, g, gm, Q, P, allreduce)
    toks = []
    zero = torch.zeros(1, dtype=torch.int64, device=z.device)
    for j in range(D):
        e, tok = cross_rank_argmax(logits, t, g, allreduce)
        toks.append(tok if j == 0 or not MTP_CHAIN_DBG else tok[:, :1])
        if j == D - 1:
            break
        tj = dict(t)
        tj.update(x=z, hnorm=t['mnorm'], e=e, pos=t['cpos'][j:j + 1], cos_sin=t['ccos'][j:j + 1], wslot=t['cwslot'][j:j + 1], sel=zero, kv=st['kv'])
        z, logits, st = mtp_block(tj, g, gm, 1, P, allreduce)
    if MTP_CHAIN_DBG:                                                                  # drafts first, then the first pick's debug values
        return torch.cat([tk[:, :1] for tk in toks] + [toks[0][:, 1:]], dim=-1), st
    return torch.cat(toks, dim=-1), st


def mtp_chain_shapes(g, gm, Q, P, D, fp8=True):
    step, w = mtp_shapes(g, gm, Q, P, fp8)
    if D > 1:
        step.update(cpos=((D - 1,), torch.int64), ccos=((D - 1, 2 * g.rot), torch.float32), cwslot=((D - 1,), torch.int64))
    VS = g.VP // g.TP
    w.update(vbias=((VS,), torch.float32), rhot=((g.TP,), torch.float32), rbase=((1,), torch.float32), emb=((VS, g.H), torch.bfloat16))
    return step, w


def mtp_geo(g, experts=None):
    """Geometry of the MTP layer: the main text geometry with one full-attention layer and the head's own expert count."""
    raw = dict(g.raw)
    raw.update(num_hidden_layers=1, layer_types=['full_attention'], num_experts=experts or int(os.environ.get('MTP_EXPERTS', g.E)))
    return Geo(raw, g.TP)


def mtp_shapes(g, gm, Q, P, fp8=True):
    """(step, weights) name -> (shape, dtype) of the mtp graph for one rank, in input order."""
    H, f32, i64 = g.H, torch.float32, torch.int64
    step = {'x': ((Q, H), f32), 'e': ((Q, H), f32), 'pos': ((Q,), i64), 'cos_sin': ((Q, 2 * g.rot), f32), 'wslot': ((Q,), i64),
            'table': ((1, P), i64), 'lo': ((1,), i64), 'sel': ((1,), i64)}
    wt = F8 if fp8 else torch.bfloat16
    w = {'hnorm': ((H,), f32), 'en': ((H,), f32), 'hn': ((H,), f32), 'fcols': ((2 * H // g.TP,), i64), 'fc': ((H, 2 * H // g.TP), wt)}
    if fp8:
        w['fc_s'] = ((H,), f32)
    w.update({'a.' + k: v for k, v in weight_shapes('gqa', gm, fp8).items()})
    w.update({'m.' + k: v for k, v in weight_shapes('moe', gm, fp8).items()})
    w['mnorm'] = ((H,), f32)
    w['w'] = ((g.VP // g.TP, H), wt)
    if fp8:
        w['w_s'] = ((g.VP // g.TP,), f32)
    return step, w


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
        s['sgate'] = ((H,), f32)
        if ROUTER == 'shard':
            s['rws'] = ((g.E, H // g.TP), f32); s['rcols'] = ((H // g.TP,), torch.int64)
        else:
            s['rw'] = ((g.E, H), f32)
        s['egu'] = ((g.E * 128, 2 * g.EIl * H // 128), wt) if EGU_T else ((g.E, 2 * g.EIl, H), wt)
        s['ed'] = ((g.E, g.EIl, H), wt)
        if fp8:
            s['egu_s'] = ((g.E, 2 * g.EIl), f32); s['ed_c'] = ((g.E, g.EIl), f32)
    return s


def state_shapes(kind, g, pages, slots):
    if kind == 'gdn':
        slots = 1 if GDN_SLOT == 'bound' else slots
        return {'kstate': ((slots, g.hl, g.Dk, g.Dv), torch.float32), 'cstate': ((slots, g.W3, g.K - 1), torch.float32)}
    if kind == 'gqa':
        if KV8:   # fp8 codes + per-(token, K|V, head) f32 scales: 8.25 KiB per page per layer and core instead of 16
            return {'kv': ((pages, 2, g.kvl, PAGE, g.D), torch.int8 if KV_CODE == 'int8' else F8), 'kvs': ((pages, 2, g.kvl, PAGE), torch.float32)}
        return {'kv': ((pages, 2, g.kvl, PAGE, g.D), torch.float32 if EXACT else torch.bfloat16)}
    return {}


def step_shapes(kind, R, Q, P, g):
    T = R * Q
    s = {'x': ((T, g.H), torch.float32)}
    if kind in ('gdn', 'gqa'):
        s['pos'] = ((T,), torch.int64)
    if kind == 'gdn' and GDN_SLOT != 'bound':
        s['slot'] = ((R,), torch.int64)
    if kind == 'gdn' and GDN_SLOT != 'bound':
        if VERIFY and Q > 1:                                                           # R streams x Q tokens each (R > 1: bench only so far)
            s['vslot'] = ((R * Q,), torch.int64)
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
    elif kind in ('mtp', 'mtpc'):                                                      # mtpc 1 Q P with MTP_CHAIN=D: D drafts in one launch
        R, Q, P = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        D = int(os.environ.get('MTP_CHAIN', '2')) if kind == 'mtpc' else 0
        gm = mtp_geo(g)
        step, wsh = mtp_chain_shapes(g, gm, Q, P, D, fp8) if D else mtp_shapes(g, gm, Q, P, fp8)
        ssh = state_shapes('gqa', gm, pages, slots)
        names, snames = list(step) + list(wsh), list(ssh)
        inputs = [zeros(*step[n]) for n in step] + [torch.ones(*wsh[n][0]) if n.endswith('_s') or n.endswith('_c') else zeros(*wsh[n]) for n in wsh]

        class MtpBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('kv', zeros(*ssh['kv']))

            def forward(self, *a):
                t = dict(zip(names, a))
                t['kv'] = self.kv
                if D:
                    toks, st = mtp_chain_block(t, g, gm, Q, P, D, allreduce)
                    self.kv = st['kv']
                    return toks
                z, logits, st = mtp_block(t, g, gm, Q, P, allreduce)
                self.kv = st['kv']
                return z, logits
        fn = MtpBlock()
        kwargs = {'enable_aliasing': True}
        meta = {'kind': 'qwen35_' + kind, 'seg': [], 'R': 1, 'Q': Q, 'P': P, 'D': D, 'pages': pages, 'slots': slots, 'inputs': names, 'states': snames, 'fp8': fp8,
                'outputs': ['drafts'] if D else ['z', 'logits'], 'mtp_experts': gm.E, 'moe_slots': MOE_SLOTS, 'router': ROUTER, 'route': ROUTE, 'row_form': ROW_FORM,
                'att_bf16': ATT_BF16, 'att_gather': ATT_GATHER, 'lin_cse': LIN_CSE}
    else:
        R, Q, P = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        seg = kind.split('_')[1:] if kind.startswith('seg_') else None       # seg_gdn_gdn_gdn_gqa: consecutive whole layers in one NEFF
        seq = kind.split('_')[1:] if kind.startswith('seq_') else None       # seq_moe_moe_...: sub-blocks (bench only)
        parts = kind.split('_')[1:] if kind.startswith('layer_') else [kind]
        step, wsh, ssh = {}, {}, {}
        if seq:
            for i, k_ in enumerate(seq):
                step.update(step_shapes(k_, R, Q, P, g))
                wsh.update({f'l{i}.{"m" if k_ == "moe" else "a"}.{k}': v for k, v in weight_shapes(k_, g, fp8).items()})
                ssh.update({f'l{i}.{k}': v for k, v in state_shapes(k_, g, pages, slots).items()})
        elif seg:
            for i, a_kind in enumerate(seg):
                step.update(step_shapes(a_kind, R, Q, P, g))
                wsh.update({f'l{i}.a.{k}': v for k, v in weight_shapes(a_kind, g, fp8).items()})
                wsh.update({f'l{i}.m.{k}': v for k, v in weight_shapes('moe', g, fp8).items()})
                if a_kind == 'gdn' and GDN_SLOT == 'bound' and R > 1:   # one state pair per row, as in the bound gdn sub-block
                    ssh.update({f'l{i}.{k}_{r}': v for r in range(R) for k, v in state_shapes(a_kind, g, pages, slots).items()})
                else:
                    ssh.update({f'l{i}.{k}': v for k, v in state_shapes(a_kind, g, pages, slots).items()})
        else:
            for i, part in enumerate(parts):
                step.update(step_shapes(part, R, Q, P, g))
                pre = ('a.', 'm.')[i] if len(parts) > 1 else ''
                wsh.update({pre + k: v for k, v in weight_shapes(part, g, fp8).items()})
            ssh = state_shapes(parts[0], g, pages, slots)
        xnames = []   # GDN_SLOT=bound verify: extra (non-aliased) state outputs after x
        if kind == 'gdn' and GDN_SLOT == 'bound':
            if R > 1:
                ssh = {f'{k}_{r}': v for r in range(R) for k, v in ssh.items()}
            if VERIFY and Q > 1:
                xnames = [f'{k}@{i}' for i in range(1, Q) for k in ('kstate', 'cstate')]
        elif seg and GDN_SLOT == 'bound' and VERIFY and Q > 1:   # bound verify segment (R = 1): per-position states of every gdn layer
            assert R == 1, 'GDN_SLOT=bound verify segments: one row'
            xnames = [f'l{j}.{k}@{i}' for j, a_kind in enumerate(seg) if a_kind == 'gdn' for i in range(1, Q) for k in ('kstate', 'cstate')]
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
                    self.register_buffer(n_.replace('.', '_'), zeros(*ssh[n_]))

            def forward(self, *a):
                t = dict(zip(names, a))
                for n_ in snames:
                    t[n_] = getattr(self, n_.replace('.', '_'))
                if seq:
                    x, st = sequence(seq, t, g, R, Q, P, allreduce, moe_mode)
                elif seg:
                    x, st = segment(seg, t, g, R, Q, P, allreduce, moe_mode)
                elif len(parts) == 1:
                    x, st = sublayer(kind, t, g, R, Q, P, allreduce, moe_mode)
                else:
                    sub = lambda pre: {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
                    common = {k: v for k, v in t.items() if '.' not in k}
                    x, st = sublayer(parts[0], {**common, **sub('a.')}, g, R, Q, P, allreduce, moe_mode)
                    x, _ = sublayer(parts[1], {**common, **sub('m.'), 'x': x}, g, R, Q, P, allreduce, moe_mode)
                for n_ in snames:
                    setattr(self, n_.replace('.', '_'), st[n_])   # mutated buffers: aliased input -> output (in place on the device)
                return (x, *[st[n_] for n_ in xnames]) if xnames else x
        fn = Block()
        kwargs = {'enable_aliasing': True} if snames else {}
        meta = {'kind': 'qwen35_' + kind, 'seg': seg or [], 'R': R, 'Q': Q, 'P': P, 'pages': pages, 'slots': slots, 'inputs': names,
                'states': snames, 'fp8': fp8, 'moe': moe_mode, 'moe_slots': MOE_SLOTS, 'moe_bmm': MOE_BMM, 'router': ROUTER, 'lin_cse': LIN_CSE, 'gdn_dec': GDN_DEC, 'gdn_form': GDN_FORM, 'row_form': ROW_FORM, 'moe_split': MOE_SPLIT, 'moe_kernel': MOE_KERNEL, 'moe_kstat': MOE_KSTAT, 'moe_kskip': MOE_KSKIP, 'moe_kyd': MOE_KYD, 'moe_kdense': MOE_KDENSE, 'moe_kfast': MOE_KFAST, 'moe_kpre': MOE_KPRE, 'egu_t': EGU_T, 'route': ROUTE, 'ar_bf16': AR_BF16, 'gdn_chunk': GDN_CHUNK, 'gdn_inv': GDN_INV, 'gdn_loop': GDN_LOOP, 'rlogit': RLOGIT, 'att_bf16': ATT_BF16, 'kv8': KV8, 'kv_code': KV_CODE, 'att_block': ATT_BLOCK, 'att_gather': ATT_GATHER, 'gdn_gather': GDN_GATHER, 'gdn_einsum': GDN_EINSUM, 'gdn_slot': GDN_SLOT, 'extra_outputs': xnames,
                'verify': bool(VERIFY and Q > 1)}
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
