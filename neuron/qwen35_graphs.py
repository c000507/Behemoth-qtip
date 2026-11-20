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
MOE_CAP = int(os.environ.get('MOE_CAP', '32'))     # MOE=group (prefill): slots per expert of the cold tier (tokens of one expert in one chunk)
MOE_HOT = int(os.environ.get('MOE_HOT', '16'))     # MOE=group: the MOE_HOT most loaded experts of the chunk run on every token (hot tier)
MOE_GE = int(os.environ.get('MOE_GE', '0'))        # MOE=group: cold-tier experts per pass (gather .. scatter); 0: all at once
MOE_GSTAT = int(os.environ.get('MOE_GSTAT', '0'))  # MOE=group: also output the MOE_GSTAT largest expert loads of the chunk (sizing MOE_CAP / MOE_HOT from real prompts)
MOE_GG = os.environ.get('MOE_GG', 'idx')           # MOE=group token gather: idx (index_select of the fp8 rows) | mm (one-hot fp8 matmul)
MOE_GD = os.environ.get('MOE_GD', 'slot2')         # MOE=group, slot outputs back to the tokens by a one-hot matmul [T, E C] x [E C, H]: slot2 (the slot
                                                   # scale folded into the one-hot matrix, bf16 operands) | slot (f32 slot outputs scaled first)
MOE_GF = os.environ.get('MOE_GF', 'onehot')       # MOE=group slot bookkeeping: onehot ([E, C, T] one-hot matrices, the Q = 256 form) | idx (wide chunks,
                                                   # Q >= 1024: slots filled by a scatter of the K T pairs, outputs read back by a row gather; moe_grouped_idx) |
                                                   # sub (the onehot form per MOE_TSUB-token sub-block of a wide chunk)
MOE_GB = os.environ.get('MOE_GB', 'sc1')            # MOE_GF=idx, slot outputs back to the tokens: sc1 / scg (assign-scatter of the slot rows into a [K T, H] pair
                                                   # table, once / per expert group) | gather (row gather of the K T pairs from the slot table: 44 ms at Q = 1024)
MOE_GBLK = int(os.environ.get('MOE_GBLK', '0'))      # MOE_GF=idx: a chunk wider than this as independent grouped blocks of MOE_GBLK tokens (0: one block)
MOE_TIERS = os.environ.get('MOE_TIERS', '')         # MOE_GF=idx: 'n:c,n:c,...' warm tiers after the MOE_HOT hot experts, in load order: n experts with c slots each
MOE_TAIL = os.environ.get('MOE_TAIL', 'nat')        # MOE_GF=idx, the experts after the tiers (MOE_CAP slots each): nat (natural order, static weights) | rank (load order)
MOE_GSLOTS = int(os.environ.get('MOE_GSLOTS', '3072'))   # MOE_GF=idx: slots per expert group (gather .. scatter pass) of the tiered / load-ordered segments
MOE_GSTAGE = os.environ.get('MOE_GSTAGE', '')        # bench only (MOE_GF=idx): stop after book | gather | gu | down (the result is NOT the MoE)
MOE_TSUB = int(os.environ.get('MOE_TSUB', '0'))    # wide prefill chunks: MOE=all and the hot tier of MOE_GF=idx in token sub-blocks of this many rows
ATT_QBLOCK = int(os.environ.get('ATT_QBLOCK', '0'))  # KV8 attention of wide prefill chunks: queries in blocks of this many tokens (x ATT_BLOCK key blocks)
WT = os.environ.get('WT', '0') == '1'   # dense projections, router, head and the experts' gate|up stored TRANSPOSED [in, out] (contraction axis first):
                                         # the M = 1 matmul then needs no on-chip transpose of the weight tiles (decode lever, docs/neuron-reap-decode-profile.md)
GDN_RSPLIT = int(os.environ.get('GDN_RSPLIT', '0'))   # decode (Q = 1): the gdn state contractions in row groups of this many (R = 8 -> 2 x 4)
WT_MOE = os.environ.get('WT_MOE', '')   # WT experts' gate|up: '' (decode: one matmul over the gathered slices relaid [H, E_ 2EIl]; MOE=all: stack) | stack | cat
PROF_NOAR = os.environ.get('PROF_NOAR', '0') == '1'   # timing probes only (WRONG output): the in-graph all-reduces are replaced by the identity
PROF_FIXED = os.environ.get('PROF_FIXED', '0') == '1'   # timing probes only (WRONG output): gathered decode MoE uses experts 0..8T-1 by a static slice (no expert gather)
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


def wmat(t, name):
    """the [in, out] weight of a projection (WT: as stored; else the transpose view of the stored [out, in])."""
    return t[name] if WT else t[name].T


WT_PROJ = {'qkv', 'z', 'b', 'a', 'o', 'q', 'k', 'v', 'sg', 'su', 'sd', 'fc'}


def to_wt(name, w):
    """A prepared tensor (the [out, in] layout of qwen35_prepare) -> its WT=1 layout: projections and the router [in, out],
    the experts' gate|up [E, H, 2 EIl], the head [H, V / TP]. Scales and every other tensor are unchanged (ed is already
    [E, EIl, H] = [in, out] per expert)."""
    if name in WT_PROJ or name == 'rw' or (name == 'w' and w.dim() == 2):
        return w.T.contiguous()
    if name == 'egu' and w.dim() == 3:
        return w.permute(0, 2, 1).contiguous()
    return w


def lin(h, t, name):
    if WT:
        w = t[name]
        h2 = h.reshape(-1, h.shape[-1])
        if EXACT:
            return torch.matmul(h2, w.float()).reshape(*h.shape[:-1], w.shape[1])
        if name + '_s' in t:
            a, h8 = quant8(h2)
            y = G.mm8(h8, w).float() * a * t[name + '_s'].reshape(1, -1)
        else:
            y = torch.matmul(h2.to(torch.bfloat16), w.to(torch.bfloat16)).float()
        return G.bf(y).reshape(*h.shape[:-1], w.shape[1])
    if EXACT:
        return torch.matmul(h.reshape(-1, h.shape[-1]), t[name].float().T).reshape(*h.shape[:-1], t[name].shape[0])
    if LIN_CSE and name + '_s' in t and h.dim() == 2:
        a, h8 = quant8(h)
        return G.bf(G.mm8(h8, t[name].T).float() * a * t[name + '_s'].reshape(1, -1))
    return G.lin(h, t, name)


def lin_partial(h, t, name):
    if WT:
        w = t[name]
        if EXACT:
            return torch.matmul(h, w.float())
        if name + '_s' in t:
            a = torch.amax(torch.abs(h.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
            return G.mm8((h.float() / a).to(F8), w).float() * a * t[name + '_s'].reshape(1, -1)
        return torch.matmul(h.to(torch.bfloat16), w.to(torch.bfloat16)).float()
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
    elif Q == 1 and GDN_RSPLIT and R > GDN_RSPLIT and R % GDN_RSPLIT == 0 and not verify:
        # decode rows in groups of GDN_RSPLIT for the state contractions (R = 8 compiles into tiny f32 tiles; R = 4 does not)
        ys, nsts = [], []
        for r0 in range(0, R, GDN_RSPLIT):
            sl = slice(r0, r0 + GDN_RSPLIT)
            nst = st[sl] * decay[sl, 0].reshape(GDN_RSPLIT, hl, 1, 1)
            mem = torch.einsum('rhkd,rhk->rhd', nst, kn[sl, 0])
            delta = (v[sl, 0] - mem) * beta[sl, 0].unsqueeze(-1)
            nst = nst + kn[sl, 0].unsqueeze(-1) * delta.unsqueeze(-2)
            ys.append(torch.einsum('rhkd,rhk->rhd', nst, qn[sl, 0]))
            nsts.append(nst)
        st = torch.cat(nsts, dim=0)
        sts = [st]
        y = torch.cat(ys, dim=0).unsqueeze(1)                                          # [R, 1, hl, Dv]
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
    # ATT_QBLOCK: the queries of a wide chunk in blocks of qb tokens, each with its own running max / sum / accumulator, so the
    # score tensor of one (query block, key block) cell keeps the size it has in the Q = 256 graphs. Every key block is
    # gathered once and used by all query blocks.
    qb = ATT_QBLOCK if ATT_QBLOCK and Q > ATT_QBLOCK and Q % ATT_QBLOCK == 0 else Q
    nq = Q // qb
    m, l, acc = [None] * nq, [None] * nq, [None] * nq
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
        kpos = torch.arange(b0 * PAGE, b0 * PAGE + keys, dtype=torch.int64, device=n.device).reshape(1, 1, 1, 1, keys)
        for i in range(nq):
            qi, pi = (qd, pos) if nq == 1 else (qd[:, i * qb:(i + 1) * qb], pos[:, i * qb:(i + 1) * qb])
            s = torch.einsum('rqhgd,rhkd->rqhgk', qi, kk).float() * (ks / math.sqrt(D))
            s = torch.where(kpos <= pi, s, torch.full_like(s, -3e38))
            mb = torch.amax(s, dim=-1)
            if m[i] is None:   # block 0 holds key 0, visible to every real row: m is finite from here on
                m[i] = mb
                p = torch.exp(s - mb.unsqueeze(-1))
                l[i] = p.sum(-1)
                acc[i] = torch.einsum('rqhgk,rhkd->rqhgd', bf(p * vs).to(dt), vv).float()
            else:
                mn = torch.maximum(m[i], mb)
                c = torch.exp(m[i] - mn)
                p = torch.exp(s - mn.unsqueeze(-1))
                l[i] = l[i] * c + p.sum(-1)
                acc[i] = acc[i] * c.unsqueeze(-1) + torch.einsum('rqhgk,rhkd->rqhgd', bf(p * vs).to(dt), vv).float()
                m[i] = mn
    acc, l = (acc[0], l[0]) if nq == 1 else (torch.cat(acc, dim=1), torch.cat(l, dim=1))
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
        logits = torch.matmul(n.to(torch.bfloat16), wmat(t, 'rw').to(torch.bfloat16)).float()
    else:
        logits = torch.matmul(n.float(), wmat(t, 'rw'))
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
    if mode == 'group' and MOE_GF == 'idx':
        top, w = route(n, t, g, ar_route or allreduce, want_top=True)                  # [T, K] expert indices, renormalised weights
        if MOE_GBLK and T > MOE_GBLK and T % MOE_GBLK == 0:
            # a chunk wider than MOE_GBLK as independent grouped blocks of MOE_GBLK tokens (each its own hot set and slots:
            # the amortisation of the expert weights saturates at 128 slot rows, i.e. at blocks of ~1024 tokens, while the
            # token gather grows with T^2); ovf = the worst block
            parts = [moe_grouped_idx(n[b:b + MOE_GBLK], top[b:b + MOE_GBLK], w[b:b + MOE_GBLK], {**t, 'pos': t['pos'][b:b + MOE_GBLK]}, g)
                     for b in range(0, T, MOE_GBLK)]
            return allreduce(shared + torch.cat([p_[0] for p_ in parts], dim=0)), {'ovf': torch.stack([p_[1] for p_ in parts], dim=0).amax(dim=0)}
        routed, ovf = moe_grouped_idx(n, top, w, t, g)
        return allreduce(shared + routed), {'ovf': ovf}
    comb = route(n, t, g, ar_route or allreduce, want_top=rowslots)                    # [T, E] | ([T, K], [T, K])
    if rowslots and MOE_DEBUG == 'route':
        top, w = comb
        comb = (w.unsqueeze(-1) * (top.unsqueeze(-1) == torch.arange(g.E, device=n.device).reshape(1, 1, g.E)).float()).sum(dim=1)
    if MOE_DEBUG in ('route', 'probs'):
        v = (comb if MOE_DEBUG == 'route' else torch.softmax(torch.matmul(n.float(), t['rw'].T), dim=-1)).reshape(T, -1).float()[:, :g.H]
        alive = sum(w.reshape(-1)[:1].float().sum() for k, w in t.items() if k not in ('x', 'norm', 'rcols')) * 0.0
        return allreduce(torch.nn.functional.pad(v, (0, g.H - v.shape[1])) + alive), {}
    if mode == 'group':
        # padding tokens (position -1) take no expert: their residual is whatever the attention made of an all-masked row
        # (it reads the scratch page, whose content is not reproducible), and in the grouped form a routed padding token
        # would occupy slots and move the hot / cold split of the REAL tokens, i.e. their fp8 rounding path. Without
        # them a real token's output depends on real tokens only, as with MOE=all.
        comb = comb * (t['pos'] >= 0).float().reshape(T, 1)
        if MOE_GF == 'sub' and MOE_TSUB and T > MOE_TSUB and T % MOE_TSUB == 0:
            # a wide chunk as T / MOE_TSUB independent grouped blocks (each its own hot set and slots: the Q = MOE_TSUB graph
            # repeated in one launch); ovf = the largest cold load of any block
            parts = [moe_grouped(n[b:b + MOE_TSUB], comb[b:b + MOE_TSUB], t, g) for b in range(0, T, MOE_TSUB)]
            routed = torch.cat([p_[0] for p_ in parts], dim=0)
            ovf = torch.stack([p_[1] for p_ in parts], dim=0).amax(dim=0)
            return allreduce(shared + routed), {'ovf': ovf}
        routed, ovf = moe_grouped(n, comb, t, g)
        return allreduce(shared + routed), {'ovf': ovf}
    groups = MOE_SPLIT if (mode == 'gather' and MOE_SPLIT and T > MOE_SPLIT and T % MOE_SPLIT == 0 and not MOE_DEBUG) else T
    if mode == 'all' and MOE_TSUB and T > MOE_TSUB and T % MOE_TSUB == 0 and not MOE_DEBUG:
        groups = MOE_TSUB   # a wide chunk in token sub-blocks: the [sub, E * 2 EIl] intermediates stay those of the Q = MOE_TSUB graph
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


def moe_grouped(n, comb, t, g):
    """Prefill MoE without running every expert on every token (MOE=group). A chunk of T tokens makes K T (token, expert)
    pairs over E experts, so an expert sees K T / E tokens on average (10 of 256 at E = 205) while MOE=all contracts it with
    all T. Two tiers, both static shapes:
      hot   the MOE_HOT most loaded experts of the chunk (generalists) are gathered by index and run on every token, as
            the decode graphs' MOE=gather does for its slot union;
      cold  every other expert e gets MOE_CAP slots: slot c holds the c-th token routed to e. The token rows are gathered
            into [E, C, H], one contraction per expert (bmm8g), and the slot outputs are added back to their tokens by a
            one-hot matmul. MOE_GE experts at a time, so the intermediates of one group stay small.
    -> (routed [T, H] f32 partial of this rank, ovf [1] f32 = the largest load among the cold experts). The result is the
    exact MoE sum iff ovf <= MOE_CAP; otherwise the engine reruns the launch with the MOE=all graph of the bucket (the MoE
    is stateless and the input residual is still intact). Same fp8 recipe as moe_routed (activation rows scaled by
    amax / 224), the down projection quantised per slot instead of per token."""
    T, E, C, S = n.shape[0], g.E, MOE_CAP, min(MOE_HOT, g.E)
    dev = n.device
    sel = (comb > 0).float()                                                           # [T, E]
    load = sel.sum(dim=0, keepdim=True)                                                # [1, E]
    ar = torch.arange(E, device=dev).float().reshape(1, E)
    hot = torch.topk(load - ar * (0.5 / E), S, dim=-1)[1].reshape(S)                   # most loaded first, ties: the lower index
    hotmask = (hot.reshape(S, 1).float() == ar).float().sum(dim=0, keepdim=True)      # [1, E]
    cold = sel * (1.0 - hotmask)
    cload = load * (1.0 - hotmask)                                                     # [1, E]
    ovf = cload.amax(dim=-1)                                                           # [1]
    if MOE_GSTAT:   # + the sorted loads of the most loaded experts (hot ones included): the engine logs them on request
        ovf = torch.cat((ovf, torch.topk(load, MOE_GSTAT, dim=-1)[0].reshape(MOE_GSTAT) if MOE_GSTAT > 0 else load.reshape(E)), dim=0)   # < 0: every expert's load, by expert
    # rank[t, e] = tokens before t routed to e: a 0 / 1 matmul (exact also with bf16 operands; blocks of 256 keep the counts exact)
    rank = torch.cat([torch.matmul(torch.tril(torch.ones(min(256, T - b), min(256, T - b), device=dev), diagonal=-1), cold[b:b + 256])
                      + (cold[:b].sum(dim=0, keepdim=True) if b else 0.0) for b in range(0, T, 256)], dim=0)
    q8 = t.get('egu_s') is not None and not EXACT
    a, n8 = quant8(n) if q8 else (None, None)                                          # [T, 1], [T, H] fp8
    ne = MOE_GE if 0 < MOE_GE < E else E
    out = None
    for e0 in range(0, E, ne):
        part = moe_cold(n, n8, a, comb, cold, rank, cload, t, g, e0, min(E, e0 + ne))
        out = part if out is None else out + part
    return moe_hot(n, comb, t, g, hot, out), ovf


def moe_tiers(E, S):
    """MOE_TIERS 'n:c,n:c,...' -> [(first rank, experts, slots per expert)] of the warm tiers that follow the S hot experts in the
    order of descending load."""
    out, r = [], S
    for item in filter(None, MOE_TIERS.replace('+', ',').split(',')):
        n_, c_ = (int(v) for v in item.split(':'))
        n_ = min(n_, E - r)
        if n_ > 0:
            out.append((r, n_, c_))
            r += n_
    return out


def moe_grouped_idx(n, top, w, t, g):
    """moe_grouped for wide chunks (MOE_GF=idx; Q >= 1024). Same contract (-> routed [T, H], ovf [1]; exact iff ovf <= MOE_CAP),
    but nothing of size [E, C, T] or [T, E C]: the one-hot slot matrices and the one-hot scatter matmul of moe_cold grow with
    T^2 (C grows with T) and would cost more than the experts at Q = 1024. Here
      * the K T (token, expert) pairs are numbered: pair (t, k) of a slotted expert e = top[t, k] gets slot off(e) + (tokens
        before t routed to e); pairs of hot experts, of padding tokens and overflowing ones get none;
      * slot -> token, slot -> combine weight and slot -> pair are scatters of K T scalars (unique targets), empty slots keep
        token 0 and weight 0;
      * the token rows are gathered per expert group, one 2-D matmul pair per expert over its slots: with >= 128 slots an
        expert's weights pass the PE array once per 128 slot rows instead of once per (at most) 32;
      * the slot outputs go back by an assign-scatter of the slot rows into a [K T + 1, H] pair table (unique targets; empty
        slots hit the dump row) that is then summed over K (MOE_GB=sc1 / scg). A row GATHER of the K T pairs from the
        [slots, H] table (MOE_GB=gather) is correct but ~5 us per row on trn1 (44 ms at Q = 1024); index_add returns garbage.
    Tiers by load (a wide chunk's sorted load profile is smooth: rank r takes about the same FRACTION of the chunk whatever
    its width, docs section 14): the MOE_HOT most loaded experts run on every token (dense rows are ~3x cheaper per row than
    slot rows, so only experts above ~25 % load belong here); MOE_TIERS = 'n:c,...' gives the next n experts c slots each
    (their weights gathered in load order); the remaining experts get MOE_CAP slots each - MOE_TAIL=rank: also in load
    order (all weights gathered), nat: in their natural order (static weight slices; the tiered experts' slots stay empty).
    ovf = MOE_CAP + the largest (load - slots) over the slotted experts, so the engine's test ovf > MOE_CAP is "an expert overflowed".
    top [T, K] int64, w [T, K] f32 from route(want_top=True)."""
    T, E, C, S, K, H, EIl = n.shape[0], g.E, MOE_CAP, min(MOE_HOT, g.E), g.topk, g.H, g.EIl
    dev = n.device
    ar = torch.arange(E, device=dev).float().reshape(1, E)
    live = (t['pos'] >= 0).float().reshape(T, 1)                                       # padding tokens take no expert (see moe_block)
    w = w * live
    onehot = (top.unsqueeze(-1).float() == ar.reshape(1, 1, E)).float()                # [T, K, E]
    sel = onehot.sum(dim=1) * live                                                     # [T, E] 0 / 1
    load = sel.sum(dim=0, keepdim=True)                                                # [1, E]
    tiers = moe_tiers(E, S)
    G0 = S + sum(n_ for _, n_, _ in tiers)                                             # experts handled in load order before the tail
    nat = MOE_TAIL == 'nat'
    # most loaded first, ties: the lower index. The whole order is needed only when the tail is slotted in load order.
    order = torch.topk(load - ar * (0.5 / E), G0 if nat else E, dim=-1)[1].reshape(-1)
    hot = order[:S]
    # segments of the slot table: (first slot, experts, slots per expert, expert index tensor or None = natural order from 0)
    segs, base = [], 0
    for r0, n_, c_ in tiers:
        segs.append((base, n_, c_, order[r0:r0 + n_]))
        base += n_ * c_
    if nat:
        segs.append((base, E, C, None))
        base += E * C
    elif G0 < E:
        segs.append((base, E - G0, C, order[G0:]))
        base += (E - G0) * C
    NS = base                                                                          # slots in all
    # per expert: its first slot, its slot count (0: hot)
    offE = torch.zeros(1, E, device=dev) + load[:, :1] * 0.0
    capE = torch.zeros(1, E, device=dev) + load[:, :1] * 0.0
    for b0, n_, c_, ex in segs:
        if ex is None:    # natural-order tail: every expert that is not hot / tiered
            inlist = (order[:G0].reshape(G0, 1).float() == ar).float().sum(dim=0, keepdim=True)     # [1, E]
            offE = offE + (1.0 - inlist) * (b0 + ar * c_)
            capE = capE + (1.0 - inlist) * c_
        else:
            hit = (ex.reshape(n_, 1).float() == ar).float()                            # [n, E] one-hot of the tier's experts
            offE = offE + (hit * (b0 + torch.arange(n_, device=dev).float().reshape(n_, 1) * c_)).sum(dim=0, keepdim=True)
            capE = capE + hit.sum(dim=0, keepdim=True) * c_
    slotted = (capE > 0.5).float()
    ovf = C + ((load - capE) * slotted - (1.0 - slotted) * 1e6).amax(dim=-1)          # [1]
    if MOE_GSTAT:
        ovf = torch.cat((ovf, torch.topk(load, MOE_GSTAT, dim=-1)[0].reshape(MOE_GSTAT) if MOE_GSTAT > 0 else load.reshape(E)), dim=0)
    cold = sel * slotted
    rank = torch.cat([torch.matmul(torch.tril(torch.ones(min(256, T - b), min(256, T - b), device=dev), diagonal=-1), cold[b:b + 256])
                      + (cold[:b].sum(dim=0, keepdim=True) if b else 0.0) for b in range(0, T, 256)], dim=0)   # [T, E] tokens before t on e
    prank = torch.einsum('tke,te->tk', onehot, rank)                                   # [T, K]
    pcold = torch.einsum('tke,te->tk', onehot, cold)                                   # [T, K] 1: a live pair of a slotted expert
    pcap = torch.einsum('tke,te->tk', onehot, capE.expand(T, E))
    poff = torch.einsum('tke,te->tk', onehot, offE.expand(T, E))
    ok = (pcold > 0.5) & (prank < pcap - 0.5)
    slot = (poff + prank).round().long()
    pairs = torch.arange(T * K, device=dev).reshape(T, K)
    idx = torch.where(ok, slot, NS + pairs).reshape(T * K)                             # unique scatter targets
    tokv = torch.div(pairs, K, rounding_mode='floor').float().reshape(T * K)
    zero = torch.zeros(NS + T * K, device=dev) + w.reshape(-1)[:1] * 0.0
    tok = zero.index_put((idx,), tokv)[:NS].round().long()                             # [NS] slot -> token (empty: 0)
    wtab = zero.index_put((idx,), w.reshape(T * K))[:NS]                               # [NS] slot -> combine weight (empty: 0)
    gu, gus, dw, dc = t['egu'], t.get('egu_s'), t['ed'], t.get('ed_c')
    q8 = gus is not None and not EXACT
    comb_hot = (w.unsqueeze(-1) * (top.unsqueeze(-1).float() == hot.float().reshape(1, 1, S)).float()).sum(dim=1)   # [T, S]
    if MOE_GSTAGE == 'book':
        return moe_hot(n, None, t, g, hot, (tok.float().sum() + wtab.sum()) * 1e-30, comb2=comb_hot), ovf
    a, n8 = quant8(n) if q8 else (None, None)
    nb = n if EXACT else n.to(torch.bfloat16)
    ys, ss = [], []
    stage = 0.0
    gi = torch.where(ok, slot, torch.full_like(slot, NS)).reshape(T * K)               # pair -> slot, NS: none
    scat = MOE_GB in ('scg', 'sc1')
    ptab = (zero + float(T * K)).index_put((idx,), pairs.reshape(T * K).float())[:NS].round().long() if scat else None   # slot -> pair (empty: the dump row K T)
    pt = torch.zeros(T * K + 1, H, device=dev) if MOE_GB == 'scg' else None
    for b0, n_, c_, ex in segs:
        ne = (MOE_GE if 0 < MOE_GE else n_) if not tiers and ex is None else max(1, MOE_GSLOTS // c_)
        for e0 in range(0, n_, ne):
            e1 = min(n_, e0 + ne)
            Eg = e1 - e0
            s0, s1 = b0 + e0 * c_, b0 + e1 * c_
            tk = tok[s0:s1]
            if ex is None:
                gug, dwg = gu[e0:e1], dw[e0:e1]
                gusg, dcg = (gus[e0:e1] if gus is not None else None), (dc[e0:e1] if dc is not None else None)
            else:
                ei = ex[e0:e1]
                gug, dwg = torch.index_select(gu, 0, ei), torch.index_select(dw, 0, ei)
                gusg = torch.index_select(gus, 0, ei) if gus is not None else None
                dcg = torch.index_select(dc, 0, ei) if dc is not None else None
            if q8:
                xg = torch.index_select(n8, 0, tk).reshape(Eg, c_, H)
                ag = torch.index_select(a, 0, tk).reshape(Eg, c_, 1)
                if MOE_GSTAGE == 'gather':
                    stage = stage + (xg.float().sum() + ag.sum()) * 1e-30
                    continue
                hgu = bmm8g(xg, gug, wt=True) * ag * gusg.reshape(Eg, 1, 2 * EIl)
            else:
                xg = torch.index_select(nb, 0, tk).reshape(Eg, c_, H)
                hgu = torch.matmul(xg, (gug.float() if EXACT else gug.to(torch.bfloat16)).transpose(1, 2)).float()
                if gusg is not None:
                    hgu = hgu * gusg.reshape(Eg, 1, 2 * EIl)
            hgu = bf(hgu).reshape(Eg, c_, 2, EIl)
            act = bf(silu_mul(hgu[:, :, 0], hgu[:, :, 1])) * wtab[s0:s1].reshape(Eg, c_, 1)   # [Eg, c, EIl]
            if dcg is not None:
                act = act * dcg.reshape(Eg, 1, EIl)
            if MOE_GSTAGE == 'gu':
                stage = stage + act.float().sum() * 1e-30
                continue
            if q8 and dcg is not None:
                s2 = torch.amax(torch.abs(act), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
                y = bmm8g((act / s2).to(F8), dwg)                                      # [Eg, c, H] f32, unscaled
            else:
                s2 = torch.ones(Eg, c_, 1, device=dev)
                y = torch.matmul(act if EXACT else act.to(torch.bfloat16), dwg.float() if EXACT else dwg.to(torch.bfloat16)).float()
            if MOE_GSTAGE == 'down':
                stage = stage + (y.sum() + s2.sum()) * 1e-30
                continue
            if MOE_GB == 'scg':     # the group's slot rows -> their pairs' rows (empty slots -> the dump row)
                pt = pt.index_put((ptab[s0:s1],), (bf(y) * s2).reshape(Eg * c_, H))
            elif MOE_GB == 'sc1':
                ys.append((bf(y) * s2).reshape(Eg * c_, H))
            else:
                ys.append(bf(y).reshape(Eg * c_, H) if EXACT else y.reshape(Eg * c_, H).to(torch.bfloat16))
                ss.append(s2.reshape(Eg * c_))
    if MOE_GSTAGE:
        return moe_hot(n, None, t, g, hot, stage, comb2=comb_hot), ovf
    if scat:
        if MOE_GB == 'sc1':
            pt = torch.zeros(T * K + 1, H, device=dev).index_put((ptab,), torch.cat(ys, dim=0))
        cold_out = pt[:T * K].reshape(T, K, H).sum(dim=1)
    else:
        ytab = torch.cat(ys + [torch.zeros(1, H, device=dev, dtype=ys[0].dtype)], dim=0)   # [NS + 1, H]; row NS: pairs without a slot
        stab = torch.cat(ss + [torch.zeros(1, device=dev)], dim=0)
        yb = torch.index_select(ytab, 0, gi).float().reshape(T, K, H) * torch.index_select(stab, 0, gi).reshape(T, K, 1)
        cold_out = yb.sum(dim=1)                                                       # [T, H]
    return moe_hot(n, None, t, g, hot, cold_out, comb2=comb_hot), ovf


def moe_cold(n, n8, a, comb, cold, rank, cload, t, g, e0, e1):
    """moe_grouped's cold tier for the experts [e0, e1) -> their sum [T, H] f32."""
    T, E, EIl, H, C = n.shape[0], e1 - e0, g.EIl, g.H, MOE_CAP
    dev = n.device
    gu, gus, dw, dc = t['egu'][e0:e1], t.get('egu_s'), t['ed'][e0:e1], t.get('ed_c')   # [E, 2EIl, H], [E, 2EIl], [E, EIl, H], [E, EIl]
    gus, dc = (gus[e0:e1] if gus is not None else None), (dc[e0:e1] if dc is not None else None)
    q8 = n8 is not None
    cold, rank, cload, comb = cold[:, e0:e1], rank[:, e0:e1], cload[:, e0:e1], comb[:, e0:e1]
    slots = torch.arange(C, device=dev).float()
    pm = cold.T.unsqueeze(1) * (rank.T.unsqueeze(1) == slots.reshape(1, C, 1)).float()   # [E, C, T] one-hot (zero rows: empty slots)
    valid = (slots.reshape(1, C) < cload.reshape(E, 1)).float()                        # [E, C]
    tok = (pm * torch.arange(T, device=dev).float().reshape(1, 1, T)).sum(dim=-1).round().long().reshape(E * C)   # empty slots: token 0, weight 0
    wsel = torch.index_select(comb.T.reshape(E * T), 0, torch.arange(E, device=dev).repeat_interleave(C) * T + tok).reshape(E, C) * valid
    if q8:
        if MOE_GG == 'mm':
            xg = G.mm8(pm.reshape(E * C, T).to(F8), n8).float().to(F8)                 # one nonzero term per element: exact
        else:
            xg = torch.index_select(n8, 0, tok)
        ag = torch.index_select(a, 0, tok).reshape(E, C, 1)
        hgu = bmm8g(xg.reshape(E, C, H), gu, wt=True) * ag * gus.reshape(E, 1, 2 * EIl)
    else:
        xg = torch.index_select(n if EXACT else n.to(torch.bfloat16), 0, tok).reshape(E, C, H)
        hgu = torch.matmul(xg, (gu.float() if EXACT else gu.to(torch.bfloat16)).transpose(1, 2)).float()
        if gus is not None:
            hgu = hgu * gus.reshape(E, 1, 2 * EIl)
    hgu = bf(hgu).reshape(E, C, 2, EIl)
    act = bf(silu_mul(hgu[:, :, 0], hgu[:, :, 1])) * wsel.unsqueeze(-1)               # [E, C, EIl]
    if dc is not None:
        act = act * dc.reshape(E, 1, EIl)
    if q8 and dc is not None and MOE_GD == 'slot2':
        # the slot scale rides on the one-hot matrix and both operands of the scatter matmul are bf16 by construction:
        # no pass over the [E C, H] slot outputs besides the contraction itself
        s2 = torch.amax(torch.abs(act), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0   # [E, C, 1]
        y = bmm8g((act / s2).to(F8), dw).reshape(E * C, H)                             # unscaled
        ps = (pm * s2).reshape(E * C, T)
        if y.device.type == 'cpu':
            return torch.matmul(bf(ps).T, bf(y))
        return torch.matmul(ps.T.to(torch.bfloat16), y.to(torch.bfloat16)).float()
    if q8 and dc is not None:
        s2 = torch.amax(torch.abs(act), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        y = bmm8g((act / s2).to(F8), dw) * s2                                          # [E, C, H]
    else:
        y = torch.matmul(act if EXACT else act.to(torch.bfloat16), dw.float() if EXACT else dw.to(torch.bfloat16)).float()
    return torch.matmul(pm.reshape(E * C, T).T, y.reshape(E * C, H))                   # [T, H]: <= K nonzero terms per element


def moe_hot(n, comb, t, g, hot, cold_out, comb2=None):
    """moe_grouped's hot tier: the gathered experts `hot` [S] on every token (moe_routed's gather form) -> cold_out + their sum.
    MOE_TSUB: the tokens in sub-blocks of that many rows (wide chunks), the gathered weights shared."""
    T, E, EIl, H, S = n.shape[0], g.E, g.EIl, g.H, hot.shape[0]
    gu, gus, dw, dc = t['egu'], t.get('egu_s'), t['ed'], t.get('ed_c')
    q8 = gus is not None and not EXACT
    if comb2 is None:
        comb2 = torch.index_select(comb, 1, hot)                                       # [T, S]
    tt = {'w': torch.index_select(gu, 0, hot).reshape(S * 2 * EIl, H)}
    if gus is not None:
        tt['w_s'] = torch.index_select(gus, 0, hot).reshape(-1)
    wd2 = torch.index_select(dw, 0, hot).reshape(S * EIl, H)
    dch = torch.index_select(dc, 0, hot).reshape(1, S, EIl) if dc is not None else None
    sub = MOE_TSUB if MOE_TSUB and T > MOE_TSUB and T % MOE_TSUB == 0 else T
    outs = []
    for b in range(0, T, sub):
        nb, cb = (n, comb2) if sub == T else (n[b:b + sub], comb2[b:b + sub])
        h2 = lin(nb, tt, 'w').reshape(sub, S, 2, EIl)
        act2 = bf(silu_mul(h2[:, :, 0], h2[:, :, 1])) * cb.unsqueeze(-1)
        if dch is not None:
            act2 = act2 * dch
        a2 = act2.reshape(sub, S * EIl)
        if q8 and dc is not None:
            s2 = torch.amax(torch.abs(a2), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
            outs.append(G.mm8((a2 / s2).to(F8), wd2).float() * s2)
        elif EXACT:
            outs.append(torch.matmul(a2, wd2.float()))
        else:
            outs.append(torch.matmul(a2.to(torch.bfloat16), wd2.to(torch.bfloat16)).float())
    return cold_out + (outs[0] if len(outs) == 1 else torch.cat(outs, dim=0))


def bmm8g(a8, w8, wt=False):
    """per-expert fp8 x fp8 contraction a8 [E, C, K] x w8 [E, K, N] (wt: w8 [E, N, K]) -> [E, C, N] f32, as E 2-D matmuls (the
    tensor-engine form of G.lin). Measured on trn1 / neuronx-cc 2.26: ONE batched matmul over the 205 experts is 3-4x slower
    and returns non-finite values; and the .float() must follow every dot directly - the fp8 x fp8 product accumulates in
    f32 only then (stacking the fp8-typed results first rounds them to E4M3: non-finite)."""
    f = (lambda x: x.float()) if a8.device.type == 'cpu' else (lambda x: x)
    return torch.stack([torch.matmul(f(a8[e]), f(w8[e].T if wt else w8[e])).float() for e in range(a8.shape[0])], dim=0)


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
        elif PROF_FIXED:   # timing probe: the first `slots` experts by a static slice (no dynamic gather); comb keeps its routing cost
            comb = comb[:, :slots] + 0.0 * comb.sum(dim=-1, keepdim=True)
        else:
            rank = used - ar * (0.5 / E)
            idx = torch.topk(rank, slots, dim=-1)[1].reshape(slots)                     # (2-D topk: a 1-D one crashed the XLA trace)
            comb = torch.index_select(comb, 1, idx)
        if PROF_FIXED and MOE_SLOTS != 'onehot':
            gu, dw = untile(gu[:slots]), dw[:slots]
            gus = gus[:slots] if gus is not None else None
            dc = dc[:slots] if dc is not None else None
        else:
            gu, dw = untile(torch.index_select(gu, 0, idx)), torch.index_select(dw, 0, idx)
            gus = torch.index_select(gus, 0, idx) if gus is not None else None
            dc = torch.index_select(dc, 0, idx) if dc is not None else None
    E_ = gu.shape[0]
    if MOE_DEBUG == 'comb':
        return None, comb
    if MOE_DEBUG == 'shared':
        return None, None
    if WT and gus is not None and not EXACT and (WT_MOE == 'stack' or (WT_MOE == '' and mode == 'all')):
        # one [T, H] x [H, 2 EIl] matmul per expert (no relayout of the expert tensor; the same products as lin)
        a, h8 = quant8(n)
        hgu = torch.stack([G.mm8(h8, gu[e]).float() for e in range(E_)], dim=1) * a.reshape(T, 1, 1) * gus.reshape(1, E_, 2 * EIl)
        hgu = G.bf(hgu).reshape(T, E_, 2, EIl)
    else:
        tt = {'w': gu.permute(1, 0, 2).reshape(g.H, E_ * 2 * EIl) if WT else gu.reshape(E_ * 2 * EIl, g.H)}   # WT: gu [E_, H, 2 EIl]
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
        return G.mm8((h / a).to(F8), wmat(t, 'w')).float() * a * t['w_s'].reshape(1, -1)
    if EXACT:
        return torch.matmul(h, wmat(t, 'w').float())
    return torch.matmul(h.to(torch.bfloat16), wmat(t, 'w')).float()


# ---------------------------------------------------------------------------------------------------------------- shapes
def weight_shapes(kind, g, fp8=True):
    """name -> (shape, dtype) of a sub-block's per-layer inputs for one rank (the prepare step writes exactly these)."""
    H, f32, b16 = g.H, torch.float32, torch.bfloat16
    wt = F8 if fp8 else b16
    s = {'norm': ((H,), f32)}

    def proj(name, out, inp):
        s[name] = ((inp, out) if WT else (out, inp), wt)
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
            s['rw'] = ((H, g.E) if WT else (g.E, H), f32)
        s['egu'] = ((g.E * 128, 2 * g.EIl * H // 128), wt) if EGU_T else ((g.E, H, 2 * g.EIl) if WT else (g.E, 2 * g.EIl, H), wt)
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
    if kind in ('gdn', 'gqa') or (kind == 'moe' and os.environ.get('MOE') == 'group'):   # the grouped MoE masks padding tokens (pos -1)
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
    allreduce = (lambda v: xm.all_reduce(xm.REDUCE_SUM, v, groups=groups)) if g.TP > 1 and not PROF_NOAR else (lambda v: v)
    if kind == 'head':
        T, S = int(sys.argv[3]), int(sys.argv[4])
        names = ['x', 'rows', 'norm', 'w'] + (['w_s'] if fp8 else [])
        shp = {'x': ((T, g.H), torch.float32), 'rows': ((S,), torch.int64), 'norm': ((g.H,), torch.float32),
               'w': (((g.H, g.VP // g.TP) if WT else (g.VP // g.TP, g.H)), F8 if fp8 else torch.bfloat16), 'w_s': ((g.VP // g.TP,), torch.float32)}
        inputs = [zeros(*shp[n]) for n in names]
        fn = lambda *a: head(dict(zip(names, a)), g, a[1])
        meta = {'kind': 'qwen35_head', 'T': T, 'S': S, 'inputs': names, 'states': [], 'fp8': fp8, 'wt': WT}
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
                if moe_mode == 'group' and kind == 'moe':   # output1: the cold tier's largest expert load (engine: > moe_cap -> rerun with MOE=all)
                    return x, st['ovf']
                return (x, *[st[n_] for n_ in xnames]) if xnames else x
        fn = Block()
        kwargs = {'enable_aliasing': True} if snames else {}
        meta = {'kind': 'qwen35_' + kind, 'seg': seg or [], 'R': R, 'Q': Q, 'P': P, 'pages': pages, 'slots': slots, 'inputs': names,
                'states': snames, 'fp8': fp8, 'moe': moe_mode, 'moe_slots': MOE_SLOTS, 'moe_bmm': MOE_BMM, 'router': ROUTER, 'lin_cse': LIN_CSE, 'gdn_dec': GDN_DEC, 'gdn_form': GDN_FORM, 'row_form': ROW_FORM, 'moe_split': MOE_SPLIT, 'moe_kernel': MOE_KERNEL, 'moe_kstat': MOE_KSTAT, 'moe_kskip': MOE_KSKIP, 'moe_kyd': MOE_KYD, 'moe_kdense': MOE_KDENSE, 'moe_kfast': MOE_KFAST, 'moe_kpre': MOE_KPRE, 'egu_t': EGU_T, 'route': ROUTE, 'ar_bf16': AR_BF16, 'gdn_chunk': GDN_CHUNK, 'gdn_inv': GDN_INV, 'gdn_loop': GDN_LOOP, 'rlogit': RLOGIT, 'att_bf16': ATT_BF16, 'kv8': KV8, 'kv_code': KV_CODE, 'att_block': ATT_BLOCK, 'att_gather': ATT_GATHER, 'gdn_gather': GDN_GATHER, 'gdn_einsum': GDN_EINSUM, 'gdn_slot': GDN_SLOT, 'extra_outputs': xnames,
                'moe_cap': MOE_CAP if moe_mode == 'group' and kind == 'moe' else 0, 'moe_hot': MOE_HOT, 'moe_gg': MOE_GG, 'moe_ge': MOE_GE, 'moe_stat': (MOE_GSTAT if MOE_GSTAT >= 0 else g.E) if moe_mode == 'group' and kind == 'moe' else 0, 'moe_gd': MOE_GD, 'moe_gf': MOE_GF, 'moe_gb': MOE_GB, 'moe_tiers': MOE_TIERS, 'moe_tail': MOE_TAIL, 'moe_gslots': MOE_GSLOTS, 'moe_gblk': MOE_GBLK, 'moe_tsub': MOE_TSUB, 'att_qblock': ATT_QBLOCK,
                'verify': bool(VERIFY and Q > 1), 'wt': WT, 'wt_moe': WT_MOE, 'gdn_rsplit': GDN_RSPLIT, 'prof_noar': PROF_NOAR, 'prof_fixed': PROF_FIXED}
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
