"""Server-contract Llama graphs for NeuronCore-v2 (trn1), tensor parallel, all projections cFP8.

One NEFF per (R, Q, P) bucket serves EVERY layer: the layers differ only in the weight and KV tensors bound
to it at run time (cpp/neuron_model.cpp), so there is no per-layer compilation.

  decode  : R requests x Q=1 token each (R padded with scratch requests)
  prefill : R=1 request x Q-token chunk (chunked prefill, the chunk attends to its own earlier chunks too)

Paged KV, exactly the server's layout (runtime/include/behemoth/attention.hpp): per layer and K/V one tensor
[pages, kv_heads_local, 16, D], page_tokens = 16, a request owns pages listed in its page table row. The
graph scatters the T new K/V rows to flat slots (page * 16 + offset), then attends over the request's pages in
key blocks of G pages with a running (max, sum, accumulator) -- flash-attention online softmax -- so no
[queries x context] score matrix is ever materialised. Key position j of the table is valid for a query at
absolute position p iff j <= p (causal and length mask in one compare), so the block loop needs no lengths.
The KV tensors are aliased input -> output (updated in place on the device).

Geometry (hidden, heads, kv heads, head dim, intermediate, vocab) comes from the checkpoint's config.json
(MODEL=/path/to/checkpoint or CONFIG=/path/config.json; default: Llama 3.1 8B) and TP from the TP env (default 2);
every meta.json records it and the runtime refuses a bucket directory traced for another geometry or TP.
Sharding: q/k/v/gate/up column-parallel (heads and kv heads divided over the ranks; with TP > kv_heads each kv head
is replicated on TP / kv_heads consecutive ranks, which hold its query heads),
o/down row-parallel with an all-reduce, lm_head by vocabulary rows padded to a multiple of TP.

Inputs (order == ORDER), per rank:
  x [T,H] f32 residual (replicated), cos_sin [T,2,D/2] f32 for the tokens' absolute positions (computed on the
  host: llama3 rope scaling, any position), pos [T] i64, wslot [T] i64, table [R,P] i64, cache [pages,2,kv_heads,16,D],
  norms and the seven FP8 projections with per-row scales.

Weight formats of a layer NEFF (meta.json "weights"):
  fp8  (default) the seven projections are row-scaled E4M3 shards, QTIP/NVFP4 tensors expanded at load (qtip_expand.cpp)
  qtip q/o/gate/up/down stay the checkpoint's 2-bit trellis words on the device and are decoded every step by the fused
       NKI kernel (qtip_nki.py); k/v are FP8 as in the checkpoints; the Hadamard rotations run in the graph (qtip_graph.py).
       Inputs per QTIP projection: words uint16 [G, 128, O*32], codebook uint32 [128, 1024], SU [1, in] and SV [1, out] f32,
       plus per-rank index vectors sel_h [H/TP] and sel_ff [FF/TP] selecting this rank's slice of a full vector.

usage: trace_model.py WORKDIR layer R Q P | head S | model R Q P L S [body]
       env: MODEL / CONFIG (geometry), TP, PAGES, WEIGHTS=fp8|qtip (layer), HEAD_ROWS=T (head), HEAD_FP8=1 (E4M3 lm_head with
            per-row scale), NCC_FLAGS
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

os.environ.setdefault('XLA_ALWAYS_ALLREDUCE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_layer import Fp8Weight, lin_fp8
import qtip_graph

LLAMA_8B = {'hidden': 4096, 'heads': 32, 'kv_heads': 8, 'head_dim': 128, 'intermediate': 14336, 'vocab': 128257, 'layers': 32}


class Geometry:
    """Model geometry plus the tensor-parallel degree; the module-level `g` is set by main() (or by a test) before tracing."""
    def __init__(self, d, tp):
        self.H, self.HEADS, self.KVHEADS, self.D = d['hidden'], d['heads'], d['kv_heads'], d['head_dim']
        self.FF, self.VOCAB, self.LAYERS, self.TP = d['intermediate'], d['vocab'], d['layers'], tp
        # kv heads are divided over the ranks, or (TP > kv_heads) each one replicated on TP / kv_heads consecutive ranks, whose
        # query heads must then all belong to it
        kv_split = self.KVHEADS % tp == 0 or (tp % self.KVHEADS == 0 and (self.HEADS // self.KVHEADS) % max(1, self.HEADS // tp) == 0)
        if self.HEADS % self.KVHEADS or self.HEADS % tp or not kv_split or self.FF % tp or self.D % 2:
            raise ValueError(f'geometry {d} is not shardable over TP={tp} (heads and intermediate must divide; kv heads must divide TP or be divided by it)')
        self.KVL = max(1, self.KVHEADS // tp)  # kv heads per rank
        self.VOCAB_PAD = (self.VOCAB + tp - 1) // tp * tp

    def meta(self):
        return {'hidden': self.H, 'heads': self.HEADS, 'kv_heads': self.KVHEADS, 'head_dim': self.D, 'intermediate': self.FF,
                'vocab': self.VOCAB, 'layers': self.LAYERS, 'tp': self.TP, 'page': PAGE}

    @staticmethod
    def from_config(path, tp):
        c = json.loads(Path(path).read_text())
        heads = int(c['num_attention_heads'])
        return Geometry({'hidden': int(c['hidden_size']), 'heads': heads, 'kv_heads': int(c.get('num_key_value_heads', heads)),
                         'head_dim': int(c.get('head_dim') or c['hidden_size'] // heads), 'intermediate': int(c['intermediate_size']),
                         'vocab': int(c['vocab_size']), 'layers': int(c['num_hidden_layers'])}, tp)


def geometry_from_env():
    tp = int(os.environ.get('TP', '2'))
    if os.environ.get('CONFIG'):
        return Geometry.from_config(os.environ['CONFIG'], tp)
    if os.environ.get('MODEL'):
        return Geometry.from_config(Path(os.environ['MODEL']) / 'config.json', tp)
    return Geometry(LLAMA_8B, tp)


g = Geometry(LLAMA_8B, 2)
PAGE = 16
KEY_BLOCK_PAGES = int(os.environ.get('KEY_BLOCK_PAGES', 128))  # G: pages per flash block for prefill (2048 keys; 32 was 6 % slower on a 14k prompt)
ATTN = os.environ.get('ATTN', 'nki')  # 'nki': prefill chunks (R = 1, Q a multiple of 128) run NKI's flash_fwd over the gathered pages; 'xla': torch-op flash loop
ATTN_DEC = os.environ.get('ATTN_DEC', 'xla')  # 'nki': decode (Q = 1, bf16 cache, up to ATTN_DEC_ROWS rows) runs nki_decode.decode_attention
ATTN_DEC_ROWS = int(os.environ.get('ATTN_DEC_ROWS', 8))
# ATTN_DEC_SHARED=1 (FP8 KV, ATTN_DEC=nki): every decode row attends over ROW 0's page table (the verify rows of ONE request): one gather
# per 128-page group for all rows (nki_decode.decode_attention_fp8_2p_shared); meta.json "shared_table": true, the runtime picks such a
# bucket only for single-request verify steps (docs/neuron-verify-kernel.md)
ATTN_DEC_SHARED = os.environ.get('ATTN_DEC_SHARED', '0') == '1'
ATTN_DEC_MIN_PAGES = int(os.environ.get('ATTN_DEC_MIN_PAGES', 128))  # below this the kernel (padded to 64 pages) is no faster than the torch-op loop
DECODE_BLOCK_PAGES = int(os.environ.get('DECODE_BLOCK_PAGES', 256))  # decode (Q = 1): few big blocks, the score row is tiny; 32-page blocks were ~50 us of serial small ops each
ORDER = ['x', 'cos_sin', 'pos', 'wslot', 'table', 'cache', 'norm1', 'norm2',
         'qf', 'qs', 'of', 'os', 'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']
# compressed QTIP layer: per QTIP projection words / codebook / SU / SV (q, o, gate, up, down), k/v FP8 as above
ORDER_QTIP = ['x', 'cos_sin', 'pos', 'wslot', 'table', 'cache', 'norm1', 'norm2', 'sel_h', 'sel_ff',
              'qw', 'ql', 'qsu', 'qsv', 'ow', 'ol', 'osu', 'osv', 'kf', 'ks', 'vf', 'vs',
              'gw', 'gl', 'gsu', 'gsv', 'uw', 'ul', 'usu', 'usv', 'dw', 'dl', 'dsu', 'dsv']
WEIGHTS = os.environ.get('WEIGHTS', 'fp8')
KV8 = os.environ.get('KV', 'bf16') == 'fp8'
KVPAIR = os.environ.get('KVPAIR', '0') == '1'  # paired K/V cache rows [pages, kvh, 2, 16, D]: one 4KB gather per (page, head), pass 2 reuses the tile
KVATT = os.environ.get('KVATT', 'deq')  # KV=fp8 attention: 'deq' (dequantise the gathered pages, bf16 loop; validated) or 'fp8' (fp8 x fp8 batched matmuls, GPU semantics: gives non-finite logits on trn1, see PERF_NOTES)  # KV cache as E4M3 bytes + one f32 scale per (K|V, token, kv head), like the GPU runtime's FP8 K/V
if KV8:
    ORDER.insert(ORDER.index('cache') + 1, 'cscale')
    ORDER_QTIP.insert(ORDER_QTIP.index('cache') + 1, 'cscale')
DOWN_BF16 = os.environ.get('DOWN_BF16', '0') == '1'  # diagnosis only: down_proj with bf16 activations (weights cast in-graph), no fp8 activation quantisation
OUT_F32 = os.environ.get('OUT_F32', '0') == '1'  # o_proj / down_proj partial sums stay f32 into the all-reduce (bf16 rounding of massive-activation channels)
WT = os.environ.get('WT') == '1'  # FP8 projections stored transposed [in, out]: contraction axis first, contiguous DMA rows
# MLP=nki (needs WT=1): gate / up / down in the tiled layout of nki_mlp.tile_weight, decode steps (T <= MLP_ROWS) through
# nki_mlp.mlp_fp8_tiled (weights as the moving operand, 4 KB DMA rows), wider steps (prefill) untile in the graph
MLP = os.environ.get('MLP', 'xla')
MLP_ROWS = int(os.environ.get('MLP_ROWS', 32))
HEAD_ORDER = ['x', 'rows', 'norm', 'w']
HEAD_FP8_ORDER = ['x', 'rows', 'norm', 'wf', 'ws']


def shapes(R, Q, P, pages):
    H, D, f8 = g.H, g.D, torch.float8_e4m3fn
    T, q, kv, ff = R * Q, g.HEADS * D // g.TP, g.KVL * D, g.FF // g.TP
    s = {
        'x': ((T, H), torch.float32), 'cos_sin': ((T, 2, D // 2), torch.float32), 'pos': ((T,), torch.int64),
        'wslot': ((T,), torch.int64), 'table': ((R, P), torch.int64),
        'cache': ((pages, g.KVL, 2, PAGE, D) if KVPAIR else (pages, 2, g.KVL, PAGE, D), torch.float8_e4m3fn if KV8 else torch.bfloat16),
        'cscale': ((pages, g.KVL, 2, PAGE) if KVPAIR else (pages, 2, g.KVL, PAGE), torch.float32),
        'norm1': ((1, H), torch.float32), 'norm2': ((1, H), torch.float32),
        'qf': ((q, H), f8), 'qs': ((q,), torch.float32), 'of': ((H, q), f8), 'os': ((H,), torch.float32),
        'kf': ((kv, H), f8), 'ks': ((kv,), torch.float32), 'vf': ((kv, H), f8), 'vs': ((kv,), torch.float32),
        'gf': ((ff, H), f8), 'gs': ((ff,), torch.float32), 'uf': ((ff, H), f8), 'us': ((ff,), torch.float32),
        'df': ((H, ff), f8), 'ds': ((H,), torch.float32),
        # compressed QTIP projections: (out_local, in_local) = q (H/TP, H), o (H, H/TP), gate/up (FF/TP, H), down (H, FF/TP)
        'sel_h': ((q,), torch.int64), 'sel_ff': ((ff,), torch.int64),
        'qw': (qtip_graph.words_shape(q, H), torch.uint16), 'ql': ((128, 1024), torch.uint32),
        'qsu': ((1, H), torch.float32), 'qsv': ((1, H), torch.float32),
        'ow': (qtip_graph.words_shape(H, q), torch.uint16), 'ol': ((128, 1024), torch.uint32),
        'osu': ((1, H), torch.float32), 'osv': ((1, H), torch.float32),
        'gw': (qtip_graph.words_shape(ff, H), torch.uint16), 'gl': ((128, 1024), torch.uint32),
        'gsu': ((1, H), torch.float32), 'gsv': ((1, g.FF), torch.float32),
        'uw': (qtip_graph.words_shape(ff, H), torch.uint16), 'ul': ((128, 1024), torch.uint32),
        'usu': ((1, H), torch.float32), 'usv': ((1, g.FF), torch.float32),
        'dw': (qtip_graph.words_shape(H, ff), torch.uint16), 'dl': ((128, 1024), torch.uint32),
        'dsu': ((1, g.FF), torch.float32), 'dsv': ((1, H), torch.float32),
    }
    if WT:
        for n in ('qf', 'of', 'kf', 'vf', 'gf', 'uf', 'df'):
            s[n] = (s[n][0][::-1], s[n][1])
    if MLP == 'nki':
        assert WT, 'MLP=nki needs WT=1'
        for n in ('gf', 'uf', 'df'):
            k, n_out = s[n][0]
            s[n] = ((k // 512 * (n_out // 1024) * 128, 4096), f8)
    return s


def untile(w, K, N):
    """nki_mlp tiled layout -> [K, N] (contraction axis first), in the graph."""
    return w.reshape(K // 512, N // 1024, 128, 4, 1024).permute(0, 3, 2, 1, 4).reshape(K, N)


def layer_order():
    return ORDER_QTIP if WEIGHTS == 'qtip' else ORDER


def flash_paged_nki(q, cache, cscale, table, pos, Q, P, kvh, rep):
    """The same attention for one request (R = 1) through neuronxcc's flash_fwd kernel (fused scale / mask / online softmax / PV).
    The kernel wants dense K^T [kvh, D, keys] and V [kvh, keys, D] and a [Q, keys] additive mask, so the pages are gathered and permuted
    here; key j of the table is valid for the query at position p iff j <= p, as in flash_paged."""
    from neuronxcc.nki.kernels.attention import FlashConfig, flash_fwd
    D, keys = g.D, P * PAGE
    kv = cache[table[0]]                                                       # [P, 2, kvh, 16, D]
    if cscale is not None:                                                     # FP8 cache: dequantise the gathered pages
        kv = (kv.float() * cscale[table[0]].unsqueeze(-1)).to(torch.bfloat16)
    k = kv[:, 0].permute(1, 3, 0, 2).reshape(1, kvh, D, keys)                  # K^T
    v = kv[:, 1].permute(1, 0, 2, 3).reshape(1, kvh, keys, D)
    qt = q.reshape(Q, kvh * rep, D).permute(1, 2, 0).reshape(1, kvh * rep, D, Q)  # heads are kv-head major
    kpos = torch.arange(keys, dtype=torch.int64, device=q.device)
    bias = torch.where(kpos[None, :] <= pos[:, None], 0., -30000.).to(torch.bfloat16).reshape(1, 1, Q, keys)
    config = FlashConfig(seq_tile_size=min(2048, keys), training=False)
    o = flash_fwd[1, kvh](qt, k, v, None, logit_bias=bias, use_causal_mask=False, mixed_precision=True, config=config)  # [1, heads, Q, D]
    return o.float().permute(0, 2, 1, 3).reshape(1, Q, kvh, rep, D)


def flash_decode_nki(q, cache, table, pos, R, P, kvh, rep):
    """R requests, one query token each, through nki_decode.decode_attention (see that module for the layouts). q [R, 1, kvh, rep, D] bf16."""
    import numpy as np
    import nki_decode as nd
    D, dev = g.D, q.device
    G = max(1, P // nd.GROUP_PAGES)
    pages = table
    if P < nd.GROUP_PAGES:                                                     # pad the tables to one group with the scratch page
        pages = torch.cat((pages, torch.full((R, nd.GROUP_PAGES - P), cache.shape[0] - 1, dtype=pages.dtype, device=dev)), dim=1)
    pg = pages.reshape(R, G, nd.GROUP_PAGES)
    half = torch.arange(2, dtype=torch.int64, device=dev).reshape(1, 1, 1, 1, 2)
    heads = torch.arange(kvh, dtype=torch.int64, device=dev).reshape(1, kvh, 1, 1, 1)
    def rows(kv):                                                              # [R, kvh, 128, G] cache row of partition 2 j + half
        base = (pg.reshape(R, 1, G, nd.GROUP_PAGES, 1) * 2 + kv) * kvh
        return ((base + heads) * 2 + half).reshape(R, kvh, G, 128).permute(0, 1, 3, 2).to(torch.int32)
    kpos = torch.tensor(nd.native_key_positions(G), dtype=torch.int64, device=dev)                          # [G, 1024]
    bias = torch.where(kpos.reshape(1, G, 1024) > pos.reshape(R, 1, 1), -30000.0, 0.0).to(torch.float32)
    bias = bias.reshape(R, G, 1, 1024).expand(R, G, nd.ROWS, 1024).contiguous()
    mask = torch.tensor(np.kron(np.eye(kvh, dtype=np.float32), np.ones((rep, 1), np.float32)), device=dev)   # [16, kvh]
    qt = (q.reshape(R, kvh, rep, D).float() * (D ** -0.5)).to(torch.bfloat16).permute(0, 1, 3, 2).contiguous()
    o = nd.decode_attention(qt, cache.reshape(-1, 1024), rows(0), rows(1), bias, mask)                       # [R * 16, D] f32
    return o.reshape(R, 1, kvh, rep, D)


def flash_decode_nki_fp8(q, cache, cscale, table, pos, R, P, kvh, rep):
    """flash_decode_nki for the FP8 cache through nki_decode.decode_attention_fp8_2p (page-head rows, 128-page groups; scales and mask in the
    kernel's native column order t * 128 + j, rows repeated over a head's query heads)."""
    import numpy as np
    import nki_decode as nd
    D, dev, GP = g.D, q.device, nd.GROUP8_PAGES
    G = -(-P // GP)
    pages = table
    if P < G * GP:                                                             # pad the tables to whole groups with the scratch page
        pages = torch.cat((pages, torch.full((R, G * GP - P), cache.shape[0] - 1, dtype=pages.dtype, device=dev)), dim=1)
    pg = pages.reshape(R, G, GP)
    heads = torch.arange(kvh, dtype=torch.int64, device=dev).reshape(1, kvh, 1, 1)
    def rows(kv):                                                              # [R, kvh, 128, G] cache row of page j of group g, head h
        return ((pg.reshape(R, 1, G, GP) * 2 + kv) * kvh + heads).permute(0, 1, 3, 2).to(torch.int32)
    kpos = torch.tensor(nd.native_key_positions8(G), dtype=torch.int64, device=dev)                       # [G, 2048]
    bias = torch.where(kpos.reshape(1, G, 2048) > pos.reshape(R, 1, 1), -30000.0, 0.0).to(torch.float32)
    bias = bias.reshape(R, G, 1, 2048).expand(R, G, nd.ROWS, 2048).contiguous()
    sc = cscale[pages].reshape(R, G, GP, 2, kvh, PAGE)                                                    # [R, G, j, K|V, h, t]
    def native(kv):                                                            # [R, G, 16 rows, 2048]: row 4 h + i, column t * 128 + j
        s = sc[:, :, :, kv].permute(0, 1, 3, 4, 2).reshape(R, G, kvh, 1, PAGE * GP)
        return s.expand(R, G, kvh, rep, PAGE * GP).reshape(R, G, kvh * rep, PAGE * GP).contiguous()
    mask = torch.tensor(np.kron(np.eye(kvh, dtype=np.float32), np.ones((rep, 1), np.float32)), device=dev)
    qt = (q.reshape(R, kvh, rep, D).float() * (D ** -0.5)).to(torch.bfloat16).permute(0, 1, 3, 2).contiguous()
    o = nd.decode_attention_fp8_2p(qt, cache.reshape(-1, PAGE * D), rows(0), rows(1), bias, mask, native(0), native(1))
    return o.reshape(R, 1, kvh, rep, D)


def flash_decode_nki_fp8_shared(q, cache, cscale, table, pos, R, P, kvh, rep):
    """flash_decode_nki_fp8 for R rows that share ONE page table (table[0]; the other table rows are ignored) through
    nki_decode.decode_attention_fp8_2p_shared: every 128-page group is gathered once for all R x 16 query rows (<= 128 partitions);
    the causal / length mask is per row (its own position), the K / V scales are repeated over the rows. Any P (padded to whole groups)."""
    import numpy as np
    import nki_decode as nd
    D, dev, GP = g.D, q.device, nd.GROUP8_PAGES
    G = -(-P // GP)
    NP = R * nd.ROWS
    assert NP <= 128, f'ATTN_DEC_SHARED: {R} rows x {nd.ROWS} query rows exceed 128 partitions'
    pages = table[:1]                                                          # [1, P]: the shared table
    if P < G * GP:
        pages = torch.cat((pages, torch.full((1, G * GP - P), cache.shape[0] - 1, dtype=pages.dtype, device=dev)), dim=1)
    pg = pages.reshape(G, GP)
    heads = torch.arange(kvh, dtype=torch.int64, device=dev).reshape(kvh, 1, 1)
    def rows(kv):                                                              # [kvh, 128, G] cache row of page j of group g, head h
        return ((pg.reshape(1, G, GP) * 2 + kv) * kvh + heads).permute(0, 2, 1).to(torch.int32)
    kpos = torch.tensor(nd.native_key_positions8(G), dtype=torch.int64, device=dev)                       # [G, 2048]
    bias = torch.where(kpos.reshape(G, 1, 2048) > pos.reshape(1, R, 1), -30000.0, 0.0).to(torch.float32)   # [G, R, 2048]
    bias = bias.reshape(G, R, 1, 2048).expand(G, R, nd.ROWS, 2048).reshape(G, NP, 2048).contiguous()
    sc = cscale[pages[0]].reshape(G, GP, 2, kvh, PAGE)                                                    # [G, j, K|V, h, t]
    def native(kv):                                                            # [G, NP, 2048]: row 16 r + 4 h + i, column t * 128 + j
        s = sc[:, :, kv].permute(0, 2, 3, 1).reshape(G, kvh, 1, PAGE * GP).expand(G, kvh, rep, PAGE * GP)
        return s.reshape(G, 1, nd.ROWS, PAGE * GP).expand(G, R, nd.ROWS, PAGE * GP).reshape(G, NP, PAGE * GP).contiguous()
    mask = torch.tensor(np.tile(np.kron(np.eye(kvh, dtype=np.float32), np.ones((rep, 1), np.float32)), (R, 1)), device=dev)   # [NP, kvh]
    qt = (q.reshape(R, kvh, rep, D).float() * (D ** -0.5)).to(torch.bfloat16).permute(0, 1, 3, 2).contiguous()
    o = nd.decode_attention_fp8_2p_shared(qt, cache.reshape(-1, PAGE * D), rows(0), rows(1), bias, mask, native(0), native(1))
    return o.reshape(R, 1, kvh, rep, D)


def flash_decode_nki_fp8_paired(q, cache, cscale, table, pos, R, P, kvh, rep):
    """flash_decode_nki_fp8 over the PAIRED cache [pages, kvh, 2, PAGE, D]: row r = p * kvh + h holds K|V in one
    4 KB row, so pass 1 gathers K and V together and pass 2 reuses the resident tile (nki_decode.decode_attention_fp8_2p_paired)."""
    import numpy as np
    import nki_decode as nd
    D, dev, GP = g.D, q.device, nd.GROUPP_PAGES
    G = -(-P // GP)
    pages = table
    if P < G * GP:
        pages = torch.cat((pages, torch.full((R, G * GP - P), cache.shape[0] - 1, dtype=pages.dtype, device=dev)), dim=1)
    pg = pages.reshape(R, G, GP)
    heads = torch.arange(kvh, dtype=torch.int64, device=dev).reshape(1, kvh, 1, 1)
    rows = ((pg.reshape(R, 1, G, GP) * kvh + heads)).permute(0, 1, 3, 2).to(torch.int32)   # [R, kvh, 128, G]
    kpos = torch.tensor(nd.native_key_positionsp(G), dtype=torch.int64, device=dev)
    bias = torch.where(kpos.reshape(1, G, 2048) > pos.reshape(R, 1, 1), -30000.0, 0.0).to(torch.float32)
    bias = bias.reshape(R, G, 1, 2048).expand(R, G, nd.ROWS, 2048).contiguous()
    sc = cscale[pages].reshape(R, G, GP, kvh, 2, PAGE)                                                    # [R, G, j, h, K|V, t]
    def native(kv):
        s = sc[:, :, :, :, kv].permute(0, 1, 3, 4, 2).reshape(R, G, kvh, 1, PAGE * GP)
        return s.expand(R, G, kvh, rep, PAGE * GP).reshape(R, G, kvh * rep, PAGE * GP).contiguous()
    mask = torch.tensor(np.kron(np.eye(kvh, dtype=np.float32), np.ones((rep, 1), np.float32)), device=dev)
    qt = (q.reshape(R, kvh, rep, D).float() * (D ** -0.5)).to(torch.bfloat16).permute(0, 1, 3, 2).contiguous()
    o = nd.decode_attention_fp8_2p_paired(qt, cache.reshape(-1, PAGE * D * 2), rows, rows, bias, mask, native(0), native(1))
    return o.reshape(R, 1, kvh, rep, D)


def flash_paged_fp8(q, cache, cscale, table, pos, R, Q, P, kvh, rep):
    """Flash loop over an FP8 cache with the GPU runtime's attention semantics (runtime/src/attention.cpp): q is quantised per (query, head) row,
    scores = (q8 . k8) q_scale k_scale / sqrt(D), probabilities are weighted by the V scale of their token, quantised per row and block to E4M3
    (the softmax sum uses the unweighted p) and multiplied with the E4M3 V pages; both matmuls run fp8 x fp8 (torch.matmul, as the projections do:
    einsum on fp8 operands keeps an fp8 result). Layout [R, kvh, Q, rep, keys]."""
    G = min(KEY_BLOCK_PAGES if Q > 1 else DECODE_BLOCK_PAGES, P)
    assert P % G == 0
    D = g.D
    scale = D ** -0.5
    neg, f8, dev = -1e30, torch.float8_e4m3fn, q.device
    qf = q.float().permute(0, 2, 1, 3, 4)                                                # [R,kvh,Q,rep,D]
    qa = torch.amax(torch.abs(qf), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0        # [R,kvh,Q,rep,1]
    q8 = (qf / qa).to(f8).reshape(R, kvh, Q * rep, D)
    m = torch.full((R, kvh, Q, rep), neg, device=dev)
    l = torch.zeros(R, kvh, Q, rep, device=dev)
    acc = torch.zeros(R, kvh, Q, rep, D, device=dev)
    qpos = pos.reshape(R, 1, Q, 1, 1)
    inblock = torch.arange(G * PAGE, dtype=torch.int64, device=dev).reshape(1, 1, 1, 1, G * PAGE)
    for b in range(P // G):
        pages = table[:, b * G:(b + 1) * G]                                              # [R,G]
        kv = cache[pages]                                                                # [R,G,2,kvh,16,D] E4M3
        sc = cscale[pages]                                                               # [R,G,2,kvh,16]
        k8 = kv[:, :, 0].permute(0, 2, 1, 3, 4).reshape(R, kvh, G * PAGE, D)
        v8 = kv[:, :, 1].permute(0, 2, 1, 3, 4).reshape(R, kvh, G * PAGE, D)
        ksc = sc[:, :, 0].permute(0, 2, 1, 3).reshape(R, kvh, 1, 1, G * PAGE)
        vsc = sc[:, :, 1].permute(0, 2, 1, 3).reshape(R, kvh, 1, 1, G * PAGE)
        s = torch.matmul(q8, k8.transpose(-1, -2)).float().reshape(R, kvh, Q, rep, G * PAGE) * qa * ksc * scale
        s = torch.where(inblock + b * G * PAGE > qpos, torch.full_like(s, neg), s)
        m_new = torch.maximum(m, s.amax(dim=-1))
        p = torch.exp(s - m_new.unsqueeze(-1))
        corr = torch.exp(m - m_new)
        l = l * corr + p.sum(dim=-1)
        pw = p * vsc
        ps = (torch.amax(pw, dim=-1, keepdim=True) / 224.0).clamp(min=1e-30)             # [R,kvh,Q,rep,1]
        p8 = (pw / ps).to(f8).reshape(R, kvh, Q * rep, G * PAGE)
        pv = torch.matmul(p8, v8).float().reshape(R, kvh, Q, rep, D) * ps
        acc = acc * corr.unsqueeze(-1) + pv
        m = m_new
    return (acc / l.unsqueeze(-1)).permute(0, 2, 1, 3, 4)


def flash_paged(q, cache, cscale, table, pos, R, Q, P, kvh, rep):
    """q [R,Q,kvh,rep,D] bf16 (rope applied) -> [R,Q,kvh,rep,D] f32. Online softmax over blocks of G pages."""
    if ATTN == 'nki' and R == 1 and Q % 128 == 0:
        return flash_paged_nki(q, cache, cscale, table, pos, Q, P, kvh, rep)
    if ATTN_DEC == 'nki' and ATTN_DEC_SHARED and Q == 1 and cscale is not None and R <= ATTN_DEC_ROWS:
        return flash_decode_nki_fp8_shared(q, cache, cscale, table, pos, R, P, kvh, rep)
    if ATTN_DEC == 'nki' and Q == 1 and cscale is None and R <= ATTN_DEC_ROWS and P >= ATTN_DEC_MIN_PAGES:
        return flash_decode_nki(q, cache, table, pos, R, P, kvh, rep)
    if ATTN_DEC == 'nki' and Q == 1 and cscale is not None and R <= ATTN_DEC_ROWS and P >= ATTN_DEC_MIN_PAGES:
        return flash_decode_nki_fp8(q, cache, cscale, table, pos, R, P, kvh, rep)
    if ATTN_DEC == 'nkikv' and Q == 1 and cscale is not None and R <= ATTN_DEC_ROWS and P >= ATTN_DEC_MIN_PAGES:
        return flash_decode_nki_fp8_paired(q, cache, cscale, table, pos, R, P, kvh, rep)
    if cscale is not None and KVATT == 'fp8':
        return flash_paged_fp8(q, cache, cscale, table, pos, R, Q, P, kvh, rep)
    G = min(KEY_BLOCK_PAGES if Q > 1 else DECODE_BLOCK_PAGES, P)
    assert P % G == 0
    D = g.D
    scale = D ** -0.5
    neg = -1e30
    dev = q.device
    m = torch.full((R, Q, kvh, rep), neg, device=dev)
    l = torch.zeros(R, Q, kvh, rep, device=dev)
    acc = torch.zeros(R, Q, kvh, rep, D, device=dev)
    qpos = pos.reshape(R, Q, 1, 1, 1)
    inblock = torch.arange(G * PAGE, dtype=torch.int64, device=dev).reshape(1, 1, 1, 1, G * PAGE)
    for b in range(P // G):
        pages = table[:, b * G:(b + 1) * G]                      # [R,G]
        kv = cache[pages]                                        # [R,G,2,kvh,16,D]: K and V pages in one gather
        if cscale is not None:                                   # FP8 cache read by the bf16 loop: dequantise
            kv = (kv.float() * cscale[pages].unsqueeze(-1)).to(torch.bfloat16)
        kb, vb = kv[:, :, 0], kv[:, :, 1]                        # [R,G,kvh,16,D]
        s = torch.einsum('rqhad,rghtd->rqhagt', q, kb).float().reshape(R, Q, kvh, rep, G * PAGE) * scale
        s = torch.where(inblock + b * G * PAGE > qpos, torch.full_like(s, neg), s)
        m_new = torch.maximum(m, s.amax(dim=-1))
        p = torch.exp(s - m_new.unsqueeze(-1))
        corr = torch.exp(m - m_new)
        l = l * corr + p.sum(dim=-1)
        pv = torch.einsum('rqhagt,rghtd->rqhad', p.reshape(R, Q, kvh, rep, G, PAGE).to(torch.bfloat16), vb).float()
        acc = acc * corr.unsqueeze(-1) + pv
        m = m_new
    return acc / l.unsqueeze(-1)


def paged_layer(R, Q, P, *args):
    import torch_xla.core.xla_model as xm
    order = layer_order()
    t = dict(zip(order, args))
    groups = [list(range(g.TP))]
    allreduce = lambda v: xm.all_reduce(xm.REDUCE_SUM, v, groups=groups)
    D, heads, kvh = g.D, g.HEADS // g.TP, g.KVL
    rep, T, half = heads // kvh, R * Q, D // 2
    H, FF = g.H, g.FF
    w = lambda a, b: Fp8Weight(t[a], t[b], WT)
    qtip = WEIGHTS == 'qtip'
    x = t['x']
    h = (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * t['norm1']).to(torch.bfloat16)
    if qtip:
        # q column-parallel: rotate the replicated input, multiply this rank's rows of W~, all-gather z, finish on the
        # full vector and keep this rank's heads
        z = qtip_graph.gather_tp(qtip_graph.kernel_matmul(qtip_graph.rotate_in(h, t['qsu'], H), t['qw'], t['ql']), t['sel_h'], H, allreduce, g.TP)
        q = torch.index_select(qtip_graph.finish(z, t['qsv'], H), -1, t['sel_h']).reshape(T, heads, D)
    else:
        q = lin_fp8(h, w('qf', 'qs')).float().reshape(T, heads, D)
    k = lin_fp8(h, w('kf', 'ks')).float().reshape(T, kvh, D)
    v = lin_fp8(h, w('vf', 'vs')).float().reshape(T, kvh, D)
    cos, sin = t['cos_sin'][:, 0].reshape(T, 1, half), t['cos_sin'][:, 1].reshape(T, 1, half)

    def rope(z):
        a, b = z[..., :half], z[..., half:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    q, k = rope(q), rope(k)
    page = torch.bitwise_right_shift(t['wslot'], 4)
    off = torch.bitwise_and(t['wslot'], PAGE - 1)
    at = (page[:, None, None], torch.arange(2, dtype=torch.int64, device=x.device)[None, :, None],
          torch.arange(kvh, dtype=torch.int64, device=x.device)[None, None, :], off[:, None, None])
    cscale = None
    if KV8:
        kk = torch.stack((k, v), dim=1)                                                  # [T,2,kvh,D] f32
        sc = torch.amax(torch.abs(kk), dim=-1).clamp(min=1e-20) / 224.0                  # one scale per (token, K|V, kv head)
        k8 = (kk / sc.unsqueeze(-1)).to(torch.float8_e4m3fn)
        if KVPAIR:
            atp = (page[:, None, None], torch.arange(kvh, dtype=torch.int64, device=x.device)[None, :, None],
                   torch.arange(2, dtype=torch.int64, device=x.device)[None, None, :], off[:, None, None])
            cache = t['cache'].index_put(atp, k8.permute(0, 2, 1, 3))
            cscale = t['cscale'].index_put(atp, sc.permute(0, 2, 1))
        else:
            cache = t['cache'].index_put(at, k8)
            cscale = t['cscale'].index_put(at, sc)
    else:
        cache = t['cache'].index_put(at, torch.stack((k, v), dim=1).to(torch.bfloat16))  # [T,2,kvh,D] -> cache[page, kv, head, off]
    attn = flash_paged(q.to(torch.bfloat16).reshape(R, Q, kvh, rep, D), cache, cscale, t['table'], t['pos'], R, Q, P, kvh, rep)
    attn = attn.reshape(T, heads * D).to(torch.bfloat16)
    # ATTN_PROFILE='attn': stop after the attention kernel (the attention half); 'noattn': skip the attention
    # kernel (attn = 0) and run the MLP tail. Unused weights stay in the graph through a dead scalar so the
    # input ORDER (and thus layer_bench binding) is unchanged.
    prof = os.environ.get('ATTN_PROFILE', '')
    if prof:
        used = {'attn': ('x', 'cos_sin', 'pos', 'wslot', 'table', 'cache', 'cscale', 'norm1', 'qf', 'qs', 'kf', 'ks', 'vf', 'vs'),
                'noattn': ('x', 'cos_sin', 'pos', 'wslot', 'table', 'cache', 'cscale', 'norm1', 'qf', 'qs', 'kf', 'ks', 'vf', 'vs',
                           'norm2', 'of', 'os', 'gf', 'gs', 'uf', 'us', 'df', 'ds')}[prof]
        dead = sum(t[k].float().reshape(-1)[0] for k in ORDER if k not in used)
        if prof == 'attn':
            return (attn.float() + 0.0 * dead, cache, cscale) if KV8 else (attn.float() + 0.0 * dead, cache)
        attn = q.reshape(T, heads * D).to(torch.bfloat16)
    if qtip:
        # o row-parallel: the rotation needs all heads (all-gather, rank order = head order), this rank multiplies its
        # columns of W~ by its slice of u, the partial z is all-reduced and finished on both ranks
        attn_full = qtip_graph.gather_tp(attn.float(), t['sel_h'], H, allreduce, g.TP)
        u = torch.index_select(qtip_graph.rotate_in(attn_full, t['osu'], H), -1, t['sel_h'])
        x = x + qtip_graph.finish(allreduce(qtip_graph.kernel_matmul(u, t['ow'], t['ol'])), t['osv'], H)
    else:
        x = x + allreduce(lin_fp8(attn, w('of', 'os'), OUT_F32).float())
    h = (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * t['norm2']).to(torch.bfloat16)
    if qtip:
        # gate | up column-parallel with one all-gather; their full outputs are replicated, so down (row-parallel)
        # needs no all-gather of its input
        zg = qtip_graph.kernel_matmul(qtip_graph.rotate_in(h, t['gsu'], H), t['gw'], t['gl'])
        zu = qtip_graph.kernel_matmul(qtip_graph.rotate_in(h, t['usu'], H), t['uw'], t['ul'])
        # gather_tp lays the ranks out one after the other, so gather [gate | up] of each rank as one block and split after
        ff_local = FF // g.TP
        sel2 = 2 * t['sel_ff'][:1] + torch.arange(2 * ff_local, dtype=torch.int64, device=x.device)  # rank * 2 * ff_local + j
        z = qtip_graph.gather_tp(torch.cat((zg, zu), dim=-1), sel2, 2 * FF, allreduce, g.TP).reshape(T, g.TP, 2, ff_local)
        gate = qtip_graph.finish(z[:, :, 0].reshape(T, FF), t['gsv'], FF)
        up = qtip_graph.finish(z[:, :, 1].reshape(T, FF), t['usv'], FF)
        act = torch.nn.functional.silu(gate) * up
        u = torch.index_select(qtip_graph.rotate_in(act, t['dsu'], FF), -1, t['sel_ff'])
        x = x + qtip_graph.finish(allreduce(qtip_graph.kernel_matmul(u, t['dw'], t['dl'])), t['dsv'], H)
    else:
        ff_local = FF // g.TP
        if MLP == 'nki' and T <= MLP_ROWS:
            import nki_mlp
            ha = torch.amax(torch.abs(h.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
            h8T = (h.float() / ha).to(torch.float8_e4m3fn).T.contiguous()
            y = nki_mlp.mlp_fp8_tiled(h8T, ha, t['gf'], t['gs'].reshape(1, -1), t['uf'], t['us'].reshape(1, -1), t['df'], t['ds'].reshape(1, -1))
            return (x + allreduce(y), cache, cscale) if KV8 else (x + allreduce(y), cache)
        if MLP == 'nki':
            t = dict(t, gf=untile(t['gf'], H, ff_local), uf=untile(t['uf'], H, ff_local), df=untile(t['df'], ff_local, H))
        act = torch.nn.functional.silu(lin_fp8(h, w('gf', 'gs'))).float() * lin_fp8(h, w('uf', 'us')).float()
        if DOWN_BF16:
            wd = t['df'] if WT else t['df'].T
            x = x + allreduce((torch.matmul(act.to(torch.bfloat16), wd.to(torch.bfloat16)).float() * t['ds'].reshape(1, -1)))
        else:
            x = x + allreduce(lin_fp8(act.to(torch.bfloat16), w('df', 'ds'), OUT_F32).float())
    return (x, cache, cscale) if KV8 else (x, cache)


LAYER_WEIGHTS = ['norm1', 'norm2', 'qf', 'qs', 'of', 'os', 'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']
SHARED = ['x', 'cos_sin', 'pos', 'wslot', 'table', 'rows', 'head_norm', 'head_w']
BODY_SHARED = SHARED[:5]  # a body segment (no head) takes only these and returns the residual stream


def head_graph(S, x, rows, norm, w, ws=None, transposed=False, argmax=False):
    """Final RMSNorm and this rank's vocab slice of lm_head for S selected rows -> logits [S, VOCAB_PAD/TP] f32.
    bf16 lm_head (w [V/TP, H]) or, with ws, an E4M3 lm_head with a per-row scale: the activation is scaled per row into
    E4M3 like the layer projections, the matmul runs fp8 x fp8 and the f32 result is rescaled (not rounded to bf16).
    transposed: w is [H, V/TP] (contraction axis first, like WT layer weights).
    argmax: return (per-row max, per-row argmax int32) over the vocab shard instead of the logits (greedy fast path)."""
    x = torch.index_select(x, 0, rows)
    h = (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * norm).to(torch.bfloat16)
    if ws is None:
        logits = (h @ (w if transposed else w.T)).float()
    else:
        a = torch.amax(torch.abs(h.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        h8 = (h.float() / a).to(torch.float8_e4m3fn)
        logits = torch.matmul(h8, w if transposed else w.T).float() * a * ws.reshape(1, -1)
    if argmax:
        mx, am = torch.max(logits, dim=-1)
        return mx, am.to(torch.int32)
    return logits


class FullModel(torch.nn.Module):
    """All L layers plus the final norm / lm_head slice as ONE graph. Inputs: SHARED, then LAYER_WEIGHTS per layer;
    the L KV tensors (K and V of a layer in one [pages, 2, kv_heads, 16, D] tensor) are module state aliased input -> output.
    Keeping the aliased tensors at one per layer matters: a graph with 64 aliased state tensors (32 layers x separate K, V) compiled
    to wrong results, 62 did not."""
    def __init__(self, R, Q, P, L, S, pages, head=True):
        super().__init__()
        self.R, self.Q, self.P, self.L, self.S, self.head = R, Q, P, L, S, head
        shape = (pages, 2, g.KVL, PAGE, g.D)
        for i in range(L):
            self.register_buffer(f'kv{i}', torch.zeros(shape, dtype=torch.bfloat16))

    def forward(self, *args):
        names = SHARED if self.head else BODY_SHARED
        shared = dict(zip(names, args[:len(names)]))
        per_layer = args[len(names):]
        x = shared['x']
        for i in range(self.L):
            t = dict(zip(LAYER_WEIGHTS, per_layer[i * len(LAYER_WEIGHTS):(i + 1) * len(LAYER_WEIGHTS)]))
            t.update(x=x, cos_sin=shared['cos_sin'], pos=shared['pos'], wslot=shared['wslot'], table=shared['table'],
                     cache=getattr(self, f'kv{i}'))
            x, cache = paged_layer(self.R, self.Q, self.P, *[t[k] for k in ORDER])
            setattr(self, f'kv{i}', cache)
        if not self.head:
            return x
        return head_graph(self.S, x, shared['rows'], shared['head_norm'], shared['head_w'])


def full_model_inputs(R, Q, P, L, S, head=True):
    s = shapes(R, Q, P, 1)
    T = R * Q
    out = [torch.zeros(T, g.H), torch.zeros(T, 2, g.D // 2), torch.zeros(T, dtype=torch.int64), torch.zeros(T, dtype=torch.int64),
           torch.zeros(R, P, dtype=torch.int64)]
    if head:
        out += [torch.zeros(S, dtype=torch.int64), torch.ones(1, g.H), torch.zeros(g.VOCAB_PAD // g.TP, g.H, dtype=torch.bfloat16)]
    for _ in range(L):
        for k in LAYER_WEIGHTS:
            shape, dtype = s[k]
            out.append(torch.zeros(shape).to(dtype) if dtype == torch.float8_e4m3fn else torch.zeros(shape, dtype=dtype))
    return out


def example_inputs(R, Q, P, pages):
    s = shapes(R, Q, P, pages)
    out = []
    for k in layer_order():
        shape, dtype = s[k]
        out.append(torch.zeros(shape).to(dtype) if dtype in (torch.float8_e4m3fn, torch.bfloat16) else torch.zeros(shape, dtype=dtype))
    return out


# Trace-time env flags recorded in meta.json "build" (provenance only; the runtime reads none of them)
BUILD_ENV_FLAGS = ['MODEL', 'CONFIG', 'TP', 'PAGES', 'WEIGHTS', 'KV', 'KVATT', 'ATTN_DEC', 'ATTN_DEC_SHARED', 'ATTN_DEC_MIN_PAGES', 'ATTN_PROFILE',
                   'WT', 'MLP', 'DOWN_BF16', 'OUT_F32', 'HEAD_ROWS', 'HEAD_FP8', 'HEAD_ARGMAX', 'UNSAFE_FP8FNCAST', 'NCC_FLAGS', 'NEURON_PLATFORM_TARGET_OVERRIDE']


def build_info():
    """Tool versions, repo commit, trace env and a timestamp for meta.json "build" (verify_graph_set.py reads it; the runtime only
    logs it with BEHEMOTH_NEURON_LOG_VERSIONS=1). Every field is best effort: a missing tool leaves the key out, never fails the trace."""
    import datetime
    info = {'timestamp': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'python': sys.version.split()[0]}
    try:
        from importlib import metadata
        for key, dist in [('neuronx_cc', 'neuronx-cc'), ('torch_neuronx', 'torch-neuronx'), ('torch_xla', 'torch-xla'), ('torch', 'torch'),
                          ('nki', 'nki'), ('libneuronxla', 'libneuronxla')]:
            try:
                info[key] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                pass
    except Exception:  # noqa: BLE001  (provenance only)
        pass
    def run(*cmd, stderr=False):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout + (r.stderr if stderr else '')).strip() if r.returncode == 0 else ''
    try:  # the compiler binary that `neuronx-cc compile` below resolves to (the venv pip version above may differ from PATH);
        # the banner "NeuronX Compiler version X" may go to stderr
        line = next((l for l in run('neuronx-cc', '--version', stderr=True).splitlines() if 'version' in l), '')
        info['neuronx_cc_bin'] = line.split('version')[-1].strip()
    except Exception:  # noqa: BLE001
        pass
    try:  # the runtime library installed where the buckets are compiled (the serving image may install another: the runtime logs both)
        info['libnrt'] = run('dpkg-query', '-W', '-f=${Version}', 'aws-neuronx-runtime-lib')
    except Exception:  # noqa: BLE001
        pass
    try:
        here = Path(__file__).resolve().parent
        info['git_commit'] = run('git', '-C', str(here), 'rev-parse', 'HEAD')
        if info['git_commit']:  # a plain copy of the tree (no .git) records neither
            info['git_dirty'] = bool(run('git', '-C', str(here), 'status', '--porcelain', '--', str(here)))
    except Exception:  # noqa: BLE001
        pass
    info['env'] = {k: os.environ[k] for k in BUILD_ENV_FLAGS if k in os.environ}
    return {k: v for k, v in info.items() if v not in ('', None)}


def main():
    global g
    workdir, kind = sys.argv[1], sys.argv[2]
    g = geometry_from_env()
    print('geometry', json.dumps(g.meta()))
    import torch_neuronx
    if kind == 'layer':
        R, Q, P = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        pages = int(os.environ.get('PAGES', '64'))
        inputs = example_inputs(R, Q, P, pages)
        kwargs = {}
        if len(sys.argv) < 7 or sys.argv[6] != '0':
            # state tensors of a Module are the only things torch_neuronx can alias input -> output
            class Layer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer('cache', inputs[layer_order().index('cache')])
                    if KV8:
                        self.register_buffer('cscale', inputs[layer_order().index('cscale')])

                def forward(self, *rest):
                    a = list(rest)
                    a.insert(layer_order().index('cache'), self.cache)
                    if KV8:
                        a.insert(layer_order().index('cscale'), self.cscale)
                        x, self.cache, self.cscale = paged_layer(R, Q, P, *a)
                    else:
                        x, self.cache = paged_layer(R, Q, P, *a)  # mutated buffer: enable_aliasing makes it in-place
                    return x
            fn = Layer()
            kwargs['enable_aliasing'] = True
            inputs = [t for k, t in zip(layer_order(), inputs) if k not in ('cache', 'cscale')]
        else:
            fn = lambda *a: paged_layer(R, Q, P, *a)
        if WEIGHTS not in ('fp8', 'qtip'):
            raise SystemExit(f'WEIGHTS={WEIGHTS}: expected fp8 or qtip')
        meta = {'kind': 'layer', 'R': R, 'Q': Q, 'P': P, 'pages': pages, 'order': layer_order(), 'weights': WEIGHTS, 'wt': WT, 'kv': 'fp8' if KV8 else 'bf16', 'mlp': 'tiled' if MLP == 'nki' else 'plain'}
        if ATTN_DEC == 'nki' and ATTN_DEC_SHARED and KV8 and Q == 1:
            meta['shared_table'] = True  # every row attends over table[0]: the runtime uses this bucket for single-request verify steps only
    elif kind == 'model':
        if WEIGHTS != 'fp8' or KV8:
            raise SystemExit('whole-model graphs are only generated for WEIGHTS=fp8 with a bf16 KV cache')
        R, Q, P, L, S = (int(a) for a in sys.argv[3:8])
        head = len(sys.argv) < 9 or sys.argv[8] != 'body'  # 'body': a segment without the head (residual out)
        pages = int(os.environ.get('PAGES', '64'))
        fn = FullModel(R, Q, P, L, S, pages, head)
        inputs = full_model_inputs(R, Q, P, L, S, head)
        kwargs = {'enable_aliasing': True}
        meta = {'kind': 'model', 'R': R, 'Q': Q, 'P': P, 'L': L, 'S': S, 'T': R * Q, 'pages': pages, 'head': head}
    else:
        S = int(sys.argv[3])
        T = int(os.environ.get('HEAD_ROWS', '256'))
        fp8 = os.environ.get('HEAD_FP8', '0') == '1'
        argmax = os.environ.get('HEAD_ARGMAX', '0') == '1'
        rows = g.VOCAB_PAD // g.TP
        inputs = [torch.zeros(T, g.H), torch.zeros(S, dtype=torch.int64), torch.ones(1, g.H)]
        kwargs = {}
        if fp8:
            inputs += [torch.zeros(*((g.H, rows) if WT else (rows, g.H))).to(torch.float8_e4m3fn), torch.ones(rows)]
            fn = lambda x, rows_, norm, wf, ws: head_graph(S, x, rows_, norm, wf, ws, WT, argmax)
        else:
            inputs += [torch.zeros(*((g.H, rows) if WT else (rows, g.H)), dtype=torch.bfloat16)]
            fn = lambda x, rows_, norm, w: head_graph(S, x, rows_, norm, w, None, WT, argmax)
        meta = {'kind': 'head', 'S': S, 'T': T, 'fp8': fp8, 'argmax': argmax, 'wt': WT, 'order': HEAD_FP8_ORDER if fp8 else HEAD_ORDER}
    meta['geometry'] = g.meta()
    meta['build'] = build_info()
    try:
        torch_neuronx.trace(fn, tuple(inputs), compiler_workdir=workdir, **kwargs)
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
        print('trace-time compile error (expected FN fp8):', str(error)[-700:])
    m = Path(workdir) / 'model'
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(m / 'graph.hlo'), str(m / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'] +
                   os.environ.get('NCC_FLAGS', '').split(), cwd=workdir, check=True)
    meta['neff'] = str(Path(workdir) / 'graph.neff')
    Path(workdir, 'meta.json').write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
