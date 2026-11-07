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
Sharding: q/k/v/gate/up column-parallel (heads and kv heads divided over the ranks, so kv_heads % TP == 0),
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
        if self.HEADS % self.KVHEADS or self.HEADS % tp or self.KVHEADS % tp or self.FF % tp or self.D % 2:
            raise ValueError(f'geometry {d} is not shardable over TP={tp} (heads, kv heads and intermediate must divide; kv_heads < TP would need K/V replication)')
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
DECODE_BLOCK_PAGES = int(os.environ.get('DECODE_BLOCK_PAGES', 256))  # decode (Q = 1): few big blocks, the score row is tiny; 32-page blocks were ~50 us of serial small ops each
ORDER = ['x', 'cos_sin', 'pos', 'wslot', 'table', 'cache', 'norm1', 'norm2',
         'qf', 'qs', 'of', 'os', 'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']
# compressed QTIP layer: per QTIP projection words / codebook / SU / SV (q, o, gate, up, down), k/v FP8 as above
ORDER_QTIP = ['x', 'cos_sin', 'pos', 'wslot', 'table', 'cache', 'norm1', 'norm2', 'sel_h', 'sel_ff',
              'qw', 'ql', 'qsu', 'qsv', 'ow', 'ol', 'osu', 'osv', 'kf', 'ks', 'vf', 'vs',
              'gw', 'gl', 'gsu', 'gsv', 'uw', 'ul', 'usu', 'usv', 'dw', 'dl', 'dsu', 'dsv']
WEIGHTS = os.environ.get('WEIGHTS', 'fp8')
WT = os.environ.get('WT') == '1'  # FP8 projections stored transposed [in, out]: contraction axis first, contiguous DMA rows
HEAD_ORDER = ['x', 'rows', 'norm', 'w']
HEAD_FP8_ORDER = ['x', 'rows', 'norm', 'wf', 'ws']


def shapes(R, Q, P, pages):
    H, D, f8 = g.H, g.D, torch.float8_e4m3fn
    T, q, kv, ff = R * Q, g.HEADS * D // g.TP, g.KVHEADS * D // g.TP, g.FF // g.TP
    s = {
        'x': ((T, H), torch.float32), 'cos_sin': ((T, 2, D // 2), torch.float32), 'pos': ((T,), torch.int64),
        'wslot': ((T,), torch.int64), 'table': ((R, P), torch.int64),
        'cache': ((pages, 2, g.KVHEADS // g.TP, PAGE, D), torch.bfloat16),
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
    return s


def layer_order():
    return ORDER_QTIP if WEIGHTS == 'qtip' else ORDER


def flash_paged(q, cache, table, pos, R, Q, P, kvh, rep):
    """q [R,Q,kvh,rep,D] bf16 (rope applied) -> [R,Q,kvh,rep,D] f32. Online softmax over blocks of G pages."""
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
    D, heads, kvh = g.D, g.HEADS // g.TP, g.KVHEADS // g.TP
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
    cache = t['cache'].index_put(at, torch.stack((k, v), dim=1).to(torch.bfloat16))  # [T,2,kvh,D] -> cache[page, kv, head, off]
    attn = flash_paged(q.to(torch.bfloat16).reshape(R, Q, kvh, rep, D), cache, t['table'], t['pos'], R, Q, P, kvh, rep)
    attn = attn.reshape(T, heads * D).to(torch.bfloat16)
    if qtip:
        # o row-parallel: the rotation needs all heads (all-gather, rank order = head order), this rank multiplies its
        # columns of W~ by its slice of u, the partial z is all-reduced and finished on both ranks
        attn_full = qtip_graph.gather_tp(attn.float(), t['sel_h'], H, allreduce, g.TP)
        u = torch.index_select(qtip_graph.rotate_in(attn_full, t['osu'], H), -1, t['sel_h'])
        x = x + qtip_graph.finish(allreduce(qtip_graph.kernel_matmul(u, t['ow'], t['ol'])), t['osv'], H)
    else:
        x = x + allreduce(lin_fp8(attn, w('of', 'os')).float())
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
        act = torch.nn.functional.silu(lin_fp8(h, w('gf', 'gs'))).float() * lin_fp8(h, w('uf', 'us')).float()
        x = x + allreduce(lin_fp8(act.to(torch.bfloat16), w('df', 'ds')).float())
    return x, cache


LAYER_WEIGHTS = ['norm1', 'norm2', 'qf', 'qs', 'of', 'os', 'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']
SHARED = ['x', 'cos_sin', 'pos', 'wslot', 'table', 'rows', 'head_norm', 'head_w']
BODY_SHARED = SHARED[:5]  # a body segment (no head) takes only these and returns the residual stream


def head_graph(S, x, rows, norm, w, ws=None, transposed=False):
    """Final RMSNorm and this rank's vocab slice of lm_head for S selected rows -> logits [S, VOCAB_PAD/TP] f32.
    bf16 lm_head (w [V/TP, H]) or, with ws, an E4M3 lm_head with a per-row scale: the activation is scaled per row into
    E4M3 like the layer projections, the matmul runs fp8 x fp8 and the f32 result is rescaled (not rounded to bf16).
    transposed: w is [H, V/TP] (contraction axis first, like WT layer weights)."""
    x = torch.index_select(x, 0, rows)
    h = (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * norm).to(torch.bfloat16)
    if ws is None:
        return (h @ (w if transposed else w.T)).float()
    a = torch.amax(torch.abs(h.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    h8 = (h.float() / a).to(torch.float8_e4m3fn)
    return torch.matmul(h8, w if transposed else w.T).float() * a * ws.reshape(1, -1)


class FullModel(torch.nn.Module):
    """All L layers plus the final norm / lm_head slice as ONE graph. Inputs: SHARED, then LAYER_WEIGHTS per layer;
    the L KV tensors (K and V of a layer in one [pages, 2, kv_heads, 16, D] tensor) are module state aliased input -> output.
    Keeping the aliased tensors at one per layer matters: a graph with 64 aliased state tensors (32 layers x separate K, V) compiled
    to wrong results, 62 did not."""
    def __init__(self, R, Q, P, L, S, pages, head=True):
        super().__init__()
        self.R, self.Q, self.P, self.L, self.S, self.head = R, Q, P, L, S, head
        shape = (pages, 2, g.KVHEADS // g.TP, PAGE, g.D)
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
                    self.register_buffer('cache', inputs[ORDER.index('cache')])

                def forward(self, *rest):
                    a = list(rest)
                    a.insert(layer_order().index('cache'), self.cache)
                    x, self.cache = paged_layer(R, Q, P, *a)  # mutated buffer: enable_aliasing makes it in-place
                    return x
            fn = Layer()
            kwargs['enable_aliasing'] = True
            inputs = [t for k, t in zip(layer_order(), inputs) if k != 'cache']
        else:
            fn = lambda *a: paged_layer(R, Q, P, *a)
        if WEIGHTS not in ('fp8', 'qtip'):
            raise SystemExit(f'WEIGHTS={WEIGHTS}: expected fp8 or qtip')
        meta = {'kind': 'layer', 'R': R, 'Q': Q, 'P': P, 'pages': pages, 'order': layer_order(), 'weights': WEIGHTS, 'wt': WT}
    elif kind == 'model':
        if WEIGHTS != 'fp8':
            raise SystemExit('whole-model graphs are only generated for WEIGHTS=fp8')
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
        rows = g.VOCAB_PAD // g.TP
        inputs = [torch.zeros(T, g.H), torch.zeros(S, dtype=torch.int64), torch.ones(1, g.H)]
        kwargs = {}
        if fp8:
            inputs += [torch.zeros(*((g.H, rows) if WT else (rows, g.H))).to(torch.float8_e4m3fn), torch.ones(rows)]
            fn = lambda x, rows_, norm, wf, ws: head_graph(S, x, rows_, norm, wf, ws, WT)
        else:
            inputs += [torch.zeros(*((g.H, rows) if WT else (rows, g.H)), dtype=torch.bfloat16)]
            fn = lambda x, rows_, norm, w: head_graph(S, x, rows_, norm, w, None, WT)
        meta = {'kind': 'head', 'S': S, 'T': T, 'fp8': fp8, 'wt': WT, 'order': HEAD_FP8_ORDER if fp8 else HEAD_ORDER}
    meta['geometry'] = g.meta()
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
