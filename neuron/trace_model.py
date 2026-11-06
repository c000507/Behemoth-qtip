"""Server-contract Llama graphs for NeuronCore-v2 (trn1), tensor parallel TP=2, all projections cFP8.

One NEFF per (R, Q, P) bucket serves EVERY layer: the 32 layers differ only in the weight and KV tensors bound
to it at run time (cpp/neuron_model.cpp), so there is no per-layer compilation.

  decode  : R requests x Q=1 token each (R padded with scratch requests)
  prefill : R=1 request x Q-token chunk (chunked prefill, the chunk attends to its own earlier chunks too)

Paged KV, exactly the server's layout (runtime/include/behemoth/attention.hpp): per layer and K/V one tensor
[pages, kv_heads_local, 16, 128], page_tokens = 16, a request owns pages listed in its page table row. The
graph scatters the T new K/V rows to flat slots (page * 16 + offset), then attends over the request's pages in
key blocks of G pages with a running (max, sum, accumulator) -- flash-attention online softmax -- so no
[queries x context] score matrix is ever materialised. Key position j of the table is valid for a query at
absolute position p iff j <= p (causal and length mask in one compare), so the block loop needs no lengths.
The KV tensors are aliased input -> output (updated in place on the device).

Inputs (order == ORDER), per rank:
  x [T,H] f32 residual (replicated), cos_sin [T,2,64] f32 for the tokens' absolute positions (computed on the
  host: llama3 rope scaling, any position), pos [T] i64, wslot [T] i64, table [R,P] i64, cache_k/cache_v,
  norms and the seven FP8 projections with per-row scales.

usage: trace_model.py WORKDIR layer|head R Q P [aliases=1]
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

TP = 2
H, HEADS, KVHEADS, D, FF, VOCAB = 4096, 32, 8, 128, 14336, 128257
VOCAB_PAD = (VOCAB + TP - 1) // TP * TP
PAGE = 16
KEY_BLOCK_PAGES = 32  # G: pages per flash block (512 keys)
ORDER = ['x', 'cos_sin', 'pos', 'wslot', 'table', 'cache_k', 'cache_v', 'norm1', 'norm2',
         'qf', 'qs', 'of', 'os', 'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']
HEAD_ORDER = ['x', 'rows', 'norm', 'w']


def shapes(R, Q, P, pages):
    T, q, kv, ff, f8 = R * Q, HEADS * D // TP, KVHEADS * D // TP, FF // TP, torch.float8_e4m3fn
    return {
        'x': ((T, H), torch.float32), 'cos_sin': ((T, 2, D // 2), torch.float32), 'pos': ((T,), torch.int64),
        'wslot': ((T,), torch.int64), 'table': ((R, P), torch.int64),
        'cache_k': ((pages, KVHEADS // TP, PAGE, D), torch.bfloat16), 'cache_v': ((pages, KVHEADS // TP, PAGE, D), torch.bfloat16),
        'norm1': ((1, H), torch.float32), 'norm2': ((1, H), torch.float32),
        'qf': ((q, H), f8), 'qs': ((q,), torch.float32), 'of': ((H, q), f8), 'os': ((H,), torch.float32),
        'kf': ((kv, H), f8), 'ks': ((kv,), torch.float32), 'vf': ((kv, H), f8), 'vs': ((kv,), torch.float32),
        'gf': ((ff, H), f8), 'gs': ((ff,), torch.float32), 'uf': ((ff, H), f8), 'us': ((ff,), torch.float32),
        'df': ((H, ff), f8), 'ds': ((H,), torch.float32),
    }


def flash_paged(q, cache_k, cache_v, table, pos, R, Q, P, kvh, rep):
    """q [R,Q,kvh,rep,D] bf16 (rope applied) -> [R,Q,kvh,rep,D] f32. Online softmax over blocks of G pages."""
    G = min(KEY_BLOCK_PAGES, P)
    assert P % G == 0
    scale = D ** -0.5
    neg = -1e30
    m = torch.full((R, Q, kvh, rep), neg)
    l = torch.zeros(R, Q, kvh, rep)
    acc = torch.zeros(R, Q, kvh, rep, D)
    qpos = pos.reshape(R, Q, 1, 1, 1)
    inblock = torch.arange(G * PAGE, dtype=torch.int64).reshape(1, 1, 1, 1, G * PAGE)
    for b in range(P // G):
        pages = table[:, b * G:(b + 1) * G]                      # [R,G]
        kb, vb = cache_k[pages], cache_v[pages]                  # [R,G,kvh,16,D]
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
    t = dict(zip(ORDER, args))
    allreduce = lambda v: xm.all_reduce(xm.REDUCE_SUM, v, groups=[list(range(TP))])
    heads, kvh = HEADS // TP, KVHEADS // TP
    rep, T, half = heads // kvh, R * Q, D // 2
    w = lambda a, b: Fp8Weight(t[a], t[b])
    x = t['x']
    h = (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * t['norm1']).to(torch.bfloat16)
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
    at = (page[:, None], torch.arange(kvh, dtype=torch.int64)[None, :], off[:, None])
    cache_k = t['cache_k'].index_put(at, k.to(torch.bfloat16))
    cache_v = t['cache_v'].index_put(at, v.to(torch.bfloat16))
    attn = flash_paged(q.to(torch.bfloat16).reshape(R, Q, kvh, rep, D), cache_k, cache_v, t['table'], t['pos'], R, Q, P, kvh, rep)
    attn = attn.reshape(T, heads * D).to(torch.bfloat16)
    x = x + allreduce(lin_fp8(attn, w('of', 'os')).float())
    h = (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * t['norm2']).to(torch.bfloat16)
    act = torch.nn.functional.silu(lin_fp8(h, w('gf', 'gs'))).float() * lin_fp8(h, w('uf', 'us')).float()
    x = x + allreduce(lin_fp8(act.to(torch.bfloat16), w('df', 'ds')).float())
    return x, cache_k, cache_v


def head_graph(S, x, rows, norm, w):
    """Final RMSNorm and this rank's vocab half of lm_head for S selected rows -> logits [S, VOCAB_PAD/2]."""
    x = torch.index_select(x, 0, rows)
    h = (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * norm).to(torch.bfloat16)
    return (h @ w.T).float()


def example_inputs(R, Q, P, pages):
    s = shapes(R, Q, P, pages)
    out = []
    for k in ORDER:
        shape, dtype = s[k]
        out.append(torch.zeros(shape).to(dtype) if dtype in (torch.float8_e4m3fn, torch.bfloat16) else torch.zeros(shape, dtype=dtype))
    return out


def main():
    workdir, kind = sys.argv[1], sys.argv[2]
    import torch_neuronx
    aliases = True
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
                    self.register_buffer('cache_k', inputs[ORDER.index('cache_k')])
                    self.register_buffer('cache_v', inputs[ORDER.index('cache_v')])

                def forward(self, *rest):
                    a = list(rest)
                    a.insert(ORDER.index('cache_k'), self.cache_k)
                    a.insert(ORDER.index('cache_v'), self.cache_v)
                    return paged_layer(R, Q, P, *a)
            fn = Layer()
            kwargs['input_output_aliases'] = {fn.cache_k: 1, fn.cache_v: 2}
            kwargs['inline_weights_to_neff'] = False
            inputs = [t for k, t in zip(ORDER, inputs) if k not in ('cache_k', 'cache_v')]
        else:
            fn = lambda *a: paged_layer(R, Q, P, *a)
        meta = {'kind': 'layer', 'R': R, 'Q': Q, 'P': P, 'order': ORDER}
    else:
        S = int(sys.argv[3])
        T = int(os.environ.get('HEAD_ROWS', '256'))
        inputs = [torch.zeros(T, H), torch.zeros(S, dtype=torch.int64), torch.ones(1, H), torch.zeros(VOCAB_PAD // TP, H, dtype=torch.bfloat16)]
        kwargs = {}
        fn = lambda x, rows, norm, w: head_graph(S, x, rows, norm, w)
        meta = {'kind': 'head', 'S': S, 'T': T, 'order': HEAD_ORDER}
    try:
        torch_neuronx.trace(fn, tuple(inputs), compiler_workdir=workdir, **kwargs)
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
    m = Path(workdir) / 'model'
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(m / 'graph.hlo'), str(m / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'],
                   cwd=workdir, check=True)
    meta['neff'] = str(Path(workdir) / 'graph.neff')
    Path(workdir, 'meta.json').write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
