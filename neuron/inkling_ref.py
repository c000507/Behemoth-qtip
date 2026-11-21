"""CPU / torch f32 reference forward of Inkling 2 Mini (`inkling2_loop`, text part): the numerical oracle for the
Neuron graphs.

Semantics follow transformers' modeling_inkling2_loop.py (PR 51050 @ 6173a2f; validated in f32 by
tests/test_inkling_ref.py against that code on a tiny random checkpoint written in the original checkpoint format,
NVFP4 + MXFP8 experts included, loaded through HF's own conversion):

  * embedding -> RMSNorm(w) (embed_norm); 24 decoder layers, 20 sliding-window (1024) + 4 global (5, 11, 17, 23);
    layers 6..17 are run `passes` (3) times with shared weights -> 48 layer VISITS per token, each with its own KV and
    conv state.
  * Block attention residual ("depth attention"): the residual stream is a list of completed blocks (embedding, then the
    partial sum of every 6 layers) plus the running partial sum. Layer inputs are softmax mixtures (learned queries)
    over the unweighted-RMS-normed [blocks..., partial] (values normed too). A block boundary layer (L % 6 == 0) closes
    its input partial into a block and restarts the partial from its attention output.
  * Loop passes 2 and 3 restart from the pre-loop state and receive 4 "messages" (q, k, v, mlp) per loop layer:
    depth attention over the previous pass's own blocks [B2, B3 partial], mixed in by rank-64 message gates
    `(1 - tanh(f)) * h + tanh(g) * m`.
  * attention: no RoPE. q = l2clip(scale * Wq x) [* log-scale tau(pos) in global layers]; k = l2clip(conv(Wk x));
    v = unweighted-rmsnorm(conv(Wv x)) per head; scores q.k / sqrt(D) + hidden-conditioned relative bias (16-dim per-head
    states, clipped, x a [16, 1024] profile bank; 0 beyond distance 1024) + causal/window mask; one sink logit per head
    and query (sink_proj x); output * sigmoid(gate_proj x) -> o_proj. Sliding layers: static depthwise causal conv (k=4)
    on k / v channels; global layers: dynamic conv (static taps + per-token per-head taps from weight_proj x) and separate
    q / k / v inputs (3 depth-attention queries, own norms). Output scales: (conv + x) / sqrt(5) static,
    (conv + x) / 3 dynamic.
  * post-attention and post-MoE static short convs over the 2304 residual channels (f32, own state each).
  * MoE: sigmoid router (f32) + selection bias, top-8, renormalised, x route_scale 8; experts [E, 2I, H] / [E, H, I],
    act = 4 tanh(g / 4) * sigmoid(g) * 25 tanh(u / 25). No shared expert.
  * final depth attention -> RMSNorm(w) -> lm_head (first 200058 rows of `unembed`) -> 30 tanh(logits / 30).

Weights are addressed by their ORIGINAL checkpoint names (`model.llm.*`), so the same loader serves the oracle and the
prepare step. `W(name)` -> f32 tensor; the expert tensors are the dequantised [E, 2I, H] gate|up (gate rows first) and
[E, H, I] down (`experts(L)`), see `dequant_experts`.

State per sequence: dict keyed by visit index v (0..47, `Cfg.visits`): {'k': [n, kvh, D], 'v': [n, kvh, D] (all past
keys / values, positions 0..n-1), 'ck', 'cv' ([C, 3] k / v conv inputs, oldest first), 'ca', 'cm' ([H, 3])}.
`forward(ids, state)` runs T new tokens of ONE sequence (prefill chunk or decode step) and returns f32 logits [T, Vu].
"""
import math

import torch
import torch.nn.functional as F

P = 'model.llm.'

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


class Cfg:
    def __init__(self, cfg):
        c = cfg.get('text_config', cfg)
        self.raw = c
        self.H = c['hidden_size']
        self.L = c['num_hidden_layers']
        self.V = c['vocab_size']
        self.Vu = c.get('unpadded_vocab_size', self.V)
        self.heads, self.kvh, self.D = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        assert c.get('swa_num_attention_heads', self.heads) == self.heads
        assert c.get('swa_num_key_value_heads', self.kvh) == self.kvh and c.get('swa_head_dim', self.D) == self.D
        self.W = c['sliding_window_size']
        self.d_rel, self.rel_extent = c['d_rel'], c['rel_extent']
        self.n_floor, self.alpha = c['log_scaling_n_floor'], c['log_scaling_alpha']
        self.eps = float(c['rms_norm_eps'])
        self.qk_clip, self.rel_clip = float(c['qk_l2_clip_threshold']), float(c['rel_l2_clip_threshold'])
        local = c.get('local_layer_ids')
        if local is not None:
            self.swa = [l in set(local) for l in range(self.L)]
        else:
            self.swa = [(l + 1) % 6 != 0 for l in range(self.L)]
        self.K = c.get('sconv_kernel_size', c.get('conv_kernel_size', 4))
        self.E, self.topk = c['n_routed_experts'], c['num_experts_per_tok']
        self.I = c.get('intermediate_size', c.get('moe_intermediate_size'))
        assert not c.get('n_shared_experts', 0)
        self.gate_cap, self.up_cap = float(c['mlp_gate_softcap']), float(c['mlp_up_softcap'])
        self.route_scale = float(c['route_scale'])
        self.final_cap = float(c['final_logit_softcapping'])
        self.blk = c['n_layers_per_attn_res_block']
        lc = c.get('loop_config') or {}
        self.ls = lc.get('core_start', c.get('loop_layer_index_start', 6))
        self.le = lc.get('core_end', c.get('loop_layer_index_end', 18))
        self.passes = lc.get('num_passes', c.get('loop_num_passes', 3))
        self.rank = lc.get('gate_rank', c.get('loop_gate_rank', 64))
        assert self.ls % self.blk == 0 and self.le % self.blk == 0 and self.ls > 0
        # visit order: (layer, pass); pass 0 for non-loop layers
        v = [(l, 0) for l in range(self.ls)]
        for p in range(self.passes):
            v += [(l, p) for l in range(self.ls, self.le)]
        v += [(l, 0) for l in range(self.le, self.L)]
        self.visits = v
        # number of completed blocks visible to the attention input of each layer (before its own boundary append)
        # = 1 (embedding) + floor((l - 1) / blk) for l >= 1
        self.nblocks = [0] + [1 + (l - 1) // self.blk for l in range(1, self.L)]

    def is_loop(self, l):
        return self.ls <= l < self.le


# ---------------------------------------------------------------------------------------------------------- math


def urms(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def rms(x, w, eps):
    return w * urms(x, eps)


def l2clip(x, thr):
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (thr / (n + 1e-6)).clamp(max=1.0)


def tau(pos, cfg):
    """log-scale factor per position [T] (global layers)."""
    return 1.0 + cfg.alpha * torch.log(((pos + 1).float() / cfg.n_floor).clamp(min=1.0))


def depth_attn(stack, q):
    """stack [T, n, H] (blocks..., partial), q [nq, H] -> [T, nq, H]: softmax(q . urms(s) / sqrt(H)) @ urms(s)."""
    s = urms(stack)
    lg = torch.einsum('qh,tnh->tqn', q, s) * (s.shape[-1] ** -0.5)
    return torch.einsum('tqn,tnh->tqh', torch.softmax(lg, dim=-1), s)


def msg_gate(h, m, down, forget, gain):
    z = torch.cat([urms(h), urms(m)], dim=-1) @ down.T
    return (1 - torch.tanh(z @ forget.T)) * h + torch.tanh(z @ gain.T) * m


def short_conv(x, w, state):
    """Static depthwise causal conv. x [T, C], w [C, K] (w[:, K-1] = current token), state [C, K-1] oldest first.
    Returns ((conv + x) / sqrt(K + 1), new state)."""
    K = w.shape[1]
    ctx = torch.cat([state.T, x], dim=0)                     # [T + K - 1, C]
    T = x.shape[0]
    y = sum(w[:, j] * ctx[j:j + T] for j in range(K))
    return (y + x) / math.sqrt(K + 1), ctx[-(K - 1):].T.contiguous()


def dyn_conv(x, u, static_w, wp, nh, state):
    """Dynamic conv of the global layers. x [T, C] (C = nh * D), u [T, H] the dynamic input, static_w [K, C]
    (static_w[0] = current token), wp [nh * K, H]. Returns ((conv + x) / sqrt(2K + 1), new state)."""
    K = static_w.shape[0]
    T, C = x.shape
    D = C // nh
    dw = (u @ wp.T).view(T, nh, 1, K)
    wts = static_w.T.reshape(nh, D, K) + dw                   # [T, nh, D, K]
    ctx = torch.cat([state.T, x], dim=0)                      # [T + K - 1, C]
    win = torch.stack([ctx[K - 1 - lag:K - 1 - lag + T] for lag in range(K)], dim=-1)   # [T, C, K], lag 0 = current
    y = (win.view(T, nh, D, K) * wts).sum(-1).reshape(T, C)
    return (y + x) / math.sqrt(2 * K + 1), ctx[-(K - 1):].T.contiguous()


def swiglu_cap(g, u, cfg):
    return torch.tanh(g / cfg.gate_cap) * cfg.gate_cap * torch.sigmoid(g) * torch.tanh(u / cfg.up_cap) * cfg.up_cap


def route(x, wg, bias, cfg):
    """sigmoid router: returns (idx [T, k], weights [T, k] (normalised, x route_scale))."""
    s = torch.sigmoid(x @ wg.T)
    idx = torch.topk(s + bias, cfg.topk, dim=-1, sorted=False).indices
    w = s.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * cfg.route_scale


def moe(x, wg, bias, gu, dn, cfg):
    idx, w = route(x, wg, bias, cfg)
    out = torch.zeros_like(x)
    for e in torch.unique(idx).tolist():
        tok, kk = torch.where(idx == e)
        h = x[tok] @ gu[e].T
        a = swiglu_cap(h[:, :cfg.I], h[:, cfg.I:], cfg)
        out.index_add_(0, tok, (a @ dn[e].T) * w[tok, kk, None])
    return out


# ------------------------------------------------------------------------------------------------ expert formats


def dequant_nvfp4(packed, scale, amax, denom):
    """packed U8 [E, R, C/2] (even element in the low nibble), scale fp8 [E, R, C/16], amax [E], denom [1] -> f32 [E, R, C]."""
    lut = torch.tensor(E2M1, dtype=torch.float32)
    p = packed.to(torch.uint8).long()
    v = torch.stack((lut[p & 0xF], lut[p >> 4]), dim=-1).reshape(*packed.shape[:2], -1)     # [E, R, C]
    E, R, C = v.shape
    v = v.view(E, R, C // 16, 16) * scale.float()[..., None]
    return (v * (amax.float() / denom.float())[:, None, None, None]).reshape(E, R, C)


def dequant_mxfp8(w, scale):
    """w fp8 e4m3 [E, R, C], scale U8 [E, R, C/32] (power of two 2^(s - 127)) -> f32."""
    E, R, C = w.shape
    v = w.float().view(E, R, C // 32, 32) * torch.exp2(scale.float() - 127)[..., None]
    return v.reshape(E, R, C)


def dequant_experts(get, L):
    """get(name) -> raw tensor (checkpoint dtype). Returns (gate_up [E, 2I, H] f32, gate rows first; down [E, H, I])."""
    pre = f'{P}layers.{L}.mlp.experts.'
    out = []
    for w in ('w13', 'w2'):
        t = get(pre + f'{w}_weight')
        s = get(pre + f'{w}_weight_scale')
        if t.dtype == torch.uint8:
            d = dequant_nvfp4(t, s, get(pre + f'{w}_weight_amax'), get(pre + f'{w}_weight_outer_scale_denominator'))
        else:
            d = dequant_mxfp8(t, s)
        out.append(d)
    gu, dn = out
    E, R, H = gu.shape
    # checkpoint rows are interleaved (gate_0, up_0, gate_1, ...): de-interleave to [gate; up] (HF `Interleave(dim=1)`)
    gu = gu.view(E, R // 2, 2, H).transpose(1, 2).reshape(E, R, H)
    return gu, dn


def expert_format(names, L):
    """'nvfp4' / 'mxfp8' from the presence of the amax tensors."""
    return 'nvfp4' if f'{P}layers.{L}.mlp.experts.w13_weight_amax' in names else 'mxfp8'


# ------------------------------------------------------------------------------------------------------ the model


class Ref:
    """W: callable name -> f32 tensor (checkpoint names); X: callable layer -> (gate_up, down) f32 expert tensors."""

    def __init__(self, cfg, W, X):
        self.c = cfg if isinstance(cfg, Cfg) else Cfg(cfg)
        self.W, self.X = W, X

    def lw(self, l, n):
        return self.W(f'{P}layers.{l}.{n}')

    # attention of one layer visit; qh / kh / vh are the normed inputs [T, H]
    def attention(self, l, qh, kh, vh, pos, st):
        c = self.c
        g = c.swa[l]
        T = qh.shape[0]
        nh, kvh, D = c.heads, c.kvh, c.D
        a = 'attn.'
        q = (qh @ self.lw(l, a + 'wq_du.weight').T).view(T, nh, D) * self.lw(l, a + 'q_l2_clip.scale')
        q = l2clip(q, c.qk_clip)
        kx = kh @ self.lw(l, a + 'wk_dv.weight').T
        vx = vh @ self.lw(l, a + 'wv_dv.weight').T
        zero = torch.zeros(kvh * D, c.K - 1)
        if g:
            k, st['ck'] = short_conv(kx, self.lw(l, a + 'k_sconv.weight').squeeze(1), st.get('ck', zero))
            v, st['cv'] = short_conv(vx, self.lw(l, a + 'v_sconv.weight').squeeze(1), st.get('cv', zero))
        else:
            k, st['ck'] = dyn_conv(kx, kh, self.lw(l, a + 'k_sconv.static_w'), self.lw(l, a + 'k_sconv.weight_proj'),
                                   kvh, st.get('ck', zero))
            v, st['cv'] = dyn_conv(vx, vh, self.lw(l, a + 'v_sconv.static_w'), self.lw(l, a + 'v_sconv.weight_proj'),
                                   kvh, st.get('cv', zero))
        k = l2clip(k.view(T, kvh, D), c.qk_clip)
        v = urms(v.view(T, kvh, D), c.eps)
        r = (qh @ self.lw(l, a + 'wr_du.weight').T).view(T, nh, c.d_rel) * self.lw(l, a + 'rel_l2_clip.scale')
        r = l2clip(r, c.rel_clip)
        if not g:
            tq = tau(pos, c)[:, None, None]
            q, r = q * tq, r * tq
        rl = r @ self.lw(l, a + 'rel_logits_proj.proj')                 # [T, nh, extent]
        sink = qh @ self.lw(l, a + 'attn_sink_proj.weight').T          # [T, nh]
        kall = torch.cat([st['k'], k], 0) if 'k' in st else k
        vall = torch.cat([st['v'], v], 0) if 'v' in st else v
        st['k'], st['v'] = kall, vall
        n = kall.shape[0]
        kpos = torch.arange(n)
        dist = pos[:, None] - kpos[None, :]                             # [T, n]
        ext = c.W if g else c.rel_extent
        rep = nh // kvh
        kr = kall.repeat_interleave(rep, dim=1)                         # [n, nh, D]
        vr = vall.repeat_interleave(rep, dim=1)
        s = torch.einsum('thd,nhd->thn', q, kr) * (D ** -0.5)
        inb = (dist >= 0) & (dist < ext)
        bias = rl.gather(-1, dist.clamp(0, ext - 1)[:, None, :].expand(T, nh, n))
        s = s + torch.where(inb[:, None, :], bias, torch.zeros(()))
        allowed = dist >= 0
        if g:
            allowed = allowed & (dist < c.W)
        s = s.masked_fill(~allowed[:, None, :], float('-inf'))
        p = torch.softmax(torch.cat([s, sink[..., None]], -1), dim=-1)[..., :-1]
        o = torch.einsum('thn,nhd->thd', p, vr).reshape(T, nh * D)
        o = o * torch.sigmoid(qh @ self.lw(l, a + 'gate_attn.weight').T)
        return o @ self.lw(l, a + 'wo_ud.weight').T

    def mgate(self, l, which, h, m):
        b = f'controllers.{which}.'
        return msg_gate(h, m, self.lw(l, b + 'down_weight'), self.lw(l, b + 'forget_weight'), self.lw(l, b + 'gain_weight'))

    def layer(self, l, h, blocks, pos, st, msgs=None):
        """One visit. h [T, H] partial, blocks [T, n, H] completed blocks. Returns new partial (the model appends the
        input partial to `blocks` itself at a boundary)."""
        c = self.c
        g = c.swa[l]
        bnd = l % c.blk == 0
        res = h
        if g:
            if l > 0:
                h = depth_attn(torch.cat([blocks, h[:, None]], 1), self.W(f'{P}attn_res.attn_queries.{l - 1}')[None])[:, 0]
            nw = self.lw(l, 'attn_norm.weight')
            if msgs is None:
                qh = kh = vh = rms(h, nw, c.eps)
            else:
                qh = rms(self.mgate(l, 'attn', h, msgs[:, 0]), nw, c.eps)
                kh = rms(self.mgate(l, 'k', h, msgs[:, 1]), nw, c.eps)
                vh = rms(self.mgate(l, 'v', h, msgs[:, 2]), nw, c.eps)
        else:
            if l > 0:
                qkv = depth_attn(torch.cat([blocks, h[:, None]], 1), self.W(f'{P}attn_res.attn_queries.{l - 1}'))
                qi, ki, vi = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            else:
                qi = ki = vi = h
            if msgs is not None:
                qi = self.mgate(l, 'attn', qi, msgs[:, 0])
                ki = self.mgate(l, 'k', ki, msgs[:, 1])
                vi = self.mgate(l, 'v', vi, msgs[:, 2])
            qh = rms(qi, self.lw(l, 'q_input_norm.weight'), c.eps)
            kh = rms(ki, self.lw(l, 'k_input_norm.weight'), c.eps)
            vh = rms(vi, self.lw(l, 'v_input_norm.weight'), c.eps)
        a = self.attention(l, qh, kh, vh, pos, st)
        zero = torch.zeros(c.H, c.K - 1)
        a, st['ca'] = short_conv(a, self.lw(l, 'attn_sconv.weight').squeeze(1), st.get('ca', zero))
        if bnd:
            blocks = torch.cat([blocks, res[:, None]], 1)
            res = a
        else:
            res = res + a
        hm = depth_attn(torch.cat([blocks, res[:, None]], 1), self.W(f'{P}attn_res.mlp_queries.{l}')[None])[:, 0]
        if msgs is not None:
            hm = self.mgate(l, 'mlp', hm, msgs[:, 3])
        gu, dn = self.X(l)
        m = moe(rms(hm, self.lw(l, 'mlp_norm.weight'), c.eps), self.lw(l, 'mlp.gate.weight'),
                self.lw(l, 'mlp.gate.bias'), gu, dn, c)
        m, st['cm'] = short_conv(m, self.lw(l, 'mlp_sconv.weight').squeeze(1), st.get('cm', zero))
        return res + m

    def messages(self, ll, prev_blocks, prev_h):
        """4 messages [T, 4, H] for loop layer index ll from the previous pass's blocks + final partial."""
        q = torch.stack([self.W(f'{P}middle_cycle_cross_pass_queries')[2 * ll],
                         self.W(f'{P}middle_cycle_cross_pass_kv_queries')[2 * ll],
                         self.W(f'{P}middle_cycle_cross_pass_kv_queries')[2 * ll + 1],
                         self.W(f'{P}middle_cycle_cross_pass_queries')[2 * ll + 1]])
        return depth_attn(torch.cat([prev_blocks, prev_h[:, None]], 1), q)

    def embed(self, ids):
        e = self.W(f'{P}embed.weight')[ids]
        return rms(e, self.W(f'{P}embed_norm.weight'), self.c.eps)

    def hidden(self, ids, state, pos0=None, x=None, trace=None):
        """Final normed hidden [T, H]. state: dict visit -> per-visit state (mutated). trace(v, h) is called after
        every visit (debug)."""
        c = self.c
        h = self.embed(ids) if x is None else x
        T = h.shape[0]
        if pos0 is None:
            pos0 = state.get('n', 0)
        pos = torch.arange(T) + pos0
        blocks = h.new_zeros(T, 0, c.H)
        vi = 0

        def run(l, h, blocks, msgs=None):
            nonlocal vi
            nh = self.layer(l, h, blocks, pos, state.setdefault(vi, {}), msgs)
            if trace is not None:
                trace(vi, nh)
            vi += 1
            if l % c.blk == 0:
                blocks = torch.cat([blocks, h[:, None]], 1)
            return nh, blocks

        for l in range(c.ls):
            h, blocks = run(l, h, blocks)
        pre_h, pre_b = h, blocks
        npre = blocks.shape[1]
        for l in range(c.ls, c.le):
            h, blocks = run(l, h, blocks)
        for p in range(1, c.passes):
            prev_b, prev_h = blocks[:, npre + 1:], h
            h, blocks = pre_h, pre_b
            for ll, l in enumerate(range(c.ls, c.le)):
                h, blocks = run(l, h, blocks, self.messages(ll, prev_b, prev_h))
        for l in range(c.le, c.L):
            h, blocks = run(l, h, blocks)
        h = depth_attn(torch.cat([blocks, h[:, None]], 1), self.W(f'{P}attn_res.output_query')[None])[:, 0]
        state['n'] = pos0 + T
        return rms(h, self.W(f'{P}norm.weight'), c.eps)

    def logits(self, h):
        c = self.c
        lg = h @ self.W(f'{P}unembed.weight')[:c.Vu].T
        return torch.tanh(lg / c.final_cap) * c.final_cap

    def forward(self, ids, state, **kw):
        return self.logits(self.hidden(ids, state, **kw))


# ------------------------------------------------------------------------------------------ random checkpoints


def quant_nvfp4(w, gen=None):
    """f32 [E, R, C] -> (packed U8 [E, R, C/2], fp8 scale [E, R, C/16], amax [E], denom [1]) in the checkpoint format
    (round to nearest E2M1 with the per-16 block scale; used to build test checkpoints)."""
    E, R, C = w.shape
    amax = w.abs().reshape(E, -1).amax(-1)
    denom = torch.tensor([6.0 * 448.0])
    g = (amax / denom)                                                     # per-expert factor
    blk = w.view(E, R, C // 16, 16)
    bs = (blk.abs().amax(-1) / 6.0) / g[:, None, None]
    bs = bs.clamp(min=2 ** -9).to(torch.float8_e4m3fn)
    q = blk / (bs.float()[..., None] * g[:, None, None, None])
    lut = torch.tensor(E2M1[:8])
    code = (q.abs()[..., None] - lut).abs().argmin(-1)
    code = code + 8 * (q < 0).long() * (code > 0).long()
    code = code.view(E, R, C)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, bs, amax.float(), denom.float()


def quant_mxfp8(w):
    E, R, C = w.shape
    blk = w.view(E, R, C // 32, 32)
    e = torch.ceil(torch.log2(blk.abs().amax(-1).clamp(min=1e-30) / 448.0)).clamp(-127, 127)
    s = (e + 127).to(torch.uint8)
    q = (blk / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn).view(E, R, C)
    return q, s


def random_checkpoint(cfg, seed=0, scale=0.05, mxfp8_layers=None, layers=None, experts=True, vocab=None):
    """Random tensors in the ORIGINAL checkpoint naming / dtypes (bf16 dense, f32 attn_res / queries / router bias,
    NVFP4 experts, MXFP8 experts in `mxfp8_layers`). layers: only these layers (device tests at real shapes);
    experts=False leaves the expert tensors out; vocab: embedding / unembedding rows (default: the config's)."""
    c = cfg if isinstance(cfg, Cfg) else Cfg(cfg)
    gen = torch.Generator().manual_seed(seed)
    mx = set(mxfp8_layers if mxfp8_layers is not None else [])
    V = vocab or c.V

    def r(*s, sc=scale, dt=torch.bfloat16):
        return (torch.randn(*s, generator=gen) * sc).to(dt)

    H, nh, kvh, D, K, E, I = c.H, c.heads, c.kvh, c.D, c.K, c.E, c.I
    t = {}
    t[P + 'embed.weight'] = r(V, H, sc=1.0)
    t[P + 'embed_norm.weight'] = (1 + r(H, sc=0.1, dt=torch.float32)).bfloat16()
    t[P + 'norm.weight'] = (1 + r(H, sc=0.1, dt=torch.float32)).bfloat16()
    t[P + 'unembed.weight'] = r(V, H, sc=0.05)
    t[P + 'attn_res.output_query'] = r(H, sc=0.5, dt=torch.float32)
    nl = c.le - c.ls
    t[P + 'middle_cycle_cross_pass_queries'] = r(2 * nl, H, sc=0.5, dt=torch.float32)
    t[P + 'middle_cycle_cross_pass_kv_queries'] = r(2 * nl, H, sc=0.5, dt=torch.float32)
    for l in (range(c.L) if layers is None else layers):
        p = f'{P}layers.{l}.'
        g = c.swa[l]
        if l > 0:
            t[f'{P}attn_res.attn_queries.{l - 1}'] = r(H, sc=0.5, dt=torch.float32) if g else r(3, H, sc=0.5, dt=torch.float32)
        t[f'{P}attn_res.mlp_queries.{l}'] = r(H, sc=0.5, dt=torch.float32)
        a = p + 'attn.'
        t[a + 'wq_du.weight'] = r(nh * D, H)
        t[a + 'wk_dv.weight'] = r(kvh * D, H)
        t[a + 'wv_dv.weight'] = r(kvh * D, H)
        t[a + 'wo_ud.weight'] = r(H, nh * D)
        t[a + 'gate_attn.weight'] = r(nh * D, H)
        t[a + 'wr_du.weight'] = r(nh * c.d_rel, H)
        t[a + 'attn_sink_proj.weight'] = r(nh, H)
        t[a + 'q_l2_clip.scale'] = (1 + r(nh, D, sc=0.2, dt=torch.float32)).bfloat16()
        t[a + 'rel_l2_clip.scale'] = (1 + r(nh, c.d_rel, sc=0.2, dt=torch.float32)).bfloat16()
        t[a + 'rel_logits_proj.proj'] = r(c.d_rel, c.W if g else c.rel_extent, sc=0.5)
        if g:
            t[a + 'k_sconv.weight'] = r(kvh * D, 1, K, sc=0.3)
            t[a + 'v_sconv.weight'] = r(kvh * D, 1, K, sc=0.3)
            t[p + 'attn_norm.weight'] = (1 + r(H, sc=0.1, dt=torch.float32)).bfloat16()
        else:
            for kv in 'kv':
                t[a + f'{kv}_sconv.static_w'] = r(K, kvh * D, sc=0.3)
                t[a + f'{kv}_sconv.weight_proj'] = r(kvh * K, H, sc=0.05)
            for n in 'qkv':
                t[p + f'{n}_input_norm.weight'] = (1 + r(H, sc=0.1, dt=torch.float32)).bfloat16()
        t[p + 'attn_sconv.weight'] = r(H, 1, K, sc=0.3)
        t[p + 'mlp_sconv.weight'] = r(H, 1, K, sc=0.3)
        t[p + 'mlp_norm.weight'] = (1 + r(H, sc=0.1, dt=torch.float32)).bfloat16()
        t[p + 'mlp.gate.weight'] = r(E, H, sc=0.1)
        t[p + 'mlp.gate.bias'] = r(E, sc=0.1, dt=torch.float32)
        if c.is_loop(l):
            for w in ('attn', 'k', 'v', 'mlp'):
                b = p + f'controllers.{w}.'
                t[b + 'down_weight'] = r(c.rank, 2 * H, sc=0.1)
                t[b + 'forget_weight'] = r(H, c.rank, sc=0.1)
                t[b + 'gain_weight'] = r(H, c.rank, sc=0.1)
        if not experts:
            continue
        e = p + 'mlp.experts.'
        w13 = torch.randn(E, 2 * I, H, generator=gen) * scale
        w2 = torch.randn(E, H, I, generator=gen) * scale
        if l in mx:
            t[e + 'w13_weight'], t[e + 'w13_weight_scale'] = quant_mxfp8(w13)
            t[e + 'w2_weight'], t[e + 'w2_weight_scale'] = quant_mxfp8(w2)
        else:
            (t[e + 'w13_weight'], t[e + 'w13_weight_scale'], t[e + 'w13_weight_amax'],
             t[e + 'w13_weight_outer_scale_denominator']) = quant_nvfp4(w13)
            (t[e + 'w2_weight'], t[e + 'w2_weight_scale'], t[e + 'w2_weight_amax'],
             t[e + 'w2_weight_outer_scale_denominator']) = quant_nvfp4(w2)
    return t


def from_tensors(cfg, t):
    """Ref over an in-memory checkpoint dict (raw dtypes)."""
    c = cfg if isinstance(cfg, Cfg) else Cfg(cfg)
    cache = {}

    def W(n):
        if n not in cache:
            cache[n] = t[n].float()
        return cache[n]

    xc = {}

    def X(l):
        if l not in xc:
            xc[l] = dequant_experts(lambda n: t[n], l)
        return xc[l]

    return Ref(c, W, X)
