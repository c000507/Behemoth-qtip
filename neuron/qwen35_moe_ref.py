"""CPU / torch f32 reference forward of Qwen3.5 MoE (`qwen3_5_moe`, text part): the numerical oracle for the Neuron graphs.

Semantics follow transformers' modeling_qwen3_5_moe.py exactly (validated bit-close in f32 by tests/test_qwen35_ref.py
for prefill and for step-by-step decode with carried state):
  * RMSNorm with zero-centred weights: y = x * rsqrt(mean(x^2) + eps) * (1 + w)  (input / post-attention / q / k / final norms)
  * Gated DeltaNet (linear_attention layers): in_proj_qkv -> depthwise causal conv1d (kernel 4) + SiLU -> per-head L2 norm of
    q, k -> beta = sigmoid(in_proj_b x), g = -exp(A_log) * softplus(in_proj_a x + dt_bias) -> recurrence per value head
    (query/key heads repeat_interleave'd onto the value heads) -> gated RMSNorm(w) * silu(in_proj_z x) -> out_proj
  * full attention: q|gate from q_proj (per head 256 q then 256 gate), q/k per-head norm, partial RoPE (first 64 dims,
    rotate_half, theta 1e7), GQA softmax attention, * sigmoid(gate), o_proj
  * MoE: softmax router over all experts, top-k, renormalised; fused experts [E, 2*I, H] / [E, H, I]; shared expert
    scaled by sigmoid(shared_expert_gate x); output = routed + shared.
The recurrence is evaluated token by token for prefill too (the chunked form of HF is mathematically identical).

State per sequence (dict by layer index): linear layers {'S': [Hv, Dk, Dv] f32, 'conv': [C, K-1] f32 (oldest first)},
attention layers {'k': [n, kvh, D], 'v': [n, kvh, D]} (all past keys). `forward` takes a batch of one sequence.

Weights: dict of HF tensor names (prefix 'model.language_model.' or 'model.') -> f32 tensors, e.g. from
`hf_model.state_dict()` or from safetensors. `random_weights(cfg)` builds a random set of the real shapes for tests.
"""
import math

import torch
import torch.nn.functional as F


class Cfg:
    """Text geometry of a qwen3_5_moe config (dict; `text_config` unwrapped)."""

    def __init__(self, cfg):
        c = cfg.get('text_config', cfg)
        self.raw = c
        self.H = c['hidden_size']
        self.L = c['num_hidden_layers']
        self.layer_types = list(c['layer_types'])
        self.eps = float(c['rms_norm_eps'])
        self.V = c['vocab_size']
        # linear attention
        self.Hk, self.Dk = c['linear_num_key_heads'], c['linear_key_head_dim']
        self.Hv, self.Dv = c['linear_num_value_heads'], c['linear_value_head_dim']
        self.K = c['linear_conv_kernel_dim']
        self.C = 2 * self.Hk * self.Dk + self.Hv * self.Dv            # conv channels (q | k | v)
        # full attention
        self.heads, self.kvh = c['num_attention_heads'], c['num_key_value_heads']
        self.D = c.get('head_dim') or self.H // self.heads
        rp = c.get('rope_parameters') or {}
        self.theta = float(rp.get('rope_theta', c.get('rope_theta', 10000.0)))
        self.rot = int(self.D * float(rp.get('partial_rotary_factor', c.get('partial_rotary_factor', 1.0))))
        self.gate = bool(c.get('attn_output_gate', True))
        # MoE
        self.E, self.topk, self.I = c['num_experts'], c['num_experts_per_tok'], c['moe_intermediate_size']
        self.SI = c.get('shared_expert_intermediate_size', 0) or 0

    def is_linear(self, l):
        return self.layer_types[l] == 'linear_attention'


def rms(x, w, eps):
    """Qwen3.5 RMSNorm: (1 + w) scaling, f32."""
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * (1.0 + w)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def rope_cos_sin(pos, rot, theta):
    """cos/sin [T, rot] for HF rotate_half RoPE over `rot` dims (emb = cat(freqs, freqs))."""
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    f = pos.to(torch.float64).reshape(-1, 1) * inv.reshape(1, -1)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos().float(), emb.sin().float()


def apply_rope(x, cos, sin):
    """x [T, heads, D], rotate the first cos.shape[-1] dims (rotate_half convention)."""
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    h = rot // 2
    rh = torch.cat((-xr[..., h:], xr[..., :h]), dim=-1)
    return torch.cat((xr * cos.unsqueeze(1) + rh * sin.unsqueeze(1), xp), dim=-1)


class Ref:
    def __init__(self, cfg, weights):
        self.g = Cfg(cfg) if not isinstance(cfg, Cfg) else cfg
        self.w = {k: v.float() for k, v in weights.items()}
        self.p = 'model.language_model.' if any(k.startswith('model.language_model.') for k in self.w) else 'model.'

    def t(self, name):
        return self.w[self.p + name] if not name.startswith('lm_head') else self.w[name]

    # ---------------------------------------------------------------------------------------------------- state
    def init_state(self):
        g = self.g
        st = {}
        for l in range(g.L):
            if g.is_linear(l):
                st[l] = {'S': torch.zeros(g.Hv, g.Dk, g.Dv), 'conv': torch.zeros(g.C, g.K - 1)}
            else:
                st[l] = {'k': torch.zeros(0, g.kvh, g.D), 'v': torch.zeros(0, g.kvh, g.D)}
        return st

    # ---------------------------------------------------------------------------------------------------- blocks
    def linear_attn(self, l, n, st):
        """n [T, H] normalised input -> [T, H]; st updated in place (conv / recurrent state after the last token)."""
        g, p = self.g, f'layers.{l}.linear_attn.'
        T = n.shape[0]
        qkv = n @ self.t(p + 'in_proj_qkv.weight').T                                     # [T, C]
        z = n @ self.t(p + 'in_proj_z.weight').T                                         # [T, Hv*Dv]
        b = n @ self.t(p + 'in_proj_b.weight').T                                         # [T, Hv]
        a = n @ self.t(p + 'in_proj_a.weight').T                                         # [T, Hv]
        cw = self.t(p + 'conv1d.weight').reshape(g.C, g.K)                               # [C, K]
        seq = torch.cat((st['conv'], qkv.T), dim=1)                                      # [C, K-1+T]
        conv = sum(seq[:, j:j + T] * cw[:, j:j + 1] for j in range(g.K))                 # [C, T]
        st['conv'] = seq[:, -(g.K - 1):].clone()
        act = F.silu(conv).T                                                             # [T, C]
        kd, vd = g.Hk * g.Dk, g.Hv * g.Dv
        q = act[:, :kd].reshape(T, g.Hk, g.Dk)
        k = act[:, kd:2 * kd].reshape(T, g.Hk, g.Dk)
        v = act[:, 2 * kd:].reshape(T, g.Hv, g.Dv)
        rep = g.Hv // g.Hk
        if rep > 1:
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        q = l2norm(q) * (g.Dk ** -0.5)
        k = l2norm(k)
        beta = torch.sigmoid(b)                                                          # [T, Hv]
        gg = -torch.exp(self.t(p + 'A_log')) * F.softplus(a + self.t(p + 'dt_bias'))     # [T, Hv]
        S = st['S']                                                                      # [Hv, Dk, Dv]
        outs = []
        for i in range(T):
            S = S * torch.exp(gg[i]).reshape(g.Hv, 1, 1)
            mem = torch.einsum('hkd,hk->hd', S, k[i])
            delta = (v[i] - mem) * beta[i].unsqueeze(-1)
            S = S + k[i].unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append(torch.einsum('hkd,hk->hd', S, q[i]))
        st['S'] = S
        o = torch.stack(outs)                                                            # [T, Hv, Dv]
        o = o * torch.rsqrt(torch.mean(o * o, dim=-1, keepdim=True) + g.eps) * self.t(p + 'norm.weight')
        o = o * F.silu(z.reshape(T, g.Hv, g.Dv))
        return o.reshape(T, vd) @ self.t(p + 'out_proj.weight').T

    def full_attn(self, l, n, pos, st):
        g, p = self.g, f'layers.{l}.self_attn.'
        T = n.shape[0]
        qg = n @ self.t(p + 'q_proj.weight').T
        if g.gate:
            qg = qg.reshape(T, g.heads, 2 * g.D)
            q, gate = qg[..., :g.D], qg[..., g.D:]
        else:
            q, gate = qg.reshape(T, g.heads, g.D), None
        k = (n @ self.t(p + 'k_proj.weight').T).reshape(T, g.kvh, g.D)
        v = (n @ self.t(p + 'v_proj.weight').T).reshape(T, g.kvh, g.D)
        q = rms(q, self.t(p + 'q_norm.weight'), g.eps)
        k = rms(k, self.t(p + 'k_norm.weight'), g.eps)
        cos, sin = rope_cos_sin(pos, g.rot, g.theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        K = torch.cat((st['k'], k)); V = torch.cat((st['v'], v))                           # [n, kvh, D]
        st['k'], st['v'] = K, V
        rep = g.heads // g.kvh
        Kr = K.repeat_interleave(rep, dim=1); Vr = V.repeat_interleave(rep, dim=1)         # [n, heads, D]
        s = torch.einsum('thd,nhd->htn', q, Kr) / math.sqrt(g.D)
        n_past = K.shape[0] - T
        kpos = torch.arange(K.shape[0]).reshape(1, 1, -1)
        qpos = (torch.arange(T) + n_past).reshape(1, T, 1)
        s = torch.where(kpos <= qpos, s, torch.full_like(s, -float('inf')))
        pr = torch.softmax(s, dim=-1)
        o = torch.einsum('htn,nhd->thd', pr, Vr)                                            # [T, heads, D]
        if gate is not None:
            o = o * torch.sigmoid(gate)
        return o.reshape(T, g.heads * g.D) @ self.t(p + 'o_proj.weight').T

    def moe(self, l, n):
        g, p = self.g, f'layers.{l}.mlp.'
        T = n.shape[0]
        logits = n @ self.t(p + 'gate.weight').T                                            # [T, E]
        probs = torch.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, g.topk, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        gu, dn = self.t(p + 'experts.gate_up_proj'), self.t(p + 'experts.down_proj')        # [E, 2I, H], [E, H, I]
        out = torch.zeros(T, g.H)
        for t in range(T):
            for j in range(g.topk):
                e = int(top_i[t, j])
                h = gu[e] @ n[t]
                act = F.silu(h[:g.I]) * h[g.I:]
                out[t] += (dn[e] @ act) * top_w[t, j]
        if g.SI:
            sh = F.silu(n @ self.t(p + 'shared_expert.gate_proj.weight').T) * (n @ self.t(p + 'shared_expert.up_proj.weight').T)
            sh = sh @ self.t(p + 'shared_expert.down_proj.weight').T
            sg = torch.sigmoid(n @ self.t(p + 'shared_expert_gate.weight').T)             # [T, 1]
            out = out + sg * sh
        return out

    # ---------------------------------------------------------------------------------------------------- forward
    def forward(self, tokens, state, positions=None, return_hidden=False):
        """tokens [T] int64 of ONE sequence continuing `state` (from init_state); positions default to the state's length.
        -> logits [T, V] f32 (and the final hidden [T, H] with return_hidden); state updated in place."""
        g = self.g
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        T = tokens.shape[0]
        if positions is None:
            past = next((state[l]['k'].shape[0] for l in range(g.L) if not g.is_linear(l)), 0)
            positions = torch.arange(T) + past
        positions = torch.as_tensor(positions, dtype=torch.int64)
        x = self.t('embed_tokens.weight')[tokens]
        for l in range(g.L):
            pl = f'layers.{l}.'
            n = rms(x, self.t(pl + 'input_layernorm.weight'), g.eps)
            x = x + (self.linear_attn(l, n, state[l]) if g.is_linear(l) else self.full_attn(l, n, positions, state[l]))
            n = rms(x, self.t(pl + 'post_attention_layernorm.weight'), g.eps)
            x = x + self.moe(l, n)
        h = rms(x, self.t('norm.weight'), g.eps)
        logits = h @ self.t('lm_head.weight').T
        return (logits, h) if return_hidden else logits


class Mtp:
    """The checkpoint's multi-token-prediction head (`mtp.*`; transformers ignores these tensors, the semantics are those of
    vLLM's Qwen3_5MultiTokenPredictor and llama.cpp's qwen35moe graph_mtp, which agree):

        z = fc(cat(rms(embed(x[p+1]), pre_fc_norm_embedding), rms(h[p], pre_fc_norm_hidden)))     (embedding FIRST)
        z = z + full_attention(rms(z, input_layernorm))          own KV cache, RoPE position p, causal over the MTP entries
        z = z + moe(rms(z, post_attention_layernorm))
        o = rms(z, mtp.norm);   logits = lm_head(o)  -> the draft for x[p+2]

    h[p] is the main model's hidden state of position p AFTER its final norm (what its lm_head reads); embed and lm_head
    are the main model's (mtp_use_dedicated_embeddings false). Drafting further tokens chains the same layer on its own
    output: h <- o, x <- the draft, position + 1 (the chained entries enter the MTP KV cache and are overwritten by the true
    (h, x) pairs once the main model has produced them). The MTP layer may have another expert count than the main layers
    (a pruned checkpoint keeps the original head): geometry from the tensors.

    weights: 'mtp.*' HF names -> tensors (experts per expert `experts.<e>.{gate,up,down}_proj.weight` or fused)."""

    def __init__(self, cfg, weights):
        g = Cfg(cfg) if not isinstance(cfg, Cfg) else cfg
        w = {k: v.float() for k, v in weights.items() if k.startswith('mtp.')}
        lp = 'mtp.layers.0.'
        inner = {'model.' + k[len('mtp.'):]: v for k, v in w.items() if '.mlp.experts.' not in k}
        if lp + 'mlp.experts.gate_up_proj' in w:
            inner['model.layers.0.mlp.experts.gate_up_proj'] = w[lp + 'mlp.experts.gate_up_proj']
            inner['model.layers.0.mlp.experts.down_proj'] = w[lp + 'mlp.experts.down_proj']
        else:
            E = w[lp + 'mlp.gate.weight'].shape[0]
            ex = lambda e, n: w[f'{lp}mlp.experts.{e}.{n}_proj.weight']
            inner['model.layers.0.mlp.experts.gate_up_proj'] = torch.stack([torch.cat((ex(e, 'gate'), ex(e, 'up'))) for e in range(E)])
            inner['model.layers.0.mlp.experts.down_proj'] = torch.stack([ex(e, 'down') for e in range(E)])
        raw = dict(g.raw)
        raw.update(num_hidden_layers=1, layer_types=['full_attention'], num_experts=inner['model.layers.0.mlp.gate.weight'].shape[0])
        self.ref = Ref(raw, inner)
        self.g = self.ref.g
        self.w = inner

    def init_state(self):
        return {'k': torch.zeros(0, self.g.kvh, self.g.D), 'v': torch.zeros(0, self.g.kvh, self.g.D)}

    def step(self, h, emb, pos, st, moe=None):
        """h [T, H] (main post-norm hidden, or the previous step's output), emb [T, H] embeddings of the NEXT tokens, pos [T]
        consecutive positions continuing st (all earlier MTP keys) -> o [T, H] (post mtp.norm; logits = o @ lm_head.T)."""
        g, t = self.g, self.ref.t
        cat = torch.cat((rms(emb, t('pre_fc_norm_embedding.weight'), g.eps), rms(h, t('pre_fc_norm_hidden.weight'), g.eps)), dim=-1)
        z = cat @ t('fc.weight').T
        z = z + self.ref.full_attn(0, rms(z, t('layers.0.input_layernorm.weight'), g.eps), torch.as_tensor(pos, dtype=torch.int64), st)
        z = z + (moe or self.ref.moe)(0, rms(z, t('layers.0.post_attention_layernorm.weight'), g.eps))
        return rms(z, t('norm.weight'), g.eps)


    def chains(self, o, pos, K, V, embed, head, steps):
        """Batched greedy draft chains (the offline acceptance study): row m continues from the true entry at position pos[m]
        whose output is o[m]; it sees the true MTP entries K / V [N, kvh, D] of positions <= pos[m] plus its own chained
        entries. -> drafts [M, steps + 1] (column 0 = argmax head(o): the draft for token pos + 2, column j for pos + 2 + j).
        Equal to step() on a copy of the state truncated after pos[m] (tests/test_qwen35_mtp_ref.py)."""
        g, t = self.g, self.ref.t
        a = 'layers.0.self_attn.'
        M_, N = o.shape[0], K.shape[0]
        rep = g.heads // g.kvh
        Kr, Vr = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
        pos = torch.as_tensor(pos, dtype=torch.int64)
        vis = (torch.arange(N).reshape(1, 1, N) <= pos.reshape(M_, 1, 1))
        drafts = [(o @ head.T).argmax(-1)]
        h, cK, cV = o, [], []
        for j in range(1, steps + 1):
            cat = torch.cat((rms(embed[drafts[-1]].float(), t('pre_fc_norm_embedding.weight'), g.eps), rms(h, t('pre_fc_norm_hidden.weight'), g.eps)), dim=-1)
            z = cat @ t('fc.weight').T
            n = rms(z, t('layers.0.input_layernorm.weight'), g.eps)
            qg = (n @ t(a + 'q_proj.weight').T).reshape(M_, g.heads, 2 * g.D)
            q, gate = qg[..., :g.D], qg[..., g.D:]
            k = (n @ t(a + 'k_proj.weight').T).reshape(M_, g.kvh, g.D)
            v = (n @ t(a + 'v_proj.weight').T).reshape(M_, g.kvh, g.D)
            cos, sin = rope_cos_sin(pos + j, g.rot, g.theta)
            q = apply_rope(rms(q, t(a + 'q_norm.weight'), g.eps), cos, sin)
            k = apply_rope(rms(k, t(a + 'k_norm.weight'), g.eps), cos, sin)
            cK.append(k.repeat_interleave(rep, dim=1))
            cV.append(v.repeat_interleave(rep, dim=1))
            ck, cv = torch.stack(cK, dim=1), torch.stack(cV, dim=1)                       # [M, j, heads, D]
            s_true = torch.einsum('mhd,nhd->mhn', q, Kr) / math.sqrt(g.D)
            s_true = torch.where(vis, s_true, torch.full_like(s_true, -float('inf')))
            s_ch = torch.einsum('mhd,mjhd->mhj', q, ck) / math.sqrt(g.D)
            pr = torch.softmax(torch.cat((s_true, s_ch), dim=-1), dim=-1)
            att = torch.einsum('mhn,nhd->mhd', pr[..., :N], Vr) + torch.einsum('mhj,mjhd->mhd', pr[..., N:], cv)
            z = z + (att * torch.sigmoid(gate)).reshape(M_, g.heads * g.D) @ t(a + 'o_proj.weight').T
            z = z + self.ref.moe(0, rms(z, t('layers.0.post_attention_layernorm.weight'), g.eps))
            h = rms(z, t('norm.weight'), g.eps)
            drafts.append((h @ head.T).argmax(-1))
        return torch.stack(drafts, dim=1)


def random_mtp_weights(cfg, seed=7, scale=0.02, experts=None, fused=False):
    """Random `mtp.*` weights of the checkpoint's shapes (per-expert tensors unless fused)."""
    raw = dict(cfg.get('text_config', cfg))
    raw.update(num_hidden_layers=1, layer_types=['full_attention'])
    if experts:
        raw['num_experts'] = experts
    g = Cfg(raw)
    base = random_weights(raw, seed=seed, scale=scale, prefix='mtp.')
    gen = torch.Generator().manual_seed(seed + 1)
    w = {k: v for k, v in base.items() if k.startswith('mtp.layers.') or k == 'mtp.norm.weight'}
    w['mtp.fc.weight'] = torch.randn(g.H, 2 * g.H, generator=gen) * (2 * g.H) ** -0.5
    w['mtp.pre_fc_norm_embedding.weight'] = torch.randn(g.H, generator=gen) * scale
    w['mtp.pre_fc_norm_hidden.weight'] = torch.randn(g.H, generator=gen) * scale
    if not fused:
        gu, dn = w.pop('mtp.layers.0.mlp.experts.gate_up_proj'), w.pop('mtp.layers.0.mlp.experts.down_proj')
        for e in range(g.E):
            w[f'mtp.layers.0.mlp.experts.{e}.gate_proj.weight'] = gu[e, :g.I].clone()
            w[f'mtp.layers.0.mlp.experts.{e}.up_proj.weight'] = gu[e, g.I:].clone()
            w[f'mtp.layers.0.mlp.experts.{e}.down_proj.weight'] = dn[e].clone()
    return w


def random_weights(cfg, seed=0, scale=0.02, prefix='model.language_model.'):
    """Random f32 weights of the real HF shapes (expert tensors fused, norms zero-centred)."""
    g = Cfg(cfg)
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen) * scale
    w = {prefix + 'embed_tokens.weight': rn(g.V, g.H), 'lm_head.weight': rn(g.V, g.H), prefix + 'norm.weight': rn(g.H)}
    for l in range(g.L):
        p = prefix + f'layers.{l}.'
        w[p + 'input_layernorm.weight'] = rn(g.H)
        w[p + 'post_attention_layernorm.weight'] = rn(g.H)
        if g.is_linear(l):
            a = p + 'linear_attn.'
            w[a + 'in_proj_qkv.weight'] = rn(g.C, g.H)
            w[a + 'in_proj_z.weight'] = rn(g.Hv * g.Dv, g.H)
            w[a + 'in_proj_b.weight'] = rn(g.Hv, g.H)
            w[a + 'in_proj_a.weight'] = rn(g.Hv, g.H)
            w[a + 'conv1d.weight'] = rn(g.C, 1, g.K) * 10
            w[a + 'A_log'] = torch.log(torch.rand(g.Hv, generator=gen) * 16 + 1e-3)
            w[a + 'dt_bias'] = torch.rand(g.Hv, generator=gen)
            w[a + 'norm.weight'] = 1 + rn(g.Dv)
            w[a + 'out_proj.weight'] = rn(g.H, g.Hv * g.Dv)
        else:
            a = p + 'self_attn.'
            w[a + 'q_proj.weight'] = rn(g.heads * g.D * (2 if g.gate else 1), g.H)
            w[a + 'k_proj.weight'] = rn(g.kvh * g.D, g.H)
            w[a + 'v_proj.weight'] = rn(g.kvh * g.D, g.H)
            w[a + 'o_proj.weight'] = rn(g.H, g.heads * g.D)
            w[a + 'q_norm.weight'] = rn(g.D)
            w[a + 'k_norm.weight'] = rn(g.D)
        m = p + 'mlp.'
        w[m + 'gate.weight'] = rn(g.E, g.H) * 5
        w[m + 'experts.gate_up_proj'] = rn(g.E, 2 * g.I, g.H)
        w[m + 'experts.down_proj'] = rn(g.E, g.H, g.I)
        if g.SI:
            w[m + 'shared_expert.gate_proj.weight'] = rn(g.SI, g.H)
            w[m + 'shared_expert.up_proj.weight'] = rn(g.SI, g.H)
            w[m + 'shared_expert.down_proj.weight'] = rn(g.H, g.SI)
            w[m + 'shared_expert_gate.weight'] = rn(1, g.H)
    return w


TINY = {  # a few layers, 8 experts: the test configuration (same code path as the 35B geometry)
    'hidden_size': 64, 'num_hidden_layers': 4,
    'layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
    'rms_norm_eps': 1e-6, 'vocab_size': 100, 'hidden_act': 'silu',
    'linear_num_key_heads': 2, 'linear_key_head_dim': 16, 'linear_num_value_heads': 4, 'linear_value_head_dim': 16,
    'linear_conv_kernel_dim': 4, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32, 'attn_output_gate': True,
    'rope_parameters': {'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'rope_type': 'default', 'mrope_section': [11, 11, 10],
                        'mrope_interleaved': True},
    'num_experts': 8, 'num_experts_per_tok': 2, 'moe_intermediate_size': 32, 'shared_expert_intermediate_size': 32,
    'max_position_embeddings': 4096, 'attention_bias': False, 'attention_dropout': 0.0, 'tie_word_embeddings': False,
    'mtp_num_hidden_layers': 0, 'mamba_ssm_dtype': 'float32',
}

TINY397 = {  # the Qwen3.5-397B-A17B shape class scaled down: 32 q heads / 2 KV heads, 16 key / 64 value linear heads (ratio 4),
    # top-10 routing, expert and shared intermediates divisible by 32 -> shardable over TP = 32 with replicated KV / key heads
    **TINY, 'hidden_size': 128, 'num_hidden_layers': 4, 'vocab_size': 160,
    'linear_num_key_heads': 16, 'linear_key_head_dim': 8, 'linear_num_value_heads': 64, 'linear_value_head_dim': 8,
    'num_attention_heads': 32, 'num_key_value_heads': 2, 'head_dim': 16,
    'num_experts': 48, 'num_experts_per_tok': 10, 'moe_intermediate_size': 64, 'shared_expert_intermediate_size': 64,
}
