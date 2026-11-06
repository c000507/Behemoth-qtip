"""Serial single-sequence Llama correctness runner using NKI primitives.

This is deliberately not an optimized model server: weights and KV cache live
in host memory, and each kernel is dispatched separately. Use simulate=True for
CPU validation; device mode requires the matching Neuron SDK/runtime/device.
"""
from functools import lru_cache
import json
import numpy as np
from backend import KernelBackend, bf16
from checkpoint import Checkpoint
from ops import rms_norm, swiglu, apply_rope, fp8_project_loop


def rope_frequencies(dim, theta, scaling):
    freq = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    if scaling:
        kind = scaling.get('rope_type', scaling.get('type'))
        if kind != 'llama3':
            raise ValueError(f'Unsupported RoPE scaling: {kind}')
        factor = float(scaling['factor'])
        low, high = float(scaling['low_freq_factor']), float(scaling['high_freq_factor'])
        original = float(scaling['original_max_position_embeddings'])
        if factor <= 0 or low <= 0 or high <= low or original <= 0:
            raise ValueError('Invalid Llama-3 RoPE parameters')
        wave = 2 * np.pi / freq
        smooth = (original / wave - low) / (high - low)
        multiplier = np.where(wave < original / high, 1.,
                     np.where(wave > original / low, 1. / factor,
                              (1. - smooth) / factor + smooth))
        freq *= multiplier
    return freq.astype(np.float32)


class LlamaRunner:
    def __init__(self, path, simulate=True):
        self.checkpoint = Checkpoint(path)
        c = self.checkpoint.config
        generation_path = self.checkpoint.root / 'generation_config.json'
        self.generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        if c.get('model_type') != 'llama' or c.get('hidden_act', 'silu') != 'silu':
            raise ValueError('Only Llama with SwiGLU is supported')
        if c.get('attention_bias', False) or c.get('mlp_bias', False):
            raise ValueError('Biased projections are not supported')
        self.hidden = int(c['hidden_size']); self.intermediate = int(c['intermediate_size'])
        self.heads = int(c['num_attention_heads']); self.kvheads = int(c['num_key_value_heads'])
        self.dim = int(c.get('head_dim', self.hidden // self.heads))
        self.layers = int(c['num_hidden_layers']); self.vocab = int(c['vocab_size'])
        self.max_context = int(c.get('max_position_embeddings', 131072))
        if self.heads <= 0 or self.kvheads <= 0 or self.heads % self.kvheads or self.dim <= 0 or self.dim > 128 or self.dim % 2:
            raise ValueError('Unsupported GQA dimensions')
        self.eps = float(c.get('rms_norm_eps', 1e-5))
        self.freq = rope_frequencies(self.dim, float(c.get('rope_theta', 10000)), c.get('rope_scaling'))
        self.backend = KernelBackend(simulate)
        self.embed = self.checkpoint.tensor('model.embed_tokens.weight').astype(np.float32)
        self.final_norm = self.checkpoint.tensor('model.norm.weight').astype(np.float32).reshape(1, -1)
        self.head = self.checkpoint.tensor('lm_head.weight')
        if self.embed.shape != (self.vocab, self.hidden) or self.head.shape != self.embed.shape or self.final_norm.shape != (1, self.hidden):
            raise ValueError('Invalid embedding, final norm or output head')
        self.reset()

    def reset(self):
        self.position = 0
        self.keys = [[] for _ in range(self.layers)]
        self.values = [[] for _ in range(self.layers)]

    @lru_cache(maxsize=1)
    def layer(self, layer):
        p = f'model.layers.{layer}.'
        c = self.checkpoint
        h, f, q, kv = self.hidden, self.intermediate, self.heads*self.dim, self.kvheads*self.dim
        shapes = {'self_attn.q_proj': (h, q), 'self_attn.k_proj': (h, kv),
                  'self_attn.v_proj': (h, kv), 'self_attn.o_proj': (q, h),
                  'mlp.gate_proj': (h, f), 'mlp.up_proj': (h, f), 'mlp.down_proj': (f, h)}
        projections = {name: c.projection(layer, name, *shape) for name, shape in shapes.items()}
        norms = [c.tensor(p + name).astype(np.float32).reshape(1, -1)
                 for name in ('input_layernorm.weight', 'post_attention_layernorm.weight')]
        if any(x.shape != (1, h) for x in norms):
            raise ValueError('Invalid layer norm weights')
        return projections, norms

    def forward_token(self, token):
        if not 0 <= token < self.vocab or self.position >= self.max_context:
            raise ValueError('Token or context position out of range')
        lengths = [len(cache) for cache in self.keys]
        try:
            return self._forward_token(token)
        except Exception:
            for layer, length in enumerate(lengths):
                del self.keys[layer][length:]
                del self.values[layer][length:]
            raise

    def _forward_token(self, token):
        state = bf16(self.embed[token:token+1])
        angle = self.freq * self.position
        for layer in range(self.layers):
            proj, norms = self.layer(layer)
            run = self.backend.run
            normalized = bf16(run(rms_norm, state, norms[0], self.eps))
            q = bf16(self.backend.linear(proj['self_attn.q_proj'], normalized)).reshape(self.heads, self.dim)
            k = bf16(self.backend.linear(proj['self_attn.k_proj'], normalized)).reshape(self.kvheads, self.dim)
            v = bf16(self.backend.linear(proj['self_attn.v_proj'], normalized)).reshape(self.kvheads, self.dim)
            def rotary(x):
                cos = np.broadcast_to(np.cos(angle).astype(np.float32), (len(x), self.dim // 2)).copy()
                sin = np.broadcast_to(np.sin(angle).astype(np.float32), (len(x), self.dim // 2)).copy()
                return bf16(run(apply_rope, x, cos, sin))
            q, k = rotary(q), rotary(k)
            self.keys[layer].append(k.copy()); self.values[layer].append(v.copy())
            attn = bf16(self.backend.attention(q, np.stack(self.keys[layer]), np.stack(self.values[layer])).reshape(1, -1))
            update = bf16(self.backend.linear(proj['self_attn.o_proj'], attn))
            state = bf16(state + update)
            normalized = bf16(run(rms_norm, state, norms[1], self.eps))
            gate = bf16(self.backend.linear(proj['mlp.gate_proj'], normalized, round_z_bf16=True))
            up = bf16(self.backend.linear(proj['mlp.up_proj'], normalized, round_z_bf16=True))
            act = bf16(run(swiglu, gate, up))
            state = bf16(state + bf16(self.backend.linear(proj['mlp.down_proj'], act)))
        normalized = bf16(self.backend.run(rms_norm, state, self.final_norm, self.eps))
        # The output head may be BF16; the tiled projection accepts either BF16 or FP8.
        scores = self.backend.run(fp8_project_loop, self.head.reshape(1, -1),
                                  np.ones((1, self.vocab), np.float32), normalized.reshape(1, -1),
                                  self.hidden, self.vocab, 1).reshape(-1)
        self.position += 1
        return scores

    def prefill(self, token_ids):
        logits = None
        for token in token_ids:
            logits = self.forward_token(int(token))
        if logits is None:
            raise ValueError('Prefill requires at least one token')
        return logits
