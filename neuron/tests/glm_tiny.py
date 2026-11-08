"""Tiny random GLM-5.3 (glm5_next) checkpoints and the glm53 C++ oracle, for the Neuron GLM graph tests.

    make_checkpoint(dir, cfg)               writes config.json + model.safetensors(.index.json), F32, seeded
    oracle_logits(binary, dir, tokens, positions, arithmetic='bf16')  -> [len(positions), vocab] via `glm53 trace`
    tiny_config(moe=True, **overrides)      a reduced geometry with every GLM feature active at short contexts
                                            (index_topk 16 -> the sparse top-k selection engages past 16 keys)
"""
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def tiny_config(moe=True, **o):
    layers = o.pop('layers', 4)
    c = {
        'model_type': 'glm5_next_text', 'hidden_size': 256, 'intermediate_size': 512, 'num_hidden_layers': layers,
        'vocab_size': 300, 'num_attention_heads': 4, 'num_key_value_heads': 4, 'q_lora_rank': 64, 'kv_lora_rank': 64,
        'qk_nope_head_dim': 32, 'qk_rope_head_dim': 0, 'v_head_dim': 32, 'mla_use_nope': True,
        'linear_attn_config': {'num_heads': 4, 'head_dim': 32, 'short_conv_kernel_size': 4, 'gate_lower_bound': -5.0},
        'hc_mult': 4, 'hc_sinkhorn_iters': 20, 'hc_eps': 1e-6, 'rms_norm_eps': 1e-5, 'mhc': True,
        'index_n_heads': 2, 'index_head_dim': 32, 'index_kpool': 4, 'index_topk': 16, 'index_kpool_always_select_tail': True,
        'index_kpool_compress': True,
        'layer_types': (['linear_attention', 'deepseek_sparse_attention'] * layers)[:layers],
        'mlp_layer_types': (['dense', 'sparse'] * layers)[:layers] if moe else ['dense'] * layers,
        'indexer_types': ['full'] * layers,
        'n_routed_experts': 8, 'num_experts_per_tok': 2, 'moe_intermediate_size': 64, 'n_shared_experts': 1,
        'n_group': 2, 'topk_group': 1, 'norm_topk_prob': True, 'routed_scaling_factor': 2.5, 'scoring_func': 'sigmoid',
        'topk_method': 'noaux_tc', 'max_position_embeddings': 512, 'eos_token_id': [298, 299], 'bos_token_id': 1,
        'swiglu_limit': 10.0, 'hidden_act': 'silu', 'attention_bias': False,
    }
    c.update(o)
    return {'model_type': 'glm5_next', 'text_config': c}


def write_safetensors(path, tensors):
    header, off, blobs = {}, 0, []
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = a.tobytes()
        header[name] = {'dtype': 'F32', 'shape': list(a.shape), 'data_offsets': [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b' ' * ((-len(h)) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def make_checkpoint(path, cfg, seed=7):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(json.dumps(cfg))
    c = cfg['text_config']
    rng = np.random.default_rng(seed)
    T = {}
    H = c['hidden_size']

    def w(name, out, inp, gain=1.0):
        T[name] = rng.standard_normal((out, inp)).astype(np.float32) * (gain / np.sqrt(inp))

    def v(name, shape, fill=None, std=0.1):
        T[name] = np.full(shape, fill, np.float32) if fill is not None else rng.standard_normal(shape).astype(np.float32) * std
    pre = 'model.language_model.'
    v(pre + 'embed_tokens.weight', (c['vocab_size'], H), std=0.5)
    v(pre + 'norm.weight', (H,), std=0.1)
    T[pre + 'norm.weight'] += 1
    w('lm_head.weight', c['vocab_size'], H, 2.0)
    la = c['linear_attn_config']
    S = c['hc_mult']
    for l in range(c['num_hidden_layers']):
        p = pre + f'layers.{l}.'
        for hc in ('hc_attn_', 'hc_ffn_'):
            w(p + hc + 'fn', S * (S + 2), S * H, 1.0)
            v(p + hc + 'base', (S * (S + 2),), std=0.3)
            v(p + hc + 'scale', (3,), std=0.05)
            T[p + hc + 'scale'] += 0.1
        v(p + 'input_layernorm.weight', (H,), 1.0)
        v(p + 'post_attention_layernorm.weight', (H,), 1.0)
        a = p + 'self_attn.'
        if c['layer_types'][l] == 'linear_attention':
            Wd = la['num_heads'] * la['head_dim']
            for k in 'qkv':
                w(a + f'{k}_proj.weight', Wd, H)
                v(a + f'{k}_conv1d.weight', (Wd, 1, la['short_conv_kernel_size']), std=0.5)
            w(a + 'f_a_proj.weight', la['head_dim'], H)
            w(a + 'f_b_proj.weight', Wd, la['head_dim'])
            w(a + 'b_proj.weight', la['num_heads'], H)
            w(a + 'g_a_proj.weight', la['head_dim'], H)
            w(a + 'g_b_proj.weight', Wd, la['head_dim'])
            w(a + 'o_proj.weight', H, Wd)
            v(a + 'A_log', (la['num_heads'],), std=0.5)
            v(a + 'dt_bias', (Wd,), std=0.5)
            v(a + 'o_norm.weight', (la['head_dim'],), 1.0)
        else:
            heads, qd, vd = c['num_attention_heads'], c['qk_nope_head_dim'], c['v_head_dim']
            w(a + 'q_a_proj.weight', c['q_lora_rank'], H)
            v(a + 'q_a_layernorm.weight', (c['q_lora_rank'],), 1.0)
            w(a + 'q_b_proj.weight', heads * qd, c['q_lora_rank'])
            w(a + 'kv_a_proj_with_mqa.weight', c['kv_lora_rank'], H)
            v(a + 'kv_a_layernorm.weight', (c['kv_lora_rank'],), 1.0)
            w(a + 'kv_b_proj.weight', heads * (qd + vd), c['kv_lora_rank'])
            w(a + 'o_proj.weight', H, heads * vd)
            ix = a + 'indexer.'
            w(ix + 'wq_b.weight', c['index_n_heads'] * c['index_head_dim'], c['q_lora_rank'])
            w(ix + 'wk.weight', c['index_head_dim'], H)
            v(ix + 'k_norm.weight', (c['index_head_dim'],), 1.0)
            v(ix + 'k_norm.bias', (c['index_head_dim'],), std=0.1)
            w(ix + 'weights_proj.weight', c['index_n_heads'], H)
            w(ix + 'index_kpool_compress_gate', c['index_head_dim'], H)
            v(ix + 'index_kpool_compress_ape', (c['index_kpool'], c['index_head_dim']), std=0.5)
        m = p + 'mlp.'

        def mlp(base, width):
            w(base + 'gate_proj.weight', width, H)
            w(base + 'up_proj.weight', width, H)
            w(base + 'down_proj.weight', H, width)
        if c['mlp_layer_types'][l] == 'dense':
            mlp(m, c['intermediate_size'])
        else:
            w(m + 'gate.weight', c['n_routed_experts'], H, 2.0)
            v(m + 'gate.e_score_correction_bias', (c['n_routed_experts'],), std=0.1)
            mlp(m + 'shared_experts.', c['moe_intermediate_size'] * c['n_shared_experts'])
            for e in range(c['n_routed_experts']):
                mlp(m + f'experts.{e}.', c['moe_intermediate_size'])
    write_safetensors(path / 'model.safetensors', T)
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in T}}))
    return path


def oracle_logits(binary, model, tokens, positions, arithmetic='bf16'):
    with tempfile.TemporaryDirectory() as d:
        req, out = Path(d) / 'req.json', Path(d) / 'trace.json'
        req.write_text(json.dumps({'tokens': list(tokens), 'logit_positions': list(positions)}))
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '8'))
        subprocess.run([binary, 'trace', '--model', str(model), '--tokens', str(req), '--out', str(out), '--arithmetic', arithmetic],
                       check=True, stderr=subprocess.DEVNULL, env=env)
        return np.array(json.loads(out.read_text())['logits'], dtype=np.float32)
