"""inkling_ref.Ref vs transformers' Inkling2Loop (PR 51050) in f32 on a tiny random checkpoint written in the ORIGINAL
checkpoint format (model.llm.* names, NVFP4 experts + MXFP8 experts in two layers), loaded by HF's own conversion.

Needs a transformers tree with models/inkling2_loop on PYTHONPATH (INKLING_TF=<.../src>), e.g.
  INKLING_TF=/path/transformers/src python neuron/tests/test_inkling_ref.py
Checks: prefill logits (T > sliding window, log-scaling active), step decode with HF's DynamicCache vs the carried state,
a multi-token continuation chunk.
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if os.environ.get('INKLING_TF'):
    sys.path.insert(0, os.environ['INKLING_TF'])

import inkling_ref as R  # noqa: E402

TINY = {
    'model_type': 'inkling2_loop_text', 'architectures': ['Inkling2LoopForCausalLM'],
    'hidden_size': 128, 'num_hidden_layers': 24, 'vocab_size': 320, 'unpadded_vocab_size': 300,
    'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32,
    'swa_num_attention_heads': 4, 'swa_num_key_value_heads': 2, 'swa_head_dim': 32,
    'd_rel': 8, 'rel_extent': 12, 'sliding_window_size': 8, 'log_scaling_n_floor': 6, 'log_scaling_alpha': 0.3,
    'rms_norm_eps': 1e-6, 'qk_l2_clip_threshold': 3.0, 'rel_l2_clip_threshold': 1.5,
    'local_layer_ids': [i for i in range(24) if (i + 1) % 6], 'sconv_kernel_size': 4,
    'final_logit_softcapping': 30.0, 'n_layers_per_attn_res_block': 6,
    'n_routed_experts': 8, 'num_experts_per_tok': 2, 'n_shared_experts': 0, 'intermediate_size': 32,
    'mlp_gate_softcap': 4.0, 'mlp_up_softcap': 25.0, 'route_scale': 8.0, 'use_gate_bias': True,
    'loop_config': {'core_start': 6, 'core_end': 18, 'num_passes': 3, 'gate_rank': 16, 'is_varloop': False},
    'eos_token_id': 1, 'pad_token_id': 0, 'torch_dtype': 'float32',
}


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    torch.manual_seed(0)
    from safetensors.torch import save_file
    from transformers import DynamicCache
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

    cfg = R.Cfg(TINY)
    t = R.random_checkpoint(cfg, seed=1, mxfp8_layers=[3, 20])
    d = tempfile.mkdtemp()
    save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(d, 'model.safetensors'), metadata={'format': 'pt'})
    json.dump(TINY, open(os.path.join(d, 'config.json'), 'w'))
    hf = Inkling2LoopForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    ref = R.from_tensors(cfg, t)

    # weight loading cross-check: the dequantised experts as HF holds them
    worst = 0.0
    for l in (0, 3, 20):
        gu, dn = ref.X(l)
        e = hf.model.layers[l].mlp.experts
        worst = max(worst, rel(gu, e.gate_up_proj.data), rel(dn, e.down_proj.data))
    print(f'experts dequant vs HF: {worst:.2e}')
    assert worst < 1e-6

    T0 = 20                                     # > window (8) and > rel extent (12); positions past n_floor (6)
    ids = torch.randint(2, cfg.Vu, (1, T0))
    with torch.no_grad():
        out = hf(input_ids=ids, use_cache=True)
    st = {}
    lr = ref.forward(ids[0], st)
    e0 = rel(lr, out.logits[0])
    print(f'prefill T={T0}: rel {e0:.2e}  top1 eq {(lr.argmax(-1) == out.logits[0].argmax(-1)).float().mean():.3f}')
    cache = out.past_key_values if hasattr(out, 'past_key_values') and out.past_key_values is not None else out.cache
    errs = []
    nxt = out.logits[0, -1].argmax().view(1, 1)
    for s in range(6):
        with torch.no_grad():
            o = hf(input_ids=nxt, cache=cache, use_cache=True)
        cache = o.past_key_values if getattr(o, 'past_key_values', None) is not None else o.cache
        lr = ref.forward(nxt[0], st)
        errs.append(rel(lr, o.logits[0]))
        nxt = o.logits[0, -1].argmax().view(1, 1)
    print('decode steps rel:', ' '.join(f'{x:.2e}' for x in errs))
    chunk = torch.randint(2, cfg.Vu, (1, 5))
    with torch.no_grad():
        o = hf(input_ids=chunk, cache=cache, use_cache=True)
    lr = ref.forward(chunk[0], st)
    e2 = rel(lr, o.logits[0])
    print(f'continuation chunk of 5: rel {e2:.2e}')
    # the same sequence in one prefill
    full = torch.cat([ids[0], torch.tensor([0] * 0, dtype=torch.long)])
    worst = max([e0, e2] + errs)
    print(f'WORST {worst:.2e}')
    assert worst < 1e-4, worst


if __name__ == '__main__':
    main()
