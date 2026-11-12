"""qwen35_moe_ref.Ref against transformers' Qwen3_5MoeForCausalLM on a tiny random-init config, f32:
prefill logits, then step-by-step decode with the state carried (HF: DynamicCache; ref: its own state dict).
Needs transformers >= 5.x with models.qwen3_5_moe (CPU only, no fla / causal_conv1d: HF falls back to its torch path).
usage: python neuron/tests/test_qwen35_ref.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_moe_ref as R

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

from transformers import Qwen3_5MoeForCausalLM  # noqa: E402
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402

# an installed fla / causal_conv1d would route to Triton GPU kernels (fails on CPU tensors): force HF's torch fallbacks
for name in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated'):
    setattr(M, name, None)
M.is_fast_path_available = False


def main():
    cfg = dict(R.TINY)
    hf_cfg = Qwen3_5MoeTextConfig(**cfg)
    hf_cfg._attn_implementation = 'eager'
    model = Qwen3_5MoeForCausalLM(hf_cfg).float().eval()
    # random weights of the real shapes into the HF module (its own init leaves norms at 0 = scale 1 and routers at 0)
    w = R.random_weights(cfg, seed=1, prefix='model.')
    sd = model.state_dict()
    missing = [k for k in sd if k not in w]
    extra = [k for k in w if k not in sd]
    assert not missing and not extra, (missing[:5], extra[:5])
    model.load_state_dict({k: v.reshape(sd[k].shape) for k, v in w.items()})
    ref = R.Ref(cfg, w)

    T = 11
    tokens = torch.randint(0, cfg['vocab_size'], (T,))
    with torch.no_grad():
        hf_pre = model(tokens.reshape(1, -1), use_cache=True)
        logits_hf = hf_pre.logits[0]
    st = ref.init_state()
    logits_ref = ref.forward(tokens, st)
    worst = 0.0
    d = (logits_ref - logits_hf).abs().max().item()
    rel = d / logits_hf.abs().max().item()
    print(f'prefill T={T}: max |diff| {d:.3e} (rel {rel:.2e}), top-1 agree {(logits_ref.argmax(-1) == logits_hf.argmax(-1)).float().mean():.2f}')
    worst = max(worst, rel)
    assert rel < 2e-5, rel

    # decode: 6 steps, HF with its cache, ref with the carried state; also a 2-token continuation chunk
    cache = hf_pre.past_key_values
    steps = torch.randint(0, cfg['vocab_size'], (6,))
    for i, tok in enumerate(steps):
        with torch.no_grad():
            out = model(tok.reshape(1, 1), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        lr = ref.forward(tok.reshape(1), st)
        d = (lr[0] - out.logits[0, 0]).abs().max().item()
        rel = d / out.logits.abs().max().item()
        worst = max(worst, rel)
        print(f'step {i}: max |diff| {d:.3e} (rel {rel:.2e})')
        assert rel < 2e-5, rel
    chunk = torch.randint(0, cfg['vocab_size'], (3,))
    with torch.no_grad():
        out = model(chunk.reshape(1, -1), past_key_values=cache, use_cache=True)
    lr = ref.forward(chunk, st)
    d = (lr - out.logits[0]).abs().max().item()
    rel = d / out.logits.abs().max().item()
    worst = max(worst, rel)
    print(f'continuation chunk of 3: max |diff| {d:.3e} (rel {rel:.2e})')
    assert rel < 2e-5, rel
    # state check: the recurrent / conv state after everything must match HF's cache
    for l in range(cfg['num_hidden_layers']):
        if cfg['layer_types'][l] == 'linear_attention':
            layer = cache.layers[l]
            ds = (layer.recurrent_states[0] - st[l]['S']).abs().max().item()
            hc = layer.conv_states[0].reshape(-1, layer.conv_states[0].shape[-1])[:, -(cfg['linear_conv_kernel_dim'] - 1):]
            dc = (hc - st[l]['conv']).abs().max().item()
            print(f'layer {l} state: recurrent max |diff| {ds:.3e}, conv {dc:.3e}')
            assert ds < 1e-5 and dc < 1e-5
    print(f'OK: worst relative logit difference {worst:.2e}')


if __name__ == '__main__':
    main()
