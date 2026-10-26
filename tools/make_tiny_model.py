"""Write a tiny random Mistral-style FP8-Dynamic checkpoint (same tensor names/dtypes as the Behemoth one) plus
random calibration tokens, to exercise the whole qtip_quant pipeline (collection, scheduling, resume, output,
reference decode) on a few MB of VRAM."""
import argparse, json, os
import numpy as np
import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--inter", type=int, default=512)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--kv_heads", type=int, default=2)
ap.add_argument("--head_dim", type=int, default=64)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--vocab", type=int, default=1000)
ap.add_argument("--nseq", type=int, default=16)
ap.add_argument("--seqlen", type=int, default=256)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
g = torch.Generator().manual_seed(0)
H, F = args.hidden, args.inter
cfg = {"architectures": ["MistralForCausalLM"], "model_type": "mistral", "hidden_size": H, "intermediate_size": F,
       "num_attention_heads": args.heads, "num_key_value_heads": args.kv_heads, "head_dim": args.head_dim,
       "num_hidden_layers": args.layers, "vocab_size": args.vocab, "rms_norm_eps": 1e-5, "rope_theta": 1e6,
       "max_position_embeddings": 4096, "hidden_act": "silu", "tie_word_embeddings": False, "dtype": "bfloat16",
       "bos_token_id": 1, "eos_token_id": 2, "sliding_window": None,
       "quantization_config": {"quant_method": "compressed-tensors"}}
json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
t = {}


def fp8(name, out, inp, std):
    w = torch.randn(out, inp, generator=g) * std * (1 + torch.rand(out, 1, generator=g))
    s = w.abs().amax(1, keepdim=True) / 448.0
    t[name + ".weight"] = (w / s).to(torch.float8_e4m3fn)
    t[name + ".weight_scale"] = s.to(torch.bfloat16)


qd, kd = args.heads * args.head_dim, args.kv_heads * args.head_dim
for L in range(args.layers):
    p = f"model.layers.{L}."
    fp8(p + "self_attn.q_proj", qd, H, 0.06)
    fp8(p + "self_attn.k_proj", kd, H, 0.06)
    fp8(p + "self_attn.v_proj", kd, H, 0.06)
    fp8(p + "self_attn.o_proj", H, qd, 0.06)
    fp8(p + "mlp.gate_proj", F, H, 0.06)
    fp8(p + "mlp.up_proj", F, H, 0.06)
    fp8(p + "mlp.down_proj", H, F, 0.04)
    t[p + "input_layernorm.weight"] = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    t[p + "post_attention_layernorm.weight"] = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
t["model.embed_tokens.weight"] = (torch.randn(args.vocab, H, generator=g) * 0.5).to(torch.bfloat16)
t["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
t["lm_head.weight"] = (torch.randn(args.vocab, H, generator=g) * 0.05).to(torch.bfloat16)
save_file(t, os.path.join(args.out, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in t}},
          open(os.path.join(args.out, "model.safetensors.index.json"), "w"))
tok = torch.randint(3, args.vocab, (args.nseq, args.seqlen), generator=g).to(torch.int32).numpy()
tok.tofile(os.path.join(args.out, "tokens.i32"))
print("ok", args.out)
