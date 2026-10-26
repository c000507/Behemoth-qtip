"""Check qtip_quant's bf16 forward / Hessian collection against an independent fp32 torch implementation of the
Mistral decoder layer (dequantized FP8 weights, W = fp8 * scale), on CPU.
Run the binary first, e.g.
  qtip_quant --mode collect --layers 0:2 --ntok 4096 --gpus 0 --keep-hessians --dump-acts D --work W --out O --tokens T
then:  python check_forward.py --tokens T --nseq 2 --layers 2 --acts D --hess W/hess
Compares: activations of sequence 0 after each layer, and H (qkv, o, gateup, down) of each layer (relative Frobenius).
"""
import argparse, json, os
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic")
ap.add_argument("--tokens", required=True)
ap.add_argument("--seq_len", type=int, default=2048)
ap.add_argument("--nseq", type=int, default=2)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--acts", required=True)
ap.add_argument("--hess", required=True)
ap.add_argument("--threads", type=int, default=32)
args = ap.parse_args()
torch.set_num_threads(args.threads)
cfg = json.load(open(os.path.join(args.model, "config.json")))
wm = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(args.model, wm[name]), "pt") as f:
        return f.get_tensor(name)


def lin_w(pre):
    return get(pre + ".weight").float() * get(pre + ".weight_scale").float().reshape(-1, 1)


H, nq, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
T = args.seq_len
tok = torch.from_numpy(np.fromfile(args.tokens, dtype=np.int32)[: args.nseq * T].reshape(args.nseq, T).astype(np.int64))
x = get("model.embed_tokens.weight")[tok].float()  # [N, T, H]
inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]
cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)


def rms(v, w):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(q):
    q1, q2 = q[..., : hd // 2], q[..., hd // 2:]
    return q * cos + torch.cat([-q2, q1], -1) * sin


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


for L in range(args.layers):
    p = f"model.layers.{L}."
    Wq, Wk, Wv, Wo = (lin_w(p + "self_attn." + k) for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
    Wg, Wu, Wd = (lin_w(p + "mlp." + k) for k in ("gate_proj", "up_proj", "down_proj"))
    ln1, ln2 = get(p + "input_layernorm.weight").float(), get(p + "post_attention_layernorm.weight").float()
    Hs = {k: torch.zeros(d, d, dtype=torch.float64) for k, d in (("qkv", H), ("o", nq * hd), ("gateup", H),
                                                                   ("down", Wd.shape[1]))}
    with torch.no_grad():
        for i in range(args.nseq):
            h = rms(x[i], ln1)
            Hs["qkv"] += (h.T @ h).double()
            q = (h @ Wq.T).view(T, nq, hd).transpose(0, 1)
            k = (h @ Wk.T).view(T, nkv, hd).transpose(0, 1)
            v = (h @ Wv.T).view(T, nkv, hd).transpose(0, 1)
            q, k = rope(q), rope(k)
            k = k.repeat_interleave(nq // nkv, 0)
            v = v.repeat_interleave(nq // nkv, 0)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(0, 1).reshape(T, nq * hd)
            Hs["o"] += (a.T @ a).double()
            x[i] = x[i] + a @ Wo.T
            h = rms(x[i], ln2)
            Hs["gateup"] += (h.T @ h).double()
            act = torch.nn.functional.silu(h @ Wg.T) * (h @ Wu.T)
            Hs["down"] += (act.T @ act).double()
            x[i] = x[i] + act @ Wd.T
    raw = np.fromfile(os.path.join(args.acts, f"act_after_layer{L}.bf16"), dtype=np.uint16).astype(np.uint32) << 16
    mine = torch.from_numpy(raw.view(np.float32)).view(T, H)
    out = {"layer": L, "act_rel_err_seq0": rel(mine, x[0])}
    fh = os.path.join(args.hess, f"layer{L}.safetensors")
    if os.path.exists(fh):
        with safe_open(fh, "pt") as f:
            for k in Hs:
                ref = (Hs[k] / (args.nseq * T)).float()
                out[f"H_{k}_rel_err"] = rel(f.get_tensor(k), ref)
    print(json.dumps(out), flush=True)
