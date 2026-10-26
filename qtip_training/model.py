"""Checkpoint format, trainable parameterization, and recomputable decoder layer."""
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import sha256_file
from .kernels import attention, qtip_linear, rmsnorm, rope, norm_weight

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")
SU_GROUPS = (("self_attn.q_proj.SU", "self_attn.k_proj.SU", "self_attn.v_proj.SU"),
             ("mlp.gate_proj.SU", "mlp.up_proj.SU"))


def validate_shared_su(values, prefix=""):
    for group in SU_GROUPS:
        if any(not torch.equal(values[prefix + group[0]], values[prefix + name]) for name in group[1:]):
            raise ValueError(f"runtime requires shared SU for {prefix}{group}")


def checkpoint_identity(path):
    path = Path(path)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    digest = hashlib.sha256()
    for name in ["config.json", "model.safetensors.index.json", *sorted(set(weight_map.values()))]:
        if Path(name).name != name:
            raise ValueError("shard names must be local filenames")
        digest.update(name.encode())
        digest.update(sha256_file(path / name).encode())
    return digest.hexdigest()


def host_copy(tensor, pin=False):
    result = torch.empty_like(tensor, device="cpu", pin_memory=pin)
    result.copy_(tensor)
    return result


class HostModel:
    """Single shared, immutable packed model. GPUs own only the small parameters."""
    def __init__(self, path, pin=False):
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text())
        c = self.config
        if c.get("rope_scaling") or c.get("sliding_window") or c.get("attention_bias") or c.get("mlp_bias"):
            raise ValueError("trainer supports full causal attention, unscaled RoPE and bias-free linears")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError("only SwiGLU is supported")
        qp = c.get("quip_params", {})
        expected = dict(L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym", td_x=16, td_y=16)
        if any(qp.get(key) != value for key, value in expected.items()):
            raise ValueError("expected the 2-bit HYB kernel-layout QTIP checkpoint")
        self.weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        for shard in set(self.weight_map.values()):
            if Path(shard).name != shard:
                raise ValueError("shard names must be local filenames")
        self.pin = pin
        self.dtypes = {}
        self.layers, self.initial = [], []
        n = c["hidden_size"]
        nh, nk = c["num_attention_heads"], c["num_key_value_heads"]
        hd = c.get("head_dim", n // nh)
        if nh % nk or hd % 2:
            raise ValueError("invalid head geometry")
        inter = c["intermediate_size"]
        shapes = ((nh * hd, n), (nk * hd, n), (nk * hd, n), (n, nh * hd),
                  (inter, n), (inter, n), (n, inter))
        for layer in range(c["num_hidden_layers"]):
            prefix = f"model.layers.{layer}."
            frozen, initial = {}, {}
            for linear, (m, width) in zip(LINEARS, shapes):
                su = self.get(prefix + linear + ".SU").float()
                sv = self.get(prefix + linear + ".SV").float()
                if self.dtypes[prefix + linear + ".SU"] != torch.bfloat16 or self.dtypes[prefix + linear + ".SV"] != torch.float32:
                    raise ValueError("trainer expects qtip_quant's BF16 SU and FP32 SV storage")
                if su.shape != (width,) or sv.shape != (m,) or width % 32 or m % 32:
                    raise ValueError(f"invalid QTIP shape: {prefix + linear}")
                if not torch.isfinite(su).all() or not torch.isfinite(sv).all():
                    raise ValueError("nonfinite signs/scales")
                scale = sv.square().mean().sqrt()
                if scale <= 0:
                    raise ValueError("SV has zero scale")
                initial[linear + ".SU"] = su
                initial[linear + ".SV"] = sv / scale
                initial[linear + ".Wscale"] = scale.reshape(())
                for key in ("trellis", "tlut"):
                    frozen[linear + "." + key] = self.get(prefix + linear + "." + key, pinned=pin)
                codes, lut = frozen[linear + ".trellis"], frozen[linear + ".tlut"]
                if codes.dtype != torch.int16 or codes.shape != (m // 16 * width // 16, 32):
                    raise ValueError("invalid trellis dtype/shape")
                if lut.shape != (512, 2) or lut.dtype != torch.float16:
                    raise ValueError("invalid HYB LUT")
                if int(self.get(prefix + linear + ".rcp")) != 0:
                    raise ValueError("row/column permuted QTIP is not supported")
                if int(self.get(prefix + linear + ".tp_rank")) != 8:
                    raise ValueError("expected unsplit (tp_rank=8) checkpoint")
            for name in NORMS:
                weight = self.get(prefix + name).float()
                if weight.shape != (n,):
                    raise ValueError("invalid RMSNorm shape")
                initial[name] = weight
            validate_shared_su(initial, "")
            for group in SU_GROUPS:
                if len({self.dtypes[prefix + key] for key in group}) != 1:
                    raise ValueError("shared SU tensors must have matching export dtypes")
            first_lut = frozen[LINEARS[0] + ".tlut"]
            if any(not torch.equal(first_lut.view(torch.int16), frozen[name + ".tlut"].view(torch.int16))
                   for name in LINEARS[1:]):
                raise ValueError(f"runtime requires one shared tlut in layer {layer}")
            self.layers.append(frozen)
            self.initial.append(initial)
        self.embedding = self.get("model.embed_tokens.weight", pinned=pin)
        self.head = self.get("lm_head.weight", pinned=pin)
        self.final_norm = self.get("model.norm.weight").float()
        if self.embedding.shape != (c["vocab_size"], n) or self.head.shape != self.embedding.shape:
            raise ValueError("embedding/lm_head shape mismatch")
        if any(key.endswith(".bias") for key in self.weight_map):
            raise ValueError("checkpoint has unsupported bias tensors")
        # Bind resumes to actual source bytes, including trellises, not just config.
        self.identity = checkpoint_identity(self.path)

    def get(self, name, pinned=False):
        with safe_open(self.path / self.weight_map[name], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
            self.dtypes[name] = tensor.dtype
            return host_copy(tensor, pinned)


def decoder_layer(x, weights, params, config, segments, positions, backend, row_chunk=256, attn_block=256):
    n, heads, kv_heads = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]
    hd = config.get("head_dim", n // heads)
    eps = config.get("rms_norm_eps", 1e-5)

    def linear(value, name):
        return qtip_linear(value, params[name + ".SU"], params[name + ".SV"], params[name + ".Wscale"],
                           weights[name + ".trellis"], weights[name + ".tlut"], backend,
                           round_z=name in ("mlp.gate_proj", "mlp.up_proj"))

    # Row checkpoints bound saved pre-MLP/linear intermediates during recompute.
    # Avoid building any autograd graph at all during the streaming forward.
    def chunks(function, value):
        chunks = []
        for row in value.split(row_chunk):
            chunks.append(checkpoint(function, row, use_reentrant=False, preserve_rng_state=False)
                          if torch.is_grad_enabled() else function(row))
        return torch.cat(chunks, dim=0)

    def qkv(value):
        value = rmsnorm(value, norm_weight(params[NORMS[0]], backend), eps)
        return torch.cat([linear(value, "self_attn." + p + "_proj") for p in ("q", "k", "v")], -1)

    q, k, v = chunks(qkv, x).split((heads * hd, kv_heads * hd, kv_heads * hd), -1)
    q = q.reshape(-1, heads, hd).transpose(0, 1)
    k = k.reshape(-1, kv_heads, hd).transpose(0, 1)
    v = v.reshape(-1, kv_heads, hd).transpose(0, 1)
    q = rope(q, positions, config.get("rope_theta", 10000.0))
    k = rope(k, positions, config.get("rope_theta", 10000.0))
    a = attention(q, k, v, segments, attn_block).transpose(0, 1).reshape(-1, heads * hd)
    x = x + chunks(lambda value: linear(value, "self_attn.o_proj"), a)

    def mlp(value):
        value = rmsnorm(value, norm_weight(params[NORMS[1]], backend), eps)
        gate = linear(value, "mlp.gate_proj")
        up = linear(value, "mlp.up_proj")
        if backend.device_rounding:
            # Runtime output_gateup computes the product in FP32, with a single
            # BF16 rounding after SiLU * up (not after SiLU alone).
            product = (F.silu(gate.float()) * up.float()).to(gate.dtype)
        else:
            product = F.silu(gate) * up
        return linear(product, "mlp.down_proj")

    return x + chunks(mlp, x)
