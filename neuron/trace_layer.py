"""Single transformer layer traced to one NEFF with resident weights.

First experiment of the graph-compiled route: a full Llama layer (RMSNorm,
QKV, RoPE, cached GQA attention, o_proj, SwiGLU MLP) runs as one traced
torch-neuronx graph. Weights are traced parameters (uploaded once, device
resident); the KV cache is a graph input/output cycled between decode steps so
activations never leave the device. FP8 and NVFP4 weights are dequantized on
the host from real checkpoint tensors; QTIP projections use random bf16
matrices unless decoded matrices are supplied (timing does not depend on the
weight values; correctness passes come later against the C++ oracle).

Attention is an unmasked softmax over the fixed context: this experiment
measures the compiled-graph execution, not generation quality. Timing mode
runs N decode steps after warmup and reports steady-state ms/step.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import ml_dtypes
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint

HEADS, KVHEADS, DIM, HIDDEN, INTERMEDIATE = 32, 8, 128, 4096, 14336


def dequant_nvfp4(projection):
    positive = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    lut = np.concatenate((positive, -positive))
    codes = np.empty((projection.packed.shape[0], projection.width), np.uint8)
    codes[:, 0::2] = projection.packed & 15
    codes[:, 1::2] = projection.packed >> 4
    dense = (lut[codes]
             * np.repeat(projection.scales.astype(np.float32), 16, axis=1)
             * projection.global_scale)
    return torch.from_numpy(dense.astype(ml_dtypes.bfloat16).astype(np.float32)).to(torch.bfloat16)


def dequant_fp8(projection):
    dense = projection.weights.astype(np.float32) * projection.scale[:, None]
    return torch.from_numpy(dense.astype(ml_dtypes.bfloat16).astype(np.float32)).to(torch.bfloat16)


def bf16(array):
    return torch.from_numpy(np.asarray(array, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)).to(torch.bfloat16)


def build_layer(model, layer, seed=5):
    cp = Checkpoint(model)
    rng = np.random.default_rng(seed)

    def projection(name, width, out):
        return cp.projection(layer, name, width, out)

    kv = projection('self_attn.k_proj', HIDDEN, KVHEADS * DIM)
    v = projection('self_attn.v_proj', HIDDEN, KVHEADS * DIM)
    weights = {
        'norm1': torch.from_numpy(cp.tensor(f'model.layers.{layer}.input_layernorm.weight').astype(np.float32).reshape(1, -1)),
        'norm2': torch.from_numpy(cp.tensor(f'model.layers.{layer}.post_attention_layernorm.weight').astype(np.float32).reshape(1, -1)),
        'kw': dequant_fp8(kv),
        'vw': dequant_fp8(v),
        'ow': bf16(rng.normal(0, .02, (HIDDEN, HIDDEN))),
        'gw': dequant_nvfp4(projection('mlp.gate_proj', HIDDEN, INTERMEDIATE)),
        'uw': dequant_nvfp4(projection('mlp.up_proj', HIDDEN, INTERMEDIATE)),
        'dw': dequant_nvfp4(projection('mlp.down_proj', INTERMEDIATE, HIDDEN)),
    }
    q = projection('self_attn.q_proj', HIDDEN, HIDDEN)
    weights['qw'] = dequant_nvfp4(q) if hasattr(q, 'packed') else bf16(rng.normal(0, .02, (HIDDEN, HIDDEN)))
    return weights


def layer_graph(x, slots, cos_sin, cache_k, cache_v,
                norm1, norm2, qw, kw, vw, ow, gw, uw, dw):
    scale = DIM ** -0.5
    half = DIM // 2
    h = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * norm1
    h = h.to(torch.bfloat16)
    q = (h @ qw.T).float().reshape(1, HEADS, DIM)
    k = (h @ kw.T).float().reshape(1, KVHEADS, DIM)
    v = (h @ vw.T).float().reshape(1, KVHEADS, DIM)
    row = torch.index_select(cos_sin, 0, slots)  # [B, 2, half]
    def rope(t, table, heads):
        cos = table[:, 0].reshape(1, 1, half).expand(1, heads, half)
        sin = table[:, 1].reshape(1, 1, half).expand(1, heads, half)
        a, b = t[..., :half], t[..., half:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    q = rope(q, row, HEADS)
    k = rope(k, row, KVHEADS)
    cache_k = cache_k.index_copy(1, slots, k.reshape(1, 1, KVHEADS, DIM))
    cache_v = cache_v.index_copy(1, slots, v.reshape(1, 1, KVHEADS, DIM))
    heads_out = []
    for head in range(HEADS):
        kh = head // (HEADS // KVHEADS)
        keys = cache_k[0, :, kh]      # [context, dim]
        values = cache_v[0, :, kh]
        scores = (q[0, head] @ keys.T) * scale
        weights = torch.softmax(scores, dim=-1)
        heads_out.append(weights @ values)
    attn = torch.stack(heads_out).reshape(1, HEADS, DIM)
    attn = attn.reshape(1, HEADS * DIM).to(torch.bfloat16)
    x = x + (attn @ ow.T).float()
    h = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * norm2
    h = h.to(torch.bfloat16)
    gate = h @ gw.T
    act = torch.nn.functional.silu(gate).float() * (h @ uw.T).float()
    x = x + (act.to(torch.bfloat16) @ dw.T).float()
    return x, cache_k, cache_v


def e4m3_to_f32(codes):
    """E4M3FN byte -> float32 with integer/float arithmetic only (no gather, no bitcast).
    Normals are (8 + m) * 2^(e - 10) with the power of two from an integer shift; subnormals
    (e == 0) are m * 2^-9. The NaN code (0x7f/0xff) is not produced by the quantizer."""
    b = codes.to(torch.int32)
    e = (b >> 3) & 15
    m = b & 7
    normal = (8 + m).to(torch.float32) * (1 << e).to(torch.float32) * (2.0 ** -10)
    value = torch.where(e == 0, m.to(torch.float32) * (2.0 ** -9), normal)
    return value * (1.0 - 2.0 * ((b >> 7) & 1).to(torch.float32))


def e2m1_to_f32(codes):
    """E2M1 nibble (0..15, bit 3 = sign) -> float32: normals (2 + m) * 2^(e - 2), subnormal m / 2."""
    b = codes.to(torch.int32)
    e = (b >> 1) & 3
    m = b & 1
    normal = (2 + m).to(torch.float32) * (1 << e).to(torch.float32) * 0.25
    value = torch.where(e == 0, m.to(torch.float32) * 0.5, normal)
    return value * (1.0 - 2.0 * ((b >> 3) & 1).to(torch.float32))


def dec_fp8(codes, scale):
    """codes uint8 [out, in], scale f32 [out] -> bf16 [out, in]."""
    return (e4m3_to_f32(codes) * scale.reshape(-1, 1)).to(torch.bfloat16)


def dec_nvfp4(packed, scales, gscale):
    """packed uint8 [out, in/2] (two E2M1 per byte, low nibble first), scales uint8 E4M3
    [out, in/16], gscale f32 [1] -> bf16 [out, in]."""
    out = packed.shape[0]
    p = packed.to(torch.int32)
    codes = torch.stack((p & 15, p >> 4), dim=-1).reshape(out, -1)
    block = (e4m3_to_f32(scales) * gscale).repeat_interleave(16, dim=1)
    return (e2m1_to_f32(codes) * block).to(torch.bfloat16)


def layer_graph_lowbit(x, slots, cos_sin, cache_k, cache_v, norm1, norm2, qw, ow,
                       kb, ks, vb, vs, gp, gs, gg, up, us, ug, dp, ds, dg):
    """Same layer, but projections stay as packed bytes in device memory and are decoded
    inside the graph each step (FP8 K/V and NVFP4 MLP via integer bit arithmetic, no table gathers)."""
    kw = dec_fp8(kb, ks)
    vw = dec_fp8(vb, vs)
    gw = dec_nvfp4(gp, gs, gg)
    uw = dec_nvfp4(up, us, ug)
    dw = dec_nvfp4(dp, ds, dg)
    return layer_graph(x, slots, cos_sin, cache_k, cache_v, norm1, norm2, qw, kw, vw, ow, gw, uw, dw)


def build_layer_lowbit(model, layer, seed=5):
    """Raw packed tensors for the low-bit graph (uint8 views; no host dequantization)."""
    cp = Checkpoint(model)
    rng = np.random.default_rng(seed)
    u8 = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.uint8))
    f32 = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    kv = cp.projection(layer, 'self_attn.k_proj', HIDDEN, KVHEADS * DIM)
    vv = cp.projection(layer, 'self_attn.v_proj', HIDDEN, KVHEADS * DIM)
    nv = {}
    for tag, name, w, o in (('g', 'mlp.gate_proj', HIDDEN, INTERMEDIATE), ('u', 'mlp.up_proj', HIDDEN, INTERMEDIATE),
                            ('d', 'mlp.down_proj', INTERMEDIATE, HIDDEN)):
        pr = cp.projection(layer, name, w, o)
        if not hasattr(pr, 'packed'):
            raise SystemExit(f'layer {layer} {name} is not NVFP4; pick a layer from 2-7,9,10')
        nv[tag] = (u8(pr.packed), u8(pr.scales), f32([pr.global_scale]))
    q = cp.projection(layer, 'self_attn.q_proj', HIDDEN, HIDDEN)
    weights = {
        'norm1': torch.from_numpy(cp.tensor(f'model.layers.{layer}.input_layernorm.weight').astype(np.float32).reshape(1, -1)),
        'norm2': torch.from_numpy(cp.tensor(f'model.layers.{layer}.post_attention_layernorm.weight').astype(np.float32).reshape(1, -1)),
        'qw': dequant_nvfp4(q) if hasattr(q, 'packed') else bf16(rng.normal(0, .02, (HIDDEN, HIDDEN))),
        'ow': bf16(rng.normal(0, .02, (HIDDEN, HIDDEN))),
        'kb': u8(kv.weights), 'ks': f32(kv.scale), 'vb': u8(vv.weights), 'vs': f32(vv.scale),
        'gp': nv['g'][0], 'gs': nv['g'][1], 'gg': nv['g'][2],
        'up': nv['u'][0], 'us': nv['u'][1], 'ug': nv['u'][2],
        'dp': nv['d'][0], 'ds': nv['d'][1], 'dg': nv['d'][2],
    }
    return weights


def cos_sin_tables(context, theta=500000.0):
    inv = theta ** (-torch.arange(0, DIM, 2, dtype=torch.float32) / DIM)
    angle = torch.arange(context, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=1).contiguous()  # [context, 2, half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--workdir', help='compiler workdir; keeps the generated NEFF and HLO for the C++ runtime harness')
    parser.add_argument('--compile-only', action='store_true', help='stop after compiling (no device run)')
    parser.add_argument('--weights', choices=('bf16', 'lowbit'), default='bf16',
                        help='bf16: host-dequantized dense weights; lowbit: packed bytes resident, decoded in-graph')
    a = parser.parse_args()
    import torch_neuronx
    import torch_xla.core.xla_model as xm

    lowbit = a.weights == 'lowbit'
    weights = build_layer_lowbit(a.model, a.layer) if lowbit else build_layer(a.model, a.layer)
    sample = {
        'x': torch.randn(1, HIDDEN),
        'slots': torch.zeros(1, dtype=torch.int64),
        'cos_sin': cos_sin_tables(a.context),
        'cache_k': torch.zeros(1, a.context, KVHEADS, DIM),
        'cache_v': torch.zeros(1, a.context, KVHEADS, DIM),
    }
    order = ['x', 'slots', 'cos_sin', 'cache_k', 'cache_v',
             'norm1', 'norm2', 'qw', 'kw', 'vw', 'ow', 'gw', 'uw', 'dw']
    if lowbit:
        order = ['x', 'slots', 'cos_sin', 'cache_k', 'cache_v', 'norm1', 'norm2', 'qw', 'ow',
                 'kb', 'ks', 'vb', 'vs', 'gp', 'gs', 'gg', 'up', 'us', 'ug', 'dp', 'ds', 'dg']
    inputs = [sample[k] if k in sample else weights[k] for k in order]
    # device bytes of the weight parameters as they are held resident (bf16 weights count 2 B/elem
    # on the device; the host tensors here are float32 holding bf16-rounded values and are cast
    # by the traced graph)
    resident = sum(weights[k].numel() * weights[k].element_size() for k in order if k in weights)

    started = time.monotonic()
    graph = torch_neuronx.trace(layer_graph_lowbit if lowbit else layer_graph, tuple(inputs),
                                compiler_workdir=a.workdir)
    compile_s = time.monotonic() - started
    if a.compile_only:
        print(json.dumps({'weights': a.weights, 'resident_weight_MB': round(resident / 1e6, 1), 'compile_s': round(compile_s, 2)}))
        return
    device = xm.xla_device()
    graph = torch_neuronx.move_trace_to_device(graph, 0)
    tensors = [t.to(device) for t in inputs]
    print(f'compiled one-NEFF layer in {compile_s:.1f}s; {a.steps} decode steps', flush=True)

    def step(tensors, position):
        x, _, cos_sin, cache_k, cache_v = tensors[:5]
        slots = torch.full((1,), position, dtype=torch.int64, device=x.device)
        return graph(x, slots, cos_sin, cache_k, cache_v, *tensors[5:])

    for i in range(a.warmup):
        x, cache_k, cache_v = step(tensors, i % a.context)
        xm.mark_step()
    started = time.monotonic()
    for i in range(a.steps):
        x, cache_k, cache_v = step(tensors, (a.warmup + i) % a.context)
        xm.mark_step()
    elapsed = time.monotonic() - started
    print(json.dumps({'weights': a.weights, 'resident_weight_MB': round(resident / 1e6, 1), 'compile_s': round(compile_s, 2), 'steps': a.steps,
                      'steady_ms_per_step': round(elapsed / a.steps * 1000, 2),
                      'steady_tokens_per_s': round(a.steps / elapsed, 2)}))


if __name__ == '__main__':
    main()
