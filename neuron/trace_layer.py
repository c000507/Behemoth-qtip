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
    codes = np.empty(projection.packed.shape + (projection.width,), np.uint8)
    codes[..., 0::2] = projection.packed & 15
    codes[..., 1::2] = projection.packed >> 4
    dense = (lut[codes]
             * np.repeat(projection.scales.astype(np.float32), 16, axis=1)
             * projection.global_scale)
    return torch.from_numpy(dense.astype(ml_dtypes.bfloat16).astype(np.float32))


def dequant_fp8(projection):
    dense = projection.weights.astype(np.float32) * projection.scale[:, None]
    return torch.from_numpy(dense.astype(ml_dtypes.bfloat16).astype(np.float32))


def bf16(array):
    return torch.from_numpy(np.asarray(array, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32))


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
    cache_k = cache_k.index_copy(0, slots, k.reshape(1, KVHEADS, DIM))
    cache_v = cache_v.index_copy(0, slots, v.reshape(1, KVHEADS, DIM))
    attn = torch.zeros((1, HEADS, DIM), dtype=torch.float32)
    for head in range(HEADS):
        kh = head // (HEADS // KVHEADS)
        keys = cache_k[:, kh]      # [context, dim]
        values = cache_v[:, kh]
        scores = (q[0, head] @ keys.T) * scale
        weights = torch.softmax(scores, dim=-1)
        attn[0, head] = weights @ values
    attn = attn.reshape(1, HEADS * DIM).to(torch.bfloat16)
    x = x + (attn @ ow.T).float()
    h = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * norm2
    h = h.to(torch.bfloat16)
    gate = h @ gw.T
    act = torch.nn.functional.silu(gate).float() * (h @ uw.T).float()
    x = x + (act.to(torch.bfloat16) @ dw.T).float()
    return x, cache_k, cache_v


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
    a = parser.parse_args()
    import torch_neuronx
    import torch_xla.core.xla_model as xm

    weights = build_layer(a.model, a.layer)
    sample = {
        'x': torch.randn(1, HIDDEN),
        'slots': torch.zeros(1, dtype=torch.int64),
        'cos_sin': cos_sin_tables(a.context),
        'cache_k': torch.zeros(1, a.context, KVHEADS, DIM),
        'cache_v': torch.zeros(1, a.context, KVHEADS, DIM),
    }
    order = ['x', 'slots', 'cos_sin', 'cache_k', 'cache_v',
             'norm1', 'norm2', 'qw', 'kw', 'vw', 'ow', 'gw', 'uw', 'dw']
    inputs = [sample[k] if k in sample else weights[k] for k in order]

    started = time.monotonic()
    graph = torch_neuronx.trace(layer_graph, tuple(inputs))
    compile_s = time.monotonic() - started
    device = torch.device('neuron')
    graph = torch_neuronx.move_trace_to_device(graph, device)
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
    print(json.dumps({'compile_s': round(compile_s, 2), 'steps': a.steps,
                      'steady_ms_per_step': round(elapsed / a.steps * 1000, 2),
                      'steady_tokens_per_s': round(a.steps / elapsed, 2)}))


if __name__ == '__main__':
    main()
