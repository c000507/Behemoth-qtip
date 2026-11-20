#!/usr/bin/env python3
"""CPU check of the KV8 attention path (qwen35_graphs.gqa_fp8) against the f32 paged attention (EXACT=1):
a prompt in chunks + decode steps over a shuffled page table, fp8 codes with per-token scales, key blocks with online softmax.
Expected: blocked == unblocked fp8 to f32 rounding (~1e-6), fp8 vs f32 cache ~1e-2 relative (e4m3 has 3 mantissa bits)."""
import os
import sys
from pathlib import Path

os.environ['EXACT'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

import qwen35_graphs as Q  # noqa: E402


def main():
    torch.manual_seed(0)
    cfg = {'hidden_size': 256, 'vocab_size': 1000, 'num_hidden_layers': 4, 'rms_norm_eps': 1e-6, 'linear_num_key_heads': 2,
           'linear_key_head_dim': 32, 'linear_num_value_heads': 4, 'linear_value_head_dim': 32, 'linear_conv_kernel_dim': 4,
           'num_attention_heads': 4, 'num_key_value_heads': 1, 'head_dim': 64, 'partial_rotary_factor': 0.25, 'attn_output_gate': True,
           'num_experts': 4, 'num_experts_per_tok': 2, 'moe_intermediate_size': 64, 'shared_expert_intermediate_size': 64,
           'layer_types': ['linear_attention'] * 3 + ['full_attention'], 'max_position_embeddings': 4096}
    g = Q.Geo(cfg, 1)
    P, pages = 16, 40
    H, D, hq, kvl = g.H, g.D, g.hq, g.kvl
    w = {'q': torch.randn(hq * 2 * D, H) / H ** 0.5, 'k': torch.randn(kvl * D, H) / H ** 0.5, 'v': torch.randn(kvl * D, H) * 3 / H ** 0.5,
         'o': torch.randn(H, hq * D) / (hq * D) ** 0.5, 'qn': 1 + 0.1 * torch.randn(D), 'kn': 1 + 0.1 * torch.randn(D)}
    table = torch.randperm(pages - 1)[:P].reshape(1, P)
    ar = lambda v: v
    state = {}
    for mode in ('f32', 'fp8', 'fp8b', 'fp8q'):
        if mode == 'f32':
            state[mode] = {'kv': torch.zeros(pages, 2, kvl, Q.PAGE, D)}
        else:
            state[mode] = {'kv': torch.zeros(pages, 2, kvl, Q.PAGE, D, dtype=torch.int8 if Q.KV_CODE == 'int8' else Q.F8), 'kvs': torch.zeros(pages, 2, kvl, Q.PAGE)}
    steps = [(0, 64), (64, 64), (128, 50)] + [(178 + i, 1) for i in range(20)]
    worst = {'fp8': 0.0, 'fp8b': 0.0, 'blk': 0.0, 'qblk': 0.0}
    for p0, n in steps:
        x = torch.randn(n, H)
        pos = torch.arange(p0, p0 + n)
        inv = 1.0 / (g.theta ** (torch.arange(0, g.rot, 2).float() / g.rot))
        ang = pos.float()[:, None] * inv[None]
        cs = torch.cat((torch.cat((ang.cos(), ang.cos()), -1), torch.cat((ang.sin(), ang.sin()), -1)), -1)
        wslot = table[0, pos // Q.PAGE] * Q.PAGE + pos % Q.PAGE
        out = {}
        for mode in ('f32', 'fp8', 'fp8b', 'fp8q'):   # fp8q: key blocks x query blocks (ATT_QBLOCK, wide prefill chunks)
            Q.ATT_BLOCK = 4 if mode in ('fp8b', 'fp8q') else 0
            Q.ATT_QBLOCK = 16 if mode == 'fp8q' else 0
            t = {**w, **state[mode], 'cos_sin': cs, 'wslot': wslot, 'table': table, 'pos': pos}
            y, st = Q.gqa_block(x, t, g, 1, n, P, ar)
            state[mode].update(st)
            out[mode] = y
        rel = lambda a, b: float((a - b).norm() / b.norm())
        worst['fp8'] = max(worst['fp8'], rel(out['fp8'], out['f32']))
        worst['fp8b'] = max(worst['fp8b'], rel(out['fp8b'], out['f32']))
        worst['blk'] = max(worst['blk'], rel(out['fp8b'], out['fp8']))
        worst['qblk'] = max(worst['qblk'], rel(out['fp8q'], out['fp8b']))
    print(Q.KV_CODE, 'max relative error over %d calls: fp8 vs f32 %.3e, fp8 blocked vs f32 %.3e, blocked vs unblocked %.3e' %
          (len(steps), worst['fp8'], worst['fp8b'], worst['blk']))
    assert worst['blk'] < 1e-5, worst
    assert worst['qblk'] < 1e-5, worst
    print('query blocks vs key blocks only: %.3e' % worst['qblk'])
    assert worst['fp8'] < 5e-2, worst
    print('OK')


if __name__ == '__main__':
    main()
