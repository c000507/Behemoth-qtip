"""Tensor-parallel (TP=2, always on) Llama decode layer with FP8 weights, compiled to ONE collectives NEFF.

Per rank: 16 of 32 query heads, 4 of 8 KV heads, 7168 of 14336 MLP channels. q/k/v/gate/up are
column-parallel (output rows split), o_proj/down_proj are row-parallel (input columns split) and
their partial sums are all-reduced across the pair of NeuronCores (replica group {0, 1}). The graph
is SPMD: the same NEFF is loaded on both ranks with rank-specific weight shards.

Steps: torch_neuronx.trace emits the HLO (its own compile fails on F8E4M3FN, NCC_EVRF051), the HLO
is retargeted to F8E4M3 and compiled with neuronx-cc; the NEFF runs from C++ (cpp/layer_bench.cpp).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_layer import (DIM, HEADS, HIDDEN, INTERMEDIATE, KVHEADS, Fp8Weight, build_layer_fp8, cos_sin_tables,
                         layer_graph, lin_fp8)

TP = 2


def shard(weights, rank):
    """Rank-local slices of the full-layer FP8 weights (column-parallel: rows; row-parallel: columns)."""
    def rows(t, n): return t[rank * n:(rank + 1) * n]
    def cols(t, n): return t[:, rank * n:(rank + 1) * n]
    q_rows, kv_rows, ff = HIDDEN // TP, KVHEADS * DIM // TP, INTERMEDIATE // TP
    return {
        'norm1': weights['norm1'], 'norm2': weights['norm2'],
        'qw': rows(weights['qw'], q_rows), 'ow': cols(weights['ow'], HIDDEN // TP),
        'kf': rows(weights['kf'], kv_rows), 'ks': rows(weights['ks'], kv_rows),
        'vf': rows(weights['vf'], kv_rows), 'vs': rows(weights['vs'], kv_rows),
        'gf': rows(weights['gf'], ff), 'gs': rows(weights['gs'], ff),
        'uf': rows(weights['uf'], ff), 'us': rows(weights['us'], ff),
        'df': cols(weights['df'], ff), 'ds': weights['ds'],  # row-parallel: every rank keeps the full output scale
    }


def tp_layer(x, slots, cos_sin, cache_k, cache_v, norm1, norm2, qw, ow,
             kf, ks, vf, vs, gf, gs, uf, us, df, ds):
    import torch_xla.core.xla_model as xm
    allreduce = lambda t: xm.all_reduce(xm.REDUCE_SUM, t, groups=[list(range(TP))])
    return layer_graph(x, slots, cos_sin, cache_k, cache_v, norm1, norm2, qw,
                       Fp8Weight(kf, ks), Fp8Weight(vf, vs), ow, Fp8Weight(gf, gs), Fp8Weight(uf, us),
                       Fp8Weight(df, ds), _lin=lin_fp8, heads=HEADS // TP, kvheads=KVHEADS // TP,
                       reduce=allreduce)


ORDER = ['x', 'slots', 'cos_sin', 'cache_k', 'cache_v', 'norm1', 'norm2', 'qw', 'ow',
         'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--workdir', required=True)
    a = parser.parse_args()
    import torch_neuronx

    weights = shard(build_layer_fp8(a.model, a.layer, 'fp8'), 0)
    sample = {
        'x': torch.randn(1, HIDDEN), 'slots': torch.zeros(1, dtype=torch.int64),
        'cos_sin': cos_sin_tables(a.context),
        'cache_k': torch.zeros(1, a.context, KVHEADS // TP, DIM),
        'cache_v': torch.zeros(1, a.context, KVHEADS // TP, DIM),
    }
    inputs = tuple(sample[k] if k in sample else weights[k] for k in ORDER)
    resident = sum(t.numel() * t.element_size() for k, t in zip(ORDER, inputs) if k in weights)
    try:
        torch_neuronx.trace(tp_layer, inputs, compiler_workdir=a.workdir)
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
    model = Path(a.workdir) / 'model'
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(model / 'graph.hlo'),
                    str(model / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1',
                    '--output', 'graph.neff'], cwd=a.workdir, check=True)
    print(json.dumps({'tp': TP, 'weights_MB_per_rank': round(resident / 1e6, 1), 'neff': str(Path(a.workdir) / 'graph.neff')}))


if __name__ == '__main__':
    main()
