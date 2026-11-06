"""TP=2 Llama decode layer whose q/o/gate/up/down projections are QTIP (2-bit trellis words resident on
the device, decoded in-graph every step); k/v stay FP8. One collectives NEFF per layer for the C++ harness.

usage: trace_qtip_layer_tp.py MODEL WORKDIR [layer] [context]       (layer must have QTIP MLP + attention)
"""
import json
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint
from qtip_tp import col_parallel, hadamard_factor, row_parallel, shard_trellis
from trace_layer import (DIM, HEADS, HIDDEN, INTERMEDIATE, KVHEADS, Fp8Weight, cos_sin_tables, layer_graph,
                         lin_fp8, to_rows)

TP = 2
PROJ = (('q', 'self_attn.q_proj', HIDDEN, HIDDEN, 'col'), ('o', 'self_attn.o_proj', HIDDEN, HIDDEN, 'row'),
        ('g', 'mlp.gate_proj', HIDDEN, INTERMEDIATE, 'col'), ('u', 'mlp.up_proj', HIDDEN, INTERMEDIATE, 'col'),
        ('d', 'mlp.down_proj', INTERMEDIATE, HIDDEN, 'row'))
ORDER = (['x', 'slots', 'cos_sin', 'cache_k', 'cache_v', 'norm1', 'norm2', 'kf', 'ks', 'vf', 'vs', 'had28']
         + [f'{p[0]}_{f}' for p in PROJ for f in ('tr', 'tl', 'su', 'sv', 'ix')])


class QtipWeight:
    def __init__(self, tr, tl, su, sv, ix, out, width, kind, factors):
        self.tr, self.tl, self.su, self.sv, self.ix = tr, tl, su, sv, ix
        self.out, self.width, self.kind, self.factors = out, width, kind, factors


def lin(h, w):
    if isinstance(w, QtipWeight):
        fn = col_parallel if w.kind == 'col' else row_parallel
        if w.kind == 'col':
            return fn(h, w.su, w.sv, w.tr, w.tl, w.out, w.width, w.factors, w.ix, TP)
        return fn(h, w.su, w.sv, w.tr, w.tl, w.out, w.width, w.factors, w.ix, TP)
    return lin_fp8(h, w)


def qtip_layer(*args):
    t = dict(zip(ORDER, args))
    factors = {28: t['had28']}
    ws = {p[0]: QtipWeight(t[f'{p[0]}_tr'], t[f'{p[0]}_tl'], t[f'{p[0]}_su'], t[f'{p[0]}_sv'], t[f'{p[0]}_ix'],
                           p[3], p[2], p[4], factors) for p in PROJ}
    return layer_graph(t['x'], t['slots'], t['cos_sin'], t['cache_k'], t['cache_v'], t['norm1'], t['norm2'],
                       ws['q'], Fp8Weight(t['kf'], t['ks']), Fp8Weight(t['vf'], t['vs']), ws['o'], ws['g'], ws['u'], ws['d'],
                       _lin=lin, heads=HEADS // TP, kvheads=KVHEADS // TP, reduce=None)


def build_inputs(model, layer, context, rank=0):
    cp = Checkpoint(model)
    inputs = {
        'x': torch.randn(1, HIDDEN), 'slots': torch.zeros(1, dtype=torch.int64), 'cos_sin': cos_sin_tables(context),
        'cache_k': torch.zeros(1, context, KVHEADS // TP, DIM), 'cache_v': torch.zeros(1, context, KVHEADS // TP, DIM),
        'norm1': torch.from_numpy(cp.tensor(f'model.layers.{layer}.input_layernorm.weight').astype(np.float32).reshape(1, -1)),
        'norm2': torch.from_numpy(cp.tensor(f'model.layers.{layer}.post_attention_layernorm.weight').astype(np.float32).reshape(1, -1)),
        'had28': torch.from_numpy(hadamard_factor(28)),
    }
    kv_rows = KVHEADS * DIM // TP
    for tag, name in (('k', 'self_attn.k_proj'), ('v', 'self_attn.v_proj')):
        p = cp.projection(layer, name, HIDDEN, KVHEADS * DIM)
        dense = torch.from_numpy(p.weights.astype(np.float32) * p.scale[:, None])[rank * kv_rows:(rank + 1) * kv_rows]
        q, s = to_rows(dense, 'fp8')
        inputs[tag + 'f'], inputs[tag + 's'] = q, s
    for key, name, width, out, kind in PROJ:
        p = cp.projection(layer, name, width, out)
        if not hasattr(p, 'trellis'):
            raise SystemExit(f'layer {layer} {name} is not QTIP')
        inputs[f'{key}_tr'] = torch.from_numpy(shard_trellis(p.trellis.view(np.int16), out, width, TP, rank, kind))
        inputs[f'{key}_tl'] = torch.from_numpy(cp.tensor(f'model.layers.{layer}.{name}.tlut').astype(np.float32))
        inputs[f'{key}_su'], inputs[f'{key}_sv'] = torch.from_numpy(p.su), torch.from_numpy(p.sv)
        n_local = (out if kind == 'col' else width) // TP
        inputs[f'{key}_ix'] = torch.arange(rank * n_local, (rank + 1) * n_local, dtype=torch.int64)
    return inputs


def main():
    model, workdir = sys.argv[1], sys.argv[2]
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    context = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    import torch_neuronx
    inputs = build_inputs(model, layer, context)
    args = tuple(inputs[k] for k in ORDER)
    resident = sum(t.numel() * t.element_size() for k, t in inputs.items() if k not in ('x', 'slots', 'cos_sin', 'cache_k', 'cache_v'))
    try:
        torch_neuronx.trace(qtip_layer, args, compiler_workdir=workdir)
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
    m = Path(workdir) / 'model'
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(m / 'graph.hlo'), str(m / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'],
                   cwd=workdir, check=True)
    print(json.dumps({'tp': TP, 'weights_MB_per_rank': round(resident / 1e6, 1), 'neff': str(Path(workdir) / 'graph.neff')}))


if __name__ == '__main__':
    main()
