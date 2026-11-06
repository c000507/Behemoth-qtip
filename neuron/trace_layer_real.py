"""TP=2 FP8 decode layer for REAL weights: all seven projections are fp8 x fp8 matmuls (QTIP, NVFP4 and FP8
checkpoint tensors are all converted to row-scaled E4M3 shards by the C++ loader), with an attention mask so a
decode step at position p only sees cache slots <= p. Compiles to one collectives NEFF; cpp/layer_run.cpp binds
the real weights to the inputs by position (input0, input1, ... in ORDER).

usage: trace_layer_real.py WORKDIR [context]
"""
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_layer import DIM, HEADS, HIDDEN, INTERMEDIATE, KVHEADS, Fp8Weight, cos_sin_tables, layer_graph, lin_fp8

TP = 2
ORDER = ['x', 'slots', 'cos_sin', 'mask', 'cache_k', 'cache_v', 'norm1', 'norm2',
         'qf', 'qs', 'of', 'os', 'kf', 'ks', 'vf', 'vs', 'gf', 'gs', 'uf', 'us', 'df', 'ds']


def shapes(context):
    """name -> (shape, dtype) of every graph input for one rank."""
    q, kv, ff = HIDDEN // TP, KVHEADS * DIM // TP, INTERMEDIATE // TP
    f8 = torch.float8_e4m3fn
    return {
        'x': ((1, HIDDEN), torch.float32), 'slots': ((1,), torch.int64), 'cos_sin': ((context, 2, DIM // 2), torch.float32),
        'mask': ((context,), torch.float32), 'cache_k': ((1, context, KVHEADS // TP, DIM), torch.float32),
        'cache_v': ((1, context, KVHEADS // TP, DIM), torch.float32),
        'norm1': ((1, HIDDEN), torch.float32), 'norm2': ((1, HIDDEN), torch.float32),
        'qf': ((q, HIDDEN), f8), 'qs': ((q,), torch.float32), 'of': ((HIDDEN, q), f8), 'os': ((HIDDEN,), torch.float32),
        'kf': ((kv, HIDDEN), f8), 'ks': ((kv,), torch.float32), 'vf': ((kv, HIDDEN), f8), 'vs': ((kv,), torch.float32),
        'gf': ((ff, HIDDEN), f8), 'gs': ((ff,), torch.float32), 'uf': ((ff, HIDDEN), f8), 'us': ((ff,), torch.float32),
        'df': ((HIDDEN, ff), f8), 'ds': ((HIDDEN,), torch.float32),
    }


def real_layer(*args, stop_after_attention=False, debug=None):
    import torch_xla.core.xla_model as xm
    t = dict(zip(ORDER, args))
    allreduce = lambda v: xm.all_reduce(xm.REDUCE_SUM, v, groups=[list(range(TP))])
    w = lambda a, b: Fp8Weight(t[a], t[b])
    out = layer_graph(t['x'], t['slots'], t['cos_sin'], t['cache_k'], t['cache_v'], t['norm1'], t['norm2'],
                       w('qf', 'qs'), w('kf', 'ks'), w('vf', 'vs'), w('of', 'os'), w('gf', 'gs'), w('uf', 'us'), w('df', 'ds'),
                       _lin=lin_fp8, heads=HEADS // TP, kvheads=KVHEADS // TP, reduce=allreduce, mask=t['mask'], stop_after_attention=stop_after_attention, debug=debug)
    if stop_after_attention or debug:  # keep every parameter in the graph so the input numbering stays the same
        keep = sum(t[k].float().reshape(-1)[0] for k in ORDER if k not in ('x', 'cache_k', 'cache_v'))
        out = (out[0] + 0.0 * keep, out[1], out[2])
    return out


def main():
    workdir = sys.argv[1]
    context = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    import torch_neuronx
    spec = shapes(context)
    sample = []
    for k in ORDER:
        shape, dtype = spec[k]
        sample.append(torch.zeros(shape, dtype=dtype) if dtype != torch.float8_e4m3fn else torch.zeros(shape).to(dtype))
    try:
        import os
        stage = os.environ.get('DEBUG_STAGE')  # attn | v | attnout
        fn = (lambda *a: real_layer(*a, stop_after_attention=(stage == 'attn'), debug=stage if stage in ('v', 'attnout', 'opart', 'o') else None)) if stage else real_layer
        torch_neuronx.trace(fn, tuple(sample), compiler_workdir=workdir)
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
    m = Path(workdir) / 'model'
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(m / 'graph.hlo'), str(m / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'],
                   cwd=workdir, check=True)
    print(json.dumps({'tp': TP, 'context': context, 'inputs': ORDER, 'neff': str(Path(workdir) / 'graph.neff')}))


if __name__ == '__main__':
    main()
