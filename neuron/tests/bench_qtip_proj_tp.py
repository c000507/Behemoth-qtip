"""One TP=2 QTIP projection (resident 2-bit shard, in-graph decode) compiled alone to measure compile
cost and step time. usage: bench_qtip_proj_tp.py MODEL WORKDIR PROJ [layer]   PROJ in q|o|g|u|d"""
from pathlib import Path
import sys

import torch
import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_tp import col_parallel, row_parallel
from trace_qtip_layer_tp import PROJ, TP, build_inputs
from trace_layer import HIDDEN

model, workdir, key = sys.argv[1], sys.argv[2], sys.argv[3]
layer = int(sys.argv[4]) if len(sys.argv) > 4 else 0
_, name, width, out, kind = next(p for p in PROJ if p[0] == key)
d = build_inputs(model, layer, 128)
names = ['x', 'had28', f'{key}_tr', f'{key}_tl', f'{key}_su', f'{key}_sv', f'{key}_ix']
x = torch.randn(1, width) if kind == 'col' else torch.randn(1, width // TP)
args = (x, d['had28'], d[f'{key}_tr'], d[f'{key}_tl'], d[f'{key}_su'], d[f'{key}_sv'], d[f'{key}_ix'])


def proj(x, had28, tr, tl, su, sv, ix):
    fn = col_parallel if kind == 'col' else row_parallel
    return fn(x, su, sv, tr, tl, out, width, {28: had28}, ix, TP)


torch_neuronx.trace(proj, args, compiler_workdir=workdir)
print('NEFF', workdir + '/graph.neff', 'weights_MB', round(d[f'{key}_tr'].numel() * 2 / 1e6, 2))
