"""Tensor-parallel (TP=2) compressed QTIP projection probes on the two NeuronCores with real layer weights: the graph
of trace_model.py's qtip path for one column-parallel (q/gate/up) or row-parallel (o/down) projection is traced with
the collectives, run through cpp/neff_io.cpp with per-rank inputs (rank shard of the words, rank index vector) and
compared with the numpy reference of test_qtip_graph_cpu.py.

usage: test_qtip_tp_device.py MODEL_DIR LAYER PROJECTION NEFF_IO_BINARY   (NEURON_RT_VISIBLE_CORES=0-1, ROOT_COMM_ID set)
"""
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('XLA_ALWAYS_ALLREDUCE', '1')
import ml_dtypes
import numpy as np
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qtip_graph
from qtip_nki import codebook_u32, decode_reference, pack_shard, tiles_of
from test_qtip_graph_cpu import H, FF, file, layer, reference, tensor  # noqa: E402

name, neff_io = sys.argv[3], sys.argv[4]
TP, T = 2, 5
col = name in ('self_attn.q_proj', 'mlp.gate_proj', 'mlp.up_proj')
out, width = {'self_attn.q_proj': (H, H), 'self_attn.o_proj': (H, H), 'mlp.gate_proj': (FF, H), 'mlp.up_proj': (FF, H),
              'mlp.down_proj': (H, FF)}[name]
base = f'model.layers.{layer}.{name}.'
trellis, tlut = tensor(file, base + 'trellis'), tensor(file, base + 'tlut').view(np.uint16)
su, sv = tensor(file, base + 'SU').astype(np.float32), tensor(file, base + 'SV').astype(np.float32)
lut = codebook_u32(tlut)
dense = decode_reference(tiles_of(trellis, out, width), lut[0])
rng = np.random.default_rng(1)
x = rng.normal(0, 1, (T, width)).astype(np.float32)
y_ref = reference(x, su, sv, dense, out, width)
allreduce = lambda v: xm.all_reduce(xm.REDUCE_SUM, v, groups=[list(range(TP))])


def col_graph(x, su, sv, words, lut, sel):           # replicated x -> this rank's slice of y
    z = qtip_graph.gather_tp(qtip_graph.kernel_matmul(qtip_graph.rotate_in(x.to(torch.bfloat16), su, width), words, lut), sel, out, allreduce)
    return torch.index_select(qtip_graph.finish(z, sv, out), -1, sel)


def row_graph(x_local, su, sv, words, lut, sel):     # this rank's slice of x -> full y (replicated)
    u = torch.index_select(qtip_graph.rotate_in(qtip_graph.gather_tp(x_local, sel, width, allreduce), su, width), -1, sel)
    return qtip_graph.finish(allreduce(qtip_graph.kernel_matmul(u, words, lut)), sv, out)


words0 = pack_shard(trellis, out, width, TP, 0, col)
sel0 = np.arange((out if col else width) // TP, dtype=np.int64)
ex = (torch.from_numpy(x if col else x[:, :width // TP].copy()), torch.from_numpy(su).reshape(1, -1), torch.from_numpy(sv).reshape(1, -1),
      torch.from_numpy(words0.view(np.int16)).view(torch.uint16), torch.from_numpy(lut.view(np.int32)).view(torch.uint32), torch.from_numpy(sel0))
work = Path('/tmp/test_qtip_tp_device') / name.replace('.', '_')
work.mkdir(parents=True, exist_ok=True)
torch_neuronx.trace(col_graph if col else row_graph, ex, compiler_workdir=str(work))
for r in range(TP):
    n = (out if col else width) // TP
    xin = x if col else x[:, r * n:(r + 1) * n].copy()
    pack_shard(trellis, out, width, TP, r, col).tofile(work / f'in3.r{r}')
    (np.arange(n, dtype=np.int64) + r * n).tofile(work / f'in5.r{r}')
    xin.astype(np.float32).tofile(work / f'in0.r{r}')
su.tofile(work / 'in1'); sv.tofile(work / 'in2'); lut.tofile(work / 'in4')
subprocess.run([neff_io, str(work / 'graph.neff'), str(work), str(TP)], check=True)
worst = 0
for r in range(TP):
    y = np.fromfile(work / f'out0.r{r}', dtype=np.float32).reshape(T, -1)
    ref = y_ref[:, r * (out // TP):(r + 1) * (out // TP)] if col else y_ref
    rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
    worst = max(worst, rel)
    print(f'{name} rank {r}: TP=2 device graph vs numpy reference relative_l2={rel:.3e}')
assert worst < 1e-2, 'TP=2 compressed projection differs from the reference'
print('OK')
