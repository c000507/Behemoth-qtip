"""Known-answer all-reduce graph: y = all_reduce_sum(x), replica group {0, 1}. usage: trace_allreduce_probe.py WORKDIR [dtype f32|bf16]"""
from pathlib import Path
import os
import sys

os.environ.setdefault('XLA_ALWAYS_ALLREDUCE', '1')  # otherwise the single-process trace drops the collective
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm

workdir = sys.argv[1]
dtype = torch.bfloat16 if len(sys.argv) > 2 and sys.argv[2] == 'bf16' else torch.float32


def probe(x):
    return xm.all_reduce(xm.REDUCE_SUM, x, groups=[[0, 1]])


torch_neuronx.trace(probe, (torch.ones(1, 8, dtype=dtype),), compiler_workdir=workdir)
print('NEFF', workdir + '/graph.neff')
