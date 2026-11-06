"""Known-answer FP8 matmul graph for probing Trn1's FP8 semantics: y = fp8_linear(a, Fp8Weight(w, s)).
usage: trace_fp8_probe.py WORKDIR     (K=256, N=16; compiled exactly like the layer graphs)"""
from pathlib import Path
import subprocess
import sys

import torch
import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_layer import Fp8Weight, fp8_linear

workdir = sys.argv[1]


def probe(a, w, s):
    return fp8_linear(a, Fp8Weight(w, s))


args = (torch.ones(1, 256), torch.zeros(16, 256).to(torch.float8_e4m3fn), torch.ones(16))
try:
    torch_neuronx.trace(probe, args, compiler_workdir=workdir)
except RuntimeError as error:
    if 'neuronx-cc failed' not in str(error):
        raise
m = Path(workdir) / 'model'
here = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(m / 'graph.hlo'), str(m / 'graph.hlo')], check=True)
subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'], cwd=workdir, check=True)
print('NEFF', workdir + '/graph.neff')
