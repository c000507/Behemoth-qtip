"""One compressed QTIP projection as a device graph (rotate_in -> fused kernel -> finish, qtip_graph, no tensor parallelism)
with real weights of a checkpoint layer, against the numpy reference of test_qtip_graph_cpu.py.

usage: NEURON_RT_VISIBLE_CORES=0 test_qtip_proj_device.py MODEL_DIR LAYER PROJECTION
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qtip_graph
from qtip_nki import codebook_u32, decode_reference, pack_tiles, tiles_of
from test_qtip_graph_cpu import H, FF, bf16, file, layer, reference, tensor  # noqa: E402  (reuses its argv parsing)

name = sys.argv[3]
out, width = {'self_attn.q_proj': (H, H), 'self_attn.o_proj': (H, H), 'mlp.gate_proj': (FF, H), 'mlp.up_proj': (FF, H),
              'mlp.down_proj': (H, FF)}[name]
base = f'model.layers.{layer}.{name}.'
trellis, tlut = tensor(file, base + 'trellis'), tensor(file, base + 'tlut').view(np.uint16)
su, sv = tensor(file, base + 'SU').astype(np.float32), tensor(file, base + 'SV').astype(np.float32)
lut = codebook_u32(tlut)
tiles = tiles_of(trellis, out, width)
dense = decode_reference(tiles, lut[0])
T = 3
rng = np.random.default_rng(1)
x = rng.normal(0, 1, (T, width)).astype(np.float32)
y_ref = reference(x, su, sv, dense, out, width)

words = torch.from_numpy(pack_tiles(tiles).view(np.int16)).view(torch.uint16)
t_lut = torch.from_numpy(lut.view(np.int32)).view(torch.uint32)
xt, su_t, sv_t = torch.from_numpy(x), torch.from_numpy(su).reshape(1, -1), torch.from_numpy(sv).reshape(1, -1)


def graph(x, su, sv, words, lut):
    u = qtip_graph.rotate_in(x.to(torch.bfloat16), su, width)
    return qtip_graph.finish(qtip_graph.kernel_matmul(u, words, lut), sv, out)


m = torch_neuronx.trace(graph, (xt, su_t, sv_t, words, t_lut), compiler_workdir='/tmp/test_qtip_proj_device')
y = m(xt, su_t, sv_t, words, t_lut.view(torch.int32)).numpy()
rel = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
print(f'{name}: device graph vs numpy reference relative_l2={rel:.3e}')
# the same with the kernel replaced by the dense CPU matmul (graph glue only) for scale
u = qtip_graph.rotate_in(xt.to(torch.bfloat16), su_t, width)
y_cpu = qtip_graph.finish(u.float() @ torch.from_numpy(dense).T, sv_t, out).numpy()
print(f'   cpu glue vs numpy reference relative_l2={np.linalg.norm(y_cpu - y_ref) / np.linalg.norm(y_ref):.3e}')
assert rel < 1e-2, 'device projection graph differs from the reference'
print('OK')
