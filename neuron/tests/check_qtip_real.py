"""Fused QTIP decode kernel on the DEVICE against the CPU reference, on a real projection of a checkpoint.

  1. the numpy mirror of qtip::decode_rows_bf16 (qtip_nki.decode_reference) is compared bit-exactly with a dump of the
     C++ reference (neuron/cpp/qtip_decode_dump.cpp) when DUMP=file ROW0=r ROWS=n are given;
  2. the kernel (traced with torch_neuronx and executed on NeuronCore 0) multiplies random bf16 inputs by this rank's
     shard of the trellis, packed exactly as the runtime packs it, and is compared with u @ W~_shard^T in float64:
     only f32 accumulation order differs, so the relative L2 must be ~1e-6.

usage: check_qtip_real.py MODEL_DIR LAYER PROJECTION TP RANK [T]
       env: DUMP / ROW0 / ROWS (step 1), NEURON_RT_VISIBLE_CORES=0
"""
import json
import os
from pathlib import Path
import struct
import sys

import ml_dtypes
import numpy as np
import torch
import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_nki import codebook_u32, decode_reference, pack_tiles, qtip_linear, shard_tiles, tiles_of, xt_layout

model, layer, name, tp, rank = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
T = int(sys.argv[6]) if len(sys.argv) > 6 else 8
column_parallel = name in ('self_attn.q_proj', 'mlp.gate_proj', 'mlp.up_proj')
config = json.loads((Path(model) / 'config.json').read_text())
H, FF = config['hidden_size'], config['intermediate_size']
out, width = {'self_attn.q_proj': (H, H), 'self_attn.o_proj': (H, H), 'mlp.gate_proj': (FF, H), 'mlp.up_proj': (FF, H),
              'mlp.down_proj': (H, FF)}[name]


def tensor(path, key):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[key]
        a, b = meta['data_offsets']
        f.seek(8 + n + a)
        raw = f.read(b - a)
    dtype = {'I16': np.int16, 'F16': np.float16, 'BF16': np.uint16, 'F32': np.float32}[meta['dtype']]
    return np.frombuffer(raw, dtype=dtype).reshape(meta['shape'])


base = f'model.layers.{layer}.{name}.'
file = Path(model) / f'model-layer-{layer:05d}.safetensors'
trellis = tensor(file, base + 'trellis')
tlut = tensor(file, base + 'tlut').view(np.uint16)
lut = codebook_u32(tlut)
tiles = tiles_of(trellis, out, width)
shard = shard_tiles(tiles, tp, rank, column_parallel)
dense = decode_reference(shard, lut[0])                                   # [out_local, in_local] bf16 values as f32
print(f'{name} layer {layer}: {out}x{width}, rank {rank}/{tp} shard {dense.shape}')

if os.environ.get('DUMP'):
    row0, rows = int(os.environ['ROW0']), int(os.environ['ROWS'])
    ref = np.fromfile(os.environ['DUMP'], dtype=np.float32).reshape(rows, width)
    full = decode_reference(tiles[row0 // 16:(row0 + rows) // 16], lut[0])
    exact = np.array_equal(full, ref)
    print(f'numpy mirror vs C++ decode_rows_bf16 rows [{row0}, {row0 + rows}): {"bit-exact" if exact else "DIFFERENT"}')
    assert exact

words = pack_tiles(shard)
G = words.shape[0]
rng = np.random.default_rng(0)
u = rng.normal(0, 1, (T, dense.shape[1])).astype(ml_dtypes.bfloat16)
xT = xt_layout(u, G)
t_words = torch.from_numpy(words.view(np.int16)).view(torch.uint16)
t_lut = torch.from_numpy(lut.view(np.int32)).view(torch.uint32)
t_x = torch.from_numpy(xT.view(np.int16)).view(torch.bfloat16)
kernel = torch_neuronx.trace(lambda w, l, x: qtip_linear(w, l, x), (t_words, t_lut, t_x), compiler_workdir='/tmp/check_qtip_real')
z = kernel(t_words, t_lut, t_x).float().numpy()
expected = u.astype(np.float64) @ dense.astype(np.float64).T
rel = np.linalg.norm(z - expected) / np.linalg.norm(expected)
print(f'device kernel vs reference decode: relative_l2={rel:.3e} max_abs={np.max(np.abs(z - expected)):.3e} (|z| ~ {np.abs(expected).mean():.2f})')
assert rel < 1e-5, 'device decode differs from the CPU reference beyond f32 accumulation rounding'
print('OK')
