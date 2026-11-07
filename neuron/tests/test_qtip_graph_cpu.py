"""The compressed-layer graph glue (qtip_graph: Hadamard, SU/SV, x^T layout, TP sharding and the all-reduce gather) on
the CPU against an independent numpy implementation of qtip::reference_forward, with real weights of a checkpoint layer.
The kernel is replaced by a dense matmul with the reference decode, so this isolates the torch side.

usage: test_qtip_graph_cpu.py MODEL_DIR LAYER [TP]
"""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qtip_graph
from qtip_nki import codebook_u32, decode_reference, shard_tiles, tiles_of

model, layer = sys.argv[1], int(sys.argv[2])
TP = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 2
config = json.loads((Path(model) / 'config.json').read_text())
H, FF = config['hidden_size'], config['intermediate_size']


def tensor(path, key):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[key]
        a, b = meta['data_offsets']
        f.seek(8 + n + a)
        raw = f.read(b - a)
    dtype = {'I16': np.int16, 'F16': np.float16, 'BF16': np.uint16, 'F32': np.float32}[meta['dtype']]
    arr = np.frombuffer(raw, dtype=dtype).reshape(meta['shape'])
    if meta['dtype'] == 'BF16':
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


def had_ref(x, n, transpose):
    """numpy qtip::hadamard over the last axis (natural-order FWHT per block, then the k x k factor)."""
    k = qtip_graph.had_k(n)
    p = n // k
    v = x.reshape(-1, k, p).astype(np.float64).copy()
    s = 1
    while s < p:
        for i in range(0, p, 2 * s):
            a = v[:, :, i:i + s].copy()
            b = v[:, :, i + s:i + 2 * s].copy()
            v[:, :, i:i + s] = a + b
            v[:, :, i + s:i + 2 * s] = a - b
        s *= 2
    if k > 1:
        h = qtip_graph.hadamard_factor(k).numpy().astype(np.float64)
        v = np.einsum('rc,bcp->brp', h.T if transpose else h, v)
    return (v / np.sqrt(n)).reshape(x.shape)


def bf16(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def reference(x, su, sv, dense, out, width):
    """qtip::reference_forward(device_rounding = true): u = bf16(hadUt(SU x)), z = u W~^T (f64), y = SV hadU(z)."""
    u = bf16(had_ref(x * su, width, True))
    z = u.astype(np.float64) @ dense.astype(np.float64).T
    return had_ref(z, out, False) * sv


file = Path(model) / f'model-layer-{layer:05d}.safetensors'

if __name__ == '__main__':
    rng = np.random.default_rng(1)
    T = 3
    worst = 0.0
    for name, out, width, col in (('self_attn.q_proj', H, H, True), ('self_attn.o_proj', H, H, False),
                                  ('mlp.gate_proj', FF, H, True), ('mlp.down_proj', H, FF, False)):
        base = f'model.layers.{layer}.{name}.'
        trellis, tlut = tensor(file, base + 'trellis'), tensor(file, base + 'tlut').view(np.uint16)
        su, sv = tensor(file, base + 'SU').astype(np.float32), tensor(file, base + 'SV').astype(np.float32)
        lut = codebook_u32(tlut)
        tiles = tiles_of(trellis, out, width)
        dense = decode_reference(tiles, lut[0])
        x = rng.normal(0, 1, (T, width)).astype(np.float32)
        y_ref = reference(x, su, sv, dense, out, width)
        # the graph, with the kernel replaced by a dense matmul of the rank's shard (same W~ values the kernel decodes)
        xt = torch.from_numpy(x)
        su_t, sv_t = torch.from_numpy(su).reshape(1, -1), torch.from_numpy(sv).reshape(1, -1)
        partial = []
        for rank in range(TP):
            shard = torch.from_numpy(decode_reference(shard_tiles(tiles, TP, rank, col), lut[0]))
            if col:
                sel = torch.arange(out // TP) + rank * (out // TP)
                u = qtip_graph.rotate_in(xt.to(torch.bfloat16), su_t, width)
                z_local = u.float() @ shard.T                                              # [T, out/TP]
                mask = torch.zeros(out).index_fill(0, sel, 1.0).reshape(1, out)
            full = torch.cat([z_local] * TP, dim=-1) * mask                              # gather_tp before the all-reduce
            else:
                sel = torch.arange(width // TP) + rank * (width // TP)
                u = torch.index_select(qtip_graph.rotate_in(xt.to(torch.bfloat16), su_t, width), -1, sel)
                full = u.float() @ shard.T                                                # partial z [T, out]
            partial.append(full)
        z = sum(partial)                                                                  # the all-reduce
        y = qtip_graph.finish(z, sv_t, out).numpy()
        rel = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
        worst = max(worst, rel)
        print(f'{name}: graph glue vs numpy reference relative_l2={rel:.3e}')
    assert worst < 1e-4, 'graph glue differs from the reference'
    print('OK')
