"""Two-rank emulation of the TP QTIP projections vs the C++ qtip_checkpoint_oracle on real layers (CPU).

usage: test_qtip_tp.py MODEL ORACLE [layer]
Checks q_proj / up_proj (column-parallel) and o_proj / down_proj (row-parallel) of a QTIP layer.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint
from qtip_tp import (col_finish, col_local, hadamard_factor, row_finish, row_local, shard_trellis)

model, oracle = sys.argv[1], sys.argv[2]
layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TP = 2
cp = Checkpoint(model)
hidden, inter = int(cp.config['hidden_size']), int(cp.config['intermediate_size'])
factors = {12: torch.from_numpy(hadamard_factor(12)), 28: torch.from_numpy(hadamard_factor(28))}

for name, width, out, parallel in (('self_attn.q_proj', hidden, hidden, 'col'), ('self_attn.o_proj', hidden, hidden, 'row'),
                                   ('mlp.up_proj', hidden, inter, 'col'), ('mlp.down_proj', inter, hidden, 'row')):
    proj = cp.projection(layer, name, width, out)
    tlut = torch.from_numpy(cp.tensor(f'model.layers.{layer}.{name}.tlut').astype(np.float32))
    su, sv = torch.from_numpy(proj.su), torch.from_numpy(proj.sv)
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([oracle, model, str(layer), name, d], check=True, capture_output=True)
        x = torch.from_numpy(np.fromfile(Path(d) / 'x.bin', dtype=np.float32).reshape(1, -1))
        expected = np.fromfile(Path(d) / 'y.bin', dtype=np.float32).reshape(1, -1)
    shards = [torch.from_numpy(shard_trellis(proj.trellis.view(np.int16), out, width, TP, r, parallel)) for r in range(TP)]
    if parallel == 'col':
        z = torch.cat([col_local(x, su, shards[r], tlut, out // TP, width, factors) for r in range(TP)], dim=-1)
        y = torch.cat([col_finish(z, sv, out, factors, torch.arange(r * out // TP, (r + 1) * out // TP)) for r in range(TP)], dim=-1)
    else:
        z = sum(row_local(x, su, shards[r], tlut, out, width // TP, width, factors,
                          torch.arange(r * width // TP, (r + 1) * width // TP)) for r in range(TP))
        y = row_finish(z, sv, out, factors)
    y = y.numpy()
    rel = np.linalg.norm(y - expected) / np.linalg.norm(expected)
    print(f'{name:18s} {parallel}-parallel TP={TP}: relative_l2={rel:.3e}')
    assert rel < 1e-2, f'{name} does not match the C++ oracle'
print('TP QTIP projections match the C++ oracle (2-rank emulation)')
