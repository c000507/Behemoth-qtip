"""CPU check of the in-graph QTIP projection (decode + RHT) against the C++ qtip_checkpoint_oracle on a real layer.

usage: test_qtip_linear_graph.py MODEL_DIR ORACLE [layer] [projection]   (power-of-two widths only)
"""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint
from trace_layer import qtip_linear

model, oracle = sys.argv[1], sys.argv[2]
layer = int(sys.argv[3]) if len(sys.argv) > 3 else 0
name = sys.argv[4] if len(sys.argv) > 4 else 'self_attn.q_proj'
cp = Checkpoint(model)
width = out = int(cp.config['hidden_size'])
proj = cp.projection(layer, name, width, out)
base = f'model.layers.{layer}.{name}.'
tlut = cp.tensor(base + 'tlut').astype(np.float32)
trellis = proj.trellis.reshape(out // 32, width // 32, 32, 2, 2).view(np.int16)
with tempfile.TemporaryDirectory() as d:
    subprocess.run([oracle, model, str(layer), name, d], check=True)
    x = np.fromfile(Path(d) / 'x.bin', dtype=np.float32).reshape(1, -1)
    expected = np.fromfile(Path(d) / 'y.bin', dtype=np.float32).reshape(1, -1)
y = qtip_linear(torch.from_numpy(x), torch.from_numpy(proj.su), torch.from_numpy(proj.sv),
                torch.from_numpy(trellis), torch.from_numpy(tlut), out, width).numpy()
rel = np.linalg.norm(y - expected) / np.linalg.norm(expected)
print(f'relative_l2={rel:.3e} max_abs={np.abs(y - expected).max():.3e}')
assert rel < 1e-2, 'in-graph QTIP projection does not match the C++ reference'
print('in-graph QTIP projection matches the C++ oracle')
