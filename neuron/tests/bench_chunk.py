"""Time one resident NVFP4 projection launch at several chunk row sizes.

Checks that larger chunks still execute correctly on device (the full
14,336-row launch previously hit a runtime timeout) and reports per-launch
and total wall times for the whole projection.
"""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint
from backend import KernelBackend

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model'); p.add_argument('--layer', type=int, default=2)
p.add_argument('--name', default='mlp.up_proj'); p.add_argument('--in-width', type=int, default=4096)
p.add_argument('--out-width', type=int, default=14336)
p.add_argument('--chunks', default='256,512,1024,2048')
a = p.parse_args()
projection = Checkpoint(a.model).projection(a.layer, a.name, a.in_width, a.out_width)
x = (0.1 * np.sin(np.arange(a.in_width, dtype=np.float32) * .071)).reshape(1, -1)
reference = None
for chunk in [int(c) for c in a.chunks.split(',')]:
    backend = KernelBackend(simulate=False, projection_chunk_rows=chunk)
    started = time.monotonic()
    result = backend.linear(projection, x)
    cold = time.monotonic() - started
    started = time.monotonic()
    result = backend.linear(projection, x)
    warm = time.monotonic() - started
    if reference is None:
        reference = result
        error = 0.0
    else:
        error = float(np.max(np.abs(result - reference)))
    print(f'chunk={chunk} launches={(a.out_width + chunk - 1)//chunk} cold_s={cold:.2f} '
          f'warm_s={warm:.2f} max_abs_diff_vs_256={error:.3g}', flush=True)
