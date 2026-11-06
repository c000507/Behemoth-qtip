"""Smoke-compile one inference operation; set NKI_ARTIFACTS_DIR externally."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops import rms_norm, swiglu, apply_rope, attention_chunk

name = sys.argv[1]
x = np.zeros((1, 128), np.float32)
specs = {
    'rms': (rms_norm, (x, np.ones_like(x), 1e-5)),
    'swiglu': (swiglu, (x, x)),
    'rope': (apply_rope, (x, np.ones((1, 64), np.float32), np.zeros((1, 64), np.float32))),
    'attention': (attention_chunk, (x, np.zeros((128, 128), np.float32), np.zeros((128, 128), np.float32), np.zeros((1, 128), np.float32), 128**-0.5)),
}
fn, args = specs[name]
try:
    fn(*args)
except ImportError as exc:
    if 'nki.runtime is not available' not in str(exc): raise
artifact = Path(__import__('os').environ['NKI_ARTIFACTS_DIR']) / 'kernel.neff'
if not artifact.is_file() or not artifact.stat().st_size:
    raise RuntimeError('No NEFF produced')
print(f'{name}: {artifact.stat().st_size} bytes')
