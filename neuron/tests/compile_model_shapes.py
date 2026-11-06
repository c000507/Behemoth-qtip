"""Compile a single Anubis-Mini-sized NKI kernel for Trn2, without a device.

Set NKI_ARTIFACTS_DIR to an empty directory; pass qtip-up, qtip-down, norm,
swiglu, had-chunks, or had-mix. See README for requirements.
"""
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_kernel import project_rotated_loop, hadamard_chunks, hadamard_mix
from qtip_layout import projection_indices, hadamard_indices
from ops import rms_norm, swiglu

name = sys.argv[1]
batch = 1
if name in ('qtip-up', 'qtip-down'):
    width, output = (4096, 14336) if name == 'qtip-up' else (14336, 4096)
    fn = project_rotated_loop
    args = (np.zeros((1,width*output//8), np.uint16), np.zeros((32,1024),np.uint32),
            *projection_indices(), np.zeros((1,width),np.float32), width, output//32, batch)
elif name == 'norm':
    width=4096; fn=rms_norm; args=(np.zeros((1,width),np.float32),np.ones((1,width),np.float32),1e-5)
elif name == 'swiglu':
    width=14336;fn=swiglu;args=(np.zeros((1,width),np.float32),np.zeros((1,width),np.float32))
elif name in ('had-chunks', 'had-mix'):
    width=14336; factor=28
    x=np.zeros((1,width),np.float32);scale=np.ones((1,width),np.float32)
    if name=='had-chunks':
        fn=hadamard_chunks;args=(x,scale,*hadamard_indices(width//factor),width,factor,batch,True)
    else:
        fn=hadamard_mix;args=(x,scale,np.ones((1,factor*factor),np.float32),width,factor,batch,False,False)
else:
    raise SystemExit('unknown component')
try:
    fn(*args)
except ImportError as exc:
    if 'nki.runtime is not available' not in str(exc): raise
artifact=Path(os.environ['NKI_ARTIFACTS_DIR'])/'kernel.neff'
if not artifact.is_file() or artifact.stat().st_size == 0: raise RuntimeError('NEFF missing')
print(f'{name}: compiled {artifact.stat().st_size} bytes')
