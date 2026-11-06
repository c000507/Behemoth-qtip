"""Validate a real checkpoint projection against an independent CPU reference.

Requires an attached Neuron device and the C++ qtip_checkpoint_oracle for
QTIP. Examples: --layer 0 --name self_attn.q_proj --in-width 4096 --out-width
4096; --layer 2 --name mlp.up_proj --in-width 4096 --out-width 14336.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint, QtipProjection, Fp8Projection, Nvfp4Projection
from backend import KernelBackend, bf16

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model');p.add_argument('--layer',type=int,required=True)
p.add_argument('--name',required=True);p.add_argument('--in-width',type=int,required=True)
p.add_argument('--out-width',type=int,required=True);p.add_argument('--oracle',default='./qtip-checkpoint-oracle')
a=p.parse_args()
print('Loading projection',a.layer,a.name,flush=True)
t=time.monotonic(); cp=Checkpoint(a.model); w=cp.projection(a.layer,a.name,a.in_width,a.out_width)
print('Loaded',type(w).__name__,'in',round(time.monotonic()-t,2),'s',flush=True)
if isinstance(w,QtipProjection):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([a.oracle,a.model,str(a.layer),a.name,d],check=True)
        x=np.fromfile(Path(d)/'x.bin',dtype=np.float32).reshape(1,-1)
        expected=np.fromfile(Path(d)/'y.bin',dtype=np.float32).reshape(1,-1)
else:
    x=(.1*np.sin(np.arange(a.in_width,dtype=np.float32)*.071)).reshape(1,-1)
    if isinstance(w,Fp8Projection):
        expected=(bf16(x) @ w.weights.astype(np.float32).T)*w.scale.reshape(1,-1)
    elif isinstance(w,Nvfp4Projection):
        pos=np.array([0,.5,1,1.5,2,3,4,6],np.float32);lut=np.concatenate((pos,-pos))
        expected=np.empty((1,a.out_width),np.float32)
        for start in range(0,a.out_width,128):
            end=min(a.out_width,start+128);packed=w.packed[start:end]
            codes=np.empty((end-start,a.in_width),np.uint8)
            codes[:,0::2]=packed & 15;codes[:,1::2]=packed >> 4
            dense=lut[codes]*np.repeat(w.scales[start:end].astype(np.float32),16,axis=1)*w.global_scale
            expected[:,start:end]=bf16(x) @ bf16(dense).T
    else: raise TypeError('Unknown projection')
print('CPU reference ready; running device projection',flush=True)
def progress(done, total):
    if done % 1024 == 0 or done == total:
        print(f'projection progress: {done}/{total} rows',flush=True)
t=time.monotonic();actual=KernelBackend(simulate=False,progress=progress).linear(w,x);elapsed=time.monotonic()-t
if not np.isfinite(actual).all():raise AssertionError('Nonfinite device output')
diff=actual.astype(np.float64)-expected.astype(np.float64)
rel=np.linalg.norm(diff)/max(np.linalg.norm(expected.astype(np.float64)),1e-30)
max_abs=np.max(np.abs(diff)); print(f'device elapsed_s={elapsed:.3f} relative_l2={rel:.8g} max_abs={max_abs:.8g}',flush=True)
if rel > .02:raise AssertionError('Projection relative L2 exceeds 0.02')
