"""Time standalone serial QTIP projection, including dispatch and compilation."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import ml_dtypes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_kernel import project_rotated_loop
from qtip_layout import projection_indices
p=argparse.ArgumentParser(); p.add_argument('--width',type=int,default=1024); p.add_argument('--rows',type=int,default=1024);p.add_argument('--runs',type=int,default=2);a=p.parse_args()
if a.width % 32 or a.rows % 32 or min(a.width,a.rows,a.runs) <= 0: p.error('dimensions must be positive multiples of 32')
rng=np.random.default_rng(29); words=rng.integers(0,65536,size=(1,a.width*a.rows//8),dtype=np.uint16)
table=rng.normal(0,.2,size=(1024,2)).astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.uint32)
lut=np.repeat((table[:,0] | (table[:,1] << 16)).reshape(1,1024),32,axis=0)
x=rng.normal(size=(1,a.width)).astype(np.float32)
args=(words,lut,*projection_indices(),x,a.width,a.rows//32,1)
for i in range(a.runs):
 t=time.perf_counter();y=project_rotated_loop(*args);elapsed=time.perf_counter()-t
 if not np.isfinite(np.asarray(y)).all(): raise RuntimeError('nonfinite output')
 print(f'run={i+1} elapsed_s={elapsed:.4f} rows={a.rows} width={a.width}',flush=True)
