"""Device-only cached versus standalone execution and timing smoke check."""
from pathlib import Path
import sys,time
import numpy as np
import ml_dtypes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nki_cache import KernelCache
from ops import fp8fn_project_loop, nvfp4_project_loop

rng=np.random.default_rng(71)
fp8lut=np.arange(256,dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(1,256)
codes=rng.choice(np.array([0,1,63,119,120,121,126,128,248,254],np.uint8),size=(1,32*64))
scale=np.ones((1,32),np.float32);x=rng.normal(size=(1,64)).astype(np.float32)
args=(codes,scale,x,fp8lut,64,32,1)
expected=fp8fn_project_loop(*args)
cache=KernelCache()
for i in range(3):
 t=time.monotonic();got=cache.run(fp8fn_project_loop,*args);dt=time.monotonic()-t
 np.testing.assert_allclose(got,expected,rtol=1e-6,atol=1e-6)
 print('cached call',i+1,'elapsed_s',round(dt,4),flush=True)
 changed=codes.copy();changed[0,0]=1 if codes[0,0]!=1 else 0
if len(cache._entries)!=1:raise AssertionError('identical signatures were not cached')
updated=(changed,scale,x,fp8lut,64,32,1)
np.testing.assert_allclose(cache.run(fp8fn_project_loop,*updated),fp8fn_project_loop(*updated),rtol=1e-6,atol=1e-6)
print('cached execution follows updated input values',flush=True)
