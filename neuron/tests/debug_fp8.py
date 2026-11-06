import sys
from pathlib import Path
import numpy as np
import ml_dtypes
import nki
import nki.language as nl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint
from backend import KernelBackend

@nki.jit
def widen(w):
    _,n=w.shape
    out=nl.ndarray((1,n),dtype=nl.float32,buffer=nl.shared_hbm)
    nl.store(out[0:1,0:n],nl.load(w[0:1,0:n],dtype=nl.float32))
    return out

codes=np.array([i for i in range(256) if i not in (127,255)],np.uint8).reshape(1,-1)
w=codes.view(ml_dtypes.float8_e4m3fn)
expected=w.astype(np.float32);actual=np.asarray(widen(w));bad=~np.isclose(actual,expected,rtol=0,atol=0,equal_nan=True)
print('FP8 widening differences:',int(bad.sum()),flush=True)
for i in np.flatnonzero(bad)[:30]:print('byte',hex(int(codes[0,i])),'expected',expected[0,i],'device',actual[0,i],flush=True)
c=Checkpoint(sys.argv[1]);v=c.projection(0,'self_attn.v_proj',4096,1024)
x=(.1*np.sin(np.arange(4096,dtype=np.float32)*.071)).reshape(1,-1)
expected=(x.astype(ml_dtypes.bfloat16).astype(np.float32) @ v.weights.astype(np.float32).T)*v.scale
actual=KernelBackend(simulate=False).linear(v,x)
bad=~np.isfinite(actual);print('nonfinite outputs:',int(bad.sum()),'CPU max abs:',float(np.max(abs(expected))),flush=True)
for i in np.flatnonzero(bad)[:8]:
 row=v.weights[i].view(np.uint8);print('row',i,'scale',float(v.scale[i]),'unique extreme codes',np.unique(row[(row & 127)>=120]).tolist(),flush=True)
