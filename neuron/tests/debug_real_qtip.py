"""Compare real QTIP input transform and first blocks on simulator and device."""
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint
from backend import KernelBackend
from qtip_layout import projection_indices
from qtip_kernel import project_rotated_loop
model=sys.argv[1]; c=Checkpoint(model);w=c.projection(0,'self_attn.q_proj',4096,4096)
x=(.1*np.sin(np.arange(4096,dtype=np.float32)*.071)).reshape(1,-1)
a=KernelBackend(simulate=True);b=KernelBackend(simulate=False)
def compare(name,cpu,gpu):
 cpu=np.asarray(cpu,dtype=np.float64);gpu=np.asarray(gpu,dtype=np.float64)
 print(name,'relative_l2=',np.linalg.norm(cpu-gpu)/max(np.linalg.norm(cpu),1e-30),'max_abs=',np.max(np.abs(cpu-gpu)),'first=',cpu.flat[:6],gpu.flat[:6],flush=True)
print('Running input rotation in simulator',flush=True)
t=time.monotonic();ref=a.transform(x,w.su,True,True,True);print('sim seconds',time.monotonic()-t,flush=True)
print('Running input rotation on device',flush=True)
t=time.monotonic();val=b.transform(x,w.su,True,True,True);print('device seconds',time.monotonic()-t,flush=True);compare('input transform',ref,val)
# 32 output rows and one 32-column block, compared with the identical codebook
# and rotated input. This isolates the packed kernel from full-width FWHT.
for col in (0,31,63,127):
 packed=w.trellis[:,col*128:(col+1)*128].copy();q=ref[:,col*32:(col+1)*32].copy()
 args=(packed,w.lut,*projection_indices(),q,32,1,1)
 compare('block '+str(col),a.run(project_rotated_loop,*args),b.run(project_rotated_loop,*args))
# Compare a full K accumulation for widely separated output block rows.
for row in (0,63,127):
    words_per_row=4096*4
    packed=w.trellis[:,row*words_per_row:(row+1)*words_per_row].copy()
    args=(packed,w.lut,*projection_indices(),ref,4096,1,1)
    print('Running full-row accumulation',row,flush=True)
    compare('row '+str(row),a.run(project_rotated_loop,*args),b.run(project_rotated_loop,*args))
# The output transform uses unrounded FP32 results.
z=np.random.default_rng(59).normal(0,1,(1,4096)).astype(np.float32)
compare('output transform',a.transform(z,w.sv,False,False,False),b.transform(z,w.sv,False,False,False))
for scale_name,scale in [('ones',np.ones((1,4096),np.float32)),('sv',w.sv)]:
 for before,rounded in [(False,False),(False,True),(True,False),(True,True)]:
    print('Testing output mode',scale_name,before,rounded,flush=True)
    compare('mode '+scale_name+' '+str(before)+' '+str(rounded),a.transform(z,scale,False,before,rounded),b.transform(z,scale,False,before,rounded))
