"""Compile/run the packed binary decoder projection with a bounded host reference.

New standalone NKI: --mode compile produces kernel.neff without a device;
--mode device requires NKI runtime + Neuron hardware and reports warm median
wall time (not profiler engine time). Legacy NKI: --mode trace emits graph.neff
for cpp/layer_bench.cpp, just like bench_qtip_nki.py. Synthetic weights measure
execution only; accuracy fitting is compare_binary_qtip.py.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import ml_dtypes

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binary_codec_nki import binary_linear


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--out-local',type=int,default=14336)
    p.add_argument('--in-local',type=int,default=8192)
    p.add_argument('--tokens',type=int,default=1)
    p.add_argument('--target',choices=['trn1','trn2'],default='trn1')
    p.add_argument('--mode',choices=['compile','trace','device'],default='compile')
    p.add_argument('--repeats',type=int,default=20)
    a=p.parse_args()
    if a.out_local<=0 or a.out_local%128 or a.in_local<=0 or a.in_local%64 or not 1<=a.tokens<=128 or a.repeats<1:
        p.error('Need output multiple of 128, input multiple of 64, tokens 1..128, repeats >= 1')
    a.out.mkdir(parents=True,exist_ok=True)
    os.environ['NKI_ARTIFACTS_DIR']=str(a.out.resolve())
    os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']=a.target
    rng=np.random.default_rng(11)
    packed=rng.integers(0,256,(a.in_local//64,a.out_local,16),dtype=np.uint8)
    d=(rng.normal(size=(128,64))/8).astype(ml_dtypes.bfloat16)
    x=rng.normal(size=(a.in_local,a.tokens)).astype(ml_dtypes.bfloat16)
    result=dict(target=a.target,shape=[a.out_local,a.in_local],tokens=a.tokens,
                packed_bytes=packed.nbytes,decoder_bytes=d.nbytes,mode=a.mode)
    if a.mode=='trace':
        import torch
        import torch_neuronx
        inputs=(torch.from_numpy(packed),torch.from_numpy(d.view(np.int16)).view(torch.bfloat16),
                torch.from_numpy(x.view(np.int16)).view(torch.bfloat16))
        torch_neuronx.trace(lambda p,d,x:binary_linear(p,d,x),inputs,compiler_workdir=str(a.out))
    else:
        try:
            got=binary_linear(packed,d,x)
        except ImportError as e:
            if a.mode!='compile' or 'nki.runtime is not available' not in str(e):
                raise
        if a.mode=='device':
            # Check every output against float64 accumulation, with BF16 rounding
            # of reconstructed weights. Only materialize one 64-column slice.
            expected=np.zeros((a.tokens,a.out_local),np.float64)
            for g in range(a.in_local//64):
                b=np.unpackbits(packed[g],axis=1,bitorder='little').astype(np.float32)*2-1
                w=(b@d.astype(np.float32)).astype(ml_dtypes.bfloat16).astype(np.float64)
                expected+=x[g*64:(g+1)*64].astype(np.float64).T@w.T
            rel=float(np.linalg.norm(got-expected)/np.linalg.norm(expected))
            if rel>2e-3:
                raise RuntimeError(f'Device numerical check failed: {rel}')
            for _ in range(3): binary_linear(packed,d,x)
            times=[]
            for _ in range(a.repeats):
                start=time.perf_counter(); binary_linear(packed,d,x)
                times.append((time.perf_counter()-start)*1000)
            result.update(relative_l2=rel,median_wall_ms=float(np.median(times)),times_ms=times,
                          timing_scope='standalone host call including runtime input/output handling; use libnrt for resident timing')
    artifact=a.out/('graph.neff' if a.mode=='trace' else 'kernel.neff')
    if not artifact.is_file() or not artifact.stat().st_size:
        raise RuntimeError(f'Missing compiled artifact: {artifact}')
    result['neff_bytes']=artifact.stat().st_size
    (a.out/'binary_bench.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
