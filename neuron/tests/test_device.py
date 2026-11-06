"""Compare standalone NKI kernels on an attached Neuron device with simulation.

Run from a matching Neuron SDK environment. This test invokes real device
execution; do not set NKI_SIMULATOR. It checks small tiles, not performance.
"""
import os
from pathlib import Path
import sys
import numpy as np
import ml_dtypes
import nki

if os.environ.get('NKI_SIMULATOR') == '1':
    raise SystemExit('NKI_SIMULATOR must be disabled for device validation')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_kernel import decode_block, project_rotated_loop, hadamard_chunks, hadamard_mix
from qtip_layout import indices, projection_indices, hadamard_indices
from ops import rms_norm, swiglu, apply_rope, attention_chunk, fp8_project_loop, nvfp4_project_loop


def compare(name, kernel, args, exact=False):
    expected = np.asarray(nki.simulate(kernel)(*args))
    got = np.asarray(kernel(*args))
    if exact:
        np.testing.assert_array_equal(got, expected, err_msg=name)
    else:
        np.testing.assert_allclose(got, expected, rtol=.02, atol=.002, err_msg=name)
    print(name, 'device matches simulator', flush=True)


def main():
    rng = np.random.default_rng(529)
    words = rng.integers(0, 65536, size=(1, 512), dtype=np.uint16)
    table = rng.normal(0, .2, size=(1024, 2)).astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.uint32)
    lut = (table[:,0] | (table[:,1] << 16)).reshape(1,1024)
    compare('trellis decode', decode_block, (words[:, :128].copy(), lut, *indices()), True)
    x = rng.normal(0,.1,size=(1,64)).astype(np.float32)
    pi = projection_indices(); rep = np.repeat(lut,32,axis=0)
    compare('QTIP projection', project_rotated_loop, (words, rep, *pi, x, 64, 2, 1))
    w = rng.normal(0,.2,size=(32,64)).astype(ml_dtypes.float8_e4m3fn)
    scale = rng.uniform(.1,1,size=(1,32)).astype(np.float32)
    compare('FP8 projection', fp8_project_loop, (w.reshape(1,-1),scale,x,64,32,1))
    packed = rng.integers(0,256,size=(1,32*32),dtype=np.uint8)
    bs = rng.uniform(.1,1,size=(1,32*4)).astype(ml_dtypes.float8_e4m3fn)
    ix = (np.arange(16,dtype=np.uint32)//2).reshape(1,16)
    shifts = ((np.arange(16,dtype=np.uint32)%2)*4).reshape(1,16)
    positive = np.array([0,.5,1,1.5,2,3,4,6],np.float32)
    codebook = np.concatenate((positive,-positive)).reshape(1,16)
    compare('NVFP4 projection', nvfp4_project_loop, (packed,bs,x,ix,shifts,codebook,.17,64,32,1))
    compare('RMSNorm', rms_norm, (x,np.ones((1,64),np.float32),1e-5))
    compare('SwiGLU', swiglu, (x,x))
    compare('RoPE', apply_rope, (x,np.ones((1,32),np.float32),np.zeros((1,32),np.float32)))
    q = x[:,:32].copy(); k = rng.normal(0,.2,size=(32,32)).astype(np.float32)
    v = rng.normal(0,.2,size=(32,32)).astype(np.float32)
    compare('attention tile', attention_chunk, (q,k,v,np.zeros((1,32),np.float32),32**-.5))
    width=384; factor=12
    x = rng.normal(0,.1,size=(1,width)).astype(np.float32)
    scale = np.ones((1,width),np.float32); matrix = np.ones((1,factor*factor),np.float32)
    compare('Hadamard chunks', hadamard_chunks, (x,scale,*hadamard_indices(32),width,factor,1,True))
    compare('Hadamard mix', hadamard_mix, (x,scale,matrix,width,factor,1,False,False))

if __name__ == '__main__': main()
