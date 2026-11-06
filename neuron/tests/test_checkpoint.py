"""Checkpoint format tests with generated QTIP and FP8 safetensors shards."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import Checkpoint, QtipProjection, Fp8Projection, Nvfp4Projection
from backend import KernelBackend, bf16
import ml_dtypes


def write_checkpoint(directory, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, array) in tensors.items():
        raw = np.asarray(array).tobytes()
        header[name] = {'dtype': dtype, 'shape': list(array.shape),
                        'data_offsets': [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    with (directory / 'layer.safetensors').open('wb') as f:
        f.write(len(encoded).to_bytes(8, 'little')); f.write(encoded); f.write(payload)
    (directory / 'config.json').write_text('{}')
    (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'layer.safetensors' for k in tensors}}))


def main():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        rng = np.random.default_rng(33)
        trellis = rng.integers(-32768, 32768, size=(16, 32), dtype=np.int16)
        # Use FP16 values exactly representable in BF16 for an independently
        # reconstructible sign-folded table.
        tlut = (rng.integers(-128, 129, size=(512, 2)) / 128).astype(np.float16)
        prefix = 'model.layers.0.self_attn.q_proj.'
        fp = 'model.layers.0.self_attn.k_proj.'
        nv = 'model.layers.0.mlp.up_proj.'
        weights = np.arange(32*64, dtype=np.uint16).astype(np.uint8).reshape(32, 64) % 0x70
        codes = rng.integers(0, 16, size=(32, 64), dtype=np.uint8)
        nvpacked = codes[:, ::2] | (codes[:, 1::2] << 4)
        nvscales = np.full((32,4), .25, dtype=ml_dtypes.float8_e4m3fn)
        tensors = {nv+'weight_packed': ('U8', nvpacked),
                   nv+'weight_scale': ('F8_E4M3', nvscales.view(np.uint8)),
                   nv+'weight_global_scale': ('F32', np.array(.75, np.float32)),
                   prefix+'trellis' : ('I16', trellis), prefix+'tlut': ('F16', tlut),
                   prefix+'SU': ('F32', np.ones(64, np.float32)), prefix+'SV': ('F32', np.ones(64, np.float32)),
                   prefix+'rcp': ('I64', np.array(0, np.int64)), prefix+'tp_rank': ('I64', np.array(0, np.int64)),
                   fp+'weight': ('F8_E4M3', weights), fp+'weight_scale': ('F32', np.ones(32, np.float32))}
        write_checkpoint(p, tensors)
        cp = Checkpoint(p)
        q = cp.projection(0, 'self_attn.q_proj', 64, 64)
        assert isinstance(q, QtipProjection) and q.trellis.shape == (1, 512) and q.lut.shape == (32, 1024)
        np.testing.assert_array_equal(q.trellis.view(np.int16).reshape(16, 32), trellis)
        cb = tlut.astype(np.float32)
        packed = q.lut[0]
        c0 = ((packed & 65535) << 16).view(np.float32)
        c1 = (packed & 0xffff0000).view(np.float32)
        np.testing.assert_array_equal(c0[:512], cb[:, 0]); np.testing.assert_array_equal(c0[512:], -cb[:, 0])
        np.testing.assert_array_equal(c1[:512], cb[:, 1]); np.testing.assert_array_equal(c1[512:], cb[:, 1])
        k = cp.projection(0, 'self_attn.k_proj', 64, 32)
        assert isinstance(k, Fp8Projection)
        np.testing.assert_array_equal(k.weights.view(np.uint8), weights)
        nvproj = cp.projection(0, 'mlp.up_proj', 64, 32)
        assert isinstance(nvproj, Nvfp4Projection)
        x = rng.normal(size=(1,64)).astype(np.float32)
        got = KernelBackend().linear(nvproj, x)
        pos = np.array([0,.5,1,1.5,2,3,4,6],np.float32)
        lut = np.concatenate((pos,-pos))
        dense = bf16(lut[codes] * .25 * .75)
        np.testing.assert_allclose(got, bf16(x) @ dense.T, rtol=3e-5, atol=2e-6)
        bad = p / 'model.safetensors.index.json'
        index = json.loads(bad.read_text()); index['weight_map'][prefix+'trellis'] = '../outside'
        bad.write_text(json.dumps(index))
        try: Checkpoint(p).tensor(prefix+'trellis')
        except ValueError: pass
        else: raise AssertionError('Unsafe checkpoint path accepted')
        print('Checkpoint QTIP, FP8, NVFP4 and path validation passed')


if __name__ == '__main__': main()
