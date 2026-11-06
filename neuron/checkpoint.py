"""Read Behemoth QTIP/FP8 tensors for the experimental Neuron backend.

Tensor data is read on demand, so opening a model index does not materialize
its weights. Tensor-parallel checkpoint shards are not yet supported.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import struct

import ml_dtypes
import numpy as np

_DTYPES = {
    'I16': np.dtype('<i2'), 'I64': np.dtype('<i8'), 'U8': np.dtype('u1'),
    'F16': np.dtype('<f2'), 'F32': np.dtype('<f4'),
    'BF16': np.dtype(ml_dtypes.bfloat16), 'F8_E4M3': np.dtype(ml_dtypes.float8_e4m3fn),
}


@dataclass
class QtipProjection:
    trellis: np.ndarray  # uint16 [1,(out/32)*(in/32)*128]
    lut: np.ndarray      # uint32 [32,1024], replicated for NKI partition gathers
    su: np.ndarray       # float32 [1,in]
    sv: np.ndarray       # float32 [1,out]
    width: int
    out: int


@dataclass
class Nvfp4Projection:
    packed: np.ndarray
    scales: np.ndarray
    global_scale: float
    width: int
    out: int


@dataclass
class Fp8Projection:
    weights: np.ndarray  # float8_e4m3fn [out,in]
    scale: np.ndarray    # float32 [out]


class Checkpoint:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        with (self.root / 'config.json').open() as f:
            self.config = json.load(f)
        with (self.root / 'model.safetensors.index.json').open() as f:
            self.index = json.load(f)['weight_map']
        self._headers = {}

    def _entry(self, name):
        if name not in self.index:
            raise KeyError(f'Missing checkpoint tensor: {name}')
        path = (self.root / self.index[name]).resolve()
        if self.root not in path.parents:
            raise ValueError('Checkpoint index contains a path outside its directory')
        if path not in self._headers:
            with path.open('rb') as f:
                size = path.stat().st_size
                raw = f.read(8)
                if len(raw) != 8:
                    raise ValueError('Invalid safetensors header')
                length = struct.unpack('<Q', raw)[0]
                if length > min(size - 8, 100_000_000):
                    raise ValueError('Invalid safetensors header length')
                header = json.loads(f.read(length))
            self._headers[path] = (8 + length, size, header)
        start, size, header = self._headers[path]
        if name not in header:
            raise ValueError(f'Index refers to absent tensor: {name}')
        entry = header[name]
        dtype = _DTYPES.get(entry['dtype'])
        if dtype is None:
            raise ValueError(f'Unsupported tensor dtype {entry["dtype"]}: {name}')
        shape = entry['shape']
        if not isinstance(shape, list) or any(not isinstance(d, int) or d < 0 for d in shape):
            raise ValueError(f'Invalid tensor shape: {name}')
        lo, hi = entry['data_offsets']
        count = int(np.prod(shape, dtype=object))
        if not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi < lo or start + hi > size or hi - lo != count * dtype.itemsize:
            raise ValueError(f'Invalid tensor offsets: {name}')
        return path, start + lo, hi - lo, dtype, tuple(shape)

    def tensor(self, name):
        path, start, size, dtype, shape = self._entry(name)
        with path.open('rb') as f:
            f.seek(start)
            data = f.read(size)
        if len(data) != size:
            raise ValueError(f'Truncated tensor: {name}')
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    def projection(self, layer, name, width, out):
        if width <= 0 or out <= 0 or width % 32 or out % 32:
            raise ValueError('Projection dimensions must be positive multiples of 32')
        base = f'model.layers.{layer}.{name}.'
        has_qtip = base + 'trellis' in self.index
        has_fp8 = base + 'weight' in self.index
        has_nvfp4 = base + 'weight_packed' in self.index
        if sum((has_qtip, has_fp8, has_nvfp4)) != 1:
            raise ValueError(f'Missing or ambiguous projection format: {base}')
        if has_nvfp4:
            packed = self.tensor(base + 'weight_packed')
            scales = self.tensor(base + 'weight_scale')
            global_scale = self.tensor(base + 'weight_global_scale').astype(np.float32).reshape(-1)
            if packed.dtype != _DTYPES['U8'] or packed.shape != (out, width // 2) or scales.dtype != _DTYPES['F8_E4M3'] or scales.shape != (out, width // 16):
                raise ValueError(f'Invalid NVFP4 projection: {base}')
            if global_scale.size != 1 or not np.isfinite(global_scale[0]) or global_scale[0] <= 0 or not np.isfinite(scales.astype(np.float32)).all():
                raise ValueError(f'Invalid NVFP4 scale: {base}')
            return Nvfp4Projection(packed, scales, float(global_scale[0]), width, out)
        if has_qtip == has_fp8:
            raise ValueError(f'Missing or ambiguous projection format: {base}')
        if has_fp8:
            weight = self.tensor(base + 'weight')
            source_scale = self.tensor(base + 'weight_scale')
            if source_scale.dtype not in (_DTYPES['BF16'], _DTYPES['F32']):
                raise ValueError(f'Invalid FP8 scale dtype: {base}')
            scale = source_scale.astype(np.float32).reshape(-1)
            if weight.dtype != _DTYPES['F8_E4M3'] or weight.shape != (out, width) or scale.shape != (out,):
                raise ValueError(f'Invalid FP8 projection: {base}')
            if not np.isfinite(weight.astype(np.float32)).all() or not np.isfinite(scale).all() or (scale < 0).any():
                raise ValueError(f'Invalid FP8 values: {base}')
            return Fp8Projection(weight, scale)
        words = self.tensor(base + 'trellis')
        su = self.tensor(base + 'SU').astype(np.float32).reshape(-1)
        sv = self.tensor(base + 'SV').astype(np.float32).reshape(-1)
        tlut = self.tensor(base + 'tlut')
        if words.dtype != _DTYPES['I16'] or words.shape != ((out // 16) * (width // 16), 32):
            raise ValueError(f'Invalid QTIP trellis: {base}')
        if su.shape != (width,) or sv.shape != (out,) or tlut.dtype != _DTYPES['F16'] or tlut.shape != (512, 2):
            raise ValueError(f'Invalid QTIP scale or LUT shape: {base}')
        if not np.isfinite(su).all() or not np.isfinite(sv).all() or not np.isfinite(tlut).all():
            raise ValueError(f'Nonfinite QTIP data: {base}')
        rcp = self.tensor(base + 'rcp')
        tp_rank = self.tensor(base + 'tp_rank')
        if rcp.dtype != _DTYPES['I64'] or rcp.size != 1 or int(rcp.reshape(-1)[0]) != 0 or tp_rank.dtype != _DTYPES['I64'] or tp_rank.size != 1:
            raise ValueError(f'Unsupported QTIP tensor-parallel checkpoint: {base}')
        vals = tlut.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.uint32)
        first = vals[:, 0]
        second = vals[:, 1] << 16
        packed = first | second
        signed = (first ^ 0x8000) | second
        lut = np.concatenate((packed, signed)).reshape(1, 1024)
        return QtipProjection(words.view(np.uint16).reshape(1, -1), np.repeat(lut, 32, axis=0),
                              su.reshape(1, -1), sv.reshape(1, -1), width, out)
