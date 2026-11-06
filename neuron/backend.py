"""Experimental NKI kernel dispatcher and serial Llama attention baseline."""
from functools import lru_cache
from pathlib import Path
import re

import ml_dtypes
import numpy as np
import nki

from checkpoint import QtipProjection, Fp8Projection, Nvfp4Projection
from ops import attention_chunk, fp8_project_loop, nvfp4_project_loop, scale_rows, fp8fn_project_loop
from qtip_kernel import project_rotated_loop, transform_pow2, transform_factored, hadamard_chunks, hadamard_mix
from qtip_layout import projection_indices, hadamard_indices, factored_hadamard_indices


def bf16(x):
    return np.asarray(x, dtype=np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


@lru_cache(maxsize=2)
def hadamard_matrix(k):
    text = (Path(__file__).resolve().parents[1] / 'runtime/src/qtip_tables.hpp').read_text()
    match = re.search(r'kHad' + str(k) + r'\[\d+\] = \{([^}]*)\}', text)
    if not match:
        raise ValueError('Missing QTIP Hadamard table')
    values = np.fromstring(match.group(1), dtype=np.int8, sep=',')
    if values.size != k * k:
        raise ValueError('Invalid QTIP Hadamard table')
    return values.reshape(k, k)


@lru_cache(maxsize=32)
def transform_tables(width, transpose):
    for k in (28, 12):
        if width % k == 0 and width // k >= 2 and not ((width // k) & (width // k - 1)):
            return transform_factored, factored_hadamard_indices(width, hadamard_matrix(k), transpose)
    if width >= 2 and not (width & (width - 1)):
        return transform_pow2, hadamard_indices(width)
    raise ValueError(f'Unsupported QTIP Hadamard dimension: {width}')


@lru_cache(maxsize=16)
def _wide_tables(p):
    return hadamard_indices(p)


@lru_cache(maxsize=16)
def _mix_matrix(factor, transpose):
    matrix = hadamard_matrix(factor)
    if transpose:
        matrix = matrix.T
    return np.ascontiguousarray(matrix, dtype=np.float32).reshape(1, -1)


class KernelBackend:
    def __init__(self, simulate=True, projection_chunk_rows=256, progress=None, cache_kernels=True):
        if projection_chunk_rows < 1:
            raise ValueError('Projection chunk size must be positive')
        self.simulate = simulate
        self.kernel_cache = None
        if not simulate and cache_kernels:
            from nki_cache import KernelCache
            self.kernel_cache = KernelCache()
        self.projection_chunk_rows = projection_chunk_rows
        self.progress = progress
        self.index = projection_indices()
        self.fp8_lut = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(1, 256)
        self.nvfp4_index = (np.arange(16, dtype=np.uint32) // 2).reshape(1, 16)
        self.nvfp4_shifts = ((np.arange(16, dtype=np.uint32) % 2) * 4).reshape(1, 16)
        positive = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
        self.nvfp4_codebook = np.concatenate((positive, -positive)).reshape(1, 16)
        self._dense_chunks = {}

    def run(self, kernel, *args, resident=()):
        if self.simulate:
            return nki.simulate(kernel)(*args)
        if self.kernel_cache is not None:
            return self.kernel_cache.run(kernel, *args, resident=resident)
        return kernel(*args)

    def transform(self, x, scale, transpose, scale_before, round_bf16):
        batch, width = x.shape
        for factor in (28, 12):
            p = width // factor
            if width % factor == 0 and p >= 2 and not (p & (p - 1)):
                partners, signs = _wide_tables(p)
                # Materialize the input by tensor scaling before indexed gathers;
                # Trn1 can otherwise return incorrect output for a loaded tile.
                input_scale = scale if scale_before else np.ones_like(scale)
                inner = self.run(hadamard_chunks, x.reshape(1, -1), input_scale, partners, signs,
                                 width, factor, batch, True, resident=(2, 3))
                matrix = _mix_matrix(factor, transpose)
                return self.run(hadamard_mix, inner, scale, matrix, width, factor, batch,
                                scale_before, round_bf16, resident=(2,)).reshape(batch, width)
        kernel, tables = transform_tables(width, transpose)
        if scale_before:
            return self.run(kernel, x, scale, *tables, True, round_bf16, resident=(2, 3, 4, 5))
        unscaled = self.run(kernel, x, np.ones_like(scale), *tables, True, False, resident=(2, 3, 4, 5))
        return self.run(scale_rows, unscaled, scale, round_bf16)

    def linear(self, projection, x, round_z_bf16=False):
        x = np.asarray(x, dtype=np.float32)
        if isinstance(projection, QtipProjection):
            batch, width = x.shape
            if width != projection.width:
                raise ValueError('QTIP input width mismatch')
            transformed = self.transform(x, projection.su, True, True, True)
            rotated = self.run(project_rotated_loop, projection.trellis, projection.lut, *self.index,
                               transformed.reshape(1, -1), width, projection.out // 32, batch,
                               resident=(0, 1, 2, 3, 4, 5))
            z = rotated.reshape(batch, projection.out)
            if round_z_bf16:
                z = bf16(z)
            return self.transform(z, projection.sv, False, False, False)
        if isinstance(projection, Nvfp4Projection):
            batch, width = x.shape
            if width != projection.width:
                raise ValueError('NVFP4 input width mismatch')
            outputs = []
            for start in range(0, projection.out, self.projection_chunk_rows):
                stop = min(start + self.projection_chunk_rows, projection.out)
                rows = stop - start
                packed, scales = self._nvfp4_chunk(projection, start, rows)
                partial = self.run(nvfp4_project_loop, packed, scales, bf16(x).reshape(1, -1),
                                   self.nvfp4_index, self.nvfp4_shifts, self.nvfp4_codebook, self.fp8_lut,
                                   projection.global_scale, width, rows, batch,
                                   resident=(0, 1, 3, 4, 5, 6))
                outputs.append(partial.reshape(batch, rows))
                if self.progress:
                    self.progress(stop, projection.out)
            return np.concatenate(outputs, axis=1)
        if isinstance(projection, Fp8Projection):
            batch, width = x.shape
            out, w = projection.weights.shape
            if width != w:
                raise ValueError('FP8 input width mismatch')
            outputs = []
            for start in range(0, out, self.projection_chunk_rows):
                stop = min(start + self.projection_chunk_rows, out)
                rows = stop - start
                weights, scale = self._fp8_chunk(projection, start, rows)
                partial = self.run(fp8fn_project_loop, weights, scale, bf16(x).reshape(1, -1),
                                   self.fp8_lut, width, rows, batch, resident=(0, 1, 3))
                outputs.append(partial.reshape(batch, rows))
                if self.progress:
                    self.progress(stop, out)
            return np.concatenate(outputs, axis=1)
        raise TypeError('Unsupported projection')

    def _nvfp4_chunk(self, projection, start, rows):
        chunks = getattr(projection, '_nvfp4_chunks', None)
        if chunks is None:
            chunks = projection._nvfp4_chunks = {}
        chunk = chunks.get((start, rows))
        if chunk is None:
            chunk = (projection.packed[start:start+rows].reshape(1, -1).copy(),
                     projection.scales[start:start+rows].view(np.uint8).reshape(1, -1).copy())
            chunks[(start, rows)] = chunk
        return chunk

    def _fp8_chunk(self, projection, start, rows):
        chunks = getattr(projection, '_fp8_chunks', None)
        if chunks is None:
            chunks = projection._fp8_chunks = {}
        chunk = chunks.get((start, rows))
        if chunk is None:
            chunk = (projection.weights[start:start+rows].view(np.uint8).reshape(1, -1).copy(),
                     projection.scale[start:start+rows].reshape(1, -1).copy())
            chunks[(start, rows)] = chunk
        return chunk

    def _dense_chunk(self, weights, start, rows):
        chunks = self._dense_chunks.get(id(weights))
        if chunks is None or chunks[0] is not weights:
            chunks = (weights, {})
            self._dense_chunks[id(weights)] = chunks
        chunk = chunks[1].get((start, rows))
        if chunk is None:
            chunk = (weights[start:start+rows].reshape(1, -1).copy(),
                     np.ones((1, rows), np.float32))
            chunks[1][(start, rows)] = chunk
        return chunk

    def dense(self, weights, x):
        """Correctness-first BF16/F32 dense projection in bounded launches."""
        batch, width = x.shape
        out, weight_width = weights.shape
        if width != weight_width:
            raise ValueError('Dense input width mismatch')
        outputs = []
        for start in range(0, out, self.projection_chunk_rows):
            stop = min(start + self.projection_chunk_rows, out)
            rows = stop - start
            chunk, ones = self._dense_chunk(weights, start, rows)
            partial = self.run(fp8_project_loop, chunk, ones, bf16(x).reshape(1, -1),
                               width, rows, batch, resident=(0, 1))
            outputs.append(partial.reshape(batch, rows))
        return np.concatenate(outputs, axis=1)

    def attention(self, query, keys, values):
        """GQA decode for one token. keys/values are [time,kv_heads,dim]."""
        qheads, dim = query.shape
        if keys.ndim != 3 or values.shape != keys.shape or keys.shape[0] < 1 or keys.shape[2] != dim:
            raise ValueError('Invalid KV cache')
        kvheads = keys.shape[1]
        if qheads % kvheads or dim > 128:
            raise ValueError('Invalid GQA head geometry')
        result = np.empty_like(query, dtype=np.float32)
        for h in range(qheads):
            kh = h // (qheads // kvheads)
            total_m, total_l = -np.inf, 0.0
            numerator = np.zeros(dim, np.float64)
            for start in range(0, keys.shape[0], 128):
                k = np.ascontiguousarray(keys[start:start+128, kh, :], dtype=np.float32)
                v = np.ascontiguousarray(values[start:start+128, kh, :], dtype=np.float32)
                mask = np.zeros((1, len(k)), np.float32)
                stats = self.run(attention_chunk, query[h:h+1], k, v, mask, dim ** -0.5)[0]
                m, den = float(stats[dim]), float(stats[dim+1])
                next_m = max(total_m, m)
                a, b = np.exp(total_m - next_m), np.exp(m - next_m)
                numerator = numerator * a + stats[:dim].astype(np.float64) * b
                total_l = total_l * a + den * b
                total_m = next_m
            result[h] = numerator / total_l
        return result
