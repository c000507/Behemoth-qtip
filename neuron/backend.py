"""Experimental NKI kernel dispatcher and serial Llama attention baseline."""
from functools import lru_cache
from pathlib import Path
import re

import ml_dtypes
import numpy as np
import nki

from checkpoint import QtipProjection, Fp8Projection, Nvfp4Projection
from ops import attention_chunk, fp8_project_loop, nvfp4_project_loop
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


class KernelBackend:
    def __init__(self, simulate=True):
        self.simulate = simulate
        self.index = projection_indices()

    def run(self, kernel, *args):
        return nki.simulate(kernel)(*args) if self.simulate else kernel(*args)

    def transform(self, x, scale, transpose, scale_before, round_bf16):
        batch, width = x.shape
        for factor in (28, 12):
            p = width // factor
            if width % factor == 0 and p >= 2 and not (p & (p - 1)):
                partners, signs = hadamard_indices(p)
                inner = self.run(hadamard_chunks, x.reshape(1, -1), scale, partners, signs,
                                 width, factor, batch, scale_before)
                matrix = hadamard_matrix(factor)
                if transpose:
                    matrix = matrix.T
                matrix = np.ascontiguousarray(matrix, dtype=np.float32).reshape(1, -1)
                return self.run(hadamard_mix, inner, scale, matrix, width, factor, batch,
                                scale_before, round_bf16).reshape(batch, width)
        kernel, tables = transform_tables(width, transpose)
        return self.run(kernel, x, scale, *tables, scale_before, round_bf16)

    def linear(self, projection, x, round_z_bf16=False):
        x = np.asarray(x, dtype=np.float32)
        if isinstance(projection, QtipProjection):
            batch, width = x.shape
            if width != projection.width:
                raise ValueError('QTIP input width mismatch')
            transformed = self.transform(x, projection.su, True, True, True)
            rotated = self.run(project_rotated_loop, projection.trellis, projection.lut, *self.index,
                               transformed.reshape(1, -1), width, projection.out // 32, batch)
            z = rotated.reshape(batch, projection.out)
            if round_z_bf16:
                z = bf16(z)
            return self.transform(z, projection.sv, False, False, False)
        if isinstance(projection, Nvfp4Projection):
            batch, width = x.shape
            if width != projection.width:
                raise ValueError('NVFP4 input width mismatch')
            ix = (np.arange(16, dtype=np.uint32) // 2).reshape(1, 16)
            shifts = ((np.arange(16, dtype=np.uint32) % 2) * 4).reshape(1, 16)
            positive = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
            codebook = np.concatenate((positive, -positive)).reshape(1, 16)
            return self.run(nvfp4_project_loop, projection.packed.reshape(1,-1),
                            projection.scales.reshape(1,-1), bf16(x).reshape(1,-1),
                            ix, shifts, codebook, projection.global_scale, width,
                            projection.out, batch).reshape(batch, projection.out)
        if isinstance(projection, Fp8Projection):
            batch, width = x.shape
            out, w = projection.weights.shape
            if width != w:
                raise ValueError('FP8 input width mismatch')
            return self.run(fp8_project_loop, projection.weights.reshape(1, -1), projection.scale.reshape(1, -1),
                            bf16(x).reshape(1, -1), width, out, batch).reshape(batch, out)
        raise TypeError('Unsupported projection')

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
