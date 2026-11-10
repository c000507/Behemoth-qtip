"""Experimental 2-bpw binary coefficients + shared BF16 matrix decoder.

Each contiguous group of 64 input weights has 128 signed binary coefficients.
W[o, g*64:(g+1)*64] ~= signs[o,g,:] @ decoder[128,64]. No codebook gather.
This is a research format, not a production checkpoint type. NumPy only; the
Neuron implementation is in binary_codec_nki.py. Fit/encode time is offline.
"""
from dataclasses import dataclass
import numpy as np
import ml_dtypes

BLOCK = 64
BITS = 128


def bf16(x):
    return np.asarray(x, dtype=ml_dtypes.bfloat16).astype(np.float32)


def _blocks(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != BLOCK or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite, nonempty [blocks,64] input')
    return x


def refine_codes(x, decoder, codes, sweeps=4):
    """Exact single-bit coordinate minimization of ||x - codes @ D||^2.

    Strictly improving flips only; ties keep their previous sign. No STE or
    hidden floating point coefficients are stored in the compressed format.
    """
    residual = x - codes @ decoder
    for sweep in range(sweeps):
        changed = 0
        for j in (range(BITS) if sweep % 2 == 0 else range(BITS - 1, -1, -1)):
            d = decoder[j]
            # Flipping b changes residual to residual + 2*b*d.
            gain = codes[:, j] * (residual @ d) + np.dot(d, d)
            flip = gain < -1e-7
            delta = np.where(flip, 2 * codes[:, j], 0)
            residual += delta[:, None] * d
            codes[:, j] -= delta
            changed += int(flip.sum())
        if not changed:
            break
    return codes


def encode(x, decoder, sweeps=8, restarts=4, seed=0):
    x = _blocks(x)
    decoder = np.asarray(decoder, np.float32)
    if decoder.shape != (BITS, BLOCK) or not np.isfinite(decoder).all():
        raise ValueError('Expected finite [128,64] decoder')
    if sweeps < 1 or restarts < 1:
        raise ValueError('sweeps and restarts must be positive')
    rng = np.random.default_rng(seed)
    best = np.empty((len(x), BITS), np.float32)
    costs = np.full(len(x), np.inf)
    for r in range(restarts):
        if r == 0:
            b = np.where(x @ decoder.T >= 0, 1., -1.).astype(np.float32)
        else:
            b = rng.choice(np.array([-1., 1.], np.float32), (len(x), BITS))
        refine_codes(x, decoder, b, sweeps)
        cost = np.square(x - b @ decoder).sum(axis=1)
        better = cost < costs
        best[better], costs[better] = b[better], cost[better]
    return best


def fit(x, iterations=20, sweeps=3, seed=0):
    """Alternating binary coordinate descent / regularized least squares.

    Train the decoder on separate blocks from those used for evaluation. The
    returned decoder is BF16-rounded, and final codes must be re-encoded with
    that stored decoder. Two orthogonal bases initialize the dictionary.
    """
    x = _blocks(x)
    if iterations < 1 or sweeps < 1:
        raise ValueError('iterations and sweeps must be positive')
    rng = np.random.default_rng(seed)
    bases = [np.linalg.qr(rng.normal(size=(BLOCK, BLOCK)))[0] for _ in range(2)]
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    d = (np.concatenate(bases) * (rms / np.sqrt(2))).astype(np.float32)
    b = encode(x, d, sweeps=sweeps, restarts=1, seed=seed)
    history = []
    for _ in range(iterations):
        gram = b.T @ b + np.eye(BITS, dtype=np.float32) * (len(x) * 1e-5)
        d = np.linalg.solve(gram, b.T @ x).astype(np.float32)
        refine_codes(x, d, b, sweeps)
        history.append(float(np.mean(np.square(x - b @ d))))
    return bf16(d), history


@dataclass
class BinaryProjection:
    # Native device layout: [input_block, output_row, packed_latent_byte].
    packed: np.ndarray
    decoder: np.ndarray

    @property
    def shape(self):
        return self.packed.shape[1], self.packed.shape[0] * BLOCK

    @property
    def storage_bytes(self):
        return self.packed.nbytes + self.decoder.nbytes

    def dense(self, device_rounding=True):
        b = unpack_codes(self.packed)
        w = b @ self.decoder.astype(np.float32)
        if device_rounding:
            w = bf16(w)
        return w.reshape(self.shape)

    def save(self, path):
        # Store BF16 as explicit uint16 bits: NumPy's .npz does not preserve
        # custom ml_dtypes metadata reliably across versions.
        np.savez(path, version=np.array(1), packed=self.packed,
                 decoder_bf16=self.decoder.view(np.uint16))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            if int(f['version']) != 1:
                raise ValueError('Unsupported binary-codec version')
            p, d = f['packed'], f['decoder_bf16']
        if p.dtype != np.uint8 or p.ndim != 3 or p.shape[2] != BITS // 8 or min(p.shape) < 1:
            raise ValueError('Invalid packed coefficients')
        if d.dtype != np.uint16 or d.shape != (BITS, BLOCK):
            raise ValueError('Invalid decoder')
        d = d.view(ml_dtypes.bfloat16)
        if not np.isfinite(d.astype(np.float32)).all():
            raise ValueError('Nonfinite decoder')
        return cls(p, d)


def pack_codes(codes):
    """[out,input_blocks,128] signs -> [input_blocks,out,16] uint8."""
    codes = np.asarray(codes)
    if codes.ndim != 3 or codes.shape[2] != BITS or not np.isin(codes, [-1, 1]).all():
        raise ValueError('Expected [out,input_blocks,128] binary signs')
    return np.ascontiguousarray(np.packbits(codes > 0, axis=-1, bitorder='little').transpose(1, 0, 2))


def unpack_codes(packed):
    bits = np.unpackbits(np.asarray(packed).transpose(1, 0, 2), axis=-1, bitorder='little')
    return bits.astype(np.float32) * 2 - 1


def quantize(weights, decoder, **encode_opts):
    w = np.asarray(weights, np.float32)
    if w.ndim != 2 or w.shape[1] % BLOCK:
        raise ValueError('Weight input width must be a multiple of 64')
    d = np.asarray(decoder, dtype=ml_dtypes.bfloat16)
    b = encode(w.reshape(-1, BLOCK), d.astype(np.float32), **encode_opts)
    return BinaryProjection(pack_codes(b.reshape(w.shape[0], -1, BITS)), d)
