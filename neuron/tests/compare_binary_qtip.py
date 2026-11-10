"""Reproducible held-out weight distortion comparison, NOT an inference speed test.

CPU dependencies: numpy, ml_dtypes, C++17 compiler with OpenMP. QTIP uses the
repository's reference Viterbi, common input normalization/rotation and identity
Hessian. Neither method sees activation calibration. Test blocks do not fit the
binary decoder. Use --source to supply a real original-weight float32 .npy slice.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binary_codec import BLOCK, fit, quantize


def hadamard(x):
    x = np.array(x, dtype=np.float32, copy=True)
    n = x.shape[-1]
    if n < 1:
        raise ValueError('Empty transform')
    if n & (n - 1):
        # Factorized orthogonal transform for e.g. 28672 = 7 * 4096.
        # DCT on the odd factor, Hadamard on its power-of-two factor. Shared
        # by both methods; this is not the production QTIP Hadamard matrix.
        power = n & -n
        odd = n // power
        z = hadamard(x.reshape(*x.shape[:-1], odd, power))
        k = np.arange(odd)[:,None]
        j = np.arange(odd)[None,:]
        dct = np.cos(np.pi / odd * (j + .5) * k) * np.sqrt(2. / odd)
        dct[0] /= np.sqrt(2.)
        return np.einsum('ij,...jk->...ik',dct.astype(np.float32),z).reshape(x.shape)
    stride = 1
    while stride < n:
        v = x.reshape(*x.shape[:-1], -1, 2, stride)
        a, b = v[..., 0, :].copy(), v[..., 1, :].copy()
        v[..., 0, :], v[..., 1, :] = a + b, a - b
        stride *= 2
    return x / np.sqrt(np.float32(n))


def metric(ref, got, x):
    err = got.astype(np.float64) - ref
    energy = float(np.square(ref.astype(np.float64)).sum())
    y = x @ ref.T
    ye = x @ err.T
    return dict(weight_nmse=float(np.square(err).sum() / energy),
                weight_relative_l2=float(np.linalg.norm(err) / np.sqrt(energy)),
                random_activation_output_relative_l2=float(np.linalg.norm(ye) / np.linalg.norm(y)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--train-blocks', type=int, default=4096)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--sweeps', type=int, default=8)
    p.add_argument('--restarts', type=int, default=4)
    p.add_argument('--test-rows', type=int, default=128)
    p.add_argument('--test-cols', type=int, default=512)
    p.add_argument('--no-rotation', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    w = np.load(args.source, allow_pickle=False) if args.source else rng.normal(size=(1024, 1024)).astype(np.float32)
    if w.ndim != 2 or not np.isfinite(w).all():
        p.error('Source must be a finite matrix')
    if args.test_rows <= 0 or args.test_rows >= w.shape[0] or args.test_rows % 32:
        p.error('test-rows must be positive, divisible by 32 and leave training rows')
    if args.test_cols <= 0 or args.test_cols > w.shape[1] or args.test_cols % 64 or w.shape[1] % 64:
        p.error('test-cols and source width must be multiples of 64; test-cols must fit')
    if args.train_blocks < 128:
        p.error('At least 128 training blocks required')
    if not args.no_rotation:
        w = hadamard(w * rng.choice([-1., 1.], w.shape[1])).T
        w = hadamard(w * rng.choice([-1., 1.], w.shape[1])).T
    # Split after the common orthogonal transform. No test blocks used to fit D.
    train = w[args.test_rows:].reshape(-1, BLOCK)
    train = train[rng.choice(len(train), min(len(train), args.train_blocks), replace=False)]
    scale = float(np.sqrt(np.mean(train.astype(np.float64) ** 2)))
    if scale == 0:
        p.error('Training weights are all zero')
    train = (train / scale).astype(np.float32)
    test = (w[:args.test_rows, :args.test_cols] / scale).astype(np.float32)
    print(f'train {train.shape}, held-out {test.shape}, rotation={not args.no_rotation}', flush=True)
    start = time.perf_counter()
    decoder, history = fit(train, iterations=args.iterations, seed=args.seed)
    fit_seconds = time.perf_counter() - start
    print(f'decoder fitted in {fit_seconds:.2f}s; train MSE {history[0]:.5f} -> {history[-1]:.5f}', flush=True)
    start = time.perf_counter()
    binary = quantize(test, decoder, sweeps=args.sweeps, restarts=args.restarts, seed=args.seed + 1)
    encode_seconds = time.perf_counter() - start
    binary.save(args.out / 'binary.npz')
    test.tofile(args.out / 'input.bin')
    np.save(args.out / 'input.npy', test)
    root = Path(__file__).resolve().parents[2]
    exe = args.out / 'qtip_reference'
    subprocess.run(['c++', '-O3', '-std=c++17', '-fopenmp', '-I'+str(root/'src'),
                    str(Path(__file__).with_name('binary_qtip_reference.cpp')),
                    str(root/'src/qtip_core.cpp'), '-o', str(exe)], check=True)
    start = time.perf_counter()
    subprocess.run([str(exe.resolve()), str(args.out/'input.bin'), str(args.out/'qtip'),
                    str(test.shape[0]), str(test.shape[1]), '0.9'], check=True)
    qtip_seconds = time.perf_counter() - start
    q = np.fromfile(args.out/'qtip.dense.bin', np.float32).reshape(test.shape)
    x = rng.normal(size=(256, test.shape[1])).astype(np.float32)
    np.save(args.out / 'activations.npy', x)
    qtip_bytes = test.size // 4 + 512*2*2 + 4
    report = dict(
        source=str(args.source) if args.source else 'synthetic_iid_gaussian',
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest() if args.source else None,
        source_shape=list(w.shape), test_shape=list(test.shape), train_blocks=len(train),
        seed=args.seed, shared_rotation=not args.no_rotation, normalization_scale=scale,
        rotation='Hadamard (DCT for any odd factor), random signs' if not args.no_rotation else 'none',
        comparison='held-out rotated weights; identity Hessian; no activation calibration',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        binary=dict(**metric(test, binary.dense(), x), payload_bits_per_weight=2,
                    decoder_bytes=binary.decoder.nbytes, storage_bytes=binary.storage_bytes,
                    effective_bits_per_weight=8*binary.storage_bytes/test.size,
                    fit_seconds=fit_seconds, encode_seconds=encode_seconds,
                    iterations=args.iterations, sweeps=args.sweeps, restarts=args.restarts,
                    training_mse=history),
        qtip=dict(**metric(test,q,x), payload_bits_per_weight=2,
                  storage_bytes=qtip_bytes, effective_bits_per_weight=8*qtip_bytes/test.size,
                  encode_seconds=qtip_seconds, scale_override=0.9,
                  encoder='src/qtip_core.cpp quantize_seq_ref; BF16 codebook decode'),
        device_latency_ms=None,
        limitations=['Not production Hessian-aware QTIP/LDLQ or teacher-guided block reconstruction.',
                     'Output error uses random Gaussian activations, not model activations or logits.',
                     'Weight-slice results do not establish perplexity or full-model quality.',
                     'CPU encoder times are offline costs, not Neuron inference latency.',
                     'Storage includes decoder/LUT and QTIP scale; common rotations excluded.'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {m:v for m,v in report[k].items() if m != 'training_mse'} for k in ['binary','qtip']}, indent=2))


if __name__ == '__main__':
    main()
