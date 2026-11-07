"""NKI FP8 matmul with the WEIGHTS as the moving operand (activation stationary), used to find out what limits a one-token decode projection.

y[T, N] = x[T, K] @ W[K, N] for T <= 128: per 128-row K tile the activation tile [128, T] is the stationary operand and a [128, 1024] weight
tile streams through as two 512-column moving matmuls accumulated in PSUM. Inputs are E4M3 bit patterns passed as uint8 (|value| <= 224 so the
FN and IEEE-style encodings agree). Result on trn1 (profile of the compiled NEFF, 4096 x 7168 weights = 29.4 MB, T = 8): DMA active 157 us
(187 GB/s), tensor engine active 107 us, i.e. DMA-bound, the same ~175-190 GB/s as the compiler's own LDWEIGHTS/MATMUL schedule
(neuron/PERF_NOTES.md "decode layer is at the DMA wall"). nisa.dge_mode.none gave 136 us (216 GB/s); hwdge is trn2 only.
Time it with the in-kernel `rep` loop (PyTorch/XLA lazy tensors only execute the last of several queued identical calls), or profile the NEFF.
"""
import sys, time, numpy as np, torch
import neuronxcc.nki as nki, neuronxcc.nki.isa as nisa, neuronxcc.nki.language as nl
import torch_xla.core.xla_model as xm

NC = 512  # moving columns per matmul instruction (one PSUM bank of f32)

def make(K, N, T, rep, wide, dge=None):
    KT = K // 128
    @nki.jit
    def k(xT, w):
        """xT [K, T] fp8 bits (uint8), w [K, N] fp8 bits (uint8) -> [T, N] f32; weights are the MOVING operand (wide = 1024)."""
        out = nl.ndarray((T, N), dtype=nl.float32, buffer=nl.shared_hbm)
        for nb in nl.affine_range(N // 1024):
            a0 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
            a1 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
            for r in nl.sequential_range(rep):
                for kt in nl.sequential_range(KT):
                    xk = nl.load(xT[nl.ds(kt * 128, 128), :]).view(nl.float8_e4m3)
                    if dge is None:
                        wt = nl.load(w[nl.ds((r * KT + kt) * 128, 128), nl.ds(nb * 1024, 1024)]).view(nl.float8_e4m3)
                    else:
                        raw = nl.ndarray((128, 1024), dtype=nl.uint8, buffer=nl.sbuf)
                        nisa.dma_copy(src=w[nl.ds((r * KT + kt) * 128, 128), nl.ds(nb * 1024, 1024)], dst=raw, dge_mode=dge)
                        wt = raw.view(nl.float8_e4m3)
                    a0 += nl.matmul(xk, wt[:, 0:NC], transpose_x=True)
                    a1 += nl.matmul(xk, wt[:, NC:2 * NC], transpose_x=True)
            nl.store(out[:, nl.ds(nb * 1024, NC)], nl.copy(a0, dtype=nl.float32))
            nl.store(out[:, nl.ds(nb * 1024 + NC, NC)], nl.copy(a1, dtype=nl.float32))
        return out
    return k

if __name__ == '__main__':
    K, N, T, wide = 4096, 7168 + 1024 - 7168 % 1024 if False else 7168, int(sys.argv[1]), 1024
    dev = xm.xla_device()
    rng = np.random.default_rng(0)
    w8 = torch.from_numpy(rng.integers(0, 120, (K, N), dtype=np.uint8)).to(dev)   # small positive e4m3 codes
    x8 = torch.from_numpy(rng.integers(0, 120, (K, T), dtype=np.uint8)).to(dev)
    times = {}
    for rep in (1, 41):
        w8 = torch.from_numpy(rng.integers(0, 120, (rep * K, N), dtype=np.uint8)).to(dev)
        k = make(K, N, T, rep, wide)
        k(x8, w8).cpu()
        t = time.time()
        for _ in range(20):
            r = k(x8, w8)
        xm.mark_step(); r.cpu()
        times[rep] = (time.time() - t) / 20 * 1e3
    print('abs ms', times)
    per = (times[41] - times[1]) / 40  # ms per full pass over the K x N weights
    print('T', T, 'wide', wide, 'ms per pass', round(per, 3), 'weights G/s', round(K * N / per / 1e6, 1), 'GB/s', round(K * N / per / 1e6, 1), flush=True)
