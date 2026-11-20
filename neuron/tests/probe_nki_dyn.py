"""PROBE (new `nki` 0.5.0 package, parser frontend): can a kernel inside torch_neuronx.trace run a loop whose trip count is a
device value, with an indirect DMA per iteration whose index vector is selected by the loop register?

  python probe_nki_dyn.py OUT NEFF_IO [variant]     (device job: one core)
variant: dyn (nl.dynamic_range over a count loaded from HBM) | skip (static trip count, out-of-bounds rows skipped by oob_mode.skip)
Kernel: acc[p, :] = sum_{g < cnt} W[idx[g, p], :]  (W [N, 512] f32, idx [G, 128] int32, cnt [1, 1] int32).
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('UNSAFE_FP8FNCAST', '1')
import numpy as np
import torch

import nki
import nki.isa as nisa
import nki.language as nl

N, G, C = 4096, 16, 512


@nki.jit
def dyn_kernel(cnt, idx, W):
    out = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.shared_hbm)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    n = nisa.register_alloc()
    nisa.register_load(n, c_sb)
    acc = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for g in nl.dynamic_range(n):
        it = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=it, src=idx.ap(pattern=[[1, 128], [1, 1]], offset=0, scalar_offset=g, indirect_dim=0))
        w = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=W.ap(pattern=[[C, 128], [1, C]], offset=0, vector_offset=it, indirect_dim=0))
        nisa.tensor_tensor(dst=acc, data1=acc, data2=w, op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out


@nki.jit
def skip_kernel(cnt, idx, W):
    out = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for g in nl.static_range(G):
        it = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=it, src=idx[g:g + 1, :].reshape((128, 1)))
        w = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=w, value=0.0)
        nisa.dma_copy(dst=w, src=W.ap(pattern=[[C, 128], [1, C]], offset=0, vector_offset=it, indirect_dim=0), oob_mode=nisa.oob_mode.skip)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=w, op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out


def main():
    out, neff_io = Path(sys.argv[1]), sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else 'dyn'
    import torch_neuronx
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import glm_repack as RP
    kern = {'dyn': dyn_kernel, 'skip': skip_kernel}[variant]
    rng = np.random.default_rng(1)
    W = torch.from_numpy(rng.standard_normal((N, C)).astype(np.float32))
    idx = torch.from_numpy(rng.integers(0, N, (G, 128)).astype(np.int32))
    cnt_v = 5
    if variant == 'skip':
        idx[cnt_v:] = N + 7                      # out of bounds -> skipped
    cnt = torch.tensor([[cnt_v]], dtype=torch.int32)
    d = out / f'probe_{variant}'
    d.mkdir(parents=True, exist_ok=True)
    try:
        torch_neuronx.trace(lambda a, b, c: kern(a, b, c) * 1.0, (cnt, idx, W), compiler_workdir=str(d))
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'],
                   cwd=d, check=True, stdout=subprocess.DEVNULL)
    io = d / 'io'
    io.mkdir(exist_ok=True)
    for i, t in enumerate([cnt, idx, W]):
        (io / f'in{i}').write_bytes(t.contiguous().numpy().tobytes())
    for cv in (5, 0, 16, 3):
        if variant == 'dyn':
            (io / 'in0').write_bytes(np.array([[cv]], np.int32).tobytes())
        else:
            cv = cnt_v
        subprocess.run([neff_io, str(d / 'graph.neff'), str(io), '1'], check=True, stdout=subprocess.DEVNULL)
        y = np.frombuffer((io / 'out0.r0').read_bytes(), np.float32).reshape(128, C)
        ref = sum(W.numpy()[idx.numpy()[g]] for g in range(cv)) if cv else np.zeros((128, C), np.float32)
        print(f'PROBE {variant} cnt={cv}: max|diff| {np.abs(y - ref).max():.3e} |ref| {np.abs(ref).max():.3f}', flush=True)


if __name__ == '__main__':
    main()
