"""Torch-side glue for QTIP projections kept compressed on the device (trace_model.py, WEIGHTS=qtip).

A QTIP linear is y = SV (.) hadU_out( hadUt_in(SU (.) x) . W~^T ): the trellis codes W~ (decoded and multiplied by
the NKI kernel qtip_nki.qtip_linear), and the randomized Hadamard transforms act on whole activation vectors. With
the weights expanded at load (qtip_expand.cpp) the rotations are folded into the dense weight and sharded with it;
with the trellis resident that is impossible (a sum of codebook entries is not a codebook entry), so the rotations
run in the graph and tensor parallelism needs the full rotated vectors:

  column-parallel (q, gate, up): u = hadUt_in(SU x) is replicated (x is); each rank multiplies its block of W~
      rows, the ranks all-gather z = W~ u (gather_tp: an all-reduce of masked copies, since torch_xla drops
      xm.all_gather from a single-process trace), both apply hadU_out + SV on the full z and keep their own slice of y.
  row-parallel (o, down): the rotation needs the full input (all-gathered attention output; gate/up outputs are
      already replicated), each rank multiplies its block of W~ columns by its slice of u, the partial z is
      all-reduced (as for the FP8 graph) and hadU_out + SV is applied to the full output on both ranks.

So a QTIP layer has two more collectives than the FP8 layer: all-gather of z for q and for gate|up (one each) and
all-gather of the attention output before o_proj (the all-reduces after o and down are the FP8 graph's). Hadamard
sizes with a 12 or 28 factor (hadK (x) H_2^p, runtime/src/qtip_tables.hpp) are supported like the C++ reference.
"""
from functools import lru_cache
from pathlib import Path
import re

import torch
import torch.nn.functional as F

IN_GROUP = 2048  # qtip_nki.IN_GROUP: input channels per 128-partition group of the kernel


@lru_cache(maxsize=2)
def hadamard_factor(k):
    """kHad12 / kHad28 from runtime/src/qtip_tables.hpp as a float32 [k, k] tensor."""
    text = (Path(__file__).resolve().parents[1] / 'runtime/src/qtip_tables.hpp').read_text()
    match = re.search(r'kHad' + str(k) + r'\[\d+\] = \{([^}]*)\}', text)
    return torch.tensor([float(v) for v in match.group(1).split(',')], dtype=torch.float32).reshape(k, k)


def had_k(n):
    """qtip::had_k(): 28 or 12 when n / k is a power of two, else 1 (n a power of two)."""
    for k in (172, 156, 140, 124, 116, 108, 60, 52, 36, 28, 20, 12):
        if n % k == 0:
            if k not in (12, 28) or (n // k) & (n // k - 1):
                raise ValueError(f'unsupported QTIP Hadamard size {n}')
            return k
    if n & (n - 1):
        raise ValueError(f'unsupported QTIP Hadamard size {n}')
    return 1


def hadamard(x, n, transpose):
    """qtip::hadamard() over the last dim of x [.., n] (f32): natural-order FWHT over the n/k power-of-two part,
    then the k x k factor (its transpose when `transpose`), scaled 1/sqrt(n)."""
    k = had_k(n)
    p = n // k
    v = x.reshape(-1, k, p)
    batch = v.shape[0]
    stride = 1
    while stride < p:
        v = v.reshape(batch, k, p // (2 * stride), 2, stride)
        v = torch.stack((v[:, :, :, 0] + v[:, :, :, 1], v[:, :, :, 0] - v[:, :, :, 1]), dim=3).reshape(batch, k, p)
        stride *= 2
    if k > 1:
        h = hadamard_factor(k).to(x.device)
        v = torch.einsum('rc,bcp->brp', h.T if transpose else h, v)
    return (v * (n ** -0.5)).reshape(x.shape)


def rotate_in(x, su, n):
    """u = bf16(hadUt_n(SU (.) x)) for x [T, n] (bf16 or f32), su [1, n]."""
    return hadamard(x.float() * su, n, True).to(torch.bfloat16)


def finish(z, sv, n):
    """y = SV (.) hadU_n(z) for z [T, n] f32."""
    return hadamard(z, n, False) * sv


def kernel_matmul(u, words, lut):
    """z [T, out_local] f32 = u [T, in_local] bf16 . W~_local^T through the fused decode kernel. The kernel wants
    x^T as [128, G, 16, T]: partition p of in-group g holds channels g*2048 + 16 p + c (zero padded ragged group)."""
    from qtip_nki import qtip_linear
    T, width = u.shape
    groups = words.shape[0]
    if groups * IN_GROUP > width:
        u = F.pad(u, (0, groups * IN_GROUP - width))
    xT = u.reshape(T, groups, 128, 16).permute(2, 1, 3, 0).contiguous()
    return qtip_linear(words, lut, xT)


def gather_tp(x_local, sel, width, allreduce, tp):
    """all-gather of x_local [T, width/tp] along the last dim in rank order, as an all-reduce of masked copies: the local
    slice is tiled tp times and multiplied by this rank's 0/1 mask (built from `sel` = rank * width/tp + arange).
    torch_xla drops xm.all_gather when it traces a single process (the compiled HLO has no all-gather at all) while
    XLA_ALWAYS_ALLREDUCE keeps all_reduce; and an index_copy / index_add whose source is the NKI kernel's output lands
    columns wrongly on the device (the same scatter from a graph input is exact), so no scatter is used here."""
    mask = torch.zeros(width, dtype=x_local.dtype, device=x_local.device).index_fill(0, sel, 1.0).reshape(1, width)
    return allreduce(torch.cat([x_local] * tp, dim=-1) * mask)


def groups_of(in_local):
    return (in_local + IN_GROUP - 1) // IN_GROUP


def words_shape(out_local, in_local):
    """Device layout of a rank's trellis words (qtip_nki.pack_tiles): uint16 [G, 128, (out_local / 16) * 32]."""
    return (groups_of(in_local), 128, out_local // 16 * 32)
