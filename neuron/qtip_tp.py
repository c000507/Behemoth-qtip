"""Tensor-parallel QTIP projections with the weights kept as packed 2-bit trellis words on the device.

Each projection decodes its local trellis shard inside the graph (dec_qtip) every step; the dense
weight never exists in HBM. The random Hadamard rotation (RHT) spans a whole input/output dimension,
so TP needs collectives beyond the usual one all-reduce:

  column-parallel (q, gate, up): y = SV * H_out(W~ u),  u = H_in(SU * x).  W~ rows are sharded, so each
      rank computes z_local = W~_local u, the ranks all-gather z, apply H_out + SV on the full vector
      and keep their own slice.
  row-parallel (o, down): the input is sharded, but u needs the full vector: all-gather x, rotate,
      each rank multiplies its W~ column shard by its slice of u, all-reduce the partial z, then
      H_out + SV on the full output (replicated on both ranks).

trellis shard layout follows the checkpoint order (i0, j0, c, j1, i1): column-parallel shards i0
(output rows, 32 per unit), row-parallel shards j0 (input columns, 32 per unit).
"""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import torch

from trace_layer import dec_qtip


@lru_cache(maxsize=2)
def hadamard_factor(k):
    text = (Path(__file__).resolve().parents[1] / 'runtime/src/qtip_tables.hpp').read_text()
    match = re.search(r'kHad' + str(k) + r'\[\d+\] = \{([^}]*)\}', text)
    values = np.array([int(v) for v in match.group(1).split(',')], np.float32)
    return values.reshape(k, k)


def had_k(n):
    """Mirror of qtip::had_k(): 28 or 12 when n / k is a power of two, else 1 (power of two)."""
    for k in (28, 12):
        if n % k == 0 and (n // k) & (n // k - 1) == 0:
            return k
    if n & (n - 1):
        raise ValueError(f'Unsupported QTIP Hadamard size {n}')
    return 1


def hadamard(x, n, factors, transpose):
    """qtip::hadamard() on the last dim: natural-order FWHT over the n/k power-of-two part, then the
    k x k QTIP factor (transposed when `transpose`), scaled 1/sqrt(n). `factors` maps k -> tensor [k, k]."""
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
        h = factors[k]
        v = torch.einsum('rc,bcp->brp', h.T if transpose else h, v)
    return (v * (n ** -0.5)).reshape(x.shape)


# --- column-parallel (q, gate, up) -----------------------------------------------------------
def col_local(x, su, trellis, tlut, out_local, width, factors):
    """z_local [T, out_local]; x is the full (replicated) activation [T, width]."""
    u = hadamard(x.float() * su.reshape(1, -1), width, factors, True).to(torch.bfloat16)
    return (u @ dec_qtip(trellis, tlut, out_local, width).T).float()


def col_finish(z, sv, out, factors, local_idx):
    """z [T, out] (all-gathered) -> this rank's slice of y; local_idx int64 [out_local] = rank*out_local + arange."""
    y = hadamard(z, out, factors, False) * sv.reshape(1, -1)
    return torch.index_select(y, -1, local_idx)


# --- row-parallel (o, down) ------------------------------------------------------------------
def row_local(x_full, su, trellis, tlut, out, in_local, width, factors, local_idx):
    """z_partial [T, out]; x_full [T, width] is the all-gathered input; trellis shards the input columns."""
    u = hadamard(x_full.float() * su.reshape(1, -1), width, factors, True).to(torch.bfloat16)
    u_local = torch.index_select(u, -1, local_idx)
    return (u_local @ dec_qtip(trellis, tlut, out, in_local).T).float()


def row_finish(z, sv, out, factors):
    return hadamard(z, out, factors, False) * sv.reshape(1, -1)


# --- graph wrappers (xm collectives, TP ranks in one replica group) --------------------------
def col_parallel(x, su, sv, trellis, tlut, out, width, factors, local_idx, tp):
    import torch_xla.core.xla_model as xm
    z_local = col_local(x, su, trellis, tlut, out // tp, width, factors)
    z = xm.all_gather(z_local, dim=-1, groups=[list(range(tp))])
    return col_finish(z, sv, out, factors, local_idx)


def row_parallel(x_local, su, sv, trellis, tlut, out, width, factors, local_idx, tp):
    import torch_xla.core.xla_model as xm
    groups = [list(range(tp))]
    x_full = xm.all_gather(x_local, dim=-1, groups=groups)
    z = xm.all_reduce(xm.REDUCE_SUM, row_local(x_full, su, trellis, tlut, out, width // tp, width, factors, local_idx),
                      groups=groups)
    return row_finish(z, sv, out, factors)


def shard_trellis(trellis, out, width, tp, rank, parallel):
    """Checkpoint trellis [(out/32)*(width/32)*32*2*2] words -> this rank's shard in dec_qtip layout."""
    t = np.asarray(trellis).reshape(out // 32, width // 32, 32, 2, 2)
    if parallel == 'col':
        n = out // 32 // tp
        return np.ascontiguousarray(t[rank * n:(rank + 1) * n])
    n = width // 32 // tp
    return np.ascontiguousarray(t[:, rank * n:(rank + 1) * n])
