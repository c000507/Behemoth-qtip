"""Fused attention-side prologue of the decode layer as one NKI kernel: RMSNorm + fp8 activation quant + q / k / v projections + RoPE.

Why: in the dec1_* / dec8_* graphs this chain is ~25 XLA ops (norm, amax, divide, cast, three fp8 matmuls with the small weights as the
STATIONARY operand, three rescales, rope slices / cats) between the layer start and the attention kernel; the weights (q 10.5 MB, k / v
2.6 MB each per rank) are only 26 % of the time the chain takes (docs/neuron-nki-fusion.md). Here the activation is stationary and the
weight tiles stream through the PE as the moving operand (the nki_mlp pattern), norm / quant / rope are vector work on [T, N] tiles.

Semantics match trace_model.paged_layer for T <= 32 rows:
  h   = bf16( x * rsqrt(mean(x^2) + 1e-5) * norm1 )
  a   = max|h| / 224 per row, h8 = E4M3(h / a)
  q   = bf16( (h8 . qf) * a * qs )   (likewise k with kf / ks, v with vf / vs); q and k RoPE'd per 128-wide head with cos_sin [T, 2, 64]
  out [T, NQ + 2 NKV] f32 = [q | k | v]   (the caller slices; q feeds the attention kernel, k / v the fp8 KV quant + page scatter)
qf [H, NQ], kf / vf [H, NKV] are the WT=1 (contraction axis first) E4M3 shards (uint8 bits or float8 tensors), qs / ks / vs [1, N] f32.
"""
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

NC = 512          # moving columns per matmul (one f32 PSUM bank)
D = 128           # head dim (RoPE pairs columns j and j + 64 inside every 128-column head)
EPS = 1e-5


def _f8(t, dtype):
    return t.view(nl.float8_e4m3) if dtype == nl.uint8 else t


@nki.jit
def qkv_fused(x, norm1, qf, qs, kf, ks, vf, vs, cos_sin):
    T, H = x.shape
    NQ, NKV = qf.shape[1], kf.shape[1]
    KT = H // 128
    out = nl.ndarray((T, NQ + 2 * NKV), dtype=nl.float32, buffer=nl.shared_hbm)
    # ---- RMSNorm + fp8 quant (T rows on the partitions) ----
    xt = nl.load(x)                                                                         # [T, H] f32
    ss = nisa.tensor_reduce(np.add, nl.multiply(xt, xt), axis=(1,), dtype=nl.float32)        # [T, 1]
    r = nl.rsqrt(nl.add(nl.multiply(ss, 1.0 / H), EPS))
    w1 = nl.broadcast_to(nl.load(norm1[0:1, :]), shape=(T, H))
    h = nl.copy(nl.multiply(nl.multiply(xt, r), w1), dtype=nl.bfloat16)                     # the graph's bf16 h
    hf = nl.copy(h, dtype=nl.float32)
    a = nl.maximum(nl.multiply(nisa.tensor_reduce(np.max, nl.abs(hf), axis=(1,), dtype=nl.float32), 1.0 / 224.0), 1e-20)   # [T, 1]
    h8 = nl.copy(nl.divide(hf, a), dtype=nl.float8_e4m3)                                    # [T, H] E4M3
    hx = nl.ndarray((128, KT * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)                     # stationary tiles [128 k, T] per K-tile
    for kt in nl.affine_range(KT):
        hx[:, nl.ds(kt * T, T)] = nl.copy(nisa.nc_transpose(h8[:, nl.ds(kt * 128, 128)]), dtype=nl.float8_e4m3)
    cos = nl.load(cos_sin[:, 0, :])                                                          # [T, 64]
    sin = nl.load(cos_sin[:, 1, :])

    def project(W, S, col0, N, NB, do_rope):
        """out[:, col0 : col0 + N] = rope?( bf16((h8 . W) * a * S) ), NB columns per weight tile (NB / 512 PSUM banks)."""
        NJ = NB // NC
        for nb in nl.affine_range(N // NB):
            accs = [nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum) for _ in nl.static_range(NJ)]
            for kt in nl.sequential_range(KT):
                wt = _f8(nl.load(W[nl.ds(kt * 128, 128), nl.ds(nb * NB, NB)]), W.dtype)         # [128 k, NB] E4M3, moving operand
                for jj in nl.static_range(NJ):
                    accs[jj] += nl.matmul(hx[:, nl.ds(kt * T, T)], wt[:, nl.ds(jj * NC, NC)], transpose_x=True)
            sc = nl.broadcast_to(nl.load(S[0:1, nl.ds(nb * NB, NB)]), shape=(T, NB))
            y = nl.ndarray((T, NB), dtype=nl.float32, buffer=nl.sbuf)
            for jj in nl.static_range(NJ):
                y[:, nl.ds(jj * NC, NC)] = nl.copy(nl.copy(nl.multiply(nl.multiply(accs[jj], a), sc[:, nl.ds(jj * NC, NC)]), dtype=nl.bfloat16), dtype=nl.float32)
            if do_rope:
                z = nl.ndarray((T, NB), dtype=nl.float32, buffer=nl.sbuf)
                for hd in nl.affine_range(NB // D):
                    ya = y[:, nl.ds(hd * D, D // 2)]
                    yb = y[:, nl.ds(hd * D + D // 2, D // 2)]
                    z[:, nl.ds(hd * D, D // 2)] = nl.subtract(nl.multiply(ya, cos), nl.multiply(yb, sin))
                    z[:, nl.ds(hd * D + D // 2, D // 2)] = nl.add(nl.multiply(yb, cos), nl.multiply(ya, sin))
                nl.store(out[:, nl.ds(col0 + nb * NB, NB)], z)
            else:
                nl.store(out[:, nl.ds(col0 + nb * NB, NB)], y)

    project(qf, qs, 0, NQ, min(NQ, 2048), True)                                              # q: 2 KB weight rows, 4 PSUM banks
    project(kf, ks, NQ, NKV, min(NKV, 512), True)
    project(vf, vs, NQ + NKV, NKV, min(NKV, 512), False)
    return out
