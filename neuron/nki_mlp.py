"""The FP8 decode MLP (gate | up, SiLU, down) of one tensor-parallel rank as one NKI kernel with the WEIGHTS as the moving operand.

Why: in the compiler's schedule of the MLP (transposed FP8 weights, M = T <= 32 rows) the weights are the stationary operand, one LDWEIGHTS
per 128 x 128 tile, and the profile shows the weight DMA stalling ~3.2 us of every ~4.6 us inside the MLP matmuls waiting on the PE
(PERF_NOTES "Where a decode layer's time goes"). Here the activation tile [128, T] is stationary and every [128, 1024] weight tile streams
through as two 512-column matmuls into PSUM, the pattern tests/bench_fp8_moving_nki.py keeps at the DMA wall.

Semantics match trace_layer.fp8_linear for each projection: the activation row is scaled into E4M3 range (amax / 224), the matmul runs
fp8 x fp8 with f32 accumulation, the output is multiplied by the row's activation scale and the column's weight scale.
  h8T [H, T] E4M3 (h / ha, transposed), ha [T, 1] f32
  gf / uf [H, FF] E4M3, gs / us [1, FF] f32   (WT=1 layout: contraction axis first)
  df [FF, H] E4M3, ds [1, H] f32
  -> y [T, H] f32 = this rank's partial down projection (the caller all-reduces it)
Weights may be uint8 bit patterns or float8 tensors (tracing a float8_e4m3fn input needs UNSAFE_FP8FNCAST=1; all codes are <= 224, where
the FN and IEEE-style encodings agree).
"""
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

NC = 512      # moving columns per matmul (one f32 PSUM bank)
NB = 1024     # weight columns per DMA tile


def _f8(t, dtype):
    return t.view(nl.float8_e4m3) if dtype == nl.uint8 else t


def _tile(n, nb):
    """Largest multiple of NC that divides n and is <= nb."""
    return max(c for c in range(NC, nb + 1, NC) if n % c == 0)


def make_mlp(NB=1024, dge=None):
    """mlp_fp8 with NB weight columns per DMA tile (NB // 512 PSUM banks per projection pass; gate and up run as separate passes when
    NB > 1024 so one pass never needs more than 4 of the 8 banks) and an optional nisa.dge_mode for the weight DMAs."""
    split = NB > 1024
    NBMAX = NB

    @nki.jit
    def mlp(h8T, ha, gf, gs, uf, us, df, ds):
        H, T = h8T.shape
        FF = gf.shape[1]
        KT, FT = H // 128, FF // 128
        NB = _tile(FF, NBMAX)                                                         # gate / up tile width (7168 -> 3584 for NBMAX 4096)
        NJ = NB // NC
        y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
        hx = nl.ndarray((128, KT * T), dtype=h8T.dtype, buffer=nl.sbuf)             # stationary activation tiles, [128, (kt, T)]
        for kt in nl.affine_range(KT):
            hx[:, nl.ds(kt * T, T)] = nl.load(h8T[nl.ds(kt * 128, 128), :])
        hs = nl.load(ha)                                                              # [T, 1]
        gate = nl.ndarray((T, FF), dtype=nl.float32, buffer=nl.sbuf)
        up = nl.ndarray((T, FF), dtype=nl.float32, buffer=nl.sbuf)
        passes = (((gf, gs, gate),), ((uf, us, up),)) if split else (((gf, gs, gate), (uf, us, up)),)
        for group in passes:                                                          # Python-level: one or two projection passes
            for nb in nl.affine_range(FF // NB):
                accs = [[nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum) for _ in nl.static_range(NJ)] for _ in group]
                for kt in nl.sequential_range(KT):
                    xk = _f8(hx[:, nl.ds(kt * T, T)], h8T.dtype)
                    for m in nl.static_range(len(group)):
                        W = group[m][0]
                        if dge is None:
                            wt = _f8(nl.load(W[nl.ds(kt * 128, 128), nl.ds(nb * NB, NB)]), W.dtype)
                        else:
                            raw = nl.ndarray((128, NB), dtype=W.dtype, buffer=nl.sbuf)
                            nisa.dma_copy(src=W[nl.ds(kt * 128, 128), nl.ds(nb * NB, NB)], dst=raw, dge_mode=dge)
                            wt = _f8(raw, W.dtype)
                        for jj in nl.static_range(NJ):
                            accs[m][jj] += nl.matmul(xk, wt[:, nl.ds(jj * NC, NC)], transpose_x=True)
                for m in nl.static_range(len(group)):
                    S, dst = group[m][1], group[m][2]
                    sc = nl.broadcast_to(nl.load(S[0:1, nl.ds(nb * NB, NB)]), shape=(T, NB))
                    for jj in nl.static_range(NJ):
                        dst[:, nl.ds(nb * NB + jj * NC, NC)] = nl.multiply(nl.multiply(accs[m][jj], hs), sc[:, nl.ds(jj * NC, NC)])
        act = nl.multiply(nl.silu(gate), up)
        # down: requantise the activation rows (amax / 224), transpose to [128, T] stationary tiles
        aa = nl.maximum(nl.multiply(nisa.tensor_reduce(np.max, nl.abs(act), axis=(1,), dtype=nl.float32), 1.0 / 224.0), 1e-20)   # [T, 1]
        a8 = nl.copy(nl.divide(act, aa), dtype=nl.float8_e4m3)
        ax = nl.ndarray((128, FT * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for ft in nl.affine_range(FT):
            ax[:, nl.ds(ft * T, T)] = nl.copy(nisa.nc_transpose(a8[:, nl.ds(ft * 128, 128)]), dtype=nl.float8_e4m3)
        NBD = _tile(H, NBMAX)                                                        # down tile width (Cydonia H = 5120 -> 2560)
        NJD = NBD // NC
        for nb in nl.affine_range(H // NBD):
            accd = [nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum) for _ in nl.static_range(NJD)]
            for ft in nl.sequential_range(FT):
                if dge is None:
                    wd = _f8(nl.load(df[nl.ds(ft * 128, 128), nl.ds(nb * NBD, NBD)]), df.dtype)
                else:
                    rawd = nl.ndarray((128, NBD), dtype=df.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=df[nl.ds(ft * 128, 128), nl.ds(nb * NBD, NBD)], dst=rawd, dge_mode=dge)
                    wd = _f8(rawd, df.dtype)
                for jj in nl.static_range(NJD):
                    accd[jj] += nl.matmul(ax[:, nl.ds(ft * T, T)], wd[:, nl.ds(jj * NC, NC)], transpose_x=True)
            dsc = nl.broadcast_to(nl.load(ds[0:1, nl.ds(nb * NBD, NBD)]), shape=(T, NBD))
            for jj in nl.static_range(NJD):
                nl.store(y[:, nl.ds(nb * NBD + jj * NC, NC)], nl.multiply(nl.multiply(accd[jj], aa), dsc[:, nl.ds(jj * NC, NC)]))
        return y
    return mlp


mlp_fp8 = make_mlp()


# ---- tiled weight layout --------------------------------------------------------------------------------------------------------------------
# A projection W [K, N] (contraction axis first) stored so that every 128-partition DMA tile is one contiguous 512 KB block whose partition
# rows are 4 KB: block (kb, nb) holds rows kb * 512 .. + 512 and columns nb * 1024 .. + 1024, partition p row = W[kb*512 + j*128 + p,
# nb*1024 : +1024] for j = 0..3 concatenated.  Flat index (((kb * NBL + nb) * 128 + p) * 4 + j) * 1024 + c, NBL = N / 1024; the tensor is
# [K / 512 * NBL * 128, 4096].  Wider SBUF rows stream faster (the 4 KB rows of a 4096-wide tile reached 254 GB/s, 1 KB rows ~205), while each
# tile still feeds only 2 PSUM banks (4 K-subtiles x 2 x 512 columns), so DMA and matmuls of consecutive blocks overlap.
TK, TN = 512, 1024


def tile_weight(w):
    """numpy [K, N] -> tiled [K / 512 * N / 1024 * 128, 4096] (same dtype)."""
    K, N = w.shape
    return np.ascontiguousarray(w.reshape(K // TK, 4, 128, N // TN, TN).transpose(0, 3, 2, 1, 4)).reshape(K // TK * (N // TN) * 128, 4 * TN)


def make_mlp_tiled(dge=None):
    @nki.jit
    def mlp(h8T, ha, gt, gs, ut, us, dt, ds):
        """h8T [H, T] E4M3, ha [T, 1]; gt / ut tiled [H, FF] E4M3, gs / us [1, FF]; dt tiled [FF, H], ds [1, H] -> y [T, H] f32 (partial)."""
        H, T = h8T.shape
        FF = gs.shape[1]
        KT, FT = H // 128, FF // 128
        y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
        hx = nl.ndarray((128, KT * T), dtype=h8T.dtype, buffer=nl.sbuf)
        for kt in nl.affine_range(KT):
            hx[:, nl.ds(kt * T, T)] = nl.load(h8T[nl.ds(kt * 128, 128), :])
        hs = nl.load(ha)
        # one f32 [T, FF] buffer (FF = 16384 per rank for Cydonia: 64 KB of a partition's 192 KB): the gate pass stores
        # gate, the up pass overwrites it with silu(gate) * up and keeps each block's |act| max
        NBL = FF // TN
        act = nl.ndarray((T, FF), dtype=nl.float32, buffer=nl.sbuf)
        mx = nl.ndarray((T, NBL), dtype=nl.float32, buffer=nl.sbuf)
        for W, S, second in ((gt, gs, False), (ut, us, True)):                        # Python-level: two projection passes
            for nb in nl.affine_range(NBL):
                a0 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
                a1 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.sequential_range(H // TK):
                    if dge is None:
                        wt = _f8(nl.load(W[nl.ds((kb * NBL + nb) * 128, 128), :]), W.dtype)
                    else:
                        raw = nl.ndarray((128, 4 * TN), dtype=W.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(src=W[nl.ds((kb * NBL + nb) * 128, 128), :], dst=raw, dge_mode=dge)
                        wt = _f8(raw, W.dtype)
                    for jj in nl.static_range(4):
                        xk = _f8(hx[:, nl.ds((kb * 4 + jj) * T, T)], h8T.dtype)
                        a0 += nl.matmul(xk, wt[:, nl.ds(jj * TN, NC)], transpose_x=True)
                        a1 += nl.matmul(xk, wt[:, nl.ds(jj * TN + NC, NC)], transpose_x=True)
                sc = nl.broadcast_to(nl.load(S[0:1, nl.ds(nb * TN, TN)]), shape=(T, TN))
                for c, acc in ((0, a0), (NC, a1)):
                    v = nl.multiply(nl.multiply(acc, hs), sc[:, nl.ds(c, NC)])
                    if second:
                        act[:, nl.ds(nb * TN + c, NC)] = nl.multiply(nl.silu(act[:, nl.ds(nb * TN + c, NC)]), v)
                    else:
                        act[:, nl.ds(nb * TN + c, NC)] = v
                if second:
                    mx[:, nl.ds(nb, 1)] = nisa.tensor_reduce(np.max, nl.abs(act[:, nl.ds(nb * TN, TN)]), axis=(1,), dtype=nl.float32)
        aa = nl.maximum(nl.multiply(nisa.tensor_reduce(np.max, mx, axis=(1,), dtype=nl.float32), 1.0 / 224.0), 1e-20)   # [T, 1]
        ra = nl.reciprocal(aa)
        a8 = nl.ndarray((T, FF), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for nb in nl.affine_range(NBL):
            a8[:, nl.ds(nb * TN, TN)] = nl.multiply(act[:, nl.ds(nb * TN, TN)], ra, dtype=nl.float8_e4m3)
        ax = nl.ndarray((128, FT * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for ft in nl.affine_range(FT):
            ax[:, nl.ds(ft * T, T)] = nl.copy(nisa.nc_transpose(a8[:, nl.ds(ft * 128, 128)]), dtype=nl.float8_e4m3)
        NBD = H // TN
        for nb in nl.affine_range(NBD):
            d0 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
            d1 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
            for kb in nl.sequential_range(FF // TK):
                if dge is None:
                    wd = _f8(nl.load(dt[nl.ds((kb * NBD + nb) * 128, 128), :]), dt.dtype)
                else:
                    rawd = nl.ndarray((128, 4 * TN), dtype=dt.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=dt[nl.ds((kb * NBD + nb) * 128, 128), :], dst=rawd, dge_mode=dge)
                    wd = _f8(rawd, dt.dtype)
                for jj in nl.static_range(4):
                    d0 += nl.matmul(ax[:, nl.ds((kb * 4 + jj) * T, T)], wd[:, nl.ds(jj * TN, NC)], transpose_x=True)
                    d1 += nl.matmul(ax[:, nl.ds((kb * 4 + jj) * T, T)], wd[:, nl.ds(jj * TN + NC, NC)], transpose_x=True)
            dsc = nl.broadcast_to(nl.load(ds[0:1, nl.ds(nb * TN, TN)]), shape=(T, TN))
            nl.store(y[:, nl.ds(nb * TN, NC)], nl.multiply(nl.multiply(d0, aa), dsc[:, 0:NC]))
            nl.store(y[:, nl.ds(nb * TN + NC, NC)], nl.multiply(nl.multiply(d1, aa), dsc[:, NC:TN]))
        return y
    return mlp


mlp_fp8_tiled = make_mlp_tiled()


def mlp_reference(h, gf, gs, uf, us, df, ds):
    """numpy f32 reference with fp8_linear's activation quantisation. h [T, H] f32; weights as float32 values of the E4M3 codes."""
    import ml_dtypes
    def q(a):
        s = np.maximum(np.abs(a).max(-1, keepdims=True), 1e-20) / 224.0
        return (a / s).astype(ml_dtypes.float8_e4m3fn).astype(np.float32), s
    h8, ha = q(h)
    gate = (h8 @ gf) * ha * gs
    up = (h8 @ uf) * ha * us
    act = gate / (1 + np.exp(-gate)) * up
    a8, aa = q(act)
    return (a8 @ df) * aa * ds
