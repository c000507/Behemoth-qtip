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


@nki.jit
def mlp_fp8(h8T, ha, gf, gs, uf, us, df, ds):
    H, T = h8T.shape
    FF = gf.shape[1]
    KT, FT = H // 128, FF // 128
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    hx = nl.ndarray((128, KT * T), dtype=h8T.dtype, buffer=nl.sbuf)                 # stationary activation tiles, [128, (kt, T)]
    for kt in nl.affine_range(KT):
        hx[:, nl.ds(kt * T, T)] = nl.load(h8T[nl.ds(kt * 128, 128), :])
    hs = nl.load(ha)                                                                  # [T, 1]
    act = nl.ndarray((T, FF), dtype=nl.float32, buffer=nl.sbuf)
    # gate | up: column blocks of NB, K accumulated in PSUM
    for nb in nl.affine_range(FF // NB):
        g0 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
        g1 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
        u0 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
        u1 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
        for kt in nl.sequential_range(KT):
            xk = _f8(hx[:, nl.ds(kt * T, T)], h8T.dtype)
            wg = _f8(nl.load(gf[nl.ds(kt * 128, 128), nl.ds(nb * NB, NB)]), gf.dtype)
            wu = _f8(nl.load(uf[nl.ds(kt * 128, 128), nl.ds(nb * NB, NB)]), uf.dtype)
            g0 += nl.matmul(xk, wg[:, 0:NC], transpose_x=True)
            g1 += nl.matmul(xk, wg[:, NC:NB], transpose_x=True)
            u0 += nl.matmul(xk, wu[:, 0:NC], transpose_x=True)
            u1 += nl.matmul(xk, wu[:, NC:NB], transpose_x=True)
        gsc = nl.broadcast_to(nl.load(gs[0:1, nl.ds(nb * NB, NB)]), shape=(T, NB))
        usc = nl.broadcast_to(nl.load(us[0:1, nl.ds(nb * NB, NB)]), shape=(T, NB))
        for c, gp, up in ((0, g0, u0), (NC, g1, u1)):                                 # Python-level unroll (PSUM tiles cannot be indexed by a loop var)
            gate = nl.multiply(nl.multiply(gp, hs), gsc[:, nl.ds(c, NC)])
            upv = nl.multiply(nl.multiply(up, hs), usc[:, nl.ds(c, NC)])
            act[:, nl.ds(nb * NB + c, NC)] = nl.multiply(nl.silu(gate), upv)
    # down: requantise the activation rows (amax / 224), transpose to [128, T] stationary tiles
    aa = nl.maximum(nl.multiply(nisa.tensor_reduce(np.max, nl.abs(act), axis=(1,), dtype=nl.float32), 1.0 / 224.0), 1e-20)   # [T, 1]
    a8 = nl.copy(nl.divide(act, aa), dtype=nl.float8_e4m3)
    ax = nl.ndarray((128, FT * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for ft in nl.affine_range(FT):
        ax[:, nl.ds(ft * T, T)] = nl.copy(nisa.nc_transpose(a8[:, nl.ds(ft * 128, 128)]), dtype=nl.float8_e4m3)
    for nb in nl.affine_range(H // NB):
        d0 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
        d1 = nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum)
        for ft in nl.sequential_range(FT):
            wd = _f8(nl.load(df[nl.ds(ft * 128, 128), nl.ds(nb * NB, NB)]), df.dtype)
            d0 += nl.matmul(ax[:, nl.ds(ft * T, T)], wd[:, 0:NC], transpose_x=True)
            d1 += nl.matmul(ax[:, nl.ds(ft * T, T)], wd[:, NC:NB], transpose_x=True)
        dsc = nl.broadcast_to(nl.load(ds[0:1, nl.ds(nb * NB, NB)]), shape=(T, NB))
        nl.store(y[:, nl.ds(nb * NB, NC)], nl.multiply(nl.multiply(d0, aa), dsc[:, 0:NC]))
        nl.store(y[:, nl.ds(nb * NB + NC, NC)], nl.multiply(nl.multiply(d1, aa), dsc[:, NC:NB]))
    return y


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
