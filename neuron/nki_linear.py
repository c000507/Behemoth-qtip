"""Fused NKI projection kernels (neuronxcc.nki, callable inside torch_neuronx.trace graphs).

The weights stay in their packed device layout in HBM. Each kernel streams a K-tile (128 input
channels on the partitions) at a time, decodes it in SBUF to bf16 and feeds the Tensor Engine, so
the dense weight matrix never exists in HBM.

NVFP4 layout (host-side repack, see nvfp4_pack):
  wp  uint8 [K/128, 128, N/2]  byte [kt, p, j]: low nibble = column 2j, high nibble = column 2j+1
                                of input channel k = kt*128 + p  (nibble: bit 3 sign, bits 0..2 E2M1)
  sce bf16  [K/16, N/2]        block scale (E4M3 decoded, global scale folded in) of columns 2j
  sco bf16  [K/16, N/2]        block scale of columns 2j + 1
  ex  bf16  [8, 128]           expansion matrix: ex[r, p] = (p // 16 == r); one 128-row K-tile spans
                                8 scale blocks, the Tensor Engine broadcasts them to the partitions
  xT  bf16  [K, B]             activations, transposed (K on partitions), B <= 128
  out f32   [B, N]
"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

NT = 256  # packed bytes (= 512 output columns) per output tile; one PSUM bank holds 512 f32 per partition


def e2m1_value(code):
    """E2M1 nibble (int tile, bit 3 = sign) -> float32 tile: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6."""
    mag = nl.bitwise_and(code, 7)
    e = nl.right_shift(mag, 1)
    f = nl.bitwise_and(mag, 1)
    pow2 = nl.left_shift(nl.copy(f, dtype=nl.int32) * 0 + 1, e)           # 2^e
    normal = nl.multiply(nl.add(f, 2), pow2, dtype=nl.float32) * 0.25     # (2 + f) * 2^e / 4, e >= 1
    value = nl.where(e == 0, nl.multiply(mag, 0.5, dtype=nl.float32), normal)
    sign = 1.0 - 2.0 * nl.right_shift(code, 3)
    return value * sign


@nki.jit
def nvfp4_linear(xT, wp, sce, sco, ex):
    K, B = xT.shape
    KT, P, HALF = wp.shape
    N = HALF * 2
    out = nl.ndarray((B, N), dtype=nl.float32, buffer=nl.shared_hbm)
    ex_t = nl.load(ex)  # [8, 128]
    for jt in nl.affine_range(HALF // NT):
        acc_e = nl.zeros((B, NT), dtype=nl.float32, buffer=nl.psum)
        acc_o = nl.zeros((B, NT), dtype=nl.float32, buffer=nl.psum)
        for kt in nl.affine_range(KT):
            x_t = nl.load(xT[kt * 128:(kt + 1) * 128, 0:B])
            w8 = nl.load(wp[kt, 0:128, jt * NT:(jt + 1) * NT])
            se = nl.load(sce[kt * 8:(kt + 1) * 8, jt * NT:(jt + 1) * NT])
            so = nl.load(sco[kt * 8:(kt + 1) * 8, jt * NT:(jt + 1) * NT])
            se_full = nl.copy(nl.matmul(ex_t, se, transpose_x=True), dtype=nl.float32)   # [128, NT]
            so_full = nl.copy(nl.matmul(ex_t, so, transpose_x=True), dtype=nl.float32)
            w_even = nl.copy(e2m1_value(nl.bitwise_and(w8, 15)) * se_full, dtype=nl.bfloat16)
            w_odd = nl.copy(e2m1_value(nl.right_shift(w8, 4)) * so_full, dtype=nl.bfloat16)
            acc_e += nl.matmul(x_t, w_even, transpose_x=True)
            acc_o += nl.matmul(x_t, w_odd, transpose_x=True)
        res_e = nl.copy(acc_e, dtype=nl.float32)
        res_o = nl.copy(acc_o, dtype=nl.float32)
        # interleave: out[:, 2j] = even, out[:, 2j + 1] = odd
        i_b = nl.arange(B)[:, None]
        i_j = nl.arange(NT)[None, :]
        nl.store(out[i_b, 2 * (jt * NT + i_j)], value=res_e)
        nl.store(out[i_b, 2 * (jt * NT + i_j) + 1], value=res_o)
    return out


@nki.jit
def nvfp4_linear_lut(xT, wp, sce, sco, ex, lut):
    """Same as nvfp4_linear, but the E2M1 -> value step is one GPSIMD table lookup per nibble plane
    (lut: bf16 [128, 16], the 16 E2M1 values replicated on every partition) instead of ~15 vector ops."""
    K, B = xT.shape
    KT, P, HALF = wp.shape
    N = HALF * 2
    out = nl.ndarray((B, N), dtype=nl.float32, buffer=nl.shared_hbm)
    ex_t = nl.load(ex)
    lut_t = nl.load(lut)
    for jt in nl.affine_range(HALF // NT):
        acc_e = nl.zeros((B, NT), dtype=nl.float32, buffer=nl.psum)
        acc_o = nl.zeros((B, NT), dtype=nl.float32, buffer=nl.psum)
        for kt in nl.affine_range(KT):
            x_t = nl.load(xT[kt * 128:(kt + 1) * 128, 0:B])
            w8 = nl.load(wp[kt, 0:128, jt * NT:(jt + 1) * NT])
            se = nl.load(sce[kt * 8:(kt + 1) * 8, jt * NT:(jt + 1) * NT])
            so = nl.load(sco[kt * 8:(kt + 1) * 8, jt * NT:(jt + 1) * NT])
            se_full = nl.copy(nl.matmul(ex_t, se, transpose_x=True), dtype=nl.bfloat16)
            so_full = nl.copy(nl.matmul(ex_t, so, transpose_x=True), dtype=nl.bfloat16)
            v_even = nl.gather_flattened(lut_t, nl.copy(nl.bitwise_and(w8, 15), dtype=nl.uint32), dtype=nl.bfloat16)
            v_odd = nl.gather_flattened(lut_t, nl.copy(nl.right_shift(w8, 4), dtype=nl.uint32), dtype=nl.bfloat16)
            w_even = nl.multiply(v_even, se_full, dtype=nl.bfloat16)
            w_odd = nl.multiply(v_odd, so_full, dtype=nl.bfloat16)
            acc_e += nl.matmul(x_t, w_even, transpose_x=True)
            acc_o += nl.matmul(x_t, w_odd, transpose_x=True)
        res_e = nl.copy(acc_e, dtype=nl.float32)
        res_o = nl.copy(acc_o, dtype=nl.float32)
        i_b = nl.arange(B)[:, None]
        i_j = nl.arange(NT)[None, :]
        nl.store(out[i_b, 2 * (jt * NT + i_j)], value=res_e)
        nl.store(out[i_b, 2 * (jt * NT + i_j) + 1], value=res_o)
    return out


def nvfp4_lut():
    """bf16 [128, 16] E2M1 value table replicated on every partition (input of nvfp4_linear_lut)."""
    import ml_dtypes
    import numpy as np
    pos = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    return np.tile(np.concatenate((pos, -pos)), (128, 1)).astype(ml_dtypes.bfloat16)


def nvfp4_pack(packed, scales, global_scale):
    """Checkpoint NVFP4 tensors -> kernel layout.

    packed uint8 [N, K/2] (two consecutive K per byte, low nibble first), scales E4M3 [N, K/16],
    global_scale float. Returns (wp, sce, sco, ex) as numpy arrays (bf16 as ml_dtypes.bfloat16)."""
    import ml_dtypes
    import numpy as np
    n, half_k = packed.shape
    k = half_k * 2
    codes = np.empty((n, k), np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    wt = codes.T.reshape(k // 128, 128, n)                      # [kt, p, n]
    wp = (wt[:, :, 0::2] | (wt[:, :, 1::2] << 4)).astype(np.uint8)
    block = scales.astype(np.float32).T * np.float32(global_scale)   # [K/16, N]
    sce = block[:, 0::2].astype(ml_dtypes.bfloat16)
    sco = block[:, 1::2].astype(ml_dtypes.bfloat16)
    ex = (np.arange(128)[None, :] // 16 == np.arange(8)[:, None]).astype(ml_dtypes.bfloat16)
    return wp, sce, sco, ex
