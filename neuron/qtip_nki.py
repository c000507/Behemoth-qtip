"""QTIP trellis decode fused with the projection matmul, as an NKI kernel (neuronxcc.nki, traces inside
torch_neuronx.trace). The weights stay in HBM as the checkpoint's 2-bit trellis words; each step the kernel decodes
them tile by tile into SBUF (bf16, the device codebook values) and feeds the Tensor Engine, so the dense matrix
never exists in HBM and a 70B layer costs ~107 MiB per rank instead of 428 MiB.

Format (runtime/include/behemoth/qtip.hpp): a 16x16 tile is a 512-bit tail-biting stream of 32 16-bit words;
weight pair t (0..127) is codebook entry e = ((s (s + 1)) >> 6) & 1023 of the 16-bit window s at bit 4t, and lands
at row D*8 + A, columns C*8 + B*2 + {0, 1} of the tile for t = A*16 + B*4 + C*2 + D. The codebook is 1024 x
(bf16, bf16) = one uint32 per entry (qtip::Codebook::bf16x2).

Device layout (host repack, `pack_shard`): words uint16 [G, 128, O*32]: partition p of in-group g holds the 32
words of every output tile for input channels g*2048 + 16p + c (c = 0..15), so one DMA brings a [128, NT*32]
block whose partition axis is the matmul's contraction axis. The kernel, per (output chunk of NT tiles, in-group):
  u_i   = w_i << 16 | w_(i+1 mod 32)                      (two instructions over strided views)
  v_k   = u >> (16 - 4k)                                  k = 0..3: the window of pair t = 4i + k (upper bits are
  p_k   = v_k * (v_k + 1)                                 harmless: only bits 0..15 of p are used, and int32 wraps)
  e_k   = (p_k >> 6) & 1023
  pair  = lut[e_k]                                        GpSimd gather into a contiguous k-major slice
then 16 matmuls (one per in-channel offset c) read a strided bf16 view of the pairs (rows A, D) as the
moving operand against x^T[c] as the stationary operand and accumulate z[T, 16 NT] in PSUM over the in-groups.
Output rows come out in (A, D) order, i.e. out = tile*16 + D*8 + A is stored through a strided view, so z is in
natural order. x^T is the rotated activation u = H_in^T (SU x), bf16, laid out [128, G, 16, T] (p, g, c, t).
"""
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

NT = 32         # output tiles per chunk: 16 * NT = 512 columns = one PSUM bank of f32 per token
IN_GROUP = 2048  # input channels per 128-partition group (16 per partition)


def tiles_of(trellis, out, width):
    """Checkpoint trellis (int16 [(out/16)*(width/16), 32] or flat) -> uint16 [out/16, width/16, 32] words per tile
    (HostLinear::word: block (i0, j0) holds its 2x2 tiles as [c][j1][i1])."""
    t = np.asarray(trellis).view(np.uint16).reshape(out // 32, width // 32, 32, 2, 2)
    return np.ascontiguousarray(t.transpose(0, 4, 1, 3, 2)).reshape(out // 16, width // 16, 32)


def pack_tiles(tiles):
    """[O, I, 32] tile words -> device layout uint16 [G, 128, O*32] (the last in-group zero padded when I is not a
    multiple of 128: zero words decode to lut[0] and the x^T padding is zero, so padded channels add nothing)."""
    o_tiles, i_tiles = tiles.shape[0], tiles.shape[1]
    if o_tiles % NT:
        raise ValueError(f'output tiles {o_tiles} must be a multiple of {NT}')
    groups = (i_tiles + 127) // 128
    packed = np.zeros((groups * 128, o_tiles, 32), np.uint16)
    packed[:i_tiles] = tiles.transpose(1, 0, 2)
    return np.ascontiguousarray(packed.reshape(groups, 128, o_tiles * 32))


def shard_tiles(tiles, tp, rank, column_parallel):
    if column_parallel:
        o = tiles.shape[0] // tp
        return tiles[rank * o:(rank + 1) * o]
    i = tiles.shape[1] // tp
    return tiles[:, rank * i:(rank + 1) * i]


def pack_shard(trellis, out, width, tp, rank, column_parallel):
    """This rank's words in device layout (column-parallel: output rows sharded; row-parallel: input columns)."""
    return pack_tiles(shard_tiles(tiles_of(trellis, out, width), tp, rank, column_parallel))


def codebook_u32(tlut_f16_bits):
    """qtip::make_codebook: entry e -> (bf16(c0 * sign), bf16(c1)) as one uint32 (low half = component 0),
    replicated on 128 partitions: uint32 [128, 1024]."""
    import ml_dtypes
    tl = np.asarray(tlut_f16_bits).view(np.uint16).reshape(512, 2)
    e = np.arange(1024)
    h0 = (tl[e & 511, 0] ^ np.where(e >> 9, 0x8000, 0)).astype(np.uint16)
    h1 = tl[e & 511, 1]
    c0 = h0.view(np.float16).astype(np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)
    c1 = h1.view(np.float16).astype(np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)
    lut = c0.astype(np.uint32) | (c1.astype(np.uint32) << 16)
    return np.ascontiguousarray(np.tile(lut, (128, 1)).astype(np.uint32))


def xt_layout(u, groups):
    """Rotated activation u [T, in_local] -> x^T [128, G, 16, T] for the kernel (zero padded ragged group)."""
    T, width = u.shape
    pad = np.zeros((T, groups * IN_GROUP), u.dtype)
    pad[:, :width] = u
    return np.ascontiguousarray(pad.reshape(T, groups, 128, 16).transpose(2, 1, 3, 0))


def decode_reference(tiles, lut_u32):
    """numpy mirror of qtip::decode_rows_bf16 on [O, I, 32] tile words -> float32 [O*16, I*16] (bf16 values)."""
    import ml_dtypes
    w = tiles.astype(np.uint32)
    i = np.arange(128) >> 2
    sh = 16 - 4 * (np.arange(128) & 3)
    u = (w[..., i] << 16) | w[..., (i + 1) & 31]
    s = (u >> sh) & 0xffff
    e = ((s * (s + 1)) >> 6) & 1023
    pair = lut_u32[e]                                                   # [O, I, 128]
    c0 = (pair & 0xffff).astype(np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
    c1 = (pair >> 16).astype(np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
    t = np.arange(128)
    A, B, C, D = t >> 4, (t >> 2) & 3, (t >> 1) & 1, t & 1
    O, I = tiles.shape[:2]
    tile = np.zeros((O, I, 16, 16), np.float32)
    tile[:, :, D * 8 + A, C * 8 + B * 2] = c0
    tile[:, :, D * 8 + A, C * 8 + B * 2 + 1] = c1
    return tile.transpose(0, 2, 1, 3).reshape(O * 16, I * 16)


def decode_pairs(w, lut, nt):
    """w: SBUF uint32 [128, nt*32] words -> SBUF uint32 [128, 4*nt*32] codebook pairs in k-major order: pair
    t = 4 i + k of tile n sits at index k*(nt*32) + n*32 + i (each lookup writes a contiguous slice: a strided gather
    destination runs the GpSimd GATHER an order of magnitude slower)."""
    i_p = nl.arange(128)[:, None, None]
    i_t = nl.arange(nt)[None, :, None]
    i_a = nl.arange(31)[None, None, :]
    i_1 = nl.arange(1)[None, None, :]
    u = nl.ndarray((128, nt * 32), dtype=nl.uint32, buffer=nl.sbuf)
    hi = nisa.tensor_scalar(w[i_p, i_t * 32 + i_a], nl.left_shift, 16, op1=nl.bitwise_or, operand1=0, dtype=nl.uint32)
    u[i_p, i_t * 32 + i_a] = nisa.tensor_tensor(hi, w[i_p, i_t * 32 + i_a + 1], nl.bitwise_or, dtype=nl.uint32)
    hi_last = nisa.tensor_scalar(w[i_p, i_t * 32 + i_1 + 31], nl.left_shift, 16, op1=nl.bitwise_or, operand1=0, dtype=nl.uint32)
    u[i_p, i_t * 32 + i_1 + 31] = nisa.tensor_tensor(hi_last, w[i_p, i_t * 32 + i_1], nl.bitwise_or, dtype=nl.uint32)
    pairs = nl.ndarray((128, 4 * nt * 32), dtype=nl.uint32, buffer=nl.sbuf)
    # unrolled by hand: a Python loop variable inside the traced kernel becomes a traced value, not a constant
    _lookup(u, lut, pairs, nt, 0, 16)
    _lookup(u, lut, pairs, nt, 1, 12)
    _lookup(u, lut, pairs, nt, 2, 8)
    _lookup(u, lut, pairs, nt, 3, 4)
    return pairs


def _lookup(u, lut, pairs, nt, k, shift):
    """pairs[k-th slice] = lut[((v (v + 1)) >> 6) & 1023], v = u_i >> shift (the window of pair t = 4 i + k)."""
    v = nisa.tensor_scalar(u, nl.right_shift, shift, op1=nl.bitwise_and, operand1=0xffff, dtype=nl.uint32)
    # tensor_tensor multiplies uint32 exactly on the hardware; scalar_tensor_tensor rounds the product through fp32
    v1 = nisa.tensor_scalar(v, nl.add, 1, op1=nl.add, operand1=0, dtype=nl.uint32)
    p = nisa.tensor_tensor(v, v1, nl.multiply, dtype=nl.uint32)
    e = nisa.tensor_scalar(p, nl.right_shift, 6, op1=nl.bitwise_and, operand1=1023, dtype=nl.uint32)
    pairs[:, k * nt * 32:(k + 1) * nt * 32] = nl.gather_flattened(lut, e)


TB = 128  # tokens per PSUM accumulator (stationary free size of the Tensor Engine)


def matmul_chunk(accs, wb, xT, g, T):
    """accs[b] PSUM [tb, NT, 8, 2] += x^T[g, block b] . W~ chunk for every 128-token block. wb: bf16 view
    [128, 4*NT*64] of the k-major pairs: value (k, tile n, word i, comp) at k*NT*64 + n*64 + 2 i + comp. In-channel
    offset c = C*8 + B*2 + comp of a tile has k = C*2 + D and i = A*4 + B for its rows (A, D), so one matmul per c
    reads the view at base (C*2)*NT*64 + 2 B + comp with strides A: 8, D: NT*64 (unrolled by hand: a traced loop
    variable cannot index a Python tuple)."""
    _column(accs, wb, xT, g, T, 0, 0)
    _column(accs, wb, xT, g, T, 1, 1)
    _column(accs, wb, xT, g, T, 2, 2)
    _column(accs, wb, xT, g, T, 3, 3)
    _column(accs, wb, xT, g, T, 4, 4)
    _column(accs, wb, xT, g, T, 5, 5)
    _column(accs, wb, xT, g, T, 6, 6)
    _column(accs, wb, xT, g, T, 7, 7)
    _column(accs, wb, xT, g, T, 8, 2 * NT * 64 + 0)
    _column(accs, wb, xT, g, T, 9, 2 * NT * 64 + 1)
    _column(accs, wb, xT, g, T, 10, 2 * NT * 64 + 2)
    _column(accs, wb, xT, g, T, 11, 2 * NT * 64 + 3)
    _column(accs, wb, xT, g, T, 12, 2 * NT * 64 + 4)
    _column(accs, wb, xT, g, T, 13, 2 * NT * 64 + 5)
    _column(accs, wb, xT, g, T, 14, 2 * NT * 64 + 6)
    _column(accs, wb, xT, g, T, 15, 2 * NT * 64 + 7)


def _column(accs, wb, xT, g, T, c, base):
    i_p = nl.arange(128)[:, None, None, None]
    i_nt = nl.arange(NT)[None, :, None, None]
    i_a = nl.arange(8)[None, None, :, None]
    i_d = nl.arange(2)[None, None, None, :]
    view = wb[i_p, i_nt * 64 + i_a * 8 + i_d * (NT * 64) + base]                      # rows (A, D) of column c
    for b, acc in enumerate(accs):
        t0 = b * TB
        x_c = nl.load(xT[:, g, c, t0:min(T, t0 + TB)])                                  # [128, tb]
        acc += nl.matmul(x_c, view, transpose_x=True)


@nki.jit
def qtip_linear(words, lut, xT):
    """z[T, O*16] f32 = u @ W~^T; words uint16 [G, 128, O*32] (pack_tiles), lut uint32 [128, 1024] (codebook_u32),
    xT bf16 [128, G, 16, T] (xt_layout). Tokens beyond 128 run as further 128-token blocks against the same decoded
    chunk (one PSUM bank each, so T <= 1024): the decode is paid once per step, not per token."""
    G, P, OW = words.shape
    O = OW // 32
    T = xT.shape[3]
    blocks = (T + TB - 1) // TB
    out = nl.ndarray((T, O * 16), dtype=nl.float32, buffer=nl.shared_hbm)
    lut_t = nl.load(lut)
    i_nt = nl.arange(NT)[None, :, None, None]
    i_a = nl.arange(8)[None, None, :, None]
    i_d = nl.arange(2)[None, None, None, :]
    for ch in nl.affine_range(O // NT):
        accs = [nl.zeros((min(TB, T - b * TB), NT, 8, 2), dtype=nl.float32, buffer=nl.psum) for b in range(blocks)]
        for g in nl.affine_range(G):
            w = nl.load(words[g, :, ch * NT * 32:(ch + 1) * NT * 32], dtype=nl.uint32)   # [128, NT*32]
            pairs = decode_pairs(w, lut_t, NT)
            matmul_chunk(accs, pairs.view(nl.bfloat16), xT, g, T)
        for b, acc in enumerate(accs):
            o_t = nl.arange(min(TB, T - b * TB))[:, None, None, None]
            nl.store(out[b * TB + o_t, (ch * NT + i_nt) * 16 + i_d * 8 + i_a], acc)
    return out


@nki.jit
def qtip_decode_dense(words, lut):
    """Timing / scratch variant: decode to HBM bf16 [128, G, O*256] (k-major pairs, no matmul)."""
    G, P, OW = words.shape
    O = OW // 32
    out = nl.ndarray((128, G, O * 256), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    lut_t = nl.load(lut)
    for ch in nl.affine_range(O // NT):
        for g in nl.affine_range(G):
            w = nl.load(words[g, :, ch * NT * 32:(ch + 1) * NT * 32], dtype=nl.uint32)
            pairs = decode_pairs(w, lut_t, NT)
            nl.store(out[:, g, ch * NT * 256:(ch + 1) * NT * 256], pairs.view(nl.bfloat16))
    return out
