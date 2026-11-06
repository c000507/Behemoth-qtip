"""Correctness-first NKI building blocks for Llama inference.

These kernels operate on small static tiles. They are not yet a serving
backend and have not been validated on device.
"""
import nki
import nki.language as nl


@nki.jit
def rms_norm(x, weights, epsilon: float):
    batch, width = x.shape
    if weights.shape != (1, width) or width < 1:
        raise ValueError("Invalid RMSNorm shapes")
    out = nl.ndarray((batch, width), dtype=nl.float32, buffer=nl.shared_hbm)
    scale = nl.load(weights[0:1, 0:width])
    for b in range(batch):
        v = nl.load(x[b:b+1, 0:width])
        ss = nl.sum(nl.multiply(v, v), axis=1, keepdims=True)
        inv = nl.rsqrt(nl.add(nl.multiply(ss, 1.0 / width), epsilon))
        nl.store(out[b:b+1, 0:width], nl.multiply(nl.multiply(v, inv), scale))
    return out


@nki.jit
def swiglu(gate, up):
    batch, width = gate.shape
    if up.shape != (batch, width):
        raise ValueError("Invalid SwiGLU shapes")
    out = nl.ndarray((batch, width), dtype=nl.float32, buffer=nl.shared_hbm)
    for b in range(batch):
        g = nl.load(gate[b:b+1, 0:width])
        u = nl.load(up[b:b+1, 0:width])
        nl.store(out[b:b+1, 0:width], nl.multiply(nl.silu(g), u))
    return out


@nki.jit
def apply_rope(x, cos, sin):
    """Llama half-split rotary embedding. Each row represents one head."""
    rows, dim = x.shape
    if dim % 2 or cos.shape != (rows, dim // 2) or sin.shape != cos.shape:
        raise ValueError("Invalid RoPE shapes")
    half = dim // 2
    out = nl.ndarray((rows, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    for r in range(rows):
        a = nl.load(x[r:r+1, 0:half])
        b = nl.load(x[r:r+1, half:dim])
        c = nl.load(cos[r:r+1, 0:half])
        s = nl.load(sin[r:r+1, 0:half])
        nl.store(out[r:r+1, 0:half], nl.subtract(nl.multiply(a, c), nl.multiply(b, s)))
        nl.store(out[r:r+1, half:dim], nl.add(nl.multiply(b, c), nl.multiply(a, s)))
    return out


@nki.jit
def attention_chunk(q, keys, values, additive_mask, inv_sqrt_dim: float):
    """One head and one nonempty cache tile; return unnormalized softmax stats.

    keys/values are [tokens, dim], q [1,dim], additive_mask [1,tokens].
    tokens and dim must fit the 128-partition SBUF/transpose constraints.
    Tiles are merged using the standard max-rescaled online softmax algorithm.
    """
    tokens, dim = keys.shape
    if tokens < 1 or tokens > 128 or dim < 1 or dim > 128:
        raise ValueError("Attention tile must be between 1 and 128 in each dimension")
    if values.shape != (tokens, dim) or q.shape != (1, dim) or additive_mask.shape != (1, tokens):
        raise ValueError("Invalid attention tile shapes")
    result = nl.ndarray((1, dim + 2), dtype=nl.float32, buffer=nl.shared_hbm)
    query = nl.broadcast_to(nl.load(q[0:1, 0:dim]), (tokens, dim))
    ktile = nl.load(keys[0:tokens, 0:dim])
    scores_col = nl.sum(nl.multiply(query, ktile), axis=1, keepdims=True)
    scores = nl.add(nl.multiply(nl.copy(nl.transpose(scores_col)), inv_sqrt_dim),
                    nl.load(additive_mask[0:1, 0:tokens]))
    maximum = nl.max(scores, axis=1, keepdims=True)
    weights = nl.exp(nl.subtract(scores, maximum))
    denom = nl.sum(weights, axis=1, keepdims=True)
    vtile = nl.copy(nl.transpose(nl.load(values[0:tokens, 0:dim])))
    numer_col = nl.sum(nl.multiply(vtile, nl.broadcast_to(weights, (dim, tokens))), axis=1, keepdims=True)
    nl.store(result[0:1, 0:dim], nl.copy(nl.transpose(numer_col)))
    nl.store(result[0:1, dim:dim+1], maximum)
    nl.store(result[0:1, dim+1:dim+2], denom)
    return result


@nki.jit
def fp8_project_loop(weights, scales, x, width: int, out_width: int, batch: int):
    """Serial, tiled E4M3 weight-only projection on already-rounded inputs.

    weights and inputs are flattened [1,N] HBM tensors. FP8 scale is per
    output channel. Width must be a multiple of 32; this baseline tiles K in
    blocks of 32 to keep SBUF use independent of model width.
    """
    import nki.isa as nisa
    if width < 1 or width % 32 or weights.shape != (1, out_width * width):
        raise ValueError("Invalid FP8 weight shape")
    if x.shape != (1, batch * width) or scales.shape != (1, out_width):
        raise ValueError("Invalid FP8 projection shapes")
    out = nl.ndarray((1, batch * out_width), dtype=nl.float32, buffer=nl.shared_hbm)
    def batch_loop(b):
        def output_loop(o):
            total = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            def input_loop(k):
                w = nl.load(weights[0:1, nl.ds(o*width + k*32, 32)], dtype=nl.float32)
                v = nl.load(x[0:1, nl.ds(b*width + k*32, 32)])
                product = nl.sum(nl.multiply(w, v), axis=1, keepdims=True)
                nisa.tensor_tensor(dst=total, data1=total, data2=product, op=nl.add)
            nl.fori_loop(0, width // 32, input_loop)
            scale = nl.load(scales[0:1, nl.ds(o, 1)])
            nl.store(out[0:1, nl.ds(b*out_width + o, 1)], nl.multiply(total, scale))
        nl.fori_loop(0, out_width, output_loop)
    nl.fori_loop(0, batch, batch_loop)
    return out


@nki.jit
def nvfp4_project_loop(packed, block_scales, x, byte_index, nibble_shift, codebook,
                       global_scale: float, width: int, out_width: int, batch: int):
    """Serial NVFP4 E2M1 projection with per-16 E4M3 and global scales."""
    import nki.isa as nisa
    if width < 1 or width % 16 or packed.shape != (1, out_width * width // 2):
        raise ValueError('Invalid NVFP4 packed weights')
    if block_scales.shape != (1, out_width * width // 16) or x.shape != (1, batch * width):
        raise ValueError('Invalid NVFP4 block scales or inputs')
    out = nl.ndarray((1, batch * out_width), dtype=nl.float32, buffer=nl.shared_hbm)
    ix = nl.load(byte_index[0:1, 0:16]); shifts = nl.load(nibble_shift[0:1, 0:16])
    lut = nl.load(codebook[0:1, 0:16])
    def batch_loop(b):
        def output_loop(o):
            total = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            def input_loop(k):
                words = nl.load(packed[0:1, nl.ds(o * width // 2 + k * 8, 8)], dtype=nl.uint32)
                codes = nl.bitwise_and(nl.right_shift(nl.gather_flattened(words, ix), shifts), 15)
                weights = nl.gather_flattened(lut, codes)
                values = nl.load(x[0:1, nl.ds(b * width + k * 16, 16)])
                scale = nl.load(block_scales[0:1, nl.ds(o * (width // 16) + k, 1)], dtype=nl.float32)
                weights = nl.multiply(nl.multiply(weights, scale), global_scale)
                weights = nl.copy(nl.copy(weights, dtype=nl.bfloat16), dtype=nl.float32)
                partial = nl.sum(nl.multiply(weights, values), axis=1, keepdims=True)
                nisa.tensor_tensor(dst=total, data1=total, data2=partial, op=nl.add)
            nl.fori_loop(0, width // 16, input_loop)
            nl.store(out[0:1, nl.ds(b*out_width + o, 1)], total)
        nl.fori_loop(0, out_width, output_loop)
    nl.fori_loop(0, batch, batch_loop)
    return out
