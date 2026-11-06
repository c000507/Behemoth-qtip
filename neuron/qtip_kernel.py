"""Experimental single 32x32 QTIP block decoder for NKI.

The output contains four streams of 128 packed BF16 pairs, in trellis state
order. Deswizzling and projection are separate steps. This is a correctness
prototype; single-partition gathers are not a performance design.
"""
import nki
import nki.language as nl


@nki.jit
def decode_block(trellis, lut, word_index, next_index, shift):
    # [1,128] contains the four interleaved 16x16 tiles of one 32x32 block.
    # Indices [4,128] address words in this block, including circular wrap.
    out = nl.ndarray((4, 128), dtype=nl.uint32, buffer=nl.shared_hbm)
    for tile in range(4):
        packed = nl.load(trellis[0:1, 0:128], dtype=nl.uint32)
        table = nl.load(lut[0:1, 0:1024])
        wi = nl.load(word_index[tile:tile + 1, 0:128])
        ni = nl.load(next_index[tile:tile + 1, 0:128])
        sh = nl.load(shift[0:1, 0:128])
        first = nl.gather_flattened(packed, wi)
        second = nl.gather_flattened(packed, ni)
        stream = nl.bitwise_or(nl.left_shift(first, 16), second)
        state = nl.bitwise_and(nl.right_shift(stream, sh), 65535)
        entry = nl.bitwise_and(nl.right_shift(nl.multiply(state, nl.add(state, 1)), 6), 1023)
        values = nl.gather_flattened(table, entry)
        nl.store(out[tile:tile + 1, 0:128], values)
    return out


@nki.jit
def project_block(trellis, lut, word_index, next_index, shift, component_shift, x):
    """Decode a 32x32 block and multiply by one already-rotated input vector.

    lut is a pre-replicated uint32 codebook [32,1024] (one per partition).
    x is FP32 holding the BF16-rounded rotated input. The result is an FP32
    32-row partial sum; caller combines K blocks and applies the output transform.
    """
    out = nl.ndarray((32, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    packed = nl.broadcast_to(nl.load(trellis[0:1, 0:128], dtype=nl.uint32), (32, 128))
    table = nl.load(lut[0:32, 0:1024])
    first = nl.gather_flattened(packed, nl.load(word_index[0:32, 0:32]))
    second = nl.gather_flattened(packed, nl.load(next_index[0:32, 0:32]))
    sh = nl.load(shift[0:32, 0:32])
    cs = nl.load(component_shift[0:32, 0:32])
    stream = nl.bitwise_or(nl.left_shift(first, 16), second)
    state = nl.bitwise_and(nl.right_shift(stream, sh), 65535)
    entry = nl.bitwise_and(nl.right_shift(nl.multiply(state, nl.add(state, 1)), 6), 1023)
    pairs = nl.gather_flattened(table, entry)
    bits = nl.left_shift(nl.bitwise_and(nl.right_shift(pairs, cs), 65535), 16)
    weights = bits.view(nl.float32)
    inputs = nl.broadcast_to(nl.load(x[0:1, 0:32]), (32, 32))
    sums = nl.sum(nl.multiply(weights, inputs), axis=1, keepdims=True)
    nl.store(out[0:32, 0:1], sums)
    return out


@nki.jit
def project_rotated(trellis, lut, word_index, next_index, shift, component_shift, x):
    """Small static-size QTIP matrix-vector prototype on rotated FP32 inputs.

    trellis is [out/32, (in/32)*128] in checkpoint block order; x is
    [batch,in]. Both dimensions must be multiples of 32. The returned
    [batch,out] partial projection still needs Hadamard and SV. This prototype
    traces Python loops and is intended only for small validation shapes.
    """
    batch, width = x.shape
    block_rows = trellis.shape[0]
    if width == 0 or width % 32 or trellis.shape[1] != (width // 32) * 128:
        raise ValueError("QTIP rotated projection requires whole 32-column blocks")
    block_cols = width // 32
    out = nl.ndarray((batch, block_rows * 32), dtype=nl.float32, buffer=nl.shared_hbm)
    table = nl.load(lut[0:32, 0:1024])
    wi = nl.load(word_index[0:32, 0:32])
    ni = nl.load(next_index[0:32, 0:32])
    sh = nl.load(shift[0:32, 0:32])
    cs = nl.load(component_shift[0:32, 0:32])
    for b in range(batch):
        for row in range(block_rows):
            total = nl.zeros((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            for col in range(block_cols):
                words = nl.load(trellis[row:row + 1, col*128:(col+1)*128], dtype=nl.uint32)
                packed = nl.broadcast_to(words, (32, 128))
                first = nl.gather_flattened(packed, wi)
                second = nl.gather_flattened(packed, ni)
                stream = nl.bitwise_or(nl.left_shift(first, 16), second)
                state = nl.bitwise_and(nl.right_shift(stream, sh), 65535)
                entry = nl.bitwise_and(nl.right_shift(nl.multiply(state, nl.add(state, 1)), 6), 1023)
                pairs = nl.gather_flattened(table, entry)
                bits = nl.left_shift(nl.bitwise_and(nl.right_shift(pairs, cs), 65535), 16)
                inputs = nl.broadcast_to(nl.load(x[b:b+1, col*32:(col+1)*32]), (32, 32))
                sums = nl.sum(nl.multiply(bits.view(nl.float32), inputs), axis=1, keepdims=True)
                total = nl.add(total, sums)
            nl.store(out[b:b+1, row*32:(row+1)*32], nl.copy(nl.transpose(total)))
    return out


@nki.jit
def transform_pow2(x, scale, partners, signs, scale_before: bool, round_bf16: bool):
    """Power-of-two normalized Hadamard and SU/SV scaling, for small widths.

    For QTIP's input transform, scale_before=True and round_bf16=True. For
    the output transform, both are False. Other Hadamard factors are not yet
    supported. Accumulation and scaling here use FP32.
    """
    batch, width = x.shape
    if width < 2 or (width & (width - 1)) or partners.shape[1] != width:
        raise ValueError("Expected a power-of-two Hadamard width of at least 2")
    if partners.shape[0] != width.bit_length() - 1 or signs.shape != partners.shape:
        raise ValueError("Invalid Hadamard stage tables")
    out = nl.ndarray((batch, width), dtype=nl.float32, buffer=nl.shared_hbm)
    factors = nl.load(scale[0:1, 0:width])
    for row in range(batch):
        value = nl.load(x[row:row+1, 0:width])
        if scale_before:
            value = nl.multiply(value, factors)
        for stage in range(width.bit_length() - 1):
            ix = nl.load(partners[stage:stage+1, 0:width])
            sign = nl.load(signs[stage:stage+1, 0:width])
            value = nl.add(nl.multiply(value, sign), nl.gather_flattened(value, ix))
        value = nl.multiply(value, 1.0 / (width ** 0.5))
        if not scale_before:
            value = nl.multiply(value, factors)
        if round_bf16:
            value = nl.copy(nl.copy(value, dtype=nl.bfloat16), dtype=nl.float32)
        nl.store(out[row:row+1, 0:width], value)
    return out


@nki.jit
def transform_factored(x, scale, partners, signs, mix_indices, mix_signs,
                       scale_before: bool, round_bf16: bool):
    """Small factor-12/28 Hadamard prototype; supplied signs choose transpose."""
    batch, width = x.shape
    factor = mix_indices.shape[0]
    if factor not in (12, 28) or width % factor or mix_indices.shape[1] != width:
        raise ValueError("Expected a QTIP factor-12 or factor-28 Hadamard")
    p = width // factor
    if p < 2 or p & (p - 1) or partners.shape != signs.shape:
        raise ValueError("Invalid QTIP Hadamard stage tables")
    if partners.shape != (p.bit_length() - 1, width) or mix_indices.shape != mix_signs.shape:
        raise ValueError("Invalid QTIP Hadamard mix tables")
    out = nl.ndarray((batch, width), dtype=nl.float32, buffer=nl.shared_hbm)
    scales = nl.load(scale[0:1, 0:width])
    for row in range(batch):
        value = nl.load(x[row:row+1, 0:width])
        if scale_before:
            value = nl.multiply(value, scales)
        for stage in range(p.bit_length() - 1):
            ix = nl.load(partners[stage:stage+1, 0:width])
            sign = nl.load(signs[stage:stage+1, 0:width])
            value = nl.add(nl.multiply(value, sign), nl.gather_flattened(value, ix))
        mixed = nl.zeros((1, width), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(factor):
            ix = nl.load(mix_indices[i:i+1, 0:width])
            sign = nl.load(mix_signs[i:i+1, 0:width])
            mixed = nl.add(mixed, nl.multiply(nl.gather_flattened(value, ix), sign))
        value = nl.multiply(mixed, 1.0 / (width ** 0.5))
        if not scale_before:
            value = nl.multiply(value, scales)
        if round_bf16:
            value = nl.copy(nl.copy(value, dtype=nl.bfloat16), dtype=nl.float32)
        nl.store(out[row:row+1, 0:width], value)
    return out


@nki.jit
def project_rotated_loop(trellis, lut, word_index, next_index, shift, component_shift, x, width: int, block_rows: int, batch: int):
    """QTIP projection using compiled loops rather than unrolling model dimensions.

    One core processes the blocks sequentially; this is a correctness-first
    baseline, not an optimized parallel implementation.
    """
    import nki.isa as nisa

    if width == 0 or width % 32 or trellis.shape != (1, block_rows * width * 4) or x.shape != (1, batch * width):
        raise ValueError("QTIP rotated projection requires whole 32-column blocks")
    out = nl.ndarray((1, batch * block_rows * 32), dtype=nl.float32, buffer=nl.shared_hbm)
    table = nl.load(lut[0:32, 0:1024])
    wi = nl.load(word_index[0:32, 0:32])
    ni = nl.load(next_index[0:32, 0:32])
    sh = nl.load(shift[0:32, 0:32])
    cs = nl.load(component_shift[0:32, 0:32])

    def batch_loop(b):
        def row_loop(row):
            total = nl.zeros((32, 1), dtype=nl.float32, buffer=nl.sbuf)
            def col_loop(col):
                words = nl.load(trellis[0:1, nl.ds(row * width * 4 + col * 128, 128)], dtype=nl.uint32)
                packed = nl.broadcast_to(words, (32, 128))
                first = nl.gather_flattened(packed, wi)
                second = nl.gather_flattened(packed, ni)
                stream = nl.bitwise_or(nl.left_shift(first, 16), second)
                state = nl.bitwise_and(nl.right_shift(stream, sh), 65535)
                entry = nl.bitwise_and(nl.right_shift(nl.multiply(state, nl.add(state, 1)), 6), 1023)
                pairs = nl.gather_flattened(table, entry)
                bits = nl.left_shift(nl.bitwise_and(nl.right_shift(pairs, cs), 65535), 16)
                inputs = nl.broadcast_to(nl.load(x[0:1, nl.ds(b * width + col * 32, 32)]), (32, 32))
                sums = nl.sum(nl.multiply(bits.view(nl.float32), inputs), axis=1, keepdims=True)
                nisa.tensor_tensor(dst=total, data1=total, data2=sums, op=nl.add)
            nl.fori_loop(0, width // 32, col_loop)
            nl.store(out[0:1, nl.ds(b * block_rows * 32 + row * 32, 32)], nl.copy(nl.transpose(total)))
        nl.fori_loop(0, block_rows, row_loop)
    nl.fori_loop(0, batch, batch_loop)
    return out


@nki.jit
def hadamard_chunks(x, scale, partners, signs, width: int, factor: int, batch: int, scale_before: bool):
    """Compute normalized power-of-two Hadamards in bounded SBUF chunks."""
    p = width // factor
    if factor not in (12, 28) or width % factor or p < 2 or p & (p-1):
        raise ValueError('Invalid chunked QTIP Hadamard geometry')
    if x.shape != (1, batch * width) or scale.shape != (1, width):
        raise ValueError('Invalid chunked QTIP input or scale')
    if partners.shape != (p.bit_length()-1, p) or signs.shape != partners.shape:
        raise ValueError('Invalid FWHT index tables')
    out = nl.ndarray((1, batch * width), dtype=nl.float32, buffer=nl.shared_hbm)
    def batch_loop(b):
        def chunk_loop(chunk):
            value = nl.load(x[0:1, nl.ds(b * width + chunk * p, p)])
            if scale_before:
                value = nl.multiply(value, nl.load(scale[0:1, nl.ds(chunk * p, p)]))
            for stage in range(p.bit_length()-1):
                ix = nl.load(partners[stage:stage+1, 0:p])
                sign = nl.load(signs[stage:stage+1, 0:p])
                value = nl.add(nl.multiply(value, sign), nl.gather_flattened(value, ix))
            nl.store(out[0:1, nl.ds(b * width + chunk * p, p)], nl.multiply(value, 1. / (p ** .5)))
        nl.fori_loop(0, factor, chunk_loop)
    nl.fori_loop(0, batch, batch_loop)
    return out


@nki.jit
def hadamard_mix(x, scale, matrix, width: int, factor: int, batch: int,
                 scale_before: bool, round_bf16: bool):
    """Apply the small Hadamard factor using bounded-SBUF compiled loops."""
    import nki.isa as nisa
    p = width // factor
    if factor not in (12, 28) or width % factor or x.shape != (1, batch * width):
        raise ValueError('Invalid QTIP factor mix geometry')
    if scale.shape != (1, width) or matrix.shape != (1, factor * factor):
        raise ValueError('Invalid QTIP factor mix table')
    out = nl.ndarray((1, batch * width), dtype=nl.float32, buffer=nl.shared_hbm)
    def batch_loop(b):
        def row_loop(row):
            total = nl.zeros((1, p), dtype=nl.float32, buffer=nl.sbuf)
            def column_loop(col):
                v = nl.load(x[0:1, nl.ds(b * width + col * p, p)])
                sign = nl.load(matrix[0:1, nl.ds(row * factor + col, 1)])
                nisa.tensor_tensor(dst=total, data1=total, data2=nl.multiply(v, sign), op=nl.add)
            nl.fori_loop(0, factor, column_loop)
            value = nl.multiply(total, 1. / (factor ** .5))
            if not scale_before:
                value = nl.multiply(value, nl.load(scale[0:1, nl.ds(row * p, p)]))
            if round_bf16:
                value = nl.copy(nl.copy(value, dtype=nl.bfloat16), dtype=nl.float32)
            nl.store(out[0:1, nl.ds(b * width + row * p, p)], value)
        nl.fori_loop(0, factor, row_loop)
    nl.fori_loop(0, batch, batch_loop)
    return out
