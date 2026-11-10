"""Tensor Engine binary decoder prototype: packed HBM -> B/D matmul -> W/x matmul.

No gather, no expanded weight matrix in HBM. Only bit unpacking uses vector
operations. Decoder is shared by all blocks in one projection. Shapes are
deliberately restricted: output multiple of 128, input multiple of 64, 1..128
tokens. Not yet wired into the serving engine.
"""
try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    LEGACY = True
except ModuleNotFoundError:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    LEGACY = False


@nki.jit
def binary_linear(packed, decoder, x):
    # packed [K/64,N,16], decoder [128,64] bf16, x [K,T] bf16
    groups, out, nbytes = packed.shape
    width, tokens = x.shape
    assert nbytes == 16 and out % 128 == 0
    assert width == groups * 64 and 1 <= tokens <= 128
    assert decoder.shape == (128, 64)
    y = nl.ndarray((tokens, out), dtype=nl.float32, buffer=nl.shared_hbm)
    d = nl.load(decoder)
    for o in nl.affine_range(out // 128):
        acc = nl.zeros((tokens, 128), dtype=nl.float32, buffer=nl.psum)
        for g in nl.affine_range(groups):
            p = nl.load(packed[g, o * 128:(o + 1) * 128, 0:16])
            b = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            for bit in nl.static_range(8):
                v = nl.bitwise_and(nl.right_shift(p, bit), 1)
                b[:, bit:128:8] = nl.add(nl.multiply(nl.copy(v, dtype=nl.bfloat16), 2.0), -1.0)
            # Unpack along free dimensions: interleaving physical partitions
            # is illegal on Trn1. One Tensor Engine transpose makes latent bits
            # the contraction axis; no GPSIMD gather or HBM expansion.
            if LEGACY:
                bt = nl.copy(nl.transpose(b), dtype=nl.bfloat16)
            else:
                transpose_dtype = nl.float32 if nisa.get_nc_version() == nisa.nc_version.gen2 else nl.bfloat16
                transpose = nl.ndarray((128, 128), dtype=transpose_dtype, buffer=nl.psum)
                nisa.nc_transpose(transpose, b)
                bt = nl.copy(transpose, dtype=nl.bfloat16)
            # d.T @ bt reconstructs [input64,output128] directly.
            if LEGACY:
                decoded = nl.matmul(d, bt, transpose_x=True)
            else:
                decoded = nl.ndarray((64, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(decoded, d, bt, accumulate=False)
            w = nl.copy(decoded, dtype=nl.bfloat16)
            a = nl.load(x[g * 64:(g + 1) * 64, :])
            if LEGACY:
                acc += nl.matmul(a, w, transpose_x=True)
            else:
                nisa.nc_matmul(acc, a, w, accumulate=True)
        nl.store(y[:, o * 128:(o + 1) * 128], nl.copy(acc, dtype=nl.float32))
    return y
