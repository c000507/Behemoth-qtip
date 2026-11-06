"""Host-side QTIP block layout helpers for the experimental NKI decoder."""
import numpy as np


def indices():
    t = np.arange(128, dtype=np.uint32)
    # [j1][i1]: tile_col then tile_row, matching interleaved word order.
    offset = np.arange(4, dtype=np.uint32)[:, None]
    first = (t[None, :] // 4) * 4 + offset
    second = ((t[None, :] // 4 + 1) % 32) * 4 + offset
    shift = np.broadcast_to(16 - 4 * (t % 4), (1, 128)).copy()
    return first, second, shift


def reconstruct(pairs):
    """Deswizzle state-order BF16 pairs into a 32x32 matrix."""
    result = np.empty((32, 32), dtype=np.float32)
    for j1 in range(2):
        for i1 in range(2):
            for t in range(128):
                a, b, c, d = t >> 4, (t >> 2) & 3, (t >> 1) & 1, t & 1
                row, col = i1 * 16 + d * 8 + a, j1 * 16 + c * 8 + b * 2
                p = int(pairs[j1 * 2 + i1, t])
                result[row, col] = np.array([(p & 0xffff) << 16], dtype=np.uint32).view(np.float32)[0]
                result[row, col + 1] = np.array([p & 0xffff0000], dtype=np.uint32).view(np.float32)[0]
    return result



def projection_indices():
    """Indices to decode each element of a 32x32 block in row-major order."""
    row = np.arange(32, dtype=np.uint32)[:, None]
    col = np.arange(32, dtype=np.uint32)[None, :]
    tile = (col // 16) * 2 + row // 16
    a = row % 8
    b = (col % 8) // 2
    c = (col % 16) // 8
    d = (row % 16) // 8
    t = a * 16 + b * 4 + c * 2 + d
    word = (t // 4) * 4 + tile
    following = ((t // 4 + 1) % 32) * 4 + tile
    shift = np.broadcast_to(16 - 4 * (t % 4), (32, 32)).copy()
    component_shift = np.broadcast_to((col % 2) * 16, (32, 32)).copy()
    return word.astype(np.uint32), following.astype(np.uint32), shift, component_shift

def hadamard_indices(width):
    """Gather indices and signs for a natural-order power-of-two FWHT."""
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    columns = np.arange(width, dtype=np.uint32)
    partners, signs = [], []
    for i in range(width.bit_length() - 1):
        bit = 1 << i
        partners.append(columns ^ bit)
        signs.append(np.where(columns & bit, -1.0, 1.0).astype(np.float32))
    return np.asarray(partners, dtype=np.uint32).reshape(-1, width), np.asarray(signs, dtype=np.float32).reshape(-1, width)


def factored_hadamard_indices(width, matrix, transpose=False):
    """Tables for (H_k ⊗ H_p) / sqrt(width), using the QTIP H_k matrix."""
    matrix = np.asarray(matrix, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hadamard factor must be square")
    k = matrix.shape[0]
    if k not in (12, 28) or width % k:
        raise ValueError("Expected a QTIP factor 12 or 28")
    p = width // k
    if p < 2 or p & (p - 1):
        raise ValueError("Hadamard power-of-two factor must be at least 2")
    columns = np.arange(width, dtype=np.uint32)
    partners = np.stack([columns ^ (1 << i) for i in range(p.bit_length()-1)])
    signs = np.stack([np.where(columns & (1 << i), -1., 1.).astype(np.float32)
                      for i in range(p.bit_length()-1)])
    rows = columns // p
    offsets = columns % p
    mix_indices = np.stack([i * p + offsets for i in range(k)]).astype(np.uint32)
    h = matrix.T if transpose else matrix
    mix_signs = np.stack([h[rows, i] for i in range(k)]).astype(np.float32)
    return partners, signs, mix_indices, mix_signs
