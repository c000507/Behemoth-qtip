"""Paged decode attention as one NKI kernel (one request, one query token, GQA over the server's 16-token pages).

Why: the torch-op decode loop gathers the request's pages into HBM (about 110 us per 8 MB block at 14k context), then runs small dots and
several elementwise passes over [rows x keys] score tiles; the profile put it at ~0.5 ms per layer at 14k context. This kernel gathers the
pages itself (indirect DMA, one half page of one kv head per SBUF partition), keeps all 16 query rows of the request (4 kv heads x 4 query
heads) in the partitions of one score tile, and fuses scale / mask / online softmax / PV.

Layout. The cache is viewed as rows of one half page of one kv head: row = ((page * 2 + kv) * kvh + h) * 2 + half, 8 tokens x 128 dims
(1024 elements) per row, so one DMA brings 128 rows = 64 pages of one head into a [128 partitions, 1024] tile (partition q = 2 j + half,
free = token t * 128 + d). Slice t of that tile is [128 blocks, 128 d] with block q holding token q % 2 * 8 + t of page j = q // 2;
a PE transpose gives K^T [d, 128 blocks]. Scores of the 4 kv heads accumulate into one PSUM tile [16 rows, 512 keys] through q tiles that are
zero except for their own head's 4 columns, so the rows of head h only see K_h. P^T slices [128 blocks, 16 rows] multiply the V slices
([128 blocks, 128 d], same block order) and the rows of head h are picked from the PV result with a 0/1 row mask.
Key order inside a group of 64 pages (1024 keys), half = 0 / 1 of 512 columns, column c = tt * 128 + q, t = 4 half + tt:
key = (q // 2) * 16 + (q % 2) * 8 + t.  The additive mask `bias` [NG, 16, 1024] f32 (0 / -30000) is built by the caller in that order.
"""
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

KVH, REP, D = 4, 4, 128
ROWS = KVH * REP          # 16 query rows
GROUP_PAGES = 64          # pages per group (128 half-page blocks = the 128 partitions)
GROUP_KEYS = GROUP_PAGES * 16


def native_key_positions(groups):
    """Key position (in the request's sequence) of every column of the bias tensor [groups, 16, 1024] (same for every row)."""
    q = np.arange(128)
    pos = np.zeros((groups, 1024), np.int64)
    for half in range(2):
        for tt in range(4):
            t = 4 * half + tt
            cols = half * 512 + tt * 128 + q
            for g in range(groups):
                pos[g, cols] = g * GROUP_KEYS + (q // 2) * 16 + (q % 2) * 8 + t
    return pos


def row_indices(table_pages, kv, groups):  # one request; stack along a leading axis for R requests
    """int32 [KVH, 128, groups]: cache row of partition q = 2 j + half for group g and kv head h (table_pages [groups * 64] page ids)."""
    p = np.asarray(table_pages).reshape(groups, GROUP_PAGES)
    j = np.arange(GROUP_PAGES)[:, None]
    half = np.arange(2)[None, :]
    out = np.zeros((KVH, 128, groups), np.int32)
    for h in range(KVH):
        for g in range(groups):
            rows = ((p[g][:, None] * 2 + kv) * KVH + h) * 2 + half      # [64, 2]
            out[h, :, g] = rows.reshape(128)
    return out


@nki.jit
def decode_attention(qt, cache, idxk, idxv, bias, mask, ksc=None, vsc=None):
    """R requests, one query token each. qt [R, KVH, D, REP] bf16 (scaled by D ** -0.5), cache [blocks, 1024] bf16, idxk / idxv [R, KVH, 128, G] int32,
    bias [R, G, 16, 1024] f32, mask [16, KVH] f32 (1 where row // 4 == h) -> o [R * 16, D] f32 (row = 16 r + 4 h + i).
    FP8 cache (ksc / vsc given, [R, G, 16, 1024] f32 in the same native column order as bias, rows repeated over a head's query heads): cache holds
    E4M3 pages (uint8 bits or a float8 tensor), converted to bf16 in SBUF; the K scale multiplies the scores, the V scale weights the probabilities
    before PV (the softmax sum uses the unweighted p), as the GPU runtime does."""
    R = qt.shape[0]
    G = idxk.shape[3]
    out = nl.ndarray((R * ROWS, D), dtype=nl.float32, buffer=nl.shared_hbm)
    i1, _ = nl.mgrid[:128, :1]
    _, jf = nl.mgrid[:128, :1024]
    rowmask = nl.load(mask)
    for r in nl.affine_range(R):
        # q tiles padded to 16 columns: head h's REP columns at 4 h, zeros elsewhere
        qp = nl.zeros((128, KVH * ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)           # [d, (h, 16)]
        tabk = nl.ndarray((128, KVH * G), dtype=nl.int32, buffer=nl.sbuf)
        tabv = nl.ndarray((128, KVH * G), dtype=nl.int32, buffer=nl.sbuf)
        for h in nl.affine_range(KVH):
            qp[:, nl.ds(h * ROWS + h * REP, REP)] = nl.load(qt[r, h])
            tabk[:, nl.ds(h * G, G)] = nl.load(idxk[r, h])
            tabv[:, nl.ds(h * G, G)] = nl.load(idxv[r, h])
        m = nl.full((ROWS, 1), -30000.0, dtype=nl.float32, buffer=nl.sbuf)
        l = nl.zeros((ROWS, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.zeros((ROWS, D), dtype=nl.float32, buffer=nl.sbuf)
        for g in nl.sequential_range(G):
            kraw = nl.ndarray((128, KVH * 1024), dtype=cache.dtype, buffer=nl.sbuf)
            vraw = nl.ndarray((128, KVH * 1024), dtype=cache.dtype, buffer=nl.sbuf)
            for h in nl.affine_range(KVH):
                nisa.dma_copy(src=cache[tabk[i1, h * G + g], jf], dst=kraw[:, nl.ds(h * 1024, 1024)])
                nisa.dma_copy(src=cache[tabv[i1, h * G + g], jf], dst=vraw[:, nl.ds(h * 1024, 1024)])
            if ksc is not None:                                                         # E4M3 pages -> bf16 (exact), scales for this group
                kb = nl.copy(kraw.view(nl.float8_e4m3) if cache.dtype == nl.uint8 else kraw, dtype=nl.bfloat16)
                vb = nl.copy(vraw.view(nl.float8_e4m3) if cache.dtype == nl.uint8 else vraw, dtype=nl.bfloat16)
                ksc_t = nl.load(ksc[r, g])
                vsc_t = nl.load(vsc[r, g])
            else:
                kb, vb = kraw, vraw
            bias_t = nl.load(bias[r, g])                                                # [16, 1024] f32
            s_sb = nl.ndarray((ROWS, 1024), dtype=nl.float32, buffer=nl.sbuf)
            for half in nl.affine_range(2):
                s_ps = nl.zeros((ROWS, 512), dtype=nl.float32, buffer=nl.psum)
                for h in nl.affine_range(KVH):
                    kt_ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                    for tt in nl.affine_range(4):
                        kt_ps[:, nl.ds(tt * 128, 128)] = nisa.nc_transpose(kb[:, nl.ds(h * 1024 + (half * 4 + tt) * 128, 128)])
                    kt_sb = nl.copy(kt_ps, dtype=nl.bfloat16)
                    s_ps += nl.matmul(qp[:, nl.ds(h * ROWS, ROWS)], kt_sb, transpose_x=True)
                if ksc is not None:
                    s_sb[:, nl.ds(half * 512, 512)] = nl.add(nl.multiply(s_ps, ksc_t[:, nl.ds(half * 512, 512)]), bias_t[:, nl.ds(half * 512, 512)])
                else:
                    s_sb[:, nl.ds(half * 512, 512)] = nl.add(s_ps, bias_t[:, nl.ds(half * 512, 512)])
            mx = nisa.tensor_reduce(np.max, s_sb, axis=(1,), dtype=nl.float32, negate=False)          # [16, 1]
            m_new = nl.maximum(m, mx)
            alpha = nl.exp(nl.subtract(m, m_new))
            p = nl.ndarray((ROWS, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
            psum_l = nl.ndarray((ROWS, 2), dtype=nl.float32, buffer=nl.sbuf)
            for half in nl.affine_range(2):
                p[:, nl.ds(half * 512, 512)] = nisa.activation_reduce(np.exp, s_sb[:, nl.ds(half * 512, 512)], bias=nl.multiply(m_new, -1.0), scale=1.0,
                                                                        reduce_op=nl.add, reduce_res=psum_l[:, nl.ds(half, 1)], dtype=nl.bfloat16)
            l[...] = nl.add(nl.multiply(l, alpha), nl.sum(psum_l, axis=1, keepdims=True))
            pw = nl.multiply(p, vsc_t, dtype=nl.bfloat16) if ksc is not None else p          # V scale weights the probabilities (not the sum)
            # P^T slices [128 blocks, 16 rows] per (half, tt)
            pt_sb = nl.ndarray((128, 8 * ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
            for c in nl.affine_range(8):
                pt_ps = nisa.nc_transpose(pw[:, nl.ds(c * 128, 128)])                       # [128, 16]
                pt_sb[:, nl.ds(c * ROWS, ROWS)] = nl.copy(pt_ps, dtype=nl.bfloat16)
            acc[...] = nl.multiply(acc, alpha)
            for h in nl.sequential_range(KVH):
                pv_ps = nl.zeros((ROWS, D), dtype=nl.float32, buffer=nl.psum)
                for t in nl.affine_range(8):
                    pv_ps += nl.matmul(pt_sb[:, nl.ds(t * ROWS, ROWS)], vb[:, nl.ds(h * 1024 + t * 128, 128)], transpose_x=True)
                acc[...] = nl.add(acc, nl.multiply(pv_ps, rowmask[:, nl.ds(h, 1)]))
            m[...] = m_new
        nl.store(out[nl.ds(r * ROWS, ROWS), :], nl.divide(acc, l))
    return out


# ---- FP8 cache variant v2 ---------------------------------------------------------------------------------------------------------------------
# One page-head (16 tokens x 128 dims = 2 KB of E4M3) per SBUF partition row, 128 pages (2048 keys) per DMA: row = (page * 2 + kv) * kvh + h.
# Slice t (token t of every page) of the [128 pages, 16 * 128] tile is [128 pages, 128 d]; K is transposed straight from the fp8 slice (the PE
# produces exact f32), probabilities are weighted by the V scale, quantised to E4M3 per row and group and multiplied with the fp8 V slices (fp8 x fp8 PV,
# as the GPU runtime does), so neither K nor V is ever converted.  Column c = t * 128 + j of a group's score tile is key g * 2048 + j * 16 + t.
GROUP8_PAGES = 128
GROUP8_KEYS = GROUP8_PAGES * 16


def native_key_positions8(groups):
    """Key position of every column [groups, 2048] of the FP8 variant's tiles."""
    j = np.arange(GROUP8_PAGES)[None, :]
    t = np.arange(16)[:, None]
    cols = (j * 16 + t).reshape(2048)                                   # column c = t * 128 + j
    return np.arange(groups)[:, None] * GROUP8_KEYS + cols[None, :]


def row_indices8(table_pages, kv, groups):
    """int32 [KVH, 128, groups]: cache row of page j of group g for kv head h (table_pages [groups * 128])."""
    p = np.asarray(table_pages).reshape(groups, GROUP8_PAGES)
    out = np.zeros((KVH, 128, groups), np.int32)
    for h in range(KVH):
        out[h] = (((p * 2 + kv) * KVH + h).T).astype(np.int32)
    return out


@nki.jit
def decode_attention_fp8(qt, cache, idxk, idxv, bias, mask, ksc, vsc):
    """R requests, one query token each, FP8 cache. qt [R, KVH, D, REP] bf16 (scaled), cache [blocks, 2048] E4M3 (uint8 bits or float8),
    idxk / idxv [R, KVH, 128, G] int32, bias / ksc / vsc [R, G, 16, 2048] f32 (native column order, rows repeated per head), mask [16, KVH] f32
    -> o [R * 16, D] f32."""
    R = qt.shape[0]
    G = idxk.shape[3]
    out = nl.ndarray((R * ROWS, D), dtype=nl.float32, buffer=nl.shared_hbm)
    i1, _ = nl.mgrid[:128, :1]
    _, jf = nl.mgrid[:128, :2048]
    rowmask = nl.load(mask)
    for r in nl.affine_range(R):
        qp = nl.zeros((128, KVH * ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
        tabk = nl.ndarray((128, KVH * G), dtype=nl.int32, buffer=nl.sbuf)
        tabv = nl.ndarray((128, KVH * G), dtype=nl.int32, buffer=nl.sbuf)
        for h in nl.affine_range(KVH):
            qp[:, nl.ds(h * ROWS + h * REP, REP)] = nl.load(qt[r, h])
            tabk[:, nl.ds(h * G, G)] = nl.load(idxk[r, h])
            tabv[:, nl.ds(h * G, G)] = nl.load(idxv[r, h])
        m = nl.full((ROWS, 1), -30000.0, dtype=nl.float32, buffer=nl.sbuf)
        l = nl.zeros((ROWS, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.zeros((ROWS, D), dtype=nl.float32, buffer=nl.sbuf)
        for g in nl.sequential_range(G):
            kraw = nl.ndarray((128, KVH * 2048), dtype=cache.dtype, buffer=nl.sbuf)
            vraw = nl.ndarray((128, KVH * 2048), dtype=cache.dtype, buffer=nl.sbuf)
            for h in nl.affine_range(KVH):
                nisa.dma_copy(src=cache[tabk[i1, h * G + g], jf], dst=kraw[:, nl.ds(h * 2048, 2048)])
                nisa.dma_copy(src=cache[tabv[i1, h * G + g], jf], dst=vraw[:, nl.ds(h * 2048, 2048)])
            if cache.dtype == nl.uint8:
                k8 = kraw.view(nl.float8_e4m3)
                v8 = vraw.view(nl.float8_e4m3)
            else:
                k8 = kraw
                v8 = vraw
            ksc_t = nl.load(ksc[r, g])
            vsc_t = nl.load(vsc[r, g])
            bias_t = nl.load(bias[r, g])
            s_sb = nl.ndarray((ROWS, 2048), dtype=nl.float32, buffer=nl.sbuf)
            for quarter in nl.affine_range(4):                                   # 4 token slices = 512 columns per score tile
                s_ps = nl.zeros((ROWS, 512), dtype=nl.float32, buffer=nl.psum)
                for h in nl.affine_range(KVH):
                    kt_ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                    for tt in nl.affine_range(4):
                        kt_ps[:, nl.ds(tt * 128, 128)] = nisa.nc_transpose(k8[:, nl.ds(h * 2048 + (quarter * 4 + tt) * 128, 128)])
                    kt_sb = nl.copy(kt_ps, dtype=nl.bfloat16)
                    s_ps += nl.matmul(qp[:, nl.ds(h * ROWS, ROWS)], kt_sb, transpose_x=True)
                s_sb[:, nl.ds(quarter * 512, 512)] = nl.add(nl.multiply(s_ps, ksc_t[:, nl.ds(quarter * 512, 512)]), bias_t[:, nl.ds(quarter * 512, 512)])
            mx = nisa.tensor_reduce(np.max, s_sb, axis=(1,), dtype=nl.float32, negate=False)
            m_new = nl.maximum(m, mx)
            alpha = nl.exp(nl.subtract(m, m_new))
            p = nl.ndarray((ROWS, 2048), dtype=nl.float32, buffer=nl.sbuf)
            psum_l = nl.ndarray((ROWS, 4), dtype=nl.float32, buffer=nl.sbuf)
            for quarter in nl.affine_range(4):
                p[:, nl.ds(quarter * 512, 512)] = nisa.activation_reduce(np.exp, s_sb[:, nl.ds(quarter * 512, 512)], bias=nl.multiply(m_new, -1.0), scale=1.0,
                                                                           reduce_op=nl.add, reduce_res=psum_l[:, nl.ds(quarter, 1)], dtype=nl.float32)
            l[...] = nl.add(nl.multiply(l, alpha), nl.sum(psum_l, axis=1, keepdims=True))
            pw = nl.multiply(p, vsc_t)                                           # V scale weights the probabilities (the sum above is unweighted)
            ps = nl.maximum(nl.multiply(nisa.tensor_reduce(np.max, pw, axis=(1,), dtype=nl.float32, negate=False), 1.0 / 224.0), 1e-30)   # [16, 1]
            p8 = nl.copy(nl.divide(pw, ps), dtype=nl.float8_e4m3)                # [16, 2048] E4M3
            pt8 = nl.ndarray((128, 16 * ROWS), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for c in nl.affine_range(16):
                pt_ps = nisa.nc_transpose(p8[:, nl.ds(c * 128, 128)])           # [128 pages, 16]
                pt8[:, nl.ds(c * ROWS, ROWS)] = nl.copy(pt_ps, dtype=nl.float8_e4m3)
            acc[...] = nl.multiply(acc, alpha)
            for h in nl.sequential_range(KVH):
                pv_ps = nl.zeros((ROWS, D), dtype=nl.float32, buffer=nl.psum)
                for t in nl.affine_range(16):
                    pv_ps += nl.matmul(pt8[:, nl.ds(t * ROWS, ROWS)], v8[:, nl.ds(h * 2048 + t * 128, 128)], transpose_x=True)
                acc[...] = nl.add(acc, nl.multiply(pv_ps, nl.multiply(rowmask[:, nl.ds(h, 1)], ps)))
            m[...] = m_new
        nl.store(out[nl.ds(r * ROWS, ROWS), :], nl.divide(acc, l))
    return out


@nki.jit
def decode_attention_fp8_2p(qt, cache, idxk, idxv, bias, mask, ksc, vsc):
    """Two-pass decode_attention_fp8 (same inputs, layouts and output). Pass 1 scores every group into one SBUF tile [16, G * 2048] with a
    per-group max (no state carried between groups, so the K gathers of later groups overlap the PE / vector work of earlier ones); one
    global max per row; pass 2 gathers V, exponentiates against the global max, quantises the weighted probabilities per group and keeps
    each group's PV in its own slice, summed at the end. No accumulator rescaling, no loop-carried m / l / acc."""
    R = qt.shape[0]
    G = idxk.shape[3]
    out = nl.ndarray((R * ROWS, D), dtype=nl.float32, buffer=nl.shared_hbm)
    i1, _ = nl.mgrid[:128, :1]
    _, jf = nl.mgrid[:128, :2048]
    rowmask = nl.load(mask)
    for r in nl.affine_range(R):
        qp = nl.zeros((128, KVH * ROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
        tabk = nl.ndarray((128, KVH * G), dtype=nl.int32, buffer=nl.sbuf)
        tabv = nl.ndarray((128, KVH * G), dtype=nl.int32, buffer=nl.sbuf)
        for h in nl.affine_range(KVH):
            qp[:, nl.ds(h * ROWS + h * REP, REP)] = nl.load(qt[r, h])
            tabk[:, nl.ds(h * G, G)] = nl.load(idxk[r, h])
            tabv[:, nl.ds(h * G, G)] = nl.load(idxv[r, h])
        s_all = nl.ndarray((ROWS, G * 2048), dtype=nl.float32, buffer=nl.sbuf)
        mx_all = nl.ndarray((ROWS, G), dtype=nl.float32, buffer=nl.sbuf)
        for g in nl.affine_range(G):                                             # pass 1: scores
            kraw = nl.ndarray((128, KVH * 2048), dtype=cache.dtype, buffer=nl.sbuf)
            for h in nl.affine_range(KVH):
                nisa.dma_copy(src=cache[tabk[i1, h * G + g], jf], dst=kraw[:, nl.ds(h * 2048, 2048)])
            if cache.dtype == nl.uint8:
                k8 = kraw.view(nl.float8_e4m3)
            else:
                k8 = kraw
            ksc_t = nl.load(ksc[r, g])
            bias_t = nl.load(bias[r, g])
            for quarter in nl.affine_range(4):
                s_ps = nl.zeros((ROWS, 512), dtype=nl.float32, buffer=nl.psum)
                for h in nl.affine_range(KVH):
                    kt_ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                    for tt in nl.affine_range(4):
                        kt_ps[:, nl.ds(tt * 128, 128)] = nisa.nc_transpose(k8[:, nl.ds(h * 2048 + (quarter * 4 + tt) * 128, 128)])
                    kt_sb = nl.copy(kt_ps, dtype=nl.bfloat16)
                    s_ps += nl.matmul(qp[:, nl.ds(h * ROWS, ROWS)], kt_sb, transpose_x=True)
                s_all[:, nl.ds(g * 2048 + quarter * 512, 512)] = nl.add(nl.multiply(s_ps, ksc_t[:, nl.ds(quarter * 512, 512)]),
                                                                         bias_t[:, nl.ds(quarter * 512, 512)])
            mx_all[:, nl.ds(g, 1)] = nisa.tensor_reduce(np.max, s_all[:, nl.ds(g * 2048, 2048)], axis=(1,), dtype=nl.float32, negate=False)
        negm = nl.multiply(nisa.tensor_reduce(np.max, mx_all, axis=(1,), dtype=nl.float32, negate=False), -1.0)      # [16, 1]
        l_all = nl.ndarray((ROWS, G * 4), dtype=nl.float32, buffer=nl.sbuf)
        acc_all = nl.ndarray((ROWS, G * D), dtype=nl.float32, buffer=nl.sbuf)
        for g in nl.affine_range(G):                                             # pass 2: probabilities and PV
            vraw = nl.ndarray((128, KVH * 2048), dtype=cache.dtype, buffer=nl.sbuf)
            for h in nl.affine_range(KVH):
                nisa.dma_copy(src=cache[tabv[i1, h * G + g], jf], dst=vraw[:, nl.ds(h * 2048, 2048)])
            if cache.dtype == nl.uint8:
                v8 = vraw.view(nl.float8_e4m3)
            else:
                v8 = vraw
            vsc_t = nl.load(vsc[r, g])
            p = nl.ndarray((ROWS, 2048), dtype=nl.float32, buffer=nl.sbuf)
            for quarter in nl.affine_range(4):
                p[:, nl.ds(quarter * 512, 512)] = nisa.activation_reduce(np.exp, s_all[:, nl.ds(g * 2048 + quarter * 512, 512)], bias=negm, scale=1.0,
                                                                           reduce_op=nl.add, reduce_res=l_all[:, nl.ds(g * 4 + quarter, 1)], dtype=nl.float32)
            pw = nl.multiply(p, vsc_t)
            ps = nl.maximum(nl.multiply(nisa.tensor_reduce(np.max, pw, axis=(1,), dtype=nl.float32, negate=False), 1.0 / 224.0), 1e-30)
            p8 = nl.copy(nl.divide(pw, ps), dtype=nl.float8_e4m3)
            pt8 = nl.ndarray((128, 16 * ROWS), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for c in nl.affine_range(16):
                pt8[:, nl.ds(c * ROWS, ROWS)] = nl.copy(nisa.nc_transpose(p8[:, nl.ds(c * 128, 128)]), dtype=nl.float8_e4m3)
            ag = nl.zeros((ROWS, D), dtype=nl.float32, buffer=nl.sbuf)
            for h in nl.sequential_range(KVH):
                pv_ps = nl.zeros((ROWS, D), dtype=nl.float32, buffer=nl.psum)
                for t in nl.affine_range(16):
                    pv_ps += nl.matmul(pt8[:, nl.ds(t * ROWS, ROWS)], v8[:, nl.ds(h * 2048 + t * 128, 128)], transpose_x=True)
                ag[...] = nl.add(ag, nl.multiply(pv_ps, rowmask[:, nl.ds(h, 1)]))
            acc_all[:, nl.ds(g * D, D)] = nl.multiply(ag, ps)
        acc = nl.zeros((ROWS, D), dtype=nl.float32, buffer=nl.sbuf)
        for g in nl.sequential_range(G):
            acc[...] = nl.add(acc, acc_all[:, nl.ds(g * D, D)])
        l = nl.sum(l_all, axis=1, keepdims=True)
        nl.store(out[nl.ds(r * ROWS, ROWS), :], nl.divide(acc, l))
    return out

