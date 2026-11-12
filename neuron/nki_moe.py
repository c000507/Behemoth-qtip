"""EXPERIMENT: the routed-expert part of a Qwen3.5 decode MoE block (one TP rank) as one NKI kernel, weights as the MOVING operand.

Semantics = qwen35_graphs.moe_routed (fp8 weights): hgu = bf16((n8 @ W_gu^T) * a * egu_s); act = bf16(silu(gate) * up) * comb * ed_c;
a2 = amax(|act|) / 224 per row; routed = (fp8(act / a2) @ W_d) * a2.  The selected experts' slices are read with indirect DMAs
(one row index per SBUF partition), never materialised in HBM.

Layouts
  egu_t [E * 128, 2048] fp8: egu_t[e * 128 + p, kt * 64 + o] = egu[e, o, kt * 128 + p]  (relayout_egu): the contraction axis H = 4096 on
        the partitions, 2 KB partition rows, one [128, 2048] DMA per slot. Per K-chunk kt the activation tile [128, T] is stationary and
        every slot contributes one 64-column moving matmul into its slice of a PSUM accumulator [T, 8 slots x 64].
  ed    [E * 32, 4096] fp8 = the stored ed [E, 32, 4096] flattened (already contraction-major, 4 KB rows): a down tile = 4 slots x 32
        rows on the 128 partitions (one indirect DMA [128, 4096]), stationary = the 4 slots' activations [128, T], 8 x 512 moving columns.
Kernel inputs (the caller builds the small ones in the graph, see graph_inputs):
  n8T [4096, T] fp8 (n / a transposed), ga [T, S * 64] f32 = a * egu_s[idx], cw [T, S * 32] f32 = comb[t, s] * ed_c[idx[s], i],
  idxg [128, S] int32 = idx[s] * 128 + p, idxd [128, ND] int32 = idx[4 d + q // 32] * 32 + q % 32 (ND = ceil(S / 4); padded slots point
  at any expert, their activation is zero) -> routed [T, 4096] f32.
"""
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

H, KT, O2, EI = 4096, 32, 64, 32     # hidden, K chunks of 128, gate|up rows per expert and rank, down rows per expert and rank
GS = 8                               # slots per PSUM accumulator (8 x 64 f32 columns = one bank)
NC = 512


def _f8(t, dtype):
    return t.view(nl.float8_e4m3) if dtype == nl.uint8 else t


def relayout_egu(egu):
    """numpy [E, 64, 4096] (any 1-byte dtype) -> [E * 128, 2048]."""
    E = egu.shape[0]
    return np.ascontiguousarray(egu.reshape(E, O2, KT, 128).transpose(0, 3, 2, 1)).reshape(E * 128, KT * O2)


def make_moe(T, S, gu='slot', act_form='slot', down='tile4', static=False, parts=('gu', 'down'), pre='', pp=0):
    """gu: 'slot' (one 64-column matmul per slot and K chunk) | 'grid' (one matmul per 8-slot group and K chunk over a strided moving view)
    act_form: 'slot' (SiLU pairing per slot) | 'grid' (3-D tiles)
    down: 'tile4' (4 slots per [128, 4096] tile) | 'half' (two passes of [128, 2048] half rows: 4 PSUM banks at a time)
    pre: 'first' | 'after': all down tiles are read into one SBUF buffer [128, ND * 4096] before / after the gate|up groups (the down
         DMAs do not depend on the activations, so they can overlap the gate|up matmuls); '' reads each tile inside the down loop.
    pp: N > 0: the gate|up groups rotate over N explicitly allocated weight buffers (group g + 1 can be read while group g is in the PE
        even if the allocator would otherwise give every group's buffer the same address); 0: one buffer per group, allocator's choice.
    static: the DMAs read fixed rows (no indirection): the DMA ceiling of the geometry (wrong result unless idx = arange).
    parts: timing probes ('gu' only skips the down projection, 'down' only feeds zeros-derived activations)."""
    ND = -(-S // 4)
    groups = [(s0, min(GS, S - s0)) for s0 in range(0, S, GS)]

    @nki.jit
    def moe(n8T, ga, cw, idxg, idxd, egu_t, ed):
        y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
        hx = nl.ndarray((128, KT * T), dtype=n8T.dtype, buffer=nl.sbuf)
        for kt in nl.affine_range(KT):
            hx[:, nl.ds(kt * T, T)] = nl.load(n8T[nl.ds(kt * 128, 128), :])
        tabg = nl.load(idxg)                                                          # [128, S]
        tabd = nl.load(idxd)                                                          # [128, ND]
        gsa = nl.load(ga)                                                             # [T, S * 64]
        cwt = nl.load(cw)                                                             # [T, S * 32]
        i1, _ = nl.mgrid[:128, :1]
        _, jg = nl.mgrid[:128, :KT * O2]
        _, jd = nl.mgrid[:128, :H]
        act = nl.zeros((T, ND * 128), dtype=nl.float32, buffer=nl.sbuf)
        if pre:
            wdall = nl.ndarray((128, ND * H), dtype=ed.dtype, buffer=nl.sbuf)

        def load_down():
            for d in nl.affine_range(ND):
                if static:
                    wdall[:, nl.ds(d * H, H)] = nl.load(ed[nl.ds(d * 128, 128), :])
                else:
                    nisa.dma_copy(src=ed[tabd[i1, d], jd], dst=wdall[:, nl.ds(d * H, H)])
        if pre == 'first':
            load_down()
        if 'gu' in parts:
            bufs = [nl.ndarray((128, GS * KT * O2), dtype=egu_t.dtype, buffer=nl.sbuf) for _ in range(pp)]
            for s0, ns in groups:                                                     # Python-level: 8-slot groups
                wraw = bufs[(s0 // GS) % pp] if pp else nl.ndarray((128, ns * KT * O2), dtype=egu_t.dtype, buffer=nl.sbuf)
                for j in nl.affine_range(ns):
                    if static:
                        wraw[:, nl.ds(j * KT * O2, KT * O2)] = nl.load(egu_t[nl.ds((s0 + j) * 128, 128), :])
                    else:
                        nisa.dma_copy(src=egu_t[tabg[i1, s0 + j], jg], dst=wraw[:, nl.ds(j * KT * O2, KT * O2)])
                if gu == 'grid':                                                      # 3-D tiles: [T, slot, 64]
                    it, iq, i6 = nl.mgrid[:T, :ns, :O2]
                    t2, q2, ii = nl.mgrid[:T, :ns, :EI]
                    acc3 = nl.zeros((T, ns, O2), dtype=nl.float32, buffer=nl.psum)
                    for kt in nl.sequential_range(KT):
                        xk = _f8(hx[:, nl.ds(kt * T, T)], n8T.dtype)
                        kp, kj, ko = nl.mgrid[:128, :ns, :O2]
                        acc3 += nisa.nc_matmul(xk, wraw[kp, kj * (KT * O2) + kt * O2 + ko])
                    hb3 = nl.multiply(acc3, gsa[it, s0 * O2 + iq * O2 + i6], dtype=nl.bfloat16)
                    sg3 = nl.silu(hb3, dtype=nl.float32)
                    pb3 = nl.multiply(sg3[t2, q2, ii], hb3[t2, q2, EI + ii], dtype=nl.bfloat16)
                    act[t2, s0 * EI + q2 * EI + ii] = nl.multiply(pb3, cwt[t2, s0 * EI + q2 * EI + ii], dtype=nl.float32)
                    continue
                acc = nl.zeros((T, ns * O2), dtype=nl.float32, buffer=nl.psum)
                for kt in nl.sequential_range(KT):
                    xk = _f8(hx[:, nl.ds(kt * T, T)], n8T.dtype)
                    for j in nl.static_range(ns):
                        acc[:, nl.ds(j * O2, O2)] += nl.matmul(xk, _f8(wraw[:, nl.ds(j * KT * O2 + kt * O2, O2)], egu_t.dtype), transpose_x=True)
                hb = nl.multiply(acc, gsa[:, nl.ds(s0 * O2, ns * O2)], dtype=nl.bfloat16)                 # [T, ns * 64] rounded to bf16
                sg = nl.silu(hb, dtype=nl.float32)
                if act_form == 'grid':
                    it, iq, ii = nl.mgrid[:T, :ns, :EI]
                    pb = nl.multiply(sg[it, iq * O2 + ii], hb[it, iq * O2 + EI + ii], dtype=nl.bfloat16)
                    act[it, s0 * EI + iq * EI + ii] = nl.multiply(pb, cwt[it, s0 * EI + iq * EI + ii], dtype=nl.float32)
                else:
                    pb = nl.ndarray((T, ns * EI), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for j in nl.affine_range(ns):
                        pb[:, nl.ds(j * EI, EI)] = nl.multiply(sg[:, nl.ds(j * O2, EI)], hb[:, nl.ds(j * O2 + EI, EI)], dtype=nl.bfloat16)
                    act[:, nl.ds(s0 * EI, ns * EI)] = nl.multiply(pb, cwt[:, nl.ds(s0 * EI, ns * EI)], dtype=nl.float32)
        else:
            act[:, 0:S * EI] = nl.copy(cwt)
        if pre == 'after':
            load_down()
        if 'down' not in parts:
            for jj in nl.affine_range(H // NC):
                nl.store(y[:, nl.ds(jj * NC, NC)], nl.copy(act[:, 0:NC]) if S * EI >= NC else nl.zeros((T, NC), dtype=nl.float32, buffer=nl.sbuf))
            return y
        aa = nl.maximum(nl.multiply(nisa.tensor_reduce(np.max, nl.abs(act), axis=(1,), dtype=nl.float32), 1.0 / 224.0), 1e-20)   # [T, 1]
        ra = nl.reciprocal(aa)
        a8 = nl.multiply(act, ra, dtype=nl.float8_e4m3)
        ax = nl.ndarray((128, ND * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for d in nl.affine_range(ND):
            ax[:, nl.ds(d * T, T)] = nl.copy(nisa.nc_transpose(a8[:, nl.ds(d * 128, 128)]), dtype=nl.float8_e4m3)
        if pre:
            accd = [nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum) for _ in nl.static_range(H // NC)]
            for d in nl.sequential_range(ND):
                for jj in nl.static_range(H // NC):
                    accd[jj] += nl.matmul(ax[:, nl.ds(d * T, T)], _f8(wdall[:, nl.ds(d * H + jj * NC, NC)], ed.dtype), transpose_x=True)
            for jj in nl.static_range(H // NC):
                nl.store(y[:, nl.ds(jj * NC, NC)], nl.multiply(accd[jj], aa))
        elif down == 'tile4':
            accd = [nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum) for _ in nl.static_range(H // NC)]
            for d in nl.sequential_range(ND):
                wd = nl.ndarray((128, H), dtype=ed.dtype, buffer=nl.sbuf)
                if static:
                    wd[:, :] = nl.load(ed[nl.ds(d * 128, 128), :])
                else:
                    nisa.dma_copy(src=ed[tabd[i1, d], jd], dst=wd)
                for jj in nl.static_range(H // NC):
                    accd[jj] += nl.matmul(ax[:, nl.ds(d * T, T)], _f8(wd[:, nl.ds(jj * NC, NC)], ed.dtype), transpose_x=True)
            for jj in nl.static_range(H // NC):
                nl.store(y[:, nl.ds(jj * NC, NC)], nl.multiply(accd[jj], aa))
        else:
            HH = H // 2
            _, jh = nl.mgrid[:128, :HH]
            for half in nl.affine_range(2):
                acch = [nl.zeros((T, NC), dtype=nl.float32, buffer=nl.psum) for _ in nl.static_range(HH // NC)]
                for d in nl.sequential_range(ND):
                    wh = nl.ndarray((128, HH), dtype=ed.dtype, buffer=nl.sbuf)
                    if static:
                        wh[:, :] = nl.load(ed[nl.ds(d * 128, 128), nl.ds(half * HH, HH)])
                    else:
                        nisa.dma_copy(src=ed[tabd[i1, d], half * HH + jh], dst=wh)
                    for jj in nl.static_range(HH // NC):
                        acch[jj] += nl.matmul(ax[:, nl.ds(d * T, T)], _f8(wh[:, nl.ds(jj * NC, NC)], ed.dtype), transpose_x=True)
                for jj in nl.static_range(HH // NC):
                    nl.store(y[:, nl.ds(half * HH + jj * NC, NC)], nl.multiply(acch[jj], aa))
        return y
    return moe


def graph_inputs(n, idx, comb, egu_s, ed_c):
    """torch (in the traced graph): n [T, H] f32 (bf16 values), idx [S] int64, comb [T, S] f32, egu_s [E, 64], ed_c [E, 32]
    -> (n8T, ga, cw, idxg, idxd) for the kernel."""
    import torch
    T, S, dev = n.shape[0], idx.shape[0], n.device
    ND = -(-S // 4)
    a = torch.amax(torch.abs(n.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    n8T = (n.float() / a).to(torch.float8_e4m3fn).T.contiguous()
    ga = a * torch.index_select(egu_s, 0, idx).reshape(1, S * O2)
    cw = (comb.unsqueeze(-1) * torch.index_select(ed_c, 0, idx).reshape(1, S, EI)).reshape(T, S * EI)
    idxg = (idx.reshape(1, S) * 128 + torch.arange(128, device=dev).reshape(128, 1)).to(torch.int32)
    ip = torch.cat((idx, idx[:4 * ND - S]), dim=0) if 4 * ND > S else idx
    idxd = (ip.reshape(ND, 4, 1) * EI + torch.arange(EI, device=dev).reshape(1, 1, EI)).reshape(ND, 128).T.contiguous().to(torch.int32)
    return n8T, ga, cw, idxg, idxd
