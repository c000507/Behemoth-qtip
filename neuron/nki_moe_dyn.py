"""EXPERIMENT: decode MoE expert part over the DISTINCT selected experts (new `nki` 0.5.0 package, parser frontend).

One TP rank of Qwen3.5-397B: 512 experts, gate|up rows egu [E, 64, 4096] fp8 (EIl = 32 per rank), down ed [E, 32, 4096] fp8,
row scales egu_s [E, 64], ed_c [E, 32]. The graph (torch, before the kernel) builds the sorted list of the distinct experts
selected by the T rows (padded to K_max = 8 G slots with unused experts), the combine weights comb[t, s] of slot s for row t (0
when row t did not select it) and the number of 8-slot groups that hold real experts, ng = ceil(n_distinct / 8). The kernel
streams ONLY those ng groups (a device-side loop, nl.dynamic_range over a register loaded from the count tensor): every
distinct expert's slice is read once and applied to all T rows.

Semantics = qwen35_graphs.moe_routed (union form): hgu = bf16((n8 @ W_gu^T) * a * egu_s); act = bf16(silu(gate) * up) * comb
* ed_c; a2 = max(amax_row(|act|) / 224, 1e-20); routed = (fp8(act / a2) @ W_d) * a2.

  phase 1 (loop g < ng): gate|up of the group's 8 slots (activation stationary, weights moving), SiLU, act_g [T, 256] f32 ->
          HBM scratch, running row amax;
  phase 2 (loop g < ng): fp8(act_g / a2), transposed to [128, 2 T] (2 down tiles of 4 slots x 32 rows) -> HBM scratch;
  phase 3 (loop g < ng): the group's 2 down tiles [128, 4096] (4 KB rows) x 8 PSUM banks of [T, 512], accumulated over all
          groups; y = acc * a2.

Layouts: egu_t [E * 128, 2048]: egu_t[e * 128 + p, kt * 64 + o] = egu[e, o, kt * 128 + p] (nki_moe.relayout_egu);
ed [E * 32, 4096] as stored. Kernel inputs (see graph_inputs): xT [128, 32 T] fp8 (chunk kt = columns kt T .. kt T + T),
ga [G, T, 512] f32, cw [G, T, 256] f32, gidx [G, 128, 8] int32, didx [G, 128, 2] int32, cnt [1, 1] int32 -> y [T, 4096] f32.
"""
import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

H, KT, O2, EI, GS = 4096, 32, 64, 32, 8


@nki.jit
def moe_dyn(xT, ga, cw, gidx, didx, cnt, egu_t, ed, dbg=0, gu=1):
    T = ga.shape[1]
    G = ga.shape[0]
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    wdump = nl.ndarray((128, GS * KT * O2), dtype=egu_t.dtype, buffer=nl.shared_hbm)
    hdump = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.shared_hbm)

    xs = nl.ndarray((128, KT * T), dtype=xT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=xT)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb)
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)

    # ---------------- phase 1: gate | up, SiLU, combine -> act (HBM), running row amax
    for g in nl.dynamic_range(ng):
        gi = nl.ndarray((128, GS), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gi, src=gidx.ap(pattern=[[GS, 128], [1, GS]], offset=0, scalar_offset=g, indirect_dim=0))
        w = nl.ndarray((128, GS * KT * O2), dtype=egu_t.dtype, buffer=nl.sbuf)
        for j in nl.static_range(GS):
            nisa.dma_copy(dst=w[:, j * KT * O2:(j + 1) * KT * O2],
                          src=egu_t.ap(pattern=[[KT * O2, 128], [1, KT * O2]], offset=0, vector_offset=gi[:, j:j + 1], indirect_dim=0))
        gs = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gs, src=ga.ap(pattern=[[GS * O2, T], [1, GS * O2]], offset=0, scalar_offset=g, indirect_dim=0))
        cs = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs, src=cw.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=0, scalar_offset=g, indirect_dim=0))
        acc = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.psum)
        if gu == 1:
            wv = w.reshape((128, GS, KT * O2))
            for kt in nl.static_range(KT):
                nisa.nc_matmul(dst=acc.reshape((T, GS, O2)), stationary=xs[:, kt * T:(kt + 1) * T],
                               moving=wv[:, :, kt * O2:(kt + 1) * O2], accumulate=(kt > 0))
        else:
            for j in nl.static_range(GS):
                for kt in nl.static_range(KT):
                    nisa.nc_matmul(dst=acc[:, j * O2:(j + 1) * O2], stationary=xs[:, kt * T:(kt + 1) * T],
                                   moving=w[:, j * KT * O2 + kt * O2:j * KT * O2 + (kt + 1) * O2], accumulate=(kt > 0))
        hb = nl.ndarray((T, GS * O2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hb, data1=acc, data2=gs, op=nl.multiply)
        if dbg == 1:
            nisa.dma_copy(dst=wdump, src=w)
            hf = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hf, src=acc)
            nisa.dma_copy(dst=hdump, src=hf)
        sg = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.silu, data=hb)
        pb = nl.ndarray((T, GS * EI), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in nl.static_range(GS):
            nisa.tensor_tensor(dst=pb[:, j * EI:(j + 1) * EI], data1=sg[:, j * O2:j * O2 + EI], data2=hb[:, j * O2 + EI:(j + 1) * O2], op=nl.multiply)
        act = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=act, data1=pb, data2=cs, op=nl.multiply)
        mg = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        ab = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ab, op=nl.abs, data=act)
        nisa.tensor_reduce(dst=mg, op=nl.maximum, data=ab, axis=1)
        nisa.tensor_tensor(dst=amax, data1=amax, data2=mg, op=nl.maximum)
        nisa.dma_copy(dst=acth.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=0, scalar_offset=g, indirect_dim=0), src=act)

    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)

    # ---------------- phase 2: quantise with the row scale, transpose to the down layout -> HBM
    for g in nl.dynamic_range(ng):
        av = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=av, src=acth.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=0, scalar_offset=g, indirect_dim=0))
        a8 = nl.ndarray((T, GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a8, data=av, op0=nl.multiply, operand0=ra)
        ax = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for i in nl.static_range(2):
            tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=a8[:, i * 128:(i + 1) * 128])
            nisa.tensor_copy(dst=ax[:, i * T:(i + 1) * T], src=tp)
        nisa.dma_copy(dst=axh.ap(pattern=[[2 * T, 128], [1, 2 * T]], offset=0, scalar_offset=g, indirect_dim=0), src=ax)

    # ---------------- phase 3: down projection, 8 PSUM banks accumulated over the groups
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    for g in nl.dynamic_range(ng):
        di = nl.ndarray((128, 2), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=di, src=didx.ap(pattern=[[2, 128], [1, 2]], offset=0, scalar_offset=g, indirect_dim=0))
        axs = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.dma_copy(dst=axs, src=axh.ap(pattern=[[2 * T, 128], [1, 2 * T]], offset=0, scalar_offset=g, indirect_dim=0))
        wd = nl.ndarray((128, 2 * H), dtype=ed.dtype, buffer=nl.sbuf)
        for i in nl.static_range(2):
            nisa.dma_copy(dst=wd[:, i * H:(i + 1) * H], src=ed.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=di[:, i:i + 1], indirect_dim=0))
        accd = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.psum)
        for jj in nl.static_range(H // 512):
            for i in nl.static_range(2):
                nisa.nc_matmul(dst=accd[:, jj * 512:(jj + 1) * 512], stationary=axs[:, i * T:(i + 1) * T],
                               moving=wd[:, i * H + jj * 512:i * H + (jj + 1) * 512], accumulate=(i > 0))
        nisa.tensor_tensor(dst=ys, data1=ys, data2=accd, op=nl.add)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    if dbg == 1:
        return y, acth, axh, wdump, hdump
    return y


def _gu_load(gidx, ga, cw, egu_t, g, off, gi, w, gs, cs):
    """the group's gate|up weights, scales and combine weights (group index = register g + static off)."""
    T = ga.shape[1]
    nisa.dma_copy(dst=gi, src=gidx.ap(pattern=[[GS, 128], [1, GS]], offset=off * 128 * GS, scalar_offset=g, indirect_dim=0))
    for j in nl.static_range(GS):
        nisa.dma_copy(dst=w[:, j * KT * O2:(j + 1) * KT * O2],
                      src=egu_t.ap(pattern=[[KT * O2, 128], [1, KT * O2]], offset=0, vector_offset=gi[:, j:j + 1], indirect_dim=0))
    nisa.dma_copy(dst=gs, src=ga.ap(pattern=[[GS * O2, T], [1, GS * O2]], offset=off * T * GS * O2, scalar_offset=g, indirect_dim=0))
    nisa.dma_copy(dst=cs, src=cw.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=off * T * GS * EI, scalar_offset=g, indirect_dim=0))


def _gu_compute(xs, w, gs, cs, amax, acth, g, off):
    T = gs.shape[0]
    acc = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.psum)
    wv = w.reshape((128, GS, KT * O2))
    for kt in nl.static_range(KT):
        nisa.nc_matmul(dst=acc.reshape((T, GS, O2)), stationary=xs[:, kt * T:(kt + 1) * T],
                       moving=wv[:, :, kt * O2:(kt + 1) * O2], accumulate=(kt > 0))
    hb = nl.ndarray((T, GS * O2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=hb, data1=acc, data2=gs, op=nl.multiply)
    sg = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=hb)
    pb = nl.ndarray((T, GS * EI), dtype=nl.bfloat16, buffer=nl.sbuf)
    if fast >= 1:   # gate x up of the GS slots in ONE vector instruction (strided views) instead of GS
        nisa.tensor_tensor(dst=pb.reshape((T, GS, 1, EI)), data1=sg.reshape((T, GS, 2, EI))[:, :, 0:1, :],
                           data2=hb.reshape((T, GS, 2, EI))[:, :, 1:2, :], op=nl.multiply)
    else:
        for j in nl.static_range(GS):
            nisa.tensor_tensor(dst=pb[:, j * EI:(j + 1) * EI], data1=sg[:, j * O2:j * O2 + EI], data2=hb[:, j * O2 + EI:(j + 1) * O2], op=nl.multiply)
    act = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=pb, data2=cs, op=nl.multiply)
    ab = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=act)
    mg = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mg, op=nl.maximum, data=ab, axis=1)
    nisa.tensor_tensor(dst=amax, data1=amax, data2=mg, op=nl.maximum)
    nisa.dma_copy(dst=acth.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=off * T * GS * EI, scalar_offset=g, indirect_dim=0), src=act)


def _d_load(didx, axh, ed, g, off, di, axs, wd):
    T2 = axs.shape[1]
    nisa.dma_copy(dst=di, src=didx.ap(pattern=[[2, 128], [1, 2]], offset=off * 256, scalar_offset=g, indirect_dim=0))
    nisa.dma_copy(dst=axs, src=axh.ap(pattern=[[T2, 128], [1, T2]], offset=off * 128 * T2, scalar_offset=g, indirect_dim=0))
    for i in nl.static_range(2):
        nisa.dma_copy(dst=wd[:, i * H:(i + 1) * H], src=ed.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=di[:, i:i + 1], indirect_dim=0))


def _d_compute(axs, wd, ys):
    T = ys.shape[0]
    accd = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.psum)
    for jj in nl.static_range(H // 512):
        for i in nl.static_range(2):
            nisa.nc_matmul(dst=accd[:, jj * 512:(jj + 1) * 512], stationary=axs[:, i * T:(i + 1) * T],
                           moving=wd[:, i * H + jj * 512:i * H + (jj + 1) * 512], accumulate=(i > 0))
    nisa.tensor_tensor(dst=ys, data1=ys, data2=accd, op=nl.add)


@nki.jit
def moe_dyn2(xT, ga, cw, gidx, didx, cnt, egu_t, ed):
    """moe_dyn with phases 1 and 3 software-pipelined over pairs of groups: two explicit buffer sets (A, B) live across the
    loop; group 2k + 1 is read into B while A computes group 2k, group 2k + 2 into A while B computes. cnt = [ng, 2 ceil(ng / 2)];
    the group arrays carry 2 spare groups (combine weight 0) for the read-ahead past the end."""
    T = ga.shape[1]
    G = ga.shape[0]
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    xs = nl.ndarray((128, KT * T), dtype=xT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=xT)
    c_sb = nl.ndarray((1, 2), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb[0:1, 0:1])
    n2 = nisa.register_alloc()
    nisa.register_load(n2, c_sb[0:1, 1:2])
    zero = nisa.register_alloc(0)
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)

    giA = nl.ndarray((128, GS), dtype=nl.int32, buffer=nl.sbuf)
    giB = nl.ndarray((128, GS), dtype=nl.int32, buffer=nl.sbuf)
    wA = nl.ndarray((128, GS * KT * O2), dtype=egu_t.dtype, buffer=nl.sbuf)
    wB = nl.ndarray((128, GS * KT * O2), dtype=egu_t.dtype, buffer=nl.sbuf)
    gsA = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
    gsB = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
    csA = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    csB = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    _gu_load(gidx, ga, cw, egu_t, zero, 0, giA, wA, gsA, csA)
    for g in nl.dynamic_range(0, n2, 2):
        _gu_load(gidx, ga, cw, egu_t, g, 1, giB, wB, gsB, csB)
        _gu_compute(xs, wA, gsA, csA, amax, acth, g, 0)
        _gu_load(gidx, ga, cw, egu_t, g, 2, giA, wA, gsA, csA)
        _gu_compute(xs, wB, gsB, csB, amax, acth, g, 1)

    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    for g in nl.dynamic_range(n2):
        av = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=av, src=acth.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=0, scalar_offset=g, indirect_dim=0))
        a8 = nl.ndarray((T, GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a8, data=av, op0=nl.multiply, operand0=ra)
        ax = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for i in nl.static_range(2):
            tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=a8[:, i * 128:(i + 1) * 128])
            nisa.tensor_copy(dst=ax[:, i * T:(i + 1) * T], src=tp)
        nisa.dma_copy(dst=axh.ap(pattern=[[2 * T, 128], [1, 2 * T]], offset=0, scalar_offset=g, indirect_dim=0), src=ax)

    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    diA = nl.ndarray((128, 2), dtype=nl.int32, buffer=nl.sbuf)
    diB = nl.ndarray((128, 2), dtype=nl.int32, buffer=nl.sbuf)
    axA = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    axB = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    wdA = nl.ndarray((128, 2 * H), dtype=ed.dtype, buffer=nl.sbuf)
    wdB = nl.ndarray((128, 2 * H), dtype=ed.dtype, buffer=nl.sbuf)
    _d_load(didx, axh, ed, zero, 0, diA, axA, wdA)
    for g in nl.dynamic_range(0, n2, 2):
        _d_load(didx, axh, ed, g, 1, diB, axB, wdB)
        _d_compute(axA, wdA, ys)
        _d_load(didx, axh, ed, g, 2, diA, axA, wdA)
        _d_compute(axB, wdB, ys)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


def _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, U):
    """phase 1 for U consecutive groups starting at register g (all loads issued first: the DMA of group u + 1 overlaps
    the matmuls of group u inside the iteration)."""
    T = ga.shape[1]
    ws = []
    for u in nl.static_range(U):
        gi = nl.ndarray((128, GS), dtype=nl.int32, buffer=nl.sbuf)
        w = nl.ndarray((128, GS * KT * O2), dtype=egu_t.dtype, buffer=nl.sbuf)
        gs = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
        cs = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        _gu_load(gidx, ga, cw, egu_t, g, u, gi, w, gs, cs)
        ws.append((w, gs, cs))
    for u in nl.static_range(U):
        _gu_compute(xs, ws[u][0], ws[u][1], ws[u][2], amax, acth, g, u)


def _ph2(acth, axh, ra, g, U):
    T = ra.shape[0]
    for u in nl.static_range(U):
        av = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=av, src=acth.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=u * T * GS * EI, scalar_offset=g, indirect_dim=0))
        a8 = nl.ndarray((T, GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a8, data=av, op0=nl.multiply, operand0=ra)
        ax = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        for i in nl.static_range(2):
            tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=a8[:, i * 128:(i + 1) * 128])
            nisa.tensor_copy(dst=ax[:, i * T:(i + 1) * T], src=tp)
        nisa.dma_copy(dst=axh.ap(pattern=[[2 * T, 128], [1, 2 * T]], offset=u * 128 * 2 * T, scalar_offset=g, indirect_dim=0), src=ax)


def _ph3(didx, axh, ed, ys, g, U, tmaj=0, st=0, g0=0, axall=None, nreal=1000000, first=0, yd=0, a2=None):
    """phase 3 for U groups: 2 U down tiles accumulated in the 8 PSUM banks, one SBUF accumulation per iteration.
    tmaj: didx stored tile-major [G, 2, 128] (moe_dyn9's slot table) instead of [G, 128, 2]."""
    T = ys.shape[0]
    tiles = []
    for u in nl.static_range(U):
        di = nl.ndarray((128, 2), dtype=nl.int32, buffer=nl.sbuf)
        axs = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        wd = nl.ndarray((128, 2 * H), dtype=ed.dtype, buffer=nl.sbuf)
        if tmaj == 1:
            if st == 2:
                nisa.dma_copy(dst=di, src=didx.ap(pattern=[[1, 128], [128, 2]], offset=(g0 + u) * 256))
                axs = axall[:, g0 + u, :]
            elif st == 1:
                nisa.dma_copy(dst=di, src=didx.ap(pattern=[[1, 128], [128, 2]], offset=(g0 + u) * 256))
                nisa.dma_copy(dst=axs, src=axh.ap(pattern=[[2 * T, 128], [1, 2 * T]], offset=(g0 + u) * 128 * 2 * T))
            else:
                nisa.dma_copy(dst=di, src=didx.ap(pattern=[[1, 128], [128, 2]], offset=u * 256, scalar_offset=g, indirect_dim=0))
                nisa.dma_copy(dst=axs, src=axh.ap(pattern=[[2 * T, 128], [1, 2 * T]], offset=u * 128 * 2 * T, scalar_offset=g, indirect_dim=0))
            for i in nl.static_range(2):
                if st == 0 or ((g0 + u) * 2 + i) * 4 < nreal:    # a tile of 4 padding slots is neither read nor multiplied
                    nisa.dma_copy(dst=wd[:, i * H:(i + 1) * H], src=ed.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=di[:, i:i + 1], indirect_dim=0))
        else:
            _d_load(didx, axh, ed, g, u, di, axs, wd)
        tiles.append((axs, wd))
    accd = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.psum)
    for jj in nl.static_range(H // 512):
        for u in nl.static_range(U):
            for i in nl.static_range(2):
                if st == 0 or ((g0 + u) * 2 + i) * 4 < nreal:
                    nisa.nc_matmul(dst=accd[:, jj * 512:(jj + 1) * 512], stationary=tiles[u][0][:, i * T:(i + 1) * T],
                                   moving=tiles[u][1][:, i * H + jj * 512:i * H + (jj + 1) * 512], accumulate=(u + i > 0))
    if yd == 1:             # the only phase-3 call (ys IS the output buffer): y = accd * a2 straight from PSUM (no memset, no add)
        nisa.tensor_scalar(dst=ys, data=accd, op0=nl.multiply, operand0=a2)
    elif first == 1:        # the first of several calls: ys = accd (no memset)
        nisa.tensor_copy(dst=ys, src=accd)
    else:
        nisa.tensor_tensor(dst=ys, data1=ys, data2=accd, op=nl.add)


@nki.jit
def moe_dyn3(xT, ga, cw, gidx, didx, cnt, egu_t, ed, U=4):
    """moe_dyn with every phase in super-iterations of U groups (dynamic loop over floor(ng / U), step U) + a tail loop of
    single groups. cnt = [ng, U floor(ng / U)]."""
    T = ga.shape[1]
    G = ga.shape[0]
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    xs = nl.ndarray((128, KT * T), dtype=xT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=xT)
    c_sb = nl.ndarray((1, 2), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb[0:1, 0:1])
    nm = nisa.register_alloc()
    nisa.register_load(nm, c_sb[0:1, 1:2])
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    for g in nl.dynamic_range(0, nm, U):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, U)
    for g in nl.dynamic_range(nm, ng):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, 1)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    for g in nl.dynamic_range(0, nm, U):
        _ph2(acth, axh, ra, g, U)
    for g in nl.dynamic_range(nm, ng):
        _ph2(acth, axh, ra, g, 1)
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    for g in nl.dynamic_range(0, nm, U):
        _ph3(didx, axh, ed, ys, g, U)
    for g in nl.dynamic_range(nm, ng):
        _ph3(didx, axh, ed, ys, g, 1)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


@nki.jit
def moe_dyn4(xT, ga, cw, gidx, didx, cnt, egu_t, ed, U=8, V=2):
    """moe_dyn3 with three loop levels per phase: super-iterations of U groups, then of V groups, then single groups.
    cnt = [b1, b2, ng]: b1 = U floor(ng / U), b2 = b1 + V floor((ng - b1) / V) (graph_inputs(levels=(U, V)))."""
    T = ga.shape[1]
    G = ga.shape[0]
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    xs = nl.ndarray((128, KT * T), dtype=xT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=xT)
    c_sb = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    b1 = nisa.register_alloc()
    nisa.register_load(b1, c_sb[0:1, 0:1])
    b2 = nisa.register_alloc()
    nisa.register_load(b2, c_sb[0:1, 1:2])
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb[0:1, 2:3])
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, 1)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    for g in nl.dynamic_range(0, b1, U):
        _ph2(acth, axh, ra, g, U)
    for g in nl.dynamic_range(b1, ng):
        _ph2(acth, axh, ra, g, 1)
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph3(didx, axh, ed, ys, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph3(didx, axh, ed, ys, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph3(didx, axh, ed, ys, g, 1)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


@nki.jit
def moe_dyn5(n8, ga, cw, gidx, didx, cnt, egu_t, ed, U=8, V=2):
    """moe_dyn4 + (a) the activation codes n8 [T, H] fp8 transposed in the kernel (32 PE transposes) instead of by the graph,
    (b) phase 2 static over all G groups: one read of the act scratch, one quantisation, 2 G transposes, one store
    (groups >= ng hold garbage that phase 3 never reads)."""
    T = ga.shape[1]
    G = ga.shape[0]
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    nr = nl.ndarray((T, H), dtype=n8.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=nr, src=n8)
    xs = nl.ndarray((128, KT * T), dtype=n8.dtype, buffer=nl.sbuf)
    for kt in nl.static_range(KT):
        tq = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tq, data=nr[:, kt * 128:(kt + 1) * 128])
        nisa.tensor_copy(dst=xs[:, kt * T:(kt + 1) * T], src=tq)
    c_sb = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    b1 = nisa.register_alloc()
    nisa.register_load(b1, c_sb[0:1, 0:1])
    b2 = nisa.register_alloc()
    nisa.register_load(b2, c_sb[0:1, 1:2])
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb[0:1, 2:3])
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph1(xs, gidx, ga, cw, egu_t, amax, acth, g, 1)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    aall = nl.ndarray((T, G, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aall, src=acth.permute((1, 0, 2)))
    a8 = nl.ndarray((T, G * GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a8, data=aall.reshape((T, G * GS * EI)), op0=nl.multiply, operand0=ra)
    axall = nl.ndarray((128, G, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for d in nl.static_range(2 * G):
        tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=a8[:, d * 128:(d + 1) * 128])
        nisa.tensor_copy(dst=axall[:, d // 2, (d % 2) * T:(d % 2 + 1) * T], src=tp)
    nisa.dma_copy(dst=axh.permute((1, 0, 2)), src=axall)
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph3(didx, axh, ed, ys, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph3(didx, axh, ed, ys, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph3(didx, axh, ed, ys, g, 1)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


def graph_inputs5(n, comb, egu_s, ed_c, kmax, levels):
    """graph glue for moe_dyn5: the distinct experts compacted with masks and sums (no top-k): pos[e] = number of used
    experts below e; slot k <- the expert with pos = k; slots >= n_distinct <- expert 0 with combine weight 0."""
    import torch
    T, E, dev = n.shape[0], comb.shape[1], n.device
    G = kmax // GS
    used = (comb > 0).float().amax(dim=0)                                                  # [E]
    ar = torch.arange(E, device=dev).float()
    before = (ar.reshape(1, E) < ar.reshape(E, 1)).float()                                 # [e, e'] = 1 when e' < e
    pos = (before * used.reshape(1, E)).sum(dim=-1)                                        # [E]
    ks = torch.arange(kmax, device=dev).float()
    sel = ((pos.reshape(1, E) == ks.reshape(kmax, 1)).float() * used.reshape(1, E))          # [K, E]
    idx = (sel * ar.reshape(1, E)).sum(dim=-1).round().long()                              # [K]
    nd = used.sum()
    valid = (ks < nd).float()                                                              # [K]
    a = torch.amax(torch.abs(n.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    n8 = (n.float() / a).to(torch.float8_e4m3fn)                                           # [T, H]
    ga = (a * torch.index_select(egu_s, 0, idx).reshape(1, kmax * O2)).reshape(T, G, GS * O2).permute(1, 0, 2).contiguous()
    cs = torch.index_select(comb, 1, idx) * valid.reshape(1, kmax)                         # [T, K]
    cw = (cs.unsqueeze(-1) * torch.index_select(ed_c, 0, idx).reshape(1, kmax, EI)).reshape(T, G, GS * EI).permute(1, 0, 2).contiguous()
    gidx = (idx.reshape(G, 1, GS) * 128 + torch.arange(128, device=dev).reshape(1, 128, 1)).to(torch.int32)
    didx = (idx.reshape(G, 2, 4, 1) * EI + torch.arange(EI, device=dev).reshape(1, 1, 1, EI)).reshape(G, 2, 128).permute(0, 2, 1).contiguous().to(torch.int32)
    ngf = torch.floor((nd + (GS - 1)) / GS)
    b1 = levels[0] * torch.floor(ngf / levels[0])
    b2 = b1 + levels[1] * torch.floor((ngf - b1) / levels[1])
    cnt = torch.stack([b1, b2, ngf]).to(torch.int32).reshape(1, 3)
    return n8, ga, cw, gidx, didx, cnt


def _gu_load6(eid, ga, cw, egu3, g, off, ei, w, gs, cs):
    """scalar-offset form: one contiguous [128, 2048] expert slice per DMA, the expert id read from SBUF (no per-partition
    index fetch)."""
    T = ga.shape[1]
    nisa.dma_copy(dst=ei, src=eid.ap(pattern=[[GS, 1], [1, GS]], offset=off * GS, scalar_offset=g, indirect_dim=0))
    for j in nl.static_range(GS):
        nisa.dma_copy(dst=w[:, j * KT * O2:(j + 1) * KT * O2],
                      src=egu3.ap(pattern=[[KT * O2, 128], [1, KT * O2]], offset=0, scalar_offset=ei[0:1, j:j + 1], indirect_dim=0))
    nisa.dma_copy(dst=gs, src=ga.ap(pattern=[[GS * O2, T], [1, GS * O2]], offset=off * T * GS * O2, scalar_offset=g, indirect_dim=0))
    nisa.dma_copy(dst=cs, src=cw.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=off * T * GS * EI, scalar_offset=g, indirect_dim=0))


def _ph1_6(xs, eid, ga, cw, egu3, amax, acth, g, U):
    T = ga.shape[1]
    ws = []
    for u in nl.static_range(U):
        ei = nl.ndarray((1, GS), dtype=nl.int32, buffer=nl.sbuf)
        w = nl.ndarray((128, GS * KT * O2), dtype=egu3.dtype, buffer=nl.sbuf)
        gs = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
        cs = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        _gu_load6(eid, ga, cw, egu3, g, u, ei, w, gs, cs)
        ws.append((w, gs, cs))
    for u in nl.static_range(U):
        _gu_compute(xs, ws[u][0], ws[u][1], ws[u][2], amax, acth, g, u)


def _ph3_6(eid, axh, ed3, ys, g, U):
    T = ys.shape[0]
    tiles = []
    for u in nl.static_range(U):
        ei = nl.ndarray((1, GS), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ei, src=eid.ap(pattern=[[GS, 1], [1, GS]], offset=u * GS, scalar_offset=g, indirect_dim=0))
        axs = nl.ndarray((128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.dma_copy(dst=axs, src=axh.ap(pattern=[[2 * T, 128], [1, 2 * T]], offset=u * 128 * 2 * T, scalar_offset=g, indirect_dim=0))
        wd = nl.ndarray((128, 2 * H), dtype=ed3.dtype, buffer=nl.sbuf)
        for j in nl.static_range(GS):
            nisa.dma_copy(dst=wd[(j % 4) * EI:(j % 4 + 1) * EI, (j // 4) * H:(j // 4 + 1) * H],
                          src=ed3.ap(pattern=[[H, EI], [1, H]], offset=0, scalar_offset=ei[0:1, j:j + 1], indirect_dim=0))
        tiles.append((axs, wd))
    accd = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.psum)
    for jj in nl.static_range(H // 512):
        for u in nl.static_range(U):
            for i in nl.static_range(2):
                nisa.nc_matmul(dst=accd[:, jj * 512:(jj + 1) * 512], stationary=tiles[u][0][:, i * T:(i + 1) * T],
                               moving=tiles[u][1][:, i * H + jj * 512:i * H + (jj + 1) * 512], accumulate=(u + i > 0))
    nisa.tensor_tensor(dst=ys, data1=ys, data2=accd, op=nl.add)


@nki.jit
def moe_dyn6(n8, ga, cw, eid, cnt, egu_t, ed, U=8, V=2):
    """moe_dyn5 with scalar-offset expert DMAs: eid [G, 1, 8] int32 expert ids (graph_inputs6)."""
    T = ga.shape[1]
    G = ga.shape[0]
    E = egu_t.shape[0] // 128
    egu3 = egu_t.reshape((E, 128, KT * O2))
    ed3 = ed.reshape((E, EI, H))
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    nr = nl.ndarray((T, H), dtype=n8.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=nr, src=n8)
    xs = nl.ndarray((128, KT * T), dtype=n8.dtype, buffer=nl.sbuf)
    for kt in nl.static_range(KT):
        tq = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tq, data=nr[:, kt * 128:(kt + 1) * 128])
        nisa.tensor_copy(dst=xs[:, kt * T:(kt + 1) * T], src=tq)
    c_sb = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    b1 = nisa.register_alloc()
    nisa.register_load(b1, c_sb[0:1, 0:1])
    b2 = nisa.register_alloc()
    nisa.register_load(b2, c_sb[0:1, 1:2])
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb[0:1, 2:3])
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph1_6(xs, eid, ga, cw, egu3, amax, acth, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph1_6(xs, eid, ga, cw, egu3, amax, acth, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph1_6(xs, eid, ga, cw, egu3, amax, acth, g, 1)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    aall = nl.ndarray((T, G, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aall, src=acth.permute((1, 0, 2)))
    a8 = nl.ndarray((T, G * GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a8, data=aall.reshape((T, G * GS * EI)), op0=nl.multiply, operand0=ra)
    axall = nl.ndarray((128, G, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for d in nl.static_range(2 * G):
        tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=a8[:, d * 128:(d + 1) * 128])
        nisa.tensor_copy(dst=axall[:, d // 2, (d % 2) * T:(d % 2 + 1) * T], src=tp)
    nisa.dma_copy(dst=axh.permute((1, 0, 2)), src=axall)
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph3_6(eid, axh, ed3, ys, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph3_6(eid, axh, ed3, ys, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph3_6(eid, axh, ed3, ys, g, 1)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


def graph_inputs6(n, comb, egu_s, ed_c, kmax, levels):
    """graph_inputs5 with expert ids eid [G, 1, 8] in place of the per-partition row indices."""
    import torch
    n8, ga, cw, gidx, didx, cnt = graph_inputs5(n, comb, egu_s, ed_c, kmax, levels)
    eid = torch.div(gidx[:, 0:1, :], 128, rounding_mode='floor').contiguous()
    return n8, ga, cw, eid, cnt


def _gu_load7(eid, gsv, csv, dcv, egu3, g, off, ei, w, gs, cs, dc):
    T = cs.shape[0]
    nisa.dma_copy(dst=ei, src=eid.ap(pattern=[[GS, 1], [1, GS]], offset=off * GS, scalar_offset=g, indirect_dim=0))
    for j in nl.static_range(GS):
        nisa.dma_copy(dst=w[:, j * KT * O2:(j + 1) * KT * O2],
                      src=egu3.ap(pattern=[[KT * O2, 128], [1, KT * O2]], offset=0, scalar_offset=ei[0:1, j:j + 1], indirect_dim=0))
    nisa.dma_copy(dst=gs, src=gsv.ap(pattern=[[0, T], [1, GS * O2]], offset=off * GS * O2, scalar_offset=g, indirect_dim=0))
    nisa.dma_copy(dst=dc, src=dcv.ap(pattern=[[0, T], [1, GS * EI]], offset=off * GS * EI, scalar_offset=g, indirect_dim=0))
    nisa.dma_copy(dst=cs, src=csv.ap(pattern=[[GS, T], [1, GS]], offset=off * T * GS, scalar_offset=g, indirect_dim=0))


def _gu_compute7(xs, w, a, gs, cs, dc, amax, acth, g, off, st=0, g0=0, aall=None, fast=0):
    """hgu = bf16((acc * a) * egu_s); act = (bf16(silu(gate) * up) * comb) * ed_c  (the torch op order)."""
    T = cs.shape[0]
    acc = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.psum)
    wv = w.reshape((128, GS, KT * O2))
    for kt in nl.static_range(KT):
        nisa.nc_matmul(dst=acc.reshape((T, GS, O2)), stationary=xs[:, kt * T:(kt + 1) * T],
                       moving=wv[:, :, kt * O2:(kt + 1) * O2], accumulate=(kt > 0))
    ha = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ha, data=acc, op0=nl.multiply, operand0=a)
    hb = nl.ndarray((T, GS * O2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=hb, data1=ha, data2=gs, op=nl.multiply)
    sg = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=hb)
    pb = nl.ndarray((T, GS * EI), dtype=nl.bfloat16, buffer=nl.sbuf)
    if fast >= 1:   # gate x up of the GS slots in ONE vector instruction (strided views) instead of GS
        nisa.tensor_tensor(dst=pb.reshape((T, GS, 1, EI)), data1=sg.reshape((T, GS, 2, EI))[:, :, 0:1, :],
                           data2=hb.reshape((T, GS, 2, EI))[:, :, 1:2, :], op=nl.multiply)
    else:
        for j in nl.static_range(GS):
            nisa.tensor_tensor(dst=pb[:, j * EI:(j + 1) * EI], data1=sg[:, j * O2:j * O2 + EI], data2=hb[:, j * O2 + EI:(j + 1) * O2], op=nl.multiply)
    pc = nl.ndarray((T, GS, EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pc, data1=pb.reshape((T, GS, EI)), data2=cs.reshape((T, GS, 1)).broadcast(2, EI), op=nl.multiply)
    if st == 2:
        act = aall[:, g0 + off, :]
    else:
        act = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=pc.reshape((T, GS * EI)), data2=dc, op=nl.multiply)
    if fast < 1:   # fast: the row amax is taken once over all the groups' activations (moe_row9), not per group
        # row amax with the vector engine only (an ACT abs between the SiLUs would switch the activation table every group)
        mx = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=act, axis=1)
        mn = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mn, op=nl.minimum, data=act, axis=1)
        nisa.tensor_scalar(dst=mn, data=mn, op0=nl.multiply, operand0=-1.0)
        nisa.tensor_tensor(dst=mx, data1=mx, data2=mn, op=nl.maximum)
        nisa.tensor_tensor(dst=amax, data1=amax, data2=mx, op=nl.maximum)
    if st == 1:
        nisa.dma_copy(dst=acth.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=(g0 + off) * T * GS * EI), src=act)
    elif st == 0:
        nisa.dma_copy(dst=acth.ap(pattern=[[GS * EI, T], [1, GS * EI]], offset=off * T * GS * EI, scalar_offset=g, indirect_dim=0), src=act)


def _ph1_7(xs, a, eid, gsv, csv, dcv, egu3, amax, acth, g, U):
    T = a.shape[0]
    ws = []
    for u in nl.static_range(U):
        ei = nl.ndarray((1, GS), dtype=nl.int32, buffer=nl.sbuf)
        w = nl.ndarray((128, GS * KT * O2), dtype=egu3.dtype, buffer=nl.sbuf)
        gs = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
        cs = nl.ndarray((T, GS), dtype=nl.float32, buffer=nl.sbuf)
        dc = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        _gu_load7(eid, gsv, csv, dcv, egu3, g, u, ei, w, gs, cs, dc)
        ws.append((w, gs, cs, dc))
    for u in nl.static_range(U):
        _gu_compute7(xs, ws[u][0], a, ws[u][1], ws[u][2], ws[u][3], amax, acth, g, u)


@nki.jit
def moe_dyn7(n8, a, gsv, csv, dcv, eid, cnt, egu_t, ed, U=8, V=2):
    """moe_dyn6 with the per-slot scales applied in the kernel (no [T, K x 64] scale tensors built by the graph):
    a [T, 1] activation scale, gsv [G, 512] = egu_s[idx], csv [G, T, 8] combine weights, dcv [G, 256] = ed_c[idx]."""
    T = csv.shape[1]
    G = csv.shape[0]
    E = egu_t.shape[0] // 128
    egu3 = egu_t.reshape((E, 128, KT * O2))
    ed3 = ed.reshape((E, EI, H))
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    nr = nl.ndarray((T, H), dtype=n8.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=nr, src=n8)
    asb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=asb, src=a)
    xs = nl.ndarray((128, KT * T), dtype=n8.dtype, buffer=nl.sbuf)
    for kt in nl.static_range(KT):
        tq = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tq, data=nr[:, kt * 128:(kt + 1) * 128])
        nisa.tensor_copy(dst=xs[:, kt * T:(kt + 1) * T], src=tq)
    c_sb = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    b1 = nisa.register_alloc()
    nisa.register_load(b1, c_sb[0:1, 0:1])
    b2 = nisa.register_alloc()
    nisa.register_load(b2, c_sb[0:1, 1:2])
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb[0:1, 2:3])
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph1_7(xs, asb, eid, gsv, csv, dcv, egu3, amax, acth, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph1_7(xs, asb, eid, gsv, csv, dcv, egu3, amax, acth, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph1_7(xs, asb, eid, gsv, csv, dcv, egu3, amax, acth, g, 1)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    aall = nl.ndarray((T, G, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aall, src=acth.permute((1, 0, 2)))
    a8 = nl.ndarray((T, G * GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a8, data=aall.reshape((T, G * GS * EI)), op0=nl.multiply, operand0=ra)
    axall = nl.ndarray((128, G, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for d in nl.static_range(2 * G):
        tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=a8[:, d * 128:(d + 1) * 128])
        nisa.tensor_copy(dst=axall[:, d // 2, (d % 2) * T:(d % 2 + 1) * T], src=tp)
    nisa.dma_copy(dst=axh.permute((1, 0, 2)), src=axall)
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph3_6(eid, axh, ed3, ys, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph3_6(eid, axh, ed3, ys, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph3_6(eid, axh, ed3, ys, g, 1)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


@nki.jit
def moe_dyn8(n8, a, gsv, csv, dcv, eid, didx, cnt, egu_t, ed, U=8, V=2):
    """moe_dyn7 with the down tiles read by per-partition indirect DMAs (didx, 2 x [128, 4096] per group) as in moe_dyn5.
    (moe_dyn7: moe_dyn6 with the per-slot scales applied in the kernel (no [T, K x 64] scale tensors built by the graph):
    a [T, 1] activation scale, gsv [G, 512] = egu_s[idx], csv [G, T, 8] combine weights, dcv [G, 256] = ed_c[idx]."""
    T = csv.shape[1]
    G = csv.shape[0]
    E = egu_t.shape[0] // 128
    egu3 = egu_t.reshape((E, 128, KT * O2))
    ed3 = ed.reshape((E, EI, H))
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    nr = nl.ndarray((T, H), dtype=n8.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=nr, src=n8)
    asb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=asb, src=a)
    xs = nl.ndarray((128, KT * T), dtype=n8.dtype, buffer=nl.sbuf)
    for kt in nl.static_range(KT):
        tq = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tq, data=nr[:, kt * 128:(kt + 1) * 128])
        nisa.tensor_copy(dst=xs[:, kt * T:(kt + 1) * T], src=tq)
    c_sb = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    b1 = nisa.register_alloc()
    nisa.register_load(b1, c_sb[0:1, 0:1])
    b2 = nisa.register_alloc()
    nisa.register_load(b2, c_sb[0:1, 1:2])
    ng = nisa.register_alloc()
    nisa.register_load(ng, c_sb[0:1, 2:3])
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph1_7(xs, asb, eid, gsv, csv, dcv, egu3, amax, acth, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph1_7(xs, asb, eid, gsv, csv, dcv, egu3, amax, acth, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph1_7(xs, asb, eid, gsv, csv, dcv, egu3, amax, acth, g, 1)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    aall = nl.ndarray((T, G, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aall, src=acth.permute((1, 0, 2)))
    a8 = nl.ndarray((T, G * GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a8, data=aall.reshape((T, G * GS * EI)), op0=nl.multiply, operand0=ra)
    axall = nl.ndarray((128, G, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for d in nl.static_range(2 * G):
        tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=a8[:, d * 128:(d + 1) * 128])
        nisa.tensor_copy(dst=axall[:, d // 2, (d % 2) * T:(d % 2 + 1) * T], src=tp)
    nisa.dma_copy(dst=axh.permute((1, 0, 2)), src=axall)
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    for g in nl.dynamic_range(0, b1, U):
        _ph3(didx, axh, ed, ys, g, U)
    for g in nl.dynamic_range(b1, b2, V):
        _ph3(didx, axh, ed, ys, g, V)
    for g in nl.dynamic_range(b2, ng):
        _ph3(didx, axh, ed, ys, g, 1)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


def graph_inputs7(n, comb, egu_s, ed_c, kmax, levels):
    """glue for moe_dyn7: mask / sum compaction of the distinct experts, small per-slot vectors only."""
    import torch
    T, E, dev = n.shape[0], comb.shape[1], n.device
    G = kmax // GS
    used = (comb > 0).float().amax(dim=0)                                                  # [E]
    ar = torch.arange(E, device=dev).float()
    before = (ar.reshape(1, E) < ar.reshape(E, 1)).float()
    pos = (before * used.reshape(1, E)).sum(dim=-1)                                        # [E] used experts below e
    ks = torch.arange(kmax, device=dev).float()
    sel = (pos.reshape(1, E) == ks.reshape(kmax, 1)).float() * used.reshape(1, E)           # [K, E]
    idx = (sel * ar.reshape(1, E)).sum(dim=-1).round().long()                              # [K] (pads: expert 0)
    nd = used.sum()
    valid = (ks < nd).float()
    a = torch.amax(torch.abs(n.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    n8 = (n.float() / a).to(torch.float8_e4m3fn)
    gsv = torch.index_select(egu_s, 0, idx).reshape(G, GS * O2)
    csv = (torch.index_select(comb, 1, idx) * valid.reshape(1, kmax)).reshape(T, G, GS).permute(1, 0, 2).contiguous()
    dcv = torch.index_select(ed_c, 0, idx).reshape(G, GS * EI)
    eid = idx.reshape(G, 1, GS).to(torch.int32)
    ngf = torch.floor((nd + (GS - 1)) / GS)
    b1 = levels[0] * torch.floor(ngf / levels[0])
    b2 = b1 + levels[1] * torch.floor((ngf - b1) / levels[1])
    cnt = torch.stack([b1, b2, ngf]).to(torch.int32).reshape(1, 3)
    return n8, a, gsv, csv, dcv, eid, cnt


def graph_inputs8(n, comb, egu_s, ed_c, kmax, levels):
    import torch
    n8, a, gsv, csv, dcv, eid, cnt = graph_inputs7(n, comb, egu_s, ed_c, kmax, levels)
    G = kmax // GS
    idx = eid.reshape(kmax).long()
    didx = (idx.reshape(G, 2, 4, 1) * EI + torch.arange(EI, device=n.device).reshape(1, 1, 1, EI)).reshape(G, 2, 128).permute(0, 2, 1).contiguous().to(torch.int32)
    return n8, a, gsv, csv, dcv, eid, didx, cnt


def _gu_load9(idl, gsl, csl, dsl, egu3, g, off, ei, w, gs, cs, dc):
    """the group's expert ids, slices and per-slot vectors from the kernel's own slot tables (built in the prologue)."""
    T = cs.shape[0]
    nisa.dma_copy(dst=ei, src=idl.ap(pattern=[[GS, 1], [1, GS]], offset=off * GS, scalar_offset=g, indirect_dim=0))
    for j in nl.static_range(GS):
        nisa.dma_copy(dst=w[:, j * KT * O2:(j + 1) * KT * O2],
                      src=egu3.ap(pattern=[[KT * O2, 128], [1, KT * O2]], offset=0, scalar_offset=ei[0:1, j:j + 1], indirect_dim=0))
    nisa.dma_copy(dst=gs, src=gsl.ap(pattern=[[0, T], [1, GS * O2]], offset=off * GS * O2, scalar_offset=g, indirect_dim=0))
    nisa.dma_copy(dst=dc, src=dsl.ap(pattern=[[0, T], [1, GS * EI]], offset=off * GS * EI, scalar_offset=g, indirect_dim=0))
    nisa.dma_copy(dst=cs, src=csl.ap(pattern=[[1, T], [T, GS]], offset=off * GS * T, scalar_offset=g, indirect_dim=0))


def _ph1_9(xs, a, idl, gsl, csl, dsl, egu3, amax, acth, g, U, order=0, st=0, g0=0, aall=None, nreal=1000000, fast=0):
    """order 1: every group's small loads (ids, scales, combine weights) first, then the weight slices, then the compute."""
    T = a.shape[0]
    ws = []
    if order == 1:
        for u in nl.static_range(U):
            ei = nl.ndarray((1, GS), dtype=nl.int32, buffer=nl.sbuf)
            gs = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
            cs = nl.ndarray((T, GS), dtype=nl.float32, buffer=nl.sbuf)
            dc = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
            if st >= 1:
                nisa.dma_copy(dst=ei, src=idl.ap(pattern=[[GS, 1], [1, GS]], offset=(g0 + u) * GS))
                nisa.dma_copy(dst=gs, src=gsl.ap(pattern=[[0, T], [1, GS * O2]], offset=(g0 + u) * GS * O2))
                nisa.dma_copy(dst=dc, src=dsl.ap(pattern=[[0, T], [1, GS * EI]], offset=(g0 + u) * GS * EI))
                nisa.dma_copy(dst=cs, src=csl.ap(pattern=[[1, T], [T, GS]], offset=(g0 + u) * GS * T))
            else:
                nisa.dma_copy(dst=ei, src=idl.ap(pattern=[[GS, 1], [1, GS]], offset=u * GS, scalar_offset=g, indirect_dim=0))
                nisa.dma_copy(dst=gs, src=gsl.ap(pattern=[[0, T], [1, GS * O2]], offset=u * GS * O2, scalar_offset=g, indirect_dim=0))
                nisa.dma_copy(dst=dc, src=dsl.ap(pattern=[[0, T], [1, GS * EI]], offset=u * GS * EI, scalar_offset=g, indirect_dim=0))
                nisa.dma_copy(dst=cs, src=csl.ap(pattern=[[1, T], [T, GS]], offset=u * GS * T, scalar_offset=g, indirect_dim=0))
            ws.append((ei, gs, cs, dc))
        wl = []
        for u in nl.static_range(U):
            w = nl.ndarray((128, GS * KT * O2), dtype=egu3.dtype, buffer=nl.sbuf)
            for j in nl.static_range(GS):
                if st >= 1 and (g0 + u) * GS + j >= nreal:      # static padding slot: nothing to stream (its combine weight is 0)
                    nisa.memset(dst=w[:, j * KT * O2:(j + 1) * KT * O2], value=0.0)
                else:
                    nisa.dma_copy(dst=w[:, j * KT * O2:(j + 1) * KT * O2],
                                  src=egu3.ap(pattern=[[KT * O2, 128], [1, KT * O2]], offset=0, scalar_offset=ws[u][0][0:1, j:j + 1], indirect_dim=0))
            wl.append(w)
        for u in nl.static_range(U):
            _gu_compute7(xs, wl[u], a, ws[u][1], ws[u][2], ws[u][3], amax, acth, g, u, st, g0, aall, fast)
        return
    for u in nl.static_range(U):
        ei = nl.ndarray((1, GS), dtype=nl.int32, buffer=nl.sbuf)
        w = nl.ndarray((128, GS * KT * O2), dtype=egu3.dtype, buffer=nl.sbuf)
        gs = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
        cs = nl.ndarray((T, GS), dtype=nl.float32, buffer=nl.sbuf)
        dc = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        _gu_load9(idl, gsl, csl, dsl, egu3, g, u, ei, w, gs, cs, dc)
        ws.append((w, gs, cs, dc))
    for u in nl.static_range(U):
        _gu_compute7(xs, ws[u][0], a, ws[u][1], ws[u][2], ws[u][3], amax, acth, g, u)


@nki.jit
def moe_dyn9(n8, a, comb, egu_s, ed_c, egu_t, ed, KM=80, U=8, V=2, dbg=0, order=1, stat=0):
    """The whole distinct-expert selection inside the kernel: the graph passes the activation codes n8 [T, H] fp8, their row
    scale a [T, 1], the router's combine matrix comb [T, E] (0 = not selected) and the stored per-expert scales.
    Prologue: used[e] = any_t comb[t, e] > 0 (PE: ones^T @ mask), pos[e] = number of used experts below e (PE: strict
    lower-triangular 0/1 matrix, exact f32 accumulation), then the slot tables (expert id, comb column, egu_s / ed_c rows,
    down-row indices) are scattered to slot pos[e] by indirect DMAs (unused experts: out-of-bounds index, skipped). Slots >=
    n_distinct keep expert 0 with zero combine weight / zero scales. Loop bounds b1 / b2 / ng computed exactly with
    compare-and-sum on one partition. Then the three phases of moe_dyn8. KM = slot capacity (multiple of 8, >= 10 T)."""
    T = comb.shape[0]
    E = comb.shape[1]
    NE = E // 128
    G = KM // GS
    egu3 = egu_t.reshape((E, 128, KT * O2))
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    idl = nl.ndarray((KM, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    csl = nl.ndarray((KM, T), dtype=nl.float32, buffer=nl.shared_hbm)
    gsl = nl.ndarray((KM, O2), dtype=nl.float32, buffer=nl.shared_hbm)
    dsl = nl.ndarray((KM, EI), dtype=nl.float32, buffer=nl.shared_hbm)
    dfl = nl.ndarray((KM, EI), dtype=nl.int32, buffer=nl.shared_hbm)

    # ---- slot tables: zero fill
    ZW = O2 if T <= O2 else T
    zf = nl.ndarray((128, ZW), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zf, value=0.0)
    zi = nl.ndarray((128, EI), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=zi, value=0)
    for r0 in nl.static_range(0, KM, 128):
        rn = min(128, KM - r0)
        nisa.dma_copy(dst=idl[r0:r0 + rn, :], src=zi[0:rn, 0:1])
        nisa.dma_copy(dst=csl[r0:r0 + rn, :], src=zf[0:rn, 0:T])
        nisa.dma_copy(dst=gsl[r0:r0 + rn, :], src=zf[0:rn, 0:O2])
        nisa.dma_copy(dst=dsl[r0:r0 + rn, :], src=zf[0:rn, 0:EI])
        nisa.dma_copy(dst=dfl[r0:r0 + rn, :], src=zi[0:rn, :])

    # ---- used experts and their slot positions
    cm = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cm, src=comb)
    mk = nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mk, data=cm, op0=nl.greater, operand0=0.0)
    onesT = nl.ndarray((T, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesT, value=1.0)
    ones1 = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones1, value=1.0)
    used = nl.ndarray((128, NE), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in nl.static_range(NE):
        uc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=uc, stationary=mk[:, c * 128:(c + 1) * 128], moving=onesT)
        nisa.tensor_scalar(dst=used[:, c:c + 1], data=uc, op0=nl.minimum, operand0=1.0)
    # strict triangle tri[e', e] = 1 when e' < e, per row chunk c' (e' = c' 128 + p)
    tri = nl.ndarray((128, NE, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c2 in nl.static_range(NE):
        dif = nl.ndarray((128, E), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=dif, pattern=[[1, E]], offset=-c2 * 128, channel_multiplier=-1)       # e - e'
        nisa.tensor_scalar(dst=tri[:, c2, :], data=dif, op0=nl.greater, operand0=0)
    sidx = nl.ndarray((128, NE), dtype=nl.int32, buffer=nl.sbuf)
    for c in nl.static_range(NE):
        pc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        for c2 in nl.static_range(NE):
            nisa.nc_matmul(dst=pc, stationary=tri[:, c2, c * 128:(c + 1) * 128], moving=used[:, c2:c2 + 1], accumulate=(c2 > 0))
        # unused experts -> index 1e6 (out of bounds: the scatter skips them)
        pf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pf, data=used[:, c:c + 1], op0=nl.multiply, operand0=-1000000.0, op1=nl.add, operand1=1000000.0)
        nisa.tensor_tensor(dst=sidx[:, c:c + 1], data1=pf, data2=pc, op=nl.add)
    # number of distinct experts on partition 0: nd = sum used
    ndp = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
    for c in nl.static_range(NE):
        nisa.nc_matmul(dst=ndp, stationary=used[:, c:c + 1], moving=ones1, accumulate=(c > 0))
    nd = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nd, src=ndp)

    # ---- scatter the slot tables
    for c in nl.static_range(NE):
        eid = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=eid, pattern=[[0, 1]], offset=c * 128, channel_multiplier=1)
        dpat = nl.ndarray((128, EI), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=dpat, pattern=[[1, EI]], offset=c * 128 * EI, channel_multiplier=EI)
        cT = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cT, src=comb.ap(pattern=[[1, 128], [E, T]], offset=c * 128))
        gr = nl.ndarray((128, O2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gr, src=egu_s[c * 128:(c + 1) * 128, :])
        dr = nl.ndarray((128, EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dr, src=ed_c[c * 128:(c + 1) * 128, :])
        vo = sidx[:, c:c + 1]
        nisa.dma_copy(dst=idl.ap(pattern=[[1, 128], [1, 1]], offset=0, vector_offset=vo, indirect_dim=0), src=eid, oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=csl.ap(pattern=[[T, 128], [1, T]], offset=0, vector_offset=vo, indirect_dim=0), src=cT, oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=gsl.ap(pattern=[[O2, 128], [1, O2]], offset=0, vector_offset=vo, indirect_dim=0), src=gr, oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=dsl.ap(pattern=[[EI, 128], [1, EI]], offset=0, vector_offset=vo, indirect_dim=0), src=dr, oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=dfl.ap(pattern=[[EI, 128], [1, EI]], offset=0, vector_offset=vo, indirect_dim=0), src=dpat, oob_mode=nisa.oob_mode.skip)

    # ---- loop bounds: ng = #{k < G: 8 k < nd}; b1 = U #{m: U (m + 1) <= ng}; b2 = b1 + V #{m: V (m + 1) <= ng - b1}
    kv = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=kv, pattern=[[1, 64]], offset=0, channel_multiplier=0)
    t8 = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t8, data=kv, op0=nl.multiply, operand0=float(GS), op1=nl.less, operand1=nd)
    ngf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ngf, op=nl.add, data=t8[:, 0:G], axis=1)
    tu = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tu, data=kv, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=float(U))
    nisa.tensor_scalar(dst=tu, data=tu, op0=nl.less_equal, operand0=ngf)
    b1f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=b1f, op=nl.add, data=tu, axis=1)
    nisa.tensor_scalar(dst=b1f, data=b1f, op0=nl.multiply, operand0=float(U))
    rem = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rem, data1=ngf, data2=b1f, op=nl.subtract)
    tv = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tv, data=kv, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=float(V))
    nisa.tensor_scalar(dst=tv, data=tv, op0=nl.less_equal, operand0=rem)
    b2f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=b2f, op=nl.add, data=tv, axis=1)
    nisa.tensor_scalar(dst=b2f, data=b2f, op0=nl.multiply, operand0=float(V), op1=nl.add, operand1=b1f)
    cnti = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cnti[:, 0:1], src=b1f)
    nisa.tensor_copy(dst=cnti[:, 1:2], src=b2f)
    nisa.tensor_copy(dst=cnti[:, 2:3], src=ngf)
    b1 = nisa.register_alloc()
    nisa.register_load(b1, cnti[0:1, 0:1])
    b2 = nisa.register_alloc()
    nisa.register_load(b2, cnti[0:1, 1:2])
    ng = nisa.register_alloc()
    nisa.register_load(ng, cnti[0:1, 2:3])
    cnth = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.shared_hbm)
    sidh = nl.ndarray((128, NE), dtype=nl.int32, buffer=nl.shared_hbm)
    if dbg == 1:
        dbt = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dbt[:, 0:1], src=b1f)
        nisa.tensor_copy(dst=dbt[:, 1:2], src=b2f)
        nisa.tensor_copy(dst=dbt[:, 2:3], src=ngf)
        nisa.tensor_copy(dst=dbt[:, 3:4], src=nd)
        nisa.dma_copy(dst=cnth, src=dbt)
        nisa.dma_copy(dst=sidh, src=sidx)

    # ---- activations (group-major views of the slot tables: the loop register indexes groups)
    idlg = idl.reshape((G, GS))
    gslg = gsl.reshape((G, GS * O2))
    cslg = csl.reshape((G, GS * T))
    dslg = dsl.reshape((G, GS * EI))
    nr = nl.ndarray((T, H), dtype=n8.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=nr, src=n8)
    asb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=asb, src=a)
    xs = nl.ndarray((128, KT * T), dtype=n8.dtype, buffer=nl.sbuf)
    for kt in nl.static_range(KT):
        tq = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tq, data=nr[:, kt * 128:(kt + 1) * 128])
        nisa.tensor_copy(dst=xs[:, kt * T:(kt + 1) * T], src=tq)
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    if stat == 1:
        # static trip counts (small T): every one of the G slot groups is streamed (unused slots: expert 0, zero weight), no
        # device-side loop, so the scheduler can overlap the groups' loads and there is no back-edge bubble
        for gq in nl.static_range(0, G, U):
            _ph1_9(xs, asb, idlg, gslg, cslg, dslg, egu3, amax, acth, 0, min(U, G - gq), 1, 1, gq)
    else:
        for g in nl.dynamic_range(0, b1, U):
            _ph1_9(xs, asb, idlg, gslg, cslg, dslg, egu3, amax, acth, g, U, order)
        for g in nl.dynamic_range(b1, b2, V):
            _ph1_9(xs, asb, idlg, gslg, cslg, dslg, egu3, amax, acth, g, V, order)
        for g in nl.dynamic_range(b2, ng):
            _ph1_9(xs, asb, idlg, gslg, cslg, dslg, egu3, amax, acth, g, 1, order)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    aall = nl.ndarray((T, G, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aall, src=acth.permute((1, 0, 2)))
    a8 = nl.ndarray((T, G * GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a8, data=aall.reshape((T, G * GS * EI)), op0=nl.multiply, operand0=ra)
    axall = nl.ndarray((128, G, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for d in nl.static_range(2 * G):
        tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=a8[:, d * 128:(d + 1) * 128])
        nisa.tensor_copy(dst=axall[:, d // 2, (d % 2) * T:(d % 2 + 1) * T], src=tp)
    nisa.dma_copy(dst=axh.permute((1, 0, 2)), src=axall)
    didx = dfl.reshape((G, 256))
    ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ys, value=0.0)
    if stat == 1:
        for gq in nl.static_range(0, G, U):
            _ph3(didx, axh, ed, ys, 0, min(U, G - gq), 1, 1, gq)
    else:
        for g in nl.dynamic_range(0, b1, U):
            _ph3(didx, axh, ed, ys, g, U, tmaj=1)
        for g in nl.dynamic_range(b1, b2, V):
            _ph3(didx, axh, ed, ys, g, V, tmaj=1)
        for g in nl.dynamic_range(b2, ng):
            _ph3(didx, axh, ed, ys, g, 1, tmaj=1)
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    if dbg == 1:
        return y, idl, csl, gsl, dfl, cnth, sidh
    return y


@nki.jit
def moe_row9(n8, a, idx, idxf, csT, egu_s, ed_c, egu_t, ed, U=8, mem=1, NR=1000000, fast=0, yd=0):
    """Row-slot STATIC form for small T (no selection logic, no device-side loop, no back-edge bubbles): slot s = t * 10 + j holds
    the j-th expert of row t (the router's top-k indices; an expert chosen by two rows is streamed twice, as in the served row
    form), padded to KM = 8 G slots with expert 0 at zero weight. Inputs from the graph: n8 [T, H] fp8, a [T, 1], idx [KM, 1]
    int32 and the same ids as f32 (idxf), csT [KM, T] f32 (combine weight of slot s for row t, 0 off its row), the stored scales.
    Prologue = 2 gathers (egu_s / ed_c rows by vector offset) + the down-row index table; then the phases of moe_dyn9 with
    static trip counts."""
    T = n8.shape[0]
    KM = idx.shape[0]
    E = egu_s.shape[0]
    G = KM // GS
    egu3 = egu_t.reshape((E, 128, KT * O2))
    y = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    acth = nl.ndarray((G, T, GS * EI), dtype=nl.float32, buffer=nl.shared_hbm)
    axh = nl.ndarray((G, 128, 2 * T), dtype=nl.float8_e4m3, buffer=nl.shared_hbm)
    gsl = nl.ndarray((KM, O2), dtype=nl.float32, buffer=nl.shared_hbm)
    dsl = nl.ndarray((KM, EI), dtype=nl.float32, buffer=nl.shared_hbm)
    dfl = nl.ndarray((KM, EI), dtype=nl.int32, buffer=nl.shared_hbm)
    for r0 in nl.static_range(0, KM, 128):
        rn = min(128, KM - r0)
        iv = nl.ndarray((rn, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=iv, src=idx[r0:r0 + rn, :])
        gb = nl.ndarray((rn, O2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gb, src=egu_s.ap(pattern=[[O2, rn], [1, O2]], offset=0, vector_offset=iv, indirect_dim=0))
        nisa.dma_copy(dst=gsl[r0:r0 + rn, :], src=gb)
        db = nl.ndarray((rn, EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=db, src=ed_c.ap(pattern=[[EI, rn], [1, EI]], offset=0, vector_offset=iv, indirect_dim=0))
        nisa.dma_copy(dst=dsl[r0:r0 + rn, :], src=db)
        ivf = nl.ndarray((rn, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ivf, src=idxf[r0:r0 + rn, :])
        nisa.tensor_scalar(dst=ivf, data=ivf, op0=nl.multiply, operand0=float(EI))
        iof = nl.ndarray((rn, EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=iof, pattern=[[1, EI]], offset=0, channel_multiplier=0)
        df = nl.ndarray((rn, EI), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=df, data=iof, op0=nl.add, operand0=ivf)
        nisa.dma_copy(dst=dfl[r0:r0 + rn, :], src=df)
    idlg = idx.reshape((G, GS))
    gslg = gsl.reshape((G, GS * O2))
    cslg = csT.reshape((G, GS * T))
    dslg = dsl.reshape((G, GS * EI))
    nr = nl.ndarray((T, H), dtype=n8.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=nr, src=n8)
    asb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=asb, src=a)
    xs = nl.ndarray((128, KT * T), dtype=n8.dtype, buffer=nl.sbuf)
    for kt in nl.static_range(KT):
        tq = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tq, data=nr[:, kt * 128:(kt + 1) * 128])
        nisa.tensor_copy(dst=xs[:, kt * T:(kt + 1) * T], src=tq)
    amax = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=amax, value=0.0)
    st = 2 if mem == 1 else 1     # mem: the activations stay in SBUF between the phases (no HBM round trips)
    if mem == 1:
        aall = nl.ndarray((T, G, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
    else:
        aall = None
    for gq in nl.static_range(0, G, U):
        _ph1_9(xs, asb, idlg, gslg, cslg, dslg, egu3, amax, acth, 0, min(U, G - gq), 1, st, gq, aall, NR, fast)
    if mem != 1:
        aall = nl.ndarray((T, G, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=aall, src=acth.permute((1, 0, 2)))
    if fast >= 1:   # the row amax of every slot's activations at once (2 reductions per call instead of 5 instructions per group)
        mn = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=amax, op=nl.maximum, data=aall.reshape((T, G * GS * EI)), axis=1)
        nisa.tensor_reduce(dst=mn, op=nl.minimum, data=aall.reshape((T, G * GS * EI)), axis=1)
        nisa.tensor_scalar(dst=mn, data=mn, op0=nl.multiply, operand0=-1.0)
        nisa.tensor_tensor(dst=amax, data1=amax, data2=mn, op=nl.maximum)
    a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a2, data=amax, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=ra, data=a2)
    a8 = nl.ndarray((T, G * GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=a8, data=aall.reshape((T, G * GS * EI)), op0=nl.multiply, operand0=ra)
    axall = nl.ndarray((128, G, 2 * T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    for d in nl.static_range(2 * G):
        tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=a8[:, d * 128:(d + 1) * 128])
        nisa.tensor_copy(dst=axall[:, d // 2, (d % 2) * T:(d % 2 + 1) * T], src=tp)
    if mem != 1:
        nisa.dma_copy(dst=axh.permute((1, 0, 2)), src=axall)
    didx = dfl.reshape((G, 256))
    yo = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    if yd >= 1 and G <= U:
        _ph3(didx, axh, ed, yo, 0, G, 1, st, 0, axall, NR, 0, 1, a2)
    else:
        ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
        if yd < 1:
            nisa.memset(dst=ys, value=0.0)
        for gq in nl.static_range(0, G, U):
            _ph3(didx, axh, ed, ys, 0, min(U, G - gq), 1, st, gq, axall, NR, 1 if (yd >= 1 and gq == 0) else 0)
        nisa.tensor_scalar(dst=yo, data=ys, op0=nl.multiply, operand0=a2)
    nisa.dma_copy(dst=y, src=yo)
    return y


@nki.jit
def moe_dense9(n8, a, csT, egu_s, ed_c, egu_t, ed, NH=1, U=4, stt=1):
    """Prefill (MOE=all) on the EGU_T layout in ONE pass over the experts, for NH x T rows (T <= 128 = the stationary operand):
    slot == expert, so there is no slot table, no gather and no indirect DMA; every group of GS experts is streamed ONCE (gate|up
    slices, down tiles, the stored scales) and contracted with each of the NH row blocks in turn. Nothing goes through HBM
    between the phases: the activations of a super-iteration (U groups) are quantised with THEIR OWN row scale (amax over the
    U * GS experts of the super-iteration, not over all E: a finer fp8 grid than the one-scale-per-row form), transposed and
    contracted with the down tiles of the same super-iteration; y accumulates accd * scale.
    n8 [NH * T, H] fp8, a [NH * T, 1], csT [NH * E, T] f32 (row h * E + e: the combine weights of expert e for the rows of block h)."""
    T = n8.shape[0] // NH
    E = egu_s.shape[0]
    G = E // GS
    egu3 = egu_t.reshape((E, 128, KT * O2))
    y = nl.ndarray((NH * T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    gslg = egu_s.reshape((G, GS * O2))
    dslg = ed_c.reshape((G, GS * EI))
    cslg = csT.reshape((NH * G, GS * T))
    xl = []
    for h in nl.static_range(NH):
        nr = nl.ndarray((T, H), dtype=n8.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nr, src=n8[h * T:(h + 1) * T, :])
        asb = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=asb, src=a[h * T:(h + 1) * T, :])
        xs = nl.ndarray((128, KT * T), dtype=n8.dtype, buffer=nl.sbuf)
        for kt in nl.static_range(KT):
            tq = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tq, data=nr[:, kt * 128:(kt + 1) * 128])
            nisa.tensor_copy(dst=xs[:, kt * T:(kt + 1) * T], src=tq)
        ys = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ys, value=0.0)
        xl.append((xs, asb, ys))
    for gq in nl.static_range(0, G, U):
        un = min(U, G - gq)
        ld = []
        for u in nl.static_range(un):
            g = gq + u
            gs = nl.ndarray((T, GS * O2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gs, src=gslg.ap(pattern=[[0, T], [1, GS * O2]], offset=g * GS * O2))
            dc = nl.ndarray((T, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=dc, src=dslg.ap(pattern=[[0, T], [1, GS * EI]], offset=g * GS * EI))
            ld.append((gs, dc))
        wl = []
        for u in nl.static_range(un):
            g = gq + u
            w = nl.ndarray((128, GS * KT * O2), dtype=egu3.dtype, buffer=nl.sbuf)
            for j in nl.static_range(GS):
                nisa.dma_copy(dst=w[:, j * KT * O2:(j + 1) * KT * O2],
                              src=egu3.ap(pattern=[[KT * O2, 128], [1, KT * O2]], offset=(g * GS + j) * 128 * KT * O2))
            wd = nl.ndarray((128, 2 * H), dtype=ed.dtype, buffer=nl.sbuf)
            for i in nl.static_range(2):
                nisa.dma_copy(dst=wd[:, i * H:(i + 1) * H], src=ed.ap(pattern=[[H, 128], [1, H]], offset=(g * 2 + i) * 128 * H))
            wl.append((w, wd))
        for h in nl.static_range(NH):
            aU = nl.ndarray((T, un, GS * EI), dtype=nl.float32, buffer=nl.sbuf)
            for u in nl.static_range(un):
                cs = nl.ndarray((T, GS), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=cs, src=cslg.ap(pattern=[[1, T], [T, GS]], offset=(h * G + gq + u) * GS * T))
                _gu_compute7(xl[h][0], wl[u][0], xl[h][1], ld[u][0], cs, ld[u][1], None, None, 0, u, 2, 0, aU, 1)
            aUf = aU.reshape((T, un * GS * EI))
            mx = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=aUf, axis=1)
            mn = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mn, op=nl.minimum, data=aUf, axis=1)
            nisa.tensor_scalar(dst=mn, data=mn, op0=nl.multiply, operand0=-1.0)
            nisa.tensor_tensor(dst=mx, data1=mx, data2=mn, op=nl.maximum)
            a2 = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=a2, data=mx, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
            ra = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=ra, data=a2)
            a8 = nl.ndarray((T, un * GS * EI), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=a8, data=aUf, op0=nl.multiply, operand0=ra)
            axs = nl.ndarray((128, 2 * un, T), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            for d in nl.static_range(2 * un):
                tp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=a8[:, d * 128:(d + 1) * 128])
                nisa.tensor_copy(dst=axs[:, d, :], src=tp)
            accd = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.psum)
            for jj in nl.static_range(H // 512):
                for d in nl.static_range(2 * un):
                    nisa.nc_matmul(dst=accd[:, jj * 512:(jj + 1) * 512], stationary=axs[:, d, :],
                                   moving=wl[d // 2][1][:, (d % 2) * H + jj * 512:(d % 2) * H + (jj + 1) * 512], accumulate=(d > 0))
            if stt == 1:
                nisa.scalar_tensor_tensor(dst=xl[h][2], data=accd, op0=nl.multiply, operand0=a2, op1=nl.add, operand1=xl[h][2])
            else:
                sc = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sc, data=accd, op0=nl.multiply, operand0=a2)
                nisa.tensor_tensor(dst=xl[h][2], data1=xl[h][2], data2=sc, op=nl.add)
    for h in nl.static_range(NH):
        nisa.dma_copy(dst=y[h * T:(h + 1) * T, :], src=xl[h][2])
    return y


def row_inputs9(top, w, T, K):
    """glue for moe_row9: the router's top-k (indices [T, K], renormalised weights [T, K]) -> idx [KM, 1] int32, idxf, csT [KM, T]."""
    import torch
    KM = -(-T * K // GS) * GS
    dev = top.device
    idxf = top.reshape(T * K, 1).float()
    cs = w.reshape(T * K, 1) * torch.eye(T, device=dev).repeat_interleave(K, dim=0)
    if KM > T * K:
        idxf = torch.cat((idxf, torch.zeros(KM - T * K, 1, device=dev)), dim=0)
        cs = torch.cat((cs, torch.zeros(KM - T * K, T, device=dev)), dim=0)
    return idxf.to(torch.int32), idxf, cs


def graph_inputs9(n):
    """glue for moe_dyn9: only the activation quantisation (as qwen35_graphs.quant8)."""
    import torch
    a = torch.amax(torch.abs(n.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    return (n.float() / a).to(torch.float8_e4m3fn), a


def distinct_slots(comb, kmax):
    """torch: comb [T, E] (0 = not selected) -> (idx [kmax] ascending distinct experts first, padded with unused experts;
    n [] number of distinct experts). The union form's own ordering (qwen35_graphs.moe_routed)."""
    import torch
    E = comb.shape[1]
    used = (comb > 0).float().amax(dim=0, keepdim=True)                                   # [1, E]
    ar = torch.arange(E, device=comb.device).float().reshape(1, E)
    idx = torch.topk(used - ar * (0.5 / E), kmax, dim=-1)[1].reshape(kmax)
    return idx, used.sum()


def graph_inputs(n, comb, egu_s, ed_c, kmax, spare=0, U=0, levels=None):
    """torch (in the traced graph): n [T, H] f32 (bf16 values), comb [T, E] f32 -> kernel inputs (xT, ga, cw, gidx, didx, cnt)."""
    import torch
    T, dev = n.shape[0], n.device
    kmax = kmax + spare * GS
    G = kmax // GS
    idx, nd = distinct_slots(comb, kmax)
    a = torch.amax(torch.abs(n.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    n8 = (n.float() / a).to(torch.float8_e4m3fn)                                           # [T, H]
    xT = n8.T.reshape(KT, 128, T).permute(1, 0, 2).reshape(128, KT * T)
    ga = (a * torch.index_select(egu_s, 0, idx).reshape(1, kmax * O2)).reshape(T, G, GS * O2).permute(1, 0, 2).contiguous()
    cs = torch.index_select(comb, 1, idx)                                                  # [T, kmax]
    cw = (cs.unsqueeze(-1) * torch.index_select(ed_c, 0, idx).reshape(1, kmax, EI)).reshape(T, G, GS * EI).permute(1, 0, 2).contiguous()
    gidx = (idx.reshape(G, 1, GS) * 128 + torch.arange(128, device=dev).reshape(1, 128, 1)).to(torch.int32)
    didx = (idx.reshape(G, 2, 4, 1) * EI + torch.arange(EI, device=dev).reshape(1, 1, 1, EI)).reshape(G, 2, 128).permute(0, 2, 1).contiguous().to(torch.int32)
    ngf = torch.floor((nd + (GS - 1)) / GS)
    cnt = ngf.to(torch.int32).reshape(1, 1)
    if levels:
        b1 = levels[0] * torch.floor(ngf / levels[0])
        b2 = b1 + levels[1] * torch.floor((ngf - b1) / levels[1])
        cnt = torch.stack([b1, b2, ngf]).to(torch.int32).reshape(1, 3)
    if U:
        cnt = torch.stack([ngf, U * torch.floor(ngf / U)]).to(torch.int32).reshape(1, 2)
    if spare:
        cnt = torch.stack([ngf, 2 * torch.floor((ngf + 1) / 2)]).to(torch.int32).reshape(1, 2)
    return xT, ga, cw, gidx, didx, cnt


def mirror(n, comb, egu, egu_s, ed, ed_c):
    """exact f32 CPU mirror of the kernel (and of moe_routed's union form) with numpy-free torch ops."""
    import torch
    T = n.shape[0]
    a = torch.amax(torch.abs(n.float()), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
    n8 = (n.float() / a).to(torch.float8_e4m3fn).float()
    used = torch.nonzero((comb > 0).any(dim=0)).reshape(-1)
    S = used.numel()
    wg = egu.float()[used].reshape(S * 2 * EI, H)
    hgu = ((n8 @ wg.T) * a * egu_s[used].reshape(1, -1)).to(torch.bfloat16).float().reshape(T, S, 2, EI)
    act = (torch.nn.functional.silu(hgu[:, :, 0]) * hgu[:, :, 1]).to(torch.bfloat16).float()
    act = (act * comb[:, used].unsqueeze(-1) * ed_c[used].reshape(1, S, EI)).reshape(T, S * EI)
    a2 = (torch.amax(torch.abs(act), dim=-1, keepdim=True) / 224.0).clamp(min=1e-20)
    return ((act / a2).to(torch.float8_e4m3fn).float() @ ed.float()[used].reshape(S * EI, H)) * a2
