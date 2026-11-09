#!/usr/bin/env python3
"""Parametric PE0.bin synthesizer for y[M,N] = x[M,K] @ w[K,N] (trn1 v2 ISA).

STATUS 2026-11-09: dataflow + addressing FULLY decoded (see bottom); the
generator itself is NOT implemented yet - implement it against
tools/mm_oracle_pairs.json (distilled per-pair ground truth for all 18
oracle buckets) and verify byte-exact with mm_oracle_pairs before any
device run.

## Completed model (all 18 buckets consistent)

Shapes: M<=128 (partition = row), K%128==0 (AS=K/128), N%128==0 (NT=ceil(N/128)).

SB layout (top-down):
  identity slice : SB [to_off, to_off+4M or 512B); to_off = 0x30000-4M,
                   EXCEPT N>=512 -> 0x2fe00, and (N>=256 and M>=64) -> 0x2fe00.
                   Staged by SP dma (sem 24, qSPSpillReload0), sizes [128,128]
                   (small) / [512,128] (big identity), to_steps [1, 262144]
                   (per-partition stride 256KB).
  x ring         : AS slots x 512B ending at to_off:
                   slot s ends at to_off - 512*(AS-1-s); LW src = end-4,
                   num=128. Streamed by pool desc1 (sem 8).
  w tiles        : family-dependent (see below).
PSUM dsts (drain-coupled! CANNOT be chosen freely - probes proved that
changing B/PV dsts breaks y; the runtime [204] patch reads exactly the
compiler-chosen finals):
  K==128 (col-mode):   final tiles @ PSUM|4M*t (bank0), dn = N if N<=256
                       else 128 (N512: 4 tiles dn128, N256: 1 tile dn256).
  K>=256, M>=96:       final @ PSUM|0 dn=N (single, direct accumulate).
  K>=256, M<=64:       final @ PSUM|0x1800 + 0x200*t (bank3!), dn=128/NT.
A temps: 4M-stride; base bank = {0x0 | 0x800 | 0x1000} per
(M,Kclass,NT) - empirically free to choose? NOT probed yet: only final/B
dst patches were probed (they break y). Probe A-dst freedom before relying
on it.
Dataflow: A phase: LW x-chunk -> weight regs, MM identity x regs -> PSUM
x-temps. B phase: LW x-temps (via SB window 0x4000+ = PSUM local alias for
some families) or x-ring direct (N>=512), MM w-tile(SB) x regs -> y.
PV phase (M<=64, K>=512 or N>=256&K>=256): identity copy of combined
tiles -> final bank3 tiles.

Family table (B_scrA = MM A operand of B phase; B LW srcs):
  K128 N<=256, M32: B_scrA=0x2f980 (SB w tile); B LW = 0x4000+4M-4 (window)
  K128 N<=256, M64: B_scrA=0x2fb00 (N128) / window; B LW = 0x4000+4M-4
  K128 N128, M96/128: B_scrA=0x2fa80 / 0x2fa00; B LW = 0x4000+4M-4
  K128 N512: B_scrA = 0x4000 const (window, all tiles); B LW = x-ring direct
  K256 N<=128: B_scrA = SB w-ring slots (0x2f780/0x2f980, 512B stride);
               B LW = window 0x4000+4M-4
  K512+ N<=128, M32: B_scrA = 0x4000+4M*s (window tiles per chunk);
               B LW = w-ring below x-ring
  K512+ N<=128, M64: B_scrA = SB 0x2ef00+N/4*s?? (0x2ef00+0x200*s);
               B LW = window 0x4000+N/4... (64_512_128: LW 0x40fc/0x41fc...)
  K>=256 N>=256, M<=64: B_scrA = 0x4000+4M*s (window); B LW = w-ring below
               x-ring (0x2f* region)
  K>=256 N>=256, M96: B_scrA = SB ring 0x2e680+0x400*s (0x2e580+0x300*s for
               N192: stride = N*4!); B LW = window 0x4000+(t+1)*?...
  PV phase LW: SB-window reads of B temps: 32: 0x427c/0x42fc (+0x80 per
               tile), 64: 0x40fc/0x41fc (64_128_256) or 0x42fc/0x43fc
               (64_256_256) / 0x44fc/0x45fc (64_512_256).

op10 slots after LW/MM: byte0=10, byte1: LW 1 at even pair / 2 odd; MM 2 in
A+PV phases, 3 in B phase. u16@6 = 32800 (A/B phases; 8320 when B MM A
operand < 0x10000 i.e. window-const), u16@10 = 3855, byte11 (ring): LW=0;
MM: A-phase 1,2,1,2 per pair; B-phase first pair 1, middle 0, LAST pair 2
(per accumulate group); PV: 1,0 (last group 1,2).
MM dst pattern in op10 slot: dst u32 @16, step u16@20 = 512, u16@22 = 1,
u16@26 = dn (tile cols), u16@28 = tick (A: 0?? see pairs json), u16@30 = 1.

Pool descs (qPoolDynamic, sems 8/9/10): desc1 x-ring (AS*512B), desc2 w
(AS*N/4 rows...), desc3 y-out (src = SB 0x4280+ drain window, dst = HBM
var2, waits 5/4 GE phases). Formats = v2.NEURON_ISA_TPB_PSEUDO_DMA_DIRECT2D
(tools/neff_isa.py).

## Remaining work
1. Implement generator from mm_oracle_pairs.json; assert byte-exact vs all
   18 oracle streams (local only).
2. Probe A-temp dst freedom (patch A dst to a free bank; if y survives,
   temps are synthesizer-chosen).
3. Pool desc synthesis (isa structs; elem sizes/steps from sp_dma patterns).
4. Device matrix: new shapes outside the 18 buckets (K384, N384/N640, M48).
"""
