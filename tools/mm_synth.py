#!/usr/bin/env python3
"""Parametric PE0.bin / Pool0.bin synthesizer for y[M,N] = x[M,K] @ w[K,N].

STATUS (2026-11-09): phase/ring model decoded from all 18 oracle buckets
(tools/shape_templates.json); PSUM dst map still unresolved - see
docs/neff-format.md "PE phase model" for the full extracted structure and
the open questions. This module is a WIP skeleton, NOT wired into lift_hip.

Verified structural facts (all buckets M{32,64,96,128} K{128..1024} N{128..640}):

Slots: 32B; every LW/MM is followed by an "op10" 32B slot:
  op10 byte0=10, byte1=b1 mode (1=multi-start, 2=cont, 3=single),
  u16@6=32800 (A/B phase) or 8320 (B-phase when MM A in low scratch),
  u16@10=3855 const, byte11=ring select (0/1/2),
  for MM: bytes 16..19 = PSUM dst u32, u16@20=512 step, u16@22=1, u16@30=1.
LW: src @16 u32 (ring slot END - 4), step u16@20 = -1, u16@26 = num
  (A phase: 128; B phase: M), ev @4..8 (0x805 first A pair, 0x405 first B
  pair, else 0).
MM: A operand @16 u32 (identity tile / B scratch tile), ev: upd 19/5 with
  0x5130 @8 on the LAST pair of each accumulation group.
EV(160): ev=0x1805 before A, ev=0x905 before B.

Address map (SB, top-down):
  identity tile end   = 0x30000, A_addr = 0x30000 - 4*M
  x ring (A phase)    = AS = K/128 slots of 512B below identity:
                        slot s ends at A_addr - 512*(AS-1-s); LW src = end-4
  w ring (B phase)    = AS slots of 512B below x ring (same formula)
  B scratch tiles     = K<=128: staged once at 0x2f980-ish (per bucket);
                        K>=256: 0x4000 + 4*M*t ring (streamed by pool desc2)
Phases:
  A: 2*AS pairs: LW x-ring[s] (b1 1 at even p, else 2), MM identity, dst PSUM-A
  B: 2*AS pairs per N-half (for NT=N/128 halves), LW w-ring, MM scratch tile,
     dst PSUM accumulates (const within a half)
  PV: iff K>=512 (M32/M128?) or N>=256: combine B halves -> final PSUM.
PSUM dst map: NOT yet solved. Observed A bases: 0x0/0x800/0x1000 (varies with
M, AS, NT in a way that does not fit a simple formula yet); B dst 0x0 (N128)
or {0x1000, 0x800} (N256 halves); PV dst {0x800} / {0x1800, 0xa00}.
Resolving it requires a device perturbation run (store y, dump PSUM window).
"""
