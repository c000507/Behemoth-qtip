#!/usr/bin/env python3
"""Own synthesis of DVE BITVEC chains (y = elementwise integer logic).

Field layouts DECODED from compiler oracles (jax gens on the trn1 box,
bit-exact validated on device 2026-11-10):

  op81 TENSOR_TENSOR_BITVEC = 64B (head 32B + data-tail 32B):
    head: [op 0x51, wl 0x10, dbg, wait u16@4 = (idx<<8)|mode, upd u16@6 =
           (idx<<8)|mode, 0x10@8, 0@10, src0-addr u16@12, ALU u16@14
           (0x800A=AND 0x800B=OR 0x800C=XOR; 0x8006=MUL on the arith twin
           op65), 0xFFF0@16, 2@18, 1,1,1@20-24, P@26, 1,1@28]
    tail: [-8P@32, 2@34, 1,1,1, P@42, 1,1, dst-addr u16@48, 0@50,
           1,1,1, P@58, 1,1]
    src1 (the second tensor operand) = NOT in the instr: TT = SB-pattern
    x B-registers, the B operand staged by pool desc1 (sem 9) - the same
    producer-consumer shape as the PE matmul.

  op83 TENSOR_SCALAR_BITVEC = 32B + op8 carrier 32B (one 64B fetch pair):
    op83:  [op 0x53, wl 0x10, dbg, wait u16@4, upd u16@6, 0x10@8, 0@10,
            src0-addr u16@12, 0@14, 0xFFF0@16, 2@18, 1,1,1, P@26, 1,1]
    op8:   [op 0x08, wl 0x10, dbg, 0x0808@4, 128@6, MODE u16@8 (3=SHR,
            10=AND, 16=SHL), 0, IMM u16@12 (shift/mask), 0,0,0,
            dst-addr u16@16, 0, 1,1,1, P@26, 1,1]

  P = n/128 (the DVE internal 128-lane pass count); P-fields = @26 (head)
  and dst-pattern @26-of-32B-tail for op81 (= @42 of the 64B view).

CRITICAL PATCH RULES (learned on device, deadlock probes):
  - upd u16 = (idx<<8)|mode - writing (mode<<8)|idx hangs the engine.
  - upd = 0 on the LAST DVE op hangs the y-out drain (the pool desc2
    chains off the DVE completion notify): every container needs exactly
    one final (19, 6) update. Intermediate ops = upd 0.
  - Pool0 P-fields (int32 = 4B/elem, byte-identical to the f32 ew pool
    except marker @62/desc = 0x0808): desc0 rel{24,32,36,40,56,60} =
    4P,4P,4P,-4P,4P,4P; desc1 rel{24,32,36,40=-8P,56,60}; desc2
    rel{36,48,60} = 4P. At P=128+ the desc2 rel48 4P value fails
    translate ("dst pseudo addr8") - larger-n staging geometry pending.
  - ACT stream (ew-A scale=1.0) = bit-exact int passthrough: f32 mul by
    1.0 preserves any finite f32 bit pattern.

MEASURED (trn1, n=4096 resident, R rounds of [shr1->y][shl1->y], P=32):
  R=16: 59.3 us avg | R=64: 66.5 us | R=256: 93.8 us
  slope = 143.75 ns/round = 8192 elem-ops -> DVE TS-BITVEC rate =
  57 G elem-ops/s (engine-bound, data resident). Consistent with the
  compiler 4-op TT chain at n=1M (55 Gop/s marginal).
  LATENCY-PER-INSTR hypothesis (36 ns per 32B slot, rate = n/72ns) is
  UNTESTED - the P=128 test is blocked on the pool desc tiling decode.
"""
import struct
from pathlib import Path

import ew_synth


def _u16(buf, off, val):
    struct.pack_into("<H", buf, off, val & 0xFFFF)


TT_ALU = {"and": 0x800A, "or": 0x800B, "xor": 0x800C, "mul": 0x8006}
TS_MODE = {"shr": 3, "and": 10, "shl": 16}
UPD_LAST = (6 << 8) | 19  # byte6 = mode 19 (SEM_ADD_IMM_COMPLETE), byte7 = idx 6


def tt_bitvec(op, src0, dst, P):
    """64B op81: SB src0 x B-regs -> dst. op in TT_ALU."""
    s = bytearray(64)
    s[0] = 81
    s[1] = 16
    _u16(s, 4, (8 << 8) | 5)     # wait sem idx 8, mode 5 (GE)
    _u16(s, 6, 0)                # upd: caller patches the final op
    _u16(s, 8, 16)
    _u16(s, 10, 0)
    _u16(s, 12, src0)
    _u16(s, 14, TT_ALU[op])
    _u16(s, 16, 0xFFF0)
    _u16(s, 18, 2)
    _u16(s, 20, 1); _u16(s, 22, 1); _u16(s, 24, 1)
    _u16(s, 26, P)
    _u16(s, 28, 1); _u16(s, 30, 1)
    _u16(s, 32, -8 * P)
    _u16(s, 34, 2)
    _u16(s, 36, 1); _u16(s, 38, 1); _u16(s, 40, 1)
    _u16(s, 42, P)
    _u16(s, 44, 1); _u16(s, 46, 1)
    _u16(s, 48, dst)
    _u16(s, 50, 0)
    _u16(s, 52, 1); _u16(s, 54, 1); _u16(s, 56, 1)
    _u16(s, 58, P)
    _u16(s, 60, 1); _u16(s, 62, 1)
    return bytes(s)


def ts_bitvec(op, src0, imm, dst, P):
    """64B op83+op8 pair: SB src0 <op> imm -> dst. op in TS_MODE.

    op83 32B head (src0 pattern) + op8 32B carrier (scalar op + dst).
    op8 32B layout (from the shl/shr/mod16 oracles):
      [0x0808@0 (hdr marker incl op=8/wl=8), 128@2, MODE u16@4, 0@6,
       IMM u16@8, 0@10, 0@12, 0@14, dst-addr u16@16, 0@18,
       1@20, 1@22, 1@24, P u16@26, 1@28, 1@30]
    """
    s = bytearray(32)
    s[0] = 83
    s[1] = 16
    _u16(s, 4, (8 << 8) | 5)
    _u16(s, 6, 0)
    _u16(s, 8, 16)
    _u16(s, 10, 0)
    _u16(s, 12, src0)
    _u16(s, 14, 0)
    _u16(s, 16, 0xFFF0)
    _u16(s, 18, 2)
    _u16(s, 20, 1); _u16(s, 22, 1); _u16(s, 24, 1)
    _u16(s, 26, P)
    _u16(s, 28, 1); _u16(s, 30, 1)
    c = bytearray(32)
    _u16(c, 0, 0x0808)
    _u16(c, 2, 128)
    _u16(c, 4, TS_MODE[op])
    _u16(c, 6, 0)
    _u16(c, 8, imm)
    _u16(c, 10, 0); _u16(c, 12, 0); _u16(c, 14, 0)
    _u16(c, 16, dst)
    _u16(c, 18, 0)
    _u16(c, 20, 1); _u16(c, 22, 1); _u16(c, 24, 1)
    _u16(c, 26, P)
    _u16(c, 28, 1); _u16(c, 30, 1)
    return bytes(s) + bytes(c)


def scale_pool_4P(pool: bytes, P: int) -> bytes:
    """Scale the 9 ew-formula pool fields from a P=32 template."""
    p = bytearray(pool)
    rels = [(0, {24: 4, 32: 4, 36: 4, 40: -4, 56: 4, 60: 4}),
            (1, {24: 4, 32: 4, 36: 4, 40: -8, 56: 4, 60: 4}),
            (2, {36: 4, 48: 4, 60: 4})]
    for k, m in rels:
        for rel, mult in m.items():
            _u16(p, 128 + 64 * k + rel, mult * P)
    return bytes(p)


def chain_streams(and_pool: bytes, base_dve: bytes, chain, n: int, P: int):
    """Assemble DVE0.bin for a chain of (op, src, imm_or_dst, dst) tuples.

    chain = list of dicts: {"kind": "ts", "op": "shr", "src": addr,
    "imm": v, "dst": addr} or {"kind": "tt", "op": "and", "src": addr,
    "dst": addr}. The LAST op carries the (19,6) completion update.
    """
    dve = bytearray(base_dve[:6 * 32])  # prologue [177][0][204][0][op160][0]
    for i, step in enumerate(chain):
        upd = i == len(chain) - 1
        if step["kind"] == "ts":
            dve += ts_bitvec(step["op"], step["src"], step["imm"], step["dst"], P)
        else:
            dve += tt_bitvec(step["op"], step["src"], step["dst"], P)
    # patch the last op's upd (head 32B of the final 64B instr)
    _u16(dve, len(dve) - 128 + 6, UPD_LAST)
    return bytes(dve)


VALIDATED = {
    "ts_shift_chain": "n=4096 P=32 R in {16,64,256}: executes, linear "
                      "timing, rate 57 Gop/s; numerics FAIL vs numpy "
                      "(bit-level mismatch TBD - numpy >> vs DVE ashr on "
                      "the y<<1 wrap path) - timing valid, numerics open",
    "tt_chain": "P>32 blocked on pool desc tiling decode "
                "(translate: dst pseudo addr8 at desc2 rel48)",
}
