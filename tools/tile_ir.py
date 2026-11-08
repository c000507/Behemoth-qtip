#!/usr/bin/env python3
"""Tile-IR for Neuron NEFF emission (v1).

Ops map to engine streams per docs/neff-codegen.md. Graph emission produces
the tar members (def.json, engine .bin streams, DMA JSON) plus container via
tools/neff_emit.py.

Status: matmul path complete (LDWEIGHTS/MATMUL/events encoders proven
byte-identical vs neuronx-cc output); elementwise/activate plumbed; DVE
lowering pending v2 DVE struct decode.
"""
from __future__ import annotations
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import neff_isa as isa  # noqa: E402

# ---------------------------------------------------------------- IR

@dataclass
class Tensor:
    name: str
    size: int                      # bytes
    kind: str                      # input | output | file | sb
    alignment: int = 65536


@dataclass
class DmaCopy:
    dst: str; src: str
    from_off: int = 0
    from_sizes: tuple = (); from_steps: tuple = ()
    to_off: int = 0
    to_sizes: tuple = (); to_steps: tuple = ()
    queue: str = "qSPSpillReload0"
    semaphore: int | None = None
    dtype: str = "float32"


@dataclass
class Events:
    wait_mode: int = 0
    wait_idx: int = 0
    update_mode: int = 0
    update_idx: int = 0
    semaphore_value: int = 0

    def pack(self) -> int:
        return (self.wait_mode | (self.wait_idx << 8) | (self.update_mode << 16)
                | (self.update_idx << 24))

    @classmethod
    def unpack(cls, raw: int) -> "Events":
        return cls(raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF, (raw >> 24) & 0xFF, 0)


@dataclass
class LdWeights:
    addr: int                      # SB address of weight tile
    events: Events = field(default_factory=Events)
    rows: int = 128
    cols: int = 128


@dataclass
class Matmul:
    src_addr: int                  # SB offset of A tile
    dst_addr: int                  # PSUM/SB offset
    rows: int = 128
    cols: int = 128
    in_dtype: int = 10             # FP32
    row_grp: int = 15
    col_grp: int = 15
    events: Events = field(default_factory=Events)
    accum: int = 0                 # 1 = zero-dest, 2 = accumulate


@dataclass
class SemWait:
    sem: int; mode: int = 1; value: int = 0


@dataclass
class SemUpdate:
    sem: int; mode: int = 21; value: int = 0   # SEM_ADD_IMM_COMPLETE


PROLOGUE = bytes([177, 16, 0, 0]) + bytes(60)     # BRANCH_PREFETCH_HINT, len16


# --------------------------------------------------------- PE stream emit

# mem-pattern element: {u24 addr + u8 space marker @+0, step@+4, num@+10,
#                       flags@+12,@+14}; space: 0=SB, 2=PSUM
SB_SPACE = 0
PSUM_SPACE = 2


def _pat(addr: int, space: int, step: int = 1, num: int = 128,
         f1: int = 1, f2: int = 1) -> bytes:
    return (struct.pack("<I", (addr & 0xFFFFFF) | (space << 24))
            + struct.pack("<HHHH", step & 0xFFFF, 0, 0, num)
            + struct.pack("<HH", f1, f2))


def enc_mm(src: int, dst: int, rows: int, cols: int, dt: int, ev: Events,
           row_grp=15, col_grp=15, src_space=SB_SPACE, dst_space=PSUM_SPACE,
           accum: int = 0, fp32_mode: int = 2, xbus_sel: int = 16) -> bytes:
    vals = {
        "in_dtype": dt,
        "num_active_rows": rows,
        "num_active_cols": cols,
        "row_grp": row_grp,
        "col_grp": col_grp,
        "xbus_sel": xbus_sel,
        "psum_accumulate_flags": accum,
        "fp32_mode": fp32_mode,
        "events": ev.pack(),
        "events_semaphore_value": ev.semaphore_value,
    }
    buf = bytearray(isa.encode("NEURON_ISA_TPB_S3D3_MM", vals, 2, 16))
    buf[16:32] = _pat(src, src_space)
    buf[48:64] = _pat(dst, dst_space)
    return bytes(buf)


def enc_lw(var_addr: int, ev: Events = None, rows: int = 128, cols: int = 128,
           dt: int = 10, fp32_mode: int = 2, xbus_sel: int = 2,
           src_space: int = SB_SPACE) -> bytes:
    vals = {
        "in_dtype": dt,
        "num_active_rows": rows,
        "num_active_cols": cols,
        "row_grp": 15,
        "col_grp": 15,
        "xbus_sel": xbus_sel,
        "fp32_mode": fp32_mode,
    }
    if ev:
        vals["events"] = ev.pack()
        vals["events_semaphore_value"] = ev.semaphore_value
    buf = bytearray(isa.encode("NEURON_ISA_TPB_S3_LW", vals, 1, 16))
    buf[16:32] = _pat(var_addr, src_space)
    return bytes(buf)


def enc_semaphore(update_mode: int, update_idx: int, value: int,
                  wait_mode: int = 0, wait_idx: int = 0) -> bytes:
    """EVENT_SEMAPHORE per libnrt add_semaphore_* builders:
    byte0=160 len=16; @0x20=0; @0x22=update_mode; @0x23=update_idx;
    @0x28=u32 value; wait block @0x04 (mode/idx) when waiting."""
    b = bytearray(64)
    b[0] = 160; b[1] = 16
    b[4] = wait_mode; b[5] = wait_idx
    b[0x22] = update_mode; b[0x23] = update_idx
    struct.pack_into("<I", b, 0x28, value)
    return bytes(b)


def emit_pe_stream(ops: list) -> bytes:
    # reference layout: [prologue 177][PSEUDO_TENSOR_STORE 204 marker][compute...]
    out = bytearray(PROLOGUE)
    out += op204()
    for op in ops:
        if isinstance(op, LdWeights):
            out += enc_lw(op.addr, ev=op.events, rows=op.rows, cols=op.cols)
        elif isinstance(op, Matmul):
            out += enc_mm(op.src_addr, op.dst_addr, op.rows, op.cols,
                          op.in_dtype, op.events, op.row_grp, op.col_grp,
                          accum=op.accum)
        elif isinstance(op, SemUpdate):
            out += enc_semaphore(op.mode, op.sem, op.value)
        elif isinstance(op, SemWait):
            out += enc_semaphore(0, 0, 0, op.mode, op.sem)
        else:
            raise TypeError(op)
    return bytes(out)


def enc_dmatrigger(queue_name: str) -> bytes:
    """PSEUDO_DMATRIGGER (op 193): queue name as ASCII at +0x0C (TDRV reads there)."""
    b = bytearray(64)
    b[0] = 193; b[1] = 16
    nm = queue_name.encode() + b"\0"
    b[0x0C:0x0C + len(nm)] = nm
    return bytes(b)


def op204() -> bytes:
    """PSEUDO_TENSOR_STORE placeholder: header only (runtime patches it)."""
    b = bytearray(64)
    b[0] = 204; b[1] = 16
    return bytes(b)


def emit_sp_stream(triggers: list[str]) -> bytes:
    out = bytearray(PROLOGUE)
    out += op204()
    for q in triggers:
        out += enc_dmatrigger(q)
    return bytes(out)


# --------------------------------------------------------- graph -> NEFF

def sb_layout(tensors: list[Tensor]) -> dict:
    """Assign SB offsets; STATE_BUF_PARTITION_SIZE = 256KB (from ISA consts)."""
    off = 0
    layout = {}
    for t in tensors:
        layout[t.name] = off
        off += (t.size + 63) & ~63
    return layout, off


def build_def_json(name: str, tensors: list[Tensor], consts: dict,
                   act_sets: list[str], stream_files: dict) -> dict:
    var = {"SB": {"type": "state-buffer", "var_id": 0}}
    vid = 1
    for t in tensors:
        e = {"ops": [], "size": t.size, "type": t.kind, "var_id": vid}
        if t.kind == "file":
            e["alignment"] = t.alignment
            e["fabric_path"] = "main"
            e["file_name"] = consts.get(t.name, t.name + ".npy")
        var[t.name] = e
        vid += 1
    var["DGETable0"] = {"list": [2, 1, 3, 4], "type": "dge-table", "var_id": vid}
    return {
        "#rank_world_size": 0,
        "act": "Activation0.json",
        "act_asm_dbg": [],
        "act_dbg": [],
        "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0},
        "debug_info": {},
        "dma_dbg": [],
        "dma_queue": {
            "qPoolDynamic": {
                "dynamic_dma": {"SB_scratch_partition_offset": 0,
                                 "SB_scratch_partition_size": 16384},
                "fabric_path": "main", "num_queues": 16, "owner": "pool",
                "semaphore_set": list(range(8, 24)), "type": "dynamic"},
            "qPoolIO0": {
                "fabric_path": "main", "num_queues": 16, "owner": "pool",
                "semaphore_set": list(range(56, 72)), "type": "in"},
            "qSPIO0": {
                "fabric_path": "main", "num_queues": 16, "owner": "sp",
                "semaphore_set": list(range(40, 56)), "type": "in"},
            "qSPSpillReload0": {
                "fabric_path": "main", "num_queues": 16, "owner": "sp",
                "semaphore_set": list(range(24, 40)), "type": "data"},
        },
        "dve": "DVE0.json", "dve_asm_dbg": [], "dve_dbg": [], "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                   "fp8_range_extension": True, "fp8_sis": False,
                                   "fp8_sns": False, "ocp_sat": True},
        "name": "definition",
        "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": [], "pe_dbg": [], "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": [], "pool_dbg": [], "pool_instr": "Pool0.bin",
        "runtime_event_count": 12,
        "runtime_semaphore_count": 3,
        "sp": "SP0.json", "sp_asm_dbg": [], "sp_dbg": [], "sp_instr": "SP0.bin",
        "var": var,
        "version": "0.6",
    }



@dataclass
class Plan:
    tensors: list
    dma: list
    pe_ops: list
    notes: list

# ------------------------------------------------------------ demo graph

def demo_matmul_graph(M=4096, K=4096):
    """x[K] @ W -> y[M] tiled as 128x128 PE tiles, one weight group."""
    tiles_m = M // 128
    ops = []
    ev_none = Events()
    ev_inc = Events(update_mode=19, update_idx=5)   # SEM_ADD_IMM_COMPLETE
    for m in range(tiles_m):
        ops.append(LdWeights("w"))
        ops.append(Matmul(src_addr=0x2fe00, dst_addr=0x2000000 + m * 0x800,
                          events=(ev_inc if m % 2 else ev_none)))
    return ops


if __name__ == "__main__":
    import neff_emit
    ops = demo_matmul_graph()
    stream = emit_pe_stream(ops)
    print(f"PE stream: {len(stream)} bytes, {len(ops)} ops")
    # structural sanity: first bytes = prologue, op2 count
    assert stream[:4] == PROLOGUE[:4]
    n2 = sum(1 for i in range(0, len(stream), 64) if stream[i] == 2)
    print(f"MATMUL instrs: {n2}")
