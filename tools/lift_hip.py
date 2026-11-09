#!/usr/bin/env python3
"""Lift HIP kernel templates (runtime/src/*.hip subset) to tile-IR.

v0 lifters (pattern-anchored per kernel, per TRANSLATION.md scope):
- tiled_projection (projection.hip): K-loop tile GEMM, WMMA 16x16x32 ->
  LDWEIGHTS/MATMUL 128x128 PE tiles + semaphore chain.
- norm_kernel (ops.hip): RMSNorm -> activation(sq) + reduce + scale plan
  (DVE reduce pending; emitted as a documented plan, not streams).

Emit: graph dir (def.json + engine bins + DMA JSON) via tile_ir + neff_emit.
"""
from __future__ import annotations
import json
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import base64
import gzip
import re

sys.path.insert(0, str(Path(__file__).parent))
import tile_ir as ir  # noqa: E402
import neff_isa as isa  # noqa: E402

SB = {"x": 179712, "w": 212224, "psum": 0x2000000}  # IO window 179712..262143


@dataclass
class Plan:
    tensors: list
    dma: list          # DMA JSON descriptors (engine SP0.json "dma")
    pe_ops: list       # tile-ir ops for PE stream
    notes: list


def lift_tiled_projection_fast(rows: int, in_features: int, out_features: int,
                               outdir: Path, name: str = "tiled_projection") -> Path:
    """Emit y[rows,out] = x[rows,in] @ W[in,out] via the verified per-shape
    PE template set (see tools/shape_templates.json + docs/neff-format.md).
    Supported buckets: M in {32,64}, K in {128,256,512}, N in {128,256}.
    Raises ValueError for shapes outside the verified set."""
    return emit_matmul(rows, in_features, out_features, outdir, name)


def lift_tiled_projection(rows: int, in_features: int, out_features: int,
                          ktile: int = 128, sem_base: int = 40) -> Plan:
    """projection.hip tiled_projection: y[rows,out] = x[rows,in] @ W[in,out].

    Neuron mapping (single m-tile, k-tile loop):
      - DMA stages x and W tiles into SB via qSPIO0 (sem 40/41 fire on arrival)
      - LDWEIGHTS pulls the W tile from SB into PE weight regs (waits sem>=2)
      - MATMUL computes 128x128 tile into PSUM
      - PSEUDO_TENSOR_STORE (op 204, runtime-patched) drains PSUM to y
    """
    k_tiles = in_features // ktile
    tensors = [
        ir.Tensor("x", rows * in_features * 4, "input"),
        ir.Tensor("w", in_features * out_features * 4, "input"),
        ir.Tensor("y", rows * out_features * 4, "output"),
    ]
    dma = [
        {"id": 0, "queue": "qSPIO0", "semaphore": 40,
         "desc": [{"op": "copy", "from": "x", "from_dtype": "float32",
                   "from_off": 0, "from_sizes": [rows * in_features, 1],
                   "from_steps": [1, 1],
                   "to": "SB", "to_dtype": "float32", "to_off": SB["x"],
                   "to_sizes": [rows * in_features, 1], "to_steps": [1, 1],
                   "num_tiling_dimensions": 1}],
         "#comment": "IO staging: x -> SB"},
        {"id": 1, "queue": "qSPIO0", "semaphore": 41,
         "desc": [{"op": "copy", "from": "w", "from_dtype": "float32",
                   "from_off": 0, "from_sizes": [in_features * out_features, 1],
                   "from_steps": [1, 1],
                   "to": "SB", "to_dtype": "float32", "to_off": SB["w"],
                   "to_sizes": [in_features * out_features, 1], "to_steps": [1, 1],
                   "num_tiling_dimensions": 1}],
         "#comment": "IO staging: w -> SB"},
    ]
    pe_ops = []
    notes = []
    for kt in range(k_tiles):
        pe_ops.append(ir.LdWeights(SB["w"], events=ir.Events(
            wait_mode=5, wait_idx=40, semaphore_value=2)))  # WAIT sem40 >= 2 (both DMAs)
        pe_ops.append(ir.Matmul(
            src_addr=SB["x"] + kt * ktile * 4,
            dst_addr=SB["psum"],
            rows=128, cols=128, in_dtype=10,
            events=ir.Events(update_mode=19, update_idx=5)))  # SEM_ADD_IMM_COMPLETE sem5
        notes.append(f"k-tile {kt}: LW(w@{SB['w']:#x}) MM(x@{SB['x']:#x}->PSUM)")
    return Plan(tensors=tensors, dma=dma, pe_ops=pe_ops, notes=notes)


def lift_norm(width: int, eps: float = 1e-5) -> Plan:
    """ops.hip norm_kernel (RMSNorm): sum(x*x)/width -> rsqrt -> x*f*w.

    Neuron mapping: activation-engine square via exp_and_others? no - square is
    its own func_id 30 (see exp_and_others.json act table); reduce = DVE
    cross-lane (op 124/125, layouts pending) - emitted as plan only.
    """
    tensors = [
        ir.Tensor("x", width * 2, "input"),
        ir.Tensor("w", width * 2, "input"),
        ir.Tensor("y", width * 2, "output"),
    ]
    return Plan(
        tensors=tensors, dma=[],
        pe_ops=[],
        notes=[
            "RMSNorm plan: DMA x tile -> SB; ACTIVATE func square_1p (func_id 30);",
            "DVE cross-lane reduce add (op 124) sum -> SP; SP computes rsqrt + scale;",
            "blocked on DVE v2 struct decode (docs/neff-isa.md open item).",
        ])


def emit_graph(plan: Plan, outdir: Path, name: str = "graph") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    sg = outdir / "sg00"
    sg.mkdir(exist_ok=True)
    # embed DVE decode tables + activation sets from a reference graph
    # (self-contained, like xla-originated graphs)
    import shutil
    ref = Path(__file__).parent / "ref_tables"
    if ref.exists():
        for tbl in ("default_opcode_table.bin", "default_control_table.bin",
                    "default_datapath_table.bin"):
            p = ref / tbl
            if p.exists():
                shutil.copy(p, sg / tbl)
    (sg / "PE0.bin").write_bytes(ir.emit_pe_stream(plan.pe_ops) if plan.pe_ops
                                  else ir.PROLOGUE)
    # SP: prologue + TENSOR_STORE + DMA triggers for queues that have descriptors
    queues = []
    for d in plan.dma:
        if d.get("queue") and d["queue"] not in queues:
            queues.append(d["queue"])
    (sg / "SP0.bin").write_bytes(ir.emit_sp_stream(queues))
    for eng in ("DVE0", "Activation0", "Pool0"):
        (sg / f"{eng}.bin").write_bytes(ir.PROLOGUE)
    # engine JSONs mirror reference graphs
    (sg / "PE0.json").write_text(json.dumps(
        {"dma": [], "engine_id": 0, "instr": "PE0.bin", "name": "PE0.json"}))
    (sg / "DVE0.json").write_text(json.dumps(
        {"dma": [], "engine_id": 0, "instr": "DVE0.bin", "name": "DVE0.json",
         "dve_tables": [{"control_table": "default_control_table.bin",
                          "datapath_table": "default_datapath_table.bin",
                          "opcode_table": "default_opcode_table.bin"}]}))
    (sg / "Pool0.json").write_text(json.dumps(
        {"dma": [], "engine_id": 0, "instr": "Pool0.bin", "name": "Pool0.json"}))
    (sg / "SP0.json").write_text(json.dumps(
        {"dma": plan.dma, "engine_id": 0, "instr": "SP0.bin", "name": "SP0.json"}))
    (sg / "Activation0.json").write_text(json.dumps(
        {"act_info": "act_info.json", "act_semaphore": 12, "activation_function_sets": [],
         "dma": [], "engine_id": 0, "instr": "Activation0.bin", "name": "Activation0.json"}))
    def_json = ir.build_def_json(name, plan.tensors, {}, [], {
        "PE0": "PE0.bin", "DVE0": "DVE0.bin", "SP0": "SP0.bin",
        "Pool0": "Pool0.bin", "Activation0": "Activation0.bin"})
    (sg / "def.json").write_text(json.dumps(def_json, indent=1))
    # top-level: neff.json TVM graph with one __kelf node, info.json
    (outdir / "info.json").write_text(json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0] * 63}))
    shapes = []
    dltype = []
    for t in plan.tensors:
        shapes.append([t.size // 4, 1])
        dltype.append("float32")
    # one shape per node output: tensors + the tvm_op node's output
    shapes.append(shapes[-1])
    dltype.append(dltype[-1])
    neff_json = {
        "nodes": (
            [{"op": "null", "name": t.name, "inputs": [],
              "output_names": [t.name], "is_param": "0"} for t in plan.tensors]
            + [{"op": "tvm_op", "name": "sg_coreV10",
                "attrs": {"func_name": "__kelf", "flatten_data": "0",
                           "num_inputs": str(sum(1 for t in plan.tensors if t.kind == 'input')),
                           "kelf": "kelf-0.json",
                           "num_outputs": str(sum(1 for t in plan.tensors if t.kind == 'output'))},
                "inputs": [[i, 0, 0] for i, t in enumerate(plan.tensors) if t.kind == 'input'],
                "output_names": [t.name for t in plan.tensors if t.kind == 'output']}]),
        "arg_nodes": [i for i, t in enumerate(plan.tensors) if t.kind == 'input'],
        "heads": [[len(plan.tensors), 0, 0]],
        "attrs": {"shape": ["list_shape", shapes], "dltype": ["list_str", dltype],
                   "storage_id": ["list_int", list(range(len(plan.tensors) + 1))]},
        "node_row_ptr": list(range(len(plan.tensors) + 2)),
        "metadata": {"signatures": {"default": {
            "inputs": {t.name: {"id": i, "dtype": "float32", "shape": [t.size // 4]}
                       for i, t in enumerate(plan.tensors)},
            "outputs": {}}}},
    }
    (outdir / "neff.json").write_text(json.dumps(neff_json))
    (outdir / "kelf-0.json").write_text(json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (outdir / "metrics.json").write_text("[]")
    # container
    import neff_emit
    neff = outdir / f"{name}.neff"
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(outdir), str(neff), "--name", f"{name}.neff"], check=True)
    return neff


if __name__ == "__main__":
    plan = lift_tiled_projection(rows=128, in_features=128, out_features=128)
    for n in plan.notes:
        print("note:", n)
    out = Path("/tmp/opencode/lifted_graph")
    neff = emit_graph(plan, out, "lifted_mm")
    print("emitted", neff, neff.stat().st_size, "bytes")
    plan2 = lift_norm(width=4096)
    for n in plan2.notes:
        print("norm:", n)


# ---------------- general K/M/N matmul emitter (per-shape verified templates)

import json as _json

def emit_matmul(M: int, K: int, N: int, outdir: Path, name: str = "matmul") -> Path:
    """Emit a matmul NEFF for (M,K,N) - FULL SYNTHESIS (see emit_matmul_synth).

    PE/Pool/ACT streams synthesized from the parameter model in mm_synth.py
    (byte-exact vs 26 compiler oracle buckets, tools/mm_synth_check.py);
    no oracle stream copy, no compiler anywhere in the emit path.
    Raises ValueError for the 2 structurally-anomalous buckets
    (32_640_256, 64_512_384) and families outside the decoded set.
    """
    return emit_matmul_synth(outdir=outdir, M=M, K=K, N=N, name=name)


def emit_matmul_synth(outdir: Path, M: int, K: int, N: int,
                      name: str = "matmul") -> Path:
    """Emit a matmul NEFF for arbitrary (M,K,N) by FULL SYNTHESIS.

    No oracle streams, no compiler, no on-box generation:
      PE0.bin   = mm_synth.synth        (own synthesis, oracle-tested)
      Pool0.bin = mm_synth.synth_pool   (own synthesis, oracle-tested)
      ACT stream= act_synth.synth_act   (own synthesis, 29/29 byte-exact)
      SP0.bin   = mm_synth.UNIVERSAL_SP_HEX   (constant across 29 buckets)
      DVE0.bin  = mm_synth.UNIVERSAL_DVE_HEX  (constant across 29 buckets)
      SP0.json  = mm_synth.identity_desc      (29/29 vs oracle dma json)
      queue_set = mm_synth.QUEUE_SETS          (constant across 29 buckets)
      identity npy = eye(128) deterministic (npy bytes match compiler)
      engine-content tables (exp/act LUTs) from tools/ref_tables (frozen
      DATA required by the engines - same standing as the DVE tables).
    Device-proven families via synth self-consistency: direct / wring /
    col among the 32/48/64-row M families with the 5 literal tables.
    """
    import mm_synth as _ms
    import act_synth as _as
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    sg = out / "sg00"
    sg.mkdir(parents=True, exist_ok=True)
    (sg / "PE0.bin").write_bytes(_ms.synth(M, K, N))
    (sg / "Pool0.bin").write_bytes(_ms.synth_pool(M, K, N))
    (sg / "Activation0.bin").write_bytes(_as.synth_act(M, K, N))
    (sg / "DVE0.bin").write_bytes(bytes.fromhex(_ms.UNIVERSAL_DVE_HEX))
    (sg / "SP0.bin").write_bytes(bytes.fromhex(_ms.UNIVERSAL_SP_HEX))
    for j in ("act_info.json", "exp_and_others.json", "exp_and_others_bkt.bin",
              "exp_and_others_ctrl.bin", "default_opcode_table.bin",
              "default_control_table.bin", "default_datapath_table.bin"):
        srcp = Path(__file__).parent / "ref_tables" / j
        if srcp.exists():
            shutil.copy(srcp, sg / j)
    # engine JSONs (own)
    (sg / "PE0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "PE0.bin", "name": "PE0.json"}))
    (sg / "Pool0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "Pool0.bin", "name": "Pool0.json"}))
    (sg / "DVE0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "DVE0.bin", "name": "DVE0.json"}))
    (sg / "Activation0.json").write_text(_json.dumps(
        {"act_info": "act_info.json", "act_semaphore": 12,
         "activation_function_sets": ["exp_and_others"], "dma": [], "engine_id": 0,
         "instr": "Activation0.bin", "name": "Activation0.json"}))
    (sg / "SP0.json").write_text(_json.dumps({
        "dma": [_ms.identity_desc(M, K, N)], "engine_id": 0,
        "instr": "SP0.bin", "name": "SP0.json"}))
    ivn, ifn, isz, ivid = ("identity_0_sg0000", "identity_0_sg0000.npy",
                           65536, 4)
    A, B, Y = M * K * 4, K * N * 4, M * N * 4
    sets = _ms.QUEUE_SETS
    d = {
        "#rank_world_size": 0,
        "act": "Activation0.json", "act_asm_dbg": [], "act_dbg": [],
        "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0}, "debug_info": {}, "dma_dbg": [],
        "dma_queue": {q: ({"dynamic_dma": {"SB_scratch_partition_offset": 0,
                                           "SB_scratch_partition_size": 16384},
                           "fabric_path": "main", "num_queues": 16, "owner": "pool",
                           "semaphore_set": s, "type": "dynamic"}
                         if q == "qPoolDynamic" else
                         {"fabric_path": "main", "num_queues": 16,
                          "owner": q.startswith("qSP") and "sp" or "pool",
                          "semaphore_set": s, "type": "data"})
                      for q, s in sets.items()},
        "dve": "DVE0.json", "dve_asm_dbg": [], "dve_dbg": [], "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                  "fp8_range_extension": True, "fp8_sis": False,
                                  "fp8_sns": False, "ocp_sat": True},
        "name": "definition", "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": [], "pe_dbg": [], "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": [], "pool_dbg": [], "pool_instr": "Pool0.bin",
        "runtime_event_count": 12, "runtime_semaphore_count": 3,
        "sp": "SP0.json",
        "sp_asm_dbg": [], "sp_dbg": [], "sp_instr": "SP0.bin",
        "var": {"SB": {"type": "state-buffer", "var_id": 0},
                "input0": {"ops": [], "size": A, "type": "input", "var_id": 1},
                "input1": {"ops": [], "size": B, "type": "input", "var_id": 2},
                "output0": {"ops": [], "size": Y, "type": "output", "var_id": 3},
                ivn: {"alignment": 65536, "fabric_path": "main", "file_name": ifn,
                      "ops": [], "size": isz, "type": "file", "var_id": ivid},
                "DGETable0": {"list": [1, 2, 3], "type": "dge-table", "var_id": 5}},
        "version": "0.6",
    }
    (sg / "def.json").write_text(_json.dumps(d, indent=1))
    import numpy as _np
    import io as _io
    idata = _np.eye(128, dtype=_np.float32).reshape(128, 128, 1, 1)
    buf = _io.BytesIO()
    _np.save(buf, idata)
    (sg / ifn).write_bytes(buf.getvalue())
    (out / "info.json").write_text(_json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0] * 63}))
    nj = {
        "nodes": [
            {"op": "null", "name": "input0", "inputs": [], "output_names": ["input0"], "is_param": "0"},
            {"op": "null", "name": "input1", "inputs": [], "output_names": ["input1"], "is_param": "0"},
            {"op": "tvm_op", "name": "sg_coreV10",
             "attrs": {"func_name": "__kelf", "flatten_data": "0", "num_inputs": "2",
                       "kelf": "kelf-0.json", "num_outputs": "1"},
             "inputs": [[0, 0, 0], [1, 0, 0]], "output_names": ["output0"]}],
        "arg_nodes": [0, 1], "heads": [[2, 0, 0]],
        "attrs": {"shape": ["list_shape", [[M, K], [K, N], [M, N]]],
                  "dltype": ["list_str", ["float32"] * 3],
                  "storage_id": ["list_int", [0, 1, 2]]},
        "node_row_ptr": [0, 1, 2, 3],
        "metadata": {"signatures": {"default": {
            "inputs": {"input0": {"id": 0, "dtype": "float32", "shape": [M, K]},
                       "input1": {"id": 1, "dtype": "float32", "shape": [K, N]}},
            "outputs": {}}}},
    }
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


def emit_elementwise(outdir: Path, n: int = 4096, name: str = "elementwise",
                     scale: float = 1.0) -> Path:
    """Emit y = scale*x*w elementwise NEFF over n f32 elements (OWN SYNTH).

    Structure fully decoded + synthesized in tools/ew_synth.py:
      n <= 128: single-pass DVE op67+op10 (no activation pass);
      n > 128:  DVE op65 pass + ACTIVATE op33/op170 with inline f32 scale.
    All n/scale dependence = u16/f32 fields (P = ceil(n/128)). Streams are
    byte-exact vs the reference oracles at n=100 (B) and n=8192 (A);
    device-proven bit-exact y=scale*x*w at 10 n values (memory 17).
    "if nout > 1:" refs: none.
    """
    import ew_synth
    # Numerics cap (roofline finding): DVE pass semaphore counter = P*16
    # increments, u16 wraps above 65535 -> wrong results at P >= 4096
    # (observed at n=1048576; correct at n<=262144). Guard until per-pass
    # re-arm is decoded.
    if n <= 0 or (n + 127) // 128 > 4095:
        raise ValueError(f"n={n} out of supported numerics range "
                         f"(1..~524160; P*16 must fit u16)")
    P = (n + 127) // 128
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    sg = out / "sg00"
    sg.mkdir(parents=True, exist_ok=True)
    streams = ew_synth.synth_streams(n, scale)
    for fn, data in streams.items():
        (sg / fn).write_bytes(data)
    # frozen engine tables (exp/act LUTs for the elementwise graph family,
    # default DVE tables) - DATA required by the engines
    import shutil as _sh
    _sh.copy(Path(__file__).parent / "ref_tables" / "elementwise" / "act_info.json", sg)
    for j in ("exp_and_others.json", "exp_and_others_bkt.bin",
              "exp_and_others_ctrl.bin"):
        _sh.copy(Path(__file__).parent / "ref_tables" / "elementwise" / j, sg)
    for j in ("default_opcode_table.bin", "default_control_table.bin",
              "default_datapath_table.bin"):
        _sh.copy(Path(__file__).parent / "ref_tables" / j, sg)
    # engine jsons + def.json (own construct, mirror of reference schema)
    (sg / "PE0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "PE0.bin", "name": "PE0.json"}))
    (sg / "DVE0.json").write_text(_json.dumps(
        {"dma": [], "dve_tables": [{"control_table": "default_control_table.bin",
                                     "datapath_table": "default_datapath_table.bin",
                                     "opcode_table": "default_opcode_table.bin"}],
         "engine_id": 0, "instr": "DVE0.bin", "name": "DVE0.json"}))
    (sg / "Pool0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "Pool0.bin", "name": "Pool0.json"}))
    (sg / "SP0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "SP0.bin", "name": "SP0.json"}))
    (sg / "Activation0.json").write_text(_json.dumps(
        {"act_info": "act_info.json", "act_semaphore": 12,
         "activation_function_sets": ["exp_and_others"], "dma": [],
         "engine_id": 0, "instr": "Activation0.bin",
         "name": "Activation0.json"}))
    debug_lists = {"pe": ["debug_info_asm_PE.dbg", "debug_info_backend_PE.dbg"],
                   "dve": ["debug_info_asm_DVE.dbg", "debug_info_backend_DVE.dbg"],
                   "pool": ["debug_info_asm_Pool.dbg", "debug_info_backend_Pool.dbg"],
                   "sp": ["debug_info_asm_SP.dbg", "debug_info_backend_SP.dbg"],
                   "act": ["debug_info_asm_Activation.dbg", "debug_info_backend_Activation.dbg"],
                   "dma": ["debug_info_backend_DMA.dbg"]}
    nbytes = n * 4
    d = {
        "#rank_world_size": 0,
        "act": "Activation0.json", "act_asm_dbg": debug_lists["act"][:1],
        "act_dbg": debug_lists["act"][1:], "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0}, "debug_info": {},
        "dma_dbg": debug_lists["dma"],
        "dma_queue": {"qPoolDynamic": {
            "dynamic_dma": {"SB_scratch_partition_offset": 0,
                             "SB_scratch_partition_size": 16384},
            "fabric_path": "main", "num_queues": 16, "owner": "pool",
            "semaphore_set": list(range(8, 24)), "type": "dynamic"}},
        "dve": "DVE0.json", "dve_asm_dbg": debug_lists["dve"][:1],
        "dve_dbg": debug_lists["dve"][1:], "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                  "fp8_range_extension": True, "fp8_sis": False,
                                  "fp8_sns": False, "ocp_sat": True},
        "name": "definition", "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": debug_lists["pe"][:1],
        "pe_dbg": debug_lists["pe"][1:], "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": debug_lists["pool"][:1],
        "pool_dbg": debug_lists["pool"][1:], "pool_instr": "Pool0.bin",
        "runtime_event_count": 12, "runtime_semaphore_count": 3,
        "sp": "SP0.json", "sp_asm_dbg": debug_lists["sp"][:1],
        "sp_dbg": debug_lists["sp"][1:], "sp_instr": "SP0.bin",
        "var": {"DGETable0": {"list": [1, 2, 3], "type": "dge-table", "var_id": 4},
                "SB": {"type": "state-buffer", "var_id": 0},
                "input0": {"ops": [], "size": nbytes, "type": "input", "var_id": 1},
                "input1": {"ops": [], "size": nbytes, "type": "input", "var_id": 2},
                "output0": {"ops": [], "size": nbytes, "type": "output", "var_id": 3}},
        "version": "0.6",
    }
    (sg / "def.json").write_text(_json.dumps(d, indent=1))
    nj = {
        "nodes": [
            {"op": "null", "name": "input0", "inputs": [], "output_names": ["input0"], "is_param": "0"},
            {"op": "null", "name": "input1", "inputs": [], "output_names": ["input1"], "is_param": "0"},
            {"op": "tvm_op", "name": "sg_coreV10",
             "attrs": {"func_name": "__kelf", "flatten_data": "0", "num_inputs": "2",
                        "kelf": "kelf-0.json", "num_outputs": "1"},
             "inputs": [[0, 0, 0], [1, 0, 0]], "output_names": ["output0"]}],
        "arg_nodes": [0, 1], "heads": [[2, 0, 0]],
        "attrs": {"shape": ["list_shape", [[1, n]] * 3],
                  "dltype": ["list_str", ["float32"] * 3],
                  "storage_id": ["list_int", [0, 1, 2]]},
        "node_row_ptr": [0, 1, 2, 3],
        "metadata": {"signatures": {"default": {
            "inputs": {"input0": {"id": 0, "dtype": "float32", "shape": [1, n]},
                        "input1": {"id": 1, "dtype": "float32", "shape": [1, n]}},
            "outputs": {}}}},
    }
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    (out / "info.json").write_text(_json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0] * 63}))
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


def emit_rms_g(outdir: Path, M: int, K: int, eps: float = 1e-6,
               name: str = "rmsg") -> Path:
    """Emit TWO outputs from one rms graph: y0 = rms(x) (unit norm),
    y1 = x*g*rsqrt(mean(x^2)+eps) (weighted). Reuses rms_synth streams
    (rms and g-mul run on the same staged x); container owns 2 real
    outputs via core num_outputs=2 + heads->core (ref pattern)."""
    import rms_synth
    if (M, K) not in rms_synth.VALIDATED:
        raise ValueError("rms synth validated only for "
                         f"{rms_synth.VALIDATED}; got ({M},{K})")
    out = Path(outdir)
    sg = out / "sg00"
    shutil.rmtree(out, ignore_errors=True)
    sg.mkdir(parents=True)
    streams = rms_synth.synth_streams(M, K, eps)
    (sg / "DVE0.bin").write_bytes(streams["DVE0.bin"])
    (sg / "Activation0.bin").write_bytes(streams["Activation0.bin"])
    (sg / "Pool0.bin").write_bytes(streams["Pool0.bin"])
    (sg / "PE0.bin").write_bytes(rms_synth.PROLOGUE)
    (sg / "SP0.bin").write_bytes(rms_synth.PROLOGUE)
    for f in ("act_info.json", "exp_and_others.json", "exp_and_others_bkt.bin",
              "exp_and_others_ctrl.bin"):
        shutil.copy(Path(__file__).parent / "ref_tables" / "elementwise" / f, sg)
    for f in ("default_opcode_table.bin", "default_control_table.bin",
              "default_datapath_table.bin"):
        shutil.copy(Path(__file__).parent / "ref_tables" / f, sg)
    (sg / "PE0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "PE0.bin", "name": "PE0.json"}))
    (sg / "SP0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "SP0.bin", "name": "SP0.json"}))
    (sg / "Pool0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "Pool0.bin", "name": "Pool0.json"}))
    (sg / "DVE0.json").write_text(_json.dumps(
        {"dma": [], "dve_tables": [{"control_table": "default_control_table.bin",
                                     "datapath_table": "default_datapath_table.bin",
                                     "opcode_table": "default_opcode_table.bin"}],
         "engine_id": 0, "instr": "DVE0.bin", "name": "DVE0.json"}))
    (sg / "Activation0.json").write_text(_json.dumps(
        {"act_info": "act_info.json", "act_semaphore": 12,
         "activation_function_sets": ["exp_and_others"], "dma": [],
         "engine_id": 0, "instr": "Activation0.bin", "name": "Activation0.json"}))
    dbg = {"pe": ["debug_info_asm_PE.dbg"], "dve": ["debug_info_asm_DVE.dbg"],
           "pool": ["debug_info_asm_Pool.dbg"], "act": ["debug_info_asm_Activation.dbg"],
           "dma": ["debug_info_backend_DMA.dbg"]}
    nb = M * K * 4
    d = {
        "#rank_world_size": 0,
        "act": "Activation0.json", "act_asm_dbg": dbg["act"], "act_dbg": [],
        "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0}, "debug_info": {},
        "dma_dbg": dbg["dma"],
        "dma_queue": {"qPoolDynamic": {
            "dynamic_dma": {"SB_scratch_partition_offset": 0,
                             "SB_scratch_partition_size": 16384},
            "fabric_path": "main", "num_queues": 16, "owner": "pool",
            "semaphore_set": list(range(8, 24)), "type": "dynamic"}},
        "dve": "DVE0.json", "dve_asm_dbg": dbg["dve"], "dve_dbg": [],
        "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                  "fp8_range_extension": True, "fp8_sis": False,
                                  "fp8_sns": False, "ocp_sat": True},
        "name": "definition", "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": dbg["pe"], "pe_dbg": [],
        "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": dbg["pool"], "pool_dbg": [],
        "pool_instr": "Pool0.bin",
        "runtime_event_count": 12, "runtime_semaphore_count": 3,
        "sp": "SP0.json", "sp_asm_dbg": [], "sp_dbg": [], "sp_instr": "SP0.bin",
        "var": {"DGETable0": {"list": [1, 2, 3, 4], "type": "dge-table",
                               "var_id": 5},
                "SB": {"type": "state-buffer", "var_id": 0},
                "input0": {"ops": [], "size": nb, "type": "input", "var_id": 1},
                "input1": {"ops": [], "size": K * 4, "type": "input", "var_id": 2},
                "output0": {"ops": [], "size": nb, "type": "output", "var_id": 3},
                "output1": {"ops": [], "size": nb, "type": "output", "var_id": 4}},
        "version": "0.6",
    }
    (sg / "def.json").write_text(_json.dumps(d, indent=1))
    nj = {"nodes": [
        {"op": "null", "name": "input0", "inputs": [], "output_names": ["input0"], "is_param": "0"},
        {"op": "null", "name": "input1", "inputs": [], "output_names": ["input1"], "is_param": "0"},
        {"op": "tvm_op", "name": "sg_coreV10",
         "attrs": {"func_name": "__kelf", "flatten_data": "0", "num_inputs": "2",
                    "kelf": "kelf-0.json", "num_outputs": "2"},
         "inputs": [[0, 0, 0], [1, 0, 0]],
         "output_names": ["output0", "output1"]}],
        "arg_nodes": [0, 1], "heads": [[2, 0, 0], [2, 1, 0]],
        "attrs": {"shape": ["list_shape", [[M, K], [K], [M, K], [M, K]]],
                  "dltype": ["list_str", ["float32"] * 4],
                  "storage_id": ["list_int", [0, 1, 2, 3]]},
        "node_row_ptr": [0, 1, 2, 3, 5],
        "metadata": {"signatures": {"default": {
            "inputs": {"input0": {"id": 0, "dtype": "float32", "shape": [M, K]},
                        "input1": {"id": 1, "dtype": "float32", "shape": [K]}},
            "outputs": {}}}}}
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    (out / "info.json").write_text(_json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0] * 63}))
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


SOFTMAX_ORACLE_DIRS = {  # frozen sg00 dirs (ground truth + tables)
    (128, 128): str(Path(__file__).parent / "oracle_sg00" / "softmax128"),
    (96, 384): str(Path(__file__).parent / "oracle_sg00" / "softmax96x384"),
}


def emit_softmax(outdir: Path, M: int, K: int, name: str = "softmax") -> Path:
    """Emit y = softmax(x, axis=-1) rowwise for x [M, K] f32 (OWN SYNTH).

    DVE max/sum reduce + row-div (op66/op67/op72), ACT exp pair - fully
    synthesized in tools/sf_synth.py (byte-exact vs softmax128 and
    M96/K384 oracles). PE/SP = frozen prologue bytes, engine tables and
    act LUTs = frozen DATA from the oracle sg00 dir.
    """
    if (M, K) not in SOFTMAX_ORACLE_DIRS:
        raise ValueError(f"no softmax table for ({M},{K})")
    import sf_synth
    sf_path = Path(SOFTMAX_ORACLE_DIRS[(M, K)])
    out = Path(outdir)
    sg = out / "sg00"
    shutil.rmtree(out, ignore_errors=True)
    sg.mkdir(parents=True)
    streams = sf_synth.synth_streams(M, K)
    (sg / "DVE0.bin").write_bytes(streams["DVE0.bin"])
    (sg / "Activation0.bin").write_bytes(streams["Activation0.bin"])
    (sg / "Pool0.bin").write_bytes(streams["Pool0.bin"])
    shutil.copy(sf_path / "PE0.bin", sg)
    shutil.copy(sf_path / "SP0.bin", sg)
    for f in ("act_info.json", "exp_and_others.json", "exp_and_others_bkt.bin",
              "exp_and_others_ctrl.bin", "default_opcode_table.bin",
              "default_control_table.bin", "default_datapath_table.bin"):
        shutil.copy(sf_path / f, sg)
    (sg / "PE0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "PE0.bin", "name": "PE0.json"}))
    (sg / "SP0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "SP0.bin", "name": "SP0.json"}))
    (sg / "Pool0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "Pool0.bin", "name": "Pool0.json"}))
    (sg / "DVE0.json").write_text(_json.dumps(
        {"dma": [], "dve_tables": [{"control_table": "default_control_table.bin",
                                     "datapath_table": "default_datapath_table.bin",
                                     "opcode_table": "default_opcode_table.bin"}],
         "engine_id": 0, "instr": "DVE0.bin", "name": "DVE0.json"}))
    (sg / "Activation0.json").write_text(_json.dumps(
        {"act_info": "act_info.json", "act_semaphore": 12,
         "activation_function_sets": ["exp_and_others"], "dma": [],
         "engine_id": 0, "instr": "Activation0.bin", "name": "Activation0.json"}))
    dbg = {"pe": ["debug_info_asm_PE.dbg", "debug_info_backend_PE.dbg"],
           "dve": ["debug_info_asm_DVE.dbg", "debug_info_backend_DVE.dbg"],
           "pool": ["debug_info_asm_Pool.dbg", "debug_info_backend_Pool.dbg"],
           "sp": ["debug_info_asm_SP.dbg", "debug_info_backend_SP.dbg"],
           "act": ["debug_info_asm_Activation.dbg", "debug_info_backend_Activation.dbg"],
           "dma": ["debug_info_backend_DMA.dbg"]}
    nbytes = M * K * 4
    d = {
        "#rank_world_size": 0,
        "act": "Activation0.json", "act_asm_dbg": dbg["act"][:1],
        "act_dbg": dbg["act"][1:], "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0}, "debug_info": {},
        "dma_dbg": dbg["dma"],
        "dma_queue": {"qPoolDynamic": {
            "dynamic_dma": {"SB_scratch_partition_offset": 0,
                             "SB_scratch_partition_size": 16384},
            "fabric_path": "main", "num_queues": 16, "owner": "pool",
            "semaphore_set": list(range(8, 24)), "type": "dynamic"}},
        "dve": "DVE0.json", "dve_asm_dbg": dbg["dve"][:1],
        "dve_dbg": dbg["dve"][1:], "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                  "fp8_range_extension": True, "fp8_sis": False,
                                  "fp8_sns": False, "ocp_sat": True},
        "name": "definition", "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": dbg["pe"][:1],
        "pe_dbg": dbg["pe"][1:], "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": dbg["pool"][:1],
        "pool_dbg": dbg["pool"][1:], "pool_instr": "Pool0.bin",
        "runtime_event_count": 12, "runtime_semaphore_count": 3,
        "sp": "SP0.json", "sp_asm_dbg": dbg["sp"][:1],
        "sp_dbg": dbg["sp"][1:], "sp_instr": "SP0.bin",
        "var": {"DGETable0": {"list": [1, 2], "type": "dge-table", "var_id": 3},
                "SB": {"type": "state-buffer", "var_id": 0},
                "input0": {"ops": [], "size": nbytes, "type": "input", "var_id": 1},
                "output0": {"ops": [], "size": nbytes, "type": "output", "var_id": 2}},
        "version": "0.6",
    }
    (sg / "def.json").write_text(_json.dumps(d, indent=1))
    nj = {"nodes": [
        {"op": "null", "name": "input0", "inputs": [], "output_names": ["input0"], "is_param": "0"},
        {"op": "tvm_op", "name": "sg_coreV10",
         "attrs": {"func_name": "__kelf", "flatten_data": "0", "num_inputs": "1",
                    "kelf": "kelf-0.json", "num_outputs": "1"},
         "inputs": [[0, 0, 0]], "output_names": ["output0"]}],
        "arg_nodes": [0], "heads": [[1, 0, 0]],
        "attrs": {"shape": ["list_shape", [[M, K], [M, K]]],
                  "dltype": ["list_str", ["float32", "float32"]],
                  "storage_id": ["list_int", [0, 1]]},
        "node_row_ptr": [0, 1, 2],
        "metadata": {"signatures": {"default": {
            "inputs": {"input0": {"id": 0, "dtype": "float32", "shape": [M, K]}},
            "outputs": {}}}}}
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    (out / "info.json").write_text(_json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0] * 63}))
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"




def emit_dequant(outdir: Path, M: int, K: int, name: str = "dequant") -> Path:
    """Emit y = q.cast(f32) * s[None, :] for q uint8 [M,K], s f32 [K].

    OWN SYNTHESIS (tools/deq_synth.py, M128 family): PE u8->f32 cast,
    DVE s-broadcast, all K/M dependence = u16 fields. Streams byte-exact
    vs compiler oracles at (128,128), (128,192), (128,256).
    Input file order: in0 = q (uint8), in1 = s (f32 K), output = f32.
    """
    import deq_synth
    if M != 128 or K < 64 or K > 256 or K % 64:
        raise ValueError(f"dequant synth validated for M=128, K in 64..256 "
                         f"mult 64; got ({M},{K})")
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    sg = out / "sg00"
    sg.mkdir(parents=True)
    streams = deq_synth.synth_streams(M, K)
    for fn, data in streams.items():
        (sg / fn).write_bytes(data)
    _sh = shutil
    _sh.copy(Path(__file__).parent / "ref_tables" / "elementwise" / "act_info.json", sg)
    for j in ("default_opcode_table.bin", "default_control_table.bin",
              "default_datapath_table.bin"):
        _sh.copy(Path(__file__).parent / "ref_tables" / j, sg)
    (sg / "PE0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "PE0.bin", "name": "PE0.json"}))
    (sg / "DVE0.json").write_text(_json.dumps(
        {"dma": [], "dve_tables": [{"control_table": "default_control_table.bin",
                                     "datapath_table": "default_datapath_table.bin",
                                     "opcode_table": "default_opcode_table.bin"}],
         "engine_id": 0, "instr": "DVE0.bin", "name": "DVE0.json"}))
    (sg / "Pool0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "Pool0.bin", "name": "Pool0.json"}))
    (sg / "SP0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "SP0.bin", "name": "SP0.json"}))
    (sg / "Activation0.json").write_text(_json.dumps(
        {"act_info": "act_info.json", "act_semaphore": 12,
         "activation_function_sets": [], "dma": [],
         "engine_id": 0, "instr": "Activation0.bin",
         "name": "Activation0.json"}))
    dbg = {"pe": ["debug_info_asm_PE.dbg"], "dve": ["debug_info_asm_DVE.dbg"],
           "pool": ["debug_info_asm_Pool.dbg"], "sp": ["debug_info_asm_SP.dbg"],
           "act": ["debug_info_asm_Activation.dbg"],
           "dma": ["debug_info_backend_DMA.dbg"]}
    d = {
        "#rank_world_size": 0,
        "act": "Activation0.json", "act_asm_dbg": dbg["act"], "act_dbg": [],
        "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0}, "debug_info": {},
        "dma_dbg": dbg["dma"],
        "dma_queue": {"qPoolDynamic": {
            "dynamic_dma": {"SB_scratch_partition_offset": 0,
                             "SB_scratch_partition_size": 16384},
            "fabric_path": "main", "num_queues": 16, "owner": "pool",
            "semaphore_set": list(range(8, 24)), "type": "dynamic"}},
        "dve": "DVE0.json", "dve_asm_dbg": dbg["dve"], "dve_dbg": [],
        "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                  "fp8_range_extension": True, "fp8_sis": False,
                                  "fp8_sns": False, "ocp_sat": True},
        "name": "definition", "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": dbg["pe"], "pe_dbg": [],
        "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": dbg["pool"], "pool_dbg": [],
        "pool_instr": "Pool0.bin",
        "runtime_event_count": 12, "runtime_semaphore_count": 3,
        "sp": "SP0.json", "sp_asm_dbg": dbg["sp"], "sp_dbg": [],
        "sp_instr": "SP0.bin",
        "var": {"DGETable0": {"list": [2, 1, 3], "type": "dge-table",
                               "var_id": 4},
                "SB": {"type": "state-buffer", "var_id": 0},
                "input0": {"ops": [], "size": M * K, "type": "input", "var_id": 1},
                "input1": {"ops": [], "size": K * 4, "type": "input", "var_id": 2},
                "output0": {"ops": [], "size": M * K * 4, "type": "output", "var_id": 3}},
        "version": "0.6",
    }
    (sg / "def.json").write_text(_json.dumps(d, indent=1))
    nj = {"nodes": [
        {"op": "null", "name": "input0", "inputs": [], "output_names": ["input0"], "is_param": "0"},
        {"op": "null", "name": "input1", "inputs": [], "output_names": ["input1"], "is_param": "0"},
        {"op": "tvm_op", "name": "sg_coreV10",
         "attrs": {"func_name": "__kelf", "flatten_data": "0", "num_inputs": "2",
                   "kelf": "kelf-0.json", "num_outputs": "1"},
         "inputs": [[0, 0, 0], [1, 0, 0]], "output_names": ["output0"]}],
        "arg_nodes": [0, 1], "heads": [[2, 0, 0]],
        "attrs": {"shape": ["list_shape", [[M, K], [K], [M, K]]],
                  "dltype": ["list_str", ["uint8", "float32", "float32"]],
                  "storage_id": ["list_int", [0, 1, 2]]},
        "node_row_ptr": [0, 1, 2, 3],
        "metadata": {"signatures": {"default": {
            "inputs": {"input0": {"id": 0, "dtype": "uint8", "shape": [M, K]},
                       "input1": {"id": 1, "dtype": "float32", "shape": [K]}},
            "outputs": {}}}}}
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    (out / "info.json").write_text(_json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0] * 63}))
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


RMS_ORACLE_DIRS = {  # (M, K) -> frozen oracle sg00 dir (tools/)
    (128, 128): str(Path(__file__).parent / "rms_oracles" / "rms128"),
    (256, 128): str(Path(__file__).parent / "oracle_sg00" / "rms256"),
}


def emit_rmsnorm(outdir: Path, M: int, K: int, eps: float = 1e-6,
                 name: str = "rmsnorm") -> Path:
    """Emit y = x * rsqrt(mean(x^2, axis=-1) + eps) for x [M, K] f32.

    OWN SYNTHESIS (tools/rms_synth.py): no PE engine. DVE MEMSET(73) +
    TENSOR_REDUCE_ARITH_OP(66) into PSUM (marker 8) over P = M/128 channel
    passes, then ACTIVATE(33) chain = setup (op122: 1.0 + K*K), sqrt/div
    (op170, 1/K + eps f32), y-mul passes per channel block. Streams byte-
    exact vs the rms128/rms256 oracles; M/K/eps parameterized. Engine
    tables = frozen DATA from ref_tables (same exp/act LUT family as the
    elementwise graph). Weighted rms = compose with emit_elementwise.
    """
    import rms_synth
    if (M, K) not in rms_synth.VALIDATED:
        raise ValueError("rms synth validated only for "
                         f"{rms_synth.VALIDATED}; got ({M},{K}) - y-block "
                         "geometry beyond M,K=128 buckets is unproven")
    out = Path(outdir)
    sg = out / "sg00"
    shutil.rmtree(out, ignore_errors=True)
    sg.mkdir(parents=True)
    streams = rms_synth.synth_streams(M, K, eps)
    (sg / "DVE0.bin").write_bytes(streams["DVE0.bin"])
    (sg / "Activation0.bin").write_bytes(streams["Activation0.bin"])
    (sg / "Pool0.bin").write_bytes(streams["Pool0.bin"])
    (sg / "PE0.bin").write_bytes(rms_synth.PROLOGUE)
    (sg / "SP0.bin").write_bytes(rms_synth.PROLOGUE)
    # frozen engine tables + activation LUT family (DATA)
    for f in ("act_info.json", "exp_and_others.json", "exp_and_others_bkt.bin",
              "exp_and_others_ctrl.bin"):
        shutil.copy(Path(__file__).parent / "ref_tables" / "elementwise" / f, sg)
    for f in ("default_opcode_table.bin", "default_control_table.bin",
              "default_datapath_table.bin"):
        shutil.copy(Path(__file__).parent / "ref_tables" / f, sg)
    # engine jsons (own, schema = reference)
    (sg / "PE0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "PE0.bin", "name": "PE0.json"}))
    (sg / "SP0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "SP0.bin", "name": "SP0.json"}))
    (sg / "Pool0.json").write_text(_json.dumps(
        {"dma": [], "engine_id": 0, "instr": "Pool0.bin", "name": "Pool0.json"}))
    (sg / "DVE0.json").write_text(_json.dumps(
        {"dma": [], "dve_tables": [{"control_table": "default_control_table.bin",
                                     "datapath_table": "default_datapath_table.bin",
                                     "opcode_table": "default_opcode_table.bin"}],
         "engine_id": 0, "instr": "DVE0.bin", "name": "DVE0.json"}))
    (sg / "Activation0.json").write_text(_json.dumps(
        {"act_info": "act_info.json", "act_semaphore": 12,
         "activation_function_sets": ["exp_and_others"], "dma": [],
         "engine_id": 0, "instr": "Activation0.bin", "name": "Activation0.json"}))
    dbg = {"pe": ["debug_info_asm_PE.dbg"], "dve": ["debug_info_asm_DVE.dbg"],
           "pool": ["debug_info_asm_Pool.dbg"], "act": ["debug_info_asm_Activation.dbg"],
           "dma": ["debug_info_backend_DMA.dbg"]}
    nbytes = M * K * 4
    d = {
        "#rank_world_size": 0,
        "act": "Activation0.json", "act_asm_dbg": dbg["act"],
        "act_dbg": [], "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0}, "debug_info": {},
        "dma_dbg": dbg["dma"],
        "dma_queue": {"qPoolDynamic": {
            "dynamic_dma": {"SB_scratch_partition_offset": 0,
                             "SB_scratch_partition_size": 16384},
            "fabric_path": "main", "num_queues": 16, "owner": "pool",
            "semaphore_set": list(range(8, 24)), "type": "dynamic"}},
        "dve": "DVE0.json", "dve_asm_dbg": dbg["dve"], "dve_dbg": [],
        "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                  "fp8_range_extension": True, "fp8_sis": False,
                                  "fp8_sns": False, "ocp_sat": True},
        "name": "definition", "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": dbg["pe"], "pe_dbg": [],
        "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": dbg["pool"], "pool_dbg": [],
        "pool_instr": "Pool0.bin",
        "runtime_event_count": 12, "runtime_semaphore_count": 3,
        "sp": "SP0.json", "sp_asm_dbg": [], "sp_dbg": [], "sp_instr": "SP0.bin",
        "var": {"DGETable0": {"list": [1, 2], "type": "dge-table", "var_id": 3},
                "SB": {"type": "state-buffer", "var_id": 0},
                "input0": {"ops": [], "size": nbytes, "type": "input", "var_id": 1},
                "output0": {"ops": [], "size": nbytes, "type": "output", "var_id": 2}},
        "version": "0.6",
    }
    (sg / "def.json").write_text(_json.dumps(d, indent=1))
    nj = {"nodes": [
        {"op": "null", "name": "input0", "inputs": [], "output_names": ["input0"], "is_param": "0"},
        {"op": "tvm_op", "name": "sg_coreV10",
         "attrs": {"func_name": "__kelf", "flatten_data": "0", "num_inputs": "1",
                   "kelf": "kelf-0.json", "num_outputs": "1"},
         "inputs": [[0, 0, 0]], "output_names": ["output0"]}],
        "arg_nodes": [0], "heads": [[1, 0, 0]],
        "attrs": {"shape": ["list_shape", [[M, K], [M, K]]],
                  "dltype": ["list_str", ["float32", "float32"]],
                  "storage_id": ["list_int", [0, 1]]},
        "node_row_ptr": [0, 1, 2],
        "metadata": {"signatures": {"default": {
            "inputs": {"input0": {"id": 0, "dtype": "float32", "shape": [M, K]}},
            "outputs": {}}}}}
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    (out / "info.json").write_text(_json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0] * 63}))
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"
