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
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

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
    ref = Path("/tmp/opencode/neff/ffn_x/sg00")
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
