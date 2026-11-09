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


# ---------------- general K/M/N matmul emitter (per-shape verified templates)

import json as _json

_SHAPE_TPL = Path(__file__).parent / "shape_templates.json"


def _generate_oracle_on_box(M, K, N, key):
    """Compile an oracle NEFF for (M,K,N) on the trn1 box via SSM + StableHLO
    bytecode + neuronx-cc; extract sg00 streams into shape_templates.json."""
    import base64 as _b64
    import gzip as _gz
    gen_py = (
        "import io, os\n"
        "import jax, jax.numpy as jnp\n"
        f"d = '/tmp/oracle_gen_{key}'\n"
        "os.makedirs(d, exist_ok=True)\n"
        "f = jax.jit(lambda a, b: a @ b)\n"
        f"lowered = f.lower(jnp.zeros(({M},{K}), jnp.float32), jnp.zeros(({K},{N}), jnp.float32))\n"
        "buf = io.BytesIO()\n"
        "lowered.compiler_ir().operation.write_bytecode(buf)\n"
        "open(d + '/graph.mlirbc', 'wb').write(buf.getvalue())\n"
        "print('gen ok')\n")
    open("/tmp/opencode/_gen_cur.b64", "w").write(_b64.b64encode(gen_py.encode()).decode())
    subprocess.run(["aws", "s3", "cp", "/tmp/opencode/_gen_cur.b64",
                    "s3://912935854432-neuron-build/tools/gen_cur.py.b64"],
                   check=True, capture_output=True)
    inner = (
        "source /home/ubuntu/work/nenv.sh 2>/dev/null\n"
        "export PATH=/opt/aws_neuronx_venv_jax_0_9/bin:/usr/local/bin:$PATH\n"
        "cd /tmp\n"
        "aws s3 cp s3://912935854432-neuron-build/tools/gen_cur.py.b64 /tmp/gen_cur.b64 1>/dev/null\n"
        "python3 -c \"import base64; open('/tmp/gen_cur.py','wb').write(base64.b64decode(open('/tmp/gen_cur.b64').read()))\"\n"
        f"/opt/aws_neuronx_venv_jax_0_9/bin/python /tmp/gen_cur.py\n"
        f"cd /tmp/oracle_gen_{key} && neuronx-cc compile graph.mlirbc --framework XLA --target trn1 --output o.neff 2>&1 | tail -3\n"
        "python3 -c \"\n"
        "import gzip, tarfile, io, base64, json\n"
        "d = open('/tmp/oracle_gen_" + key + "/o.neff','rb').read()\n"
        "t = tarfile.open(fileobj=io.BytesIO(gzip.decompress(d[1024:])))\n"
        "out = {}\n"
        "for m in t.getmembers():\n"
        "    if m.name.startswith('sg00/') and (m.name.endswith('.bin') or m.name.endswith('.json')) and all(x not in m.name for x in ('debug','bkt','exp_and','act_info','default_','value_')):\n"
        "        out[m.name] = base64.b64encode(t.extractfile(m).read()).decode()\n"
        "print('NEFFDATA:' + json.dumps(out))\"\n")
    open('/tmp/opencode/_gen_inner.sh', 'w').write(inner)
    r = subprocess.run(["bash", "/tmp/opencode/ssm_run.sh", "/tmp/opencode/_gen_inner.sh", "300"],
                       capture_output=True, text=True, timeout=320)
    out = r.stdout
    marker = out.find("NEFFDATA:")
    if marker < 0:
        raise RuntimeError(f"oracle gen failed for {key}: {out[-400:]} {r.stderr[-200:]}")
    line = [l for l in out[marker:].split('\n') if 'NEFFDATA:' in l][0]
    data = json.loads(line.split('NEFFDATA:', 1)[1])
    sg_dir = Path(f"/tmp/opencode/oracle_{key}/sg00")
    sg_dir.mkdir(parents=True, exist_ok=True)
    for nm, b in data.items():
        (sg_dir / Path(nm).name).write_bytes(_b64.b64decode(b))
    ident = [(vn, vv['file_name'], vv['size'], vv['var_id'])
             for vn, vv in json.load(open(sg_dir / 'def.json'))['var'].items()
             if vv.get('type') == 'file'][0]
    sp = json.load(open(sg_dir / 'SP0.json'))
    tpl = _json.loads(_SHAPE_TPL.read_text())
    tpl[key] = {
        "pool_hex": open(sg_dir / 'Pool0.bin', 'rb').read().hex(),
        "pe_hex": open(sg_dir / 'PE0.bin', 'rb').read().hex(),
        "sp_dma": sp['dma'], "identity_var": ident,
        "queue_sets": {q: v.get('semaphore_set')
                       for q, v in json.load(open(sg_dir / 'def.json'))['dma_queue'].items()},
        "dve_hex": open(sg_dir / 'DVE0.bin', 'rb').read().hex(),
        "act_hex": open(sg_dir / 'Activation0.bin', 'rb').read().hex(),
        "sp_hex": open(sg_dir / 'SP0.bin', 'rb').read().hex(),
    }
    _json.dump(tpl, _SHAPE_TPL.open('w'))


def emit_matmul(M: int, K: int, N: int, outdir: Path, name: str = "matmul") -> Path:
    """Emit a matmul NEFF for (M,K,N) using the verified per-shape template set.

    Supported: K in {128,256,512}, M in {32,64}, N in {128,256} (6 buckets).
    Streams byte-copy from the compiler oracle for the bucket; container
    (header/tar/jsons/DGE) is emitted by tools/neff_emit.py.
    """
    import struct as _s
    key = f"{M}_{K}_{N}"
    tpl = _json.loads(_SHAPE_TPL.read_text())
    if key not in tpl:
        _generate_oracle_on_box(M, K, N, key)
        tpl = _json.loads(_SHAPE_TPL.read_text())
        if key not in tpl:
            raise ValueError(f"oracle generation failed for {key}")
    t = tpl[key]
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    sg = out / "sg00"
    sg.mkdir(parents=True, exist_ok=True)
    (sg / "PE0.bin").write_bytes(bytes.fromhex(t["pe_hex"]))
    (sg / "Pool0.bin").write_bytes(bytes.fromhex(t["pool_hex"]))
    (sg / "DVE0.bin").write_bytes(bytes.fromhex(t["dve_hex"]))
    (sg / "Activation0.bin").write_bytes(bytes.fromhex(t["act_hex"]))
    (sg / "SP0.bin").write_bytes(bytes.fromhex(t["sp_hex"]))
    for j in ("PE0.json","Pool0.json","DVE0.json","Activation0.json","SP0.json",
              "act_info.json","exp_and_others.json","exp_and_others_bkt.bin",
              "exp_and_others_ctrl.bin","default_opcode_table.bin",
              "default_control_table.bin","default_datapath_table.bin"):
        src = Path("/tmp/opencode/mm_x/sg00") / j
        if src.exists():
            shutil.copy(src, sg / j)
    ivn, ifn, isz, ivid = t["identity_var"]
    A, B, Y = M*K*4, K*N*4, M*N*4
    d = {
        "#rank_world_size": 0,
        "act": "Activation0.json", "act_asm_dbg": [], "act_dbg": [],
        "act_instr": "Activation0.bin",
        "cc_streams": {"num_streams": 0}, "debug_info": {}, "dma_dbg": [],
        "dma_queue": {q: ({"dynamic_dma": {"SB_scratch_partition_offset": 0,
                                            "SB_scratch_partition_size": 16384},
                            "fabric_path": "main", "num_queues": 16, "owner": "pool",
                            "semaphore_set": sets, "type": "dynamic"}
                          if q == "qPoolDynamic" else
                          {"fabric_path": "main", "num_queues": 16, "owner": q.startswith("qSP") and "sp" or "pool",
                           "semaphore_set": sets, "type": "data"})
                      for q, sets in t["queue_sets"].items()},
        "dve": "DVE0.json", "dve_asm_dbg": [], "dve_dbg": [], "dve_instr": "DVE0.bin",
        "fp8_conversion_config": {"emax_fp8_e4m3": 7, "emax_fp8_e5m2": 14,
                                   "fp8_range_extension": True, "fp8_sis": False,
                                   "fp8_sns": False, "ocp_sat": True},
        "name": "definition", "neff_features": [],
        "pe": "PE0.json", "pe_asm_dbg": [], "pe_dbg": [], "pe_instr": "PE0.bin",
        "pool": "Pool0.json", "pool_asm_dbg": [], "pool_dbg": [], "pool_instr": "Pool0.bin",
        "runtime_event_count": 12, "runtime_semaphore_count": 3,
        "sp": "SP0.json", "sp_asm_dbg": [], "sp_dbg": [], "sp_instr": "SP0.bin",
        "var": {"SB": {"type": "state-buffer", "var_id": 0},
                "input0": {"ops": [], "size": A, "type": "input", "var_id": 1},
                "input1": {"ops": [], "size": B, "type": "input", "var_id": 2},
                "output0": {"ops": [], "size": Y, "type": "output", "var_id": 3},
                ivn: {"alignment": 65536, "fabric_path": "main", "file_name": ifn,
                       "ops": [], "size": isz, "type": "file", "var_id": ivid},
                "DGETable0": {"list": [1,2,3], "type": "dge-table", "var_id": 5}},
        "version": "0.6",
    }
    (sg / "def.json").write_text(_json.dumps(d, indent=1))
    sp = {"dma": t["sp_dma"], "engine_id": 0, "instr": "SP0.bin", "name": "SP0.json"}
    (sg / "SP0.json").write_text(_json.dumps(sp))
    import numpy as _np, io as _io
    idata = _np.eye(128, dtype=_np.float32).reshape(128, 128, 1, 1)
    buf = _io.BytesIO()
    _np.save(buf, idata)  # verified byte-identical to compiler output
    (sg / ifn).write_bytes(buf.getvalue())
    (out / "info.json").write_text(_json.dumps({
        "tool_version": "behemoth-reverse-0.1", "kelp_version": "Dynamic",
        "name": f"{name}.neff", "num_tpb": 1, "num_tpb_requested": 1,
        "num_tpb_per_node": [1] + [0]*63}))
    nj = {
        "nodes": [
            {"op": "null", "name": "input0", "inputs": [], "output_names": ["input0"], "is_param": "0"},
            {"op": "null", "name": "input1", "inputs": [], "output_names": ["input1"], "is_param": "0"},
            {"op": "tvm_op", "name": "sg_coreV10",
             "attrs": {"func_name": "__kelf", "flatten_data": "0", "num_inputs": "2",
                        "kelf": "kelf-0.json", "num_outputs": "1"},
             "inputs": [[0,0,0],[1,0,0]], "output_names": ["output0"]}],
        "arg_nodes": [0,1], "heads": [[2,0,0]],
        "attrs": {"shape": ["list_shape", [[M,K],[K,N],[M,N]]],
                   "dltype": ["list_str", ["float32"]*3],
                   "storage_id": ["list_int", [0,1,2]]},
        "node_row_ptr": [0,1,2,3],
        "metadata": {"signatures": {"default": {
            "inputs": {"input0": {"id": 0, "dtype": "float32", "shape": [M,K]},
                        "input1": {"id": 1, "dtype": "float32", "shape": [K,N]}},
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


def emit_matmul_synth(outdir: Path, M: int, K: int, N: int,
                      name: str = "matmul") -> Path:
    """Emit a matmul NEFF for arbitrary (M,K,N) via mm_synth synthesis.

    PE0.bin = mm_synth.synth (byte-exact model vs 18 oracle streams),
    Pool0.bin = mm_synth.synth_pool, SP/DVE = universal templates.
    ACTIVATE stream = nearest oracle bucket by (K-class, NT).
    Literal tables (d0/d1 descs) fall back to interpolated formulas.
    """
    import mm_synth as _ms
    key = f"{M}_{K}_{N}"
    tpls = _json.loads(_SHAPE_TPL.read_text())
    NT = (N + 127) // 128
    kc = 0 if K <= 128 else (1 if K <= 512 else 2)
    # nearest bucket: same K-class and NT, else any with matching NT then K-class
    def score(bm, bk, bn):
        bnt = (bn + 127) // 128
        bkc = 0 if bk <= 128 else (1 if bk <= 512 else 2)
        return (abs(bnt - NT), abs(bkc - kc), abs(bm - M))
    best = min(tpls, key=lambda k: score(*map(int, k.split("_"))))
    t = tpls[best]
    if key in tpls:
        # exact-shape oracle exists: byte-copy its full stream set. The
        # compiler SB allocation varies per compile (fresh K384/N384 put w
        # at 0x18000) and the oracle variant is device-proven; synth() is
        # the path for shapes outside the oracle set.
        return emit_matmul(M, K, N, outdir, name)
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    sg = out / "sg00"
    sg.mkdir(parents=True, exist_ok=True)
    (sg / "PE0.bin").write_bytes(_ms.synth(M, K, N))
    (sg / "Pool0.bin").write_bytes(_ms.synth_pool(M, K, N))
    (sg / "DVE0.bin").write_bytes(bytes.fromhex(t["dve_hex"]))
    (sg / "Activation0.bin").write_bytes(bytes.fromhex(t["act_hex"]))
    (sg / "SP0.bin").write_bytes(bytes.fromhex(t["sp_hex"]))
    for j in ("PE0.json", "Pool0.json", "DVE0.json", "Activation0.json",
              "SP0.json", "act_info.json", "exp_and_others.json",
              "exp_and_others_bkt.bin", "exp_and_others_ctrl.bin",
              "default_opcode_table.bin", "default_control_table.bin",
              "default_datapath_table.bin"):
        srcp = Path("/tmp/opencode/mm_x/sg00") / j
        if srcp.exists():
            shutil.copy(srcp, sg / j)
    ivn, ifn, isz, ivid = t["identity_var"]
    A, B, Y = M * K * 4, K * N * 4, M * N * 4
    sets = t["queue_sets"]
    pv = t["sp_dma"]
    # identity refill desc: [512 if pv-kind else 4M, 128] rows, to_off = tof
    kind, fdst, fdn = _ms.final_scheme(M, K, N)
    is_pv = kind == "pv" or (kind == "col" and len(fdst) > 1)
    rows = 512 if is_pv else 4 * M
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
    sp = {"dma": t["sp_dma"], "engine_id": 0, "instr": "SP0.bin",
          "name": "SP0.json"}
    # patch identity desc sizes for the new shape
    import copy as _copy
    sp = _copy.deepcopy(sp)
    for desc in sp["dma"]:
        for dd in desc.get("desc", []):
            if str(dd.get("to_off", 0)).startswith("196") or dd.get("to") == "SB":
                dd["from_sizes"] = [rows, 128]
                dd["to_sizes"] = [rows, 128]
    (sg / "SP0.json").write_text(_json.dumps(sp))
    import numpy as _np, io as _io
    idata = _np.eye(128, dtype=_np.float32).reshape(128, 128, 1, 1)
    buf = _io.BytesIO()
    _np.save(buf, idata)
    (sg / ifn).write_bytes(buf.getvalue())
    if nout > 1:
        # compiler tars carry no info.json; keep it for single-output wrappers
        pass
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
    """Emit y = scale*x*w elementwise NEFF over n f32 elements.

    Template = freshly compiled neuronx-cc 2.28 oracles (tools/ew_templates/),
    NOT the old ref26bb template: the old compiler's final ACTIVATE reads its
    scale from a reg/ptr that is never staged (per-boot garbage scale - the
    "0.6x drift"); the new template carries the scale as an inline f32 imm in
    the op170 continuation (@228), patchable per-call.

    Structure (decoded from 4 oracles n in {100, 2048, 6144, 8192}):
      P = ceil(n/128) DVE channel passes; per-channel count = P everywhere.
      n <= 128: template B (DVE op67 + op10, no activation engine pass);
      n > 128:  template A (DVE op65, ACTIVATE op33/op170 with f32 scale imm).
    All n-dependence is u16 - patched offsets verified against all 4 oracles.
    """
    TPL = Path(__file__).parent / "ew_templates"
    tpl = TPL / ("B" if n <= 128 else "A")
    if n <= 0 or (n > 128 and (n + 127) // 128 > 16383):
        raise ValueError(f"n={n} out of supported range (1..2098176)")
    P = (n + 127) // 128
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(tpl, out)
    sg = out / "sg00"

    def u16(buf, off, val):
        struct.pack_into("<H", buf, off, val & 0xFFFF)

    if n > 128:
        neg4P, neg8P = 65536 - 4 * P, 65536 - 8 * P
        dve = bytearray(open(sg / "DVE0.bin", "rb").read())
        assert dve[192] == 65  # op170 cont slot opcode byte varies (128/0)
        u16(dve, 192 + 26, P)
        u16(dve, 224 + 0, neg8P)
        u16(dve, 224 + 10, P)
        u16(dve, 224 + 26, P)
        (sg / "DVE0.bin").write_bytes(bytes(dve))
        act = bytearray(open(sg / "Activation0.bin", "rb").read())
        assert act[192] == 33 and act[224] == 170
        struct.pack_into("<f", act, 224 + 4, scale)
        u16(act, 192 + 16, neg4P)
        u16(act, 192 + 26, P)
        u16(act, 224 + 26, P)
        (sg / "Activation0.bin").write_bytes(bytes(act))
        pool = bytearray(open(sg / "Pool0.bin", "rb").read())
        assert pool[128] == 212 and pool[192] == 212 and pool[256] == 212
        u16(pool, 128 + 24, 4 * P)                       # x-in desc
        u16(pool, 160 + 4, 4 * P); u16(pool, 160 + 8, neg4P); u16(pool, 160 + 28, 4 * P)
        u16(pool, 192 + 24, 4 * P)                       # w-in desc
        u16(pool, 224 + 4, 4 * P); u16(pool, 224 + 8, neg8P); u16(pool, 224 + 28, 4 * P)
        u16(pool, 288 + 4, 4 * P)                        # y-out desc + rearm
        u16(pool, 288 + 16, 4 * P); u16(pool, 288 + 28, 4 * P)
        (sg / "Pool0.bin").write_bytes(bytes(pool))
    else:
        dve = bytearray(open(sg / "DVE0.bin", "rb").read())
        assert dve[224] == 10
        u16(dve, 224 + 2, n)
        struct.pack_into("<f", dve, 224 + 8, scale)  # scale imm inside op10
        (sg / "DVE0.bin").write_bytes(bytes(dve))
        pool = bytearray(open(sg / "Pool0.bin", "rb").read())
        assert pool[128] == 212 and pool[192] == 212 and pool[256] == 212
        for base in (160, 224, 288):                     # rearm pseudo: opcode=n
            u16(pool, base + 0, n); u16(pool, base + 24, n)
        (sg / "Pool0.bin").write_bytes(bytes(pool))

    nbytes = n * 4
    d = _json.loads((sg / "def.json").read_text())
    for vn in ("input0", "input1", "output0"):
        d["var"][vn]["size"] = nbytes
    (sg / "def.json").write_text(_json.dumps(d))
    if nout > 1:
        # compiler tars carry no info.json; keep it for single-output wrappers
        pass
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
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


SOFTMAX_ORACLE_DIRS = {  # (M, K) -> extracted compiler oracle sg00 dir
    (128, 128): "/tmp/opencode/oracle_softmax128/sg00",
}


def emit_softmax(outdir: Path, M: int, K: int, name: str = "softmax") -> Path:
    """Emit y = exp(x - max) / sum(exp(x - max)) rowwise, f32 [M, K].

    Compiler-lowering oracle byte-copy (DVE TENSOR_REDUCE_ARITH_OP(66) for
    max + sum, ACTIVATE(33) exp via LUT tables; exp_and_others_bkt +
    act_info ship with the oracle)."""
    if (M, K) not in SOFTMAX_ORACLE_DIRS:
        raise ValueError(f"no softmax oracle for ({M},{K})")
    sg_src = Path(SOFTMAX_ORACLE_DIRS[(M, K)])
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "sg00").mkdir(parents=True)
    for f in sg_src.iterdir():
        shutil.copy(f, out / "sg00" / f.name)
    if nout > 1:
        # compiler tars carry no info.json; keep it for single-output wrappers
        pass
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
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


DEQUANT_ORACLE_DIRS = {  # (M, K) -> extracted compiler oracle sg00 dir
    (128, 128): "/tmp/opencode/oracle_dequant128/sg00",
}


def emit_dequant(outdir: Path, M: int, K: int, name: str = "dequant") -> Path:
    """Emit y = q.cast(f32) * s[None, :] for q uint8 [M,K], s f32 [K].

    Compiler-lowering oracle byte-copy (cast happens on PE, scale on DVE).
    Input file order: in0 = q (uint8), in1 = s (f32 K), output = f32.
    """
    if (M, K) not in DEQUANT_ORACLE_DIRS:
        raise ValueError(f"no dequant oracle for ({M},{K})")
    sg_src = Path(DEQUANT_ORACLE_DIRS[(M, K)])
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "sg00").mkdir(parents=True)
    for f in sg_src.iterdir():
        shutil.copy(f, out / "sg00" / f.name)
    if nout > 1:
        # compiler tars carry no info.json; keep it for single-output wrappers
        pass
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
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


RMS_ORACLE_DIRS = {  # (M, K) -> extracted compiler oracle sg00 dir
    (128, 128): "/tmp/opencode/oracle_rms128/sg00",
    (256, 128): "/tmp/opencode/oracle_rms256/sg00",
}


def emit_rmsnorm(outdir: Path, M: int, K: int, eps: float = 1e-6,
                 name: str = "rmsnorm") -> Path:
    """Emit y = x * rsqrt(mean(x^2, axis=-1) + eps) for x [M, K] f32.

    Compiler-lowering oracle byte-copy (no PE: DVE MEMSET(73) +
    TENSOR_REDUCE_ARITH_OP(66) into PSUM bank0 via DVE marker 8, then a
    3-pass ACTIVATE(33) chain for sqrt/div/y-mul). eps = f32 @360 in
    Activation0.bin (verified: 1e-6 vs 1e-3 differ only there).
    Weighted rms (x * w * rs) = compose with emit_elementwise on the host.
    """
    import struct as _s
    if (M, K) not in RMS_ORACLE_DIRS:
        raise ValueError(f"no rms oracle for ({M},{K}); buckets: {list(RMS_ORACLE_DIRS)}")
    sg_src = Path(RMS_ORACLE_DIRS[(M, K)])
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "sg00").mkdir(parents=True)
    for f in sg_src.iterdir():
        shutil.copy(f, out / "sg00" / f.name)
    act = bytearray((out / "sg00" / "Activation0.bin").read_bytes())
    _s.pack_into("<f", act, 360, eps)
    (out / "sg00" / "Activation0.bin").write_bytes(bytes(act))
    if nout > 1:
        # compiler tars carry no info.json; keep it for single-output wrappers
        pass
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
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"

# ---------------- generalized on-box oracle generation ----------------

_SSM_RUN = "/tmp/opencode/ssm_run.sh"


def gen_oracle_custom(key: str, gen_py: str, timeout: int = 300) -> Path:
    """Compile an arbitrary jax snippet on the trn1 box and extract sg00.

    gen_py must write <mlirbc> to /tmp/oracle_gen_<key>/graph.mlirbc.
    Chunked transfer with per-chunk length validation + refill rounds
    (SSM stdout drops chunks; first line carries a Success/Failed prefix).
    Returns the extracted sg00 dir.
    """
    open("/tmp/opencode/_gc.b64", "w").write(base64.b64encode(gen_py.encode()).decode())
    subprocess.run(["aws", "s3", "cp", "/tmp/opencode/_gc.b64",
                    "s3://912935854432-neuron-build/tools/gc.py.b64"],
                   check=True, capture_output=True)
    inner = (
        "source /home/ubuntu/work/nenv.sh 2>/dev/null\n"
        "export PATH=/opt/aws_neuronx_venv_jax_0_9/bin:/usr/local/bin:$PATH\n"
        "cd /tmp\n"
        "aws s3 cp s3://912935854432-neuron-build/tools/gc.py.b64 /tmp/gc.b64 1>/dev/null\n"
        "python3 -c \"import base64; open('/tmp/gc.py','wb').write(base64.b64decode(open('/tmp/gc.b64').read()))\"\n"
        f"/opt/aws_neuronx_venv_jax_0_9/bin/python /tmp/gc.py\n"
        f"cd /tmp/oracle_gen_{key} && neuronx-cc compile graph.mlirbc --framework XLA --target trn1 --output o.neff 2>&1 | tail -1\n"
        "cat > /tmp/fb3.py <<'PYEOF'\n"
        "import gzip as gz, tarfile, io, base64, json\n"
        f"d = open('/tmp/oracle_gen_{key}/o.neff','rb').read()\n"
        "t = tarfile.open(fileobj=io.BytesIO(gz.decompress(d[1024:])))\n"
        "out = {}\n"
        "for m in t.getmembers():\n"
        "    if m.name.startswith('sg00/') and all(x not in m.name for x in ('debug',)):\n"
        "        out[m.name.split('/')[-1]] = base64.b64encode(t.extractfile(m).read()).decode()\n"
        "blob = base64.b64encode(gz.compress(json.dumps(out).encode(), 9)).decode()\n"
        "open('/tmp/rmsblob.b64','w').write(blob)\n"
        "print('NB:' + str(len(blob)))\n"
        "PYEOF\n"
        "/opt/aws_neuronx_venv_jax_0_9/bin/python /tmp/fb3.py")
    open('/tmp/opencode/_gc_inner.sh', 'w').write(inner)
    m = None
    r = None
    for attempt in range(3):
        r = subprocess.run(["bash", _SSM_RUN, '/tmp/opencode/_gc_inner.sh', str(timeout)],
                           capture_output=True, text=True, timeout=timeout + 30)
        m = re.search(r'NB:(\d+)', r.stdout)
        if m:
            break
    assert m, f"gen failed {key}: {r.stdout[-300:] if r else 'no run'}"
    nb, n = int(m.group(1)), (int(m.group(1)) + 7999) // 8000
    have = {}
    for rnd in range(15):
        missing = [i for i in range(n) if len(have.get(i, '')) != min(8000, nb - i * 8000)]
        if not missing:
            break
        fetch = ("cat > /tmp/fc2.py <<'PYEOF'\nimport base64\n"
                 "blob = open('/tmp/rmsblob.b64').read()\nCH = 8000\nfor i in ["
                 + ",".join(map(str, missing)) + "]:\n"
                 "    print(f'NCH:{i}:' + blob[i*CH:(i+1)*CH])\nPYEOF\n"
                 "/opt/aws_neuronx_venv_jax_0_9/bin/python /tmp/fc2.py\n")
        open('/tmp/opencode/_gc_fetch.sh', 'w').write(fetch)
        r2 = subprocess.run(["bash", _SSM_RUN, '/tmp/opencode/_gc_fetch.sh', '120'],
                            capture_output=True, text=True, timeout=150)
        for line in r2.stdout.split('\n'):
            m2 = re.search(r'NCH:(\d+):(.*)$', line)
            if m2:
                i, sg = int(m2.group(1)), m2.group(2)
                if len(sg) == min(8000, nb - i * 8000):
                    have[i] = sg
    assert len(have) == n, f"chunks {sorted(have)}/{n}"
    blob = ''.join(have[i] for i in sorted(have))
    data = json.loads(gzip.decompress(base64.b64decode(blob)))
    sg = Path(f"/tmp/opencode/oracle_{key}/sg00")
    sg.mkdir(parents=True, exist_ok=True)
    for nm, b in data.items():
        (sg / nm).write_bytes(base64.b64decode(b))
    return sg


def _wrap_graph(outdir: Path, sg_src: Path, inputs, name: str = "graph",
                out_shape=None) -> Path:
    """inputs = [(dtype, [dims])...] in arg order; builds wrapper + neff."""
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "sg00").mkdir(parents=True)
    for f in sg_src.iterdir():
        shutil.copy(f, out / "sg00" / f.name)
    nin = len(inputs)
    nodes = [{"op": "null", "name": f"input{i}", "inputs": [],
              "output_names": [f"input{i}"], "is_param": "0"} for i in range(nin)]
    sg_idx = nin
    nout = len(out_shape) if (out_shape and isinstance(out_shape[0], list)) else 1
    nodes.append({"op": "tvm_op", "name": "sg_coreV10",
                  "attrs": {"func_name": "__kelf", "flatten_data": "0",
                            "num_inputs": str(nin), "kelf": "kelf-0.json",
                            "num_outputs": str(nout)},
                  "inputs": [[i, 0, 0] for i in range(nin)],
                  "output_names": [f"output{i}" for i in range(nout)]})
    if nout > 1:
        # compiler tars carry no info.json; keep it for single-output wrappers
        pass
    shapes = [[d for d in dims] for _, dims in inputs]
    outs = out_shape if (out_shape and isinstance(out_shape[0], list)) else [out_shape or list(inputs[0][1])]
    shapes.extend(outs)
    nj = {"nodes": nodes, "arg_nodes": list(range(nin)),
          "heads": [[sg_idx, i, 0] for i in range(len(outs))],
          "attrs": {"shape": ["list_shape", shapes],
                    "dltype": ["list_str", [dt for dt, _ in inputs] + ["float32"] * len(outs)],
                    "storage_id": ["list_int", list(range(nin + len(outs)))]},
          "node_row_ptr": list(range(nin + 1)) + [nin + len(outs)],
          "metadata": {"signatures": {"default": {
              "inputs": {f"input{i}": {"id": i, "dtype": dt, "shape": dims}
                         for i, (dt, dims) in enumerate(inputs)},
              "outputs": {}}}}}
    shapes[-1] = _json.loads(_json.dumps(shapes[-1]))
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


def _jax_gen(body: str) -> str:
    return ("import io, os\nimport jax, jax.numpy as jnp\n"
            "d = os.environ['ODIR']\nos.makedirs(d, exist_ok=True)\n"
            + body +
            "\nbuf = io.BytesIO()\nlowered.compiler_ir().operation.write_bytecode(buf)\n"
            "open(d + '/graph.mlirbc','wb').write(buf.getvalue())\nprint('gen ok')\n")


def emit_rmsnorm_gen(outdir: Path, M: int, K: int, eps: float = 1e-6,
                     name: str = "rmsnorm") -> Path:
    """Arbitrary-shape RMSNorm via on-box oracle generation + eps patch."""
    key = f"rms_{M}x{K}"
    body = (f"def f(a):\n"
            f"    return a * (1.0 / jnp.sqrt(jnp.mean(a*a, axis=-1, keepdims=True) + {eps!r}))\n"
            f"lowered = jax.jit(f).lower(jnp.zeros(({M},{K}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    # eps is baked into the compiled graph via codegen; do NOT patch ACT
    # (@360 is a bucket-path-only offset and corrupts fresh layouts).
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K])], name)


def emit_softmax_gen(outdir: Path, M: int, K: int, name: str = "softmax") -> Path:
    key = f"softmax_{M}x{K}"
    body = (f"lowered = jax.jit(lambda a: jax.nn.softmax(a, axis=-1)).lower("
            f"jnp.zeros(({M},{K}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K])], name)


def emit_dequant_gen(outdir: Path, M: int, K: int, name: str = "dequant") -> Path:
    key = f"dequant_{M}x{K}"
    body = (f"lowered = jax.jit(lambda q, s: q.astype(jnp.float32) * s[None, :]).lower("
            f"jnp.zeros(({M},{K}), jnp.uint8), jnp.zeros(({K},), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("uint8", [M, K]), ("float32", [K])], name)


def emit_rmsnorm_fused(outdir: Path, M: int, K: int, eps: float = 1e-6,
                       name: str = "rmsnorm_fused") -> Path:
    """Weighted RMSNorm y = x * g * rsqrt(mean(x^2)+eps) in ONE graph."""
    key = f"rmsw_{M}x{K}"
    body = (f"def f(a, g):\n"
            f"    return a * g * (1.0 / jnp.sqrt(jnp.mean(a*a, axis=-1, keepdims=True) + {eps!r}))\n"
            f"lowered = jax.jit(f).lower(jnp.zeros(({M},{K}), jnp.float32), "
            f"jnp.zeros(({K},), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K]), ("float32", [K])], name)


def emit_matmul_gen(outdir: Path, M: int, K: int, N: int, name: str = "matmul") -> Path:
    """Arbitrary-shape matmul y = x[M,K] @ w[K,N] via on-box oracle gen."""
    key = f"mm_{M}x{K}x{N}"
    body = (f"lowered = jax.jit(lambda a, b: a @ b).lower("
            f"jnp.zeros(({M},{K}), jnp.float32), jnp.zeros(({K},{N}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K]), ("float32", [K, N])], name,
                       out_shape=[M, N])


def emit_sdpa_gen(outdir: Path, h: int, M: int, S: int, d: int, scale: float,
                  name: str = "sdpa") -> Path:
    """Attention through proven paths: y = softmax(q@k^T*scale)@v per head.
    q[h,M,d], k[h,S,d], v[h,S,d] -> y[h,M,d]. Single graph, on-box gen."""
    key = f"sdpa_{h}_{M}_{S}_{d}"
    body = (f"def f(q, k, v):\n"
            f"    a = jnp.einsum('hmd,hsd->hms', q, k) * {scale!r}\n"
            f"    p = jax.nn.softmax(a, axis=-1)\n"
            f"    return jnp.einsum('hms,hsd->hmd', p, v)\n"
            f"lowered = jax.jit(f).lower(jnp.zeros(({h},{M},{d}), jnp.float32), "
            f"jnp.zeros(({h},{S},{d}), jnp.float32), jnp.zeros(({h},{S},{d}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg,
                       [("float32", [h, M, d]), ("float32", [h, S, d]), ("float32", [h, S, d])],
                       name, out_shape=[h, M, d])


def emit_block_gen(outdir: Path, M: int, K: int, H: int, h: int, d: int,
                   eps: float = 1e-6, name: str = "block") -> Path:
    """Full decoder block in ONE graph:
    y = ffn(rms(x + o(sdpa(rms(x)*g1) @ wo)) * g2)
    qkv fused proj, h=h heads of dim d (K = h*d for qkv split).
    Inputs: x[M,K] g1[K] wqkv[K,3K] wo[K,K] w1[K,H] w2[H,K] g2[K]
    """
    key = f"block_{M}_{K}_{H}_{h}_{d}"
    body = (
        f"def f(x, g1, wqkv, wo, w1, w2, g2):\n"
        f"    def rms(v, g):\n"
        f"        return v * g * (1.0 / jnp.sqrt(jnp.mean(v*v, axis=-1, keepdims=True) + {eps!r}))\n"
        f"    qkv = rms(x, g1) @ wqkv\n"
        f"    q, k, v = qkv[..., :{K}], qkv[..., {K}:2*{K}], qkv[..., 2*{K}:]\n"
        f"    q = q.reshape({M}, {h}, {d}).transpose(1, 0, 2)\n"
        f"    k = k.reshape({M}, {h}, {d}).transpose(1, 0, 2)\n"
        f"    v = v.reshape({M}, {h}, {d}).transpose(1, 0, 2)\n"
        f"    a = jnp.einsum('hmd,hnd->hmn', q, k) * {1.0 / d**0.5:.8f}\n"
        f"    p = jax.nn.softmax(a, axis=-1)\n"
        f"    o = jnp.einsum('hmn,hnd->hmd', p, v)\n"
        f"    o = o.transpose(1, 0, 2).reshape({M}, {K}) @ wo\n"
        f"    t = rms(x + o, g2)\n"
        f"    return (jax.nn.gelu(t @ w1) @ w2)\n"
        f"lowered = jax.jit(f).lower(jnp.zeros(({M},{K}), jnp.float32), jnp.zeros(({K},), jnp.float32), "
        f"jnp.zeros(({K},{3*K}), jnp.float32), jnp.zeros(({K},{K}), jnp.float32), "
        f"jnp.zeros(({K},{H}), jnp.float32), jnp.zeros(({H},{K}), jnp.float32), jnp.zeros(({K},), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg,
                       [("float32", [M, K]), ("float32", [K]), ("float32", [K, 3 * K]),
                        ("float32", [K, K]), ("float32", [K, H]), ("float32", [H, K]),
                        ("float32", [K])],
                       name, out_shape=[M, K])


def emit_be_block_gen(outdir: Path, K: int = 12288, hq: int = 96, hkv: int = 8,
                      d: int = 128, H: int = 28672, theta: float = 1e6,
                      eps: float = 1e-5, S: int = 1, name: str = "beblock") -> Path:
    """Behemoth decode block (runtime/src arch): post-norm, RoPE pair(d,d+64)
    theta 1e6, GQA, SwiGLU, RMSNorm eps 1e-5. M=1 decode step w/ KV cache.
    Inputs: x[K] w_in[K] wq[K,hq*d] wk[K,hkv*d] wv[K,hkv*d] wo[hq*d,K]
            w_post[K] wg[K,H] wu[K,H] wd[H,K] kc[hkv,S,d] vc[hkv,S,d] pos(i32)
    Outputs: y[K] k_new[hkv,d] v_new[hkv,d]
    """
    key = f"beblk_{K}_{hq}_{hkv}_{d}_{H}_{S}"
    body = (
        f"def f(x, w_in, wq, wk, wv, wo, w_post, wg, wu, wd, kc, vc, pos):\n"
        f"    def rms(v, g):\n"
        f"        return v * g * (1.0 / jnp.sqrt(jnp.mean(v*v) + {eps!r}))\n"
        f"    xn = rms(x, w_in)\n"
        f"    q = (xn @ wq).reshape({hq}, {d})\n"
        f"    k = (xn @ wk).reshape({hkv}, {d})\n"
        f"    v = (xn @ wv).reshape({hkv}, {d})\n"
        f"    inv = {theta!r} ** (-jnp.arange({d//2}) * 2.0 / {d})\n"
        f"    ang = pos[0] * inv\n"
        f"    c, s = jnp.cos(ang), jnp.sin(ang)\n"
        f"    def rope(t):\n"
        f"        a, b = t[:, :{d//2}], t[:, {d//2}:]\n"
        f"        return jnp.concatenate([a*c - b*s, b*c + a*s], -1)\n"
        f"    q, k = rope(q), rope(k)\n"
        f"    Kall = jnp.concatenate([kc, k[:, None]], 1)\n"
        f"    Vall = jnp.concatenate([vc, v[:, None]], 1)\n"
        f"    Kr = jnp.repeat(Kall, {hq//hkv}, 0)\n"
        f"    Vr = jnp.repeat(Vall, {hq//hkv}, 0)\n"
        f"    sc = jnp.einsum('md,msd->ms', q, Kr) * {1.0/d**0.5:.8f}\n"
        f"    p = jax.nn.softmax(sc, -1)\n"
        f"    o = (jnp.einsum('ms,msd->md', p, Vr).reshape({hq*d},) @ wo)\n"
        f"    res = x + o\n"
        f"    t = rms(res, w_post)\n"
        f"    g = t @ wg\n"
        f"    u = t @ wu\n"
        f"    h = g / (1.0 + jnp.exp(-g)) * u\n"
        f"    return res + h @ wd, k, v\n"
        f"lowered = jax.jit(f).lower(\n"
        f"    jnp.zeros(({K},), jnp.float32), jnp.zeros(({K},), jnp.float32),\n"
        f"    jnp.zeros(({K},{hq*d}), jnp.float32), jnp.zeros(({K},{hkv*d}), jnp.float32),\n"
        f"    jnp.zeros(({K},{hkv*d}), jnp.float32), jnp.zeros(({hq*d},{K}), jnp.float32),\n"
        f"    jnp.zeros(({K},), jnp.float32), jnp.zeros(({K},{H}), jnp.float32),\n"
        f"    jnp.zeros(({K},{H}), jnp.float32), jnp.zeros(({H},{K}), jnp.float32),\n"
        f"    jnp.zeros(({hkv},{S},{d}), jnp.float32), jnp.zeros(({hkv},{S},{d}), jnp.float32),\n"
        f"    jnp.zeros((1,), jnp.int32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py, timeout=600)
    ins = [("float32", [K]), ("float32", [K]), ("float32", [K, hq * d]),
           ("float32", [K, hkv * d]), ("float32", [K, hkv * d]),
           ("float32", [hq * d, K]), ("float32", [K]), ("float32", [K, H]),
           ("float32", [K, H]), ("float32", [H, K]),
           ("float32", [hkv, S, d]), ("float32", [hkv, S, d]), ("int32", [1])]
    return _wrap_graph(Path(outdir), sg, ins, name,
                       out_shape=[[K], [hkv, d], [hkv, d]])
