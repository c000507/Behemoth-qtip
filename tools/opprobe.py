#!/usr/bin/env python3
"""op123 TENSOR_DEQUANTIZE + op104 GATHER existence probes.

Base = validated 1-input shr container (bvw_shr): x (4096 f32) staged by
pool desc0 into DVE-0 region (src addr 0), y drained by desc1 from 0x4000.

op123 (v3.NEURON_ISA_TPB_S3D3_TENS_DEQUANT_STRUCT): 64B instr, wl=16.
  validator constraints (libnrt is_valid_s3d3_tens_dequant): opcode>=123
  (or 1..4), wl==0x10, num_active_channels>0, dequant_fmt in 1..4,
  group_size in {0..7,8,...}; events valid.
  Layout guess = op83-head + dtype block @32-37 + dst@44 (+ i_p dst@48).

op104 GATHER: no DWARF struct (only DMA_GATHER_XPOSE = op241 desc).
  Layout guess modeled on op123/DMA patterns; idx = x region itself
  (values 0..96 are valid in-bounds indices) -> expected y = x[x].
"""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bv_pscale import ORACLE, pu16, patch_pool_P  # reuses validated base path

UPD = (6 << 8) | 19
WAIT = (8 << 8) | 5


def op123_instr(src, dst, P, in_dt, out_dt, fmt, group, dst_at_48=False):
    s = bytearray(64)
    s[0], s[1] = 123, 16
    pu16(s, 4, WAIT)
    pu16(s, 6, UPD)
    pu16(s, 8, 16)
    struct.pack_into("<I", s, 12, src)
    pu16(s, 16, 0xFFF0)
    pu16(s, 18, 2)
    for off in (20, 22, 24):
        pu16(s, off, 1)
    pu16(s, 26, P)
    pu16(s, 28, 1); pu16(s, 30, 1)
    s[32] = in_dt; s[33] = out_dt; s[34] = 32; s[35] = fmt; s[36] = group
    if dst_at_48:
        struct.pack_into("<I", s, 48, dst)
    else:
        struct.pack_into("<I", s, 44, dst)
    pu16(s, 58, P)
    return bytes(s)


def op104_instr(src, idx, dst, P, elem=4, dtype=10):
    s = bytearray(64)
    s[0], s[1] = 104, 16
    pu16(s, 4, WAIT)
    pu16(s, 6, UPD)
    pu16(s, 8, 16)
    s[12] = 0            # semaphore
    s[13] = 0            # dma_configs
    s[14] = P & 0xFF     # idx_num_active_channels
    s[15] = 0            # src_idx_bound
    struct.pack_into("<I", s, 16, src)   # src_start_addr imm
    pu16(s, 20, 2)
    for off in (22, 24):
        pu16(s, off, 1)
    pu16(s, 26, P)
    pu16(s, 28, 1); pu16(s, 30, 1)
    struct.pack_into("<I", s, 36, idx)   # src_idx_start_addr imm
    struct.pack_into("<I", s, 40, dst)   # dst_start_addr imm
    pu16(s, 52, 1)       # dst_step_elem_1
    pu16(s, 60, elem)    # elem_size
    s[62] = dtype        # dtype
    s[63] = 0            # flags gather_dim
    pu16(s, 58, P)
    return bytes(s)


def build(out: Path, dve_ops: bytes):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ORACLE, out)
    sg = out / "sg00"
    dve = bytearray(open(sg / "DVE0.bin", "rb").read()[:4 * 32]) + dve_ops
    (sg / "DVE0.bin").write_bytes(bytes(dve))
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out.parent / f"{out.name}.neff"),
                    "--name", out.name], check=True, capture_output=True)
    return out.parent / f"{out.name}.neff"


def main():
    pd = Path("/tmp/opencode/opprobe")
    pd.mkdir(parents=True, exist_ok=True)
    made = []
    for fmt in (1, 2, 3, 4):
        made.append(build(pd / f"dq_f{fmt}_f32",
                          op123_instr(0, 0x4000, 32, 10, 10, fmt, 8)))
    made.append(build(pd / "dq_f1_i8", op123_instr(0, 0x4000, 32, 2, 10, 1, 8)))
    made.append(build(pd / "dq_f2_i8", op123_instr(0, 0x4000, 32, 2, 10, 2, 8)))
    made.append(build(pd / "dq_f1_ip", op123_instr(0, 0x4000, 32, 10, 10, 1, 8, dst_at_48=True)))
    made.append(build(pd / "gth_v1", op104_instr(0, 0, 0x4000, 32)))
    made.append(build(pd / "gth_v2", op104_instr(0, 0, 0x4000, 32, elem=4, dtype=2)))
    for m in made:
        print(m)


if __name__ == "__main__":
    main()
