# Project memory: NEFF reverse engineering (LLVM→NEFF compiler project)

State snapshot 2026-11-08 (late). Owner away — work autonomously, no questions. Resume = read this + docs/neff-format.md + docs/neff-isa-v2-fields.md.

## MILESTONE REACHED (all verified on-device)
Full reverse-engineering chain WORKS: NEFF container ↔ ISA decode ↔ ISA encode ↔ emit ↔ trn1 execute.
1. Container: unpack (tools/neff_unpack.py) + rebuild (tools/neff_emit.py) — rebuilt ffn1_full passes nrt_load on trn1, identical tensors.
2. ISA: complete v2 field maps extracted from neuron-dbg cgo accessors (tools/isa_fields.py → tools/neff-isa-fields.json, 4389 fields, 46 v2 structs).
3. Encoder: tools/neff_isa.py round-trips ALL 6144 MATMUL instrs in ffn1 PE0.bin byte-identical.
4. DEVICE CONTROL PROVEN: patched all 6144 matmuls (num_active_rows 128→64), re-emitted, ran on trn1 via neff_io2 harness — output differs from reference exactly as expected (all 4096 outputs differ).
5. Harness: /tmp/nrtd on box has neff_info + neff_io2 (generic single-rank runner, inputs by order dir/in0.., outputs dir/out0..). neff_io2.cpp source also at s3://912935854432-neuron-build/tools/neff_io2.b64. NOTE /tmp on box is wiped on reboot — rebuild harnesses from b64 files (info.c inline needed too, transport via b64 not heredoc).
6. OPCODE→STRUCT DISPATCH: op1=LDWEIGHTS→v2.S3_LW (6402/6402 byte-identical), op2=MATMUL→v2.S3D3_MM (6402/6402). DVE arith (65/66/67/107) + CTRL (160/204) v2 layouts NOT in accessor set — live in libnrt add_* builders (SP/CTRL done in docs/neff-isa-sp.md) and hwdecode tables (DVE TODO). v2-only real opcode = 212 (Pool, PSEUDO_DMA_DIRECT2D likely).
7. SYNTHETIC NEFF ON DEVICE: tools/tile_ir.py + tools/lift_hip.py emit a from-scratch graph (lifted tiled_projection) that passes nrt_load on trn1 — tensors w/x/y correct. Schema fixes learned: kelf-0.json graphs[].definition="sg00/def.json"; def.json needs var/dma_queue sets (qSPIO0 type=in sem 40-55, qPoolIO0 56-71, qPoolDynamic 8-23, qSPSpillReload0 24-39); IO inputs ARE valid DMA sources but only on IO queues; DVE tables must be embedded (copied from reference); neff.json needs one shape per node output.

## Goal
ROCm/HIP kernels (runtime/src/*.hip) → better Neuron kernels/graphs → direct NEFF emission, bypassing neuronx-cc frontends. Repo = BEHEMOTH-REVERSE.

## What is solved (verified, see docs/neff-format.md + docs/neff-isa.md + docs/neff-isa-sp.md)
- NEFF v2 = 1024B header + gzip(tar). Unpacker: tools/neff_unpack.py (tested). Digest A = md5(gz) word-swapped @0xAC.
- ISA BREAKTHROUGH: neuron-dbg (aws-neuronx-tools deb) + libndbg.so carry DWARF; tools/dwarf_extract.py + tools/dwarf_resolve.py dump 355 NEURON_ISA defs + 105 consts + full opcode enum (~180 ops) + DTYPE/WAIT_MODE/UPDATE_MODE/NOTIFIC_KIND enums (semaphore contract).
- Instruction = 32B, engines fetch 64B (INST_NBYTES=64). Header u8 opcode, u8 word_len(16), u8 debug_cmd, u8 debug_hint. PE array 128x128, 64 regs, 128 ch DVE.
- libnrt has 95 add_* builder fns (SP scalar assembler) — disassembled, byte layouts in docs/neff-isa-sp.md (64 builders decoded by subagent).
- v2-only opcodes exist: 8, 10, 128, 212 (not in common enum).
- GPSIMD DISCOVERY: trn1 has Xtensa Q7 cores; AWS ships aws-neuronx-gpsimd-customop-lib deb = full Xtensa GCC toolchain + C/C++ custom-op API (neuron_hbm_allocate, neuron_memcpy DMA, dataram alloc). libnrt gpsimd_build_ctx_* wraps nrtucode ll_builder. Sanctioned custom-kernel path #2 (besides NKI).
- nrtucode core types with ext-ISA libs: 6(1 lib, tonga/inf1?), 13(4 libs), 21(4), 29(4) — likely sunda=13, cayman=21, mariana=29. Getters take (int, out*) — crash w/o args.
- dlopen of libnrtucode_extisa.so WORKS on this box (earlier hang was my bad null-arg calls, not ctor). Assembler experiments possible locally!
- ISA ENCODER PROVEN: tools/neff_isa.py round-trips ALL 6144 MATMUL (S3D3_MM) instrs in ffn1 PE0.bin byte-identical. MEM_PATTERN layout: elem stride 16B {addr u32 @b, step u16 @b+4, num u16 @b+10, flags u16 @b+12,@b+14}; src base=16 dst base=48 (only dim0 fits in dst for MM). Remaining unmatched fields = u16 passthroughs.
- trn1 REACHABLE: /tmp/opencode/ssm_run.sh <script.sh> runs bash on box (aws ssm send-command + base64). Box has /home/ubuntu/work/nenv.sh (env), /home/ubuntu/codex-neuron-test/venv (nki 0.7, neuronx-cc 2.28), libnrt 2.33.10. Instance was STOPPED — I started it (i-0d79198d32a0df40f, running now). Old NOTES.md instance id is same one. PATH quirk: /usr/local/bin/aws not on nenv PATH, use export.
- VALIDATION LOOP PROVEN: upload NEFF via s3://912935854432-neuron-build/neff/ → box pulls → /tmp/nrtd/neff_info (built from inline info.c, links libnrt) → nrt_load OK, tensor list matches def.json.

## Key artifacts on disk
- /tmp/opencode/neff/ : 3 sample NEFFs + extracted *_x/ dirs (mh=flash_mh_m128_d64_h64, ffn1_full, flash_attn_m2048_d64). /tmp is volatile — re-pull from s3://912935854432-neuron-build/ if gone.
- /tmp/opencode/rtd/ : aws-neuronx-runtime-lib 2.35.30 (libnrt.so UNSTRIPPED 58k syms, headers include/nrt/, libnrtucode_extisa.so). Re-pull: apt.repos.neuron.amazonaws.com pool/main/a/aws-neuronx-runtime-lib + aws-neuron-runtime.
- /tmp/opencode/xtools/ : aws-neuronx-tools 2.33.38 deb — neuron-dbg (35MB, NOT stripped, WITH DWARF) = ISA SOURCE. /tmp/opencode/dbg.dwarf (78MB text) + /tmp/opencode/ndbg.dwarf (337MB) + /tmp/opencode/isa.json + isa-fields.json (4389 fields) + isa_layouts.md + isa_raw.md + addfns.txt (95 SP builders disasm) + libnrt.asm (214MB full disasm) + extisa.asm.
- /tmp/opencode/gpsimd/ : aws-neuronx-gpsimd-customop-lib 0.25.12 = Xtensa Q7 toolchain + custom_op.h API (hbm_allocate/memcpy DMA/dataram).
- /tmp/opencode/ssm_run.sh : runs bash script on trn1 box via `aws ssm send-command` + base64 pipe (session-manager-plugin NOT installed; send-command works). trn1 = i-0d79198d32a0df40f, /home/ubuntu/work/nenv.sh env wrapper.
- trn1 libnrt is 2.33.10 (older than local 2.35.30) — note version skew.

## Docs (all in repo, uncommitted)
- docs/neff-format.md — container spec (header, tar members, def.json, DMA JSON, consts).
- docs/neff-isa.md — instruction format, opcode enum (~180), DTYPE/WAIT/UPDATE/NOTIFIC enums, geometry consts, v2-only opcodes 8/10/128/212.
- docs/neff-isa-v2-fields.md — 46 v2 struct byte layouts (MATMUL/AC/DMA/PSEUDO etc) — THE encoding spec.
- docs/neff-isa-sp.md — 64 SP scalar builder byte layouts (subagent).
- tools/: neff_unpack.py, dwarf_extract.py, dwarf_resolve.py, isa_fields.py.

## Todo list (live in session tracker; mirror here)
1. [HIGH] Make synthetic lifted graph EXECUTE with correct numerics on trn1 (PE compute path: DMA staging order, PE addressing of w/x/y via mem patterns vs IO ring offsets, SP prologue/PSEUDO_TENSOR_STORE epilogue semantics)
2. [HIGH] Generalize encoder: DVE arith (65/66/67/107) + CTRL (160/204) v2 layouts via libnrt add_* disasm + hwdecode tables; then norm/silu lowering completes ops.hip subset
3. [HIGH] Semaphore scheduling contract details (runtime_semaphore_count semantics, which sems for PE<->DMA handoff)
4. [HIGH] Tile-loop IR generalization: lift more projection.hip/qtip.hip kernels (fused_projection, dequant GEMM), bounds/tail handling
5. [MED] GPSIMD/Xtensa path eval (custom C++ ops) as alternative for dequant/attention glue
6. [MED] Perf vs neuronx-cc/NKI baselines; collectives cc_streams for TP
7. [MED] LLVM/MLIR decision for scalar/control-flow frontend
8. NOTE: trn1 i-0d79198d32a0df40f STOPPED (I stopped it after validation). Start: aws ec2 start-instances --instance-ids i-0d79198d32a0df40f (~60s to SSM-ready)

## Committed (branch rocm-to-neff)
- d9285e7: RE foundation (format+ISA+encoder+device round-trip)
- 4bf59cf: tile-IR + lifter + first synthetic NEFF loading on device

## Conventions
- Caveman terse (per user config), but code/docs normal.
- Don't commit unless asked. Tools go in tools/, docs in docs/.
- Files staged uncommitted: docs/neff-format.md, tools/neff_unpack.py (git status shows A).
