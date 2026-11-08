# NEFF container format (reverse engineered)

Derived empirically from three compiled graphs (`flash_mh_m128_d64_h64.neff`,
`ffn1_full.neff`, `flash_attn_m2048_d64.neff`, compiled by neuronx-cc
2.26.6360.0 for trn1/"sunda") plus static analysis of `libnrt.so`
(aws-neuronx-runtime-lib 2.35.30) and `libnrtucode_extisa.so` from
https://apt.repos.neuron.amazonaws.com.

## Top level

```
NEFF file := header(1024 bytes) | gzip( tar archive )
```

Confirmed by `neff_parse` in libnrt (address 0x72f930 in 2.35.30): it calls
`archive_read_new` / `archive_read_support_format_tar` /
`archive_read_support_filter_gzip` / `archive_read_open_memory` (libarchive)
after computing SHA-256 and MD5 over the buffer. Payload gzip stream starts at
exactly offset 0x400.

### 1024-byte header (v2, little endian)

Observed fields (offsets from 0):

| offset | type | meaning (verified across 3 graphs) |
|--------|------|-------------------------------------|
| 0x00   | u32  | version = 2                         |
| 0x04   | u32  | 0                                   |
| 0x08   | u64  | 0x400 = payload (tar.gz) start offset / header size |
| 0x10   | u64  | payload byte size (file_size - 1024) |
| 0x18   | u64  | 2                                   |
| 0xA0   |      | zeros through 0xAB                  |
| 0xA8   | u32  | 1 (digest count)                    |
| 0xAC   | 16B  | MD5 of the gzip payload stream, stored as 4 little-endian-swapped u32 words (VERIFIED: matches md5(file[1024:]) on all 3 graphs) |
| 0xCC   | 16B  | MD5 digest B, same word-swapped encoding; source unknown (NOT md5 of: tar payload, name+size+data walk, header w/ digests zeroed, name, header+gz) |
| 0xDC   | char[64] | NUL-padded file name (plaintext) |
| 0x1D8  | u32  | 1                                   |
| 0x1E0  | u32  | 1                                   |
| 0x228  | u64  | 0x0010_0010_0000_0000 (seen in all 3) |

Rest of the 1024 bytes: zeros in these samples. Digest coverage not yet
pinned down (candidates: gzip stream, tar contents, header itself). All
remaining structure must be validated against libnrt's header parser before
writing an emitter.

`neff_get_header_from_buffer` (libnrt) only validates: buffer != null,
size > 0x3ff, and `*(u64*)(header+8) >= size` - i.e. the 1024-byte header is
mandatory, and field@0x8 is authoritative for total size.

## tar members

```
info.json                                   build metadata
neff.json                                   TVM graph-runtime JSON
kelf-0.json                                 per-kernel ("kelf" = kernel ELF-ish) config
metrics.json                                compiler metrics
debug_info/debug_info_penguin.dbg_sgNNNNNN  BIR debug: maps engine asm -> BIR -> HLO (protobuf w/ text strings)
sg00/def.json                               subgraph definition: engine files, DMA queues, semaphores, vars
sg00/tensor_map.json                        input/output/const tensors (name, dtype, layout, .npy file)
sg00/PE0.bin + PE0.json                     processing engine (tensor core) instruction stream
sg00/DVE0.bin + DVE0.json                   vector engine stream + dve_tables refs
sg00/Activation0.bin + Activation0.json     activation engine stream + act function set LUTs
sg00/Pool0.bin + Pool0.json                 DMA/pool engine stream
sg00/SP0.bin + SP0.json                     scalar engine stream + DMA descriptor JSON ("dma" array)
sg00/act_info.json                          activation function set table (bkt/ctl LUT layout)
sg00/debug_info_asm_<ENGINE>.dbg            protobuf: ordered asm-instruction -> BIR-instruction mapping
sg00/debug_info_backend_<ENGINE>.dbg        protobuf: backend-level debug
```

`sg00` = subgraph 0; multi-subgraph graphs use sg01, sg02... (one per
`num_tpb` tensor pipe block in info.json).

### info.json

```json
{"tool_version": "...", "kelp_version": "Dynamic", "name": "...",
 "num_tpb": 1, "num_tpb_requested": 1, "num_tpb_per_node": [1,0,...]}
```

### neff.json (TVM graph)

Standard TVM graph-runtime format: `nodes` (op "null" = placeholder/param,
`tvm_op` with `func_name: "__kelf"` and `kelf: "kelf-0.json"`), `arg_nodes`,
`heads`, attrs `shape`/`dltype`/`storage_id`, `node_row_ptr`,
`metadata.signatures` with named inputs/outputs incl. dtype+shape.

### sg00/def.json (the loader script)

Key fields:
- `var`: named memory objects with `type` in {input, output, file, state-buffer,
  dge-table}, `var_id`, size, alignment, fabric_path. `SB` = on-chip state
  buffer (SBUF), `DGETable0` = DMA gather-engine descriptor table.
- `dma_queue`: named queues (`qSPSpillReload0` data, `qPoolDynamic` dynamic) with
  owner engine, `semaphore_set` ids, num_queues.
- per-engine pointers: `pe`/`dve`/`act`/`sp`/`pool` -> their JSON + `*_instr` bin +
  `*_asm_dbg`/`*_dbg` files.
- `fp8_conversion_config`, `runtime_semaphore_count`, `version: "0.6"`.

### DMA descriptors (sgNN/<ENGINE>.json `"dma"` array)

Data movement is NOT machine code - it is JSON:

```json
{"id":0, "queue":"qSPSpillReload0", "semaphore":8,
 "desc":[{"op":"copy",
          "from":"identity_3659_sg0000","from_dtype":"float32",
          "from_off":0,"from_sizes":[512,128],"from_steps":[1,512],
          "to":"SB","to_dtype":"float32","to_off":196096,
          "to_sizes":[512,128],"to_steps":[1,262144]}],
 "#comment":"I-3663-0-block-1"}
```

`from`/`to` name entries in `var` (HBM tensors or SB). `*_off` byte offsets,
`*_sizes`/`*_steps` strided tile shape. Runtime compiles these into DMA
descriptors at load.

### Engine instruction streams (PE0.bin etc.)

Every stream = [64B prologue: opcode 177 BRANCH_PREFETCH_HINT, inst_word_len 16,
rest zero] followed by 32-byte instructions (pairs per 64B line). No explicit
HALT/EXIT terminator - execution stops at stream end. Verified identical
prologue across PE/DVE/Pool/Activation/SP in all three sample graphs.

Runtime-execution preamble (ib_add_neff_exec_preamble in libnrt, appended by
the runtime at exec, not present in the NEFF): add_notify, add_dma_tail_inc,
ev_wait(ioq code switch), add_sync_barrier, add_args_table_swap.

### Emitter status (tools/neff_emit.py)

Byte-level NEFF rebuild from unpacked members verified: rebuilt ffn1_full
(header+gzip(tar)+md5-A digest, digest B zeroed, tail zeroed) passes
`nrt_load` on trn1 (libnrt 2.33.10) with identical tensor table. Tail fields
and digest B not required by the loader in this configuration.

### Activation engine

Pure LUT machine: `act_info.json` picks function sets (exp/sigmoid/tanh/gelu/...)
whose `*_bkt.bin`/`*_ctrl.bin` bucket tables also ship in the runtime, not the
NEFF. `exp_and_others.json` shows the per-function control record layout
(func_id, PWL control indices, exp_offset, thresholds, special-case results).

## Everything needed to (a) parse, (b) emit NEFF is present

1. Parser: header + tar.gz + JSON members (done, see tools/neff_unpack.py).
2. Graph level: emit valid `neff.json` + `info.json` + `def.json` + DMA
   descriptor JSON. All JSON schemas above.
3. Engine streams: either (a) call `nrtucode_ll_builder_*` from the runtime to
   assemble, or (b) decode ISA from libnrt/hwdecode tables and write own
   assembler. (a) is the sanctioned path; (b) is the independent path.
4. Validation: `nrt_load` parses+validates without executing (needs neuron-rtd;
   full validation needs device). `cpp/neff_info.cpp` pattern loads a NEFF and
   prints tensors - reuse as round-trip checker on the trn1 box.

## Open items

- Header field-by-field semantics beyond offsets 0x00-0x10 and the name field;
  digest inputs (digest B tried: md5 of tar, name+size+data walk, header w/
  digests zeroed, name, header+gz - none match).
- Engine instruction encodings: 32-byte fixed instructions confirmed (word
  histograms). Shared preamble across all engines: first instr opcode 0x10b1
  (entry), SP streams also use 0x10cc/0x10c1. PE opcodes seen: 0x1001/0x1002/
  0x10a/0x20a/0x30a/0x10a0; DVE: 0x106b + operand-encodings 0x200d0a/0x80100a;
  Pool: 0x10d4/0x10080/0x107e/0x800f00/0x801300. Low 12 bits look like opcode
  class, upper bits carry fields. Full field decode pending.
- DGE (DMA gather engine) table format (`DGETable0: {list:[2,1,3,4], type:"dge-table"}`).
- collectives NEFF additions (`kelf` cc_streams, allreduce routing).
- Whether newer neuronx-cc emits v3 headers (NEURON versions >2.26 untested).

## Resolved during RE

- Const weights: large tensors (w1/w2/w3, 32MB) are `type: input` vars - the
  runtime binds them via nrt_tensor_write before execute. Only tiny consts
  (e.g. identity matrix for activation-engine control, `*_CRSM.npy`) are
  embedded in the tar as `file` vars.
- DVE opcode/control/datapath tables and activation bkt/ctrl bins ARE embedded
  in tar for xla/torch-originated graphs (ffn1_full, flash_attn) under
  `sg00/default_*.bin` + `sg00/*_bkt.bin`; the NKI-originated mh graph instead
  relies on runtime-provided defaults. An emitter should embed for
  self-containment.
- debug_info_asm_<ENGINE>.dbg protobuf: f1 = description, f2 = backend dbg
  filename, repeated f3 groups each containing one varint BIR-instruction id
  (repeated ~4x, one per asm instr emitted for that BIR instr), f4 = 1. Partial
  coverage only (~460 BIR ids vs 12k PE instrs in ffn) - covers "interesting"
  instructions, not NOPs/padding.
- dlopen of libnrtucode_extisa.so hangs on this box (constructor waits for
  hardware); its API must be studied statically or driven on the trn1
  instance.

## Graph semantics (from working minimal reference, nki-cache kernel-26bb)

Verified 2026-11-08 on trn1: emitter-rebuilt container (own header/tar/json +
reference engine streams) produces BIT-IDENTICAL output to the original.

- DMA staging is done by COMPILED descriptors in Pool0.bin (op 212
  PSEUDO_DMA_DIRECT2D), NOT by SP0.json dma entries (those were empty in the
  minimal reference). SP0.json dma appears to serve spill/reload only.
- op212 addressing: src/dst start_addr = u64 {marker<<56 | payload}.
  marker 32 (0x20) = HBM var reference (payload = ioq mr/DGE-table lookup);
  marker 16 (0x10) = scratch address (payload = byte offset, e.g. 0x82c000).
  The DGE table (def.json var DGETable0, type "dge-table") is the IO VAR-ID
  LIST in input/output order (e.g. [1,2,3] for vars x=1,weights=2,y=3).
  WRONG DGE list = "Failed to find io name for mr N" at execute.
  MISSING DGE table = "Address from table is not allowed" at load.
- op212 has sem_increment=16 and `semaphore` field = the DESCRIPTOR's
  completion semaphore; descriptor block index byte @0x0A corresponds to the
  pool-dynamic semaphore slot (first descriptors use sems 8,9,10...).
- waits: WAIT_SEM_GE_IMM(5) in the events block (@4 mode, @5 idx, @8 value).
- Engine program layout: [177 prologue][204 PSEUDO_TENSOR_STORE marker][ops...];
  Pool adds trailing 164 NOP; compute in DVE (ops 65/66 + EVENT_SEMAPHORE 160)
  and Activation (ops 35/33) with function-set LUTs (act_info.json +
  <set>.json + _bkt.bin/_ctrl.bin MUST be shipped complete - a truncated LUT
  yields all-NaN outputs).
- neff.json: heads points at the tvm_op NODE INDEX; node_row_ptr has
  len(nodes)+1 entries; shapes/storage cover all node outputs.

## Elementwise numerics proven (kernel-26bb replica, trn1 2026-11-09)

Container emitted by tools/neff_emit.py + lift_hip.py with reference engine
streams runs BIT-IDENTICAL to the original NEFF. Decode of that kernel:
- Pool op212 descriptors: dst marker16 scratch addrs 0x82c000 (x) / 0x2c000
  (w); DVE op65 (TENSOR_TENSOR_ARITH = multiply) reads scratch in ITS address
  space (space1, addr = pool addr >> 4); ACTIVATE(33) applies a constant
  scale; result written back to 0x82c000; op212-out (waits sem6 GE 3) copies
  to HBM var (DGE table[2]).
- Semaphore units: op212 sem_increment = 16 per descriptor; DVE waits GE 16;
  sem6 counts ENGINE EVENTS (+1 each), op212-out waits GE 3 (3 compute instrs).
- Numerics: y = 3.3214889 * x * w (constant = kernel-specific activation imm).

## PE (matmul) path - open items

Single-tile y[128,128] = x@w via [177][204][LW][MM] with pool-op212 staging
still yields all-zero PSUM drain. Variants ruled out: space0/space1, addr>>4,
full addr, psum_accumulate_flags SINGLE(3). Reference FFN instrs replicate
byte-identically (tools/neff_isa round-trip), so field VALUES in a working
context are the missing piece - next step: perturb the KNOWN-WORKING ffn
container (identity-matmul graph) one instruction at a time:
1. run ffn with controlled inputs, map y tiles to staged SB regions
2. confirms PE src/dst address <-> SB offset mapping and PSUM row layout
   (dst = 0x2000000 + row*0x800 suggests partition-per-row)
3. then reuse the mapping for the lifted GEMM stream.
Working reference streams live in /tmp/opencode/ref26bb/sg00/ and
/tmp/opencode/neff/ffn_x/sg00/ (volatile; re-pull from nki-cache on box).

## PE ucode loading + slot structure (2026-11-09, neuron-dbg findings)

- Sequencer disasm (neuron-dbg `capture` + `disasm`) shows each engine runs a
  tiny RUNTIME-GENERATED orchestration program ([SOM][EVSEM][WR config][NOP
  wait event][DRN]) - IDENTICAL for working ffn and my containers. The
  PE0.bin/DVE0.bin streams are HW UCODE loaded into each engine's instruction
  memory; the sequencer just pokes start/config CSRs.
- v2 PE stream slot rules (from working ffn stream): 32B slots DENSE, every
  instruction followed by a pad slot: zeros after 177/204/EVENT_SEMAPHORE(160),
  op10 (v2-only) after LDWEIGHTS/MATMUL - op10 carries the PSUM accumulate
  mode (b1=1 MULTI_START after LW, b1=2 MULTI_END after MM; b1=3 = SINGLE
  variant seen). Back-to-back instrs without pads are silently DROPPED by the
  v2 translator (result: zeros, no load error).
- ffn program shape: [177][0][204][0][160 wait sem24 GE16][0][LW wait sem8
  GE16][op10(1)][MM upd19/5][op10(2)][LW][op10(2)][MM]... (6144 LW + 6144 MM,
  y = identity @ staged tiles per-tile).
- ffn pool op212 dst addresses (marker16) = SB byte offsets 0x2bc00-descending
  = exactly the LW src region; LW step -1 compensates the compiler's
  descending chunk order. marker32 = HBM var (DGE table idx + byte offset).
- DGE table = IO var-id list; wrong list = "Failed to find io name for mr N".
- NRT 1004 = exec completed with engine errors; 1203 = DMA abort (bad pool
  descs); 5 = timeout (IO refill ring never drained by truncated consumer).
- Elementwise (DVE) path: FULLY WORKING numeric pipeline (see above).
- PE matmul path: ucode accepted (with pads) but compute still yields zeros.
  Remaining unknowns: LW B-tile addressing semantics, PSUM drain mapping.
  Next step: compile trivial torch.matmul via torch_neuronx.trace on the box
  (nki-cache kernels are mostly elementwise; only cache kernel with PE =
  complex attention) -> clean minimal PE ucode oracle with known shapes.

## PE matmul numerics proven (2026-11-09, trace-matmul oracle)

Generated a minimal matmul NEFF by hand-writing serialized StableHLO
(`jax.jit(lambda a,b: a@b).lower(...).compiler_ir().operation.write_bytecode()`
-> `neuronx-cc compile graph.mlirbc --framework XLA --target trn1`).
`graph.hlo` from torch_neuronx.trace is a SERIALIZED HloProto (binary), not text.

The resulting NEFF (32x128 @ 128x128 f32) executes on trn1 via neff_io2 with
maxdiff 0 (A@B). AND my emitter's rebuild (my header/tar/neff.json/def.json/DGE
+ compiler streams + identity npy file var) also executes A@B correctly
(maxdiff 2e-6 = f32 noise). Template decoded:

- def vars: SB 0, input0 1, input1 2, output0 3, identity file var 4
  (value_sg0000_identity_N_CRSM.npy shipped in tar), DGETable0 5 = [1,2,3].
- dma_queue: qPoolDynamic sems [8,9,10], qSPSpillReload0 [24,25,26].
- SP0.json dma: ONE desc: identity file -> SB 196480, sem 24 (qSPSpillReload0).
- Pool0: [177][0][204][0][op212 input0 16KB -> SB 0x2fd80 sem8][0-pads? no:
  op212 descs also carry implicit pads? NO - pool stream dense 64B instrs,
  no 32B pads: op212@2, op212@3, op212@4, op164 NOP@5].
- PE0 (32B slots, every instr + pad): [177][0][204][0][160 wait 5/24 GE16][0]
  [LW wait 5/8 GE16 src=SB 0x2ff7c step-1 num128][op10 b1=1]
  [MM A=SB 0x2ff80 (identity tile) dst PSUM 0x2000000 upd 19/5][op10 b1=2]
  [LW src 0x2ff7c no wait][op10 b1=2]
  [MM same upd 19/5][op10 b1=3 SINGLE]
  [160 wait 5/9 GE16][0][LW wait 5/4 ...][op10(1)][MM][op10(2)][LW][op10(2)]
  [MM upd 19/5][op10(3)].
- The compiler lowers x@w as identity-tile matmuls on partitions (A operand =
  identity staged tile; B operand = LW-loaded reversed input tile); groups
  chain: first op10 after LW = b1=1 (MULTI_START), between groups b1=2
  (MULTI_MID/END), final group ends b1=3 (SINGLE).
- op212 pool descs here: marker32 src = HBM var (DGE idx), dst marker16 = SB
  byte offset; sem_increment=16; y-out desc waits sem4 GE2 (!; sem4 = dynamic
  scratch slot, outside def queue sets - wait, wait_idx=4 means desc waits on
  sem 4 = ... it's the compiler's internal sem numbering, works as shipped).

## Multi-K-tile GEMM chaining (mm2: 32x256 @ 256x128, trn1 verified)

Compiler structure for K > 128 (PE contraction per instr = 128... in practice
K chunks): K-loop = N-tile-loop of [LW][op10][MM][op10] groups whose LW/MM
SB addresses are the SAME every iteration. Chaining works via a
PRODUCER-CONSUMER RING on the dynamic queue:
- qPoolDynamic has SB_scratch partition + desc#N (dst SB+0x0, src = next
  input0 K-chunk, waits sem(GE count)) re-stages the next chunk into the
  same SB slot after the PE consumed the previous one.
- op10 byte10 ∈ {0,1,2} selects the ring slot/half being read.
- pool descs: [input0 tile0 -> SB 0x2fb80 sem8], [input1 tile0 -> 0x2f780
  sem9], [refill input0 chunk -> SB 0x0 waits 5/4 GE2 sem10].
Verified: my emitter's byte-copy rebuild of mm2 (K=256, 8 chained groups)
executes A@B exactly (maxdiff 4e-6).
Lifter implication: arbitrary K = (K/128 - 1) refill descs + (K/64?) PE
groups with op10 byte10 alternating ring halves; exact per-group cadence
still needs one more decode pass (dynamic queue ring semantics).

## K=512 cadence (mm3 oracle, trn1 2026-11-09)

K=512, N=128, M=32: PE stream = 80 slots:
1. Identity/A phase (slots 6-37): 16 [LW desc][MM] pairs; LW srcs cycle 4 chunk
   addrs (0x2f7fc/0x2f9fc/0x2fbfc/0x2fdfc, step -1) each used TWICE; MM A =
   identity 0x2fe00, dst PSUM 0x2001000/0x2001080/0x2001100/0x2001180 (4
   output sub-tiles, stride 0x80 = 128B); op10 b1: 1 at each 4-chunk group
   start, 2 else; op10 "addr" field = PSUM tick target, byte10 = 32.
2. B/PV phase (slots 40-71): 16 pairs, LW srcs cycle 0x2effc/0x2f1fc/0x2f3fc/
   0x2f5fc; MM dst PSUM 0x4000/0x4080/0x4100/0x4180; op10 b1=3 (SINGLE per
   pair), byte10 = 32.
3. PV pass (slots 72-79): 2 pairs num=128, dst PSUM 0x800; LW src 0x427c.
4. Event sems bracket phases: wait 5/24 (identity), 5/8, 5/9, upd 19/5 at
   group boundaries (slots 36, 70, 78).
mm3.neff + full mm3_x extracted streams = /tmp/opencode/neff/mm3.neff +
/tmp/opencode/mm3_x/ (volatile). Ring/chunk cadence per pool descs still to
be decoded locally from mm3_x/sg00/Pool0.bin (my hand-parse on box used wrong
struct offsets; use tools/neff_isa.py decode).

## mm3 pool descs decoded (tools/neff_isa.py, 2026-11-09)

K=512/N=128/M=32 pool = only 3 descs (no explicit refill descs!):
- @2 sem8: src HBM var0 (A, DGE ti0, offset 0) -> SB 0x2f600, 2D transfer
  covering 4 ring slots of 512B (0x2f600..0x2f7ff; LW A-side addrs = slot
  ends 0x2f7fc/0x2f9fc/0x2fbfc/0x2fdfc, step -1 = slot-read reversed).
- @3 sem9: src HBM var1 (B, ti1) -> SB 0x2ee00, 4 slots (B-side LW ends
  0x2effc/0x2f1fc/0x2f3fc/0x2f5fc).
- @4 sem10: src SB 0x4280 -> HBM var2 (y-out, ti2), waits 5/4 GE 3.

KEY: qPoolDynamic (type=dynamic) RE-EXECUTES the desc set per cycle with the
HBM src offset AUTO-ADVANCING (streaming DMA) - that is the entire refill
ring. LW addrs repeat because the same SB slots are re-staged with the next
K-chunk. Cadence per K:
- slots = K/128 ring slots (K=128:1 @0x2ff80-area, 256:2, 512:4)
- pairs: K128=2, K256=8, K512=32 (A-phase + B-phase split appears at K>=512
  with 4 PSUM sub-tiles + PV pass; K<=256 = single-phase, no PV)
- passes over the ring: K512 uses each slot 2x (chunk128 × 2 halves?) -
  consistent with M=32: PE contraction chunk = 128, M=32 x passes... exact
  pair-count formula still needs one more shape (e.g. K=512 M=64) but the
  streaming-DMA mechanism is now fully identified.

Confirmed cadence (mm/mm2/mm3):
- A-desc elem_size = (K/128) x 512B staged per ring pass; B-desc = 512B+
  (B-side reads may route via DGE/HBM at K>=256: mm2 B-side LW addrs were
  DGE-range, mm3 B-side = SB slots)
- y-out desc src = PSUM base (0x4080/0x4100/0x4280 - varies w/ N tiles),
  waits 5/4 GE (num phases).
NEXT SESSION (implementation, decode done): write general emitter in
lift_hip.py/tile_ir.py emitting: 3 pool descs (A ring, B ring, y-out),
identity refill SP dma, PE stream = phase cadence per mm/mm2/mm3 templates
(pair counts: K128=2, K256=8, K512=32; b1 modes 1=start-of-chunk-group,
2=mid, 3=per-pair-single at K>=512 B-side; upd 19/5 at group ends),
validate x@w vs numpy for K in {128,256,512}, M in {32,64}, N in {128,256}.

## General matmul emitter rules (decoded from 6 oracles: K{128,256,512} x M{32,64} x N{128,256})

Params: KS=K/64 k-step pairs per pass; AS=K/128 ring slots (512B each);
NT=N/64 PSUM sub-tiles (64B stride); passes: A-phase then B-phase then PV
(PV only when NT>2? K128/N128 has no PV: mm 2 pairs total = 1 pass each).

Observed sequences (k = 0..KS-1, s = 0..AS-1 = k//2, tile = 0..NT-1):
- A-phase pairs (NT x KS): LW src = SB A_ring[s] end (0x?fc), step=-1, num=128,
  wait 5/8 GE16 only on k=0 of tile 0 (sem8 = A stage desc); MM A=identity
  0x2fe00 step1 num=M, dst PSUM 0x2000000, upd 19/5 on last k of tile;
  op10 b1 = 1 if k==0 else 2, after LW AND after MM (b1 same).
  PSUM dst during A-phase = 0x2fe00?? NO: mm_32_512_256 A-phase MM dst
  pattern = identity 0x2fe00 (A operand!), dst PSUM 0x2000000 fixed.
  LW addrs cycle s=0,0,1,1,2,2,3,3 for K512 (2 k-steps per slot); s=k//2.
  For K256: slots at 0x2fc00+, K128: 0x2fe00-512 slot.
- B-phase (NT x KS): LW src = SB B_ring[b] end, step=-1, num=128 (b = k//2
  offset by NT? mm_32_512_256: B slots 0x2ee00+0x200*b+0x1fc for b in
  0..3 first NT-pass, then 0x2ee00+2048+... second pass); MM A = SB
  scratch tile (0x4000 + 0x80*tile for K512N256; 0x2f980 for N256K128 -
  scratch tile addrs live ABOVE 0x4000 when NT>2, below when small);
  dst PSUM sub-tile 0x4000+0x80*tile (K512N256) / SB scratch (K128N256);
  upd 19/5 at last k of tile.
- PV phase (only NT>=4? K512 shapes): LW = PSUM 0x427c/0x42fc... descending
  reads of the B-phase PSUM tiles, num = 32/64; MM A = identity, num=128
  cols, dst PSUM 0x800. Two pairs per N-tile-half.
- y-out pool desc: src SB PV PSUM (0x4280/0x42fc...), dst HBM var, waits
  5/4 GE (num phases = 3).

IMPLEMENTATION NOTE: exact scratch tile placement (0x4000 vs 0x2f980 base)
= compiler allocator choice; any free SB region works as long as LW/MM
addresses are consistent. Emitter validates byte-exactly against the six
archived oracle streams (oracles_x/mm_*).
