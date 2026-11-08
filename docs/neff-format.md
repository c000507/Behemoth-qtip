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
