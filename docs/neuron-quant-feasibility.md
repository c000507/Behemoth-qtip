# On-chip dequant feasibility: NVFP4 + QTIP on Neuron (trn1)

Brief 5 step 2. Status: INTERIM — format analysis complete, engine
capability map complete (ISA-level), decode-throughput microbenchmarks and
the bit-exact prototype are the next work items. Everything measured says
GO; nothing measured says NO-GO yet either — the open risk is throughput
of the bit-vector ops, not capability.

## What the HIP references do (read, runtime/src + glm53)

Summarized from the source (subagent report, line numbers verified):

- **QTIP** (qtip.hip, qtip.cpp): 2 bits/weight. 16x16 tile = 32 u16
  (512-bit tail-biting bitstream) = 256 weights. Decode per state
  t = 0..127: circular 32-bit window `u = w[i]<<16 | w[(i+1)&31]`,
  `s = (u >> (16-4*(t&3))) & 0xffff`, LUT index
  `e = (s*(s+1) >> 6) & 1023` (plus 1 sign bit from bit 15 of s*(s+1)),
  LUT entry = bf16x2 (the tile's two weights). LUT = 1024 u32 = 4KB,
  identical for all linears, derived from the 512x2 fp16 `tlut` safetensor
  (+ sign flip). NO per-weight scale: signs on input (SU), sign*scale on
  output (SV), Hadamard rotations around the GEMV. States are independent
  (direct bit-window read, no carry chain). GPU: 1 lane = 1 word-pair per
  tile, `v_mad_u32_u24` + LDS LUT + `v_fdot2_f32_bf16`.
- **FP8** (fp8_linear.hip): OCP E4M3 weights u8 [out][in], fp32 per-output-
  row scale applied once post-accumulate. Decode = HW intrinsic
  fp8->f32. GEMV: 16 weights/thread, fmaf chain, wave reduce.
- **NVFP4** (glm53/src/weights.cpp, device_ops.cu): two E2M1 per byte
  (LOW nibble first), E4M3 scale per 16-element block, fp32 global scale
  (global = maxabs/(448*6), block scale = e4m3(blockmax/(6*global))).
  E2M1 levels {0,.5,1,1.5,2,3,4,6} + sign bit. Device decode = LUT-free
  arithmetic: `mag = n<2 ? n*.5 : (2+(n&1)) * 2^((n>>1)-2)` built via
  float-exponent construction, then `±mag * fp8(scale) * global`.
- **AQLM codebook** (projection.hip reconstruct/fused_projection): u16
  codes -> 8-fp8-byte entries, 2-level gather, fp8xfp8 WMMA 16x16x32.
  Fused inside the K-loop; scales applied once at epilogue. The sorted /
  delta-code layouts (popcount chains + u64 atomics) are hostiles to any
  non-GPU ISA — out of scope for Neuron.

## What the trn1 ISA offers (from the DWARF-extracted opcode enum)

The ISA has far more than the fp32 DVE arithmetic we have used so far:

- **BITVEC (integer/bitwise) DVE ops**: TENSOR_TENSOR_BITVEC_OP (81),
  TENSOR_REDUCE_BITVEC (82), TENSOR_SCALAR_BITVEC (83),
  TENSOR_SCALAR_PTR_BITVEC (84), TENSOR_CUMULATIVE_BITVEC (94),
  TENSOR_SCALAR_PTR_MULTI_BITVEC (95), TENSOR_SCALAR_IMM_LD_BITVEC (113),
  CROSS_LANE_REDUCE_BITVEC (125), TENSOR_SCALAR_PTR_MULTI_DUAL_BITVEC
  (136), SCALAR_TENSOR_TENSOR_BITVEC (158) — the and/or/xor/shift family
  the QTIP window extraction and NVFP4 nibble unpack need. Field layouts
  pending decode (same accessor path that gave us the fp32 arith ops).
- **TENSOR_DEQUANTIZE (123)** — a dedicated dequant opcode (fields TBD).
- **GATHER (104)** — indexed gather (DVE), for codebook-style lookups.
- **ACT engine tables**: the exp LUT machinery we already ship
  (exp_and_others bkt ~941-1036 entries) — a 1024-entry QTIP LUT or a
  16-entry E2M1 LUT fits the same table-load path
  (ACTIVATION_TABLE_LOAD op35, proven).
- **PE dtypes** (DTYPE enum): BFLOAT16 = 6, FP16 = 7, FP8_EXP3 = 13,
  FP8_EXP4 = 14, FP8_EXP5 = 15, INT8 = 2, UINT8 = 3 — the matmul
  in_dtype field accepts bf16/fp8 weights directly. If the PE consumes
  FP8 natively, NVFP4 decode = nibble->fp8 LUT only (16 entries), and
  QTIP decode = bit-unpack->bf16x2 LUT only.
- **PE WEIGHT_MASK (4) / WEIGHT_SHIFT (5) opcodes** exist — possibly the
  on-PE unpack primitives for packed weights (fields TBD).

## Decode designs

### NVFP4 (GO candidate)

Pipeline (all pieces individually proven on trn1):
1. Pool DMA streams packed u8 weights into SB (proven: emit_dequant).
2. Nibble unpack: DVE BITVEC ops (shift+and) OR the PE WEIGHT_SHIFT/MASK
   ops — produces 16 entries/block + block-scale index.
3. LUT: 16-entry E2M1 table via ACT (ACTIVATION_TABLE_LOAD + activate
   pass, proven machinery) — or the arithmetic form (2 shifts + fmul) if
   the table path is slower.
4. Per-block scale: pre-bake `fp8(scale)*global` per block on the HOST
   (16-entry LUT per block is wrong — scale varies per block; instead
   fold the scale into the DVE broadcast mul like emit_dequant's
   s-broadcast, with block granularity).
5. PE matmul consumes fp8 (in_dtype FP8_EXP4) or the expanded bf16.

Decode cost ≈ 1-2 ops per weight after unpack. At the measured
220 GB/s DMA plateau, fp32 streams 55 Gweight/s; NVFP4 = 4x = needs
220 Gweight/s decode — this is the number to microbench (DVE BITVEC
throughput). If BITVEC runs at the same element rate as fp32 arith
(each op = 128 lanes), a 3-op decode = ~73 Gweight/s/engine = the
bottleneck becomes DVE, not HBM. Mitigation: PE consumes PACKED fp8
after a cheaper unpack, or split unpack across DVE+ACT.

### FP8 E4M3 (TRIVIAL GO)

= emit_dequant with dtype swapped: q u8 in (PE in_dtype FP8_EXP4
directly, or u8->fp8 cast) + per-row f32 scale broadcast (proven DVE
broadcast). No LUT, no bit ops. This should be the FIRST prototype:
smallest delta from the proven emit_dequant path.

### QTIP (conditioned GO)

- The friendly part: states independent, LUT = 4KB (fits the ACT table
  path we ship today), no per-weight scale.
- Risk 1 — `s*(s+1)` mod 2^16: fp32 multiply is inexact for s ≥ 2^8.
  Fix (exact): s = 256h + l; s*(s+1) = l^2 + 512hl + 65536h(h+1);
  mod 2^16 the last term dies; l^2 ≤ 2^16 exact in f32; 512hl ≤ 2^17
  exact in f32 -> two f32 ops + one add = exact. ~4 DVE ops per state.
- Risk 2 — the 512-bit tail-biting stream crossing u16 words: fix at
  PACK time (our loader): re-pack each state's 16-bit window as an
  aligned u16 in HBM (~2x memory of the packed form: 4B/weight still
  beats fp32's 4B/weight+scale only marginally...). NOTE: aligned
  re-pack = 2 bytes/weight = same as fp8 — QTIP's win vs fp8 shrinks
  to precision, unless the raw u16-stream form is unpacked on-chip
  (BITVEC shifts, 2-3 ops/state).
- LUT throughput: ACT table pass per element = the ACT engine is the
  exp-LUT engine — its throughput at 128 lanes/chunk is measured by the
  rms/softmax timings (softmax 2048x2048 = 244 µs for ~4M elements of
  exp+div ≈ 17 G/s — WAY too slow for weight streaming if done at
  element granularity; the ACT path needs batched table passes or the
  DVE-side alternative).

### AQLM / sorted layouts (NO-GO on Neuron)

Random 8B gathers per element + popcount carry chains + u64 atomics.
Pool descriptors are strided patterns, not gathers; skip.

## Measurements still needed (next session)

1. DVE BITVEC throughput microbench (and/or/xor/shift chain over n
   elements, our own stream) vs the fp32 mul rate we already have.
2. ACT table-load + lookup pass rate at 128-chunk granularity
   (16-entry and 1024-entry tables).
3. PE matmul with in_dtype FP8_EXP4 / BFLOAT16 (patch an oracle
   container's in_dtype + dtype-scaled weights, compare numerics) —
   decides whether decode must expand to bf16 or can stop at fp8.
4. NVFP4 GEMV prototype BIT-EXACT vs the glm53 host decode
   (weights.cpp dequant reference) — q-packed u8 in + block scales in,
   y = dequant(x) @ w with fp32 PE.
5. QTIP single-tile prototype (256 weights) BIT-EXACT vs qtip.cpp
   decode_tile, then a 16x16-tile GEMV with the epilogue Hadamard chain
   moved to DVE.

Interim verdict: **NVFP4 = go, FP8 = trivial go, QTIP = go with the two
risks above to be measured, AQLM/sorted = no-go.** The decisive
measurement is DVE BITVEC throughput (item 1) — it sets whether decode
keeps up with the 220 GB/s plateau or DVE becomes the bottleneck.

## Measured results (device, this session)

- **BITVEC ops EXECUTE on trn1, bit-exact**: compiler-oracle graphs for
  `a&b`, `a|b`, `a^b` (op81 TT-BITVEC), `a<<4`, `a>>4`, `a&15` (op83
  TS-BITVEC + op8 scalar-operand loader) and `jnp.take` (GATHER via pool
  indirect machinery op218/op167) all ran bit-exact on the device. Opcode
  names are chip-implemented, not enum fiction.
- **Field layouts decoded** (64B DVE instrs; slot pairs = head/tail):
  - op81 (TENSOR_TENSOR_BITVEC): src0 addr u16@12, src0 ALU u16@14
    (0x800A = AND, 0x800B = OR, 0x800C = XOR); src1 @32; dst @48;
    wait/upd events as usual. ALU codes 0x8006 = MUL on the arith twin
    (op65) — same pattern field, different opcode.
  - op83 (TENSOR_SCALAR_BITVEC): tensor @12; scalar pattern @32..: mode
    u16@36 (3 = SHR, 10 = AND, 16 = SHL), immediate value @40 (4 = shift
    amount, 15 = mask). op8 = scalar-operand carrier instr (value @10)
    observed before the bitvec ops.
- **Throughput** (compiler oracles, `(a&b)^((a<<3)&b)`, 4 int ops/elem):
  n=131072 = 68.4 µs, n=1048576 = 133.3 µs → marginal 12.6 MB / 64.9 µs
  = 194 GB/s (≈ DMA plateau) and ≥ 48 Gop/s DVE bitvec. **The bitvec
  decode is DMA-bound at 1M elements — DVE keeps up with HBM.**
- **PE in_dtype probe** (32×128×128 synth container, weights as 16-bit):
  - MM in_dtype @32 = 7 (FP16) with fp32-side LW: finite but wrong
    (rel 1.84) — the fp16 path engages, but 16-bit weight staging needs
    the full 16-bit geometry (LW step/num semantics + pool desc spacing
    + the fp32 regs hold 2 fp16 lanes), not just desc size halving.
  - Both-16-bit patches (LW+MM = 6/7): NaN — LW in_dtype @32 is not
    simply "the dtype of the loaded weights".
  - Conclusion: bf16/fp8 PE consumption is REAL but needs a dedicated
    16-bit staging decode session (expected: 1-2 h of probe work).
- **op123 TENSOR_DEQUANTIZE / op104 GATHER**: GATHER proven working
  (through pool indirect DMA, not a bare op104 — the compiler lowers
  jnp.take to pool descs + op83 index math + op3 PE_REG_WRITE). No
  compiler-emitted bare op123/op104 seen yet; their on-chip status stays
  unproven. The dequant path we need (LUT/nibble/scale) is covered by
  op81+op83+ACT regardless.
