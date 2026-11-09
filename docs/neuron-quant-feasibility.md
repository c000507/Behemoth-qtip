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

## DVE engine-rate measurement (resident-data microbench)

Method: TS-BITVEC shift chains over RESIDENT SB data (staged once, no
per-round DMA), n = 4096, P = 32, R rounds of [ashr 1 -> y][shl 1 -> y],
our own synthesized stream (tools/bv_synth.py) inside the validated
and-oracle container:

| rounds | elem-ops (2/elem/round) | mean us | min us |
|---|---|---|---|
| 16  | 131072  | 59.3 | 57.6 |
| 64  | 524288  | 66.5 | 64.7 |
| 256 | 2097152 | 93.8 | 91.9 |

Slope = 143.75 ns/round → **DVE TS-BITVEC rate = 57 G elem-ops/s**
(engine-bound, data resident). Matches the compiler's own 4-op TT chain
at n = 1M (55 Gop/s marginal). The earlier "48+ Gop/s DMA-bound" number
was a lower bound; this is the engine rate.

Latency structure: ~36 ns per 32B slot (72 ns per op83+op8 pair
covering 4096 elements at P = 32). Whether the rate is per-instr latency
(rate would scale with P: 227 Gop/s at P = 128) or fixed throughput is
OPEN — the P = 128 test hits an undecoded pool-desc tiling field
("Failed to translate dst pseudo addr8 for pseudo dma direct 2d" at
desc2 rel48 = 4P = 512; 128/4096 both valid, 512 rejected).

Decode-implication at 57 Gop/s: NVFP4 needs ~2 DVE ops/weight → ~110
Gop/s to keep up with the 220 GB/s plateau at 4 values/byte → DVE-side
decode alone would cap effective streaming at ~26 Gweight/s = HALF the
fp32 rate (55 Gw/s) → NVFP4 loses UNLESS (a) the per-instr-latency
hypothesis holds (P-scaling buys 4x) or (b) the LUT+scale half of the
decode moves to the ACT engine (parallel) or (c) the PE consumes packed
fp8 directly (in_dtype path, pending 16-bit staging decode).

Chain-semantics gotchas (device-proven):
- upd u16 = (idx<<8)|mode; byte-swapped (mode<<8)|idx runs but notifies
  the wrong semaphore class → silently wrong numerics (observed).
- upd = 0 on the final DVE op = y-out drain never fires = execute
  timeout (status 5).
- op8 = the op83 scalar-carrier (same 64B fetch pair), NOT a separate
  instruction stream element.
- The R-sweep numerics mismatch at imm=1/self-dst (y<<1→y) is OPEN:
  timing unaffected (dst-chain is data-flow independent); bit-exact
  validation of long chains pending (imm=4 validated single-op).

## Chain numerics status (open, scoped)

Multi-op TS chains inside the 2-input and-oracle container run + scale
linearly but produce wrong values even for a single [ashr 4 src=0x888]
op that is byte-identical to the validated shr-oracle op except the src
address. The DVE-address <-> SB-staging mapping is therefore
container-shape dependent (2-input 3-desc containers differ from
1-input 2-desc ones). RESOLUTION PATH (next session): build the NVFP4
decode chain in the **1-input container shape** (the shr-oracle
structure: in0 at DVE-0, y at 0x4000, pool = 2 descs) which is fully
validated end-to-end; the decode = [shr4][and 15] nibble streams +
ACT-LUT(16-entry E2M1) + TT-mul by block-scale = 1-input-shaped ops.

Device-proven bit-exact single ops (for the record): a&b, a|b, a^b
(op81), a<<4, a>>4, a&15 (op83+op8), jnp.take. Rate 57 Gop/s resident.

## P-scaling, ACT rate, op123/op104 hw tests (2026-11-10, device)

**P-scaling test** (tools/bv_pscale.py; n = 4096 fixed, R = 64 rounds of
[shr1->y][shl1->y], P decoupled from n by patching the P fields; 200-iter
timing; baseline single-op container = 61.5 us avg / 57.3 min):

| P | avg us | min us | per-op incremental (vs baseline, /128 ops) | rate at n=4096 |
|---|---|---|---|---|
| 8   | 66.5 | 64.8 | ~58 ns | 70 Gop/s |
| 16  | 67.4 | 65.3 | ~62 ns | 65 Gop/s |
| 32  | 68.5 | 65.9 | ~67 ns | 61 Gop/s |
| 64  | 71.2 | 68.8 | ~90 ns | 45 Gop/s |
| 128 | 77.7 | 74.2 | ~132 ns | 31 Gop/s |

- **Rate FALLS with P at fixed n** (per-op ~= 54 ns + 0.61 ns x P). The
  "num_active_channels parallelism buys 4x at P=128" hypothesis is DEAD:
  P = per-channel overhead, not a parallelism knob, at fixed n.
- All P values (incl. P = 128) EXECUTE on the 2-desc pool with rel48 =
  4P = 512 — the earlier translate failure ("dst pseudo addr8") was
  specific to the 3-desc ew pool layout, not a general P limit.
- P is not a coverage gate either: P = 8 still processed all 4096
  elements (y = 2x everywhere from the shl op).
- Numerics of chained [shr][shl] on a shared dst buffer = y = 2x at ALL
  P (adjacent-slot write->read race on 0x4000; the shr result is not
  visible to the next op). Timing unaffected; long-chain bit-exact
  numerics stay OPEN (ping-pong dst buffers = the obvious next fix).
- Field map for n/P scaling extracted from fresh shr oracles
  (n = 2048/4096/8192, all neuronx-cc 2.26): pool is n-INDEPENDENT
  (tensor sizes live in the DGE/shape side) — only 4P fields scale:
  desc0 rel{24,36,60} = 4P, rel40 = -4P; desc1 rel{36,48,60} = 4P;
  DVE op83 head rel16 = -4P, rel26 = P; op8 carrier rel26 = P.
  SP0/PE0/Activation streams are constant across n.

**ACT engine rate** (compiler oracles, exp vs mul x1.0 at same shapes,
same wrapper — marginal isolates the ACT LUT pass):

| n | mul x1.0 | exp | exp - mul | ACT marginal |
|---|---|---|---|---|
| 131072  | 62.0 us | 65.1 us | 3.1 us  | 42 G exp/s |
| 1048576 | 80.6 us | 91.3 us | 10.7 us | 98 G exp/s |

Fit: ACT exp ~= 2 us fixed + 8.3 ns/elem -> **~121 G exp/s marginal**
(917504 extra elems / 7.6 us extra). Supersedes the 17 G/s softmax-based
guess — the ACT LUT pipeline is ~7x faster than feared. Pool/DMA
marginal in the compiler mul containers = 7.34 MB / 18.6 us = ~395 GB/s
(our synth-desc plateau = 220 GB/s; compiler desc granularity is better).

**op123 TENSOR_DEQUANTIZE: NO** (does not execute on trn1 in any tested
placement). 64B instr built per v3.NEURON_ISA_TPB_S3D3_TENS_DEQUANT
fields + is_valid_s3d3_tens_dequant constraints (wl = 0x10, nac > 0,
dequant_fmt 1..4, group_size {0..7,8}); variants: fmt 1..4, in/out
dtype FP32/INT8/UINT8/FP16, group 0/8, step 1/0xFFF0, dst@44/@48.
Placements: DVE stream, POOL stream (between descs), ACT stream, PE
stream — ALL return `execute failed: 1004` (NRT_EXEC_COMPLETED_WITH_ERR).
Corroborating: the DVE hwdecode table (default_opcode_table.bin) has no
0x7b entry (has 0x68 = 104). op123 = dead on trn1; dequant goes through
op81/op83 + ACT LUT as designed.

**op104 GATHER (bare): NO** (no numerics-positive execution found).
DVE/ACT placements = 1004. POOL placement (op104 between descs)
COMPLETES CLEAN but y is all zeros — with dst@40 AND dst@44 variants,
idx = x-region self-indexing (expected y = x[x] check) — i.e. the pool
engine tolerates/skips it without producing values. The compiler never
emits bare op104 (it lowers gathers to the pool indirect DMA machinery
+ op83 + op3), which is the proven usable path.

## Updated go/no-go arithmetic (all engines, measured)

Need at the 220 GB/s synth plateau: NVFP4 = 2 values/byte ->
**440 G values/s**; fp8 = 1 value/byte -> 220 G values/s; fp32
baseline = 55 G values/s.

- DVE bitvec decode ceiling (measured, worst case at fixed n):
  **57-70 G elem-ops/s, falling with P**. NVFP4 unpack = >=1 op/value
  -> 57-70 G values/s = **6-8x short of 440**. NO-GO via DVE decode.
- ACT LUT pass (measured): ~121 G/s marginal — above fp32 parity
  (55 G/s), below fp8 (220) and NVFP4 (440) needs. As a PIPELINE stage
  after DVE unpack the min rate still binds at DVE 57-70.
- OPEN rate-law question RESOLVED (same session, R-amplified sweep,
  tools/bv_pscale.py --n): per-op time is **INDEPENDENT of n** and
  linear in P: **t_op ~= 60 ns + 0.75 ns x P** (P = channels),
  validated n = 4096 -> 1048576 (at P = 32: 64-70 ns/op at EVERY n;
  P = 1024 -> ~1.0 us/op; P = 8192 -> ~10.3 us/op, slight
  superlinearity ~1.26 ns/ch at the top). n-independence means the DVE
  streams all n elements inside one op (coverage is pattern/DGE-side,
  not P-side; confirmed by P = 8 still processing all 4096 elems).

### Final decode arithmetic (measured law)

Sustained DVE rate at natural P = n/128: asymptote = 128/0.75 ns =
**~171 G elem-ops/s**; measured 102 G/s at n = 1M (P = 8192), 128 G/s
at n = 131k (P = 1024). Per-pass cost over a whole tensor = one op.

- NVFP4: unpack = 2 DVE passes (AND 15 | SHR 4) + scale-mul = 1 more
  DVE pass; ACT LUT (121 G/s) runs in parallel. DVE-bound:
  171/3 ~= **57 G values/s** vs 440 needed = **13% of plateau. NO-GO.**
- fp8 E4M3: values need no unpack; ACT LUT (256-entry) = 121 G/s < 220
  G values/s needed = 55%; DVE scale-mul pass 171 G/s = 78%. Both
  engines short alone; pipelined DVE+ACT still <= ~121-171 Gv/s.
  **NO-GO via DVE/ACT; GO only via PE in_dtype = FP8_EXP4 (14), which
  skips decode entirely — gated on the 16-bit staging geometry
  decode.**
- QTIP: ~1 DVE op/value (window extract) + 0.5 ACT lookup/value +
  Hadamard epilogue adds (~2 DVE/value) -> DVE ~3 ops/value = ~57
  G values/s vs ~390 G values/s demanded at 4.5 bits/weight.
  **NO-GO.** (The earlier ACT-rate fear is retired — 121 G/s measured —
  but the DVE op-per-value wall decides it.)
- Bottom line: no software dequant path on DVE/ACT can keep up with
  the DMA plateau at 4/8-bit densities. The only measured-free GO
  candidates are (a) PE-native packed dtypes (fp8 in_dtype; 16-bit
  LW/desc staging decode = the gating probe session), (b) GPSIMD
  Xtensa glue (deferred), or (c) staying in fp32/bf16 with the 40%
  compiler headroom already measured.

## PE-native low-bit weights (measured, 2026-11-10)

Method: compiler (neuronx-cc 2.26) jax oracles at 32x128x128 with
bfloat16 / float16 / float8_e4m3fn inputs, byte-diffed against the
fp32 twin; probe containers = patched streams + jsons + identity npy.

**bf16 and f16 weights WORK (the open door opens for 16-bit):**
- both-bf16 32x128x128 compiler container = **BIT-EXACT** vs torch
  bf16 matmul (maxdiff 0.0). f16 GEMV M1 K4096 N4096 = f16-level
  numerics (rel 4.9e-4 vs torch f16) and runs.
- in_dtype byte = @+32 of the 64B LW(op1)/MM(op2) instr (10 = FP32,
  6 = BFLOAT16, 14 = FP8_EXP4). LW head@16 addr/step pseudo-fields
  scale as f32_value/(elem/4) (e.g. B-LW = 0x4000 + (M-1)*elem).
- **Uniform-dtype pipeline law**: in_dtype = the dtype of the data IN
  SB/PSUM at that pipeline point. The compiler's own mixed lowering
  (x f32 x w bf16) produces NaN on device; a w-only-16bit patch of the
  f32 container = NaN too (the B-phase LW re-reads the A-phase x-temps
  through a 16-bit window from an f32 PSUM). 16-bit = all-or-nothing
  (or needs a dtype-conversion stage that the compiler doesn't emit).
- Pool w-desc (desc1): rel{24,36,60} stride fields 0x0200 -> 0x0100
  (f32 -> bf16); rel62 = dtype marker pair 0x0a0a/0x0606.
- def.json var sizes = bytes; identity npy = eye(128).reshape(
  128,128,1,1) in the compute dtype (f32 descr <f4, bf16 descr <V2).
- LOADER FINDINGS (nrt_load 2.33.10): def.json is ORDER-SENSITIVE
  (popping + re-adding the identity var key = load fail 2 with
  identical content); the identity npy reader accepts descr <f4/<V2
  but REJECTS <u8/<V1 (u8 fp8 eye = load fail on the bf16 base).

**GEMV weight-streaming measurements** (compiler containers, M in
{1,4,8}, square K=N in {4096,12288}, 200-iter timing avg):

| shape | fp32 us | bf16 us | speedup | w GB/s f32 | w GB/s bf16 |
|---|---|---|---|---|---|
| M1 4096x4096   | 405.7  | 208.2  | 1.95x | 41 | 40 |
| M4 4096x4096   | 363.7  | 221.0  | 1.65x | 45 | 37 |
| M8 4096x4096   | 356.1  | 215.9  | 1.65x | 46 | 38 |
| M1 12288x12288 | 2778.4 | 1300.2 | 2.14x | 218 | 232 |
| M4 12288x12288 | 2553.8 | 1294.7 | 1.97x | 237 | 234 |
| M8 12288x12288 | 2558.4 | 1300.2 | 1.97x | 236 | 234 |

- fp32 GEMV numerics = exact vs numpy (<= 2.1e-6 rel); bf16 = bf16-
  level (rel 2e-4..2e-3 vs fp32 torch ref, accumulation-order noise).
- **Weight streaming is byte-bound: halving weight bytes = ~2x
  latency at every M tested.** The compiler's stream saturates the
  ~220-230 GB/s plateau only at K,N = 12288; at 4096x4096 it streams
  at only ~40 GB/s (schedule not overlapping enough at small sizes —
  headroom OUR emitter could claim: a synth bf16 stream at the
  220 GB/s plateau would be ~5x faster than the compiler at 4096^2).

**FP8_EXP4 (in_dtype 14): EXECUTES but numerics = open (negative
so far).** A fully-patched fp8 container (all four core instrs
in_dtype = 14, desc markers 0x0e0e + /4 strides, fp8 input files,
fp8 dltype) LOADS and RUNS on device — the opcode-level dtype is
accepted at execute. Numerics = garbage (maxdiff ~46), unchanged when
the identity-npy byte layout + A-MM stride are adjusted — the
mismatch sits deeper (w-side B-phase pattern/desc strides for 1B/elem
and/or the identity-dtype wall above). fp8 GEMV speedup = NOT yet
measurable. Residual decode work: B-phase w-pattern stride/num fields
for 1B elements + a loader-legal identity in an 8-bit pipeline
(craft V2-npy bytes that the 1B read pattern sees as an fp8 eye).

Verdict update: **bf16/f16 weight-streaming = GO (2x measured, bit-
exact)**; **fp8 = blocked on data plumbing, opcode accepted** (promising
but unproven); NVFP4/QTIP via DVE/ACT = NO-GO (measured, previous
section). Our-emitter priority: synth bf16 weight path at the 220 GB/s
plateau (big win over the compiler's 40 GB/s at 4096^2), then fp8
plumbing probes if the win justifies it.
