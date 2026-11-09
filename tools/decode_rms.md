# RMSNorm oracle decode (on-box neuronx-cc 2.28, rms128 = [128,128] f32)

## Graph
y = x * rsqrt(mean(x^2, axis=-1) + 1e-6), x = input0 [128,128] f32 (64KB).

## Engines: NO PE. DVE + ACT only. Pool = 3 descs (input in sem8, out sem9,
+ op164 tail). rms vars: input0 (65536, vid 1), output0 (65536, vid 2),
DGETable0 vid 3. dma_queue = qPoolDynamic only.

## Opcode names (from isa.json DWARF enum, /tmp/opencode/isa.json)
- DVE 66 = TENSOR_REDUCE_ARITH_OP (the rowsum!)
- DVE 73 = MEMSET (zeroes the reduce accumulator bank)
- DVE 67 = TENSOR_MUL-ish (elementwise B template op67 = mul)
- DVE 10 = dst-pattern/move (same opcode as PE op10)
- ACT 33 = ACTIVATE (elemwise/nonlinear engine, scale f32 @36 known)
- ACT 35 = ACTIVATION_TABLE_LOAD (LUT machinery)
- ACT 122 = LOAD_POOL_ARGUMENT (carries f32 1.0 @4 + int 16384 @8!)
- ACT 170 = TENSOR_LOAD

## DVE0.bin (384B = 8 slots) for rms128
- s4/s5 = op73 MEMSET 64B pair: @6-7 = 13 06; s5[0] = 05, s5[37-38] = 40 00
- s6/s7 = op66 TENSOR_REDUCE_ARITH_OP pair:
  s6: @4-5 = (mode 5, idx 4) wait (pool-in sem4), @6-7 = 13 06, @8 = 01
  s7 = op10 dst: byte1 = 0x0a (DVE wl = 10, NOT 16!), @2-3 = 80 00,
      @4-5 = 04 02 (u16 0x0204), dst @16-19 = 0x08000000 = marker8|bank0
      (DVE PSUM write marker = 8, NOT the PE MM marker 2!), @20-21 = 04 02,
      @24-25 = 08 42, @28.. steps = 1s
- => reduce accumulator = PSUM bank0 written via DVE marker 8.

## Activation0.bin (448B) for rms128
- s4 = op160 wait (mode5, idx8, value 16)
- s6 = op35 ACTIVATION_TABLE_LOAD (empty body here; table via exp_and_others)
- s8 = op33 ACTIVATE: wait (5,6) (sem6 = DVE done), @6-7 = 13 04, @8-9 = 1
- s9 = op122 LOAD_POOL_ARGUMENT: wl 0x0a, @2 = 80, @3 = 1e (30),
      f32 1.0 @4-7, u32 16384 @8-11 (= 128*128?), u32 @16 = 0x4008
- s10 = op33: wait (5,6), @8 = 2
- s11 = op170 TENSOR_LOAD: @2-3 = 80 80, payload bytes 00 00 03 cb / d3 78 63 50
- s12 = op33: wait (5,3) (sem3!), @8 = 1
- s13 = op170: bytes 01 84 20 00 ...
- => 3 ACTIVATE passes chained via ACT sems (6 then 3): the sqrt/div via
  LUT + the final y multiply. f32 constants visible: 1.0 (s9), 4.0 (s9 @16).

## rsqrt[16384] oracle: DVE/ACT/PE all EMPTY (128B prologues only!)
1D graphs compile to nothing (bug or unsupported) - use 2D micro-oracles.

## NEXT (decode TODO)
1. op66 field semantics: reduce axis/stride/num (bytes @6-7 = 13 06 config?),
   per-pass element counts, 2D vs 1D reduce modes.
2. ACT chain semantics: which op33 = div, which = sqrt (LUT mode bits),
   f32 constants per pass; then how to inject the *g scale and eps.
3. exp_and_others tables: rms128 needs act_info.json + tables transferred
   (excluded from current extraction - re-pull with filters off).
4. Emit path: RMSNorm emitter = DVE0(384B template w/ patched sizes) +
   ACT(448B template) + Pool(3 descs, sem8 in / sem9 out) + wrapper.
   M/N generalization = same u16-field discipline as matmul/elementwise.

## 2026-11-09 (later): oracle container wrapper status
- rms128 oracle re-extracted with FULL sg00 (19 files incl act_info,
  exp_and_others tables, PE0.json etc) via chunked transfer with
  per-chunk length validation (SSM stdout drops/prefixes lines:
  ALWAYS re.search not match; validate chunk len == 8000 or tail).
  Blob persisted on box at /tmp/rmsblob.b64 (38576B), fetcher = /tmp/fc2.py.
- Wrapper built (neff.json 1-input rms graph + def.json from oracle).
- DEVICE: nrt_load = "Invalid NEFF buffer" - root cause NOT yet found:
  the tgz transfer was flaky (stale object / partial extraction), so the
  loaded neff may have been truncated. NEXT SESSION: re-upload with
  checksum, verify on box (md5), then re-test load. If load still fails,
  compare def.json/neff.json against a KNOWN-GOOD matmul wrapper
  (emit_matmul output) field by field - rms def.json is the compiler's own.
- sp_dma identity facts (fresh oracles): 32_128_384 rows [512,128]
  to_off tof-N(=0x2fe00); 64_128_384 rows [256,128] to_off tof(0x2ff00);
  64_1024_128 rows [512,128] to_off tof-4M(0x2fe00); 32_640_128
  rows [512,128] to_off 0x2fe00 (= tof-384, NOT tof!); 32_384_256
  rows [512,128] to_off 0x2fe00 (= tof, big layout). to_off rule =
  shape literal, table-driven for now.
- Perf harness (timing.cpp) still at /home/ubuntu/work/nrtd on box.

## 2026-11-09 (final): all lifted ops device-validated
- rms128 oracle container: PASS bit-accurate (2.4e-07) after md5-verified
  transfer ("Invalid NEFF buffer" was purely transfer flakiness).
- emit_rmsnorm 4/4 PASS: M128/M256 x eps {1e-6, 1e-3} (eps f32 @360 ACT).
- emit_softmax PASS: 1.6e-06 (exp LUT tables ship in oracle sg00).
- emit_dequant PASS BIT-EXACT (0.0): q uint8 cast on PE + scale broadcast.
- weighted rms full-chain (rms + elementwise g compose): PASS 1.8e-07.
- perf (timing.cpp rebuilt on box, 200 iters): synth 64_384_128 = 62.7us
  avg vs oracle = 62.4us = parity; rms 61.5us, softmax 62.0us; all
  launch-bound (~60us floor at these sizes).
- Box gotchas: /tmp dirs get wiped by overlapping SSM runs - use
  /home/ubuntu/work/pf for multi-step box work; S3 GET flaky (~1 in 3)
  - always md5-verify + retry loop.
