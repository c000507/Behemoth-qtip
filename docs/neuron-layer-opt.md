# Devstral layer optimization: baseline, bound model, packing tests (brief 9)

All timings on i-0adbefb8af607c610, trn1.2xlarge, both NCs for the TP
runs (repo cpp/layer_bench.cpp + NEURON_RT_VISIBLE_CORES=0-1 +
NEURON_RT_ROOT_COMM_ID), 100-step steady state.

## 1. Baseline on the real graphs (cydonia-24b-fp8-kv)

| graph | rows | ms/step | per-row ms |
|---|---|---|---|
| dec1_128 | 1 | **1.320** | 1.320 |
| dec4_128 | 4 | 1.642 | 0.411 |
| dec8_128 | 8 | 1.797 | 0.225 |

- Weight tensors per rank (unpacked dec1_128 def.json, float8e4): MLP
  tiled = 83.9 MB x 3 = 251.7 MB; q/k/v/o shards = 10.5 + 10.5 + 2.6 +
  2.6 = 26.2 MB; total **278 MB fp8 weights per rank per layer**.
  KV cache tensor = 91.1 MB fp8 (a capacity, not per-execute traffic at
  M=1; only touched pages move).
- Marginal cost per extra row = only ~68 us -> the weight stream
  (278 MB) happens ONCE per step regardless of rows; the per-row work
  is activations/KV only. Batch amortizes the stream almost perfectly
  (8 rows = 1.797 ms total = 4.4 tok/s-equivalent throughput for
  multi-request serving).

## 2. The bound model (corrected — supersedes the 179 GB/s PE claim)

- The 179 GB/s figure (128 elem/cycle x 1.4 GHz) = the PE array's
  OPERAND rate, confirmed by the repo's fp8 lm_head data
  (262 MB / 1.46 ms = 179.5 GB/s). It is the rate at which the PE
  consumes f32/bf16 OPERANDS, not fp8 bytes.
- dec1_128 streams 278 MB of fp8 weights in 1.320 ms = **210 GB/s
  effective** — ABOVE the 179 GB/s operand floor. Therefore the layer
  cannot be PE-operand-bound: it pipelines HBM->SB (fp8) -> dequant to
  f32 in SB -> PE (f32 operands, 716 GB/s equivalent). The binding
  stages are the fp8 DMA and the dequant, which must overlap the PE.
- Isolated M=1 matvec (docs/neuron-pe-bound.md): bf16 12288^2 =
  1378 us = 228 GB/s (DMA serial ceiling); fp8 = 1065 us (150 GB/s
  effective, 1.29x over bf16, NOT 2x): the gap = the in-graph dequant
  pass. The tiled layer does better than the isolated matvec
  (210 vs 150 GB/s) because its MLP tiling overlaps the pipeline.

## 3. Ceiling arithmetic for "faster layer graphs"

- If the weight stream ran at the compiler's elementwise-mul geometry
  (~395 GB/s aggregate, ONE continuous multi-packet desc — decoded in
  docs/neff-opt.md): 278 MB / 395 = 0.70 ms; add the measured
  non-overlapped tail (~0.2-0.35 ms: KV page gather + scales + all-
  reduce + attention glue) -> **0.9-1.05 ms/layer** -> 40 layers =
  36-42 ms + head 2 ms + host ~3 ms = **41-47 ms/token = 21-24 t/s**
  IF the dequant stays hidden inside the stream.
- At the measured parallel-DMA ceiling only (240-265 GB/s): 278/250 =
  1.11 ms + the same tail = ~1.3-1.45 ms = little gain. **The upside
  is entirely in whether the mul-geometry (continuous desc, no
  per-slice semaphore round-trips) can carry the tiled MLP weight
  streams while the consumer paces them.**
- Hard ceiling (any geometry): the HBM read of 278 MB at the observed
  ~395 GB/s envelope = 0.70 ms/layer = 28 ms/40 layers = ~25-26 t/s
  with a zero-cost head. Beyond that = not streaming every weight per
  token (rejected for now) or a faster consumer dtype path.

## 4. Packing tests done so far (matvec level, isolated)

| variant | time | note |
|---|---|---|
| fp8 + per-row scale (f32) | 1065 us | reference fp8 matvec |
| fp8 SCALE-FOLDED (no scale input; W pre-quantized at +-0.06) | **1063 us** | **NO CHANGE** — the scale multiply was not the cost; the fp8->f32 cast pipeline is |

Negative result: folding the dequant scale away buys nothing; the
dequant cost is the cast itself, so packing scales differently
(beside/interleaved) cannot speed the matvec. The remaining packing
lever is the DMA chunk geometry, not the scale placement.

## 5. What is NOT yet done (next session)

- Devstral-dimension layer-graph VARIANTS through the repo trace
  pipeline (trace_layer_tp.py hardcodes 8B dims: HIDDEN 4096,
  INTERMEDIATE 14336; Devstral = 5120/32768). Candidate variants:
  fused QKV, fused gate/up, MLP tile shape, scale placement, KV gather
  placement. Each compile = tens of minutes; needs a dedicated session.
- The 395-geometry test on a real weight stream: recompiling the MLP
  tile with the elementwise-mul descriptor form and measuring both
  ranks. This is the decisive experiment for the 21-24 t/s range.
- 2-rank numerics validation vs the C++ oracle (the runner fills
  deterministic inputs; numerics = checked separately).

## Verdict

- The user is NOT at the PE floor; the layer is at 210 GB/s effective
  vs a 250-400 GB/s DMA envelope. Realistic upside = 1.15-1.6x per
  layer (15 -> 17-24 t/s), entirely from DMA geometry and dequant
  overlap, NOT from the vector engines (rejected) and NOT from scale
  packing (measured no-op).
- dec1 vs dec4/dec8: batching amortizes the stream almost perfectly;
  for single-stream latency the graph variants above are the only
  lever.
