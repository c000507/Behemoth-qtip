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

## 3. THE DECISIVE EXPERIMENT: read-only DMA rate (2026-11-10)

The 395 GB/s mul number = read+write. Read-only probes:

| probe | traffic | marginal rate |
|---|---|---|
| jnp.sum(x) n=262144->1048576 (4 MB read + 4 B write per execute) | 3 MB delta / 14.3 us | **210 GB/s** |
| GEMV M=1 12288^2 bf16 (302 MB weight read + 4 KB output) | 302 MB / 1322 us | **228 GB/s** |
| mul x*1.0 (4 MB read + 4 MB write, HBM->HBM direct copy) | 7.34 MB / 18.6 us | 395 total |

Conclusion: the 395 GB/s geometry = the **HBM->HBM direct copy**
(DIRECT2D: ~198 read + ~198 write concurrent, saturating the bus). A
pure one-direction read — which is what a weight stream is — tops out
at **210-230 GB/s regardless of descriptor geometry**. The compiler's
GEMV ring (66 auto-advancing descs) is ALREADY at that ceiling; the
real dec1_128 layer at 210 GB/s effective = **92-100% of the
read ceiling**.

**Verdict on brief9 questions:**
1. Faster per-layer graphs: the weight read = at the per-direction DMA
   ceiling; the remaining lever = the non- overlapped tail only
   (~8-15%: perfect-overlap floor ~1.15-1.20 ms/layer -> 17-19 t/s).
   The 21-24 t/s estimate in the previous revision is RETRACTED — it
   assumed the 395 geometry transfers to reads; it does not.
2. Packing the weights differently: no. The read rate is
   geometry-independent at the ceiling; the scale-fold test = no-op;
   interleaving cannot exceed the one-direction bus rate. fp8 already
   minimizes bytes. The only real levers left: batch (68 us marginal
   per row), not streaming every weight per token (rejected), or a
   different hardware path (rejected for now).

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

## 6. Read-only measurement details (2026-11-10, decisive experiment)

- sumread probes: compiled via the oracle pipeline (jax sum lowers to
  a reduce over a staged 4 MB f32 tensor; output = 4 B). Marginal
  between n=262144 (65.2 us) and n=1048576 (79.5 us): 3 MB / 14.3 us
  = 210 GB/s. Caveat: the reduce adds ~8-9 us at 1M elements, so the
  TRUE pure-read rate sits between 210 (reduce-mixed) and the GEMV's
  228 (a matvec output is 4 KB = effectively read-only, no reduce).
- The mul-geometry decode (docs/neff-opt.md): ONE op212 desc,
  continuous, 64 KiB packets, packet-count field = bytes/64KiB; both
  src and dst = HBM vars (DIRECT2D) — hence the 2-direction 395.
- Weight streams = src HBM -> dst SB: one-direction by definition.
  The observed per-direction DMA ceiling across every probe:
  210 (sum, reduce-mixed), 224-266 (serial desc chain, r then w),
  228 (GEMV ring, pure read), 240-265 (2 concurrent reads + write).
  A single-direction stream never exceeded ~266; reads top at ~228.
