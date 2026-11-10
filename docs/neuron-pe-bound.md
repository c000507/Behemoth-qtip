# Neuron PE/DMA bound for fp8 decode (brief 8 follow-through)

## Isolated M=1 matvec matrix (trn1, single NC, 200-iter avg, neurond/timing)

| shape (w = KxN) | w bytes bf16 | bf16 us | bf16 GB/s | w bytes fp8 | fp8 us | fp8 GB/s | speedup |
|---|---|---|---|---|---|---|---|
| 12288 x 12288 | 302 MB | 1378 | 228 | 151 MB | 1065 | 150 | 1.29x |
| 32768 x 5120 | 336 MB | 1808 | 192 | 168 MB | 1361 | 129 | 1.33x |

- bf16 rate = 192-228 GB/s = at the DMA serial ceiling measured in the
  DMA-only sweep (224-266): **the bf16 M=1 matvec is DMA-bound.**
- fp8 = 1.29-1.33x, NOT 2x. fp8 does NOT hit the naive 2x (DMA
  halving would give ~690 us at 12288^2); it also does not sit exactly
  at the 179 GB/s PE-operand floor (that would be ~900 us incl. the
  56 us launch floor). The ~220 us gap between the floor and the
  measurement is consistent with an in-graph dequant pass (fp8 -> f32
  mul-by-scale over all weights feeding the PE) that is not fully
  hidden.
- Numerics: bf16 maxdiff vs torch bf16 = 8.7/14.3 (bf16-level); fp8
  = 0.33/0.55 absolute on row-scale-224 quantized weights (max |ref|
  ~3) = the expected fp8 rounding. Both compiled with the repo
  pipeline: torch_neuronx.trace -> tools_patch_fp8_hlo.py
  (F8E4M3FN -> F8E4M3) -> neuronx-cc 2.26.

## The 179 GB/s figure: confirmed, but it is NOT the fp8 bottleneck

- Repo head data (PERF_NOTES): fp8 lm_head 262 MB / 1.46 ms of PE
  streaming = **179.5 GB/s** = the 128-elements-per-cycle x 1.4 GHz
  moving-operand rate. That number is the rate at which the PE array
  consumes OPERANDS (f32 after dequant: 128 elem/cycle = 716 GB/s of
  f32, or bf16 operands at 358 GB/s).
- Therefore an fp8 weight stream that is DEQUANTIZED BEFORE the PE is
  not limited by 179 GB/s of fp8 bytes; it is limited by
  (a) the DMA that moves the fp8 bytes (measured ceiling 224-266
  serial / 240-265 parallel; compiler mul geometry reaches ~395), and
  (b) the dequant pass that reads fp8 + scale and writes f32.
- fp8 buys wall time only to the extent the DMA + dequant pipeline
  hides behind each other. Measured: 1.29-1.33x on an isolated matvec.

## Real dec1_128 layer (Devstral, TP=2, both ranks, layer_bench)

- 1.320 ms/layer steady-state (100 steps), 23 inputs / 3 outputs,
  372 MB inputs per rank.
- Weight tensors per rank (from the unpacked def.json): MLP tiled
  = 83.9 MB x 3 = 251.7 MB; attention q/k/v/o shards = 10.5+10.5+2.6+2.6
  = 26.2 MB -> **278 MB fp8 weights per rank per layer**.
- KV cache tensor = 91.1 MB fp8 (in + out): only the touched pages are
  gathered per execute at M=1; the full-tensor sizes are capacities.
- Implied effective weight-stream rate = 278 MB / 1.32 ms =
  **210 GB/s** — ABOVE the 179 GB/s PE-operand floor, consistent with
  the dequant-into-SB picture: the layer pipelines DMA(fp8) ->
  dequant(SB, f32) -> PE(f32 operands), so its bound is the DMA +
  dequant overlap, not the PE operand rate.
- Token arithmetic at the measured 1.32 ms: 40 layers = 52.8 ms +
  head ~2 ms + host ~3 ms ≈ 58 ms/token ≈ 17 t/s ceiling for the
  dec1 bucket; the user's 15 t/s = the launch gaps + the older 1.46 ms
  profile. The headroom is in per-layer time, which is DMA/dequant
  bound, i.e. exactly what brief 9's packing/geometry experiments
  attack.

## FP8 toolchain status (brief 6 step 4, settled)

- neuronx-cc (2.26/2.28) rejects F8E4M3FN on TRN1/TRN2
  (NCC_EVRF051: "not supported, target TRN3 or later"); the
  --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 flag is not accepted by
  the torch_neuronx/jax plugin lower path (NCC_EARG002).
- Working path (the repo's own): trace -> patch HLO F8E4M3FN -> F8E4M3
  (identical encoding below 240) -> compile. Row-scale-224 quantized
  weights keep every code < 240.
- fp8 weights as an INPUT dtype (Devstral dec graphs) = what the real
  graphs already do: float8e4 inputs + f32 scales, dequant in-graph.

## Bottom line

The fp8 decode is not at a hard PE wall; it is at a
DMA-geometry + dequant-overlap wall. Isolated matvec evidence: fp8
1.29-1.33x over bf16 with a naive per-row-scale dequant; the layer
achieves 210 GB/s effective vs the 224-266 (ours) / ~395 (compiler
elementwise geometry) DMA envelope. Closing that gap = descriptor
geometry + dequant placement/packing (brief 9), worth up to ~1.2-1.6x
per layer if the 395 GB/s geometry can be applied to the weight
streams and the dequant stays hidden.
