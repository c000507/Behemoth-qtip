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

## 7. Payload accounting + engine-concurrency sweep (brief 10, 2026-11-10)

Payload accounting: the mul "395 GB/s" counts read+write bytes of the
HBM->HBM copy: 3.67 MB read + 3.67 MB write in 18.6 us = **198 GB/s
payload** — BELOW the 228 GB/s pure-read GEMV. The copy is not a faster
path; it never was.

Concurrency sweep (single NC, read-only, jnp.sum over N independent
4 MB inputs; compile-time graphs, 100-200 iters):

| streams | slice | total read | avg us | aggregate GB/s |
|---|---|---|---|---|
| 1  | 4 MB | 4.2 MB   | 78.7  | 184 |
| 2  | 4 MB | 8.4 MB   | 94.9  | 216 |
| 4  | 4 MB | 16.8 MB  | 127.6 | 234 |
| 8  | 4 MB | 33.6 MB  | 190.3 | 250 |
| 16 | 4 MB | 67.1 MB  | 315.1 | 259 |
| 32 | 1 MB | 33.6 MB  | 191.1 | 248 |
| 4  | 8 MB | 134.2 MB | 565.1 | **264** |

- Marginal N=1->8: 29.4 MB / 111.6 us = 263 GB/s. The aggregate rises
  monotonically to N=16 and the knee is at ~8-16 streams; the shared
  ceiling = **~260-265 GB/s per core**, NOT a per-engine 14 GB/s wall.
- Slice size matters mildly: 8 MB slices (264) > 4 MB (250-259) >
  1 MB (248) at saturation.
- The compiler GEMV ring (228 GB/s, 578 descs on the 16-sem/16-queue
  qPoolDynamic set) behaves like ~1-2 EFFECTIVE concurrent streams,
  not 16: the ring's descriptor chain is serialized despite the 16
  semaphores. That is the gap.
- Both ranks in layer_bench already read concurrently: 210 GB/s per
  core = 420 GB/s chip aggregate, well under chip HBM (~820). The
  per-core multi-stream ceiling (260-265) is therefore the real
  per-core limit, and the layer sits 20-25% below it.
- fp8-matvec dequant exposure (arithmetic): 1065 us total - 56 floor
  - 151 MB/228 GB/s (=662 us, if the DMA ran at the single-stream
  read rate) = ~347 us (~33%) of exposed dequant/gaps. With the
  multi-stream DMA (260+) the exposed fraction grows unless the
  dequant is overlapped better.

Verdict (brief 10 item 6): there IS read-side headroom: splitting the
weight stream into >=8 concurrent descriptor streams with >=4-8 MB
contiguous runs should lift the per-core read from ~210-228 to
~260-265 GB/s = the dec1_128 layer from 1.32 to ~1.06-1.15 ms ->
t/s = 1000/(40*1.10+5) ~= 20-21. Unverified: whether the layer's
consumer pacing (NKI attention, tiled MLP) tolerates the deeper
concurrency, and cross-core DMA behavior beyond the 2-rank layer bench.

## 8. Brief 11: split-stream fp8 matvec at the real MLP shape — KILLED

32768x5120 fp8 + per-row scales, single NC, 100 iters, weight split
along the output rows into S independent batched matvecs (contiguous
32 KB rows, disjoint outputs, concatenated):

| S | avg us | eff GB/s (w bytes / (t-56)) | maxdiff |
|---|---|---|---|
| 1  | 1340.8 | 131 | 0.0056 |
| 2  | 1337.8 | 131 | 0.0056 |
| 4  | 1337.0 | 131 | 0.0056 |
| 8  | 1338.8 | 131 | 0.0056 |
| 16 | 1454.7 | 120 | 0.0056 |

- All variants bit-identical to each other and to the plain matvec
  (1361 us, earlier matrix). **No split helps; S=16 is 8% worse.**
- Effective rate = 131 GB/s vs the read ceiling 184-264: the fp8
  matvec is NOT DMA-bound at this shape — it is **dequant-bound**: the
  in-graph cast materializes fp8 x4 as f32 through SB (168 MB fp8 ->
  672 MB f32 write + re-read), and that SB-side traffic + the cast
  pipeline, not the HBM read, sets the time.
- fp8 vs bf16 at the same shape: 1341 vs 1808 us = 1.35x — the fp8
  byte savings are almost entirely consumed by the dequant.
- Therefore brief 11 step 2 (applying the split to the real layer via
  the Devstral-dim trace) is MOOT: there is no single-NC win to carry.
  The layer's 210 GB/s effective already exceeds the isolated matvec's
  131 because its tiling hides the dequant better.

Net verdict across briefs 9-11: the Devstral decode layer is bounded
by (a) the one-direction HBM read ceiling (~210-230 GB/s per core,
geometry-independent) and (b) the fp8 dequant SB traffic, which the
layer's tiling already hides about as well as anything measured.
Remaining levers: batch amortization (68 us/row), or not streaming all
weights per token. Single-stream t/s upside from graph work: ~1.1x
at most (17 -> ~19 t/s).

## 9. Brief 12: the real layer is ALREADY native fp8 x fp8 — dequant claim corrected

FACT-FIND (local, dec1_128 unpacked NEFF): the PE stream = 5850 LW +
6026 MM 64B instructions; in_dtype histogram (byte @+32):
**LW: 5578 x FP8_EXP4(14)**, 240 x f32(10), 24 x bf16(6), 8 x u8(3);
**MM: 5754 x FP8_EXP4(14)**, 240 x f32, 24 x bf16, 8 x u8.
The seven big projections run as **native fp8 x fp8 PE matmuls**
(the repo fp8_linear form: the activation row-scaled into E4M3,
fp8 x fp8 dot with f32 accumulate, output x act_scale x weight_scale).
The brief-11 "dequant f32 in SB" inference for the REAL layer was
WRONG — it applies only to the isolated cast-form matvec. The real
layer never casts the weights.

Isolated native-fp8 matvec (same retarget pipeline, 100 iters, w = 168 MB fp8):

| form | 32768x5120 | 5120x32768 | split S=1/4/8 |
|---|---|---|---|
| native fp8 x fp8 | 1430/1429/1533 us = 122 GB/s | 1239/1240/1237 us = 142 GB/s | flat, no effect |
| cast form (brief 11) | 1341 us = 131 GB/s | - | flat |

- The native form is SLOWER than the cast form at this isolated shape
  (1430 vs 1341) — the 8B-layer table (native 1.84 vs cast 3.96) does
  not transfer to a single isolated matvec; the real layer's win comes
  from its deep tiling/pipelining, which the isolated matvec lacks.
- The isolated matvecs (122-142 GB/s effective) are ALL below the real
  layer's 210 GB/s effective. The layer pipelines better than anything
  we can build standalone from single matmuls.
- Splitting (S=1..8) = flat in every form: brief 11 step 3 answered
  definitively NO.

Revised bound picture: the PE consumes fp8 directly (no dequant); the
layer's 210 GB/s effective fp8-byte rate sits between the 179 GB/s
one-element-per-cycle operand rate and the ~358 GB/s packed rate, and
80-92% of the one-direction read ceiling (228-264). The remaining
lever = the non-overlapped tail (~0.1-0.2 ms) = the layer ~1.15-1.2 ms
-> ~17-19 t/s. No packing, splitting, or dtype-form change moves it.

## 10. Brief 13: WHY standalone matvecs are slower than the layer (differential, measured)

Structure diff (PE stream, per 64B LW/MM pair):

| graph | weight bytes | LW pairs | bytes/LW pair | descs | KB/desc |
|---|---|---|---|---|---|
| isolated native matvec (any shape) | 168 MB | 20482 | **8.2 KB** | 323-355 | ~520 |
| isolated MLP chain (3x84MB, real deps) | 252 MB | 30982 | **8.1 KB** | - | - |
| real dec1_128 layer | 343 MB | 5850 | **58.6 KB** | 566 | ~560 |

- The compiler picks **8.1-8.2 KB PE tiles for ANY standalone matmul**
  (every shape tested: [5120,32768], [40960,4096], [20480,4096]; every
  form: native, cast, split, 8-separate-inputs) and **58.6 KB tiles
  only inside the full-layer compile**. Tile choice is layer-context-
  dependent and not controllable from the trace.
- At M=1 the matmul is PE pair-latency-bound at 8 KB tiles: measured
  62.6 ns per LW+MM pair (168 MB / 20482 pairs / 1.28 ms) -> the PE
  chain caps the standalone graph at **~130-142 GB/s** and the DMA
  waits for it. Splitting (S=1..8), separate input tensors (8x), shape
  changes, and the real MLP chain all land at the same ~130-142.
- HLO decode corrections: the real layer = 2 computations (SPMD ranks);
  the attention = ONE fused [44488, 2048] fp8 weight (91.1 MB) inside an
  AwsNeuronCustomNativeKernel; the MLP = 3 x [20480, 4096] fp8 + [1,
  16384] scales. **Weights per rank = 91.1 + 3 x 83.9 = 343 MB** (not
  278 — the 91.1 MB tensor had been mislabeled as the KV cache; it is
  the fused attention weight). The KV cache = the small page tensors +
  a 91.1 MB output written back each layer.
- REAL layer traffic per execute = 343 MB read + 91.1 MB KV write-back
  (+ touched-page reads) = ~434 MB in 1.320 ms = **329 GB/s bus
  aggregate** (reads at the ~260 multi-stream ceiling + the writes
  concurrent), ~83% of the ~2 x 395-class chip-bus envelope with both
  ranks running. The layer is BUS-bound, not PE-bound and not
  descriptor-bound.

Verdict: the 122-142 vs 210 gap = the compiler's PE tile size (8 KB
standalone vs 58.6 KB in-layer), which amortizes the M=1 LW+MM pair
latency; and the layer additionally overlaps its KV write-back with
the weight reads. Neither is reproducible in a standalone compiled
matmul, and neither is controllable from the trace. Practical ceiling:
the layer at ~83% of the achievable bus = the remaining upside ~1.05-
1.15x (17-19 t/s), consistent with briefs 10-12. Implied t/s for the
realistic improvements: tail polish only.

## 11. Brief 14: KV-copy hypothesis KILLED; final accounting

Bucket-scaling test (repo layer_bench, 2 ranks, 100 steps):

| bucket | cache capacity (pages x 32 KB) | ms/step |
|---|---|---|
| dec1_32  | 22.8 MB  | **1.299** |
| dec1_128 | 91.1 MB  | **1.319** |
| dec1_512 | 364.4 MB | **1.433** |

- dec1_32 -> dec1_128 costs only **20 us** for a 4x cache. A full
  91 MB cache read+write copy per layer would cost ~0.4-0.55 ms at the
  measured rates: **there is no full-cache copy.** The KV write-back
  touches only the current page(s) (the NKI kernel writes the new K/V
  rows); output1 is capacity-sized and only partly written.
- dec1_512's +114 us = the attention gather/kernel group-loop scaling
  with the gather span (~0.16 us/page = ~32 KB/page at the DMA rate),
  not a copy.
- Corrected accounting (HLO-verified): weights per rank = **91.1 MB
  fused attention weight [44488,2048] + 3 x 83.9 MB MLP shards =
  343 MB**; total inputs = 372.1 MB per rank per layer read from HBM
  (weights + tables + scales + touched pages). Input21 = the KV cache
  (5561 x 8 x 16 x 128 fp8 = 91,111,424 B) staged once at load; the
  566 pool descriptors = the per-execute HBM reads of all inputs
  (~372 MB), NOT a cache copy.
- Implied per-execute read rate = 372 MB / 1.32 ms = **282 GB/s** =
  at/above the single-NC multi-stream read ceiling (260-265): the
  layer is at the practical memory-system ceiling with both ranks
  running.
- Brief14 task 3 (no-copy variant) = MOOT: no copy exists to remove.
  The remaining per-layer lever = the attention gather span (use the
  bucket that matches the actual context; already done by the user's
  bucketing) and the ~10% launch/tail polish. t/s ceiling ~17-19.

Note: input21 = the KV cache; the 91.1 MB fused attention WEIGHT lives
in the HLO as p1.497 [44488, 2048] = a separate input among the 23
(the 10.5/2.6 MB tensors include q/o shards and kernel tables). The
brief-13 "input21 = attention weight" relabel was wrong; the supervisor's
original accounting was right.

## 12. Brief 15: per-token breakdown — where the non-layer time goes

Residency (settled): input21 [5561,2,4,16,128] = the FULL-context KV
pool (identical in dec1_32 and dec1_128 def.json = bucket-independent).
Weights 278 MB x 40 layers = 11.1 GB + KV 91 MB x 40 = 3.6 GB =
**14.7 GB < 16 GB per core = weights are RESIDENT on device.** The
serving stack's STAGE_DIR=/dev/shm + STREAM_AHEAD=8 = load-time host
staging only (each layer's NEFF+weights uploaded once at model load,
<=8 ahead); per-token there is NO weight upload. The 1.32 ms/layer =
the HBM re-read of the resident weights per execute (the DMA geometry
already covered).

Measured (real graphs, 2 ranks, layer_bench):

| measurement | result |
|---|---|
| head1 (lm_head, bf16, 671 MB/rank) | **3.371 ms/token** |
| dec1_128 sync, 40 steps | 1.327 ms/layer |
| dec1_128 ASYNC inflight=8, 40 steps | **1.133 ms/layer** |
| dec1_128 ASYNC, 200 steps | 1.215 ms/layer |

- **The launch gap = 0.11-0.19 ms/layer and it is hideable** with
  NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=8: 40 x 0.19 = up to
  **7.8 ms/token**. The repo already sets this in cpp/neuron_model.cpp
  (setenv) — whether the user's Devstral serving path inherits it must
  be checked in their stack (image behemoth-neuron-server).
- Head = 3.37 ms bf16 (671 MB read/rank); the repo's HEAD_FP8=1
  (quantized lm_head rows, max 224) = measured 2.1 ms in PERF_NOTES =
  **-1.3 ms/token** on the shelf.
- Token arithmetic (measured device floor): layers 40 x 1.15-1.21 =
  46-48.5 ms + head 3.37 = **49-52 ms device**. The user's 15 t/s =
  66.7 ms/token -> **host-side = ~15-17 ms/token (23-25%)**: embedding,
  RoPE rows, page-table updates, logits read-back (131072 x f32 =
  524 KB = trivial), sampling over a 131k vocab, per-step host sync.
  This is now the LARGEST single chunk and it is outside libnrt.
- Concrete fixes, largest first:
  1. Verify/set NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=8 in the
     serving env (or the repo's neuron_engine.cpp path) = -5..-8 ms.
  2. HEAD_FP8=1 head buckets = -1.3 ms.
  3. Host: overlap sampling/readback with the next step's first layer
     (async execute of layer 0 while sampling); batch the host ops.
  Expected: 15 -> 17-19 t/s with 1+2 and ~19-21 with 1-3, if the host
  is really ~15 ms; measuring the true host split needs the real
  engine run (BEHEMOTH_NEURON_STEP_LOG=1) which needs the full prepared
  model set — flagged as the remaining unverified item.

## 13. Brief 16: fused one-call-per-token kernel — DEAD ON EXISTING DATA

The repo already built and measured this: trace_model.py "model R Q P L S"
compiles L layers (+ the optional head segment) into ONE graph, and
SERVING.md documents the sweep (neuronx-cc 2.26, torch-neuronx 2.9):

| path | 30 layers, 8 rows, 128 pages | per-layer |
|---|---|---|
| one fused graph (1 launch) | 48.2 ms | 1.61 ms |
| 30 separate executes (SYNC) | 51.7 ms | 1.72 ms |
| dec1_128 async inflight=8 (my brief-15 measurement, Devstral) | - | **1.133 ms** |

- The fused graph's entire benefit = removing the launch gap = **7%
  over SYNC**. The async-inflight execute queue removes the SAME gap
  (measured 1.327 -> 1.133 = 15% on the Devstral dec1 bucket) WITHOUT
  the fused graph's costs: the sweep showed the fused per-layer device
  time is never better than the per-layer path (numerics match to fp8
  noise; a layer-31 binding bug exists in the fused mode).
- The Devstral scale-up (N=2..40 fused compiles, new tiling, ~900
  input tensors bound per execute, cc_stream count limits) would cost
  hours of compile per point to chase a benefit the async path already
  delivers. NOT BUILT — killed by existing data.
- What remains for one-call-per-token = the HOST-side 15-17 ms/token
  (embedding, RoPE rows, page tables, sampling over 131k vocab,
  readback): moving those on-device (embedding gather + argmax are
  cheap graph ops; sampling via passed-in Gumbel noise) is the real
  lever, worth ~15-17 ms IF the host really costs that much — needs
  the real engine step-log (BEHEMOTH_NEURON_STEP_LOG=1) to split host
  vs device before building anything.
- Practical order of wins for 15 -> ~19 t/s without new graphs:
  1) verify NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=8 in the serving
     image (-7.8 ms/token if missing), 2) HEAD_FP8=1 (-1.3 ms),
  3) host-side: overlap sampling/readback with the next step's first
     layer execute (-5..-10 ms if the host is really 15-17 ms).

## 14. Brief 17: the faithful token loop (neuron/cpp/token_loop.cpp)

Harness: dec1_128 loaded 40x (one model per layer, distinct fp8 weight
buffers, TP=2 collectives), non-weight tensors (KV pool, tables,
scales) SHARED across layers by name, the layer-i output tensor feeds
layer i+1 (the same nrt_tensor_t in both sets), the EHS head at the
end, the host per step = position/slot writes only. Both ranks, 64+
tokens. Memory: ~12 GB/rank (the sharing = required: with distinct KV
pools = OOM at 40 layers).

| run | ms/token | t/s |
|---|---|---|
| full loop (40 layers + ehs), 64 tok | **67.96** | 14.7 |
| full loop, 128 tok | 68.99 | 14.5 |
| no-ehs (40 layers only) | 71.63 (noise-band) | 14.0 |

- **The loop calibrates to the user's serving: 68 vs 66.7 ms/token
  (1.5%).** The user's 15 t/s is now EXPLAINED end to end.
- Decomposition: 41 executes x ~1.33 ms (layer_bench per-layer) =
  54.5 ms; the measured 68 = **~330 us per execute of model-rotation
  overhead** (40-model rotation; layer_bench has zero model switches).
  The ehs head = noise-level inside the loop.
- **ASYNC CORRECTION (retracts brief 15's 1.133 finding)**: libnrt
  2.33 explicitly logs "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS is
  no longer supported and will be ignored" — the implicit async mode
  was REMOVED. The brief-15 1.133 vs 1.327 difference = run-to-run
  variance. The real async = the explicit nrta_execute_schedule API
  (nrt_async.h) = the genuine remaining lever: it schedules executes
  without the per-execute host round trip and may also hide the
  330 us model-switch cost. Not yet wired (moderate engineering).
- Context/batch (from the earlier layer_bench sweeps, both ranks):
  dec1_32/128/512 = 1.299/1.319/1.433 ms (the gather span grows
  ~0.16 us/page); dec4/dec8_128 = 1.642/1.797 ms = the batch
  amortization (68 us/row marginal).
- Floor estimate: the layer device work = 372 MB read/execute at the
  one-direction read ceiling (~230-260 GB/s/core) + the KV write-back
  riding concurrently = 1.32-1.43 ms/layer = the device floor
  53-57 ms/token = **17.5-19 t/s with ZERO overhead**. The rotation
  overhead (13 ms) and the host are the only remaining chunks, both
  addressable via the explicit async API + the on-device head (built).
