# Qwen3.5-397B fp8 on trn1.32xlarge: why batched decode does not scale, and what was fixed

Branch `qwen397-batch` (from `qwen397-fp8` 27655f4). Read docs/neuron-qwen397.md sections 8-8.10 first: this continues 8.9
(batching). All numbers: instance i-03935aa4aec5c547c, TP = 32, all 32 NeuronCores, short context (P = 32 pages), 2026-11-12/13.

## 1. Method

* `layer_bench` (cpp/layer_bench.cpp) was giving wrong numbers for anything with state: it bound the aliased state outputs to
  fresh tensors, so every launch copied the whole KV cache (seg20 R = 1: 17.4 ms per launch against 6.7 in the engine), and it
  did not hold `/dev/cpu_dma_latency`. Now `layer_bench NEFF steps ranks nstates` aliases the last n inputs to `output<s+1>`
  as the engine does, holds the latency fd, and reports the fastest and the median of 8 drained batches. With that the 20-layer
  segments reproduce the engine: 5.9 / 8.7 / 13.4 / 23.2 ms per launch at R = 1 / 2 / 4 / 8 = 17.7 / 26 / 40 / 70 ms per step
  (engine `layers_ms` 20.1 / - / 37.6 / 63.5).
* In-launch cost of one sub-block: bench graphs `seq_<kind>_<kind>...` (`sequence()` in qwen35_graphs.py: N sub-blocks of one
  kind, each with its own weights and state) at N = 2 and N = 8; (t8 - t2) / 6 is the cost of one block inside a launch, free of
  the 0.3 ms launch + rendezvous. MoE stages by `MOE_DEBUG` the same way.
* `neuron-profile capture --collectives-workers-per-node 32 --collectives-worker-count 32` on the N = 2 graphs (rank 0): bytes
  read from HBM, DMA / engine active times, instruction counts.

## 2. Decomposition of the baseline (bucket set g4: ROUTE=rank, `cache[table]` gather)

In-launch ms per sub-block:

| | R = 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| gdn block (x45 per step) | 0.090 | 0.143 | 0.174 | 0.174-0.213 | 0.287 |
| gqa block, P = 32 (x15) | 0.152 | 0.207 | 0.257 | **1.11-1.27** | 4.58 |
| moe block (x60) | 0.174 | 0.282 | 0.406 | **0.606** | 2.99 |

ms per decode step (45 gdn + 15 gqa + 60 moe + 3 launches of 0.3 ms), and the fitted cost of one more row:

| component | R = 1 | 2 | 4 | 8 | slope 1 -> 8, ms per extra row | at the DMA wall (250 GB/s) for R = 8 |
|---|---|---|---|---|---|---|
| gdn (5 projections, conv, recurrence, state slot read / write) | 4.1 | 6.4 | 7.8 | 7.8-9.6 | 0.5-0.8 | 5.7 MB per block = 1.0 ms per step: op bound |
| gqa (scores over 512 keys, f32 K\|V gather per row) | 2.3 | 3.1 | 3.9 | 16.7-19.0 | cliff at 8 rows | 9.8 MB per block = 0.6 ms |
| moe: pre-norm, shared expert, all-reduce, residual | ~4.4 | | 4.4 | 4.7 | ~0 | |
| moe: router matmul + softmax + top-10 (`rank`: [T, 512, 512] compares) | ~1 | | 4.5 | **7.6** | 0.95 | 8.4 MB f32 router per block = 2.0 ms per step |
| moe: slot union (topk over the used experts, comb gather) | 0.4 | | 1.3 | 3.1 | 0.4 | |
| moe: expert gather + gate/up + SiLU | \ | | 7.8 | **14.0** | \ | 21 MB per block = 5.0 ms |
| moe: expert gather + down | / 4.6 | | 5.0 | 5.0 | / 2.1 | 10.5 MB per block = 2.5 ms |
| moe total | 10.4 | 16.9 | 24.4 | **36.4** | 3.7 | 40.7 MB per block = 9.8 ms |
| launches (3 x 0.3) | 0.9 | 0.9 | 0.9 | 0.9 | 0 | |
| **segments, sum** | 17.7 | 27.3 | 37.0 | 62-66 | 6.4-6.9 | |
| segments, engine `layers_ms` | 20.1 | | 37.6 | 63.5 | 6.2 | |
| head (R rows x 7760 logits per rank) | 0.5 | 0.6 | 0.6 | 0.6 | 0.01 | |
| host: thread hand-off, logits assembly (R x 1 MB copied serially), OpenMP sampling | 3.5 | 3.7 | 3.8 | 4.5 | 0.15 | |
| **step wall** | 24.3 | (41.9 in the R = 4 bucket) | 43.0 | 69.0 | 6.4 | |

What the profiler says about the hypotheses in the brief:

* **"MoE bytes = R x 10 slices, even when rows share experts": true, and the graph reads each of them exactly once.**
  `hbm_read_bytes` per MoE block: 12.9 MB at R = 1 (8.4 router f32 + 3.9 gathered + 0.4 shared), 16.8 MB at R = 2, 40.7 MB at
  R = 8 (8.4 + 80 x 0.39 + 0.4); `hbm_write_bytes` ~0.3 MB. The gather is not materialised: the selected slices stream through
  software-dynamic DMA straight into the matmuls (81 MB in 0.488 ms active = **166 GB/s**, not 250). The slot list already is
  the union of the rows' experts (sorted, padded with unused experts); a static graph must provision 10 R slots, so the ~65-70
  distinct experts of 8 real rows cannot be exploited without a data-dependent fallback (15 % of the bytes at best).
  8 identical prompts at 63 ms instead of 69 is not explained by fewer bytes (the bytes are the same): not investigated further.
* **MoE is not only bytes.** At R = 8 the block takes 0.61 ms with the DMA active 0.31 ms, tensor engine 0.29 ms (6200 matmul
  instructions per block: one LDWEIGHTS + MATMUL pair per 64-row slot tile, so the count follows the slots), scalar 0.19, activation
  0.23, vector 0.19 ms. **The top-10 selection alone was 0.13 ms per block (7.6 ms per step, 12 % of the step)**: the
  topk-free `rank` form of 8.4 compares [T, 512, 512] = 2 M elements in 6 elementwise passes.
* **The attention gather is per row, and at 8 rows it fell off a cliff**: `cache[table].float()` (8 rows x 32 pages = 8.4 MB
  f32) is lowered at R = 8 to a static-DMA copy through HBM: 27 MB read + 8.6 MB written per block, 0.47 ms of static DMA
  (R = 4: 7.5 MB read, nothing written). 15 gqa layers = 17-19 ms of the 63.5 ms step. The same happens to the MoE at 16 rows:
  the 160-slot gather turns into a static scan of the expert tensors (570 MB read per block, 2.4 ms): that is the R = 16
  "disproportionately slow" MoE of 8.9.
* **gdn**: 5.7 MB per block, DMA 0.07 ms; the rest is ~1300 small instructions per block (scalar / activation / vector
  0.06-0.075 ms each at R = 8). Per row +0.012-0.018 ms per block. `GDN_GATHER=flat` and `GDN_EINSUM=elem` (probes, default
  off) change nothing measurable (0.148-0.193 against 0.174 at R = 8; 0.093-0.099 against 0.090 at R = 1).
* **all-reduces, head, launches**: not a row cost (head 0.5-0.6 ms for 1..8 rows).
* **host**: 3.5-4.5 ms per step at every R (15 % of the R = 1 step): the sampling OpenMP region wakes / spins a 128-thread
  team for <= 8 rows (and its spinning threads delay the 32 rank threads of the next launch), and the 32 logit shards were
  read into per-step vectors and copied serially.
* **bucket padding**: 2 streams ran in the R = 4 bucket (41.9 ms where an R = 2 bucket costs 26).
* The first extra row is the dearest (R = 1 -> 2: +2.6 ms per 20-layer launch, +7.7 ms per step; 2 -> 4: +1.2 ms per row and
  launch; 4 -> 8: +0.8). Profile R = 1 -> 2 of the MoE: +30 % bytes but +66 % matmul instructions and +70-75 % scalar / activation /
  vector instructions: the compiler's schedule for T = 1 is special-cased; nothing in the graph source to fix.

## 3. Fixes (all measured; `ROUTE=pad64 ATT_GATHER=flat` are graph build flags, the rest is the engine)

| fix | per block, ms | per step at 8 rows |
|---|---|---|
| **`ROUTE=pad64`**: top-10 membership by `torch.topk` on the rows padded to 64 (the prefill shape, where the device's topk is correct; 4 / 8 rows are not, 8.4); the weights are then computed as in the rank form | moe 0.174 / 0.282 / 0.406 / 0.606 -> 0.166 / 0.258 / 0.358 / **0.473** at R = 1 / 2 / 4 / 8 | -8.0 ms |
| **`ATT_GATHER=flat`**: `index_select` over the flattened page table instead of `cache[table]` | gqa 0.152 / 0.207 / 0.257 / 1.27 / 4.58 -> 0.145 / 0.184 / 0.196 / **0.283** / 0.522 at R = 1 / 2 / 4 / 8 / 16 | -13 to -15 ms |
| 20-layer segment, both flags (layer_bench, ms per launch) | R = 1 / 2 / 4 / 8: 5.90 / 8.69 / 13.40 / 23.24 -> 5.73 / 8.31 / 10.70 / **14.04** | 69.7 -> 42.1 ms in the 3 launches |
| engine: every rank reads its vocabulary shard straight into the logits buffer; `OMP_NUM_THREADS=8 OMP_WAIT_POLICY=active` for the sampling loop | host 3.5-4.5 -> **1.0-1.6 ms** per step (forward - layers - head 2.4 -> 0.07 ms at R = 1, sample 1.1-5 -> 0.92) | -3 ms at every R |
| engine: row buckets made of segment NEFFs only (no sub-block graphs needed for R = 2 / 3 / 6), the narrowest row bucket that has a page bucket for the rows | 2 streams 41.9 -> 26.3 ms; 3 streams 30.1; 5-6 streams 41.3 | |

Numerics: on the device, single-block NEFFs with random weights of the real shapes (`tests/test_qwen35_device.py`,
`DEVICE_OUT` keeps the outputs): `ROUTE=pad64` vs `rank` (routing stage and whole MoE block, R = 2 / 4 / 8) and
`ATT_GATHER=flat` vs `index` (R = 2 / 8) are **bit-identical** (max |diff| 0, all 32 ranks equal). The fixes do not change
what a block computes.

Tried and dropped (measured, no gain):

| idea | result |
|---|---|
| `ROUTE=tour` (exact tournament top-k: 8x fewer compared elements) | 0.222 ms route stage at R = 8 against 0.212 (rank) and 0.122 (pad64): more, smaller ops cost more. The element count was the wrong model. `ROUTE=rank2` (3 passes): 0.181 |
| `MOE_SLOTS=onehot` | 0.593 against 0.606 (noise) |
| `MOE_SPLIT` (rows in groups, one slot union per group) at R = 8: groups of 4 / 2 / 1 | 0.76 / 0.81 / 0.80 ms per block against 0.47-0.61: worse |
| dense all-experts MoE (`MOE=all`) at 16 / 32 rows | 1.99 / 2.37 ms per block (201 MB per block): 120-142 ms per step for the MoE alone. Crossover with the gather (0.036-0.066 ms per row and block) is beyond 32 rows: never useful for decode |
| R = 16 with `MOE_SPLIT=8` (two 80-slot unions, avoids the 160-slot cliff) | moe 2.99 -> 1.12 ms per block; see section 4 for the R = 16 segment |
| gdn gather / einsum forms | nothing (section 2) |
| bf16 router weights (8.4 -> 4.2 MB per block, -1 ms per step at the measured DMA rate) | not built: a bf16 x bf16 dot would round the router logits to bf16 unless the compiler keeps the f32 accumulate; needs its own numerics check |

## 4. Result (engine, 8 distinct prompts of validate_seg.sh, 100 new tokens, quiet box)

Bucket set T2: prefill sub-block graphs Q = 64 / 256 of g4 + `seg20` R in {1, 2, 3, 4, 6, 8} x P in {32, 128, 512} built with
`ROUTE=pad64 ATT_GATHER=flat` + heads; 43 NEFFs load next to the weights (12.04 GiB + 1.04 GiB state per core).
`OMP_NUM_THREADS=8 OMP_WAIT_POLICY=active`.

| streams | bucket | before: ms/step, tok/s | now: `layers_ms` | now: ms/step | aggregate tok/s |
|---|---|---|---|---|---|
| 1 | R = 1 | 24.3, 40.9 | 17.6 | **19.1** | **52.3** |
| 2 | R = 2 (was R = 4) | 41.9, 47.5 | 24.7 | **26.3** | **75.5** |
| 3 | R = 3 | - | 28.4 | 30.1 | 99.3 |
| 4 | R = 4 | 43.0, 93.0 | 32.3 | **33.9** | **117.5** |
| 6 (5.3 rows on average) | R = 6 | - | 39.6 | 41.3 | 128 (6 full rows: 145) |
| 8 (7.3 rows on average) | R = 8 | 69.0, 116 (106) | 42.7 | **45.9** | **159** (8 full rows: 174) |
| 16 | 2 x R = 8 per step | 258, 59 | 84.1 | **87.8** | **174** |

Per extra row: 6.2 -> 3.6 ms per step (1 -> 2: 7.1; 2 -> 4: 3.8 per row; 4 -> 8: 2.6 per row).

Tokens: the R = 1 stream is identical to the validated configuration's (100 / 100). Batched rows are identical to the old
batched output when the requests are scheduled the same way (4 / 4 rows in one 4-stream run, 3 / 4 and 4 / 8 in others): the
prompts are prefilled one after the other while earlier requests already decode, so which decode bucket a request's first
tokens run in depends on arrival timing; rows that change bucket take the other branch of a near tie (8.9). Oracle on the
batched rows: see ORACLE below.

## 5. Is 300 tok/s at 8 streams (<= 26.7 ms per step) reachable? No. Floor with this layout: 33-38 ms (210-240 tok/s)

Per step and core at 8 rows the graphs must read: 60 x 31.5 MB of selected expert slices (every selected expert is read in
full once per step across the 32 cores, whatever the sharding: 80 x 12.6 MB per layer) + 60 x 8.4 MB router (f32) + 0.27 GB
attention weights = **2.7 GB**: 10.7 ms at the 250 GB/s wall, **16 ms at the 166 GB/s the gathered stream reaches**. The R = 1
step already costs 17.6 ms in the segments with only 1.06 GB (6.4 ms of DMA): ~11-13 ms of it is op-count bound (norms,
recurrence, routing, all-reduces, ~100 small ops per layer), and that part does not shrink with more rows.

| component at 8 rows | now, ms per step | floor | what blocks it |
|---|---|---|---|
| moe gather + gate/up + down | 19 | 7.6 (bytes at 250 GB/s) - 11.4 (at 166 GB/s) | 1.9 GB per step and core; 6200 slot-tile matmul instructions per block; needs a hand-written kernel that streams the selected slices as wide tiles |
| moe router + top-10 + slot union | 5.7 | ~3 | f32 router 0.5 GB per step; topk |
| moe base (norm, shared expert, all-reduce) | 4.7 | 4.7 | op count, same as R = 1 |
| gdn | 7.8-9.6 | 4.1 (the R = 1 cost) + per-row state work | op count; +0.5-0.8 ms per row |
| gqa | 4.2 | 2.3-3 | per-row K\|V gather (4.2 MB at 8 rows x 32 pages) |
| launches + head | 1.5 | 1.5 | |
| host | 1.6-3 | ~1 | argmax over 248k logits per row; device argmax would remove ~1 ms |
| **total** | **45.9** | **~33-38** | |

The coordinator's arithmetic (31-35 ms, 230-260 tok/s) is right in structure; measured corrections: the gathered stream runs
at 166 GB/s, not 250 (+4 ms), dedupe is not available to a static graph (80 slots, not 65-70), and the R = 1 base is 17.6 ms
of which only a third is bytes. 300 tok/s would need the MoE bytes AND all per-row small-op costs to be free: even with a
zero-cost MoE gather the step is 17.6 (R = 1 segments) + 3-5 (gdn / gqa rows) + 1.5 + 1 = 23-25 ms, i.e. the 26.7 ms budget
leaves 2-3 ms for 1.9 GB of expert bytes that need 7.6 ms at the wall.

Beyond 8 rows the aggregate saturates near 175-190 tok/s: every further row adds 3.9 MB per layer and core (2.5-4 ms per
step), so 16 streams = two R = 8 launches (174 tok/s).

What would change the picture (not done):

* An NKI kernel for the gathered MoE (indirect DMA of whole [64 | 32, 4096] slices as 4 KB rows, weights as the moving
  operand as in nki_mlp.py): target 250 GB/s and ~1/4 of the matmul instructions: -8 to -11 ms at 8 rows (-> ~35 ms, 230
  tok/s). The expert slices are stored output-major ([E, 64, 4096]); the kernel needs the contraction axis on the
  partitions, i.e. a re-laid-out copy of the experts (the engine can build it at load). Earlier indirect-DMA kernels in this
  repo measured 9-137 GB/s depending on row size (PERF_NOTES), so the gain is not certain.
* Fewer bytes per token: nothing in the layout helps (EP concentrates the same bytes on fewer cores, 8.7). Only a smaller
  top-k or shared experts across rows (a model change) would.
* Device-side argmax / top-k of the head for greedy and top-k sampling: -1 ms per step.
* A second instance: throughput per stream is latency bound here (19 ms per token single stream), the batch only adds
  ~3.6 ms per row.

## 6. Serving settings

* Graphs: `ROUTE=pad64 ATT_GATHER=flat` for every decode bucket (`MOE=gather`, `PAGES=4097 SLOTS=17`); prefill buckets as
  before. Row buckets R in {1, 2, 3, 4, 6, 8}; page ladder R = 1 to 4096, R = 4 to 2048, R = 8 to 1024, R = 2 / 3 / 6 to 512
  (a row set that needs more pages falls to the next wider bucket that has them, else it is split).
* Engine env: `OMP_NUM_THREADS=8 OMP_WAIT_POLICY=active` (the sampling loop has at most 8 rows per launch; a 128-thread
  team costs 1-5 ms per step). `BEHEMOTH_NEURON_STEP_LOG=1` prints `exec_ms` / `plan_ms` next to `layers_ms`.
* Do not serve R = 16 (section 4).

---

# Part 2 (branch `qwen397-batch2`, 2026-11-12/13): second pass over the same question

Continues sections 1-6 (bucket set T2 = `ROUTE=pad64 ATT_GATHER=flat`, 45.9 ms per step at 8 streams). Same instance, same
method (`seq_<kind>...` chains at N = 2 / 8 for the in-launch cost of one block, 20-layer segments in `layer_bench`, the
engine step log). Bucket sets of this part: **T3** (gathered decode, R = 1..16) and **T64** (dense decode, R = 64).

## 7. Decomposition against the row count, as served now (T3)

In-launch ms per sub-block (best variant per R, see 8), x 45 gdn + 15 gqa + 60 moe + 3 launches = one decode step:

| per block, ms | R = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn | 0.092 | 0.143 | | 0.163 | | | | 0.165 | | | 0.287 |
| gqa, P = 32 (`flat`) | 0.149 | 0.184 | | 0.196 | | | | 0.271 | | | 0.522 |
| moe, T2 (`pad64`, replicated router, slot union) | 0.171 | 0.257 | | 0.352 | | | | 0.472 | | | 2.99 |
| moe, now | 0.123 | 0.202 | 0.278 | 0.325 | 0.380 | 0.454 | 0.457 | 0.459 | 0.785 | 0.720 | 0.871 (2 unions of 80) / 2.77 |

| per decode step, ms | R = 1 | 2 | 3 | 4 | 6 | 8 | slope 1 -> 8 per extra row | what bounds it |
|---|---|---|---|---|---|---|---|---|
| gdn x 45 (5 projections, conv, recurrence, per-row state slot read / write) | 4.1 | 6.4 | | 7.3 | | 7.4 | 0.47 | op count; 5.7 MB per block = 1.0 ms per step at the DMA wall |
| gqa x 15 (512 keys per row, per-row K\|V gather) | 2.2 | 2.8 | | 2.9 | | 4.1 | 0.27 | op count; per-row gather |
| moe: norm, shared expert, all-reduce, residual | ~4.4 | | | 4.4 | | 4.7 | ~0 | op count |
| moe: router (+ its all-reduce), softmax, top-10 | 1.5 | | | ~3 | | ~4.5 | 0.4 | was 8.4 MB of f32 router per block and rank (T2: 3.3 ms at R = 1); now 0.26 MB |
| moe: slot selection (union of the rows' experts) | 0 (row slots) | 0 | 0 | 0 | 3 | 3.1 | 0.45 | topk(10 R of 512) + combine-weight gather; only R <= 4 can skip it (8.2) |
| moe: 10 R expert slices, gate / up, SiLU, down | 1.5 | | | ~12 | | 15.3 | 2.0 | 3.9 MB per row and block: 0.94 ms per step and row at 250 GB/s, 1.42 at the 166 GB/s the gathered stream reaches |
| launches (3 x 0.3) | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0 | |
| **segments (3 launches), layer_bench** | **14.5** | 21.5 | 24.4 | 28.3 | 39.0 | **41.2** | 3.8 | |
| segments, engine `layers_ms` | 14.9 | 22.0 | 24.9 | 29.3 | 39.8 | 41.6 | 3.8 | |
| head (R rows x 7760 logits per rank) | 0.48 | 0.59 | 0.60 | 0.62 | 0.62 | 0.56 | 0.01 | |
| host: logits read, greedy argmax over 248k logits per row, scheduler | 0.2 | 0.2 | 0.3 | 0.4 | 0.4 | 0.55 | 0.05 | was 1.0-1.6 ms (and +3 ms on steps with 8 fresh rows, 8.3) |
| **step wall** | **15.6** | **22.8** | **25.8** | **30.3** | **40.8** | **42.7** | 3.9 | |
| aggregate tok/s | 64.0 | 87.6 | 116 | 132 | 147 | **187** | | |

The cost of one more row is not constant:

* **1 -> 2 rows costs 7 ms, 2 -> 4 rows 3.7 ms per row, 6 -> 8 rows 1.0 ms per row.** The last figure is the byte floor
  (3.9 MB of expert slices per row and block = 0.94 ms per step at 250 GB/s): at 6-8 rows a further row is already as
  cheap as this layout allows. The whole excess sits below 6 rows, in steps of the compiler's schedule, not in bytes:
  per block the second row adds 0.079 ms to the MoE (0.024 would be its bytes), 0.051 ms to gdn (the two rows' state
  slots), 0.035 to gqa.
* **The MoE block has plateaus and cliffs in the slot count**: 0.325 ms at 40 slots, 0.380 at 50, **0.454 / 0.457 /
  0.459 at 60 / 70 / 80** (rows 7 and 8 are free once the sixth is paid), then 0.72-0.79 at 100-120 slots and 2.8-2.9 ms at
  140-160 (the gather becomes a static scan of the expert tensors, section 2). **8 rows is the last good gathered
  bucket**; 16 rows as two unions of 80 slots (`MOE_SPLIT=8`) cost 0.87 ms per block.
* A plain MLP-like block (RMSNorm, fp8 quantisation, two fp8 matmuls, SiLU, residual: `tests/bench_layout_graph.py`)
  costs 0.034 / 0.037 / 0.037 ms at 1 / 2 / 8 rows, so row-major activations are not the problem (a column-major
  variant, hidden axis first, is slower: 0.044-0.047). What scales badly with rows is everything indexed per row:
  the expert gather, the state slots, the K|V page gather, the router's top-k.

## 8. Fixes of this pass

| fix | effect |
|---|---|
| **`ROUTER=shard`**: every rank held the whole f32 router (8.4 MB read per MoE block = 65 % of the block's bytes at one row). Now rank r holds the router columns of its hidden slice (`rws` [512, 128], `rcols`; the engine derives them at load from the prepared replicated router) and the partial logits are summed by one more f32 all-reduce (15-30 us inside a launch). Every rank receives the same bits, so every rank selects the same experts (device test: ranks differ by 0) | moe block 0.171 -> 0.144 (R = 1), 0.257 -> 0.221 (2), 0.352 -> 0.339 (4), 0.472 -> 0.459 (8); 20-layer segment 5.68 -> 5.10 ms at R = 1 |
| **`MOE_SLOTS=row`** (R <= 4): slot r K + j = the j-th expert of row r, straight from the router's top-k; no union over the rows, no top-k over the experts, no [T, E] combine matrix. `ROW_FORM=launder` (the indices through f32 and back) is the variant that the compiler streams at 2-4 rows | segment 5.10 -> **4.83** (R = 1), 7.92 -> **7.20** (2), 8.88 -> **8.12** (3), 10.22 -> **9.43** (4) ms per launch. At 6 / 8 rows every row form is a cliff (the gather is materialised through HBM: 38 MB written per block, 1.1-5 ms), so R >= 6 keeps the union |
| sampling: greedy argmax + finite check of a contiguous row in one vectorised pass (`runtime/src/sampling.cpp`) | 0.92 -> 0.10-0.27 ms per step (75 us per 248k-logit row instead of 900) |
| embedding table mapped with `MAP_POPULATE` | the first use of a token's embedding row was a page fault inside the step: steps with 8 fresh rows spent 3.1 ms between `forward` entry and the launch (T2: 46.6 ms forward for 43.5 ms exec); now 0.1 ms |
| `GDN_DEC=1` (decode conv state by a static slice), `LIN_CSE=1` (one fp8 quantisation per projected tensor) | bit-identical; within noise (gdn 0.177 -> 0.165 at R = 8) |

Measured and not used:

| idea | result |
|---|---|
| `MOE_BMM=1` (every row contracts only its own 10 slots, batched matmul) | equal to `row` at R = 1 / 2 (4.84 / 7.18 ms per launch), worse at 3 / 4 (9.63 / 11.2), 5 ms per block at 8 rows |
| row slots at 8 rows through a second top-k (`ROW_FORM=topk`), which the compiler streams | 0.468 ms per block against 0.459 for the union: the union stage itself was never the cost |
| `GDN_FORM=fused` (decode recurrence with ONE contraction of the stored state: mem = d (S k), out = d (S q) + (k . q) delta; equal up to f32 rounding) | sub-block: 0.163 -> 0.135 (R = 4), 0.156 -> 0.145 (R = 8), but 0.72 ms at R = 1 (!); in the 20-layer segments -0.1..-0.25 ms per launch at best, not reproducible between runs. Not served (it is not bit-identical) |
| column-major activations (hidden axis first) | slower (section 7) |
| R = 10 / 12 / 14 buckets | cliff (section 7); R = 5 / 7 would cost 0.380 / 0.457 ms per MoE block: an R = 5 bucket would save ~4 ms per step for exactly 5 streams, not built |

Numerics (`tests/test_qwen35_device.py`, single MoE blocks with random weights of the real shapes, device vs device):
`ROUTER=shard` changes the router logits in the last f32 bit (32 partial sums instead of one dot product): block output
max |diff| 1.4e-6..9e-5 (rel 5e-7..1.4e-5) against `pad64` at R = 1..8, all 32 ranks bit-equal; the row forms are
bit-identical to the union form on top of that. Device vs CPU mirror unchanged (3.2e-2..3.6e-2 rel L2, the fp8 level).

## 9. Result of this pass (engine step log, distinct prompts of `prompts8.txt`, 100 new tokens, quiet box)

Bucket set T3 / T3F (50 NEFFs: prefill sub-block graphs of g4 + decode `seg20` R in {1, 2, 3, 4, 6, 8, 16} + heads),
`OMP_NUM_THREADS=8 OMP_WAIT_POLICY=active`. "before" = the configuration of docs/neuron-qwen397.md 8.13 (g10).

| streams | bucket | g10 (8.13) | T2 (section 4) | **T3: ms per step** | `layers_ms` | **aggregate tok/s** |
|---|---|---|---|---|---|---|
| 1 | R = 1 | 24.5 ms, 41 | 19.1, 52.3 | **15.6** | 14.9 | **64.0** |
| 2 | R = 2 | (R = 4 bucket) | 26.3, 75.5 | **22.8** | 22.0 | **87.6** |
| 3 | R = 3 | | 30.1, 99.3 | **25.8** | 24.9 | **116** |
| 4 | R = 4 | 41.4, 96.5 | 33.9, 117.5 | **30.3** | 29.3 | **132** |
| 6 | R = 6 | | 41.3, 145 | 40.8 | 39.8 | 147 |
| 8 | R = 8 | 56.2, 128 | 47.6 (8 rows) / 44.9 (7 rows), 168 | **42.7** | 41.6 | **187** (8 full rows; 171 over the run with 7.3 rows on average) |
| 16 | R = 16 (`MOE_SPLIT=8`: two unions of 80 slots) | 258, 59 | 87.8, 174 (2 x R = 8) | **73.3** | 71.6 | **218** |
| 1 after an 8k fill | R = 1, P = 512 (`ATT_BF16=1`) | 25.7, 38.8 | | **16.9** | | 59.0 |

* Tokens: the R = 1 stream is identical to the validated configuration's (100 / 100, and the 8k continuation 32 / 32
  against both of agent A's references). Batched rows: 5 of 8 identical to T2's over 100 tokens, the others leave at
  tokens 26-45 (near ties; the sharded router changes the logits in the last f32 bit, and a row's bucket depends on
  arrival timing as before). Oracle: section 12.
* All 50 NEFFs of T3F load next to the weights (12.05 GiB + 1.04 GiB state per core).

## 10. Is 300 tok/s at 8 streams reachable? No

The step at 8 rows is 42.7 ms; 300 tok/s needs 26.7. What is left, per step:

| at 8 rows | ms | floor | comment |
|---|---|---|---|
| 80 expert slices per MoE block: gather, gate / up, SiLU, down | 15.3 | 7.6 (bytes at 250 GB/s) .. 11.4 (166 GB/s) | 1.9 GB per step and core. The tensor engine ingests 128 weight elements per cycle whatever the operand roles, and every selected slice must pass through it once: this part cannot go below the byte floor |
| MoE router + top-10 + slot union | 7.6 | ~3 | compiler schedule (the union is needed at >= 6 rows, 8.2) |
| MoE base (norm, shared expert, all-reduce, residual) | 4.7 | 4.7 | op count, same as R = 1 |
| gdn | 7.4 | 4.1 (R = 1) | per-row state slots |
| gqa | 4.1 | 2.2 | per-row K\|V gather |
| launches, head, host | 2.0 | 1.5 | |
| **total** | **42.7** | **~24-28 only if every item reached its floor at once** | |

Even with a perfect MoE kernel (expert part at the byte floor: -4 to -8 ms) the step is 35-39 ms (205-230 tok/s): the
non-expert work alone (router, base, gdn, gqa, launches, host = 26 ms) already is the whole 26.7 ms budget. Reaching 300
tok/s at 8 streams would need BOTH the expert part at the 250 GB/s wall AND the per-row overheads of everything else
removed (the R = 1 cost of 14.5 ms + 7 rows x 1.0 ms of bytes = 21.5 ms is the true floor of this layout), i.e. a
hand-written kernel for the whole layer, not only the MoE. A second instance (2 x 4 streams = 2 x 132 = 264 tok/s; 2 x 8
= 374) or a model-side change (fewer experts per token) are the other ways.

What does scale: the aggregate keeps growing with more streams only up to 8 rows per launch (187), 16 rows in one
launch give 218, and beyond that the gathered MoE falls off (section 7).

## 11. Merged serving set (T3M) and oracle

T3M = agent A's `g10` prefill ladder (sub-block Q = 64 / 256 / 512, prefill segments `seg_gdn_gdn_gdn_gqa_1_{256,512}_P`
with `AR_BF16=1` (`ATT_BF16=1` at Q = 512)) + the decode buckets of this pass; 71 NEFFs, loads next to the weights (12.05
GiB + 1.04 GiB state per core), `BEHEMOTH_PREFILL_CHUNK=512`. Archive: `s3://912935854432-neuron-build/qwen397/batch2/graphs/T3M.tar`
(`<bucket>/graph.neff` + `meta.json`; md5s in `T3M.MANIFEST.txt` inside `logs_e2e.tgz`).

| check on T3M | result |
|---|---|
| 1 stream | 15.8 ms per step, 63.2 tok/s; tokens = T3 = the validated configuration (100 / 100) |
| 8 streams | 43.0 ms per step at 8 rows, 186 tok/s; all 8 rows identical to the T3 run |
| 8k-token prompt (Q = 512 prefill segments) | prefill 881 tok/s after the first chunk; the 32 continuation tokens identical to both of agent A's references; decode 16.9 ms per token on a quiet box (T3F) |

Oracle (f32 teacher-forced HF model on the CPU, `tests/test_qwen35_oracle.py`, the 8 rows of the 8-stream run, 100 new
tokens each): **T3 722 / 734 (98.4 %)**; T2 725 / 733; agent A's configuration (`v20_d8`) 724 / 733. Every disagreement
but one is the oracle's second choice (margins 0.2-1.6). The three configurations are equally valid fp8 realisations.

## 12. Serving settings

* Decode graphs (`MOE=gather PAGES=4097 SLOTS=17`, all with `ROUTE=pad64 ATT_GATHER=flat ROUTER=shard GDN_DEC=1 LIN_CSE=1`):
  R = 1: `MOE_SLOTS=row`, P 32 .. 4096; R = 2 / 3: `MOE_SLOTS=row ROW_FORM=launder`, P 32 / 128 / 512; R = 4: the same,
  P 32 .. 2048; R = 6 / 8: slot union (T2's R = 6 graphs; R = 8 P = 32 rebuilt with the sharded router), P to 512 / 1024;
  R = 16: `MOE_SPLIT=8`, P = 32. `ATT_BF16=1` for the new P >= 128 buckets. Choose the variant per R by measuring the
  20-layer segment: the compiler's lowering of the expert gather changes with the row count (section 8).
* Engine: this branch (router shards derived at load: no change to the prepared set; `MAP_POPULATE` embedding; vectorised
  greedy sampling). Env `OMP_NUM_THREADS=8 OMP_WAIT_POLICY=active`, `/dev/cpu_dma_latency` writable.
* `--max-batch-requests 16` only if 16-row steps (73 ms, 13.6 tok/s per stream) are acceptable; 8 otherwise.
