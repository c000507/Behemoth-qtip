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
