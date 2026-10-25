# Decode performance, round 2: lane-owns-row layout, fused elementwise chain

Date: 2026-10-26. Machine: 2x Radeon AI PRO R9700 (gfx1201, Wave32, 32 WGPs, 64 KiB LDS),
TP=2, kernel 7.0.0-38, amdgpu DKMS 7.1.9, ROCm 10.1 (HIP 7.16, clang 24). Baseline: commit
`a12a658` (sorted layout + `sorted-lds`, `BATCHED_SORTED_REPORT.md`). Commits of this round:
`b0b3d7e` (lane layout + kernels), `f66ba14` (fused elementwise), plus the report/results commit.
Raw numbers: `perf/results/round2/` (bench JSON `final_rows*.json`, e2e `*_c{1,4}.json`,
`launch_rocm10.txt`), scripts `perf/e2e_round2.sh`, `perf/server_cycle.sh`, `perf/bench_sweep.sh`,
`perf/summarize_round2.py`.

**GPU clock setting for every number below:** OverDrive `OD_SCLK_OFFSET -500 MHz`,
`OD_VDDGFX_OFFSET 0 mV`, `power1_cap 210 W`, perf level `auto`, both cards (the point chosen in
`refusal_bench/GPU_OD_LIMIT_REPORT.md`, applied by `~/gpu_od_apply.sh` from `~/kernel.sh`; verified
in sysfs before every GPU run). It is ~9 % slower than the stock clocks the earlier reports used
(rows=1 sorted layer 0.537 -> 0.598 ms), so all comparisons here are against baselines
re-measured at this setting.

## TL;DR

| | baseline `a12a658` (sorted) | this round (`--code-layout lane`) | change |
| --- | ---: | ---: | ---: |
| c=1 inter-token (GPU step) | 85.7 ms (11.7 tok/s) | **76.4 ms (13.1 tok/s)** | -11 % |
| c=4 step | 222.1 ms (18.0 tok/s aggregate) | **121.6 ms (32.9 tok/s aggregate)** | -45 % (1.83x) |
| c=8 step | **both GPUs fell off the PCIe bus** during the baseline c=8 warm-up (section 6) | not run (same risk) | - |
| projections per layer per rank, rows 1 / 2 / 3 / 4 / 8 (bench, lane-only modes) | 0.619 / 1.007 / 1.379 / 1.772 / 1.961 ms | **0.687 / 0.731 / 0.936 / 0.956 / 1.677 ms** | +11 % / -27 % / -32 % / -46 % / -14 % |
| rows 16 / 32 / 64 / 256 (sorted-lds / sorted-dense decoding the lane layout) | 2.107 / 2.569 / 3.008 / 9.839 | 2.031 / 2.420 / 2.932 / 9.410 | -4 % / -6 % / -3 % / -4 % |

* **Lane-owns-row layout (`--code-layout lane`, new default for the server):** every 16-row tile
  is stored so that lane `l` of a Wave32 owns output row `l/2` and walks that row's codes in sorted
  order; chunk `j` holds the `j`-th entry of each lane's own stream. Accumulation is per lane in
  registers (fp32 via the `v_dot4_f32_fp8_fp8` accumulator operand): **no LDS atomics, no fixed-point
  split, no scalar loads in the hot loop.** `M` batch rows cost `M` activation reads per entry
  instead of `M` atomics, so rows 2-4 run in one pass at ~1.0 ms/layer (the old per-row `sorted-sN`
  passes cost rows x 0.6 ms, and `sorted-lds` 1.9-2.0 ms). The codebook gathers stay L0-resident:
  a CPU simulation of the real layer-0 codes (`lane_sim`, section 2) gives a 97.5 % L0 hit rate for
  8 lock-step waves (vs 6 L2 line requests per 32 entries for the fully sorted stream).
* **Elementwise fusion** (`f66ba14`): norm+quantise into the projection workspace once for q/k/v
  and once for gate/up, bf16 written by the merge kernel, rope(q)+rope(k)+quantise(q) in one
  kernel, cast+residual in one kernel: 15 of ~42 launches per layer gone, rounding bit-identical.
  Measured on the sorted layout: **c=1 85.7 -> 77.6 ms (-8.1 ms), c=4 222 -> 212 ms.**
* **Decode attention partition 32 keys** (`BEHEMOTH_ATTN_PARTITION`, default was effectively
  no partition below 1024 tokens): a single request's 48 heads spread over ceil(ctx/32) x 48 blocks
  instead of 48 one-wave blocks: c=1 79.9 -> 76.4 ms, c=4 124.6 -> 121.6 ms.
* **The lane kernel alone is 3 % slower than `sorted-s8` at c=1 end to end (79.9 vs 77.6 ms)** and
  11 % slower in the (MALL-assisted) bench; it is 1.7x faster at c=4. With the attention change the
  lane server is faster than the sorted one at every concurrency. Diagnostics (section 4) show
  the rows=1 kernel is bound by the idx/nibble stream and VALU, not by the gather or LDS: with the
  gather and LDS reads removed it still takes 0.134 ms for gate (vs 0.144 complete). Closing the
  last 3 % at c=1 needs fewer stream bytes per entry (section 7).
* **gfx12 cache policy of the stream loads matters and the earlier finding does not transfer:**
  for the lane kernels, `SCOPE_SYS` loads (policy 24, what `sorted-lds` uses for its decode) are
  8-10 % slower end to end than regular loads (c=1 87.5 vs 79.9 ms, c=4 132 vs 125 ms); non-temporal
  hints are 25 % slower in the bench. Regular loads are the default (`lane_stream_policy = 0`,
  `BEHEMOTH_LANE_POLICY` overrides).
* **HIP graphs still do not help on ROCm 10.1:** 1000 back-to-back empty kernels 2.28-2.75 us each,
  the same 1000 kernels as a graph 2.32 us each (`perf/results/round2/launch_rocm10.txt`).
* Hardware: the baseline c=8 warm-up (8 x 24 tokens, `sorted` layout, stock `a12a658` binary) took
  both GPUs off the bus 12 minutes after the OD agent handed over; recovered with a BMC power cycle
  and `~/kernel.sh`. No c=8 measurements were attempted afterwards. c=4 was clean for the
  ~20 server cycles of this round.

## 1. Design: lane-owns-row layout (`include/behemoth/sorted_layout.hpp`, `src/sorted_layout.cpp`)

The sorted layout of the previous round sorts all `16 x groups` entries of a tile by code; every
entry then has an arbitrary row, so the per-entry result must be scattered into 16 row accumulators
(`ds_add_u64`, ~10.7 clk per 32 entries, the limiter at rows=1 and `M` atomics per entry for `M`
rows). The lane layout keeps the sortedness *per row*:

* Tile = 16 output rows x up to 2047 groups (one group tile for every projection of this model;
  `lane_layout(max_groups_per_tile)` balances several tiles when `in/8 > 2047`). For row `r` the
  codes of the tile's groups are sorted; even positions go to lane `2r`, odd to lane `2r+1`.
* Chunk `j` = the `j`-th item of every lane's stream. Storage is lane-major inside a segment of
  `S = 32` chunks: `idx[(seg*32 + lane)*32 + j] = group | (delta & 31) << 11` (uint16) and
  `hi[(seg*32 + lane)*4 + j/8]` nibble `j%8` = `delta >> 5`, so a lane loads its whole segment with
  four 16-byte idx loads and one 16-byte nibble load, and nothing is shared between lanes.
  `segment_codes[seg*32 + lane]` is the lane's running code before the segment (any segment can
  start independently: splits, decode). `code_j = (code_{j-1} + delta_j) & 0xffff`, 9-bit deltas.
* Dummies have `group == 2047` (the kernels clamp the group to a zero activation slot): gap
  escapes (`delta 511`, gap > 510 happens for ~1 in 5e5 entries at this code density), tail
  padding of the last segment, rows beyond `out`, and optional drift-limiter waits
  (`drift_limit`, off by default: section 2 shows it is not needed at full K).
* Bytes per entry: 2 (idx) + 0.5 (nibble) + 0.0625 (segment codes) = 2.56 B before dummies; on
  layer 0: gate 58.3 MB vs 56.7 MB (sorted) vs 44.0 MB (2-byte codes), 3.3 % dummies (segment
  padding + escapes), q 25.1 MB, o 26.2 MB (8.3 %). Same KV budget as the sorted layout: 4569 pages
  (73k tokens) at `--memory-fraction 0.9`.
* Encoder: `lane_layout()` (OpenMP over tiles, a few seconds per rank); `lane_layout_decode()`
  round-trips exactly (`core_tests`: tails in rows and groups, clustered codes forcing escapes,
  several group tiles, drift limiter).

### 1.1 `lane-sN` kernel (`lane_projection<R, W>`, `src/projection.hip`)

* `R` = batch rows per pass (1, 2 or 4; `passes = ceil(rows/4)`, `grid.y`), block = `W` waves
  (8 for R <= 2, 16 for R = 4) = `W` consecutive row tiles of the same group tile and split;
  persistent blocks loop over (group tile, tile group, split) items in stride (`lane_blocks()`
  picks a divisor of the item count so every block does the same number of items). The
  activation tile (`xs[group][R]`, 8R bytes per group, slot `groups` zero) is loaded into LDS once
  per group tile; for R=4 the two 16-byte halves of a group live in separate regions so the two
  `ds_read_b128` of an entry spread over all eight 4-bank LDS groups (0.39 -> 0.22 ms for gate at
  rows=4: the R=4 kernel is LDS-conflict-bound, section 4).
* Per segment a lane holds 80 bytes in flight (the next segment prefetches while the current one is
  processed, including across items), per chunk: idx word and nibble extraction, code chain
  (`+ delta & 0xffff`), 8-byte codebook gather, R x 8 bytes of activations from LDS, 2R `dot4`
  into R fp32 accumulators. At the end the two lanes of a row add their halves (`__shfl_xor`) and
  write `partial[(gt*splits + split)][rows][out]`; `merge_projection` applies the scales (and now
  writes bf16 directly when asked, see 1.3).
* `lane-s1/s2/s4/s8` differ in the number of segment splits per tile (parallelism for small
  projections: k/v have 32 tiles). The autotuner compares them (plus `sorted-lds` from 5 rows and
  BLAS where available) per weight for rows <= 16, exactly like the other modes.
* fp32 accumulation (the same precision class as the WMMA paths, not the exact int64 of
  `sorted-sN`); cross-path checks pass for all seven layer-0 projections at every row count
  benchmarked (1-8, 16, 32, 64, 256), `behemoth_gpu_tests` covers the lane layout on
  in = 128..4104, out = 32..256 (37/65/129/513-group tails), rows 1, 2, 7, 16, 33, 64, 65, 130, scaled
  and unscaled, with output guards.

### 1.2 `sorted-lds` / `sorted-dense` on the lane layout

`lane_decode_tile` walks the segments of a tile (two per wave in flight, lane-private loads) and
writes `ct[g*16 + lane/2] = code` for the block's K window; `sorted_lds_gemm` and
`sorted_lds_reconstruct` are templated on the encoding (`SortedArgs`). Rows 16-64 and prefill are
3-6 % faster than with the sorted encoding (fewer, wider loads in the decode phase; section 3).

### 1.3 Fused elementwise chain (`f66ba14`: `src/ops.hip`, `src/model.cpp`, `ProjectionRunner::Call`)

| before (per layer) | after |
| --- | --- |
| `rms_norm` -> 3 x (`quantize_rows` + kernel(s) + `merge` + `cast_scaled`) for q/k/v | `rms_norm_quantized` (fp8 + scales straight into the projection workspace) -> 3 x (kernel(s) + `merge<bf16>`) |
| `rope(q)`, `rope(k)`, `quantize_rows(q)` | `rope_quantize` (one launch, grid rows x (48 + 4) heads) |
| o: `cast_scaled` + `residual_add` | `residual_add_scaled` |
| `rms_norm` -> 2 x (`quantize_rows` + ... + `cast_scaled`) for gate/up | `rms_norm_quantized` -> 2 x (kernel(s) + `merge<bf16>`) |
| down: `cast_scaled` + `residual_add` | `residual_add_scaled` |

15 launches per layer removed (1320 per token, ~3 ms of dispatch at 2.3 us) plus the bf16 and fp32
intermediate passes; rounding is bit-identical (`norm_quantize_kernel` reproduces
`norm_kernel` + `quantize_kernel`, the bf16 merge reproduces `cast_kernel`, `rope_quantize_kernel`
reproduces `rope_kernel` + `quantize_kernel`). The lm_head also takes its input from
`rms_norm_quantized`. `ProjectionRunner::Call { input, output, bf16_output, apply_row_scale }` with
`input == nullptr` means "the workspace already holds the quantised rows" (`prepare_input()`
sizes and pads it); modes that do not end in a merge (fused, BLAS, single-window `sorted-lds`)
write fp32 and are cast afterwards inside `run()`, so every mode stays correct.

Measured on the sorted layout (same kernels as the baseline otherwise): c=1 85.7 -> 77.6 ms per
step, c=4 222.1 -> 211.8 ms. The profiled c=1 step still shows ~7.6 ms of "elementwise" GPU
events; most of it is the event overhead of five profiled regions per layer plus the launch floor
of the remaining ~27 kernels per layer.

## 2. Why the gathers stay local: CPU simulation (`lane_sim.cpp`, layer 0, real codes)

Per-row order statistics drift: at step `j` row `r`'s code is the `2j`-th smallest of its 1536
codes, mean `j * 85`, standard deviation up to 836 codes (6.7 KiB). An LRU model of a 32 KiB /
128-byte-line L0 shared by 8 lock-step waves (8 row tiles) touching 32 codebook entries per step:

| groups per tile | drift limit | dummies | L0 hit rate | L2 lines per entry | code span per step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1536 (full K, used) | off | 0 % | **0.975** | 0.025 | 27 KiB |
| 1536 | 3000 codes | 1.3 % | 0.978 | 0.022 | 24.5 KiB |
| 768 | off | 0 % | 0.905 | 0.095 | 38.5 KiB |
| 512 | off | 0 % | 0.822 | 0.178 | 47 KiB |
| 1536, L0 = 16 KiB | off | 0 % | 0.788 | 0.212 | 27 KiB |
| fully sorted stream (previous layout), for reference | | | | 0.192 | |

q/k/down behave the same (0.976-0.980), o_proj (768 groups) 0.909. Full-K tiles were therefore
kept (`max_groups_per_tile = 2047`), which also caps `R` at 4 (R x 8 x 1792 bytes of activations
must fit the 64 KiB LDS); the drift limiter is implemented but off. The GPU diagnostics in section 4
confirm it: the gather costs 0.02 ms of the 0.144 ms gate kernel at rows=1.

## 3. Projection benchmark (`behemoth_projection_bench MODEL rows 0 1 all`, layer 0, rank-0 shard, device 1)

Per layer per rank, medians of seven interleaved rounds, timings include quantisation and merge.
"lane-only" = best of the modes a `--code-layout lane` server can run (no 2-byte codes resident),
"sorted-only" likewise for `--code-layout sorted`; "group-major" = stock codes (`pipelined-*`).
All cross-path checks pass (`final_rows*.json`).

| rows | group-major best | sorted-only best | **lane-only best** | lane mode | per row (lane) |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 1.628 | 0.619 (sorted-s8) | **0.687** | lane-s4/s8 | 0.687 |
| 2 | 1.650 | 1.007 (sorted-s4) | **0.731** | lane-s8 | 0.366 |
| 3 | 1.660 | 1.379 (sorted-s8) | **0.936** | lane-s8 | 0.312 |
| 4 | 1.689 | 1.772 (sorted-s4) / 1.959 (sorted-lds) | **0.956** | lane-s8 | 0.239 |
| 5 | 1.703 | 1.960 (sorted-lds) | **1.656** | lane-s8, 2 passes | 0.331 |
| 6 | 1.706 | 1.974 | **1.668** | lane-s8 | 0.278 |
| 7 | 1.705 | 1.979 | **1.671** | lane-s8 | 0.239 |
| 8 | 1.711 | 1.993 | **1.677** | lane-s8 (sorted-lds 1.961) | 0.210 |
| 16 | 1.783 | 2.107 | 2.031 | sorted-lds | 0.127 |
| 32 | 2.023 | 2.569 | 2.420 | sorted-lds | 0.076 |
| 64 | 2.443 | 3.008 | 2.932 | sorted-lds | 0.046 |
| 256 | 9.309 | 9.839 | 9.410 | sorted-dense | 0.037 |

Per projection at rows=1 (lane / sorted): q 0.065 / 0.067, k 0.037 / 0.034, v 0.037 / 0.034,
o 0.077 / 0.067, gate 0.144 / 0.130, up 0.145 / 0.131, down 0.134 / 0.135. At rows=4 (lane):
q 0.107, o 0.110, gate 0.214, up 0.212, down 0.202.

The bench re-reads one 58 MB layout, which the 64 MiB MALL partly serves; the sorted kernel's
plain loads profit from that more than the lane kernel did with `SCOPE_SYS` loads (first builds),
which is one reason the bench gap at rows=1 (11 %) is larger than the end-to-end gap (3 %).

Auto selections (`"auto"` rows in the JSON) are within 1-2 % of the forced best for the lane
layout at every row count; the bench's `LAYOUT=lane|sorted-lane|all` runs the autotuner per
projection like the server.

## 4. Where the lane kernel's time goes (diagnostic variants, `BEHEMOTH_LANE_DIAG`, gate, bench)

| variant | rows=1 | rows=4 |
| --- | ---: | ---: |
| complete kernel (before the split-half LDS layout) | 0.156 ms | 0.391 ms |
| no codebook gather (`diag 2`) | 0.135 | 0.384 |
| no LDS activation read (`diag 4`) | 0.154 | **0.190** |
| neither (`diag 6`: stream + VALU + launch/merge) | 0.134 | 0.155 |
| gather from one cache line (`diag 8`) | 0.135 | 0.388 |
| complete, 16-byte LDS reads | | 0.270 |
| complete, 16-byte reads + split-half layout (shipped) | 0.144 (policy 0) | 0.217 |

* rows=1: the gather is worth 0.02 ms and the LDS read nothing; 0.134 ms of 0.144 is the stream
  (58 MB at ~430 GB/s with the MALL) plus VALU plus fixed costs. Four earlier variants were tried
  on the way: 2-byte per-lane idx loads with 4 chunks in flight (0.160 ms), whole-segment loads
  of the bit-plane encoding (0.189 ms: 194-256 VGPRs because the compiler kept the whole unrolled
  segment live; fixed by uniform tile descriptors and a batch switch), scalar plane loads
  prefetched (0.157), then the lane-private nibbles (0.144, 87 VGPRs for R=1, 143 for R=4).
* rows=4 is LDS-bound: 32 random bytes of activations per entry; `ds_read_b64` x4 with the
  interleaved layout cost 0.20 ms of 0.39, two `ds_read_b128` 0.08, the split-half layout 0.06.
  The remaining per-entry cost (0.217 vs 0.155 without LDS) is the inherent bank-conflict rate of
  32 random 16-byte reads over eight 4-bank groups (expected max load ~8.5 of 4 ideal).
* Stream cache policy (bench, rows=1 layer sums): regular 0.619-0.626, `SCOPE_DEV` 0.616,
  `SCOPE_SYS` 0.702, non-temporal 0.837-0.841, `SYS+NT` 1.48. End to end (section 5): regular 79.9
  / 124.6 ms, DEV 79.3 / 126.1, SYS 87.5 / 132.3.

## 5. End to end (`perf/server_cycle.sh`, `perf/e2e_round2.sh`; 2 GPUs, zero-copy transport, `--token-budget 1024 --prefill-chunk 256`)

39-token prompt, per concurrency two warm-up runs (first-use autotuning for the batch size) then
`2c` requests of 96 tokens; "GPU step" = `gpu_step_seconds` delta / (2 x 96 steps) from `/metrics`
(prefill of 39 tokens is one chunk, so it counts as one step per wave of requests).

| build / layout / stream policy | c=1 GPU step (tok/s) | c=4 GPU step (aggregate tok/s) | TTFT p50 c=1 / c=4 |
| --- | ---: | ---: | ---: |
| baseline `a12a658`, sorted | 85.7 ms (11.7) | 222.1 ms (18.0) | 363 / 927 ms |
| this round, sorted (fusion only) | 77.6 (12.9) | 211.8 (18.9) | 329 / 940 |
| this round, lane, first kernel (2-byte loads), SYS | 93.2 | 179.6 (22.3) | 313 / 893 |
| this round, lane, first kernel, regular loads | 81.9 | 303.0 (autotuner picked badly: one run) | 331 / 973 |
| this round, lane, final kernel, SYS (24) | 87.5 (11.4) | 132.3 (30.2) | 316 / 876 |
| **this round, lane, final kernel, regular loads (default)** | **79.9 (12.5)** | **124.6 (32.1)** | 321 / 672 |
| this round, lane, final kernel, DEV scope (16) | 79.3 (12.6) | 126.1 (31.7) | 332 / 952 |
| **+ decode attention partition 32 keys (final, default)** | **76.4 (13.1)** | **121.6 (32.9)** | 318 / 670 |
| + attention partition 128 keys | 79.9 | 127.2 | 318 / 911 |
| this round, lane, `--profile true` (events add overhead) | 91.4 | 136.3 | 335 / 894 |

Profiled c=1 step (lane, regular loads; rank 0 GPU event totals, inflated by ~5 ms of events):
projections 54.1 ms, attention + KV 12.0 ms, elementwise 7.6 ms, collectives 4.0 ms, lm_head 0.3 ms.
Projections match the bench (0.687 ms x 88 = 60 ms with the merge and partial traffic the bench
includes). Attention at 12 ms for a ~90-token context is the next largest item (section 7).

c=8 was not measured: the baseline c=8 warm-up (8 requests x 24 tokens through the stock `a12a658`
server, 00:22) took both GPUs off the bus (`amdgpu 0000:cc:00.0: device lost from bus!`, vendor ID
`ffff` on both root ports), the same signature as the earlier incidents, at the OD agent's chosen
210 W point which had only been validated at c=4. Recovery: BMC `ForceOff`, 20 s, `On`,
`bash ~/kernel.sh` (00:27, setting re-applied and verified). No further c=8 load was run.

## 6. Checks

* `ctest --test-dir build-rocm`: core (lane round trip added), gpu_correctness, gpu_transport pass;
  `behemoth_gpu_tests` (device 1) and `--two-gpu` pass on every build that was benchmarked;
  the bench cross-path checks pass for all seven layer-0 projections at rows 1-8, 16, 32, 64, 256
  on all four layouts (`final_rows*.json`, `cross_path_failures=0`).
* One bug caught by re-running the GPU tests before an e2e cycle: the R=1 activation slot of the
  split-half LDS layout (server produced garbage, load_bench failed); fixed before the final build.

## 7. What is left

1. **rows=1 stream bytes.** The lane kernel at rows=1 is stream/VALU-bound at ~430 GB/s equivalent;
   the only way to beat `sorted-s8` there by more than noise is fewer bytes per entry: 16-chunk
   segments (halves the segment padding, 1.6 % of bytes), 8-bit deltas with escape dummies (2.5 ->
   2.38 B, ~0.3 % escapes at this density), or dropping the group field to 10 bits where
   `in/8 <= 1024`... (not for this model). Expected: <= 5 % at c=1.
2. **rows 5-8 at one pass.** R=8 needs 96-112 KiB of activations per block; with 768/896-group
   tiles it fits (56 KiB) at the cost of a wider gather window (sim: 0.905 L0 hit rate) and 2x LDS
   traffic per entry. Worth trying for a c=8 server once the hardware can run c=8.
3. **Attention at c=1 (12 ms/token for 90 tokens before the partition change):** the kernel is
   still one wave per block with a serial 32-key loop; partition 32 recovers 3.5 ms by using more
   blocks, a multi-wave block (one wave per key tile, LDS reduction) or fusing `append_kv` into it
   would recover more. Partition 32 falls back to 1024 when the split-K workspace (256 MB) would
   overflow at very long contexts.
4. The remaining ~27 launches per layer (2.3 us each) are ~5.5 ms/token; graphs do not help
   (measured again on ROCm 10.1). Only kernel fusion across the attention/KV chain is left.

## 8. Files

* `include/behemoth/sorted_layout.hpp`, `src/sorted_layout.cpp`: `LaneLayout`, `lane_layout()`,
  `lane_layout_decode()`.
* `src/projection.hip`: `LaneSeg`, `lane_decode_tile/window`, `lane_projection<R, W>`,
  `SortedArgs`, `sorted_lds_gemm<..., LANE>`, `sorted_lds_reconstruct<DB, LANE>`,
  `merge_projection<T>`, `ProjectionRunner::Call`, `prepare_input`, lane candidates,
  `BEHEMOTH_LANE_DIAG` / `BEHEMOTH_LANE_POLICY`.
* `src/ops.hip`: `rms_norm_quantized`, `residual_add_scaled`, `rope_quantize`; `src/model.cpp`:
  fused forward pass, `BEHEMOTH_ATTN_PARTITION`.
* `include/behemoth/gpu.hpp`: `GpuProjection::upload_lane`, `has_lane`, lane modes,
  `projection_mode_layout`; `include/behemoth/checkpoint.hpp`: `CodeLayout::Lane`.
* `tests/core_tests.cpp`, `tests/gpu_tests.cpp`, `tools/projection_bench.cpp`.
* `perf/`: `e2e_round2.sh`, `server_cycle.sh`, `gpu_check.sh`, `gpu_check2.sh`, `bench_sweep.sh`,
  `summarize_round2.py`, `lane_sim.cpp`, `results/round2/`.

## 9. Server state at hand-over

```
cd /home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
setsid nohup build-rocm/behemoth_server --model /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output \
    --devices 0,1 --transport zerocopy --code-layout lane --memory-fraction 0.9 \
    --token-budget 1024 --prefill-chunk 256 --host 0.0.0.0 --port 1337 --threads 4 --profile false \
    > /tmp/beh_server.log 2>&1 < /dev/null & echo $! > /tmp/beh_server.pid
```
