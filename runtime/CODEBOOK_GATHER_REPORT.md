# Breaking the AQLM 1x16 codebook-gather wall with a codebook-major ("sorted") code layout

Investigation date: 2026-10-25, 2x Radeon AI PRO R9700 (gfx1201), ROCm 7.2.4.
Builds on `DECODE_PERF_REPORT.md` (gather-rate wall: ~68 G random 8-byte gathers/s
chip-wide, 1x16 projections at ~120-135 GB/s of codes, 126 ms/token/rank).
Commits in this tree: `c6737c9` (layout encoder + microbenchmarks), `86d099a`
(runtime integration: `--code-layout sorted`, modes `sorted-s4/s8/s16/sorted-dense`),
plus the follow-up commit with the coalesced `sorted-dense` scratch. Raw numbers:
`perf/results/sorted/`, `perf/results/e2e/sorted_*`.

## TL;DR

* **The wall is the visit order, not the data.** Re-ordering each 32-output-row slab of a
  projection by codebook index (lossless, done once at load time) turns the random
  8-byte codebook gather into a near-streaming read: 32 consecutive lanes touch ~2
  cache lines instead of 32. Gather-only microbenchmark: **278 G gathers/s on the sorted
  stream vs 64 G/s random (4.4x)**; the stream is then DRAM-bound (628 GB/s).
* **Accumulating the scattered outputs is the new problem, and fp32 LDS atomics are
  unusable on gfx1201: `ds_add_f32` runs at 0.26 lane-ops/clk/CU (serialised), while
  `ds_add_u32` runs at 5.6 and `ds_add_u64` at 3.0.** The kernel therefore accumulates
  *exact* fixed-point int64 (products of two e4m3 values are multiples of 2^-18 and
  the sums stay below 2^50): the result is bit-exact versus a double-precision
  reference (max error 0 in the microbenchmark), i.e. more accurate than the fp32 WMMA
  path it replaces.
* **Measured on the real layer-0 weights (`behemoth_projection_bench`, rows=1, layer 0,
  all cross-path checks pass):** gate/up 0.116 ms (was 0.348-0.355, **3.0x**), down 0.110
  (was 0.350, 3.2x), q 0.063 (was 0.159, 2.5x), o 0.057 (was 0.158, 2.8x), k/v unchanged
  (0.038; tiny layouts). **Per layer 0.539 ms vs 1.436 ms (2.66x); per token per rank
  47 ms of projections instead of 126 ms.** The bench re-reads one 52 MB layout, which
  the 64 MiB MALL partly serves; the MALL-free microbenchmark (4 rotating copies) gives
  gate 0.130-0.147 ms, q 0.065, down 0.124-0.157 — still 2.3-2.7x.
* **End-to-end (`behemoth_server --code-layout sorted`, single request): inter-token p50
  82.5 ms (12.1 tok/s) with the host transport and 73.6 ms (13.6 tok/s) with the zero-copy
  transport prototype, versus 163 ms (6.1 tok/s) for the stock group-major layout: 2.0-2.2x.** Decode at concurrency 8 is *slower* with the sorted-only
  layout (337 ms/step vs 198) because the sorted kernel streams the layout once per
  batch row; see section 7 for the batched story and the `sorted-dense` fallback.
* **Cost: the sorted layout is 2.38-2.5 B per 8-weight group instead of 2.0 B (+19-30 %,
  depending on code density; the floor is ~2.18 B).** It replaces the group-major codes
  in GPU memory (both do not fit in 32 GiB), so a server runs *either* layout:
  `--code-layout sorted` is a low-concurrency / interactive configuration today.
* Not pursued further: codebook relabeling / output permutation (cannot help with uniform
  code usage, section 8.1), the LDS-slice bucketed variant (dominated by the fully sorted
  stream, section 8.2), fp32 lane-private accumulators (measured, slower: section 3.3).

## 1. The layout

For every projection the output rows are cut into tiles of 32 rows (one Wave32 per tile).
Inside a tile all `32 x groups` entries `(row, group, code)` are sorted by `code`
(counting sort, stable) and stored as 32-entry chunks:

* `idx[chunk*32 + lane] = row_local << 11 | group_local` (uint16; 2048 > 1792 groups covers
  every projection of the model, `group_local == 2047` is a dummy slot).
* `planes[chunk*3 + k]` = bit `k` of the 32 code deltas (bit-plane layout). A lane's code is
  the running code at the start of its segment plus the *inclusive* prefix of the deltas:
  three `v_bcnt` on `plane & lanes_le_me`, no cross-lane shuffles. The wave-uniform total
  (`s_bcnt1`) advances the running code.
* Dummy entries (`group_local = 2047`, arbitrary delta) bridge gaps larger than 7 and pad the
  last chunk of a tile. The kernel keeps a zero activation in LDS slot 2047, so dummies
  contribute exactly 0 without any masking or branch.
* Chunks are grouped into segments of 16 (512 entries); `segment_codes` stores the code of
  each segment's first entry so blocks/waves can start anywhere. Tile descriptors:
  `{chunk_begin, chunk_count, segment_begin, segment_count}`.

Bytes per entry: 2 (idx) + 0.375 (planes) + 0.004 (segment codes) = **2.38 B**, times
(1 + dummy fraction). Information-theoretically, sorting a stream of `N` entries over 65536
codes saves `log2 N!` bits of ordering: at density 0.75 entries/code (32 rows x 1536 groups)
the floor is ~17.4 bits = 2.18 B/entry, so the fixed-width scheme is within 10 % of optimal.
Alternatives considered: 2-bit deltas (dummies 8-17 %, no faster), 4-bit (2.5 B, 0 % dummies,
no faster), 256x256 / 128x512 tiles with 16-bit indices and per-tile codebook passes
(amplifies L2 codebook traffic to 8 B/entry; the 32-row x 2048-group stream reads the 512
KiB codebook once per 49152 entries = 10.7 B/entry from L2, which is not the limiter).

Measured dummy fractions on layer 0 (code usage is least uniform there): gate/up/down 0.42 %,
q 4.2 %, k 3.4 %, v 0.7 %, o 9.1 % (o has 768 groups per rank, density 0.375). Layout bytes
versus 2-byte codes: gate 52.6 MB vs 44.0 (+19.5 %), q 23.4 vs 18.9 (+24 %), o 24.5 vs 18.9
(+30 %), k 1.94 vs 1.57 (+23 %). Per rank: ~18.5 GB instead of 15.2 GB; the server reported
5957 KV pages (95k tokens) with the sorted layout at `--memory-fraction 0.9`.

Encoder: `src/sorted_layout.cpp` (`sorted_layout()`, OpenMP over tiles, exact round-trip
check `sorted_layout_decode()`); load-time cost is a few seconds per rank.

## 2. Microbenchmarks (`perf/micro_sorted.hip`, `perf/results/sorted/`)

### 2.1 LDS throughput (`micro_sorted lds`, 2048 blocks x 256 threads, random addresses)

| operation | Gop/s | lane-ops/clk/CU |
| --- | ---: | ---: |
| `ds_add_f32` random, 32..4096 slots | 39-40 | **0.26** (serialised; conflicts irrelevant) |
| `ds_add_u32` random, 256 slots | 847 | 5.63 |
| `ds_add_u64` random, 32 slots | 451 | 3.00 |
| `ds_read_b64` random, 2048 entries (16 KiB) | 613 | 4.07 |
| lane-private fp32 RMW `[row][lane]` (read+add+write) | 725 | 4.82 |
| `ds_read_b64` + lane-private RMW pair | 435 | 2.89 |

The ISA was checked (`--save-temps`): the fp32 atomics are native `ds_add_f32`, not CAS
loops. This single number kills the obvious "sorted stream + fp32 LDS atomics" design.

### 2.2 Sorted vs random gather (`micro_sorted gather`, gate shape, 22 M entries)

| stream | G gathers/s | 2-byte-code-equivalent GB/s | gathers/clk/CU |
| --- | ---: | ---: | ---: |
| random (idx holds the code, current layout) | 58-64 | 117-128 | 0.40 |
| sorted, 2-bit planes (8.3 % dummies) | **275-277** | 550-553 | 1.84 (DRAM-bound: 2.44 B/entry x 277 G/s = 628 GB/s) |
| sorted, 3-bit planes (0.4 % dummies) | 207-210 | 413-420 | 1.38 |

### 2.3 Full GEMV prototypes (synthetic uniform codes, `micro_sorted gemv`, `perf/results/sorted/gemv_*.txt`)

Timing re-reads the same layout (MALL-assisted) unless noted. gate shape = 22 M entries.

| variant | gate ms | q ms | down ms | note |
| --- | ---: | ---: | ---: | --- |
| fp32 `ds_add_f32` atomics into `y[256]` (first prototype) | 0.70-0.80 | 0.31-0.34 | 0.70-0.80 | atomic-serialised, 2x slower than pipelined |
| fp32 lane-private slots `ys[row][lane]`, 4 KiB/wave | 0.173-0.182 | 0.079-0.085 | 0.179-0.184 | LDS-capacity limits occupancy |
| **int64 `ds_add_u64`, exact fixed point (shipped)** | **0.111-0.119** | **0.054** | **0.108-0.120** | max error 0 vs double reference |
| same, MALL-free (4 rotating layout copies) | 0.130-0.147 | 0.065 | 0.124-0.157 | 340-420 GB/s actual |
| diagnostic: atomics replaced by plain stores | 0.058 | 0.034 | 0.062 | 850-900 GB/s "actual" (MALL) |
| x read from global/L0 instead of LDS | 0.115-0.132 | 0.061 | 0.129-0.140 | no gain |
| 3-stage software pipeline (prefetch idx/planes, gathers, LDS) | 0.115-0.128 | 0.057-0.067 | 0.114-0.126 | no gain (MALL-free either) |
| W=16 waves/block, dynamic LDS, C=1/2/4 accumulator copies, U=2/4/8, splits 4-32 | +-10 % | | | noise-level |

What limits the shipped kernel: per 32 entries the wave issues one random `ds_read_b64`
(x[group], ~8 clk) and one `ds_add_u64` (~10.7 clk) -> ~1.7 entries/clk/CU of LDS capacity;
the kernel runs at 1.3 (MALL) / 1.0-1.1 (MALL-free) entries/clk/CU, i.e. 60-75 % of the LDS
bound and 55-70 % of DRAM. VALU is ~30 ops/entry (decode 9, addresses 5, dot 2, fixed-point
split 8, loop), i.e. 2.1 entries/clk/CU at most — close enough to matter. The diagnostic row
(no atomics) confirms that removing the LDS accumulation would reach DRAM speed.

### 2.4 Exact fixed-point accumulation

`v = dot8(e4m3 weights, e4m3 activations)` is a sum of 8 products of values `m * 2^k`
(m <= 15, k in [-9, 5]), hence an integer multiple of 2^-18 with |v| < 2^21. `n = v * 2^18`
is an integer-valued float < 2^39 and is split exactly into `a = floor(v * 4)` and
`b = n - a * 2^16 < 2^16` (both exact in fp32), packed into an int64 and added with
`ds_add_u64`. The first attempt (`floor(v * 2^-14)` / `fma`) lost bits for small negative
values (max error 0.03 in the synthetic test); the shipped split has error 0. The output
is `float(double(sum) * 2^-18)`, so the only rounding left is inside `v_dot4_f32_fp8_fp8`.

## 3. The shipped kernel (`src/projection.hip`, `sorted_projection<W=8, U=4, C=2>`)

* Block = 8 waves = 8 consecutive 32-row tiles; `grid = (ceil(tiles/8) * splits, rows)`. The
  block loads the fp8 activation row into LDS (16 KiB, slot 2047 = 0) and zeroes 2 copies of
  32 int64 accumulators per wave (lane parity selects the copy to halve same-address
  serialisation). Each wave walks its split's segments: per 4 chunks it loads 4 x (u16 idx,
  3 planes via `s_load`), computes codes (popcounts), gathers 4 x 8 B from the codebook
  (L0 hits), reads `x[group]` from LDS, two `v_dot4_f32_fp8_fp8`, fixed-point split, one
  `ds_add_u64`. At the end lane `r` sums the copies for row `r` and writes
  `partial[split][row][r]`; the existing `merge_projection` applies the input scale,
  codebook scale and row scale. Modes `sorted-s4 / sorted-s8 / sorted-s16` differ only in the
  number of segment splits per tile (parallelism vs partial-buffer traffic); the autotuner
  measures them like any other mode (`choice_report`, `perf/results/sorted/bench_rows*.json`).
* `sorted-dense`: `sorted_reconstruct` streams the layout once into a dense FP8 scratch
  (group-major so that the dense pipelined WMMA kernels load 32 consecutive columns per
  wave), then runs `pipelined-64 / pipelined-r2-64 / pipelined-r4-64` on it. BLAS also
  reconstructs from the sorted layout (row-major) when no codes are present.
* Tails: rows beyond `out` are simply absent from the last tile; groups beyond `in/8` never
  occur. `behemoth_gpu_tests` runs every sorted mode on in = 128..4104, out = 32..256
  (including 37/65/129/513-group tails), rows 1/2/7/16/33, scaled and unscaled, with
  output guards; the bench's cross-path check passes for all seven layer-0 projections at
  rows 1, 8 and 32.
* Memory: `--code-layout sorted` uploads only the sorted layout (`GpuProjection::upload`
  with `CodeLayout::Sorted`); `upload_sorted()` adds it next to the codes (tests / bench).

## 4. Projection benchmark (`behemoth_projection_bench MODEL rows 0 dev sorted`, layer 0, rank-0 shard)

Medians over 7 interleaved rounds, timings include activation quantisation, same harness as
`DECODE_PERF_REPORT.md`. "logical GB/s" = 2-byte code bytes / time.

### 4.1 rows = 1 (`perf/results/sorted/bench_rows1.json`)

| projection | codes MB | best stock mode | ms | best sorted mode | ms | speed-up | logical GB/s |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| q 12288->6144 | 18.9 | pipelined-64 | 0.160 | sorted-s8 | **0.063** | 2.5x | 299 |
| k 12288->512 | 1.57 | pipelined-64 | 0.038 | sorted-s16 | 0.039 | 1.0x | 41 |
| v 12288->512 | 1.57 | pipelined-128 | 0.038 | sorted-s16 | 0.038 | 1.0x | 41 |
| o 6144->12288 | 18.9 | pipelined-64 | 0.157 | sorted-s4 | **0.057** | 2.7x | 330 |
| gate 12288->14336 | 44.0 | pipelined-64 | 0.353 | sorted-s16 | **0.116** | 3.1x | 381 |
| up 12288->14336 | 44.0 | pipelined-128 | 0.355 | sorted-s16 | **0.117** | 3.0x | 377 |
| down 14336->12288 | 44.0 | pipelined-128 | 0.358 | sorted-s4 | **0.110** | 3.3x | 400 |
| **layer** | 173 | | **1.436** | auto (selects sorted) | **0.539** | **2.66x** | |

Auto selects a sorted mode for q/o/gate/up/down and pipelined-64 for k (tie). k/v are 16 tiles
= 2 blocks x splits; their layouts are 1.9 MB and the kernel is launch/latency-bound there.

### 4.2 rows = 8 and rows = 32 (`bench_rows8.json`, `bench_rows32.json`)

Stock best per layer: rows 8 = 1.47-1.50 ms (pipelined-64/128/s2048), rows 32 = 1.76 ms
(pipelined-r2-128). Sorted-layout best per layer: rows 8 = 2.76 ms (per-row passes,
0.083 ms per row-pass of gate because the 8 passes share the layout in L2/MALL), rows 32 =
10.6 ms. Per-projection numbers and the `sorted-dense` analysis are in section 7.

## 5. End-to-end (`perf/e2e.sh`, `behemoth_load_bench`, 2 GPUs, `perf/results/e2e/`)

| configuration | c | inter-token p50 / p95 (ms) | tok/s | TTFT p50 (ms) |
| --- | ---: | ---: | ---: | ---: |
| baseline 965c1cfa, group-major, host transport | 1 | 163.1 / 168.1 | 6.1 | 787 |
| baseline | 8 | 197.9 / 204.0 (per step) | 40.4 aggregate | 2440 |
| **sorted layout, host transport** | 1 | **82.5 / 87.1** | **12.1** | 1641 |
| sorted layout, host transport | 8 | 337.6 / 344.4 (per step) | 23.7 aggregate | 4049 |
| **sorted layout, zero-copy transport (prototype)** | 1 | **73.6 / 77.7** | **13.6** | 1642 |
| sorted layout, zero-copy transport | 8 | 334.5 / 340.0 (per step) | 23.9 aggregate | 3984 |

No GPU faults during these runs (concurrency kept <= 8).

(`load_bench tok/s` in the raw JSON includes TTFT and request setup; the inter-token numbers
are the server's own rolling percentiles.) Single-request decode budget per rank now:
projections ~47-55 ms (MALL-free estimate), collectives ~12-15 ms (pinned-host), attention
+ KV ~9 ms, elementwise + launches ~7.5 ms = ~80 ms, matching the measured 82.5 ms. With the
zero-copy transport prototype (-8.5 ms) and the elementwise fusions from the previous report
(-4 ms) the single-request floor with this layout is ~70 ms/token (~14 tok/s).

## 6. Projected throughput

* **Batch 1:** 82.5 ms/token measured (12.1 tok/s) vs 163 ms with the host transport;
  **73.6 ms/token (13.6 tok/s, 2.2x) with `--transport zerocopy`**. Remaining budget:
  ~50 ms projections, ~3-4 ms collectives, ~9 ms attention/KV, ~7.5 ms elementwise and
  launches, ~3 ms other; the previous report's fusion items (-4 to -6 ms) would give
  ~65 ms (15 tok/s).
* **Batch 8:** the per-row sorted passes cost 2.76 ms/layer (243 ms/step of projections):
  measured 337 ms/step = 23.7 tok/s aggregate, worse than the stock 198 ms/step (40 tok/s).
  `sorted-dense` does not rescue this (section 7). Projected with zero-copy collectives:
  ~330 ms/step (24 tok/s) sorted vs ~190 ms/step (42 tok/s) stock.
* **Batch 32:** stock `pipelined-r2-128` 1.76 ms/layer (155 ms/step of projections,
  ~140 tok/s projected aggregate with zero-copy collectives). Sorted layout: 10.6 ms/layer
  = 930 ms/step of projections (~33 tok/s aggregate). The sorted layout cannot use the r2/r4
  gather-sharing kernels (they need the codes), so it is the wrong layout for batch >= 4.

## 7. Batched decode with the sorted layout

The sorted kernel's cost is per (entry, row): every batch row needs its own LDS `x` read
and its own int64 atomic, so an M-row pass is LDS-bound at ~M x the single-row LDS cost;
measured M=2 gave no per-row gain (0.231 ms per 2 rows vs 0.111 per 1). Options measured:

* `sorted-sN` with rows > 1: `grid.y = rows` independent passes (each re-reads the layout;
  the passes share L2/MALL so 8 rows cost 0.66 ms for gate = 0.083 ms/row).
* `sorted-dense`: stream the layout once into a dense FP8 scratch (2.39 B read + 8 B
  written per entry) and run the dense pipelined WMMA kernels. The first version wrote the
  scratch row-major and the dense kernels loaded 8 B per lane with a 12 KiB stride
  (3.6-3.9 ms for gate at rows 8/32); the follow-up commit writes it group-major so
  column loads coalesce.

Measured per-layer projection time (layer 0, rank 0, best mode per projection):

| rows | stock best (group-major codes) | sorted layout best | sorted modes chosen |
| ---: | ---: | ---: | --- |
| 1 | 1.436 ms | 0.539 ms | sorted-s4/s8/s16 |
| 8 | 1.472-1.503 ms | 2.76 ms (q 0.317, k 0.052, v 0.051, o 0.330, gate 0.667, up 0.670, down 0.671) | sorted-s4/s8 per-row passes (sorted-dense 0.87-4.7, blas 1.0-3.0) |
| 32 | 1.758-1.765 ms | 10.6 ms (q 1.08, k 0.093, v 0.092, o 1.25, gate 2.69, up 2.74, down 2.68) | sorted-dense for q/k/v/o, sorted-s4 for gate/up/down (blas 2.6-3.0) |

`sorted-dense` did not deliver: the dense reconstruct writes one 8-byte group per lane at a
random (row, group) position, so every 64-byte sector is assembled from 8 writes that are
spread over the whole 32-row stream; L2 cannot hold the 384 KiB-per-tile outputs of ~100
concurrent tiles, and the partial-sector write-backs cost ~8x the dense bytes (gate: 4.7 ms
for stream + reconstruct + dense WMMA, versus 0.43 ms for the stock `pipelined-r2-128`).
Writing the scratch group-major (coalesced dense loads) fixed the GEMM half but not the
scatter half; q/k/v/o gain (q 1.47 -> 0.87 ms at rows 8) because their streams are shorter.

Recommendation: run `--code-layout sorted` for interactive / low-concurrency serving (1-2
concurrent requests: 2x lower inter-token latency), keep the group-major layout for
throughput serving (>= 8 concurrent). A hybrid that keeps both layouts resident is ruled
out by memory (15.2 + 18.5 GB per rank > 32 GiB); a hybrid at the *layer* level (sorted
for some layers) only interpolates. The real fix for batched decode on this layout is a
kernel that accumulates several rows per entry without LDS atomics (e.g. lane-owns-row
order-statistic streams with register accumulators, section 8.3), not pursued here.

## 8. Alternatives evaluated

### 8.1 Codebook relabeling / output-row permutation (dropped)

Cache-line sharing between the 32 lanes of a gather requires the 32 codes of one group across
32 adjacent output columns to fall into few 128-byte lines (16 codebook entries each). A
relabeling is one global permutation of the 65536 codes; the sets of co-occurring codes are
different for every (group, column-tile) pair (22 M entries / 32 = 690 k different sets per
projection), so no single permutation can compact more than a negligible fraction of them.
With uniform code usage (15.9-16 bits of entropy, `perf/results/code_histogram.txt`) the
expected number of distinct lines touched by a 32-lane gather is 32 x (1 - 31/8192) ~ 31.9
before and after any relabeling or any row permutation (a row permutation changes *which*
32 columns form a wave, i.e. draws another random set). Quantified and dropped.

### 8.2 Two-level bucketing / LDS codebook slices (dropped analytically)

Sorting entries into 2048-code buckets and staging each 16 KiB codebook slice in LDS turns
the gather into an LDS lookup — but the fully sorted stream already turns it into an L0 hit
(2 lines per 32 lanes), the LDS lookup is not faster (`ds_read_b64` random: 4 lane-ops/clk/CU
vs the code-stream limit of ~2 entries/clk/CU), it costs a staging barrier per bucket and,
unless entries are also sorted within the bucket, 11 bits of within-bucket code per entry
(3.4 B/entry). The codebook traffic from L2 is identical (one 512 KiB pass per 32-row
stream). Strictly dominated by the fully sorted stream.

### 8.3 Lane-owns-row ("order statistic") streams

Alternative layout: lane k owns row k of a 32-row tile and walks its own row's codes in sorted
order; chunk j holds the j-th smallest code of each row. No atomics, register accumulators,
and M batch rows cost only M LDS `x` reads per entry (no M atomics) — potentially 4x faster
than the stock kernels at rows = 8. The price is gather locality: the 32 codes of a chunk are
spread over ~6-7 KiB (order-statistic spread), so L0 hit rates depend on 8-16 waves per WGP
staying in lock-step (window ~27 KiB vs 32 KiB L0). Entropy-wise it costs the same ~18 bits
per entry (11-bit group + ~7-bit per-lane delta). Not prototyped; the most promising route to
a sorted-layout kernel that also wins at batch 8-32.

### 8.4 fp32 accumulation variants (measured, section 2.3)

fp32 LDS atomics: 0.26 lane-ops/clk/CU, kernel 2x slower than the stock pipelined kernels.
fp32 lane-private slots (`ys[row][lane]`, conflict-free RMW): correct, but 4 KiB of LDS per
wave caps occupancy at 8-12 waves/WGP; 0.17-0.18 ms for gate (2x the int64 kernel).

## 9. Risks and caveats

* **Memory**: +3.3 GB per rank (18.5 vs 15.2 GB); KV capacity shrinks accordingly
  (5957 pages / 95k tokens at `--memory-fraction 0.9`). Both layouts cannot be resident.
* **Batched throughput regression with the sorted-only layout** (section 7). Default remains
  `group-major`; `sorted` is opt-in.
* **MALL inflation of micro/bench numbers**: a 52 MB layout fits the 64 MiB infinity cache
  across repeated timing iterations. MALL-free microbenchmarks are 10-25 % slower than the
  bench medians; the end-to-end result (82.5 ms) is the ground truth and is consistent with
  the MALL-free numbers.
* **Numerics**: exact integer accumulation; outputs match the fused WMMA path within 1e-7
  absolute on layer 0 (cross-path checks) and the double reference exactly in synthetic tests.
* **Layer-0 code skew**: dummies 4-9 % for q/k/o on layer 0 (code usage entropy 14-15.6 bits
  there); other layers are near-uniform (expected < 1 %). Per-projection delta-bit selection
  (4 bits when groups <= 1024) would remove most of o_proj's 9 %; not done (0.5 ms/token).
* **gfx1201-specific**: the design hinges on measured LDS atomic rates (fp32 serialised,
  int64 fast) and the 32 KiB L0. Other GPUs need the LDS microbenchmark re-run first.
* **Prefill / TTFT**: with the sorted-only layout prefill chunks run through BLAS or
  `sorted-dense` (both reconstruct a dense weight from the stream, 2.6-5 ms per large
  projection); TTFT was 1641 ms vs 787 ms for the 39-token prompt. A prefill path that
  does not scatter 8-byte writes (e.g. per-tile LDS transposes over a group window) is
  needed if prompts are long.
* **Load time**: layout encoding adds a few seconds per rank (OpenMP over 32-row tiles).

## 10. Files

* `include/behemoth/sorted_layout.hpp`, `src/sorted_layout.cpp` — layout, encoder, decoder.
* `perf/micro_sorted.hip` — LDS atomics/RMW, sorted vs random gather, GEMV prototypes
  (fp32 atomics, fp32 lane-private, int64 exact, diagnostics, software pipeline, MALL-free
  via `NCOPY=4`). Results: `perf/results/sorted/{lds,gather,gemv_*}.txt`.
* `src/projection.hip` — `sorted_projection`, `sorted_reconstruct`, modes
  `sorted-s4/s8/s16/sorted-dense`, autotuner candidates, dense group-major scratch.
* `include/behemoth/gpu.hpp` — `GpuProjection` sorted buffers, `upload_sorted`,
  `CodeLayout::Sorted`; `src/model.cpp` / `src/server.cpp` — `--code-layout sorted`.
* `tools/projection_bench.cpp` (`LAYOUT=sorted|all`), `tests/gpu_tests.cpp` (sorted modes on
  all shapes), `perf/e2e.sh` (`CONFIGS=... CONCURRENCIES="1 8"`).
* Results: `perf/results/sorted/bench_rows{1,8,32}.json`, `perf/results/e2e/sorted_*`.
