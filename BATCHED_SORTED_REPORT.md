# Batched decode on the codebook-major ("sorted") AQLM layout: one layout for batch 1 and batch 8-64

Investigation date: 2026-10-25, 2x Radeon AI PRO R9700 (gfx1201, Wave32, 32 WGPs, 64 KiB LDS),
ROCm 7.2.4, TP=2. Builds on `CODEBOOK_GATHER_REPORT.md` (sorted layout: rows=1 2.66x faster,
but rows=8 1.9x slower and rows=32 6x slower than the group-major layout) and
`DECODE_PERF_REPORT.md` (the 1x16 gather wall: ~68 G random 8-byte codebook gathers/s chip-wide).
Commits: `6f12141` (layout: 16-row tiles, per-chunk base codes, microbenchmarks), `ed914a7`
(`sorted-lds` batched mode, LDS-based dense reconstruct), plus the results/report commit.
Raw numbers: `perf/results/batched/` (`bench_rows{1,8,32,64,256}.json`, `v1/` = before K windows,
`e2e_*`), microbenchmark source `perf/micro_batched.hip`.

## TL;DR

* **One layout now serves both regimes.** `--code-layout sorted` keeps the rows=1 stream kernel
  (**0.54 ms/layer/rank, 2.7x faster than group-major**) and adds `sorted-lds`, a batched kernel
  that decodes each 16-row tile's sorted stream into an LDS code tile and then runs the same
  random-gather + WMMA scheme as the group-major kernels, reading codes from LDS. Measured per
  layer (layer 0, rank 0, best mode per projection, `behemoth_projection_bench ... all`):

  | rows | group-major best (ms) | sorted layout before (ms) | **sorted layout now (ms)** | target | ratio now/stock |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 1.477 | 0.539 | **0.542** (auto 0.537) | <= 0.6 | 0.37 |
  | 8 | 1.519 | 2.76 | **1.748** | <= 1.5 | 1.15 |
  | 32 | 1.807 | 10.6 | **2.149** | <= 2.0 | 1.19 |
  | 64 | 2.192 | (n/a) | **2.639** | <= 2.5 | 1.20 |
  | 256 (prefill) | 8.360 (pipelined-r4-64) / 12.9 (stock auto: blas) | 15-30 (sorted-dense/blas) | **8.959** (sorted-dense) | <= stock | 1.07 (0.70 vs stock auto) |

  Batched decode on the sorted layout went from 1.9x / 6x slower than group-major to 15-20 %
  slower; prefill-sized batches are within 7 % of the best group-major kernel and faster than
  the stock auto choice. The rows=8/32/64 targets are missed by 15-20 % (see section 7 for why
  and what is left).
* **The batched kernel is gather-bound at the same chip-wide wall as the group-major kernels**
  (22 M gathers for gate in ~0.37 ms either way); the overhead over group-major is the decode
  phase (stream read 0.07 ms per gate when not overlapped) plus a few per cent of launch and
  merge cost. The random gather dominates, so the sorted layout cannot be *faster* than
  group-major for batches >= 8 with this design; it is now roughly at parity.
* **Two findings that cost most of the investigation:**
  1. **A stream that allocates in the GPU caches halves the random-gather rate of the other
     blocks.** With ordinary loads the fused decode+gather kernel took 0.75-0.80 ms for gate
     (decode alone 0.07, gathers alone 0.38); reading the stream with the gfx12 `SCOPE_SYS`
     cache policy (buffer loads, cachepolicy 24) brought it to 0.41-0.44 ms. `nontemporal`,
     `SCOPE_SE`/`SCOPE_DEV` and the `LU`/`NT` temporal hints did not help; streaming only half
     the bytes did not help either (the effect is not proportional to volume).
  2. **rocWMMA's fp8 `load_matrix_sync` is byte-granular**: an A/B fragment loaded from LDS cost
     more than the gather it feeds (the LDS variant was 2x slower than loading from global).
     The shipped kernel builds the `v_wmma_f32_16x16x16_fp8_fp8` operands directly: lane `l`
     holds `A[row l%16][k 8*(l/16)..+8]` (one 8-byte activation load) and
     `B[k 8*(l/16)..+8][col l%16]`, which is exactly one gathered 8-byte codebook entry. No
     LDS staging of operands at all; the LDS holds only the 2-byte codes.
* **Layout changes (lossless, load time):** tiles of 16 output rows (a 16 x 1792-group code tile
  is 56 KiB, so it fits in LDS), every 32-entry chunk carries its own base code (no serial
  segment scan; eight chunks decode per wave step from one 16-byte load per lane), 4-bit deltas
  (0.3-2 % dummies at the lower 16-row density; o_proj 7 %). Cost: 2.57-2.75 B per 8-weight
  group, **19.8 GB per rank vs 18.5 GB (32-row/3-bit) and 15.2 GB (group-major)**; the KV budget
  shrinks accordingly (section 8).
* **End-to-end was not measured: both GPUs fell off the PCIe bus at the first request of the
  `batched_zerocopy` e2e run** (section 6; same failure class as the c=16 incident in the
  previous report, now at c=1). The machine needs a power cycle; the server could not be left
  running. Projected from the bench: c=1 unchanged at ~74 ms/token, c=8 ~210-220 ms/step (was
  335), TTFT ~1 s (was 1.6 s); to be confirmed, host transport first.

## 1. Design

### 1.1 Layout (`include/behemoth/sorted_layout.hpp`, `src/sorted_layout.cpp`)

Unchanged principle: inside a tile all `(row, group, code)` entries are sorted by code and stored
in 32-entry chunks (`idx` = `row_local << 11 | group_local`, bit-plane deltas). Changes:

* `sorted_rows_per_tile` 32 -> **16**. The batched kernel needs one tile's codes in LDS:
  16 rows x 1792 groups x 2 B = 56 KiB (down_proj), 48 KiB for the 1536-group projections. Code
  density per tile halves (16 x 1536 / 65536 = 0.375 entries per code), which the earlier report
  measured as still near-streaming for the rows=1 kernel (o_proj at density 0.375 ran at 330
  logical GB/s). Measured: rows=1 per layer 0.542 ms vs 0.539 ms with 32-row tiles.
* **Per-chunk base codes** (`chunk_codes[chunk]`, uint16, +2 B per 32 entries): the code of an
  entry is `chunk_code + inclusive prefix of the chunk's deltas`, so any chunk decodes without
  the segment running code. The batched decode can therefore process eight chunks per wave step
  (four lanes per chunk, eight entries per lane) without a serial scan, and the rows=1 kernel
  replaces one popcount-total per chunk by a uniform load. The segment table is still produced
  by the encoder for `sorted_layout_decode()` but no kernel uses it and it is not uploaded.
* Default delta width 3 -> **4 bits** (`sorted_default_delta_bits`): at density 0.375 the 3-bit
  variant has 6.6 % dummies (gate) for 2.60 B/entry; 4-bit has 0.4 % dummies for 2.57 B/entry
  and ~3 % fewer entries to process. The kernels are templated on `DB` (3 or 4).

### 1.2 `sorted-lds` (`sorted_lds_gemm<RT, U, DB>`, `src/projection.hip`)

One block of 512 threads (16 waves) per (16-row tile, K window, 64-row batch group):

1. **Decode phase** (`sorted_decode_tile`): the 16 waves walk the tile's chunks in octets; lane
   `l` loads 8 entries of chunk `o*8 + l/4` with one 16-byte buffer load, the chunk's 4 planes
   and base code, and writes `ct[group * 16 + row] = code` into LDS (2-byte writes; groups
   outside the block's K window are skipped). Four octets are in flight per wave. The loads use
   `__builtin_amdgcn_raw_buffer_load_*` with cachepolicy 24 (`SCOPE_SYS`) so the stream does
   not displace the codebook (finding 1 above). Cost when run alone: 0.05-0.07 ms for a gate
   projection (57 MB; 850-1100 GB/s apparent with the MALL, ~600 GB/s MALL-free).
2. **GEMM phase**: lanes are (column `l%16`, k-half `l/16`). A unit is 2U groups; lane half `h`
   owns groups `g0 + h*U + i`, so its U activation slices are contiguous (one 16-byte load per
   two groups per row tile). Per group the lane reads the code from LDS (conflict-free: 16
   consecutive uint16 per half-wave), gathers the 8-byte codebook entry (the B operand) and
   issues `RT` WMMAs (RT = 1/2/4 row tiles of 16 batch rows; U = 8 for RT=1, 4 otherwise, so a
   wave keeps 8 or 4 gathers plus the next unit's loads in flight; 82/76/127 VGPRs, no spills).
   Waves split the K units round-robin and reduce their accumulators through the (now free)
   code-tile LDS at the end.
3. **K windows**: windows of <= 1024 groups (`sorted_lds_geometry`): 1536-group projections use
   two windows of 768 (24 KiB code tile), down_proj two of 896 (28 KiB). Two blocks then fit
   per WGP (the reduction scratch needs 16 waves x RT KiB, so RT=4 still ends at 64 KiB and one
   block), and one block's stream decode overlaps the other's gathers. Each window scans the
   whole tile stream once (2x stream traffic). Measured gain 6-7 % at rows <= 32 (gate M=16:
   0.445 -> 0.415 ms MALL-free); 3-4 windows cost more than they save. Window partial sums go
   to `partial[window][rows][out]` and the existing `merge_projection` applies the scales; a
   single window writes the scaled output directly.
4. Rows > 64: `gridDim.y` row groups of 64, each re-decoding the tile (decode is cheap relative
   to the 4x gather); candidate up to 512 rows, the autotuner compares it with `sorted-dense`.

Tails: output rows beyond `out` are absent from the stream, groups beyond `in/8` get zero B
operands; the quantised activation workspace is padded to whole 64-row groups plus one K unit
and zeroed on allocation so the WMMA tiles may read past the real rows/columns (finite padding
times zero weights). `behemoth_gpu_tests` covers every sorted mode on in = 128..4104, out =
32..256 with 37/65/129/513-group tails, rows 1, 2, 7, 16, 33, 64, 65 and 130; the bench's
cross-path checks pass for all seven layer-0 projections at rows 1/8/32/64/256.

### 1.3 `sorted-dense` and BLAS: `sorted_lds_reconstruct<DB>`

Same decode phase, then each block writes its 16 dense rows coalesced (row-major for hipBLASLt,
group-major for the dense pipelined WMMA kernels): 0.23-0.5 ms for a 176 MB gate weight versus
2.6-5 ms for the old scattered 8-byte reconstruct. `sorted-dense` is the prefill winner
(section 4: rows=256 8.96 ms/layer vs 8.36 for the best group-major kernel and 12.9 for the stock
auto choice, hipBLASLt after `reconstruct_kernel`).

### 1.4 Autotuner candidates (`ProjectionRunner::run`)

rows <= 16: `sorted-lds`, `sorted-s8/s16` (rows <= 8), `sorted-s4` (rows <= 2), plus the
group-major kernels when codes are present. rows 17-64: `sorted-lds`, `sorted-dense`,
`pipelined-r2/r4`, BLAS. rows > 64: BLAS, `sorted-dense`, `sorted-lds` (<= 512 rows),
`pipelined-r4-64` (<= 512 rows, group-major; it beats hipBLASLt up to a few hundred rows).
Auto selects `sorted-s4/s8/s16` at rows=1 (0.537 ms/layer, 0.9 % over the forced best) and
`sorted-lds` at rows 8/32/64 when only the sorted layout is resident.

## 2. Microbenchmarks (`perf/micro_batched.hip`, synthetic uniform codes, gate shape = 22 M entries)

All times ms, device 0, median of 20. "MALL-free" = `NCOPY=4` rotating layout copies.

### 2.1 rows=1 stream kernel on 16-row tiles (`micro_batched gemv`)

| layout | gate | q | down | note |
| --- | ---: | ---: | ---: | --- |
| 32-row tiles, 3-bit (previous report, shipped kernel) | 0.111-0.119 | 0.054 | 0.108-0.120 | |
| 16-row tiles, 3-bit, chunk codes | 0.121-0.135 | 0.055-0.064 | 0.120-0.127 | 6.6 % dummies |
| **16-row tiles, 4-bit, chunk codes** | **0.112-0.122** | **0.053-0.061** | **0.110-0.117** | 0.4 % dummies, exact (max error 0) |

### 2.2 Batched kernel anatomy (gate, M=16 rows, RT=1, U=8, 16 waves; `phase*` modes)

| variant | ms | what it shows |
| --- | ---: | --- |
| decode only (stream -> LDS codes, no gathers) | 0.050-0.075 | DRAM/MALL-bound decode phase |
| gathers on hash-random codes, no decode (`HASH`) | 0.365-0.405 | the random-gather floor of this pipeline (= group-major `pipelined-64` 0.353-0.37) |
| group-major 2-byte codes from global, same pipeline (`GMCODES`) | 0.375-0.42 | same floor |
| gathers to one line + decode (`hitgather`) | 0.10-0.16 | decode + pipeline without gather misses |
| **decode + gathers, plain loads** | **0.72-0.80** | the stream halves the gather rate |
| decode + gathers, `nontemporal` / `SCOPE_SE` / `SCOPE_DEV` / `LU` stream loads | 0.68-0.78 | no help |
| half the stream bytes, plain loads | 0.78 | not proportional to volume |
| **decode + gathers, `SCOPE_SYS` stream loads** | **0.41-0.44** | shipped |
| + two K windows (two blocks per WGP) | 0.38-0.42 (MALL-free 0.415) | shipped |
| three / four windows | 0.47-0.51 | extra stream scans cost more |
| padded activation stride (`in + 64`) | +-2 % | not the limiter |
| rocWMMA `load_matrix_sync` A from LDS instead of global | 0.85 (vs 0.40) | byte-granular LDS loads; motivates raw operands |

The same table at M=64 (RT=4, U=4): HASH 0.49-0.52, shipped 0.54-0.59 (the decode phase is
mostly hidden behind the 4x heavier GEMM phase, +3-12 %); at M=32 (RT=2): HASH 0.41-0.43,
shipped 0.47-0.49.

### 2.3 Dense reconstruct from the sorted layout (`micro_batched recon`, gate, 176 MB dense)

row-major 0.45-0.58 ms (W=16: 0.45), group-major 0.49-0.59 ms, i.e. 300-390 GB/s of dense
writes; the old scattered reconstruct took 2.6-5 ms.

## 3. Projection benchmark, rows = 1 (`perf/results/batched/bench_rows1.json`)

| projection | stock best | ms | sorted best | ms | ratio |
| --- | --- | ---: | --- | ---: | ---: |
| q 12288->6144 | pipelined-128 | 0.161 | sorted-s4 | 0.062 | 0.39 |
| k 12288->512 | pipelined-128 | 0.038 | sorted-s16 | 0.032 | 0.85 |
| v 12288->512 | pipelined-128 | 0.038 | sorted-s16 | 0.032 | 0.85 |
| o 6144->12288 | pipelined-64 | 0.157 | sorted-s4 | 0.066 | 0.42 |
| gate 12288->14336 | pipelined-64 | 0.361 | sorted-s8 | 0.116 | 0.32 |
| up 12288->14336 | pipelined-n128-64 | 0.362 | sorted-s8 | 0.117 | 0.32 |
| down 14336->12288 | pipelined-64 | 0.361 | sorted-s4 | 0.117 | 0.33 |
| **layer** | | **1.477** | auto | **0.537** | **0.37** |

`sorted-lds` at rows=1 is 1.65 ms/layer (gather-bound like the stock kernels); the autotuner
never picks it below rows ~3.

## 4. Projection benchmark, batched (`bench_rows{8,32,64,256}.json`)

`sorted-lds` time per projection (ms), stock best in parentheses:

| projection | rows 8 | rows 32 | rows 64 | rows 256 (sorted-dense / sorted-lds) |
| --- | ---: | ---: | ---: | ---: |
| q | 0.191 (0.165) | 0.252 (0.203) | 0.289 (0.257) | 0.981 / 1.073 (0.895) |
| k | 0.041 (0.038) | 0.043 (0.041) | 0.047 (0.045) | 0.130 / 0.145 (0.125) |
| v | 0.041 (0.039) | 0.043 (0.041) | 0.047 (0.045) | 0.132 / 0.148 (0.125) |
| o | 0.178 (0.163) | 0.213 (0.202) | 0.300 (0.243) | 0.997 / 1.085 (0.900) |
| gate | 0.433 (0.371) | 0.540 (0.446) | 0.655 (0.543) | 2.250 / 2.497 (2.115) |
| up | 0.433 (0.371) | 0.536 (0.438) | 0.658 (0.537) | 2.249 / 2.478 (2.124) |
| down | 0.431 (0.372) | 0.523 (0.437) | 0.643 (0.522) | 2.219 / 2.405 (2.077) |
| **layer** | **1.748 (1.519)** | **2.149 (1.807)** | **2.639 (2.192)** | **8.959 / 9.831 (8.360)** |
| per row | 0.219 ms | 0.067 ms | 0.041 ms | 0.035 ms |

Stock modes at rows 8/32/64/256: pipelined-64/128/n128, pipelined-r2-128, pipelined-r4-64,
pipelined-r4-64. All cross-path checks pass (outputs within the existing tolerance of the fused
reference; the sorted-lds accumulation is fp32 WMMA like the group-major kernels).

Per rank per token (88 layers): rows=8 154 ms of projections (stock 134), rows=32 189 ms
(159), rows=64 232 ms (193), i.e. 19 / 5.9 / 3.6 ms per row.

## 5. Where the remaining 15-20 % goes

For gate at rows=8: sorted-lds 0.433 ms vs stock 0.371. Decomposition from section 2.2: the
random-gather floor of the pipeline is ~0.37-0.40 (equal to stock); the decode phase costs
0.05-0.07 of which the two-window overlap hides roughly half; the per-call extras (second
window's stream scan on DRAM, `merge_projection`, two extra launches) are ~0.01. The stock
kernels have no decode phase, so parity is the ceiling of this design, and the gap is the
non-overlapped part of the decode plus ~3 % of fixed costs.

What would close it (not done): a persistent block per WGP that double-buffers two K windows
in LDS (2 x 24 KiB) and dedicates 4 of its 16 waves to decoding window `w+1` while the others
gather window `w`, so the decode is fully hidden; estimated gain 0.04 ms per large projection
(~0.2 ms/layer at rows 8, which would land at ~1.55 ms vs the 1.5 target). Beyond parity, only a
kernel that avoids the random gather for batches (none found: dense-in-LDS reconstruct does not
fit, lane-owns-row ordering loses the L0 locality, int64-atomic accumulation scales with rows)
could make the sorted layout faster than group-major at batch >= 8.

## 6. End-to-end (`perf/e2e.sh`, `CONFIGS=batched_zerocopy CONCURRENCIES="1 8"`, 2 GPUs)

**Not measured: both GPUs fell off the PCIe bus at the first request of the run.** Timeline
(`perf/results/batched/gpu_fault_dmesg.txt`, `perf/results/batched/e2e_run.log`,
`perf/results/e2e/server_batched_zerocopy.log`): the server (commit `ed914a7` build,
`--code-layout sorted --transport zerocopy --memory-fraction 0.9 --prefill-chunk 256`, KV pages
4578 = 73k tokens) loaded both ranks and reported `Listening` at 15:11:27 (uptime 9755 s). The
c=1 warm-up request (39-token prompt: prefill rows=39 -> autotuning `sorted-lds`/`sorted-dense`/
BLAS, then rows=1 decode) hung immediately; at uptime 9758 s the kernel logged the audio
functions of both GPUs (`0000:33:00.1`, `0000:cc:00.1`) as `device inaccessible`; `load_bench`
timed out after 600 s with `active_requests 1, completion_tokens 0`; killing the server made
amdgpu report `MES failed to respond`, `GPU reset begin` and **`device lost from bus!` for both
GPUs** (uptime 10388-10513 s). The server process is stuck in `D` state
(`amddrm_sched_entity_flush`), TTM workers are blocked, `rocm-smi` no longer answers. Per the
safety rule the investigation stopped there (no reboot, no further GPU work); a power cycle is
needed before anything else runs.

This is the same failure class as the concurrency-16 incident in `DECODE_PERF_REPORT.md`
(stock build, host transport), now at concurrency 1 with this build. The faults in `dmesg`
before it (`micro_batched pid 23931`, uptime 8018 s) are the deliberate out-of-bounds diagnostic
run (`GMCODES+decode` variant) of the microbenchmark and recovered normally. What is new in this
run relative to the sorted-layout e2e that completed earlier today (`sorted_zerocopy`, 73.6
ms/token): the `sorted-lds` kernels, the `SCOPE_SYS` buffer loads for the stream, the padded
activation workspace, and ~1.3 GB more VRAM in use. The same kernels ran for ~20 minutes of
`behemoth_gpu_tests` and `behemoth_projection_bench` on device 0 without any fault, but only
device 0 and never together with the zero-copy transport (GPU spin-waits on fine-grained host
memory). Recommended order after the power cycle: (1) `behemoth_gpu_tests` on device 1
(`HIP_VISIBLE_DEVICES=1`) and `behemoth_projection_bench ... 1 all` to cover the second GPU;
(2) e2e with `--transport host` (`CONFIGS=batched_host`) before `zerocopy`; (3) if it recurs,
A/B the stream cache policy by setting `sorted_stream_policy` to 0 in `src/projection.hip`
(`SCOPE_SYS` is the one access mode this build adds that the earlier, working sorted build did
not use), then the padded-workspace memset, then the layout size. Expected e2e numbers if the
projections behave as in the bench: c=1 ~74 ms/token (unchanged: same rows=1 kernel speed),
c=8 ~210-220 ms/step (sorted-lds 1.75 ms/layer vs 2.76 before: ~245 ms of projections become
~155 ms) and TTFT ~0.9-1.0 s (sorted-dense prefill at 9 ms/layer vs 15-30 before).

## 7. Targets vs measured

| rows | target (ms/layer) | measured | status |
| ---: | ---: | ---: | --- |
| 1 | <= 0.6 | 0.54 | met |
| 8 | <= 1.5 | 1.75 | missed by 17 % (stock 1.52) |
| 32 | <= 2.0 | 2.15 | missed by 7 % (stock 1.81) |
| 64 | <= 2.5 | 2.64 | missed by 6 % (stock 2.19) |
| 256 (prefill) | <= stock | 8.96 vs 8.36 best / 12.9 stock auto | met vs stock auto, 7 % over best kernel |

## 8. Memory

Layer-0 layout bytes (2-byte codes -> sorted): q 18.9 -> 24.7 MB (+31 %, 2.1 % dummies), k 1.57
-> 2.05, v 1.57 -> 2.03, o 18.9 -> 25.9 (+37 %, 7.2 % dummies: 768 groups, density 0.19), gate/up
44.0 -> 56.7 (+29 %, 0.4 %), down 44.0 -> 56.6 (+29 %). Per rank: **19.8 GB** vs 18.5 GB for the
32-row/3-bit layout and 15.2 GB group-major. With 32 GiB per GPU the sorted layout leaves
~1.3 GB less for KV pages than before (see the server log in `perf/results/e2e/` for the page
count at `--memory-fraction 0.9`). Cheap follow-up: per-projection delta width (5 bits for the
768-group o_proj would remove most of its 7 % dummies).

## 9. Files

* `include/behemoth/sorted_layout.hpp`, `src/sorted_layout.cpp`: 16-row tiles, `chunk_codes`,
  4-bit default, decoder verifies chunk codes.
* `src/projection.hip`: `sorted_projection<W,U,C,DB>` (rows=1, chunk codes), `sorted_decode_tile`
  (SCOPE_SYS buffer loads), `sorted_lds_gemm<RT,U,DB>` (raw WMMA operands, K windows),
  `sorted_lds_reconstruct<DB>`, `sorted-lds` mode, candidate lists, padded activation workspace.
* `include/behemoth/gpu.hpp`: `GpuProjection::sorted_chunk_codes`, `sorted_bytes()`,
  `ProjectionMode::SortedLds`.
* `tests/gpu_tests.cpp`: rows 64/65/130. `tools/projection_bench.cpp`: layout bytes.
* `perf/micro_batched.hip`: all prototypes and diagnostics (`gemv`, `gemm`, `gemm2`, `recon`,
  `phase*` modes, `NCOPY`, `WIN`). `perf/summarize_layouts.py`: stock vs sorted tables.
* `perf/e2e.sh`: `batched_host` / `batched_zerocopy` configs, `OUT` override.
* `perf/results/batched/`: bench JSON (rows 1/8/32/64/256, `v1/` before K windows), test logs,
  `e2e_run.log`, `gpu_fault_dmesg.txt`, `server_debug_host.log` (second start attempt, hung at
  device init because the GPUs were already gone).

## 10. Server state at hand-over

`behemoth_server` is **not running**: PID 32465 (started 15:23 with `--transport host` to
reproduce the hang) is stuck in `D` state inside amdgpu after the bus loss and cannot be killed;
PID 31246 (the e2e server) is defunct. The intended hand-over configuration, to be started after
the power cycle (port 1337; profiling off):

```
cd /home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
setsid nohup build-rocm/behemoth_server --model /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output \
    --devices 0,1 --transport zerocopy --code-layout sorted --memory-fraction 0.9 \
    --token-budget 1024 --prefill-chunk 256 --host 0.0.0.0 --port 1337 --threads 4 --profile false \
    > /tmp/beh_server.log 2>&1 < /dev/null & echo $! > /tmp/beh_server.pid
```

Use `--transport host` first if the zero-copy transport is under suspicion for the bus loss.
