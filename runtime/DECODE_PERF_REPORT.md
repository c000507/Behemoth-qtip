# Behemoth-R1-123B-v2 AQLM 1x16 decode performance on 2x Radeon AI PRO R9700 (gfx1201)

Investigation date: 2026-10-25. Baseline: HF codebase branch `965c1cfa` (git baseline
`b0c3698` in this tree). Prototypes: commit `d21bdca` and later in this tree.
All raw numbers: `cpp_runtime/perf/results/`. Microbenchmarks: `cpp_runtime/perf/*.hip`,
`perf/*.py`.

## TL;DR

* **The AQLM 1x16 projections are not DRAM-bandwidth bound and cannot be made so on this
  GPU.** They are bound by the GPU-wide rate of random cache-line requests into L2
  caused by the 65536-entry (512 KiB) FP8 codebook gather: ~28.8 line requests per
  clock for the whole chip (~68 G gathers/s). One 8-byte gather per 8 weights means a
  hard floor of **~112 ms/token/rank for 1x16 at batch 1** (15.2 GB of codes per rank
  per token at ~135 GB/s "logical" code bandwidth). The production `pipelined-64/128`
  kernels already run at 120 GB/s, i.e. within ~12 % of that floor. Occupancy, MLP,
  LDS barriers, VGPRs, instruction count: none of these are the limiter (section 2).
* **A format with codebook tables <= 16 KiB decodes at DRAM bandwidth** on the same
  code stream: a 2x8 (two 256-entry books, 4 KiB) decode kernel measured 490-830 GB/s
  of codes (DRAM-capped ~600 GB/s for real weights), i.e. **~25-30 ms/token/rank for
  projections, ~45 ms/token end-to-end (~22 tok/s) instead of 167 ms (6 tok/s).**
  Getting there requires re-quantising (AQLM 2x8 / 2x10, QuIP#-E8P, QTIP, GGUF IQ2_*),
  or the calibration-free "codebook re-encoding" at 3 bpw described in section 3.3
  (measured reconstruction error 17.5 %, projected ~58 ms/token).
* **Batched decode already amortises the gather:** rows 1 -> 16 cost 1.44 -> 1.52 ms
  per layer on one rank (per-row cost 126 -> 8.4 ms). The old kernels fall off a cliff
  at 17+ rows (2x at 32). The new `pipelined-r2-*/r4-*` modes (prototype, all GPU tests
  and cross-path checks pass) look each weight tile up once for 32/64 rows:
  **rows=32: 1.75 ms/layer (was 2.96), rows=64: 2.12 ms/layer (was ~5.9 est.)**, i.e.
  per-row projection cost 4.8 ms and 2.9 ms.
* **Collectives:** the pinned-host all-reduce costs 67 us per 48 KiB message (176 per
  token = ~12 ms, matches the profiled 14-17 ms). A stream-ordered zero-copy exchange
  through fine-grained host memory (prototype `--transport zerocopy`) does it in
  **18.8 us (3.3 ms/token)**; RCCL SHM with P2P disabled is not usable (seconds per op /
  hangs). E2E results in section 6.
* **Baseline end-to-end (measured):** 163 ms/token inter-token at concurrency 1 (5.9-6.1
  tok/s), 198 ms/step at concurrency 8 (40 tok/s aggregate). **Both GPUs fell off the
  PCIe bus during the concurrency-16 baseline run** (`amdgpu: device lost from bus`,
  section 6.2); the machine needs a power cycle and the prototype end-to-end numbers
  could not be taken.
* **Launch overhead:** 2.3 us per kernel back-to-back; HIP graphs do not help (2.28 us).
  ~42 kernels + 4 memcpys per layer ≈ 3.7 k launches/token ≈ 8.5 ms floor; fusing the
  tiny elementwise kernels is worth ~4 ms/token.

Single-request decode budget (per rank, ms/token):

| component | measured (965c1cfa) | floor with 1x16 | with <=16 KiB codebooks |
| --- | ---: | ---: | ---: |
| projections | 126-132 | ~112 (gather rate) | 25-30 (DRAM) |
| collectives | 14-17 | 3.3 (zero-copy, measured) | 3.3 |
| attention + KV (ctx 75) | 9 | ~3 (fusion) | ~3 |
| elementwise + launches | 7.5 | ~3.5 | ~3.5 |
| lm_head | 0.4 | 0.4 | 0.4 |
| **total** | **~167** | **~122 (8.2 tok/s)** | **~40-45 (22-25 tok/s)** |

## 1. Hardware facts measured

* HIP reports 32 multiprocessors (WGPs) x 64 waves, 64 CUs, 2.35 GHz, 64 KiB LDS per
  WGP (also the per-block max), 8 MiB L2 (256 B lines reported by rocminfo), 256-bit
  GDDR6 at 1258 MHz -> 640 GB/s nominal. `v_dot4_f32_fp8_fp8` and
  `v_cvt_pk_f32_fp8` are available on gfx1201 (used by the microbenchmark kernels).
* `perf/results/stream.txt`: streaming read of 1 GiB with 128-bit loads: **637 GB/s**
  (99.5 % of nominal). Bandwidth is not the problem.
* `perf/results/launch.txt`: 1000 back-to-back empty kernels: **2.23-2.62 us/launch**;
  the same 1000 kernels as a HIP graph: 2.28 us/kernel. Graphs do not reduce dispatch
  cost on this stack.

## 2. Diagnosis: why 1x16 lookups run at ~120 GB/s

### 2.1 Random-gather microbenchmark (`perf/micro.hip gather`, `perf/results/gather.txt`)

Each thread streams 16-bit codes (coalesced) and gathers one 8-byte codebook entry per
code from a table of the given size; U independent chains per thread, 2048 blocks x 256
threads. Numbers are per-GPU gathers/s and the equivalent code-stream bandwidth.

| table | where it fits | G gathers/s | code GB/s | gathers/clk/CU |
| ---: | --- | ---: | ---: | ---: |
| 4-16 KiB | L0 (32 KiB/CU) | 280-307 | 560-613 | 1.9-2.0 (code-stream limited) |
| 32 KiB | L0 thrashing | 245-264 | 490-528 | 1.6-1.8 |
| 64 KiB | L2 | 120-122 | 240 | 0.80 |
| 64 KiB in LDS | LDS | 173-205 | 345-410 | 1.15-1.37 |
| 128 KiB | L2 | 84 | 168 | 0.56 |
| 256 KiB | L2 | 72.5 | 145 | 0.48 |
| **512 KiB (1x16 book)** | L2 | **67.7** | **135** | **0.45** |

U (4/8/16) and the block count (1024-4096) change nothing for tables >= 64 KiB: the
limiter is not per-wave memory-level parallelism.

### 2.2 It is a chip-wide L2 request-rate limit, not a per-CU limit (`perf/results/gatherblocks.txt`)

512 KiB table, 8 waves per block, varying the number of active WGPs:

| blocks (WGPs) | G gathers/s | per-block gathers/clk | GPU gathers/clk |
| ---: | ---: | ---: | ---: |
| 1 | 4.6 | 1.96 | 2.0 |
| 8 | 34.9 | 1.86 | 14.9 |
| 16 | 53.3 | 1.42 | 22.7 |
| 32 | 67.7 | 0.90 | 28.8 |
| 64-256 | 67.4 | 0.45-0.11 | 28.7 |

A single WGP can issue ~2 L0-missing gathers per clock; the whole chip saturates at
**~28.8 random 128-B line requests per clock (~68 G/s)**. Every 8-byte gather from a
512 KiB table misses the 32 KiB L0 (~6 % hit rate) and costs one L2 line request.
Multiply: 15.2 GB of codes per rank per token = 7.6 G gathers -> **112 ms**. The
pipelined-64 kernel (0.159 ms for q: 18.9 MB codes = 9.4 M gathers -> 59 G/s = 87 % of
the ceiling) is already close.

Consequences:

* More waves, wider loads, deeper prefetch, fewer barriers: no gain (microbenchmark
  insensitivity to U/blocks; my register-only `gemv_1x16` kernel without LDS/barriers,
  `perf/micro.hip gemv`, reaches 120-125 GB/s = same ceiling as pipelined-64/128).
* FP8 activation path, dequant-then-WMMA structure, VGPR count (49-69), LDS barriers:
  irrelevant at this point; the WMMA with 15/16 zero rows at batch 1 is free.
* `rocprofv3` counter collection (TCP_REQ/GL2C_*) returned zeros or crashed (signal 6)
  for these kernels on ROCm 7.2.4/gfx1201; the microbenchmarks above are the evidence.

### 2.3 Code usage is uniform -> no hot-set trick (`perf/results/code_histogram.txt`)

Entropy of code usage per projection is 15.9-16.0 bits (of 16) for all layers except
layer 0 (14.2-15.6). Coverage of the 8192 most frequent entries (what a 64 KiB LDS cache
could hold) is 16-18 % (uniform = 12.5 %), 32-40 % only in layer 0. Caching a hot subset
in LDS/L0 would remove at most ~15 % of L2 gathers: not worth the occupancy loss.

### 2.4 Infeasible: per-group dot-product LUT

A "codebook x activation" table for 1x16 is 65536 entries per 8-input group: 65536 x
1536 groups x 2 B = **201 MB per projection input** (1.4 GB per layer), rebuilt every
token and then read at random. Dead on arrival; only viable for <= 256-entry books
(see 3.3), where it is unnecessary because the direct dot4 kernel is already DRAM-bound.

## 3. Single-request options (ranked)

| # | option | projections ms/token | e2e ms/token (tok/s) | effort | risk |
| --- | --- | ---: | ---: | --- | --- |
| 1 | Re-quantise to a small-codebook format (AQLM 2x8 / 2x10, QuIP# E8P, QTIP, GGUF IQ2_XS) + dot4 GEMV kernel (prototype exists) | 25-30 | 40-45 (22-25) | high: calibration run for 123B (days of multi-GPU compute, needs the bf16 original which is not on this machine), new loader/kernels (~1-2 days) | quality change depends on method; 2x8 AQLM is a bit worse than 1x16 in the AQLM paper; QTIP/E8P are equal or better at 2 bpw |
| 2 | Calibration-free codebook re-encoding to 3x8 (3 bpw) or 4x8 (4 bpw) | 38 / 50 | ~55 / ~67 (18 / 15) | medium: offline script (minutes per model, already written), new code layout + kernel (~2 days) | **measured** weight reconstruction error 17.5 % (3x8), 10 % (4x8) relative to the 1x16 weights; expect small ppl hit for 4x8, moderate for 3x8; needs ppl eval |
| 3 | Zero-copy collectives (prototype `--transport zerocopy`) | - | -8 to -10 | done (prototype) | GPU spin-wait on host flag; abort/hang handling needed for production |
| 4 | Fuse elementwise kernels (quantize once for q/k/v and gate/up, merge+cast, cast+residual+norm) | - | -3 to -4 | low (1 day) | none |
| 5 | Attention/KV small-batch fusion (rope+quantize+append) | - | -2 to -4 | low | none |
| 6 | Squeeze the last 12 % of the 1x16 gather ceiling (persistent blocks, tail balance) | 126 -> ~115 | -10 | medium | diminishing returns |
| 7 | Fused gate+up, HIP graphs, more MLP / wider loads / different tile geometry | ~0 | ~0 | - | measured: no effect (sections 1, 2) |
| 8 | Pipeline parallel (one GPU per request) instead of TP | 2x worse per request | - | - | each GPU would gather the full 15.2 GB of... no: 30 GB per token per GPU (2x the per-rank ceiling), only useful with >= 2 concurrent requests, equivalent to batching |

Honest bottom line for batch 1 with the existing 1x16 checkpoint: **~120-125 ms/token
(8-8.5 tok/s)** after items 3-6, versus 167 ms today. Reaching 35-45 ms needs a format
change (item 1 or 2).

### 3.1 Why formats with small codebooks hit DRAM speed (measured)

`perf/micro.hip gemv` runs a hypothetical additive 2x8 format (code = hi byte | lo byte,
two 256-entry 8-dim FP8 books = 4 KiB) on the real projection shapes with random codes,
using `v_dot4_f32_fp8_fp8` (2 dot4 per 8 weights per book), 8 groups per thread in
flight, no LDS, split-K partials + merge:

| shape | 1x16 kernel (ceiling) | 2x8 books in LDS | 2x8 books via L0 |
| --- | ---: | ---: | ---: |
| q 12288->6144 (18.9 MB) | 0.153 ms, 123 GB/s | 0.038 ms, 496 GB/s | 0.028 ms, 677 GB/s |
| gate/up 12288->14336 (44 MB) | 0.352 ms, 125 GB/s | 0.077 ms, 574 GB/s | 0.053 ms, 829 GB/s |
| down 14336->12288 (44 MB) | 0.353 ms, 125 GB/s | 0.077 ms, 574 GB/s | 0.053 ms, 834 GB/s |

Values above 640 GB/s are MALL (64 MiB infinity cache) assisted because the benchmark
re-reads the same 44 MB tensor; real decode streams 15 GB per token, so expect ~600
GB/s -> **~25-27 ms/token/rank for all projections.** The L0 path (plain global loads
from a 4 KiB table) beats LDS: no staging, no bank conflicts, 32 KiB L0 keeps a <= 16 KiB
table resident while codes stream through.

Format candidates with <= 16 KiB tables: AQLM 2x8 (4 KiB), AQLM 2x10 (16 KiB, 2.5 bpw),
GGUF IQ2_XXS/IQ2_XS/IQ2_S (E8-lattice grids of 2-8 KiB + signs/scales, 2.06-2.5 bpw),
QuIP# E8P (65536 virtual entries decoded from a 256-entry base table + sign flips),
QTIP (trellis, no table, ~10 VALU ops per weight: compute cost ~8 ms/token on this
GPU, acceptable).

### 3.2 Why 1x16 -> LDS / "2x8 split" without re-quantisation does not work

* 512 KiB does not fit the 64 KiB LDS; even a 64 KiB LDS table only reaches 1.37
  gathers/clk/CU (section 2.1) and forces one block per WGP.
* A 1x16 code is an arbitrary index: it cannot be split into two 8-bit sub-codes without
  re-fitting the codes.

### 3.3 Calibration-free codebook re-encoding (`perf/codebook_reencode.py`, `perf/results/codebook_reencode*.txt`)

Idea: keep the AQLM codes' *information* but re-express each of the 65536 codebook
vectors (8 FP8 values) as a sum of M entries from small books (K entries each); rewrite
the code tensor offline (each 16-bit code -> M small codes). No calibration data, no
original weights needed; fit takes 10-60 s per projection on CPU (616 projections ->
~2-10 h single-threaded, trivially parallel). Weighted by actual code usage.

| re-encoding | bpw | table | weighted rel. error of dequantised weights |
| --- | ---: | ---: | ---: |
| FP8 runtime codebook (what the server already does) | 2.0 | 512 KiB | 2.6 % |
| 2x8 (K=256) | 2.00 | 4 KiB | **31 %** (too lossy) |
| 2x10 (K=1024) | 2.50 | 16 KiB | 21.6 % |
| 2x11 (K=2048) | 2.75 | 32 KiB | 17.5 % |
| 3x8 (K=256) | 3.00 | 6 KiB | 17.5 % |
| 3x9 (K=512) | 3.38 | 12 KiB | 13.5 % |
| 4x8 (K=256) | 4.00 | 8 KiB | 10.2 % |

Interpretation: AQLM 2-bit weights already carry ~35-45 % relative quantisation noise
versus the bf16 model; adding e independent relative error gives total
sqrt(0.4^2 + e^2): 4x8 -> +3 %, 3x8 -> +9 %, 2x8 -> +22 % of weight noise. 3x8 at 1.5x
the code bytes gives a DRAM-bound **~38 ms/token/rank projections (3x faster than
today)**; 4x8 ~50 ms. This is the fastest path that needs no calibration run, but it
must be validated with a perplexity/benchmark comparison before use; a short
fine-tune of the small books against layer outputs (as AQLM does) would recover part of
the loss.

## 4. Collectives (`perf/micro_coll.hip`, `perf/results/coll*.txt`)

Two-GPU all-reduce latency, 12288 floats (one decode row), 200 iterations:

| transport | per all-reduce | per token (176) |
| --- | ---: | ---: |
| current pinned-host staged (D2H, sync, 3 host barriers, CPU sum, H2D, sync) | 67 us | 11.8 ms |
| **zero-copy kernel-driven (prototype)** | **18.8 us** | **3.3 ms** |
| RCCL SHM, NCCL_P2P_DISABLE=1 | smoke test: 0.024 GB/s (~1 s for 10 ops); 10 back-to-back ops hang beyond 300 s | unusable |

Larger messages (zero-copy reads the peer buffer over PCIe at ~5 GB/s, uncached):
4 rows 80.6 -> 46.8 us, 8 rows 111.7 -> 85.2 us, 16 rows 159.9 vs 182.6 us (host wins).
The prototype therefore uses zero-copy for <= 8 rows (`zero_copy_rows`) and the staged
path above. For 16-64 rows the right design is a stream-ordered memcpy exchange
(D2H to own host buffer, flag, peer H2D) or reduce-scatter + all-gather; estimated
~100 us per 32-row all-reduce versus ~250 us today (not prototyped).

The collective cost is latency (host rendezvous + 2 stream syncs + 2 memcpy launches),
not bandwidth: 48 KiB at PCIe 5.0 x16 is ~2 us.

## 5. Batched decode

Per-layer, per-rank projection time (best mode, `behemoth_projection_bench`, layer 0,
device 1, `perf/results/base_rows*.json`, `multirow_rows*.json`):

| rows | 965c1cfa best | per row | prototype best | per row |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.436 ms | 126 ms/token | - | - |
| 8 | 1.471 | 16.2 | - | - |
| 16 | 1.524 | 8.4 | 1.52 (unchanged) | 8.4 |
| 32 | 2.960 (tiled/pipelined, 2 row tiles) | 8.1 | **1.746 (pipelined-r2-128)** | **4.8** |
| 64 | ~5.9 (2 x rows=32, est.) | 8.1 | **2.121 (pipelined-r4-64)** | **2.9** |

BLAS after dequant is never competitive here: the dequant is itself the 7.6 G-gather
problem (`reconstruct_kernel`) plus a full FP8 weight write/read (1.07 ms for q, 2.65 ms
for gate at rows=32 versus 0.20 / 0.43 ms for the multi-row pipelined kernel). It would
only win when the dequantised weight is reused across >= ~10 row tiles (>= 160 rows),
which is prefill territory.

Prototype details (`src/projection.hip`, modes `pipelined-r2-64/128`,
`pipelined-r4-64/128`): the block owns 32 or 64 rows; the weight tile is gathered and
staged once, `mb` is loaded once per K step and multiplied against 2/4 activation
tiles into 2/4 accumulators. 89 VGPRs (r2-128) / 129 VGPRs (r4-128), 13-17 KiB LDS. Auto
candidates for 17-64 compressed rows are the r2/r4 modes plus BLAS/fused; the server
reserves split-K scratch for 64 rows. `behemoth_gpu_tests` (every registered mode,
rows 1/2/7/16/33, tails) and the bench cross-path checks pass.

Aggregate projection throughput per rank at batch N (88 layers): N=8: 61 tok/s,
N=16: 119, N=32: 183 (prototype), N=64: 341 (prototype) -- before attention,
collectives and elementwise. Real aggregate numbers: section 6.

## 6. End-to-end measurements

Script: `perf/e2e.sh` (server start, warm-up runs per concurrency so first-use autotuning
is excluded, then `behemoth_load_bench` with 2x concurrency requests of 96 tokens, prompt
39 tokens). Results: `perf/results/e2e/`.

### 6.1 Baseline binary (965c1cfa, `--transport host`, `--profile false`)

| concurrency | load_bench aggregate tok/s (includes TTFT) | server inter-token p50 / p95 | steady-state aggregate (c / inter-token) | TTFT p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.86 | 163.1 / 168.1 ms | 6.1 tok/s | 787 ms |
| 8 | 20.5 | 197.9 / 204.0 ms | 40 tok/s | 2.44 s (p95 4.49 s) |
| 16 | **server hung** (16 active requests, GPU step counter frozen for 14 min) | - | - | - |

The batch-8 step (198 ms) is 35 ms slower than batch 1: ~20 ms of that is the staged
all-reduce at 8 rows (112 us x 176, section 4), the rest attention/elementwise growth.
The 2.4-4.5 s TTFT at concurrency 8 is prefill of 8 x 39 tokens through the generic
fused/BLAS path plus decode-priority scheduling; prefill was out of scope here.

### 6.2 Hardware failure during the run (important)

While the baseline server was serving the concurrency-16 run, both GPUs fell off the
PCIe bus. Kernel log (`perf/results/e2e/gpu_fault_dmesg.txt`):

* 11:46:32 `snd_hda_intel 0000:33:00.1 / 0000:cc:00.1: Unable to change power state from D3hot to D0, device inaccessible` (both cards' audio functions, first sign of PCIe trouble on both boards; decode was still working).
* ~11:55-12:00 server stops making progress at 16 concurrent requests (the pinned-host transport waits on an SDMA copy that never completes).
* 12:01:24 ptrace attach (gdb, my diagnostic) -> 12:01:31 `MES(0) failed to respond to msg=REMOVE_QUEUE ... GPU reset begin! Source: 3 ... device lost from bus! ... GPU reset end with ret = -19` on 33:00.0, and 12:01:33 the same on cc:00.0.
* 12:03 `ring sdma0 timeout`, `GPU Recovery Failed: -19`, devcoredump written for card2; from then on every access reports `device lost from bus`.

Both GPUs are gone until the machine is power-cycled (not done: reboots were out of
scope). The last server process (PID 31505, the prototype `--transport zerocopy` start
from `perf/e2e.sh`) is stuck in D state in `amddrm_sched_entity_flush` and cannot be
killed; the two earlier servers were killed by exact PID. Because of this, **the
prototype builds have no end-to-end server measurement**; their component measurements
(sections 4 and 5) stand, and `behemoth_gpu_tests` / `--two-gpu` / bench cross-path
checks passed on the prototype build before the failure.

I cannot tell whether the hang at 16 requests was a software bug or the beginning of
the hardware failure: the same binary served concurrency 1 and 8 correctly minutes
before, and the audio functions of both boards were already inaccessible at 11:46.
Earlier in the session `rocprofv3` (counter collection) aborted with SIGABRT on GPU 0 and
had to be killed; GPU tests and the two-GPU smoke passed afterwards. Re-run
`perf/e2e.sh` after a power cycle: it skips configurations that already have results and
records hangs with a timeout instead of blocking.

Projected from the component measurements (to be confirmed by `perf/e2e.sh`):
prototype `--transport zerocopy` at batch 1: 163 - 8.5 = **~155 ms/token (6.5 tok/s)**;
at batch 8: 198 - 4.7 = ~193 ms/step (41 tok/s); at batch 32 with the multi-row kernels:
projections 154 ms + collectives ~45 ms (staged, 1.5 MiB messages) + attention/elementwise
~25 ms = ~225 ms/step -> **~140 tok/s aggregate**, versus ~330 ms/step (97 tok/s) with
the old kernels; at batch 64: ~187 + 80 + 35 = ~300 ms -> ~210 tok/s.

## 7. Recommended roadmap

1. **Ship the two prototypes** after hardening: multi-row pipelined tiles (done, tests
   pass) and the zero-copy transport (add abort/timeout handling for the spin-wait, keep
   the staged path as fallback, extend to 16-64 rows with a stream-ordered memcpy
   exchange). Expected: batch-1 ~155 ms/token, batch-32 projections 1.7x faster.
2. **Cheap kernel-count cuts** (1 day): quantise the normed input once for q/k/v and
   gate/up (saves 3 launches/layer and 2 x 12288-element passes), fuse merge+cast and
   cast+residual+rms_norm, fuse rope+quantize+append_kv. ~4-6 ms/token.
3. **Decide the format question.** The 1x16 codebook is the wall: ~112 ms/token/rank is
   the best any kernel can do on gfx1201. Choose between
   (a) re-quantising from the bf16 Behemoth-R1-123B-v2 to a small-table 2-2.5 bpw format
   (best quality/speed, needs calibration compute and the original weights);
   (b) the calibration-free 3x8/4x8 re-encoding of the existing checkpoint
   (`perf/codebook_reencode.py`), 3-4 bpw, 2.5-3.3x projection speed-up, needs a
   perplexity check. Either way the decode kernel is the `gemv_2x8`-style dot4 GEMV
   already measured at DRAM speed (extend to 3-4 books and multi-row WMMA for batching).
4. After (3), collectives and elementwise dominate (~15 of ~45 ms): zero-copy
   collectives and fusion become mandatory, and prefill (`--prefill-chunk`) should move
   to a dequant-tile + WMMA path with 64-256 rows per weight tile.

## 8. Files

* `perf/micro.hip` -- stream, gather, gatherblocks, launch, gemv (1x16 and 2x8 prototypes)
* `perf/micro_coll.hip` -- host-staged vs zero-copy vs RCCL all-reduce latency
* `perf/code_histogram.py`, `perf/codebook_reencode.py`, `perf/summarize_bench.py`, `perf/e2e.sh`
* `perf/results/` -- all raw outputs referenced above
* `src/projection.hip`, `include/behemoth/gpu.hpp` -- multi-row pipelined modes
* `src/transport.hip`, `src/model.cpp`, `src/server.cpp`, `include/behemoth/model.hpp` -- `--transport zerocopy`
