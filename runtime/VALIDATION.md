# Validation status

No R9700 GPU is attached to the development workspace; hardware results below
are explicitly identified as user-reported.

## Checked locally

- C++20 host build using GCC 14.2.
- Host correctness tests under AddressSanitizer and UndefinedBehaviorSanitizer:
  OCP E4M3 exhaustive roundtrips and independently computed nearest-value checks,
  BF16 rounding, F16 file decoding, independent scalar attention, causal masking,
  page boundaries/alias rejection, head sharding, KV append, page reclamation,
  scheduler progress under oversubscription and preemption, seeded sampling,
  partial stop strings, UTF-8 streaming, protocol validation, and safetensors bounds.
- Tokenizer tests using the actual tokenizer.model and tokenizer_config.json from
  pinned checkpoint revision `61a69e73fcdbcd8a0b81237c0092aa93518a5c52`.
- Complete Release CMake compile/link using the ROCm 7.2.1 SDK: clang 22,
  rocWMMA 2.2, hipBLASLt 1.2.2, RCCL 2.27.7, Boost 1.83,
  nlohmann JSON 3.11.3, and SentencePiece 0.2.1.
- Native `gfx1201` HIP objects, GPU test/benchmark executables, full-model/block
  test executables, and HTTP server/client test executables compile and link.
- Generated attention assembly contains `v_wmma_f32_16x16x16_fp8_fp8` instructions
  and `.wavefront_size: 32` metadata. This confirms code generation, not execution.
- The current sanitized host suite passes 78,341 assertions, including an exact
  blocked index transpose with partial tiles and the maximum uint16 code.
  Release host tests, pinned-tokenizer tests, shell syntax, and staged whitespace
  checks also pass.

ROCm packages were extracted into a temporary local toolchain to perform the
compile/link check. This does not supply GPU hardware or establish that kernels
execute correctly. Downloaded packages, credentials, model assets, build products,
and third-party source trees are not committed.

## User-reported R9700 results for commit 853bc14

- `core_correctness`, tokenizer tests and checkpoint inspection passed.
- Dual-GPU pinned-host transport passed with P2P unavailable; both cards report
  PCIe 5.0 x16 and BAR apertures covering VRAM.
- The unmodified one-GPU suite failed its conversion check for exactly two of
  10,263 values: positive and negative infinity. The native constructor produced
  signed NaNs (`0x7f`/`0xff`), while the reference saturated to `0x7e`/`0xfe`.
- With only those two inputs excluded locally, all other GPU tests passed:
  finite conversion, KV append, projections, prefill, decode, split/merge,
  guarded output buffers and stream checks. This is not recorded as a passing
  unmodified GPU suite.

The runtime wrapper now explicitly handles signed infinities and NaNs, leaving
native finite-input conversion intact. Infinities remain in the test, and both
startup and full conversion tests now include signed NaNs and maximum finite
FP32 inputs. The user subsequently reported all tests passing after this correction.

## User-reported projection profile after 733122f

Per-rank projection totals are 268.7 / 263.6 ms, about 87% of the reported GPU
step totals. Collectives (including host rendezvous waits) are 14.4 / 19.7 ms,
attention/cache 9.0 / 8.9 ms, elementwise 7.7 / 7.6 ms, and LM head 0.43 ms each.
For one decode row, gate/up select fused at about 0.86 ms each; down selects
split-K at 0.53 ms, Q at 0.27 ms, O at 0.26 ms, and K/V at 0.06 ms each.
These measurements prioritize projection work. They do not independently
establish which hardware resource limits it. The runtime FP8 codebook occupies
512 KiB, versus 1 MiB for the source F16 representation.

Commit `7a88e6d` added group-major index uploads and four-wave tiled candidates.
Both layouts and all forced paths are included in the GPU reference tests and
real-checkpoint benchmark. Static gfx1201 assembly metadata reports Wave32,
zero private-segment bytes, and the following resources:

| Kernel | Threads | LDS bytes | VGPRs/thread |
| --- | ---: | ---: | ---: |
| fused / split-K | 32 | 2,048 | 41 |
| tiled-64 | 128 | 6,144 | 36 |
| tiled-128 | 128 | 11,264 | 47 |

This establishes compilation and resource allocation, not correctness, achieved
occupancy, cache traffic, bandwidth, or speedup. The benchmark queries runtime
kernel attributes as well. The server now separates execution wall time from
profile-collection time to help reconcile GPU totals with end-to-end throughput.

## User-reported R9700 results after 7a88e6d

The user reports decode improving from 307 to 178 ms/token (about 5.6 tokens/s,
1.72x throughput) and time to first token falling from 377 to 202 ms. The standalone
projection benchmark's estimate agrees with decode timing. At context 75, the
reported step wall time is 179.8 ms:

| Stage | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| Projections | 142.6 ms | 140.7 ms |
| Collectives, including host waits | 13.2 ms | 15.3 ms |
| Attention and cache | 8.8 ms | 8.7 ms |
| Elementwise | 7.6 ms | 7.5 ms |
| LM head | 0.39 ms | 0.39 ms |

Every compressed projection uses the default group-major layout. Auto selects
mostly tiled-128, with tiled-64 for K/V and one-row gate/up. The dense LM head
uses BLAS. The user reports coherent output with thinking enabled. This is an
end-to-end observation, not a numerical quality evaluation or a report that all
expanded GPU tests passed. The earlier 1.25 tokens/s result preceded the 307 ms
baseline; it is not the baseline for the 1.72x comparison.

## Kernel changes in d434962

New `tiled-lds-64` / `tiled-lds-128` variants retain the tile loads and matrix
computations, replacing full HIP barriers with local-memory release/acquire
fences and block synchronization. Only LDS is shared between waves. Generated
gfx1201 assembly omits the old `global_inv` and combined global/LDS waits while
preserving LDS synchronization. Register/LDS counts match the original tiles.

New `pipelined-64` / `pipelined-128` candidates additionally group independent
index loads before their codebook loads and prefetch the next tile into registers
before current-tile WMMA. The aligned compressed group-major specialization omits
dense/layout/tail branches and bounds uint32 code addressing at dispatch. Generic
variants support the other layouts, dense weights and tails. The compiler emits
grouped index and codebook loads, native FP8 WMMA, and no cache invalidation at
LDS barriers. Compiled resources for both aligned and generic variants are:

| Kernel | Threads | LDS bytes | VGPRs/thread | Private bytes | Wave size |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiled-lds-64 | 128 | 6,144 | 36 | 0 | 32 |
| tiled-lds-128 | 128 | 11,264 | 47 | 0 | 32 |
| pipelined-64 | 128 | 6,144 | 49 | 0 | 32 |
| pipelined-128 | 128 | 11,264 | 69 | 0 | 32 |

These static observations do not establish execution correctness, achieved
overlap, occupancy or improvement. The higher pipeline register counts may cost
occupancy. All previous candidates remain in the benchmark and autotuner; no
speedup is assumed. FP8 products, scaling, split boundaries, precision and memory
budget are unchanged. No GPU speed measurement for these new candidates is
available from the development environment.

## User-reported R9700 results after d434962

The user reports decode improving from 178 to 167 ms/token (about 6.0 tokens/s,
6.6% higher throughput) and time to first token falling from 202 to 191 ms. At
context 75, the step wall time is 169.1 ms:

| Stage | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| Projections | 131.6 ms | 131.6 ms |
| Collectives, including host waits | 13.6 ms | 13.8 ms |
| Attention and cache | 8.8 ms | 8.7 ms |
| Elementwise | 7.5 ms | 7.5 ms |
| LM head | 0.40 ms | 0.39 ms |

The standalone projection benchmark predicts about 132 ms. Auto selects
pipelined-64 or pipelined-128 for every compressed projection, and dense BLAS
for the LM head. The user reports coherent output with thinking enabled.
Projections remain about 78% of step wall time. No numerical test-suite result
was supplied with this timing report.

## Geometry candidates in ce03e68

That update retained the original four-wave / 1024-input pipelines and
added eight candidates: K tiles of 64/128 with either 32 or 128 output columns
(two/eight waves, retaining 1024-input partitions), and K tiles of 64/128 with
512 or 2048 inputs per partition (retaining 64 output columns / four waves).
Output geometry and split width are varied separately. Scales and FP8 products
are unchanged; changing the partition width changes FP32 summation grouping.

Geometry metadata is reported per dispatch candidate, including thread
count, output columns, K tile, partition width and aligned/generic choice. The
runtime resource report also queries HIP's theoretical active-block/wave limits
for each kernel and thread count; it does not measure achieved occupancy. The
minimum partition remains 512, fitting the existing preallocated partial
workspace. No persistent GPU allocations or KV capacity changes are needed.

Full ROCm Release compilation/linking passes. Static gfx1201 metadata reports
Wave32 and zero private-segment bytes for all pipeline variants:

| Output columns | K tile | Threads | LDS bytes | Aligned VGPRs | Generic VGPRs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 64 | 3,584 | 52 | 54 |
| 32 | 128 | 64 | 6,656 | 76 | 76 |
| 64 | 64 | 128 | 6,144 | 49 | 49 |
| 64 | 128 | 128 | 11,264 | 69 | 69 |
| 128 | 64 | 256 | 11,264 | 49 | 49 |
| 128 | 128 | 256 | 20,480 | 64 | 66 |

The 64-column register/LDS counts hold for all three partition widths. Static
resource differences do not establish achieved occupancy or speedup. The new
geometries have not run on a GPU in the development workspace; previous results
do not validate them.

## User-reported R9700 results after ce03e68

The wider geometry search produced no end-to-end gain: decode remains
167 ms/token. The reported one-row projection benchmark, per rank, is:

| Projection | Auto | Fastest forced mode | Fastest ms | Previous auto |
| --- | ---: | --- | ---: | ---: |
| Q | 0.196 | pipelined-s512-64 | 0.169 | 0.165 |
| K | 0.048 | pipelined-64 | 0.047 | 0.041 |
| V | 0.048 | pipelined-n32-128 | 0.048 | 0.042 |
| O | 0.187 | pipelined-64 | 0.164 | 0.162 |
| Gate | 0.377 | pipelined-s2048-64 | 0.366 | 0.368 |
| Up | 0.419 | pipelined-s2048-64 | 0.374 | 0.363 |
| Down | 0.375 | pipelined-n128-128 | 0.369 | 0.357 |
| Sum/block | 1.650 | | 1.537 | 1.498 |

The best forced times show no kernel gain over the preceding build. Auto differs
from the fastest forced result by about 16% for Q and 12% for up in this run.
This identifies a selection/measurement issue, without proving the relative
contribution of cache sharing, clock/cache state, or timing noise.

## Current autotuning correction: host tested, GPU validation pending

Code inspection identified one three-launch mean per candidate, fixed candidate
order, shape-only choice sharing across weight tensors, and auto measured first
in a separate benchmark phase from forced modes. Gate/up and K/V inherited one
shape decision despite different codebooks. Tuning also excluded input
quantization while the standalone benchmark included it.

The current changes keep BLAS descriptors/algorithms cached by shape and split
out kernel choices. Up to 16 rows use weight identity, shape/layout/rows and
row-scaling policy; reupload changes identity and cannot inherit a stale choice
even if GPU allocation addresses are reused. Larger-prefill choices retain
shape sharing to bound cold startup. Search-policy changes clear choices.
The default conservative search uses the two proven pipelines for model-sized
compressed decode. Expanded geometries remain forced options and are included
only when exhaustive search is explicitly selected. Dense/large-batch choices
retain BLAS/fused, and 4..16 compressed rows may also consider available BLAS.

Timing now warms each path, calibrates 1..32-call batches targeting 0.5 ms, runs
seven interleaved forward/reverse rounds, and confirms the two finalists with
three ABBA cycles. Switching to the later finalist requires a median gain above
2% and wins above that margin in at least two cycles. Both timing paths include
quantization and reconstruction. Raw samples and confirmation gains are exported.
Weight-specific calibration can add seconds on first use of a new small batch;
cache hits do not tune. No measured GPU latency improvement is claimed.

The standalone benchmark warms all forced paths before auto tuning, then
interleaves repeated-batch measurements. Its auto record names the exact selected
mode and reports a paired ABBA comparison against that forced mode, including
sample arrays, medians and relative auto overhead. Row scaling matches each TP
projection's role. The server reports grouped selections and timing ranges to
avoid serializing every weight's raw samples on every step.

The sanitized CPU suite passes deterministic clock-drift/outlier cases, a noisy
17-candidate case, tie-margin behavior, single-candidate behavior and invalid
sample rejection. Both host build configurations and the complete ROCm Release
compile/link pass. GPU tests now additionally check separate same-shape weights,
cache hits, reupload invalidation, row-scaling keys, policy invalidation, compact
report counts, and exact auto/selected-forced output equality. Those GPU checks
and the revised benchmark have not executed in this environment. Device kernels,
precision, reserved scratch and KV allocation policy are unchanged by this update.

## JSON equality regression reported after 452f100

The user reports the one-GPU suite failed three times at "Cache hit unexpectedly
retuned the weight". Before/after selection dumps were identical: untimed fields
contained in-memory NaNs, which nlohmann JSON prints as null but compares unequal
to themselves. This failure did not demonstrate retuning. The subsequent
auto/selected-forced comparison and remaining GPU checks did not execute.

Every untimed candidate and legacy timing field now exports an actual JSON null.
The structural equality assertion is retained. A host regression covers nested
untimed fields, finite values, self/copy equality, serialization roundtrips and
nonfinite conversion. The sanitized host suite passes 78,341 assertions, and the
full ROCm Release build passes. The corrected GPU suite still needs rerunning.

## User-reported performance after 452f100

Conservative one-row, group-major benchmarking reports auto within 1% of the
fastest forced mode for every projection:

| Projection | Auto ms | Fastest forced ms | Mode |
| --- | ---: | ---: | --- |
| Q | 0.165 | 0.164 | pipelined-64 |
| K | 0.038 | 0.038 | pipelined-128 |
| V | 0.038 | 0.038 | pipelined-128 |
| O | 0.163 | 0.163 | pipelined-64 |
| Gate | 0.355 | 0.356 | pipelined-128 |
| Up | 0.354 | 0.354 | pipelined-128 |
| Down | 0.357 | 0.355 | pipelined-128 |
| Sum/block | 1.470 | | |

The isolated estimate is about 129 ms over 88 layers, but the profiled server
reports decode at 171 ms/token (p95 174 ms) and TTFT at 199 ms. Projection totals
are 133.7 / 131.8 ms; collectives are 14.9 / 16.8 ms. This is not an end-to-end
gain over the preceding 167 ms result. The user observes mostly pipelined-64 in
the cached selection summary, including duplicate shapes at 12 rows.

The compact summary groups all cached batch sizes by shape/scaling/mode; 12-row
entries do not establish which kernels one-row decode used, and duplicate groups
do not uniquely identify K/V or gate/up. The isolated benchmark uses layer 0,
rank 0 and one GPU, while serving uses all layers and both rank-local code tensors.
The supplied data does not isolate active-row selection, cold versus warmed
weights/cache state, GPU clocks/concurrency, or profiling overhead as the cause
of the remaining server/benchmark difference.

## Current diagnostics update

The compact selection summary is cached and invalidated on choice creation,
completed tuning, search-policy changes and forced-mode changes. Previously,
profiling scanned/grouped all weight choices every step. GPU regression checks
cover counts and invalidation; no measured overhead reduction is claimed.

New `/metrics/tuning` and `/metrics/tuning/all` endpoints expose named weight
choices for the last completed step (layer 0 plus head, or all layers plus head).
They use the actual query/sample row counts and TP scaling flags, include both
ranks and raw calibration samples, and omit dormant batch-size choices. They do
not retune or launch kernels and also work with stage profiling disabled. The
snapshot holds the model execution lock and should be fetched outside timed
generation. It reports execution and profile-collection times separately when
profiling is enabled.

Full-model tests now exercise a two-row prefill followed by one-row decode and
check that the snapshot follows the active rows without including old prefill
choices. HTTP tests check named snapshot availability and row filtering. These
GPU/live-server checks are compiled but remain pending on the target machine.
This diagnostics update does not change the tuning tournament or device kernels.

## Still requires the target machine

Persistent-worker reuse, exception propagation, and draining are covered in the
host suite. Expanded GPU projection tests force fused, split-K, tiled, tiled-LDS
and pipelined paths with both layouts against independent scalar references,
including nonaligned tails, dense weights and unscaled row-parallel outputs.
Every registered projection mode is exercised. N=64/128 cover aligned
specialization; K=2048 covers repeated tile prefetches and split partitions.
Additional K=4096 / N=256 and K=4104 / N=129 cases at rows 1/7 cover multiple
2048-input partitions, multiple 128-column tiles, and simultaneous K/N tails,
with both layouts, dense/compressed weights and scaled/unscaled outputs.
Sentinel guard regions
surround projection outputs; outputs are poisoned before every comparison so a
missing store cannot inherit an earlier path's correct output. Test codes
explicitly include 0 and 65,535.
The real-checkpoint projection benchmark and server profile report hardware
timings; no throughput improvement is assumed.

- Expanded unmodified GPU suite, including selection-cache regression checks.
- Real KV reuse under concurrent serving and optional RCCL-SHM transport.
- Complete-block comparison against the independent scalar C++ implementation.
- Full-model finite logits, repeated execution, generation and cache reclamation.
- Live HTTP streaming, concurrency, stop strings and disconnect cancellation.
- Memory-pressure serving and recomputation with actual device allocations.
- Numerical/quality differences from the source AQLM model.
- Throughput, latency, memory use, dispatch tuning and useful concurrency limits.

Run the commands in [README.md](README.md), or start with
`bash cpp_runtime/scripts/validate.sh BUILD_DIRECTORY MODEL_DIRECTORY`.
The script fails on a missing GPU or a numerical failure; it never records skipped
hardware validation as a pass. Run `behemoth_http_tests` after starting the server.

The independent CPU attention reference uses higher-precision scalar arithmetic
to expose indexing/scaling mistakes. GPU tolerances account for FP32 accumulation
and BF16 output rounding; separate quantization diagnostics must not be used to
justify an implementation mismatch.

The developer has not independently reproduced the user-reported hardware
results. The end-to-end timings above are user-reported; no full-model quality
result or measured performance for the newest candidates is claimed.
