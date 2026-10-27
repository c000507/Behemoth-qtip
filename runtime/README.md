# Behemoth C++ ROCm runtime

For the new opt-in BF16 prefill-transport experiment, benchmark correction,
validation status and remaining bottlenecks, see
[PREFILL_INVESTIGATION.md](PREFILL_INVESTIGATION.md). It is disabled by default;
`--qtip-bf16-prefill-rows 512` enables it for qualifying pure-prefill micro-batches
and adds rounding that needs hardware/quality evaluation.

Model-specific C++20 implementation for **two Radeon AI PRO R9700 (`gfx1201`) GPUs**.
It retains AQLM 1x16/8-weight compression, uses native OCP E4M3 matrix operands and
KV storage, BF16 residuals/activations/embedding and norm weights, and FP32
accumulation, scales, normalization, softmax, collective reductions, and logits.
There is no FP16 compute or precision fallback. F16 checkpoint bytes are decoded
as a file format, never instantiated as runtime FP16 tensors.

**The best user-reported decode is 167 ms/token (about 6 tokens/s) after `d434962`**,
versus 178 ms after `7a88e6d` and the earlier 307 ms baseline. That run's time to
first token was 191 ms.
The latest `452f100` report shows improved isolated selection (1.47 ms/block),
but server decode is 171 ms/token with profiling enabled; the bench gain has not
carried over. Its GPU suite stopped at a JSON NaN equality bug, fixed in
`cd84683`; the remaining checks still require rerunning. This update adds named,
last-step tuning snapshots and caches profiling summaries. See
[VALIDATION.md](VALIDATION.md) for the exact hardware-validation status.

## Build on the R9700 machine

Use Linux with matching ROCm **7.2.1 or newer** HIP, rocWMMA, hipBLASLt, and RCCL
development packages. The compile check uses ROCm 7.2.1 / clang 22 / rocWMMA 2.2.
Also install a C++20 compiler, CMake >=3.25, Ninja, OpenMP development support, and
Boost >=1.74 headers (Ubuntu: `build-essential cmake ninja-build libboost-dev`).
CMake fetches nlohmann JSON 3.11.3 and SentencePiece 0.2.1. Runtime and tests need
no Python, PyTorch, Rust, or Triton.

From the repository root:

```sh
cmake -S cpp_runtime -B cpp_runtime/build-rocm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBEHEMOTH_ROCM=ON -DBEHEMOTH_SERVER=ON -DBEHEMOTH_TOKENIZER=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build cpp_runtime/build-rocm -j 4
```

The matrix kernels force `-mno-wavefrontsize64`, assert rocWMMA's Wave32 mode,
and reject other GPU architectures. ROCm versions that define
`__AMDGCN_WAVEFRONT_SIZE` also receive an explicit `==32` preprocessor check.
ROCm 7.2.1 clang does not expose that macro, so the compiler flag, rocWMMA
constant, device-property check, and actual matrix-instruction smoke test enforce
the same requirement. OCP types are `__hip_fp8_e4m3`, `rocwmma::float8_t`, and
`HIP_R_8F_E4M3`; no FNUZ operations are selected.

The runtime converter explicitly saturates signed infinities to +/-448 and preserves
signed NaNs. The native gfx1201 constructor maps infinities to NaN, so the wrapper
handles those special values before using native conversion for finite inputs.
Host and GPU tests retain infinities, signed NaNs, and maximum finite FP32 values.

`roc::rocwmma` is used to locate headers but not linked as a public interface:
that package exports `-x hip`, which would otherwise break host C++ compilation.

## Download the checkpoint

Use an existing local `output` checkpoint, or download the pinned revision with
the shell script (curl only):

```sh
bash cpp_runtime/scripts/download_model.sh /models/behemoth-aqlm
```

Pinned model revision: `61a69e73fcdbcd8a0b81237c0092aa93518a5c52`.
The seven shards contain 32,728,473,600 tensor bytes. Allow at least 64 GiB of
storage and preferably 128 GiB of host RAM. The loader memory-maps shards, checks
offsets/overlap/shapes, and converts only the tensors being loaded.

## Run correctness tests, in order

```sh
# Host reference, metadata, scheduler, checkpoint and streaming utilities.
ctest --test-dir cpp_runtime/build-rocm -R '^core_correctness$' --output-on-failure

# One GPU: native FP8 conversion/matrix smoke, attention against independent
# references, KV append, projection, masking, page boundaries, split/merge,
# stream isolation and guarded output buffers. No model download required.
cpp_runtime/build-rocm/behemoth_gpu_tests

# Two GPUs: PCIe/BAR/IOMMU diagnostics and pinned-host reduction/gather.
# No peer access is enabled or required. No model download required.
cpp_runtime/build-rocm/behemoth_gpu_tests --two-gpu

cpp_runtime/build-rocm/behemoth_tokenizer_tests /models/behemoth-aqlm
cpp_runtime/build-rocm/behemoth_inspect /models/behemoth-aqlm

# Loads the full model, executes one chosen block on both GPUs, and compares
# against an independent scalar C++ reference. CPU reference is intentionally
# slow: about 1.38 billion MACs for the chosen block.
cpp_runtime/build-rocm/behemoth_block_tests /models/behemoth-aqlm 0

# Full-model finite logits, repeated execution, decode, and cache reclamation.
cpp_runtime/build-rocm/behemoth_model_tests /models/behemoth-aqlm
```

Or run the same sequence with:

```sh
bash cpp_runtime/scripts/validate.sh cpp_runtime/build-rocm /models/behemoth-aqlm
```

Randomized GPU comparisons must meet both `RMS <= 0.001 + 0.01*reference_RMS`
and `abs(error) <= 0.01 + 0.02*abs(reference)` per element. Known-answer cases
require exact results. Reference probability quantization uses the same tile and
partition boundaries. Differences from attention without probability quantization
are reported separately and must not be confused with implementation errors.

The GPU runner fails if hardware is absent; it does not silently skip GPU tests.
Do not change tolerances merely to make a failure pass. Save the complete failure
output, ROCm version, GPU architecture, and test seed for investigation.

## Serve

```sh
cpp_runtime/build-rocm/behemoth_server \
  --model /models/behemoth-aqlm --devices 0,1 --transport host \
  --memory-fraction 0.9 --token-budget 1024 --prefill-chunk 256 \
  --host 127.0.0.1 --port 8080 --threads 4
```

Weights and fixed workspace are allocated before computing the common KV-page
budget from both devices. The page allocator includes FP32 K/V scale metadata.
Each cached token costs **92,928 bytes per GPU** across 88 layers (90.75 KiB).
Page IDs match across ranks, while each rank stores four of the eight KV heads.

There is no fixed active-request count. Continuous batching prioritizes decode,
limits work per iteration, and chunks prefills. KV pages grow on demand. Under
pressure, newer unpinned requests are paused and later recomputed from saved
tokens. Sampling state is preserved and previously emitted output is not repeated.
The oldest request is allowed to make progress when prefills fill the pool.
Requests that cannot fit alone or exceed 131,072 total tokens are rejected or
ended at the available context limit.

Endpoints:

- `POST /v1/chat/completions`: text messages, streaming/nonstreaming, temperature
  (default 0), top-k/top-p, seed, stop strings, and output-token limits.
- `GET /v1/models`, `GET /health`, `GET /metrics`.
- `GET /metrics/tuning`: last-step choices and calibration samples for layer 0
  and the LM head on both ranks; `/metrics/tuning/all` includes all 88 layers.

Tools, multimodal messages, multiple completions, and unsupported logit processors
are rejected. The exact checkpoint chat template and legacy SentencePiece behavior
are implemented in C++. Streaming handles partial stop strings, incomplete UTF-8,
disconnect cancellation, and bounded network write queues.

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cgg507/Behemoth-R1-123B-v2-aqlm","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'

# Run while the actual GPU server is listening, in another terminal:
cpp_runtime/build-rocm/behemoth_http_tests 127.0.0.1 8080
```

### Prefix cache

Enabled by default (`--prefix-cache-slots 10`, `--prefix-cache-gb 64`, `--prefix-cache-min-tokens 64`,
`--prefix-cache-gpu true`, `--prefix-cache-staging-pages 32`, `--prefix-cache-restore-pace 0`;
`--prefix-cache-slots 0` disables it). A prompt
whose token ids start with a cached sequence restores that KV (from pinned host memory, or by taking over the
pages a recent sequence left on the GPUs) and prefills only the remainder; finished and cancelled requests are
cached. Design, tests and measurements: [PREFIX_CACHE_REPORT.md](PREFIX_CACHE_REPORT.md). Tests:
`behemoth_model_tests MODEL prefix` (save/restore round trip, logits vs cold prefill).
Against a running server, use the native `behemoth_http_tests HOST PORT` for HTTP coverage.
Benchmark shell scripts require `jq` for JSON processing; legacy scripting clients have been removed.

The HTTP test exercises protocol errors, deterministic streaming/nonstreaming
output, concurrent requests, stop strings, client disconnects, and page reclamation.
`/metrics` reports engine-side rolling TTFT/inter-token percentiles (last 1024
samples), active requests, free pages, preemptions, tokens, and GPU-step time.

## PCIe transport: P2P is assumed unavailable

`--transport host` is the default: device-to-pinned-host copies, FP32 CPU summation,
then host-to-device copies, with abortable rendezvous between rank workers. It
does not call `hipDeviceEnablePeerAccess` or use device-to-device copies.

`--transport rccl-shm` is optional. It forces `NCCL_P2P_DISABLE=1` and enables SHM.
Test that path separately before selecting it:

```sh
cpp_runtime/build-rocm/behemoth_gpu_tests --two-gpu-rccl
```

Diagnostics print PCI addresses, current link speed/width, the largest memory BAR,
whether it covers VRAM, and whether `iommu=pt`/`iommu.passthrough=1` is present.
BAR aperture coverage is an observation, not proof of Resizable BAR capability.
These settings are not prerequisites for the default host-staged transport.
Container shared-memory limits matter when opting into RCCL SHM.

## Benchmarks

### Diagnosing low decode throughput

Projection dispatch caches complete hipBLASLt plans and uses persistent rank
workers. Compressed weights retain 16-bit AQLM indices and a 512 KiB FP8 codebook
per projection (the serialized F16 table is 1 MiB). Logical index bytes divided
by runtime do not establish physical memory bandwidth or prove a cache bottleneck.

Indices now default to group-major order: neighboring output lanes read adjacent
indices. The loader transposes each local shard once, without retaining duplicate
GPU tensors or changing the checkpoint. `--code-layout row-major` restores the
previous layout for comparison. Both layouts work with every projection path.

The `tiled-64` and `tiled-128` candidates use four Wave32 waves per block,
64 output columns, packed eight-weight global loads and LDS stores, and a padded
column-major weight tile. Their K tiles are 64 and 128, with 1024-input split-K
partitions and deterministic FP32 merging. Padding avoids a power-of-two LDS
column stride; neither kernel stages the entire codebook in LDS. They use 6 KiB
and 11 KiB of static LDS.

Commit `d434962` added two controlled comparisons:

- `tiled-lds-64` / `tiled-lds-128` retain the existing tile computations and
  loads, but synchronize only LDS. HIP's full `__syncthreads()` emits global
  waits and read-cache invalidation; these kernels share only LDS between waves.
  The replacement uses local-memory release/acquire fences and a block barrier.
- `pipelined-64` / `pipelined-128` also batch independent index loads before
  codebook gathers and prefetch the next tile into registers before current-tile
  WMMA. Fully aligned compressed group-major projections omit dense/layout/tail
  branches and use bounded uint32 index addressing. Other shapes retain a
  generic implementation. LDS-only barriers allow the register loads to remain
  outstanding across synchronization. The compiler retains one K=32 fragment
  pair at a time to bound register pressure.

Those tile families use the same 1024-input split width, scales, FP8 products and
FP32 merge. No extra persistent GPU weight tensors or KV storage are added. The
pipelines use more registers than the original tiles; lookup parallelism and
occupancy remain competing considerations.

Commit `ce03e68` extended the pipelines with eight geometry candidates. Each
family below has K tiles of 64 and 128; the final suffix names that tile size.
Output columns and K partitions are varied separately, so timings can identify
which change helps each projection shape:

| Modes | Output columns/block | Waves/block | K partition |
| --- | ---: | ---: | ---: |
| `pipelined-64`, `pipelined-128` (proven decode defaults) | 64 | 4 | 1024 |
| `pipelined-n32-64`, `pipelined-n32-128` | 32 | 2 | 1024 |
| `pipelined-n128-64`, `pipelined-n128-128` | 128 | 8 | 1024 |
| `pipelined-s512-64`, `pipelined-s512-128` | 64 | 4 | 512 |
| `pipelined-s2048-64`, `pipelined-s2048-128` | 64 | 4 | 2048 |

Smaller output tiles create more blocks; larger tiles share activation staging
across more waves. Smaller K partitions create more independent work, at the cost
of additional partial results and merging; larger partitions do the reverse.
These are hardware tradeoffs, not assumed speedups. Split-K changes FP32
summation grouping; the independent reference comparisons retain their existing
tolerances. The smallest partition remains 512, so the preallocated partial
workspace and the KV budget do not increase.

Auto now defaults to `--projection-search conservative`: model-sized compressed
decode shapes compare `pipelined-64` and `pipelined-128`. Small generic shapes
also compare fused, and batches of 4..16 tokens may compare available BLAS.
Dense projections and larger batches compare BLAS/fused. All forced modes remain
available. `--projection-search exhaustive` opts back into the full kernel set
for batches of up to 16 tokens, including the geometry variants above.

BLAS descriptors/algorithms remain shared by shape. Choices for up to 16 tokens
are keyed by weight identity, dimensions, layout, batch rows and row-scaling
policy: gate/up, K/V and different layers no longer inherit one same-shape
decision. Reuploading through `GpuProjection::upload` changes its identity.
Larger-prefill choices retain shape caching to bound startup cost. Changing the
search policy clears choices and retains BLAS plans.

Each candidate warms before timing. Calibration chooses 1..32 repeated full
calls per sample, targeting at least 0.5 ms where that cap permits. Seven
rotated forward/reverse rounds feed median measurements. The two finalists are
then measured in three A-B-B-A cycles, with a 2% minimum confirmed gain in at
least two cycles to replace the earlier finalist. Autotuning now includes input
quantization and reconstruction, matching benchmark timing scope. These steps
reduce susceptibility to clock drift and occasional scheduling stalls; they do
not establish a speedup without hardware measurements.

**Cold tuning costs more:** a new small-batch size can tune hundreds of separate
weight selections and take seconds before caching completes. Warm every batch
size used in a throughput test. Exclude first-use tuning from warmed decode and
TTFT results, and report cold latency separately. Cache hits do not retime kernels.

First rebuild and rerun `behemoth_gpu_tests` and the block/model tests. Projection
tests explicitly exercise both index layouts, every tile family, fused and
split-K kernels, cached auto dispatch, dense weights, row-parallel scaling,
guarded outputs, and tails at K=1032 / N=37/65 against independent scalar
references. N=64/128 exercise the aligned specializations; N=65 exercises partial
output tiles. K=2048 exercises prefetch across multiple tiles and split
partitions. Additional cases at K=4096 / N=256 and
K=4104 / N=129 exercise multiple 2048-input partitions, multiple 128-column
tiles, and simultaneous K/N tails. Every registered mode is tested. They also
check FP8 hipBLASLt where an algorithm is available.
Selection tests additionally cover same-shape weight isolation, reuploads,
scaled/unscaled keys, policy changes, cache hits, and exact agreement between
auto output and its selected forced kernel.

For a diagnostic run, append these options to the existing server command:

```sh
--projection-mode auto --profile true
```

Generate enough tokens to get past first-use autotuning, then collect:

```sh
curl -s http://127.0.0.1:8080/metrics > profile.json
```

`last_step_profile` contains per-rank GPU event totals for projections, attention
and KV operations, elementwise work, vocabulary projection, and collectives. It
also contains actual context lengths, batch size, selected projection methods,
tuning summaries, and kernel register/LDS/local-memory attributes.
Resource entries also include HIP's theoretical maximum active blocks/waves per
multiprocessor for each kernel and block size. These are resource limits, not
measured occupancy or a prediction that a small grid fills the GPU.
The server groups `projection_dispatch` by shape, row-scaling policy and selected
mode. `selection_count` and `selected_ms_min/max` summarize cached choices without
rebuilding every weight's samples on each token. The summary is now cached until
a selection, search policy or forced-mode override changes. This removes the
previous per-token scan/grouping of hundreds of entries when profiling is enabled.
These groups include **all cached batch sizes**: a `rows=12` entry may describe
an earlier prefill rather than one-row decode. Duplicate dimensions can represent
different selected modes and do not uniquely identify K/V or gate/up.
The standalone benchmark exports
detailed per-weight decisions: candidate medians, raw round/confirmation samples,
batch repeat counts, geometry, search policy and confirmation gains. Untested
candidates have null timing. Its previous timing fields remain available, but
their scope now includes quantization as indicated in the report.
`execution_wall_ms` measures model execution and logit download;
`profile_collection_ms` measures diagnostic event/JSON collection; `wall_ms`
includes both. Collective timings include host rendezvous waits. Events add
measurement overhead during execution too, so disable profiling again for final
throughput measurements. First-use tuning steps must be excluded.

To inspect the server's actual current choices, let a warmed request finish,
then fetch:

```sh
curl -s http://127.0.0.1:8080/metrics/tuning > server-tuning.json
# Optional full-model distribution of choices:
curl -s http://127.0.0.1:8080/metrics/tuning/all > server-tuning-all.json
```

These snapshots label the layer and projection, rank/device, current query/sample
row counts, scaling policy, selected mode, raw candidate samples and confirmation
gains. They select the choices used by the **last completed step**, excluding
dormant batch sizes. Samples are cached first-use calibration, not fresh per-step
timings. The snapshot works with `--profile false`, and includes the latest wall /
execution / profile-collection times when profiling is enabled. Reading it takes
the model execution lock; fetch after generation completes, outside latency tests.
For `cache_scope=shape`, calibration samples may originate from another weight
that uses the same shared prefill choice.

Compare layer 0, rank 0, row 1 with the default one-row benchmark. The benchmark
measures one GPU and one layer's weights with repeated self-warming. The server
runs both ranks and streams every layer's weights; its rank-local code indices,
cache/clock state and workload differ. A single-layer timing multiplied by 88
does not establish end-to-end improvement. Measure warmed serving with
`--profile false` separately from the profiling run.

Compare native paths on the real checkpoint with one GPU, without loading the
full model or requiring tensor-parallel communication:

```sh
# Model directory, batch-token rows, layer, device, layout, auto-search policy.
# Repeat for rows 1/2/4/8/16/32 and both GPUs.
cpp_runtime/build-rocm/behemoth_projection_bench /models/behemoth-aqlm 1 0 0 group-major conservative \
  > projection-bench.json

# Explicitly compare the wider search using the same benchmark protocol.
cpp_runtime/build-rocm/behemoth_projection_bench /models/behemoth-aqlm 1 0 0 group-major exhaustive \
  > projection-bench-exhaustive.json
```

The tool reports per-projection timings for fused, split-K, all three tile
families, hipBLASLt and auto, regardless of auto-search policy. Both layouts can
be requested with `both`. Every forced path warms before auto tunes. Timings use
adaptive repeated batches in seven interleaved rounds, with a self-warm call
before each sample. They include input quantization and reconstruction where
applicable. The row-scaling flag matches each projection's TP role, including
unscaled O/down partials. Logical code GB/s is tensor bytes/time, not
measured HBM traffic. It compares every output with the first layout's fused
path; independent scalar validation remains in `behemoth_gpu_tests`. An
unavailable forced BLAS path is reported explicitly.

The auto record includes `selected_mode`, its full `auto_tuning` decision, and a
`same_kernel_comparison` against that exact forced mode in paired A-B-B-A order.
`relative_auto_overhead` helps distinguish dispatch overhead or timing drift from
an actual choice between different kernels. A small auto/forced difference is
not evidence of a faster kernel; compare the same-kernel samples and their spread.

`--projection-mode` accepts `auto`, `fused`, `split-k`, `blas`, `tiled-64`,
`tiled-128`, `tiled-lds-64`, `tiled-lds-128`, `pipelined-64`, or `pipelined-128`.
It also accepts each geometry mode listed in the table above.
Forced split-K/tiled/pipelined modes reserve more workspace for large prefill
batches; auto caps these candidates at 16 rows and reserves bounded scratch
**before** sizing the KV pool. Forced BLAS fails if a shape has no native FP8
algorithm. All paths retain FP8 operands, BF16 residuals, FP32 accumulation, and
the no-P2P host transport.

```sh
# Context, requests, query tokens per request. Last argument 1 measures decode.
cpp_runtime/build-rocm/behemoth_bench 4096 1 1
cpp_runtime/build-rocm/behemoth_bench 4096 8 1
cpp_runtime/build-rocm/behemoth_bench 8192 1 256
cpp_runtime/build-rocm/behemoth_bench 32768 1 1

# Live server: prompt file, concurrency, requests, maximum output tokens.
cpp_runtime/build-rocm/behemoth_load_bench prompt.txt 8 32 128
```

The kernel benchmark reports GPU/HIP versions, time, workspace and a numerical
comparison against its scalar reference. The load benchmark reports actual prompt
token counts, aggregate completion throughput, request-latency percentiles and
server rolling metrics. Use prompt files of appropriate token lengths to measure
1K/4K/8K/32K workloads. Sweep concurrency until the KV budget becomes limiting.

The initial attention implementation uses one Wave32 workgroup per query tile/head
and gathers 32 keys from two pages. Long decode contexts can split into 1024-key
partitions with FP32 merging. There is no full score-matrix allocation. Further
GQA reuse, tile tuning, and performance claims require measurements on the cards.

## CPU-only development

```sh
cmake -S cpp_runtime -B cpp_runtime/build-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBEHEMOTH_SANITIZE=ON \
  -DBEHEMOTH_TOKENIZER=ON -DBEHEMOTH_HTTP_TOOLS=ON
cmake --build cpp_runtime/build-host -j 4
ctest --test-dir cpp_runtime/build-host --output-on-failure
cpp_runtime/build-host/behemoth_tokenizer_tests /models/behemoth-aqlm
```

For the dependency-minimal core tests, leave the tokenizer and HTTP options OFF.
The CPU reference benchmark is explicitly a reference/quantization diagnostic,
not a GPU-performance or GPU-correctness result.
