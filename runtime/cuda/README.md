# Shared CUDA inference: Behemoth QTIP and GLM-5.3-Flash

`behemoth_cuda` and `behemoth_cuda_server` select the architecture from the
checkpoint's `config.json`:

| Model profile | CUDA backend |
|---|---|
| Llama/Mistral-layout Behemoth-R1-123B-v2 QTIP | 88 layers, hidden 12288, GQA 96/8, RoPE, QTIP HYB and optional FP8 K/V projections |
| Behemoth adaptive mixed-format (`qtip_quant --mode adaptive`) | same geometry; per-layer QTIP or NVFP4 (E2M1 + E4M3 block scales) projections, selected per linear from the checkpoint tensors |
| Llama 3.3 70B Instruct QTIP (`qtip_quant` output) | 80 layers, hidden 8192, GQA 64/8, `head_dim` 128, FFN 28672, vocab 128256, rope_theta 5e5 with `llama3` rope_scaling, byte-level BPE tokenizer, stop on the `generation_config.json` eos list (`<|end_of_text|>`, `<|eot_id|>`, ...); single sequence, one GPU (no TP2/batching) |
| `glm5_next` / `glm5_next_text` | Original 320B MoE or custom dense config, KDA, NoPE sparse MLA, mHC, FP8/BF16 and NVFP4 MLPs |

The Llama/Mistral QTIP profile reads its geometry from `config.json` (Behemoth, Llama 3.3 70B or any
llama-layout shape whose dimensions are `2^p`, `12*2^p` or `28*2^p` multiples of 32 with `2^p <= 8192`),
supports `rope_scaling` type `llama3` only, and requires the HYB L16/K2/V2 codebook; other codebooks and
unsupported shapes are rejected. The batched / TP2 fast profile (`BEHEMOTH_CUDA_FAST`) is Behemoth-only and
refuses other geometries. A 70B 2-bit checkpoint needs more than 16 GB of VRAM (trellis ~17.5 GB + bf16
embeddings/head ~4.2 GB).
Llama 3.3 verification: `behemoth_cuda_llama_tests` (llama3 frequency table and RoPE kernel vs a fp64 HF
transcription; synthetic Llama-geometry-scaled QTIP, QTIP+NVFP4 and plain-RoPE checkpoints end to end vs a CPU
reference; Llama 3 tokenizer/chat template with a synthetic tokenizer; config rejection) and the 8192/28672
projection shapes in `behemoth_cuda_tests`. Llama checkpoints written with `qtip_quant --kv-format fp8` from a BF16 source (K/V as F8_E4M3 `weight` + BF16 `weight_scale`, `qtip_quant.kv_format = "fp8"`, in QTIP and NVFP4 layers) load through the same per-linear FP8 W8A16 path as Behemoth and are covered by a synthetic CUDA-vs-CPU case in `behemoth_cuda_llama_tests`. Not verified: real Llama weights/tokenizer, a full-size run.
GLM dimensions and dense/MoE layer types come from its configuration. Tokenizer
selection follows the tokenizer assets: SentencePiece for Behemoth, or byte-level
BPE for GLM and Llama-3-style assets. The checkpoint's actual Jinja template is
used for chat.

## Build

Requires C++20, CMake 3.25+, a CUDA toolkit supporting the selected GPU, PCRE2,
Boost headers, and optionally OpenMP. SentencePiece is fetched during configure.
For an A100:

```sh
cmake -S runtime -B runtime/build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBEHEMOTH_CUDA=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build runtime/build-cuda -j8
OMP_NUM_THREADS=8 ctest --test-dir runtime/build-cuda --output-on-failure
```

Use architecture `75` for RTX 2080 Ti / Turing. CUDA 12.4 was used for the A100
execution checks. CUDA 12.9 also compiles the QTIP kernels for sm_75. The kernels
use integer BF16 storage and FP32 arithmetic, without requiring BF16 or FP8
tensor-core instructions. Configure ROCm and CUDA in separate build directories.

The optional Ampere fast backend adds NCCL TP2, batched decode, cuBLAS prefill,
and cuDNN causal GQA. For two RTX A6000s, configure with:

```sh
cmake -S runtime -B runtime/build-cuda-fast -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBEHEMOTH_CUDA=ON -DBEHEMOTH_CUDA_FAST=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNCCL_ROOT=/path/to/nccl -DCUDNN_ROOT=/path/to/cudnn
cmake --build runtime/build-cuda-fast -j8
OMP_NUM_THREADS=8 ctest --test-dir runtime/build-cuda-fast --output-on-failure
```

NCCL/cuDNN roots contain `include` and `lib`; NVIDIA Python wheel directories
are also supported. The build fetches pinned cuDNN frontend headers. This path
currently requires the Behemoth geometry and FP8 K/V projection weights. GLM
continues to use its existing single-GPU backend. Bulk fast prefill requires
sm_80 or newer; use the default build for the Turing single-GPU path.

## Generate or serve either architecture

```sh
runtime/build-cuda/behemoth_cuda --model /models/Behemoth-R1-123B-v2-QTIP-2bit \
  --prompt 'What is 2 + 2? Answer briefly.' --max-tokens 64 --context 32768

runtime/build-cuda/behemoth_cuda_server --model /models/Behemoth-R1-123B-v2-QTIP-2bit \
  --device 0 --context 32768 --name behemoth-qtip --host 127.0.0.1 --port 1337

# The same executables select GLM automatically:
runtime/build-cuda/behemoth_cuda_server --model /models/glm53-nvfp4 \
  --device 0 --context 32768 --name glm53 --host 127.0.0.1 --port 1337
```

The CLI accepts a user `--prompt`, a `--messages` JSON array, or exact `--tokens`
JSON. `--trace FILE` records teacher-forced logits at the token request's
`logit_positions` (defaults to the last position). A trace-only invocation can
set `--max-tokens 0`.

The HTTP server provides health/models/chat completions, stop strings, seeded
temperature/top-k/top-p sampling, usage, SSE streaming, and `reasoning_content`.
It resets sequence state between requests. The fast build accepts `--tp 2` for
Behemoth and uses bulk prefill. Requests are still serialized; tool calls,
structured outputs, and a continuous batching scheduler are not implemented. It
binds to loopback by default; use SSH forwarding to access it remotely.

## Execution and validation limits

For Behemoth, compressed QTIP and FP8 projection weights remain resident on the
GPU. QTIP GEMV fuses the tail-biting trellis decoder with matrix-vector products;
the input/output Hadamard transforms preserve the SU/SV conventions of the CPU
reference. FP8 K/V scales apply after FP32 accumulation. Mixed-format layers
written by `qtip_quant --mode adaptive` run NVFP4 projections through the shared
GLM gemv (register-widened E2M1 with bf16 rounding); the format is probed per
linear from the layer tensors. Activations and the
RoPE GQA KV cache are BF16. This cache differs from the original ROCm runtime's
FP8 cache. The 123B public mixed checkpoint occupies about 33.3 GiB with a 4k
context allocation; a 32k cache adds about 9.6 GiB. An adaptive checkpoint with
N NVFP4 layers adds roughly 1.3 GiB per NVFP4 layer over the all-QTIP size.

GLM uses the [shared GLM kernels](../../glm53/README.md), with GPU attention and
recurrent state and streamed large weights. Its prefill replays tokens in order.

The fast Behemoth backend shards QTIP rows in the rotated basis and KV heads
between GPUs. It reconstructs the global output Hadamard transform after each
projection. FP8 K/V scales apply after FP32 accumulation. Decode uses CUDA graphs
and tiled GQA; prefill decodes one projection at a time into a BF16 cuBLAS
workspace and uses cuDNN attention. Batch-8 prefill groups whole sequences to
limit live activation memory. Activations and KV remain BF16.

The native model API exposes `prefill_batch` and `step_batch` for batches up to
eight. Measure actual TP2 execution, separately timing prefill and decode, with:

```sh
runtime/build-cuda-fast/behemoth_cuda_bench \
  --model /models/Behemoth-R1-123B-v2-QTIP-2bit --tp 2 --batch 8 \
  --prompt-tokens 8192 --decode-tokens 256 --warmup 1 --output batch8.json
```

Use `--batch 1` for single-sequence decode. The benchmark uses fixed lengths,
ignores EOS, and reports aggregate batch throughput. Its batched model execution
does not measure HTTP scheduler throughput. Set `BEHEMOTH_CUDA_GRAPHS=0` for an
uncaptured comparison, or `BEHEMOTH_CUDA_PROFILE_PREFILL=profile.json` to record
projection phase timings for the first prefill group.
`BEHEMOTH_CUDA_PROFILE_DECODE=decode.json` records stage timings for the first
uncaptured decode step. Event instrumentation adds overhead; use it to locate
bottlenecks and disable it for throughput measurements.
The benchmark also reports per-step latency summaries. Set
`BEHEMOTH_CUDA_BENCH_STEP_TIMES=1` to retain all step times in its JSON output.
Use `--trace-logits FILE` for numerical comparisons: it records logits after
bulk prefill and the final decode step, plus all generated token IDs. Trace
serialization happens outside the timed regions; warmup runs do not write it.

The fast build also provides a full-checkpoint regression for short prompts:

```sh
BEHEMOTH_CUDA_TP_TRANSPORT=host runtime/build-cuda-fast/behemoth_cuda_model_tests \
  --model /models/Behemoth-R1-123B-v2-QTIP-2bit
```

It compares a cold 32-token prefill and decode against the same request after a
1024-token request grows the arena. Logits must match bit for bit. This catches
short-prefill split-workspace errors and stale state across resets. Configure
`BEHEMOTH_CUDA_TEST_MODEL=/path/to/checkpoint` to include it in CTest; the checkpoint
is not downloaded by the test.

Small-batch QTIP decode uses 32 output rows per warp and reuses activation
fragments across two tensor-core products. It is enabled for batches 2–16.
Set `BEHEMOTH_CUDA_QTIP_MMA_WIDE=0` for the earlier kernel; `1` also selects
the tensor-core path for batch 1, which changes its FP32 summation order.

Decode attention uses register-level tensor-core products. Set
`BEHEMOTH_CUDA_ATTENTION_REGISTER=0` to compare the earlier WMMA staging path.
Both paths retain BF16 probability rounding and the same split-context reduction.
For contexts through 8448 tokens, decode softmax retains scores and exponentials
in registers, preserving the earlier reduction tree. Set
`BEHEMOTH_CUDA_RESIDENT_SOFTMAX=0` for an A/B comparison.

Prefill uses fused Hadamard transforms. Set `BEHEMOTH_CUDA_FUSED_HAD=0` for
the earlier path. The fused kernels keep each token's intermediate values on-chip for widths 12288 and 28672
and at least 256 rows. Unsupported geometry, insufficient shared memory, and
unsafe overlapping layouts use the earlier two-pass transform. The MLP up
projection uses a separate output buffer so its rank-major input remains intact.

NCCL defaults to shared-memory transport (`NCCL_P2P_DISABLE=1`), two channels,
and a 1 MiB buffer after a hosted A6000 pair showed a slow peer path. Explicit
NCCL environment settings override these defaults; measure on your topology.
On hosts with working CUDA peer access, set
`BEHEMOTH_CUDA_TP_TRANSPORT=peer` to use GPU store kernels for small AllGather
payloads and copy engines for large prefill transfers. `peer-dma` uses copy
engines for every transfer and provides an A/B comparison. The peer tests
check exact rank data, repeated buffer reuse, and cross-device graph replay.
The default remains `nccl` because peer-copy performance varies greatly by host.
Peer transport now checks small and bulk transfers against known byte patterns
at startup. Some hosted PCIe systems advertise peer access but return incorrect
data. Such a host is rejected before model loading; use `nccl` there. The fast
build adds `cuda_nccl_collectives` to test exact rank data and graph replay through
NCCL. Run `behemoth_cuda_peer_tests --dma` to isolate the peer DMA path.

`BEHEMOTH_CUDA_TP_TRANSPORT=host` uses copy engines for transfers larger than
1 MiB, pipelining D2H and H2D through two 16 MiB pinned slots per sender.
Smaller decode transfers stay on NCCL to avoid multi-device memcpy graph launch
overhead. This path requires no GPU peer access and reserves 64 MiB of host RAM.
`cuda_host_collectives` checks exact data, ring-slot reuse, odd tails and graph
replay; it can also be run with `behemoth_cuda_peer_tests --host`.

## Experimental QTIP split-K benchmark

The Behemoth trellis GEMV can split each input reduction across independent CTAs.
The partial FP32 sums are combined inside the output Hadamard transform, before
the MLP's optional BF16 rounding. There are no atomics or extra reduction launches.
The summation order changes, so this requires numerical and full-model validation.
The default remains one split; GLM and FP8 projections do not use this setting.

```sh
# Tests split counts 1/2/3/4/8/16/32 against the independent CPU reference:
OMP_NUM_THREADS=8 ctest --test-dir runtime/build-cuda -R cuda_qtip_correctness --output-on-failure

# Warm projection microbenchmark: GPU event times, five batches, median per call.
# Omit --model to use synthetic weights at the actual Behemoth projection sizes.
runtime/build-cuda/behemoth_cuda_qtip_bench \
  --model /models/Behemoth-R1-123B-v2-QTIP-2bit --iterations 20

# Opt in for full-model A/B checks using the same prompt, seed and context:
BEHEMOTH_CUDA_QTIP_SPLITS=16 runtime/build-cuda/behemoth_cuda \
  --model /models/Behemoth-R1-123B-v2-QTIP-2bit \
  --prompt 'Explain briefly why the sky is blue.' --context 32768 --max-tokens 128
```

The benchmark checks output differences against the unsplit path and reports
CSV timings for q, gate and down projections (covering the three matrix shapes).
It includes the input and output Hadamard transforms. Repeated isolated calls
have different cache behavior from whole-model decoding: these timings cannot
be presented as a tokens-per-second improvement. Split-K and batched kernels
have passed GPU reference checks on RTX A6000s. Changed FP32 summation order can
change greedy output; these checks do not establish full-model quality.

Attention-score scratch space is now reserved at model load to avoid device
allocation/free at each new sequence length. This adds `96 * context * 4` bytes
of persistent scratch, or 12 MiB at a 32k context.

## Validated baseline

On an A100 SXM 80GB, the following passed at commit `021c434`, before split-K:

- CPU core and structured-output regressions.
- CUDA trellis GEMV versus the independent double-precision QTIP reference,
  including Hadamard factors 1/12/28, transpose direction, and rotated-output rounding.
- FP8 output-channel scaling and causal GQA prefix/group mapping.
- Shared CLI dispatch for tiny GLM dense/MoE source and NVFP4 checkpoints, with
  logits identical to the standalone GLM CUDA path.
- Shared HTTP GLM dispatch and reset consistency.
- The standalone GLM CUDA kernel and 33-token model tests.
- Loading all 88 layers of `cgg507/Behemoth-R1-123B-v2-QTIP-2bit` and generating
  the correct answer `4` to the arithmetic smoke prompt above.
- Real-checkpoint HTTP generation and reset consistency: a 64-token explanation
  of Rayleigh scattering took about 18.8 seconds including 13 prompt tokens.

Additionally, on an RTX A4000 (sm_86, CUDA 12.0): the NVFP4 mixed-format
loader/gemv unit test (`behemoth_cuda_tests`, `nvfp4_test`) and the synthetic
adaptive-quantizer suite (`qtip_adaptive_tests`) pass.

Detailed hardware, checkpoint revision, and measurements are in
[VALIDATION.md](VALIDATION.md).

These checks establish the tested paths, not full-model quality or absence of
repetition. Turing and R9700 execution still need checks on their respective
hardware; A100 execution does not establish identical behavior on those devices.
