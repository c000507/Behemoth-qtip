# Shared CUDA inference: Behemoth QTIP and GLM-5.3-Flash

`behemoth_cuda` and `behemoth_cuda_server` select the architecture from the
checkpoint's `config.json`:

| Model profile | CUDA backend |
|---|---|
| Llama/Mistral-layout Behemoth-R1-123B-v2 QTIP | 88 layers, hidden 12288, GQA 96/8, RoPE, QTIP HYB and optional FP8 K/V projections |
| `glm5_next` / `glm5_next_text` | Original 320B MoE or custom dense config, KDA, NoPE sparse MLA, mHC, FP8/BF16 and NVFP4 MLPs |

The Llama/Mistral QTIP profile currently uses the supported Behemoth geometry
and HYB L16/K2/V2 codebook; other Llama sizes and other QTIP codebooks are rejected.
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
It resets sequence state between requests. Tool calls, structured outputs,
batching, and tensor parallelism are not implemented in this CUDA server. It
binds to loopback by default; use SSH forwarding to access it remotely.

## Execution and validation limits

For Behemoth, compressed QTIP and FP8 projection weights remain resident on the
GPU. QTIP GEMV fuses the tail-biting trellis decoder with matrix-vector products;
the input/output Hadamard transforms preserve the SU/SV conventions of the CPU
reference. FP8 K/V scales apply after FP32 accumulation. Activations and the
RoPE GQA KV cache are BF16. This cache differs from the original ROCm runtime's
FP8 cache. The 123B public mixed checkpoint occupies about 33.3 GiB with a 4k
context allocation; a 32k cache adds about 9.6 GiB.

GLM uses the [shared GLM kernels](../../glm53/README.md), with GPU attention and
recurrent state and streamed large weights. Both backends currently replay
prefill one token at a time and run one request on one GPU.

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
be presented as a tokens-per-second improvement. GPU correctness and performance
of split-K are pending; compilation is not an execution check.

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

Detailed hardware, checkpoint revision, and measurements are in
[VALIDATION.md](VALIDATION.md).

These checks establish the tested paths, not full-model quality or absence of
repetition. Turing and R9700 execution still need checks on their respective
hardware; A100 execution does not establish identical behavior on those devices.
