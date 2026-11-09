# GLM-5.3-Flash text backend

The [shared CUDA CLI/server](../runtime/cuda/README.md) also loads these GLM
models and Llama/Mistral-layout Behemoth QTIP checkpoints through the same entry
points. The standalone GLM build below remains available for HIP and CUDA.

This standalone C++ backend loads the original **320B MoE GLM-5.3-Flash** and the
supplied **80-layer dense conversion**. It implements their text architectures,
MLP-only NVFP4 conversion, incremental generation, logit traces, and a small HTTP
server. It does not use the Mistral-specific GQA engine in `runtime/`.

The optional CUDA and HIP backends execute KDA, sparse MLA, mHC, routing,
normalization, SwiGLU, and weight-decode GEMV on the GPU. Activations, KDA state,
compressed MLA latents, and pooled index keys stay on the device. Linear weights
are kept resident on their GPU up to a per-device budget and are otherwise
streamed from the mmap-backed host checkpoint for each projection. The budget is
derived from the free memory at load (leaving room for the attention caches at
the requested context, the decoded MLA expansions, constants and a 4 GiB margin);
`GLM53_RESIDENT_GIB=<GiB>` sets it explicitly (`0` streams everything, the
original behaviour). Weights become resident in execution order on the first
token, so for a MoE checkpoint that does not fit, the experts hit first stay and
the rest stream. Small constants and MLA expansion weights are retained on the
GPU. The CPU schedules layers and experts, samples tokens, and reads logits; MoE
routing transfers only the selected IDs and weights for expert loading.

### Several GPUs

`--tp N` (the standalone CLI with `--device 0`, and the shared `behemoth_cuda` /
`behemoth_cuda_server`) runs every layer on N consecutive GPUs with
Megatron-style **tensor parallelism**; `--layer-split N` instead keeps the earlier
**layer pipeline** (contiguous layer ranges per GPU, one 64 KB residual hand-off
per boundary, a shared indexer stays with its full indexer) for comparison. The
tensor split shards:

| Block | Sharded over ranks | Replicated | Exchange |
|---|---|---|---|
| Dense MLP, shared and routed experts | `gate_proj`/`up_proj` rows and `down_proj` columns (intermediate channels) | router (FP32, run on rank 0; expert IDs reach the host once per layer) | one FP32 all-reduce per MLP block; for MoE the shared and routed partials travel side by side in that one all-reduce, then each expert output is rounded to bf16 and weighted exactly as on one GPU |
| KDA | `q/k/v_proj`, `f_b/g_b_proj`, `b_proj` rows, conv taps, `dt_bias`, `A_log`, the recurrent and convolution state, `o_proj` columns (whole heads) | `f_a/g_a_proj` (128-wide low-rank inputs), `o_norm` | one all-reduce after `o_proj` |
| Sparse MLA | `q_b_proj` rows, the decoded `kv_b_proj` expansion, scores, `o_proj` columns (whole heads) | `q_a_proj`, `kv_a_proj_with_mqa`, their norms, the latent cache and the whole indexer (keys, gates, pools, top-k) | one all-reduce after `o_proj` |
| mHC, norms, residual streams | - | everything (the streams are 64 KB; mixing is 24 dot products) | - |
| `lm_head` | vocabulary rows (uneven shares allowed) | final norm | none: the host concatenates each rank's logits |
| Embeddings | - | host-side row lookup, uploaded to every rank | - |

The latent cache and indexer are replicated rather than sharded because the
latent is one 512-wide vector per token (0.6 GiB per rank at 32k context for the
70B) while sharding it would add an all-gather of the latent and a broadcast of
the top-k ids per MLA layer; replicated ranks compute identical selections since
their inputs are bitwise equal (the all-reduce result is identical on every rank)
and the kernels are deterministic. Partial sums are exchanged in FP32 and rounded
to bf16 once, so a tensor-parallel rank differs from the single-GPU path only by
the FP32 summation order inside the row-parallel projections. Heads and MLP
widths must divide by the rank count (the 70B and 320B geometries divide by 2, 4
and 8; NVFP4 column slices need 16-channel alignment, FP8 block scales
128-channel alignment). Collectives use NCCL when the build found it
(`GLM53_NCCL`, default ON) and otherwise a copy-engine all-reduce (peer or
host-staged copies into per-rank slots, summed in a fixed order);
`GLM53_TP_TRANSPORT=copy|nccl` overrides. `GLM53_TP_SAME_DEVICE=1` puts every
rank on `--device` as separate streams, which is how the tests exercise TP2 and
TP4 on one GPU (NCCL refuses a repeated device, so this always uses the copy
collective). Memory per GPU for the dense 70B at TP2 is about 76 GiB bf16 and
38 GiB with NVFP4 MLPs (2.8 GiB of that replicated), plus the latent cache
(0.6 GiB at 32k context) and scratch.

CUDA targets **RTX 2080 Ti / Turing (sm_75)** and newer NVIDIA devices. HIP targets
**R9700 / gfx1201**, using wave32. Fused GEMV decodes BF16/F16/F32, channel/block
FP8, or NVFP4 as it reads weights, with BF16 operand rounding and FP32 accumulation.
BF16 values are stored as integer bits and converted in registers, so no BF16,
FP8, or FP4 tensor-core instructions are needed on Turing. There is no FP16
substitution that would narrow BF16's exponent range. The standalone GPU build
requires the CUDA/HIP runtime, without a BLAS dependency.

Prefill replays tokens sequentially (one GEMV pass per token, no batched GEMM)
and requests are serialized. When the budget does not cover the weights,
host-to-device transfers dominate: the bf16 dense 70B streamed entirely ran at
3.2 s/token on an RTX PRO 6000 Blackwell. GLM-5.3-Flash (320B MoE) performance
and quality have not been measured on a real checkpoint.

| Configuration | Original MoE | Supplied dense |
|---|---:|---:|
| Hidden size | 4,096 | 8,192 |
| Layers | 45 | 80 |
| KDA / sparse MLA layers | 34 / 11 | 60 / 20 |
| Dense / MoE MLP layers | 3 / 42 | 80 / 0 |
| Query low-rank dimension | 1,536 | 4,096 |
| Vocabulary / context | 154,880 / 1,048,576 | 128,256 / 131,072 |

Dimensions and layer types come from `text_config`; stale MoE fields do not turn
dense layers into experts. Tensor prefixes `model.language_model.` and `model.`
are supported. The source tensor names otherwise follow the original GLM
checkpoint. The supplied conversion's initialization note says its new attention
needs distillation or continued pretraining; loading it does not establish its
quality.

Attention supports these checkpoints' NoPE MLA and KDA layout. Unsupported RoPE
MLA and mismatched expanded KV/query-head configurations are rejected.

## Build

Dependencies: CMake 3.25+, C++20, PCRE2 development files, and optionally OpenMP.
The server also needs Boost headers and pthreads. Python is only used by optional
validation scripts, not inference or quantization.

```sh
cmake -S glm53 -B build-glm53 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGLM53_SERVER=ON
cmake --build build-glm53 -j 8
OMP_NUM_THREADS=8 ctest --test-dir build-glm53 --output-on-failure
```

For R9700 inference, configure a separate ROCm build:

```sh
cmake -S glm53 -B build-glm53-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGLM53_SERVER=ON -DGLM53_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-glm53-hip -j 8
ctest --test-dir build-glm53-hip --output-on-failure
```

For RTX 2080 Ti inference, use a CUDA toolkit that still supports Turing (CUDA
12.9 was used for compilation) and configure:

```sh
cmake -S glm53 -B build-glm53-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGLM53_SERVER=ON -DGLM53_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-glm53-cuda -j 8
ctest --test-dir build-glm53-cuda --output-on-failure
```

Choose one GPU provider per build. Both expose the same `--device 0` CLI/server
interface; `--device cpu` (the default) uses the CPU reference. For other NVIDIA
devices, set `CMAKE_CUDA_ARCHITECTURES` accordingly: `120` for RTX PRO 6000
Blackwell (CUDA 12.8 builds this tree for sm_120 without changes; the kernels use
no architecture-specific intrinsics). `--tp N` / `--layer-split N` select tensor
parallelism or the layer pipeline over N GPUs (see above).

The GPU tests compare F32/BF16 and FP8/NVFP4 projections with CPU widening, test
KDA recurrence, check sparse top-k ordering, partial pools and causal tails, and
compare tiny dense/MoE source and NVFP4 models over 33 tokens with state reset.
They also check row and column slices of every weight format against the full
projection, and run TP2 and TP4 on tiny dense and MoE models (ranks emulated on
one GPU when fewer are present) against the CPU reference and the single-GPU
path. `scripts/glm_tp2_validate.sh MODEL_DIR` runs the two-GPU validation on a
real checkpoint (tests, teacher-forced logits of TP2 vs the layer split, decode
rate).
They return CTest skip code 77 without an available GPU. Numerical BF16 model
checks allow relative L2 error below 2% and maximum absolute logit error below
0.03 per position; these thresholds are test criteria, not measured results.

## Inspect, convert, and generate

```sh
build-glm53/glm53 inspect --model /models/glm-source
build-glm53/glm53 quantize --model /models/glm-source --out /models/glm-nvfp4
build-glm53/glm53 generate --model /models/glm-nvfp4 --messages chat.json --steps 64
build-glm53-hip/glm53 generate --model /models/glm-nvfp4 --messages chat.json --steps 64 --device 0
build-glm53-cuda/glm53 generate --model /models/glm-nvfp4 --messages chat.json --steps 64 --device 0
```

`chat.json` is an array, for example:

```json
[{"role":"system","content":"Be concise."},{"role":"user","content":"Hello."}]
```

Conversion writes a **new directory**, using a temporary sibling and rename on
success. It converts dense, routed-expert, and shared-expert gate/up/down
projections to weight-only NVFP4: two E2M1 values per byte, low nibble first, one
E4M3 scale per 16 input channels, and one F32 global scale per matrix. Packed
weights use a versioned `glm53_quant` marker and bundled safetensors shards.
Non-MLP tensors retain their source dtype and bytes, including FP8 scales.
Both original block-128 FP8 (`weight_scale_inv`) and converted per-output-channel
FP8 (`weight_scale`) are supported. Execution widens scaled weights to BF16.
GPU GEMV widens values in registers rather than expanding the entire MLP matrix.

This converter uses nearest-value weight reconstruction; it has no calibration,
teacher reconstruction, QAT, or GLM QTIP path. Conversion is not resumable. Keep
space for source and output, plus transient expanded matrices and shard buffers.
The original model's MLP payload alone is about **172 GB** in this format; the
dense model's is about **31.7 GB**, before attention, embeddings, and other tensors.
The complete models are not assumed to fit in two 32 GB GPUs.

Tokenizer loading applies the checkpoint's byte-level BPE, regex splitting,
special tokens, and actual Jinja chat template. Both the GLM tokenizer and the
Llama-3-style tokenizer are supported. Serialized tokenizer truncation and padding
are ignored, including the supplied tokenizer's 2,048-token limit. Raw text
encoding adds no implicit BOS; the chat template supplies its own special tokens.

## HTTP and traces

```sh
build-glm53-hip/glm53_server --model /models/glm-nvfp4 --device 0 --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

The server provides `/health`, `/v1/models`, and `/v1/chat/completions`, with greedy
or temperature/top-k/top-p sampling, seed, stop strings, usage, and SSE streaming.
It resets recurrent state per request. `<think>` blocks, including reopened
blocks, go into `reasoning_content`; a template-prefilled `<think>` starts in that
channel. Requests run serially. Tool calls, multimodal input, structured outputs,
prefix caching, and batching are not implemented here. Vision and MTP weights
are preserved by conversion but are not executed.

For a teacher-forced logit comparison, create a request such as
`{"tokens":[1,2,3],"logit_positions":[0,1,2]}` and run:

```sh
build-glm53/glm53 trace --model /models/glm-nvfp4 --tokens request.json \
  --out trace.json --arithmetic fp32
```

`bf16` is the default execution arithmetic; `fp32` is for the CPU reference.
The MLA implementation uses latent-space absorption rather than expanding every
cached K/V head. It is algebraically equivalent in FP32, but rounding differs
from an eager BF16 implementation. No BF16 logit-parity or perplexity claim is made.

## Validation

CPU CTest covers both actual configuration geometries, changed attention
dimensions in a tiny dense model, KDA state, sparse MLA, mHC, MoE, state reset,
FP8 scale layouts, NVFP4 reconstruction, tokenizer behavior, and live HTTP
streaming/nonstreaming consistency. An independent Transformers oracle compares
seven incremental tokens with full-sequence eager execution in tiny dense and
MoE models, both before and after NVFP4 conversion:

```sh
python -m pip install torch transformers safetensors tokenizers jinja2
OMP_NUM_THREADS=4 python glm53/tests/hf_parity.py --build build-glm53
python glm53/tests/tokenizer_parity.py --binary build-glm53/glm53 --model /models/glm-source
python glm53/tests/tokenizer_parity.py --binary build-glm53/glm53 --model /models/dense-source
```

The oracle requires Transformers with `glm5_next` support (validated with
5.22.1). FP32 tiny-model comparisons had relative L2 error below `4e-7` and maximum
absolute error below `6e-7`. Both real tokenizer families matched Hugging Face
token IDs and Jinja rendering, including a 4,097-token input. These checks do not
establish full-checkpoint accuracy, absence of repetition, long-context quality,
or full-checkpoint GPU correctness. The CUDA and HIP builds compile and link for
sm_75 and gfx1201 respectively. The CUDA kernel tests, tiny dense/MoE source and
NVFP4 model comparisons, and state-reset tests now also pass on an A100 SXM 80GB
with CUDA 12.4. The shared CUDA CLI/server dispatch tests pass on that device.
Full-size GLM checkpoints and execution on Turing/R9700 remain unvalidated. The
local CUDA 12.9 SDK needed the four standard `sinpi`/`cospi` declarations marked
`noexcept` to match this environment's newer glibc; no SDK changes are shipped
in this repository.

The GLM CPU correctness suite also passes with address/undefined sanitizers;
the existing runtime's core and structured-output regression suites pass after
sharing its reasoning-stream parser with the new server.

Architecture references: [official checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash),
[Transformers GLM5Next](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm5_next).
The Jinja interpreter is vendored [Google Minja](../third_party/minja/README.md).
