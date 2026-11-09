# CUDA execution validation

## RTX PRO 6000 Blackwell, GLM-5.3 dense 70B — 2026-11-09

Two RTX PRO 6000 Blackwell (96 GB, sm_120) on a RunPod host, driver 595.91.07,
CUDA 12.8.93, GCC 11.4, CMake 4.4. A llama.cpp evaluation shared both GPUs
during every measurement below (26-45 GB and 20-90% utilization each), so the
timings are approximate upper bounds.

Build: `-DBEHEMOTH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` and the standalone
`glm53` CUDA build both compile without source changes (one glibc
`-Wstringop-overflow` note from `string_fortified.h`, no CUDA warnings). The
kernels carry no `__CUDA_ARCH__` guards or tensor-core paths that depend on the
architecture; FP8/NVFP4 are decoded with integer arithmetic.

Tests: all 8 runtime CTests pass (`cuda_qtip_correctness`, `cuda_llama_correctness`,
`cuda_batch_correctness`, `cuda_architecture_dispatch`, core, structured; the two
peer-collective tests skip with one visible device) and all 4 standalone GLM
tests pass (`glm53_gpu`, `glm53_gpu_models` incl. tiny dense/MoE source and
NVFP4 models GPU vs CPU, HTTP). The Transformers 5.22.2 oracle
(`glm53/tests/hf_parity.py`) passes for tiny dense and MoE models before and
after NVFP4 conversion (relative L2 below 6e-7).

Real checkpoint: the bf16 `glm5_next_text` conversion of Llama-3.3-70B (80
layers, KDA + NoPE sparse MLA every fourth layer, mHC, Llama-3 vocabulary,
161 GB) loads with `--tp 2` as layers 0-39 / 40-79. With the shared GPUs the
automatic budget held 64 GiB of weights per device and streamed 19 GB per token
(the whole checkpoint streamed from host ran at 3.2 s/token before the residency
change).

Teacher-forced logits (`--trace`, every position) against an independent fp32
CPU reference (Transformers `Glm5NextForConditionalGeneration` eager full-sequence
forward, chunked KDA and dense masked attention, on the same checkpoint):

| Prompt | Positions | Top-1 agreement | KL(ref‖CUDA) mean / max | max abs logit diff |
|---|---:|---:|---:|---:|
| Llama-3 chat (system + question) | 53 | 51/53 (96.2%) | 0.0014 / 0.025 | 1.59 |
| PG-19 prose (BOS + 60 tokens) | 61 | 60/61 (98.4%) | 0.0003 / 0.0011 | 0.26 |

The chat mismatches are near-ties in the reference (top-2 gaps 0.051 and 0.012
logits); the PG-19 mismatch has a gap of 0.084. Both implementations predict
`Paris` after the assistant header.
The Llama-3 chat template rendered by the runtime's Minja tokenizes to the same
ids as the HF tokenizer's rendering of the same messages (53 identical ids).

Greedy generation through the Llama-3 template, `--tp 2`: "What is the capital
of France? Answer in one word." gives `Paris.` and stops on `<|eot_id|>`;
"Explain in two sentences why the sky is blue." gives a correct two-sentence
Rayleigh-scattering explanation (51 tokens). Decode ran at 0.93 tok/s with
19 GB of weights still streaming per token next to the co-tenant; prefill is the
same per-token cost, since GLM prefill replays tokens.

Not verified on this hardware: the original GLM-5.3-Flash 320B MoE checkpoint
(no disk space; MoE routing, routed/shared experts and NVFP4 experts are covered
by the tiny-model GPU-vs-CPU tests and the Transformers oracle only), HIP
compilation of the layer split, and throughput without a co-tenant.


## RTX A6000 fast backend — 2026-11-05

Two RTX A6000s on a RunPod US-TX-1 host, driver 595.91.07, CUDA 12.4.131,
GCC 11.4, sm_86. The fast build uses NCCL 2.21.5 and cuDNN 9.1. There is no
NVLink; the GPUs appear as NODE in the topology report.

All six CTests pass: core, structured outputs, QTIP, batched kernels, peer
collectives, and architecture dispatch. Batched projections are checked against
the independent CPU reference, including factors 1/12/28, split reductions,
rotated-output rounding, rank-major layouts, aliased buffers, and partial row
and token tiles. Native decode GQA checks cache isolation and a group of twelve
query heads sharing each KV head. cuDNN prefill checks strided query heads,
causal masking, and unequal sequence lengths against high-precision attention
(relative L2 0.0018332, maximum absolute error 0.00195312 for that fixture).

The two-GPU peer test checks byte-exact AllGather with 56 KiB and 8 MiB per
rank, repeated send-buffer reuse, and three graph replays with changing pinned
host inputs. `cudaMemcpyPeerAsync` was rejected during capture; the implemented
UVA `cudaMemcpyAsync` path passes both uncaptured and captured checks.

The 32-row QTIP register kernel matches the earlier tensor-core products
bit for bit across partial row/token tiles, split counts 1/3/8, and batches
2/7/8/16/17/65. Batch 1 is checked against the independent CPU projection
reference at input widths 12288 and 28672; its default remains the SIMT kernel.
On real checkpoint shards at batch 8, isolated product times decreased from
0.097/0.188/0.186 ms to 0.062/0.137/0.128 ms for Q/gate/down respectively.
These microbenchmarks do not establish whole-model throughput.

The complete portable build also compiles for sm_75 and sm_80. Local CPU
regressions pass; local GPU tests skip because that workspace has no GPU.
This does not establish Turing execution, model quality, or the requested
20/100/500 tok/s throughput targets. Full-model fast-path measurements and
quality comparisons are still in progress.

Measured before the main rebase, with the public mixed FP8-K/V checkpoint,
TP2 peer transport, BF16 KV, 8192 prompt tokens and 256 decode steps:

| Revision at measurement | Batch | Decode tok/s (aggregate) | Prefill tok/s |
|---|---:|---:|---:|
| `895cfd9` | 1 | 15.54 | 454.49 |
| `3624ea7` | 8 | 60.79 | 470.16 |

These are complete model runs with zero warmup runs and fixed generation
lengths. They remain below the requested targets. The rebased equivalents are
`e5e492a` and `8d37814`; performance has not yet been remeasured after the rebase.

On the replacement US-KS-2 A6000 host, both peer GPU stores and peer DMA left
the receiving rank's sentinel bytes unchanged (first byte 238 instead of 83),
despite driver-reported peer access. Those model timings are invalid. The new
startup check rejects both peer modes on this host. NCCL with peer access disabled
passes exact 56 KiB, 224 KiB and 8 MiB AllGather checks, repeated buffer reuse,
and three graph replays with changing inputs. The other six CTests pass here,
including the rebased Llama geometry tests. Peer execution still requires a host
that passes its own integrity checks.

The prefill Hadamard candidate matches the per-token GPU reference bit for bit
at widths 12288 and 28672 with 257 rows, FP32/BF16 inputs, both orientations,
rotated-output rounding, rank-major layouts and in-place fallback cases.
At 8192 rows, isolated input/output transforms decreased from about 2.6 to
1.3 ms at width 12288 and from 6.6–6.9 to 3.1–3.9 ms at width 28672.
Register-resident softmax preserves attention outputs exactly through context
8448 and falls back above that bound; batch-8 8k attention decreased from
0.391 to 0.353 ms. Fused split reduction/scale/BF16 compaction matches the
separate kernels exactly, and every finite E4M3 code decodes exactly, including
signed zero and subnormals.

Using NCCL on all 88 layers, the combined candidate matches the rebased baseline
bit for bit after 256-token bulk prefill and 32 greedy decode steps at batches 1 and 8.
The complete prefill/final logits and generated tokens have identical trace
SHA256 values, respectively:
`d08f611db878398c554b9e90589c0a6bfd98507039f1509a2b9c146036b5c27f` and
`d50d5143ad342112a1ed81a875ddcecc1d62d8f4a3755d195852c3db59b3b25d`.
This finite comparison does not establish long-context model quality.

The staged host AllGather also passes exact data and graph replay checks on
US-KS-2, including an 8 MiB + 13-byte transfer through 1 MiB ring slots. Eight
non-peer CTests pass, including NCCL and host collectives; the peer test still
rejects this host. Full 88-layer traces at batches 1 and 8 match the NCCL
reference exactly after 256-token prefill and 32 decode steps.

For a 448 MiB payload per rank on this host, uncaptured host-staged AllGather
takes 103 ms versus 136 ms with NCCL. Small memcpy graphs have higher host launch
overhead, so inference keeps transfers through 1 MiB on NCCL. This is an opt-in
alternative; it does not establish the requested full-model throughput targets.

Final US-KS-2 measurements at source `b5108f2`, TP2, BF16 KV and 8192/256 tokens:

| Transport | Batch | Warmup runs | Decode tok/s | Prefill tok/s |
|---|---:|---:|---:|---:|
| Host bulk staging, NCCL small transfers | 1 | 1 | 15.05 | 245.55 |
| Host bulk staging, NCCL small transfers | 8 | 0 | 51.53 | 242.58 |

The targets remain unmet. The same host's NCCL-only candidate runs measured
14.15/51.59 decode tok/s and 193.23/199.23 prefill tok/s at batches 1/8 with
zero warmups. These measurements should not be compared as a code regression
against the earlier host's working peer transport. Exact timings, warmup counts,
environment and numerical check hashes are in the [result artifact](results/a6000-us-ks-2.json).

A subsequent HTTP smoke test exposed an under-allocation for cold short prompts:
the small-prefill matmul uses split-K, while prefill reserved only one partial.
The arena now reserves the actual split count below 128 padded rows. The optional
`cuda_model_prefill` regression passes on the complete checkpoint: a cold 32-token
prefill and its next decode step match bit for bit after a 1024-token request
grows the arena and the model resets. The 8k benchmark uses the unchanged bulk
allocation path.
After this fix, the TP2 server with context 32768 passes two identical greedy
HTTP requests for `What is 2 + 2?`: both return `4`, populate `reasoning_content`,
and stop on EOS. The server's health endpoint returns HTTP 200.

## Historical A100 baseline — 2026-11-04

The following report describes baseline commit `021c434`. At the time of this baseline report, the subsequent opt-in split-K
kernel and expanded split-count tests had not yet been executed on a GPU.
RunPod could not restart the existing A100 pod because its host had no free GPUs.
Do not attribute the execution results below to the experimental kernel.

Offline candidate checks: the complete CUDA build succeeds with CUDA 12.9.86,
GCC 14.2, and both `sm_75` and `sm_80` targets. The two CPU CTests pass; the
two GPU CTests skip because this workspace has no GPU. The benchmark's `--help`
runs, and a timing request exits 77 with an explicit GPU-unavailable message.
The default CPU-only build also passes its two regression tests.

Hardware: RunPod A100-SXM4-80GB, compute capability 8.0, driver 595.71.05.
Build: CUDA 12.4.131, GCC 11.4.0, Release, sm_80. The QTIP kernels also compile
for sm_75 with CUDA 12.9; this report contains no Turing/R9700 execution results.

Checkpoint: `cgg507/Behemoth-R1-123B-v2-QTIP-2bit`, Hugging Face revision
`fe6b677e7436c7b817f6f15670110352e679f93c`. Download: 99 files, 33,779,669,339 bytes.
It has 88 layers, QTIP HYB q/o and MLP projections, FP8 K/V projections, and BF16
embeddings/head/norms. It is executed by this repository's native CUDA engine.

## Tests

The runtime CUDA build passes all four CTests:

- `core_correctness`
- `structured_outputs`
- `cuda_qtip_correctness`
- `cuda_architecture_dispatch`

The trellis/RHT test uses the independent double-precision CPU reference across
Hadamard factors 1, 12 and 28, with and without rotated-output BF16 rounding,
including 12288-wide inputs/outputs and 28672-wide outputs. The largest observed
relative L2 error was `9.47e-5`, maximum absolute error `6.11e-5`.
FP8 channel-scale and causal GQA prefix/group tests matched the rounded CPU
results exactly for their fixtures. The shared CLI's GLM dense/MoE source and
NVFP4 traces matched the standalone GLM CUDA traces exactly. Shared HTTP GLM
dispatch and state reset passed.

The standalone GLM CUDA build also passes its four CTests, including GPU kernel
and 33-token source/NVFP4 model comparisons. The default CPU runtime build passes
its two regression tests.

## Full Behemoth runs

All 88 layers load successfully. Device use reported by the loader:

| Allocated context | Device use |
|---|---:|
| 4096 | 33.27 GiB |
| 32768 | 42.89 GiB |

With prompt `What is 2 + 2? Answer briefly.`, greedy generation produced `4` and
stopped on EOS. Prefill was 15 tokens in 3.715 seconds; the short decode emitted
two tokens in 0.493 seconds. This is a smoke test, not a throughput benchmark.

The server was then started with a 32768 context allocation. Two identical
greedy, seeded HTTP requests for `Explain briefly why the sky is blue.` returned
identical output. Each request processed 13 prompt tokens and produced 64
completion tokens, taking 18.80 and 18.77 seconds end to end. The reply began:

> The sky appears blue due to a phenomenon called Rayleigh scattering. When
> sunlight enters Earth's atmosphere, it contains all the colors of the visible
> spectrum ...

The 64-token limit truncated the paragraph as expected. These results demonstrate
execution, coherent samples, and deterministic reset. They do not establish
perplexity, long-context accuracy, or absence of repetition. A 32k cache allocation
does not imply that a 32k prompt was tested.
