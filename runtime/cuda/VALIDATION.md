# CUDA execution validation

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
