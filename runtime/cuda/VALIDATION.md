# CUDA execution validation — 2026-11-04

This report describes baseline commit `021c434`. The subsequent opt-in split-K
kernel and expanded split-count tests have not yet been executed on a GPU.
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
