# Training attention backward performance

## GPU measurements through `5ee3d77`

The user reported these steady GPU measurements after pulling `04a8fc9`
(paired-wave dK/dV) and `5ee3d77` (balanced 64-key dK/dV). All seven CTest tests
passed on the former and all nine on the latter, including the wave and
balanced32 A/B paths. No MES errors were observed in either run.

| Layer 0, 32k sequence (`qtip_bench`) | `a067e7e` | `04a8fc9` | `5ee3d77` |
| --- | ---: | ---: | ---: |
| Forward | 1.85 s | 1.85 s | 1.85 s |
| Backward | 26.5 s | 8.32 s | 6.23 s |
| Forward + backward | 28.4 s | 10.2 s | 8.1 s |
| Estimated 88-layer step | 42 min | 15 min | 11.9 min |

| Attention-only case | `a067e7e` backward | `04a8fc9` backward | `5ee3d77` backward | `5ee3d77` dQ | `5ee3d77` dK/dV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8k, one document | 0.90 s | 0.30 s | 0.19 s | 54 ms | 129 ms |
| 32k, one document | 32.0 s | 5.42 s | 3.43 s | 1.22 s | 2.18 s |

Useful backward throughput reached approximately 30 TFLOP/s at 8k and 27
TFLOP/s at 32k, up from 19 and 17 TFLOP/s on `04a8fc9`. dK/dV improved from
237 to 129 ms at 8k and from 4.18 to 2.18 s at 32k. The earlier staged dK/dV
kernel took 840 ms at 8k. Layer timings and attention-only timings come from
different benchmarks; use full-layer runs for step-time estimates.

## Balanced score work and greater Q/dO reuse

In the paired-wave kernel, the dK wave computes QK, dP and dK while the dV
wave computes only dV. The balanced kernel gives QK and dK to one wave and
dP and dV to the other. FP32 dP crosses through a padded LDS tile before the
dK wave computes P/dS; BF16 P then crosses to the dV wave. Each wave retains
one K/V operand rather than both. Query heads accumulate in a fixed order,
with one writer per dK/dV element and no attention-gradient atomics.

Four pairs own 64 keys and reuse the same staged 16-query Q/dO block. Q/dO
global loads per key-query pair are halved relative to `04a8fc9`. The tradeoff
is one extra barrier per query tile (four instead of three) and the FP32 dP
exchange. The seven backward matrix products per causal pair are unchanged.
No global workspace, allocation, or model precision change is introduced.

`balanced32` uses the same algorithm over 32 keys to separate the effect of
balancing the score work from the effect of greater load reuse. It also
provides a smaller tile for benchmarking short or heavily packed sequences.
Both `wave` (`04a8fc9`) and `staged` (`a067e7e`) remain selectable.

## Optional 128-key experiment

The validated 64-key `balanced` kernel remains the default. `balanced128`
extends the same algorithm to eight wave pairs and 128 keys, sharing the
16-query Q/dO block twice as widely again. K/V loads at kernel entry are
batched into 64 rows, reusing the same 16 KiB query staging buffer. Only the
small P/dP exchange buffers grow; no global workspace is added. The steady
query loop still has four barriers and the same matrix products per pair.

The larger block has 512 threads instead of 256. It can reduce Q/dO traffic,
but larger barriers, fewer independent blocks and extra masked work near
packed-document boundaries may offset the benefit. GPU timing must determine
whether it helps. The compiler emits unchanged instructions for the 32- and
64-key balanced specializations, after normalizing assembly labels and comments.

ROCm 7.0 clang 20, `-O3`, gfx1201 Wave32 compilation reports:

| `QTIP_TRAIN_ATTN_DKV` | Keys/block | Threads | VGPRs | LDS bytes | Scratch bytes | VGPR spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged` | 16 | 128 | 159 | 30,720 | 0 | 0 |
| `wave` | 32 | 128 | 215 | 17,600 | 0 | 0 |
| `balanced32` | 32 | 128 | 175 | 20,160 | 0 | 0 |
| `balanced` (default) | 64 | 256 | 166 | 23,744 | 0 | 0 |
| `balanced128` | 128 | 512 | 166 | 30,912 | 0 | 0 |

Local validation: host/device HIP object compilation, GPU-test source syntax
checking, the CPU training test, and 246 host WMMA-layout reference cases for
the two balanced tile sizes passed. The layout cases cover sequence tails,
GQA groups 1/3/4/12/32, packed boundaries and one-token documents; worst gradient
relative error against dense FP32/BF16 reference products was 0.036%. The host
model is a supplementary layout check, not execution of the HIP kernel.

For the 128-key addition, HIP object compilation and the CPU training test
passed. An extended host model passed 369 cases across 32/64/128-key tiles,
including both batches of K/V staging and partial final blocks. On identical
inputs, all tile sizes produced bitwise equal model gradients; worst relative
error against dense reference products was 0.055%. This still does not execute
the GPU kernel or measure its performance.

The user has now validated GPU execution and measured the 64-key balanced
kernel as recorded above. There was no AMD GPU in the development workspace;
**GPU correctness and speed of `balanced128` remain unmeasured.** Run the full GPU CTest
suite and compare 8k, 32k contiguous and 32k with 2k documents using the
[A/B commands](FINE_TUNING.md#evaluate-and-benchmark) (search that page for
`for dkv in balanced balanced128`). Then benchmark the full layer with
the same GEMM tuning settings and warmup policy.

## Step-time targets from the reported baseline

Using 1.85 + 6.23 = 8.08 s/layer on `5ee3d77` and holding other work constant:

| Assumption | Layer estimate | 88-layer step estimate |
| --- | ---: | ---: |
| dK/dV time equals dQ: 2.18 s becomes 1.22 s | 7.12 s | 10.4 min |
| dK/dV matches dQ matrix throughput: four products vs three, about 1.63 s | 7.53 s | 11.0 min |
| dK/dV time halves again: 2.18 s becomes 1.09 s | 6.99 s | 10.3 min |
| All attention backward time removed | 4.65 s | 6.8 min |

These are bounds and targets derived from the baseline, not measurements of
`balanced128`. Matching dQ matrix throughput now saves approximately 49 s per
88-layer step. GEMMs, forward recomputation, transfers and other work must
still be measured in the full trainer.
