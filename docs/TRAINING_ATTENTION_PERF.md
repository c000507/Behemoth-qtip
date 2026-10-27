# Training attention backward performance

## GPU baseline: `04a8fc9`

The user reported these steady GPU measurements after pulling `04a8fc9`
(paired-wave dK/dV). All seven CTest tests passed, including
`native_gpu_staged_dkv`; no MES errors were observed.

| Layer 0, 32k sequence (`qtip_bench`) | `a067e7e` | `04a8fc9` |
| --- | ---: | ---: |
| Forward | 1.85 s | 1.85 s |
| Backward | 26.5 s | 8.32 s |
| Forward + backward | 28.4 s | 10.2 s |
| Estimated 88-layer step | 42 min | 15 min |

| Attention-only case | Earlier backward | Paired-wave backward | Paired-wave dQ | Paired-wave dK/dV |
| --- | ---: | ---: | ---: | ---: |
| 8k, one document | 0.90 s | 0.30 s | 57 ms | 237 ms |
| 32k, one document | 32.0 s | 5.42 s | 1.24 s | 4.18 s |

Useful backward throughput was approximately 19 TFLOP/s at 8k and 17 TFLOP/s
at 32k. At 8k, the preceding staged dK/dV kernel took 840 ms. Layer timings and
attention-only timings come from different benchmarks; use full-layer runs for
step-time estimates.

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

ROCm 7.0 clang 20, `-O3`, gfx1201 Wave32 compilation reports:

| `QTIP_TRAIN_ATTN_DKV` | Keys/block | Threads | VGPRs | LDS bytes | Scratch bytes | VGPR spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `staged` | 16 | 128 | 159 | 30,720 | 0 | 0 |
| `wave` | 32 | 128 | 215 | 17,600 | 0 | 0 |
| `balanced32` | 32 | 128 | 175 | 20,160 | 0 | 0 |
| `balanced` (default) | 64 | 256 | 166 | 23,744 | 0 | 0 |

Local validation: host/device HIP object compilation, GPU-test source syntax
checking, the CPU training test, and 246 host WMMA-layout reference cases for
the two balanced tile sizes passed. The layout cases cover sequence tails,
GQA groups 1/3/4/12/32, packed boundaries and one-token documents; worst gradient
relative error against dense FP32/BF16 reference products was 0.036%. The host
model is a supplementary layout check, not execution of the HIP kernel.

There was no AMD GPU in the development workspace. **GPU correctness and
speed of the balanced variants remain unmeasured.** Run the full GPU CTest
suite and compare 8k, 32k contiguous and 32k with 2k documents using the
[A/B commands](FINE_TUNING.md#evaluate-and-benchmark) (search that page for
`for dkv in wave balanced32 balanced`). Then benchmark the full layer with
the same GEMM tuning settings and warmup policy.

## Step-time targets from the reported baseline

Using the rounded 10.2 s/layer baseline and holding other work constant:

| Assumption | Layer estimate | 88-layer step estimate |
| --- | ---: | ---: |
| dK/dV time equals dQ: 4.18 s becomes 1.24 s | 7.26 s | 10.6 min |
| dK/dV matches dQ matrix throughput: four products vs three, about 1.65 s | 7.67 s | 11.2 min |
| All attention backward time removed | 4.78 s | 7.0 min |

These are bounds and targets derived from the baseline, not measurements of
the balanced kernels. GEMMs, forward recomputation, transfers and other work
must still be measured in the full trainer.
