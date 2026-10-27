# Prefill compute and remaining headroom

Reviewed commit: `c42cf03`, round 5. Calculations use the measurements in
[PERF_ROUND5_REPORT.md](PERF_ROUND5_REPORT.md); **no new GPU timings were collected**.
This executor has no GPU or access to .229. Native CPU accounting, a gfx1201
attention compilation/resource inspection, and host syntax checking were possible.

The application is not sustaining peak matrix throughput. At 2,048 query rows,
projection work averages 60.1 TFLOP/s per GPU at short context. That is 31.4% of
boost-clock BF16 matrix peak, while earlier isolated GEMMs reached 110–134 TFLOP/s.
Long-context attention still has substantial headroom after the round-5 improvement.
Communication and synchronization also limit the full pipeline.

## Hardware denominator

AMD specifies **191 TFLOP/s dense FP16 matrix**, **383 TFLOP/s dense FP8 matrix**,
64 CUs, up to 2.920 GHz, and 640 GB/s memory bandwidth per R9700:
[official specifications](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html).
AMD's [RDNA 4 WMMA rate table](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/)
gives the same throughput for BF16 and FP16, 1,024 FLOPs/clock/CU; FP8 is twice that.
The calculated peaks are 191.365/382.730 TFLOP/s; the calculator uses rounded
191.4/382.8. The pair therefore has approximately **382.8 TFLOP/s BF16** or
**765.6 TFLOP/s FP8** peak capacity.

The sparse/INT4 marketing numbers do not apply: QTIP prefill decodes weights to
BF16 and uses dense BF16-input, FP32-accumulate hipBLASLt GEMMs. FP32 accumulation
does not turn these into vector FP32 operations. Default attention is FP8 Q/P/K/V;
`BEHEMOTH_ATTN_QP=bf16` changes Q/P arithmetic, while K/V remain stored as FP8.
BF16 prefill *exchange* is a separate setting.

The percentages below are useful-matrix work divided by boost-clock peak over
wall time. They are not GPU-busy percentages, issue-counter measurements, or
clock-adjusted utilization. Power caps, sustained frequency, SFU/VALU work,
transforms, redundant tile work, and memory/communication costs all matter.

## Projection arithmetic

Model: 88 layers, hidden 12,288, MLP 28,672, 96 query heads, 8 KV heads, head dim
128. Count each multiply-add as two FLOPs. Per layer:

```
weights = 2 * 12288^2 + 2 * 12288 * 1024 + 3 * 12288 * 28672
        = 1,384,120,320
projection FLOPs/token = 2 * weights * 88 = 243,605,176,320
```

For a 2,048-token step, actual TP=2 GEMM shapes and work are:

| Stage | Weight shape per GPU, output × input | TFLOP per GPU, all 88 layers |
|---|---:|---:|
| stacked q/k/v | 7,168 × 12,288 | 31.75 |
| o | 6,144 × 12,288 | 27.21 |
| stacked gate/up | 28,672 × 12,288 | 126.99 |
| down | 6,144 × 28,672 | 63.50 |
| **Total** | | **249.45** |

The pair performs **498.90 TFLOP per 2,048-token step** in these projections.
The MLP accounts for 76.4%. The benchmark samples just the final row, so lm_head
adds only 0.000805 TFLOP per step, not that amount times 2,048. The calculator
accepts `--sampled-rows` for scoring workloads that evaluate more vocabulary rows.
The sign/Hadamard transforms and trellis decode are extra operations outside this
matrix-work accounting.

## Attention and achieved throughput

For B new tokens following C already-cached tokens, the number of valid causal
query/key pairs is `B*C + B*(B+1)/2`. Both QK and PV contribute:

```
attention FLOPs = 4 * 88 * 96 * 128 * (B*C + B*(B+1)/2)
```

At B=2,048 this gives:

| Existing prefix C | Projection TFLOP, pair | Attention TFLOP, pair | Observed tok/s | Step seconds | Precision-weighted peak fraction |
|---:|---:|---:|---:|---:|---:|
| 0 | 498.90 | 9.08 | 493.5 | 4.150 | 31.7% |
| 8,192 | 498.90 | 81.64 | 423.2 | 4.839 | 29.1% |
| 16,384 | 498.90 | 154.21 | 369.7 | 5.540 | 27.2% |
| 32,768 | 498.90 | 299.35 | 290.8 | 7.043 | 24.1% |

For default FP8 attention, the fraction is:

```
(projection_FLOPs / pair_BF16_peak + attention_FLOPs / pair_FP8_peak
 + sampled_head_FLOPs / pair_FP8_peak) / step_wall_seconds
```

The original round-5 table does not record `BEHEMOTH_ATTN_QP`. FP8 is the source
code default and the assumption above. If those runs used BF16 Q/P, the four
fractions would instead be approximately 32.0%, 31.3%, 30.8%, and 29.6%.
Either interpretation leaves significant headroom.

These are **trailing 2,048-token step rates**, not whole-prompt TTFT. For a whole
cold 32,768-token prompt, projections cost 7.982 PFLOP and attention 2.322 PFLOP.
A cold prompt visits a range of prefix lengths; dividing 32k by the slowest
trailing-step rate does not give measured whole-prompt TTFT.

The useful attention count excludes masked elements still processed within a
16-query/32-key boundary tile, and padding on partial tiles. It also excludes
softmax, FP8 conversions, scaling, and merge/reduction operations.

## What round 5 improved, and what remains

Holding projection work constant, subtract the context-0 latency from each
long-context latency. Dividing the additional per-GPU attention FLOPs by that
time gives:

| Prefix | Before round 5, TFLOP/s/GPU | Round 5, TFLOP/s/GPU |
|---:|---:|---:|
| 8,192 | 22.9 | 52.6 |
| 16,384 | 23.6 | 52.2 |
| 32,768 | 23.9 | 50.2 |

This is a **marginal estimate**, assuming non-attention costs stay approximately
constant. It is not a timed isolated-attention result. Its consistency across
three contexts supports attention as the growing cost. Round 5 roughly doubled
that rate, but approximately 50 TFLOP/s is only 13% of FP8 boost-clock matrix peak.
Softmax, conversion, register lifetime, LDS, and synchronization must be profiled
before deciding which resource is actually limiting this kernel.

An illustrative scenario: halving only the extra context-dependent cost at 32k
would change the 2,048-token step from 7.043 to 5.596 seconds, about **366 tok/s
(+26%)**. That is an Amdahl calculation, not a promised kernel speedup.

Compile-only resource inspection with ROCm 7.1 / AMD clang 20, gfx1201:

| Attention variant | VGPRs | Scratch bytes/lane | Reported waves/SIMD limit | LDS bytes/block |
|---|---:|---:|---:|---:|
| previous FP8 | 204 | 0 | 7 | 24,576 |
| round-5 FP8 | 169 | 0 | 8 | 12,288 |
| previous BF16 Q/P | 256 | 64 | 5 | 28,928 |
| round-5 BF16 Q/P | 230 | 80 | 6 | 17,664 |

The default FP8 kernel has no compiler-reported spills or scratch. BF16's nonzero
scratch is worth investigating separately; the round-5 compiler reports zero
VGPR spills despite that local storage. These are compiler estimates, not
measured resident occupancy. Production uses a different toolchain (the reports
cite ROCm 7.2), so collect its resource data before making register-limit changes.

## Ranked improvements to test

1. **Attention inner-loop work and register lifetime.** The round-5 kernel already
   avoids probability LDS traffic and most fragment rearrangement. Next compare
   key tiles 32/64 and head groups 12/6/4, while tracking registers, scratch, LDS,
   matrix instructions, exp/convert instructions, and elapsed time. Larger query
   tiles are not automatically better: output accumulators already consume 64
   FP32 registers per lane, and doubling queries doubles that state. AMD's
   [global transpose loads](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/)
   are another experiment for V staging, respecting 16-token physical page
   boundaries and the instruction's operand layout. Do not assume CDNA/gfx1250
   LDS transpose instructions exist on gfx1201. Preserve Q/P precision and
   existing masking semantics while testing these variants.

2. **PCIe critical path.** BF16 exchange still sends **106 KiB/token/layer per
   rank in each direction**: qkv 14 KiB, attention 12, o 12, gate/up 56, down 12.
   At 2,048 tokens and 88 layers that is **19.562 GB each way**. The earlier
   measured 12.5 GB/s duplex path on GPU 0 gives an ideal wire floor of **1.565 s**
   per step; GPU 1 was roughly twice as fast. This rate may have changed and needs
   a fresh measurement. Check every upstream PCIe hop and actual LnkSta, CPU/NUMA
   placement and concurrent copies. Do not attribute the asymmetry to a specific
   hardware cause without those measurements. Raising useful duplex bandwidth
   to 25 GB/s would halve that wire floor; the wall-time gain depends on overlap.

3. **Direct BF16 GEMM output into the exchange block.** `qtip_project` currently
   enables direct output only for `!bf16`, so BF16 prefill writes an FP32 result
   and then converts/copies it with `reduce_publish`. Add a hipBLASLt BF16-output
   plan with FP32 accumulation, including output dtype in the plan key and
   autotune. This can remove approximately **69.4 GB of logical intermediate
   write/read traffic per GPU per 2,048-token step**, across the four projections.
   This is not measured DRAM traffic: caches and GEMM implementation affect the
   physical cost. It is a smaller, concrete optimization; benchmark exact output
   types/shapes and test BF16 rounding plus held-out quality before enabling it.

4. **Jointly tune GEMM shapes and overlap.** Re-measure 1,024/2,048/4,096 query
   rows, one versus two micro-batches, and DMA piece counts 1/2/4/8. Use exact
   batch sizes and output dtypes in the GEMM tuning. Earlier four-micro-batch
   experiments regressed; increasing their count without a measured reason is
   not an improvement. The previously measured 110–134 TFLOP/s isolated GEMMs
   are 57–70% of BF16 boost peak and should be refreshed under the same clocks
   and power cap as the full-model benchmark.

5. **Sequence-parallel, layer-at-a-time cold prefill.** This larger redesign was
   already identified in PREFILL_INVESTIGATION.md. For a 32k prompt, current TP
   activation exchange is about **313 GB per rank per direction**. Exchanging the
   peer's packed weight shard once per prompt is about 15.2 GB per rank, plus KV
   redistribution. It needs transient peer-weight/dense scratch, balanced causal
   query positions, layer-local KV gathering, final normal TP KV ownership,
   memory budgeting, and a decode scheduling policy. It does not require global
   token pruning or changing prefix-cache semantics. The byte savings are not
   a measured speedup or a ready-to-enable implementation.

The BF16 projection-only theoretical ceiling is approximately **1,571 tok/s** for
the pair; this ignores attention and all overhead. At the earlier isolated GEMM
rates, the analogous ceiling is **903–1,100 tok/s**. Consequently, a 4x gain over
493.5 tok/s cannot come from optimizing this same BF16 projection pipeline alone.
FP8 projection GEMMs might raise the math ceiling, but would additionally quantize
decoded QTIP weights/activations and require separate quality validation.

## Instrumentation and reproducibility

Added the CPU-only `behemoth_prefill_compute` executable and saved analytical
results in `perf/results/prefill/round5_compute_accounting.json`. They label
supplied throughput as measured input, and predictions as analytical work.

The existing event profiling had two limitations: `measure(Mb, ...)` recorded
only micro-batch 0, and `dense_projections_ms` combined decode, GEMMs, and waits.
The instrumentation now records all micro-batches and adds `qtip_gemm_ms`,
`qtip_weight_decode_ms`, and a separately marked tiled-scratch fallback. The
high-level timing remains available. `gpu_event_totals_are_sums_across_streams`
explicitly warns that nested/overlapping event intervals are not wall-clock busy
time. Benchmark output also records attention Q/P precision and the actual
micro-batch count. These changes do not change numerical operations or serving defaults.

Build the accounting executable without HIP:

```sh
cmake -S runtime -B build-prefill-analysis -DCMAKE_BUILD_TYPE=Release
cmake --build build-prefill-analysis --target behemoth_prefill_compute
build-prefill-analysis/behemoth_prefill_compute \
  --rows 2048 --context 32768 --tokens-per-second 290.8
```

It accepts `--config MODEL/config.json`, `--attention bf16`, different peaks and
sampled-row counts. It requires no model weights, GPU, or Python.

On .229, explicitly match the BF16 exchange and attention mode, and profile the
single-stream path first so that GEMM event sums are interpretable:

```sh
BEHEMOTH_QTIP_BF16_PREFILL_ROWS=512 BEHEMOTH_ATTN_QP=fp8 \
BEHEMOTH_BENCH_PROFILE=1 BEHEMOTH_BENCH_PATHS=dense \
  build-rocm/behemoth_prefill_bench MODEL 2048 0,8192,16384,32768 5
```

Then repeat without profiling and with `BEHEMOTH_BENCH_PATHS=overlap`; profiling
itself adds overhead. Match clocks/power/temperature and use repeated medians.
Trace both ranks' compute, DMA, waits and overlap with rocprofv3. In particular,
dividing full-step FLOPs by an overlapping sum of event durations does not give
aggregate hardware throughput. The actual-clock roofline and instruction/cache
counters are still needed to establish physical saturation and its cause.

Validated locally: the CPU tool builds through CMake; default/model-config and
boundary arithmetic checks; native runtime CPU tests; gfx1201 compilation of
attention with resource reports; host syntax checking of the profiling changes.
GPU execution, timing and quality tests remain pending on target hardware.
