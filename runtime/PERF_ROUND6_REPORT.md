# Performance round 6

Hardware: 2-GPU tensor parallel (`--devices 0,1`), Behemoth-R1-123B-v2-QTIP-2bit, .229.
All prefill numbers: `behemoth_prefill_bench MODEL 2048 0,8192,16384,32768 3`,
`BEHEMOTH_BENCH_PATHS=dense BEHEMOTH_QTIP_BF16_PREFILL_ROWS=512 BEHEMOTH_ATTN_QP=fp8`
(trailing 2048-token step after the given prefix).

## Profiled baseline (42ac8f9)

`BEHEMOTH_BENCH_PROFILE=1`, dense, rank 0 event sums (ms per 2048-token step; single stream,
so the sums are serial time):

| ctx | wall | qtip_gemm | qtip_weight_decode | collectives (exchange) | elementwise | attention_and_cache |
|----:|-----:|----------:|-------------------:|-----------------------:|------------:|--------------------:|
| 0     | 4400 | 2111 | 208 | 1565 | 366 | 137 |
| 32768 | 7248 | 2090 | 207 | 1644 | 368 | 2922 |

(Rank 1: GEMM 2338/2324, decode 237, collectives 1229/1230.) The GEMMs ran at 110-139
TFLOP/s in the startup autotune.

## A: BF16 GEMM output straight into the exchange block

`qtip_project` wrote FP32 GEMM output and then `reduce_publish` rounded/copied it into the
BF16 exchange block. hipBLASLt now has BF16-output plans (FP32 accumulation; output type in
the plan and autotune keys, `DenseMatmul::gemm/tune` overloads for `BFloat16 *`), and the dense
copy-engine path writes this rank's half of the exchange block directly for BF16 payloads
(gate/up always, q/k/v, o, down in BF16-exchange prefill steps). `BEHEMOTH_QTIP_BF16_DIRECT=0`
restores the old path. The BF16-output GEMMs are also faster by themselves (autotune
134-175 vs 111-139 TFLOP/s: half the output bytes).

| ctx | direct off tok/s | direct on tok/s | change |
|----:|-----------------:|----------------:|-------:|
| 0     | 489.4 | 541.9 | +10.7 % |
| 8192  | 419.4 | 459.5 | +9.6 % |
| 16384 | 367.3 | 398.0 | +8.4 % |
| 32768 | 289.7 | 308.5 | +6.5 % |

Profile at ctx 0 with direct on: wall 3833 ms, qtip_gemm 1821, collectives 1375,
elementwise 280 (from 4400 / 2111 / 1565 / 366).

Quality: greedy top-k outputs (`behemoth_eval topk`, 4 prompts x 48 tokens) identical on/off;
held-out perplexity (16384 tokens) 2.94061 vs 2.94071 (round 5, rows 512). New unit test:
BF16-output GEMM equals the rounded FP32 GEMM except rare 1-ulp differences
(accumulation order). Model prefill/prefix tests pass.

## B: prefill attention kernel (`gqa_prefill_kernel`, `src/attention.hip`)

Iterated with the standalone `perf/attention_round4/long_attention.cpp` (one rank: 48 query /
4 KV heads, 2048 queries after the given prefix, 20 reps, ms per call). ISA inspection of the
round-5 FP8 loop showed it was VALU/latency bound, not matrix bound: an IEEE division and a
full `expf` per probability, the NaN/inf-guarded scalar FP8 conversion, the V transposition
compiled to branches (dynamic register indexing; this was also the BF16 variant's 80 B
scratch), and `__syncthreads()` waiting on `s_wait_loadcnt 0`, which drained the K/V register
prefetch at every barrier.

| variant | FP8 prefix 0 | 8k | 16k | 32k | BF16 Q/P 32k | VGPR fp8/bf16, scratch |
|---|---:|---:|---:|---:|---:|---|
| round 5 (base) | 1.272 | 9.411 | 17.535 | 33.162 | 38.144 | 170/225, bf16 80 B |
| v3: LDS-only barriers + select-only V transpose + exp2/rcp/packed FP8 cvt | 0.651 | 4.615 | 8.750 | 16.890 | 30.423 | 173/223, 0 |
| v4: + 64-key staged tile for FP8 (2 x 32 sub-tiles per barrier pair) | 0.645 | 4.570 | 8.654 | 16.676 | 30.758 | 184/233 |
| v5: + unmasked path for fully visible key tiles, no division in the P scale | 0.599 | 4.169 | 7.855 | 14.964 | 28.345 | 185/223 |
| **v6: + O^T accumulated by the WMMA (P-scale units)** (kept) | **0.576** | **4.022** | **7.510** | **14.436** | **27.535** | 188/214, 0 |
| v6 + sticky P scale (reuse previous scale within 2x) | 0.591 | 4.072 | 7.607 | 14.639 | 27.547 | |
| v7: v6 + score scale folded into the exp FMA | 0.560 | 3.984 | 7.432 | 14.295 | 27.534 | 180/220 |
| v6, head groups of 6 waves (2 blocks per KV head/tile) | 0.678 | 4.918 | 9.255 | 17.443 | | 206, occupancy 7 |
| v6, head groups of 4 waves | 0.643 | 4.779 | 8.971 | 17.145 | | 206, occupancy 6 |

Kept v6 (v7 within noise, head groups 6/4 are 20 % slower: K/V staging repeats per block
and the residency does not improve). FP8 attention is 2.3x faster at 32k (about 118
TFLOP/s per GPU of useful causal work vs 52), BF16 Q/P 1.39x. 64-key tiles stay FP8-only
(BF16 at 64 keys spills, 140 B scratch).

Correctness: against the CPU kernel reference (`reference_attention`, same FP8 P semantics)
the errors are unchanged (fp8 1.66e-3, bf16 2.32-2.34e-3 rel rms; vs exact float attention
unchanged); against the round-5 kernel on the round-5 A/B case list (causal, mixed prefix,
decode rows, key partitions 0/512/2048, both precisions) rel rms <= 2.8e-4. `ctest`,
`behemoth_gpu_tests`, model prefill/prefix tests pass.

End to end (direct BF16 GEMM output on in both columns):

| ctx | A only tok/s | A + B tok/s | change | vs round-5 baseline (493.5/423.2/369.7/290.8) |
|----:|------------:|------------:|-------:|------:|
| 0     | 541.9 | 549.8 | +1.5 % | +11 % |
| 8192  | 459.5 | 512.6 | +11.6 % | +21 % |
| 16384 | 398.0 | 479.1 | +20.4 % | +30 % |
| 32768 | 308.5 | 422.7 | +37.0 % | +45 % |

`BEHEMOTH_ATTN_QP=bf16`: 471.9 tok/s at 8k, 336.6 at 32k. Greedy outputs identical; held-out
perplexity 2.93935 (A only: 2.94061). No MES errors during the round.

Remaining at 32k (profile, A only): attention ~2.9 s of 6.6 s per step before B; after B
the projection GEMMs (~1.8 s), PCIe exchange (~1.4 s) and elementwise kernels dominate again.
