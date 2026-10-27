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
