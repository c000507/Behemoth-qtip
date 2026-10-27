# Performance round 5

Hardware: 2-GPU tensor parallel (`--devices 0,1`), Behemoth-R1-123B-v2-QTIP-2bit.

## Changes

- New long-context prefill attention kernel (`src/attention.hip`, `src/attention.cpp`).
  The round-4 kernel is kept for A/B via `BEHEMOTH_ATTN_PREFILL_V1=1`.
- Fused decode residual add in the QTIP path (`src/qtip.hip`, `qtip_gpu.hpp`, `model.cpp`).
- `behemoth_server` now defaults to `--qtip-bf16-prefill-rows 512` (BF16 QTIP prefill
  exchange); `--qtip-bf16-prefill-rows 0` restores the FP32 exchange.
- `behemoth_prefill_bench`: `BEHEMOTH_BENCH_DECODE`, `BEHEMOTH_BENCH_ZC_ROWS`,
  `BEHEMOTH_BENCH_PROFILE` knobs.

## Prefill (dense, 2048-token chunk, bf16 prefill rows 512, 1 rep)

| context | base 54f533b tok/s | round 5 tok/s | change |
|--------:|-------------------:|--------------:|-------:|
| 0       | 478.3 | 493.5 | +3 % |
| 8192    | 349.0 | 423.2 | +21 % |
| 16384   | 278.2 | 369.7 | +33 % |
| 32768   | 197.6 | 290.8 | +47 % |

## Quality

- Greedy outputs identical base vs round 5.
- Perplexity: 2.93925 (BF16 exchange off), 2.94071 (rows 512); baseline 2.94004 / 2.93983.
- Decode speed unchanged within noise (+0-2 %).

## Decode at higher row counts (ctx 0)

| rows | zero-copy rows 8 | zero-copy rows 16 |
|-----:|-----------------:|------------------:|
| 8    | 85.1 ms | 86.6 ms |
| 12   | 173.4 ms | 173.7 ms |
| 16   | 175.6 ms | 180.4 ms |

Raising the zero-copy exchange buffer to 16 rows does not remove the step from
8 to 12+ rows, so the copy-engine fallback for >8-row all-gathers is not the cause
of the decode slowdown. Still open.
