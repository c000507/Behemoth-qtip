# QTIP prefill, decode-first scheduling and batched Hadamard transforms

Date: 2026-10-27. Machine: 192.168.0.229, 2x Radeon AI PRO R9700 (gfx1201), stock clocks
(`auto` DPM, 300 W cap, verified), ROCm 7.2, no P2P. Model: Behemoth-R1-123B-v2-QTIP-2bit
(88 layers, TP=2). "Before" = the production binary (cpp_runtime build, `--token-budget 1024
--prefill-chunk 256`), "after" = `runtime/build-rocm/behemoth_server` from this branch with

```
--transport zerocopy --code-layout lane --memory-fraction 0.9 --token-budget 2048 --prefill-chunk 2048
--mixed-prefill-tokens 512 --prefill-interval 24 --threads 4 --profile false
```

Raw results: `/home/charles/qtip_prefill_bench/` on .229 (`before_*.jsonl`, `final_*.jsonl`,
`s_*_mixed.jsonl`, `prefill_bench`/`rprof*` logs). Client: `bench2.py` there (streaming,
per-token timestamps; decode concurrency runs use temperature 0.8, the 100-token story prompt,
128 output tokens; first request after each restart discarded).

## Results

| | before | after | |
|---|---:|---:|---|
| prefill 2k TTFT / tok/s | 10.34 s / 193 | 4.99 s / 401 | 2.1x |
| prefill 8k | 44.2 s / 181 | 21.7 s / 369 | 2.0x |
| prefill 16k | 93.3 s / 171 | 51.3 s / 312 | 1.8x |
| prefill 32k | 209.4 s / 153 | 126.1 s / 254 | 1.7x |
| decode c=1, short context (ms/token) | 43.8 | 44.3 | = |
| decode c=1 after 2k / 8k / 16k / 32k prompt | 46.7 / 51.0 / 52.9 / 56.0 | 47.2 / 51.1 / 53.0 / 56.3 | = |
| c=4 aggregate tok/s, ITL p50 / p95 (ms) | 35.9, 111.5 / 114.8 | 52.3, 76.1 / 77.1 | 1.46x |
| c=8 | 55.4, 145.5 / 147.7 | 79.7, 99.7 / 101.5 | 1.44x |
| c=16 | 66.6, 240.9 / 245.5 | 140.3, 114.8 / 118.0 | 2.1x |
| 4 decoding + 16k prompt: decode ITL p50 / p95 during the prefill | 1716 / 2003 ms | 78 / 81 ms | |
| same: max decode gap / 16k TTFT (prefill tok/s) | 2028 ms / 108.7 s (147) | 2916 ms / 130.5 s (123) | |
| 4 decoding + two 8k prompts: ITL p50 / p95, TTFTs | - | 78 / 90 ms, 61.7 s / 124.0 s | |
| KV pages (tokens) | 8604 (137,664) | 8053 (128,848) | -6.4 % |
| VRAM used per GPU (incl. KV) | - | 30.9 / 34.2 GB | |
| startup | ~4 min | ~4 min (+14 s GEMM autotune) | |

The 4x prefill target is **not** reached: 1.7-2.1x end to end. The reason is measured, not
guessed (section "Where prefill time goes"): at 2048-token steps GPU 0's PCIe link (13-14 GB/s
to host per direction, vs 27-28 GB/s on GPU 1) has to move ~26 GB each way per step for the
rotated-row tensor-parallel all-gathers, and bf16 GEMM compute at the measured sustained rate
adds as much again; overlapping the two recovers only ~10 %.

Quality: `behemoth_eval ppl` on 4096 tokens of chat text: old binary 2.26121, new binary WMMA
path 2.26121 (identical bits), dense path 256-token chunks 2.26533, dense 2048-token chunks
2.26050 (with or without micro-batch overlap; after GEMM autotune 2.26062). All within the
run-to-run spread of summation-order changes; decode-sized steps are bit-identical to before.

## 1. Dense QTIP prefill

**Path.** For steps of >= `--qtip-dense-rows` (default 512) rows every linear is decoded once
per step into a bf16 scratch (`qtip::decode_dense`, one whole shard: 28672 x 12288 = 704 MB, the
largest gate/up shard) and multiplied with hipBLASLt bf16 x bf16 -> fp32 (fp32 accumulation,
`qtip::DenseMatmul`, `src/qtip_dense.hip`). Smaller steps keep the WMMA trellis kernel; decode
(<= 4 rows) keeps GEMV. fp32-payload projections (q/k/v, o, down) write the GEMM output directly
into the exchange buffer (no slice-reduce copy).

- `decode_dense` was rewritten for bandwidth: one 256-thread block decodes a 32 x 256 tile into
  LDS and writes it with 16-byte stores: 530 GB/s written (the old test-only kernel did 2-byte
  scattered stores). Decoding all four shards of a layer costs 2.6 ms per rank.
- hipBLASLt's first heuristic algorithm is 20-35 % slower than its best candidate for these
  shapes (gate/up 105 vs 136 TFLOP/s, down 85 vs 115): at startup each shape (q/k/v, o, gate/up,
  down) is timed for all heuristic candidates at the batch sizes the dense path sees (512, 1024,
  2048 for budget 2048); later GEMMs pick the algorithm tuned at the nearest batch >= theirs when
  the heuristic offers it for their size. Measured in isolation: 110-134 TFLOP/s.
- Crossover (`behemoth_qtip_bench`, one layer, one GPU, before autotune): matmul time per layer
  WMMA vs dense = 1.19 / 5.15 ms at 64 rows, 2.35 / 6.10 at 128, 4.68 / 7.00 at 256, ~9.4 / 9.52
  at 512, ~18.7 / 17.0 at 1024, ~37 / 31.1 at 2048. Dense wins from ~512 rows; with autotune
  the crossover moves to ~400, 512 was kept.

**Larger chunks.** `--token-budget` may now be up to 4096 for QTIP (AQLM stays <= 1024;
attention layouts up to 4096 queries). The lm_head/logit buffers are sized by the sampled rows
(`RuntimeOptions::sampled_rows`, default min(budget, 256); the scheduler caps requests per step
at `--max-batch-requests` 256), and QTIP no longer allocates the AQLM projection buffers, so a
2048 budget costs only 551 pages (-6.4 %) including the 704 MB scratch. Step-level timing
(`behemoth_prefill_bench`, one request, medians):

| rows @ context | WMMA | dense | dense + 2 overlapped micro-batches |
|---|---:|---:|---:|
| 256 @ 0 | 948 ms (270 tok/s) | - | - |
| 1024 @ 0 | 5572 ms (184) | 2983 ms (343) | 3277 ms (312) |
| 2048 @ 0 | 14566 ms (141) | 5713 ms (359) | 4981 ms (411) |
| 1024 @ 8192 | 6223 ms (165) | 3731 ms (274) | 3597 ms (285) |
| 2048 @ 8192 | 16024 ms (128) | 7222 ms (284) | 7491 ms (273) |
| 1024 @ 16384 | - | 5019 ms (204) | 4723 ms (217) |
| 2048 @ 16384 | - | 8665 ms (236) | 7939 ms (258) |

(The 256-row WMMA step was 1290 ms before this work at 2k context: the DMA transport and the
row-batched transforms below already cut it to ~950-1080 ms.)

**Transport.** Profiling the old path at a 209-row step: 622 ms of 1143 ms were collectives —
the staged host transport (blocking D2H, host barrier, H2D of *both* halves, per exchange).
Steps of >= 9 rows now use stream-ordered copy-engine all-gathers (`DmaChannel` in model.cpp):
the rank stream copies its half to pinned host memory in up to 4 pieces and records an event per
piece; a copy stream waits for the peer's piece events with cross-device `hipStreamWaitEvent`
(a command-processor barrier) and copies only the peer's half in, so D2H and H2D overlap
(PCIe is full duplex: 12.5 GB/s each way on GPU 0 concurrently). Two things measured and
rejected on the way: kernel-issued PCIe stores (the zero-copy publish kernels) reach only
~7 GB/s for large messages, and `hipStreamWaitValue32` is implemented as a spinning
`__amd_rocclr_streamOpsWait` kernel (5 s of them per 2048-row step on GPU 1) that steals CUs
from concurrent GEMMs. Decode steps (<= 8 rows) keep the fused zero-copy kernels unchanged.

**Micro-batch overlap.** Dense steps of >= 2x the dense threshold run as two micro-batches on two
streams per rank (own copy streams and DMA channels), issued step-interleaved per layer, coupled
only by events: the shared decoded weight (micro-batch 0 decodes after micro-batch 1's previous
GEMM), and the KV append of the earlier micro-batch before the later one's attention. Results
are bit-identical to the single dense step (full-model test). Gain is +9-15 % at 0 and 16k
context and -4 % at 8k (table above); `BEHEMOTH_QTIP_MICRO_BATCHES=4` was slower,
`GPU_MAX_HW_QUEUES=16` made no difference.

### Where prefill time goes (rocprofv3, 2048-row step, GPU 0)

Per step and rank the all-gathers move 144 KB per token per layer in each direction (q/k/v fp32
28 KB, attention bf16 12 KB, o fp32 24 KB, gate/up bf16 56 KB, down fp32 24 KB): 26 GB per
2048-token step. `bw`/`bw2` (in the bench dir): GPU 0 13.8 GB/s D2H / 13.4 H2D, 12.5 duplex;
GPU 1 28.0 / 26.2. Timeline of an overlapped step at ~2-4k context: 5.9 s wall, of which GPU 0
compute-busy 4.4 s (GEMM 3.25 s summed over concurrent kernels, attention 0.60, weight decode
0.34, transforms/epilogues 0.60), copy engines busy 3.5 s, copies with no kernel running 1.4 s.
So PCIe of GPU 0 alone bounds a 2048-token step at >= ~2.1 s (~1000 tok/s) even with perfect
overlap, and the bf16 GEMMs at the sustained rate under the 300 W cap need another ~2-3 s;
reaching 700-800 tok/s would need both perfectly hidden behind each other. At 16-32k context
prefill attention grows to ~1.5-3 s per step (the round-4 GQA prefill kernel runs at
~15-25 TFLOP/s) and dominates the decline from 401 to 254 tok/s.

## 2. Decode-first scheduling

`SchedulerPolicy` (scheduler.hpp), server flags:

- `--mixed-prefill-tokens N`: at most N prefill tokens in a step that also decodes (0 = old
  behavior: the token budget);
- `--prefill-interval K`: while requests decode, only every K-th step carries prefill tokens
  (fairness: the first mixed step after a decode-only phase prefills at once);
- `--max-batch-requests N` (256).

Decodes always enter a step first; without decodes prefill uses the full budget in
`--prefill-chunk` pieces; queued prompts prefill in arrival order. Defaults in code are the old
behavior (0 / 1); the recommended production flags are `--mixed-prefill-tokens 512
--prefill-interval 24`.

Trade-off curve (4 decoding streams at c=4, ITL 75-76 ms without prefill; a 16k prompt
arrives after 15 s; budget/chunk 2048):

| setting | decode ITL p50 / p95 / max during prefill | 16k TTFT | prefill tok/s | decode agg tok/s during |
|---|---:|---:|---:|---:|
| cap 0 (no limit) | 6132 / 7755 / 7755 ms | 52.1 s | 307 | 0.7 |
| cap 64 | 383 / 431 / 437 | 96.1 s | 167 | 10.5 |
| cap 16 | 155 / 174 / 182 | 155.4 s | 103 | 25.8 |
| cap 8 | 124 / 140 / 144 | 248.4 s | 64 | 31.3 |
| cap 4 | 117 / 130 / 135 | 327.7 s | 49 | 29.7 |
| cap 256, interval 8 | 76 / 1289 / 1430 | 110.1 s | 145 | 18.1 |
| **cap 512, interval 24 (chosen)** | **76-78 / 79-81 / 2589-2916** | **126-130 s** | **123-127** | 23 |
| before (budget 1024, chunk 256) | 1716 / 2003 / 2028 | 108.7 s | 147 | 2.4 |

Small mixed pieces cost ~5 ms per prefill token (WMMA rows + ~2 ms of PCIe per token + attention
over the long context), so a cap that keeps every step within 1.5x of the decode step allows
only ~4 tokens per step (49 tok/s prefill). The interval setting meets the p95 target (1.05x)
with 2.5x that prefill rate; the price is one ~2.5-2.9 s pause every ~24 tokens while a
512-token dense piece runs. For smooth streaming at the cost of p95 use `--mixed-prefill-tokens
16` (2x ITL, 103 tok/s). Two queued 8k prompts: decodes keep p95 90 ms, prompts finish FIFO.

Pure-decode concurrency (also part of this item):

- **Sampling.** At temperature > 0 without top-k the sampler fully sorted the vocabulary with
  an indirect comparator: 7.5 ms per token on this CPU (~22 ms per step in the engine).
  It is now a radix sort of (value desc, index asc) keys — the same permutation, so identical
  tokens (`sampler_equivalence` tests unchanged): 2.25 ms. Requests of a step are sampled in
  parallel on a small pool (each has its own RNG: identical results).
- **Bug fixed:** the engine indexed the logits rows by the position of a request in the step
  instead of its position among sampled requests; a non-final prefill chunk ahead of a sampled
  request (several queued prompts) sampled the wrong row.
- Steps of 9-16 rows use the DMA all-gathers instead of the staged host transport.
- c=4/8/16 aggregate 35.9/55.4/66.6 -> 52.3/79.7/140.3 tok/s (temperature 0.8).

Mixed steps: multi-query attention now splits keys (`prefill_attention_partition`) when the
largest request has few query tiles (a small prefill piece over a long context).

## 3. Batched Hadamard transforms

All RHT transforms/epilogues had one 256-thread block per (1024-chunk, row): each block re-read
the whole row (K = 12 or 28 times) for the `hadK` mix. Row-batched variants (one block per row,
whole row in registers, each input element read once, the K outputs computed from registers,
gate/up in two passes through `h`) are used for >= 64 rows (`--qtip-row-transforms`); the
o/down residual epilogue is fused with the next input transform (rms norm + SU + hadUt) from
registers, including across layers. The arithmetic per element is unchanged: bit-identical
(unit tests at 1 and 70 rows, full-model logits identical). Per layer and rank
(`behemoth_qtip_bench`):

| rows | per-chunk kernels | row-batched (+ fusion) | x88 per step |
|---:|---:|---:|---|
| 64 | 0.30 ms | 0.18 ms | 26 -> 15 ms |
| 256 | 0.83 ms | 0.44 ms | 73 -> 38 ms |
| 1024 | 3.44 ms | 1.72 ms | 303 -> 151 ms |
| 2048 | 7.02 ms | 3.44 ms | 618 -> 303 ms |

At decode sizes the per-chunk kernels stay (1 row: 0.05 vs 0.20 ms). The remaining cost is
memory traffic (gate/up epilogue 1.29 ms at 2048 rows = ~700 MB moved); a GEMM formulation of the
Kronecker factor would not reduce that.

## Tests

On the final build: `ctest` (core incl. new `scheduler_policy`, gpu, two-gpu transport, qtip)
pass; `behemoth_qtip_tests` with layer 40 + python reference vectors pass (new: dense prefill
matmul vs double reference and vs WMMA at 1..513 rows incl. piecewise scratch, rel_rms
<= 6e-6 like the WMMA kernel; fast decode bit-exact; row-batched transforms and the fused
residual+input bit-identical at 1/70 rows); `behemoth_block_tests` QTIP layers 0 and 87 and
AQLM layer 0 (host transport) pass; `behemoth_model_tests` AQLM and the new QTIP `prefill`
mode (WMMA vs dense: logits rel 3.7e-2 with equal argmax, the same order as re-chunking the
WMMA prefill alone, 3.4e-2; overlapped == single dense step bit for bit; mixed-step chunked
prefill) pass; GPU attention tests include the mixed-step key split. Python:
`tests/test_runtime_alignment.py` + `tests/test_training.py` results before/after are in the
"Python tests" line below. The fine-tuner's HIP bridge (qtip_training/native, which compiles
`runtime/src/qtip.hip` alone) still builds and has no hipBLASLt dependency (DenseMatmul lives
in `qtip_dense.hip`); the `LinearView` overloads are unchanged.

Python tests (`tests/test_training.py tests/test_runtime_alignment.py`, quantbench venv with
pytest/cmake from a private `--target` dir, HIP bridge built from the respective tree): 36
passed before (HEAD archive) and 36 passed after.

Pre-existing failures, identical with the production binary (cpp_runtime build): the AQLM
block test with the `zerocopy` transport (`max_abs=0.03 pass=0`; the host transport passes) and
`behemoth_model_tests` on the QTIP model ("Tuning snapshot omitted projections": the test
expects AQLM's eight autotuned projections). The AQLM model test and the QTIP `prefill` mode
pass.

## Known gaps

- 4x prefill not reached (1.7-2.1x); limits: GPU 0 PCIe bandwidth (13 GB/s), bf16 GEMM
  sustained rate, prefill attention at long contexts. Options not done: bf16 all-gather of
  q/k/v, o, down rotated outputs during prefill (-26 % PCIe bytes; changes numerics like the
  existing bf16 gate/up exchange), a faster GQA prefill attention kernel, overlap scheduling
  that keeps PCIe busy across stage boundaries.
- Mixed small prefill pieces are expensive (~5 ms/token); the chosen interval policy trades a
  ~2.5-2.9 s pause every ~24 decode tokens for p95 and prefill rate.
- GEMM autotune keys on (shape, batch); untuned batch sizes use the nearest tuned algorithm
  only when hipBLASLt's heuristic list contains it, else its first choice.
- The mixed-step attention key split and overlap micro-batches are QTIP-only; AQLM is unchanged.
