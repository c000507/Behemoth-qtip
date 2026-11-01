# Prefill round 2: FP8 K/V GEMM, PCIe/NUMA, fp32 audit

Date: 2026-11-01. Machine: 192.168.0.229 (X12DAi-N6, 2x Ice Lake, 2x R9700, ROCm 7.2, no P2P).
Checkpoint: `Behemoth-R1-123B-v2-QTIP-2bit-gmcot-fp8kv` (separate FP8 K/V projections).
Baseline: `812dfe1` (`runtime/build-rocm`); new: branch `perf/prefill` (`runtime/build-prefill`).
Server flags: production command (`BEHEMOTH_ATTN_QP=bf16`, budget/chunk 2048, mixed 512,
interval 24) plus `--prefix-cache-slots 0` for cold prefill. Raw data, scripts and traces:
`/home/charles/prefill_r2/` (`bench2.py` there also counts `reasoning_content` deltas; the
old client counted only `content`, so think-block outputs showed 0 tokens).

## Result

| | 10-27 report (old ckpt) | 812dfe1 (fp8kv) | perf/prefill | vs 812dfe1 |
|---|---:|---:|---:|---:|
| TTFT 2k (tok/s) | 4.99 s (401) | 5.59-6.34 s (316-358) | 4.15-4.94 s (405-481) | -22 % |
| TTFT 8k | 21.7 s (369) | 24.8 s (322) | 18.6 s (430) | -25 % |
| TTFT 16k | 51.3 s (312) | 51.7 s (309) | 38.2 s (419) | -26 % |
| TTFT 32k | 126.1 s (254) | 111.9 s (286) | 85.0 s (376) | -24 % |
| 4 decoding + 16k prompt: 16k TTFT | 126-130 s | 123.4 s | 108.1 s (96.7 s with `BEHEMOTH_QTIP_BF16_EXCHANGE_ALL=1`) | -12 % (-22 %) |
| same: decode ITL p50 / p95 / max during | 76-78 / 79-81 / 2589-2916 | 70.1 / 72.2 / 2479 | 70.2 / 72.9 / 1993 (64.9 / 68.2 / 1750) | |
| decode c=1 ms/token | 44.3 | 44.3 | 44.1 (43.9) | = |
| decode c=4 / c=8 ITL p50 | 76.1 / 99.7 | 67.3 / 86.1 | 67.5 / 86.1 (62.6 / 82.7) | = (-7 % / -4 %) |
| KV capacity (tokens) | 128,848 | 119,792 | 119,264 | -0.4 % |

The fp8kv checkpoint had regressed short-prompt prefill against the 10-27 numbers: its FP8
K/V projections at prefill sizes ran in a naive 16x64-tile WMMA kernel at ~2.6 TFLOP/s,
1.65 s of every 2048-row step. That is the main fix here.

Engine step (`behemoth_prefill_bench ... 2048 CTX 3`, dense+overlap, medians, ms):

| ctx | 812dfe1 | perf/prefill | tok/s |
|---:|---:|---:|---:|
| 0 | 6451 | 4858 | 317 -> 422 |
| 8192 | 6287 | 4854 | 326 -> 422 |
| 16384 | 7524 | 5384 | 272 -> 380 |
| 32768 | 8124 | 6555 | 252 -> 312 |

## Where a 2048-row step goes (rocprofv3, GPU 0, 2 micro-batches)

Kernel times are summed over the two micro-batch streams (they run concurrently); "busy" is
the union. Under rocprof the steps are 6.65 / 7.74 s (base) and 5.09 / 6.21 s (new).

| | base ctx 0 | new ctx 0 | new ctx 16k |
|---|---:|---:|---:|
| hipBLASLt GEMMs (q/k/v, o, gate/up, down; new: + fp8 K/V) | 2913 | 2946 | 2943 |
| fp8 K/V old WMMA kernel / new expand + scale | 1645 | 20 | 20 |
| QTIP weight decode | 308 | 304 | 301 |
| attention | 89 | 88 | 1234 |
| gate/up epilogue / residual epilogues / transforms / qkv epilogue | 320 / 158 / 88 / 53 | 317 / 161 / 87 / 53 | 310 / 166 / 89 / 53 |
| rope+quant, norm, other | 69 | 89 | 70 |
| **kernels busy (union)** | 4347 | **2778** | 3901 |
| copies D2H / H2D busy | 1439 / 1480 | 1439 / 1478 | 1438 / 1477 |
| any copy busy / copy with no kernel running | 2862 / 1969 | **2864 / 1972** | 2861 / 1942 |
| idle (no kernel, no copy) | 338 | 340 | 369 |

GPU 1: kernels busy 3802 ms at 16k, copies 780 / 788 ms (twice GPU 0's rate), idle 1347 ms:
it waits for GPU 0.

After the fix GPU 0 is link bound at short context: 2.86 s of copies vs 2.78 s of kernels in
a 5.09 s step, with 1.97 s of copies exposed. At 16k attention (1.2 s summed) is the next
largest compute term.

## PCIe / NUMA

Bytes: every pure-prefill exchange is already bf16 (`--qtip-bf16-prefill-rows 512`):
q/k/v 14 KiB, attention 12, o 12, gate/up 56, down 12 = 106 KiB per row and layer, per rank
and direction: **19.6 GB each way per 2048-row step** (~153 GB each way for a 16k prompt).
Measured in the trace: GPU 0 19.6 GB / 1.44 s = 13.6 GB/s D2H, 13.2 GB/s H2D; GPU 1
25.1 / 24.8 GB/s.

Cause of GPU 0's half rate: **its root-port link trained at PCIe Gen3.**

```
GPU 0 33:00.0 (CPU1 Slot5, NUMA 0): 30:02.0 root port LnkCap 16GT/s, LnkSta 8GT/s x16;
      31:00.0 (card switch upstream) LnkSta "8GT/s (downgraded)"; endpoint 33:00.0 32GT/s
GPU 1 cc:00.0 (CPU2 Slot2, NUMA 1): c9:02.0 16GT/s x16; ca:00.0 "16GT/s (downgraded)"
```

Root-port target speed is 16 GT/s, no AER errors are logged, slot is "PCI-E 4.0 X16". The
endpoint's own link (32 GT/s, to the card's on-board switch) is what the old startup
diagnostic printed, which hid this; it now also prints `slowest_upstream_link=` (8.0 GT/s
for GPU 0).

NUMA is not a factor (refuted): `dup.hip` microbenchmark (`/home/charles/prefill_r2/`),
16 MB messages, pinned buffers with `hipHostMallocNumaUser` under `numactl --membind 0` and
`--membind 1` (placement verified with numastat): identical rates, GPU 0 13.6 / 13.2 GB/s
one-way, 9.9 GB/s per direction duplex; GPU 1 27.3 / 25.4, duplex 20.6. Default (non-NUMA)
allocation gives the same. UPI is not limiting.

Duplex in the runtime: in the trace GPU 0's D2H and H2D overlap for only 53 ms of 1.4 s.
The microbenchmark with the same DmaChannel protocol (piece events, cross-device waits)
does overlap (16 MB all-gather 1.78 ms vs 2.5 ms serial) when both ranks reach the
exchange together; in the model the ranks reach exchanges at different times and the
copies serialize. Also checked and ruled out: stream count/HW-queue sharing (up to 12
streams: no change; `GPU_MAX_HW_QUEUES=16` slower), piece count (4 is best: 6 / 8 / 16
pieces 3-7 % slower end to end), 3 or 4 micro-batches (-4 % / -18 % at ctx 0).

**Recommendation (hardware, not done):** get GPU 0's link to Gen4. Options, least
invasive first: check the BIOS per-slot link-speed setting for CPU1 Slot5 (Auto/Gen4);
retrain the link (`setpci -s 30:02.0 CAP_EXP+0x10.w=0x20:0x20`, then re-read LnkSta;
with the GPU idle); reseat the card / riser; move GPU 0 to CPU1 Slot1 (16:02.0, free x16).
Expected: GPU 0 copy time 2.86 -> ~1.45 s per 2048-row step, i.e. ~5.1 -> ~3.7-4.0 s
(+25-35 % prefill), and decode exchange latency down too.

## fp32 audit (what is materialized in memory or on the wire)

Bytes per rank per 2048-row pure-prefill step unless noted; a 16k prompt is ~7.8 such steps.
Line numbers: `runtime/src/model.cpp` at the branch head.

| fp32 tensor | where | bytes | status |
|---|---|---:|---|
| q/k/v, o, down exchange payloads, **pure prefill** (>= 512 rows) | `qtip_exchange` 1192-1276 | 0 (already bf16) | bf16 since round 5 |
| same, **mixed decode+prefill steps** (<= 512 prefill rows + decodes) | `bf16_projection` 1776 | +38 KiB/row/layer: 1.77 GB each way per 516-row mixed step (~0.13 s on GPU 0) | **new flag** `BEHEMOTH_QTIP_BF16_EXCHANGE_ALL=1`: mixed 16k TTFT 108.1 -> 96.7 s, max decode pause 1993 -> 1750 ms |
| same, **decode steps** | same | +38 KiB/row/layer: 3.3 MB (c=1) / 27 MB (c=8) per step | same flag: c=4 ITL 67.5 -> 62.6 ms, c=8 86.1 -> 82.7, c=1 44.1 -> 43.9 |
| gate/up, attention payloads | | 0 (bf16) | bf16 |
| hipBLASLt GEMM outputs | `qtip_project` 1308-1393 | 0 in bf16 steps (bf16 plans write the exchange block) | fp32 only in fp32-payload steps (above) |
| WMMA split-K partials (steps < 512 rows) | `qtip_partial` 544/860, summed by `reduce_publish` 1200/1233/1261 | slices x rows x out; 235 MB reserved at budget 2048 | **keep**: split-K partial sums; bf16 partials would add rounding per slice. VRAM opportunity: dense steps only use rows x 1024 of it (FP8 K/V acc), the reservation could shrink to the WMMA maximum (~511 rows) |
| new FP8 K/V GEMM accumulator | `fp8_kv_dense` 1504 | 8.4 MB/layer, 0.74 GB written+read per step (~2.5 ms) | **keep**: per-channel scale must be applied before the single bf16 rounding; removable via a hipBLASLt alpha-vector epilogue (~2 ms/step, not worth it now) |
| exchange host buffers (DmaChannel, zero-copy slots) | 142, 293 | typed float, carry bf16 bytes | n/a (byte buffers) |
| lm_head logits (`raw`, `gathered`, pinned `logits_host`) | 850-851, 2152 | 64 KiB + 128 KiB per sampled row (1 row per prefill step) | **keep**: sampling precision; bf16 logits step is 0.125 at \|logit\| 16-32 (changes temperature sampling and greedy ties); <0.1 ms/step |
| KV scales `k_scale`/`v_scale` | 523 | 4 B per 128 B fp8 vector (3 % of KV) | keep: part of the FP8 KV format; bf16 would gain ~1.5 % KV capacity and change KV numerics |
| `query_scale` (fp8 Q scales) | 535 | 393 KB/layer | keep: scale, negligible |
| SU/SV sign/scale vectors | 512-513 | constants | keep |
| attention split-key partials | `attention_workspace` 856 | <= 32 MB; partition 0 (unused) in overlapped prefill steps | keep: softmax merge state |
| staged HostTransport buffers + host sum | 293 | not used in production (zero-copy + DMA) | n/a |

Quality of the new flag: held-out perplexity (4096 / 2048 tokens of `ppl_text.txt`):
chunk 16 (decode-sized steps) 2.58447 -> 2.58465 (+0.007 %); chunk 256 (WMMA steps)
2.24448 -> 2.24634 (+0.08 %). Greedy (5 short + 2k + 8k prompts, 80 tokens): 5 of 7
identical, 2 diverge after 106 / 84 characters, both coherent. It is off by default
because it changes decode numerics; recommended on after a longer quality check.

## Changes (commit 47d81f4 on `perf/prefill`)

1. **Dense FP8 K/V GEMM** (`fp8_kv_dense`, default on, `BEHEMOTH_FP8_KV_DENSE_ROWS`,
   default 65, 0 restores the old kernels): steps of >= 65 rows expand the stacked K|V
   shard to bf16 exactly (`fp8_expand_pair`, 25 MB per micro-batch scratch), run one
   hipBLASLt GEMM (autotuned at startup, 145-162 TFLOP/s) into the fp32 accumulator and
   store bf16(acc x scale) (`fp8_scale_pair`, the GEMV kernels' rounding). Rows <= 64
   keep the row-batched GEMVs (decode unchanged). Results differ from `fp8_linear` only in
   summation order (18 of 66,560 / 165 of 614,400 outputs differ by an ulp on layer 40);
   perplexity 2.24532 -> 2.24538 at chunk 2048.
2. `BEHEMOTH_QTIP_BF16_EXCHANGE_ALL=1` (default off): see the audit.
3. Startup diagnostic: `slowest_upstream_link=` per GPU.

Tests (build-prefill): `behemoth_tests` (104,477 assertions) pass; `behemoth_gpu_tests`
and `--two-gpu` pass; `behemoth_qtip_tests` with the real fp8kv layer 40 pass (new: dense
FP8 K/V pair vs reference at 65 and 600 rows, vs `fp8_linear`).

## Also measured, not adopted

- `--token-budget 4096 --prefill-chunk 4096`: engine step +22 % at ctx 0, +7 % at 8k, +5 %
  at 16k; server TTFT 8k 18.6 -> 16.8 s, 16k unchanged, but 32k **184.9 s** (85.0 s at 2048,
  not investigated; the engine bench at 4096 rows and 24-28k context is normal, 337-363
  tok/s), 9 % fewer KV pages and 41 s autotune. Keep 2048.
- DMA pieces 6/8/16, 3/4 micro-batches: slower (above).

## Remaining ideas, ranked (expected TTFT gain / effort)

1. **GPU 0 link to Gen4** (above): +25-35 % prefill, also decode/mixed; BIOS/slot work only.
2. **Duplex-aligned exchange**: copies serialize (1.97 s exposed per step). If the ranks'
   D2H/H2D overlapped as in the microbenchmark (9.9 vs 13.4 GB/s one-way on GPU 0), copy
   time drops ~2.9 -> ~2.0 s: up to ~0.9 s per 2048-row step (~15 %). Needs scheduling work
   (e.g. issue micro-batch 0's H2D against micro-batch 1's D2H, or let GPU 1 lead so GPU 0's
   receive overlaps its send). Medium effort, uncertain.
3. **Enable `BEHEMOTH_QTIP_BF16_EXCHANGE_ALL`** after longer quality checks: mixed 16k
   TTFT -10 %, c=4/c=8 decode -7/-4 %. Trivial.
4. **FP8 (e4m3 + per-row scale) gate/up exchange**: -26 % exchange bytes, ~0.6 s per step on
   GPU 0; needs a perplexity study (rotated activations, 2-bit weights). Medium.
5. **Long-context attention**: 1.2 s summed per step at 16k, ~2.5 s at 32k. Medium-high.
6. Sequence-parallel MLP with host-resident peer weights: only pays at >= 4096-row chunks
   (weight traffic ~11.6 GB H2D per step regardless of rows). High effort; not recommended
   while the link is Gen3.
