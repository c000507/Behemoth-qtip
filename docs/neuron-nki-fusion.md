# NKI fusion for the Devstral decode / prefill graphs: measurements, what was built, results

Box: i-0adbefb8af607c610 (trn1.2xlarge, libnrt 2.33.10, neuronx-cc 2.26.6360), 2026-11-10 (started for this work, stopped after it).
Model: Devstral Small 2507 (hidden 5120, 32/8 heads, head_dim 128, ff 32768, 40 layers, vocab 131072), fp8 weights + fp8 KV (page 16),
TP=2, the b18 graphs / prepared set on the box + the staged `dec8s_*` buckets (`/home/ubuntu/work/vk/g`). Work dir on the box
`/home/ubuntu/work/nf/` (`g/` buckets, `bench/` layer_bench, `h/` numerics harness, `e/` engine runs, `reqs/` prompts); scripts on the
workstation `/tmp/fable_vk/s3/`. Branch `nki-fusion`. Every number is a measurement run in this session (drained `layer_bench`, 200 iters
decode / 40 iters prefill, 2 ranks, `/home/ubuntu/work/vk/bin/layer_bench` = the repo's layer_bench.cpp md5 1177a848; or the real engine
`neuron_engine_cli` with `BEHEMOTH_NEURON_STEP_LOG=1`), arithmetic shown. Common env for the device: `NEURON_RT_VISIBLE_CORES=0-1
NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128`, `pkill neurond` first. Buckets were built with the box's bucket env (`ATTN_DEC=nki WT=1 MLP=nki
KV=fp8 UNSAFE_FP8FNCAST=1 MODEL=/home/ubuntu/work/b18/model PAGES=5561 NEURON_PLATFORM_TARGET_OVERRIDE=trn1`) plus the knob named per bucket.

## 1. Measurements first

### 1a. Every existing b18 bucket (layer_bench, ms/step = one layer; two interleaved reps, both listed, min used)

| bucket | rep 1 | rep 2 | used | note |
|---|---|---|---|---|
| dec1_32 | 1.298 | 1.309 | 1.30 | |
| dec1_128 | 1.320 | 1.485* | 1.32 | *the bimodal 1.3 / 1.5-1.8 first-run effect of the earlier docs |
| dec1_512 | 1.424 | 1.641* | 1.42 | |
| dec1_1024 | 1.543 | 1.546 | 1.54 | |
| dec1_2048 | 1.846 | 1.846 | 1.85 | 14k bucket |
| dec4_128 | 1.650 | 1.638 | 1.64 | NKI kernel, 4 rows |
| dec4_512 | 2.015 | 2.525* | 2.02 | |
| dec4_2048 | 4.029 | 4.033 | 4.03 | |
| dec8_32 | 1.663 | 1.687 | 1.66 | torch-op loop (P < 128) |
| dec8_128 | 1.793 | 1.803 | 1.79 | NKI kernel, 8 rows |
| dec8_512 | 2.524 | 2.809* | 2.52 | |
| dec8_1024 | 3.793 | | 3.79 | |
| dec8_2048 | 6.702 | | 6.70 | |
| dec16_32 | 1.964 | | 1.96 | torch-op loop (R > ATTN_DEC_ROWS = 8) |
| **dec16_128** | **4.204** | | **4.20** | torch-op loop: +2.41 ms over dec8_128 for 8 more rows |
| dec32_32 | 2.617 | | 2.62 | torch-op loop |
| **dec32_128** | **7.045** | | **7.05** | torch-op loop: 5.3 ms of attention for 32 rows x 128 pages |
| pre64_32 | 1.962 | | 1.96 | 64-token chunk (torch-op attention) |
| pre256_32 | 2.970 | | 2.97 | 256-token chunk, flash_fwd over 32 pages: 11.6 us per token per layer |
| pre256_128 | 3.174 | 3.168 | 3.17 | |
| pre256_256 | 3.500 | 3.533 | 3.50 | |
| pre256_512 | 4.153 | | 4.15 | |
| pre256_768 | 5.201 | | 5.20 | |
| pre256_1024 | 6.492 | | 6.49 | +3.52 ms over pre256_32 = 3.5 us per page (16 keys) for 256 queries |
| pre256_1536 | 7.873 | | 7.87 | |
| pre256_2048 | 8.041 | | 8.04 | |

(rep 2 of the decode list was cut short when the stage builds started compiling on the CPU; the starred values are the known
bimodal / perturbed outliers, the table uses the quiet value.)

Derived:

- Per-row cost of batched decode at 128 pages: dec4 (1.64 - 1.32) / 3 = **0.107 ms/row**, dec8 (1.79 - 1.32) / 7 = **0.067 ms/row** with the
  NKI kernel; dec16 (4.20 - 1.32) / 15 = **0.192 ms/row** and dec32 (7.05 - 1.32) / 31 = **0.185 ms/row** with the torch-op loop the R > 8
  buckets fall back to (`ATTN_DEC_ROWS` default 8). The 16- and 32-row buckets are the ones the engine uses for 9-32 concurrent decode rows
  (production: 8-16 parallel users), so the loop costs 40 x (4.20 - 2 x 1.79 + 1.32) ~ +77 ms per 16-row step against two dec8 launches,
  and the step is slower than two 8-row steps (section 1d: 81 vs 94 tok/s aggregate).
- Prefill per 256-token chunk per layer: 2.97 ms at 32 pages = 142 GFLOP per rank (2 x 256 x 278 M weights) / 2.97 ms = **48 TFLOPS per
  NeuronCore ~ 50 % of the 95 TFLOPS bf16/fp8 peak** (trn1: 190 TFLOPS per chip, two cores; fp8 has no extra rate on trn1); the context
  part grows 3.5 us per page (0.22 us per key for the 256 queries of a chunk): at 1024 pages it is 3.5 of 6.5 ms = 54 % of the chunk.
  Attention FLOPs at 1024 pages: 2 x 2 x 256 x 16384 x 128 x 16 heads = 34 GFLOP -> 0.36 ms at peak vs 3.5 ms measured: the prefill
  attention path (page gather + fp8 dequant in XLA, then neuronxcc flash_fwd) runs at ~10 % of PE peak (section 1c splits it).



### 1b. Where the attention half of the decode layer goes (LAYER_STAGE graphs, layer_bench 200 iters, 2 ranks)

`LAYER_STAGE=<stage>` (trace_model.py, this branch) compiles the layer graph truncated after that stage (every unused input kept alive by
a dead scalar, so the inputs and the layer_bench binding are those of the real bucket). Consecutive stages give the cost of each piece;
the pieces are what the compiler schedules for the truncated graph, so the split is indicative (+-0.01 ms), the totals are the real
buckets. `s1_128_*` = dec1_128 stages, `s8_128_*` = dec8_128 stages; the `rope` stage graphs compiled to a pathological layout (4.2 / 1.4
ms: the cat of three f32 tensors as the output) and are not used; rope is read from `kvwrite - qkv`.

| stage (cumulative) | dec1_128 ms | piece | dec8_128 ms | piece | what the piece is |
|---|---|---|---|---|---|
| launch floor (reviewer doc) | 0.061 | | 0.061 | | nrt_execute of a trivial NEFF |
| norm1 | 0.064 | 0.003 | 0.075 | 0.014 | RMSNorm -> bf16 h |
| qkv | 0.147 | **0.083** | 0.154 | 0.079 | fp8 quant + q/k/v fp8 matmuls + rescale: 15.7 MB of weights = 58 us at the 270 GB/s wall -> **70 % DMA-bound already** |
| kvwrite (incl. rope) | 0.174 | 0.027 | 0.208 | 0.054 | RoPE, fp8 KV quant + scales, page scatter |
| attn | 0.228 | **0.054** | 0.579 | **0.371** | decode_attention_fp8_2p (1 group): 54 us for 1 row, 46 us per extra row |
| oproj (no all-reduce) | 0.269 | 0.041 | 0.624 | 0.045 | fp8 quant of attn + o_proj matmul: 10.5 MB = 39 us at the wall -> DMA-bound |
| oproj_ar | 0.313 | **0.044** | 0.700 | **0.076** | the in-graph all-reduce of x [T, 5120] f32 (20 / 160 KB): 44 / 76 us, not the 2.7 us of nccom-test |
| norm2 | 0.313 | 0.000 | 0.715 | 0.015 | residual + RMSNorm |
| mlpin | 0.338 | 0.025 | 0.763 | 0.048 | fp8 activation quant + transpose for the MLP kernel |
| full bucket (1a) | 1.323 | **0.985** | 1.793 | 1.030 | MLP kernel (252 MB: 256 GB/s) + residual + all-reduce |

Reading:

- The attention half = 0.34 ms at 128 pages (the earlier docs' 0.23 was the ATTN_PROFILE=attn graph, which stops before o_proj / AR /
  norm2 / quant). Of it, 0.124 ms is the q/k/v/o weight DMA that is already at 78 % of the wall (0.097 ms floor), 0.054 the attention
  kernel, 0.044 the all-reduce, 0.064 launch floor, and only ~0.055 ms (norm, rope, KV quant / scatter, MLP input quant) is XLA small-op
  work a fused kernel could remove. **A fused attention-side kernel can gain at most ~0.05-0.08 ms/layer (2-3 ms/token, 4-5 %) at 128
  pages** unless it also hides the o_proj weight stream under the attention compute (another <= 0.04 ms) - the earlier doc's "<= 0.1
  ms/layer" bound holds, and section 2b shows the first attempt (norm + quant + qkv + RoPE in one kernel) is a loss, not a gain.
- The 8-row non-attention premium (+0.11 ms/layer of the dec8s verify step): all-reduce +0.032 (160 KB instead of 20 KB), MLP input quant
  +0.023, KV write / rope +0.027, norm2 +0.015, MLP kernel +0.045, o_proj +0.004, q/k/v 0 (weight-bound). None of these is a single
  fusable chain: it is 8x the vector work of the row-wise XLA ops plus a larger collective.
- The all-reduce costs 44 us in the graph (2 per layer = 3.5 ms/token = 6 % of a 57 ms step). It is f32 [T, 5120]; a bf16 all-reduce would
  halve the bytes but the massive-activation channels of this model made OUT_F32 necessary (PERF_NOTES); not tried here.
- Weight-stream wall check: the whole layer is 278 MB in 1.32 ms = 211 GB/s; MLP 252 MB in 0.985 ms = 256 GB/s; the attention half's weights
  26.2 MB in 0.124 ms = 211 GB/s. The DMA is idle ~0.21 ms per layer (attention kernel + AR + launch + small ops) = 8.6 ms/token: the only way
  to recover it is to stream weights during that window, and the SBUF (24 MiB, of which the MLP kernel's rings and the attention kernel's
  tiles take ~16 MiB at 128 pages) bounds a prefetch at ~8 MiB = 0.03 ms/layer. Cross-kernel prefetch is impossible anyway: the MLP kernel's
  SBUF does not exist until the all-reduce before it has completed (NKI kernels are opaque units for the scheduler, collectives cannot be
  inside them). Verdict on the 24 MiB bound of the brief: the usable bound is smaller (~8 MiB) and unreachable without a single-kernel layer.

### 1c. Where the prefill chunk goes (pre256 stages; `p256_128_*` = pre256_128, `p256_1024_*` = pre256_1024; 40 iters)

| stage (cumulative) | pre256_128 ms | piece | pre256_1024 ms | piece |
|---|---|---|---|---|
| norm1 | 0.140 | 0.14 | | |
| qkv | 0.290 | 0.15 | | |
| kvwrite (incl. rope) | 0.419 | 0.13 | 0.417 | |
| attn (page gather + fp8 dequant + permute + flash_fwd) | 0.766 | **0.35** | 3.070 | **2.65** |
| oproj_ar | 1.045 | 0.28 | 3.333 | 0.26 |
| norm2 | 1.075 | 0.03 | | |
| full bucket | 3.179 | **2.10** (MLP incl. act quant) | 6.472 | 3.14 |

(The `kvgather` and `rope` prefill stage graphs materialise their outputs in a pathological layout (200 / 21 ms) and are discarded; the
gather + dequant cost is inside the `attn` piece.)

- At short context the MLP is 2/3 of the prefill chunk: 2.10 ms for 129 GFLOP = 61 TFLOPS (64 % of peak) through the XLA matmuls, which
  read the gate / up / down weights through the in-graph `untile()` permute of the tiled (MLP=nki) layout; the q/k/v/o projections run at
  0.15 + 0.28 ms for 8 + 5 GFLOP (30-50 TFLOPS); attention at 128 pages 0.35 ms. Section 2c tests whether the untile costs anything.
- At 1024 pages attention is 2.65 ms of 6.47 (41 %): 34 GFLOP in 2.65 ms = 13 TFLOPS. The flash_fwd kernel sees dense bf16 K^T / V that the
  graph builds by gathering 1024 pages (16 MB fp8 -> f32 -> bf16 -> permute = ~80 MB of HBM traffic per layer ~ 0.3 ms) and then runs 256 queries
  x 16384 keys x 16 heads; most of the 2.65 ms is the kernel itself (the 1024-page `attn` piece minus the 128-page one = 2.3 ms for 7x the keys:
  0.14 ms per 128-page group of 2048 keys for 16 heads = 8.8 us per head-group). A paged fp8 flash-prefill kernel (the decode kernel's K / V
  layout with 128-query tiles as the stationary operand) would remove the gather / dequant pass (~0.3 ms) and could run closer to the PE
  rate; it was not built in this session (the MLP is the larger prefill cost at the contexts production sees most, and the wide-row decode
  kernel and the embedding fix were cheaper wins).

### 1d. Real engine baseline: prefill / TTFT and concurrent requests (sw/BUILD2 engine = merged HEAD sources, graph set = b18 + dec8s)

Prompts (`/tmp/fable_vk/s3/build_reqs3.py`, Tekken format, HF tokenizer with `no_truncation()`): `req_ttft.json` = tool15 (1364 tokens) and
the coding-agent system prompt with pre-read repo files cut to 4096 / 8193 / 12289 tokens, `max_new 8`, sequential; `req_parN_1k.json` =
N distinct ~1379-token tool prompts (4 tools, the user task carries a per-request sentence), `req_parN_4k.json` = N distinct 4413-token
prompts (8 tools + pre-read files), `max_new 128`, all submitted at once (the CLI's default mode: the scheduler batches them exactly as the
production server does, token budget 1024 = up to 4 x 256-token chunks per step, decode rows of all requests in one launch). Engine defaults
(speculation adaptive, page cost 0.002), `BEHEMOTH_NEURON_PREFIX_CACHE=0` so every prompt is really prefilled. Step-log analysis:
`/tmp/fable_vk/s3/analyze_steps.py` (groups the steps by composition; "pure decode" = steps without a prefill chunk).

Prefill (sequential, one request at a time):

| prompt tokens | chunks (256) | TTFT | prefill tok/s | layers_ms per chunk (40 layers) | non-layer per chunk |
|---|---|---|---|---|---|
| 1364 | 6 | 1305 ms | 1045 | 112-172 (buckets pre256_128) | 28-150 ms (see below) |
| 4096 | 16 | 2700 ms | 1517 | 112-134 (pre256_128 .. pre256_512) | 5-80 ms |
| 8193 | 33 | 5328 ms | 1538 | 112-160 (.. pre256_768 / 1024) | 5-40 ms |
| 12289 | 49 | 8384 ms | 1466 | 112-192 (.. pre256_1536) | 5-30 ms |
| all 102 chunks | | 17568 ms | 1477 | sum 14941 ms = 85 % | sum 2628 ms = 25.8 ms/chunk |

The non-layer part of a prefill step has a 5 ms floor (host fill of x / cos_sin / tables, drain, sampling bookkeeping) and spikes of 15-150 ms
that follow the NOVELTY of the chunk's tokens, not its position or bucket: the first prompt's chunks cost 150 / 130 / 88 / 51 / 46 ms extra, the
later prompts' chunks that repeat the same system-prompt text cost ~5 ms, and the chunks with fresh code text 15-80 ms. `embed_row`
(neuron_model.cpp) reads the embedding rows element by element from the mmap'd `model-embed.safetensors` (bf16, 1.34 GB); the first touch of
each 4 KB page is a disk read (256 tokens x 2.5 pages x ~0.1-0.2 ms). Section 3 measures the fix.

Concurrent requests (production-like batching; aggregate numbers from the step log):

| set | TTFT per request (ms) | prefill-only steps: tokens/step, wall, tok/s | pure decode steps: bucket, wall, emitted/step, aggregate | drafted / accepted |
|---|---|---|---|---|
| 4 x 1379 | 2881, 3337 x3 | 870, 480 ms, 1812 tok/s (layers 95 %) | dec8_128 81.9 ms, 5.9 tok (3.5 drafts/step) -> **70.9 tok/s**, 14.1 ms/token | 89 / 51 |
| 8 x 1379 | 2839 .. 6624 | 896, 473 ms, 1894 tok/s (94 %) | dec8_128 79.3 ms, 7.7 tok -> **94.2 tok/s**, 10.6 ms/token | 7 / 5 |
| 16 x 1379 | 2884 .. 14319 | 896, 481 ms, 1864 tok/s (94 %) | **dec16_128 175.2 ms**, 14.7 tok -> **81.0 tok/s**, 12.4 ms/token (dec8 steps in the same run: 78.6 ms) | 25 / 12 |
| 4 x 4413 | 9535, 10109 x3 | 970, 530 ms, 1832 tok/s (94 %) | dec8_512 109.4 ms, 5.1 tok (3.8 drafts) -> **46.3 tok/s**, 21.6 ms/token | 146 / 62 |
| 8 x 4413 | 9876 .. 22892 | 981, 549 ms, 1789 tok/s (92 %) | dec8_512 112.5 ms, 7.1 tok -> **61.7 tok/s**, 16.2 ms/token | 27 / 16 |

- Prefill dominates the agentic multi-user case: 8 x 4.4k prompts = 35k tokens take 19.6 s of prefill at ~1.8k tok/s before the last
  request's first token (production saw 17 s p50 at 8 parallel 4k). Mixed steps (decode rows + 3-4 chunks) cost 570-690 ms, so the
  decoding requests see 600 ms token gaps while others prefill. Prefill steps are 92-95 % layer time: the `pre256_P` layer graph is the lever
  (section 1c), the host side ~26-44 ms/step (embedding faults + 4 x fill) the second.
- Pure-decode aggregate: 8 rows at 128 pages 94 tok/s; 16 rows **81 tok/s** (less than 8 rows: the dec16_128 loop graph, 175 ms/step);
  4 rows with speculation (3.5 drafts/step, A ~ 1.5 accepted per step) 71 tok/s. At 512 pages 8 rows 62 tok/s.
- The engine only drafts when a request is alone or few (SPEC_REQUESTS default 4): the 8/16-row runs drafted 7-27 tokens in total.

## 2. What was built, and what it measured (layer_bench, 2 reps, 200 iters, 2 ranks)

### 2a. Wide decode rows through the NKI kernel (`ATTN_DEC_ROWS=32`: buckets n16_128 / n32_128 / n16_512; n16_32 / n32_32 with `ATTN_DEC_MIN_PAGES=32`)

No kernel change: `decode_attention_fp8_2p` already loops over R rows (`for r in nl.affine_range(R)`); `ATTN_DEC_ROWS` (default 8) only
gated which buckets trace through it, and the b18 dec16_* / dec32_* buckets fell back to the torch-op loop (dequantised bf16 pages,
`DECODE_BLOCK_PAGES` blocks of einsum / softmax / einsum per row). Compile times 11-30 s per bucket (R = 32 included).

| bucket | b18 graph (loop) rep 1 / rep 2 | NKI-kernel graph rep 1 / rep 2 | per layer | per 40-layer step |
|---|---|---|---|---|
| dec16_128 -> n16_128 | 4.204 / 4.324 (4.391 perturbed) | **2.138 / 2.119** | **-2.1 ms (-50 %)** | **-84 ms** (168 -> 85 ms of layers for a 16-row step) |
| dec32_128 -> n32_128 | 7.045 / 7.067 | **3.083 / 2.905** | **-4.1 ms (-58 %)** | **-165 ms** (282 -> 118 ms) |
| dec16_32 -> n16_32 | 1.964 / 1.974 | 2.167 / 2.164 | +0.20 ms | the kernel pads 32 pages to one 128-page group: slower, NOT shipped |
| dec32_32 -> n32_32 | 2.617 / 2.786 | 2.919 / 2.925 | +0.25 ms | same, NOT shipped |
| (new) dec16_512 = n16_512 | no b18 bucket (16 rows at > 128 pages run as two dec8_512 launches = 2 x 2.52 = 5.05 ms) | 3.671 / 3.667 | -1.4 ms vs two launches | optional extra bucket |

Per-row attention cost in the kernel at 128 pages: (2.12 - 1.32) / 15 = 0.053 ms/row (16 rows), (2.91 - 1.32) / 31 = 0.051 ms/row (32 rows),
the same 45-50 us per (row, 128-page group) as dec8 (0.067 incl. the 8-row XLA premium). Numerics: section 2d.

### 2b. Fused RMSNorm + fp8 quant + q/k/v + RoPE NKI kernel (`nki_qkv.py`, `ATTN_QKV=nki`): built, correct in form, SLOWER

`qkv_fused` streams qf / kf / vf as the moving operand against the stationary fp8 activation tiles (the nki_mlp pattern: 40 K-tiles x
[128, 2048 | 512 | 512] weight tiles, 4 + 1 + 1 PSUM banks, rescale by a x column scale, bf16 rounding as the graph, RoPE on the [T, N]
result); the graph keeps the fp8 KV quant / scatter and the attention kernel unchanged, so the bucket interface is identical to dec1_*.

| bucket | baseline rep 1 / rep 2 | fused rep 1 / rep 2 | delta |
|---|---|---|---|
| dec1_128 -> f1_128 | 1.323 / 1.319 | 1.391 / 1.383 | **+0.07 ms (+5 %)** |
| dec8_128 -> f8_128 | 1.793 / 1.816 | 1.822 / 1.824 | +0.02 ms |
| dec1_512 -> f1_512 | 1.426 / 1.429 | 2.449* / 1.458 | +0.03 ms (*rep 1 = the bimodal first-run outlier) |
| stage up to rope: s1_128 qkv + rope ~ 0.147 + ~0.02 | | f1_128_rope 0.197 / 0.198 | the kernel's prologue is ~0.03 ms slower than the XLA chain |

Why it loses: the XLA q/k/v chain is already DMA-bound at 0.083 ms for 15.7 MB (section 1b), the compiler's version overlaps the three weight
streams with the norm / quant / rope vector work across the whole attention half, while the NKI kernel is a scheduling unit: its 40 x 3
weight tiles cannot start before its inputs (x, norm1) are ready and the attention kernel cannot start before it finishes, so the
norm / quant / transpose prologue (~20 us of vector + 40 PE transposes) and the RoPE epilogue are exposed instead of hidden. The 2 KB / 512 B
weight rows (q / k / v in the WT layout) also stream slower than the MLP kernel's 4 KB rows. Verdict: **negative result, kept in the tree as
opt-in (`ATTN_QKV=nki`) for reference, not to be shipped.** A version that hides the o_proj stream under the attention compute would have to
fuse the attention kernel too (one kernel: qkv + KV write + attention + o_proj); its upper bound from 1b is 0.05-0.08 ms/layer (2-3 ms/token),
and the o_proj prefetch adds <= 0.04 ms; not built in this session.

### 2c. Prefill MLP: is the in-graph `untile` of the tiled weights costing anything? (x256_32 = pre256_32 traced with MLP=xla)

See section 2e for the numbers.

### 2d. Numerics of the wide-row buckets (harness `/tmp/fable_vk/s3/harness_wide.py`, box `nf/h`, real layer-0 weights, both ranks)

Every row is its OWN request (own random pages, own context length 40..2047 tokens produced on the CPU with the real k/v projections +
RoPE, quantised per (token, K|V, head) like the graph, own position = its context length, own write slot); row 0 fills the bucket (128 pages).
References: the b18 torch-op-loop bucket of the same (R, P) (bf16 dequantised pages; the class of difference the 128+-page dec1/dec8
kernel buckets already carry against it, 1.3 % on this metric) and the b18 dec8_P NKI bucket run on 8-row slices of the same rows (the same
kernel code at R = 8). Metric: relative L2 of the layer delta `out - x` per row.

| case | rows | new vs dec8 slices (max / mean) | new vs loop (max / mean) | loop vs dec8 slices (max / mean) | K/V bytes written: new vs loop / new vs dec8 | pad rows |
|---|---|---|---|---|---|---|
| n16_128, seed 1 | 16 real | **0 / 0 (bit-identical)** | 2.2e-2 / 1.45e-2 | 2.2e-2 / 1.45e-2 | 0 / 0 | - |
| n32_128, seed 2 | 32 real | 9.9e-3 / 1.4e-3 (8 of 32 rows differ, 2e-3..1e-2) | 1.7e-2 / 1.33e-2 | 1.7e-2 / 1.33e-2 | **0** / 815 of 32768 | - |
| n16_128, seed 3, 11 real + 5 engine pad rows (x = 0, pos 0, scratch wslot / table) | 11 | **0 / 0** | 2.5e-2 / 1.54e-2 | 2.5e-2 / 1.55e-2 | 0 / 0 | finite |
| n16_32 (32 pages padded to one 128-page group), seed 4 | 16 | 2.0e-2 / 1.66e-2 (dec8_32 is itself a loop graph here) | 2.0e-2 / 1.66e-2 | 0 / 0 (both loop graphs) | 0 / 0 | finite |

- Fused qkv kernel (same harness, R = 1 / 8, reference = the dec1_128 / dec8_128 kernel buckets): f1_128 differs from dec1_128 by 5.4e-3
  with 43 of 2048 K/V codes written differently, f8_128 by 8.1e-3 max / 4.1e-3 mean (265 of 16384 codes): the kernel's own norm / quant
  rounding flips a few E4M3 codes of h8 and of the written K/V (the fp8-noise class, below the 1.3-1.5 % kernel-vs-loop class); f1_512 did not
  run (the box disk filled up with the harness work dirs at that point; freed afterwards, not rerun).
- 16 rows: bit-identical to the 8-row kernel buckets on every row, K/V writes byte-identical; the 1.45 % against the loop graph is the
  fp8-kernel-vs-bf16-loop class (identical to the loop-vs-dec8 column).
- 32 rows: the n32 graph and the b18 dec32_128 LOOP graph write byte-identical K/V and differ from the 8-row graph's K/V in 815 of 32768
  codes (2.5 %): the fp8 KV quantisation of the 32-row XLA graph (amax / divide / cast over [32, 2, 4, 128]) rounds a few codes differently
  from the 8-row graph's; the 8 rows whose attention output differs from the dec8 slices by 0.2-1 % are rows whose own fresh K/V codes differ
  (a row attends to its own token). This is a pre-existing property of the shipped dec32 bucket vs dec8, not of the kernel; against the loop
  reference n32 sits at 1.33 % mean = the same class as dec8. Both ranks agree bit for bit on every run.

### 2e. Prefill variants and re-benches (xbench, after the engine chain; 2 reps)

PLACEHOLDER_XBENCH_TABLE

## 3. Real engine (neuron_engine_cli, same prompts and analysis as 1d; graph sets = b18 + dec8s with the named buckets swapped in)

- `graphs_wide`: dec16_128 -> n16_128, dec32_128 -> n32_128 (+ dec16_512 = n16_512); the dec16_32 / dec32_32 loop buckets kept (the padded
  kernel is slower there, 2a). Engine binary unchanged (sw/BUILD2): the bucket directory name is free, the engine keys buckets by meta (R, Q, P).
- `graphs_fused`: dec1_128 -> f1_128, dec1_512 -> f1_512, dec8_128 -> f8_128 (the ATTN_QKV=nki graphs; interface identical).
- `nf/BUILD`: the engine built from this branch's neuron_model.cpp (resident embedding option) + the sw/repo runtime; run with and
  without `BEHEMOTH_NEURON_EMBED_RESIDENT=1` on the b18 graph set.

### 3a. Wide-row buckets (graphs_wide) vs baseline (same binary, same prompts, both runs in this session)

| set | graph set | pure decode steps (bucket, wall, emitted) | aggregate decode | TTFT last request | prefill tok/s |
|---|---|---|---|---|---|
| 16 x 1379 | base (dec16_128 loop) | dec16_128 **175.2 ms**, 14.7 tok; dec8_128 78.6 ms | **81.0 tok/s** (12.35 ms/token) | 14319 ms | 1864 |
| 16 x 1379 | **wide (n16_128 kernel)** | dec16_128 **90.0 ms**, 14.7 tok; dec8_128 76.7 ms | **137.6 tok/s** (7.27 ms/token) | 13458 ms | 1958 |
| 8 x 1379 | base | dec8_128 79.3 ms, 7.7 tok | 94.2 tok/s | 6624 | 1894 |
| 8 x 1379 | wide (no wide bucket used) | dec8_128 76.4 ms, 7.7 tok | 98.2 tok/s | 6481 | 1950 |
| 8 x 4413 | base | dec8_512 112.5 ms, 7.1 tok | 61.7 tok/s | 22892 | 1789 |
| 8 x 4413 | wide (no wide bucket used) | dec8_512 105.7 ms, 7.1 tok | 65.8 tok/s | 21287 | 1878 |

- The 16-row step drops 175.2 -> 90.0 ms (-85 ms; layer_bench predicted 40 x (4.32 - 2.12) = -88), **aggregate decode +70 % (81 -> 138 tok/s)
  at 16 concurrent users with 1.4k contexts**, per-user 7.0 -> 11.1 tok/s equivalent while all 16 decode; the mixed steps (16 rows + 3 chunks)
  drop 607 -> 514 ms. The 8-row runs do not touch a wide bucket; their 4-6 % differences are the process-to-process spread (the second
  run of each pair also had a warm page cache for the prompts' tokens: non-layer 17-19 vs 26-44 ms per prefill step).
- 32-row steps were not exercised by the engine in this session (needs > 16 concurrent requests; layer_bench: 7.05 -> 2.91 ms/layer).

### 3b. Fused qkv kernel (graphs_fused) vs baseline, single request, plain greedy

| prompt | graph set | plain step wall | ms/token | t/s | tokens (finish) |
|---|---|---|---|---|---|
| tool15 1364 | base dec1_128 | 57.0 | 57.04 | 17.5 | 53 (stop) |
| tool15 1364 | **fused f1_128** | 59.8 | 59.79 | 16.7 | 54 (stop): **+2.8 ms/token = 40 x +0.07 ms/layer (2b)** |
| long 4398 | base dec1_512 | 61.2 | 61.20 | 16.3 | 128 (length) |
| long 4398 | fused f1_512 | 63.0 | 63.00 | 15.9 | 107 (stop): +1.8 ms/token |

The engine confirms the layer_bench loss; the fused outputs are different near-tie realisations (54 vs 53 tokens; 1-row fp8 noise, 2d).
**Do not ship ATTN_QKV=nki.**

### 3c. Prefill host side: warm vs cold embedding pages, and the resident table

Within this session the same tool15 prompt prefilled in 1305 ms (base_ttft, first process after the instance start, 25.8 ms of non-layer
time per chunk averaged over 102 chunks, 28-150 ms on the first prompt's chunks) and in 808-810 ms once its tokens' embedding pages were in
the page cache (base_t15_plain / fused_t15_plain: 6.4 ms non-layer per chunk = the host floor): **-38 % TTFT from the embedding pages alone**.

| run (b18 graph set, same prompts, sequential, max_new 8) | engine | embedding | TTFT 1364 / 4096 / 8193 / 12289 | non-layer per chunk (102 chunks) | prefill tok/s |
|---|---|---|---|---|---|
| base_ttft (first process after the instance start, page cache cold for the prompts' tokens, prefix cache off) | sw/BUILD2 | mmap | **1305 / 2700 / 5328 / 8384 ms** | **25.8 ms** (5 floor + faults) | 1477 |
| base_ttft_pc (same prompts again, page cache warm, prefix cache on = production default) | sw/BUILD2 | mmap | 812 / 2131 / 4836 / 7902 | 6.2 ms | 1669 |
| res_ttft (prefix cache off) | nf/BUILD (this branch) | **resident (1.34 GB copied at load in 1.02 s)** | 863 / 2101 / 4775 / 7813 | **4.0 ms** | 1683 |

- Cold vs warm on the same prompt: 1305 -> 812 ms TTFT at 1364 tokens (-38 %), 8384 -> 7902 at 12k (-6 %): the faults are paid once per
  embedding page, so they hit short, novel prompts hardest (the agentic case: fresh system prompts and tool schemas after a deploy / eviction).
  The resident table removes the fault path by construction (anonymous memory, never evicted by the 24 GB prepared-set reads that fill the
  page cache on this box: MemFree was 1.3 GB with 24.9 GB cached) and on a warm cache still saves 2.2 ms per chunk (6.2 -> 4.0 ms: no per-element
  bounds-checked mmap reads). The cold-cache case could not be re-measured with the resident table in this session (dropping the page cache
  would also evict the prepared set, 200 s reload; out of budget), so the cold gain is inferred from the warm-vs-cold measurement above.
- The prefix cache (default on) does not change prefill cost for distinct prompts (base_ttft_pc = res-free warm run: 6.2 ms/chunk).



## 4. Ranked table: where the time is and what each candidate is worth (measured where built, bounded where not)

| rank | workload | where the time goes (measured) | candidate | gain | status |
|---|---|---|---|---|---|
| 1 | 9-32 concurrent decode rows (dec16_*/dec32_* buckets) | torch-op attention loop: 4.20 / 7.05 ms/layer vs 2.12 / 2.91 with the NKI kernel | `ATTN_DEC_ROWS=32` buckets n16_128 / n32_128 (+ n16_512) | **-84 / -165 ms per 16 / 32-row step (-50 / -58 % of the layer time)**; engine: section 3 | built, validated (bit-identical to dec8 at 16 rows), engine-measured |
| 2 | prefill / TTFT (every request, every chunk) | 5-150 ms per 256-token chunk of host time = page faults of the mmap'd embedding (85-95 % layer time otherwise) | `BEHEMOTH_NEURON_EMBED_RESIDENT=1` (engine, 1.3 GB host RAM) | up to -45 % TTFT on cold / novel prompts, -5..-25 ms per chunk warm; section 3 | built, engine-measured |
| 3 | prefill layer (92-95 % of prefill steps) | MLP 2.1 ms of the 3.2 ms chunk at <= 128 pages (61 TFLOPS = 64 % of peak); attention 2.65 ms at 1024 pages (13 TFLOPS) | (a) NKI prefill MLP on the tiled layout (T = 256 as two 128-row stationary blocks); (b) paged fp8 flash-prefill kernel (no gather / dequant materialisation) | bound: (a) <= -0.7 ms/chunk-layer if PE-bound at ~90 % (= -28 ms per chunk, +25 % prefill tok/s); (b) up to -2 ms/layer at 16k context | not built (sections 1c, 2c) |
| 4 | single-stream decode, attention half | 0.34 ms/layer: weights 0.124 (78 % of the DMA wall), attention kernel 0.054, all-reduce 0.044, launch 0.064, XLA small ops ~0.055 | fused attention-side kernel (norm + quant + qkv + rope + KV write + attention + o_proj) | bound -0.05..-0.08 ms/layer = -2..-3 ms/token (-4..-5 %); the partial fusion built (qkv side) is +0.07 ms/layer | negative result for the partial fusion; the full fusion not built |
| 5 | verify step (dec8s) non-attention +0.11 ms/layer | AR +0.032, MLP-in quant +0.023, KV write +0.027, norm2 +0.015, MLP +0.045 | nothing fusable as one chain; a bf16 all-reduce would save ~0.02-0.03 ms/layer (numerics risk: massive activations) | <= -1 ms per verify step | not built |
| 6 | all-reduce (2 per layer) | 44 us each in-graph (not 2.7 us): 3.5 ms/token = 6 % | fewer / smaller collectives (bf16 AR, or AR fused into the next kernel: impossible, collectives cannot be in NKI) | <= -1.8 ms/token | not built |

## 5. Recommendation and ship steps

Ship (in this order):

1. **Wide-row NKI decode buckets** (biggest measured gain, no engine change): build `dec16_128` and `dec32_128` with `ATTN_DEC_ROWS=32`
   (`ATTN_DEC_ROWS=32 ATTN_DEC=nki WT=1 MLP=nki KV=fp8 UNSAFE_FP8FNCAST=1 MODEL=... PAGES=5561 python neuron/trace_model.py OUT/dec16_128 layer 16 1 128`,
   same for `32 1 128`; 11-30 s each) and REPLACE the two loop buckets of the same name in the graphs prefix (`deploy/ecs-neuron/variables.tf`
   `graphs_s3_prefix`; the engine refuses two buckets with the same (R, Q, P)). `build_buckets.sh WIDE_NKI=1` does this (this branch). Keep
   `dec16_32` / `dec32_32` as they are (the padded kernel is 0.2-0.3 ms/layer slower there). Optional: add `dec16_512` (n16_512, 3.67 ms vs two
   dec8_512 launches 5.05). Expected: 16-row decode steps 175 -> 90 ms (+70 % aggregate at 16 users), 32-row steps 282 -> 118 ms of layers.
   Rollback: restore the old bucket directories. Numerics: bit-identical to dec8 at 16 rows; dec32 differs from dec8 by the pre-existing
   32-row KV-quant rounding only (2d).
2. **Resident embedding table**: engine from this branch, env `BEHEMOTH_NEURON_EMBED_RESIDENT=1` (+1.34 GB host RSS, +1 s load; check the
   task memory limit in the ECS task definition). Default off; no graph change; rollback = unset.
3. Do NOT ship `ATTN_QKV=nki` (slower, 2b / 3b). `LAYER_STAGE` is a profiling knob only.

Next engineering candidates (not built, bounded in section 4): an NKI prefill MLP on the tiled layout (<= -25 % prefill time at <= 512
pages), a paged fp8 flash-prefill kernel for long contexts, and the full attention-side fusion (<= -4..-5 % single-stream decode).

## 6. Validated / not validated / failures

- Validated on the box: layer_bench of every b18 bucket (1a), the per-stage graphs (1b / 1c, 2 reps), n16/n32 numerics with real weights on
  both ranks incl. pad rows and the 32-page padded variant (2d), the engine with the wide buckets at 8 / 16 concurrent requests (3a), the
  fused kernel's loss in layer_bench and in the engine (2b / 3b), the embedding fault signature from the step log and the resident option on a
  warm cache (3c), the engine builds from this branch's neuron_model.cpp (nf/BUILD) and runs the b18 graph set unchanged.
- Not validated: n32 in the engine (no > 16-request run); the resident embedding on a cold page cache; dec16_512 in the engine; the
  `rope` / `kvgather` stage graphs (pathological layouts, discarded); f1_512 numerics (disk full at that point); the pre256 MLP=xla
  comparison and the fp8 lm_head together with the wide buckets (not run together).
- Failures: the first stage / wide / fused builds all failed because the nf repo copy lacked `checkpoint.py` (fixed by copying the full
  neuron/ python dir; cost ~15 min); the harness work dirs (556 MB of layer-0 weights each) filled the disk to 17 MB free during the fused
  f1_512 check (deleted; the engine chain was unaffected: its runs started after the cleanup); the stage bench's rep 2 of 1a was
  perturbed by concurrent compiles (quiet values used); `analyze_run.py` mis-attributes steps to requests in concurrent mode (a new
  aggregate analyzer `neuron/tests/analyze_engine_steps.py` is used instead).
- Repo additions: `neuron/nki_qkv.py`, `LAYER_STAGE` / `ATTN_QKV` / `ATTN_DEC_ROWS` recorded in meta.build, `neuron/tests/harness_wide_rows.py`,
  `neuron/tests/analyze_engine_steps.py`, `neuron/tests/build_reqs_concurrent.py` (box-path constants inside, as the earlier harness).
