# Shared-page-table verify attention (dec8s buckets): design, validation, measurements

Box: i-0adbefb8af607c610 (trn1.2xlarge, libnrt 2.33.10, neuronx-cc 2.26.6360), 2026-11-10. Model: Devstral Small 2507 (fp8 weights,
fp8 KV, TP=2, 40 layers), the b18 graphs / prepared set on the box. Every number here is a measurement run in this session (drained
layer_bench, 200 iters, 2 ranks; or the real engine with `BEHEMOTH_NEURON_STEP_LOG=1`), with the arithmetic shown. Work dir on the box:
`/home/ubuntu/work/vk/` (copy of the b18 repo + the files of this branch; buckets in `vk/g`, graph dir `vk/graphs`, harness `vk/h`,
engine runs `vk/e`). Branch `verify-kernel`.

## 1. The problem (docs/neuron-reviewer-experiments.md 1a / 1b)

A speculative verify step runs the request's scheduled token + 7 drafts as 8 rows of a `dec8_P` layer graph. `make_inference_batch`
(neuron_engine.cpp) emits ONE request with `count = 1 + drafts` tokens and one page-table row; `NeuronModel::forward` (neuron_model.cpp,
the row path for `count <= kRowPathTokens = 8`) turns it into 8 rows of one decode launch, each row with the SAME `table_row(L, i, req)`
copy of the request's table, its own `pos` and `wslot = page * 16 + pos % 16`; pad rows (when fewer than R rows) get `x = 0`, `pos = 0`,
`wslot = scratch * 16` and an all-scratch table. The graph scatters every row's K/V (`index_put` at `wslot`) BEFORE the attention, and the
attention mask is `key position <= row position` (`bias` built from `pos`), so row i sees the committed context plus draft rows `<= i`
(positions `p_0 .. p_i`, all inside the request's pages), never rows `> i`: causality inside the verify window is the position mask alone.

`decode_attention_fp8_2p` (nki_decode.py) loops `for r in nl.affine_range(R)` and, per row, gathers every 128-page group of K (pass 1) and
V (pass 2) and runs the per-group work (16 `nc_transpose` of K slices, 16 score matmuls, scale / bias / max, 16 P^T transposes, 64 PV
matmuls, exp / quant) for that row's 16 query rows (4 kv heads x 4 query heads) in a [16, 512] score tile. For 8 rows that is 8 gathers
and 8x the per-group PE work of a dec1 step: measured verify premium 40 x (dec8 - dec1) = +15 / +19 / +44 / +99 ms per step at 32 / 128 /
512 / 1024 pages, which makes drafting a loss at 4.4k context (69.7 vs 61.4 ms/token).

Per 128-page group of the existing kernel (per row): K gather 4 x 256 KB (one DMA per kv head, 128 pages x 2 KB), V the same; `ksc`,
`vsc`, `bias` loads of [16, 2048] f32 (128 KB each); PE: 16 K transposes [128 pages, 128 d], 16 score matmuls [16 x 128] x [128 x 512],
16 P^T transposes, 64 PV matmuls [16 x 128 pages] x [128 pages x 128 d] fp8; Vector / Scalar: scale-multiply + bias-add over [16, 2048],
max, exp with sum, V-scale multiply, max / divide / fp8 cast, copies PSUM -> SBUF. ~45 us per group (docs/neuron-attention-fusion.md).

## 2. Design: one gather per group for all rows

All R rows of a verify step share one page table, so `decode_attention_fp8_2p_shared` (nki_decode.py, same module) gathers each group
ONCE and batches the rows into the PARTITION axis of every per-group tile:

* query rows = partitions: `NP = R x 16` (R requests x 4 kv heads x 4 query heads) <= 128. Row `16 r + 4 h + i` is request r, kv head h,
  query head i (the layout of the existing kernel's output, so the caller's reshape is unchanged).
* stationary q tile per kv head h: `qp[:, h * NP : (h + 1) * NP]` = [128 d, NP] bf16, zero except the REP columns of (r, h) at `16 r + 4 h`;
  score matmul `s_ps[NP, 512] += qp_h^T x K_h^T` as before, so the rows of other heads get 0 from K_h and the 4 heads accumulate into one
  PSUM tile. PE time is set by the moving operand (512 key columns), not by the 16 -> 128 stationary rows.
* per-row causal / length mask: `bias [G, NP, 2048]` f32 built by the caller from each row's position (`key position > pos_row -> -30000`),
  loaded per group as [NP, 2048] (one DMA of 1 MB instead of 8 x 128 KB). K / V scales are per key, the same for every row: `ksc / vsc
  [G, NP, 2048]` with the 16-row block repeated over the R rows (the XLA expand writes 1 MB per group, as the old graph did for 8 rows).
* pass 1 (scores of all rows, one K gather per group, per-group max), global max per row, pass 2 (exp against the global max, V-scale
  weighting, per-row per-group fp8 quantisation of P, P^T slices [128 pages, NP], 64 PV matmuls [NP x 128 pages] x [128 pages x 128 d]
  fp8 x fp8, head pick with the [NP, 4] row mask, per-group PV slices summed at the end): the two-pass structure of `decode_attention_fp8_2p`
  unchanged, every tile widened from 16 to NP partitions. No loop-carried DMA destinations in affine loops; the gathers are the same
  `nisa.dma_copy(src=cache[tab[i1, h * G + g], jf])` form.
* SBUF per partition (trn1: 24 MB / 128 = 192 KB), G groups of 128 pages: `s_all` [NP, G x 2048] f32 = 8 KB x G (64 KB at 1024 pages),
  `kraw` / `vraw` 8 KB each (4 heads x 2 KB, up to 2 buffers), `ksc_t` + `bias_t` 16 KB, `p` / `pw` 8 KB each, `p8` 2 KB, `pt8` [128, 16 x NP]
  fp8 = 2 KB, `acc_all` 512 B x G, `qp` 1 KB, tables 32 B x G: ~120-160 KB at G = 8 incl. double buffering. These are PER-PARTITION
  sizes and identical to the old kernel's (which already used 16 of the 128 partitions with the same free sizes), so the footprint is the
  same and G = 8 (1024 pages) compiles; G = 16 (2048 pages, the 14k bucket) would need `s_all` = 128 KB and was not built.
* PSUM (8 banks x 2 KB per partition): `s_ps` [NP, 512] f32 = 2 KB (1 bank), `kt_ps` [128, 512] f32 = 2 KB, `pv_ps` [NP, 128] f32 = 512 B,
  transposes [128, NP] f32 = 512 B: as before.
* arithmetic per row is the SAME as the old kernel's in the SAME order (per-element PE accumulations over the 128-d / 128-page contractions,
  the same 4 quarter sums, the same per-group fp8 quantisation and group sum), so the result is expected bit-identical to dec8 (it is,
  section 4).

Graph / runtime wiring (opt-in, nothing changes without it):

* `trace_model.py`: `ATTN_DEC_SHARED=1` (with the bucket defaults ATTN_DEC=nki KV=fp8) routes Q = 1 decode graphs with `R x 16 <= 128`
  rows through `flash_decode_nki_fp8_shared` (gathers from `table[0]`, builds `bias` per row and the repeated scales) for ANY P (pads to
  whole 128-page groups, so dec8s_32 also uses the kernel where dec8_32 used the torch-op loop); `meta.json` gets `"shared_table": true`.
  `build_buckets.sh SHARED_VERIFY=1` adds `dec8s_32/128/512/1024`.
* `neuron_model.cpp`: `Meta.shared_table` (default false), buckets of the same (R, Q, P) but different `shared_table` are distinct, and
  `pick(..., shared)` only returns shared-table buckets when asked: `forward` asks for one only for a multi-row launch whose rows ALL belong
  to one request (the verify step), falling back to the per-row buckets otherwise. A graph set without `shared_table` buckets runs exactly
  as before. The step log (`last_profile`) gains `"decode_bucket": "dec<R>_<P>[s]"`.

## 3. Diffs (branch verify-kernel)

* `neuron/nki_decode.py`: + `decode_attention_fp8_2p_shared` (~75 lines) next to `decode_attention_fp8_2p`.
* `neuron/trace_model.py`: + `ATTN_DEC_SHARED`, + `flash_decode_nki_fp8_shared`, the dispatch line in `flash_paged`, `meta['shared_table']`.
* `neuron/cpp/neuron_model.cpp`: `Meta.shared_table`, meta parse, bucket dedupe key, `pick(..., shared)`, the single-request test in the
  decode row loop, `decode_bucket` in the profile (~15 lines).
* `neuron/build_buckets.sh`: `SHARED_VERIFY=1` block.
* `neuron/cpp/neuron_engine.cpp`: unchanged (the gate change is a separate commit, section 7).

## 4. Validation (numerics, on the box, real layer-0 weights, both ranks)

Harness `/home/ubuntu/work/vk/h/harness.py` (workstation copy `/tmp/fable_vk/harness.py`): the same layer inputs through the old `dec8_P`
NEFF, the new `dec8s_P` NEFF and, row by row with the cache carried over (row i runs with rows > i NOT yet in the cache), the `dec1_P`
NEFF, all via `neuron/cpp/neff_io.cpp` (`NEFF_IO_STATES=2`, 2 ranks). Inputs: the real layer-0 shards of the b18 prepared set (all 14
projection tensors + norms, per rank), residual rows = real embedding rows of random tokens, a committed context of CTX tokens whose K/V
were produced on the CPU with the real k/v projections + RoPE at their positions and quantised per (token, K|V, head) to E4M3 exactly as
the graph does, the 8 verify rows at positions CTX .. CTX+7 writing into the table's pages (`wslot`), PAGES = 5561 (the b18 pool, scratch
5560), table pages a random subset. Metric: relative L2 of the layer delta `out - x` per row; the fp8-noise reference is old dec8 vs dec1.

| case | P | ctx (pages used / P) | rows | new vs old (max / mean) | old vs dec1 (max / mean) | new vs dec1 (max / mean) | K/V bytes written: new vs old / vs dec1 |
|---|---|---|---|---|---|---|---|
| a | 128 | 1000 (63 / 128) | 8 | **0 / 0** (bit-identical) | 3.6e-7 / 6.2e-8 | 3.6e-7 / 6.2e-8 | 0 / 0 |
| b | 128 | 2000 (126 / 128) | 8 | 0 / 0 | 3.2e-7 / 4.0e-8 | 3.2e-7 / 4.0e-8 | 0 / 0 |
| c (pad rows) | 128 | 1000 | 5 real + 3 pad (x = 0, pos 0, scratch wslot / table) | 0 / 0 | 2.4e-7 / 1.1e-7 | 2.4e-7 / 1.1e-7 | 0 / 0; pad rows finite |
| d (not a multiple of 128) | 512 | 4400 (276 / 512 = 3 groups, 2.16 used) | 8 | 0 / 0 | 2.1e-7 / 7.4e-8 | 2.1e-7 / 7.4e-8 | 0 / 0 |
| e | 1024 | 12000 (751 / 1024, G = 8) | 8 | 0 / 0 | 2.9e-7 / 3.6e-8 | 2.9e-7 / 3.6e-8 | 0 / 0 |
| f | 32 | 300 (20 / 32, padded to one 128-page group) | 8 | 1.8e-2 / 1.3e-2 | 9.2e-7 / 4.9e-7 | 1.8e-2 / 1.3e-2 | 0 / 0 |

- At 128 / 512 / 1024 pages the new kernel is BIT-IDENTICAL to the old dec8 (per row the same arithmetic in the same order, batched over
  partitions), and both are within 3.6e-7 of the row-by-row dec1 reference: the dec1 reference runs row i WITHOUT rows i+1..7 in the cache,
  so any leak of later rows through the mask would show here (a 0.7 % weight shift at 1000 keys ~ 1e-2 on this metric); 1e-7 is the
  all-reduce / fusion noise between the 1-row and 8-row graphs.
- Causal negative control (case a inputs, every row told `pos = CTX + 7`, i.e. allowed to see all 8 rows): rows 0-6 change by 5.0 / 2.6 /
  2.2 / 2.2 / 1.6 / 2.4 / 3.4 % (relative to the layer delta), row 7 by exactly 0 (it already sees everything) -> the mask is live and the
  rows attend to the fresh K/V of the other rows exactly when allowed; old and new kernels agree bit-for-bit on the control too.
- Case f (32 pages): `dec8_32` and `dec1_32` do NOT use the NKI kernel (P < ATTN_DEC_MIN_PAGES = 128: the bf16 torch-op loop over
  dequantised pages), `dec8s_32` runs the fp8 kernel over one padded group, so 1.3 % is the fp8-kernel-vs-bf16-loop difference (the class
  of difference the 128+-page buckets already have against the loop; the kernel's own result at 32 pages was not separately referenced).
- Verified: K/V rows written by the graph (cache + scales) are byte-identical between old / new / dec1; both ranks return the same x.

## 5. layer_bench (200 iters, 2 ranks, drained; `/home/ubuntu/work/vk/bin/layer_bench` = the repo's layer_bench.cpp md5 1177a848)

All runs of this session are listed (`bench.txt` rep 2 ran while g++ built the engine, `bench2.txt` was a quiet sweep); the "used" value
is the median of the quiet runs. The dec1 buckets show a bimodal 1.30 / 1.80 ms behaviour when run first after an idle period (the
reviewer doc's "first run outlier"); dec8 / dec8s never did.

| bucket | runs (ms/step) | used |
|---|---|---|
| dec1_32 | 1.304, 1.841*, 1.804*, 1.308 | 1.31 |
| dec8_32 | 1.675, 1.680, 1.679, 1.676, 1.674 | 1.68 |
| **dec8s_32** | 1.500, 1.508, 1.499, 1.505, 1.508 | **1.505** |
| dec1_128 | 1.321, 1.326, 1.330, 1.800*, 1.332, 1.826* | 1.33 |
| dec8_128 | 1.803, 2.555*, 1.798, 2.011, 1.801, 1.810 | 1.80 |
| **dec8s_128** | 1.506, 1.815 (g++ running), 1.509, 1.500, 1.508, 1.507 | **1.507** |
| dec1_512 | 1.768*, 1.426, 1.431, 1.424, 1.435 | 1.43 |
| dec8_512 | 2.534, 2.534, 2.529, 2.533, 2.532 | 2.53 |
| **dec8s_512** | 2.121 / 2.346 (g++ running), 1.649, 1.673, 1.653 | **1.653** |
| dec1_1024 | 1.793 / 1.670 (g++ running), 1.544, 1.555, 1.545 | 1.545 |
| dec8_1024 | 6.023 (g++), 3.618, 3.639, 3.621, 3.879 | 3.63 |
| **dec8s_1024** | 1.915, 1.924, 1.918, 1.936, 1.924 | **1.924** |
| ATTN_PROFILE=attn dec1_128 / dec8_128 / dec8s_128 | 0.232, 0.231, 0.244, 0.238, 0.232 / 0.595, 0.581, 0.583, 0.582, 0.581 / 0.296, 0.297, 0.297, 0.297, 0.296 | 0.232 / 0.582 / 0.297 |
| ATTN_PROFILE=noattn dec1_128 / dec8_128 | 1.284, 1.275, 1.634*, 1.278, 1.278 / 1.392, 1.611*, 1.387, 1.390, 1.629* | 1.278 / 1.390 |

Derived (arithmetic shown; old premiums from the reviewer doc's table 1a, same method):

| pages | dec8s - dec1 (ms/layer) | verify premium new = 40 x | old premium (dec8 - dec1) x 40 | saved per verify step |
|---|---|---|---|---|
| 32 | 1.505 - 1.31 = 0.195 | **7.8 ms** | 0.37 x 40 = 14.8 | 7.0 |
| 128 | 1.507 - 1.33 = 0.177 | **7.1 ms** | 0.47 x 40 = 18.8 | 11.7 |
| 512 | 1.653 - 1.43 = 0.223 | **8.9 ms** | 1.10 x 40 = 44.0 | 35.1 |
| 1024 | 1.924 - 1.545 = 0.379 | **15.2 ms** | 2.09 x 40 = 83.4 (reviewer doc: 98.8 with its 4.02 median) | 68-84 |

Where the remaining 0.18 ms/layer at 128 pages goes (ATTN_PROFILE halves, same bucket set): attention half dec8s - dec1 = 0.297 - 0.232 =
**0.065 ms** (the kernel: 3 x 1 MB bias / scale loads + XLA expands per group instead of 3 x 128 KB, and the 128-column stationary
loads of 96 PE ops per group), non-attention half dec8 - dec1 = 1.390 - 1.278 = **0.112 ms** (8-row RMSNorm / q k v / RoPE / KV scatter /
o_proj / MLP activation quant: not attention, not addressed here); 0.065 + 0.112 = 0.177 = the measured difference. The target "dec1 + 0.12
ms/layer" is met for the attention part (0.065) and missed overall (0.18-0.22) by the 8-row non-attention cost. At 1024 pages the
kernel adds 0.38 - 0.11 = 0.27 ms = 8 groups x ~34 us: ~0.03 ms/group over dec1's per-group cost (bias / scale traffic 3 MB per group
at ~270 GB/s = 11 us, plus the wider PE stationary loads).

## 6. Real engine (neuron_engine_cli, sequential, greedy, max_new 128, BEHEMOTH_NEURON_STEP_LOG=1, same prompts as the reviewer doc)

Graph dir `/home/ubuntu/work/vk/graphs` = symlinks to every b18 bucket + `dec8s_32/128/512/1024`; `new` = `vk/BUILD/neuron_engine_cli`
built from this branch's cpp (+ the b18 runtime/), `base` = the SAME build procedure on the b18 cpp sources (no shared buckets, b18
graphs). Prompts: `req_tool.json` = [long 4398, short 2243, long 4398] tokens (512-page buckets), `req_tool15.json` = [short15 1364 x 2]
(128-page buckets). ms/token = decode wall after the first token / tokens (analyze_run.py, as in the reviewer doc). The step log's
`decode_bucket` field confirms the bucket of every launch (`dec8_512s` = the shared bucket; the plain steps `dec1_512`).

| ctx | config | engine | verify steps / plain | A (accepted per verify) | verify wall ms | plain wall ms | ms/token | t/s | vs the old-kernel best |
|---|---|---|---|---|---|---|---|---|---|
| 4398 | plain (SPEC_TOKENS=0) | base | 0 / 127 | - | - | 61.06 | 61.06 | 16.4 | (doc 61.4) |
| 4398 | plain | new | 0 / 127 | - | - | 61.21 | 61.21 | 16.3 | same graph path |
| 4398 | forced (PAGE_COST=0) | base (old dec8_512) | 60 / 18 | 0.50 | 107.5 | 59.9 | 69.72 | 14.3 | doc 69.75 reproduced |
| 4398 | **forced** | **new (dec8s_512)** | 60 / 18 | 0.50 | **72.7** | 60.0 | **50.40 / 50.23** (2 runs) | **19.8** | **+21 % over plain, +38 % over old forced** |
| 4398 | adaptive (default cost 0.01) | new | 0 / 127 | never drafts | - | 61.2 | 61.2 | 16.3 | gate off at 512 pages, as before |
| 4398 | adaptive, PAGE_COST=0.002 | new | 24 / 71 and 41 / 48 (repeat) | 0.54 / 0.93 | 73.5 | 60.8 | 56.3 / 46.5 | 17.7 / 21.5 | gate still skips ~half the steps |
| 2243 | forced | base | 25 / 7 | 0.80 | 107.6 | 59.1 | 59.69 | 16.7 | |
| 2243 | **forced** | **new** | 25 / 7 | 0.80 | 72.7 | 59.4 | **42.97** | **23.3** | +42 % over plain (61.0) |
| 1364 | plain | new | 0 / 52 | - | - | 57.1 | 57.09 | 17.5 | (doc 57.05) |
| 1364 | forced | base (old dec8_128) | 27 / 7 | 0.70 | 78.7 | 56.0 | 47.50 / 47.34 | 21.1 | doc 47.49 reproduced |
| 1364 | **forced** | **new (dec8s_128)** | 27 / 7 | 0.70 | **67.3** | 56.4 | **41.74 / 41.52** | **24.0** | +37 % over plain, +14 % over old forced |
| 1364 | adaptive (0.01) | new | 7 / 41 and 12 / 33 | 0.71 / 0.58 | 69.6 / 67.6 | 56.8 | 53.1 / 51.5 | 18.8 / 19.4 | gate skips most steps (doc 54.3) |
| 1364 | adaptive, PAGE_COST=0.002 | new | 27 / 7 | 0.70 | 67.5 | 56.0 | 41.80 / 41.56 | 23.9 / 24.1 | = forced (overhead 1.8 ms < 10 % of plain: always try) |

- Verify premium in the engine: 512 pages 72.7 - 60.0 = **+12.7 ms** per verify step (was +47.5; layers alone 8.9 from section 5, the rest
  the 8-row head + host); 128 pages 67.3 - 56.4 = **+10.9 ms** (was +22.4; layers 7.1). Break-even acceptance = premium / plain step =
  0.21 (512) and 0.19 (128) accepted drafts per step; measured A = 0.50-0.93.
- Output text: the new forced runs produce EXACTLY the texts of the old-kernel forced runs (all 220-char prefixes identical, same token
  counts 109 / 53 / 109 and 54 / 54, same finish reasons), consistent with the bit-identical layer outputs. Forced vs plain differ after ~60
  chars as the reviewer doc already noted (8-row vs 1-row graph near-tie flips, 1e-7 per layer). The 0.002-gate 4.4k repeat ran a different
  verify/plain step mix and diverged into a 128-token answer (same first 60 chars).
- Expected production t/s with the shared buckets and drafting on at agentic contexts: **19.8-24 t/s at 1.4-4.4k** with the stock
  prompt-lookup drafter (vs 16.3-17.5 plain, 14.3-21.1 forced today), i.e. the lower end of the reviewer doc's 22-26 estimate (that
  estimate assumed +4.8 ms per verify step; measured +10.9-12.7 because of the 8-row non-attention cost).

## 7. Adaptive gate (BEHEMOTH_NEURON_SPEC_PAGE_COST) with the new verify cost

The gate (neuron_engine.cpp:423-428) models the verify overhead as `page_cost x drafts x bucket_pages` and skips drafting when
`overhead > 0.1 x plain_step_ms && alpha x plain_step_ms < 2 x overhead`. Measured overheads with dec8s: 128 pages 10.9 ms, 512 pages 12.7
ms (engine), 1024 pages ~15 ms layers + head (layer_bench only): nearly flat in pages, so no page cost fits both ends: 0.01 (today's
default) gives 9.0 / 35.8 / 71.7 ms, i.e. it never drafts at 512+ pages where the measured gain is the largest (+21 % at A = 0.5).
Proposed default **0.002**: 128 pages -> 1.8 ms (< 10 % of a 56 ms step: always try = the forced behaviour, measured 41.8 ms/token = 23.9
t/s); 512 -> 7.2 ms (> 6.1: drafts while alpha >= 2 x 7.2 / 61 = 0.24, vs the break-even 0.21; measured 56.3 / 46.5 ms/token = 17.7 / 21.5
t/s on the two 4.4k runs, both better than plain 61.2, below forced 50.4 because the EMA dips below 0.24 after two empty steps); 1024 ->
14.3 ms (alpha >= 0.47; break-even ~0.3 with a ~19 ms premium: conservative; not engine-tested). 0.001 would make 512 pages "always try"
(= the forced 19.8 t/s measured) at the price of 1024 pages drafting from alpha >= 0.24 (below its break-even 0.3): not proposed without
a 1024-page engine run. Separate commit; `BEHEMOTH_NEURON_SPEC_PAGE_COST` still overrides it, and WITHOUT dec8s buckets the old premiums
apply, so a deployment that changes the default must ship the buckets (0.002 with the old dec8_512 would draft at +47.5 ms per step).

## 8. Validated vs not

- Validated: compile of dec8s_32/128/512/1024 (PAGES 5561); numerics on both ranks with real layer-0 weights at 32/128/512/1024 pages
  incl. pad rows, a non-multiple-of-128 page count and the causal control; drained layer_bench on both ranks; the real engine end to end
  (real weights, 40 layers, both prompts, forced / adaptive / plain, outputs identical to the old-kernel forced runs); the engine falls
  back to the per-row buckets for plain steps (`dec1_*` in the log) and, by construction, for multi-request launches (not exercised:
  the CLI ran one request at a time; the pick rule is a 3-line `all_of` on the request id).
- Not validated: multi-request steps with the shared buckets present (the fallback path, code-reviewed only); 2048 pages / 14k context
  (dec8s_2048 not built: `s_all` would be 128 KB per partition; the 14k verify premium is from the 32->1024 curve only); layer 1-39
  weights in the harness (layer 0 only; the engine run covers all layers at the token level); seeded / non-greedy sampling; the 0.002 gate
  at 1024 pages; the `dec4_*` buckets (no shared variant built: `R x 16 = 64` partitions would work the same way).
- Not done: shrinking the remaining +0.065 ms/layer of the kernel (in-kernel iota mask instead of the 1 MB bias load; the 3 x 1 MB
  per-group scale / bias traffic) and the 8-row non-attention +0.11 ms/layer, which is now the larger part of the verify premium.

## 9. Ship steps

1. Build the buckets with the box's bucket env: `SHARED_VERIFY=1 neuron/build_buckets.sh OUT 5561` (adds `dec8s_32 dec8s_128 dec8s_512
   dec8s_1024`; or `ATTN_DEC_SHARED=1 ATTN_DEC=nki WT=1 MLP=nki KV=fp8 UNSAFE_FP8FNCAST=1 MODEL=... PAGES=5561 python neuron/trace_model.py
   OUT/dec8s_512 layer 8 1 512`, 12-14 s each) and upload them next to the existing buckets under the Terraform `graphs_s3_prefix`
   (deploy/ecs-neuron/variables.tf, default `graphs/cydonia-24b-fp8-kv`; one subdirectory per bucket with graph.neff + meta.json; the
   `meta.json` carries `"shared_table": true`, the directory name is free).
2. Rebuild the server image from this branch (neuron_model.cpp is needed: an OLD binary refuses a graph dir with dec8s buckets as "two fp8
   layer NEFFs for the same bucket R=8 Q=1 P=..."; the new binary with the old graph set behaves exactly as before). Tag e.g.
   `neuron-server:verify-kernel-<sha>`.
3. Env: nothing required; `BEHEMOTH_NEURON_SPEC_PAGE_COST` default changes to 0.002 in the separate gate commit (override to 0.01 to
   keep today's gating, or 0 to force drafting as measured above).
4. Check: the step log (`BEHEMOTH_NEURON_STEP_LOG=1`) shows `"decode_bucket":"dec8_<P>s"` on verify steps and `dec1_<P>` on plain steps;
   verify wall ~ plain + 11-13 ms at 128-512 pages.
5. Rollback: delete the `dec8s_*` bucket directories from the graphs prefix (the engine then never picks the shared path; the new binary
   is otherwise identical) or set `BEHEMOTH_NEURON_SPEC_PAGE_COST=0.01`; no data-format change (KV layout, prepared shards, heads untouched).


## Deployment guard (added when shipping)

The engine default for `BEHEMOTH_NEURON_SPEC_PAGE_COST` is now chosen at load: 0.002 only when the graph set contains shared-table verify buckets
(`NeuronModel::has_shared_verify()`, any decode bucket with `shared_table`), otherwise 0.01 as before. Reason: with per-row dec8 verify (+47 ms per
verify step at 512 pages) a 0.002 gate drafts at a loss, so a new image deployed BEFORE the dec8s_* buckets reach the graphs prefix must keep the old
gate. An explicit `BEHEMOTH_NEURON_SPEC_PAGE_COST` still overrides either default. Compile-checked locally (g++ -fsyntax-only on neuron_model.cpp and
neuron_engine.cpp); not run on the box.
