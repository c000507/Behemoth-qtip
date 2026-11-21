# Ornith REAP20 v3 on trn1: the WT (transposed-weight) graph set as a drop-in for the production set

Box i-0cd98c451f018c0a3 (trn1.2xlarge, 2 NeuronCores, TP = 2), aws-neuronx-runtime-lib 2.33.10, neuronx-cc 2.26. Model
`models/ornith-reap20-v3-fp8` (prepared, fp8, unchanged), production graph set `graphs/ornith-reap20-v3-kv8-seg20-mtp-pf1024`,
production args and env (`deploy/ecs-neuron/models.auto.tfvars`, "ornith"). Branch `ornith-wt` (from `ornith-decode-prof` 8803336).
Session 2026-11-21. Nothing was pushed or deployed. Follows docs/neuron-reap-decode-profile.md (why WT: the compiler transposes
every `[out, in]` fp8 weight tile on chip at decode; `WT=1` stores them `[in, out]`).

## TL;DR

* **The WT set is a drop-in replacement and is faster everywhere.** Decode (engine step, MTP on): **1 row 16.8 -> 12.6 ms, 1 row
  with a draft 25.3 -> 18.9, 2 rows 23.8 -> 18.2, 4 rows 34.3 -> 27.3, 5-8 rows 100.4 -> 45.4 ms** (8-row bound segments with
  `GDN_RSPLIT=4`); 1 stream at the client 62 -> 81 tok/s. Prefill on the frozen corpus **5-12 % faster** (TTFT 501: 0.58 -> 0.53 s,
  6715: 5.07 -> 4.71, 23763: 18.42 -> 17.29, 50111: 43.51 -> 41.18 s; 4 x 7k aggregate 1327 -> 1432 tok/s), decode stall during a
  prefill 730 -> 686 ms median.
* **The prefill MoE blocker is solved without a second weight copy**: the grouped MoE (Q = 256 / 1024) is natural in WT and
  faster (4.6 -> 3.8, 12.8 -> 11.9 ms); dense `MOE=all` at Q = 64 in groups of 5 experts is faster too (3.1 -> 2.4 ms). The only
  slower NEFFs are the dense overflow reruns of the grouped MoE (Q = 256 / 1024: 6.5 -> 9.5, 22.8 -> 38.1 ms; one fallback seen in
  this session's parity runs, ~0.02 % of launches in the earlier load statistics).
* **Parity**: 3 of 7 parity prompts bit-identical to production (decode, heads, gdn / gqa, grouped MoE give production's bits);
  prompts with a Q = 64 piece differ at the fp8 noise level (rel L2 0.09-0.12, top-1 7 / 7, 44 / 46 greedy steps, flips at
  margins 0.20 / 0.04; f32-reference error 0.0816 vs 0.0827). Two starts bit-identical, prefix-cache hits bit-identical to cold,
  0 non-finite in the soak and every run.
* **Engine**: load-time transpose (18.8 GiB, +9 s startup), no new model prefix; refuses mixed WT / non-WT sets.
  **HBM**: 16142 MiB per core (production 16087), ~178 MiB headroom (233).
* **Deploy** (section 8): image from `ornith-wt` FIRST, then the new graph prefix (copy of `tmp/ornith-wt/graphs-wt/`), args / env
  / model unchanged. Nothing blocks it apart from the usual canary; `MOE_SLOTS=row` (-0.4 ms) is not included (not identical).

## 1. Prefill MoE under WT

The decode-profile session left WT blocked on the prefill MoE: through the decode code path, `MOE=all` relays the whole
`[E, H, 2 EIl]` gate|up tensor to `[H, E 2 EIl]` every launch (moe_1_64_32 3.1 -> 14.6 ms). A second, untransposed copy of
the experts does not fit (section 4). What was found (layer_bench, ms per launch, best of 8; production = the `-pf1024` NEFF):

| NEFF | form | production | WT |
|---|---|---|---|
| moe_1_64_32 (`MOE=all`, Q = 64: the primary prefill MoE of <= 64-token pieces) | one relayout of all 205 experts (`WT_MOE=''`) | 3.10 | 14.56 |
| | per expert, results stacked (`WT_MOE=stack`, decode-profile) | | 26.2 |
| | **`WT_MOE=grp`, experts in groups of `WT_GRP`** (`egu_mm`): 4 / 8 / 16 / 52 / 69 / 103 | | 64.4 / 54.6 / 41.2 / 26.8 / 25.7 / 24.2 |
| | `WT_MOE=grp` with **5**, 35, 41 experts per group | | **2.43**, 2.48, 2.50 |
| | `WT_MOE=tr` (transposed orientation: outputs `[E 2 EIl, T]`, down projection contracting the leading axis; CPU bit-identical) | | grp 5 / 41: 2.49 / 2.52 (neuronx-cc canonicalises it to the same NEFF); per expert 26.6 |
| moeg_1_256_32 (grouped, Q = 256: hot 24 + cold 32 slots) | hot tier relaid (`''`) / groups of 8 | 4.62 (4.40 in the set bench) | **3.78** / 3.93 |
| moeg_1_1024_32 (grouped idx, Q = 1024: hot 32 + cold 128 slots) | `''` / groups of 8 | 12.79 | **11.94** / 12.38 |
| moe_1_256_32 (`MOE=all`: ONLY the grouped MoE's overflow rerun) | `''` / grp 5 / grp 35 / grp 41 | 6.46 | 17.4 / 138 / 69.7 / 62.7 |
| | 64-token sub-blocks (`MOE_TSUB=64`) x grp 5 / grp 41; 128-token sub-blocks x grp 5 | | **9.53** / 9.88; 72.6 |
| moe_1_1024_32 (`MOE=all`, overflow rerun only) | grp 8 / grp 41 / **`MOE_TSUB=64` grp 5** | 22.8 | 146 / 233 / **38.1** |
| decode gather MoE, R = 1 / 2 / 4 / 8 (8 R gathered slices) | `''` (relayout of the gathered slices) | 0.263 / 0.339 / 0.502 / 0.831 | **0.232 / 0.327 / 0.460 / 0.704** (grp 16 / 32 at R = 8: 0.99 / 0.82) |

Readings:

* **The grouped prefill MoE is the natural WT form and is faster than production**: its cold tier is one `[C, H] x [H, 2 EIl]`
  matmul per expert, which with WT needs no weight transpose (`bmm8g(wt=not WT)`), and the hot tier relays only 24-32 gathered
  slices. Q = 256: 4.6 -> 3.8 ms, Q = 1024: 12.8 -> 11.9 ms per launch.
* **`MOE=all` under WT is good only with at most 64 token rows per matmul and particular group widths** (5, 35 or 41 experts
  per matmul: those NEFFs compile in 16-40 s to ~0.8 MB; every other width compiles to a 4-5 MB NEFF that runs 8-20x slower).
  This is a neuronx-cc 2.26 lowering cliff, reproducible but not understood (no profile was taken); 128 or more rows per
  matmul fall off it at every width tried. So: Q = 64 `MOE=all` with groups of 5 is **2.43 ms, 22 % faster than
  production**; the Q = 256 / 1024 `MOE=all` NEFFs, which only run when a grouped launch overflows its slots, use 64-token
  sub-blocks of the same form and are **1.5x / 1.7x slower than production (9.5 / 38.1 ms)**.
* TTFT cost of the slower fallback: a fallback happened in 5 of 22 560 grouped launches at Q = 256 and in none of ~3 600 at
  Q = 1024 in the earlier sessions (docs/neuron-qwen35-moe.md 13.7, 14.3), i.e. +3 ms (Q = 256) / +15 ms (Q = 1024) per
  fallback, < 0.01 % of prefill time on that traffic. Worst case (every launch of a 1024 chunk overflowing): 40 x (11.9 + 38.1)
  = 2.0 s per chunk instead of 40 x (12.8 + 22.8) = 1.4 s (+40 %).
* Device vs CPU mirror (random real-shape weights, `devtest.sh`): every WT MoE NEFF equals the production value of its kind
  (moe_1_64 3.391e-2 = 3.391e-2, moeg 256 3.483e-2, moeg 1024 3.464e-2, moe_8 3.60e-2; the final fallbacks moe_1_256 3.501e-2, moe_1_1024 3.488e-2), ranks bit-identical.
  CPU: every WT form gives bit-identical logits to the production layout in the fp8 mirror (`test_qwen35_graphs_cpu.py` digest
  line; MOE all / gather, grouped onehot / idx with forced overflow).
* Not pursued: the NKI MoE kernels (`MOE_KERNEL=row/dyn` do not build with WT; out of scope).

## 2. The set

`graphs-wt` = every NEFF of the production `-pf1024` set rebuilt with `WT=1` and the build flags recorded in its own
`meta.json` (`deploy/neuron-reap/wt/wtspecs.py`; checked: 7 production NEFFs rebuilt from the same specs WITHOUT `WT` have
metas identical to production, only keys the old metas lack differ, all at their defaults), plus six new 8-row segments.
96 NEFF directories (production: 90):

| group | NEFFs | WT form |
|---|---|---|
| decode segments R = 1 / 2 / 4, P = 32 .. 4096 | `seg20_{1,2,4}_{32,128,512,1024,2048,4096}` (18) | `WT=1`, gathered experts relaid (`WT_MOE=''`) |
| **new: 8-row bound segments** | `seg20_8_{32,128,512,1024,2048,4096}` (6) | `WT=1 GDN_SLOT=bound GDN_RSPLIT=4` (no dynamic state DMA) |
| verify segments (MTP, 1 draft) | `segv20_1_2_*` (6) | `WT=1 VERIFY=1` |
| MTP head | `mtp_1_1_32`, `mtp_1_2_32` | `WT=1` (`fc` and the shared `head.w` [in, out]; its own experts relaid at load like the main ones) |
| decode sub-blocks R = 1 / 2 / 4 / 8 | gdn / gqa (P ladder) / moe | `WT=1` (as production: R = 1 / 2 / 4 gdn bound, R = 8 gdn dynamic; unused while the segment ladders are complete) |
| heads | `head_{1_1,2_2,4_4,8_8,64_1,256_1,1024_1}` | `WT=1` |
| prefill Q = 64 | gdn, gqa ladder, **moe (`WT_MOE=grp WT_GRP=5`)**, head | |
| prefill Q = 256 / 1024 | gdn, gqa ladders, moeg (grouped, `WT_MOE=''`), **moe fallback (`MOE_TSUB=64 WT_MOE=grp WT_GRP=5`)**, head | |

Build: `deploy/neuron-reap/wt/wtbuild.sh OUT <production set> WT=1 'moe_1_64_32@WT_MOE=grp,WT_GRP=5' 'moe_1_256_32@WT_MOE=grp,WT_GRP=5,MOE_TSUB=64' 'moe_1_1024_32@WT_MOE=grp,WT_GRP=5,MOE_TSUB=64'`
(small NEFFs 2 in parallel, segments and Q = 1024 MoE one at a time: seg20 R = 1-4 120-160 s / 15 GB each, R = 8 220-300 s; whole set ~2 h
on the trn1.2xlarge). NEFF bytes are not reproducible between two identical builds (different md5); the metas are.

Device vs CPU mirror (`devtest.sh`, random real-shape weights, the mirror in WT mode): all 53 testable WT sub-block NEFFs
2.5e-2 .. 4.4e-2, the values of the production NEFFs of the same kinds, both ranks bit-identical (gdn bound R = 2 / 4 are not
supported by the test harness; they and the segments, heads and MTP graphs are covered end to end, section 7).

Per NEFF (layer_bench, ms per launch, production -> WT; quiet box):

| NEFF | prod | WT |
|---|---|---|
| seg20_1_32 / 128 / 512 / 1024 / 2048 / 4096 | 7.06 / 7.01 / 7.68 / 8.44 / 9.59 / 12.47 | **5.32 / 5.37 / 5.99 / 6.80 / 7.93 / 10.77** |
| seg20_2_32 / 512 / 4096 | 10.42 / 11.14 / 20.03 | 7.94 / 8.72 / 17.42 |
| seg20_4_32 / 512 / 4096 | 15.56 / 17.01 / 34.41 | 12.46 / 14.03 / 31.43 |
| seg20_8_32 .. 4096 (new) | - (R = 8 ran 80 sub-block launches) | 21.30 / 20.38 / 23.67 / 28.26 / 39.30 / 66.59 (P = 32 / 128 / 512 / 1024 / 2048 / 4096) |
| head_1_1 / head_8_8 | 1.94 / 2.15 | 1.20 / 1.24 |
| mtp_1_1_32 / mtp_1_2_32 | 4.50 / 2.63 | 1.69 / 1.74 |
| gdn_1_1_32 / gdn_1_64_32 / gdn_1_256_32 / gdn_1_1024_32 | 0.30 / 0.54 / 1.11 / 4.01 | 0.24 / 0.47 / 1.03 / 3.76 |
| gqa_1_64_32 / gqa_1_256_4096 / gqa_1_1024_2048 | 0.47 / 10.6 / 19.1 | 0.41 / 10.5 / 19.0 |
| moe_1_64_32 / moeg_1_256_32 / moeg_1_1024_32 | 3.10 / 4.40 / 12.8 | 2.43 / 3.78 / 11.9 |
| moe_1_256_32 / moe_1_1024_32 (overflow reruns only) | 6.46 / 22.8 | 9.55 / 38.1 |

## 3. Engine

`neuron/cpp/qwen35_model.cpp` (engine image of this branch):

* Every graph carries `wt` from its meta. `load_weights` refuses a set whose NEFFs disagree (`Qwen3.5 NEFFs disagree on the weight
  layout (meta wt): ... - one graph set must be one layout`; MTP graphs included, they bind `head.w`), refuses WT with the `egu_t`
  layout or `ROUTER=shard`, and refuses WT-stored weights (`prepared.json` `"wt": true`) under a non-WT set. Old metas have no `wt`:
  the production set loads exactly as before (BUILD2 serves it unchanged).
* WT set over the normal prepared dir: **load-time transpose** of the tensors `to_wt` names (2-D `qkv z b a o q k v sg su sd fc rw w`
  [out, in] -> [in, out]; `egu` [E, 2 EIl, H] -> [E, H, 2 EIl], per expert, 4 threads per rank), MTP tensors included. 18.8 GiB over
  the two ranks, +9 s of weight load (weights resident 50 -> 59 s, server up 87 -> 97 s). No new model prefix is needed.
  A prepared dir already in WT layout (offline `mkwt.py`, `"wt": true`) is uploaded as is.
* Tested: the decode-profile minimal WT set on BUILD2 over the production prepared dir returns the same three 256-token greedy
  texts as the production buckets (md5 b1a83cab / e42e1fc4 / 3323d6df); a set with one non-WT head is refused at load.

## 4. HBM

The runtime's accounting per NeuronCore (`mem.sh`, server up with MTP, production args), MiB:

| | total | tensors (weights, state, MTP, IO) | shared scratchpad | model code | DMA rings | runtime | NEFFs |
|---|---|---|---|---|---|---|---|
| production `-pf1024` | 16087 | 14341 | 640 | 810 | 244 | 37 | 90 |
| **WT set** | **16142** | 14341 | 640 | 875 | 218 | 53 | 96 |

Allocations fail at ~16320 MiB (docs/neuron-qwen35-moe.md 14.8), so the headroom per core goes from **~233 to ~178 MiB**.
The WT NEFFs themselves are smaller (no transpose instructions); the +65 MiB of code is the six 8-row segments (7-18 MB NEFFs)
and the 64-token sub-blocked fallbacks (moe_1_1024_32 is a 12 MB NEFF). The weights take the same bytes in either layout (one
copy; the relayout is in place at load). Spare levers if a future bucket needs room: the decode sub-blocks of R = 1 / 2 / 4 / 8
(44 NEFFs; their share of the 875 MiB of code was not measured) are not used while every row bucket has its complete segment ladder.

## 5. Measured: decode

Same session, same box, fresh server per run, production args (`--prefill-chunk 1024 --token-budget 1032 --max-batch-requests 8
--prefix-cache-gb 4`) and env (`BUCKET_FIT`, prefix cache, MTP adaptive); only the graph set and the engine differ: production =
`graphs/...-pf1024` + the engine of 8803336 (= production engine sources), WT = `graphs-wt` + the engine of this branch
(`dec.sh`: short prompts, 1 / 2 / 4 / 8 concurrent streams x 256 new tokens). Engine step log, median ms per decode step:

| rows in the step | production, MTP on | **WT, MTP on** | production, MTP off | **WT, MTP off** |
|---|---|---|---|---|
| 1 (no draft) | 16.8 | **12.6** | 16.8 | **12.5** |
| 1 with an MTP draft (verify segment + MTP head) | 25.3 | **18.9** | - | - |
| 2 | 23.8 | **18.2** | 23.8 | **18.0** |
| 3-4 | 34.3 | **27.3** | 34.3 | **27.3** |
| 5-8 | 100.4 | **45.4** | 100.4 | **45.4** |

At the client (1 stream, MTP on, 256 tokens): 16.2 -> **12.3 ms per token (62 -> 81 tok/s)**; 8 streams 91-101 -> 42-46 ms per
token (aggregate ~80 -> ~175 tok/s). The decode-profile estimates (12.6 / 18.0 / 27.3 / 45.5) are met. The MTP draft step costs
18.9 ms instead of 25.3 (mtp_1_x 4.5 -> 1.7 ms, verify segments by the same WT gain), so the adaptive MTP keeps its benefit.

**`MOE_SLOTS=row`** (task 7: the 1-row decode MoE takes the router's top-8 as slots, no union top-k; `seg20_1_*` + `moe_1_1_32`
rebuilt, set `wtrow`): seg20_1_32 5.32 -> 5.11 ms, seg20_1_4096 10.77 -> 10.55 ms per launch; engine 1-row step 12.5 -> **12.1 ms**
(MTP off), 12.6 -> 12.2 (MTP on), draft step 18.9 -> 19.0. **Identity check fails**: prefill-end logits bit-identical (7 / 7), but
every decode step differs (rel L2 0.03-0.14: the changed order of the 8-expert down-projection sum, amplified through 40 fp8 layers),
greedy 38 / 41 steps, 3 flips at margins 0.32 / 0.09 / 0.13. Not in the candidate set; it is a -0.4 ms option for whoever accepts
fp8-noise-level output changes (same class as the 256 -> 1024 chunk change that was deployed).

## 6. Measured: prefill, concurrency, decode stall

Prompt matrix (`matrix.sh`, the frozen corpus; TTFT at the client, in brackets prompt tokens / TTFT):

| prompt (tokens) | production | **WT** |
|---|---|---|
| 82 | 0.46 s | **0.41 s** |
| 501 | 0.58 s (868 tok/s) | **0.53 s (941)** |
| 3029 | 2.22 s (1367) | **2.07 s (1463)** |
| 6715 | 5.07 s (1326) | **4.71 s (1425)** |
| 23763 | 18.42 s (1290) | **17.29 s (1374)** |
| 50111 | 43.51 s (1152) | **41.18 s (1217)** |
| 4 concurrent x ~7k (28392): last first token, aggregate | 21.4 s, 1327 tok/s | **19.8 s, 1432 tok/s** |
| 8 concurrent x ~3.8k (30293) | 25.4 s, 1194 tok/s | **22.7 s, 1334 tok/s** |
| 4 concurrent x ~60 | 1.09 s | **0.90 s** |

A full 1024-token chunk in the engine: 787 -> **737 ms** wall (1301 -> 1389 tok/s; forward: gdn 126.6 -> 119.5, gqa 104.5 -> 102.7,
grouped MoE 520 -> 480 ms). Prefill is 5-12 % faster at every size, and decoding streams that share the box with a prefill
are faster too (concurrency rows).

Decode streams during a prefill (`stall.sh`, gap between streamed chunks of a decoding stream while a long prompt prefills):

| | production | WT |
|---|---|---|
| 1 decoder, 20.5k prompt: gap median / p90 / max | 730 / 862 / 891 ms | **686 / 810 / 840 ms** |
| 4 decoders, 26.8k prompt | 791 / 909 / 910 ms | **735 / 854 / 856 ms** |
| 1 decoder, 6k prompt: max | 760 ms | **708 ms** |
| the prefill itself (20.5k / 26.8k) | 16.1 s / 22.2 s | **15.1 s / 20.7 s** |

The stall is one decode step per 1024-token prefill chunk in both (section 14 of docs/neuron-qwen35-moe.md); WT shortens it by
the faster chunk.

## 7. Parity, determinism, prefix cache, fault soak

* **Greedy identity vs production** (`par.sh`, MTP off, 7 prompts 82 .. 23763 tokens x 8 greedy steps, logits dumps): **3 of 7
  prompts (501, 3029, 23763 tokens) are bit-identical on all 8 records** - the WT decode segments, heads, gdn / gqa prefill and
  grouped MoE (Q = 256 / 1024) give the same bits as production. The 4 prompts with a Q = 64 piece (82, 1625, 5251, 6715 tokens:
  the new `moe_1_64_32` group form accumulates in another order) differ by rel L2 0.09-0.12 (KL <= 0.005), the fp8 noise band
  (production 256 vs 1024 chunks: 0.08-0.16). Prefill-end top-1 7 / 7; greedy steps 44 / 46 agree, the 2 flips at production
  margins **0.20 and 0.04** (the following steps are not compared after a flip). 0 non-finite.
* **f32 reference** (`qwen35_prepared_ref.py` references of section 13.5, prefill-end logits): 82 tokens: production 0.0827 (KL
  0.0056), WT **0.0816 (KL 0.0015)**; 501 tokens: 0.1174 / 0.1174 (identical bits). Top-1 equal to the reference in all.
* **Determinism**: two fresh starts of the WT server, 56 logits records: **bit-identical**.
* **Prefix cache** (`cache.sh`, `PREFIX_ALIGN=1`, 1024 grid): full hit, partial hit (+487 tokens), the same pair for a prompt with
  a padded tail, and a cold second server: all **bit-identical** to cold. Full hit of the 6715-token prompt: TTFT 0.73 s.
* **Fault soak** (`soak.sh`): 3 fresh starts x (8 concurrent x up to 1024 tokens, 4 x 1024, 1 x 2048, a 6k prompt + 512),
  MTP adaptive on, the 8-row bound segments in use: **0 non-finite, 0 errors** in 42 requests; plus 0 non-finite in every other run
  of the session (decode, matrix, stall, parity, cache: ~200 requests, ~400 prefill chunks). With MTP adaptive on, single-stream
  texts can differ between starts (the draft decision follows measured step times); MTP off they are bit-identical (above).
* Overflow fallback: one grouped launch fell back to the dense rerun in both the production and the WT parity runs (the same
  prompt), i.e. the WT fallback path ran end to end.

## 8. Deploy recipe (not applied)

Nothing was deployed. What a deploy needs:

1. **Graph prefix**: copy `s3://behemoth-neuron-models-912935854432/tmp/ornith-wt/graphs-wt/` (96 directories of `graph.neff` +
   `meta.json`, 220 MB, `MANIFEST.md5`: md5 of every file) to a NEW prefix, e.g. `graphs/ornith-reap20-v3-kv8-seg20-mtp-pf1024-wt/`.
   It replaces the whole `-pf1024` set (one layout per set: a mixed set is refused at load). Rebuild from source with
   `deploy/neuron-reap/wt/wtbuild.sh` + `fb.sh` (section 2).
2. **Engine image** from branch `ornith-wt` (engine code in commit 1eeda00, any later commit of the branch; the WT load path in `qwen35_model.cpp`). The new image runs the
   current `-pf1024` prefix unchanged (no `wt` in its metas), so the image goes first, the prefix second. An OLD image with the
   new prefix would bind untransposed weights to WT NEFFs (garbage, no error): **never deploy the prefix before the image.**
3. **Model**: unchanged (`models/ornith-reap20-v3-fp8`, MTP dir included); the engine transposes at load (+~10 s startup).
4. **Args / env**: unchanged (`--prefill-chunk 1024 --token-budget 1032 --max-batch-requests 8 --prefix-cache-gb 4`,
   `BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1`, prefix cache, MTP adaptive). `models.auto.tfvars`: only `graphs_s3_prefix` of "ornith".
5. **Rollback**: the old prefix (the new image keeps serving it bit-identically to the old engine path: no `wt` metas, no relayout).
6. **Watch**: decode `ms/step` (expect 12.6 / 18 / 27 / 45 at 1 / 2 / 4 / 8 rows), `prefill_tokens_per_s` (+5-12 %), the log line
   `weights resident in ... (WT layout: 18807 MiB transposed at load)`, `moe_fallbacks` (a fallback now costs 9.5 / 38 ms instead
   of 6.5 / 23), HBM 16142 MiB per core (`mem.sh`).

## 9. Not done / left

* The dense `MOE=all` fallbacks at Q = 256 / 1024 are 1.5x / 1.7x slower than production (a neuronx-cc lowering cliff of the
  WT layout above 64 rows per matmul, section 1); harmless at the measured fallback rates, a cost only for pathological routing.
* Prompts with a Q = 64 piece are not bit-identical to production (new MoE form; fp8 noise level, f32-reference error equal or lower).
* Not done (out of scope, recommendations): MoE routing + expert gather as one NKI kernel (~27 us exposed per MoE at R = 1,
  ~1.1 ms per step), a gqa decode attention kernel (195 us per gqa sub-block, ~0.8-1.1 ms per step), a fused gdn decode kernel
  (~0.5-0.8 ms) - docs/neuron-reap-decode-profile.md section 7; NKI MoE kernels with WT (`MOE_KERNEL` does not build with it).
* The unused decode sub-blocks (R = 1 / 2 / 4 / 8) could be dropped from the set for HBM headroom (amount not measured); kept for a
  drop-in identical structure.
