# Ornith REAP20 v3 decode on trn1: where the 17 ms per token goes (device profile), and the weight-layout fix

Box i-0cd98c451f018c0a3 (trn1.2xlarge, 2 NeuronCores, TP = 2), aws-neuronx-runtime-lib 2.33.10, neuronx-cc 2.26, neuron-profile
2.31.13. Model `models/ornith-reap20-v3-fp8` (prepared, fp8), graph set `graphs/ornith-reap20-v3-kv8-seg20-mtp-pf1024` (the deployed
one), engine of branch `rocm-to-neff` 8eb7f41 with the production env and args (`deploy/ecs-neuron/models.auto.tfvars`, "ornith").
Branch `ornith-decode-prof`. Session 2026-11-20. Nothing was pushed or deployed.

**Follow-up (2026-11-21, branch `ornith-wt`): the complete WT set, its prefill MoE forms, the engine load path and the
same-session measurements are in docs/neuron-reap-wt.md.**

## TL;DR

* **The 1-row decode step is not weight-streaming bound and not latency bound: it is compute bound on the tensor engine (PE) and the
  scalar/activation engine (ACT), because neuronx-cc transposes every fp8 weight tile on chip, every step.** The prepared weights are
  stored `[out, in]`; the M = 1 matmul needs the contraction axis on the SBUF partitions, so per 128 x 128 weight tile the compiled
  NEFF runs a PE transpose (`MATMUL` against identity, moving = the 128-column tile, into PSUM), an ACT `COPY` PSUM -> SBUF back to
  fp8 (one per four tiles, 464 ns each), a second `LDWEIGHTS`, then the real matvec. In a 20-layer segment that is ~40k extra PE
  instructions and ~3.7 ms of ACT time per launch; the head does the same (5133 ACT copies, 1.65 ms).
  The DMA queues run at ~205 GB/s while active but are idle 40 % of the time.
* **Fix: store the weights transposed (`[in, out]`, new graph switch `WT=1`).** Same bytes, same fp8 math, no on-chip transposes.
  Measured on the device:
  segment 7.08 -> 5.34 ms per 20 layers, head 1.94 -> 1.20 ms, and **end to end (engine, 1 stream, MTP off) 16.8 -> 12.6 ms per
  token (59.5 -> 79.5 tok/s), greedy output byte-identical to the production layout on 3 / 3 x 256-token requests**, 0 nonfinite.
* All levels of concurrency gain (engine, MTP off): 2 rows 23.8 -> 18.0 ms, 4 rows 34.3 -> 27.3 ms.
* After WT the remaining per-step device time is 10.4 ms of segments + 1.14 ms head (DMA-bound, at the wall) against a
  bandwidth floor of ~6.6 ms. The rest is the MoE routing chain (router -> top-k -> expert gather: DMA idle ~25-30 us per MoE), the
  gqa decode attention (195 us per gqa sub-block, ~60 us of it weights), 80 in-graph all-reduces (~10 us each effective) and 3
  launches. Ranked levers in section 7: with WT plus the measured `MOE_SLOTS=row` the step is ~12.1 ms; reaching <= 10 ms needs
  2-3 of the kernel-level items (routing+gather, gqa attention, gdn decode) on top.
* **R = 8 (106 ms per step) is two things: 81 sub-block launches (no 8-row segment, because the bound 8-row gdn compiled to 5.0 ms)
  and the 8-row gdn state contraction itself (1.36 ms per gdn even dynamic).** Splitting the 8 rows' state contractions into two
  groups of 4 in the graph (new `GDN_RSPLIT=4`) makes the bound 8-row gdn 0.50 ms (from 5.0 bound / 1.36 dynamic), which makes a
  bound 8-row 20-layer segment possible: **21.3 ms per launch with WT, i.e. ~44 ms per 8-row step instead of 106** (est.).
* Compiler flags (`-O1`, `-O3`, `--model-type=transformer`), `LIN_CSE`, `ROUTER=shard`, `AR_BF16`, `RLOGIT=bf16`, `GDN_FORM=fused`,
  `GDN_DEC`, `ROUTE=thr`, `ATT_GATHER=flat` change a 4-layer segment by <= 2 % (or make it slower): not levers.

Engine end to end (prod env, short prompts, 256 new tokens, MTP off, `--prefill-chunk 64`, median step ms; the WT set = WT
decode segments R = 1 / 2 / 4 / 8 (the 8-row one bound with `GDN_RSPLIT=4`), WT heads, WT sub-blocks, WT Q = 64 prefill; the control =
the same bucket names from the production set; `dec.sh`):

| streams (rows) | production (deployed set, chunk 1024) | control (prod buckets, chunk 64) | **WT set** | tok/s per stream, prod -> WT |
|---|---|---|---|---|
| 1 | 16.8 | 16.8 | **12.6** | 59.5 -> **79.4** |
| 2 | 23.8 | 23.8 | **18.0** | 42 -> 56 |
| 4 | 34.3 | 34.3 | **27.3** | 29 -> 37 |
| 8 | 101.0 | 100.7 | **45.5** | 9.9 -> **22** (aggregate 79 -> 176 tok/s) |
| 1, MTP on (production) | 16.2 ms per token (steps with a draft 25.3 ms) | | not built | 61.7 |

## 1. Method

* **Step reproduction**: the production server (prod env and args), short prompts, 256 new tokens per stream, the engine step log and
  `BEHEMOTH_NEURON_QWEN35_LTRACE` for the host split (`deploy/neuron-reap/decode-prof/dec.sh`, `dec1.sh`).
* **Per-launch device time**: `layer_bench` (both ranks, aliased states, drained batches; ms per launch = best of 8 batches).
* **Device profile**: `neuron-profile capture -n graph.neff --collectives-workers-per-node 2 --collectives-worker-count 2
  --num-exec 5 --profile-nth-exec 5` (both ranks), `neuron-profile view --output-format summary-json` and `--output-format json`
  (`prof.sh`); the json (356 MB for a 20-layer segment) is reduced on the box by `ana.py` (per-engine busy time as the union of
  the `active_time` intervals, DMA bytes / active time per queue, collectives, and **per sub-block intervals: every sub-block ends
  with its all-reduce, so the all-reduce completion times cut the timeline into gdn / gqa / moe pieces**) and `cls.py` (instructions
  of one interval by engine / opcode / dtype / tile shape / PSUM source). Inputs are the profiler's (zero-filled) tensors: routing
  then always picks the same 8 experts, which does not change the instruction stream or the byte counts.
* Profiling artefact: the capture starts the two ranks independently, so the first all-reduce of a launch waits for the other
  rank (a 4.4 ms `BARRIER` in the segment capture, 3-7 ms in the 4/8-row sub-block captures). It is excluded below (the steady-state
  figures match `layer_bench` within 3 %: segment 6.87 ms device vs 7.08 ms per launch).
* Experiments: graph variants built with the production decode settings (`deploy/neuron-reap/lbuild.sh`: MOE=gather, KV8=1,
  GDN_SLOT=bound, PAGES 4097, SLOTS 9) plus one switch each, on a 4-layer segment `seg_gdn_gdn_gdn_gqa` (R = 1, P = 32) unless
  noted; timing-only probes `PROF_NOAR=1` (all-reduce -> identity) and `PROF_FIXED=1` (MoE uses a static slice of 8 experts, no
  gather) produce wrong output by design.

## 2. Measured breakdown of the production 1-row step

Production server (`graphs/...-pf1024`, prod args `--prefill-chunk 1024 --token-budget 1032 --max-batch-requests 8`, prod env),
engine step log, median ms per decode step (`run/p_nomtp`, `run/p_mtp`):

| rows in the step | 1 | 2 | 3-4 | 5-8 |
|---|---|---|---|---|
| MTP off | 16.8 | 23.8 | 34.3 | 100.7-101.0 |
| MTP on (1 stream adaptive k = 1) | 16.8 (no draft) / 25.3 (with a draft): 16.2 ms per token | 23.8 | 34.3 | 101 |

R = 5-8 rows all take the 8-row path (sub-blocks, see section 8).

Host side (LTRACE, rank 0, ms per step, MTP off): wall 16.6-16.7 = dispatch 0.01 + input writes 0.04 + host gaps 0.01-0.02 + **2
segment launches 14.4-14.5** + **head 2.05** + logits read 0.08.

Device side (neuron-profile, `seg20_1_32` = 20 layers, rank 0; rank 1 within 0.5 %; head_1_1):

| component (per step) | ms | how measured | weight / state bytes per rank | DMA floor @ 240 GB/s | x floor |
|---|---|---|---|---|---|
| 30 gdn sub-blocks (attention part of the linear-attention layers) | 5.10 (30 x 170 us) | per-interval, seg profile | 16.8 MB weights + 2 MB state each | 2.36 | 2.2 |
| 10 gqa sub-blocks (P = 32 bucket) | 2.47 (10 x 247 us) | " | 13.6 MB weights + KV pages | 0.57 | 4.3 |
| 40 MoE sub-blocks | 6.04 (40 x 151 us) | " | 15.8 MB (8 experts 12.6 + shared 1.6 + f32 router 1.7) | 2.64 | 2.3 |
| (of which in-graph all-reduces, 80 x 8 KiB f32) | 0.39 (80 x 4.9 us) | cc_ops of the profile | | latency | |
| segment tails / launch-internal | 0.1 | | | | |
| **2 segment launches, device** | **13.74** | | 1.32 GB (hbm_read 659 MB per launch) | 5.5 | 2.5 |
| segment launch overhead (engine 7.23 / launch vs device 6.87) | 0.7 | LTRACE - profile | | | |
| **head_1_1, device** | **1.87** | head profile | 254 MB | 1.06 | 1.8 |
| head launch overhead | 0.18 | | | | |
| host (dispatch, input writes, gaps, logits read) | 0.15 | LTRACE | | | |
| **step** | **16.6-16.8** | engine | 1.59 GB | 6.6 | 2.5 |

Per engine, one 20-layer segment (6.87 ms of device time):

| engine | busy ms | instructions | note |
|---|---|---|---|
| PE (tensor) | 4.22 (61 %) | 86094 MATMUL + 86048 LDWEIGHTS | about half are weight-tile transposes (below) |
| ACT (scalar) | 4.10 (60 %) | 12913 ACTIVATE | 3.7 ms of it = PSUM -> SBUF fp32 -> fp8 copies of transposed weight tiles |
| DVE (vector) | 1.29 | 6179 | norms, recurrence elementwise, routing |
| Pool (gpsimd) | 0.25 | 4971 | DGE descriptors of the expert gather |
| DMA (all queues) | 3.95 active (57 %) | 200k packets of 3.4 KB on the software-DGE queue | 675 MB at 205 GB/s while active |
| collectives | 0.20 (40 all-reduces) | | 2-10 us each, trigger delay 0.5 us |
| all engines and DMA idle | ~0.25 | | 4-15 us per sub-block |

Per sub-block (means over the segment, us; PE / ACT / DVE busy, DMA MB moved in the interval and its active time):

| sub-block | wall | PE | ACT | DVE | Pool | DMA MB / us | AR | idle |
|---|---|---|---|---|---|---|---|---|
| gdn | 170 | 116 | 124 | 35 | 1.4 | 14.5 / 95 | 3.2 | 4 |
| moe | 151 | 85 | 79 | 25 | 10 | 21.4 / 108 (prefetch crosses intervals) | 6.3 | 7 |
| gqa | 247 | 153 | 132 | 52 | 4.4 | 9.1 / 94 | 4.6 | 16 |

The device is busy all the time (idle ~2 %), the DMA only ~57 %: **the step is bound by PE + ACT instruction streams, and the
dominant instruction class in both is the on-chip weight transpose.**

## 3. Root cause: the compiler transposes every weight tile on chip, every step

`cls.py` over one gdn / moe / gqa window of the segment profile (`prof/p1/seg_w2_8.json`), instructions by class:

| sub-block | class | count | engine time | what it is |
|---|---|---|---|---|
| gdn | MATMUL fp8 128x128, moving 128 columns | 1024 | 192 us summed (188 ns each) | transpose of a [128 out x 128 in] weight tile against identity into PSUM |
| | ACT COPY fp32 -> fp8, PSUM source, [128 x 512] | 258 | **120 us (464 ns each)** | the transposed tiles back to SBUF as fp8 (4 tiles per copy) |
| | LDWEIGHTS fp8 128x128 | 2048 | 171 us | 2 per tile: the transpose's identity side + the transposed tile |
| | MATMUL fp8 128x128, moving 1 column | 1024 | 146 us | the actual matvec |
| moe | transposes / ACT copies / matvecs | 608 / 152 / 864 | ACT 71 us | gate_up (8 x 512 x 2048) and the shared expert are transposed; the down projection `ed` is stored [E, in, out] already (256 tiles, no transpose) |
| gqa | MATMUL fp8 128x32 moving 1 column + LDWEIGHTS 128x32 | 2048 + 2048 | 288 + 163 us | q projection in 32-column tiles |
| | transposes / ACT copies | 800 / 208 | ACT 97 us | k, v, o |
| head | MATMUL / LDWEIGHTS | 32837 / 32837 | PE 1.48 ms | 2 per 16 KB tile |
| | ACT COPY | 5133 | **ACT 1.65 ms (88 % busy)** | the head is ACT-bound |

The production layout is `[out, in]` (row-scaled E4M3, one scale per output row: `qwen35_prepare.py`); torch's `x @ W.T` lowers to a
dot whose stationary operand needs the contraction (`in`) axis on the 128 SBUF partitions. A DMA cannot transpose 1-byte elements
efficiently, so neuronx-cc loads the tile as stored and transposes it on the PE, through PSUM (f32) and the ACT engine. Nothing in
the compiler flags changes this (section 6). The experts' down projection, already stored contraction-first, shows the cheap path:
one LDWEIGHTS + one matvec per tile and no ACT work.

## 4. The fix: `WT=1` (weights stored [in, out])

`neuron/qwen35_graphs.py`: `WT=1` stores the dense projections (gdn qkv / z / a / b / o, gqa q / k / v / o, shared expert sg / su / sd),
the router `rw` and the head `w` as `[in, out]` and the experts' gate|up `egu` as `[E, H, 2 EIl]` (`ed` unchanged); `lin` /
`lin_partial` / `route` / `head` multiply by the stored matrix directly. `to_wt(name, tensor)` converts a prepared tensor; the CPU
mirror test runs with it (`WT=1 python neuron/tests/test_qwen35_graphs_cpu.py`: EXACT 5e-7, fp8 numbers identical to the default
layout). The fp8 values, scales and the order of every dot product are unchanged; only the memory layout differs.

Measured (layer_bench, ms per launch; production layout -> WT):

| NEFF | prod | WT | |
|---|---|---|---|
| seg20_1_32 (20 layers) | 7.08 | **5.34** | -1.73 per launch, -3.47 ms per step |
| seg4 (gdn, gdn, gdn, gqa) | 1.513 | 1.18-1.19 | |
| head_1_1 (254 MB fp8) | 1.94 | **1.20** | -0.75 ms per step; DMA-bound: 239 GB/s while active, 98 % DMA duty |
| gdn_1_1_32 / gqa_1_1_32 / moe_1_1_32 | 0.283 / 0.351 / 0.246 | 0.241 / 0.320 / 0.229 | |
| gdn_4_1_32 / moe_4_1_32 | 0.426 / 0.499 | 0.347 / 0.452 | |
| gdn_8_1_32 (dynamic) / moe_8_1_32 / gqa_8_1_32 | 1.357 / 0.828 / 0.541 | 1.285 / 0.710 / 0.484 | |

Profile of the WT segment (`prof/p4`, 5.22 ms device): PE 2.41 ms (from 4.22, 52k MATMUL instead of 86k), ACT 0.93 ms (from 4.10),
DVE 1.32, DMA 3.51 ms active (67 %) at 213 GB/s, the same 659 MB read; per sub-block gdn 112 us, moe 125, gqa 195.

**End to end**: a minimal WT graph set (`sets/wt`: the WT seg20_1_32 + WT head_1_1 + WT 1-row sub-blocks + WT Q = 64 prefill gdn /
gqa / moe / head_64_1) over a WT copy of the prepared dir (`mkwt.py`: the same transposes as `to_wt`), against the same buckets
from the production set over the production prepared dir, engine of 8eb7f41, prod env, `--prefill-chunk 64`, MTP off, 1 stream,
3 requests x 256 tokens (`dec1.sh`):

| | ms / step (median) | tok/s (server metric) | LTRACE: segments / head per step | outputs |
|---|---|---|---|---|
| production layout | 16.8 | 59.4-59.5 | 14.45 / 2.05 | |
| **WT** | **12.6** | **79.4-79.5** | **10.93 / 1.31** | **3 / 3 byte-identical** (md5 e42e1fc4, 3323d6df, ...), 0 nonfinite |

Multi-stream with the full WT set: see the table at the top (R = 2 18.0, R = 4 27.3, R = 8 45.5 ms). Outputs at 2-8 concurrent
streams differ between runs in BOTH layouts (arrival timing changes the batch composition; e.g. the control's d4 / d8 md5s differ from
the production run's on the same prompts, and the WT run reproduces some of the production variants, e.g. d4/0 = 2ef22167); the
1-stream outputs are identical. 0 nonfinite in all runs.

Caveat found: with WT, the prefill MoE (`MOE=all`, all 205 experts) through the decode form relays the whole expert tensor
(`[E, H, 2 EIl] -> [H, E 2 EIl]`) every launch: moe_1_64_32 went 133 -> 592 ms per 40 layers in the end-to-end run. `WT_MOE=stack`
(one `[T, H] x [H, 2 EIl]` matmul per expert) was meant to avoid it, but it does not: layer_bench `moe_1_64_32` prod 3.12 ms, WT (relayout) 14.6 ms, WT_MOE=stack 26.2 ms. **The Q = 64 / 256 / 1024
prefill MoE needs its own WT form before WT can ship** (a second, untransposed copy of gate|up does not fit in HBM; candidates: the grouped prefill MoE with the transposed
gather, or a per-expert-group relayout; neither was built). TTFT in
the WT end-to-end runs is correspondingly worse (prefill not representative). The grouped prefill MoE
(`MOE=group`, `moeg_*`) and the MTP / verify graphs have not been converted (section 9).

## 5. After WT: breakdown and roofline per component

1-row step with WT (engine LTRACE 12.4 ms wall: segments 10.93, head 1.31, host 0.16; device from the WT profiles):

| component (per step) | prod ms | WT ms | DMA floor @ 240 GB/s | WT x floor | what is left (profile) |
|---|---|---|---|---|---|
| 30 gdn | 5.10 | 3.36 (112 us) | 2.36 | 1.4 | DMA ~80 % duty; DVE 38 us (norms, conv, recurrence elementwise) |
| 40 moe | 6.04 | 5.00 (125 us) | 2.64 | 1.9 | routing chain (router -> softmax -> top-k -> one-hot -> slot top-k -> DGE) ~25-30 us with the DMA idle, then the 12.6 MB expert gather at ~195 GB/s |
| 10 gqa (P = 32) | 2.47 | 1.95 (195 us) | 0.57 | 3.4 | attention compute: PE 104 us, DVE 57, ACT 53 (int8 KV dequant, scores, softmax) |
| 80 all-reduces | in the above | 0.5 device (6.2 us each) / 0.8 effective (probe) | latency | | |
| head | 1.87 | 1.14 | 1.06 | 1.08 | at the wall |
| launch overhead (3 launches) + host | 1.0 | 0.8 | | | |
| **step** | **16.8** | **12.6** | **6.6** | 1.9 | |

The PE, which bounded the production layout, is now 46 % busy (2.41 ms per segment); the DMA is 67 % busy at 213 GB/s.

## 6. Experiments that do not pay (4-layer segment, ms per launch; base 1.513-1.537, WT base 1.181-1.193)

| switch | prod layout | with WT | reading |
|---|---|---|---|
| `LIN_CSE=1` (shared fp8 activation scaling) | 1.515 | | 0 |
| `ROUTER=shard` (router split over ranks + f32 all-reduce) | 1.578 | | worse (+1 all-reduce per MoE) |
| `AR_BF16=1` (bf16 all-reduce) | 1.521 | 1.181 | 0: the 8 KiB all-reduce is latency bound |
| `NCC_FLAGS=-O1` / `-O3` / `--model-type=transformer` | 1.521 / 1.516 / 1.534 | | 0 |
| `RLOGIT=bf16` (router logits in bf16) | 1.499 | 1.202 | ~0 |
| `GDN_FORM=fused` / `GDN_DEC=1` | 1.505 / 1.501 | 1.184 | ~0 |
| `ROUTE=thr` | 1.522 | 1.179 | 0 |
| `ATT_GATHER=flat` | | 1.203 | 0 |
| `MOE_KERNEL=row` / `dyn` (NKI MoE kernels) | | build fails with WT (EGU_T reshape) | not pursued |
| `MOE_SLOTS=row` (the router's top-8 indices as slots, no union top-k) | 1.488 | **1.141** | **-11 us per MoE** (-0.45 ms per step) |
| `MOE_SLOTS=row` + `RLOGIT=bf16` | | 1.126 | -15 us per MoE (router numerics change) |
| probe `PROF_NOAR=1` (all-reduces removed) | 1.401 | 1.102 | 14 / 10 us per all-reduce: 1.1 / 0.8 ms per step upper bound |
| probe `PROF_FIXED=1` (no expert gather) | 1.473 | 1.076 | 12 / 27 us per MoE: 0.5 / 1.1 ms per step upper bound |
| probes both | 1.348 | | |

On the 20-layer segment with WT: `MOE_SLOTS=row` 5.344 -> **5.131 ms** per launch (-0.43 ms per step; `x8/seg20wtrow_1_32`); not yet
checked end to end for token identity (it changes the order of the 8-expert down-projection sum, i.e. f32 rounding only).

## 7. Ranked levers to <= 10 ms per step (1 row, no MTP)

Baseline 16.8 ms (production). Measured items first, then estimates (from the probes / per-sub-block floors above):

| # | lever | saving per step | status / evidence | effort |
|---|---|---|---|---|
| 1 | **WT weight layout** (no on-chip transposes) | **-4.2 ms** (16.8 -> 12.6) | measured end to end, 1-stream output identical | medium: prepare / engine relayout + rebuild every NEFF; the prefill MoE form is open (section 4) |
| 2 | `MOE_SLOTS=row` (no union top-k at R = 1) | -0.43 | measured (seg20, layer_bench) | none (graph switch), parity check |
| 3 | MoE routing + expert gather off the critical path (fused router / top-k / gather NKI kernel issuing the expert DMAs from the indices; or keep the DMA busy with the next layer's static weights during routing) | -0.6 to -1.0 | probe `PROF_FIXED`: 27 us per MoE exposed (1.1 ms) | high (NKI) |
| 4 | gqa decode attention kernel (1 row, int8 KV, P = 32..4096) | -0.8 to -1.1 | 195 us vs ~60 us weights per gqa | high (NKI; the Llama path's ATTN_DEC=nki is the template) |
| 5 | gdn decode kernel (conv step + q/k norms + gating + recurrence + gated norm fused) | -0.5 to -0.8 | 112 us vs 79 us floor; DVE 38 us | high (NKI) |
| 6 | fewer launches: one 40-layer segment, head inside it | -0.3 to -0.4 | ~0.2 ms engine-vs-device per launch; seg40 compile needs > 30 GB RAM (seg20 peaks at 15 GB) | low-medium (bigger build host; engine change for the head) |
| 7 | all-reduces (80 x 8 KiB) | -0.2 to -0.3 | probe `PROF_NOAR`: 10 us effective each (0.8 ms); bf16 does not help (latency); only overlap / fewer would | high, uncertain |
| - | head | 0 | at the DMA wall after WT | |
| - | flags, LIN_CSE, ROUTER=shard, AR_BF16, RLOGIT, GDN forms | 0 | measured (section 6) | |

Sum: 12.6 (WT) - 0.43 (row) = **~12.2 ms measured-or-switch (82 tok/s)**; with items 3-6 at their midpoints ~9.4 ms (~106 tok/s).
**<= 10 ms (100 tok/s) without MTP is reachable only with WT plus at least two of the NKI items 3-5** (or 3 + 4 + 6). The hard
floor of this layout is ~6.6 ms of weight bytes + ~0.5 ms of unavoidable latency (all-reduces, launch) = ~7 ms (~140 tok/s).
MTP (k = 1) currently buys 4 % at 1 stream (16.8 -> 16.2 ms per token); its verify segment and MTP head get the same WT gain.

## 8. R = 8 (and R = 2 / 4)

Production at 5-8 rows: **101 ms per step**. There is no 8-row segment in the set (the bound 8-row gdn compiled to 5.0 ms per gdn;
docs/neuron-reap-launches.md), so a step is 80 sub-block launches + head: 30 x gdn_8 (dynamic state) 1.36 + 10 x gqa_8 0.54 + 40 x
moe_8 0.83 + head_8_8 2.15 = 79 ms of in-launch time + 81 launches x ~0.27 ms.

Profile of the 8-row sub-blocks (`prof/p3`):
* **gdn_8**: the per-row state contractions (`einsum('rhkd,rhk->rhd')` over 8 rows x 16 heads of a [128 x 128] f32 state) are
  compiled into thousands of tiny f32 PE tiles (`LDWEIGHTS fp32 16x128` 2054 x 502 ns, plus f32 state transposes `MATMUL fp32 128x16
  mov128` 2048) and ACT/DVE copies; with bound slots the same form compiles even worse (5.0 ms). At R = 4 the compiler picks a
  clean form (0.43 ms, `fp32 4x128 mov4` tiles).
* **moe_8**: up to 64 distinct experts = 104 MB per launch: DMA-bound at ~207 GB/s (floor ~0.42 ms) plus the weight transposes
  (PE 0.5 ms, ACT 0.47 ms busy).
* gqa_8: 0.54 ms (weight transposes + 8-row attention).

Fix measured: **`GDN_RSPLIT=4`** (the state contractions in two groups of 4 rows, exact): bound gdn_8 5.0 -> **0.50 ms** (WT),
dynamic 1.36 -> 0.65; with WT moe_8 0.83 -> 0.72, gqa_8 0.54 -> 0.48, head_8_8 2.15 -> 1.24. The bound 8-row gdn makes an 8-row
segment safe (no dynamic state DMA, the nonfinite-race path of docs/neuron-reap-nonfinite.md): **seg20 R = 8 WT + GDN_RSPLIT = 21.3
ms per launch** (221 s / 15.1 GB compile). End to end with the WT set: **8 streams 101 -> 45.5 ms per step (aggregate 79 -> 176
tok/s)**, 0 nonfinite. The remaining 8-row floor is the MoE bytes (~4.2 GB per step = ~17 ms at the wall).

R = 2 / R = 4 (layer_bench per 20-layer launch, prod -> WT): 10.44 -> 7.96 and 15.56 -> 12.47 ms; end to end 23.8 -> 18.0 and
34.3 -> 27.3 ms per step. R = 3 uses the 4-row bucket, R = 5-7 the 8-row one (padding), as before.

## 9. What a deployment of WT needs (nothing was deployed)

The engine binds ONE copy of every weight tensor to all NEFFs (prefill, decode, MTP, verify), and HBM has no room for a second
layout (tensors 14.3 of 16 GiB per core), so WT is all-or-nothing per graph set:

1. **Weights in the WT layout.** Either a load-time relayout in `qwen35_model.cpp` when the set's metas say `wt` (the same
   pattern as the existing `egu_t` relayout: per tensor a host-side transpose before `put`; ~17 GB over two ranks, a few seconds of
   CPU per GB), or a WT prepared set (`mkwt.py` / `to_wt` rules: 2-D projections, `rw` and the head `w` transposed, `egu`
   `[E, 2 EIl, H] -> [E, H, 2 EIl]`; `ed`, scales, norms, conv, embeddings unchanged; MTP `mtp.bin` the same rules). The
   engine's `ROUTER=shard` slicing reads `rw` as `[E, H]` and would need the transposed indexing (ROUTER=shard is not used).
2. **Every NEFF of the set rebuilt with `WT=1`**: decode segments and sub-blocks (R = 1, 2, 4, 8), heads, the Q = 64 / 256 / 1024
   prefill buckets, MTP (`mtp_*`, `mtpc`) and the verify segments (`segv20_*`). Not done / not validated here: the grouped prefill
   MoE (`moe_grouped_idx` / `moe_grouped` index `egu` as `[E, 2 EIl, H]`: needs the transposed gather and its own timing), the MTP
   and verify graphs (they use the same `lin` / `moe_routed`, so `WT=1` should carry over, but untested), the chunked gdn prefill and
   blocked gqa prefill at Q = 256 / 1024 (only Q = 64 was built; same `lin` calls).
3. Validation as for the earlier graph-set changes: logits parity (`par.sh`), prefix-cache exactness, the prompt matrix (TTFT must not
   regress: check the prefill MoE forms), fault soak. The 1-row decode outputs were byte-identical in the end-to-end test here.

Order of work suggested: WT for the decode segments + heads + the prefill buckets (largest, measured win), with `MOE_SLOTS=row`
and `GDN_RSPLIT=4` (R = 8 segment) in the same rebuild.

## 10. Files

* `neuron/qwen35_graphs.py`: `WT`, `WT_MOE`, `GDN_RSPLIT`, `to_wt()`; probes `PROF_NOAR`, `PROF_FIXED`. `neuron/tests/test_qwen35_graphs_cpu.py`: `WT=1` hook.
* `deploy/neuron-reap/decode-prof/`: box drivers (`setup.sh`, `dec.sh`, `dec1.sh`, `xb.sh`, `prof.sh`, `ana.py`, `cls.py`, `ext.py`, `mkwt.py`, `lb.sh`, `srv.sh`).
* Raw results: `s3://behemoth-neuron-models-912935854432/tmp/ornith-decode-prof/` (`log/` layer_bench tables, `prof/` profile
  summaries + analyses, `run/` engine runs).
