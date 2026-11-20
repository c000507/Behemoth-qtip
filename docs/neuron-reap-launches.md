# Ornith REAP20 v3 decode on trn1: fewer launches (20-layer segment NEFFs)

Box i-0ffdea4dcf9b4c089 (trn1.2xlarge), aws-neuronx-runtime-lib 2.33.10, neuronx-cc 2.26, model
`cgg507/Ornith-1.5-35B-A3B-REAP20-GM-RP-v3-FP8` (/mnt/scratch/v3prep), production graph set `graphs/ornith-reap20-v3-kv8-bound`
(/mnt/scratch/v3gb: PAGES 4097, SLOTS 9, KV8 int8, bound gdn_{1,2,4}_1_32, dynamic gdn_8_1_32). Branch `reap-launches`
(from reap-nonfinite). Server args and env as production (`--prefill-chunk 256 --max-batch-requests 8 --prefix-cache-gb 4`,
prefix cache env on). Box work dir /mnt/scratch/launches (engine b2, NEFFs g/, graph sets gs_*, logs j*.log,
run/<tag>/server.log). Session 2026-11-15, 11:50-14:30 UTC.

## TL;DR

* **Measured**: the ~46 ms decode step is NOT host work between launches (all host gaps + input writes + logits read = 0.2 ms
  per step). It is the cost *inside* each synchronous `nrt_execute`: an Ornith sub-block NEFF costs 0.47-0.58 ms per call in the
  engine but 0.18-0.28 ms when the same NEFF is launched back to back (layer_bench). About 0.3 ms per launch is the cost of
  switching to another NEFF + the completion round trip; x 80 launches = ~23 ms of the 45 ms.
* **Lever that pays: 20-layer decode segment NEFFs** (`seg_gdn_gdn_gdn_gqa x 5`, the existing `seg_` graph kind, now with
  `GDN_SLOT=bound` state slots): 81 launches -> 3 per token. **1 stream 22.1 -> 49.8 tok/s (44.9 -> 20.0 ms/step, 2.25x)**,
  2 streams 17.9 -> 30.4, 4 streams 15.5 -> 23.5 tok/s per stream; at 9-11k context 19 / 14.7 -> 38-40 tok/s.
  Token-identical to production on 13 / 13 requests (short, 9-11k GM prompts with `<think>`, tool calls, 16-turn history,
  prefix-cache hits), 10 / 10 fresh starts x 2 x 300 tokens with 0 nonfinite and 20 / 20 identical outputs, HBM +12 MB per core.
* **Second lever, env only: `OMP_WAIT_POLICY=PASSIVE`.** The sampling code's idle OpenMP threads spin (7 threads at 20-30 % CPU
  each on the 8-vCPU box) and delayed the start of the two rank-worker jobs by 1.5-3 ms per step (new LTRACE `dispatch`
  column: 2.7 -> 0.01 ms). With segments: **1 stream 49.8 -> 59.6 tok/s (16.6 ms/step), 2 streams 42.0, 4 streams 29.2,
  8 streams 10.0**; on the production graphs alone 22.1 -> 24.8 tok/s. Token-identical (`OMP_NUM_THREADS=4` does the same).
* **Combined: 22.1 -> 59.6 tok/s at 1 stream (2.7x), 17.9 -> 42.0 at 2, 15.5 -> 29.2 at 4, 9.6 -> 10.0 at 8 streams.**
* Did not pay: async execution (+22 % alone, nothing on top of segments), R = 8 segments with bound gdn (5.5 tok/s, the
  known bad bound 8-row compile), 40-layer segment (compile OOM on 30 GB RAM; would save one launch of ~0.3 ms).

## 1. Where the time goes (measured)

New engine switch `BEHEMOTH_NEURON_QWEN35_LTRACE=N`: per (R, Q) bucket and rank, host-side split of every `execute()` call;
one stderr line per N calls. Columns are ms per decode step (rank 0; rank 1 the same within 0.1 ms):

| config (1 stream unless noted) | wall | input writes | host gaps between calls | in nrt_execute: gdn / gqa / moe / segment | head | logits read | launches |
|---|---|---|---|---|---|---|---|
| production, R = 1 | **44.9** | 0.04 | 0.07 | 15.29 / 5.81 / 18.97 / - | 2.05 | 0.09 | 80 + head |
| production, R = 4 | 63.9 | 0.04 | 0.13 | 20.24 / 7.64 / 30.91 / - | 2.22 | 0.33 | 80 + head |
| `ASYNC=1` (implicit async + drains), R = 1 | 37.0 | 0.04 | 0.06 | 2.38 / 1.28 / 3.81 (submit only) / - | 26.69 (head call waits for the queue) | 0.08 | 80 + head |
| seg4 (4-layer segments), R = 1 | 21.8 | 0.04 | 0.02 | - / - / - / 17.16 | 2.05 | 0.09 | 10 + head |
| **seg20 (20-layer segments), R = 1** | **20.0** | 0.04 | 0.01 | - / - / - / 15.15 | 2.05 | 0.08 | 2 + head |
| seg20, R = 2 | 26.6 | 0.04 | 0.04 | 23.33 | 2.24 | 0.17 | 2 + head |
| seg20, R = 4 | 36.3-38.8 | 0.04 | 0.01-0.09 | 32.5-34.6 | 2.22 | 0.31 | 2 + head |

layer_bench (same NEFF back to back, both ranks, best of 8 batches of 50; `LB_ASYNC=1` = implicit async + drain per batch):

| NEFF | sync ms / launch | async ms / launch | in the engine (production), ms / call | engine - bench |
|---|---|---|---|---|
| gdn_1_1_32 (bound) | 0.277 | 0.216 | 0.510 | +0.23 |
| gqa_1_1_32 | 0.277 | 0.279 | 0.581 | +0.30 |
| moe_1_1_32 | 0.177 | 0.185 | 0.474 | +0.30 |
| head_1_1 | 1.883 | 1.883 | 2.05 | +0.17 |
| layer_gdn_moe_1_1_32 (fused, bound) | 0.370 | 0.372 | - | (vs 0.454 for the two parts) |
| layer_gqa_moe_1_1_32 (fused) | 0.420 | 0.421 | - | (vs 0.454) |
| seg4_1_32 (4 layers) | 1.411 | 1.410 | 1.716 | +0.30 |
| gdn_4_1_32 / moe_4_1_32 / gqa_4_1_32 | 0.346 / 0.428 / 0.451 | same | 0.675 / 0.773 / 0.764 | +0.32 / +0.35 / +0.31 |

Reading: the in-NEFF work of a 1-row step is ~20 ms (30 x 0.277 + 10 x 0.277 + 40 x 0.177 + 1.88); production spends 45 ms.
The extra ~0.3 ms per call is not host code (gaps 0.07 ms per step in total, tensor sets are cached), and implicit async only
hides a third of it (37 ms): the rest is runtime / device side per launch (switching the core to another NEFF's
instruction and DMA rings, the per-execute collectives setup, completion). It is paid per launch, also by a 4-layer segment
(1.72 in the engine vs 1.41 back to back), so the only cure is fewer launches. The 20-layer segment also lets the compiler
overlap layer boundaries: 15.15 ms for 40 layers in-graph vs ~18 ms for the sub-blocks back to back.
neuron-profile was not needed for this (the per-launch split above answers it); neuron-monitor was used for HBM only.

## 2. Levers

### 2.1 Segment NEFFs (kept)

`neuron/qwen35_graphs.py` already had `seg_<attn>_<attn>...` (consecutive whole layers, each attention + its MoE, in one NEFF;
the 397B serves 20-layer segments). Changes: the gdn state of a segment can be `GDN_SLOT=bound` (R > 1: one state pair per
row, `l<i>.kstate_<r>`), so no dynamic state DMA is reintroduced; the engine binds the slot slices of layer l + i
(`layer_set` / `bind_state`; gqa states of a segment bind whole). The engine already picked the longest segment for a bucket
and falls back to the sub-block NEFFs where no segment matches (e.g. a P bucket without segment), so the switch is the
graph directory. Build: `deploy/neuron-reap/lbuild.sh OUT seg20_<R>_<P>:seg_gdn_gdn_gdn_gqa_x5:<R>:1:<P>` (MOE=gather, KV8=1
int8, GDN_SLOT=bound, PAGES 4097, SLOTS 9: the production settings). Compile per NEFF on the trn1.2xlarge: 130-160 s (R = 1),
200-215 s (R = 4), ~15 GB peak RAM each, so one at a time next to a server (two in parallel + a server got OOM-killed).

| decode tok/s per stream (server metrics, 250 tokens, short prompt) | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| production (kv8-bound) | 22.1 | 17.9 | 15.5 | 9.6 |
| + seg4 R = 1 | 45.5 | | | |
| + seg20 R = 1, 2, 4 | **49.8** | **30.4** | **23.5** | 9.6 (sub-blocks) |
| + seg20 R = 8 bound | | | | 5.5 (dropped) |

Final numbers on a quiet box are in section 3.

Context (1 stream, prefix cache on, production vs seg20, same requests): 9.5k-token GM prompt 19.0 -> 37.5-38.9 tok/s, 11.5k
14.7 -> 39.2 tok/s (the 1024-page segment bucket); a request whose page count exceeds every segment P uses the sub-block
graphs as before.

**Correctness**: `pc.py seq` cases s1, s2, a1, a2, a3, t1, t2, h1, h2, c0, s1, a1, h1 (short chat, 8k-9.5k GM system prompts
with `<think>`, tool call + tool result, 16-turn GM history ~11.5k, an 8k shared prefix; the repeats are prefix-cache hits),
200 tokens each: **13 / 13 token-identical to production** (seg run vs production run; the production run's own cached vs cold
repeats: 13 / 13, i.e. cached == cold in both). At 1 / 2 / 4 streams the bench outputs are identical too (section 3). The
segment is a different compile of the same ops, so near-tie forks are possible in principle; none showed in these runs.

**Fault soak**: 10 fresh starts x 2 requests x 300 tokens on the seg20 set (while graph compiles ran on the CPU):
**0 nonfinite, 20 / 20 outputs identical** (one md5). The segments contain no dynamic state DMA (bound slots).

**HBM per core** (neuron-monitor, after load): production 14.772 GiB (tensors 13.603, shared scratchpad 0.812, model code 0.341,
runtime 0.009, constants 0.006); final set (production + 18 segment NEFFs seg20_{1,2,4}_1_{32,128,512,1024,2048,4096}) **15.179 GiB**
(model code 0.341 -> 0.741, runtime 0.014, rest unchanged): +0.41 GiB, 0.82 GiB of the 16 GiB core left. It ran 38k-token
requests (b1 / b2) without error. If that headroom is too small, drop the R = 2 / 4 segments at P 2048 / 4096 (~0.1 GiB:
those requests fall back to sub-blocks).

### 2.2 Async execution (`BEHEMOTH_NEURON_QWEN35_ASYNC=1`, kept as an opt-in switch, not recommended)

Reworked: implicit async (`NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`, default depth 128) plus
`nrt_async_drain_queued_execs` before every host read of a launch output and at the end of every execute() / MTP launch, so host
writes (inputs, state spill / restore, prefix-cache copies) only touch an idle core. The earlier version (docs/neuron-qwen35-moe.md
10b) had no drains and diverged at token 11; this one is token-identical (3 / 3 cases). Production graphs: 22.1 -> 26.9 tok/s
(1 stream), 15.5 -> 18.2 (4 streams); with segments it adds nothing (47.1 vs 49.8 tok/s at 1 stream): with 3 launches per step
there is nothing left to overlap. The explicit queue API of nrt_async.h (`nrta_execute_schedule`, `nrta_tensor_write`,
sequence ids per execution unit) would allow overlapping the next step's input writes, but the remaining host share is 0.2 ms.

### 2.3 Fused layer NEFFs (`layer_gdn_moe`, `layer_gqa_moe`)

They compile on trn1 (39 s each) and save 0.03-0.08 ms per layer in-launch (layer_bench above), but they keep 41 launches, so
at ~0.3 ms per launch they cap near 30 ms per step (the earlier trn1 run of the old fused set: 31.5 ms); not served, superseded
by segments. (The first engine run refused the gqa fused NEFF because its meta said gdn_slot bound; fixed: the bound flag
only applies to graphs that contain gdn layers.)

### 2.4 Row MoE kernel and collectives

Not pursued: with segments the step is 15 ms of in-graph compute for 40 layers + 2 ms head; the MoE sub-block is 0.18 ms of a
1-row layer (in-launch ~0.15 by the segment slope), and the per-layer all-reduces are in-graph and latency bound (8 KiB f32
residuals, bf16 halves bytes, not latency). These are the next levers for the remaining 4x to the bandwidth roofline (section 4).

## 3. Combined (final numbers)

Quiet box (no compiles), engine b2/b3, production server args + prefix-cache env, decode tok/s per stream from the server
metrics (250 tokens, short prompts; two runs each, identical within 0.1; the last-arriving stream of a batch shown first):

| config | 1 | 2 | 4 | 8 | 1 stream, 9.5k prompt | 1 stream, 38k prompt |
|---|---|---|---|---|---|---|
| production (kv8-bound) | 22.1 (45 ms) | 17.7 / 17.9 | 14.9 / 15.5 | 9.1 / 9.6 | 20.8 | 17.6 |
| production + `OMP_WAIT_POLICY=PASSIVE` | 24.8 | 19.1 / 19.5 | 16.5 / 16.8 | 9.5 / 10.0 | | |
| + seg20_{1,2,4} (final set) | 49.8 (20.0 ms) | 35.8 / 36.9 | 24.7 / 26.3 | 9.1 / 9.6 | 43.1 | 31.6-32.2 |
| **final set + PASSIVE (recommended)** | **59.6 (16.6 ms)** | **42.0** | **27.2 / 29.2** | **9.4 / 10.0** | | |
| final + dynamic-gdn R = 8 segment `seg20d_8_32` (not recommended) | | | | 9.9 / 10.6 | | |

LTRACE of the recommended config, R = 1: wall 16.6 ms = dispatch 0.01 + input writes 0.04 + gaps 0.01-0.02 + 2 segment
launches 14.4 + head 2.05 + logits read 0.08; R = 4: 34.0 ms (segments 31.4).

Validation of the recommended config: s1 / t1 token-identical to production; 1 / 2 / 4-stream bench outputs within the
production's own batch-composition variants (at 2 and 4 concurrent streams the outputs depend on arrival timing in BOTH sets:
over 4 repeats each, d2 showed the same 2 variants and d4 the same 3 variants in production and segment runs; 1 stream is
identical); fault soak 5 fresh starts x 2 x 300 tokens + 8 concurrent + 6 prefix-cache requests: 0 nonfinite, the 10 single-stream outputs identical to each other and to the 10 of the non-PASSIVE soak (one md5); prefix-cache soak (pc.py soak, 6 short requests, 5 of them served from the cache at 256) clean.

The R = 8 segment with dynamic gdn state (+10 %) is left out: it puts the dynamic state gather back into a NEFF, which the
nonfinite study showed racing in 1-, 2- and 4-row gdn NEFFs; R = 8 stays on the production sub-blocks (dynamic gdn_8_1_32,
0 races in 10000 harness iterations).

## 4. What is left

* 40-layer segment: 1 launch fewer (~0.3 ms, 1.5 %); the trace + compile needs > 30 GB RAM (OOM-killed twice). Build it on a
  bigger CPU box if wanted.
* R = 8: the bound 8-row gdn compiles badly (also as a sub-block, docs/neuron-reap-nonfinite.md): 175 ms per step as a
  segment; the dynamic-gdn R = 8 segment gives only +10 % and reintroduces dynamic state DMA. 8 streams stay at ~10 tok/s; a
  better 8-row gdn formulation is the open item.
* In-graph time now dominates: 15 ms for 40 layers (~0.38 ms per layer) + 2.05 ms head (254 MB fp8 per rank: ~3x its byte
  time). The bandwidth roofline is ~5 ms per step; next: the segment's op count per layer (norms, routing, fp8 activation
  scaling: `LIN_CSE`, `ROUTER=shard`), the row MoE kernel at R = 2-4, and the head.
* MTP (other agent): the verify buckets are still sub-block graphs; a verify segment (R = 1, Q = 1 + k, `VERIFY=1`) needs the
  bound-verify state outputs inside `segment()` (not done).

## 5. Production recipe (patches, NOT applied)

1. **Engine** (`deploy/neuron-reap/launches/engine.patch`, = `git diff b36ec5b..reap-launches -- neuron/cpp/qwen35_model.cpp
   neuron/cpp/layer_bench.cpp`; `git apply --check` passes on BEHEMOTH-REVERSE at f0af7af): bound segments, LTRACE, ASYNC with
   drains. Graph sets without segments behave exactly as before. **Deploy the engine before the graphs** (an older engine
   rejects a bound segment NEFF at load).
2. **Graphs**: `s3://912935854432-neuron-build/reap/launches/graphs-seg20/` holds the segment NEFF directories (graph.neff +
   meta.json each): **NOT staged** - the box's instance role (neuron-inf2-role) has no s3:PutObject on
`912935854432-neuron-build/reap/launches/` (AccessDenied); per instructions no presigned URLs / local credentials were used.
The 18 directories are on the box's EBS root at `/root/launches-graphs-seg20/` (107 MB, `MANIFEST.md5`; survives a stop,
unlike /mnt/scratch) and in /mnt/scratch/launches/g/. Rebuild from source instead: `SRC=<reap-launches tree> bash
deploy/neuron-reap/lbuild.sh OUT seg20_<R>_<P>:seg_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa:<R>:1:<P>`
for R in 1 2 4, P in 32 128 512 1024 2048 4096 (JOBS=1 on a 32 GB box: ~15 GB peak RAM each, 130-220 s each, ~1 h total). The new graph prefix = a copy of `graphs/ornith-reap20-v3-kv8-bound` plus these directories, e.g.
   `graphs/ornith-reap20-v3-kv8-seg20` in the models bucket. Generator change (`deploy/neuron-reap/launches/graphs.patch`,
   qwen35_graphs.py + lbuild.sh) to rebuild them from source.
3. **tfvars** (`deploy/neuron-reap/launches/tfvars.patch`): `graphs_s3_prefix = "graphs/ornith-reap20-v3-kv8-seg20"` and
   `OMP_WAIT_POLICY = "PASSIVE"` in extra_env (the env alone is safe to ship first: +12 % on the current graphs). No
   server-argument change (ASYNC stays off).
