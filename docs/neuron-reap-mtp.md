# MTP speculative decoding for the Ornith GM tune on the trn1.2xlarge, on the 20-layer segment decode graphs

Model `cgg507/Ornith-1.5-35B-A3B-REAP20-GM-RP-v3-FP8` (Qwen3.5-35B-A3B arch, REAP-20), TP = 2 fp8, int8 KV8, PAGES 4097, SLOTS 9,
box i-0ffdea4dcf9b4c089. Branch `reap-mtp2` (from `reap-launches`: bound gdn graphs + seg20 decode segments + prefix cache).
Server args and env as production (`--prefill-chunk 256 --max-batch-requests 8 --prefix-cache-gb 4`, prefix-cache env,
`OMP_WAIT_POLICY=PASSIVE`). Box work dir /mnt/scratch/mtp2 (engine b1 / b2, NEFFs g/, graph sets gs_*, run/<tag>/server.log,
logs st.log / soak1.log / sp.log / ad.log). Drivers: `deploy/neuron-reap/mtp2/` (+ `deploy/neuron-reap/pc.py`, new cases k1, k2
code and u1 plain chat). Session 2026-11-15, 13:45-17:00 UTC. The first pass (old sub-block graphs, 2026-11-15 morning) is
summarised in section 9.

## TL;DR

* **New: bound verify SEGMENTS** (`segv20_1_Q_P`: the seg20 decode segment traced with `VERIFY=1`, R = 1, Q = 1 + k tokens; the
  per-position gdn states of all 15 gdn layers are extra outputs bound to the vslot slices, no dynamic state DMA). A verify step
  is 2 segment launches + head, like a decode step. Per-layer verify graphs would cost 80 launches (~45 ms per step) and lose
  against the 16.7 ms segment decode step outright, so the mixed design was not built.
* **Exactness: proven.** Strict mode with the verify bucket pinned (`STRICT_Q=4`, new): k = 1, 2, 3 token-identical to k = 0 on
  13 / 13 requests each (11 prompts: GM with `<think>`, tool call + tool result, 16-turn RP history 11.5k, code, plain chat,
  9.5k GM, 38k; plus 2 prefix-cache hits), also with 30 % forced draft corruption (rejections at every position) and with the
  prefix cache off. The first pass's k = 3 mismatch is numerics of the Q = 2 / 3 / 4 bucket compiles, not rollback: unpinned
  strict k = 3 differs from strict k = 0 exactly as the pinned Q = 4 family differs from the Q = 2 family (section 3).
* **Faults: 0.** k = 1 soak 10 fresh starts x 5 requests: 0 `Nonfinite`, the 10 starts' outputs identical (one md5). All
  strict / speed runs (~25 server starts, ~250 requests, k = 0..3): 0 nonfinite.
* **Speed (1 stream, 256 tokens, quiet box): k = 1 is the only k that pays, and only on predictable text.** k = 0 59.7 tok/s
  short / 51 at 9.5-11.5k / 35.5 at 38k. k = 1: GM world-bible 9.5k **+28 %** (65.6), tool call **+20 %** (61.0), code **+21 %**
  (72.1), 38k GM **+25 %** (44.4), short GM +8 %, RP history +1 %, plain chat -1 %, creative story -10 % (53.2). A k = 1 step
  costs 25 ms (verify segments 19.5-22 ms + head 2.2 + draft ~3) vs 16.7 ms: break-even at 1.50 tokens per step (first-draft
  acceptance 50 %) at short context, 1.41 at 9.5k, 1.31 at 38k. k = 2 / 3 lose (the Q = 3 / 4 segments cost more than the
  second / third draft brings: acceptance of drafts 2 and 3 is ~15-25 %).
* **Auto-off rule** (engine change: `BEHEMOTH_NEURON_SPEC_MTP_ADAPT=1` now also adapts spec_mtp = 1, i.e. k in {0, 1} by the
  measured acceptance and step times, every 16th step probes k = 1): story / chat prompts run at k = 0 speed (59.0 vs 53.2 fixed
  k = 1), predictable prompts keep most of the gain (cached GM 9.5k 68.2, 38k 49.0, code 66.4), with a ~100-200 token lag
  after a low-acceptance request (section 5.3). 2+ streams never draft (no bound R > 1 verify buckets; section 6: a 2-row
  verify segment would gain at most ~5 %).
* **HBM: k = 1 set fits, the full Q = 2..4 set does not.** gs_k1 (production + seg20 + 6 segv Q = 2 + mtp_1_{1,2}_32 + head
  weights) **15.687 GiB per core** (+0.51 GiB over the seg20 production set, 0.31 GiB left); 16 verify segments (Q = 2..4) fail
  to allocate the KV cache at load. Host RSS 18.2 GB with the prefix cache filled (17.0 GB without), fits 32 GiB.
* **Verdict: GO for k = 1 + ADAPT on single-stream decode** (+20-28 % on GM world-bible / tool-call / code / long-context
  requests, no loss on chat / story, unchanged at 2-8 streams; exact in strict mode; 0 faults; fits HBM with 0.31 GiB to spare).
  **NO-GO for k >= 2** (15-35 % slower than k = 1, and the Q = 3 / 4 set does not fit next to Q = 2 at all page buckets). Recipe
  in section 8 (graphs + head staged under /root/mtp2-stage on the box, patches in `deploy/neuron-reap/mtp2/`).

## 1. What was built

* **Head weights**: `qwen35_fetch_mtp.py cgg507/Ornith-1.5-35B-A3B-REAP20-GM-RP-v3-FP8` (HTTP range requests on
  `model-mtp.safetensors`, 1254 tensors, 693 MB, ~2 min) and `qwen35_prepare.py RAW OUT --mtp --tp 2` -> /mnt/scratch/mtp2/v3mtp
  (`mtp.json`, `r{0,1}/mtp.bin` 344 MB each; 205 experts). Engine: `Qwen35 MTP: ready ... per core 0.38313 GiB (head weights +
  its KV cache)`.
* **Generator** (`neuron/qwen35_graphs.py`): a `seg_...` kind traced with `GDN_SLOT=bound VERIFY=1` and Q > 1 lists
  `l<j>.kstate@<i>` / `l<j>.cstate@<i>` (gdn layer j, position i = 1..Q-1) as extra outputs; `gdn_block` already returned them.
  **Engine** (`qwen35_model.cpp`): the load check accepted bound verify graphs only as gdn sub-blocks; now also as segments
  (R = 1). Binding needed no change (`bind_state` already maps `l<j>.kstate@<i>` to layer l + j's slice at vslot[i]).
  `BEHEMOTH_NEURON_QWEN35_STRICT_Q=q`: strict launches use a verify bucket of at least q tokens (test switch).
  `neuron_engine.cpp`: ADAPT for spec_mtp = 1 (section 5.3).
* **NEFFs** (`deploy/neuron-reap/mtp2/g1.sh` = lbuild.sh specs `segv20_1_<Q>_<P>:seg_gdn_gdn_gdn_gqa x5:1:<Q>:<P>:VERIFY=1`,
  MOE=gather, KV8 int8, GDN_SLOT=bound): Q = 2 at P = 32, 128, 512, 1024, 2048, 4096; Q = 3 / 4 at P = 128, 512, 1024, 2048, 4096.
  158-212 s each, ~15 GB peak RAM, one at a time (NEFF 4.7-8.0 MB). `mtp_1_{1..4}_32` (24 s each) and `head_3_3` (16 s).

## 2. Graph sets and HBM per core (neuron-monitor after load)

| set | NEFFs | per core | model code | tensors |
|---|---|---|---|---|
| production kv8-bound (launches doc) | 54 | 14.772 GiB | 0.341 | 13.603 |
| + seg20 (launches doc, gs_final) | 72 | 15.179 | 0.741 | 13.603 |
| **gs_k1** = + segv20_1_2_{32..4096} + mtp_1_{1,2}_32 + MTP head | 80 | **15.687** | 0.862 | 13.988 |
| gs_k23 = + segv Q = 2, 3, 4 at P 128 / 1024 + mtp_1_{1..4} + head_3_3 | 83 | 15.726 | 0.896 | 13.991 |
| gs_mtpF = all 16 segv + mtp_1_{1..4} + head_3_3 | 93 | **does not load** (`alloc L039.attn.kv failed`, 3 / 3 starts) | | |

The MTP head + its 64 MiB KV cache is +0.385 GiB of tensors; each verify segment ~15-20 MB of code (shared scratchpad unchanged
at 0.812). So k = 1 fits with 0.31 GiB left; a production set with k = 2 / 3 at every page bucket does not.

## 3. Exactness (strict mode, greedy, 200 new tokens; prefix cache on unless noted)

Strict mode (`STRICT_VERIFY=1 STRICT_PAGES=4096`) runs every token through the verify segments at 4096 pages, so streams with and
without drafts share one graph family. Cases (pc.py) s1, s2 (short GM + `<think>`), a1 (9.5k GM), a3 (multi-turn on a1's
prefix), t1 (tool call), t2 (tool result turn), h1 (16-turn RP history 11.5k), k1 (code), u1 (plain chat), b1 (38k GM), then s1,
a1 again (prefix-cache hits at 256 / 9472 tokens). Reference: strict k = 0.

| run | vs | result |
|---|---|---|
| stq1: k = 1, STRICT_Q = 4 | stq0 (k = 0, Q = 4) | **13 / 13 identical** |
| stq2: k = 2, STRICT_Q = 4 | stq0 | **13 / 13 identical** (10 in the first run, whose server was OOM-killed on the host by a parallel graph compile at the 38k prompt; b1, s1, a1 in a rerun) |
| stq3: k = 3, STRICT_Q = 4 | stq0 | **13 / 13 identical** |
| stq3c: k = 3 + `SPEC_MTP_CORRUPT=30` (forced rejections) | stq0 | **13 / 13 identical** |
| stq3n: k = 3, prefix cache OFF | stq0 (cache on) | **13 / 13 identical** (and cache hits == cold) |
| stn3: k = 3 unpinned (Q = 2 / 3 / 4 by draft count) | stn0 (k = 0 unpinned = Q = 2) | 0 / 13 (first difference at char 22-761) |
| stn0 (Q = 2 family) | stq0 (Q = 4 family) | 0 / 13 (char 22-226): the two compiles round differently |

The separating test: the state rollback (vslot commit of the accepted position, reserved slots, mtp_draft commit), the C-state
and the page-bucket pinning are exact - pinning only the verify Q makes every k, with or without forced rejections and with or
without the prefix cache, bit-identical. What differs is the Q = 2 / 3 / 4 segment compiles themselves: the same tokens through
the Q = 2 and the Q = 4 graph fork within 22-226 characters, and unpinned k = 3 (which mixes Q = 2, 3, 4 launches) follows
whichever bucket it runs. Numerics-only. (The forks are near-ties of the greedy argmax; acceptance at k = 1 is 0.4-0.85 per
prompt type, i.e. the draft head's top-1 agrees with the verify graph's on that fraction of positions.)

Normal mode (production-like): k = 0 decodes through `seg20_1_1_P`, k = 1 through `segv20_1_2_P`: another compile, so the k = 1
stream is not token-identical to the k = 0 stream (s1: first difference at char 734), as on the 397B. It is deterministic: the
10 soak starts gave one output.

## 4. Faults

| config | fresh starts / requests | Nonfinite |
|---|---|---|
| k = 1 normal mode, gs_mtp (segv Q = 2 at P 32 / 128): `pc.py soak 3` (s1, s2, s1 cached) + u1 + k1 per start | **10 / 50** | **0** (10 / 10 outputs identical) |
| strict k = 0..3 (section 3), 9 starts | 9 / ~110 | 0 |
| speed runs k = 0..3, gs_k1 / gs_k23 (section 5) | 6 / ~150 incl. 2-8 streams, 38k | 0 |
| ADAPT k <= 1 (b2): 1 + 3 fresh starts | 4 / 38 | 0 (section 5.3) |

The verify segments contain no dynamic state DMA (GDN_SLOT=bound), so the 1-row race of the first pass (docs/neuron-reap-nonfinite.md)
is gone for k >= 2 too (strict k = 2 / 3: 0 faults over 5 starts; the first pass faulted 2 / 2 runs).

## 5. Speed (quiet box, per-request decode tok/s from the server request_metrics; tok/step = tokens emitted per decode step)

### 5.1 1 stream by prompt type and context, 256 new tokens (gs_k1; `pc.py seq`)

| prompt (tokens) | k = 0 | k = 1 (tok/step) | k = 1 vs k = 0 | k = 1 no prefix cache |
|---|---|---|---|---|
| s1 short GM `<think>` (453) | 59.7 | 64.6 (1.63) | +8 % | 64.2 |
| s2 short GM (449, cached 256) | 59.7 | 57.1 (1.44) | -4 % | 56.7 |
| k1 code, no think (75) | 59.6 | **72.1** (1.82) | **+21 %** | 71.6 |
| k2 code request in the 9.5k GM context (9502) | 50.8 | 54.2 (1.51) | +7 % | 54.3 |
| u1 plain chat, no think (33) | 59.7 | 59.1 (1.49) | -1 % | 58.6 |
| t1 tool call (9825) | 51.0 | **61.0** (1.71) | **+20 %** | 61.3 |
| h1 16-turn RP history (11520) | 50.9 | 51.7 (1.44) | +1 % | 51.6 |
| a1 GM 9.5k world bible (9486) | 51.1 | **65.6** (1.82) | **+28 %** | 65.3 |
| a3 multi-turn on a1 (9552, cached 9472) | 51.3 | 55.4 (1.53) | +8 % | 55.2 |
| b1 GM 38k (38027) | 35.5 | **44.4** (1.64) | **+25 %** | 45.3 |
| lbench story prompts, no think (26) | 59.4 | 53.2 (1.34) | **-10 %** | 53.0 |

Context: 1k / 8-11k / 38k = 59.7 / 51 / 35.5 tok/s at k = 0. The prefix cache changes TTFT only (identical decode rates).

### 5.2 Streams (lbench chat prompts, 250 tokens, per stream; k > 0 drafts only while one stream decodes)

| streams | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| 1 | 59.4 | 53.2 (story prompt: 1.34 tok/step) | 42.1 (1.37) | 34.7 (1.37) |
| 2 | 40.6 / 42.1 | 40.6 / 42.1 | 40.4 / 41.9 | 40.6 / 42.0 |
| 4 | 27.1-29.1 | 27.1-29.1 | 28.4-29.2 | 27.2-29.1 |
| 8 | 9.4-9.9 | 9.4-10.0 | 9.4-10.0 | 9.4-10.0 |

k = 2 / 3 by prompt (set gs_k23: Q = 2 / 3 / 4 verify segments at P 128 and 1024 only, so b1 at 38k ran without drafts;
k = 1 on this set reproduced gs_k1 within 0.1 tok/s):

| prompt | k = 0 | k = 1 | k = 2 (tok/step) | k = 3 (tok/step) |
|---|---|---|---|---|
| s1 short GM | 59.7 | **64.6** | 53.4 (1.73) | 44.5 (1.75) |
| k1 code | 59.6 | **72.1** | 57.1 (1.86) | 48.8 (1.91) |
| u1 plain chat | 59.7 | **59.1** | 47.2 (1.53) | 39.7 (1.56) |
| t1 tool call 9.8k | 51.0 | **61.0** | 51.6 (1.82) | 42.7 (1.81) |
| h1 RP history 11.5k | 50.9 | **51.7** | 42.3 (1.50) | 39.3 (1.67) |
| a1 GM 9.5k | 51.1 | **65.6** | 54.1 (1.91) | 48.1 (2.03) |

The second and third drafts add 0.05-0.2 tokens per step and cost a Q = 3 / 4 segment (verify step +4-6 ms each): k = 2 is
15-20 % slower than k = 1 everywhere, k = 3 25-35 % slower; on the old launch-bound graphs (first pass) the extra rows were
nearly free, now they are not.

### 5.3 Step cost and the auto-off rule

LTRACE (rank 0, ms per step, R = 1): decode Q = 1 wall 16.6-17.5 = segments 14.4-15.3 + head 2.05 + read 0.08; verify Q = 2 wall
22.0-24.6 = segments 19.5-22.1 + head 2.17 + read 0.16, plus the draft launch (mtp_1_2_32 / mtp_1_1_32) ~2.5-3 ms outside the
step. So a k = 1 step is ~25 ms at short context (27.7 at 9.5k, 36.9 at 38k) against 16.7 / 19.6 / 28.2 ms for k = 0: the
2-token segment is 35-50 % more device time than the 1-token one (MoE reads ~1.7x the experts, attention twice the rows), the
launch overhead MTP used to amortise is gone. Break-even tokens per step = 1.50 / 1.41 / 1.31 (first-draft acceptance 50 / 41 /
31 %). Measured acceptance by type: code 0.83, GM world-bible / tool calls 0.71-0.83, short GM 0.44-0.64, RP history and
chat 0.44-0.49, creative story 0.34.

**Auto-off rule: `BEHEMOTH_NEURON_SPEC_MTP_ADAPT=1` with `SPEC_MTP=1`.** The engine's adaptive chooser (written for the 397B,
k = 1..4) picks the k maximising (1 + P(draft 1 accepted)) / (verify_ms(k) + draft_ms(k)) from running EMAs of the measured
step times and acceptance, and every 16th step probes one draft more; it was disabled for spec_mtp = 1, which is now allowed
(k in {0, 1}). It learned verify_ms = 17.9 / 24.6 ms and draft_ms 3.1 ms on this box, i.e. it drafts while P(accept) > ~0.53.
Same sequence of requests as 5.1 (engine b2, gs_k1, one server, requests in order):

| prompt | k = 0 | k = 1 | ADAPT (drafted / accepted, tok/step) |
|---|---|---|---|
| lbench story x 2 | 59.4 | 53.2 | **59.0 / 58.9** (20 / 7, 15 / 4: off) |
| s1 short GM | 59.7 | 64.6 | 59.9 (16 / 10: still off after the story prompts) |
| s2 short GM | 59.7 | 57.1 | 59.6 |
| k1 code | 59.6 | 72.1 | 66.4 (103 / 78: switched on mid-request) |
| k2 code in 9.5k context | 50.8 | 54.2 | 54.8 |
| u1 plain chat | 59.7 | 59.1 | 58.4 (15 / 2: off) |
| t1 tool call | 51.0 | 61.0 | 64.3 (256 tokens here vs 164: other text) |
| h1 RP history | 50.9 | 51.7 | 50.6 |
| a1 GM 9.5k | 51.1 | 65.6 | 54.2 (81 / 49: switched on late) |
| a3 multi-turn | 51.3 | 55.4 | 50.4 |
| b1 GM 38k | 35.5 | 44.4 | **49.0** (132 / 113) |
| s1 / a1 again (cached) | 59.9 / 51.3 | 65.0 / 65.8 | **65.0 / 68.2** |

ADAPT removes the losses (story / chat run at k = 0 speed) and keeps most of the gains, but the acceptance EMA is global and
slow (weight 0.05 per drafted step, one probe per 16 steps while off), so after a low-acceptance request it takes ~100-200
tokens to switch back on (s1, a1 above). Follow-up if wanted: reset the acceptance estimate per request or weight the probes
higher. ADAPT soak: 4 fresh starts x 7-17 requests, 0 nonfinite. Note that with ADAPT the k of a step depends on measured timings, so
normal-mode outputs are not reproducible run to run (every step is a greedy decode through either the seg20 decode or the
verify graph family; fixed k = 1 is deterministic). Rule for production: **MTP on for 1 decoding stream, k = 1, ADAPT=1; off (automatic) with 2+ streams; k >= 2 never.**

## 6. Where MTP loses, and 2+ streams

* Low-acceptance text (story writing, plain chat, RP turns): -1 to -10 % at k = 1 without ADAPT.
* k >= 2: every extra draft adds a verify row (Q = 3 / 4 segments) for a 15-25 % chance of an extra token.
* 2+ streams: the engine drafts only for one decoding stream (`mtp_max_drafts(n > 1)` needs R x Q verify buckets; bound verify
  is R = 1 only). Estimate for 2 streams: two R = 1 Q = 2 verify launches + 2 drafts ~50 ms for 2 x 1.7 tokens = 34 tok/s per
  stream < 42.0 at k = 0 (loses); a bound R = 2 x Q = 2 verify segment (T = 4, ~seg20_4 cost 31-34 ms) + 2 drafts ~38 ms for
  1.7 tokens each = ~44.7 tok/s per stream vs 42.0: +5 % at best, not worth a new state layout and 6 more NEFFs in 0.31 GiB.
  The first pass's ">= 4 streams off" was the same rule (it never drafted with company, BY_ROWS empty).

## 7. Memory

HBM: section 2 (k = 1 set 15.687 GiB per core of 16). Host RSS of the server (gs_k1, MTP loaded): 18.2 GB after the speed
matrix with the prefix-cache host tier filled (4 GB KV pages + state snapshots), 17.0 GB with the prefix cache off; fits the
32 GiB production host (the dev box has 30 GiB and OOM-killed a server once when a 15 GB graph compile ran next to it).

## 8. Production recipe (NOT applied, NOT pushed)

1. **Engine**: `deploy/neuron-reap/mtp2/engine.patch` = `git diff b36ec5b..reap-mtp2 -- neuron/cpp/qwen35_model.cpp
   neuron/cpp/neuron_engine.cpp neuron/cpp/layer_bench.cpp` (the launches engine.patch + bound verify segments + STRICT_Q + ADAPT
   k = 1). `git apply --check` passes on BEHEMOTH-REVERSE f0af7af. Graph sets without verify segments behave as before; an older
   engine rejects the bound verify segments at load, so deploy the engine first.
2. **Graphs**: new prefix `graphs/ornith-reap20-v3-kv8-seg20-mtp` = a copy of `graphs/ornith-reap20-v3-kv8-seg20` (the launches
   set: kv8-bound + seg20_{1,2,4}_1_P) + the 8 directories of `/root/mtp2-stage/graphs-mtp/` on the box: `segv20_1_2_{32,128,512,1024,2048,4096}`,
   `mtp_1_1_32`, `mtp_1_2_32` (graph.neff + meta.json each, 36 MB). Staging target `s3://912935854432-neuron-build/reap/mtp/graphs-mtp/`.
3. **Head weights**: `/root/mtp2-stage/mtp/` (`mtp.json`, `r0/mtp.bin`, `r1/mtp.bin`, 689 MB) -> `models/ornith-reap20-v3-fp8/mtp/`
   in the models bucket (staging `s3://912935854432-neuron-build/reap/mtp/mtp/`). The entrypoint pulls the model prefix
   recursively (also in the stream-load phase 1, which only excludes `model-layer-*`), so it lands in
   `/data/models/ornith-reap20-v3-fp8/mtp`. md5 + sizes: `/root/mtp2-stage/MANIFEST.md5`.
4. **tfvars** (`deploy/neuron-reap/mtp2/tfvars.patch`, against f0af7af, includes the launches tfvars change): graphs prefix
   `graphs/ornith-reap20-v3-kv8-seg20-mtp`, `OMP_WAIT_POLICY=PASSIVE`, `BEHEMOTH_NEURON_MTP_DIR=/data/models/ornith-reap20-v3-fp8/mtp`,
   `BEHEMOTH_NEURON_SPEC_MTP=1`, `BEHEMOTH_NEURON_SPEC_MTP_ADAPT=1`. No server-argument change. Rollback = drop the three
   MTP env lines (the graph set works without them).
5. **Generator** (to rebuild the NEFFs from source): `deploy/neuron-reap/mtp2/graphs.patch` (qwen35_graphs.py + lbuild.sh from
   the production version 58745c9) and `tools.patch` (qwen35_fetch_mtp.py + the `--mtp` block-fp8 fix of qwen35_prepare.py).

## 9. First pass (superseded; old sub-block graphs, before the segment and nonfinite fixes)

Head + 29 sub-block verify / mtp NEFFs on v3gkv8_4k (HBM 16.34 of 17.18 GB): k = 1 14.9 -> 23.6-31.5 tok/s at 1 stream
because the 80-launch decode step (45 ms) was launch-bound and a 2-token verify cost about the same; k = 2 / 3 faulted (1-row
dynamic gdn state DMA race, since fixed by GDN_SLOT=bound); strict k = 3 0 / 4 identical (cause now separated: Q-bucket
numerics, section 3).
