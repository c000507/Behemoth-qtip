# Ornith REAP20 v3 FP8 (Qwen3.5 35B-A3B, 205 experts) on trn1.2xlarge, TP=2: performance and the nonfinite fault

Box: i-0ffdea4dcf9b4c089 (trn1.2xlarge, us-east-1), aws-neuronx-runtime-lib 2.33.10, aws-neuronx-dkms 2.29.0.
Model /mnt/scratch/v3prep (TP=2 fp8, lm_head fp8 e4m3 + per-row f32 scale), graphs /mnt/scratch/v3g1k (PAGES=1025, 35 NEFFs),
baseline numbers in STATUS-reap.log.

## Fault: "Nonfinite model logits" (1-row decode)

**Update (2026-11-15, session 3): mechanism found and fixed without the speed cost, see docs/neuron-reap-nonfinite.md** (dynamic
state-slot DMA in the gdn NEFFs; `GDN_SLOT=bound` graphs + engine slot slices: 0 faults, 22.1 tok/s at 1 stream).

Symptom: single-stream greedy decode stops with "Nonfinite model logits" at a run-to-run varying position; the server must be
restarted afterwards (the NaN sits in the request's gdn state slot, and `state * keep` with keep = 0 does not clear NaN, so later requests
on that slot fault too). Baseline here: 7 of 8 fresh starts faulted within 300 tokens (scan, 2026-11-15); earlier 6/8 and 3/5.
Faulting is decided per step, not per server start: one start survived 3 x 300 tokens, another faulted on its second request.

### How it was localised (all on the box, scripts in /mnt/scratch: v3fault.sh, v3one.py, v3first.sh, v3dd.py, v3cmp.py)

1. `BEHEMOTH_NEURON_QWEN35_NANSCAN=1` (new, free until a fault): on a launch with nonfinite logits it reads back both residual buffers and
   every layer state per rank / slot. 7 faults: the first anomalous tensor is always in ONE gdn layer on ONE rank (either rank), the other
   rank's same layer clean, e.g. rank 0 L13 kstate fully NaN with its conv state finite; rank 0 L20 conv state holding 4.5e26 (rank 1: 16.8,
   normal ~15); in 2 cases both ranks' gdn input already NaN (L0/L1). State magnitudes before the fault are normal (kstate max ~1-2),
   so this is not numeric growth / overflow, and not stale state (states are zeroed at load, slot 0 is reused cleanly).
2. In one fault only 4 elements of the conv state were garbage (1.4e21 at tap 0 of channels 1531..1535): tap 0 is a copy of an input that
   earlier steps had already consumed cleanly, so those bytes were *read* wrongly inside the launch, not computed.
3. Switch matrix (fresh starts x 300 tokens):

| config | faults / starts | tok/s | outputs identical across starts |
|---|---|---|---|
| baseline (1-row decode NEFFs) | 7 / 8 | 21.3 | no (differ from token ~2 of the answer, also within one process) |
| `BEHEMOTH_NEURON_QWEN35_BARRIER=1` (ranks aligned per launch) | 1 / 1 (cut) | 21.4 | - |
| `NEURON_RT_DBG_DISABLE_DGE=1` | 1 / 1 (cut) | 21.4 | - |
| `BEHEMOTH_NEURON_QWEN35_SYNC=4` (blocking device read after every launch) | 3 / 3 | 21.1 | - |
| `NEURON_RT_DBG_FORCE_STRICT_ORDERING=1` | **0 / 8** | 8.2 | **8 / 8 identical** |
| `BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R=4` (1 row padded into the 4-row NEFFs) | **0 / 8** | 15.1 | **8 / 8 identical** |
| 2-row decode set (/mnt/scratch/v3gr2) + `MIN_DECODE_R=2` | 0 / 6 starts (12 x 300 tok) | 18.0 | no (4 variants in 8) |
| alt gdn_1_1 NEFF `GDN_FORM=fused` | 1 fault / 9 requests (5 starts) | 21.35 | no (5 variants in 8) |
| alt gdn_1_1 NEFF `GDN_DEC=1` | 1 / 1 (cut) | 16.5-18.6 | - |
| alt gdn_1_1 NEFF compiled `-O1` | 3 / 3 | 21.3 | - |

   Host-side ordering (barrier between ranks, blocking read after each launch) changes nothing; runtime strict ordering (libnrt:
   "Strict ordering is forced. Disabling HW decode mode") removes both the faults and the run-to-run divergence. So the race is inside
   the execution of a NEFF (engine / DMA-queue synchronisation), not between launches, ranks or host writes.
4. Which NEFF: `NEURON_X_DUMP=2` (max |x| per real row after every launch, rank 0) of 4 normal runs diffed against a strict-ordering
   reference run (same NEFFs, so bit-exact): prefill identical in all; the first divergence is **always right after a 1-row gdn launch
   (`gdn_1_1_32`) whose input residual was still bit-identical** (L14 pos 32, L13 pos 76, and L17 pos 89 where it was the NaN itself);
   the 4th run stayed identical for all 9600 launches.

### Root cause (best evidence)

The compiled 1-row gdn decode NEFF (`gdn_1_1_32`, neuronx-cc default flags, runtime 2.33.10) is nondeterministic on identical inputs:
occasionally a few values are read before they are written (stale on-chip bytes) inside the launch. Usually that only perturbs the
output (the known run-to-run forks); when the stale bytes are huge or NaN the state and residual go nonfinite. It disappears when the
runtime serialises engines (strict ordering) and did not show in the 4-row decode NEFFs (18/18 deterministic); the 2-row gdn NEFF still shows it
(nondeterministic), so it is a compiler/runtime scheduling hazard of the small-row gdn graphs, not the model math. It is not GDN state
numerics (no growth), not uninitialised state, not host launch order or host-side rank alignment, not DGE. Re-formulating the graph
(fused recurrence, static conv slice) or -O1 does not remove it. Report-worthy to the Neuron compiler team with gdn_1_1_32 as repro.

### Workaround (opt-in; default unchanged)

`BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R=4` (engine, commit 636b3b5): decode launches never use the 1-row NEFFs; a single stream rides in the
4-row bucket with 3 padding rows (scratch slot / page). Cost: 1 stream 21.3 -> 15.1 tok/s (47 -> 66 ms/step, = the 4-row step time);
4 and 8 streams unchanged (they already use the 4/8-row NEFFs, which never faulted). Faster but not proven race-free: the 2-row set
(`/mnt/scratch/v3gr2`, `MIN_DECODE_R=2`) at 18.0 tok/s. Debug-only alternative: `NEURON_RT_DBG_FORCE_STRICT_ORDERING=1` (8.2 tok/s).
Tokens: with the workaround the output is deterministic across starts; it differs from the 1-row path from the first decode step on (a
different NEFF; near-tie at token 2), which is expected and the 1-row baseline itself is not reproducible.

### Verification of the workaround

`BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R=4`, engine /mnt/scratch/v3fbuild2, graphs /mnt/scratch/v3g1k, 1 stream, greedy, 300 tokens per fresh
server start: **10 / 10 starts without a fault** (run vr4; plus 8 / 8 earlier, minr4: 18 / 18 in total, baseline 7 / 8 faulted), and **all
18 outputs token-identical** (reasoning + answer). Decode 15.1 tok/s on a quiet box (10.5-14.5 in the starts that overlapped graph
compiles). For the record: x-dump diff and NANSCAN tools stay in the engine as opt-in debug switches.

Open: a fast race-free 1-row path. Options not yet tried: report the gdn_1_1_32 NEFF to the Neuron compiler team; compiler flags beyond
-O1 / -O3 (O3 and the GDN_GATHER=flat / GDN_EINSUM=elem NEFFs are compiled in /mnt/scratch/v3galt_* but not screened); double-buffered
(non-aliased) gdn states; the 4/8-row decode NEFFs are unaffected, so batched serving does not need the switch.

## fp8 lm_head + 8-bit KV cache (2026-11-15)

**fp8 lm_head: already in place.** `qwen35_prepare.shard_head` quantises the head with `RP.to_fp8_rows` whenever `--weights fp8`;
/mnt/scratch/v3prep/r{0,1}/head.bin = 254.8 MB/rank (`w` f8e4m3 [124160, 2048] + per-row f32 `w_s`), head NEFF meta `fp8: true`. No change made.

**KV8** = the existing graph switch `KV8=1` (codes + per-(token, K|V, head) f32 scales; engine binds `kv` + `kvs` generically). Graph sets
(gqa rebuilt, gdn/moe/head linked from /mnt/scratch/v3gr2, built by /mnt/scratch/v3kv8.sh OUT PAGES "P list" CODE):
/mnt/scratch/v3gkv8 (int8, PAGES 1025, 109 s), /mnt/scratch/v3gkv8e4 (e4m3, PAGES 1025), /mnt/scratch/v3gkv8_4k (int8, PAGES 4097, P up to
4096, 221 s). ATT_BLOCK not needed (compiles and runs at P = 4096). Default stays bf16 KV (graph dir choice).

Accuracy vs bf16 KV (both fp8 head; `MIN_DECODE_R=4` so decode is deterministic; 3 chat prompts x 64 greedy tokens, logits from
`NEURON_LOGITS_DUMP`, compared step by step until the first argmax change, since after that the inputs differ):

| KV | prompt 0 | prompt 1 | prompt 2 | rel L2 logit error (mean / max over compared steps) |
|---|---|---|---|---|
| int8 | 64/64 identical | 34/64, then near-tie (bf16 margin 0.157) | 47/64, near-tie (0.069) | 0.068/0.160, 0.048/0.110, 0.068/0.119 |
| e4m3 | 35/64 (margin 0.188) | 23/64 (0.271) | 9/64 (0.015) | 0.060/0.159, 0.054/0.108, 0.073/0.137 |

int8 holds the greedy path clearly longer; the rel-L2 numbers are similar for both and include the step-0 prefill logits; no f32 oracle run
(no time), so treat them as relative, not absolute. Recommendation: int8 codes.

Speed (engine v3fbuild2 + `MIN_DECODE_R=4`, i.e. the fault-safe config; per-request decode tok/s from the server metrics):

| | bf16 KV | int8 KV |
|---|---|---|
| 1 stream, short | 15.2 tok/s (66 ms) | 14.9 (67 ms) |
| 4 streams, short | 15.1 each (66 ms/step, ~60 agg) | 14.5-14.8 each (~68 ms) |
| 8 streams, short | 9.7 each (103 ms/step, 78 agg) | 9.65 each (104 ms) |
| 1 stream after an 8.7k prompt | 11.6 tok/s, prefill 455 tok/s | 13.3 tok/s, prefill 469 tok/s |
| state per core (1025 pages) | 0.432 GiB | 0.355 GiB |

KV8 costs ~2% at short context and is faster at 8k (half the key bytes gathered). Fault rate with KV8 on the default 1-row path:
1 fault in 4 fresh starts (and the 1-row speed pass faulted at token 5): KV8 does not change the gdn race.

**64k context with KV8, PAGES 4097** (/mnt/scratch/v3gkv8_4k, 54 NEFFs): loads and runs (the bf16 4097-page set failed at first execute).
Per core: 12.9955 GiB weights + 0.593 GiB state (engine); neuron-monitor `memory_used` per NeuronCore: tensors 14.606 GB, shared scratchpad
0.872 GB, model code 0.366 GB, runtime 9.8 MB, constants 6.9 MB (~15.86 GB of 17.18 GB). 32391-token prompt: prefill 428 tok/s, decode
19.0 tok/s; 64827-token prompt: prefill 375 tok/s, decode 17.3 tok/s. **New max context: 64k tokens (4096 pages x 16)**, was 16k.
(These two long requests ran on the 1-row path without the fault switch and happened not to fault; for serving combine with `MIN_DECODE_R=4`.)
