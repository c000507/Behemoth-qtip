# "Nonfinite model logits" on the Ornith GM tune (trn1, TP = 2 fp8): mechanism and fix

Box i-0ffdea4dcf9b4c089 (trn1.2xlarge), aws-neuronx-runtime-lib 2.33.10, neuronx-cc 2.26.6360.0, model
`cgg507/Ornith-1.5-35B-A3B-REAP20-GM-RP-v3-FP8` (/mnt/scratch/v3prep), production graph set /mnt/scratch/v3gkv8_4k
(= `graphs/ornith-reap20-v3-kv8`). Branch `reap-nonfinite` (from reap-prefix-cache). Earlier study: docs/neuron-reap-perf.md
("Fault" section). Everything below was measured in one session (2026-11-15, 07:00-09:45 UTC).

## TL;DR

* **Mechanism**: the race sits in the **runtime-indexed (dynamic) DMA that gathers / scatters the request's state slot** inside
  the gdn NEFFs (`kstate[slot]`, `cstate[slot]`, `index_put((slot,), ...)` on the aliased `[slots, ...]` state tensors).
  Occasionally the launch uses bytes that the *previous NEFF on the core* left on chip instead of the gathered slot (best
  reading: a missing wait between that DMA and its consumer; the exact instruction-level hazard was not decoded from the NEFF). With the same NEFF launched back to back those bytes are that NEFF's own previous values, so the
  race is invisible; in the real model the previous launch is a moe / gqa / head NEFF, and the stale bytes become garbage
  (1e21-1e26) or NaN. Not aliasing, not host ordering, not the moe / gqa dynamic gathers, no runtime switch except strict
  ordering.
* **Fix**: compile the gdn graphs with `GDN_SLOT=bound`: each row's state is its own one-slot input/output tensor (no slot
  input) and the engine binds the slot's slice of the layer state (`nrt_tensor_allocate_slice`, cached per slot). No dynamic
  state DMA is left in the NEFF.
* **Result** (1 stream, fresh starts x 2 requests x 300 tokens, interleaved A/B, same engine):
  **bound 10 / 10 starts clean (20 / 20 requests), 22.1 tok/s, 20 / 20 outputs token-identical**; the production
  1-row NEFF in the same session **faulted in 10 / 10 starts** (at tokens 40-276), 21.3 tok/s. The `MIN_DECODE_R=4`
  workaround that production runs gives 15.1 tok/s: the fix is **+46 % decode at 1 stream** and costs nothing elsewhere
  (4 streams +5 %, 8 streams unchanged with the dynamic 8-row NEFF kept; section 3). MTP k >= 2 is unblocked as well.

## 1. A minimal repro: `neuron/cpp/neff_race.cpp`

The server needs ~2 minutes per fresh start, and a fault is a rare event; so a NEFF-level harness came first. It loads one
NEFF on both ranks like the engine (states aliased in place), and repeats: restore one state slot from a saved copy, execute,
optionally execute a *peer* NEFF, read the residual output and the slot back, compare bit-for-bit with iteration 0. Identical
inputs, so every difference is a race. `RACE_PEER_NAN=1` fills the peer's inputs with NaN (0xff), so that a stale read turns into
NaN instead of a plausible number.

```
NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 RACE_PEER_NAN=1 \
  neff_race v3gkv8_4k/gdn_1_1_32/graph.neff 20000 2 v3gkv8_4k/moe_1_1_32/graph.neff
```

Iterations with a difference, per rank (r0 / r1), out of N (both ranks run every iteration; x is all-reduced, so a NaN on one
rank shows on both):

| main NEFF | peer | N | diff iters r0 / r1 (nonfinite) | ms / iter |
|---|---|---|---|---|
| gdn_1_1_32 | none (same NEFF back to back) | 3000 | 0 / 0 | 1.08 |
| gdn_1_1_32 | moe_1_1_32 (pattern inputs) | 3000 | 1 / 1 (values off by a few 1e-5: a read of wrong bytes in a pure copy, the conv tap 1 <- tap 2 shift) | 1.73 |
| gdn_1_1_32 | moe_1_1_32 NaN | 20000 | **8 / 7 (4)** | 1.67 |
| gdn_1_1_32, same, repeated | moe_1_1_32 NaN | 20000 | 8 / 7 (4); 9 / 10 (5) with a rebuilt identical-source NEFF | 1.67 |
| gdn_1_1_32 + `NEURON_RT_DBG_FORCE_STRICT_ORDERING=1` | moe NaN | 10000 | **0 / 0** | 3.19 |
| gdn_1_1_32, 2 ms sleep after the peer | moe NaN | 10000 | 1 / 0 | 3.74 |
| gdn_1_1_32 | gdn bound (itself free of dynamic state DMA), NaN | 20000 | 16 / 11 (7) | 1.69 |
| gdn_1_1_32 | gqa_1_1_32 NaN | 20000 | 8 / 7 (4) | 1.77 |
| gdn_1_1_32 | head_1_1 NaN | 20000 | 6 / 9 (3) | 3.26 |
| **gdn_1_1_32 `GDN_SLOT=bound`** | moe NaN | 2 x 20000 | **0 / 0, 0 / 0** | 1.62 |
| moe_1_1_32 (expert gather, dynamic) | gqa NaN | 20000 | 0 / 0 | 1.00 |
| gqa_1_1_32 (KV page gather / scatter, dynamic) | moe NaN | 20000 | 0 / 0 | 1.03 |
| gdn_4_1_32 (4 rows, dynamic) | moe_4_1_32 NaN | 4000 + 20000 | 2 / 2, then 0 / 0 | 4.06 |
| gdn_4_1_32 `GDN_SLOT=bound` | moe_4_1_32 NaN | 20000 | 0 / 0 | 4.12 |
| gdn_8_1_32 (dynamic) / bound | moe_8_1_32 NaN | 10000 each | 0 / 0 and 0 / 0 | 7.80 / 11.22 |

What this separates:

* **Intra-NEFF, not between launches**: a 2 ms pause after the peer does not remove it, host writes / reads between launches
  are present in every iteration, strict ordering removes it.
* **Needs a different NEFF before it**, but *any* different NEFF: also a peer that has no dynamic DMA at all (the bound gdn), so
  it is not stale DMA descriptors from another NEFF's dynamic DMA; it is stale on-chip data read too early.
* **The dynamic state gather / scatter is the racing part**: the same graph with the slot bound by the host (only the state
  access changes: no slot input, state `[1, ...]`, no index_put) has 0 differences where the original has 15-27.
* **Not the in-place aliasing**: `RACE_SPLIT=1` (state outputs bound to separate tensors, copied back by the host) 7 / 7 (2), the
  same rate as aliased.
* The moe expert gather and the gqa page gather / scatter (also runtime-indexed DMA) did not race in 20000 iterations each;
  the 4-row gdn NEFF races about an order of magnitude less often than the 1-row one (2 in 24000), which matches serving
  (4-row runs looked deterministic in the earlier 18-start test but are not proven race-free).

Runtime switches, 1-row gdn + moe NaN peer, 20000 iterations each (baseline in the same batch 8 / 7 and 9 / 10):

| switch | diff r0 / r1 (nonfinite) | ms / iter |
|---|---|---|
| `NEURON_RT_DBG_HW_DECODE_TOGGLE=1` | refused: "Hardware decode requested but not supported on Sunda" (trn1 has no HW decode: the strict-ordering log line about it is irrelevant here) | - |
| `NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1` | 9 / 10 (3) | 1.71 |
| `NEURON_RT_DISABLE_PACK_DESCRIPTORS=1` | 5 / 6 (4) | 1.67 |
| `NEURON_RT_DBG_EARLY_COMPLETION=0` | 10 / 8 (3) | 1.68 |
| `NEURON_RT_DBG_EARLY_POSTING=0` | 5 / 3 (0) | 1.67 |
| `NEURON_RT_DBG_DISABLE_DGE=1` | 4 / 7 (3) | 1.67 |
| `RACE_SPLIT=1` (non-aliased state outputs) | 7 / 7 (2) | 1.86 |
| `NEURON_RT_DBG_FORCE_STRICT_ORDERING=1` | 0 / 0 (10000 iterations) | 3.19 (1.9x) |

None of the cheap switches changes the rate (EARLY_POSTING=0 is within noise of the baseline at these counts); strict ordering
is the only runtime fix and it serialises everything (8.2 tok/s in serving).

## 2. The fix: `GDN_SLOT=bound`

**Graph** (`neuron/qwen35_graphs.py`, env `GDN_SLOT=bound`, default off = unchanged graphs):

* R = 1 decode / prefill: inputs `kstate [1, hl, Dk, Dv]`, `cstate [1, W, K-1]` (one slot), no `slot` input; the new state is the
  output (aliased in place, as before), no gather, no index_put.
* R > 1 decode: one state pair per row, `kstate_<r>` / `cstate_<r>`.
* VERIFY buckets (R = 1, Q drafts): the row's state in, the state after position 0 back into it (aliased), the states after
  positions 1..Q-1 as extra (non-aliased) outputs `kstate@<i>` / `cstate@<i>` (meta `extra_outputs`), no `vslot` input.
* meta `gdn_slot: "bound"`. The numerics are the same formulas as before (only the state access differs); the NEFF is a
  different compile, so tokens can differ from the dynamic NEFF at near ties (as any rebuild).

**Engine** (`neuron/cpp/qwen35_model.cpp`): graphs whose meta says `gdn_slot: bound` get, per state input / output, the slice of
the layer's state tensor at the launch's slot (`nrt_tensor_allocate_slice`, cached per (state, slot, rank)); tensor sets are
cached per (graph, layer, parity, rank, slots of the launch). Decode / prefill use `L.slot[row]`, verify uses `L.vslot[i]`.
State allocation, spill / restore, prefix-cache snapshots and NANSCAN are unchanged (the full `[slots, ...]` tensors still
exist; the graph just sees one slot of them). Graph sets without bound NEFFs run exactly as before. The switch is the graph
directory: bound NEFFs in the set = new path; the default stays the old graphs. **A bound NEFF needs the new engine** (an old
engine would size the state tensors from it); deploy the engine before the graphs.

## 3. Serving results

Engine /mnt/scratch/nf/eng (pcbuild2 + this change), `--prefill-chunk 256 --max-batch-requests 8`, no `MIN_DECODE_R`,
`v3fault.sh` (fresh server start, 2 sequential greedy requests x 300 tokens), the two configurations alternated start by start:

| graphs | starts | requests completed / faulted | decode tok/s | outputs identical |
|---|---|---|---|---|
| production v3gkv8_4k (dynamic gdn_1_1_32) | 10 | 2 / **10** (faults at tokens 248, 92, 143, 214, 158, 49, 267, 276, 40, 100) | 21.3 | 2 variants in 2 |
| v3gkv8_4k + bound gdn_1_1_32 (/mnt/scratch/nf/gset) | 10 | **20 / 0** | **22.1** | **20 / 20** |
| for reference (earlier sessions): `MIN_DECODE_R=4` | 18 | 18 / 0 | 15.1 | 18 / 18 |

Probability of 0 faults in 20 requests at the baseline's rate (>= 10 faults in 12 requests) is negligible (< 1e-10 for any
per-request fault rate >= 0.6). The bound NEFF is also 3-4 % faster than the dynamic one (no gather / scatter DMA; harness
1.62 vs 1.67 ms with peer, serving 22.1 vs 21.3 tok/s).

More, engine /mnt/scratch/nf/eng2 (the generalised engine of the commit: R > 1 and verify bound too) and the bound NEFFs
of /mnt/scratch/nf/g3 (`deploy/neuron-reap/nfbuild.sh dec verify`):

| test | graphs | result |
|---|---|---|
| 1 stream, 5 fresh starts x 2 x 300 tokens | gset2 = production + bound gdn_{1,2,4,8}_1_32 | **10 / 10 requests clean**, 22.07-22.18 tok/s |
| 8 concurrent requests (8.6k shared GM system prompt, 200 tokens each), 3 starts per set | gset2 vs production | 0 nonfinite in either (48 requests); outputs differ between starts in BOTH sets at the same characters (13, 157): the known batch-composition effect (docs/neuron-reap-prefix-cache.md, 8-stream baseline), not the race |
| MTP k = 3, **no MIN_DECODE_R**, 3 fresh starts x 5 prompts (s1, s2, t1, h1, a1; 256 tokens) | gmtp2 = v3gmtp + bound decode + bound verify gdn_1_{2,3,4}_32 | **15 / 15 clean**, outputs **identical across the 3 starts**; 22.4-28.4 tok/s (k = 3 first pass with MIN_DECODE_R=4: 21.3-28.0, and 1 fault in 8 requests) |

Harness for the bound R > 1 graphs: gdn_4_1_32 bound 0 / 0 in 20000, gdn_8_1_32 bound 0 / 0 in 10000 (the dynamic gdn_8_1_32
also 0 / 0 in 10000; the dynamic 4-row 2 in 24000).

**Decode speed** (`v3bench.py dec S 250`, per-request decode tok/s from the server metrics, engine eng2, quiet box, 2 runs each):

| streams | production set (dynamic gdn) | + bound gdn_1_1 only (gset) | + bound gdn_{1,2,4,8} (gset2) |
|---|---|---|---|
| 1 | 21.3 (faults; with MIN_DECODE_R=4: 15.1) | **22.1** | 22.1 |
| 4 | 14.8 (= gset: same NEFFs) | 14.78 (14.25 the last-arriving stream) | **15.55** (bound 4-row: +5 %) |
| 8 | 9.64 (= gset) | 9.64 | **5.1** (bound 8-row: -47 %) |

The bound 8-row NEFF compiles badly (1.14 MB vs ~0.24 MB for the others, about 2x the step time); the bound 1-, 2- and 4-row
NEFFs are as fast or faster than the dynamic ones. So the production set keeps the dynamic gdn_8_1_32 (0 / 0 in the harness,
never seen to fault in serving) and binds the slot in R = 1, 2, 4 and in the verify buckets. (The production set's own
bench start faulted on its warm-up request and stopped the engine: one more baseline fault.)

**Recommended set, with the production prefix-cache env** (gfin = production + bound gdn_{1,2,4}_1_32; `--prefix-cache-gb 4`,
`PREFIX_CACHE=1 PREFIX_ALIGN=1 PREFIX_INTERVAL=2048 PREFIX_STATE_GB=4 PREFIX_HOST_GB=4`, no MIN_DECODE_R), 3 fresh starts, each
`dec` 1 / 2 / 4 / 8 streams x 250 tokens + `pc.py soak 9` (8 of 9 restored from the cache at 256):

| streams | decode tok/s per stream (3 starts) | today's production (MIN_DECODE_R=4) |
|---|---|---|
| 1 | **22.06 / 22.05 / 22.11** | 15.1 |
| 2 | 17.7-17.95 | 15.1 (padded to 4 rows) |
| 4 | 14.95-15.55 | 14.8 |
| 8 | 9.12-9.64 | 9.65 |

0 nonfinite in 3 starts (15 bench + 27 soak requests); soak outputs **9 / 9 identical across all 3 starts** (cached == cold
was already shown for ALIGN; the bound slices see the same memory the snapshot / restore copies use). MTP k = 1 on the same
set + bound verify (gmtp3): 5 prompts, 0 nonfinite, 29.45 / 26.22 / 29.5 / 24.96 / 31.54 tok/s (s1, s2, t1, h1, a1;
first pass with MIN_DECODE_R=4: 29.6 / 23.6 / 29.7 / 25.1 / 31.5), i.e. MTP no longer needs MIN_DECODE_R either.

Totals for the fixed 1-row path this session: 21 fresh starts (10 A/B + 1 smoke + 5 + 2 bench + 3 final), 66 single-stream
requests (300 / 250 / 160 tokens; 27 of them prefix-cache soak), plus 48
8-concurrent, 15 MTP k = 3 and 5 MTP k = 1 requests: **0 nonfinite**. Dynamic 1-row NEFF in the same session: 11 of 11
starts faulted.

## 4. Production patch (not applied)

1. **Engine**: branch `reap-nonfinite` (= reap-prefix-cache + commit 8d8f917; only `neuron/cpp/qwen35_model.cpp` changes
   in the engine). Graph sets without bound NEFFs behave exactly as before, so it can ship ahead of the graphs.
2. **Graphs**: in `graphs/ornith-reap20-v3-kv8`, replace these NEFF directories with the bound builds (staged, see below):
   `gdn_1_1_32`, `gdn_2_1_32`, `gdn_4_1_32` (keep `gdn_8_1_32`, `gdn_1_64_32`, `gdn_1_256_32` and everything else). For MTP
   (`v3gmtp`-style set) also `gdn_1_2_32`, `gdn_1_3_32`, `gdn_1_4_32` (bound verify). Rebuild from source with
   `SRC=<tree of reap-nonfinite> bash deploy/neuron-reap/nfbuild.sh OUT dec verify` (then drop OUT/gdn_8_1_32) on any trn1
   box with the venv, ~1-2 min per NEFF.
3. **Env**: remove `BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R = "4"` (`deploy/ornith-nonfinite.tfvars.patch`). Nothing else.
4. Order: engine first, then graphs (an old engine loading a bound NEFF would size the state tensors for one slot).

Staged (not in any behemoth-* bucket): `s3://912935854432-neuron-build/reap/nf/graphs-bound/<dir>/` with `graph.neff`,
`meta.json` per directory (gdn_1_1_32, gdn_2_1_32, gdn_4_1_32, gdn_1_2_32, gdn_1_3_32, gdn_1_4_32; md5 of graph.neff
947320e717a2..., 576d688f27a4..., 724000542b62..., 7f22f405ddf2..., 1452da00bbb1..., d7b90a997a7b...), and the engine binary
built on the box (pcbuild2 objects + this qwen35_model.cpp) at `s3://912935854432-neuron-build/reap/nf/engine-reap-nonfinite/neuron_server`
for reference (production should build from the branch). The meta's `neff` field points at the box path; the engine ignores it.
Copy the six directories over the same-named ones of a copy of the graph prefix (MTP: all six; without MTP: the first three).

## 5. What was ruled out (this session and the previous one)

* Host side: rank barrier per launch, blocking device read after every launch, a pause between launches (harness).
* Runtime: DGE off, HW execution barrier, descriptor packing off, early completion / early posting off (table above); HW decode
  toggle is not available on trn1.
* In-place aliasing of state input and output (harness `RACE_SPLIT=1`, same rate); NEFF rebuild from the same source (same rate).
* Earlier: fused recurrence form (1 / 9 faults), static conv slice (`GDN_DEC=1`), `-O1` (3 / 3): all keep the dynamic slot access.
* The moe expert gather and gqa KV page gather / scatter: no race in 20000 iterations each (harness).
* Not seen on trn2 (~30 runs) and on the 397B; consistent with a trn1 (neuronx-cc 2.26 / runtime 2.33) synchronisation bug
  around the dynamic DMA of the gdn state gather, i.e. a compiler / runtime issue, but one we no longer depend on.

## 6. Repro for the Neuron team

`neuron/cpp/neff_race.cpp` + `/mnt/scratch/v3gkv8_4k/gdn_1_1_32/graph.neff` (+ `moe_1_1_32/graph.neff` as the peer); no
weights needed (inputs are a pattern, peer inputs NaN): `RACE_PEER_NAN=1 neff_race gdn.neff 20000 2 moe.neff` shows 7-16
differing iterations per rank in 20000 on trn1.2xlarge, runtime 2.33.10, compiler 2.26.6360.0; 0 with
`NEURON_RT_DBG_FORCE_STRICT_ORDERING=1` and 0 with the slot-bound variant of the same graph. The HLO of both graphs is in
the graph directories (`model/graph.hlo`).
