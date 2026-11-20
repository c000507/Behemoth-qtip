# Prefix cache for the Ornith GM tune on the trn1.2xlarge (Qwen3.5 35B-A3B REAP20 v3 FP8, TP = 2, KV8, PAGES 4097)

Model `cgg507/Ornith-1.5-35B-A3B-REAP20-GM-RP-v3-FP8` (30 gated-DeltaNet + 10 full-attention layers, 205 experts), graph set
`/mnt/scratch/v3gkv8_4k` (= production's `graphs/ornith-reap20-v3-kv8`, int8 KV, 4096 pages, 8 state slots), prepared set
`/mnt/scratch/v3prep`, box i-0ffdea4dcf9b4c089. Server args as production (`--prefill-chunk 256 --max-batch-requests 8`) and
`BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R=4` in every equality run (the 1-row gdn NEFF race makes runs without it non-reproducible,
docs/neuron-reap-perf.md). Mechanism: docs/neuron-qwen35-prefix-cache.md (branch qwen397-release2); this doc is what it
takes on this model and box. Branch `reap-prefix-cache` (off `reap-perf-opencode` c21c6c6).

## 1. Summary

* **Production's engine already has the code path** (ornith-prod 908e23c carries `neuron/cpp/`, `runtime/` verbatim from
  reap-perf c21c6c6: `BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE`, state_save / state_restore / kv_copy in qwen35_model.cpp). The
  snapshots are host copies of the gdn state slots (`nrt_tensor_read` / `nrt_tensor_write`): **no extra graphs, nothing new
  to upload for the graph set**.
* It works out of the box on this model: hits are exact for repeats and aligned partial matches, TTFT 20.9 s -> 0.27 s at
  9.5k tokens and 93.9 s -> 0.30 s at 38k.
* Two engine problems found and fixed (both behind the existing switch; default behaviour of every other path unchanged):
  1. **Host RAM**: the K/V host arena was a zero-filled `std::vector`, so `--prefix-cache-gb` was resident at start
     (8 GB cap: RSS 6.3 -> 14.1 GB before the first request). On a 32 GiB host that is the difference between safe and not.
     Now lazily backed (RSS at start 6.32 GB with a 4 GB cap).
  2. **Next-turn hits were not exact**: the default tail snapshot (8 tokens before the prompt's end, with an extra chunk
     split there) restores at an off-grid position, the following prefill chunks then run on another partition, and the
     gdn prefill's arithmetic depends on the partition: a3 / h2 (next turn after an edited / fixed answer) diverged from the
     cold run at characters 55 / 60. The split also changes cache MISSES against the cache-off server (8 of 11 prompts
     differ after 17-135 characters: same family of near ties, but production output would change the day the cache is
     switched on). New opt-in **`BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN=1`**: snapshots only on the 256-token chunk grid, the
     tail one at the last grid point at or before prompt - 8, no split, no end snapshot. **With it every cached and every
     cold response is token-identical to the cache-off server** (section 3).
  3. (Found while testing 2.) With END off, requests that have no snapshot (short prompts with long answers) were still
     cached and evicted the useful slots (12 short benchmark requests emptied all 10 slots). Snapshot-less recurrent
     sequences are no longer cached.
* Recommended production config (section 6): `BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE=1`, `..._PREFIX_ALIGN=1`,
  `..._PREFIX_INTERVAL=2048`, `..._PREFIX_STATE_GB=4`, `BEHEMOTH_NEURON_PREFIX_HOST_GB=4`, server arg `--prefix-cache-gb 4`,
  engine built from this branch (the RSS fix matters on the 32 GiB host; without the new engine use `--prefix-cache-gb 2`).

## 2. Sizes and host memory budget

| item | size |
|---|---|
| recurrent snapshot (30 gdn layers x 2 ranks: kstate f32 [16 heads, 128, 128] = 1 MiB + conv f32 [4096, 3] = 48 KiB per layer and rank) | **62.81 MiB** (engine: "snapshots of 62.8125 MiB") |
| snapshot cost (device -> host, both ranks in parallel) | 30.7 ms (288 snapshots, 8846 ms), restore ~25 ms |
| K/V page (10 gqa layers x 2 ranks x (8 KiB int8 codes + 128 B scales), 16 tokens) | 162.5 KiB = **10.4 KB per token** |
| 9.5k-token conversation, interval 2048: K/V + 5 snapshots | 99 MB + 314 MiB |
| 38k-token conversation, interval 2048: K/V + 19 snapshots | 395 MB + 1.19 GiB (interval 1024: 2.3 GiB) |
| server RSS, cache off (after load / after 38k requests) | 6.30 / 6.75 GB |
| RSS, cache on, OLD engine, `--prefix-cache-gb 8` (start / after the test set / peak) | 14.09 / 18.09 GB (arena resident + 4 GB snapshots) |
| RSS, cache on, new engine, `--prefix-cache-gb 4`, STATE_GB 4 (start / after the test set / end) | 6.32 / 9.81 / 8.11 GB |

Upper bound with the recommended caps: 6.8 GB + 4 GB K/V + 4 GB snapshots + ~0.5 GB transient = ~15.3 GB, inside the ~20
GiB the 2xlarge leaves after the resident weights. **Max safe values on the 32 GiB host: K/V + snapshot caps <= 12 GB
together** (e.g. 4 + 8 or 6 + 6) with the new engine; with the OLD engine (arena resident) `--prefix-cache-gb` counts in
full from the start, so keep it <= 2 there. 4 GB of K/V holds ~385k tokens, i.e. 10 slots of 38k: the slot count (10), not
the bytes, limits K/V; the snapshot cap is the binding one (4 GB = 64 snapshots = three 38k conversations at interval 2048).

## 3. Correctness (greedy, 160 new tokens, `MIN_DECODE_R=4`, request list in `pc.py`)

Prompts: s1 / s2 short GM prompts (453 / 449 tokens, shorter than the interval), a1 / a2 9.5k world-bible system prompt +
different questions (partial hit), a3 = a1 + fixed assistant answer + new user turn (next turn), t1 / t2 tool calls
(`roll_dice`, `lookup_npc`) and the turn after the tool result, h1 / h2 11.5k-token RP chat history (14 turns) + next turn,
b1 / b2 38k system prompt + different questions. All with the default `<think>` opening (GM/RP reasoning). Tool-call ids
are random per response and excluded from the comparison.

| run | config | result |
|---|---|---|
| ref vs off | cache on, default TAIL=8, every request a miss, vs cache off | **3 / 11 identical**: the tail chunk split alone changes the output (near ties, diverging after 17-135 chars) |
| on1 vs ref | default (interval 1024, TAIL 8, END on): repeats (restored prompt - 8), partial at 9216 / 37888, tool turn 2 (9817) | **13 / 15 identical**; a3 and h2 (next turn, restored at the unaligned tail 9478 / 11512) differ at char 55 / 60 |
| on2 vs ref | second pass after evictions (4 GB snapshot cap, 24 slot evictions, 229 snapshots dropped) | 11 / 11 identical (4 hits, 7 misses after eviction) |
| on c4 / c4b vs off c4 | 4 concurrent requests on the shared 9.5k prefix, cold then cached (9478-9482 restored) | 4 / 4 and 4 / 4 identical |
| **align vs off** | ALIGN=1, interval 2048: repeats at 256 / 9472 / 11264 / 37888, partial 8192, next turn 9472 / 11264, tool turn 9728 | **12 / 12 identical, against the CACHE-OFF server** |
| **align2 vs off** | second pass, all hits (256, 9472, 11264, 11520) | **7 / 7 identical** |
| **align c4 vs off c4** | 4 concurrent, restored 8192 each | **4 / 4 identical** |
| align c8 / on c8 vs off c8 | 8 concurrent | 1 / 8 and 0 / 8 identical (see below) |
| eviction | ALIGN, 3 slots, 0.3 GB K/V, 0.5 GB snapshots: 19 requests incl. 3 x 38k (do not fit), 6 slot evictions, 290 snapshots dropped (cap) | **19 / 19 identical to cache off** (8 hits, 11 misses after eviction; 0 failed saves) |
| without MIN_DECODE_R | ALIGN, 1-row decode NEFFs | first (cold, cached 0) request: `Nonfinite model logits` at token 111, engine stopped: the known 1-row race, not the cache. Equality / soak runs therefore all use MIN_DECODE_R=4 |
| **soak** | 10 fresh starts x 9 short requests (s1 / s2, 120 tokens; 8 of 9 restored at 256), ALIGN + MIN_DECODE_R=4 | **0 nonfinite in 10 starts / 90 requests** (previous cache-on rate without the workaround: 11 / 14 starts); **90 / 90 outputs identical across starts**, cached == cold within each start |

8 streams: with 8 decoding rows the step uses the 8-row bucket, with <= 4 the 4-row one, and which tokens run in which
depends on when each stream's prefill ends (a cached request prefills 8-300 tokens, a cold one 9.5k): another realisation,
not a cache error. Baseline: two back-to-back 8-concurrent cold runs on the SAME cache-off server agree on only 3 / 8
(5 differ after 13-350 characters), so 8-stream output is not reproducible with or without the cache.

The prompts shorter than the snapshot interval are covered by the tail snapshot (s1 again: 445 restored by default, 256 =
the grid point with ALIGN; s2 shares s1's first 256 tokens and restores there).

## 4. Performance (same box, quiet except where noted)

TTFT from `request_metrics` (`ttft_ms`, `cached_tokens`; ms):

| prompt | cache off | cold, cache on (default / ALIGN) | identical prompt again (default / ALIGN) | partial (shared system prompt, new question) | next turn |
|---|---|---|---|---|---|
| s1, 453 tokens | 1130 | 1418 / 1196 | 261 (445 cached) / 582 (256) | - | - |
| a1, 9486 | 20308 | 20890 / 21175 | 271 (9478) / 281 (9472) | 1153 (9216 of 9482) / 3276 (8192, interval 2048) | 866 (9478) / 639 (9472) |
| h1, 11520 | 24622 | 25239 / 25575 | 253 (11512) / 599 (11264) | - | 599 (11512) / 941 (11264) |
| t2 (tool result turn), 9901 | 21225 | - | 855 (9817) / 630 (9728) | - | - |
| b1, 38027 | 92332 | 93873 / 92890 | 303 (38019) / 796 (37888) | 1376 (37888 of 38023) / 1068 | - |

Cold prefill with the cache on costs 1.6-2.9 % more (snapshots, 31 ms each; with the default the extra tail chunk too).
ALIGN costs up to one chunk (256 tokens, ~0.55 s) more per hit than the default's split tail, and the partial-match restore
point moves to the interval grid (interval 2048: up to 2048 tokens recomputed = ~4.5 s worst case; interval 1024 halves
that for 2x the snapshot memory).

Decode, `v3bench.py dec N 250` (per-stream tok/s from request_metrics; wall s for 250 tokens):

| streams | cache off | cache on, default | cache on, ALIGN |
|---|---|---|---|
| 1 | 14.88 (17.08 s) | 14.84 (17.09 s) | 14.90 (17.02 s) |
| 4 | 14.83 (17.99 s) | 14.76 (17.95 s) | 14.85 (17.85 s) |
| 8 | 9.66 (28.02 s) | 9.51 (28.20 s) | 9.63 (27.89 s) |

Unchanged within noise. `/metrics` -> `prefix_cache` after the default run: hits 24 / misses 31, device_hits 11, host_hits
13, restored_tokens 276701, evictions 24, failed_saves 0, snapshots taken 288 / dropped 229 (cap) / failed 0.

Observed, not fixed: with 8 concurrent requests on one shared prefix 1-2 of 8 missed (cached 0) although a matching slot
existed (both builds); outputs correct, only the TTFT of that request is cold. 4 concurrent never missed.

## 5. Engine changes (branch reap-prefix-cache)

* `neuron_engine.cpp`: host arena `std::unique_ptr<std::byte[]>` (default-initialised; was a zero-filled vector).
* `BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN` (default 0): snapshots only at multiples of `--prefill-chunk`; tail snapshot at
  the last grid point <= prompt - `PREFIX_TAIL`; no `set_prefill_split`; END default off; `turn_slack` = tail + chunk.
* `Scheduler::set_prefill_align` (`runtime/`): a prefill chunk that starts off the grid ends at the next grid point (a
  chunk shortened by the token budget under concurrency gets back on the grid). Only set by ALIGN.
* Recurrent sequences without any snapshot are not inserted into the cache.
* Host tests (`neuron/cpp/host_test.sh`): prefix_state_test 51 -> 71 checks (ALIGN: same prompt, partial, next turn exact
  and edited, budget-shortened chunks under 6-way concurrency, 12 snapshot-less requests do not evict; arena RSS +0 MB for
  a 1 GB cap, the old engine fails it); neuron_engine_test 181 and request_metrics 19 unchanged and passing.
  `python3 tools/no_cc_check.py`: A=10 B=0 C=0.

## 6. Production (not applied)

Patch: `deploy/ornith-prefix-cache.tfvars.patch` (against BEHEMOTH-REVERSE ornith-prod, `deploy/ecs-neuron/ornith.auto.tfvars`):

```
backend_extra_server_args = "--prefill-chunk 256 --max-batch-requests 8 --prefix-cache-gb 4"
backend_extra_env = {
  STREAM_LOAD                                = "0"
  BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R        = "4"
  BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE        = "1"
  BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN        = "1"
  BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL     = "2048"
  BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB     = "4"
  BEHEMOTH_NEURON_PREFIX_HOST_GB             = "4"
}
```

The image must carry the engine of this branch (the ALIGN switch does not exist in 908e23c; there it would be ignored
and the default TAIL = 8 mode would run, with the arena resident: then use `--prefix-cache-gb 2`, accept that cached
next-turn answers and the miss outputs change at the near-tie level, and drop ALIGN). Graph set and model: unchanged.
Start log must show `Prefix cache: 10 slots, host tier 3.99992 GB (24038 pages of 162 KiB)` and `recurrent state
snapshots of 62.8125 MiB every 2048 prefill tokens, on the 256-token chunk grid, ...`. Rollback: remove the env lines
(the cache is off without `PREFIX_CACHE=1`).
