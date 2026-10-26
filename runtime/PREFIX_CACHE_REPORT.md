# Prefix cache in host memory (behemoth_server)

Date: 2026-10-27. Machine: 192.168.0.229, 2x Radeon AI PRO R9700 (gfx1201), stock clocks (`auto` DPM,
300 W cap), ROCm 7.2, no P2P. Model: Behemoth-R1-123B-v2-QTIP-2bit (88 layers, TP=2). Server flags as in
production (`--transport zerocopy --code-layout lane --memory-fraction 0.9 --token-budget 2048
--prefill-chunk 2048 --mixed-prefill-tokens 512 --prefill-interval 24 --threads 4 --profile false`) plus the
prefix-cache flags named per run. Raw results: `/home/charles/prefix_cache_bench/` on .229 (`<tag>_<mode>.jsonl`,
`srv_*.log`, `e2e_*.log`, `model_test_prefix*.log`); client: `runtime/perf/prefix_cache_bench.py`.

## Summary

| turn-2 TTFT after a turn 1 of | cache off | host restore (`--prefix-cache-gpu false`) | GPU tier (default) |
|---|---:|---:|---:|
| 2k tokens | 5.31 s | 0.39 s | 0.24-0.38 s |
| 8k | 24.4 s | 0.43 s | 0.31-0.45 s |
| 16k | 54.5 s | 0.51 s | 0.53 s |
| 32k | 133.0 s | 0.86 s | 0.57 s |
| 8k shared system prompt, 2nd conversation | 24.4 s | 0.22 s | 0.16 s |

Turn 2 = turn-1 prompt + 32-token reply + a short new user message (48 new tokens). With the cache, turn 2
reuses exactly the turn-1 prompt (see "What matches" for why not the reply) and prefills the remaining
48 tokens.

| per GPU (rank 0 = GPU 0, ~13 GB/s link / rank 1 = GPU 1, ~27 GB/s link) | 2k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|
| bytes per GPU | 0.19 GB | 0.74 GB | 1.49 GB | 2.98 GB |
| restore, GPU time (slower rank) | 15 ms | 58 ms | 114 ms | 233 ms |
| restore GB/s rank 0 / rank 1 | 12.3 / 18.6 | 12.9 / 24.3 | 13.1 / 25.1 | 12.8 / 24.1 |
| save, GPU time | 14.6 ms | 56 ms | 109 ms | 224 ms |
| save GB/s rank 0 / rank 1 | 13.0 / 25.0 | 13.4 / 26.4 | 13.7 / 27.1 | 13.3 / 25.9 |
| host RAM per slot (both GPUs) | 0.37 GB | 1.49 GB | 2.98 GB | 5.95 GB |

The copies run at the links' copy-engine rates measured in PREFILL_SCHED_REPORT.md (13.8 / 28.0 GB/s D2H).

- **Save overhead on concurrent decode:** none measurable. 4 streams decoding at 77.0 / 77.9 / 78.5 ms
  ITL p50 / p95 / max; while a 16k-token request finished and its 1.49 GB per GPU were saved (111 ms GPU
  time): 77.4 / 78.3 / 78.3 ms (cache off, same moment: 77.1 / 77.9 / 78.0 ms).
- **Correctness:** restored prefix + tail prefill gives logits **bit-identical** to a cold prefill with the
  same step split; save -> restore into permuted pages -> save reproduces every host byte on both ranks. A
  restored prefix + short tail vs one long cold prefill differs like any change of step split
  (rel_rms 0.042 vs 0.040 for cold-split vs cold-single, equal argmax); repeated greedy turns gave identical
  text.
- **Defaults:** 10 slots, 64 GB cap (pinned at startup, 29 s), min 64 tokens, GPU tier on, 2 x 47.6 MB device
  staging per GPU (KV pool 8053 -> 7988 pages, -0.8 %).
- **Incidents (dmesg `MES(1) failed to respond to msg=INVALIDATE_TLBS` on GPU 0, count 0 -> 3, GPUs stayed
  up, no hang/NaN, all later runs fine):** (1) 03:44:34, while an earlier build grew the pinned arena
  (hipHostMalloc) during serving - fixed: the arena is now allocated in full at startup, and no failure
  occurred in any of the ~1.5 h of serving benchmarks afterwards; (2)+(3) both at 04:47:44, during the
  regression suite right after `ctest` finished (gpu_correctness ran 04:42-04:47; the next ones were the
  pre-existing two-GPU transport and QTIP kernel tests, which do not execute prefix-cache code); re-running
  those tests and the prefix test four times did not reproduce it. See "Host memory".

## Design

### What matches

A request matches on the exact token ids the engine prefills (after chat templating and tokenization):
the longest common prefix over all slots, capped at prompt length - 1 (the last prompt token is always
computed: its logits are sampled). Matching is token-granular; restores copy whole 16-token pages and a match
ending inside a page restores that page too: the tail prefill starts at the match position and overwrites
the stale entries of that page before anything reads them (attention reads positions < length, and the
append runs before attention within a step). Matches shorter than `--prefix-cache-min-tokens` are misses.

**The previous reply is usually not reused.** The cache stores prompt + *generated* token ids, but the next
turn's prompt contains the client's copy of the reply re-rendered by the template as `" " + text + "</s>"`
and re-tokenized. With this model replies start with `<think>` emitted directly after `[/INST]`, while
the template's `" " + text` puts a space in front, which SentencePiece tokenizes differently
(`▁<`, `think`, `>` / a separate `▁`); the sequences therefore diverge at the first reply token (measured:
0 of 48 reply tokens reused, every time; clients that strip the thinking block diverge there anyway). Turn N therefore reuses everything up to the end of turn N-1's *prompt* and prefills the last
reply + the new message; with long conversations that is still nearly all of the context. Substituting
the generated ids for a re-rendered reply would change what the engine prefills compared with a cold
request, so it was not done (exact-token semantics as specified).

### Slots and eviction (`PrefixCache`, src/prefix_cache.cpp, no GPU code, unit-tested)

- A slot = token sequence + one host page index per 16-token page (a host page index names the page of
  *both* ranks) + optionally its GPU pages (GPU tier) + readiness.
- **Insert** (every request that ends with KV for >= min tokens - finished, stop string, length, or cancelled by
  the client, also during prefill: `history[0:cached]`, i.e. all tokens with KV):
  - nothing is saved when an indexed slot already covers the sequence;
  - indexed slots that are a prefix of it are replaced, and so is the slot it was restored from when that
    match covered the old slot's whole prompt (= the next turn of the same conversation, whose old reply
    did not match). So a conversation occupies one slot; a match ending inside the old prompt (shared system
    prompt, edited earlier message, regenerate) keeps the old slot;
  - pages that lie entirely inside the restored prefix are *shared* with the old slot (reference counted,
    no copy, no extra memory); only the partial page and new pages are copied from the GPUs;
  - LRU eviction to stay within `--prefix-cache-slots` and the host page cap. Plain LRU: prefill time and
    KV bytes both grow ~linearly with length, so cost per byte is about constant and GreedyDual-size
    degenerates to LRU; the slot count bounds bookkeeping only. One-off requests shorter than
    `--prefix-cache-min-tokens` are not saved, so they do not churn slots.
- **References instead of pins:** host pages are reference counted by slots, by *leases* (a matched prefix
  held from submit until the request ends) and by in-flight saves, so evicting a slot never frees pages a
  copy is reading or writing; they are freed when the last user lets go. Pages still referenced count
  against the cap.
- Host pages are handed out lowest index first, so a sequence's pages are mostly consecutive in the arena
  (one copy per run of consecutive pages).

### Request flow (src/engine.cpp, src/scheduler.cpp)

1. `Engine::submit`: lookup. On a hit the request is marked *restoring* (never scheduled or preempted) and
   holds a lease.
2. Engine loop, between steps: once the slot is ready (its save may still be in flight when the next turn
   arrives quickly) the request either takes over the slot's resident GPU pages (**GPU-tier hit**: no copy;
   the slot keeps only its host copy) or gets fresh pool pages and a **host restore** is issued. If the pool
   cannot hold the whole prefix, the part that fits is restored (whole pages). The copy runs while other
   requests keep stepping (all at once, or at most `--prefix-cache-restore-pace` pages per step while
   requests decode); completion is polled at step boundaries, then `cached = matched tokens` and the
   request prefills only the rest (a full-prompt match is a one-token, decode-shaped step).
3. When a request ends, the scheduler keeps its pages (`set_retain_finished`), the engine offers its tokens
   to the cache and issues the save of the pages to copy; the GPU pages stay out of the pool until the copy
   completes, then become the slot's GPU tier (or go back to the pool). A client cancel that arrives while a
   restore copies is applied when the copy lands.
4. The scheduler's reclaim hook frees GPU-tier pages (LRU) before it would preempt anything; GPU-tier pages
   are therefore free memory in practice, but not counted in `kv_pages_free` (`gpu_tier_pages` in metrics).
5. HIP copies are issued by the engine thread outside the engine mutex; all cache bookkeeping is under it.

### GPU copies (src/kv_transfer.hip, TensorParallelModel::kv_save / kv_restore in src/model.cpp)

Per layer and rank a KV page is 8 KB K + 8 KB V (4 heads x 16 tokens x 128 FP8 E4M3) + 2 x 256 B FP32
scales = 16,896 B; 88 layers = 1,486,848 B per page and rank (92,928 B per token, as the pool). A
pack kernel gathers up to 32 pages x 88 layers (grid pages x layers, 16-byte words) into a device staging
buffer, the copy engine moves it to the pinned host pages (one `hipMemcpyAsync` per run of consecutive host
pages); restores do the reverse with an unpack kernel. Two staging buffers per rank ping-pong between a
kernel stream and a copy stream, chained by events, so packing overlaps copying. Transfers execute in issue
order on both ranks and concurrently with the model's streams (different pages). Host layout is page-major
(page i of a slot is one contiguous 1.49 MB block per rank), which makes prefix sharing and partial restores
plain page lists.

### Host memory

- A pinned arena per rank: `hipHostMalloc(portable)` in 64-page (95 MB) blocks, bound to the GPU's NUMA node
  (`set_mempolicy(MPOL_BIND)` + `hipHostMallocNumaUser`; checked in `/proc/<pid>/numa_maps`: rank 0 blocks on
  node 0, rank 1 on node 1). Only copy-engine transfers between device staging and this pinned memory; no
  pageable D2H.
- **Reserved in full at startup** (cap 64 GB: 29 s, both ranks in parallel), before serving. An earlier build
  grew the arena on demand from a background thread while the GPUs ran; one `MES(1) failed to respond to
  msg=INVALIDATE_TLBS` on GPU 0 coincided with that growth (pinned bytes went 15.0 -> 15.4 GB at that moment)
  - mapping new pinned memory into the GPU address space needs TLB invalidations, the same mechanism as the
  pageable-copy hang noted before. Growing on demand also stalled the engine (~50 ms per block and rank). The
  transfer events are created at startup too, so serving does no allocations.
- The cap counts both ranks' pages (`--prefix-cache-gb 64` = 21,522 pages = ~344k tokens, i.e. 10 x 32k slots
  fit); 64 GB stays pinned for the server's lifetime. Lower it with `--prefix-cache-gb` on a busy host.

### Flags

| flag | default | meaning |
|---|---|---|
| `--prefix-cache-slots N` | 10 | cached sequences; 0 disables the cache (no staging, no arena, old behaviour) |
| `--prefix-cache-gb X` | 64 | host memory cap in 1e9 bytes for both GPUs' pages, pinned at startup |
| `--prefix-cache-min-tokens N` | 64 | shortest sequence saved / prefix restored |
| `--prefix-cache-gpu true\|false` | true | GPU tier: finished sequences keep their pool pages until needed |
| `--prefix-cache-staging-pages N` | 32 | pages per device staging buffer (1..64), two buffers per GPU |
| `--prefix-cache-restore-pace N` | 0 | while requests decode, issue at most N restore pages per step (0: all at once) |
| `BEHEMOTH_KV_NUMA=0` | on | do not bind the arena to the GPUs' NUMA nodes |
| `BEHEMOTH_PREFIX_DEBUG=1` | off | log every retire / save / restore decision to stderr |

`/metrics` -> `prefix_cache`: `hits`, `misses`, `gpu_tier_hits`, `host_restores`, `partial_restores`,
`tokens_reused`, `restore_ms_last/p50` (GPU time, slower rank), `restore_wall_ms_*` (issue -> seen by the
engine), `restore_gbps_rank{0,1}_{last,p50}`, the same for saves, `saves_of_cancelled`, `shared_pages`,
`slots_used/slots_max`, `cached_tokens`, `host_bytes_used/cap/pinned`, `gpu_tier_pages`, `evictions`,
`saves_skipped`, `issue_ms_total/max` (engine-thread enqueue time), `transfers_in_flight`, `saves_pending`,
`kv_pages_unaccounted` (pool pages not free, not owned by a request, a save or the GPU tier: always 0).
Responses carry `usage.prompt_tokens_details.cached_tokens`; streaming responses send a final usage chunk
when the request has `"stream_options": {"include_usage": true}`.

## Tests

- `behemoth_tests` (ctest `core_correctness`): new `prefix_cache` (LCP matching incl. partial pages and the
  prompt-1 cap, min tokens, extension with page sharing, superseding, covered sequences, shared system prompt
  tie-breaking, slot-count LRU eviction, host-cap eviction with leased pages, oversized sequences skipped
  without evicting, GPU tier take/reclaim, abandon, next-turn replacement vs shared-prompt retention) and
  `scheduler_restore` (waiting requests not scheduled, restore/finish/abort, full-prompt match as a decode
  step, retained pages, reclaim before preemption). PASS (83,660 assertions).
- `behemoth_model_tests MODEL prefix` (GPU, full model): a 1700-token random prompt (107 pages, 4 transfer
  chunks, partial last page) is saved, restored into permuted pages (different physical ids), saved again:
  every host byte equal on both ranks; restored prefix + 60-token tail logits **bit-identical** to a cold
  prefill split 1700 + 60; a restore of 1690 tokens (ends inside a page, tail recomputes the rest) vs a cold
  1690 + 70 split: rel_rms 0.029-0.046, same greedy token (not bit-equal: the KV came from a 1700-row step);
  restored 1700 + 60 vs one cold 1760-row step: rel_rms 0.036-0.056, same greedy token, exactly the
  cold-split vs cold-single difference of the same process (the dense GEMM autotune picks per process, so
  this varies 0.04-0.06 between runs). Copy rates in the test: save 10-12 / 17-23 GB/s, restore 13.1-13.4 /
  25.1-26.0 GB/s (rank 0 / rank 1, 0.16 GB per GPU). PASS (5 runs; one early version that required equal
  argmax across different step splits failed once on these near-flat random-token logits, the check is now
  top-5 containment plus rel_rms < 0.1).
- `tests/prefix_cache_e2e.py` against the server (default config, `--prefix-cache-gpu false`, and
  `--prefix-cache-gpu false --prefix-cache-restore-pace 128`): turn-2 reuse
  and TTFT (4081-token prompt: 10.4-10.6 s -> 0.34-0.60 s), the next turn replaces the conversation's slot,
  repeated turn hits prompt-1 tokens with identical greedy text, a client cancel after 12 streamed chunks is
  saved and reused, a request cancelled 8 s into a 25k-token prefill caches its 4096 prefilled tokens,
  shared system prompt (2027 of 2034 tokens), 4 concurrent 3-turn conversations (all later turns hit), no
  leaked pages, caps respected. PASSED in all configurations.
- Regression suites on the final build: `ctest` (core, gpu, two-gpu transport, qtip: 4/4 pass),
  `behemoth_gpu_tests --two-gpu` pass, `behemoth_qtip_tests` layer 40 + reference vectors pass (251 PASS
  lines, as before), `behemoth_block_tests` QTIP layers 0 / 87 pass, AQLM layer 0 host transport pass,
  `behemoth_model_tests` QTIP `prefill` mode pass, AQLM model test pass, tokenizer test pass. Pre-existing
  failures unchanged (identical output to PREFILL_SCHED_REPORT.md): AQLM block test with `zerocopy`
  (`max_abs=0.03 pass=0`) and the QTIP default model test ("Tuning snapshot omitted projections"). The
  Python fine-tuner tests were not rerun: nothing they build or import changed (`qtip.hip`, `LinearView`
  and the qtip_training tree are untouched).

## Benchmarks

`prefix_cache_bench.py`, greedy unless noted, each conversation prefixed with a unique session line.

### Turn-2 TTFT (`turn2`)

Turn 1: the 2k/8k/16k/32k story prompt, 32 tokens; turn 2: + reply + "Summarize the story so far in one
sentence." (48 tokens more).

| turn-1 context | turn-1 TTFT (cold) | turn-2 prompt / reused | turn-2 TTFT off | host restore | GPU tier |
|---|---:|---:|---:|---:|---:|
| 2k | 4.8-6.1 s | 2063 / 2015 | 5.31 s | 0.39 s | 0.24-0.38 s |
| 8k | 23.7-24.6 s | 8064 / 8016 | 24.4 s | 0.43 s | 0.31-0.45 s |
| 16k | 53.1-55.0 s | 16062 / 16014 | 54.5 s | 0.51 s | 0.53 s |
| 32k | 129.3-131.2 s | 32064 / 32016 | 133.0 s | 0.86 s | 0.57 s |

Speed-up of turn 2: 14-22x at 2k, 54-79x at 8k, ~105x at 16k, 155-235x at 32k. The cache-on TTFT is the
restore (host: 15-233 ms GPU time, table above; GPU tier: none) plus prefilling the 48 remaining tokens at
2-32k context in a WMMA step (0.2-0.5 s); 2k and 8k ran twice on the GPU-tier server (spread shown). The
first-round 16k GPU-tier number of an earlier build (1.43 s) was inflated by on-demand arena growth, which
the final build no longer does. Decode after the restore runs at the cold-path rate (ITL p50 47 / 51 / 53 /
56 ms at 2k / 8k / 16k / 32k, identical with and without the cache).

### Shared system prompt (`shared`)

8k-token system prompt, two conversations with different questions: second conversation reuses 8020 of
8028 tokens; TTFT 0.16 s (GPU tier) / 0.22 s (host) vs 24.4 s without the cache.

### Save overhead (`save`)

4 streams decode (temperature 0.8) while a 16k prompt prefills and finishes (1 output token); ITL of the 4
streams in the window starting when the long request ends (its save: 111 ms GPU time, 13.4 / 25.4 GB/s):

| | ITL p50 / p95 / max before | in the save window | after |
|---|---:|---:|---:|
| cache on | 77.0 / 77.9 / 78.5 ms | 77.4 / 78.3 / 78.3 ms (28 gaps) | 77.3 / - / 79.7 ms |
| cache off | 74.8 / 75.8 / 78.1 ms | 77.1 / 77.9 / 78.0 ms (26 gaps) | 77.1 / - / 78.6 ms |

The D2H runs on the copy engine and the pack kernel is a device-local copy; the decode steps' zero-copy
exchanges were not slowed measurably. Enqueueing a transfer (between steps, on the engine thread) took at
most 6.2 ms (a 2048-page restore; host pages are mostly consecutive, one copy per run). Saves during
*prefill* of another request share GPU 0's link with the prefill all-gathers (copy engines) and delay that
prefill by roughly the save time (not measured separately).

### Restore while other streams decode (`restore`, `--prefix-cache-gpu false`)

4 streams decode (77 ms ITL); then turn 2 of a conversation with a 32k (16k) turn 1 is sent and restored
from host memory:

| | turn-2 TTFT | restore GPU / wall | decode ITL in the restore window p50 / max |
|---|---:|---:|---:|
| 32k, whole restore at once (default) | 0.80 s | 233 / 279 ms | 280 / 365 ms (12 gaps) |
| 16k, whole restore at once | 0.54 s | 117 / 179 ms | 180 / 262 ms |
| 32k, `--prefix-cache-restore-pace 128` | 1.94 s | 232 / 1422 ms | 89 / 363 ms (74 gaps) |

Unlike saves, restores do slow concurrent decode: the host-to-GPU DMA saturates GPU 0's inbound link, which
the decode steps' zero-copy exchanges also use to read the peer's partial results from host memory, so
the 2-3 decode steps that overlap a 32k restore take ~280 ms instead of 77 (the max includes the step that
prefills the restored request's 32-token tail at 32k context). Saves (D2H) compete only with the exchanges'
posted writes and were not measurable. `--prefix-cache-restore-pace N` issues at most N pages per step
while requests decode: 128 pages (190 MB per GPU per step) cut the decode impact to ~89 ms steps at the
price of +1.1 s TTFT for the restored request. Default 0 (whole restore at once): the total extra decode
delay (~0.6 s for a 32k restore) is far below what the cold path causes (a 32k prefill under
`--prefill-interval 24` pauses the decoders ~2.5-2.9 s every 24 tokens for ~2 minutes).


### 10 slots and the memory cap (`fill`, `--prefix-cache-gpu false --prefix-cache-gb 12`)

- 12 new 2k-token conversations after the turn-2 runs: `slots_used` rises to 10 and stays there, one LRU
  eviction per new conversation; turn 2 of the first conversation is then a miss (cold, 5.0 s), turn 2 of the
  last one a hit (2017 tokens, 0.14 s).
- With the 12 GB cap (12.18 GB pinned, 4035 pages), the 32k turn-1 save (5.95 GB) evicted older slots;
  `host_bytes_used` peaked at 11.9 GB and never exceeded the cap; slots dropped to 4-5 while large ones were
  cached.
- RAM per slot = 185,856 B per token (both GPUs), page-granular: 0.37 / 1.49 / 2.98 / 5.95 GB for 2k / 8k /
  16k / 32k. At the 64 GB default, 10 x 32k-token slots (59.5 GB) fit.

## Limits and notes

- Previous replies are rarely reused (template re-tokenization, see "What matches"); the prompt part is.
- The GPU tier holds finished sequences' pages (reclaimed before any preemption); `kv_pages_free` is lower
  while it is populated. Disable with `--prefix-cache-gpu false`.
- Restore completion is polled at step boundaries: a restore finishing during a long prefill step of another
  request is used at the next step (the request could not join earlier anyway).
- A preempted request that had restored its prefix recomputes it from scratch (no re-lookup) and is then not
  page-shared with its slot.
- 64 GB (default cap) is pinned for the server lifetime and costs 29 s at startup.
- Cache contents are lost on restart; cached KV is the runtime's FP8 KV, so it is only valid for the same
  model and binary numerics (it is never persisted).
- AQLM models use the same KV layout and code path but the GPU prefix test was run on the QTIP model only.
