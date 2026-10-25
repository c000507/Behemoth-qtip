# PERF round 3 report (items 1-4 of BEHEMOTH_PERF_IDEAS.md)

Date: 2026-10-26. Base: `ddd2c03` (PERF_ROUND2 state). Machine: 192.168.0.229,
both GPUs at OD -500 MHz / 210 W (verified before every measurement round;
`fastpoll` + `gpu_poll2.sh` running). Production server flags throughout:
`--transport zerocopy --code-layout lane --memory-fraction 0.9 --token-budget
1024 --prefill-chunk 256 --threads 4 --profile false`, port 1337.

Load: c=1 = `behemoth_load_bench prompt 1 3 128` (one request at a time, 128
output tokens, first/autotune request discarded, `inter_token_p50_ms` from
`/metrics`); c=4 = `behemoth_load_bench prompt 4 8 96` (aggregate tok/s and
step p50). Baselines measured on the fresh `ddd2c03` server at the start of
the round.

## Results per item

| | baseline | item 1 scratch | item 2 sampling | item 3 pre-warm | item 4 fusions (final) |
|---|---|---|---|---|---|
| commit | `ddd2c03` | `8de0fe5` | `86e68e7` | `79121e0` | `c0a329a` |
| KV pages (`/metrics`) | 4569 | 4727 | 4727 | 4647 | 4666 |
| token capacity | 73,104 | 75,632 | 75,632 | 74,352 | 74,656 |
| c=1 ms/token (ITL p50) | 74.3 | ~75 * | ~75 * | 74.5 | 74.7 |
| c=1 tok/s | 13.25 | ~13 * | 13.09 | 13.18 | 13.13 |
| c=4 step p50 (ms) | 113.2 | ~113 * | 112.9 | 113.4 | 113.8 |
| c=4 aggregate tok/s | 15.70 | — | 18.1 | 31.95 | 31.51 |
| first request TTFT after boot | multi-second stall (per-weight autotune) | — | — | 0.78 s | 0.78 s |
| fresh prompt length TTFT (109-tok) | 1.446 s | — | — | 1.13 s | — |
| startup | ~210 s (load only) | — | — | ~290 s (load + 78 s pre-warm) | ~290 s |

\* item 1/2 c=1 runs followed tuning-heavy batches on the same server; treat as
"no regression" rather than deltas. The c=4 aggregate jump (15.7 -> 18.1 ->
31.9 tok/s at identical step p50) is the removal of host-side sampling cost and
of autotune stalls during concurrency ramp-down, not a GPU step-time change.

## Item 1: right-size projection scratch + attention workspace

Landed: attention workspace 256 -> 32 MB (partition-32 decode still fits ~10k
contexts at c=4; the candidate check falls back to the 1024-key partition
beyond that, prefill needs none). Split-K partial reservation unified at
min(token_budget, 64) rows = 117 MB.

The ideas doc's premise of a 1.88 GB recovery did not apply: the production
server runs projection mode `auto`, which previously took the 64-row (117 MB)
reserve; 1.88 GB was only held under explicit `--projection-mode split-k|...`
flags. Net real saving = the 224 MB workspace: KV pages 4569 -> 4727.
A 3.4k-token prefill (14 x 256-row chunks) and all tests pass; c=1/c=4
unchanged.

## Item 2: host sampling path

Landed: `sample_logits` uses `std::partial_sort` for top-k (exact prefix of
the old total order - the comparator is a strict total order, so identity is
mathematical, not statistical), thread-local reused buffers, and a new
`sample_logits(low, high)` split-vocabulary overload; `TensorParallelModel::
forward` returns a span over one persistent pinned buffer filled by a single
async D2H; the engine samples the rank halves in place (no interleave copy, no
per-step vectors).

Isolated sampler cost on the server CPU: 0.11-0.13 ms/token (old path:
~1.5-4 ms sort + allocations). Token identity: new core_tests case compares
the pre-rewrite sampler against the new one (contiguous + split forms) over
2592 seeded draws x {temperature 0/0.7/1, top_k 0/1/2/40/1000/32768, top_p
0.05/0.5/0.9/1.0, tie-heavy/outlier logits, n=513/32768} - all identical.
Exact path (no top-k) keeps the full sort by design.

## Item 3: autotuner pre-warm + packed async metadata upload

Findings that reshaped the item: tuning at decode row counts (<= 16) is
weight-scoped - all 616 weights re-tune, ~15-20 s per row count; prefill-sized
row counts (> 16) are shape-scoped and pay the observed ~1.3 s. So the
"1.3 s TTFT spike" is two different effects, and only the first stalls the
server badly (fresh-server first decode step).

Landed: at startup the model runs synthetic batches at decode rows 1..4 (the
concurrency this box is licensed for by the GPU-safety limits; override with
`BEHEMOTH_PREWARM_ROWS`, disable with `BEHEMOTH_PREWARM=0`) on a temporary
64-page cache, then sizes the final cache - which also means the cache
accounts for the fully grown tuning scratch. GpuMetadata packs
offsets/positions/lengths/page-table/tokens behind one pinned staging buffer
and uploads with a single async copy (was five blocking pageable memcpys per
step).

Cost: 78 s pre-warm at boot (measured; rows 5..8 would add ~60 s more, rows
16/32/64 and the prefill chunk ~20 s each - not worth it). Warming 16/32/64
and the chunk was tried and reverted (244 s). New prompt lengths still pay
one ~1.3 s shape tuning inside their prefill (1.13 s measured for a 109-token
prompt vs 1.446 s before; the rest of the old spike was host-side, now gone).
First request after boot: 0.78 s TTFT.

## Item 4: micro-fusions

Landed (kept):
- `silu_multiply_quantized`: silu(gate)*up rounds to bf16 and fp8-quantises
  straight into the projection workspace (per-row max in registers). Removes
  the mlp bf16 round trip + one launch per layer. Bit-identical.
- `zero_copy_reduce_residual_dev`: one fused kernel sums the rank partials and
  applies the scaled residual (identical rounding to reduce +
  residual_add_scaled); one launch and one full device pass per layer-end
  collective less, 176 launches per decode token removed.

Dropped after measurement (per HARD RULES, reverted and documented):
- Publish-in-merge (`merge_projection` writing partials straight to the pinned
  host buffer): c=4 step p50 113 -> 123 ms. The merge-shaped scalar host
  stores plus the consumer reading its own partial back over PCIe cost more
  than the device round trip they replace. (A/B-isolated with temporary env
  toggles: fusions off 114.2 ms; silu-only 114.5 ms; collective-only 123.3 ms;
  publish-in-merge is the whole regression.)
- Attention-epilogue quantisation for the o_proj input: structurally blocked.
  The fp8 GEMM kernels factor a single input_scale per row out of the dot
  product; the attention epilogue sees 128-wide per-head rows, and per-head
  scales cannot be factored out without one accumulator per head in every
  projection kernel.

Profiled c=1 step (events): collectives_ms 4.0 -> 2.8; elementwise loses the
residual pass; end-to-end c=1/c=4 unchanged within noise (74.7 / 113.8 ms).

## Tests

After every item: `ctest` 3/3, `behemoth_gpu_tests --two-gpu` PASS,
`behemoth_block_tests MODEL 0` PASS with rms=0.00 (bit-identical to the CPU
reference, including both landed fusions). model_tests now runs the zero-copy
transport so the fused collective path is covered in CI.

## Bottom line

- +97 KV pages (+1.5k tokens) net (item 1 minus pre-warm scratch growth);
  the ideas doc's +1,100 pages was based on a mis-read reserve.
- Host-side per-token cost ~0.13 ms (was 1.5-4 ms); no GPU idle gap for
  sampling or batch metadata.
- Fresh-server first request 0.78 s TTFT (was a multi-second stall);
  c=4 8x96 aggregate 15.7 -> 31.5-32.5 tok/s at unchanged step time; new
  prompt lengths pay one ~1.1 s shape tuning.
- Startup +78 s (pre-warm, reported); BEHEMOTH_PREWARM=0 reverts to lazy.
- 176 launches + one device pass per token removed; publish-in-merge and
  attention-epilogue quantisation documented dead with numbers/reasons.
