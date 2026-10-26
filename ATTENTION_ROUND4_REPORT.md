# Attention round 4: GQA-aware decode/prefill kernels, BF16 Q/P option

Commits: `567b4da` (GQA decode + prefill kernels, partition sizing, tests),
`9f78f65` (BF16 Q/P option), this report. Base: `79a608b` (round 3).

**Status: both GPUs were lost (PCI vendor id `ffff` on 33:00.0 and cc:00.0) at
11:36:28 during the end-to-end benchmark of the new build, 64 s into the
20.8k-token prefill request.** Per the safety rules all GPU work stopped
there: no recovery/reboot was attempted, the BF16 Q/P end-to-end and
repetition comparison did not run, nothing was uploaded to HF, and the
production server is NOT serving (process 3473232 still exists but its
devices are gone). Section 5 has the timeline and the poller data. Every
number below was measured before the loss; the kernel work is complete and
tested, the end-to-end picture is partial.

## 1. Problem 1: long-context decode attention

### Before

`attention_kernel`: one 32-thread wave per (request, query head, 16-query
tile, key partition); every query head re-read its kv head (12x), every tile
was a 16-row matrix even for one decode query, ~29 KB LDS per wave gave 2
waves per WGP. Per layer per rank, one decode query (`/tmp/long_attention`,
partition 32 = production choice): 2k 1.1 ms, 13k 6.4-6.8 ms, 32k 15-17 ms,
40k 19-21 ms. Production: 74 ms/token at a 39-token prompt, 348 ms/token at
6.1k context, **1017 ms/token at 20.8k**; TTFT 69.6 s at 6.1k and **614 s at
20.8k** (the same kernel ran prefill: 14.6 ms/layer for a 256-query chunk at
2k context, 42 ms at 6k).

### After: `gqa_decode_kernel` (`src/attention.hip`)

* Used when every request carries exactly one query (`max_queries() == 1`).
* One 128-thread block (4 waves) per (request, kv head, key partition). The
  12 query heads sharing the kv head form the 16-row WMMA tile (rows 12..15
  zero); each 32-key K/V tile is loaded once with 16-byte loads (register
  prefetch one tile ahead), used by the 12 heads via the same FP8
  16x16x32 fragments as before. Thread t owns softmax row t/8 and keys
  (t%8)*4..+4 (8-lane shuffle reductions), then output columns (t%8)*16..+16
  of that row in registers; LDS 21 KB (Q, K, V, P, scores, product).
* Semantics unchanged: FP8 Q/K/V decoding and per-vector scales, scores x
  q_scale x k_scale / sqrt(128), fp32 online softmax, P quantised to FP8 per
  row against the 32-key tile maximum (times v_scale), split-K partials
  (max, denominator, 128 floats) and the same `merge_kernel`; output/stat
  buffers and `attention_workspace_bytes` unchanged.
* Partition sizing: `decode_attention_partition(layout, workspace, requested,
  target_blocks)` picks the smallest multiple of 32 keys such that about
  512 (request, kv head, partition) blocks cover the longest context, grows
  it until the partials fit the 32 MB workspace, and returns 0 (unsplit) for
  single-tile contexts. `BEHEMOTH_ATTN_PARTITION` forces a partition,
  `BEHEMOTH_ATTN_BLOCKS` the target. At 13k/c=1 this gives 128 keys (408
  blocks), at 32k 256, at c=4 with 13k/517/8191/2049 416.
* The old kernel remains behind `BEHEMOTH_ATTN_LEGACY=1` for A/B.

### Also: `gqa_prefill_kernel` (multi-query path)

The 614 s TTFT made the prefill kernel the larger user-visible problem, so
the multi-query path got a GQA kernel too (the task allowed keeping the old
one; it is kept under the legacy switch). 12 waves (one per query head of the
kv group, 384 threads) per (request, kv head, 16-query tile, key partition).
Q stays in A fragments, scores and the output accumulate in WMMA accumulator
fragments (gfx12 wave32 layout lane l -> column l%16, element i -> row
8*(l/16)+i, verified at startup by `layout_probe` in `verify_device`; the
softmax row reductions are 16-lane shuffles), only P (FP8 per row / tile
max, or BF16) goes through a 512 B per-wave LDS tile. K/V tiles are shared
by the 12 waves (prefetched one tile ahead), keys past the tile's last query
position are skipped. LDS 24 KB (Q staging reused), 204 VGPRs, no spills
(a `sched_barrier` per column fragment keeps one PV product in flight).
Supports split-K partials like the other kernels (tests use it with
partition 32 on multi-query batches).

### Kernel results (`/tmp/long_attention`, device 0, server idle, ms per layer per rank)

Decode, one query per request, default FP8, auto partition (`*` = chosen by
`decode_attention_partition`); accuracy vs `reference_attention` (same
partition) is 1.6-1.7e-3 relative RMS for every row, identical to the old
kernel.

| context (tokens)            | before (part 32) | after auto      | after part 32 | after part 1024 |
|-----------------------------|------------------|-----------------|---------------|-----------------|
| 90 (c=1 / c=4)              | ~0.08 (est.)     | 0.020 / 0.025   | same          | same            |
| 2048                        | 1.1              | **0.046-0.058** (32) | 0.046     | 0.090-0.099     |
| 2048 x4 requests            | -                | **0.057-0.061** (64) | 0.160     | 0.090           |
| 8192                        | -                | **0.080-0.086** (64) | 0.18-0.23 | 0.088           |
| 13000                       | 6.4-6.8          | **0.085-0.088** (128)| 0.31-0.32 | 0.096-0.106     |
| 16384                       | -                | **0.099-0.104** (128)| 0.38-0.39 | 0.098           |
| 32768                       | 15-17            | **0.138** (256) | 0.66-0.67     | 0.140           |
| 40000                       | 19-21            | **0.153** (320) | 0.77-0.82     | 0.227           |
| 13000+517+8191+2049 (c=4)   | -                | **0.111-0.113** (416)| 0.47-0.55 | 0.190           |

Targets were 13k <= 0.3 ms and 32k <= 0.7 ms: reached with 3.5x / 5x margin
(13k 74x faster, 32k ~110x, 2k 20x; 13 MB of KV per layer per rank in 87 us
is ~150 GB/s, so there is still 3-4x to bandwidth, mostly the fixed per-block
Q load / partial write and the serial merge). Partition 32 (the old default)
is now the worst choice at long context because of the merge over hundreds
of partials; the auto choice is best or within noise everywhere.

Prefill chunk (256 queries at the end of the context, unsplit):

| context | before (legacy kernel) | after FP8 | after BF16 Q/P |
|---------|------------------------|-----------|----------------|
| 2048    | 14.6                   | **0.645** | 0.816          |
| 6144    | 42.0                   | **1.846** | 2.051          |

22-23x; 1.85 ms at 6k is ~21 TFLOP/s, so prefill attention is no longer the
TTFT bottleneck (projections are).

### End to end (`behemoth_load_bench`, production flags, first request discarded)

ITL = (latency of a 128-token request - latency of a 1-token request) / 127
on the same prompt; the c=4 number is the p50 request latency for 8 requests
x 96 tokens at concurrency 4.

| case                              | before (79a608b)            | after (9f78f65, default FP8)       |
|-----------------------------------|-----------------------------|------------------------------------|
| c=1, 39-token prompt              | 73.6 ms/token               | **68.6 ms/token**                  |
| c=4, 39-token prompt, 96 tokens   | 12.00 s p50 (~123 ms/step)  | **10.27 s p50 (~105 ms/step)**     |
| c=1, 6107-token prompt TTFT       | 69.6 s                      | **25.4 s**                         |
| c=1, 6107-token prompt decode     | 348 ms/token                | **65 ms/token**                    |
| c=1, 20772-token prompt TTFT      | 613.8 s                     | GPUs lost 64 s into the prefill    |
| c=1, 20772-token prompt decode    | 1017 ms/token               | not measured                       |

(The "13k" prompt file tokenised to 20.8k tokens; the 6.1k one is the
closest to the requested ~13k that completed on both builds.) Long-context
decode is now flat: 65 ms/token at 6.1k vs 68.6 at 39 tokens (noise level),
i.e. attention is no longer visible in the decode step. The remaining TTFT at
6.1k (25 s for 24 chunks of 256) is the projection path.

## 2. Problem 2: BF16 Q and P option

`BEHEMOTH_ATTN_QP=bf16` (model.cpp) sets `AttentionDeviceView::q_bf16` to the
rotated BF16 query buffer that `rope_quantize` already writes. All three
kernels are templated on the operand type: with BF16, Q is used unquantised
(q_scale ignored), K and V are widened exactly from their FP8 cache values
into BF16 tiles (16x16x16 fragments), and P goes into the PV product as BF16
with pscale = 1 instead of per-tile FP8. K/V stay FP8 in the cache.
`reference_attention` takes the queries from `AttentionData::q_bf16` when it
is populated; the tests run both kernels in this mode against the reference
with `quantize_probabilities = false`.

Microbench accuracy (relative RMS vs exact double attention on the original
float Q/K/V; `refVsEx` is the scalar reference with the same quantisation,
so the kernel adds nothing measurable on top):

| context      | all-FP8 (default) | BF16 Q/P, FP8 KV | kernel vs its reference (FP8 / BF16) |
|--------------|-------------------|------------------|--------------------------------------|
| 2048 (2 seeds)| 4.80 % / 4.35 %  | 3.65 % / 3.45 %  | 1.7e-3 / 2.2e-3                      |
| 8192         | 3.75 %            | 3.16 %           | 1.6e-3 / 2.4e-3                      |
| 13000 (2)    | 5.13 % / 3.57 %   | 3.87 % / 3.09 %  | 1.7e-3 / 2.3-2.6e-3                  |
| 16384        | 3.32 %            | 2.99 %           | 1.7e-3 / 2.5e-3                      |
| 32768        | 3.13 %            | 2.83 %           | 1.6e-3 / 2.1e-3                      |
| 40000        | 3.33 %            | 2.97 %           | 1.7e-3 / 2.5e-3                      |
| prefill 6144 x 256 q | 4.84 %    | 3.70 %           | 1.7e-3 / 2.3e-3                      |

So FP8 Q + FP8 P account for roughly a quarter of the per-layer attention
error; the FP8 KV cache is the rest (3-4 %). The BF16 kernel's 2.2-2.6e-3 vs
the double reference is the BF16 rounding of P (the reference keeps P in
double).

Cost: decode 13k 0.087 -> 0.14 ms/layer (+60 %), 32k 0.138 -> 0.22 ms,
2k 0.05 -> 0.06 ms (LDS 32 KB, 150 VGPRs: 2 blocks/WGP instead of 3);
prefill chunk +10-25 % (2k 0.645 -> 0.816, 6k 1.85 -> 2.05 ms). At 88 layers
this is ~5 ms/token at 13k and ~2 ms/token at short context, i.e. cheap
relative to a ~70 ms step.

Not done because of the GPU loss: the end-to-end greedy continuation /
repeated n-gram comparison (scripts are ready: `/tmp/gen_compare.py <tag>
<prompt> 512` writes `/tmp/gen_<tag>_<prompt>.json` with distinct-word ratio
and repeated 3/4/8-gram fractions; prompts `/tmp/long8k.txt` (~8k tokens of
these reports) and `/tmp/story8k.txt`), and the block test under
`BEHEMOTH_ATTN_QP=bf16`. The default therefore stays all-FP8; the option is
opt-in.

BF16 K/V: not implemented. It is not a kernel-side switch: the cache pages,
`append_kernel`, `reference_append`, the KV scale arrays, the page-capacity
computation and the host cache tests all assume 1 byte per element plus a
scale, and it doubles KV memory (93 KB/token/GPU -> ~180 KB). The
measurement above says it would remove most of the remaining 3 % error.

## 3. Tests

Before the server restart, on the final code (`9f78f65` contents):

* `ctest --test-dir cpp_runtime/build-rocm`: core_correctness,
  gpu_correctness (149 s), gpu_transport_correctness all passed.
* `behemoth_gpu_tests --two-gpu`: PASS.
* `behemoth_block_tests <model> 0` (server stopped): block 0 rms 0.00 vs
  reference, pass=1.
* `tests/gpu_tests.cpp` additions: single-query contexts 4096 / 13000 /
  33000, mixed decode batches {13000,517,8191,2049} and {33,4097,1,2048},
  each with partitions 0, 32, 1024, the automatic choice at 512 and 2048
  target blocks; decode queries not at the end of the context (masked tail
  keys, partition 256); a 96/8-head mixed decode batch; BF16 Q/P on the
  decode and the prefill kernel; `decode_attention_partition` policy and
  workspace-fit checks. The existing prefill/multi-query, known-answer,
  guard and stream checks run unchanged on the new prefill kernel.

Commit `567b4da` (GQA kernels without the BF16 wiring) was built but, because
it was split out after the GPUs were lost, not run on its own; its FP8
kernels are byte-for-byte the instantiations that the full test run above
exercised, the only differences are the removed BF16 dispatch/fields.

`/tmp/long_attention.cpp` (copy in `perf/attention_round4/long_attention.cpp`)
was extended with the BF16 mode, the automatic partition, prefill-chunk specs
and a single-spec command line (`./long_attention <reps> <len,len,..>
[queries]`, `BEHEMOTH_ATTN_LEGACY=1` for the old kernel).

## 4. Not done / left for the next round

* HF upload: the `codebase` branch head was verified as `3da79b1d...`
  (unchanged), but nothing was uploaded after the crash. Command, once the
  machine is back and the user wants it:
  `rm -rf /tmp/beh_push && mkdir -p /tmp/beh_push/cpp_runtime && git archive HEAD | tar -x -C /tmp/beh_push/cpp_runtime && /home/charles/venvs/nvfp4/bin/hf upload cgg507/Behemoth-R1-123B-v2-aqlm /tmp/beh_push/cpp_runtime cpp_runtime --revision codebase --commit-message "attention round 4: GQA decode/prefill kernels, BF16 Q/P option"`.
* End-to-end numbers at ~13k and 20k context, c=4 at long context, the BF16
  Q/P end-to-end / repetition comparison, and the decision whether BF16 Q/P
  should become the default.
* Decode kernel is at ~150 GB/s of ~600; a fused merge (last block merges)
  and a 16-byte Q load would be the next steps if attention shows up again.

## 5. GPU loss

Timeline (26 Oct 2026, host clock):

* 10:14-11:12 old server (79a608b) serving the "before" benchmark, then
  idle; kernel microbenchmarks and `behemoth_gpu_tests` ran beside it.
  OD state verified at 10:14: `OD_SCLK_OFFSET -500Mhz`, power1_cap 210 W on
  both cards; pollers (`fastpoll`, `gpu_poll2.sh`) running.
* 11:12-11:28 server stopped (PID recorded in
  `/tmp/beh_server_cmdline_before.txt`), block test, new server started
  (PID 3473232, production flags, default attention mode), "Listening" at
  ~11:28 after the 77 s pre-warm.
* 11:28-11:35 "after" benchmark: 39-token c=1 and c=4, 6107-token TTFT and
  128-token decode: all completed (table above).
* 11:35:24 the 20772-token 1-token request (TTFT measurement) started.
* 11:36:28 `gpu_poll2.csv` shows both vendor ids `ffff`; last good sample
  11:36:27.9: P0 206 W, P1 200 W, sclk 2943 / 2938 MHz, 63 / 61 C. dmesg:
  `amdgpu 0000:33:00.0: device lost from bus!` and the same for cc:00.0,
  then repeating SMU bus errors. `fastpoll` has logged `P0=0W P1=0W` since.
* 11:52 noticed (the metrics endpoint still answered with
  `active_requests: 1`); load generators killed by PID; no further GPU work.

What the pollers show about the regime: in the 20 s before the loss both
cards sat at 193-206 W and 2.89-2.96 GHz continuously (the prefill kernel
keeps both ranks busy for the whole chunk). Over the whole day the sampled
peaks were P0 263 W / P1 281 W and 3.09 GHz under the OLD build (before
10:50) and P0 280 W / P1 294 W, 3.08 GHz under the new one, i.e. the 210 W
cap and the -500 MHz offset did not bound the sampled power or clock in
either period; the new prefill kernel made the high-power phase sustained
(a 20k prefill is ~80 chunks back to back) rather than making it higher.
The 6107-token prefill (25 s, same load pattern) completed. Whether a lower
cap, a longer per-chunk gap or a single-rank prefill ordering avoids the
transient is for the user to decide; no clock/power setting was changed by
this work.

State left: server process 3473232 alive but without devices (not killed,
to avoid touching the driver further); `fastpoll` and `gpu_poll2.sh` still
running; working tree clean at this report's commit; build-rocm contains the
final build; the old server command line is in
`/tmp/beh_server_cmdline_before.txt`; benchmark logs in `/tmp/e2e_before.log`,
`/tmp/e2e_after.log`, `/tmp/tests_round4.log`, `/tmp/block_round4.log`.

## 6. Files

* `src/attention.hip`: `Operand<BF16>`, `score_fragment`, `product_fragment`,
  `attention_kernel<BF16>` (legacy), `gqa_decode_kernel<BF16>`,
  `gqa_prefill_kernel<BF16>`, `layout_probe`, `merge_kernel` (unrolled),
  `decode_attention_partition`, `paged_attention` dispatch
  (`BEHEMOTH_ATTN_LEGACY`), `verify_device` layout check.
* `include/behemoth/gpu.hpp`: `AttentionDeviceView::q_bf16`,
  `decode_attention_partition`.
* `include/behemoth/attention.hpp`, `src/attention.cpp`:
  `AttentionData::q_bf16`, reference BF16 query path.
* `src/model.cpp`: partition choice via `decode_attention_partition`
  (`BEHEMOTH_ATTN_PARTITION`, `BEHEMOTH_ATTN_BLOCKS`), `BEHEMOTH_ATTN_QP=bf16`.
* `tests/gpu_tests.cpp`: long-context, mixed-batch, BF16 and partition-policy
  cases.
* `perf/attention_round4/long_attention.cpp`: the microbenchmark.
