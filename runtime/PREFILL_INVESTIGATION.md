# Further prefill investigation

Review baseline: `7874c01`, [PREFILL_SCHED_REPORT.md](PREFILL_SCHED_REPORT.md) and
[PREFIX_CACHE_REPORT.md](PREFIX_CACHE_REPORT.md). No new GPU performance or
quality measurements were made: this workspace has neither ROCm nor GPU devices.
The changes below are an opt-in experiment, not a measured speedup.

## Findings

Dense GEMMs, asynchronous DMA all-gathers, row-batched Hadamards and overlapping
micro-batches already remove much of the initial overhead. Larger chunks alone
are unlikely to produce another large gain. The published 2048-row profile has
3.25 s of GEMM work, 0.60 s of attention, 0.34 s of weight decoding and 0.60 s of
transforms on GPU 0; copy engines are busy for 3.5 s, with 1.4 s exposed without
compute. Wall time is about 5.9 s. Long-context attention grows to 1.5–3 s/step.

For warm conversations, preserve the exact-token prefix cache: reported turn-2
TTFT is 0.24–0.86 s versus 5–133 s cold. Globally pruning tokens would complicate
reuse of those cached higher-layer states. The reply-tokenization mismatch matters
for long reasoning replies, but substituting generated token IDs would violate
cold-request semantics.

## Implemented experiment: BF16 prefill exchange

Transport the remaining FP32 q/k/v, o and down rotated outputs as BF16, as gate/up
already are. The receiver converts the payload into FP32 registers for the same
Hadamard/SV operations. GEMM accumulation remains FP32.

Per rank, per token, per layer, in **each direction**:

| Stage | Current | Experiment |
|---|---:|---:|
| q/k/v | 28 KiB | 14 KiB |
| attention output | 12 KiB | 12 KiB |
| o | 24 KiB | 12 KiB |
| gate/up | 56 KiB | 56 KiB |
| down | 24 KiB | 12 KiB |
| **Total** | **144 KiB** | **106 KiB** |

This is **26.4% fewer exchange bytes**: 26.58 -> 19.56 GB for 2048 tokens x 88
layers, per rank per direction. At the report's 12.5 GB/s duplex bottleneck,
the ideal wire-time floor falls from 2.13 to 1.56 s. That saves about 0.56 s of
communication, not 26.4% of whole-model latency. A tentative end-to-end target
is around 5–15%, depending on exposed communication and attention; it may be
smaller. This is not a credible route to a 4x speedup.

`behemoth_server` default: **512 (on)** since round 5 (PERF_ROUND5_REPORT.md); `--qtip-bf16-prefill-rows 0`
turns it off. Library/tool default (`RuntimeOptions`): 0. The
threshold applies to each overlap micro-batch; valid nonzero values are 64..4096.
A step containing **any one-query request** retains the previous precision,
protecting decode-only and mixed decode/prefill steps. AQLM is unchanged.

BF16 consumers cover per-chunk and row-batched q/k/v and residual epilogues,
including residual+next-input fusion. They avoid a full FP32 expansion of the
gathered payload. Buffer capacity and weight/KV storage are unchanged. Existing
FP32 APIs remain available for the runtime, tests and training bridge.

**Numerical change:** this rounds rotated outputs before Hadamard rather than
only rounding the final projection output. It is not lossless. GPU unit tests
compare new epilogues **bit for bit against FP32 epilogues with explicitly
BF16-rounded inputs**; that checks implementation, not model quality. Do not
enable by default without held-out perplexity, continuations and cache-restore
testing on .229. Restart between A/B configurations so a prefix-cache entry
cannot cross modes; no hot toggle is provided.

## Benchmark correction

`behemoth_prefill_bench` previously set query positions to `ctx..ctx+rows` without
prefilling `0..ctx`. Nonzero-context runs could read uninitialized KV beyond the
short prompts a previous case filled. It now initializes the whole prefix on
the same physical page table before each case, outside timing. Repetition still
overwrites the measured query range. Existing long-context **step-benchmark**
results should be rerun. This does not establish a problem with server/client
TTFT measurements, where the server prefills the full actual prompt.

## Validation and reproduction

Completed here:

- CPU core tests: **83,696 assertions passed**, including payload sizes, disabled
  mode, threshold boundaries and mixed/decode exclusion.
- Python trainer/runtime alignment: **33 passed, 3 GPU tests skipped**.
- CPU-only syntax checking of `behemoth_eval`.

Pending on .229: compiling the modified HIP kernels, GPU tests, full-model quality
and actual A/B timing. Full runtime compilation is unavailable here: HIP and
Boost headers are absent. New tests in `behemoth_qtip_tests` cover both ranks,
rows 1/70, row-batched on/off, residuals and fusion. Full-model modes
`prefill-bf16` and `prefix-bf16` reuse the prefill/mixed-step and host-save/restore
checks with a 64-row threshold to exercise small cases.

After building the runtime on .229:

```sh
runtime/build-rocm/behemoth_qtip_tests
runtime/build-rocm/behemoth_model_tests MODEL prefill-bf16
runtime/build-rocm/behemoth_model_tests MODEL prefix-bf16

# Each invocation loads the model once. Prefix initialization is not timed.
BEHEMOTH_BENCH_PATHS=dense,overlap BEHEMOTH_QTIP_BF16_PREFILL_ROWS=0 \
  runtime/build-rocm/behemoth_prefill_bench MODEL 512,1024,2048 0,8192,16384 3
BEHEMOTH_BENCH_PATHS=dense,overlap BEHEMOTH_QTIP_BF16_PREFILL_ROWS=512 \
  runtime/build-rocm/behemoth_prefill_bench MODEL 512,1024,2048 0,8192,16384 3

# Preliminary quality check: identical corpus and token/chunk settings.
BEHEMOTH_QTIP_BF16_PREFILL_ROWS=0 \
  runtime/build-rocm/behemoth_eval MODEL ppl HELDOUT_CHAT_TEXT 16384 2048
BEHEMOTH_QTIP_BF16_PREFILL_ROWS=512 \
  runtime/build-rocm/behemoth_eval MODEL ppl HELDOUT_CHAT_TEXT 16384 2048
```

Then compare actual GM-CoT prompts/continuations and general text. Repeat baseline
controls to distinguish normal GEMM tuning/summation changes from added rounding.
Test whole steps, both overlap paths, cached-prefix plus tail and concurrent
decode. Throughput alone is insufficient to accept the mode. The tool-only
environment variable selects it in `behemoth_eval` and `behemoth_prefill_bench`;
the server uses the explicit CLI flag.

## Larger opportunities

1. **Long-context attention.** Its measured 15–25 TFLOP/s and growing share of
   16–32k prefill make it the next kernel target. Profile register occupancy,
   softmax and K/V loads; compare smaller GQA head groups or larger query tiles.
   Changing these without gfx1201 compiler resource reports and timings risks
   losing more reuse than it gains in occupancy.
2. **Sequence-parallel, layer-at-a-time cold prefill.** Distribute prompt rows
   across GPUs, exchange each layer's compressed weight shard once, process
   local rows with the full layer, then exchange K/V rather than five wide
   activation vectors. Half the packed model is roughly 15 GB per rank per
   whole prompt versus hundreds of GB of activation exchange for 32k today.
   The extra code shard/dense scratch must be temporary; the full model does
   not fit comfortably on each GPU. This could preserve weights and produce
   normal head-sharded KV for decode, but needs balanced query positions, KV
   redistribution, scratch budgeting and decode scheduling. It is not implemented
   or benchmarked; the byte comparison is not a speedup prediction.
3. **PCIe topology.** Establish why GPU 0 sustains ~13 GB/s versus ~27 GB/s on
   GPU 1 by checking the complete upstream link path, not only endpoint widths.
   The prefix-cache arena already uses GPU-local NUMA allocation and shows the
   same asymmetry, so NUMA placement alone is not a demonstrated fix.

These directions retain the prefix-cache advantage without requiring TokenGate
training or acceptance of context loss.
