# Native QTIP fine-tuning

The fine-tuner and data preparer are standalone C++20/HIP executables. Build,
prepare, train, resume, evaluate, export, benchmark, and runtime-trace checking
require **no Python, PyTorch, Triton, libtorch, or Python bindings**. The former
`qtip_training` package and Python training scripts have been replaced.
The remaining legacy Python reference, calibration, benchmark and test utilities
have also been removed from the repository. Quantization accepts pre-tokenized
raw calibration data; benchmark shell scripts use `jq` for JSON processing.

## Build

Dependencies: CMake 3.25+, a C++20 compiler, OpenSSL development headers, threads,
and optionally OpenMP. The GPU build additionally needs ROCm with HIP,
hipBLASLt and rocWMMA headers. It targets gfx1201, matching the runtime. The
preparer uses native SentencePiece; CMake uses an installed CMake package or
fetches the pinned v0.2.1 C++ release. An offline build can supply
`-DFETCHCONTENT_SOURCE_DIR_SPM=/path/to/sentencepiece`.

```sh
cmake -S training -B build-training -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DQTIP_TRAIN_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-training -j8
ctest --test-dir build-training --output-on-failure
```

For CPU verification, omit `-DQTIP_TRAIN_HIP=ON` and ROCm arguments. CPU mode
rejects models wider than 1024; it is a numerical test backend, not a practical
way to train Behemoth. `-DQTIP_TRAIN_TOKENIZER=OFF` skips the preparer and its
SentencePiece dependency. All executable help is available via `--help`.

## Prepare data

Use the calibration split metadata to preserve the established held-out
conversations. The native shuffle uses `std::mt19937_64`; newly prepared row
order differs from the former Python shuffle. Recorded held-out IDs are
preserved exactly. Existing version-1 packed datasets remain readable unchanged.

```sh
build-training/qtip_prepare \
  --data /home/charles/train_data/train_cot_final.jsonl \
  --tokenizer /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output \
  --split-metadata /mnt/nvme/qtip_quant/calib_gmcot_512x8192.i32.json \
  --general /mnt/nvme/qtip_quant/calib_general.i32 \
  --general-seqlen 2048 --seqlen 24576 --gm-frac .8 \
  --out /mnt/nvme/qtip_quant/train_gmcot_24k

build-training/qtip_prepare \
  --data /home/charles/train_data/train_cot_final.jsonl \
  --tokenizer /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output \
  --split-metadata /mnt/nvme/qtip_quant/calib_gmcot_512x8192.i32.json \
  --split heldout --seqlen 24576 --gm-frac 1 \
  --out /mnt/nvme/qtip_quant/heldout_gmcot_24k
```

The preparer implements the runtime's legacy SentencePiece tokenizer and chat
format (`<s>`, `[INST]`, `[SYSTEM_PROMPT]`, assistant space and `</s>`). It
strips old inline thinking and inserts `reasoning_content` inside `<think>`.
It tokenizes the complete rendered conversation, records assistant spans using
SentencePiece byte offsets, and supervises the assistant end-of-turn token.
Unsupported added-token behavior is rejected. Arbitrary Hugging Face chat
Jinja templates are not supported.

Output: checksummed `metadata.json` plus raw `tokens.i32`, `loss.u8`,
`segments.i32`, `positions.i32`, all shaped `[nseq,seqlen]`. Attention and shifted
labels stop at document boundaries. RoPE positions restart per document and
continue across windows of a long document. Final short rows are padded and
masked. Each source token is used at most once; insufficient general text is an
error. The mixture is approximate at packed-row granularity. An optional
`--max-tokens` on the preparer limits storage; the trainer separately limits its
single traversal of the data.

## Train, resume, export

```sh
build-training/qtip_finetune \
  --model /mnt/nvme/models/Behemoth-R1-123B-v2-QTIP-2bit-gmcot \
  --data /mnt/nvme/qtip_quant/train_gmcot_24k \
  --devices 0,1 --work /mnt/nvme/qtip_quant/qat-pilot \
  --max-tokens 500000 --warmup-steps 10 \
  --rows 1024 --loss-rows 128 --checkpoint-every 10 \
  --out /mnt/nvme/models/Behemoth-R1-123B-v2-QTIP-2bit-gmcot-ft
```

Resume with the same arguments and `--resume /mnt/nvme/qtip_quant/qat-pilot/latest.json`.
If `--out` already exists, use a new export path or omit export. For an
interruption test use `--stop-after-steps N`, then remove that option when
resuming. It is an absolute step limit and does not change the LR schedule.
Training stops at the smaller of one dataset traversal and the token budget
rounded down to whole packed sequences, including a partial final GPU batch.

Trainable parameters: relaxed SU tied across q/k/v and gate/up, separate SU for
o/down, per-linear relaxed SV and scalar Wscale, and RMSNorm weights. Trellis,
LUT, embedding, and lm_head stay frozen. Forward uses BF16 effective SU/norms,
BF16 decoded weights and GEMM operands, FP32 accumulation, BF16 gate/up rotated
outputs, and BF16 activation/residual boundaries. Backward uses explicit
straight-through adjoints and recomputation. AdamW keeps FP32 master parameters
and moments, lower norm LR, norm-only decay, global clipping, and warmup followed
by cosine decay. Loss uses assistant targets and one global supervised-token
denominator across all active GPUs. Idle replicas receive the same update on a
partial final step.

Each GPU uses a fixed arena (default 24 decimal GB, `--arena-gb`), two packed
weight slots, two BF16 activation staging slots, and a copy/offload stream with
events. Packed host weights are registered once and shared by replicas. Layer
input checkpoints live in pinned BF16 host memory. Decoded BF16 matrices are
reused within a layer pass. MLP intermediates and vocabulary logits are
chunked; attention uses native BF16 WMMA tiles and never allocates a sequence
squared tensor. Transfers use registered weights or reusable pinned staging.
At 24,576 tokens, activation checkpoints require about 53 GB of pinned host RAM
per GPU, in addition to packed weights. Arena exhaustion reports a clear error.
Actual peak memory and throughput must be measured on the target machine.

Checkpoints are native format v3: `state.safetensors` (FP32 parameters and Adam
moments), checksummed `metadata.json`, and an atomic `latest.json`. State files
and directories are synced before publishing the latest pointer. Resume checks
model shards, dataset identity, parameter layout, device count, backend and
schedule. GPU atomic reductions can introduce small floating-point differences
between repeated runs. Old Python `.pt` optimizer checkpoints cannot be resumed;
use their already exported HF model as a new training source. The native tool
never unpickles a file or invokes Python.

Export creates a new directory, preserves all original tensor names, shapes,
dtypes and frozen tensor bytes, folds Wscale into FP32 SV, and rounds SU/norms
to their original BF16 precision. SU aliases stay tied. Export refuses source
mutation and an existing destination.

## Evaluate and benchmark

```sh
build-training/qtip_finetune --eval-only --devices 0 \
  --model /mnt/nvme/models/Behemoth-R1-123B-v2-QTIP-2bit-gmcot-ft \
  --data /mnt/nvme/qtip_quant/heldout_gmcot_24k

# Model, device, sequence length, layer index, MLP chunk rows[, arena GB (24), repeats (1)]:
build-training/qtip_bench /mnt/nvme/models/Behemoth-R1-123B-v2-QTIP-2bit-gmcot 0 24576 0 1024
```

The benchmark reports decode, forward, backward seconds and peak arena bytes.
Training emits JSON loss, gradient norm, LR, step time and packed tokens/second.
With repeats greater than one, the layer is decoded once and run repeatedly with
unchanged inputs/parameters and cleared gradients. Each iteration prints timings;
the final report gives medians excluding the first iteration. This separates
GEMM tuning and other first-call setup from steady execution:

```sh
QTIP_TRAIN_GEMM_TUNE=1 build-training/qtip_bench MODEL 0 8192 0 1024 24 4
```

The native trainer shares the input Hadamard transform across Q/K/V and across
gate/up. HIP transforms write BF16 GEMM operands directly, including in projection
backward, avoiding an FP32 transform buffer followed by a separate conversion.
The CPU reference preserves its original rounding behavior. Persistent activations
and gradients still use FP32 storage; this is not a full BF16 activation-storage
conversion.

SU/SV and RMSNorm parameter gradients use coalesced two-pass column reductions.
RMSNorm no longer issues one global atomic addition per activation (about
53.5 billion per 24,576-token, 88-layer training pass, including final norm).
Scratch space is `4 * ceil(rows / 256) * columns` bytes, about 4.5 MiB for a
24,576-by-12,288 reduction. Scalar Wscale reductions still use a small number of
atomic additions. RoPE caches sine/cosine per batch, dimension and theta; batch
changes invalidate the cache even when their lengths match. At 24,576 tokens and
head dimension 128 the cache uses 12 MiB. Contiguous host transfers use up to
8 MiB per copy; each loss chunk downloads its row losses together rather than
synchronizing for every four-byte value.

Optional GEMM tuning benchmarks up to eight valid hipBLASLt heuristic candidates
on the actual device, using one warmup and three timed calls per candidate. The
winner is cached per matrix geometry for that process. Enable it with:

```sh
QTIP_TRAIN_GEMM_TUNE=1 build-training/qtip_bench MODEL 0 24576 0 1024
# The same environment variable works with qtip_finetune.
```

Tuning is off by default. It prints the selected candidate and timing to stderr
and adds setup work to the first call of each geometry, so first-step timing
includes tuning. Choices are not persisted across processes. The GPU test suite
checks both the default and tuned paths; it skips both when no GPU is available.
Numerical and performance validation on gfx1201 is still required for these
optimizations; CPU tests and compilation alone cannot establish GPU speedups.

To isolate training attention without loading a model, run the native attention
benchmark. Start small, then use the actual packed-row and document lengths:

```sh
ctest --test-dir build-training -R native_gpu --output-on-failure
# Sequence length, device, document length, repeats, heads, KV heads, arena GB:
build-training/qtip_attention_bench 512 0 512 3 96 8 24
build-training/qtip_attention_bench 24576 0 24576 3 96 8 24
# Same packed row, with separate causal documents of at most 2048 tokens:
build-training/qtip_attention_bench 24576 0 2048 3 96 8 24
```

It reports each iteration immediately, discards the first as warmup, and reports
median forward/backward time, useful matrix TFLOP/s and peak arena bytes.
Backward timing includes the gradient copy/rounding and softmax correction.
Useful FLOPs count causal, unmasked pairs; they exclude tile padding, reductions,
conversions, and memory traffic. These are attention-only timings, not full-step
predictions. The CPU mode (`32 cpu 32 1 8 2`) is for checking the executable.

Training attention backward assigns each dQ tile and each dK/dV tile to one
thread block. It accumulates gradients in registers and writes each element
once, including the reduction across grouped query heads. It uses document end
boundaries to stop the dK/dV scan. The previous query-owned backward issued
contended global atomic additions for every key tile: for one contiguous
24,576-token sequence with 96 query heads and eight KV heads, that was
464,460,447,744 FP32 atomic additions per layer. The replacement performs
50,331,648 ordinary dK/dV element stores per layer, with no attention gradient
atomics. This operation-count reduction is **not** an equivalent speedup claim.

The tradeoff is recomputation: dQ and dK/dV independently calculate scores and
probability gradients, giving seven matrix products per causal pair in backward
instead of five. An extra FP32 softmax-correction buffer uses
`4 * heads * sequence_length` bytes (9 MiB at 96 heads and 24,576 tokens). There
is no quadratic or per-query-head dK/dV workspace. GPU numerical checks and
timings must run on gfx1201; compilation and CPU tests alone do not establish
GPU correctness or speedup. `--rows` chunks the MLP and does not shorten attention.

Training forward now uses native gfx1201 BF16 WMMA operands. A wave owns 16
queries for one head; all query heads sharing a KV head stage K/V together.
Scores and the output accumulator stay in registers, softmax uses a cross-half
wave shuffle, and probabilities use BF16 with FP32 accumulation. The forward
and backward recomputation use the same path. Packed-document starts and causal
masks are applied per query, including all-masked intermediate key tiles. LSE
remains in natural-log units for backward even though softmax uses `exp2`.
The earlier rocWMMA forward is retained for A/B checks and GQA groups above 32:

```sh
QTIP_TRAIN_ATTN_FORWARD=legacy build-training/qtip_attention_bench 8192 0
QTIP_TRAIN_ATTN_FORWARD=native build-training/qtip_attention_bench 8192 0
# Compare full layers with warmup excluded, GEMM tuning disabled in both runs:
QTIP_TRAIN_ATTN_FORWARD=legacy build-training/qtip_bench MODEL 0 32768 0 1024 24 4
QTIP_TRAIN_ATTN_FORWARD=native build-training/qtip_bench MODEL 0 32768 0 1024 24 4
```

The native forward is the default for GQA groups up to 32. This implementation
uses the runtime kernel's operand layout, register accumulation, and parallel
softmax; it does not use the runtime's FP8 cache or probability quantization.
Run GPU correctness tests and measure both paths on gfx1201 before relying on
speedup estimates.

Backward also has native BF16 WMMA kernels. dQ shares K/V staging across a GQA
group, keeps Q/dO and its output accumulator in registers, and computes score
and probability derivatives within each wave. dK/dV use paired waves: one owns
dK and computes QK scores/dS, the other owns dV and computes dP. They exchange
FP32 dP and BF16 P through LDS. Each wave keeps one K/V operand and its gradient
accumulator in registers. Four pairs share a 16-query Q/dO load across 64 keys,
staged in both orientations directly. This doubles Q/dO reuse compared with
`04a8fc9` and distributes its two score products between waves. The tradeoff is
one additional barrier per query tile and the small FP32 dP exchange. There is
no in-place Q/dO transpose or extra global workspace.

Heads reduce in a fixed order with one writer per gradient element. Gradients
accumulate in FP32 registers and are rounded at the existing HIP boundary. Both
kernels preserve document and causal masks; there is no quadratic or per-query-head
gradient workspace and no attention-gradient atomic addition. The seven backward
matrix products per causal pair are unchanged. The earlier backward remains
available for A/B checks:

```sh
ctest --test-dir build-training -R native_gpu --output-on-failure
QTIP_TRAIN_ATTN_BACKWARD=legacy build-training/qtip_attention_bench 8192 0
QTIP_TRAIN_ATTN_BACKWARD=native build-training/qtip_attention_bench 8192 0
# Separate GPU-event timings of dQ and dK/dV, in milliseconds on stderr:
QTIP_TRAIN_ATTN_PROFILE=1 build-training/qtip_attention_bench 32768 0
```

`QTIP_TRAIN_ATTN_DKV` only applies to native backward and keeps dQ unchanged:

| Value | dK/dV implementation |
| --- | --- |
| `balanced` (default) | Balanced score work; four wave pairs own 64 keys |
| `balanced32` | Same algorithm; two wave pairs own 32 keys |
| `wave` | GPU-validated paired-wave kernel from `04a8fc9` |
| `staged` | Native dK/dV kernel from `a067e7e` |

CTest registers all four native dK/dV paths against the CPU reference,
including strided views, 16/32/64-key tile tails, GQA groups through 32, packed
boundaries, one-token documents, and repeat-call determinism. Run these checks
before measuring the isolated change, then compare the steady medians and
`dkv_ms` (stderr) for contiguous and packed sequences:

```sh
cmake --build build-training -j8
ctest --test-dir build-training --output-on-failure
for dkv in wave balanced32 balanced; do
  QTIP_TRAIN_ATTN_BACKWARD=native QTIP_TRAIN_ATTN_DKV=$dkv QTIP_TRAIN_ATTN_PROFILE=1 \
    build-training/qtip_attention_bench 8192 0 8192 3 96 8 24
  QTIP_TRAIN_ATTN_BACKWARD=native QTIP_TRAIN_ATTN_DKV=$dkv QTIP_TRAIN_ATTN_PROFILE=1 \
    build-training/qtip_attention_bench 32768 0 32768 3 96 8 24
  QTIP_TRAIN_ATTN_BACKWARD=native QTIP_TRAIN_ATTN_DKV=$dkv QTIP_TRAIN_ATTN_PROFILE=1 \
    build-training/qtip_attention_bench 32768 0 2048 3 96 8 24
done
```

The user reported all seven CTest tests passing on `04a8fc9`, no MES errors,
and 32k layer time improving from 28.4 to 10.2 seconds. The balanced variants
compile for gfx1201 with zero spills but still require GPU execution and timing.
See [training attention performance](TRAINING_ATTENTION_PERF.md) for the measured
baseline, compiler resource counts, and step-time bounds.

The profile flag adds a synchronization after the two kernels. It excludes the
softmax-correction kernel, gradient copy/rounding, and forward recomputation;
the benchmark's backward wall time includes the correction and copy/rounding.
Native backward is used by default for GQA groups up to 32; larger groups use
the earlier kernels. These new kernels require on-device numerical validation
and timings; compilation and CPU tests alone do not establish GPU correctness
or throughput.

For strict runtime parity, make a JSON request containing `tokens` and
`logit_positions` (1–128 positions). Capture the actual native runtime output:

```sh
BEHEMOTH_ATTN_QP=bf16 build-training/qtip_check --capture \
  --model MODEL --eval-binary /path/to/behemoth_eval \
  --prompt request.json --trace runtime-trace.json --chunk 256
build-training/qtip_check --model MODEL --trace runtime-trace.json --device 0
```

Capture executes `behemoth_eval` directly and stamps a model digest. Checking
requires exactly matching logits and exits nonzero on a mismatch; error and
argmax diagnostics do not turn approximate agreement into a pass. BF16 GEMM
summation and tiled attention need not be bit-identical to the serving kernels.
Full-model runtime parity is still an explicit pending verification gate.

## Verification status

The native CPU suite checks parameter gradients through a complete two-layer
transformer against finite differences, document/assistant masks, generalized
Hadamard inversion, single-batch overfitting, immutable export tensors, tied SU,
bit-exact interrupted/resumed CPU training, and checkpoint identity rejection.
Native SentencePiece tests cover chat boundaries, assistant EOS, Unicode,
reasoning, packing, padding and continued positions. No Python is used by these
tests. `--reference` selects unrounded CPU math for numerical differentiation;
normal CPU execution emulates BF16 boundaries.

Both CPU and gfx1201 HIP builds compile and link (GPU compile checked with ROCm 7.1).
The user reported passing GPU execution/gradient checks and real-layer timings
on `04a8fc9`; the new balanced dK/dV variants still need those GPU checks.
Full-logit runtime parity, held-out quality, and the 0.5M/5M-token runs remain
separate validation steps. No speed or quality result is inferred from CPU tests.
