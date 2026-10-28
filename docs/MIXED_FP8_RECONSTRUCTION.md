# FP8 K/V weights and teacher-guided block reconstruction

This workflow preserves the original source checkpoint's `k_proj` and `v_proj`
weights as scaled FP8 E4M3, leaving Q/O and the MLP in QTIP 2-bit. It can upgrade
an existing QTIP model or produce mixed checkpoints during fresh quantization.
The original FP8 checkpoint is required: widening a 2-bit reconstruction cannot
recover the original K/V weights.

For Behemoth's 88 layers and two 1024-by-12288 K/V projections per layer, the
change adds approximately 1.66 decimal GB of weight payload across both GPUs.
It does not change the precision of the attention KV cache.

## Build and validation

Build the training tools with the existing HIP configuration, then rebuild the
runtime so its loader and projection kernels understand the mixed format:

```sh
cmake -S training -B build-training -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DQTIP_TRAIN_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-training -j8
cmake --build runtime/build-rocm -j8
ctest --test-dir build-training --output-on-failure
ctest --test-dir runtime/build-rocm -R 'core_correctness|gpu_correctness|qtip_correctness' --output-on-failure
```

Local development validation passed CPU conversion/export, mixed-layer finite
differences, padding-aware reconstruction loss, best-validation selection and
completed-block resume, including AddressSanitizer/UndefinedBehaviorSanitizer
runs. Runtime CPU tests cover split FP8/scale shards and malformed checkpoints.
Modified HIP kernels compile for gfx1201, and the engine
host code compiles against the ROCm headers. There is no GPU in the development
workspace; GPU numerical execution, real-model memory/timing and repetition
quality remain unmeasured. GPU tests exercise W8A16 GEMV/WMMA projections and
the Q-only tensor-parallel epilogue, plus mixed-student/full-FP8-teacher layer
forward/backward comparisons against the CPU. The CPU fixture's reconstruction improvement
is a numerical test, not evidence of improved Behemoth generation quality.

## Upgrade an existing model

```sh
build-training/qtip_fp8_kv \
  --model /models/behemoth-qtip \
  --source /models/Behemoth-R1-123B-v2-FP8-Dynamic \
  --out /models/behemoth-qtip-fp8kv
```

The output must be a new directory. K/V weight bytes and per-output scales are
copied from the source; their trellis/SU/SV/LUT tensors are removed. All other
tensors are copied without changes. Geometry, embedding/head values and
available tokenizer files must agree with the source. The output index and
configuration are rewritten and published through a temporary directory.

Fresh `qtip_quant` runs default to `--kv-format fp8`. K/V are excluded
from QTIP optimization and copied directly from the source. Use that flag again
with `--mode assemble`; `--kv-format qtip` retains the legacy all-QTIP format
and must be used when resuming or assembling an older all-QTIP work directory.
A work directory with completed layers of another format is rejected.

The inference server automatically selects the mixed path from the checkpoint;
no new server flag is needed. K/V remain stored on GPU as FP8, with BF16 input
activations and FP32 accumulation. Decode uses a direct weight-reading GEMV;
prefill widens FP8 weight tiles exactly into BF16 LDS operands for WMMA. This is
**W8A16**, not FP8 activation training. Original channel scales are applied before
BF16 projection-output rounding. Each rank computes its own original-space K/V
heads; only Q retains the QTIP transform and rotated-output all-gather.

Mixed training layers use the generic composition of the native operations;
QTIP-only layers keep their fused path. Measure mixed-model memory and timing
before scheduling a long reconstruction or QAT run.

First compare the upgraded model against the original QTIP model on fixed
prompts and sampling settings. This isolates the precision change before fitting.

## Prepare complete, masked conversations

Reconstruction consumes the native prepared dataset format, including document
segments, positions and padding. It requires a distinct held-out conversation
split; validation conversations are never used for gradient updates. For RP-only
data, use `--gm-frac 1` so no general-text file is required:

```sh
build-training/qtip_prepare --data rp.jsonl --tokenizer /models/behemoth-qtip-fp8kv \
  --out rp-32k-train --split train --gm-frac 1 --seqlen 32768
build-training/qtip_prepare --data rp.jsonl --tokenizer /models/behemoth-qtip-fp8kv \
  --out rp-32k-heldout --split heldout --gm-frac 1 --seqlen 32768
```

Use the same `--split-metadata FILE` for both when retaining an existing holdout.
The reconstruction path uses the trainer's bounded-memory attention and honors
these masks. This does not retrofit long-context tiling or metadata masks into
the older Hessian collector in `src/model.cpp`.

## Fit blocks against the source FP8 teacher

```sh
build-training/qtip_reconstruct \
  --model /models/behemoth-qtip-fp8kv \
  --teacher /models/Behemoth-R1-123B-v2-FP8-Dynamic \
  --data rp-32k-train --validation-data rp-32k-heldout \
  --work reconstruction-work --out /models/behemoth-qtip-reconstructed \
  --device 0 --teacher-device 1 --arena-gb 24 --teacher-arena-gb 12 \
  --samples 4 --validation-samples 1 --steps 64 --eval-every 8 \
  --lr 1e-4 --norm-lr 1e-5 --rows 1024
```

Start with `--layers 0:2 --steps 16` and separate pilot work/output directories
to measure memory, loss and speed. Omit `--layers` for all layers. `--rows` chunks
the MLP; it does not cut the attention context. The two replicas use separate
devices by default. Co-locating them requires budgeting the sum of both arenas.
Host memory holds source FP8 and student compressed weights plus cached block
inputs/targets: about 160 GB of weights for this model, with a few 32k samples
adding tens of GB. An 88-layer activation-checkpoint tape is not allocated.

For each layer:

1. Run the frozen source FP8 block on teacher-prefix activations to cache targets.
2. Run the student block on activations from the already reconstructed student
   prefix. The two paths are propagated independently, including validation.
3. Minimize mean squared block-output error over all nonpadding tokens. This is
   a teacher output objective, not next-token cross-entropy or a hard-label loss.
4. Optimize this block's SU/SV, positive log-ratio Wscale and norm parameters,
   using per-group clipping. FP8 K/V weights/scales and QTIP trellis codes stay
   frozen. This pass does not search new discrete trellis assignments.
5. Retain the best held-out step, including the unchanged starting block as a
   candidate; propagate its outputs into the next student block.

The source teacher is evaluated by native BF16-activation/FP32-accumulate
operations using the original FP8 weights and scales. It is not the inference
runtime's FP8 KV-cache/activation arithmetic. Held-out block MSE selects an
update but does not establish reduced repetition; compare full generations,
held-out loss, task quality and reasoning-to-answer transitions separately.

Progress is logged as JSON. After every completed block, parameter state and its
checksum are checkpointed under `--work`. Repeat the same command with `--resume`
to continue; earlier accepted blocks are replayed to rebuild the two activation
paths. Model/data identities and fitting options must match. The final model is
written to a new directory, with a reconstruction report in `config.json`.
Training data, teacher weights and source checkpoints are never overwritten.

## Checkpoint representation

`config.json` marks mixed checkpoints with `qtip_quant.kv_format = "fp8"`.
Each K/V projection contains:

- `.weight`: `F8_E4M3 [out, in]` in the original, unrotated orientation.
- `.weight_scale`: BF16 or FP32 per-output-channel scales, with `out` elements.

There is no K/V `.trellis`, `.SU`, `.SV` or `.tlut`. The remaining projections
use the existing QTIP format. Malformed, ambiguous, nonfinite or incorrectly
declared FP8 projections are rejected. Native QAT/evaluation/export also accept
the mixed checkpoint and leave its FP8 K/V tensors frozen.
