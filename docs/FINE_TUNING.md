# QTIP continuous-parameter fine-tuning

This implements the phase 3 trainer in [PLAN.md](PLAN.md), with portable PyTorch
kernels and a native adapter to the committed QTIP kernels under `runtime/`.
The native backend targets ROCm/gfx1201; PyTorch names the GPU API `torch.cuda`.
Install a PyTorch build
matching the ROCm version on .229, then `pip install -r requirements-train.txt`.
The quantizer's CMake targets are unchanged.

## Implementation and current verification

- Shared packed weights in pinned host RAM; two device weight slots per GPU.
  Events prevent slot reuse until compute finishes; a copy stream prefetches the
  next weights. Forward activation checkpoints use a separate offload stream.
  Backward prefetches checkpoints into two reusable device slots in reverse
  layer order and recomputes the layer. MLP/QKV intermediates are row-checkpointed.
- Independent sequences per GPU; one pinned-host gradient sum per optimizer
  step. Each local loss uses the global assistant-token denominator, so sequences
  with different assistant counts have the correct weight. A final partial batch
  also updates idle replicas, preserving identical AdamW state.
- FP32 trainable SU, relaxed SV, one Wscale per linear, both layer norms and
  final norm. AdamW uses separate sign/scale and norm groups; signs/scales have
  zero decay. Embeddings, head, codes, LUT, rcp and tp_rank stay frozen.
- SU is tied across q/k/v and across gate/up, matching `load_qtip_layer` in the
  runtime. Gradients from those uses sum into one parameter; clipping, host
  reduction and AdamW update it once. Input checkpoints with unequal grouped SU,
  unequal SU dtypes or differing per-layer LUT bit patterns are rejected. Export
  checks these constraints too. This fixes the original trainer's independent-SU
  parameterization, whose exports the runtime would reject after training.
  The trainer accepts the quantizer's BF16 SU / FP32 SV storage convention.
- The reference trellis forward and transpose decode output-row tiles instead of
  constructing a dense layer weight. The Hadamard uses butterfly passes and the
  quantizer's generated hadK tables, including their nonsymmetric orientation.
  Attention uses online tiled softmax and an analytic tiled backward; neither
  direction allocates T x T storage. Loss streams token-row chunks through the
  frozen head and recomputes them in backward, avoiding full sequence logits.
- Atomic, fsynced checkpoints contain trainables, FP32 AdamW state, next data
  position, token counters and schedule settings. Source/model shard and dataset
  SHA256 identities prevent resuming on different weights or data. The model is
  hashed once on load, so startup includes a full sequential checkpoint read.
  Checkpoint and gradient D2H transfers use reusable pinned staging buffers.
- Export retains the original tensor names, shapes, dtypes, config and index.
  Learned Wscale is folded into SV. Only SU/SV and norm values change. Trellises,
  LUTs, embeddings and head remain byte-identical as tensors. Export writes a new
  directory atomically and refuses an existing destination or source overwrite.

Run the verification suite from the repo root:

```sh
OMP_NUM_THREADS=2 python -m pytest -q tests/test_training.py tests/test_runtime_alignment.py
```

Tests cover upstream reference vectors, finite differences for every backward
operation, a dense autograd single-layer oracle, resident-vs-streamed multilayer
gradients, single-batch overfit, host reduction, exact CPU crash/resume, and
frozen-tensor preservation/export reload. GPU slot reuse and two-GPU reduction
tests are included and skip when the corresponding hardware is absent. The
alignment suite builds a CPU shared library directly from `runtime/src/qtip.cpp`
and checks the Python decoder, BF16 projection rounding and straight-through
adjoints against that implementation. It requires CMake and a C++20 compiler;
the build uses vendored JSON headers and downloads nothing.

**Still pending:** compiling/exercising the HIP bridge on .229, bit-identical
`behemoth_server` logits, a real-layer comparison, and target memory, stability
and performance measurements. Runtime projection precision is implemented, but
the trainer's tiled attention still differs from paged FP8 K/V and FP8/BF16 Q/P
attention in the server. Full-server parity is therefore **not established**;
the strict checker is expected to expose differences. Native transpose uses
runtime-decoded BF16 row tiles and FP32 BLAS, not a fused transposed trellis GEMM.
Neither backend substantiates the plan's throughput estimates. No Behemoth
training run has been performed. Complete the
plan's real-layer/runtime checks before the pilot. Phase 1 calibration/re-quant
and phase 2's AQLM comparison/repetition-generation harness remain separate work.

## Prepare data

Reuse the held-out IDs from the existing calibration metadata. Omitting
`--split-metadata` recreates the original 10% split with seed 0, but supplying it
explicitly is preferable. Assistant reasoning is inlined exactly as in
`make_calib_gmcot.py`. Assistant text, `<think>` and end-of-turn tokens are loss
targets; user/system/header tokens are masked. Token offsets are taken from a
single full-template tokenization, avoiding separately tokenized BPE boundaries.
A fast tokenizer with a prefix-stable chat template is required; incompatible
templates fail explicitly.

```sh
python tools/make_train_gmcot.py \
  --data /home/charles/train_data/train_cot_final.jsonl \
  --tokenizer /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output \
  --split-metadata /mnt/nvme/qtip_quant/calib_gmcot_512x8192.i32.json \
  --general /mnt/nvme/qtip_quant/calib_2048x2048.i32 \
  --seqlen 24576 --gm-frac 0.8 \
  --out /mnt/nvme/qtip_quant/train_gmcot_24k

python tools/make_train_gmcot.py \
  --data /home/charles/train_data/train_cot_final.jsonl \
  --tokenizer /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output \
  --split-metadata /mnt/nvme/qtip_quant/calib_gmcot_512x8192.i32.json \
  --split heldout --gm-frac 1 --seqlen 24576 \
  --out /mnt/nvme/qtip_quant/heldout_gmcot_24k
```

The output directory contains raw `tokens.i32`, `loss.u8`, `segments.i32`,
`positions.i32` ([nseq, seqlen]) and checksummed `metadata.json`. Short
conversations are packed without cross-document attention or boundary targets.
Each document has its own RoPE positions. Long conversations are split into
windows; final short rows are padded and masked. General calibration sequences
retain their own document boundaries. Neither GM conversations nor general
sequences are sampled repeatedly. The 80/20 share is approximate at row
granularity, and the last GM row may contain padding. The trainer traverses the
prepared, seeded order once at most.

Prepare all training data once to permit 0.5M/5M token budgets against the same
data identity. `--max-tokens` on the data preparer optionally limits storage; it
is not needed for the pilot.

## Runtime backend and parity

Build the bridge on .229 (ROCm PyTorch must also be installed):

```sh
cmake -S qtip_training/native -B build-training -DCMAKE_BUILD_TYPE=Release \
  -DQTIP_TRAINING_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-training -j8
```

This compiles the actual `runtime/src/qtip.hip` into
`build-training/libqtip_training_hip.so`. It needs HIP but does not link the
server, RCCL, tokenizer or libtorch. A C ABI lets PyTorch own the two weight slots,
outputs, scratch and current stream. The runtime's owning GpuLinear API remains
available; a new non-owning LinearView overload avoids uploading/copying a third
layer. `QTIP_TRAINING_LIBRARY` may select an absolute alternate library path.

`--backend auto` (the default) selects `runtime` on GPUs and `runtime-reference`
on CPU. `runtime` fails early if ROCm, gfx1201, or the built library is missing;
it never silently falls back. Forward uses the runtime GEMV/WMMA and slice
reduction kernels; transpose decodes bounded row tiles through the same runtime
decoder and accumulates FP32 BLAS products. Runtime Hadamards handle the real
1024/12288/28672 dimensions; tiny verification dimensions use the reference
butterfly. All native calls run on PyTorch's current device stream. LUT cache
entries include tensor versions, so slot pointer reuse cannot select an old LUT.

`runtime-reference` emulates the projection's BF16 rotated inputs and BF16 LUT,
including BF16 gate/up rotated-output exchange. SU and norms are rounded to their
export BF16 precision for forward, with straight-through gradients. SV/Wscale
are combined in FP32 before the output multiply. SwiGLU rounds once after the
FP32 SiLU/product. Pure `reference` retains unrounded differentiable math for
finite-difference checks; runtime rounding uses straight-through adjoint tests
because its numerical finite-difference derivative is zero between thresholds.

An optional `--backend package.module` may expose `create_backend()` returning:

```python
class Backend:
    device_rounding = True  # Enable runtime BF16 rounding with straight-through gradients.

    def hadamard(self, x, transpose=False):
        # Return hadU(x), or hadUt(x) for transpose=True, normalized by sqrt(n).
        ...

    def matmul(self, x, trellis, tlut, m, n, transpose=False):
        # Frozen rotated matrix W is m x n, kernel-layout I16 trellis.
        # False: x[..., n] @ W.T -> [..., m]
        # True:  x[..., m] @ W   -> [..., n]
        ...

def create_backend():
    return Backend()
```

The trainer handles SU/SV/Wscale and their analytic gradients. Kernel operations
must use PyTorch's **current device stream**, preserve input dtype/device, return
new outputs without mutating inputs, and never retain a weight-slot reference
after the operation. Inputs and packed weights are already on that device.
Production compute is bf16 at layer boundaries and fp32 for rotations and
accumulation. Pure reference verification also exercises float64; the native
adapter accepts FP32 compute tensors. Forward matmul rounds rotated inputs to
BF16; transpose must preserve FP32 upstream gradients. Adapters must define and
test conversion behavior against this contract.

For runtime parity, `behemoth_eval` now has a `trace` mode accepting exact token
ids and emitting selected-position logits without tokenizer/BOS changes. The
capture tool converts this real runtime output into a safetensors trace tied
to the source model's SHA256 identity. A prompt file supplies `tokens` and optional
`logit_positions` (defaults to the last token). Only a single contiguous document
is supported, matching Session's paged cache; packed-document inputs are rejected.

```sh
python tools/capture_qat_runtime.py --eval-binary runtime/build/behemoth_eval \
  --model /path/to/quant --prompt /path/to/prompt.safetensors \
  --out /path/to/server-trace.safetensors --chunk 256
```

Trace tensors are `tokens` (I64 [T]), `logit_positions` (I64 [N]), `logits` (F32
[N,vocab]), optionally `segments` and `positions` (I64 [T]). Select a few logit
positions to keep the trace small. The checker compares every selected logit
exactly and exits nonzero on **any** difference, with the maximum absolute error:

```sh
python tools/check_qat_runtime.py --model /path/to/quant \
  --trace /path/to/server-trace.safetensors --device cuda:0 \
  --backend runtime
```

A successful approximate comparison is insufficient for the bit-identical check.
Attention, tensor-parallel stacking/slice reductions, and final-head arithmetic
can still differ. No tolerance option turns an approximate match into a pass.
The checker rejects a captured source-model identity mismatch.

## Pilot, resume, export, and score

After hardware/runtime verification, use the GM-calibrated quant as input. The
following selects the native adapter. Resolve the outstanding runtime/real-layer
checks before treating this as a production training command.

```sh
python tools/finetune_qtip.py \
  --model /mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-2bit-gmcot \
  --data /mnt/nvme/qtip_quant/train_gmcot_24k \
  --work /mnt/nvme/qtip_quant/ft-pilot --devices cuda:0,cuda:1 \
  --backend runtime --max-tokens 500000 \
  --lr 1e-4 --norm-lr 1e-5 --warmup-steps 10 --clip 1 \
  --checkpoint-every 10 \
  --out /mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-2bit-gmcot-ft-pilot
```

At 24,576 tokens, a 500,000-token budget consumes 20 sequences (491,520 packed
tokens), or 10 steps across two GPUs. It includes warmup by default; choose the
pilot schedule deliberately. The LR uses linear warmup and cosine decay over the
selected budget; norm LR defaults to one tenth of the sign/scale LR. Norm-only
weight decay defaults to 0.01. Loss is normalized by actual supervised tokens.

`train.jsonl` reports loss, unclipped gradient norm, packed and supervised tokens,
throughput, elapsed time, peak host RSS and peak allocated VRAM per GPU.
`latest.pt` is atomically replaced every N steps and at completion. Allocation
includes all per-layer host checkpoints up front; ensure the planned host RAM is
available. Only a bounded row tile is dense during the native transpose. A full model's
decoded weights and full sequence logits are never resident.

For a crash restart, repeat **the same command/settings**, omit an already
existing `--out` directory, and add:

```sh
--resume /mnt/nvme/qtip_quant/ft-pilot/latest.pt
```

`--stop-after-steps N` permits a clean interruption without changing the schedule
or data identity. A different source, prepared data, replica count, backend,
precision, chunking, seed, budget or optimizer schedule is rejected on resume.
The training path has no dropout/sampling and saves CPU RNG state. GPU arithmetic
determinism still depends on the backend/device. A new 5M run uses a new work
directory and `--max-tokens 5000000`; no training ever wraps back to the start of
the prepared epoch.

Training checkpoint format is now **v2**, reflecting the tied parameter groups.
Earlier v1 checkpoints are rejected: their independent SU optimizer moments
cannot be copied into tied groups without changing training semantics. Native
library SHA256 also participates in the resume signature.

Score a complete exported model on the held-out data:

```sh
python tools/finetune_qtip.py \
  --model /path/to/exported-ft-model --data /mnt/nvme/qtip_quant/heldout_gmcot_24k \
  --work /mnt/nvme/qtip_quant/ft-eval --devices cuda:0,cuda:1 \
  --eval-only --max-tokens 100000000
```

The final record reports assistant NLL/perplexity, including reasoning tokens.
The same packed format supports general-text evaluation with all text tokens
marked as targets. `--eval-only` does not update weights or write training
checkpoints. Export is optional; the small training checkpoint is always saved.
