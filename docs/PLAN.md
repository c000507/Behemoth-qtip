# Plan: GM-CoT calibration + streaming fine-tune for QTIP Behemoth

Status: native C++20/HIP conversion 2026-10-27. Phase 3 uses standalone training/ executables;
training, data preparation, checkpointing and export have no Python dependency. Shared SU groups
and the native runtime QTIP decoder are preserved.
Real-layer/GPU verification, full-logit parity and phases 1/2/4 remain pending.
See [FINE_TUNING.md](FINE_TUNING.md) for commands, backend contract and current limitations.
Hardware: 192.168.0.229, 2x Radeon AI PRO R9700 (gfx1201, 32 GB), 377 GB RAM, ROCm 10.1, GPUs at DPM level 1.
Time and memory numbers are estimates until the pilot measures them.

## Goal

Recover quality lost by the 2-bit QTIP quantisation of Behemoth-R1-123B-v2 on the user's actual workload
(Game-Master chat with reasoning, "GM-CoT"), without a teacher model and without LoRA. Some drift towards the
GM-CoT style is acceptable.

Two levers:
1. Calibrate the quantiser (Hessians) on GM-CoT at long context instead of generic text.
2. Fine-tune the continuous QTIP parameters of the quantised model with next-token cross-entropy on GM-CoT
   (AdamW). The model format stays unchanged, so the runtime needs no changes.

## Data

- GM-CoT: `/home/charles/train_data/train_cot_final.jsonl` on .229. 731 conversations, 18,183 assistant turns,
  ~72M characters (~18M tokens), median conversation ~28k tokens / 37 messages.
- CoT is in assistant `reasoning_content`; Behemoth's chat template ignores that field, so it is inlined as
  `<think>\n...\n</think>\n\n` + content (legacy inline `<think>` blocks in `content` are stripped).
  Implemented by the native `qtip_prepare` executable.
- Held-out set: 73 conversations (10 %, seed 0), ids recorded in the calibration `.json` files. Never used for
  calibration or training.
- General text share: from the existing calibration mix (OpenWebText + smol-smoltalk).
- Defaults: sequence length 24,576 tokens; mix 80 % GM-CoT / 20 % general. Conversations shorter than the
  window are packed with document masking (no attention across conversation boundaries).

## Phase 0 - prerequisites (in progress / done)

- Baseline QTIP 2-bit model with generic calibration: `/mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-2bit`
  (4.19M tokens, seq 2048).
- QTIP decode support in `behemoth_server` (trellis-decode GEMV/GEMM, RHT/Hadamard, TP design). The trainer
  reuses these kernels.
- Done: GM-CoT calibration files `calib_gmcot_512x8192.i32` and `calib_gmcot_2048x2048.i32` (4.19M tokens each,
  75/25 mix). The legacy generator is retired; new data uses `qtip_prepare`.

## Phase 1 - long-context calibration and re-quant (`qtip_quant`)

1. Attention in `LayerRunner` currently materialises S/P of size 12 x T x T per KV group (~77 GB at 32k).
   Replace it with query-block tiling (e.g. 4,096 query rows per tile) or a flash-style kernel, so memory is
   O(T x block).
2. Process the per-token buffers (qkv, gate/up, y) in row chunks so 24-32k sequences fit in VRAM.
3. Calibration file at 24k from native `qtip_prepare`, with mask/segment-id files
   that the quantizer attention must honour.
4. Parity: tiled attention vs the current implementation on layers 0-1 at seq 2k and 8k (activations and
   Hessians within bf16 noise).
5. Full re-quant from GM-CoT Hessians (~5-6 h on both GPUs).

Output: `Behemoth-R1-123B-v2-QTIP-2bit-gmcot`.

## Phase 2 - evaluation harness (can overlap with phase 1)

- Metrics:
  - perplexity on assistant tokens (incl. `<think>`) of the 73 held-out conversations;
  - repetition: repeated-n-gram rate in 512-token continuations of held-out prompts (greedy and temperature 0.8);
  - perplexity on a general-text sample.
- Implementation: a scoring path through the runtime (prompt log-probs endpoint, or an offline tool that links
  the runtime library).
- Compare: AQLM 1x16 (current production), QTIP generic calibration, QTIP GM-CoT calibration, and the fine-tuned
  model.

## Phase 3 - streaming fine-tuner (new code in this repo)

### Execution model

- All 88 QTIP layers (~31 GB packed) live in pinned host RAM, shared by both GPUs.
- Per GPU: two weight slots and a copy stream (double buffer). While layer i computes in one slot, layer i+1
  loads into the other; events order copy and compute. Backward streams the layers in reverse order with the
  same scheme. One layer is ~350 MB, so ~14 ms over PCIe Gen4 versus ~1 s of compute per 32k sequence: loads
  are hidden.
- Data parallel: each GPU streams the full model over its own sequences. Gradients of the trainable parameters
  (~100 MB) are all-reduced once per step through pinned host memory.
- Activation checkpoints: one per layer (input hidden state, 32k x 12288 bf16 ~805 MB), copied to pinned host
  memory during forward and read back during backward, also double-buffered. ~71 GB per 32k sequence per GPU.
  Everything inside a layer is recomputed from its checkpoint in backward.
- All large device<->host transfers go through reused pinned staging buffers (avoids the amdgpu
  `MES failed to respond to msg=INVALIDATE_TLBS` issue seen with pageable D2H copies).

### Trainable parameters (no LoRA)

- SU (RHT input signs, relaxed): shared by q/k/v and by gate/up, separate for o and down,
  as required by the runtime's stacked projections. These groups are tied during optimization.
- Per linear: SV (relaxed output signs) and Wscale.
- RMSNorm weights (input and post-attention per layer, final norm).
- Frozen: trellis codes, LUT, embeddings, lm_head.
- Total ~25M parameters. Exported in the same QTIP checkpoint format (only these tensors change).

### Kernels

- Forward: the runtime's trellis-decode GEMM + Hadamard transforms.
- Backward:
  - transposed trellis-decode GEMM for input gradients;
  - SU/SV/Wscale gradients as elementwise products/reductions of the rotated activations and gradients;
  - RMSNorm backward, SwiGLU backward, RoPE backward;
  - attention backward, tiled/flash-style, with document masking.
- Loss: fused cross-entropy over the 32,768-token vocabulary in chunks (logits never stored in full); loss
  only on assistant tokens (system and user tokens masked).

### Optimiser and schedule

- AdamW in fp32 on the GPU for all trainable parameters (state ~300 MB). Muon was considered and dropped:
  without LoRA there are no 2D trainable matrices for it to act on.
- Learning rate ~1e-4 (scales/signs), lower for norms; warmup then cosine decay; gradient clipping 1.0; no
  weight decay on signs/scales.
- At most 1 epoch over the GM-CoT training split.
- Checkpoints every N steps (trainable tensors + optimiser state + data position); resumable after crashes or
  power loss.

### Verification before any real run

- Finite-difference gradient checks for every backward kernel on a tiny random model.
- GPU layer forward + backward against the standalone C++ reference and runtime QTIP decoder.
- Streaming forward logits bit-identical to `behemoth_server` on the same model and prompt.
- Overfit a single batch (loss must fall towards zero).

## Phase 4 - runs

1. Pilot: 0.5M tokens (~1-2 h). Measure throughput, memory and loss curve.
2. Main: 5M tokens (~10-12 h). Evaluate with phase 2 against the un-tuned GM-CoT quant.
3. Optional: up to the full epoch (~20M tokens, ~1.5-2 days) only if held-out metrics are still improving.

Output: `Behemoth-R1-123B-v2-QTIP-2bit-gmcot-ft`, served by `behemoth_server` without changes.

## Resources (estimates)

| Item | Estimate |
|---|---|
| Re-quant (phase 1) | 5-6 h on 2 GPUs |
| Fine-tune 5M tokens | 10-12 h on 2 GPUs |
| Fine-tune 20M tokens | 1.5-2 days |
| Host RAM peak | ~31 GB weights + ~142 GB checkpoints (2 GPUs x 32k) + staging, < 250 GB of 377 GB |
| VRAM per GPU | 2 weight slots (~0.7 GB) + one layer's activations/recompute at 32k (~10-15 GB) + attention tiles + optimiser |
| PCIe | ~1-2 GB/s average per GPU, well below link capacity |
| Disk | ~37 GB per quantised model + checkpoints (small) |

## Risks

| Risk | Mitigation |
|---|---|
| Overfitting to GM style | low LR, <= 1 epoch, 20 % general text, held-out evaluation |
| Signs/scales/norms alone recover too little | measure after the 5M run; next options are a trainable lm_head or a full-weight fine-tune before quantisation (needs more host memory) |
| GPU stability on long runs | DPM level 1, pollers, MES-safe pinned transfers, resumable checkpoints |
| Backward bugs | gradient checks, GPU/C++ reference comparisons, single-batch overfit, before real runs |
| Power loss mid-run | checkpoint every N steps; quantiser and trainer both resume |

## Order of work

1. Phase 0 completes (baseline quant done; runtime QTIP support merged).
2. Phase 1 (tiled attention, 24k calibration, re-quant) and phase 2 (evaluation) in parallel.
3. Phase 3 trainer, verified on a tiny model and one real layer.
4. Phase 4 pilot, main run, evaluation; then decide on the optional full epoch.
