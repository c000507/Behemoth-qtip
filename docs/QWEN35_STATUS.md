# Qwen3.5 support — status

Branch `qwen35`. Reference implementation: `nmoe` (read-only). Test model: Qwen3.5-0.8B (dense,
`/mnt/nvme/models/Qwen3.5-0.8B` on .229, BF16 safetensors).

## What works (tested)

### Quantizer (`qtip_quant`)

- **Config parsing** (`src/qwen35.cpp`): `qwen3_5` / `qwen3_5_moe` config.json incl. nested
  `text_config`, `layer_types` / `full_attention_interval`, linear-attn dims, `rope_parameters`
  (partial_rotary_factor 0.25, theta 1e7), MoE fields, tie_word_embeddings. Unit test in
  `tests/test_cpu.cpp` (`test_qwen_config`, runs on every `qtip_tests_cpu`).
- **Checkpoint reader** (`read_qwen_layer`): all BF16 tensors of dense 0.8B + FP8 fallback in BOTH
  layouts (per-row `weight_scale`, 128×128 block `weight_scale_inv` — the 122B-A10B-FP8 layout),
  BF16/F32 `dt_bias`/`A_log`/gated-norm dtypes, prefix autodetect (`model.language_model.`).
- **CPU fp32 reference forward** (`qwen_cpu_forward`): full stack incl. Gated DeltaNet, gated full
  attention, **and MoE** (router softmax→top-k→renorm, routed experts, sigmoid-gated shared expert).
  - Parity vs HF transformers 5.17 fp32, dense 0.8B (10-token prompt): logits max|diff| 0.063
    (rel ~0.4%), argmax 10/10; per-layer residual max|diff| <= 0.006.
  - Parity vs HF transformers on the **tiny synthetic MoE** checkpoint: logits max|diff| 1.5e-5
    (rel 3e-6), argmax 16/16.
- **GPU layer runner** (`src/qwen_runner.cpp`, `src/qwen_kernels.hip`): both layer types + dense MLP
  + **MoE** (routing kernels, per-expert w²-weighted Hessians, pooled/shared Hessians, expert
  forward for chaining). (GPU parity test pending GPU availability — see open issues.)
- **Quantize pipeline**: per-layer-type Hessian groups, QTIP 2-bit HYB output with original HF
  tensor names; **MoE quantize**: per-expert groups with starved-expert pooled fallback
  (`--moe-min-tok`), shared expert, router + `e_score_correction_bias` + `shared_expert_gate`
  copied verbatim; MoE Hessian file format (`format=moe`). Untied lm_head written by assemble.
- Behemoth (Mistral) path untouched: same binary, dispatch by `model_type`.

### Runtime

- **`QwenModel`** (`runtime/src/qwen_model.cpp`): single-GPU correctness-first inference. Trellis
  GEMV/WMMA kernels per linear, generic K×P Hadamard, fused input transform, gated attention
  kernels (bf16 KV cache, q/k norm + partial RoPE + sigmoid output gate), per-token recurrent
  delta prefill/decode with persistent fp32 state, **untied lm_head**, **bf16-linear fallback**
  (unquantized checkpoints: linears load plain `weight` bf16 and run dense GEMMs — enables
  bf16-vs-quantized comparison of the same path), **MoE block** (router, top-k, per-expert
  trellis GEMV or bf16 GEMM on compact expert token lists, sigmoid-gated shared expert).
  `qwen_eval` tool: `--ppl`, `--greedy`.
- Builds clean (`build`, `build-runtime`, gfx1201). Runtime validation pending GPUs.

### Baselines recorded

- CPU bf16 ppl (dense 0.8B, our reference forward, 512-token windows): **6.2137** over 4088
  predictions of `/tmp/qwen_eval.i32` (re-measured on the current eval file; the older 5.7415 was a
  slightly different eval set).
- **Quantized-reference ppl (0.8B QTIP-2bit, `qwen_qt_ref --ppl`, double-precision trellis decode):
  9.3136** over 3584 predictions of the same corpus — the first 0.8B quantization used only 202k
  calibration tokens; re-quantization with 4M tokens pending GPU. This separates quantization
  quality from runtime correctness: the GPU runtime should MATCH ~9.31 on the old checkpoint.
- Bug found on the way: the embed/norm tensor scan (assemble + runtime loader) picked
  `mtp.norm.weight` over `model.language_model.norm.weight` (last-match-wins) — every earlier
  runtime eval applied the WRONG final norm. Fixed (explicit candidates); 0.8B model-embed
  regenerated.
- Tiny MoE (`~/qwen_tiny_moe` on .229, seed 777): CPU-ref logits `/tmp/tiny_logits.bin` (16 tokens,
  max|diff| vs HF 1.5e-5), argmax `386 1919 1389 457 75 561 165 926 1058 1413 718 772 186 642 666 884`;
  CPU-ref ppl over 1022 predictions of `/tmp/tiny_ppl.bin`: **3437.42** (random weights).
- HF transformers logits for the dense-0.8B fox prompt in `/tmp/hf_logits.npy` (.229).

## Pending / open issues

1. **GPU slots**: since the 123B finetune ended, the other agent has saturated both GPUs
   continuously (behemoth_server, 123B qtip_quant, behemoth_eval, qtip_reconstruct; a 2 h
   gap-watcher found no idle window). All GPU validation is scripted and ready in
   `tools/qwen35_gpu_validate.sh` (steps 1-6, one GPU process at a time; also
   `/tmp/qwen35_gap_run.sh <cmd>` on .229 waits for an amd-smi gap and runs it). Queued:
   (a) runtime ppl/greedy of the existing 0.8B QTIP checkpoint with the fixed final norm vs the
   quantized reference (expect ≈ 9.31) — validates/clears the suspected "layer-6 DeltaNet runtime
   bug" (evidence now points at quantization drift + the final-norm bug, not the delta kernels:
   the layer test showed layer-0 delta out rel 3.2e-3 after the host-pointer fixes);
   (b) re-quantize 0.8B with 4M calibration tokens (`/tmp/qwen_calib.i32`, 2048x2048 on .229);
   (c) qtref + runtime ppl of the new checkpoint vs the 6.2137 bf16 baseline (same eval file);
   (d) tiny-MoE Hessian collection + quantize + GPU-runtime vs CPU-ref parity (bf16 and quantized);
   (e) `qtip_tests_gpu` parity suite.
2. **122B-A10B**: full plan (memory/time/disk) in docs/QWEN35.md; FP8 block-scale reading
   implemented; run NOT executed per task instructions.
3. Runtime is single-GPU / single-sequence (eval); no scheduler/server integration, prefix cache
   disabled by design for Qwen3.5 (delta state cannot resume from KV pages alone).
4. Trainer: rejects Qwen3.5 with a clear "not supported in the trainer yet" error (config/loader
   awareness only, per the task's minimum).

## Commits

- `5542ab8` design note docs/QWEN35.md
- `b6812e0` + fixes: qwen config/reader/CPU ref/GPU runner/main integration
- runtime: qwen kernels + model + eval tool (see `git log qwen35`)
- `6337a50` MoE: per-expert Hessians, quantize path, runtime MoE + bf16 fallback, tiny-MoE
  generator, HF-validated CPU MoE reference, trainer guard, 122B FP8 block-scale reading + plan
