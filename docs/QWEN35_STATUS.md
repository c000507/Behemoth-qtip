# Qwen3.5 support — status

Branch `qwen35`. Reference implementation: `nmoe` (read-only). Test model: Qwen3.5-0.8B (dense,
`/mnt/nvme/models/Qwen3.5-0.8B` on .229, BF16 safetensors).

## What works (tested)

### Quantizer (`qtip_quant`)

- **Config parsing** (`src/qwen35.cpp`): `qwen3_5` / `qwen3_5_moe` config.json incl. nested
  `text_config`, `layer_types` / `full_attention_interval`, linear-attn dims, `rope_parameters`
  (partial_rotary_factor 0.25, theta 1e7), MoE fields, tie_word_embeddings. Unit test in
  `tests/test_cpu.cpp` (`test_qwen_config`, runs on every `qtip_tests_cpu`).
- **Checkpoint reader** (`read_qwen_layer`): all BF16 tensors of dense 0.8B + FP8+scale fallback,
  BF16/F32 `dt_bias`/`A_log`/gated-norm dtypes, prefix autodetect (`model.language_model.`).
- **CPU fp32 reference forward** (`qwen_cpu_forward`, `qwen_cpu_layer`): full stack incl. Gated
  DeltaNet (conv ring, l2norm-in-sum q/k, decay/beta gates, delta rule, gated RMSNorm) and gated
  full attention (zero-centred q/k norms, partial RoPE rotate_half on 64/256 dims, output gate).
  **Parity vs HF transformers 5.17 fp32** (10-token prompt, per-layer hooks):
  logits max|diff| 0.063 (scale ~17, rel ~0.4%), argmax 10/10; per-layer residual max|diff| <= 0.006.
  Bug found on the way: attention scores originally dotted q·q instead of q·k (fixed, bisected via
  HF layer hooks + implied-attention-weight least squares).
- **GPU layer runner** (`src/qwen_runner.cpp`, `src/qwen_kernels.hip`): per-token recurrent delta
  step kernel + gated-attention GEMM path for Hessian collection, both layer types + dense MLP.
  (GPU parity test written; pending GPU availability — see open issues.)
- **Quantize pipeline**: per-layer-type Hessian groups (linear: qkv+z+b+a share the normed-hidden
  Hessian; out_proj group over value_dim; gateup/down; full: q/k/v, o, gateup, down), QTIP 2-bit
  HYB output with original HF tensor names, unquantized small tensors (conv1d, A_log, dt_bias,
  norms, q/k norms) copied verbatim, assemble keeps the qwen3_5 config + tokenizer files, tied
  embeddings (no lm_head tensor).
- Behemoth (Mistral) path untouched: same binary, dispatch by `model_type`.

### Runtime

- **`QwenModel`** (`runtime/src/qwen_model.cpp`): single-GPU correctness-first inference. Trellis
  GEMV/WMMA kernels per linear (rotate -> matmul -> slice-sum -> hadU -> SV epilogue), generic
  K×P Hadamard (K ∈ {1,12,20,28}, so 3584 = 28×128 works, unlike the stock 1024-chunk kernels),
  fused input transform (zero-centred norm + SU + hadUt), gated attention kernels (bf16 KV cache,
  q/k norm + partial RoPE + sigmoid output gate), per-token recurrent delta prefill/decode with
  persistent fp32 state, tied lm_head via bf16 GEMM. `qwen_eval` tool: `--ppl`, `--greedy`.
- Builds clean (`build-runtime`, gfx1201). Runtime validation pending a quantized checkpoint
  (needs the quantizer run, which needs the GPUs).

### Baselines recorded

- CPU bf16 ppl (our reference forward, 512-token windows, 4088 predictions, code-heavy corpus):
  **5.7415** (`qwen35_ref ... --ppl`).
- HF transformers logits for "The quick brown fox jumps over the lazy dog." in `/tmp/hf_logits.npy`.

## Pending / open issues

1. **GPU slots**: the other agent holds both GPUs (trainer + 123B quantization). Blocked on:
   GPU parity test (`QWEN35_MODEL=... qtip_tests_gpu`), 0.8B Hessian collection + quantization,
   runtime ppl/greedy of the quantized model.
2. **MoE**: config parsing + checkpoint reading implemented; Hessian collection (per-expert) and
   quantize path reject MoE with a clear error (`MoE quantization not implemented yet`), same in
   the runtime loader. Router softmax→top-8→renorm + sigmoid-gated shared expert implemented in
   the CPU reference only.
3. Runtime is single-GPU / single-sequence (eval); no scheduler/server integration, prefix cache
   disabled by design for Qwen3.5 (delta state cannot resume from KV pages alone).
4. Untied lm_head (`tie_word_embeddings=false`, e.g. 122B) rejected by the runtime loader for now.
5. Trainer: untouched (optional per task).

## Commits

- `5542ab8` design note docs/QWEN35.md
- `b6812e0` + fixes: qwen config/reader/CPU ref/GPU runner/main integration
- runtime: qwen kernels + model + eval tool (see `git log qwen35`)
