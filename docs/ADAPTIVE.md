# Adaptive mixed-format quantization (`--mode adaptive`)

`qtip_quant --mode adaptive` produces a mixed-format Behemoth checkpoint: every transformer layer is
quantized with the existing QTIP 2-bit HYB pipeline (Hessians, RHT, block LDL, LDLQ + Viterbi), an
output-quality metric is measured against the original layer, and layers whose metric exceeds
`--kl-threshold` are re-quantized to NVFP4 (block-scaled FP4 E2M1, the same tensor format as the GLM
converter) instead. Per-layer decisions land in `adaptive_report.json` and
`config.json:qtip_quant.layer_formats` so loaders can tell the formats apart.

```
qtip_quant --mode adaptive --tokens calibration/tokens.i32 \
  --out /mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-ADAPTIVE --work /mnt/nvme/qtip_quant/work-adaptive \
  --gpus 0,1 --kl-threshold 0.05
```

| flag | default | meaning |
|---|---|---|
| `--kl-threshold X` | 0.05 | metric value above which the layer is written as NVFP4 |
| `--kl-metric lmkl\|mse` | lmkl | layer quality metric (see below) |
| `--kl-tokens N` | 512 | probe document length (tokens) for the metric |
| `--adaptive-next-input original\|quantized` | original | source of the next layer's input activations |

All other flags (`--kv-format`, `--sigma-reg`, `--vit-batch`, `--layers`, resume behavior) are shared
with the classic pipeline. K/V projections follow `--kv-format` exactly as before (default: source FP8
preserved, in both QTIP and NVFP4 layers).

## Per-layer flow

For each layer L, strictly sequentially (chains on all GPUs run in lockstep; layer L+1 is only
collected after L was decided):

1. **QTIP quantization** of q/o/gate/up/down (plus k/v with `--kv-format qtip`) with the unchanged
   existing pipeline on the calibration activations.
2. **Metric vs the original layer** on a probe document: the first `--kl-tokens` calibration tokens,
   embedded once and advanced through the layers (see "chaining" below). The original FP8→bf16 layer
   and the candidate quantized layer run on the *same* probe input; the metric compares their outputs.
3. **Decision**: if `metric(QTIP) > threshold`, the QTIP result is discarded and the layer is
   quantized to NVFP4 from the original weights (nearest-value reconstruction, no calibration), the
   metric is re-measured and logged for the NVFP4 layer as well. The layer then uses NVFP4
   regardless of whether its metric is below the QTIP one (NVFP4 is a capacity fallback, not a
   per-layer argmin).
4. **Record**: the format, `kl_qtip`, `kl_nvfp4` (or `mse_*` with `--kl-metric mse`), the threshold,
   and the metric name are written to `<out>/adaptive_report.json` (rewritten atomically per layer),
   `<work>/quant_log.jsonl`, `<work>/progress.json` (`format` per layer) and, at assemble time, to
   `config.json`.

## Metric definitions

There is no LM head inside a transformer layer, so both metrics are defined as *sandwich* measures:
one candidate layer with all remaining layers (`L+1..87`) and the final norm + `lm_head` kept
**original**.

- `lmkl` (**default**, `--kl-metric lmkl`): mean token-level KL divergence of the next-token
  distribution,
  `KL(p_orig || p_quant) = Σ_t Σ_v p_orig(t,v) · (log p_orig(t,v) − log p_quant(t,v)) / T`, where
  `p_orig`/`p_quant` are softmaxes of the logits obtained by pushing the original/quantized layer-L
  output through the original layers `L+1..87`, the final RMSNorm and `lm_head`, on the T-token probe
  document (all T positions; each position's distribution predicts the next token). Units: nats/token.
  `--kl-threshold 0.05` is a starting point, not a tuned value; inspect `adaptive_report.json` and
  re-run with a threshold placed inside the observed KL distribution. The KL is computed in double
  precision from fp32 logits.
- `mse` (`--kl-metric mse`): normalized MSE of the layer output itself,
  `Σ ||y_quant − y_orig||² / Σ ||y_orig||²` over the probe tokens (bf16 values compared as fp32). No
  forward through the remaining layers; use when the lm_head pass is too expensive.

The preferred metric is `lmkl` (it measures what matters: next-token damage of the layer under
original context); `mse` is the documented fallback.

## Chaining (`--adaptive-next-input`)

- `original` (**default**, matches the classic pipeline): the calibration chain, the Hessians, the
  probe and the KL context always come from the *original* model. Each layer's quality is measured in
  isolation; errors do not compound across layers in the measurement. Resume works as usual.
- `quantized`: after each decision the chain activations are recomputed through the *chosen* layer
  (QTIP hatW or NVFP4 dequantized, bf16), so layer L+1's Hessians, probe input and KL context see the
  actual quantized model state (progressive/sequential quantization). This costs one extra activation
  snapshot (~2x chain host RAM) and one extra layer forward per layer, and **resume is not
  supported** in this mode (the quantized chain state of finished layers cannot be reconstructed
  without re-quantizing); the tool refuses to start if the work directory already contains finished
  layers.

## Output format

The HF directory layout is unchanged (`model-layer-XXXXX.safetensors` per layer); only the tensors
inside a layer differ by format:

- QTIP layers: `trellis/SU/SV/tlut/rcp/tp_rank` per linear, exactly as `--mode run` writes them.
- NVFP4 layers: `<linear>.weight_packed` (U8 `[m, n/2]`, two E2M1 values per byte, low nibble first),
  `<linear>.weight_scale` (F8_E4M3 `[m, n/16]`, one scale per 16 consecutive input channels of a row)
  and `<linear>.weight_global_scale` (F32 scalar) — the same format `glm53 quantize` writes for GLM
  MLPs. NVFP4 quantization starts from the exactly-decoded original FP8 weights.
- FP8-preserved K/V (default `--kv-format fp8`): `weight` + `weight_scale` as before, in both formats.

`config.json` gains `qtip_quant.layer_formats` (a per-layer list, e.g.
`["qtip", "nvfp4", ...]`) and `qtip_quant.adaptive` (metric/threshold/kl_tokens/next_input).
`adaptive_report.json` in the output directory records one entry per layer:
`{layer, format, adaptive: {metric, threshold, kl_qtip, kl_nvfp4, ...}}`.

## Runtime support

- `runtime/` CUDA engine (`behemoth_cuda`, `behemoth_cuda_server`): **supported**. Each linear is
  loaded by probing its tensors (`trellis` → QTIP, `weight_packed` → NVFP4 shared GLM gemv, `weight`
  → FP8 K/V), so mixed checkpoints run per-layer without configuration. Validated by
  `runtime/tests/cuda_tests.cpp` (loader + NVFP4 gemv vs a bf16 reference).
- ROCm 2-GPU engine (`behemoth_server`, runtime/src/model.cpp): **follow-up**. Exact gaps:
  `load_qtip_layer` assumes every linear is trellis-or-FP8 (SU/tlut sharing checks and the stacked
  rotated-input matmuls do not represent NVFP4 linears); supporting them needs per-linear dispatch in
  the decode/prefill projection paths (dense bf16 dequant path like `qtip_dense` is the simplest) and
  the ~4x larger per-layer payload accounted for in the memory plan.
- `behemoth::qtip::reference_forward` / `reference_block` (CPU reference): trellis/FP8 only; NVFP4
  HostLinears are rejected implicitly (empty trellis) — trivial to add by dequantizing when needed.

## Validation status

- `qtip_adaptive_tests` (built with both backends): a synthetic 3-layer FP8 checkpoint is quantized
  with thresholds/metrics that force each branch — all-QTIP, all-NVFP4 (KL fallback), MSE metric with
  quantized-input chaining — checking layer tensors, report/config contents and NVFP4 decode
  fidelity. Passes on the CUDA build (RTX A4000, sm_86).
- Full-size Behemoth run: **not executed** (no model weights on this box). VRAM note: quantization
  temporaries and the evaluation runners alternate on GPU 0 (runners are freed before the next
  layer's QTIP pass); peak is roughly the classic pipeline's peak plus the lm_head
  (vocab x hidden bf16) during evaluation.

## Llama 3.3 70B (BF16 source, 128256 vocab)

Adaptive mode works with BF16 sources and the Llama 3.3 geometry (dims from `config.json`, llama3
RoPE scaling in the probe/chain forward). BF16 linears are used directly for both the QTIP and the NVFP4
candidates (no FP8 K/V preservation; `kv_format` is `qtip`). Memory: the metric probe's lm_head GEMM is tiled
over vocab rows (`logit_tile`, default about 256 MB of fp32 logits per tile) instead of materialising
`T x 128256` logits, so the device holds the bf16 head (128256 x 8192 x 2 B ~ 2.1 GB) plus the two
evaluation runners. Verified only on the synthetic Llama-geometry checkpoint (`qtip_llama_tests`: tiled and
untiled metrics equal, all-QTIP and all-NVFP4 outputs load in the CUDA runtime); a full-size run and its VRAM
peak on a 16 GB card are **not** verified.
