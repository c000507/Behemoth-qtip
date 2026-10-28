# Qwen3.5 architecture support — design note

Goal: quantizer (`qtip_quant`), runtime (`behemoth_server`/`behemoth_eval`) support the Qwen3.5 family
(dense + MoE, hybrid Gated DeltaNet / gated full attention) alongside the existing Behemoth (Mistral-Large
dense) path, which must not change. Reference implementation: `nmoe` (read-only), in particular
`nmoe/docs/qwen35_model_math.md` (exact math), `src/model/config.cpp`, `src/hip/delta_chunk.hip`,
`src/hip/resident.hip` (`k_dn_delta`), `src/hip/moe_fused.hip`.

## Qwen3.5 architecture summary (verified against Qwen3.5-0.8B and Qwen3.5-122B-A10B configs)

- Hybrid stack: `layer_types` = `linear_attention` (Gated DeltaNet) : `full_attention` = 3:1
  (`full_attention_interval: 4`; full at layer idx 3,7,11,…). `mlp_only_layers` empty.
- Full attention: `num_attention_heads` q heads (8 for 0.8B, 32 for 122B), `num_key_value_heads`
  (2 everywhere), `head_dim` 256, GQA. **Output gate**: `q_proj` out dim = `nq*2*hd`, rows interleaved
  per head `[q(256) | gate(256)]`; `out = attn * sigmoid(gate)`. `attn_output_gate: true`.
- q/k norms: zero-centered RMSNorm per head over the full 256 dims (Qwen3_5RMSNorm:
  `y = x_f32*rsqrt(mean(x²)+eps)*(1+w)`).
- Partial RoPE: `partial_rotary_factor 0.25` → rotary_dim 64 of 256, rotate_half convention on the
  64-dim slice, `rope_theta 1e7`. mrope sections collapse to plain positions for text-only.
- Linear attention (Gated DeltaNet): `linear_num_key_heads` (16), `linear_key_head_dim` (128),
  `linear_num_value_heads` (16 for 0.8B, 64 for 122B), `linear_value_head_dim` (128),
  `linear_conv_kernel_dim` 4. `key_dim = nk*kd`, `value_dim = nv*vd`, `conv_dim = 2*key_dim + value_dim`.
- RMSNorm eps 1e-6, zero-centered weights for all standard norms; `linear_attn.norm` is the gated
  per-v-head RMSNorm (plain weight near 1) `y = rmsnorm(x)*w*silu(z_head)`.
- Dense MLP (small models): SwiGLU `intermediate_size`. MoE (`qwen3_5_moe`): `num_experts` (256),
  `num_experts_per_tok` (8), `moe_intermediate_size`, `shared_expert_intermediate_size`, router
  `mlp.gate` softmax → top-k → renormalize, shared expert gated by `sigmoid(shared_expert_gate(x))`.
- Embeddings tied for small models (`tie_word_embeddings`), vocab 248320, eos 248044.
- HF tensor names (text model prefix `model.layers.N.` for dense, `model.language_model.layers.N.` when
  nested under `text_config`; embed `model.embed_tokens.weight`, final norm `model.norm.weight`).

Forward math (decode, per v-head; see nmoe doc for exact fp32 discipline):

```
xn = rmsnorm_zc(x, input_layernorm)                       # (1+w) convention
qkv = in_proj_qkv(xn)  -> [q key_dim | k key_dim | v value_dim]
z   = in_proj_z(xn)    -> value_dim
b   = in_proj_b(xn); a = in_proj_a(xn) -> nv
conv ring (per channel, K=4): push raw qkv_t; y[c] = sum_j ring[c][j]*w[c][j]; qkv_t = silu(y)
per v-head i (k/q head = i / (nv/nk)):
  q = l2norm(q)*(1/sqrt(kd))       # l2norm = x * rsqrt(SUM(x²)+1e-6), NOT mean
  k = l2norm(k)
  beta = sigmoid(b[i]); g = -exp(A_log[i]) * softplus(a[i] + dt_bias[i])
  S *= exp(g); kv = S k; delta = (v - kv) * beta; S += k ⊗ delta; o = S q
  o = rmsnorm(o) * norm_w[i] * silu(z_head)
y = out_proj(o concat) + x
```

## What changes where

### Quantizer (src/)

- `ModelConfig`: add `arch` (Behemoth | Qwen35Dense | Qwen35Moe), `layer_types` per layer, linear-attn
  dims, `rot_dim`, `attn_output_gate`, MoE fields. `load_model_config` unwraps `text_config`, detects
  `model_type` `qwen3_5`/`qwen3_5_moe`; Behemoth path unchanged.
- Weight reading: `read_layer` dispatches per arch. Qwen3.5 checkpoints are BF16 (or FP8 for the 122B) —
  read BF16 tensors into fp32 host buffers (`LinearIn::w_f32` path of `quantize_group`, already
  supported); norms `ln_in/ln_post` bf16; extra small tensors kept bf16/f32 unquantized: `conv1d.weight`
  `[conv_dim,1,4]`, `A_log[nv]`, `dt_bias[nv]` f32, `linear_attn.norm.weight[vd]`, `q_norm/k_norm[hd]`,
  router + `e_score_correction_bias` + `shared_expert_gate` (MoE), all copied verbatim to the output.
- `LayerRunner::forward_batch`: per-layer-type dispatch.
  - Full attention: existing GEMM attention path generalized (hd, nq, nkv from config) + zero-centered
    q/k norm kernel + partial RoPE kernel + output gate `*sigmoid(gate)` before `o_proj`.
  - DeltaNet: per-token recurrent kernel over all sequences in the batch (grid `seqs × nv`): conv ring
    step, l2norm, beta/g decay, delta-rule state update (fp32 S per head), gated RMSNorm. Loop over T
    positions; state buffers `[seqs][nv][kd*vd]` fp32 + conv ring `[seqs][conv_dim][K]` f32.
  - MoE: router (bf16 GEMV) → top-k softmax renorm → per-expert activation buffers; Hessians per expert
    (gateup/down per expert; only tokens routed to the expert contribute). Shared expert like dense MLP.
- Hessian groups (per layer, same 4-slot `HessAcc` layout `qkv/o/gateup/down`):
  - linear layer: `qkv` = {in_proj_qkv, in_proj_z, in_proj_b, in_proj_a} (input = xn, dim hidden),
    `o` = {out_proj} (input dim `nv*vd`), `gateup`/`down` as dense (or per-expert + shared for MoE).
  - full layer: `qkv` = {q_proj (m=nq*2*hd incl gate rows), k_proj, v_proj}, `o` = {o_proj},
    `gateup`/`down` as dense.
- Quantized output keeps **original HF tensor names** under `model.layers.N.…` (+ `.trellis/.SU/.SV/…`
  suffixes exactly as today), so the runtime maps names 1:1. Small tensors copied unquantized.
  `assemble()` for Qwen3.5 keeps the original `model_type`/`layer_types` config (drops
  `quantization_config`), adds `quip_params`/`qtip_quant`, copies tokenizer files, handles
  `tie_word_embeddings` (no lm_head tensor written; runtime ties).
- Which linears get QTIP (2-bit HYB): in_proj_qkv/z/b/a, out_proj, q/k/v/o_proj, mlp gate/up/down
  (dense), experts + shared expert (MoE). NOT quantized: router gate, e_score_correction_bias,
  shared_expert_gate, conv1d, A_log, dt_bias, all norms, embeddings (embed kept bf16 as today).

### Runtime (runtime/)

- `checkpoint.hpp/cpp`: `ModelConfig` reads real values (no whitelist for qwen3.5; existing Mistral
  validation unchanged for llama/mistral), incl. layer_types, linear dims, MoE, tie_word_embeddings,
  eos. New `Arch` field selects forward path.
- Model/TP: Qwen3.5 initially **single-GPU** (0.8B 2-bit ≈ 0.3 GB weights; 122B 2-bit ≈ 31 GB fits one
  32 GB R9700 tightly; TP split of DeltaNet heads / attention heads / experts is the follow-up).
  Buffers sized from config (no 12288/88 magic numbers on the qwen path). Existing Behemoth TP=2 path
  untouched.
- Attention (full layers): new kernel file for Qwen: bf16 KV cache (not FP8-quantized initially),
  head_dim 256, GQA 8/2 (generic over config), partial RoPE (rot 64) + zero-centered q/k norm,
  sigmoid output gate. Decode + prefill variants (correctness first: scalar per (seq,head) tile kernels,
  WMMA later).
- DeltaNet: recurrent decode kernel (port of nmoe `k_dn_delta`, one block per v-head, fp32 S) and
  prefill via the same kernel looping over positions (correctness first; chunked delta-rule
  (`delta_chunk.hip`) port later if time). Per-sequence state: conv ring f32 `[conv_dim][K]` + S f32
  `[nv][kd*vd]` per linear layer per sequence, managed next to the KV page pool (separate pool; freed on
  sequence end). **Prefix cache disabled for Qwen3.5** (delta state snapshotting is the follow-up).
- MoE: router softmax top-8 renorm (fp32), routed experts as bf16 GEMV per active token initially,
  shared expert with sigmoid gate. (Quantized experts via the same QTIP GEMV as dense once Hessians
  land.)
- Tokenizer: Qwen3.5 ships `vocab.json` + `merges.txt` (GPT2 byte-level BPE), no `tokenizer.model`.
  New BPE tokenizer backend; chat template ChatML (`<|im_start|>…`) from `chat_template.jinja`-style
  hardcode for v1. EOS 248044.
- eval: perplexity/greedy vs CPU reference and vs bf16 (unquantized) path.

### Trainer (training/) — optional, last

DeltaNet backward port from nmoe `src/train/ops_qwen.hip` (chunked delta-rule backward, gated norm
backward, conv backward, l2norm backward, router backward) — only if the quantizer/runtime milestones
are done and stable.

## TP plan (when Qwen3.5 TP=2 lands)

- DeltaNet: split v-heads across GPUs (nv even); in_proj_qkvz rows, out_proj cols, conv channels, S
  states follow the head split; partial `out_proj` results all-reduced like the existing o_proj halves.
- Full attention: split q heads (and kv heads if >2); KV pages per rank own their kv heads.
- MoE: experts split evenly across ranks (128/128), router replicated, activation all-gather or
  P2P per top-k pair (nmoe streams experts instead; for decode, expert-split with token gather is
  simpler here).

## State cache design

- Per sequence slot: `[n_lin_layers]` × (conv ring `[conv_dim][K]` f32, S `[nv][kd*vd]` f32). 0.8B:
  18 layers × (8192·4·4 B + 16·128·128·4 B) ≈ 18 × (128 KB + 1 MB) ≈ 20 MB per sequence. 122B:
  36 × (49152·16 B + 64·128·128·4 B) = 36 × (768 KB + 4 MB) ≈ 172 MB per sequence — fine for a few
  concurrent sequences, paged/allocated from a pool on first token, freed on finish/preempt-restore.
- Interaction with preemption: delta state saved to host with the sequence's other state (KV pages
  already round-trip via host tiers).
- Prefix cache: disabled for Qwen3.5 initially (`--prefix-cache` ignored with a warning), because a
  KV-page hit is not enough to resume a DeltaNet stack (needs S+conv snapshot at the same position).
  Follow-up: snapshot S+conv at prefix boundaries (nmoe keeps 4 checkpoint slots; same idea).

## Test plan

- Unit: config parse (0.8B + 122B configs), tensor-name mapping, CPU reference forward of a full layer
  stack vs nmoe logits (0.8B, short prompt, bf16), quantize→dequantize round trip of a tiny linear.
- Quantizer: 0.8B Hessians + 2-bit QTIP quantize; ppl (`behemoth_eval --ppl`) vs bf16 path; expect
  2-bit HYB degradation similar to Behemoth numbers in QTIP_QUANT_REPORT.md.
- Runtime: greedy short prompts vs CPU reference tokens; ppl parity quantized runtime vs quantizer-side
  proxy; MoE variant config-only smoke (clear error until implemented).

## MoE implementation (done)

- Routing (quantizer kernels `k_q_route*` / runtime `q_moe_route*`): softmax → top-k (deterministic,
  ties to the lowest index) → renormalise, one block per token; a second pass + single-block scan
  builds compact per-expert token lists (atomicAdd; Hessian GEMMs are order-independent).
  Matches the CPU reference and nmoe `k_router_topk` bit-for-bit in the selection rule.
- **Per-expert Hessians**: each routed expert accumulates `Σ_t w_t² x_t x_tᵀ` for gateup and the
  same-weighted activation Hessian for down, where `w_t` is the renormalised routing weight. Rationale:
  the expert's contribution to the residual is `w · E(x)`, so a weight perturbation ΔW costs
  `w·ΔW·x` — the Hessian is the input covariance under the w²-weighted routing distribution.
  Each expert is normalised by its own `Σ w²` (an average xxᵀ, scale independent of routing mass).
- **Starved experts** (< `--moe-min-tok` (default 64) routed calibration tokens) fall back to the
  pooled all-token Hessian (= the shared expert's), which is the natural flatter input distribution;
  QTIP degrades gracefully on it. The fallback is logged and counted in `quant_log.jsonl`
  (`experts.starved_pooled`).
- Shared expert uses the pooled (unweighted) Hessians — HessAcc slots 2/3; routed experts live in
  `MoeHessAcc` (gate/down per expert + per-expert `Σw²` and token counts). Hessian file: safetensors
  with `qkv`/`o`/`gateup`/`down` (pooled), `stats` F32 [2E] and `e{i}.gateup`/`e{i}.down`,
  metadata `format=moe`.
- Quantized output: `mlp.experts.{e}.{gate,up,down}_proj` and `mlp.shared_expert.*` as QTIP;
  `mlp.gate.weight` (router), `mlp.gate.e_score_correction_bias`, `mlp.shared_expert_gate.weight`
  stay unquantised (copied verbatim). The softmax router ignores `e_score_correction_bias`
  (matches nmoe: the bias only affects sigmoid/noaux_tc routers), it is stored for completeness.
- Runtime `moe_block`: router bf16 GEMM → routing kernels → per expert gather + (trellis
  transform→GEMV→hadU→SV or plain bf16 GEMM) → weighted scatter-add into an fp32 accumulator;
  shared expert with `sigmoid(shared_expert_gate)`; state += accumulator. Untied `lm_head.weight`
  supported. Linears load either QTIP trellis or plain bf16 weights (unquantized checkpoints run
  the identical surrounding code — bf16-vs-quantized comparison).

## Qwen3.5-122B-A10B plan (not executed)

Checkpoint facts (read from `/mnt/evo/Qwen3.5-122B-A10B-FP8`, config + tensor headers):
text model H=3072, 48 layers (36 linear / 12 full), full attention nq32/nkv2/hd256 (A=8192),
DeltaNet nk16/kd128, nv64/vd128 (key 2048, value 8192, conv 12288), vocab 248320,
`tie_word_embeddings=false`, MoE per layer: 256 experts top-8, moe_inter 1024, shared_inter 1024,
sigmoid-gated shared expert, softmax router (no `scoring_func` override). FP8-E4M3 weights with
**128×128 block scales** `weight_scale_inv` BF16 [out/128][in/128] (e.g. [8,24] next to [1024,3072]);
`read_w16` dequantises both layouts (per-row `weight_scale` and block `weight_scale_inv`, scale
multiplies on dequant — transformers FP8 convention). Router/conv/A_log/dt_bias/gates are BF16 in
the checkpoint (`modules_to_not_convert`).

Memory/time (2× R9700 32 GB, 377 GB host RAM, ~27 GB free on /mnt/nvme):

| item | size |
|---|---|
| per-expert Hessians (F32, GPU + hess file) | 256×(3072²+1024²)×4 ≈ **10.7 GB/layer** |
| expert weights bf16 on GPU (collection) | 256×3×1024×3072×2 ≈ 4.8 GB/layer |
| MoE collection peak per GPU | ≈ 17 GB (Hessians + weights + activations) ✓ |
| host peak at deposit | partials 2×10.7 + combined 10.7 + writer copy 10.7 ≈ 43 GB ✓ |
| hess disk peak | backlog × 10.7 GB → run with **`--backlog 1`**, delete after quantize ✓ |
| quantized weights (2-bit, all linears) | ≈ 30 GB + embed/lm_head bf16 3.05 GB ≈ **33 GB total** |

- Collection: 4M calibration tokens ≈ 38 PFLOP forward + ~equal Hessian GEMM work → ~10–20 min GPU
  total across both GPUs (experts see topk-8 × the tokens, all GEMM-bound).
- Quantize: 2 attn groups + 512 expert groups + 2 shared groups per layer; fp64 LDL (3072²/1024²)
  ~3–6 s each → **~40–60 min per layer** (GPU Viterbi + CPU LDL overlap), 48 layers ≈ 1.5–2 days on
  two GPUs in parallel. The per-expert groups are embarrassingly parallel if more CPUs help.
- Runtime: 33 GB of weights does not leave KV room on one 32 GB GPU → serve with TP=2 (experts
  sharded 128/128, router replicated; see TP plan above); single-GPU eval works with small context
  and lm_head streaming. Untied lm_head is a bf16 GEMM (`lm_head.weight` stays bf16).
- Do NOT write outputs to /mnt/nvme beyond the hess backlog: quantized model ≈ 33 GB — target a
  different disk.
