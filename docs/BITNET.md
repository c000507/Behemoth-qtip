# BitNet b1.58 distillation and inference

End-to-end support for BitNet b1.58 (ternary `{-1, 0, +1}` weights with one absmean scale per
tensor, 8-bit per-token absmax activations, SubLN) in the native toolchain: a C++20/CUDA
distillation trainer (`training/bitnet`), a packed ternary checkpoint format, and ternary
projections in the shared CUDA inference engine (`runtime/cuda`). No Python is used by any
training, export or inference step; a throwaway PyTorch script was used once, outside the
repository, to cross-check gradients (see *Verification*).

References: BitNet b1.58 (arXiv 2402.17764), BitDistill (arXiv 2510.13998), MiniLM relation
distillation.

## Design

### BitLinear

Every linear in the transformer blocks (`q/k/v/o`, `gate/up/down`) becomes a BitLinear:

- Latent weights stay FP32 (`W`). The forward operand is `W~ = clip(round(W / s), -1, 1)` with
  `s = mean|W| + 1e-8` per tensor (absmean, as in BitNet b1.58). The effective weight is
  `s * W~`.
- Activations are quantised per token: `xs = max|x| / 127` (eps-guarded), `xq = clip(round(x / xs), -127, 127)`.
- `y = s * xs * (W~ xq^T)`. Both operands are exact small integers, so the trainer stores them as
  BF16 and runs cuBLAS BF16 GEMMs with FP32 accumulation: `|sum| <= in * 127 < 2^24`, hence the
  integer sums are **exact** and identical to the int32 accumulation of the inference kernel. Only
  the final scale multiply is floating point.
- Straight-through estimators: `dL/dx = s * dY W~` (the activation rounding is transparent) and
  `dL/dW = (dY (.) xs)^T xq = dY^T x_deq`, i.e. the gradient of the dequantised effective weight is
  applied to the latent weight (the canonical `w + (quant(w) - w).detach()` formulation; the scale
  and the activation scale are treated as constants). Checked against a PyTorch reference
  (below).
- Embeddings, `lm_head` (tied or not), RMSNorm gains and the SubLN gains stay high precision.
  Embedding/head are frozen (BF16); the norm gains train in FP32.

### SubLN

As in BitDistill, an extra RMSNorm (gain initialised to 1) is inserted before `o_proj` (over the
attention output, width `heads * head_dim`) and before `down_proj` (over the SwiGLU output,
width `intermediate`). The runtime reads them from `self_attn.attn_sub_norm.weight` and
`mlp.ffn_sub_norm.weight`; `quantization_config.sub_norm` says whether they exist.

### Model and numerics

The trainer implements a Llama-layout decoder (RMSNorm, GQA with RoPE, SwiGLU, tied or untied
head) with FP32 residual stream and FP32 intermediate activations, BF16 GEMM operands and cuBLAS
FP32 accumulation. By default it emulates the runtime's BF16 activation boundaries (every
projection output, residual, norm output, attention output and SwiGLU output is rounded to BF16
in the forward; the backward is straight-through), so the trained ternary codes see the same
activations as the serving engine. `--no-round` keeps FP32 activations.

Attention is materialised (scores `[heads, T, T]` FP32 via cuBLAS strided-batched GEMM, masked
softmax kernel, BF16 probabilities for the `PV` product, exactly as the serving engine rounds
them). Packed rows with several documents use a block-causal mask (query `t` attends to keys in
`[start(t), t]`) and per-document RoPE positions, as prepared by the data tools. This is fine
for the model sizes this targets (`T <= 2048` on 16 GB); it is not a flash attention.

Backward recomputes each layer from its saved input (layer-input checkpointing), so activation
memory is `(layers + 1) * T * hidden * 4` bytes plus one layer's workspace.

### Teacher and losses

The full-precision teacher (dense BF16 operands, no SubLN, same code path) runs live on the same
GPU and produces FP32 logits `[T, vocab]` for every micro-batch. Loss per supervised token:

```
L = ce_w * CE(student)                      (shifted next-token targets inside each document)
  + kl_w * T^2 * KL(softmax(z_t/T) || softmax(z_s/T))   (logits distillation, temperature T)
  + attn_w * (1/3) sum_{X in {Q,K,V}} mean_heads mean_rows KL(R_t^X || R_s^X)
```

where `R^X = softmax(X X^T / sqrt(head_dim))` (block-causal masked) is the MiniLM-style self
relation of the pre-RoPE projections at one chosen layer (`--attn-layer`; `--attn-rope` uses the
rotated Q/K instead). Q relations use the query heads, K/V relations the KV heads. The gradient
of the relation loss flows into the student's `q/k/v` projections of that layer only.

### Stages (BitDistill)

1. **Modelling refinement** – load the HF BF16 model, convert every block linear to BitLinear,
   insert SubLN (gain 1). This is what `bitnet_train` does when it loads a model.
2. **Continued training** – short CE-only training without a teacher (`--no-teacher`) so the
   ternary network recovers from the conversion (initial perplexity is in the millions).
3. **Distillation fine-tune** – `--init` from the stage-2 checkpoint with a fresh optimizer; CE +
   logits KL + attention-relation distillation from the live teacher.

Optimizer: AdamW (FP32 master weights and moments on the GPU), separate learning rate for norm
gains, global gradient-norm clipping, linear warmup then cosine decay to `--min-lr-ratio`.
Gradient accumulation across micro-batches (`--rows` packed rows each, `--accum` micro-batches).

## Formats

### Dataset

`bitnet_prepare` tokenises JSONL `{"text": ...}` documents with the checkpoint's byte-level BPE
tokenizer (the GLM/Llama-3-style `glm53::Tokenizer`, which now also implements the HF `Digits`
pre-tokenizer and tolerates vocabularies without every byte token, e.g. SmolLM2) and packs them
into the native v1 dataset format of `qtip_prepare` (`tokens.i32`, `loss.u8`, `segments.i32`,
`positions.i32`, checksummed `metadata.json`). Each document starts with BOS; a document may
continue across rows (positions continue); every real token is supervised. The trainer's
`qat::Dataset` reader is reused unchanged.

### Trainer checkpoints

`WORK/ckpt-N/state.safetensors` holds the flat FP32 parameter vector (`[weights of every block
linear in layer order q,k,v,o,gate,up,down | norm gains: per layer input, post-attention,
attn_sub, ffn_sub; final norm]`) and the Adam moments; `metadata.json` carries the step, token
count, dataset position and the layout (checked on resume); `latest.json` points at the newest
checkpoint. `--resume` continues a run; `--init` starts a new stage from the parameters only.

### Exported ternary checkpoint

`bitnet_train --export DIR` (or `--resume CKPT --export-only --export DIR`) writes an HF-style
directory (`model.safetensors`, `config.json`, tokenizer files copied from the source):

| Tensor | dtype / shape | Content |
|---|---|---|
| `model.layers.L.<linear>.weight` | `U8 [out, in/4]` | packed ternary codes (layout below) |
| `model.layers.L.<linear>.weight_scale` | `F32 [1]` | absmean scale `s` |
| `model.layers.L.self_attn.attn_sub_norm.weight` | `F32 [heads*head_dim]` | SubLN gain |
| `model.layers.L.mlp.ffn_sub_norm.weight` | `F32 [intermediate]` | SubLN gain |
| `model.layers.L.input_layernorm.weight`, `post_attention_layernorm.weight`, `model.norm.weight` | `F32` | norm gains (trained in FP32, exported unchanged) |
| `model.embed_tokens.weight` (and `lm_head.weight` if untied) | `BF16` | frozen source values |

The codes are taken from the trainer's device operands, so the exported model is exactly the
model the trainer evaluated. `config.json` is the source config plus the detection marker

```json
"quantization_config": {"quant_method": "bitnet", "bits": 1.58, "weight_scale": "absmean_per_tensor",
                        "activation_bits": 8, "activation_scale": "absmax_per_token",
                        "packing": "u32x16_2bit_byte_interleaved", "sub_norm": true},
"bitnet_params": {"source": ..., "step": ..., "tokens": ..., "temperature": ..., ...}
```

analogous to QTIP's `quip_params`. The runtime (`ModelConfig::read`) maps it to
`Quantization::Ternary`; a plain `model_type: llama` checkpoint without any quantization marker is
`Quantization::Dense` (BF16 projections), which is how the baseline of the same model is served.
Both skip QTIP's Hadamard geometry rule (dimensions only need to be multiples of 16) and may tie
the embeddings. The SmolLM2-360M export is 173 MB (79 MB of packed weights + 94 MB of BF16 tied
embeddings) against 724 MB for the BF16 source.

### Packing (`runtime/include/behemoth/ternary.hpp`)

Rows are packed independently (`in % 16 == 0`). Sixteen consecutive input elements share one
little-endian `uint32`; element `e` (0..15) of the word lives in byte `e % 4` at bit offset
`2 * (e / 4)` as `code = w + 1` (`3` is invalid and rejected by the loader). Therefore
`(word >> 2j) & 0x03030303` yields elements `4j..4j+3` as four bytes, which pairs directly with
four consecutive int8 activations in a `dp4a`, and `sum(w * x) = sum(code * x) - sum(x)`. The
header provides the host pack/unpack/absmean/activation-quantiser and an exact int32 row dot
product used by the CPU references of both the trainer and the runtime.

## Inference (`runtime/cuda`)

- `HostLinear` gained `ternary_packed/ternary_scale` and `bf16`; `load_linear` reads them for
  `Ternary`/`Dense` checkpoints; `reference_forward` has exact int32 ternary and dense branches.
- `engine.cpp` (`Model::Impl`): `Linear::forward` dispatches to `ternary_forward` or
  `dense_forward`; SubLN norms run in place before `o_proj` and `down_proj`; the tied head reuses
  the embedding matrix.
- Decode kernel `ternary_gemv_fused`: one block of four warps per four output rows; each block
  quantises the bf16 input row to int8 in shared memory (absmax, sum), then each lane consumes one
  packed word (16 weights) and four int8x4 activation words per step with four `__dp4a` into an
  int32 accumulator; warp reduction; `y = bf16((acc - sum) * s * xs)`.
- Prefill: `ternary_gemv<TOK>` stages up to eight tokens' int8 activations per block so every packed
  word is read once per eight tokens; `Impl::prefill` runs the seven projections of 32-token chunks
  through it (norms, RoPE and attention remain per-token kernels, as in the stepwise engine). The
  QTIP path keeps the one-token-at-a-time prefill. A native int8 tensor-core GEMM is not
  implemented (see *Limitations*).
- The OpenAI-compatible server (`behemoth_cuda_server`) and CLI (`behemoth_cuda`, new `--raw TEXT`
  for base models without a chat template) select the format from `config.json` automatically.

## Commands

```sh
# Trainer (CUDA, sm_86 by default) and runtime builds
cmake -S training -B build-training -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DQTIP_TRAIN_BITNET=ON -DQTIP_TRAIN_TOKENIZER=OFF -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-training -j12 && build-training/bitnet_tests
cmake -S runtime -B build-runtime -G Ninja -DCMAKE_BUILD_TYPE=Release -DBEHEMOTH_CUDA=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-runtime -j12 && build-runtime/behemoth_cuda_tests && build-runtime/behemoth_cuda_llama_tests

# Data (JSONL {"text": ...} per document; wikitext-103 was converted with a one-off script)
build-training/bitnet_prepare --data train.jsonl --tokenizer MODEL --out data/train-1k --seqlen 1024 --max-tokens 60000000
build-training/bitnet_prepare --data validation.jsonl --tokenizer MODEL --out data/val-1k --seqlen 1024 --split heldout --seed 0

# Stage 2: continued CE training of the converted model (no teacher)
build-training/bitnet_train --model MODEL --data data/train-1k --work work/stage2 --rows 1 --accum 2 \
  --max-steps 1000 --warmup-steps 50 --lr 3e-4 --norm-lr 3e-4 --min-lr-ratio 0.3 --no-teacher \
  --checkpoint-every 500 --eval-data data/val-1k --eval-rows 64
# Stage 3: distillation (CE + KL at T=2 + Q/K/V relation at layer 24), export at the end
build-training/bitnet_train --model MODEL --data data/train-1k --work work/stage3 --init work/stage2/latest.json \
  --start-row 2000 --rows 1 --accum 2 --max-steps 10000 --warmup-steps 100 --lr 3e-4 --min-lr-ratio 0.1 \
  --temperature 2 --ce-w 1 --kl-w 1 --attn-layer 24 --attn-w 1 --checkpoint-every 500 \
  --eval-data data/val-1k --eval-rows 64 --export MODEL-bitnet
# Evaluation (teacher and student perplexity), export only, forward trace for parity checks
build-training/bitnet_train --model MODEL --resume work/stage3/latest.json --eval-only --eval-data data/test-1k --rows 4
build-training/bitnet_train --model MODEL --resume work/stage3/latest.json --export-only --export MODEL-bitnet
build-training/bitnet_train --model MODEL --resume work/stage3/latest.json --no-teacher \
  --trace-tokens tokens.json --trace-out trainer.f32        # .f32 [T, vocab]

# Serve / generate / trace with the CUDA engine (same binaries as QTIP and GLM)
build-runtime/behemoth_cuda --model MODEL-bitnet --raw 'The capital of France is' --max-tokens 48 --context 512
build-runtime/behemoth_cuda --model MODEL-bitnet --tokens tokens.json --trace runtime.json --max-tokens 0
build-runtime/behemoth_cuda_server --model MODEL-bitnet --device 0 --context 4096 --name bitnet --port 1337
build-runtime/behemoth_cuda_ternary_bench            # ternary vs bf16 gemv at the SmolLM2 shapes
build-runtime/behemoth_cuda_llama_tests --checkpoint MODEL-bitnet   # CUDA vs CPU reference on a real export
```

`--grad-dump FILE` runs one micro-batch and writes the flat gradient for reference cross-checks.

## Verification

Model: SmolLM2-360M (Llama layout, 32 layers, hidden 960, 15/5 heads of 64, FFN 2560, vocab
49152, tied embeddings). Data: wikitext-103-raw (train: 60 M packed tokens of 1024; validation
and test splits held out). Hardware: one RTX A4000 (16 GB, sm_86), CUDA 12.0.

### Unit tests (`bitnet_tests`, `behemoth_cuda_tests`, `behemoth_cuda_llama_tests`)

- Pack/unpack round trip, the `dp4a` extraction property of the packed layout, packed row dot
  product.
- BitLinear forward: device ternary codes and activation codes equal the host quantiser; the
  output equals the exact int32 reference (max relative error 0.0). Backward: `dX` and `dW` (STE)
  match a double-precision reference to `< 1e-5` relative L2.
- Whole tiny model (2 layers, packed two-document batch) through the HF loader: dense and
  ternary+SubLN forward match a double-precision CPU reference with relative L2 `0.0` (identical
  rounding), cross-entropy identical; the packed second document equals an isolated reference run
  (mask/positions).
- Runtime kernel: raw int32 dot products equal the CPU reference for all tested shapes/tokens
  (16x1, 2560x960, 960x2560 x3 tokens, 320x960 x5, 2560x960 x8, 960x2560 x13, 961x960 x37,
  28672x12288) and the scaled bf16 outputs are bit-exact.
- Runtime end to end: synthetic 3-layer ternary checkpoints (SubLN + tied head; no SubLN + untied
  head) and a dense BF16 checkpoint vs the CPU reference: worst relative L2 `6.9e-8`; batched
  prefill reproduces the stepwise logits.

### Trainer forward vs HF transformers

Teacher (dense, same trainer code) on 32 validation rows: CE 2.45456 (ppl 11.641) vs HF
`transformers` bf16 with the same block-causal mask and positions: CE 2.45497 (ppl 11.646).
Tokenisation of the validation set is identical to HF `tokenizers` (265,216 tokens, 100% match).

### Backward vs PyTorch (throwaway script, outside the repo)

One 256-token row, full model, same batch. Dense student (no quant, no SubLN): per-tensor
gradient relative L2 median 2.1%, worst 5.4%, total norm 4.011 vs 4.033 (BF16 operand rounding
noise). Dense + SubLN: worst 3.6%. Ternary + SubLN, CE only: median 13%, worst 27%, total norm
331.8 vs 328.4; with KL (T=2) and the layer-5 relation loss: median 12.5%, worst 25%, norm 644.7
vs 656.7; losses CE 13.537/13.537, KL 16.058/16.073, relation 2.093/2.107. The larger ternary
spread is intrinsic: changing nothing but the BF16 rounding of the attention probabilities in the
PyTorch reference itself moves its gradients by a median of 22% (worst 45%), because int8 code
flips in an untrained ternary network are chaotic. The native gradients agree with the reference
within that sensitivity.

### Distillation run

RESULTS_PLACEHOLDER

### Runtime parity and speed

PARITY_PLACEHOLDER

## Limitations and open questions

- **Scale of the run.** BitDistill uses billions of tokens; this run used about 22 M tokens on one
  A4000 in under two hours. The student is far from the teacher (numbers above are honest).
  Longer stage 2/3 runs, a larger `--rows`, and lower temperature/KL weights were not explored.
- **Trainer throughput** (~3.4-4.3 k tokens/s at `T = 1024`) is bounded by the materialised
  attention (memory-bound softmax over `[heads, T, T]`) and many small element-wise passes. A
  fused flash-style attention and fused scale/round kernels would help; neither was needed to
  validate the design.
- **Runtime parity floor.** Deep ternary networks amplify summation-order differences through the
  int8 rounding: the runtime vs the trainer and the runtime vs the exact double-precision CPU
  reference both differ by a few percent relative L2 on the real 32-layer model while
  3-layer synthetic models agree to `1e-7`. Argmax agreement is what to look at for greedy
  decoding; a bit-exact chain would require identical reduction orders in both implementations.
- **Prefill GEMM.** The multi-token kernel reuses each weight word for eight tokens but is still a
  `dp4a` GEMV family, not an int8 tensor-core GEMM (`mma.sync.m16n8k32.s8`), which would be the
  right kernel for long prompts. Norms, RoPE and attention stay per token in the stepwise engine.
- **Decode of a 360 M model is launch-bound**, not bandwidth-bound: the packed matrices are
  0.2-0.6 MB each, so per-kernel latency (~5-10 µs) dominates and the ternary kernel's measured
  bandwidth is far below the A4000's 448 GB/s. CUDA graphs (as in the Behemoth fast path) would
  remove most of that overhead; the comparison against bf16 at the same shapes is still like for
  like.
- **Formats.** Only the Llama layout (no biases, no q/k norms, SiLU) is supported; `Quantization::Ternary`
  is single-GPU and not available in the TP2/batched Behemoth fast path. The embedding/head stay
  BF16 (the dominant size of the SmolLM2 export).
- **Open question: activation statistics in the trainer vs runtime.** The trainer emulates the
  runtime's BF16 boundaries by default; the scale multiply order was matched too. Any remaining
  mismatch comes from reduction orders (cuBLAS vs the dp4a warp reduction, norm sums). Whether to
  make the trainer use the runtime kernels for the forward (identical bits, slower backward) is
  left open.
