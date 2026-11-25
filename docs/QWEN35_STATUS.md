# Qwen3.5 support — status

Branch `qwen35`, merged into main as `merge-qwen35` (2026-11-24). Reference implementation: `nmoe` (read-only). Test model: Qwen3.5-0.8B (dense,
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
- Builds clean (`build`, `runtime/build-rocm`, gfx1201); the quantizer side also builds as CUDA
  (`-DQTIP_CUDA=ON`, qwen_kernels goes through the same .hip -> .cu copy as the other kernels).
  Runtime validated on ROCm — see "ROCm end-to-end (merged tree)" below.

### Baselines recorded

- CPU bf16 ppl (dense 0.8B, our reference forward, 512-token windows): **6.2137** over 4088
  predictions of `/tmp/qwen_eval.i32` (re-measured on the current eval file; the older 5.7415 was a
  slightly different eval set).
- **Quantized-reference ppl (0.8B QTIP-2bit, `qwen_qt_ref --ppl`, double-precision trellis decode):
  9.3136** (stale — see the ROCm section: does not reproduce, runtime == qt_ref now) over 3584 predictions of the same corpus — the first 0.8B quantization used only 202k
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

## ROCm end-to-end (merged tree, 2026-11-24, .229 gfx1201, dense 0.8B)

Token files (regenerated, `~/qwen35_merge_rocm/data` on .229, script `mk_tokens_merge.py`):
`legacy_eval.i32` = the original `/tmp/qwen_eval.i32` recipe (first 4096 tokens of the source-code
calibration text — **in-sample for the old 202k-token calibration**, so not a fair eval for it);
`calib4m.i32` = 2048 x 2048 tokens wikipedia (shard 0) + pg19 (train shard 0);
held-out `eval_wiki.i32` (wikipedia shard 40) and `eval_pg19.i32` (pg19 test), 8192 tokens each.
All ppl numbers: 512-token windows, stride 448 (`qwen_eval --window 512`, same windows as
`qwen35_ref --ppl`), unless noted.

**GPU parity, quantizer layer runner** (`tools/qwen_gpu_parity`, bf16 weights + bf16 residual vs
the CPU fp32 reference, 19-token prompt, all 24 layers): isolated per-layer residual rel-L2 <= 4.7e-3
(tol 2e-2), chained <= 1.1e-2 (tol 5e-2), final logits max|diff| 0.31 of max |logit| 20.9 (1.5%),
argmax 19/19. `QWEN35_MODEL=... qtip_tests_gpu`: layer 0 (DeltaNet) rel 1.2e-2, layer 3 (full) 1.2e-2.

**Runtime bf16** (original HF checkpoint through the bf16-linear fallback):
- per-layer residual vs CPU fp32 reference (19 tokens): rel-L2 3.7e-3 (layer 0) .. 9.6e-3 (layer 23);
  last-token logits max|diff| 0.11 of 15.3, argmax equal.
- greedy 24 tokens vs HF transformers fp32 (`model.generate`, 5 prompts): 4/5 identical over all 24
  tokens; "In 1905, Albert Einstein published" diverges at token 18 (near tie), re-converges.
- ppl: legacy_eval **6.2151** (CPU reference 6.2137), eval_wiki 17.79, eval_pg19 20.69.
- `qwen_gpu_tests` (ctest `qwen_gpu_correctness`): q_gateup, delta-step chunk/decode continuity,
  prefill-vs-decode logits (max|diff| 0.12 of 15.3, argmax " Paris") — passes.

**Runtime quantized vs the CPU quantized reference** (`qwen_qt_ref`, double-precision trellis
decode): on the old 202k-token checkpoint, first 1030 legacy tokens (2 x 513 windows) both give
**6.1699** (identical to 4 digits); 19-token prompt KL(qt_ref || runtime) = 4e-4, per-layer residual
rel-L2 <= 1.7e-2. Chunk size (16 vs 256) changes ppl by < 0.1%. The documented 9.3136 does not
reproduce: the same checkpoint/eval/windows (513/512, 3584 predictions) give 10.05 in the runtime,
and qt_ref agrees with the runtime where both were run, so 9.31 is a stale measurement (probably
taken before the model-embed regeneration of the final-norm fix), not a runtime gap.

**Quantization, 4M tokens** (`qtip_quant --mode run --gpus 0,1 --ntok 4194304 --seq-len 2048
--batch-rows 4096`, both GPUs shared with another job): 19.4 min wall, 11.4 GB host RSS, 0.64 GB
checkpoint. Per-linear proxy errors 0.01–0.046 (worst: mlp.up_proj of layers 0–5, self_attn.o_proj).

| checkpoint | legacy_eval | eval_wiki | eval_pg19 |
|---|---|---|---|
| bf16 (runtime) | 6.215 | 17.79 | 20.69 |
| QTIP-2bit, 202k code tokens (old) | 8.98 (in-sample) | 60.77 | 67.97 |
| QTIP-2bit, 4M wiki+pg19 tokens (new) | 24.14 | 28.82 | 31.28 |

Greedy on the new checkpoint is fluent but less factual ("The capital of France is the capital of
the Kingdom of France", fibonacci code correct, "Water boils at ... 20°C").

### Runtime bugs fixed during the ROCm bring-up (commit d79b9ab)

1. `q_gateup` computed `f2bf(bf2f(silu(g)) * u)`: the float was implicitly converted to `uint16_t`,
   so **every dense MLP contributed ~0** (trellis and bf16 path) — the earlier "runtime bug" hunt.
2. bf16 linears reuse the SV epilogues with an all-ones vector sized max(inter, hidden) = 3584 <
   q_proj width 4096: the last q/gate columns were scaled by garbage.
3. The causal-conv ring of `q_delta_step` was indexed by the row within the chunk: every decode
   step (and chunked prefill past 256 tokens) used a wrong conv history. Now indexed by absolute
   position (`pos0`).
4. Loader: HF checkpoints use `model.language_model.layers.*`; codebook only built when a trellis
   exists; `max_context` no longer reset by `load()`; debug dumps sync the stream first.

## Qwen3.8-27B-FP8 serving (branch qwen27-fp8, .229: 2x R9700 gfx1201, TP2, 2026-11-25)

FP8 block-scaled checkpoint served natively: W8A8 FP8 WMMA GEMMs (128x128 weight block scales,
dynamic per-token x 128-column activation scales), tensor parallel over both GPUs (heads /
intermediate / vocab shards, device-flag all-reduce through pinned host memory, no P2P on this box),
paged KV (64-token pages) + per-slot DeltaNet state, continuous batching with HIP-graph decode steps,
packed multi-sequence prefill, bf16 WMMA flash attention for prefill, OpenAI front-end
(`runtime/tools/qwen_openai_server.py` -> `qwen_worker`).

### FP8 KV cache (`--kv bf16 | k8v16 | fp8`, env `QWEN_KV`; serving default `fp8`)
Layout: e4m3 (OCP) bytes `[pool row][kv head][256]` + one fp32 scale per (pool row, kv head),
scale = amax / 448, written by the K/V append kernels (`q_qk_norm_rope`, `q_rope_b`). Readers:
decode `attn_b` and split-K fold the K scale into the q.k dot and the V scale into the
probabilities; flash prefill converts e4m3 -> bf16 exactly into the LDS tiles and applies the K
scale to S and the V scale to P. 260 B per head-row vs 512 B bf16 (K+V fp8: 2x KV tokens).

Quality (TP2, chunk 2048; ppl noise between equivalent runs is ~0.1-0.2%):

| KV | wiki 512-win | pg19 512-win | wiki514 | decode-path ppl (wiki) | batch test 16x48 | needles 16k-32k |
|---|---|---|---|---|---|---|
| bf16 | 8.2104 | 8.8522 | 3.7322 | 7.7149 | 16/16 | 4/4 |
| k8v16 | 8.2287 (+0.22%) | 8.8570 | 3.7153 | 7.7187 | 16/16 | 4/4 |
| fp8 (K+V) | 8.2279 (+0.21%) | 8.8473 (-0.06%) | 3.7269 | 7.7192 (+0.06%) | 16/16 | 4/4 |

Greedy (serving path, 64 tokens) vs bf16 KV: prompts 1/3/4 identical, 0 and 2 diverge at 47 / 60
(near-ties); needle answers (passphrase at 10-85% depth of 16k/20k/24k/32k pg19 prompts) identical.

Decode aggregate tok/s (`qwen_eval --serve-seqs 64 --bench-batch`, forked context):

| ctx | B | bf16 | fp8 |
|---|---|---|---|
| 1k | 1 / 16 / 32 / 64 | 31.0 / 354 / 554 / 702 | 31.0 / 355 / 558 / 711 |
| 8k | 1 / 16 / 32 / 64 | 30.7 / 298 / 431 / (no room) | 30.8 / 317 / 469 / 572 |
| 32k | 1 / 8 / 16 | 27.4 / 111 / (no room) | 29.6 / 157 / 231 |

Prefill (single sequence, warm): 2k 1784 vs 1778 tok/s, 8k 2650 vs 2630 (-0.7%, flash +3%),
32k 2022 vs 2004.

KV pool / max concurrency (max-context 131072): 32 slots 400k -> 789k tokens (@8k 32 -> 32,
@32k 12 -> 24); 64 slots 320k -> 629k tokens (@8k 39 -> 64, @32k 9 -> 19).
Server e2e (64 slots, fp8): 64x256 702 tok/s aggregate (TTFT p50 0.93 s; 559 with 32 slots + queue),
32x512 562 tok/s (TTFT p50 0.39 s).

### MTP speculative decoding (`--mtp`, draft k = 1)
The checkpoint's MTP head (`mtp.fc` on `[enorm(embed(t+1)) | hnorm(h_t)]` with h_t the post-final-norm
hidden, as vLLM's qwen3_next_mtp; one gated full-attention layer + dense MLP; `mtp.norm`; shared
embed / fp8 lm_head) is loaded as an extra TP-sharded layer with its own fp8 KV pages. Its K/V is
written during prefill (shifted prompt tokens) and by every plain decode step, so MTP can be switched
per step. Verify = one batched decode graph with 2 rows per sequence `[last, draft]`; `q_delta_bt`
runs both DeltaNet steps per sequence (bit-identical to two `q_delta_b` steps) and stores the state
after row 1 in a spare state entry; a rejected draft just switches the slot's state entry (conv ring /
KV are positional and get overwritten). Then one MTP pass (1-2 rows per sequence) drafts the next
token. Worker: greedy = exact match, sampling = speculative acceptance rule `min(1, p/q)` + residual
`max(0, p-q)` on the sampler's top-64 distributions; adaptive on/off from the measured
verify/plain step ratio and an acceptance EMA (`--mtp-max N` for a fixed threshold).

Correctness: greedy MTP == plain greedy token-identical: `qwen_eval --mtp-test` 16 pg19 + 16 wiki
prompts x 64 tokens, alone and batched (32/32 each); through the server (1x400 thinking/answer,
8x300 concurrent) identical text MTP on vs off.

Acceptance (k = 1): wiki 0.82, pg19 0.75; chat (768 tokens, T=0 / 0.7): code 0.89 / 0.87, code +
thinking 0.79 / 0.76, essay 0.81 / 0.83, explanation + thinking 0.79, math + thinking 0.81.

Decode (qwen_eval --bench-batch, pg19 context, acceptance 0.80), aggregate tok/s plain -> MTP:

| ctx | B=1 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| 1k | 30.8 -> 50.9 (1.65x) | 113 -> 178 | 206 -> 310 | 352 -> 498 | 462 -> 559 | 552 -> 639 (1.16x) |
| 8k | 30.6 -> 50.1 (1.64x) | 106 -> 164 | 185 -> 273 | 298 -> 410 | 374 -> 452 | 433 -> 505 (1.17x) |

Step cost (verify + draft vs plain): 1.09x at B=1 ... 1.55x at B=32 (verify rows = 2B, > 64 rows not
supported -> B > 32 always plain). Server e2e (64 slots, temp 0.7, no thinking): 1x256 31.3 -> 50.8
tok/s, 4x512 114 -> 175, 8x512 207 -> 310, 16x512 358 -> 495, 32x512 563 -> 643, 64x256 709 -> 707.
Cost: KV pool 629k -> 430k tokens with 64 slots (32 spare DeltaNet state entries x 72 MB per GPU +
the MTP layer's KV); `--mtp-seqs 16` halves the spare entries (MTP then only up to 16 sequences).

Server: `qwen_openai_server.py --worker qwen_worker --model Qwen3.8-27B-FP8 --gpus 0,1 --tp --port 8027
--max-context 131072 --max-seqs 64 --kv fp8 --chunk 2048 --mtp --mtp-seqs 32 --mtp-max 0`.

## Pending / open issues

1. **2-bit quality on 0.8B**: +62% wiki / +51% pg19 ppl over bf16 with 4M general calibration tokens
   (the old 202k code-only calibration was far worse out of domain: 60.8 wiki). Proxy errors look
   normal for 2-bit QTIP; a sub-1B hybrid model is simply fragile at 2 bits. Not investigated further
   (time box): candidates are keeping the small DeltaNet/attention projections or the first MLPs at
   higher precision, or a fine-tuning/recovery pass (the trainer rejects Qwen3.5).
   Still queued from the earlier plan: tiny-MoE quantize + runtime parity (step 5 of
   `tools/qwen35_gpu_validate.sh`).
2. **122B-A10B**: full plan (memory/time/disk) in docs/QWEN35.md; FP8 block-scale reading
   implemented; run NOT executed per task instructions.
3. Runtime is single-GPU / single-sequence (eval); no scheduler/server integration, prefix cache
   disabled by design for Qwen3.5 (delta state cannot resume from KV pages alone).
4. Trainer: rejects Qwen3.5 with a clear "not supported in the trainer yet" error (config/loader
   awareness only, per the task's minimum).

## Commits
- merge into main: `28a6440` (merge-qwen35; Pipeline port into src/pipeline.{hpp,cpp}), `d79b9ab`
  (runtime fixes + qwen_gpu_tests + qwen_gpu_parity + `qwen_eval --window`)

- `5542ab8` design note docs/QWEN35.md
- `b6812e0` + fixes: qwen config/reader/CPU ref/GPU runner/main integration
- runtime: qwen kernels + model + eval tool (see `git log qwen35`)
- `6337a50` MoE: per-expert Hessians, quantize path, runtime MoE + bf16 fallback, tiny-MoE
  generator, HF-validated CPU MoE reference, trainer guard, 122B FP8 block-scale reading + plan
