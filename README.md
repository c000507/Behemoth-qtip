# qtip_quant

For GLM-5.3-Flash (original 320B MoE and the custom dense conversion), see the
[standalone GLM text backend](glm53/README.md), including MLP-only NVFP4 conversion
and CUDA/HIP inference kernels with their current validation limits.

C++/HIP (ROCm, gfx1201) implementation of **QTIP** quantization ([arXiv 2406.11235](https://arxiv.org/abs/2406.11235),
reference code [Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip)) for **Behemoth-R1-123B-v2**
(Mistral-Large layout, 88 layers), using both R9700 GPUs of the box.

Code: the paper's 2-bit HYB code. Bitshift trellis L=16, K=2 bits/weight, V=2, `quantlut_sym` with a
2^9-entry tunable LUT, 16x16 tiles (one tail-biting trellis of 256 weights per tile), kernel (`for_kernel`) layout.
The quantizer gives each layer one LDLQ/Viterbi pass. A standalone C++20/HIP streaming QAT
trainer tunes continuous signs/scales and norms without LoRA, with no Python dependency; see
[fine-tuning instructions and verification status](docs/FINE_TUNING.md).

Pipeline:

1. **Calibration**: packed 2048-token sequences, 75% OpenWebText and 25% SmolTalk chats. This is the same data
   choice as the original quantbench calibration mix.
2. **Hessian collection**: the original FP8 model is run in bf16 layer by layer. FP8 weights are used exactly, and
   the per-channel scale is applied in fp32 to the GEMM output.
   - Causal GQA attention uses native WMMA query/key tiles and online softmax, with no full sequence-length-squared
     score/probability buffers. Q/K/V and attention outputs remain BF16; softmax and accumulation are FP32.
   - `H = XᵀX/N` is accumulated for the inputs of q/k/v (shared), o, gate/up (shared) and down.
   - Accumulation is bf16-input, fp32-accumulate (hipBLASLt).
   - Each GPU owns half of the sequences and runs its own layer chain. The two partial Hessians are summed on the
     host.
   - Activations stay in host RAM (bf16, about 98 GB for 4.19M tokens).
3. **Quantization**: one Hessian group at a time (q, o, gate+up, down), preserving source FP8 K/V by default
   (`--kv-format qtip` quantizes q+k+v for the legacy format; for BF16 sources `--kv-format fp8` converts K/V to FP8 instead).
   - **RHT**: random sign vectors SU/SV, then the reference's Hadamard factorisation `hadK ⊗ H_{2^p}` on both
     sides (12288 = 12·1024, 28672 = 28·1024, 1024 = 2^10).
   - **Damping**: `regularize_H` (σ = 1e-2 · mean diag).
   - **Block LDL**: `block_LDL(HRr, 16)`, computed in fp64 on the CPU (blocked AVX-512 Cholesky). It runs in a side
     thread, so the LDL of the next group overlaps the GPU work of the current one.
   - **Wscale**: `Wscale = rms(Wr) / (rms(lut)·0.9)`.
   - **LDLQ**: `lib/algo/ldlq.py` with 128-column buffers. The error feedback inside a buffer uses an fp32 kernel.
     The `prod_cache` update across buffers uses a bf16 hi/lo split GEMM (≈fp32 accurate).
   - **Viterbi** on the GPU, two passes per tile as in `bitshift_codebook.quantize`:
     - pass 1: rolled sequence, to get the overlap state;
     - pass 2: constrained (tail-biting).
4. **Output**: `trellis` (int16, packed and swizzled exactly as `quantize_finetune_llama` does for the kernel
   layout), `SU`, `SV` (×Wscale, as `hfize_llama`/`unpack_quip` store it), `tlut`, `rcp`, `tp_rank`, plus the
   layer norms. The result is an HF directory in the QTIP-hfized llama layout.


## Inference runtime (`runtime/`)

For NVIDIA GPUs, the [shared CUDA engine](runtime/cuda/README.md) serves both the
Llama/Mistral-layout Behemoth QTIP checkpoint and GLM-5.3-Flash models, selecting
the architecture from `config.json`.

`runtime/` is the Behemoth C++/HIP inference runtime and OpenAI-compatible server (`behemoth_server`), imported
with its history from the AQLM codebase. It serves both the AQLM 1x16 model and QTIP 2-bit models written by
`qtip_quant` (detected from `quip_params` in `config.json`): trellis-decode GEMV/WMMA kernels, Hadamard
transforms and rotated-row tensor parallelism over the two GPUs. Details and measurements:
`runtime/README.md`, `runtime/VALIDATION.md` and the reports in `runtime/`.

Build (from this repository root):

```sh
cmake -S runtime -B runtime/build-rocm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBEHEMOTH_ROCM=ON -DBEHEMOTH_SERVER=ON -DBEHEMOTH_TOKENIZER=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build runtime/build-rocm -j 16
```

Serve a QTIP model on both GPUs (recommended flags, see `runtime/PREFILL_SCHED_REPORT.md`):

```sh
runtime/build-rocm/behemoth_server --model /mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-2bit \
  --devices 0,1 --transport zerocopy --code-layout lane --memory-fraction 0.9 \
  --token-budget 2048 --prefill-chunk 2048 --mixed-prefill-tokens 512 --prefill-interval 24 \
  --host 0.0.0.0 --port 1337 --threads 4 --profile false
```

(The prefix cache is enabled at its defaults, see below; add `--prefix-cache-slots 0` to disable it.)

The model directory needs `tokenizer.model` next to `tokenizer.json` (copy it from the source model if
`qtip_quant --mode assemble` did not). Single-request decode is ~44 ms/token short context (~52 ms at 16k) on
2x R9700, against ~69-74 ms/token for the AQLM 1x16 model.

Fresh quantization keeps the source FP8 K/V projection weights while keeping
the other projections in QTIP 2-bit (`--kv-format qtip` selects legacy all-2-bit).
`qtip_fp8_kv` upgrades an existing checkpoint;
`qtip_reconstruct` performs sequential teacher-guided block fitting. See
[FP8 K/V and block reconstruction](docs/MIXED_FP8_RECONSTRUCTION.md).

`qtip_quant --mode adaptive` quantizes each layer to QTIP 2-bit and re-quantizes layers whose
next-token KL (or normalized MSE) against the original model exceeds `--kl-threshold` to NVFP4,
writing a mixed-format checkpoint (`adaptive_report.json` + `config.json:qtip_quant.layer_formats`;
served by the CUDA runtime, see [adaptive mixed-format quantization](docs/ADAPTIVE.md)).

For repeated paragraphs or reasoning loops, `qtip_diagnose` captures/replays exact
tokens and compares source/student probabilities and block outputs; see
[repetition diagnosis](docs/REPETITION_DIAGNOSIS.md).

QTIP prefill and scheduling options (`runtime/PREFILL_SCHED_REPORT.md`):

| flag / env | default | meaning |
|---|---|---|
| `--token-budget N` | 1024 | tokens per step; QTIP allows up to 4096 (AQLM 1024) |
| `--prefill-chunk N` | 256 | prefill tokens per request per step (when nothing is decoding) |
| `--mixed-prefill-tokens N` | 0 | at most N prefill tokens in a step that also decodes (0: no cap) |
| `--prefill-interval K` | 1 | while requests decode, only every K-th step carries prefill tokens |
| `--max-batch-requests N` | 256 | requests (sampled rows) per step |
| `--qtip-dense-rows N` / `BEHEMOTH_QTIP_DENSE_ROWS` | 512 | steps of >= N rows decode each linear to bf16 and use hipBLASLt (0: always the WMMA trellis kernel) |
| `--qtip-row-transforms N` / `BEHEMOTH_QTIP_ROW_TRANSFORMS` | 64 | steps of >= N rows use the row-batched Hadamard kernels (bit-identical) |
| `BEHEMOTH_QTIP_OVERLAP_ROWS` | 2x dense rows | steps of >= N rows run as two overlapped micro-batches (0: off) |
| `BEHEMOTH_QTIP_MICRO_BATCHES` | 2 | micro-batches per overlapped step (max 4) |
| `BEHEMOTH_QTIP_DENSE_TUNE` | 64 | hipBLASLt candidates timed per dense GEMM shape at startup (0: heuristic only) |
| `BEHEMOTH_QTIP_DMA_ROWS` | 9 | steps of >= N rows exchange through the copy-engine channels |
| `BEHEMOTH_ATTN_PREFILL_BLOCKS` | 512 | key-split target for multi-query steps with few query tiles (0: off) |

Step-level prefill timings without the server: `runtime/build-rocm/behemoth_prefill_bench MODEL [rows,...] [ctx,...] [reps]`.

Prefix cache (on by default, `runtime/PREFIX_CACHE_REPORT.md`): when a prompt's token ids (after chat
templating) start with a cached sequence, that KV is restored instead of prefilled and only the rest of the prompt
is prefilled. Every request that ends - finished, stopped, or cancelled by the client, also mid-prefill - is cached
(prompt + generated tokens), so the next turn of a conversation and prompts sharing a system prompt start from the
cached KV. Slots live in pinned host memory (both GPUs' KV pages, ~186 KB per token, i.e. ~6 GB per 32k-token
slot); the most recent ones also keep their GPU pages until the KV pool needs them (a GPU-tier hit needs no copy).
Turn-2 TTFT after a 32k-token turn 1: ~0.6 s (GPU tier) / ~0.9 s (host restore, 13 / 24 GB/s per GPU) instead of
133 s. `usage.prompt_tokens_details.cached_tokens` reports the reused prompt tokens (streaming:
`"stream_options": {"include_usage": true}`), `/metrics` has a `prefix_cache` section.

| flag | default | meaning |
|---|---|---|
| `--prefix-cache-slots N` | 10 | cached sequences (LRU; the next turn of a conversation replaces its slot); 0 disables the cache |
| `--prefix-cache-gb X` | 64 | host memory cap (both GPUs' pages, 1e9 bytes); pinned in full at startup (~29 s for 64 GB), LRU slots are evicted to stay under it |
| `--prefix-cache-min-tokens N` | 64 | shortest sequence saved and shortest prefix restored |
| `--prefix-cache-gpu true\|false` | true | keep a finished sequence's KV pages on the GPUs until the pool needs them |
| `--prefix-cache-staging-pages N` | 32 | pages per device staging buffer (two per GPU, 47.6 MB each at 32; taken from the KV pool) |
| `--prefix-cache-restore-pace N` | 0 | while requests decode, issue at most N restore pages per step (smoother ITL for the others, later TTFT for the restored request; 0: all at once) |

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j32
```
Requirements: ROCm (hipcc/clang, hipBLASLt), OpenMP. `third_party/nlohmann/json.hpp` is vendored.
`src/qtip_constants.hpp` contains committed reference Hadamard tables and the HYB LUT
(k-means, seed 0); generating them is not a build step. All repository Python utilities
have been removed. The remaining benchmark shell scripts use `jq` for JSON processing. Historical reports retain
provenance and results from the retired tools; they are not current run instructions.

## Usage

```
# Supply pre-tokenized calibration data as raw little-endian int32 [nseq, seqlen].
# For GM-CoT, build the native preparer (see docs/FINE_TUNING.md):
./build-training/qtip_prepare --data conversations.jsonl --tokenizer MODEL \
    --split-metadata calibration-split.json --gm-frac 1 --seqlen 2048 \
    --max-tokens 4194304 --out calibration

# full run (collect + quantize), both GPUs, resumable
./build/qtip_quant --tokens calibration/tokens.i32 \
    --out /mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-2bit --work /mnt/nvme/qtip_quant/work --gpus 0,1
```

Flags (`--help`):

| flag | default | meaning |
|---|---|---|
| `--mode run\|collect\|quantize\|assemble\|adaptive` | run | `collect`: Hessians only (kept on disk). `quantize`: layers whose Hessian file exists in `--hess-dir`. `assemble`: write config/index/embeddings. `adaptive`: per-layer QTIP-vs-NVFP4 selection (see [docs/ADAPTIVE.md](docs/ADAPTIVE.md); extra flags `--kl-threshold`, `--kl-metric lmkl|mse`, `--kl-tokens`, `--adaptive-next-input original|quantized`) |
| `--model DIR` | /mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic | FP8-Dynamic (compressed-tensors) checkpoint, read only |
| `--tokens FILE`, `--ntok N`, `--seq-len N` | –, 4194304, 2048 | calibration tokens (raw int32 [nseq][seq_len]) |
| `--layers A:B` | all | layer range [A, B) |
| `--gpus 0,1` | 0,1 | HIP devices |
| `--batch-rows N` | 4096 | tokens per forward batch, rounded up to whole sequences (at least `--seq-len`) |
| `--hess-dir DIR` | `<work>/hess` | Hessian files `layer<L>.safetensors` (F32 qkv/o/gateup/down, normalised by token count) |
| `--backlog N` | 2 | max layers with collected but unquantized Hessians (5.1 GB RAM + disk each) |
| `--keep-hessians` | off | do not delete a layer's Hessians after it is quantized |
| `--vit-batch N` | 1024 | trellises per Viterbi launch (260 KB GPU scratch each) |
| `--sigma-reg`, `--scale-override` | 1e-2, 0.9 | as the reference |
| `--ldl-threads N` | 16 | CPU threads per fp64 LDL |
| `--dump-dir DIR` | – | write W and the reconstructed weight per linear (fp32), for validation |
| `--dump-acts DIR` | – | debug: activations of sequence 0 after each layer |
| `--seed N` | 0 | sign-vector seed |

Attention is tiled automatically, including for 32k calibration sequences. For
example, add `--seq-len 32768` when the token file contains 32k rows. The default
batch then becomes one full 32k sequence. This removes 72 GiB of score/probability
workspace per GPU at 32k for Behemoth, plus the old intermediate attention output.
Q/K/V, model activations, decoded source weights and Hessians still need memory.
The quantizer continues to treat each raw token row as one causal document with
positions starting at zero; it does not consume the preparer's document masks.

Numerical/causal checks and an optional full-shape benchmark:

```sh
ctest --test-dir build -R gpu_attention --output-on-failure
./build/qtip_attention_tests --device 0 --bench 32768
```

The tests cover FP64 attention agreement, comparison with the previous
GEMM/softmax path, GQA/head layouts, partial tiles, output guards and future-token
isolation. The benchmark checks uniform attention against prefix means and
reports attention-only time. Local validation compiled the HIP kernels for
gfx1201 and checked host code; GPU execution and timings require GPU hardware.

### Resume

`<work>/progress.json` records every finished layer (per-linear proxy error, timings) and every complete Hessian
file.

- A restart skips finished layers, as long as their `model-layer-XXXXX.safetensors` exists.
- A restart quantizes layers whose Hessians are on disk.
- Only the remaining layers are collected. The GPU chains first replay the forward pass (without Hessians) up to
  the first layer that still needs collecting.

Layer outputs and Hessians are written atomically (tmp + rename). Per-linear results are also appended to
`<work>/quant_log.jsonl`. The human-readable log is `<work>/qtip_quant.log`.

## Output format

The output follows the layout `quantize_llama/hfize_llama.py` produces, loadable by QTIP's
`model.llama.LlamaForCausalLM` / `lib.utils.unsafe_import.model_from_hf_path`.

`config.json` is the source config with these changes:
- `model_type: llama`
- `architectures: [LlamaForCausalLM]`
- `quip_params: {L:16, K:2, V:2, tlut_bits:9, decode_mode: quantlut_sym, td_x:16, td_y:16, ...}`
- `torch_dtype: bfloat16`

The index is `model.safetensors.index.json`. Shards are:
- `model-embed.safetensors`: embed_tokens, norm, lm_head (bf16, copied)
- `model-layer-XXXXX.safetensors`: one per layer, containing

| tensor | dtype / shape | content |
|---|---|---|
| `...{q,k,v,o}_proj.trellis`, `...{gate,up,down}_proj.trellis` | I16 [(m/16)(n/16), 32] | packed tail-biting trellis bitstreams, kernel layout |
| `.SU` | BF16 [n] | input sign vector |
| `.SV` | F32 [m] | output sign vector × Wscale |
| `.tlut` | F16 [512, 2] | HYB LUT (identical for all linears) |
| `.rcp`, `.tp_rank` | I64 [] | 0, 8 (no tensor-parallel split) |
| `input_layernorm.weight`, `post_attention_layernorm.weight` | BF16 [12288] | copied |

Decoding (what an inference runtime has to implement) for an m×n linear:

```
y = SV ⊙ hadU_m( hadUt_n(x ⊙ SU) · Ŵᵀ )
```
- `hadU_k(v) = (hadK ⊗ H_{k/K}) v / √k`, and `hadUt` uses `hadKᵀ`.
- `Ŵ[16b+r, 16c+s]` = `lut[state]` of trellis (b, c) at sequence position `p = INV_PERMUTE[16r+s]` (component `p&1` of
  step `p>>1`).
- The states are a sliding 16-bit window over the 512-bit tail-biting stream: start bit 4·step, wrapping around.
- `lut[s] = tlut[((s(s+1)) >> 6) & 511]`, with component 0 negated when bit 15 of s(s+1) is set.

The native decoder and the committed numerical fixtures verify this reconstruction.

## Llama 3.3 70B Instruct (`meta-llama/Llama-3.3-70B-Instruct`)

Besides Behemoth, `qtip_quant` (classic and `--mode adaptive`) and the shared CUDA runtime
(`behemoth_cuda`, `behemoth_cuda_server`) support the Llama 3.3 70B layout. All dimensions come from
`config.json` (hidden, FFN, layers, q/kv heads, `head_dim`, vocab, `rms_norm_eps`, `rope_theta`); Behemoth
output is unchanged. Unsupported shapes are rejected with an error (hidden/FFN/head dims must be a multiple
of 32 and `2^p`, `12*2^p` or `28*2^p` with `2^p <= 8192`; `rope_scaling` other than `llama3` is rejected).

- Geometry: 80 layers, hidden 8192 (= 2^13, no hadK factor), 64/8 heads, `head_dim` 128, FFN 28672
  (= 28*1024), vocab 128256, untied embeddings. Hadamard dims: q/o in 8192 out 8192, k/v out 1024,
  gate/up in 8192 out 28672, down in 28672 out 8192 (the FWHT kernels now allow a 2^13 chunk).
- **llama3 RoPE scaling** (factor 8, low/high_freq_factor 1/4, original_max_position_embeddings 8192): the
  inverse-frequency table is computed in fp64 on the host (HF `_compute_llama3_parameters`) and used by the
  quantizer's calibration forward (`k_qkv_rope_scaled`) and by the CUDA runtime (`rope_scaled`). The
  assembled `config.json` keeps the source `rope_scaling` block verbatim.
- **BF16 sources**: the quantizer detects BF16 (vs FP8-Dynamic) source linears from `q_proj`'s dtype. There
  are no FP8 K/V to preserve, so `--kv-format` defaults to `qtip` (all 7 linears quantized). An explicit
  `--kv-format fp8` is accepted: k_proj/v_proj are converted to FP8-Dynamic (E4M3 `weight` + per-output-channel
  BF16 `weight_scale` = absmax/448, round-to-nearest-even, saturating; same layout as an FP8-Dynamic source, so the
  runtime loads them as W8A16 exactly like Behemoth) and only q/o/MLP are QTIP/NVFP4. Relative weight L2 error
  of the conversion is about 2.6% (Anubis-Mini-8B layers 0-2). They are the smallest, most error-sensitive
  linears (v_proj is the worst by proxy) and cost about +6 MB/layer on the 8B (1024x4096 k+v: 8 MB FP8 vs
  about 2 MB QTIP). Adaptive mode keeps k/v FP8 in QTIP and NVFP4 layers and the KL evaluation uses the
  dequantized FP8 weights. The work directory remembers the K/V format (`progress.json`); resuming with a
  different `--kv-format` is refused.
- **Adaptive mode**: the lm_head logits GEMM for the metric probe runs in vocab-row tiles (about 256 MB of
  fp32 logits at most, `Options::logit_tile` overrides it), so the 128256-row head only needs its bf16 copy
  (~2.1 GB) resident on the device. Tiled and untiled metrics are asserted equal.
- **Runtime**: stop tokens come from `generation_config.json` `eos_token_id` (list), else `config.json`
  (Behemoth: 2). Tokenizer: the Llama 3 `tokenizer.json` (byte-level BPE, `Split` regex + ByteLevel) goes
  through the existing BPE path; the checkpoint's Jinja chat template renders the header/`<|eot_id|>` tokens
  (the current date is supplied as `date_string`, as `strftime_now` is in HF). The batched / TP2 fast profile
  remains Behemoth-only; Llama 3.3 runs single-sequence on one GPU. The ROCm runtime (`behemoth_server`)
  rejects non-Behemoth geometries.
- Memory note: a 2-bit 70B model is ~17.5 GB of trellis data plus ~4.2 GB of bf16 embeddings/head, so the
  single-GPU CUDA runner needs a card larger than 16 GB (e.g. A100 40/80 GB); a 16 GB A4000 only holds the
  synthetic test checkpoints and the quantizer's adaptive evaluation.

**Verified** (RTX A4000, CUDA 12.0): a synthetic Llama-3.3-geometry-scaled checkpoint (hidden 256, FFN 896 =
28*32, 8 q / 1 kv heads, `head_dim` 32, theta 5e5 + llama3 scaling) is generated, quantized (QTIP,
adaptive all-QTIP / all-NVFP4, tiled head) and run end to end; CUDA logits match a CPU reference of the whole
model (relative L2 <= 1.4e-2; the unscaled-RoPE reference differs by ~1.0, so the test is sensitive to the
scaling), and the quantized projections are within the expected 2-bit / NVFP4 distance of the BF16 source.
`tools/llama_synthetic_e2e.sh` chains the quantizer and runtime tests. The `llama3` frequency table and the
RoPE kernels are checked against a literal fp64 transcription of the HF function (real Llama 3.3 parameters
included); 8192x8192, 1024x8192, 28672x8192 and 8192x28672 QTIP projections are checked against the CPU
reference. **Not verified** (no Llama weights or tokenizer on this box): real-checkpoint quality/perplexity,
the real `tokenizer.json`/`chat_template` of the release (a synthetic Llama-3-style tokenizer and a
reproduction of the 3.x template structure are tested; the tool-calling branches of the real template are
not), a full 80-layer / 128256-vocab run and its memory peak on a 16 GB card, multi-shard BF16 reading at
full size, and everything on ROCm.

## Tests

- `build/qtip_tests_cpu tests/refvec`: compares against the committed upstream reference fixtures in `tests/refvec`:
  - Hadamard (hadU/hadUt for 96, 1024, 12288, 28672)
  - lut / _PERMUTE
  - fp64 Viterbi vs `bitshift_codebook.quantize`
  - pack_trellis + kernel swizzle + unpack round trip
  - block LDL vs `block_LDL`, and blocked Cholesky vs naive
- `build/qtip_tests_gpu [--device N] [--bench B] [--shape m n]`:
  - GPU Hadamard/transpose/GEMM layouts
  - GPU Viterbi vs fp64 reference (identical paths, decode(encode) exact, tail-biting)
  - GPU LDLQ vs a CPU port of `ldlq.LDLQ` (bit-identical trellises)
  - end-to-end `quantize_group` with a CPU decode of the packed output
  - `--shape` times one real-sized linear with a synthetic Hessian
- `build/qtip_llama_tests [--device N] [--keep DIR]`: Llama 3.3 support on a synthetic BF16
  Llama-geometry checkpoint: config parsing/rejection, 8192/28672 Hadamard, llama3 RoPE kernel vs fp64 HF
  reference, QTIP and adaptive (all-QTIP, all-NVFP4, tiled lm_head) runs, kv-format handling.
  `--keep DIR` retains the checkpoints for `behemoth_cuda_llama_tests --checkpoint` (see
  `tools/llama_synthetic_e2e.sh`).
- `build/qtip_adaptive_tests [--device N]`: adaptive mixed-format mode on a synthetic 3-layer FP8
  checkpoint — forces every branch (all-QTIP, KL-forced NVFP4 fallback, MSE metric with
  quantized-input chaining) and checks layer tensors, `adaptive_report.json`, `config.json`
  `layer_formats`, and NVFP4 decode fidelity.
- Native runtime projection checks: `behemoth_qtip_tests` verifies decode and forward math;
  `qtip_check` compares selected logits against a captured `behemoth_eval` trace.
  See [fine-tuning validation](docs/FINE_TUNING.md) for commands and pending GPU gates.

See `QTIP_QUANT_REPORT.md` for results and timings.
