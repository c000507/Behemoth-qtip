# qtip_quant — QTIP 2-bit quantizer for Behemoth-R1-123B-v2 on 2x R9700 (ROCm)

Date: 2026-10-26. Repo: `.229:/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/qtip_quant`
(local git, not pushed). README.md covers build, usage, flags, resume and the output format.

## Status in one paragraph

The binary is complete and works end to end: calibration → Hessian collection on both GPUs → QTIP HYB 2-bit
quantization → QTIP-format HF directory, and it is resumable.

What has been validated:
- **Unit level** against the unmodified reference python: Hadamard, codebook/LUT, Viterbi, packing, LDL, and LDLQ
  (bit-identical trellises).
- **Pipeline on a tiny random Mistral checkpoint**, end to end, including resume and reference decode of the
  written files. The QTIP reference's own `decode_compressed` + RHT inverse reproduces the binary's weights to
  1e-7 relative.
- **Real-model forward pass / Hessians** on layers 0–1.

What is **not done**:
- real-layer (layer 40) proxy parity against the python reference;
- measured real per-layer wall times on 1 vs 2 GPUs.

Why: the only stop window was aborted after its first step. GPU0 (33:00.0) logged
`MES(1) failed to respond to msg=INVALIDATE_TLBS` (68 lines, 13:09:15–13:09:27) while qtip_quant was collecting
layers 0–1. Under the GPU-safety rule I stopped all GPU work at that point. Both GPUs are still present, and
behemoth_server was restarted with its exact command line (healthy). The timings below are therefore
measured kernels/stages plus projections. The remaining validation is one scripted command (see "Next window").

## Design

| stage | where / precision | notes |
|---|---|---|
| FP8 weights | GPU | fp8 e4m3 decoded exactly. For the forward pass, the weights are exact-fp8 bf16 and the per-channel scale is applied to the fp32 GEMM output. For quantization, W = fp8·scale in fp32 (exact) |
| calibration | `tools/make_calib.py` | quantbench data choice: 75% OpenWebText, 25% SmolTalk chats (chat template), packed 2048-token sequences, BOS/EOS per document. Default 2048×2048 = 4.19M tokens (`/mnt/nvme/qtip_quant/calib_2048x2048.i32`) |
| forward (Hessian inputs) | GPU, bf16, hipBLASLt | Original model, HF-Mistral semantics: RMSNorm, RoPE θ=1e6 rotate_half, GQA 96/8, causal softmax, SiLU-MLP. Attention uses GEMMs + softmax kernel, 4096-token batches. Activations live in host RAM as bf16 (98 GB for 4.19M tokens) |
| Hessians | GPU, bf16 inputs, fp32 accumulate | H = XᵀX/N for qkv, o, gate/up, down. Only 10 of 16 blocks of a 4×4 block partition are accumulated (symmetry), then mirrored (−37% Hessian FLOPs). Each GPU runs its own layer chain over half the sequences; the partials are summed on the host and saved as `hess/layer<L>.safetensors` (5.1 GB/layer) |
| RHT | GPU fp32 | `matmul_hadUt` exactly as in the reference: natural-order FWHT on 2^p chunks, then the `hadK` mix (12288 = 12·1024, 28672 = 28·1024, 1024 = 2^10; tables generated from the reference's `get_had*`). SU, SV random ±1 |
| damping | GPU | `regularize_H`: diag += 1e-2·mean(diag) |
| block LDL | **CPU fp64** | Blocked right-looking Cholesky with an AVX-512 6×16 microkernel and a vectorised panel solve, then `block_LDL(·,16)` and zeroing of the diagonal blocks. Speed: 12 s at n=28672 and ~1.5 s at n=12288 on 16 threads. It runs in a side thread, so group g+1 is factored while the GPU runs LDLQ on group g |
| Wscale | GPU | rms(Wr) / (rms(fp16 lut)·0.9), per linear |
| LDLQ | GPU | `ldlq.LDLQ`, buf_cols=128, for_kernel `_PERMUTE`. Feedback inside a buffer: fp32 kernel. `prod_cache` update: one hipBLASLt GEMM with K-concatenated bf16 hi/lo splits (≈fp32, ~1e-5 rel) |
| Viterbi | GPU | Two passes per 16×16 tile as in `bitshift_codebook.quantize`: (1) rolled by 128, giving the overlap = state₆₄>>4; (2) tail-biting constrained. One 1024-thread workgroup per trellis; each thread owns 64 states (16 predecessors × 4 successor groups). Per thread in registers: fp16 LUT pairs, \|l\|², fp32 costs via `v_dot2_f32_f16`; argmin index tagged into the low mantissa bits and `v_min3`. Group minima in LDS (double-buffered, one barrier per step). 4-bit argmins stored to global and backtracked by one thread per trellis. 166 VGPRs, no spills. Input rounded to fp16 like the reference; costs fp32 (the reference's are fp16) |
| packing | CPU | `pack_trellis` + the has_kernel byte swizzle, verified bit-exactly against the reference |
| output | — | `model-layer-XXXXX.safetensors`: `trellis` I16, `SU` BF16, `SV` F32 (= sign·Wscale, as `unpack_quip` stores it), `tlut` F16, `rcp`/`tp_rank` I64, layer norms. Plus `model-embed.safetensors` (embed/norm/lm_head bf16), `config.json` (llama + `quip_params`), index, tokenizer files |
| scheduling | 2 GPU worker threads | Each worker prefers a ready layer to quantize; otherwise it advances its own collection chain. `--backlog` (default 2) bounds collected-but-unquantized layers (RAM/disk); a chain may always complete a layer its partner has started (no deadlock). Weights are prefetched from NFS by an LRU layer cache shared by both chains and the quantizer |
| resume | `work/progress.json` | Done layers are skipped, Hessians on disk are quantized, and the chains replay the forward pass (no Hessians) up to the first layer still to collect. Writes are atomic. Verified bit-reproducible: re-collected and re-quantized layers have identical md5 |

Deliberate differences from the reference (all quality-neutral or better; each is noted in `config.json` → `qtip_quant`):
- **One SU per Hessian group.** q/k/v share one SU and one LDL, as do gate/up; the reference draws one SU per
  linear. Given a shared SU and L, LDLQ is row-independent, so this is an exact fusion and halves the LDL work. The
  stored SU is identical for the linears in a group, which the format allows.
- **Viterbi costs and LDL precision.** Viterbi costs are fp32, where the reference uses fp16. LDL is fp64, where the
  reference uses fp32 `torch.linalg.cholesky`. The codebook is fp16, matching the reference with an fp16 model
  (with a bf16 model the reference would use a bf16 LUT).
- **Sign vectors** come from a deterministic splitmix RNG (`--seed`), not torch.randn.

## Validation results

| test | result |
|---|---|
| Hadamard hadU/hadUt, n = 96, 1024, 12288, 28672 (CPU fp64 vs reference) | max diff ≤ 1.3e-7 (the reference itself divides by an fp32 √n) |
| Hadamard GPU fp32 vs fp64 | ≤ 1.3e-6 |
| LUT (quantlut_sym of tlut, fp16) and `_PERMUTE` | identical to reference |
| CPU fp64 Viterbi vs python `bitshift_codebook.quantize` (24 seqs) | 24/24 identical state paths |
| **GPU Viterbi vs fp64 Viterbi (48 seqs, 3 scales)** | 48/48 identical paths, cost excess 0; tail-biting and trellis links hold; decode(encode(x)) == returned values exactly |
| pack_trellis + kernel swizzle vs reference; unswizzle/unpack round trip | bit-identical / exact |
| block LDL vs reference `block_LDL` (n=96); blocked Cholesky vs naive (n=1000) | 1.5e-8 (fp32 output) / 4.6e-15 |
| **GPU LDLQ vs CPU port of `ldlq.LDLQ` (m=64, n=256)** | proxy identical (0.053032 both); 64/64 trellises bit-identical |
| `quantize_group` end to end + CPU decode of the packed output | decoded proxy = reported proxy (5 digits) |
| **Tiny random Mistral checkpoint, full pipeline** (3 layers, 2 GPUs): reference `decode_compressed` + reference `matmul_hadU` on the written files vs the binary's own reconstruction | rel. Frobenius diff 8–10e-8 on all 21 linears; reference `QuantizedLinear.forward` vs x·Ŵᵀ: 3e-4 (fp16 internals); proxy from the reference decode = proxy reported by the binary |
| resume (delete outputs/Hessians, rerun) | replays the chain, re-collects only what is missing, bit-identical outputs |
| **Real model, layers 0–1, 2×2048 tokens** (`tools/check_forward.py`, fp32 torch reimplementation on CPU) | activations rel err 0.59% / 0.54% (bf16 level); Hessian rel Frobenius err: qkv 0.14% / 0.03%, o 0.21% / 0.54%, gate/up 0.10% / 0.22%, down 1.3% / 0.42% |
| **Real layer 40 vs reference QTIP** (proxy within 1–2%) | **not run** (GPU stop, see top). The previous agent's reference numbers (32k-token Hessians, which are deleted) were: k 0.0022, q 0.0027, o 0.0123, down 0.0185 |

Real-shape run with a synthetic Hessian (`qtip_tests_gpu --shape 1024 12288`, i.e. k_proj size, H from 4096
correlated samples): proxy 0.0062. This cannot be compared numerically with the real k_proj, since the Hessian is
different. It shows the full path at real n.

## Timings

All GPU numbers were measured while behemoth_server was running on the same GPU (it held about 90% of VRAM and
the stability test was driving it), at the earlier −500 MHz OD setting. Since 13:00 the cards are pinned to DPM
level 1 (~2.1–2.3 GHz), so expect roughly 0–20% slower kernels now.

Measured:

| piece | number |
|---|---|
| Viterbi, 2 passes, 2048 trellises | 4.5 µs per trellis-pass aggregate → **28.3 M weights/s per GPU** (pass 1 + pass 2) |
| LDLQ incl. Viterbi, 1024×12288 | 0.46 s (Viterbi 0.42 s) → 27 M weights/s |
| RHT of W and H at n=12288 | 0.37 s |
| fp64 Cholesky (16 threads) | n=8192 0.53 s, n=28672 12.0 s (0.66 TFLOPS); runs on the CPU, overlapped |
| forward + Hessians, real layer 1, 4096 tokens | 0.22 s → **18.3 k tokens/s per GPU**, 81.5 TFLOPS effective (bf16 hipBLASLt) |
| layer weights over NFS (1.38 GB fp8) | 28.4 s cold (~49 MB/s; prefetched one layer ahead) |
| Hessian file write (5.1 GB/layer) | 4.6–4.8 s |

Projected per layer (Behemoth layer = 1.384 G weights):

| | 1 GPU | 2 GPUs |
|---|---|---|
| quantization (Viterbi ≈ 49 s + RHT/proxy/pack/IO ≈ 10 s; LDL hidden on CPU) | ~60 s | ~30 s/layer throughput (one layer per GPU) |
| Hessian collection, 4.19M tokens | ~230 s | ~115 s (token halves in parallel) |
| **total per layer** | **~290 s** | **~145 s** |
| **full model, 88 layers** | **~7 h** | **~3.5–4 h** |

For comparison, the python reference is ≈25 min per layer per GPU for quantization alone (≈23 h on 2 GPUs). That
makes the quantization step ≈25× faster. Collection now dominates, and with fewer calibration tokens (`--ntok`) the
total shrinks proportionally:

- 1M tokens: ≈1.6 h on 2 GPUs (projected).
- The paper/QTIP default is 6144 × 4096 = 25M tokens. This box cannot hold those activations in RAM; ≈4M tokens is
  the practical setting here.

Memory and disk:
- **GPU** (projected from the buffer sizes): ~10.5 GB for the forward + Hessian accumulators, plus ~12 GB peak for
  the gate+up group. Fits in 32 GB, but only with behemoth_server stopped.
- **Host RAM**: 98 GB activations (4.19M tokens) + 5–15 GB Hessians + ~6 GB layer cache.
- **Disk**: Hessian backlog ≤ 2 layers × 5.1 GB; output ≈ 30.4 GB (2 bpw, about 345 MB per layer) + 1.6 GB
  embeddings/lm_head. Total under 45 GB, within the 60 GB budget.
- **Currently on /mnt/nvme/qtip_quant**: 17 MB (calibration tokens + logs). All temporaries have been removed.

## Next window (remaining validation, ~45 min, one command)

```
cd /home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/qtip_quant
setsid nohup bash tools/window_validate.sh "2 3 4 5 6" > /dev/null 2>&1 < /dev/null &   # log: /mnt/nvme/qtip_quant/window.log
```

The script:
- **Before it starts:** stops the server by exact PID and records its command line.
- **Before every step:** checks GPU presence, DPM `manual` / not on the top level, and new MES errors in dmesg.
- **On exit (any exit):** restarts the server with the identical command line.

Steps:
- **2**: layer 40/41 Hessians, 512k tokens, one GPU.
- **3**: layer 40 on 1 GPU, with dumps.
- **4**: layers 40+41 on 2 GPUs.
- **5**: reference QTIP on the same Hessians. q/k/v/o full; MLP on 4096-row subsets.
- **6**: full pipeline with 4.19M tokens, layers 0–1, both GPUs.

After it:

```
QTIP_DIR=/mnt/nvme/quantbench/qtip /home/charles/venvs/quantbench/bin/python tools/check_decode.py \
  --out /mnt/nvme/qtip_quant/q1/out --layer 40 --dump /mnt/nvme/qtip_quant/q1/dump \
  --hess /mnt/nvme/qtip_quant/par/hess/layer40.safetensors --forward
```

Then compare `q1/work/quant_log.jsonl` with `ref_results.jsonl`.

Before running it, someone should look at the GPU0 MES messages. They appeared first at 12:37 during the −50 mV
test, then again at 13:09 during qtip_quant step 1. That step was a short single-GPU forward of 4096 tokens plus
hipMalloc/hipFree of the Hessian buffers. The messages could be a lingering firmware state on GPU0 since the 12:37
hang, rather than something specific to this workload.

## What an inference runtime needs for this format

The ROCm runtime (cpp_runtime) currently decodes AQLM codebooks. Supporting this format means a QTIP decoder:

- **Per linear:**
  - Fold `x ⊙ SU`, then a fast Hadamard on the input: `hadUt_n` = FWHT-1024 per chunk + the 12- or 28-point mix.
  - Matmul with the trellis-decoded weight.
  - `hadU_m` on the output, then `⊙ SV`.
- **Trellis decode:** each 16×16 tile is 32 uint16 words, i.e. a 512-bit tail-biting stream. Weight pair t is
  `lut[(stream >> 4t) & 0xffff]` as a 16-bit window with wrap-around, where
  `lut[s] = (±tlut[((s(s+1))>>6)&511][0], tlut[…][1])`, sign from bit 15 of s(s+1).
- **Element ordering:** follow `_PERMUTE` and the 2×2-tile word interleave of the kernel layout (exactly
  `lib/utils/kernel_decompress.decode_compressed`, or QTIP's CUDA `decompress_matvec_16_9_2_1` kernels).
- **Implementation choices:**
  - decode-to-bf16 tiles + WMMA for prefill;
  - a fused decode-GEMV for decode, with the 512×2 fp16 tlut in LDS (≈2 bit/weight of reads);
  - Hadamard kernels for 12288 and 28672 on both sides of every linear.
- **No trainable components** (no FT): SU/SV/tlut are as quantized. Layer-wise / end-to-end fine-tuning would be
  later work.
