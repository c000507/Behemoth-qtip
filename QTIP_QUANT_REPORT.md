# qtip_quant — QTIP 2-bit quantizer for Behemoth-R1-123B-v2 on 2x R9700 (ROCm)

Date: 2026-10-26. Repo: `.229:/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/qtip_quant`
(local git, not pushed). README.md covers build, usage, flags, resume and the output format.

## Status in one paragraph

The binary is complete and works end to end: calibration → Hessian collection on both GPUs → QTIP HYB 2-bit
quantization → QTIP-format HF directory, and it is resumable.

What has been validated:
- **Unit level** against the unmodified reference python: Hadamard, codebook/LUT, Viterbi, packing, LDL, and LDLQ
  (bit-identical trellises).
- **Pipeline on a tiny random Mistral checkpoint**, end to end, including resume and reference decode.
- **Real-model forward pass / Hessians** on layers 0–1.
- **Real layer 40, quantized from 512k-token Hessians, against the python reference on the same Hessians:**
  - the proxy error matches within 0.5% on every linear;
  - the reference decoder reproduces the written weights to 1.6e-7 relative;
  - the reference `QuantizedLinear` forward agrees to 3e-4 (fp16 internals).
- **Measured wall times**, at DPM level 1 (2.1–2.3 GHz) with behemoth_server stopped:
  - 1 layer quantized on 1 GPU in 81–92 s;
  - 2 layers on 2 GPUs in 82.6 s (41 s/layer throughput);
  - Hessian collection at 18.7 k tokens/s per GPU.

What is **not done**: step 6, the 2-GPU full-pipeline run with all 4.19M tokens.
- Its first attempt (14:16) was cut by a power loss of the whole box; see the MES section.
- After the reboot the model's NFS export (192.168.0.130:/mnt/evo) is empty, so the run cannot read the checkpoint.
- Its 4.19M-token per-layer time below is therefore projected from the measured 512k-token throughput.

**MES `INVALIDATE_TLBS` errors: root-caused and fixed in our code** (commit `ca70f68`, details below and in
`refusal_bench/GPU0_MES_INVALIDATE_TLBS.md`):
- They came from large device→host copies into pageable `std::vector`s: the Hessian download, plus the HRr and
  trellis downloads in the quantizer.
- They are not specific to GPU0.
- All such copies now go through a pinned staging pair. Step 1 went from 35 MES lines to 0, and steps 2–5 had 0.

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
| **Real layer 40 vs reference QTIP**, same 512k-token Hessians | proxy within 0.5% on all 7 linears; the reference decoder reproduces the written weights to 1.6e-7 (see "Validation results — real layer 40") |

Real-shape run with a synthetic Hessian (`qtip_tests_gpu --shape 1024 12288`, i.e. k_proj size, H from 4096
correlated samples): proxy 0.0062. This cannot be compared numerically with the real k_proj, since the Hessian is
different. It shows the full path at real n.

## MES `INVALIDATE_TLBS` errors (root cause and fix)

Symptom: bursts of `amdgpu 0000:33:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS`, seen during step 1
(13:09 and 13:41).

**Cause:** a large `hipMemcpy` device→host into pageable, CPU-resident memory, here a freshly `resize()`d
`std::vector`.
- HIP locks those pages on the fly through KFD SVM-range ioctls.
- On gfx1201 with amdgpu DKMS 7.1.9 and MES fw 0x93, MES answers some of the resulting TLB invalidations with a
  failure status, roughly every 1–10 ms during the copy (this is not a timeout).
- `gmc_v12_0_flush_gpu_tlb_pasid` ignores the error and has no fallback.

**Evidence:**
- strace plus kmsg markers put every burst inside `deposit_partial`'s downloads of the 4 Hessian accumulators
  (604 MB ×3 + 3.3 GB per layer).
- A minimal HIP test (`/mnt/nvme/qtip_quant/mes/mes_test.cpp`) reproduces it:
  - D2H of 3 GB into a zeroed vector: 13 lines on GPU0, **5 on GPU1** (so not card-specific);
  - 0.56 GB: 1 line; 50 MB: 0;
  - a re-used vector: only its first copy;
  - zero lines for H2D from pageable memory, D2H into `hipHostMalloc`, `hipHostRegister`/`Unregister`,
    `hipMalloc`/`hipFree`, and D2H into untouched mmap.
- PCIe AER counters on 33:00.0 and its root port are zero.

**Fix** (`ca70f68`): `d2h_staged()` (src/staging.cpp), a per-thread double-buffered 2×64 MB pinned staging pair.
- Used for the Hessian download and for the quantizer's HRr, states and dump downloads.
- Step 1: 35 lines → **0** in 4 runs (GPU0 ×3, GPU1 ×1), with md5-identical Hessians.
- Steps 2, 3, 4 and 5 then ran with 0 new lines.

**Upstream:**
- Same message in [ROCm/TheRock#9299](https://github.com/ROCm/TheRock/issues/9299) and
  [#9279](https://github.com/ROCm/TheRock/issues/9279) (R9700 / gfx1201, open, no fix).
- The amd-gfx "Rework GPU TLB invalidation" series (Sep 2026) drops the gmc12 MES TLB-invalidation path in favour
  of SDMA. It is not in DKMS 7.1.9.

## Validation results — real layer 40 (validation window 13:58–14:57)

Setup:
- Hessians for layers 40/41 from 512k calibration tokens (256×2048), collected on GPU0 (step 2).
- Layer 40 quantized on 1 GPU with dumps (step 3); layers 40+41 on 2 GPUs (step 4).
- The python reference QTIP (`tools/ref_quantize.py`, unmodified QTIP code) on the same Hessian file (step 5).
- Reference decode of the written file (`tools/check_decode.py --forward --hess`, CPU).
- Both python tools read W from the binary's `W.f32` dumps (`--wdump`, new). These are exactly fp8·scale in fp32,
  bit-identical to decoding the checkpoint, because the NFS model mount was gone after the power loss.

Proxy error tr(ΔHΔᵀ)/tr(WHWᵀ), unregularised H, original domain:

| linear | shape | qtip_quant | python reference | Δ | ref decode of written file: proxy / rel. diff vs binary hatW / forward rel. err |
|---|---|---|---|---|---|
| q_proj | 12288×12288 | 0.004004 | 0.004005 | −0.02% | 0.004004 / 1.6e-7 / 2.9e-4 |
| k_proj | 1024×12288 | 0.003127 | 0.003129 | −0.05% | 0.003127 / 1.3e-7 / 2.9e-4 |
| v_proj | 1024×12288 | 0.030882 | 0.030885 | −0.01% | 0.030882 / 1.3e-7 / 2.9e-4 |
| o_proj | 12288×12288 | 0.024279 | 0.024278 | +0.00% | 0.024279 / 1.6e-7 / 2.9e-4 |
| gate_proj | 28672×12288 | 0.013521 (same 4096 rows: 0.013692) | 0.013624 (4096 rows) | +0.50% | 0.013521 / 1.6e-7 / 2.9e-4 |
| up_proj | 28672×12288 | 0.030481 (same rows: 0.030625) | 0.030635 (4096 rows) | −0.03% | 0.030481 / 1.6e-7 / 2.9e-4 |
| down_proj | 12288×28672 | 0.040284 (same rows: 0.038011) | 0.038076 (4096 rows) | −0.17% | 0.040284 / 1.6e-7 / 2.9e-4 |

For the MLP:
- The reference ran on a random 4096-row subset (`--rows 4096`, seed 1234), for time.
- The binary's proxy is restated on the same rows (`mes/subset_proxy.py`).
- The binary quantized the full matrix, so its SU, SV and Wscale differ from the reference's. A ±0.5% difference
  is therefore at the level of RNG noise.

The other checks:
- **The decoder** (`decode_compressed` + `matmul_hadU`, SU, SV) reproduces the binary's dumped hatW to 1.3–1.6e-7
  relative, i.e. fp32 rounding, on all 7 linears.
- **`proxy_ref_decode`** equals the proxy the binary reported, to ≥4 digits.
- **Parity of the layer quantization (steps 3 vs 4).** Layer 40 got identical proxies on 1 GPU (step 3) and on
  2 GPUs (step 4).
- **The earlier reference run** with 32k-token Hessians (from a previous agent; not comparable) had k 0.0022,
  q 0.0027, o 0.0123, down 0.0185. With 512k tokens the Hessian is better conditioned, and the error is higher but
  more representative.

## Timings (measured at DPM level 1, behemoth_server stopped)

| piece | measured |
|---|---|
| forward only (no Hessians), 524288 tokens/layer, 1 GPU | 20.7 s/layer → **25.3 k tok/s**, 72.6 TFLOPS |
| forward + Hessians, 524288 tokens, 1 GPU (layers 40, 41) | 28.1 / 27.8 s → **18.7–18.8 k tok/s**, 83–84 TFLOPS |
| Hessian D2H + host sum + save (5.1 GB/layer, staged) | ~10 s per layer (save 4.7 s) |
| quantize 1 layer, 1 GPU (step 3, incl. dumps) | **91.5 s** (qkv 13.3, o 11.4, gate/up 42.4, down 41.1; Viterbi 50 s of it) |
| quantize layers 40+41, 2 GPUs concurrently (step 4) | **82.6 s for 2 layers** (81.4 / 82.4 s each) → 41 s/layer |
| python reference, layer 40 on 1 GPU | attention 373 s; MLP extrapolated from the 4096-row subsets ≈ 1100 s → ≈ 25 min/layer, **≈17× slower** |
| CPU fp64 LDL, n = 28672 (down) | 17–22 s; it shows up inside the down group, while the LDLs for n = 12288 (2.3–2.7 s) are mostly hidden |
| layer weights from NFS (1.38 GB fp8) | ~26 s cold (step 1), prefetched one layer ahead; 0 s warm |

Not measured: step 6 (Hessian collection with all 4.19M tokens on 2 GPUs). It was interrupted by the power loss at
14:16, and could not be repeated because the NFS export was empty after the reboot. Collection is linear in tokens
(the GEMMs dominate; 512k already runs at 84 TFLOPS), so from the measured 18.7 k tok/s per GPU:

| | 1 GPU | 2 GPUs |
|---|---|---|
| Hessian collection, 512k tokens | 28 s + ~10 s save (measured) | ~14 s + ~10 s (projected) |
| Hessian collection, 4.19M tokens | ~224 s + ~10 s | ~112 s + ~10 s (token halves in parallel) |
| quantization | ~82–92 s | ~41 s/layer throughput |
| **per layer, 4.19M tokens** | **~320 s** | **~160 s** (≈ (2×112 + 82 + 10) GPU-s / 2) |
| **full model, 88 layers, 4.19M tokens** | **~7.8 h** | **~3.9 h** |
| full model, 1M tokens | ~3.1 h | ~1.6 h |

On 2 GPUs, the earlier 3.5–4 h projection is confirmed. Collection is ~70% of the time.

Memory and disk:
- **Host RAM**: 103.1 GB activations at 4.19M tokens (measured at the step-6 start).
- **Disk**: Hessians 5.1 GB per layer; output 346.7 MB per layer (≈30.5 GB for 88 layers) + embeddings.
- **GPU**: fits with behemoth_server stopped (steps 2–4 ran fine).

## Remaining

Step 6, once 192.168.0.130:/mnt/evo is back:

```
bash /mnt/nvme/qtip_quant/window_noserver.sh "6"     # 2 GPUs, layers 0-1, 4.19M tokens; log: /mnt/nvme/qtip_quant/window.log
```

Watch the box during it. It is the first sustained dual-GPU forward pass, and the 14:16 power loss happened at its
start.

Full run:

```
./build/qtip_quant --tokens /mnt/nvme/qtip_quant/calib_2048x2048.i32 \
    --out /mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-2bit --work /mnt/nvme/qtip_quant/work --gpus 0,1
```

It takes ~4 h with the server down.

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
