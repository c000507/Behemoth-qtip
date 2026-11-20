# Qwen3.5 MoE (`qwen3_5_moe`) on trn1.2xlarge, TP=2: design

Target checkpoints: `Qwen/Qwen3.5-35B-A3B` (256 experts) and REAP-pruned variants such as
`Ttimms/Ornith-1.5-35B-A3B-REAP-50-bf16` (128 experts in every layer, 19.0 B parameters, bf16, one `model.safetensors`,
no MTP tensors). Facts below were read from the two `config.json`s and from the safetensors headers (HTTP range requests
on the header only; the checkpoint itself is never on the dev host - it goes HF -> S3 inside the account).
The reference semantics are `transformers` 5.14.1 `models/qwen3_5_moe/modeling_qwen3_5_moe.py` (available locally in the
system python; the Ornith config was written by transformers 5.15.1).

Status log (what exists / was validated) is at the end of this file.

## 1. Architecture (from config + HF modeling code)

| | value |
|---|---|
| layers | 40, `layer_types` = (3 x `linear_attention`, 1 x `full_attention`) x 10 -> 30 Gated DeltaNet layers, 10 GQA layers |
| hidden | 2048, RMSNorm eps 1e-6, **weights stored zero-centred: `y = norm(x) * (1 + w)`** (Qwen3-Next convention; applies to input/post-attention norms, q_norm/k_norm, final norm; NOT to `linear_attn.norm` which is a plain gated RMSNorm `y = norm(x) * w * silu(z)`) |
| Gated DeltaNet | 16 key heads x 128, 32 value heads x 128 (query/key heads `repeat_interleave(2)` onto value heads: value heads 2i, 2i+1 share key head i), `in_proj_qkv` [8192, 2048] (rows: q 0..2047, k 2048..4095, v 4096..8191), `in_proj_z` [4096, 2048] (output gate), `in_proj_b` [32, 2048] (beta = sigmoid), `in_proj_a` [32, 2048] (`g = -exp(A_log) * softplus(a + dt_bias)`), depthwise causal `conv1d` [8192, 1, 4] over the qkv channels + SiLU, `A_log` [32], `dt_bias` [32], `norm.weight` [128], `out_proj` [2048, 4096]. q and k are L2-normalised per head (eps 1e-6), q scaled by 1/sqrt(128). Recurrence per value head: `S = S*exp(g); mem = S^T k; S += k (v - mem)^T beta; o = S^T q` (S is [Dk, Dv] f32). |
| full attention | 16 q heads x 256, 2 KV heads x 256, `q_proj` [8192, 2048] = per head 512 rows = 256 q then 256 **output gate** (`attn_output_gate`: `o = attn * sigmoid(gate)` before `o_proj`), `k_proj`/`v_proj` [512, 2048], `o_proj` [2048, 4096], `q_norm`/`k_norm` [256] per head (1+w), RoPE on the **first 64 dims only** (`partial_rotary_factor` 0.25, rotate_half within those 64: pairs (i, i+32)), theta 1e7; mRoPE sections collapse to plain 1-D RoPE for text (all three position streams are equal), scale 1/16. |
| MoE (every layer) | router `gate.weight` [E, 2048]: **softmax over all E experts, top-8, renormalise** (no bias, no groups, no scaling factor); experts stored fused `experts.gate_up_proj` [E, 1024, 2048] (gate rows 0..511, up 512..1023) and `experts.down_proj` [E, 2048, 512]; SiLU(gate) * up; shared expert `shared_expert.{gate,up,down}_proj` (512 wide) scaled by `sigmoid(shared_expert_gate.weight [1, 2048] . x)`; output = routed + shared. |
| vocab / head | 248320, `embed_tokens` and `lm_head` untied, bf16 |
| MTP | `mtp.*` tensors exist in the Qwen checkpoint (`mtp_num_hidden_layers` 1, per-expert `mtp.layers.0.mlp.experts.N.*`), none in Ornith (0): ignored |
| vision | `model.visual.*`: ignored (text only) |

Tensor names (prefix `model.language_model.layers.N.`): `input_layernorm.weight`, `post_attention_layernorm.weight`,
`linear_attn.{A_log, conv1d.weight, dt_bias, in_proj_a.weight, in_proj_b.weight, in_proj_qkv.weight, in_proj_z.weight,
norm.weight, out_proj.weight}`, `self_attn.{q_proj, k_proj, v_proj, o_proj}.weight`, `self_attn.{q_norm, k_norm}.weight`,
`mlp.gate.weight`, `mlp.experts.{gate_up_proj, down_proj}` (3-D, no `.weight` suffix), `mlp.shared_expert.{gate,up,down}_proj.weight`,
`mlp.shared_expert_gate.weight`; plus `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight`, `lm_head.weight`.
Both checkpoints use this fused layout (the Qwen one is 14 shards + index; Ornith is one 19 GB file whose header is 132 KB).

## 2. Op inventory per layer type (decode, one token per row)

Linear-attention layer: RMSNorm(1+w) -> qkv / z / a / b projections (4 matmuls, one activation) -> conv1d step (3 old
columns from the conv state + new column, 4-tap dot per channel, SiLU; conv state shift) -> per-head L2-norm of q, k ->
beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias) -> recurrence (per value head: scale S, 2 small matvecs, 1 outer
product) -> gated RMSNorm(w) * silu(z) -> out_proj (row-parallel, all-reduce) -> residual -> RMSNorm(1+w) -> MoE.

Full-attention layer: RMSNorm -> q|gate, k, v projections -> per-head RMSNorm(1+w) of q and k -> partial RoPE (64 dims) ->
KV scatter into the paged cache -> paged attention over the request's pages (GQA 8:1 per rank) -> * sigmoid(gate) ->
o_proj (row-parallel, all-reduce) -> residual -> RMSNorm -> MoE.

MoE (both): router matmul [T, 2048] x [2048, E] f32 -> softmax -> top-8 -> renormalise (replicated on both ranks, on the
device, exact and identical on both ranks because both run the same NEFF on the same input) -> gather the 8 selected
experts' slices (index_select on the expert axis of the resident [E, ...] tensors) -> gate_up matmul [T, 2048] x
[2048, 8 x 2 x 256] -> SiLU * up * combine weight -> down [T, 8 x 256] x [8 x 256, 2048] (folded into one contraction, as
`glm_graphs.moe_block`) -> + shared expert (TP-split 256 wide, gate scalar replicated) -> one all-reduce of the partial sums.
With T rows the gather has `min(8 T, E)` slots; the prefill graphs run every local expert (`MOE=all`).

## 3. Sharding under TP=2 (every rank runs the same NEFF; rank data are inputs)

| tensor | per-rank slice (rank r) | shape per rank |
|---|---|---|
| `in_proj_qkv` | q rows [1024 r, 1024 (r+1)), k rows 2048 + same, v rows 4096 + [2048 r, 2048 (r+1)) -> key heads [8r, 8r+8), value heads [16r, 16r+16) (consistent with the interleaved q/k repeat) | [4096, 2048] |
| `conv1d.weight` | the same 4096 channels | [4096, 4] f32 |
| `in_proj_z` | rows [2048 r, 2048 (r+1)) | [2048, 2048] |
| `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias` | heads [16 r, 16 r + 16) | [16, 2048], [16] |
| `linear_attn.norm.weight` | replicated | [128] |
| `out_proj` | columns [2048 r, 2048 (r+1)) (row-parallel, all-reduce) | [2048, 2048] |
| `q_proj` | rows [4096 r, 4096 (r+1)) = q heads [8r, 8r+8) with their gates | [4096, 2048] |
| `k_proj`, `v_proj` | rows [256 r, 256 (r+1)) = KV head r (no replication needed: kv_heads == TP) | [256, 2048] |
| `q_norm`, `k_norm` | replicated | [256] |
| `o_proj` | columns [2048 r, 2048 (r+1)) | [2048, 2048] |
| router `gate.weight` | replicated, f32 (0.26 / 0.52 M params) | [E, 2048] |
| `experts.gate_up_proj` | **expert tensor parallel (ETP = 2, EP = 1)**: gate rows [256 r, 256 (r+1)) and up rows 512 + same of EVERY expert | [E, 512, 2048] |
| `experts.down_proj` | input channels [256 r, 256 (r+1)) of every expert, stored transposed [E, 256, 2048] with per-input-channel scale (so the 8 experts fold into one K = 2048 contraction) | [E, 256, 2048] |
| shared expert | gate/up rows [256 r, ...), down columns likewise; `shared_expert_gate` replicated | [256, 2048] x 2, [2048, 256] |
| `lm_head` | vocabulary rows [124160 r, ...) | [124160, 2048] |
| embeddings | host (bf16 table, 1.02 GB host RAM, as the GLM backend) | - |

Why ETP rather than EP for the experts: with 2 ranks and top-8 the gathered-expert graph has a static slot count. Under
EP (64 experts per rank) a rank may receive all 8 selected experts, so each rank must gather 8 full experts (worst case)
and reads 8 x 3.1 MB; under ETP each rank reads 8 half-experts (8 x 1.6 MB), always balanced, and the memory footprint is
identical. ETP also keeps the all-reduce count at one per MoE (shared + routed partials). The GLM MoE code already
implements exactly this layout (`EP=1`, `MOE=gather`), so the Qwen graphs reuse `moe_block` with a softmax router.

## 4. Decode-step state

* Linear layers (30): per layer and rank, `kstate` f32 [slots, 16, 128, 128] (1 MiB per slot) and `cstate` f32
  [slots, 4096, 3] (48 KiB per slot: the last 3 qkv columns). One slot per live sequence, keyed by its first page, LRU
  spill to host RAM (the GLM backend's `acquire_slot` / `copy_slot`), aliased in/out in the NEFF, zeroed by the graph
  when the row's position is 0. 31.4 MiB per sequence per rank over the 30 layers; 32 slots = 1.0 GiB per rank.
* Full-attention layers (10): the server's paged KV, `[pages, 2 (K|V), 1 kv head, 16, 256]` bf16 per layer and rank
  = 16 KiB per page per layer, **10 KiB per token per rank over the model**. 2048 pages (32k tokens) = 320 MiB per rank;
  the 262k context of the config would be 2.5 GiB. FP8 KV is not needed for memory here (and the bf16 NKI decode kernel
  is the validated one); the KV gather at long context is 10 layers instead of 32-40, so it matters less than for Llama.
* Step inputs: `x` f32 [T, 2048], `pos` i64 [T], `slot` i64 [R] (linear layers), `wslot` i64 [T] and `table` i64 [R, P]
  (attention layers), `cos_sin` f32 [T, 128] or computed in-graph from `pos` (64 rotary dims; the host path of
  trace_model.py is reused: fp64 cos/sin per absolute position).

## 5. Memory per rank (16 GiB HBM per NeuronCore)

Per-rank parameters per layer: linear-attention block 16.8 M, full-attention block 13.6 M, experts E x 1.573 M
(= 3.146 M per expert / 2 ranks), shared expert 1.57 M, router E x 2048. Whole model (Ornith, E = 128): 19.0 B
parameters = the 38.0 GB bf16 `model.safetensors` (37 981 269 704 bytes in S3) -> ~19 GB fp8 in total, ~8.8 B parameters
per rank at TP = 2 (experts split along their intermediate axis, embeddings on the host, lm_head by vocabulary rows).

| variant | weights, 40 layers | lm_head shard | KV (2048 pages) | GDN slots (32) | total | fits? |
|---|---|---|---|---|---|---|
| 128 experts, bf16 | 16.33 GiB | 0.47 GiB | 0.31 | 1.0 | 18.1 GiB | **no** |
| 128 experts, fp8 (E4M3 + row scales, the stack's format) | 8.17 GiB | 0.24 GiB (fp8 head) | 0.31 | 1.0 | **9.7 GiB** | **yes** (6 GiB spare: more pages / slots) |
| 256 experts, bf16 | 31.4 GiB | | | | | no |
| 256 experts, fp8 | 15.68 GiB | 0.24 | 0.31 | 1.0 | 17.2 GiB | **no** (experts alone 15.0 GiB) |
| 256 experts, experts NVFP4 (0.5 B/weight + scales) + fp8 rest | ~8.5 GiB | 0.24 | 0.31 | 1.0 | ~10 GiB | yes, but needs the NVFP4 NKI expert path (`nki_linear.py` exists for dense layers only) |

So on trn1.2xlarge the **REAP-128 variant in fp8 is the only configuration that fits with the existing formats**; the
full 256-expert model needs sub-8-bit experts (or TP >= 4 on a bigger instance). Everything below is designed for both
E values (E is a geometry parameter) but will only be run with E = 128.

## 6. Expected decode cost (batch 1)

Bytes touched per token per rank (fp8): 30 x (16.8 + 8 x 1.573 + 1.57 + 0.26) MB + 10 x (13.6 + 12.6 + 1.57 + 0.26) MB
= **1.22 GB of layer weights + 254 MB lm_head shard** (~3.0 B active parameters model-wide: 1.0 B routed, 0.13 B shared,
0.6 B attention/linear, 0.5 B head, plus the host embedding), independent of E (routing picks 8 either way).

| | fp8 | bf16 (if it fitted) |
|---|---|---|
| layer DMA floor at the measured per-core 214 / 257 GB/s wall | 5.7 / 4.7 ms | 11.4 / 9.5 ms |
| head | 1.2 / 1.0 ms | 2.4 / 2.0 ms |
| DMA-floor tokens/s | **~145-175** | ~70-85 |

The floor will not be reached at first: the GLM measurements on this box put the fixed cost of a sub-block launch with
many small ops at ~0.5 ms (routing, top-k, gathers, the recurrence's small matmuls, the all-reduce 12-22 us) and a fused
layer at ~1.0-1.5 ms beyond its weight streaming. Realistic first numbers: 40 layers x (0.14 ms weights + 0.5-1.0 ms fixed)
+ head ~ **25-45 ms/token (22-40 tok/s)**; with the small ops fused into NKI kernels (routing + gather, conv step +
recurrence) and 2-4 layers per NEFF, 10-15 ms/token (65-100 tok/s) is the plausible ceiling on this instance. The gather
of 8 expert slices is 8 DMAs of 1 MB (gate|up) + 8 of 0.5 MB (down) per rank per layer: large contiguous rows, which the
DMA wall numbers were measured on. Prefill is dominated by `MOE=all` (every expert on every token: 201 MB per rank per
layer at E = 128 -> ~1 ms per layer at the wall, so a 16-token chunk costs about a decode step, as for GLM).

## 7. What must change in the engine (vs. the GLM backend, which is the template)

1. **Graphs** (`neuron/qwen35_graphs.py`, modelled on `glm_graphs.py`): kinds `gdn` (linear attention), `gqa` (paged
   attention), `moe`, fused `layer_gdn_moe` / `layer_gqa_moe`, `head`; no mHC (x is [T, 2048] f32); norms `1 + w`;
   softmax router; Qwen's gated RMSNorm with `silu(z)`; q/k per-head norms + partial RoPE + attention output gate; conv
   state [channels, K-1]; recurrence as in `kda_block` without the KDA-specific f/g projections. Decode buckets R rows,
   prefill R = 1 x Q (token loop in-graph for the recurrence, as GLM; a chunked/WY prefill kernel is later work).
2. **Prepared format** (`neuron/qwen35_prepare.py`, modelled on `glm_repack.py`): `prepared.json` with
   `format: qwen35-moe-neuron-prepared`, `embed.bin`, `r{rank}/L{layer}.bin`, `r{rank}/head.bin`; tensors named as the
   graphs' `weight_shapes`; fp8 row-scaled E4M3 (`WT=1` transposed [in, out] for the dense projections where the
   runtime's tiled/transposed path is used, experts kept [E, out, in] because they are gathered by expert then used as
   one 2-D matmul). Reads the fused 3-D expert tensors directly (slices per rank along the intermediate axis). Norm weights
   are stored as `1 + w` so the graphs use a plain multiply.
3. **Runtime** (`neuron/cpp/qwen35_model.cpp` behind `Backend`, or `glm_model.cpp` generalised: it already does
   state slots, paged caches, by-name binding, fused/sub-block plans, the head and spill/restore). Differences: streams = 1,
   attention kinds `gdn`/`gqa` instead of `kda`/`mla`, KV state is the K|V page tensor (one aliased state per gqa layer),
   `cos_sin` step input, `is_qwen35_prepared` dispatch in `NeuronModel`, `recurrent()` already disables speculation and
   the prefix cache for any `Backend`. Tokenizer: the Qwen `tokenizer.json` (byte-level BPE, 248k ids, HF `tokenizers`
   format with `<|im_start|>` chat template) - `llama_text.cpp` loads Llama-3 / Tekken `tokenizer.json`; whether its
   pre-tokeniser regex matches Qwen's must be checked (open item).
4. **Routing on device**: yes (one-hot top-k + `index_select` gather inside the NEFF, exact and identical on both ranks).
   Host routing would need a device->host->device round trip per layer (40 per token) and is ruled out.
5. **Weights do not fit on the box's disk** (3.7 GB free; the fp8 prepared set is ~17 GB for two ranks, the bf16 source
   19 GB): the prepare step must run where the checkpoint is (S3 -> a host with disk or RAM: the box has 32 GB RAM, one
   layer at a time is ~0.9 GB bf16) and the loader must **stream `r{rank}/L{layer}.bin` from S3 into HBM one layer at a
   time** (no local copy; 17 GB at ~100 MB/s is ~3 min of load) - or the data volume is resized / an instance store
   attached (**decision for the supervisor; not done here**). Graph work uses random weights of the real shapes.

## 8. Risks / open questions

* `torch.topk` on [T, E] with E = 128/256 and the one-hot formulation traced fine for GLM (E up to 288); the softmax
  router is simpler than noaux_tc.
* The recurrence at 16 heads x 128 x 128 f32 per rank is 1 MiB of state read + written per layer per row: 30 layers =
  60 MiB per token of state traffic per rank, ~0.3 ms at the wall - acceptable.
* Prefill recurrence as an in-graph token loop over Q = 16 is what GLM does; Q = 64/128 unrolls to large graphs.
* The 256-expert model does not fit in fp8 (section 5).
* Numerics: fp8 experts with per-row scales were within 3-4 % per block of the exact mirror on GLM; Qwen's softmax
  router may flip near-tied experts between the fp8 device and the bf16 oracle - discrete, as observed for GLM.

## 9. Status log (2026-11-11, branch qwen35-moe)

**Exists**
* `neuron/qwen35_moe_ref.py`: f32 CPU reference (prefill + step decode, state dict per layer); `neuron/tests/test_qwen35_ref.py`
  validates it against transformers 5.14.1 `Qwen3_5MoeForCausalLM` (torch fallbacks forced, fla/causal_conv1d bypassed) on a
  tiny random-init config (4 layers, 8 experts): prefill, 6 decode steps with HF's DynamicCache vs the carried state, a
  3-token continuation chunk, recurrent / conv states: worst relative logit difference **4.3e-7** (f32 noise).
* `neuron/qwen35_graphs.py`: TP graphs (`gdn`, `gqa`, `moe`, fused `layer_gdn_moe` / `layer_gqa_moe`, `head`), same
  trace / meta.json contract as `glm_graphs.py` (inputs by name, aliased states, `enable_aliasing`); `EXACT=1` CPU mode.
* `neuron/qwen35_prepare.py`: HF safetensors (fused 3-D expert tensors) -> per-rank prepared files in the GLM prepared format
  (`prepared.json`, `embed.bin`, `r{rank}/L{layer}.bin`, `r{rank}/head.bin`), fp8 row-scaled E4M3 (down_proj per input
  channel), norms stored as 1 + w; `shard_layer` / `shard_head` are also the sharding the tests use.
* `neuron/tests/test_qwen35_graphs_cpu.py`: the graphs' CPU mirror for 2 ranks vs the oracle (padded prefill chunk, two
  interleaved sequences as decode rows, slot / page state): EXACT **3.6e-7** rel L2 (MOE=all and gather), bf16 8.7e-3,
  fp8 ~0.1 on the tiny random model (top-1 unchanged).
* `neuron/qwen35_build_buckets.sh`, `neuron/tests/test_qwen35_device.py` (random real-shape weights vs mirror through
  `neff_io`), `neuron/tests/test_qwen35_real_layer.py` (one real layer: oracle vs fp8 mirror vs fused NEFF).

**Validated on the trn1.2xlarge (TP=2, E = 128 config, neuronx-cc of `/opt/aws_neuronx_venv_pytorch_2_9`)**
* Every bucket traced and compiled in 11-20 s (NEFFs 130-260 KB): gdn / gqa / moe x R {1, 4} x Q 1, Q = 16 prefill
  chunks (gdn token loop, moe all-experts), gqa P {4, 128}, fused layer_gdn_moe / layer_gqa_moe R = 1, bf16 variants.
* Device vs the exact CPU mirror (random weights of the real shapes, both ranks bit-identical): **bf16 4.1-4.8e-3**
  (rounding noise), **fp8 2.7-5.2e-2** on all 12 fp8 buckets - the same level as GLM's fp8 blocks (E4M3 activation
  rounding boundaries), so the fp8 gap is the format, not a graph bug.
* Real Ornith weights (pulled from S3 to the box's /mnt/scratch; layer 0 gdn, layer 3 gqa, fused NEFFs): device vs mirror
  3.6-3.7e-2, fp8 mirror vs f32 oracle 5.5-7.0e-2, **device vs oracle 5.5-6.8e-2** per layer.
* `layer_bench` (200 steps, 2 ranks, drained; PAGES=65 unless noted), ms per launch:

| NEFF | fp8 | bf16 |
|---|---|---|
| gdn 1x1 | 0.336-0.338 | 0.357 |
| gqa 1x1 P=4 (PAGES 65 / 257) | 0.379 / 0.529 | 0.667 (PAGES 65) |
| gqa 1x1 P=128 (PAGES 257) | 0.527 | |
| gqa 4x1 P=4 / P=128 (PAGES 257) | 0.614 / 0.744 | |
| moe gather 1x1 (8 slots) / 4x1 (32 slots) | 0.247-0.249 / 0.494 | 0.257 |
| moe all, 1x16 prefill chunk (201 MB per rank) | 1.926 (= 104 GB/s) | |
| gdn 4x1 / 1x16 | 0.506 / 0.835 | |
| **layer_gdn_moe 1x1** / **layer_gqa_moe 1x1** (fused) | **0.478** / **0.519** | |

  Head (fp8, 1 row, 254 MB per rank): **2.195 ms** (116 GB/s; the Llama path's transposed `WT=1` head reaches the wall).
  So a batch-1 decode token is 30 x 0.478 + 10 x 0.519 + 2.2 = **21.7 ms of launches -> ~46 tok/s** before any optimisation; 4 rows ~41 ms per step (~100 tok/s aggregate). The DMA floor of section 6 is
  5-6 ms: the launches are fixed-cost bound (gdn 16.8 MB in 0.34 ms = 50 GB/s; moe gather ~15 MB in 0.25 ms = 60 GB/s),
  as predicted. Note the gqa cost grows with PAGES (0.38 -> 0.53 ms from 65 to 257 pages, same P): the in-graph scatter
  touches the whole aliased cache; the Llama path's scatter formulation / NKI decode kernel is the fix.
* The prepare step ran on the box over the real checkpoint (`/mnt/scratch/ornith`, pulled from S3 at ~290 MB/s): TP = 2,
  fp8, ~30 s per layer, 8.5 GiB per rank + 971 MB `embed.bin`, 17.85 GiB in all (89 files, shapes as section 3), archived
  at `s3://912935854432-neuron-build/qwen35/prepared/Ornith-REAP50-tp2-fp8/` (19.19 GB, 89 objects; the instance store
  is wiped on stop). Per core at load: 8.5 GiB weights + head 0.24 GiB + KV + state slots: fits the 16 GiB HBM (section 5).
* Head graph built (`head 1 1`, fp8) and benched; the gqa / gdn / moe buckets of the status table are in
  `/home/ubuntu/work/qw/{g128,g128b,gbf16,ghead}` on the box (persistent root disk), logs in `/home/ubuntu/work/qw/logs`.

## 10. End to end (2026-11-11, second session)

**Runtime** `neuron/cpp/qwen35_model.cpp` (generated from `glm_model.cpp`, same prepared format and step logic): sub-block kinds
`gdn` / `gqa` / `moe` and fused `layer_gdn_moe` / `layer_gqa_moe`, the residual a plain [T, 2048] f32 vector, host-computed
RoPE rows (`cos_sin`, double precision, 64 rotary dims, theta from the config), the KV cache as the gqa layers' aliased
state, GDN recurrent + conv state in LRU slots keyed by the sequence's first page (spill / restore to host RAM as GLM),
eos = config eos + `generation_config.json`'s (`<|im_end|>` 248046 for chat). `NeuronModel` dispatches on
`prepared.json` `format: qwen35-moe-neuron-prepared`; `recurrent()` is true for any `Backend`, so the engine already
disables prompt-lookup speculation (a rejected draft would have advanced the state) and the prefix cache (restored K/V
pages carry no recurrent state) - the choice for this family: **no prefix cache, no speculation**; a preempted request is
re-prefilled from position 0 (the graph zeroes its slot). Scheduler / page pool unchanged (16-token pages for the 10 gqa
layers; the 30 linear layers have no pages). `NEURON_X_DUMP=1|2` reads the residual back after every launch (debug).
Fix found on the way: `layer_set` / `head_set` of the GLM backend inserted into a shared `std::map` from both rank threads
concurrently (UB; it hung the engine on the first runs) - now under a mutex in both backends.

**Tokenizer / template**: the server path (`glm53::Tokenizer`: tokenizer.json ByteLevel BPE + minja) needed one change -
accept Qwen's `NFC` normalizer (not applied; UTF-8 input is NFC in practice). `neuron/cpp/tok_check` +
`tests/test_qwen35_tokenizer.py`: 7 texts (code, CJK, emoji, contractions, control tokens), 3 chat conversations (system,
multi-turn with `<think>`), 1 tool conversation (tools offered, assistant tool_calls, tool result) = **0 mismatches** vs HF
`AutoTokenizer` for both the Qwen and the Ornith tokenizer.json. Qwen's tool-call *output* format (`<tool_call><function=..>`
XML) is not parsed by runtime/src/tools.cpp (GLM / Mistral formats only): tools render correctly into the prompt, but
`tool_calls` in answers are not extracted for this family (gap).

**Generation on the box** (prepared fp8 set from S3, `neuron_engine_cli`, greedy, enable_thinking=false), prompt
"Explain in two paragraphs why the sky is blue.": *"The sky appears blue primarily due to how light interacts with the
molecules of air in the atmosphere and how our eyes perceive those colors. Sunlight, which appears white to us, is actually
composed of all the colors of the visible spectrum, each corresponding to a different wavelength. When this light travels
through the sky, it collides with the gases and particles in the air, causing the light to scatter. Because blue light has a
shorter wavelength than other colors like red or yellow, it gets scattered much more widely in all directions. This
phenomenon, known as scattering, ..."*; the Fibonacci prompt gives a correct iterative function with a docstring.

**Oracle** (`tests/test_qwen35_oracle.py`: the f32 CPU reference over the device's prompt + generated tokens in one pass,
weights streamed from model.safetensors, ~70 s for 57 tokens): teacher-forced top-1 agreement **35/35** (sky, 48-token
run), **34/35** (Fibonacci; the miss at margin 0.19), **107/112** over a 112-token sky answer (misses at margins
0.02-0.55 = near ties, as expected from fp8). The fp8 CPU mirror in prefill form agrees 105/112 with the device.

**Fused-layer fault**: with the fused `layer_gdn_moe` NEFF the residual becomes entirely non-finite after some gdn layer at
a position that differs from run to run (104 / 122 / 144; layers 13 / 17 / 22), from finite inputs and state, and the CPU
fp8 mirror of the same math is finite through all 40 layers; `-O1` and `--auto-cast none` do not change it. The
sub-block graphs (`gdn` + `moe`, `gqa` + `moe`: two launches per layer) run 200 tokens clean. Non-deterministic failure of
one compiled graph = a scheduling / DMA-ordering fault in that NEFF (compare SERVING.md's layer-31 anomaly); not fixable
from the trace. **The served set uses sub-block graphs** (about +4.5 ms per token); fused gqa layers were not seen failing
but are left out as well.

**Not done**
* Qwen tool-call output parsing (above); vision / MTP ignored.
* Wider decode buckets (R 8+), prefill chunks > 16 (the token loop unrolls), NKI kernels for the small ops,
  the 256-expert model (does not fit in fp8, section 5).

## 10b. Buckets to 64k context and measurements (sub-block graph set, trn1.2xlarge, TP=2, fp8)

Bucket set `s3://912935854432-neuron-build/qwen35/graphs/Ornith-REAP50/` (45 NEFFs, 42 MB, built by
`qwen35_build_buckets.sh` with `PAGES=4097 SLOTS=9 MOE_DEC=gather HEAD=1 DECODE_R="1 4 8" PREFILL_Q="64 256"
P="32 128 512 1024 2048 4096"`): `gdn` / `moe` x R {1, 4, 8} and Q {64, 256}; `gqa` x the same x P {32, 128, 512, 1024,
2048, 4096}; heads 1/4/8 rows and 64/256-row prefill. Compile 10-50 s per NEFF, the 256-token gdn prefill (unrolled
recurrence) 14 min. The fused `layer_*` set compiled as well (41 NEFFs) but is not served (section 10).
HBM per core at load: **8.45 GiB weights + 0.90 GiB state** (KV 10 layers x 4097 pages x 16 KiB = 640 MiB bf16 for 64k
tokens, 8 GDN slots 0.26 GiB), ~6.6 GiB free: 2048 more pages or 100+ slots would fit. The KV scatter is now the 4-index
`index_put` of trace_model.py: the gqa layer costs the same at 4097 pages as at 65 (0.47 vs 0.48 ms), so long context
only pays the key gather / softmax (below). KV in fp8 is not needed for memory here.

Measured (engine step log, `tests/qwen35_measure.sh`, greedy, prompt + repeated filler text for the fills):

| | decode ms/step | tok/s | prefill (256-token chunks) |
|---|---|---|---|
| 1 stream, short context (23-token prompt, 200 new) | **47.2** | 21.2 | |
| 4 streams, short | 66.4 (4 rows) | 60.3 aggregate | |
| 1 stream after an 8k fill | 49.0 | 20.4 | 8033 tokens in 13.6 s = **592 tok/s** |
| 1 stream after a 32k fill | 60.4 | 16.6 | 32033 in 65.2 s = 491 tok/s |
| 1 stream after a 64k fill (the 4096-page bucket) | 74.9 | 13.4 | 64033 in 165.7 s = 386 tok/s |
| 4 streams after 8k fills each | 76.2 (4 rows) | 52.5 aggregate | 4 x 8k in 53.9 s = 596 tok/s |
| implicit async launches (`BEHEMOTH_NEURON_QWEN35_ASYNC=1`) | 38.0 / 55.1 (4 rows) | 26.3 / 72.7 | **unsafe**: greedy output diverges from the synchronous run at token 11 (a race between the host-side input writes and in-flight launches); off by default, as for GLM |

Where the time goes: 80 sub-block launches per token; layer_bench puts the device work at 30 x (0.34 + 0.25) + 10 x
(0.38 + 0.25) + 2.2 (head) = 26 ms, the engine measures 47 ms, i.e. ~0.26 ms of host launch overhead per launch. The fused
set did 31.5 ms/token (41 launches) before its fault showed; a race-free async path or fused NEFFs that compute correctly
would give 25-30 ms/token (33-40 tok/s). Long context adds 0.35 ms per gqa layer per 32k keys (dense softmax + 64 MB
gather at 64k). The 64k prefill at 386 tok/s is attention-bound in the Q=256 gqa graphs (dense scores over the bucket).

Validation at long context: the f32 oracle over the 8k-fill run (prompt 8033 + 32 generated tokens, teacher forced, the
512-page gqa buckets): **28/32 top-1 agreement**, the 4 misses at margins 0.045-0.67 (the filler prompt is repetitive text,
so the continuation is a near-tie-rich list). ~45 CPU minutes; 8k is the longest feasible on the box - the 32k / 64k runs
produced coherent continuations through the 2048 / 4096-page buckets but were not oracle-checked.

## 11. Deploying this family (not applied; for the user)

The ECS/Terraform stack (`deploy/ecs-neuron`) serves a model directory + a graph directory pulled from the deployment
bucket (`aws_s3_bucket.models`, not the build bucket). For Ornith-REAP50:

* copy `s3://912935854432-neuron-build/qwen35/prepared/Ornith-REAP50-tp2-fp8/` (89 objects, 19.2 GB: prepared.json,
  embed.bin, r0/, r1/, config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json) to
  `s3://<models bucket>/<model_s3_prefix>/` and `s3://912935854432-neuron-build/qwen35/graphs/Ornith-REAP50/` (45 buckets,
  graph.neff + meta.json each, 42 MB, MANIFEST.json with md5s) to `s3://<models bucket>/<graphs_s3_prefix>/`;
* `terraform.tfvars`: `model_s3_prefix = "models/ornith-reap50-tp2-fp8"`, `graphs_s3_prefix = "graphs/ornith-reap50"`,
  `backend_model_name = "ornith-1.5-35b-a3b"`, `prepared_s3_prefix = ""` (the Llama streaming format does not apply),
  `backend_extra_env = { STREAM_LOAD = "0" }` (the entrypoint's streaming path expects `model-layer-*` files; with
  STREAM_LOAD=0 it pulls the whole prepared directory, then `neuron_server --model <dir>` dispatches on prepared.json),
  `backend_image_tag` = an image built from this branch (the backend binary must contain qwen35_model.cpp and the NFC
  tokenizer change; `deploy/ecs-neuron/backend/build_and_push.sh`), `backend_root_volume_gb` unchanged if the instance
  store holds the data dir (19 GB prepared + graphs), else >= 60;
* trn1.2xlarge per core: 8.45 GiB weights + 0.90 GiB state (640 MiB KV for 4096 pages = 64k tokens of context across all
  requests, 0.26 GiB for 8 GDN slots) - the pool is the 4096 pages; `max_context` 65536 (the 4096-page bucket);
* no Terraform resources change; nothing was applied here.

## 12. Recovery (2026-11-12) and the run-to-run nondeterminism it exposed

The original clone (/tmp, commits 637b362..73e0556) was lost in a reboot. This branch was rebuilt from the S3 code snapshot
(`qwen35/code/src.tgz` + `fix1..fix5.tgz`) and, for what postdated it (sections 10, 10b, 11, the 8k oracle line,
`tests/qwen35_measure.sh`), from the session transcript. The original commit hashes cannot come back; the history here is a
regrouped sequence with the same content. Off-host backup: `s3://912935854432-neuron-build/qwen35/code/qwen35-moe.bundle`
(`git bundle` of 2c4f7af..qwen35-moe).

Acceptance on the box from the recovered tree (engine rebuilt, prepared set + the 45 graph buckets pulled from S3, all
md5s equal to MANIFEST.json): CPU tests identical to before (ref vs HF 4.28e-7; mirror EXACT 3.57e-7, bf16 8.70e-3, fp8
1.00e-1), tokenizer test 0 mismatches, short-context decode **47.3 ms/token** (198 steps; 47.2 before).

**Greedy output is not reproducible run to run** (this was missed on 2026-11-11, where each configuration ran once):
* 200-token sky run today: forks from the 11 Nov `short1` stream at token 11 ("...interacts with matter..." instead of
  "...with the molecules of air..."); both are fluent answers.
* 5 further 40-token runs of the identical request: one equals the 11 Nov `short1` stream token for token, one equals the
  11 Nov `run200` stream, others fork at tokens 11 / 32 / 37, and **one stopped at step 10 with "Nonfinite model logits"**
  - on the sub-block graph set that section 10 called clean.
* 8 runs with `NEURON_X_DUMP=1` (the residual read back after every launch, i.e. a host-device synchronisation between
  launches): 8 / 8 finite, 7 identical streams, 1 fork at token 32.
So nothing was lost in the recovery, but two conclusions of section 10 / 10b need correcting: the non-finite fault is not
specific to the fused `layer_gdn_moe` NEFF (it is rarer with sub-block graphs, not absent), and the async divergence at
token 11 does not by itself show async to be worse than the synchronous path, which forks at the same token. The
dependence on launch timing (forks and faults mostly disappear when every launch is followed by a blocking read) points
at an ordering problem between consecutive launches that share device tensors (the residual ping-pong buffers and the
in-place aliased state: read-modify-write of `kstate` / `cstate` / `kv`), not at the math (the CPU mirror is finite and
deterministic). **Open, and it blocks production use**: next steps are a per-launch synchronisation A/B without the dump
cost, non-aliased state outputs (copy-out) for the gdn graphs, and the same repeat test on the GLM backend, which shares
this code path.

## 13. Prefill of the production model: where the time went, and the faster prefill set (2026-11-20, branch ornith-prefill)

Production (ECS task def `behemoth-backend:24`, image `34c68eb`, branch `ornith-seg-mtp`): Ornith-1.5-35B-A3B-REAP-20pct v3
(205 experts, top-8), TP=2 fp8, int8 KV, graph set `graphs/ornith-reap20-v3-kv8-seg20-mtp`, `--prefill-chunk 256
--max-batch-requests 8 --prefix-cache-gb 4`, prefix cache with state snapshots, MTP. Everything below was measured on the dev
trn1.2xlarge (i-0cd98c451f018c0a3, runtime-lib 2.33.10, neuronx-cc 2.26) with the production artefacts pulled from S3, the
production engine built from `34c68eb` and the production environment. Prompts are deterministic slices of the repo's sources
(`git archive 600b70f neuron runtime`) wrapped in one user message; sizes are the server's `prompt_tokens`.

The graph builder that produced the production NEFFs lived in `behemoth-reap-perf` (branch `reap-mtp2`); commit `600b70f`
brings it (KV8, `GDN_SLOT=bound`, segments, MTP, `GDN_CHUNK` / `GDN_INV`, the reap docs) into this tree. The engine sources
were already identical.

### 13.1 Measured: a 256-token prefill chunk in production

Prefill runs sub-block NEFFs: per chunk 30 `gdn_1_256_32` + 10 `gqa_1_256_P` + 40 `moe_1_256_32` launches (+ the head on the
last chunk), 80-81 synchronous `nrt_execute` calls per rank. Engine step log (rank 0, ms inside `nrt_execute`, median chunk of
single-stream prompts up to 24k tokens):

| | ms per chunk | per launch | layer_bench (same NEFF back to back) |
|---|---|---|---|
| gdn (30 launches) | **246** | 8.2 | 8.01 |
| moe, all experts (40) | **267** | 6.7 | 6.85 |
| gqa (10), context < 2k / 8k / 16k / 32k / 64k (P = 128 / 512 / 1024 / 2048 / 4096) | 12 / 26 / ~53 / ~121 / ~255 (26 measured; the others 10 x bench + launch cost) | 1.1 .. 25 | 1.13 / 2.42 / 5.04 / 11.85 / 25.2 |
| head (last chunk only) | 2 | | 1.95 |
| host: input writes 0.4, set lookup 0.1, plan 0.1; state snapshot 31 ms every 2048 tokens | ~4 | | |
| **chunk** | **541** (short context) .. 770 (64k) | | |

So a chunk is device time inside the launches: 45 % GDN recurrence, 49 % MoE, 5 % attention at short context (33 % at 64k).
The ~0.3 ms per launch that an engine launch costs over a back-to-back one (docs/neuron-reap-launches.md) is 24 ms of the 541;
host code is under 1 %.

### 13.2 Roofline of a 256-token chunk (per rank = per NeuronCore)

trn1 tensor engine: 128 x 128 MACs per cycle at 1.4 GHz = 22.9 TMAC/s with all 128 stationary rows in use; the moving operand
streams at 179 G elements/s (docs/neuron-pe-bound.md), i.e. a weight tensor passes the array once per 128 token rows:
a 256-token chunk streams every weight byte it touches TWICE. HBM DMA ~200-260 GB/s.

| block | work per chunk and rank | roof | production | off by |
|---|---|---|---|---|
| gdn projections (qkv, z, a, b, o: 16.8 MB fp8) | 2 x 16.8 MB = 4.3 GMAC | 0.19 ms | | |
| gdn recurrence, 16 value heads x 128 x 128 f32 state | chunked form ~0.7 GMAC; token loop: 256 x 4 passes over 1 MB of state | 0.03 ms | | |
| **gdn layer** | | **~0.25 ms** | 8.0 ms | **32x** |
| gqa projections (13.6 MB) | 2 x 13.6 MB | 0.15 ms | | |
| gqa scores + values, N keys: 256 x 8 heads x N x 256 x 2 MAC | N = 8k / 32k / 64k: 8.6 / 34 / 69 GMAC | 0.4 / 1.5 / 3.0 ms | 2.4 / 11.9 / 25.2 ms | 5-8x |
| **moe, all experts**: 205 x (512 x 2048 gate\|up + 256 x 2048 down) = 322 MB fp8 | 2 x 322 MB = 82.5 GMAC | **3.6 ms** | 6.9 ms | **1.9x** |
| moe, only the routed experts (8 of 205 per token) | 3.2 GMAC useful, but every expert is hit by ~10 of the 256 tokens: all 322 MB must pass the array once | 1.8 ms | - | |
| head (one row, 254 MB) | | 1.4 ms | 1.95 ms | 1.4x |
| **chunk, short context** | 30 gdn + 10 gqa + 40 moe | **~155 ms (1650 tok/s)**; ~85 ms with a routed MoE | **541 ms (470 tok/s)** | **3.5x** |

Why the dense 24B Devstral prefills 3-4x faster on the same box: its chunk streams 40 x 278 MB of weights with nothing else in
the way (0.21 ms per layer and token row block, at the DMA / PE wall). This model's prefill streams MORE weight bytes than
Devstral's (`MOE=all` touches all 12.9 GB of experts per chunk, twice; the "3B active" only holds for decode), runs the GDN
recurrence as 256 unrolled token steps on the vector engine, and computes dense softmax passes over the whole key bucket.

### 13.3 What was changed (in order of payoff)

**1. GDN prefill: chunked gated delta rule + bound state slots (graphs only; no engine change).**
`GDN_CHUNK=128 GDN_INV=diag GDN_SLOT=bound` for `gdn_1_256_32` (`GDN_CHUNK=64` for `gdn_1_64_32`): the recurrence in matmul
form (the algebra of `gdn_chunks2`, validated for the 397B) with the state as the request's own slot tensor.

| NEFF (layer_bench, ms) | token loop, dynamic slot (prod) | chunked, dynamic slot | token loop, bound | **chunked, bound** |
|---|---|---|---|---|
| gdn_1_256_32 | 8.01 | 7.64 | (14 min compile, not built) | **1.11** (chunk 64: 1.13) |
| gdn_1_64_32 | 2.22 | 7.14 | 2.14 | **0.54** |

The chunked form alone gains nothing here (unlike the token loop it leaves a ~6.5 ms fixed cost around the runtime-indexed
state slot); bound alone gains nothing either. Together: 7.2x, 246 -> 39 ms per chunk. Compile 40-70 s instead of 14 min.
Device vs CPU mirror (random real-shape weights): 4.13e-2 / 4.22e-2, the production loop NEFFs 4.14e-2 / 4.23e-2.

**2. Attention prefill in key blocks (graphs only).** `ATT_BLOCK` (the online-softmax loop of `gqa_fp8` that existed for the
256k-context set) for the Q=256 buckets: the scores / exp / weighted-value passes of one block of 128-256 pages stay small
instead of making ~8 elementwise passes over a [256, 8, keys] f32 tensor in HBM (537 MB at P=4096).

| gqa_1_256_P, layer_bench ms | P=512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| production (one block) | 2.42 | 5.04 | 11.85 | 25.2 |
| `ATT_BLOCK` = 64 / 128 / 256 / 512 / 1024 at P=4096 | | | | 16.6 / 10.8 / **10.5** / 15.1 / 16.3 |
| chosen: `ATT_BLOCK=128` (P=512), `256` (P >= 1024) | **1.71** | **2.91** | **5.42** | **10.5** |

Device vs mirror 4.19e-2 (production 4.20e-2). P=32 / 128 are unchanged (1.04 / 1.13 ms).

**3. MoE prefill by expert groups, `MOE=group` (graphs + engine).** Real routing of this model (MOE_GSTAT build, 6480 + 2880
MoE launches over code and prose prompts): of a chunk's 2048 (token, expert) pairs the three most loaded experts take up to
255 / 224 / 198 of the 256 tokens (generalists), the 9th up to 128, the 17th up to 48, the 25th up to 34, the 33rd up to 27;
the mean is 10. So
`moe_grouped()` runs two tiers with static shapes: the `MOE_HOT` most loaded experts of the chunk are gathered and run on
every token (the decode graphs' `MOE=gather` form), every other expert gets `MOE_CAP` token slots (rows gathered by index,
one 2-D fp8 matmul per expert for gate|up and for down, the slot outputs added back to their tokens by a one-hot bf16 matmul
with the slot's fp8 scale folded in). The graph also returns the largest cold-expert load; the engine (kind `moeg`) reads it
after the launch (4 bytes) and, when it exceeds `MOE_CAP`, reruns the layer's MoE with the bucket's `MOE=all` NEFF from the
untouched input residual - the result is always the exact routed sum. All ranks exchange their verdict first, so they cannot
disagree on a launch whose collectives need them all. Padding tokens of a short chunk (position -1; the graph takes `pos`)
are routed to no expert: see 13.5 for why that matters.

Fallback rate P(load of the (S+1)-th expert > C), code / prose prompts, %: (S=16, C=32) 24 / 31; (16, 40) 0.9 / 1.7;
(16, 48) 0 / 0; (24, 28) 1.1 / 3.4; **(24, 32) 0.00 / 0.14**; (24, 36) 0 / 0; (32, 24) 0.03 / 0.31.

| moe NEFF, Q=256, layer_bench ms | |
|---|---|
| `MOE=all` (production; with `ROUTE=thr` 6.42) | 6.85 |
| group, one batched matmul over the experts (first attempt) | 14-61, and non-finite on the device |
| group, all 205 experts in one pass: C=16 / 24 / 32 (S=16..32) | 3.4 / 11.9 / 22.6 (a cliff once the slot tensors of a pass exceed ~27 MB) |
| group, `MOE_GE=69` experts per pass: (C, S) = (32, 24) / (40, 24) / (48, 16) | **4.41** / 4.74 / 5.26 |
| group, `MOE_GE` = 41 / 16 / 8 at (32, 24) | 5.00 / 6.89 / 9.65 (0.25 ms per pass) |

Chosen `MOE_CAP=32 MOE_HOT=24 MOE_GE=69`: 4.4 ms, 4.7 ms per launch in the engine (incl. the load read), 267 -> 188 ms per
chunk; fallbacks in the full prompt matrix: 5 of 22 560 launches (0.02 %), none in the parity and cache runs (8 720 launches). Device vs mirror 3.48e-2 (MOE=all 3.50e-2). It is still
2.4x above the routed roof (1.8 ms): the router, the shared expert and the hot tier, the gathers, the one-hot scatter and
the per-pass overhead cost as much as the expert matmuls themselves.
Two neuronx-cc traps found on the way: a batched matmul over the 205 experts is slow AND non-finite; an fp8 x fp8 matmul
only accumulates in f32 when its `.float()` follows the dot directly (stacking the fp8-typed results first rounds them to
E4M3 and returns NaN) - `bmm8g()` documents both.

**4. Launch / host overhead: measured, not changed.** Host code is < 1 % of a chunk. The 80 launches cost ~24 ms (now 10 % of
a chunk); prefill segments (`seg_gdn_gdn_gdn_gqa` at Q=256, 10 launches) would recover most of it but cannot hold the grouped
MoE (its rerun must restart from the MoE's own input, and a segment has already advanced the gdn / KV state).

### 13.4 Before / after

Same box, same session, same prompts, production env; engine `34c68eb` + production graphs vs the engine of this branch +
the new prefill set (final build, 13.7).

Per 256-token chunk (rank 0, ms in `nrt_execute`): gdn 246 -> **39**, moe 267 -> **188**, gqa (P=128 .. 4096) 11 .. 255 ->
11 .. 106, total at short context 541 -> **248** (1030 tok/s).

| prompt (tokens) | TTFT before | TTFT after | prefill tok/s before -> after |
|---|---|---|---|
| 82 | 0.67 s | 0.38 s | (one padded chunk) |
| 501 | 1.16 s | 0.59 s | 461 -> 971 |
| 3029 | 6.54 s | 3.08 s | 469 -> 1009 |
| 6715 | 14.42 s | 6.80 s | 469 -> 1001 |
| 23763 | 54.30 s | 25.01 s | 439 -> 956 |
| 50111 | 129.26 s | 58.26 s | 388 -> 863 |
| 4 concurrent x ~7k (28392 in all): first token of the first / last stream | 52.5 / 61.5 s | 24.3 / 28.8 s | 462 -> 986 aggregate |
| 8 concurrent x ~3.8k (30293 in all) | 28.7 / 68.6 s | 13.4 / 33.6 s | 442 -> 903 aggregate |
| 4 concurrent x ~60 | 1.58 s | 0.90 s | |

(tok/s = the server's `request_metrics.prefill_tokens_per_s`; TTFT = first streamed token at the client.)

Decode is untouched (same decode NEFFs; engine step log, ms per step, before -> after): 1 stream with one MTP draft 25.4 ->
25.3 (15.5-17.5 ms per token at the client in both), 1 stream no draft 18.3 -> 18.0, 4 rows 34.4 -> 34.3, 8 rows 106.4 ->
106.1. HBM: the set loads next to the weights as before (81 NEFFs, 13.32 GiB weights + 0.66 GiB state + 0.38 GiB MTP per
core) and ran the 50k-token prompt through the P=4096 buckets.

### 13.5 Parity

* CPU: `EXACT=1 PREFILL_MOE=group` mirror vs the f32 oracle 2.9e-7 (MOE=all 2.7e-7), fp8 6.8e-2 (6.7e-2), bf16 4.9e-3;
  the overflow rerun path (`MOE_CAP=1`) reproduces MOE=all exactly; `test_qwen35_gdn_chunk2.py` <= 2.5e-6.
* Device vs CPU mirror per NEFF (random real-shape weights, `test_qwen35_device.py`): all new NEFFs 3.5-4.2e-2, equal to the
  production NEFF of the same block; both ranks bit-identical.
* End to end, `NEURON_LOGITS_DUMP`, MTP off, 7 prompts (82 .. 23763 tokens) x 8 greedy steps, final set vs production:
  **prefill-end top-1 7 / 7**, logits rel L2 0.092-0.168, KL 0.0005-0.0058; over the following greedy steps 32 / 36 compared
  steps agree, the 4 flips at production margins 0.00 / 0.01 / 0.05 / 0.34. 0 non-finite logits in every run of the session
  (parity, matrix, cache tests: ~2500 prefill chunks on the new graphs).
* Is 0.1 rel L2 "the same"? `tests/qwen35_prepared_ref.py` (new): the graph functions in EXACT mode over the DEQUANTISED
  prepared weights, the whole prompt as one chunk = an f32 reference that isolates what the device adds (fp8 activations, bf16,
  int8 KV, block form). Prefill-end logits vs that reference, rel L2 (KL) for the 82-token / 501-token prompt:

  | build | 82 tokens | 501 tokens |
  |---|---|---|
  | production | 0.083 (0.0013) | 0.092 (0.0011) |
  | GDN change only | 0.075 (0.0059) | 0.095 (0.0032) |
  | GDN + grouped MoE (C=40, S=16) | 0.083 (0.0011) | 0.081 (0.0030) |
  | **final set** (GDN + blocked attention + grouped MoE C=32, S=24) | 0.077 (0.0013) | 0.117 (0.0017) |

  top-1 equal to the reference in all. Production itself sits 8-9 % from the f32 result; the new forms are at the same level
  (0.075-0.117 over the variants), and two builds differ from each other like two draws of that fp8 noise. Only two reference
  prompts (the reference costs 17 CPU minutes for 501 tokens on the box): this bounds the error class, it is not a statistic.
* **Run-to-run reproducibility.** Production prefill is bit-reproducible (two server starts: 56 / 56 logits records
  identical), and the prefix cache relies on it. The GDN / attention changes keep that (two starts bit-identical). The first
  grouped-MoE build did NOT: prompts whose last chunk was a padded 256-token chunk differed by 0.04-0.09 rel L2 between two
  starts. Cause: a padding row's residual is not reproducible (its all-masked attention row reads the scratch page, which
  the padding tokens of a launch overwrite in arbitrary order); MOE=all keeps rows independent, but in the grouped form a
  routed padding row occupied slots and moved the hot / cold split - i.e. the fp8 rounding path - of the real tokens. Fixed
  in the graph (padding tokens take no expert, commit `76b905a`): two server starts of the final set are bit-identical on
  all 56 records, also across a cold second server in the cache test.
* New engine + production graphs: all 56 logits records bit-identical to the production engine.
* Device vs CPU mirror of the final `moeg_1_256_32`: 3.48e-2; layer_bench of the final set 1.13 / 0.54 (gdn 256 / 64),
  1.68 / 2.91 / 5.42 / 10.46 (gqa P=512 .. 4096), 4.38 ms (moeg).

### 13.6 Prefix cache

The state snapshots are the same tensors (bound NEFFs read / write the request's slice of the layer state), so the cache
needs nothing new. `PREFIX_ALIGN=1` exactness holds for the new set as for production (logits dump, MTP off, 8 steps each):
a repeated 6715-token prompt (6656 cached tokens), the same prompt with 487 more tokens (6656 cached, 546 prefilled after
the restore), and the same pair for a 5251-token prompt whose tail is a padded 256-token chunk (5120 cached) all return
logits **bit-identical** to a cold server. TTFT of the extended 7202-token turn: 1.43 -> 0.79 s; of the full hit 0.26 -> 0.21 s.

### 13.7 Building the set, and what a deployment needs (nothing was pushed or deployed)

Graphs (7 NEFFs, ~4 min with 7 parallel compiles on the box; needs the prepared set's `config.json`):

    SRC=<tree of this branch> CONFIG=<prepared dir>/config.json bash deploy/neuron-reap/pfbuild.sh /mnt/scratch/gpf
    #   gdn_1_256_32   GDN_CHUNK=128 GDN_INV=diag GDN_SLOT=bound      gdn 1 256 32
    #   gdn_1_64_32    GDN_CHUNK=64  GDN_INV=diag GDN_SLOT=bound      gdn 1 64 32
    #   gqa_1_256_512  KV8=1 ATT_BLOCK=128                            gqa 1 256 512
    #   gqa_1_256_{1024,2048,4096}  KV8=1 ATT_BLOCK=256               gqa 1 256 P
    #   moeg_1_256_32  MOE=group MOE_CAP=32 MOE_HOT=24 MOE_GE=69      moe 1 256 32
    # common: TP=2 PAGES=4097 SLOTS=9 WEIGHTS=fp8 (KV_CODE int8 is the default)

The NEFFs built by that script in this session - the exact files of the final measurements - are in
`s3://behemoth-neuron-models-912935854432/tmp/ornith-prefill/graphs-pf/` (scratch prefix; 7 directories with `graph.neff` +
`meta.json`, `MANIFEST.md5`); logs and per-request JSON of every run under
`.../tmp/ornith-prefill/{run,log}/`. Engine: `neuron/cpp/build.sh OUT srv` from this branch. The measurement drivers are
in `deploy/neuron-reap/prefill/` (`matrix.sh` prompt matrix, `par.sh` + `cmp.py` logits parity, `cache.sh` prefix-cache
exactness, `ref.sh` + `cmpref.py` prepared-weights reference, `loads.py` expert-load table, `lb.sh` / `devtest.sh` per-NEFF
bench and device-vs-mirror test, `gb.sh` ad-hoc NEFF builds).

To deploy (for the user):

1. Graph prefix: copy `graphs/ornith-reap20-v3-kv8-seg20-mtp/` to a NEW prefix (e.g. `graphs/ornith-reap20-v3-kv8-seg20-mtp-pf1/`),
   replace `gdn_1_256_32`, `gdn_1_64_32`, `gqa_1_256_{512,1024,2048,4096}` by the new directories and ADD `moeg_1_256_32`
   (the plain `moe_1_256_32` must stay: it is the overflow rerun and the bucket's consistency check). 81 NEFFs.
2. Backend image from this branch (the engine with the `moeg` kind). Order matters only for the grouped MoE: the production
   engine does not know `moe_cap` and would treat `moeg_1_256_32` as a second `moe` NEFF of the bucket, so the image must be
   live before (or together with) the new prefix. The gdn / gqa NEFFs alone run on the production engine unchanged
   (measured: 746 tok/s at 8k from the two gdn NEFFs alone) - a graphs-only first step is possible by leaving `moeg_1_256_32` out.
3. `deploy/ecs-neuron/models.auto.tfvars`: `graphs_s3_prefix` of the `ornith` entry -> the new prefix; nothing else (same
   server args, same env; `--prefill-chunk 256` stays, the snapshot grid is unchanged).
4. Rollback: the old prefix (and, if wanted, the old image; the new engine runs the old prefix bit-identically to the
   production engine, 56 / 56 logits records). Kill switch without a redeploy of graphs: `BEHEMOTH_NEURON_QWEN35_MOE_GROUP=0`
   (the engine ignores the grouped NEFF; prefill MoE back to MOE=all).
5. Watch after a deploy: `request_metrics.prefill_tokens_per_s` (expect ~2x), and with `BEHEMOTH_NEURON_STEP_LOG=1` the
   `moe_fallbacks` / `moe_grouped` counters of the step profile. The fallback rate was measured on code and technical prose
   only; role-play prose may route differently. A fallback costs one extra MOE=all launch (6.9 ms) - with every launch
   falling back a chunk would take ~520 ms, still not slower than production - and `MOE_CAP=40 MOE_HOT=24 MOE_GE=69`
   (4.74 ms, no fallback seen on either corpus) is the conservative build.

### 13.8 Not done / left

* MoE: 4.4 ms per layer vs 1.8 ms routed roof and 6.9 ms before. The next step is the expert part as an NKI kernel
  (token-sorted, the gather / contraction / scatter of one expert without leaving SBUF; `nki_moe_dyn.py` has the decode-side
  kernels) or larger chunks, where the grouped form amortises (its cost is per chunk, MOE=all's per token) - but a Q=512
  bucket ladder does not fit next to this model's 13.3 GiB of weights per core without dropping other buckets, and the cold
  tier hits a cliff when one pass's slot tensors exceed ~27 MB (kept below it by `MOE_GE`).
* Launch overhead (~24 ms of the 248 ms chunk): prefill segments are incompatible with the per-MoE rerun (13.3.4).
* Short prompts: a 65..255-token prompt still pays one padded 256-token chunk (0.39 s for 82 tokens; 0.67 s before).
* The Q=64 buckets keep MOE=all and the one-block attention (used for <= 64-token prompts and for nothing else on the grid).
* GDN prefill is at 1.1 ms vs ~0.25 ms roof: the projections + conv over 256 tokens and the bf16 all-reduce are what is left.
* Parity against the HF-checkpoint oracle (`test_qwen35_oracle.py`) was not rerun (no raw checkpoint in the two buckets used
  here, and nothing was fetched from outside the account); the prepared-weights reference of 13.5 replaces it, on two prompts.
* Decode at long context (29-31 ms per token at 50k) still uses the one-block attention; `ATT_BLOCK` for the decode / segment
  buckets is the same switch but was out of scope (decode numbers had to stay unchanged).
