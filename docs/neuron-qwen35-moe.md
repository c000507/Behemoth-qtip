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
