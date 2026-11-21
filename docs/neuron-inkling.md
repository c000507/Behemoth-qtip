# Inkling 2 Mini (`inkling2_loop`, text) on trn1.2xlarge, TP=2

Target: `OpenUncensored/Inkling-2-Mini-Uncensored` (rev `748f78e`, Apache-2.0; base `thinkingmachines/Inkling-2-Mini`;
only the 20 sliding-layer attention output projections differ from the base). Reference semantics: transformers PR 51050
(`models/inkling2_loop`, commit `6173a2f`), whose `modeling_inkling2_loop.py` is the reviewed code the oracle follows.
Vision tower and the MTP / DSpark head are ignored. Branch `inkling` (from `ornith-wt` 1eeda00).

Status log and measured numbers: section 9 onwards.

## 1. Architecture (config + HF code)

| | value |
|---|---|
| parameters | 26 B total, 3.4 B active per token |
| layers | 24 decoder layers: 20 sliding-window (window 1024) + 4 global (5, 11, 17, 23); layers 6..17 are LOOPED 3 times with shared weights -> **48 layer visits per token**, each with its own K/V and conv state |
| hidden | 2304, RMSNorm eps 1e-6 with plain `w * norm(x)` weights; vocab 201 024 embedding rows, 200 058 lm_head rows (`unembed` padded), untied |
| attention | 24 q heads, 8 KV heads, head_dim 128, **no RoPE**. q = l2clip(scale * Wq x) (threshold sqrt(512)), global layers also x tau(pos) = 1 + 0.1 log(max((pos+1)/8192, 1)); k = l2clip(conv(Wk x)); v = unweighted-rmsnorm(conv(Wv x)) per head. Scores q.k/sqrt(128) + a **hidden-state-conditioned relative bias** (per head a 16-dim state r = l2clip(scale * Wr x) [* tau] times a [16, 1024] bank of profiles -> one bias per backward distance < 1024, 0 beyond) + causal / window mask, one **sink logit** per head and query (Ws x); output x sigmoid(Wgate x) -> Wo |
| short convs (kernel 4, f32) | sliding layers: static depthwise conv on the k / v channels, out = (conv + x) / sqrt(5); global layers: DYNAMIC conv (static taps + per-token per-head taps from weight_proj x), out = (conv + x) / 3. Every layer: post-attention and post-MoE convs over the 2304 residual channels |
| depth attention ("block attention residuals") | the residual stream is a list of completed blocks (embedding, then the partial sum of every 6 layers) + the running partial. Layer inputs are softmax mixtures of the unweighted-RMS-normed [blocks..., partial] with learned queries (1 for sliding inputs / MLP inputs, 3 for the q / k / v inputs of global layers); a block-boundary layer (l % 6 == 0) closes its input partial into a block and restarts the partial from its attention output; the final norm reads a depth-attention mixture of everything |
| loop | passes 2 and 3 restart from the pre-loop state (partial = B1, blocks = [embedding]) and get 4 messages per loop layer (q, k, v, MLP inputs): depth attention over the PREVIOUS pass's [B2, final partial], mixed in by rank-64 message gates `(1 - tanh(Wf z)) h + tanh(Wg z) m`, z = Wd [urms h, urms m] |
| MoE (every layer) | 128 experts, top-8, no shared expert, intermediate 1152; router sigmoid(x Wr) (f32), selection by sigmoid + bias, weights = sigmoid scores renormalised x 8; act = 4 tanh(g/4) sigmoid(g) 25 tanh(u/25) |
| head | final depth attention -> RMSNorm -> lm_head -> 30 tanh(logits / 30) |
| checkpoint | one 19.5 GB shard: experts **NVFP4** in layers 0..19 (E2M1 codes, fp8 scale per 16, per-expert amax / denominator), **MXFP8** in 20..23 (E4M3, 2^(s-127) per 32); gate|up rows interleaved (g0, u0, g1, ...); everything else BF16 / F32 |

Tensor names (original checkpoint, `model.llm.`): `embed`, `embed_norm`, `unembed`, `norm`, `attn_res.{attn_queries.<l-1>, mlp_queries.<l>, output_query}`,
`middle_cycle_cross_pass_{queries, kv_queries}` ([24, 2304] each: (q, mlp) and (k, v) per loop layer), per layer
`attn.{wq_du, wk_dv, wv_dv, wo_ud, gate_attn, wr_du, attn_sink_proj, q_l2_clip.scale, rel_l2_clip.scale, rel_logits_proj.proj}`,
sliding `attn.{k,v}_sconv.weight` + `attn_norm`, global `attn.{k,v}_sconv.{static_w, weight_proj}` + `{q,k,v}_input_norm`,
`attn_sconv`, `mlp_sconv`, `mlp_norm`, `mlp.gate.{weight, bias}`, `mlp.experts.w13_*` / `w2_*`, loop layers `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight`.

## 2. Decomposition for the NeuronCores

**Visits.** A token runs the 48 visits in order (0..5: layers 0..5; 6..41: layers 6..17 x 3 passes; 42..47: layers 18..23).
One graph = one visit (attention + MoE fused) or a **segment** of consecutive visits. Visit kinds: `s` sliding, `g` global,
`S` / `G` the same in loop passes 2-3 (with message gates). The visit sequence is `sssssg` x 4 (visits 0-5, 6-11, 12-17,
42-47) and `SSSSSG` x 4 (18-41): **a decode token is 8 segment launches of two NEFF kinds + the head**.

**Residual x [T, 7, 2304] f32** (replicated on both ranks): slot 0 embedding, 1-3 the completed blocks B1-B3, 4 the partial,
5 / 6 the previous pass's B2 / final partial (message sources). The host writes the (pre-normed) embedding into slots 0 and 4.

**ctl [17] f32 per visit** (a device constant per visit, built by the prepare step / engine from the visit table): the
attention-input and MLP-input block masks (0 visible, -3e38 hidden), the boundary flag and the block slot it writes, the
first-layer flag (layer 0 has no attention input), the loop-restart flag (first loop layer of passes 2 and 3: message
sources <- (B2, partial), partial <- B1). One NEFF per kind therefore serves every visit of that kind; the difference
between visits is data, applied with exact 0/1 arithmetic.

**Near / far attention.** Every visit keeps a K|V buffer of the request's last NB tokens (NB = 1024 = window = rel_extent)
in LOGICAL order (entry i of a row whose first new token is at p0 holds position p0 - NB + i), shifted by the graph after
each launch. With that order the relative bias of every (query, key) pair is a static skew of the query's profile
(pad / flatten / reshape, no gather, no runtime offset); positions < 0 are masked. Sliding visits attend over [buffer;
chunk] (window mask); global visits add the FAR keys from the server's paged K/V (only keys at distance >= 1024, which
carry no bias, and which are never the chunk's own keys, so there is no read-after-write inside a launch). The buffer and
the conv states are bound per row (the engine binds the request slot's one-slot slice: no runtime-indexed state DMA, the
lesson of docs/neuron-reap-nonfinite.md); a prefill chunk with padding updates them by the count of valid tokens (one-hot
row selection, exact). A row starting at position 0 starts from zero state (the graph multiplies the bound state by
[p0 > 0]).

## 3. Sharding (TP = 2, rank r)

| tensor | rank slice | per rank |
|---|---|---|
| wq / gate_attn | heads [12 r, 12 r + 12) | [1536, 2304] |
| wk / wv, k / v convs (+ dynamic taps) | KV heads [4 r, 4 r + 4) | [512, 2304] |
| wo | input columns of those heads (row-parallel, all-reduce) | [2304, 1536] |
| wr (rel states), q / rel scales, sink | the rank's heads | [192, 2304], [12, 2304] |
| rel profile bank, norms, depth queries, convs over the residual, message gates, router (f32) + bias | replicated | |
| experts | ETP: gate / up rows [576 r, 576 r + 576) and down input channels of EVERY expert (one all-reduce per MoE) | gate\|up [128, 1152, 2304], down [128, 576, 2304] (transposed, per-input-channel scale) |
| lm_head | rows [100 029 r, ...) of the 200 058 | [100029, 2304] |
| embedding | host (pre-normed bf16) | |

Formats (`inkling_prepare.py`): dense projections and experts E4M3 with a per-output-row f32 scale (max |code| 224; experts'
down per input channel), sink / dynamic-tap / message-gate weights bf16, everything small f32. WT=1 graphs take the
projections / router / gates / head transposed [in, out] (the engine relays at load).

## 4. Memory per NeuronCore

| | GiB |
|---|---|
| experts (24 layers x 128 x 1728 x 2304 fp8) | 11.39 |
| attention projections (24 x 13.5 MB), router f32, message gates bf16 | 0.40 |
| lm_head shard fp8 | 0.21 |
| **weights** | **12.0** |
| per sequence slot: 48 K\|V buffers (1024 x 2 x 4 x 128 bf16 = 2 MiB) + 48 conv states (67.6 KB) | 0.097 |
| paged far K/V: 8 global visits x 32 KiB per 16-token page (2049 pages = 32k tokens) | 0.50 |

## 5. Decode cost model (batch 1, per rank)

Per visit: attention 13.5 MB + 8 gathered experts 31.9 MB + router 1.2 MB (+ 2.4 MB message gates in passes 2-3) +
the K|V buffer 2 MB = ~49 MB; x 48 visits = 2.35 GB, + head 0.23 GB: **~2.6 GB per token per rank -> DMA floor ~12 ms at
214 GB/s** (~80 tok/s). The loop triples the per-token bytes of layers 6..17: Inkling's active bytes are those of a
48-layer model.

## 6. Gates (trn1.2xlarge i-0cd98c451f018c0a3, neuronx-cc 2.26, 2026-11-21)

**GATE 1 - oracle == HF.** `neuron/inkling_ref.py` (f32) against transformers PR 51050 (f32):
* tiny random checkpoint written in the ORIGINAL format (NVFP4 + MXFP8 experts, loaded through HF's own conversion;
  `tests/test_inkling_ref.py`, local): prefill (20 tokens > window 8 and rel extent 12, log-scaling active), 6 decode steps
  with HF's DynamicCache, a 5-token continuation chunk: worst rel **6.4e-6**;
* the real checkpoint on the box CPU (`tests/test_inkling_oracle_hf.py`: HF's dequant op cached per layer as f32 memmaps,
  ~100 GB on the instance store; the oracle reads the same files): the oracle's own NVFP4 / MXFP8 dequant is bit-identical
  to HF's (layers 0 and 20); chat prompt (27 tokens) and raw text (5 tokens) + 24 oracle-greedy tokens each, HF teacher-forced:
  prefill logits rel **5.2e-7 / 3.3e-7**, decode steps max **9.9e-7**, max |d| 3e-5, **top-1 48 / 48**. Text:
  `<|message_model|><|content_thinking|>The user asks for explanation in two short paragraphs why sky is blue. ...` and
  ` Paris.\n\nA. True\nB. False ...`. PASS.

**Graph mirror.** `tests/test_inkling_graphs_cpu.py` (TP = 2 ranks as threads, prepare sharding of a tiny original-format
checkpoint, prefill chunks with padding, 2-row decode with bound states, slot reuse at position 0): EXACT **6.7e-6** vs the
oracle (also WT, gathered MoE, 6-visit segments). bf16 / fp8 numerics on the TINY random model diverge over 48 visits (routing
flips amplify; local per-visit bf16 error 2-7e-3), so those are judged on the device and end to end instead.

**Device vs CPU mirror** (`tests/test_inkling_device.py`, random real-shape weights, the real visits' ctl, one launch per NEFF,
both ranks bit-identical): decode visits s / g / S / G **3.5e-2 / 3.5e-2 / 5.9e-2 / 5.1e-2** (the stack's fp8 level), heads
1.3-1.6e-2; prefill Q = 256 visits: per-token median 5.6e-2, a few tokens up to 0.49 (top-8 routing flips between the two fp8
computations; the K|V buffer and k / v conv states agree to 1e-3), block rel 0.13-0.16. The random router had to be scaled to a
realistic logit spread: with saturated sigmoids many scores tie at 1.0 and the device and CPU pick different experts.

**GATE 2 - fp8 re-encode cost** (`tests/inkling_kl.py`, f32 oracle, reference = exact dequantised experts + bf16 dense;
held-out text: 2304 tokens of the repo's prose in one pass; 5 chat-template prompts, 134 positions):

| variant | text KL mean / p99 / max (nats) | text top-1 | chat KL mean | chat top-1 |
|---|---|---|---|---|
| x8: experts E4M3 per output row (gate\|up) / input channel (down) - the prepared format | 0.0075 / 0.055 / 0.62 | 95.7 % | 0.030 | 99.3 % |
| x8d8: x8 + dense projections and head E4M3 per row (what the graphs use) | 0.0109 / 0.084 / 0.34 | 94.9 % | 0.048 | 92.5 % |
| xb128: experts with one scale per 128 input columns | 0.0069 / 0.056 / 0.23 | 96.1 % | (not run) | |

Small (mean KL ~0.01 nats per token): per-row fp8 is kept; 128-column blocks would buy little (0.0069 vs 0.0075). PASS.

**GATE 3 - end to end.** `neuron_engine_cli` on the prepared set, greedy, 3 chat prompts x 32 tokens, the oracle teacher-forced
over prompt + device tokens (`tests/inkling_gate3.py`): **91 / 96 top-1** against the fp8-weight oracle (x8d8), 90 / 96 against
the exact-weight oracle; every miss is a near tie (oracle margin 0.03-0.33, the device's token being the oracle's runner-up);
0 non-finite. Device text: *"<|message_model|><|content_thinking|>The user asks for an explanation in two short paragraphs why
the sky is blue. Need answer concise, two short paragraphs. Need ensure accuracy"*; *"... capital of Australia and why it was
chosen. Need answer factual. Need ensure accuracy. Australia capital is Canberra. Why chosen"*.

## 7. Performance (trn1.2xlarge, TP = 2, the WT set, engine of this branch)

Graph set `s3://behemoth-neuron-models-912935854432/tmp/inkling/graphs/inkling-wt-f512/` (26 NEFFs, `MANIFEST.md5`;
built by `deploy/neuron-inkling/build.sh` with `WT=1 PAGES=2049`, specs as below), prepared set
`.../tmp/inkling/prepared/tp2-fp8/` (59 objects, 26.7 GB incl. the 0.93 GB normed embedding).

* decode: `seg_sssssg_R_1_P` / `seg_SSSSSG_R_1_P` for R in {1, 2, 4} x P in {64, 512} (`MOE=gather`, `ROUTE=pad64` for R > 1),
  heads 1 / 2 / 4 rows; 8 segment launches + head per token;
* prefill (256-token chunks): every visit SPLIT into an attention NEFF and a MoE NEFF (`PART=attn` / `PART=moe`;
  `MOE=all`, MoE parts `WT_MOE=grp WT_GRP=8`), `ga` / `Ga` (the global visits' attention with far keys) at P in {64, 512},
  head 256 -> 1 row.

| | ms / step | tok/s |
|---|---|---|
| decode, 1 stream (engine step, median of 64) | **26.7** (head 1.25) | **37.4** (client 37.2 through neuron_server) |
| decode, 2 streams | 44.0 | 45.4 aggregate |
| decode, 4 streams | 77.8 | 51.4 aggregate |
| decode, 8 streams (two 4-row launches; an 8-row bucket was slower, 200.6 ms) | 155.6 | 51.4 aggregate |
| decode after a 1k / 4k / 7.8k-token prompt, 1 stream | 28.9 / 28.2 / 28.1 | ~35 |

| prompt | TTFT | prefill tok/s |
|---|---|---|
| 27 tokens (one padded chunk) | 1.0 s | |
| 1026 | 5.62 s | 183 |
| 4026 | 23.8 s | 169 |
| 7826 | 48.9 s | 160 |

0 non-finite logits in every run. Before the WT relayout and the segments, decode was 45.8 ms (49 single-visit launches),
then 34.0 ms (non-WT segments); the WT layout took the segments 3.20 -> 2.89 / 3.63 -> 3.19 ms and the head 2.04 -> 1.11 ms.

**Against the DMA floor** (section 5: ~2.6 GB per token per rank at batch 1 -> ~12 ms at 214 GB/s): decode is at 2.2x the floor
(26.7 ms); a segment of 6 visits (~0.28 GB) takes 2.9-3.2 ms = ~90 GB/s. At 4 rows the routed experts of the union (~29 of 128
per visit) dominate: ~7.5 GB per step -> ~35 ms floor vs 77.8 ms.

**Prefill** is far from its roof (the MoE streams all 510 MB of experts per visit and rank: 48 x 510 MB = 24.5 GB per chunk
= ~0.12 s at the DMA wall, attention and projections far less): per 256-token chunk at short context 20 x (8.9 + 8.9) + 20 x (9.4 + 9.1) + 4 x (16.4 + 13.5) +
4 x (Ga + Gm) ~= 0.97 s of launches; beyond 64 pages (1k tokens) the P = 512 attention of the 8 global visits costs **107 ms
each** (dense far scores over 8192 keys, the Ornith lesson of section 13.2 of docs/neuron-qwen35-moe.md) - that is why 1k+ prompts
run at 160-183 tok/s. Measured per NEFF (layer_bench, ms): `sa` 8.89, `sm` 8.88, `Sa` 9.43, `Sm` 9.06, `ga` 16.4 (P 64) /
107.4 (P 512), `gm` 13.5.

What the measurements showed on the way:
* a FUSED global prefill visit (attention with paged far keys + MOE=all in one NEFF) takes **117 ms**; either half alone is
  16.8 / 21.7 ms; split into two NEFFs 16.1 + 9.3 = 25.4 ms (the loop's `S` visit likewise 28.9 -> 17.5). Expert groups inside
  the MoE (`MOE_EG`) made it worse (s 61, g 108-120 ms).
* the MoE of the WT layout in prefill: one relaid matmul over 128 experts 15.4 ms, groups of 32 experts **679 ms**, of 8 9.0 ms
  (non-WT 8.1).
* the 8-row decode bucket: `torch.topk` over 8 rows returns NaN on trn1 (`ROUTE=pad64` fixes it); under WT the gathered
  gate|up of the 64 slots in one matmul is NaN too (`WT_MOE=grp WT_GRP=16` fixes it, but the result is slower than two
  4-row launches, so the set has no 8-row bucket).

## 8. HBM per NeuronCore (runtime accounting with the set loaded)

| | MiB |
|---|---|
| tensors (weights 12.01 GiB + state 1.37 GiB: 9 slots x 48 K\|V buffers and conv states, 8 x 2049-page far K/V) | 11 839 |
| shared scratchpad (the max over the loaded NEFFs) | 448 |
| DMA rings | 1 320 |
| model code | 360 |
| **total / peak** | **14 007 / 16 032** (allocation fails at ~16 300) |

32k context does **not** fit with the current graph forms: the P = 2048 NEFFs (decode segments and the global prefill attention)
raise the shared scratchpad to 2.5 GiB and the DMA rings to 3.0 GiB, and the load fails allocating the experts. The served
set therefore stops at P = 512 pages = **8192 tokens** of context per request (the pool has 2048 pages, so the KV pool is not
the limit). Getting to 32k needs the far attention in key blocks with an online softmax (as `ATT_BLOCK` of the Qwen graphs),
which bounds both the scratchpad and the 107 ms cliff above.

## 9. Serving

* `neuron_server --model <prepared dir> --prefill-chunk 256 --max-batch-requests 8` with `BEHEMOTH_NEURON_NEFF_DIR=<graph
  set>`; `NeuronModel` dispatches on `prepared.json` `format: inkling2-neuron-prepared` (`inkling_model.cpp`). Verified: an
  OpenAI chat completion returns `content` *"The capital of France is Paris."* with the thinking in `reasoning_content`,
  `finish_reason` stop (eos `<|content_model_end_sampling|>` 200006), 37.2 tok/s; streaming emits `reasoning_content` deltas.
* Tokenizer: `tokenizer.json` (o200k-style Split regex + ByteLevel) through the server's path, 0 mismatches vs HF on 10 texts;
  chat template: `neuron/inkling_chat_template.jinja` (the checkpoint's template minus what minja lacks: a trailing dict comma,
  `| float`, tojson / dictsort keyword arguments) renders identically to HF on 4 conversations (plain, system, multi-turn,
  reasoning history); the prepare step installs it as `chat_template.jinja` and keeps the original as `chat_template.orig.jinja`.
  The model's message markers reach the reasoning splitter as `<think>` / `</think>` (`llama_text.cpp`;
  `ChatBackend::drop_stray_think_close` drops the `</think>` that the message end after the answer becomes).
* Env: `OMP_WAIT_POLICY=PASSIVE`; `BEHEMOTH_NEURON_INKLING_SLOTS` (default 9 = 8 sequences + scratch; each slot is 99 MiB per
  core); `BEHEMOTH_NEURON_INKLING_NOSEG=1` (single-visit graphs), `BEHEMOTH_NEURON_INKLING_SPLIT=1` (prefer split visits),
  `NEURON_X_DUMP=1|2` (residual after every launch).
* Not supported: prefix cache (state_bytes = 0: the engine keeps it off), speculation, MTP, vision, the model's tool-call
  OUTPUT format (`<|message_model|>NAME<|content_invoke_tool_json|>{...}`; tools are not offered by the server for this
  family), per-request `reasoning_effort` (the template's default effort 0.9 is used, as HF without the argument).

## 10. Proposed `models` entry (deploy/ecs-neuron/models.auto.tfvars; NOT applied - copy the artefacts first)

Copy `tmp/inkling/prepared/tp2-fp8/` to `models/inkling-2-mini-uncensored-tp2-fp8/` and `tmp/inkling/graphs/inkling-wt-f512/`
to `graphs/inkling-2-mini-wt-f512/` in the models bucket, and build the backend image from this branch (it must contain
`inkling_model.cpp`, the split-visit planner and the tokenizer / reasoning changes; the box's first engine build predated the
split planner and ran a split part as the whole visit - garbage output - so image and graph set must match).

```hcl
  inkling = {
    model_s3_prefix    = "models/inkling-2-mini-uncensored-tp2-fp8"
    graphs_s3_prefix   = "graphs/inkling-2-mini-wt-f512"   # 26 NEFFs: WT decode segments R 1/2/4 x P 64/512, split prefill visits
    prepared_s3_prefix = ""                               # the prepared dir (r0/ r1/ L<layer>.bin) is the model dir itself
    model_ids          = ["inkling-2-mini-uncensored"]
    model_name         = "inkling-2-mini-uncensored"
    shm_size_mib       = 6144
    extra_server_args  = "--prefill-chunk 256 --max-batch-requests 8"   # 8192-token context per request (P = 512)
    extra_env = {
      STREAM_LOAD     = "0"
      OMP_WAIT_POLICY = "PASSIVE"
    }
    use_spot      = false
    spot_fallback = false
  }
```

Data residency: us-east-1 is fine for this model (only the 397B case is AU-bound).

## 11. Open items (in order of payoff)

1. **Far attention in key blocks** (online softmax) for the global visits: removes the 107 ms P = 512 prefill cliff (1k+ prompts
   would go from ~170 to ~250-270 tok/s) and is the prerequisite for 16k / 32k context (scratchpad / DMA-ring budget, section 8).
2. Prefill MoE: 8.9-13.5 ms per visit for 510 MB = 40-60 GB/s; a grouped (routed-only) MoE as in the Qwen graphs
   (`MOE=group`) would roughly halve it.
3. Decode: 2.2x the DMA floor; the next levers are an int8 K|V buffer (2 MiB per visit and row read every step), the router in
   bf16 (1.2 MB f32 per visit), launch overhead (8 + 1 launches; fewer, longer segments), device-side argmax / top-k for the
   200k vocabulary (host sampling measured ~3.7 ms per step before `OMP_WAIT_POLICY=PASSIVE`).
4. Tool-call output parsing for Inkling's format; per-request `reasoning_effort` into the template; prefix cache with state
   snapshots (the buffers + conv states are the per-sequence state, 99 MiB per core).
5. The 8-row decode bucket under WT (correct with `WT_MOE=grp WT_GRP=16` but slower than two 4-row launches).

## 12. Files

`neuron/inkling_ref.py` (oracle), `neuron/inkling_graphs.py` (graphs), `neuron/inkling_prepare.py` (prepare),
`neuron/inkling_chat_template.jinja`, `neuron/cpp/inkling_model.cpp` (engine), `deploy/neuron-inkling/build.sh`;
tests `neuron/tests/test_inkling_{ref,graphs_cpu,device,oracle_hf,tokenizer}.py`, `inkling_kl.py`, `inkling_gate3.py`.
