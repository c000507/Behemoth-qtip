# Inkling 2 Mini (`inkling2_loop`, text) on trn1.2xlarge, TP=2

Target: `OpenUncensored/Inkling-2-Mini-Uncensored` (rev `748f78e`, Apache-2.0; base `thinkingmachines/Inkling-2-Mini`;
only the 20 sliding-layer attention output projections differ from the base). Reference semantics: transformers PR 51050
(`models/inkling2_loop`, commit `6173a2f`), whose `modeling_inkling2_loop.py` is the reviewed code the oracle follows.
Vision tower and the MTP / DSpark head are ignored. Branch `inkling` (from `ornith-wt` 1eeda00).

Status: served set `inkling-wt-32k` (31 NEFFs): **32 768 tokens of context**, prefill ~290-350 tok/s flat to 30k tokens, decode
24.1 - 24.8 ms per token (41 tok/s), prefix cache, tool calls, per-request reasoning effort. Sections 1-6 describe the model and the
decomposition; sections 7-8 are the first bring-up (8k set, kept for the record); **section 13 is the current state** (what
changed for 32k, the measurements, the rooflines).

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

## 7. Performance of the first set (`inkling-wt-f512`, 8k context) - superseded by section 13

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

## 8. HBM per NeuronCore, first set (superseded by 13.6)

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
* Round 2 (section 13.7): prefix cache with state snapshots, tool calls (offered, parsed, constrained for `required` / a
  named function), per-request `reasoning_effort`. Not supported: speculation / the DSpark head (13.8), vision.

## 10. Proposed `models` entry (deploy/ecs-neuron/models.auto.tfvars; NOT applied - copy the artefacts first)

Copy `tmp/inkling/prepared/tp2-fp8/` to `models/inkling-2-mini-uncensored-tp2-fp8/` and `tmp/inkling/graphs/inkling-wt-32k/`
to `graphs/inkling-2-mini-wt-32k/` in the models bucket, and build the backend image from this branch: graph set and engine
must match (the set needs the engine of section 13: far-page buckets, loop-pass prefill graphs with zero gates, the
attention / MoE split; an older engine fails at load or, as the first bring-up showed, runs a split part as a whole visit).
The prepared directory's `chat_template.jinja` must be `neuron/inkling_chat_template.jinja` of this branch (tools / call
JSON variables); `inkling_prepare.py` installs it, for the existing prepared set copy the file over.

```hcl
  inkling = {
    model_s3_prefix    = "models/inkling-2-mini-uncensored-tp2-fp8"
    graphs_s3_prefix   = "graphs/inkling-2-mini-wt-32k"   # WT decode segments R 1/2/4 x P 64/512/2048, split prefill visits Q 256
    prepared_s3_prefix = ""                               # the prepared dir (r0/ r1/ L<layer>.bin) is the model dir itself
    model_ids          = ["inkling-2-mini-uncensored"]
    model_name         = "inkling-2-mini-uncensored"
    shm_size_mib       = 6144
    extra_server_args  = "--prefill-chunk 256 --max-batch-requests 8"   # 32768 tokens of K/V pool (one request may use all of it)
    extra_env = {
      STREAM_LOAD                             = "0"
      OMP_WAIT_POLICY                         = "PASSIVE"
      BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE     = "1"      # state snapshots (198 MiB each on the host)
      BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN     = "1"
      BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL  = "2048"
      BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB  = "4"
    }
    use_spot      = false
    spot_fallback = false
  }
```

Data residency: us-east-1 is fine for this model (only the 397B case is AU-bound).

## 11. Open items

See 13.9 (the list of the first bring-up is closed: far attention in key blocks, 32k context, prefix cache, tools,
`reasoning_effort` are done; the routed prefill MoE and the 8-row bucket were measured and do not pay in this graph form).

## 12. Files

`neuron/inkling_ref.py` (oracle), `neuron/inkling_graphs.py` (graphs), `neuron/inkling_prepare.py` (prepare),
`neuron/inkling_chat_template.jinja`, `neuron/cpp/inkling_model.cpp` (engine), `deploy/neuron-inkling/build.sh` (one NEFF per
spec), `deploy/neuron-inkling/set.sh` (the served set); serving: `runtime/src/tools.cpp` (Inkling call format),
`runtime/src/protocol.cpp` (`render_vars`), `neuron/cpp/llama_text.cpp`, `glm53/src/tokenizer.cpp` (template variables);
tests `neuron/tests/test_inkling_{ref,graphs_cpu,device,oracle_hf,tokenizer}.py`, `inkling_kl.py`, `inkling_gate3.py`,
`neuron/cpp/inkling_tools_test.cpp` (host_test.sh).

## 13. Round 2: 32k context, prefill and decode speed, serving completeness (2026-11-22)

Same box (i-0cd98c451f018c0a3, trn1.2xlarge), same prepared set (`tmp/inkling/prepared/tp2-fp8/`, unchanged), new graphs and
engine. Graph set `s3://behemoth-neuron-models-912935854432/tmp/inkling/graphs/inkling-wt-32k/` (31 NEFFs, `MANIFEST.md5`),
built by `deploy/neuron-inkling/set.sh` (`WT=1 PAGES=2049`).

### 13.1 Before / after

| | first set (`inkling-wt-f512`) | this set (`inkling-wt-32k`) |
|---|---|---|
| context per request | 8 192 (the P = 2048 NEFFs did not load) | **32 768** (the whole page pool; needle found at 3k / 16k / 30k) |
| HBM per core, set loaded | ~16.0 GiB: full (rings 1 320, scratch 448, code 360; 13.6) | **14 663 MiB with the server up (rings 114, scratch 448, code 325); ~1.6 GiB free** |
| TTFT 1 026 / 4 026 / 16 026 / 30 026 prompt tokens | 5.62 / 23.8 s / - / - (183 / 169 tok/s) | **3.58 / 11.61 / 50.05 / 97.65 s (287 / 347 / 320 / 307 tok/s)** |
| TTFT 27 tokens | 1.0 s | 0.78 s |
| decode 1 stream (engine step, median) | 26.7 ms (37.4 tok/s) | **24.1 ms (41.4 tok/s) up to 2k tokens of context, 24.8 ms (40.3 tok/s) beyond** |
| decode 2 / 4 / 8 streams | 44.0 / 77.8 / 155.6 ms (45.4 / 51.4 / 51.4 tok/s aggregate) | **41.1 / 71.9 / 143.1 ms (48.7 / 55.6 / 55.9)** |
| decode after a 1k / 4k / 16k / 30k prompt | 28.9 / 28.2 ms / - / - | 25.5 / 27.0 / 30.9 / 30.8 ms |
| gate 3 (top-1 vs the fp8 oracle, 3 x 32 tokens) | 91 / 96 | **93 / 96** (misses at oracle margins 0.11 - 0.36) |
| long-context gate | - | 24 / 24 top-1 vs the fp8 oracle on a 2 526-token prompt (far attention active); needle retrieved at 3k / 16k / 30k |
| non-finite logits | 0 | 0 |
| serving | no prefix cache, no tools, fixed effort | prefix cache (state snapshots), tool calls, `reasoning_effort` (13.7) |

### 13.2 Why the first graphs were slow and did not scale (device profiles, `neuron-profile`)

The Q = 256 sliding attention part (`sa`, 8.77 ms) was **not** compute bound: the compute engines were busy 3.1 ms, the DMA
queues 7.9 ms, moving 2.6 GB of HBM for ~30 MB of logical data:

* **2.3 GB: the softmax output re-read from HBM once per head and key tile.** The scores were laid out `[query, kv head,
  rep, key]`; the value contraction is per kv head, so neuronx-cc transposed the 16 MB probability tensor piecewise for
  every tile of the dot. Fix: a **head-major layout** `[kv head, rep x query, key]` with the two contractions as batched
  matmuls over the kv heads (bf16 operands): 8.71 -> 4.84 ms.
* 55 MB / 22k packets: `cat([scores, sink])` before the softmax (a one-column misalignment of a 16 MB tensor). Fix: no
  softmax op; the sink enters through the running max / sum of an online softmax: -> 3.92 ms.
* 430k DMA packets of ~136 bytes: the relative-bias skew (pad / flatten / reshape shifts row t by t elements, which no
  engine does per partition). Fix `SKEW=bank`: the bias of query t is a matmul of its 16 relative states with ITS shifted
  copy of the profile bank (256 static pads of the [16, 1024] bank, shifts along the free dimension): -> **3.36 ms**; DMA
  rings of the NEFF 82 -> 2 MiB, scratchpad 128 -> 64 MiB. (`SKEW=bits`, log2 shift-and-select passes: 28.8 ms.)

The global visits' far attention (`ga`, 107 ms at P = 512 = 8k keys) had a second problem: **contracted straight out of the
page gather, the keys stay in their 16-token pages and every page becomes its own matmul with its own weight load** (164k
`MATMUL` + 164k `LDWEIGHTS`, 10 GB of spill / reload DMA at P = 2048). What works:

| far-key form, `ga` Q = 256, P = 2048 (32k keys), layer_bench ms | |
|---|---|
| one block, f32 (the first set's form; P = 512 only) | 107 at P = 512; P = 2048 does not load |
| key blocks of 256 pages, online softmax, bf16 dots | 45.9 |
| + head-major layout | 45.5 |
| + query blocks of 64 | 32.4 |
| blocks of 64 pages | 36.6 |
| keys / values of a block MATERIALISED by an exact multiply (the 2^-3 part of 1 / sqrt(D) on the keys, a power of two undone after the dot on the values; `FARK=dk`), blocks of 128 pages | **32.8** |
| the same with blocks of 256 pages / + query blocks of 64 | 319 (!) / 44.0 |
| blocks of 128 pages without the materialising multiply | compiler error NCC_IBIR229 (state buffer allocation) |

So the far attention now costs ~1.75 ms per block of 2048 keys and 256 queries (6.3 M scores): ~3.6 G score elements / s,
which is the activation engine's rate for the mask / exp / cast passes over a tensor that does not fit SBUF - that, not the
contraction, is the floor of this graph form. Buckets: P = 64 / 512 / 2048 far pages = 2 048 / 9 216 / 32 768 tokens of
context (the page table of a launch holds only the FAR keys, distance >= 1024; the engine picks the bucket by
`pages - 64`): 10.4 / 16.7 / 31.7 ms per global visit in the engine.

One numerics fact found on the way: **a bf16 x bf16 matmul returns a bf16-ROUNDED result on trn1 even when `.float()` follows
directly** (unlike fp8 x fp8, which accumulates into f32): `mmtest.py`, device == bf16(f32 reference) bit for bit. The
attention scores (|s| <= 45) therefore carry up to half a bf16 ulp (<= 0.06) of rounding. Device vs the f32-attention CPU
mirror: sliding part 7.4e-3 (the first set's f32 form: 7.6e-3), global part with far blocks 2.5e-2 - 2.9e-2 (first set's
P = 64 form 2.3e-2); gate 3 went from 91 to 93 of 96.

### 13.3 Prefill: where a 256-token chunk goes now, and the rooflines

Engine, rank 0, ms inside `nrt_execute` per 256-token chunk (96 launches: every visit split into attention + MoE; the 12
first-pass visit kinds run through the loop-pass graphs `S` / `G` with ZERO message gates bound by the engine - the gates
are then the identity, exactly - which halves the prefill NEFFs):

| | first set | now | per launch |
|---|---|---|---|
| sliding attention, 40 visits | ~366 (layer_bench sum) | **169 - 185** | 4.2 - 4.6 (layer_bench 3.4) |
| global attention, 8 visits: context <= 2k / <= 9k / <= 32k | 131 / 859 / - | **83 / 134 / 254** | 10.4 / 16.7 / 31.7 |
| MoE, 48 visits (dense: every expert on every token, `WT_MOE=grp WT_GRP=8`) | ~467 | **446 - 463** | 9.3 - 9.6 |
| host (embedding write 2.5 ms, launch dispatch) | | ~10 - 25 | |
| **chunk** | 0.97 s .. 1.6 s | **0.72 .. 0.92 s** | 280 - 355 tok/s |

**The dense MoE is at its roof.** Profile of the MoE part (9.0 ms): the tensor engine is busy 6.0 of the 9.0 ms with 58.5k
`MATMUL` + `LDWEIGHTS` pairs, each a full 128 x 128 x 128 tile; the work is 256 tokens x 128 experts x (2304 x 1152 +
576 x 2304) = **130 GMAC per visit and rank = 5.7 ms at the array's 22.9 TMAC/s**. 48 visits: 0.27 s per 256 tokens, i.e.
**~940 tok/s is the ceiling of any all-experts prefill of this model on a trn1 core pair even with free attention**; the
measured chunk is at 37 % of that ceiling (MoE 1.6x its roof, attention on top). `WT_GRP=4`: 470 ms (a cliff), `16`: 12.8.

**The routed-only MoE does not pay for this model** (built, exact on the CPU, run on the device with the overflow rerun; not
in the served set). `MOE=group` (the Qwen3.5 `moe_grouped_idx` with Inkling's activation: the `MOE_HOT` most loaded
experts of the chunk on every token, every other expert on `MOE_CAP` token slots, the largest slotted load returned, the
engine reruns the visit's MoE with the dense NEFF when it overflows - both ranks vote; the conv state is held back by the
graph in that case):

| MoE part, layer_bench ms | Q = 256 | Q = 1024 (per 256 tokens) |
|---|---|---|
| dense (`MOE_TSUB=256` at Q = 1024) | 9.06 | 35.4 (8.85) |
| grouped, hot 16, cap 40 / 128 | 8.98 | 27.0 (6.75) |

Two reasons. (1) **Inkling's routing is far more concentrated than Qwen3.5's.** Loads of the r-th most loaded expert in a
1024-token chunk (`MOE_GSTAT=24`, 384 MoE launches of a 9k-token prose prompt; median / maximum, tokens of 1024): r = 1:
649 / 1024, 2: 474 / 1024, 4: 327 / 957, 8: 220 / 495, 12: 167 / 236, 16: 134 / 191, 20: 114 / 165, 24: 97 / 140. The top 24
experts take ~62 % of the 8192 (token, expert) pairs; with hot 16 / cap 128 **49 % of the launches overflowed** and were
rerun dense (a 1024-token chunk then took 3.3 s). A fitting split would be ~hot 24 / cap 160. (2) **A slot row costs ~5x a
dense row**: per (row, expert) the dense form takes 0.27 us, the slot form 1.4 us (61 us per expert at 40 slots, 175 us at
128): every per-expert matmul reloads its 243 weight tiles for at most 128 rows, and the softcapped activation
(3 transcendental passes per element) runs on the padded slots. Profile of the grouped NEFF: tensor engine 15.7 ms (132k
matmul / weight-load pairs), vector engine 10.5 ms, activation engine 10.4 ms of 29.3 ms. With hot 24 / cap 160 the estimate
is ~30 ms per 1024 tokens against 35.4 dense: not worth a second bucket, its 576 MiB scratchpad, 3 s decode stalls and a
1024-token cache grid. The routed roof itself is ~3 ms per 1024 tokens (8 192 pairs x 3.98 MMAC = 33 GMAC, + 510 MB of
weights once): reaching it needs a kernel that streams an expert's weights once past all its token rows (NKI), not this
graph form.

A wide chunk does not help the attention either: `Sa` Q = 1024 with 256-token query blocks 12.7 ms (3.2 per 256 against
3.4), `Ga` Q = 1024 P = 2048 117.5 ms (29.4 per 256 against 32.8; 7 min compile, 576 MiB scratchpad). The served set keeps
the 256-token chunk only. `PRE_Q="256 1024"` in `set.sh` builds the wide bucket (the engine picks the chunk bucket by cost).

### 13.4 Decode

| engine step, median ms | first set | now |
|---|---|---|
| 1 row | 26.7 | **24.8 (6-visit segments) / 24.1 (12-visit segments: contexts up to 2k tokens)** |
| 2 / 4 rows | 44.0 / 77.8 | 41.1 / 71.9 |
| 8 rows (two 4-row launches) | 155.6 | 143.1 |
| head | 1.25 | 1.25 |

What changed: the attention forms of 13.2 in the segments (segment launch 2.91 -> 2.60 ms in layer_bench; P = 2048: 3.3 ms), and 12-visit segments for the 1-row, P = 64 bucket.
Profile of a 6-visit segment (3.3 ms in the profiler): 307 MB read per launch (= the cost model of section 5: 49 MB per
visit), DMA queues active 2.06 ms at ~150 GB/s, tensor engine 0.95 ms (22.7k matmul + weight-load pairs), i.e. the segment
is DMA bound at ~70 % of the wall rate; 12 % of its bytes are the K|V buffer shift (2 MiB read + 2 MiB written per visit).
At the 214 GB/s wall a segment would take 1.43 ms: **decode is at ~1.8x the DMA floor** (was 2.2x). Engine launches cost
~0.4 ms each over layer_bench.

Measured / decided, in the order of the brief:
* **fewer launches**: 12-visit segments (`seg_sssssgsssssg`, `seg_sssssgSSSSSG`, `seg_SSSSSGSSSSSG`, `seg_SSSSSGsssssg`: 4 launches + head
  instead of 8 + head) take the step from 24.8 to **24.1 ms** (segments 8 x 3.01 -> 4 x 5.63 ms: 0.2 ms per saved launch,
  not the 0.4 ms a launch costs over layer_bench). Built for 1 row / P = 64 only (2.7 - 3.5 min compile each). A 24-visit
  segment could not be traced: the tracer holds the 24 visits' placeholder weights, > 20 GB of host RAM, OOM-killed twice.
  Greedy text of the 12- and 6-visit sets differs from the first near tie on (token 6 of the first test prompt, "for
  explanation" / "for an explanation": the same tie that separates the runs of the first set), gate 3 below.
* **int8 / fp8 sliding K|V, ring instead of shift**: not done. The buffer is 4 % (read) + 8 % (shift write) of a segment's
  bytes; a ring needs the bias row rotated by a runtime offset (a gather, ~5 us per row on trn1) - worth ~1.5 ms per token
  at best.
* **bf16 router**: 1.2 MB of 49 MB per visit (2.5 %): not done.
* **on-device argmax / sampling**: not needed - with `OMP_WAIT_POLICY=PASSIVE` the host side is 0.13 ms sampling + 0.07 -
  0.1 ms logits read per step.
* **8 rows**: the union of 8 rows' experts is ~51 of 128 per visit (204 MB): the DMA floor of an 8-row step is ~50 ms, of a
  4-row step ~35 ms. Aggregate throughput is flat from 4 rows on (55.6 -> 55.9 tok/s): the target "8 rows > 2x one row"
  (> 80 tok/s) would need the 8-row step under 100 ms, i.e. within 2x of its floor, and the 8-row bucket measured 200 ms
  (WT needs `WT_MOE=grp WT_GRP=16` there). Not reached; two 4-row launches stay.
* **long context**: + 0.7 ms per segment at P = 2048 (the 64 MiB page gather of the global visit): 30.8 ms per token at 30k.

### 13.5 Gates of this set

* CPU mirror (`tests/test_inkling_graphs_cpu.py`, `EXACT=1`): 5.4e-6 - 6.7e-6 vs the oracle with far-key blocks
  (`ATT_BLOCK=1`), query blocks (`ATT_QBLOCK=2 / 4`), `SKEW=bank / banke / reshape / bits`, `FARK=dk / kd`, the grouped MoE
  (`PREFILL_MOE=group`: 240 launches, 0 fallbacks with hot 2 / cap 4; 152 forced fallbacks with hot 1 / cap 2, the held-back
  state checked), the loop-pass graphs on first-pass visits (`UPPER=1`: bit-identical results), WT, segments.
* Device vs CPU mirror (random real-shape weights): sliding attention part 7.4e-3, global parts 2.5e-2 (P = 2048, 16 key
  blocks) / 3.0e-2 (`G`, P = 512), per token median 2.9e-2 / max 6e-2 (uniform: no edge effect), both ranks bit-identical;
  6-visit decode segments cannot be judged this way (one top-8 flip in visit 2 of a single random token: 0.26 new,
  0.43 first set against the same mirror; the first visit agrees: attention 9e-3, MoE 5e-2).
* **Gate 3** (3 chat prompts x 32 greedy tokens, oracle teacher-forced, fp8 weights): **93 / 96** top-1 (31 + 32 + 30), the
  three misses at oracle margins 0.36 / 0.33 / 0.11; 0 non-finite.
* **Long-context gates**: (a) a 2 526-token chat prompt (prose), 24 greedy device tokens, the oracle teacher-forced over all
  2 550 positions with the fp8 weights (`pf.sh` + `inkling_gate3.py`, 33 min on the box CPU): **24 / 24 top-1**; every
  query beyond position 1024 attends far keys through the P = 512 bucket (4 key blocks). (b) the needle runs below at
  16k and 30k tokens (P = 2048, 16 key blocks). An oracle run at 16k+ tokens was not attempted (hours on this CPU).
* Needle (`needle.sh`: a code at the very start of the prompt, asked for at the end; `reasoning_effort` low; 256-token
  chunks): **found at 2 963, 15 964 and 29 965 prompt tokens**, 0 non-finite in 430 steps; also at 8 964 tokens through the
  Q = 1024 bucket with the grouped MoE and 188 overflow reruns.

### 13.6 HBM per NeuronCore (driver accounting, `mem.sh`, server up and idle)

| MiB | **served set: 31 NEFFs** |
|---|---|
| tensors (weights 12.01 GiB, 8 sequence slots x 99 MiB of state, 2048-page far K/V, IO buffers) | 13 741 |
| shared scratchpad (the largest NEFF's: the P = 2048 global prefill attention) | 448 |
| DMA rings | 114 |
| model code | 325 |
| **total** (after the smoke run 14 689; high-water mark of the whole session, profiles included: 15 484) | **14 663** |

Allocations fail at ~16 300 MiB: **~1.6 GiB per core are free**. The first set's recorded 14 007 MiB was a sample taken while
its process was unloading (its tensors figure, 11 839, is 1.9 GiB short of weights + state); its real steady state was its
recorded peak, ~16.0 GiB - full - of which **1 320 MiB were DMA rings** (the skew and the transposes of 13.2). That, not the
context length, is why the P = 2048 NEFFs could not be added then. The same sampling caveat applies to readings taken with
`memtest.sh`; use `mem.sh` against a running server.

What the room could hold: the Q = 1024 bucket (scratchpad 576, + ~310 rings, + ~230 code, + 132 IO: fits, ~0.9 GiB left;
not served, 13.3), or a 4096-page pool (64k tokens: + 512 MiB, every global NEFF rebuilt with `PAGES=4097` and a P = 4096
bucket, ~+1.4 ms per decode segment and ~60 ms per global prefill visit at full length) - not built.

### 13.7 Serving

* **Prefix cache**: `inkling_model.cpp` implements the recurrent-family contract (`state_bytes` / `state_save` /
  `state_restore`: one sequence's 48 K|V buffers + conv states, 99 MiB per core = 198 MiB per snapshot; `kv_page_bytes` /
  `kv_copy`: the 8 global visits' far K/V, 512 KiB per page for both cores). Enabled as for Qwen3.5:
  `BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE=1`, `BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN=1`, `..._PREFIX_INTERVAL=2048`,
  `..._PREFIX_STATE_GB=4` (20 snapshots). Measured through
  `neuron_server` (7 495-token prompt): cold 24.2 s; the same request again **2.50 s** (7 424 tokens restored, identical
  text); the conversation extended by a turn 2.55 s (7 424 of 7 512 tokens restored).
* **`reasoning_effort`** (`none | minimal | low | medium | high | xhigh | max`) goes into the template's "Thinking effort
  level" preamble (`ChatBackend::render_vars`; `none` needs no constraint for this family). Measured: the same question
  with `none` / `low` / default: 72 / 55 / 84 completion tokens (prompt 23 / 25 / 25 tokens: "level: 0" / "0.2" / "0.9").
* **Tools**: offered through the checkpoint's template (`tool_declare` system message; the specs and the calls of the
  history are written exactly as HF's `tojson(sort_keys, separators)` does - `glm53::Tokenizer::chat_tools` passes
  `tools_json` / `call_json`, minja's `tojson` cannot). Output format
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name": NAME, "args": {...}}<|end_message|>`: `ToolCallParser` modes
  `InkHead` / `InkArgs` (`runtime/src/tools.cpp`), a call grammar for `tool_choice` required / a named function;
  `tool_choice` auto is parsed, not constrained (every message starts with the same marker, so there is no trigger token).
  `neuron/tests/test_inkling_tokenizer.py`: rendering identical to HF for tools, tool results and efforts (9 conversations, 0
  mismatches); `neuron/cpp/inkling_tools_test.cpp` (host_test.sh): 51 checks. On the device:
  `auto` -> `finish_reason: tool_calls`, `get_weather {"city":"Paris","unit":"celsius"}` after a short reasoning message; the
  tool-result turn answers "Paris is currently 18°C with light rain."; `required` -> ONE constrained call in 1.7 s (the
  grammar allows one call: the constrained answer cannot think first, and forced past the message end the model kept
  calling until `max_tokens`); streaming emits `reasoning_content` and `tool_calls` deltas.
* Env additions: `BEHEMOTH_NEURON_INKLING_NOGROUP=1` (ignore grouped MoE NEFFs), `BEHEMOTH_NEURON_INKLING_MOE_LOAD_LOG=1`
  (log the loads a grouped launch returns). Step profile: `by` (rank 0's ms by launch class), `moe_grouped`,
  `moe_fallbacks`, `moe_load_max`, `state_saves`, `state_restores`.

### 13.8 DSpark / MTP head (`mtp.safetensors`): investigated, not built

Header (636 MB bf16, 54 tensors): three dense decoder layers (hidden 2304, MLP 9216: 3 x 127 MB each; attention with a
second K / V projection pair `*_ctx` onto the trunk's states), a `fuse_weight` [4, 4], `mask_embed`, and a "markov head"
(`markov_w1` [201 024, 256], `rnn_proj` [768, 5120], `hidden_out`). The transformers PR that the oracle follows has no code
for it, so there is no reference to validate a draft path against. Payoff estimate without it: a verify step of 1 + k tokens
costs what a (1 + k)-row decode step costs here, because the rows' experts are disjoint - 1.66x for k = 1 (41.1 vs 24.8 ms) -
plus ~3 ms for the draft (0.32 GB of fp8 weights per rank and draft pass + a head). Tokens per step 1 + p: break-even at an
acceptance of p = 0.78, **1.07x at p = 0.89** (the first-draft rate Qwen3.5's MTP reaches); more drafts do not help (4 rows
cost 2.9x one row). Not worth the 0.32 GiB and the verify buckets on trn1; it would pay on a core where a second row is nearly free.

### 13.9 What is left (in order of payoff)

1. **Prefill MoE as an NKI kernel** (60 % of a chunk, at its all-experts roof): one pass of an expert's weights past its
   own token rows; the routing statistics of 13.3 say 24 experts take 62 % of the pairs, so hot tier + slots is the right
   shape, the slot rows just must not cost 5x. Ceiling of the dense form ~940 tok/s, routed roof several times higher.
2. Far attention: 1.75 ms per 2048 keys x 256 queries is activation-engine bound in this form (mask, exp, cast as
   separate passes over an HBM-resident tensor); a fused score -> exp -> value kernel would make 32k prompts cost what
   2k prompts cost. int8 far K/V would halve the decode gather (0.7 ms per segment at 32k).
3. Decode: 1.8x the DMA floor; ring K|V buffers, fewer / longer segments beyond 12 visits (a 24-visit trace needs
   > 20 GB of host RAM on the 32 GB box), the 8-row bucket.
4. 64k context: fits the HBM on paper (13.6), needs the global NEFFs rebuilt with `PAGES=4097` and a P = 4096 bucket.
5. Structured outputs for this family: the constraint path does not know that every answer starts with
   `<|message_model|>` (not looked at); tool_choice auto is unconstrained.
