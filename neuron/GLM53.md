# GLM-5.3-Flash on Neuron (Trainium): feasibility, layout, status

Scope (as decided): **FP8 weights fully resident** (no QTIP 2-bit, no host/NVMe weight or expert streaming), FP8 K/V
(the MLA latent cache), tensor parallel + expert parallel across up to 32 NeuronCores of a **trn1.32xlarge** (trn2 noted).
Llama / Mistral path: unchanged. The new binaries give logits bit-identical to an origin/rocm-to-neff build on the
8B oracle set (13/13 top-1, mean KL 0.022); `neuron/cpp/host_test.sh` passes (160 checks).
Both variants of the reference implementation in `glm53/` are covered: the original **320B MoE** (45 layers, 42 MoE) and
the **80-layer dense conversion**. Everything below that says "measured" was run on the trn1.2xlarge build host
(2 NeuronCores, TP=2) on reduced configurations; full-size numbers are projections from those measurements.

## 1. What runs where

| | trn1.2xlarge (2 cores, 32 GB HBM, 32 GB RAM) | trn1.32xlarge (32 cores, 512 GB HBM, 512 GB RAM) | trn2.48xlarge (64 logical cores, 1.5 TB HBM) |
|---|---|---|---|
| dense 80L, FP8 (75 GiB) | no (38 GiB per core at TP=2) | **yes**, TP=8 .. 32 | yes, TP=4 .. 64 |
| MoE 320B, FP8 (292 GiB) | no | **yes**, TP=EP=32 only (9.6 GiB per core) | yes, comfortably |
| reduced configs (tests below) | **yes**, TP=2 | | |

trn1.32xlarge quota is not available on this account yet (the On-Demand Trn vCPU quota is 64; trn1.32xlarge needs 128).
Nothing in the code is hard-wired to TP=2: TP, EP and the expert-TP degree (ETP = TP / EP) are trace-time and repack-time
parameters (`TP=`, `EP=` / `--tp`, `--ep`), the runtime takes them from the prepared directory and refuses NEFFs traced for
another layout.

## 2. Memory budget (FP8, per NeuronCore of 16 GiB)

Computed from the exact per-rank tensor shapes the repacker writes (`glm_graphs.weight_shapes`; E4M3 projections with f32
row scales, f32 mHC / norm / router tensors, bf16 `kv_b`, replicated tensors counted on every core):

| variant | layout | weights + head per core | MLA cache per token (all MLA layers, replicated) | KDA state per sequence | left for caches (16 GiB - weights - 1 GiB) |
|---|---|---|---|---|---|
| dense 80L | TP=8 | 10.67 + 0.12 GiB | 30.0 KiB bf16 / **20.1 KiB FP8** | 32 MiB | 4.2 GiB: ~220k cached tokens (FP8) or ~130 KDA slots |
| dense 80L | TP=16 | 6.16 + 0.06 GiB | same | 16 MiB | 8.8 GiB |
| dense 80L | TP=32 | 3.91 + 0.03 GiB | same | 8 MiB | 11.1 GiB |
| MoE 320B | TP=16, EP=16 | 18.7 GiB | | | **does not fit** |
| MoE 320B | TP=32, EP=32 (or EP=8 x ETP=4, or EP=1 x ETP=32) | 9.60 + 0.02 GiB | 16.5 KiB bf16 / **11.0 KiB FP8** | 4.5 MiB | 5.4 GiB: ~500k cached tokens (FP8) |

Whole-model FP8 sizes: dense ~75 GiB, MoE ~292 GiB (305 B routed-expert weights; the MTP layer and vision tower are not
loaded). The EP / ETP split does not change the bytes per core, only which expert slices a core holds.

Notes:
* The MLA latent (512) and the indexer's pooling keys / gates (2 x 128) are shared by all heads, so every core holds the
  whole cache of every MLA layer (replicated, not sharded). At 131k context the dense model's cache is 2.6 GiB per core
  per sequence (FP8 latent). Sharding it would need attention data parallelism (requests split over core groups),
  not implemented.
* KDA state is f32 [heads/TP, 128, 128] per layer and sequence; it is a fixed cost per live sequence, independent of
  context. The runtime keeps `SLOTS-1` sequences resident and spills the least recently used to host RAM (exact).
* Host RAM (512 GB on trn1.32xlarge): the repacker processes one layer at a time; the server keeps only the bf16
  embedding table (dense 2.0 GiB, MoE 1.2 GiB) and spilled KDA states.

## 3. Graphs and parallel layout (implemented)

`neuron/glm_graphs.py` (traced with torch-neuronx, compiled by neuronx-cc 2.26) - one NEFF per sub-block kind and bucket:

| sub-block | sharding over TP ranks | collectives |
|---|---|---|
| mHC pre/post (every sub-block) | replicated (S=4 streams x hidden residual, 20 Sinkhorn iterations on 4x4) | - |
| KDA | q/k/v, short conv, f_b, g_b, b, A_log, dt_bias by linear head; f_a, g_a replicated; recurrence per head on the device (f32 state, token loop in-graph for prefill chunks) | 1 all-reduce (o_proj) |
| sparse MLA | q_a, kv_a and the whole indexer replicated (every rank computes the same top-k); q_b, kv_b by head; absorbed latent attention; top-k pool selection on the device (`index_topk / index_kpool` pools + always-visible tail); static plain causal mask when the bucket has <= `index_topk` keys | 1 all-reduce (o_proj) |
| dense SwiGLU (with `swiglu_limit`) | gate/up column, down row | 1 all-reduce |
| MoE | router f32 replicated, sigmoid + bias + group limit + top-8 on the device (one-hot masks); routed experts: EP groups x ETP slices; shared expert split over all ranks | 1 all-reduce (shared + routed partials) |
| head | final stream mean + norm, lm_head by vocabulary rows | - (host gathers) |

Expert execution modes: `MOE=all` (every local expert on every token: static, the choice for prefill and wide batches) and
`MOE=gather` (decode: only the `top_k x T` selected local experts' weight slices are gathered, exact). Bytes read per core
for one decode token and MoE layer: all = E/EP experts / ETP; gather = min(E/EP, 8 T) experts / ETP. For batch-1 decode the
best layout is therefore **EP=1, ETP=32 with MOE=gather** (8 expert slices of 1/32 each: 6.3 MB per layer) - with the same
memory footprint as EP=32.

Runtime (`neuron/cpp/glm_model.cpp`, behind `NeuronModel`, used by `neuron_engine_cli` / `neuron_server` when the model
directory is a `glm_repack.py` prepared directory; the Llama/Mistral path is untouched):
* weights uploaded from per-rank prepared files, inputs bound by name (`meta.json` inputs + aliased states);
* MLA caches paged with the server's 16-token pages and page tables; KDA state in slots keyed by the sequence's first page,
  reset in-graph at position 0 (new or preempted sequence), LRU spill/restore to host;
* prefill chunks padded with position -1 tokens (no state update); decode rows padded with a scratch slot / page;
* the engine disables prompt-lookup speculation (a rejected draft would advance the recurrent state) and the prefix cache
  (restored pages carry no KDA state) for this model family.
Server: GLM tokenizer + the checkpoint's own Jinja chat template (`<think>` reasoning split as on the GPU server) and
**GLM tool calls** (`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`): tools rendered by the
template, `tool_choice` required / named / auto with a grammar over the tools' parameter schemas, streaming parser
(runtime/src/tools.cpp, also usable by the GPU server).

## 4. Expected throughput on trn1.32xlarge (projection)

Measured constants (trn1.2xlarge, this repo): FP8 weight streaming at decode ~190 GB/s per core (PERF_NOTES "DMA wall");
the per-launch cost of a GLM sub-block at hidden 8192 / TP=2 (section 5); intra-chip all-reduce 12-22 us. Not measured:
all-reduce latency over NeuronLink across 16 chips (assumed 50-100 us for a [rows, hidden] f32 vector).

Per decode step (batch 1): every layer costs (weight bytes per core / 190 GB/s) + a fixed cost + its all-reduces.
The reduced-size runs (section 5) bound the fixed cost: **~0.5 ms per sub-block launch** where weights are negligible
(tiny model: 8 launches in 4.2 ms; mostly the mHC pre/post chain - 20 Sinkhorn iterations of tiny ops - routing /
selection ops and the launch), and **~2.0 ms per real 8192-wide layer at TP=2** beyond its 2.6 ms of weight streaming
(4.6 ms measured; it includes attention / KDA compute on 32 heads, which shrinks with TP, and the mHC chain, which does
not). One NEFF per layer (`FUSED=1`) removed 0.47 ms per real layer (18.5 -> 16.6 ms for 4 layers, same logits to 1e-7
on the tiny model). The projection therefore spans fixed = 0.5 .. 1.5 ms per (fused) layer:

| variant, layout (FP8, trn1.32xlarge) | weights read per core per token | fixed (80 / 45 layers) | all-reduces | projected decode, 1 stream | 8 streams (aggregate) |
|---|---|---|---|---|---|
| dense, TP=8 | 10.8 GB -> 57 ms | 40-120 ms | 160 x ~0.05 ms | **~105-185 ms/token (5-10 tok/s)** | ~40-70 tok/s |
| dense, TP=32 | 3.9 GB -> 21 ms | 40-120 ms | 160 x ~0.1 ms | **~75-155 ms/token (6-13 tok/s)** | ~50-100 tok/s |
| MoE, TP=EP=32 (MOE=all) | 9.6 GB -> 51 ms | 23-68 ms | 90 x ~0.1 ms | **~85-130 ms/token (8-12 tok/s)** | ~60-90 tok/s |
| MoE, TP=32, EP=1 x ETP=32 (MOE=gather) | ~0.5 GB -> 3 ms | 23-68 ms | 90 x ~0.1 ms | **~35-80 ms/token (12-28 tok/s)** | ~80-200 tok/s |

Assumptions to verify on the instance: the fixed per-launch cost does not grow with TP (the mHC chain is replicated
work), cross-chip all-reduce of a few-KB vector at 32 ranks costs ~50-100 us, the head (vocab-sharded, 33 MB per core)
is ~0.5 ms, and the gathered expert slices of 64 columns (ETP=32) stream near the measured DMA rate.

Prefill: KDA runs its recurrence token by token inside a chunk (no chunked WY form yet). A 16-token chunk of the
4-layer real model takes 25.5 ms of layer time at TP=2 (one decode token: 18.5 ms), i.e. the chunk costs little more than a
decode step; the full dense model would prefill at roughly 16 tokens per (80 x ~1-2 ms) at TP=32, ~100-200 tok/s with
16-token chunks, more with wider chunks (64 / 128 tokens: traceable, the token loop is unrolled in the graph).
The fixed cost is the lever: an NKI kernel for mHC pre/post (one call instead of ~100 small ops), multi-layer NEFFs
attack it (asynchronous execution did not: section 5).

## 5. Measured on trn1.2xlarge

Oracle = `glm53` CPU reference (`glm53 trace`, BF16 arithmetic). "Mirror" = the graph functions of `glm_graphs.py` run on the
CPU (ranks as threads), which is what the NEFFs compute; FP8 in the mirror is exact (E4M3 codes widened, f32 sums).

**Real weights: the 80-layer dense conversion truncated to its first 4 layers** (layers 0-2 KDA, layer 3 sparse MLA;
hidden 8192, 64 heads, vocab 128256; `/mnt/evo/dense70b_nvfp4/p2s1150_bf16`, repacked TP=2 FP8), a 39-token English prompt:

| | rel L2 of logits (median / max over positions) | top-1 agreement | mean KL |
|---|---|---|---|
| CPU mirror, BF16 weights, TP=1 vs oracle | 5.5e-3 / 6.8e-3 | 39/39 | 8.4e-5 |
| CPU mirror, FP8 weights, TP=2 vs oracle | 5.2e-2 / 6.0e-2 | 38/39 | 4.7e-3 |
| **device, FP8, TP=2, one-token decode steps vs oracle** | **5.1e-2 / 6.0e-2** | **38/39** | **5.2e-3** |
| device, 16-token prefill chunks (chunk ends) vs oracle | 5.3e-2 | 3/3 | 4.7e-3 |
| device: 2 sequences as rows of each step, sequence 0 vs single-row run | 1.9e-2 | 38/39 | 6e-4 |

The FP8 loss (~5 % logit rel L2, KL 5e-3 on this 4-layer prefix) is the format's (row-scaled E4M3 weights and
activations); the device adds nothing measurable on top of the exact mirror. Long context with the sparse selection active (2112-token prompt, 16-token prefill chunks through the
`P=160` buckets, oracle at chunk ends 2047 / 2063 / 2079 / 2095 / 2111; from 2052 keys on only the top 512 of the
complete 4-token pools plus the tail are visible): rel L2 4.6-5.5e-2, KL 3.7-5.5e-3 at every position - the same as at
short context - top-1 3/5 (the two misses are near-ties at these KL values).

Timing, same model, trn1.2xlarge TP=2 (synchronous launches): weights resident in 7.6 s (2.37 GiB per core);
decode 22.4 ms/token for 1 row (layers 18.5 ms = 4.6 ms per layer, head 3.6 ms), 26.2 ms for 2 rows; a 16-token prefill
chunk 29.4 ms. Fused layer NEFFs (`FUSED=1`, 4 launches instead of 8): decode 20.4 ms/token (layers 16.6 ms), a 16-token
chunk 27.6 ms. 4 rows per step: 26.2 ms, fused 24.7 ms (162 tok/s aggregate vs 49 tok/s single stream). Async execution
(`BEHEMOTH_NEURON_GLM_ASYNC=1`) gained <= 4 % and one single-row run hung, so it stays off. Through `neuron_server` (OpenAI chat API, streaming, `/metrics`): TTFT 83 ms for a 42-token
prompt, ITL p50 25.9 ms (the 4-layer prefix produces no meaningful text, as expected).

**Tiny random checkpoints** (hidden 256, 4 layers KDA/MLA alternating, dense or MoE with 8 experts / top-2 / 2 groups,
`index_topk` 16 so the top-k pool selection is active from 17 keys on):
* CPU mirror vs oracle (`tests/test_glm_cpu.py`, all pass): TP=1 BF16 rel L2 median 7-8e-3, top-1 100 %; TP=2, EP=2 and
  EP=1 (expert TP), MOE=gather, chunked and padded prefill, two interleaved sequences: 1.3-1.5e-2 without selection; with
  selection, single positions differ by up to 40 % where a near-tied pool choice flips (then carried by the KDA state):
  discrete, not an error. FP8 per sub-block vs BF16: 3-5 %.
* Each sub-block NEFF on one NeuronCore (BF16, `tests/test_glm_device.py`) vs the mirror: all kinds, prefill and decode
  buckets, worst rel L2 7e-3 (states included).
* TP=2 sharded NEFFs through `cpp/neff_io.cpp` (`tests/test_glm_fp8_device.py`): BF16 <= 8.7e-3 for every kind; FP8 1-4 %
  against the exact mirror (the FP8 matmul itself matches the mirror to 1e-7, `tests/glm_fp8_probe.py`; the remaining
  difference is activations landing on the other side of an E4M3 rounding boundary after bf16-level differences
  upstream); expert TP (EP=1, ETP=2) 3.4e-2.
* Runtime end to end (`glm_run`): BF16 TP=2 decode vs oracle rel L2 median 2.2e-2 (dense) / 5.3e-2 (MoE, routing and
  selection flips); 16-token prefill chunk ends vs decode 0.9-1.3e-2; 3 sequences on 2 KDA slots (130 spills, 129
  restores) **bit-identical** to the same run with enough slots, 2 % slower; fused layer NEFFs bit-identical to sub-block
  NEFFs (1e-7) and 29 % faster (4.17 -> 2.97 ms per step).
* Tool calls (`runtime/tests/structured_tests.cpp` `glm_tools`, all pass; `neuron_server_fake` with the real GLM-5.3
  tokenizer and template): grammar accepts / rejects by schema (required keys, integer values, enums), streaming parser,
  history with assistant `tool_calls` and `tool` results rendered by the checkpoint template; `tool_choice` required and
  auto return `tool_calls`.

## 6. What remains, and what needs the big instance

Needs **trn1.32xlarge** (quota pending) or trn2:
* any full-size run: the dense model needs TP >= 8 (10.7 GiB per core), the MoE TP = 32 (9.6 GiB per core); all graphs,
  the repacker and the runtime are TP / EP generic but have only run at TP <= 2. Multi-chip collectives (TP > 2 over
  NeuronLink) are the main untested piece; the Llama path's TP > 2 code has the same status.
* full-size accuracy (logit diffs against the oracle over many positions) and the throughput table of section 4.
* the MoE checkpoint itself: not on the local disk nor in `s3://behemoth-neuron-models-912935854432/`. Download
  `zai-org/GLM-5.3-Flash` (original layout; the repacker reads block-128 FP8 `weight_scale_inv`, BF16, or per-channel FP8,
  not the NVFP4 conversion) - roughly 320 GB - onto the instance's NVMe; repacking needs one layer at a time in RAM.

Not done / known limits (any instance):
* MOE=gather: the first device trace crashed in torch-xla (a 1-D `topk`); with the 2-D form the gathered-expert MoE
  graph matches the CPU mirror on one core (BF16, rel L2 3.8e-3), and TP=2 with EP=1 / ETP=2 as FP8 NEFFs: each gathered MoE block within 3.2-3.5 % of the exact mirror (the
  usual FP8 rounding-boundary level), end to end as good as MOE=all against the oracle. Its speed gain needs a large E/EP (EP=1 or
  8 on trn1.32xlarge) and is unmeasured.
* fixed per-launch cost (~0.5 ms, section 4) dominates at TP=32: an NKI mHC kernel and multi-layer NEFFs are the next
  steps; async execution is opt-in only (small gain, one hang observed, and unsafe across KDA spills).
* MLA attention is single-pass over all keys of the bucket (no flash blocking): fine to a few thousand keys per bucket;
  long contexts need the blocked form (as `trace_model.py` does for Llama) and a top-k *gather* of the selected 2048 keys
  instead of masking all keys (today the selection saves no work).
* the MLA cache is replicated on every core (latent shared by all heads); KV=fp8 (E4M3 latent + per-token scale) is
  implemented and checked on the CPU mirror (3-5 % per block), not yet run on the device.
* KDA prefill is a per-token loop in the graph (chunked WY form not implemented); no prefix cache, no speculation for
  GLM (both off by design: recurrent state); a sequence's KDA state is keyed by its first page.
* tool calls are implemented for GLM's own format (the MoE checkpoint's tokenizer); the dense conversion ships a
  Llama-3.1 tokenizer / template whose tool format is not implemented on Neuron.
* shared indexer layers (`indexer_types: shared`) and RoPE MLA are rejected (neither checkpoint uses them); MTP and
  vision weights are not loaded.

## 7. How to run

```sh
# repack (TP / EP of the target; a reduced model: --layers N)
python neuron/glm_repack.py /models/glm-5.3-flash /models/glm-tp32 --tp 32 --ep 1
# graphs (one directory per sub-block kind and bucket; MOE_DEC=gather for the decode MoE graphs)
MODEL=/models/glm-5.3-flash TP=32 EP=1 DECODE_R="1 4 8" PREFILL_Q="16" P="64 256" PAGES=4097 SLOTS=33 KV=fp8 MOE_DEC=gather \
  neuron/glm_build_buckets.sh /neffs/glm-tp32
# teacher-forced logits / timing, then the server
glm_run /models/glm-tp32 /neffs/glm-tp32 tokens.json out.f32 [1|16|rows2]
NEURON_RT_VISIBLE_CORES=0-31 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 BEHEMOTH_NEURON_NEFF_DIR=/neffs/glm-tp32 \
  neuron_server --model /models/glm-tp32 --port 8080
```
Checks: `tests/test_glm_cpu.py` (CPU mirror vs the glm53 oracle), `tests/test_glm_device.py` (each sub-block NEFF vs the
mirror, one core), `tests/test_glm_fp8_device.py` (TP-sharded NEFFs through `cpp/neff_io.cpp`), `tests/glm_fp8_probe.py`
(the device's FP8 matmul semantics), `runtime/tests/structured_tests.cpp` (`glm_tools`), `neuron/cpp/host_test.sh`.
