# GLM-5.3-Flash on Neuron (Trainium): feasibility, layout, status

Scope (as decided): **FP8 weights fully resident** (no QTIP 2-bit, no host/NVMe weight or expert streaming), FP8 K/V
(the MLA latent cache), tensor parallel + expert parallel across up to 32 NeuronCores of a **trn1.32xlarge** (trn2 noted).
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

GLMSTEP_TABLE

## 5. Measured on trn1.2xlarge

MEASURED

## 6. What remains, and what needs the big instance

REMAINING

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
