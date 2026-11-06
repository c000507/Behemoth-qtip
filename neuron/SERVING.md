# Serving Llama on NeuronCores (trn1, TP=2)

The server's own `Engine` / `Scheduler` / `PagePool` run unchanged on top of a Neuron model backend. Weights, the
paged KV cache and all activations stay on the two NeuronCores; the host supplies token embeddings, RoPE rows, page
tables and write slots each step and reads back the logits of the sampled rows.

```
runtime/src/server.cpp (HTTP, unchanged except the chat-backend hook)
  -> behemoth::Engine            neuron/cpp/neuron_engine.cpp   scheduler, sampling, cancel, metrics
     -> neuron::NeuronModel      neuron/cpp/neuron_model.cpp    forward(InferenceBatch) -> logits (libnrt, C++)
        -> NEFFs                 neuron/trace_model.py          paged-KV flash-attention TP=2 cFP8 graphs
```

## Build and run (on a trn1.2xlarge)

```sh
# 1. NEFFs (Python venv with torch-neuronx; one directory per bucket, PAGES = KV pages incl. one scratch page)
PAGES=2049 python neuron/trace_model.py OUT/dec1_32  layer 1 1 32       # R rows x Q queries x P table pages
PAGES=2049 python neuron/trace_model.py OUT/pre256_128 layer 1 256 128
HEAD_ROWS=1 python neuron/trace_model.py OUT/head1 head 1               # T rows in, S rows out
# 2. binaries (g++ + libnrt only)
neuron/cpp/build.sh BUILD server        # neuron_engine_cli and neuron_server
# 3. serve (checkpoint = Behemoth QTIP / NVFP4 / FP8 Llama layout, 8B geometry)
NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 BEHEMOTH_NEURON_NEFF_DIR=OUT \
  BUILD/neuron_server --model /path/to/checkpoint --port 8080
```

`neuron_server` is `runtime/src/server.cpp` linked against the Neuron `Engine` and a Llama-3 byte-level BPE
`Tokenizer` (`neuron/cpp/llama_text.cpp`, the checkpoint's chat template through `set_chat_backend`). It speaks the
same OpenAI chat API with streaming, stop strings, `/health`, `/metrics`. `neuron_engine_cli MODEL requests.json` drives the
`Engine` with token-id requests (all submitted together so the scheduler batches them) for tests.

## What each piece does

**Graphs (`trace_model.py`).** One NEFF per bucket serves every layer; the 32 layers differ only in the weight and KV
tensors bound to it, so nothing is compiled per layer. Buckets:

* `layer R Q P`: R requests x Q query tokens each over a page table of P pages. Decode is `R x 1`, chunked prefill is
  `1 x Q`. The runtime picks the cheapest bucket (R x P or Q x P) that covers the step, so a single short chat runs the
  `1 x 1 x 32` graph and a 14k-token context runs `P=1024`.
* `head T S`: final RMSNorm and this rank's half of `lm_head` for S selected rows of T.
* `model R Q P L S`: the same layers plus head as ONE graph (see "Single NEFF" below).

Per layer and rank: RMSNorm, q/k/v (column-parallel) -> RoPE -> KV scatter -> paged flash attention -> o_proj
(row-parallel, all-reduce) -> RMSNorm -> gate/up -> SwiGLU -> down (row-parallel, all-reduce).

* **cFP8.** All seven projections are `fp8 x fp8` matmuls on the tensor engine in decode AND prefill: the activation is
  scaled per row (amax/224) into E4M3, the weight is E4M3 with a per-output-row scale, the f32 accumulator is rescaled.
  QTIP, NVFP4 and FP8 checkpoint tensors are all expanded to that form at load (`qtip_expand.cpp`; QTIP is decoded and
  the randomized Hadamard folded in, so QTIP layers need no extra collectives). Attention score and PV matmuls are bf16.
* **Paged KV.** `[pages, 2 (K,V), kv_heads_local, 16, 128]` bf16 per layer and rank: the server's 16-token pages and
  page tables (`AttentionLayout`), K and V of a page in one tensor so a page gather is one DMA and each layer has a
  single aliased state tensor. The graph scatters the step's K/V to flat slots (`page * 16 + offset`), updated in place
  on the device (`enable_aliasing`). Page `pages-1` is a scratch page for padding rows.
* **Flash attention.** Keys are processed in blocks of 32 pages (512 keys) with running max / sum / accumulator (online
  softmax); the `[queries x context]` score matrix is never formed, and causal + length masking is one compare
  `key_position <= query_position` per block, so no per-request lengths are needed. This is the flash algorithm written in
  torch ops and compiled by neuronx-cc; the NKI `flash_fwd` kernel needs a dense `q_len == k_len` layout and was not used
  because chunked prefill over paged KV has neither.
* **Extended positions.** cos/sin rows are computed on the host in fp64 for each token's absolute position with the
  checkpoint's llama3 rope scaling, so any position up to the NEFF's page table (16k here, more buckets for more) works.

**Runtime (`neuron_model.cpp`).** `libnrt` through the C API, no Python: two ranks, one `RankWorker` thread each, tensor
sets built once per layer and ping-pong residual buffer. Weight upload (4 threads) takes ~85 s for the 8B. `forward`
splits the scheduler's mixed batch into decode rows (one launch, padded to the bucket's R with scratch pages) and
prefill chunks (one request per launch), writes the step tensors, runs the layers, the head for the sampled rows, and
returns `[n_sample][vocab]` logits that the engine samples on the host (greedy, temperature, top-k/top-p, per-request RNG).

**Engine (`neuron_engine.cpp`).** The GPU engine's contract (`submit` / `cancel` / `metrics` / `shutdown`, callbacks per
token, finish reasons, TTFT/ITL percentiles) with the scheduler's decode-first policy, page allocation, preemption and
chunked prefill. `Scheduler` got a configurable vocabulary size (default unchanged, 32768; Llama 3 has 128257).

## Verification (trn1.2xlarge, 8B adaptive QTIP/NVFP4 checkpoint, oracle = the CUDA runtime on an RTX A4000)

See `PERF_NOTES.md` for the numbers; the checks are:

* teacher-forced logits at 13 positions (5 ... 1699, around the 255/256/257 chunk boundary) and 9 positions up to 14999
  (past the llama3 original-context 8192) against the CUDA runtime: top-1 agrees at 22 of 22, mean KL(top-256) 0.018-0.020;
* 13 mixed concurrent requests (different lengths, greedy / seeded sampling, one cancelled) produce exactly the tokens
  of the same requests run one at a time, within a bucket choice; seeded requests are reproducible;
* HTTP: chat completion, SSE streaming, 8 concurrent requests, `finish_reason: stop` on `<|eot_id|>`.

## One NEFF per step?

The default is **one layer NEFF reused for all 32 layers** (nothing compiled per layer; 32 launches per step plus the head).
`trace_model.py ... model R Q P L S [body]` also compiles L layers (plus optionally the head) into ONE graph; the
runtime chains such "segments" (`body` segments that return the residual, then a head segment or the separate head NEFF),
so a step is 1-3 launches. Findings on this toolchain (neuronx-cc 2.26, torch-neuronx 2.9, Trn1):

* The launch overhead removed is small: 30 layers, 8 rows, 128 pages take 48.2 ms as one graph against 51.7 ms as 30
  launches (7 %); the rest is device time. A single short chat is 38 ms per token either way.
* Graphs with 4, 8, 16, 24, 28, 30 and 31 layers, chains of 16-layer segments, and segments bound to any layers 0..30
  (including repeated or shifted ones) match the per-layer path to fp8 noise (relative L2 of the residual 0.02-0.05).
* **Binding layer 31 of the 8B checkpoint into any multi-layer graph gives wrong results** (residual relative error 0.6-1.6,
  top-1 changes), whatever its position in the graph, with 1 or 8 rows, `-O1` or `-O2`, separate or merged K/V, either
  upload order, KV tensor swapped. Its weights are sane (no E4M3 exponent-15 codes, scales in the range of the other
  layers) and the same tensors are correct when run as a single-layer launch (the 22 oracle positions above). Not root-caused,
  so the 32-layer 8B uses the layer NEFF; the segment path is only used when the bucket directory contains `model`
  graphs, and each compile took 3-11 min and up to 13 GB of host memory.
* Debug hooks for this: `NEURON_LAYERS=n` (first n layers), `NEURON_USE_LAYERS=1` (ignore model graphs),
  `NEURON_DUMP_X=prefix` (residual after 16/31/32 layers), `NEURON_LAYER_ORDER="0-15,16-30"`,
  `NEURON_SEG_LAYERS="0-15;16-30,30"`, `NEURON_SEG_SHIFT=n`, `NEURON_LOGITS_DUMP=file`.

## Not carried over

* Prefix cache (KV save/restore) and structured outputs: `Engine::submit` rejects a constraint instead of ignoring it.
* FP8 K/V (the GPU server's KV format): this backend keeps bf16 K/V; FP8 K/V would double the pool and halve the gather.
* Model geometry: the graphs are traced for Llama 3.1 8B (hidden 4096, 32/8 heads); the C++ checks this and refuses others.
* The numerics of a bucket are not bit-identical to another bucket's (different fusion in neuronx-cc, fp8 rounding
  amplifies it, ~6-8 % relative logit difference between the 1-row and 8-row decode graphs), so greedy text can diverge
  late between requests served by different buckets; quality against the oracle is the same.
