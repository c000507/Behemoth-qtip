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
* after the geometry/TP refactor and the speculative engine: the same 22 positions give 22 / 22 top-1, mean KL 0.019 (bf16 head);
  32 concurrent requests: ITL p50 62 ms, ~425 tok/s with and without drafting enabled (drafting does not engage at 32 rows);
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

## Model geometry and tensor parallelism

`trace_model.py` takes the geometry (hidden, heads, kv heads, head dim, intermediate, vocab) from a checkpoint's
`config.json` (`MODEL=/path/to/checkpoint` or `CONFIG=config.json`; the default is Llama 3.1 8B) and the TP degree from
`TP` (default 2), and writes both into every `meta.json`. The runtime takes TP and the geometry from the buckets,
compares them with the checkpoint and refuses a bucket directory traced for another geometry or a mixed TP (buckets
from before this field are taken as 8B / TP=2). Sharding: q/k/v/gate/up column-parallel (`heads`, `kv_heads` and
`intermediate` must divide by TP; `kv_heads < TP` would need K/V replication and is refused), o/down row-parallel with an
all-reduce, lm_head by vocabulary rows padded to a multiple of TP, embeddings on the host. Every rank runs the same NEFF
(`nrt_load_collectives(..., rank, TP)`); TP > 2 (several chips, `NEURON_RT_ROOT_COMM_ID` on the shared host) is
implemented in the same code path but has only been run at TP=2 on the 2-core instance. The load report prints the
per-core footprint (`weights + KV`), e.g. the 8B: 4.3 GiB weights + 2.1 GiB KV for 2048 pages.

Formats: every projection of a layer is expanded to a row-scaled E4M3 shard at load (`qtip_expand.cpp`): QTIP trellis
(decoded, Hadamard and SU/SV folded in, any `2^p / 12*2^p / 28*2^p` dimension the quantizer accepts), NVFP4 packed
(`weight_packed/weight_scale/weight_global_scale`), FP8 (`weight` + per-row scale), mixed per layer (the adaptive
checkpoints choose per layer). The checkpoint's `kv_format` does not matter: K/V are produced by the graph in bf16.

**70B on a trn1.2xlarge (Anubis 70B: 80 layers, hidden 8192, 64/8 heads, ff 28672).** The geometry traces and compiles
(dec1_32 layer 34 s, pre64_32 99 s, heads 2 min) and the loader handles its QTIP layers, but the FP8 expansion needs 428 MiB
per layer per rank, 34 GiB per core for 80 layers: more than the 16 GiB HBM, so only a layer-truncated 70B (`NEURON_LAYERS=n`)
runs on this instance. The checkpoint itself is 2 bits/weight: ~107 MiB per layer per rank, 8.6 GiB for 80 layers, plus the
lm_head (1 GiB bf16, 0.5 GiB E4M3) and KV (2.6 GiB per core for 1024 pages = 16k tokens), 12-13 GiB per core: it fits only
with the trellis kept resident and decoded in the graph. See `PERF_NOTES.md` ("70B") for what that costs on this hardware
and for the TP=8 FP8 alternative.

## Speculative decoding (prompt lookup)

A decode step of a request may carry up to `BEHEMOTH_NEURON_SPEC_TOKENS` (default 7, 0 disables) draft tokens after its
scheduled token: the tokens that followed the most recent earlier occurrence of the last `BEHEMOTH_NEURON_SPEC_NGRAM` (3)
tokens of the history (prompt + output). The model runs the request's tokens as rows of ONE decode bucket (the graph scatters
every row's K/V before the attention, so row i attends to rows < i of the same launch) and returns logits for every row;
the engine samples each row and accepts draft i when row i - 1 sampled exactly it, stopping at the first mismatch. The token
sampled at the mismatching row is still exact, so the accepted sequence is distributed as in sequential decoding (greedy
output is identical; seeded sampling consumes the RNG differently). `Scheduler::reserve` allocates the pages the draft
positions need and `Scheduler::accept` advances a request by an already-computed token. Only steps with at most
`BEHEMOTH_NEURON_SPEC_REQUESTS` (4) decode requests draft, and never past `BEHEMOTH_NEURON_SPEC_ROWS` (8) rows in total, since
rows are what a decode bucket costs and a wide step has none to spare. A verify step is also dearer at long context (the K/V
gather is rows x pages: `dec8_1024` costs 44 ms more per step than `dec1_1024`), so drafting is adaptive per request: it drafts
while its running acceptance (accepted drafts per verify step, EMA from an optimistic start) times the plain step time exceeds
`BEHEMOTH_NEURON_SPEC_PAGE_COST` (0.01 ms) x drafts x the pages of the bucket the step would land in, and a request that stopped
recovers slowly to retry. Continuations of up to 8 tokens (draft verification, the tail of a chunked prompt) use the row path; prompts still
start in prefill buckets. The step log prints `drafts=` and `tokens=` per step, `/metrics` has `drafted_tokens` and
`accepted_drafts`.

## Not carried over

* Prefix cache (KV save/restore) and structured outputs: `Engine::submit` rejects a constraint instead of ignoring it.
* FP8 K/V (the GPU server's KV format): this backend keeps bf16 K/V; FP8 K/V would double the pool and halve the gather.
* The numerics of a bucket are not bit-identical to another bucket's (different fusion in neuronx-cc, fp8 rounding
  amplifies it, ~6-8 % relative logit difference between the 1-row and 8-row decode graphs), so greedy text can diverge
  late between requests served by different buckets; quality against the oracle is the same.
