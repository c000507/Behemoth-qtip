# Devstral fused-graph experiment (brief 16, measured on device)

Geometry: Devstral Small 2507 dims (hidden 5120, heads 32, kv 8, dim 128,
intermediate 32768, vocab 131072, 40 layers, TP=2), traced through the
repo's trace_model.py model-mode via a CONFIG json (the geometry is
CONFIG-driven, no code changes needed). Timing variant uses a bf16 KV
cache because the repo's whole-model path requires it (the fp8-KV path
needs 2 aliased state tensors per layer: the known 62-alias wall caps a
fused fp8-KV graph at ~31 layers, so a 40-layer fp8-KV fused graph is
structurally impossible).

## Fused N-layer graphs vs async executes (both ranks, 100 iters)

| graph | inputs | MB/rank | ms/execute | ms/layer |
|---|---|---|---|---|
| fused N=2 (bf16-KV) | 42 | 1235.7 | 7.414-7.668 | **3.7-3.8** |
| fused N=4 (bf16-KV) | 76 | 1800.3 | 11.80-12.12 | **2.95-3.03** |
| dec1_128 sync (fp8-KV) | 23 | 372.1 | 1.327 | 1.327 |
| dec1_128 async inflight=8 | 23 | 372.1 | **1.133-1.193** | 1.133-1.193 |

Compile: 39 s (N=2) / 61 s (N=4) — cheap. Performance: **the fused
per-layer time is 2.6-3.4x WORSE than async per-layer executes.** Even
discounting the bf16-KV gather (2x KV bytes), the fused path loses
badly: the compiler's cross-layer tiling degrades (the same finding as
the repo's 8B one-graph sweep, where the fused path beat only the SYNC
path by 7% — the async-inflight path removes the same launch gap
without the tiling penalty). Layer fusion is DEAD on Devstral dims,
now measured, not extrapolated.

## Embedding + final norm + fp8 head + argmax: ONE call works

Graph: token id -> embed gather -> final-norm scale -> activation
row-scaled into E4M3 -> fp8 x fp8 lm_head shard [65536, 5120] ->
argmax -> next token id. Single execute, single NC (the full TP head
adds the all-gather/argmax-merge; the shard here is exactly what one
rank runs):

| graph | ms/execute | note |
|---|---|---|
| head1 bf16 (repo head bucket, 671 MB/rank) | 3.371 | no embed/argmax |
| **embed+norm+fp8-head+argmax (this graph, 335 MB shard)** | **2.387** | token id in, token id out, greedy |

The one-call head+embed+argmax replaces (the bf16 head execute + the
host embedding + the host sampling) with ONE 2.4 ms execute. Measured
greedy output = a valid token id.

## Per-token projection (measured pieces)

| path | arithmetic | ms/token | t/s |
|---|---|---|---|
| current serving (user) | 66.7 | 15 |
| layers async + bf16 head + host | 40x1.19 + 3.37 + host | 47.6 + host | host=16.7 -> 15 |
| layers async + ehs one-call + minimal host | 47.6 + 2.39 + ~2 | **~52** | **~19** |

- The async env (NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=8) = -0.19
  ms/layer = -7.8 ms/token if the serving path is not already setting
  it (the repo sets it in cpp/neuron_model.cpp).
- The ehs one-call = -1 ms vs the bf16 head + it removes the host
  embedding + sampling from the critical path (the remaining host =
  RoPE rows + page tables + the 8 B token readback).
- Layer fusion contributes NOTHING (measured worse).

## What the user must change

1. Verify NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=8 in the
   behemoth-neuron-server image env (the repo sets it in
   cpp/neuron_model.cpp; the tfvars/serving env must not override it).
2. Build head buckets as the ehs graph (trace: the same pipeline; the
   weights = the fp8 lm_head shard + the embedding table as inputs)
   and switch the serving loop to: token-id -> ehs execute -> token id.
3. Keep per-layer dec1_* executes with the async env (do NOT fuse).
