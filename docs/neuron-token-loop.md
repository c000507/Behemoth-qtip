# Devstral token loop (brief 17 + addendum, measured)

Harness: neuron/cpp/token_loop.cpp. Faithful decode loop on both ranks
(TP=2, collectives, real Devstral dims, dec1_128 bucket, 100-200 iters
steady state). Weights 278 MB/layer/rank resident, fp8, random-valid
buffers. EHS head graph (embed gather + norm + fp8 head + argmax) on
device. Host per step = position/slot int64 writes.

## Variants (ms/token, t/s, 128 tokens unless noted)

| variant | 64 tok | 128 tok | load |
|---|---|---|---|
| A: 40 loaded models (one nrt_load per layer, as the serving does) | 67.96 (14.7 t/s) | 68.99 (14.5) | 1.04 s |
| **S: ONE loaded model + 40 prebuilt tensor sets** | **62.15 (16.1)** | **59.46 (16.8)** | **0.05 s** |

- The dec1 NEFF is generic (weights/KV are inputs): loading it once and
  executing 40x with 40 prebuilt input tensor sets kills the model-
  rotation overhead: **-9.5 ms/token, +2.1 t/s, load 1.04 -> 0.05 s.**
  This is a one-line serving-stack change (cpp/neuron_model.cpp: one
  nrt_load_collectives per rank, per-layer tensor sets instead of 40
  model handles).
- The calibrated baseline (variant A, 68 ms) matches the user's
  measured 66.7 ms/token (15 t/s) within 1.5%: the serving is now
  explained end to end.

## Where the remaining 59.5 ms sits

| chunk | ms | note |
|---|---|---|
| 40 layer executes x ~1.33 ms device | ~53 | the read ceiling (372 MB/layer at 260-282 GB/s effective) |
| EHS head+embed+argmax execute | ~2.4 | measured standalone; noise-level in the loop |
| residual host/launch over 41 executes | ~4 | ~120 us/execute; the explicit-async target |

## The async env + the explicit API

- **NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS is REMOVED in libnrt
  2.33** (the runtime logs it is ignored). The user's Dockerfile
  installs NEURON_RUNTIME_VERSION empty = latest from the neuron apt
  repo = the same >=2.33 family: **the env does nothing on the user's
  stack either.** Brief-15's 1.133-vs-1.327 delta = run-to-run
  variance; retracted.
- The explicit replacement = nrta_execute_schedule (nrt_async.h):
  posts to the COMPUTE XU queue, returns a sequence number. BUT the
  dec1 layer graphs contain collectives, and the async collectives
  path (nrta_cc_prepare/nrta_cc_schedule) is marked "work-in-progress
  and subject to change" in the header. Wiring async executes around
  collective-containing graphs = the WIP area; not attempted (hang
  risk). If AWS ships it, the expected gain = the ~4 ms residual
  host/launch (17.5-18 t/s).

## Floor

The device work = 372 MB read/layer at the one-direction read ceiling
(~260-282 GB/s effective, measured) = 1.32-1.43 ms/layer = 53-57
ms/token = **17.5-19 t/s with zero host overhead**. Everything above
the floor = the host/launch residue (~4 ms) and the head (~2.4 ms,
partially overlappable). The context sweep (dec1_32/128/512 =
1.299/1.319/1.433) and the batch sweep (dec4/dec8 = 1.642/1.797,
68 us/row marginal) are in docs/neuron-layer-opt.md.

## Serving-stack changes (exact)

1. cpp/neuron_model.cpp: replace the per-layer nrt_load_collectives
   with ONE load per rank + per-layer tensor sets (the dec1 NEFF is
   generic; the weights/KV are inputs). = -9.5 ms/token, -1 s load.
2. The head bucket: build the ehs graph (token id -> token id) and
   drop the host embedding/sampling from the step loop. = -1 ms +
   host off the critical path.
3. No fusing (docs/neuron-fused.md: the fused graphs = 2.6-3.4x worse
   per layer). No async env (removed in 2.33).

## Real Devstral Small 2507 end-to-end on the box (brief 18)

Setup: real prepared weights (40 x 556 MB shards = 20.7 GiB), real
graphs (all buckets), real config/tokenizer, repo engine
(neuron_engine_cli built from the repo with the Dockerfile flags).
16-token prompt ("The capital of France is Paris. The largest planet
in the solar system is"), greedy, 32 new tokens.

| mode | wall | ms/token | t/s |
|---|---|---|---|
| greedy, spec decoding ON (the default; 63 drafted / 11 accepted) | 1.42 s | 44.5 | 22.5 |
| greedy, BEHEMOTH_NEURON_SPEC_TOKENS=0 (plain) | 1.86 s | **58.2** | **17.2** |

Step log (spec off): layers_ms = 38.07 (**0.95 ms/layer**), head_ms =
14.26, forward_ms = 54.96, launches 1. The output text = coherent
greedy continuation (Paris -> Jupiter -> Pacific Ocean -> Asia ->
Russia...).

## THE FINDING: the repo engine ALREADY IS variant S

SERVING.md: "One NEFF per bucket serves every layer; the layers differ
only in the weight and KV tensors bound to it at run time
(cpp/neuron_model.cpp)". The engine loads ONE dec1_128 model handle per
rank and binds 40 per-layer weight/KV tensor sets — exactly the
variant-S scheme I built in token_loop.cpp. My token_loop variant-A
(40 separate model handles, 68 ms/token) = an artificial pattern the
real engine never had; there is NOTHING TO PATCH. The ask "patch the
engine to variant S" = the shipped design.

Cross-validation: the real engine's 38.1 ms / 40 layers = 0.95 ms/layer
is BETTER than my synthetic loops (layer_bench 1.32, token_loop variant
S 59.5 ms/token) — the engine's RankWorker pipelines uploads against
executes across the layer sequence, which none of my synthetic loops
do. The loop numbers were conservative, not optimistic.

## What actually bounds the real engine now

| chunk | ms | share |
|---|---|---|
| 40 layers | 38.1 | 69% |
| head (bf16, head1 bucket, 671 MB/rank) | **14.3** | 26% |
| upload + other | 2.6 | 5% |

- The layers at 0.95 ms = BELOW my 1.32 synthetic estimate: the engine
  overlaps uploads with execution across layers. The read-ceiling floor
  (0.9-1.05 ms/layer-equivalent) = CONFIRMED REACHED.
- **The head = 14.3 ms = the largest remaining chunk** (26%): the bf16
  head1 bucket reads 671 MB/rank at only ~47 GB/s effective — far
  below the 210-260 the matmuls get. Levers: HEAD_FP8 (the repo has
  HEAD_FP8=1 = E4M3 lm_head rows, half the bytes), the head8/head16
  multi-row buckets, or the ehs-style one-call head. Getting the head
  to even 130 GB/s = -6 ms/token = 58.2 -> ~52 = ~19 t/s.
- The spec decoding (the default ON) = 44.5 ms/token = 22.5 t/s
  measured — the prompt-lookup drafts amortize the weight stream; the
  acceptance depends on the workload's repetition.

## Verdict for the user

- The engine = already the one-load + per-layer-tensor-set design;
  no patch is needed or possible (it is the shipped architecture).
- The measured real-model greedy decode = **17.2 t/s plain, 22.5 t/s
  with the default spec decoding** on this box (the user's 15 t/s =
  the serving-stack overheads: the longer contexts, the HTTP stack,
  the batch scheduling, possibly an older runtime).
- The next lever = the head (14.3 ms, bf16): HEAD_FP8=1 buckets or the
  ehs-style fused head = up to -6..-12 ms/token.
