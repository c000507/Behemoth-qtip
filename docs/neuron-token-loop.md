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
