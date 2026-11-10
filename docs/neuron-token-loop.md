# Devstral token loop (brief 17 + addendum, measured)

> CORRECTION (2026-11-10, review): several numbers below are submit-rate
> artifacts and are superseded — see the inline CORRECTION notes. Summary:
> (1) "0.95 ms/layer" and "293 GB/s sustained" in the real-engine step log
> measure 40 async nrt_execute SUBMITS with no read; with the default async
> depth 8 the first 32 submits are free and the last 8 block at the device
> rate, so 38.5 ms / 32 = 1.20 ms and the 9.5 ms drain / 8 = 1.19 ms — the
> real per-layer device time is ~1.2-1.3 ms, consistent with layer_bench
> 1.30-1.33 minus the launch gap. (2) The 372 MB/layer "read per execute"
> was a capacity sum (weights 278 MB + the 91.1 MB KV pool + cscale): the
> KV pool is NOT read per execute (bucket scaling, profile hbm_read 278.8
> MB); the layer reads 278 MB at ~211 GB/s over 1.32 ms. (3) The head
> 14.3 ms was retracted by brief 20: it is mostly the layer drain, not the
> head matmul. (4) "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS removed in
> 2.33" is wrong: the env is ignored because depth-8 async is the default.
>
> DRAINED RE-MEASUREMENT (2026-11-10, clocks now stop after a drain
> nrt_tensor_read): layer_bench dec1_128 = 1.321-1.326 ms/step (unchanged
> vs 1.331 — the 200-step back-to-back loop is already device-saturated,
> so the bench figure was never submit-side); token_loop variant S =
> 58.74 ms/token = 17.0 t/s (was 59.46). Honest device floor: ~1.2-1.3
> ms/layer (the depth-8 re-derivation ~1.19-1.20 pure device; the bench
> 1.32 includes the inter-execute gaps) = ~48-52 ms/40 layers + ~3.4 ms
> head + ~4 ms host = the measured 56-58 ms/token, 17-19 t/s.

Harness: neuron/cpp/token_loop.cpp. Faithful decode loop on both ranks
(TP=2, collectives, real Devstral dims, dec1_128 bucket, 100-200 iters
steady state). Weights 278 MB/layer/rank resident, fp8, random-valid
buffers. EHS head graph (embed gather + norm + fp8 head + argmax) on
device. Host per step = position/slot int64 writes.

## Variants (ms/token, t/s, 128 tokens unless noted)

| variant | 64 tok | 128 tok | load |
|---|---|---|---|
| A: 40 loaded models (one nrt_load per layer; NOT what the serving does — see the CORRECTION below) | 67.96 (14.7 t/s) | 68.99 (14.5) | 1.04 s |
| **S: ONE loaded model + 40 prebuilt tensor sets** | **62.15 (16.1)** | **59.46 (16.8)** | **0.05 s** |

- The dec1 NEFF is generic (weights/KV are inputs): loading it once and
  executing 40x with 40 prebuilt input tensor sets kills the model-
  rotation overhead: **-9.5 ms/token, +2.1 t/s, load 1.04 -> 0.05 s.**
  This is a one-line serving-stack change (cpp/neuron_model.cpp: one
  nrt_load_collectives per rank, per-layer tensor sets instead of 40
  model handles).
- ~~The calibrated baseline (variant A, 68 ms) matches the user's
  measured 66.7 ms/token (15 t/s) within 1.5%: the serving is now
  explained end to end.~~ CORRECTION (2026-11-10): false. The engine loads
  one model per bucket and binds per-layer tensor sets (neuron_model.cpp,
  SERVING.md "One NEFF per bucket serves every layer"); variant A = an
  artificial pattern the engine never had, and the 1.5% match = a
  coincidence (this harness does no host work or readback, so it cannot
  explain the user's serving overheads).

## Where the remaining 59.5 ms sits

| chunk | ms | note |
|---|---|---|
| 40 layer executes x ~1.33 ms device | ~53 | CORRECTION (2026-11-10): the layer reads 278 MB of weights per execute (the 91.1 MB KV pool is written once at load and only the touched pages are written per step); 278 MB / 1.32 ms = 211 GB/s effective |
| EHS head+embed+argmax execute | ~2.4 | measured standalone; noise-level in the loop |
| residual host/launch over 41 executes | ~4 | ~120 us/execute; the explicit-async target |

## The async env + the explicit API

- ~~NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS is REMOVED in libnrt
  2.33~~ CORRECTION (2026-11-10): the env var is logged as ignored
  because depth-8 async execution is the DEFAULT in 2.33 (PERF_NOTES
  "async depth 8 (now the default in the runtime)"), not because async
  was removed. Brief-15's 1.133-vs-1.327 delta = the missing-drain
  signature (40 vs 200 steps: 32/40 vs 192/200 of the device work inside
  the timed window), not run-to-run variance. The user's image runtime
  version remains unverified (their Dockerfile pins nothing).
- The explicit replacement = nrta_execute_schedule (nrt_async.h):
  posts to the COMPUTE XU queue, returns a sequence number. BUT the
  dec1 layer graphs contain collectives, and the async collectives
  path (nrta_cc_prepare/nrta_cc_schedule) is marked "work-in-progress
  and subject to change" in the header. Wiring async executes around
  collective-containing graphs = the WIP area; not attempted (hang
  risk). If AWS ships it, the expected gain = the ~4 ms residual
  host/launch (17.5-18 t/s).

## Floor

The device work = 278 MB of weights per layer (CORRECTION 2026-11-10:
the earlier 372 MB = the capacity sum incl. the resident KV pool, which is
not read per execute) at the one-direction read ceiling (~211 GB/s
effective over the measured 1.32 ms; 260-264 GB/s = the multi-stream
ceiling) = 1.32-1.43 ms/layer = 53-57 ms/token = **17.5-19 t/s with zero
host overhead**. Everything above
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
3. No fusing (docs/neuron-fused.md: the fused graphs lose; CORRECTION
   2026-11-10: ~1.6x worse per layer after removing the head time and
   the retracted baseline, not 2.6-3.4x). No async env knob (it is
   ignored; depth-8 is the default).

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

Step log (spec off): layers_ms = 38.07, head_ms = 14.26, forward_ms =
54.96, launches 1. The output text = coherent greedy continuation
(Paris -> Jupiter -> Pacific Ocean -> Asia -> Russia...).
CORRECTION (2026-11-10): "0.95 ms/layer" measures the SUBMIT rate (40
async executes, depth-8 queue): the per-layer device time = ~1.2-1.3 ms
(38.5/32 blocked submits = 1.20; the 9.5 ms drain / 8 = 1.19), matching
layer_bench 1.30-1.33 minus the launch gap. The "engine faster than my
synthetic loops" cross-validation below rests on the artifact and is
retracted; the variant-S 59.5 ms/token number stands as measured (the
harness includes its own sync reads).

## THE FINDING: the repo engine ALREADY IS variant S

SERVING.md: "One NEFF per bucket serves every layer; the layers differ
only in the weight and KV tensors bound to it at run time
(cpp/neuron_model.cpp)". The engine loads ONE dec1_128 model handle per
rank and binds 40 per-layer weight/KV tensor sets — exactly the
variant-S scheme I built in token_loop.cpp. My token_loop variant-A
(40 separate model handles, 68 ms/token) = an artificial pattern the
real engine never had; there is NOTHING TO PATCH. The ask "patch the
engine to variant S" = the shipped design.

Cross-validation: ~~the real engine's 0.95 ms/layer is BETTER than my
synthetic loops...~~ RETRACTED (2026-11-10): 0.95 = the submit-rate
artifact; the device per layer = ~1.2-1.3 ms in both the engine and the
loops. The RankWorker upload/execute pipelining claim is unproven.

## What actually bounds the real engine now

| chunk | ms | share |
|---|---|---|
| 40 layers (device ~1.2 ms each + submit skew) | 48-52 device | ~85% |
| head graph device + read (brief 20 breakdown) | ~3.4 + 0.2 device | ~6% |
| host (embedding, sampling, readback, loop) | ~4 | ~8% |

CORRECTION (2026-11-10, brief 20): the "40 layers 38.1 / head 14.3"
split above measured submits, not the device. The head 14.3 ms is NOT a
head cost: ~9.5 ms of it = the layer drain and ~3.4 ms = the head graph
itself; the "671 MB at ~47 GB/s, the largest remaining chunk, -6..-12
ms/token" analysis is retracted. The remaining levers = the layer
device time (the weight bytes: already fp8; spec decoding = 22.5 t/s
measured) and the host/launch residue.
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
