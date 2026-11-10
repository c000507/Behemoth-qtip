# Brief 20: why the engine's head costs 14.3 ms, and the device-argmax experiment

Real Devstral Small 2507, spec off unless noted, 64 greedy tokens, the box
(trn1, libnrt 2.33.10), the engine built from the repo.

## 1. Breakdown of head_ms (BEHEMOTH_NEURON_STEP_LOG timers, ctx 16)

The engine executes the layer NEFF 40x asynchronously (the launches return
before the device finishes; the runtime notes say so) and the head's
`nrt_tensor_read` is the synchronisation point. New timers split the head
phase (neuron_model.cpp `phase_head_submit_ms` = the submit wall,
`phase_head_read_ms` = the submit-to-read wall; neuron_engine.cpp
`sample_ms`):

| chunk | ms |
|---|---|
| layers: 40 async submits | 38.2-38.7 (0.95/submit) |
| head submit | 1.2 |
| **head read (= drain + head device + readback)** | **13.0-13.3** |
| gather (2 ranks' 65536-shards into [vocab]) | 0.04 |
| host sampling block (incl. forward wait accounting) | ~0.5 net |

The head read = the ~9.5 ms waiting for the queued layer device work (the
layer device time ≈ 1.2 ms vs the 0.95 ms submit rate) + the head graph
device time (3.4 ms standalone) + ~0.2 ms readback. **The brief's premise is
wrong: the ~11 ms is NOT logits readback / cross-rank merge / sampling** (the
gather = 0.04, the sampling = ~0.5, the readback transfer = ~0.2); it is the
layer drain. The token is device-bound: ~48 ms layers + ~3.4 ms head + ~4 ms
host = ~56 ms. A device argmax can only remove the ~0.7 ms of
gather+sample+readback, not the drain.

## 2. Device argmax (BEHEMOTH_NEURON_DEVICE_ARGMAX=1), built and measured

- Graphs (trace_model.py `HEAD_ARGMAX=1`): head1arg/head8arg/head64arg/
  head256arg/head1argf8 - the head graph returns per-row (max f32, argmax
  int32) instead of the [S, 65536] shard.
- Engine (neuron_model.cpp + neuron_engine.cpp): the Meta/HeadBucket carry
  `argmax`; the head tensor set adds output1; the pick prefers an argmax head
  and falls back to a logits head; the greedy gate (temperature==0, no logit
  processing, no constraints, env=1) routes the whole step through the device
  path; the host merges the two ranks' (max, idx) candidates. The
  spec-decode verify rows are covered (they are greedy compares by design).
  Default (env unset) = the old behaviour; non-greedy = always the old path.

| run | itl_p50 | tokens |
|---|---|---|
| baseline bf16 head, spec off, 64 tok | 55.99 | X |
| **device argmax, spec off** | **63.96** | **identical to X (64/64)** |
| baseline spec on (7 drafts) | 53.95 | Y |
| device argmax, spec on | 63.89 | **identical to Y** |
| 1.2k ctx baseline / argmax | 56.98 / 65.0 | identical |
| non-greedy (temperature 0.8) with the flag on | works unchanged (the gate routes it to the old path) | — |
| fp8 head + device argmax | nrt status 2 at head execute | not validated |

**The device argmax is 8 ms/token SLOWER**: torch.max's index tracking
lowers to a serial scalar-engine argmax over the 65536-wide shard (the
head1arg NEFF is 3.7 MB vs 861 KB for head8arg - a pessimized compile; the
head device time goes 3.4 -> ~11 ms). Token identity is perfect, so the
machinery is correct, but it does not ship. The ehs one-call head (jax
pipeline, fp8 head + argmax = 2.39 ms total) remains the proven fast
head+argmax form; porting the engine onto it is the follow-up if the head
ever becomes the binding term (today the drain is).

## 3. What would actually move the number

- The drain (≈9.5 ms/step) = the layer device time: only weight-byte or
  kernel-time cuts move it (spec decoding already delivers 22.5 t/s on the
  real engine).
- HEAD_FP8=1 head buckets (the repo's own quantized lm_head): ~2.1 ms device
  vs 3.4 = about -1.3 ms/token; optional, not the binding term.
- The head 14.3 ms is an accounting artifact of the async submit rate; the
  engine needs no fix there.

## 4. What the user changes in their stack

Nothing ships: the device-argmax path stays default-off behind
BEHEMOTH_NEURON_DEVICE_ARGMAX and is slower. If a future head-side win lands
(ehs-style head buckets), the change = the graph dir contents + an image
rebuild (behemoth-neuron-server tag -> Terraform backend_image_tag); no env
or bucket names change for the current behaviour.

## Validated vs not

- Validated: the head-phase breakdown (timers, 4 runs consistent), token
  identity of the device-argmax path (3 configs x 64 tokens), the
  non-greedy fallback, the fallback pick when no argmax bucket covers a
  shape.
- Not validated / rejected: the device-argmax performance (negative), the
  fp8 head + argmax combination (execute status 2), any shipping change.
