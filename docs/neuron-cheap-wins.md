# Brief 22: cheap wins + gap accounting for Devstral Small decode (trn1.2xlarge, 2026-11-10)

Reference point: real engine, spec off, short prompt = 58.2 ms/token = 17.2 t/s
(brief 18; t/s below = 1000 / (58.2 - saved_ms)). Layer baseline dec1_128,
both ranks, drained (layer_bench 200 iters) = 1.327 ms/step re-measured this
session. All graphs: Devstral geometry (hidden 5120, 32/8 heads, ff 32768,
vocab 131072, 40 layers, TP 2), neuronx-cc 2.26 on the box.

## 1. Head: fp8 lm_head bucket vs bf16 (layer_bench, 200 iters, drained)

| variant | inputs MB/rank | ms/execute | saved vs bf16 | implied t/s |
|---|---|---|---|---|
| head1 bf16 (b18 bucket, re-timed same boot) | 671.1 | **3.414** | - | 17.2 |
| **head1 fp8** (`HEAD_FP8=1 WT=1 HEAD_ROWS=1 trace_model.py <dir> head 1`, built this session at `/home/ubuntu/work/b22/head1f8`) | 335.8 | **1.815** | **1.60 ms** | **17.7** |
| ehs one-call (embed+norm+fp8-head+argmax, 1-NC, `b16/wd_ehs/graph.neff`) | 1677.7 | 2.381 | 1.03 ms device + ~2 ms host (embed+sample leave the critical path) | ~18 |

- The fp8 head = a direct drop-in for the repo's head bucket contract (the
  engine already reads `meta.fp8` and enforces one head format per model, the
  brief-20 B2 check). PERF_NOTES estimated -1.3 ms for the 8B; on Devstral's
  671 MB bf16 shard the measured save is **1.6 ms/token**.
- The ehs graph re-confirms brief 48 (2.381 vs 2.387 ms). It is faster than
  the bf16 head but SLOWER than the fp8 head alone (it carries a 1.34 GB
  embedding gather per execute); its extra value = removing the host
  embedding + sampling from the step.
- Token identity NOT checked (needs a real-engine run; the documented fp8
  head behavior is near-tie top-1 flips, PERF_NOTES "13/13 and 8/9"). Not
  claimed here.

**Kept: the fp8 head1 bucket.** Exact user change: rebuild the head bucket
with `HEAD_FP8=1 WT=1` (`build_buckets.sh` already honors it for heads via
`HEAD_FP8`) and ship the bucket dir; the serving image needs no env change
(`neuron_model.cpp` picks up `meta.fp8`). -1.6 ms/token.

## 2. Compiler flags on dec1_128 (recompiled via trace_model NCC_FLAGS, 200 iters)

All variants recompiled fresh on this box with the bucket defaults
(WT=1 MLP=nki KV=fp8 ATTN_DEC=nki). Caveat: the variants got the trace
default `PAGES=64` KV pool (1 MB) while the b18 baseline has 5561 pages
(91 MB); per brief 14 the KV capacity costs ~20 us per 4x, so the bias
favors the variants by at most a few tens of us — the deltas are below the
keep bar regardless.

| variant | NCC_FLAGS | ms/step | delta | verdict |
|---|---|---|---|---|
| baseline (b18 graph) | none | 1.327 | - | - |
| optlevel 3 | `--optlevel 3` | 1.324 | -0.003 | reject (< 0.03) |
| optlevel 1 | `--optlevel 1` | 1.321 | -0.006 | reject (< 0.03) |
| model-type | `--model-type transformer` | **1.423** | +0.096 | reject (regression) |
| saturate | `--enable-saturate-infinity` | 1.346 | +0.019 | reject |

All four accepted by neuronx-cc 2.26 and compiled fine. Nothing clears the
0.03 ms/layer keep bar: the layer is weight-DMA-bound (briefs 12-14), and no
flag changes the byte count or the descriptor geometry. **Nothing kept.**

## 3. Gap accounting: device-side inter-execute gap

Method: layer_bench wall for N back-to-back executes with one drain read at
the end (the marginal W(N)-W(N-1) = the device cost of one more execute
including any inter-execute idle; nrt_execute is async, submits hide behind
the device).

| graph | W(1) | marginal W(2)-W(1) | steady W(200)/200 |
|---|---|---|---|
| dec1_128 | 1.350 | **1.292** | 1.327-1.346 (1.346 this boot) |
| dec1_32 | 1.311 | 1.293 | 1.309 |

- The marginal execute costs the same as the steady-state step: **no
  measurable device-side inter-execute gap** beyond run-to-run noise
  (~20-30 us). The per-layer wall ≈ the device time.
- Cross-check from the brief-19 neuron-profile of dec1_32 (profiler 2.31.13):
  `total_active_time` = 1.2817 ms (DMA active 1.103 ms, tensor 0.908 ms,
  hbm_read 278.8 MB) against the drained bench 1.30-1.31 ms → device idle
  ≈ 20-30 us/layer.
- Implied per-token cost: ≤ ~1.2 ms over 40 layers (~2 % of 58.2) — below
  the 0.1 ms/layer action threshold. **2-4-layer fp8-KV segment graphs NOT
  built** (FullModel refuses KV=fp8 whole-model anyway; a segment would
  chase ≤ 1.2 ms/token).
- Tooling note: the NTFF instruction trace is RECONSTRUCTED timing (per-op
  START/END derived from the static stream + DMA events, `EXEC TIME` field
  empty), not wall-clock — do not gap-account from `show-session -t`. The
  b19-style summary JSON (`total_active_time` et al.) is the usable profile
  output; dec1_128's own summary was not regenerable this session (the
  2.31.13 tool prints no report and neuron-expector's `view` is UI-only), so
  the dec1_32 numbers + the bench marginals carry the conclusion.

## 4. All-reduce latency (nccom-test, 2 workers, bf16, 200 iters)

`nccom-test -r 2 allr -b 8K -e 32K -d bf16 -n 200 -w 20`:

| size | avg time |
|---|---|
| 8 KiB | 2.49 us |
| 32 KiB | 3.09 us |

The layer's collectives = 2 all-reduces (o_proj, down_proj) x 40 layers =
80/token, each moving hidden 5120 x 4 B = 20 KB → ~2.7 us each →
**~0.22 ms/token = 0.4 % of the token**. Not the feared 1-2 ms, far below
the 5 ms flag. Collectives are a non-issue for decode; no change.

## Verdict

| step | kept | change for the stack |
|---|---|---|
| fp8 head | **yes (-1.6 ms/token → ~17.7 t/s)** | rebuild head1 with `HEAD_FP8=1 WT=1`; token identity unverified |
| compiler flags | no | - |
| inter-execute gap | none found (≤ 30 us/layer) | no segment fusion |
| all-reduce | none needed (0.22 ms/token) | - |

Validated: all timings both ranks (or 1-NC as marked), drained clocks, same
boot. Not validated: fp8-head token identity end to end, the ehs host-side
saving inside the real engine loop.
