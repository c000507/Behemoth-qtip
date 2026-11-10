# Neuron (trn1) roofline measurements

Brief 5 step 1. All numbers from trn1 i-0adbefb8af607c610 (libnrt 2.33.10,
single NeuronCore visible, /dev/neuron0), `nrt_execute` latency via
timing.cpp, 200 iters, mean and min. Generators: tools/roofline_gen.py.

## Setup

- Single-rank graph containers: our own synthesizers (fp32) and
  neuronx-cc-compiled reference graphs (offline comparator, v2.26-era
  compiler on the box venv).
- "DMA bytes" = total HBM traffic per launch (inputs + outputs).

## 1. Launch floor and per-op cost (chained single graph)

One graph, N chained elementwise ops (relu*1.0001, 4096 f32), compiled by
neuronx-cc (comparator), 200 iters:

| ops in graph | mean us | min us |
|---|---|---|
| 2  | 56.6 | 54.5 |
| 4  | 56.5 | 54.4 |
| 8  | 58.5 | 56.3 |
| 16 | 60.5 | 58.8 |

- Fixed launch/execute floor ≈ **56 µs** per nrt_execute (runtime + host
  round trip; independent of graph content at these sizes).
- Marginal per-op cost inside one graph ≈ **0.27 µs/op**.
- Consequence: N separate containers each pay the floor (7 single-op
  containers = 449.9 µs vs 77.3 µs fused transformer block, memory 26).
  Fusion across ops in ONE graph is the dominant win at small/medium
  sizes; the floor disappears only behind real work (see 3).

## 2. DMA plateau (elementwise y = 2.5·x·w, fp32)

n f32 elements ×3 arrays (2 in, 1 out). Our synthesizer (DVE mul):
numerics BIT-EXACT ≤ n=262144; at n=1048576 numerics FAIL (semaphore
counter overflow, P·16 > 65535) though the launch still completes.

| n | DMA bytes | mean us | min us | marginal GB/s | aggregate GB/s |
|---|---|---|---|---|---|
| 16383   | 196 KB  | 57.4  | 55.5 | –    | (floor) |
| 65536   | 786 KB  | 60.3  | 58.5 | 270  | (floor-dominated) |
| 262144  | 3.1 MB  | 73.2  | 68.9 | 183  | 43 (naive) |
| 1048576 | 12.6 MB | 114.2 | 108.4 | 230 | 110 (naive) |

Transfer-dominated rate (subtract 56 µs floor): ≈ **220 GB/s aggregate**
for 512B-granular chunked DMA through the pool engine. This is the
practical per-NeuronCore HBM<->SB plateau the runtime achieves at this
chunk size. (Theoretical per-core share is higher; the gap is chunk
granularity + per-chunk semaphore traffic.)

**Elementwise numerics cap found**: elementwise synth is correct only for
P = ceil(n/128) ≤ ~4095 (P·16 ≤ 65535, u16 semaphore wrap). Fix =
per-pass semaphore re-arm or 32-bit counters (decode needed).

## 2b. DMA-only sweep (no compute in the path, 2026-11-10)

Containers: the validated shr shape with the DVE work stripped — pool
desc0 = HBM->SB (x-in, sem 8) chained to desc1 = SB->HBM (y-out, wait
sem 8 GE 16); desc events: wait = u16@4, upd = u16@6; an assigned sem
(u32@12 = sem<<8) is auto-incremented +16 by the transfer completion.
The 4P desc fields = BYTES PER CHANNEL = n*4/128 — they MUST scale
with n (desc1 GE value = 16 = one desc completion). Run through the
fault-isolated neurond (tools/neurond/neurond.cpp, per-request child
process, 240 s kill). traffic = in + out bytes per execute.

Serial chain (HBM->SB then SB->HBM, 200-iter avg):

| n (elem) | bytes/dir | total | avg us | marginal GB/s (avg-56) |
|---|---|---|---|---|
| 128      | 512 B  | 1 KB    | 59.0  | ~0 (floor) |
| 512      | 2 KB   | 4 KB    | 58.9  | ~1 |
| 2048     | 8 KB   | 16 KB   | 59.2  | ~5 |
| 8192     | 32 KB  | 64 KB   | 59.4  | 19  |
| 16384    | 64 KB  | 128 KB  | 59.9  | 33  |
| 32768    | 128 KB | 256 KB  | 63.5  | 34  |
| 65536    | 256 KB | 512 KB  | 60.9  | 105 |
| 131072   | 512 KB | 1 MB    | 64.1  | 123 |
| 262144   | 1 MB   | 2 MB    | 66.9  | 183 |
| 524288   | 2 MB   | 4 MB    | 74.3  | 219 |
| 1048576  | 4 MB   | 8 MB    | 91.7  | 224 |

- Ceiling (serial) = **~220-225 GB/s aggregate** at >= 2 MB/dir;
  transfers < 128 KB/dir run at 1-100 GB/s (fixed per-transfer cost
  ~3-5 us + small-chunk inefficiency).
- PARALLEL variant (compiler 2-input and-oracle, 3 descs: 2x in
  concurrent + y-out; numerics a&b verified): n=131072: 67.3 us
  (1.5 MB, 133-163 GB/s); n=524288: 82.7 us (6 MB, 225-248); n=
  1048576: 105.4 us (12 MB, marginal 259-264 GB/s). **Aggregate DMA
  ceiling ~240-265 GB/s**; the 220 GB/s proxy number = right order,
  mildly conservative.
- The compiler mul marginal (395 GB/s, quant-feasibility doc) exceeds
  the DMA-only probes: its desc geometry pipelines better than our
  2-desc serial chain — the pseudo-desc internals (packet splitting at
  DMA_MAX_DESC_COUNT_PER_PACKET=64 x 64 KiB) are opaque; treat
  250-400 GB/s as the desc-geometry-dependent envelope.
- **Queue sweep = IMPOSSIBLE via def.json num_queues**: q1/q4 = exec
  timeout (5) at a shape that passes with the default 16 — the
  semaphore wiring is tied to the 16-sem set (descs reference sems
  8/9/10 of qPoolDynamic's set); shrinking the queue count breaks it.
  Parallelism must come from MORE DESCRIPTORS (more streams), not
  fewer queues.
- **Size cap = 4 MB/dir per pseudo-desc**: n*4/128 bytes-per-channel
  overflows u16 at n > 2M elem AND DMA_MAX_DESC_COUNT_PER_PACKET(64) x
  64 KiB = 4 MB; the 8 MB probe hard-faulted the exec unit (101).
  Bigger transfers need a real multi-desc chain (undecoded).
- "Both cores" = not exercised: single-graph containers bind 1 NC;
  2-core aggregate needs either 2 concurrent processes (device has 2
  NCs) or cc_streams collectives (deferred).

## 3. Weight streaming (decode-shaped matmuls, fp32, compiler)

w[K,N] streams from HBM through the SB w-region in 512B row-chunks; the
multi-round machinery repeats [A-phase][B-phase] per w slice. Our
synthesizer currently covers K ≤ 1024 (single round); K ≥ 2048 = the
multi-round path (compiler only, decode pending).

| M×K×N | w bytes | mean us | min us | w-stream GB/s (floor-sub) |
|---|---|---|---|---|
| 32×1024×128 (ours, synth) | 512 KB | 62.7 | – | 35 |
| 32×2048×128 (oracle)      | 1.0 MB | 68.2 | 64.0 | 84 |
| 32×4096×128 (oracle)      | 2.1 MB | 71.8 | 68.7 | 139 |
| 512×2048×512 (oracle)     | 4.2 MB | 136.0 | 129.6 | – |

- Marginal w-streaming rate at K4096/N128 ≈ **133-139 GB/s** = ~60% of
  the DMA-only plateau (see the DMA sweep below). NOT an arithmetic
  bug: the w rows = N x elem = 512 B at N=128 = the SMALL-CHUNK DMA
  regime (chunk size = row width, set by N). The GEMV sweep (K=N=4096,
  16 KB rows) reaches 165-243 GB/s. Consistent picture: the DMA is
  chunk-size-bound, not fixed at 220.
- 512×2048×512 = 1.07 GFLOP in 136 µs = **7.9 TFLOP/s fp32** (incl. all
  overheads) — the only shape where compute is visible above the floor.
- Our synth containers = byte-exact vs compiler streams at the covered
  shapes → same timings (parity confirmed earlier: 62.7 vs 62.4 µs).

## 4. Engine overlap

Inferred from timings (libnrt exposes no per-engine utilization):
- 512×2048×512: 8.4 MB DMA + 1.07 GFLOP in 136 µs → DMA 62 GB/s AND
  compute 7.9 TFLOP/s concurrently = DMA and PE overlap well inside one
  graph at large M.
- Small-M GEMV: DMA-bound; PE idle (M ≤ 128 adds nothing but launch
  floor + staging).

## 5. Dtypes

Our emitters: fp32 only so far. The PE in_dtype field accepts the DTYPE
enum (BFLOAT16, FP16, FP8_E4M3/E5M2, INT8) — bf16/fp8 weight variants are
untested; expected effect = 2-4× lower w-stream bytes for the same GB/s.
(Deferred to the quant-feasibility study, docs/neuron-quant-feasibility.md.)

## Conclusion — where the headroom is

1. **Launch floor (56 µs)** dominates anything below ~1 MB of traffic per
   graph. Multi-op fusion in one graph (0.27 µs/op marginal) removes it.
   This is already proven (5.8× block fusion, memory 26) and is the main
   lever our emitter stack exploits.
2. **Chunk-size-bound streaming**: the DMA plateau is ~220-265 GB/s
   aggregate for >= 512 KB transfers but collapses to 1-140 GB/s below
   ~256 KB/dir (and for 512 B-row layouts like N=128 weights: 133-139
   GB/s). Chunk size is largely set by the tensor row geometry; the
   lever = re-packing weights/layouts so the DMA sees big contiguous
   runs, IF the PE consumption pattern allows it (open question — the
   LW pattern reads N-row-shaped tiles).
3. **Low-bit weights** multiply the same lever: NVFP4 = 4× fewer w bytes
   than fp32 on top of headroom 2 — quant study next
   (docs/neuron-quant-feasibility.md).
4. M = 1-8 GEMV (true decode): not yet synthesized (synth families start
   at M = 32). The compiler oracle numbers above are the reference until
   the M<32 family is decoded.
