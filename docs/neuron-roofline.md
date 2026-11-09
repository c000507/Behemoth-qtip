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

- Marginal w-streaming rate at K4096 ≈ **139 GB/s** = ~60% of the
  elementwise DMA plateau (220 GB/s): the 512B chunk granularity costs
  ~40% at decode shapes.
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
2. **Decode-shape weight streaming runs at ~60% of the DMA plateau**
   (139 vs 220 GB/s): bigger staging chunks (or multi-round pipelining
   with fewer, larger descs) could buy ~1.7× on bandwidth-bound GEMV.
   The compiler does not do this; ours can (chunk size is our choice in
   the pool descs).
3. **Low-bit weights** multiply the same lever: NVFP4 = 4× fewer w bytes
   than fp32 on top of headroom 2 — quant study next
   (docs/neuron-quant-feasibility.md).
4. M = 1-8 GEMV (true decode): not yet synthesized (synth families start
   at M = 32). The compiler oracle numbers above are the reference until
   the M<32 family is decoded.
