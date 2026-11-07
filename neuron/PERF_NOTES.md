# Neuron performance notes (trn1.2xlarge, 8B Anubis-Mini checkpoint, layer 2/0)

All numbers: weights and KV cache resident on the device, run from C++ (`cpp/layer_bench.cpp`, libnrt,
no Python), synthetic inputs (timing only), batch 1 unless stated. Instance: trn1.2xlarge, 2 NeuronCores (v2),
neuronx-cc 2.26, torch-neuronx 2.9. Accuracy was only checked where stated.

## Decode layer, one core

| weights in device memory | resident | ms / layer-step |
|---|---|---|
| bf16 dense | 437 MB | 2.06 |
| int8 + row scale, cast in graph | 253 MB | 4.78 |
| FP8 E4M3, cast to bf16 in graph | 253 MB | 3.96 |
| FP8 x FP8 native matmul (activation scaled into E4M3 range) | 253 MB | 1.84 |
| FP8/NVFP4 packed, table-lookup decode in XLA | 243 MB | 56 |
| FP8/NVFP4 packed, bit-arithmetic decode in XLA | 175 MB | 100 |

**TP=2 (both NeuronCores), FP8 x FP8, real all-reduce after o_proj and down_proj:** about 1.04 ms / layer for decode
(real layers 0-31 of the 8B, 1.02-1.07 ms; 126 MB of weights per rank).
**Prefill, TP=2, FP8 projections (q/o still bf16 placeholders in this graph):** 128 tok 1.32 ms, 512 tok 3.13 ms,
2048 tok 12.77 ms per layer (attention materializes T x T); about 409 ms for a 2048-token prompt over 32 layers.

> Correction: TP numbers measured before commit "Force collectives into traced TP graphs" were invalid. `torch_xla`
> drops `xm.all_reduce` / `xm.all_gather` when it sees one process, so those graphs contained no collective at all and
> each rank returned its own partial sum (known-answer probe: ranks sending 1 and 2 got 1 and 2). Set
> `XLA_ALWAYS_ALLREDUCE=1` (trace_layer.py now does) and verify with `cpp/allreduce_probe.cpp` (both ranks must read 3).

## Facts about the toolchain
- Trn1/Trn2 reject `F8E4M3FN` (`NCC_EVRF051`); the flag the message suggests is not accepted by 2.26/2.28.
  `tools_patch_fp8_hlo.py` retargets the HLO to `F8E4M3` (same encoding below 240) before `neuronx-cc compile`.
- A graph with `xm.all_reduce(groups=[[0,1]])` traces to a collectives NEFF; load it on each rank with
  `nrt_load_collectives` and `NEURON_RT_ROOT_COMM_ID=host:port`.
- Legacy `neuronxcc.nki` kernels trace inside `torch_neuronx.trace`.
- `neuronx-distributed` does not import in the installed venv (transformers version mismatch).

## Packed-weight decode: correctness is done, speed is not
- In-graph QTIP decode + RHT matches the C++ oracle (rel L2 1.7e-3); TP=2 forms (q/up column-parallel, o/down
  row-parallel, with all-gather / all-reduce) match it too (`tests/test_qtip_tp.py`).
- A single TP=2 q_proj shard (2048 x 4096, 2.1 MB) costs 165 ms / step, 98 s and 9.5 GB RAM to compile; a whole
  QTIP layer compile reached 27.5 GB of the 32 GB host and was killed.
- NVFP4 NKI kernel (decode in SBUF, Tensor Engine matmul) is correct (simulator rel L2 2.2e-3) but at batch 1
  takes 0.9 ms for a 4096 x 1024 slice (no-decode floor 0.125 ms), and 16 ms for 2048 tokens (about 1% of Tensor
  Engine peak): the loop structure, not the decode, limits the batched variant.
- Rough ceiling for decoding packed 4-bit/2-bit weights on the Vector/GPSIMD engines: about 4-5 G weights/s per
  core. An 8B model (7.5 G weights) would need about 1.6 s per token on one core if every weight is decoded each step.
  Decode is paid per weight, so it only amortizes with many tokens per step (prefill, big batches).

## Open
- Numerics of the FP8 weight path at model level (about 4% per MLP matmul vs the NVFP4 reference).
- Tiled/pipelined NKI matmul skeleton, then QTIP trellis decode on top of it.
- Whole-layer / whole-model assembly and KV cache residency across steps.

## Load-time QTIP -> FP8 expansion (C++, `cpp/qtip_expand.cpp`)
The checkpoint stays QTIP; at load the host decodes the trellis, applies the Hadamard rotations and SU/SV and
writes this rank's E4M3 row-scaled shard (max 224) for the TP FP8 graph (no extra collectives at run time).
Measured on layer 0 of the 8B with `cpp/qtip_expand_test.cpp`, TP=2, 8 vCPU, weight-only error vs
`reference_forward` (FP8 activations add more):

| projection | split | expand (both ranks) | relative L2 |
|---|---|---|---|
| q_proj 4096x4096 | column | 0.39 s | 2.85% |
| o_proj 4096x4096 | row | 0.39 s | 2.43% |
| gate_proj 14336x4096 | column | 1.80 s | 2.70% |
| up_proj 14336x4096 | column | 1.82 s | 2.61% |
| down_proj 4096x14336 | row | 1.85 s | 2.69% |

About 6 s per QTIP layer, so roughly 3 minutes for 32 layers at load. Resident cost is 1 byte per weight (QTIP is
0.25), which is the price of running QTIP layers at the FP8 decode speed (0.99 ms / layer at TP=2).

## Real-layer check (C++, `cpp/layer_run.cpp`, TP=2, one decode step at position 0)
Real weights of the 8B bound to the corrected TP graph (`trace_layer_real.py`; QTIP and NVFP4 expanded/re-encoded to
FP8 shards at load, FP8 k/v re-scaled), compared with a double-precision CPU layer (per-format reference forward):

| layer | MLP / attention format | relative L2 vs CPU reference | ms / step |
|---|---|---|---|
| 0 | QTIP | 0.27% | 1.05 |
| 2 | NVFP4 MLP, QTIP attention | 1.0% | 1.03 |
| 5 | NVFP4 MLP | 1.4% | 1.07 |
| 12 | QTIP | 1.9% | 1.02 |
| 31 | QTIP | 4.6% | 1.04 |

The device error equals a CPU emulation of FP8 weights + per-row FP8 activations (layer 31: 4.57% vs 4.57%), i.e. it is
the cost of the FP8 path itself. The residual stream dominates these outputs; error accumulation over 32 layers, longer
contexts (RoPE positions > 0, multi-token KV) and end-to-end token quality are measured in the next section (whole model against
the CUDA runtime: top-1 22 / 22, mean KL 0.02, positions to 15k).

## Server path (trn1.2xlarge, TP=2, 8B adaptive QTIP/NVFP4, cFP8, paged bf16 KV; see SERVING.md)

Measured through `behemoth::Engine` / `neuron_server`, end to end (embedding on the host, 32 layer launches, head, host sampling).

| | |
|---|---|
| model resident (weights expand + upload) | 85 s |
| decode, 1 request, ctx 100 (graph `1x1x32`) | 38 ms/token (26 tok/s); layers 32 ms, head 2.5 ms |
| decode, 1 request, ctx 300 / 1.5k / 14k (this session's binary) | 42.0 / 45.7 / 77.0 ms/token; with prompt-lookup drafts 34.6 / 44.8 / 77.3 (9.2 on text that repeats its prompt), see "Single-stream decode" |
| decode, 1 request, ctx 1.5k / 6k / 14k | 42 / 57 / 74 ms/token |
| decode, 4 / 8 concurrent requests (300-token prompts) | ITL 44 / 46.5 ms, ~90 / ~170 tok/s aggregate |
| prefill, 256-token chunks, ctx <= 2k | 3.3-3.4k tok/s (77 ms per 256) |
| prefill 4096 / 8192 / 14000-token prompt | 1.9 s / 4.4 s / 10.5 s (2.2k / 1.9k / 1.3k tok/s): attention grows with context |
| scheduler oversubscription (6 x 6000-token prompts + 400 new in a 32k-token pool) | 8 preemptions, all 6 complete, pool fully freed |

Accuracy against the CUDA runtime on the same checkpoint (teacher-forced logits, KL over the oracle's top-256):

| positions | top-1 agree | mean KL | worst KL |
|---|---|---|---|
| 13 positions 5 .. 1699 (around the 255/256/257 chunk edge and the 16-token page edges) | 13 / 13 | 0.020 | 0.124 (pos 255) |
| 9 positions 2047 .. 14999 (past the llama3 original 8192 context) | 9 / 9 | 0.018 | 0.100 (pos 2047) |

Batch behaviour: 13 concurrent requests (mixed lengths 6..1200 tokens, seeded sampling, one cancelled after 5 tokens) return the
same tokens as the same requests run one at a time when served by the same graphs; seeded requests reproduce. Different graphs
(1-, 4-, 8-row decode, 64- vs 256-token prefill) differ in fp8 rounding (~6-8 % relative logit difference between two
compilations of the same math), so greedy text can diverge late between requests that land in different buckets.

Where the time goes: decode is ~1.0 ms per layer at one row and 1.6 ms at 8 rows x 128 pages (the paged gather moves
`rows x pages x 16 KB x 2` per layer), so the bucket choice (fewest rows x pages that cover the step) matters more than launch
overhead (a one-graph 30-layer step is only 7 % faster). Prefill at long context is bound by the bf16 flash loop (score tiles
in f32 through the vector/scalar engines), not by the fp8 projections; an NKI fused flash kernel over paged KV is the next step.

### Single-stream decode (one request)

Measured through `neuron_engine_cli` with `BEHEMOTH_NEURON_STEP_LOG=1` (ctx = prompt of 300 tokens, `dec1_32` / `dec8_32`
buckets, 60 new tokens; ms/token = wall time of the decode steps / tokens emitted):

| | ms/token | per step |
|---|---|---|
| ctx 300, no drafts | 42.1 | layers 38.9 (1.22 ms/layer incl. launch), head 2.5, host ~0.6 |
| ctx 1.5k, no drafts | 57.8 | layers 44.4 (`dec1_128`) |
| ctx 14k, no drafts | see table below | `dec1_1024` |

Where the 42 ms go: 32 x `nrt_execute` of the layer NEFF at ~1.0 ms device time each plus ~0.1-0.2 ms launch gap (the
one-graph experiment in SERVING.md removed 7 % = ~0.12 ms per launch), the head 2.5 ms, host (embedding, RoPE rows, logits
read-back, sampling) under 1 ms. The per-layer device time at one row is the tensor engine streaming the 126 MB/rank of E4M3
weights: neuron-profile showed 0.65 ms PE-active per layer with 27.7k PE instructions for 110 MB of HBM reads, i.e. ~4 KB of
weight per instruction, ~128 weight bytes per cycle at 1.4 GHz = ~180 GB/s per core, which is the PE array's moving-operand
rate (one 128-element column per cycle; Trn1 runs FP8 at the BF16 rate, no double rate). At M=1 that rate is the same whether
the weight is the stationary or the moving operand, so fusing q/k/v and gate/up or an NKI matmul cannot cut the ~0.65 ms floor,
only the ~0.35 ms of non-overlapped scalar/DMA work around it; the only way past it is to not stream every weight per token
(speculative decoding below, or spreading the matvec over the vector/GPSIMD engines as well, not attempted). This analysis
of the instruction count is from the earlier profile, not re-profiled here.

**E4M3 lm_head** (`HEAD_FP8=1` head buckets; the runtime quantizes the lm_head rows per row at load, max 224): the head
launch goes from 2.5 to 2.1 ms although it reads half the bytes (262 instead of 525 MB per rank), which is what the element
rate above predicts: at one row the PE consumes 128 weight elements per cycle in either dtype (525 MB bf16 and 262 MB E4M3 are
both 1.46 ms of streaming), so FP8 buys HBM bytes and capacity, not time. Accuracy against the CUDA oracle: 13/13 top-1 and
mean KL 0.021 at the short positions, 8/9 top-1 (position 8191 flips at a near tie, KL 0.012) and mean KL 0.020 at the long
ones, against 22/22 and 0.019-0.020 with the bf16 head. Kept optional (bf16 heads stay the default); a bucket directory must
use one head format throughout (the runtime refuses a mix, since lm_head is uploaded in one format).

**Prompt-lookup speculative decoding** (SERVING.md "Speculative decoding"): the draft rows run in the `dec8_32` bucket, whose
layer costs ~1.4-1.5 ms instead of ~1.2 ms (8 rows x 32 pages gathered), so a verify step costs ~15 % more than a plain step
and emits 1 + accepted tokens. Measured while NEFF compiles ran on the host (absolute numbers ~10 % high; relative ones hold):

| prompt (300 tokens), 60 new tokens | drafts / step | accepted | ms/step | ms/token | greedy output |
|---|---|---|---|---|---|
| free-form continuation, no drafts | 0 | - | 48.9 | 48.9 | reference |
| free-form, up to 3 drafts | 60 over 47 steps | 12 (20 %) | 53.9 | 42.9 | diverges at token 9 (dec8 vs dec1 bucket numerics, see below) |
| free-form, up to 7 drafts | 122 over 45 steps | 14 (11 %) | 56.1 | 42.8 | same divergence |
| text that repeats its prompt, no drafts | 0 | - | 47.6 | 47.6 | reference |
| repeating text, up to 3 drafts | 51 over 18 steps | 41 (80 %) | 60.8 | 18.6 | identical to the reference for all 60 tokens |

The free-form divergence is the known 6-8 % relative logit difference between two compilations of the same math (the
1-row and 8-row graphs), not the acceptance logic: two copies of the free-form request run together (both in `dec8`, no
drafts) agree with each other and diverge from the 1-row run at token 1, and the repeating prompt, where every step
verifies drafts, reproduces the sequential output exactly.

Clean numbers (nothing else on the host; `BEHEMOTH_NEURON_SPEC_TOKENS=7`, drafting always on):

| context | no drafts, ms/token | 7 drafts, ms/token | accepted / drafted | verify-step layers vs plain |
|---|---|---|---|---|
| 300, free-form | 42.0 | **34.4** (3 drafts: 35.5) | 14 / 122 | 38.5 vs 36.7 ms (`dec8_32` vs `dec1_32`) |
| 300, repeating text | 42.1 | **9.3** | 48 / 70 | greedy output identical to the no-draft run |
| 1.5k, free-form | 45.7 | 48.9 | 7 / 210 | 52.3 vs 40.5 ms (`dec8_128`) |
| 14k, free-form | 77.0 | 105.0 | 3 / 133 | 114.8 vs 71.1 ms (`dec8_1024`) |
| 32 concurrent (300-token outputs) | ITL p50 62.0 ms, 425 tok/s | ITL p50 62.2 ms, 420 tok/s | no drafting at 32 rows | unchanged |

The verify bucket's cost is the paged K/V gather of rows x pages (0.006-0.013 ms per draft row and page), so at long context
drafting must pay for +14 / +44 ms per step with 3-5 % acceptance and loses. Hence the adaptive rule (SERVING.md): a request
drafts only while its acceptance EMA x plain step time exceeds 0.01 ms x drafts x bucket pages, and retries slowly. With it:

| context, 7 drafts, adaptive (default) | no drafts | adaptive | drafted / accepted |
|---|---|---|---|
| 300, free-form | 42.0 | **34.6** | 122 / 14 (verify within 10 % of a plain step: always on) |
| 300, repeating text | 42.1 | **9.2** | 70 / 48, greedy output identical |
| 1.5k, free-form | 45.7 | 44.8 | 49 / 3 (drafts in a third of the steps) |
| 14k, free-form | 77.0 | 77.3 | 0 (a verify step would cost +57 %) |

Before the margin and the cheap-verify rule the adaptive policy oscillated at 1.5k (drafting every other step, 50.1 ms/token)
and throttled the cheap case (39.6 at ctx 300); both are in the commit history.

Not done for single-stream latency, with what the measurements say about each: fusing q/k/v and gate/up (the PE element rate is
the floor, see above; the gain would be part of the ~0.4 ms/layer of non-overlapped work, unmeasured); fewer launches (7 % at
best, and the 32-layer graph is blocked by the layer-31 problem, SERVING.md); an NKI paged-attention kernel for the 14k case
(+35 ms/token of bf16 K/V gather and per-head matmuls, the biggest remaining item at long context); FP8 K/V (halves that gather).

### 70B on trn1 (Anubis 70B QTIP, 80 layers, hidden 8192, 64/8 heads, ff 28672, vocab 128256)

Weights per layer: 855.6 M (q, o 67.1 M each, k, v 16.8 M each, gate, up, down 234.9 M each). Per core at TP=2:

| resident form | per layer | 80 layers | note |
|---|---|---|---|
| FP8 shards (today's load path) | 428 MiB | 33.4 GiB | does not fit the 16 GiB HBM (a layer-truncated run is possible, below) |
| QTIP trellis as stored (2 bit + SU/SV + tlut) | 107 MiB | 8.6 GiB | fits with lm_head (1 GiB bf16 / 0.5 GiB E4M3) and KV (2.6 GiB for 1024 pages x 80 layers bf16; 1.3 GiB as FP8 K/V): 12-13 GiB per core |
| FP8 shards at TP=8 (trn1.32xlarge) | 107 MiB | 8.6 GiB | fits; each core then streams 107 MiB of weights per layer, like the 8B at TP=2 today (~1 ms/layer: ~80-100 ms/token plus the cross-chip all-reduces; not run) |

Decoding the trellis in the graph is the only way to run the 70B on this instance, and the earlier kernels bound its speed:
the plain-XLA QTIP decode was 56-100 ms per 8B layer (one 2048 x 4096 shard 165 ms), the NKI NVFP4 decode reaches ~4.6 G
weights/s per core (0.9 ms for a 4096 x 1024 slice), and the Vector/GPSIMD ceiling for bit-unpack + lookup is ~4-5 G
weights/s per core. A 70B layer has 428 M weights per rank: ~90-100 ms per layer, ~7-8 s per decode token with every weight
decoded every step (both cores in parallel). Decode only amortizes over tokens per step, so a chunked prefill (256 tokens per
launch) and wide batches would be usable, single-stream decode would not. Host RAM is not the limit: the loader expands one
layer at a time (3.4 GiB f32 per 70B layer, 4 layers in flight on the 8-vCPU host) and 21 GiB of trellis data streams from disk.

**70B truncated run (verified on this instance, real Anubis 70B checkpoint, `NEURON_LAYERS=8`).** Buckets traced with
`MODEL=/home/ubuntu/models/anubis-70b` (dec1_32, pre64_32, head1, head64; PAGES=1025): the loader expands the 70B QTIP layers
(Hadamard sizes 8192 and 28672 = 7 x 4096) and the model is resident in 124 s: 4.17 GiB weights per core for 8 layers plus the
bf16 lm_head (1.0 GiB) + 0.25 GiB KV. A one-row decode step over the 8 layers takes 28-29 ms of layer time, 3.5 ms per 70B layer
(2.9x the 8B's 1.2 ms for 3.4x the weights: still the weight stream), and the head 4.7 ms; extrapolated to 80 layers that would be
~285 ms/token at TP=2 if the FP8 weights fit. 8B buckets against the 70B checkpoint are refused ("NEFF ... was traced for
hidden 4096 heads 32/8 ... the checkpoint is hidden 8192 heads 64/8 ... retrace with MODEL=..."). The numerics of the truncated
model were not checked (an 8-layer prefix has no oracle); the expansion code is the one the 8B passes the CUDA oracle with.

### Wide decode buckets (16 / 32 rows)

Per-layer decode at one row is bound by the tensor engine streaming the 126 MB/rank of weights (neuron-profile: 0.65 ms of
~1.0 ms), so single-stream latency has little left; aggregate throughput comes from more rows per launch. Gather cost is
rows x table pages, so the page bucket must be tight (contexts ~420 tokens, 300 new each):

| bucket | ITL p50 | aggregate |
|---|---|---|
| 8 rows x 128 pages (before) | 46.5 ms | ~170 tok/s |
| 16 rows x 128 pages | 75 ms | 202 tok/s |
| 32 rows x 128 pages | 164 ms | 172 tok/s |
| 16 rows x 32 pages | 54 ms | 275 tok/s |
| 32 rows x 32 pages | 68 ms | 381 tok/s |

Host side of a 32-row step was 15 ms of 74 (sampling 32 rows of 128k logits one after another, plus zero-filling the 16 MB logits
buffer). Sampling now runs rows in parallel (OpenMP) and the buffer is not zeroed: step wall 74 -> 61 ms, ITL p50 59.5 ms,
~446 tok/s at 32 concurrent requests. What is left is device time: layers 52 ms (1.6 ms/layer, tensor engine 0.82 ms active,
scalar engine 0.97 ms), head 4 ms. `BEHEMOTH_NEURON_STEP_LOG=1` prints every step's profile.

## Compressed QTIP (trellis resident, decoded in the graph; SERVING.md "Compressed QTIP layers")

Design question: (1) decode a tile into SBUF and feed the Tensor Engine (fused), (2) decode a whole layer into an HBM
scratch buffer per step and reuse the FP8 graph, or (3) a per-layer hybrid with FP8-expanded layers in the spare HBM. Measured
first (one NeuronCore, `cpp/layer_bench.cpp`, `tests/bench_qtip_nki.py`, the 70B gate shard at TP=2: 14336 x 8192 = 117 M
weights, 29.4 MB of trellis words):

| kernel (`qtip_nki.py`) | ms / step | G weights/s/core |
|---|---|---|
| fused decode + matmul, T = 1 | 22.6 | 5.2 |
| fused decode + matmul, T = 128 tokens (one decode shared by the 128 rows) | 30.1 | 3.9 (0.24 ms per token) |
| decode to HBM bf16 only (option 2's first half, no matmul) | 22.0 | 5.3 |

`neuron-profile` of the T = 1 kernel: GpSimd 18.4 ms of 20.6 ms active (the 896 `GATHER` instructions: 17.95 ms), Vector
2.1 ms (all the uint32 shift / multiply / mask work), Tensor 0.7 ms, DMA negligible. The codebook lookup (one GpSimd gather per
pair of weights, 1024-entry bf16x2 table, ~3.4 cycles per lookup per GpSimd core) is the whole cost; everything else overlaps.
Things that did not help: writing the gathered pairs through a strided destination vs contiguous slices (same speed, the
latter kept because the matmul view absorbs the interleave for free), `nisa.local_gather` (returns one value per partition
of a 16-partition group for every index, so only 1/16 of its output is usable here: slower), a 16-bit table (the index must be
uint32). A 65536-entry table indexed by the raw window would remove the s(s+1) arithmetic but needs 256 KB per partition
(SBUF is 192 KB) and the arithmetic is not the bottleneck anyway. Decoding to HBM costs the same as fusing, so option (2)
would only add the scratch traffic and a second launch: the fused kernel is used, (3) is the `auto` policy on top of it.

Two hardware facts found on the way (the simulator does not model either): `nisa.scalar_tensor_tensor` rounds a uint32
product through fp32 (the low bits of s(s+1) are lost; `nisa.tensor_tensor` multiplies exactly), and `torch_xla` drops
`xm.all_gather` from a single-process trace (the HLO has no all-gather at all; `XLA_ALWAYS_ALLREDUCE` only keeps all_reduce),
so the compressed graph gathers with an all-reduce of zero-padded slices.

Correctness: the kernel matches the numpy mirror of `qtip::decode_rows_bf16` on the simulator (2e-7) and, on the device, the C++
reference decode of real layers (8B layer 0 q_proj, layer 12 down_proj with the ragged 7168-wide input group; 70B layer 0 q_proj,
layer 40 gate_proj, layer 79 down_proj, both ranks, 8 and 200 tokens): relative L2 1.2-2.0e-7, i.e. f32 accumulation order only;
the numpy mirror itself is bit-exact against `cpp/qtip_decode_dump.cpp` rows of the 8B and 70B.

**FP8 weight layout (side finding, both paths).** The FP8 projections stream weights at ~145 GB/s (gate dot 29.4 MB in 201 us in
the layer profile) against a 410 GB/s HBM. A single-projection microbenchmark of the 8B gate shard at M = 1 (`fp8 x fp8`,
one core, 200 steps incl. launch): weight stored `[out, in]` and multiplied as `h @ w.T` (today) 0.270 ms; stored pre-transposed
`[in, out]` and multiplied as `h @ w` 0.219 ms. The transposed form sits at the PE element-rate floor (29.4 M elements / 128 per
cycle / 1.4 GHz = 0.16 ms + launch), the current layout pays ~25 % on top of it for the strided [128 in x 512 out] tile DMA.
Not changed here (it touches the default fast path's upload and graph); worth ~0.1 ms of the ~1.0 ms 8B decode layer.

**Status (end of this session, honest).** Implemented and pushed: the fused NKI decode+matmul kernel, the WEIGHTS=qtip layer
graph, the runtime's compressed / expanded / auto modes with per-format NEFF variants, the C++ packing (bit-identical to the
Python packing on q/down/gate shards), and the device checks above. Verified on hardware: the kernel against the C++ reference
decode on real 8B and 70B layers (1e-7); one compressed projection as a device graph incl. the Hadamards (1.7e-5); TP=2 probes
on both cores with real weights (`tests/test_qtip_tp_device.py` + `cpp/neff_io.cpp`): row-parallel o_proj 8.5e-7, column-parallel
q_proj 1.5e-5 with the final (tile + mask) gather; the full compressed 8B loads in 11-17 s (2.07 GiB weights + 2.0 GiB KV per core
vs 3.74 GiB expanded) and runs end to end. NOT yet verified: the end-to-end 8B oracle comparison (22 positions) and the 70B
generation in compressed mode. The last e2e run before the gather fix was wrong (KL ~4): root-caused to `index_copy` of the kernel
output (see above) and fixed in the graph (`gather_tp`), but with the retraced dec1_32 / pre64_32 qtip buckets the 2-position 8B smoke run is STILL wrong (top-1 0/2, KL 4.3) while the
isolated TP=2 projection probes pass: the remaining defect is in the full layer graph only (candidates, untested: the two
identically shaped gate/up kernel calls in one graph (CSE / aliasing), the T = 64 prefill chunk vs the T <= 8 probes, the FP8 k/v
path next to the kernels, the sel-based index_select of the q heads). Diagnose with NEURON_LAYERS=1 NEURON_DUMP_LAYERS=1
NEURON_DUMP_X (the padding rows x = 0 already match the fp8 path to 2.7 %, the real tokens blow up to |x| ~ 2000). Measured device time of a compressed 8B layer (neuron-profile, dec1_32, 1 row): 22.9 ms (GATHER 16.6 ms)
= the kernel numbers above; the engine's wall time per compressed layer was ~62 ms (host-side, unexplained, not yet profiled).
Projection from the kernel rate: 70B ~80 ms per layer per step, ~6.5 s per decode token, ~30 tok/s for 256-token prefill chunks;
the 70B compressed footprint is 8.6 GiB trellis + 0.66 GiB FP8 k/v + 1 GiB bf16 head + 2.6 GiB KV (16k) = ~12.9 GiB per core.
Next steps: find that defect, then retrace the qtip buckets (q8/q70 chain on the instance: `chain8.sh`), rerun `qtest.sh BUCKETS compressed`, `q70run.sh`,
profile the host-side per-launch overhead, and re-run the default FP8 oracle / 32-concurrent check (the default path was not
modified except for name-based input binding; its 2-position smoke check gave top-1 2/2, KL 0.004 after the change).

**Update: the end-to-end defect was found and fixed.** The gate|up all-gather gathered each rank's `[gate | up]` block with a mask built for a
`[gate (all ranks) | up (all ranks)]` layout, so rank 0's up slice got its gate values and rank 1's gate slice its up values (gather_tp lays ranks out
one after the other). Zero rows hid it (z = 0), which is why padding rows matched while real tokens blew up, and the q / o probes never touched
it. trace_model.py now gathers `[gate | up]` per rank and splits after the gather. Compressed 8B (24 compressed + 8 expanded layers; both NEFF
variants per bucket), retraced dec1_32 / pre64_32, first three oracle positions: top-1 3/3, mean KL(top-256) 0.0029 (pos 5 / 17 / 100).
Compressed 8B (24 compressed + 8 expanded layers, pre64_128 / dec1_128 / pre64_32 / dec1_32 graphs), 13 oracle positions 5 .. 1699:
top-1 13/13, mean KL(top-256) 0.0057 (worst 0.030 at pos 256), versus ~0.02 for the FP8-expanded path (the compressed layers skip the FP8 weight rounding).
The 256-token compressed prefill graph does not compile in practice (neuronx-cc walrus_driver used 24 GB for 50 min in anti_dependency_analyzer); use 64-token chunks.
