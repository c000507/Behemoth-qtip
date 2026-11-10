# Brief 19: attention fusion in the Devstral decode layer — measurements and verdict

All numbers from the box (trn1, libnrt 2.33.10, cc 2.26, the real Devstral
Small 2507 prepared weights + the repo engine), 2 ranks, layer_bench 200
steps / the engine `BEHEMOTH_NEURON_STEP_LOG=1`.

## What the decode layer actually is

`dec1_*` graphs (trace_model.py, ATTN_DEC=nki MLP=nki KV=fp8 WT=1) contain
**TWO** NKI custom calls, not four: `decode_attention_fp8_2p` (paged decode
attention) and `mlp` (`nki_mlp.mlp_fp8_tiled`). The XLA region between/around
them: RMSNorm, q/k/v fp8 projections, RoPE, fp8 KV quant + page scatter,
o_proj fp8 projection, two all-reduces, residual adds. `P < 128` buckets
(dec1_32) use the XLA torch-loop attention instead of the NKI kernel
(ATTN_DEC_MIN_PAGES).

## Profile: where the per-layer time goes

layer_bench (isolated execute) vs the engine: ~~the bench carries a constant
~0.38 ms/execute of runtime per-call overhead the engine's back-to-back
submits hide~~ CORRECTION (2026-11-10, review): the engine's step-log
"layers_ms" numbers below measure async nrt_execute SUBMITS (depth-8 queue:
32 of 40 free, 8 blocked at the device rate), so 0.95/0.98/1.06/1.17 are
submit-rate artifacts; the per-layer device time = ~1.2-1.3 ms in both the
bench and the engine. The bench/engine column gap is partly the missing-drain
signature, not a constant overhead.

| ctx (P) | bench full | engine layers/step | attn half (bench) | MLP tail = F−A |
|---|---|---|---|---|
| 32 | 1.300 | 0.95 | 0.210 | 1.090 |
| 128 | 1.331 | 0.98 | 0.232 | 1.099 |
| 512 | 1.428 | 1.06 | 0.333 | 1.095 |
| 1024 | 1.553 | 1.17 | 0.479 | 1.074 |

neuron-profile of dec1_32 (one execute, instrumented window 3.30 ms; the DMA
is not inflated, the compute engines are):

- hbm_read = 278.8 MB/execute = the full weight stream (confirmed).
- **DMA active = 1.08 ms @ 258 GB/s** (software-dynamic = the weight tiles).
- PE 27%, ACT 8%, scalar 4% of the window; union-active ~39%.
- DMA timeline (dma_throughput): ~285 GB/s during the MLP stream, **zero
  during the attention half** in an isolated execute.

Interpretation: the layer is **weight-DMA-bound**. The MLP tail (1.09 ms)
= the 252 MB/rank stream at ~231-258 GB/s. The attention half (0.21-0.48 ms)
is the only DMA-idle stretch inside one execute.

## ~~The engine already overlaps what fusion would add~~

~~The engine runs 40 layer executes back-to-back and sustains
278.8 MB / 0.95 ms = 293 GB/s — above any single-execute rate, i.e. the
next layer's weight DMA already fills the previous layer's DMA-idle
attention window. Candidate (d) "overlap the MLP weight prefetch under
attention" is already the engine's behavior.~~ CORRECTION (2026-11-10,
review): 0.95 ms = the submit-rate artifact (the async depth-8 queue); the
device per layer = ~1.2-1.3 ms. The cross-layer overlap statement is
UNPROVEN — it rested entirely on 0.95/293. What survives: the itl_p50
growth with context (56.1/56.9/61.0/65.9 ms) and the paired-gather zero
gain. Measured attention exposure in the engine (submit-rate numbers):

| ctx | layers/step | vs ctx 16 |
|---|---|---|
| 16 | 0.95 | — |
| 1 500 | 0.98 | +0.03 |
| 4 000 | 1.06 | +0.11 |
| 14 000 | 1.17 | +0.22 |

itl_p50: 56.1 / 56.9 / 61.0 / 65.9 ms. ~~Even at 14k the layer stays
~DMA-saturated (295 MB / 1.17 ms = 252 GB/s)~~ (CORRECTION 2026-11-10:
1.17 ms was the submit rate; the device rate at 14k is not established —
with 278.8 + gather MB over ~1.2 ms the layer sits below the 264 ceiling).
The growth = the attention kernel's per-group compute (~45 µs per 128-page group: nc_transposes, score
and PV matmuls, quant/copy traffic), which scales with G.

## Fusion candidates: built, measured, verdicts

| candidate | status | verdict |
|---|---|---|
| (a) merge 4 custom calls | premise wrong: 2 calls exist | moot |
| (b) fuse norm+qkv+rope+KV-write into the attention kernel | not built | the all-reduce after o_proj splits the chain (no collectives in NKI kernels); upside ≤ ~0.1 ms/layer (the XLA small-op trim) = ~4 ms/token, the AR + megakernel risk high |
| (c) fuse attn-out → o_proj | not built | o_proj = one 13 MB fp8 matmul, already minimal |
| (d) MLP weight prefetch under attention | the overlap statement is unproven (the 293 GB/s figure was a submit-rate artifact) | unproven; the fusion-dead verdict rests on the itl growth and the paired-gather zero gain instead |
| (e') paired K/V gather (new kernel `decode_attention_fp8_2p_paired` + cache layout [pages, kvh, 2, 16, D], one 4 KB row per (page, head), V resident for pass 2) | built + compiled + measured | **zero gain**: 1.42 vs 1.42 ms (P=512), 1.55 vs 1.55 (P=1024). The exposure is per-group compute, not gather transport. Requires the prefill buckets to share the layout (the mismatch diverges generation: 12 vs 32 tokens), so it is serving-invasive for no measured win — killed |
| context/bucket choice | measured | already optimal (the engine picks the cheapest covering bucket) |

## Verdict

Attention fusion in the Devstral decode layer is a **dead end at the serving
contexts**: the layer is bound by the fp8 weight DMA (~232 GB/s over the
1.2 ms device time; the earlier "saturated at 293 GB/s" = a submit-rate
artifact
by the engine's cross-layer pipelining), and the attention's remaining
exposure only appears at 4k+ context (≤ 0.22 ms/layer, ~9 ms/token at 14k)
where it is per-group kernel compute, not fusion-addressable without deep NKI
pipeline work whose upside is ≤ ~7% at 14k only.

The real levers, in order: ~~**the head (14.3 ms/token = 25% at every
context, bf16 lm_head 671 MB/rank at only ~47 GB/s effective)**~~
(CORRECTION 2026-11-10, brief 20: the 14.3 ms head figure = mostly the
layer drain — retracted), spec decoding (already on by default: 44.5
ms/token = 22.5 t/s measured), batching (68 µs/row marginal), and
weight-byte reduction (already fp8).

## What is validated vs not

- Validated: the layer profile split (bench + engine + neuron-profile agree),
  the DMA timeline, the paired-gather no-gain result, the layout-consistency
  requirement (prefill and decode must share the KV layout).
- Not validated: any fusion shipping to the engine (none qualifies), the
  paired kernel's numerics (diverges under the mixed-layout test; killed
  before a full-consistency build since the speed result was zero anyway).
