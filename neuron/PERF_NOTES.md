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
contexts (RoPE positions > 0, multi-token KV) and end-to-end token quality are NOT measured yet.
