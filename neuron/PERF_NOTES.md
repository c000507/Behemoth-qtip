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

**TP=2 (both NeuronCores), FP8 x FP8:** 0.99 ms / layer (126 MB per rank), all-reduce after o_proj and down_proj.
**Prefill, TP=2, FP8:** 128 tok 1.24 ms, 512 tok 2.78 ms, 2048 tok 11.76 ms per layer (attention materializes T x T).

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
