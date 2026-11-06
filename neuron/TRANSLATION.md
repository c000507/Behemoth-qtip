# Notes on an automated HIP -> Neuron translation path

Assessment based on porting the QTIP ROCm/HIP runtime (`runtime/src/*.hip`,
`runtime/cuda/*.cu`) to hand-written NKI 0.7 kernels in `neuron/`. This is a
feasibility note, not a working translator.

## What the HIP source looks like

The ROCm kernels are HIP C++ (`runtime/src/kernels.hip`, `qtip.hip`,
`qtip_dense.hip`, `fp8_linear.hip`, `ops.hip`, `attention.hip`,
`kv_transfer.hip`, `transport.hip`) plus CUDA-targeted C++ under
`runtime/cuda/`. They are ordinary GPU kernels: grid/block launches, shared
memory, warp shuffles, tensor-core WMMA fragments, async copies. Host code
drives them through HIP streams and device buffers.

## What Neuron actually accepts

There is no LLVM-level path from arbitrary GPU machine code to Neuron:

- Trainium executes NEFF containers produced by `neuronx-cc`. The supported
  front ends are XLA/HLO, PyTorch(neuronx)/TorchEager and NKI. NKI lowers
  Python (`nki.language`) to NKI-IR/MLIR and then to NEFF.
- An "HIP -> LLVM -> Neuron" route would have to lower to the same MLIR
  dialects neuronx-cc consumes (the "penguin" IR used by the internal XLA
  path is not a documented or stable input). In practice the only sanctioned
  compiler entry points are the ones above.
- The device model is also different enough that semantics, not just code
  format, must change: 2 NeuronCores per process visible as partitioned
  engines (GEMM engine, vector engine, scalar engine, DMA engines with
  explicit gather/scatter and on-chip SBUF/PSUM staging). Warp shuffles,
  shared-memory barriers and WMMA fragments have no 1:1 equivalent; the
  closest NKI primitives are `nl.load` DMA tiles, partition-local compute,
  PSUM accumulation and the tensor engine via `nl.matmul`-style ops.

## What a realistic translator would target

Translating the *math* (loop nests over tiles) rather than the *thread
model* is the only plausible scope:

1. Parseable subset: HIP kernels that are "for each output tile: load tiles,
   FMA loops, store" map naturally to NKI `fori_loop` kernels. The QTIP
   trellis decode, FP8/NVFP4 dequant-GEMV, Hadamard transforms and RMSNorm
   in this repo are all in that class; the port in `neuron/` was written
   kernel-by-kernel from the HIP semantics and validated against the C++
   reference, which is effectively what a translator would have to emit.
2. Hard cases: attention (paged KV, online softmax across tiles, FP8 P
   quantization) needs multi-buffer orchestration that NKI expresses
   differently; the all-gather/transport kernels use peer memory that has
   no Neuron analogue within one NKI launch; WMMA tensor-core layouts
   deswizzle differently (the QTIP trellis bit layout in NKI mirrors the
   m16n8k16 A-fragment order the CUDA path uses, which is why the same
   index tables could be reused).
3. Practical alternative: keep the CUDA/HIP runtime as the reference
   implementation and generate NKI from a shared IR the project controls.
   Given the number of kernels (about a dozen) and how much of the win is
   in scheduling (chunking, residency, fusion) rather than instruction
   selection, the ROI of a full translator is low compared to maintaining
   the NKI kernels with the existing oracle-based cross-validation
   (`neuron/tests/test_qtip_kernel.py`, `model_oracle.cpp`).

## Conclusion

No automatic path exists today, and the compile-stack entry points make a
source-level C++ translator the only viable shape: it would parse the HIP
kernels' tile loops and emit NKI Python. For this codebase that translator
would cover the projection/decode kernels (they are already proven in NKI)
and still leave attention, transport and scheduling as hand work. Writing
the NKI kernels directly with C++ oracle validation - what this branch does
- is the shorter path to a correct result.
