# ROCm-to-Neuron feasibility

## Finding

LLVM/MLIR could provide infrastructure for a restricted translator, but there is no documented `HIP/AMDGPU LLVM IR -> NEFF` input path in the Neuron SDK. The supported graph path is XLA HLO -> `neuronx-cc` -> NEFF; custom tile kernels can be expressed in NKI and included in a compiled graph. NEFF is an executable container for a particular Neuron target, not a flavor of LLVM IR. NKI itself uses MLIR, but that does not imply a stable public interface that accepts arbitrary LLVM IR or AMDGPU intrinsics.

Sources: [NEFF generation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/compiler/neuronx-cc/how-to-generate-neff.html), [NKI compiler](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-compiler.html), [NKI and MLIR](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/get-started/about/index.html).

## Why direct lowering is difficult here

* `runtime/src/qtip.hip` uses Wave32 lane layouts, AMDGCN builtins, inline assembly, LDS bank replication, barriers and system-scope atomics. These are hardware-specific implementation choices; translating each instruction would not recover an efficient Neuron tile program. Its peer synchronization also needs a different runtime/collective design.
* `runtime/src/attention.hip` uses rocWMMA fragments and GPU-specific tiling. Neuron's matrix engine has its own tile, memory and operand layout constraints.
* `runtime/src/ops.hip` has simpler elementwise and reduction operations that are plausible candidates for a restricted translator, provided it understands barriers, reduction semantics, bounds, and BF16 rounding.
* HIP host allocations, streams, launch configuration, hipBLAS and the model server cannot be represented just by translating kernel LLVM IR; execution, memory management, collectives and serving need a Neuron backend.

For QTIP the model format is portable in principle, but the kernel must preserve the exact tail-biting trellis windows, LUT indexing and sign, and weight layout. An unpacked dense implementation is useful as a correctness baseline but loses the compressed-weight bandwidth benefit during inference.

## Recommended route

1. Implement a portable semantic reference for a QTIP projection using the existing packed checkpoint layout. Compare its decoded values against the existing CPU decoder and its outputs against ROCm, including tile and bitstream boundaries. Preserve the existing scale and transform steps, not just the core matrix product.
2. Build an NKI QTIP decode-and-project prototype: bring packed tiles and the LUT into SBUF, decode via bit operations and indexed gather where supported, then multiply in tiles. Evaluate fused decode with matmul against pre-decoding to BF16. The indexed gather API has hardware layout restrictions, so feasibility and speed must be measured on the selected chip. [NKI local gather](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/nki/api/generated/nki.isa.local_gather.html).
3. Use the Neuron graph/compiler or an established Neuron model implementation for attention, normalization, FP8 K/V projections and model execution; add custom NKI operations where needed. Verify format-specific FP8 scaling and target dtype support rather than assuming the existing bytes are native Neuron FP8.
4. Only then build an MLIR-based source translator for a deliberately limited HIP subset. Start with cast/elementwise and RMSNorm; recognize whole reduction/matmul patterns before emitting HLO or generated NKI. Fail explicitly on inline AMD assembly, unsupported intrinsics, barriers or atomics. LLVM IR can be a diagnostic/input experiment, but optimized AMDGPU IR may have already lost the high-level structure needed for tiling. Generating supported NKI source is a more stable boundary than relying on internal NKI IR.
5. Compile for a named Neuron chip and run on hardware; compare per-operation outputs, attention/GQA and KV-cache behavior, layer logits and end-to-end generation, then profile decode and prefill at the desired context and batch sizes. Compilation and CPU simulation do not establish device performance or full hardware correctness. [NKI CPU simulator](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html).

## Open decisions and validation

Choose a target instance and pin a Neuron SDK version first: NKI feature and chip support are version-dependent, and the current [NKI FAQ](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/nki_faq.html) lists Trn2/Trn3 for its current release. Do not assume current NKI features run on Inf2 or Trn1. Determine whether Neuron supports the precise integer operations and gather patterns needed for QTIP efficiently, whether a tiled unpack+matmul beats an HBM BF16 baseline, and how quantized weights, KV cache, sampling and tensor parallelism integrate with the chosen serving stack.

This investigation did not compile a NEFF or run Neuron code: `neuronx-cc` and a Neuron device were not present in this workspace.
