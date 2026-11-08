# HIP → tile-IR → NEFF codegen design

Scope per docs/rocm-to-neuron.md + neuron/TRANSLATION.md: translate the tile
math (loop nests over tiles), not the CUDA thread model.

## Target execution model (from RE, docs/neff-isa*.md)

- Tensors live in HBM; engines see SB (state buffer) tiles via DMA.
- DMA movement is declarative JSON (def.json dma_queue + <ENGINE>.json dma
  descriptors: from/to var, offsets, strided sizes/steps). The runtime compiles
  descriptors; we do NOT emit DMA machine code for v1.
- PE (tensor engine) executes 64B-line instructions: LDWEIGHTS (op 1) loads a
  weight tile into PE, MATMUL (op 2) = 128x128 tile FMA into PSUM with inline
  events block (wait_mode/wait_idx @4/5, update_mode/update_idx @6/7,
  semaphore_value @8). Both fully encodable (tools/neff_isa.py, byte-identical
  round-trip on 6402+6402 instructions in ffn1_full).
- Activation engine: ACTIVATE (op 33) runs PWL-LUT function sets (exp/sigmoid/
  silu/tanh/gelu...); function sets + bkt/ctrl LUT microcode ship with the
  runtime or embed in tar.
- DVE (vector engine) does stream permutes/reduces/elementwise (ops 65/66/67/
  107/114...). v2 field layouts NOT yet decoded (missing from accessor set;
  sources: hwdecode tables, neuron-explorer). v1 avoids DVE where possible:
  use activation-engine function sets for nonlinearity, SP for scalar glue.
- SP (scalar engine) programs are built by libnrt add_* builders (95 fns, see
  docs/neff-isa-sp.md) - control flow, semaphore ops, DMA triggers.
- Sequencing = semaphores: producer instrs set UPDATE_MODE=*_COMPLETE,
  consumers WAIT_MODE=WAIT_FOR_SEM_* (see docs/neff-isa.md enums). def.json
  dma_queue.semaphore_set pre-allocates semaphore ids.

## Tile-IR (tools/tile_ir.py)

Graph = list of ops over named tensors with static shapes. Ops:

    dma(dst, src, from_off, sizes, steps, to_off, to_sizes, to_steps, sem)
    ldweights(tile_var, sem)
    matmul(dst, src, rows=128, cols=128, in_dtype, events, sem)
    activate(dst, src, func_set, func_id, channels)
    sem_wait(sem_id, mode, value) / sem_update(sem_id, mode, value)
    barrier()

Lowering rules (v1):
- One sb graph per subgraph (sg00), PE stream = [prologue 64B op177] + per-tile
  [LDWEIGHTS, MATMUL] pairs + EVENT_SEMAPHORE glue, exactly the pattern ffn
  graphs use (LDWEIGHTS+MATMUL pairs, SEM_INC_COMPLETE every other matmul).
- Weight/activation tiles pre-staged by DMA JSON from HBM vars (type input).
- Outputs: PSEUDO_TENSOR_STORE epilogue (op 204) as in reference graphs.
- def.json vars: SB (STATE_BUF_PARTITION_SIZE from ISA consts), inputs/outputs
  with sizes, DGETable0, queues qSPSpillReload0/qPoolDynamic as in samples.

## Lift from HIP (runtime/src/*.hip subset)

Recognized kernel classes (ops.hip/projection.hip):
1. elementwise over n (`cast`, `silu`, `residual`, `residual_scaled`) →
   activation-engine function sets (silu_and_others etc.) or DVE arith.
2. row-reduce + scale (`norm_kernel` = RMSNorm: sum(x^2) → rsqrt → x*f*w) →
   v1: decompose into activation(sq)+DVE reduce+SP scale; long-term DVE native.
3. quantize variants (bf16→fp8 per-channel scale) → activation cast + scale.
4. tiling GEMM (`projection`, `qtip` dequant-GEMM) → LDWEIGHTS/MATMUL tiles
   (the dequant path can fold the trellis decode into weight pre-staging DMA
   + activation LUT, or run on GPSIMD Xtensa).
5. attention (online softmax, paged KV) → multi-tile orchestration; phase 2.

The lifter parses the restricted subset (row-block loop nests with
__shared__ reductions, elementwise maps); anything else is rejected with a
location, per TRANSLATION.md's conclusion that attention/scheduling stay
hand-written for now.

## Validation

Round-trip harness on trn1 (tools in /tmp/nrtd on box, source at
s3://912935854432-neuron-build/tools/*.b64): neff_info (nrt_load + tensor
table), neff_io2 (single-rank run, dir/in0.. → dir/out0..). Numeric oracle =
repo's C++/HIP reference kernels.
