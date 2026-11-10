# neff-opt: NEFF DMA post-processor — geometry study + tool

STATUS (2026-11-10, review): PROTOTYPE / NON-DELIVERABLE — the tool is
compiler-version-dependent BY DESIGN (it rewrites descriptors that a
neuronx-cc release emitted) and is paused; the geometry study below is
the durable part.

Purpose: rewrite the DMA descriptor geometry of neuronx-cc-compiled
NEFFs so weight streaming approaches the pure-DMA ceiling. neuronx-cc
stays in the loop (it compiles the input graphs); this tool only
rewrites descriptors in place. C++20, tools/neff_opt/.

## Measured rates (trn1, 200-iter, neurond)

| graph | geometry | GB/s (marginal) |
|---|---|---|
| compiler elementwise mul (x*1.0, n=1M) | ONE desc, all engines idle | **395** (CORRECTION 2026-11-10: read+write of an HBM->HBM copy = 198 GB/s payload) |
| compiler GEMV M1 K=N=12288 bf16 | 66-desc circular ring | 232 |
| compiler GEMV M1/4/8 K=N=4096/12288 (bf16) | 66-desc circular ring | 152-243 |
| our serial 2-desc DMA-only probe | in->out chained | 224-266 |
| our parallel 2-in probe (compiler and-oracle) | 2 in + y-out | 240-265 |

## The 395 GB/s geometry (decoded from bytes)

The compiler's elementwise `x * 1.0` container (n = 1M f32):
- Pool0.bin = 256 B total: prologue [177][204] + **ONE op212 desc** +
  op164 tail (waits sem 8 GE 16).
- DVE0/Activation0/SP0/PE0 = 128 B idle prologues each. **The mul was
  folded away** — x*1.0 = a pure DMA copy; the desc reads the HBM x var
  and writes the HBM y var directly (marker 0x0a0a = f32 pair; DGE
  routes both vars).
- The desc's size field (word rel36/rel27 pair) = n*4B / 64 KiB =
  **packet count at 64 KiB per packet** (n=131072 -> 8; n=1048576 ->
  64; DMA_MAX_TRANSFER_SIZE_PER_DESC_KIB = 64). One desc = a continuous
  multi-packet stream: no per-desc semaphore round-trips anywhere.
- Marginal rate between n=131072 and n=1048576 = 7.34 MB / 18.6 us =
  **395 GB/s aggregate** (read+write concurrent on the HBM bus).

## The GEMV weight-stream geometry (decoded from bytes)

gemv_m1_k4096_n4096_bf16 (w = 33.5 MB bf16 streamed per execute):
- Pool0.bin = 6528 B = 198 slots = **66 op212 descs interleaved 2:1
  with 33 op160 event gates** (+ prologue + tail).
- Every engine stream (PE/DVE/ACT/SP) = idle prologue only — the
  graph = M=1 GEMV: the PE work happens inside... (the matmul engine
  program is in PE0; the pool descs stage w).
- Each desc: sem = 8 + (desc_idx mod 16) — **a 16-deep semaphore
  ring**; src = HBM w var with an **auto-advancing address register**
  (marker 0xeec0-family: the offset nibble steps ee->fe->de->ce->be->ae
  per desc), dst = SB w-region (0x2000 family); size fields 0x80/0x0004
  pairs; dtype marker 0x0606 (bf16).
- The desc list = CIRCULAR: 66 descs x ~508 KB ≈ 33.5 MB — the list
  loops ~8x per execute with the hardware auto-advancing the HBM
  offset. The first descs carry no wait; later groups are gated by
  op160 event instrs (2 descs per gate).
- Rate = 222-244 GB/s marginal = **~60-65% of the pure-DMA ceiling**
  (the "ceiling" here = the 395 copy figure = 198 GB/s payload per
  direction, CORRECTION 2026-11-10 — compare against the one-direction
  read ceiling 210-228 / multi-stream 260-265 instead).

## Conclusion of the diff (what the post-processor can and cannot do)

1. The GEMV w-stream descriptors are ALREADY a continuous
   auto-advancing ring — not a naive small-desc serial chain. The
   222-244 vs 395 gap is therefore NOT a descriptor-chaining artifact;
   the ring is paced by its consumer (the PE weight-load loop) and/or
   by the 16-sem ring depth, neither of which a descriptor rewrite
   touches.
2. Rewriting opportunities that remain (v1 scope, conservative):
   - merge/extend where descs ARE plainly serial small slices of one
     contiguous HBM->SB run with no consumer pacing between them;
   - nothing else without decoding the consumer side.
3. Opaque so far: exact packet-split behavior inside one desc
   (DMA_MAX_DESC_COUNT_PER_PACKET = 64 x 64 KiB), the op160 gate
   semantics, the SB region map of the GEMV ring, whether the 66-desc
   list is re-armed by the runtime or free-loops.

## neffopt v1 (tools/neff_opt/)

`neffopt in.neff out.neff [--report FILE]`:
- unpacks the NEFF (1024 B header + gzip/tar), locates sg00/Pool0.bin,
- parses every op212 descriptor + op160 gate into a typed model,
- classifies the pool: `mul-form` (single continuous desc), `ring`
  (circular auto-advancing w-stream), `serial-chain`, `unknown`,
- DEFAULT = pass-through unchanged (bit-identical output) with a
  per-graph report; rewrites are gated behind flags and only apply to
  patterns whose consumer wiring is decoded,
- `--report` writes the geometry + a byte-diff summary (which bytes
  changed and why).

Safety rules (brief6 step 3): anything with collectives/cc_streams,
multi-output heads the tool cannot model, or desc forms outside the
decoded set = pass-through + reason in the report.

## FP8 (brief6 step 4)

Pending on-device probe: neuronx-cc 2.26 rejected the first jax
fp8-matmul lower (gen produced no o.neff; error capture pending).
Findings will be appended here.
