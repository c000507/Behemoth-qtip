# Inference attention audit — 2026-10-29

Scope: paragraph repetition on the latest mixed QTIP/FP8-K/V checkpoint, starting
at commit `37358ac`. No definite attention implementation defect was identified
by source inspection. This is not a GPU numerical acceptance result: this
workspace has neither a GPU device nor the full model checkpoint.

## Inspection findings

- The runtime accepts 96 query heads / 8 KV heads, or the tensor-parallel
  48 / 4 layout. The corresponding GQA ratio is 12. General attention maps
  `query_head / 12`; the optimized kernels assign heads `kv_head * 12 + wave`.
- Mixed K/V projections use original-space output rows 0–511 on rank 0 and
  512–1023 on rank 1. These correspond to four 128-dimensional KV heads per
  rank. Q's transformed output is gathered before selecting each rank's 48 heads.
- K/V projection scales are applied once before BF16 rounding. K receives RoPE
  before cache insertion; V does not. Cache append and attention reads both use
  `[physical_page][local_kv_head][token_within_16_token_page][dimension]`.
- Causal masks use absolute query positions, including chunked prefill. Split
  attention merges partial numerators and denominators after rescaling by each
  partition's softmax maximum. Empty partitions contribute zero.
- In overlapped prefill, attention waits for the preceding micro-batch's cache
  append event before reading its keys.

The runtime still quantizes the KV cache to FP8. By default it also uses FP8
queries and tile probabilities. Preserving K/V **weights** in FP8 does not change
these attention precisions. `BEHEMOTH_ATTN_QP=bf16` keeps queries/probabilities in
BF16, with the same FP8 cache. RoPE uses FP32 trigonometry and rounds rotated
vectors to BF16; this also differs from source paths that round cosine/sine
tables before multiplication. These are numerical differences to measure,
not demonstrated causes of repetition.

## Added acceptance checks

Three CTest cases run the same checks in separate processes so the environment
selects the default native kernels, the general legacy kernel, and prefill v1:

```sh
cmake --build runtime/build-rocm -j8
ctest --test-dir runtime/build-rocm -R '^attention_audit_' --output-on-failure -j1
```

Coverage includes:

- FP8 K/V projections at the production shape, both tensor-parallel output
  shards, and GEMV/WMMA execution, using sparse selector weights with an exact
  analytical answer; existing random projection and Q-only epilogue checks.
- Independent known-answer GQA head/request routing through shuffled cache
  pages, for both 48/4 and 96/8 head layouts.
- Full and partial query tiles spanning 16-, 32-, and 64-token boundaries,
  unequal request lengths, and masked cache tails.
- The same queries evaluated as prefill and individual decode calls.
- Changes to future K/V must leave preceding outputs bitwise unchanged.
- FP8 and BF16 Q/P modes, unsplit and split/merge attention, output guards,
  cache append, and the existing long-context checks through 33k tokens.

The scalar reference checks outputs and softmax statistics. Analytical routing
and causal-isolation checks supplement that reference so a shared reference
indexing mistake cannot explain all passing results.

Local results: runtime CPU `core_correctness` passed; the inference attention
HIP source compiled for gfx1201; the expanded GPU test source passed host syntax
checks. The GPU cases above have **not** run locally.

## Diagnose the repeating prompt

Use the same checkpoint, prompt, seed, sampling settings, and output limit.
Restart the server for each comparison; these environment settings are read
once per process.

| Run | Environment before the usual server command | What changes |
|---|---|---|
| Baseline | `BEHEMOTH_ATTN_LEGACY=0 BEHEMOTH_ATTN_PREFILL_V1=0 BEHEMOTH_ATTN_QP=fp8` | Current kernels/precision |
| General kernel | `BEHEMOTH_ATTN_LEGACY=1 BEHEMOTH_ATTN_QP=fp8` | Attention implementation |
| Higher Q/P precision | `BEHEMOTH_ATTN_LEGACY=0 BEHEMOTH_ATTN_PREFILL_V1=0 BEHEMOTH_ATTN_QP=bf16` | Query/probability precision |

A consistent quality difference warrants comparing teacher-forced logits or
held-out loss, since small numerical differences can change sampled generation.
Passing synthetic tests alone cannot establish that the full model's repetition
is unrelated to inference. For the model-level cache/prefill paths, also run:

```sh
runtime/build-rocm/behemoth_model_tests MODEL_DIRECTORY prefill
runtime/build-rocm/behemoth_model_tests MODEL_DIRECTORY prefix
```

Those existing tests compare full/chunked/overlapped prefill and restored versus
cold prefix-cache execution using the actual checkpoint.
