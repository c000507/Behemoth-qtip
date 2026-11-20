#pragma once
#include "kernels.hpp"
#include <memory>
namespace behemoth::cuda_runtime {
// Whole-chunk prefill operators for ternary (BitNet b1.58) and dense BF16 llama checkpoints.
// Projections are tensor-core GEMMs over all rows of a chunk (cuBLAS: bf16 x bf16 -> fp32 for dense,
// int8 x int8 -> int32 for ternary weights unpacked on the fly), attention is a causal
// scores-GEMM / masked softmax / values-GEMM per KV head, and every elementwise step (BF16 rounding,
// RoPE, residual, SwiGLU, KV-cache write) is one launch per chunk. All rounding points of the
// stepwise path are kept (projection outputs, probabilities, attention outputs are rounded to BF16
// before their next use); only fp32 summation order differs. Ternary projections are exact integers
// and therefore bit-identical to the stepwise dp4a kernel.
class PrefillOps {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    // An unrounded projection result, [rows][width]: fp32 (meta == nullptr) or int32 ternary dot
    // products with the per-row activation scale in meta[2 * row]. Valid until the slot is reused.
    struct Product {
        const void *z;
        const float *meta;
        float wscale;
        int width;
    };
    explicit PrefillOps(Stream); // the stream's device must be current
    ~PrefillOps();
    // Whether cuBLAS on this device runs the bf16 -> fp32 and int8 -> int32 GEMMs (probed once with
    // tiny operands; BF16 tensor cores need sm_80+). Callers fall back to the stepwise path otherwise.
    bool supported() const;
    // RMSNorm of every row (the arithmetic of glm53::gpu::norm); in place is allowed.
    void norm(const B16 *x, const float *gain, B16 *y, int rows, int width, float epsilon);
    // Per-row int8 absmax quantisation of the input of the following ternary() calls.
    void quantize(const B16 *x, int rows, int in);
    Product dense(const B16 *weight, const B16 *x, int in, int out, int rows, int slot = 0);
    Product ternary(const uint8_t *packed, float wscale, int in, int out, int rows, int slot = 0);
    void store(Product, B16 *y, int rows);                  // y = bf16(product)
    void add(Product, B16 *x, int rows);                    // x = bf16(x + bf16(product))
    void swiglu(Product gate, Product up, B16 *y, int rows); // y = bf16(silu(gate) * up)
    // RoPE at positions position.. (inv_freq != nullptr: llama3 table). Queries are written grouped
    // by KV head, [kvheads][rows][qheads / kvheads][dim]; keys keep the cache layout [rows][kvheads][dim].
    void rope_queries(Product, B16 *grouped, int rows, int position, int qheads, int kvheads, int dim, float theta,
                      const float *inv_freq);
    void rope_keys(Product, B16 *keys, int rows, int position, int kvheads, int dim, float theta,
                   const float *inv_freq);
    // Causal GQA for rows new tokens at cache rows position..position+rows-1 (keys/values already
    // appended): out [rows][qheads][dim].
    // Scores are materialised for at most `floats` fp32 cells per pass (default 32M = 128 MB).
    void score_budget(size_t floats);
    void attention(const B16 *grouped_queries, const B16 *keys, const B16 *values, B16 *out, int position, int rows,
                   int qheads, int kvheads, int dim);
};
} // namespace behemoth::cuda_runtime
