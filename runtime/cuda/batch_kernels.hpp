#pragma once
#include "kernels.hpp"
namespace behemoth::cuda_runtime {
// Token-major activations; split-major, token-major FP32 products.
void qtip_multiply_batch(QtipMatrix, const B16 *, float *, int stride, int splits, int tokens, Stream,
                         int implementation = -1);
void qtip_decode_matrix(QtipMatrix, B16 *, Stream);
void bf16_multiply_batch(const B16 *weight, const B16 *input, float *out, int in, int rows, int tokens,
                         Stream);
void fp8_multiply_batch(const uint8_t *weight, const float *scale, const B16 *input, B16 *out, int in,
                        int rows, int tokens, Stream, float *workspace = nullptr);
void embedding_batch(const B16 *, B16 *, const int *tokens, int rows, int hidden, Stream);
void norm_batch(const B16 *, const float *, B16 *, int rows, int hidden, float epsilon, Stream,
                bool resident = true);
void rope_batch(B16 *, B16 *, const int *positions, int rows, int sequence, int qheads, int kvheads, int dim,
                float theta, Stream, int query_stride = 0);
void cache_batch(const B16 *, const B16 *, B16 *, B16 *, const int *positions, int batch, int sequence,
                 int context, int kvwidth, Stream);
void gqa_batch(const B16 *, const B16 *, const B16 *, B16 *, float *, const int *positions, int batch,
               int context, int max_tokens, int qheads, int kvheads, int dim, Stream, int query_stride = 0,
               float *value_workspace = nullptr, bool register_mma = true);
void select_rows(const B16 *, B16 *, const int *, int rows, int width, Stream);
void reduce_products(float *, size_t elements, int splits, Stream);
void scale_to_b16(const float *, const float *, B16 *, int rows, int width, Stream);
void decode_fp8_matrix(const uint8_t *, B16 *, size_t elements, Stream);
} // namespace behemoth::cuda_runtime
