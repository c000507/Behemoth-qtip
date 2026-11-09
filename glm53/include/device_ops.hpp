#pragma once
#include "gpu_runtime.hpp"
namespace glm53::gpu {
using B16 = uint16_t;
struct Matrix {
    const void *data;
    const float *scales;
    const uint8_t *block_scales;
    int rows, cols, format; // 0 F32, 1 FP8, 2 NVFP4, 3 F16, 4 BF16
    bool block;
    float global;
};
void gemv(Matrix, const B16 *, void *, bool fp32, Stream);
void decode_matrix(Matrix, B16 *, Stream);
void from_float(const float *, B16 *, int, Stream);
void to_float(const B16 *, float *, int, Stream);
void replicate(const B16 *, B16 *, int hidden, int streams, Stream);
void mean_streams(const B16 *, B16 *, int hidden, int streams, Stream);
void norm(const B16 *, const float *, const float *, B16 *, int n, float eps, bool centered, Stream);
void swiglu(B16 *, const B16 *, int n, float limit, Stream);
void axpy(float *, const B16 *, int n, float alpha, Stream);
// out[i] = sum over ranks r of slots[r * n + i]; fixed summation order, so every rank agrees bitwise.
void sum_slots(const float *slots, int ranks, int n, float *out, Stream);
// MoE combination from the reduced expert outputs [shared, routed 0..experts-1] (each n floats):
// out = bf16(bf16(shared) + sum_e factors[e] * bf16(routed_e)).
void combine_experts(const float *outputs, const float *factors, int experts, int n, B16 *out, Stream);
void float_matvec(const float *, const B16 *, float *, int rows, int cols, Stream);
void hc_pre(const B16 *, const float *, const float *, const float *, B16 *, float *, float *, float *,
            int hidden, int streams, int iters, float eps, float hc_eps, Stream);
void hc_post(const B16 *, const B16 *, const float *, const float *, B16 *, int hidden, int streams, Stream);
void convolution(B16 *, B16 *, B16 *, B16 *, const float *, const float *, const float *, int width,
                 int kernel, Stream);
void kda(const B16 *, const B16 *, const B16 *, const B16 *, const B16 *, const float *, const float *,
         float *, B16 *, int heads, int dim, float lower, Stream);
void gated_norm(B16 *, const B16 *, const float *, int heads, int dim, float eps, Stream);
void pool_key(const B16 *, const B16 *, const float *, B16 *, int pool, int dim, Stream);
void index_scores(const B16 *, const B16 *, const B16 *, float *, int pools, int heads, int dim, Stream);
void select_keys(float *, int *, int *, int pools, int padded, int pool, int topk, int position, Stream);
void absorb_query(const B16 *, const B16 *, float *, int heads, int qdim, int vdim, int rank, Stream);
void mla(const float *, const B16 *, const B16 *, const int *, B16 *, float *, float *, int selected,
         int heads, int qdim, int vdim, int rank, Stream);
void route(float *, const float *, int *, float *, float *, int *, int experts, int topk, int groups,
           int top_groups, bool normalize, float scale, Stream);
} // namespace glm53::gpu
