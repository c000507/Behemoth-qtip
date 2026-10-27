#pragma once
#include "train.hpp"
#include <hip/hip_runtime.h>
namespace qat::gpu {
// half: elements are BF16 (uint16_t) at p, else FP32.
struct View {
    float *p;
    int rows, cols, ld;
    int half;
};
inline View view(Mat m) { return {m.p, m.rows, m.cols, m.ld, m.half ? 1 : 0}; }
void binary(View, View, View, int, bool, hipStream_t);
void cast(View, uint16_t *, bool, hipStream_t);
void rounded(View, hipStream_t);
void scalar_reduce(View, View, View, hipStream_t);
inline constexpr int reduction_rows = 256;
void column_reduce(View, View, View, View, View, hipStream_t);
// partial = column partials of a * b (* scale[r]); d[c] += total * factor[c].
void column_reduce_factor(View a, View b, View scale, View factor, View partial, View d, hipStream_t);
// dc = column sums of dy * c; gsv += dc * scale[0]; gscale[0] += sum(dc * sv).
void sv_reduce(View dy, View c, View sv, View scale, View partial, View gsv, View gscale, hipStream_t);
void rms(View, View, View, View, float, hipStream_t);
void rms_back(View, View, View, View, View, hipStream_t);
void rope_angles(View, int, float, const int32_t *, hipStream_t);
void rope(View, View, int, int, View, bool, hipStream_t);
void swiglu(View, View, View, hipStream_t);
void swiglu_back(View, View, View, View, View, hipStream_t);
void embedding(View, const uint16_t *, const int32_t *, int, hipStream_t);
void loss(View, const int32_t *, float *, float, bool, hipStream_t);
void adam(float *, const float *, float *, float *, size_t, bool, uint64_t, float, float, hipStream_t);
void flash(View, View, View, View, View, int, int, const int32_t *, hipStream_t);
void flash_back_native(View, View, View, View, View, View, View, View, View, int, int, const int32_t *,
                       const int32_t *, hipStream_t);
void flash_back(View, View, View, View, View, View, View, View, View, View, int, int, const int32_t *,
                const int32_t *, hipStream_t);
} // namespace qat::gpu
namespace qtip_training {
// Row transform of n = 1024 * K (K = 1, 12, 28) wide rows, one block per row:
// c = had(x') or hadT(x') / sqrt(n), x' = (round_in ? bf16(x) : x) * pre * pre_scale[0];
// y = bf16(c * post * post_scale[0]), then y = bf16(residual + y) if residual.
// Every pointer except in is optional; *_half selects BF16 element storage.
struct RowTransform {
    const void *in = nullptr;
    long ld_in = 0;
    int in_half = 0, round_in = 0, transpose = 0;
    const float *pre = nullptr, *pre_scale = nullptr;
    void *c = nullptr;
    long ldc = 0;
    int c_half = 0;
    const float *post = nullptr, *post_scale = nullptr;
    const void *res = nullptr;
    long ldres = 0;
    int res_half = 0;
    void *y = nullptr;
    long ldy = 0;
    int y_half = 0;
};
bool transform_supported(int n);
void transform(const RowTransform &, int rows, int n, hipStream_t);
// Legacy entry point (FP32 c and/or BF16 y = bf16(c * post)).
void had_rows(const void *, int, long, int, int, int, const float *, int, float *, long, uint16_t *, long,
              const float *, hipStream_t);
// Fused RMSNorm + input transform for n = 1024 or 12288: xn = bf16(x * r * w),
// u = bf16(hadT(su * xn) / sqrt(n)) (contiguous GEMM operand).
void rms_had(const void *x, int x_half, long ld, int rows, int n, const float *w, float eps, const float *su,
             void *xn, int xn_half, long ldxn, float *r, uint16_t *u, long ldu, hipStream_t);
// dp = dxsu * su; dg = dp * up * silu'(gate); du = dp * silu(gate).
void swiglu_bwd(const void *dxsu, int dxsu_half, long ldd, const float *su, const void *gate, int gate_half,
                long ldg, const void *up, int up_half, long ldu, void *dg, void *du, int out_half, long ldo,
                int rows, int cols, hipStream_t);
// RMSNorm backward with dy = dxsu * su (n = 1024 or 12288): dh += r (dyw - xh mean(dyw xh)).
void rms_bwd(const void *x, int x_half, long ldx, const float *r, const float *w, const float *su,
             const void *dxsu, int dxsu_half, long ldd, float *dh, long ldh, int rows, int n, hipStream_t);
} // namespace qtip_training
