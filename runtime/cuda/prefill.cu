#include "prefill.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
namespace behemoth::cuda_runtime {
namespace {
using glm53::gpu::Buffer;
using Product = PrefillOps::Product;
__device__ __forceinline__ float f32(B16 x) { return __uint_as_float(unsigned(x) << 16); }
__device__ __forceinline__ B16 b16(float x) {
    unsigned u = __float_as_uint(x);
    return B16((u + 0x7fffU + ((u >> 16) & 1)) >> 16);
}
__device__ __forceinline__ float sigmoid(float x) { return x >= 0 ? 1 / (1 + expf(-x)) : expf(x) / (1 + expf(x)); }
// the projection output as the stepwise path sees it: rounded to BF16
__device__ __forceinline__ float value(const Product &p, int row, size_t i) {
    if (p.meta)
        return f32(b16(float(static_cast<const int32_t *>(p.z)[i]) * p.wscale * p.meta[2 * row]));
    return f32(b16(static_cast<const float *>(p.z)[i]));
}
// One block per row; reduction tree of glm53::gpu::norm (warp sums, then a sum over the warps).
__global__ void norm_rows(const B16 *x, const float *w, B16 *out, int n, float eps) {
    __shared__ float partial[32];
    x += size_t(blockIdx.x) * n;
    out += size_t(blockIdx.x) * n;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float a = f32(x[i]);
        sum += a * a;
    }
    for (int d = 16; d; d /= 2)
        sum += __shfl_down_sync(0xffffffffU, sum, d);
    if (!lane)
        partial[warp] = sum;
    __syncthreads();
    sum = threadIdx.x < blockDim.x / 32 ? partial[lane] : 0;
    if (!warp)
        for (int d = 16; d; d /= 2)
            sum += __shfl_down_sync(0xffffffffU, sum, d);
    if (!threadIdx.x)
        partial[0] = sum;
    __syncthreads();
    const float inv = rsqrtf(partial[0] / n + eps);
    // every thread has read all source values before any in-place write
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = b16(f32(x[i]) * inv * w[i] + 0.f);
}
// packed 2-bit codes (behemoth::ternary layout, sixteen weights per word) -> int8 {-1, 0, 1}
__global__ void unpack_ternary(const uint32_t *packed, uint4 *out, size_t words) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= words)
        return;
    const uint32_t word = packed[i];
    uint32_t v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t code = (word >> (2 * j)) & 0x03030303u;
        const uint32_t plus = (code >> 1) & 0x01010101u, minus = ~code & ~(code >> 1) & 0x01010101u;
        v[j] = plus | minus * 255u;
    }
    out[i] = make_uint4(v[0], v[1], v[2], v[3]);
}
__global__ void store_product(Product p, B16 *y, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = b16(value(p, int(i / p.width), i));
}
__global__ void add_product(Product p, B16 *x, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = b16(f32(x[i]) + value(p, int(i / p.width), i));
}
__global__ void swiglu_product(Product gate, Product up, B16 *y, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const int row = int(i / gate.width);
        const float a = value(gate, row, i), b = value(up, row, i);
        y[i] = b16(a * sigmoid(a) * b);
    }
}
// One thread per rotated pair. grouped != 0: source heads [row][heads][dim] are written as
// [heads / group][row][group][dim]; otherwise the layout is kept.
__global__ void rope_product(Product p, B16 *out, int rows, int position, int heads, int group, int dim,
                             float theta, const float *inv_freq) {
    const int half = dim / 2;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * heads * half)
        return;
    const int d = int(i % half), h = int(i / half % heads), row = int(i / half / heads);
    const float angle = float(position + row) * (inv_freq ? inv_freq[d] : powf(theta, -float(2 * d) / dim));
    float s, c;
    sincosf(angle, &s, &c);
    const size_t src = (size_t(row) * heads + h) * dim + d;
    const float a = value(p, row, src), b = value(p, row, src + half);
    const size_t dst =
        group ? ((size_t(h / group) * rows + row) * group + h % group) * dim + d : src;
    out[dst] = b16(a * c - b * s);
    out[dst + half] = b16(b * c + a * s);
}
// One block per (kv head, query, group member) row of fp32 scores [columns]; the query at cache row
// first + q sees keys 0..first + q. Same arithmetic as the stepwise softmax; probabilities are
// rounded to BF16 and the masked tail is zero.
__global__ void causal_softmax(float *scores, B16 *probabilities, int columns, int queries, int group, int first,
                               float scale) {
    __shared__ float red[256];
    const int tid = threadIdx.x, n = first + int(blockIdx.x / group % queries) + 1;
    float *x = scores + size_t(blockIdx.x) * columns;
    B16 *p = probabilities + size_t(blockIdx.x) * columns;
    float top = -INFINITY;
    for (int i = tid; i < n; i += 256) {
        x[i] *= scale;
        top = fmaxf(top, x[i]);
    }
    red[tid] = top;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (tid < d)
            red[tid] = fmaxf(red[tid], red[tid + d]);
        __syncthreads();
    }
    top = red[0];
    __syncthreads();
    float sum = 0;
    for (int i = tid; i < n; i += 256) {
        const float a = expf(x[i] - top);
        x[i] = a;
        sum += a;
    }
    red[tid] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (tid < d)
            red[tid] += red[tid + d];
        __syncthreads();
    }
    sum = red[0];
    for (int i = tid; i < columns; i += 256)
        p[i] = i < n ? b16(x[i] / sum) : B16(0);
}
// fp32 [kvheads][queries][group][dim] -> bf16 [queries][kvheads * group][dim]
__global__ void attention_output(const float *grouped, B16 *out, int queries, int kvheads, int group, int dim) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(queries) * kvheads * group * dim)
        return;
    const int d = int(i % dim), h = int(i / dim % (kvheads * group)), q = int(i / dim / (kvheads * group));
    out[i] = b16(grouped[((size_t(h / group) * queries + q) * group + h % group) * dim + d]);
}
void launched() { glm53::gpu::checked(cudaGetLastError(), "CUDA prefill kernel launch"); }
void blas(cublasStatus_t status, const char *stage) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(stage) + ": cuBLAS status " + std::to_string(int(status)));
}
unsigned blocks(size_t n) { return unsigned((n + 255) / 256); }
} // namespace
struct PrefillOps::Impl {
    Stream stream;
    cublasHandle_t handle{};
    std::array<Buffer<float>, 2> z; // projection results (fp32 or int32)
    Buffer<int8_t> weights, codes;  // unpacked ternary matrix, quantised activations
    Buffer<float> meta, scores, grouped;
    Buffer<B16> probabilities;
    const B16 *input = nullptr;
    size_t budget = size_t(32) << 20; // fp32 score cells per attention pass
    int strided = -1; // strided-batched bf16 GEMM availability (probed on first use)
    explicit Impl(Stream s) : stream(s) {
        blas(cublasCreate(&handle), "create cuBLAS handle");
        blas(cublasSetStream(handle, s), "set cuBLAS stream");
    }
    ~Impl() {
        (void)cudaStreamSynchronize(stream);
        (void)cublasDestroy(handle);
    }
    // C[i] (m x n, fp32) = op(A[i]) B[i] for `count` equally strided BF16 operand pairs
    void gemm_batch(cublasOperation_t op, int m, int n, int k, const B16 *a, int lda, long long sa, const B16 *b,
                    int ldb, long long sb, float *c, int ldc, long long sc, int count) {
        const float one = 1, zero = 0;
        if (strided) {
            auto status = cublasGemmStridedBatchedEx(handle, op, CUBLAS_OP_N, m, n, k, &one, a, CUDA_R_16BF, lda, sa,
                                                     b, CUDA_R_16BF, ldb, sb, &zero, c, CUDA_R_32F, ldc, sc, count,
                                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (status == CUBLAS_STATUS_SUCCESS) {
                strided = 1;
                return;
            }
            if (strided == 1)
                blas(status, "batched attention GEMM");
            strided = 0;
        }
        for (int i = 0; i < count; ++i)
            blas(cublasGemmEx(handle, op, CUBLAS_OP_N, m, n, k, &one, a + i * sa, CUDA_R_16BF, lda, b + i * sb,
                              CUDA_R_16BF, ldb, &zero, c + i * sc, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "attention GEMM");
    }
};
PrefillOps::PrefillOps(Stream s) : impl(std::make_unique<Impl>(s)) {}
PrefillOps::~PrefillOps() = default;
void PrefillOps::norm(const B16 *x, const float *gain, B16 *y, int rows, int width, float epsilon) {
    norm_rows<<<rows, 256, 0, impl->stream>>>(x, gain, y, width, epsilon);
    launched();
}
void PrefillOps::quantize(const B16 *x, int rows, int in) {
    (void)rows, (void)in;
    impl->input = x;
}
Product PrefillOps::dense(const B16 *weight, const B16 *x, int in, int out, int rows, int slot) {
    auto &z = impl->z.at(size_t(slot));
    z.ensure(size_t(rows) * out, impl->stream);
    const float one = 1, zero = 0;
    blas(cublasGemmEx(impl->handle, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one, weight, CUDA_R_16BF, in, x,
                      CUDA_R_16BF, in, &zero, z.p, CUDA_R_32F, out, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
         "dense prefill GEMM");
    return {z.p, nullptr, 1, out};
}
Product PrefillOps::ternary(const uint8_t *packed, float wscale, int in, int out, int rows, int slot) {
    if (in % 16)
        throw std::invalid_argument("ternary projection needs in % 16 == 0");
    auto &z = impl->z.at(size_t(slot));
    z.ensure(size_t(rows) * out, impl->stream);
    // dp4a multi-token kernel; its raw int32 dot products are the product (meta lives in its scratch)
    impl->codes.ensure(ternary_scratch_bytes(in, rows), impl->stream);
    impl->weights.ensure(size_t(rows) * out * 2, impl->stream);
    ternary_forward(packed, wscale, impl->input, reinterpret_cast<B16 *>(impl->weights.p), in, out,
                    reinterpret_cast<uint8_t *>(impl->codes.p), impl->stream, rows,
                    reinterpret_cast<int32_t *>(z.p));
    float *meta = reinterpret_cast<float *>(impl->codes.p + size_t(rows) * in);
    if (rows == 1) // the single-token kernel quantises internally and leaves no metadata
        ternary_quantize(impl->input, impl->codes.p, meta, in, 1, impl->stream);
    return {z.p, meta, wscale, out};
}
void PrefillOps::store(Product p, B16 *y, int rows) {
    const size_t n = size_t(rows) * p.width;
    store_product<<<blocks(n), 256, 0, impl->stream>>>(p, y, n);
    launched();
}
void PrefillOps::add(Product p, B16 *x, int rows) {
    const size_t n = size_t(rows) * p.width;
    add_product<<<blocks(n), 256, 0, impl->stream>>>(p, x, n);
    launched();
}
void PrefillOps::swiglu(Product gate, Product up, B16 *y, int rows) {
    const size_t n = size_t(rows) * gate.width;
    swiglu_product<<<blocks(n), 256, 0, impl->stream>>>(gate, up, y, n);
    launched();
}
void PrefillOps::rope_queries(Product p, B16 *grouped, int rows, int position, int qheads, int kvheads, int dim,
                              float theta, const float *inv_freq) {
    rope_product<<<blocks(size_t(rows) * qheads * dim / 2), 256, 0, impl->stream>>>(
        p, grouped, rows, position, qheads, qheads / kvheads, dim, theta, inv_freq);
    launched();
}
void PrefillOps::rope_keys(Product p, B16 *keys, int rows, int position, int kvheads, int dim, float theta,
                           const float *inv_freq) {
    rope_product<<<blocks(size_t(rows) * kvheads * dim / 2), 256, 0, impl->stream>>>(p, keys, rows, position,
                                                                                    kvheads, 0, dim, theta, inv_freq);
    launched();
}
void PrefillOps::score_budget(size_t floats) { impl->budget = floats; }
void PrefillOps::attention(const B16 *queries, const B16 *keys, const B16 *values, B16 *out, int position, int rows,
                           int qheads, int kvheads, int dim) {
    const int group = qheads / kvheads, kv = kvheads * dim;
    Stream s = impl->stream;
    // bound the materialised scores
    const int step =
        int(std::clamp<size_t>(impl->budget / (size_t(qheads) * (position + rows)), 1, size_t(rows)));
    for (int a = 0; a < rows; a += step) {
        const int nq = std::min(step, rows - a), columns = position + a + nq, wide = nq * group;
        const size_t cells = size_t(kvheads) * wide * columns;
        impl->scores.ensure(cells, s);
        impl->probabilities.ensure(cells, s);
        impl->grouped.ensure(size_t(kvheads) * wide * dim, s);
        // scores[kh] (columns x wide) = K[kh]^T (columns x dim) Q[kh] (dim x wide)
        impl->gemm_batch(CUBLAS_OP_T, columns, wide, dim, keys, kv, dim, queries + size_t(a) * group * dim, dim,
                         (long long)rows * group * dim, impl->scores, columns, (long long)wide * columns, kvheads);
        causal_softmax<<<kvheads * wide, 256, 0, s>>>(impl->scores, impl->probabilities, columns, nq, group,
                                                     position + a, rsqrtf(float(dim)));
        launched();
        // out[kh] (dim x wide) = V[kh] (dim x columns) P[kh] (columns x wide)
        impl->gemm_batch(CUBLAS_OP_N, dim, wide, columns, values, kv, dim, impl->probabilities, columns,
                         (long long)wide * columns, impl->grouped, dim, (long long)wide * dim, kvheads);
        attention_output<<<blocks(size_t(nq) * qheads * dim), 256, 0, s>>>(impl->grouped,
                                                                           out + size_t(a) * qheads * dim, nq,
                                                                           kvheads, group, dim);
        launched();
    }
}
} // namespace behemoth::cuda_runtime
