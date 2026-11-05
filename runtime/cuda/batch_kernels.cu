#include "batch_kernels.hpp"
#include <algorithm>
#include <array>
#include <cuda_bf16.h>
#include <mma.h>
namespace behemoth::cuda_runtime {
namespace {
namespace wm = nvcuda::wmma;
__device__ float f32(B16 x) { return __uint_as_float(unsigned(x) << 16); }
__device__ B16 b16(float x) {
    unsigned u = __float_as_uint(x);
    return B16((u + 0x7fffU + ((u >> 16) & 1)) >> 16);
}
__device__ float fp8(uint8_t x) {
    int e = (x >> 3) & 15, m = x & 7;
    float v = e ? __uint_as_float(unsigned(e + 120) << 23 | unsigned(m) << 20) : float(m) / 512;
    return e == 15 && m == 7 ? nanf("") : (x & 128 ? -v : v);
}
__device__ void decode_tile(QtipMatrix w, const uint32_t *lut, int br, int chunk, B16 *a, int ld) {
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    size_t base = (size_t(br) * (w.in / 32) + chunk) * 128;
    for (int quad = warp; quad < 4; quad += blockDim.x / 32) {
        unsigned u = unsigned(w.trellis[base + lane * 4 + quad]) << 16 |
                     w.trellis[base + ((lane + 1) & 31) * 4 + quad];
        int j1 = quad / 2, i1 = quad % 2;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned s = (u >> (16 - 4 * t)) & 65535U;
            unsigned pair = lut[((s * (s + 1)) >> 6) & 1023U];
            int r = i1 * 16 + (t & 1) * 8 + lane / 4;
            int c = j1 * 16 + (t / 2) * 8 + (lane % 4) * 2;
            *reinterpret_cast<uint32_t *>(a + size_t(r) * ld + c) = pair;
        }
    }
}
__global__ void decode_matrix(QtipMatrix w, B16 *out) {
    __shared__ uint32_t lut[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        lut[i] = w.lut[i];
    __syncthreads();
    decode_tile(w, lut, blockIdx.x, blockIdx.y, out + size_t(blockIdx.x) * 32 * w.in + blockIdx.y * 32, w.in);
}
// A is row-major (32 x 32), B column-major (32 x NT). No transposition of
// activations is needed. Every weight is decoded once for all NT token columns.
template <int Format, int NT, int Threads>
__global__ void tensor_product(QtipMatrix w, const B16 *dense, const uint8_t *low, const float *scale,
                               const B16 *x, void *out, int stride, int tokens) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) unsigned char shared[];
    auto *lut = reinterpret_cast<uint32_t *>(shared);
    auto *a = reinterpret_cast<B16 *>(shared + 4096);
    auto *b = a + 1024;
    int br = blockIdx.x, token_base = blockIdx.z * NT;
    constexpr int Columns = NT == 16 ? 1 : 2;
    constexpr int Fragments = NT / (16 * Columns);
    int warp = threadIdx.x / 32, row_tile = warp / Columns, column_tile = warp % Columns;
    wm::fragment<wm::accumulator, 16, 16, 16, float> acc[Fragments];
    for (int n = 0; n < Fragments; ++n)
        wm::fill_fragment(acc[n], 0.f);
    if constexpr (Format == 0)
        for (int i = threadIdx.x; i < 1024; i += Threads)
            lut[i] = w.lut[i];
    __syncthreads();
    int chunks = w.in / 32, first = chunks * blockIdx.y / gridDim.y;
    int last = chunks * (blockIdx.y + 1) / gridDim.y;
    for (int chunk = first; chunk < last; ++chunk) {
        if constexpr (Format == 0)
            decode_tile(w, lut, br, chunk, a, 32);
        else
            for (int i = threadIdx.x; i < 1024; i += Threads) {
                size_t at = size_t(br * 32 + i / 32) * w.in + chunk * 32 + i % 32;
                a[i] = Format == 1 ? dense[at] : b16(fp8(low[at]));
            }
        for (int i = threadIdx.x; i < NT * 32; i += Threads) {
            int token = token_base + i / 32;
            b[i] = token < tokens ? x[size_t(token) * w.in + chunk * 32 + i % 32] : B16(0);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> af;
            wm::load_matrix_sync(af, reinterpret_cast<const __nv_bfloat16 *>(a + row_tile * 16 * 32 + kk),
                                 32);
#pragma unroll
            for (int n = 0; n < Fragments; ++n) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major> bf;
                wm::load_matrix_sync(bf,
                                     reinterpret_cast<const __nv_bfloat16 *>(
                                         b + (column_tile * 16 + n * Columns * 16) * 32 + kk),
                                     32);
                wm::mma_sync(acc[n], af, bf, acc[n]);
            }
        }
        __syncthreads();
    }
    auto *values = reinterpret_cast<float *>(a);
    for (int n = 0; n < Fragments; ++n)
        wm::store_matrix_sync(values + (column_tile * 16 + n * Columns * 16) * 32 + row_tile * 16, acc[n], 32,
                              wm::mem_col_major);
    __syncthreads();
    for (int i = threadIdx.x; i < NT * 32; i += Threads) {
        int token = token_base + i / 32, row = br * 32 + i % 32;
        if (token >= tokens)
            continue;
        if constexpr (Format == 2)
            static_cast<B16 *>(out)[size_t(token) * stride + row] = b16(values[i] * scale[row]);
        else
            static_cast<float *>(out)[(size_t(blockIdx.y) * tokens + token) * stride + row] = values[i];
    }
#endif
}
bool native_bf16() {
    int device = 0;
    glm53::gpu::checked(cudaGetDevice(&device), "get matmul device");
    static thread_local std::array<int, 64> major = [] {
        std::array<int, 64> a{};
        a.fill(-1);
        return a;
    }();
    if (device >= 64)
        return false;
    if (major[device] < 0) {
        cudaDeviceProp properties{};
        glm53::gpu::checked(cudaGetDeviceProperties(&properties, device), "query tensor core support");
        major[device] = properties.major;
    }
    return major[device] >= 8;
}
// QTIP's four decoded pairs already have the register ordering required by
// mma.m16n8k16. Keep them in registers instead of staging a 32x32 BF16 tile.
__global__ void trellis_mma8(QtipMatrix w, const B16 *x, float *out, int stride, int tokens) {
#if __CUDA_ARCH__ >= 800
    __shared__ uint32_t lut[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        lut[i] = w.lut[i];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    int row = blockIdx.x * (blockDim.x / 32) * 16 + warp * 16;
    if (row >= w.out)
        return;
    int chunks = w.in / 32;
    int first = chunks * blockIdx.y / gridDim.y;
    int last = chunks * (blockIdx.y + 1) / gridDim.y;
    int token_base = blockIdx.z * 8;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int chunk = first; chunk < last; ++chunk) {
        size_t base = (size_t(row / 32) * chunks + chunk) * 128;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            unsigned word = w.trellis[base + lane * 4 + half * 2 + (warp & 1)];
            unsigned next = __shfl_sync(0xffffffffU, word, (lane + 1) & 31);
            unsigned u = word << 16 | next;
            uint32_t a[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                unsigned state = (u >> (16 - 4 * t)) & 65535U;
                a[t] = lut[((state * (state + 1)) >> 6) & 1023U];
            }
            int token = token_base + lane / 4;
            size_t at = size_t(token) * w.in + chunk * 32 + half * 16 + (lane % 4) * 2;
            uint32_t b0 = 0, b1 = 0;
            if (token < tokens) {
                b0 = *reinterpret_cast<const uint32_t *>(x + at);
                b1 = *reinterpret_cast<const uint32_t *>(x + at + 8);
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
        }
    }
    int token = token_base + (lane % 4) * 2;
    size_t at = (size_t(blockIdx.y) * tokens + token) * stride + row + lane / 4;
    if (token < tokens) {
        out[at] = c0;
        out[at + 8] = c2;
    }
    if (token + 1 < tokens) {
        out[at + stride] = c1;
        out[at + stride + 8] = c3;
    }
#endif
}
template <int Format>
void launch(QtipMatrix w, const B16 *dense, const uint8_t *low, const float *scale, const B16 *x, void *out,
            int stride, int splits, int tokens, Stream stream) {
    if (w.in % 32 || w.out % 32 || tokens <= 0 || splits <= 0 || splits > w.in / 32 || stride < w.out)
        throw std::runtime_error("Invalid CUDA tensor matmul geometry");
    if (tokens <= 16) {
        constexpr int shared = 4096 + 2048 + 1024;
        tensor_product<Format, 16, 64><<<dim3(w.out / 32, splits, (tokens + 15) / 16), 64, shared, stream>>>(
            w, dense, low, scale, x, out, stride, tokens);
    } else {
        constexpr int shared = 4096 + 32 * 64 * 4;
        tensor_product<Format, 64, 128>
            <<<dim3(w.out / 32, splits, (tokens + 63) / 64), 128, shared, stream>>>(w, dense, low, scale, x,
                                                                                    out, stride, tokens);
    }
    glm53::gpu::checked(cudaGetLastError(), "CUDA tensor matmul launch");
}
__device__ float warp_sum(float x) {
    for (int d = 16; d; d /= 2)
        x += __shfl_down_sync(0xffffffffU, x, d);
    return x;
}
__global__ void embed(const B16 *weights, B16 *out, const int *ids, int rows, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width)
        out[i] = weights[size_t(ids[i / width]) * width + i % width];
}
__global__ void rms(const B16 *x, const float *weight, B16 *out, int width, float epsilon) {
    __shared__ float red[8];
    int row = blockIdx.x, tid = threadIdx.x;
    x += size_t(row) * width;
    out += size_t(row) * width;
    float sum = 0;
    for (int i = tid; i < width; i += 256) {
        float v = f32(x[i]);
        sum += v * v;
    }
    sum = warp_sum(sum);
    if (!(tid & 31))
        red[tid / 32] = sum;
    __syncthreads();
    sum = tid < 8 ? red[tid] : 0;
    if (tid < 32)
        sum = warp_sum(sum);
    if (!tid)
        red[0] = sum;
    __syncthreads();
    float inv = rsqrtf(red[0] / width + epsilon);
    for (int i = tid; i < width; i += 256)
        out[i] = b16(f32(x[i]) * inv * weight[i]);
}
__global__ void rotate(B16 *q, B16 *k, const int *positions, int sequence, int qh, int kh, int dim,
                       float theta, int qstride) {
    int h = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    if (d >= dim / 2)
        return;
    B16 *x = h < qh ? q + size_t(row) * qstride + h * dim : k + (size_t(row) * kh + h - qh) * dim;
    int position = positions[row / sequence] + row % sequence;
    float angle = position * powf(theta, -float(2 * d) / dim), s, c;
    sincosf(angle, &s, &c);
    float a = f32(x[d]), b = f32(x[d + dim / 2]);
    x[d] = b16(a * c - b * s);
    x[d + dim / 2] = b16(b * c + a * s);
}
__global__ void cache(const B16 *k, const B16 *v, B16 *keys, B16 *values, const int *positions, int batch,
                      int sequence, int context, int width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(batch) * sequence * width) {
        int row = i / width, b = row / sequence;
        int at = positions[b] + row % sequence;
        size_t dst = (size_t(b) * context + at) * width + i % width;
        keys[dst] = k[i];
        values[dst] = v[i];
    }
}
__global__ void scores(const B16 *q, const B16 *k, float *out, const int *positions, int context, int qh,
                       int kh, int dim, int qstride) {
    int h = blockIdx.y % qh, b = blockIdx.y / qh, t = blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x & 31, head = h / (qh / kh);
    float sum = 0;
    if (t <= positions[b])
        for (int d = lane; d < dim; d += 32)
            sum += f32(q[size_t(b) * qstride + h * dim + d]) *
                   f32(k[((size_t(b) * context + t) * kh + head) * dim + d]);
    sum = warp_sum(sum);
    if (!lane && t <= positions[b])
        out[(size_t(b) * qh + h) * context + t] = sum * rsqrtf(float(dim));
}
__global__ void softmax(float *x, const int *positions, int context, int qh) {
    __shared__ float red[256];
    int h = blockIdx.x, tid = threadIdx.x, n = positions[h / qh] + 1;
    x += size_t(h) * context;
    float top = -INFINITY;
    for (int i = tid; i < n; i += 256)
        top = fmaxf(top, x[i]);
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
        float v = expf(x[i] - top);
        x[i] = v;
        sum += v;
    }
    red[tid] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (tid < d)
            red[tid] += red[tid + d];
        __syncthreads();
    }
    sum = red[0];
    for (int i = tid; i < n; i += 256)
        x[i] = f32(b16(x[i] / sum));
}
__global__ void weighted_value(const B16 *v, const float *p, B16 *out, const int *positions, int context,
                               int qh, int kh, int dim) {
    int h = blockIdx.x % qh, b = blockIdx.x / qh, d = threadIdx.x;
    if (d >= dim)
        return;
    int head = h / (qh / kh), n = positions[b] + 1;
    float sum = 0;
    for (int t = 0; t < n; ++t)
        sum +=
            p[(size_t(b) * qh + h) * context + t] * f32(v[((size_t(b) * context + t) * kh + head) * dim + d]);
    out[(b * qh + h) * dim + d] = b16(sum);
}
__global__ void tensor_scores(const B16 *q, const B16 *k, float *out, const int *positions, int context,
                              int qh, int kh, int stride) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) unsigned char shared[];
    auto *a = reinterpret_cast<B16 *>(shared);
    auto *b = a + 16 * 32;
    int head = blockIdx.y % kh, batch = blockIdx.y / kh, group = qh / kh;
    int start = blockIdx.x * 64, warp = threadIdx.x / 32;
    wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
    wm::fill_fragment(acc, 0.f);
    for (int part = 0; part < 128; part += 32) {
        for (int i = threadIdx.x; i < 16 * 32; i += 128) {
            int g = i / 32;
            a[i] = g < group ? q[size_t(batch) * stride + (head * group + g) * 128 + part + i % 32] : B16(0);
        }
        for (int i = threadIdx.x; i < 64 * 32; i += 128) {
            int t = start + i / 32;
            b[i] = t < context && t <= positions[batch]
                       ? k[((size_t(batch) * context + t) * kh + head) * 128 + part + i % 32]
                       : B16(0);
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> af;
            wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major> bf;
            wm::load_matrix_sync(af, reinterpret_cast<const __nv_bfloat16 *>(a + kk), 32);
            wm::load_matrix_sync(bf, reinterpret_cast<const __nv_bfloat16 *>(b + warp * 16 * 32 + kk), 32);
            wm::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    auto *values = reinterpret_cast<float *>(shared);
    wm::store_matrix_sync(values + warp * 16, acc, 64, wm::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 64; i += 128) {
        int g = i / 64, t = start + i % 64;
        if (g < group && t < context && t <= positions[batch])
            out[(size_t(batch) * qh + head * group + g) * context + t] = values[i] * rsqrtf(128.f);
    }
#endif
}
__global__ void tensor_values(const B16 *v, const float *p, float *out, const int *positions, int context,
                              int qh, int kh) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) unsigned char shared[];
    auto *a = reinterpret_cast<B16 *>(shared);
    auto *b = a + 16 * 32;
    int head = blockIdx.y % kh, batch = blockIdx.y / kh, group = qh / kh;
    int start = blockIdx.x * 128, warp = threadIdx.x / 32;
    wm::fragment<wm::accumulator, 16, 16, 16, float> acc[2];
    wm::fill_fragment(acc[0], 0.f);
    wm::fill_fragment(acc[1], 0.f);
    for (int part = 0; part < 128; part += 32) {
        for (int i = threadIdx.x; i < 16 * 32; i += 128) {
            int g = i / 32, t = start + part + i % 32;
            a[i] = g < group && t < context && t <= positions[batch]
                       ? b16(p[(size_t(batch) * qh + head * group + g) * context + t])
                       : B16(0);
        }
        for (int i = threadIdx.x; i < 32 * 128; i += 128) {
            int t = start + part + i / 128;
            b[i] = t < context && t <= positions[batch]
                       ? v[((size_t(batch) * context + t) * kh + head) * 128 + i % 128]
                       : B16(0);
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> af;
            wm::load_matrix_sync(af, reinterpret_cast<const __nv_bfloat16 *>(a + kk), 32);
            for (int n = 0; n < 2; ++n) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::row_major> bf;
                wm::load_matrix_sync(
                    bf, reinterpret_cast<const __nv_bfloat16 *>(b + kk * 128 + warp * 16 + n * 64), 128);
                wm::mma_sync(acc[n], af, bf, acc[n]);
            }
        }
        __syncthreads();
    }
    auto *values = reinterpret_cast<float *>(shared);
    for (int n = 0; n < 2; ++n)
        wm::store_matrix_sync(values + warp * 16 + n * 64, acc[n], 128, wm::mem_row_major);
    __syncthreads();
    size_t at = (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * 16 * 128;
    for (int i = threadIdx.x; i < 16 * 128; i += 128)
        out[at + i] = values[i];
#endif
}
__global__ void sum_values(const float *partials, B16 *out, int parts, int qh, int kh) {
    int h = blockIdx.x % qh, batch = blockIdx.x / qh, d = threadIdx.x, group = qh / kh;
    size_t at = ((size_t(batch) * kh + h / group) * parts * 16 + h % group) * 128 + d;
    float sum = partials[at];
    for (int part = 1; part < parts; ++part)
        sum += partials[at + size_t(part) * 16 * 128];
    out[(size_t(batch) * qh + h) * 128 + d] = b16(sum);
}
__global__ void gather_rows(const B16 *x, B16 *y, const int *selected, int rows, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width)
        y[i] = x[size_t(selected[i / width]) * width + i % width];
}
__global__ void reduce(float *x, size_t n, int splits) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float sum = x[i];
    for (int part = 1; part < splits; ++part)
        sum += x[size_t(part) * n + i];
    x[i] = sum;
}
__global__ void scale_output(const float *x, const float *scale, B16 *y, size_t n, int width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = b16(x[i] * scale[i % width]);
}
__global__ void decode_fp8(const uint8_t *x, B16 *y, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = b16(fp8(x[i]));
}
void launch_check() { glm53::gpu::checked(cudaGetLastError(), "CUDA batch operation launch"); }
} // namespace
void qtip_multiply_batch(QtipMatrix w, const B16 *x, float *z, int stride, int splits, int tokens, Stream s) {
    if (tokens == 1 || !native_bf16())
        qtip_multiply(w, x, z, stride, splits, s, tokens);
    else if (tokens <= 16) {
        if (w.in % 32 || w.out % 32 || stride < w.out || tokens <= 0 || splits <= 0 || splits > 32 ||
            splits > w.in / 32)
            throw std::runtime_error("Invalid register QTIP matmul geometry");
        trellis_mma8<<<dim3((w.out + 127) / 128, splits, (tokens + 7) / 8), 256, 0, s>>>(w, x, z, stride,
                                                                                       tokens);
        glm53::gpu::checked(cudaGetLastError(), "register QTIP matmul launch");
    } else
        launch<0>(w, nullptr, nullptr, nullptr, x, z, stride, splits, tokens, s);
}
void qtip_decode_matrix(QtipMatrix w, B16 *out, Stream s) {
    decode_matrix<<<dim3(w.out / 32, w.in / 32), 128, 0, s>>>(w, out);
    glm53::gpu::checked(cudaGetLastError(), "decode QTIP matrix");
}
void bf16_multiply_batch(const B16 *w, const B16 *x, float *y, int in, int rows, int tokens, Stream s) {
    if (tokens == 1 || !native_bf16()) {
        for (int t = 0; t < tokens; ++t)
            glm53::gpu::gemv({w, nullptr, nullptr, rows, in, 4, false, 1}, x + size_t(t) * in,
                             y + size_t(t) * rows, true, s);
    } else
        launch<1>({nullptr, nullptr, nullptr, nullptr, in, rows}, w, nullptr, nullptr, x, y, rows, 1, tokens,
                  s);
}
void fp8_multiply_batch(const uint8_t *w, const float *scale, const B16 *x, B16 *y, int in, int rows,
                        int tokens, Stream s, float *workspace) {
    if (tokens == 1 || !native_bf16()) {
        for (int t = 0; t < tokens; ++t)
            fp8_forward(w, scale, x + size_t(t) * in, y + size_t(t) * rows, in, rows, s);
    } else if (workspace) {
        int parts = std::min(32, in / 32);
        launch<3>({nullptr, nullptr, nullptr, nullptr, in, rows}, nullptr, w, nullptr, x, workspace, rows,
                  parts, tokens, s);
        reduce_products(workspace, size_t(tokens) * rows, parts, s);
        scale_to_b16(workspace, scale, y, tokens, rows, s);
    } else
        launch<2>({nullptr, nullptr, nullptr, nullptr, in, rows}, nullptr, w, scale, x, y, rows, 1, tokens,
                  s);
}
void embedding_batch(const B16 *w, B16 *x, const int *ids, int rows, int width, Stream s) {
    embed<<<(rows * width + 255) / 256, 256, 0, s>>>(w, x, ids, rows, width);
    launch_check();
}
void norm_batch(const B16 *x, const float *w, B16 *y, int rows, int width, float eps, Stream s) {
    rms<<<rows, 256, 0, s>>>(x, w, y, width, eps);
    launch_check();
}
void rope_batch(B16 *q, B16 *k, const int *positions, int rows, int sequence, int qh, int kh, int dim,
                float theta, Stream s, int qstride) {
    rotate<<<dim3(qh + kh, rows), 128, 0, s>>>(q, k, positions, sequence, qh, kh, dim, theta,
                                               qstride ? qstride : qh * dim);
    launch_check();
}
void cache_batch(const B16 *k, const B16 *v, B16 *keys, B16 *values, const int *positions, int batch,
                 int sequence, int context, int width, Stream s) {
    cache<<<(size_t(batch) * sequence * width + 255) / 256, 256, 0, s>>>(k, v, keys, values, positions, batch,
                                                                         sequence, context, width);
    launch_check();
}
void gqa_batch(const B16 *q, const B16 *k, const B16 *v, B16 *out, float *work, const int *positions,
               int batch, int context, int max_tokens, int qh, int kh, int dim, Stream s, int qstride,
               float *value_workspace) {
    if (value_workspace && dim == 128 && qh / kh <= 16 && native_bf16()) {
        tensor_scores<<<dim3((max_tokens + 63) / 64, batch * kh), 128, 16 * 32 * 2 + 64 * 32 * 2, s>>>(
            q, k, work, positions, context, qh, kh, qstride ? qstride : qh * dim);
        softmax<<<batch * qh, 256, 0, s>>>(work, positions, context, qh);
        int parts = (max_tokens + 127) / 128;
        tensor_values<<<dim3(parts, batch * kh), 128, 16 * 32 * 2 + 32 * 128 * 2, s>>>(
            v, work, value_workspace, positions, context, qh, kh);
        sum_values<<<batch * qh, 128, 0, s>>>(value_workspace, out, parts, qh, kh);
        launch_check();
        return;
    }
    scores<<<dim3((max_tokens + 3) / 4, batch * qh), 128, 0, s>>>(q, k, work, positions, context, qh, kh, dim,
                                                                  qstride ? qstride : qh * dim);
    softmax<<<batch * qh, 256, 0, s>>>(work, positions, context, qh);
    weighted_value<<<batch * qh, 128, 0, s>>>(v, work, out, positions, context, qh, kh, dim);
    launch_check();
}
void select_rows(const B16 *x, B16 *y, const int *selected, int rows, int width, Stream s) {
    gather_rows<<<(rows * width + 255) / 256, 256, 0, s>>>(x, y, selected, rows, width);
    launch_check();
}
void reduce_products(float *x, size_t n, int splits, Stream s) {
    if (splits > 1)
        reduce<<<(n + 255) / 256, 256, 0, s>>>(x, n, splits);
    launch_check();
}
void scale_to_b16(const float *x, const float *scale, B16 *y, int rows, int width, Stream s) {
    size_t n = size_t(rows) * width;
    scale_output<<<(n + 255) / 256, 256, 0, s>>>(x, scale, y, n, width);
    launch_check();
}
void decode_fp8_matrix(const uint8_t *x, B16 *y, size_t n, Stream s) {
    decode_fp8<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    launch_check();
}
} // namespace behemoth::cuda_runtime
