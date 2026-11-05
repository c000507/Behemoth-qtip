#include "../src/qtip_tables.hpp"
#include "kernels.hpp"
#include <cmath>
#include <stdexcept>
namespace behemoth::cuda_runtime {
namespace {
__constant__ int8_t had12[144], had28[784];
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
__device__ float warp_sum(float x) {
    for (int d = 16; d; d /= 2)
        x += __shfl_down_sync(0xffffffffU, x, d);
    return x;
}
template <bool Partial>
__global__ void fwht(const B16 *x, const float *z, const float *su, float *out, int p, bool round_z,
                     int splits = 1, int stride = 0, int width = 0, bool rank_major = false,
                     bool b16_z = false) {
    extern __shared__ float a[];
    int block = blockIdx.x;
    for (int i = threadIdx.x; i < p; i += blockDim.x) {
        int j = block * p + i, offset = blockIdx.y * width;
        int source = offset + j;
        if (rank_major) {
            int half = width / 2;
            source = ((j / half) * gridDim.y + blockIdx.y) * half + j % half;
        }
        float v;
        if (x)
            v = f32(x[source]) * su[j];
        else {
            v = b16_z ? f32(reinterpret_cast<const B16 *>(z)[source]) : z[source];
            if constexpr (Partial)
                for (int part = 1; part < splits; ++part)
                    v += z[size_t(part) * stride + source];
            if (round_z)
                v = f32(b16(v));
        }
        a[i] = v;
    }
    __syncthreads();
    for (int s = 1; s < p; s *= 2) {
        for (int t = threadIdx.x; t < p / 2; t += blockDim.x) {
            int i = (t / s) * (2 * s) + t % s;
            float v = a[i], w = a[i + s];
            a[i] = v + w;
            a[i + s] = v - w;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < p; i += blockDim.x)
        out[blockIdx.y * width + block * p + i] = a[i];
}
__global__ void cross_had(const float *x, const float *sv, B16 *out, int k, int p, bool transpose) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= p)
        return;
    const int8_t *h = k == 12 ? had12 : had28;
    float sum = 0;
    int offset = blockIdx.z * k * p;
    for (int c = 0; c < k; ++c)
        sum += (k == 1 ? 1 : transpose ? h[c * k + r] : h[r * k + c]) * x[offset + c * p + i];
    int at = r * p + i;
    out[offset + at] = b16(sum * rsqrtf(float(k * p)) * (sv ? sv[at] : 1));
}
// A prefill CTA computes every small-Hadamard output row for 32 columns.
// Share the FP32 inputs across warps instead of reading them K times through L2.
template <int K>
__global__ void cross_had_rows(const float *x, const float *sv, B16 *out, int p, bool transpose) {
    __shared__ float tile[K * 32];
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    int start = blockIdx.x * 32;
    const int8_t *h = K == 12 ? had12 : had28;
    size_t offset = size_t(blockIdx.y) * K * p;
    for (int j = threadIdx.x; j < K * 32; j += blockDim.x) {
        int column = start + j % 32;
        tile[j] = column < p ? x[offset + (j / 32) * p + column] : 0.f;
    }
    __syncthreads();
    int i = start + lane;
    if (i >= p)
        return;
    for (int r = warp; r < K; r += blockDim.x / 32) {
        float sum = 0;
#pragma unroll
        for (int c = 0; c < K; ++c)
            sum += (transpose ? h[c * K + r] : h[r * K + c]) * tile[c * 32 + lane];
        int at = r * p + i;
        out[offset + at] = b16(sum * rsqrtf(float(K * p)) * (sv ? sv[at] : 1));
    }
}
void cross_transform(const float *x, const float *sv, B16 *out, int k, int p, bool transpose, int tokens,
                     Stream stream) {
    if (tokens >= 32 && k == 12)
        cross_had_rows<12><<<dim3((p + 31) / 32, tokens), 256, 0, stream>>>(x, sv, out, p, transpose);
    else if (tokens >= 32 && k == 28)
        cross_had_rows<28><<<dim3((p + 31) / 32, tokens), 256, 0, stream>>>(x, sv, out, p, transpose);
    else
        cross_had<<<dim3((p + 255) / 256, k, tokens), 256, 0, stream>>>(x, sv, out, k, p, transpose);
}
template <bool Split> __global__ void trellis_gemv(QtipMatrix w, const B16 *x, float *out, int stride) {
    __shared__ uint32_t table[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        table[i] = w.lut[i];
    __syncthreads();
    int lane = threadIdx.x & 31, br = blockIdx.x * 4 + threadIdx.x / 32;
    if (br >= w.out / 32)
        return;
    int A = lane / 4, B = lane % 4, n2 = w.in / 32;
    float acc[4] = {0, 0, 0, 0};
    // Each CTA decodes a contiguous input slice. This supplies enough independent
    // work even when the output has fewer row blocks than the GPU has SMs.
    int first = 0, last = n2;
    if constexpr (Split) {
        first = n2 * blockIdx.y / gridDim.y;
        last = n2 * (blockIdx.y + 1) / gridDim.y;
    }
    size_t offset = size_t(blockIdx.z) * stride;
    x += size_t(blockIdx.z) * w.in;
    if constexpr (Split)
        offset += size_t(blockIdx.y) * gridDim.z * stride;
    for (int j0 = first; j0 < last; ++j0) {
        size_t base = (size_t(br) * n2 + j0) * 128;
        ushort4 a = reinterpret_cast<const ushort4 *>(w.trellis + base)[lane];
        ushort4 b = reinterpret_cast<const ushort4 *>(w.trellis + base)[(lane + 1) & 31];
        const uint16_t av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
        for (int j1 = 0; j1 < 2; ++j1)
            for (int i1 = 0; i1 < 2; ++i1) {
                uint32_t u = uint32_t(av[j1 * 2 + i1]) << 16 | bv[j1 * 2 + i1];
                for (int t = 0; t < 4; ++t) {
                    uint32_t s = (u >> (16 - 4 * t)) & 0xffffU, e = ((s * (s + 1)) >> 6) & 1023U,
                             pair = table[e];
                    int col = j0 * 32 + j1 * 16 + (t / 2) * 8 + B * 2;
                    acc[i1 * 2 + (t & 1)] +=
                        f32(B16(pair)) * f32(x[col]) + f32(B16(pair >> 16)) * f32(x[col + 1]);
                }
            }
    }
    for (int i = 0; i < 4; ++i) {
        acc[i] += __shfl_down_sync(0xffffffffU, acc[i], 2, 4);
        acc[i] += __shfl_down_sync(0xffffffffU, acc[i], 1, 4);
        if (!B)
            out[offset + br * 32 + (i / 2) * 16 + (i % 2) * 8 + A] = acc[i];
    }
}
__global__ void fp8_gemv(const uint8_t *w, const float *scales, const B16 *x, B16 *y, int in, int rows) {
    int r = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
    float sum = 0;
    if (r < rows)
        for (int c = lane; c < in; c += 32)
            sum += fp8(w[size_t(r) * in + c]) * f32(x[c]);
    sum = warp_sum(sum);
    if (!lane && r < rows)
        y[r] = b16(sum * scales[r]);
}
__global__ void embed_kernel(const B16 *w, B16 *y, int token, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = w[size_t(token) * n + i];
}
__global__ void add_kernel(B16 *x, const B16 *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = b16(f32(x[i]) + f32(y[i]));
}
__global__ void rope_kernel(B16 *q, B16 *k, int position, int qheads, int dim, float theta) {
    int h = blockIdx.x, d = threadIdx.x;
    if (d >= dim / 2)
        return;
    B16 *x = h < qheads ? q + size_t(h) * dim : k + size_t(h - qheads) * dim;
    float angle = float(position) * powf(theta, -float(2 * d) / dim), s, c;
    sincosf(angle, &s, &c);
    float a = f32(x[d]), b = f32(x[d + dim / 2]);
    x[d] = b16(a * c - b * s);
    x[d + dim / 2] = b16(b * c + a * s);
}
// llama3 rope_scaling: inv_freq[dim/2] precomputed on the host in double precision
__global__ void rope_scaled_kernel(B16 *q, B16 *k, int position, int qheads, int dim, const float *inv_freq) {
    int h = blockIdx.x, d = threadIdx.x;
    if (d >= dim / 2)
        return;
    B16 *x = h < qheads ? q + size_t(h) * dim : k + size_t(h - qheads) * dim;
    float angle = float(position) * inv_freq[d], s, c;
    sincosf(angle, &s, &c);
    float a = f32(x[d]), b = f32(x[d + dim / 2]);
    x[d] = b16(a * c - b * s);
    x[d + dim / 2] = b16(b * c + a * s);
}
__global__ void gqa_scores(const B16 *q, const B16 *k, float *out, int tokens, int qheads, int kvheads,
                           int dim) {
    int h = blockIdx.y, t = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31,
        kh = h / (qheads / kvheads);
    float sum = 0;
    if (t < tokens)
        for (int d = lane; d < dim; d += 32)
            sum += f32(q[h * dim + d]) * f32(k[(size_t(t) * kvheads + kh) * dim + d]);
    sum = warp_sum(sum);
    if (!lane && t < tokens)
        out[h * tokens + t] = sum * rsqrtf(float(dim));
}
__global__ void probabilities(float *x, int n) {
    __shared__ float red[256];
    int h = blockIdx.x, tid = threadIdx.x;
    float top = -INFINITY;
    for (int i = tid; i < n; i += 256)
        top = fmaxf(top, x[h * n + i]);
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
        float a = expf(x[h * n + i] - top);
        x[h * n + i] = a;
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
    for (int i = tid; i < n; i += 256)
        x[h * n + i] = f32(b16(x[h * n + i] / sum));
}
__global__ void gqa_value(const B16 *v, const float *p, B16 *out, int tokens, int qheads, int kvheads,
                          int dim) {
    int h = blockIdx.x, d = threadIdx.x, kh = h / (qheads / kvheads);
    if (d < dim) {
        float sum = 0;
        for (int t = 0; t < tokens; ++t)
            sum += p[h * tokens + t] * f32(v[(size_t(t) * kvheads + kh) * dim + d]);
        out[h * dim + d] = b16(sum);
    }
}
int factor(int n) {
    for (int k : {28, 12})
        if (n % k == 0 && ((n / k) & (n / k - 1)) == 0)
            return k;
    if (n > 0 && (n & (n - 1)) == 0)
        return 1;
    throw std::runtime_error("Unsupported CUDA Hadamard dimension");
}
void launch_check() { glm53::gpu::checked(cudaGetLastError(), "Behemoth CUDA kernel launch"); }
} // namespace
void initialize_tables() {
    glm53::gpu::checked(cudaMemcpyToSymbol(had12, behemoth::qtip::kHad12, sizeof(had12)), "upload had12");
    glm53::gpu::checked(cudaMemcpyToSymbol(had28, behemoth::qtip::kHad28, sizeof(had28)), "upload had28");
}
void qtip_prepare(QtipMatrix w, const B16 *x, float *scratch, B16 *rotated, Stream s, int tokens,
                  bool rank_major) {
    int ki = factor(w.in), pi = w.in / ki;
    // pi * 4 bytes of dynamic shared memory in fwht: 8192 (Llama 3.3 hidden) needs 32 KiB
    if (w.in % 32 || pi > 8192)
        throw std::runtime_error("Unsupported CUDA QTIP geometry");
    fwht<false><<<dim3(ki, tokens), 256, size_t(pi) * 4, s>>>(x, nullptr, w.su, scratch, pi, false, 1, 0,
                                                              w.in, rank_major);
    cross_transform(scratch, nullptr, rotated, ki, pi, true, tokens, s);
    launch_check();
}
void qtip_multiply(QtipMatrix w, const B16 *rotated, float *z, int stride, int splits, Stream s, int tokens) {
    if (w.in % 32 || w.out % 32 || stride < w.out || splits < 1 || splits > 32 || splits > w.in / 32)
        throw std::runtime_error("Invalid CUDA QTIP projection geometry");
    if (splits == 1)
        trellis_gemv<false><<<dim3((w.out / 32 + 3) / 4, 1, tokens), 128, 0, s>>>(w, rotated, z, stride);
    else
        trellis_gemv<true><<<dim3((w.out / 32 + 3) / 4, splits, tokens), 128, 0, s>>>(w, rotated, z, stride);
    launch_check();
}
void qtip_finish(QtipMatrix w, const float *z, B16 *y, float *scratch, bool rz, int splits, Stream s,
                 int tokens, bool rank_major) {
    int ko = factor(w.out), po = w.out / ko;
    if (w.out % 32 || po > 8192 || splits < 1 || splits > 32)
        throw std::runtime_error("Invalid CUDA QTIP output geometry");
    if (splits == 1) {
        fwht<false><<<dim3(ko, tokens), 256, size_t(po) * 4, s>>>(nullptr, z, nullptr, scratch, po, rz, 1, 0,
                                                                  w.out, rank_major);
    } else {
        fwht<true><<<dim3(ko, tokens), 256, size_t(po) * 4, s>>>(nullptr, z, nullptr, scratch, po, rz, splits,
                                                                 tokens * w.out, w.out, rank_major);
    }
    cross_transform(scratch, w.sv, y, ko, po, false, tokens, s);
    launch_check();
}
void qtip_finish_b16(QtipMatrix w, const B16 *z, B16 *y, float *scratch, int tokens, Stream s,
                     bool rank_major) {
    int ko = factor(w.out), po = w.out / ko;
    fwht<false><<<dim3(ko, tokens), 256, size_t(po) * 4, s>>>(nullptr, reinterpret_cast<const float *>(z),
                                                              nullptr, scratch, po, false, 1, 0, w.out,
                                                              rank_major, true);
    cross_transform(scratch, w.sv, y, ko, po, false, tokens, s);
    launch_check();
}
void qtip_forward(QtipMatrix w, const B16 *x, B16 *y, float *scratch, B16 *rotated, float *z, bool rz,
                  Stream s, int splits) {
    qtip_prepare(w, x, scratch, rotated, s);
    qtip_multiply(w, rotated, z, w.out, splits, s);
    qtip_finish(w, z, y, scratch, rz, splits, s);
}
void fp8_forward(const uint8_t *w, const float *scale, const B16 *x, B16 *y, int in, int out, Stream s) {
    fp8_gemv<<<(out + 3) / 4, 128, 0, s>>>(w, scale, x, y, in, out);
    launch_check();
}
void nvfp4_forward(const uint8_t *packed, const uint8_t *block_scales, float global, const B16 *x, B16 *y, int in,
                   int out, Stream s) {
    // shared GLM gemv (format 2): register-widened E2M1 x E4M3 x global with bf16 rounding, fp32 accumulation
    glm53::gpu::Matrix m{packed, nullptr, block_scales, out, in, 2, false, global};
    glm53::gpu::gemv(m, x, y, false, s);
    glm53::gpu::checked(cudaGetLastError(), "Behemoth CUDA NVFP4 gemv");
}
void embedding(const B16 *w, B16 *y, int t, int h, Stream s) {
    embed_kernel<<<(h + 255) / 256, 256, 0, s>>>(w, y, t, h);
    launch_check();
}
void residual(B16 *x, const B16 *y, int n, Stream s) {
    add_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    launch_check();
}
void rope(B16 *q, B16 *k, int p, int qh, int kh, int d, float theta, Stream s) {
    rope_kernel<<<qh + kh, 128, 0, s>>>(q, k, p, qh, d, theta);
    launch_check();
}
void rope_scaled(B16 *q, B16 *k, int p, int qh, int kh, int d, const float *inv_freq, Stream s) {
    rope_scaled_kernel<<<qh + kh, 128, 0, s>>>(q, k, p, qh, d, inv_freq);
    launch_check();
}
void gqa(const B16 *q, const B16 *k, const B16 *v, B16 *out, float *scores, int n, int qh, int kh, int d,
         Stream s) {
    gqa_scores<<<dim3((n + 3) / 4, qh), 128, 0, s>>>(q, k, scores, n, qh, kh, d);
    probabilities<<<qh, 256, 0, s>>>(scores, n);
    gqa_value<<<qh, 128, 0, s>>>(v, scores, out, n, qh, kh, d);
    launch_check();
}
} // namespace behemoth::cuda_runtime
