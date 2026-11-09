#include "device_ops.hpp"
#include <cmath>
#if defined(GLM53_CUDA)
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#endif
namespace glm53::gpu {
namespace {
__device__ float f32(B16 x) { return __uint_as_float(unsigned(x) << 16); }
__device__ B16 b16(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        return B16((u >> 16) | 0x40);
    return B16((u + 0x7fffU + ((u >> 16) & 1)) >> 16);
}
__device__ float fp8(uint8_t x) {
    int e = (x >> 3) & 15, m = x & 7;
    float v = e ? __uint_as_float(unsigned(e + 120) << 23 | unsigned(m) << 20) : float(m) / 512;
    if (e == 15 && m == 7)
        v = nanf("");
    return x & 128 ? -v : v;
}
__device__ float sigmoid(float x) { return x >= 0 ? 1 / (1 + expf(-x)) : expf(x) / (1 + expf(x)); }
__device__ float shuffle(float x, int d) {
#if defined(GLM53_CUDA)
    return __shfl_down_sync(0xffffffffU, x, d);
#else
    return __shfl_down(x, d, 32);
#endif
}
__device__ float warp_sum(float x) {
    for (int d = 16; d; d /= 2)
        x += shuffle(x, d);
    return x;
}
__device__ float block_sum(float x) {
    __shared__ float partial[32];
    x = warp_sum(x);
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (!lane)
        partial[warp] = x;
    __syncthreads();
    x = threadIdx.x < blockDim.x / 32 ? partial[lane] : 0;
    if (!warp)
        x = warp_sum(x);
    if (!threadIdx.x)
        partial[0] = x;
    __syncthreads();
    float result = partial[0];
    __syncthreads(); // callers may immediately reuse this reduction's shared storage
    return result;
}
__device__ float weight(Matrix w, size_t i) {
    float v;
    if (w.format == 0)
        v = static_cast<const float *>(w.data)[i];
    else if (w.format == 4)
        return f32(static_cast<const B16 *>(w.data)[i]);
    else if (w.format == 3)
        v = __half2float(static_cast<const __half *>(w.data)[i]);
    else if (w.format == 1) {
        size_t r = i / w.cols, c = i % w.cols;
        size_t s = w.block ? (r / 128) * ((w.cols + 127) / 128) + c / 128 : r;
        v = fp8(static_cast<const uint8_t *>(w.data)[i]) * w.scales[s];
    } else {
        uint8_t packed = static_cast<const uint8_t *>(w.data)[i / 2];
        int code = (packed >> (4 * (i & 1))) & 15;
        int n = code & 7;
        float magnitude =
            n < 2 ? float(n) * .5f : float(2 + (n & 1)) * __uint_as_float(unsigned((n >> 1) + 125) << 23);
        v = (code & 8 ? -magnitude : magnitude) * fp8(w.block_scales[i / 16]) * w.global;
    }
    // W16A16 semantics without BF16/FP8 tensor-core requirements (sm_75).
    return f32(b16(v));
}
__device__ float low(unsigned u) { return __uint_as_float(u << 16); }
__device__ float high(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__global__ void gemv_kernel(Matrix w, const B16 *x, void *out, bool full, bool vector) {
    int row = int(blockIdx.x) * 4 + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    float sum = 0;
    if (row < w.rows) {
        if (vector) {
            // BF16 rows with 16-byte loads: eight weights and eight activations per lane and step. The
            // products are the same bf16-exact values as the scalar path; only the summation order differs.
            const uint4 *wr = reinterpret_cast<const uint4 *>(static_cast<const B16 *>(w.data) + size_t(row) * w.cols);
            const uint4 *xv = reinterpret_cast<const uint4 *>(x);
            for (int c = lane; c < w.cols / 8; c += 32) {
                uint4 a = wr[c], b = xv[c];
                sum = fmaf(low(a.x), low(b.x), sum);
                sum = fmaf(high(a.x), high(b.x), sum);
                sum = fmaf(low(a.y), low(b.y), sum);
                sum = fmaf(high(a.y), high(b.y), sum);
                sum = fmaf(low(a.z), low(b.z), sum);
                sum = fmaf(high(a.z), high(b.z), sum);
                sum = fmaf(low(a.w), low(b.w), sum);
                sum = fmaf(high(a.w), high(b.w), sum);
            }
        } else
            for (int c = lane; c < w.cols; c += 32)
                sum = fmaf(weight(w, size_t(row) * w.cols + c), f32(x[c]), sum);
    }
    sum = warp_sum(sum);
    if (!lane && row < w.rows) {
        if (full)
            static_cast<float *>(out)[row] = sum;
        else
            static_cast<B16 *>(out)[row] = b16(sum);
    }
}
__global__ void decode_kernel(Matrix w, B16 *out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(w.rows) * w.cols)
        out[i] = b16(weight(w, i));
}
template <bool to_b16> __global__ void convert_kernel(const void *in, void *out, int n) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        if constexpr (to_b16)
            static_cast<B16 *>(out)[i] = b16(static_cast<const float *>(in)[i]);
        else
            static_cast<float *>(out)[i] = f32(static_cast<const B16 *>(in)[i]);
    }
}
__global__ void replicate_kernel(const B16 *in, B16 *out, int hidden, int streams) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < hidden * streams)
        out[i] = in[i % hidden];
}
__global__ void mean_kernel(const B16 *in, B16 *out, int hidden, int streams) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < hidden) {
        float a = 0;
        for (int s = 0; s < streams; ++s)
            a += f32(in[s * hidden + i]) / streams;
        out[i] = b16(a);
    }
}
__global__ void norm_kernel(const B16 *x, const float *w, const float *bias, B16 *out, int n, float eps,
                            bool centered) {
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += f32(x[i]);
    float mean = centered ? block_sum(sum) / n : 0;
    sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float a = f32(x[i]) - mean;
        sum += a * a;
    }
    float inv = rsqrtf(block_sum(sum) / n + eps);
    // Every thread has read all source values before any in-place writes.
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = b16((f32(x[i]) - mean) * inv * w[i] + (bias ? bias[i] : 0));
}
__global__ void swiglu_kernel(B16 *gate, const B16 *up, int n, float limit) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = f32(gate[i]), b = f32(up[i]);
        a = a > limit ? limit : a;
        b = b > limit ? limit : b < -limit ? -limit : b;
        gate[i] = b16(a * sigmoid(a) * b);
    }
}
__global__ void axpy_kernel(float *out, const B16 *x, int n, float alpha) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] += alpha * f32(x[i]);
}
__global__ void sum_slots_kernel(const float *slots, int ranks, int n, float *out) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = 0;
        for (int r = 0; r < ranks; ++r)
            a += slots[size_t(r) * n + i];
        out[i] = a;
    }
}
__global__ void combine_kernel(const float *outputs, const float *factors, int experts, int n, B16 *out) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = f32(b16(outputs[i]));
        for (int e = 0; e < experts; ++e)
            a += factors[e] * f32(b16(outputs[size_t(e + 1) * n + i]));
        out[i] = b16(a);
    }
}
__global__ void float_matvec_kernel(const float *w, const B16 *x, float *y, int rows, int cols) {
    int r = int(blockIdx.x) * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
    float sum = 0;
    if (r < rows)
        for (int c = lane; c < cols; c += 32)
            sum += w[size_t(r) * cols + c] * f32(x[c]);
    sum = warp_sum(sum);
    if (!lane && r < rows)
        y[r] = sum;
}
__global__ void hc_coeff(const B16 *x, const float *fn, float *coeff, int width, float eps) {
    float ss = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float a = f32(x[i]);
        ss += a * a;
    }
    float inv = rsqrtf(block_sum(ss) / width + eps), sum = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        sum += fn[size_t(blockIdx.x) * width + i] * (f32(x[i]) * inv);
    sum = block_sum(sum);
    if (!threadIdx.x)
        coeff[blockIdx.x] = sum;
}
__global__ void hc_mix(const float *coeff, const float *base, const float *scale, float *pre, float *post,
                       float *mix, int n, int iters, float eps) {
    if (threadIdx.x)
        return;
    for (int i = 0; i < n; ++i) {
        pre[i] = sigmoid(coeff[i] * scale[0] + base[i]) + eps;
        post[i] = f32(b16(2 * sigmoid(coeff[n + i] * scale[1] + base[n + i])));
        float maximum = -INFINITY, sum = 0;
        for (int j = 0; j < n; ++j)
            maximum = fmaxf(maximum, coeff[2 * n + i * n + j] * scale[2] + base[2 * n + i * n + j]);
        for (int j = 0; j < n; ++j) {
            float a = expf(coeff[2 * n + i * n + j] * scale[2] + base[2 * n + i * n + j] - maximum);
            mix[i * n + j] = a;
            sum += a;
        }
        for (int j = 0; j < n; ++j)
            mix[i * n + j] = mix[i * n + j] / sum + eps;
    }
    for (int iter = 0; iter < iters; ++iter) {
        if (iter)
            for (int i = 0; i < n; ++i) {
                float sum = eps;
                for (int j = 0; j < n; ++j)
                    sum += mix[i * n + j];
                for (int j = 0; j < n; ++j)
                    mix[i * n + j] /= sum;
            }
        for (int j = 0; j < n; ++j) {
            float sum = eps;
            for (int i = 0; i < n; ++i)
                sum += mix[i * n + j];
            for (int i = 0; i < n; ++i)
                mix[i * n + j] /= sum;
        }
    }
    for (int i = 0; i < n * n; ++i)
        mix[i] = f32(b16(mix[i]));
}
__global__ void hc_collapse(const B16 *x, const float *pre, B16 *out, int hidden, int n) {
    int d = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d < hidden) {
        float a = 0;
        for (int i = 0; i < n; ++i)
            a += pre[i] * f32(x[i * hidden + d]);
        out[d] = b16(a);
    }
}
__global__ void hc_post_kernel(const B16 *x, const B16 *y, const float *post, const float *mix, B16 *out,
                               int hidden, int n) {
    int at = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < hidden * n) {
        int s = at / hidden, d = at % hidden;
        float a = post[s] * f32(y[d]);
        for (int j = 0; j < n; ++j)
            a += mix[j * n + s] * f32(x[j * hidden + d]);
        out[at] = b16(a);
    }
}
__global__ void conv_kernel(B16 *q, B16 *k, B16 *v, B16 *history, const float *qw, const float *kw,
                            const float *vw, int width, int kernel) {
    int channel = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= width)
        return;
    for (int part = 0; part < 3; ++part) {
        B16 *x = part == 0 ? q : part == 1 ? k : v;
        const float *w = part == 0 ? qw : part == 1 ? kw : vw;
        size_t at = (size_t(part) * width + channel) * (kernel - 1);
        float a = f32(x[channel]) * w[size_t(channel) * kernel + kernel - 1];
        for (int j = 0; j < kernel - 1; ++j)
            a += f32(history[at + j]) * w[size_t(channel) * kernel + j];
        for (int j = 0; j < kernel - 2; ++j)
            history[at + j] = history[at + j + 1];
        if (kernel > 1)
            history[at + kernel - 2] = x[channel];
        x[channel] = b16(a * sigmoid(a));
    }
}
__global__ void kda_kernel(const B16 *q, const B16 *k, const B16 *v, const B16 *g, const B16 *beta,
                           const float *dt, const float *alog, float *state, B16 *out, int dim, float lower) {
    extern __shared__ float values[];
    float *qn = values, *kn = qn + dim, *decay = kn + dim;
    int h = blockIdx.x;
    float qs = 0, ks = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = f32(q[h * dim + i]), b = f32(k[h * dim + i]);
        qs += a * a;
        ks += b * b;
    }
    float qi = rsqrtf(block_sum(qs) + 1e-6f) * rsqrtf(float(dim));
    float ki = rsqrtf(block_sum(ks) + 1e-6f);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        qn[i] = f32(q[h * dim + i]) * qi;
        kn[i] = f32(k[h * dim + i]) * ki;
        decay[i] = expf(lower * sigmoid(expf(alog[h]) * (f32(g[h * dim + i]) + dt[h * dim + i])));
    }
    __syncthreads();
    float b = f32(b16(sigmoid(f32(beta[h]))));
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float memory = 0;
        for (int i = 0; i < dim; ++i) {
            size_t at = (size_t(h) * dim + i) * dim + d;
            state[at] *= decay[i];
            memory += state[at] * kn[i];
        }
        float delta = (f32(v[h * dim + d]) - memory) * b, sum = 0;
        for (int i = 0; i < dim; ++i) {
            size_t at = (size_t(h) * dim + i) * dim + d;
            state[at] += kn[i] * delta;
            sum += state[at] * qn[i];
        }
        out[h * dim + d] = b16(sum);
    }
}
__global__ void gated_norm_kernel(B16 *x, const B16 *gate, const float *w, int dim, float eps) {
    int h = blockIdx.x;
    float ss = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = f32(x[h * dim + i]);
        ss += a * a;
    }
    float inv = rsqrtf(block_sum(ss) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int at = h * dim + i;
        x[at] = b16(f32(x[at]) * inv * w[i] * sigmoid(f32(gate[at])));
    }
}
__global__ void pool_kernel(const B16 *keys, const B16 *gates, const float *ape, B16 *out, int pool,
                            int dim) {
    int d = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= dim)
        return;
    float top = -INFINITY, den = 0, key = 0;
    for (int t = 0; t < pool; ++t)
        top = fmaxf(top, f32(gates[t * dim + d]) + ape[t * dim + d]);
    for (int t = 0; t < pool; ++t)
        den += expf(f32(gates[t * dim + d]) + ape[t * dim + d] - top);
    for (int t = 0; t < pool; ++t) {
        float prob = f32(b16(expf(f32(gates[t * dim + d]) + ape[t * dim + d] - top) / den));
        key += f32(b16(prob * f32(keys[t * dim + d])));
    }
    out[d] = b16(key);
}
__global__ void index_kernel(const B16 *query, const B16 *keys, const B16 *weights, float *out, int heads,
                             int dim) {
    int pool = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float sum = 0;
    for (int h = warp; h < heads; h += blockDim.x / 32) {
        float dot = 0;
        for (int d = lane; d < dim; d += 32)
            dot += f32(query[h * dim + d]) * f32(keys[size_t(pool) * dim + d]);
        dot = warp_sum(dot);
        if (!lane)
            sum += fmaxf(0.f, dot * rsqrtf(float(dim))) * f32(weights[h]) * rsqrtf(float(heads));
    }
    sum = block_sum(sum);
    if (!threadIdx.x)
        out[pool] = sum;
}
__device__ bool better(float a, int ai, float b, int bi) { return a > b || (a == b && ai < bi); }
__global__ void sort_chunks(float *scores, int *ids, int n, int padded, int final_size) {
    __shared__ float s[1024];
    __shared__ int idx[1024];
    int start = blockIdx.x * 1024;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        int g = start + i;
        s[i] = g < n ? scores[g] : -INFINITY;
        idx[i] = g;
        if (final_size) {
            s[i] = g < padded ? scores[g] : -INFINITY;
            idx[i] = g < padded ? ids[g] : g;
        }
    }
    __syncthreads();
    // Initial bitonic chunks alternate direction. Later launches finish a global merge.
    int first = final_size ? 512 : 1;
    for (int size = final_size ? final_size : 2; size <= (final_size ? final_size : 1024); size *= 2) {
        for (int step = final_size ? first : size / 2; step; step /= 2) {
            for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
                int j = i ^ step;
                if (j > i) {
                    bool desc = ((start + i) & size) == 0;
                    bool swap =
                        desc ? better(s[j], idx[j], s[i], idx[i]) : better(s[i], idx[i], s[j], idx[j]);
                    if (swap) {
                        float a = s[i];
                        s[i] = s[j];
                        s[j] = a;
                        int b = idx[i];
                        idx[i] = idx[j];
                        idx[j] = b;
                    }
                }
            }
            __syncthreads();
        }
        if (final_size)
            break;
    }
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        if (start + i < padded) {
            scores[start + i] = s[i];
            ids[start + i] = idx[i];
        }
}
__global__ void sort_global(float *scores, int *ids, int n, int size, int step) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x, j = i ^ step;
    if (i >= n || j >= n || j <= i)
        return;
    bool desc = (i & size) == 0;
    bool swap =
        desc ? better(scores[j], ids[j], scores[i], ids[i]) : better(scores[i], ids[i], scores[j], ids[j]);
    if (swap) {
        float a = scores[i];
        scores[i] = scores[j];
        scores[j] = a;
        int b = ids[i];
        ids[i] = ids[j];
        ids[j] = b;
    }
}
__global__ void select_kernel(const int *ids, int *out, int pools, int pool, int count, int position) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    int full = count * pool, tail = position + 1 - pools * pool;
    if (i < full + tail)
        out[i] = i < full ? ids[i / pool] * pool + i % pool : pools * pool + i - full;
}
__global__ void query_kernel(const B16 *q, const B16 *kv, float *out, int qdim, int vdim, int rank) {
    int j = int(blockIdx.x) * blockDim.x + threadIdx.x, h = blockIdx.y;
    if (j < rank) {
        float sum = 0;
        for (int d = 0; d < qdim; ++d)
            sum += f32(q[h * qdim + d]) * f32(kv[(size_t(h) * (qdim + vdim) + d) * rank + j]);
        out[h * rank + j] = sum;
    }
}
__global__ void mla_scores(const float *q, const B16 *latent, const int *selected, float *scores, int count,
                           int qdim, int rank) {
    int t = blockIdx.x * 4 + threadIdx.x / 32, h = blockIdx.y, lane = threadIdx.x & 31;
    float dot = 0;
    if (t < count)
        for (int j = lane; j < rank; j += 32)
            dot += q[h * rank + j] * f32(latent[size_t(selected[t]) * rank + j]);
    dot = warp_sum(dot);
    if (!lane && t < count)
        scores[h * count + t] = dot * rsqrtf(float(qdim));
}
__global__ void softmax_kernel(float *scores, int n) {
    // Use max reduction through shared memory; all 256 threads take every barrier.
    __shared__ float s[256];
    int h = blockIdx.x, tid = threadIdx.x;
    float top = -INFINITY;
    for (int t = tid; t < n; t += blockDim.x)
        top = fmaxf(top, scores[h * n + t]);
    s[tid] = top;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (tid < d)
            s[tid] = fmaxf(s[tid], s[tid + d]);
        __syncthreads();
    }
    top = s[0];
    float sum = 0;
    for (int t = tid; t < n; t += blockDim.x) {
        float a = expf(scores[h * n + t] - top);
        scores[h * n + t] = a;
        sum += a;
    }
    sum = block_sum(sum);
    for (int t = tid; t < n; t += blockDim.x)
        scores[h * n + t] = f32(b16(scores[h * n + t] / sum));
}
__global__ void latent_sum(const B16 *latent, const int *selected, const float *scores, float *out, int count,
                           int rank) {
    int j = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y;
    if (j < rank) {
        float sum = 0;
        for (int t = 0; t < count; ++t)
            sum += scores[h * count + t] * f32(latent[size_t(selected[t]) * rank + j]);
        out[h * rank + j] = sum;
    }
}
__global__ void expand_value(const float *latent, const B16 *kv, B16 *out, int qdim, int vdim, int rank) {
    int d = blockIdx.x * 4 + threadIdx.x / 32, h = blockIdx.y, lane = threadIdx.x & 31;
    float sum = 0;
    if (d < vdim)
        for (int j = lane; j < rank; j += 32)
            sum += latent[h * rank + j] * f32(kv[(size_t(h) * (qdim + vdim) + qdim + d) * rank + j]);
    sum = warp_sum(sum);
    if (!lane && d < vdim)
        out[h * vdim + d] = b16(sum);
}
__global__ void route_kernel(float *scores, const float *bias, int *ids, float *weights, float *gs,
                             int *allowed, int experts, int topk, int groups, int top_groups, bool normalize,
                             float scale) {
    if (threadIdx.x)
        return;
    int per = experts / groups;
    for (int i = 0; i < experts; ++i)
        scores[i] = sigmoid(scores[i]);
    for (int g = 0; g < groups; ++g) {
        float first = -INFINITY, second = -INFINITY;
        for (int e = 0; e < per; ++e) {
            float a = scores[g * per + e] + bias[g * per + e];
            if (a > first) {
                second = first;
                first = a;
            } else
                second = fmaxf(second, a);
        }
        gs[g] = first + second;
        allowed[g] = 0;
    }
    for (int i = 0; i < top_groups; ++i) {
        int best = -1;
        for (int g = 0; g < groups; ++g)
            if (!allowed[g] && (best < 0 || gs[g] > gs[best]))
                best = g;
        allowed[best] = 1;
    }
    float den = normalize ? 1e-20f : 1;
    for (int i = 0; i < topk; ++i) {
        int best = -1;
        for (int e = 0; e < experts; ++e)
            if (allowed[e / per]) {
                bool used = false;
                for (int j = 0; j < i; ++j)
                    used |= ids[j] == e;
                if (!used && (best < 0 || scores[e] + bias[e] > scores[best] + bias[best]))
                    best = e;
            }
        ids[i] = best;
        if (normalize)
            den += scores[best];
    }
    for (int i = 0; i < topk; ++i)
        weights[i] = scores[ids[i]] / den * scale;
}
void launched() { checked(last_error(), "GPU kernel launch"); }
} // namespace
void gemv(Matrix w, const B16 *x, void *y, bool f, Stream s) {
    bool vector = w.format == 4 && w.cols % 8 == 0 && (reinterpret_cast<uintptr_t>(w.data) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(x) & 15) == 0;
    gemv_kernel<<<(w.rows + 3) / 4, 128, 0, s>>>(w, x, y, f, vector);
    launched();
}
void decode_matrix(Matrix w, B16 *out, Stream s) {
    decode_kernel<<<(size_t(w.rows) * w.cols + 255) / 256, 256, 0, s>>>(w, out);
    launched();
}
void from_float(const float *x, B16 *y, int n, Stream s) {
    convert_kernel<true><<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    launched();
}
void to_float(const B16 *x, float *y, int n, Stream s) {
    convert_kernel<false><<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    launched();
}
void replicate(const B16 *x, B16 *y, int h, int n, Stream s) {
    replicate_kernel<<<(h * n + 255) / 256, 256, 0, s>>>(x, y, h, n);
    launched();
}
void mean_streams(const B16 *x, B16 *y, int h, int n, Stream s) {
    mean_kernel<<<(h + 255) / 256, 256, 0, s>>>(x, y, h, n);
    launched();
}
void norm(const B16 *x, const float *w, const float *b, B16 *y, int n, float e, bool c, Stream s) {
    norm_kernel<<<1, 256, 0, s>>>(x, w, b, y, n, e, c);
    launched();
}
void swiglu(B16 *a, const B16 *b, int n, float l, Stream s) {
    swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, b, n, l);
    launched();
}
void axpy(float *a, const B16 *b, int n, float k, Stream s) {
    axpy_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, b, n, k);
    launched();
}
void sum_slots(const float *slots, int ranks, int n, float *out, Stream s) {
    sum_slots_kernel<<<(n + 255) / 256, 256, 0, s>>>(slots, ranks, n, out);
    launched();
}
void combine_experts(const float *outputs, const float *factors, int experts, int n, B16 *out, Stream s) {
    combine_kernel<<<(n + 255) / 256, 256, 0, s>>>(outputs, factors, experts, n, out);
    launched();
}
void float_matvec(const float *w, const B16 *x, float *y, int r, int c, Stream s) {
    float_matvec_kernel<<<(r + 3) / 4, 128, 0, s>>>(w, x, y, r, c);
    launched();
}
void hc_pre(const B16 *x, const float *fn, const float *base, const float *scale, B16 *y, float *post,
            float *mix, float *scratch, int h, int n, int it, float e, float he, Stream s) {
    hc_coeff<<<n *(n + 2), 256, 0, s>>>(x, fn, scratch, h * n, e);
    hc_mix<<<1, 32, 0, s>>>(scratch, base, scale, scratch + n * (n + 2), post, mix, n, it, he);
    hc_collapse<<<(h + 255) / 256, 256, 0, s>>>(x, scratch + n * (n + 2), y, h, n);
    launched();
}
void hc_post(const B16 *x, const B16 *y, const float *p, const float *m, B16 *out, int h, int n, Stream s) {
    hc_post_kernel<<<(h * n + 255) / 256, 256, 0, s>>>(x, y, p, m, out, h, n);
    launched();
}
void convolution(B16 *q, B16 *k, B16 *v, B16 *hist, const float *qw, const float *kw, const float *vw, int w,
                 int n, Stream s) {
    conv_kernel<<<(w + 255) / 256, 256, 0, s>>>(q, k, v, hist, qw, kw, vw, w, n);
    launched();
}
void kda(const B16 *q, const B16 *k, const B16 *v, const B16 *g, const B16 *b, const float *dt,
         const float *a, float *state, B16 *out, int h, int d, float l, Stream s) {
    kda_kernel<<<h, 256, size_t(d) * 12, s>>>(q, k, v, g, b, dt, a, state, out, d, l);
    launched();
}
void gated_norm(B16 *x, const B16 *g, const float *w, int h, int d, float e, Stream s) {
    gated_norm_kernel<<<h, 256, 0, s>>>(x, g, w, d, e);
    launched();
}
void pool_key(const B16 *k, const B16 *g, const float *a, B16 *out, int p, int d, Stream s) {
    pool_kernel<<<(d + 255) / 256, 256, 0, s>>>(k, g, a, out, p, d);
    launched();
}
void index_scores(const B16 *q, const B16 *k, const B16 *w, float *out, int p, int h, int d, Stream s) {
    if (p) {
        index_kernel<<<p, 128, 0, s>>>(q, k, w, out, h, d);
        launched();
    }
}
void select_keys(float *scores, int *ids, int *out, int pools, int padded, int pool, int topk, int pos,
                 Stream s) {
    if (pools) {
        sort_chunks<<<(padded + 1023) / 1024, 256, 0, s>>>(scores, ids, pools, padded, 0);
        for (int size = 2048; size <= padded; size *= 2) {
            for (int step = size / 2; step >= 1024; step /= 2)
                sort_global<<<(padded + 255) / 256, 256, 0, s>>>(scores, ids, padded, size, step);
            sort_chunks<<<(padded + 1023) / 1024, 256, 0, s>>>(scores, ids, pools, padded, size);
        }
    }
    int count = pools < topk / pool ? pools : topk / pool;
    int n = count * pool + pos + 1 - pools * pool;
    select_kernel<<<(n + 255) / 256, 256, 0, s>>>(ids, out, pools, pool, count, pos);
    launched();
}
void absorb_query(const B16 *q, const B16 *kv, float *out, int h, int qd, int vd, int r, Stream s) {
    query_kernel<<<dim3((r + 255) / 256, h), 256, 0, s>>>(q, kv, out, qd, vd, r);
    launched();
}
void mla(const float *q, const B16 *latent, const B16 *kv, const int *selected, B16 *out, float *scores,
         float *weighted, int n, int h, int qd, int vd, int r, Stream s) {
    mla_scores<<<dim3((n + 3) / 4, h), 128, 0, s>>>(q, latent, selected, scores, n, qd, r);
    softmax_kernel<<<h, 256, 0, s>>>(scores, n);
    latent_sum<<<dim3((r + 255) / 256, h), 256, 0, s>>>(latent, selected, scores, weighted, n, r);
    expand_value<<<dim3((vd + 3) / 4, h), 128, 0, s>>>(weighted, kv, out, qd, vd, r);
    launched();
}
void route(float *s, const float *b, int *ids, float *w, float *gs, int *a, int e, int k, int g, int tg,
           bool n, float scale, Stream stream) {
    route_kernel<<<1, 32, 0, stream>>>(s, b, ids, w, gs, a, e, k, g, tg, n, scale);
    launched();
}
} // namespace glm53::gpu
