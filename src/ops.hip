#include "behemoth/gpu.hpp"
#include "device_numeric.hpp"

namespace behemoth {
namespace {
__global__ void cast_kernel(const float *x, const float *scale, uint16_t *y, size_t n, int width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = device::bf16(x[i] * (scale ? scale[i % size_t(width)] : 1.f));
}
__global__ void norm_kernel(const uint16_t *x, const uint16_t *w, uint16_t *y, int width,
                            float epsilon) {
    int row = int(blockIdx.x), lane = int(threadIdx.x);
    __shared__ float sum[256];
    float s = 0;
    for (int d = lane; d < width; d += 256) {
        float v = device::bf16(x[size_t(row) * size_t(width) + size_t(d)]);
        s += v * v;
    }
    sum[lane] = s;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (lane < n)
            sum[lane] += sum[lane + n];
        __syncthreads();
    }
    float factor = rsqrtf(sum[0] / float(width) + epsilon);
    for (int d = lane; d < width; d += 256) {
        size_t i = size_t(row) * size_t(width) + size_t(d);
        y[i] = device::bf16(device::bf16(x[i]) * factor * device::bf16(w[d]));
    }
}
// One block per row; each thread keeps its width/256 normalised values in
// registers between the two reductions, so the quantisation needs no second
// read of the row. Rounding matches norm_kernel (bf16 value) + quantize_kernel.
template <int PER_THREAD>
__global__ void norm_quantize_kernel(const uint16_t *x, const uint16_t *w, uint8_t *q,
                                     float *scales, int width, float epsilon) {
    const int row = int(blockIdx.x), lane = int(threadIdx.x);
    __shared__ float red[256];
    float v[PER_THREAD];
    float s = 0;
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        const int d = lane + i * 256;
        v[i] = d < width ? device::bf16(x[size_t(row) * size_t(width) + size_t(d)]) : 0.f;
        s += v[i] * v[i];
    }
    red[lane] = s;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (lane < n)
            red[lane] += red[lane + n];
        __syncthreads();
    }
    const float factor = rsqrtf(red[0] / float(width) + epsilon);
    __syncthreads();
    float m = 0;
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        const int d = lane + i * 256;
        if (d < width) {
            v[i] = device::bf16(device::bf16(v[i] * factor * device::bf16(w[d])));
            m = fmaxf(m, fabsf(v[i]));
        }
    }
    red[lane] = m;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (lane < n)
            red[lane] = fmaxf(red[lane], red[lane + n]);
        __syncthreads();
    }
    const float scale = device::scale(red[0]);
    if (lane == 0)
        scales[row] = scale;
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        const int d = lane + i * 256;
        if (d < width)
            q[size_t(row) * size_t(width) + size_t(d)] = device::fp8(v[i] / scale);
    }
}
__global__ void residual_scaled_kernel(uint16_t *state, const float *x, const float *scale, size_t n,
                                       int width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float t = device::bf16(device::bf16(x[i] * (scale ? scale[i % size_t(width)] : 1.f)));
        state[i] = device::bf16(device::bf16(state[i]) + t);
    }
}
// grid = (rows, heads + kv_heads), block = 128: blocks < heads rotate one q head
// and quantise it (per-head scale), the rest rotate one k head in place.
__global__ void rope_quantize_kernel(uint16_t *q, uint16_t *k, const int *positions, int heads,
                                     int kv_heads, float theta, uint8_t *q8, float *q_scale) {
    const int row = int(blockIdx.x), h = int(blockIdx.y), lane = int(threadIdx.x);
    const bool query = h < heads;
    uint16_t *x = query ? q + (size_t(row) * size_t(heads) + size_t(h)) * 128
                        : k + (size_t(row) * size_t(kv_heads) + size_t(h - heads)) * 128;
    const int d = lane & 63, half = lane >> 6; // lanes 0-63: pair (d, d+64) handled by lane d
    __shared__ float rotated[128];
    __shared__ float red[128];
    if (half == 0) {
        float angle = float(positions[row]) * powf(theta, -float(2 * d) / 128), s, c;
        sincosf(angle, &s, &c);
        const float a = device::bf16(x[d]), b = device::bf16(x[d + 64]);
        rotated[d] = device::bf16(device::bf16(a * c - b * s));
        rotated[d + 64] = device::bf16(device::bf16(b * c + a * s));
    }
    __syncthreads();
    x[lane] = device::bf16(rotated[lane]);
    if (!query)
        return;
    red[lane] = fabsf(rotated[lane]);
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (lane < s)
            red[lane] = fmaxf(red[lane], red[lane + s]);
        __syncthreads();
    }
    const float scale = device::scale(red[0]);
    const size_t head_row = size_t(row) * size_t(heads) + size_t(h);
    if (lane == 0)
        q_scale[head_row] = scale;
    q8[head_row * 128 + size_t(lane)] = device::fp8(rotated[lane] / scale);
}
__global__ void residual_kernel(uint16_t *state, const uint16_t *x, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        state[i] = device::bf16(device::bf16(state[i]) + device::bf16(x[i]));
}
__global__ void silu_kernel(const uint16_t *gate, const uint16_t *up, uint16_t *y, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = device::bf16(gate[i]);
        y[i] = device::bf16(x / (1 + expf(-x)) * device::bf16(up[i]));
    }
}
__global__ void rope_kernel(uint16_t *x, const int *positions, int rows, int heads, float theta) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * size_t(heads) * 64)
        return;
    int row = int(i / (size_t(heads) * 64)), d = int(i % 64);
    size_t base = (i / 64) * 128 + size_t(d);
    float angle = float(positions[row]) * powf(theta, -float(2 * d) / 128), s, c;
    sincosf(angle, &s, &c);
    float a = device::bf16(x[base]), b = device::bf16(x[base + 64]);
    x[base] = device::bf16(a * c - b * s);
    x[base + 64] = device::bf16(b * c + a * s);
}
__global__ void embed_kernel(const uint16_t *embedding, const int *tokens, uint16_t *out, size_t n,
                             int width) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = embedding[size_t(tokens[i / size_t(width)]) * size_t(width) + i % size_t(width)];
}
} // namespace
void cast_scaled(const float *x, const float *scale, BFloat16 *y, int rows, int width,
                 hipStream_t stream) {
    size_t n = size_t(rows) * size_t(width);
    if (n) {
        cast_kernel<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(
            x, scale, reinterpret_cast<uint16_t *>(y), n, width);
        HIP_CHECK(hipGetLastError());
    }
}
void rms_norm(const BFloat16 *x, const BFloat16 *w, BFloat16 *y, int rows, int width, float epsilon,
              hipStream_t stream) {
    if (rows) {
        norm_kernel<<<unsigned(rows), 256, 0, stream>>>(
            reinterpret_cast<const uint16_t *>(x), reinterpret_cast<const uint16_t *>(w),
            reinterpret_cast<uint16_t *>(y), width, epsilon);
        HIP_CHECK(hipGetLastError());
    }
}
void rms_norm_quantized(const BFloat16 *x, const BFloat16 *w, uint8_t *q, float *scales, int rows,
                        int width, float epsilon, hipStream_t stream) {
    if (width < 1 || width > 256 * 64 || !x || !w || !q || !scales)
        throw std::invalid_argument("Invalid fused norm/quantisation buffers or width");
    if (!rows)
        return;
    auto launch = [&]<int PER>() {
        norm_quantize_kernel<PER><<<unsigned(rows), 256, 0, stream>>>(
            reinterpret_cast<const uint16_t *>(x), reinterpret_cast<const uint16_t *>(w), q, scales,
            width, epsilon);
    };
    const int per = (width + 255) / 256;
    if (per <= 8)
        launch.template operator()<8>();
    else if (per <= 16)
        launch.template operator()<16>();
    else if (per <= 32)
        launch.template operator()<32>();
    else if (per <= 48)
        launch.template operator()<48>();
    else
        launch.template operator()<64>();
    HIP_CHECK(hipGetLastError());
}
void residual_add_scaled(BFloat16 *state, const float *x, const float *row_scale, int rows,
                         int width, hipStream_t stream) {
    const size_t n = size_t(rows) * size_t(width);
    if (n) {
        residual_scaled_kernel<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(
            reinterpret_cast<uint16_t *>(state), x, row_scale, n, width);
        HIP_CHECK(hipGetLastError());
    }
}
void rope_quantize(BFloat16 *q, BFloat16 *k, const int *positions, int rows, int heads,
                   int kv_heads, float theta, uint8_t *q8, float *q_scale, hipStream_t stream) {
    if (rows < 0 || heads < 1 || kv_heads < 1 || !q || !k || !positions || !q8 || !q_scale)
        throw std::invalid_argument("Invalid rope/quantisation buffers");
    if (rows) {
        rope_quantize_kernel<<<dim3(unsigned(rows), unsigned(heads + kv_heads)), 128, 0, stream>>>(
            reinterpret_cast<uint16_t *>(q), reinterpret_cast<uint16_t *>(k), positions, heads,
            kv_heads, theta, q8, q_scale);
        HIP_CHECK(hipGetLastError());
    }
}
void residual_add(BFloat16 *state, const BFloat16 *x, size_t n, hipStream_t stream) {
    if (n) {
        residual_kernel<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(
            reinterpret_cast<uint16_t *>(state), reinterpret_cast<const uint16_t *>(x), n);
        HIP_CHECK(hipGetLastError());
    }
}
void silu_multiply(const BFloat16 *gate, const BFloat16 *up, BFloat16 *y, size_t n,
                   hipStream_t stream) {
    if (n) {
        silu_kernel<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(
            reinterpret_cast<const uint16_t *>(gate), reinterpret_cast<const uint16_t *>(up),
            reinterpret_cast<uint16_t *>(y), n);
        HIP_CHECK(hipGetLastError());
    }
}
void rope(BFloat16 *x, const int *pos, int rows, int heads, float theta, hipStream_t stream) {
    size_t n = size_t(rows) * size_t(heads) * 64;
    if (n) {
        rope_kernel<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(reinterpret_cast<uint16_t *>(x),
                                                                    pos, rows, heads, theta);
        HIP_CHECK(hipGetLastError());
    }
}
void embed(const BFloat16 *w, const int *tokens, BFloat16 *y, int rows, int width,
           hipStream_t stream) {
    size_t n = size_t(rows) * size_t(width);
    if (n) {
        embed_kernel<<<unsigned(ceil_div(n, 256)), 256, 0, stream>>>(
            reinterpret_cast<const uint16_t *>(w), tokens, reinterpret_cast<uint16_t *>(y), n,
            width);
        HIP_CHECK(hipGetLastError());
    }
}
void select_rows(const BFloat16 *x, const int *rows, BFloat16 *y, int count, int width,
                 hipStream_t stream) {
    embed(x, rows, y, count, width, stream);
}
} // namespace behemoth
