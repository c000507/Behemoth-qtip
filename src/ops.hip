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
