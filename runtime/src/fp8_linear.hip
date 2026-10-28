// Mixed QTIP/FP8 projections. W8A16: E4M3 weights, BF16 activations, FP32 sums.
#include "behemoth/qtip_gpu.hpp"
#include "device_numeric.hpp"
namespace behemoth::qtip {
namespace {
using Reg = __bf16 __attribute__((ext_vector_type(8)));
using Acc = float __attribute__((ext_vector_type(8)));
__device__ int at(int r, int d) { return r * 128 + (d ^ ((r & 15) * 8)); }
__device__ Reg load(const uint16_t *p, int row, int d) {
    return __builtin_bit_cast(Reg, *reinterpret_cast<const uint4 *>(p + at(row, d)));
}
__global__ void fp8_gemv(const uint8_t *w, const float *scale, int n, const uint16_t *x,
                          int ldx, uint16_t *y, int ldy) {
    __shared__ float sums[8];
    int feature = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    float value = 0;
    for (int d = tid; d < n; d += 256)
        value = fmaf(device::bf16(x[size_t(row) * ldx + d]),
                      device::fp8(w[size_t(feature) * n + d]), value);
    for (int s = 16; s; s /= 2) value += __shfl_down(value, s);
    if (tid % 32 == 0) sums[tid / 32] = value;
    __syncthreads();
    if (tid == 0) {
        value = 0;
        for (int i = 0; i < 8; ++i) value += sums[i];
        y[size_t(row) * ldy + feature] = device::bf16(value * scale[feature]);
    }
}
__global__ void fp8_gemm(const uint8_t *w, const float *scale, int n, int out, const uint16_t *x,
                          int ldx, int rows, uint16_t *y, int ldy) {
    __shared__ __align__(16) uint16_t input[16 * 128], weight[64 * 128];
    int tid = threadIdx.x, wave = tid / 32, c = tid % 16, h = tid % 32 / 16;
    int first = blockIdx.x * 16, feature0 = blockIdx.y * 64;
    Acc acc{};
    for (int base = 0; base < n; base += 128) {
        for (int i = tid; i < 16 * 128; i += 128) {
            int r = i / 128, d = i % 128;
            input[at(r, d)] = first + r < rows && base + d < n ? x[size_t(first + r) * ldx + base + d] : 0;
        }
        for (int i = tid; i < 64 * 128; i += 128) {
            int r = i / 128, d = i % 128;
            weight[at(r, d)] = feature0 + r < out && base + d < n
                ? device::bf16(device::fp8(w[size_t(feature0 + r) * n + base + d])) : 0;
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int d = 32 * (s / 2) + 16 * h + 8 * (s % 2);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                load(weight, wave * 16 + c, d), load(input, c, d), acc);
        }
        __syncthreads();
    }
    if (first + c < rows)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int feature = feature0 + wave * 16 + 8 * h + i;
            if (feature < out)
                y[size_t(first + c) * ldy + feature] = device::bf16(acc[i] * scale[feature]);
        }
}
} // namespace
void fp8_linear(const GpuFp8Linear &w, const BFloat16 *x, int ldx, int rows,
                BFloat16 *y, int ldy, hipStream_t s) {
    if (w.in <= 0 || w.rows <= 0 || rows <= 0 || ldx < w.in || ldy < w.rows ||
        w.weight.size() != size_t(w.in) * size_t(w.rows) || w.scale.size() != size_t(w.rows))
        throw std::invalid_argument("Invalid FP8 projection call");
    if (rows <= 4)
        fp8_gemv<<<dim3(w.rows, rows), 256, 0, s>>>(w.weight.data(), w.scale.data(), w.in,
            reinterpret_cast<const uint16_t *>(x), ldx, reinterpret_cast<uint16_t *>(y), ldy);
    else
        fp8_gemm<<<dim3((rows + 15) / 16, (w.rows + 63) / 64), 128, 0, s>>>(w.weight.data(), w.scale.data(),
            w.in, w.rows, reinterpret_cast<const uint16_t *>(x), ldx, rows, reinterpret_cast<uint16_t *>(y), ldy);
    HIP_CHECK(hipGetLastError());
}
} // namespace behemoth::qtip
