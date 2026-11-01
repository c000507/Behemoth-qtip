// Mixed QTIP/FP8 projections. W8A16: E4M3 weights, BF16 activations, FP32 sums.
#include "behemoth/qtip_gpu.hpp"
#include "device_numeric.hpp"
#include <cstdlib>
#include <cstring>
namespace behemoth::qtip {
namespace {
using Reg = __bf16 __attribute__((ext_vector_type(8)));
using Acc = float __attribute__((ext_vector_type(8)));
__device__ int at(int r, int d) { return r * 128 + (d ^ ((r & 15) * 8)); }
__device__ Reg load(const uint16_t *p, int row, int d) {
    return __builtin_bit_cast(Reg, *reinterpret_cast<const uint4 *>(p + at(row, d)));
}
// Legacy single-row GEMV: one block per (feature, row), one byte of weight per thread per step.
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
// Row-batched GEMV for decode-sized steps (rows <= 64): one block per (feature, group of R
// rows). Each thread owns 4 consecutive inputs of every 1024-wide step: one 32-bit load of four
// FP8 weights, then R 64-bit loads of four BF16 activations. The weight shard is streamed from
// HBM exactly once per row group (R rows share every weight byte); the activations (rows x n
// BF16, <= 1.5 MB) stay in L2. Replaces the old rows > 4 path, a 16x64-tile WMMA GEMM whose grid
// held only out/64 blocks (8 for a 512-row K or V shard): 260 ms per step of fp8 K/V at 8 rows
// versus 9 ms at 1 row (perf/batch-decode).
template <int R>
__global__ __launch_bounds__(256) void fp8_gemv_rows(const uint8_t *__restrict__ w,
                                                     const float *__restrict__ scale, int n,
                                                     const uint16_t *__restrict__ x, int ldx,
                                                     int rows, uint16_t *__restrict__ y, int ldy) {
    __shared__ float sums[8][R];
    const int feature = blockIdx.x, row0 = int(blockIdx.y) * R, tid = threadIdx.x;
    const uint8_t *wf = w + size_t(feature) * size_t(n);
    const int live = min(R, rows - row0);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        acc[r] = 0.f;
    for (int d = tid * 4; d < n; d += 1024) {
        const uint32_t packed = *reinterpret_cast<const uint32_t *>(wf + d);
        const float w0 = device::fp8(uint8_t(packed)), w1 = device::fp8(uint8_t(packed >> 8)),
                    w2 = device::fp8(uint8_t(packed >> 16)), w3 = device::fp8(uint8_t(packed >> 24));
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < live) {
                const uint2 xv =
                    *reinterpret_cast<const uint2 *>(x + size_t(row0 + r) * size_t(ldx) + d);
                acc[r] = fmaf(device::bf16(uint16_t(xv.x)), w0, acc[r]);
                acc[r] = fmaf(device::bf16(uint16_t(xv.x >> 16)), w1, acc[r]);
                acc[r] = fmaf(device::bf16(uint16_t(xv.y)), w2, acc[r]);
                acc[r] = fmaf(device::bf16(uint16_t(xv.y >> 16)), w3, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        for (int s = 16; s; s /= 2)
            acc[r] += __shfl_down(acc[r], s);
    if (tid % 32 == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
            sums[tid / 32][r] = acc[r];
    __syncthreads();
    if (tid < live) {
        float value = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            value += sums[i][tid];
        y[size_t(row0 + tid) * size_t(ldy) + feature] = device::bf16(value * scale[feature]);
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
// BEHEMOTH_FP8_KV_KERNEL=legacy restores the per-row GEMV (rows <= 4) + WMMA GEMM dispatch.
bool legacy_dispatch() {
    static const bool legacy = [] {
        const char *v = std::getenv("BEHEMOTH_FP8_KV_KERNEL");
        return v && std::strcmp(v, "legacy") == 0;
    }();
    return legacy;
}
} // namespace
void fp8_linear(const GpuFp8Linear &w, const BFloat16 *x, int ldx, int rows,
                BFloat16 *y, int ldy, hipStream_t s) {
    if (w.in <= 0 || w.rows <= 0 || rows <= 0 || ldx < w.in || ldy < w.rows ||
        w.weight.size() != size_t(w.in) * size_t(w.rows) || w.scale.size() != size_t(w.rows))
        throw std::invalid_argument("Invalid FP8 projection call");
    const auto *xs = reinterpret_cast<const uint16_t *>(x);
    auto *ys = reinterpret_cast<uint16_t *>(y);
    // Row-batched GEMV: 4-wide loads need n % 4 == 0 and 8-byte aligned activation rows.
    const bool batched = !legacy_dispatch() && rows <= 64 && w.in % 4 == 0 && ldx % 4 == 0 &&
                         (reinterpret_cast<uintptr_t>(x) & 7) == 0;
    if (batched) {
        auto launch = [&](auto kernel, int r) {
            kernel<<<dim3(w.rows, (rows + r - 1) / r), 256, 0, s>>>(
                w.weight.data(), w.scale.data(), w.in, xs, ldx, rows, ys, ldy);
        };
        if (rows <= 1)
            launch(fp8_gemv_rows<1>, 1);
        else if (rows <= 2)
            launch(fp8_gemv_rows<2>, 2);
        else if (rows <= 4)
            launch(fp8_gemv_rows<4>, 4);
        else if (rows <= 8)
            launch(fp8_gemv_rows<8>, 8);
        else
            launch(fp8_gemv_rows<16>, 16);
    } else if (rows <= 4)
        fp8_gemv<<<dim3(w.rows, rows), 256, 0, s>>>(w.weight.data(), w.scale.data(), w.in, xs, ldx,
                                                    ys, ldy);
    else
        fp8_gemm<<<dim3((rows + 15) / 16, (w.rows + 63) / 64), 128, 0, s>>>(
            w.weight.data(), w.scale.data(), w.in, w.rows, xs, ldx, rows, ys, ldy);
    HIP_CHECK(hipGetLastError());
}
} // namespace behemoth::qtip
