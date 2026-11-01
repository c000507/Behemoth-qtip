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
    // 16 consecutive inputs per thread per 4096-wide step: one 128-bit weight load, two 128-bit
    // activation loads per row.
    for (int d = tid * 16; d < n; d += 4096) {
        const uint4 packed = *reinterpret_cast<const uint4 *>(wf + d);
        float wv[16];
        const uint32_t pw[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            wv[4 * q] = device::fp8(uint8_t(pw[q]));
            wv[4 * q + 1] = device::fp8(uint8_t(pw[q] >> 8));
            wv[4 * q + 2] = device::fp8(uint8_t(pw[q] >> 16));
            wv[4 * q + 3] = device::fp8(uint8_t(pw[q] >> 24));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < live) {
                const uint4 *xp =
                    reinterpret_cast<const uint4 *>(x + size_t(row0 + r) * size_t(ldx) + d);
                const uint4 xa = xp[0], xb = xp[1];
                const uint32_t px[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    acc[r] = fmaf(device::bf16(uint16_t(px[q])), wv[2 * q], acc[r]);
                    acc[r] = fmaf(device::bf16(uint16_t(px[q] >> 16)), wv[2 * q + 1], acc[r]);
                }
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
// Decode-sized K and V in one launch (perf/single-decode): one wave per output feature of
// either shard (features [0, k.rows) are K, the rest V), 8 waves per block. A lane owns 16
// consecutive inputs of every 512-wide step; the 24 steps of n = 12288 are issued in batches of
// 8 independent 128-bit loads, so every wave keeps 4 KiB of weight reads in flight (the previous
// kernel's 512 blocks x 3 dependent steps streamed the 12 MB of K/V weights at ~200 GB/s).
// Numerics as fp8_gemv_rows: fp32 fma over the inputs in ascending order per lane, wave sum,
// bf16(value * scale).
constexpr int pair_waves = 8;
template <int R>
__global__ __launch_bounds__(pair_waves * 32) void fp8_gemv_pair(
    const uint8_t *__restrict__ wk, const float *__restrict__ sk, int k_rows,
    const uint8_t *__restrict__ wv, const float *__restrict__ sv, int v_rows, int n,
    const uint16_t *__restrict__ x, int ldx, int rows, uint16_t *__restrict__ yk,
    uint16_t *__restrict__ yv, int ldy) {
    const int lane = int(threadIdx.x) & 31,
              feature = __builtin_amdgcn_readfirstlane(int(blockIdx.x) * pair_waves + int(threadIdx.x >> 5));
    if (feature >= k_rows + v_rows)
        return;
    const bool is_k = feature < k_rows;
    const int f = is_k ? feature : feature - k_rows;
    const uint8_t *wf = (is_k ? wk : wv) + size_t(f) * size_t(n);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        acc[r] = 0.f;
    for (int d0 = lane * 16; d0 < n; d0 += 8 * 512) {
        uint4 packed[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = d0 + i * 512;
            packed[i] = d < n ? *reinterpret_cast<const uint4 *>(wf + d) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = d0 + i * 512;
            if (d >= n)
                break;
            float wv16[16];
            const uint32_t pw[4] = {packed[i].x, packed[i].y, packed[i].z, packed[i].w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                wv16[4 * q] = device::fp8(uint8_t(pw[q]));
                wv16[4 * q + 1] = device::fp8(uint8_t(pw[q] >> 8));
                wv16[4 * q + 2] = device::fp8(uint8_t(pw[q] >> 16));
                wv16[4 * q + 3] = device::fp8(uint8_t(pw[q] >> 24));
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < rows) {
                    const uint4 *xp = reinterpret_cast<const uint4 *>(x + size_t(r) * size_t(ldx) + d);
                    const uint4 xa = xp[0], xb = xp[1];
                    const uint32_t px[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        acc[r] = fmaf(device::bf16(uint16_t(px[q])), wv16[2 * q], acc[r]);
                        acc[r] = fmaf(device::bf16(uint16_t(px[q] >> 16)), wv16[2 * q + 1], acc[r]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        for (int s = 16; s; s /= 2)
            acc[r] += __shfl_down(acc[r], s);
    if (lane == 0) {
        const float scale = is_k ? sk[f] : sv[f];
        uint16_t *y = is_k ? yk : yv;
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (r < rows)
                y[size_t(r) * size_t(ldy) + size_t(f)] = device::bf16(acc[r] * scale);
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
// Prefill-sized K/V (perf/prefill): exact E4M3 -> BF16 expansion of the stacked K|V shard
// (E4M3 values are representable in BF16), 16 weights per thread.
__global__ __launch_bounds__(256) void fp8_expand(const uint8_t *__restrict__ w, size_t n16,
                                                  uint16_t *__restrict__ out) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n16)
        return;
    const uint4 b = reinterpret_cast<const uint4 *>(w)[i];
    const uint32_t words[4] = {b.x, b.y, b.z, b.w};
    uint32_t packed[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t word = words[j / 2], shift = 16 * (j % 2);
        const uint16_t lo = device::bf16(device::fp8(uint8_t(word >> shift)));
        const uint16_t hi = device::bf16(device::fp8(uint8_t(word >> (shift + 8))));
        packed[j] = uint32_t(lo) | (uint32_t(hi) << 16);
    }
    auto *o = reinterpret_cast<uint4 *>(out) + 2 * i;
    o[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    o[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}
// y = bf16(acc * scale[feature]) for the stacked K|V GEMM output acc[rows][k_rows + v_rows]
// (the rounding of the GEMV kernels: one per output, after the fp32 scale).
__global__ __launch_bounds__(256) void fp8_scale_split(const float *__restrict__ acc, int rows,
                                                       const float *__restrict__ sk, int k_rows,
                                                       const float *__restrict__ sv, int v_rows,
                                                       uint16_t *__restrict__ yk, uint16_t *__restrict__ yv,
                                                       int ldy) {
    const int features = k_rows + v_rows;
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(rows) * size_t(features))
        return;
    const int r = int(i / size_t(features)), f = int(i % size_t(features));
    if (f < k_rows)
        yk[size_t(r) * ldy + f] = device::bf16(acc[i] * sk[f]);
    else
        yv[size_t(r) * ldy + (f - k_rows)] = device::bf16(acc[i] * sv[f - k_rows]);
}
} // namespace
bool fp8_pair_dense_eligible(const GpuFp8Linear &k, const GpuFp8Linear &v) {
    return k.in == v.in && k.in % 16 == 0 && k.rows > 0 && v.rows > 0 &&
           k.weight.size() == size_t(k.in) * size_t(k.rows) && v.weight.size() == size_t(v.in) * size_t(v.rows) &&
           k.scale.size() == size_t(k.rows) && v.scale.size() == size_t(v.rows);
}
void fp8_expand_pair(const GpuFp8Linear &k, const GpuFp8Linear &v, BFloat16 *w, hipStream_t s) {
    if (!fp8_pair_dense_eligible(k, v))
        throw std::invalid_argument("Invalid FP8 K/V pair expansion");
    auto expand = [&](const GpuFp8Linear &l, BFloat16 *out) {
        const size_t n16 = l.weight.size() / 16;
        fp8_expand<<<unsigned((n16 + 255) / 256), 256, 0, s>>>(l.weight.data(), n16,
                                                              reinterpret_cast<uint16_t *>(out));
    };
    expand(k, w);
    expand(v, w + size_t(k.rows) * size_t(k.in));
    HIP_CHECK(hipGetLastError());
}
void fp8_scale_pair(const float *acc, int rows, const GpuFp8Linear &k, const GpuFp8Linear &v,
                    BFloat16 *yk, BFloat16 *yv, int ldy, hipStream_t s) {
    if (rows < 1 || ldy < k.rows || ldy < v.rows)
        throw std::invalid_argument("Invalid FP8 K/V scale call");
    const size_t n = size_t(rows) * size_t(k.rows + v.rows);
    fp8_scale_split<<<unsigned((n + 255) / 256), 256, 0, s>>>(
        acc, rows, k.scale.data(), k.rows, v.scale.data(), v.rows, reinterpret_cast<uint16_t *>(yk),
        reinterpret_cast<uint16_t *>(yv), ldy);
    HIP_CHECK(hipGetLastError());
}
void fp8_linear(const GpuFp8Linear &w, const BFloat16 *x, int ldx, int rows,
                BFloat16 *y, int ldy, hipStream_t s) {
    if (w.in <= 0 || w.rows <= 0 || rows <= 0 || ldx < w.in || ldy < w.rows ||
        w.weight.size() != size_t(w.in) * size_t(w.rows) || w.scale.size() != size_t(w.rows))
        throw std::invalid_argument("Invalid FP8 projection call");
    const auto *xs = reinterpret_cast<const uint16_t *>(x);
    auto *ys = reinterpret_cast<uint16_t *>(y);
    // Row-batched GEMV: 128-bit loads need n % 16 == 0 and 16-byte aligned activation rows.
    const bool batched = !legacy_dispatch() && rows <= 64 && w.in % 16 == 0 && ldx % 8 == 0 &&
                         (reinterpret_cast<uintptr_t>(x) & 15) == 0;
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
bool fp8_linear_pair(const GpuFp8Linear &k, const GpuFp8Linear &v, const BFloat16 *x, int ldx,
                     int rows, BFloat16 *yk, BFloat16 *yv, int ldy, hipStream_t s) {
    static const bool enabled = [] {
        const char *e = std::getenv("BEHEMOTH_FP8_KV_PAIR");
        return !e || std::atoi(e) != 0;
    }();
    if (!enabled || legacy_dispatch() || rows < 1 || rows > 4 || k.in != v.in || k.in % 16 ||
        ldx % 8 || (reinterpret_cast<uintptr_t>(x) & 15) || ldy < k.rows || ldy < v.rows ||
        k.weight.size() != size_t(k.in) * size_t(k.rows) || v.weight.size() != size_t(v.in) * size_t(v.rows) ||
        k.scale.size() != size_t(k.rows) || v.scale.size() != size_t(v.rows))
        return false;
    const int features = k.rows + v.rows;
    const dim3 grid((features + pair_waves - 1) / pair_waves);
    auto launch = [&](auto kernel) {
        kernel<<<grid, pair_waves * 32, 0, s>>>(
            k.weight.data(), k.scale.data(), k.rows, v.weight.data(), v.scale.data(), v.rows, k.in,
            reinterpret_cast<const uint16_t *>(x), ldx, rows, reinterpret_cast<uint16_t *>(yk),
            reinterpret_cast<uint16_t *>(yv), ldy);
    };
    if (rows == 1)
        launch(fp8_gemv_pair<1>);
    else if (rows == 2)
        launch(fp8_gemv_pair<2>);
    else
        launch(fp8_gemv_pair<4>);
    HIP_CHECK(hipGetLastError());
    return true;
}
} // namespace behemoth::qtip
