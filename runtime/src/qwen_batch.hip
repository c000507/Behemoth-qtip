// Qwen3.5 batched-serving kernels: W8A8 FP8 WMMA GEMMs (gfx12 v_wmma_f32_16x16x16_fp8_fp8) with
// dynamic per-token x 128-column activation scales and the checkpoint's 128x128 weight block scales.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "behemoth/gpu.hpp"
#include "behemoth/qwen_batch.hpp"

namespace behemoth {
namespace {
__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
__device__ __forceinline__ v8f wmma8(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, v8f c) {
    const v2i a = {int(a0), int(a1)}, b = {int(b0), int(b1)};
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
// 4 floats -> 4 OCP e4m3 bytes (values already within +-448)
__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c, float d) {
    int r = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    r = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, r, true);
    return uint32_t(r);
}

// one wave per (row, 128-column block); lane = 4 consecutive columns
template <class T>
__device__ __forceinline__ void load4(const T *p, float *v);
template <>
__device__ __forceinline__ void load4<uint16_t>(const uint16_t *p, float *v) {
    const uint2 u = *reinterpret_cast<const uint2 *>(p);
    v[0] = bf2f(uint16_t(u.x));
    v[1] = bf2f(uint16_t(u.x >> 16));
    v[2] = bf2f(uint16_t(u.y));
    v[3] = bf2f(uint16_t(u.y >> 16));
}
template <>
__device__ __forceinline__ void load4<float>(const float *p, float *v) {
    const float4 u = *reinterpret_cast<const float4 *>(p);
    v[0] = u.x;
    v[1] = u.y;
    v[2] = u.z;
    v[3] = u.w;
}
template <class T>
__global__ __launch_bounds__(256) void act_quant_kernel(const T *__restrict__ x, int ldx, int rows, int n,
                                                        uint8_t *__restrict__ q, float *__restrict__ s) {
    const int nb = n / 128;
    const long gid = long(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (gid >= long(rows) * nb) return;
    const int r = int(gid / nb), b = int(gid % nb);
    float v[4];
    load4<T>(x + long(r) * ldx + b * 128 + lane * 4, v);
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor(am, off, 32));
    const float sc = am > 0.f ? am / 448.0f : 1.0f;
    float o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = fminf(448.f, fmaxf(-448.f, v[i] / sc));
    *reinterpret_cast<uint32_t *>(q + long(r) * n + b * 128 + lane * 4) = pack_fp8x4(o[0], o[1], o[2], o[3]);
    if (lane == 0) s[long(r) * nb + b] = sc;
}

// ---------------------------------------------------------------- decode-sized W8A8 GEMM
// Block = 8 waves = 128 output features (wave: 16 features, A operand = weights straight from global
// into registers), all 16*MT tokens of the K window staged in LDS (B operand). K split over
// gridDim.y; split s writes raw sums of its K blocks to y + s * split_stride.
// WMMA operand layout (gfx12 wave32): lane l -> A row (feature) l%16 / B column (token) l%16; k
// slots 8*(l/16)..+8. For the k-step pair j of a 128-column block lane half h holds the 16 bytes
// [32j + 16h, +16): the first 8 feed the even step, the last 8 the odd step (same k mapping for A
// and B). D: lane holds features 8*(l/16)+i (i < 8) of token l%16.
constexpr int XS_PAD = 144;  // LDS row stride (bytes) of 128-byte token rows
template <int MT, bool G>
__global__ __launch_bounds__(256) void w8a8_small_kernel(const uint8_t *__restrict__ w, const float *__restrict__ ws,
                                                         int m, int n, const uint8_t *__restrict__ x,
                                                         const float *__restrict__ xs, int rows,
                                                         float *__restrict__ y, int ldy, long split_stride,
                                                         int kbps) {
    __shared__ __align__(16) uint8_t xt[16 * MT][XS_PAD];
    __shared__ float xss[16 * MT];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, c = lane & 15, h = lane >> 4;
    const int nkb = n / 128;
    const int f0 = int(blockIdx.x) * 128 + wave * 16;
    const int kb0 = int(blockIdx.y) * kbps, kb1 = min(nkb, kb0 + kbps);
    const bool active = f0 < m;
    const uint8_t *wr = w + size_t(active ? f0 + c : 0) * n + 16 * h;
    const float *wsr = ws + size_t(active ? f0 / 128 : 0) * nkb;
    v8f acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 wc[4], wn[4];
    const int kl = kb1 - 1;
#pragma unroll
    for (int j = 0; j < 4; ++j) wc[j] = *reinterpret_cast<const uint4 *>(wr + size_t(kb0) * 128 + 32 * j);
#pragma unroll
    for (int j = 0; j < 4; ++j) wn[j] = *reinterpret_cast<const uint4 *>(wr + size_t(min(kb0 + 1, kl)) * 128 + 32 * j);
    for (int kb = kb0; kb < kb1; ++kb) {
        if (!G) {
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * MT * 8; i += 256) {
            const int r = i >> 3, qq = i & 7;
            *reinterpret_cast<uint4 *>(&xt[r][qq * 16]) =
                *reinterpret_cast<const uint4 *>(x + size_t(r) * n + size_t(kb) * 128 + qq * 16);
        }
        if (threadIdx.x < 16 * MT) xss[threadIdx.x] = xs[size_t(threadIdx.x) * nkb + kb];
        __syncthreads();
        }
        uint4 wn2[4];  // two K blocks of weight loads in flight
        const int kn = min(kb + 2, kl);
#pragma unroll
        for (int j = 0; j < 4; ++j) wn2[j] = *reinterpret_cast<const uint4 *>(wr + size_t(kn) * 128 + 32 * j);
        const float sw = wsr[kb];
        v8f tmp[MT];
#pragma unroll
        for (int t = 0; t < MT; ++t) tmp[t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                const uint4 b = G ? *reinterpret_cast<const uint4 *>(x + size_t(16 * t + c) * n + size_t(kb) * 128 + 32 * j + 16 * h)
                                  : *reinterpret_cast<const uint4 *>(&xt[16 * t + c][32 * j + 16 * h]);
                tmp[t] = wmma8(wc[j].x, wc[j].y, b.x, b.y, tmp[t]);
                tmp[t] = wmma8(wc[j].z, wc[j].w, b.z, b.w, tmp[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            const float sc = sw * (G ? xs[size_t(16 * t + c) * nkb + kb] : xss[16 * t + c]);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][i] = fmaf(tmp[t][i], sc, acc[t][i]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wc[j] = wn[j];
            wn[j] = wn2[j];
        }
    }
    if (!active) return;
    float *yo = y + long(blockIdx.y) * split_stride;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int tok = 16 * t + c;
        if (tok < rows) {
            float4 *p = reinterpret_cast<float4 *>(yo + size_t(tok) * ldy + f0 + 8 * h);
            p[0] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
            p[1] = make_float4(acc[t][4], acc[t][5], acc[t][6], acc[t][7]);
        }
    }
}
// y[r][j] = sum_s partial[s][r][j] (fixed order)
__global__ void splitk_reduce_kernel(const float *__restrict__ p, int splits, int rows, int m, long sstride,
                                     float *__restrict__ y, int ldy) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= long(rows) * m / 4) return;
    const long e = i * 4;
    const int r = int(e / m), j = int(e % m);
    float4 a = *reinterpret_cast<const float4 *>(p + size_t(r) * m + j);
    for (int s = 1; s < splits; ++s) {
        const float4 b = *reinterpret_cast<const float4 *>(p + s * sstride + size_t(r) * m + j);
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        a.w += b.w;
    }
    *reinterpret_cast<float4 *>(y + size_t(r) * ldy + j) = a;
}

// ---------------------------------------------------------------- prefill-sized W8A8 GEMM
// Block tile 128 features x 128 tokens, 8 waves as 4 (32 features) x 2 (64 tokens); both operands
// staged in LDS per 128-column K block (one weight scale per block tile and K block).
template <int TT>
__global__ __launch_bounds__(256) void w8a8_big_kernel(const uint8_t *__restrict__ w, const float *__restrict__ ws,
                                                       int m, int n, const uint8_t *__restrict__ x,
                                                       const float *__restrict__ xs, int rows,
                                                       float *__restrict__ y, int ldy) {
    __shared__ __align__(16) uint8_t wt[128][XS_PAD], xt[32 * TT][XS_PAD];
    __shared__ float xss[32 * TT];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, c = lane & 15, h = lane >> 4;
    const int wf = wave & 3, wk = wave >> 2;
    const int nkb = n / 128;
    const int fb = int(blockIdx.y) * 128, tb = int(blockIdx.x) * 32 * TT;  // token blocks fastest: weight tile reuse in L2
    v8f acc[2][TT];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[a][t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    // global -> register staging: 4 x 16 B of weights and of tokens per thread per K block
    uint4 rw[4], rx[TT];
#define GLOAD(kb)                                                                                                  \
    _Pragma("unroll") for (int p = 0; p < 4; ++p) {                                                                \
        const int i = threadIdx.x + 256 * p, r = i >> 3, qq = i & 7;                                               \
        rw[p] = *reinterpret_cast<const uint4 *>(w + size_t(fb + r) * n + size_t(kb) * 128 + qq * 16);             \
        if (p < TT) {                                                                                              \
            const int tr = min(tb + r, rows - 1);                                                                  \
            rx[p] = *reinterpret_cast<const uint4 *>(x + size_t(tr) * n + size_t(kb) * 128 + qq * 16);             \
        }                                                                                                          \
    }
    GLOAD(0);
    const float *wsr = ws + size_t(fb / 128) * nkb;
    for (int kb = 0; kb < nkb; ++kb) {
        __syncthreads();
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int i = threadIdx.x + 256 * p, r = i >> 3, qq = i & 7;
            *reinterpret_cast<uint4 *>(&wt[r][qq * 16]) = rw[p];
            if (p < TT) *reinterpret_cast<uint4 *>(&xt[r][qq * 16]) = rx[p];
        }
        if (threadIdx.x < 32 * TT) xss[threadIdx.x] = xs[size_t(min(tb + int(threadIdx.x), rows - 1)) * nkb + kb];
        __syncthreads();
        if (kb + 1 < nkb) { GLOAD(kb + 1); }
        const float sw = wsr[kb];
        v8f tmp[2][TT];
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int t = 0; t < TT; ++t) tmp[a][t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint4 av[2], bv[TT];
#pragma unroll
            for (int a = 0; a < 2; ++a) av[a] = *reinterpret_cast<const uint4 *>(&wt[wf * 32 + 16 * a + c][32 * j + 16 * h]);
#pragma unroll
            for (int t = 0; t < TT; ++t) bv[t] = *reinterpret_cast<const uint4 *>(&xt[wk * 16 * TT + 16 * t + c][32 * j + 16 * h]);
#pragma unroll
            for (int a = 0; a < 2; ++a)
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    tmp[a][t] = wmma8(av[a].x, av[a].y, bv[t].x, bv[t].y, tmp[a][t]);
                    tmp[a][t] = wmma8(av[a].z, av[a].w, bv[t].z, bv[t].w, tmp[a][t]);
                }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float sc = sw * xss[wk * 16 * TT + 16 * t + c];
#pragma unroll
            for (int a = 0; a < 2; ++a)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[a][t][i] = fmaf(tmp[a][t][i], sc, acc[a][t][i]);
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        const int tok = tb + wk * 16 * TT + 16 * t + c;
        if (tok >= rows) continue;
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            float4 *p = reinterpret_cast<float4 *>(y + size_t(tok) * ldy + fb + wf * 32 + 16 * a + 8 * h);
            p[0] = make_float4(acc[a][t][0], acc[a][t][1], acc[a][t][2], acc[a][t][3]);
            p[1] = make_float4(acc[a][t][4], acc[a][t][5], acc[a][t][6], acc[a][t][7]);
        }
    }
}
#undef GLOAD

// ---------------------------------------------------------------- bf16 GEMV, any row count
// grid (row groups of 8, feature groups of 8): consecutive blocks share a weight tile (L2 reuse).
__global__ __launch_bounds__(256) void bf16_gemv_any_kernel(const uint16_t *__restrict__ w, int m, int n,
                                                            const uint16_t *__restrict__ x, int ldx, int rows,
                                                            float *__restrict__ y, int ldy) {
    constexpr int R = 8;
    const int lane = threadIdx.x & 31;
    const int j = int(blockIdx.y) * 8 + (threadIdx.x >> 5);
    const int row0 = int(blockIdx.x) * R;
    if (j >= m) return;
    const int live = min(R, rows - row0);
    const uint16_t *wr = w + size_t(j) * n;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int d = lane * 8; d < n; d += 256) {
        const uint4 pk = *reinterpret_cast<const uint4 *>(wr + d);
        const uint32_t pw[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < live) {
                const uint4 xa = *reinterpret_cast<const uint4 *>(x + size_t(row0 + r) * ldx + d);
                const uint32_t px[4] = {xa.x, xa.y, xa.z, xa.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    acc[r] = fmaf(bf2f(uint16_t(px[q])), bf2f(uint16_t(pw[q])), acc[r]);
                    acc[r] = fmaf(bf2f(uint16_t(px[q] >> 16)), bf2f(uint16_t(pw[q] >> 16)), acc[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (lane == 0 && r < live) y[size_t(row0 + r) * ldy + j] = v;
    }
}

// 128x128 block quantisation of bf16 weights: one block (256 threads) per (row block, col block)
__global__ __launch_bounds__(256) void quant_weight_kernel(const uint16_t *__restrict__ w, int m, int n,
                                                           uint8_t *__restrict__ w8, float *__restrict__ ws) {
    const int rb = blockIdx.y, cb = blockIdx.x, nb = n / 128;
    __shared__ float red[8];
    float am = 0.f;
    for (int i = threadIdx.x; i < 128 * 128; i += 256) {
        const int r = rb * 128 + i / 128, cc = cb * 128 + i % 128;
        if (r < m) am = fmaxf(am, fabsf(bf2f(w[size_t(r) * n + cc])));
    }
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor(am, off, 32));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    am = 0.f;
    for (int i = 0; i < 8; ++i) am = fmaxf(am, red[i]);
    const float sc = am > 0.f ? am / 448.f : 1.f;
    for (int i = threadIdx.x * 4; i < 128 * 128; i += 1024) {
        const int r = rb * 128 + i / 128, cc = cb * 128 + i % 128;
        if (r >= m) continue;
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) v[k] = fminf(448.f, fmaxf(-448.f, bf2f(w[size_t(r) * n + cc + k]) / sc));
        *reinterpret_cast<uint32_t *>(w8 + size_t(r) * n + cc) = pack_fp8x4(v[0], v[1], v[2], v[3]);
    }
    if (threadIdx.x == 0) ws[size_t(rb) * nb + cb] = sc;
}
}  // namespace

void q_act_quant(const uint16_t *x, int ldx, int rows, int n, uint8_t *q, float *s, hipStream_t st) {
    if (n % 128 || ldx % 4) throw std::runtime_error("q_act_quant: n % 128");
    const long waves = long(rows) * (n / 128);
    act_quant_kernel<uint16_t><<<unsigned((waves + 7) / 8), 256, 0, st>>>(x, ldx, rows, n, q, s);
    HIP_CHECK(hipGetLastError());
}
void q_act_quant_f32(const float *x, int ldx, int rows, int n, uint8_t *q, float *s, hipStream_t st) {
    if (n % 128 || ldx % 4) throw std::runtime_error("q_act_quant_f32: n % 128");
    const long waves = long(rows) * (n / 128);
    act_quant_kernel<float><<<unsigned((waves + 7) / 8), 256, 0, st>>>(x, ldx, rows, n, q, s);
    HIP_CHECK(hipGetLastError());
}

int q_w8a8_splits(int m, int n) {
    // aim for >= ~2048 waves streaming weights; each split keeps >= 2 K blocks
    const int fblocks = (m + 127) / 128, nkb = n / 128;
    static const int target = std::getenv("QWEN_SPLITW") ? std::atoi(std::getenv("QWEN_SPLITW")) : 256;
    int want = std::max(1, target / std::max(1, fblocks));
    want = std::min(want, std::max(1, nkb / 2));
    const int kbps = (nkb + want - 1) / want;
    return (nkb + kbps - 1) / kbps;
}
size_t q_w8a8_small_scratch(int m, int n, int rows) {
    const int s = q_w8a8_splits(m, n);
    return s > 1 ? size_t(s) * size_t(rows) * size_t(m) : 0;
}
void q_w8a8_small(const uint8_t *w, const float *ws, int m, int n, const uint8_t *x8, const float *xs, int rows,
                  float *y, int ldy, float *partial, hipStream_t st) {
    if (m % 16 || n % 128 || rows > 64 || rows < 1) throw std::runtime_error("q_w8a8_small: shape");
    const int nkb = n / 128, splits = q_w8a8_splits(m, n);
    const int kbps = (nkb + splits - 1) / splits;
    const dim3 grid(unsigned((m + 127) / 128), unsigned(splits));
    float *out = splits > 1 ? partial : y;
    const int ldo = splits > 1 ? m : ldy;
    const long ss = long(rows) * m;
    if (splits > 1 && !partial) throw std::runtime_error("q_w8a8_small: partial scratch required");
    const int mt = (rows + 15) / 16;
    // B operand straight from global (L1/L2) for <= 32 tokens, LDS-staged beyond (register pressure)
#define QW8(MT, G) w8a8_small_kernel<MT, G><<<grid, 256, 0, st>>>(w, ws, m, n, x8, xs, rows, out, ldo, ss, kbps)
    switch (mt) {
    case 1: QW8(1, true); break;
    case 2: QW8(2, true); break;
    case 3: QW8(3, false); break;
    default: QW8(4, false); break;
    }
#undef QW8
    HIP_CHECK(hipGetLastError());
    if (splits > 1) {
        const long n4 = long(rows) * m / 4;
        splitk_reduce_kernel<<<unsigned((n4 + 255) / 256), 256, 0, st>>>(partial, splits, rows, m, ss, y, ldy);
        HIP_CHECK(hipGetLastError());
    }
}
void q_w8a8_big(const uint8_t *w, const float *ws, int m, int n, const uint8_t *x8, const float *xs, int rows,
                float *y, int ldy, hipStream_t st) {
    if (m % 128 || n % 128) throw std::runtime_error("q_w8a8_big: m, n % 128");
    static const int tt = std::getenv("QWEN_BIG_TT") ? std::atoi(std::getenv("QWEN_BIG_TT")) : 4;
    if (tt == 2) {
        const dim3 grid(unsigned((rows + 63) / 64), unsigned(m / 128));
        w8a8_big_kernel<2><<<grid, 256, 0, st>>>(w, ws, m, n, x8, xs, rows, y, ldy);
    } else {
        const dim3 grid(unsigned((rows + 127) / 128), unsigned(m / 128));
        w8a8_big_kernel<4><<<grid, 256, 0, st>>>(w, ws, m, n, x8, xs, rows, y, ldy);
    }
    HIP_CHECK(hipGetLastError());
}
void q_w8a8(const uint8_t *w, const float *ws, int m, int n, const uint8_t *x8, const float *xs, int rows,
            float *y, int ldy, float *partial, hipStream_t st) {
    if (rows <= 64) q_w8a8_small(w, ws, m, n, x8, xs, rows, y, ldy, partial, st);
    else q_w8a8_big(w, ws, m, n, x8, xs, rows, y, ldy, st);
}
void q_bf16_gemv_any(const uint16_t *w, int m, int n, const uint16_t *x, int ldx, int rows, float *y, int ldy,
                     hipStream_t st) {
    if (n % 256 || ldx % 8) throw std::runtime_error("q_bf16_gemv_any: n % 256");
    const dim3 grid(unsigned((rows + 7) / 8), unsigned((m + 7) / 8));
    bf16_gemv_any_kernel<<<grid, 256, 0, st>>>(w, m, n, x, ldx, rows, y, ldy);
    HIP_CHECK(hipGetLastError());
}
void q_quant_weight_bf16(const uint16_t *w, int m, int n, uint8_t *w8, float *ws, hipStream_t st) {
    if (n % 128) throw std::runtime_error("q_quant_weight_bf16: n % 128");
    quant_weight_kernel<<<dim3(unsigned(n / 128), unsigned((m + 127) / 128)), 256, 0, st>>>(w, m, n, w8, ws);
    HIP_CHECK(hipGetLastError());
}

}  // namespace behemoth

// ------------------------------------------------------------------ tensor-parallel all-reduce (2 ranks)
// Each rank pushes its row-parallel partial into a pinned host slot (zero-copy writes) and raises a
// host flag; the peer spins on the flag, then state = bf16(state + (p_rank0 + p_rank1)) (fixed order:
// both ranks compute identical bits). Exchange sequence numbers live in device memory, so the kernels
// can be captured in HIP graphs and need no host synchronisation. bf16 mode rounds BOTH partials to
// bf16 (the pushed one and the rank's own copy) to keep the ranks bit-identical.
namespace behemoth {
namespace {
__device__ __forceinline__ unsigned ld_flag(const unsigned *p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void last_block(unsigned *ctr, unsigned *flag, unsigned flag_val, unsigned *seq,
                                           unsigned seq_val) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = atomicAdd(ctr, 1u);
        if (done == gridDim.x - 1) {
            *ctr = 0;
            __threadfence_system();
            if (flag) __hip_atomic_store(flag, flag_val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            if (seq) *seq = seq_val;
        }
    }
}
__global__ void ar_push_kernel(float *__restrict__ part, long n4, char *hbase, long slot_bytes, int bf16,
                               unsigned *hflag, const unsigned *seq, unsigned *ctr, int nslot) {
    const unsigned s = *seq;
    const int slot = int(s % unsigned(nslot));
    char *dst = hbase + slot * slot_bytes;
    for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += long(gridDim.x) * blockDim.x) {
        float4 v = reinterpret_cast<float4 *>(part)[i];
        if (bf16) {
            const uint16_t a = f2bf(v.x), b = f2bf(v.y), c = f2bf(v.z), d = f2bf(v.w);
            reinterpret_cast<uint2 *>(dst)[i] = make_uint2(uint32_t(a) | (uint32_t(b) << 16), uint32_t(c) | (uint32_t(d) << 16));
            reinterpret_cast<float4 *>(part)[i] = make_float4(bf2f(a), bf2f(b), bf2f(c), bf2f(d));
        } else {
            reinterpret_cast<float4 *>(dst)[i] = v;
        }
    }
    last_block(ctr, hflag + slot * 16, s + 1, nullptr, 0);
}
__global__ void ar_pull_kernel(uint16_t *__restrict__ state, const float *__restrict__ own, const char *hbase,
                               long slot_bytes, int bf16, long n4, int rank, const unsigned *peer_flag,
                               unsigned *seq, unsigned *ctr, int nslot) {
    const unsigned s = *seq;
    const int slot = int(s % unsigned(nslot));
    if (peer_flag && threadIdx.x == 0)
        while (ld_flag(peer_flag + slot * 16) < s + 1) __builtin_amdgcn_s_sleep(2);
    __syncthreads();
    const char *src = hbase + slot * slot_bytes;
    for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += long(gridDim.x) * blockDim.x) {
        const float4 o = reinterpret_cast<const float4 *>(own)[i];
        float4 p;
        if (bf16) {
            const uint2 u = *reinterpret_cast<const uint2 *>(src + i * 8);
            p = make_float4(bf2f(uint16_t(u.x)), bf2f(uint16_t(u.x >> 16)), bf2f(uint16_t(u.y)), bf2f(uint16_t(u.y >> 16)));
        } else {
            p = *reinterpret_cast<const float4 *>(src + i * 16);
        }
        const float4 a = rank == 0 ? o : p, b = rank == 0 ? p : o;
        uint2 st = reinterpret_cast<uint2 *>(state)[i];
        const float r0 = bf2f(uint16_t(st.x)) + (a.x + b.x), r1 = bf2f(uint16_t(st.x >> 16)) + (a.y + b.y);
        const float r2 = bf2f(uint16_t(st.y)) + (a.z + b.z), r3 = bf2f(uint16_t(st.y >> 16)) + (a.w + b.w);
        st.x = uint32_t(f2bf(r0)) | (uint32_t(f2bf(r1)) << 16);
        st.y = uint32_t(f2bf(r2)) | (uint32_t(f2bf(r3)) << 16);
        reinterpret_cast<uint2 *>(state)[i] = st;
    }
    last_block(ctr, nullptr, 0, seq, s + 1);
}
// bf16 round of the partial: dst bf16 copy for the DMA push, part rounded in place
__global__ void ar_round_kernel(float *part, uint2 *dst, long n4) {
    for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += long(gridDim.x) * blockDim.x) {
        const float4 v = reinterpret_cast<float4 *>(part)[i];
        const uint16_t a = f2bf(v.x), b = f2bf(v.y), c = f2bf(v.z), d = f2bf(v.w);
        dst[i] = make_uint2(uint32_t(a) | (uint32_t(b) << 16), uint32_t(c) | (uint32_t(d) << 16));
        reinterpret_cast<float4 *>(part)[i] = make_float4(bf2f(a), bf2f(b), bf2f(c), bf2f(d));
    }
}
__global__ void ar_flag_kernel(unsigned *hflag, const unsigned *seq, int nslot) {
    const unsigned s = *seq;
    __threadfence_system();
    __hip_atomic_store(hflag + int(s % unsigned(nslot)) * 16, s + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void ar_wait_kernel(const unsigned *peer_flag, const unsigned *seq, int nslot) {
    const unsigned s = *seq;
    while (ld_flag(peer_flag + int(s % unsigned(nslot)) * 16) < s + 1) __builtin_amdgcn_s_sleep(2);
}
}  // namespace

void q_ar_round_bf16(float *part, void *dst, long n, hipStream_t st) {
    const long n4 = n / 4;
    ar_round_kernel<<<unsigned(std::min<long>(1024, (n4 + 255) / 256)), 256, 0, st>>>(part, static_cast<uint2 *>(dst), n4);
    HIP_CHECK(hipGetLastError());
}
void q_ar_flag(unsigned *hflag, const unsigned *seq, int nslot, hipStream_t st) {
    ar_flag_kernel<<<1, 1, 0, st>>>(hflag, seq, nslot);
    HIP_CHECK(hipGetLastError());
}
void q_ar_wait(const unsigned *peer_flag, const unsigned *seq, int nslot, hipStream_t st) {
    ar_wait_kernel<<<1, 1, 0, st>>>(peer_flag, seq, nslot);
    HIP_CHECK(hipGetLastError());
}
void q_ar_add_local(uint16_t *state, const float *own, const void *peer_dev, bool bf16, long n, int rank, unsigned *seq,
                    unsigned *ctr, hipStream_t st) {
    const long n4 = n / 4;
    const unsigned blocks = unsigned(std::min<long>(1024, (n4 + 255) / 256));
    ar_pull_kernel<<<blocks, 256, 0, st>>>(state, own, static_cast<const char *>(peer_dev), 0, bf16 ? 1 : 0, n4, rank,
                                           nullptr, seq, ctr, 1);
    HIP_CHECK(hipGetLastError());
}

void q_ar_push(float *part, long n, void *hbase, long slot_bytes, bool bf16, unsigned *hflag, const unsigned *seq,
               unsigned *ctr, int nslot, hipStream_t st) {
    const long n4 = n / 4;
    // 64 blocks: grid-stride (256 blocks hit a ~20 us floor on zero-copy writes on this box)
    const unsigned blocks = unsigned(std::min<long>(64, (n4 + 255) / 256));
    ar_push_kernel<<<blocks, 256, 0, st>>>(part, n4, static_cast<char *>(hbase), slot_bytes, bf16 ? 1 : 0, hflag, seq,
                                           ctr, nslot);
    HIP_CHECK(hipGetLastError());
}
void q_ar_pull_add(uint16_t *state, const float *own, const void *peer_hbase, long slot_bytes, bool bf16, long n,
                   int rank, const unsigned *peer_flag, unsigned *seq, unsigned *ctr, int nslot, hipStream_t st) {
    const long n4 = n / 4;
    const unsigned blocks = unsigned(std::min<long>(1024, (n4 + 255) / 256));
    static const bool spin_all = std::getenv("QWEN_AR_SPIN_ALL") != nullptr;
    if (!spin_all) ar_wait_kernel<<<1, 1, 0, st>>>(peer_flag, seq, nslot);  // one spinner, not one per block
    ar_pull_kernel<<<blocks, 256, 0, st>>>(state, own, static_cast<const char *>(peer_hbase), slot_bytes, bf16 ? 1 : 0,
                                           n4, rank, spin_all ? peer_flag : nullptr, seq, ctr, nslot);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth

// ------------------------------------------------------------------ Gated DeltaNet, whole chunk (v2)
// conv (silu'd causal conv of q|k|v, f32 [T][conv_dim]) -> prep: l2-normalise q (x 1/sqrt(kd)) and k
// in place, beta / decay into the qkvz row's b / a columns -> seq: one block per v-head, thread d owns
// state column S[:, d] in registers, q/k of 16 tokens staged in LDS, no per-token block reductions
// (raw o written over the conv v section) -> norm: gated rmsnorm of o -> bf16 out.
namespace behemoth {
namespace {
__device__ __forceinline__ float silu_d(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus_d(float x) { return x > 20.0f ? x : __logf(1.0f + __expf(x)); }
__global__ void dconv_kernel(const float *qkvz, long W, const uint16_t *conv_w, const float *ring, int T, int pos0,
                             int conv_dim, int K, float *conv) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= long(T) * conv_dim) return;
    const int t = int(i / conv_dim), c = int(i % conv_dim);
    const int tp = pos0 + t;
    float a = 0.f;
    for (int k = 0; k < K; ++k) {
        const int p = tp - K + 1 + k;
        float x = 0.f;
        if (p >= pos0) x = qkvz[long(p - pos0) * W + c];
        else if (p >= 0) x = ring[long(c) * K + (p % K)];
        a += x * bf2f(conv_w[long(c) * K + k]);
    }
    conv[i] = silu_d(a);
}
__global__ void dring_kernel(const float *qkvz, long W, float *ring, int T, int pos0, int conv_dim, int K) {
    const int c = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    for (int p = max(pos0, pos0 + T - K); p < pos0 + T; ++p) ring[long(c) * K + (p % K)] = qkvz[long(p - pos0) * W + c];
}
// grid T, block 32 * nk (wave per k head; lane = 4 dims)
__global__ void dprep_kernel(float *conv, float *qkvz, long W, int conv_dim, int value_dim, int nk, int nv,
                             const float *a_dt) {
    const int t = blockIdx.x, kh = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int key_dim = nk * 128;
    float *cr = conv + long(t) * conv_dim;
    float q[4], k[4], qs = 0, ks = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q[i] = cr[kh * 128 + lane * 4 + i];
        k[i] = cr[key_dim + kh * 128 + lane * 4 + i];
        qs += q[i] * q[i];
        ks += k[i] * k[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        qs += __shfl_xor(qs, off, 32);
        ks += __shfl_xor(ks, off, 32);
    }
    const float iq = rsqrtf(qs + 1e-6f) * rsqrtf(128.f), ik = rsqrtf(ks + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        cr[kh * 128 + lane * 4 + i] = q[i] * iq;
        cr[key_dim + kh * 128 + lane * 4 + i] = k[i] * ik;
    }
    for (int hv = threadIdx.x; hv < nv; hv += blockDim.x) {
        float *row = qkvz + long(t) * W;
        const float beta = 1.0f / (1.0f + __expf(-row[conv_dim + value_dim + hv]));
        const float decay = __expf(-__expf(a_dt[hv]) * softplus_d(row[conv_dim + value_dim + nv + hv] + a_dt[nv + hv]));
        row[conv_dim + value_dim + hv] = beta;
        row[conv_dim + value_dim + nv + hv] = decay;
    }
}
constexpr int DT = 16;
// grid (nv, 4 column groups), block 64: thread (col = 32 * group + tid / 2, half = tid & 1) owns
// S[64 * half .. +64][col]; the two halves of a dot product meet through one lane shuffle.
__global__ __launch_bounds__(64) void dseq_kernel(float *__restrict__ conv, const float *__restrict__ qkvz, long W,
                                                  float *__restrict__ S, int T, int conv_dim, int value_dim, int nk,
                                                  int nv) {
    const int hv = blockIdx.x, col = int(blockIdx.y) * 32 + (threadIdx.x >> 1), half = threadIdx.x & 1;
    const int kh = hv / (nv / nk), key_dim = nk * 128;
    float s[64];
    float *Sh = S + long(hv) * 128 * 128 + long(half) * 64 * 128 + col;
#pragma unroll
    for (int i = 0; i < 64; ++i) s[i] = Sh[long(i) * 128];
    __shared__ float4 sq[DT][32], sk[DT][32];
    __shared__ float sb[DT], sd[DT];
    for (int t0 = 0; t0 < T; t0 += DT) {
        const int nt = min(DT, T - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < nt * 64; i += 64) {
            const int tt = i >> 6, part = i & 63;
            const float *cr = conv + long(t0 + tt) * conv_dim;
            if (part < 32) sq[tt][part] = *reinterpret_cast<const float4 *>(cr + kh * 128 + part * 4);
            else sk[tt][part - 32] = *reinterpret_cast<const float4 *>(cr + key_dim + kh * 128 + (part - 32) * 4);
        }
        if (threadIdx.x < nt) {
            const float *row = qkvz + long(t0 + threadIdx.x) * W;
            sb[threadIdx.x] = row[conv_dim + value_dim + hv];
            sd[threadIdx.x] = row[conv_dim + value_dim + nv + hv];
        }
        __syncthreads();
        for (int tt = 0; tt < nt; ++tt) {
            float *vp = conv + long(t0 + tt) * conv_dim + 2 * key_dim + hv * 128 + col;
            const float vv = *vp, decay = sd[tt], beta = sb[tt];
            float m0 = 0, m1 = 0, m2 = 0, m3 = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float4 k4 = sk[tt][half * 16 + i];
                s[4 * i] *= decay;
                s[4 * i + 1] *= decay;
                s[4 * i + 2] *= decay;
                s[4 * i + 3] *= decay;
                m0 = fmaf(s[4 * i], k4.x, m0);
                m1 = fmaf(s[4 * i + 1], k4.y, m1);
                m2 = fmaf(s[4 * i + 2], k4.z, m2);
                m3 = fmaf(s[4 * i + 3], k4.w, m3);
            }
            float mem = (m0 + m1) + (m2 + m3);
            const float mo = __shfl_xor(mem, 1, 32);
            mem = half ? mo + mem : mem + mo;  // same order in both lanes
            const float delta = (vv - mem) * beta;
            float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float4 k4 = sk[tt][half * 16 + i], q4 = sq[tt][half * 16 + i];
                s[4 * i] = fmaf(k4.x, delta, s[4 * i]);
                s[4 * i + 1] = fmaf(k4.y, delta, s[4 * i + 1]);
                s[4 * i + 2] = fmaf(k4.z, delta, s[4 * i + 2]);
                s[4 * i + 3] = fmaf(k4.w, delta, s[4 * i + 3]);
                o0 = fmaf(s[4 * i], q4.x, o0);
                o1 = fmaf(s[4 * i + 1], q4.y, o1);
                o2 = fmaf(s[4 * i + 2], q4.z, o2);
                o3 = fmaf(s[4 * i + 3], q4.w, o3);
            }
            float o = (o0 + o1) + (o2 + o3);
            const float oo = __shfl_xor(o, 1, 32);
            if (!half) *vp = o + oo;
        }
    }
#pragma unroll
    for (int i = 0; i < 64; ++i) Sh[long(i) * 128] = s[i];
}
// grid (T, nv), block 128: out = bf16(o * rsqrt(mean(o^2)+eps) * w * silu(z))
__global__ __launch_bounds__(128) void dnorm_kernel(const float *conv, const float *qkvz, long W, const uint16_t *dn_norm,
                                                    uint16_t *out, int conv_dim, int value_dim, int nk, float eps) {
    const int t = blockIdx.x, hv = blockIdx.y, d = threadIdx.x;
    const float o = conv[long(t) * conv_dim + 2 * nk * 128 + hv * 128 + d];
    float ss = o * o;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    __shared__ float red[4];
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    ss = red[0] + red[1] + red[2] + red[3];
    const float z = qkvz[long(t) * W + conv_dim + hv * 128 + d];
    out[long(t) * value_dim + hv * 128 + d] = f2bf(o * rsqrtf(ss / 128.f + eps) * bf2f(dn_norm[d]) * silu_d(z));
}
}  // namespace

void q_delta_chunk2(float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm, const float *a_dt, float *ring,
                    float *S, uint16_t *out, float *conv, int T, int pos0, int conv_dim, int value_dim, int nk, int nv,
                    int K, float eps, hipStream_t st) {
    const long W = long(conv_dim) + value_dim + 2 * nv;
    const long n = long(T) * conv_dim;
    dconv_kernel<<<unsigned((n + 255) / 256), 256, 0, st>>>(qkvz, W, conv_w, ring, T, pos0, conv_dim, K, conv);
    HIP_CHECK(hipGetLastError());
    dring_kernel<<<unsigned((conv_dim + 255) / 256), 256, 0, st>>>(qkvz, W, ring, T, pos0, conv_dim, K);
    HIP_CHECK(hipGetLastError());
    dprep_kernel<<<unsigned(T), unsigned(32 * nk), 0, st>>>(conv, qkvz, W, conv_dim, value_dim, nk, nv, a_dt);
    HIP_CHECK(hipGetLastError());
    dseq_kernel<<<dim3(unsigned(nv), 4), 64, 0, st>>>(conv, qkvz, W, S, T, conv_dim, value_dim, nk, nv);
    HIP_CHECK(hipGetLastError());
    dnorm_kernel<<<dim3(unsigned(T), unsigned(nv)), 128, 0, st>>>(conv, qkvz, W, dn_norm, out, conv_dim, value_dim, nk,
                                                                  eps);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth

// ------------------------------------------------------------------ batched decode (one token per sequence)
// Rows b < B carry (token, slot, pos) in meta = [tokens | slots | pos] (3 x bmax ints). KV: 64-token
// pages, page table ptab[slot][maxpages]. DeltaNet ring / state: per-slot pools.
namespace behemoth {
namespace {
__device__ __forceinline__ float bsum256(float v, float *red) {  // block of 256: sum to every thread
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < int(blockDim.x >> 5); ++i) r += red[i];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ long kvrow_b(const int *pt, int t) { return long(pt[t >> 6]) * 64 + (t & 63); }

// grid (nq + nkv, B), block hd (256)
__global__ __launch_bounds__(256) void rope_b_kernel(const float *__restrict__ z, int Wf, uint16_t *__restrict__ qout,
                                                     uint16_t *__restrict__ G, uint16_t *__restrict__ kc,
                                                     uint16_t *__restrict__ vc, const int *__restrict__ meta, int bmax,
                                                     const int *__restrict__ ptab, int maxpages, int nq, int nkv, int rot,
                                                     float theta, const uint16_t *__restrict__ q_norm,
                                                     const uint16_t *__restrict__ k_norm, float eps) {
    constexpr int hd = 256;
    const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    const int slot = meta[bmax + b], pos = meta[2 * bmax + b];
    const float *zr = z + long(b) * Wf;
    const bool is_q = h < nq;
    float raw;
    long kvr = 0;
    if (is_q) {
        raw = rbf(zr[h * 2 * hd + d]);
        G[long(b) * nq * hd + h * hd + d] = f2bf(zr[h * 2 * hd + hd + d]);
    } else {
        const int kh = h - nq;
        raw = rbf(zr[nq * 2 * hd + kh * hd + d]);
        kvr = kvrow_b(ptab + long(slot) * maxpages, pos);
        vc[(kvr * nkv + kh) * hd + d] = f2bf(zr[nq * 2 * hd + nkv * hd + kh * hd + d]);
    }
    __shared__ float red[8], sh[hd];
    const float ss = bsum256(raw * raw, red);
    const float inv = rsqrtf(ss / hd + eps);
    const uint16_t *w = is_q ? q_norm : k_norm;
    const float sv = raw * inv * (1.0f + bf2f(w[d]));
    sh[d] = sv;
    __syncthreads();
    float o;
    if (d < rot / 2) {
        const float inv_freq = __powf(theta, -2.0f * float(d) / float(rot));
        const float ang = float(pos) * inv_freq;
        o = sv * cosf(ang) - sh[d + rot / 2] * sinf(ang);
    } else if (d < rot) {
        const int dd = d - rot / 2;
        const float inv_freq = __powf(theta, -2.0f * float(dd) / float(rot));
        const float ang = float(pos) * inv_freq;
        o = sv * cosf(ang) + sh[dd] * sinf(ang);
    } else {
        o = sv;
    }
    if (is_q) qout[long(b) * nq * hd + h * hd + d] = f2bf(o);
    else kc[(kvr * nkv + (h - nq)) * hd + d] = f2bf(o);
}

// grid (maxsplit, nkv, B), block 256 (8 waves): split of 512 positions; in-block ordered merge of
// the 8 wave partials; one (m, l, acc[256]) partial per (b, q head, split).
constexpr int ASPLIT = 512;
template <int GQ>
__global__ __launch_bounds__(256) void attn_b_kernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ kc,
                                                     const uint16_t *__restrict__ vc, const int *__restrict__ meta,
                                                     int bmax, const int *__restrict__ ptab, int maxpages, int nq,
                                                     int nkv, int maxsplit, float scale, float *__restrict__ pml,
                                                     float *__restrict__ pacc) {
    constexpr int HD = 256, DPL = HD / 32;
    const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
    const int len = meta[2 * bmax + b] + 1;
    if (split * ASPLIT >= len) return;
    const int *pt = ptab + long(meta[bmax + b]) * maxpages;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ float qs[GQ][HD];
    __shared__ float wm[8][GQ], wl[8][GQ];
    __shared__ float red[8][HD];
    for (int i = threadIdx.x; i < GQ * HD; i += 256) {
        const int g = i / HD, d = i % HD;
        qs[g][d] = bf2f(q[long(b) * nq * HD + long(kh * GQ + g) * HD + d]) * scale;
    }
    __syncthreads();
    const int t_beg = split * ASPLIT, t_end = min(len, t_beg + ASPLIT);
    float m[GQ], l[GQ], acc[GQ][DPL];
#pragma unroll
    for (int g = 0; g < GQ; ++g) {
        m[g] = -1e30f;
        l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[g][i] = 0.f;
    }
    const long kstride = long(nkv) * HD;
    for (int base = t_beg + wave * 32; base < t_end; base += 8 * 32) {
        const int t = base + lane;
        const bool ok = t < t_end;
        float dot[GQ];
#pragma unroll
        for (int g = 0; g < GQ; ++g) dot[g] = 0.f;
        if (ok) {
            const uint4 *kp = reinterpret_cast<const uint4 *>(kc + kvrow_b(pt, t) * kstride + long(kh) * HD);
#pragma unroll 4
            for (int c = 0; c < HD / 8; ++c) {
                const uint4 kv = kp[c];
                const uint32_t w4[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float k0 = bf2f(uint16_t(w4[e])), k1 = bf2f(uint16_t(w4[e] >> 16));
#pragma unroll
                    for (int g = 0; g < GQ; ++g)
                        dot[g] = fmaf(qs[g][c * 8 + 2 * e + 1], k1, fmaf(qs[g][c * 8 + 2 * e], k0, dot[g]));
                }
            }
        }
        float p[GQ];
#pragma unroll
        for (int g = 0; g < GQ; ++g) {
            float cm = ok ? dot[g] : -1e30f;
            for (int off = 16; off > 0; off >>= 1) cm = fmaxf(cm, __shfl_xor(cm, off, 32));
            const float mn = fmaxf(m[g], cm);
            const float corr = __expf(m[g] - mn);
            m[g] = mn;
            p[g] = ok ? __expf(dot[g] - mn) : 0.f;
            l[g] = l[g] * corr + p[g];
#pragma unroll
            for (int i = 0; i < DPL; ++i) acc[g][i] *= corr;
        }
        const int nj = min(32, t_end - base);
        const long vrow0 = kvrow_b(pt, base);  // base is 32-aligned: the 32 positions share one page
        for (int j = 0; j < nj; ++j) {
            const uint4 vv = *reinterpret_cast<const uint4 *>(vc + (vrow0 + j) * kstride + long(kh) * HD + lane * DPL);
            const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
            float v[DPL];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[2 * e] = bf2f(uint16_t(w4[e]));
                v[2 * e + 1] = bf2f(uint16_t(w4[e] >> 16));
            }
#pragma unroll
            for (int g = 0; g < GQ; ++g) {
                const float pg = __shfl(p[g], j, 32);
#pragma unroll
                for (int i = 0; i < DPL; ++i) acc[g][i] = fmaf(pg, v[i], acc[g][i]);
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GQ; ++g) {
        float ls = l[g];
        for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor(ls, off, 32);
        if (lane == 0) {
            wm[wave][g] = m[g];
            wl[wave][g] = ls;
        }
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < GQ; ++g) {
        float M = -1e30f;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, wm[w][g]);
        const float f = wl[wave][g] > 0.f ? __expf(m[g] - M) : 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) red[wave][lane * DPL + i] = acc[g][i] * f;
        __syncthreads();
        const int d = threadIdx.x;
        float a = 0.f, L = 0.f;
        for (int w = 0; w < 8; ++w) {
            a += red[w][d];
            L += wl[w][g] > 0.f ? wl[w][g] * __expf(wm[w][g] - M) : 0.f;
        }
        const long e = (long(b) * nq + kh * GQ + g) * maxsplit + split;
        pacc[e * HD + d] = a;
        if (d == 0) {
            pml[2 * e] = M;
            pml[2 * e + 1] = L;
        }
        __syncthreads();
    }
}
// grid (nq, B), block 256
__global__ __launch_bounds__(256) void attn_b_combine_kernel(const float *pml, const float *pacc, int maxsplit,
                                                             const int *meta, int bmax, const uint16_t *G,
                                                             uint16_t *out, int nq) {
    constexpr int HD = 256;
    const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    const int len = meta[2 * bmax + b] + 1, ns = (len + ASPLIT - 1) / ASPLIT;
    const long e0 = (long(b) * nq + h) * maxsplit;
    float M = -1e30f;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, pml[2 * (e0 + s)]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float w = __expf(pml[2 * (e0 + s)] - M);
        den += w * pml[2 * (e0 + s) + 1];
        num += w * pacc[(e0 + s) * HD + d];
    }
    const float gt = bf2f(G[long(b) * nq * HD + h * HD + d]);
    out[long(b) * nq * HD + h * HD + d] = f2bf(num / den / (1.0f + __expf(-gt)));
}

// grid (nv, B), block 256: thread (col = tid / 2, half = tid & 1)
__global__ __launch_bounds__(256) void delta_b_kernel(const float *__restrict__ qkvz, long W,
                                                      const uint16_t *__restrict__ conv_w,
                                                      const uint16_t *__restrict__ dn_norm,
                                                      const float *__restrict__ a_dt, float *__restrict__ ring,
                                                      float *__restrict__ S, uint16_t *__restrict__ out,
                                                      const int *__restrict__ meta, int bmax, int conv_dim,
                                                      int value_dim, int nk, int nv, int K, float eps) {
    const int hv = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const int slot = meta[bmax + b], pos = meta[2 * bmax + b];
    const int kh = hv / (nv / nk), key_dim = nk * 128;
    const float *row = qkvz + long(b) * W;
    float *rg = ring + long(slot) * conv_dim * K;
    __shared__ float sq[128], sk[128], sv[128], red[8];
    // causal conv of this head's q / k (threads < 128) and v (threads >= 128) channel
    float cq = 0.f, ck = 0.f;
    {
        const int i = tid & 127;
        const int c0 = tid < 128 ? kh * 128 + i : 2 * key_dim + hv * 128 + i;
        const int c1 = key_dim + kh * 128 + i;  // k (threads < 128 only)
        auto conv1 = [&](int c) {
            const float x = row[c];
            float a = 0.f;
            for (int k = 0; k < K - 1; ++k) {
                const int p = pos - K + 1 + k;
                a += (p >= 0 ? rg[long(c) * K + (p % K)] : 0.f) * bf2f(conv_w[long(c) * K + k]);
            }
            a += x * bf2f(conv_w[long(c) * K + K - 1]);
            return silu_d(a);
        };
        cq = conv1(c0);
        if (tid < 128) ck = conv1(c1);
    }
    __syncthreads();  // all ring reads of this block done before the writes below
    {
        const int i = tid & 127;
        const int c0 = tid < 128 ? kh * 128 + i : 2 * key_dim + hv * 128 + i;
        rg[long(c0) * K + (pos % K)] = row[c0];
        if (tid < 128) rg[long(key_dim + kh * 128 + i) * K + (pos % K)] = row[key_dim + kh * 128 + i];
    }
    const float qs = bsum256(tid < 128 ? cq * cq : 0.f, red);
    const float ks = bsum256(tid < 128 ? ck * ck : 0.f, red);
    if (tid < 128) {
        sq[tid] = cq * rsqrtf(qs + 1e-6f) * rsqrtf(128.f);
        sk[tid] = ck * rsqrtf(ks + 1e-6f);
    } else {
        sv[tid - 128] = cq;
    }
    __syncthreads();
    const float beta = 1.0f / (1.0f + __expf(-row[conv_dim + value_dim + hv]));
    const float decay = __expf(-__expf(a_dt[hv]) * softplus_d(row[conv_dim + value_dim + nv + hv] + a_dt[nv + hv]));
    const int col = tid >> 1, half = tid & 1;
    float *Sh = S + (long(slot) * nv + hv) * 128 * 128 + long(half) * 64 * 128 + col;
    float s[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) s[i] = Sh[long(i) * 128] * decay;
    float m0 = 0, m1 = 0, m2 = 0, m3 = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        m0 = fmaf(s[4 * i], sk[half * 64 + 4 * i], m0);
        m1 = fmaf(s[4 * i + 1], sk[half * 64 + 4 * i + 1], m1);
        m2 = fmaf(s[4 * i + 2], sk[half * 64 + 4 * i + 2], m2);
        m3 = fmaf(s[4 * i + 3], sk[half * 64 + 4 * i + 3], m3);
    }
    float mem = (m0 + m1) + (m2 + m3);
    const float mo = __shfl_xor(mem, 1, 32);
    mem = half ? mo + mem : mem + mo;
    const float delta = (sv[col] - mem) * beta;
    float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        s[4 * i] = fmaf(sk[half * 64 + 4 * i], delta, s[4 * i]);
        s[4 * i + 1] = fmaf(sk[half * 64 + 4 * i + 1], delta, s[4 * i + 1]);
        s[4 * i + 2] = fmaf(sk[half * 64 + 4 * i + 2], delta, s[4 * i + 2]);
        s[4 * i + 3] = fmaf(sk[half * 64 + 4 * i + 3], delta, s[4 * i + 3]);
        o0 = fmaf(s[4 * i], sq[half * 64 + 4 * i], o0);
        o1 = fmaf(s[4 * i + 1], sq[half * 64 + 4 * i + 1], o1);
        o2 = fmaf(s[4 * i + 2], sq[half * 64 + 4 * i + 2], o2);
        o3 = fmaf(s[4 * i + 3], sq[half * 64 + 4 * i + 3], o3);
    }
#pragma unroll
    for (int i = 0; i < 64; ++i) Sh[long(i) * 128] = s[i];
    float o = (o0 + o1) + (o2 + o3);
    const float oo = __shfl_xor(o, 1, 32);
    o = half ? oo + o : o + oo;
    const float os = bsum256(half ? 0.f : o * o, red);
    if (!half) {
        const float zz = row[conv_dim + hv * 128 + col];
        out[long(b) * value_dim + hv * 128 + col] = f2bf(o * rsqrtf(os / 128.f + eps) * bf2f(dn_norm[col]) * silu_d(zz));
    }
}

// Top-K (K <= 64) of each row: 4-pass 8-bit radix select of the K-th largest key, then collect.
// grid rows, block 1024. Output unsorted.
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ __launch_bounds__(1024) void topk_kernel(const float *__restrict__ logits, int V, int K, int voff,
                                                    float *__restrict__ oval, int *__restrict__ oid, int ostride) {
    const float *x = logits + long(blockIdx.x) * V;
    __shared__ unsigned hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need, cgt, ceq;
    if (threadIdx.x == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_need = K;
    }
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int i = threadIdx.x; i < V; i += 1024) {
            const uint32_t k = fkey(x[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int need = s_need, cum = 0, sel = 0;
            for (int bin = 255; bin >= 0; --bin) {
                if (cum + int(hist[bin]) >= need) {
                    sel = bin;
                    need -= cum;
                    break;
                }
                cum += int(hist[bin]);
            }
            s_need = need;
            s_prefix = prefix | (uint32_t(sel) << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }
    const uint32_t T = s_prefix;
    const int need = s_need, ngt = K - need;
    if (threadIdx.x == 0) {
        cgt = 0;
        ceq = 0;
    }
    __syncthreads();
    float *ov = oval + long(blockIdx.x) * ostride;
    int *oi = oid + long(blockIdx.x) * ostride;
    for (int i = threadIdx.x; i < V; i += 1024) {
        const uint32_t k = fkey(x[i]);
        if (k > T) {
            const int p = atomicAdd(&cgt, 1);
            ov[p] = x[i];
            oi[p] = i + voff;
        } else if (k == T) {
            const int p = atomicAdd(&ceq, 1);
            if (p < need) {
                ov[ngt + p] = x[i];
                oi[ngt + p] = i + voff;
            }
        }
    }
}
}  // namespace

void q_rope_b(const float *z, int Wf, uint16_t *qout, uint16_t *G, uint16_t *kc, uint16_t *vc, const int *meta, int bmax,
              int B, const int *ptab, int maxpages, int nq, int nkv, int hd, int rot, float theta, const uint16_t *q_norm,
              const uint16_t *k_norm, float eps, hipStream_t st) {
    if (hd != 256) throw std::runtime_error("q_rope_b: head dim 256 only");
    rope_b_kernel<<<dim3(unsigned(nq + nkv), unsigned(B)), 256, 0, st>>>(z, Wf, qout, G, kc, vc, meta, bmax, ptab,
                                                                         maxpages, nq, nkv, rot, theta, q_norm, k_norm,
                                                                         eps);
    HIP_CHECK(hipGetLastError());
}
size_t q_attn_b_scratch(int bmax, int nq, int max_context) {
    const int maxsplit = (max_context + ASPLIT - 1) / ASPLIT;
    return size_t(bmax) * nq * maxsplit * (256 + 2);
}
void q_attn_b(const uint16_t *q, const uint16_t *kc, const uint16_t *vc, const uint16_t *G, uint16_t *out,
              const int *meta, int bmax, int B, const int *ptab, int maxpages, int nq, int nkv, int max_context,
              float scale, float *scratch, hipStream_t st) {
    const int maxsplit = (max_context + ASPLIT - 1) / ASPLIT;
    float *pml = scratch, *pacc = scratch + size_t(bmax) * nq * maxsplit * 2;
    const dim3 grid{unsigned(maxsplit), unsigned(nkv), unsigned(B)};
    switch (nq / nkv) {
    case 1: attn_b_kernel<1><<<grid, 256, 0, st>>>(q, kc, vc, meta, bmax, ptab, maxpages, nq, nkv, maxsplit, scale, pml, pacc); break;
    case 2: attn_b_kernel<2><<<grid, 256, 0, st>>>(q, kc, vc, meta, bmax, ptab, maxpages, nq, nkv, maxsplit, scale, pml, pacc); break;
    case 4: attn_b_kernel<4><<<grid, 256, 0, st>>>(q, kc, vc, meta, bmax, ptab, maxpages, nq, nkv, maxsplit, scale, pml, pacc); break;
    case 6: attn_b_kernel<6><<<grid, 256, 0, st>>>(q, kc, vc, meta, bmax, ptab, maxpages, nq, nkv, maxsplit, scale, pml, pacc); break;
    default: throw std::runtime_error("q_attn_b: unsupported GQA group");
    }
    HIP_CHECK(hipGetLastError());
    attn_b_combine_kernel<<<dim3(unsigned(nq), unsigned(B)), 256, 0, st>>>(pml, pacc, maxsplit, meta, bmax, G, out, nq);
    HIP_CHECK(hipGetLastError());
}
void q_delta_b(const float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm, const float *a_dt, float *ring, float *S,
               uint16_t *out, const int *meta, int bmax, int B, int conv_dim, int value_dim, int nk, int nv, int K,
               float eps, hipStream_t st) {
    const long W = long(conv_dim) + value_dim + 2 * nv;
    delta_b_kernel<<<dim3(unsigned(nv), unsigned(B)), 256, 0, st>>>(qkvz, W, conv_w, dn_norm, a_dt, ring, S, out, meta, bmax,
                                                                   conv_dim, value_dim, nk, nv, K, eps);
    HIP_CHECK(hipGetLastError());
}
void q_topk(const float *logits, int rows, int V, int K, int voff, float *oval, int *oid, int ostride, hipStream_t st) {
    if (K > 64 || K < 1) throw std::runtime_error("q_topk: K <= 64");
    topk_kernel<<<unsigned(rows), 1024, 0, st>>>(logits, V, K, voff, oval, oid, ostride);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth
