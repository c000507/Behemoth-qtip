// Training-only row transforms for the fast trainer engine (qtip_training/fast.py).
// One 256-thread block per row; thread t holds positions 4t..4t+3 of each 1024-chunk, so a row
// is read once and every output chunk is mixed from registers (as runtime/src/qtip.hip's row
// kernels). y = hadU(x') or hadUt(x') / sqrt(n), x' = pre (.) [bf16](x); optional fp32 output y
// and bf16 output bf16(post (.) y). Strided rows: the trainer passes column slices of stacked
// q|k|v and gate|up GEMM outputs directly.
#include "../../runtime/src/device_numeric.hpp"
#include "../../runtime/src/qtip_tables.hpp"
#include <hip/hip_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
__constant__ int8_t t_had12[144];
__constant__ int8_t t_had28[784];

void check(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string("HIP: ") + hipGetErrorString(e));
}

void tables() {
    static std::mutex mutex;
    static bool done[64] = {};
    int device = 0;
    check(hipGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (device < 0 || device >= 64)
        throw std::runtime_error("device index");
    if (!done[device]) {
        check(hipMemcpyToSymbol(HIP_SYMBOL(t_had12), behemoth::qtip::kHad12, sizeof(behemoth::qtip::kHad12)));
        check(hipMemcpyToSymbol(HIP_SYMBOL(t_had28), behemoth::qtip::kHad28, sizeof(behemoth::qtip::kHad28)));
        done[device] = true;
    }
}

using behemoth::device::bf16;

__device__ __forceinline__ void fwht1024(float v[4], float *scratch) {
    {
        const float a = v[0], b = v[1], c = v[2], d = v[3];
        v[0] = a + b;
        v[1] = a - b;
        v[2] = c + d;
        v[3] = c - d;
    }
    {
        const float a = v[0], b = v[1], c = v[2], d = v[3];
        v[0] = a + c;
        v[1] = b + d;
        v[2] = a - c;
        v[3] = b - d;
    }
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor(v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
#pragma unroll
    for (int m = 1; m < 8; m <<= 1) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
            scratch[4 * tid + i] = v[i];
        __syncthreads();
        const int partner = tid ^ (32 * m);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = scratch[4 * partner + i];
            v[i] = (wave & m) ? o - v[i] : v[i] + o;
        }
    }
}

template <int K> __device__ __forceinline__ float sign(int r, int c) {
    if constexpr (K == 12)
        return float(t_had12[r * 12 + c]);
    else
        return float(t_had28[r * 28 + c]);
}

template <bool BF16> __device__ __forceinline__ void load4(const void *p, size_t i, float out[4]) {
    if constexpr (BF16) {
        const uint2 v = *reinterpret_cast<const uint2 *>(static_cast<const uint16_t *>(p) + i);
        out[0] = bf16(uint16_t(v.x & 0xffffU));
        out[1] = bf16(uint16_t(v.x >> 16));
        out[2] = bf16(uint16_t(v.y & 0xffffU));
        out[3] = bf16(uint16_t(v.y >> 16));
    } else {
        const float4 v = *reinterpret_cast<const float4 *>(static_cast<const float *>(p) + i);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
    }
}

template <int K, bool IN_BF16>
__global__ __launch_bounds__(256) void had_rows_kernel(const void *in, long ld_in, const float *pre,
                                                       int round_in, int transpose, float *out32,
                                                       long ld32, uint16_t *out16, long ld16,
                                                       const float *post) {
    __shared__ float scratch[1024];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    constexpr int n = K * 1024;
    float t[K][4];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        load4<IN_BF16>(in, size_t(row) * size_t(ld_in) + size_t(k * 1024 + j), t[k]);
        if (round_in) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                t[k][i] = bf16(bf16(t[k][i]));
        }
        if (pre) {
            const float4 s = *reinterpret_cast<const float4 *>(pre + k * 1024 + j);
            t[k][0] *= s.x;
            t[k][1] *= s.y;
            t[k][2] *= s.z;
            t[k][3] *= s.w;
        }
    }
    const float scale = rsqrtf(float(n));
    for (int k2 = 0; k2 < K; ++k2) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        if constexpr (K == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = t[0][i];
        } else {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const float s = transpose ? sign<K>(k, k2) : sign<K>(k2, k);
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    v[i] += s * t[k][i];
            }
        }
        fwht1024(v, scratch);
        const int e = k2 * 1024 + j;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] *= scale;
        if (out32)
            *reinterpret_cast<float4 *>(out32 + size_t(row) * size_t(ld32) + size_t(e)) =
                make_float4(v[0], v[1], v[2], v[3]);
        if (out16) {
            float w[4] = {v[0], v[1], v[2], v[3]};
            if (post) {
                const float4 s = *reinterpret_cast<const float4 *>(post + e);
                w[0] *= s.x;
                w[1] *= s.y;
                w[2] *= s.z;
                w[3] *= s.w;
            }
            *reinterpret_cast<uint2 *>(out16 + size_t(row) * size_t(ld16) + size_t(e)) =
                make_uint2(uint32_t(bf16(w[0])) | uint32_t(bf16(w[1])) << 16,
                           uint32_t(bf16(w[2])) | uint32_t(bf16(w[3])) << 16);
        }
    }
}

__device__ __forceinline__ float block_sum(float v, float *red) {
    for (int m = 16; m; m >>= 1)
        v += __shfl_xor(v, m);
    __syncthreads();
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        s += red[w];
    return s;
}

__device__ __forceinline__ void store_bf16x4(uint16_t *p, const float v[4]) {
    *reinterpret_cast<uint2 *>(p) = make_uint2(uint32_t(bf16(v[0])) | uint32_t(bf16(v[1])) << 16,
                                               uint32_t(bf16(v[2])) | uint32_t(bf16(v[3])) << 16);
}
__device__ __forceinline__ float4 ld4(const float *p) { return *reinterpret_cast<const float4 *>(p); }

// Fused RMSNorm + input transform (n = 12288): xn = bf16((x * r) * w), r = rsqrt(mean(x^2) + eps),
// u = bf16(hadUt(su (.) xn)). Optional xn / r outputs (backward needs them). grid (rows).
__global__ __launch_bounds__(256) void rms_had_kernel(const uint16_t *x, long ld, const float *w, float eps,
                                                      const float *su, uint16_t *xn, long ldxn, float *r,
                                                      uint16_t *u, long ldu) {
    constexpr int K = 12, n = K * 1024;
    __shared__ float scratch[1024];
    __shared__ float red[8];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    float t[K][4];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        load4<true>(x, size_t(row) * size_t(ld) + size_t(k * 1024 + j), t[k]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            ss += t[k][i] * t[k][i];
    }
    const float factor = rsqrtf(block_sum(ss, red) / float(n) + eps);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int e = k * 1024 + j;
        const float4 wv = ld4(w + e), sv = ld4(su + e);
        const float ww[4] = {wv.x, wv.y, wv.z, wv.w}, ss4[4] = {sv.x, sv.y, sv.z, sv.w};
        float y[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            y[i] = bf16(bf16(t[k][i] * factor * ww[i]));
        if (xn)
            store_bf16x4(xn + size_t(row) * size_t(ldxn) + size_t(e), y);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            t[k][i] = ss4[i] * y[i];
    }
    if (r && threadIdx.x == 0)
        r[row] = factor;
    const float scale = rsqrtf(float(n));
    for (int k2 = 0; k2 < K; ++k2) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float s = sign<K>(k, k2); // hadUt
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] += s * t[k][i];
        }
        fwht1024(v, scratch);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] *= scale;
        store_bf16x4(u + size_t(row) * size_t(ldu) + size_t(k2 * 1024 + j), v);
    }
}

// gate|up epilogue + SwiGLU (I = 28672): z fp32 [rows][ldz] (gate at 0, up at I); gate/up use
// bf16(z) (the runtime's bf16 exchange), c = hadU(.)/sqrt(I), g = bf16(c_g svs_g), u = bf16(c_u svs_u),
// product = bf16(silu(g) u). Optional fp32 c_g / c_u. grid (rows).
__global__ __launch_bounds__(256) void gateup_kernel(const float *z, long ldz, const float *svs_g, const float *svs_u,
                                                     uint16_t *gate, uint16_t *up, uint16_t *product, long ldo,
                                                     float *cg, float *cu, long ldc) {
    constexpr int K = 28, I = K * 1024;
    __shared__ float scratch[1024];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    const float scale = rsqrtf(float(I));
    float t[K][4];
    for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            load4<false>(z, size_t(row) * size_t(ldz) + size_t(part * I + k * 1024 + j), t[k]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                t[k][i] = bf16(bf16(t[k][i]));
        }
        const float *svs = part ? svs_u : svs_g;
        float *c = part ? cu : cg;
        for (int k2 = 0; k2 < K; ++k2) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const float s = sign<K>(k2, k); // hadU
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    v[i] += s * t[k][i];
            }
            fwht1024(v, scratch);
            const int e = k2 * 1024 + j;
            const size_t o = size_t(row) * size_t(ldo) + size_t(e);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] *= scale;
            if (c)
                *reinterpret_cast<float4 *>(c + size_t(row) * size_t(ldc) + size_t(e)) = make_float4(v[0], v[1], v[2], v[3]);
            const float4 sv = ld4(svs + e);
            float y[4] = {v[0] * sv.x, v[1] * sv.y, v[2] * sv.z, v[3] * sv.w};
            if (!part) {
                store_bf16x4(gate + o, y);
            } else {
                store_bf16x4(up + o, y);
                const uint2 gv = *reinterpret_cast<const uint2 *>(gate + o); // this thread's own store
                const float g[4] = {bf16(uint16_t(gv.x & 0xffffU)), bf16(uint16_t(gv.x >> 16)),
                                    bf16(uint16_t(gv.y & 0xffffU)), bf16(uint16_t(gv.y >> 16))};
                float p[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    p[i] = g[i] / (1.f + expf(-g[i])) * bf16(bf16(y[i]));
                store_bf16x4(product + o, p);
            }
        }
    }
}

// Column sums with atomics: acc[c] += sum_r a[r][c] * b[r][c] (* rs[r]). a fp32, b fp32/bf16.
// Block: 256 threads x 4 columns, ROWS rows. grid (n / 1024, ceil(rows / ROWS)).
template <bool B16>
__global__ __launch_bounds__(256) void colsum_kernel(const float *a, long lda, const void *b, long ldb,
                                                     const float *rs, int rows, int rows_per_block, float *acc) {
    const int c = int(blockIdx.x) * 1024 + 4 * int(threadIdx.x);
    const int r0 = int(blockIdx.y) * rows_per_block, r1 = min(rows, r0 + rows_per_block);
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int r = r0; r < r1; ++r) {
        const float4 av = ld4(a + size_t(r) * size_t(lda) + size_t(c));
        float bv[4];
        load4<B16>(b, size_t(r) * size_t(ldb) + size_t(c), bv);
        const float f = rs ? rs[r] : 1.f;
        s[0] += av.x * bv[0] * f;
        s[1] += av.y * bv[1] * f;
        s[2] += av.z * bv[2] * f;
        s[3] += av.w * bv[3] * f;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        atomicAdd(acc + c + i, s[i]);
}

// SwiGLU backward: dp = dxsu * su_d; dg = dp * u * silu'(g); du = dp * silu(g) (fp32 outputs, may alias
// dxsu for dg). Elementwise over [rows][I]; grid (I / 1024, rows).
__global__ __launch_bounds__(256) void swiglu_bwd_kernel(const float *dxsu, const float *su, const uint16_t *gate,
                                                         const uint16_t *up, float *dg, float *du, int cols) {
    const size_t o = size_t(blockIdx.y) * size_t(cols) + size_t(blockIdx.x) * 1024 + 4 * threadIdx.x;
    const int e = int(blockIdx.x) * 1024 + 4 * int(threadIdx.x);
    const float4 d = ld4(dxsu + o), s = ld4(su + e);
    float g[4], u[4];
    load4<true>(gate, o, g);
    load4<true>(up, o, u);
    const float dp[4] = {d.x * s.x, d.y * s.y, d.z * s.z, d.w * s.w};
    float a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float sg = 1.f / (1.f + expf(-g[i]));
        b[i] = dp[i] * g[i] * sg;
        a[i] = dp[i] * u[i] * (sg * (1.f + g[i] * (1.f - sg)));
    }
    *reinterpret_cast<float4 *>(dg + o) = make_float4(a[0], a[1], a[2], a[3]);
    *reinterpret_cast<float4 *>(du + o) = make_float4(b[0], b[1], b[2], b[3]);
}

// RMSNorm backward, row part (n = 12288): dxhat = dxsu * su * w, xhat = x * r,
// dh += r * (dxhat - xhat * mean(dxhat * xhat)). grid (rows).
__global__ __launch_bounds__(256) void rms_bwd_kernel(const uint16_t *x, long ldx, const float *r, const float *w,
                                                      const float *su, const float *dxsu, long ldd, float *dh,
                                                      long ldh) {
    constexpr int K = 12, n = K * 1024;
    __shared__ float red[8];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    const float rr = r[row];
    float xh[K][4], dx[K][4];
    float dot = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int e = k * 1024 + j;
        load4<true>(x, size_t(row) * size_t(ldx) + size_t(e), xh[k]);
        const float4 d = ld4(dxsu + size_t(row) * size_t(ldd) + size_t(e)), s = ld4(su + e), ww = ld4(w + e);
        dx[k][0] = d.x * s.x * ww.x;
        dx[k][1] = d.y * s.y * ww.y;
        dx[k][2] = d.z * s.z * ww.z;
        dx[k][3] = d.w * s.w * ww.w;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            xh[k][i] *= rr;
            dot += dx[k][i] * xh[k][i];
        }
    }
    const float mean = block_sum(dot, red) / float(n);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        float *p = dh + size_t(row) * size_t(ldh) + size_t(k * 1024 + j);
        float4 h = *reinterpret_cast<float4 *>(p);
        h.x += rr * (dx[k][0] - xh[k][0] * mean);
        h.y += rr * (dx[k][1] - xh[k][1] * mean);
        h.z += rr * (dx[k][2] - xh[k][2] * mean);
        h.w += rr * (dx[k][3] - xh[k][3] * mean);
        *reinterpret_cast<float4 *>(p) = h;
    }
}

// RoPE on head_dim 128 (pairs d, d + 64): forward bf16 [rows][ld] (heads x 128) -> bf16 [rows][heads][128];
// backward (adjoint) bf16 -> fp32. cos/sin fp32 [rows][128]. grid (heads, rows), 64 threads.
template <bool BACKWARD>
__global__ __launch_bounds__(64) void rope_kernel(const uint16_t *in, long ld, const float *cs, const float *sn,
                                                  void *out, int heads) {
    const int h = int(blockIdx.x), row = int(blockIdx.y), d = int(threadIdx.x);
    const uint16_t *p = in + size_t(row) * size_t(ld) + size_t(h) * 128;
    const float a = bf16(p[d]), b = bf16(p[d + 64]);
    const float c0 = cs[size_t(row) * 128 + d], c1 = cs[size_t(row) * 128 + d + 64];
    const float s0 = sn[size_t(row) * 128 + d], s1 = sn[size_t(row) * 128 + d + 64];
    const size_t o = (size_t(row) * size_t(heads) + size_t(h)) * 128;
    if constexpr (!BACKWARD) {
        uint16_t *q = static_cast<uint16_t *>(out) + o;
        q[d] = bf16(a * c0 + (-b) * s0);
        q[d + 64] = bf16(b * c1 + a * s1);
    } else {
        float *q = static_cast<float *>(out) + o;
        q[d] = a * c0 + b * s1;
        q[d + 64] = b * c1 - a * s0;
    }
}

template <int K>
void launch(const void *in, int in_bf16, long ld_in, int rows, const float *pre, int round_in,
            int transpose, float *out32, long ld32, uint16_t *out16, long ld16, const float *post,
            hipStream_t stream) {
    if (in_bf16)
        had_rows_kernel<K, true><<<rows, 256, 0, stream>>>(in, ld_in, pre, round_in, transpose, out32,
                                                          ld32, out16, ld16, post);
    else
        had_rows_kernel<K, false><<<rows, 256, 0, stream>>>(in, ld_in, pre, round_in, transpose, out32,
                                                           ld32, out16, ld16, post);
    check(hipGetLastError());
}
} // namespace

namespace qtip_training {
void had_rows(const void *in, int in_bf16, long ld_in, int rows, int n, int transpose, const float *pre,
              int round_in, float *out32, long ld32, uint16_t *out16, long ld16, const float *post,
              hipStream_t stream) {
    if (rows <= 0)
        return;
    if (!out32 && !out16)
        throw std::invalid_argument("had_rows needs an output");
    auto aligned = [](const void *p, size_t a) { return (reinterpret_cast<uintptr_t>(p) % a) == 0; };
    if (!aligned(in, in_bf16 ? 8 : 16) || ld_in % 4 || ld32 % 4 || ld16 % 4 || (out32 && !aligned(out32, 16)) ||
        (out16 && !aligned(out16, 8)) || (pre && !aligned(pre, 16)) || (post && !aligned(post, 16)))
        throw std::invalid_argument("had_rows alignment");
    tables();
    if (n == 1024)
        launch<1>(in, in_bf16, ld_in, rows, pre, round_in, transpose, out32, ld32, out16, ld16, post, stream);
    else if (n == 12288)
        launch<12>(in, in_bf16, ld_in, rows, pre, round_in, transpose, out32, ld32, out16, ld16, post, stream);
    else if (n == 28672)
        launch<28>(in, in_bf16, ld_in, rows, pre, round_in, transpose, out32, ld32, out16, ld16, post, stream);
    else
        throw std::invalid_argument("had_rows supports n = 1024, 12288, 28672");
}
void rms_had(const uint16_t *x, long ld, int rows, int n, const float *w, float eps, const float *su, uint16_t *xn,
             long ldxn, float *r, uint16_t *u, long ldu, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (n != 12288 || ld % 4 || ldxn % 4 || ldu % 4)
        throw std::invalid_argument("rms_had supports n = 12288 with 4-element aligned rows");
    tables();
    rms_had_kernel<<<rows, 256, 0, stream>>>(x, ld, w, eps, su, xn, ldxn, r, u, ldu);
    check(hipGetLastError());
}
void gateup(const float *z, long ldz, int rows, int inter, const float *svs_g, const float *svs_u, uint16_t *gate,
            uint16_t *up, uint16_t *product, long ldo, float *cg, float *cu, long ldc, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (inter != 28672 || ldz % 4 || ldo % 4 || ldc % 4)
        throw std::invalid_argument("gateup supports intermediate 28672");
    tables();
    gateup_kernel<<<rows, 256, 0, stream>>>(z, ldz, svs_g, svs_u, gate, up, product, ldo, cg, cu, ldc);
    check(hipGetLastError());
}
void colsum(const float *a, long lda, const void *b, int b16, long ldb, const float *rs, int rows, int n,
            float *acc, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (n % 1024 || lda % 4 || ldb % 4)
        throw std::invalid_argument("colsum needs n % 1024 == 0");
    const int per = 64;
    const dim3 grid(unsigned(n / 1024), unsigned((rows + per - 1) / per));
    if (b16)
        colsum_kernel<true><<<grid, 256, 0, stream>>>(a, lda, b, ldb, rs, rows, per, acc);
    else
        colsum_kernel<false><<<grid, 256, 0, stream>>>(a, lda, b, ldb, rs, rows, per, acc);
    check(hipGetLastError());
}
void swiglu_bwd(const float *dxsu, const float *su, const uint16_t *gate, const uint16_t *up, float *dg, float *du,
                int rows, int cols, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (cols % 1024)
        throw std::invalid_argument("swiglu_bwd needs cols % 1024 == 0");
    swiglu_bwd_kernel<<<dim3(unsigned(cols / 1024), unsigned(rows)), 256, 0, stream>>>(dxsu, su, gate, up, dg, du, cols);
    check(hipGetLastError());
}
void rms_bwd(const uint16_t *x, long ldx, const float *r, const float *w, const float *su, const float *dxsu, long ldd,
             float *dh, long ldh, int rows, int n, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (n != 12288)
        throw std::invalid_argument("rms_bwd supports n = 12288");
    rms_bwd_kernel<<<rows, 256, 0, stream>>>(x, ldx, r, w, su, dxsu, ldd, dh, ldh);
    check(hipGetLastError());
}
void rope(const uint16_t *in, long ld, int rows, int heads, int head_dim, const float *cs, const float *sn, void *out,
          int backward, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (head_dim != 128)
        throw std::invalid_argument("rope kernel supports head_dim 128");
    if (backward)
        rope_kernel<true><<<dim3(unsigned(heads), unsigned(rows)), 64, 0, stream>>>(in, ld, cs, sn, out, heads);
    else
        rope_kernel<false><<<dim3(unsigned(heads), unsigned(rows)), 64, 0, stream>>>(in, ld, cs, sn, out, heads);
    check(hipGetLastError());
}
} // namespace qtip_training
