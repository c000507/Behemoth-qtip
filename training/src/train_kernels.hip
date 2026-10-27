// Native training row transforms and fused elementwise epilogues.
// One 256-thread block per row; thread t holds positions 4t..4t+3 of each 1024-chunk, so a row
// is read once and every output chunk is mixed from registers (as runtime/src/qtip.hip's row
// kernels). Inputs and outputs are FP32 or BF16 (activation storage); arithmetic is FP32.
#include "hip_ops.hpp"
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


__device__ __forceinline__ float rnd(float x) { return bf16(bf16(x)); }
__device__ __forceinline__ float4 ld4(const float *p) { return *reinterpret_cast<const float4 *>(p); }
__device__ __forceinline__ void load4v(const void *p, int half, size_t i, float out[4]) {
    if (half)
        load4<true>(p, i, out);
    else
        load4<false>(p, i, out);
}
__device__ __forceinline__ void store4v(void *p, int half, size_t i, const float v[4]) {
    if (half)
        *reinterpret_cast<uint2 *>(static_cast<uint16_t *>(p) + i) =
            make_uint2(uint32_t(bf16(v[0])) | uint32_t(bf16(v[1])) << 16,
                       uint32_t(bf16(v[2])) | uint32_t(bf16(v[3])) << 16);
    else
        *reinterpret_cast<float4 *>(static_cast<float *>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
}

// See qtip_training::RowTransform. The output transform of every projection
// (GEMM output -> saved c, scaled/rounded y, optional residual add) and every
// input transform (activation or gradient -> BF16 GEMM operand) is one launch.
template <int K, bool IN_HALF>
__global__ __launch_bounds__(256) void transform_kernel(qtip_training::RowTransform a) {
    __shared__ float scratch[1024];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    constexpr int n = K * 1024;
    const float pre_scale = a.pre_scale ? *a.pre_scale : 1.f, post_scale = a.post_scale ? *a.post_scale : 1.f;
    float t[K][4];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        load4<IN_HALF>(a.in, size_t(row) * size_t(a.ld_in) + size_t(k * 1024 + j), t[k]);
        if (a.round_in) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                t[k][i] = rnd(t[k][i]);
        }
        if (a.pre) {
            const float4 s = ld4(a.pre + k * 1024 + j);
            t[k][0] *= s.x * pre_scale;
            t[k][1] *= s.y * pre_scale;
            t[k][2] *= s.z * pre_scale;
            t[k][3] *= s.w * pre_scale;
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
                const float s = a.transpose ? sign<K>(k, k2) : sign<K>(k2, k);
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
        if (a.c)
            store4v(a.c, a.c_half, size_t(row) * size_t(a.ldc) + size_t(e), v);
        if (a.y) {
            float w[4] = {v[0], v[1], v[2], v[3]};
            if (a.post) {
                const float4 s = ld4(a.post + e);
                w[0] *= s.x * post_scale;
                w[1] *= s.y * post_scale;
                w[2] *= s.z * post_scale;
                w[3] *= s.w * post_scale;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                w[i] = rnd(w[i]);
            if (a.res) {
                float r[4];
                load4v(a.res, a.res_half, size_t(row) * size_t(a.ldres) + size_t(e), r);
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    w[i] = rnd(r[i] + w[i]);
            }
            store4v(a.y, a.y_half, size_t(row) * size_t(a.ldy) + size_t(e), w);
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

// Fused RMSNorm + input transform: xn = bf16((x * r) * w), r = rsqrt(mean(x^2) + eps),
// u = bf16(hadUt(su (.) xn) / sqrt(n)). Optional xn / r outputs (backward needs them). grid (rows).
template <int K, bool IN_HALF>
__global__ __launch_bounds__(256) void rms_had_kernel(const void *x, long ld, const float *w, float eps,
                                                      const float *su, void *xn, int xn_half, long ldxn,
                                                      float *r, uint16_t *u, long ldu) {
    constexpr int n = K * 1024;
    __shared__ float scratch[1024];
    __shared__ float red[8];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    float t[K][4];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        load4<IN_HALF>(x, size_t(row) * size_t(ld) + size_t(k * 1024 + j), t[k]);
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
            y[i] = rnd(t[k][i] * factor * ww[i]);
        if (xn)
            store4v(xn, xn_half, size_t(row) * size_t(ldxn) + size_t(e), y);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            t[k][i] = ss4[i] * y[i];
    }
    if (r && threadIdx.x == 0)
        r[row] = factor;
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
                const float s = sign<K>(k, k2); // hadUt
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    v[i] += s * t[k][i];
            }
        }
        fwht1024(v, scratch);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] *= scale;
        store4v(u, 1, size_t(row) * size_t(ldu) + size_t(k2 * 1024 + j), v);
    }
}

// SwiGLU backward: dp = dxsu * su; dg = dp * u * silu'(g); du = dp * silu(g).
// Elementwise over [rows][cols]; grid (cols / 1024, rows).
__global__ __launch_bounds__(256) void swiglu_bwd_kernel(const void *dxsu, int dxsu_half, long ldd,
                                                         const float *su, const void *gate, int gate_half,
                                                         long ldg, const void *up, int up_half, long ldu,
                                                         void *dg, void *du, int out_half, long ldo) {
    const size_t row = blockIdx.y;
    const int e = int(blockIdx.x) * 1024 + 4 * int(threadIdx.x);
    float d[4], g[4], u[4];
    load4v(dxsu, dxsu_half, row * size_t(ldd) + size_t(e), d);
    load4v(gate, gate_half, row * size_t(ldg) + size_t(e), g);
    load4v(up, up_half, row * size_t(ldu) + size_t(e), u);
    const float4 s = ld4(su + e);
    const float dp[4] = {d[0] * s.x, d[1] * s.y, d[2] * s.z, d[3] * s.w};
    float a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float sg = 1.f / (1.f + expf(-g[i]));
        a[i] = dp[i] * u[i] * sg * (1.f + g[i] * (1.f - sg));
        b[i] = dp[i] * g[i] * sg;
    }
    store4v(dg, out_half, row * size_t(ldo) + size_t(e), a);
    store4v(du, out_half, row * size_t(ldo) + size_t(e), b);
}

// RMSNorm backward, row part: dxhat = dxsu * su * w, xhat = x * r,
// dh += r * (dxhat - xhat * mean(dxhat * xhat)). grid (rows).
template <int K>
__global__ __launch_bounds__(256) void rms_bwd_kernel(const void *x, int x_half, long ldx, const float *r,
                                                      const float *w, const float *su, const void *dxsu,
                                                      int dxsu_half, long ldd, float *dh, long ldh) {
    constexpr int n = K * 1024;
    __shared__ float red[8];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    const float rr = r[row];
    float xh[K][4], dx[K][4];
    float dot = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int e = k * 1024 + j;
        float d[4];
        load4v(x, x_half, size_t(row) * size_t(ldx) + size_t(e), xh[k]);
        load4v(dxsu, dxsu_half, size_t(row) * size_t(ldd) + size_t(e), d);
        const float4 s = ld4(su + e), ww = ld4(w + e);
        dx[k][0] = d[0] * s.x * ww.x;
        dx[k][1] = d[1] * s.y * ww.y;
        dx[k][2] = d[2] * s.z * ww.z;
        dx[k][3] = d[3] * s.w * ww.w;
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

template <int K> void launch(const qtip_training::RowTransform &a, int rows, hipStream_t stream) {
    if (a.in_half)
        transform_kernel<K, true><<<rows, 256, 0, stream>>>(a);
    else
        transform_kernel<K, false><<<rows, 256, 0, stream>>>(a);
    check(hipGetLastError());
}
bool aligned(const void *p, size_t a) { return (reinterpret_cast<uintptr_t>(p) % a) == 0; }
size_t width(int half) { return half ? 8 : 16; }
} // namespace

namespace qtip_training {
bool transform_supported(int n) { return n == 1024 || n == 12288 || n == 28672; }
void transform(const RowTransform &a, int rows, int n, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (!a.c && !a.y)
        throw std::invalid_argument("row transform needs an output");
    if (!aligned(a.in, width(a.in_half)) || a.ld_in % 4 || a.ldc % 4 || a.ldy % 4 || a.ldres % 4 ||
        (a.c && !aligned(a.c, width(a.c_half))) || (a.y && !aligned(a.y, width(a.y_half))) ||
        (a.res && !aligned(a.res, width(a.res_half))) || (a.pre && !aligned(a.pre, 16)) ||
        (a.post && !aligned(a.post, 16)))
        throw std::invalid_argument("row transform alignment");
    tables();
    if (n == 1024)
        launch<1>(a, rows, stream);
    else if (n == 12288)
        launch<12>(a, rows, stream);
    else if (n == 28672)
        launch<28>(a, rows, stream);
    else
        throw std::invalid_argument("row transform supports n = 1024, 12288, 28672");
}
void had_rows(const void *in, int in_bf16, long ld_in, int rows, int n, int transpose, const float *pre,
              int round_in, float *out32, long ld32, uint16_t *out16, long ld16, const float *post,
              hipStream_t stream) {
    RowTransform a;
    a.in = in;
    a.in_half = in_bf16;
    a.ld_in = ld_in;
    a.transpose = transpose;
    a.pre = pre;
    a.round_in = round_in;
    a.c = out32;
    a.ldc = ld32;
    a.y = out16;
    a.ldy = ld16;
    a.y_half = 1;
    a.post = post;
    transform(a, rows, n, stream);
}
void rms_had(const void *x, int x_half, long ld, int rows, int n, const float *w, float eps, const float *su,
             void *xn, int xn_half, long ldxn, float *r, uint16_t *u, long ldu, hipStream_t stream) {
    if (rows <= 0)
        return;
    if ((n != 12288 && n != 1024) || ld % 4 || ldxn % 4 || ldu % 4 || !aligned(x, width(x_half)) ||
        (xn && !aligned(xn, width(xn_half))) || !aligned(u, 8) || !aligned(w, 16) || !aligned(su, 16))
        throw std::invalid_argument("rms_had supports n = 1024 or 12288 with 4-element aligned rows");
    tables();
    if (n == 1024) {
        if (x_half)
            rms_had_kernel<1, true><<<rows, 256, 0, stream>>>(x, ld, w, eps, su, xn, xn_half, ldxn, r, u, ldu);
        else
            rms_had_kernel<1, false><<<rows, 256, 0, stream>>>(x, ld, w, eps, su, xn, xn_half, ldxn, r, u, ldu);
    } else {
        if (x_half)
            rms_had_kernel<12, true><<<rows, 256, 0, stream>>>(x, ld, w, eps, su, xn, xn_half, ldxn, r, u, ldu);
        else
            rms_had_kernel<12, false>
                <<<rows, 256, 0, stream>>>(x, ld, w, eps, su, xn, xn_half, ldxn, r, u, ldu);
    }
    check(hipGetLastError());
}
void swiglu_bwd(const void *dxsu, int dxsu_half, long ldd, const float *su, const void *gate, int gate_half,
                long ldg, const void *up, int up_half, long ldu, void *dg, void *du, int out_half, long ldo,
                int rows, int cols, hipStream_t stream) {
    if (rows <= 0)
        return;
    if (cols % 1024 || ldd % 4 || ldg % 4 || ldu % 4 || ldo % 4)
        throw std::invalid_argument("swiglu_bwd needs cols % 1024 == 0 and aligned rows");
    swiglu_bwd_kernel<<<dim3(unsigned(cols / 1024), unsigned(rows)), 256, 0, stream>>>(
        dxsu, dxsu_half, ldd, su, gate, gate_half, ldg, up, up_half, ldu, dg, du, out_half, ldo);
    check(hipGetLastError());
}
void rms_bwd(const void *x, int x_half, long ldx, const float *r, const float *w, const float *su,
             const void *dxsu, int dxsu_half, long ldd, float *dh, long ldh, int rows, int n,
             hipStream_t stream) {
    if (rows <= 0)
        return;
    if ((n != 12288 && n != 1024) || ldx % 4 || ldd % 4 || ldh % 4)
        throw std::invalid_argument("rms_bwd supports n = 1024 or 12288");
    if (n == 1024)
        rms_bwd_kernel<1><<<rows, 256, 0, stream>>>(x, x_half, ldx, r, w, su, dxsu, dxsu_half, ldd, dh, ldh);
    else
        rms_bwd_kernel<12><<<rows, 256, 0, stream>>>(x, x_half, ldx, r, w, su, dxsu, dxsu_half, ldd, dh, ldh);
    check(hipGetLastError());
}
} // namespace qtip_training
