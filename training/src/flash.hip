// Training attention: bounded-memory 16-query / 32-key BF16 WMMA tiles.
// Q/K/V stay token-major; GQA and packed-document boundaries are explicit.
#include "../../runtime/src/device_numeric.hpp"
#include "hip_ops.hpp"
#include <cmath>
#include <rocwmma/rocwmma.hpp>
namespace qat::gpu {
using BF = rocwmma::bfloat16_t;
using behemoth::device::bf16;
#if defined(__HIP_DEVICE_COMPILE__)
static_assert(rocwmma::Constants::AMDGCN_WAVE_SIZE == 32);
#endif
__device__ float val(View x, int t, int h, int d) { return x.p[size_t(t) * x.ld + h * 128 + d]; }
// C16x16 = op(A)16xK * op(B)Kx16, with general leading dimensions.
template <bool TA, bool TB>
__device__ void product(const uint16_t *a, int lda, const uint16_t *b, int ldb, float *c, int ldc, int k) {
    using LA = std::conditional_t<TA, rocwmma::col_major, rocwmma::row_major>;
    using LB = std::conditional_t<TB, rocwmma::col_major, rocwmma::row_major>;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> out;
    rocwmma::fill_fragment(out, 0.f);
    for (int i = 0; i < k; i += 16) {
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, BF, LA> aa;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, BF, LB> bb;
        rocwmma::load_matrix_sync(aa, reinterpret_cast<const BF *>(a) + (TA ? i * lda : i), lda);
        rocwmma::load_matrix_sync(bb, reinterpret_cast<const BF *>(b) + (TB ? i : i * ldb), ldb);
        rocwmma::mma_sync(out, aa, bb, out);
    }
    rocwmma::store_matrix_sync(c, out, ldc, rocwmma::mem_row_major);
}
__global__ void forward(View qv, View kv, View vv, View ov, View lv, int heads, int kvheads,
                        const int32_t *starts) {
    __shared__ __align__(16) uint16_t q[16 * 128], k[32 * 128], v[32 * 128], p[16 * 32];
    __shared__ __align__(16) float scores[16 * 32], tile[16 * 128], acc[16 * 128], mx[16], den[16],
        rescale[16];
    int tid = threadIdx.x, wave = tid / 32, head = blockIdx.y, kh = head / (heads / kvheads),
        first = blockIdx.x * 16, T = qv.rows;
    for (int i = tid; i < 16 * 128; i += 128) {
        int r = i / 128, d = i % 128;
        q[i] = first + r < T ? bf16(val(qv, first + r, head, d)) : 0;
        acc[i] = 0;
    }
    if (tid < 16) {
        mx[tid] = -INFINITY;
        den[tid] = 0;
    }
    __syncthreads();
    // Start at the earliest document represented by the query tile.
    int begin = starts[first];
    for (int key = begin; key < (first + 16 < T ? first + 16 : T); key += 32) {
        for (int i = tid; i < 32 * 128; i += 128) {
            int t = key + i / 128, d = i % 128;
            k[i] = t < T ? bf16(val(kv, t, kh, d)) : 0;
            v[i] = t < T ? bf16(val(vv, t, kh, d)) : 0;
        }
        __syncthreads();
        if (wave == 0) {
            product<false, true>(q, 128, k, 128, scores, 32, 128);
            product<false, true>(q, 128, k + 16 * 128, 128, scores + 16, 32, 128);
        }
        __syncthreads();
        if (tid < 16) {
            int t = first + tid;
            float top = -INFINITY;
            for (int j = 0; j < 32; ++j) {
                bool valid = t < T && key + j <= t && key + j >= starts[t];
                float s = valid ? scores[tid * 32 + j] * .08838834764831845f : -INFINITY;
                scores[tid * 32 + j] = s;
                top = fmaxf(top, s);
            }
            float next = fmaxf(mx[tid], top);
            float alpha = isfinite(mx[tid]) ? expf(mx[tid] - next) : 0;
            rescale[tid] = alpha;
            float ss = 0;
            for (int j = 0; j < 32; ++j) {
                float prob = isfinite(scores[tid * 32 + j]) ? expf(scores[tid * 32 + j] - next) : 0;
                p[tid * 32 + j] = bf16(prob);
                ss += prob;
            }
            den[tid] = den[tid] * alpha + ss;
            mx[tid] = next;
        }
        __syncthreads();
        for (int col = wave; col < 8; col += 4)
            product<false, false>(p, 32, v + col * 16, 128, tile + col * 16, 128, 32);
        __syncthreads();
        for (int i = tid; i < 16 * 128; i += 128)
            acc[i] = acc[i] * rescale[i / 128] + tile[i];
        __syncthreads();
    }
    for (int i = tid; i < 16 * 128; i += 128) {
        int t = first + i / 128;
        if (t < T)
            ov.p[size_t(t) * ov.ld + head * 128 + i % 128] =
                __uint_as_float(uint32_t(bf16(acc[i] / den[i / 128])) << 16);
    }
    if (tid < 16 && first + tid < T)
        lv.p[size_t(head) * lv.ld + first + tid] = mx[tid] + logf(den[tid]);
}
__global__ void backward(View qv, View kv, View vv, View ov, View lv, View dy, View dqv, View dkv, View dvv,
                         int heads, int kvheads, const int32_t *starts) {
    __shared__ __align__(16) uint16_t q[16 * 128], dout[16 * 128], k[32 * 128], v[32 * 128], p[16 * 32],
        ds[16 * 32];
    __shared__ __align__(16) float scores[16 * 32], dp[16 * 32], dq[16 * 128], tile[32 * 128], delta[16];
    int tid = threadIdx.x, wave = tid / 32, head = blockIdx.y, kh = head / (heads / kvheads),
        first = blockIdx.x * 16, T = qv.rows;
    for (int i = tid; i < 16 * 128; i += 128) {
        int t = first + i / 128, d = i % 128;
        q[i] = t < T ? bf16(val(qv, t, head, d)) : 0;
        dout[i] = t < T ? bf16(val(dy, t, head, d)) : 0;
        dq[i] = 0;
    }
    if (tid < 16) {
        int t = first + tid;
        float dot = 0;
        if (t < T)
            for (int d = 0; d < 128; ++d)
                dot += val(dy, t, head, d) * val(ov, t, head, d);
        delta[tid] = dot;
    }
    __syncthreads();
    for (int key = starts[first]; key < (first + 16 < T ? first + 16 : T); key += 32) {
        for (int i = tid; i < 32 * 128; i += 128) {
            int t = key + i / 128, d = i % 128;
            k[i] = t < T ? bf16(val(kv, t, kh, d)) : 0;
            v[i] = t < T ? bf16(val(vv, t, kh, d)) : 0;
        }
        __syncthreads();
        if (wave == 0) {
            product<false, true>(q, 128, k, 128, scores, 32, 128);
            product<false, true>(q, 128, k + 16 * 128, 128, scores + 16, 32, 128);
            product<false, true>(dout, 128, v, 128, dp, 32, 128);
            product<false, true>(dout, 128, v + 16 * 128, 128, dp + 16, 32, 128);
        }
        __syncthreads();
        for (int i = tid; i < 16 * 32; i += 128) {
            int t = first + i / 32, j = key + i % 32;
            float prob = t < T && j <= t && j >= starts[t]
                             ? expf(scores[i] * .08838834764831845f - lv.p[size_t(head) * lv.ld + t])
                             : 0;
            p[i] = bf16(prob);
            ds[i] = bf16(prob * (dp[i] - delta[i / 32]) * .08838834764831845f);
        }
        __syncthreads();
        for (int col = wave; col < 8; col += 4)
            product<false, false>(ds, 32, k + col * 16, 128, tile + col * 16, 128, 32);
        __syncthreads();
        for (int i = tid; i < 16 * 128; i += 128)
            dq[i] += tile[i];
        __syncthreads();
        for (int block = 0; block < 2; ++block)
            for (int col = wave; col < 8; col += 4)
                product<true, false>(ds + block * 16, 32, q + col * 16, 128,
                                     tile + block * 16 * 128 + col * 16, 128, 16);
        __syncthreads();
        for (int i = tid; i < 32 * 128; i += 128) {
            int t = key + i / 128;
            if (t < T)
                atomicAdd(dkv.p + size_t(t) * dkv.ld + kh * 128 + i % 128, tile[i]);
        }
        __syncthreads();
        for (int block = 0; block < 2; ++block)
            for (int col = wave; col < 8; col += 4)
                product<true, false>(p + block * 16, 32, dout + col * 16, 128,
                                     tile + block * 16 * 128 + col * 16, 128, 16);
        __syncthreads();
        for (int i = tid; i < 32 * 128; i += 128) {
            int t = key + i / 128;
            if (t < T)
                atomicAdd(dvv.p + size_t(t) * dvv.ld + kh * 128 + i % 128, tile[i]);
        }
        __syncthreads();
    }
    for (int i = tid; i < 16 * 128; i += 128) {
        int t = first + i / 128;
        if (t < T)
            dqv.p[size_t(t) * dqv.ld + head * 128 + i % 128] = dq[i];
    }
}
void flash(View q, View k, View v, View out, View lse, int h, int kh, const int32_t *b, hipStream_t s) {
    forward<<<dim3((q.rows + 15) / 16, h), 128, 0, s>>>(q, k, v, out, lse, h, kh, b);
}
void flash_back(View q, View k, View v, View out, View lse, View dy, View dq, View dk, View dv, int h, int kh,
                const int32_t *b, hipStream_t s) {
    backward<<<dim3((q.rows + 15) / 16, h), 128, 0, s>>>(q, k, v, out, lse, dy, dq, dk, dv, h, kh, b);
}
} // namespace qat::gpu
