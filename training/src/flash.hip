// Training attention: bounded-memory 16-query / 32-key BF16 WMMA tiles.
// Q/K/V stay token-major; GQA and packed-document boundaries are explicit.
#include "../../runtime/src/device_numeric.hpp"
#include "hip_ops.hpp"
#include <cmath>
#include <cstdlib>
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
        if (wave < 2)
            product<false, true>(q, 128, k + wave * 16 * 128, 128, scores + wave * 16, 32, 128);
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
using Acc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
using NativeReg = __bf16 __attribute__((ext_vector_type(8)));
using NativeAcc = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ NativeAcc native_mma(NativeReg a, NativeReg b, NativeAcc c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ int native_kbase(int s, int h) { return 32 * (s >> 1) + 16 * h + 8 * (s & 1); }
// A wave owns one query head and 16 queries. All heads of a GQA group share
// staged K/V. Transposed products keep scores, softmax, and O in registers;
// native vector operand loads avoid rocWMMA's elementwise LDS gathers.
__global__ void forward_native(View qv, View kv, View vv, View ov, View lv, int group,
                               const int32_t *starts) {
    constexpr int KROW = 256, VROW = 72, KBYTES = 32 * KROW;
    __shared__ __align__(16) unsigned char shared[KBYTES + 128 * VROW];
    int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    int c = lane % 16, h = lane / 16, first = blockIdx.x * 16;
    int kh = blockIdx.y, head = kh * group + wave, query = first + c, T = qv.rows;
    bool live = query < T;
    int doc = live ? starts[query] : T;
    NativeReg qr[8];
    // Reuse the K/V storage for four Q heads at a time, then keep Q in VGPRs.
    for (int lo = 0; lo < group; lo += 4) {
        int count = group - lo < 4 ? group - lo : 4;
        for (int i = tid; i < count * 16 * 128; i += blockDim.x) {
            int w = i / (16 * 128), row = i / 128 % 16, d = i % 128;
            reinterpret_cast<uint16_t *>(shared)[i] =
                first + row < T ? bf16(val(qv, first + row, kh * group + lo + w, d)) : 0;
        }
        __syncthreads();
        if (wave >= lo && wave < lo + count) {
            const unsigned char *row = shared + ((wave - lo) * 16 + c) * KROW;
#pragma unroll
            for (int s = 0; s < 8; ++s)
                qr[s] = __builtin_bit_cast(NativeReg,
                                           *reinterpret_cast<const uint4 *>(row + native_kbase(s, h) * 2));
        }
        __syncthreads();
    }
    NativeAcc acc[8]{};
    float mx = -INFINITY, den = 0;
    int limit = first + 16 < T ? first + 16 : T;
    for (int base = starts[first]; base < limit; base += 32) {
        for (int i = tid; i < 32 * 128; i += blockDim.x) {
            int key = i / 128, d = i % 128, t = base + key;
            uint16_t k = t < limit ? bf16(val(kv, t, kh, d)) : 0;
            uint16_t v = t < limit ? bf16(val(vv, t, kh, d)) : 0;
            *reinterpret_cast<uint16_t *>(shared + key * KROW + ((d * 2) ^ ((key & 7) * 16))) = k;
            *reinterpret_cast<uint16_t *>(shared + KBYTES + d * VROW + key * 2) = v;
        }
        __syncthreads();
        NativeAcc sc[2]{};
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            int key = 16 * (c >> 3) + 8 * n + (c & 7);
            const unsigned char *row = shared + key * KROW;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                auto kr = __builtin_bit_cast(
                    NativeReg,
                    *reinterpret_cast<const uint4 *>(row + ((native_kbase(s, h) * 2) ^ ((key & 7) * 16))));
                sc[n] = native_mma(kr, qr[s], sc[n]);
            }
        }
        float top = -INFINITY;
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int key = base + 16 * h + 8 * n + i;
                float score = live && key >= doc && key <= query
                                  ? sc[n][i] * (0.08838834764831845f * 1.4426950408889634f)
                                  : -INFINITY;
                sc[n][i] = score;
                top = fmaxf(top, score);
            }
        top = fmaxf(top, __shfl_xor(top, 16));
        float next = fmaxf(mx, top);
        float alpha = mx == next ? 1.f : __builtin_amdgcn_exp2f(mx - next);
        float shift = isfinite(next) ? next : 0.f;
        NativeReg pr[2];
        float sum = 0;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            uint16_t bits[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float p = __builtin_amdgcn_exp2f(sc[n][i] - shift);
                sum += p;
                bits[i] = bf16(p);
            }
            pr[n] = __builtin_bit_cast(NativeReg, bits);
        }
        sum += __shfl_xor(sum, 16);
        den = den * alpha + sum;
        mx = next;
#pragma unroll
        for (int nd = 0; nd < 8; ++nd) {
            NativeAcc out = acc[nd];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                out[i] *= alpha;
            const unsigned char *row = shared + KBYTES + (16 * nd + c) * VROW;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const unsigned char *p = row + (16 * h + 8 * n) * 2;
                uint2 lo = *reinterpret_cast<const uint2 *>(p), hi = *reinterpret_cast<const uint2 *>(p + 8);
                auto vr = __builtin_bit_cast(NativeReg, make_uint4(lo.x, lo.y, hi.x, hi.y));
                out = native_mma(vr, pr[n], out);
            }
            acc[nd] = out;
        }
        __syncthreads();
    }
    if (live) {
        float inv = 1.f / den;
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                ov.p[size_t(query) * ov.ld + head * 128 + 16 * nd + 8 * h + i] =
                    __uint_as_float(uint32_t(bf16(acc[nd][i] * inv)) << 16);
        if (h == 0)
            lv.p[size_t(head) * lv.ld + query] = mx * 0.6931471805599453f + logf(den);
    }
}
// Accumulate two 16-wide reduction tiles without spilling the output to LDS.
__device__ void accumulate(const uint16_t *a, const uint16_t *b, Acc &out) {
    for (int i = 0; i < 32; i += 16) {
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, BF, rocwmma::row_major> aa;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, BF, rocwmma::row_major> bb;
        rocwmma::load_matrix_sync(aa, reinterpret_cast<const BF *>(a + i), 32);
        rocwmma::load_matrix_sync(bb, reinterpret_cast<const BF *>(b + i * 128), 128);
        rocwmma::mma_sync(out, aa, bb, out);
    }
}
__global__ void backward_q(View qv, View kv, View vv, View lv, View dy, View dqv, View delta_v, int heads,
                           int kvheads, const int32_t *starts) {
    __shared__ __align__(16) uint16_t q[16 * 128], dout[16 * 128], k[32 * 128], v[32 * 128], ds[16 * 32];
    __shared__ __align__(16) float scores[16 * 32], dp[16 * 32];
    int tid = threadIdx.x, wave = tid / 32, head = blockIdx.y, kh = head / (heads / kvheads),
        first = blockIdx.x * 16, T = qv.rows;
    Acc dq[2];
    for (int j = 0; j < 2; ++j)
        rocwmma::fill_fragment(dq[j], 0.f);
    for (int i = tid; i < 16 * 128; i += 128) {
        int t = first + i / 128, d = i % 128;
        q[i] = t < T ? bf16(val(qv, t, head, d)) : 0;
        dout[i] = t < T ? bf16(val(dy, t, head, d)) : 0;
    }
    __syncthreads();
    for (int key = starts[first]; key < (first + 16 < T ? first + 16 : T); key += 32) {
        for (int i = tid; i < 32 * 128; i += 128) {
            int t = key + i / 128, d = i % 128;
            k[i] = t < T ? bf16(val(kv, t, kh, d)) : 0;
            v[i] = t < T ? bf16(val(vv, t, kh, d)) : 0;
        }
        __syncthreads();
        if (wave < 2)
            product<false, true>(q, 128, k + wave * 16 * 128, 128, scores + wave * 16, 32, 128);
        else
            product<false, true>(dout, 128, v + (wave - 2) * 16 * 128, 128, dp + (wave - 2) * 16, 32, 128);
        __syncthreads();
        for (int i = tid; i < 16 * 32; i += 128) {
            int t = first + i / 32, j = key + i % 32;
            float prob = t < T && j <= t && j >= starts[t]
                             ? expf(scores[i] * .08838834764831845f - lv.p[size_t(head) * lv.ld + t])
                             : 0;
            float delta = t < T ? delta_v.p[size_t(head) * delta_v.ld + t] : 0;
            ds[i] = bf16(prob * (dp[i] - delta) * .08838834764831845f);
        }
        __syncthreads();
        for (int j = 0; j < 2; ++j)
            accumulate(ds, k + (wave + j * 4) * 16, dq[j]);
        __syncthreads();
    }
    float *tile = reinterpret_cast<float *>(k);
    for (int j = 0; j < 2; ++j)
        rocwmma::store_matrix_sync(tile + (wave + j * 4) * 16, dq[j], 128, rocwmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 16 * 128; i += 128) {
        int t = first + i / 128;
        if (t < T)
            dqv.p[size_t(t) * dqv.ld + head * 128 + i % 128] = tile[i];
    }
}
// One wave computes each softmax row's correction once for both backward passes.
__global__ void backward_delta(View out, View dy, View delta, int heads) {
    int lane = threadIdx.x % 32;
    size_t row = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= size_t(out.rows) * heads)
        return;
    int t = row / heads, h = row % heads;
    float sum = 0;
    for (int d = lane; d < 128; d += 32)
        sum += val(out, t, h, d) * val(dy, t, h, d);
    for (int offset = 16; offset; offset /= 2)
        sum += __shfl_down(sum, offset);
    if (lane == 0)
        delta.p[size_t(h) * delta.ld + t] = sum;
}

// Each CTA owns 16 keys for one KV head. It reduces all associated query heads
// and causal query rows in registers, then writes dK/dV once. No atomics and no
// sequence-sized per-query-head gradient workspace are needed.
__global__ void backward_kv(View qv, View kv, View vv, View lv, View dy, View delta_v, View dkv, View dvv,
                            int heads, int kvheads, const int32_t *starts, const int32_t *ends) {
    __shared__ __align__(16) uint16_t k[16 * 128], v[16 * 128], q[32 * 128], dout[32 * 128], p[16 * 32],
        ds[16 * 32];
    __shared__ __align__(16) float scores[16 * 32], dp[16 * 32];
    int tid = threadIdx.x, wave = tid / 32, kh = blockIdx.y, first = blockIdx.x * 16, T = qv.rows;
    Acc dk[2], dv[2];
    for (int j = 0; j < 2; ++j) {
        rocwmma::fill_fragment(dk[j], 0.f);
        rocwmma::fill_fragment(dv[j], 0.f);
    }
    for (int i = tid; i < 16 * 128; i += 128) {
        int t = first + i / 128, d = i % 128;
        k[i] = t < T ? bf16(val(kv, t, kh, d)) : 0;
        v[i] = t < T ? bf16(val(vv, t, kh, d)) : 0;
    }
    // A key tile can cross document boundaries: use the last key's document
    // end, then apply the per-query start mask inside the tile.
    int end = ends[first + 15 < T ? first + 15 : T - 1];
    int group = heads / kvheads;
    for (int head = kh * group; head < (kh + 1) * group; ++head) {
        for (int query = first; query < end; query += 32) {
            for (int i = tid; i < 32 * 128; i += 128) {
                int t = query + i / 128, d = i % 128;
                q[i] = t < end ? bf16(val(qv, t, head, d)) : 0;
                dout[i] = t < end ? bf16(val(dy, t, head, d)) : 0;
            }
            __syncthreads();
            if (wave < 2)
                product<false, true>(k, 128, q + wave * 16 * 128, 128, scores + wave * 16, 32, 128);
            else
                product<false, true>(v, 128, dout + (wave - 2) * 16 * 128, 128, dp + (wave - 2) * 16, 32,
                                     128);
            __syncthreads();
            for (int i = tid; i < 16 * 32; i += 128) {
                int key = first + i / 32, t = query + i % 32;
                bool valid = key < T && t < end && key <= t && key >= starts[t];
                float prob =
                    valid ? expf(scores[i] * .08838834764831845f - lv.p[size_t(head) * lv.ld + t]) : 0;
                float delta = t < end ? delta_v.p[size_t(head) * delta_v.ld + t] : 0;
                p[i] = bf16(prob);
                ds[i] = bf16(prob * (dp[i] - delta) * .08838834764831845f);
            }
            __syncthreads();
            for (int j = 0; j < 2; ++j) {
                int col = (wave + j * 4) * 16;
                accumulate(ds, q + col, dk[j]);
                accumulate(p, dout + col, dv[j]);
            }
            __syncthreads();
        }
    }
    // Reuse the now-dead query staging buffer for guarded stores at the tail.
    float *tile = reinterpret_cast<float *>(q);
    for (int j = 0; j < 2; ++j)
        rocwmma::store_matrix_sync(tile + (wave + j * 4) * 16, dk[j], 128, rocwmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 16 * 128; i += 128)
        if (first + i / 128 < T)
            dkv.p[size_t(first + i / 128) * dkv.ld + kh * 128 + i % 128] = tile[i];
    __syncthreads();
    for (int j = 0; j < 2; ++j)
        rocwmma::store_matrix_sync(tile + (wave + j * 4) * 16, dv[j], 128, rocwmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 16 * 128; i += 128)
        if (first + i / 128 < T)
            dvv.p[size_t(first + i / 128) * dvv.ld + kh * 128 + i % 128] = tile[i];
}

void flash(View q, View k, View v, View out, View lse, int h, int kh, const int32_t *b, hipStream_t s) {
    const char *setting = std::getenv("QTIP_TRAIN_ATTN_FORWARD");
    bool legacy = setting && std::string(setting) == "legacy";
    require(!setting || legacy || std::string(setting) == "native",
            "QTIP_TRAIN_ATTN_FORWARD must be native or legacy");
    int group = h / kh;
    if (!legacy && group <= 32)
        forward_native<<<dim3((q.rows + 15) / 16, kh), group * 32, 0, s>>>(q, k, v, out, lse, group, b);
    else
        forward<<<dim3((q.rows + 15) / 16, h), 128, 0, s>>>(q, k, v, out, lse, h, kh, b);
}
void flash_back(View q, View k, View v, View out, View lse, View dy, View dq, View dk, View dv, View delta,
                int h, int kh, const int32_t *b, const int32_t *ends, hipStream_t s) {
    backward_delta<<<(size_t(q.rows) * h + 7) / 8, 256, 0, s>>>(out, dy, delta, h);
    const char *setting = std::getenv("QTIP_TRAIN_ATTN_BACKWARD");
    bool legacy = setting && std::string(setting) == "legacy";
    require(!setting || legacy || std::string(setting) == "native",
            "QTIP_TRAIN_ATTN_BACKWARD must be native or legacy");
    if (!legacy && h / kh <= 32) {
        flash_back_native(q, k, v, lse, dy, dq, dk, dv, delta, h, kh, b, ends, s);
        return;
    }
    backward_q<<<dim3((q.rows + 15) / 16, h), 128, 0, s>>>(q, k, v, lse, dy, dq, delta, h, kh, b);
    backward_kv<<<dim3((q.rows + 15) / 16, kh), 128, 0, s>>>(q, k, v, lse, dy, delta, dk, dv, h, kh, b, ends);
}
} // namespace qat::gpu
