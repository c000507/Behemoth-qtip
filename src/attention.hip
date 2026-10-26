#include "behemoth/gpu.hpp"
#include "device_numeric.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <rocwmma/rocwmma.hpp>
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
static_assert(rocwmma::Constants::AMDGCN_WAVE_SIZE == 32, "rocWMMA must use Wave32");
#endif

namespace behemoth {
namespace {
using F8 = rocwmma::float8_t;
static_assert(sizeof(F8) == 1);
__device__ F8 f8(uint8_t x) {
    return __builtin_bit_cast(F8, x);
}
struct Meta {
    int qh, kh, pages, stride, requests, queries, max_q;
    const int *offsets, *pos, *len, *table;
};
Meta meta(const GpuMetadata &m) {
    const auto &l = m.layout();
    return {l.query_heads,      l.kv_heads,       l.physical_pages, l.table_stride,
            l.requests(),       l.queries(),      m.max_queries(),  m.offsets,
            m.positions,        m.lengths,        m.table};
}
using BF = rocwmma::bfloat16_t;
static_assert(sizeof(BF) == 2);
// Operand storage for the matrix tiles: FP8 bytes by default; BF16 when the
// query/probability precision option is active (K/V are then widened exactly
// from their FP8 cache values, so only Q and P gain precision).
template <bool BF16> struct Operand {
    using type = F8;
    using raw = uint8_t;
    static constexpr int block_k = 32;
    __device__ static raw from_fp8(uint8_t x) {
        return x;
    }
    __device__ static raw zero() {
        return 0;
    }
};
template <> struct Operand<true> {
    using type = BF;
    using raw = uint16_t;
    static constexpr int block_k = 16;
    __device__ static raw from_fp8(uint8_t x) {
        return device::bf16(device::fp8(x));
    }
    __device__ static raw zero() {
        return 0;
    }
};
template <bool BF16, class Raw>
__device__ inline void load_query_element(const AttentionDeviceView &a, size_t index, Raw &out) {
    static_assert(!BF16, "BF16 queries are not wired yet");
    out = a.q[index];
}
// scores[16 x 32] = Q[16 x 128] . K[32 x 128]^T for key columns [n, n+16).
template <bool BF16, class Raw>
__device__ inline void score_fragment(const Raw *q, const Raw *k, float *scores, int n) {
    using T = typename Operand<BF16>::type;
    constexpr int KS = Operand<BF16>::block_k;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, KS, float> c;
    rocwmma::fill_fragment(c, 0.f);
    for (int d = 0; d < 128; d += KS) {
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, KS, T, rocwmma::row_major> qa;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, KS, T, rocwmma::col_major> kb;
        rocwmma::load_matrix_sync(qa, reinterpret_cast<const T *>(q) + d, 128);
        rocwmma::load_matrix_sync(kb, reinterpret_cast<const T *>(k) + n * 128 + d, 128);
        rocwmma::mma_sync(c, qa, kb, c);
    }
    rocwmma::store_matrix_sync(scores + n, c, 32, rocwmma::mem_row_major);
}
// product[16 x 128] columns [n, n+16) = P[16 x 32] . V[32 x 128].
template <bool BF16, class Raw>
__device__ inline void product_fragment(const Raw *p, const Raw *v, float *product, int n) {
    using T = typename Operand<BF16>::type;
    constexpr int KS = Operand<BF16>::block_k;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, KS, float> c;
    rocwmma::fill_fragment(c, 0.f);
    for (int kk = 0; kk < 32; kk += KS) {
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, KS, T, rocwmma::row_major> pa;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, KS, T, rocwmma::row_major> vb;
        rocwmma::load_matrix_sync(pa, reinterpret_cast<const T *>(p) + kk, 32);
        rocwmma::load_matrix_sync(vb, reinterpret_cast<const T *>(v) + kk * 128 + n, 128);
        rocwmma::mma_sync(c, pa, vb, c);
    }
    rocwmma::store_matrix_sync(product + n, c, 128, rocwmma::mem_row_major);
}
// Probability operand: FP8 relative to the per-row tile scale, or BF16 unscaled.
template <bool BF16> __device__ inline typename Operand<BF16>::raw probability(float x, float pscale) {
    if constexpr (BF16)
        return device::bf16(x);
    else
        return device::fp8(x / pscale);
}
// General (prefill / multi-query) kernel: one wave per (request, query head,
// 16-query tile, key partition). Unchanged semantics from round 3.
template <bool BF16>
__global__ void attention_kernel(Meta m, AttentionDeviceView a, float *partial, int parts,
                                 int partition_keys) {
    using Raw = typename Operand<BF16>::raw;
    const int lane = int(threadIdx.x), req = int(blockIdx.x), head = int(blockIdx.y),
              part = int(blockIdx.z) % parts;
    const int tile = int(blockIdx.z) / parts, first = m.offsets[req] + tile * 16,
              stop = m.offsets[req + 1], kh = head / 12;
    if (first >= stop)
        return;
    __shared__ __attribute__((aligned(16))) Raw q[16 * 128], k[32 * 128], v[32 * 128], p[16 * 32];
    __shared__ float scores[16 * 32], product[16 * 128], out[16 * 128], mx[16], den[16], alpha[16],
        pscale[16], ks[32], vs[32];
    for (int i = lane; i < 16 * 128; i += 32) {
        int row = i / 128, d = i % 128, g = first + row;
        if (g < stop)
            load_query_element<BF16>(a, (size_t(g) * size_t(m.qh) + size_t(head)) * 128 + size_t(d),
                                     q[i]);
        else
            q[i] = Operand<BF16>::zero();
        out[i] = 0;
    }
    if (lane < 16) {
        mx[lane] = -INFINITY;
        den[lane] = 0;
    }
    __syncthreads();
    int begin = partition_keys ? part * partition_keys : 0;
    int end = partition_keys ? min(m.len[req], begin + partition_keys) : m.len[req];
    for (int base = begin; base < end; base += 32) {
        for (int i = lane; i < 32 * 128; i += 32) {
            int t = base + i / 128, d = i % 128;
            bool valid = t < end;
            int page = valid ? m.table[req * m.stride + t / 16] : 0;
            size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
            k[i] = valid ? Operand<BF16>::from_fp8(a.k[loc * 128 + size_t(d)]) : Operand<BF16>::zero();
            v[i] = valid ? Operand<BF16>::from_fp8(a.v[loc * 128 + size_t(d)]) : Operand<BF16>::zero();
        }
        int t = base + lane;
        bool valid = t < end;
        int page = valid ? m.table[req * m.stride + t / 16] : 0;
        size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
        ks[lane] = valid ? a.k_scale[loc] : 1;
        vs[lane] = valid ? a.v_scale[loc] : 1;
        __syncthreads();
        for (int n = 0; n < 32; n += 16)
            score_fragment<BF16>(q, k, scores, n);
        __syncthreads();
        if (lane < 16) {
            int g = first + lane;
            const float qscale =
                BF16 || g >= stop ? 1.f : a.q_scale[size_t(g) * size_t(m.qh) + size_t(head)];
            float maximum = -INFINITY;
            for (int j = 0; j < 32; ++j) {
                bool keep = g < stop && base + j < end && base + j <= m.pos[g];
                float score =
                    keep ? scores[lane * 32 + j] * qscale * ks[j] * 0.08838834764831845f : -INFINITY;
                scores[lane * 32 + j] = score;
                maximum = fmaxf(maximum, score);
            }
            float next = fmaxf(mx[lane], maximum);
            alpha[lane] = isfinite(mx[lane]) ? expf(mx[lane] - next) : 0;
            float sum = den[lane] * alpha[lane], weighted_max = 0;
            for (int j = 0; j < 32; ++j) {
                float probability =
                    isfinite(scores[lane * 32 + j]) ? expf(scores[lane * 32 + j] - next) : 0;
                sum += probability;
                scores[lane * 32 + j] = probability * vs[j];
                weighted_max = fmaxf(weighted_max, scores[lane * 32 + j]);
            }
            pscale[lane] = BF16 ? 1.f : device::scale(weighted_max);
            for (int j = 0; j < 32; ++j)
                p[lane * 32 + j] = probability<BF16>(scores[lane * 32 + j], pscale[lane]);
            mx[lane] = next;
            den[lane] = sum;
        }
        __syncthreads();
        for (int n = 0; n < 128; n += 16)
            product_fragment<BF16>(p, v, product, n);
        __syncthreads();
        for (int i = lane; i < 16 * 128; i += 32)
            out[i] = out[i] * alpha[i / 128] + product[i] * pscale[i / 128];
        __syncthreads();
    }
    for (int i = lane; i < 16 * 128; i += 32) {
        int g = first + i / 128;
        if (g >= stop)
            continue;
        size_t row = size_t(g) * size_t(m.qh) + size_t(head);
        if (parts == 1)
            reinterpret_cast<uint16_t *>(a.output)[row * 128 + size_t(i % 128)] =
                device::bf16(den[i / 128] > 0 ? out[i] / den[i / 128] : 0);
        else
            partial[(row * size_t(parts) + size_t(part)) * 130 + 2 + size_t(i % 128)] = out[i];
    }
    if (lane < 16 && first + lane < stop) {
        size_t row = size_t(first + lane) * size_t(m.qh) + size_t(head);
        if (parts == 1) {
            if (a.maximum)
                a.maximum[row] = mx[lane];
            if (a.denominator)
                a.denominator[row] = den[lane];
        } else {
            size_t i = (row * size_t(parts) + size_t(part)) * 130;
            partial[i] = mx[lane];
            partial[i + 1] = den[lane];
        }
    }
}
// GQA decode kernel (round 4): one 128-thread block per (request, kv head,
// key partition) when every request has exactly one query. The 12 query
// heads sharing the kv head form the 16-row matrix tile (rows 12..15 are
// zero padding), so each K/V tile is read from memory once and used by all
// 12 heads. Tile semantics (FP8 Q/K/V decoding and scales, per-row FP8 P
// relative to the 32-key tile maximum, fp32 online softmax, split-K
// partials) are identical to attention_kernel; only the work decomposition
// changes. Thread t owns softmax row t/8 (head), keys (t%8)*4..+4 of the
// tile and output columns (t%8)*16..+16 of that row. K/V tiles are
// prefetched into registers one tile ahead (16-byte loads).
inline constexpr int gqa_threads = 128;
template <bool BF16>
__global__ __launch_bounds__(gqa_threads) void gqa_decode_kernel(Meta m, AttentionDeviceView a,
                                                                 float *partial, int parts,
                                                                 int partition_keys) {
    using Raw = typename Operand<BF16>::raw;
    const int tid = int(threadIdx.x), req = int(blockIdx.x), kh = int(blockIdx.y),
              part = int(blockIdx.z);
    const int g = m.offsets[req];
    if (g >= m.offsets[req + 1])
        return;
    const int wave = tid >> 5, r = tid >> 3, jq = (tid & 7) * 4, c0 = (tid & 7) * 16;
    const int head0 = kh * 12, len = m.len[req], pos = m.pos[g];
    const int begin = partition_keys ? part * partition_keys : 0;
    const int end = partition_keys ? min(len, begin + partition_keys) : len;
    const int limit = min(end, pos + 1); // keys after the query position are masked anyway
    __shared__ __attribute__((aligned(16))) Raw q[16 * 128], k[32 * 128], v[32 * 128], p[16 * 32];
    __shared__ __attribute__((aligned(16))) float scores[16 * 32], product[16 * 128], ks[32], vs[32];
    for (int i = tid; i < 16 * 128; i += gqa_threads) {
        int row = i / 128, d = i % 128;
        if (row < 12)
            load_query_element<BF16>(
                a, (size_t(g) * size_t(m.qh) + size_t(head0 + row)) * 128 + size_t(d), q[i]);
        else
            q[i] = Operand<BF16>::zero();
    }
    const float qscale =
        BF16 || r >= 12 ? 1.f : a.q_scale[size_t(g) * size_t(m.qh) + size_t(head0 + r)];
    float mx = -INFINITY, den = 0, out[16];
    for (float &o : out)
        o = 0;
    // Register prefetch: thread t loads 16-byte chunks t and t+128 of the
    // 32x128 K tile and the same of V (chunk -> key chunk/8, offset chunk%8*16).
    uint4 kreg[2], vreg[2];
    float sreg = 1;
    auto fetch = [&](int base) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            int chunk = tid + c * gqa_threads, t = base + (chunk >> 3), d = (chunk & 7) * 16;
            bool valid = t < limit;
            int page = valid ? m.table[req * m.stride + t / 16] : 0;
            size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
            if (valid) {
                kreg[c] = *reinterpret_cast<const uint4 *>(a.k + loc * 128 + size_t(d));
                vreg[c] = *reinterpret_cast<const uint4 *>(a.v + loc * 128 + size_t(d));
            } else {
                kreg[c] = uint4{0, 0, 0, 0};
                vreg[c] = uint4{0, 0, 0, 0};
            }
        }
        if (tid < 64) {
            int t = base + (tid & 31);
            bool valid = t < limit;
            int page = valid ? m.table[req * m.stride + t / 16] : 0;
            size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
            sreg = valid ? (tid < 32 ? a.k_scale[loc] : a.v_scale[loc]) : 1.f;
        }
    };
    auto store = [&] {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            int chunk = tid + c * gqa_threads, key = chunk >> 3, d = (chunk & 7) * 16;
            if constexpr (BF16) {
                const uint8_t *kb = reinterpret_cast<const uint8_t *>(&kreg[c]),
                              *vb = reinterpret_cast<const uint8_t *>(&vreg[c]);
                uint16_t kw[16], vw[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    kw[i] = Operand<true>::from_fp8(kb[i]);
                    vw[i] = Operand<true>::from_fp8(vb[i]);
                }
                uint4 *kd = reinterpret_cast<uint4 *>(k + key * 128 + d),
                      *vd = reinterpret_cast<uint4 *>(v + key * 128 + d);
                kd[0] = *reinterpret_cast<const uint4 *>(kw);
                kd[1] = *reinterpret_cast<const uint4 *>(kw + 8);
                vd[0] = *reinterpret_cast<const uint4 *>(vw);
                vd[1] = *reinterpret_cast<const uint4 *>(vw + 8);
            } else {
                *reinterpret_cast<uint4 *>(k + key * 128 + d) = kreg[c];
                *reinterpret_cast<uint4 *>(v + key * 128 + d) = vreg[c];
            }
        }
        if (tid < 32)
            ks[tid] = sreg;
        else if (tid < 64)
            vs[tid - 32] = sreg;
    };
    if (begin < limit)
        fetch(begin);
    for (int base = begin; base < limit; base += 32) {
        store();
        if (base + 32 < limit)
            fetch(base + 32);
        __syncthreads();
        if (wave < 2)
            score_fragment<BF16>(q, k, scores, wave * 16);
        __syncthreads();
        float s[4], tile_max = -INFINITY;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            int j = jq + jj;
            bool keep = r < 12 && base + j < limit;
            s[jj] = keep ? scores[r * 32 + j] * qscale * ks[j] * 0.08838834764831845f : -INFINITY;
            tile_max = fmaxf(tile_max, s[jj]);
        }
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 2));
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 4));
        const float next = fmaxf(mx, tile_max), alpha = isfinite(mx) ? expf(mx - next) : 0;
        float sum = 0, weighted_max = 0, pv[4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            float probability = isfinite(s[jj]) ? expf(s[jj] - next) : 0;
            sum += probability;
            pv[jj] = probability * vs[jq + jj];
            weighted_max = fmaxf(weighted_max, pv[jj]);
        }
        sum += __shfl_xor(sum, 1);
        sum += __shfl_xor(sum, 2);
        sum += __shfl_xor(sum, 4);
        weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 1));
        weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 2));
        weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 4));
        const float pscale = BF16 ? 1.f : device::scale(weighted_max);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
            p[r * 32 + jq + jj] = probability<BF16>(pv[jj], pscale);
        den = den * alpha + sum;
        mx = next;
        __syncthreads();
        product_fragment<BF16>(p, v, product, wave * 32);
        product_fragment<BF16>(p, v, product, wave * 32 + 16);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i)
            out[i] = out[i] * alpha + product[r * 128 + c0 + i] * pscale;
    }
    if (r >= 12)
        return;
    const size_t row = size_t(g) * size_t(m.qh) + size_t(head0 + r);
    if (parts == 1) {
        uint16_t *o = reinterpret_cast<uint16_t *>(a.output) + row * 128 + size_t(c0);
#pragma unroll
        for (int i = 0; i < 16; ++i)
            o[i] = device::bf16(den > 0 ? out[i] / den : 0);
        if ((tid & 7) == 0) {
            if (a.maximum)
                a.maximum[row] = mx;
            if (a.denominator)
                a.denominator[row] = den;
        }
    } else {
        float *dst = partial + (row * size_t(parts) + size_t(part)) * 130;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            dst[2 + c0 + i] = out[i];
        if ((tid & 7) == 0) {
            dst[0] = mx;
            dst[1] = den;
        }
    }
}
// GQA prefill kernel (round 4): one block of 12 waves per (request, kv head,
// 16-query tile, key partition); wave w owns query head kh*12+w, so the 12
// heads sharing a kv head read each K/V tile from LDS once. Scores and the
// output accumulate in WMMA accumulator fragments, whose gfx12 wave32 layout
// is lane l -> column l%16, element i -> row 8*(l/16)+i (verified at startup
// by verify_device). Only P (per-row FP8 against the 32-key tile maximum, or
// BF16) goes through LDS between the two matrix products. Semantics match
// attention_kernel / reference_attention.
inline constexpr int prefill_waves = 12, prefill_threads = prefill_waves * 32;
template <bool BF16>
__global__ __launch_bounds__(prefill_threads) void gqa_prefill_kernel(Meta m, AttentionDeviceView a,
                                                                      float *partial, int parts,
                                                                      int partition_keys) {
    using Raw = typename Operand<BF16>::raw;
    using T = typename Operand<BF16>::type;
    constexpr int KS = Operand<BF16>::block_k, DSTEPS = 128 / KS;
    const int tid = int(threadIdx.x), wave = tid >> 5, lane = tid & 31, req = int(blockIdx.x),
              kh = int(blockIdx.y), part = int(blockIdx.z) % parts, tile = int(blockIdx.z) / parts;
    const int first = m.offsets[req] + tile * 16, stop = m.offsets[req + 1], head = kh * 12 + wave;
    if (first >= stop)
        return;
    // Shared storage: Q staging (reused afterwards for K, V, P and the scales).
    constexpr int q_bytes = 16 * 128 * int(sizeof(Raw)), kv_bytes = 32 * 128 * int(sizeof(Raw)),
                  p_bytes = 16 * 32 * int(sizeof(Raw));
    constexpr int stage_bytes = 12 * 16 * 128, // 24 KB: 12 FP8 or 6 BF16 query tiles per round
                  loop_bytes = 2 * kv_bytes + prefill_waves * p_bytes + 2 * 32 * 4;
    constexpr int shared_bytes = stage_bytes > loop_bytes ? stage_bytes : loop_bytes;
    __shared__ __attribute__((aligned(16))) unsigned char shared[shared_bytes];
    Raw *k = reinterpret_cast<Raw *>(shared), *v = reinterpret_cast<Raw *>(shared + kv_bytes),
        *p = reinterpret_cast<Raw *>(shared + 2 * kv_bytes) + wave * 16 * 32;
    float *ks = reinterpret_cast<float *>(shared + 2 * kv_bytes + prefill_waves * p_bytes),
          *vs = ks + 32;
    // Per-lane row bookkeeping: rows 8*(lane/16)+i.
    const int row0 = 8 * (lane >> 4), col = lane & 15;
    float qscale[8], mx[8], den[8];
    int pos[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int g = first + row0 + i;
        const bool live = g < stop;
        qscale[i] = BF16 || !live ? 1.f : a.q_scale[size_t(g) * size_t(m.qh) + size_t(head)];
        pos[i] = live ? m.pos[g] : -1;
        mx[i] = -INFINITY;
        den[i] = 0;
    }
    // Stage Q through LDS (zero padded) and keep it in A fragments.
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, KS, T, rocwmma::row_major> qf[DSTEPS];
    constexpr int waves_per_round = stage_bytes / q_bytes;
    for (int round = 0; round < prefill_waves; round += waves_per_round) {
        const int lo = round, hi = min(prefill_waves, round + waves_per_round);
        for (int i = tid; i < (hi - lo) * 16 * 128; i += prefill_threads) {
            const int w = lo + i / (16 * 128), e = i % (16 * 128), row = e / 128, d = e % 128;
            const int g = first + row;
            Raw value = Operand<BF16>::zero();
            if (g < stop)
                load_query_element<BF16>(
                    a, (size_t(g) * size_t(m.qh) + size_t(kh * 12 + w)) * 128 + size_t(d), value);
            reinterpret_cast<Raw *>(shared)[(w - lo) * 16 * 128 + e] = value;
        }
        __syncthreads();
        if (wave >= lo && wave < hi) {
            const Raw *q = reinterpret_cast<const Raw *>(shared) + (wave - lo) * 16 * 128;
#pragma unroll
            for (int s = 0; s < DSTEPS; ++s)
                rocwmma::load_matrix_sync(qf[s], reinterpret_cast<const T *>(q) + s * KS, 128);
        }
        __syncthreads();
    }
    rocwmma::fragment<rocwmma::accumulator, 16, 16, KS, float> acc[8];
#pragma unroll
    for (int n = 0; n < 8; ++n)
        rocwmma::fill_fragment(acc[n], 0.f);
    const int begin = partition_keys ? part * partition_keys : 0;
    const int end = partition_keys ? min(m.len[req], begin + partition_keys) : m.len[req];
    // Keys past the tile's last query position are masked for every row.
    int tile_last = -1;
    for (int g = first; g < min(stop, first + 16); ++g)
        tile_last = max(tile_last, m.pos[g]);
    const int limit = min(end, tile_last + 1);
    // Cooperative K/V tile prefetch: threads 0..255 own 16-byte chunk tid of K
    // and of V (key chunk/8, offset chunk%8*16); threads 256..319 the scales.
    uint4 kreg{}, vreg{};
    float sreg = 1;
    auto fetch = [&](int base) {
        if (tid < 256) {
            const int t = base + (tid >> 3), d = (tid & 7) * 16;
            const bool valid = t < limit;
            const int page = valid ? m.table[req * m.stride + t / 16] : 0;
            const size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
            if (valid) {
                kreg = *reinterpret_cast<const uint4 *>(a.k + loc * 128 + size_t(d));
                vreg = *reinterpret_cast<const uint4 *>(a.v + loc * 128 + size_t(d));
            } else {
                kreg = uint4{0, 0, 0, 0};
                vreg = uint4{0, 0, 0, 0};
            }
        } else if (tid < 320) {
            const int t = base + (tid & 31);
            const bool valid = t < limit;
            const int page = valid ? m.table[req * m.stride + t / 16] : 0;
            const size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
            sreg = valid ? (tid < 288 ? a.k_scale[loc] : a.v_scale[loc]) : 1.f;
        }
    };
    auto store = [&] {
        if (tid < 256) {
            const int key = tid >> 3, d = (tid & 7) * 16;
            if constexpr (BF16) {
                const uint8_t *kb = reinterpret_cast<const uint8_t *>(&kreg),
                              *vb = reinterpret_cast<const uint8_t *>(&vreg);
                uint16_t kw[16], vw[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    kw[i] = Operand<true>::from_fp8(kb[i]);
                    vw[i] = Operand<true>::from_fp8(vb[i]);
                }
                uint4 *kd = reinterpret_cast<uint4 *>(k + key * 128 + d),
                      *vd = reinterpret_cast<uint4 *>(v + key * 128 + d);
                kd[0] = *reinterpret_cast<const uint4 *>(kw);
                kd[1] = *reinterpret_cast<const uint4 *>(kw + 8);
                vd[0] = *reinterpret_cast<const uint4 *>(vw);
                vd[1] = *reinterpret_cast<const uint4 *>(vw + 8);
            } else {
                *reinterpret_cast<uint4 *>(k + key * 128 + d) = kreg;
                *reinterpret_cast<uint4 *>(v + key * 128 + d) = vreg;
            }
        } else if (tid < 288) {
            ks[tid - 256] = sreg;
        } else if (tid < 320) {
            vs[tid - 288] = sreg;
        }
    };
    if (begin < limit)
        fetch(begin);
    for (int base = begin; base < limit; base += 32) {
        store();
        if (base + 32 < limit)
            fetch(base + 32);
        __syncthreads();
        // Scores for this wave's head: two 16-key column fragments.
        rocwmma::fragment<rocwmma::accumulator, 16, 16, KS, float> sc[2];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            rocwmma::fill_fragment(sc[n], 0.f);
#pragma unroll
            for (int s = 0; s < DSTEPS; ++s) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, KS, T, rocwmma::col_major> kb;
                rocwmma::load_matrix_sync(kb, reinterpret_cast<const T *>(k) + n * 16 * 128 + s * KS,
                                          128);
                rocwmma::mma_sync(sc[n], qf[s], kb, sc[n]);
            }
        }
        // Online softmax on the accumulator layout: element i of fragment n is
        // row row0+i, key n*16+col.
        float alpha[8], pscale[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float tile_max = -INFINITY;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int j = n * 16 + col, t = base + j;
                const bool keep = t < limit && t <= pos[i];
                const float s = keep ? sc[n].x[i] * qscale[i] * ks[j] * 0.08838834764831845f
                                     : -INFINITY;
                sc[n].x[i] = s;
                tile_max = fmaxf(tile_max, s);
            }
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 2));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 4));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 8));
            const float next = fmaxf(mx[i], tile_max);
            alpha[i] = isfinite(mx[i]) ? expf(mx[i] - next) : 0;
            float sum = 0, weighted_max = 0, pv[2];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const float probability = isfinite(sc[n].x[i]) ? expf(sc[n].x[i] - next) : 0;
                sum += probability;
                pv[n] = probability * vs[n * 16 + col];
                weighted_max = fmaxf(weighted_max, pv[n]);
            }
            sum += __shfl_xor(sum, 1);
            sum += __shfl_xor(sum, 2);
            sum += __shfl_xor(sum, 4);
            sum += __shfl_xor(sum, 8);
            weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 1));
            weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 2));
            weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 4));
            weighted_max = fmaxf(weighted_max, __shfl_xor(weighted_max, 8));
            pscale[i] = BF16 ? 1.f : device::scale(weighted_max);
            den[i] = den[i] * alpha[i] + sum;
            mx[i] = next;
#pragma unroll
            for (int n = 0; n < 2; ++n)
                p[(row0 + i) * 32 + n * 16 + col] = probability<BF16>(pv[n], pscale[i]);
        }
        // This wave alone writes and reads its P tile.
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, KS, T, rocwmma::row_major> pf[32 / KS];
#pragma unroll
        for (int kk = 0; kk < 32 / KS; ++kk)
            rocwmma::load_matrix_sync(pf[kk], reinterpret_cast<const T *>(p) + kk * KS, 32);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            rocwmma::fragment<rocwmma::accumulator, 16, 16, KS, float> c;
            rocwmma::fill_fragment(c, 0.f);
#pragma unroll
            for (int kk = 0; kk < 32 / KS; ++kk) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, KS, T, rocwmma::row_major> vb;
                rocwmma::load_matrix_sync(vb, reinterpret_cast<const T *>(v) + kk * KS * 128 + n * 16,
                                          128);
                rocwmma::mma_sync(c, pf[kk], vb, c);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[n].x[i] = acc[n].x[i] * alpha[i] + c.x[i] * pscale[i];
            // Keep one column fragment in flight: without the barrier the
            // scheduler hoists all eight products and spills the accumulators.
            __builtin_amdgcn_sched_barrier(0);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int g = first + row0 + i;
        if (g >= stop)
            continue;
        const size_t row = size_t(g) * size_t(m.qh) + size_t(head);
        if (parts == 1) {
            uint16_t *o = reinterpret_cast<uint16_t *>(a.output) + row * 128;
#pragma unroll
            for (int n = 0; n < 8; ++n)
                o[n * 16 + col] = device::bf16(den[i] > 0 ? acc[n].x[i] / den[i] : 0);
            if (col == 0) {
                if (a.maximum)
                    a.maximum[row] = mx[i];
                if (a.denominator)
                    a.denominator[row] = den[i];
            }
        } else {
            float *dst = partial + (row * size_t(parts) + size_t(part)) * 130;
#pragma unroll
            for (int n = 0; n < 8; ++n)
                dst[2 + n * 16 + col] = acc[n].x[i];
            if (col == 0) {
                dst[0] = mx[i];
                dst[1] = den[i];
            }
        }
    }
}
// Startup check of the accumulator layout the prefill kernel relies on.
__global__ void layout_probe(int *bad) {
    __shared__ float m[16 * 16];
    const int lane = int(threadIdx.x);
    for (int i = lane; i < 256; i += 32)
        m[i] = float(i);
    __syncthreads();
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> c;
    rocwmma::load_matrix_sync(c, m, 16, rocwmma::mem_row_major);
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> d;
    rocwmma::load_matrix_sync(d, m, 16, rocwmma::mem_row_major);
    int errors = c.num_elements != 8 || d.num_elements != 8;
    for (int i = 0; i < 8; ++i) {
        const int expected = (8 * (lane / 16) + i) * 16 + lane % 16;
        errors += int(c.x[i]) != expected;
        errors += int(d.x[i]) != expected;
    }
    if (errors)
        atomicAdd(bad, errors);
}
__global__ void merge_kernel(AttentionDeviceView a, const float *partial, int rows, int parts) {
    int row = int(blockIdx.x), d = int(threadIdx.x);
    if (row >= rows || d >= 128)
        return;
    const float *base = partial + size_t(row) * size_t(parts) * 130;
    float maximum = -INFINITY;
    for (int p = 0; p < parts; ++p)
        maximum = fmaxf(maximum, base[size_t(p) * 130]);
    float sum = 0, out = 0;
#pragma unroll 8
    for (int p = 0; p < parts; ++p) {
        const float *i = base + size_t(p) * 130;
        float weight = isfinite(i[0]) ? expf(i[0] - maximum) : 0;
        sum += i[1] * weight;
        out += i[2 + d] * weight;
    }
    reinterpret_cast<uint16_t *>(a.output)[size_t(row) * 128 + size_t(d)] =
        device::bf16(sum > 0 ? out / sum : 0);
    if (d == 0) {
        if (a.maximum)
            a.maximum[row] = maximum;
        if (a.denominator)
            a.denominator[row] = sum;
    }
}

__global__ void append_kernel(Meta m, AttentionDeviceView a, const uint16_t *kin,
                              const uint16_t *vin) {
    int q = int(blockIdx.x), h = int(blockIdx.y), lane = int(threadIdx.x), req = 0;
    while (req + 1 < m.requests && q >= m.offsets[req + 1])
        ++req;
    int t = m.pos[q], page = m.table[req * m.stride + t / 16];
    size_t dest = (size_t(page) * size_t(m.kh) + size_t(h)) * 16 + size_t(t % 16),
           src = (size_t(q) * size_t(m.kh) + size_t(h)) * 128;
    float k = lane < 128 ? device::bf16(kin[src + size_t(lane)]) : 0,
          v = lane < 128 ? device::bf16(vin[src + size_t(lane)]) : 0;
    __shared__ float km[128], vm[128];
    km[lane] = fabsf(k);
    vm[lane] = fabsf(v);
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (lane < s) {
            km[lane] = fmaxf(km[lane], km[lane + s]);
            vm[lane] = fmaxf(vm[lane], vm[lane + s]);
        }
        __syncthreads();
    }
    float sk = device::scale(km[0]), sv = device::scale(vm[0]);
    a.k[dest * 128 + size_t(lane)] = device::fp8(k / sk);
    a.v[dest * 128 + size_t(lane)] = device::fp8(v / sv);
    if (lane == 0) {
        a.k_scale[dest] = sk;
        a.v_scale[dest] = sv;
    }
}
__global__ void quantize_kernel(const uint16_t *input, uint8_t *out, float *scales, int width) {
    int row = int(blockIdx.x), lane = int(threadIdx.x);
    __shared__ float maximum[256];
    float m = 0;
    for (int i = lane; i < width; i += 256)
        m = fmaxf(m, fabsf(device::bf16(input[size_t(row) * size_t(width) + size_t(i)])));
    maximum[lane] = m;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s)
            maximum[lane] = fmaxf(maximum[lane], maximum[lane + s]);
        __syncthreads();
    }
    float scale = device::scale(maximum[0]);
    if (lane == 0)
        scales[row] = scale;
    for (int i = lane; i < width; i += 256)
        out[size_t(row) * size_t(width) + size_t(i)] =
            device::fp8(device::bf16(input[size_t(row) * size_t(width) + size_t(i)]) / scale);
}
__global__ void conversion_kernel(const float *x, uint8_t *y, int n) {
    int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        y[i] = device::fp8(x[i]);
}
__global__ void matrix_probe(float *result) {
    __shared__ __attribute__((aligned(16))) F8 a[16 * 32], b[32 * 16];
    int lane = int(threadIdx.x);
    for (int i = lane; i < 512; i += 32) {
        a[i] = f8(device::fp8(1.f));
        b[i] = f8(device::fp8(1.f));
    }
    __syncthreads();
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major> ma;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::row_major> mb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> c;
    rocwmma::fill_fragment(c, 0.f);
    rocwmma::load_matrix_sync(ma, a, 32);
    rocwmma::load_matrix_sync(mb, b, 16);
    rocwmma::mma_sync(c, ma, mb, c);
    rocwmma::store_matrix_sync(result, c, 16, rocwmma::mem_row_major);
}
} // namespace
void verify_device() {
    hipDeviceProp_t p{};
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&p, device));
    if (std::string(p.gcnArchName).find("gfx1201") != 0 || p.warpSize != 32)
        throw std::runtime_error(
            "Native FP8 runtime requires gfx1201 / wave32; no precision fallback");
    DeviceBuffer<float> result(256);
    matrix_probe<<<1, 32>>>(result.data());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (float v : result.download())
        if (v != 32.f)
            throw std::runtime_error("Native FP8 matrix-instruction smoke test failed");
    DeviceBuffer<int> bad;
    bad.upload(std::vector<int>{0});
    layout_probe<<<1, 32>>>(bad.data());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    if (bad.download()[0])
        throw std::runtime_error("WMMA accumulator layout differs from the prefill kernel's model");
    std::array<float, 10> source{448,
                                 1000,
                                 -0.f,
                                 1.0625f,
                                 INFINITY,
                                 -INFINITY,
                                 std::numeric_limits<float>::max(),
                                 -std::numeric_limits<float>::max(),
                                 std::bit_cast<float>(0x7fc00000U),
                                 std::bit_cast<float>(0xffc00000U)};
    DeviceBuffer<float> input;
    input.upload(source);
    DeviceBuffer<uint8_t> encoded(source.size());
    fp8_conversion_probe(input.data(), encoded.data(), int(source.size()), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto values = encoded.download();
    for (size_t i = 0; i < source.size(); ++i)
        if (values[i] != encode_e4m3(source[i]))
            throw std::runtime_error("OCP E4M3 smoke test failed (saturation/signed zero/NaN/RNE)");
}
GpuMetadata::~GpuMetadata() {
    if (staging_)
        (void)hipHostFree(staging_);
}
void GpuMetadata::ensure_staging(size_t ints, hipStream_t) {
    if (ints <= staging_capacity_)
        return;
    if (staging_)
        (void)hipHostFree(staging_);
    staging_capacity_ = ints * 2; // one-time headroom for typical batch growth
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&staging_),
                            staging_capacity_ * sizeof(int), hipHostMallocPortable));
}
void GpuMetadata::upload(const AttentionLayout &l, std::span<const int> tokens_in,
                         hipStream_t stream) {
    valid_ = false;
    l.validate();
    host_ = l;
    max_queries_ = max_length_ = 0;
    for (int r = 0; r < l.requests(); ++r) {
        max_queries_ =
            std::max(max_queries_, l.query_offsets[size_t(r + 1)] - l.query_offsets[size_t(r)]);
        max_length_ = std::max(max_length_, l.lengths[size_t(r)]);
    }
    const size_t r = size_t(l.requests()), queries = size_t(l.queries()),
                 stride = size_t(l.table_stride);
    const size_t ints = (r + 1) + queries + r + r * stride + queries;
    ensure_staging(ints, stream);
    device_.resize(ints);
    size_t o = 0;
    auto put = [&](const int *src, size_t n) {
        if (n)
            std::memcpy(staging_ + o, src, n * sizeof(int));
        o += n;
    };
    put(l.query_offsets.data(), r + 1);
    put(l.positions.data(), queries);
    put(l.lengths.data(), r);
    put(l.page_table.data(), r * stride);
    put(tokens_in.data(), tokens_in.size());
    offsets = device_.data();
    positions = offsets + (r + 1);
    lengths = positions + queries;
    table = lengths + r;
    tokens = table + r * stride;
    if (ints) {
        if (stream)
            HIP_CHECK(hipMemcpyAsync(device_.data(), staging_, ints * sizeof(int),
                                     hipMemcpyHostToDevice, stream));
        else
            HIP_CHECK(hipMemcpy(device_.data(), staging_, ints * sizeof(int),
                                hipMemcpyHostToDevice));
    }
    valid_ = true;
}
size_t attention_workspace_bytes(const AttentionLayout &l, int partition) {
    l.validate();
    if (partition < 0 || partition % 32)
        throw std::invalid_argument("Partition must be a multiple of 32");
    int parts = partition
                    ? int(ceil_div(size_t(*std::max_element(l.lengths.begin(), l.lengths.end())),
                                   size_t(partition)))
                    : 1;
    return parts == 1 ? 0
                      : checked_mul(
                            checked_mul(size_t(l.queries()) * size_t(l.query_heads), size_t(parts)),
                            130 * sizeof(float));
}
int decode_attention_partition(const AttentionLayout &l, size_t workspace_bytes, int requested,
                               int target_blocks) {
    l.validate();
    const int max_len = *std::max_element(l.lengths.begin(), l.lengths.end());
    const size_t blocks = size_t(l.requests()) * size_t(l.kv_heads);
    int partition = requested;
    if (partition <= 0) {
        // Enough (request, kv head, partition) blocks to fill the GPU, each
        // owning whole 32-key tiles; one partition when the context is short.
        size_t parts = std::max<size_t>(1, ceil_div(size_t(std::max(target_blocks, 1)), blocks));
        partition = int(ceil_div(ceil_div(size_t(max_len), parts), size_t(32)) * 32);
    }
    partition = std::max(32, int(ceil_div(size_t(partition), size_t(32)) * 32));
    while (partition < max_len && attention_workspace_bytes(l, partition) > workspace_bytes)
        partition *= 2;
    return partition >= max_len ? 0 : partition;
}
void paged_attention(const GpuMetadata &metadata, AttentionDeviceView a, void *workspace,
                     size_t bytes, int partition, hipStream_t stream) {
    const auto &l = metadata.layout();
    if (partition < 0 || partition % 32)
        throw std::invalid_argument("Partition must be a multiple of 32");
    int parts = partition ? int(ceil_div(size_t(metadata.max_length()), size_t(partition))) : 1;
    size_t needed =
        parts == 1
            ? 0
            : checked_mul(checked_mul(size_t(l.queries()) * size_t(l.query_heads), size_t(parts)),
                          130 * sizeof(float));
    if (!a.q || !a.q_scale || !a.k || !a.v || !a.k_scale || !a.v_scale || !a.output ||
        (needed && (!workspace || bytes < needed)))
        throw std::invalid_argument("Missing/undersized attention buffers");
    if (!l.queries())
        return;
    auto m = meta(metadata);
    float *partial = static_cast<float *>(workspace);
    static const bool legacy = [] {
        const char *v = std::getenv("BEHEMOTH_ATTN_LEGACY");
        return v && std::atoi(v) != 0;
    }();
    if (m.max_q == 1 && !legacy) {
        // Decode: every request carries one query -> GQA kernel over kv heads.
        dim3 grid(unsigned(m.requests), unsigned(m.kh), unsigned(parts));
        gqa_decode_kernel<false><<<grid, gqa_threads, 0, stream>>>(m, a, partial, parts,
                                                                        partition);
    } else if (!legacy) {
        // Prefill / multi-query: 12 waves (one per query head of the kv group) per
        // (request, kv head, 16-query tile, key partition).
        dim3 grid(unsigned(m.requests), unsigned(m.kh),
                  unsigned(ceil_div(size_t(m.max_q), 16)) * unsigned(parts));
        gqa_prefill_kernel<false><<<grid, prefill_threads, 0, stream>>>(m, a, partial, parts,
                                                                            partition);
    } else {
        dim3 grid(unsigned(m.requests), unsigned(m.qh),
                  unsigned(ceil_div(size_t(m.max_q), 16)) * unsigned(parts));
        attention_kernel<false><<<grid, 32, 0, stream>>>(m, a, partial, parts, partition);
    }
    HIP_CHECK(hipGetLastError());
    if (parts > 1) {
        merge_kernel<<<unsigned(m.queries * m.qh), 128, 0, stream>>>(a, partial, m.queries * m.qh,
                                                                     parts);
        HIP_CHECK(hipGetLastError());
    }
}
void append_kv(const GpuMetadata &metadata, AttentionDeviceView a, const BFloat16 *k,
               const BFloat16 *v, hipStream_t stream) {
    (void)metadata.layout(); // validated once per batch, rather than 88 times per token
    if (!k || !v || !a.k || !a.v || !a.k_scale || !a.v_scale)
        throw std::invalid_argument("Missing KV append buffers");
    auto m = meta(metadata);
    if (!m.queries)
        return;
    append_kernel<<<dim3(unsigned(m.queries), unsigned(m.kh)), 128, 0, stream>>>(
        m, a, reinterpret_cast<const uint16_t *>(k), reinterpret_cast<const uint16_t *>(v));
    HIP_CHECK(hipGetLastError());
}
void quantize_rows(const BFloat16 *x, uint8_t *y, float *s, int rows, int width,
                   hipStream_t stream) {
    if (rows < 0 || width < 1 || !x || !y || !s)
        throw std::invalid_argument("Invalid quantization buffers/dimensions");
    if (rows) {
        quantize_kernel<<<unsigned(rows), 256, 0, stream>>>(reinterpret_cast<const uint16_t *>(x),
                                                            y, s, width);
        HIP_CHECK(hipGetLastError());
    }
}
void fp8_conversion_probe(const float *x, uint8_t *y, int n, hipStream_t stream) {
    if (n > 0) {
        conversion_kernel<<<unsigned(ceil_div(size_t(n), 256)), 256, 0, stream>>>(x, y, n);
        HIP_CHECK(hipGetLastError());
    }
}
} // namespace behemoth
