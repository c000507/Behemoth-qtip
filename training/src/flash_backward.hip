// Native gfx1201 BF16 attention adjoints. Each output tile has one owner.
#include "../../runtime/src/device_numeric.hpp"
#include "behemoth/gpu.hpp"
#include "hip_ops.hpp"
#include "view_device.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace qat::gpu {
namespace {
using Reg = __bf16 __attribute__((ext_vector_type(8)));
using Acc = float __attribute__((ext_vector_type(8)));
using behemoth::device::bf16;
constexpr float scale = .08838834764831845f;
__device__ __forceinline__ Acc mma(Reg a, Reg b, Acc c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ int kb(int s, int h) { return 32 * (s >> 1) + 16 * h + 8 * (s & 1); }
__device__ __forceinline__ float get(View x, int t, int head, int d) {
    return qat::gpu::load(x, size_t(t) * x.ld + head * 128 + d);
}
// Row-major tiles are XOR-swizzled in eight-BF16 chunks. The same helper
// handles 128-wide feature rows and 32-wide query/probability rows.
template <int Width> __device__ __forceinline__ int at(int r, int d) {
    return r * Width + (d ^ ((r & (Width / 8 - 1)) * 8));
}
template <int Width> __device__ __forceinline__ Reg load(const uint16_t *p, int row, int d) {
    return __builtin_bit_cast(Reg, *reinterpret_cast<const uint4 *>(p + at<Width>(row, d)));
}

// One wave per query head, sharing K/V across its GQA group. Q and dO are
// resident in registers. K is staged in both orientations for QK and dQ.
__global__ void dq_kernel(View q, View k, View v, View lse, View dy, View delta, View dq, int group,
                          const int32_t *starts) {
    __shared__ __align__(16) uint16_t shared[3 * 32 * 128];
    uint16_t *kr = shared, *vr = kr + 32 * 128, *kt = vr + 32 * 128;
    int tid = threadIdx.x, wave = tid / 32, c = tid % 16, h = tid % 32 / 16;
    int first = blockIdx.x * 16, kh = blockIdx.y, head = kh * group + wave, t = first + c;
    bool live = t < q.rows;
    Reg qr[8], dr[8];
    // Stage four heads at a time; shared storage is reused by the key loop.
    for (int part = 0; part < 2; ++part) {
        View src = part ? dy : q;
        for (int lo = 0; lo < group; lo += 4) {
            int count = min(4, group - lo);
            for (int i = tid; i < count * 16 * 128; i += blockDim.x) {
                int row = i / 128 % 16, w = i / (16 * 128), d = i % 128;
                shared[i] = first + row < q.rows ? bf16(get(src, first + row, kh * group + lo + w, d)) : 0;
            }
            __syncthreads();
            if (wave >= lo && wave < lo + count)
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    Reg value = __builtin_bit_cast(
                        Reg,
                        *reinterpret_cast<const uint4 *>(shared + ((wave - lo) * 16 + c) * 128 + kb(s, h)));
                    if (part)
                        dr[s] = value;
                    else
                        qr[s] = value;
                }
            __syncthreads();
        }
    }
    Acc out[8]{};
    float logsum = live ? lse.p[size_t(head) * lse.ld + t] : 0;
    float correction = live ? delta.p[size_t(head) * delta.ld + t] : 0;
    int doc = live ? starts[t] : q.rows;
    int limit = min(first + 16, q.rows);
    for (int base = starts[first]; base < limit; base += 32) {
        for (int i = tid; i < 32 * 128; i += blockDim.x) {
            int row = i / 128, d = i % 128;
            uint16_t key = base + row < limit ? bf16(get(k, base + row, kh, d)) : 0;
            kr[at<128>(row, d)] = key;
            vr[at<128>(row, d)] = base + row < limit ? bf16(get(v, base + row, kh, d)) : 0;
            kt[at<32>(d, row)] = key;
        }
        __syncthreads();
        Reg ds[2];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            Acc score{}, dp{};
            int row = 16 * (c >> 3) + 8 * n + (c & 7);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                score = mma(load<128>(kr, row, kb(s, h)), qr[s], score);
                dp = mma(load<128>(vr, row, kb(s, h)), dr[s], dp);
            }
            uint16_t bits[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int key = base + 16 * h + 8 * n + i;
                float p = live && key >= doc && key <= t
                              ? __builtin_amdgcn_exp2f((score[i] * scale - logsum) * 1.4426950408889634f)
                              : 0;
                bits[i] = bf16(p * (dp[i] - correction) * scale);
            }
            ds[n] = __builtin_bit_cast(Reg, bits);
        }
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int s = 0; s < 2; ++s)
                out[nd] = mma(load<32>(kt, 16 * nd + c, kb(s, h)), ds[s], out[nd]);
        __syncthreads();
    }
    if (live)
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                store_rounded(dq, size_t(t) * dq.ld + head * 128 + 16 * nd + 8 * h + i, out[nd][i]);
}

// Four waves own the 128 output dimensions of 16 keys. All GQA heads reduce
// into those registers. Q/dO staging is transposed in place after scoring so
// both gradient products use vector LDS loads, with no global atomics.
__global__ void dkv_staged_kernel(View q, View k, View v, View lse, View dy, View delta, View dk, View dv, int group,
                           const int32_t *starts, const int32_t *ends) {
    __shared__ __align__(16) uint16_t kr[16 * 128], vr[16 * 128], qr[32 * 128], dr[32 * 128], p[16 * 32],
        ds[16 * 32];
    __shared__ float scores[16 * 32], dps[16 * 32];
    int tid = threadIdx.x, wave = tid / 32, c = tid % 16, h = tid % 32 / 16;
    int first = blockIdx.x * 16, kh = blockIdx.y;
    Acc ak[2]{}, av[2]{};
    for (int i = tid; i < 16 * 128; i += 128) {
        int row = i / 128, d = i % 128;
        kr[at<128>(row, d)] = first + row < q.rows ? bf16(get(k, first + row, kh, d)) : 0;
        vr[at<128>(row, d)] = first + row < q.rows ? bf16(get(v, first + row, kh, d)) : 0;
    }
    int end = ends[min(first + 15, q.rows - 1)];
    for (int head = kh * group; head < (kh + 1) * group; ++head)
        for (int base = first; base < end; base += 32) {
            for (int i = tid; i < 32 * 128; i += 128) {
                int row = i / 128, d = i % 128;
                qr[at<128>(row, d)] = base + row < end ? bf16(get(q, base + row, head, d)) : 0;
                dr[at<128>(row, d)] = base + row < end ? bf16(get(dy, base + row, head, d)) : 0;
            }
            __syncthreads();
            Acc result{};
            int sub = (wave % 2) * 16;
#pragma unroll
            for (int s = 0; s < 8; ++s)
                result = mma(load<128>(wave < 2 ? kr : vr, c, kb(s, h)),
                             load<128>(wave < 2 ? qr : dr, sub + c, kb(s, h)), result);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                (wave < 2 ? scores : dps)[(8 * h + i) * 32 + sub + c] = result[i];
            __syncthreads();
            for (int i = tid; i < 16 * 32; i += 128) {
                int key = first + i / 32, t = base + i % 32;
                bool valid = key < q.rows && t < end && key <= t && key >= starts[t];
                float prob =
                    valid ? __builtin_amdgcn_exp2f((scores[i] * scale - lse.p[size_t(head) * lse.ld + t]) *
                                                   1.4426950408889634f)
                          : 0;
                float correction = t < end ? delta.p[size_t(head) * delta.ld + t] : 0;
                p[at<32>(i / 32, i % 32)] = bf16(prob);
                ds[at<32>(i / 32, i % 32)] = bf16(prob * (dps[i] - correction) * scale);
            }
            // Each thread owns one feature column while transposing all 32 rows.
            Reg tq[4], td[4];
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    tq[s][e] = __builtin_bit_cast(__bf16, qr[at<128>(8 * s + e, tid)]);
                    td[s][e] = __builtin_bit_cast(__bf16, dr[at<128>(8 * s + e, tid)]);
                }
            __syncthreads();
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                *reinterpret_cast<uint4 *>(qr + at<32>(tid, 8 * s)) = __builtin_bit_cast(uint4, tq[s]);
                *reinterpret_cast<uint4 *>(dr + at<32>(tid, 8 * s)) = __builtin_bit_cast(uint4, td[s]);
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                int d = (wave + j * 4) * 16 + c;
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    ak[j] = mma(load<32>(ds, c, kb(s, h)), load<32>(qr, d, kb(s, h)), ak[j]);
                    av[j] = mma(load<32>(p, c, kb(s, h)), load<32>(dr, d, kb(s, h)), av[j]);
                }
            }
            __syncthreads();
        }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int t = first + 8 * h + i, d = (wave + j * 4) * 16 + c;
            if (t < q.rows) {
                store_rounded(dk, size_t(t) * dk.ld + kh * 128 + d, ak[j][i]);
                store_rounded(dv, size_t(t) * dv.ld + kh * 128 + d, av[j][i]);
            }
        }
}

// A pair of waves owns 16 keys: one computes scores/dS and dK, the other
// accumulates dV. Two pairs share a 16-query Q/dO load. K/V and dK/dV stay in
// registers; only P crosses between waves. Stage both Q/dO orientations directly
// to avoid the old score/dP exchange and in-place LDS transpose. Splitting the
// output accumulators between waves avoids spilling 128 FP32 values per lane.
__global__ __launch_bounds__(128) void dkv_wave_kernel(
    View q, View k, View v, View lse, View dy, View delta, View dk, View dv, int group,
    const int32_t *starts, const int32_t *ends) {
    __shared__ __align__(16) uint16_t shared[4 * 16 * 128], probs[32 * 16];
    __shared__ float logs[16], corrections[16];
    __shared__ int docs[16];
    uint16_t *qr = shared, *dr = qr + 16 * 128, *qt = dr + 16 * 128, *dt = qt + 16 * 128;
    int tid = threadIdx.x, wave = tid / 32, c = tid % 16, h = tid % 32 / 16;
    int begin = blockIdx.x * 32, pair = wave / 2, first = begin + pair * 16;
    int kh = blockIdx.y, key = first + c;
    bool live = key < q.rows, value_wave = wave & 1;
    Reg kr[8], vr[8];
    // Reuse the query staging space to load the two key tiles once.
    for (int part = 0; part < 2; ++part) {
        View src = part ? v : k;
        for (int i = tid; i < 32 * 128; i += 128) {
            int row = i / 128, d = i % 128;
            shared[i] = begin + row < q.rows ? bf16(get(src, begin + row, kh, d)) : 0;
        }
        __syncthreads();
        if (!value_wave)
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                Reg value = __builtin_bit_cast(
                    Reg, *reinterpret_cast<const uint4 *>(shared + (pair * 16 + c) * 128 + kb(s, h)));
                if (part)
                    vr[s] = value;
                else
                    kr[s] = value;
            }
        __syncthreads();
    }
    Acc out[8]{};
    // The last key can belong to a later packed document. Per-query masking
    // excludes cross-document pairs, including tiles spanning many documents.
    int end = ends[min(begin + 31, q.rows - 1)];
    for (int head = kh * group; head < (kh + 1) * group; ++head)
        for (int base = begin; base < end; base += 16) {
            for (int i = tid; i < 16 * 128; i += 128) {
                int row = i / 128, d = i % 128;
                uint16_t a = base + row < end ? bf16(get(q, base + row, head, d)) : 0;
                uint16_t b = base + row < end ? bf16(get(dy, base + row, head, d)) : 0;
                qr[at<128>(row, d)] = a;
                dr[at<128>(row, d)] = b;
                qt[at<16>(d, row)] = a;
                dt[at<16>(d, row)] = b;
            }
            if (tid < 16) {
                int t = base + tid;
                logs[tid] = t < end ? lse.p[size_t(head) * lse.ld + t] : 0;
                corrections[tid] = t < end ? delta.p[size_t(head) * delta.ld + t] : 0;
                docs[tid] = t < end ? starts[t] : q.rows;
            }
            __syncthreads();
            Reg ds;
            if (!value_wave) {
                // C(query,key): lane c owns one key; accumulator components
                // are queries 8*h..8*h+7. This is the reduction orientation
                // required by both Q^T*dS and dO^T*P.
                Acc score{}, dp{};
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    score = mma(load<128>(qr, c, kb(s, h)), kr[s], score);
                    dp = mma(load<128>(dr, c, kb(s, h)), vr[s], dp);
                }
                uint16_t pb[8], sb[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int row = 8 * h + i, t = base + row;
                    float prob = live && t < end && key <= t && key >= docs[row]
                                     ? __builtin_amdgcn_exp2f((score[i] * scale - logs[row]) *
                                                              1.4426950408889634f)
                                     : 0;
                    pb[i] = bf16(prob);
                    sb[i] = bf16(prob * (dp[i] - corrections[row]) * scale);
                }
                ds = __builtin_bit_cast(Reg, sb);
                *reinterpret_cast<uint4 *>(probs + at<16>(pair * 16 + c, 8 * h)) =
                    __builtin_bit_cast(uint4, pb);
            }
            __syncthreads();
            Reg derivative = value_wave ? load<16>(probs, pair * 16 + c, 8 * h) : ds;
#pragma unroll
            for (int nd = 0; nd < 8; ++nd)
                out[nd] = mma(load<16>(value_wave ? dt : qt, 16 * nd + c, 8 * h), derivative, out[nd]);
            __syncthreads();
        }
    if (live) {
        View dst = value_wave ? dv : dk;
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                store_rounded(dst, size_t(key) * dst.ld + kh * 128 + 16 * nd + 8 * h + i, out[nd][i]);
    }
}

// Balance score work between the pair: the dK wave computes QK^T, the dV
// wave computes dO*V^T. Only dP and P cross waves; each gradient still has
// one writer and reduces query heads in a fixed order. Four pairs share
// Q/dO loads over 64 keys; eight pairs extend that reuse to 128 keys.
template <int KeyTiles>
__global__ __launch_bounds__(KeyTiles * 64) void dkv_balanced_kernel(
    View q, View k, View v, View lse, View dy, View delta, View dk, View dv, int group,
    const int32_t *starts, const int32_t *ends) {
    static_assert(KeyTiles == 2 || KeyTiles == 4 || KeyTiles == 8);
    constexpr int KeyRows = KeyTiles * 16, Threads = KeyTiles * 64;
    constexpr int StageSize = 4 * 16 * 128;
    constexpr int LoadRows = KeyRows < 64 ? KeyRows : 64;
    // Reuse query staging for up to 64 K/V rows at a time, so the 128-key
    // variant needs no larger query buffer. These batches run only at entry.
    __shared__ __align__(16) uint16_t shared[StageSize], probs[KeyRows * 16];
    // A padded 20-float row keeps both halves aligned for vector LDS access
    // while distributing neighboring key rows across the memory banks.
    __shared__ __align__(16) float dps[KeyRows * 20];
    __shared__ float logs[16], corrections[16];
    __shared__ int docs[16];
    uint16_t *qr = shared, *dr = qr + 16 * 128, *qt = dr + 16 * 128, *dt = qt + 16 * 128;
    int tid = threadIdx.x, wave = tid / 32, c = tid % 16, h = tid % 32 / 16;
    int begin = blockIdx.x * KeyRows, pair = wave / 2, first = begin + pair * 16;
    int kh = blockIdx.y, key = first + c;
    bool live = key < q.rows, value_wave = wave & 1;
    Reg kv[8];
    for (int part = 0; part < 2; ++part) {
        View src = part ? v : k;
        for (int lo = 0; lo < KeyRows; lo += LoadRows) {
            for (int i = tid; i < LoadRows * 128; i += Threads) {
                int row = i / 128, d = i % 128;
                shared[i] = begin + lo + row < q.rows ? bf16(get(src, begin + lo + row, kh, d)) : 0;
            }
            __syncthreads();
            if (value_wave == bool(part) &&
                (KeyRows <= 64 || (pair * 16 >= lo && pair * 16 < lo + LoadRows)))
#pragma unroll
                for (int s = 0; s < 8; ++s)
                    kv[s] = __builtin_bit_cast(
                        Reg, *reinterpret_cast<const uint4 *>(shared + (pair * 16 - lo + c) * 128 + kb(s, h)));
            __syncthreads();
        }
    }
    Acc out[8]{};
    int end = ends[min(begin + KeyRows - 1, q.rows - 1)];
    for (int head = kh * group; head < (kh + 1) * group; ++head)
        for (int base = begin; base < end; base += 16) {
            for (int i = tid; i < 16 * 128; i += Threads) {
                int row = i / 128, d = i % 128;
                uint16_t a = base + row < end ? bf16(get(q, base + row, head, d)) : 0;
                uint16_t b = base + row < end ? bf16(get(dy, base + row, head, d)) : 0;
                qr[at<128>(row, d)] = a;
                dr[at<128>(row, d)] = b;
                qt[at<16>(d, row)] = a;
                dt[at<16>(d, row)] = b;
            }
            if (tid < 16) {
                int t = base + tid;
                logs[tid] = t < end ? lse.p[size_t(head) * lse.ld + t] : 0;
                corrections[tid] = t < end ? delta.p[size_t(head) * delta.ld + t] : 0;
                docs[tid] = t < end ? starts[t] : q.rows;
            }
            __syncthreads();
            Acc result{};
#pragma unroll
            for (int s = 0; s < 8; ++s)
                result = mma(load<128>(value_wave ? dr : qr, c, kb(s, h)), kv[s], result);
            using HalfAcc = float __attribute__((ext_vector_type(4)));
            float *dp_row = dps + (pair * 16 + c) * 20 + 8 * h;
            if (value_wave) {
                *reinterpret_cast<HalfAcc *>(dp_row) = __builtin_shufflevector(result, result, 0, 1, 2, 3);
                *reinterpret_cast<HalfAcc *>(dp_row + 4) =
                    __builtin_shufflevector(result, result, 4, 5, 6, 7);
            }
            __syncthreads();
            Reg ds;
            if (!value_wave) {
                Acc dp = __builtin_shufflevector(*reinterpret_cast<const HalfAcc *>(dp_row),
                                                *reinterpret_cast<const HalfAcc *>(dp_row + 4),
                                                0, 1, 2, 3, 4, 5, 6, 7);
                uint16_t pb[8], sb[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int row = 8 * h + i, t = base + row;
                    float prob = live && t < end && key <= t && key >= docs[row]
                                     ? __builtin_amdgcn_exp2f((result[i] * scale - logs[row]) *
                                                              1.4426950408889634f)
                                     : 0;
                    pb[i] = bf16(prob);
                    sb[i] = bf16(prob * (dp[i] - corrections[row]) * scale);
                }
                ds = __builtin_bit_cast(Reg, sb);
                *reinterpret_cast<uint4 *>(probs + at<16>(pair * 16 + c, 8 * h)) =
                    __builtin_bit_cast(uint4, pb);
            }
            __syncthreads();
            Reg derivative = value_wave ? load<16>(probs, pair * 16 + c, 8 * h) : ds;
#pragma unroll
            for (int nd = 0; nd < 8; ++nd)
                out[nd] = mma(load<16>(value_wave ? dt : qt, 16 * nd + c, 8 * h), derivative, out[nd]);
            __syncthreads();
        }
    if (live) {
        View dst = value_wave ? dv : dk;
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                store_rounded(dst, size_t(key) * dst.ld + kh * 128 + 16 * nd + 8 * h + i, out[nd][i]);
    }
}
} // namespace

void flash_back_native(View q, View k, View v, View lse, View dy, View dq, View dk, View dv, View delta,
                       int h, int kh, const int32_t *starts, const int32_t *ends, hipStream_t stream) {
    const char *setting = std::getenv("QTIP_TRAIN_ATTN_PROFILE");
    require(!setting || std::string(setting) == "0" || std::string(setting) == "1",
            "QTIP_TRAIN_ATTN_PROFILE must be 0 or 1");
    bool profile = setting && std::string(setting) == "1";
    const char *dkv_setting = std::getenv("QTIP_TRAIN_ATTN_DKV");
    std::string dkv = dkv_setting ? dkv_setting : "balanced";
    require(dkv == "balanced" || dkv == "balanced32" || dkv == "balanced128" || dkv == "wave" ||
                dkv == "staged",
            "QTIP_TRAIN_ATTN_DKV must be balanced, balanced32, balanced128, wave or staged");
    struct Events {
        hipEvent_t begin{}, middle{}, end{};
        ~Events() {
            if (begin)
                (void)hipEventDestroy(begin);
            if (middle)
                (void)hipEventDestroy(middle);
            if (end)
                (void)hipEventDestroy(end);
        }
    } e;
    if (profile) {
        HIP_CHECK(hipEventCreate(&e.begin));
        HIP_CHECK(hipEventCreate(&e.middle));
        HIP_CHECK(hipEventCreate(&e.end));
        HIP_CHECK(hipEventRecord(e.begin, stream));
    }
    dq_kernel<<<dim3((q.rows + 15) / 16, kh), (h / kh) * 32, 0, stream>>>(q, k, v, lse, dy, delta, dq, h / kh,
                                                                          starts);
    if (profile)
        HIP_CHECK(hipEventRecord(e.middle, stream));
    if (dkv == "staged")
        dkv_staged_kernel<<<dim3((q.rows + 15) / 16, kh), 128, 0, stream>>>(
            q, k, v, lse, dy, delta, dk, dv, h / kh, starts, ends);
    else if (dkv == "wave")
        dkv_wave_kernel<<<dim3((q.rows + 31) / 32, kh), 128, 0, stream>>>(
            q, k, v, lse, dy, delta, dk, dv, h / kh, starts, ends);
    else if (dkv == "balanced32")
        dkv_balanced_kernel<2><<<dim3((q.rows + 31) / 32, kh), 128, 0, stream>>>(
            q, k, v, lse, dy, delta, dk, dv, h / kh, starts, ends);
    else if (dkv == "balanced128")
        dkv_balanced_kernel<8><<<dim3((q.rows + 127) / 128, kh), 512, 0, stream>>>(
            q, k, v, lse, dy, delta, dk, dv, h / kh, starts, ends);
    else
        dkv_balanced_kernel<4><<<dim3((q.rows + 63) / 64, kh), 256, 0, stream>>>(
            q, k, v, lse, dy, delta, dk, dv, h / kh, starts, ends);
    if (profile) {
        HIP_CHECK(hipEventRecord(e.end, stream));
        HIP_CHECK(hipEventSynchronize(e.end));
        float qms = 0, kvms = 0;
        HIP_CHECK(hipEventElapsedTime(&qms, e.begin, e.middle));
        HIP_CHECK(hipEventElapsedTime(&kvms, e.middle, e.end));
        int device = 0;
        HIP_CHECK(hipGetDevice(&device));
        std::cerr << json({{"attention_backward_device", device},
                           {"seqlen", q.rows},
                           {"heads", h},
                           {"kv_heads", kh},
                           {"dkv_kernel", dkv},
                           {"dq_ms", qms},
                           {"dkv_ms", kvms}})
                         .dump()
                  << '\n';
    }
}
} // namespace qat::gpu
