// Microbenchmarks for batched decode / prefill on the codebook-major ("sorted")
// AQLM 1x16 layout with 16-row tiles and per-chunk base codes.
//
// Kernels:
//   sorted_gemv16      rows=1 int64-exact stream kernel on 16-row tiles (baseline check)
//   sorted_lds_gemm    one block per 16-row tile: decode the tile's sorted stream into an
//                      LDS code tile [groups][16] (2-byte LDS writes, 8 chunks per wave
//                      step, no serial scan), then every wave gathers weight tiles from the
//                      LDS codes and runs WMMA against up to 64 activation rows.
//   sorted_lds_recon   same decode phase, then coalesced dense FP8 weight writes
//                      (row-major or group-major) for prefill (BLAS / dense WMMA).
// Build (from cpp_runtime/):
//   hipcc --offload-arch=gfx1201 -O3 -std=c++20 -mno-wavefrontsize64 -fopenmp -Iinclude \
//         -I/opt/rocm/include perf/micro_batched.hip src/sorted_layout.cpp -o perf/micro_batched
// Usage: micro_batched [gemv|gemm|recon|all] [device]   (NCOPY=4 defeats the MALL)
#include "behemoth/sorted_layout.hpp"
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using behemoth::SortedLayout;
using behemoth::SortedTileDesc;
using F8 = rocwmma::float8_t;

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e = (x);                                                                        \
        if (e != hipSuccess) {                                                                     \
            fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, hipGetErrorString(e), __FILE__,         \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

struct Timer {
    hipEvent_t a, b;
    Timer() {
        CK(hipEventCreate(&a));
        CK(hipEventCreate(&b));
    }
    template <class F> double ms(F f, int iters = 20) {
        f();
        CK(hipDeviceSynchronize());
        std::vector<float> t;
        for (int i = 0; i < iters; ++i) {
            CK(hipEventRecord(a));
            f();
            CK(hipEventRecord(b));
            CK(hipEventSynchronize(b));
            float el;
            CK(hipEventElapsedTime(&el, a, b));
            t.push_back(el);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    }
};

constexpr int TR = 16, G = 2048, GB = 11;

__device__ __forceinline__ uint64_t exact_fixed18(float v) {
    const float a_f = floorf(v * 4.f);
    const float b_f = fmaf(a_f, -65536.f, v * 262144.f);
    const int32_t a = int32_t(a_f);
    const uint32_t b = uint32_t(b_f);
    const uint32_t lo = (uint32_t(a) << 16) | b, hi = uint32_t(a >> 16);
    return (uint64_t(hi) << 32) | lo;
}
__device__ __forceinline__ void wave_lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ------------------------------------------------------------ rows=1 stream kernel
// Per-chunk base codes replace the segment running code (one uniform load per chunk
// instead of a popcount total). grid = (ceil(tiles/W) * splits), block W*32.
template <int W, int UNROLL, int C, int DB>
__global__ void __launch_bounds__(W * 32)
sorted_gemv16(const uint8_t *__restrict__ x, const uint16_t *__restrict__ idx,
              const uint32_t *__restrict__ planes, const uint16_t *__restrict__ chunk_codes,
              const SortedTileDesc *__restrict__ tiles, const uint8_t *__restrict__ book,
              float *__restrict__ partial, int in, int out, int tile_count, int splits) {
    constexpr int THREADS = W * 32;
    __shared__ uint64_t xs[G];
    __shared__ uint64_t ys[W * 32 * C];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int tg = blockIdx.x / splits, split = blockIdx.x % splits, groups = in / 8;
    const int tile = tg * W + wave;
    for (int i = tid; i < G; i += THREADS)
        xs[i] = i < groups ? reinterpret_cast<const uint64_t *>(x)[i] : 0;
    for (int i = tid; i < W * 32 * C; i += THREADS)
        ys[i] = 0;
    __syncthreads();
    if (tile < tile_count) {
        const SortedTileDesc t = tiles[tile];
        const uint32_t le = (2u << lane) - 1;
        const uint32_t c0 = __builtin_amdgcn_readfirstlane(
                           t.chunk_begin + (t.chunk_count * uint32_t(split)) / uint32_t(splits)),
                       c1 = __builtin_amdgcn_readfirstlane(
                           t.chunk_begin + (t.chunk_count * uint32_t(split + 1)) / uint32_t(splits));
        uint64_t *yi = ys + wave * (32 * C) + (lane & (C - 1)) * 32;
        auto step = [&](uint32_t cc, int n) {
            uint32_t e[UNROLL], off[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if (u >= n)
                    break;
                e[u] = idx[size_t(cc + u) * 32 + lane];
                uint32_t incl = chunk_codes[cc + u];
#pragma unroll
                for (int k = 0; k < DB; ++k)
                    incl += uint32_t(__builtin_popcount(planes[size_t(cc + u) * DB + k] & le)) << k;
                off[u] = incl << 3;
            }
            uint2 w[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                if (u < n)
                    w[u] = *reinterpret_cast<const uint2 *>(book + off[u]);
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if (u >= n)
                    break;
                const int r = int(e[u] >> GB), grp = int(e[u] & (G - 1));
                const uint64_t xv = xs[grp];
                float v = __builtin_amdgcn_dot4_f32_fp8_fp8(w[u].x, uint32_t(xv), 0.f);
                v = __builtin_amdgcn_dot4_f32_fp8_fp8(w[u].y, uint32_t(xv >> 32), v);
                __hip_atomic_fetch_add(&yi[r], exact_fixed18(v), __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_WORKGROUP);
            }
        };
        uint32_t c = c0;
        for (; c + UNROLL <= c1; c += UNROLL)
            step(c, UNROLL);
        for (; c < c1; ++c)
            step(c, 1);
        wave_lds_sync();
        int64_t acc = 0;
        const uint64_t *base = ys + wave * (32 * C);
#pragma unroll
        for (int k = 0; k < C; ++k)
            acc += int64_t(base[k * 32 + lane]);
        const int r = tile * TR + lane;
        if (lane < TR && r < out)
            partial[size_t(split) * size_t(out) + r] = float(double(acc) * (1.0 / 262144.0));
    }
}

// ------------------------------------------------------- decode a tile into LDS
// ct[g * 16 + row] = code. 8 chunks per wave step (4 lanes per chunk, 8 entries per
// lane as one 16-byte load); U steps are loaded before any is decoded.
typedef unsigned u32x4_t __attribute__((ext_vector_type(4)));
template <int DB, int U, int AUX = -1, bool HALF = false>
__device__ __forceinline__ void decode_tile_to_lds(uint16_t *ct, const SortedTileDesc &t,
                                                   const uint16_t *__restrict__ idx,
                                                   const uint32_t *__restrict__ planes,
                                                   const uint16_t *__restrict__ chunk_codes,
                                                   int groups, int wave, int waves, int lane,
                                                   int gbeg = 0) { // window [gbeg, gbeg + groups)
    constexpr bool NT = AUX == -2;
    const uint32_t octets = HALF ? (t.chunk_count + 15) >> 4 : (t.chunk_count + 7) >> 3;
    const __amdgpu_buffer_rsrc_t ridx = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t *>(idx), 0, 0x7fffffff, 0x31004000),
                                 rpl = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint32_t *>(planes), 0, 0x7fffffff, 0x31004000),
                                 rcc = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t *>(chunk_codes), 0, 0x7fffffff, 0x31004000);
    const uint32_t sub = lane >> 2, part = lane & 3;
    for (uint32_t o = wave; o < octets; o += waves * U) {
        uint4 e[U];
        uint32_t pl[U][DB], base[U];
        bool valid[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint32_t rel = (o + u * waves) * 8 + sub;
            valid[u] = (o + u * waves) < octets && rel < t.chunk_count;
            const uint32_t cc = t.chunk_begin + (valid[u] ? rel : 0);
            if constexpr (AUX >= 0) {
                const u32x4_t v = __builtin_amdgcn_raw_buffer_load_b128(ridx, int((size_t(cc) * 32 + part * 8) * 2), 0, AUX);
                e[u] = uint4{v.x, v.y, v.z, v.w};
#pragma unroll
                for (int k = 0; k < DB; ++k)
                    pl[u][k] = __builtin_amdgcn_raw_buffer_load_b32(rpl, int((size_t(cc) * DB + k) * 4), 0, AUX);
                base[u] = __builtin_amdgcn_raw_buffer_load_b16(rcc, int(size_t(cc) * 2), 0, AUX);
            } else if constexpr (NT) {
                const u32x4_t v = __builtin_nontemporal_load(
                    reinterpret_cast<const u32x4_t *>(idx + size_t(cc) * 32 + part * 8));
                e[u] = uint4{v.x, v.y, v.z, v.w};
#pragma unroll
                for (int k = 0; k < DB; ++k)
                    pl[u][k] = __builtin_nontemporal_load(planes + size_t(cc) * DB + k);
                base[u] = __builtin_nontemporal_load(chunk_codes + cc);
            } else {
                e[u] = *reinterpret_cast<const uint4 *>(idx + size_t(cc) * 32 + part * 8);
#pragma unroll
                for (int k = 0; k < DB; ++k)
                    pl[u][k] = planes[size_t(cc) * DB + k];
                base[u] = chunk_codes[cc];
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!valid[u])
                continue;
            const uint32_t ew[4] = {e[u].x, e[u].y, e[u].z, e[u].w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t p = part * 8 + j, mask = (2u << p) - 1;
                uint32_t code = base[u];
#pragma unroll
                for (int k = 0; k < DB; ++k)
                    code += uint32_t(__builtin_popcount(pl[u][k] & mask)) << k;
                const uint32_t ent = (ew[j >> 1] >> (16 * (j & 1))) & 0xffffu;
                const int r = int(ent >> GB), g = int(ent & (G - 1)) - gbeg;
                if (g >= 0 && g < groups)
                    ct[g * TR + r] = uint16_t(code);
            }
        }
    }
}

// ------------------------------------------------------------- batched GEMM
// grid = (tile_count, ceil(rows / (16*RT))), block = W*32 (W from blockDim), dynamic LDS:
//   max(gpad*32, W*RT*1024) + W * 16 * (KT*8+16)
// x must be readable for 16*RT*gridDim.y rows and gpad*8 columns (padded workspace).
template <int RT, int KT, int DB, int PHASE = 0> // PHASE 1: decode only, 2: gemm only
__global__ void __launch_bounds__(512)
sorted_lds_gemm(const uint8_t *__restrict__ x, const uint16_t *__restrict__ idx,
                const uint32_t *__restrict__ planes, const uint16_t *__restrict__ chunk_codes,
                const SortedTileDesc *__restrict__ tiles, const uint8_t *__restrict__ book,
                float *__restrict__ y, const float *__restrict__ xs, const float *__restrict__ ws,
                float bs, bool apply_rows, int rows, int in, int out, int gpad) {
    static_assert(KT == 4 || KT == 8 || KT == 16);
    constexpr int LDB = KT * 8 + 16, BST = TR * LDB, GPL = KT / 2; // gathers per lane per step
    extern __shared__ __attribute__((aligned(16))) uint8_t lds[];
    uint16_t *ct = reinterpret_cast<uint16_t *>(lds);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, waves = blockDim.x >> 5;
    const int tile = blockIdx.x, groups = in / 8, row0 = blockIdx.y * (TR * RT);
    const SortedTileDesc t = tiles[tile];
    if (PHASE != 2)
        decode_tile_to_lds<DB, 4>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane);
    __syncthreads();
    if (PHASE == 1) {
        if (tid < 16 && ct[tid] == 0xfffe && ct[tid + 16] == 0x1234)
            y[tid] = 1.f;
        return;
    }
    const int red_bytes = max(gpad * 32, waves * RT * 1024);
    F8 *b = reinterpret_cast<F8 *>(lds + red_bytes + wave * BST);
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> acc[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r)
        rocwmma::fill_fragment(acc[r], 0.f);
    const int steps = gpad / KT;
    const int col = lane & 15, kc0 = lane >> 4;
    auto gather = [&](int s, uint64_t (&w)[GPL]) {
        uint32_t code[GPL];
#pragma unroll
        for (int i = 0; i < GPL; ++i)
            code[i] = ct[(s * KT + kc0 + 2 * i) * TR + col];
        asm volatile("" ::: "memory");
#pragma unroll
        for (int i = 0; i < GPL; ++i)
            w[i] = (s * KT + kc0 + 2 * i) < groups
                       ? reinterpret_cast<const uint64_t *>(book)[code[i]]
                       : 0;
    };
    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major>;
    constexpr int KS = KT * 8 / 32; // WMMA K sub-steps per step
    auto load_a = [&](int s, FragA (&a)[KS][RT]) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#pragma unroll
            for (int r = 0; r < RT; ++r)
                if (PHASE == 3)
                    rocwmma::load_matrix_sync(a[ks][r], b + ks * 32 + r, LDB);
                else
                    rocwmma::load_matrix_sync(
                        a[ks][r], reinterpret_cast<const F8 *>(x) + size_t(row0 + r * 16) * size_t(in) +
                                      size_t(s) * (KT * 8) + ks * 32,
                        in);
    };
    uint64_t w[GPL];
    FragA a[KS][RT];
    int s = wave;
    if (s < steps) {
        gather(s, w);
        load_a(s, a);
    }
    for (; s < steps; s += waves) {
#pragma unroll
        for (int i = 0; i < GPL; ++i)
            *reinterpret_cast<uint64_t *>(b + col * LDB + (kc0 + 2 * i) * 8) = w[i];
        FragA an[KS][RT];
        if (s + waves < steps) {
            gather(s + waves, w);
            load_a(s + waves, an);
        }
        wave_lds_sync();
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::col_major> mb;
            rocwmma::load_matrix_sync(mb, b + ks * 32, LDB);
#pragma unroll
            for (int r = 0; r < RT; ++r)
                rocwmma::mma_sync(acc[r], a[ks][r], mb, acc[r]);
        }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#pragma unroll
            for (int r = 0; r < RT; ++r)
                a[ks][r] = an[ks][r];
        wave_lds_sync();
    }
    __syncthreads(); // every wave is done with the code tile: reuse it for the reduction
    float *red = reinterpret_cast<float *>(lds) + wave * (RT * 256);
#pragma unroll
    for (int r = 0; r < RT; ++r)
        rocwmma::store_matrix_sync(red + r * 256, acc[r], 16, rocwmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < RT * 256; i += blockDim.x) {
        float sum = 0;
        for (int wv = 0; wv < waves; ++wv)
            sum += reinterpret_cast<const float *>(lds)[wv * (RT * 256) + i];
        const int row = row0 + i / 16, c = tile * TR + i % 16;
        if (row < rows && c < out)
            y[size_t(row) * size_t(out) + size_t(c)] =
                sum * xs[row] * bs * (apply_rows ? ws[c] : 1.f);
    }
}


// ------------------------------------------------- batched GEMM, raw WMMA operands
// gfx12 Wave32 v_wmma_f32_16x16x16_fp8_fp8 operand layout: lane l holds A[row l%16][k 8*(l/16)..+8]
// (8 consecutive activations of one batch row = one 8-byte load) and B[k 8*(l/16)..+8][col l%16]
// (= one gathered codebook entry of output column l%16, group g0 + l/16). No LDS staging:
// the 8-byte codebook entry is the B operand. D: lane l holds rows 8*(l/16)+i, column l%16.
__device__ __forceinline__ uint32_t hash_code(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v & 0xffffu;
}
typedef int v2i_t __attribute__((ext_vector_type(2)));
typedef float v8f_t __attribute__((ext_vector_type(8)));
__device__ __forceinline__ v8f_t wmma_fp8(uint64_t a, uint64_t b, v8f_t c) {
    v2i_t av = {int(uint32_t(a)), int(uint32_t(a >> 32))}, bv = {int(uint32_t(b)), int(uint32_t(b >> 32))};
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, c);
}
// grid = (tile_count, ceil(rows / (16*RT))), block = W*32, dynamic LDS = max(gpad*32, W*RT*1024).
// gpad: groups rounded up to a multiple of 2*U. x readable for 16*RT*gridDim.y rows, gpad*8 cols.
template <int RT, int U, int DB, int PHASE = 0>
__global__ void __launch_bounds__(512)
sorted_lds_gemm2(const uint8_t *__restrict__ x, const uint16_t *__restrict__ idx,
                 const uint32_t *__restrict__ planes, const uint16_t *__restrict__ chunk_codes,
                 const SortedTileDesc *__restrict__ tiles, const uint8_t *__restrict__ book,
                 float *__restrict__ y, const float *__restrict__ xs, const float *__restrict__ ws,
                 float bs, bool apply_rows, int rows, int in, int out, int gpad) {
    extern __shared__ __attribute__((aligned(16))) uint8_t lds[];
    uint16_t *ct = reinterpret_cast<uint16_t *>(lds);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, waves = blockDim.x >> 5;
    const int tile = blockIdx.x, row0 = blockIdx.y * (TR * RT);
    const int ldx = (PHASE & (1 << 21)) ? in + 64 : in; // bit 21: padded activation stride
    // PHASE bit 20: K windows of gpad groups each (blockIdx.z), outputs atomically added.
    const int gbeg = (PHASE & (1 << 20)) ? int(blockIdx.z) * gpad : 0;
    const int groups = min(in / 8 - gbeg, gpad);
    const SortedTileDesc t = tiles[tile];
    if (PHASE & 32) { // read the stream, discard it: L2 pollution test
        uint32_t acc0 = 0;
        const uint32_t octets = (t.chunk_count + 7) >> 3;
        for (uint32_t o = wave; o < octets; o += waves) {
            const uint32_t rel = o * 8 + (lane >> 2);
            const uint32_t cc = t.chunk_begin + (rel < t.chunk_count ? rel : 0);
            const uint4 e = *reinterpret_cast<const uint4 *>(idx + size_t(cc) * 32 + (lane & 3) * 8);
            acc0 ^= e.x ^ e.y ^ e.z ^ e.w ^ planes[size_t(cc) * DB] ^ chunk_codes[cc];
        }
        if (acc0 == 0x12345678u)
            ct[lane] = 1;
    } else if ((PHASE & 3) != 2 && (!(PHASE & (128 | 512)) || (PHASE & 1024))) {
        if (PHASE & 16)
            decode_tile_to_lds<DB, 4, -2>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane, gbeg);
        else if ((PHASE & 8192) && (PHASE & (1 << 14)))
            decode_tile_to_lds<DB, 4, ((PHASE >> 15) & 31), true>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane, gbeg);
        else if (PHASE & 8192)
            decode_tile_to_lds<DB, 4, -1, true>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane, gbeg);
        else if (PHASE & (1 << 14))
            decode_tile_to_lds<DB, 4, ((PHASE >> 15) & 31)>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane, gbeg);
        else if (PHASE & 2048)
            decode_tile_to_lds<DB, 8>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane, gbeg);
        else if (PHASE & 4096)
            decode_tile_to_lds<DB, 2>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane, gbeg);
        else
            decode_tile_to_lds<DB, 4>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane, gbeg);
    }
    __syncthreads();
    if ((PHASE & 3) == 1) {
        if (tid < 16 && ct[tid] == 0xfffe && ct[tid + 16] == 0x1234)
            y[tid] = 1.f;
        return;
    }
    // PHASE bits: 4 = all gathers hit one line (code 0), 8 = no x loads (diagnostics).
    const int col = lane & 15, gh = lane >> 4;
    const uint64_t *b64 = reinterpret_cast<const uint64_t *>(book);
    const uint64_t *x64 = reinterpret_cast<const uint64_t *>(x);
    const int units = gpad / (2 * U);
    // Unit = 2U groups; lane half h owns groups g0 + h*U + i (i < U), so its U activation
    // slices are contiguous (8U bytes); substep i multiplies slice i of both halves.
    struct Unit {
        uint64_t w[U], a[U][RT];
    };
    auto load = [&](int unit, Unit &un) {
        const int g0 = unit * (2 * U) + gh * U; // window-relative group
        uint32_t code[U];
#pragma unroll
        for (int i = 0; i < U; ++i) {
            code[i] = (PHASE & 4)     ? 0u
                      : (PHASE & 512)   ? hash_code(uint32_t(tile) * 65536u + uint32_t(g0 + i) * 16u + uint32_t(col))
                      : (PHASE & 128) ? uint32_t(idx[size_t(gbeg + g0 + i) * size_t(out) + size_t(tile * TR + col)])
                                      : uint32_t(ct[(g0 + i) * TR + col]);
            if (PHASE & 256)
                code[i] &= 8191u; // 64 KiB codebook working set
        }
        asm volatile("" ::: "memory");
#pragma unroll
        for (int i = 0; i < U; ++i)
            un.w[i] = (g0 + i) < groups ? b64[code[i]] : 0;
        if (!(PHASE & 8)) {
#pragma unroll
            for (int r = 0; r < RT; ++r) {
                const uint64_t *xr = x64 + (size_t(row0 + r * 16 + col) * size_t(ldx) >> 3) + size_t(gbeg + g0);
                if constexpr (U % 2 == 0) {
#pragma unroll
                    for (int i = 0; i < U; i += 2) {
                        const uint4 v = *reinterpret_cast<const uint4 *>(xr + i);
                        un.a[i][r] = uint64_t(v.x) | (uint64_t(v.y) << 32);
                        un.a[i + 1][r] = uint64_t(v.z) | (uint64_t(v.w) << 32);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < U; ++i)
                        un.a[i][r] = xr[i];
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < U; ++i)
#pragma unroll
                for (int r = 0; r < RT; ++r)
                    un.a[i][r] = un.w[i] ^ uint64_t(r);
        }
        asm volatile("" ::: "memory");
    };
    v8f_t acc[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r)
        acc[r] = v8f_t{};
    auto compute = [&](const Unit &un) {
#pragma unroll
        for (int i = 0; i < U; ++i)
#pragma unroll
            for (int r = 0; r < RT; ++r)
                acc[r] = wmma_fp8(un.a[i][r], un.w[i], acc[r]);
    };
    if constexpr (PHASE & 64) {
        // Two units in flight at all times (three register slots, loop unrolled by 3).
        Unit ua, ub, uc;
        int u = wave;
        const int w1 = waves, w2 = 2 * waves;
        if (u < units)
            load(u, ua);
        if (u + w1 < units)
            load(u + w1, ub);
        for (; u < units; u += 3 * waves) {
            if (u + w2 < units)
                load(u + w2, uc);
            compute(ua);
            if (u + w1 >= units)
                break;
            if (u + w2 + w1 < units)
                load(u + w2 + w1, ua);
            compute(ub);
            if (u + w2 >= units)
                break;
            if (u + w2 + w2 < units)
                load(u + w2 + w2, ub);
            compute(uc);
        }
    } else {
        Unit cur, nxt;
        int u = wave;
        if (u < units)
            load(u, cur);
        for (; u < units; u += waves) {
            if (u + waves < units)
                load(u + waves, nxt);
            compute(cur);
            cur = nxt;
        }
    }
    __syncthreads(); // every wave is done with the code tile: reuse it for the reduction
    float *red = reinterpret_cast<float *>(lds);
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            red[((wave * RT + r) * 8 + i) * 32 + lane] = acc[r][i];
    __syncthreads();
    for (int e = tid; e < RT * 256; e += blockDim.x) {
        const int r = e >> 8, i = (e >> 5) & 7, l = e & 31;
        float sum = 0;
        for (int wv = 0; wv < waves; ++wv)
            sum += red[((wv * RT + r) * 8 + i) * 32 + l];
        const int row = row0 + r * 16 + 8 * (l >> 4) + i, c = tile * TR + (l & 15);
        if (row < rows && c < out) {
            if (PHASE & (1 << 20))
                atomicAdd(&y[size_t(row) * size_t(out) + size_t(c)], sum * xs[row] * bs * (apply_rows ? ws[c] : 1.f));
            else
                y[size_t(row) * size_t(out) + size_t(c)] = sum * xs[row] * bs * (apply_rows ? ws[c] : 1.f);
        }
    }
}

// ------------------------------------------------------- dense reconstruct
// grid = tile_count, block = W*32, dynamic LDS = gpad*32.
template <int DB>
__global__ void __launch_bounds__(512)
sorted_lds_recon(const uint16_t *__restrict__ idx, const uint32_t *__restrict__ planes,
                 const uint16_t *__restrict__ chunk_codes, const SortedTileDesc *__restrict__ tiles,
                 const uint8_t *__restrict__ book, uint8_t *__restrict__ weight, int in, int out,
                 bool group_major) {
    extern __shared__ __attribute__((aligned(16))) uint8_t lds[];
    uint16_t *ct = reinterpret_cast<uint16_t *>(lds);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, waves = blockDim.x >> 5;
    const int tile = blockIdx.x, groups = in / 8;
    const SortedTileDesc t = tiles[tile];
    decode_tile_to_lds<DB, 4>(ct, t, idx, planes, chunk_codes, groups, wave, waves, lane);
    __syncthreads();
    const uint64_t *b64 = reinterpret_cast<const uint64_t *>(book);
    uint64_t *w64 = reinterpret_cast<uint64_t *>(weight);
    const int rows_here = min(TR, out - tile * TR);
    if (group_major) {
        // consecutive lanes: consecutive rows of one group (128-byte runs)
        for (int i = tid; i < groups * TR; i += blockDim.x) {
            const int r = i & 15, g = i >> 4;
            if (r < rows_here)
                w64[size_t(g) * size_t(out) + size_t(tile * TR + r)] = b64[ct[g * TR + r]];
        }
    } else {
        for (int r = 0; r < rows_here; ++r)
            for (int g = tid; g < groups; g += blockDim.x)
                w64[size_t(tile * TR + r) * size_t(groups) + size_t(g)] = b64[ct[g * TR + r]];
    }
}

// ---------------------------------------------------------------- host side
static float fp8_to_f(uint8_t v) {
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    if (e == 15 && m == 7)
        return NAN;
    float f = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -f : f;
}
static uint8_t f_to_fp8(float f) {
    if (f == 0)
        return 0;
    uint8_t s = f < 0 ? 0x80 : 0;
    f = std::fabs(f);
    int e;
    float m = std::frexp(f, &e);
    e -= 1;
    m *= 2;
    if (e < -6) {
        int q = int(std::round(f * 512.f));
        if (q > 7)
            q = 7;
        return s | uint8_t(q);
    }
    if (e > 8)
        return s | 0x7e;
    int q = int(std::round((m - 1) * 8));
    if (q == 8) {
        q = 0;
        ++e;
        if (e > 8)
            return s | 0x7e;
    }
    return s | uint8_t((e + 7) << 3) | uint8_t(q);
}
template <class T> T *upload(const std::vector<T> &v) {
    T *p;
    CK(hipMalloc(&p, std::max<size_t>(v.size() * sizeof(T), 16)));
    CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
struct Shape {
    const char *name;
    int in, out;
};
static const Shape shapes[] = {{"q 12288->6144", 12288, 6144},
                               {"gate/up 12288->14336", 12288, 14336},
                               {"down 14336->12288", 14336, 12288}};
struct DevLayout {
    uint16_t *idx, *cc;
    uint32_t *pl;
    SortedTileDesc *tiles;
};
static DevLayout up(const SortedLayout &L) {
    return {upload(L.idx), upload(L.chunk_codes), upload(L.planes), upload(L.tiles)};
}
static void freeL(DevLayout &d) {
    CK(hipFree(d.idx));
    CK(hipFree(d.cc));
    CK(hipFree(d.pl));
    CK(hipFree(d.tiles));
}

int main(int argc, char **argv) {
    std::string what = argc > 1 ? argv[1] : "all";
    int dev = argc > 2 ? atoi(argv[2]) : 0;
    CK(hipSetDevice(dev));
    hipDeviceProp_t p{};
    CK(hipGetDeviceProperties(&p, dev));
    printf("# %s WGPs=%d clock=%d MHz\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    const int NCOPY = getenv("NCOPY") ? atoi(getenv("NCOPY")) : 1;
    Timer T;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> hbook(65536 * 8);
    for (auto &v : hbook)
        v = f_to_fp8(nd(rng) * 0.5f);
    std::vector<float> bookf(hbook.size());
    for (size_t i = 0; i < bookf.size(); ++i)
        bookf[i] = fp8_to_f(hbook[i]);
    uint8_t *book = upload(hbook);
    constexpr int MMAX = 64;
    for (const auto &s : shapes) {
        const int groups = s.in / 8;
        std::vector<uint16_t> codes(size_t(s.out) * groups);
        for (auto &c : codes)
            c = uint16_t(rng());
        const int rows_pad = MMAX;
        const int gpad16 = (groups + 15) / 16 * 16;
        std::vector<uint8_t> hx(size_t(rows_pad) * (s.in + 64) + size_t(gpad16) * 8 + 256, 0);
        for (size_t i = 0; i < size_t(MMAX) * s.in; ++i)
            hx[i] = f_to_fp8(nd(rng) * 0.3f);
        // padded-stride copy (bit 21 variants): row m at m*(in+64)
        std::vector<uint8_t> hxp(hx.size(), 0);
        for (int m = 0; m < MMAX; ++m)
            std::copy_n(&hx[size_t(m) * s.in], s.in, &hxp[size_t(m) * (s.in + 64)]);
        uint8_t *dxp = upload(hxp);
        uint8_t *dx = upload(hx);
        std::vector<float> xf(size_t(MMAX) * s.in);
        for (size_t i = 0; i < xf.size(); ++i)
            xf[i] = fp8_to_f(hx[i]);
        std::vector<float> ref(size_t(MMAX) * s.out);
#pragma omp parallel for
        for (int o = 0; o < s.out; ++o) {
            std::vector<float> wrow(size_t(s.in));
            for (int g = 0; g < groups; ++g)
                std::copy_n(&bookf[size_t(codes[size_t(o) * groups + g]) * 8], 8, &wrow[g * 8]);
            for (int m = 0; m < MMAX; ++m) {
                double acc = 0;
                const float *xx = &xf[size_t(m) * s.in];
                for (int k = 0; k < s.in; ++k)
                    acc += double(wrow[k]) * xx[k];
                ref[size_t(m) * s.out + o] = float(acc);
            }
        }
        std::vector<float> ones(MMAX, 1.f), onesw(s.out, 1.f);
        float *dxs = upload(ones), *dws = upload(onesw);
        float *y;
        CK(hipMalloc(&y, size_t(MMAX) * s.out * 4));
        float *partial;
        CK(hipMalloc(&partial, size_t(32) * s.out * 4));
        auto check = [&](int M, const char *tag) {
            std::vector<float> hy(size_t(M) * s.out);
            CK(hipMemcpy(hy.data(), y, hy.size() * 4, hipMemcpyDeviceToHost));
            double maxerr = 0, maxref = 0;
            for (size_t i = 0; i < hy.size(); ++i) {
                maxerr = std::max(maxerr, double(std::fabs(hy[i] - ref[i])));
                maxref = std::max(maxref, double(std::fabs(ref[i])));
            }
            printf("    %s maxerr=%.3g/%.3g\n", tag, maxerr, maxref);
        };
        printf("# %s entries=%zu (2B codes = %.1f MB)\n", s.name, codes.size(),
               codes.size() * 2 / 1e6);
        for (int DB : {3, 4}) {
            auto L = behemoth::sorted_layout(s.in, s.out, codes.data(), TR, G, DB, 16);
            std::vector<uint16_t> back;
            bool ok = behemoth::sorted_layout_decode(L, s.in, s.out, back) && back == codes;
            printf("  layout R=16 G=2048 DB=%d: chunks=%zu dummies=%.2f%% bytes/entry=%.3f "
                   "(%.1f MB) decode=%s\n",
                   DB, L.chunks(), 100. * L.dummies / L.entries, double(L.bytes()) / L.entries,
                   L.bytes() / 1e6, ok ? "ok" : "FAIL");
            std::vector<DevLayout> dl;
            for (int k = 0; k < NCOPY; ++k)
                dl.push_back(up(L));
            std::vector<uint16_t *> dgm; // group-major codes [groups][out] (old layout)
            {
                std::vector<uint16_t> gm(codes.size());
                for (int o = 0; o < s.out; ++o)
                    for (int g = 0; g < groups; ++g)
                        gm[size_t(g) * s.out + o] = codes[size_t(o) * groups + g];
                for (int k = 0; k < NCOPY; ++k)
                    dgm.push_back(upload(gm));
            }
            const int tiles = int(L.tiles.size());
            int iter = 0;
            auto gemv = [&]<int W, int U, int C, int DBT>(int splits) {
                if (DBT != DB)
                    return;
                const int blocks = (tiles + W - 1) / W * splits;
                auto launch = [&] {
                    auto &d = dl[(iter++) % NCOPY];
                    sorted_gemv16<W, U, C, DBT><<<blocks, W * 32>>>(dx, d.idx, d.pl, d.cc, d.tiles,
                                                                    book, partial, s.in, s.out,
                                                                    tiles, splits);
                };
                double ms = T.ms(launch);
                printf("  gemv16 W=%d U=%d C=%d splits=%2d: %7.4f ms  logical(2B) %6.1f GB/s  "
                       "actual %6.1f GB/s\n",
                       W, U, C, splits, ms, codes.size() * 2 / ms / 1e6, L.bytes() / ms / 1e6);
                // correctness: merge splits on host
                std::vector<float> hp(size_t(splits) * s.out), hy(s.out);
                CK(hipMemcpy(hp.data(), partial, hp.size() * 4, hipMemcpyDeviceToHost));
                double maxerr = 0;
                for (int o = 0; o < s.out; ++o) {
                    float v = 0;
                    for (int k = 0; k < splits; ++k)
                        v += hp[size_t(k) * s.out + o];
                    maxerr = std::max(maxerr, double(std::fabs(v - ref[o])));
                }
                printf("    maxerr=%.3g\n", maxerr);
            };
            if (what == "all" || what == "gemv") {
                gemv.template operator()<8, 4, 2, 3>(4);
                gemv.template operator()<8, 4, 2, 3>(8);
                gemv.template operator()<8, 4, 2, 4>(4);
                gemv.template operator()<8, 4, 2, 4>(8);
            }
            auto gemm = [&]<int RT, int KT, int DBT, int PHASE = 0>(int W, int M) {
                if (DBT != DB)
                    return;
                hipFuncAttributes fa{};
                CK(hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(sorted_lds_gemm<RT, KT, DBT, PHASE>)));
                const int gpad = (groups + KT - 1) / KT * KT;
                const size_t lds = size_t(std::max(gpad * 32, W * RT * 1024)) + size_t(W) * TR * (KT * 8 + 16);
                if (lds > 65536) {
                    printf("  gemm RT=%d KT=%d W=%d: LDS %zu too large\n", RT, KT, W, lds);
                    return;
                }
                dim3 grid(tiles, (M + TR * RT - 1) / (TR * RT));
                auto launch = [&] {
                    auto &d = dl[(iter++) % NCOPY];
                    sorted_lds_gemm<RT, KT, DBT, PHASE><<<grid, W * 32, lds>>>(
                        dx, d.idx, d.pl, d.cc, d.tiles, book, y, dxs, dws, 1.f, false, M, s.in,
                        s.out, gpad);
                };
                CK(hipFuncSetAttribute(reinterpret_cast<const void *>(sorted_lds_gemm<RT, KT, DBT, PHASE>),
                                       hipFuncAttributeMaxDynamicSharedMemorySize, int(lds)));
                double ms = T.ms(launch);
                printf("  gemm%s M=%2d RT=%d KT=%2d W=%2d lds=%5zu vgpr=%d spill=%zu: %7.4f ms  logical(2B) %6.1f GB/s  "
                       "stream %6.1f GB/s  %.4f ms/row\n",
                       PHASE == 1 ? "-decode" : PHASE == 2 ? "-gemmonly" : PHASE == 3 ? "-gemm-ldsA" : "", M, RT, KT, W, lds, fa.numRegs, fa.localSizeBytes, ms, codes.size() * 2 / ms / 1e6, L.bytes() / ms / 1e6,
                       ms / M);
                if (PHASE == 0) {
                    launch();
                    check(M, "gemm");
                }
            };
            if (what == "phase") {
                for (int W : {4, 8, 16}) {
                    gemm.template operator()<1, 8, 3, 1>(W, 16);
                    gemm.template operator()<1, 8, 3, 2>(W, 16);
                    gemm.template operator()<1, 8, 3, 3>(W, 16);
                    gemm.template operator()<4, 8, 3, 2>(W, 64);
                    gemm.template operator()<4, 8, 3, 3>(W, 64);
                    gemm.template operator()<1, 4, 3, 2>(W, 16);
                }
            }
            if (what == "all" || what == "gemm") {
                for (int W : {4, 8}) {
                    gemm.template operator()<1, 8, 3>(W, 8);
                    gemm.template operator()<1, 8, 3>(W, 16);
                    gemm.template operator()<2, 8, 3>(W, 32);
                    gemm.template operator()<4, 8, 3>(W, 64);
                    gemm.template operator()<1, 4, 3>(W, 16);
                    gemm.template operator()<4, 4, 3>(W, 64);
                    gemm.template operator()<1, 8, 4>(W, 16);
                    gemm.template operator()<4, 8, 4>(W, 64);
                }
                gemm.template operator()<1, 16, 3>(4, 16);
                gemm.template operator()<4, 16, 3>(4, 64);
            }

            auto gemm2 = [&]<int RT, int U, int DBT, int PHASE = 0>(int W, int M) {
                if (DBT != DB)
                    return;
                hipFuncAttributes fa{};
                CK(hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(sorted_lds_gemm2<RT, U, DBT, PHASE>)));
                int gpad = (groups + 2 * U - 1) / (2 * U) * (2 * U);
                int windows = 1;
                if (PHASE & (1 << 20)) {
                    const int WIN = getenv("WIN") ? atoi(getenv("WIN")) : 1024;
                    windows = (groups + WIN - 1) / WIN;
                    gpad = ((groups + windows - 1) / windows + 2 * U - 1) / (2 * U) * (2 * U);
                }
                const size_t lds = size_t(std::max(gpad * 32, W * RT * 1024));
                if (lds > 65536) {
                    printf("  gemm2 RT=%d U=%d W=%d: LDS %zu too large\n", RT, U, W, lds);
                    return;
                }
                dim3 grid(tiles, (M + TR * RT - 1) / (TR * RT), windows);
                auto launch = [&] {
                    const int k = (iter++) % NCOPY;
                    auto &d = dl[k];
                    if (PHASE & (1 << 20))
                        CK(hipMemsetAsync(y, 0, size_t(M) * s.out * 4));
                    sorted_lds_gemm2<RT, U, DBT, PHASE><<<grid, W * 32, lds>>>(
                        (PHASE & (1 << 21)) ? dxp : dx, (PHASE & 128) ? dgm[k] : d.idx, d.pl, d.cc, d.tiles, book, y, dxs, dws,
                        1.f, false, M, s.in, s.out, gpad);
                };
                CK(hipFuncSetAttribute(reinterpret_cast<const void *>(sorted_lds_gemm2<RT, U, DBT, PHASE>),
                                       hipFuncAttributeMaxDynamicSharedMemorySize, int(lds)));
                double ms = T.ms(launch);
                printf("  gemm2%s%s%s M=%2d RT=%d U=%2d W=%2d lds=%5zu vgpr=%d spill=%zu: %7.4f ms  logical(2B) %6.1f GB/s  "
                       "stream %6.1f GB/s  %.4f ms/row\n",
                       (std::string((PHASE & (1 << 20)) ? "-WIN" + std::to_string(windows) : "") + ((PHASE & (1 << 21)) ? "-XPAD" : "")).c_str(),
                       (PHASE & 8192) ? "-HALFSTREAM" : (PHASE & (1 << 14)) ? (std::string("-AUX") + std::to_string((PHASE >> 15) & 31)).c_str() : "",
                       (PHASE & 1023) == 1 ? "-decode" : PHASE == 2 ? "-gemmonly" : PHASE == 4 ? "-hitgather" : PHASE == 8 ? "-nox" : PHASE == 12 ? "-hit-nox" : PHASE == 16 ? "-NT" : PHASE == 17 ? "-NT-decode" : (PHASE & 1023) == 512 ? "-HASH" : (PHASE & 1535) == 1536 ? "-HASH+decode" : PHASE == 1536 + 2048 ? "-HASH+decode8" : PHASE == 1536 + 4096 ? "-HASH+decode2" : PHASE == 1 + 2048 ? "-decode8" : PHASE == 1 + 4096 ? "-decode2" : PHASE == 1152 ? "-GMCODES+decode" : PHASE == 34 ? "-gemm+stream" : PHASE == 38 ? "-hit+stream" : PHASE == 258 ? "-gemmonly-64K" : PHASE == 290 ? "-gemm+stream-64K" : PHASE == 256 ? "-full-64K" : PHASE == 384 ? "-GMCODES-64K" : PHASE == 128 ? "-GMCODES" : PHASE == 192 ? "-GMCODES-ring3" : (PHASE & 64) ? "-ring3" : "", M, RT, U, W, lds, fa.numRegs, fa.localSizeBytes, ms, codes.size() * 2 / ms / 1e6, L.bytes() / ms / 1e6,
                       ms / M);
                if ((PHASE == 0 || PHASE == 64 || PHASE == 128 || (PHASE & 1023) == 0) && what != "phase2x") {
                    if (PHASE & (1 << 20))
                        CK(hipMemsetAsync(y, 0, size_t(M) * s.out * 4));
                    launch();
                    check(M, "gemm2");
                }
            };
            if (what == "phase10") {
                constexpr int A24 = (1 << 14) + (24 << 15), WN = 1 << 20, XP = 1 << 21;
                for (int W : {16}) {
                    gemm2.template operator()<1, 8, 3, A24 + WN>(W, 16);
                    gemm2.template operator()<1, 8, 3, A24 + WN + XP>(W, 16);
                    gemm2.template operator()<1, 8, 3, 512>(W, 16);
                    gemm2.template operator()<1, 8, 3, 512 + XP>(W, 16);
                    gemm2.template operator()<2, 4, 3, A24 + WN>(W, 32);
                    gemm2.template operator()<2, 4, 3, A24 + WN + XP>(W, 32);
                    gemm2.template operator()<4, 4, 3, A24 + WN>(W, 64);
                    gemm2.template operator()<4, 4, 3, A24 + WN + XP>(W, 64);
                    gemm2.template operator()<4, 2, 3, A24 + WN + XP>(W, 64);
                    gemm2.template operator()<4, 4, 3, 512 + XP>(W, 64);
                }
            }
            if (what == "phase9") {
                constexpr int A24 = (1 << 14) + (24 << 15), WN = 1 << 20;
                for (int W : {8, 16}) {
                    gemm2.template operator()<1, 8, 3, A24>(W, 16);
                    gemm2.template operator()<1, 8, 3, A24 + WN>(W, 16);
                    gemm2.template operator()<1, 8, 3, 512>(W, 16);
                    gemm2.template operator()<2, 4, 3, A24>(W, 32);
                    gemm2.template operator()<2, 4, 3, A24 + WN>(W, 32);
                    gemm2.template operator()<2, 4, 3, 512>(W, 32);
                    gemm2.template operator()<4, 4, 3, A24>(W, 64);
                    gemm2.template operator()<4, 4, 3, A24 + WN>(W, 64);
                    gemm2.template operator()<4, 4, 3, 512>(W, 64);
                }
            }
            if (what == "phase8") {
                constexpr int HD = 512 + 1024, A24 = (1 << 14) + (24 << 15), A25 = (1 << 14) + (25 << 15), A27 = (1 << 14) + (27 << 15);
                for (int W : {8, 16}) {
                    gemm2.template operator()<1, 8, 3, 512>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + A24>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + A25>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + A27>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + A24 + 8192>(W, 16);
                    gemm2.template operator()<1, 8, 3, A24>(W, 16);
                    gemm2.template operator()<2, 4, 3, 512>(W, 32);
                    gemm2.template operator()<2, 4, 3, HD + A24>(W, 32);
                    gemm2.template operator()<2, 4, 3, A24>(W, 32);
                    gemm2.template operator()<4, 4, 3, 512>(W, 64);
                    gemm2.template operator()<4, 4, 3, HD + A24>(W, 64);
                    gemm2.template operator()<4, 4, 3, A24>(W, 64);
                }
            }
            if (what == "phase7") {
                constexpr int HD = 512 + 1024;
                for (int W : {8}) {
                    gemm2.template operator()<1, 8, 3, 512>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + 8192>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + (1 << 14) + (0 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + (1 << 14) + (1 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + (1 << 14) + (3 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + (1 << 14) + (8 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + (1 << 14) + (16 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + (1 << 14) + (17 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, HD + (1 << 14) + (24 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, 1 + (1 << 14) + (16 << 15)>(W, 16);
                    gemm2.template operator()<1, 8, 3, 1 + (1 << 14) + (24 << 15)>(W, 16);
                }
            }
            if (what == "phase6") {
                for (int W : {8, 16}) {
                    gemm2.template operator()<1, 4, 3, 512>(W, 16);
                    gemm2.template operator()<1, 4, 3, 512 + 1024>(W, 16);
                    gemm2.template operator()<1, 8, 3, 512>(W, 16);
                    gemm2.template operator()<1, 8, 3, 512 + 1024>(W, 16);
                    gemm2.template operator()<1, 8, 3, 512 + 1024 + 2048>(W, 16);
                    gemm2.template operator()<1, 8, 3, 512 + 1024 + 4096>(W, 16);
                    gemm2.template operator()<1, 8, 3, 1>(W, 16);
                    gemm2.template operator()<1, 8, 3, 1 + 2048>(W, 16);
                    gemm2.template operator()<1, 8, 3, 1 + 4096>(W, 16);
                    gemm2.template operator()<2, 4, 3, 512>(W, 32);
                    gemm2.template operator()<2, 4, 3, 512 + 1024>(W, 32);
                    gemm2.template operator()<4, 2, 3, 512>(W, 64);
                    gemm2.template operator()<4, 2, 3, 512 + 1024>(W, 64);
                    gemm2.template operator()<4, 4, 3, 512>(W, 64);
                    gemm2.template operator()<4, 4, 3, 512 + 1024>(W, 64);
                }
            }
            if (what == "all" || what == "gemm2") {
                for (int W : {4, 8, 16}) {
                    gemm2.template operator()<1, 8, 3>(W, 16);
                    gemm2.template operator()<1, 8, 3, 2>(W, 16);
                    gemm2.template operator()<1, 4, 3>(W, 16);
                    gemm2.template operator()<1, 16, 3>(W, 16);
                    gemm2.template operator()<2, 8, 3>(W, 32);
                    gemm2.template operator()<2, 4, 3>(W, 32);
                    gemm2.template operator()<4, 4, 3>(W, 64);
                    gemm2.template operator()<4, 4, 3, 2>(W, 64);
                    gemm2.template operator()<4, 2, 3>(W, 64);
                    gemm2.template operator()<1, 8, 4>(W, 8);
                    gemm2.template operator()<1, 8, 4>(W, 16);
                    gemm2.template operator()<2, 8, 4>(W, 32);
                    gemm2.template operator()<4, 4, 4>(W, 64);
                }
            }
            if (what == "all" || what == "recon") {
                uint8_t *dw;
                CK(hipMalloc(&dw, size_t(s.in) * s.out));
                const int gpad = gpad16;
                for (bool gm : {false, true})
                    for (int W : {4, 8, 16}) {
                        auto launch = [&] {
                            auto &d = dl[(iter++) % NCOPY];
                            if (DB == 3)
                                sorted_lds_recon<3><<<tiles, W * 32, gpad * 32>>>(
                                    d.idx, d.pl, d.cc, d.tiles, book, dw, s.in, s.out, gm);
                            else
                                sorted_lds_recon<4><<<tiles, W * 32, gpad * 32>>>(
                                    d.idx, d.pl, d.cc, d.tiles, book, dw, s.in, s.out, gm);
                        };
                        double ms = T.ms(launch);
                        printf("  recon %s W=%2d: %7.4f ms  dense write %6.1f GB/s  stream %6.1f "
                               "GB/s\n",
                               gm ? "group-major" : "row-major  ", W, ms,
                               double(s.in) * s.out / ms / 1e6, L.bytes() / ms / 1e6);
                    }
                // verify row-major
                if (DB == 3)
                    sorted_lds_recon<3><<<tiles, 256, gpad * 32>>>(dl[0].idx, dl[0].pl, dl[0].cc,
                                                                    dl[0].tiles, book, dw, s.in,
                                                                    s.out, false);
                else
                    sorted_lds_recon<4><<<tiles, 256, gpad * 32>>>(dl[0].idx, dl[0].pl, dl[0].cc,
                                                                    dl[0].tiles, book, dw, s.in,
                                                                    s.out, false);
                std::vector<uint8_t> hw(size_t(s.in) * s.out);
                CK(hipMemcpy(hw.data(), dw, hw.size(), hipMemcpyDeviceToHost));
                size_t bad = 0;
                for (int o = 0; o < s.out && bad < 10; ++o)
                    for (int g = 0; g < groups; ++g)
                        if (memcmp(&hw[size_t(o) * s.in + g * 8],
                                   &hbook[size_t(codes[size_t(o) * groups + g]) * 8], 8)) {
                            ++bad;
                            break;
                        }
                printf("    recon row-major verify: %s\n", bad ? "FAIL" : "ok");
                CK(hipFree(dw));
            }
            for (auto &d : dl)
                freeL(d);
            for (auto *g : dgm)
                CK(hipFree(g));
        }
        CK(hipFree(dx));
        CK(hipFree(dxp));
        CK(hipFree(y));
        CK(hipFree(partial));
        CK(hipFree(dxs));
        CK(hipFree(dws));
    }
    return 0;
}
