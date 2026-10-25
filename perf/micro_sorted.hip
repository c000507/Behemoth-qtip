// Microbenchmarks for the codebook-major ("sorted tile") AQLM 1x16 layout.
// Build (from cpp_runtime/):
//   hipcc --offload-arch=gfx1201 -O3 -std=c++20 -mno-wavefrontsize64 -Iinclude \
//         perf/micro_sorted.hip src/sorted_layout.cpp -o perf/micro_sorted
// Usage: micro_sorted [lds|gather|gemv|all] [device]
#include "behemoth/sorted_layout.hpp"
#include <hip/hip_runtime.h>
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

__device__ __forceinline__ void lds_atomic_add(float *p, float v) {
    __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// ------------------------------------------------------------- LDS throughput
// Random ds_add_f32 into a table of `slots` floats; each thread does `iters`
// adds with a cheap LCG address stream (address generation is ~3 VALU ops).
template <int MODE> // 0 = atomic add, 1 = ds_read_b64, 2 = read + add
__global__ void lds_random(uint32_t seed, int slots_mask, int iters, float *out) {
    __shared__ float table[4096];
    __shared__ uint64_t xt[2048];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        table[i] = 0.f;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        xt[i] = i;
    __syncthreads();
    uint32_t s = seed ^ (threadIdx.x * 2654435761u) ^ (blockIdx.x * 40503u);
    uint64_t acc = 0;
    float f = 1.f;
    for (int i = 0; i < iters; ++i) {
        s = s * 1664525u + 1013904223u;
        int a = int(s >> 8) & slots_mask;
        if constexpr (MODE == 0)
            lds_atomic_add(&table[a], f);
        else if constexpr (MODE == 1)
            acc += xt[a & 2047];
        else if constexpr (MODE == 2) {
            acc += xt[a & 2047];
            lds_atomic_add(&table[a], f);
        } else if constexpr (MODE == 3) {
            __hip_atomic_fetch_add(reinterpret_cast<uint32_t *>(&table[a]), 1u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (MODE == 6) {
            __hip_atomic_fetch_add(&xt[a & 31], uint64_t(1),
                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        } else if constexpr (MODE == 4) {
            // lane-private slot: [row][lane] layout, row random in 0..31
            float *slot = &table[((a & 31) << 5) | (threadIdx.x & 31)];
            *slot += f;
        } else {
            acc += xt[a & 2047];
            float *slot = &table[((a & 31) << 5) | (threadIdx.x & 31)];
            *slot += f;
        }
        f += 1e-7f;
    }
    __syncthreads();
    if (acc == 0x1234567 || table[threadIdx.x] == 12345.f)
        out[0] = float(acc);
}

// ---------------------------------------------------- sorted vs random gather
// Gather-only kernel on the encoded sorted stream: idx+planes -> code ->
// 8-byte codebook read, xor-accumulated. `random`: read the code from idx
// directly (same bytes, random addresses), the current layout's access pattern.
template <int DB, bool random>
__global__ void sorted_gather(const uint16_t *__restrict__ idx, const uint32_t *__restrict__ planes,
                              const uint16_t *__restrict__ seg_codes, uint32_t chunks_per_segment,
                              uint32_t segments, const uint64_t *__restrict__ book, uint32_t *out) {
    const int lane = threadIdx.x & 31;
    const uint32_t wave_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5,
                   waves = (gridDim.x * blockDim.x) >> 5;
    const uint32_t lt = (1u << lane) - 1;
    uint64_t acc = 0;
    for (uint32_t seg = wave_global; seg < segments; seg += waves) {
        uint32_t running = seg_codes[seg];
        const uint32_t c0 = seg * chunks_per_segment, c1 = c0 + chunks_per_segment;
        for (uint32_t c = c0; c < c1; ++c) {
            uint32_t e = idx[c * 32 + lane];
            uint32_t code;
            if constexpr (random) {
                code = e;
            } else {
                uint32_t pre = 0, tot = 0, d = 0;
#pragma unroll
                for (int k = 0; k < DB; ++k) {
                    uint32_t p = planes[c * DB + k];
                    pre += __builtin_popcount(p & lt) << k;
                    tot += __builtin_popcount(p) << k;
                    d |= ((p >> lane) & 1) << k;
                }
                code = (running + pre + d) & 0xffff;
                running += tot;
            }
            acc ^= book[code];
        }
    }
    if (acc == 0xdeadbeef)
        out[0] = uint32_t(acc);
}

// ------------------------------------------------------------ sorted GEMV
// Layout: tile = 32 output rows x G groups (G > in/8, so one group tile),
// idx = row(5) << 11 | group(11); group G-1 is the zero "dummy" slot. A wave
// owns one tile's stream; a block has W waves (W consecutive tiles) and
// handles segment range `split` of each.
// KIND 0: fp32 lane-private accumulators ys[row][lane] (conflict-free RMW).
// KIND 1: exact fixed-point int64 accumulators via ds_add_u64 (2^-18 units),
//         C copies per row selected by lane % C to spread same-address atomics.
// x: fp8 [M][in]; partial: [splits][M][out] fp32.
__device__ __forceinline__ uint64_t exact_fixed18(float v) {
    // v is a sum of products of e4m3 values: an integer multiple of 2^-18 with
    // |v| < 2^21, so n = v*2^18 is an integer-valued float below 2^39. Split
    // n = a*2^16 + b with 0 <= b < 2^16: a = floor(v*4) (|a| < 2^23, exact),
    // b = n - a*2^16 (< 2^16, exact in fp32). No rounding anywhere.
    const float a_f = floorf(v * 4.f);
    const float b_f = fmaf(a_f, -65536.f, v * 262144.f);
    const int32_t a = int32_t(a_f);
    const uint32_t b = uint32_t(b_f);
    const uint32_t lo = (uint32_t(a) << 16) | b, hi = uint32_t(a >> 16);
    return (uint64_t(hi) << 32) | lo;
}
template <int G, int DB, int M, int W, int UNROLL, int KIND, int C = 1>
__global__ void __launch_bounds__(W * 32)
sorted_gemv(const uint8_t *__restrict__ x, const uint16_t *__restrict__ idx,
            const uint32_t *__restrict__ planes, const uint16_t *__restrict__ seg_codes,
            const SortedTileDesc *__restrict__ tiles, const uint8_t *__restrict__ book,
            float *__restrict__ partial, int in, int out, int tile_count, int splits,
            int chunks_per_segment) {
    constexpr int GB = 11, THREADS = W * 32;
    static_assert(G <= 2048 && (G & (G - 1)) == 0 && (C & (C - 1)) == 0 && C <= 32);
    __shared__ uint64_t xs[G * M];
    __shared__ float ysf[KIND == 0 ? W * 32 * M * 32 : 1];
    __shared__ uint64_t ysi[KIND == 1 ? W * 32 * M * C : 1];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int tg = blockIdx.x / splits, split = blockIdx.x % splits, groups = in / 8;
    const int tile = tg * W + wave;
    for (int i = tid; i < G * M; i += THREADS) {
        const int g = i / M, m = i % M;
        xs[i] = g < groups ? reinterpret_cast<const uint64_t *>(x)[size_t(m) * groups + g] : 0;
    }
    if constexpr (KIND == 0) {
        for (int i = tid; i < W * 32 * M * 32; i += THREADS)
            ysf[i] = 0.f;
    } else {
        for (int i = tid; i < W * 32 * M * C; i += THREADS)
            ysi[i] = 0;
    }
    __syncthreads();
    if (tile < tile_count) {
        const SortedTileDesc t = tiles[tile];
        const uint32_t le = (2u << lane) - 1; // lanes <= this one
        const uint32_t s0 = (t.segment_count * uint32_t(split)) / uint32_t(splits),
                       s1 = (t.segment_count * uint32_t(split + 1)) / uint32_t(splits);
        float *yw = ysf + wave * (32 * M * 32);
        uint64_t *yi = ysi + wave * (32 * M * C) + (lane & (C - 1)) * (32 * M);
        for (uint32_t seg = s0; seg < s1; ++seg) {
            uint32_t running = seg_codes[t.segment_begin + seg];
            const uint32_t c0 = __builtin_amdgcn_readfirstlane(
                                   t.chunk_begin + seg * uint32_t(chunks_per_segment)),
                           c1 = __builtin_amdgcn_readfirstlane(min(
                               t.chunk_begin + t.chunk_count, c0 + uint32_t(chunks_per_segment)));
            uint32_t c = c0;
            auto step = [&](uint32_t cc, int n) {
                uint32_t e[UNROLL], off[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    if (u >= n)
                        break;
                    e[u] = idx[(cc + u) * 32 + lane];
                    uint32_t incl = 0, tot = 0;
#pragma unroll
                    for (int k = 0; k < DB; ++k) {
                        const uint32_t p = planes[(cc + u) * DB + k];
                        incl += __builtin_popcount(p & le) << k;
                        tot += __builtin_popcount(p) << k;
                    }
                    off[u] = (running + incl) << 3; // byte offset into the codebook
                    running += tot;
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
                    const int row = int(e[u] >> GB), grp = int(e[u] & (G - 1));
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        const uint64_t xv = xs[grp * M + m];
                        float v = __builtin_amdgcn_dot4_f32_fp8_fp8(w[u].x, uint32_t(xv), 0.f);
                        v = __builtin_amdgcn_dot4_f32_fp8_fp8(w[u].y, uint32_t(xv >> 32), v);
                        if constexpr (KIND == 0) {
                            yw[(row * M + m) * 32 + lane] += v;
                        } else {
                            __hip_atomic_fetch_add(&yi[row * M + m], exact_fixed18(v),
                                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
                        }
                    }
                }
            };
            for (; c + UNROLL <= c1; c += UNROLL)
                step(c, UNROLL);
            for (; c < c1; ++c)
                step(c, 1);
        }
        // Reduce / convert and store this wave's 32 rows (lane = row).
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float sum = 0.f;
            if constexpr (KIND == 0) {
                for (int j = 0; j < 32; ++j)
                    sum += yw[(lane * M + m) * 32 + ((j + lane) & 31)];
            } else {
                int64_t acc = 0;
                const uint64_t *base = ysi + wave * (32 * M * C);
#pragma unroll
                for (int k = 0; k < C; ++k)
                    acc += int64_t(base[k * (32 * M) + lane * M + m]);
                sum = float(double(acc) * (1.0 / 262144.0));
            }
            const int r = tile * 32 + lane;
            if (r < out)
                partial[(size_t(split) * M + m) * size_t(out) + r] = sum;
        }
    }
}
__global__ void merge(const float *partial, float *y, int n, int parts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float s = 0;
    for (int p = 0; p < parts; ++p)
        s += partial[size_t(p) * n + i];
    y[i] = s;
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

int main(int argc, char **argv) {
    std::string what = argc > 1 ? argv[1] : "all";
    int dev = argc > 2 ? atoi(argv[2]) : 0;
    CK(hipSetDevice(dev));
    hipDeviceProp_t p{};
    CK(hipGetDeviceProperties(&p, dev));
    printf("# %s WGPs=%d clock=%d MHz\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    const double clk = p.clockRate * 1e3;
    Timer T;
    float *sink;
    CK(hipMalloc(&sink, 4096));

    if (what == "all" || what == "lds") {
        printf("# LDS random throughput, 2048 blocks x 256 threads, 4096 ops/thread\n");
        const int iters = 4096, blocks = 2048, threads = 256;
        const double ops = double(blocks) * threads * iters;
        for (int slots : {32, 256, 1024, 4096}) {
            double ms = T.ms([&] {
                lds_random<0><<<blocks, threads>>>(1, slots - 1, iters, sink);
            }, 5);
            printf("ds_add_f32 random slots=%5d : %7.3f ms  %6.1f Gop/s  ops/clk/CU=%.2f\n", slots,
                   ms, ops / ms / 1e6, ops / (ms * 1e-3) / 64 / clk);
        }
        {
            double ms = T.ms([&] { lds_random<1><<<blocks, threads>>>(1, 2047, iters, sink); }, 5);
            printf("ds_read_b64 random 2048 entries : %7.3f ms  %6.1f Gop/s  ops/clk/CU=%.2f\n",
                   ms, ops / ms / 1e6, ops / (ms * 1e-3) / 64 / clk);
            ms = T.ms([&] { lds_random<2><<<blocks, threads>>>(1, 255, iters, sink); }, 5);
            printf("read_b64 + add_f32 (256 slots)  : %7.3f ms  %6.1f Gpair/s  pairs/clk/CU=%.2f\n",
                   ms, ops / ms / 1e6, ops / (ms * 1e-3) / 64 / clk);
            ms = T.ms([&] { lds_random<3><<<blocks, threads>>>(1, 255, iters, sink); }, 5);
            printf("ds_add_u32 random 256 slots     : %7.3f ms  %6.1f Gop/s  ops/clk/CU=%.2f\n",
                   ms, ops / ms / 1e6, ops / (ms * 1e-3) / 64 / clk);
            ms = T.ms([&] { lds_random<6><<<blocks, threads>>>(1, 255, iters, sink); }, 5);
            printf("ds_add_u64 random 32 slots      : %7.3f ms  %6.1f Gop/s  ops/clk/CU=%.2f\n",
                   ms, ops / ms / 1e6, ops / (ms * 1e-3) / 64 / clk);
            ms = T.ms([&] { lds_random<4><<<blocks, threads>>>(1, 4095, iters, sink); }, 5);
            printf("lane-private RMW [row][lane]    : %7.3f ms  %6.1f Gop/s  ops/clk/CU=%.2f\n",
                   ms, ops / ms / 1e6, ops / (ms * 1e-3) / 64 / clk);
            ms = T.ms([&] { lds_random<5><<<blocks, threads>>>(1, 4095, iters, sink); }, 5);
            printf("read_b64 + lane-private RMW     : %7.3f ms  %6.1f Gpair/s  pairs/clk/CU=%.2f\n",
                   ms, ops / ms / 1e6, ops / (ms * 1e-3) / 64 / clk);
        }
        {
            int v = 0;
            CK(hipDeviceGetAttribute(&v, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev));
            printf("LDS per multiprocessor (WGP): %d bytes, per block %zu\n", v, p.sharedMemPerBlock);
        }
    }

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> hbook(65536 * 8);
    for (auto &v : hbook)
        v = f_to_fp8(nd(rng) * 0.5f);
    uint64_t *book = reinterpret_cast<uint64_t *>(upload(hbook));

    if (what == "all" || what == "gather") {
        // Synthetic gate_proj-shaped code tensor, random codes.
        const int in = 12288, out = 14336, groups = in / 8;
        std::vector<uint16_t> codes(size_t(out) * groups);
        for (auto &c : codes)
            c = uint16_t(rng());
        printf("# gather-only on the sorted stream (gate shape, %zu entries)\n", codes.size());
        for (int DB : {2, 3}) {
            auto L = behemoth::sorted_layout(in, out, codes.data(), 32, 2048, DB, 64);
            std::vector<uint16_t> back;
            bool ok = behemoth::sorted_layout_decode(L, in, out, back) && back == codes;
            printf("layout R=32 G=2048 DB=%d: chunks=%zu dummies=%.2f%% bytes/entry=%.3f decode=%s\n",
                   DB, L.chunks(), 100. * L.dummies / L.entries, double(L.bytes()) / L.entries,
                   ok ? "ok" : "FAIL");
            auto *didx = upload(L.idx);
            auto *dpl = upload(L.planes);
            auto *dseg = upload(L.segment_codes);
            // Only whole segments: drop tile tails for the microbenchmark.
            uint32_t segments = uint32_t(L.chunks() / 64);
            double n = double(segments) * 64 * 32;
            auto run = [&](auto kern, const char *name, int blocks) {
                double ms = T.ms([&] {
                    kern<<<blocks, 256>>>(didx, dpl, dseg, 64, segments, book, (uint32_t *)sink);
                });
                printf("  %-7s DB=%d blocks=%5d : %7.3f ms  %7.2f Ggather/s  2B-code-equiv %6.1f "
                       "GB/s  gathers/clk/CU=%.2f\n",
                       name, DB, blocks, ms, n / ms / 1e6, n * 2 / ms / 1e6,
                       n / (ms * 1e-3) / 64 / clk);
            };
            if (DB == 2) {
                run(sorted_gather<2, false>, "sorted", 1024);
                run(sorted_gather<2, false>, "sorted", 4096);
                run(sorted_gather<2, true>, "random", 1024);
                run(sorted_gather<2, true>, "random", 4096);
            } else {
                run(sorted_gather<3, false>, "sorted", 1024);
                run(sorted_gather<3, false>, "sorted", 4096);
            }
            CK(hipFree(didx));
            CK(hipFree(dpl));
            CK(hipFree(dseg));
        }
    }

    if (what == "all" || what == "gemv") {
        for (const auto &s : shapes) {
            const int groups = s.in / 8;
            std::vector<uint16_t> codes(size_t(s.out) * groups);
            for (auto &c : codes)
                c = uint16_t(rng());
            constexpr int MMAX = 8;
            std::vector<uint8_t> hx(size_t(MMAX) * s.in);
            for (auto &v : hx)
                v = f_to_fp8(nd(rng) * 0.3f);
            uint8_t *dx = upload(hx);
            // reference for M rows
            std::vector<float> xf(hx.size()), bookf(hbook.size());
            for (size_t i = 0; i < xf.size(); ++i)
                xf[i] = fp8_to_f(hx[i]);
            for (size_t i = 0; i < bookf.size(); ++i)
                bookf[i] = fp8_to_f(hbook[i]);
            std::vector<float> ref(size_t(MMAX) * s.out);
            for (int m = 0; m < MMAX; ++m)
                for (int o = 0; o < s.out; ++o) {
                    double acc = 0;
                    for (int g = 0; g < groups; ++g) {
                        const float *w = &bookf[size_t(codes[size_t(o) * groups + g]) * 8];
                        const float *xx = &xf[size_t(m) * s.in + size_t(g) * 8];
                        for (int j = 0; j < 8; ++j)
                            acc += double(w[j]) * xx[j];
                    }
                    ref[size_t(m) * s.out + o] = float(acc);
                }
            printf("# %s entries=%zu (2B codes = %.1f MB)\n", s.name, codes.size(),
                   codes.size() * 2 / 1e6);
            float *partial, *y;
            CK(hipMalloc(&partial, size_t(64) * MMAX * s.out * 4));
            CK(hipMalloc(&y, size_t(MMAX) * s.out * 4));
            auto run = [&]<int G, int DB, int M, int W, int UNROLL, int KIND, int C>(int splits) {
                auto L = behemoth::sorted_layout(s.in, s.out, codes.data(), 32, G, DB, 64);
                if (L.group_tiles != 1)
                    throw std::runtime_error("G must cover all groups");
                auto *didx = upload(L.idx);
                auto *dpl = upload(L.planes);
                auto *dseg = upload(L.segment_codes);
                auto *dt = upload(L.tiles);
                const int tiles = int(L.tiles.size());
                const int blocks = (tiles + W - 1) / W * splits;
                auto launch = [&] {
                    sorted_gemv<G, DB, M, W, UNROLL, KIND, C><<<blocks, W * 32>>>(
                        dx, didx, dpl, dseg, dt, reinterpret_cast<const uint8_t *>(book), partial,
                        s.in, s.out, tiles, splits, 64);
                };
                double ms = T.ms(launch);
                merge<<<(M * s.out + 255) / 256, 256>>>(partial, y, M * s.out, splits);
                std::vector<float> hy(size_t(M) * s.out);
                CK(hipMemcpy(hy.data(), y, hy.size() * 4, hipMemcpyDeviceToHost));
                double maxerr = 0, maxref = 0;
                for (size_t i = 0; i < hy.size(); ++i) {
                    maxerr = std::max(maxerr, double(std::fabs(hy[i] - ref[i])));
                    maxref = std::max(maxref, double(std::fabs(ref[i])));
                }
                const double bytes = double(L.bytes());
                printf("  %s C=%d G=%4d DB=%d M=%d W=%d U=%d splits=%2d blocks=%5d dummies=%.2f%%: %7.4f ms  "
                       "logical(2B) %6.1f GB/s  actual %6.1f GB/s  %.3f B/entry  %.3f ms/row "
                       "maxerr=%.3g/%.3g\n",
                       KIND == 0 ? "fp32-lane" : "int64-atm", C, G, DB, M, W, UNROLL, splits, blocks,
                       100. * L.dummies / L.entries, ms, codes.size() * 2 / ms / 1e6,
                       bytes / ms / 1e6, bytes / codes.size(), ms / M, maxerr, maxref);
                CK(hipFree(didx));
                CK(hipFree(dpl));
                CK(hipFree(dseg));
                CK(hipFree(dt));
            };
            run.template operator()<2048, 3, 1, 8, 4, 0, 1>(8);
            run.template operator()<2048, 3, 1, 8, 4, 1, 1>(8);
            run.template operator()<2048, 3, 1, 8, 4, 1, 2>(8);
            run.template operator()<2048, 3, 1, 8, 4, 1, 4>(8);
            run.template operator()<2048, 3, 1, 8, 2, 1, 2>(8);
            run.template operator()<2048, 3, 1, 8, 8, 1, 2>(8);
            run.template operator()<2048, 3, 1, 8, 4, 1, 2>(4);
            run.template operator()<2048, 3, 1, 8, 4, 1, 2>(16);
            run.template operator()<2048, 3, 1, 4, 4, 1, 2>(16);
            run.template operator()<2048, 2, 1, 8, 4, 1, 2>(8);
            run.template operator()<2048, 4, 1, 8, 4, 1, 2>(8);
            run.template operator()<2048, 3, 2, 8, 4, 1, 2>(8);
            CK(hipFree(dx));
            CK(hipFree(partial));
            CK(hipFree(y));
        }
    }
    return 0;
}
