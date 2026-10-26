// Microbenchmarks for AQLM 1x16 decode on gfx1201.
// Build: hipcc --offload-arch=gfx1201 -O3 -mno-wavefrontsize64 micro.hip -o micro
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

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e = (x);                                                                        \
        if (e != hipSuccess) {                                                                     \
            fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, hipGetErrorString(e), __FILE__,         \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

typedef float f2v __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------- stream read
__global__ void stream_read(const uint4 *__restrict__ p, size_t n, uint32_t *out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(gridDim.x) * blockDim.x;
    uint32_t acc = 0;
    for (; i + 3 * stride < n; i += 4 * stride) {
        uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.w ^ b.x ^ b.w ^ c.x ^ c.w ^ d.x ^ d.w;
    }
    for (; i < n; i += stride) {
        uint4 a = p[i];
        acc ^= a.x ^ a.w;
    }
    if (acc == 0xdeadbeef)
        out[0] = acc;
}

// ------------------------------------------------- global random gather (8B)
// Each thread: U independent (code load -> table gather) chains per iteration.
template <int U>
__global__ void gather_global(const uint16_t *__restrict__ codes, size_t n,
                              const uint64_t *__restrict__ table, uint32_t mask, uint32_t *out) {
    size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t acc = 0;
    for (; i + (U - 1) * stride < n; i += U * stride) {
        uint32_t c[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
            c[u] = codes[i + u * stride] & mask;
        uint64_t v[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
            v[u] = table[c[u]];
#pragma unroll
        for (int u = 0; u < U; ++u)
            acc ^= v[u];
    }
    if (acc == 0xdeadbeef)
        out[0] = uint32_t(acc);
}

// Same but the table is staged into LDS (table bytes <= 64 KiB).
template <int U>
__global__ void gather_lds(const uint16_t *__restrict__ codes, size_t n,
                           const uint64_t *__restrict__ table, uint32_t mask, uint32_t *out) {
    extern __shared__ uint64_t lds[];
    const uint32_t entries = mask + 1;
    for (uint32_t j = threadIdx.x; j < entries; j += blockDim.x)
        lds[j] = table[j];
    __syncthreads();
    size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t acc = 0;
    for (; i + (U - 1) * stride < n; i += U * stride) {
        uint32_t c[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
            c[u] = codes[i + u * stride] & mask;
        uint64_t v[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
            v[u] = lds[c[u]];
#pragma unroll
        for (int u = 0; u < U; ++u)
            acc ^= v[u];
    }
    if (acc == 0xdeadbeef)
        out[0] = uint32_t(acc);
}

// ------------------------------------------------------------- GEMV 1x16 rows=1
// codes: group-major [groups][out] uint16; book: uint64[65536] (8 fp8 each);
// x: fp8 [in]; partial: [parts][out] fp32.  Thread owns CPT adjacent columns,
// processes U groups per iteration (U*CPT independent gathers in flight).
template <int U, int CPT, int THREADS>
__global__ void __launch_bounds__(THREADS)
gemv_1x16(const uint16_t *__restrict__ codes, const uint64_t *__restrict__ book,
          const uint8_t *__restrict__ x, float *__restrict__ partial, int in, int out,
          int part_groups) {
    static_assert(CPT == 1 || CPT == 2 || CPT == 4);
    const int col = (blockIdx.x * THREADS + threadIdx.x) * CPT;
    const int g0 = blockIdx.y * part_groups, groups = in / 8;
    const int g1 = min(groups, g0 + part_groups);
    float acc[CPT];
#pragma unroll
    for (int c = 0; c < CPT; ++c)
        acc[c] = 0.f;
    const uint2 *xg = reinterpret_cast<const uint2 *>(x);
    int g = g0;
    for (; g + U <= g1; g += U) {
        uint32_t code[U][CPT];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint16_t *p = codes + size_t(g + u) * out + col;
            if constexpr (CPT == 1) {
                code[u][0] = *p;
            } else if constexpr (CPT == 2) {
                uint32_t v = *reinterpret_cast<const uint32_t *>(p);
                code[u][0] = v & 0xffff;
                code[u][1] = v >> 16;
            } else {
                uint2 v = *reinterpret_cast<const uint2 *>(p);
                code[u][0] = v.x & 0xffff;
                code[u][1] = v.x >> 16;
                code[u][2] = v.y & 0xffff;
                code[u][3] = v.y >> 16;
            }
        }
        uint64_t w[U][CPT];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < CPT; ++c)
                w[u][c] = book[code[u][c]];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint2 xv = xg[g + u]; // wave-uniform -> scalar load
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w[u][c]), xv.x, acc[c]);
                acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w[u][c] >> 32), xv.y, acc[c]);
            }
        }
    }
    for (; g < g1; ++g) {
        const uint2 xv = xg[g];
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            uint64_t wv = book[codes[size_t(g) * out + col + c]];
            acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(wv), xv.x, acc[c]);
            acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(wv >> 32), xv.y, acc[c]);
        }
    }
    float *dst = partial + size_t(blockIdx.y) * out + col;
    if constexpr (CPT == 1)
        dst[0] = acc[0];
    else if constexpr (CPT == 2)
        *reinterpret_cast<float2 *>(dst) = make_float2(acc[0], acc[1]);
    else
        *reinterpret_cast<float4 *>(dst) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

// Hypothetical 2x8 additive format (same 16 bits / 8 weights): two 256-entry
// books of 8 fp8 each (4 KiB total) in LDS; w = b0[hi] + b1[lo] is folded into
// two dot4 pairs per group (no explicit add).  Measures what a small-codebook
// format would decode at on this GPU.
template <int U, int CPT, int THREADS, bool LDS>
__global__ void __launch_bounds__(THREADS)
gemv_2x8(const uint16_t *__restrict__ codes, const uint64_t *__restrict__ book,
         const uint8_t *__restrict__ x, float *__restrict__ partial, int in, int out,
         int part_groups) {
    __shared__ uint64_t lbook[512];
    const uint64_t *b = book;
    if constexpr (LDS) {
        for (int j = threadIdx.x; j < 512; j += THREADS)
            lbook[j] = book[j];
        __syncthreads();
        b = lbook;
    }
    const int col = (blockIdx.x * THREADS + threadIdx.x) * CPT;
    const int g0 = blockIdx.y * part_groups, groups = in / 8;
    const int g1 = min(groups, g0 + part_groups);
    float acc[CPT];
#pragma unroll
    for (int c = 0; c < CPT; ++c)
        acc[c] = 0.f;
    const uint2 *xg = reinterpret_cast<const uint2 *>(x);
    int g = g0;
    for (; g + U <= g1; g += U) {
        uint32_t code[U][CPT];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint16_t *p = codes + size_t(g + u) * out + col;
            if constexpr (CPT == 1) {
                code[u][0] = *p;
            } else if constexpr (CPT == 2) {
                uint32_t v = *reinterpret_cast<const uint32_t *>(p);
                code[u][0] = v & 0xffff;
                code[u][1] = v >> 16;
            } else {
                uint2 v = *reinterpret_cast<const uint2 *>(p);
                code[u][0] = v.x & 0xffff;
                code[u][1] = v.x >> 16;
                code[u][2] = v.y & 0xffff;
                code[u][3] = v.y >> 16;
            }
        }
        uint64_t w0[U][CPT], w1[U][CPT];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                w0[u][c] = b[code[u][c] >> 8];
                w1[u][c] = b[256 + (code[u][c] & 255)];
            }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint2 xv = xg[g + u];
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w0[u][c]), xv.x, acc[c]);
                acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w0[u][c] >> 32), xv.y, acc[c]);
                acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w1[u][c]), xv.x, acc[c]);
                acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w1[u][c] >> 32), xv.y, acc[c]);
            }
        }
    }
    for (; g < g1; ++g) {
        const uint2 xv = xg[g];
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            uint32_t cd = codes[size_t(g) * out + col + c];
            uint64_t w0 = b[cd >> 8], w1 = b[256 + (cd & 255)];
            acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w0), xv.x, acc[c]);
            acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w0 >> 32), xv.y, acc[c]);
            acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w1), xv.x, acc[c]);
            acc[c] = __builtin_amdgcn_dot4_f32_fp8_fp8(uint32_t(w1 >> 32), xv.y, acc[c]);
        }
    }
    float *dst = partial + size_t(blockIdx.y) * out + col;
    for (int c = 0; c < CPT; ++c)
        dst[c] = acc[c];
}

__global__ void empty_kernel(float *p) {
    if (threadIdx.x == 1023)
        p[0] = 1.f;
}
__global__ void merge(const float *partial, float *y, int out, int parts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out)
        return;
    float s = 0;
    for (int p = 0; p < parts; ++p)
        s += partial[size_t(p) * out + i];
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
static uint8_t f_to_fp8(float f) { // crude nearest for finite small inputs
    if (f == 0)
        return 0;
    uint8_t s = f < 0 ? 0x80 : 0;
    f = std::fabs(f);
    int e;
    float m = std::frexp(f, &e); // f = m * 2^e, m in [0.5,1)
    e -= 1;
    m *= 2; // m in [1,2)
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

int main(int argc, char **argv) {
    std::string what = argc > 1 ? argv[1] : "all";
    int dev = argc > 2 ? atoi(argv[2]) : 0;
    CK(hipSetDevice(dev));
    hipDeviceProp_t p{};
    CK(hipGetDeviceProperties(&p, dev));
    printf("# %s WGPs=%d clock=%d MHz\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    Timer T;
    uint32_t *sink;
    CK(hipMalloc(&sink, 4096));

    if (what == "all" || what == "stream") {
        size_t bytes = size_t(1) << 30; // 1 GiB
        uint4 *buf;
        CK(hipMalloc(&buf, bytes));
        CK(hipMemset(buf, 1, bytes));
        for (int blocks : {256, 512, 1024, 2048, 4096})
            for (int threads : {256, 512}) {
                double ms = T.ms([&] { stream_read<<<blocks, threads>>>(buf, bytes / 16, sink); });
                printf("stream_read blocks=%d threads=%d : %.3f ms  %.1f GB/s\n", blocks, threads, ms,
                       bytes / ms / 1e6);
            }
        CK(hipFree(buf));
    }

    if (what == "all" || what == "gather") {
        size_t n = size_t(64) << 20; // 64M codes = 128 MiB of codes
        std::vector<uint16_t> h(n);
        std::mt19937 rng(1);
        for (auto &v : h)
            v = uint16_t(rng());
        uint16_t *codes;
        CK(hipMalloc(&codes, n * 2));
        CK(hipMemcpy(codes, h.data(), n * 2, hipMemcpyHostToDevice));
        uint64_t *table;
        CK(hipMalloc(&table, 8 << 20));
        CK(hipMemset(table, 3, 8 << 20));
        printf("# gather: code stream 2B/code + 8B random gather from table of given size\n");
        for (int log2e : {9, 10, 11, 12, 13, 14, 15, 16}) { // 4KB .. 512KB
            uint32_t mask = (1u << log2e) - 1;
            size_t tbytes = size_t(mask + 1) * 8;
            auto run = [&](auto kern, const char *name, int U, int blocks, int threads, size_t lds) {
                double ms = T.ms([&] { kern<<<blocks, threads, lds>>>(codes, n, table, mask, sink); });
                printf("%-13s table=%6zu KB U=%2d blocks=%5d thr=%3d : %7.3f ms  %7.2f Ggather/s  "
                       "codes %6.1f GB/s  gathers/clk/CU=%.2f\n",
                       name, tbytes / 1024, U, blocks, threads, ms, n / ms / 1e6, n * 2 / ms / 1e6,
                       n / (ms * 1e-3) / 64 / (p.clockRate * 1e3));
            };
            run(gather_global<4>, "global", 4, 2048, 256, 0);
            run(gather_global<8>, "global", 8, 2048, 256, 0);
            run(gather_global<16>, "global", 16, 1024, 256, 0);
            run(gather_global<16>, "global", 16, 4096, 256, 0);
            if (tbytes <= 65536) {
                run(gather_lds<8>, "lds", 8, 2048, 256, tbytes);
                run(gather_lds<16>, "lds", 16, 2048, 256, tbytes);
            }
        }
        CK(hipFree(codes));
        CK(hipFree(table));
    }

    if (what == "launch") {
        float *dummy;
        CK(hipMalloc(&dummy, 4096));
        hipStream_t s;
        CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        for (int blocks : {1, 48, 1152}) {
            double ms = T.ms([&] {
                for (int i = 0; i < 1000; ++i)
                    empty_kernel<<<blocks, 256, 0, s>>>(dummy);
                CK(hipStreamSynchronize(s));
            }, 5);
            printf("1000 back-to-back empty kernels (blocks=%d, 256 thr): %.3f ms -> %.2f us/launch\n", blocks, ms, ms);
        }
        // hipGraph with 1000 nodes
        hipGraph_t g;
        hipGraphExec_t ge;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        for (int i = 0; i < 1000; ++i)
            empty_kernel<<<48, 256, 0, s>>>(dummy);
        CK(hipStreamEndCapture(s, &g));
        CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        double ms = T.ms([&] {
            CK(hipGraphLaunch(ge, s));
            CK(hipStreamSynchronize(s));
        }, 5);
        printf("hipGraph of 1000 empty kernels: %.3f ms -> %.2f us/kernel\n", ms, ms);
        // realistic small elementwise kernel chain: 12288-element bf16 ops
        return 0;
    }
    if (what == "gatherblocks") {
        size_t n = size_t(16) << 20;
        std::vector<uint16_t> h(n);
        std::mt19937 rng(1);
        for (auto &v : h)
            v = uint16_t(rng());
        uint16_t *codes;
        CK(hipMalloc(&codes, n * 2));
        CK(hipMemcpy(codes, h.data(), n * 2, hipMemcpyHostToDevice));
        uint64_t *table;
        CK(hipMalloc(&table, 1 << 20));
        CK(hipMemset(table, 3, 1 << 20));
        uint32_t mask = 65535;
        printf("# 512 KB table, 8B gathers, varying active WGPs (blocks of 256 threads = 8 waves)\n");
        for (int blocks : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
            double ms = T.ms([&] { gather_global<16><<<blocks, 256>>>(codes, n, table, mask, sink); }, 5);
            double gps = n / ms / 1e6;
            printf("blocks=%4d : %8.3f ms  %7.2f Ggather/s  per-block gathers/clk=%.3f  GPU gathers/clk=%.1f\n",
                   blocks, ms, gps, gps * 1e9 / blocks / (p.clockRate * 1e3), gps * 1e9 / (p.clockRate * 1e3));
        }
        for (int blocks : {64, 128}) {
            double ms = T.ms([&] { gather_global<16><<<blocks, 1024>>>(codes, n, table, mask, sink); }, 5);
            double gps = n / ms / 1e6;
            printf("blocks=%4d x1024thr: %8.3f ms  %7.2f Ggather/s  GPU gathers/clk=%.1f\n", blocks, ms, gps, gps * 1e9 / (p.clockRate * 1e3));
        }
        return 0;
    }
    if (what == "all" || what == "gemv" || what == "gemv2x8") {
        struct Shape {
            const char *name;
            int in, out;
        } shapes[] = {{"q 12288->6144", 12288, 6144},   {"k/v 12288->512", 12288, 512},
                      {"o 6144->12288", 6144, 12288},   {"gate/up 12288->14336", 12288, 14336},
                      {"down 14336->12288", 14336, 12288}};
        // Codebook: random-ish fp8 values of modest magnitude.
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<uint8_t> hbook(65536 * 8);
        for (auto &v : hbook)
            v = f_to_fp8(nd(rng) * 0.5f);
        uint64_t *book;
        CK(hipMalloc(&book, 65536 * 8));
        CK(hipMemcpy(book, hbook.data(), 65536 * 8, hipMemcpyHostToDevice));
        for (auto &s : shapes) {
            size_t ncodes = size_t(s.in / 8) * s.out;
            std::vector<uint16_t> hcodes(ncodes);
            for (auto &v : hcodes)
                v = uint16_t(rng());
            std::vector<uint8_t> hx(s.in);
            for (auto &v : hx)
                v = f_to_fp8(nd(rng) * 0.3f);
            uint16_t *codes;
            uint8_t *x;
            float *partial, *y;
            CK(hipMalloc(&codes, ncodes * 2));
            CK(hipMalloc(&x, s.in));
            CK(hipMalloc(&partial, size_t(64) * s.out * 4));
            CK(hipMalloc(&y, s.out * 4));
            CK(hipMemcpy(codes, hcodes.data(), ncodes * 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(x, hx.data(), s.in, hipMemcpyHostToDevice));
            // reference (1x16)
            std::vector<float> ref(s.out, 0.f), xf(s.in);
            for (int i = 0; i < s.in; ++i)
                xf[i] = fp8_to_f(hx[i]);
            std::vector<float> bookf(65536 * 8);
            for (size_t i = 0; i < bookf.size(); ++i)
                bookf[i] = fp8_to_f(hbook[i]);
            if (what != "gemv2x8")
                for (int o = 0; o < s.out; ++o) {
                    double acc = 0;
                    for (int g = 0; g < s.in / 8; ++g) {
                        const float *w = &bookf[size_t(hcodes[size_t(g) * s.out + o]) * 8];
                        for (int j = 0; j < 8; ++j)
                            acc += double(w[j]) * xf[g * 8 + j];
                    }
                    ref[o] = float(acc);
                }
            printf("# %s codes=%.1f MB\n", s.name, ncodes * 2 / 1e6);
            auto report = [&](const char *name, int U, int CPT, int threads, int part_groups,
                              double ms, bool check) {
                int parts = (s.in / 8 + part_groups - 1) / part_groups;
                std::string status;
                if (check) {
                    merge<<<(s.out + 255) / 256, 256>>>(partial, y, s.out, parts);
                    std::vector<float> hy(s.out);
                    CK(hipMemcpy(hy.data(), y, s.out * 4, hipMemcpyDeviceToHost));
                    double maxerr = 0, maxref = 0;
                    for (int o = 0; o < s.out; ++o) {
                        maxerr = std::max(maxerr, double(std::fabs(hy[o] - ref[o])));
                        maxref = std::max(maxref, double(std::fabs(ref[o])));
                    }
                    char b[64];
                    snprintf(b, sizeof b, " maxerr=%.3g (maxref=%.3g)", maxerr, maxref);
                    status = b;
                }
                int blocks = (s.out / CPT + threads - 1) / threads * parts;
                printf("  %-8s U=%2d CPT=%d thr=%3d partK=%5d blocks=%5d waves=%5d : %7.4f ms  "
                       "%6.1f GB/s%s\n",
                       name, U, CPT, threads, part_groups * 8, blocks, blocks * threads / 32, ms,
                       ncodes * 2 / ms / 1e6, status.c_str());
            };
            auto run1 = [&](auto kern, const char *name, int U, int CPT, int threads, int partK) {
                int part_groups = partK / 8;
                int parts = (s.in / 8 + part_groups - 1) / part_groups;
                dim3 grid((s.out / CPT + threads - 1) / threads, parts);
                double ms = T.ms([&] {
                    kern<<<grid, threads>>>(codes, book, x, partial, s.in, s.out, part_groups);
                });
                report(name, U, CPT, threads, part_groups, ms, what != "gemv2x8");
            };
            if (what != "gemv2x8") {
                run1(gemv_1x16<4, 1, 256>, "1x16", 4, 1, 256, 512);
                run1(gemv_1x16<8, 1, 256>, "1x16", 8, 1, 256, 512);
                run1(gemv_1x16<8, 2, 256>, "1x16", 8, 2, 256, 512);
                run1(gemv_1x16<4, 4, 256>, "1x16", 4, 4, 256, 512);
                run1(gemv_1x16<8, 4, 256>, "1x16", 8, 4, 256, 512);
                run1(gemv_1x16<8, 2, 128>, "1x16", 8, 2, 128, 256);
                run1(gemv_1x16<8, 2, 128>, "1x16", 8, 2, 128, 512);
                run1(gemv_1x16<8, 2, 128>, "1x16", 8, 2, 128, 1024);
                run1(gemv_1x16<8, 4, 128>, "1x16", 8, 4, 128, 256);
                run1(gemv_1x16<8, 4, 128>, "1x16", 8, 4, 128, 512);
                run1(gemv_1x16<16, 2, 128>, "1x16", 16, 2, 128, 512);
                run1(gemv_1x16<16, 1, 256>, "1x16", 16, 1, 256, 512);
                run1(gemv_1x16<16, 1, 256>, "1x16", 16, 1, 256, 1024);
            }
            // 2x8 hypothetical (no reference; book reused as two 256-entry tables)
            auto run2 = [&](auto kern, const char *name, int U, int CPT, int threads, int partK) {
                int part_groups = partK / 8;
                int parts = (s.in / 8 + part_groups - 1) / part_groups;
                dim3 grid((s.out / CPT + threads - 1) / threads, parts);
                double ms = T.ms([&] {
                    kern<<<grid, threads>>>(codes, book, x, partial, s.in, s.out, part_groups);
                });
                report(name, U, CPT, threads, part_groups, ms, false);
            };
            run2(gemv_2x8<8, 2, 256, true>, "2x8-lds", 8, 2, 256, 512);
            run2(gemv_2x8<8, 4, 256, true>, "2x8-lds", 8, 4, 256, 512);
            run2(gemv_2x8<8, 2, 128, true>, "2x8-lds", 8, 2, 128, 512);
            run2(gemv_2x8<8, 4, 128, true>, "2x8-lds", 8, 4, 128, 512);
            run2(gemv_2x8<8, 2, 256, false>, "2x8-l0", 8, 2, 256, 512);
            run2(gemv_2x8<8, 4, 256, false>, "2x8-l0", 8, 4, 256, 512);
            run2(gemv_2x8<8, 4, 128, false>, "2x8-l0", 8, 4, 128, 512);
            CK(hipFree(codes));
            CK(hipFree(x));
            CK(hipFree(partial));
            CK(hipFree(y));
        }
        CK(hipFree(book));
    }
    return 0;
}
