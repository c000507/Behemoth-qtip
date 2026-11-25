// Micro-benchmark of the W8A8 GEMMs (src/qwen_batch.hip) vs the fp8 GEMV, on TP2 shard shapes.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

#include "behemoth/gpu.hpp"
#include "behemoth/qwen_batch.hpp"
#include "behemoth/qwen_kernels.hpp"

using namespace behemoth;
// all-reduce microbenchmark between devices 0 and 1 (device-flag protocol, zero-copy vs DMA pull)
static void ar_bench() {
    const int H = 5120;
    struct R {
        hipStream_t s;
        DeviceBuffer<float> part, recv;
        DeviceBuffer<uint16_t> state;
        DeviceBuffer<unsigned> ctr;
        char *host = nullptr;
        unsigned *flags = nullptr;
    } r[2];
    const long slot = 64L * H * 4;
    for (int d = 0; d < 2; ++d) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipStreamCreateWithFlags(&r[d].s, hipStreamNonBlocking));
        r[d].part.resize(64 * H);
        r[d].recv.resize(64 * H);
        r[d].state.resize(64 * H);
        r[d].ctr.resize(4);
        HIP_CHECK(hipMemset(r[d].ctr.data(), 0, 16));
        HIP_CHECK(hipMemset(r[d].part.data(), 0, r[d].part.bytes()));
        HIP_CHECK(hipMemset(r[d].state.data(), 0, r[d].state.bytes()));
    }
    for (int d = 0; d < 2; ++d) {
        HIP_CHECK(hipSetDevice(std::getenv("AR_LOCAL") ? d : 1 - d));
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&r[d].host), size_t(slot) * 4,
                                hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&r[d].flags), 4 * 64, hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
        std::memset(r[d].flags, 0, 256);
    }
    for (int rows : {1, 8, 16, 32, 64})
        for (int bf = 0; bf < 2; ++bf)
            for (int dma = 0; dma < 2; ++dma) {
                const long n = long(rows) * H;
                const int it = 200;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i)
                    for (int d = 0; d < 2; ++d) {
                        HIP_CHECK(hipSetDevice(d));
                        R &a = r[d], &b = r[1 - d];
                        q_ar_push(a.part.data(), n, a.host, slot, bf, a.flags, a.ctr.data(), a.ctr.data() + 1, 4, a.s);
                        if (!dma) {
                            q_ar_pull_add(a.state.data(), a.part.data(), b.host, slot, bf, n, d, b.flags, a.ctr.data(),
                                          a.ctr.data() + 2, 4, a.s);
                        } else {
                            q_ar_wait(b.flags, a.ctr.data(), 4, a.s);
                            HIP_CHECK(hipMemcpyAsync(a.recv.data(), b.host + (i % 4) * slot, size_t(n) * (bf ? 2 : 4),
                                                     hipMemcpyHostToDevice, a.s));
                            q_ar_add_local(a.state.data(), a.part.data(), a.recv.data(), bf, n, d, a.ctr.data(),
                                           a.ctr.data() + 2, a.s);
                        }
                    }
                for (int d = 0; d < 2; ++d) {
                    HIP_CHECK(hipSetDevice(d));
                    HIP_CHECK(hipStreamSynchronize(r[d].s));
                }
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
                std::printf("ar rows %2d %s %s: %.1f us/exchange\n", rows, bf ? "bf16" : "fp32", dma ? "dma " : "zc  ", us);
            }
}

int main(int argc, char **argv) {
    if (argc > 1 && !std::strcmp(argv[1], "add")) {  // q_ar_add_local at prefill size
        const long n = (argc > 2 ? std::atol(argv[2]) : 2048L) * 5120;
        DeviceBuffer<uint16_t> st(static_cast<size_t>(n)), pe(static_cast<size_t>(n) * 2);
        DeviceBuffer<float> own(static_cast<size_t>(n));
        DeviceBuffer<unsigned> ctr(4);
        HIP_CHECK(hipMemset(st.data(), 0, st.bytes()));
        HIP_CHECK(hipMemset(pe.data(), 0, pe.bytes()));
        HIP_CHECK(hipMemset(own.data(), 0, own.bytes()));
        HIP_CHECK(hipMemset(ctr.data(), 0, 16));
        hipEvent_t a, b;
        HIP_CHECK(hipEventCreate(&a));
        HIP_CHECK(hipEventCreate(&b));
        for (int bf = 1; bf >= 0; --bf) {
            HIP_CHECK(hipEventRecord(a, nullptr));
            for (int i = 0; i < 20; ++i) q_ar_add_local(st.data(), own.data(), pe.data(), bf, n, 0, ctr.data(), ctr.data() + 2, nullptr);
            HIP_CHECK(hipEventRecord(b, nullptr));
            HIP_CHECK(hipEventSynchronize(b));
            float ms;
            HIP_CHECK(hipEventElapsedTime(&ms, a, b));
            std::printf("add_local bf16 %d: %.3f ms\n", bf, ms / 20);
        }
        return 0;
    }
    if (argc > 1 && !std::strcmp(argv[1], "ar")) {
        ar_bench();
        return 0;
    }
    struct Shape { int m, n; const char *name; };
    const Shape shapes[] = {{17408, 5120, "gateup"}, {5120, 8704, "down"}, {8192, 5120, "qkvz"}, {5120, 3072, "out"},
                            {7168, 5120, "qkv"}, {124160, 5120, "lm_head"}};
    hipStream_t s;
    HIP_CHECK(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    for (const Shape &sh : shapes) {
        const int maxr = 4096;
        DeviceBuffer<uint8_t> w(size_t(sh.m) * sh.n), xq(size_t(maxr) * sh.n);
        DeviceBuffer<float> ws(size_t(sh.m / 128) * (sh.n / 128)), xs(size_t(maxr) * (sh.n / 128)), y(size_t(maxr) * sh.m),
            part(std::max<size_t>(1, q_w8a8_small_scratch(sh.m, sh.n, 64)));
        DeviceBuffer<uint16_t> xb(size_t(maxr) * sh.n);
        HIP_CHECK(hipMemset(w.data(), 0x22, w.bytes()));
        HIP_CHECK(hipMemset(xq.data(), 0x22, xq.bytes()));
        HIP_CHECK(hipMemset(xb.data(), 0x22, xb.bytes()));
        std::vector<float> one(ws.size(), 1.f), onex(xs.size(), 1.f);
        ws.upload(one);
        xs.upload(onex);
        const double wbytes = double(sh.m) * sh.n;
        auto timeit = [&](auto fn, int it) {
            fn();
            HIP_CHECK(hipEventRecord(e0, s));
            for (int i = 0; i < it; ++i) fn();
            HIP_CHECK(hipEventRecord(e1, s));
            HIP_CHECK(hipEventSynchronize(e1));
            float ms;
            HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
            return double(ms) / it;
        };
        double g1 = timeit([&] { q_fp8_gemv(w.data(), ws.data(), sh.m, sh.n, xb.data(), sh.n, 1, y.data(), sh.m, s); }, 20);
        double g8 = timeit([&] { q_fp8_gemv(w.data(), ws.data(), sh.m, sh.n, xb.data(), sh.n, 8, y.data(), sh.m, s); }, 20);
        std::printf("%-8s %6dx%-5d gemv r1 %.3f ms (%.0f GB/s)  r8 %.3f ms | splits %d\n", sh.name, sh.m, sh.n, g1,
                    wbytes / g1 / 1e6, g8, q_w8a8_splits(sh.m, sh.n));
        for (int r : {1, 8, 16, 32, 48, 64}) {
            double t = timeit([&] { q_w8a8_small(w.data(), ws.data(), sh.m, sh.n, xq.data(), xs.data(), r, y.data(), sh.m, part.data(), s); }, 20);
            std::printf("   small rows %3d: %.3f ms (%.0f GB/s, %.1f TFLOPS)\n", r, t, wbytes / t / 1e6,
                        2.0 * wbytes * r / t / 1e9);
        }
        for (int r : {128, 512, 2048, 4096}) {
            double t = timeit([&] { q_w8a8_big(w.data(), ws.data(), sh.m, sh.n, xq.data(), xs.data(), r, y.data(), sh.m, s); }, 5);
            double tq = timeit([&] { q_act_quant(xb.data(), sh.n, r, sh.n, xq.data(), xs.data(), s); }, 5);
            std::printf("   big rows %4d: %.3f ms (%.1f TFLOPS)  act quant %.3f ms\n", r, t, 2.0 * wbytes * r / t / 1e9, tq);
        }
    }
    return 0;
}
