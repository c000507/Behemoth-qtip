// Micro-benchmark of the W8A8 GEMMs (src/qwen_batch.hip) vs the fp8 GEMV, on TP2 shard shapes.
#include <chrono>
#include <cstdio>
#include <vector>

#include "behemoth/gpu.hpp"
#include "behemoth/qwen_batch.hpp"
#include "behemoth/qwen_kernels.hpp"

using namespace behemoth;
int main() {
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
