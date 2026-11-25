// GPU tests for src/qwen_batch.hip: activation fp8 quantisation and the W8A8 WMMA GEMMs against a
// double-precision CPU reference built from the same quantised operands; batch invariance of the
// decode GEMM (a row's bits do not depend on the other rows / the row count).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "behemoth/gpu.hpp"
#include "behemoth/qwen_batch.hpp"

using namespace behemoth;

static float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp(float(m) / 8.f, -6) : std::ldexp(1.f + float(m) / 8.f, e - 7);
    return s ? -v : v;
}
static uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
static float bf2f(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static int fails = 0;
#define CHECK(c, ...)                                                                                              \
    do {                                                                                                           \
        if (!(c)) {                                                                                                \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                       \
            std::printf(__VA_ARGS__);                                                                              \
            std::printf("\n");                                                                                     \
            ++fails;                                                                                               \
        }                                                                                                          \
    } while (0)

static void test_gemm(int m, int n, int rows, std::mt19937 &rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.5f, 2.f);
    std::vector<uint8_t> w(size_t(m) * n);
    for (auto &b : w) {
        b = uint8_t(byte(rng));
        if ((b & 0x7f) == 0x7f) b = 0x7e;
    }
    std::vector<float> ws(size_t(m / 128) * (n / 128));
    for (auto &s : ws) s = ud(rng) * 1e-3f;
    const int prow = std::max(rows, 64);
    std::vector<uint16_t> x(size_t(prow) * n);
    for (size_t i = 0; i < x.size(); ++i) x[i] = f2bf(nd(rng) * (1 + (i % 7)));
    DeviceBuffer<uint8_t> dw, dq;
    DeviceBuffer<float> dws, dxs, dy, dy2, dp;
    DeviceBuffer<uint16_t> dx;
    dw.upload(w);
    dws.upload(ws);
    dx.upload(x);
    dq.resize(size_t(prow) * n);
    dxs.resize(size_t(prow) * (n / 128));
    dy.resize(size_t(rows) * m);
    dy2.resize(size_t(rows) * m);
    dp.resize(std::max<size_t>(1, q_w8a8_small_scratch(m, n, 64)));
    q_act_quant(dx.data(), n, prow, n, dq.data(), dxs.data(), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto q = dq.download();
    auto xs = dxs.download();
    // quantisation error bound (e4m3: 3 mantissa bits)
    double qerr = 0;
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < n; ++k) {
            const float xv = bf2f(x[size_t(r) * n + k]), s = xs[size_t(r) * (n / 128) + k / 128];
            const float dv = e4m3(q[size_t(r) * n + k]) * s;
            qerr = std::max(qerr, double(std::fabs(dv - xv)) / (std::fabs(xv) / 16 + s / 64));
        }
    CHECK(qerr <= 1.0, "act quant error ratio %.3f", qerr);
    if (rows <= 64) q_w8a8_small(dw.data(), dws.data(), m, n, dq.data(), dxs.data(), rows, dy.data(), m, dp.data(), nullptr);
    else q_w8a8_big(dw.data(), dws.data(), m, n, dq.data(), dxs.data(), rows, dy.data(), m, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto y = dy.download();
    double maxrel = 0;
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < m; ++j) {
            double ref = 0, mag = 0;
            for (int k = 0; k < n; ++k) {
                const double a = double(e4m3(q[size_t(r) * n + k])) * xs[size_t(r) * (n / 128) + k / 128];
                const double b = double(e4m3(w[size_t(j) * n + k])) * ws[size_t(j / 128) * (n / 128) + k / 128];
                ref += a * b;
                mag += std::fabs(a * b);
            }
            maxrel = std::max(maxrel, std::fabs(ref - y[size_t(r) * m + j]) / (mag + 1e-30));
        }
    CHECK(maxrel < 1e-5, "gemm m %d n %d rows %d: max err / sum|ab| %.3g", m, n, rows, maxrel);
    std::printf("gemm m %d n %d rows %d: err %.3g, quant err ratio %.3f\n", m, n, rows, maxrel, qerr);
    if (rows > 1 && rows <= 64) {  // batch invariance: rows 0..r1 alone give the same bits
        const int r1 = std::max(1, rows / 3);
        q_w8a8_small(dw.data(), dws.data(), m, n, dq.data(), dxs.data(), r1, dy2.data(), m, dp.data(), nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        auto y2 = dy2.download();
        CHECK(std::memcmp(y2.data(), y.data(), size_t(r1) * m * 4) == 0, "batch invariance rows %d vs %d", r1, rows);
    }
}

int main() {
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        std::printf("no GPU, skipped\n");
        return 0;
    }
    std::mt19937 rng(1234);
    test_gemm(256, 512, 1, rng);
    test_gemm(384, 1024, 5, rng);
    test_gemm(128, 2048, 16, rng);
    test_gemm(256, 640, 37, rng);
    test_gemm(512, 384, 64, rng);
    test_gemm(256, 512, 100, rng);
    test_gemm(384, 768, 300, rng);
    std::printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
