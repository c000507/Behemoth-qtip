// Qwen3.5 runtime GPU regression tests (gfx1201):
//  1. q_gateup vs host silu(g) * u (the kernel once truncated silu(g) to an integer: every MLP was 0)
//  2. q_delta_step chunk continuity: one T-row chunk == the same rows split into chunks / single-row
//     decode steps with the conv ring and delta state carried over (ring indexed by absolute position)
//  3. (needs a dense Qwen3.5 checkpoint, argv[1] or /mnt/nvme/models/Qwen3.5-0.8B; skipped if absent)
//     QwenModel prefill(N) last-token logits == prefill(N-k) + k decode steps, both argmax and values,
//     on the original HF bf16 checkpoint (bf16-linear fallback, "model.language_model." prefix).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#include "behemoth/qwen_kernels.hpp"
#include "behemoth/qwen_model.hpp"

using namespace behemoth;

static int g_fail = 0;
#define EXPECT(c, ...)                                                                             \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            std::printf("  FAIL: ");                                                               \
            std::printf(__VA_ARGS__);                                                              \
            std::printf("\n");                                                                     \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static float bf(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static uint16_t tobf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

static void test_gateup(hipStream_t s) {
    std::printf("[qwen] q_gateup vs host\n");
    const int rows = 3, F = 300;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> g(size_t(rows) * F), u(g.size()), sv(size_t(F), 1.0f);
    for (auto &x : g) x = nd(rng);
    for (auto &x : u) x = nd(rng);
    DeviceBuffer<float> dg, du, dsv;
    dg.upload(g);
    du.upload(u);
    dsv.upload(sv);
    DeviceBuffer<uint16_t> dh(g.size());
    q_gateup(dg.data(), du.data(), dsv.data(), dsv.data(), dh.data(), rows, F, F, s);
    HIP_CHECK(hipStreamSynchronize(s));
    auto h = dh.download();
    double worst = 0, mag = 0;
    for (size_t i = 0; i < h.size(); ++i) {
        float gb = bf(tobf(g[i])), ub = bf(tobf(u[i]));
        float ref = gb / (1.f + std::exp(-gb)) * ub;
        worst = std::max(worst, double(std::fabs(bf(h[i]) - ref)) / (std::fabs(ref) + 1e-2));
        mag += std::fabs(bf(h[i]));
    }
    std::printf("  worst rel err %.2e, mean |h| %.3f\n", worst, mag / double(h.size()));
    EXPECT(worst < 1e-2, "q_gateup mismatch (worst rel %.3e)", worst);
    EXPECT(mag > 0.1 * double(h.size()), "q_gateup output collapsed");
}

static void test_delta_chunks(hipStream_t s) {
    std::printf("[qwen] q_delta_step chunked == single chunk\n");
    const int nk = 2, nv = 4, kd = 32, vd = 32, K = 4, T = 13;
    const int key_dim = nk * kd, V = nv * vd, conv_dim = 2 * key_dim + V;
    const int W = conv_dim + V + 2 * nv;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> qkvz(size_t(T) * W), a_dt(size_t(2 * nv));
    for (auto &x : qkvz) x = nd(rng);
    for (auto &x : a_dt) x = 0.3f * nd(rng);
    std::vector<uint16_t> conv(static_cast<size_t>(conv_dim) * K), dn(static_cast<size_t>(vd));
    for (auto &x : conv) x = tobf(0.5f * nd(rng));
    for (auto &x : dn) x = tobf(1.f + 0.1f * nd(rng));
    DeviceBuffer<float> d_adt;
    d_adt.upload(a_dt);
    DeviceBuffer<uint16_t> d_conv, d_dn;
    d_conv.upload(conv);
    d_dn.upload(dn);
    auto run = [&](const std::vector<int> &chunks) {
        DeviceBuffer<float> ring(size_t(conv_dim) * K), S(size_t(nv) * kd * vd), z;
        HIP_CHECK(hipMemsetAsync(ring.data(), 0, ring.bytes(), s));
        HIP_CHECK(hipMemsetAsync(S.data(), 0, S.bytes(), s));
        std::vector<uint16_t> out;
        int pos = 0;
        for (int rows : chunks) {
            std::vector<float> part(qkvz.begin() + long(pos) * W, qkvz.begin() + long(pos + rows) * W);
            z.upload(part);
            DeviceBuffer<uint16_t> o(size_t(rows) * V);
            for (int t = 0; t < rows; ++t)
                q_delta_step(z.data(), d_conv.data(), d_dn.data(), d_adt.data(), ring.data(), S.data(), o.data(),
                             rows, t, pos, conv_dim, V, kd, vd, nk, nv, K, 1e-6f, s);
            HIP_CHECK(hipStreamSynchronize(s));
            auto h = o.download();
            out.insert(out.end(), h.begin(), h.end());
            pos += rows;
        }
        return out;
    };
    auto whole = run({T});
    auto split = run({5, 1, 1, 2, 4});
    auto steps = run(std::vector<int>(size_t(T), 1));
    double d1 = 0, d2 = 0, mag = 0;
    for (size_t i = 0; i < whole.size(); ++i) {
        d1 = std::max(d1, double(std::fabs(bf(whole[i]) - bf(split[i]))));
        d2 = std::max(d2, double(std::fabs(bf(whole[i]) - bf(steps[i]))));
        mag = std::max(mag, double(std::fabs(bf(whole[i]))));
    }
    std::printf("  max|whole - split| %.3e, max|whole - per-token| %.3e (max|out| %.3f)\n", d1, d2, mag);
    EXPECT(d1 <= 1e-6 * std::max(1.0, mag) && d2 <= 1e-6 * std::max(1.0, mag), "delta chunk continuity");
}

static void test_model_decode(const std::filesystem::path &dir) {
    std::printf("[qwen] model prefill vs decode consistency (%s)\n", dir.c_str());
    QwenModel m(dir, 0, 256, 512);
    const int V = m.config().vocab;
    // "In 1905, Albert Einstein published" + "The quick brown fox" + "The capital of France is"
    const std::vector<int> toks = {623, 220, 16, 24, 15, 20, 11, 16993, 52237, 4501,
                                   760, 3841, 13477, 37550, 760, 6511, 314, 9338, 369};
    const int N = int(toks.size()), k = 6;
    std::vector<float> a(static_cast<size_t>(V)), b(static_cast<size_t>(V));
    m.reset();
    m.prefill(toks.data(), size_t(N), a.data());
    m.reset();
    m.prefill(toks.data(), size_t(N - k), nullptr);
    for (int i = N - k; i < N; ++i) m.decode(toks[size_t(i)], b.data());
    int am = int(std::max_element(a.begin(), a.end()) - a.begin());
    int bm = int(std::max_element(b.begin(), b.end()) - b.begin());
    double maxd = 0, scale = 0;
    for (int v = 0; v < V; ++v) {
        maxd = std::max(maxd, double(std::fabs(a[size_t(v)] - b[size_t(v)])));
        scale = std::max(scale, double(std::fabs(a[size_t(v)])));
    }
    std::printf("  argmax prefill %d decode %d, logits max|d| %.4f (max|logit| %.2f)\n", am, bm, maxd, scale);
    EXPECT(am == bm, "prefill/decode argmax differ");
    EXPECT(maxd < 0.03 * scale, "prefill/decode logits differ (%.4f)", maxd);
    // the 0.8B answers "The capital of France is" with " Paris" (11751) in the CPU fp32 reference
    if (dir.filename() == "Qwen3.5-0.8B") EXPECT(am == 11751, "expected ' Paris' (11751), got %d", am);
}

int main(int argc, char **argv) {
    hipStream_t s;
    HIP_CHECK(hipStreamCreate(&s));
    test_gateup(s);
    test_delta_chunks(s);
    std::filesystem::path dir = argc >= 2 ? argv[1] : "/mnt/nvme/models/Qwen3.5-0.8B";
    if (std::filesystem::exists(dir / "config.json")) test_model_decode(dir);
    else std::printf("[qwen] model test skipped (no %s)\n", dir.c_str());
    (void)hipStreamDestroy(s);
    std::printf(g_fail ? "QWEN GPU TESTS FAILED (%d)\n" : "ALL QWEN GPU TESTS PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
