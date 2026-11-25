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
#include "behemoth/qwen_kernels.hpp"

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

// flash prefill attention vs the scalar split-K kernel (same paged bf16 cache)
static void test_flash(int rows, int pos0, std::mt19937 &rng) {
    const int nq = 12, nkv = 2, hd = 256, len = pos0 + rows;
    const int pages = (len + 63) / 64 + 1;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> q(size_t(rows) * nq * 2 * hd), g(size_t(rows) * nq * hd), kc(size_t(pages) * 64 * nkv * hd),
        vc(kc.size());
    for (auto &x : q) x = f2bf(nd(rng));
    for (auto &x : g) x = f2bf(nd(rng));
    for (auto &x : kc) x = f2bf(nd(rng));
    for (auto &x : vc) x = f2bf(nd(rng));
    std::vector<int> pt;
    for (int i = 0; i < pages; ++i) pt.push_back((i * 7 + 3) % pages);  // shuffled pages
    DeviceBuffer<uint16_t> dq, dg, dk, dv, o1(size_t(rows) * nq * hd), o2(size_t(rows) * nq * hd);
    DeviceBuffer<int> dpt;
    DeviceBuffer<float> scr(q_attn_split_scratch(rows, nq, nkv, hd, len) + 16);
    dq.upload(q);
    dg.upload(g);
    dk.upload(kc);
    dv.upload(vc);
    dpt.upload(pt);
    const float sc = 1.0f / 16.0f;
    q_attn_split(dq.data(), dk.data(), dv.data(), dg.data(), o1.data(), rows, pos0, nq, nkv, hd, sc, scr.data(), scr.size(),
                 nullptr, dpt.data());
    q_flash_prefill(dq.data(), dk.data(), dv.data(), dg.data(), o2.data(), rows, pos0, nq, nkv, hd, sc, dpt.data(), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto a = o1.download(), b = o2.download();
    double md = 0, mm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        md = std::max(md, double(std::fabs(bf2f(a[i]) - bf2f(b[i]))));
        mm = std::max(mm, double(std::fabs(bf2f(a[i]))));
    }
    CHECK(md < 0.02 * mm + 1e-3, "flash rows %d pos0 %d: max diff %.4g (max |o| %.3g)", rows, pos0, md, mm);
    std::printf("flash rows %d pos0 %d: max diff %.4g (max |o| %.3g)\n", rows, pos0, md, mm);
}

// ---- fp8 KV cache (QKvCache with scales): random e4m3 caches with power-of-two scales, so the
// dequantised values are exact in bf16 and every fp8 read path can be compared with its bf16 twin.
struct KvPair {
    std::vector<uint8_t> k8, v8;
    std::vector<float> ks, vs;
    std::vector<uint16_t> kb, vb;  // bf16 of e4m3 * scale (exact)
};
static KvPair make_kv(size_t rows, int nkv, int hd, std::mt19937 &rng) {
    std::uniform_int_distribution<int> sign(0, 1), ex(4, 8), man(0, 7), sc(-3, 2);
    KvPair p;
    const size_t n = rows * nkv * hd;
    p.k8.resize(n);
    p.v8.resize(n);
    p.kb.resize(n);
    p.vb.resize(n);
    p.ks.resize(rows * nkv);
    p.vs.resize(rows * nkv);
    for (size_t i = 0; i < rows * nkv; ++i) {
        p.ks[i] = std::ldexp(1.f, sc(rng));
        p.vs[i] = std::ldexp(1.f, sc(rng));
    }
    for (size_t i = 0; i < n; ++i) {
        p.k8[i] = uint8_t(sign(rng) << 7 | ex(rng) << 3 | man(rng));
        p.v8[i] = uint8_t(sign(rng) << 7 | ex(rng) << 3 | man(rng));
        p.kb[i] = f2bf(e4m3(p.k8[i]) * p.ks[i / hd]);
        p.vb[i] = f2bf(e4m3(p.v8[i]) * p.vs[i / hd]);
    }
    return p;
}
static double max_abs_diff(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, double *mag = nullptr) {
    double md = 0, mm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        md = std::max(md, double(std::fabs(bf2f(a[i]) - bf2f(b[i]))));
        mm = std::max(mm, double(std::fabs(bf2f(a[i]))));
    }
    if (mag) *mag = mm;
    return md;
}
// prefill read paths: flash + split-K on e4m3 K (and V) vs the same kernels on the exact bf16 twin
static void test_flash_fp8(int rows, int pos0, bool v8, std::mt19937 &rng) {
    const int nq = 12, nkv = 2, hd = 256, len = pos0 + rows;
    const int pages = (len + 63) / 64 + 1;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> q(size_t(rows) * nq * 2 * hd), g(size_t(rows) * nq * hd);
    for (auto &x : q) x = f2bf(nd(rng) * 0.25f);
    for (auto &x : g) x = f2bf(nd(rng));
    KvPair kv = make_kv(size_t(pages) * 64, nkv, hd, rng);
    std::vector<int> pt;
    for (int i = 0; i < pages; ++i) pt.push_back((i * 5 + 1) % pages);
    DeviceBuffer<uint16_t> dq, dg, dkb, dvb, o[4];
    DeviceBuffer<uint8_t> dk8, dv8;
    DeviceBuffer<float> dks, dvs;
    DeviceBuffer<int> dpt;
    for (auto &x : o) x.resize(size_t(rows) * nq * hd);
    DeviceBuffer<float> scr(q_attn_split_scratch(rows, nq, nkv, hd, len) + 16);
    dq.upload(q);
    dg.upload(g);
    dkb.upload(kv.kb);
    dvb.upload(kv.vb);
    dk8.upload(kv.k8);
    dv8.upload(kv.v8);
    dks.upload(kv.ks);
    dvs.upload(kv.vs);
    dpt.upload(pt);
    QKvCache cb, c8;
    cb.k = dkb.data();
    cb.v = dvb.data();
    c8.k = dk8.data();
    c8.ks = dks.data();
    c8.v = v8 ? static_cast<void *>(dv8.data()) : static_cast<void *>(dvb.data());
    c8.vs = v8 ? dvs.data() : nullptr;
    const float sc = 1.0f / 16.0f;
    q_flash_prefill(dq.data(), cb, dg.data(), o[0].data(), rows, pos0, nq, nkv, hd, sc, dpt.data(), nullptr);
    q_flash_prefill(dq.data(), c8, dg.data(), o[1].data(), rows, pos0, nq, nkv, hd, sc, dpt.data(), nullptr);
    q_attn_split(dq.data(), cb, dg.data(), o[2].data(), rows, pos0, nq, nkv, hd, sc, scr.data(), scr.size(), nullptr, dpt.data());
    q_attn_split(dq.data(), c8, dg.data(), o[3].data(), rows, pos0, nq, nkv, hd, sc, scr.data(), scr.size(), nullptr, dpt.data());
    HIP_CHECK(hipDeviceSynchronize());
    const auto a = o[0].download(), b = o[1].download(), c = o[2].download(), d = o[3].download();
    CHECK(std::memcmp(a.data(), b.data(), a.size() * 2) == 0, "flash fp8 (v8 %d) rows %d pos0 %d: not bit-identical to bf16 twin",
          int(v8), rows, pos0);
    double mm = 0;
    const double d1 = max_abs_diff(c, d, &mm), d2 = max_abs_diff(a, d);
    CHECK(d1 < 4e-3 * mm + 1e-4, "split fp8 (v8 %d) vs bf16 twin: %.4g (max |o| %.3g)", int(v8), d1, mm);
    CHECK(d2 < 0.02 * mm + 1e-3, "flash fp8 vs split fp8 (v8 %d): %.4g (max |o| %.3g)", int(v8), d2, mm);
    std::printf("kv fp8 (v8 %d) rows %d pos0 %d: split vs twin %.3g, flash vs split %.3g (max |o| %.3g)\n", int(v8), rows,
                pos0, d1, d2, mm);
}
// decode read path: attn_b on e4m3 caches vs attn_b on the bf16 twin and vs split-K (1 row per
// sequence); B rows of different slots / lengths; batch invariance (B = 1 row 0 == B row 0, bits)
static void test_attn_b_fp8(int B, bool v8, std::mt19937 &rng) {
    const int nq = 12, nkv = 2, hd = 256, maxctx = 4096, maxpages = maxctx / 64;
    const int slots = B, pages = slots * maxpages;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> lenu(1, maxctx - 1);
    KvPair kv = make_kv(size_t(pages) * 64, nkv, hd, rng);
    std::vector<int> ptab(size_t(slots) * maxpages);
    for (int i = 0; i < slots * maxpages; ++i) ptab[size_t(i)] = (i * 37 + 11) % pages;  // permutation (gcd 1)
    std::vector<int> meta(size_t(3) * B);
    for (int b = 0; b < B; ++b) {
        meta[size_t(b)] = 0;
        meta[size_t(B + b)] = (b * 3 + 1) % slots;
        meta[size_t(2 * B + b)] = b == 0 ? 600 : lenu(rng);  // position (length - 1)
    }
    std::vector<uint16_t> q(size_t(B) * nq * hd), g(size_t(B) * nq * hd);
    for (auto &x : q) x = f2bf(nd(rng) * 0.25f);
    for (auto &x : g) x = f2bf(nd(rng));
    DeviceBuffer<uint16_t> dq, dg, dkb, dvb, o0, o1, o2;
    DeviceBuffer<uint8_t> dk8, dv8;
    DeviceBuffer<float> dks, dvs, scr(q_attn_b_scratch(B, nq, maxctx));
    DeviceBuffer<int> dptab, dmeta, dmeta1;
    dq.upload(q);
    dg.upload(g);
    dkb.upload(kv.kb);
    dvb.upload(kv.vb);
    dk8.upload(kv.k8);
    dv8.upload(kv.v8);
    dks.upload(kv.ks);
    dvs.upload(kv.vs);
    dptab.upload(ptab);
    dmeta.upload(meta);
    std::vector<int> meta1 = {meta[0], meta[size_t(B)], meta[size_t(2 * B)]};
    dmeta1.upload(meta1);
    o0.resize(q.size());
    o1.resize(q.size());
    o2.resize(size_t(nq) * hd);
    QKvCache cb, c8;
    cb.k = dkb.data();
    cb.v = dvb.data();
    c8.k = dk8.data();
    c8.ks = dks.data();
    c8.v = v8 ? static_cast<void *>(dv8.data()) : static_cast<void *>(dvb.data());
    c8.vs = v8 ? dvs.data() : nullptr;
    const float sc = 1.0f / 16.0f;
    q_attn_b(dq.data(), cb, dg.data(), o0.data(), dmeta.data(), B, B, dptab.data(), maxpages, nq, nkv, maxctx, sc, scr.data(), nullptr);
    q_attn_b(dq.data(), c8, dg.data(), o1.data(), dmeta.data(), B, B, dptab.data(), maxpages, nq, nkv, maxctx, sc, scr.data(), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    q_attn_b(dq.data(), c8, dg.data(), o2.data(), dmeta1.data(), 1, 1, dptab.data(), maxpages, nq, nkv, maxctx, sc, scr.data(), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    const auto a = o0.download(), b = o1.download(), b1 = o2.download();
    double mm = 0;
    const double d1 = max_abs_diff(a, b, &mm);
    CHECK(d1 < 4e-3 * mm + 1e-4, "attn_b fp8 (v8 %d) B %d vs bf16 twin: %.4g (max |o| %.3g)", int(v8), B, d1, mm);
    CHECK(std::memcmp(b.data(), b1.data(), b1.size() * 2) == 0, "attn_b fp8 batch invariance B %d", B);
    // row b vs split-K on the same fp8 cache (one row at position pos, the slot's page table)
    double d2 = 0;
    for (int r = 0; r < B; ++r) {
        const int slot = meta[size_t(B + r)], pos = meta[size_t(2 * B + r)];
        DeviceBuffer<uint16_t> q2, g2, os(size_t(nq) * hd);
        std::vector<uint16_t> qq(size_t(nq) * 2 * hd), gg(g.begin() + size_t(r) * nq * hd, g.begin() + size_t(r + 1) * nq * hd);
        for (int h = 0; h < nq; ++h)
            for (int d = 0; d < hd; ++d) qq[size_t(h) * 2 * hd + d] = q[(size_t(r) * nq + h) * hd + d];
        q2.upload(qq);
        g2.upload(gg);
        DeviceBuffer<float> s2(q_attn_split_scratch(1, nq, nkv, hd, pos + 1) + 16);
        q_attn_split(q2.data(), c8, g2.data(), os.data(), 1, pos, nq, nkv, hd, sc, s2.data(), s2.size(), nullptr,
                     dptab.data() + size_t(slot) * maxpages);
        HIP_CHECK(hipDeviceSynchronize());
        const auto o = os.download();
        for (int i = 0; i < nq * hd; ++i)
            d2 = std::max(d2, double(std::fabs(bf2f(o[size_t(i)]) - bf2f(b[size_t(r) * nq * hd + i]))));
    }
    CHECK(d2 < 4e-3 * mm + 1e-4, "attn_b fp8 vs split fp8 (v8 %d): %.4g", int(v8), d2);
    std::printf("attn_b fp8 (v8 %d) B %d: vs twin %.3g, vs split %.3g (max |o| %.3g)\n", int(v8), B, d1, d2, mm);
}
// write paths: q_rope_b / q_qk_norm_rope into e4m3 caches == e4m3 quantisation of the bf16 cache rows
// (scale = amax / 448, |deq - x| <= |x| / 16 + scale / 64), same rows in both kernels
static void check_quant_rows(const std::vector<uint16_t> &ref, const std::vector<uint8_t> &q8, const std::vector<float> &qs,
                             int hd, const std::vector<long> &rows_idx, const char *what) {
    double worst = 0, sworst = 0;
    for (long idx : rows_idx) {
        float am = 0;
        for (int d = 0; d < hd; ++d) am = std::max(am, std::fabs(bf2f(ref[size_t(idx) * hd + d])));
        const float s = qs[size_t(idx)];
        sworst = std::max(sworst, double(std::fabs(s - am / 448.f)) / (am / 448.f + 1e-30));
        for (int d = 0; d < hd; ++d) {
            const float x = bf2f(ref[size_t(idx) * hd + d]), dq = e4m3(q8[size_t(idx) * hd + d]) * s;
            worst = std::max(worst, double(std::fabs(dq - x)) / (std::fabs(x) / 16 + s / 64 + 1e-30));
        }
    }
    CHECK(worst <= 1.0 && sworst < 1e-6, "%s: quant err ratio %.3f, scale err %.3g", what, worst, sworst);
    std::printf("%s: quant err ratio %.3f, scale rel err %.3g (%zu rows)\n", what, worst, sworst, rows_idx.size());
}
static void test_kv_write_fp8(std::mt19937 &rng) {
    const int nq = 12, nkv = 2, hd = 256, rot = 64, B = 5, maxpages = 4, pages = 8;
    const int Wf = nq * 2 * hd + 2 * nkv * hd;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> z(size_t(B) * Wf);
    for (size_t i = 0; i < z.size(); ++i) z[i] = nd(rng) * (1.f + float(i % 13));
    std::vector<uint16_t> qn(hd), kn(hd);
    for (auto &x : qn) x = f2bf(nd(rng) * 0.1f);
    for (auto &x : kn) x = f2bf(nd(rng) * 0.1f);
    std::vector<int> ptab(size_t(B) * maxpages), meta(size_t(3) * B);
    for (size_t i = 0; i < ptab.size(); ++i) ptab[i] = int(i % pages);
    for (int b = 0; b < B; ++b) {
        meta[size_t(B + b)] = b;
        meta[size_t(2 * B + b)] = 7 * b + 3;
    }
    const size_t n = size_t(pages) * 64 * nkv * hd;
    DeviceBuffer<float> dz, dks(size_t(pages) * 64 * nkv), dvs(size_t(pages) * 64 * nkv);
    DeviceBuffer<uint16_t> dqn, dkn, bq(size_t(B) * nq * hd), bG(size_t(B) * nq * hd), kb(n), vb(n);
    DeviceBuffer<uint8_t> k8(n), v8(n);
    DeviceBuffer<int> dptab, dmeta;
    dz.upload(z);
    dqn.upload(qn);
    dkn.upload(kn);
    dptab.upload(ptab);
    dmeta.upload(meta);
    QKvCache cb, c8;
    cb.k = kb.data();
    cb.v = vb.data();
    c8.k = k8.data();
    c8.v = v8.data();
    c8.ks = dks.data();
    c8.vs = dvs.data();
    q_rope_b(dz.data(), Wf, bq.data(), bG.data(), cb, dmeta.data(), B, B, dptab.data(), maxpages, nq, nkv, hd, rot, 1e7f,
             dqn.data(), dkn.data(), 1e-6f, nullptr);
    q_rope_b(dz.data(), Wf, bq.data(), bG.data(), c8, dmeta.data(), B, B, dptab.data(), maxpages, nq, nkv, hd, rot, 1e7f,
             dqn.data(), dkn.data(), 1e-6f, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<long> idx;
    for (int b = 0; b < B; ++b) {
        const int pos = meta[size_t(2 * B + b)];
        const long row = long(ptab[size_t(b) * maxpages + pos / 64]) * 64 + pos % 64;
        for (int h = 0; h < nkv; ++h) idx.push_back(row * nkv + h);
    }
    check_quant_rows(kb.download(), k8.download(), dks.download(), hd, idx, "rope_b K e4m3");
    check_quant_rows(vb.download(), v8.download(), dvs.download(), hd, idx, "rope_b V e4m3");
    // prefill append: rows consecutive positions 0.. of one sequence (contiguous)
    const int rows = 70;
    std::vector<uint16_t> q(size_t(rows) * nq * 2 * hd), k(size_t(rows) * nkv * hd), v(k.size());
    for (auto &x : q) x = f2bf(nd(rng));
    for (auto &x : k) x = f2bf(nd(rng) * 3.f);
    for (auto &x : v) x = f2bf(nd(rng) * 2.f);
    DeviceBuffer<uint16_t> dq1, dk1, dv1, dq2, dk2, dv2, G(size_t(rows) * nq * hd);
    dq1.upload(q);
    dk1.upload(k);
    dv1.upload(v);
    dq2.upload(q);
    dk2.upload(k);
    dv2.upload(v);
    q_qk_norm_rope(dq1.data(), dk1.data(), dv1.data(), G.data(), cb, rows, 0, nq, nkv, hd, rot, 1e7f, dqn.data(), dkn.data(),
                   1e-6f, nullptr);
    q_qk_norm_rope(dq2.data(), dk2.data(), dv2.data(), G.data(), c8, rows, 0, nq, nkv, hd, rot, 1e7f, dqn.data(), dkn.data(),
                   1e-6f, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    idx.clear();
    for (long i = 0; i < long(rows) * nkv; ++i) idx.push_back(i);
    check_quant_rows(kb.download(), k8.download(), dks.download(), hd, idx, "qk_norm_rope K e4m3");
    check_quant_rows(vb.download(), v8.download(), dvs.download(), hd, idx, "qk_norm_rope V e4m3");
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
    test_flash(16, 0, rng);
    test_flash(37, 5, rng);
    test_flash(200, 300, rng);
    test_flash_fp8(16, 0, false, rng);
    test_flash_fp8(37, 5, true, rng);
    test_flash_fp8(200, 300, true, rng);
    test_attn_b_fp8(1, true, rng);
    test_attn_b_fp8(7, false, rng);
    test_attn_b_fp8(16, true, rng);
    test_kv_write_fp8(rng);
    std::printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
