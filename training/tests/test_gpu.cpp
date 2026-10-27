#include "hip_ops.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace qat;
int main() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
        std::cout << "no HIP device; skipped\n";
        return 77;
    }
    try {
        auto cpu = cpu_ops(true), gpu = hip_ops(0, size_t(512) << 20);
        int T = 17;
        Config c({{"hidden_size", 1024},
                  {"intermediate_size", 1024},
                  {"num_attention_heads", 8},
                  {"num_key_value_heads", 2},
                  {"head_dim", 128},
                  {"num_hidden_layers", 1},
                  {"vocab_size", 16},
                  {"quip_params",
                   {{"L", 16},
                    {"K", 2},
                    {"V", 2},
                    {"tlut_bits", 9},
                    {"decode_mode", "quantlut_sym"},
                    {"td_x", 16},
                    {"td_y", 16}}}});
        Batch b;
        b.tokens.assign(T, 0);
        b.segments.resize(T);
        b.positions.resize(T);
        for (int i = 0; i < T; ++i) {
            b.segments[i] = i < 7 ? 0 : 1;
            b.positions[i] = i < 7 ? i : i - 7;
        }
        b.boundaries();
        std::vector<uint16_t> frozen(16 * 1024);
        for (size_t i = 0; i < frozen.size(); ++i)
            frozen[i] = behemoth::BFloat16(float(std::sin(i * .11) * .1)).bits;
        auto cw = cpu->frozen(frozen, 16, 1024), gw = gpu->frozen(frozen, 16, 1024);
        auto ce = cpu->embedding(cw, b), ge = gpu->embedding(gw, b);
        auto compare = [&](Mat a, Mat d, double tol) {
            std::vector<float> x(a.size()), y(d.size());
            cpu->download(a, x);
            gpu->download(d, y);
            require(x.size() == y.size(), "GPU shape mismatch");
            double ss = 0, ref = 0, max = 0;
            for (size_t i = 0; i < x.size(); ++i) {
                require(std::isfinite(x[i]) && std::isfinite(y[i]), "nonfinite GPU result");
                ss += double(x[i] - y[i]) * (x[i] - y[i]);
                ref += double(x[i]) * x[i];
                max = std::max(max, std::abs(double(x[i]) - y[i]));
            }
            require(std::sqrt(ss / std::max(ref, 1e-20)) < tol || max < 2e-5,
                    "GPU relative error " + std::to_string(std::sqrt(ss / std::max(ref, 1e-20))));
        };
        auto pair = [&](int r, int cols, double phase) {
            std::vector<float> v(size_t(r) * cols);
            for (size_t i = 0; i < v.size(); ++i)
                v[i] = bf(float(std::sin(i * .13 + phase) * .15));
            auto a = cpu->alloc(r, cols), d = gpu->alloc(r, cols);
            cpu->upload(a, v);
            gpu->upload(d, v);
            return std::pair(a, d);
        };
        compare(ce, ge, 1e-6);
        compare(cpu->had(ce, true), gpu->had(ge, true), 1e-4);
        auto [w, z] = pair(1, 1024, .3);
        auto cn = cpu->rms(ce, w, c.eps), gn = gpu->rms(ge, z, c.eps);
        compare(cn.y, gn.y, .006);
        auto [dy, gdy] = pair(T, 1024, .8);
        auto dw = cpu->alloc(1, 1024, true), gdw = gpu->alloc(1, 1024, true);
        compare(cpu->rms_backward(ce, cn, w, dy, dw), gpu->rms_backward(ge, gn, z, gdy, gdw), .002);
        compare(dw, gdw, .002);
        for (int rows : {1, 17, 255, 256, 257, 513}) {
            for (int cols : {31, 128, 1024}) {
                // Strided views and partial tiles exercise the coalesced, two-pass
                // reduction. Accumulators start nonzero and are updated twice.
                auto [ax, bx] = pair(rows, cols + 16, .2);
                auto [ad, bd] = pair(rows, cols + 16, .7);
                ax = ax.col(0, cols);
                bx = bx.col(0, cols);
                ad = ad.col(0, cols);
                bd = bd.col(0, cols);
                auto [ac, bc] = pair(1, cols, .4);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    cpu->colsum(ax, ad, ac);
                    gpu->colsum(bx, bd, bc);
                }
                compare(ac, bc, .0001);
                auto [aw, bw] = pair(1, cols, .3);
                auto an = cpu->rms(ax, aw, c.eps), bn = gpu->rms(bx, bw, c.eps);
                auto [ag, bg] = pair(1, cols, .5);
                for (int repeat = 0; repeat < 2; ++repeat)
                    compare(cpu->rms_backward(ax, an, aw, ad, ag), gpu->rms_backward(bx, bn, bw, bd, bg),
                            .002);
                compare(ag, bg, .002);
            }
        }
        compare(cpu->swiglu(ce, dy), gpu->swiglu(ge, gdy), .007);
        auto cg = cpu->swiglu_backward(ce, dy, ce), gg = gpu->swiglu_backward(ge, gdy, ge);
        compare(cg.first, gg.first, .002);
        compare(cg.second, gg.second, .002);
        compare(cpu->rope(ce, c.heads, c, b), gpu->rope(ge, c.heads, c, b), .006);
        Batch shifted = b;
        for (auto &pos : shifted.positions)
            pos += 1000;
        gpu->embedding(gw, shifted);
        for (float theta : {1000000.f, 10000.f, 1000000.f}) {
            Config rc = c;
            rc.theta = theta;
            for (bool back : {false, true})
                compare(cpu->rope(ce, c.heads, rc, shifted, back), gpu->rope(ge, c.heads, rc, shifted, back),
                        .006);
        }
        gpu->embedding(gw, b);
        compare(cpu->rope(ce, c.heads, c, b), gpu->rope(ge, c.heads, c, b), .006);
        std::vector<Weight> cpu_weights{cw, cw}, gpu_weights{gw, gw};
        auto cm = cpu->had_mm(ce, cpu_weights, w), gm = gpu->had_mm(ge, gpu_weights, z);
        for (int i = 0; i < 2; ++i)
            compare(cm[i], gm[i], .003);
        for (int width : {1024, 12288, 28672}) {
            auto [ax, bx] = pair(3, width + 16, .1);
            ax = ax.col(0, width);
            bx = bx.col(0, width);
            auto [ap, bp] = pair(1, width, .3);
            std::vector<uint16_t> data(size_t(width) * 16);
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = behemoth::BFloat16(float(std::sin(i * .17) * .02)).bits;
            for (bool transposed : {false, true}) {
                auto aw = cpu->frozen(data, transposed ? width : 16, transposed ? 16 : width);
                auto bw = gpu->frozen(data, transposed ? width : 16, transposed ? 16 : width);
                auto am = cpu->had_mm(ax, std::span(&aw, 1), ap, transposed);
                auto bm = gpu->had_mm(bx, std::span(&bw, 1), bp, transposed);
                compare(am.front(), bm.front(), .006);
            }
        }
        auto [q, gq] = pair(T, 1024, .1);
        auto [k, gk] = pair(T, 256, .4);
        auto [v, gv] = pair(T, 256, .9);
        auto ca = cpu->attention(q, k, v, c, b), ga = gpu->attention(gq, gk, gv, c, b);
        compare(ca.out, ga.out, .012);
        compare(ca.lse, ga.lse, .001);
        auto cb = cpu->attention_backward(ca, dy, c, b), gb = gpu->attention_backward(ga, gdy, c, b);
        for (int i = 0; i < 3; ++i)
            compare(cb[i], gb[i], .04);
        // Exercise both ownership directions, partial tiles, GQA reductions,
        // document boundaries inside key tiles, and nontrivial softmax scores.
        for (int length : {1, 17, 33, 63, 64, 65, 129, 257}) {
            for (int group : {1, 3, 4, 12, 32}) {
                if (group == 32 && length != 65)
                    continue;
                Config ac = c;
                ac.kv = 2;
                ac.heads = group * ac.kv;
                ac.n = ac.heads * ac.dim;
                for (int packing : {0, 1, 2}) {
                    bool packed = packing != 0;
                    Batch ab;
                    ab.tokens.assign(length, 0);
                    ab.positions.resize(length);
                    ab.segments.resize(length);
                    for (int t = 0; t < length; ++t) {
                        // Include one-token docs and boundaries inside both
                        // the 16-key wave tile and the 32-key block tile.
                        int start = !packed ? 0 : t < 7 ? 0 : t < 31 ? 7 : t < 32 ? 31 : t < 63 ? 32
                                                        : t < 64 ? 63 : t < 70 ? 64 : 70;
                        if (packing == 2)
                            start = t; // Every token is its own document.
                        ab.segments[t] = start;
                        ab.positions[t] = t - start;
                    }
                    ab.boundaries();
                    gpu->embedding(gw, ab); // Upload packed boundary metadata.
                    auto attention_pair = [&](int cols, unsigned seed, float scale) {
                        // Extra columns deliberately exercise strided Q/K/V/dY views.
                        auto a = cpu->alloc(length, cols + 16).col(0, cols);
                        auto d = gpu->alloc(length, cols + 16).col(0, cols);
                        std::vector<float> data(size_t(length) * cols);
                        for (float &x : data) {
                            seed = seed * 1664525u + 1013904223u;
                            x = bf((float(seed >> 8) / 8388608.f - 1.f) * scale);
                        }
                        cpu->upload(a, data);
                        gpu->upload(d, data);
                        return std::pair(a, d);
                    };
                    auto [aq, bq] = attention_pair(ac.n, 11, packed ? 3.f : 1.f);
                    auto [ak, bk] = attention_pair(ac.kv * 128, 19, packed ? 3.f : 1.f);
                    auto [av, bv] = attention_pair(ac.kv * 128, 37, 1.f);
                    auto [ad, bd] = attention_pair(ac.n, 43, 1.f);
                    compare(cpu->rope(aq, ac.heads, ac, ab), gpu->rope(bq, ac.heads, ac, ab), .006);
                    compare(cpu->rope(ak, ac.kv, ac, ab, true), gpu->rope(bk, ac.kv, ac, ab, true), .006);
                    auto aa = cpu->attention(aq, ak, av, ac, ab);
                    auto ba = gpu->attention(bq, bk, bv, ac, ab);
                    compare(aa.out, ba.out, .012);
                    compare(aa.lse, ba.lse, .001);
                    auto ag = cpu->attention_backward(aa, ad, ac, ab);
                    auto bg = gpu->attention_backward(ba, bd, ac, ab);
                    for (int i = 0; i < 3; ++i)
                        compare(ag[i], bg[i], .04);
                    auto repeated = gpu->attention_backward(ba, bd, ac, ab);
                    for (int i = 0; i < 3; ++i) {
                        std::vector<float> x(bg[i].size()), y(repeated[i].size());
                        gpu->download(bg[i], x);
                        gpu->download(repeated[i], y);
                        require(x == y, "attention backward must be deterministic");
                    }
                }
            }
        }
        auto cx = cpu->mm(ce, cw), gx = gpu->mm(ge, gw);
        compare(cx, gx, .002);
        compare(cpu->mm(cx, cw, true), gpu->mm(gx, gw, true), .005);
        std::vector<int32_t> labels(T, 3);
        labels[2] = -100;
        double cl = cpu->cross_entropy(cx, labels, T, true), gl = gpu->cross_entropy(gx, labels, T, true);
        require(std::abs(cl - gl) < .001, "GPU CE loss");
        compare(cx, gx, .003);
        std::cout << "native GPU primitives and packed GQA attention gradients passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
