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
        compare(cpu->swiglu(ce, dy), gpu->swiglu(ge, gdy), .007);
        auto cg = cpu->swiglu_backward(ce, dy, ce), gg = gpu->swiglu_backward(ge, gdy, ge);
        compare(cg.first, gg.first, .002);
        compare(cg.second, gg.second, .002);
        compare(cpu->rope(ce, c.heads, c, b), gpu->rope(ge, c.heads, c, b), .006);
        auto [q, gq] = pair(T, 1024, .1);
        auto [k, gk] = pair(T, 256, .4);
        auto [v, gv] = pair(T, 256, .9);
        auto ca = cpu->attention(q, k, v, c, b), ga = gpu->attention(gq, gk, gv, c, b);
        compare(ca.out, ga.out, .012);
        compare(ca.lse, ga.lse, .001);
        auto cb = cpu->attention_backward(ca, dy, c, b), gb = gpu->attention_backward(ga, gdy, c, b);
        for (int i = 0; i < 3; ++i)
            compare(cb[i], gb[i], .04);
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
