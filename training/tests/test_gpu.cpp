#include "common.hpp"
#include "hip_ops.hpp"
#include "safetensors.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <map>
#include <iostream>
#include <random>
#include <unistd.h>
using namespace qat;
namespace {
// One-layer model with HIP-supported widths (hidden = intermediate = 1024,
// eight 128-wide heads) for whole-layer CPU/GPU gradient comparisons.
void write_layer_model(const fs::path &p, int format) {
    fs::create_directories(p);
    const int n = 1024, inter = 1024, vocab = 64;
    json c = {{"hidden_size", n},
              {"intermediate_size", inter},
              {"num_attention_heads", 8},
              {"num_key_value_heads", 8},
              {"head_dim", 128},
              {"num_hidden_layers", 1},
              {"vocab_size", vocab},
              {"rms_norm_eps", 1e-5},
              {"rope_theta", 10000},
              {"hidden_act", "silu"},
              {"quip_params",
               {{"L", 16},
                {"K", 2},
                {"V", 2},
                {"tlut_bits", 9},
                {"decode_mode", "quantlut_sym"},
                {"td_x", 16},
                {"td_y", 16}}}};
    if (format == 1) c["qtip_quant"]["kv_format"] = "fp8";
    if (format == 2) c.erase("quip_params");
    durable_json(p / "config.json", c);
    qtip::SafeTensorsWriter w;
    json index = {{"weight_map", json::object()}};
    std::deque<std::vector<uint16_t>> keep16; // stable references
    std::deque<std::vector<uint8_t>> keep8;
    std::deque<std::vector<float>> keep32;
    auto add = [&](std::string name, std::string type, std::vector<int64_t> shape, const void *data,
                   size_t bytes) {
        w.add(name, type, shape, data, bytes);
        index["weight_map"][name] = "model.safetensors";
    };
    std::mt19937 rng(23);
    std::normal_distribution<float> normal(0, 1);
    auto dense = [&](std::string name, int r, int cols, float scale, float bias) {
        auto &v = keep16.emplace_back(size_t(r) * cols);
        for (auto &x : v)
            x = behemoth::BFloat16(bias + scale * normal(rng)).bits;
        add(name, "BF16", cols == 1 ? std::vector<int64_t>{r} : std::vector<int64_t>{r, cols}, v.data(),
            v.size() * 2);
    };
    std::string prefix = "model.layers.0.";
    auto &lut = keep16.emplace_back(1024);
    for (auto &x : lut)
        x = qtip::f32_to_fp16(.5f * normal(rng));
    for (int i = 0; i < 7; ++i) {
        int out = i == 4 || i == 5 ? inter : n, in = i == 6 ? inter : n;
        std::string b = prefix + linears[i] + ".";
        if (format == 2 || (format == 1 && (i == 1 || i == 2))) {
            auto &fp8 = keep8.emplace_back(size_t(out) * in);
            for (auto &x : fp8) x = behemoth::encode_e4m3(8.f * normal(rng));
            auto &scales = keep32.emplace_back(out, .0025f);
            add(b + "weight", "F8_E4M3", {out, in}, fp8.data(), fp8.size());
            add(b + "weight_scale", "F32", {out}, scales.data(), scales.size() * 4);
            continue;
        }
        auto &codes = keep16.emplace_back(size_t(out) * in / 8);
        for (auto &x : codes)
            x = uint16_t(rng());
        add(b + "trellis", "I16", {out / 16 * (in / 16), 32}, codes.data(), codes.size() * 2);
        add(b + "tlut", "F16", {512, 2}, lut.data(), lut.size() * 2);
        // q/k/v and gate/up share (tie) SU.
        int group = i <= 2 ? 0 : i == 3 ? 3 : i <= 5 ? 4 : 6;
        auto &su = keep16.emplace_back(in);
        for (int j = 0; j < in; ++j)
            su[j] = behemoth::BFloat16((j * 7 + group) % 3 == 0 ? -1.f : 1.f).bits;
        add(b + "SU", "BF16", {in}, su.data(), su.size() * 2);
        auto &sv = keep32.emplace_back(out);
        for (int j = 0; j < out; ++j)
            sv[j] = (j % 2 ? -.02f : .02f) * (1 + .3f * std::sin(j * .7f));
        add(b + "SV", "F32", {out}, sv.data(), sv.size() * 4);
        static int64_t rcp = 0, tp = 8;
        add(b + "rcp", "I64", {}, &rcp, 8);
        add(b + "tp_rank", "I64", {}, &tp, 8);
    }
    dense(prefix + norm1, n, 1, .1f, 1.f);
    dense(prefix + norm2, n, 1, .1f, 1.f);
    dense("model.norm.weight", n, 1, .1f, 1.f);
    dense("model.embed_tokens.weight", vocab, n, .5f, 0.f);
    dense("lm_head.weight", vocab, n, .05f, 0.f);
    w.write((p / "model.safetensors").string());
    durable_json(p / "model.safetensors.index.json", index);
}
// Full forward/backward of a one-layer model on CPU (BF16-boundary emulation)
// and GPU in both activation storage modes. Exercises every fused path.
void layer_check(int format) {
    fs::path root = fs::temp_directory_path() / ("qtip-gpu-layer-" + std::to_string(getpid()));
    fs::create_directories(root);
    struct Cleanup {
        fs::path p;
        ~Cleanup() { fs::remove_all(p); }
    } cleanup{root};
    write_layer_model(root / "model", format);
    Model m(root / "model", format == 2);
    const int T = 40;
    Batch b;
    for (int t = 0; t < T; ++t) {
        b.tokens.push_back((t * 13 + 5) % m.config.vocab);
        b.segments.push_back(t < 17 ? 0 : 17);
        b.positions.push_back(t < 17 ? t : t - 17);
        b.labels.push_back(t % 5 == 0 ? -100 : (t * 7 + 3) % m.config.vocab);
    }
    b.boundaries();
    Options o;
    o.rows = 16; // three MLP chunks, the last one partial
    o.loss_rows = 16;
    size_t denominator = b.targets();
    Replica cpu(m, -1, T, 0, true);
    double cl = cpu.run(b, o, denominator, true);
    auto cg = cpu.grad_values();
    const char *saved = std::getenv("QTIP_TRAIN_STORAGE");
    std::string restore = saved ? saved : "";
    std::vector<float> fp32_grads;
    double fp32_loss = 0;
    for (const char *storage : {"fp32", "bf16"}) {
        setenv("QTIP_TRAIN_STORAGE", storage, 1);
        Replica gpu(m, 0, T, size_t(2) << 30, true);
        double gl = gpu.run(b, o, denominator, true);
        auto gg = gpu.grad_values();
        // Relative L2 error per parameter kind; scalar Wscale gradients are
        // cancellation-heavy sums and get a looser bound.
        std::map<std::string, std::pair<double, double>> kinds;
        for (auto &p : m.params) {
            auto kind = p.name.substr(p.name.rfind('.') + 1);
            for (size_t i = p.offset; i < p.offset + p.size; ++i) {
                require(std::isfinite(gg[i]), "nonfinite GPU layer gradient");
                kinds[kind].first += double(gg[i] - cg[i]) * (gg[i] - cg[i]);
                kinds[kind].second += double(cg[i]) * cg[i];
            }
        }
        json report = {{"format", format}, {"storage", storage}, {"cpu_loss", cl / denominator}, {"gpu_loss", gl / denominator}};
        for (auto &[kind, e] : kinds) {
            double rel = std::sqrt(e.first / std::max(e.second, 1e-30));
            report[kind + "_rel_error"] = rel;
            require(rel < (kind == "Wscale" ? .06 : .02), "GPU layer " + kind + " gradient differs from CPU");
        }
        std::cout << report.dump() << '\n';
        require(std::abs(cl - gl) < 2e-3 * std::abs(cl), "GPU layer loss differs from CPU");
        if (std::string(storage) == "fp32") {
            fp32_grads = gg;
            fp32_loss = gl;
        } else {
            double ss = 0, ref = 0;
            for (size_t i = 0; i < gg.size(); ++i) {
                ss += double(gg[i] - fp32_grads[i]) * (gg[i] - fp32_grads[i]);
                ref += double(fp32_grads[i]) * fp32_grads[i];
            }
            std::cout << json({{"bf16_vs_fp32_gradient_rel_error", std::sqrt(ss / ref)},
                               {"bf16_vs_fp32_loss_diff", (gl - fp32_loss) / denominator}})
                             .dump()
                      << '\n';
        }
    }
    if (saved)
        setenv("QTIP_TRAIN_STORAGE", restore.c_str(), 1);
    else
        unsetenv("QTIP_TRAIN_STORAGE");
}
} // namespace
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
        {
            auto [target, gpu_target] = pair(T, 1024, .7);
            auto gradient = cpu->alloc(T, 1024), gpu_gradient = gpu->alloc(T, 1024);
            std::vector<int32_t> segments(T, 0);
            segments.back() = -1;
            double reference = cpu->mse(q, target, segments, gradient);
            double actual = gpu->mse(gq, gpu_target, segments, gpu_gradient);
            require(std::abs(reference - actual) < 1e-7, "GPU reconstruction MSE mismatch");
            compare(gradient, gpu_gradient, .002);
        }
        auto ca = cpu->attention(q, k, v, c, b), ga = gpu->attention(gq, gk, gv, c, b);
        compare(ca.out, ga.out, .012);
        compare(ca.lse, ga.lse, .001);
        auto cb = cpu->attention_backward(ca, dy, c, b), gb = gpu->attention_backward(ga, gdy, c, b);
        for (int i = 0; i < 3; ++i)
            compare(cb[i], gb[i], .04);
        // Exercise both ownership directions, partial tiles, GQA reductions,
        // document boundaries inside key tiles, and nontrivial softmax scores.
        for (int length : {1, 17, 33, 63, 64, 65, 127, 128, 129, 257}) {
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
                        // Include one-token docs and boundaries inside the
                        // 16-key wave tile and the 32/64/128-key block tiles.
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
        for (int format : {0, 1, 2}) layer_check(format);
        std::cout << "native GPU primitives, packed GQA gradients, QTIP, mixed and FP8 teacher layers passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
