// Native BitNet trainer tests: packing round trip, BitLinear forward/backward against an exact CPU
// reference, and the full tiny-model forward (dense teacher and ternary SubLN student) against a
// double-precision CPU reference through the HF loader. Exit 77 skips without a CUDA device.
#include "behemoth/ternary.hpp"
#include "bitnet.hpp"
#include "kernels.hpp"
#include "safetensors.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

using namespace bitnet;
namespace tern = behemoth::ternary;
namespace {
int failures = 0;
void expect(bool ok, const std::string &what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    } else
        std::cerr << "ok: " << what << "\n";
}
float bf16_round(float x) { return float(behemoth::BFloat16(x)); }
float from_bits(uint16_t bits) {
    behemoth::BFloat16 b;
    b.bits = bits;
    return float(b);
}
template <class T> std::vector<T> dl(const T *p, size_t n) {
    std::vector<T> h(n);
    gpu::check(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
template <class T> void ul(T *p, const std::vector<T> &h) {
    gpu::check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
double rel_l2(const std::vector<float> &a, const std::vector<double> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i)
        num += (a[i] - b[i]) * (a[i] - b[i]), den += b[i] * b[i];
    return std::sqrt(num / std::max(den, 1e-30));
}

void pack_test(std::mt19937 &rng) {
    const size_t out = 5, in = 48;
    std::vector<int8_t> q(out * in);
    std::uniform_int_distribution<int> d(-1, 1);
    for (auto &v : q)
        v = int8_t(d(rng));
    std::vector<uint8_t> packed(out * tern::packed_bytes(in));
    tern::pack(q, out, in, packed);
    expect(tern::unpack(packed, out, in) == q, "pack/unpack round trip");
    // dp4a layout: (word >> 2j) & 0x03030303 holds elements 4j..4j+3 of the word as bytes
    bool layout = true;
    for (size_t r = 0; r < out && layout; ++r)
        for (size_t wd = 0; wd < in / 16; ++wd) {
            uint32_t word;
            std::memcpy(&word, packed.data() + r * in / 4 + wd * 4, 4);
            for (unsigned j = 0; j < 4; ++j) {
                const uint32_t bytes = (word >> (2 * j)) & 0x03030303u;
                for (unsigned b = 0; b < 4; ++b)
                    if (int((bytes >> (8 * b)) & 0xffu) - 1 != q[r * in + wd * 16 + 4 * j + b])
                        layout = false;
            }
        }
    expect(layout, "packed word layout matches dp4a extraction");
    std::vector<int8_t> x(in);
    std::uniform_int_distribution<int> dx(-127, 127);
    for (auto &v : x)
        v = int8_t(dx(rng));
    int32_t ref = 0;
    for (size_t i = 0; i < in; ++i)
        ref += q[i] * x[i];
    expect(tern::dot_row(packed.data(), x.data(), in) == ref, "packed row dot product");
}

void bitlinear_test(std::mt19937 &rng) {
    const int out = 40, in = 64, T = 7;
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> w(size_t(out) * in), x(size_t(T) * in), dy(size_t(T) * out);
    for (auto &v : w)
        v = nd(rng);
    for (auto &v : x)
        v = nd(rng) * 20;
    for (auto &v : dy)
        v = nd(rng);
    cudaStream_t s = cudaStreamPerThread;
    Geometry g{in, 16, 1, 1, 16, 16, 1, 1e-5f, 1e4f, true};
    Model m(g, true, true, true);
    Linear L;
    L.out = out, L.in = in, L.offset = 0;
    L.wq.reset(size_t(out) * in), L.dscale.reset(1);
    Dev<float> dw(size_t(out) * in), dxd(size_t(T) * in), dyd(size_t(T) * out), yd(size_t(T) * out), xs(T),
        xd(size_t(T) * in), gradw(size_t(out) * in), scratch(1);
    Dev<uint16_t> xq(size_t(T) * in), dyb(size_t(T) * out), dyb2(size_t(T) * out);
    ul(dw.p, w), ul(xd.p, x), ul(dyd.p, dy);
    gpu::ternarize(dw.p, w.size(), L.wq.p, L.dscale.p, scratch.p, s);
    gpu::check(cudaStreamSynchronize(s), "sync");
    L.scale = dl(L.dscale.p, 1)[0];
    // weight codes against the host quantiser at the device scale
    auto wq = dl(L.wq.p, w.size());
    std::vector<int8_t> q(w.size());
    bool codes = true;
    for (size_t i = 0; i < w.size(); ++i) {
        q[i] = tern::ternarize(w[i], L.scale);
        if (from_bits(wq[i]) != float(q[i]))
            codes = false;
    }
    expect(codes, "device ternary codes match host quantiser");
    expect(std::fabs(L.scale - tern::absmean(w)) < 1e-6f * tern::absmean(w) + 1e-9f, "absmean scale");
    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetStream(h, s);
    gpu::act_quant(xd.p, xq.p, xs.p, T, in, true, s);
    gemm(h, false, true, T, out, in, xq.p, in, L.wq.p, in, yd.p, out);
    gpu::scale_rows(yd.p, xs.p, L.scale, yd.p, nullptr, T, out, s);
    gpu::check(cudaStreamSynchronize(s), "sync");
    auto y = dl(yd.p, size_t(T) * out);
    auto xs_h = dl(xs.p, T);
    auto xq_h = dl(xq.p, size_t(T) * in);
    std::vector<int8_t> xqi(size_t(T) * in);
    std::vector<float> xs_ref(T);
    bool acts = true;
    for (int t = 0; t < T; ++t) {
        xs_ref[t] = tern::quantize_activation(std::span<const float>(x).subspan(size_t(t) * in, in),
                                              std::span<int8_t>(xqi).subspan(size_t(t) * in, in));
        if (std::fabs(xs_ref[t] - xs_h[t]) > 1e-6f * xs_ref[t])
            acts = false;
        for (int i = 0; i < in; ++i)
            if (from_bits(xq_h[size_t(t) * in + i]) != float(xqi[size_t(t) * in + i]))
                acts = false;
    }
    expect(acts, "activation quantisation matches host");
    std::vector<uint8_t> packed(size_t(out) * in / 4);
    tern::pack(q, size_t(out), size_t(in), packed);
    double worst = 0;
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < out; ++o) {
            const int32_t acc = tern::dot_row(packed.data() + size_t(o) * in / 4, xqi.data() + size_t(t) * in, in);
            const double ref = double(acc) * L.scale * xs_ref[t];
            worst = std::max(worst, std::fabs(ref - y[size_t(t) * out + o]) / (std::fabs(ref) + 1e-3));
        }
    expect(worst < 1e-5, "BitLinear forward equals exact int32 reference (max rel err " + std::to_string(worst) + ")");
    // backward: dx = scale * dy W~ ; dW = (dy * xs)^T xq
    gpu::check(cudaMemset(gradw.p, 0, gradw.bytes()), "memset");
    gpu::to_bf16(dyd.p, dyb.p, dy.size(), s);
    gemm(h, false, false, T, in, out, dyb.p, out, L.wq.p, in, dxd.p, in, L.scale, 0.f);
    gpu::scale_rows(dyd.p, xs.p, 1.f, nullptr, dyb2.p, T, out, s);
    gemm(h, true, false, out, in, T, dyb2.p, out, xq.p, in, gradw.p, in, 1.f, 1.f);
    gpu::check(cudaStreamSynchronize(s), "sync");
    auto dx = dl(dxd.p, size_t(T) * in);
    auto gw = dl(gradw.p, size_t(out) * in);
    std::vector<double> dx_ref(size_t(T) * in), gw_ref(size_t(out) * in);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < in; ++i) {
            double a = 0;
            for (int o = 0; o < out; ++o)
                a += double(bf16_round(dy[size_t(t) * out + o])) * q[size_t(o) * in + i];
            dx_ref[size_t(t) * in + i] = a * L.scale;
        }
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) {
            double a = 0;
            for (int t = 0; t < T; ++t)
                a += double(bf16_round(dy[size_t(t) * out + o] * xs_ref[t])) * xqi[size_t(t) * in + i];
            gw_ref[size_t(o) * in + i] = a;
        }
    expect(rel_l2(dx, dx_ref) < 1e-5, "BitLinear dX (STE) matches reference");
    expect(rel_l2(gw, gw_ref) < 1e-5, "BitLinear dW (STE) matches reference");
    cublasDestroy(h);
}

// CPU double-precision reference of the Llama forward with optional BitLinear/SubLN.
struct Ref {
    Geometry g;
    bool quant, subln, round = false; // round: BF16 boundaries like the trainer's default / the runtime
    std::map<std::string, std::vector<float>> w;
    void rnd(std::vector<double> &x) const {
        if (round)
            for (auto &v : x)
                v = bf16_round(float(v));
    }
    std::vector<double> linear(const std::string &name, const std::vector<double> &x, int T, int in, int out) {
        const auto &W = w.at(name);
        std::vector<double> y(size_t(T) * out);
        if (!quant) {
            for (int t = 0; t < T; ++t)
                for (int o = 0; o < out; ++o) {
                    double a = 0;
                    for (int i = 0; i < in; ++i)
                        a += double(bf16_round(W[size_t(o) * in + i])) * bf16_round(float(x[size_t(t) * in + i]));
                    y[size_t(t) * out + o] = a;
                }
            return y;
        }
        const float scale = tern::absmean(W);
        std::vector<int8_t> q(W.size());
        for (size_t i = 0; i < W.size(); ++i)
            q[i] = tern::ternarize(W[i], scale);
        for (int t = 0; t < T; ++t) {
            std::vector<float> xf(static_cast<size_t>(in));
            for (int i = 0; i < in; ++i)
                xf[size_t(i)] = float(x[size_t(t) * in + i]);
            std::vector<int8_t> xq(static_cast<size_t>(in));
            const float xs = tern::quantize_activation(xf, xq);
            for (int o = 0; o < out; ++o) {
                int32_t acc = 0;
                for (int i = 0; i < in; ++i)
                    acc += q[size_t(o) * in + i] * xq[size_t(i)];
                y[size_t(t) * out + o] = double(acc) * scale * xs;
            }
        }
        return y;
    }
    std::vector<double> rms(const std::vector<double> &x, const std::vector<float> &gain, int T, int n) {
        std::vector<double> y(x.size());
        for (int t = 0; t < T; ++t) {
            double ss = 0;
            for (int i = 0; i < n; ++i)
                ss += x[size_t(t) * n + i] * x[size_t(t) * n + i];
            const double r = 1 / std::sqrt(ss / n + g.eps);
            for (int i = 0; i < n; ++i)
                y[size_t(t) * n + i] = x[size_t(t) * n + i] * r * gain[size_t(i)];
        }
        return y;
    }
    std::vector<double> logits(const std::vector<int32_t> &tokens, const std::vector<int32_t> &starts) {
        const int T = int(tokens.size()), n = g.n, qd = g.heads * g.dim, kd = g.kv * g.dim, group = g.heads / g.kv;
        std::vector<double> h(size_t(T) * n);
        const auto &E = w.at("model.embed_tokens.weight");
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < n; ++i)
                h[size_t(t) * n + i] = bf16_round(E[size_t(tokens[size_t(t)]) * n + i]);
        for (int l = 0; l < g.layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            auto xn = rms(h, w.at(p + "input_layernorm.weight"), T, n);
            rnd(xn);
            auto q = linear(p + "self_attn.q_proj.weight", xn, T, n, qd);
            auto k = linear(p + "self_attn.k_proj.weight", xn, T, n, kd);
            auto v = linear(p + "self_attn.v_proj.weight", xn, T, n, kd);
            rnd(q), rnd(k), rnd(v);
            auto rope = [&](std::vector<double> &x, int heads) {
                for (int t = 0; t < T; ++t)
                    for (int hh = 0; hh < heads; ++hh)
                        for (int d = 0; d < g.dim / 2; ++d) {
                            const float inv = std::pow(g.theta, -2.f * float(d) / float(g.dim));
                            const float ang = float(t) * inv;
                            const double c = std::cos(ang), s = std::sin(ang);
                            double *vv = &x[(size_t(t) * heads + hh) * g.dim];
                            const double a = vv[d], b = vv[d + g.dim / 2];
                            vv[d] = a * c - b * s;
                            vv[d + g.dim / 2] = b * c + a * s;
                        }
            };
            rope(q, g.heads), rope(k, g.kv);
            std::vector<double> att(size_t(T) * qd);
            for (int hh = 0; hh < g.heads; ++hh) {
                const int gi = hh / group;
                for (int t = 0; t < T; ++t) {
                    std::vector<double> sc(size_t(T), -1e300);
                    double mx = -1e300;
                    for (int j = starts[size_t(t)]; j <= t; ++j) {
                        double a = 0;
                        for (int d = 0; d < g.dim; ++d)
                            a += bf16_round(float(q[(size_t(t) * g.heads + hh) * g.dim + d])) *
                                 bf16_round(float(k[(size_t(j) * g.kv + gi) * g.dim + d]));
                        sc[size_t(j)] = a / std::sqrt(double(g.dim));
                        mx = std::max(mx, sc[size_t(j)]);
                    }
                    double z = 0;
                    for (int j = starts[size_t(t)]; j <= t; ++j)
                        z += std::exp(sc[size_t(j)] - mx);
                    for (int d = 0; d < g.dim; ++d) {
                        double a = 0;
                        for (int j = starts[size_t(t)]; j <= t; ++j)
                            a += bf16_round(float(std::exp(sc[size_t(j)] - mx) / z)) *
                                 bf16_round(float(v[(size_t(j) * g.kv + gi) * g.dim + d]));
                        att[(size_t(t) * g.heads + hh) * g.dim + d] = a;
                    }
                }
            }
            rnd(att);
            if (subln) {
                att = rms(att, w.at(p + "self_attn.attn_sub_norm.weight"), T, qd);
                rnd(att);
            }
            auto o = linear(p + "self_attn.o_proj.weight", att, T, qd, n);
            rnd(o);
            for (size_t i = 0; i < h.size(); ++i)
                h[i] += o[i];
            rnd(h);
            auto xn2 = rms(h, w.at(p + "post_attention_layernorm.weight"), T, n);
            rnd(xn2);
            auto gate = linear(p + "mlp.gate_proj.weight", xn2, T, n, g.inter);
            auto up = linear(p + "mlp.up_proj.weight", xn2, T, n, g.inter);
            rnd(gate), rnd(up);
            std::vector<double> sact(gate.size());
            for (size_t i = 0; i < gate.size(); ++i)
                sact[i] = gate[i] / (1 + std::exp(-gate[i])) * up[i];
            rnd(sact);
            if (subln) {
                sact = rms(sact, w.at(p + "mlp.ffn_sub_norm.weight"), T, g.inter);
                rnd(sact);
            }
            auto d = linear(p + "mlp.down_proj.weight", sact, T, g.inter, n);
            rnd(d);
            for (size_t i = 0; i < h.size(); ++i)
                h[i] += d[i];
            rnd(h);
        }
        auto xf = rms(h, w.at("model.norm.weight"), T, n);
        std::vector<double> out(size_t(T) * g.vocab);
        for (int t = 0; t < T; ++t)
            for (int vv = 0; vv < g.vocab; ++vv) {
                double a = 0;
                for (int i = 0; i < n; ++i)
                    a += double(bf16_round(E[size_t(vv) * n + i])) * bf16_round(float(xf[size_t(t) * n + i]));
                out[size_t(t) * g.vocab + vv] = a;
            }
        return out;
    }
};

void model_test(std::mt19937 &rng) {
    const Geometry g{64, 96, 4, 2, 16, 128, 2, 1e-5f, 1e4f, true};
    const fs::path dir = fs::temp_directory_path() / ("bitnet-test-" + std::to_string(rng()));
    fs::create_directories(dir);
    std::map<std::string, std::vector<float>> w;
    std::normal_distribution<float> nd(0.f, 0.08f);
    auto tensor = [&](const std::string &name, size_t count, float scale, float bias = 0) {
        std::vector<float> v(count);
        for (auto &x : v)
            x = bias + nd(rng) * scale;
        w[name] = v;
    };
    tensor("model.embed_tokens.weight", size_t(g.vocab) * g.n, 1);
    tensor("model.norm.weight", size_t(g.n), 0.3f, 1);
    for (int l = 0; l < g.layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        tensor(p + "self_attn.q_proj.weight", size_t(g.heads * g.dim) * g.n, 1);
        tensor(p + "self_attn.k_proj.weight", size_t(g.kv * g.dim) * g.n, 1);
        tensor(p + "self_attn.v_proj.weight", size_t(g.kv * g.dim) * g.n, 1);
        tensor(p + "self_attn.o_proj.weight", size_t(g.n) * g.heads * g.dim, 1);
        tensor(p + "mlp.gate_proj.weight", size_t(g.inter) * g.n, 1);
        tensor(p + "mlp.up_proj.weight", size_t(g.inter) * g.n, 1);
        tensor(p + "mlp.down_proj.weight", size_t(g.n) * g.inter, 1);
        tensor(p + "input_layernorm.weight", size_t(g.n), 0.3f, 1);
        tensor(p + "post_attention_layernorm.weight", size_t(g.n), 0.3f, 1);
        w[p + "self_attn.attn_sub_norm.weight"] = std::vector<float>(size_t(g.heads * g.dim), 1.f);
        w[p + "mlp.ffn_sub_norm.weight"] = std::vector<float>(size_t(g.inter), 1.f);
    }
    qtip::SafeTensorsWriter writer;
    for (auto &[name, v] : w) {
        if (name.find("sub_norm") != std::string::npos)
            continue;
        std::vector<uint16_t> b(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            b[i] = behemoth::BFloat16(v[i]).bits;
            v[i] = from_bits(b[i]); // the reference sees the stored bf16 values
        }
        std::vector<int64_t> shape;
        if (name.find("norm") != std::string::npos)
            shape = {int64_t(v.size())};
        else if (name.starts_with("model.embed"))
            shape = {g.vocab, g.n};
        else
            shape = {1, int64_t(v.size())}; // only numel is checked by the loader
        writer.add(name, "BF16", shape, b.data(), b.size() * 2);
    }
    writer.write((dir / "model.safetensors").string());
    {
        std::ofstream f(dir / "config.json");
        f << json({{"hidden_size", g.n}, {"intermediate_size", g.inter}, {"num_attention_heads", g.heads},
                   {"num_key_value_heads", g.kv}, {"vocab_size", g.vocab}, {"num_hidden_layers", g.layers},
                   {"rms_norm_eps", g.eps}, {"rope_theta", g.theta}, {"tie_word_embeddings", true},
                   {"model_type", "llama"}})
                 .dump();
    }
    qtip::SafeTensorsReader reader(dir.string());
    cudaStream_t s = cudaStreamPerThread;
    const int T = 24;
    qat::Batch b;
    std::uniform_int_distribution<int> tok(0, g.vocab - 1);
    for (int t = 0; t < T; ++t) {
        b.tokens.push_back(tok(rng));
        b.segments.push_back(t < 10 ? 0 : 1);
        b.positions.push_back(t < 10 ? t : t - 10);
        b.labels.push_back(t + 1 < T && b.segments.back() == (t + 1 < 10 ? 0 : 1) ? 0 : -100);
    }
    b.boundaries();
    for (int t = 0; t < T; ++t)
        if (b.labels[size_t(t)] == 0)
            b.labels[size_t(t)] = b.tokens[size_t(t) + 1];
    // reference positions restart at segment boundaries: emulate by rotating with the actual positions
    Options o;
    o.loss_rows = 8;
    for (int variant = 0; variant < 4; ++variant) {
        const bool quant = variant % 2 == 1, round = variant >= 2;
        Model m(g, quant, quant, quant);
        m.round = round;
        m.load_hf(reader, s);
        gpu::check(cudaStreamSynchronize(s), "sync");
        Trainer tr(m, nullptr, o, s);
        Batch db = upload_batch({b}, s);
        std::vector<float> logits;
        auto rep = tr.evaluate(m, db, nullptr, &logits);
        gpu::check(cudaStreamSynchronize(s), "sync");
        Ref ref{g, quant, quant, round, w};
        // The reference rotates with angle t. Rows 0..9 are the first document (positions == t);
        // the second document restarts positions, so it is compared against an isolated reference run,
        // which also checks that the packed attention mask isolates the documents.
        std::vector<int32_t> starts(b.starts.begin(), b.starts.end());
        auto want = ref.logits(b.tokens, starts);
        std::vector<float> got(logits.begin(), logits.begin() + 10 * g.vocab);
        std::vector<double> ref10(want.begin(), want.begin() + 10 * g.vocab);
        const double err = rel_l2(got, ref10);
        const std::string what = std::string(quant ? "ternary SubLN" : "dense") + (round ? " (bf16 boundaries)" : "");
        expect(err < 2e-3, what + " forward matches CPU reference (rel L2 " + std::to_string(err) + ")");
        std::vector<int32_t> t2(b.tokens.begin() + 10, b.tokens.end()), s2(size_t(T - 10), 0);
        auto want2 = ref.logits(t2, s2);
        std::vector<float> got2(logits.begin() + 10 * g.vocab, logits.end());
        const double err2 = rel_l2(got2, want2);
        expect(err2 < 2e-3, what + " packed second document matches isolated reference (rel L2 " +
                                std::to_string(err2) + ")");
        // CE consistency: mean CE from the kernel equals the reference's CE on supervised rows
        double ce = 0;
        int count = 0;
        for (int t = 0; t < T; ++t) {
            if (b.labels[size_t(t)] < 0)
                continue;
            const double *row = (t < 10 ? want.data() : want2.data()) + size_t(t < 10 ? t : t - 10) * g.vocab;
            double mx = -1e300;
            for (int vv = 0; vv < g.vocab; ++vv)
                mx = std::max(mx, row[vv]);
            double z = 0;
            for (int vv = 0; vv < g.vocab; ++vv)
                z += std::exp(row[vv] - mx);
            ce += mx + std::log(z) - row[b.labels[size_t(t)]];
            ++count;
        }
        expect(count == int(rep.count) && std::fabs(ce / count - rep.ce / rep.count) < 2e-3,
               "cross-entropy matches reference (" + std::to_string(ce / count) + " vs " +
                   std::to_string(rep.ce / rep.count) + ")");
    }
    fs::remove_all(dir);
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "no CUDA device: skipping\n";
        return 77;
    }
    try {
        std::mt19937 rng(12345);
        pack_test(rng);
        bitlinear_test(rng);
        model_test(rng);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << (failures ? "FAILED" : "all tests passed") << "\n";
    return failures ? 1 : 0;
}
