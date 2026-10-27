// Single-layer Qwen runtime check: the full GPU layer pipeline (input transform, trellis matmuls,
// delta step / gated attention, MLP epilogues) vs a CPU replica that uses qtip::reference_forward
// (device rounding) for every linear. usage: qwen_rt_tests <model_dir> layer
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "behemoth/qtip.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/qwen_kernels.hpp"
#include "behemoth/qwen_model.hpp"

using namespace behemoth;
using json = nlohmann::json;

namespace {
float b16(uint16_t b) { return float(BFloat16::from_bits(b)); }
uint16_t to16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
}  // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::fprintf(stderr, "usage: qwen_rt_tests <model_dir> layer\n");
        return 1;
    }
    std::filesystem::path dir = argv[1];
    int layer = std::atoi(argv[2]);
    QwenRuntimeConfig c = QwenRuntimeConfig::read(dir);
    const int H = c.hidden, F = c.inter, hd = c.hd, nq = c.nq, nkv = c.nkv, V = c.value_dim();
    const bool full = c.is_full(layer);
    std::string fn = "model-layer-" + std::string(layer < 10 ? "0000" : "000") + std::to_string(layer) +
                     ".safetensors";
    SafeTensorFile f(dir / fn);
    std::string p = "model.layers." + std::to_string(layer) + ".";
    auto load_lin = [&](const std::string &name) {
        std::string b = p + name + ".";
        qtip::HostLinear h;
        auto tre = f.bytes(b + "trellis");
        h.trellis.resize(tre.size() / 2);
        std::memcpy(h.trellis.data(), tre.data(), tre.size());
        auto su = f.floats(b + "SU"), sv = f.floats(b + "SV");
        h.su.resize(su.size());
        for (size_t i = 0; i < su.size(); ++i) h.su[i] = su[i];
        h.sv.resize(sv.size());
        for (size_t i = 0; i < sv.size(); ++i) h.sv[i] = sv[i];
        auto tl = f.bytes(b + "tlut");
        h.tlut.resize(tl.size() / 2);
        std::memcpy(h.tlut.data(), tl.data(), tl.size());
        h.in = int(h.su.size());
        h.out = int(h.sv.size());
        return h;
    };
    auto load16 = [&](const std::string &name) {
        auto v = f.floats(p + name);
        std::vector<uint16_t> r(v.size());
        for (size_t i = 0; i < v.size(); ++i) r[i] = to16(v[i]);
        return r;
    };
    // input: 8 rows from the embedding + noise
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    std::vector<uint16_t> x0(size_t(8) * H);
    if (std::getenv("RANDX")) {
        for (auto &v : x0) v = to16(0.5f * nd(rng));
    } else {
        SafeTensorFile ef(dir / "model-embed.safetensors");
        auto emb = ef.floats("model.language_model.embed_tokens.weight");
        for (int t = 0; t < 8; ++t)
            for (int i = 0; i < H; ++i)
                x0[size_t(t) * H + size_t(i)] = to16(emb[size_t((t + 100) % 1000) * H + size_t(i)] +
                                                     0.02f * nd(rng));
    }
    hipStream_t s;
    HIP_CHECK(hipStreamCreate(&s));
    const int rows = 8;

    // ---------------- GPU pipeline (mirrors QwenModel::forward_chunk for one layer)
    auto lin_gpu = [&](qtip::HostLinear &h, qtip::GpuLinear &gl, qtip::GpuCodebook &lut,
                       DeviceBuffer<float> &su, DeviceBuffer<float> &sv) {
        gl.upload({std::span<const uint16_t>(h.trellis)}, h.in);
        su.upload(h.su);
        sv.upload(h.sv);
        qtip::Codebook cb = qtip::make_codebook(std::span<const uint16_t>(h.tlut));
        lut.upload(cb);
    };
    auto input_t = [&](const uint16_t *x, const DeviceBuffer<uint16_t> *norm, const float *su, int n,
                       DeviceBuffer<uint16_t> &u) {
        q_input_transform(x, rows, n, norm ? norm->data() : nullptr, c.eps, su, n, u.data(), n, s);
    };
    auto matmul_z = [&](qtip::GpuLinear &gl, qtip::GpuCodebook &lut, DeviceBuffer<uint16_t> &u, int n,
                        DeviceBuffer<float> &z) {
        int slices = qtip::matmul_slices(gl.view(), rows);
        static DeviceBuffer<float> partial;
        partial.resize(size_t(slices) * rows * gl.rows);
        qtip::matmul(gl.view(), lut.lut.data(), reinterpret_cast<const BFloat16 *>(u.data()), n, rows,
                     partial.data(), slices, s);
        q_sum_slices(partial.data(), slices, size_t(rows) * gl.rows, z.data(), s);
    };
    DeviceBuffer<uint16_t> d_state, d_u, d_q, d_k, d_v, d_g, d_ao, d_o, d_gu;
    DeviceBuffer<float> d_z, d_z2;
    d_state.upload(x0);
    d_u.resize(size_t(rows) * std::max({H, V, F, nq * hd}));
    d_z.resize(size_t(rows) * std::max({2 * F, H, V}));
    d_z2.resize(size_t(rows) * std::max({2 * F, H, V}));
    d_q.resize(size_t(rows) * nq * 2 * hd);
    d_k.resize(size_t(rows) * nkv * hd);
    d_v.resize(size_t(rows) * nkv * hd);
    d_g.resize(size_t(rows) * nq * hd);
    d_ao.resize(size_t(rows) * nq * hd);
    d_o.resize(size_t(rows) * V);
    d_gu.resize(size_t(rows) * F);
    auto ln_in_h = load16("input_layernorm.weight"), ln_post_h = load16("post_attention_layernorm.weight");
    DeviceBuffer<uint16_t> ln_in, ln_post;
    ln_in.upload(ln_in_h);
    ln_post.upload(ln_post_h);
    std::vector<float> y_cpu;  // computed alongside

    if (full) {
        auto hq = load_lin("self_attn.q_proj"), hk = load_lin("self_attn.k_proj"),
             hv = load_lin("self_attn.v_proj"), ho = load_lin("self_attn.o_proj");
        qtip::GpuLinear gq, gk, gv, go;
        qtip::GpuCodebook lq;
        DeviceBuffer<float> suq, svq, svk, svv, svo, suk, suv;
        lin_gpu(hq, gq, lq, suq, svq);
        lin_gpu(hk, gk, lq, suk, svk);
        lin_gpu(hv, gv, lq, suv, svv);
        DeviceBuffer<float> su_ho;
        lin_gpu(ho, go, lq, su_ho, svo);
        DeviceBuffer<float> su_o;
        su_o.upload(ho.su);
        {
            // rt_tests-style control: random input through the same kernel call
            std::vector<uint16_t> rx(size_t(rows) * H);
            std::vector<float> rxf(size_t(rows) * H);
            for (size_t i = 0; i < rx.size(); ++i) {
                float v = 0.5f * nd(rng);
                rxf[i] = float(BFloat16::from_bits(to16(v)));
                rx[i] = to16(v);
            }
            DeviceBuffer<uint16_t> rd_x, rd_u;
            rd_x.upload(rx);
            rd_u.resize(size_t(rows) * H);
            q_input_transform(rd_x.data(), rows, H, nullptr, 0, suq.data(), H, rd_u.data(), H, s);
            HIP_CHECK(hipStreamSynchronize(s));
            auto ug2 = rd_u.download();
            double du2 = 0;
            for (int r = 0; r < rows; ++r) {
                std::vector<double> t2(H);
                for (int i = 0; i < H; ++i) t2[size_t(i)] = double(rxf[size_t(r) * H + i]) * hq.su[size_t(i)];
                qtip::hadamard(t2.data(), H, true);
                for (int i = 0; i < H; ++i)
                    du2 = std::max(du2, double(std::fabs(double(b16(ug2[size_t(r) * H + i])) - t2[size_t(i)])));
            }
            std::printf("control (random, no norm) u diff: %.5f\n", du2);
            {
                std::vector<double> g(H);
                for (int i = 0; i < H; ++i) g[size_t(i)] = double(b16(ug2[i]));
                qtip::hadamard(g.data(), H, false);  // inverse (same FWHT)
                double dinv = 0;
                for (int i = 0; i < H; ++i)
                    dinv = std::max(dinv, double(std::fabs(g[size_t(i)] - double(rxf[size_t(i)]) * hq.su[size_t(i)])));
                std::printf("ctl inverse-hadamard residual: %.5f (su-consistent if ~0)\n", dinv);
                int worst = 0;
                for (int i = 1; i < H; ++i)
                    if (std::fabs(g[size_t(i)] - double(rxf[size_t(i)]) * hq.su[size_t(i)]) >
                        std::fabs(g[size_t(worst)] - double(rxf[size_t(worst)]) * hq.su[size_t(worst)]))
                        worst = i;
                std::printf("worst i=%d inv %.4f xsu %.4f | first 16 mismatches:",
                            worst, g[size_t(worst)],
                            double(rxf[size_t(worst)]) * hq.su[size_t(worst)]);
                int shown = 0;
                for (int i = 0; i < H && shown < 16; ++i)
                    if (std::fabs(g[size_t(i)] - double(rxf[size_t(i)]) * hq.su[size_t(i)]) > 0.05) {
                        std::printf(" %d", i);
                        ++shown;
                    }
                std::printf("\n");
            }
            std::printf("ctl gpu[:6]: %.4f %.4f %.4f %.4f\n", b16(ug2[0]), b16(ug2[1]), b16(ug2[2]), b16(ug2[3]));
            {
                std::vector<double> t2(H);
                for (int i = 0; i < H; ++i) t2[size_t(i)] = double(rxf[i]) * hq.su[size_t(i)];
                qtip::hadamard(t2.data(), H, true);
                std::printf("ctl cpu[:6]: %.4f %.4f %.4f %.4f\n", t2[0], t2[1], t2[2], t2[3]);
            }
        }
        std::printf("stage: qkv input_transform\n");
        input_t(d_state.data(), std::getenv("NONORM") ? nullptr : &ln_in, suq.data(), H, d_u);
        std::printf("stage: q matmul\n");
        matmul_z(gq, lq, d_u, H, d_z);
        q_hadamard_rows(d_z.data(), rows, hq.out, hq.out, false, s);
        q_sv_cast(d_z.data(), svq.data(), d_q.data(), rows, hq.out, hq.out, s);
        std::printf("stage: k\n");
        matmul_z(gk, lq, d_u, H, d_z);
        q_hadamard_rows(d_z.data(), rows, hk.out, hk.out, false, s);
        q_sv_cast(d_z.data(), svk.data(), d_k.data(), rows, hk.out, hk.out, s);
        std::printf("stage: v\n");
        matmul_z(gv, lq, d_u, H, d_z);
        q_hadamard_rows(d_z.data(), rows, hv.out, hv.out, false, s);
        q_sv_cast(d_z.data(), svv.data(), d_v.data(), rows, hv.out, hv.out, s);
        DeviceBuffer<uint16_t> kc(size_t(64) * nkv * hd), vc(size_t(64) * nkv * hd);
        auto qn_h = load16("self_attn.q_norm.weight"), kn_h = load16("self_attn.k_norm.weight");
        DeviceBuffer<uint16_t> qn, kn;
        qn.upload(qn_h);
        kn.upload(kn_h);
        std::printf("stage: rope\n");
        {
            // stepwise: rotated input u vs host double replica (qtip::hadamard)
            auto ug = d_u.download();
            std::vector<float> xs2(rows * H);
            for (size_t i = 0; i < xs2.size(); ++i) xs2[i] = b16(x0[i]);
            double du = 0;
            for (int r = 0; r < rows; ++r) {
                std::vector<double> t(H);
                double ss = 0;
                for (int i = 0; i < H; ++i) ss += double(xs2[size_t(r) * H + i]) * xs2[size_t(r) * H + i];
                float inv = float(1.0 / std::sqrt(ss / H + c.eps));
                for (int i = 0; i < H; ++i) {
                    float v;
                    if (std::getenv("NONORM")) v = xs2[size_t(r) * H + i];
                    else v = b16(to16(xs2[size_t(r) * H + i] * inv)) * (1.0f + b16(ln_in_h[size_t(i)]));
                    t[size_t(i)] = double(v * hq.su[size_t(i)]);
                }
                qtip::hadamard(t.data(), H, true);
                if (r == 0) {
                    std::vector<double> t2(H);
                    for (int i = 0; i < H; ++i)
                        t2[size_t(i)] = double(b16(to16(xs2[size_t(i)] * inv)) *
                                               (1.0f + b16(ln_in_h[size_t(i)])) * hq.su[size_t(i)]);
                    qtip::hadamard(t2.data(), H, false);
                    double df = 0;
                    for (int i = 0; i < H; ++i)
                        df = std::max(df, double(std::fabs(double(b16(ug[size_t(i)])) - t2[size_t(i)])));
                    std::printf("hadU variant diff row0: %.5f\n", df);
                }
                if (r == 0) {
                    std::printf("u gpu[:8]: ");
                    for (int i = 0; i < 8; ++i) std::printf("%.4f ", double(b16(ug[size_t(i)])));
                    std::printf("\nu cpu[:8]: ");
                    for (int i = 0; i < 8; ++i) std::printf("%.4f ", t[size_t(i)]);
                    std::printf("\nu gpu[1016:1024]: ");
                    for (int i = 1016; i < 1024; ++i) std::printf("%.4f ", double(b16(ug[size_t(i)])));
                    std::printf("\nu cpu[1016:1024]: ");
                    for (int i = 1016; i < 1024; ++i) std::printf("%.4f ", t[size_t(i)]);
                    std::printf("\n");
                }
                for (int i = 0; i < H; ++i)
                    du = std::max(du, double(std::fabs(double(b16(ug[size_t(r) * H + size_t(i)])) - t[size_t(i)])));
                if (r == 0) {
                    // no-hadamard hypothesis: u == bf16(x*inv*(1+w)*su)
                    double dh = 0;
                    for (int i = 0; i < H; ++i) {
                        float v = b16(to16(xs2[size_t(i)] * inv));
                        v *= (1.0f + b16(ln_in_h[size_t(i)])) * hq.su[size_t(i)];
                        dh = std::max(dh, double(std::fabs(double(b16(ug[size_t(i)])) - double(v))));
                    }
                    std::printf("no-hadamard hypothesis diff: %.5f\n", dh);
                }
            }
            std::printf("u diff %.5f\n", du);
        }
        std::printf("rope ptrs q=%p k=%p v=%p g=%p kc=%p vc=%p qn=%p kn=%p\n", (void *)d_q.data(),
                    (void *)d_k.data(), (void *)d_v.data(), (void *)d_g.data(), (void *)kc.data(),
                    (void *)vc.data(), (void *)qn.data(), (void *)kn.data());
        q_qk_norm_rope(d_q.data(), d_k.data(), d_v.data(), d_g.data(), kc.data(), vc.data(), rows, 0,
                       nq, nkv, hd, c.rot_dim, c.theta, qn.data(), kn.data(), c.eps, s);
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("rope synced\n");
        {
            auto qg = d_q.download();
            // compare with the CPU replica below is done later; print a sample of the GPU q
            std::printf("gpu q row0 head0 [:6]: %.4f %.4f %.4f %.4f %.4f %.4f\n", b16(qg[0]),
                        b16(qg[1]), b16(qg[2]), b16(qg[3]), b16(qg[4]), b16(qg[5]));
        }
        std::printf("stage: attn\n");
        q_attn(d_q.data(), kc.data(), vc.data(), d_g.data(), d_ao.data(), rows, 0, nq, nkv, hd,
               1.0f / std::sqrt(float(hd)), s);
        std::printf("stage: o input_transform\n");
        {
            // attention-only check: CPU attention consuming the GPU's own q/k/v/g
            auto qg = d_q.download(), kg = d_k.download(), vg = d_v.download(), gg = d_g.download();
            auto aog = d_ao.download();
            double da = 0, sa = 1e-9;
            for (int r = 0; r < rows; ++r)
                for (int h = 0; h < nq; ++h) {
                    int kh2 = h / (nq / nkv);
                    float mx = -INFINITY;
                    std::vector<float> sc(r + 1);
                    for (int i = 0; i <= r; ++i) {
                        float d2 = 0;
                        for (int d2i = 0; d2i < hd; ++d2i++)
                            d2 += b16(qg[size_t(r) * nq * 2 * hd + size_t(h) * 2 * hd + size_t(d2i)]) *
                                  b16(kg[size_t(i) * nkv * hd + size_t(kh2) * hd + size_t(d2i)]);
                        sc[size_t(i)] = d2 / std::sqrt(float(hd));
                        mx = std::max(mx, sc[size_t(i)]);
                    }
                    float sum = 0;
                    for (int i = 0; i <= r; ++i) sum += sc[size_t(i)] = std::exp(sc[size_t(i)] - mx);
                    for (int d2i = 0; d2i < hd; ++d2i) {
                        float acc = 0;
                        for (int i = 0; i <= r; ++i)
                            acc += sc[size_t(i)] / sum *
                                   b16(vg[size_t(i) * nkv * hd + size_t(kh2) * hd + size_t(d2i)]);
                        float g2 = b16(gg[size_t(r) * nq * hd + size_t(h) * hd + size_t(d2i)]);
                        float want = acc / (1.0f + std::exp(-g2));
                        da = std::max(da, double(std::fabs(double(b16(aog[size_t(r) * nq * hd + size_t(h) * hd + size_t(d2i)])) - want)));
                        sa = std::max(sa, double(std::fabs(want)));
                    }
                }
            std::printf("attn-only diff: %.5f scale %.3f rel %.2e\n", da, sa, da / sa);
        }
        input_t(d_ao.data(), nullptr, su_o.data(), nq * hd, d_u);
        std::printf("stage: o matmul\n");
        matmul_z(go, lq, d_u, nq * hd, d_z);
        q_hadamard_rows(d_z.data(), rows, H, H, false, s);
        q_sv_add(d_z.data(), svo.data(), d_state.data(), rows, H, H, H, s);

        // ---- CPU replica (reference_forward per linear)
        auto proj_ref = [](qtip::HostLinear &h, const std::vector<float> &x) {
            return qtip::reference_forward(h, x, true);
        };
        std::vector<float> xs(rows * H);
        for (size_t i = 0; i < xs.size(); ++i) xs[i] = b16(x0[i]);
        auto normed = [&](const std::vector<float> &x, const std::vector<uint16_t> &w) {
            std::vector<float> y(x.size());
            for (int r = 0; r < rows; ++r) {
                double ss = 0;
                for (int i = 0; i < H; ++i) ss += double(x[size_t(r) * H + i]) * x[size_t(r) * H + i];
                float inv = float(1.0 / std::sqrt(ss / H + c.eps));
                for (int i = 0; i < H; ++i)
                    y[size_t(r) * H + i] =
                        float(b16(to16(x[size_t(r) * H + i] * inv))) * (1.0f + b16(w[size_t(i)]));
            }
            return y;
        };
        std::vector<float> xn = normed(xs, ln_in_h);
        auto qf = proj_ref(hq, xn), kf = proj_ref(hk, xn), vf = proj_ref(hv, xn);
        // q/k norm + rope + attention on CPU (fp32; matches the kernels' semantics)
        auto rope = [&](float *v, int pos) {
            for (int j = 0; j < c.rot_dim / 2; ++j) {
                float fr = std::pow(c.theta, -2.0f * float(j) / float(c.rot_dim));
                float ang = float(pos) * fr, co = std::cos(ang), si = std::sin(ang);
                float a = v[j], b2 = v[j + c.rot_dim / 2];
                v[j] = a * co - b2 * si;
                v[j + c.rot_dim / 2] = b2 * co + a * si;
            }
        };
        std::vector<float> Q(rows * nq * hd), K(rows * nkv * hd), Vv = vf, G(rows * nq * hd);
        {
            auto qg = d_q.download(), kg = d_k.download(), gg = d_g.download();
            auto vf16 = d_v.download();
            double dq = 0, dk = 0, dv = 0, dg = 0;
            for (int r = 0; r < rows; ++r)
                for (int h = 0; h < nq; ++h)
                    for (int d = 0; d < hd; ++d)
                        dq = std::max(dq, double(std::fabs(b16(qg[size_t(r) * nq * 2 * hd + size_t(h) * 2 * hd + size_t(d)]) - Q[size_t(r) * nq * hd + size_t(h) * hd + size_t(d)])));
            for (size_t i = 0; i < kg.size(); ++i) dk = std::max(dk, double(std::fabs(b16(kg[i]) - K[i])));
            for (size_t i = 0; i < vf16.size(); ++i) dv = std::max(dv, double(std::fabs(b16(vf16[i]) - Vv[i])));
            for (size_t i = 0; i < gg.size(); ++i) dg = std::max(dg, double(std::fabs(b16(gg[i]) - G[i])));
            std::printf("pre-attn: dq %.4f dk %.4f dv %.4f dg %.4f\n", dq, dk, dv, dg);
            std::printf("gate gpu[:4]: %.4f %.4f | qbuf gate region [:4]: %.4f %.4f | cpu G[:4]: %.4f %.4f | ref qf gate[:4]: %.4f %.4f\n",
                        b16(gg[0]), b16(gg[1]), b16(qg[hd]), b16(qg[hd + 1]), G[0], G[1],
                        qf[hd], qf[hd + 1]);
        }
        for (int r = 0; r < rows; ++r) {
            for (int h = 0; h < nq; ++h) {
                float *qh = &qf[size_t(r) * nq * 2 * hd + size_t(h) * 2 * hd];
                float *gh = qh + hd;
                double ss = 0;
                for (int d = 0; d < hd; ++d) ss += double(qh[d]) * qh[d];
                float inv = float(1.0 / std::sqrt(ss / hd + c.eps));
                for (int d = 0; d < hd; ++d) {
                    Q[size_t(r) * nq * hd + size_t(h) * hd + d] =
                        b16(to16(qh[d] * inv)) * (1.0f + b16(qn_h[size_t(d)]));
                    G[size_t(r) * nq * hd + size_t(h) * hd + d] = gh[d];
                }
                rope(&Q[size_t(r) * nq * hd + size_t(h) * hd], r);
            }
            for (int h = 0; h < nkv; ++h) {
                float *kh = &kf[size_t(r) * nkv * hd + size_t(h) * hd];
                double ss = 0;
                for (int d = 0; d < hd; ++d) ss += double(kh[d]) * kh[d];
                float inv = float(1.0 / std::sqrt(ss / hd + c.eps));
                for (int d = 0; d < hd; ++d)
                    K[size_t(r) * nkv * hd + size_t(h) * hd + d] =
                        b16(to16(kh[d] * inv)) * (1.0f + b16(kn_h[size_t(d)]));
                rope(&K[size_t(r) * nkv * hd + size_t(h) * hd], r);
            }
        }
        std::vector<float> ao(rows * nq * hd);
        for (int r = 0; r < rows; ++r)
            for (int h = 0; h < nq; ++h) {
                int kh = h / (nq / nkv);
                float mx = -INFINITY;
                std::vector<float> sc(r + 1);
                for (int i = 0; i <= r; ++i) {
                    float d2 = 0;
                    for (int d = 0; d < hd; ++d)
                        d2 += Q[size_t(r) * nq * hd + size_t(h) * hd + d] *
                              K[size_t(i) * nkv * hd + size_t(kh) * hd + d];
                    sc[size_t(i)] = d2 / std::sqrt(float(hd));
                    mx = std::max(mx, sc[size_t(i)]);
                }
                float sum = 0;
                for (int i = 0; i <= r; ++i) sum += sc[size_t(i)] = std::exp(sc[size_t(i)] - mx);
                for (int d = 0; d < hd; ++d) {
                    float acc = 0;
                    for (int i = 0; i <= r; ++i)
                        acc += sc[size_t(i)] / sum * Vv[size_t(i) * nkv * hd + size_t(kh) * hd + d];
                    float g = G[size_t(r) * nq * hd + size_t(h) * hd + d];
                    ao[size_t(r) * nq * hd + size_t(h) * hd + d] =
                        acc / (1.0f + std::exp(-g));
                }
            }
        auto of = proj_ref(ho, ao);
        std::vector<float> y1(rows * H);
        for (size_t i = 0; i < y1.size(); ++i) y1[i] = xs[i] + b16(to16(of[i]));
        y_cpu = y1;
    } else {
        // ---------------- linear (Gated DeltaNet) layer
        auto hq = load_lin("linear_attn.in_proj_qkv"), hz = load_lin("linear_attn.in_proj_z"),
             ho = load_lin("linear_attn.out_proj");
        qtip::GpuLinear gq, gz, go;
        qtip::GpuCodebook lq;
        DeviceBuffer<float> suq, svq, svz, svo, su_o;
        lin_gpu(hq, gq, lq, suq, svq);
        lin_gpu(hz, gz, lq, suq /*same su: group shared*/, svz);
        DeviceBuffer<float> su_z;
        su_z.upload(hz.su);
        lin_gpu(ho, go, lq, su_o, svo);
        const int Wq = c.conv_dim(), nv = c.lin_nv, kd = c.lin_kd, vd = c.lin_vd;
        const int Wz = Wq + V + 2 * nv;
        d_z.resize(size_t(rows) * Wz);
        d_z2.resize(size_t(rows) * Wz);
        DeviceBuffer<float> zbig(size_t(rows) * Wz);
        auto proj_scatter = [&](qtip::GpuLinear &gl, qtip::GpuCodebook &lut, DeviceBuffer<uint16_t> &uu,
                                int n, DeviceBuffer<float> &sv, long col) {
            int slices = qtip::matmul_slices(gl.view(), rows);
            static DeviceBuffer<float> part;
            part.resize(size_t(slices) * rows * gl.rows);
            qtip::matmul(gl.view(), lut.lut.data(), reinterpret_cast<const BFloat16 *>(uu.data()), n,
                         rows, part.data(), slices, s);
            q_sum_slices(part.data(), slices, size_t(rows) * gl.rows, d_z2.data(), s);
            q_hadamard_rows(d_z2.data(), rows, gl.rows, gl.rows, false, s);
            q_sv_cast_f32(d_z2.data(), sv.data(), zbig.data() + col, rows, gl.rows, gl.rows, Wz, s);
        };
        input_t(d_state.data(), &ln_in, suq.data(), H, d_u);
        {
            auto ug3 = d_u.download();
            std::vector<float> xs3(rows * H);
            for (size_t i = 0; i < xs3.size(); ++i) xs3[i] = b16(x0[i]);
            double du3 = 0;
            for (int r = 0; r < rows; ++r) {
                std::vector<double> t3(H);
                double ss3 = 0;
                for (int i = 0; i < H; ++i)
                    ss3 += double(xs3[size_t(r) * H + i]) * xs3[size_t(r) * H + i];
                float inv3 = float(1.0 / std::sqrt(ss3 / H + c.eps));
                for (int i = 0; i < H; ++i)
                    t3[size_t(i)] = double(b16(to16(xs3[size_t(r) * H + i] * inv3)) *
                                           (1.0f + b16(ln_in_h[size_t(i)])) * hq.su[size_t(i)]);
                qtip::hadamard(t3.data(), H, true);
                for (int i = 0; i < H; ++i)
                    du3 = std::max(du3, double(std::fabs(double(b16(ug3[size_t(r) * H + i])) - t3[i])));
            }
            std::printf("delta-branch u diff: %.5f\n", du3);
        }
        proj_scatter(gq, lq, d_u, H, svq, 0);
        proj_scatter(gz, lq, d_u, H, svz, Wq);
        // b/a: raw bf16 gemms on the plain normed hidden state (mirror QwenModel)
        {
            auto bt = f.bytes(p + "linear_attn.in_proj_b.weight");
            auto at = f.bytes(p + "linear_attn.in_proj_a.weight");
            std::vector<uint16_t> ba(bt.size() / 2 + at.size() / 2);
            std::memcpy(ba.data(), bt.data(), bt.size());
            std::memcpy(ba.data() + bt.size() / 2, at.data(), at.size());
            DeviceBuffer<uint16_t> d_ba;
            d_ba.upload(ba);
            DeviceBuffer<uint16_t> h_norm;
            h_norm.upload(x0);
            q_zc_norm(h_norm.data(), rows, H, ln_in.data(), c.eps, s);
            qtip::DenseMatmul dense;
            dense.gemm(nv, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows,
                       zbig.data() + Wq + V, Wz, reinterpret_cast<const BFloat16 *>(d_ba.data()), s);
            dense.gemm(nv, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows,
                       zbig.data() + Wq + V + nv, Wz,
                       reinterpret_cast<const BFloat16 *>(d_ba.data()) + size_t(nv) * H, s);
        }
        auto convw = load16("linear_attn.conv1d.weight");
        auto dnn = load16("linear_attn.norm.weight");
        DeviceBuffer<uint16_t> d_conv, d_dn;
        d_conv.upload(convw);
        d_dn.upload(dnn);
        DeviceBuffer<float> ring(size_t(Wq) * c.conv_k), S(size_t(nv) * kd * vd), a_dt(2 * size_t(nv));
        {
            auto al = f.floats(p + "linear_attn.A_log");
            auto db = f.floats(p + "linear_attn.dt_bias");
            std::vector<float> aux(2 * size_t(nv));
            for (size_t i = 0; i < al.size(); ++i) aux[i] = al[i];
            for (size_t i = 0; i < db.size(); ++i) aux[nv + i] = db[i];
            a_dt.upload(aux);
        }
        {
            auto zb = zbig.download();
            std::printf("zbig t0: qkv[:3] %.4f %.4f | z[:3] %.4f %.4f | b[:2] %.4f %.4f | a[:2] %.4f %.4f\n",
                        zb[0], zb[1], zb[Wq], zb[Wq + 1], zb[Wq + V], zb[Wq + V + 1],
                        zb[Wq + V + nv], zb[Wq + V + nv + 1]);
        }
        HIP_CHECK(hipMemsetAsync(ring.data(), 0, ring.bytes(), s));
        HIP_CHECK(hipMemsetAsync(S.data(), 0, S.bytes(), s));
        for (int t = 0; t < rows; ++t)
            q_delta_step(zbig.data(), d_conv.data(), d_dn.data(), a_dt.data(), ring.data(), S.data(),
                         d_o.data(), rows, t, Wq, V, kd, vd, c.lin_nk, nv, c.conv_k, c.eps, s);
        HIP_CHECK(hipStreamSynchronize(s));
        {
            auto og = d_o.download();
            std::printf("delta out gpu[:6]: %.4f %.4f %.4f %.4f\n", b16(og[0]), b16(og[1]), b16(og[2]),
                        b16(og[3]));
        }
        input_t(d_o.data(), nullptr, su_o.data(), V, d_u);
        int slices = qtip::matmul_slices(go.view(), rows);
        static DeviceBuffer<float> part2;
        part2.resize(size_t(slices) * rows * go.rows);
        qtip::matmul(go.view(), lq.lut.data(), reinterpret_cast<const BFloat16 *>(d_u.data()), V, rows,
                     part2.data(), slices, s);
        q_sum_slices(part2.data(), slices, size_t(rows) * go.rows, d_z.data(), s);
        q_hadamard_rows(d_z.data(), rows, H, H, false, s);
        q_sv_add(d_z.data(), svo.data(), d_state.data(), rows, H, H, H, s);
        HIP_CHECK(hipStreamSynchronize(s));

        // ---- CPU replica
        auto lin_ref = [](qtip::HostLinear &h, const std::vector<float> &x) {
            return qtip::reference_forward(h, x, true);
        };
        std::vector<float> xs(rows * H);
        for (size_t i = 0; i < xs.size(); ++i) xs[i] = b16(x0[i]);
        std::vector<float> xn(rows * H);
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int i = 0; i < H; ++i) ss += double(xs[size_t(r) * H + i]) * xs[size_t(r) * H + i];
            float inv = float(1.0 / std::sqrt(ss / H + c.eps));
            for (int i = 0; i < H; ++i)
                xn[size_t(r) * H + i] =
                    b16(to16(xs[size_t(r) * H + i] * inv)) * (1.0f + b16(ln_in_h[size_t(i)]));
        }
        auto qkvf = lin_ref(hq, xn), zf = lin_ref(hz, xn);
        // b/a
        auto b16vec = [&](const std::string &nm) {
            auto v = f.floats(p + nm);
            std::vector<float> r(v.size());
            for (size_t i = 0; i < v.size(); ++i) r[i] = v[i];
            return r;
        };
        auto bw = b16vec("linear_attn.in_proj_b.weight"), aw = b16vec("linear_attn.in_proj_a.weight");
        std::vector<float> y(rows * H, 0.0f);
        // shared-sequence CPU replica (faithful port of the HF-validated qtip_quant CPU reference)
        {
            std::vector<float> qkv(Wq), zz(V), ba(2 * size_t(nv));
            std::vector<float> ringv(size_t(Wq) * c.conv_k, 0.0f);
            std::vector<float> Sv(size_t(nv) * kd * vd, 0.0f);
            auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
            auto sp = [](float x) { return x > 20.0f ? x : std::log(1.0f + std::exp(x)); };
            auto alog = f.floats(p + "linear_attn.A_log");
            auto dtb = f.floats(p + "linear_attn.dt_bias");
            std::vector<float> out(rows * V);
            for (int t = 0; t < rows; ++t) {
                for (int i = 0; i < Wq; ++i) qkv[size_t(i)] = qkvf[size_t(t) * Wq + i];
                for (int i = 0; i < V; ++i) zz[size_t(i)] = zf[size_t(t) * V + i];
                for (int e = 0; e < 2 * nv; ++e) {
                    const std::vector<float> &w = e < nv ? bw : aw;
                    float acc = 0;
                    for (int i = 0; i < H; ++i)
                        acc += xn[size_t(t) * H + size_t(i)] * w[size_t(e % nv) * H + size_t(i)];
                    ba[size_t(e)] = acc;
                }
                for (int ch = 0; ch < Wq; ++ch) {
                    float *rg = ringv.data() + size_t(ch) * c.conv_k;
                    for (int j = 0; j + 1 < c.conv_k; ++j) rg[j] = rg[j + 1];
                    rg[c.conv_k - 1] = qkv[size_t(ch)];
                    float acc = 0;
                    const uint16_t *cw2 = convw.data() + size_t(ch) * c.conv_k;
                    for (int j = 0; j < c.conv_k; ++j) acc += rg[j] * b16(cw2[size_t(j)]);
                    qkv[size_t(ch)] = acc / (1.0f + std::exp(-acc));
                }
                for (int hv = 0; hv < nv; ++hv) {
                    const int kh2 = hv / (nv / c.lin_nk);
                    float *qh = qkv.data() + size_t(kh2) * kd;
                    float *kh_ = qkv.data() + c.key_dim() + size_t(kh2) * kd;
                    const float *vh = qkv.data() + 2 * c.key_dim() + size_t(hv) * vd;
                    float qs = 0, ks = 0;
                    for (int d2 = 0; d2 < kd; ++d2) {
                        qs += qh[d2] * qh[d2];
                        ks += kh_[d2] * kh_[d2];
                    }
                    float iq = 1.0f / std::sqrt(qs + 1e-6f) / std::sqrt(float(kd));
                    float ik = 1.0f / std::sqrt(ks + 1e-6f);
                    for (int d2 = 0; d2 < kd; ++d2) {
                        qh[d2] *= iq;
                        kh_[d2] *= ik;
                    }
                    const float beta = sig(ba[size_t(hv)]);
                    const float g = -std::exp(alog[size_t(hv)]) * sp(ba[size_t(nv + hv)] + dtb[size_t(hv)]);
                    const float decay = std::exp(g);
                    float *Sh = Sv.data() + size_t(hv) * kd * vd;
                    float kvm[256], dl[256], ov[256];
                    for (int dv2 = 0; dv2 < vd; ++dv2) kvm[dv2] = 0;
                    for (int dk = 0; dk < kd; ++dk) {
                        const float k_dk = kh_[dk];
                        float *srow = Sh + size_t(dk) * vd;
                        for (int dv2 = 0; dv2 < vd; ++dv2) {
                            srow[dv2] *= decay;
                            kvm[dv2] += srow[dv2] * k_dk;
                        }
                    }
                    for (int dv2 = 0; dv2 < vd; ++dv2) {
                        dl[dv2] = (vh[dv2] - kvm[dv2]) * beta;
                        ov[dv2] = 0;
                    }
                    for (int dk = 0; dk < kd; ++dk) {
                        const float k_dk = kh_[dk], q_dk = qh[dk];
                        float *srow = Sh + size_t(dk) * vd;
                        for (int dv2 = 0; dv2 < vd; ++dv2) {
                            const float sv2 = srow[dv2] + k_dk * dl[dv2];
                            srow[dv2] = sv2;
                            ov[dv2] += sv2 * q_dk;
                        }
                    }
                    float ss2 = 0;
                    for (int dv2 = 0; dv2 < vd; ++dv2) ss2 += ov[dv2] * ov[dv2];
                    const float oinv = 1.0f / std::sqrt(ss2 / vd + c.eps);
                    if (t < 1 && hv < 2)
                        std::printf("t%d hv%d: |ov| %.5g oinv %.3g beta %.3f decay %.4f dl0 %.4f vh0 %.4f k0 %.4f q0 %.4f\n",
                                    t, hv, std::sqrt(ss2), oinv, beta, decay, dl[0], vh[0], kh_[0], qh[0]);
                    const uint16_t *nw = dnn.data();
                    for (int dv2 = 0; dv2 < vd; ++dv2) {
                        float z = zz[size_t(hv) * vd + dv2];
                        out[size_t(t) * V + size_t(hv) * vd + dv2] =
                            ov[dv2] * oinv * b16(nw[size_t(dv2)]) * (z / (1.0f + std::exp(-z)));
                    }
                }
            }
            std::printf("delta out cpu[:6]: %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
            auto og = d_o.download();
            double ddo = 0, sc = 1e-9;
            for (size_t i = 0; i < out.size(); ++i) {
                ddo = std::max(ddo, double(std::fabs(double(b16(og[i])) - out[i])));
                sc = std::max(sc, double(std::fabs(out[i])));
            }
            std::printf("delta out diff: %.5f scale %.3f rel %.2e\n", ddo, sc, ddo / sc);
            auto of2 = lin_ref(ho, out);
            for (size_t i = 0; i < y.size(); ++i) y[i] = xs[i] + b16(to16(of2[i]));
        }
        auto gpu_state2 = d_state.download();
        double maxd2 = 0, scale2 = 1e-9;
        for (size_t i = 0; i < y.size(); ++i) {
            double g2 = b16(gpu_state2[i]);
            maxd2 = std::max(maxd2, std::fabs(g2 - y[i]));
            scale2 = std::max(scale2, double(std::fabs(y[i])));
        }
        std::printf("layer %d (lin) post-attn: max|diff| %.5f scale %.3f rel %.2e -> %s\n", layer,
                    maxd2, scale2, maxd2 / scale2, maxd2 / scale2 < 1.5e-1 ? "ok" : "FAIL");
        return maxd2 / scale2 < 1.5e-1 ? 0 : 1;
    }

    auto gpu_state = d_state.download();
    double maxd = 0, scale = 1e-9;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        double g = b16(gpu_state[i]);
        maxd = std::max(maxd, std::fabs(g - y_cpu[i]));
        scale = std::max(scale, double(std::fabs(y_cpu[i])));
    }
    std::printf("layer %d (%s) post-attn: max|diff| %.5f scale %.3f rel %.2e -> %s\n", layer,
                full ? "full" : "lin", maxd, scale, maxd / scale, maxd / scale < 5e-2 ? "ok" : "FAIL");
    return maxd / scale < 5e-2 ? 0 : 1;
}
