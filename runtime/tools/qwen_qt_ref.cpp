// Full CPU forward of a QTIP-quantized Qwen3.5 checkpoint (dense or MoE): every quantized linear is
// evaluated with qtip::reference_forward (trellis decode + double-precision matvec, the same math the
// GPU runtime implements), unquantized tensors (norms, conv, A_log/dt_bias, router, b/a, shared
// expert gate) run as bf16 host GEMMs. This is the reference for GPU-runtime parity: it measures the
// QUANTIZED model, so runtime-vs-this isolates runtime bugs from quantization error.
// usage: qwen_qt_ref <model_dir> [tokens_file] [T] [logits_out.bin] [x] [--ppl]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "behemoth/qtip.hpp"
#include "behemoth/qwen_model.hpp"

using namespace behemoth;
using json = nlohmann::json;

namespace {

float b16(uint16_t h) { return float(BFloat16::from_bits(h)); }
float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
float softplus_f(float x) { return x > 20.0f ? x : std::log1pf(std::exp(x)); }

struct QtLayer {
    bool full = false, moe = false;
    qtip::HostLinear lq, lk, lv, lo;          // full attention
    qtip::HostLinear l_in_qkv, l_in_z, l_out;  // linear attention
    qtip::HostLinear lgate, lup, ldown;        // dense MLP
    std::vector<qtip::HostLinear> e_gate, e_up, e_down;
    qtip::HostLinear sh_gate, sh_up, sh_down;
    std::vector<float> w_router, w_router_bias, w_sg;  // bf16-valued (exact bf16 values in f32)
    std::vector<float> q_norm, k_norm, conv_w, dn_norm, a_log, dt_bias;
    std::vector<float> w_ba;  // [2nv][H]
    std::vector<float> ln_in, ln_post;
};

float dot(const std::vector<float>& a, const std::vector<float>& b, size_t off) {
    double s = 0;
    for (size_t i = 0; i < b.size(); ++i) s += double(a[off + i]) * b[i];
    return float(s);
}

static std::vector<float> tovec(const FloatTensorView& v) {
    std::vector<float> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = v[i];
    return r;
}

// y[T][m] = X[T][n] . W[m][n]^T
void mm(std::vector<float>& y, const std::vector<float>& X, const std::vector<float>& W, int T, int m,
        int n) {
    y.assign(size_t(T) * m, 0);
#pragma omp parallel for schedule(static) collapse(2)
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < m; ++r) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += double(X[size_t(t) * n + i]) * W[size_t(r) * n + i];
            y[size_t(t) * m + r] = float(s);
        }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qwen_qt_ref <model_dir> [tokens_file] [T] [logits_out.bin] [x] [--ppl]\n");
        return 1;
    }
    std::string dir = argv[1];
    QwenRuntimeConfig c = QwenRuntimeConfig::read(dir);
    const int H = c.hidden, hd = c.hd, nq = c.nq, nkv = c.nkv, F = c.inter;
    const int kd = c.lin_kd, vd = c.lin_vd, nk = c.lin_nk, nv = c.lin_nv;
    const int key_dim = c.key_dim(), V = c.value_dim(), conv_dim = c.conv_dim();
    std::printf("model: %d layers (%d full), H %d, %s\n", c.layers,
                int(std::count(c.full_attn.begin(), c.full_attn.end(), 1)), H,
                c.num_experts > 0 ? ("moe E" + std::to_string(c.num_experts)).c_str() : "dense");
    auto layer_file = [&](int L) {
        char b[64];
        std::snprintf(b, sizeof b, "model-layer-%05d.safetensors", L);
        return std::filesystem::path(dir) / b;
    };
    auto vec16 = [&](const SafeTensorFile& f, const std::string& n) { return tovec(f.floats(n)); };
    auto vec_opt = [&](const SafeTensorFile& f, const std::string& n) {
        try {
            return tovec(f.floats(n));
        } catch (...) {
            return std::vector<float>{};
        }
    };
    // ---- layers
    std::vector<QtLayer> layers(size_t(c.layers));
    for (int L = 0; L < c.layers; ++L) {
        QtLayer& ql = layers[size_t(L)];
        ql.full = c.is_full(L);
        SafeTensorFile f(layer_file(L));
        std::string p = "model.layers." + std::to_string(L) + ".";
        // generic HostLinear loader: dims from SU/SV (qtip::load_linear hardcodes Behemoth dims)
        auto lin = [&](qtip::HostLinear& h, const std::string& name) {
            std::string b = p + name + ".";
            auto tre = f.bytes(b + "trellis");
            h.trellis.resize(tre.size() / 2);
            std::memcpy(h.trellis.data(), tre.data(), tre.size());
            h.su = tovec(f.floats(b + "SU"));
            h.sv = tovec(f.floats(b + "SV"));
            auto tl = f.bytes(b + "tlut");
            h.tlut.resize(tl.size() / 2);
            std::memcpy(h.tlut.data(), tl.data(), tl.size());
            h.in = int(h.su.size());
            h.out = int(h.sv.size());
        };
        if (ql.full) {
            lin(ql.lq, "self_attn.q_proj");
            lin(ql.lk, "self_attn.k_proj");
            lin(ql.lv, "self_attn.v_proj");
            lin(ql.lo, "self_attn.o_proj");
            ql.q_norm = vec16(f, p + "self_attn.q_norm.weight");
            ql.k_norm = vec16(f, p + "self_attn.k_norm.weight");
        } else {
            lin(ql.l_in_qkv, "linear_attn.in_proj_qkv");
            lin(ql.l_in_z, "linear_attn.in_proj_z");
            lin(ql.l_out, "linear_attn.out_proj");
            ql.conv_w = vec16(f, p + "linear_attn.conv1d.weight");
            ql.a_log = tovec(f.floats(p + "linear_attn.A_log"));
            ql.dt_bias = tovec(f.floats(p + "linear_attn.dt_bias"));
            ql.dn_norm = vec16(f, p + "linear_attn.norm.weight");
            std::vector<float> wb = vec16(f, p + "linear_attn.in_proj_b.weight");
            std::vector<float> wa = vec16(f, p + "linear_attn.in_proj_a.weight");
            ql.w_ba = wb;
            ql.w_ba.insert(ql.w_ba.end(), wa.begin(), wa.end());
        }
        ql.moe = c.num_experts > 0;
        if (ql.moe) {
            const int E = c.num_experts;
            ql.w_router = vec16(f, p + "mlp.gate.weight");
            ql.w_router_bias = vec_opt(f, p + "mlp.gate.e_score_correction_bias");
            ql.e_gate.resize(size_t(E));
            ql.e_up.resize(size_t(E));
            ql.e_down.resize(size_t(E));
            for (int e = 0; e < E; ++e) {
                lin(ql.e_gate[size_t(e)], "mlp.experts." + std::to_string(e) + ".gate_proj");
                lin(ql.e_up[size_t(e)], "mlp.experts." + std::to_string(e) + ".up_proj");
                lin(ql.e_down[size_t(e)], "mlp.experts." + std::to_string(e) + ".down_proj");
            }
            lin(ql.sh_gate, "mlp.shared_expert.gate_proj");
            lin(ql.sh_up, "mlp.shared_expert.up_proj");
            lin(ql.sh_down, "mlp.shared_expert.down_proj");
            if (c.shared_gate) ql.w_sg = vec16(f, p + "mlp.shared_expert_gate.weight");
        } else {
            lin(ql.lgate, "mlp.gate_proj");
            lin(ql.lup, "mlp.up_proj");
            lin(ql.ldown, "mlp.down_proj");
        }
        ql.ln_in = vec16(f, p + "input_layernorm.weight");
        ql.ln_post = vec16(f, p + "post_attention_layernorm.weight");
    }
    // ---- embed / final norm / lm_head
    SafeTensorFile ef(std::filesystem::path(dir) / "model-embed.safetensors");
    std::vector<float> emb = vec_opt(ef, "model.language_model.embed_tokens.weight");
    if (emb.empty()) emb = vec16(ef, "model.embed_tokens.weight");
    std::vector<float> final_norm = vec_opt(ef, "model.language_model.norm.weight");
    if (final_norm.empty()) final_norm = vec16(ef, "model.norm.weight");
    if (emb.empty() || final_norm.empty()) {
        std::fprintf(stderr, "missing embed/final norm in model-embed.safetensors (bad assemble?)\n");
        return 1;
    }
    std::vector<float> lmh;
    if (!c.tied) lmh = vec16(ef, "lm_head.weight");
    const std::vector<float>& head = c.tied ? emb : lmh;

    // ---- tokens
    std::vector<int32_t> toks;
    int T = 16;
    if (argc >= 3) {
        std::ifstream tf(argv[2], std::ios::binary | std::ios::ate);
        if (!tf) { std::fprintf(stderr, "cannot open tokens file\n"); return 1; }
        size_t sz = size_t(tf.tellg()) / 4;
        T = argc >= 4 ? std::atoi(argv[3]) : int(sz);
        toks.resize(size_t(T));
        tf.seekg(0);
        tf.read(reinterpret_cast<char*>(toks.data()), std::streamsize(size_t(T) * 4));
    } else {
        toks = {2288, 4062, 13771, 30135, 517, 33984, 511, 8411, 11658, 13, 2288, 4062, 13771, 30135, 517, 33984};
    }
    const bool do_ppl = argc >= 7 && std::string(argv[6]) == "--ppl";
    double nll = 0;
    long cnt = 0;

    auto run_prompt = [&](const int32_t* tokens, int tcnt, std::vector<float>* logits_out) {
        // residual stream
        std::vector<float> x(size_t(tcnt) * H), xn(size_t(tcnt) * H), y;
        for (int t = 0; t < tcnt; ++t)
            for (int i = 0; i < H; ++i)
                x[size_t(t) * H + i] = emb[size_t(tokens[t]) * H + i];
        auto zc_norm = [&](std::vector<float>& v, const std::vector<float>& w, int row) {
            double ss = 0;
            for (int i = 0; i < H; ++i) {
                float vv = v[size_t(row) * H + i];
                ss += double(vv) * vv;
            }
            float inv = float(1.0 / std::sqrt(ss / H + c.eps));
            for (int i = 0; i < H; ++i)
                v[size_t(row) * H + i] =
                    float(BFloat16(v[size_t(row) * H + i] * inv)) * (1.0f + w[size_t(i)]);
        };
        for (int L = 0; L < c.layers; ++L) {
            const QtLayer& ql = layers[size_t(L)];
            xn = x;
            for (int t = 0; t < tcnt; ++t) zc_norm(xn, ql.ln_in, t);
            if (ql.full) {
                const int Wq = nq * 2 * hd;
                std::vector<float> q = qtip::reference_forward(ql.lq, xn, true, true);
                std::vector<float> k = qtip::reference_forward(ql.lk, xn, true, true);
                std::vector<float> v = qtip::reference_forward(ql.lv, xn, true, true);
                const int gq = nq / nkv;
                const float qk = 1.0f / std::sqrt(float(hd));
#pragma omp parallel for collapse(2)
                for (int t = 0; t < tcnt; ++t)
                    for (int h = 0; h < nq + nkv; ++h) {
                        float* vec = h < nq ? q.data() + size_t(t) * Wq + size_t(h) * 2 * hd
                                            : k.data() + size_t(t) * nkv * hd + size_t(h - nq) * hd;
                        const std::vector<float>& w = h < nq ? ql.q_norm : ql.k_norm;
                        double ss = 0;
                        for (int d = 0; d < hd; ++d) ss += double(vec[d]) * vec[d];
                        float inv = float(1.0 / std::sqrt(ss / hd + c.eps));
                        for (int d = 0; d < hd; ++d)
                            vec[d] = float(BFloat16(vec[d] * inv)) * (1.0f + w[size_t(d)]);
                        for (int d = 0; d < c.rot_dim / 2; ++d) {
                            float inv_freq = std::pow(c.theta, -2.0f * float(d) / float(c.rot_dim));
                            float ang = float(t) * inv_freq;
                            float cc = std::cos(ang), ss2 = std::sin(ang);
                            float a = vec[d], b = vec[d + c.rot_dim / 2];
                            vec[d] = a * cc - b * ss2;
                            vec[d + c.rot_dim / 2] = b * cc + a * ss2;
                        }
                    }
                std::vector<float> out(size_t(tcnt) * nq * hd);
                for (int t = 0; t < tcnt; ++t)
                    for (int h = 0; h < nq; ++h) {
                        const float* qh = q.data() + size_t(t) * Wq + size_t(h) * 2 * hd;
                        const float* gate = qh + hd;
                        const int kh = h / gq;
                        float mx = -INFINITY;
                        std::vector<float> row(t + 1);
                        for (int i = 0; i <= t; ++i) {
                            float s = 0;
                            for (int d = 0; d < hd; ++d)
                                s += qh[d] * k[size_t(i) * nkv * hd + size_t(kh) * hd + d];
                            s *= qk;
                            row[size_t(i)] = s;
                            mx = std::max(mx, s);
                        }
                        float sum = 0;
                        for (int i = 0; i <= t; ++i) sum += row[size_t(i)] = std::exp(row[size_t(i)] - mx);
                        for (int d = 0; d < hd; ++d) {
                            float acc = 0;
                            for (int i = 0; i <= t; ++i)
                                acc += row[size_t(i)] / sum *
                                       v[size_t(i) * nkv * hd + size_t(kh) * hd + d];
                            out[size_t(t) * nq * hd + size_t(h) * hd + d] =
                                acc * sigmoid_f(gate[d]);
                        }
                    }
                y = qtip::reference_forward(ql.lo, out, true, true);
            } else {
                // Gated DeltaNet
                std::vector<float> qkv = qtip::reference_forward(ql.l_in_qkv, xn, true, true);
                std::vector<float> zz = qtip::reference_forward(ql.l_in_z, xn, true, true);
                std::vector<float> ba;
                mm(ba, xn, ql.w_ba, tcnt, 2 * nv, H);
                std::vector<float> ring(size_t(conv_dim) * c.conv_k, 0.0f);
                std::vector<float> S(size_t(nv) * kd * vd, 0.0f);
                std::vector<float> o(size_t(tcnt) * V, 0.0f);
                const int group = nv / nk;
                std::vector<float> kv_mem(vd), delt(vd), ov(vd);
                for (int t = 0; t < tcnt; ++t) {
                    const size_t row = size_t(t) * conv_dim;
                    for (int ch = 0; ch < conv_dim; ++ch) {
                        float* rg = ring.data() + size_t(ch) * c.conv_k;
                        for (int j = 0; j + 1 < c.conv_k; ++j) rg[j] = rg[j + 1];
                        rg[c.conv_k - 1] = qkv[row + ch];
                        float acc = 0;
                        for (int j = 0; j < c.conv_k; ++j) acc += rg[j] * ql.conv_w[size_t(ch) * c.conv_k + j];
                        qkv[row + ch] = silu_f(acc);
                    }
                    for (int hv = 0; hv < nv; ++hv) {
                        const int kh = hv / group;
                        float* qh = qkv.data() + row + size_t(kh) * kd;
                        float* kvv = qkv.data() + row + key_dim + size_t(kh) * kd;
                        const float* vh = qkv.data() + row + 2 * key_dim + size_t(hv) * vd;
                        double qs = 0, ks = 0;
                        for (int d = 0; d < kd; ++d) { qs += double(qh[d]) * qh[d]; ks += double(kvv[d]) * kvv[d]; }
                        float iq = float(1.0 / std::sqrt(qs + 1e-6)) * (1.0f / std::sqrt(float(kd)));
                        float ik = float(1.0 / std::sqrt(ks + 1e-6));
                        for (int d = 0; d < kd; ++d) { qh[d] *= iq; kvv[d] *= ik; }
                        const float beta = sigmoid_f(ba[size_t(t) * 2 * nv + hv]);
                        const float g = -std::exp(ql.a_log[size_t(hv)]) *
                                        softplus_f(ba[size_t(t) * 2 * nv + nv + hv] + ql.dt_bias[size_t(hv)]);
                        const float decay = std::exp(g);
                        float* Sh = S.data() + size_t(hv) * kd * vd;
                        for (int d = 0; d < vd; ++d) kv_mem[d] = 0;
                        for (int dk = 0; dk < kd; ++dk) {
                            const float k_dk = kvv[dk];
                            float* srow = Sh + size_t(dk) * vd;
                            for (int dv = 0; dv < vd; ++dv) {
                                srow[dv] *= decay;
                                kv_mem[dv] += srow[dv] * k_dk;
                            }
                        }
                        for (int dv = 0; dv < vd; ++dv) {
                            delt[dv] = (vh[dv] - kv_mem[dv]) * beta;
                            ov[dv] = 0;
                        }
                        for (int dk = 0; dk < kd; ++dk) {
                            const float k_dk = kvv[dk], q_dk = qh[dk];
                            float* srow = Sh + size_t(dk) * vd;
                            for (int dv = 0; dv < vd; ++dv) {
                                const float s2 = srow[dv] + k_dk * delt[dv];
                                srow[dv] = s2;
                                ov[dv] += s2 * q_dk;
                            }
                        }
                        double ss = 0;
                        for (int dv = 0; dv < vd; ++dv) ss += double(ov[dv]) * ov[dv];
                        const float inv = float(1.0 / std::sqrt(ss / vd + c.eps));
                        for (int dv = 0; dv < vd; ++dv)
                            o[size_t(t) * V + size_t(hv) * vd + dv] =
                                ov[dv] * inv * ql.dn_norm[size_t(dv)] *
                                silu_f(zz[size_t(t) * V + size_t(hv) * vd + dv]);
                    }
                }
                y = qtip::reference_forward(ql.l_out, o, true, true);
            }
            for (size_t i = 0; i < x.size(); ++i) x[i] = float(BFloat16(x[i] + float(BFloat16(y[i]))));
            // ---- MLP
            xn = x;
            for (int t = 0; t < tcnt; ++t) zc_norm(xn, ql.ln_post, t);
            if (!ql.moe) {
                std::vector<float> g = qtip::reference_forward(ql.lgate, xn, true, true);
                std::vector<float> u = qtip::reference_forward(ql.lup, xn, true, true);
                std::vector<float> act(size_t(tcnt) * F);
                for (size_t i = 0; i < act.size(); ++i) act[i] = silu_f(g[i]) * u[i];
                y = qtip::reference_forward(ql.ldown, act, true, true);
            } else {
                const int E = c.num_experts, K = c.topk;
                std::vector<float> rlog(E);
                y.assign(size_t(tcnt) * H, 0.0f);
                for (int t = 0; t < tcnt; ++t) {
                    const std::vector<float> xt(xn.begin() + size_t(t) * H,
                                                xn.begin() + size_t(t + 1) * H);
                    for (int e = 0; e < E; ++e) rlog[size_t(e)] = dot(xt, ql.w_router, size_t(e) * H);
                    float mx = -INFINITY;
                    for (int e = 0; e < E; ++e) mx = std::max(mx, rlog[size_t(e)]);
                    float sum = 0;
                    for (int e = 0; e < E; ++e) sum += rlog[size_t(e)] = std::exp(rlog[size_t(e)] - mx);
                    for (int e = 0; e < E; ++e) rlog[size_t(e)] /= sum;
                    int ids[64];
                    float wts[64];
                    for (int j = 0; j < K; ++j) {
                        int best = -1;
                        for (int e = 0; e < E; ++e)
                            if (best < 0 || rlog[size_t(e)] > rlog[size_t(best)]) best = e;
                        ids[j] = best;
                        wts[j] = rlog[size_t(best)];
                        rlog[size_t(best)] = -1;
                    }
                    float wsum = 0;
                    for (int j = 0; j < K; ++j) wsum += wts[j];
                    std::vector<float> yr(H, 0.0f);
                    for (int j = 0; j < K; ++j) {
                        const float w = wts[j] / wsum;
                        const int e = ids[j];
                        std::vector<float> g = qtip::reference_forward(ql.e_gate[size_t(e)], xt, true, true);
                        std::vector<float> u = qtip::reference_forward(ql.e_up[size_t(e)], xt, true, true);
                        std::vector<float> act(c.moe_inter);
                        for (int i = 0; i < c.moe_inter; ++i) act[size_t(i)] = silu_f(g[size_t(i)]) * u[size_t(i)];
                        std::vector<float> d = qtip::reference_forward(ql.e_down[size_t(e)], act, true, true);
                        for (int i = 0; i < H; ++i) yr[size_t(i)] += w * d[size_t(i)];
                    }
                    if (c.shared_gate) {
                        float sg = sigmoid_f(dot(xt, ql.w_sg, 0));
                        std::vector<float> g = qtip::reference_forward(ql.sh_gate, xt, true, true);
                        std::vector<float> u = qtip::reference_forward(ql.sh_up, xt, true, true);
                        std::vector<float> act(c.shared_inter);
                        for (int i = 0; i < c.shared_inter; ++i)
                            act[size_t(i)] = silu_f(g[size_t(i)]) * u[size_t(i)];
                        std::vector<float> d = qtip::reference_forward(ql.sh_down, act, true, true);
                        for (int i = 0; i < H; ++i) yr[size_t(i)] += sg * d[size_t(i)];
                    }
                    for (int i = 0; i < H; ++i) y[size_t(t) * H + i] = yr[size_t(i)];
                }
            }
            for (size_t i = 0; i < x.size(); ++i) x[i] = float(BFloat16(x[i] + float(BFloat16(y[i]))));
        }
        // final norm + head
        std::vector<float> xf = x;
        for (int t = 0; t < tcnt; ++t) {
            double ss = 0;
            for (int i = 0; i < H; ++i) ss += double(xf[size_t(t) * H + i]) * xf[size_t(t) * H + i];
            float inv = float(1.0 / std::sqrt(ss / H + c.eps));
            for (int i = 0; i < H; ++i)
                xf[size_t(t) * H + i] =
                    float(BFloat16(xf[size_t(t) * H + i] * inv)) * (1.0f + final_norm[size_t(i)]);
        }
        std::vector<float> logits;
        mm(logits, xf, head, tcnt, c.vocab, H);
        if (logits_out) *logits_out = logits;
        return logits;
    };

    if (do_ppl) {
        // 512-token windows like qwen35_ref
        std::vector<int32_t> all = toks;
        if (int(all.size()) < 514) { std::fprintf(stderr, "ppl needs >= 514 tokens\n"); return 1; }
        std::vector<float> lg;
        for (size_t w = 0; w + 513 < all.size(); w += 512) {
            std::vector<int32_t> win(all.begin() + w, all.begin() + w + 513);
            run_prompt(win.data(), int(win.size()), &lg);
            for (int t = 0; t + 1 < int(win.size()); ++t) {
                const float* row = lg.data() + size_t(t) * c.vocab;
                float mx = -INFINITY;
                for (long v = 0; v < c.vocab; ++v) mx = std::max(mx, row[size_t(v)]);
                double lse = 0;
                for (long v = 0; v < c.vocab; ++v) lse += std::exp(double(row[size_t(v)]) - mx);
                lse = std::log(lse) + mx;
                nll += lse - row[size_t(win[size_t(t + 1)])];
                cnt++;
            }
            std::fprintf(stderr, "window %zu: %ld predictions\r", w / 512, cnt);
        }
        std::printf("\nppl over %ld predictions (512-token windows): %.4f\n", cnt, std::exp(nll / cnt));
    } else {
        std::vector<float> logits;
        double t0 = double(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
        run_prompt(toks.data(), T, &logits);
        std::printf("forward: %d tokens in %.1fs\n", T,
                    double(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()) - t0);
        for (int t = 0; t < std::min(T, 16); ++t) {
            int am = 0;
            for (long v = 1; v < c.vocab; ++v)
                if (logits[size_t(t) * c.vocab + v] > logits[size_t(t) * c.vocab + am]) am = int(v);
            std::printf("argmax[%d] = %d\n", t, am);
        }
        if (argc >= 5) {
            std::ofstream f(argv[4], std::ios::binary);
            f.write(reinterpret_cast<const char*>(logits.data()),
                    std::streamsize(logits.size() * 4));
        }
    }
    return 0;
}
