// Qwen3.5 single-GPU QTIP runtime model. Loads the qtip_quant output (trellis linears + bf16 small
// tensors), runs hybrid Gated DeltaNet / gated full attention layers with a bf16 KV cache and
// per-layer fp32 delta state. Correctness-first: every projection goes through the trellis GEMV/WMMA
// kernels (rotate -> matmul -> slice sum -> hadU -> SV), the lm_head is a bf16 GEMM on the embedding.
#include "behemoth/qwen_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "behemoth/checkpoint.hpp"
#include "behemoth/qtip.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/qwen_kernels.hpp"

namespace behemoth {
namespace {
using json = nlohmann::json;

json read_json(const std::filesystem::path &p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return json::parse(f);
}

std::vector<uint16_t> bf16_raw(const Checkpoint &ck, const std::string &name) {
    auto v = ck.bf16(name);
    std::vector<uint16_t> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        float x = float(v[i]);
        uint32_t u;
        std::memcpy(&u, &x, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        r[i] = uint16_t(u >> 16);
    }
    return r;
}
}  // namespace

QwenRuntimeConfig QwenRuntimeConfig::read(const std::filesystem::path &dir) {
    json j = read_json(dir / "config.json");
    const json &t = j.contains("text_config") ? j["text_config"] : j;
    std::string mt = j.value("model_type", std::string(""));
    if (mt != "qwen3_5" && mt != "qwen3_5_moe" && t.value("model_type", "") != "qwen3_5_text")
        throw std::runtime_error("not a qwen3_5 checkpoint: " + dir.string());
    QwenRuntimeConfig c;
    c.hidden = t.at("hidden_size");
    c.layers = t.at("num_hidden_layers");
    c.vocab = t.at("vocab_size");
    c.eps = t.value("rms_norm_eps", 1e-6f);
    c.nq = t.at("num_attention_heads");
    c.nkv = t.at("num_key_value_heads");
    c.hd = t.value("head_dim", c.hidden / c.nq);
    c.attn_output_gate = t.value("attn_output_gate", true);
    c.lin_nk = t.at("linear_num_key_heads");
    c.lin_kd = t.at("linear_key_head_dim");
    c.lin_nv = t.at("linear_num_value_heads");
    c.lin_vd = t.at("linear_value_head_dim");
    c.conv_k = t.value("linear_conv_kernel_dim", 4);
    c.tied = t.value("tie_word_embeddings", false);
    float partial = 0.25f, theta = 1e7f;
    if (t.contains("rope_parameters")) {
        partial = t["rope_parameters"].value("partial_rotary_factor", partial);
        theta = t["rope_parameters"].value("rope_theta", theta);
    }
    partial = t.value("partial_rotary_factor", partial);
    theta = t.value("rope_theta", theta);
    c.theta = theta;
    c.rot_dim = int(float(c.hd) * partial) & ~1;
    c.num_experts = t.value("num_experts", t.value("n_routed_experts", 0));
    c.inter = t.value("intermediate_size", t.value("moe_intermediate_size", 0));
    c.full_attn.assign(size_t(c.layers), 0);
    if (t.contains("layer_types") && !t["layer_types"].is_null()) {
        for (int i = 0; i < c.layers; ++i)
            c.full_attn[size_t(i)] = (t["layer_types"][size_t(i)] == "full_attention") ? 1 : 0;
    } else {
        int interval = t.value("full_attention_interval", 4);
        for (int i = 0; i < c.layers; ++i) c.full_attn[size_t(i)] = ((i + 1) % interval == 0) ? 1 : 0;
    }
    return c;
}

// ------------------------------------------------------------------ impl
struct QwenModel::Impl {
    struct Linear {
        qtip::GpuLinear lin;
        DeviceBuffer<float> sv;  // [out]
    };
    struct LayerW {
        bool full = false;
        Linear lq, lk, lv, lo, lgate, lup, ldown;   // full attention + dense mlp
        Linear l_in_qkv, l_in_z, l_in_b, l_in_a, l_out;  // linear attention
        DeviceBuffer<uint16_t> ln_in, ln_post;      // raw zc weights (kernels add 1)
        DeviceBuffer<uint16_t> q_norm, k_norm;      // full attention (raw zc)
        DeviceBuffer<uint16_t> conv_w, dn_norm;     // linear attention
        DeviceBuffer<float> a_dt;                   // [2nv] = A_log | dt_bias
        DeviceBuffer<float> su0, su1, su2, su3;     // shared SU per Hessian group
        DeviceBuffer<float> ring, dstate;           // per-layer delta state (f32)
        DeviceBuffer<uint16_t> kc, vc;              // KV cache [maxctx][nkv][hd] bf16
    };
    QwenRuntimeConfig cfg;
    int device = 0, chunk = 256;
    hipStream_t stream{};
    qtip::DenseMatmul dense;
    qtip::GpuCodebook codebook;
    std::vector<LayerW> layers;
    DeviceBuffer<uint16_t> embed, final_norm;
    // scratch (sized for chunk rows)
    DeviceBuffer<uint16_t> state, u, q, k, v, gate, attn_out, o_in, gu;
    DeviceBuffer<float> partial, z, z2, logits_d;
    DeviceBuffer<int> tokens_d;
    std::vector<float> logits_h, logits_all;
    int pos = 0;
    size_t max_slices = 0;

    void load(const std::filesystem::path &dir);
    qtip::HostLinear load_host(const Checkpoint &ck, int layer, const std::string &name, int in, int out);
    void upload_linear(Linear &L, const qtip::HostLinear &h);
    void forward_chunk(const int *tokens_h, int rows);
    void lm_head_rows(int row0, int nrows);  // logits_all[row0..] from state rows
    void reset();
    void prefill(const int *tokens, size_t n, float *last_logits);
    void decode(int token, float *logits);
    std::vector<int> generate(const int *prompt, size_t n_prompt, size_t n_new);
    double perplexity(const int *tokens, size_t n);
};

qtip::HostLinear QwenModel::Impl::load_host(const Checkpoint &ck, int layer, const std::string &name,
                                            int in, int out) {
    const std::string base = "model.layers." + std::to_string(layer) + "." + name + ".";
    ck.tensor(base + "trellis");  // existence + index check
    const SafeTensorFile &f = ck.tensor(base + "trellis");
    qtip::HostLinear l;
    l.in = in;
    l.out = out;
    auto raw = f.bytes(base + "trellis");
    l.trellis.resize(raw.size() / 2);
    std::memcpy(l.trellis.data(), raw.data(), raw.size());
    auto floats = [&](const std::string &n, size_t size) {
        auto v = f.floats(n);
        if (v.size() != size) throw std::runtime_error("invalid QTIP vector " + n);
        return std::vector<float>(v.begin(), v.end());
    };
    l.su = floats(base + "SU", size_t(in));
    l.sv = floats(base + "SV", size_t(out));
    auto t = f.bytes(base + "tlut");
    l.tlut.resize(1024);
    std::memcpy(l.tlut.data(), t.data(), t.size());
    return l;
}

void QwenModel::Impl::upload_linear(Linear &L, const qtip::HostLinear &h) {
    L.lin.upload({std::span<const uint16_t>(h.trellis)}, h.in);
    L.sv.upload(h.sv);
}

void QwenModel::Impl::load(const std::filesystem::path &dir) {
    Checkpoint ck(dir);
    cfg = QwenRuntimeConfig::read(dir);
    if (cfg.num_experts > 0)
        throw std::runtime_error("Qwen3.5 MoE checkpoints are not supported by this runtime yet");
    if (!cfg.tied)
        throw std::runtime_error("untied lm_head (separate lm_head.weight) not implemented yet");
    const bool f0 = cfg.is_full(0);
    qtip::HostLinear probe =
        load_host(ck, 0, f0 ? "self_attn.q_proj" : "linear_attn.in_proj_qkv", cfg.hidden,
                  f0 ? cfg.nq * 2 * cfg.hd : cfg.conv_dim());
    qtip::Codebook cb = qtip::make_codebook(std::span<const uint16_t>(probe.tlut));
    codebook.upload(cb);
    const int H = cfg.hidden, V = cfg.value_dim(), A = cfg.nq * cfg.hd;
    layers.resize(size_t(cfg.layers));
    for (int L = 0; L < cfg.layers; ++L) {
        LayerW &w = layers[size_t(L)];
        w.full = cfg.is_full(L);
        std::string p = "model.layers." + std::to_string(L) + ".";
        auto up16 = [&](DeviceBuffer<uint16_t> &d, const std::string &name) {
            d.upload(bf16_raw(ck, p + name));
        };
        auto up_su = [&](DeviceBuffer<float> &d, const qtip::HostLinear &h) { d.upload(h.su); };
        if (w.full) {
            auto hq = load_host(ck, L, "self_attn.q_proj", H, cfg.nq * 2 * cfg.hd);
            upload_linear(w.lq, hq);
            upload_linear(w.lk, load_host(ck, L, "self_attn.k_proj", H, cfg.nkv * cfg.hd));
            upload_linear(w.lv, load_host(ck, L, "self_attn.v_proj", H, cfg.nkv * cfg.hd));
            upload_linear(w.lo, load_host(ck, L, "self_attn.o_proj", A, H));
            up_su(w.su0, hq);
            auto ho = load_host(ck, L, "self_attn.o_proj", A, H);
            up_su(w.su1, ho);
            up16(w.q_norm, "self_attn.q_norm.weight");
            up16(w.k_norm, "self_attn.k_norm.weight");
            w.kc.resize(size_t(cfg.max_context) * cfg.nkv * cfg.hd);
            w.vc.resize(size_t(cfg.max_context) * cfg.nkv * cfg.hd);
        } else {
            auto hq = load_host(ck, L, "linear_attn.in_proj_qkv", H, cfg.conv_dim());
            upload_linear(w.l_in_qkv, hq);
            upload_linear(w.l_in_z, load_host(ck, L, "linear_attn.in_proj_z", H, V));
            upload_linear(w.l_in_b, load_host(ck, L, "linear_attn.in_proj_b", H, cfg.lin_nv));
            upload_linear(w.l_in_a, load_host(ck, L, "linear_attn.in_proj_a", H, cfg.lin_nv));
            upload_linear(w.l_out, load_host(ck, L, "linear_attn.out_proj", V, H));
            up_su(w.su0, hq);
            auto ho = load_host(ck, L, "linear_attn.out_proj", V, H);
            up_su(w.su1, ho);
            up16(w.conv_w, "linear_attn.conv1d.weight");
            up16(w.dn_norm, "linear_attn.norm.weight");
            {
                auto al = ck.tensor(p + "linear_attn.A_log").floats(p + "linear_attn.A_log");
                auto db = ck.tensor(p + "linear_attn.dt_bias").floats(p + "linear_attn.dt_bias");
                std::vector<float> aux(al.begin(), al.end());
                aux.insert(aux.end(), db.begin(), db.end());
                w.a_dt.upload(aux);
            }
            w.ring.resize(size_t(cfg.conv_dim()) * cfg.conv_k);
            w.dstate.resize(size_t(cfg.lin_nv) * cfg.lin_kd * cfg.lin_vd);
        }
        auto hg = load_host(ck, L, "mlp.gate_proj", H, cfg.inter);
        upload_linear(w.lgate, hg);
        upload_linear(w.lup, load_host(ck, L, "mlp.up_proj", H, cfg.inter));
        upload_linear(w.ldown, load_host(ck, L, "mlp.down_proj", cfg.inter, H));
        up_su(w.su2, hg);
        auto hd = load_host(ck, L, "mlp.down_proj", cfg.inter, H);
        up_su(w.su3, hd);
        up16(w.ln_in, "input_layernorm.weight");
        up16(w.ln_post, "post_attention_layernorm.weight");
    }
    embed.upload(bf16_raw(ck, "model.embed_tokens.weight"));
    final_norm.upload(bf16_raw(ck, "model.norm.weight"));
    // scratch
    const int R = chunk;
    const int Wz = std::max({cfg.nq * 2 * cfg.hd + 2 * cfg.nkv * cfg.hd,
                             cfg.conv_dim() + V + 2 * cfg.lin_nv, 2 * cfg.inter});
    state.resize(size_t(R) * H);
    u.resize(size_t(R) * std::max({H, V, A, cfg.inter}));
    q.resize(size_t(R) * cfg.nq * 2 * cfg.hd);
    k.resize(size_t(R) * cfg.nkv * cfg.hd);
    v.resize(size_t(R) * cfg.nkv * cfg.hd);
    gate.resize(size_t(R) * A);
    attn_out.resize(size_t(R) * A);
    o_in.resize(size_t(R) * V);
    gu.resize(size_t(R) * cfg.inter);
    z.resize(size_t(R) * Wz);
    z2.resize(size_t(R) * Wz);
    tokens_d.resize(size_t(R));
    for (auto &w : layers) {
        qtip::LinearView views[] = {w.lq.lin.view(),  w.lk.lin.view(),  w.lv.lin.view(),
                                    w.lo.lin.view(),  w.lgate.lin.view(), w.lup.lin.view(),
                                    w.ldown.lin.view(), w.l_in_qkv.lin.view(), w.l_in_z.lin.view(),
                                    w.l_in_b.lin.view(), w.l_in_a.lin.view(), w.l_out.lin.view()};
        for (auto gv : views)
            if (gv.rows) max_slices = std::max<size_t>(max_slices, qtip::matmul_slices(gv, R));
    }
    partial.resize(max_slices * size_t(R) * Wz);
    logits_all.resize(size_t(std::min(R, 16)) * cfg.vocab);
    logits_d.resize(size_t(std::min(R, 16)) * cfg.vocab);
    logits_h.resize(size_t(cfg.vocab));
}

// logits of state rows [row0, row0+nrows) -> logits_all (sub-batches of <= 16 rows)
void QwenModel::Impl::lm_head_rows(int row0, int nrows) {
    const int H = cfg.hidden;
    q_zc_norm(state.data() + size_t(row0) * H, nrows, H, final_norm.data(), cfg.eps, stream);
    for (int r0 = 0; r0 < nrows; r0 += 16) {
        int nr = std::min(16, nrows - r0);
        dense.gemm(cfg.vocab, H, state.data() + size_t(row0 + r0) * H, H, nr,
                   logits_d.data() + size_t(r0) * cfg.vocab, cfg.vocab, embed.data(), stream);
        HIP_CHECK(hipMemcpyAsync(logits_all.data() + size_t(r0) * cfg.vocab,
                                 logits_d.data() + size_t(r0) * cfg.vocab, size_t(nr) * cfg.vocab * 4,
                                 hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

void QwenModel::Impl::forward_chunk(const int *tokens_h, int rows) {
    const QwenRuntimeConfig &c = cfg;
    const int H = c.hidden, A = c.nq * c.hd, V = c.value_dim(), F = c.inter;
    const int pos0 = pos;
    HIP_CHECK(hipMemcpyAsync(tokens_d.data(), tokens_h, size_t(rows) * 4, hipMemcpyHostToDevice, stream));
    q_embed_gather(embed.data(), tokens_d.data(), state.data(), rows, H, stream);
    // one trellis projection: rotated input is already in u (row stride n); z <- sum slices (dense f32)
    auto proj = [&](const Linear &lin, int n, float *zdst) {
        int slices = qtip::matmul_slices(lin.lin.view(), rows);
        qtip::matmul(lin.lin.view(), codebook.lut.data(), u.data(), n, rows, partial.data(), slices,
                     stream);
        q_sum_slices(partial.data(), slices, size_t(rows) * lin.lin.rows, zdst, stream);
    };
    for (int L = 0; L < c.layers; ++L) {
        LayerW &w = layers[size_t(L)];
        q_input_transform(state.data(), rows, H, w.ln_in.data(), c.eps, w.su0.data(), H, u.data(), H,
                          stream);
        if (w.full) {
            const int Wq = c.nq * 2 * c.hd, Wk = c.nkv * c.hd;
            proj(w.lq, H, z.data());
            q_hadamard_rows(z.data(), rows, Wq, Wq, false, stream);
            q_sv_cast(z.data(), w.lq.sv.data(), q.data(), rows, Wq, Wq, stream);
            proj(w.lk, H, z.data());
            q_hadamard_rows(z.data(), rows, Wk, Wk, false, stream);
            q_sv_cast(z.data(), w.lk.sv.data(), k.data(), rows, Wk, Wk, stream);
            proj(w.lv, H, z.data());
            q_hadamard_rows(z.data(), rows, Wk, Wk, false, stream);
            q_sv_cast(z.data(), w.lv.sv.data(), v.data(), rows, Wk, Wk, stream);
            q_qk_norm_rope(q.data(), k.data(), v.data(), gate.data(), w.kc.data(), w.vc.data(), rows,
                           pos0, c.nq, c.nkv, c.hd, c.rot_dim, c.theta, w.q_norm.data(),
                           w.k_norm.data(), c.eps, stream);
            q_attn(q.data(), w.kc.data(), w.vc.data(), gate.data(), attn_out.data(), rows, pos0, c.nq,
                   c.nkv, c.hd, 1.0f / std::sqrt(float(c.hd)), stream);
            q_input_transform(attn_out.data(), rows, A, nullptr, 0, w.su1.data(), A, u.data(), A,
                              stream);
            proj(w.lo, A, z.data());
            q_hadamard_rows(z.data(), rows, H, H, false, stream);
            q_sv_add(z.data(), w.lo.sv.data(), state.data(), rows, H, H, H, stream);
        } else {
            const int Wq = c.conv_dim(), nv = c.lin_nv;
            const int Wz = Wq + V + 2 * nv;
            auto proj_scatter = [&](const Linear &lin, int n, long col_off) {
                proj(lin, n, z2.data());
                q_hadamard_rows(z2.data(), rows, lin.lin.rows, lin.lin.rows, false, stream);
                q_sv_cast_f32(z2.data(), lin.sv.data(), z.data() + col_off, rows, lin.lin.rows,
                              lin.lin.rows, Wz, stream);
            };
            proj_scatter(w.l_in_qkv, H, 0);
            proj_scatter(w.l_in_z, H, Wq);
            proj_scatter(w.l_in_b, H, Wq + V);
            proj_scatter(w.l_in_a, H, Wq + V + nv);
            for (int t = 0; t < rows; ++t)
                q_delta_step(z.data(), w.conv_w.data(), w.dn_norm.data(), w.a_dt.data(), w.ring.data(),
                             w.dstate.data(), o_in.data(), rows, t, Wq, V, c.lin_kd, c.lin_vd,
                             c.lin_nk, nv, c.conv_k, c.eps, stream);
            q_input_transform(o_in.data(), rows, V, nullptr, 0, w.su1.data(), V, u.data(), V, stream);
            proj(w.l_out, V, z.data());
            q_hadamard_rows(z.data(), rows, H, H, false, stream);
            q_sv_add(z.data(), w.l_out.sv.data(), state.data(), rows, H, H, H, stream);
        }
        q_input_transform(state.data(), rows, H, w.ln_post.data(), c.eps, w.su2.data(), H, u.data(), H,
                          stream);
        proj(w.lgate, H, z.data());
        q_hadamard_rows(z.data(), rows, F, F, false, stream);
        proj(w.lup, H, z2.data());
        q_hadamard_rows(z2.data(), rows, F, F, false, stream);
        q_gateup(z.data(), z2.data(), w.lgate.sv.data(), w.lup.sv.data(), gu.data(), rows, F, F,
                 stream);
        q_input_transform(gu.data(), rows, F, nullptr, 0, w.su3.data(), F, u.data(), F, stream);
        proj(w.ldown, F, z.data());
        q_hadamard_rows(z.data(), rows, H, H, false, stream);
        q_sv_add(z.data(), w.ldown.sv.data(), state.data(), rows, H, H, H, stream);
    }
    pos += rows;
}

void QwenModel::Impl::reset() {
    pos = 0;
    for (auto &w : layers) {
        if (w.ring.size()) HIP_CHECK(hipMemsetAsync(w.ring.data(), 0, w.ring.bytes(), stream));
        if (w.dstate.size()) HIP_CHECK(hipMemsetAsync(w.dstate.data(), 0, w.dstate.bytes(), stream));
        if (w.kc.size()) {
            HIP_CHECK(hipMemsetAsync(w.kc.data(), 0, w.kc.bytes(), stream));
            HIP_CHECK(hipMemsetAsync(w.vc.data(), 0, w.vc.bytes(), stream));
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

void QwenModel::Impl::prefill(const int *tokens, size_t n, float *last_logits) {
    for (size_t i = 0; i < n; i += size_t(chunk)) {
        int rows = int(std::min<size_t>(chunk, n - i));
        forward_chunk(tokens + i, rows);
        if (last_logits && i + size_t(rows) == n) {
            lm_head_rows(rows - 1, 1);
            std::memcpy(last_logits, logits_all.data(), size_t(cfg.vocab) * 4);
        }
    }
}

void QwenModel::Impl::decode(int token, float *logits) {
    forward_chunk(&token, 1);
    if (logits) {
        lm_head_rows(0, 1);
        std::memcpy(logits, logits_all.data(), size_t(cfg.vocab) * 4);
    }
}

std::vector<int> QwenModel::Impl::generate(const int *prompt, size_t n_prompt, size_t n_new) {
    reset();
    std::vector<int> out;
    prefill(prompt, n_prompt, logits_h.data());
    int next = 0;
    for (size_t i = 0; i < size_t(cfg.vocab); ++i)
        if (logits_h[size_t(i)] > logits_h[size_t(next)]) next = i;
    for (size_t g = 0; g < n_new; ++g) {
        out.push_back(next);
        decode(next, logits_h.data());
        next = 0;
        for (size_t i = 0; i < size_t(cfg.vocab); ++i)
            if (logits_h[size_t(i)] > logits_h[size_t(next)]) next = i;
    }
    return out;
}

double QwenModel::Impl::perplexity(const int *tokens, size_t n) {
    if (n < 2) throw std::runtime_error("perplexity needs at least 2 tokens");
    reset();
    double nll = 0;
    long count = 0;
    // chunk [i, i+rows): logits of row r (position i+r) predict token at i+r+1
    for (size_t i = 0; i + 1 < n; i += size_t(chunk)) {
        int rows = int(std::min<size_t>(size_t(chunk), n - i));
        forward_chunk(tokens + i, rows);
        int nrows_log = int(std::min<size_t>(size_t(rows) - 1 + 1, n - 1 - i));  // rows with a target
        for (int r0 = 0; r0 < nrows_log; r0 += 16) {
            int nr = std::min(16, nrows_log - r0);
            lm_head_rows(r0, nr);
            for (int r = 0; r < nr; ++r) {
                size_t predict_at = i + size_t(r0 + r) + 1;
                float lp = logits_all[size_t(r) * cfg.vocab + size_t(tokens[predict_at])];
                nll += -std::log(std::max(lp, 1e-30f));
                count++;
            }
        }
    }
    return std::exp(nll / double(count));
}

// ------------------------------------------------------------------ public
QwenModel::QwenModel(const std::filesystem::path &dir, int device, int chunk, int max_context)
    : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    impl_->chunk = std::max(1, chunk);
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipStreamCreateWithFlags(&impl_->stream, hipStreamNonBlocking));
    impl_->cfg = QwenRuntimeConfig::read(dir);
    impl_->cfg.max_context = max_context;
    impl_->load(dir);
    impl_->reset();
}
QwenModel::~QwenModel() { (void)hipStreamDestroy(impl_->stream); }
const QwenRuntimeConfig &QwenModel::config() const { return impl_->cfg; }
void QwenModel::reset() { impl_->reset(); }
int QwenModel::position() const { return impl_->pos; }
void QwenModel::prefill(const int *tokens, size_t n, float *last_logits) {
    impl_->prefill(tokens, n, last_logits);
}
void QwenModel::decode(int token, float *logits) { impl_->decode(token, logits); }
std::vector<int> QwenModel::generate(const int *prompt, size_t n_prompt, size_t n_new) {
    return impl_->generate(prompt, n_prompt, n_new);
}
double QwenModel::perplexity(const int *tokens, size_t n) { return impl_->perplexity(tokens, n); }

}  // namespace behemoth
