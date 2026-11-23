// Qwen3.5 single-GPU QTIP runtime model. Loads the qtip_quant output (trellis linears + bf16 small
// tensors), runs hybrid Gated DeltaNet / gated full attention layers with a bf16 KV cache and
// per-layer fp32 delta state. Correctness-first: every projection goes through the trellis GEMV/WMMA
// kernels (rotate -> matmul -> slice sum -> hadU -> SV), the lm_head is a bf16 GEMM on the embedding.
#include "behemoth/qwen_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "behemoth/checkpoint.hpp"
#include "behemoth/qtip.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/qwen_kernels.hpp"

namespace behemoth {
// Minimal safetensors-directory reader for the qwen path (behemoth::Checkpoint validates the
// Mistral config and cannot load qwen3_5 checkpoints).
using json = nlohmann::json;
struct QwenCheckpoint {
    std::filesystem::path root;
    std::map<std::string, std::string> index;
    mutable std::map<std::string, std::shared_ptr<SafeTensorFile>> files;

    explicit QwenCheckpoint(std::filesystem::path p) : root(std::move(p)) {
        std::ifstream f(root / "model.safetensors.index.json");
        if (!f) throw std::runtime_error("no model.safetensors.index.json in " + root.string());
        json idx = json::parse(f);
        for (auto it = idx["weight_map"].begin(); it != idx["weight_map"].end(); ++it)
            index[it.key()] = it.value();
    }
    const SafeTensorFile &tensor(const std::string &name) const {
        auto it = index.find(name);
        if (it == index.end()) throw std::runtime_error("tensor not found: " + name);
        auto &ptr = files[it->second];
        if (!ptr) ptr = std::make_shared<SafeTensorFile>(root / it->second);
        return *ptr;
    }
    std::vector<BFloat16> bf16(const std::string &name) const {
        auto v = tensor(name).floats(name);
        std::vector<BFloat16> r(v.size());
        for (size_t i = 0; i < v.size(); ++i) r[i] = BFloat16(v[i]);
        return r;
    }
};
}  // namespace behemoth

namespace behemoth {
namespace {

json read_json(const std::filesystem::path &p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return json::parse(f);
}

std::vector<uint16_t> bf16_raw(const QwenCheckpoint &ck, const std::string &name) {
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
    if (c.num_experts > 0) {
        c.topk = t.value("num_experts_per_tok", t.value("n_experts_per_tok", 8));
        c.moe_inter = t.value("moe_intermediate_size", c.inter);
        c.inter = c.moe_inter;
        c.shared_inter = t.value("shared_expert_intermediate_size", c.moe_inter);
        c.shared_gate = t.value("shared_expert_gated", true);
        std::string scoring = t.value("scoring_func", "softmax");
        if (scoring != "softmax")
            throw std::runtime_error("only softmax router scoring supported (got " + scoring + ")");
    }
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
    // One linear: either a QTIP trellis (rotate -> matmul -> slice sum -> hadU -> SV epilogue) or a
    // plain bf16 [out][in] weight (unquantized checkpoints; the bf16-vs-quantized comparison then
    // runs the identical surrounding code).
    struct Linear {
        qtip::GpuLinear lin;
        DeviceBuffer<float> sv;  // [out]
        bool bf16 = false;
        int m = 0, n = 0;               // bf16 dims
        DeviceBuffer<uint16_t> w;       // bf16 weights [m][n]
        DeviceBuffer<float> su;         // trellis: input SU (per-Hessian-group shared)
        int rows() const { return bf16 ? m : lin.rows; }
        int in_dim() const { return bf16 ? n : lin.in; }
    };
    struct LayerW {
        bool full = false;
        bool moe = false;
        Linear lq, lk, lv, lo, lgate, lup, ldown;   // full attention + dense mlp
        Linear l_in_qkv, l_in_z, l_out;                 // linear attention (b/a stay bf16)
        std::vector<Linear> e_gate, e_up, e_down;       // MoE routed experts
        Linear sh_gate, sh_up, sh_down;                 // MoE shared expert
        DeviceBuffer<uint16_t> w_router;                // MoE router bf16 [E][H]
        DeviceBuffer<uint16_t> w_sg;                    // shared_expert_gate bf16 [H]
        DeviceBuffer<uint16_t> w_ba;                     // [2*nv][hidden] bf16 in_proj_b|a
        DeviceBuffer<uint16_t> ln_in, ln_post;      // raw zc weights (kernels add 1)
        DeviceBuffer<uint16_t> q_norm, k_norm;      // full attention (raw zc)
        DeviceBuffer<uint16_t> conv_w, dn_norm;     // linear attention
        DeviceBuffer<float> a_dt;                   // [2nv] = A_log | dt_bias
        DeviceBuffer<float> ring, dstate;           // per-layer delta state (f32)
        DeviceBuffer<uint16_t> kc, vc;              // KV cache [maxctx][nkv][hd] bf16
    };
    QwenRuntimeConfig cfg;
    int device = 0, chunk = 256;
    hipStream_t stream{};
    qtip::DenseMatmul dense;
    qtip::GpuCodebook codebook;
    std::vector<LayerW> layers;
    DeviceBuffer<uint16_t> embed, final_norm, lm_head_w;  // lm_head_w empty -> tied (use embed)
    // scratch (sized for chunk rows)
    DeviceBuffer<uint16_t> state, h_norm, u, q, k, v, gate, attn_out, o_in, gu, eX;
    DeviceBuffer<float> partial, z, z2, logits_d;
    DeviceBuffer<float> ones_f;  // all-1 "SV" for bf16 experts (max of inter/hidden)
    // MoE scratch
    DeviceBuffer<float> rlog, rwts, y_moe, sg_logit, slot_w;
    DeviceBuffer<int> rids, rcnt, rbase, rrun, slot_tok;
    DeviceBuffer<int> tokens_d;
    std::vector<float> logits_h, logits_all;
    int pos = 0;
    size_t max_slices = 0;

    void load(const std::filesystem::path &dir);
    qtip::HostLinear load_host(const QwenCheckpoint &ck, int layer, const std::string &name, int in, int out);
    void upload_linear(Linear &L, const qtip::HostLinear &h);
    void load_linear(Linear &L, const QwenCheckpoint &ck, int layer, const std::string &name, int in,
                     int out);  // trellis or bf16 fallback
    void run_linear(const Linear &L, const uint16_t *x, int ldx, int rows, float *out);
    void forward_chunk(const int *tokens_h, int rows);
    void moe_block(LayerW &w, int rows);
    void lm_head_rows(int row0, int nrows);  // logits_all[row0..] from state rows
    void reset();
    void prefill(const int *tokens, size_t n, float *last_logits);
    void decode(int token, float *logits);
    std::vector<int> generate(const int *prompt, size_t n_prompt, size_t n_new);
    double perplexity(const int *tokens, size_t n);
};

qtip::HostLinear QwenModel::Impl::load_host(const QwenCheckpoint &ck, int layer, const std::string &name,
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
        std::vector<float> r(size);
        for (size_t i = 0; i < size; ++i) r[i] = v[i];
        return r;
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

// Loads `model.layers.<layer>.<name>.` as a QTIP trellis linear, or — when the checkpoint stores a
// plain BF16 `weight` tensor there instead (unquantized model) — as a bf16 linear.
void QwenModel::Impl::load_linear(Linear &L, const QwenCheckpoint &ck, int layer, const std::string &name,
                                  int in, int out) {
    const std::string base = "model.layers." + std::to_string(layer) + "." + name + ".";
    if (ck.index.count(base + "trellis")) {
        auto h = load_host(ck, layer, name, in, out);
        if (h.in != in || h.out != out)
            throw std::runtime_error("linear dims mismatch for " + base + ": trellis " +
                                     std::to_string(h.out) + "x" + std::to_string(h.in) + " vs config " +
                                     std::to_string(out) + "x" + std::to_string(in));
        upload_linear(L, h);
        L.su.upload(h.su);
        L.bf16 = false;
        L.m = out;
        L.n = in;
    } else if (ck.index.count(base + "weight")) {
        L.bf16 = true;
        L.m = out;
        L.n = in;
        L.w.upload(bf16_raw(ck, base + "weight"));
    } else {
        throw std::runtime_error("no trellis or bf16 weight for " + base);
    }
}

// y[rows][out] = x[rows][ldx] . L^T (fp32 out, contiguous rows; trellis slice-summed or dense bf16
// GEMM).
void QwenModel::Impl::run_linear(const Linear &L, const uint16_t *x, int ldx, int rows, float *y) {
    if (L.bf16) {
        dense.gemm(L.m, L.n, reinterpret_cast<const BFloat16 *>(x), ldx, rows, y, L.m,
                   reinterpret_cast<const BFloat16 *>(L.w.data()), stream);
    } else {
        int slices = qtip::matmul_slices(L.lin.view(), rows);
        qtip::matmul(L.lin.view(), codebook.lut.data(), reinterpret_cast<const BFloat16 *>(x), ldx, rows,
                     partial.data(), slices, stream);
        q_sum_slices(partial.data(), slices, size_t(rows) * L.lin.rows, y, stream);
    }
}

void QwenModel::Impl::load(const std::filesystem::path &dir) {
    QwenCheckpoint ck(dir);
    cfg = QwenRuntimeConfig::read(dir);
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
        w.moe = cfg.num_experts > 0;
        std::string p = "model.layers." + std::to_string(L) + ".";
        auto up16 = [&](DeviceBuffer<uint16_t> &d, const std::string &name) {
            d.upload(bf16_raw(ck, p + name));
        };
        if (w.full) {
            load_linear(w.lq, ck, L, "self_attn.q_proj", H, cfg.nq * 2 * cfg.hd);
            load_linear(w.lk, ck, L, "self_attn.k_proj", H, cfg.nkv * cfg.hd);
            load_linear(w.lv, ck, L, "self_attn.v_proj", H, cfg.nkv * cfg.hd);
            load_linear(w.lo, ck, L, "self_attn.o_proj", A, H);
            up16(w.q_norm, "self_attn.q_norm.weight");
            up16(w.k_norm, "self_attn.k_norm.weight");
            w.kc.resize(size_t(cfg.max_context) * cfg.nkv * cfg.hd);
            w.vc.resize(size_t(cfg.max_context) * cfg.nkv * cfg.hd);
        } else {
            load_linear(w.l_in_qkv, ck, L, "linear_attn.in_proj_qkv", H, cfg.conv_dim());
            load_linear(w.l_in_z, ck, L, "linear_attn.in_proj_z", H, V);
            load_linear(w.l_out, ck, L, "linear_attn.out_proj", V, H);
            {  // b|a: raw bf16 rows [2nv][H]
                std::vector<uint16_t> b = bf16_raw(ck, p + "linear_attn.in_proj_b.weight");
                std::vector<uint16_t> a = bf16_raw(ck, p + "linear_attn.in_proj_a.weight");
                b.insert(b.end(), a.begin(), a.end());
                w.w_ba.upload(b);
            }
            up16(w.conv_w, "linear_attn.conv1d.weight");
            up16(w.dn_norm, "linear_attn.norm.weight");
            {
                auto al = ck.tensor(p + "linear_attn.A_log").floats(p + "linear_attn.A_log");
                auto db = ck.tensor(p + "linear_attn.dt_bias").floats(p + "linear_attn.dt_bias");
                std::vector<float> aux;
                for (size_t i = 0; i < al.size(); ++i) aux.push_back(al[i]);
                for (size_t i = 0; i < db.size(); ++i) aux.push_back(db[i]);
                w.a_dt.upload(aux);
            }
            w.ring.resize(size_t(cfg.conv_dim()) * cfg.conv_k);
            w.dstate.resize(size_t(cfg.lin_nv) * cfg.lin_kd * cfg.lin_vd);
        }
        if (w.moe) {
            const int E = cfg.num_experts, F = cfg.moe_inter, S = cfg.shared_inter;
            up16(w.w_router, "mlp.gate.weight");
            if (cfg.shared_gate) up16(w.w_sg, "mlp.shared_expert_gate.weight");
            w.e_gate.resize(size_t(E));
            w.e_up.resize(size_t(E));
            w.e_down.resize(size_t(E));
            for (int e = 0; e < E; ++e) {
                std::string ep = "mlp.experts." + std::to_string(e) + ".";
                load_linear(w.e_gate[size_t(e)], ck, L, ep + "gate_proj", H, F);
                load_linear(w.e_up[size_t(e)], ck, L, ep + "up_proj", H, F);
                load_linear(w.e_down[size_t(e)], ck, L, ep + "down_proj", F, H);
            }
            load_linear(w.sh_gate, ck, L, "mlp.shared_expert.gate_proj", H, S);
            load_linear(w.sh_up, ck, L, "mlp.shared_expert.up_proj", H, S);
            load_linear(w.sh_down, ck, L, "mlp.shared_expert.down_proj", S, H);
        } else {
            load_linear(w.lgate, ck, L, "mlp.gate_proj", H, cfg.inter);
            load_linear(w.lup, ck, L, "mlp.up_proj", H, cfg.inter);
            load_linear(w.ldown, ck, L, "mlp.down_proj", cfg.inter, H);
        }
        up16(w.ln_in, "input_layernorm.weight");
        up16(w.ln_post, "post_attention_layernorm.weight");
    }
    std::string embed_name, norm_name;
    for (std::string nm : {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"})
        if (ck.index.count(nm)) embed_name = nm;
    for (std::string nm : {"model.language_model.norm.weight", "model.norm.weight"})
        if (ck.index.count(nm)) norm_name = nm;
    if (embed_name.empty() || norm_name.empty())
        throw std::runtime_error("cannot find embed_tokens / final norm in the checkpoint index");
    embed.upload(bf16_raw(ck, embed_name));
    final_norm.upload(bf16_raw(ck, norm_name));
    if (!cfg.tied) {
        if (!ck.index.count("lm_head.weight"))
            throw std::runtime_error("tie_word_embeddings=false but no lm_head.weight");
        lm_head_w.upload(bf16_raw(ck, "lm_head.weight"));
    }
    // scratch
    const int R = chunk;
    const int FS = std::max(cfg.inter, cfg.shared_inter);
    const int Wz = std::max({cfg.nq * 2 * cfg.hd + 2 * cfg.nkv * cfg.hd,
                             cfg.conv_dim() + V + 2 * cfg.lin_nv, 2 * FS});
    state.resize(size_t(R) * H);
    h_norm.resize(size_t(R) * H);
    u.resize(size_t(R) * std::max({H, V, A, FS}));
    q.resize(size_t(R) * cfg.nq * 2 * cfg.hd);
    k.resize(size_t(R) * cfg.nkv * cfg.hd);
    v.resize(size_t(R) * cfg.nkv * cfg.hd);
    gate.resize(size_t(R) * A);
    attn_out.resize(size_t(R) * A);
    o_in.resize(size_t(R) * V);
    gu.resize(size_t(R) * FS);
    eX.resize(size_t(R) * H);
    z.resize(size_t(R) * Wz);
    z2.resize(size_t(R) * Wz);
    tokens_d.resize(size_t(R));
    ones_f.resize(size_t(std::max(FS, H)));
    {
        std::vector<float> ones(ones_f.size(), 1.0f);
        ones_f.upload(ones);
    }
    rlog.resize(size_t(R) * std::max(1, cfg.num_experts));
    rwts.resize(size_t(R) * std::max(1, cfg.topk));
    y_moe.resize(size_t(R) * H);
    sg_logit.resize(size_t(R));
    slot_w.resize(size_t(R) * std::max(1, cfg.topk));
    rids.resize(size_t(R) * std::max(1, cfg.topk));
    rcnt.resize(size_t(std::max(1, cfg.num_experts)));
    rbase.resize(size_t(std::max(1, cfg.num_experts)));
    rrun.resize(size_t(std::max(1, cfg.num_experts)));
    slot_tok.resize(size_t(R) * std::max(1, cfg.topk));
    for (auto &w : layers) {
        std::vector<const Linear *> lins = {&w.lq, &w.lk,      &w.lv,   &w.lo,   &w.lgate, &w.lup,
                                            &w.ldown, &w.l_in_qkv, &w.l_in_z, &w.l_out, &w.sh_gate,
                                            &w.sh_up, &w.sh_down};
        for (auto &e : w.e_gate) lins.push_back(&e);
        for (auto &e : w.e_up) lins.push_back(&e);
        for (auto &e : w.e_down) lins.push_back(&e);
        for (const Linear *gv : lins)
            if (!gv->bf16 && gv->lin.rows)
                max_slices = std::max<size_t>(max_slices, qtip::matmul_slices(gv->lin.view(), R));
    }
    partial.resize(max_slices * size_t(R) * Wz);
    logits_all.resize(size_t(std::min(R, 16)) * cfg.vocab);
    logits_d.resize(size_t(std::min(R, 16)) * cfg.vocab);
    logits_h.resize(size_t(cfg.vocab));
}

// logits of state rows [row0, row0+nrows) -> logits_all (sub-batches of <= 16 rows)
void QwenModel::Impl::lm_head_rows(int row0, int nrows) {
    const int H = cfg.hidden;
    const uint16_t *head = lm_head_w.size() ? lm_head_w.data() : embed.data();
    q_zc_norm(state.data() + size_t(row0) * H, nrows, H, final_norm.data(), cfg.eps, stream);
    for (int r0 = 0; r0 < nrows; r0 += 16) {
        int nr = std::min(16, nrows - r0);
        dense.gemm(cfg.vocab, H,
                   reinterpret_cast<const BFloat16 *>(state.data() + size_t(row0 + r0) * H), H, nr,
                   logits_d.data() + size_t(r0) * cfg.vocab, cfg.vocab,
                   reinterpret_cast<const BFloat16 *>(head), stream);
        HIP_CHECK(hipMemcpyAsync(logits_all.data() + size_t(r0) * cfg.vocab,
                                 logits_d.data() + size_t(r0) * cfg.vocab, size_t(nr) * cfg.vocab * 4,
                                 hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    if (const char *lp = std::getenv("QWEN_LOGITS_OUT")) {
        static bool once = false;
        if (!once) {
            once = true;
            std::ofstream f(lp, std::ios::binary);
            f.write(reinterpret_cast<const char *>(logits_all.data()), std::streamsize(size_t(nrows) * cfg.vocab * 4));
        }
    }
    if (std::getenv("QWEN_RT_DBG")) {
        float *row = logits_all.data();
        for (int k = 0; k < 3; ++k) {
            int am = 0;
            for (long v = 1; v < cfg.vocab; ++v)
                if (row[size_t(v)] > row[size_t(am)]) am = int(v);
            std::fprintf(stderr, "logit top%d: id %d val %.4f; samples: %.3f %.3f %.3f\n", k, am,
                         row[size_t(am)], row[0], row[1], row[100]);
            row[size_t(am)] = -1e30f;
        }
    }
}

// MoE MLP block: router (softmax -> top-k -> renorm), routed experts on compact per-expert token
// lists, sigmoid-gated shared expert. Experts are QTIP trellis linears or plain bf16 (same Linear).
void QwenModel::Impl::moe_block(LayerW &w, int rows) {
    const QwenRuntimeConfig &c = cfg;
    const int H = c.hidden, E = c.num_experts, K = c.topk, F = c.moe_inter, S = c.shared_inter;
    const bool e16 = !w.e_gate.empty() && w.e_gate[0].bf16;
    // h_norm = zc-norm(state, ln_post): router + shared-gate + bf16-expert input
    HIP_CHECK(hipMemcpyAsync(h_norm.data(), state.data(), size_t(rows) * H * 2,
                             hipMemcpyDeviceToDevice, stream));
    q_zc_norm(h_norm.data(), rows, H, w.ln_post.data(), c.eps, stream);
    dense.gemm(E, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows, rlog.data(), E,
               reinterpret_cast<const BFloat16 *>(w.w_router.data()), stream);
    HIP_CHECK(hipMemsetAsync(rcnt.data(), 0, rcnt.bytes(), stream));
    q_moe_route(rlog.data(), rows, E, K, rwts.data(), rids.data(), rcnt.data(), stream);
    q_moe_scan(rcnt.data(), rbase.data(), E, stream);
    HIP_CHECK(hipMemsetAsync(rrun.data(), 0, rrun.bytes(), stream));
    q_moe_route_fill(rlog.data(), rows, E, K, rbase.data(), rrun.data(), slot_tok.data(), slot_w.data(),
                     stream);
    std::vector<int> cnt{size_t(E)}, base{size_t(E)};
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(cnt.data(), rcnt.data(), size_t(E) * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(base.data(), rbase.data(), size_t(E) * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemsetAsync(y_moe.data(), 0, size_t(rows) * H * 4, stream));
    // routed experts
    for (int e = 0; e < E; ++e) {
        const int ne = cnt[size_t(e)];
        if (ne <= 0) continue;
        const Linear &lg = w.e_gate[size_t(e)];
        const Linear &lu = w.e_up[size_t(e)];
        const Linear &ld = w.e_down[size_t(e)];
        const int *tok_e = slot_tok.data() + base[size_t(e)];
        const float *w_e = slot_w.data() + base[size_t(e)];
        q_moe_gather(state.data(), H, tok_e, eX.data(), ne, stream);  // raw rows
        if (!e16) {
            q_input_transform(eX.data(), ne, H, w.ln_post.data(), c.eps, lg.su.data(), H, u.data(), H,
                              stream);
            run_linear(lg, u.data(), H, ne, z.data());
            q_hadamard_rows(z.data(), ne, F, F, false, stream);
            run_linear(lu, u.data(), H, ne, z2.data());
            q_hadamard_rows(z2.data(), ne, F, F, false, stream);
            q_gateup(z.data(), z2.data(), lg.sv.data(), lu.sv.data(), gu.data(), ne, F, F, stream);
            q_input_transform(gu.data(), ne, F, nullptr, 0, ld.su.data(), F, u.data(), F, stream);
            run_linear(ld, u.data(), F, ne, z.data());
            q_hadamard_rows(z.data(), ne, H, H, false, stream);
            q_moe_scatter_sv_add(y_moe.data(), z.data(), ld.sv.data(), tok_e, w_e, ne, H, stream);
        } else {
            q_zc_norm(eX.data(), ne, H, w.ln_post.data(), c.eps, stream);
            run_linear(lg, eX.data(), H, ne, z.data());
            run_linear(lu, eX.data(), H, ne, z2.data());
            q_gateup(z.data(), z2.data(), ones_f.data(), ones_f.data(), gu.data(), ne, F, F, stream);
            run_linear(ld, gu.data(), F, ne, z.data());
            q_moe_scatter_sv_add(y_moe.data(), z.data(), ones_f.data(), tok_e, w_e, ne, H, stream);
        }
    }
    // shared expert (+ sigmoid gate)
    const bool s16 = w.sh_gate.bf16;
    if (!s16) {
        q_input_transform(state.data(), rows, H, w.ln_post.data(), c.eps, w.sh_gate.su.data(), H,
                          u.data(), H, stream);
        run_linear(w.sh_gate, u.data(), H, rows, z.data());
        q_hadamard_rows(z.data(), rows, S, S, false, stream);
        run_linear(w.sh_up, u.data(), H, rows, z2.data());
        q_hadamard_rows(z2.data(), rows, S, S, false, stream);
        q_gateup(z.data(), z2.data(), w.sh_gate.sv.data(), w.sh_up.sv.data(), gu.data(), rows, S, S,
                 stream);
        q_input_transform(gu.data(), rows, S, nullptr, 0, w.sh_down.su.data(), S, u.data(), S, stream);
        run_linear(w.sh_down, u.data(), S, rows, z.data());
        q_hadamard_rows(z.data(), rows, H, H, false, stream);
        if (c.shared_gate) {
            dense.gemm(1, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows, sg_logit.data(),
                       1, reinterpret_cast<const BFloat16 *>(w.w_sg.data()), stream);
            q_moe_sv_add_gated(z.data(), w.sh_down.sv.data(), sg_logit.data(), state.data(), rows, H,
                               stream);
        } else {
            q_moe_sv_add_gated(z.data(), w.sh_down.sv.data(), nullptr, state.data(), rows, H, stream);
        }
    } else {
        run_linear(w.sh_gate, h_norm.data(), H, rows, z.data());
        run_linear(w.sh_up, h_norm.data(), H, rows, z2.data());
        q_gateup(z.data(), z2.data(), ones_f.data(), ones_f.data(), gu.data(), rows, S, S, stream);
        run_linear(w.sh_down, gu.data(), S, rows, z.data());
        if (c.shared_gate) {
            dense.gemm(1, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows, sg_logit.data(),
                       1, reinterpret_cast<const BFloat16 *>(w.w_sg.data()), stream);
            q_moe_sv_add_gated(z.data(), ones_f.data(), sg_logit.data(), state.data(), rows, H, stream);
        } else {
            q_moe_sv_add_gated(z.data(), ones_f.data(), nullptr, state.data(), rows, H, stream);
        }
    }
    q_moe_residual_add(state.data(), y_moe.data(), rows, H, stream);
}

void QwenModel::Impl::forward_chunk(const int *tokens_h, int rows) {
    const QwenRuntimeConfig &c = cfg;
    const int H = c.hidden, A = c.nq * c.hd, V = c.value_dim(), F = c.inter;
    const int pos0 = pos;
    HIP_CHECK(hipMemcpyAsync(tokens_d.data(), tokens_h, size_t(rows) * 4, hipMemcpyHostToDevice, stream));
    q_embed_gather(embed.data(), tokens_d.data(), state.data(), rows, H, stream);
    if (std::getenv("QWEN_RT_DBG")) {
        std::vector<uint16_t> h = state.download();
        std::fprintf(stderr, "embed row0: %.4f %.4f %.4f %.4f  raw %04x %04x\n",
                     float(BFloat16::from_bits(h[0])), float(BFloat16::from_bits(h[1])),
                     float(BFloat16::from_bits(h[2])), float(BFloat16::from_bits(h[3])), h[0], h[1]);
    }
    // one trellis projection: rotated input is already in u (row stride n); z <- sum slices (dense f32)
    auto proj = [&](const Linear &lin, int n, float *zdst) {
        int slices = qtip::matmul_slices(lin.lin.view(), rows);
        qtip::matmul(lin.lin.view(), codebook.lut.data(), reinterpret_cast<const BFloat16 *>(u.data()), n,
                     rows, partial.data(), slices, stream);
        q_sum_slices(partial.data(), slices, size_t(rows) * lin.lin.rows, zdst, stream);
    };
    for (int L = 0; L < c.layers; ++L) {
        LayerW &w = layers[size_t(L)];
        const bool lin16 = w.full ? w.lq.bf16 : w.l_in_qkv.bf16;  // bf16 (unquantized) checkpoint
        auto zc_hidden = [&]() {  // h_norm = zc-norm(state, ln_in): plain input of bf16 linears
            HIP_CHECK(hipMemcpyAsync(h_norm.data(), state.data(), size_t(rows) * H * 2,
                                     hipMemcpyDeviceToDevice, stream));
            q_zc_norm(h_norm.data(), rows, H, w.ln_in.data(), c.eps, stream);
        };
        if (w.full) {
            const int Wq = c.nq * 2 * c.hd, Wk = c.nkv * c.hd;
            if (!lin16) {
                q_input_transform(state.data(), rows, H, w.ln_in.data(), c.eps, w.lq.su.data(), H,
                                  u.data(), H, stream);
                proj(w.lq, H, z.data());
                q_hadamard_rows(z.data(), rows, Wq, Wq, false, stream);
                q_sv_cast(z.data(), w.lq.sv.data(), q.data(), rows, Wq, Wq, stream);
                proj(w.lk, H, z.data());
                q_hadamard_rows(z.data(), rows, Wk, Wk, false, stream);
                q_sv_cast(z.data(), w.lk.sv.data(), k.data(), rows, Wk, Wk, stream);
                proj(w.lv, H, z.data());
                q_hadamard_rows(z.data(), rows, Wk, Wk, false, stream);
                q_sv_cast(z.data(), w.lv.sv.data(), v.data(), rows, Wk, Wk, stream);
            } else {
                zc_hidden();
                run_linear(w.lq, h_norm.data(), H, rows, z.data());
                q_sv_cast(z.data(), ones_f.data(), q.data(), rows, Wq, Wq, stream);
                run_linear(w.lk, h_norm.data(), H, rows, z.data());
                q_sv_cast(z.data(), ones_f.data(), k.data(), rows, Wk, Wk, stream);
                run_linear(w.lv, h_norm.data(), H, rows, z.data());
                q_sv_cast(z.data(), ones_f.data(), v.data(), rows, Wk, Wk, stream);
            }
            q_qk_norm_rope(q.data(), k.data(), v.data(), gate.data(), w.kc.data(), w.vc.data(), rows,
                           pos0, c.nq, c.nkv, c.hd, c.rot_dim, c.theta, w.q_norm.data(),
                           w.k_norm.data(), c.eps, stream);
            q_attn(q.data(), w.kc.data(), w.vc.data(), gate.data(), attn_out.data(), rows, pos0, c.nq,
                   c.nkv, c.hd, 1.0f / std::sqrt(float(c.hd)), stream);
            if (!w.lo.bf16) {
                q_input_transform(attn_out.data(), rows, A, nullptr, 0, w.lo.su.data(), A, u.data(), A,
                                  stream);
                proj(w.lo, A, z.data());
                q_hadamard_rows(z.data(), rows, H, H, false, stream);
                q_sv_add(z.data(), w.lo.sv.data(), state.data(), rows, H, H, H, stream);
            } else {
                run_linear(w.lo, attn_out.data(), A, rows, z.data());
                q_moe_residual_add(state.data(), z.data(), rows, H, stream);
            }
        } else {
            const int Wq = c.conv_dim(), nv = c.lin_nv;
            const int Wz = Wq + V + 2 * nv;
            if (!lin16) {
                auto proj_scatter = [&](const Linear &lin, int n, long col_off) {
                    q_input_transform(state.data(), rows, H, w.ln_in.data(), c.eps, lin.su.data(), H,
                                      u.data(), H, stream);
                    proj(lin, n, z2.data());
                    q_hadamard_rows(z2.data(), rows, lin.lin.rows, lin.lin.rows, false, stream);
                    q_sv_cast_f32(z2.data(), lin.sv.data(), z.data() + col_off, rows, lin.lin.rows,
                                  lin.lin.rows, Wz, stream);
                };
                proj_scatter(w.l_in_qkv, H, 0);
                proj_scatter(w.l_in_z, H, Wq);
            } else {
                zc_hidden();
                run_linear(w.l_in_qkv, h_norm.data(), H, rows, z2.data());
                q_scatter_f32(z.data(), Wz, 0, z2.data(), Wq, rows, Wq, stream);
                run_linear(w.l_in_z, h_norm.data(), H, rows, z2.data());
                q_scatter_f32(z.data(), Wz, Wq, z2.data(), V, rows, V, stream);
            }
            // b/a: unquantised bf16 GEMMs on the plain (zc-normed) hidden state
            if (!lin16) {
                HIP_CHECK(hipMemcpyAsync(h_norm.data(), state.data(), size_t(rows) * H * 2,
                                         hipMemcpyDeviceToDevice, stream));
                q_zc_norm(h_norm.data(), rows, H, w.ln_in.data(), c.eps, stream);
            }
            dense.gemm(nv, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows,
                       z.data() + Wq + V, Wz, reinterpret_cast<const BFloat16 *>(w.w_ba.data()), stream);
            dense.gemm(nv, H, reinterpret_cast<const BFloat16 *>(h_norm.data()), H, rows,
                       z.data() + Wq + V + nv, Wz,
                       reinterpret_cast<const BFloat16 *>(w.w_ba.data()) + size_t(cfg.lin_nv) * H,
                       stream);
            for (int t = 0; t < rows; ++t)
                q_delta_step(z.data(), w.conv_w.data(), w.dn_norm.data(), w.a_dt.data(), w.ring.data(),
                             w.dstate.data(), o_in.data(), rows, t, Wq, V, c.lin_kd, c.lin_vd,
                             c.lin_nk, nv, c.conv_k, c.eps, stream);
            if (!w.l_out.bf16) {
                q_input_transform(o_in.data(), rows, V, nullptr, 0, w.l_out.su.data(), V, u.data(), V,
                                  stream);
                proj(w.l_out, V, z.data());
                q_hadamard_rows(z.data(), rows, H, H, false, stream);
                q_sv_add(z.data(), w.l_out.sv.data(), state.data(), rows, H, H, H, stream);
            } else {
                run_linear(w.l_out, o_in.data(), V, rows, z.data());
                q_moe_residual_add(state.data(), z.data(), rows, H, stream);
            }
        }
        if (std::getenv("QWEN_RT_DBG")) {
            std::vector<uint16_t> h = state.download();
            double ss = 0;
            for (int i = 0; i < H; ++i) {
                float v;
                uint16_t b = h[size_t(i)];
                uint32_t u = uint32_t(b) << 16;
                std::memcpy(&v, &u, 4);
                ss += double(v) * v;
            }
            std::fprintf(stderr, "L%d %s post-attn norm %.4f\n", L, w.full ? "full" : "lin", std::sqrt(ss / H));
        }
        if (w.moe) {
            moe_block(w, rows);
        } else if (!w.lgate.bf16) {
            q_input_transform(state.data(), rows, H, w.ln_post.data(), c.eps, w.lgate.su.data(), H,
                              u.data(), H, stream);
            proj(w.lgate, H, z.data());
            q_hadamard_rows(z.data(), rows, F, F, false, stream);
            proj(w.lup, H, z2.data());
            q_hadamard_rows(z2.data(), rows, F, F, false, stream);
            q_gateup(z.data(), z2.data(), w.lgate.sv.data(), w.lup.sv.data(), gu.data(), rows, F, F,
                     stream);
            q_input_transform(gu.data(), rows, F, nullptr, 0, w.ldown.su.data(), F, u.data(), F, stream);
            proj(w.ldown, F, z.data());
            q_hadamard_rows(z.data(), rows, H, H, false, stream);
            q_sv_add(z.data(), w.ldown.sv.data(), state.data(), rows, H, H, H, stream);
        } else {
            HIP_CHECK(hipMemcpyAsync(h_norm.data(), state.data(), size_t(rows) * H * 2,
                                     hipMemcpyDeviceToDevice, stream));
            q_zc_norm(h_norm.data(), rows, H, w.ln_post.data(), c.eps, stream);
            run_linear(w.lgate, h_norm.data(), H, rows, z.data());
            run_linear(w.lup, h_norm.data(), H, rows, z2.data());
            q_gateup(z.data(), z2.data(), ones_f.data(), ones_f.data(), gu.data(), rows, F, F, stream);
            run_linear(w.ldown, gu.data(), F, rows, z.data());
            q_moe_residual_add(state.data(), z.data(), rows, H, stream);
        }
        if (std::getenv("QWEN_RT_DBG")) {
            std::vector<uint16_t> h = state.download();
            double ss = 0;
            for (int i = 0; i < H; ++i) {
                float v;
                uint16_t b = h[size_t(i)];
                uint32_t u = uint32_t(b) << 16;
                std::memcpy(&v, &u, 4);
                ss += double(v) * v;
            }
            std::fprintf(stderr, "L%d mlp norm %.4f\n", L, std::sqrt(ss / H));
        }
        if (const char *dbg = std::getenv("QWEN_DUMP_LAYERS")) {
            static int dump_rows = 0;
            if (!dump_rows) dump_rows = rows;
            if (rows == dump_rows && pos == 0) {
                std::vector<uint16_t> h = state.download();
                std::ofstream f(std::string(dbg) + "/rt_layer" + std::to_string(L) + ".bin",
                                std::ios::binary);
                f.write(reinterpret_cast<const char *>(h.data()),
                        std::streamsize(size_t(rows) * H * 2));
            }
        }
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
                const float *row = logits_all.data() + size_t(r) * cfg.vocab;
                float mx = -INFINITY;
                for (long v = 0; v < cfg.vocab; ++v) mx = std::max(mx, row[size_t(v)]);
                double lse = 0;
                for (long v = 0; v < cfg.vocab; ++v) lse += std::exp(double(row[size_t(v)]) - mx);
                lse = std::log(lse) + mx;
                nll += lse - row[size_t(tokens[predict_at])];
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
