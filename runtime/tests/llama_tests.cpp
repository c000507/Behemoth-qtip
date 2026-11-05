// Llama 3.3 70B support tests (CUDA): llama3 RoPE scaling against an fp64 reference of HF's
// _compute_llama3_parameters, and end-to-end inference of synthetic Llama-3.3-geometry-scaled QTIP /
// mixed QTIP+NVFP4 checkpoints (same GQA 8:1, FFN 3.5x with the 28*2^p Hadamard factor, untied
// embeddings, rope_theta 5e5 + llama3 rope_scaling, generation_config eos list) against a CPU
// reference of the whole model. usage: behemoth_cuda_llama_tests [--checkpoint DIR]...
// Real checkpoints given with --checkpoint (e.g. qtip_quant output) get the same CPU comparison.
#include "behemoth/checkpoint.hpp"
#include "behemoth/cuda.hpp"
#include "behemoth/numeric.hpp"
#include "behemoth/qtip.hpp"
#include "kernels.hpp"
#include "safetensors.hpp"
#include <algorithm>
#include <ctime>
#include <array>
#include <bit>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
using namespace behemoth;
namespace cg = glm53::gpu;
namespace cu = behemoth::cuda_runtime;
namespace fs = std::filesystem;
using json = nlohmann::json;
namespace {
void require(bool ok, const std::string &text) {
    if (!ok)
        throw std::runtime_error(text);
}
// ---------------------------------------------------------------- llama3 RoPE reference
// Literal fp64 transcription of transformers.modeling_rope_utils._compute_llama3_parameters.
std::vector<double> hf_llama3(int dim, double base, const RopeScalingLlama3 &r) {
    std::vector<double> inv(size_t(dim / 2)), out(inv.size());
    for (int i = 0; i < dim / 2; ++i)
        inv[size_t(i)] = 1.0 / std::pow(base, double(2 * i) / dim);
    const double low_freq_wavelen = double(r.original_max_position_embeddings) / r.low_freq_factor;
    const double high_freq_wavelen = double(r.original_max_position_embeddings) / r.high_freq_factor;
    for (size_t i = 0; i < inv.size(); ++i) {
        const double wavelen = 2 * M_PI / inv[i];
        double v = wavelen > low_freq_wavelen ? inv[i] / r.factor : inv[i];
        const double smooth = (double(r.original_max_position_embeddings) / wavelen - r.low_freq_factor) /
                              (r.high_freq_factor - r.low_freq_factor);
        const double smoothed = (1 - smooth) * v / r.factor + smooth * v;
        const bool medium = !(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen);
        out[i] = medium ? smoothed : v;
    }
    return out;
}
void inv_freq_test() {
    const RopeScalingLlama3 real{8.0, 1.0, 4.0, 8192};
    for (auto [dim, theta, r] :
         {std::tuple{128, 500000.0, real}, std::tuple{32, 500000.0, RopeScalingLlama3{8.0, 1.0, 4.0, 64}},
          std::tuple{64, 10000.0, RopeScalingLlama3{4.0, 2.0, 5.0, 4096}}}) {
        auto actual = llama3_inv_freq(dim, theta, r);
        auto expected = hf_llama3(dim, theta, r);
        require(actual.size() == expected.size(), "inv_freq size");
        int scaled = 0, interpolated = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::abs(double(actual[i]) - expected[i]) <= 1e-6 * expected[i], "llama3 inv_freq mismatch");
            const double plain = 1.0 / std::pow(theta, double(2 * i) / dim);
            scaled += std::abs(expected[i] - plain / r.factor) <= 1e-12 * plain;
            interpolated += std::abs(expected[i] - plain) > 1e-9 * plain &&
                            std::abs(expected[i] - plain / r.factor) > 1e-9 * plain;
        }
        require(scaled > 0 && interpolated > 0 && actual[0] == 1.0f * float(expected[0]),
                "llama3 test must exercise scaled, interpolated and untouched bands");
    }
    // Llama 3.3 70B published values: the highest frequency is untouched, the lowest is divided by 8.
    auto f = llama3_inv_freq(128, 500000.0, real);
    require(f.size() == 64 && f[0] == 1.0f, "llama3 inv_freq[0] must be 1");
    require(std::abs(double(f[63]) - std::pow(500000.0, -126.0 / 128) / 8) < 1e-9, "llama3 lowest frequency");
    std::cout << "llama3 inv_freq matches fp64 HF reference (dims 128/32/64)\n";
}
void rope_kernel_test(cg::Stream s) {
    constexpr int qh = 4, kh = 2, dim = 128;
    const RopeScalingLlama3 real{8.0, 1.0, 4.0, 8192};
    auto table = llama3_inv_freq(dim, 500000.0, real);
    auto hf = hf_llama3(dim, 500000.0, real);
    cg::Buffer<float> d_table, result;
    cg::Buffer<uint16_t> q, k;
    d_table.ensure(table.size(), s);
    cg::checked(cg::h2d(d_table, table.data(), table.size() * 4, s), "upload rope table");
    q.ensure(qh * dim, s);
    k.ensure(kh * dim, s);
    result.ensure((qh + kh) * dim, s);
    cg::Pinned<float> host;
    host.ensure((qh + kh) * dim);
    std::mt19937 gen(5);
    std::normal_distribution<float> normal(0, 1);
    for (int position : {0, 1, 5, 777, 8191, 8192, 40000, 131071}) {
        std::vector<float> in((qh + kh) * dim);
        for (auto &v : in)
            v = float(BFloat16(normal(gen)));
        std::vector<uint16_t> bq, bk;
        for (int i = 0; i < qh * dim; ++i)
            bq.push_back(BFloat16(in[size_t(i)]).bits);
        for (int i = qh * dim; i < (qh + kh) * dim; ++i)
            bk.push_back(BFloat16(in[size_t(i)]).bits);
        cg::checked(cg::h2d(q, bq.data(), bq.size() * 2, s), "upload q");
        cg::checked(cg::h2d(k, bk.data(), bk.size() * 2, s), "upload k");
        cu::rope_scaled(q, k, position, qh, kh, dim, d_table, s);
        cg::to_float(q, result, qh * dim, s);
        cg::to_float(k, result.p + qh * dim, kh * dim, s);
        cg::checked(cg::d2h(host.p, result, (qh + kh) * dim * 4, s), "read rope");
        cg::checked(cg::sync(s), "finish rope");
        double e = 0, n = 0;
        for (int h = 0; h < qh + kh; ++h)
            for (int d = 0; d < dim / 2; ++d) {
                // HF: float32 position * float32 inv_freq, then cos/sin (rotate_half pairing d, d+dim/2)
                const double angle = double(float(position) * float(hf[size_t(d)]));
                const double a = in[size_t(h * dim + d)], b = in[size_t(h * dim + d + dim / 2)];
                const double o1 = a * std::cos(angle) - b * std::sin(angle),
                             o2 = b * std::cos(angle) + a * std::sin(angle);
                e += std::pow(host.p[h * dim + d] - o1, 2) + std::pow(host.p[h * dim + d + dim / 2] - o2, 2);
                n += o1 * o1 + o2 * o2;
            }
        const double relative = std::sqrt(e / n);
        std::cout << "rope_scaled position " << position << " relative L2 " << relative << '\n';
        require(relative < 0.006, "CUDA llama3 RoPE mismatch");
    }
}
// ---------------------------------------------------------------- synthetic checkpoint
struct Geometry {
    int hidden = 256, inter = 896, layers = 2, nq = 8, nkv = 1, hd = 32, vocab = 320;
    double theta = 500000;
    bool llama3 = true;
    bool nvfp4_layer1 = false; // mixed adaptive-style checkpoint: layer 1 projections are NVFP4
    std::vector<int> eos{261, 264}; // <|end_of_text|>, <|eot_id|> of the synthetic tokenizer
    bool tokenizer = true;
};
std::pair<int, int> shape_of(const Geometry &g, const std::string &name) {
    if (name == "self_attn.q_proj" || name == "self_attn.o_proj")
        return {g.hidden, g.nq * g.hd};
    if (name == "self_attn.k_proj" || name == "self_attn.v_proj")
        return {g.hidden, g.nkv * g.hd};
    if (name == "mlp.gate_proj" || name == "mlp.up_proj")
        return {g.hidden, g.inter};
    return {g.inter, g.hidden};
}
void add_qtip(::qtip::SafeTensorsWriter &w, const std::string &base, int in, int out, std::mt19937 &gen,
              float gain) {
    std::vector<uint16_t> trellis(size_t(out / 16) * size_t(in / 16) * 32), tlut(1024);
    for (auto &v : trellis)
        v = uint16_t(gen());
    for (auto &v : tlut)
        v = uint16_t((gen() & 0x8000) | (0x3000 + gen() % 0x1000));
    std::vector<float> su(static_cast<size_t>(in)), sv(static_cast<size_t>(out));
    for (auto &v : su)
        v = gen() % 2 ? 1.f : -1.f;
    for (auto &v : sv)
        v = (gen() % 2 ? 1.f : -1.f) * (gain * 0.35f / std::sqrt(float(in)));
    const int64_t zero = 0, ranks = 8;
    w.add(base + "trellis", "I16", {int64_t(trellis.size() / 32), 32}, trellis.data(), trellis.size() * 2);
    w.add(base + "SU", "F32", {in}, su.data(), su.size() * 4);
    w.add(base + "SV", "F32", {out}, sv.data(), sv.size() * 4);
    w.add(base + "tlut", "F16", {512, 2}, tlut.data(), tlut.size() * 2);
    w.add(base + "rcp", "I64", {}, &zero, 8);
    w.add(base + "tp_rank", "I64", {}, &ranks, 8);
}
void add_nvfp4(::qtip::SafeTensorsWriter &w, const std::string &base, int in, int out, std::mt19937 &gen,
               float gain) {
    std::normal_distribution<float> normal(0, gain / std::sqrt(float(in)));
    std::vector<float> dense(size_t(out) * in);
    float maximum = 0;
    for (auto &v : dense) {
        v = normal(gen);
        maximum = std::max(maximum, std::abs(v));
    }
    const float global = maximum / (448.f * 6);
    constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    auto fp4 = [&](float x) {
        float a = std::abs(x), best = INFINITY;
        int code = 0;
        for (int i = 0; i < 8; ++i)
            if (std::abs(a - levels[i]) < best) {
                best = std::abs(a - levels[i]);
                code = i;
            }
        return uint8_t(code | (std::signbit(x) ? 8 : 0));
    };
    std::vector<uint8_t> packed(dense.size() / 2), scales(dense.size() / 16);
    for (size_t b = 0; b < scales.size(); ++b) {
        float m = 0;
        for (int j = 0; j < 16; ++j)
            m = std::max(m, std::abs(dense[b * 16 + j]));
        scales[b] = encode_e4m3(m / (6 * global));
        const float scale = decode_e4m3(scales[b]) * global;
        for (int j = 0; j < 16; j += 2)
            packed[b * 8 + j / 2] = fp4(scale ? dense[b * 16 + j] / scale : 0) |
                                    (fp4(scale ? dense[b * 16 + j + 1] / scale : 0) << 4);
    }
    w.add(base + "weight_packed", "U8", {out, in / 2}, packed.data(), packed.size());
    w.add(base + "weight_scale", "F8_E4M3", {out, in / 16}, scales.data(), scales.size());
    w.add(base + "weight_global_scale", "F32", {}, &global, 4);
}
void add_bf16(::qtip::SafeTensorsWriter &w, const std::string &name, std::vector<int64_t> shape, size_t n,
              std::mt19937 &gen, float mean, float sd) {
    std::normal_distribution<float> normal(mean, sd);
    std::vector<uint16_t> bits(n);
    for (auto &b : bits)
        b = BFloat16(normal(gen)).bits;
    w.add(name, "BF16", shape, bits.data(), n * 2);
}
void write_tokenizer(const fs::path &root);
fs::path write_checkpoint(const fs::path &root, const Geometry &g, unsigned seed) {
    fs::remove_all(root);
    fs::create_directories(root);
    std::mt19937 gen(seed);
    ::qtip::SafeTensorsWriter w;
    add_bf16(w, "model.embed_tokens.weight", {g.vocab, g.hidden}, size_t(g.vocab) * g.hidden, gen, 0, 1);
    add_bf16(w, "lm_head.weight", {g.vocab, g.hidden}, size_t(g.vocab) * g.hidden, gen, 0,
             1.f / std::sqrt(float(g.hidden)));
    add_bf16(w, "model.norm.weight", {g.hidden}, size_t(g.hidden), gen, 1, .1f);
    json layer_formats = json::array();
    for (int l = 0; l < g.layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        add_bf16(w, p + "input_layernorm.weight", {g.hidden}, size_t(g.hidden), gen, 1, .1f);
        add_bf16(w, p + "post_attention_layernorm.weight", {g.hidden}, size_t(g.hidden), gen, 1, .1f);
        const bool fp4 = g.nvfp4_layer1 && l == 1;
        layer_formats.push_back(fp4 ? "nvfp4" : "qtip");
        for (const char *name : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
            auto [in, out] = shape_of(g, name);
            // large q/k and o gains make attention (and therefore RoPE) dominate the logits
            const std::string n = name;
            const float gain = n == "self_attn.q_proj" || n == "self_attn.k_proj" ? 8.f : n == "self_attn.o_proj" ? 4.f : 1.f;
            (fp4 ? add_nvfp4 : add_qtip)(w, p + name + ".", in, out, gen, gain);
        }
    }
    w.write((root / "model-00000.safetensors").string());
    json weight_map = json::object();
    for (const auto &name : w.names())
        weight_map[name] = "model-00000.safetensors";
    std::ofstream(root / "model.safetensors.index.json") << json{{"metadata", json::object()}, {"weight_map", weight_map}}.dump();
    json c = {{"architectures", {"LlamaForCausalLM"}}, {"model_type", "llama"}, {"hidden_size", g.hidden},
              {"intermediate_size", g.inter}, {"num_hidden_layers", g.layers},
              {"num_attention_heads", g.nq}, {"num_key_value_heads", g.nkv}, {"head_dim", g.hd},
              {"vocab_size", g.vocab}, {"hidden_act", "silu"}, {"tie_word_embeddings", false},
              {"rms_norm_eps", 1e-5}, {"rope_theta", g.theta}, {"max_position_embeddings", 131072},
              {"bos_token_id", 260}, {"eos_token_id", g.eos[0]}, {"attention_bias", false}, {"mlp_bias", false},
              {"quip_params", {{"codebook", "bitshift"}, {"codebook_version", 0}, {"L", 16}, {"K", 2}, {"V", 2},
                               {"tlut_bits", 9}, {"decode_mode", "quantlut_sym"}, {"td_x", 16}, {"td_y", 16},
                               {"split_for_tp", false}, {"skip_list", nullptr}}},
              {"qtip_quant", {{"kv_format", "qtip"}, {"layer_formats", layer_formats}}}};
    if (g.llama3)
        // Llama 3.3 factors; original_max_position_embeddings scaled with the toy head_dim so that
        // the scaled, interpolated and untouched frequency bands are all hit by short test sequences.
        c["rope_scaling"] = {{"factor", 8.0}, {"low_freq_factor", 1.0}, {"high_freq_factor", 4.0},
                             {"original_max_position_embeddings", 64}, {"rope_type", "llama3"}};
    else
        c["rope_scaling"] = nullptr;
    std::ofstream(root / "config.json") << c.dump(2);
    std::ofstream(root / "generation_config.json") << json{{"bos_token_id", 260}, {"eos_token_id", g.eos}}.dump();
    if (g.tokenizer)
        write_tokenizer(root);
    return root;
}
// ---------------------------------------------------------------- CPU reference model
float bf(double v) { return float(BFloat16(float(v))); }
struct Reference {
    Checkpoint cp;
    const ModelConfig &c = cp.config;
    bool scaled;
    std::vector<double> hf;
    std::vector<std::vector<float>> keys, values; // [layer][token][kv]
    int position = 0;
    std::vector<std::array<behemoth::qtip::HostLinear, 7>> linears;
    explicit Reference(const fs::path &root, bool use_scaling) : cp(root), scaled(use_scaling && cp.config.rope_llama3) {
        if (scaled)
            hf = hf_llama3(c.dim, c.theta, c.rope);
        keys.resize(size_t(c.layers));
        values.resize(size_t(c.layers));
        const char *names[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
        for (int l = 0; l < c.layers; ++l) {
            linears.emplace_back();
            for (int i = 0; i < 7; ++i)
                linears.back()[size_t(i)] = behemoth::qtip::load_linear(cp, l, names[i]);
        }
    }
    std::vector<float> project(const behemoth::qtip::HostLinear &h, const std::vector<float> &x, bool rz) const {
        std::vector<float> y;
        if (h.is_nvfp4()) {
            constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
            y.resize(size_t(h.out));
            for (int o = 0; o < h.out; ++o) {
                double sum = 0;
                for (int i = 0; i < h.in; ++i) {
                    const size_t at = size_t(o) * h.in + size_t(i);
                    uint8_t code = (h.nvfp4_packed[at / 2] >> (4 * (at & 1))) & 15;
                    float scale = decode_e4m3(h.nvfp4_scales[at / 16]) * h.nvfp4_global;
                    sum += double(bf((code & 8 ? -1.f : 1.f) * levels[code & 7] * scale)) * x[size_t(i)];
                }
                y[size_t(o)] = float(sum);
            }
        } else
            y = behemoth::qtip::reference_forward(h, x, true, rz);
        for (auto &v : y)
            v = bf(v);
        return y;
    }
    std::vector<float> norm(const std::vector<float> &x, const std::string &weight) const {
        auto w = cp.bf16(weight);
        double squares = 0;
        for (float v : x)
            squares += double(v) * v;
        const float inv = float(1 / std::sqrt(squares / double(x.size()) + c.epsilon));
        std::vector<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            out[i] = bf(x[i] * inv * float(w[i]));
        return out;
    }
    void rope(std::vector<float> &v, int heads) const {
        for (int h = 0; h < heads; ++h)
            for (int d = 0; d < c.dim / 2; ++d) {
                double angle;
                if (scaled)
                    angle = double(float(position) * float(hf[size_t(d)]));
                else
                    angle = position * std::pow(double(c.theta), -double(2 * d) / c.dim);
                float &a = v[size_t(h * c.dim + d)], &b = v[size_t(h * c.dim + d + c.dim / 2)];
                const double na = a * std::cos(angle) - b * std::sin(angle),
                             nb = b * std::cos(angle) + a * std::sin(angle);
                a = bf(na);
                b = bf(nb);
            }
    }
    std::vector<float> step(int token) {
        auto embed = cp.bf16("model.embed_tokens.weight");
        std::vector<float> x(static_cast<size_t>(c.hidden));
        for (int i = 0; i < c.hidden; ++i)
            x[size_t(i)] = float(embed[size_t(token) * c.hidden + size_t(i)]);
        const int kv = c.kv_heads * c.dim, group = c.q_heads / c.kv_heads;
        for (int l = 0; l < c.layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            auto &lin = linears[size_t(l)];
            auto h = norm(x, p + "input_layernorm.weight");
            auto q = project(lin[0], h, false), k = project(lin[1], h, false), v = project(lin[2], h, false);
            rope(q, c.q_heads);
            rope(k, c.kv_heads);
            keys[size_t(l)].insert(keys[size_t(l)].end(), k.begin(), k.end());
            values[size_t(l)].insert(values[size_t(l)].end(), v.begin(), v.end());
            const int n = position + 1;
            std::vector<float> attended(static_cast<size_t>(c.q_heads * c.dim));
            for (int head = 0; head < c.q_heads; ++head) {
                const int kh = head / group;
                std::vector<double> score(static_cast<size_t>(n));
                double top = -INFINITY, total = 0;
                for (int t = 0; t < n; ++t) {
                    double s = 0;
                    for (int d = 0; d < c.dim; ++d)
                        s += double(q[size_t(head * c.dim + d)]) * keys[size_t(l)][size_t(t * kv + kh * c.dim + d)];
                    score[size_t(t)] = s / std::sqrt(double(c.dim));
                    top = std::max(top, score[size_t(t)]);
                }
                for (auto &s : score) {
                    s = std::exp(s - top);
                    total += s;
                }
                for (auto &s : score)
                    s = bf(s / total);
                for (int d = 0; d < c.dim; ++d) {
                    double s = 0;
                    for (int t = 0; t < n; ++t)
                        s += score[size_t(t)] * values[size_t(l)][size_t(t * kv + kh * c.dim + d)];
                    attended[size_t(head * c.dim + d)] = bf(s);
                }
            }
            auto o = project(lin[3], attended, false);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = bf(x[i] + o[i]);
            h = norm(x, p + "post_attention_layernorm.weight");
            auto gate = project(lin[4], h, true), up = project(lin[5], h, true);
            for (size_t i = 0; i < gate.size(); ++i)
                gate[i] = bf(gate[i] / (1 + std::exp(-gate[i])) * up[i]);
            auto down = project(lin[6], gate, false);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = bf(x[i] + down[i]);
        }
        auto normed = norm(x, "model.norm.weight");
        auto head = cp.bf16("lm_head.weight");
        std::vector<float> logits(static_cast<size_t>(c.vocab));
        for (int v = 0; v < c.vocab; ++v) {
            double s = 0;
            for (int i = 0; i < c.hidden; ++i)
                s += double(float(head[size_t(v) * c.hidden + size_t(i)])) * normed[size_t(i)];
            logits[size_t(v)] = float(s);
        }
        ++position;
        return logits;
    }
};
double relative(const std::vector<float> &a, const std::vector<float> &b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]), "nonfinite CUDA logits");
        e += std::pow(double(a[i]) - b[i], 2);
        n += double(b[i]) * b[i];
    }
    return std::sqrt(e / std::max(n, 1e-30));
}
// returns {relative L2 of the scaled reference, of the unscaled reference} at the last token
std::pair<double, double> compare(const fs::path &root, int tokens, const std::string &label) {
    cu::Model model(root, 0, 128, 1, 1);
    Reference scaled(root, true), plain(root, false);
    const auto &c = scaled.c;
    require(model.config.vocab == c.vocab && model.config.eos == c.eos, "model config must follow config.json");
    std::mt19937 gen(77);
    double worst = 0, plain_last = 0, scaled_last = 0;
    for (int t = 0; t < tokens; ++t) {
        const int token = int(gen() % unsigned(c.vocab));
        auto actual = model.step(token);
        auto expected = scaled.step(token);
        auto other = plain.step(token);
        scaled_last = relative(actual, expected);
        plain_last = relative(actual, other);
        worst = std::max(worst, scaled_last);
        require(actual.size() == size_t(c.vocab), "logit count");
        require(scaled_last < 0.02, label + ": CUDA/CPU logits mismatch");
    }
    // prefill must reproduce the last stepwise logits
    std::mt19937 again(77);
    std::vector<int> ids;
    for (int t = 0; t < tokens; ++t)
        ids.push_back(int(again() % unsigned(c.vocab)));
    auto prefilled = model.prefill(ids);
    Reference check(root, true);
    std::vector<float> last;
    for (int id : ids)
        last = check.step(id);
    require(relative(prefilled, last) < 0.02, label + ": prefill mismatch");
    std::cout << label << ": layers " << c.layers << " hidden " << c.hidden << " ffn " << c.intermediate << " heads "
              << c.q_heads << '/' << c.kv_heads << " hd " << c.dim << " vocab " << c.vocab << " rope "
              << (c.rope_llama3 ? "llama3" : "plain") << ": worst relative L2 " << worst << " (unscaled-rope reference "
              << plain_last << ")\n";
    return {scaled_last, plain_last};
}

// ---------------------------------------------------------------- Llama 3 style tokenizer + chat template
std::string byte_unicode(int c) {
    std::string out;
    if (c < 0x80)
        out += char(c);
    else {
        out += char(0xc0 | (c >> 6));
        out += char(0x80 | (c & 63));
    }
    return out;
}
const char *kLlama3Template = R"(
{{- bos_token }}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
)";
void write_tokenizer(const fs::path &root) {
    // 256 byte tokens (ids = byte values, GPT-2 byte-to-unicode mapping), a few merges, Llama 3 specials
    std::vector<std::string> mapped(256);
    int next = 256;
    for (int b = 0; b < 256; ++b)
        mapped[size_t(b)] = byte_unicode((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 ? b : next++);
    json vocab = json::object();
    for (int b = 0; b < 256; ++b)
        vocab[mapped[size_t(b)]] = b;
    vocab["he"] = 256;
    vocab["ll"] = 257;
    vocab["hell"] = 258;
    vocab["hello"] = 259;
    json added = json::array(), decoder = json::object();
    const char *specials[] = {"<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                              "<|eot_id|>"};
    for (int i = 0; i < 5; ++i) {
        json t = {{"id", 260 + i}, {"content", specials[i]}, {"single_word", false}, {"lstrip", false},
                  {"rstrip", false}, {"normalized", false}, {"special", true}};
        added.push_back(t);
        decoder[std::to_string(260 + i)] = t;
    }
    const char *regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    json tokenizer = {{"version", "1.0"}, {"added_tokens", added}, {"normalizer", nullptr},
                      {"pre_tokenizer",
                       {{"type", "Sequence"},
                        {"pretokenizers",
                         {{{"type", "Split"}, {"pattern", {{"Regex", regex}}}, {"behavior", "Isolated"}, {"invert", false}},
                          {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}}}}}},
                      {"decoder", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}}},
                      {"model", {{"type", "BPE"}, {"ignore_merges", true}, {"vocab", vocab},
                                 {"merges", {"h e", "l l", "he ll", "hell o"}}}}};
    std::ofstream(root / "tokenizer.json") << tokenizer.dump();
    json config = {{"bos_token", "<|begin_of_text|>"}, {"eos_token", "<|eot_id|>"},
                   {"added_tokens_decoder", decoder}, {"chat_template", kLlama3Template},
                   {"model_max_length", 131072}};
    std::ofstream(root / "tokenizer_config.json") << config.dump();
}
void tokenizer_test(const fs::path &root) {
    cu::Tokenizer tokenizer(root);
    using Ids = std::vector<int>;
    require(tokenizer.encode("hello") == Ids{259}, "BPE merges / ignore_merges");
    require(tokenizer.encode("hello 123456") == (Ids{259, ' ', '1', '2', '3', '4', '5', '6'}),
            "Llama 3 pre-tokenizer regex (digits in groups of 3, leading space on words)") ;
    require(tokenizer.encode("<|eot_id|>hello<|end_of_text|>") == (Ids{264, 259, 261}), "special token matching");
    require(tokenizer.decode({259, 32, 'x'}) == "hello x", "byte-level decode");
    require(tokenizer.decode({260, 259, 264}) == "hello" && tokenizer.decode({260, 259, 264}, true) ==
                                                              "<|begin_of_text|>hello<|eot_id|>",
            "special token decode");
    char date[32];
    std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    std::strftime(date, sizeof date, "%d %b %Y", &local);
    const std::string prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                               "Cutting Knowledge Date: December 2023\nToday Date: " +
                               std::string(date) + "\n\n";
    auto rendered = tokenizer.chat(json::array({{{"role", "system"}, {"content", "  Be brief. "}},
                                                {{"role", "user"}, {"content", "hello"}},
                                                {{"role", "assistant"}, {"content", "hello"}},
                                                {{"role", "user"}, {"content", "hello"}}}));
    require(rendered == prefix + "Be brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
                                 "<|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>"
                                 "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
                                 "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "Llama 3 chat template rendering:\n" + rendered);
    auto no_system = tokenizer.chat(json::array({{{"role", "user"}, {"content", "hello"}}}));
    require(no_system == prefix + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
                                  "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "chat template without system message:\n" + no_system);
    auto ids = tokenizer.encode(no_system);
    require(ids.front() == 260 && ids[1] == 262 && std::count(ids.begin(), ids.end(), 264) == 2 &&
                ids[ids.size() - 3] == 263 && ids.back() == '\n',
            "header/eot special ids in the encoded prompt");
    require(tokenizer.decode(ids, true) == no_system, "chat prompt round trip");
    std::cout << "Llama 3 tokenizer/chat template (synthetic tokenizer.json) OK\n";
}

void rejection_test(const fs::path &root) {
    std::ifstream in(root / "config.json");
    const json good = json::parse(in);
    auto rejected = [&](const char *what, auto &&mutate) {
        json c = good;
        mutate(c);
        fs::create_directories(root / "bad");
        const fs::path file = root / "bad" / "config.json";
        std::ofstream(file) << c.dump();
        try {
            ModelConfig::read(file);
        } catch (const std::invalid_argument &) {
            return;
        }
        throw std::runtime_error(std::string("config must be rejected: ") + what);
    };
    (void)ModelConfig::read(root / "config.json");
    rejected("yarn rope", [](json &c) { c["rope_scaling"] = {{"rope_type", "yarn"}, {"factor", 4.0}}; });
    rejected("hidden 8200 (no Hadamard factor)", [](json &c) { c["hidden_size"] = 8200; });
    rejected("hidden not multiple of 32", [](json &c) { c["hidden_size"] = 100; });
    rejected("pow2 part > 8192", [](json &c) { c["intermediate_size"] = 28 * 16384; });
    rejected("tied embeddings", [](json &c) { c["tie_word_embeddings"] = true; });
    rejected("sliding window", [](json &c) { c["sliding_window"] = 4096; });
    rejected("eos outside vocabulary", [](json &c) { c["eos_token_id"] = 99999; c.erase("rope_scaling"); });
    fs::remove_all(root / "bad");
    std::cout << "unsupported geometries are rejected with errors\n";
}

// Quantizer output vs its BF16 source: layer 0 projections applied to random inputs (the quantized
// weights, decoded by the runtime reference, must approximate the source weights).
void fidelity(const fs::path &checkpoint, const fs::path &source, double limit) {
    ::qtip::SafeTensorsReader reader(source.string());
    Reference ref(checkpoint, true);
    const char *names[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
    std::mt19937 gen(5);
    std::normal_distribution<float> normal(0, 1);
    for (int i = 0; i < 7; ++i) {
        const auto &h = ref.linears[0][size_t(i)];
        const std::string name = std::string("model.layers.0.") + names[i] + ".weight";
        auto raw = reader.read(name);
        require(raw.size() == size_t(h.in) * h.out * 2, "source weight shape " + name);
        std::vector<float> x(size_t(h.in));
        for (auto &v : x)
            v = float(BFloat16(normal(gen)));
        auto y = ref.project(h, x, false);
        double e = 0, n = 0;
        for (int o = 0; o < h.out; ++o) {
            double sum = 0;
            for (int k = 0; k < h.in; ++k) {
                uint16_t b;
                std::memcpy(&b, raw.data() + (size_t(o) * h.in + size_t(k)) * 2, 2);
                sum += double(std::bit_cast<float>(uint32_t(b) << 16)) * x[size_t(k)];
            }
            e += std::pow(y[size_t(o)] - sum, 2);
            n += sum * sum;
        }
        const double relative = std::sqrt(e / n);
        std::cout << "fidelity " << checkpoint.filename().string() << " " << names[i] << " relative L2 " << relative << '\n';
        require(relative < limit, std::string("quantized projection too far from the BF16 source: ") + names[i]);
    }
}
} // namespace
int main(int argc, char **argv) {
    int count = 0;
    auto status = cg::device_count(&count);
    if (status == cg::no_device || status == cg::insufficient_driver || (status == cg::success && !count))
        return 77;
    cg::Stream s{};
    try {
        cg::checked(status, "GPU discovery");
        cg::checked(cg::stream_create(&s), "test stream");
        cu::initialize_tables();
        inv_freq_test();
        rope_kernel_test(s);
        cg::checked(cg::stream_destroy(s), "test stream shutdown");
        s = nullptr;
        const fs::path tmp = fs::temp_directory_path() / "behemoth_llama_tests";
        write_checkpoint(tmp / "llama_qtip", Geometry{}, 1);
        tokenizer_test(tmp / "llama_qtip");
        rejection_test(tmp / "llama_qtip");
        // Llama-3.3-geometry-scaled: GQA 8:1, FFN 3.5x (28*2^p Hadamard factor), llama3 rope_scaling
        auto [good, bad] = compare(write_checkpoint(tmp / "llama_qtip", Geometry{}, 1), 24, "synthetic Llama QTIP");
        require(bad > 2 * good && bad > 0.03,
                "the test must be sensitive to llama3 RoPE scaling (unscaled reference should differ)");
        Geometry mixed;
        mixed.nvfp4_layer1 = true;
        mixed.layers = 3;
        compare(write_checkpoint(tmp / "llama_mixed", mixed, 2), 16, "synthetic Llama QTIP+NVFP4 (adaptive layout)");
        Geometry plain;
        plain.llama3 = false;
        plain.theta = 1000000;
        plain.eos = {2};
        plain.tokenizer = false;
        compare(write_checkpoint(tmp / "llama_plain", plain, 3), 16, "synthetic plain-RoPE llama geometry");
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == "--checkpoint")
                compare(argv[i + 1], 12, std::string("checkpoint ") + argv[i + 1]);
        for (int i = 1; i + 2 < argc; ++i)
            if (std::string(argv[i]) == "--fidelity") // --fidelity CHECKPOINT SOURCE LIMIT
                fidelity(argv[i + 1], argv[i + 2], std::stod(argv[i + 3]));
        fs::remove_all(tmp);
        std::cout << "PASS Llama 3.3 geometry, llama3 RoPE scaling and end-to-end CUDA vs CPU reference\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (s)
            (void)cg::stream_destroy(s);
        return 1;
    }
}
