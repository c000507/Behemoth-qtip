// Full-model Llama reference for indexed Behemoth QTIP checkpoints (adaptive
// QTIP/FP8/NVFP4 layers). Mirrors neuron/llama.py's BF16 rounding points in
// double precision; with --fp8-kv attention switches to the ROCm runtime's
// FP8 KV-cache arithmetic (quantized Q/K/V and probabilities, 32-token tiles).
//
// usage: model_oracle MODEL_DIR OUT_DIR TOKEN [TOKEN...] [--fp8-kv]
// Writes OUT_DIR/logits_<i>.bin (F32 [vocab]) for prompt token i and prints
// the greedy argmax after the last prompt token.
#include "behemoth/attention.hpp"
#include "behemoth/checkpoint.hpp"
#include "behemoth/qtip.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using behemoth::BFloat16;
using behemoth::Checkpoint;
using behemoth::ModelConfig;

float bf(double x) { return float(BFloat16(float(x))); }

template <typename T> void save(const std::filesystem::path &path, const std::vector<T> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
    if (!f) throw std::runtime_error("Failed to write oracle output");
}

std::vector<float> float_tensor(const Checkpoint &cp, const std::string &name, size_t expect) {
    auto v = cp.tensor(name).floats(name);
    if (v.size() != expect)
        throw std::invalid_argument("Unexpected tensor size: " + name);
    std::vector<float> out(expect);
    for (size_t i = 0; i < expect; ++i)
        out[i] = v[i];
    return out;
}

// NVFP4 forward matching the GPU gemv: dequantized weights (E2M1 x E4M3 block
// scale x global scale) rounded to BF16, BF16-rounded input, fp64 accumulation.
std::vector<float> nvfp4_forward(const behemoth::qtip::HostLinear &l, std::span<const float> x) {
    if (x.size() != size_t(l.in))
        throw std::invalid_argument("NVFP4 input width");
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    std::vector<float> y(size_t(l.out));
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < l.out; ++o) {
        const uint8_t *row = l.nvfp4_packed.data() + size_t(o) * size_t(l.in / 2);
        const uint8_t *scales = l.nvfp4_scales.data() + size_t(o) * size_t(l.in / 16);
        double sum = 0;
        for (int i = 0; i < l.in; ++i) {
            const uint8_t byte = row[size_t(i / 2)];
            const int code = (i % 2) ? byte >> 4 : byte & 15;
            const double w = double(lut[code]) * double(behemoth::decode_e4m3(scales[size_t(i / 16)])) *
                             double(l.nvfp4_global);
            sum += double(bf(x[size_t(i)])) * double(bf(w));
        }
        y[size_t(o)] = float(sum);
    }
    return y;
}

std::vector<float> project(const Checkpoint &cp, int layer, const std::string &name,
                           std::span<const float> x, bool round_z) {
    auto l = behemoth::qtip::load_linear(cp, layer, name);
    if (l.is_nvfp4()) {
        if (round_z) throw std::invalid_argument("NVFP4 round_z not used");
        return nvfp4_forward(l, x);
    }
    return behemoth::qtip::reference_forward(l, x, true, round_z);
}

// BF16 input/output RMSNorm with fp64 squares (runtime reference semantics).
std::vector<float> norm(const std::vector<float> &x, const std::vector<float> &weights, float eps) {
    double squares = 0;
    for (float v : x)
        squares += double(v) * double(v);
    const double factor = 1 / std::sqrt(squares / double(x.size()) + double(eps));
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = bf(double(x[i]) * factor * double(weights[i]));
    return out;
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: model_oracle MODEL_DIR OUT_DIR TOKEN [TOKEN...] [--fp8-kv]\n";
        return 2;
    }
    std::filesystem::path root = argv[1], out_dir = argv[2];
    std::vector<int> tokens;
    bool fp8_kv = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--fp8-kv")
            fp8_kv = true;
        else
            tokens.push_back(std::stoi(a));
    }
    std::filesystem::create_directories(out_dir);
    Checkpoint cp(root);
    const ModelConfig &c = cp.config;
    std::cout << "model layers=" << c.layers << " hidden=" << c.hidden << " heads=" << c.q_heads
              << " kv_heads=" << c.kv_heads << " dim=" << c.dim << " vocab=" << c.vocab
              << " fp8_kv=" << fp8_kv << '\n';
    const int heads = c.q_heads, kvheads = c.kv_heads, dim = c.dim, group = heads / kvheads;

    std::vector<float> freq(size_t(dim / 2));
    if (c.rope_llama3) {
        freq = behemoth::llama3_inv_freq(dim, c.theta, c.rope);
    } else {
        for (int d = 0; d < dim / 2; ++d)
            freq[size_t(d)] = float(std::pow(double(c.theta), -2.0 * d / dim));
    }

    const size_t embed_n = size_t(c.vocab) * size_t(c.hidden);
    auto embed = float_tensor(cp, "model.embed_tokens.weight", embed_n);
    auto head_w = float_tensor(cp, "lm_head.weight", embed_n);
    auto final_norm = float_tensor(cp, "model.norm.weight", size_t(c.hidden));

    // Per layer: BF16-valued K/V cache (or FP8 entries + scales with --fp8-kv).
    struct LayerCache {
        std::vector<float> k, v;                    // [t][kvheads][dim] bf16 values
        std::vector<uint8_t> kq, vq;                // [t][kvheads][dim] e4m3
        std::vector<float> kq_scale, vq_scale;      // [t][kvheads]
    };
    std::vector<LayerCache> cache(size_t(c.layers));
    for (int token_index = 0; token_index < int(tokens.size()); ++token_index) {
        const int token = tokens[size_t(token_index)];
        if (token < 0 || token >= c.vocab)
            throw std::invalid_argument("Token id out of range");
        std::vector<float> state(size_t(c.hidden));
        for (int i = 0; i < c.hidden; ++i)
            state[size_t(i)] = float(embed[size_t(token) * size_t(c.hidden) + size_t(i)]);
        std::vector<float> cos(size_t(dim / 2)), sin(size_t(dim / 2));
        for (int d = 0; d < dim / 2; ++d) {
            const double angle = double(freq[size_t(d)]) * token_index;
            cos[size_t(d)] = float(std::cos(angle));
            sin[size_t(d)] = float(std::sin(angle));
        }
        for (int layer = 0; layer < c.layers; ++layer) {
            const std::string p = "model.layers." + std::to_string(layer) + ".";
            auto normalized = norm(state, float_tensor(cp, p + "input_layernorm.weight",
                                                       size_t(c.hidden)), c.epsilon);
            const int q_width = heads * dim, kv_width = kvheads * dim;
            auto q = project(cp, layer, "self_attn.q_proj", normalized, false);
            auto k = project(cp, layer, "self_attn.k_proj", normalized, false);
            auto v = project(cp, layer, "self_attn.v_proj", normalized, false);
            for (auto *vec : {&q, &k, &v})
                for (float &value : *vec)
                    value = bf(value);
            auto rotary = [&](std::vector<float> &x, int row_width) {
                for (int r = 0; r < int(x.size()) / row_width; ++r)
                    for (int d = 0; d < row_width / 2; ++d) {
                        const float a = x[size_t(r) * row_width + d],
                                    b = x[size_t(r) * row_width + row_width / 2 + d];
                        x[size_t(r) * row_width + d] =
                            bf(double(a) * cos[size_t(d)] - double(b) * sin[size_t(d)]);
                        x[size_t(r) * row_width + row_width / 2 + d] =
                            bf(double(b) * cos[size_t(d)] + double(a) * sin[size_t(d)]);
                    }
            };
            rotary(q, q_width);
            rotary(k, kv_width);
            LayerCache &lc = cache[size_t(layer)];
            const int time = int(lc.k.size()) / size_t(kv_width);
            lc.k.resize(lc.k.size() + kv_width);
            lc.v.resize(lc.v.size() + kv_width);
            std::copy(k.begin(), k.end(), lc.k.begin() + size_t(time) * kv_width);
            std::copy(v.begin(), v.end(), lc.v.begin() + size_t(time) * kv_width);
            if (fp8_kv) {
                lc.kq.resize(lc.k.size());
                lc.vq.resize(lc.v.size());
                lc.kq_scale.resize(lc.kq_scale.size() + kvheads);
                lc.vq_scale.resize(lc.vq_scale.size() + kvheads);
                for (int h = 0; h < kvheads; ++h) {
                    auto kq = behemoth::quantize(std::span<const float>(lc.k.data() + size_t(time) * kv_width + size_t(h) * dim, dim));
                    auto vq = behemoth::quantize(std::span<const float>(lc.v.data() + size_t(time) * kv_width + size_t(h) * dim, dim));
                    std::copy(kq.values.begin(), kq.values.end(),
                              lc.kq.begin() + size_t(time) * kv_width + size_t(h) * dim);
                    std::copy(vq.values.begin(), vq.values.end(),
                              lc.vq.begin() + size_t(time) * kv_width + size_t(h) * dim);
                    lc.kq_scale[size_t(time) * kvheads + h] = kq.scale;
                    lc.vq_scale[size_t(time) * kvheads + h] = vq.scale;
                }
            }
            std::vector<float> attn(q_width, 0.f);
            for (int h = 0; h < heads; ++h) {
                const int kh = h / group;
                const double scale = 1 / std::sqrt(double(dim));
                std::vector<double> scores(size_t(time + 1));
                double maximum = -INFINITY;
                for (int t = 0; t <= time; ++t) {
                    double dot = 0;
                    if (fp8_kv) {
                        const size_t loc = size_t(t) * kv_width + size_t(kh) * dim;
                        for (int d = 0; d < dim; ++d)
                            dot += double(q[size_t(h) * dim + d]) *
                                   double(behemoth::decode_e4m3(lc.kq[loc + d]));
                        dot *= double(lc.kq_scale[size_t(t) * kvheads + kh]);
                    } else {
                        for (int d = 0; d < dim; ++d)
                            dot += double(q[size_t(h) * dim + d]) *
                                   double(lc.k[size_t(t) * kv_width + size_t(kh) * dim + d]);
                    }
                    scores[size_t(t)] = dot * scale;
                    maximum = std::max(maximum, scores[size_t(t)]);
                }
                std::vector<double> out(dim, 0.0), probability(size_t(time + 1));
                double denominator = 0;
                for (int t = 0; t <= time; ++t) {
                    probability[size_t(t)] = std::exp(scores[size_t(t)] - maximum);
                    denominator += probability[size_t(t)];
                }
                if (fp8_kv) {
                    // Runtime FP8 P: quantize p*v_scale per 32-token tile with its own scale.
                    for (int start = 0; start <= time; start += behemoth::key_tile) {
                        const int count = std::min(behemoth::key_tile, time + 1 - start);
                        float weighted_max = 0;
                        for (int j = 0; j < count; ++j) {
                            const double p = probability[size_t(start + j)] *
                                             double(lc.vq_scale[size_t(start + j) * kvheads + kh]);
                            probability[size_t(start + j)] = p;
                            weighted_max = std::max(weighted_max, float(p));
                        }
                        const float ps = weighted_max == 0 ? 1.f
                            : std::max(weighted_max / 448.f, std::numeric_limits<float>::min());
                        for (int j = 0; j < count; ++j) {
                            const double w = double(behemoth::decode_e4m3(
                                behemoth::encode_e4m3(float(probability[size_t(start + j)]) / ps))) * ps;
                            const size_t loc = size_t(start + j) * kv_width + size_t(kh) * dim;
                            for (int d = 0; d < dim; ++d)
                                out[size_t(d)] += w * double(behemoth::decode_e4m3(lc.vq[loc + d]));
                        }
                    }
                } else {
                    for (int t = 0; t <= time; ++t) {
                        const double p = probability[size_t(t)] / denominator;
                        for (int d = 0; d < dim; ++d)
                            out[size_t(d)] += p * double(lc.v[size_t(t) * kv_width + size_t(kh) * dim + d]);
                    }
                }
                for (int d = 0; d < dim; ++d)
                    attn[size_t(h) * dim + d] = bf(out[size_t(d)] / denominator);
            }
            auto update = project(cp, layer, "self_attn.o_proj", attn, false);
            for (int i = 0; i < c.hidden; ++i)
                state[size_t(i)] = bf(double(state[size_t(i)]) + double(bf(update[size_t(i)])));
            normalized = norm(state, float_tensor(cp, p + "post_attention_layernorm.weight",
                                                  size_t(c.hidden)), c.epsilon);
            auto gate = project(cp, layer, "mlp.gate_proj", normalized, true);
            auto up = project(cp, layer, "mlp.up_proj", normalized, true);
            std::vector<float> act(size_t(gate.size()));
            for (size_t i = 0; i < gate.size(); ++i) {
                const double g = double(bf(gate[i]));
                act[i] = bf(g / (1 + std::exp(-g)) * double(bf(up[i])));
            }
            auto down = project(cp, layer, "mlp.down_proj", act, false);
            for (int i = 0; i < c.hidden; ++i)
                state[size_t(i)] = bf(double(state[size_t(i)]) + double(bf(down[size_t(i)])));
            std::cout << "token " << token_index << " layer " << layer << " done\n";
        }
        auto normalized = norm(state, final_norm, c.epsilon);
        std::vector<float> logits(size_t(c.vocab));
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < c.vocab; ++o) {
            double sum = 0;
            for (int i = 0; i < c.hidden; ++i)
                sum += double(bf(normalized[size_t(i)])) *
                       double(head_w[size_t(o) * size_t(c.hidden) + size_t(i)]);
            logits[size_t(o)] = float(sum);
        }
        save(out_dir / ("logits_" + std::to_string(token_index) + ".bin"), logits);
        const int argmax = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::cout << "token " << token_index << " (id " << token << ") argmax " << argmax << " max "
                  << *std::max_element(logits.begin(), logits.end()) << '\n';
    }
}
