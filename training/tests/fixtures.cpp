#include "common.hpp"
#include "safetensors.hpp"
#include "train.hpp"
#include <cmath>
#include <fstream>
#include <random>

namespace qat {
void write_tiny_model(const fs::path &p, int layers) {
    fs::create_directories(p);
    json c = {{"hidden_size", 64},
              {"intermediate_size", 96},
              {"num_attention_heads", 4},
              {"num_key_value_heads", 2},
              {"head_dim", 16},
              {"num_hidden_layers", layers},
              {"vocab_size", 96},
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
    durable_json(p / "config.json", c);
    qtip::SafeTensorsWriter w;
    json index = {{"weight_map", json::object()}};
    auto add = [&](std::string name, std::string type, std::vector<int64_t> shape, const void *data,
                   size_t bytes) {
        w.add(name, type, shape, data, bytes);
        index["weight_map"][name] = "model.safetensors";
    };
    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0, .13f);
    auto dense = [&](std::string name, int r, int cols, bool norm) {
        std::vector<uint16_t> v(size_t(r) * cols);
        for (auto &x : v)
            x = behemoth::BFloat16(norm ? 1.f : normal(rng)).bits;
        add(name, "BF16", cols == 1 ? std::vector<int64_t>{r} : std::vector<int64_t>{r, cols}, v.data(),
            v.size() * 2);
    };
    for (int l = 0; l < layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".";
        std::vector<uint16_t> lut(1024);
        for (auto &x : lut)
            x = qtip::f32_to_fp16(normal(rng));
        for (int i = 0; i < 7; ++i) {
            int out = i == 1 || i == 2 ? 32 : i == 4 || i == 5 ? 96 : 64, in = i == 6 ? 96 : 64;
            std::string b = prefix + linears[i] + ".";
            std::vector<uint16_t> codes(size_t(out) * in / 8);
            for (auto &x : codes)
                x = uint16_t(rng());
            add(b + "trellis", "I16", {out / 16 * (in / 16), 32}, codes.data(), codes.size() * 2);
            add(b + "tlut", "F16", {512, 2}, lut.data(), lut.size() * 2);
            std::vector<uint16_t> su(in);
            for (int j = 0; j < in; ++j)
                su[j] = behemoth::BFloat16(j % 3 == 0 ? -1.f : 1.f).bits;
            add(b + "SU", "BF16", {in}, su.data(), su.size() * 2);
            std::vector<float> sv(out, .4f);
            for (int j = 0; j < out; ++j)
                if (j % 2)
                    sv[j] *= -1;
            add(b + "SV", "F32", {out}, sv.data(), sv.size() * 4);
            int64_t rcp = 0, tp = 8;
            add(b + "rcp", "I64", {}, &rcp, 8);
            add(b + "tp_rank", "I64", {}, &tp, 8);
        }
        dense(prefix + norm1, 64, 1, true);
        dense(prefix + norm2, 64, 1, true);
    }
    dense("model.norm.weight", 64, 1, true);
    dense("model.embed_tokens.weight", 96, 64, false);
    dense("lm_head.weight", 96, 64, false);
    w.write((p / "model.safetensors").string());
    durable_json(p / "model.safetensors.index.json", index);
}
void write_tiny_data(const fs::path &p, int sequences) {
    fs::create_directories(p);
    json m = {{"format_version", 1},         {"split", "train"},
              {"nseq", sequences},           {"seqlen", 5},
              {"files", json::object()},     {"training_conversations", {0, 1}},
              {"holdout_conversations", {2}}};
    std::vector<int32_t> tokens, segments, positions;
    std::vector<uint8_t> loss;
    for (int s = 0; s < sequences; ++s)
        for (int i = 0; i < 5; ++i) {
            tokens.push_back(3 + (s + i) % 9);
            segments.push_back(i < 3 ? 0 : 1);
            positions.push_back(i < 3 ? i : i - 3);
            loss.push_back(i != 0);
        }
    auto write = [&](const char *n, const void *d, size_t bytes) {
        std::ofstream f(p / n, std::ios::binary);
        f.write(static_cast<const char *>(d), bytes);
        f.close();
        m["files"][n] = sha256(p / n);
    };
    write("tokens.i32", tokens.data(), tokens.size() * 4);
    write("segments.i32", segments.data(), segments.size() * 4);
    write("positions.i32", positions.data(), positions.size() * 4);
    write("loss.u8", loss.data(), loss.size());
    durable_json(p / "metadata.json", m);
}
} // namespace qat
