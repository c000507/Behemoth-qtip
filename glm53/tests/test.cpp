#include "glm53.hpp"
#include "safetensors.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <unistd.h>
using namespace glm53;
namespace {
json tiny(bool dense = false) {
    return {
        {"model_type", "glm5_next"},
        {"text_config",
         {{"model_type", "glm5_next_text"},
          {"hidden_size", 16},
          {"intermediate_size", 32},
          {"num_hidden_layers", 4},
          {"vocab_size", 32},
          {"num_attention_heads", 2},
          {"num_key_value_heads", 2},
          {"q_lora_rank", 8},
          {"kv_lora_rank", 4},
          {"qk_nope_head_dim", 4},
          {"qk_rope_head_dim", 0},
          {"v_head_dim", 4},
          {"mla_use_nope", true},
          {"linear_attn_config",
           {{"num_heads", 2}, {"head_dim", 4}, {"short_conv_kernel_size", 3}, {"gate_lower_bound", -5.f}}},
          {"hc_mult", 4},
          {"hc_sinkhorn_iters", 20},
          {"hc_eps", 1e-6},
          {"rms_norm_eps", 1e-5},
          {"mhc", true},
          {"index_n_heads", 2},
          {"index_head_dim", 4},
          {"index_kpool", 2},
          {"index_topk", 4},
          {"index_kpool_always_select_tail", true},
          {"layer_types",
           {"linear_attention", "deepseek_sparse_attention", "linear_attention",
            "deepseek_sparse_attention"}},
          {"mlp_layer_types",
           dense ? json({"dense", "dense", "dense", "dense"}) : json({"dense", "sparse", "dense", "sparse"})},
          {"indexer_types", {"full", "full", "full", "full"}},
          {"n_routed_experts", 4},
          {"num_experts_per_tok", 2},
          {"moe_intermediate_size", 16},
          {"n_shared_experts", 1},
          {"n_group", 2},
          {"topk_group", 1},
          {"norm_topk_prob", true},
          {"routed_scaling_factor", 2.5},
          {"max_position_embeddings", 128},
          {"eos_token_id", {30, 31}},
          {"bos_token_id", 1},
          {"swiglu_limit", 10.f}}}};
}
void fixture(const fs::path &path, bool dense, bool server = false) {
    fs::create_directories(path);
    auto cfg = tiny(dense);
    if (dense) {
        cfg["text_config"]["q_lora_rank"] = 16;
        cfg["text_config"]["qk_nope_head_dim"] = 6;
        cfg["text_config"]["v_head_dim"] = 8;
        cfg["text_config"]["num_attention_heads"] = 3;
        cfg["text_config"]["num_key_value_heads"] = 3;
        cfg["text_config"]["linear_attn_config"]["num_heads"] = 3;
    }
    if (server) {
        cfg["text_config"]["vocab_size"] = 263;
        cfg["text_config"]["eos_token_id"] = {262};
    }
    {
        std::ofstream f(path / "config.json");
        f << cfg;
    }
    Config c(cfg);
    qtip::SafeTensorsWriter w;
    json map = json::object();
    std::mt19937 rng(41);
    std::normal_distribution<float> normal(0, .08f);
    auto tensor = [&](const std::string &name, std::vector<int64_t> shape, float fill = NAN) {
        size_t n = 1;
        for (auto d : shape)
            n *= size_t(d);
        Vec v(n);
        for (float &a : v)
            a = std::isnan(fill) ? normal(rng) : fill;
        w.add(name, "F32", shape, v.data(), v.size() * 4);
        map[name] = "model.safetensors";
    };
    auto linear = [&](const std::string &name, int r, int col) { tensor(name + ".weight", {r, col}); };
    std::string pre = "model.language_model.";
    tensor(pre + "embed_tokens.weight", {c.vocab, c.hidden});
    tensor(pre + "norm.weight", {c.hidden}, 1);
    linear("lm_head", c.vocab, c.hidden);
    for (int l = 0; l < c.layers; ++l) {
        std::string p = pre + "layers." + std::to_string(l) + ".";
        for (auto hc : {"hc_attn_", "hc_ffn_"}) {
            tensor(p + hc + "fn", {24, 4 * c.hidden});
            tensor(p + hc + "base", {24}, 0);
            tensor(p + hc + "scale", {3}, .1);
        }
        tensor(p + "input_layernorm.weight", {c.hidden}, 1);
        tensor(p + "post_attention_layernorm.weight", {c.hidden}, 1);
        std::string a = p + "self_attn.";
        if (c.attention[l] == "linear_attention") {
            int width = c.linear_heads * c.linear_dim;
            for (auto kind : {"q", "k", "v"}) {
                linear(a + kind + "_proj", width, c.hidden);
                tensor(a + kind + "_conv1d.weight", {width, 1, c.conv});
            }
            linear(a + "f_a_proj", c.linear_dim, c.hidden);
            linear(a + "f_b_proj", width, c.linear_dim);
            linear(a + "b_proj", c.linear_heads, c.hidden);
            linear(a + "g_a_proj", c.linear_dim, c.hidden);
            linear(a + "g_b_proj", width, c.linear_dim);
            linear(a + "o_proj", c.hidden, width);
            tensor(a + "A_log", {c.linear_heads}, 0);
            tensor(a + "dt_bias", {width}, 0);
            tensor(a + "o_norm.weight", {c.linear_dim}, 1);
        } else {
            linear(a + "q_a_proj", c.qrank, c.hidden);
            tensor(a + "q_a_layernorm.weight", {c.qrank}, 1);
            linear(a + "q_b_proj", c.heads * c.qdim, c.qrank);
            linear(a + "kv_a_proj_with_mqa", c.kvrank, c.hidden);
            tensor(a + "kv_a_layernorm.weight", {c.kvrank}, 1);
            linear(a + "kv_b_proj", c.heads * (c.qdim + c.vdim), c.kvrank);
            linear(a + "o_proj", c.hidden, c.heads * c.vdim);
            if (c.indexers[l] == "full") {
                a += "indexer.";
                linear(a + "wq_b", c.index_heads * c.index_dim, c.qrank);
                linear(a + "wk", c.index_dim, c.hidden);
                tensor(a + "k_norm.weight", {c.index_dim}, 1);
                tensor(a + "k_norm.bias", {c.index_dim}, 0);
                linear(a + "weights_proj", c.index_heads, c.hidden);
                tensor(a + "index_kpool_compress_gate", {c.index_dim, c.hidden});
                tensor(a + "index_kpool_compress_ape", {c.pool, c.index_dim});
            }
        }
        auto mlp = [&](const std::string &a, int width) {
            linear(a + "gate_proj", width, c.hidden);
            linear(a + "up_proj", width, c.hidden);
            linear(a + "down_proj", c.hidden, width);
        };
        if (c.mlp[l] == "dense")
            mlp(p + "mlp.", c.intermediate);
        else {
            linear(p + "mlp.gate", c.experts, c.hidden);
            tensor(p + "mlp.gate.e_score_correction_bias", {c.experts});
            mlp(p + "mlp.shared_experts.", c.expert_intermediate);
            for (int e = 0; e < c.experts; ++e)
                mlp(p + "mlp.experts." + std::to_string(e) + ".", c.expert_intermediate);
        }
    }
    w.write((path / "model.safetensors").string());
    {
        std::ofstream f(path / "model.safetensors.index.json");
        f << json({{"weight_map", map}});
    }
}
template <class F> void rejects(F f) {
    bool bad = false;
    try {
        f();
    } catch (const std::exception &) {
        bad = true;
    }
    check(bad, "expected rejection");
}
void tokenizer_fixture(const fs::path &p) {
    fs::create_directories(p);
    json vocab = json::object();
    std::vector<std::string> chars;
    int next = 256;
    auto u8 = [](int c) {
        return c < 128 ? std::string(1, char(c)) : std::string{char(0xc0 | (c >> 6)), char(0x80 | (c & 63))};
    };
    for (int b = 0; b < 256; ++b) {
        int code = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) ? b : next++;
        vocab[u8(code)] = b;
        chars.push_back(u8(code));
    }
    vocab["ab"] = 256;
    vocab["123"] = 257;
    vocab["12"] = 258;
    json added = json::array();
    int id = 259;
    for (auto token : {"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"})
        added.push_back({{"id", id++},
                         {"content", token},
                         {"special", true},
                         {"lstrip", false},
                         {"rstrip", false},
                         {"single_word", false},
                         {"normalized", false}});
    json j = {
        {"truncation", {{"max_length", 2}}},
        {"model",
         {{"type", "BPE"}, {"byte_fallback", false}, {"vocab", vocab}, {"merges", {"a b", "1 2", "12 3"}}}},
        {"normalizer", nullptr},
        {"added_tokens", added},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           json::array({{{"type", "Split"},
                         {"pattern",
                          {{"Regex", "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
                                     " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}},
                         {"behavior", "Isolated"},
                         {"invert", false}},
                        {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", false}}})}}},
        {"decoder", {{"type", "ByteLevel"}}}};
    {
        std::ofstream f(p / "tokenizer.json");
        f << j;
    }
    {
        std::ofstream f(p / "tokenizer_config.json");
        f << json(
            {{"chat_template",
              R"({{ bos_token }}{% set first = messages.0 %}{% set effort = 'max'|capitalize %}{% if effort != 'Max' %}{{ raise_exception('capitalize mismatch') }}{% endif %}{% for m in messages %}{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\n\n' + m.content|trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %})"},
             {"bos_token", "<|begin_of_text|>"},
             {"eos_token", "<|eot_id|>"}});
    }
}
void fp8_loader(const fs::path &path) {
    fs::create_directory(path);
    qtip::SafeTensorsWriter writer;
    json map = json::object();
    auto add = [&](const std::string &name, const std::string &dtype, std::vector<int64_t> shape,
                   const void *data, size_t bytes) {
        writer.add(name, dtype, shape, data, bytes);
        map[name] = "model.safetensors";
    };
    float embedding = 1;
    add("model.embed_tokens.weight", "F32", {1, 1}, &embedding, 4);
    // Partial blocks exercise both scale axes and the ceil-divided block stride.
    constexpr int rows = 130, cols = 144;
    std::vector<uint8_t> data(rows * cols);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = behemoth::encode_e4m3(float(int(i % 37) - 18) / 8);
    Vec channel(rows), block{.125f, .25f, .5f, 2.f};
    for (int r = 0; r < rows; ++r)
        channel[r] = float(1 + r % 7) / 3;
    add("channel.weight", "F8_E4M3", {rows, cols}, data.data(), data.size());
    add("channel.weight_scale", "F32", {rows, 1}, channel.data(), channel.size() * 4);
    add("block.weight", "F8_E4M3", {rows, cols}, data.data(), data.size());
    add("block.weight_scale_inv", "F32", {2, 2}, block.data(), block.size() * 4);
    add("bad.weight", "F8_E4M3", {rows, cols}, data.data(), data.size());
    add("bad.weight_scale_inv", "F32", {4}, block.data(), block.size() * 4);
    writer.write((path / "model.safetensors").string());
    {
        std::ofstream f(path / "model.safetensors.index.json");
        f << json({{"weight_map", map}});
    }
    Weights w(path);
    auto c = w.linear("channel"), b = w.linear("block");
    for (int r = 0; r < rows; ++r)
        for (int col = 0; col < cols; ++col) {
            size_t i = size_t(r) * cols + col;
            float raw = behemoth::decode_e4m3(data[i]);
            check(c.data[i] == float(behemoth::BFloat16(raw * channel[r])), "FP8 channel scaling");
            check(b.data[i] == float(behemoth::BFloat16(raw * block[(r / 128) * 2 + col / 128])),
                  "FP8 block scaling");
        }
    rejects([&] { w.linear("bad"); });
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--fixture") {
            fixture(argv[2], false);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--dense-fixture") {
            fixture(argv[2], true);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--server-fixture") {
            fixture(argv[2], true, true);
            tokenizer_fixture(argv[2]);
            return 0;
        }
        auto root = fs::temp_directory_path() / ("glm53-test-" + std::to_string(getpid()));
        fs::create_directory(root);
        Config original(tiny());
        Config published(read_json(fs::path(GLM53_FIXTURES) / "original_config.json"));
        Config supplied(read_json(fs::path(GLM53_FIXTURES) / "dense_config.json"));
        check(published.layers == 45 && published.hidden == 4096 && published.experts == 288 &&
                  published.summary()["kda_layers"] == 34 && published.summary()["moe_layers"] == 42,
              "published 320B configuration");
        check(supplied.layers == 80 && supplied.hidden == 8192 && supplied.qrank == 4096 &&
                  supplied.vocab == 128256 && supplied.summary()["moe_layers"] == 0 &&
                  supplied.summary()["kda_layers"] == 60,
              "supplied dense configuration");
        check(original.layers == 4 && original.mlp[1] == "sparse", "config parse");
        Config custom(tiny(true));
        check(custom.mlp[1] == "dense" && custom.experts == 4, "unused MoE fields must not force routing");
        auto bad = tiny();
        bad["text_config"]["layer_types"] = {"linear_attention"};
        rejects([&] { Config c(bad); });
        bad = tiny();
        bad["text_config"]["index_kpool"] = 0;
        rejects([&] { Config c(bad); });
        fp8_loader(root / "fp8");
        Vec state;
        auto y = kda_step({1, 0}, {1, 0}, {2, 3}, {0, 0}, {1}, state, 1, 2);
        check(std::abs(y[0] - std::sqrt(2.f)) < 1e-5 && std::abs(y[1] - 3 / std::sqrt(2.f)) < 1e-5,
              "KDA delta update");
        auto z = kda_step({0, 1}, {0, 1}, {4, 5}, {std::log(.5f), std::log(.25f)}, {1}, state, 1, 2);
        check(std::abs(state[0] - 1) < 1e-5 && std::abs(z[0] - 4 / std::sqrt(2.f)) < 1e-5,
              "KDA channel decay/state continuation");
        Vec values(32);
        for (int i = 0; i < 32; ++i)
            values[i] = float(i - 16) / 4;
        auto q = quantize_nvfp4(values, 2, 16);
        auto dq = dequantize_nvfp4(q);
        double error = 0, ss = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            error += std::pow(values[i] - dq[i], 2);
            ss += values[i] * values[i];
        }
        check(error / ss < .025, "NVFP4 reconstruction error");
        check(dequantize_nvfp4(quantize_nvfp4(Vec(16), 1, 16)) == Vec(16), "zero NVFP4 tensor");
        rejects([&] { quantize_nvfp4(Vec(15), 1, 15); });
        tokenizer_fixture(root / "tokenizer");
        Tokenizer tokenizer(root / "tokenizer");
        check(tokenizer.encode("ab") == std::vector<int>{256}, "BPE merge order");
        check(tokenizer.encode("123123") == std::vector<int>({257, 257}), "Llama-3 three-digit regex");
        for (const auto &s : {std::string("Café 中文 😀\n\n I’M not 1234567"), std::string(4096, 'a')})
            check(tokenizer.decode(tokenizer.encode(s)) == s, "Unicode byte BPE roundtrip/truncation");
        check(tokenizer.encode("<|begin_of_text|>")[0] == 259, "special token split");
        check(tokenizer.chat(json::array({{{"role", "user"}, {"content", " hello "}}})) ==
                  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_"
                  "header_id|>assistant<|end_header_id|>\n\n",
              "Llama-3 template");
        for (bool dense : {false, true}) {
            auto path = root / (dense ? "dense" : "moe");
            fixture(path, dense);
            Model a(path, false);
            std::vector<Vec> baseline;
            for (int t : {1, 2, 3, 4, 5, 6, 7})
                baseline.push_back(a.step(t));
            a.reset();
            size_t at = 0;
            for (int t : {1, 2, 3, 4, 5, 6, 7})
                check(a.step(t) == baseline[at++], "state reset/non-deterministic replay");
            convert_nvfp4(path, root / (dense ? "dense-nv" : "moe-nv"));
            Weights before(path), after(root / (dense ? "dense-nv" : "moe-nv"));
            auto n = before.prefix + "layers.0.self_attn.q_proj.weight";
            check(before.tensor(n).data == after.tensor(n).data, "non-MLP weights changed");
            Model quant(root / (dense ? "dense-nv" : "moe-nv"), false);
            for (int t : {1, 2, 3, 4}) {
                auto logits = quant.step(t);
                check(logits.size() == 32, "quantized output vocabulary");
                for (float v : logits)
                    check(std::isfinite(v), "nonfinite quantized logits");
            }
        }
        fs::remove_all(root);
        std::cout << "PASS GLM53 dense/MoE KDA/MLA state replay, NVFP4, tokenizer and config\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
