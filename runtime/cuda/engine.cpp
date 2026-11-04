#include "behemoth/cuda.hpp"
#include "behemoth/qtip.hpp"
#include "glm53.hpp"
#include "kernels.hpp"
#include "minja/minja.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
namespace behemoth::cuda_runtime {
namespace {
using namespace glm53::gpu;
template <class T> void upload(Buffer<T> &d, const T *src, size_t n, Stream s) {
    d.ensure(n, s);
    checked(h2d(d, src, n * sizeof(T), s), "upload CUDA weights");
}
struct Linear {
    int in = 0, out = 0;
    Buffer<B16> trellis;
    Buffer<float> su, sv, scales;
    Buffer<uint32_t> lut;
    Buffer<uint8_t> fp8;
    void load(const Checkpoint &cp, int layer, const std::string &name, Stream s) {
        auto h = behemoth::qtip::load_linear(cp, layer, name);
        in = h.in;
        out = h.out;
        if (h.is_fp8()) {
            upload(fp8, h.fp8.data(), h.fp8.size(), s);
            upload(scales, h.fp8_scale.data(), h.fp8_scale.size(), s);
        } else {
            auto cb = behemoth::qtip::make_codebook(h.tlut);
            upload(trellis, h.trellis.data(), h.trellis.size(), s);
            upload(su, h.su.data(), h.su.size(), s);
            upload(sv, h.sv.data(), h.sv.size(), s);
            upload(lut, cb.bf16x2.data(), cb.bf16x2.size(), s);
        }
        checked(sync(s), "finish weight upload");
    }
    void forward(const Buffer<B16> &x, Buffer<B16> &y, Buffer<float> &scratch, Buffer<B16> &rotated,
                 Buffer<float> &z, bool rz, Stream s) const {
        y.ensure(out, s);
        if (fp8.p)
            fp8_forward(fp8, scales, x, y, in, out, s);
        else {
            scratch.ensure(std::max(in, out), s);
            rotated.ensure(in, s);
            z.ensure(out, s);
            qtip_forward({trellis, su, sv, lut, in, out}, x, y, scratch, rotated, z, rz, s);
        }
    }
};
json read(const fs::path &p) {
    std::ifstream f(p);
    check(bool(f), "Cannot read " + p.string());
    return json::parse(f);
}
} // namespace
struct Model::Impl {
    Checkpoint checkpoint;
    int device, context, position = 0;
    Stream stream{};
    bool failed = false;
    struct Layer {
        std::array<Linear, 7> linear;
        Buffer<float> norm1, norm2;
        Buffer<B16> keys, values;
    };
    std::vector<Layer> layers;
    Buffer<B16> embeddings, head, x, normed, q, k, v, attended, projected, gate, up, rotated;
    Buffer<float> final_norm, scratch, z, scores, logits;
    Pinned<float> readback;
    void vector(const std::string &name, Buffer<float> &out) {
        auto view = checkpoint.tensor(name).floats(name);
        check(view.size() == 12288, "Invalid normalization shape: " + name);
        Vec data(view.size());
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = view[i];
            check(std::isfinite(data[i]), "Nonfinite normalization weight");
        }
        upload(out, data.data(), data.size(), stream);
        checked(sync(stream), "upload normalization");
    }
    void dense(const std::string &name, Buffer<B16> &out) {
        const auto &f = checkpoint.tensor(name);
        auto shape = f.shape(name);
        check(shape == std::vector<size_t>{32768, 12288}, "Invalid embedding/head shape");
        size_t n = size_t(32768) * 12288;
        out.ensure(n, stream);
        if (f.dtype(name) == "BF16") {
            auto bytes = f.bytes(name);
            checked(h2d(out, bytes.data(), bytes.size(), stream), "upload dense BF16");
        } else {
            auto data = checkpoint.bf16(name);
            checked(h2d(out, data.data(), data.size() * 2, stream), "upload dense converted weights");
            checked(sync(stream), "finish converted weight upload");
        }
        checked(sync(stream), "finish dense weight upload");
    }
    Impl(const fs::path &path, int d, int ctx)
        : checkpoint(path), device(d), context(ctx), layers(checkpoint.config.layers) {
        check(checkpoint.config.quantization == Quantization::Qtip, "CUDA runner requires a QTIP checkpoint");
        check(ctx > 0 && ctx <= checkpoint.config.max_context, "Invalid CUDA context limit");
        checked(set_device(d), "select CUDA device");
        checked(stream_create(&stream), "create CUDA stream");
        try {
            initialize_tables();
            dense("model.embed_tokens.weight", embeddings);
            dense("lm_head.weight", head);
            vector("model.norm.weight", final_norm);
            const std::array<const char *, 7> names = {
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
            for (int l = 0; l < int(layers.size()); ++l) {
                auto &p = layers[l];
                auto prefix = "model.layers." + std::to_string(l) + ".";
                vector(prefix + "input_layernorm.weight", p.norm1);
                vector(prefix + "post_attention_layernorm.weight", p.norm2);
                for (int i = 0; i < 7; ++i)
                    p.linear[i].load(checkpoint, l, names[i], stream);
                p.keys.ensure(size_t(ctx) * 1024, stream);
                p.values.ensure(size_t(ctx) * 1024, stream);
                std::cerr << "Loaded CUDA QTIP layer " << l + 1 << '/' << layers.size() << '\n';
            }
            x.ensure(12288, stream);
            normed.ensure(12288, stream);
            attended.ensure(12288, stream);
            logits.ensure(32768, stream);
            readback.ensure(32768);
            size_t free = 0, total = 0;
            checked(cudaMemGetInfo(&free, &total), "CUDA memory info");
            std::cerr << "CUDA QTIP ready: " << double(total - free) / (1ull << 30) << " GiB used, context "
                      << ctx << '\n';
        } catch (...) {
            (void)sync(stream);
            (void)stream_destroy(stream);
            stream = nullptr;
            throw;
        }
    }
    ~Impl() {
        (void)set_device(device);
        if (stream) {
            (void)sync(stream);
            (void)stream_destroy(stream);
        }
    }
    Vec step(int token) {
        check(!failed, "Reset the CUDA model after an inference failure");
        check(token >= 0 && token < 32768 && position < context,
              "Token or context exceeds CUDA model limits");
        checked(set_device(device), "select CUDA device");
        try {
            embedding(embeddings, x, token, 12288, stream);
            for (auto &layer : layers) {
                norm(x, layer.norm1, nullptr, normed, 12288, 1e-5f, false, stream);
                layer.linear[0].forward(normed, q, scratch, rotated, z, false, stream);
                layer.linear[1].forward(normed, k, scratch, rotated, z, false, stream);
                layer.linear[2].forward(normed, v, scratch, rotated, z, false, stream);
                rope(q, k, position, 96, 8, 128, 1000000.f, stream);
                checked(d2d(layer.keys.p + size_t(position) * 1024, k, 2048, stream),
                        "append CUDA key cache");
                checked(d2d(layer.values.p + size_t(position) * 1024, v, 2048, stream),
                        "append CUDA value cache");
                scores.ensure(size_t(96) * (position + 1), stream);
                gqa(q, layer.keys, layer.values, attended, scores, position + 1, 96, 8, 128, stream);
                layer.linear[3].forward(attended, projected, scratch, rotated, z, false, stream);
                residual(x, projected, 12288, stream);
                norm(x, layer.norm2, nullptr, normed, 12288, 1e-5f, false, stream);
                layer.linear[4].forward(normed, gate, scratch, rotated, z, true, stream);
                layer.linear[5].forward(normed, up, scratch, rotated, z, true, stream);
                swiglu(gate, up, 28672, INFINITY, stream);
                layer.linear[6].forward(gate, projected, scratch, rotated, z, false, stream);
                residual(x, projected, 12288, stream);
            }
            norm(x, final_norm, nullptr, normed, 12288, 1e-5f, false, stream);
            gemv({head, nullptr, nullptr, 32768, 12288, 4, false, 1}, normed, logits, true, stream);
            checked(d2h(readback.p, logits, 32768 * 4, stream), "read CUDA logits");
            checked(sync(stream), "finish CUDA token");
            Vec out(readback.p, readback.p + 32768);
            for (float a : out)
                check(std::isfinite(a), "Nonfinite CUDA logits");
            ++position;
            return out;
        } catch (...) {
            failed = true;
            throw;
        }
    }
};
Model::Model(const fs::path &p, int device, int context) : config{32768, context, {2}} {
    auto raw = read(p / "config.json");
    auto kind = raw.value("model_type", "");
    if (kind == "glm5_next" || kind == "glm5_next_text") {
        glm53::Config c(raw);
        check(context > 0 && context <= c.max_context, "Invalid GLM context limit");
        config = {c.vocab, context, c.eos};
        backend_ = glm53::gpu_backend(p, device);
    } else
        impl_ = std::make_unique<Impl>(p, device, context);
}
Model::~Model() = default;
Vec Model::step(int token) {
    check(token >= 0 && token < config.vocab && position_ < config.max_context,
          "Token/context outside configured limits");
    auto out = backend_ ? backend_->step(token) : impl_->step(token);
    ++position_;
    return out;
}
void Model::reset() {
    position_ = 0;
    if (backend_) {
        backend_->reset();
        return;
    }
    checked(set_device(impl_->device), "select CUDA device");
    checked(sync(impl_->stream), "reset CUDA sequence");
    impl_->position = 0;
    impl_->failed = false;
}
Tokenizer::Tokenizer(const fs::path &p) : settings_(read(p / "tokenizer_config.json")) {
    auto tokenizer = read(p / "tokenizer.json");
    if (tokenizer.at("model").at("type") == "BPE" && tokenizer.at("decoder").at("type") == "ByteLevel") {
        bpe_ = std::make_unique<glm53::Tokenizer>(p);
        return;
    }
    tokenizer_ = std::make_unique<behemoth::Tokenizer>(p);
    if (!settings_.contains("chat_template")) {
        std::ifstream f(p / "chat_template.jinja");
        check(bool(f), "Missing chat template");
        settings_["chat_template"] = std::string(std::istreambuf_iterator<char>(f), {});
    }
    for (const auto &i : settings_.at("added_tokens_decoder").items())
        added_[std::stoi(i.key())] = i.value().at("content");
}
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string &text) const {
    return bpe_ ? bpe_->encode(text) : tokenizer_->encode(text);
}
std::string Tokenizer::decode(const std::vector<int> &ids, bool special) const {
    if (bpe_)
        return bpe_->decode(ids, special);
    if (!special)
        return tokenizer_->decode(ids);
    std::string out;
    bool first = true;
    for (int id : ids)
        out += piece(id, first, true);
    return out;
}
std::string Tokenizer::piece(int id, bool &first, bool special) const {
    if (bpe_)
        return bpe_->decode({id}, special);
    if (special && tokenizer_->is_special(id)) {
        auto it = added_.find(id);
        if (it != added_.end()) {
            first = false;
            return it->second;
        }
        return "";
    }
    return tokenizer_->piece(id, first);
}
std::string Tokenizer::chat(const json &messages) const {
    if (bpe_)
        return bpe_->chat(messages);
    check(messages.is_array() && !messages.empty(), "Messages must be a nonempty array");
    for (const auto &m : messages) {
        auto r = m.at("role").get<std::string>();
        check((r == "system" || r == "user" || r == "assistant") && m.at("content").is_string() &&
                  !m.contains("tool_calls"),
              "Only text chat is supported");
    }
    auto source = settings_.at("chat_template");
    check(source.is_string(), "Expected string chat template");
    auto token = [&](const char *name) {
        auto x = settings_.at(name);
        return x.is_string() ? x.get<std::string>() : x.at("content").get<std::string>();
    };
    nlohmann::ordered_json inputs = {{"messages", nlohmann::ordered_json::parse(messages.dump())},
                                     {"bos_token", token("bos_token")},
                                     {"eos_token", token("eos_token")},
                                     {"add_generation_prompt", true}};
    return minja::Parser::parse(source.get<std::string>(), {true, true, false})
        ->render(minja::Context::make(minja::Value(inputs)));
}
} // namespace behemoth::cuda_runtime
