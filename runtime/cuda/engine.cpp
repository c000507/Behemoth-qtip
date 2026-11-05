#include "behemoth/cuda.hpp"
#include "behemoth/qtip.hpp"
#include "glm53.hpp"
#include "kernels.hpp"
#if defined(BEHEMOTH_CUDA_FAST)
#include "tensor_model.hpp"
#endif
#include "minja/minja.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
namespace behemoth::cuda_runtime {
namespace {
using namespace glm53::gpu;
struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {
        checked(cudaGetDevice(&previous), "read current CUDA device");
        checked(set_device(device), "select CUDA peer device");
    }
    ~DeviceGuard() { (void)set_device(previous); }
};
struct DeviceEvent {
    int device;
    cudaEvent_t event{};
    explicit DeviceEvent(int d) : device(d) {
        DeviceGuard guard(d);
        checked(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create CUDA peer event");
    }
    ~DeviceEvent() {
        DeviceGuard guard(device);
        (void)cudaEventDestroy(event);
    }
};
struct PeerContext {
    int primary, device;
    Stream stream{};
    DeviceEvent ready, done;
    Buffer<B16> input;
    Buffer<float> partial;
    PeerContext(int p, int d) : primary(p), device(d), ready(p), done(d) {
        for (auto [from, to] : {std::pair{p, d}, std::pair{d, p}}) {
            DeviceGuard guard(from);
            int supported = 0;
            checked(cudaDeviceCanAccessPeer(&supported, from, to), "query CUDA peer access");
            check(supported != 0, "TP=2 requires bidirectional CUDA peer access");
            auto error = cudaDeviceEnablePeerAccess(to, 0);
            if (error == cudaErrorPeerAccessAlreadyEnabled)
                (void)cudaGetLastError();
            else
                checked(error, "enable CUDA peer access");
        }
        DeviceGuard guard(d);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create CUDA peer stream");
        input.ensure(28672, stream);
        partial.ensure(size_t(28672) * 32 / 2, stream);
    }
    ~PeerContext() {
        DeviceGuard guard(device);
        if (stream) {
            (void)sync(stream);
            input.clear();
            partial.clear();
            (void)stream_destroy(stream);
        }
    }
};
struct PeerWeights {
    int device;
    Buffer<B16> trellis;
    Buffer<uint32_t> lut;
    explicit PeerWeights(int d) : device(d) {}
    ~PeerWeights() {
        DeviceGuard guard(device);
        trellis.clear();
        lut.clear();
    }
};
template <class T> void upload(Buffer<T> &d, const T *src, size_t n, Stream s) {
    d.ensure(n, s);
    checked(h2d(d, src, n * sizeof(T), s), "upload CUDA weights");
}
struct Linear {
    int in = 0, out = 0, splits = 1;
    Buffer<B16> trellis;
    Buffer<float> su, sv, scales;
    Buffer<uint32_t> lut;
    Buffer<uint8_t> fp8;
    Buffer<uint8_t> nv_packed, nv_scales;
    float nv_global = 0;
    PeerContext *parallel = nullptr;
    std::unique_ptr<PeerWeights> peer;
    void load(const Checkpoint &cp, int layer, const std::string &name, Stream s, int requested_splits,
              PeerContext *context) {
        parallel = context;
        auto h = behemoth::qtip::load_linear(cp, layer, name);
        in = h.in;
        out = h.out;
        splits = std::min(requested_splits, in / 32);
        if (h.is_fp8()) {
            upload(fp8, h.fp8.data(), h.fp8.size(), s);
            upload(scales, h.fp8_scale.data(), h.fp8_scale.size(), s);
        } else if (h.is_nvfp4()) {
            nv_global = h.nvfp4_global;
            upload(nv_packed, h.nvfp4_packed.data(), h.nvfp4_packed.size(), s);
            upload(nv_scales, h.nvfp4_scales.data(), h.nvfp4_scales.size(), s);
        } else {
            auto cb = behemoth::qtip::make_codebook(h.tlut);
            size_t local = h.trellis.size() / (parallel ? 2 : 1);
            upload(trellis, h.trellis.data(), local, s);
            upload(su, h.su.data(), h.su.size(), s);
            upload(sv, h.sv.data(), h.sv.size(), s);
            upload(lut, cb.bf16x2.data(), cb.bf16x2.size(), s);
            if (parallel) {
                check(out % 64 == 0, "TP=2 QTIP outputs must be divisible by 64");
                DeviceGuard guard(parallel->device);
                peer = std::make_unique<PeerWeights>(parallel->device);
                upload(peer->trellis, h.trellis.data() + local, local, parallel->stream);
                upload(peer->lut, cb.bf16x2.data(), cb.bf16x2.size(), parallel->stream);
                checked(sync(parallel->stream), "finish peer weight upload");
            }
        }
        checked(sync(s), "finish weight upload");
    }
    void forward(const Buffer<B16> &x, Buffer<B16> &y, Buffer<float> &scratch, Buffer<B16> &rotated,
                 Buffer<float> &z, bool rz, Stream s) const {
        y.ensure(out, s);
        if (fp8.p)
            fp8_forward(fp8, scales, x, y, in, out, s);
        else if (nv_packed.p)
            nvfp4_forward(nv_packed, nv_scales, nv_global, x, y, in, out, s);
        else {
            scratch.ensure(std::max(in, out), s);
            rotated.ensure(in, s);
            z.ensure(size_t(out) * splits, s);
            QtipMatrix full{trellis, su, sv, lut, in, out};
            if (!parallel)
                qtip_forward(full, x, y, scratch, rotated, z, rz, s, splits);
            else {
                qtip_prepare(full, x, scratch, rotated, s);
                checked(cudaEventRecord(parallel->ready.event, s), "record TP input ready");
                {
                    DeviceGuard guard(parallel->device);
                    checked(cudaStreamWaitEvent(parallel->stream, parallel->ready.event, 0),
                            "wait for TP input");
                    checked(cudaMemcpyPeerAsync(parallel->input, parallel->device, rotated, parallel->primary,
                                                size_t(in) * 2, parallel->stream),
                            "broadcast TP input");
                    qtip_multiply({peer->trellis, nullptr, nullptr, peer->lut, in, out / 2}, parallel->input,
                                  parallel->partial, out / 2, splits, parallel->stream);
                    cudaMemcpy3DPeerParms copy{};
                    size_t width = size_t(out / 2) * sizeof(float);
                    copy.srcPtr = make_cudaPitchedPtr(parallel->partial, width, width, splits);
                    copy.dstPtr = make_cudaPitchedPtr(z.p + out / 2, width * 2, width * 2, splits);
                    copy.srcDevice = parallel->device;
                    copy.dstDevice = parallel->primary;
                    copy.extent = make_cudaExtent(width, splits, 1);
                    checked(cudaMemcpy3DPeerAsync(&copy, parallel->stream), "gather TP partials");
                    checked(cudaEventRecord(parallel->done.event, parallel->stream), "record TP completion");
                }
                auto local = full;
                local.out /= 2;
                qtip_multiply(local, rotated, z, out, splits, s);
                checked(cudaStreamWaitEvent(s, parallel->done.event, 0), "wait for TP projection");
                qtip_finish(full, z, y, scratch, rz, splits, s);
            }
        }
    }
};
json read(const fs::path &p) {
    std::ifstream f(p);
    check(bool(f), "Cannot read " + p.string());
    return json::parse(f);
}
int qtip_splits(int tp) {
    const char *value = std::getenv("BEHEMOTH_CUDA_QTIP_SPLITS");
    if (!value)
        return tp == 2 ? 16 : 1;
    int n = 0;
    const char *end = value + std::char_traits<char>::length(value);
    auto result = std::from_chars(value, end, n);
    check(result.ec == std::errc{} && result.ptr == end && n >= 1 && n <= 32,
          "BEHEMOTH_CUDA_QTIP_SPLITS must be an integer in 1..32");
    return n;
}
} // namespace
struct Model::Impl {
    Checkpoint checkpoint;
    int device, context, position = 0;
    Stream stream{};
    std::unique_ptr<PeerContext> peer;
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
    Impl(const fs::path &path, int d, int ctx, int tp)
        : checkpoint(path), device(d), context(ctx), layers(checkpoint.config.layers) {
        int splits = qtip_splits(tp);
        check(tp == 1 || tp == 2, "CUDA tensor parallel degree must be 1 or 2");
        check(checkpoint.config.quantization == Quantization::Qtip, "CUDA runner requires a QTIP checkpoint");
        check(ctx > 0 && ctx <= checkpoint.config.max_context, "Invalid CUDA context limit");
        checked(set_device(d), "select CUDA device");
        checked(stream_create(&stream), "create CUDA stream");
        try {
            if (tp == 2) {
                int count = 0;
                checked(device_count(&count), "count CUDA devices");
                check(d >= 0 && d + 1 < count, "TP=2 requires two CUDA devices starting at --device");
                peer = std::make_unique<PeerContext>(d, d + 1);
            }
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
                    p.linear[i].load(checkpoint, l, names[i], stream, splits, peer.get());
                p.keys.ensure(size_t(ctx) * 1024, stream);
                p.values.ensure(size_t(ctx) * 1024, stream);
                std::cerr << "Loaded CUDA QTIP layer " << l + 1 << '/' << layers.size() << '\n';
            }
            x.ensure(12288, stream);
            normed.ensure(12288, stream);
            attended.ensure(12288, stream);
            logits.ensure(32768, stream);
            // Scores are scratch, so reserve them once instead of cudaMalloc/free
            // at every new sequence length (cudaFree synchronizes the device).
            scores.ensure(size_t(96) * ctx, stream);
            readback.ensure(32768);
            size_t free = 0, total = 0;
            checked(cudaMemGetInfo(&free, &total), "CUDA memory info");
            std::cerr << "CUDA QTIP ready: " << double(total - free) / (1ull << 30) << " GiB used, context "
                      << ctx << '\n';
        } catch (...) {
            if (peer) {
                DeviceGuard guard(peer->device);
                (void)sync(peer->stream);
            }
            (void)sync(stream);
            (void)stream_destroy(stream);
            stream = nullptr;
            throw;
        }
    }
    ~Impl() {
        (void)set_device(device);
        if (peer) {
            DeviceGuard guard(peer->device);
            (void)sync(peer->stream);
        }
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
            if (peer) {
                DeviceGuard guard(peer->device);
                (void)sync(peer->stream);
            }
            failed = true;
            throw;
        }
    }
};
Model::Model(const fs::path &p, int device, int context, int tp, int max_batch)
    : config{32768, context, {2}} {
    auto raw = read(p / "config.json");
    auto kind = raw.value("model_type", "");
    if (kind == "glm5_next" || kind == "glm5_next_text") {
        check(tp == 1, "GLM tensor parallel inference is not implemented yet");
        glm53::Config c(raw);
        check(context > 0 && context <= c.max_context, "Invalid GLM context limit");
        config = {c.vocab, context, c.eos};
        backend_ = glm53::gpu_backend(p, device);
    } else {
#if defined(BEHEMOTH_CUDA_FAST)
        if (tp == 2 || max_batch > 1 || std::getenv("BEHEMOTH_CUDA_FAST_PATH")) {
            tensor_ = std::make_unique<TensorModel>(p, device, context, tp, max_batch);
            return;
        }
#else
        check(max_batch == 1, "Rebuild with BEHEMOTH_CUDA_FAST=ON for batched inference");
#endif
        impl_ = std::make_unique<Impl>(p, device, context, tp);
    }
}
Model::~Model() = default;
Vec Model::step(int token) {
#if defined(BEHEMOTH_CUDA_FAST)
    if (tensor_) {
        auto out = tensor_->step({token});
        ++position_;
        return std::move(out[0]);
    }
#endif
    check(token >= 0 && token < config.vocab && position_ < config.max_context,
          "Token/context outside configured limits");
    auto out = backend_ ? backend_->step(token) : impl_->step(token);
    ++position_;
    return out;
}
Vec Model::prefill(const std::vector<int> &tokens) {
    auto result = prefill_batch({tokens});
    return std::move(result[0]);
}
std::vector<Vec> Model::prefill_batch(const std::vector<std::vector<int>> &tokens) {
#if defined(BEHEMOTH_CUDA_FAST)
    if (tensor_) {
        auto out = tensor_->prefill(tokens);
        position_ = 0;
        for (const auto &row : tokens)
            position_ = std::max(position_, int(row.size()));
        return out;
    }
#endif
    check(tokens.size() == 1 && !tokens[0].empty(), "This backend requires a single prefill sequence");
    reset();
    Vec out;
    for (int id : tokens[0])
        out = step(id);
    return {std::move(out)};
}
std::vector<Vec> Model::step_batch(const std::vector<int> &tokens) {
#if defined(BEHEMOTH_CUDA_FAST)
    if (tensor_) {
        auto out = tensor_->step(tokens);
        ++position_;
        return out;
    }
#endif
    check(tokens.size() == 1, "This backend requires a single decode sequence");
    return {step(tokens[0])};
}
void Model::reset() {
    position_ = 0;
#if defined(BEHEMOTH_CUDA_FAST)
    if (tensor_) {
        tensor_->reset();
        return;
    }
#endif
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
