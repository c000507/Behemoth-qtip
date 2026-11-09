#pragma once
#include "behemoth/checkpoint.hpp"
#include <functional>
#include <set>
namespace glm53 {
namespace fs = std::filesystem;
using json = nlohmann::json;
using Vec = std::vector<float>;
void check(bool, const std::string &);
json read_json(const fs::path &);
struct Config {
    int hidden, intermediate, layers, vocab, heads, qrank, kvrank, qdim, vdim;
    int linear_heads, linear_dim, conv, streams, sinkhorn, index_heads, index_dim, pool, topk;
    int experts, expert_topk, expert_intermediate, shared, groups, top_groups;
    int max_context, bos = -1;
    float eps, hc_eps, gate_lower, swiglu_limit, routed_scale;
    bool norm_topk, tail;
    std::vector<std::string> attention, mlp, indexers;
    std::vector<int> eos;
    explicit Config(const json &);
    json summary() const;
};
class Tokenizer {
    json config_;
    std::map<std::string, int> vocab_, merges_, added_;
    std::vector<std::string> decode_;
    std::vector<std::string> patterns_;
    std::string byte_map_[256];
    std::map<std::string, unsigned char> inverse_;
    bool prefix_space_ = false, ignore_merges_ = false;
    void bpe(std::string_view, std::vector<int> &) const;
    std::string render(const nlohmann::ordered_json &messages, bool generation, const nlohmann::ordered_json &tools) const;

  public:
    explicit Tokenizer(const fs::path &);
    std::vector<int> encode(const std::string &) const;
    std::string decode(const std::vector<int> &, bool special = false) const;
    std::string chat(const json &, bool generation = true) const;
    // OpenAI messages with tool calls / tool results and the offered tools (array or null), rendered by a chat
    // template that takes them (GLM's): assistant tool_calls arguments given as JSON text are passed as objects.
    std::string chat_tools(const nlohmann::ordered_json &messages, const nlohmann::ordered_json &tools) const;
    json summary() const;
    size_t size() const { // token ids (highest id + 1)
        return decode_.size();
    }
};
struct Tensor {
    std::vector<size_t> shape;
    Vec data;
};
struct Nvfp4 {
    std::vector<uint8_t> packed, scales;
    float global;
    int rows, cols;
};
Nvfp4 quantize_nvfp4(const Vec &, int rows, int cols);
Vec dequantize_nvfp4(const Nvfp4 &);
class Weights {
    fs::path path_;
    json index_;
    mutable std::map<std::string, std::shared_ptr<behemoth::SafeTensorFile>> files_;
    mutable std::map<std::string, Tensor> cached_;

  public:
    std::string prefix;
    explicit Weights(const fs::path &);
    bool has(const std::string &) const;
    const behemoth::SafeTensorFile &file(const std::string &) const;
    Tensor tensor(const std::string &) const;
    Vec row(const std::string &, int index, int width) const;
    Tensor linear(const std::string &) const;
    const Tensor &cached_linear(const std::string &name, bool bf16 = false) const {
        auto key = name + (bf16 ? "#bf16" : "#fp32");
        if (!cached_.contains(key)) {
            auto t = linear(name);
            if (bf16)
                for (float &v : t.data)
                    v = float(behemoth::BFloat16(v));
            cached_[key] = std::move(t);
        }
        return cached_.at(key);
    }
    const json &index() const { return index_; }
};
void convert_nvfp4(const fs::path &, const fs::path &);
struct KdaState {
    Vec recurrent, convolution;
};
Vec kda_step(const Vec &, const Vec &, const Vec &, const Vec &, const Vec &, Vec &, int heads, int dim);
struct Hyper {
    Vec collapsed, post, mixing;
};
Hyper hyper_pre(const Vec &, const Tensor &, const Vec &, const Vec &, const Config &);
Vec hyper_post(const Vec &, const Vec &, const Hyper &, const Config &);
class InferenceBackend {
  public:
    virtual ~InferenceBackend() = default;
    virtual Vec step(int token) = 0;
    virtual void reset() = 0;
};
class Model {
    Weights weights_;
    struct State {
        KdaState kda;
        Vec latent, index_keys, index_gates;
        std::vector<int> last_topk;
    };
    std::vector<State> states_;
    size_t position_ = 0;
    bool bf16_;
    std::function<Vec(const Weights &, const std::string &, const Vec &)> projector_;
    std::shared_ptr<InferenceBackend> backend_;
    Vec matvec(const std::string &, const Vec &);
    Vec norm(const Vec &, const std::string &, float);
    Vec mlp(int, const Vec &);
    Vec attention(int, const Vec &);
    Vec round(Vec) const;

  public:
    Config config;
    explicit Model(const fs::path &, bool bf16 = true);
    void set_projector(std::function<Vec(const Weights &, const std::string &, const Vec &)> p) {
        check(!backend_, "cannot combine a full backend with a projection callback");
        projector_ = std::move(p);
    }
    void set_backend(std::shared_ptr<InferenceBackend> b) {
        check(b && !projector_ && position_ == 0, "set a backend before processing tokens");
        backend_ = std::move(b);
    }
    Vec step(int token);
    void reset();
};
std::function<Vec(const Weights &, const std::string &, const Vec &)> gpu_projector(int device);
std::shared_ptr<InferenceBackend> gpu_backend(const fs::path &, int device);
// GPU execution options. Layers are split into contiguous ranges over `devices` consecutive GPUs
// starting at `device` (activations cross between GPUs, weights do not). Linear weights stay resident
// on their GPU up to `resident_bytes` per device (<0: automatic from free memory at load, leaving room
// for the attention caches at `context` tokens; 0: stream every projection as before). The
// GLM53_RESIDENT_GIB environment variable overrides the automatic budget.
struct GpuOptions {
    int device = 0, devices = 1, context = 0;
    long long resident_bytes = -1;
};
std::shared_ptr<InferenceBackend> gpu_backend(const fs::path &, const GpuOptions &);
} // namespace glm53
