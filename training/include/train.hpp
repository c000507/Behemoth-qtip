#pragma once
#include "behemoth/qtip.hpp"
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace qat {
namespace fs = std::filesystem;
using json = nlohmann::json;
void require(bool condition, const std::string &message);
json read_json(const fs::path &);
std::string sha256(const fs::path &);
std::string model_identity(const fs::path &);
void durable_json(const fs::path &, const json &);
void sync_file(const fs::path &);
void sync_directory(const fs::path &);
float bf(float);
struct Config {
    int n, inter, heads, kv, dim, vocab, layers;
    float eps, theta;
    explicit Config(const json &);
};
inline const std::vector<std::string> linears = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                                 "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
                                                 "mlp.down_proj"};
inline const std::string norm1 = "input_layernorm.weight", norm2 = "post_attention_layernorm.weight";
std::string su_name(int linear);
struct Param {
    std::string name;
    size_t offset, size;
    bool norm;
};
struct LayerData {
    std::vector<behemoth::qtip::HostLinear> linear;
    std::map<std::string, size_t> params;
};
class Model {
    std::map<std::string, std::shared_ptr<behemoth::SafeTensorFile>> files_;

  public:
    fs::path path;
    json config_json, index;
    Config config;
    std::string identity;
    std::vector<LayerData> layers;
    std::vector<Param> params;
    std::vector<float> initial;
    size_t final_norm = 0;
    std::vector<uint16_t> embedding, head;
    explicit Model(const fs::path &);
    const behemoth::SafeTensorFile &file(const std::string &) const;
    size_t add_param(const std::string &, std::span<const float>, bool norm = false);
    void export_model(const fs::path &, std::span<const float>) const;
};
struct Batch {
    std::vector<int32_t> tokens, labels, segments, positions, starts, ends;
    size_t targets() const;
    void boundaries();
};
class Dataset {
    fs::path root_;

  public:
    json metadata;
    std::string identity;
    size_t count, length;
    explicit Dataset(const fs::path &, bool train = true);
    Batch get(size_t) const;
};
// Float tensors are views into a shared allocation. GPU buffers are allocated
// from a fixed arena; all temporary reuse stays ordered on one compute stream.
struct Mat {
    std::shared_ptr<void> owner;
    float *p = nullptr;
    int rows = 0, cols = 0, ld = 0;
    size_t size() const { return size_t(rows) * cols; }
    Mat row(int a, int b) const;
    Mat col(int a, int b) const;
};
struct Weight {
    std::shared_ptr<void> owner;
    const void *p = nullptr;
    int out = 0, in = 0;
    bool bf16 = false, transposed = false;
};
struct Norm {
    Mat y, r;
};
struct Attention {
    Mat q, k, v, out, lse;
};
struct Ops {
    bool rounding = false;
    virtual ~Ops() = default;
    virtual Mat alloc(int, int, bool zero = false) = 0;
    virtual void copy(Mat, Mat) = 0;
    virtual void upload(Mat, std::span<const float>) = 0;
    virtual void download(Mat, std::span<float>) = 0;
    virtual void zero(Mat) = 0;
    virtual void add(Mat dst, Mat a, Mat b, bool round = false) = 0;
    virtual void mul(Mat dst, Mat a, Mat b) = 0;
    virtual void round(Mat) = 0;
    virtual void colsum(Mat a, Mat b, Mat accumulator) = 0;
    virtual void scale_sum(Mat a, Mat b, Mat accumulator) = 0;
    virtual Mat had(Mat, bool transpose = false, Mat pre = {}) = 0;
    virtual Norm rms(Mat, Mat weight, float eps) = 0;
    virtual Mat rms_backward(Mat x, Norm, Mat weight, Mat dy, Mat dw) = 0;
    virtual Weight frozen(std::span<const uint16_t>, int out, int in) = 0;
    virtual std::vector<Weight> decode(const LayerData &) = 0;
    virtual Mat mm(Mat x, const Weight &, bool transpose = false) = 0;
    // Shared input transform; HIP writes BF16 GEMM operands directly. CPU keeps
    // the reference operation order (and unrounded finite-difference mode).
    virtual std::vector<Mat> had_mm(Mat x, std::span<const Weight> weights, Mat pre = {},
                                    bool transpose = false) {
        auto u = had(x, true, pre);
        std::vector<Mat> result;
        for (const auto &w : weights)
            result.push_back(mm(u, w, transpose));
        return result;
    }
    virtual Mat embedding(const Weight &, const Batch &) = 0;
    virtual Mat rope(Mat, int heads, const Config &, const Batch &, bool backward = false) = 0;
    virtual Mat swiglu(Mat gate, Mat up) = 0;
    virtual std::pair<Mat, Mat> swiglu_backward(Mat gate, Mat up, Mat dy) = 0;
    virtual Attention attention(Mat q, Mat k, Mat v, const Config &, const Batch &) = 0;
    virtual std::vector<Mat> attention_backward(const Attention &, Mat dy, const Config &, const Batch &) = 0;
    virtual double cross_entropy(Mat logits, std::span<const int32_t> labels, float normalizer,
                                 bool backward) = 0;
    virtual void adam(Mat p, Mat g, Mat m, Mat v, const std::vector<Param> &, uint64_t step, float lr,
                      float norm_lr, float decay, float clip) = 0;
    virtual void synchronize() {}
    // Prefetch callbacks borrow immutable, pinned model storage. CPU uses no-ops.
    virtual void configure_streaming(const Model &, int) {}
    virtual void prefetch(int, int, bool) {}
    virtual Mat checkpoint(int, Mat, bool save) = 0;
    virtual void acquire(int) {}
    virtual void release(int) {}
    virtual size_t peak_bytes() const { return 0; }
};
std::unique_ptr<Ops> cpu_ops(bool rounding = false);
std::unique_ptr<Ops> hip_ops(int device, size_t arena_bytes);
struct LinearSave {
    Mat x, c;
    int index;
};
class Layer {
    Ops &o_;
    const Model &m_;
    const LayerData &data_;
    Mat parameters_, gradients_;
    std::vector<Weight> weights_;
    Mat parameter(const std::string &) const;
    Mat gradient(const std::string &) const;
    Mat effective(const std::string &) const;
    Mat projection(Mat, int, LinearSave *save = nullptr);
    std::vector<Mat> projections(Mat, int first, int count, LinearSave *saves = nullptr);
    Mat projection_backward(const LinearSave &, Mat);
    struct AttnSave {
        Norm norm;
        Mat xn, h;
        Attention attention;
        std::vector<LinearSave> qkv;
        LinearSave out;
    };
    Mat attention_half(Mat, const Batch &, AttnSave *);
    struct MlpSave {
        Norm norm;
        LinearSave gate, up, down;
        Mat g, u;
    };
    Mat mlp(Mat, MlpSave *);

  public:
    Layer(Ops &, const Model &, int layer, Mat parameters, Mat gradients);
    Mat forward(Mat, const Batch &, int chunk);
    Mat backward(Mat x, Mat dy, const Batch &, int chunk);
};
struct Options {
    fs::path model, data, work, output, resume;
    std::vector<int> devices{-1}; // -1 CPU verification only
    uint64_t max_tokens = 500000, warmup = 10, checkpoint_every = 10, stop_after = 0;
    int rows = 1024, loss_rows = 128;
    float lr = 1e-4f, norm_lr = 1e-5f, decay = .01f, clip = 1;
    size_t arena_bytes = size_t(24) * 1000 * 1000 * 1000;
    bool eval = false, rounding = true;
    json signature(const Model &, const Dataset &) const;
};
struct Progress {
    uint64_t step = 0, position = 0, tokens = 0, supervised = 0;
};
class Replica {
  public:
    const Model &model;
    std::unique_ptr<Ops> ops;
    Mat parameters, gradients, first, second;
    Weight head, embedding;
    Replica(const Model &, int device, int seqlen, size_t arena, bool rounding);
    double run(const Batch &, const Options &, size_t denominator, bool backward);
    std::vector<float> values();
    std::vector<float> grad_values();
};
void save_checkpoint(const fs::path &, Replica &, const Progress &, const json &signature);
Progress load_checkpoint(const fs::path &, const std::vector<std::unique_ptr<Replica>> &, const json &);
int train(const Options &);
void write_tiny_model(const fs::path &, int layers = 2);
void write_tiny_data(const fs::path &, int sequences = 3);
} // namespace qat
