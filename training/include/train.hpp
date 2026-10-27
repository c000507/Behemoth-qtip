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
// `half` marks BF16 element storage (HIP activation storage only): p then
// addresses uint16_t elements and ld/rows/cols still count elements.
struct Mat {
    std::shared_ptr<void> owner;
    float *p = nullptr;
    int rows = 0, cols = 0, ld = 0;
    bool half = false;
    size_t size() const { return size_t(rows) * cols; }
    size_t element() const { return half ? 2 : 4; }
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
// Output transform of a QTIP linear, c = had(round_in ? bf16(z) : z) and
// y = bf16(c * post * post_scale[0]), optionally y = bf16(residual + y).
// The default implementation may round z in place.
struct Transform {
    bool round_in = false, keep_c = false, want_y = true;
    Mat post, post_scale, residual, y;
};
struct Transformed {
    Mat y, c;
};
struct Ops {
    bool rounding = false;
    virtual ~Ops() = default;
    virtual Mat alloc(int, int, bool zero = false) = 0;
    // Activation and activation-gradient storage (HIP may use BF16).
    virtual Mat act(int r, int c) { return alloc(r, c); }
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
    // `rounded`: the caller rounds every output to BF16 before use, so a backend
    // may return BF16 GEMM outputs.
    virtual std::vector<Mat> had_mm(Mat x, std::span<const Weight> weights, Mat pre = {},
                                    bool transpose = false, bool rounded = false) {
        (void)rounded;
        auto u = had(x, true, pre);
        std::vector<Mat> result;
        for (const auto &w : weights)
            result.push_back(mm(u, w, transpose));
        return result;
    }
    // Fusable compositions. Defaults define the reference operation order;
    // HIP replaces them with fused kernels.
    virtual std::vector<Mat> rms_had_mm(Mat x, Mat w, float eps, Mat su, std::span<const Weight> weights,
                                        Norm &norm, bool rounded = false) {
        norm = rms(x, w, eps);
        return had_mm(norm.y, weights, su, false, rounded);
    }
    virtual Transformed transform(Mat z, const Transform &t) {
        if (t.round_in)
            round(z);
        Transformed r;
        auto c = had(z);
        if (t.keep_c)
            r.c = c;
        if (!t.want_y)
            return r;
        auto y = alloc(c.rows, c.cols);
        if (t.post.p) {
            auto s = t.post;
            if (t.post_scale.p) {
                s = alloc(1, t.post.cols);
                mul(s, t.post, t.post_scale);
            }
            mul(y, c, s);
        } else
            copy(y, c);
        round(y);
        if (t.residual.p) {
            auto h = t.y.p ? t.y : alloc(y.rows, y.cols);
            add(h, t.residual, y, true);
            y = h;
        } else if (t.y.p) {
            copy(t.y, y);
            y = t.y;
        }
        r.y = y;
        return r;
    }
    // sum_j mm(had(x_j * sv_j * scale_j, transposed), w_j, transposed GEMM).
    virtual Mat had_mm_sum(std::span<const Mat> x, std::span<const Mat> sv, std::span<const Mat> scale,
                           std::span<const Weight> weights) {
        Mat total;
        for (size_t j = 0; j < x.size(); ++j) {
            auto svs = alloc(1, sv[j].cols);
            mul(svs, sv[j], scale[j]);
            auto d = had_mm(x[j], weights.subspan(j, 1), svs, true).front();
            if (!total.p)
                total = d;
            else
                add(total, total, d);
        }
        return total;
    }
    // SV/Wscale gradients from dy and the saved output transform c.
    virtual void sv_grads(Mat dy, Mat c, Mat sv, Mat scale, Mat gsv, Mat gscale) {
        auto dc = alloc(1, sv.cols, true);
        colsum(dy, c, dc);
        auto temp = alloc(1, sv.cols);
        mul(temp, dc, scale);
        add(gsv, gsv, temp);
        scale_sum(dc, sv, gscale);
    }
    virtual std::pair<Mat, Mat> swiglu_backward_su(Mat gate, Mat up, Mat dxsu, Mat su) {
        auto dp = alloc(dxsu.rows, dxsu.cols);
        mul(dp, dxsu, su);
        return swiglu_backward(gate, up, dp);
    }
    // dh += rms_backward(x, dy = dxsu * su); dw += norm-weight gradient.
    virtual void rms_backward_su(Mat x, Norm n, Mat w, Mat su, Mat dxsu, Mat dw, Mat dh) {
        auto dy = alloc(dxsu.rows, dxsu.cols);
        mul(dy, dxsu, su);
        auto d = rms_backward(x, n, w, dy, dw);
        add(dh, dh, d);
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
    mutable std::map<std::string, Mat> effective_;
    Mat parameter(const std::string &) const;
    Mat gradient(const std::string &) const;
    Mat effective(const std::string &, bool rounded = true) const;
    std::vector<Mat> epilogues(std::vector<Mat> z, int first, LinearSave *saves, Mat input, Mat residual = {},
                               Mat dst = {});
    Mat projection(Mat, int, LinearSave *save = nullptr, Mat residual = {}, Mat dst = {});
    // Backward through `count` projections sharing one input (and SU): returns
    // dxsu = had(sum_j W_j^T had^T(dy_j * svs_j)) and, if requested, bf16(dxsu * su).
    Transformed projections_backward(const LinearSave *, int count, Mat *dy, bool rounded_dx);
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
    Mat mlp(Mat, MlpSave *, Mat dst = {});

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
