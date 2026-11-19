// Native BitNet b1.58 distillation trainer (C++20 + CUDA + cuBLAS). See docs/BITNET.md.
#pragma once
#include "kernels.hpp"
#include "train.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace qtip {
class SafeTensorsReader;
}
namespace bitnet {
namespace fs = std::filesystem;
using json = nlohmann::json;
using qat::require;

// RAII device allocation.
template <class T> struct Dev {
    T *p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) { reset(count); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
    Dev(Dev &&o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
    Dev &operator=(Dev &&o) noexcept {
        if (this != &o) {
            free();
            p = o.p, n = o.n;
            o.p = nullptr, o.n = 0;
        }
        return *this;
    }
    ~Dev() { free(); }
    void reset(size_t count);
    void free();
    size_t bytes() const { return n * sizeof(T); }
};

struct Options {
    fs::path model, data, work, resume, export_dir, eval_data, trace_tokens, trace_out;
    bool eval_only = false, export_only = false, no_quant = false, no_subln = false, attn_distill_rope = false;
    int rows = 1, accum = 1, loss_rows = 256, max_steps = 0, warmup = 20, checkpoint_every = 50, device = 0,
        attn_layer = -1, eval_rows = 0, log_every = 1, stop_after = 0, seed = 1;
    long max_tokens = 1'000'000;
    float lr = 1e-4f, norm_lr = 1e-4f, decay = 0.f, clip = 1.f, temperature = 2.f, ce_w = 1.f, kl_w = 1.f,
          attn_w = 0.f, min_lr_ratio = 0.1f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f;
};

// One linear layer. `w` views the trainable flat parameter vector (fp32 latent weights) of the student;
// the frozen teacher keeps only `wq` (dense bf16).
struct Linear {
    std::string name;
    int out = 0, in = 0;
    size_t offset = 0;           // offset into the flat parameter vector (student)
    Dev<uint16_t> wq;            // GEMM operand [out, in]: ternary values or dense bf16
    float scale = 1;             // absmean scale (student), host copy after ternarize
    Dev<float> dscale;           // device scale
};
struct LayerParams {
    Linear lin[7]; // q k v o gate up down
    size_t norm1 = 0, norm2 = 0, sub_attn = 0, sub_ffn = 0; // offsets of norm gains
};
struct Geometry {
    int n, inter, heads, kv, dim, vocab, layers;
    float eps, theta;
    bool tied;
};

struct Workspace; // per-sequence activation buffers (defined in model.cpp)

class Model {
  public:
    Geometry g;
    bool quant;     // BitLinear (true) or dense bf16 operands
    bool subln;     // SubLN before o_proj / down_proj
    bool trainable; // latent fp32 weights in the flat parameter vector (student); teacher: operands only
    std::vector<LayerParams> layers;
    size_t final_norm = 0;
    size_t weight_params = 0, norm_params = 0, total_params = 0; // flat layout: [weights | norms]
    Dev<float> params, grad, m, v; // student only (teacher: params holds nothing)
    Dev<uint16_t> embed, head;     // bf16 frozen [vocab, n]; head aliases embed when tied
    Dev<float> final_norm_w;        // teacher's final norm (student: inside params)
    std::vector<Dev<float>> teacher_norms; // teacher norm gains per layer (2 each)
    Dev<float> scratch;             // small device scratch
    Model(const Geometry &, bool quant, bool subln, bool trainable);
    // Load bf16 HF Llama weights (teacher: into wq/norm buffers; student: latent fp32 params + wq rebuild).
    void load_hf(const qtip::SafeTensorsReader &, cudaStream_t);
    void load_params(const std::vector<float> &host, cudaStream_t); // student flat params
    std::vector<float> host_params() const;
    void rebuild_operands(cudaStream_t); // ternarize student latent weights into wq
    const float *norm_ptr(size_t offset, int layer, int which) const;
    const float *param_ptr(size_t offset) const { return params.p + offset; }
    float *grad_ptr(size_t offset) const { return grad.p + offset; }
    json layout() const;
};

struct Batch {
    int rows = 0;
    Dev<int32_t> tokens, labels, positions, starts;
    std::vector<int32_t> h_labels;
    int targets = 0;
};
Batch upload_batch(const std::vector<qat::Batch> &, cudaStream_t);

struct LossReport {
    double ce = 0, kl = 0, attn = 0, count = 0;
};
class Trainer {
  public:
    Trainer(Model &student, Model *teacher, const Options &, cudaStream_t);
    ~Trainer();
    // Forward + backward on one batch; accumulates gradients. Returns summed losses.
    LossReport step(const Batch &);
    // Forward only; per-row CE written to row_loss (host) when requested. Returns summed CE and count.
    LossReport evaluate(Model &, const Batch &, std::vector<float> *row_loss, std::vector<float> *logits = nullptr);
    double grad_norm();
    std::vector<float> gradient() const;
    void optimizer_step(float lr, float norm_lr, float clip_scale, int step);
    void zero_grad();
    cublasHandle_t handle() const { return blas_; }

  private:
    Model &student_;
    Model *teacher_;
    Options opt_;
    cudaStream_t s_;
    cublasHandle_t blas_ = nullptr;
    std::unique_ptr<Workspace> ws_;
    void ensure(int rows);
    void forward_layer(Model &, int layer, const Batch &, Workspace &, bool keep_qkv);
    void backward_layer(Model &, int layer, const Batch &, Workspace &);
    double logits_loss(Model &, const Batch &, Workspace &, bool backward, const float *teacher, LossReport &,
                       std::vector<float> *row_loss, std::vector<float> *logits);
    void teacher_forward(const Batch &, Workspace &);
    void relation_loss(const Batch &, Workspace &, int layer);
};
// Flat gemm helper (row-major semantics): C[M,N] = alpha op(A) op(B) + beta C, bf16 inputs, fp32 output.
void gemm(cublasHandle_t, bool transA, bool transB, int M, int N, int K, const uint16_t *A, int lda,
          const uint16_t *B, int ldb, float *C, int ldc, float alpha = 1.f, float beta = 0.f, int batch = 1,
          long strideA = 0, long strideB = 0, long strideC = 0);

// Checkpoints (safetensors + metadata.json + latest.json) and export of the packed ternary model.
struct State {
    int step = 0;
    long tokens = 0;
    size_t position = 0; // dataset row index
};
void save_checkpoint(const fs::path &dir, const Model &, const State &, const Options &, const json &extra);
State load_checkpoint(const fs::path &dir, Model &, cudaStream_t);
void export_model(const fs::path &dest, const fs::path &source, const Model &, const json &info);
Geometry geometry(const json &config);
} // namespace bitnet
