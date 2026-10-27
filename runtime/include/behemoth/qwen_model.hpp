#pragma once
// Qwen3.5 single-GPU QTIP inference model (hybrid Gated DeltaNet / gated full attention, dense MLP).
// Correctness-first runtime for eval (greedy / perplexity): recurrent per-token delta-rule prefill,
// bf16 KV cache attention on full-attention layers, trellis GEMV decode + dense GEMM prefill per
// linear. MoE checkpoints are rejected with a clear error until implemented.
#include "qtip_gpu.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace behemoth {

struct QwenRuntimeConfig {
    // parsed from the quantized checkpoint's config.json (original qwen3_5 fields)
    int layers = 0, hidden = 0, inter = 0, vocab = 0;
    int nq = 0, nkv = 0, hd = 0, rot_dim = 0;
    float theta = 1e7f, eps = 1e-6f;
    bool attn_output_gate = true;
    int lin_nk = 0, lin_kd = 0, lin_nv = 0, lin_vd = 0, conv_k = 0;
    bool tied = true;
    int num_experts = 0;
    std::vector<unsigned char> full_attn;  // per layer
    int max_context = 8192;
    int key_dim() const { return lin_nk * lin_kd; }
    int value_dim() const { return lin_nv * lin_vd; }
    int conv_dim() const { return 2 * key_dim() + value_dim(); }
    bool is_full(int l) const { return full_attn[size_t(l)] != 0; }
    static QwenRuntimeConfig read(const std::filesystem::path &model_dir);
};

class QwenModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    // device: HIP device index. chunk: max prefill tokens per pass (>=1).
    QwenModel(const std::filesystem::path &model_dir, int device, int chunk = 256,
              int max_context = 8192);
    ~QwenModel();
    QwenModel(const QwenModel &) = delete;
    QwenModel &operator=(const QwenModel &) = delete;
    const QwenRuntimeConfig &config() const;
    // Resets all state (KV, conv rings, delta states, position).
    void reset();
    int position() const;
    // Prefills tokens (advances position); logits of the LAST token -> host [vocab] (optional out).
    void prefill(const int *tokens, size_t n, float *last_logits = nullptr);
    // One decode step (advances position by one); logits -> host [vocab] (optional out).
    void decode(int token, float *logits = nullptr);
    // Convenience: prefill + greedy generation of `n` tokens, returns generated ids.
    std::vector<int> generate(const int *prompt, size_t n_prompt, size_t n_new);
    // Mean negative log-likelihood of the given tokens (assumes position 0; resets first).
    double perplexity(const int *tokens, size_t n);
};

}  // namespace behemoth
