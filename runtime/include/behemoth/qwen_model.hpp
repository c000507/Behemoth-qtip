#pragma once
// Qwen3.5 single-GPU QTIP inference model (hybrid Gated DeltaNet / gated full attention, dense or
// MoE MLP). Correctness-first runtime for eval (greedy / perplexity): recurrent per-token
// delta-rule prefill, bf16 KV cache attention on full-attention layers, trellis GEMV decode +
// dense GEMM prefill per linear. Linears load either QTIP trellis tensors or plain bf16 weights
// (unquantized checkpoints), enabling bf16-vs-quantized comparison of the same code path.
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
    int num_experts = 0, topk = 0, moe_inter = 0, shared_inter = 0;
    bool shared_gate = true;
    std::vector<unsigned char> full_attn;  // per layer
    int max_context = 8192;
    int key_dim() const { return lin_nk * lin_kd; }
    int value_dim() const { return lin_nv * lin_vd; }
    int conv_dim() const { return 2 * key_dim() + value_dim(); }
    bool is_full(int l) const { return full_attn[size_t(l)] != 0; }
    static QwenRuntimeConfig read(const std::filesystem::path &model_dir);
};

// Batched serving (continuous batching): max_seqs sequence slots, KV in a shared page pool sized from
// free GPU memory (kv_tokens > 0 caps it), reserve_gb left free per GPU.
struct QwenServeOptions {
    int max_seqs = 0;  // 0: legacy single-sequence model
    long kv_tokens = 0;
    double reserve_gb = 1.5;
    bool mtp = false;   // load the checkpoint's MTP head (mtp.*) for speculative decoding
    int mtp_seqs = 16;  // max sequences per verify pass (one extra DeltaNet state entry each)
};
struct QwenCand {
    float logit;
    int id;
};

class QwenModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    // device: HIP device index. chunk: max prefill tokens per pass (>=1).
    QwenModel(const std::filesystem::path &model_dir, int device, int chunk = 256,
              int max_context = 8192);
    // Pipeline split over several GPUs: contiguous layer ranges per device (even split, or
    // QWEN_SPLIT="k1,k2,..": first layer of each later stage); hidden state handed off per chunk.
    // tensor_parallel (2 GPUs): every layer sharded by heads / intermediate / vocab instead.
    QwenModel(const std::filesystem::path &model_dir, const std::vector<int> &devices, int chunk = 256,
              int max_context = 8192, bool tensor_parallel = false);
    QwenModel(const std::filesystem::path &model_dir, const std::vector<int> &devices, int chunk, int max_context,
              bool tensor_parallel, const QwenServeOptions &opt);
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

    // ---- serving API (QwenServeOptions::max_seqs > 0). Slots 0..max_seqs-1.
    int max_seqs() const;
    long kv_pages_total() const;
    long kv_pages_free() const;
    int page_tokens() const;
    int topk() const;  // candidates per row
    int seq_len(int slot) const;
    void seq_start(int slot);  // fresh sequence in slot (state reset, pages released)
    void seq_free(int slot);
    // Appends n tokens (chunked prefill). cand: topk() candidates of the last token (sorted). false:
    // not enough KV pages (nothing done).
    bool seq_prefill(int slot, const int *tokens, size_t n, QwenCand *cand);
    // One decode step for n sequences (token i appended to slots[i]); cand [n][topk()]. false: out
    // of KV pages (nothing done).
    bool decode_batch(const int *slots, const int *tokens, int n, QwenCand *cand);
    // Several sequences' prompt pieces in ONE forward pass (sum of n <= chunk()): shares the weight
    // reads among short prompts. cand: [number of parts with want][topk()], in order.
    struct PrefillPart {
        int slot;
        const int *tokens;
        size_t n;
        bool want;  // candidates of the piece's last token (the prompt's final piece)
        int next = -1;  // !want: the prompt token after the piece (MTP K/V of its last row)
    };
    bool prefill_multi(const std::vector<PrefillPart> &parts, QwenCand *cand);
    int chunk() const;
    // Copies sequence src (state + KV) into slot dst (benchmarks / prefix reuse).
    void seq_fork(int src, int dst);

    // ---- MTP speculative decoding (QwenServeOptions::mtp and an mtp.* head in the checkpoint).
    // Per sequence: last = newest sampled token (position seq_len, not in the KV yet), draft = MTP
    // guess for the token after it. Step: decode_verify (main model on [last, draft]) -> caller
    // accepts / rejects the draft from cand -> spec_accept -> mtp_draft for the next draft. decode_batch
    // stays usable (it keeps the MTP K/V complete), e.g. above a batch-size threshold.
    bool has_mtp() const;
    int mtp_seqs() const;  // max sequences per decode_verify
    // cand [2n][topk()]: row 2i = candidates after last (verifies drafts[i]), 2i+1 = after the draft.
    bool decode_verify(const int *slots, const int *tokens, const int *drafts, int n, QwenCand *cand);
    // accepted[i] (i over the last decode_verify's sequences): 1 commits last + draft, 0 commits last only.
    void spec_accept(const int *accepted);
    // cand [n][topk()]: MTP candidates for the token after last[i] (draft distribution).
    bool mtp_draft(const int *slots, const int *last, int n, QwenCand *cand);
};

}  // namespace behemoth
