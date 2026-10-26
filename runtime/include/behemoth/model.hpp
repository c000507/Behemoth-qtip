#pragma once
#include "attention.hpp"
#include "scheduler.hpp"
#include <array>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>

namespace behemoth {
struct InferenceBatch {
    AttentionLayout layout;
    std::vector<int> tokens, sample_rows;
    std::vector<uint64_t> sample_ids;
};
// Default largest number of sampled rows (requests) per step (RuntimeOptions::sampled_rows).
inline constexpr size_t max_sampled_rows = 256;
struct RuntimeOptions {
    std::filesystem::path model;
    std::array<int, 2> devices{0, 1};
    double memory_fraction = .9;
    // token_budget: AQLM <= 1024, QTIP <= 4096 (dense prefill chunks).
    size_t token_budget = 1024, prefill_chunk = 256;
    SchedulerPolicy scheduler; // decode-first prefill limits (scheduler.hpp)
    // Rows per step that may request logits (sizes the lm_head buffers and pinned logits);
    // 0: min(token_budget, max_sampled_rows). Tools that sample every prefill row set it to
    // the token budget.
    size_t sampled_rows = 0;
    int qtip_dense_rows = -1;     // >= 0 overrides qtip::dense_rows_threshold (0 disables)
    int qtip_row_transforms = -1; // >= 0 overrides qtip::row_transform_threshold (0 disables)
    bool rccl_shared_memory = false; // default: explicit pinned-host staging, no P2P assumptions
    bool zero_copy_transport = false; // --transport zerocopy: kernel-driven host-memory exchange
    // Largest batch (rows) exchanged by the zero-copy kernels (0: 8). QTIP batches of more
    // than 8 rows use the copy-engine channels instead (PREFILL_SCHED_REPORT.md).
    size_t zero_copy_rows = 0;
    bool profile = false;
    std::string projection_mode = "auto";
    std::string projection_search = "conservative";
    std::string code_layout = "group-major";
};
void tensor_parallel_smoke(std::array<int, 2> devices = {0, 1}, bool rccl_shared_memory = false);
class TensorParallelModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit TensorParallelModel(const RuntimeOptions &);
    ~TensorParallelModel();
    int cache_pages() const;
    nlohmann::json last_profile() const;
    nlohmann::json projection_tuning(int layer = 0) const;
    // Full-vocab logits for the batch's sampled rows in a persistent pinned
    // host buffer, rank-blocked: [rank0 n x 16384 | rank1 n x 16384]. The
    // span is valid until the next forward call. Sampling uses the split
    // sample_logits overload; no per-step host staging vectors exist.
    std::span<const float> forward(const InferenceBatch &);
    std::vector<BFloat16> block_for_validation(int layer, std::span<const BFloat16> input);
};
InferenceBatch make_inference_batch(const Scheduler &, const ScheduledBatch &);
// QTIP prefill steps of at least `rows` rows run as two overlapped micro-batches (dense path,
// copy-engine transport). -1: 2x the dense threshold (default, env BEHEMOTH_QTIP_OVERLAP_ROWS),
// 0: never.
void set_qtip_overlap_rows(int rows);
} // namespace behemoth
