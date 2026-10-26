#pragma once
#include "attention.hpp"
#include "prefix_cache.hpp"
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
    // Experimental: BF16 rotated-output exchange for QTIP pure-prefill micro-batches
    // of at least this size (0: off; otherwise 64..4096). Adds BF16 rounding to
    // q/k/v, o and down; decode-only and mixed steps keep existing numerics.
    int qtip_bf16_prefill_rows = 0;
    bool rccl_shared_memory = false; // default: explicit pinned-host staging, no P2P assumptions
    bool zero_copy_transport = false; // --transport zerocopy: kernel-driven host-memory exchange
    // Largest batch (rows) exchanged by the zero-copy kernels (0: 8). QTIP batches of more
    // than 8 rows use the copy-engine channels instead (PREFILL_SCHED_REPORT.md).
    size_t zero_copy_rows = 0;
    bool profile = false;
    std::string projection_mode = "auto";
    std::string projection_search = "conservative";
    std::string code_layout = "group-major";
    // Prefix cache (PREFIX_CACHE_REPORT.md): device staging pages per rank and buffer for KV
    // page save/restore (two buffers per rank, allocated before the KV pool is sized; 0: no
    // transfers). The engine sets it when the prefix cache is enabled.
    int kv_transfer_pages = 0;
    PrefixCacheOptions prefix_cache{0, 0, 64, true}; // slots 0: disabled (server default 10)
    double prefix_cache_gb = 64;                     // host memory cap (both ranks), 1e9 bytes
    size_t prefix_cache_restore_pace = 0; // restore pages issued per step while decoding (0: all)
};
void tensor_parallel_smoke(std::array<int, 2> devices = {0, 1}, bool rccl_shared_memory = false);
class TensorParallelModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    uint64_t kv_start(bool save, std::span<const int> gpu, std::span<const int> host);

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

    // Prefix-cache KV transfers (requires kv_transfer_pages > 0). A host page index names one
    // 16-token KV page of both ranks (kv_page_bytes() each) in a pinned, NUMA-local arena that
    // grows in 64-page blocks and is reused, never freed before shutdown.
    size_t kv_page_bytes() const;
    // Grows the arena to >= pages (both ranks in parallel, outside the transfer lock;
    // ~50 ms per 64-page block per rank here). Thread-safe. Call it while the GPUs are idle:
    // mapping pinned memory while kernels run triggered "MES failed to respond to
    // msg=INVALIDATE_TLBS" on this machine (PREFIX_CACHE_REPORT.md); the engine reserves the
    // whole cap at startup.
    void kv_host_reserve(size_t pages);
    size_t kv_host_pages() const; // host pages backed by the arena
    size_t kv_host_bytes() const; // pinned bytes allocated, both ranks
    std::span<const std::byte> kv_host_page(int rank, int page) const;
    // Asynchronous copies on dedicated streams (pack/unpack kernels + copy engines through two
    // device staging buffers per rank); they run concurrently with forward() and are executed
    // in issue order. The caller keeps the GPU pages out of every batch until complete.
    uint64_t kv_save(std::span<const int> gpu_pages, std::span<const int> host_pages);
    uint64_t kv_restore(std::span<const int> host_pages, std::span<const int> gpu_pages);
    // True once complete (the id is then forgotten); gpu_ms: per-rank GPU time of the copy.
    bool kv_poll(uint64_t transfer, std::array<float, 2> *gpu_ms = nullptr);
    // Blocks until the transfer is complete (kv_poll then returns true).
    void kv_wait(uint64_t transfer);
};
InferenceBatch make_inference_batch(const Scheduler &, const ScheduledBatch &);
// QTIP prefill steps of at least `rows` rows run as two overlapped micro-batches (dense path,
// copy-engine transport). -1: 2x the dense threshold (default, env BEHEMOTH_QTIP_OVERLAP_ROWS),
// 0: never.
void set_qtip_overlap_rows(int rows);
} // namespace behemoth
