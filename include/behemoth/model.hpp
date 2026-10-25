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
struct RuntimeOptions {
    std::filesystem::path model;
    std::array<int, 2> devices{0, 1};
    double memory_fraction = .9;
    size_t token_budget = 1024, prefill_chunk = 256;
    bool rccl_shared_memory = false; // default: explicit pinned-host staging, no P2P assumptions
    bool zero_copy_transport = false; // --transport zerocopy: kernel-driven host-memory exchange
    size_t zero_copy_rows = 8;        // largest batch (rows) exchanged by the zero-copy path
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
    std::vector<std::vector<float>> forward(const InferenceBatch &);
    std::vector<BFloat16> block_for_validation(int layer, std::span<const BFloat16> input);
};
InferenceBatch make_inference_batch(const Scheduler &, const ScheduledBatch &);
} // namespace behemoth
