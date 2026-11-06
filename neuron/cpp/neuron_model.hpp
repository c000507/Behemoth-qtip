#pragma once
// Llama on two NeuronCores (TP=2) behind the server's model contract: forward(InferenceBatch) -> logits.
//
// All weights (FP8 shards expanded from QTIP / NVFP4 / FP8 checkpoint tensors at load), the paged KV cache and
// the activations stay on the devices; the host only supplies token embeddings, RoPE rows, page tables and
// slots each step, and reads back the logits of the sampled rows. One layer NEFF per (rows, queries, table
// pages) bucket serves all layers (trace_model.py); a head NEFF does the final norm and lm_head.
#include "behemoth/model.hpp"
#include <filesystem>
#include <memory>
#include <span>

namespace neuron {
struct ModelOptions {
    std::filesystem::path model;    // Behemoth/QTIP checkpoint directory
    std::filesystem::path neff_dir; // subdirectories with meta.json + graph.neff (trace_model.py outputs)
    int load_threads = 4;           // layers expanded and uploaded concurrently
    bool verbose = true;
};

class NeuronModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit NeuronModel(const ModelOptions &);
    ~NeuronModel();
    int cache_pages() const;  // usable KV pages (one more is kept as scratch for padding rows)
    int vocab() const;
    const std::vector<int> &eos() const;
    int max_context() const;
    // Logits of batch.sample_rows in order: [n_sample][vocab] (valid until the next call).
    std::span<const float> forward(const behemoth::InferenceBatch &);
    nlohmann::json last_profile() const;
};
} // namespace neuron
