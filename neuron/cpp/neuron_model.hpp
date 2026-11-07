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
    // QTIP layers: "expanded" decodes them to FP8 shards at load (the fast path, needs WEIGHTS=fp8 layer NEFFs);
    // "compressed" keeps the trellis resident and decodes in the graph every step (WEIGHTS=qtip NEFFs, ~1/4 of the
    // HBM, far slower per token); "auto" compresses everything, then expands layers from 0 up while the HBM budget
    // (hbm_gib - hbm_reserve_gib, minus KV, head and the compressed rest) allows. A mixed model needs both NEFF kinds.
    std::string qtip_mode = "expanded";
    double hbm_gib = 16.0, hbm_reserve_gib = 2.0;
};

class NeuronModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit NeuronModel(const ModelOptions &);
    ~NeuronModel();
    int cache_pages() const;  // usable KV pages (one more is kept as scratch for padding rows)
    int vocab() const;
    int max_decode_rows() const;          // largest R of the decode buckets (rows a step can verify at once)
    int decode_bucket_pages(int pages) const; // table pages of the smallest decode bucket covering `pages` (0: none)
    const std::vector<int> &eos() const;
    int max_context() const;
    // Logits of batch.sample_rows in order: [n_sample][vocab] (valid until the next call).
    std::span<const float> forward(const behemoth::InferenceBatch &);
    nlohmann::json last_profile() const;
    // Prefix cache: host bytes of one 16-token page (K/V and FP8 scales of every layer and rank), and synchronous copies
    // between pool pages and host pages of `arena` (host page h at arena + h * kv_page_bytes()). Call between forward()s.
    size_t kv_page_bytes() const;
    void kv_save(std::span<const int> device_pages, std::span<const int> host_pages, std::byte *arena);
    void kv_restore(std::span<const int> host_pages, std::span<const int> device_pages, const std::byte *arena);
};
} // namespace neuron
