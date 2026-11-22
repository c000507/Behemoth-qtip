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
#include <stdexcept>

namespace neuron {
struct ModelOptions {
    std::filesystem::path model;    // Behemoth/QTIP checkpoint directory
    std::filesystem::path neff_dir; // subdirectories with meta.json + graph.neff (trace_model.py outputs)
    int load_threads = 8;           // layers expanded and uploaded concurrently (8: one per trn1.2xlarge vCPU, 67 s vs 89 s at 4 for 24B FP8)
    bool verbose = true;
    // QTIP layers: "expanded" decodes them to FP8 shards at load (the fast path, needs WEIGHTS=fp8 layer NEFFs);
    // "compressed" keeps the trellis resident and decodes in the graph every step (WEIGHTS=qtip NEFFs, ~1/4 of the
    // HBM, far slower per token); "auto" compresses everything, then expands layers from 0 up while the HBM budget
    // (hbm_gib - hbm_reserve_gib, minus KV, head and the compressed rest) allows. A mixed model needs both NEFF kinds.
    std::string qtip_mode = "expanded";
    double hbm_gib = 16.0, hbm_reserve_gib = 2.0;
};

// A model family served behind NeuronModel's contract by its own graphs and runtime (the Llama graphs are Impl).
// GLM-5.3 (glm5_next: KDA, sparse MLA, mHC, MoE) is one: glm_model.cpp, serving a glm_repack.py prepared directory.
class Backend {
  public:
    virtual ~Backend() = default;
    virtual int cache_pages() const = 0;
    virtual int vocab() const = 0;
    virtual int max_decode_rows() const = 0;
    virtual int decode_bucket_pages(int pages) const = 0;
    virtual const std::vector<int> &eos() const = 0;
    virtual int max_context() const = 0;
    virtual std::span<const float> forward(const behemoth::InferenceBatch &) = 0;
    virtual nlohmann::json last_profile() const = 0;
    // MTP speculative decoding (qwen35_model.cpp): 0 / {} for backends without it
    virtual int mtp_max_drafts() const { return 0; }
    virtual int mtp_max_drafts(int, int) const { return 0; } // per request, in a step of n decode requests at a page count
    virtual std::vector<int> mtp_draft(std::span<const int>, int, std::span<const int>, int) { return {}; }
    // Prefix cache of a recurrent family (qwen35_model.cpp): 0 / no-ops for backends without it (the engine then keeps
    // the cache off). K/V pages as NeuronModel::kv_save / kv_restore; the recurrent state of ONE sequence (identified by
    // its first page) as state_bytes() host bytes.
    virtual size_t kv_page_bytes() const { return 0; }
    virtual void kv_copy(std::span<const int>, std::span<const int>, std::byte *, bool) { throw std::logic_error("no K/V copies for this family"); }
    virtual size_t state_bytes() const { return 0; }
    virtual void state_save(int, std::byte *) { throw std::logic_error("no state snapshots for this family"); }
    virtual void state_restore(int, const std::byte *) { throw std::logic_error("no state snapshots for this family"); }
    virtual void state_drop(int) {}
};
// options.model is a GLM prepared directory (prepared.json written by neuron/glm_repack.py).
bool is_glm_prepared(const std::filesystem::path &);
std::unique_ptr<Backend> make_glm_backend(const ModelOptions &);
// options.model is a Qwen3.5 MoE prepared directory (prepared.json written by neuron/qwen35_prepare.py): qwen35_model.cpp.
bool is_qwen35_prepared(const std::filesystem::path &);
std::unique_ptr<Backend> make_qwen35_backend(const ModelOptions &);
// options.model is an Inkling 2 Mini prepared directory (prepared.json written by neuron/inkling_prepare.py): inkling_model.cpp.
bool is_inkling_prepared(const std::filesystem::path &);
std::unique_ptr<Backend> make_inkling_backend(const ModelOptions &);

class NeuronModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<Backend> backend_; // set instead of impl_ for a non-Llama family (GLM)

  public:
    explicit NeuronModel(const ModelOptions &);
    ~NeuronModel();
    int cache_pages() const;  // usable KV pages (one more is kept as scratch for padding rows)
    int vocab() const;
    int max_decode_rows() const;          // largest R of the decode buckets (rows a step can verify at once)
    int decode_bucket_pages(int pages) const; // table pages of the smallest decode bucket covering `pages` (0: none)
    bool has_shared_verify() const;           // the graph set has shared-table verify buckets (dec8s_*)
    const std::vector<int> &eos() const;
    int max_context() const;
    // Recurrent state per sequence (GLM's KDA layers): the engine then neither drafts (rejected draft rows would advance
    // the state) nor uses the prefix cache (restored K/V pages carry no recurrent state).
    bool recurrent() const;
    // Logits of batch.sample_rows in order: [n_sample][vocab] (valid until the next call).
    // want_argmax: greedy rows may be sampled on the device when an argmax head bucket covers the launch
    // (device_token(row) >= 0 then; -1 rows still come from the returned logits).
    std::span<const float> forward(const behemoth::InferenceBatch &, bool want_argmax = false);
    bool device_argmax_used() const;
    int device_token(size_t logits_row) const;
    nlohmann::json last_profile() const;
    // MTP speculative decoding by the checkpoint's own draft head (Qwen3.5; docs/neuron-qwen397-mtp.md): the largest number of
    // drafts a verify step takes (0: none), and after the sampling of a step that was one decode request at `position` whose
    // emitted tokens are `next`: commit the recurrent state of the accepted prefix and draft `want` tokens after next.back().
    int mtp_max_drafts() const;
    int mtp_max_drafts(int requests, int pages) const;
    std::vector<int> mtp_draft(std::span<const int> pages, int position, std::span<const int> next, int want);
    // Prefix cache: host bytes of one 16-token page (K/V and FP8 scales of every layer and rank), and synchronous copies
    // between pool pages and host pages of `arena` (host page h at arena + h * kv_page_bytes()). Call between forward()s.
    size_t kv_page_bytes() const;
    void kv_save(std::span<const int> device_pages, std::span<const int> host_pages, std::byte *arena);
    void kv_restore(std::span<const int> host_pages, std::span<const int> device_pages, const std::byte *arena);
    // Recurrent families with state snapshots (Qwen3.5; docs/neuron-qwen35-prefix-cache.md): host bytes of one sequence's
    // recurrent state over all layers and ranks (0: the family has none or cannot snapshot it: no prefix cache then).
    // A sequence is identified by its first page. Call between forward()s, at a COMMITTED position only (after
    // mtp_draft() of a verify step): state_save copies the state the sequence has now (the state after its last
    // forwarded / accepted token), state_restore makes `src` the state of the sequence that will continue on that first
    // page (its next token is at the snapshot's position, > 0), state_drop forgets a finished sequence's state.
    size_t state_bytes() const;
    void state_save(int first_page, std::byte *dst);
    void state_restore(int first_page, const std::byte *src);
    void state_drop(int first_page);
};
} // namespace neuron
