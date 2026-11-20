// Qwen3.5 MoE (qwen3_5_moe: Gated DeltaNet linear attention, GQA full attention, MoE) on NeuronCores behind
// NeuronModel's contract: forward(InferenceBatch) -> logits. Derived from glm_model.cpp (the GLM-5.3 backend): same
// prepared-directory format (neuron/qwen35_prepare.py: prepared.json, embed.bin, r<rank>/L<layer>.bin, r<rank>/head.bin),
// same step logic; the differences are the sub-block kinds (gdn | gqa attention, moe MLP, fused layer_<attn>_moe), the
// residual being a plain [T, hidden] vector (no hyper-connection streams), the host-computed RoPE cos/sin rows of the
// full-attention layers (partial rotary, 64 dims, theta 1e7) and the KV cache being the state of the gqa layers.
//
// State: the gqa layers' K/V cache is paged with the server's 16-token pages (page tables from the scheduler, page
// pages-1 is scratch). Gated DeltaNet recurrent + conv state lives in [slots, ...] tensors per linear layer, one slot per
// sequence: a sequence is identified by its first page (stable while it lives; a new or re-prefilled sequence starts at
// position 0, where the graph zeroes its state). Slots are an LRU cache: an evicted sequence's state is copied to host
// memory and copied back when it continues, so more sequences may be live than there are slots. The last slot is scratch
// (padding rows). Padding tokens of a prefill chunk have position -1 (the graph leaves the state untouched). The engine
// disables prompt-lookup speculation for this family (NeuronModel::recurrent()); the prefix cache is opt-in
// (BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE=1): the engine snapshots a sequence's recurrent state between steps
// (state_save / state_restore below) next to the K/V pages of the gqa layers (kv_copy) and restores only at snapshot
// positions (docs/neuron-qwen35-prefix-cache.md).
#include "neuron_model.hpp"
#include "behemoth/worker.hpp"
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

using namespace behemoth;

namespace neuron {
namespace {
constexpr int PAGE = 16;
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void check(NRT_STATUS status, const std::string &what) {
    if (status != NRT_SUCCESS)
        throw std::runtime_error("Qwen35: " + what + " failed: nrt status " + std::to_string(int(status)));
}
struct Dev {
    nrt_tensor_t *t = nullptr;
    size_t bytes = 0;
};
Dev alloc(int core, size_t bytes, const std::string &name) {
    Dev d;
    d.bytes = bytes;
    check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, core, bytes, name.c_str(), &d.t), "alloc " + name);
    return d;
}
void put(const Dev &d, const void *src, size_t bytes, size_t offset = 0) { check(nrt_tensor_write(d.t, src, offset, bytes), "write"); }
// BEHEMOTH_NEURON_QWEN35_ASYNC=1 (docs/neuron-reap-launches.md): implicit async execution (nrt_execute queues the launch and returns);
// every host read of a launch's output, and the end of every execute() / mtp_launch(), first drains the rank's queue, so host
// writes (next step's inputs, state spill / restore, prefix-cache copies) only ever happen on an idle core.
bool async_exec() {
    static const bool on = std::getenv("BEHEMOTH_NEURON_QWEN35_ASYNC") && std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_ASYNC"));
    return on;
}
int ltrace_every() {
    static const int n = std::getenv("BEHEMOTH_NEURON_QWEN35_LTRACE") ? std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_LTRACE")) : 0;
    return n;
}
void drain(int core) {
    if (async_exec())
        check(nrt_async_drain_queued_execs(core), "drain");
}
void zero(const Dev &d) {
    std::vector<uint8_t> z(std::min<size_t>(d.bytes, 64u << 20), 0);
    for (size_t off = 0; off < d.bytes; off += z.size())
        put(d, z.data(), std::min(z.size(), d.bytes - off), off);
}
std::vector<char> slurp(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot read " + p.string());
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
json read_json(const std::filesystem::path &p) {
    std::ifstream f(p);
    if (!f)
        throw std::runtime_error("cannot read " + p.string());
    return json::parse(f);
}
float bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// All rank threads meet here after every launch when BEHEMOTH_NEURON_QWEN35_BARRIER=1: no rank starts launch i + 1 (whose
// all-reduce pairs with the other ranks' launch i + 1) before every rank has returned from launch i. Spins (launches are
// sub-millisecond); a failed rank releases the others.
struct SpinBarrier {
    std::atomic<int> count{0}, generation{0};
    std::atomic<bool> failed{false};
    int n;
    explicit SpinBarrier(int ranks) : n(ranks) {}
    void wait() {
        const int g = generation.load();
        if (count.fetch_add(1) + 1 == n) {
            count.store(0);
            generation.fetch_add(1);
            return;
        }
        while (generation.load() == g) {
            if (failed.load())
                throw std::runtime_error("Qwen35: another rank failed");
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
};

// One NEFF: a sub-block kind ("gdn", "gqa", "dense", "moe"), a whole layer ("layer_<attn>_<mlp>": both sub-blocks,
// weights named a.* / m.*) or "head", for one bucket.
struct Graph {
    std::string kind, attn, mlp; // attn / mlp: the sub-blocks it contains ("" for none)
    std::vector<std::string> seg; // "seg_gdn_gdn_gdn_gqa": the attention kinds of the consecutive whole layers it runs
    bool has_gqa = false;         // binds a page table (gqa sub-block, or a segment containing one)
    bool has_gdn = false;         // runs gated-delta layers (gdn sub-block, or a segment containing one): defines a decode row bucket
    bool bound = false;           // meta gdn_slot "bound": a gdn graph that takes every row's state as its own one-slot tensor (no slot / vslot
                                  // input); the engine binds the slices of the launch's slots (docs/neuron-reap-nonfinite.md: no dynamic state DMA)
    std::vector<std::string> bound_states; // bound: the NEFF's state inputs in order ("kstate" or "kstate_<row>"); states then holds the base names
    std::vector<std::string> extra;        // bound verify: outputs 1.. before the states, "kstate@<i>": the state after position i -> slot vslot[i]
    bool verify = false;          // speculative verify bucket (R = 1, Q tokens of one request, input vslot: per-position gdn state slots)
    bool mtp = false;             // the MTP head (kind mtp): outputs z, logits, then the aliased MTP KV cache
    size_t out0 = 0;              // mtp graphs: bytes of output 0
    int chain = 0;                // kind mtpc: the head's first pass + chain - 1 chained passes in ONE launch (in-graph argmax across the
                                  // ranks, vocabulary-sharded embedding table); outputs drafts [1, chain] f32, then the aliased KV cache
    int R = 0, Q = 0, P = 0, T = 0, S = 0, pages = 0, slots = 0;
    int moe_stat = 0;             // kind moeg: output1 holds moe_stat more values after the load (the largest expert loads, for the load log)
    int moe_cap = 0;              // kind moeg (meta moe "group", "moe_cap" > 0): the prefill MoE by expert groups (qwen35_graphs.moe_grouped);
                                  // output1 = the largest token count of a grouped expert, > moe_cap: the launch is rerun with the moe NEFF
    int egu_t = -1;                          // meta "egu_t": experts stored as egu_t [E * 128, 2048] (nki_moe.relayout_egu); -1: no MoE
    std::vector<std::string> inputs, states; // NEFF inputs: inputs..., then the aliased states (module buffers)
    std::vector<nrt_model_t *> model;        // [rank]
    std::vector<size_t> input_bytes;          // by NEFF input index
    std::vector<size_t> state_bytes;
    std::filesystem::path neff;
};
} // namespace

class Qwen35Model final : public Backend {
    ModelOptions options;
    json prepared;
    std::filesystem::path root;
    int tp = 1, hidden = 0, streams = 1, rot = 0, vocab_ = 0, vocab_shard = 0, layers = 0, pages_total = 0, slots_total = 0, max_context_ = 0;
    std::vector<std::string> attn_kind, mlp_kind;
    std::vector<int> eos_;
    double theta = 1e7;
    // embedding table (bf16 [vocab][hidden]) mapped from embed.bin
    const uint16_t *embed = nullptr;
    size_t embed_map_bytes = 0;
    void *embed_map = nullptr;
    std::vector<Graph> graphs;
    struct Rank {
        std::map<std::string, Dev> t; // "L003.attn.q", "L003.attn.kstate", "head.w"
    };
    std::vector<Rank> ranks;
    std::vector<std::unique_ptr<RankWorker>> workers;
    // step buffers per (R, Q) group and rank; tables per (R, Q, P)
    struct Step {
        Dev x[2], pos, slot, vslot, wslot, cos_sin;
        std::map<int, Dev> table, rows, logits; // table by P; head rows / logits by the head's S
    };
    std::map<std::pair<int, int>, std::vector<Step>> steps;
    // tensor sets per (graph, layer, parity, rank); bound graphs (Graph::bound) also per the launch's slots
    std::map<std::tuple<std::string, int, int>, Dev> state_slices; // (layer state name, slot, rank): one slot of a layer state
    std::map<std::tuple<const Graph *, int, int, int, std::vector<int64_t>>, std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *>> bound_sets;
    std::map<std::tuple<const Graph *, int, int, int>, std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *>> sets;
    std::mutex sets_mutex; // layer_set / head_set run on every rank's worker thread concurrently
    std::vector<float> logits;
    json profile = json::object();
    // GDN state slots: sequence (first page) -> slot, LRU order; spilled states on the host
    std::map<int, int> slot_of;
    std::list<int> lru; // sequences with a slot, most recent last
    std::map<int, std::vector<std::vector<char>>> spilled; // sequence -> [rank] bytes of every GDN layer's slot
    uint64_t spills = 0, restores = 0, state_saves = 0, state_restores = 0;
    // ---- MTP speculative decoding (docs/neuron-qwen397-mtp.md). The head's weights come from BEHEMOTH_NEURON_MTP_DIR
    // (qwen35_prepare.py --mtp: mtp.json + r<rank>/mtp.bin, tensors "MTP.*"); it needs mtp graphs and verify buckets.
    json mtp_manifest;
    std::filesystem::path mtp_root;
    bool mtp_ready = false;
    struct MtpStep {
        Dev e, pos, cos_sin, wslot, lo, sel;
        std::map<int, Dev> table; // by P
    };
    std::map<int, std::vector<MtpStep>> mtp_steps; // by the graph's Q, per rank
    std::vector<std::array<Dev, 2>> mtp_z;          // per rank: the chained residual, ping-pong
    struct MtpChain {
        Dev cpos, ccos, cwslot, drafts;
    };
    std::map<int, std::vector<MtpChain>> mtp_chain; // by the chain length D, per rank
    std::vector<int> mtp_chain_out;                 // the drafts of the last chain launch
    std::array<float, 64> mtp_chain_raw{};
    std::vector<Dev> mtp_logits_dev;                // per rank: this rank's vocabulary shard of the draft row
    std::vector<float> mtp_logits;
    struct MtpSeq {
        int lo = 0, next = -1; // MTP cache entries exist for positions lo .. next - 1
    };
    std::map<int, MtpSeq> mtp_seq; // by sequence (first page)
    // the decode / verify requests of the last forward(): what mtp_draft() commits and drafts from. One entry per request
    // (row `row` of an R x Q launch; vslot: the row's per-position state slots, empty for a plain decode row); x_valid: the
    // launch's residual buffer still holds this row (no later launch of the same forward() reused the bucket)
    struct Last {
        bool valid = false, x_valid = true;
        int seq = -1, position = 0, count = 0, Q = 0, parity = 0, R = 1, row = 0;
        std::vector<int64_t> vslot;
        int launch = 0; // which execute() of the forward()
    };
    std::vector<Last> lasts;
    int last_requests = 1; // decode / verify requests of the last forward()
    Last last;
    std::map<std::tuple<int, int, int, int, int>, Dev> x_slices; // (R, Q, parity, row, rank): a row's residual rows inside an R x Q step buffer
    std::set<int> reserved; // slots holding per-position states of an uncommitted verify launch
    int final_parity = 0;
    double mtp_ms = 0, mtp_pmin = 0, mtp_last_p = 1;
    uint64_t mtp_launches = 0;
    std::map<int, std::vector<int64_t>> mtp_static; // by Q: table | lo | sel as last written (rewritten only when they change)

  public:
    explicit Qwen35Model(const ModelOptions &o) : options(o), root(o.model) {
        prepared = read_json(root / "prepared.json");
        const auto &g = prepared.at("geometry");
        tp = g.at("tp");
        hidden = g.at("hidden");
        rot = g.at("rot");
        theta = rope_theta(prepared.at("config"));
        vocab_ = g.at("vocab");
        layers = g.at("layers");
        if (const char *n = std::getenv("NEURON_LAYERS"))
            layers = std::min(layers, std::atoi(n));
        vocab_shard = (vocab_ + tp - 1) / tp;
        for (const auto &l : prepared.at("layers")) {
            attn_kind.push_back(l.at("attn"));
            mlp_kind.push_back(l.at("mlp"));
        }
        eos_ = prepared.at("eos").get<std::vector<int>>();
        merge_generation_eos();
        max_context_ = prepared.at("max_context");
        map_embedding();
        if (const char *d = std::getenv("BEHEMOTH_NEURON_MTP_DIR"); d && *d && std::filesystem::exists(std::filesystem::path(d) / "mtp.json")) {
            mtp_root = d;
            if (const char *pm = std::getenv("BEHEMOTH_NEURON_SPEC_MTP_PMIN"))
                mtp_pmin = std::atof(pm);
            mtp_manifest = read_json(mtp_root / "mtp.json");
            if (mtp_manifest.at("geometry").at("tp") != tp || mtp_manifest.at("geometry").at("hidden") != hidden)
                throw std::runtime_error("Qwen35: the MTP set " + mtp_root.string() + " was prepared for another TP / hidden size");
        }
        // BEHEMOTH_NEURON_QWEN35_ASYNC=1: implicit async execution (launches return before the device finishes; the logits
        // read synchronises), hides launch gaps. Off by default: a GDN state spill / restore between two launches of one
        // step relies on the previous launch having finished.
        if (async_exec()) // the queue depth: a whole decode step (<= 81 launches) may be in flight
            setenv("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", std::getenv("BEHEMOTH_NEURON_QWEN35_ASYNC_DEPTH") ? std::getenv("BEHEMOTH_NEURON_QWEN35_ASYNC_DEPTH") : "128", 0);
        hold_cpu_latency();
        check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init");
        ranks.resize(size_t(tp));
        for (int r = 0; r < tp; ++r)
            workers.push_back(std::make_unique<RankWorker>());
        discover();
        load_graphs();
        load_weights();
        if (options.verbose) {
            size_t w = 0, s = 0;
            for (const auto &[name, d] : ranks[0].t)
                (name.find("state") != std::string::npos || name.ends_with(".kv") || name.ends_with(".kvs") ? s : w) += d.bytes;
            std::cerr << "Qwen35: " << layers << " layers, TP " << tp << ", " << graphs.size() << " NEFFs, " << pages_total - 1 << " pages, "
                      << slots_total - 1 << " GDN state slots; per core " << double(w) / (1 << 30) << " GiB weights + " << double(s) / (1 << 30)
                      << " GiB state\n";
        }
    }
    ~Qwen35Model() override {
        if (embed_map)
            munmap(embed_map, embed_map_bytes);
    }

    // Keep the CPUs out of deep idle states for the life of the process (PM QoS: an open fd on /dev/cpu_dma_latency holding
    // 0 us). Every launch is a rendezvous of all rank threads in a sub-millisecond all-reduce; a rank thread waking from a
    // deep C-state stalls all of them. Measured on trn1.32xlarge, TP=32, fused decode: 81-106 ms/token with jitter (45-129)
    // without it, 48.6 ms with it. Needs write access to the device (root, or chmod); BEHEMOTH_NEURON_CPU_DMA_LATENCY=-1 disables.
    int latency_fd = -1;
    void hold_cpu_latency() {
        const char *e = std::getenv("BEHEMOTH_NEURON_CPU_DMA_LATENCY");
        const int32_t us = e ? std::atoi(e) : 0;
        if (us < 0)
            return;
        latency_fd = ::open("/dev/cpu_dma_latency", O_WRONLY);
        if (latency_fd < 0 || ::write(latency_fd, &us, sizeof us) != sizeof us) {
            std::cerr << "Qwen35: cannot hold /dev/cpu_dma_latency at " << us << " us (" << std::strerror(errno)
                      << "): decode will be slower and jittery; run as root or make the device writable\n";
            if (latency_fd >= 0)
                ::close(latency_fd);
            latency_fd = -1;
        } else if (options.verbose)
            std::cerr << "Qwen35: holding /dev/cpu_dma_latency at " << us << " us\n";
    }

    void map_embedding() {
        const auto &e = prepared.at("files").at("embed.bin").at(0);
        const auto path = root / "embed.bin";
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("cannot open " + path.string());
        embed_map_bytes = std::filesystem::file_size(path);
        // MAP_POPULATE: every embedding page resident and mapped before serving. Without it the first use of a token's row
        // was a page fault in the step (cold page cache: a disk read; 8 fresh rows cost ~3 ms of a 45 ms step).
        embed_map = mmap(nullptr, embed_map_bytes, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
        ::close(fd);
        if (embed_map == MAP_FAILED)
            throw std::runtime_error("mmap embed.bin");
        if (e.at("dtype") != "bf16" || e.at("shape").at(0) != vocab_ || e.at("shape").at(1) != hidden)
            throw std::runtime_error("embed.bin is not bf16 [vocab][hidden]");
        const size_t offset = e.at("offset"), need = size_t(vocab_) * size_t(hidden) * 2;
        if (offset > embed_map_bytes || need > embed_map_bytes - offset)
            throw std::runtime_error("embed.bin is smaller than its layout entry");
        embed = reinterpret_cast<const uint16_t *>(static_cast<const char *>(embed_map) + size_t(e.at("offset")));
    }

    void discover() {
        const auto &geo = prepared.at("geometry");
        for (const auto &dir : std::filesystem::directory_iterator(options.neff_dir)) {
            const auto meta_path = dir.path() / "meta.json";
            if (!std::filesystem::exists(meta_path))
                continue;
            const json m = read_json(meta_path);
            const std::string kind = m.value("kind", "");
            if (kind.rfind("qwen35_", 0) != 0)
                continue;
            const auto &mg = m.at("geometry");
            for (const char *k : {"hidden", "vocab", "heads", "kv_heads", "head_dim", "rot", "linear_key_heads", "linear_value_heads", "linear_dim",
                                  "conv", "experts", "top_k", "expert_intermediate", "shared_intermediate", "tp"})
                if (mg.at(k) != geo.at(k))
                    throw std::runtime_error("NEFF " + dir.path().string() + " was traced for " + std::string(k) + " " + mg.at(k).dump() +
                                             ", the prepared model has " + geo.at(k).dump() + " (retrace with MODEL=... TP=... EP=...)");
            Graph gr;
            gr.kind = kind.substr(7);
            if (gr.kind.rfind("layer_", 0) == 0) {
                const size_t us = gr.kind.find('_', 6);
                gr.attn = gr.kind.substr(6, us - 6);
                gr.mlp = gr.kind.substr(us + 1);
            } else if (gr.kind.rfind("seg_", 0) == 0) {
                for (size_t at = 4; at < gr.kind.size();) {
                    const size_t us = std::min(gr.kind.find('_', at), gr.kind.size());
                    gr.seg.push_back(gr.kind.substr(at, us - at));
                    at = us + 1;
                }
            } else if (gr.kind == "gdn" || gr.kind == "gqa")
                gr.attn = gr.kind;
            else if (gr.kind == "mtp")
                gr.mtp = true;
            else if (gr.kind == "mtpc") {
                gr.mtp = true;
                gr.chain = m.value("D", 0);
                static const bool use_chain = !std::getenv("BEHEMOTH_NEURON_MTP_CHAIN") || std::atoi(std::getenv("BEHEMOTH_NEURON_MTP_CHAIN"));
                if (gr.chain < 1 || !use_chain)
                    continue;
            }
            else if (gr.kind != "head")
                gr.mlp = gr.kind;
            gr.moe_cap = gr.kind == "moe" ? m.value("moe_cap", 0) : 0;
            gr.moe_stat = m.value("moe_stat", 0);
            if (gr.moe_cap > 0) { // the grouped prefill MoE: its own kind, the bucket's plain "moe" NEFF stays the fallback
                static const bool off = std::getenv("BEHEMOTH_NEURON_QWEN35_MOE_GROUP") && !std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_MOE_GROUP"));
                if (off)
                    continue;
                gr.kind = "moeg";
            }
            if (gr.mtp && (mtp_manifest.is_null() || m.value("mtp_experts", 0) != mtp_manifest.at("geometry").at("experts").get<int>()))
                continue; // no MTP set loaded (or one with another expert count): the head's graphs stay unused
            gr.has_gqa = gr.attn == "gqa" || std::find(gr.seg.begin(), gr.seg.end(), "gqa") != gr.seg.end();
            gr.has_gdn = gr.attn == "gdn" || std::find(gr.seg.begin(), gr.seg.end(), "gdn") != gr.seg.end();
            gr.inputs = m.at("inputs").get<std::vector<std::string>>();
            gr.verify = std::find(gr.inputs.begin(), gr.inputs.end(), "vslot") != gr.inputs.end();
            gr.bound = gr.has_gdn && m.value("gdn_slot", std::string()) == "bound"; // (a gqa-only graph traced with GDN_SLOT=bound has no gdn state)
            gr.states = m.at("states").get<std::vector<std::string>>();
            if (gr.bound) {
                gr.verify = m.value("verify", false);
                gr.bound_states = gr.states;
                gr.states.clear();
                for (const auto &n : gr.bound_states) { // "kstate_2" -> "kstate"
                    const std::string base = n.substr(0, n.find('_'));
                    if (std::find(gr.states.begin(), gr.states.end(), base) == gr.states.end())
                        gr.states.push_back(base);
                }
                gr.extra = m.value("extra_outputs", std::vector<std::string>());
            }
            if (std::any_of(gr.inputs.begin(), gr.inputs.end(), [](const std::string &n) { return n == "egu" || n.ends_with(".egu"); }))
                gr.egu_t = m.value("egu_t", false) ? 1 : 0;
            gr.neff = dir.path() / "graph.neff";
            if (gr.kind == "head") {
                gr.T = m.at("T");
                gr.S = m.at("S");
            } else {
                gr.R = m.at("R");
                gr.Q = m.at("Q");
                gr.P = m.at("P");
                gr.T = gr.R * gr.Q;
                gr.pages = m.at("pages");
                gr.slots = m.at("slots");
                if (gr.attn == "gqa") {
                    if (pages_total && pages_total != gr.pages)
                        throw std::runtime_error("Qwen3.5 NEFFs disagree on the page count");
                    pages_total = gr.pages;
                }
                if (gr.attn == "gdn") {
                    if (slots_total && slots_total != gr.slots)
                        throw std::runtime_error("Qwen3.5 NEFFs disagree on the state slot count");
                    slots_total = gr.slots;
                }
                if (gr.Q > 1 && gr.R != 1 && !gr.verify) // R x Q with vslot: verify buckets of several requests
                    throw std::runtime_error("Qwen3.5 prefill buckets must have R = 1");
            }
            graphs.push_back(std::move(gr));
        }
        if (graphs.empty())
            throw std::runtime_error("no Qwen3.5 NEFFs (meta kind qwen35_*) in " + options.neff_dir.string());
        if (!pages_total || !slots_total)
            throw std::runtime_error("Qwen3.5 NEFFs need gqa and gdn buckets");
        // every (R, Q) group needs every sub-block kind the model uses, and a head for its token count
        std::set<std::pair<int, int>> groups;
        for (const auto &g : graphs)
            if (g.kind != "head" && !g.mtp)
                groups.insert({g.R, g.Q});
        for (auto [R, Q] : groups) {
            for (int l = 0; l < layers;) {
                if (const Graph *sg = segment_graph(l, R, Q, 1)) { // row buckets may consist of segment NEFFs only
                    l += int(sg->seg.size());
                    continue;
                }
                if (!layer_graph(l, R, Q, 1) && (!find(attn_kind[size_t(l)], R, Q, 1) || !find(mlp_kind[size_t(l)], R, Q, 1)))
                    throw std::runtime_error("Qwen35 bucket R=" + std::to_string(R) + " Q=" + std::to_string(Q) + " has neither a layer_" +
                                             attn_kind[size_t(l)] + "_" + mlp_kind[size_t(l)] + " NEFF nor both sub-block NEFFs for layer " +
                                             std::to_string(l));
                ++l;
            }
            if (!head_for(R * Q, 1))
                throw std::runtime_error("Qwen35 bucket R=" + std::to_string(R) + " Q=" + std::to_string(Q) + " has no head NEFF for " +
                                         std::to_string(R * Q) + " rows");
        }
    }
    // the cheapest graph of `kind` with exactly (R, Q) and at least P table pages (P ignored without an mla sub-block)
    const Graph *find(const std::string &kind, int R, int Q, int P) const {
        const Graph *best = nullptr;
        for (const auto &g : graphs)
            if (g.kind == kind && g.R == R && g.Q == Q && (!g.has_gqa || g.P >= P) && (!best || g.P < best->P))
                best = &g;
        return best;
    }
    // any graph containing attention sub-block `attn` (kda / mla) for (R, Q) (R = 0: any), at least P pages
    const Graph *find_attn(const std::string &attn, int R, int Q, int P) const {
        const Graph *best = nullptr;
        for (const auto &g : graphs)
            if (g.attn == attn && (!R || (g.R == R && g.Q == Q)) && (attn != "gqa" || g.P >= P) && (!best || g.P < best->P))
                best = &g;
        return best;
    }
    // the fused graph of layer l (attention + MLP in one NEFF) for (R, Q), if traced
    const Graph *layer_graph(int l, int R, int Q, int P) const {
        return find("layer_" + attn_kind[size_t(l)] + "_" + mlp_kind[size_t(l)], R, Q, P);
    }
    // a segment graph whose layers are exactly layers l.. of the model, for (R, Q), at least P pages
    const Graph *segment_graph(int l, int R, int Q, int P) const {
        static const bool off = std::getenv("BEHEMOTH_NEURON_QWEN35_NOSEG") && std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_NOSEG"));
        const Graph *best = nullptr;
        for (const auto &g : graphs) {
            if (off || g.seg.empty() || g.R != R || g.Q != Q || (g.has_gqa && g.P < P) || l + int(g.seg.size()) > layers)
                continue;
            bool same = true;
            for (size_t i = 0; i < g.seg.size(); ++i)
                same &= g.seg[i] == attn_kind[size_t(l) + i];
            if (same && (!best || g.seg.size() > best->seg.size() || (g.seg.size() == best->seg.size() && g.P < best->P)))
                best = &g;
        }
        return best;
    }
    // the smallest table width >= pages among the (R, Q) bucket's graphs with a gqa part (0: none)
    int bucket_pages(int R, int Q, int pages) const {
        int best = 0;
        for (const auto &g : graphs)
            if (g.has_gqa && g.R == R && g.Q == Q && g.P >= pages && (!best || g.P < best))
                best = g.P;
        return best;
    }
    const Graph *head_for(int T, int S) const {
        const Graph *best = nullptr;
        for (const auto &g : graphs)
            if (g.kind == "head" && g.T == T && g.S >= S && (!best || g.S < best->S))
                best = &g;
        return best;
    }

    void load_graphs() {
        for (auto &gr : graphs) {
            const auto neff = slurp(gr.neff);
            gr.model.resize(size_t(tp));
            std::vector<std::future<void>> jobs;
            for (int r = 0; r < tp; ++r)
                jobs.push_back(workers[size_t(r)]->submit([&, r] {
                    if (tp > 1 && gr.kind != "head")
                        check(nrt_load_collectives(neff.data(), neff.size(), r, -1, uint32_t(r), uint32_t(tp), &gr.model[size_t(r)]),
                              "nrt_load_collectives " + gr.neff.string());
                    else
                        check(nrt_load(neff.data(), neff.size(), r, -1, &gr.model[size_t(r)]), "nrt_load " + gr.neff.string());
                }));
            for (auto &j : jobs)
                j.get();
            nrt_tensor_info_array_t *info = nullptr;
            check(nrt_get_model_tensor_info(gr.model[0], &info), "tensor info");
            const size_t n = gr.inputs.size() + (gr.bound ? gr.bound_states.size() : gr.states.size());
            gr.input_bytes.assign(n, 0);
            for (uint64_t i = 0; i < info->tensor_count; ++i) {
                const auto &t = info->tensor_array[i];
                if (t.usage != NRT_TENSOR_USAGE_INPUT)
                    continue;
                const size_t index = std::stoul(std::string(t.name).substr(5));
                if (index >= n)
                    throw std::runtime_error(gr.neff.string() + ": input " + t.name + " beyond meta inputs + states");
                gr.input_bytes[index] = t.size;
            }
            if (gr.bound && ((gr.seg.empty() && gr.attn != "gdn") || (gr.verify && gr.R != 1)))
                throw std::runtime_error(gr.neff.string() + ": gdn_slot bound needs a gdn sub-block graph or a segment (verify: R = 1)");
            for (size_t s = 0; s < gr.states.size(); ++s) // state_bytes: the whole layer state (all slots), also for bound graphs
                gr.state_bytes.push_back(gr.input_bytes[gr.inputs.size() + s] * (gr.bound && gr.seg.empty() ? size_t(gr.slots) : 1)); // bound: row 0's kstate, cstate first
                // (segments: only the sub-block graphs size the layer states, a segment's state_bytes stay unused)
            if (gr.mtp) { // outputs: z [1, hidden] f32, logits [1, vocab shard] f32, then the aliased KV cache
                std::map<std::string, size_t> out;
                for (uint64_t i = 0; i < info->tensor_count; ++i)
                    if (info->tensor_array[i].usage == NRT_TENSOR_USAGE_OUTPUT)
                        out[info->tensor_array[i].name] = info->tensor_array[i].size;
                gr.out0 = out["output0"];
                if (gr.chain ? out["output0"] < size_t(gr.chain) * 4 || out["output1"] != gr.state_bytes.at(0)
                             : out["output0"] != size_t(hidden) * 4 || out["output1"] != size_t(vocab_shard) * 4 || out["output2"] != gr.state_bytes.at(0))
                    throw std::runtime_error(gr.neff.string() + ": unexpected MTP output layout");
            }
            nrt_free_model_tensor_info(info);
            if (options.verbose)
                std::cerr << "Qwen35 NEFF " << gr.kind << " R=" << gr.R << " Q=" << gr.Q << " P=" << gr.P << " T=" << gr.T << " S=" << gr.S << "\n";
        }
        // step buffers per (R, Q) group: residual ping-pong, positions, write slots, state slots, one table per P, head rows / logits
        for (const auto &gr : graphs) {
            if (gr.kind == "head")
                continue;
            if (gr.mtp) {
                auto &v = mtp_steps[gr.Q];
                if (v.empty()) {
                    v.resize(size_t(tp));
                    for (int r = 0; r < tp; ++r) {
                        auto &m = v[size_t(r)];
                        m.e = alloc(r, size_t(gr.Q) * size_t(hidden) * 4, "mtp_e");
                        m.pos = alloc(r, size_t(gr.Q) * 8, "mtp_pos");
                        m.cos_sin = alloc(r, size_t(gr.Q) * size_t(2 * rot) * 4, "mtp_cos_sin");
                        m.wslot = alloc(r, size_t(gr.Q) * 8, "mtp_wslot");
                        m.lo = alloc(r, 8, "mtp_lo");
                        m.sel = alloc(r, 8, "mtp_sel");
                    }
                }
                if (!v[0].table.count(gr.P))
                    for (int r = 0; r < tp; ++r)
                        v[size_t(r)].table[gr.P] = alloc(r, size_t(gr.P) * 8, "mtp_table");
                if (gr.chain && !mtp_chain.count(gr.chain)) {
                    auto &c = mtp_chain[gr.chain];
                    c.resize(size_t(tp));
                    const size_t n = size_t(std::max(gr.chain - 1, 1));
                    for (int r = 0; r < tp; ++r)
                        c[size_t(r)] = {alloc(r, n * 8, "mtp_cpos"), alloc(r, n * size_t(2 * rot) * 4, "mtp_ccos"), alloc(r, n * 8, "mtp_cwslot"),
                                        alloc(r, gr.out0, "mtp_drafts")};
                }
                if (mtp_z.empty()) {
                    mtp_z.resize(size_t(tp));
                    mtp_logits_dev.resize(size_t(tp));
                    for (int r = 0; r < tp; ++r) {
                        mtp_z[size_t(r)] = {alloc(r, size_t(hidden) * 4, "mtp_z0"), alloc(r, size_t(hidden) * 4, "mtp_z1")};
                        mtp_logits_dev[size_t(r)] = alloc(r, size_t(vocab_shard) * 4, "mtp_logits");
                    }
                }
                continue;
            }
            auto &v = steps[{gr.R, gr.Q}];
            if (v.empty()) {
                v.resize(size_t(tp));
                for (int r = 0; r < tp; ++r) {
                    auto &s = v[size_t(r)];
                    const size_t T = size_t(gr.T);
                    s.x[0] = alloc(r, T * size_t(streams * hidden) * 4, "x0");
                    s.x[1] = alloc(r, T * size_t(streams * hidden) * 4, "x1");
                    s.pos = alloc(r, T * 8, "pos");
                    s.wslot = alloc(r, T * 8, "wslot");
                    s.slot = alloc(r, size_t(gr.R) * 8, "slot");
                    s.vslot = alloc(r, T * 8, "vslot");
                    s.cos_sin = alloc(r, T * size_t(2 * rot) * 4, "cos_sin");
                    for (const auto &h : graphs)
                        if (h.kind == "head" && h.T == gr.T && !s.rows.count(h.S)) {
                            s.rows[h.S] = alloc(r, size_t(h.S) * 8, "rows");
                            s.logits[h.S] = alloc(r, size_t(h.S) * size_t(vocab_shard) * 4, "logits");
                        }
                }
            }
            if (gr.has_gqa && !v[0].table.count(gr.P))
                for (int r = 0; r < tp; ++r)
                    v[size_t(r)].table[gr.P] = alloc(r, size_t(gr.R) * size_t(gr.P) * 8, "table");
        }
    }

    void load_weights() {
        const auto t0 = Clock::now();
        const auto &files = prepared.at("files");
        // ROUTER=shard graphs (inputs m.rws / m.rcols): rank r holds the router columns of its hidden slice, derived here
        // from the prepared replicated router [E, H] f32. The full router is uploaded only when a loaded NEFF still binds
        // it (60 x 8.4 MB per core otherwise saved).
        bool need_rws = false, need_rw = false;
        // EGU_T graphs (MOE_KERNEL=dyn): the gate|up experts re-laid out here, egu [E, 64, 4096] -> egu_t[e * 128 + p, kt * 64 + o] =
        // egu[e, o, kt * 128 + p]. Every MoE-holding NEFF must agree (the expert buffers are shared, no second copy).
        int egu_t = -1;
        // The MTP head graphs (kind mtp / mtpc) are exempt: the head has its own weights ("MTP.*", 0.2 GiB per core), kept in the
        // layout its graphs were built with (the stored one), whatever the main set uses.
        for (const auto &g : graphs)
            if (g.egu_t >= 0 && !g.mtp) {
                if (egu_t >= 0 && egu_t != g.egu_t)
                    throw std::runtime_error("Qwen3.5 NEFFs disagree on the expert layout (meta egu_t): " + g.neff.string());
                egu_t = g.egu_t;
            }
        for (const auto &g : graphs)
            for (const auto &n : g.inputs) {
                need_rws |= n == "rws" || n.ends_with(".rws");
                need_rw |= n == "rw" || n.ends_with(".rw");
            }
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                auto upload = [&](const std::string &file, const std::string &prefix, const std::filesystem::path *dir = nullptr, const json *table = nullptr) {
                    const auto data = slurp((dir ? *dir : root) / file);
                    for (const auto &e : (table ? *table : files).at(file)) {
                        const std::string name = prefix + e.at("name").get<std::string>();
                        const size_t bytes = e.at("bytes"), offset = e.at("offset");
                        if (offset > data.size() || bytes > data.size() - offset)
                            throw std::runtime_error(file + ": " + name + " lies outside the file");
                        if (name.ends_with(".mlp.rw") && need_rws) {
                            const size_t cols = size_t(hidden / tp), E = bytes / (4 * size_t(hidden));
                            std::vector<float> slice(E * cols);
                            const char *src = data.data() + offset;
                            for (size_t x = 0; x < E; ++x)
                                std::memcpy(slice.data() + x * cols, src + (x * size_t(hidden) + size_t(r) * cols) * 4, cols * 4);
                            std::vector<int64_t> rc(cols);
                            for (size_t c = 0; c < cols; ++c)
                                rc[c] = int64_t(size_t(r) * cols + c);
                            const std::string base = name.substr(0, name.size() - 2);
                            Dev ds = alloc(r, slice.size() * 4, base + "rws"), dc = alloc(r, rc.size() * 8, base + "rcols");
                            put(ds, slice.data(), slice.size() * 4);
                            put(dc, rc.data(), rc.size() * 8);
                            ranks[size_t(r)].t[base + "rws"] = ds;
                            ranks[size_t(r)].t[base + "rcols"] = dc;
                            if (!need_rw)
                                continue;
                        }
                        Dev d = alloc(r, bytes, name);
                        if (egu_t == 1 && name.ends_with(".mlp.egu") && !name.starts_with("MTP.")) {
                            const size_t H = size_t(hidden), O = 64, KT = H / 128, E = bytes / (O * H);
                            if (E * O * H != bytes || H % 128)
                                throw std::runtime_error(name + ": unexpected expert tensor size for the egu_t layout");
                            std::vector<uint8_t> t(bytes);
                            const auto *src = reinterpret_cast<const uint8_t *>(data.data() + offset);
                            for (size_t e = 0; e < E; ++e)
                                for (size_t o = 0; o < O; ++o)
                                    for (size_t kt = 0; kt < KT; ++kt)
                                        for (size_t p = 0; p < 128; ++p)
                                            t[((e * 128 + p) * KT + kt) * O + o] = src[(e * O + o) * H + kt * 128 + p];
                            put(d, t.data(), bytes);
                        } else
                            put(d, data.data() + offset, bytes);
                        ranks[size_t(r)].t[name] = d;
                    }
                };
                upload("r" + std::to_string(r) + "/head.bin", "head.");
                for (int l = 0; l < layers; ++l)
                    upload("r" + std::to_string(r) + "/" + layer_file(l), layer_prefix(l));
                if (const Graph *mg = mtp_graph(0)) { // the MTP head: its layer, fc, norms ("MTP.*") and its own paged KV cache
                    upload("r" + std::to_string(r) + "/mtp.bin", "MTP.", &mtp_root, &mtp_manifest.at("files"));
                    Dev d = alloc(r, mg->state_bytes.at(0), "MTP.kv");
                    zero(d);
                    ranks[size_t(r)].t["MTP.kv"] = d;
                    if (!mtp_chain.empty()) { // chain graphs: this rank's slice of the embedding table, its vocabulary mask, its one-hot
                        const size_t vs = size_t(vocab_shard), first = size_t(r) * vs, have = first < size_t(vocab_) ? std::min(vs, size_t(vocab_) - first) : 0;
                        std::vector<uint16_t> emb(vs * size_t(hidden), 0);
                        std::memcpy(emb.data(), embed + first * size_t(hidden), have * size_t(hidden) * 2);
                        std::vector<float> vbias(vs, -3e38f), rhot(size_t(tp), 0.f);
                        std::fill_n(vbias.begin(), have, 0.f);
                        rhot[size_t(r)] = 1.f;
                        const float rbase = float(first);
                        auto constant = [&](const char *name, const void *src, size_t bytes) {
                            Dev c = alloc(r, bytes, name);
                            put(c, src, bytes);
                            ranks[size_t(r)].t[name] = c;
                        };
                        constant("MTP.emb", emb.data(), emb.size() * 2);
                        constant("MTP.vbias", vbias.data(), vbias.size() * 4);
                        constant("MTP.rhot", rhot.data(), rhot.size() * 4);
                        constant("MTP.rbase", &rbase, 4);
                    }
                }
                // state tensors, zeroed, sized by the NEFFs
                for (int l = 0; l < layers; ++l) {
                    const Graph *g = find_attn(attn_kind[size_t(l)], 0, 0, 1);
                    for (size_t s = 0; s < g->states.size(); ++s) {
                        const std::string name = layer_prefix(l) + "attn." + g->states[s];
                        Dev d = alloc(r, g->state_bytes[s], name);
                        zero(d);
                        ranks[size_t(r)].t[name] = d;
                    }
                }
            }));
        for (auto &j : jobs)
            j.get();
        if (options.verbose)
            std::cerr << "Qwen35: weights resident in " << ms_since(t0) / 1000 << " s\n";
        mtp_ready = mtp_graph(1) && mtp_max_drafts() > 0;
        if (options.verbose && !mtp_manifest.is_null()) {
            size_t b = 0;
            for (const auto &[name, d] : ranks[0].t)
                if (name.rfind("MTP.", 0) == 0)
                    b += d.bytes;
            std::cerr << "Qwen35 MTP: " << (mtp_ready ? "ready" : "NOT usable (needs mtp_1_1_P graphs and VERIFY segment buckets)") << ", up to "
                      << mtp_max_drafts() << " drafts per step, per core " << double(b) / (1 << 30) << " GiB (head weights + its KV cache)\n";
        }
    }
    // the MTP graph for Q entries (0: any) with the widest key window
    const Graph *mtp_graph(int Q) const {
        const Graph *best = nullptr;
        for (const auto &g : graphs)
            if (g.mtp && !g.chain && (!Q || g.Q == Q) && (!best || g.P > best->P))
                best = &g;
        return best;
    }
    // the chain graph drafting exactly D tokens after a first pass over Q entries
    const Graph *chain_graph(int Q, int D) const {
        const Graph *best = nullptr;
        for (const auto &g : graphs)
            if (g.chain == D && g.Q == Q && (!best || g.P > best->P))
                best = &g;
        return best;
    }
    // the narrowest verify bucket (Q, P) for `count` tokens of a request spanning `pages` pages ({0, 0}: none)
    std::pair<int, int> verify_bucket(int count, int pages) const {
        for (int q : verify_Q())
            if (q >= count)
                if (const int P = bucket_pages(1, q, pages))
                    return {q, P};
        return {0, 0};
    }
    // the cheapest verify bucket (R, Q, P) for `n` requests of at most `count` tokens each spanning `pages` pages ({0, 0, 0}: none)
    std::tuple<int, int, int> verify_bucket_rows(int n, int count, int pages) const {
        std::tuple<int, int, int> best{0, 0, 0};
        for (const auto &g : graphs)
            if (g.verify && g.has_gdn && g.R >= n && g.Q >= count && g.R * g.Q <= slots_total - 1)
                if (const int P = bucket_pages(g.R, g.Q, pages); P && (!std::get<2>(best) || g.R * g.Q < std::get<0>(best) * std::get<1>(best)))
                    best = {g.R, g.Q, P};
        return best;
    }
    // drafts per request a step of `n` decode requests at `pages` pages can verify in one launch
    int mtp_max_drafts(int n, int pages) const override {
        if (!mtp_ready)
            return 0;
        int q = 0;
        for (const auto &g : graphs)
            if (g.verify && g.has_gdn && (n == 1 ? g.R == 1 : g.R >= n) && g.R * g.Q <= slots_total - 1 && bucket_pages(g.R, g.Q, pages) && head_for(g.R * g.Q, 1))
                q = std::max(q, g.Q);
        return std::max(q - 1, 0);
    }
    static bool strict_verify() {
        static const bool on = std::getenv("BEHEMOTH_NEURON_QWEN35_STRICT_VERIFY") && std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_STRICT_VERIFY"));
        return on;
    }
    std::set<int> verify_Q() const {
        std::set<int> q;
        for (const auto &g : graphs)
            if (g.verify && g.has_gdn && g.R == 1)
                q.insert(g.Q);
        return q;
    }
    std::pair<int, int> first_group() const { return steps.begin()->first; }
    static double rope_theta(const json &cfg) {
        const json &c = cfg.contains("text_config") ? cfg.at("text_config") : cfg;
        if (c.contains("rope_parameters") && c.at("rope_parameters").contains("rope_theta"))
            return c.at("rope_parameters").at("rope_theta").get<double>();
        return c.value("rope_theta", 10000.0);
    }
    // cos | sin of position p for the `rot` rotary dims (HF rotate_half layout: emb = cat(f, f)), computed in double
    void rope_row(float *row, int position) const {
        const int half = rot / 2;
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * i / rot), a = double(position) * inv;
            row[i] = row[i + half] = float(std::cos(a));
            row[rot + i] = row[rot + i + half] = float(std::sin(a));
        }
    }
    // generation_config.json's eos_token_id (e.g. <|im_end|> for chat) joins the config's eos
    void merge_generation_eos() {
        std::ifstream f(root / "generation_config.json");
        const auto gc = json::parse(f, nullptr, false);
        if (gc.is_discarded() || !gc.contains("eos_token_id"))
            return;
        const auto &e = gc.at("eos_token_id");
        for (const auto &v : (e.is_array() ? e : json::array({e})))
            if (std::find(eos_.begin(), eos_.end(), v.get<int>()) == eos_.end())
                eos_.push_back(v.get<int>());
    }
    static std::string layer_file(int l) {
        char b[32];
        std::snprintf(b, sizeof b, "L%03d.bin", l);
        return b;
    }
    static std::string layer_prefix(int l) {
        char b[32];
        std::snprintf(b, sizeof b, "L%03d.", l);
        return b;
    }

    // Input / output tensor sets of graph g for layer l reading residual buffer `parity`.
    // Bound graphs: `slots` = the launch's state slots (decode / prefill: one per row; verify: vslot, one per position).
    std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *> layer_set(const Graph &g, int l, int parity, int r, const std::vector<int64_t> *slots = nullptr) {
        if (g.bound && !slots)
            throw std::runtime_error("Qwen35: bound gdn graph launched without slots");
        auto key = std::make_tuple(&g, l, parity, r);
        std::lock_guard lock(sets_mutex);
        if (g.bound) {
            if (auto it = bound_sets.find({&g, l, parity, r, *slots}); it != bound_sets.end())
                return it->second;
        } else if (auto it = sets.find(key); it != sets.end())
            return it->second;
        auto &st = steps.at({g.R, g.Q})[size_t(r)];
        auto &rank = ranks[size_t(r)];
        // weights / states: sub-block graphs by their own sub-block, fused layers by a.* (attention) / m.* (MLP); states
        // always belong to the attention sub-block
        const std::string sub = g.attn.empty() ? "mlp." : "attn.";
        auto bind = [&](const std::string &name) -> const Dev & {
            if (name == "x") return st.x[parity];
            if (name == "pos") return st.pos;
            if (name == "slot") return st.slot;
            if (name == "vslot") return st.vslot;
            if (name == "wslot") return st.wslot;
            if (name == "cos_sin") return st.cos_sin;
            if (name == "table") return st.table.at(g.P);
            std::string full = layer_prefix(l) + sub + name;
            if (!g.seg.empty()) { // "l<i>.a.<w>" / "l<i>.m.<w>" / "l<i>.<state>" of layer l + i
                const size_t dot = name.find('.');
                const int li = l + std::stoi(name.substr(1, dot - 1));
                const std::string rest = name.substr(dot + 1);
                full = layer_prefix(li) + (rest.rfind("a.", 0) == 0 ? "attn." + rest.substr(2) : rest.rfind("m.", 0) == 0 ? "mlp." + rest.substr(2) : "attn." + rest);
            } else
            if (name.rfind("a.", 0) == 0) full = layer_prefix(l) + "attn." + name.substr(2);
            if (name.rfind("m.", 0) == 0) full = layer_prefix(l) + "mlp." + name.substr(2);
            auto it = rank.t.find(full);
            if (it == rank.t.end())
                throw std::runtime_error("Qwen35: layer " + std::to_string(l) + " has no tensor " + full + " for the " + g.kind + " NEFF");
            return it->second;
        };
        // bound graphs: "kstate" / "kstate_<i>" / "kstate@<i>" -> the slice of the layer's kstate at slot (*slots)[i]
        // bound segments: "l<j>.kstate" / "l<j>.kstate_<i>" -> layer l + j; their gqa states ("l<j>.kv") bind whole
        auto bind_state = [&](const std::string &name) -> const Dev & {
            if (!g.bound)
                return bind(name);
            const size_t cut = name.find_first_of("_@");
            const std::string base = name.substr(0, cut);
            const size_t dot = base.find('.');
            const std::string leaf = dot == std::string::npos ? base : base.substr(dot + 1);
            if (leaf != "kstate" && leaf != "cstate")
                return bind(name);
            const int i = cut == std::string::npos ? 0 : std::stoi(name.substr(cut + 1));
            const Dev &whole = bind(base);
            const int slot = int(slots->at(size_t(i)));
            if (slot < 0 || slot >= slots_total)
                throw std::runtime_error("Qwen35: bound gdn graph: slot " + std::to_string(slot) + " out of range");
            const int li = dot == std::string::npos ? l : l + std::stoi(base.substr(1, dot - 1));
            const std::string full = layer_prefix(li) + "attn." + leaf;
            auto &sl = state_slices[{full, slot, r}];
            if (!sl.t) {
                sl.bytes = whole.bytes / size_t(slots_total);
                check(nrt_tensor_allocate_slice(whole.t, size_t(slot) * sl.bytes, sl.bytes, (full + ".slot").c_str(), &sl.t), "state slice");
            }
            return sl;
        };
        nrt_tensor_set_t *in = nullptr, *out = nullptr;
        check(nrt_allocate_tensor_set(&in), "set");
        check(nrt_allocate_tensor_set(&out), "set");
        const std::vector<std::string> &snames = g.bound ? g.bound_states : g.states;
        std::vector<std::string> names = g.inputs;
        names.insert(names.end(), snames.begin(), snames.end());
        for (size_t i = 0; i < names.size(); ++i) {
            const Dev &d = i < g.inputs.size() ? bind(names[i]) : bind_state(names[i]);
            if (d.bytes != g.input_bytes[i])
                throw std::runtime_error("Qwen35: " + g.kind + " input " + names[i] + " of layer " + std::to_string(l) + " expects " +
                                         std::to_string(g.input_bytes[i]) + " B, has " + std::to_string(d.bytes));
            check(nrt_add_tensor_to_tensor_set(in, ("input" + std::to_string(i)).c_str(), d.t), "add input");
        }
        check(nrt_add_tensor_to_tensor_set(out, "output0", st.x[1 - parity].t), "add output");
        if (g.moe_cap)
            check(nrt_add_tensor_to_tensor_set(out, "output1", moe_load_dev_locked(g, r).t), "add moe load output");
        for (size_t e = 0; e < g.extra.size(); ++e) // bound verify: per-position states, not aliased
            check(nrt_add_tensor_to_tensor_set(out, ("output" + std::to_string(e + 1)).c_str(), bind_state(g.extra[e]).t), "add extra output");
        for (size_t s = 0; s < snames.size(); ++s)
            check(nrt_add_tensor_to_tensor_set(out, ("output" + std::to_string(g.extra.size() + s + 1)).c_str(), bind_state(snames[s]).t), "add state output");
        if (g.bound)
            return bound_sets[{&g, l, parity, r, *slots}] = {in, out};
        return sets[key] = {in, out};
    }
    // the grouped MoE's load output (f32 [1]), one device tensor per graph and rank
    std::map<std::pair<const Graph *, int>, Dev> moe_load;
    const Dev &moe_load_dev_locked(const Graph &g, int r) {
        auto &d = moe_load[{&g, r}];
        if (!d.t)
            d = alloc(r, 4 * size_t(1 + g.moe_stat), "moe_load");
        return d;
    }
    const Dev &moe_load_dev(const Graph &g, int r) {
        std::lock_guard lock(sets_mutex);
        return moe_load_dev_locked(g, r);
    }
    int moe_grouped = 0, moe_fallbacks = 0; // per forward(): grouped MoE launches / reruns with the dense NEFF (rank 0's count)
    float moe_load_max = 0;
    std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *> head_set(const Graph &h, int R, int Q, int r, int parity) {
        auto key = std::make_tuple(&h, R * 1000 + Q, parity, r);
        std::lock_guard lock(sets_mutex);
        if (auto it = sets.find(key); it != sets.end())
            return it->second;
        auto &st = steps.at({R, Q})[size_t(r)];
        auto &rank = ranks[size_t(r)];
        nrt_tensor_set_t *in = nullptr, *out = nullptr;
        check(nrt_allocate_tensor_set(&in), "set");
        check(nrt_allocate_tensor_set(&out), "set");
        for (size_t i = 0; i < h.inputs.size(); ++i) {
            const auto &n = h.inputs[i];
            const Dev &d = n == "x" ? st.x[parity] : n == "rows" ? st.rows.at(h.S) : rank.t.at("head." + n);
            if (d.bytes != h.input_bytes[i])
                throw std::runtime_error("Qwen35: head input " + n + " size mismatch");
            check(nrt_add_tensor_to_tensor_set(in, ("input" + std::to_string(i)).c_str(), d.t), "add head input");
        }
        check(nrt_add_tensor_to_tensor_set(out, "output0", st.logits.at(h.S).t), "add head output");
        return sets[key] = {in, out};
    }

    // ---- GDN state slots ----------------------------------------------------------------------------------------
    size_t slot_state_bytes(int l, size_t s) const {
        const Graph *g = find_attn("gdn", 0, 0, 1);
        return g->state_bytes[s] / size_t(slots_total);
    }
    void copy_slot(int seq, int slot, bool save) {
        auto &blob = spilled[seq];
        if (save)
            blob.assign(size_t(tp), {});
        std::vector<std::future<void>> jobs;
        const Graph *g = find_attn("gdn", 0, 0, 1);
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                size_t at = 0;
                for (int l = 0; l < layers; ++l) {
                    if (attn_kind[size_t(l)] != "gdn")
                        continue;
                    for (size_t s = 0; s < g->states.size(); ++s) {
                        const Dev &d = ranks[size_t(r)].t.at(layer_prefix(l) + "attn." + g->states[s]);
                        const size_t n = slot_state_bytes(l, s);
                        if (save) {
                            blob[size_t(r)].resize(at + n);
                            check(nrt_tensor_read(d.t, blob[size_t(r)].data() + at, size_t(slot) * n, n), "read state");
                        } else
                            check(nrt_tensor_write(d.t, blob[size_t(r)].data() + at, size_t(slot) * n, n), "write state");
                        at += n;
                    }
                }
            }));
        for (auto &j : jobs)
            j.get();
        if (!save)
            spilled.erase(seq);
    }
    // The slot of sequence `seq` (its first page) whose chunk starts at `position`; `pinned`: slots of this launch.
    int acquire_slot(int seq, int position, std::set<int> &pinned) {
        auto touch = [&](int s) {
            lru.remove(s);
            lru.push_back(s);
        };
        if (auto it = slot_of.find(seq); it != slot_of.end()) {
            if (position == 0)
                spilled.erase(seq);
            touch(seq);
            pinned.insert(it->second);
            return it->second;
        }
        if (position > 0 && !spilled.count(seq))
            throw std::runtime_error("Qwen35: no GDN state for the sequence on page " + std::to_string(seq) + " at position " +
                                     std::to_string(position));
        // a free slot, else evict the least recently used unpinned one (its state goes to the host)
        const int slot0 = free_slot(pinned);
        slot_of[seq] = slot0;
        lru.push_back(seq);
        pinned.insert(slot0);
        if (position > 0) {
            copy_slot(seq, slot0, false);
            ++restores;
        } else
            spilled.erase(seq);
        return slot0;
    }
    // A slot no sequence owns (nor a pending verify launch), else the least recently used unpinned one, its state spilled.
    int free_slot(std::set<int> &pinned) {
        std::set<int> used = reserved;
        for (auto [s, sl] : slot_of)
            used.insert(sl);
        int slot = -1;
        for (int sl = 0; sl < slots_total - 1 && slot < 0; ++sl)
            if (!used.count(sl) && !pinned.count(sl))
                slot = sl;
        if (slot < 0) {
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                const int victim = *it, vs = slot_of.at(victim);
                if (pinned.count(vs) || reserved.count(vs))
                    continue;
                copy_slot(victim, vs, true);
                ++spills;
                slot_of.erase(victim);
                lru.erase(it);
                slot = vs;
                break;
            }
            if (slot < 0)
                throw std::runtime_error("Qwen35: more sequences in one launch than GDN state slots");
        }
        return slot;
    }

    // ---- prefix cache (docs/neuron-qwen35-prefix-cache.md) -----------------------------------------------------------
    // Host bytes of one sequence's recurrent state: [rank][gdn layer][kstate slot | cstate slot] (copy_slot's layout).
    size_t state_rank_bytes() const {
        const Graph *g = find_attn("gdn", 0, 0, 1);
        size_t n = 0, per = 0;
        if (!g)
            return 0;
        for (size_t s = 0; s < g->states.size(); ++s)
            per += g->state_bytes[s] / size_t(slots_total);
        for (int l = 0; l < layers; ++l)
            n += attn_kind[size_t(l)] == "gdn" ? per : 0;
        return n;
    }
    size_t state_bytes() const override { return state_rank_bytes() * size_t(tp); }
    // The state the sequence has now: read from its slot (it keeps the slot), or copied from its spilled blob.
    void state_save(int seq, std::byte *dst) override {
        const size_t rb = state_rank_bytes();
        if (auto sp = spilled.find(seq); sp != spilled.end() && !slot_of.count(seq)) {
            for (int r = 0; r < tp; ++r) {
                if (sp->second.at(size_t(r)).size() != rb)
                    throw std::runtime_error("Qwen35: spilled state size mismatch");
                std::memcpy(dst + size_t(r) * rb, sp->second[size_t(r)].data(), rb);
            }
            return;
        }
        auto it = slot_of.find(seq);
        if (it == slot_of.end())
            throw std::runtime_error("Qwen35: no GDN state to snapshot for the sequence on page " + std::to_string(seq));
        const int slot = it->second;
        std::vector<std::future<void>> jobs;
        const Graph *g = find_attn("gdn", 0, 0, 1);
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                size_t at = size_t(r) * rb;
                for (int l = 0; l < layers; ++l) {
                    if (attn_kind[size_t(l)] != "gdn")
                        continue;
                    for (size_t s = 0; s < g->states.size(); ++s) {
                        const Dev &d = ranks[size_t(r)].t.at(layer_prefix(l) + "attn." + g->states[s]);
                        const size_t n = slot_state_bytes(l, s);
                        check(nrt_tensor_read(d.t, dst + at, size_t(slot) * n, n), "read state");
                        at += n;
                    }
                }
            }));
        for (auto &j : jobs)
            j.get();
        ++state_saves;
    }
    // `src` becomes the state of the sequence that continues on first page `seq`: kept as its spilled blob, which
    // acquire_slot() writes into a slot at the sequence's first launch (position > 0). A slot / blob a finished sequence
    // left on that page is forgotten.
    void state_restore(int seq, const std::byte *src) override {
        state_drop(seq);
        const size_t rb = state_rank_bytes();
        auto &blob = spilled[seq];
        blob.assign(size_t(tp), {});
        for (int r = 0; r < tp; ++r)
            blob[size_t(r)].assign(reinterpret_cast<const char *>(src) + size_t(r) * rb, reinterpret_cast<const char *>(src) + size_t(r + 1) * rb);
        ++state_restores;
    }
    void state_drop(int seq) override {
        if (auto it = slot_of.find(seq); it != slot_of.end()) {
            reserved.erase(it->second);
            slot_of.erase(it);
            lru.remove(seq);
        }
        spilled.erase(seq);
    }
    // K/V pages of the gqa layers: host page = [rank][gqa layer][state tensor: this page's bytes] (kv, and kvs with KV8:
    // codes + scales, copied bit for bit; bf16 / f32 caches likewise). Every gqa state tensor is [pages, ...].
    std::vector<size_t> kv_strides() const {
        const Graph *g = find_attn("gqa", 0, 0, 1);
        std::vector<size_t> st;
        if (!g)
            return st;
        for (size_t s = 0; s < g->states.size(); ++s) {
            if (g->state_bytes[s] % size_t(pages_total))
                return {};
            st.push_back(g->state_bytes[s] / size_t(pages_total));
        }
        return st;
    }
    size_t kv_rank_bytes() const {
        size_t per = 0, n = 0;
        for (size_t b : kv_strides())
            per += b;
        for (int l = 0; l < layers; ++l)
            n += attn_kind[size_t(l)] == "gqa" ? per : 0;
        return n;
    }
    size_t kv_page_bytes() const override { return kv_rank_bytes() * size_t(tp); }
    void kv_copy(std::span<const int> device, std::span<const int> host, std::byte *arena, bool save) override {
        if (device.size() != host.size())
            throw std::invalid_argument("kv copy: page lists differ in length");
        for (int p : device)
            if (p < 0 || p >= pages_total - 1)
                throw std::invalid_argument("kv copy: page outside the pool");
        const Graph *g = find_attn("gqa", 0, 0, 1);
        const auto strides = kv_strides();
        const size_t rank_bytes = kv_rank_bytes(), page_bytes = rank_bytes * size_t(tp);
        if (!g || !rank_bytes)
            throw std::logic_error("Qwen35: no pageable K/V state");
        // runs of consecutive device AND host pages move in one transfer per layer and tensor, straight from / to the arena
        std::vector<std::pair<size_t, size_t>> runs;
        for (size_t i = 0; i < device.size();) {
            size_t j = i + 1;
            while (j < device.size() && device[j] == device[j - 1] + 1)
                ++j;
            runs.push_back({i, j});
            i = j;
        }
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                std::vector<std::byte> buf;
                size_t base = size_t(r) * rank_bytes;
                for (int l = 0; l < layers; ++l) {
                    if (attn_kind[size_t(l)] != "gqa")
                        continue;
                    for (size_t s = 0; s < strides.size(); ++s) {
                        const Dev &t = ranks[size_t(r)].t.at(layer_prefix(l) + "attn." + g->states[s]);
                        const size_t stride = strides[s];
                        for (auto [a, b] : runs) {
                            const size_t n = b - a;
                            buf.resize(n * stride);
                            if (save) {
                                check(nrt_tensor_read(t.t, buf.data(), size_t(device[a]) * stride, n * stride), "kv save");
                                for (size_t k = 0; k < n; ++k)
                                    std::memcpy(arena + size_t(host[a + k]) * page_bytes + base, buf.data() + k * stride, stride);
                            } else {
                                for (size_t k = 0; k < n; ++k)
                                    std::memcpy(buf.data() + k * stride, arena + size_t(host[a + k]) * page_bytes + base, stride);
                                check(nrt_tensor_write(t.t, buf.data(), size_t(device[a]) * stride, n * stride), "kv restore");
                            }
                        }
                        base += stride;
                    }
                }
            }));
        for (auto &j : jobs)
            j.get();
    }

    // ---- one launch: R rows x Q tokens through every layer, then the head for the sampled rows ------------------------
    struct Launch {
        int R = 0, Q = 0, P = 0;
        std::vector<float> x;
        std::vector<int64_t> pos, wslot, table, slot, vslot;
        bool verify = false;
        std::vector<float> cos_sin; // [T][2 rot]: cos | sin rows of each token's position
        std::vector<int> head_rows, destination;
    };
    Launch fresh(int R, int Q, int P) const {
        Launch L;
        L.R = R; L.Q = Q; L.P = P;
        const size_t T = size_t(R * Q);
        L.x.assign(T * size_t(streams * hidden), 0.f);
        L.pos.assign(T, 0);
        L.wslot.assign(T, int64_t(pages_total - 1) * PAGE);
        L.table.assign(size_t(R) * size_t(P), pages_total - 1);
        L.slot.assign(size_t(R), slots_total - 1);
        L.vslot.assign(T, slots_total - 1);
        L.cos_sin.assign(T * size_t(2 * rot), 0.f);
        for (size_t t = 0; t < T; ++t)
            rope_row(L.cos_sin.data() + t * size_t(2 * rot), 0);
        return L;
    }
    void fill_token(Launch &L, int t, int token, int position, int page) const {
        if (token < 0 || token >= vocab_)
            throw std::runtime_error("token outside the vocabulary");
        float *x = L.x.data() + size_t(t) * size_t(streams * hidden);
        const uint16_t *e = embed + size_t(token) * size_t(hidden);
        for (int h = 0; h < hidden; ++h)
            x[h] = bf16_to_f32(e[h]);
        for (int s = 1; s < streams; ++s)
            std::memcpy(x + size_t(s) * size_t(hidden), x, size_t(hidden) * 4);
        L.pos[size_t(t)] = position;
        L.wslot[size_t(t)] = int64_t(page) * PAGE + position % PAGE;
        rope_row(L.cos_sin.data() + size_t(t) * size_t(2 * rot), position);
    }
    double layers_ms = 0, head_ms = 0, exec_ms = 0, plan_ms = 0; // exec: submit -> all ranks joined; plan: execute() before the submit
    std::vector<std::array<double, 7>> prof; // per rank, per forward(): see execute()
    // BEHEMOTH_NEURON_QWEN35_LTRACE=N (docs/neuron-reap-launches.md): per (R, Q) bucket and rank, the host-side split of a
    // launch() (execute() call): input writes, the nrt_execute call time per NEFF class, the gaps between consecutive calls,
    // head execute, drain + logits read; one stderr line per bucket every N launches. Off (0) by default.
    struct LTrace {
        long steps = 0;
        double wall = 0;                                        // execute(): submit -> all ranks joined
        double join = 0;                                        // last rank's job done -> execute() sees all futures
        std::array<std::array<double, 12>, 2> r{};               // [rank<2][put, gap, gdn, gqa, moe, seg, layer, head, read, n_gdn+gqa+moe, n_seg+layer, tail]
    };
    std::map<std::pair<int, int>, LTrace> ltrace;
    std::array<double, 2> job_ms{};
    std::mutex ltrace_mutex;
    int launches = 0;
    // BEHEMOTH_NEURON_QWEN35_NANSCAN=1 (debug, fault study): when a launch's logits contain a non-finite value, read back
    // every rank's residual buffers and every layer's state tensors and report, per layer / state / slot, the non-finite
    // count and max |finite value| (gdn states f32, gqa kv bf16). Costs nothing until a fault: only the logits are checked.
    void nan_scan(const Launch &L, const float *dest, int parity) {
        size_t bad = 0;
        for (size_t i = 0; i < L.head_rows.size(); ++i)
            for (int v = 0; v < vocab_; ++v)
                bad += !std::isfinite(dest[size_t(L.destination[i]) * size_t(vocab_) + size_t(v)]);
        static int scans = 0;
        if (!bad || scans >= 2)
            return;
        ++scans;
        std::cerr << "NANSCAN launch R=" << L.R << " Q=" << L.Q << " pos0=" << L.pos[0] << " slot0=" << L.slot[0] << " nonfinite logits=" << bad
                  << " final parity=" << parity << "\n";
        for (size_t i = 0; i < L.head_rows.size(); ++i)
            for (int r = 0; r < tp; ++r) {
                size_t n = 0;
                const size_t first = size_t(r) * size_t(vocab_shard), end = std::min(size_t(vocab_), first + size_t(vocab_shard));
                for (size_t v = first; v < end; ++v)
                    n += !std::isfinite(dest[size_t(L.destination[i]) * size_t(vocab_) + v]);
                std::cerr << "NANSCAN logits row " << i << " rank " << r << " nonfinite " << n << "/" << (end - first) << "\n";
            }
        for (int r = 0; r < tp; ++r) {
            auto &st = steps.at({L.R, L.Q})[size_t(r)];
            for (int p = 0; p < 2; ++p) {
                std::vector<float> xr(st.x[p].bytes / 4);
                check(nrt_tensor_read(st.x[p].t, xr.data(), 0, xr.size() * 4), "nanscan x");
                size_t n = 0; float m = 0;
                for (float v : xr) { if (!std::isfinite(v)) ++n; else m = std::max(m, std::fabs(v)); }
                std::cerr << "NANSCAN rank " << r << " x[" << p << "] nonfinite " << n << "/" << xr.size() << " max " << m << "\n";
            }
            for (int l = 0; l < layers; ++l)
                for (const auto &[name, d] : ranks[size_t(r)].t) {
                    const std::string pre = layer_prefix(l) + "attn.";
                    if (name.rfind(pre, 0) != 0)
                        continue;
                    const std::string s = name.substr(pre.size());
                    if (s != "kstate" && s != "cstate" && s != "kv")
                        continue;
                    std::vector<char> buf(d.bytes);
                    check(nrt_tensor_read(d.t, buf.data(), 0, d.bytes), "nanscan state");
                    const bool f32 = s != "kv";
                    const size_t elems = f32 ? d.bytes / 4 : d.bytes / 2;
                    const int groups = f32 ? slots_total : 1;
                    const size_t per = elems / size_t(groups);
                    std::string line;
                    size_t total = 0;
                    for (int gidx = 0; gidx < groups; ++gidx) {
                        size_t n = 0, huge = 0, h0 = SIZE_MAX, h1 = 0, n0 = SIZE_MAX, n1 = 0; float m = 0;
                        for (size_t e = size_t(gidx) * per; e < size_t(gidx + 1) * per; ++e) {
                            float v;
                            if (f32) std::memcpy(&v, buf.data() + e * 4, 4);
                            else { uint16_t u; std::memcpy(&u, buf.data() + e * 2, 2); v = bf16_to_f32(u); }
                            const size_t i = e - size_t(gidx) * per;
                            if (!std::isfinite(v)) { ++n; n0 = std::min(n0, i); n1 = i; }
                            else { m = std::max(m, std::fabs(v)); if (std::fabs(v) > 1e4f) { ++huge; h0 = std::min(h0, i); h1 = i; } }
                        }
                        total += n;
                        line += " [" + std::to_string(gidx) + "] " + std::to_string(n) + "/" + std::to_string(m);
                        if (huge) // element index within the slot ([W, K-1] for cstate: channel = i / (K-1), tap = i % (K-1))
                            line += " huge " + std::to_string(huge) + " @" + std::to_string(h0) + ".." + std::to_string(h1);
                        if (n && n < per)
                            line += " nf@" + std::to_string(n0) + ".." + std::to_string(n1);
                    }
                    std::cerr << "NANSCAN rank " << r << " L" << l << " " << s << " nonfinite " << total << " (slot nonfinite/max:" << line << ")\n";
                }
        }
    }
    void execute(Launch &L, float *dest) {
        const auto tp0 = Clock::now();
        const Graph *head = nullptr;
        if (!L.head_rows.empty()) {
            head = head_for(L.R * L.Q, int(L.head_rows.size()));
            if (!head)
                throw std::runtime_error("Qwen35: no head NEFF for " + std::to_string(L.head_rows.size()) + " rows");
        }
        // the launch plan: a segment NEFF covering the next layers when traced for this bucket, else the fused layer NEFF,
        // else the attention and MLP sub-block NEFFs; each entry is (graph, first layer)
        std::vector<std::pair<const Graph *, int>> plan;
        for (int l = 0; l < layers;) {
            if (const Graph *sg = segment_graph(l, L.R, L.Q, L.P)) {
                plan.push_back({sg, l});
                l += int(sg->seg.size());
                continue;
            }
            if (const Graph *f = layer_graph(l, L.R, L.Q, L.P))
                plan.push_back({f, l});
            else {
                plan.push_back({find(attn_kind[size_t(l)], L.R, L.Q, L.P), l});
                const Graph *grouped = mlp_kind[size_t(l)] == "moe" ? find("moeg", L.R, L.Q, 0) : nullptr;
                plan.push_back({grouped ? grouped : find(mlp_kind[size_t(l)], L.R, L.Q, 0), l});
                if (!plan[plan.size() - 2].first || !plan.back().first)
                    throw std::runtime_error("Qwen35: bucket R=" + std::to_string(L.R) + " Q=" + std::to_string(L.Q) + " P=" + std::to_string(L.P) +
                                             " has no graph for layer " + std::to_string(l));
            }
            ++l;
        }
        // every table width the bucket's mla graphs use gets the request pages, padded with the scratch page
        std::map<int, std::vector<int64_t>> tables;
        for (const auto &entry : plan)
            {
                const Graph *g = entry.first;
                if (g->has_gqa && !tables.count(g->P)) {
                    auto &t = tables[g->P];
                    t.assign(size_t(L.R) * size_t(g->P), pages_total - 1);
                    for (int r = 0; r < L.R; ++r)
                        std::copy_n(L.table.begin() + std::ptrdiff_t(r * L.P), std::min(L.P, g->P), t.begin() + std::ptrdiff_t(r * g->P));
                }
            }
        std::vector<int64_t> rows(head ? size_t(head->S) : 0, 0);
        for (size_t i = 0; i < L.head_rows.size(); ++i)
            rows[i] = L.head_rows[i];
        std::vector<std::future<void>> jobs;
        double t_layers = 0, t_head = 0;
        static const bool use_barrier = std::getenv("BEHEMOTH_NEURON_QWEN35_BARRIER") && std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_BARRIER"));
        SpinBarrier barrier(tp);
        const std::unique_ptr<std::atomic<int>[]> moe_votes(new std::atomic<int>[size_t(layers * tp)]()); // per layer and rank: 0 none, 1 fits, 2 overflow
        plan_ms += ms_since(tp0);
        const auto te0 = Clock::now();
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
              try {
                auto &st = steps.at({L.R, L.Q})[size_t(r)];
                const auto t0 = Clock::now();
                put(st.x[0], L.x.data(), L.x.size() * 4);
                put(st.pos, L.pos.data(), L.pos.size() * 8);
                put(st.wslot, L.wslot.data(), L.wslot.size() * 8);
                put(st.slot, L.slot.data(), L.slot.size() * 8);
                if (L.verify)
                    put(st.vslot, L.vslot.data(), L.vslot.size() * 8);
                put(st.cos_sin, L.cos_sin.data(), L.cos_sin.size() * 4);
                for (const auto &[width, t] : tables)
                    put(st.table.at(width), t.data(), t.size() * 8);
                int parity = 0;
                double *pr = prof[size_t(r)].data(); // set lookup, gdn / gqa / moe execute, input writes, head execute, logits read
                pr[4] += ms_since(t0);
                static const int sync_bytes = std::getenv("BEHEMOTH_NEURON_QWEN35_SYNC") ? std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_SYNC")) : 0;
                static const int dump_x = std::getenv("NEURON_X_DUMP") ? std::atoi(std::getenv("NEURON_X_DUMP")) : 0; // 1: report non-finite layers, 2: every layer
                std::vector<float> xr;
                const int ltrace_n = ltrace_every();
                std::array<double, 12> lt{};
                auto lt_prev = Clock::now();
                lt[0] = ms_since(t0);
                lt[11] = std::chrono::duration<double, std::milli>(t0 - te0).count(); // dispatch: execute() submit -> this rank's job starts
                for (const auto &[g, l] : plan) {
                    {
                        const auto ts = Clock::now();
                        auto [in, o] = layer_set(*g, l, parity, r, g->bound ? (g->verify ? &L.vslot : &L.slot) : nullptr);
                        const auto te = Clock::now();
                        check(nrt_execute(g->model[size_t(r)], in, o), g->kind + " execute");
                        if (g->moe_cap) {
                            // grouped MoE: exact iff no grouped expert got more tokens than it has slots. Every rank computes the
                            // same routing from the same residual bits; the ranks still exchange their verdicts and rerun when
                            // ANY of them saw an overflow, so they can never disagree on a launch whose collectives need them all.
                            drain(r);
                            float load = 0;
                            check(nrt_tensor_read(moe_load_dev(*g, r).t, &load, 0, 4), "read moe load");
                            moe_votes[size_t(l * tp + r)].store(load <= float(g->moe_cap) ? 1 : 2);
                            bool over = false;
                            for (int o = 0; o < tp; ++o) {
                                int v;
                                while (!(v = moe_votes[size_t(l * tp + o)].load())) {
                                    if (barrier.failed.load())
                                        throw std::runtime_error("Qwen35: another rank failed");
#if defined(__x86_64__)
                                    __builtin_ia32_pause();
#endif
                                }
                                over |= v == 2;
                            }
                            if (over) {
                                const Graph *dense = find(mlp_kind[size_t(l)], g->R, g->Q, 0);
                                auto [in2, o2] = layer_set(*dense, l, parity, r);
                                check(nrt_execute(dense->model[size_t(r)], in2, o2), "moe fallback execute");
                            }
                            if (r == 0) {
                                ++moe_grouped;
                                moe_fallbacks += over;
                                moe_load_max = std::max(moe_load_max, load);
                                static const bool log_load = std::getenv("BEHEMOTH_NEURON_QWEN35_MOE_LOAD_LOG") != nullptr;
                                if (log_load) {
                                    std::string top;
                                    if (g->moe_stat) { // the chunk's largest expert loads, descending
                                        std::vector<float> st(size_t(g->moe_stat));
                                        check(nrt_tensor_read(moe_load_dev(*g, r).t, st.data(), 4, st.size() * 4), "read moe stats");
                                        top = " top";
                                        for (float v : st)
                                            top += " " + std::to_string(int(v));
                                    }
                                    std::cerr << "moe_load layer " << l << " pos " << L.pos[0] << " load " << load << (over ? " FALLBACK" : "") << top << "\n";
                                }
                            }
                        }
                        const auto tx = Clock::now();
                        pr[0] += std::chrono::duration<double, std::milli>(te - ts).count();
                        pr[g->attn == "gdn" ? 1 : g->attn == "gqa" ? 2 : 3] += std::chrono::duration<double, std::milli>(tx - te).count();
                        if (ltrace_n) {
                            const int c = !g->seg.empty() ? 5 : !g->attn.empty() && !g->mlp.empty() ? 6 : g->attn == "gdn" ? 2 : g->attn == "gqa" ? 3 : 4;
                            lt[1] += std::chrono::duration<double, std::milli>(te - lt_prev).count();
                            lt[size_t(c)] += std::chrono::duration<double, std::milli>(tx - te).count();
                            lt[c >= 5 ? 10 : 9] += 1;
                            lt_prev = tx;
                        }
                        parity ^= 1;
                        if (use_barrier)
                            barrier.wait();
                        if (sync_bytes || dump_x)
                            drain(r);
                        if (sync_bytes) { // BEHEMOTH_NEURON_QWEN35_SYNC=n: blocking read of n bytes of the new residual on EVERY rank
                            char probe[64]; // after every launch (the section-12 A/B: host-device synchronisation without the dump cost)
                            check(nrt_tensor_read(st.x[parity].t, probe, 0, size_t(std::min(sync_bytes, 64))), "sync read");
                        }
                        if (dump_x && r == 0) { // residual after this launch: max |x| per row, first non-finite layer
                            xr.resize(L.x.size());
                            check(nrt_tensor_read(st.x[parity].t, xr.data(), 0, xr.size() * 4), "read x");
                            std::string rows;
                            bool bad = false;
                            for (int t = 0; t < L.R * L.Q; ++t) {
                                float m = 0; bool fin = true;
                                for (int h = 0; h < hidden; ++h) { const float v = xr[size_t(t) * size_t(hidden) + size_t(h)]; if (!std::isfinite(v)) fin = false; else m = std::max(m, std::fabs(v)); }
                                bad |= !fin;
                                rows += (fin ? " " : " NONFINITE ") + std::to_string(m);
                            }
                            if (bad || dump_x > 1)
                                std::cerr << "x after layer " << l << " (" << g->kind << ", pos " << L.pos[0] << "):" << rows << "\n";
                        }
                    }
                }
                // NEURON_H_DUMP=file: the final residual (BEFORE the final norm) of every real token of the launch, appended as
                // records {int64 slot, int64 position, f32 [hidden]} (rank 0's copy; all ranks hold the same bits). Debug only:
                // the offline MTP acceptance study (tests/qwen35_mtp_study.py) reads it.
                static const char *h_dump = std::getenv("NEURON_H_DUMP");
                if (h_dump && r == 0) {
                    drain(r);
                    xr.resize(L.x.size());
                    check(nrt_tensor_read(st.x[parity].t, xr.data(), 0, xr.size() * 4), "read x");
                    std::ofstream f(h_dump, std::ios::binary | std::ios::app);
                    for (int t = 0; t < L.R * L.Q; ++t) {
                        const int64_t rec[2] = {L.slot[size_t(t / L.Q)], L.pos[size_t(t)]};
                        if (rec[0] == slots_total - 1 || rec[1] < 0)
                            continue;
                        f.write(reinterpret_cast<const char *>(rec), sizeof rec);
                        f.write(reinterpret_cast<const char *>(xr.data() + size_t(t) * size_t(hidden)), std::streamsize(size_t(hidden) * 4));
                    }
                }
                const auto t1 = Clock::now();
                if (head) {
                    put(st.rows.at(head->S), rows.data(), rows.size() * 8);
                    auto [in, o] = head_set(*head, L.R, L.Q, r, parity);
                    const auto th = Clock::now();
                    check(nrt_execute(head->model[size_t(r)], in, o), "head execute");
                    drain(r);
                    const auto tr = Clock::now();
                    // each rank reads its vocabulary shard of every sampled row straight into the row's place in the
                    // logits buffer (32 threads in parallel; no per-step shard vectors, no serial 1 MB-per-row copy)
                    const size_t first = size_t(r) * size_t(vocab_shard);
                    if (first < size_t(vocab_)) {
                        const size_t n = std::min(size_t(vocab_) - first, size_t(vocab_shard));
                        for (size_t i = 0; i < L.head_rows.size(); ++i)
                            check(nrt_tensor_read(st.logits.at(head->S).t, dest + size_t(L.destination[i]) * size_t(vocab_) + first,
                                                  i * size_t(vocab_shard) * 4, n * 4), "read logits");
                    }
                    pr[5] += std::chrono::duration<double, std::milli>(tr - th).count();
                    pr[6] += ms_since(tr);
                    if (ltrace_n) {
                        lt[1] += std::chrono::duration<double, std::milli>(th - lt_prev).count();
                        lt[7] += std::chrono::duration<double, std::milli>(tr - th).count();
                        lt[8] += ms_since(tr);
                    }
                } else {
                    const auto td = Clock::now();
                    drain(r);
                    if (ltrace_n)
                        lt[8] += ms_since(td);
                }
                if (ltrace_n && r < 2) {
                    job_ms[size_t(r)] = ms_since(te0); // submit -> this rank's job done
                    std::lock_guard lock(ltrace_mutex);
                    auto &acc = ltrace[{L.R, L.Q}].r[size_t(r)];
                    for (size_t i = 0; i < lt.size(); ++i)
                        acc[i] += lt[i];
                }
                if (r == 0) {
                    t_layers = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    t_head = ms_since(t1);
                }
              } catch (...) {
                barrier.failed.store(true);
                throw;
              }
            }));
        for (auto &j : jobs)
            j.get();
        if (const int ltrace_n = ltrace_every()) {
            std::lock_guard lock(ltrace_mutex);
            auto &acc = ltrace[{L.R, L.Q}];
            acc.wall += ms_since(te0);
            acc.join += ms_since(te0) - std::max(job_ms[0], tp > 1 ? job_ms[1] : 0.0);
            if (++acc.steps == ltrace_n) {
                const double n = double(acc.steps);
                char b[1024];
                for (int r = 0; r < std::min(tp, 2); ++r) {
                    const auto &a = acc.r[size_t(r)];
                    std::snprintf(b, sizeof b, "LTRACE R=%d Q=%d rank=%d steps=%ld wall=%.3f dispatch=%.3f join=%.3f put=%.3f gap=%.3f gdn=%.3f gqa=%.3f moe=%.3f seg=%.3f layer=%.3f head=%.3f read=%.3f launches=%.1f+%.1f (ms per launch() call)\n",
                                  L.R, L.Q, r, acc.steps, acc.wall / n, a[11] / n, acc.join / n, a[0] / n, a[1] / n, a[2] / n, a[3] / n, a[4] / n, a[5] / n, a[6] / n, a[7] / n, a[8] / n, a[9] / n, a[10] / n);
                    std::cerr << b;
                }
                acc = {};
            }
        }
        static const bool nanscan = std::getenv("BEHEMOTH_NEURON_QWEN35_NANSCAN") && std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_NANSCAN"));
        if (nanscan && head)
            nan_scan(L, dest, int(plan.size() % 2));
        // BEHEMOTH_NEURON_QWEN35_XCHECK=1 (debug): after every launch compare the final residual of all ranks bitwise (every
        // rank should hold the same all-reduced bits); report the first launches whose residuals differ.
        static const bool xcheck = std::getenv("BEHEMOTH_NEURON_QWEN35_XCHECK") && std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_XCHECK"));
        if (xcheck && tp > 1) {
            static int reports = 0;
            static long checked = 0;
            const int p = int(plan.size() % 2);
            std::vector<std::vector<float>> xs(size_t(tp), std::vector<float>(L.x.size()));
            for (int r = 0; r < tp; ++r)
                check(nrt_tensor_read(steps.at({L.R, L.Q})[size_t(r)].x[p].t, xs[size_t(r)].data(), 0, L.x.size() * 4), "xcheck read");
            ++checked;
            size_t n = 0; double m = 0;
            for (size_t i = 0; i < L.x.size(); ++i)
                if (std::memcmp(&xs[0][i], &xs[1][i], 4)) { ++n; m = std::max(m, double(std::fabs(xs[0][i] - xs[1][i]))); }
            if (n && reports < 20) {
                ++reports;
                std::cerr << "XCHECK launch " << checked << " R=" << L.R << " Q=" << L.Q << " pos0=" << L.pos[0] << ": " << n << "/" << L.x.size()
                          << " residual elements differ between rank 0 and 1, max |diff| " << m << "\n";
            }
        }
        final_parity = int(plan.size() % 2);
        exec_ms += ms_since(te0);
        layers_ms += t_layers;
        head_ms += t_head;
        ++launches;
    }

    // ---- MTP head ---------------------------------------------------------------------------------------------------
    std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *> mtp_set(const Graph &g, bool first_pass, int parity, int r, int R = 1, int row = 0) {
        auto key = std::make_tuple(&g, first_pass ? (R > 1 ? -100 - R * 64 - row : -1) : -2, parity, r);
        std::lock_guard lock(sets_mutex);
        if (auto it = sets.find(key); it != sets.end())
            return it->second;
        auto &ms = mtp_steps.at(g.Q)[size_t(r)];
        auto &rank = ranks[size_t(r)];
        auto bind = [&](const std::string &name) -> const Dev & {
            // x: the residual BEFORE its norm hnorm: the main model's final residual of the last step (final norm), or the
            // previous pass's own output (mtp.norm)
            if (name == "x" && first_pass && R > 1) { // the request's Q rows of the R x Q launch's residual: a slice of that buffer
                auto &sl = x_slices[{R, g.Q, parity, row, r}];
                if (!sl.t) {
                    const size_t bytes = size_t(g.Q) * size_t(hidden) * 4;
                    check(nrt_tensor_allocate_slice(steps.at({R, g.Q})[size_t(r)].x[parity].t, size_t(row) * bytes, bytes, "mtp_x_slice", &sl.t), "slice");
                    sl.bytes = bytes;
                }
                return sl;
            }
            if (name == "x") return first_pass ? steps.at({1, g.Q})[size_t(r)].x[parity] : mtp_z[size_t(r)][size_t(parity)];
            if (name == "e") return ms.e;
            if (name == "pos") return ms.pos;
            if (name == "cos_sin") return ms.cos_sin;
            if (name == "wslot") return ms.wslot;
            if (name == "table") return ms.table.at(g.P);
            if (name == "lo") return ms.lo;
            if (name == "sel") return ms.sel;
            if (name == "cpos") return mtp_chain.at(g.chain)[size_t(r)].cpos;
            if (name == "ccos") return mtp_chain.at(g.chain)[size_t(r)].ccos;
            if (name == "cwslot") return mtp_chain.at(g.chain)[size_t(r)].cwslot;
            std::string full = "MTP." + name;
            if (name == "hnorm") full = first_pass ? "head.norm" : "MTP.norm";
            else if (name == "mnorm") full = "MTP.norm";
            else if (name == "w" || name == "w_s") full = "head." + name;
            else if (name.rfind("a.", 0) == 0) full = "MTP.attn." + name.substr(2);
            else if (name.rfind("m.", 0) == 0) full = "MTP.mlp." + name.substr(2);
            auto it = rank.t.find(full);
            if (it == rank.t.end())
                throw std::runtime_error("Qwen35: no tensor " + full + " for the mtp NEFF");
            return it->second;
        };
        nrt_tensor_set_t *in = nullptr, *out = nullptr;
        check(nrt_allocate_tensor_set(&in), "set");
        check(nrt_allocate_tensor_set(&out), "set");
        std::vector<std::string> names = g.inputs;
        names.insert(names.end(), g.states.begin(), g.states.end());
        for (size_t i = 0; i < names.size(); ++i) {
            const Dev &d = bind(names[i]);
            if (d.bytes != g.input_bytes[i])
                throw std::runtime_error("Qwen35: mtp input " + names[i] + " expects " + std::to_string(g.input_bytes[i]) + " B, has " + std::to_string(d.bytes));
            check(nrt_add_tensor_to_tensor_set(in, ("input" + std::to_string(i)).c_str(), d.t), "add input");
        }
        if (g.chain) {
            check(nrt_add_tensor_to_tensor_set(out, "output0", mtp_chain.at(g.chain)[size_t(r)].drafts.t), "add output");
            check(nrt_add_tensor_to_tensor_set(out, "output1", rank.t.at("MTP.kv").t), "add state output");
            return sets[key] = {in, out};
        }
        check(nrt_add_tensor_to_tensor_set(out, "output0", mtp_z[size_t(r)][first_pass ? 0 : size_t(1 - parity)].t), "add output");
        check(nrt_add_tensor_to_tensor_set(out, "output1", mtp_logits_dev[size_t(r)].t), "add output");
        check(nrt_add_tensor_to_tensor_set(out, "output2", rank.t.at("MTP.kv").t), "add state output");
        return sets[key] = {in, out};
    }
    // One launch of the MTP head: entries (position[i], next token[i]) for i < n of a bucket of g.Q rows, logits of row n - 1.
    int mtp_launch(const Graph &g, bool first_pass, int parity, std::span<const int> pages, int position, std::span<const int> next, int first_page, int lo,
                   int xR = 1, int xrow = 0) {
        const size_t Q = size_t(g.Q), n = next.size();
        std::vector<float> e(Q * size_t(hidden), 0.f), cs(Q * size_t(2 * rot), 0.f);
        std::vector<int64_t> pos(Q, -1), wslot(Q, int64_t(pages_total - 1) * PAGE), table(size_t(g.P), pages_total - 1);
        for (size_t i = 0; i < Q; ++i) {
            const int p = position + int(i);
            rope_row(cs.data() + i * size_t(2 * rot), i < n ? p : 0);
            if (i >= n)
                continue;
            if (next[i] < 0 || next[i] >= vocab_ || size_t(p / PAGE) >= pages.size())
                throw std::runtime_error("Qwen35: MTP entry outside the vocabulary / the request's pages");
            const uint16_t *row = embed + size_t(next[i]) * size_t(hidden);
            for (int h = 0; h < hidden; ++h)
                e[i * size_t(hidden) + size_t(h)] = bf16_to_f32(row[h]);
            pos[i] = p - first_page * PAGE; // key index inside the window (the table's first key is position first_page * 16)
            wslot[i] = int64_t(pages[size_t(p / PAGE)]) * PAGE + p % PAGE;
        }
        for (int j = 0; j < g.P; ++j)
            if (size_t(first_page + j) < pages.size())
                table[size_t(j)] = pages[size_t(first_page + j)];
        // chain graph: the entries of the chained passes follow the last first-pass entry (positions, RoPE rows, write slots)
        const size_t C = g.chain > 1 ? size_t(g.chain - 1) : 0;
        std::vector<int64_t> cpos(C), cwslot(C);
        std::vector<float> ccos(C * size_t(2 * rot));
        for (size_t j = 0; j < C; ++j) {
            const int p = position + int(n) + int(j);
            if (size_t(p / PAGE) >= pages.size())
                throw std::runtime_error("Qwen35: MTP chain entry outside the request's pages");
            cpos[j] = p - first_page * PAGE;
            cwslot[j] = int64_t(pages[size_t(p / PAGE)]) * PAGE + p % PAGE;
            rope_row(ccos.data() + j * size_t(2 * rot), p);
        }
        float chain_out[64] = {};
        const int64_t lo_rel = std::max(0, lo - first_page * PAGE), sel = int64_t(n) - 1;
        std::vector<int64_t> stat = table;
        stat.push_back(lo_rel);
        stat.push_back(sel);
        stat.push_back(g.P);
        const bool fresh_static = mtp_static[g.Q] != stat;
        if (fresh_static)
            mtp_static[g.Q] = stat;
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                auto &ms = mtp_steps.at(g.Q)[size_t(r)];
                put(ms.e, e.data(), e.size() * 4);
                put(ms.pos, pos.data(), pos.size() * 8);
                put(ms.cos_sin, cs.data(), cs.size() * 4);
                put(ms.wslot, wslot.data(), wslot.size() * 8);
                if (fresh_static) {
                    put(ms.table.at(g.P), table.data(), table.size() * 8);
                    put(ms.lo, &lo_rel, 8);
                    put(ms.sel, &sel, 8);
                }
                if (C) {
                    auto &mc = mtp_chain.at(g.chain)[size_t(r)];
                    put(mc.cpos, cpos.data(), C * 8);
                    put(mc.ccos, ccos.data(), ccos.size() * 4);
                    put(mc.cwslot, cwslot.data(), C * 8);
                }
                auto [in, o] = mtp_set(g, first_pass, parity, r, xR, xrow);
                check(nrt_execute(g.model[size_t(r)], in, o), "mtp execute");
                drain(r);
                if (g.chain) { // every rank holds the same drafts (all-reduced): rank 0's copy
                    if (r == 0)
                        check(nrt_tensor_read(mtp_chain.at(g.chain)[0].drafts.t, chain_out, 0, std::min(g.out0, size_t(256))), "read drafts");
                    return;
                }
                const size_t first = size_t(r) * size_t(vocab_shard);
                if (first < size_t(vocab_))
                    check(nrt_tensor_read(mtp_logits_dev[size_t(r)].t, mtp_logits.data() + first, 0, std::min(size_t(vocab_) - first, size_t(vocab_shard)) * 4),
                          "read mtp logits");
            }));
        for (auto &j : jobs)
            j.get();
        ++mtp_launches;
        if (g.chain) { // an id outside the vocabulary (an exact tie between two ranks) ends the chain
            mtp_chain_out.clear();
            std::copy_n(chain_out, 64, mtp_chain_raw.begin());
            for (int j = 0; j < std::min(g.chain, 16); ++j) {
                const float v = chain_out[j];
                if (!(v >= 0.f) || v >= float(vocab_))
                    break;
                mtp_chain_out.push_back(int(v + 0.5f));
            }
            return mtp_chain_out.empty() ? -1 : mtp_chain_out[0];
        }
        int best = 0;
        for (int v = 1; v < vocab_; ++v)
            if (mtp_logits[size_t(v)] > mtp_logits[size_t(best)])
                best = v;
        const float top = mtp_logits[size_t(best)];
        if (!std::isfinite(top))
            throw std::runtime_error("Qwen35: non-finite MTP logits");
        if (mtp_pmin > 0) { // the draft's probability under the head (tokens more than 16 below the top do not count)
            double sum = 0;
            for (int v = 0; v < vocab_; ++v)
                if (mtp_logits[size_t(v)] > top - 16.f)
                    sum += std::exp(double(mtp_logits[size_t(v)] - top));
            mtp_last_p = 1.0 / sum;
        }
        return best;
    }

    // ---- Backend --------------------------------------------------------------------------------------------------
    int mtp_max_drafts() const override {
        const auto q = verify_Q();
        return mtp_graph(1) && !q.empty() ? *q.rbegin() - 1 : 0;
    }
    // After the sampling of a step that was ONE decode / verify request at `position` (its scheduled token): `next` = the
    // tokens the step emitted (the sampled token, then one per accepted draft). Commits the gdn state of the last accepted
    // position, writes the MTP cache entries of positions position .. position + accepted (pairs: hidden of the position,
    // embedding of the next token) and drafts `want` tokens after next.back() by chaining the head on its own output.
    std::vector<int> mtp_draft(std::span<const int> pages, int position, std::span<const int> next, int want) override {
        const auto t0 = Clock::now();
        std::vector<int> drafts;
        const int accepted = int(next.size()) - 1;
        auto it = pages.empty() ? lasts.end() : std::find_if(lasts.begin(), lasts.end(), [&](const Last &l) { return l.seq == pages[0] && l.position == position; });
        if (it == lasts.end() || accepted < 0 || accepted >= it->count)
            throw std::logic_error("Qwen35: mtp_draft does not follow the last step");
        last = *it;
        lasts.erase(it);
        if (accepted > 0) // the per-position state of the last accepted draft becomes the sequence's state; no copy
            slot_of.at(last.seq) = int(last.vslot[size_t(accepted)]);
        for (size_t i = 1; i < last.vslot.size(); ++i)
            reserved.erase(int(last.vslot[i]));
        if (!last.x_valid)
            return drafts;
        const Graph *g1 = mtp_graph(last.Q), *gc = mtp_graph(1);
        if (!mtp_ready || want <= 0 || !g1)
            return drafts;
        want = std::min(want, mtp_max_drafts());
        // only as many drafts as a verify bucket of the next step's context can take (wide buckets may stop at fewer pages)
        while (want > 0 && mtp_max_drafts(last_requests, (position + accepted + 2 + want + PAGE - 1) / PAGE) < want)
            --want;
        if (want <= 0)
            return drafts;
        auto &ms = mtp_seq[last.seq];
        if (ms.next != position) // the cache has no entry for the previous position (first step after prefill / a batched step)
            ms.lo = position;
        ms.next = position + accepted + 1;
        mtp_logits.resize(size_t(vocab_));
        // the window: the g.P pages ending at the page of the last position this call writes
        auto window = [&](const Graph &g) { return std::max(0, (position + accepted + want - 1) / PAGE - g.P + 1); };
        // BEHEMOTH_NEURON_SPEC_MTP_PMIN=p: a draft the head gives less than p is dropped and ends the chain (it would be
        // rejected with probability ~1 - p and cost a verify row; the next step is then narrower)
        int launches = 1;
        if (const Graph *gch = mtp_pmin > 0 ? nullptr : chain_graph(last.Q, want)) { // the whole chain in one launch
            mtp_launch(*gch, true, last.parity, pages, position, next, window(*gch), ms.lo, last.R, last.row);
            drafts = mtp_chain_out;
            static int debug = std::getenv("BEHEMOTH_NEURON_MTP_CHAIN_DEBUG") ? std::atoi(std::getenv("BEHEMOTH_NEURON_MTP_CHAIN_DEBUG")) : 0;
            if (debug > 0) { // the first launches: the chain's raw output against the host-chained launches (which rewrite the same entries)
                --debug;
                std::string line = "mtp chain Q=" + std::to_string(gch->Q) + " D=" + std::to_string(want) + " raw";
                for (size_t j = 0; j < std::min(gch->out0 / 4, size_t(64)); ++j)
                    line += " " + std::to_string(mtp_chain_raw[j]).substr(0, 9);
                line += " host";
                int tok = mtp_launch(*g1, true, last.parity, pages, position, next, window(*g1), ms.lo, last.R, last.row);
                line += " " + std::to_string(tok);
                for (int j = 1; j < want; ++j) {
                    const int one[1] = {tok};
                    tok = mtp_launch(*gc, false, (j - 1) % 2, pages, position + accepted + j, one, window(*gc), ms.lo);
                    line += " " + std::to_string(tok);
                }
                std::cerr << line << "\n";
            }
            mtp_ms = ms_since(t0);
            profile["mtp_ms"] = mtp_ms;
            profile["mtp_launches"] = launches;
            return drafts;
        }
        drafts.push_back(mtp_launch(*g1, true, last.parity, pages, position, next, window(*g1), ms.lo, last.R, last.row));
        for (int j = 1; j < want && mtp_last_p >= mtp_pmin; ++j, ++launches) {
            const int tok[1] = {drafts.back()};
            drafts.push_back(mtp_launch(*gc, false, (j - 1) % 2, pages, position + accepted + j, tok, window(*gc), ms.lo));
        }
        if (mtp_last_p < mtp_pmin)
            drafts.pop_back();
        mtp_ms = ms_since(t0);
        profile["mtp_ms"] = mtp_ms;
        profile["mtp_launches"] = launches;
        return drafts;
    }
    int cache_pages() const override { return pages_total - 1; }
    int vocab() const override { return vocab_; }
    int max_decode_rows() const override {
        int r = 0;
        for (const auto &g : graphs)
            if (g.has_gdn && g.Q == 1) r = std::max(r, g.R);
        return std::min(r, slots_total - 1);
    }
    int decode_bucket_pages(int pages) const override {
        int best = 0;
        for (const auto &g : graphs)
            if (g.has_gqa && g.Q == 1 && g.P >= pages && (!best || g.P < best)) best = g.P;
        return best;
    }
    const std::vector<int> &eos() const override { return eos_; }
    int max_context() const override {
        int p = 0;
        for (const auto &g : graphs)
            if (g.has_gqa) p = std::max(p, g.P);
        return std::min(p * PAGE, max_context_);
    }
    json last_profile() const override { return profile; }

    std::span<const float> forward(const InferenceBatch &batch) override {
        const auto t0 = Clock::now();
        layers_ms = head_ms = exec_ms = plan_ms = 0;
        moe_grouped = moe_fallbacks = 0;
        moe_load_max = 0;
        prof.assign(size_t(tp), {});
        launches = 0;
        const auto &lay = batch.layout;
        logits.resize(batch.sample_rows.size() * size_t(vocab_));
        std::vector<int> sample_of(size_t(lay.queries()), -1);
        for (size_t s = 0; s < batch.sample_rows.size(); ++s)
            sample_of[size_t(batch.sample_rows[s])] = int(s);
        auto pages_needed = [&](int req) { return (lay.lengths[size_t(req)] + PAGE - 1) / PAGE; };
        auto page_at = [&](int req, int j) { return lay.page_table[size_t(req) * size_t(lay.table_stride) + size_t(j)]; };
        auto fill_table = [&](Launch &L, int row, int req) {
            for (int j = 0; j < L.P; ++j) {
                const int page = j < pages_needed(req) ? page_at(req, j) : -1;
                L.table[size_t(row) * size_t(L.P) + size_t(j)] = page < 0 ? pages_total - 1 : page;
            }
        };
        // decode rows (one token per request) and prefill requests (several tokens: one R = 1 launch per chunk)
        // a request whose rows are ALL sampled and more than one: a speculative verify chunk (scheduled token + drafts)
        std::vector<int> rows, multi, verify;
        const std::set<int> vq = verify_Q();
        for (int req = 0; req < lay.requests(); ++req) {
            const int first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
            int sampled = 0;
            for (int i = 0; i < count; ++i)
                sampled += sample_of[size_t(first + i)] >= 0;
            // BEHEMOTH_NEURON_QWEN35_STRICT_VERIFY=1: a lone decode token also runs in the verify bucket (one token + padding), so
            // every token of a stream is computed by the same graph family whether drafts were attached or not: speculative
            // and non-speculative greedy streams are then comparable bit for bit (the R = 1 decode graph rounds differently)
            const bool strict = strict_verify();
            if ((count > 1 && sampled == count) || (strict && count == 1 && sampled == 1 && lay.requests() == 1 && !vq.empty() && lay.positions[size_t(first)] > 0)) {
                if (vq.lower_bound(count) == vq.end())
                    throw std::runtime_error("Qwen35: no verify bucket for " + std::to_string(count) + " tokens");
                verify.push_back(req);
            } else
                (count == 1 ? rows : multi).push_back(req);
        }
        reserved.clear(); // verify launches nobody committed: the sequences keep the state of their scheduled token
        lasts.clear();
        auto note = [&](Last l) { // a launch reusing a bucket overwrites the residual rows of the earlier one
            l.launch = launches; // (execute() has counted it: the rows of one launch share the id)
            for (auto &o : lasts)
                if (o.R == l.R && o.Q == l.Q && o.launch != l.launch)
                    o.x_valid = false;
            lasts.push_back(std::move(l));
        };
        // several decode requests of which at least one carries drafts: ONE R x Q verify launch (a request without drafts is
        // a row of one token + padding) when such a bucket exists; else the R = 1 verify launches + the batched rows below
        if (!verify.empty() && verify.size() + rows.size() >= 2) {
            std::vector<int> all = verify;
            all.insert(all.end(), rows.begin(), rows.end());
            std::sort(all.begin(), all.end());
            int maxc = 1, pages = 1;
            for (int req : all) {
                maxc = std::max(maxc, lay.query_offsets[size_t(req) + 1] - lay.query_offsets[size_t(req)]);
                pages = std::max(pages, pages_needed(req));
            }
            const auto [R, Q, P] = verify_bucket_rows(int(all.size()), maxc, pages);
            if (P && head_for(R * Q, 1)) {
                auto L = fresh(R, Q, P);
                L.verify = true;
                std::set<int> pinned;
                std::vector<Last> made;
                for (size_t i = 0; i < all.size(); ++i) {
                    const int req = all[i], first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
                    const int position = lay.positions[size_t(first)];
                    std::vector<int64_t> vs(static_cast<size_t>(count), int64_t(0));
                    L.slot[i] = vs[0] = acquire_slot(page_at(req, 0), position, pinned);
                    for (int j = 0; j < Q; ++j) {
                        const size_t t = i * size_t(Q) + size_t(j);
                        if (j < count) {
                            fill_token(L, int(t), batch.tokens[size_t(first + j)], position + j, page_at(req, (position + j) / PAGE));
                            if (j) {
                                vs[size_t(j)] = free_slot(pinned);
                                pinned.insert(int(vs[size_t(j)]));
                            }
                            L.vslot[t] = vs[size_t(j)];
                            if (sample_of[size_t(first + j)] >= 0) {
                                L.head_rows.push_back(int(t));
                                L.destination.push_back(sample_of[size_t(first + j)]);
                            }
                        } else
                            L.pos[t] = -1;
                    }
                    fill_table(L, int(i), req);
                    made.push_back({true, true, page_at(req, 0), position, count, Q, 0, R, int(i), vs});
                }
                for (size_t t = all.size() * size_t(Q); t < size_t(R * Q); ++t)
                    if (t % size_t(Q))
                        L.pos[t] = -1;
                execute(L, logits.data());
                for (auto &m : made) {
                    m.parity = final_parity;
                    reserved.insert(m.vslot.begin() + 1, m.vslot.end());
                    note(std::move(m));
                }
                verify.clear();
                rows.clear();
            }
        }
        std::set<int> decode_R;
        std::set<int> prefill_Q;
        for (const auto &g : graphs)
            if (g.has_gdn && !g.verify) (g.Q == 1 ? decode_R.insert(g.R) : prefill_Q.insert(g.Q));
        for (int req : verify) {
            const int first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
            // strict mode: the table width must not depend on how many drafts ride along. A chunk p..p+k picks its bucket by
            // its LAST position, so near a bucket edge (e.g. 512 -> 1024 pages at 8192 tokens) the k = 2 stream computed tokens
            // p, p + 1 in the wider bucket where the k = 0 stream used the narrower one: other bits from there on (docs/neuron-
            // qwen397-mtp.md section 17). Strict launches therefore use the widest verify bucket, or the narrowest one with at
            // least BEHEMOTH_NEURON_QWEN35_STRICT_PAGES pages when that is set (and covers the request).
            int vpages = pages_needed(req);
            if (strict_verify()) {
                static const int pin = std::getenv("BEHEMOTH_NEURON_QWEN35_STRICT_PAGES") ? std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_STRICT_PAGES")) : 0;
                int widest = 0;
                for (const auto &g : graphs)
                    if (g.verify && g.has_gdn && g.R == 1 && g.has_gqa)
                        widest = std::max(widest, g.P);
                vpages = std::max(vpages, pin > 0 ? pin : widest);
            }
            // BEHEMOTH_NEURON_QWEN35_STRICT_Q=q (strict mode): every launch in a verify bucket of at least q tokens, so streams with
            // different draft counts run the same graph (Q buckets are separate compiles that may round differently)
            static const int strict_q = std::getenv("BEHEMOTH_NEURON_QWEN35_STRICT_Q") ? std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_STRICT_Q")) : 0;
            const auto [Q, P] = verify_bucket(strict_verify() ? std::max(count, strict_q) : count, vpages);
            if (!P)
                throw std::runtime_error("Qwen35: no verify bucket for " + std::to_string(count) + " tokens with " + std::to_string(pages_needed(req)) + " pages");
            auto L = fresh(1, Q, P);
            L.verify = true;
            std::set<int> pinned;
            const int position = lay.positions[size_t(first)];
            L.slot[0] = L.vslot[0] = acquire_slot(page_at(req, 0), position, pinned);
            for (int i = 0; i < Q; ++i) {
                if (i < count) {
                    fill_token(L, i, batch.tokens[size_t(first + i)], position + i, page_at(req, (position + i) / PAGE));
                    if (i) { // the state after draft i goes to a spare slot; mtp_draft() makes the accepted one the sequence's
                        L.vslot[size_t(i)] = free_slot(pinned);
                        pinned.insert(int(L.vslot[size_t(i)]));
                    }
                    L.head_rows.push_back(i);
                    L.destination.push_back(sample_of[size_t(first + i)]);
                } else
                    L.pos[size_t(i)] = -1;
            }
            fill_table(L, 0, req);
            execute(L, logits.data());
            note({true, true, page_at(req, 0), position, count, Q, final_parity, 1, 0, std::vector<int64_t>(L.vslot.begin(), L.vslot.begin() + count)});
            reserved.insert(L.vslot.begin() + 1, L.vslot.begin() + count);
        }
        last_requests = std::max<int>(1, int(lasts.size() + rows.size()));
        const int maxR = std::min(decode_R.empty() ? 0 : *decode_R.rbegin(), slots_total - 1);
        // BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R=n (opt-in, nonfinite-fault workaround study, docs/neuron-reap-perf.md): decode
        // launches use a row bucket of at least n rows (fewer rows are padded with the scratch slot / page), i.e. never the
        // 1-row decode NEFFs when n > 1.
        static const int min_r = std::getenv("BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R") ? std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R")) : 0;
        for (size_t at = 0; at < rows.size();) {
            if (!maxR)
                throw std::runtime_error("Qwen35: no decode buckets");
            size_t n = std::min(rows.size() - at, size_t(maxR));
            const int want = std::min(std::max(int(n), min_r), maxR);
            int R = *decode_R.lower_bound(want), pages = 1;
            for (size_t i = 0; i < n; ++i)
                pages = std::max(pages, pages_needed(rows[at + i]));
            // the narrowest row bucket >= n that has a page bucket for these rows (wide buckets stop at fewer pages)
            int P = 0;
            for (auto it = decode_R.lower_bound(want); it != decode_R.end() && !P; ++it)
                if (*it <= slots_total - 1 && (P = bucket_pages(*it, 1, pages)))
                    R = *it;
            if (!P) { // no bucket holds n rows at this context: split the rows over narrower buckets that do
                for (auto it = decode_R.rbegin(); it != decode_R.rend() && !P; ++it)
                    if (*it < int(n) && (P = bucket_pages(*it, 1, pages)))
                        R = *it;
                if (!P)
                    throw std::runtime_error("Qwen35: no decode bucket with " + std::to_string(pages) + " pages");
                n = size_t(R);
                P = 0;
                pages = 1;
                for (size_t i = 0; i < n; ++i)
                    pages = std::max(pages, pages_needed(rows[at + i]));
                P = bucket_pages(R, 1, pages);
            }
            auto L = fresh(R, 1, P);
            std::set<int> pinned;
            for (size_t i = 0; i < n; ++i) {
                const int req = rows[at + i], ti = lay.query_offsets[size_t(req)], position = lay.positions[size_t(ti)];
                fill_token(L, int(i), batch.tokens[size_t(ti)], position, page_at(req, position / PAGE));
                fill_table(L, int(i), req);
                L.slot[i] = acquire_slot(page_at(req, 0), position, pinned);
                if (sample_of[size_t(ti)] >= 0) {
                    L.head_rows.push_back(int(i));
                    L.destination.push_back(sample_of[size_t(ti)]);
                }
            }
            execute(L, logits.data());
            for (size_t i = 0; i < n; ++i) { // every decode row can start a draft chain (its residual row: a slice of the bucket's buffer)
                const int req = rows[at + i], ti = lay.query_offsets[size_t(req)];
                note({true, true, page_at(req, 0), lay.positions[size_t(ti)], 1, 1, final_parity, R, int(i), {}});
            }
            at += n;
        }
        for (int req : multi) {
            const int first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
            for (int done = 0; done < count;) {
                const int left = count - done;
                // the smallest chunk bucket covering what is left, else the largest
                int Q = prefill_Q.empty() ? 0 : *prefill_Q.rbegin();
                if (auto it = prefill_Q.lower_bound(left); it != prefill_Q.end()) Q = *it;
                // BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1 (opt-in; sets with wide chunk buckets, e.g. Q = 2048): when the covering
                // bucket would be padded by more than 25 %, run the largest bucket that is completely filled instead and
                // leave the rest to the next launch (600 tokens left: 512 + 256 rows instead of one 2048-row launch)
                static const bool fit = std::getenv("BEHEMOTH_NEURON_QWEN35_BUCKET_FIT") && std::atoi(std::getenv("BEHEMOTH_NEURON_QWEN35_BUCKET_FIT"));
                if (fit && Q * 4 > left * 5) {
                    auto it = prefill_Q.upper_bound(left);
                    if (it != prefill_Q.begin()) Q = *std::prev(it);
                }
                if (!Q) { // no prefill buckets: run the prompt as decode rows of one request
                    Q = 1;
                }
                const int take = std::min(left, Q);
                const int PQ = bucket_pages(1, Q, pages_needed(req));
                if (!PQ)
                    throw std::runtime_error("Qwen35: no prefill bucket Q=" + std::to_string(Q) + " with " + std::to_string(pages_needed(req)) + " pages");
                auto L = fresh(1, Q, PQ);
                std::set<int> pinned;
                for (int i = 0; i < Q; ++i) {
                    if (i < take) {
                        const int ti = first + done + i, position = lay.positions[size_t(ti)];
                        fill_token(L, i, batch.tokens[size_t(ti)], position, page_at(req, position / PAGE));
                    } else
                        L.pos[size_t(i)] = -1; // padding: no state update, scratch write slot
                }
                fill_table(L, 0, req);
                L.slot[0] = acquire_slot(page_at(req, 0), lay.positions[size_t(first + done)], pinned);
                const int last = first + done + take - 1;
                if (sample_of[size_t(last)] >= 0) {
                    L.head_rows.push_back(take - 1);
                    L.destination.push_back(sample_of[size_t(last)]);
                }
                execute(L, logits.data());
                done += take;
            }
        }
        if (const char *dump = std::getenv("NEURON_LOGITS_DUMP")) {
            std::ofstream f(dump, std::ios::binary | std::ios::app);
            f.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * 4));
        }
        std::array<double, 7> p0 = prof[0], pmax = prof[0];
        for (const auto &p : prof)
            for (size_t i = 0; i < 7; ++i)
                pmax[i] = std::max(pmax[i], p[i]);
        profile = {{"launches", launches}, {"forward_ms", ms_since(t0)}, {"layers_ms", layers_ms}, {"head_ms", head_ms}, {"exec_ms", exec_ms}, {"plan_ms", plan_ms},
                   {"r0", {{"set", p0[0]}, {"gdn", p0[1]}, {"gqa", p0[2]}, {"moe", p0[3]}, {"put", p0[4]}, {"head", p0[5]}, {"read", p0[6]}}},
                   {"rmax", {{"set", pmax[0]}, {"gdn", pmax[1]}, {"gqa", pmax[2]}, {"moe", pmax[3]}, {"put", pmax[4]}, {"head", pmax[5]}, {"read", pmax[6]}}},
                   {"tokens", batch.tokens.size()}, {"moe_grouped", moe_grouped}, {"moe_fallbacks", moe_fallbacks}, {"moe_load_max", moe_load_max}, {"kda_spills", spills}, {"kda_restores", restores}, {"state_saves", state_saves}, {"state_restores", state_restores}};
        return logits;
    }
};

bool is_qwen35_prepared(const std::filesystem::path &p) {
    const auto f = p / "prepared.json";
    if (!std::filesystem::exists(f))
        return false;
    return read_json(f).value("format", "") == "qwen35-moe-neuron-prepared";
}
std::unique_ptr<Backend> make_qwen35_backend(const ModelOptions &o) { return std::make_unique<Qwen35Model>(o); }
} // namespace neuron
