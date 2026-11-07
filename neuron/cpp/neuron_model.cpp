#include "neuron_model.hpp"
#include "qtip_expand.hpp"
#include "behemoth/checkpoint.hpp"
#include "behemoth/worker.hpp"
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

using namespace behemoth;

namespace neuron {
namespace {
constexpr int PAGE = 16; // tokens per KV page (the server's AttentionLayout)
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void check(NRT_STATUS status, const std::string &what) {
    if (status != NRT_SUCCESS)
        throw std::runtime_error(what + " failed: nrt status " + std::to_string(int(status)));
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
void put(const Dev &d, const void *src, size_t bytes, size_t offset = 0) {
    check(nrt_tensor_write(d.t, src, offset, bytes), "write");
}
void zero(const Dev &d) {
    std::vector<uint8_t> z(std::min<size_t>(d.bytes, 64u << 20), 0);
    for (size_t off = 0; off < d.bytes; off += z.size())
        put(d, z.data(), std::min(z.size(), d.bytes - off), off);
}

// The geometry a NEFF was traced for (meta.json "geometry", trace_model.py). Buckets without it are from the
// first generator version, which only produced Llama 3.1 8B graphs at TP=2.
struct Geometry {
    int hidden = 4096, heads = 32, kv_heads = 8, dim = 128, intermediate = 14336, vocab = 128257, layers = 32, tp = 2;
    static Geometry of(const ModelConfig &c, int tp) {
        return {c.hidden, c.q_heads, c.kv_heads, c.dim, c.intermediate, c.vocab, c.layers, tp};
    }
    static Geometry parse(const nlohmann::json &j) {
        Geometry g;
        g.hidden = j.at("hidden"); g.heads = j.at("heads"); g.kv_heads = j.at("kv_heads"); g.dim = j.at("head_dim");
        g.intermediate = j.at("intermediate"); g.vocab = j.at("vocab"); g.layers = j.value("layers", 0); g.tp = j.at("tp");
        return g;
    }
    bool operator==(const Geometry &o) const { // layer count is not a graph property (one layer NEFF serves all layers)
        return hidden == o.hidden && heads == o.heads && kv_heads == o.kv_heads && dim == o.dim && intermediate == o.intermediate &&
               vocab == o.vocab && tp == o.tp;
    }
    std::string str() const {
        return "hidden " + std::to_string(hidden) + " heads " + std::to_string(heads) + "/" + std::to_string(kv_heads) + " dim " +
               std::to_string(dim) + " ff " + std::to_string(intermediate) + " vocab " + std::to_string(vocab) + " TP " + std::to_string(tp);
    }
};

struct Meta {
    std::string kind;
    int R = 0, Q = 0, P = 0, S = 0, T = 0, L = 0; // kind "model": L layers in this graph (a bucket's total after discover())
    bool head = true;                             // kind "model": ends with the final norm + lm_head (else outputs the residual)
    bool fp8 = false;                             // kind "head": E4M3 lm_head with per-row scale (5 inputs) instead of bf16 (4)
    bool wt = false;                              // kind "layer": FP8 projections stored transposed [in][out] (trace_model.py WT=1)
    std::string weights = "fp8";                  // kind "layer": "fp8" (seven E4M3 shards) or "qtip" (q/o/gate/up/down trellis resident)
    std::vector<std::string> order;               // kind "layer": input names (trace_model.py ORDER / ORDER_QTIP), "cache" is aliased
    Geometry geometry;
    std::filesystem::path neff;
};

// Layer NEFF inputs of the first generator version (no "order" in meta.json): the FP8 layer.
const std::vector<std::string> kFp8Order = {"x", "cos_sin", "pos", "wslot", "table", "cache", "norm1", "norm2", "qf", "qs", "of", "os",
                                            "kf", "ks", "vf", "vs", "gf", "gs", "uf", "us", "df", "ds"};
// NEFF input order: the meta order with the aliased KV state moved to the end (torch_neuronx puts module state last).
std::vector<std::string> neff_input_names(const Meta &m) {
    std::vector<std::string> names;
    for (const auto &n : m.order.empty() ? kFp8Order : m.order)
        if (n != "cache") names.push_back(n);
    names.push_back("cache");
    return names;
}
constexpr int kFp8Inputs = 22;
// Continuations of up to this many tokens run as rows of a decode bucket instead of a prefill chunk.
constexpr int kRowPathTokens = 8;

struct LayerWeights {
    bool compressed = false;     // QTIP q/o/gate/up/down resident as trellis words (WEIGHTS=qtip NEFF) instead of FP8 shards
    Dev norm1, norm2, w[14], kv; // w: qf qs of os kf ks vf vs gf gs uf us df ds; kv: [pages, 2 (K, V), kv_heads, 16, dim] bf16
    struct Packed { Dev words, lut, su, sv; } q[5]; // compressed layers: q, o, gate, up, down (w keeps only kf ks vf vs)
    size_t bytes() const {
        size_t b = norm1.bytes + norm2.bytes;
        for (const auto &d : w) b += d.bytes;
        for (const auto &p : q) b += p.words.bytes + p.lut.bytes + p.su.bytes + p.sv.bytes;
        return b;
    }
};

struct RankState {
    int core = 0;
    std::vector<LayerWeights> layers;
    Dev head_w, head_norm, head_ws; // head_w: bf16 rows, or E4M3 rows + head_ws scales when an fp8 head NEFF is loaded
    Dev sel_h, sel_ff;              // compressed layers: this rank's slice of a full hidden / intermediate vector (int64 indices)
};

struct LayerBucket {
    Meta meta;
    std::vector<nrt_model_t *> model;         // [rank]: the meta.weights variant (the first layer NEFF found for this R, Q, P)
    // Layer NEFFs of the same (R, Q, P) traced for the other weight format: a layer runs the variant of its own format.
    struct Variant { Meta meta; std::vector<nrt_model_t *> model; };
    std::map<std::string, Variant> variants;  // by weights ("fp8", "qtip"), including meta.weights -> {meta, model}
    const std::vector<nrt_model_t *> &model_for(bool compressed) const { return variants.at(compressed ? "qtip" : "fp8").model; }
    // Whole-model buckets run as a chain of segments (one segment when the toolchain compiles all layers into one graph).
    struct Seg {
        Meta meta;
        int first_layer = 0;
        std::vector<int> layers; // layer index bound to each of the segment's L slots
        std::vector<nrt_model_t *> model;
        std::vector<nrt_tensor_set_t *> in, out; // [rank]
    };
    std::vector<Seg> segs;
    struct PerRank {
        Dev x[2], cos_sin, pos, wslot, table;
        std::vector<std::array<nrt_tensor_set_t *, 2>> in, out; // [layer][parity]; whole-model buckets: [0][0]
        Dev rows, logits;                                      // whole-model buckets only
    };
    std::vector<PerRank> rank;
    int tokens() const { return meta.R * meta.Q; }
    bool full() const { return !segs.empty(); }
    bool in_graph_head() const { return full() && segs.back().meta.head; }
};
struct HeadBucket {
    Meta meta;
    std::vector<nrt_model_t *> model;
    struct PerRank {
        Dev rows, logits;
        // by the layer bucket whose residual buffer feeds the head, then by parity of the buffer holding the final x
        std::map<const LayerBucket *, std::array<nrt_tensor_set_t *, 2>> in;
        std::array<nrt_tensor_set_t *, 2> out{};
    };
    std::vector<PerRank> rank;
};

std::vector<char> slurp(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot read " + p.string());
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
} // namespace

struct NeuronModel::Impl {
    ModelOptions options;
    ModelConfig config;
    std::unique_ptr<SafeTensorFile> embed_file;
    FloatTensorView embed, final_norm;
    std::vector<float> inv_freq;
    int tp = 0, hidden = 0, dim = 0;                           // from the NEFF geometry (checked against config.json)
    int vocab_ = 0, vocab_shard = 0, pages_total = 0, kv_local = 0; // vocab_shard: lm_head rows per rank (vocab padded to TP)
    bool head_transposed = false;                              // the head NEFFs take lm_head as [H][V/TP] (WT=1)
    bool head_fp8 = false;                                     // the head NEFFs take E4M3 lm_head rows
    std::vector<RankState> ranks;
    std::vector<LayerBucket> decode, prefill;
    std::vector<HeadBucket> heads;
    std::vector<Meta> segments; // whole-model graphs found by discover(), before they are assembled into buckets
    std::vector<float> logits;
    std::vector<std::unique_ptr<RankWorker>> workers;
    nlohmann::json profile = nlohmann::json::object();
    double phase_upload_ms = 0, phase_layers_ms = 0, phase_head_ms = 0; // rank 0, summed over the launches of a forward
    size_t weight_bytes_rank = 0, kv_bytes_rank = 0;                   // resident per NeuronCore, for the load report
    bool weights_transposed = false, wt_seen = false;                  // FP8 NEFFs take [in][out] projections (meta.wt, all alike)
    std::vector<bool> compressed;                                      // per layer: QTIP projections resident as trellis (decide_modes)
    int n_compressed = 0;

    explicit Impl(const ModelOptions &o) : options(o), config(ModelConfig::read(o.model / "config.json")) {
        if (const char *n = std::getenv("NEURON_LAYERS")) // debugging: run only the first n layers (and whole-model NEFFs traced with L=n)
            config.layers = std::min(config.layers, std::atoi(n));
        vocab_ = config.vocab;
        hidden = config.hidden;
        dim = config.dim;
        discover();
        auto needs_head = [](const std::vector<LayerBucket> &v) { return std::any_of(v.begin(), v.end(), [](const auto &b) { return !b.in_graph_head(); }); };
        if (decode.empty() || prefill.empty() || ((needs_head(decode) || needs_head(prefill)) && heads.empty()))
            throw std::runtime_error("need decode and prefill NEFF buckets (and head NEFFs for per-layer buckets) in " + options.neff_dir.string());
        vocab_shard = (vocab_ + tp - 1) / tp;
        kv_local = config.kv_heads / tp;
        // Implicit async execution (launches return before the device finishes; the logits read synchronises): hides the per-layer
        // launch gap, 10 % on single-stream decode and more with wide batches. A value set by the caller wins.
        setenv("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "8", 0);
        check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init");
        ranks.resize(size_t(tp));
        for (int r = 0; r < tp; ++r) {
            ranks[size_t(r)].core = r;
            workers.push_back(std::make_unique<RankWorker>());
        }
        embed_file = std::make_unique<SafeTensorFile>(o.model / "model-embed.safetensors");
        embed = embed_file->floats("model.embed_tokens.weight");
        final_norm = embed_file->floats("model.norm.weight");
        if (config.rope_llama3)
            inv_freq = llama3_inv_freq(dim, config.theta, config.rope);
        else {
            inv_freq.resize(size_t(dim / 2));
            for (int d = 0; d < dim / 2; ++d)
                inv_freq[size_t(d)] = float(std::pow(double(config.theta), -2.0 * d / dim));
        }
        load_buckets();
        load_weights();
    }

    // Every bucket must have been traced for this checkpoint's geometry and one common TP; the TP decides the ranks.
    void accept(const Meta &m) {
        if (!tp)
            tp = m.geometry.tp;
        if (m.geometry.tp != tp)
            throw std::runtime_error("NEFF buckets in " + options.neff_dir.string() + " were traced for different TP degrees");
        if (!(m.geometry == Geometry::of(config, tp)))
            throw std::runtime_error("NEFF " + m.neff.string() + " was traced for " + m.geometry.str() + ", the checkpoint is " +
                                     Geometry::of(config, tp).str() + " (retrace with MODEL=" + options.model.string() + ")");
    }

    void discover() {
        for (const auto &entry : std::filesystem::directory_iterator(options.neff_dir)) {
            const auto meta_path = entry.path() / "meta.json";
            if (!entry.is_directory() || !std::filesystem::exists(meta_path))
                continue;
            std::ifstream f(meta_path);
            auto j = nlohmann::json::parse(f);
            Meta m;
            m.kind = j.at("kind");
            m.neff = entry.path() / "graph.neff";
            if (j.contains("geometry"))
                m.geometry = Geometry::parse(j.at("geometry"));
            if (m.kind == "model") {
                if (std::getenv("NEURON_USE_LAYERS"))
                    continue;
                m.R = j.at("R"); m.Q = j.at("Q"); m.P = j.at("P"); m.L = j.at("L"); m.S = j.at("S");
                m.head = j.value("head", true);
                accept(m);
                segments.push_back(m);
            } else if (m.kind == "layer") {
                m.R = j.at("R"); m.Q = j.at("Q"); m.P = j.at("P");
                m.weights = j.value("weights", "fp8");
                m.wt = j.value("wt", false);
                if (m.weights == "fp8") {
                    if (wt_seen && weights_transposed != m.wt)
                        throw std::runtime_error("FP8 layer NEFFs in " + options.neff_dir.string() + " mix transposed (WT=1) and plain weights");
                    weights_transposed = m.wt;
                    wt_seen = true;
                }
                if (j.contains("order"))
                    for (const auto &n : j.at("order")) m.order.push_back(n);
                if (m.weights != "fp8" && m.weights != "qtip")
                    throw std::runtime_error("NEFF " + m.neff.string() + " has unknown weights format " + m.weights);
                accept(m);
                auto &set = m.Q == 1 ? decode : prefill;
                LayerBucket *same = nullptr;
                for (auto &b : set)
                    if (!b.full() && b.meta.R == m.R && b.meta.Q == m.Q && b.meta.P == m.P) same = &b;
                if (same) {
                    if (same->variants.count(m.weights))
                        throw std::runtime_error("two " + m.weights + " layer NEFFs for the same bucket R=" + std::to_string(m.R) + " Q=" +
                                                 std::to_string(m.Q) + " P=" + std::to_string(m.P) + " in " + options.neff_dir.string());
                    same->variants[m.weights] = {m, {}};
                } else {
                    LayerBucket b;
                    b.meta = m;
                    b.variants[m.weights] = {m, {}};
                    set.push_back(std::move(b));
                }
            } else {
                m.S = j.at("S"); m.T = j.at("T");
                m.fp8 = j.value("fp8", false);
                m.wt = j.value("wt", false);
                accept(m);
                if (!heads.empty() && (heads.front().meta.fp8 != m.fp8 || heads.front().meta.wt != m.wt))
                    throw std::runtime_error("head NEFFs mix bf16 / fp8 or transposed / plain lm_head formats");
                head_fp8 = m.fp8;
                head_transposed = m.wt;
                HeadBucket b;
                b.meta = m;
                heads.push_back(std::move(b));
            }
        }
        if (!tp)
            throw std::runtime_error("no NEFF buckets (meta.json + graph.neff directories) in " + options.neff_dir.string());
        // Whole-model graphs of one (R, Q, P): body segments (residual out), optionally ending in one head segment (logits
        // out), chained over the layers in order. Used when their layer counts add up to the checkpoint's; without a head
        // segment the separate head NEFF of the same token count produces the logits.
        std::map<std::array<int, 3>, std::vector<Meta>> groups;
        for (const auto &m : segments)
            groups[{m.R, m.Q, m.P}].push_back(m);
        for (auto &[key, metas] : groups) {
            const Meta *head = nullptr;
            const Meta *body = nullptr;
            for (const auto &m : metas) {
                if (m.head)
                    head = &m;
                else if (!body || m.L > body->L)
                    body = &m;
            }
            int remaining = config.layers - (head ? head->L : 0);
            LayerBucket bucket;
            while (body && remaining >= body->L) {
                bucket.segs.push_back({*body, 0, {}, {}, {}, {}});
                remaining -= body->L;
            }
            if (remaining != 0 || (!head && !body))
                continue; // layer counts do not add up (a partial-depth probe graph)
            if (head)
                bucket.segs.push_back({*head, 0, {}, {}, {}, {}});
            int first = 0;
            const char *shift = std::getenv("NEURON_SEG_SHIFT"); // debugging: bind later segments to layers shifted by this
            for (size_t g = 0; g < bucket.segs.size(); ++g) {
                bucket.segs[g].first_layer = first + (g > 0 && shift ? std::atoi(shift) : 0);
                for (int i = 0; i < bucket.segs[g].meta.L; ++i)
                    bucket.segs[g].layers.push_back(bucket.segs[g].first_layer + i);
                first += bucket.segs[g].meta.L;
            }
            if (const char *map = std::getenv("NEURON_SEG_LAYERS")) { // debugging: "0-15;16-30,30" explicit layers per segment
                const std::string text = map;
                size_t g = 0;
                for (size_t at = 0; at <= text.size() && g < bucket.segs.size(); ++g) {
                    const size_t semi = std::min(text.find(';', at), text.size());
                    std::string part = text.substr(at, semi - at);
                    bucket.segs[g].layers.clear();
                    for (size_t q = 0; q <= part.size();) {
                        const size_t comma = std::min(part.find(',', q), part.size());
                        const std::string item = part.substr(q, comma - q);
                        const size_t dash = item.find('-');
                        const int lo = std::stoi(item.substr(0, dash)), hi = dash == std::string::npos ? lo : std::stoi(item.substr(dash + 1));
                        for (int l = lo; l <= hi; ++l) bucket.segs[g].layers.push_back(l);
                        q = comma + 1;
                    }
                    at = semi + 1;
                }
            }
            bucket.meta = head ? *head : *body;
            bucket.meta.L = config.layers;
            bucket.meta.head = head != nullptr;
            (bucket.meta.Q == 1 ? decode : prefill).push_back(std::move(bucket));
        }
        auto by_size = [](const auto &a, const auto &b) { return a.meta.P != b.meta.P ? a.meta.P < b.meta.P : a.meta.Q != b.meta.Q ? a.meta.Q < b.meta.Q : a.meta.L > b.meta.L; };
        std::sort(decode.begin(), decode.end(), by_size);
        std::sort(prefill.begin(), prefill.end(), by_size);
        std::sort(heads.begin(), heads.end(), [](const auto &a, const auto &b) { return a.meta.T != b.meta.T ? a.meta.T < b.meta.T : a.meta.S < b.meta.S; });
    }

    size_t kv_bytes() const { return size_t(pages_total) * 2 * size_t(kv_local) * PAGE * size_t(dim) * 2; }

    // Load every NEFF on both cores and create its step tensors. Tensor sets are built after the weights exist.
    void load_buckets() {
        auto load_layer = [&](LayerBucket &b) {
            std::vector<std::vector<char>> neffs;
            std::vector<std::string> variant_names; // per-layer buckets: the variant each NEFF belongs to
            if (b.full())
                for (const auto &seg : b.segs) neffs.push_back(slurp(seg.meta.neff));
            else
                for (auto &[name, v] : b.variants) {
                    neffs.push_back(slurp(v.meta.neff));
                    variant_names.push_back(name);
                    v.model.resize(size_t(tp));
                }
            b.model.resize(size_t(tp));
            b.rank.resize(size_t(tp));
            for (auto &seg : b.segs) {
                seg.model.resize(size_t(tp));
                seg.in.resize(size_t(tp));
                seg.out.resize(size_t(tp));
            }
            std::vector<std::future<void>> jobs;
            for (int r = 0; r < tp; ++r)
                jobs.push_back(workers[size_t(r)]->submit([&, r] {
                    for (size_t g = 0; g < neffs.size(); ++g) {
                        const auto &neff = neffs[g];
                        nrt_model_t *&handle = b.full() ? b.segs[g].model[size_t(r)] : b.variants.at(variant_names[g]).model[size_t(r)];
                        check(nrt_load_collectives(neff.data(), neff.size(), r, -1, uint32_t(r), uint32_t(tp), &handle), "nrt_load_collectives");
                        if (!b.full() && variant_names[g] == b.meta.weights)
                            b.model[size_t(r)] = handle;
                        nrt_tensor_info_array_t *info = nullptr;
                        check(nrt_get_model_tensor_info(handle, &info), "tensor info");
                        const Meta &m = b.full() ? b.segs[g].meta : b.variants.at(variant_names[g]).meta;
                        if (options.verbose && r == 0)
                            std::cerr << (b.full() ? "model" : "layer") << " NEFF R=" << m.R << " Q=" << m.Q << " P=" << m.P << " L=" << (b.full() ? m.L : 1)
                                      << " head=" << m.head << " weights=" << m.weights << ": " << info->tensor_count << " tensors\n";
                        const int base = b.full() ? (m.head ? 8 : 5) : 0;
                        const int layer_inputs = b.full() ? kFp8Inputs : int(neff_input_names(m).size());
                        const int kv_index = b.full() ? base + 16 * m.L : layer_inputs - 1;
                        int ins = 0;
                        for (uint64_t i = 0; i < info->tensor_count; ++i)
                            if (info->tensor_array[i].usage == NRT_TENSOR_USAGE_INPUT) {
                                const auto &t = info->tensor_array[i];
                                if (std::stoul(std::string(t.name).substr(5)) == size_t(kv_index)) {
                                    const int pages = int(t.size / size_t(2 * kv_local * PAGE * dim * 2));
                                    if (pages_total && pages != pages_total)
                                        throw std::runtime_error("NEFFs disagree on the page count");
                                    pages_total = pages;
                                }
                                ++ins;
                            }
                        if (ins != (b.full() ? base + 17 * m.L : layer_inputs))
                            throw std::runtime_error("NEFF has " + std::to_string(ins) + " inputs, unexpected for its layer count");
                    }
                    auto &pr = b.rank[size_t(r)];
                    const size_t T = size_t(b.tokens());
                    pr.x[0] = alloc(r, T * size_t(hidden) * 4, "x0");
                    pr.x[1] = alloc(r, T * size_t(hidden) * 4, "x1");
                    pr.cos_sin = alloc(r, T * size_t(dim) * 4, "cos_sin");
                    pr.pos = alloc(r, T * 8, "pos");
                    pr.wslot = alloc(r, T * 8, "wslot");
                    pr.table = alloc(r, size_t(b.meta.R) * size_t(b.meta.P) * 8, "table");
                    if (b.in_graph_head()) {
                        pr.rows = alloc(r, size_t(b.meta.S) * 8, "rows");
                        pr.logits = alloc(r, size_t(b.meta.S) * size_t(vocab_shard) * 4, "logits");
                    }
                }));
            for (auto &j : jobs)
                j.get();
        };
        for (auto &b : decode) load_layer(b);
        for (auto &b : prefill) load_layer(b);
        for (auto &h : heads) {
            const auto neff = slurp(h.meta.neff);
            h.model.resize(size_t(tp));
            h.rank.resize(size_t(tp));
            for (int r = 0; r < tp; ++r) {
                check(nrt_load(neff.data(), neff.size(), r, 1, &h.model[size_t(r)]), "nrt_load head");
                h.rank[size_t(r)].rows = alloc(r, size_t(h.meta.S) * 8, "rows");
                h.rank[size_t(r)].logits = alloc(r, size_t(h.meta.S) * size_t(vocab_shard) * 4, "logits");
            }
        }
        if (!pages_total)
            throw std::runtime_error("no KV page count found");
    }

    void upload_layer(int layer) {
        char name[64];
        std::snprintf(name, sizeof name, "model-layer-%05d.safetensors", layer);
        SafeTensorFile file(options.model / name);
        struct P { const char *name; bool col; int slot; };
        const P projs[] = {{"self_attn.q_proj", true, 0}, {"self_attn.o_proj", false, 2}, {"self_attn.k_proj", true, 4},
                           {"self_attn.v_proj", true, 6}, {"mlp.gate_proj", true, 8}, {"mlp.up_proj", true, 10}, {"mlp.down_proj", false, 12}};
        const std::string base = "model.layers." + std::to_string(layer) + ".";
        const size_t H = size_t(hidden);
        std::vector<float> n1(H), n2(H);
        {
            auto a = file.floats(base + "input_layernorm.weight"), b = file.floats(base + "post_attention_layernorm.weight");
            for (size_t i = 0; i < H; ++i) { n1[i] = a[i]; n2[i] = b[i]; }
        }
        std::vector<uint8_t> zeros(kv_bytes(), 0);
        for (int r = 0; r < tp; ++r) {
            auto &lw = ranks[size_t(r)].layers[size_t(layer)];
            lw.norm1 = alloc(r, H * 4, "norm1");
            lw.norm2 = alloc(r, H * 4, "norm2");
            put(lw.norm1, n1.data(), H * 4);
            put(lw.norm2, n2.data(), H * 4);
            lw.kv = alloc(r, zeros.size(), "kv_cache");
            put(lw.kv, zeros.data(), zeros.size());
        }
        const bool compress = compressed[size_t(layer)];
        for (const auto &p : projs) {
            const bool kv_proj = p.slot == 4 || p.slot == 6;
            if (compress && !kv_proj) {
                // trellis words stay compressed: pack this rank's shard for the fused decode kernel (qtip_nki.py layout)
                const auto l = qtip::load_linear(file, layer, p.name, &config);
                const int which = p.slot == 0 ? 0 : p.slot == 2 ? 1 : p.slot == 8 ? 2 : p.slot == 10 ? 3 : 4; // q o gate up down
                for (int r = 0; r < tp; ++r) {
                    const QtipPacked packed = pack_qtip_shard(l, tp, r, p.col);
                    auto &q = ranks[size_t(r)].layers[size_t(layer)].q[which];
                    q.words = alloc(r, packed.words.size() * 2, p.name);
                    q.lut = alloc(r, packed.lut.size() * 4, p.name);
                    q.su = alloc(r, packed.su.size() * 4, p.name);
                    q.sv = alloc(r, packed.sv.size() * 4, p.name);
                    put(q.words, packed.words.data(), packed.words.size() * 2);
                    put(q.lut, packed.lut.data(), packed.lut.size() * 4);
                    put(q.su, packed.su.data(), packed.su.size() * 4);
                    put(q.sv, packed.sv.data(), packed.sv.size() * 4);
                }
                continue;
            }
            auto shards = load_projection_shards(file, layer, p.name, config, tp, p.col);
            for (int r = 0; r < tp; ++r) {
                auto &lw = ranks[size_t(r)].layers[size_t(layer)];
                const auto &s = shards[size_t(r)];
                lw.w[p.slot] = alloc(r, s.w.size(), p.name);
                lw.w[p.slot + 1] = alloc(r, s.scale.size() * 4, p.name);
                if (weights_transposed) { // NEFFs traced with WT=1 take [in][out]
                    std::vector<uint8_t> t(s.w.size());
                    for (int i = 0; i < s.rows; ++i)
                        for (int j = 0; j < s.cols; ++j)
                            t[size_t(j) * size_t(s.rows) + size_t(i)] = s.w[size_t(i) * size_t(s.cols) + size_t(j)];
                    put(lw.w[p.slot], t.data(), t.size());
                } else
                    put(lw.w[p.slot], s.w.data(), s.w.size());
                put(lw.w[p.slot + 1], s.scale.data(), s.scale.size() * 4);
            }
        }
    }

    // Which layers keep their QTIP projections compressed (options.qtip_mode), and whether the NEFF buckets can run the
    // result: a compressed layer needs a WEIGHTS=qtip layer NEFF in every bucket, an expanded one a WEIGHTS=fp8 NEFF.
    void decide_modes() {
        compressed.assign(size_t(config.layers), false);
        std::vector<bool> compressible(size_t(config.layers), false);
        std::vector<size_t> packed_bytes(size_t(config.layers), 0), expanded_bytes(size_t(config.layers), 0);
        const int H = hidden, FF = config.intermediate;
        struct P { const char *name; bool col; int out, in; };
        const P projs[] = {{"self_attn.q_proj", true, H, H}, {"self_attn.o_proj", false, H, H}, {"mlp.gate_proj", true, FF, H},
                           {"mlp.up_proj", true, FF, H}, {"mlp.down_proj", false, H, FF}};
        for (int layer = 0; layer < config.layers; ++layer) {
            char name[64];
            std::snprintf(name, sizeof name, "model-layer-%05d.safetensors", layer);
            SafeTensorFile file(options.model / name);
            bool all = true;
            for (const auto &p : projs) {
                all = all && is_qtip_projection(file, layer, p.name);
                packed_bytes[size_t(layer)] += qtip_packed_bytes(p.out, p.in, tp, p.col);
                expanded_bytes[size_t(layer)] += size_t(p.out) * size_t(p.in) / size_t(tp) + size_t(p.col ? p.out / tp : p.out) * 4;
            }
            compressible[size_t(layer)] = all;
        }
        const int n_compressible = int(std::count(compressible.begin(), compressible.end(), true));
        if (options.qtip_mode == "compressed" || options.qtip_mode == "auto")
            compressed = compressible;
        else if (options.qtip_mode != "expanded")
            throw std::runtime_error("BEHEMOTH_NEURON_QTIP must be expanded, compressed or auto (got " + options.qtip_mode + ")");
        if (options.qtip_mode == "auto") {
            // fill the HBM budget with expanded layers (from layer 0 up): every layer's fixed part (norms, FP8 k/v, KV cache),
            // the lm_head and the compressed form of every QTIP layer are committed first
            const double budget = (options.hbm_gib - options.hbm_reserve_gib) * double(1 << 30);
            double used = double(vocab_shard) * double(hidden) * (head_fp8 ? 1.0 : 2.0) + double(vocab_shard) * 4;
            for (int layer = 0; layer < config.layers; ++layer) {
                used += double(kv_bytes()) + 2.0 * hidden * 4 + 2.0 * (size_t(config.kv_heads / tp) * size_t(dim) * size_t(hidden) + size_t(config.kv_heads / tp) * size_t(dim) * 4);
                used += compressible[size_t(layer)] ? double(packed_bytes[size_t(layer)]) : double(expanded_bytes[size_t(layer)]);
            }
            for (int layer = 0; layer < config.layers; ++layer) {
                if (!compressible[size_t(layer)]) continue;
                const double extra = double(expanded_bytes[size_t(layer)]) - double(packed_bytes[size_t(layer)]);
                if (used + extra > budget) break;
                used += extra;
                compressed[size_t(layer)] = false;
            }
            if (options.verbose)
                std::cerr << "QTIP auto: planned " << used / double(1 << 30) << " GiB of a " << budget / double(1 << 30) << " GiB budget per core\n";
        }
        n_compressed = int(std::count(compressed.begin(), compressed.end(), true));
        if (options.verbose)
            std::cerr << "QTIP mode " << options.qtip_mode << ": " << n_compressed << " of " << n_compressible << " QTIP layers compressed (trellis resident), "
                      << config.layers - n_compressed << " layers expanded to FP8\n";
        const bool need_qtip = n_compressed > 0, need_fp8 = n_compressed < config.layers;
        for (const auto *group : {&decode, &prefill})
            for (const auto &b : *group) {
                if (b.full()) {
                    if (need_qtip)
                        throw std::runtime_error("whole-model NEFF buckets cannot run compressed QTIP layers (NEURON_USE_LAYERS=1 or BEHEMOTH_NEURON_QTIP=expanded)");
                    continue;
                }
                for (const auto &[need, name] : {std::pair{need_qtip, "qtip"}, std::pair{need_fp8, "fp8"}})
                    if (need && !b.variants.count(name))
                        throw std::runtime_error("layer bucket R=" + std::to_string(b.meta.R) + " Q=" + std::to_string(b.meta.Q) + " P=" + std::to_string(b.meta.P) +
                                                 " has no WEIGHTS=" + name + " layer NEFF, which BEHEMOTH_NEURON_QTIP=" + options.qtip_mode + " needs for " +
                                                 (std::string(name) == "qtip" ? "the compressed" : "the expanded") + " layers (retrace with WEIGHTS=" + name + ")");
            }
    }

    void load_weights() {
        const auto t0 = Clock::now();
        decide_modes();
        for (auto &rs : ranks) {
            rs.layers.resize(size_t(config.layers));
            for (int l = 0; l < config.layers; ++l) rs.layers[size_t(l)].compressed = compressed[size_t(l)];
        }
        if (n_compressed) // this rank's slice of a full hidden / intermediate vector (compressed graphs select it after the rotation)
            for (int r = 0; r < tp; ++r) {
                std::vector<int64_t> h(size_t(hidden / tp)), ff(size_t(config.intermediate / tp));
                for (size_t i = 0; i < h.size(); ++i) h[i] = int64_t(r) * int64_t(h.size()) + int64_t(i);
                for (size_t i = 0; i < ff.size(); ++i) ff[i] = int64_t(r) * int64_t(ff.size()) + int64_t(i);
                ranks[size_t(r)].sel_h = alloc(r, h.size() * 8, "sel_h");
                ranks[size_t(r)].sel_ff = alloc(r, ff.size() * 8, "sel_ff");
                put(ranks[size_t(r)].sel_h, h.data(), h.size() * 8);
                put(ranks[size_t(r)].sel_ff, ff.data(), ff.size() * 8);
            }
        std::atomic<int> next{0};
        std::mutex log;
        std::vector<std::thread> pool;
        std::exception_ptr failure;
        for (int i = 0; i < std::max(1, options.load_threads); ++i)
            pool.emplace_back([&] {
                for (;;) {
                    const int layer = next.fetch_add(1);
                    if (layer >= config.layers)
                        return;
                    try {
                        upload_layer(layer);
                    } catch (...) {
                        std::lock_guard lock(log);
                        failure = std::current_exception();
                        next = config.layers;
                        return;
                    }
                    if (options.verbose) {
                        std::lock_guard lock(log);
                        std::cerr << "layer " << layer << " uploaded (" << ms_since(t0) / 1e3 << " s)\n";
                    }
                }
            });
        for (auto &t : pool)
            t.join();
        if (failure)
            std::rethrow_exception(failure);
        // lm_head shards (rows padded to a multiple of TP; bf16 as stored, or E4M3 + row scale for fp8 head NEFFs) and the final norm
        const size_t H = size_t(hidden);
        std::vector<float> norm(H);
        for (size_t i = 0; i < H; ++i)
            norm[i] = final_norm[i];
        const auto head_bytes = embed_file->bytes("lm_head.weight");
        const auto head_floats = embed_file->floats("lm_head.weight");
        for (int r = 0; r < tp; ++r) {
            auto &rs = ranks[size_t(r)];
            rs.head_norm = alloc(r, H * 4, "final_norm");
            put(rs.head_norm, norm.data(), H * 4);
            const size_t first = size_t(r) * size_t(vocab_shard), last = std::min(size_t(vocab_), first + size_t(vocab_shard));
            const size_t rows = last > first ? last - first : 0;
            if (head_fp8) {
                std::vector<float> w(rows * H);
                for (size_t i = 0; i < w.size(); ++i)
                    w[i] = head_floats[first * H + i];
                const Fp8Shard s = quantize_rows(w.data(), int(rows), hidden);
                rs.head_w = alloc(r, size_t(vocab_shard) * H, "lm_head");
                rs.head_ws = alloc(r, size_t(vocab_shard) * 4, "lm_head_scale");
                zero(rs.head_w);
                zero(rs.head_ws);
                if (rows) {
                    if (head_transposed) {
                        std::vector<uint8_t> t(size_t(vocab_shard) * H);
                        for (size_t i = 0; i < rows; ++i)
                            for (size_t j = 0; j < H; ++j)
                                t[j * size_t(vocab_shard) + i] = s.w[i * H + j];
                        put(rs.head_w, t.data(), t.size());
                    } else
                        put(rs.head_w, s.w.data(), s.w.size());
                    put(rs.head_ws, s.scale.data(), s.scale.size() * 4);
                }
            } else {
                rs.head_w = alloc(r, size_t(vocab_shard) * H * 2, "lm_head");
                zero(rs.head_w);
                if (rows && head_transposed) {
                    std::vector<uint16_t> t(size_t(vocab_shard) * H);
                    const uint16_t *src = reinterpret_cast<const uint16_t *>(head_bytes.data()) + first * H;
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < H; ++j)
                            t[j * size_t(vocab_shard) + i] = src[i * H + j];
                    put(rs.head_w, t.data(), t.size() * 2);
                } else if (rows)
                    put(rs.head_w, head_bytes.data() + first * H * 2, rows * H * 2);
            }
        }
        for (const auto &lw : ranks[0].layers) {
            weight_bytes_rank += lw.bytes();
            kv_bytes_rank += lw.kv.bytes;
        }
        weight_bytes_rank += ranks[0].head_w.bytes + ranks[0].head_ws.bytes + ranks[0].head_norm.bytes + ranks[0].sel_h.bytes + ranks[0].sel_ff.bytes;
        if (const char *swap = std::getenv("NEURON_SWAP_TENSORS")) { // debugging: layer TARGET reuses layer FROM's tensors; list of 0 norm1, 1 norm2, 2 + k weight slot k
            const int from = std::atoi(std::getenv("NEURON_SWAP_FROM") ? std::getenv("NEURON_SWAP_FROM") : "30");
            const int target = std::atoi(std::getenv("NEURON_SWAP_TARGET") ? std::getenv("NEURON_SWAP_TARGET") : "31");
            const std::string list = swap;
            for (size_t at = 0; at < list.size();) {
                const size_t comma = std::min(list.find(',', at), list.size());
                const int i = std::atoi(list.substr(at, comma - at).c_str());
                for (int r = 0; r < tp; ++r) {
                    auto &dst = ranks[size_t(r)].layers[size_t(target)];
                    const auto &src = ranks[size_t(r)].layers[size_t(from)];
                    if (i == 0) dst.norm1 = src.norm1;
                    else if (i == 1) dst.norm2 = src.norm2;
                    else dst.w[size_t(i - 2)] = src.w[size_t(i - 2)];
                }
                at = comma + 1;
            }
        }
        build_sets();
        if (options.verbose)
            std::cerr << "model resident on " << tp << " NeuronCores in " << ms_since(t0) / 1e3 << " s, " << pages_total - 1 << " KV pages; per core "
                      << weight_bytes_rank / double(1 << 30) << " GiB weights + " << kv_bytes_rank / double(1 << 30) << " GiB KV ("
                      << Geometry::of(config, tp).str() << ", " << config.layers << " layers, " << n_compressed << " compressed)\n";
    }

    void build_sets() {
        // Whole-model bucket: one tensor set. Inputs: step tensors + head, then 16 tensors per layer, then the 2L KV state
        // tensors (one K/V tensor per layer); outputs: logits, then the same KV tensors (updated in place).
        auto build_full = [&](LayerBucket &b) {
            for (int r = 0; r < tp; ++r) {
                auto &pr = b.rank[size_t(r)];
                for (size_t g = 0; g < b.segs.size(); ++g) {
                    auto &seg = b.segs[g];
                    nrt_tensor_info_array_t *info = nullptr;
                    check(nrt_get_model_tensor_info(seg.model[size_t(r)], &info), "tensor info");
                    // x chains through the segments: segment g reads x[g % 2] and (a body segment) writes x[(g + 1) % 2]
                    std::vector<const Dev *> by_index = {&pr.x[g % 2], &pr.cos_sin, &pr.pos, &pr.wslot, &pr.table};
                    if (seg.meta.head) {
                        by_index.push_back(&pr.rows);
                        by_index.push_back(&ranks[size_t(r)].head_norm);
                        by_index.push_back(&ranks[size_t(r)].head_w);
                    }
                    std::vector<const Dev *> state;
                    for (int l : seg.layers) {
                        const auto &lw = ranks[size_t(r)].layers[size_t(l)];
                        by_index.push_back(&lw.norm1);
                        by_index.push_back(&lw.norm2);
                        for (const auto &w : lw.w) by_index.push_back(&w);
                        state.push_back(&lw.kv);
                    }
                    by_index.insert(by_index.end(), state.begin(), state.end());
                    nrt_tensor_set_t *in = nullptr, *out = nullptr;
                    check(nrt_allocate_tensor_set(&in), "set");
                    check(nrt_allocate_tensor_set(&out), "set");
                    size_t extra = 0;
                    for (uint64_t i = 0; i < info->tensor_count; ++i) {
                        const auto &t = info->tensor_array[i];
                        const std::string name = t.name;
                        if (t.usage == NRT_TENSOR_USAGE_INPUT) {
                            const size_t index = std::stoul(name.substr(5));
                            if (index >= by_index.size() || by_index[index]->bytes != t.size)
                                throw std::runtime_error("whole-model input " + name + " does not match the host tensor");
                            check(nrt_add_tensor_to_tensor_set(in, t.name, by_index[index]->t), "add input");
                        } else if (name == "output0") {
                            check(nrt_add_tensor_to_tensor_set(out, t.name, (seg.meta.head ? pr.logits : pr.x[(g + 1) % 2]).t), "add output");
                        } else {
                            if (extra >= state.size() || state[extra]->bytes != t.size)
                                throw std::runtime_error("whole-model output " + name + " does not match a KV tensor");
                            check(nrt_add_tensor_to_tensor_set(out, t.name, state[extra++]->t), "add kv output");
                        }
                    }
                    seg.in[size_t(r)] = in;
                    seg.out[size_t(r)] = out;
                }
            }
        };
        // the device tensor behind a layer NEFF input name (trace_model.py ORDER / ORDER_QTIP) for layer l, parity p
        auto tensor_for = [&](const std::string &name, int r, const LayerBucket::PerRank &pr, const LayerWeights &lw, int p) -> const Dev & {
            static const std::map<std::string, int> fp8_slot = {{"qf", 0}, {"qs", 1}, {"of", 2}, {"os", 3}, {"kf", 4}, {"ks", 5}, {"vf", 6},
                                                                {"vs", 7}, {"gf", 8}, {"gs", 9}, {"uf", 10}, {"us", 11}, {"df", 12}, {"ds", 13}};
            if (name == "x") return pr.x[p];
            if (name == "cos_sin") return pr.cos_sin;
            if (name == "pos") return pr.pos;
            if (name == "wslot") return pr.wslot;
            if (name == "table") return pr.table;
            if (name == "cache") return lw.kv;
            if (name == "norm1") return lw.norm1;
            if (name == "norm2") return lw.norm2;
            if (name == "sel_h") return ranks[size_t(r)].sel_h;
            if (name == "sel_ff") return ranks[size_t(r)].sel_ff;
            if (auto it = fp8_slot.find(name); it != fp8_slot.end()) return lw.w[it->second];
            if (name.size() >= 2 && std::string("qogud").find(name[0]) != std::string::npos) { // qw ql qsu qsv ... dsv
                const auto &q = lw.q[size_t(std::string("qogud").find(name[0]))];
                const std::string field = name.substr(1);
                if (field == "w") return q.words;
                if (field == "l") return q.lut;
                if (field == "su") return q.su;
                if (field == "sv") return q.sv;
            }
            throw std::runtime_error("unknown layer NEFF input " + name);
        };
        auto build = [&](LayerBucket &b) {
            if (b.full()) {
                build_full(b);
                return;
            }
            for (int r = 0; r < tp; ++r) {
                auto &pr = b.rank[size_t(r)];
                pr.in.resize(ranks[size_t(r)].layers.size());
                pr.out.resize(pr.in.size());
                for (size_t l = 0; l < pr.in.size(); ++l) {
                    const auto &lw = ranks[size_t(r)].layers[l];
                    const auto &variant = b.variants.at(lw.compressed ? "qtip" : "fp8");
                    const std::vector<std::string> names = neff_input_names(variant.meta);
                    nrt_tensor_info_array_t *info = nullptr;
                    check(nrt_get_model_tensor_info(variant.model[size_t(r)], &info), "tensor info");
                    for (int p = 0; p < 2; ++p) {
                        nrt_tensor_set_t *in = nullptr, *out = nullptr;
                        check(nrt_allocate_tensor_set(&in), "set");
                        check(nrt_allocate_tensor_set(&out), "set");
                        int extra = 0;
                        for (uint64_t i = 0; i < info->tensor_count; ++i) {
                            const auto &t = info->tensor_array[i];
                            const std::string name = t.name;
                            if (t.usage == NRT_TENSOR_USAGE_INPUT) {
                                const size_t index = std::stoul(name.substr(5));
                                if (index >= names.size())
                                    throw std::runtime_error("layer NEFF input " + name + " beyond its meta order");
                                const Dev &dev = tensor_for(names[index], r, pr, lw, p);
                                if (dev.bytes != t.size)
                                    throw std::runtime_error("input " + name + " (" + names[index] + ") expects " + std::to_string(t.size) + " B, host has " +
                                                             std::to_string(dev.bytes) + " (layer " + std::to_string(l) + ")");
                                check(nrt_add_tensor_to_tensor_set(in, t.name, dev.t), "add input");
                            } else if (name == "output0") {
                                check(nrt_add_tensor_to_tensor_set(out, t.name, pr.x[1 - p].t), "add output");
                            } else { // the updated KV state, written back into the same device buffer
                                const Dev &kv = lw.kv;
                                if (extra++ != 0 || kv.bytes != t.size)
                                    throw std::runtime_error("unexpected extra output " + name);
                                check(nrt_add_tensor_to_tensor_set(out, t.name, kv.t), "add kv output");
                            }
                        }
                        pr.in[l][size_t(p)] = in;
                        pr.out[l][size_t(p)] = out;
                    }
                }
            }
        };
        for (auto &b : decode) build(b);
        for (auto &b : prefill) build(b);
        // head sets: input0 = the residual buffer of each layer bucket with the same token count, rows, norm, w (, scale) -> logits.
        for (auto &h : heads)
            for (int r = 0; r < tp; ++r) {
                auto &hr = h.rank[size_t(r)];
                for (int p = 0; p < 2; ++p) {
                    check(nrt_allocate_tensor_set(&hr.out[size_t(p)]), "set");
                    check(nrt_add_tensor_to_tensor_set(hr.out[size_t(p)], "output0", hr.logits.t), "head out");
                }
                for (const auto *group : {&decode, &prefill})
                    for (const auto &b : *group) {
                        if (b.in_graph_head() || b.tokens() != h.meta.T)
                            continue;
                        auto &sets = hr.in[&b];
                        for (int p = 0; p < 2; ++p) {
                            check(nrt_allocate_tensor_set(&sets[size_t(p)]), "set");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input0", b.rank[size_t(r)].x[p].t), "head in");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input1", hr.rows.t), "head in");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input2", ranks[size_t(r)].head_norm.t), "head in");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input3", ranks[size_t(r)].head_w.t), "head in");
                            if (h.meta.fp8)
                                check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input4", ranks[size_t(r)].head_ws.t), "head in");
                        }
                    }
            }
    }

    // ---- per-step host data for one launch of a layer bucket ----------------------------------------------------
    struct Launch {
        const LayerBucket *bucket = nullptr;
        std::vector<float> x, cos_sin;
        std::vector<int64_t> pos, wslot, table;
        std::vector<int> head_rows;   // rows of x to produce logits for
        std::vector<int> destination; // output row (in `logits`) of each head row
    };

    void embed_row(int token, float *out) const {
        if (token < 0 || token >= vocab_)
            throw std::runtime_error("token outside the vocabulary");
        for (size_t i = 0; i < size_t(hidden); ++i)
            out[i] = embed[size_t(token) * size_t(hidden) + i];
    }
    void rope_row(int position, float *out) const { // [cos x dim/2, sin x dim/2] from fp64 angles, any position
        for (int d = 0; d < dim / 2; ++d) {
            const double angle = double(position) * double(inv_freq[size_t(d)]);
            out[d] = float(std::cos(angle));
            out[dim / 2 + d] = float(std::sin(angle));
        }
    }

    void fill_row(Launch &L, int row, int token, int position, int page_id) {
        embed_row(token, L.x.data() + size_t(row) * size_t(hidden));
        rope_row(position, L.cos_sin.data() + size_t(row) * size_t(dim));
        L.pos[size_t(row)] = position;
        L.wslot[size_t(row)] = int64_t(page_id) * PAGE + position % PAGE;
    }

    // Merge the ranks' vocab slices of the sampled rows into dest.
    void gather_logits(const Launch &L, const std::vector<std::vector<float>> &out, float *dest) const {
        for (size_t i = 0; i < L.head_rows.size(); ++i) {
            float *row = dest + size_t(L.destination[i]) * size_t(vocab_);
            for (int r = 0; r < tp; ++r) {
                const size_t first = size_t(r) * size_t(vocab_shard);
                if (first >= size_t(vocab_))
                    continue;
                const size_t count = std::min(size_t(vocab_) - first, size_t(vocab_shard));
                std::copy_n(out[size_t(r)].data() + i * size_t(vocab_shard), count, row + first);
            }
        }
    }

    void execute(Launch &L, float *dest) {
        const auto &b = *L.bucket;
        if (b.in_graph_head()) {
            if (L.head_rows.size() > size_t(b.meta.S))
                throw std::runtime_error("more sampled rows than the whole-model NEFF produces");
            std::vector<int64_t> rows(size_t(b.meta.S), 0);
            for (size_t i = 0; i < L.head_rows.size(); ++i)
                rows[i] = L.head_rows[i];
            std::vector<std::vector<float>> out(static_cast<size_t>(tp));
            std::vector<std::future<void>> jobs;
            for (int r = 0; r < tp; ++r)
                jobs.push_back(workers[size_t(r)]->submit([&, r] {
                    auto &pr = const_cast<LayerBucket &>(b).rank[size_t(r)];
                    const auto t_start = Clock::now();
                    put(pr.x[0], L.x.data(), L.x.size() * 4);
                    put(pr.cos_sin, L.cos_sin.data(), L.cos_sin.size() * 4);
                    put(pr.pos, L.pos.data(), L.pos.size() * 8);
                    put(pr.wslot, L.wslot.data(), L.wslot.size() * 8);
                    put(pr.table, L.table.data(), L.table.size() * 8);
                    put(pr.rows, rows.data(), rows.size() * 8);
                    const auto t_uploaded = Clock::now();
                    for (size_t g = 0; g < b.segs.size(); ++g) {
                        const auto &seg = b.segs[g];
                        check(nrt_execute(seg.model[size_t(r)], seg.in[size_t(r)], seg.out[size_t(r)]), "whole-model execute");
                        if (r == 0 && !seg.meta.head)
                            dump_x(b, seg.first_layer + seg.meta.L, pr.x[(g + 1) % 2]);
                    }
                    const auto t_executed = Clock::now();
                    out[size_t(r)].resize(size_t(b.meta.S) * size_t(vocab_shard));
                    check(nrt_tensor_read(pr.logits.t, out[size_t(r)].data(), 0, out[size_t(r)].size() * 4), "read logits");
                    if (r == 0) {
                        phase_upload_ms += std::chrono::duration<double, std::milli>(t_uploaded - t_start).count();
                        phase_layers_ms += std::chrono::duration<double, std::milli>(t_executed - t_uploaded).count();
                        phase_head_ms += ms_since(t_executed);
                    }
                }));
            for (auto &j : jobs)
                j.get();
            gather_logits(L, out, dest);
            return;
        }
        const HeadBucket *head = nullptr;
        if (!L.head_rows.empty())
            for (const auto &h : heads)
                if (h.meta.T == b.tokens() && h.meta.S >= int(L.head_rows.size())) {
                    head = &h;
                    break;
                }
        if (!L.head_rows.empty() && !head)
            throw std::runtime_error("no head NEFF for " + std::to_string(L.head_rows.size()) + " rows of " + std::to_string(b.tokens()));
        int layers = config.layers;
        if (!b.full() && std::getenv("NEURON_LAYER_ORDER")) { // debugging order: its length decides the final parity
            layers = 0;
            std::string text = std::getenv("NEURON_LAYER_ORDER");
            for (size_t at = 0; at < text.size();) {
                const size_t comma = std::min(text.find(',', at), text.size()), dash = text.find('-', at);
                layers += std::stoi(text.substr(dash + 1, comma - dash - 1)) - std::stoi(text.substr(at, dash - at)) + 1;
                at = comma + 1;
            }
        }
        const int final_parity = (b.full() ? int(b.segs.size()) : layers) % 2;
        std::vector<int64_t> rows(head ? size_t(head->meta.S) : 0, 0);
        for (size_t i = 0; i < L.head_rows.size(); ++i)
            rows[i] = L.head_rows[i];
        std::vector<std::vector<float>> out(static_cast<size_t>(tp));
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                auto &pr = const_cast<LayerBucket &>(b).rank[size_t(r)];
                const auto t_start = Clock::now();
                put(pr.x[0], L.x.data(), L.x.size() * 4);
                put(pr.cos_sin, L.cos_sin.data(), L.cos_sin.size() * 4);
                put(pr.pos, L.pos.data(), L.pos.size() * 8);
                put(pr.wslot, L.wslot.data(), L.wslot.size() * 8);
                put(pr.table, L.table.data(), L.table.size() * 8);
                const auto t_uploaded = Clock::now();
                if (b.full()) {
                    for (size_t g = 0; g < b.segs.size(); ++g) {
                        const auto &seg = b.segs[g];
                        check(nrt_execute(seg.model[size_t(r)], seg.in[size_t(r)], seg.out[size_t(r)]), "segment execute");
                        if (r == 0)
                            dump_x(b, seg.first_layer + seg.meta.L, pr.x[(g + 1) % 2]);
                    }
                } else {
                    std::vector<int> order(static_cast<size_t>(layers));
                    for (int l = 0; l < layers; ++l) order[size_t(l)] = l;
                    if (const char *spec = std::getenv("NEURON_LAYER_ORDER")) { // debugging: run these layer ranges, "0-15,15-30"
                        order.clear();
                        std::string text = spec;
                        for (size_t at = 0; at < text.size();) {
                            const size_t comma = std::min(text.find(',', at), text.size());
                            const std::string part = text.substr(at, comma - at);
                            const size_t dash = part.find('-');
                            for (int l = std::stoi(part.substr(0, dash)); l <= std::stoi(part.substr(dash + 1)); ++l) order.push_back(l);
                            at = comma + 1;
                        }
                    }
                    for (size_t step = 0; step < order.size(); ++step) {
                        const size_t l = size_t(order[step]), parity = step % 2;
                        check(nrt_execute(b.model_for(compressed[l])[size_t(r)], pr.in[l][parity], pr.out[l][parity]), "layer execute");
                        if (r == 0)
                            dump_x(b, int(step) + 1, pr.x[(step + 1) % 2]);
                    }
                }
                const auto t_layers = Clock::now();
                if (r == 0) {
                    phase_upload_ms += std::chrono::duration<double, std::milli>(t_uploaded - t_start).count();
                    phase_layers_ms += std::chrono::duration<double, std::milli>(t_layers - t_uploaded).count();
                }
                if (head) {
                    auto &hr = const_cast<HeadBucket *>(head)->rank[size_t(r)];
                    put(hr.rows, rows.data(), rows.size() * 8);
                    check(nrt_execute(head->model[size_t(r)], hr.in.at(&b)[size_t(final_parity)], hr.out[size_t(final_parity)]), "head execute");
                    out[size_t(r)].resize(size_t(head->meta.S) * size_t(vocab_shard));
                    check(nrt_tensor_read(hr.logits.t, out[size_t(r)].data(), 0, out[size_t(r)].size() * 4), "read logits");
                    if (r == 0)
                        phase_head_ms += ms_since(t_layers);
                }
            }));
        for (auto &j : jobs)
            j.get();
        if (head)
            gather_logits(L, out, dest);
    }

    // Debugging: NEURON_DUMP_X=prefix writes rank 0's residual stream (f32 [tokens][hidden]) after layers 16 and 32 (when the
    // execution passes there) to prefix.x16 / prefix.x32, from either execution path.
    void dump_x(const LayerBucket &b, int layers_done, const Dev &x) {
        const char *prefix = std::getenv("NEURON_DUMP_X");
        if (!prefix)
            return;
        bool wanted = layers_done == 16 || layers_done == 32 || layers_done == 31;
        if (const char *list = std::getenv("NEURON_DUMP_LAYERS")) { // "1,2,8": also after these many layers
            wanted = false;
            for (std::string text = list; !text.empty();) {
                const size_t comma = std::min(text.find(','), text.size());
                wanted = wanted || std::stoi(text.substr(0, comma)) == layers_done;
                text = comma < text.size() ? text.substr(comma + 1) : "";
            }
        }
        if (!wanted)
            return;
        std::vector<float> v(size_t(b.tokens()) * size_t(hidden));
        check(nrt_tensor_read(x.t, v.data(), 0, v.size() * 4), "read x");
        std::ofstream(std::string(prefix) + ".x" + std::to_string(layers_done), std::ios::binary).write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * 4));
    }

    std::span<const float> forward(const InferenceBatch &batch) {
        const auto t0 = Clock::now();
        phase_upload_ms = phase_layers_ms = phase_head_ms = 0;
        const auto &lay = batch.layout;
        const int scratch = pages_total - 1;
        logits.resize(batch.sample_rows.size() * size_t(vocab_)); // every sampled row is overwritten by execute()
        // which tokens sample, and which logits row each owns (several per request when drafts are verified)
        std::vector<int> sample_of(size_t(lay.queries()), -1);
        for (size_t s = 0; s < batch.sample_rows.size(); ++s)
            sample_of[size_t(batch.sample_rows[s])] = int(s);
        auto pages_needed = [&](int req) { return (lay.lengths[size_t(req)] + PAGE - 1) / PAGE; };
        auto table_row = [&](Launch &L, int slot, int req) {
            const auto &b = *L.bucket;
            for (int j = 0; j < b.meta.P; ++j) {
                const int page = j < pages_needed(req) ? lay.page_table[size_t(req) * size_t(lay.table_stride) + size_t(j)] : scratch;
                L.table[size_t(slot) * size_t(b.meta.P) + size_t(j)] = page < 0 ? scratch : page;
            }
        };
        auto fresh = [&](LayerBucket &b) {
            Launch L;
            L.bucket = &b;
            const size_t T = size_t(b.tokens());
            L.x.assign(T * size_t(hidden), 0.f);
            L.cos_sin.assign(T * size_t(dim), 0.f);
            L.pos.assign(T, 0);
            L.wslot.assign(T, int64_t(scratch) * PAGE);
            L.table.assign(size_t(b.meta.R) * size_t(b.meta.P), scratch);
            for (size_t t = 0; t < T; ++t)
                rope_row(0, L.cos_sin.data() + t * size_t(dim));
            return L;
        };
        // Bucket choice: the cheapest (rows x table pages, or queries x table pages) NEFF that covers the work; a
        // whole-model bucket wins a tie with the per-layer one.
        auto cost = [](const LayerBucket &b) { return double(b.tokens()) * double(b.meta.P) + (b.full() ? 0. : .5); };
        auto pick = [&](std::vector<LayerBucket> &set, int need_rows, int need_queries, int pages) -> LayerBucket * {
            LayerBucket *best = nullptr;
            for (auto &b : set)
                if (b.meta.R >= need_rows && b.meta.Q >= need_queries && b.meta.P >= pages && (!best || cost(b) < cost(*best)))
                    best = &b;
            return best;
        };
        int launches = 0, max_rows = 0, max_queries = 0;
        for (const auto &b : decode) max_rows = std::max(max_rows, b.meta.R);
        for (const auto &b : prefill) max_queries = std::max(max_queries, b.meta.Q);
        // Rows of the decode launches: one-token requests, and short continuations (a decode token followed by draft
        // tokens to verify, or the tail of a chunked prompt) as one row per token. The graph scatters every row's K/V
        // before the attention, so a request's later rows attend to its earlier rows of the same launch.
        struct Row { int req, token_index; };
        std::vector<Row> rows;
        std::vector<int> multi;
        for (int req = 0; req < lay.requests(); ++req) {
            const int first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
            if (count == 1 || (count <= std::min(max_rows, kRowPathTokens) && lay.positions[size_t(first)] > 0))
                for (int i = 0; i < count; ++i) rows.push_back({req, first + i});
            else
                multi.push_back(req);
        }
        // all rows in one launch when a bucket fits, else in groups of the largest R
        for (size_t at = 0; at < rows.size();) {
            size_t n = rows.size() - at;
            auto pages_of = [&](size_t count) {
                int pages = 1;
                for (size_t i = 0; i < count; ++i) pages = std::max(pages, pages_needed(rows[at + i].req));
                return pages;
            };
            LayerBucket *bucket = pick(decode, int(n), 1, pages_of(n));
            if (!bucket) {
                n = std::min<size_t>(n, size_t(max_rows));
                bucket = pick(decode, int(n), 1, pages_of(n));
                if (!bucket)
                    throw std::runtime_error("no decode NEFF bucket large enough for " + std::to_string(pages_of(n)) + " pages");
            }
            auto L = fresh(*bucket);
            for (size_t i = 0; i < n; ++i) {
                const int req = rows[at + i].req, token_index = rows[at + i].token_index;
                const int position = lay.positions[size_t(token_index)];
                const int page = lay.page_table[size_t(req) * size_t(lay.table_stride) + size_t(position / PAGE)];
                fill_row(L, int(i), batch.tokens[size_t(token_index)], position, page);
                table_row(L, int(i), req);
                if (sample_of[size_t(token_index)] >= 0) {
                    L.head_rows.push_back(int(i));
                    L.destination.push_back(sample_of[size_t(token_index)]);
                }
            }
            execute(L, logits.data());
            ++launches;
            at += n;
        }
        // prefill chunks: one request per launch, the smallest query bucket that covers what is left of the prompt
        for (int req : multi) {
            const int first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
            for (int done = 0; done < count;) {
                const int pages = pages_needed(req), left = count - done;
                LayerBucket *bucket = pick(prefill, 1, std::min(left, max_queries), pages);
                if (!bucket)
                    throw std::runtime_error("no prefill NEFF bucket large enough for " + std::to_string(pages) + " pages");
                const int take = std::min(left, bucket->meta.Q);
                auto L = fresh(*bucket);
                for (int i = 0; i < take; ++i) {
                    const int token_index = first + done + i, position = lay.positions[size_t(token_index)];
                    const int page = lay.page_table[size_t(req) * size_t(lay.table_stride) + size_t(position / PAGE)];
                    fill_row(L, i, batch.tokens[size_t(token_index)], position, page);
                }
                table_row(L, 0, req);
                if (sample_of[size_t(first + done + take - 1)] >= 0) {
                    L.head_rows.push_back(take - 1);
                    L.destination.push_back(sample_of[size_t(first + done + take - 1)]);
                }
                execute(L, logits.data());
                ++launches;
                done += take;
            }
        }
        if (const char *dump = std::getenv("NEURON_LOGITS_DUMP")) { // append the sampled rows (f32 [n][vocab]) for offline comparison
            std::ofstream f(dump, std::ios::binary | std::ios::app);
            f.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * 4));
            std::ofstream ids(std::string(dump) + ".ids", std::ios::app);
            for (auto id : batch.sample_ids)
                ids << id << '\n';
        }
        profile = {{"launches", launches}, {"forward_ms", ms_since(t0)}, {"upload_ms", phase_upload_ms}, {"layers_ms", phase_layers_ms}, {"head_ms", phase_head_ms}, {"tokens", batch.tokens.size()}};
        return logits;
    }
};

NeuronModel::NeuronModel(const ModelOptions &o) : impl_(std::make_unique<Impl>(o)) {}
NeuronModel::~NeuronModel() = default;
int NeuronModel::cache_pages() const { return impl_->pages_total - 1; }
int NeuronModel::vocab() const { return impl_->vocab_; }
int NeuronModel::max_decode_rows() const {
    int r = 0;
    for (const auto &b : impl_->decode) r = std::max(r, b.meta.R);
    return r;
}
int NeuronModel::decode_bucket_pages(int pages) const {
    int best = 0;
    for (const auto &b : impl_->decode)
        if (b.meta.P >= pages && (!best || b.meta.P < best)) best = b.meta.P;
    return best;
}
const std::vector<int> &NeuronModel::eos() const { return impl_->config.eos; }
int NeuronModel::max_context() const {
    int p = 0;
    for (const auto &b : impl_->decode) p = std::max(p, b.meta.P);
    return std::min(p * PAGE, impl_->config.max_context);
}
std::span<const float> NeuronModel::forward(const InferenceBatch &b) { return impl_->forward(b); }
nlohmann::json NeuronModel::last_profile() const { return impl_->profile; }
} // namespace neuron
