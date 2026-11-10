#include "neuron_model.hpp"
#include "qtip_expand.hpp"
#include "behemoth/checkpoint.hpp"
#include "behemoth/worker.hpp"
#include <cstdint>
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_version.h> // nrt_get_version (BEHEMOTH_NEURON_LOG_VERSIONS=1 provenance log)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
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
// Streamed loading (deploy/ecs-neuron/backend/entrypoint.sh STREAM_LOAD=1): layer files arrive while the server is
// already loading; BEHEMOTH_NEURON_WAIT_READY=1 makes every layer open wait for "<file>.ready" (fail on "<file>.failed").
void wait_ready(const std::filesystem::path &file) {
    if (!std::getenv("BEHEMOTH_NEURON_WAIT_READY"))
        return;
    const char *t = std::getenv("BEHEMOTH_NEURON_WAIT_TIMEOUT_S");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(t ? std::atoi(t) : 1800);
    const auto ready = file.string() + ".ready", failed = file.string() + ".failed";
    while (!std::filesystem::exists(ready)) {
        if (std::filesystem::exists(failed))
            throw std::runtime_error("download failed: " + file.string());
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("timed out waiting for " + ready);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

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
    bool argmax = false;                          // kind "head": outputs (per-row max, argmax int32) instead of logits (device greedy)
    bool kv8 = false;                             // kind "layer": FP8 KV cache (E4M3 bytes + f32 scale per K|V, token, head), trace_model.py KV=fp8
    bool wt = false;                              // kind "layer": FP8 projections stored transposed [in][out] (trace_model.py WT=1)
    bool mlp_tiled = false;                       // kind "layer": gate / up / down in nki_mlp.tile_weight's layout (trace_model.py MLP=nki)
    bool shared_table = false;                    // kind "layer": every row attends over table row 0 (trace_model.py ATTN_DEC_SHARED=1): only for
                                                  // launches whose rows all belong to ONE request (a verify step); never picked otherwise
    std::string weights = "fp8";                  // kind "layer": "fp8" (seven E4M3 shards) or "qtip" (q/o/gate/up/down trellis resident)
    std::vector<std::string> order;               // kind "layer": input names (trace_model.py ORDER / ORDER_QTIP), "cache" is aliased
    Geometry geometry;
    std::filesystem::path neff;
    std::string build_compiler, build_libnrt;   // meta.json "build" (trace_model.py build_info): neuronx-cc / libnrt versions the bucket was
                                                  // compiled with; provenance only, logged with BEHEMOTH_NEURON_LOG_VERSIONS=1, never enforced
};

// meta.json "build" is optional provenance; any other unknown key is ignored (nlohmann value()/at() on known keys only).
void parse_build(const nlohmann::json &j, Meta &m) {
    if (!j.contains("build") || !j.at("build").is_object())
        return;
    const auto &b = j.at("build");
    m.build_compiler = b.value("neuronx_cc_bin", b.value("neuronx_cc", std::string()));
    m.build_libnrt = b.value("libnrt", std::string());
}

// BEHEMOTH_NEURON_LOG_VERSIONS=1: log the libnrt version the engine loaded and the compiler / libnrt versions recorded in the buckets'
// meta.json, with a WARNING (stderr, never an error) when the buckets mix compiler versions or were compiled against another libnrt.
// Off by default: nothing is printed and nothing changes.
void log_versions(const std::vector<Meta> &metas) {
    if (!std::getenv("BEHEMOTH_NEURON_LOG_VERSIONS"))
        return;
    std::string loaded;
    nrt_version_t v{};
    if (nrt_get_version(&v, sizeof v) == NRT_SUCCESS)
        loaded = std::to_string(v.rt_major) + "." + std::to_string(v.rt_minor) + "." + std::to_string(v.rt_patch) + "." + std::to_string(v.rt_maintenance);
    std::cerr << "Neuron versions: libnrt loaded " << (loaded.empty() ? std::string("unknown (nrt_get_version failed)") : loaded) << "\n";
    std::map<std::string, int> compilers, libnrts;
    int without = 0;
    for (const auto &m : metas) {
        if (m.build_compiler.empty() && m.build_libnrt.empty()) { ++without; continue; }
        ++compilers[m.build_compiler.empty() ? "unknown" : m.build_compiler];
        ++libnrts[m.build_libnrt.empty() ? "unknown" : m.build_libnrt];
    }
    for (const auto &[c, n] : compilers) std::cerr << "Neuron versions: " << n << " bucket(s) compiled with neuronx-cc " << c << "\n";
    if (without) std::cerr << "Neuron versions: " << without << " bucket(s) without build info in meta.json (older trace_model.py)\n";
    if (compilers.size() > 1) std::cerr << "WARNING: the NEFF buckets mix neuronx-cc versions (see above); rebuild the set with one compiler\n";
    for (const auto &[l, n] : libnrts) // the recorded version is the dpkg version string ("2.33.10.0-hash"), the loaded one is numeric
        if (l != "unknown" && !loaded.empty() && l.rfind(loaded, 0) != 0)
            std::cerr << "WARNING: " << n << " bucket(s) were compiled on a host with aws-neuronx-runtime-lib " << l << ", this process loaded libnrt "
                      << loaded << " (ok when the compiler matches; see docs/neuron-build-pinning.md)\n";
}

// Layer NEFF inputs of the first generator version (no "order" in meta.json): the FP8 layer.
const std::vector<std::string> kFp8Order = {"x", "cos_sin", "pos", "wslot", "table", "cache", "norm1", "norm2", "qf", "qs", "of", "os",
                                            "kf", "ks", "vf", "vs", "gf", "gs", "uf", "us", "df", "ds"};
// NEFF input order: the meta order with the aliased KV state moved to the end (torch_neuronx puts module state last).
std::vector<std::string> neff_input_names(const Meta &m) {
    std::vector<std::string> names;
    bool scale = false;
    for (const auto &n : m.order.empty() ? kFp8Order : m.order) {
        if (n == "cscale") scale = true;
        if (n != "cache" && n != "cscale") names.push_back(n);
    }
    names.push_back("cache");
    if (scale) names.push_back("cscale");
    return names;
}
constexpr int kFp8Inputs = 22;
// Continuations of up to this many tokens run as rows of a decode bucket instead of a prefill chunk.
constexpr int kRowPathTokens = 8;

struct LayerWeights {
    bool compressed = false;     // QTIP q/o/gate/up/down resident as trellis words (WEIGHTS=qtip NEFF) instead of FP8 shards
    Dev norm1, norm2, w[14], kv, ks; // ks: FP8 KV scales [pages, 2, kv_heads, 16] f32 (kv8 only); w: qf qs of os kf ks vf vs gf gs uf us df ds; kv: [pages, 2 (K, V), kv_heads, 16, dim] bf16
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
    bool argmax = false;
    std::vector<nrt_model_t *> model;
    struct PerRank {
        Dev rows, logits, amx;
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
    // Prepared shards (BEHEMOTH_NEURON_PREPARED_DIR / BEHEMOTH_NEURON_WRITE_PREPARED): every layer's per-rank device bytes as uploaded
    // (transposed, tiled, row scales, norms) in one file per layer, so a load is read + copy with no expansion pass.
    std::filesystem::path prepared_dir, write_prepared_dir;
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
    double phase_head_submit_ms = 0, phase_head_read_ms = 0, phase_gather_ms = 0;
    std::vector<int> device_tokens;             // destination row -> token (device argmax path; -1: not produced)
    bool used_device_argmax = false;
    size_t weight_bytes_rank = 0, kv_bytes_rank = 0;                   // resident per NeuronCore, for the load report
    bool kv8 = false, kv8_seen = false;                                // FP8 KV cache: cache bytes + a scale tensor per layer (meta.kv)
    bool weights_transposed = false, wt_seen = false;
    bool mlp_tiled = false, mlp_seen = false;                          // FP8 NEFFs take tiled gate / up / down (meta.mlp, all alike)                  // FP8 NEFFs take [in][out] projections (meta.wt, all alike)
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
        kv_local = std::max(1, config.kv_heads / tp); // tp > kv_heads: one replicated kv head per rank
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
        inv_freq = rope_inv_freq(config);
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
            parse_build(j, m);
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
                m.kv8 = j.value("kv", "bf16") == "fp8";
                if (kv8_seen && kv8 != m.kv8)
                    throw std::runtime_error("layer NEFFs mix FP8 and bf16 KV caches");
                kv8 = m.kv8;
                kv8_seen = true;
                m.mlp_tiled = j.value("mlp", std::string("plain")) == "tiled";
                m.shared_table = j.value("shared_table", false);
                if (m.weights == "fp8") {
                    if (mlp_seen && mlp_tiled != m.mlp_tiled)
                        throw std::runtime_error("FP8 layer NEFFs in " + options.neff_dir.string() + " mix tiled (MLP=nki) and plain MLP weights");
                    mlp_tiled = m.mlp_tiled;
                    mlp_seen = true;
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
                    if (!b.full() && b.meta.R == m.R && b.meta.Q == m.Q && b.meta.P == m.P && b.meta.shared_table == m.shared_table) same = &b;
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
                m.argmax = j.value("argmax", false);
                m.wt = j.value("wt", false);
                accept(m);
                // lm_head is uploaded in ONE format and bound to every head bucket, so fp8/wt must agree
                // across ALL heads regardless of the argmax flag (a mixed fp8-argmax/bf16-logits set binds
                // wrong-size weights and fails at execute).
                if (!heads.empty() && (heads.front().meta.fp8 != m.fp8 || heads.front().meta.wt != m.wt))
                    throw std::runtime_error("head NEFFs mix bf16 / fp8 or transposed / plain lm_head formats");
                if (heads.empty()) { // one upload format: fixed by the first head enumerated
                    head_fp8 = m.fp8;
                    head_transposed = m.wt;
                }
                HeadBucket b;
                b.meta = m;
                b.argmax = m.argmax;
                heads.push_back(std::move(b));
            }
        }
        for (const auto &h : heads) // an argmax head without a same-shape logits head leaves non-greedy steps unservable
            if (h.argmax && !std::any_of(heads.begin(), heads.end(), [&](const HeadBucket &g) {
                    return !g.argmax && g.meta.T == h.meta.T;
                }))
                throw std::runtime_error("argmax head NEFF for T=" + std::to_string(h.meta.T) +
                                         " has no logits-head fallback (non-greedy steps need one); stage both formats");
        if (!tp)
            throw std::runtime_error("no NEFF buckets (meta.json + graph.neff directories) in " + options.neff_dir.string());
        if (std::getenv("BEHEMOTH_NEURON_LOG_VERSIONS")) { // provenance log (off by default), before any device work
            std::vector<Meta> all(segments);
            for (const auto &b : decode) for (const auto &[n, v] : b.variants) all.push_back(v.meta);
            for (const auto &b : prefill) for (const auto &[n, v] : b.variants) all.push_back(v.meta);
            for (const auto &h : heads) all.push_back(h.meta);
            log_versions(all);
        }
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

    size_t kv_bytes() const { return size_t(pages_total) * 2 * size_t(kv_local) * PAGE * size_t(dim) * (kv8 ? 1 : 2); }
    size_t kv_scale_bytes() const { return size_t(pages_total) * 2 * size_t(kv_local) * PAGE * 4; }
    // Prefix cache: one host page holds this 16-token page's K/V (and FP8 scales) of every layer of every rank,
    // laid out [rank][layer][K/V bytes | scales]. Runs of consecutive device pages move in one transfer per layer and tensor.
    size_t kv_page_stride() const { return 2 * size_t(kv_local) * PAGE * size_t(dim) * (kv8 ? 1 : 2); }
    size_t kv_scale_page_stride() const { return kv8 ? 2 * size_t(kv_local) * PAGE * 4 : 0; }
    size_t kv_page_rank_bytes() const { return size_t(config.layers) * (kv_page_stride() + kv_scale_page_stride()); }
    void kv_copy(std::span<const int> device, std::span<const int> host, std::byte *arena, bool save) {
        if (device.size() != host.size())
            throw std::invalid_argument("kv copy: page lists differ in length");
        for (int p : device)
            if (p < 0 || p >= pages_total - 1)
                throw std::invalid_argument("kv copy: page outside the pool");
        const size_t rank_bytes = kv_page_rank_bytes(), page_bytes = rank_bytes * size_t(tp);
        std::vector<std::pair<size_t, size_t>> runs; // [begin, end) into device: consecutive page ids
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
                for (int l = 0; l < config.layers; ++l) {
                    const auto &lw = ranks[size_t(r)].layers[size_t(l)];
                    const size_t base = size_t(r) * rank_bytes + size_t(l) * (kv_page_stride() + kv_scale_page_stride());
                    for (int which = 0; which < (kv8 ? 2 : 1); ++which) {
                        const Dev &t = which ? lw.ks : lw.kv;
                        const size_t stride = which ? kv_scale_page_stride() : kv_page_stride(), sub = which ? kv_page_stride() : 0;
                        for (auto [a, b] : runs) {
                            const size_t n = b - a;
                            buf.resize(n * stride);
                            if (save) {
                                check(nrt_tensor_read(t.t, buf.data(), size_t(device[a]) * stride, n * stride), "kv save");
                                for (size_t k = 0; k < n; ++k)
                                    std::memcpy(arena + size_t(host[a + k]) * page_bytes + base + sub, buf.data() + k * stride, stride);
                            } else {
                                for (size_t k = 0; k < n; ++k)
                                    std::memcpy(buf.data() + k * stride, arena + size_t(host[a + k]) * page_bytes + base + sub, stride);
                                check(nrt_tensor_write(t.t, buf.data(), size_t(device[a]) * stride, n * stride), "kv restore");
                            }
                        }
                    }
                }
            }));
        for (auto &j : jobs)
            j.get();
    }

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
                        const int kv_index = b.full() ? base + 16 * m.L : layer_inputs - (m.kv8 ? 2 : 1);
                        int ins = 0;
                        for (uint64_t i = 0; i < info->tensor_count; ++i)
                            if (info->tensor_array[i].usage == NRT_TENSOR_USAGE_INPUT) {
                                const auto &t = info->tensor_array[i];
                                if (std::stoul(std::string(t.name).substr(5)) == size_t(kv_index)) {
                                    const int pages = int(t.size / size_t(2 * kv_local * PAGE * dim * (m.kv8 ? 1 : 2)));
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
                h.rank[size_t(r)].logits = alloc(r, size_t(h.meta.S) * (h.argmax ? 1 : size_t(vocab_shard)) * 4, "logits");
                if (h.argmax)
                    h.rank[size_t(r)].amx = alloc(r, size_t(h.meta.S) * 4, "amx");
            }
        }
        if (!pages_total)
            throw std::runtime_error("no KV page count found");
    }

    void upload_layer(int layer) {
        char name[64];
        std::snprintf(name, sizeof name, "model-layer-%05d.safetensors", layer + layer_offset());
        std::filesystem::path src = options.model / name;
        if (!prepared_dir.empty())
            src = prepared_dir / prepared_name(layer);
        wait_ready(src);
        if (!prepared_dir.empty())
            upload_prepared(layer, src);
        else
            upload_layer_file(layer, src);
        // streamed loading: tell the downloader this file is consumed, and drop the staged copy if asked (tmpfs staging)
        if (std::getenv("BEHEMOTH_NEURON_WAIT_READY")) {
            std::ofstream(src.string() + ".done").put('\n');
            if (std::getenv("BEHEMOTH_NEURON_UNLINK_LAYERS")) {
                std::error_code ec;
                const auto target = std::filesystem::canonical(src, ec);
                if (!ec)
                    std::filesystem::remove(target, ec);
            }
        }
    }

    static std::string prepared_name(int layer) {
        char name[64];
        std::snprintf(name, sizeof name, "model-layer-%05d.bin", layer + layer_offset());
        return name;
    }
    // What the prepared bytes depend on: a directory written for another layout must not be loaded.
    nlohmann::json prepared_layout() const {
        return {{"format", 1}, {"tp", tp}, {"wt", weights_transposed}, {"mlp_tiled", mlp_tiled}, {"hidden", hidden},
                {"intermediate", config.intermediate}, {"q_heads", config.q_heads}, {"kv_heads", config.kv_heads}, {"dim", dim},
                {"layers", config.layers}, {"layer_offset", layer_offset()}};
    }
    void check_prepared() {
        if (const char *d = std::getenv("BEHEMOTH_NEURON_PREPARED_DIR"))
            prepared_dir = d;
        if (const char *d = std::getenv("BEHEMOTH_NEURON_WRITE_PREPARED"))
            write_prepared_dir = d;
        if ((!prepared_dir.empty() || !write_prepared_dir.empty()) && n_compressed)
            throw std::runtime_error("prepared shards cover FP8-expanded layers only (QTIP layers kept compressed)");
        if (!prepared_dir.empty()) {
            wait_ready(prepared_dir / "layout.json");
            std::ifstream f(prepared_dir / "layout.json");
            const auto have = nlohmann::json::parse(f);
            if (have != prepared_layout())
                throw std::runtime_error("prepared shards in " + prepared_dir.string() + " were written for " + have.dump() + ", this load needs " +
                                         prepared_layout().dump());
            if (options.verbose)
                std::cerr << "loading prepared shards from " << prepared_dir << "\n";
        }
        if (!write_prepared_dir.empty()) {
            std::filesystem::create_directories(write_prepared_dir);
            std::ofstream(write_prepared_dir / "layout.json") << prepared_layout().dump(1) << "\n";
        }
    }
    // file: "BSHD0001", u32 entries, then per entry u32 rank, u32 slot (0..13: w[], 100 / 101: norm1 / norm2), u64 bytes; then the data
    struct PreparedEntry { uint32_t rank, slot; uint64_t bytes; };
    void write_prepared(int layer, const std::vector<PreparedEntry> &entries, const std::vector<std::vector<uint8_t>> &data) {
        const auto path = write_prepared_dir / prepared_name(layer);
        const auto tmp = path.string() + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary);
            const uint32_t n = uint32_t(entries.size());
            f.write("BSHD0001", 8);
            f.write(reinterpret_cast<const char *>(&n), 4);
            f.write(reinterpret_cast<const char *>(entries.data()), std::streamsize(entries.size() * sizeof(PreparedEntry)));
            for (const auto &d : data)
                f.write(reinterpret_cast<const char *>(d.data()), std::streamsize(d.size()));
            if (!f)
                throw std::runtime_error("cannot write " + tmp);
        }
        std::filesystem::rename(tmp, path);
    }
    void init_layer_state(int layer) { // KV cache (zeros) and FP8 scales (ones) of every rank
        std::vector<uint8_t> zeros(kv_bytes(), 0);
        for (int r = 0; r < tp; ++r) {
            auto &lw = ranks[size_t(r)].layers[size_t(layer)];
            lw.kv = alloc(r, zeros.size(), "kv_cache");
            put(lw.kv, zeros.data(), zeros.size());
            if (kv8) { // scales start at 1 so an unwritten page dequantises to 0 * 1
                std::vector<float> ones(kv_scale_bytes() / 4, 1.f);
                lw.ks = alloc(r, kv_scale_bytes(), "kv_scale");
                put(lw.ks, ones.data(), kv_scale_bytes());
            }
        }
    }
    void upload_prepared(int layer, const std::filesystem::path &path) {
        std::ifstream f(path, std::ios::binary);
        char magic[8];
        uint32_t n = 0;
        f.read(magic, 8);
        f.read(reinterpret_cast<char *>(&n), 4);
        if (!f || std::string(magic, 8) != "BSHD0001" || n > 16 * 64) // 16 entries per rank, up to 64 ranks
            throw std::runtime_error("bad prepared shard file " + path.string());
        std::vector<PreparedEntry> entries(n);
        f.read(reinterpret_cast<char *>(entries.data()), std::streamsize(n * sizeof(PreparedEntry)));
        std::vector<uint8_t> buf;
        for (const auto &e : entries) {
            if (e.rank >= uint32_t(tp) || !(e.slot < 14 || e.slot == 100 || e.slot == 101))
                throw std::runtime_error("bad prepared entry in " + path.string());
            auto &lw = ranks[size_t(e.rank)].layers[size_t(layer)];
            Dev &d = e.slot == 100 ? lw.norm1 : e.slot == 101 ? lw.norm2 : lw.w[e.slot];
            buf.resize(size_t(e.bytes));
            f.read(reinterpret_cast<char *>(buf.data()), std::streamsize(e.bytes));
            if (!f)
                throw std::runtime_error("truncated prepared shard file " + path.string());
            d = alloc(int(e.rank), size_t(e.bytes), "prepared");
            put(d, buf.data(), buf.size());
        }
        init_layer_state(layer);
    }

    void upload_layer_file(int layer, const std::filesystem::path &path) {
        SafeTensorFile file(path);
        struct P { const char *name; bool col; int slot; };
        const P projs[] = {{"self_attn.q_proj", true, 0}, {"self_attn.o_proj", false, 2}, {"self_attn.k_proj", true, 4},
                           {"self_attn.v_proj", true, 6}, {"mlp.gate_proj", true, 8}, {"mlp.up_proj", true, 10}, {"mlp.down_proj", false, 12}};
        const std::string base = "model.layers." + std::to_string(layer + layer_offset()) + ".";
        const size_t H = size_t(hidden);
        std::vector<float> n1(H), n2(H);
        {
            auto a = file.floats(base + "input_layernorm.weight"), b = file.floats(base + "post_attention_layernorm.weight");
            for (size_t i = 0; i < H; ++i) { n1[i] = a[i]; n2[i] = b[i]; }
        }
        // BEHEMOTH_NEURON_WRITE_PREPARED: keep a copy of every uploaded weight buffer for this layer's prepared shard file
        const bool record = !write_prepared_dir.empty();
        std::vector<PreparedEntry> rec_entries;
        std::vector<std::vector<uint8_t>> rec_data;
        auto upload = [&](int r, uint32_t slot, Dev &d, const void *src, size_t bytes) {
            put(d, src, bytes);
            if (record) {
                rec_entries.push_back({uint32_t(r), slot, uint64_t(bytes)});
                rec_data.emplace_back(static_cast<const uint8_t *>(src), static_cast<const uint8_t *>(src) + bytes);
            }
        };
        for (int r = 0; r < tp; ++r) {
            auto &lw = ranks[size_t(r)].layers[size_t(layer)];
            lw.norm1 = alloc(r, H * 4, "norm1");
            lw.norm2 = alloc(r, H * 4, "norm2");
            upload(r, 100, lw.norm1, n1.data(), H * 4);
            upload(r, 101, lw.norm2, n2.data(), H * 4);
        }
        init_layer_state(layer);
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
                    // 64 x 64 blocks over threads: the element-wise strided transpose was most of the load time
                    constexpr int B = 64;
                    const int rb = (s.rows + B - 1) / B, cb = (s.cols + B - 1) / B;
#pragma omp parallel for schedule(static)
                    for (int blk = 0; blk < rb * cb; ++blk) {
                        const int i0 = (blk / cb) * B, j0 = (blk % cb) * B, i1 = std::min(s.rows, i0 + B), j1 = std::min(s.cols, j0 + B);
                        for (int i = i0; i < i1; ++i)
                            for (int j = j0; j < j1; ++j)
                                t[size_t(j) * size_t(s.rows) + size_t(i)] = s.w[size_t(i) * size_t(s.cols) + size_t(j)];
                    }
                    if (mlp_tiled && (p.slot == 8 || p.slot == 10 || p.slot == 12)) {
                        // nki_mlp.tile_weight: [K][N] -> blocks (kb, nb) of 128 partition rows of 4 x 1024 (rows kb*512 + j*128 + q)
                        const size_t K = size_t(s.cols), N = size_t(s.rows);
                        if (K % 512 || N % 1024)
                            throw std::runtime_error(std::string(p.name) + ": tiled MLP needs K % 512 == 0 and N % 1024 == 0");
                        std::vector<uint8_t> tiled(t.size());
                        const size_t NBL = N / 1024;
#pragma omp parallel for schedule(static)
                        for (size_t k = 0; k < K; ++k) {
                            const size_t kb = k / 512, jj = (k % 512) / 128, q = k % 128;
                            for (size_t nb = 0; nb < NBL; ++nb)
                                std::memcpy(&tiled[(((kb * NBL + nb) * 128 + q) * 4 + jj) * 1024], &t[k * N + nb * 1024], 1024);
                        }
                        t.swap(tiled);
                    }
                    upload(r, uint32_t(p.slot), lw.w[p.slot], t.data(), t.size());
                } else
                    upload(r, uint32_t(p.slot), lw.w[p.slot], s.w.data(), s.w.size());
                upload(r, uint32_t(p.slot + 1), lw.w[p.slot + 1], s.scale.data(), s.scale.size() * 4);
            }
        }
        if (record)
            write_prepared(layer, rec_entries, rec_data);
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
        // qtip_quant.layer_formats (written by the quantiser and repack_compressed_tensors.py) says which layers are QTIP without
        // opening the layer files, so a streamed load can start uploading layer 0 while later layers are still downloading
        std::vector<std::string> formats;
        try {
            std::ifstream cf(options.model / "config.json");
            const auto cj = nlohmann::json::parse(cf);
            if (cj.contains("qtip_quant") && cj.at("qtip_quant").contains("layer_formats"))
                for (const auto &f : cj.at("qtip_quant").at("layer_formats"))
                    formats.push_back(f.get<std::string>());
        } catch (...) {
            formats.clear();
        }
        if (int(formats.size()) < config.layers + layer_offset())
            formats.clear();
        for (int layer = 0; layer < config.layers; ++layer) {
            bool all = true;
            if (!formats.empty())
                all = formats[size_t(layer + layer_offset())] == "qtip";
            else {
                char name[64];
                std::snprintf(name, sizeof name, "model-layer-%05d.safetensors", layer + layer_offset());
                wait_ready(options.model / name);
                SafeTensorFile file(options.model / name);
                for (const auto &p : projs)
                    all = all && is_qtip_projection(file, layer, p.name);
            }
            for (const auto &p : projs) {
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
                used += double(kv_bytes()) + 2.0 * hidden * 4 + 2.0 * (size_t(kv_local) * size_t(dim) * size_t(hidden) + size_t(kv_local) * size_t(dim) * 4);
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
        check_prepared();
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
            kv_bytes_rank += lw.kv.bytes + lw.ks.bytes;
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
            if (name == "cscale") return lw.ks;
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
                            } else { // the updated KV state (cache, then its scales), written back into the same device buffers
                                const Dev &kv = extra == 0 ? lw.kv : lw.ks;
                                if (extra++ > (variant.meta.kv8 ? 1 : 0) || kv.bytes != t.size)
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
                    if (h.argmax)
                        check(nrt_add_tensor_to_tensor_set(hr.out[size_t(p)], "output1", hr.amx.t), "head amx out");
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
        bool want_argmax = false;     // device argmax head for this launch (falls back to the logits head when unavailable)
    };

    void embed_row(int token, float *out) const {
        if (token < 0 || token >= vocab_)
            throw std::runtime_error("token outside the vocabulary");
        for (size_t i = 0; i < size_t(hidden); ++i)
            out[i] = embed[size_t(token) * size_t(hidden) + i];
        if (const char *file = std::getenv("NEURON_X_IN")) { // debugging: replace the embedding row by f32 [hidden] from this file
            std::ifstream in(file, std::ios::binary);
            in.read(reinterpret_cast<char *>(out), std::streamsize(size_t(hidden) * 4));
        }
    }
    void rope_row(int position, float *out) const { // [cos x dim/2, sin x dim/2] from fp64 angles, any position (YaRN: x mscale)
        const double m = config.rope_attention_factor;
        for (int d = 0; d < dim / 2; ++d) {
            const double angle = double(position) * double(inv_freq[size_t(d)]);
            out[d] = float(m * std::cos(angle));
            out[dim / 2 + d] = float(m * std::sin(angle));
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
            const auto t_g = Clock::now();
            gather_logits(L, out, dest);
            phase_gather_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_g).count();
            return;
        }
        const HeadBucket *head = nullptr;
        if (!L.head_rows.empty()) {
            for (const auto &h : heads)
                if (h.meta.T == b.tokens() && h.meta.S >= int(L.head_rows.size()) && h.argmax == L.want_argmax) {
                    head = &h;
                    break;
                }
            if (!head && L.want_argmax) { // no argmax head for this shape: fall back to the logits head
                for (const auto &h : heads)
                    if (h.meta.T == b.tokens() && h.meta.S >= int(L.head_rows.size()) && !h.argmax) {
                        head = &h;
                        L.want_argmax = false;
                        break;
                    }
            }
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
        std::vector<std::vector<int32_t>> amx(static_cast<size_t>(tp));
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
                // drain before stopping the layers clock: the executes are async (the runtime queues them),
                // so reading a few bytes of the residual stream forces the wait for THIS rank's device work
                // (the collectives inside the graphs sync both ranks).
                {
                    float drain[16];
                    check(nrt_tensor_read(pr.x[0].t, drain, 0, sizeof(drain)), "read x drain");
                    check(nrt_tensor_read(pr.x[1].t, drain, 0, sizeof(drain)), "read x drain");
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
                    const auto t_head_submit = Clock::now();
                    if (head->argmax && L.want_argmax) {
                        out[size_t(r)].resize(size_t(L.head_rows.size()));
                        check(nrt_tensor_read(hr.logits.t, out[size_t(r)].data(), 0, out[size_t(r)].size() * 4), "read head max");
                        amx[size_t(r)].resize(size_t(L.head_rows.size()));
                        check(nrt_tensor_read(hr.amx.t, amx[size_t(r)].data(), 0, amx[size_t(r)].size() * 4), "read head argmax");
                    } else {
                        out[size_t(r)].resize(size_t(head->meta.S) * size_t(vocab_shard));
                        check(nrt_tensor_read(hr.logits.t, out[size_t(r)].data(), 0, out[size_t(r)].size() * 4), "read logits");
                    }
                    if (r == 0) {
                        phase_head_submit_ms += std::chrono::duration<double, std::milli>(t_head_submit - t_layers).count();
                        phase_head_read_ms += ms_since(t_head_submit);
                        phase_head_ms += ms_since(t_layers);
                    }
                }
            }));
        for (auto &j : jobs)
            j.get();
        if (head && head->argmax && L.want_argmax) {
            // Merge the two ranks' (max, argmax-shard-index) candidates. vocab_shard = ceil(vocab/TP) with the
            // padded tail on the last rank, so idx + r*vocab_shard can exceed vocab-1 only for ranks that own
            // padded rows - harmless for vocab=131072/TP=2 (131072 is even, no padding) but not general; ties
            // keep rank 0 (the lowest shard = the same convention as the host argmax).
            for (size_t i = 0; i < L.head_rows.size(); ++i) {
                int best_r = 0;
                float best = out[0][i];
                for (int r = 1; r < tp; ++r)
                    if (out[size_t(r)][i] > best) {
                        best = out[size_t(r)][i];
                        best_r = r;
                    }
                if (L.destination[i] >= 0 && size_t(L.destination[i]) < device_tokens.size())
                    device_tokens[size_t(L.destination[i])] = int(amx[size_t(best_r)][i]) + best_r * vocab_shard;
            }
        } else if (head) {
            const auto t_g = Clock::now();
            gather_logits(L, out, dest);
            phase_gather_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_g).count();
        }
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

    bool device_argmax_used() const { return used_device_argmax; }
    int device_token(size_t row) const { return row < device_tokens.size() ? device_tokens[row] : -1; }

    std::span<const float> forward(const InferenceBatch &batch, bool want_argmax = false) {
        const auto t0 = Clock::now();
        phase_upload_ms = phase_layers_ms = phase_head_ms = 0;
        phase_head_submit_ms = phase_head_read_ms = phase_gather_ms = 0;
        device_tokens.assign(batch.sample_rows.size(), -1);
        used_device_argmax = false;
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
        // shared: true picks among the shared-table buckets only (every row reads table row 0, so the launch must hold the rows of
        // ONE request); false (the default) never picks one, so a graph set without such buckets behaves as before.
        auto pick = [&](std::vector<LayerBucket> &set, int need_rows, int need_queries, int pages, bool shared = false) -> LayerBucket * {
            LayerBucket *best = nullptr;
            for (auto &b : set)
                if (b.meta.shared_table == shared && b.meta.R >= need_rows && b.meta.Q >= need_queries && b.meta.P >= pages && (!best || cost(b) < cost(*best)))
                    best = &b;
            return best;
        };
        int launches = 0, max_rows = 0, max_queries = 0;
        std::string decode_bucket; // step log: the last decode launch's bucket, "dec<R>_<P>" (+ "s" for a shared-table bucket)
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
        // Short requests first, so they share the wide buckets (which have short page tables) and a long one does
        // not drag them into a long-table launch; stable keeps each request's rows in token order.
        std::stable_sort(rows.begin(), rows.end(), [&](const Row &a, const Row &b) { return pages_needed(a.req) < pages_needed(b.req); });
        // all rows in one launch when a bucket fits, else the largest group a bucket covers
        for (size_t at = 0; at < rows.size();) {
            size_t n = rows.size() - at;
            auto pages_of = [&](size_t count) {
                int pages = 1;
                for (size_t i = 0; i < count; ++i) pages = std::max(pages, pages_needed(rows[at + i].req));
                return pages;
            };
            LayerBucket *bucket = nullptr;
            // a multi-row launch whose rows all belong to one request (a verify step: the scheduled token + its drafts) may use a
            // shared-table bucket (one page gather for all rows); any other mix uses the per-row buckets as before
            if (n > 1 && std::all_of(rows.begin() + std::ptrdiff_t(at), rows.begin() + std::ptrdiff_t(at + n), [&](const Row &r) { return r.req == rows[at].req; }))
                bucket = pick(decode, int(n), 1, pages_of(n), true);
            if (!bucket)
                bucket = pick(decode, int(n), 1, pages_of(n));
            while (!bucket && n > 1) {
                n = std::min(n - 1, size_t(max_rows));
                bucket = pick(decode, int(n), 1, pages_of(n));
            }
            if (!bucket)
                throw std::runtime_error("no decode NEFF bucket large enough for " + std::to_string(pages_of(n)) + " pages");
            auto L = fresh(*bucket);
            decode_bucket = "dec" + std::to_string(bucket->meta.R) + "_" + std::to_string(bucket->meta.P) + (bucket->meta.shared_table ? "s" : "");
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
            L.want_argmax = want_argmax;
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
                L.want_argmax = want_argmax;
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
        used_device_argmax = want_argmax && std::any_of(device_tokens.begin(), device_tokens.end(), [](int t) { return t >= 0; });
        profile = {{"launches", launches}, {"forward_ms", ms_since(t0)}, {"upload_ms", phase_upload_ms}, {"layers_ms", phase_layers_ms},
                   {"head_ms", phase_head_ms}, {"head_submit_ms", phase_head_submit_ms}, {"head_read_ms", phase_head_read_ms},
                   {"gather_ms", phase_gather_ms}, {"tokens", batch.tokens.size()}, {"decode_bucket", decode_bucket}};
        return logits;
    }
};

NeuronModel::NeuronModel(const ModelOptions &o) {
    if (is_glm_prepared(o.model))
        backend_ = make_glm_backend(o);
    else
        impl_ = std::make_unique<Impl>(o);
}
NeuronModel::~NeuronModel() = default;
bool NeuronModel::recurrent() const { return backend_ != nullptr; }
int NeuronModel::cache_pages() const { return backend_ ? backend_->cache_pages() : impl_->pages_total - 1; }
int NeuronModel::vocab() const { return backend_ ? backend_->vocab() : impl_->vocab_; }
int NeuronModel::max_decode_rows() const {
    if (backend_)
        return backend_->max_decode_rows();
    int r = 0;
    for (const auto &b : impl_->decode) r = std::max(r, b.meta.R);
    return r;
}
int NeuronModel::decode_bucket_pages(int pages) const {
    if (backend_)
        return backend_->decode_bucket_pages(pages);
    int best = 0;
    for (const auto &b : impl_->decode)
        if (b.meta.P >= pages && (!best || b.meta.P < best)) best = b.meta.P;
    return best;
}
bool NeuronModel::has_shared_verify() const {
    if (backend_)
        return false;
    for (const auto &b : impl_->decode)
        if (b.meta.shared_table)
            return true;
    return false;
}
const std::vector<int> &NeuronModel::eos() const { return backend_ ? backend_->eos() : impl_->config.eos; }
int NeuronModel::max_context() const {
    if (backend_)
        return backend_->max_context();
    int p = 0;
    for (const auto &b : impl_->decode) p = std::max(p, b.meta.P);
    return std::min(p * PAGE, impl_->config.max_context);
}
std::span<const float> NeuronModel::forward(const InferenceBatch &b, bool want_argmax) {
    if (backend_)
        return backend_->forward(b);
    return impl_->forward(b, want_argmax);
}
bool NeuronModel::device_argmax_used() const { return !backend_ && impl_->device_argmax_used(); }
int NeuronModel::device_token(size_t logits_row) const { return backend_ ? -1 : impl_->device_token(logits_row); }
nlohmann::json NeuronModel::last_profile() const { return backend_ ? backend_->last_profile() : impl_->profile; }
// Prefix-cache copies exist only for the Llama graphs (the engine disables the cache for recurrent models).
size_t NeuronModel::kv_page_bytes() const { return backend_ ? 0 : impl_->kv_page_rank_bytes() * size_t(impl_->tp); }
void NeuronModel::kv_save(std::span<const int> device_pages, std::span<const int> host_pages, std::byte *arena) {
    if (backend_)
        throw std::logic_error("no prefix cache for this model");
    impl_->kv_copy(device_pages, host_pages, arena, true);
}
void NeuronModel::kv_restore(std::span<const int> host_pages, std::span<const int> device_pages, const std::byte *arena) {
    if (backend_)
        throw std::logic_error("no prefix cache for this model");
    impl_->kv_copy(device_pages, host_pages, const_cast<std::byte *>(arena), false);
}
} // namespace neuron
