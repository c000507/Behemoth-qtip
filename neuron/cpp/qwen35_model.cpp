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
// disables prompt-lookup speculation and the prefix cache for this family (NeuronModel::recurrent()).
#include "neuron_model.hpp"
#include "behemoth/worker.hpp"
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <algorithm>
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

// One NEFF: a sub-block kind ("gdn", "gqa", "dense", "moe"), a whole layer ("layer_<attn>_<mlp>": both sub-blocks,
// weights named a.* / m.*) or "head", for one bucket.
struct Graph {
    std::string kind, attn, mlp; // attn / mlp: the sub-blocks it contains ("" for none)
    int R = 0, Q = 0, P = 0, T = 0, S = 0, pages = 0, slots = 0;
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
        Dev x[2], pos, slot, wslot, cos_sin;
        std::map<int, Dev> table, rows, logits; // table by P; head rows / logits by the head's S
    };
    std::map<std::pair<int, int>, std::vector<Step>> steps;
    // tensor sets per (graph, layer, parity, rank)
    std::map<std::tuple<const Graph *, int, int, int>, std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *>> sets;
    std::mutex sets_mutex; // layer_set / head_set run on every rank's worker thread concurrently
    std::vector<float> logits;
    json profile = json::object();
    // GDN state slots: sequence (first page) -> slot, LRU order; spilled states on the host
    std::map<int, int> slot_of;
    std::list<int> lru; // sequences with a slot, most recent last
    std::map<int, std::vector<std::vector<char>>> spilled; // sequence -> [rank] bytes of every GDN layer's slot
    uint64_t spills = 0, restores = 0;

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
        // BEHEMOTH_NEURON_QWEN35_ASYNC=1: implicit async execution (launches return before the device finishes; the logits
        // read synchronises), hides launch gaps. Off by default: a GDN state spill / restore between two launches of one
        // step relies on the previous launch having finished.
        if (const char *a = std::getenv("BEHEMOTH_NEURON_QWEN35_ASYNC"); a && std::atoi(a))
            setenv("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "8", 0);
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
                (name.find("state") != std::string::npos || name.ends_with(".kv") ? s : w) += d.bytes;
            std::cerr << "Qwen35: " << layers << " layers, TP " << tp << ", " << graphs.size() << " NEFFs, " << pages_total - 1 << " pages, "
                      << slots_total - 1 << " GDN state slots; per core " << double(w) / (1 << 30) << " GiB weights + " << double(s) / (1 << 30)
                      << " GiB state\n";
        }
    }
    ~Qwen35Model() override {
        if (embed_map)
            munmap(embed_map, embed_map_bytes);
    }

    void map_embedding() {
        const auto &e = prepared.at("files").at("embed.bin").at(0);
        const auto path = root / "embed.bin";
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("cannot open " + path.string());
        embed_map_bytes = std::filesystem::file_size(path);
        embed_map = mmap(nullptr, embed_map_bytes, PROT_READ, MAP_SHARED, fd, 0);
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
            } else if (gr.kind == "gdn" || gr.kind == "gqa")
                gr.attn = gr.kind;
            else if (gr.kind != "head")
                gr.mlp = gr.kind;
            gr.inputs = m.at("inputs").get<std::vector<std::string>>();
            gr.states = m.at("states").get<std::vector<std::string>>();
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
                if (gr.Q > 1 && gr.R != 1)
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
            if (g.kind != "head")
                groups.insert({g.R, g.Q});
        for (auto [R, Q] : groups) {
            for (int l = 0; l < layers; ++l)
                if (!layer_graph(l, R, Q, 1) && (!find(attn_kind[size_t(l)], R, Q, 1) || !find(mlp_kind[size_t(l)], R, Q, 1)))
                    throw std::runtime_error("Qwen35 bucket R=" + std::to_string(R) + " Q=" + std::to_string(Q) + " has neither a layer_" +
                                             attn_kind[size_t(l)] + "_" + mlp_kind[size_t(l)] + " NEFF nor both sub-block NEFFs for layer " +
                                             std::to_string(l));
            if (!head_for(R * Q, 1))
                throw std::runtime_error("Qwen35 bucket R=" + std::to_string(R) + " Q=" + std::to_string(Q) + " has no head NEFF for " +
                                         std::to_string(R * Q) + " rows");
        }
    }
    // the cheapest graph of `kind` with exactly (R, Q) and at least P table pages (P ignored without an mla sub-block)
    const Graph *find(const std::string &kind, int R, int Q, int P) const {
        const Graph *best = nullptr;
        for (const auto &g : graphs)
            if (g.kind == kind && g.R == R && g.Q == Q && (g.attn != "gqa" || g.P >= P) && (!best || g.P < best->P))
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
            const size_t n = gr.inputs.size() + gr.states.size();
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
            for (size_t s = 0; s < gr.states.size(); ++s)
                gr.state_bytes.push_back(gr.input_bytes[gr.inputs.size() + s]);
            nrt_free_model_tensor_info(info);
            if (options.verbose)
                std::cerr << "Qwen35 NEFF " << gr.kind << " R=" << gr.R << " Q=" << gr.Q << " P=" << gr.P << " T=" << gr.T << " S=" << gr.S << "\n";
        }
        // step buffers per (R, Q) group: residual ping-pong, positions, write slots, state slots, one table per P, head rows / logits
        for (const auto &gr : graphs) {
            if (gr.kind == "head")
                continue;
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
                    s.cos_sin = alloc(r, T * size_t(2 * rot) * 4, "cos_sin");
                    for (const auto &h : graphs)
                        if (h.kind == "head" && h.T == gr.T && !s.rows.count(h.S)) {
                            s.rows[h.S] = alloc(r, size_t(h.S) * 8, "rows");
                            s.logits[h.S] = alloc(r, size_t(h.S) * size_t(vocab_shard) * 4, "logits");
                        }
                }
            }
            if (gr.attn == "gqa" && !v[0].table.count(gr.P))
                for (int r = 0; r < tp; ++r)
                    v[size_t(r)].table[gr.P] = alloc(r, size_t(gr.R) * size_t(gr.P) * 8, "table");
        }
    }

    void load_weights() {
        const auto t0 = Clock::now();
        const auto &files = prepared.at("files");
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                auto upload = [&](const std::string &file, const std::string &prefix) {
                    const auto data = slurp(root / file);
                    for (const auto &e : files.at(file)) {
                        const std::string name = prefix + e.at("name").get<std::string>();
                        const size_t bytes = e.at("bytes"), offset = e.at("offset");
                        if (offset > data.size() || bytes > data.size() - offset)
                            throw std::runtime_error(file + ": " + name + " lies outside the file");
                        Dev d = alloc(r, bytes, name);
                        put(d, data.data() + offset, bytes);
                        ranks[size_t(r)].t[name] = d;
                    }
                };
                upload("r" + std::to_string(r) + "/head.bin", "head.");
                for (int l = 0; l < layers; ++l)
                    upload("r" + std::to_string(r) + "/" + layer_file(l), layer_prefix(l));
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
    std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *> layer_set(const Graph &g, int l, int parity, int r) {
        auto key = std::make_tuple(&g, l, parity, r);
        std::lock_guard lock(sets_mutex);
        if (auto it = sets.find(key); it != sets.end())
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
            if (name == "wslot") return st.wslot;
            if (name == "cos_sin") return st.cos_sin;
            if (name == "table") return st.table.at(g.P);
            std::string full = layer_prefix(l) + sub + name;
            if (name.rfind("a.", 0) == 0) full = layer_prefix(l) + "attn." + name.substr(2);
            if (name.rfind("m.", 0) == 0) full = layer_prefix(l) + "mlp." + name.substr(2);
            auto it = rank.t.find(full);
            if (it == rank.t.end())
                throw std::runtime_error("Qwen35: layer " + std::to_string(l) + " has no tensor " + full + " for the " + g.kind + " NEFF");
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
                throw std::runtime_error("Qwen35: " + g.kind + " input " + names[i] + " of layer " + std::to_string(l) + " expects " +
                                         std::to_string(g.input_bytes[i]) + " B, has " + std::to_string(d.bytes));
            check(nrt_add_tensor_to_tensor_set(in, ("input" + std::to_string(i)).c_str(), d.t), "add input");
        }
        check(nrt_add_tensor_to_tensor_set(out, "output0", st.x[1 - parity].t), "add output");
        for (size_t s = 0; s < g.states.size(); ++s)
            check(nrt_add_tensor_to_tensor_set(out, ("output" + std::to_string(s + 1)).c_str(), bind(g.states[s]).t), "add state output");
        return sets[key] = {in, out};
    }
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
        std::set<int> used;
        for (auto [s, sl] : slot_of)
            used.insert(sl);
        int slot = -1;
        for (int sl = 0; sl < slots_total - 1 && slot < 0; ++sl)
            if (!used.count(sl))
                slot = sl;
        if (slot < 0) {
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                const int victim = *it, vs = slot_of.at(victim);
                if (pinned.count(vs))
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
        slot_of[seq] = slot;
        lru.push_back(seq);
        pinned.insert(slot);
        if (position > 0) {
            copy_slot(seq, slot, false);
            ++restores;
        } else
            spilled.erase(seq);
        return slot;
    }

    // ---- one launch: R rows x Q tokens through every layer, then the head for the sampled rows ------------------------
    struct Launch {
        int R = 0, Q = 0, P = 0;
        std::vector<float> x;
        std::vector<int64_t> pos, wslot, table, slot;
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
    double layers_ms = 0, head_ms = 0;
    int launches = 0;
    void execute(Launch &L, float *dest) {
        const Graph *head = nullptr;
        if (!L.head_rows.empty()) {
            head = head_for(L.R * L.Q, int(L.head_rows.size()));
            if (!head)
                throw std::runtime_error("Qwen35: no head NEFF for " + std::to_string(L.head_rows.size()) + " rows");
        }
        // per layer: the fused layer NEFF when traced for this bucket, else the attention and MLP sub-block NEFFs
        std::vector<std::vector<const Graph *>> plan(static_cast<size_t>(layers));
        for (int l = 0; l < layers; ++l) {
            if (const Graph *f = layer_graph(l, L.R, L.Q, L.P))
                plan[size_t(l)] = {f};
            else
                plan[size_t(l)] = {find(attn_kind[size_t(l)], L.R, L.Q, L.P), find(mlp_kind[size_t(l)], L.R, L.Q, 0)};
        }
        // every table width the bucket's mla graphs use gets the request pages, padded with the scratch page
        std::map<int, std::vector<int64_t>> tables;
        for (const auto &v : plan)
            for (const Graph *g : v)
                if (g->attn == "gqa" && !tables.count(g->P)) {
                    auto &t = tables[g->P];
                    t.assign(size_t(L.R) * size_t(g->P), pages_total - 1);
                    for (int r = 0; r < L.R; ++r)
                        std::copy_n(L.table.begin() + std::ptrdiff_t(r * L.P), std::min(L.P, g->P), t.begin() + std::ptrdiff_t(r * g->P));
                }
        std::vector<int64_t> rows(head ? size_t(head->S) : 0, 0);
        for (size_t i = 0; i < L.head_rows.size(); ++i)
            rows[i] = L.head_rows[i];
        std::vector<std::vector<float>> out(static_cast<size_t>(tp));
        std::vector<std::future<void>> jobs;
        double t_layers = 0, t_head = 0;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                auto &st = steps.at({L.R, L.Q})[size_t(r)];
                const auto t0 = Clock::now();
                put(st.x[0], L.x.data(), L.x.size() * 4);
                put(st.pos, L.pos.data(), L.pos.size() * 8);
                put(st.wslot, L.wslot.data(), L.wslot.size() * 8);
                put(st.slot, L.slot.data(), L.slot.size() * 8);
                put(st.cos_sin, L.cos_sin.data(), L.cos_sin.size() * 4);
                for (const auto &[width, t] : tables)
                    put(st.table.at(width), t.data(), t.size() * 8);
                int parity = 0;
                static const int dump_x = std::getenv("NEURON_X_DUMP") ? std::atoi(std::getenv("NEURON_X_DUMP")) : 0; // 1: report non-finite layers, 2: every layer
                std::vector<float> xr;
                for (int l = 0; l < layers; ++l)
                    for (const Graph *g : plan[size_t(l)]) {
                        auto [in, o] = layer_set(*g, l, parity, r);
                        check(nrt_execute(g->model[size_t(r)], in, o), g->kind + " execute");
                        parity ^= 1;
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
                const auto t1 = Clock::now();
                if (head) {
                    put(st.rows.at(head->S), rows.data(), rows.size() * 8);
                    auto [in, o] = head_set(*head, L.R, L.Q, r, parity);
                    check(nrt_execute(head->model[size_t(r)], in, o), "head execute");
                    out[size_t(r)].resize(size_t(head->S) * size_t(vocab_shard));
                    check(nrt_tensor_read(st.logits.at(head->S).t, out[size_t(r)].data(), 0, out[size_t(r)].size() * 4), "read logits");
                }
                if (r == 0) {
                    t_layers = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    t_head = ms_since(t1);
                }
            }));
        for (auto &j : jobs)
            j.get();
        layers_ms += t_layers;
        head_ms += t_head;
        ++launches;
        for (size_t i = 0; i < L.head_rows.size(); ++i) {
            float *row = dest + size_t(L.destination[i]) * size_t(vocab_);
            for (int r = 0; r < tp; ++r) {
                const size_t first = size_t(r) * size_t(vocab_shard);
                if (first >= size_t(vocab_))
                    continue;
                std::copy_n(out[size_t(r)].data() + i * size_t(vocab_shard), std::min(size_t(vocab_) - first, size_t(vocab_shard)), row + first);
            }
        }
    }

    // ---- Backend --------------------------------------------------------------------------------------------------
    int cache_pages() const override { return pages_total - 1; }
    int vocab() const override { return vocab_; }
    int max_decode_rows() const override {
        int r = 0;
        for (const auto &g : graphs)
            if (g.attn == "gdn" && g.Q == 1) r = std::max(r, g.R);
        return std::min(r, slots_total - 1);
    }
    int decode_bucket_pages(int pages) const override {
        int best = 0;
        for (const auto &g : graphs)
            if (g.attn == "gqa" && g.Q == 1 && g.P >= pages && (!best || g.P < best)) best = g.P;
        return best;
    }
    const std::vector<int> &eos() const override { return eos_; }
    int max_context() const override {
        int p = 0;
        for (const auto &g : graphs)
            if (g.attn == "gqa") p = std::max(p, g.P);
        return std::min(p * PAGE, max_context_);
    }
    json last_profile() const override { return profile; }

    std::span<const float> forward(const InferenceBatch &batch) override {
        const auto t0 = Clock::now();
        layers_ms = head_ms = 0;
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
        std::vector<int> rows, multi;
        for (int req = 0; req < lay.requests(); ++req) {
            const int count = lay.query_offsets[size_t(req) + 1] - lay.query_offsets[size_t(req)];
            (count == 1 ? rows : multi).push_back(req);
        }
        std::set<int> decode_R;
        std::set<int> prefill_Q;
        for (const auto &g : graphs)
            if (g.attn == "gdn") (g.Q == 1 ? decode_R.insert(g.R) : prefill_Q.insert(g.Q));
        const int maxR = std::min(decode_R.empty() ? 0 : *decode_R.rbegin(), slots_total - 1);
        for (size_t at = 0; at < rows.size();) {
            if (!maxR)
                throw std::runtime_error("Qwen35: no decode buckets");
            size_t n = std::min(rows.size() - at, size_t(maxR));
            int R = *decode_R.lower_bound(int(n)), pages = 1;
            for (size_t i = 0; i < n; ++i)
                pages = std::max(pages, pages_needed(rows[at + i]));
            const Graph *g = find_attn("gqa", R, 1, pages);
            if (!g)
                throw std::runtime_error("Qwen35: no decode bucket R=" + std::to_string(R) + " with " + std::to_string(pages) + " pages");
            auto L = fresh(R, 1, g->P);
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
            at += n;
        }
        for (int req : multi) {
            const int first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
            for (int done = 0; done < count;) {
                const int left = count - done;
                // the smallest chunk bucket covering what is left, else the largest
                int Q = prefill_Q.empty() ? 0 : *prefill_Q.rbegin();
                if (auto it = prefill_Q.lower_bound(left); it != prefill_Q.end()) Q = *it;
                if (!Q) { // no prefill buckets: run the prompt as decode rows of one request
                    Q = 1;
                }
                const int take = std::min(left, Q);
                const Graph *g = find_attn("gqa", 1, Q, pages_needed(req));
                if (!g)
                    throw std::runtime_error("Qwen35: no prefill bucket Q=" + std::to_string(Q) + " with " + std::to_string(pages_needed(req)) + " pages");
                auto L = fresh(1, Q, g->P);
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
        profile = {{"launches", launches}, {"forward_ms", ms_since(t0)}, {"layers_ms", layers_ms}, {"head_ms", head_ms},
                   {"tokens", batch.tokens.size()}, {"kda_spills", spills}, {"kda_restores", restores}};
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
