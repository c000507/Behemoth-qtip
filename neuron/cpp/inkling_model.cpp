// Inkling 2 Mini (inkling2_loop, text) on NeuronCores behind NeuronModel's contract: forward(InferenceBatch) -> logits.
// Derived from qwen35_model.cpp (same prepared-file format, rank threads, tensor sets, slot LRU with host spill); graphs from
// neuron/inkling_graphs.py, weights from neuron/inkling_prepare.py (prepared.json format "inkling2-neuron-prepared").
//
// A token runs 48 layer VISITS (layers 6..17 three times, shared weights): one NEFF per visit (attention + MoE) or per
// SEGMENT of consecutive visits (seg_sssssg / seg_SSSSSG), then the head. The residual is x [T, 7, hidden] f32 (embedding,
// three completed blocks, the partial sum, the previous loop pass's two message sources); the host writes the normed
// embedding into slots 0 and 4. Each visit gets its small control vector ctl (which blocks are visible, block boundary,
// loop restart) as a device constant "V<v>.ctl".
//
// State: per visit and SEQUENCE SLOT a K|V buffer of the last 1024 tokens (logical order) and the conv states, bound per
// row (one-slot slices of [slots, ...] tensors: no runtime-indexed state DMA, docs/neuron-reap-nonfinite.md); the 8 visits
// of the 4 global layers also hold the server's paged K/V (keys farther than 1024 tokens). Sequences are identified by
// their first page; slots are an LRU cache with host spill (as the Qwen3.5 GDN slots); the last slot is scratch. A row at
// position 0 starts from zero state (the graph zeroes it). No prefix cache, no speculation (recurrent family).
#include "neuron_model.hpp"
#include "behemoth/worker.hpp"
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
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
        throw std::runtime_error("Inkling: " + what + " failed: nrt status " + std::to_string(int(status)));
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
// WT layout (inkling_graphs.to_wt): 2-D projections / router / message gates / head [out, in] -> [in, out]; egu per expert
bool wt_2d(const std::string &base) {
    static const std::set<std::string> proj = {"q", "gt", "k", "v", "o", "r", "sk", "kw", "vw", "rw", "w"};
    return proj.count(base) > 0 || base.rfind("gd_", 0) == 0 || base.rfind("gf_", 0) == 0 || base.rfind("gg_", 0) == 0;
}
void transpose_into(uint8_t *dst, const uint8_t *src, size_t rows, size_t cols, size_t el) {
    constexpr size_t B = 64;
    for (size_t r0 = 0; r0 < rows; r0 += B)
        for (size_t c0 = 0; c0 < cols; c0 += B) {
            const size_t r1 = std::min(rows, r0 + B), c1 = std::min(cols, c0 + B);
            for (size_t r = r0; r < r1; ++r)
                for (size_t c = c0; c < c1; ++c)
                    std::memcpy(dst + (c * rows + r) * el, src + (r * cols + c) * el, el);
        }
}
bool to_wt(const std::string &name, const json &shape, size_t bytes, const uint8_t *src, std::vector<uint8_t> &out) {
    const std::string base = name.substr(name.rfind('.') + 1);
    std::vector<size_t> sh;
    for (const auto &d : shape)
        sh.push_back(d.get<size_t>());
    size_t n = 1;
    for (size_t d : sh)
        n *= d;
    if (!n || bytes % n)
        throw std::runtime_error(name + ": shape does not match its byte count");
    const size_t el = bytes / n;
    if (sh.size() == 2 && wt_2d(base)) {
        out.resize(bytes);
        transpose_into(out.data(), src, sh[0], sh[1], el);
        return true;
    }
    if (sh.size() == 3 && base == "egu") {
        out.resize(bytes);
        const size_t E = sh[0], per = sh[1] * sh[2] * el;
        std::vector<std::thread> th;
        for (size_t k = 0; k < 4; ++k)
            th.emplace_back([&, k] {
                for (size_t e = k; e < E; e += 4)
                    transpose_into(out.data() + e * per, src + e * per, sh[1], sh[2], el);
            });
        for (auto &t : th)
            t.join();
        return true;
    }
    return false;
}

// One NEFF: a visit kind (s | S | g | G), a segment of visit kinds, or the head.
struct Graph {
    std::string kind;                // "s", "seg_sssssg", "head"
    std::string kinds;               // the visit kinds it runs, in order ("" for the head)
    bool has_g = false;              // binds a page table (contains a global visit)
    int R = 0, Q = 0, P = 0, T = 0, S = 0, pages = 0;
    bool wt = false;
    std::vector<std::string> inputs, states; // NEFF inputs, then the aliased states
    std::vector<nrt_model_t *> model;        // [rank]
    std::vector<size_t> input_bytes;
    std::filesystem::path neff;
};
} // namespace

class InklingModel final : public Backend {
    ModelOptions options;
    json prepared;
    std::filesystem::path root;
    int tp = 1, hidden = 0, ns = 7, vocab_ = 0, vocab_shard = 0, nvis = 0, pages_total = 0, slots_total = 0, max_context_ = 0;
    std::vector<int> visit_layer;
    std::string visit_kind;
    std::vector<std::vector<float>> visit_ctl;
    std::vector<int> eos_;
    const uint16_t *embed = nullptr;
    size_t embed_map_bytes = 0;
    void *embed_map = nullptr;
    std::vector<Graph> graphs;
    struct Rank {
        std::map<std::string, Dev> t; // "L003.q", "V12.buf", "V05.kv", "V07.ctl", "head.w"
    };
    std::vector<Rank> ranks;
    std::vector<std::unique_ptr<RankWorker>> workers;
    struct Step {
        Dev x[2], pos, wslot;
        std::map<int, Dev> table, rows, logits;
    };
    std::map<std::pair<int, int>, std::vector<Step>> steps;
    std::map<std::tuple<std::string, int, int>, Dev> state_slices; // (state name, slot, rank)
    std::map<std::tuple<const Graph *, int, int, int, std::vector<int64_t>>, std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *>> sets;
    std::mutex sets_mutex;
    std::vector<float> logits;
    json profile = json::object();
    size_t buf_bytes[2] = {0, 0}, conv_bytes = 0; // per slot: buf of s visits / g visits, conv
    std::map<int, int> slot_of;
    std::list<int> lru;
    std::map<int, std::vector<std::vector<char>>> spilled;
    uint64_t spills = 0, restores = 0;
    int launches = 0;
    double exec_ms = 0;
    std::array<double, 4> prof_ms{}; // rank 0: put, layers, head, read

  public:
    explicit InklingModel(const ModelOptions &o) : options(o), root(o.model) {
        prepared = read_json(root / "prepared.json");
        const auto &g = prepared.at("geometry");
        tp = g.at("tp");
        hidden = g.at("hidden");
        ns = g.at("ns");
        vocab_ = g.at("vocab");
        vocab_shard = (vocab_ + tp - 1) / tp;
        for (const auto &v : prepared.at("visits")) {
            visit_layer.push_back(v.at("layer"));
            visit_kind += v.at("kind").get<std::string>();
            visit_ctl.push_back(v.at("ctl").get<std::vector<float>>());
        }
        nvis = int(visit_layer.size());
        if (const char *n = std::getenv("BEHEMOTH_NEURON_INKLING_SLOTS"))
            slots_total = std::max(2, std::atoi(n));
        else
            slots_total = 9;
        eos_ = prepared.at("eos").get<std::vector<int>>();
        merge_generation_eos();
        max_context_ = prepared.at("max_context");
        map_embedding();
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
                (name[0] == 'V' ? s : w) += d.bytes;
            std::cerr << "Inkling: " << nvis << " visits, TP " << tp << ", " << graphs.size() << " NEFFs, " << pages_total - 1 << " pages, "
                      << slots_total - 1 << " sequence slots; per core " << double(w) / (1 << 30) << " GiB weights + " << double(s) / (1 << 30)
                      << " GiB state\n";
        }
    }
    ~InklingModel() override {
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
        embed_map = mmap(nullptr, embed_map_bytes, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
        ::close(fd);
        if (embed_map == MAP_FAILED)
            throw std::runtime_error("mmap embed.bin");
        if (e.at("dtype") != "bf16" || e.at("shape").at(1) != hidden || e.at("shape").at(0).get<int>() < vocab_)
            throw std::runtime_error("embed.bin is not bf16 [vocab][hidden]");
        embed = reinterpret_cast<const uint16_t *>(static_cast<const char *>(embed_map) + size_t(e.at("offset")));
    }
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

    void discover() {
        const auto &geo = prepared.at("geometry");
        for (const auto &dir : std::filesystem::directory_iterator(options.neff_dir)) {
            const auto meta_path = dir.path() / "meta.json";
            if (!std::filesystem::exists(meta_path))
                continue;
            const json m = read_json(meta_path);
            const std::string kind = m.value("kind", "");
            if (kind.rfind("inkling_", 0) != 0)
                continue;
            const auto &mg = m.at("geometry");
            for (const char *k : {"hidden", "vocab", "heads", "kv_heads", "head_dim", "window", "rel_extent", "experts", "top_k",
                                  "expert_intermediate", "tp", "ns", "ctl", "visits"})
                if (mg.at(k) != geo.at(k))
                    throw std::runtime_error("NEFF " + dir.path().string() + " was traced for " + std::string(k) + " " + mg.at(k).dump() +
                                             ", the prepared model has " + geo.at(k).dump());
            Graph gr;
            gr.kind = kind.substr(8);
            gr.inputs = m.at("inputs").get<std::vector<std::string>>();
            gr.states = m.at("states").get<std::vector<std::string>>();
            gr.wt = m.value("wt", false);
            gr.neff = dir.path() / "graph.neff";
            if (gr.kind == "head") {
                gr.T = m.at("T");
                gr.S = m.at("S");
            } else {
                for (const auto &k : m.at("kinds"))
                    gr.kinds += k.get<std::string>();
                gr.has_g = gr.kinds.find_first_of("gG") != std::string::npos;
                gr.R = m.at("R");
                gr.Q = m.at("Q");
                gr.P = m.at("P");
                gr.T = gr.R * gr.Q;
                if (gr.Q > 1 && gr.R != 1)
                    throw std::runtime_error("Inkling prefill buckets must have R = 1");
                if (gr.has_g) {
                    gr.pages = m.at("pages");
                    if (pages_total && pages_total != gr.pages)
                        throw std::runtime_error("Inkling NEFFs disagree on the page count");
                    pages_total = gr.pages;
                }
            }
            graphs.push_back(std::move(gr));
        }
        if (graphs.empty())
            throw std::runtime_error("no Inkling NEFFs (meta kind inkling_*) in " + options.neff_dir.string());
        if (!pages_total)
            throw std::runtime_error("Inkling NEFFs need global-layer buckets");
        std::set<std::pair<int, int>> groups;
        for (const auto &g : graphs)
            if (g.kind != "head")
                groups.insert({g.R, g.Q});
        for (auto [R, Q] : groups) {
            for (int v = 0; v < nvis;) {
                const Graph *g = visit_graph(v, R, Q, 1);
                if (!g)
                    throw std::runtime_error("Inkling bucket R=" + std::to_string(R) + " Q=" + std::to_string(Q) + " has no graph for visit " +
                                             std::to_string(v) + " (kind " + visit_kind[size_t(v)] + ")");
                v += int(g->kinds.size());
            }
            if (!head_for(R * Q, 1))
                throw std::runtime_error("Inkling bucket R=" + std::to_string(R) + " Q=" + std::to_string(Q) + " has no head NEFF");
        }
    }
    // the graph running visits v.. for (R, Q) with >= P pages: the longest segment matching the visit kinds, else the visit's own
    const Graph *visit_graph(int v, int R, int Q, int P) const {
        static const bool noseg = std::getenv("BEHEMOTH_NEURON_INKLING_NOSEG") && std::atoi(std::getenv("BEHEMOTH_NEURON_INKLING_NOSEG"));
        const Graph *best = nullptr;
        for (const auto &g : graphs) {
            if (g.kind == "head" || g.R != R || g.Q != Q || (g.has_g && g.P < P) || v + int(g.kinds.size()) > nvis)
                continue;
            if (noseg && g.kinds.size() > 1)
                continue;
            if (visit_kind.compare(size_t(v), g.kinds.size(), g.kinds) != 0)
                continue;
            if (!best || g.kinds.size() > best->kinds.size() || (g.kinds.size() == best->kinds.size() && g.P < best->P))
                best = &g;
        }
        return best;
    }
    int bucket_pages(int R, int Q, int pages) const {
        int best = 0;
        for (const auto &g : graphs)
            if (g.has_g && g.R == R && g.Q == Q && g.P >= pages && (!best || g.P < best))
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
            // per-slot state sizes, from any graph's first row
            for (size_t s = 0; s < gr.states.size(); ++s) {
                const std::string &nm = gr.states[s];
                const std::string leaf = nm.substr(nm.find('.') == std::string::npos ? 0 : nm.find('.') + 1);
                const std::string base = leaf.substr(0, leaf.find('_'));
                const size_t b = gr.input_bytes[gr.inputs.size() + s];
                const int vi = nm.find('.') == std::string::npos ? 0 : std::stoi(nm.substr(1, nm.find('.') - 1));
                const char kind = gr.kinds.empty() ? 's' : gr.kinds[size_t(vi)];
                if (base == "buf")
                    buf_bytes[kind == 'g' || kind == 'G'] = b;
                else if (base == "conv")
                    conv_bytes = b;
            }
            nrt_free_model_tensor_info(info);
            if (options.verbose)
                std::cerr << "Inkling NEFF " << gr.kind << " R=" << gr.R << " Q=" << gr.Q << " P=" << gr.P << " T=" << gr.T << " S=" << gr.S << "\n";
        }
        for (const auto &gr : graphs) {
            if (gr.kind == "head")
                continue;
            auto &v = steps[{gr.R, gr.Q}];
            if (v.empty()) {
                v.resize(size_t(tp));
                for (int r = 0; r < tp; ++r) {
                    auto &s = v[size_t(r)];
                    const size_t T = size_t(gr.T);
                    s.x[0] = alloc(r, T * size_t(ns * hidden) * 4, "x0");
                    s.x[1] = alloc(r, T * size_t(ns * hidden) * 4, "x1");
                    s.pos = alloc(r, T * 8, "pos");
                    s.wslot = alloc(r, T * 8, "wslot");
                    for (const auto &h : graphs)
                        if (h.kind == "head" && h.T == gr.T && !s.rows.count(h.S)) {
                            s.rows[h.S] = alloc(r, size_t(h.S) * 8, "rows");
                            s.logits[h.S] = alloc(r, size_t(h.S) * size_t(vocab_shard) * 4, "logits");
                        }
                }
            }
            if (gr.has_g && !v[0].table.count(gr.P))
                for (int r = 0; r < tp; ++r)
                    v[size_t(r)].table[gr.P] = alloc(r, size_t(gr.R) * size_t(gr.P) * 8, "table");
        }
        if (!buf_bytes[0] || !buf_bytes[1] || !conv_bytes)
            throw std::runtime_error("Inkling: the graph set does not size the s / g buffers and conv states");
    }

    static std::string vname(int v, const std::string &leaf) {
        char b[16];
        std::snprintf(b, sizeof b, "V%02d.", v);
        return b + leaf;
    }
    static std::string lprefix(int l) {
        char b[16];
        std::snprintf(b, sizeof b, "L%03d.", l);
        return b;
    }

    void load_weights() {
        const auto t0 = Clock::now();
        const auto &files = prepared.at("files");
        const bool wt = graphs.front().wt;
        for (const auto &g : graphs)
            if (g.wt != wt)
                throw std::runtime_error("Inkling NEFFs disagree on the weight layout (meta wt)");
        const bool relayout = wt && !prepared.value("wt", false);
        const int L = prepared.at("geometry").at("layers");
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
                        if (std::vector<uint8_t> t; relayout && to_wt(name, e.at("shape"), bytes, reinterpret_cast<const uint8_t *>(data.data() + offset), t))
                            put(d, t.data(), bytes);
                        else
                            put(d, data.data() + offset, bytes);
                        ranks[size_t(r)].t[name] = d;
                    }
                };
                upload("r" + std::to_string(r) + "/head.bin", "head.");
                for (int l = 0; l < L; ++l) {
                    char f[32];
                    std::snprintf(f, sizeof f, "L%03d.bin", l);
                    upload("r" + std::to_string(r) + "/" + f, lprefix(l));
                }
                for (int v = 0; v < nvis; ++v) {
                    const auto &c = visit_ctl[size_t(v)];
                    Dev d = alloc(r, c.size() * 4, vname(v, "ctl"));
                    put(d, c.data(), c.size() * 4);
                    ranks[size_t(r)].t[vname(v, "ctl")] = d;
                    const bool g = visit_kind[size_t(v)] == 'g' || visit_kind[size_t(v)] == 'G';
                    for (auto [leaf, per] : {std::pair<const char *, size_t>{"buf", buf_bytes[g]}, {"conv", conv_bytes}}) {
                        Dev s = alloc(r, per * size_t(slots_total), vname(v, leaf));
                        zero(s);
                        ranks[size_t(r)].t[vname(v, leaf)] = s;
                    }
                    if (g) {
                        const Graph *gg = nullptr;
                        for (const auto &x : graphs)
                            if (x.has_g && !gg)
                                gg = &x;
                        size_t kvb = 0;
                        for (size_t s = 0; s < gg->states.size(); ++s)
                            if (gg->states[s] == "kv" || gg->states[s].ends_with(".kv"))
                                kvb = gg->input_bytes[gg->inputs.size() + s];
                        Dev k = alloc(r, kvb, vname(v, "kv"));
                        zero(k);
                        ranks[size_t(r)].t[vname(v, "kv")] = k;
                    }
                }
            }));
        for (auto &j : jobs)
            j.get();
        if (options.verbose)
            std::cerr << "Inkling: weights resident in " << ms_since(t0) / 1000 << " s" << (relayout ? " (WT relayout at load)" : "") << "\n";
    }

    // tensor sets of graph g running visits v.. with residual parity `parity`, rows bound to `slots`
    std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *> visit_set(const Graph &g, int v, int parity, int r, const std::vector<int64_t> &slots) {
        std::lock_guard lock(sets_mutex);
        if (auto it = sets.find({&g, v, parity, r, slots}); it != sets.end())
            return it->second;
        auto &st = steps.at({g.R, g.Q})[size_t(r)];
        auto &rank = ranks[size_t(r)];
        auto bind = [&](const std::string &name) -> const Dev & {
            if (name == "x") return st.x[parity];
            if (name == "pos") return st.pos;
            if (name == "wslot") return st.wslot;
            if (name == "table") return st.table.at(g.P);
            int vi = v;
            std::string leaf = name;
            if (g.kinds.size() > 1) {
                const size_t dot = name.find('.');
                vi = v + std::stoi(name.substr(1, dot - 1));
                leaf = name.substr(dot + 1);
            }
            std::string full;
            const std::string base = leaf.substr(0, leaf.find('_'));
            if (leaf == "ctl" || leaf == "kv")
                full = vname(vi, leaf);
            else if (base == "buf" || base == "conv") {
                const int row = leaf.find('_') == std::string::npos ? 0 : std::stoi(leaf.substr(leaf.find('_') + 1));
                const int slot = int(slots.at(size_t(row)));
                full = vname(vi, base);
                const Dev &whole = rank.t.at(full);
                auto &sl = state_slices[{full, slot, r}];
                if (!sl.t) {
                    sl.bytes = whole.bytes / size_t(slots_total);
                    check(nrt_tensor_allocate_slice(whole.t, size_t(slot) * sl.bytes, sl.bytes, (full + ".slot").c_str(), &sl.t), "state slice");
                }
                return sl;
            } else
                full = lprefix(visit_layer[size_t(vi)]) + leaf;
            auto it = rank.t.find(full);
            if (it == rank.t.end())
                throw std::runtime_error("Inkling: no tensor " + full + " for the " + g.kind + " NEFF (visit " + std::to_string(vi) + ")");
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
                throw std::runtime_error("Inkling: " + g.kind + " input " + names[i] + " (visit " + std::to_string(v) + ") expects " +
                                         std::to_string(g.input_bytes[i]) + " B, has " + std::to_string(d.bytes));
            check(nrt_add_tensor_to_tensor_set(in, ("input" + std::to_string(i)).c_str(), d.t), "add input");
        }
        check(nrt_add_tensor_to_tensor_set(out, "output0", st.x[1 - parity].t), "add output");
        for (size_t s = 0; s < g.states.size(); ++s)
            check(nrt_add_tensor_to_tensor_set(out, ("output" + std::to_string(s + 1)).c_str(), bind(g.states[s]).t), "add state output");
        return sets[{&g, v, parity, r, slots}] = {in, out};
    }
    std::pair<nrt_tensor_set_t *, nrt_tensor_set_t *> head_set(const Graph &h, int R, int Q, int r, int parity) {
        std::lock_guard lock(sets_mutex);
        const std::vector<int64_t> key{R, Q};
        if (auto it = sets.find({&h, -1, parity, r, key}); it != sets.end())
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
                throw std::runtime_error("Inkling: head input " + n + " size mismatch");
            check(nrt_add_tensor_to_tensor_set(in, ("input" + std::to_string(i)).c_str(), d.t), "add head input");
        }
        check(nrt_add_tensor_to_tensor_set(out, "output0", st.logits.at(h.S).t), "add head output");
        return sets[{&h, -1, parity, r, key}] = {in, out};
    }

    // ---- sequence slots ----------------------------------------------------------------------------------------------
    void copy_slot(int seq, int slot, bool save) {
        auto &blob = spilled[seq];
        if (save)
            blob.assign(size_t(tp), {});
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                size_t at = 0;
                for (int v = 0; v < nvis; ++v)
                    for (const char *leaf : {"buf", "conv"}) {
                        const Dev &d = ranks[size_t(r)].t.at(vname(v, leaf));
                        const size_t n = d.bytes / size_t(slots_total);
                        if (save) {
                            blob[size_t(r)].resize(at + n);
                            check(nrt_tensor_read(d.t, blob[size_t(r)].data() + at, size_t(slot) * n, n), "read state");
                        } else
                            check(nrt_tensor_write(d.t, blob[size_t(r)].data() + at, size_t(slot) * n, n), "write state");
                        at += n;
                    }
            }));
        for (auto &j : jobs)
            j.get();
        if (!save)
            spilled.erase(seq);
    }
    int acquire_slot(int seq, int position, std::set<int> &pinned) {
        if (auto it = slot_of.find(seq); it != slot_of.end()) {
            if (position == 0)
                spilled.erase(seq);
            lru.remove(seq);
            lru.push_back(seq);
            pinned.insert(it->second);
            return it->second;
        }
        if (position > 0 && !spilled.count(seq))
            throw std::runtime_error("Inkling: no state for the sequence on page " + std::to_string(seq) + " at position " + std::to_string(position));
        std::set<int> used;
        for (auto [s, sl] : slot_of)
            used.insert(sl);
        int slot = -1;
        for (int sl = 0; sl < slots_total - 1 && slot < 0; ++sl)
            if (!used.count(sl) && !pinned.count(sl))
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
                throw std::runtime_error("Inkling: more sequences in one launch than state slots");
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
    void state_drop(int seq) override {
        if (auto it = slot_of.find(seq); it != slot_of.end()) {
            slot_of.erase(it);
            lru.remove(seq);
        }
        spilled.erase(seq);
    }

    // ---- one launch ------------------------------------------------------------------------------------------------
    struct Launch {
        int R = 0, Q = 0, P = 0;
        std::vector<float> x;
        std::vector<int64_t> pos, wslot, table, slot;
        std::vector<int> head_rows, destination;
    };
    Launch fresh(int R, int Q, int P) const {
        Launch L;
        L.R = R; L.Q = Q; L.P = P;
        const size_t T = size_t(R * Q);
        L.x.assign(T * size_t(ns * hidden), 0.f);
        L.pos.assign(T, -1);
        L.wslot.assign(T, int64_t(pages_total - 1) * PAGE);
        L.table.assign(size_t(R) * size_t(P), pages_total - 1);
        L.slot.assign(size_t(R), slots_total - 1);
        return L;
    }
    void fill_token(Launch &L, int t, int token, int position, int page) const {
        if (token < 0 || token >= vocab_)
            throw std::runtime_error("token outside the vocabulary");
        float *x = L.x.data() + size_t(t) * size_t(ns * hidden);
        const uint16_t *e = embed + size_t(token) * size_t(hidden);
        for (int h = 0; h < hidden; ++h)
            x[h] = x[4 * hidden + h] = bf16_to_f32(e[h]);
        L.pos[size_t(t)] = position;
        L.wslot[size_t(t)] = int64_t(page) * PAGE + position % PAGE;
    }
    void execute(Launch &L, float *dest) {
        const Graph *head = nullptr;
        if (!L.head_rows.empty()) {
            head = head_for(L.R * L.Q, int(L.head_rows.size()));
            if (!head)
                throw std::runtime_error("Inkling: no head NEFF for " + std::to_string(L.head_rows.size()) + " rows");
        }
        std::vector<std::pair<const Graph *, int>> plan;
        for (int v = 0; v < nvis;) {
            const Graph *g = visit_graph(v, L.R, L.Q, L.P);
            if (!g)
                throw std::runtime_error("Inkling: bucket R=" + std::to_string(L.R) + " Q=" + std::to_string(L.Q) + " P=" + std::to_string(L.P) +
                                         " has no graph for visit " + std::to_string(v));
            plan.push_back({g, v});
            v += int(g->kinds.size());
        }
        std::map<int, std::vector<int64_t>> tables;
        for (const auto &[g, v] : plan)
            if (g->has_g && !tables.count(g->P)) {
                auto &t = tables[g->P];
                t.assign(size_t(L.R) * size_t(g->P), pages_total - 1);
                for (int r = 0; r < L.R; ++r)
                    std::copy_n(L.table.begin() + std::ptrdiff_t(r * L.P), std::min(L.P, g->P), t.begin() + std::ptrdiff_t(r * g->P));
            }
        std::vector<int64_t> rows(head ? size_t(head->S) : 0, 0);
        for (size_t i = 0; i < L.head_rows.size(); ++i)
            rows[i] = L.head_rows[i];
        static const int dump_x = std::getenv("NEURON_X_DUMP") ? std::atoi(std::getenv("NEURON_X_DUMP")) : 0;
        const auto te0 = Clock::now();
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < tp; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                auto &st = steps.at({L.R, L.Q})[size_t(r)];
                const auto t0 = Clock::now();
                put(st.x[0], L.x.data(), L.x.size() * 4);
                put(st.pos, L.pos.data(), L.pos.size() * 8);
                put(st.wslot, L.wslot.data(), L.wslot.size() * 8);
                for (const auto &[width, t] : tables)
                    put(st.table.at(width), t.data(), t.size() * 8);
                const auto t1 = Clock::now();
                int parity = 0;
                std::vector<float> xr;
                for (const auto &[g, v] : plan) {
                    auto [in, o] = visit_set(*g, v, parity, r, L.slot);
                    check(nrt_execute(g->model[size_t(r)], in, o), g->kind + " execute");
                    parity ^= 1;
                    if (dump_x && r == 0) {
                        xr.resize(L.x.size());
                        check(nrt_tensor_read(st.x[parity].t, xr.data(), 0, xr.size() * 4), "read x");
                        std::string line;
                        bool bad = false;
                        for (int t = 0; t < L.R * L.Q; ++t) {
                            float m = 0;
                            bool fin = true;
                            for (int h = 0; h < hidden; ++h) {
                                const float val = xr[size_t(t) * size_t(ns * hidden) + size_t(4 * hidden + h)];
                                if (!std::isfinite(val)) fin = false; else m = std::max(m, std::fabs(val));
                            }
                            bad |= !fin;
                            line += (fin ? " " : " NONFINITE ") + std::to_string(m);
                        }
                        if (bad || dump_x > 1)
                            std::cerr << "x after visit " << v << " (" << g->kind << ", pos " << L.pos[0] << "):" << line << "\n";
                    }
                }
                const auto t2 = Clock::now();
                if (head) {
                    put(st.rows.at(head->S), rows.data(), rows.size() * 8);
                    auto [in, o] = head_set(*head, L.R, L.Q, r, parity);
                    check(nrt_execute(head->model[size_t(r)], in, o), "head execute");
                    const auto t3 = Clock::now();
                    const size_t first = size_t(r) * size_t(vocab_shard);
                    if (first < size_t(vocab_)) {
                        const size_t n = std::min(size_t(vocab_) - first, size_t(vocab_shard));
                        for (size_t i = 0; i < L.head_rows.size(); ++i)
                            check(nrt_tensor_read(st.logits.at(head->S).t, dest + size_t(L.destination[i]) * size_t(vocab_) + first,
                                                  i * size_t(vocab_shard) * 4, n * 4), "read logits");
                    }
                    if (r == 0) {
                        prof_ms[2] += std::chrono::duration<double, std::milli>(t3 - t2).count();
                        prof_ms[3] += ms_since(t3);
                    }
                }
                if (r == 0) {
                    prof_ms[0] += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    prof_ms[1] += std::chrono::duration<double, std::milli>(t2 - t1).count();
                }
            }));
        for (auto &j : jobs)
            j.get();
        exec_ms += ms_since(te0);
        launches += int(plan.size()) + (head ? 1 : 0);
    }

    int cache_pages() const override { return pages_total - 1; }
    int vocab() const override { return vocab_; }
    int max_decode_rows() const override {
        int r = 0;
        for (const auto &g : graphs)
            if (g.kind != "head" && g.Q == 1) r = std::max(r, g.R);
        return std::min(r, slots_total - 1);
    }
    int decode_bucket_pages(int pages) const override {
        int best = 0;
        for (const auto &g : graphs)
            if (g.has_g && g.Q == 1 && g.P >= pages && (!best || g.P < best)) best = g.P;
        return best;
    }
    const std::vector<int> &eos() const override { return eos_; }
    int max_context() const override {
        int p = 0;
        for (const auto &g : graphs)
            if (g.has_g) p = std::max(p, g.P);
        return std::min(p * PAGE, max_context_);
    }
    json last_profile() const override { return profile; }

    std::span<const float> forward(const InferenceBatch &batch) override {
        const auto t0 = Clock::now();
        exec_ms = 0;
        prof_ms = {};
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
        std::vector<int> rows, multi;
        for (int req = 0; req < lay.requests(); ++req) {
            const int count = lay.query_offsets[size_t(req) + 1] - lay.query_offsets[size_t(req)];
            (count == 1 ? rows : multi).push_back(req);
        }
        std::set<int> decode_R, prefill_Q;
        for (const auto &g : graphs)
            if (g.kind != "head") (g.Q == 1 ? decode_R.insert(g.R) : prefill_Q.insert(g.Q));
        const int maxR = std::min(decode_R.empty() ? 0 : *decode_R.rbegin(), slots_total - 1);
        for (size_t at = 0; at < rows.size();) {
            if (!maxR)
                throw std::runtime_error("Inkling: no decode buckets");
            size_t n = std::min(rows.size() - at, size_t(maxR));
            int pages = 1;
            for (size_t i = 0; i < n; ++i)
                pages = std::max(pages, pages_needed(rows[at + i]));
            int R = 0, P = 0;
            for (auto it = decode_R.lower_bound(int(n)); it != decode_R.end() && !P; ++it)
                if (*it <= slots_total - 1 && (P = bucket_pages(*it, 1, pages)))
                    R = *it;
            if (!P)
                throw std::runtime_error("Inkling: no decode bucket for " + std::to_string(n) + " rows with " + std::to_string(pages) + " pages");
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
            at += n;
        }
        for (int req : multi) {
            const int first = lay.query_offsets[size_t(req)], count = lay.query_offsets[size_t(req) + 1] - first;
            for (int done = 0; done < count;) {
                const int left = count - done;
                int Q = prefill_Q.empty() ? 1 : *prefill_Q.rbegin();
                if (auto it = prefill_Q.lower_bound(left); it != prefill_Q.end()) Q = *it;
                const int take = std::min(left, Q);
                const int PQ = bucket_pages(Q == 1 ? 1 : 1, Q, pages_needed(req));
                if (!PQ)
                    throw std::runtime_error("Inkling: no prefill bucket Q=" + std::to_string(Q) + " with " + std::to_string(pages_needed(req)) + " pages");
                auto L = fresh(1, Q, PQ);
                std::set<int> pinned;
                for (int i = 0; i < take; ++i) {
                    const int ti = first + done + i, position = lay.positions[size_t(ti)];
                    fill_token(L, i, batch.tokens[size_t(ti)], position, page_at(req, position / PAGE));
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
        profile = {{"launches", launches}, {"forward_ms", ms_since(t0)}, {"exec_ms", exec_ms},
                   {"r0", {{"put", prof_ms[0]}, {"layers", prof_ms[1]}, {"head", prof_ms[2]}, {"read", prof_ms[3]}}},
                   {"tokens", batch.tokens.size()}, {"spills", spills}, {"restores", restores}};
        return logits;
    }
};

bool is_inkling_prepared(const std::filesystem::path &p) {
    const auto f = p / "prepared.json";
    if (!std::filesystem::exists(f))
        return false;
    return read_json(f).value("format", "") == "inkling2-neuron-prepared";
}
std::unique_ptr<Backend> make_inkling_backend(const ModelOptions &o) { return std::make_unique<InklingModel>(o); }
} // namespace neuron
