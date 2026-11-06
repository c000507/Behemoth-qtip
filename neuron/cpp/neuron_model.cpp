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
constexpr int TP = 2, DIM = 128;
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

struct Meta {
    std::string kind;
    int R = 0, Q = 0, P = 0, S = 0, T = 0, L = 0; // L > 0: whole-model graph (L layers + head, kind "model")
    std::filesystem::path neff;
};

// Layer NEFF input order after torch_neuronx moved the aliased KV state to the end.
constexpr int kInputs = 23;
enum In { X, COS_SIN, POS, WSLOT, TABLE, NORM1, NORM2, QF, QS, OF, OS, KF, KS, VF, VS, GF, GS, UF, US, DF, DS, CK, CV };

struct LayerWeights {
    Dev norm1, norm2, w[14], ck, cv; // w: qf qs of os kf ks vf vs gf gs uf us df ds
};

struct RankState {
    int core = 0;
    std::vector<LayerWeights> layers;
    Dev head_w, head_norm;
};

struct LayerBucket {
    Meta meta;
    std::array<nrt_model_t *, TP> model{};
    struct PerRank {
        Dev x[2], cos_sin, pos, wslot, table;
        std::vector<std::array<nrt_tensor_set_t *, 2>> in, out; // [layer][parity]; whole-model buckets: [0][0]
        Dev rows, logits;                                      // whole-model buckets only
    };
    std::array<PerRank, TP> rank;
    int tokens() const { return meta.R * meta.Q; }
    bool full() const { return meta.L > 0; }
};
struct HeadBucket {
    Meta meta;
    std::array<nrt_model_t *, TP> model{};
    struct PerRank {
        Dev rows, logits;
        // by the layer bucket whose residual buffer feeds the head, then by parity of the buffer holding the final x
        std::map<const LayerBucket *, std::array<nrt_tensor_set_t *, 2>> in;
        std::array<nrt_tensor_set_t *, 2> out{};
    };
    std::array<PerRank, TP> rank;
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
    int vocab_ = 0, vocab_half = 0, pages_total = 0, kv_local = 0;
    std::array<RankState, TP> ranks;
    std::vector<LayerBucket> decode, prefill;
    std::vector<HeadBucket> heads;
    std::vector<float> logits;
    std::vector<std::unique_ptr<RankWorker>> workers;
    nlohmann::json profile = nlohmann::json::object();
    double phase_upload_ms = 0, phase_layers_ms = 0, phase_head_ms = 0; // rank 0, summed over the launches of a forward

    explicit Impl(const ModelOptions &o) : options(o), config(ModelConfig::read(o.model / "config.json")) {
        if (config.hidden != 4096 || config.intermediate != 14336 || config.q_heads != 32 || config.kv_heads != 8 || config.dim != DIM)
            throw std::runtime_error("the Neuron graphs are traced for the Llama 3.1 8B geometry (trace_model.py constants)");
        if (const char *n = std::getenv("NEURON_LAYERS")) // debugging: run only the first n layers (and whole-model NEFFs traced with L=n)
            config.layers = std::min(config.layers, std::atoi(n));
        vocab_ = config.vocab;
        vocab_half = (vocab_ + TP - 1) / TP;
        kv_local = config.kv_heads / TP;
        check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init");
        for (int r = 0; r < TP; ++r) {
            ranks[size_t(r)].core = r;
            workers.push_back(std::make_unique<RankWorker>());
        }
        discover();
        auto needs_head = [](const std::vector<LayerBucket> &v) { return std::any_of(v.begin(), v.end(), [](const auto &b) { return !b.full(); }); };
        if (decode.empty() || prefill.empty() || ((needs_head(decode) || needs_head(prefill)) && heads.empty()))
            throw std::runtime_error("need decode and prefill NEFF buckets (and head NEFFs for per-layer buckets) in " + options.neff_dir.string());
        embed_file = std::make_unique<SafeTensorFile>(o.model / "model-embed.safetensors");
        embed = embed_file->floats("model.embed_tokens.weight");
        final_norm = embed_file->floats("model.norm.weight");
        if (config.rope_llama3)
            inv_freq = llama3_inv_freq(DIM, config.theta, config.rope);
        else {
            inv_freq.resize(DIM / 2);
            for (int d = 0; d < DIM / 2; ++d)
                inv_freq[size_t(d)] = float(std::pow(double(config.theta), -2.0 * d / DIM));
        }
        load_buckets();
        load_weights();
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
            if (m.kind == "layer" || m.kind == "model") {
                if (m.kind == "model" && std::getenv("NEURON_USE_LAYERS"))
                    continue;
                m.R = j.at("R"); m.Q = j.at("Q"); m.P = j.at("P");
                if (m.kind == "model") {
                    m.L = j.at("L"); m.S = j.at("S");
                    if (m.L != config.layers)
                        continue; // a partial-depth graph (compile-time probe)
                }
                LayerBucket b;
                b.meta = m;
                (m.Q == 1 ? decode : prefill).push_back(std::move(b));
            } else {
                m.S = j.at("S"); m.T = j.at("T");
                HeadBucket b;
                b.meta = m;
                heads.push_back(std::move(b));
            }
        }
        auto by_size = [](const auto &a, const auto &b) { return a.meta.P != b.meta.P ? a.meta.P < b.meta.P : a.meta.Q != b.meta.Q ? a.meta.Q < b.meta.Q : a.meta.L > b.meta.L; };
        std::sort(decode.begin(), decode.end(), by_size);
        std::sort(prefill.begin(), prefill.end(), by_size);
        std::sort(heads.begin(), heads.end(), [](const auto &a, const auto &b) { return a.meta.T != b.meta.T ? a.meta.T < b.meta.T : a.meta.S < b.meta.S; });
    }

    // Load every NEFF on both cores and create its step tensors. Tensor sets are built after the weights exist.
    void load_buckets() {
        auto load_layer = [&](LayerBucket &b) {
            const auto neff = slurp(b.meta.neff);
            std::vector<std::future<void>> jobs;
            for (int r = 0; r < TP; ++r)
                jobs.push_back(workers[size_t(r)]->submit([&, r] {
                    check(nrt_load_collectives(neff.data(), neff.size(), r, -1, uint32_t(r), TP, &b.model[size_t(r)]), "nrt_load_collectives");
                    nrt_tensor_info_array_t *info = nullptr;
                    check(nrt_get_model_tensor_info(b.model[size_t(r)], &info), "tensor info");
                    if (options.verbose && r == 0) {
                        std::cerr << "layer NEFF R=" << b.meta.R << " Q=" << b.meta.Q << " P=" << b.meta.P << ":";
                        for (uint64_t i = 0; i < info->tensor_count; ++i)
                            std::cerr << ' ' << info->tensor_array[i].name << '(' << info->tensor_array[i].size << ')';
                        std::cerr << '\n';
                    }
                    int ins = 0;
                    for (uint64_t i = 0; i < info->tensor_count; ++i)
                        if (info->tensor_array[i].usage == NRT_TENSOR_USAGE_INPUT) {
                            const auto &t = info->tensor_array[i];
                            const size_t index = std::stoul(std::string(t.name).substr(5));
                            if (index == size_t(b.full() ? 8 + 16 * b.meta.L : int(CK))) {
                                const int pages = int(t.size / size_t(kv_local * 16 * DIM * 2));
                                if (pages_total && pages != pages_total)
                                    throw std::runtime_error("layer NEFFs disagree on the page count");
                                pages_total = pages;
                            }
                            ++ins;
                        }
 if (ins != (b.full() ? 8 + 18 * b.meta.L : kInputs))
                        throw std::runtime_error("layer NEFF has " + std::to_string(ins) + " inputs, expected " + std::to_string(kInputs));
                    auto &pr = b.rank[size_t(r)];
                    const size_t T = size_t(b.tokens());
                    pr.x[0] = alloc(r, T * 4096 * 4, "x0");
                    pr.x[1] = alloc(r, T * 4096 * 4, "x1");
                    pr.cos_sin = alloc(r, T * 2 * (DIM / 2) * 4, "cos_sin");
                    pr.pos = alloc(r, T * 8, "pos");
                    pr.wslot = alloc(r, T * 8, "wslot");
                    pr.table = alloc(r, size_t(b.meta.R) * size_t(b.meta.P) * 8, "table");
                    if (b.full()) {
                        pr.rows = alloc(r, size_t(b.meta.S) * 8, "rows");
                        pr.logits = alloc(r, size_t(b.meta.S) * size_t(vocab_half) * 4, "logits");
                    }
                }));
            for (auto &j : jobs)
                j.get();
        };
        for (auto &b : decode) load_layer(b);
        for (auto &b : prefill) load_layer(b);
        for (auto &h : heads) {
            const auto neff = slurp(h.meta.neff);
            for (int r = 0; r < TP; ++r) {
                check(nrt_load(neff.data(), neff.size(), r, 1, &h.model[size_t(r)]), "nrt_load head");
                h.rank[size_t(r)].rows = alloc(r, size_t(h.meta.S) * 8, "rows");
                h.rank[size_t(r)].logits = alloc(r, size_t(h.meta.S) * size_t(vocab_half) * 4, "logits");
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
        std::vector<float> n1(4096), n2(4096);
        {
            auto a = file.floats(base + "input_layernorm.weight"), b = file.floats(base + "post_attention_layernorm.weight");
            for (size_t i = 0; i < 4096; ++i) { n1[i] = a[i]; n2[i] = b[i]; }
        }
        std::vector<uint8_t> zeros(size_t(pages_total) * size_t(kv_local) * 16 * DIM * 2, 0);
        for (int r = 0; r < TP; ++r) {
            auto &lw = ranks[size_t(r)].layers[size_t(layer)];
            lw.norm1 = alloc(r, 4096 * 4, "norm1");
            lw.norm2 = alloc(r, 4096 * 4, "norm2");
            put(lw.norm1, n1.data(), 4096 * 4);
            put(lw.norm2, n2.data(), 4096 * 4);
            lw.ck = alloc(r, zeros.size(), "cache_k");
            lw.cv = alloc(r, zeros.size(), "cache_v");
            put(lw.ck, zeros.data(), zeros.size());
            put(lw.cv, zeros.data(), zeros.size());
        }
        for (const auto &p : projs) {
            auto shards = load_projection_shards(file, layer, p.name, config, TP, p.col);
            for (int r = 0; r < TP; ++r) {
                auto &lw = ranks[size_t(r)].layers[size_t(layer)];
                const auto &s = shards[size_t(r)];
                lw.w[p.slot] = alloc(r, s.w.size(), p.name);
                lw.w[p.slot + 1] = alloc(r, s.scale.size() * 4, p.name);
                put(lw.w[p.slot], s.w.data(), s.w.size());
                put(lw.w[p.slot + 1], s.scale.data(), s.scale.size() * 4);
            }
        }
    }

    void load_weights() {
        const auto t0 = Clock::now();
        for (auto &rs : ranks)
            rs.layers.resize(size_t(config.layers));
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
        // lm_head halves (bf16, rows padded to an even vocabulary) and the final norm
        const auto head = embed_file->bytes("lm_head.weight");
        std::vector<float> norm(4096);
        for (size_t i = 0; i < 4096; ++i)
            norm[i] = final_norm[i];
        for (int r = 0; r < TP; ++r) {
            auto &rs = ranks[size_t(r)];
            rs.head_w = alloc(r, size_t(vocab_half) * 4096 * 2, "lm_head");
            rs.head_norm = alloc(r, 4096 * 4, "final_norm");
            put(rs.head_norm, norm.data(), 4096 * 4);
            const size_t first = size_t(r) * size_t(vocab_half), last = std::min(size_t(vocab_), first + size_t(vocab_half));
            zero(rs.head_w);
            if (last > first)
                put(rs.head_w, head.data() + first * 4096 * 2, (last - first) * 4096 * 2);
        }
        build_sets();
        if (options.verbose)
            std::cerr << "model resident on " << TP << " NeuronCores in " << ms_since(t0) / 1e3 << " s, " << pages_total - 1 << " KV pages\n";
    }

    void build_sets() {
        // Whole-model bucket: one tensor set. Inputs: step tensors + head, then 16 tensors per layer, then the 2L KV state
        // tensors (K, V interleaved per layer); outputs: logits, then the same KV tensors (updated in place).
        auto build_full = [&](LayerBucket &b) {
            for (int r = 0; r < TP; ++r) {
                auto &pr = b.rank[size_t(r)];
                nrt_tensor_info_array_t *info = nullptr;
                check(nrt_get_model_tensor_info(b.model[size_t(r)], &info), "tensor info");
                std::vector<const Dev *> by_index = {&pr.x[0], &pr.cos_sin, &pr.pos, &pr.wslot, &pr.table, &pr.rows, &ranks[size_t(r)].head_norm,
                                                     &ranks[size_t(r)].head_w};
                for (int l = 0; l < b.meta.L; ++l) {
                    const auto &lw = ranks[size_t(r)].layers[size_t(l)];
                    by_index.push_back(&lw.norm1);
                    by_index.push_back(&lw.norm2);
                    for (const auto &w : lw.w) by_index.push_back(&w);
                }
                std::vector<const Dev *> state;
                for (int l = 0; l < b.meta.L; ++l) {
                    state.push_back(&ranks[size_t(r)].layers[size_t(l)].ck);
                    state.push_back(&ranks[size_t(r)].layers[size_t(l)].cv);
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
                        check(nrt_add_tensor_to_tensor_set(out, t.name, pr.logits.t), "add output");
                    } else {
                        if (extra >= state.size() || state[extra]->bytes != t.size)
                            throw std::runtime_error("whole-model output " + name + " does not match a KV tensor");
                        check(nrt_add_tensor_to_tensor_set(out, t.name, state[extra++]->t), "add kv output");
                    }
                }
                pr.in.assign(1, {in, in});
                pr.out.assign(1, {out, out});
            }
        };
        auto build = [&](LayerBucket &b) {
            if (b.full()) {
                build_full(b);
                return;
            }
            for (int r = 0; r < TP; ++r) {
                auto &pr = b.rank[size_t(r)];
                nrt_tensor_info_array_t *info = nullptr;
                check(nrt_get_model_tensor_info(b.model[size_t(r)], &info), "tensor info");
                pr.in.resize(ranks[size_t(r)].layers.size());
                pr.out.resize(pr.in.size());
                for (size_t l = 0; l < pr.in.size(); ++l)
                    for (int p = 0; p < 2; ++p) {
                        const auto &lw = ranks[size_t(r)].layers[l];
                        nrt_tensor_set_t *in = nullptr, *out = nullptr;
                        check(nrt_allocate_tensor_set(&in), "set");
                        check(nrt_allocate_tensor_set(&out), "set");
                        const Dev *by_index[kInputs] = {&pr.x[p], &pr.cos_sin, &pr.pos, &pr.wslot, &pr.table, &lw.norm1, &lw.norm2,
                                                        &lw.w[0], &lw.w[1], &lw.w[2], &lw.w[3], &lw.w[4], &lw.w[5], &lw.w[6], &lw.w[7],
                                                        &lw.w[8], &lw.w[9], &lw.w[10], &lw.w[11], &lw.w[12], &lw.w[13], &lw.ck, &lw.cv};
                        int extra = 0;
                        for (uint64_t i = 0; i < info->tensor_count; ++i) {
                            const auto &t = info->tensor_array[i];
                            const std::string name = t.name;
                            if (t.usage == NRT_TENSOR_USAGE_INPUT) {
                                const size_t index = std::stoul(name.substr(5));
                                if (by_index[index]->bytes != t.size)
                                    throw std::runtime_error("input " + name + " expects " + std::to_string(t.size) + " B, host has " +
                                                             std::to_string(by_index[index]->bytes));
                                check(nrt_add_tensor_to_tensor_set(in, t.name, by_index[index]->t), "add input");
                            } else if (name == "output0") {
                                check(nrt_add_tensor_to_tensor_set(out, t.name, pr.x[1 - p].t), "add output");
                            } else { // extra outputs: the updated KV state, written back into the same device buffers
                                const Dev &kv = extra++ == 0 ? lw.ck : lw.cv;
                                if (kv.bytes != t.size)
                                    throw std::runtime_error("unexpected extra output " + name);
                                check(nrt_add_tensor_to_tensor_set(out, t.name, kv.t), "add kv output");
                            }
                        }
                        pr.in[l][size_t(p)] = in;
                        pr.out[l][size_t(p)] = out;
                    }
            }
        };
        for (auto &b : decode) build(b);
        for (auto &b : prefill) build(b);
        // head sets: input0 = the residual buffer of each layer bucket with the same token count, rows, norm, w -> logits.
        for (auto &h : heads)
            for (int r = 0; r < TP; ++r) {
                auto &hr = h.rank[size_t(r)];
                for (int p = 0; p < 2; ++p) {
                    check(nrt_allocate_tensor_set(&hr.out[size_t(p)]), "set");
                    check(nrt_add_tensor_to_tensor_set(hr.out[size_t(p)], "output0", hr.logits.t), "head out");
                }
                for (const auto *group : {&decode, &prefill})
                    for (const auto &b : *group) {
                        if (b.full() || b.tokens() != h.meta.T)
                            continue;
                        auto &sets = hr.in[&b];
                        for (int p = 0; p < 2; ++p) {
                            check(nrt_allocate_tensor_set(&sets[size_t(p)]), "set");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input0", b.rank[size_t(r)].x[p].t), "head in");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input1", hr.rows.t), "head in");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input2", ranks[size_t(r)].head_norm.t), "head in");
                            check(nrt_add_tensor_to_tensor_set(sets[size_t(p)], "input3", ranks[size_t(r)].head_w.t), "head in");
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
        for (size_t i = 0; i < 4096; ++i)
            out[i] = embed[size_t(token) * 4096 + i];
    }
    void rope_row(int position, float *out) const { // [cos x64, sin x64] from fp64 angles, any position
        for (int d = 0; d < DIM / 2; ++d) {
            const double angle = double(position) * double(inv_freq[size_t(d)]);
            out[d] = float(std::cos(angle));
            out[DIM / 2 + d] = float(std::sin(angle));
        }
    }

    void fill_row(Launch &L, int row, int token, int position, int page_id) {
        embed_row(token, L.x.data() + size_t(row) * 4096);
        rope_row(position, L.cos_sin.data() + size_t(row) * DIM);
        L.pos[size_t(row)] = position;
        L.wslot[size_t(row)] = int64_t(page_id) * 16 + position % 16;
    }

    void execute(Launch &L, float *dest) {
        const auto &b = *L.bucket;
        if (b.full()) {
            if (L.head_rows.size() > size_t(b.meta.S))
                throw std::runtime_error("more sampled rows than the whole-model NEFF produces");
            std::vector<int64_t> rows(size_t(b.meta.S), 0);
            for (size_t i = 0; i < L.head_rows.size(); ++i)
                rows[i] = L.head_rows[i];
            std::vector<std::vector<float>> out(TP);
            std::vector<std::future<void>> jobs;
            for (int r = 0; r < TP; ++r)
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
                    check(nrt_execute(b.model[size_t(r)], pr.in[0][0], pr.out[0][0]), "whole-model execute");
                    const auto t_executed = Clock::now();
                    out[size_t(r)].resize(size_t(b.meta.S) * size_t(vocab_half));
                    check(nrt_tensor_read(pr.logits.t, out[size_t(r)].data(), 0, out[size_t(r)].size() * 4), "read logits");
                    if (r == 0) {
                        phase_upload_ms += std::chrono::duration<double, std::milli>(t_uploaded - t_start).count();
                        phase_layers_ms += std::chrono::duration<double, std::milli>(t_executed - t_uploaded).count();
                        phase_head_ms += ms_since(t_executed);
                    }
                }));
            for (auto &j : jobs)
                j.get();
            for (size_t i = 0; i < L.head_rows.size(); ++i) {
                float *row = dest + size_t(L.destination[i]) * size_t(vocab_);
                for (int r = 0; r < TP; ++r) {
                    const size_t first = size_t(r) * size_t(vocab_half), count = std::min(size_t(vocab_) - first, size_t(vocab_half));
                    std::copy_n(out[size_t(r)].data() + i * size_t(vocab_half), count, row + first);
                }
            }
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
        const int layers = config.layers, final_parity = layers % 2;
        std::vector<int64_t> rows(head ? size_t(head->meta.S) : 0, 0);
        for (size_t i = 0; i < L.head_rows.size(); ++i)
            rows[i] = L.head_rows[i];
        std::vector<std::vector<float>> out(TP);
        std::vector<std::future<void>> jobs;
        for (int r = 0; r < TP; ++r)
            jobs.push_back(workers[size_t(r)]->submit([&, r] {
                auto &pr = const_cast<LayerBucket &>(b).rank[size_t(r)];
                const auto t_start = Clock::now();
                put(pr.x[0], L.x.data(), L.x.size() * 4);
                put(pr.cos_sin, L.cos_sin.data(), L.cos_sin.size() * 4);
                put(pr.pos, L.pos.data(), L.pos.size() * 8);
                put(pr.wslot, L.wslot.data(), L.wslot.size() * 8);
                put(pr.table, L.table.data(), L.table.size() * 8);
                const auto t_uploaded = Clock::now();
                for (int l = 0; l < layers; ++l)
                    check(nrt_execute(b.model[size_t(r)], pr.in[size_t(l)][size_t(l % 2)], pr.out[size_t(l)][size_t(l % 2)]), "layer execute");
                const auto t_layers = Clock::now();
                if (r == 0) {
                    phase_upload_ms += std::chrono::duration<double, std::milli>(t_uploaded - t_start).count();
                    phase_layers_ms += std::chrono::duration<double, std::milli>(t_layers - t_uploaded).count();
                }
                if (head) {
                    auto &hr = const_cast<HeadBucket *>(head)->rank[size_t(r)];
                    put(hr.rows, rows.data(), rows.size() * 8);
                    check(nrt_execute(head->model[size_t(r)], hr.in.at(&b)[size_t(final_parity)], hr.out[size_t(final_parity)]), "head execute");
                    out[size_t(r)].resize(size_t(head->meta.S) * size_t(vocab_half));
                    check(nrt_tensor_read(hr.logits.t, out[size_t(r)].data(), 0, out[size_t(r)].size() * 4), "read logits");
                    if (r == 0)
                        phase_head_ms += ms_since(t_layers);
                }
            }));
        for (auto &j : jobs)
            j.get();
        if (!head)
            return;
        for (size_t i = 0; i < L.head_rows.size(); ++i) {
            float *row = dest + size_t(L.destination[i]) * size_t(vocab_);
            for (int r = 0; r < TP; ++r) {
                const size_t first = size_t(r) * size_t(vocab_half), count = std::min(size_t(vocab_) - first, size_t(vocab_half));
                std::copy_n(out[size_t(r)].data() + i * size_t(vocab_half), count, row + first);
            }
        }
    }

    std::span<const float> forward(const InferenceBatch &batch) {
        const auto t0 = Clock::now();
        phase_upload_ms = phase_layers_ms = phase_head_ms = 0;
        const auto &lay = batch.layout;
        const int scratch = pages_total - 1;
        logits.assign(batch.sample_rows.size() * size_t(vocab_), 0.f);
        // which request samples, and which logits row it owns
        std::vector<int> sample_index(size_t(lay.requests()), -1);
        for (size_t s = 0, req = 0; s < batch.sample_rows.size(); ++s) {
            while (batch.sample_rows[s] >= lay.query_offsets[req + 1]) ++req;
            sample_index[req] = int(s);
        }
        auto pages_needed = [&](int req) { return (lay.lengths[size_t(req)] + 15) / 16; };
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
            L.x.assign(T * 4096, 0.f);
            L.cos_sin.assign(T * DIM, 0.f);
            L.pos.assign(T, 0);
            L.wslot.assign(T, int64_t(scratch) * 16);
            L.table.assign(size_t(b.meta.R) * size_t(b.meta.P), scratch);
            for (size_t t = 0; t < T; ++t)
                rope_row(0, L.cos_sin.data() + t * DIM);
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
        std::vector<int> single, multi;
        for (int req = 0; req < lay.requests(); ++req)
            (lay.query_offsets[size_t(req) + 1] - lay.query_offsets[size_t(req)] == 1 ? single : multi).push_back(req);
        int launches = 0, max_rows = 0, max_queries = 0;
        for (const auto &b : decode) max_rows = std::max(max_rows, b.meta.R);
        for (const auto &b : prefill) max_queries = std::max(max_queries, b.meta.Q);
        // decode rows (one token each): all of them in one launch when a bucket fits, else in groups of the largest R
        for (size_t at = 0; at < single.size();) {
            size_t n = single.size() - at;
            auto pages_of = [&](size_t count) {
                int pages = 1;
                for (size_t i = 0; i < count; ++i) pages = std::max(pages, pages_needed(single[at + i]));
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
                const int req = single[at + i], token_index = lay.query_offsets[size_t(req)];
                const int position = lay.positions[size_t(token_index)];
                const int page = lay.page_table[size_t(req) * size_t(lay.table_stride) + size_t(position / 16)];
                fill_row(L, int(i), batch.tokens[size_t(token_index)], position, page);
                table_row(L, int(i), req);
                if (sample_index[size_t(req)] >= 0) {
                    L.head_rows.push_back(int(i));
                    L.destination.push_back(sample_index[size_t(req)]);
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
                    const int page = lay.page_table[size_t(req) * size_t(lay.table_stride) + size_t(position / 16)];
                    fill_row(L, i, batch.tokens[size_t(token_index)], position, page);
                }
                table_row(L, 0, req);
                if (done + take == count && sample_index[size_t(req)] >= 0) {
                    L.head_rows.push_back(take - 1);
                    L.destination.push_back(sample_index[size_t(req)]);
                }
                execute(L, logits.data());
                ++launches;
                done += take;
            }
        }
        profile = {{"launches", launches}, {"forward_ms", ms_since(t0)}, {"upload_ms", phase_upload_ms}, {"layers_ms", phase_layers_ms}, {"head_ms", phase_head_ms}, {"tokens", batch.tokens.size()}};
        return logits;
    }
};

NeuronModel::NeuronModel(const ModelOptions &o) : impl_(std::make_unique<Impl>(o)) {}
NeuronModel::~NeuronModel() = default;
int NeuronModel::cache_pages() const { return impl_->pages_total - 1; }
int NeuronModel::vocab() const { return impl_->vocab_; }
const std::vector<int> &NeuronModel::eos() const { return impl_->config.eos; }
int NeuronModel::max_context() const {
    int p = 0;
    for (const auto &b : impl_->decode) p = std::max(p, b.meta.P);
    return std::min(p * 16, impl_->config.max_context);
}
std::span<const float> NeuronModel::forward(const InferenceBatch &b) { return impl_->forward(b); }
nlohmann::json NeuronModel::last_profile() const { return impl_->profile; }
} // namespace neuron
