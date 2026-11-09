#include "device_ops.hpp"
#include "glm53.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
namespace glm53 {
namespace {
using namespace gpu;
// A linear weight kept on its device for the model's lifetime (within the residency budget).
struct Held {
    Buffer<uint8_t> raw, block_scales;
    Buffer<float> scales;
    Matrix matrix{};
};
struct Resident {
    Tensor host;
    Buffer<float> data;
};
struct Context {
    int device;
    Stream stream{};
    Event done{};
    // Shared staging for streamed (non-resident) matrices.
    Buffer<uint8_t> raw, block_scales;
    Buffer<float> scales;
    Pinned<float> readback;
    Pinned<int> read_indices;
    std::map<std::string, Tensor> host_scales;
    std::map<std::string, Held> resident;
    std::map<std::string, Resident> constants;
    size_t resident_budget = 0, resident_used = 0, streamed_bytes = 0;
    int streamed_count = 0;
    explicit Context(int d) : device(d) {
        checked(set_device(d), "select GPU");
#if defined(GLM53_CUDA)
        cudaDeviceProp prop{};
        checked(cudaGetDeviceProperties(&prop, d), "CUDA properties");
        check(prop.major >= 7 && prop.warpSize == 32, "CUDA inference requires sm_70 or newer");
#else
        hipDeviceProp_t prop{};
        checked(hipGetDeviceProperties(&prop, d), "HIP properties");
        check(prop.warpSize == 32, "HIP inference requires wave32 (R9700/gfx1201)");
#endif
        checked(stream_create(&stream), "create GPU stream");
        if (event_create(&done) != success) {
            (void)stream_destroy(stream);
            throw std::runtime_error("create GPU event");
        }
    }
    ~Context() {
        (void)set_device(device);
        (void)sync(stream);
        resident.clear();
        constants.clear();
        raw.clear();
        block_scales.clear();
        scales.clear();
        (void)event_destroy(done);
        (void)stream_destroy(stream);
    }
    void activate() { checked(set_device(device), "select GPU"); }
    size_t free_memory() {
        activate();
        size_t free = 0, total = 0;
        checked(memory_info(&free, &total), "GPU memory info");
        return free;
    }
    const float *constant(const Weights &w, const std::string &name, size_t size) {
        if (!constants.contains(name)) {
            Resident r;
            r.host = w.tensor(name);
            check(r.host.data.size() == size, "GPU constant geometry mismatch: " + name);
            r.data.ensure(size, stream);
            auto it = constants.emplace(name, std::move(r)).first;
            const auto &a = it->second;
            checked(h2d(a.data.p, a.host.data.data(), size * 4, stream), "upload constant");
        }
        check(constants.at(name).host.data.size() == size, "GPU constant size changed");
        return constants.at(name).data;
    }
    // Uploads a linear weight. `keep` requests residency: the matrix stays on the device when the
    // budget allows, otherwise it streams through the shared staging buffers like before.
    Matrix matrix(const Weights &w, const std::string &base, bool keep = true) {
        if (keep)
            if (auto it = resident.find(base); it != resident.end())
                return it->second.matrix;
        bool nv = w.has(base + ".weight_packed");
        auto name = nv ? base + ".weight_packed" : w.has(base) ? base : base + ".weight";
        const auto &f = w.file(name);
        auto shape = f.shape(name);
        check(shape.size() == 2 && shape[0] > 0 && shape[0] <= INT32_MAX && shape[1] > 0 &&
                  shape[1] <= INT32_MAX / 2,
              "GPU matrix geometry outside supported range: " + name);
        int r = int(shape[0]), c = int(shape[1]) * (nv ? 2 : 1);
        auto bytes = f.bytes(name);
        auto dtype = f.dtype(name);
        // Side data: NVFP4 block scales or FP8 scales.
        std::span<const std::byte> side;
        const Tensor *fp8_scales = nullptr;
        Matrix m{nullptr, nullptr, nullptr, r, c, 0, false, 1};
        if (nv) {
            check(dtype == "U8" && c % 16 == 0 && !w.has(base + ".weight"), "invalid NVFP4 matrix");
            m.format = 2;
            auto sn = base + ".weight_scale";
            const auto &sf = w.file(sn);
            check(sf.dtype(sn) == "F8_E4M3" && sf.shape(sn) == std::vector<size_t>{size_t(r), size_t(c / 16)},
                  "invalid NVFP4 block-scale geometry");
            side = sf.bytes(sn);
            auto gs = w.tensor(base + ".weight_global_scale");
            check(gs.data.size() == 1 && gs.data[0] > 0, "invalid NVFP4 global scale");
            m.global = gs.data[0];
        } else if (dtype == "F8_E4M3") {
            m.format = 1;
            m.block = w.has(base + ".weight_scale_inv");
            auto sn = base + (m.block ? ".weight_scale_inv" : ".weight_scale");
            if (!host_scales.contains(sn))
                host_scales.emplace(sn, w.tensor(sn));
            fp8_scales = &host_scales.at(sn);
            check(m.block ? fp8_scales->shape ==
                                std::vector<size_t>{size_t((r + 127) / 128), size_t((c + 127) / 128)}
                          : fp8_scales->data.size() == size_t(r),
                  "invalid FP8 scale geometry");
            for (float a : fp8_scales->data)
                check(a >= 0 && std::isfinite(a), "invalid FP8 scale");
        } else if (dtype == "BF16")
            m.format = 4;
        else if (dtype == "F16")
            m.format = 3;
        else
            check(dtype == "F32", "unsupported GPU linear dtype: " + dtype);
        size_t total = bytes.size() + side.size() + (fp8_scales ? fp8_scales->data.size() * 4 : 0);
        Held held;
        bool pin = keep && resident_used + total <= resident_budget;
        Buffer<uint8_t> &raw_target = pin ? held.raw : raw, &side_target = pin ? held.block_scales : block_scales;
        Buffer<float> &scale_target = pin ? held.scales : scales;
        raw_target.ensure(bytes.size(), stream);
        checked(h2d(raw_target, bytes.data(), bytes.size(), stream), "upload matrix");
        m.data = raw_target;
        if (!side.empty()) {
            side_target.ensure(side.size(), stream);
            checked(h2d(side_target, side.data(), side.size(), stream), "upload NVFP4 scales");
            m.block_scales = side_target;
        }
        if (fp8_scales) {
            scale_target.ensure(fp8_scales->data.size(), stream);
            checked(h2d(scale_target, fp8_scales->data.data(), fp8_scales->data.size() * 4, stream),
                    "upload FP8 scales");
            m.scales = scale_target;
        }
        if (pin) {
            // Streamed uploads come from pageable mmap memory and complete before cudaMemcpyAsync returns;
            // a resident copy also outlives this call, so no extra synchronization is needed here.
            held.matrix = m;
            resident.emplace(base, std::move(held));
            resident_used += total;
        } else if (keep) {
            // One-shot uploads (keep == false, e.g. the decoded MLA expansion) are not per-token traffic.
            ++streamed_count;
            streamed_bytes += total;
        }
        return m;
    }
    int linear(const Weights &w, const std::string &base, const B16 *x, size_t width, Buffer<B16> &out) {
        auto m = matrix(w, base);
        check(size_t(m.cols) == width, "GPU linear input width mismatch: " + base);
        out.ensure(m.rows, stream);
        gemv(m, x, out, false, stream);
        return m.rows;
    }
};
int power_of_two(int n) {
    int p = 1;
    while (p < n) {
        check(p <= INT32_MAX / 2, "GPU context too large");
        p *= 2;
    }
    return p;
}
// Per-device activation scratch: every device of a layer split owns one set.
struct Scratch {
    Buffer<B16> x, next, collapsed, normalized, branch, embed;
    Buffer<B16> q, k, v, low, forget, beta, gate, attended, qr, latent, iq, ik, iw, ig;
    Buffer<B16> mlp_gate, mlp_up, expert, logits;
    Buffer<float> post, mix, hc_scratch, query, attention_scores, weighted, pool_scores;
    Buffer<float> router_scores, router_weights, group_scores, accumulated, host_output, upload_embedding;
    Buffer<int> pool_ids, expert_ids, allowed;
};
size_t env_bytes(const char *name, bool &set) {
    const char *value = std::getenv(name);
    set = value != nullptr;
    if (!value)
        return 0;
    double gib = 0;
    try {
        size_t used = 0;
        gib = std::stod(value, &used);
        check(used == std::char_traits<char>::length(value) && gib >= 0 && std::isfinite(gib),
              "invalid value");
    } catch (const std::exception &) {
        throw std::runtime_error(std::string(name) + " must be a non-negative number of GiB");
    }
    return size_t(gib * double(size_t(1) << 30));
}
class DeviceModel final : public InferenceBackend {
    Weights weights;
    Config c;
    std::vector<std::unique_ptr<Context>> contexts;
    std::vector<Scratch> scratch;
    std::vector<int> layer_context; // which context runs each layer
    struct State {
        Buffer<float> recurrent;
        Buffer<B16> convolution, latent, keys, gates, pooled, kv;
        Buffer<int> selected;
        int count = 0;
    };
    std::vector<State> state;
    int position = 0;
    bool failed = false, reported = false;
    Vec embedding_row;
    // Set per layer by step(): the context (device) and its scratch currently in use.
    Context *ctx = nullptr;
    Scratch *sc = nullptr;
    const float *constant(const std::string &name, size_t size) { return ctx->constant(weights, name, size); }
    int linear(const std::string &name, const Buffer<B16> &in, size_t width, Buffer<B16> &out) {
        return ctx->linear(weights, name, in, width, out);
    }
    void rms(const Buffer<B16> &in, Buffer<B16> &out, const std::string &name, int width) {
        out.ensure(width, ctx->stream);
        norm(in, constant(name, width), nullptr, out, width, c.eps, false, ctx->stream);
    }
    void dense(const std::string &p, Buffer<B16> &out) {
        int width = linear(p + "gate_proj", sc->normalized, c.hidden, sc->mlp_gate);
        check(linear(p + "up_proj", sc->normalized, c.hidden, sc->mlp_up) == width, "GPU SwiGLU width mismatch");
        swiglu(sc->mlp_gate, sc->mlp_up, width, c.swiglu_limit, ctx->stream);
        check(linear(p + "down_proj", sc->mlp_gate, width, out) == c.hidden, "GPU MLP output width mismatch");
    }
    void mlp(int layer, const std::string &p) {
        auto &s = *sc;
        if (c.mlp[layer] == "dense") {
            dense(p, s.branch);
            return;
        }
        // Routing intentionally stays FP32, including the router's source weights.
        s.router_scores.ensure(c.experts, ctx->stream);
        float_matvec(constant(p + "gate.weight", size_t(c.experts) * c.hidden), s.normalized, s.router_scores,
                     c.experts, c.hidden, ctx->stream);
        s.expert_ids.ensure(c.expert_topk, ctx->stream);
        s.router_weights.ensure(c.expert_topk, ctx->stream);
        s.group_scores.ensure(c.groups, ctx->stream);
        s.allowed.ensure(c.groups, ctx->stream);
        route(s.router_scores, constant(p + "gate.e_score_correction_bias", c.experts), s.expert_ids,
              s.router_weights, s.group_scores, s.allowed, c.experts, c.expert_topk, c.groups, c.top_groups,
              c.norm_topk, c.routed_scale, ctx->stream);
        ctx->read_indices.ensure(c.expert_topk);
        ctx->readback.ensure(c.expert_topk);
        checked(d2h(ctx->read_indices.p, s.expert_ids, size_t(c.expert_topk) * sizeof(int), ctx->stream),
                "read expert IDs");
        checked(d2h(ctx->readback.p, s.router_weights, size_t(c.expert_topk) * 4, ctx->stream),
                "read router weights");
        checked(sync(ctx->stream), "route experts");
        std::vector<int> ids(ctx->read_indices.p, ctx->read_indices.p + c.expert_topk);
        Vec factors(ctx->readback.p, ctx->readback.p + c.expert_topk);
        dense(p + "shared_experts.", s.branch);
        s.accumulated.ensure(c.hidden, ctx->stream);
        to_float(s.branch, s.accumulated, c.hidden, ctx->stream);
        for (int i = 0; i < c.expert_topk; ++i) {
            check(ids[i] >= 0 && ids[i] < c.experts && std::isfinite(factors[i]), "invalid GPU expert route");
            dense(p + "experts." + std::to_string(ids[i]) + ".", s.expert);
            axpy(s.accumulated, s.expert, c.hidden, factors[i], ctx->stream);
        }
        from_float(s.accumulated, s.branch, c.hidden, ctx->stream);
    }
    void attention(int layer, const std::string &p) {
        auto &st = state[layer];
        auto &s = *sc;
        auto stream = ctx->stream;
        if (c.attention[layer] == "linear_attention") {
            int width = c.linear_heads * c.linear_dim;
            check(linear(p + "q_proj", s.normalized, c.hidden, s.q) == width &&
                      linear(p + "k_proj", s.normalized, c.hidden, s.k) == width &&
                      linear(p + "v_proj", s.normalized, c.hidden, s.v) == width,
                  "GPU KDA projection geometry");
            size_t history = size_t(3) * width * (c.conv - 1), memory = size_t(width) * c.linear_dim;
            if (!st.recurrent.p) {
                st.recurrent.ensure(memory, stream);
                checked(zero(st.recurrent, memory * 4, stream), "clear KDA memory");
                st.convolution.ensure(history, stream);
                if (history)
                    checked(zero(st.convolution, history * 2, stream), "clear KDA convolution");
            }
            convolution(s.q, s.k, s.v, st.convolution, constant(p + "q_conv1d.weight", size_t(width) * c.conv),
                        constant(p + "k_conv1d.weight", size_t(width) * c.conv),
                        constant(p + "v_conv1d.weight", size_t(width) * c.conv), width, c.conv, stream);
            check(linear(p + "f_a_proj", s.normalized, c.hidden, s.low) == c.linear_dim &&
                      linear(p + "f_b_proj", s.low, c.linear_dim, s.forget) == width &&
                      linear(p + "b_proj", s.normalized, c.hidden, s.beta) == c.linear_heads,
                  "GPU KDA gate geometry");
            s.attended.ensure(width, stream);
            kda(s.q, s.k, s.v, s.forget, s.beta, constant(p + "dt_bias", width),
                constant(p + "A_log", c.linear_heads), st.recurrent, s.attended, c.linear_heads, c.linear_dim,
                c.gate_lower, stream);
            check(linear(p + "g_a_proj", s.normalized, c.hidden, s.low) == c.linear_dim &&
                      linear(p + "g_b_proj", s.low, c.linear_dim, s.gate) == width,
                  "GPU KDA output gate geometry");
            gated_norm(s.attended, s.gate, constant(p + "o_norm.weight", c.linear_dim), c.linear_heads,
                       c.linear_dim, c.eps, stream);
            check(linear(p + "o_proj", s.attended, width, s.branch) == c.hidden, "GPU KDA output geometry");
            return;
        }
        check(linear(p + "q_a_proj", s.normalized, c.hidden, s.qr) == c.qrank, "GPU MLA low-rank query geometry");
        rms(s.qr, s.qr, p + "q_a_layernorm.weight", c.qrank);
        check(linear(p + "q_b_proj", s.qr, c.qrank, s.q) == c.heads * c.qdim, "GPU MLA query geometry");
        check(linear(p + "kv_a_proj_with_mqa", s.normalized, c.hidden, s.latent) == c.kvrank,
              "GPU MLA latent geometry");
        rms(s.latent, s.latent, p + "kv_a_layernorm.weight", c.kvrank);
        int capacity = std::min(c.max_context, std::max(256, power_of_two(position + 1)));
        st.latent.ensure(size_t(capacity) * c.kvrank, stream, true);
        checked(d2d(st.latent.p + size_t(position) * c.kvrank, s.latent, size_t(c.kvrank) * 2, stream),
                "append MLA latent");
        st.selected.ensure(c.topk + c.pool, stream);
        if (c.indexers[layer] == "shared") {
            int previous = layer - 1;
            while (previous >= 0 && !state[previous].count)
                --previous;
            check(previous >= 0, "GPU shared indexer has no preceding result");
            check(layer_context[previous] == layer_context[layer],
                  "GPU layer split must keep a shared indexer on the device of its full indexer");
            st.count = state[previous].count;
            checked(d2d(st.selected, state[previous].selected, size_t(st.count) * sizeof(int), stream),
                    "reuse sparse indices");
        } else {
            auto ix = p + "indexer.";
            check(linear(ix + "wq_b", s.qr, c.qrank, s.iq) == c.index_heads * c.index_dim &&
                      linear(ix + "wk", s.normalized, c.hidden, s.ik) == c.index_dim &&
                      linear(ix + "index_kpool_compress_gate", s.normalized, c.hidden, s.ig) == c.index_dim &&
                      linear(ix + "weights_proj", s.normalized, c.hidden, s.iw) == c.index_heads,
                  "GPU indexer geometry");
            norm(s.ik, constant(ix + "k_norm.weight", c.index_dim), constant(ix + "k_norm.bias", c.index_dim),
                 s.ik, c.index_dim, 1e-6f, true, stream);
            st.keys.ensure(size_t(c.pool) * c.index_dim, stream);
            st.gates.ensure(size_t(c.pool) * c.index_dim, stream);
            int slot = position % c.pool, pools = (position + 1) / c.pool;
            checked(d2d(st.keys.p + size_t(slot) * c.index_dim, s.ik, size_t(c.index_dim) * 2, stream),
                    "append index key");
            checked(d2d(st.gates.p + size_t(slot) * c.index_dim, s.ig, size_t(c.index_dim) * 2, stream),
                    "append index gate");
            st.pooled.ensure(size_t((capacity + c.pool - 1) / c.pool) * c.index_dim, stream, true);
            if (slot == c.pool - 1)
                pool_key(st.keys, st.gates,
                         constant(ix + "index_kpool_compress_ape", size_t(c.pool) * c.index_dim),
                         st.pooled.p + size_t(pools - 1) * c.index_dim, c.pool, c.index_dim, stream);
            int padded = power_of_two(std::max(1, pools));
            s.pool_scores.ensure(std::max(1024, padded), stream);
            s.pool_ids.ensure(std::max(1024, padded), stream);
            index_scores(s.iq, st.pooled, s.iw, s.pool_scores, pools, c.index_heads, c.index_dim, stream);
            select_keys(s.pool_scores, s.pool_ids, st.selected, pools, padded, c.pool, c.topk, position, stream);
            st.count = std::min(pools, c.topk / c.pool) * c.pool + (position + 1) % c.pool;
        }
        if (!st.kv.p) {
            // Decoded once into the layer's own buffer; the packed source need not stay resident.
            auto m = ctx->matrix(weights, p + "kv_b_proj", false);
            check(m.rows == c.heads * (c.qdim + c.vdim) && m.cols == c.kvrank, "GPU MLA expansion geometry");
            st.kv.ensure(size_t(m.rows) * m.cols, stream);
            decode_matrix(m, st.kv, stream);
        }
        s.query.ensure(size_t(c.heads) * c.kvrank, stream);
        s.weighted.ensure(size_t(c.heads) * c.kvrank, stream);
        s.attention_scores.ensure(size_t(c.heads) * st.count, stream);
        s.attended.ensure(size_t(c.heads) * c.vdim, stream);
        absorb_query(s.q, st.kv, s.query, c.heads, c.qdim, c.vdim, c.kvrank, stream);
        mla(s.query, st.latent, st.kv, st.selected, s.attended, s.attention_scores, s.weighted, st.count, c.heads,
            c.qdim, c.vdim, c.kvrank, stream);
        check(linear(p + "o_proj", s.attended, size_t(c.heads) * c.vdim, s.branch) == c.hidden,
              "GPU MLA output geometry");
    }
    // Switch to the context of `index`, carrying the residual streams over from the previous device.
    void use(int index) {
        Context *next = contexts[index].get();
        if (ctx == next)
            return;
        Scratch &target = scratch[index];
        if (ctx) {
            checked(event_record(ctx->done, ctx->stream), "record GPU hand-off");
            size_t n = size_t(c.streams) * c.hidden;
            next->activate();
            target.x.ensure(n, next->stream);
            target.next.ensure(n, next->stream);
            checked(stream_wait(next->stream, ctx->done), "wait for previous GPU");
            checked(copy_peer(target.x, next->device, sc->x, ctx->device, n * sizeof(B16), next->stream),
                    "transfer residual streams between GPUs");
        } else
            next->activate();
        ctx = next;
        sc = &target;
    }
    // Contiguous layer ranges with roughly equal checkpoint bytes per device; a shared indexer stays with
    // the full indexer it reuses, and the head goes with the last layer.
    void assign_layers(int devices) {
        layer_context.assign(c.layers, 0);
        if (devices <= 1)
            return;
        std::vector<double> bytes(c.layers, 0);
        auto prefix = weights.prefix + "layers.";
        for (const auto &it : weights.index().at("weight_map").items()) {
            const std::string &name = it.key();
            if (!name.starts_with(prefix))
                continue;
            size_t dot = name.find('.', prefix.size());
            if (dot == std::string::npos)
                continue;
            int layer = -1;
            auto digits = std::string_view(name).substr(prefix.size(), dot - prefix.size());
            auto result = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
            if (result.ec != std::errc{} || result.ptr != digits.data() + digits.size() || layer < 0 ||
                layer >= c.layers)
                continue;
            bytes[layer] += double(weights.file(name).bytes(name).size());
        }
        double total = 0;
        for (double b : bytes)
            total += b;
        double running = 0;
        int device = 0;
        for (int l = 0; l < c.layers; ++l) {
            // Advance when the running total has crossed this device's share, never splitting a shared
            // indexer from its source layer.
            if (device + 1 < devices && running >= total * (device + 1) / devices &&
                !(c.attention[l] == "deepseek_sparse_attention" && c.indexers[l] == "shared"))
                ++device;
            layer_context[l] = device;
            running += bytes[l];
        }
    }
    void report() {
        if (reported)
            return;
        reported = true;
        for (size_t i = 0; i < contexts.size(); ++i) {
            const auto &x = *contexts[i];
            int first = -1, last = -1;
            for (int l = 0; l < c.layers; ++l)
                if (layer_context[l] == int(i)) {
                    if (first < 0)
                        first = l;
                    last = l;
                }
            std::cerr << "GLM GPU " << x.device << ": layers " << first << "-" << last << ", resident "
                      << double(x.resident_used) / double(size_t(1) << 30) << " GiB in " << x.resident.size()
                      << " matrices (budget " << double(x.resident_budget) / double(size_t(1) << 30)
                      << " GiB); streamed " << x.streamed_count << " matrices, "
                      << double(x.streamed_bytes) / double(size_t(1) << 30) << " GiB per token\n";
        }
    }

  public:
    DeviceModel(const fs::path &path, const GpuOptions &options)
        : weights(path), c(read_json(path / "config.json")), state(c.layers) {
        check(c.linear_dim <= 2048, "GPU KDA head dimension exceeds shared-memory limit");
        check(c.expert_topk <= (c.experts / std::max(1, c.groups)) * c.top_groups ||
                  std::none_of(c.mlp.begin(), c.mlp.end(), [](const auto &t) { return t == "sparse"; }),
              "not enough experts in selected groups");
        check(options.device >= 0 && options.devices >= 1, "invalid GPU device selection");
        int count = 0;
        checked(device_count(&count), "count GPUs");
        check(options.device + options.devices <= count, "GPU layer split needs consecutive devices from --device");
        int context = options.context > 0 ? std::min(options.context, c.max_context) : c.max_context;
        for (int i = 0; i < options.devices; ++i)
            contexts.push_back(std::make_unique<Context>(options.device + i));
        scratch.resize(contexts.size());
        assign_layers(options.devices);
        bool set = false;
        size_t fixed = env_bytes("GLM53_RESIDENT_GIB", set);
        for (size_t i = 0; i < contexts.size(); ++i) {
            auto &x = *contexts[i];
            if (options.resident_bytes >= 0)
                x.resident_budget = size_t(options.resident_bytes);
            else if (set)
                x.resident_budget = fixed;
            else {
                // Leave room for the attention caches at `context` tokens, the decoded MLA expansions, the
                // FP32 constants and activations, plus a fixed margin for the allocator and other users.
                int mla = 0;
                for (int l = 0; l < c.layers; ++l)
                    mla += layer_context[l] == int(i) && c.attention[l] == "deepseek_sparse_attention";
                size_t per_layer = size_t(context) * c.kvrank * 2 + size_t(context / c.pool + 1) * c.index_dim * 2 +
                                   size_t(c.heads) * (c.qdim + c.vdim) * c.kvrank * 2 +
                                   size_t(c.heads) * (c.topk + c.pool) * 4;
                size_t headroom = size_t(4) << 30;
                headroom += size_t(mla) * per_layer + size_t(c.vocab) * 8;
                size_t free = x.free_memory();
                x.resident_budget = free > headroom ? free - headroom : 0;
            }
        }
    }
    ~DeviceModel() override {
        for (auto &x : contexts) {
            (void)set_device(x->device);
            (void)sync(x->stream);
        }
    }
    void reset() override {
        for (int l = 0; l < c.layers; ++l) {
            auto &s = state[l];
            auto &x = *contexts[layer_context[l]];
            x.activate();
            checked(sync(x.stream), "finish previous GPU sequence");
            if (s.recurrent.p)
                checked(zero(s.recurrent, s.recurrent.capacity * 4, x.stream), "reset KDA state");
            if (s.convolution.p)
                checked(zero(s.convolution, s.convolution.capacity * 2, x.stream), "reset convolution");
            s.count = 0;
        }
        position = 0;
        failed = false;
    }
    Vec step(int token) override {
        check(!failed, "GPU inference failed; reset before reusing this sequence");
        check(token >= 0 && token < c.vocab && position < c.max_context,
              "token/context outside GPU model limits");
        try {
            ctx = nullptr;
            use(layer_context[0]);
            auto s = ctx->stream;
            embedding_row = weights.row(weights.prefix + "embed_tokens.weight", token, c.hidden);
            sc->upload_embedding.ensure(c.hidden, s);
            sc->embed.ensure(c.hidden, s);
            checked(h2d(sc->upload_embedding, embedding_row.data(), embedding_row.size() * 4, s),
                    "upload embedding row");
            from_float(sc->upload_embedding, sc->embed, c.hidden, s);
            sc->x.ensure(size_t(c.streams) * c.hidden, s);
            sc->next.ensure(size_t(c.streams) * c.hidden, s);
            replicate(sc->embed, sc->x, c.hidden, c.streams, s);
            for (int l = 0; l < c.layers; ++l) {
                use(layer_context[l]);
                s = ctx->stream;
                sc->collapsed.ensure(c.hidden, s);
                sc->post.ensure(c.streams, s);
                sc->mix.ensure(c.streams * c.streams, s);
                sc->hc_scratch.ensure(c.streams * (c.streams + 3), s);
                auto p = weights.prefix + "layers." + std::to_string(l) + ".";
                for (bool ffn : {false, true}) {
                    auto hc = p + (ffn ? "hc_ffn_" : "hc_attn_");
                    hc_pre(sc->x, constant(hc + "fn", size_t(c.streams) * (c.streams + 2) * c.streams * c.hidden),
                           constant(hc + "base", c.streams * (c.streams + 2)), constant(hc + "scale", 3),
                           sc->collapsed, sc->post, sc->mix, sc->hc_scratch, c.hidden, c.streams, c.sinkhorn,
                           c.eps, c.hc_eps, s);
                    rms(sc->collapsed, sc->normalized,
                        p + (ffn ? "post_attention_layernorm.weight" : "input_layernorm.weight"), c.hidden);
                    if (ffn)
                        mlp(l, p + "mlp.");
                    else
                        attention(l, p + "self_attn.");
                    hc_post(sc->x, sc->branch, sc->post, sc->mix, sc->next, c.hidden, c.streams, s);
                    std::swap(sc->x, sc->next);
                }
            }
            sc->collapsed.ensure(c.hidden, s);
            mean_streams(sc->x, sc->collapsed, c.hidden, c.streams, s);
            rms(sc->collapsed, sc->normalized, weights.prefix + "norm.weight", c.hidden);
            check(linear("lm_head", sc->normalized, c.hidden, sc->logits) == c.vocab,
                  "GPU output vocabulary mismatch");
            sc->host_output.ensure(c.vocab, s);
            to_float(sc->logits, sc->host_output, c.vocab, s);
            ctx->readback.ensure(c.vocab);
            checked(d2h(ctx->readback.p, sc->host_output, size_t(c.vocab) * 4, s), "read logits");
            checked(sync(s), "GPU inference synchronization");
            Vec out(ctx->readback.p, ctx->readback.p + c.vocab);
            for (float v : out)
                check(std::isfinite(v), "nonfinite GPU logits");
            ++position;
            report();
            return out;
        } catch (...) {
            failed = true;
            throw;
        }
    }
};
struct Projector {
    Context ctx;
    Buffer<float> source, result;
    Buffer<B16> input;
    explicit Projector(int d) : ctx(d) {}
    ~Projector() {
        (void)set_device(ctx.device);
        (void)sync(ctx.stream);
    }
    Vec operator()(const Weights &w, const std::string &base, const Vec &x) {
        ctx.activate();
        source.ensure(x.size(), ctx.stream);
        input.ensure(x.size(), ctx.stream);
        checked(h2d(source, x.data(), x.size() * 4, ctx.stream), "upload projection input");
        from_float(source, input, int(x.size()), ctx.stream);
        auto m = ctx.matrix(w, base, false);
        check(size_t(m.cols) == x.size(), "GPU projection width mismatch");
        result.ensure(m.rows, ctx.stream);
        gemv(m, input, result, true, ctx.stream);
        ctx.readback.ensure(m.rows);
        checked(d2h(ctx.readback.p, result, size_t(m.rows) * 4, ctx.stream), "read projection");
        checked(sync(ctx.stream), "GPU projection synchronization");
        return Vec(ctx.readback.p, ctx.readback.p + m.rows);
    }
};
} // namespace
std::function<Vec(const Weights &, const std::string &, const Vec &)> gpu_projector(int device) {
    auto p = std::make_shared<Projector>(device);
    return [p](const Weights &w, const std::string &name, const Vec &x) { return (*p)(w, name, x); };
}
std::shared_ptr<InferenceBackend> gpu_backend(const fs::path &path, const GpuOptions &options) {
    return std::make_shared<DeviceModel>(path, options);
}
std::shared_ptr<InferenceBackend> gpu_backend(const fs::path &path, int device) {
    GpuOptions options;
    options.device = device;
    return gpu_backend(path, options);
}
} // namespace glm53
