#include "device_ops.hpp"
#include "glm53.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace glm53 {
namespace {
using namespace gpu;
struct Context {
    int device;
    Stream stream{};
    Buffer<uint8_t> raw, block_scales;
    Buffer<float> scales;
    Pinned<float> readback;
    Pinned<int> read_indices;
    std::map<std::string, Tensor> host_scales;
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
    }
    ~Context() {
        (void)set_device(device);
        (void)sync(stream);
        raw.clear();
        block_scales.clear();
        scales.clear();
        (void)stream_destroy(stream);
    }
    Matrix matrix(const Weights &w, const std::string &base) {
        bool nv = w.has(base + ".weight_packed");
        auto name = nv ? base + ".weight_packed" : w.has(base) ? base : base + ".weight";
        const auto &f = w.file(name);
        auto shape = f.shape(name);
        check(shape.size() == 2 && shape[0] > 0 && shape[0] <= INT32_MAX && shape[1] > 0 &&
                  shape[1] <= INT32_MAX / 2,
              "GPU matrix geometry outside supported range: " + name);
        int r = int(shape[0]), c = int(shape[1]) * (nv ? 2 : 1);
        auto bytes = f.bytes(name);
        raw.ensure(bytes.size(), stream);
        checked(h2d(raw, bytes.data(), bytes.size(), stream), "upload matrix");
        Matrix m{raw, nullptr, nullptr, r, c, 0, false, 1};
        auto dtype = f.dtype(name);
        if (nv) {
            check(dtype == "U8" && c % 16 == 0 && !w.has(base + ".weight"), "invalid NVFP4 matrix");
            m.format = 2;
            auto sn = base + ".weight_scale";
            const auto &sf = w.file(sn);
            check(sf.dtype(sn) == "F8_E4M3" && sf.shape(sn) == std::vector<size_t>{size_t(r), size_t(c / 16)},
                  "invalid NVFP4 block-scale geometry");
            auto sb = sf.bytes(sn);
            block_scales.ensure(sb.size(), stream);
            checked(h2d(block_scales, sb.data(), sb.size(), stream), "upload NVFP4 scales");
            m.block_scales = block_scales;
            auto gs = w.tensor(base + ".weight_global_scale");
            check(gs.data.size() == 1 && gs.data[0] > 0, "invalid NVFP4 global scale");
            m.global = gs.data[0];
        } else if (dtype == "F8_E4M3") {
            m.format = 1;
            m.block = w.has(base + ".weight_scale_inv");
            auto sn = base + (m.block ? ".weight_scale_inv" : ".weight_scale");
            if (!host_scales.contains(sn))
                host_scales.emplace(sn, w.tensor(sn));
            const auto &s = host_scales.at(sn);
            check(m.block ? s.shape == std::vector<size_t>{size_t((r + 127) / 128), size_t((c + 127) / 128)}
                          : s.data.size() == size_t(r),
                  "invalid FP8 scale geometry");
            for (float a : s.data)
                check(a >= 0 && std::isfinite(a), "invalid FP8 scale");
            scales.ensure(s.data.size(), stream);
            checked(h2d(scales, s.data.data(), s.data.size() * 4, stream), "upload FP8 scales");
            m.scales = scales;
        } else if (dtype == "BF16")
            m.format = 4;
        else if (dtype == "F16")
            m.format = 3;
        else
            check(dtype == "F32", "unsupported GPU linear dtype: " + dtype);
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
struct Resident {
    Tensor host;
    Buffer<float> data;
};
int power_of_two(int n) {
    int p = 1;
    while (p < n) {
        check(p <= INT32_MAX / 2, "GPU context too large");
        p *= 2;
    }
    return p;
}
class DeviceModel final : public InferenceBackend {
    Weights weights;
    Config c;
    Context ctx;
    struct State {
        Buffer<float> recurrent;
        Buffer<B16> convolution, latent, keys, gates, pooled, kv;
        Buffer<int> selected;
        int count = 0;
    };
    std::vector<State> state;
    std::map<std::string, Resident> constants;
    int position = 0;
    bool failed = false;
    Vec embedding_row;
    Buffer<B16> x, next, collapsed, normalized, branch, embed;
    Buffer<B16> q, k, v, low, forget, beta, gate, attended, qr, latent, iq, ik, iw, ig;
    Buffer<B16> mlp_gate, mlp_up, expert, logits;
    Buffer<float> post, mix, hc_scratch, query, attention_scores, weighted, pool_scores;
    Buffer<float> router_scores, router_weights, group_scores, accumulated, host_output, upload_embedding;
    Buffer<int> pool_ids, expert_ids, allowed;
    const float *constant(const std::string &name, size_t size) {
        if (!constants.contains(name)) {
            Resident r;
            r.host = weights.tensor(name);
            check(r.host.data.size() == size, "GPU constant geometry mismatch: " + name);
            r.data.ensure(size, ctx.stream);
            auto it = constants.emplace(name, std::move(r)).first;
            const auto &a = it->second;
            checked(h2d(a.data.p, a.host.data.data(), size * 4, ctx.stream), "upload constant");
        }
        check(constants.at(name).host.data.size() == size, "GPU constant size changed");
        return constants.at(name).data;
    }
    int linear(const std::string &name, const Buffer<B16> &in, size_t width, Buffer<B16> &out) {
        return ctx.linear(weights, name, in, width, out);
    }
    void rms(const Buffer<B16> &in, Buffer<B16> &out, const std::string &name, int width) {
        out.ensure(width, ctx.stream);
        norm(in, constant(name, width), nullptr, out, width, c.eps, false, ctx.stream);
    }
    void dense(const std::string &p, Buffer<B16> &out) {
        int width = linear(p + "gate_proj", normalized, c.hidden, mlp_gate);
        check(linear(p + "up_proj", normalized, c.hidden, mlp_up) == width, "GPU SwiGLU width mismatch");
        swiglu(mlp_gate, mlp_up, width, c.swiglu_limit, ctx.stream);
        check(linear(p + "down_proj", mlp_gate, width, out) == c.hidden, "GPU MLP output width mismatch");
    }
    void mlp(int layer, const std::string &p) {
        if (c.mlp[layer] == "dense") {
            dense(p, branch);
            return;
        }
        // Routing intentionally stays FP32, including the router's source weights.
        router_scores.ensure(c.experts, ctx.stream);
        float_matvec(constant(p + "gate.weight", size_t(c.experts) * c.hidden), normalized, router_scores,
                     c.experts, c.hidden, ctx.stream);
        expert_ids.ensure(c.expert_topk, ctx.stream);
        router_weights.ensure(c.expert_topk, ctx.stream);
        group_scores.ensure(c.groups, ctx.stream);
        allowed.ensure(c.groups, ctx.stream);
        route(router_scores, constant(p + "gate.e_score_correction_bias", c.experts), expert_ids,
              router_weights, group_scores, allowed, c.experts, c.expert_topk, c.groups, c.top_groups,
              c.norm_topk, c.routed_scale, ctx.stream);
        ctx.read_indices.ensure(c.expert_topk);
        ctx.readback.ensure(c.expert_topk);
        checked(d2h(ctx.read_indices.p, expert_ids, size_t(c.expert_topk) * sizeof(int), ctx.stream),
                "read expert IDs");
        checked(d2h(ctx.readback.p, router_weights, size_t(c.expert_topk) * 4, ctx.stream),
                "read router weights");
        checked(sync(ctx.stream), "route experts");
        std::vector<int> ids(ctx.read_indices.p, ctx.read_indices.p + c.expert_topk);
        Vec factors(ctx.readback.p, ctx.readback.p + c.expert_topk);
        dense(p + "shared_experts.", branch);
        accumulated.ensure(c.hidden, ctx.stream);
        to_float(branch, accumulated, c.hidden, ctx.stream);
        for (int i = 0; i < c.expert_topk; ++i) {
            check(ids[i] >= 0 && ids[i] < c.experts && std::isfinite(factors[i]), "invalid GPU expert route");
            dense(p + "experts." + std::to_string(ids[i]) + ".", expert);
            axpy(accumulated, expert, c.hidden, factors[i], ctx.stream);
        }
        from_float(accumulated, branch, c.hidden, ctx.stream);
    }
    void attention(int layer, const std::string &p) {
        auto &st = state[layer];
        auto s = ctx.stream;
        if (c.attention[layer] == "linear_attention") {
            int width = c.linear_heads * c.linear_dim;
            check(linear(p + "q_proj", normalized, c.hidden, q) == width &&
                      linear(p + "k_proj", normalized, c.hidden, k) == width &&
                      linear(p + "v_proj", normalized, c.hidden, v) == width,
                  "GPU KDA projection geometry");
            size_t history = size_t(3) * width * (c.conv - 1), memory = size_t(width) * c.linear_dim;
            if (!st.recurrent.p) {
                st.recurrent.ensure(memory, s);
                checked(zero(st.recurrent, memory * 4, s), "clear KDA memory");
                st.convolution.ensure(history, s);
                if (history)
                    checked(zero(st.convolution, history * 2, s), "clear KDA convolution");
            }
            convolution(q, k, v, st.convolution, constant(p + "q_conv1d.weight", size_t(width) * c.conv),
                        constant(p + "k_conv1d.weight", size_t(width) * c.conv),
                        constant(p + "v_conv1d.weight", size_t(width) * c.conv), width, c.conv, s);
            check(linear(p + "f_a_proj", normalized, c.hidden, low) == c.linear_dim &&
                      linear(p + "f_b_proj", low, c.linear_dim, forget) == width &&
                      linear(p + "b_proj", normalized, c.hidden, beta) == c.linear_heads,
                  "GPU KDA gate geometry");
            attended.ensure(width, s);
            kda(q, k, v, forget, beta, constant(p + "dt_bias", width), constant(p + "A_log", c.linear_heads),
                st.recurrent, attended, c.linear_heads, c.linear_dim, c.gate_lower, s);
            check(linear(p + "g_a_proj", normalized, c.hidden, low) == c.linear_dim &&
                      linear(p + "g_b_proj", low, c.linear_dim, gate) == width,
                  "GPU KDA output gate geometry");
            gated_norm(attended, gate, constant(p + "o_norm.weight", c.linear_dim), c.linear_heads,
                       c.linear_dim, c.eps, s);
            check(linear(p + "o_proj", attended, width, branch) == c.hidden, "GPU KDA output geometry");
            return;
        }
        check(linear(p + "q_a_proj", normalized, c.hidden, qr) == c.qrank, "GPU MLA low-rank query geometry");
        rms(qr, qr, p + "q_a_layernorm.weight", c.qrank);
        check(linear(p + "q_b_proj", qr, c.qrank, q) == c.heads * c.qdim, "GPU MLA query geometry");
        check(linear(p + "kv_a_proj_with_mqa", normalized, c.hidden, latent) == c.kvrank,
              "GPU MLA latent geometry");
        rms(latent, latent, p + "kv_a_layernorm.weight", c.kvrank);
        int capacity = std::min(c.max_context, std::max(256, power_of_two(position + 1)));
        st.latent.ensure(size_t(capacity) * c.kvrank, s, true);
        checked(d2d(st.latent.p + size_t(position) * c.kvrank, latent, size_t(c.kvrank) * 2, s),
                "append MLA latent");
        st.selected.ensure(c.topk + c.pool, s);
        if (c.indexers[layer] == "shared") {
            int previous = layer - 1;
            while (previous >= 0 && !state[previous].count)
                --previous;
            check(previous >= 0, "GPU shared indexer has no preceding result");
            st.count = state[previous].count;
            checked(d2d(st.selected, state[previous].selected, size_t(st.count) * sizeof(int), s),
                    "reuse sparse indices");
        } else {
            auto ix = p + "indexer.";
            check(linear(ix + "wq_b", qr, c.qrank, iq) == c.index_heads * c.index_dim &&
                      linear(ix + "wk", normalized, c.hidden, ik) == c.index_dim &&
                      linear(ix + "index_kpool_compress_gate", normalized, c.hidden, ig) == c.index_dim &&
                      linear(ix + "weights_proj", normalized, c.hidden, iw) == c.index_heads,
                  "GPU indexer geometry");
            norm(ik, constant(ix + "k_norm.weight", c.index_dim), constant(ix + "k_norm.bias", c.index_dim),
                 ik, c.index_dim, 1e-6f, true, s);
            st.keys.ensure(size_t(c.pool) * c.index_dim, s);
            st.gates.ensure(size_t(c.pool) * c.index_dim, s);
            int slot = position % c.pool, pools = (position + 1) / c.pool;
            checked(d2d(st.keys.p + size_t(slot) * c.index_dim, ik, size_t(c.index_dim) * 2, s),
                    "append index key");
            checked(d2d(st.gates.p + size_t(slot) * c.index_dim, ig, size_t(c.index_dim) * 2, s),
                    "append index gate");
            st.pooled.ensure(size_t((capacity + c.pool - 1) / c.pool) * c.index_dim, s, true);
            if (slot == c.pool - 1)
                pool_key(st.keys, st.gates,
                         constant(ix + "index_kpool_compress_ape", size_t(c.pool) * c.index_dim),
                         st.pooled.p + size_t(pools - 1) * c.index_dim, c.pool, c.index_dim, s);
            int padded = power_of_two(std::max(1, pools));
            pool_scores.ensure(std::max(1024, padded), s);
            pool_ids.ensure(std::max(1024, padded), s);
            index_scores(iq, st.pooled, iw, pool_scores, pools, c.index_heads, c.index_dim, s);
            select_keys(pool_scores, pool_ids, st.selected, pools, padded, c.pool, c.topk, position, s);
            st.count = std::min(pools, c.topk / c.pool) * c.pool + (position + 1) % c.pool;
        }
        if (!st.kv.p) {
            auto m = ctx.matrix(weights, p + "kv_b_proj");
            check(m.rows == c.heads * (c.qdim + c.vdim) && m.cols == c.kvrank, "GPU MLA expansion geometry");
            st.kv.ensure(size_t(m.rows) * m.cols, s);
            decode_matrix(m, st.kv, s);
        }
        query.ensure(size_t(c.heads) * c.kvrank, s);
        weighted.ensure(size_t(c.heads) * c.kvrank, s);
        attention_scores.ensure(size_t(c.heads) * st.count, s);
        attended.ensure(size_t(c.heads) * c.vdim, s);
        absorb_query(q, st.kv, query, c.heads, c.qdim, c.vdim, c.kvrank, s);
        mla(query, st.latent, st.kv, st.selected, attended, attention_scores, weighted, st.count, c.heads,
            c.qdim, c.vdim, c.kvrank, s);
        check(linear(p + "o_proj", attended, size_t(c.heads) * c.vdim, branch) == c.hidden,
              "GPU MLA output geometry");
    }

  public:
    DeviceModel(const fs::path &path, int device)
        : weights(path), c(read_json(path / "config.json")), ctx(device), state(c.layers) {
        check(c.linear_dim <= 2048, "GPU KDA head dimension exceeds shared-memory limit");
        check(c.expert_topk <= (c.experts / std::max(1, c.groups)) * c.top_groups ||
                  std::none_of(c.mlp.begin(), c.mlp.end(), [](const auto &t) { return t == "sparse"; }),
              "not enough experts in selected groups");
    }
    ~DeviceModel() override {
        (void)set_device(ctx.device);
        (void)sync(ctx.stream);
    }
    void reset() override {
        checked(set_device(ctx.device), "select GPU");
        checked(sync(ctx.stream), "finish previous GPU sequence");
        for (auto &s : state) {
            if (s.recurrent.p)
                checked(zero(s.recurrent, s.recurrent.capacity * 4, ctx.stream), "reset KDA state");
            if (s.convolution.p)
                checked(zero(s.convolution, s.convolution.capacity * 2, ctx.stream), "reset convolution");
            s.count = 0;
        }
        position = 0;
        failed = false;
    }
    Vec step(int token) override {
        check(!failed, "GPU inference failed; reset before reusing this sequence");
        check(token >= 0 && token < c.vocab && position < c.max_context,
              "token/context outside GPU model limits");
        checked(set_device(ctx.device), "select GPU");
        try {
            auto s = ctx.stream;
            embedding_row = weights.row(weights.prefix + "embed_tokens.weight", token, c.hidden);
            upload_embedding.ensure(c.hidden, s);
            embed.ensure(c.hidden, s);
            checked(h2d(upload_embedding, embedding_row.data(), embedding_row.size() * 4, s),
                    "upload embedding row");
            from_float(upload_embedding, embed, c.hidden, s);
            x.ensure(size_t(c.streams) * c.hidden, s);
            next.ensure(size_t(c.streams) * c.hidden, s);
            replicate(embed, x, c.hidden, c.streams, s);
            collapsed.ensure(c.hidden, s);
            post.ensure(c.streams, s);
            mix.ensure(c.streams * c.streams, s);
            hc_scratch.ensure(c.streams * (c.streams + 3), s);
            for (int l = 0; l < c.layers; ++l) {
                auto p = weights.prefix + "layers." + std::to_string(l) + ".";
                for (bool ffn : {false, true}) {
                    auto hc = p + (ffn ? "hc_ffn_" : "hc_attn_");
                    hc_pre(x, constant(hc + "fn", size_t(c.streams) * (c.streams + 2) * c.streams * c.hidden),
                           constant(hc + "base", c.streams * (c.streams + 2)), constant(hc + "scale", 3),
                           collapsed, post, mix, hc_scratch, c.hidden, c.streams, c.sinkhorn, c.eps, c.hc_eps,
                           s);
                    rms(collapsed, normalized,
                        p + (ffn ? "post_attention_layernorm.weight" : "input_layernorm.weight"), c.hidden);
                    if (ffn)
                        mlp(l, p + "mlp.");
                    else
                        attention(l, p + "self_attn.");
                    hc_post(x, branch, post, mix, next, c.hidden, c.streams, s);
                    std::swap(x, next);
                }
            }
            mean_streams(x, collapsed, c.hidden, c.streams, s);
            rms(collapsed, normalized, weights.prefix + "norm.weight", c.hidden);
            check(linear("lm_head", normalized, c.hidden, logits) == c.vocab,
                  "GPU output vocabulary mismatch");
            host_output.ensure(c.vocab, s);
            to_float(logits, host_output, c.vocab, s);
            ctx.readback.ensure(c.vocab);
            checked(d2h(ctx.readback.p, host_output, size_t(c.vocab) * 4, s), "read logits");
            checked(sync(s), "GPU inference synchronization");
            Vec out(ctx.readback.p, ctx.readback.p + c.vocab);
            for (float v : out)
                check(std::isfinite(v), "nonfinite GPU logits");
            ++position;
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
        checked(set_device(ctx.device), "select GPU");
        source.ensure(x.size(), ctx.stream);
        input.ensure(x.size(), ctx.stream);
        checked(h2d(source, x.data(), x.size() * 4, ctx.stream), "upload projection input");
        from_float(source, input, int(x.size()), ctx.stream);
        auto m = ctx.matrix(w, base);
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
std::shared_ptr<InferenceBackend> gpu_backend(const fs::path &path, int device) {
    return std::make_shared<DeviceModel>(path, device);
}
} // namespace glm53
