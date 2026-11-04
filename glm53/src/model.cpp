#include "glm53.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace glm53 {
namespace {
float sigmoid(float x) { return x >= 0 ? 1 / (1 + std::exp(-x)) : std::exp(x) / (1 + std::exp(x)); }
float bf(float x) { return float(behemoth::BFloat16(x)); }
Vec mul(const Tensor &w, const Vec &x) {
    check(w.shape.size() == 2 && w.shape[1] == x.size(), "matrix/input geometry mismatch");
    Vec y(w.shape[0]);
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < y.size(); ++r) {
        double sum = 0;
        for (size_t c = 0; c < x.size(); ++c)
            sum += double(w.data[r * x.size() + c]) * x[c];
        y[r] = float(sum);
    }
    return y;
}
Vec rms(const Vec &x, const Vec &w, float eps) {
    check(x.size() == w.size(), "RMS norm width mismatch");
    double ss = 0;
    for (float a : x)
        ss += double(a) * a;
    float inv = float(1 / std::sqrt(ss / x.size() + eps));
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] * inv * w[i];
    return y;
}
} // namespace
Vec kda_step(const Vec &q, const Vec &k, const Vec &v, const Vec &g, const Vec &beta, Vec &state, int heads,
             int dim) {
    size_t width = size_t(heads) * dim;
    check(q.size() == width && k.size() == width && v.size() == width && g.size() == width &&
              beta.size() == size_t(heads),
          "KDA input geometry mismatch");
    if (state.empty())
        state.assign(width * dim, 0);
    check(state.size() == width * dim, "KDA state geometry mismatch");
    Vec out(width);
#pragma omp parallel for schedule(static)
    for (int h = 0; h < heads; ++h) {
        double qs = 0, ks = 0;
        for (int i = 0; i < dim; ++i) {
            qs += double(q[h * dim + i]) * q[h * dim + i];
            ks += double(k[h * dim + i]) * k[h * dim + i];
        }
        float qi = float(1 / std::sqrt(qs + 1e-6)) / std::sqrt(float(dim)),
              ki = float(1 / std::sqrt(ks + 1e-6));
        for (int d = 0; d < dim; ++d) {
            float memory = 0;
            for (int i = 0; i < dim; ++i) {
                size_t at = (size_t(h) * dim + i) * dim + d;
                state[at] *= std::exp(g[h * dim + i]);
                memory += state[at] * k[h * dim + i] * ki;
            }
            float delta = (v[h * dim + d] - memory) * beta[h], sum = 0;
            for (int i = 0; i < dim; ++i) {
                size_t at = (size_t(h) * dim + i) * dim + d;
                state[at] += k[h * dim + i] * ki * delta;
                sum += state[at] * q[h * dim + i] * qi;
            }
            out[h * dim + d] = sum;
        }
    }
    return out;
}
Hyper hyper_pre(const Vec &x, const Tensor &fn, const Vec &base, const Vec &scale, const Config &c) {
    int n = c.streams, d = c.hidden;
    check(x.size() == size_t(n) * d && base.size() == size_t(n * (n + 2)) && scale.size() == 3,
          "mHC geometry mismatch");
    Vec normalized = rms(x, Vec(x.size(), 1), c.eps), coeff = mul(fn, normalized);
    check(coeff.size() == base.size(), "mHC projection shape mismatch");
    Hyper h{Vec(d), Vec(n), Vec(size_t(n) * n)};
    for (int i = 0; i < n; ++i) {
        float pre = sigmoid(coeff[i] * scale[0] + base[i]) + c.hc_eps;
        h.post[i] = 2 * sigmoid(coeff[n + i] * scale[1] + base[n + i]);
        for (int j = 0; j < d; ++j)
            h.collapsed[j] += pre * x[size_t(i) * d + j];
    }
    for (int i = 0; i < n; ++i) {
        float maximum = -INFINITY;
        for (int j = 0; j < n; ++j)
            maximum = std::max(maximum, coeff[2 * n + i * n + j] * scale[2] + base[2 * n + i * n + j]);
        float sum = 0;
        for (int j = 0; j < n; ++j) {
            float a = std::exp(coeff[2 * n + i * n + j] * scale[2] + base[2 * n + i * n + j] - maximum);
            h.mixing[i * n + j] = a;
            sum += a;
        }
        for (int j = 0; j < n; ++j)
            h.mixing[i * n + j] = h.mixing[i * n + j] / sum + c.hc_eps;
    }
    for (int iter = 0; iter < c.sinkhorn; ++iter) {
        if (iter)
            for (int i = 0; i < n; ++i) {
                float s = c.hc_eps;
                for (int j = 0; j < n; ++j)
                    s += h.mixing[i * n + j];
                for (int j = 0; j < n; ++j)
                    h.mixing[i * n + j] /= s;
            }
        for (int j = 0; j < n; ++j) {
            float s = c.hc_eps;
            for (int i = 0; i < n; ++i)
                s += h.mixing[i * n + j];
            for (int i = 0; i < n; ++i)
                h.mixing[i * n + j] /= s;
        }
    }
    return h;
}
Vec hyper_post(const Vec &residual, const Vec &value, const Hyper &h, const Config &c) {
    check(value.size() == size_t(c.hidden) && residual.size() == size_t(c.hidden) * c.streams,
          "mHC output shape mismatch");
    Vec out(residual.size());
    for (int i = 0; i < c.streams; ++i)
        for (int d = 0; d < c.hidden; ++d) {
            float a = h.post[i] * value[d];
            for (int j = 0; j < c.streams; ++j)
                a += h.mixing[j * c.streams + i] * residual[size_t(j) * c.hidden + d];
            out[size_t(i) * c.hidden + d] = a;
        }
    return out;
}
Model::Model(const fs::path &p, bool bf16) : weights_(p), bf16_(bf16), config(read_json(p / "config.json")) {
    states_.resize(config.layers);
}
void Model::reset() {
    if (backend_)
        backend_->reset();
    position_ = 0;
    states_.assign(config.layers, {});
}
Vec Model::round(Vec x) const {
    if (bf16_)
        for (float &v : x)
            v = bf(v);
    return x;
}
Vec Model::matvec(const std::string &name, const Vec &x) {
    if (projector_)
        return round(projector_(weights_, name, x));
    auto w = weights_.has(name) ? weights_.tensor(name) : weights_.linear(name);
    if (bf16_)
        for (float &v : w.data)
            v = bf(v);
    return round(mul(w, x));
}
Vec Model::norm(const Vec &x, const std::string &name, float eps) {
    return round(rms(x, weights_.tensor(name).data, eps));
}
Vec Model::mlp(int l, const Vec &x) {
    const auto &c = config;
    std::string p = weights_.prefix + "layers." + std::to_string(l) + ".mlp.";
    auto dense = [&](const std::string &base) {
        auto a = matvec(base + "gate_proj", x), b = matvec(base + "up_proj", x);
        check(a.size() == b.size(), "SwiGLU width mismatch");
        for (size_t i = 0; i < a.size(); ++i) {
            float gate = std::min(a[i], c.swiglu_limit),
                  up = std::clamp(b[i], -c.swiglu_limit, c.swiglu_limit);
            a[i] = gate * sigmoid(gate) * up;
        }
        return matvec(base + "down_proj", round(std::move(a)));
    };
    if (c.mlp[l] == "dense")
        return dense(p);
    auto scores = mul(weights_.linear(p + "gate"), x),
         bias = weights_.tensor(p + "gate.e_score_correction_bias").data;
    check(scores.size() == size_t(c.experts) && bias.size() == scores.size(), "router shape mismatch");
    Vec choice(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] = sigmoid(scores[i]);
        choice[i] = scores[i] + bias[i];
    }
    std::vector<int> groups(c.groups);
    std::iota(groups.begin(), groups.end(), 0);
    Vec gs(c.groups);
    int per = c.experts / c.groups;
    for (int g = 0; g < c.groups; ++g) {
        float first = -INFINITY, second = -INFINITY;
        for (int i = 0; i < per; ++i) {
            float a = choice[g * per + i];
            if (a > first) {
                second = first;
                first = a;
            } else
                second = std::max(second, a);
        }
        gs[g] = first + second;
    }
    std::sort(groups.begin(), groups.end(),
              [&](int a, int b) { return gs[a] == gs[b] ? a < b : gs[a] > gs[b]; });
    std::set<int> allowed(groups.begin(), groups.begin() + c.top_groups);
    std::vector<int> order;
    for (int i = 0; i < c.experts; ++i)
        if (allowed.contains(i / per))
            order.push_back(i);
    check(order.size() >= size_t(c.expert_topk), "not enough experts in selected groups");
    std::partial_sort(order.begin(), order.begin() + c.expert_topk, order.end(),
                      [&](int a, int b) { return choice[a] == choice[b] ? a < b : choice[a] > choice[b]; });
    float denominator = 1e-20f;
    if (c.norm_topk)
        for (int i = 0; i < c.expert_topk; ++i)
            denominator += scores[order[i]];
    else
        denominator = 1;
    auto out = dense(p + "shared_experts.");
    for (int i = 0; i < c.expert_topk; ++i) {
        auto y = dense(p + "experts." + std::to_string(order[i]) + ".");
        check(y.size() == out.size(), "expert output shape mismatch");
        float factor = scores[order[i]] / denominator * c.routed_scale;
        for (size_t j = 0; j < out.size(); ++j)
            out[j] += factor * y[j];
    }
    return round(out);
}
Vec Model::attention(int l, const Vec &x) {
    const auto &c = config;
    auto &s = states_[l];
    std::string p = weights_.prefix + "layers." + std::to_string(l) + ".self_attn.";
    if (c.attention[l] == "linear_attention") {
        int width = c.linear_heads * c.linear_dim;
        Vec projected[3] = {matvec(p + "q_proj", x), matvec(p + "k_proj", x), matvec(p + "v_proj", x)};
        if (s.kda.convolution.empty())
            s.kda.convolution.assign(size_t(3) * width * (c.conv - 1), 0);
        for (int part = 0; part < 3; ++part) {
            auto w = weights_.tensor(p +
                                     std::string(part == 0   ? "q"
                                                 : part == 1 ? "k"
                                                             : "v") +
                                     "_conv1d.weight");
            check(projected[part].size() == size_t(width) && w.data.size() == size_t(width) * c.conv,
                  "KDA convolution geometry mismatch");
            for (int channel = 0; channel < width; ++channel) {
                size_t at = (size_t(part) * width + channel) * (c.conv - 1);
                float a = projected[part][channel] * w.data[size_t(channel) * c.conv + c.conv - 1];
                for (int j = 0; j < c.conv - 1; ++j)
                    a += s.kda.convolution[at + j] * w.data[size_t(channel) * c.conv + j];
                for (int j = 0; j < c.conv - 2; ++j)
                    s.kda.convolution[at + j] = s.kda.convolution[at + j + 1];
                if (c.conv > 1)
                    s.kda.convolution[at + c.conv - 2] = projected[part][channel];
                projected[part][channel] = bf16_ ? bf(a * sigmoid(a)) : a * sigmoid(a);
            }
        }
        auto g = matvec(p + "f_b_proj", matvec(p + "f_a_proj", x)), beta = matvec(p + "b_proj", x),
             dt = weights_.tensor(p + "dt_bias").data, alog = weights_.tensor(p + "A_log").data;
        check(g.size() == size_t(width) && dt.size() == g.size() && alog.size() == size_t(c.linear_heads),
              "KDA gate geometry mismatch");
        for (int h = 0; h < c.linear_heads; ++h)
            for (int d = 0; d < c.linear_dim; ++d) {
                size_t i = size_t(h) * c.linear_dim + d;
                g[i] = c.gate_lower * sigmoid(std::exp(alog[h]) * (g[i] + dt[i]));
            }
        for (float &v : beta)
            v = bf16_ ? bf(sigmoid(v)) : sigmoid(v);
        auto y = round(kda_step(projected[0], projected[1], projected[2], g, beta, s.kda.recurrent,
                                c.linear_heads, c.linear_dim));
        auto gate = matvec(p + "g_b_proj", matvec(p + "g_a_proj", x)),
             w = weights_.tensor(p + "o_norm.weight").data;
        check(w.size() == size_t(c.linear_dim) && gate.size() == y.size(),
              "KDA output norm geometry mismatch");
        for (int h = 0; h < c.linear_heads; ++h) {
            double ss = 0;
            for (int d = 0; d < c.linear_dim; ++d)
                ss += double(y[h * c.linear_dim + d]) * y[h * c.linear_dim + d];
            float inv = float(1 / std::sqrt(ss / c.linear_dim + c.eps));
            for (int d = 0; d < c.linear_dim; ++d) {
                size_t i = size_t(h) * c.linear_dim + d;
                y[i] = y[i] * inv * w[d] * sigmoid(gate[i]);
            }
        }
        return matvec(p + "o_proj", round(y));
    }
    auto qr = norm(matvec(p + "q_a_proj", x), p + "q_a_layernorm.weight", c.eps),
         q = matvec(p + "q_b_proj", qr);
    auto latent = norm(matvec(p + "kv_a_proj_with_mqa", x), p + "kv_a_layernorm.weight", c.eps);
    check(latent.size() == size_t(c.kvrank) && q.size() == size_t(c.heads) * c.qdim, "MLA geometry mismatch");
    s.latent.insert(s.latent.end(), latent.begin(), latent.end());
    std::vector<int> selected;
    if (c.indexers[l] == "shared") {
        int previous = l - 1;
        while (previous >= 0 && states_[previous].last_topk.empty())
            --previous;
        check(previous >= 0, "missing shared indexer state");
        selected = states_[previous].last_topk;
    } else {
        std::string ix = p + "indexer.";
        auto iq = matvec(ix + "wq_b", qr), ik = matvec(ix + "wk", x),
             gates = matvec(ix + "index_kpool_compress_gate", x),
             wg = weights_.tensor(ix + "k_norm.weight").data, bs = weights_.tensor(ix + "k_norm.bias").data,
             weights = matvec(ix + "weights_proj", x);
        check(iq.size() == size_t(c.index_heads) * c.index_dim && ik.size() == size_t(c.index_dim) &&
                  gates.size() == ik.size() && wg.size() == ik.size() && bs.size() == ik.size() &&
                  weights.size() == size_t(c.index_heads),
              "indexer geometry mismatch");
        double mean = std::accumulate(ik.begin(), ik.end(), 0.) / ik.size(), variance = 0;
        for (float a : ik)
            variance += (a - mean) * (a - mean);
        float inv = float(1 / std::sqrt(variance / ik.size() + 1e-6));
        for (size_t i = 0; i < ik.size(); ++i)
            ik[i] = float((ik[i] - mean) * inv) * wg[i] + bs[i];
        ik = round(ik);
        s.index_keys.insert(s.index_keys.end(), ik.begin(), ik.end());
        s.index_gates.insert(s.index_gates.end(), gates.begin(), gates.end());
        auto ape = weights_.tensor(ix + "index_kpool_compress_ape").data;
        check(ape.size() == size_t(c.pool) * c.index_dim, "indexer pool position shape mismatch");
        int pools = int(position_ + 1) / c.pool;
        Vec scores(pools);
        std::vector<int> order(pools);
        std::iota(order.begin(), order.end(), 0);
        for (int pool = 0; pool < pools; ++pool) {
            Vec key(c.index_dim);
            for (int d = 0; d < c.index_dim; ++d) {
                float top = -INFINITY;
                for (int t = 0; t < c.pool; ++t)
                    top = std::max(top, s.index_gates[(size_t(pool) * c.pool + t) * c.index_dim + d] +
                                            ape[t * c.index_dim + d]);
                float den = 0;
                for (int t = 0; t < c.pool; ++t)
                    den += std::exp(s.index_gates[(size_t(pool) * c.pool + t) * c.index_dim + d] +
                                    ape[t * c.index_dim + d] - top);
                for (int t = 0; t < c.pool; ++t) {
                    float prob = std::exp(s.index_gates[(size_t(pool) * c.pool + t) * c.index_dim + d] +
                                          ape[t * c.index_dim + d] - top) /
                                 den;
                    float term = (bf16_ ? bf(prob) : prob) *
                                 s.index_keys[(size_t(pool) * c.pool + t) * c.index_dim + d];
                    key[d] += bf16_ ? bf(term) : term;
                }
                if (bf16_)
                    key[d] = bf(key[d]);
            }
            for (int h = 0; h < c.index_heads; ++h) {
                double dot = 0;
                for (int d = 0; d < c.index_dim; ++d)
                    dot += double(iq[h * c.index_dim + d]) * key[d];
                scores[pool] += std::max(0.f, float(dot) / std::sqrt(float(c.index_dim))) * weights[h] /
                                std::sqrt(float(c.index_heads));
            }
        }
        int choose = std::min(pools, c.topk / c.pool);
        std::partial_sort(order.begin(), order.begin() + choose, order.end(), [&](int a, int b) {
            return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
        });
        for (int i = 0; i < choose; ++i)
            for (int t = 0; t < c.pool; ++t)
                selected.push_back(order[i] * c.pool + t);
        if (c.tail)
            for (int t = pools * c.pool; t <= int(position_); ++t)
                selected.push_back(t);
    }
    s.last_topk = selected;
    check(!selected.empty(), "sparse MLA selected no visible keys");
    const auto &kv = weights_.cached_linear(p + "kv_b_proj", bf16_);
    check(kv.shape == std::vector<size_t>{size_t(c.heads) * (c.qdim + c.vdim), size_t(c.kvrank)},
          "MLA expansion geometry mismatch");
    Vec out(size_t(c.heads) * c.vdim);
#pragma omp parallel for schedule(static)
    for (int h = 0; h < c.heads; ++h) {
        Vec query(c.kvrank), weighted(c.kvrank), scores(selected.size());
        for (int j = 0; j < c.kvrank; ++j) {
            double sum = 0;
            for (int d = 0; d < c.qdim; ++d)
                sum +=
                    double(q[h * c.qdim + d]) * kv.data[(size_t(h) * (c.qdim + c.vdim) + d) * c.kvrank + j];
            query[j] = float(sum);
        }
        float top = -INFINITY;
        for (size_t t = 0; t < selected.size(); ++t) {
            double score = 0;
            for (int j = 0; j < c.kvrank; ++j)
                score += double(query[j]) * s.latent[size_t(selected[t]) * c.kvrank + j];
            scores[t] = float(score) / std::sqrt(float(c.qdim));
            top = std::max(top, scores[t]);
        }
        float den = 0;
        for (float &score : scores) {
            score = std::exp(score - top);
            den += score;
        }
        for (size_t t = 0; t < selected.size(); ++t) {
            float prob = scores[t] / den;
            if (bf16_)
                prob = bf(prob);
            for (int j = 0; j < c.kvrank; ++j)
                weighted[j] += prob * s.latent[size_t(selected[t]) * c.kvrank + j];
        }
        for (int d = 0; d < c.vdim; ++d) {
            double sum = 0;
            for (int j = 0; j < c.kvrank; ++j)
                sum += double(weighted[j]) *
                       kv.data[(size_t(h) * (c.qdim + c.vdim) + c.qdim + d) * c.kvrank + j];
            out[h * c.vdim + d] = float(sum);
        }
    }
    return matvec(p + "o_proj", round(out));
}
Vec Model::step(int token) {
    if (backend_) {
        auto logits = backend_->step(token);
        ++position_;
        return logits;
    }
    const auto &c = config;
    check(token >= 0 && token < c.vocab && position_ < size_t(c.max_context),
          "token/context outside model limits");
    auto embed = round(weights_.row(weights_.prefix + "embed_tokens.weight", token, c.hidden));
    Vec x(size_t(c.streams) * c.hidden);
    for (int h = 0; h < c.streams; ++h)
        std::copy(embed.begin(), embed.end(), x.begin() + size_t(h) * c.hidden);
    for (int l = 0; l < c.layers; ++l) {
        std::string p = weights_.prefix + "layers." + std::to_string(l) + ".";
        for (bool feedforward : {false, true}) {
            std::string hc = feedforward ? "hc_ffn_" : "hc_attn_";
            auto h = hyper_pre(x, weights_.tensor(p + hc + "fn"), weights_.tensor(p + hc + "base").data,
                               weights_.tensor(p + hc + "scale").data, c);
            auto n =
                norm(round(h.collapsed),
                     p + (feedforward ? "post_attention_layernorm.weight" : "input_layernorm.weight"), c.eps);
            auto y = feedforward ? mlp(l, n) : attention(l, n);
            if (bf16_) {
                for (float &v : h.post)
                    v = bf(v);
                for (float &v : h.mixing)
                    v = bf(v);
            }
            x = round(hyper_post(x, y, h, c));
        }
    }
    Vec hidden(c.hidden);
    for (int i = 0; i < c.streams; ++i)
        for (int d = 0; d < c.hidden; ++d)
            hidden[d] += x[size_t(i) * c.hidden + d] / c.streams;
    ++position_;
    return matvec("lm_head", norm(round(hidden), weights_.prefix + "norm.weight", c.eps));
}
} // namespace glm53
