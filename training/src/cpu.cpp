#include "train.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace qat {
namespace {
struct Cpu final : Ops {
    std::vector<Mat> checkpoints;
    explicit Cpu(bool round) { rounding = round; }
    Mat alloc(int r, int c, bool clear = false) override {
        require(r >= 0 && c > 0 && size_t(r) <= SIZE_MAX / size_t(c) / 4, "tensor size overflow");
        std::shared_ptr<void> p(new float[size_t(r) * c], [](void *p) { delete[] static_cast<float *>(p); });
        Mat m{p, static_cast<float *>(p.get()), r, c, c};
        if (clear)
            zero(m);
        return m;
    }
    void copy(Mat d, Mat s) override {
        require(d.rows == s.rows && d.cols == s.cols, "copy shape");
        for (int r = 0; r < d.rows; ++r)
            std::copy_n(s.p + size_t(r) * s.ld, s.cols, d.p + size_t(r) * d.ld);
    }
    void upload(Mat d, std::span<const float> s) override {
        require(d.size() == s.size(), "upload shape");
        for (int r = 0; r < d.rows; ++r)
            std::copy_n(s.data() + size_t(r) * d.cols, d.cols, d.p + size_t(r) * d.ld);
    }
    void download(Mat s, std::span<float> d) override {
        require(s.size() == d.size(), "download shape");
        for (int r = 0; r < s.rows; ++r)
            std::copy_n(s.p + size_t(r) * s.ld, s.cols, d.data() + size_t(r) * s.cols);
    }
    void zero(Mat d) override {
        for (int r = 0; r < d.rows; ++r)
            std::fill_n(d.p + size_t(r) * d.ld, d.cols, 0.f);
    }
    void add(Mat d, Mat a, Mat b, bool round = false) override {
        require(a.rows == d.rows && a.cols == d.cols && b.rows == d.rows && b.cols == d.cols, "add shape");
        for (int r = 0; r < d.rows; ++r)
            for (int c = 0; c < d.cols; ++c) {
                float v = a.p[size_t(r) * a.ld + c] + b.p[size_t(r) * b.ld + c];
                d.p[size_t(r) * d.ld + c] = round && rounding ? bf(v) : v;
            }
    }
    void mul(Mat d, Mat a, Mat b) override {
        require(d.rows == a.rows && d.cols == a.cols && (b.rows == 1 || b.rows == d.rows) &&
                    (b.cols == 1 || b.cols == d.cols),
                "multiply shape");
        for (int r = 0; r < d.rows; ++r)
            for (int c = 0; c < d.cols; ++c)
                d.p[size_t(r) * d.ld + c] = a.p[size_t(r) * a.ld + c] *
                                            b.p[size_t(b.rows == 1 ? 0 : r) * b.ld + (b.cols == 1 ? 0 : c)];
    }
    void round(Mat d) override {
        if (rounding)
            for (int r = 0; r < d.rows; ++r)
                for (int c = 0; c < d.cols; ++c)
                    d.p[size_t(r) * d.ld + c] = bf(d.p[size_t(r) * d.ld + c]);
    }
    void colsum(Mat a, Mat b, Mat g) override {
        require(a.rows == b.rows && a.cols == b.cols && g.rows == 1 && g.cols == a.cols, "colsum shape");
        for (int c = 0; c < a.cols; ++c) {
            double v = 0;
            for (int r = 0; r < a.rows; ++r)
                v += double(a.p[size_t(r) * a.ld + c]) * b.p[size_t(r) * b.ld + c];
            g.p[c] += float(v);
        }
    }
    void scale_sum(Mat a, Mat b, Mat g) override {
        require(a.rows == b.rows && a.cols == b.cols && g.size() == 1, "scalar reduce shape");
        double v = 0;
        for (int r = 0; r < a.rows; ++r)
            for (int c = 0; c < a.cols; ++c)
                v += double(a.p[size_t(r) * a.ld + c]) * b.p[size_t(r) * b.ld + c];
        g.p[0] += float(v);
    }
    Mat had(Mat x, bool transpose = false, Mat pre = {}) override {
        auto y = alloc(x.rows, x.cols);
        std::vector<double> row(x.cols);
        for (int r = 0; r < x.rows; ++r) {
            for (int c = 0; c < x.cols; ++c)
                row[c] = double(x.p[size_t(r) * x.ld + c]) * (pre.p ? pre.p[c] : 1.f);
            behemoth::qtip::hadamard(row.data(), x.cols, transpose);
            for (int c = 0; c < x.cols; ++c)
                y.p[size_t(r) * y.ld + c] = float(row[c]);
        }
        return y;
    }
    Norm rms(Mat x, Mat w, float eps) override {
        Norm a{alloc(x.rows, x.cols), alloc(x.rows, 1)};
        for (int r = 0; r < x.rows; ++r) {
            double ss = 0;
            for (int c = 0; c < x.cols; ++c)
                ss += double(x.p[size_t(r) * x.ld + c]) * x.p[size_t(r) * x.ld + c];
            float rr = float(1 / std::sqrt(ss / x.cols + eps));
            a.r.p[r] = rr;
            for (int c = 0; c < x.cols; ++c) {
                float v = x.p[size_t(r) * x.ld + c] * rr * w.p[c];
                a.y.p[size_t(r) * a.y.ld + c] = rounding ? bf(v) : v;
            }
        }
        return a;
    }
    Mat rms_backward(Mat x, Norm a, Mat w, Mat dy, Mat dw) override {
        auto dx = alloc(x.rows, x.cols);
        for (int r = 0; r < x.rows; ++r) {
            double dot = 0;
            float rr = a.r.p[r];
            for (int c = 0; c < x.cols; ++c) {
                float xh = x.p[size_t(r) * x.ld + c] * rr;
                dot += double(dy.p[size_t(r) * dy.ld + c]) * w.p[c] * xh;
                dw.p[c] += dy.p[size_t(r) * dy.ld + c] * xh;
            }
            for (int c = 0; c < x.cols; ++c) {
                float xh = x.p[size_t(r) * x.ld + c] * rr;
                dx.p[size_t(r) * dx.ld + c] =
                    rr * (dy.p[size_t(r) * dy.ld + c] * w.p[c] - xh * float(dot / x.cols));
            }
        }
        return dx;
    }
    Weight frozen(std::span<const uint16_t> data, int out, int in) override {
        require(data.size() == size_t(out) * in, "frozen shape");
        auto a = alloc(out, in);
        for (size_t j = 0; j < data.size(); ++j)
            a.p[j] = std::bit_cast<float>(uint32_t(data[j]) << 16);
        return {a.owner, a.p, out, in, false, false};
    }
    std::vector<Weight> decode(const LayerData &l) override {
        std::vector<Weight> w;
        for (auto &lin : l.linear) {
            auto a = alloc(lin.out, lin.in);
            auto cb = behemoth::qtip::make_codebook(lin.tlut);
            if (rounding)
                behemoth::qtip::decode_rows_bf16(lin, cb, 0, lin.out, a.p);
            else
                behemoth::qtip::decode_rows(lin, cb, 0, lin.out, a.p);
            w.push_back({a.owner, a.p, lin.out, lin.in, false, false});
        }
        return w;
    }
    Mat mm(Mat x, const Weight &w, bool transpose = false) override {
        int reduce = transpose ? w.out : w.in, cols = transpose ? w.in : w.out;
        require(x.cols == reduce, "GEMM shape");
        auto y = alloc(x.rows, cols);
        const float *p = static_cast<const float *>(w.p);
        for (int r = 0; r < x.rows; ++r)
            for (int c = 0; c < cols; ++c) {
                double sum = 0;
                for (int k = 0; k < reduce; ++k) {
                    float a = x.p[size_t(r) * x.ld + k];
                    if (rounding)
                        a = bf(a);
                    sum += double(a) * p[transpose ? size_t(k) * w.in + c : size_t(c) * w.in + k];
                }
                y.p[size_t(r) * y.ld + c] = float(sum);
            }
        return y;
    }
    Mat embedding(const Weight &w, const Batch &b) override {
        auto x = alloc(int(b.tokens.size()), w.in);
        const float *p = static_cast<const float *>(w.p);
        for (int r = 0; r < x.rows; ++r) {
            require(b.tokens[r] >= 0 && b.tokens[r] < w.out, "token outside vocabulary");
            std::copy_n(p + size_t(b.tokens[r]) * w.in, w.in, x.p + size_t(r) * x.ld);
        }
        return x;
    }
    Mat rope(Mat x, int heads, const Config &c, const Batch &b, bool backward = false) override {
        require(x.cols == heads * c.dim && x.rows == int(b.tokens.size()), "RoPE shape");
        auto y = alloc(x.rows, x.cols);
        for (int r = 0; r < x.rows; ++r)
            for (int h = 0; h < heads; ++h)
                for (int j = 0; j < c.dim / 2; ++j) {
                    double angle = b.positions[r] * std::pow(double(c.theta), -2. * j / c.dim);
                    float cs = float(std::cos(angle)), sn = float(std::sin(angle));
                    size_t a = size_t(r) * x.ld + h * c.dim + j, d = a + c.dim / 2;
                    float u = x.p[a], v = x.p[d];
                    size_t o = size_t(r) * y.ld + h * c.dim + j;
                    y.p[o] = u * cs + (backward ? v : -v) * sn;
                    y.p[o + c.dim / 2] = v * cs + (backward ? -u : u) * sn;
                }
        if (!backward)
            round(y);
        return y;
    }
    Mat swiglu(Mat g, Mat u) override {
        auto y = alloc(g.rows, g.cols);
        for (int r = 0; r < g.rows; ++r)
            for (int c = 0; c < g.cols; ++c) {
                float v = g.p[size_t(r) * g.ld + c];
                y.p[size_t(r) * y.ld + c] = v / (1 + std::exp(-v)) * u.p[size_t(r) * u.ld + c];
            }
        round(y);
        return y;
    }
    std::pair<Mat, Mat> swiglu_backward(Mat g, Mat u, Mat dy) override {
        auto dg = alloc(g.rows, g.cols), du = alloc(g.rows, g.cols);
        for (int r = 0; r < g.rows; ++r)
            for (int c = 0; c < g.cols; ++c) {
                float v = g.p[size_t(r) * g.ld + c], s = 1 / (1 + std::exp(-v)),
                      d = dy.p[size_t(r) * dy.ld + c];
                dg.p[size_t(r) * dg.ld + c] = d * u.p[size_t(r) * u.ld + c] * s * (1 + v * (1 - s));
                du.p[size_t(r) * du.ld + c] = d * v * s;
            }
        return {dg, du};
    }
    Attention attention(Mat q, Mat k, Mat v, const Config &c, const Batch &b) override {
        Attention a{q, k, v, alloc(q.rows, q.cols), alloc(c.heads, q.rows)};
        std::vector<double> scores(q.rows);
        double scale = 1 / std::sqrt(c.dim);
        for (int t = 0; t < q.rows; ++t)
            for (int h = 0; h < c.heads; ++h) {
                int kh = h / (c.heads / c.kv);
                double top = -INFINITY;
                for (int j = b.starts[t]; j <= t; ++j) {
                    double s = 0;
                    for (int d = 0; d < c.dim; ++d)
                        s += double(q.p[size_t(t) * q.ld + h * c.dim + d]) *
                             k.p[size_t(j) * k.ld + kh * c.dim + d];
                    scores[j] = s * scale;
                    top = std::max(top, scores[j]);
                }
                double den = 0;
                for (int j = b.starts[t]; j <= t; ++j)
                    den += std::exp(scores[j] - top);
                a.lse.p[size_t(h) * a.lse.ld + t] = float(top + std::log(den));
                for (int d = 0; d < c.dim; ++d) {
                    double s = 0;
                    for (int j = b.starts[t]; j <= t; ++j)
                        s += std::exp(scores[j] - top) / den * v.p[size_t(j) * v.ld + kh * c.dim + d];
                    a.out.p[size_t(t) * a.out.ld + h * c.dim + d] = float(s);
                }
            }
        round(a.out);
        return a;
    }
    std::vector<Mat> attention_backward(const Attention &a, Mat dy, const Config &c,
                                        const Batch &b) override {
        if (rounding) {
            auto effective = alloc(dy.rows, dy.cols);
            copy(effective, dy);
            round(effective);
            dy = effective;
        }
        auto dq = alloc(a.q.rows, a.q.cols, true), dk = alloc(a.k.rows, a.k.cols, true),
             dv = alloc(a.v.rows, a.v.cols, true);
        double scale = 1 / std::sqrt(c.dim);
        for (int t = 0; t < a.q.rows; ++t)
            for (int h = 0; h < c.heads; ++h) {
                int kh = h / (c.heads / c.kv);
                double delta = 0;
                for (int d = 0; d < c.dim; ++d)
                    delta += double(dy.p[size_t(t) * dy.ld + h * c.dim + d]) *
                             a.out.p[size_t(t) * a.out.ld + h * c.dim + d];
                for (int j = b.starts[t]; j <= t; ++j) {
                    double s = 0, dp = 0;
                    for (int d = 0; d < c.dim; ++d) {
                        s += double(a.q.p[size_t(t) * a.q.ld + h * c.dim + d]) *
                             a.k.p[size_t(j) * a.k.ld + kh * c.dim + d];
                        dp += double(dy.p[size_t(t) * dy.ld + h * c.dim + d]) *
                              a.v.p[size_t(j) * a.v.ld + kh * c.dim + d];
                    }
                    double p = std::exp(s * scale - a.lse.p[size_t(h) * a.lse.ld + t]),
                           ds = p * (dp - delta) * scale;
                    for (int d = 0; d < c.dim; ++d) {
                        dq.p[size_t(t) * dq.ld + h * c.dim + d] +=
                            float(ds * a.k.p[size_t(j) * a.k.ld + kh * c.dim + d]);
                        dk.p[size_t(j) * dk.ld + kh * c.dim + d] +=
                            float(ds * a.q.p[size_t(t) * a.q.ld + h * c.dim + d]);
                        dv.p[size_t(j) * dv.ld + kh * c.dim + d] +=
                            float(p * dy.p[size_t(t) * dy.ld + h * c.dim + d]);
                    }
                }
            }
        round(dq);
        round(dk);
        round(dv);
        return {dq, dk, dv};
    }
    double cross_entropy(Mat x, std::span<const int32_t> labels, float denominator, bool backward) override {
        require(labels.size() == size_t(x.rows) && denominator > 0, "loss shape");
        double total = 0;
        for (int r = 0; r < x.rows; ++r) {
            int label = labels[r];
            float *row = x.p + size_t(r) * x.ld;
            if (label == -100) {
                if (backward)
                    std::fill_n(row, x.cols, 0.f);
                continue;
            }
            require(label >= 0 && label < x.cols, "target outside vocabulary");
            double top = *std::max_element(row, row + x.cols), sum = 0;
            for (int c = 0; c < x.cols; ++c)
                sum += std::exp(double(row[c]) - top);
            double lse = top + std::log(sum);
            total += lse - row[label];
            if (backward)
                for (int c = 0; c < x.cols; ++c)
                    row[c] = float((std::exp(double(row[c]) - lse) - (c == label ? 1. : 0.)) / denominator);
        }
        return total;
    }
    void adam(Mat p, Mat g, Mat m, Mat v, const std::vector<Param> &params, uint64_t step, float lr,
              float nlr, float decay, float clip) override {
        double ss = 0;
        for (size_t i = 0; i < g.size(); ++i) {
            require(std::isfinite(g.p[i]), "nonfinite gradient");
            ss += double(g.p[i]) * g.p[i];
        }
        float factor = clip > 0 ? float(std::min(1., double(clip) / (std::sqrt(ss) + 1e-6))) : 1.f;
        float b1 = 1 - std::pow(.9f, float(step)), b2 = 1 - std::pow(.999f, float(step));
        for (auto &a : params)
            for (size_t j = 0; j < a.size; ++j) {
                size_t i = a.offset + j;
                float gg = g.p[i] * factor;
                m.p[i] = .9f * m.p[i] + .1f * gg;
                v.p[i] = .999f * v.p[i] + .001f * gg * gg;
                float rate = a.norm ? nlr : lr;
                p.p[i] *= 1 - rate * (a.norm ? decay : 0);
                p.p[i] -= rate * (m.p[i] / b1) / (std::sqrt(v.p[i] / b2) + 1e-8f);
            }
    }
    void configure_streaming(const Model &m, int) override { checkpoints.resize(m.layers.size()); }
    Mat checkpoint(int layer, Mat x, bool save) override {
        if (save) {
            if (!checkpoints.at(layer).p)
                checkpoints[layer] = alloc(x.rows, x.cols);
            copy(checkpoints[layer], x);
            return {};
        }
        auto y = alloc(checkpoints.at(layer).rows, checkpoints.at(layer).cols);
        copy(y, checkpoints[layer]);
        return y;
    }
};
} // namespace
std::unique_ptr<Ops> cpu_ops(bool rounding) { return std::make_unique<Cpu>(rounding); }
#ifndef QTIP_TRAIN_HIP
std::unique_ptr<Ops> hip_ops(int, size_t) {
    throw std::runtime_error("this build has no HIP backend; rebuild with -DQTIP_TRAIN_HIP=ON");
}
#endif
} // namespace qat
