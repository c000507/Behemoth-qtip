#include "train.hpp"

namespace qat {
Layer::Layer(Ops &o, const Model &m, int layer, Mat p, Mat g)
    : o_(o), m_(m), data_(m.layers.at(layer)), parameters_(p), gradients_(g), weights_(o.decode(data_)) {}
Mat Layer::parameter(const std::string &name) const {
    const auto &p = m_.params.at(data_.params.at(name));
    return parameters_.col(int(p.offset), int(p.offset + p.size));
}
Mat Layer::gradient(const std::string &name) const {
    const auto &p = m_.params.at(data_.params.at(name));
    return gradients_.col(int(p.offset), int(p.offset + p.size));
}
// Parameters are fixed for the lifetime of a Layer (one pass), so the BF16
// effective SU/norm vectors are rounded once and reused by every MLP chunk.
// SV vectors are copied once into separately aligned storage for vector loads.
Mat Layer::effective(const std::string &name, bool rounded) const {
    auto key = (rounded ? "r:" : "c:") + name;
    auto it = effective_.find(key);
    if (it != effective_.end())
        return it->second;
    auto p = parameter(name), v = o_.alloc(p.rows, p.cols);
    o_.copy(v, p);
    if (rounded)
        o_.round(v);
    effective_.emplace(key, v);
    return v;
}
// Output transforms (and residual adds) of GEMM results z[j] = W_{first+j} u.
std::vector<Mat> Layer::epilogues(std::vector<Mat> z, int first, LinearSave *saves, Mat input, Mat residual,
                                  Mat dst) {
    std::vector<Mat> result;
    for (size_t j = 0; j < z.size(); ++j) {
        int i = first + int(j);
        Transform t;
        t.round_in = i == 4 || i == 5;
        t.keep_c = saves != nullptr;
        t.post = effective(linears[i] + ".SV", false);
        t.post_scale = parameter(linears[i] + ".Wscale");
        t.residual = residual;
        t.y = dst;
        auto r = o_.transform(z[j], t);
        z[j] = {};
        if (saves)
            saves[j] = {input, r.c, i};
        result.push_back(r.y);
    }
    return result;
}
Mat Layer::projection(Mat x, int i, LinearSave *save, Mat residual, Mat dst) {
    auto z = o_.had_mm(x, std::span(weights_).subspan(i, 1), effective(su_name(i)));
    return epilogues(std::move(z), i, save, x, residual, dst).front();
}
Transformed Layer::projections_backward(const LinearSave *s, int count, Mat *dy, bool rounded_dx) {
    int first = s[0].index;
    std::vector<Mat> sv, scale;
    for (int j = 0; j < count; ++j) {
        require(s[j].index == first + j && su_name(s[j].index) == su_name(first),
                "grouped projections must share SU");
        const auto &name = linears[s[j].index];
        sv.push_back(effective(name + ".SV", false));
        scale.push_back(parameter(name + ".Wscale"));
        o_.sv_grads(dy[j], s[j].c, sv.back(), scale.back(), gradient(name + ".SV"), gradient(name + ".Wscale"));
    }
    auto da = o_.had_mm_sum(std::span(dy, count), sv, scale, std::span(weights_).subspan(first, count));
    Transform t;
    t.keep_c = true;
    t.want_y = rounded_dx;
    if (rounded_dx)
        t.post = effective(su_name(first));
    auto r = o_.transform(da, t);
    da = {};
    o_.colsum(r.c, s[0].x, gradient(su_name(first)));
    return r;
}
Mat Layer::attention_half(Mat x, const Batch &b, AttnSave *save) {
    Norm norm;
    auto z = o_.rms_had_mm(x, effective(norm1), m_.config.eps, effective(su_name(0)),
                           std::span(weights_).subspan(0, 3), norm);
    std::vector<LinearSave> qkv(3);
    auto projected = epilogues(std::move(z), 0, save ? qkv.data() : nullptr, norm.y);
    auto q = projected[0], k = projected[1], v = projected[2];
    projected.clear();
    q = o_.rope(q, m_.config.heads, m_.config, b);
    k = o_.rope(k, m_.config.kv, m_.config, b);
    auto a = o_.attention(q, k, v, m_.config, b);
    LinearSave out;
    auto h = projection(a.out, 3, save ? &out : nullptr, x);
    if (save)
        *save = {norm, norm.y, h, a, qkv, out};
    return h;
}
Mat Layer::mlp(Mat h, MlpSave *save, Mat dst) {
    Norm norm;
    auto z = o_.rms_had_mm(h, effective(norm2), m_.config.eps, effective(su_name(4)),
                           std::span(weights_).subspan(4, 2), norm, true);
    LinearSave gu[2], ds;
    auto projected = epilogues(std::move(z), 4, save ? gu : nullptr, norm.y);
    auto g = projected[0], u = projected[1];
    projected.clear();
    auto product = o_.swiglu(g, u);
    auto out = projection(product, 6, save ? &ds : nullptr, h, dst);
    if (save)
        *save = {norm, gu[0], gu[1], ds, g, u};
    return out;
}
Mat Layer::forward(Mat x, const Batch &b, int chunk) {
    auto h = attention_half(x, b, nullptr), out = o_.act(x.rows, x.cols);
    for (int start = 0; start < x.rows; start += chunk) {
        int end = std::min(start + chunk, x.rows);
        (void)mlp(h.row(start, end), nullptr, out.row(start, end));
    }
    return out;
}
Mat Layer::backward(Mat x, Mat dy, const Batch &b, int chunk) {
    AttnSave saved;
    auto h = attention_half(x, b, &saved);
    // The residual-stream gradient stays FP32: it accumulates every chunk.
    auto dh = o_.alloc(dy.rows, dy.cols);
    o_.copy(dh, dy);
    for (int start = 0; start < x.rows; start += chunk) {
        int end = std::min(start + chunk, x.rows);
        MlpSave s;
        (void)mlp(h.row(start, end), &s);
        auto rows = dy.row(start, end);
        auto dp = projections_backward(&s.down, 1, &rows, false).c;
        auto [dg, du] = o_.swiglu_backward_su(s.g, s.u, dp, effective(su_name(6)));
        dp = {};
        s.g = {};
        s.u = {};
        s.down = {};
        LinearSave gu[2] = {s.gate, s.up};
        Mat dgu[2] = {dg, du};
        dg = {};
        du = {};
        s.gate = {};
        s.up = {};
        auto dxsu = projections_backward(gu, 2, dgu, false).c;
        dgu[0] = {};
        dgu[1] = {};
        gu[0] = {};
        gu[1] = {};
        o_.rms_backward_su(h.row(start, end), s.norm, effective(norm2), effective(su_name(4)), dxsu,
                           gradient(norm2), dh.row(start, end));
    }
    h = {};
    saved.h = {};
    auto da = projections_backward(&saved.out, 1, &dh, true).y;
    saved.out = {};
    auto d = o_.attention_backward(saved.attention, da, m_.config, b);
    saved.attention = {};
    da = {};
    d[0] = o_.rope(d[0], m_.config.heads, m_.config, b, true);
    d[1] = o_.rope(d[1], m_.config.kv, m_.config, b, true);
    auto dxsu = projections_backward(saved.qkv.data(), 3, d.data(), false).c;
    d.clear();
    saved.qkv.clear();
    o_.rms_backward_su(x, saved.norm, effective(norm1), effective(su_name(0)), dxsu, gradient(norm1), dh);
    return dh;
}
} // namespace qat
