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
Mat Layer::effective(const std::string &name) const {
    auto p = parameter(name), v = o_.alloc(p.rows, p.cols);
    o_.copy(v, p);
    o_.round(v);
    return v;
}
Mat Layer::projection(Mat x, int i, LinearSave *save) {
    auto su = effective(su_name(i));
    auto u = o_.had(x, true, su);
    auto z = o_.mm(u, weights_.at(i));
    if (i == 4 || i == 5)
        o_.round(z);
    auto c = o_.had(z);
    auto sv = parameter(linears[i] + ".SV"), scale = parameter(linears[i] + ".Wscale");
    auto svs = o_.alloc(1, sv.cols);
    o_.mul(svs, sv, scale);
    auto y = o_.alloc(c.rows, c.cols);
    o_.mul(y, c, svs);
    o_.round(y);
    if (save)
        *save = {x, c, i};
    return y;
}
Mat Layer::projection_backward(const LinearSave &s, Mat dy) {
    const auto &name = linears[s.index];
    auto sv = parameter(name + ".SV"), scale = parameter(name + ".Wscale");
    auto dc = o_.alloc(1, sv.cols, true);
    o_.colsum(dy, s.c, dc);
    auto temp = o_.alloc(1, sv.cols);
    o_.mul(temp, dc, scale);
    o_.add(gradient(name + ".SV"), gradient(name + ".SV"), temp);
    o_.scale_sum(dc, sv, gradient(name + ".Wscale"));
    auto svs = o_.alloc(1, sv.cols);
    o_.mul(svs, sv, scale);
    auto db = o_.had(dy, true, svs);
    auto da = o_.mm(db, weights_[s.index], true);
    auto dxsu = o_.had(da);
    o_.colsum(dxsu, s.x, gradient(su_name(s.index)));
    auto dx = o_.alloc(dxsu.rows, dxsu.cols);
    o_.mul(dx, dxsu, effective(su_name(s.index)));
    return dx;
}
Mat Layer::attention_half(Mat x, const Batch &b, AttnSave *save) {
    auto norm = o_.rms(x, effective(norm1), m_.config.eps);
    std::vector<LinearSave> qkv(3);
    auto q = projection(norm.y, 0, save ? &qkv[0] : nullptr),
         k = projection(norm.y, 1, save ? &qkv[1] : nullptr),
         v = projection(norm.y, 2, save ? &qkv[2] : nullptr);
    q = o_.rope(q, m_.config.heads, m_.config, b);
    k = o_.rope(k, m_.config.kv, m_.config, b);
    auto a = o_.attention(q, k, v, m_.config, b);
    LinearSave out;
    auto y = projection(a.out, 3, save ? &out : nullptr);
    auto h = o_.alloc(x.rows, x.cols);
    o_.add(h, x, y, true);
    if (save)
        *save = {norm, norm.y, h, a, qkv, out};
    return h;
}
Mat Layer::mlp(Mat h, MlpSave *save) {
    auto norm = o_.rms(h, effective(norm2), m_.config.eps);
    LinearSave gs, us, ds;
    auto g = projection(norm.y, 4, save ? &gs : nullptr), u = projection(norm.y, 5, save ? &us : nullptr);
    auto product = o_.swiglu(g, u);
    auto y = projection(product, 6, save ? &ds : nullptr);
    auto out = o_.alloc(h.rows, h.cols);
    o_.add(out, h, y, true);
    if (save)
        *save = {norm, gs, us, ds, g, u};
    return out;
}
Mat Layer::forward(Mat x, const Batch &b, int chunk) {
    auto h = attention_half(x, b, nullptr), out = o_.alloc(x.rows, x.cols);
    for (int start = 0; start < x.rows; start += chunk) {
        int end = std::min(start + chunk, x.rows);
        auto y = mlp(h.row(start, end), nullptr);
        o_.copy(out.row(start, end), y);
    }
    return out;
}
Mat Layer::backward(Mat x, Mat dy, const Batch &b, int chunk) {
    AttnSave saved;
    auto h = attention_half(x, b, &saved);
    auto dh = o_.alloc(dy.rows, dy.cols);
    o_.copy(dh, dy);
    for (int start = 0; start < x.rows; start += chunk) {
        int end = std::min(start + chunk, x.rows);
        MlpSave s;
        (void)mlp(h.row(start, end), &s);
        auto dp = projection_backward(s.down, dy.row(start, end));
        auto [dg, du] = o_.swiglu_backward(s.g, s.u, dp);
        auto dxn = projection_backward(s.gate, dg), dxu = projection_backward(s.up, du);
        o_.add(dxn, dxn, dxu);
        auto d = o_.rms_backward(h.row(start, end), s.norm, effective(norm2), dxn, gradient(norm2));
        o_.add(dh.row(start, end), dh.row(start, end), d);
    }
    h = {};
    saved.h = {};
    auto da = projection_backward(saved.out, dh);
    saved.out = {};
    auto d = o_.attention_backward(saved.attention, da, m_.config, b);
    saved.attention = {};
    da = {};
    d[0] = o_.rope(d[0], m_.config.heads, m_.config, b, true);
    d[1] = o_.rope(d[1], m_.config.kv, m_.config, b, true);
    auto dxn = projection_backward(saved.qkv[0], d[0]);
    d[0] = {};
    for (int i = 1; i < 3; ++i) {
        auto part = projection_backward(saved.qkv[i], d[i]);
        o_.add(dxn, dxn, part);
        d[i] = {};
    }
    saved.qkv.clear();
    auto dx = o_.rms_backward(x, saved.norm, effective(norm1), dxn, gradient(norm1));
    o_.add(dh, dh, dx);
    return dh;
}
} // namespace qat
