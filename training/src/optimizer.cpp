#include "train.hpp"
#include <algorithm>
#include <cmath>

namespace qat {
const char *group_name(Group g) {
    switch (g) {
    case Group::su:
        return "su";
    case Group::sv:
        return "sv";
    case Group::wscale:
        return "wscale";
    case Group::norm:
        return "norm";
    }
    return "?";
}
Group group_of(const std::string &name, bool norm) {
    if (norm)
        return Group::norm;
    if (name.ends_with(".SU"))
        return Group::su;
    if (name.ends_with(".SV"))
        return Group::sv;
    require(name.ends_with(".Wscale"), "unknown parameter group: " + name);
    return Group::wscale;
}
GradientStats gradient_stats(const std::vector<Param> &params, std::span<const float> g, size_t top) {
    GradientStats s;
    std::array<double, group_count> ss{};
    double total = 0;
    for (size_t k = 0; k < params.size(); ++k) {
        const auto &p = params[k];
        require(p.offset + p.size <= g.size(), "gradient shorter than parameter layout");
        double t = 0;
        for (size_t i = p.offset; i < p.offset + p.size; ++i) {
            require(std::isfinite(g[i]), "nonfinite gradient; checkpoint unchanged");
            t += double(g[i]) * g[i];
        }
        ss[int(p.group)] += t;
        total += t;
        s.top.emplace_back(std::sqrt(t), k);
    }
    // Alignment padding between tensors is zeroed and never written, so the
    // per-tensor sums are the whole-buffer norm.
    s.norm = std::sqrt(total);
    for (int i = 0; i < group_count; ++i)
        s.group[i] = std::sqrt(ss[i]);
    top = std::min(top, s.top.size());
    std::partial_sort(s.top.begin(), s.top.begin() + top, s.top.end(), std::greater<>());
    s.top.resize(top);
    return s;
}
json GradientStats::describe(const std::vector<Param> &params) const {
    json groups = json::object(), tensors = json::array();
    for (int i = 0; i < group_count; ++i)
        groups[group_name(Group(i))] = group[i];
    for (auto [n, k] : top) {
        const auto &name = params.at(k).name;
        json t = {{"name", name}, {"group", group_name(params[k].group)}, {"norm", n}, {"layer", nullptr}};
        std::string prefix = "model.layers.";
        if (name.starts_with(prefix)) {
            auto rest = name.substr(prefix.size());
            auto dot = rest.find('.');
            t["layer"] = std::stoi(rest.substr(0, dot));
            auto module = rest.substr(dot + 1);
            // "self_attn.q_proj.SU" -> "self_attn.q_proj"; "input_layernorm.weight" -> "input_layernorm".
            t["linear"] = module.substr(0, module.rfind('.'));
        } else
            t["linear"] = name.substr(0, name.rfind('.'));
        tensors.push_back(t);
    }
    return {{"grad_norms", groups}, {"grad_top", tensors}};
}
Optimizer::Optimizer(const Model &m, const Options &o) : model_(m), options_(o) {
    for (size_t k = 0; k < m.params.size(); ++k) {
        const auto &p = m.params[k];
        if (p.group != Group::wscale) {
            device_params_.push_back(p);
            continue;
        }
        require(p.size == 1, "Wscale must be a scalar");
        scales.push_back(k);
        initial.push_back(m.initial.at(p.offset));
        require(initial.back() > 0 && std::isfinite(initial.back()), "Wscale must be positive");
    }
    ratio.assign(scales.size(), 0.f);
    first.assign(scales.size(), 0.f);
    second.assign(scales.size(), 0.f);
}
float Optimizer::effective(size_t i) const {
    float w = float(double(initial[i]) * std::exp(double(ratio[i])));
    require(w > 0 && std::isfinite(w), "Wscale left the FP32 range");
    return w;
}
void Optimizer::to_log_scale(std::span<float> g) const {
    for (size_t i = 0; i < scales.size(); ++i) {
        auto &x = g[model_.params[scales[i]].offset];
        x = float(double(x) * effective(i));
    }
}
std::array<float, group_count> Optimizer::clip(std::span<float> g, const GradientStats &s) const {
    std::array<float, group_count> factor;
    float global = options_.clip > 0 ? float(std::min(1., double(options_.clip) / (s.norm + 1e-6))) : 1.f;
    factor.fill(global);
    if (global != 1.f)
        for (float &x : g)
            x *= global;
    return factor;
}
void Optimizer::step(const std::vector<Replica *> &replicas, std::span<const float> g, uint64_t step,
                     float rate) {
    require(step > 0, "optimizer steps are 1-based");
    // Same constants and FP32 operation order as the device AdamW (no decay on Wscale).
    float b1 = 1 - std::pow(.9f, float(step)), b2 = 1 - std::pow(.999f, float(step));
    float lr = options_.lr * rate;
    for (size_t i = 0; i < scales.size(); ++i) {
        float gg = g[model_.params[scales[i]].offset];
        first[i] = .9f * first[i] + .1f * gg;
        second[i] = .999f * second[i] + .001f * gg * gg;
        ratio[i] -= lr * (first[i] / b1) / (std::sqrt(second[i] / b2) + 1e-8f);
    }
    for (auto *r : replicas) {
        r->ops->upload(r->gradients, g);
        r->ops->adam(r->parameters, r->gradients, r->first, r->second, device_params_, step, options_.lr * rate,
                     options_.norm_lr * rate, options_.decay, 0);
        publish(*r);
        r->ops->synchronize();
    }
}
void Optimizer::publish(Replica &r) const {
    for (size_t i = 0; i < scales.size(); ++i) {
        float w = effective(i);
        int offset = int(model_.params[scales[i]].offset);
        r.ops->upload(r.parameters.col(offset, offset + 1), std::span(&w, 1));
    }
}
} // namespace qat
