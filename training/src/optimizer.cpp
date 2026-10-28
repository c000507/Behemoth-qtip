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
} // namespace qat
