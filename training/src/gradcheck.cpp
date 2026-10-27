// Real-data single-layer gradient capture for numerical A/B checks (for
// example QTIP_TRAIN_STORAGE=fp32 against bf16, or two builds).
#include "train.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <tuple>
using namespace qat;
namespace {
std::vector<float> read_floats(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    require(bool(f), "cannot open " + p.string());
    f.seekg(0, std::ios::end);
    size_t bytes = size_t(f.tellg());
    f.seekg(0);
    std::vector<float> v(bytes / 4);
    f.read(reinterpret_cast<char *>(v.data()), std::streamsize(bytes));
    return v;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc >= 5 && std::string(argv[2]) == "compare") {
            Model m(argv[1]);
            auto a = read_floats(argv[3]), b = read_floats(argv[4]);
            require(a.size() == b.size() && a.size() >= m.initial.size(), "capture size mismatch");
            // Kind -> (squared error, squared reference, dot, squared candidate).
            std::map<std::string, std::array<double, 4>> kinds;
            int layer = argc > 5 ? std::stoi(argv[5]) : 0;
            std::string prefix = "model.layers." + std::to_string(layer) + ".";
            for (auto &p : m.params) {
                if (!p.name.starts_with(prefix))
                    continue;
                auto kind = p.name.substr(prefix.size());
                auto dot = kind.find('.');
                kind = kind.find("proj") != std::string::npos
                           ? kind.substr(kind.rfind('.') + 1) + ":" + kind.substr(0, kind.rfind('.'))
                           : kind.substr(0, dot);
                for (auto key : {kind, kind.substr(0, kind.find(':'))})
                    for (size_t i = p.offset; i < p.offset + p.size; ++i) {
                        auto &e = kinds[key];
                        e[0] += double(a[i] - b[i]) * (a[i] - b[i]);
                        e[1] += double(a[i]) * a[i];
                        e[2] += double(a[i]) * b[i];
                        e[3] += double(b[i]) * b[i];
                    }
            }
            auto &dx = kinds["layer_input_gradient"];
            for (size_t i = m.initial.size(); i < a.size(); ++i) {
                dx[0] += double(a[i] - b[i]) * (a[i] - b[i]);
                dx[1] += double(a[i]) * a[i];
                dx[2] += double(a[i]) * b[i];
                dx[3] += double(b[i]) * b[i];
            }
            json out;
            for (auto &[k, e] : kinds)
                out[k] = {{"rel_error", std::sqrt(e[0] / std::max(e[1], 1e-300))},
                          {"cosine", e[2] / std::sqrt(std::max(e[1] * e[3], 1e-300))}};
            std::cout << out.dump(1) << '\n';
            return 0;
        }
        if (argc >= 6 && std::string(argv[2]) == "full") {
            // Whole-model loss and parameter gradient of one packed sequence.
            Model m(argv[1]);
            Dataset d(argv[3]);
            auto b = d.get(std::stoul(argv[4]));
            int device = argc > 6 ? std::stoi(argv[6]) : 0;
            Options o;
            if (argc > 7)
                o.rows = std::stoi(argv[7]);
            Replica r(m, device, int(d.length), size_t(31e9), true);
            double loss = r.run(b, o, b.targets(), true);
            auto grads = r.grad_values();
            double ss = 0;
            for (float g : grads)
                ss += double(g) * g;
            std::ofstream f(argv[5], std::ios::binary);
            f.write(reinterpret_cast<const char *>(grads.data()), std::streamsize(grads.size() * 4));
            require(bool(f), "capture write failed");
            std::cout << json({{"loss", loss / b.targets()}, {"grad_norm", std::sqrt(ss)}}).dump() << '\n';
            return 0;
        }
        if (argc >= 5 && std::string(argv[2]) == "top") {
            // Largest per-parameter contributions to the squared gradient-norm difference.
            Model m(argv[1]);
            auto a = read_floats(argv[3]), b = read_floats(argv[4]);
            require(a.size() >= m.initial.size() && b.size() >= m.initial.size(), "capture size mismatch");
            std::vector<std::tuple<double, double, double, std::string>> rows;
            double ta = 0, tb = 0;
            for (auto &p : m.params) {
                double na = 0, nb = 0, e = 0;
                for (size_t i = p.offset; i < p.offset + p.size; ++i) {
                    na += double(a[i]) * a[i];
                    nb += double(b[i]) * b[i];
                    e += double(a[i] - b[i]) * (a[i] - b[i]);
                }
                ta += na;
                tb += nb;
                rows.emplace_back(e, na, nb, p.name);
            }
            // Relative L2 error by parameter kind and excluding Wscale scalars.
            std::map<std::string, std::array<double, 2>> kinds;
            for (auto &[e, na, nb, name] : rows) {
                (void)nb;
                auto kind = name.substr(name.rfind('.') + 1);
                for (auto key : {kind, std::string(kind == "Wscale" ? "Wscale" : "all_but_Wscale")}) {
                    kinds[key][0] += e;
                    kinds[key][1] += na;
                }
                if (kind == "Wscale" && name.find("q_proj") == std::string::npos &&
                    name.find("k_proj") == std::string::npos) {
                    kinds["Wscale_not_qk"][0] += e;
                    kinds["Wscale_not_qk"][1] += na;
                }
            }
            if (std::getenv("QTIP_GRADCHECK_LAYERS")) {
                // Per-layer relative error of all non-Wscale parameters.
                std::map<int, std::array<double, 2>> layers;
                for (auto &[e, na, nb, name] : rows) {
                    (void)nb;
                    if (!name.starts_with("model.layers.") || name.ends_with("Wscale"))
                        continue;
                    int l = std::stoi(name.substr(13));
                    layers[l][0] += e;
                    layers[l][1] += na;
                }
                json per;
                for (auto &[l, v] : layers)
                    per.push_back(std::round(1000 * std::sqrt(v[0] / std::max(v[1], 1e-300))) / 1000);
                std::cout << json({{"per_layer_rel_error", per}}).dump() << '\n';
            }
            json summary = {{"norm_a", std::sqrt(ta)}, {"norm_b", std::sqrt(tb)}};
            for (auto &[k, v] : kinds)
                summary[k + "_rel_error"] = std::sqrt(v[0] / std::max(v[1], 1e-300));
            std::sort(rows.rbegin(), rows.rend());
            std::cout << summary.dump() << '\n';
            for (size_t i = 0; i < std::min<size_t>(rows.size(), 15); ++i)
                std::cout << json({{"param", std::get<3>(rows[i])},
                                   {"norm_a", std::sqrt(std::get<1>(rows[i]))},
                                   {"norm_b", std::sqrt(std::get<2>(rows[i]))},
                                   {"diff", std::sqrt(std::get<0>(rows[i]))}})
                                 .dump()
                          << '\n';
            return 0;
        }
        if (argc < 6) {
            std::cerr << "qtip_gradcheck MODEL DATA SEQUENCE SEQLEN OUT.f32 [LAYER=0] [DEVICE=0] [ROWS=1024] "
                         "[ARENA_GB=31]\n"
                         "qtip_gradcheck MODEL compare A.f32 B.f32 [LAYER=0]\n";
            return 1;
        }
        Model m(argv[1]);
        Dataset d(argv[2]);
        size_t index = std::stoul(argv[3]);
        int T = std::stoi(argv[4]), l = argc > 6 ? std::stoi(argv[6]) : 0, device = argc > 7 ? std::stoi(argv[7]) : 0,
            rows = argc > 8 ? std::stoi(argv[8]) : 1024;
        double arena = argc > 9 ? std::stod(argv[9]) : 31;
        require(T > 0 && size_t(T) <= d.length && l >= 0 && l < m.config.layers, "invalid sequence/layer");
        auto b = d.get(index);
        for (auto *v : {&b.tokens, &b.labels, &b.segments, &b.positions})
            v->resize(T);
        b.boundaries();
        Replica r(m, device, T, size_t(arena * 1e9), true);
        auto &o = *r.ops;
        // Layer l input: run the preceding layers forward from real embeddings.
        auto x = o.embedding(r.embedding, b);
        for (int i = 0; i <= l; ++i) {
            o.prefetch(i, i % 2, false);
            o.acquire(i % 2);
            if (i < l) {
                Layer layer(o, m, i, r.parameters, r.gradients);
                x = layer.forward(x, b, rows);
            }
            if (i < l)
                o.release(i % 2);
        }
        // Deterministic output gradient with a realistic scale (1 / tokens).
        auto dy = o.alloc(T, m.config.n);
        std::vector<float> g(dy.size());
        uint32_t seed = 12345;
        for (auto &v : g) {
            seed = seed * 1664525u + 1013904223u;
            v = (float(seed >> 8) / 8388608.f - 1.f) / float(T);
        }
        o.upload(dy, g);
        o.zero(r.gradients);
        Mat dx;
        {
            Layer layer(o, m, l, r.parameters, r.gradients);
            dx = layer.backward(x, dy, b, rows);
        }
        o.synchronize();
        o.release(l % 2);
        auto grads = r.grad_values();
        std::vector<float> dxv(dx.size());
        o.download(dx, dxv);
        std::ofstream f(argv[5], std::ios::binary);
        f.write(reinterpret_cast<const char *>(grads.data()), std::streamsize(grads.size() * 4));
        f.write(reinterpret_cast<const char *>(dxv.data()), std::streamsize(dxv.size() * 4));
        require(bool(f), "capture write failed");
        std::cout << json({{"layer", l}, {"seqlen", T}, {"sequence", index}, {"peak_bytes", o.peak_bytes()}}).dump()
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "qtip_gradcheck: " << e.what() << '\n';
        return 1;
    }
}
