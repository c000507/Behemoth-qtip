#include "common.hpp"
#include "safetensors.hpp"
#include "train.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <unistd.h>
using namespace qat;
namespace qat {
void write_tiny_model(const fs::path &p, int layers) {
    fs::create_directories(p);
    json c = {{"hidden_size", 64},
              {"intermediate_size", 96},
              {"num_attention_heads", 4},
              {"num_key_value_heads", 2},
              {"head_dim", 16},
              {"num_hidden_layers", layers},
              {"vocab_size", 96},
              {"rms_norm_eps", 1e-5},
              {"rope_theta", 10000},
              {"hidden_act", "silu"},
              {"quip_params",
               {{"L", 16},
                {"K", 2},
                {"V", 2},
                {"tlut_bits", 9},
                {"decode_mode", "quantlut_sym"},
                {"td_x", 16},
                {"td_y", 16}}}};
    durable_json(p / "config.json", c);
    qtip::SafeTensorsWriter w;
    json index = {{"weight_map", json::object()}};
    auto add = [&](std::string name, std::string type, std::vector<int64_t> shape, const void *data,
                   size_t bytes) {
        w.add(name, type, shape, data, bytes);
        index["weight_map"][name] = "model.safetensors";
    };
    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0, .13f);
    auto dense = [&](std::string name, int r, int cols, bool norm) {
        std::vector<uint16_t> v(size_t(r) * cols);
        for (auto &x : v)
            x = behemoth::BFloat16(norm ? 1.f : normal(rng)).bits;
        add(name, "BF16", cols == 1 ? std::vector<int64_t>{r} : std::vector<int64_t>{r, cols}, v.data(),
            v.size() * 2);
    };
    for (int l = 0; l < layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".";
        std::vector<uint16_t> lut(1024);
        for (auto &x : lut)
            x = qtip::f32_to_fp16(normal(rng));
        for (int i = 0; i < 7; ++i) {
            int out = i == 1 || i == 2 ? 32 : i == 4 || i == 5 ? 96 : 64, in = i == 6 ? 96 : 64;
            std::string b = prefix + linears[i] + ".";
            std::vector<uint16_t> codes(size_t(out) * in / 8);
            for (auto &x : codes)
                x = uint16_t(rng());
            add(b + "trellis", "I16", {out / 16 * (in / 16), 32}, codes.data(), codes.size() * 2);
            add(b + "tlut", "F16", {512, 2}, lut.data(), lut.size() * 2);
            std::vector<uint16_t> su(in);
            for (int j = 0; j < in; ++j)
                su[j] = behemoth::BFloat16(j % 3 == 0 ? -1.f : 1.f).bits;
            add(b + "SU", "BF16", {in}, su.data(), su.size() * 2);
            std::vector<float> sv(out, .4f);
            for (int j = 0; j < out; ++j)
                if (j % 2)
                    sv[j] *= -1;
            add(b + "SV", "F32", {out}, sv.data(), sv.size() * 4);
            int64_t rcp = 0, tp = 8;
            add(b + "rcp", "I64", {}, &rcp, 8);
            add(b + "tp_rank", "I64", {}, &tp, 8);
        }
        dense(prefix + norm1, 64, 1, true);
        dense(prefix + norm2, 64, 1, true);
    }
    dense("model.norm.weight", 64, 1, true);
    dense("model.embed_tokens.weight", 96, 64, false);
    dense("lm_head.weight", 96, 64, false);
    w.write((p / "model.safetensors").string());
    durable_json(p / "model.safetensors.index.json", index);
}
void write_tiny_data(const fs::path &p, int sequences) {
    fs::create_directories(p);
    json m = {{"format_version", 1},         {"split", "train"},
              {"nseq", sequences},           {"seqlen", 5},
              {"files", json::object()},     {"training_conversations", {0, 1}},
              {"holdout_conversations", {2}}};
    std::vector<int32_t> tokens, segments, positions;
    std::vector<uint8_t> loss;
    for (int s = 0; s < sequences; ++s)
        for (int i = 0; i < 5; ++i) {
            tokens.push_back(3 + (s + i) % 9);
            segments.push_back(i < 3 ? 0 : 1);
            positions.push_back(i < 3 ? i : i - 3);
            loss.push_back(i != 0);
        }
    auto write = [&](const char *n, const void *d, size_t bytes) {
        std::ofstream f(p / n, std::ios::binary);
        f.write(static_cast<const char *>(d), bytes);
        f.close();
        m["files"][n] = sha256(p / n);
    };
    write("tokens.i32", tokens.data(), tokens.size() * 4);
    write("segments.i32", segments.data(), segments.size() * 4);
    write("positions.i32", positions.data(), positions.size() * 4);
    write("loss.u8", loss.data(), loss.size());
    durable_json(p / "metadata.json", m);
}
} // namespace qat
void close(double a, double b, double abs = 2e-4, double rel = .02) {
    require(std::abs(a - b) < abs + rel * std::max(std::abs(a), std::abs(b)),
            "mismatch " + std::to_string(a) + " vs " + std::to_string(b));
}
std::vector<float> get(Ops &o, Mat x) {
    std::vector<float> v(x.size());
    o.download(x, v);
    return v;
}
int main(int argc, char **argv) {
    fs::path root = fs::temp_directory_path() / ("qtip-native-test-" + std::to_string(getpid()));
    try {
        fs::create_directories(root);
        write_tiny_model(root / "model");
        write_tiny_data(root / "data");
        Model m(root / "model");
        Dataset data(root / "data");
        auto b = data.get(0);
        require(b.labels == std::vector<int32_t>({4, 5, -100, 7, -100}), "shifted assistant/boundary mask");
        {
            // Per-group/per-tensor norms: every tensor gets a distinct known norm.
            std::vector<float> g(m.initial.size(), 0.f);
            std::array<double, group_count> expect{};
            double total = 0;
            for (size_t k = 0; k < m.params.size(); ++k) {
                auto &p = m.params[k];
                double norm = .01 * double(k + 1);
                float each = float(norm / std::sqrt(double(p.size)));
                double ss = 0;
                for (size_t i = 0; i < p.size; ++i) {
                    g[p.offset + i] = (i % 2 ? -1 : 1) * each;
                    ss += double(each) * each;
                }
                expect[int(p.group)] += ss;
                total += ss;
            }
            auto stats = gradient_stats(m.params, g);
            close(stats.norm, std::sqrt(total), 1e-9, 1e-9);
            for (int i = 0; i < group_count; ++i)
                close(stats.group[i], std::sqrt(expect[i]), 1e-9, 1e-9);
            require(stats.top.size() == 3 && stats.top[0].second == m.params.size() - 1 &&
                        stats.top[1].second == m.params.size() - 2 && stats.top[2].second == m.params.size() - 3,
                    "top tensors not ordered by norm");
            auto d = stats.describe(m.params);
            require(d["grad_top"][0]["name"] == "model.norm.weight" && d["grad_top"][0]["layer"].is_null() &&
                        d["grad_top"][0]["group"] == "norm",
                    "final norm description");
            require(d["grad_top"][1]["layer"] == 1 && d["grad_top"][1]["linear"] == "post_attention_layernorm",
                    "layer/linear description");
            for (auto k : {"su", "sv", "wscale", "norm"})
                require(d["grad_norms"].contains(k), std::string("missing group norm ") + k);
            int counts[group_count] = {};
            for (auto &p : m.params)
                ++counts[int(p.group)];
            require(counts[int(Group::su)] == 2 * 4 && counts[int(Group::sv)] == 2 * 7 &&
                        counts[int(Group::wscale)] == 2 * 7 && counts[int(Group::norm)] == 2 * 2 + 1,
                    "parameter group assignment");
            g[m.params[3].offset] = std::nanf("");
            bool rejected = false;
            try {
                gradient_stats(m.params, g);
            } catch (...) {
                rejected = true;
            }
            require(rejected, "nonfinite gradient accepted");
        }
        {
            // Group clipping limits each group independently; global mode scales all.
            std::vector<float> g(m.initial.size(), 0.f);
            for (auto &p : m.params)
                for (size_t j = 0; j < p.size; ++j)
                    g[p.offset + j] = p.group == Group::wscale ? 100.f : .001f;
            auto before = gradient_stats(m.params, g);
            Options go;
            Optimizer grouped(m, go);
            auto gg = g;
            auto f = grouped.clip(gg, before);
            auto after = gradient_stats(m.params, gg);
            close(after.group[int(Group::wscale)], 1, 1e-5, 1e-5);
            for (auto k : {Group::su, Group::sv, Group::norm}) {
                require(f[int(k)] == 1.f, "unclipped group scaled");
                close(after.group[int(k)], before.group[int(k)], 0, 1e-12);
            }
            go.group_clip[int(Group::wscale)] = 0;
            go.group_clip[int(Group::sv)] = 1e-4f;
            gg = g;
            f = grouped.clip(gg, before);
            after = gradient_stats(m.params, gg);
            require(f[int(Group::wscale)] == 1.f, "disabled group clip applied");
            double sv = before.group[int(Group::sv)];
            close(after.group[int(Group::sv)], 1e-4 * sv / (sv + 1e-6), 1e-12, 1e-5);
            Options gl;
            gl.clip_mode = Options::ClipMode::global;
            Optimizer global(m, gl);
            gg = g;
            f = global.clip(gg, before);
            close(gradient_stats(m.params, gg).norm, 1, 1e-5, 1e-5);
            for (int i = 0; i < group_count; ++i)
                require(f[i] == f[0] && f[0] < 1, "global clip must scale every group equally");
            // Per-group LR: the first Adam step moves each parameter by its LR.
            Options lo;
            lo.lr = .01f;
            lo.norm_lr = .002f;
            lo.wscale_lr = .003f;
            lo.decay = 0;
            Replica fresh(m, -1, 5, 0, false);
            Optimizer stepper(m, lo);
            std::vector<float> ones(m.initial.size(), 0.f);
            for (auto &p : m.params)
                std::fill_n(ones.begin() + p.offset, p.size, .5f);
            stepper.step({&fresh}, ones, 1, 1.f);
            auto moved = fresh.values();
            for (size_t k = 0; k < m.params.size(); ++k) {
                auto &p = m.params[k];
                for (size_t j = p.offset; j < p.offset + p.size; ++j) {
                    double delta = moved[j] - m.initial[j];
                    if (p.group == Group::wscale)
                        close(delta / m.initial[j], std::expm1(-.003), 1e-6, 1e-3);
                    else
                        close(delta, p.norm ? -.002 : -.01, 1e-6, 1e-3);
                }
            }
            for (float d : stepper.ratio)
                close(d, -.003, 1e-7, 1e-4);
        }
        auto o = cpu_ops();
        auto x = o->alloc(3, 96);
        std::vector<float> v(x.size());
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = float(std::sin(double(i) * .3));
        o->upload(x, v);
        auto y = o->had(x);
        auto z = o->had(y, true);
        auto out = get(*o, z);
        for (size_t i = 0; i < v.size(); ++i)
            close(v[i], out[i], 1e-5, 1e-5);
        auto h28 = o->alloc(1, 224);
        std::vector<float> v28(224);
        for (size_t i = 0; i < v28.size(); ++i)
            v28[i] = float(std::cos(i * .7));
        o->upload(h28, v28);
        auto inv28 = get(*o, o->had(o->had(h28), true));
        for (size_t i = 0; i < v28.size(); ++i)
            close(v28[i], inv28[i], 1e-5, 1e-5);
        auto decoded = o->decode(m.layers[0]);
        for (int i = 0; i < 7; ++i) {
            auto &lin = m.layers[0].linear[i];
            auto input = o->alloc(1, lin.in), su = o->alloc(1, lin.in), sv = o->alloc(1, lin.out);
            std::vector<float> values(lin.in);
            for (int j = 0; j < lin.in; ++j)
                values[j] = float(std::sin(j * .37));
            o->upload(input, values);
            o->upload(su, lin.su);
            o->upload(sv, lin.sv);
            auto c = o->had(o->mm(o->had(input, true, su), decoded[i])), actual = o->alloc(1, lin.out);
            o->mul(actual, c, sv);
            auto oracle = behemoth::qtip::reference_forward(lin, values), got = get(*o, actual);
            for (size_t j = 0; j < got.size(); ++j)
                close(oracle[j], got[j], 2e-6, 1e-5);
        }
        Options opt;
        opt.model = root / "model";
        opt.data = root / "data";
        opt.work = root / "full";
        opt.max_tokens = 1000;
        opt.warmup = 0;
        opt.rows = 2;
        opt.loss_rows = 2;
        opt.rounding = false;
        opt.checkpoint_every = 0;
        Replica r(m, -1, 5, 0, false);
        double base = r.run(b, opt, b.targets(), true) / b.targets();
        auto analytic = r.grad_values(), params = r.values();
        const float eps = .002f;
        // A representative coordinate from every SU, SV, scale and norm parameter,
        // through the entire recomputed two-layer transformer and masked CE.
        for (auto &p : m.params) {
            size_t j = p.offset + p.size / 3;
            auto pert = params;
            pert[j] += eps;
            r.ops->upload(r.parameters, pert);
            double hi = r.run(b, opt, 1, false);
            pert[j] -= 2 * eps;
            r.ops->upload(r.parameters, pert);
            double lo = r.run(b, opt, 1, false);
            close((hi - lo) / (2 * eps * b.targets()), analytic[j], 8e-5, .035);
        }
        r.ops->upload(r.parameters, params);
        {
            // Wscale is trained as d = log(Wscale / Wscale0): check dL/dd against
            // finite differences in d, away from d = 0 so the chain rule must use
            // the current (not initial) Wscale.
            Optimizer fd(m, opt);
            for (size_t i = 0; i < fd.scales.size(); ++i)
                fd.ratio[i] = .2f * float(int(i % 3) - 1);
            fd.publish(r);
            auto shifted = r.values();
            r.run(b, opt, b.targets(), true);
            auto g = r.grad_values();
            fd.to_log_scale(g);
            size_t checked = 0;
            for (size_t i = 0; i < fd.scales.size(); ++i) {
                size_t j = m.params[fd.scales[i]].offset;
                require(shifted[j] == fd.effective(i), "published Wscale differs from Wscale0 * exp(d)");
                auto pert = shifted;
                float d = fd.ratio[i];
                fd.ratio[i] = d + eps;
                pert[j] = fd.effective(i);
                r.ops->upload(r.parameters, pert);
                double hi = r.run(b, opt, 1, false);
                fd.ratio[i] = d - eps;
                pert[j] = fd.effective(i);
                r.ops->upload(r.parameters, pert);
                double lo = r.run(b, opt, 1, false);
                fd.ratio[i] = d;
                close((hi - lo) / (2 * eps * b.targets()), g[j], 8e-5, .035);
                checked += std::abs(g[j]) > 1e-3;
            }
            require(checked * 2 > fd.scales.size(), "degenerate Wscale gradient check");
            r.ops->upload(r.parameters, params);
        }
        // Input adjoint through masked attention and chunked MLP recomputation.
        auto input = r.ops->embedding(r.embedding, b), upstream = r.ops->alloc(5, 64);
        std::vector<float> seed(upstream.size());
        for (size_t i = 0; i < seed.size(); ++i)
            seed[i] = float(std::sin(i * .17)) * .01f;
        r.ops->upload(upstream, seed);
        Layer layer(*r.ops, m, 0, r.parameters, r.gradients);
        auto dx = get(*r.ops, layer.backward(input, upstream, b, 2)), iv = get(*r.ops, input);
        auto objective = [&]() {
            auto result = get(*r.ops, layer.forward(input, b, 2));
            double sum = 0;
            for (size_t i = 0; i < result.size(); ++i)
                sum += double(result[i]) * seed[i];
            return sum;
        };
        for (size_t j : {size_t(3), size_t(79), size_t(217)}) {
            input.p[j] = iv[j] + eps;
            double hi = objective();
            input.p[j] = iv[j] - eps;
            double lo = objective();
            input.p[j] = iv[j];
            close((hi - lo) / (2 * eps), dx[j], 1e-5, .01);
        }
        // Unequal target counts must receive token weighting across replicas.
        auto short_batch = data.get(1);
        short_batch.labels.assign(5, -100);
        short_batch.labels[0] = 4;
        Replica other(m, -1, 5, 0, false);
        size_t total = b.targets() + short_batch.targets();
        r.run(b, opt, total, true);
        other.run(short_batch, opt, total, true);
        auto ga = r.grad_values(), gb = other.grad_values();
        r.run(b, opt, b.targets(), true);
        other.run(short_batch, opt, short_batch.targets(), true);
        auto local_a = r.grad_values(), local_b = other.grad_values();
        for (size_t i = 0; i < ga.size(); ++i)
            close(ga[i] + gb[i], (local_a[i] * b.targets() + local_b[i] * short_batch.targets()) / total,
                  1e-7, 1e-4);
        {
            // An untrained export reproduces the source model: SU/norms bit-exact,
            // SV = (SV / rms) * rms within FP32 rounding.
            Replica fresh(m, -1, 5, 0, false);
            m.export_model(root / "export-initial", fresh.values());
            Model same(root / "export-initial");
            for (size_t l = 0; l < m.layers.size(); ++l)
                for (int i = 0; i < 7; ++i) {
                    auto &a = m.layers[l].linear[i], &e = same.layers[l].linear[i];
                    require(a.su == e.su, "untrained SU export changed");
                    for (size_t j = 0; j < a.sv.size(); ++j)
                        close(a.sv[j], e.sv[j], 0, 1e-6);
                }
            require(same.initial.size() == m.initial.size(), "untrained export layout");
            for (auto &p : m.params)
                if (p.norm)
                    for (size_t j = p.offset; j < p.offset + p.size; ++j)
                        require(same.initial[j] == m.initial[j], "untrained norm export changed");
        }
        auto fit = opt;
        fit.lr = .025f;
        fit.norm_lr = .01f;
        fit.decay = 0;
        fit.clip = 1;
        Optimizer fitter(m, fit);
        for (int step = 1; step <= 80; ++step) {
            r.run(b, fit, b.targets(), true);
            auto g = r.grad_values();
            fitter.to_log_scale(g);
            fitter.clip(g, gradient_stats(m.params, g));
            fitter.step({&r}, g, step, 1);
        }
        double fitted = r.run(b, opt, 1, false) / b.targets();
        require(fitted < base * .45, "tiny overfit did not converge");
        double moved = 0;
        for (float d : fitter.ratio)
            moved = std::max(moved, double(std::abs(d)));
        require(moved > 1e-3, "Wscale did not train");
        auto trained = r.values();
        m.export_model(root / "export", trained);
        Model exported(root / "export");
        for (size_t i = 0; i < fitter.scales.size(); ++i) {
            // Export folds the effective Wscale into FP32 SV exactly as before.
            auto &scale = m.params[fitter.scales[i]];
            require(trained[scale.offset] == fitter.effective(i), "device Wscale differs from optimizer");
            auto &sv = m.params[fitter.scales[i] - 1];
            require(sv.group == Group::sv && sv.name.substr(0, sv.name.size() - 2) ==
                                                 scale.name.substr(0, scale.name.size() - 6),
                    "SV/Wscale pairing");
            auto layer = std::stoi(scale.name.substr(13));
            int linear = int(std::find_if(linears.begin(), linears.end(),
                                          [&](auto &n) { return scale.name.find(n) != std::string::npos; }) -
                             linears.begin());
            auto &e = exported.layers[layer].linear[linear].sv;
            for (size_t j = 0; j < sv.size; ++j)
                require(e[j] == trained[sv.offset + j] * fitter.effective(i), "exported SV is not SV * Wscale");
        }
        for (auto &it : m.index["weight_map"].items()) {
            std::string name = it.key();
            if (name.ends_with(".SU") || name.ends_with(".SV") || name.ends_with("layernorm.weight") ||
                name == "model.norm.weight")
                continue;
            auto a = m.file(name).bytes(name), e = exported.file(name).bytes(name);
            require(std::equal(a.begin(), a.end(), e.begin(), e.end()), "frozen tensor changed: " + name);
        }
        require(exported.layers[0].linear[0].su == exported.layers[0].linear[1].su, "tied SU export");
        {
            std::ostringstream log;
            auto *previous = std::cout.rdbuf(log.rdbuf());
            try {
                train(opt);
            } catch (...) {
                std::cout.rdbuf(previous);
                throw;
            }
            std::cout.rdbuf(previous);
            std::istringstream lines(log.str());
            std::string line;
            int steps = 0;
            while (std::getline(lines, line)) {
                auto j = json::parse(line);
                if (!j.contains("step"))
                    continue;
                ++steps;
                double ss = 0;
                for (auto k : {"su", "sv", "wscale", "norm"})
                    ss += std::pow(j["grad_norms"][k].get<double>(), 2);
                close(std::sqrt(ss), j["grad_norm"].get<double>(), 1e-9, 1e-6);
                require(j["grad_top"].size() == 3 && j["grad_top"][0]["norm"] >= j["grad_top"][1]["norm"],
                        "step record top tensors");
            }
            require(steps == 3, "step records missing");
        }
        auto partial = opt;
        partial.work = root / "resume";
        partial.stop_after = 1;
        train(partial);
        partial.stop_after = 0;
        partial.resume = partial.work / "latest.json";
        train(partial);
        auto state = [&](fs::path p) {
            auto n = read_json(p / "latest.json")["path"].get<std::string>();
            return std::make_unique<behemoth::SafeTensorFile>(p / n / "state.safetensors");
        };
        auto a = state(opt.work), e = state(partial.work);
        for (auto n : {"parameters", "adam_m", "adam_v", "wscale_log_ratio"}) {
            auto av = a->bytes(n), ev = e->bytes(n);
            require(std::equal(av.begin(), av.end(), ev.begin(), ev.end()), "resume is not bit exact");
        }
        auto sig = opt.signature(m, data);
        std::vector<std::unique_ptr<Replica>> rs;
        rs.push_back(std::make_unique<Replica>(m, -1, 5, 0, false));
        auto rejects = [&](const fs::path &path, const json &signature, const std::string &expect) {
            Optimizer scratch(m, opt);
            try {
                load_checkpoint(path, rs, scratch, signature);
            } catch (const std::exception &error) {
                require(std::string(error.what()).find(expect) != std::string::npos,
                        std::string("unexpected rejection: ") + error.what());
                return;
            }
            throw std::runtime_error("checkpoint accepted: expected " + expect);
        };
        auto changed = sig;
        changed["lr"] = .02;
        rejects(partial.work, changed, "differs");
        for (auto [key, value] : {std::pair<const char *, json>{"clip_mode", "global"}, {"wscale_lr", 1e-3}}) {
            changed = sig;
            changed[key] = value;
            rejects(partial.work, changed, "differs");
        }
        {
            // Format 3 (raw Wscale) conversion: build a v3 snapshot from a step-1
            // v4 snapshot by mapping the log-ratio state back to raw Wscale moments.
            auto v3src = opt;
            v3src.work = root / "v3src";
            v3src.stop_after = 1;
            train(v3src);
            auto name = read_json(v3src.work / "latest.json")["path"].get<std::string>();
            auto src = v3src.work / name;
            behemoth::SafeTensorFile f(src / "state.safetensors");
            auto vec = [&](const char *n) {
                auto v = f.floats(n);
                std::vector<float> out(v.size());
                for (size_t i = 0; i < out.size(); ++i)
                    out[i] = v[i];
                return out;
            };
            auto P = vec("parameters"), M = vec("adam_m"), V = vec("adam_v"), D = vec("wscale_log_ratio");
            auto M0 = M, V0 = V;
            Optimizer ref(m, opt);
            for (size_t i = 0; i < ref.scales.size(); ++i) {
                auto off = m.params[ref.scales[i]].offset;
                float w = P[off];
                M[off] = float(double(M[off]) / w);
                V[off] = float(double(V[off]) / (double(w) * w));
            }
            auto v3 = root / "v3";
            fs::create_directories(v3);
            qtip::SafeTensorsWriter w;
            w.add("parameters", "F32", {int64_t(P.size())}, P.data(), P.size() * 4);
            w.add("adam_m", "F32", {int64_t(M.size())}, M.data(), M.size() * 4);
            w.add("adam_v", "F32", {int64_t(V.size())}, V.data(), V.size() * 4);
            w.write((v3 / "state.safetensors").string());
            auto meta = read_json(src / "metadata.json");
            for (auto key : {"wscale_param", "clip_mode", "group_clip", "wscale_lr"})
                meta["signature"].erase(key);
            meta["signature"]["format"] = 3;
            meta["state_sha256"] = sha256(v3 / "state.safetensors");
            durable_json(v3 / "metadata.json", meta);
            Optimizer converted(m, opt);
            load_checkpoint(v3, rs, converted, sig);
            auto Q = rs[0]->values();
            std::vector<float> cm(Q.size()), cv(Q.size());
            rs[0]->ops->download(rs[0]->first, cm);
            for (size_t i = 0; i < converted.scales.size(); ++i) {
                auto off = m.params[converted.scales[i]].offset;
                close(converted.ratio[i], D[i], 1e-6, 1e-5);
                close(converted.first[i], M0[off], 1e-12, 1e-5);
                close(converted.second[i], V0[off], 1e-15, 1e-5);
                close(Q[off], P[off], 0, 1e-6);
                Q[off] = P[off];
                require(cm[off] == 0, "converted Wscale moment left in device slot");
            }
            require(Q == P, "converted parameters changed");
            // Training continues from the converted snapshot and lands next to the
            // uninterrupted run.
            auto cont = opt;
            cont.work = root / "v3work";
            cont.resume = v3;
            train(cont);
            auto x = state(opt.work), y = state(cont.work);
            auto xp = x->floats("parameters"), yp = y->floats("parameters");
            for (size_t i = 0; i < xp.size(); ++i)
                close(xp[i], yp[i], 1e-6, 1e-3);
            auto bad = meta;
            bad["signature"]["lr"] = .02;
            durable_json(v3 / "metadata.json", bad);
            rejects(v3, sig, "differs");
            bad = meta;
            bad["signature"]["format"] = 2;
            durable_json(v3 / "metadata.json", bad);
            rejects(v3, sig, "unsupported checkpoint format 2");
        }
        auto unsupervised = opt;
        unsupervised.data = root / "no-targets";
        unsupervised.work = root / "skipped";
        fs::copy(root / "data", unsupervised.data, fs::copy_options::recursive);
        {
            std::fstream mask(unsupervised.data / "loss.u8", std::ios::in | std::ios::out | std::ios::binary);
            char zero[5] = {};
            mask.write(zero, 5);
        }
        auto um = read_json(unsupervised.data / "metadata.json");
        um["files"]["loss.u8"] = sha256(unsupervised.data / "loss.u8");
        durable_json(unsupervised.data / "metadata.json", um);
        train(unsupervised);
        auto un = read_json(unsupervised.work / "latest.json")["path"].get<std::string>();
        auto progress = read_json(unsupervised.work / un / "metadata.json");
        require(progress["position"] == 3 && progress["step"] == 2, "unsupervised window not skipped");
        if (argc > 1 && std::string(argv[1]) == "--keep")
            std::cout << "fixtures: " << root << '\n';
        else
            fs::remove_all(root);
        std::cout << "native tests passed: gradients, masks, overfit " << base << " -> " << fitted
                  << ", frozen export, exact resume, identity rejection\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "native tests: " << e.what() << " (fixtures " << root << ")\n";
        return 1;
    }
}
