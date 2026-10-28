#include "train.hpp"
#include "safetensors.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace qat;
namespace {
void check(bool ok, const std::string &what) { require(ok, what); }
template<class F> void rejects(F f) { bool failed = false; try { f(); } catch (const std::exception &) { failed = true; } check(failed, "expected rejection"); }
std::vector<float> read(Ops &o, Mat x) { std::vector<float> out(x.size()); o.download(x, out); return out; }
void source_model(const Model &m, const fs::path &path) {
    fs::create_directory(path);
    auto config = m.config_json;
    config.erase("quip_params");
    config["quantization_config"] = {{"format", "float-quantized"}};
    durable_json(path / "config.json", config);
    qtip::SafeTensorsWriter weights, scales;
    json index = {{"weight_map", json::object()}};
    for (int l = 0; l < m.config.layers; ++l)
        for (int i = 0; i < 7; ++i) {
            const auto &w = m.layers[l].linear[i];
            std::vector<float> identity(size_t(w.in) * w.in);
            for (int j = 0; j < w.in; ++j) identity[size_t(j) * w.in + j] = 1;
            auto dense = behemoth::qtip::reference_forward(w, identity);
            std::vector<uint8_t> fp8(size_t(w.out) * w.in);
            std::vector<uint16_t> scale(w.out);
            for (int r = 0; r < w.out; ++r) {
                std::vector<float> row(w.in);
                for (int c = 0; c < w.in; ++c) row[c] = dense[size_t(c) * w.out + r] * 1.15f;
                auto q = behemoth::quantize(row);
                scale[r] = behemoth::BFloat16(q.scale).bits;
                std::copy(q.values.begin(), q.values.end(), fp8.begin() + size_t(r) * w.in);
            }
            auto base = "model.layers." + std::to_string(l) + "." + linears[i] + ".";
            weights.add(base + "weight", "F8_E4M3", {w.out, w.in}, fp8.data(), fp8.size());
            scales.add(base + "weight_scale", "BF16", {w.out}, scale.data(), scale.size() * 2);
            index["weight_map"][base + "weight"] = "weights.safetensors";
            index["weight_map"][base + "weight_scale"] = "scales.safetensors";
        }
    for (const auto &item : m.index.at("weight_map").items()) {
        if (!item.key().ends_with("layernorm.weight") && item.key() != "model.norm.weight" &&
            item.key() != "model.embed_tokens.weight" && item.key() != "lm_head.weight") continue;
        const auto &file = m.file(item.key());
        auto shape = file.shape(item.key()); auto raw = file.bytes(item.key());
        weights.add(item.key(), file.dtype(item.key()), {shape.begin(), shape.end()}, raw.data(), raw.size());
        index["weight_map"][item.key()] = "weights.safetensors";
    }
    weights.write((path / "weights.safetensors").string());
    scales.write((path / "scales.safetensors").string());
    durable_json(path / "model.safetensors.index.json", index);
}
void validation_data(const fs::path &path) {
    write_tiny_data(path, 1);
    for (const auto &name : {"tokens.i32", "segments.i32"}) {
        std::vector<int32_t> data(5);
        { std::ifstream f(path / name, std::ios::binary); f.read(reinterpret_cast<char *>(data.data()), 20); }
        for (auto &v : data) v += std::string(name) == "tokens.i32" ? 4 : 100;
        { std::ofstream f(path / name, std::ios::binary); f.write(reinterpret_cast<const char *>(data.data()), 20); }
    }
    auto metadata = read_json(path / "metadata.json");
    metadata["split"] = "heldout";
    metadata["training_conversations"] = json::array();
    metadata["evaluated_conversations"] = {100, 101};
    metadata["holdout_conversations"] = {100, 101};
    for (const auto &name : {"tokens.i32", "segments.i32"}) metadata["files"][name] = sha256(path / name);
    durable_json(path / "metadata.json", metadata);
}
void finite_difference(const Model &m, const Batch &batch) {
    Replica r(m, -1, int(batch.tokens.size()), size_t(1e9), false, false);
    auto &o = *r.ops;
    auto x = o.embedding(r.embedding, batch);
    auto target = o.alloc(x.rows, x.cols, true);
    auto y = Layer(o, m, 0, r.parameters, r.gradients).forward(x, batch, 3);
    auto dy = o.alloc(y.rows, y.cols);
    o.zero(r.gradients);
    o.mse(y, target, batch.segments, dy);
    (void)Layer(o, m, 0, r.parameters, r.gradients).backward(x, dy, batch, 3);
    auto g = r.grad_values();
    for (const auto &name : {"self_attn.q_proj.SU", "self_attn.q_proj.SV", "self_attn.q_proj.Wscale", norm1.c_str(), norm2.c_str()}) {
        auto p = m.params.at(m.layers[0].params.at(name));
        const float eps = .001f;
        double scores[2];
        for (int sign = 0; sign < 2; ++sign) {
            auto v = m.initial; v[p.offset] += sign ? eps : -eps; o.upload(r.parameters, v);
            auto out = Layer(o, m, 0, r.parameters, r.gradients).forward(x, batch, 3);
            scores[sign] = o.mse(out, target, batch.segments);
        }
        double fd = (scores[1] - scores[0]) / (2 * eps);
        check(std::abs(fd - g[p.offset]) < 2e-5 + .02 * std::max(std::abs(fd), std::abs(double(g[p.offset]))),
              std::string("mixed block gradient mismatch: ") + name);
    }
}
} // namespace
int main() {
    auto root = fs::temp_directory_path() / ("qtip-reconstruction-" + std::to_string(getpid()));
    try {
        fs::create_directory(root);
        write_tiny_model(root / "quantized", 2);
        write_tiny_data(root / "data", 2);
        validation_data(root / "heldout");
        Model original(root / "quantized");
        source_model(original, root / "teacher");
        rejects([&] { Model denied(root / "teacher"); });
        replace_kv_fp8(original.path, root / "teacher", root / "mixed");
        Model mixed(root / "mixed"), teacher(root / "teacher", true);
        for (int l = 0; l < mixed.config.layers; ++l) {
            for (int i = 0; i < 7; ++i) {
                const auto &w = mixed.layers[l].linear[i];
                if (i == 1 || i == 2) {
                    check(w.is_fp8() && w.trellis.empty() && w.su.empty(), "FP8 K/V retained QTIP payload");
                    check(w.fp8 == teacher.layers[l].linear[i].fp8 && w.fp8_scale == teacher.layers[l].linear[i].fp8_scale,
                          "source FP8 K/V changed in conversion");
                } else check(!w.is_fp8() && w.trellis == original.layers[l].linear[i].trellis, "unrelated quantized projection changed");
            }
            check(!mixed.layers[l].params.contains("self_attn.k_proj.Wscale") &&
                  !mixed.layers[l].params.contains("self_attn.v_proj.SV"), "FP8 K/V must stay frozen");
        }
        Dataset dataset(root / "data");
        finite_difference(mixed, dataset.get(0));
        auto ops = cpu_ops();
        auto a = ops->alloc(2, 3), b = ops->alloc(2, 3, true), g = ops->alloc(2, 3);
        ops->upload(a, std::vector<float>{1, 2, 3, 100, 200, 300});
        std::vector<int32_t> segments{0, -1};
        check(std::abs(ops->mse(a, b, segments, g) - 14. / 3) < 1e-6, "padding entered reconstruction MSE");
        auto gradient = read(*ops, g);
        check(gradient[3] == 0 && gradient[4] == 0 && gradient[5] == 0 && std::abs(gradient[0] - 2.f / 3) < 1e-6,
              "MSE normalization or padding gradient");
        rejects([&] { ops->mse(a, b, std::vector<int32_t>{-1, -1}); });
        ReconstructionOptions o;
        o.model = mixed.path; o.teacher = teacher.path; o.data = root / "data"; o.validation_data = root / "heldout";
        o.work = root / "work"; o.output = root / "reconstructed";
        o.device = o.teacher_device = -1; o.samples = 2; o.validation_samples = 1;
        o.steps = 16; o.eval_every = 2; o.rows = 3; o.lr = .02f; o.norm_lr = .002f;
        reconstruct(o);
        auto report = read_json(o.work / "reconstruction.json");
        check(report["blocks"].size() == 2 && report["next_layer"] == 2, "missing reconstructed blocks");
        bool improved = false;
        for (const auto &block : report["blocks"]) {
            double before = block["validation_mse_before"], after = block["validation_mse_after"];
            check(after <= before, "reconstruction accepted a worse held-out block");
            improved |= after < before;
        }
        check(improved, "teacher-guided reconstruction did not improve any block");
        Model fitted(o.output);
        for (int l = 0; l < 2; ++l)
            for (int i = 0; i < 7; ++i)
                check(i == 1 || i == 2 ? fitted.layers[l].linear[i].fp8 == mixed.layers[l].linear[i].fp8 &&
                                           fitted.layers[l].linear[i].fp8_scale == mixed.layers[l].linear[i].fp8_scale
                                      : fitted.layers[l].linear[i].trellis == mixed.layers[l].linear[i].trellis,
                      "reconstruction changed frozen weights");
        o.resume = true; o.output = root / "resumed";
        reconstruct(o);
        Model resumed(o.output);
        check(resumed.identity == fitted.identity, "completed-block resume changed the export");
        o.steps += 1; o.output = root / "bad-resume";
        rejects([&] { reconstruct(o); });
        auto overlap = read_json(root / "heldout" / "metadata.json");
        overlap["evaluated_conversations"] = {0}; durable_json(root / "heldout" / "metadata.json", overlap);
        o.resume = false; o.work = root / "bad-work";
        rejects([&] { reconstruct(o); });
        fs::remove_all(root);
        std::cout << "PASS mixed FP8 conversion, frozen export, gradients, masked MSE, sequential teacher reconstruction and resume\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << " (fixtures: " << root << ")\n";
        return 1;
    }
}
