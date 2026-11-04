#include "glm53.hpp"
#include "safetensors.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
namespace glm53 {
Weights::Weights(const fs::path &p)
    : path_(fs::canonical(p)), index_(read_json(path_ / "model.safetensors.index.json")) {
    if (has("model.language_model.embed_tokens.weight"))
        prefix = "model.language_model.";
    else if (has("model.embed_tokens.weight"))
        prefix = "model.";
    else
        throw std::runtime_error("GLM53 embedding tensor not found");
    bool packed = false;
    for (const auto &it : index_.at("weight_map").items()) {
        std::string name = it.value();
        check(fs::path(name).filename() == name, "unsafe shard path");
        packed |= it.key().ends_with(".weight_packed");
    }
    if (packed) {
        auto q = read_json(path_ / "config.json").at("glm53_quant");
        check(q.at("mlp_format") == "nvfp4" && q.at("block_size") == 16 && q.at("format_version") == 1,
              "unsupported packed GLM53 weight format");
    }
}
bool Weights::has(const std::string &n) const { return index_.at("weight_map").contains(n); }
const behemoth::SafeTensorFile &Weights::file(const std::string &n) const {
    std::string name = index_.at("weight_map").at(n);
    if (!files_.contains(name))
        files_[name] = std::make_shared<behemoth::SafeTensorFile>(path_ / name);
    return *files_.at(name);
}
Tensor Weights::tensor(const std::string &n) const {
    const auto &f = file(n);
    Tensor t{f.shape(n), {}};
    auto a = f.floats(n);
    t.data.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        t.data[i] = a[i];
        check(std::isfinite(t.data[i]), "nonfinite tensor " + n);
    }
    return t;
}
Vec Weights::row(const std::string &name, int index, int width) const {
    const auto &f = file(name);
    auto shape = f.shape(name);
    check(shape.size() == 2 && shape[1] == size_t(width) && index >= 0 && size_t(index) < shape[0],
          "embedding row outside tensor geometry");
    auto values = f.floats(name);
    Vec out(width);
    for (int i = 0; i < width; ++i) {
        out[i] = values[size_t(index) * width + i];
        check(std::isfinite(out[i]), "nonfinite embedding");
    }
    return out;
}
Tensor Weights::linear(const std::string &base) const {
    if (has(base + ".weight_packed")) {
        const std::string packed = base + ".weight_packed", scales = base + ".weight_scale";
        const auto &f = file(packed);
        auto shape = f.shape(packed);
        check(f.dtype(packed) == "U8" && shape.size() == 2 && !has(base + ".weight"),
              "invalid NVFP4 packed tensor");
        int r = int(shape[0]), c = int(shape[1] * 2);
        auto s = file(scales).shape(scales);
        check(s == std::vector<size_t>{size_t(r), size_t(c / 16)} && file(scales).dtype(scales) == "F8_E4M3",
              "invalid NVFP4 scale shape/type");
        auto global = tensor(base + ".weight_global_scale");
        check(global.data.size() == 1, "invalid NVFP4 global scale");
        auto a = f.bytes(packed), b = file(scales).bytes(scales);
        Nvfp4 w{{}, {}, global.data[0], r, c};
        w.packed.resize(a.size());
        w.scales.resize(b.size());
        std::memcpy(w.packed.data(), a.data(), a.size());
        std::memcpy(w.scales.data(), b.data(), b.size());
        return {{size_t(r), size_t(c)}, dequantize_nvfp4(w)};
    }
    auto name = base + ".weight";
    const auto &f = file(name);
    auto shape = f.shape(name);
    check(shape.size() == 2, "linear weight must be a matrix: " + name);
    if (f.dtype(name) != "F8_E4M3")
        return tensor(name);
    int r = int(shape[0]), c = int(shape[1]);
    auto bytes = f.bytes(name);
    Tensor out{shape, Vec(size_t(r) * c)};
    bool block = has(base + ".weight_scale_inv");
    auto scale = tensor(base + (block ? ".weight_scale_inv" : ".weight_scale"));
    check(block ? scale.shape == std::vector<size_t>{(size_t(r) + 127) / 128, (size_t(c) + 127) / 128}
                : scale.data.size() == size_t(r),
          "FP8 scale geometry mismatch: " + base);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) {
            size_t si = block ? size_t(i / 128) * ((c + 127) / 128) + j / 128 : i;
            float s = scale.data[si];
            check(s >= 0 && std::isfinite(s), "invalid FP8 weight scale");
            float v = behemoth::decode_e4m3(uint8_t(bytes[size_t(i) * c + j])) * s;
            check(std::isfinite(v), "nonfinite FP8 weight");
            out.data[size_t(i) * c + j] = float(behemoth::BFloat16(v));
        }
    return out;
}
namespace {
constexpr float levels[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
uint8_t fp4(float x) {
    float a = std::abs(x), best = INFINITY;
    int code = 0;
    for (int i = 0; i < 8; ++i) {
        float e = std::abs(a - levels[i]);
        if (e < best || (e == best && (i % 2 == 0))) {
            best = e;
            code = i;
        }
    }
    return uint8_t(code | (std::signbit(x) ? 8 : 0));
}
} // namespace
Nvfp4 quantize_nvfp4(const Vec &x, int r, int c) {
    check(r > 0 && c > 0 && c % 16 == 0 && x.size() == size_t(r) * c,
          "NVFP4 rows/cols must align to 16 input channels");
    float maximum = 0;
    for (float v : x) {
        check(std::isfinite(v), "nonfinite NVFP4 input");
        maximum = std::max(maximum, std::abs(v));
    }
    Nvfp4 w{{}, {}, maximum ? maximum / (448.f * 6) : 1.f, r, c};
    check(w.global > 0 && std::isfinite(w.global), "NVFP4 global scale underflow/overflow");
    w.packed.resize(x.size() / 2);
    w.scales.resize(x.size() / 16);
    for (size_t b = 0; b < w.scales.size(); ++b) {
        float m = 0;
        for (int j = 0; j < 16; ++j)
            m = std::max(m, std::abs(x[b * 16 + j]));
        uint8_t s = behemoth::encode_e4m3(m / (6 * w.global));
        w.scales[b] = s;
        float scale = behemoth::decode_e4m3(s) * w.global;
        for (int j = 0; j < 16; j += 2)
            w.packed[b * 8 + j / 2] =
                fp4(scale ? x[b * 16 + j] / scale : 0) | (fp4(scale ? x[b * 16 + j + 1] / scale : 0) << 4);
    }
    return w;
}
Vec dequantize_nvfp4(const Nvfp4 &w) {
    check(w.rows > 0 && w.cols > 0 && w.cols % 16 == 0 && w.packed.size() == size_t(w.rows) * w.cols / 2 &&
              w.scales.size() == size_t(w.rows) * w.cols / 16 && w.global > 0 && std::isfinite(w.global),
          "invalid NVFP4 payload");
    Vec out(size_t(w.rows) * w.cols);
    for (size_t i = 0; i < out.size(); ++i) {
        uint8_t code = (w.packed[i / 2] >> (i % 2 * 4)) & 15;
        float s = behemoth::decode_e4m3(w.scales[i / 16]);
        check(s >= 0 && std::isfinite(s), "invalid NVFP4 block scale");
        out[i] = float(behemoth::BFloat16((code & 8 ? -1.f : 1.f) * levels[code & 7] * s * w.global));
    }
    return out;
}
void convert_nvfp4(const fs::path &source, const fs::path &dest) {
    check(!read_json(source / "config.json").contains("glm53_quant"),
          "checkpoint is already quantized; use the original FP8/BF16 source");
    Weights weights(source);
    Config c(read_json(source / "config.json"));
    auto out = fs::absolute(dest).lexically_normal(), src = fs::canonical(source);
    check(!fs::exists(out), "quantized output must be a new directory");
    for (auto p = out; !p.empty() && p != p.parent_path(); p = p.parent_path())
        check(p != src, "output cannot be inside source");
    fs::create_directories(out.parent_path());
    auto tmp = fs::path(out.string() + ".tmp-" + std::to_string(getpid()));
    check(!fs::exists(tmp), "stale output directory");
    fs::create_directory(tmp);
    try {
        std::set<std::string> bases;
        for (int l = 0; l < c.layers; ++l) {
            std::string p = weights.prefix + "layers." + std::to_string(l) + ".mlp.";
            for (auto part : {"gate_proj", "up_proj", "down_proj"}) {
                if (c.mlp[l] == "dense")
                    bases.insert(p + part);
                else {
                    bases.insert(p + "shared_experts." + part);
                    for (int e = 0; e < c.experts; ++e)
                        bases.insert(p + "experts." + std::to_string(e) + "." + part);
                }
            }
        }
        json map = json::object();
        size_t number = 0;
        qtip::SafeTensorsWriter packed_writer;
        size_t packed_bytes = 0, completed = 0;
        auto flush = [&] {
            if (!packed_bytes)
                return;
            std::string shard = "mlp-" + std::to_string(number++) + ".safetensors";
            packed_writer.write((tmp / shard).string());
            for (const auto &n : packed_writer.names())
                map[n] = shard;
            packed_writer = {};
            packed_bytes = 0;
            std::cerr << "NVFP4: " << completed << '/' << bases.size() << " MLP projections\n";
        };
        for (const auto &base : bases) {
            auto t = weights.linear(base);
            int r = int(t.shape[0]), col = int(t.shape[1]);
            auto q = quantize_nvfp4(t.data, r, col);
            size_t bytes = q.packed.size() + q.scales.size() + 4;
            if (packed_bytes && packed_bytes + bytes > (256u << 20))
                flush();
            packed_writer.add(base + ".weight_packed", "U8", {r, col / 2}, q.packed.data(), q.packed.size());
            packed_writer.add(base + ".weight_scale", "F8_E4M3", {r, col / 16}, q.scales.data(),
                              q.scales.size());
            packed_writer.add(base + ".weight_global_scale", "F32", {}, &q.global, 4);
            packed_bytes += bytes;
            ++completed;
        }
        flush();
        std::map<std::string, std::vector<std::string>> shards;
        for (const auto &it : weights.index().at("weight_map").items())
            shards[it.value()].push_back(it.key());
        for (const auto &[shard, names] : shards) {
            qtip::SafeTensorsWriter writer;
            for (const auto &name : names) {
                bool replaced = false;
                for (const auto &suffix : {".weight", ".weight_scale", ".weight_scale_inv"})
                    if (name.ends_with(suffix) &&
                        bases.contains(name.substr(0, name.size() - std::string(suffix).size())))
                        replaced = true;
                if (replaced)
                    continue;
                const auto &f = weights.file(name);
                auto shape = f.shape(name);
                auto raw = f.bytes(name);
                check(!fs::exists(tmp / shard), "quantized shard name collision");
                writer.add(name, f.dtype(name), {shape.begin(), shape.end()}, raw.data(), raw.size());
                map[name] = shard;
            }
            if (!writer.names().empty())
                writer.write((tmp / shard).string());
        }
        for (const auto &e : fs::directory_iterator(src))
            if (e.is_regular_file() && e.path().extension() != ".safetensors" &&
                e.path().filename() != "config.json" && e.path().filename() != "model.safetensors.index.json")
                fs::copy_file(e.path(), tmp / e.path().filename());
        auto config = read_json(src / "config.json");
        config["glm53_quant"] = {
            {"mlp_format", "nvfp4"}, {"block_size", 16}, {"format_version", 1}, {"activations", "bf16"}};
        {
            std::ofstream f(tmp / "config.json");
            f << config.dump(2);
            check(bool(f), "write config failed");
        }
        uint64_t bytes = 0;
        for (const auto &e : fs::directory_iterator(tmp))
            if (e.path().extension() == ".safetensors")
                bytes += e.file_size();
        {
            std::ofstream f(tmp / "model.safetensors.index.json");
            f << json({{"metadata", {{"total_size", bytes}}}, {"weight_map", map}}).dump(2);
            check(bool(f), "write index failed");
        }
        fs::rename(tmp, out);
    } catch (...) {
        fs::remove_all(tmp);
        throw;
    }
}
} // namespace glm53
