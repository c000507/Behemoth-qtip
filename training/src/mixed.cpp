#include "train.hpp"
#include "safetensors.hpp"
#include <cstring>
#include <set>
#include <unistd.h>

namespace qat {
void replace_kv_fp8(const fs::path &root, const fs::path &source, const fs::path &destination) {
    Model student(root);
    auto teacher_root = fs::canonical(source);
    Config teacher(read_json(teacher_root / "config.json"));
    const auto &c = student.config;
    require(c.n == teacher.n && c.inter == teacher.inter && c.layers == teacher.layers &&
                c.heads == teacher.heads && c.kv == teacher.kv && c.dim == teacher.dim &&
                c.vocab == teacher.vocab && c.theta == teacher.theta && c.eps == teacher.eps,
            "source/student architecture mismatch");
    auto index = read_json(teacher_root / "model.safetensors.index.json");
    std::map<std::string, std::shared_ptr<behemoth::SafeTensorFile>> sources;
    auto file = [&](const std::string &name) -> const behemoth::SafeTensorFile & {
        auto shard = index.at("weight_map").at(name).get<std::string>();
        require(fs::path(shard).filename() == shard, "unsafe source shard path");
        if (!sources.contains(shard)) sources[shard] = std::make_shared<behemoth::SafeTensorFile>(teacher_root / shard);
        return *sources.at(shard);
    };
    for (auto name : {"model.embed_tokens.weight", "lm_head.weight"}) {
        auto a = student.file(name).floats(name), b = file(name).floats(name);
        require(a.size() == b.size(), "source embedding/head shape mismatch");
        for (size_t i = 0; i < a.size(); ++i)
            require(bf(a[i]) == bf(b[i]), "source embedding/head differs from quantized model");
    }
    for (auto name : {"tokenizer.model", "tokenizer_config.json"})
        if (fs::exists(student.path / name) && fs::exists(teacher_root / name))
            require(sha256(student.path / name) == sha256(teacher_root / name), "source tokenizer mismatch");
    auto out = fs::absolute(destination).lexically_normal();
    require(!fs::exists(out), "output directory exists");
    for (auto parent = out; !parent.empty() && parent != parent.parent_path(); parent = parent.parent_path())
        require(parent != student.path && parent != teacher_root, "output cannot be inside either source model");
    fs::create_directories(out.parent_path());
    auto tmp = fs::path(out.string() + ".tmp-" + std::to_string(getpid()));
    require(!fs::exists(tmp), "stale conversion output");
    fs::create_directory(tmp);
    try {
        std::set<std::string> prefixes, shards;
        qtip::SafeTensorsWriter kv;
        json map = json::object();
        const std::string kv_shard = "model-fp8-kv.safetensors";
        for (int l = 0; l < c.layers; ++l)
            for (int i : {1, 2}) {
                const auto prefix = "model.layers." + std::to_string(l) + "." + linears[i] + ".";
                prefixes.insert(prefix);
                const auto &f = file(prefix + "weight");
                auto w = behemoth::qtip::load_fp8_linear(f, file(prefix + "weight_scale"), l, linears[i]);
                require(w.in == c.n && w.out == c.kv * c.dim, "source FP8 K/V shape mismatch");
                for (auto suffix : {"weight", "weight_scale"}) {
                    auto name = prefix + suffix;
                    const auto &tensor = file(name);
                    auto shape = tensor.shape(name);
                    auto raw = tensor.bytes(name);
                    kv.add(name, tensor.dtype(name), {shape.begin(), shape.end()}, raw.data(), raw.size());
                    map[name] = kv_shard;
                }
            }
        for (const auto &item : student.index.at("weight_map").items()) shards.insert(item.value());
        for (const auto &shard : shards) {
            qtip::SafeTensorsWriter writer;
            for (const auto &item : student.index.at("weight_map").items()) {
                if (item.value() != shard) continue;
                bool replaced = false;
                for (const auto &prefix : prefixes) replaced |= item.key().starts_with(prefix);
                if (replaced) continue;
                const auto &f = student.file(item.key());
                auto shape = f.shape(item.key());
                auto raw = f.bytes(item.key());
                writer.add(item.key(), f.dtype(item.key()), {shape.begin(), shape.end()}, raw.data(), raw.size());
                map[item.key()] = shard;
            }
            if (!writer.names().empty()) {
                require(shard != kv_shard, "FP8 shard name collides with non-K/V tensors");
                writer.write((tmp / shard).string());
                sync_file(tmp / shard);
            }
        }
        kv.write((tmp / kv_shard).string());
        sync_file(tmp / kv_shard);
        for (const auto &entry : fs::directory_iterator(student.path))
            if (entry.is_regular_file() && entry.path().extension() != ".safetensors" &&
                entry.path().filename() != "config.json" && entry.path().filename() != "model.safetensors.index.json") {
                fs::copy_file(entry.path(), tmp / entry.path().filename());
                sync_file(tmp / entry.path().filename());
            }
        if (!fs::exists(tmp / "tokenizer.model") && fs::exists(teacher_root / "tokenizer.model")) {
            fs::copy_file(teacher_root / "tokenizer.model", tmp / "tokenizer.model");
            sync_file(tmp / "tokenizer.model");
        }
        auto config = student.config_json;
        config["qtip_quant"]["kv_format"] = "fp8";
        config["qtip_quant"]["fp8_kv_source"] = model_identity(teacher_root);
        uint64_t bytes = 0;
        for (const auto &entry : fs::directory_iterator(tmp))
            if (entry.path().extension() == ".safetensors") bytes += entry.file_size();
        durable_json(tmp / "config.json", config);
        durable_json(tmp / "model.safetensors.index.json", {{"metadata", {{"total_size", bytes}}}, {"weight_map", map}});
        sync_directory(tmp);
        fs::rename(tmp, out);
        sync_directory(out.parent_path());
    } catch (...) {
        fs::remove_all(tmp);
        throw;
    }
}
} // namespace qat
