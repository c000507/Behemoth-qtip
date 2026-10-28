#include "common.hpp"
#include "safetensors.hpp"
#include "train.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <openssl/evp.h>
#include <random>
#include <set>
#include <sstream>
#include <unistd.h>

namespace qat {
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
json read_json(const fs::path &p) {
    std::ifstream f(p);
    require(bool(f), "cannot read " + p.string());
    return json::parse(f);
}
std::string sha256(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    require(bool(f), "cannot hash " + p.string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA256 init failed");
    std::vector<char> block(8 << 20);
    while (f) {
        f.read(block.data(), block.size());
        require(EVP_DigestUpdate(ctx.get(), block.data(), size_t(f.gcount())) == 1, "SHA256 update");
    }
    require(f.eof(), "hash read failed");
    unsigned char bytes[32];
    unsigned size = 0;
    require(EVP_DigestFinal_ex(ctx.get(), bytes, &size) == 1 && size == 32, "SHA256 finish");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto c : bytes)
        out << std::setw(2) << unsigned(c);
    return out.str();
}
std::string model_identity(const fs::path &p) {
    auto index = read_json(p / "model.safetensors.index.json");
    std::set<std::string> files;
    for (auto &v : index.at("weight_map").items()) {
        std::string name = v.value();
        require(fs::path(name).filename() == name, "unsafe shard path");
        files.insert(name);
    }
    std::string hashes = sha256(p / "config.json") + sha256(p / "model.safetensors.index.json");
    for (const auto &name : files)
        hashes += name + ":" + sha256(p / name) + ";";
    unsigned char digest[32];
    unsigned n = 0;
    require(EVP_Digest(hashes.data(), hashes.size(), digest, &n, EVP_sha256(), nullptr) == 1 && n == 32,
            "model digest");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto c : digest)
        out << std::setw(2) << unsigned(c);
    return out.str();
}
void sync_directory(const fs::path &p) {
    int fd = ::open(p.c_str(), O_RDONLY | O_DIRECTORY);
    require(fd >= 0, "open directory " + p.string());
    int rc = ::fsync(fd);
    ::close(fd);
    require(rc == 0, "directory fsync");
}
void sync_file(const fs::path &p) {
    int fd = ::open(p.c_str(), O_RDONLY);
    require(fd >= 0, "open " + p.string());
    int rc = ::fsync(fd);
    ::close(fd);
    require(rc == 0, "file fsync " + p.string());
}
void durable_json(const fs::path &p, const json &j) {
    auto tmp = p.string() + ".tmp";
    {
        std::ofstream f(tmp);
        require(bool(f), "write " + tmp);
        f << j.dump(2) << '\n';
        f.flush();
        require(bool(f), "JSON write failed");
    }
    int fd = ::open(tmp.c_str(), O_RDONLY);
    require(fd >= 0, "open temp JSON");
    int rc = ::fsync(fd);
    ::close(fd);
    require(rc == 0, "JSON fsync");
    fs::rename(tmp, p);
    sync_directory(p.parent_path());
}
float bf(float x) { return float(behemoth::BFloat16(x)); }
Config::Config(const json &c)
    : n(c.at("hidden_size")), inter(c.at("intermediate_size")), heads(c.at("num_attention_heads")),
      kv(c.at("num_key_value_heads")), dim(c.value("head_dim", n / std::max(heads, 1))),
      vocab(c.at("vocab_size")), layers(c.at("num_hidden_layers")), eps(c.value("rms_norm_eps", 1e-5f)),
      theta(c.value("rope_theta", 1e6f)) {
    require(n > 0 && inter > 0 && heads > 0 && kv > 0 && heads % kv == 0 && dim > 0 && dim % 2 == 0 &&
                vocab > 1 && layers > 0,
            "invalid model geometry");
    require(eps > 0 && std::isfinite(eps) && theta > 0 && std::isfinite(theta), "invalid norm/RoPE config");
    require(c.value("hidden_act", std::string("silu")) == "silu", "only SwiGLU supported");
    for (auto key : {"rope_scaling", "sliding_window", "attention_bias", "mlp_bias"})
        require(!c.contains(key) || c[key].is_null() || c[key] == false || c[key] == 0,
                std::string("unsupported ") + key);
    const auto &q = c.at("quip_params");
    require(q.at("L") == 16 && q.at("K") == 2 && q.at("V") == 2 && q.at("tlut_bits") == 9 &&
                q.at("decode_mode") == "quantlut_sym" && q.at("td_x") == 16 && q.at("td_y") == 16,
            "expected 2-bit HYB QTIP");
}
std::string su_name(int i) { return linears.at(i == 1 || i == 2 ? 0 : i == 5 ? 4 : i) + ".SU"; }
const behemoth::SafeTensorFile &Model::file(const std::string &name) const {
    auto key = index.at("weight_map").at(name).get<std::string>();
    return *files_.at(key);
}
size_t Model::add_param(const std::string &name, std::span<const float> value, bool norm) {
    size_t id = params.size();
    require(std::all_of(value.begin(), value.end(), [](float x) { return std::isfinite(x); }),
            "nonfinite " + name);
    // The native row transforms load sign/norm vectors in aligned groups of four.
    initial.resize((initial.size() + 3) / 4 * 4, 0.f);
    params.push_back({name, initial.size(), value.size(), norm, group_of(name, norm)});
    initial.insert(initial.end(), value.begin(), value.end());
    return id;
}
Model::Model(const fs::path &root)
    : path(fs::canonical(root)), config_json(read_json(path / "config.json")),
      index(read_json(path / "model.safetensors.index.json")), config(config_json) {
    for (auto &v : index.at("weight_map").items()) {
        std::string key = v.value();
        require(fs::path(key).filename() == key, "unsafe shard path");
        if (!files_.contains(key))
            files_[key] = std::make_shared<behemoth::SafeTensorFile>(path / key);
        require(!v.key().ends_with(".bias"), "bias tensors unsupported");
    }
    auto floats = [&](const std::string &name) {
        auto v = file(name).floats(name);
        std::vector<float> out(v.size());
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = v[i];
        return out;
    };
    int qd = config.heads * config.dim, kd = config.kv * config.dim;
    const std::vector<std::pair<int, int>> shapes = {
        {qd, config.n},           {kd, config.n},           {kd, config.n},          {config.n, qd},
        {config.inter, config.n}, {config.inter, config.n}, {config.n, config.inter}};
    for (int l = 0; l < config.layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".";
        LayerData layer;
        for (int i = 0; i < 7; ++i) {
            auto [m, n] = shapes[i];
            require(m % 32 == 0 && n % 32 == 0, "QTIP dimensions need multiples of 32");
            auto base = prefix + linears[i] + ".";
            behemoth::qtip::HostLinear w;
            w.in = n;
            w.out = m;
            auto &f = file(base + "trellis");
            require(f.dtype(base + "trellis") == "I16" &&
                        f.shape(base + "trellis") == std::vector<size_t>{size_t(m / 16) * size_t(n / 16), 32},
                    "invalid trellis");
            auto raw = f.bytes(base + "trellis");
            w.trellis.resize(raw.size() / 2);
            std::memcpy(w.trellis.data(), raw.data(), raw.size());
            auto &fl = file(base + "tlut");
            require(fl.dtype(base + "tlut") == "F16" &&
                        fl.shape(base + "tlut") == std::vector<size_t>{512, 2},
                    "invalid tlut");
            raw = fl.bytes(base + "tlut");
            w.tlut.resize(1024);
            std::memcpy(w.tlut.data(), raw.data(), raw.size());
            (void)behemoth::qtip::make_codebook(w.tlut);
            require(file(base + "SU").dtype(base + "SU") == "BF16" &&
                        file(base + "SV").dtype(base + "SV") == "F32",
                    "expected BF16 SU / FP32 SV");
            w.su = floats(base + "SU");
            w.sv = floats(base + "SV");
            require(w.su.size() == size_t(n) && w.sv.size() == size_t(m), "invalid sign vector size");
            for (auto [key, expected] : {std::pair{"rcp", int64_t(0)}, {"tp_rank", int64_t(8)}}) {
                auto &s = file(base + key);
                require(s.dtype(base + key) == "I64" && s.bytes(base + key).size() == 8, "invalid scalar");
                int64_t v;
                std::memcpy(&v, s.bytes(base + key).data(), 8);
                require(v == expected, "unsupported split/permuted checkpoint");
            }
            auto canonical = su_name(i);
            if (canonical == linears[i] + ".SU")
                layer.params[canonical] = add_param(prefix + canonical, w.su);
            else {
                const auto &p = params.at(layer.params.at(canonical));
                require(std::equal(w.su.begin(), w.su.end(), initial.begin() + p.offset),
                        "runtime requires tied SU");
            }
            double ss = 0;
            for (float v : w.sv)
                ss += double(v) * v;
            float scale = float(std::sqrt(ss / m));
            require(scale > 0 && std::isfinite(scale), "invalid Wscale");
            auto signs = w.sv;
            for (float &v : signs)
                v /= scale;
            layer.params[linears[i] + ".SV"] = add_param(base + "SV", signs);
            layer.params[linears[i] + ".Wscale"] = add_param(base + "Wscale", std::span(&scale, 1));
            if (!layer.linear.empty())
                require(w.tlut == layer.linear[0].tlut, "runtime requires one LUT per layer");
            layer.linear.push_back(std::move(w));
        }
        for (const auto &key : {norm1, norm2}) {
            auto v = floats(prefix + key);
            require(v.size() == size_t(config.n), "invalid norm");
            layer.params[key] = add_param(prefix + key, v, true);
        }
        layers.push_back(std::move(layer));
    }
    auto final = floats("model.norm.weight");
    require(final.size() == size_t(config.n), "invalid final norm");
    final_norm = add_param("model.norm.weight", final, true);
    auto frozen = [&](const std::string &key) {
        auto v = floats(key);
        require(v.size() == size_t(config.vocab) * config.n, "invalid embedding/head");
        std::vector<uint16_t> out(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            out[i] = behemoth::BFloat16(v[i]).bits;
        return out;
    };
    embedding = frozen("model.embed_tokens.weight");
    head = frozen("lm_head.weight");
    identity = model_identity(path);
}
void Model::export_model(const fs::path &dest, std::span<const float> values) const {
    require(values.size() == initial.size(), "export parameter size mismatch");
    auto out = fs::absolute(dest).lexically_normal();
    require(!fs::exists(out), "export destination exists");
    for (auto p = out; !p.empty() && p != p.parent_path(); p = p.parent_path())
        require(p != path, "export inside source");
    for (auto p = path; !p.empty() && p != p.parent_path(); p = p.parent_path())
        require(p != out, "export contains source");
    fs::create_directories(out.parent_path());
    auto tmp = out.string() + ".tmp-" + std::to_string(getpid());
    require(!fs::exists(tmp), "stale export temp directory");
    fs::create_directory(tmp);
    try {
        std::map<std::string, std::vector<float>> replacement;
        for (int l = 0; l < config.layers; ++l) {
            auto prefix = "model.layers." + std::to_string(l) + ".";
            for (int i = 0; i < 7; ++i) {
                const auto &a = params[layers[l].params.at(su_name(i))];
                replacement[prefix + linears[i] + ".SU"] = {values.begin() + a.offset,
                                                            values.begin() + a.offset + a.size};
                const auto &v = params[layers[l].params.at(linears[i] + ".SV")],
                           &s = params[layers[l].params.at(linears[i] + ".Wscale")];
                auto &r = replacement[prefix + linears[i] + ".SV"];
                r.resize(v.size);
                for (size_t j = 0; j < v.size; ++j)
                    r[j] = values[v.offset + j] * values[s.offset];
            }
            for (auto &key : {norm1, norm2}) {
                const auto &p = params[layers[l].params.at(key)];
                replacement[prefix + key] = {values.begin() + p.offset, values.begin() + p.offset + p.size};
            }
        }
        const auto &pn = params[final_norm];
        replacement[pn.name] = {values.begin() + pn.offset, values.begin() + pn.offset + pn.size};
        for (auto &[shard, f] : files_) {
            qtip::SafeTensorsWriter writer;
            for (auto &it : index.at("weight_map").items())
                if (it.value() == shard) {
                    auto name = it.key(), dtype = f->dtype(name);
                    auto shape = f->shape(name);
                    std::vector<int64_t> dims(shape.begin(), shape.end());
                    auto raw = f->bytes(name);
                    if (replacement.contains(name)) {
                        auto &v = replacement[name];
                        require(std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }),
                                "nonfinite export");
                        if (dtype == "BF16") {
                            std::vector<uint16_t> b(v.size());
                            for (size_t j = 0; j < v.size(); ++j) {
                                b[j] = behemoth::BFloat16(v[j]).bits;
                                require(std::isfinite(float(behemoth::BFloat16{v[j]})),
                                        "BF16 export overflow");
                            }
                            writer.add(name, dtype, dims, b.data(), b.size() * 2);
                        } else {
                            require(dtype == "F32", "unsupported trainable export dtype");
                            writer.add(name, dtype, dims, v.data(), v.size() * 4);
                        }
                    } else
                        writer.add(name, dtype, dims, raw.data(), raw.size());
                }
            writer.write((fs::path(tmp) / shard).string());
            int fd = ::open((fs::path(tmp) / shard).c_str(), O_RDONLY);
            require(fd >= 0 && ::fsync(fd) == 0, "export fsync");
            ::close(fd);
        }
        for (auto &entry : fs::directory_iterator(path))
            if (entry.is_regular_file() && entry.path().extension() != ".safetensors") {
                auto d = fs::path(tmp) / entry.path().filename();
                fs::copy_file(entry.path(), d);
                int fd = ::open(d.c_str(), O_RDONLY);
                require(fd >= 0 && ::fsync(fd) == 0, "export sidecar fsync");
                ::close(fd);
            }
        sync_directory(tmp);
        fs::rename(tmp, out);
        sync_directory(out.parent_path());
    } catch (...) {
        fs::remove_all(tmp);
        throw;
    }
}
size_t Batch::targets() const {
    return size_t(std::count_if(labels.begin(), labels.end(), [](int v) { return v != -100; }));
}
void Batch::boundaries() {
    require(tokens.size() == segments.size() && tokens.size() == positions.size(), "invalid batch geometry");
    starts.resize(tokens.size());
    ends.resize(tokens.size());
    for (size_t a = 0; a < tokens.size();) {
        size_t b = a + 1;
        while (b < tokens.size() && segments[b] == segments[a])
            ++b;
        for (size_t i = a; i < b; ++i) {
            starts[i] = int(a);
            ends[i] = int(b);
        }
        a = b;
    }
}
Dataset::Dataset(const fs::path &p, bool train) : root_(p), metadata(read_json(p / "metadata.json")) {
    require(std::endian::native == std::endian::little, "packed datasets require a little-endian host");
    require(metadata.at("format_version") == 1 && (!train || metadata.at("split") == "train"),
            "expected v1 training dataset");
    count = metadata.at("nseq");
    length = metadata.at("seqlen");
    require(count > 0 && length >= 2 && length <= INT32_MAX && count <= SIZE_MAX / length / 4,
            "invalid dataset shape");
    for (auto [name, width] :
         {std::pair{"tokens.i32", 4}, {"loss.u8", 1}, {"segments.i32", 4}, {"positions.i32", 4}}) {
        require(fs::file_size(p / name) == count * length * size_t(width), "dataset size mismatch");
        require(sha256(p / name) == metadata.at("files").at(name).get<std::string>(),
                std::string("dataset checksum mismatch: ") + name);
    }
    std::set<int> held;
    for (int i : metadata.value("holdout_conversations", std::vector<int>{}))
        held.insert(i);
    for (int i : metadata.value("training_conversations", std::vector<int>{}))
        require(!held.contains(i), "held-out conversation leaked into training");
    identity = sha256(p / "metadata.json");
}
Batch Dataset::get(size_t i) const {
    require(i < count, "dataset position outside epoch");
    Batch b;
    b.tokens.resize(length);
    b.segments.resize(length);
    b.positions.resize(length);
    b.labels.assign(length, -100);
    std::vector<uint8_t> mask(length);
    auto read = [&](const char *name, void *dst, size_t width) {
        std::ifstream f(root_ / name, std::ios::binary);
        f.seekg(std::streamoff(i * length * width));
        f.read(static_cast<char *>(dst), std::streamsize(length * width));
        require(bool(f), "short dataset read");
    };
    read("tokens.i32", b.tokens.data(), 4);
    read("segments.i32", b.segments.data(), 4);
    read("positions.i32", b.positions.data(), 4);
    read("loss.u8", mask.data(), 1);
    for (size_t t = 0; t < length; ++t)
        require(b.positions[t] >= 0 && mask[t] <= 1 && b.segments[t] >= -1,
                "invalid positions/loss/segments");
    for (size_t t = 0; t + 1 < length; ++t)
        if (mask[t + 1] && b.segments[t] >= 0 && b.segments[t] == b.segments[t + 1])
            b.labels[t] = b.tokens[t + 1];
    b.boundaries();
    return b;
}
Mat Mat::row(int a, int b) const {
    require(a >= 0 && b >= a && b <= rows, "bad row slice");
    return {owner, reinterpret_cast<float *>(reinterpret_cast<char *>(p) + size_t(a) * ld * element()), b - a, cols,
            ld, half};
}
Mat Mat::col(int a, int b) const {
    require(a >= 0 && b >= a && b <= cols, "bad column slice");
    return {owner, reinterpret_cast<float *>(reinterpret_cast<char *>(p) + size_t(a) * element()), rows, b - a, ld,
            half};
}
} // namespace qat
