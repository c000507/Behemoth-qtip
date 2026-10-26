#include "behemoth/checkpoint.hpp"
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace behemoth {
const char *code_layout_name(CodeLayout layout) {
    switch (layout) {
    case CodeLayout::RowMajor:
        return "row-major";
    case CodeLayout::GroupMajor:
        return "group-major";
    case CodeLayout::Sorted:
        return "sorted";
    case CodeLayout::Lane:
        return "lane";
    }
    throw std::invalid_argument("Invalid code layout");
}
CodeLayout parse_code_layout(std::string_view name) {
    if (name == "row-major")
        return CodeLayout::RowMajor;
    if (name == "group-major")
        return CodeLayout::GroupMajor;
    if (name == "sorted")
        return CodeLayout::Sorted;
    if (name == "lane")
        return CodeLayout::Lane;
    throw std::invalid_argument("Code layout must be row-major, group-major, sorted or lane");
}
std::vector<uint16_t> group_major_codes(const HostProjection &p) {
    if (p.in <= 0 || p.in % 8 || p.out <= 0 ||
        p.codes.size() != checked_mul(size_t(p.out), size_t(p.in / 8)))
        throw std::invalid_argument("Invalid code matrix for transposition");
    const size_t groups = size_t(p.in / 8), outputs = size_t(p.out);
    std::vector<uint16_t> result(p.codes.size());
    // A blocked transpose keeps the host initialization pass cache friendly.
    for (size_t o0 = 0; o0 < outputs; o0 += 32)
        for (size_t g0 = 0; g0 < groups; g0 += 32)
            for (size_t o = o0; o < std::min(outputs, o0 + 32); ++o)
                for (size_t g = g0; g < std::min(groups, g0 + 32); ++g)
                    result[g * outputs + o] = p.codes[o * groups + g];
    return result;
}
float FloatTensorView::operator[](size_t i) const {
    if (i >= size())
        throw std::out_of_range("Tensor element");
    if (format == Format::f32) {
        float v;
        std::memcpy(&v, data.data() + i * 4, 4);
        return v;
    }
    uint16_t u;
    std::memcpy(&u, data.data() + i * 2, 2);
    return format == Format::f16_file ? decode_checkpoint_f16(u)
                                      : std::bit_cast<float>(uint32_t(u) << 16);
}
static nlohmann::json read_json(const std::filesystem::path &p) {
    std::ifstream f(p);
    if (!f)
        throw std::runtime_error("Cannot read " + p.string());
    return nlohmann::json::parse(f);
}
ModelConfig ModelConfig::read(const std::filesystem::path &path) {
    auto j = read_json(path);
    ModelConfig c;
    // QTIP checkpoints (qtip_quant --mode assemble) are QTIP-hfized llama configs of the same
    // Mistral-Large architecture: model_type llama + quip_params, no quantization_config.
    const bool qtip = j.contains("quip_params");
    if (j.at("hidden_size") != c.hidden || j.at("intermediate_size") != c.intermediate ||
        j.at("num_hidden_layers") != c.layers || j.at("num_attention_heads") != c.q_heads ||
        j.at("num_key_value_heads") != c.kv_heads || j.at("head_dim") != c.dim ||
        j.at("vocab_size") != c.vocab || j.at("model_type") != (qtip ? "llama" : "mistral") ||
        j.at("hidden_act") != "silu" || j.at("tie_word_embeddings") != false ||
        (j.contains("sliding_window") && !j.at("sliding_window").is_null()) ||
        (!qtip && !j.contains("sliding_window")))
        throw std::invalid_argument("Checkpoint is not the supported Behemoth architecture");
    if (qtip) {
        auto q = j.at("quip_params");
        if (q.at("codebook") != "bitshift" || q.at("L") != 16 || q.at("K") != 2 || q.at("V") != 2 ||
            q.at("tlut_bits") != 9 || q.at("decode_mode") != "quantlut_sym" || q.at("td_x") != 16 ||
            q.at("td_y") != 16 || (q.contains("split_for_tp") && q.at("split_for_tp") != false) ||
            (q.contains("skip_list") && !q.at("skip_list").is_null()) ||
            j.contains("quantization_config"))
            throw std::invalid_argument("Unsupported QTIP format (needs HYB L16 K2 V2 tlut_bits 9 "
                                        "quantlut_sym 16x16, no TP split, no skip list)");
        c.quantization = Quantization::Qtip;
    } else {
        auto q = j.at("quantization_config");
        if (q.at("quant_method") != "aqlm" || q.at("in_group_size") != 8 ||
            q.at("out_group_size") != 1 || q.at("num_codebooks") != 1 ||
            q.at("nbits_per_codebook") != 16)
            throw std::invalid_argument("Unsupported AQLM format");
    }
    c.max_context = j.at("max_position_embeddings");
    c.epsilon = j.at("rms_norm_eps");
    c.theta = j.at("rope_theta");
    if (c.max_context != 131072 || c.epsilon != 1e-5f || c.theta != 1000000.f)
        throw std::invalid_argument("Unsupported position/normalization configuration");
    return c;
}
SafeTensorFile::SafeTensorFile(const std::filesystem::path &path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("Cannot open " + path.string());
    try {
        struct stat s{};
        if (fstat(fd_, &s) || s.st_size < 10)
            throw std::runtime_error("Invalid safetensors file");
        bytes_ = size_t(s.st_size);
        void *p = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED)
            throw std::runtime_error("mmap failed");
        mapping_ = static_cast<const std::byte *>(p);
        uint64_t n = 0;
        std::memcpy(&n, mapping_, 8);
        if (std::endian::native != std::endian::little || n > 100000000 || n > bytes_ - 8)
            throw std::runtime_error("Invalid safetensors header");
        data_start_ = 8 + size_t(n);
        header_ = nlohmann::json::parse(reinterpret_cast<const char *>(mapping_ + 8),
                                        reinterpret_cast<const char *>(mapping_ + data_start_));
        std::vector<std::pair<size_t, size_t>> intervals;
        for (auto it = header_.begin(); it != header_.end(); ++it)
            if (it.key() != "__metadata__") {
                auto raw = bytes(it.key());
                (void)raw;
                auto o = it.value().at("data_offsets").get<std::vector<size_t>>();
                intervals.emplace_back(o[0], o[1]);
            }
        std::sort(intervals.begin(), intervals.end());
        size_t previous = 0;
        for (auto [begin, end] : intervals) {
            if (begin != previous)
                throw std::runtime_error("Safetensors gaps/overlap");
            previous = end;
        }
        if (previous != bytes_ - data_start_)
            throw std::runtime_error("Safetensors trailing/truncated data");
    } catch (...) {
        if (mapping_)
            munmap(const_cast<std::byte *>(mapping_), bytes_);
        ::close(fd_);
        throw;
    }
}
SafeTensorFile::~SafeTensorFile() {
    if (mapping_)
        munmap(const_cast<std::byte *>(mapping_), bytes_);
    if (fd_ >= 0)
        ::close(fd_);
}
std::vector<size_t> SafeTensorFile::shape(const std::string &name) const {
    return header_.at(name).at("shape").get<std::vector<size_t>>();
}
std::string SafeTensorFile::dtype(const std::string &name) const {
    return header_.at(name).at("dtype").get<std::string>();
}
std::span<const std::byte> SafeTensorFile::bytes(const std::string &name) const {
    auto offsets = header_.at(name).at("data_offsets").get<std::vector<size_t>>();
    if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > bytes_ - data_start_)
        throw std::runtime_error("Tensor offsets out of bounds");
    std::string d = dtype(name);
    size_t width = d == "F32" ? 4
                   : (d == "F16" || d == "BF16" || d == "I16") ? 2
                   : d == "I64" ? 8 // QTIP rcp / tp_rank scalars
                                : 0;
    if (!width)
        throw std::runtime_error("Unsupported checkpoint dtype: " + d);
    size_t count = 1;
    for (size_t n : shape(name))
        count = checked_mul(count, n);
    if (checked_mul(count, width) != offsets[1] - offsets[0])
        throw std::runtime_error("Tensor byte/shape mismatch");
    return {mapping_ + data_start_ + offsets[0], offsets[1] - offsets[0]};
}
float SafeTensorFile::value(const std::string &name, size_t i) const {
    return floats(name)[i];
}
FloatTensorView SafeTensorFile::floats(const std::string &name) const {
    auto d = dtype(name);
    auto f = d == "F16"    ? FloatTensorView::Format::f16_file
             : d == "BF16" ? FloatTensorView::Format::bf16
                           : FloatTensorView::Format::f32;
    if (d != "F16" && d != "BF16" && d != "F32")
        throw std::invalid_argument("Expected floating tensor");
    return {bytes(name), f};
}
uint16_t SafeTensorFile::code(const std::string &name, size_t i) const {
    if (dtype(name) != "I16")
        throw std::invalid_argument("Expected I16 codes");
    auto d = bytes(name);
    if (i >= d.size() / 2)
        throw std::out_of_range("Code element");
    uint16_t u;
    std::memcpy(&u, d.data() + i * 2, 2);
    return u;
}
Checkpoint::Checkpoint(std::filesystem::path root)
    : root_(std::move(root)), config(ModelConfig::read(root_ / "config.json")) {
    index_ = read_json(root_ / "model.safetensors.index.json")
                 .at("weight_map")
                 .get<std::map<std::string, std::string>>();
    for (const auto &[key, file] : index_) {
        (void)key;
        if (std::filesystem::path(file).filename() != file)
            throw std::invalid_argument("Unsafe shard path");
    }
}
const SafeTensorFile &Checkpoint::tensor(const std::string &name) const {
    auto file = index_.at(name);
    auto it = files_.find(file);
    if (it == files_.end())
        it = files_.emplace(file, std::make_shared<SafeTensorFile>(root_ / file)).first;
    return *it->second;
}
std::vector<BFloat16> Checkpoint::bf16(const std::string &name) const {
    auto view = tensor(name).floats(name);
    std::vector<BFloat16> out(view.size());
    for (size_t i = 0; i < view.size(); ++i) {
        float value = view[i];
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite checkpoint weight: " + name);
        out[i] = BFloat16(value);
    }
    return out;
}
HostProjection Checkpoint::projection(int layer, const std::string &name, int rank,
                                      bool column) const {
    if (layer < 0 || layer >= 88 || rank < 0 || rank > 1)
        throw std::invalid_argument("Invalid layer/rank");
    std::string base = "model.layers." + std::to_string(layer) + "." + name;
    const auto &ct = tensor(base + ".codes");
    auto shape = ct.shape(base + ".codes");
    if (shape.size() != 3 || shape[2] != 1 || shape[0] % 2 || shape[1] % 2)
        throw std::invalid_argument("Invalid AQLM code shape");
    int full_out = int(shape[0]), full_in = int(shape[1]) * 8;
    int expected_in = 12288, expected_out = 12288;
    if (name == "self_attn.k_proj" || name == "self_attn.v_proj")
        expected_out = 1024;
    else if (name == "mlp.gate_proj" || name == "mlp.up_proj")
        expected_out = 28672;
    else if (name == "mlp.down_proj")
        expected_in = 28672;
    else if (name != "self_attn.q_proj" && name != "self_attn.o_proj")
        throw std::invalid_argument("Unknown projection");
    if (full_in != expected_in || full_out != expected_out || ct.dtype(base + ".codes") != "I16")
        throw std::invalid_argument("Projection dimensions/dtype disagree with architecture");
    HostProjection p;
    p.in = column ? full_in : full_in / 2;
    p.out = column ? full_out / 2 : full_out;
    p.codes.resize(size_t(p.out) * size_t(p.in / 8));
    p.row_scale.resize(size_t(p.out));
    auto raw = ct.bytes(base + ".codes");
    auto scales = tensor(base + ".scales").floats(base + ".scales");
    if (scales.size() != size_t(full_out))
        throw std::invalid_argument("Invalid row scales");
    for (int o = 0; o < p.out; ++o) {
        int src_o = o + (column ? rank * p.out : 0), src_i = column ? 0 : rank * (p.in / 8);
        size_t src = size_t(src_o) * size_t(full_in / 8) + size_t(src_i);
        std::memcpy(p.codes.data() + size_t(o) * size_t(p.in / 8), raw.data() + src * 2,
                    size_t(p.in / 8) * 2);
        p.row_scale[size_t(o)] = scales[size_t(src_o)];
        if (!std::isfinite(p.row_scale[size_t(o)]))
            throw std::invalid_argument("Nonfinite row scale");
    }
    const auto &bt = tensor(base + ".codebooks");
    if (bt.shape(base + ".codebooks") != std::vector<size_t>{1, 65536, 1, 8})
        throw std::invalid_argument("Invalid codebook shape");
    auto bv = bt.floats(base + ".codebooks");
    std::vector<float> values(65536 * 8);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = bv[i];
    auto qb = quantize(values);
    p.codebook = std::move(qb.values);
    p.codebook_scale = qb.scale;
    return p;
}
std::vector<float> reference_projection_unscaled(const HostProjection &p,
                                                 std::span<const float> x) {
    if (p.in <= 0 || p.out <= 0 || p.in % 8 || x.size() % size_t(p.in) ||
        p.codes.size() != size_t(p.out) * size_t(p.in / 8) || p.codebook.size() != 65536 * 8 ||
        p.row_scale.size() != size_t(p.out))
        throw std::invalid_argument("Projection shape mismatch");
    std::vector<float> y(x.size() / size_t(p.in) * size_t(p.out));
    std::vector<float> book(p.codebook.size());
    for (size_t i = 0; i < book.size(); ++i)
        book[i] = decode_e4m3(p.codebook[i]);
    for (size_t t = 0; t < x.size() / size_t(p.in); ++t) {
        auto q = quantize(x.subspan(t * size_t(p.in), size_t(p.in)));
        std::vector<float> input(q.values.size());
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = decode_e4m3(q.values[i]);
        for (int o = 0; o < p.out; ++o) {
            double sum = 0;
            for (int i = 0; i < p.in; ++i) {
                auto code = p.codes[size_t(o) * size_t(p.in / 8) + size_t(i / 8)];
                sum += double(input[size_t(i)]) * book[size_t(code) * 8 + size_t(i % 8)];
            }
            y[t * size_t(p.out) + size_t(o)] = float(sum * q.scale * p.codebook_scale);
        }
    }
    return y;
}
std::vector<float> reference_projection(const HostProjection &p, std::span<const float> x) {
    auto y = reference_projection_unscaled(p, x);
    for (size_t i = 0; i < y.size(); ++i)
        y[i] *= p.row_scale[i % size_t(p.out)];
    return y;
}
} // namespace behemoth
