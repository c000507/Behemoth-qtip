#pragma once
#include "numeric.hpp"
#include <filesystem>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>

namespace behemoth {
struct FloatTensorView {
    std::span<const std::byte> data;
    enum class Format { f16_file, bf16, f32 } format;
    size_t size() const {
        return data.size() / (format == Format::f32 ? 4 : 2);
    }
    float operator[](size_t i) const;
};
struct ModelConfig {
    int layers = 88, hidden = 12288, intermediate = 28672, q_heads = 96, kv_heads = 8, dim = 128,
        vocab = 32768;
    int max_context = 131072;
    float epsilon = 1e-5f, theta = 1000000;
    static ModelConfig read(const std::filesystem::path &);
};
class SafeTensorFile {
    int fd_ = -1;
    size_t bytes_ = 0, data_start_ = 0;
    const std::byte *mapping_ = nullptr;
    nlohmann::json header_;

  public:
    explicit SafeTensorFile(const std::filesystem::path &);
    ~SafeTensorFile();
    SafeTensorFile(const SafeTensorFile &) = delete;
    SafeTensorFile &operator=(const SafeTensorFile &) = delete;
    std::vector<size_t> shape(const std::string &) const;
    std::string dtype(const std::string &) const;
    std::span<const std::byte> bytes(const std::string &) const;
    float value(const std::string &, size_t index) const;
    uint16_t code(const std::string &, size_t index) const;
    FloatTensorView floats(const std::string &) const;
};
struct HostProjection {
    int in = 0, out = 0;
    std::vector<uint16_t> codes;
    std::vector<uint8_t> codebook;
    std::vector<float> row_scale;
    float codebook_scale = 1;
};
enum class CodeLayout { RowMajor, GroupMajor, Sorted };
CodeLayout parse_code_layout(std::string_view);
const char *code_layout_name(CodeLayout);
std::vector<uint16_t> group_major_codes(const HostProjection &);
class Checkpoint {
    std::filesystem::path root_;
    std::map<std::string, std::string> index_;
    mutable std::map<std::string, std::shared_ptr<SafeTensorFile>> files_;

  public:
    ModelConfig config;
    explicit Checkpoint(std::filesystem::path);
    const SafeTensorFile &tensor(const std::string &) const;
    std::vector<BFloat16> bf16(const std::string &) const;
    HostProjection projection(int layer, const std::string &name, int rank,
                              bool column_parallel) const;
};
std::vector<float> reference_projection(const HostProjection &, std::span<const float> input);
std::vector<float> reference_projection_unscaled(const HostProjection &,
                                                 std::span<const float> input);
std::vector<BFloat16> reference_block(const Checkpoint &, int layer,
                                      std::span<const BFloat16> input);
} // namespace behemoth
