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
// Weight format of the checkpoint: AQLM 1x16 (the original runtime format) or QTIP 2-bit HYB
// (qtip_quant output, QTIP-hfized llama layout with config.quip_params; see qtip.hpp).
enum class Quantization { Aqlm, Qtip };
// llama3 RoPE frequency scaling (HF rope_scaling type "llama3", e.g. Llama 3.3 70B).
struct RopeScalingLlama3 {
    double factor = 0, low_freq_factor = 0, high_freq_factor = 0;
    long original_max_position_embeddings = 0;
};
// YaRN RoPE (HF rope_scaling type "yarn", e.g. Devstral 2 / ministral3): frequency ramp between interpolated and
// extrapolated dims; the attention factor scales cos and sin (so q.k by its square), as HF and vLLM do.
struct RopeScalingYarn {
    double factor = 0, beta_fast = 32, beta_slow = 1, mscale = 0, mscale_all_dim = 0, attention_factor = 0;
    long original_max_position_embeddings = 0;
    bool truncate = true;
};
struct ModelConfig {
    Quantization quantization = Quantization::Aqlm;
    bool fp8_kv = false; // Mixed QTIP checkpoint: original E4M3 K/V projections.
    int layers = 88, hidden = 12288, intermediate = 28672, q_heads = 96, kv_heads = 8, dim = 128,
        vocab = 32768;
    int max_context = 131072;
    float epsilon = 1e-5f, theta = 1000000;
    bool rope_llama3 = false, rope_yarn = false;
    RopeScalingLlama3 rope;
    RopeScalingYarn yarn;
    double rope_attention_factor = 1; // multiplies cos and sin (YaRN mscale); 1 otherwise
    std::vector<int> eos{2};
    int bos = 1;
    static ModelConfig read(const std::filesystem::path &);
};
// inv_freq[d] (d < dim/2) with llama3 wavelength scaling; fp64 host computation (HF
// _compute_llama3_parameters), returned as floats for the device tables.
std::vector<float> llama3_inv_freq(int dim, double theta, const RopeScalingLlama3 &);
// inv_freq[d] with YaRN scaling (HF _compute_yarn_parameters), fp64, returned as floats.
std::vector<float> yarn_inv_freq(int dim, double theta, const RopeScalingYarn &);
// The checkpoint's RoPE frequencies: plain theta, llama3 or YaRN.
std::vector<float> rope_inv_freq(const ModelConfig &);
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
    bool contains(const std::string &name) const { return header_.contains(name); }
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
enum class CodeLayout { RowMajor, GroupMajor, Sorted, Lane };
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
    bool has_tensor(const std::string &name) const { return index_.contains(name); }
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
