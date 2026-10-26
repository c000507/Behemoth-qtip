#include "behemoth/attention.hpp"
#include "behemoth/checkpoint.hpp"
#include <array>

namespace behemoth {
namespace {
std::vector<BFloat16> norm(std::span<const BFloat16> x, const std::vector<BFloat16> &weights) {
    double squares = 0;
    for (auto v : x)
        squares += double(float(v)) * float(v);
    double factor = 1 / std::sqrt(squares / double(x.size()) + 1e-5);
    std::vector<BFloat16> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = BFloat16(float(double(float(x[i])) * factor * float(weights[i])));
    return out;
}
std::vector<float> floats(std::span<const BFloat16> x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = float(x[i]);
    return out;
}
std::vector<BFloat16> rounded(const std::vector<float> &x) {
    std::vector<BFloat16> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = BFloat16(x[i]);
    return out;
}
} // namespace
std::vector<BFloat16> reference_block(const Checkpoint &checkpoint, int layer,
                                      std::span<const BFloat16> input) {
    if (input.size() != 12288 || layer < 0 || layer >= 88)
        throw std::invalid_argument(
            "Block reference accepts one 12288-wide token at position zero");
    std::vector<BFloat16> residual(input.begin(), input.end());
    auto prefix = "model.layers." + std::to_string(layer) + ".";
    auto normalized = norm(input, checkpoint.bf16(prefix + "input_layernorm.weight"));
    auto x = floats(normalized);
    auto col = [&](const char *name, int rank, std::span<const float> values) {
        return rounded(
            reference_projection(checkpoint.projection(layer, name, rank, true), values));
    };
    std::array<std::vector<BFloat16>, 2> attention;
    for (int rank = 0; rank < 2; ++rank) {
        auto q = col("self_attn.q_proj", rank, x), k = col("self_attn.k_proj", rank, x),
             v = col("self_attn.v_proj", rank, x);
        // At position zero RoPE is the identity. Storage/indexing is independently scalar.
        AttentionLayout l;
        l.physical_pages = 1;
        l.table_stride = 1;
        l.query_offsets = {0, 1};
        l.positions = {0};
        l.lengths = {1};
        l.page_table = {0};
        AttentionData a;
        a.q.resize(48 * 128);
        a.q_scale.resize(48);
        a.k.assign(4 * 16 * 128, 0);
        a.v = a.k;
        a.k_scale.assign(4 * 16, 1);
        a.v_scale = a.k_scale;
        for (int h = 0; h < 48; ++h) {
            auto fq = floats(std::span(q).subspan(size_t(h) * 128, 128));
            auto quantized = quantize(fq);
            a.q_scale[size_t(h)] = quantized.scale;
            std::copy(quantized.values.begin(), quantized.values.end(), a.q.begin() + h * 128);
        }
        reference_append(l, a, k, v);
        attention[size_t(rank)] = rounded(reference_attention(l, a, true).output);
    }
    auto row_add = [&](const char *name, const std::array<std::vector<BFloat16>, 2> &values) {
        auto p0 = checkpoint.projection(layer, name, 0, false),
             p1 = checkpoint.projection(layer, name, 1, false);
        auto y0 = reference_projection_unscaled(p0, floats(values[0])),
             y1 = reference_projection_unscaled(p1, floats(values[1]));
        for (size_t i = 0; i < residual.size(); ++i) {
            float update = (y0[i] + y1[i]) * p0.row_scale[i];
            residual[i] = BFloat16(float(residual[i]) + float(BFloat16(update)));
        }
    };
    row_add("self_attn.o_proj", attention);
    normalized = norm(residual, checkpoint.bf16(prefix + "post_attention_layernorm.weight"));
    x = floats(normalized);
    std::array<std::vector<BFloat16>, 2> mlp;
    for (int rank = 0; rank < 2; ++rank) {
        auto gate = col("mlp.gate_proj", rank, x), up = col("mlp.up_proj", rank, x);
        mlp[size_t(rank)].resize(gate.size());
        for (size_t i = 0; i < gate.size(); ++i) {
            float g = float(gate[i]);
            mlp[size_t(rank)][i] = BFloat16(g / (1 + std::exp(-g)) * float(up[i]));
        }
    }
    row_add("mlp.down_proj", mlp);
    return residual;
}
} // namespace behemoth
