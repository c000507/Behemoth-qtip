#pragma once
#include "numeric.hpp"
#include <cstddef>
#include <vector>

namespace behemoth {
inline constexpr int head_dim = 128, page_tokens = 16, key_tile = 32, query_tile = 16;
struct AttentionLayout {
    int query_heads = 48, kv_heads = 4, physical_pages = 0, table_stride = 0;
    std::vector<int> query_offsets; // requests + 1
    std::vector<int> positions;     // total queries, absolute sequence positions
    std::vector<int> lengths;       // requests, including newly appended KV
    std::vector<int> page_table;    // requests * table_stride
    void validate() const;
    int queries() const {
        return query_offsets.empty() ? 0 : query_offsets.back();
    }
    int requests() const {
        return int(lengths.size());
    }
};
struct AttentionData {
    std::vector<uint8_t> q, k, v;
    std::vector<float> q_scale, k_scale, v_scale;
    std::vector<BFloat16> q_bf16; // optional unquantised queries (BF16 Q/P mode)
    void validate(const AttentionLayout &) const;
};
struct AttentionResult {
    std::vector<float> output;               // before BF16 rounding
    std::vector<float> maximum, denominator; // pre-normalization online-softmax statistics
};
// Independent scalar reference. partition_keys=0 is unsplit; otherwise a multiple of 32.
// With a.q_bf16 populated the queries are taken from it (q/q_scale ignored); pass
// quantize_probabilities=false for the BF16 Q/P mode (P then stays unquantised).
AttentionResult reference_attention(const AttentionLayout &, const AttentionData &,
                                    bool quantize_probabilities = true, int partition_keys = 0);
void reference_append(const AttentionLayout &, AttentionData &, std::span<const BFloat16> k,
                      std::span<const BFloat16> v);
struct ErrorMetrics {
    double rms_error{}, reference_rms{}, max_abs{};
    bool pass{};
};
ErrorMetrics compare_attention(std::span<const float> actual, std::span<const float> expected);
} // namespace behemoth
