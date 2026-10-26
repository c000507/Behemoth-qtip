#pragma once
// Host-only QTIP prefill transport policy. Payload sizes are per rank, per token,
// per layer, in each direction; both rank halves use the same element precision.
#include <cstddef>

namespace behemoth {
inline constexpr size_t qtip_fp32_exchange_bytes =
    7168 * sizeof(float) + 6144 * 2 + 6144 * sizeof(float) + 28672 * 2 + 6144 * sizeof(float);
inline constexpr size_t qtip_bf16_exchange_bytes = (7168 + 6144 + 6144 + 28672 + 6144) * 2;

inline bool qtip_bf16_prefill_enabled(int rows, int threshold, bool has_decode) {
    // Never change any decode-only or mixed decode/prefill step. Thresholds
    // below 64 are rejected by the runtime/server before loading the model.
    return threshold >= 64 && rows >= threshold && !has_decode;
}
} // namespace behemoth
