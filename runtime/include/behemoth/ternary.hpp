// BitNet b1.58 ternary weight format shared by the trainer, exporter and runtime.
//
// Weights are {-1, 0, +1} with one FP32 scale per tensor (absmean of the latent
// weights). Activations are quantised per token to int8 with an absmax scale.
//
// Packing (per row, `in` must be a multiple of 16): sixteen consecutive input
// elements share one little-endian uint32 word. Element e (0..15) of a word is
// stored in byte e % 4 at bit offset 2 * (e / 4) as code = w + 1 in {0, 1, 2}
// (3 is invalid). Therefore (word >> 2j) & 0x03030303 yields elements
// 4j..4j+3 as four bytes, which pairs directly with four consecutive int8
// activations in a dp4a-style product: sum(w * x) = sum(code * x) - sum(x).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace behemoth::ternary {
inline constexpr int group = 16; // elements per packed word
inline constexpr int act_max = 127;

inline size_t packed_bytes(size_t in) {
    if (in % group)
        throw std::invalid_argument("ternary rows need a multiple of 16 inputs");
    return in / 4;
}
// code of element e within its 16-element word: byte e%4, bit offset 2*(e/4)
inline unsigned code_shift(unsigned e) { return 8 * (e % 4) + 2 * (e / 4); }

// Absmean quantiser: scale = mean|w| (eps-guarded); q = clip(round(w / scale), -1, 1).
inline float absmean(std::span<const float> w) {
    double s = 0;
    for (float v : w)
        s += std::fabs(v);
    return float(s / double(w.empty() ? 1 : w.size())) + 1e-8f;
}
inline int8_t ternarize(float w, float scale) {
    float r = std::nearbyint(w / scale);
    return int8_t(r < -1 ? -1 : r > 1 ? 1 : r);
}
// Packs `out` rows of `in` ternary values (int8 in {-1,0,1}) into [out, in/4] bytes.
inline void pack(std::span<const int8_t> q, size_t out, size_t in, std::span<uint8_t> dst) {
    const size_t rb = packed_bytes(in);
    if (q.size() != out * in || dst.size() != out * rb)
        throw std::invalid_argument("ternary pack size mismatch");
    std::memset(dst.data(), 0, dst.size());
    for (size_t r = 0; r < out; ++r)
        for (size_t i = 0; i < in; ++i) {
            const int v = q[r * in + i];
            if (v < -1 || v > 1)
                throw std::invalid_argument("ternary value outside {-1,0,1}");
            const unsigned sh = code_shift(unsigned(i % group));
            uint32_t word;
            uint8_t *p = dst.data() + r * rb + (i / group) * 4;
            std::memcpy(&word, p, 4);
            word |= uint32_t(v + 1) << sh;
            std::memcpy(p, &word, 4);
        }
}
inline int8_t unpack_one(const uint8_t *row, size_t i) {
    uint32_t word;
    std::memcpy(&word, row + (i / group) * 4, 4);
    const unsigned c = (word >> code_shift(unsigned(i % group))) & 3u;
    if (c == 3)
        throw std::runtime_error("invalid ternary code 3");
    return int8_t(int(c) - 1);
}
inline std::vector<int8_t> unpack(std::span<const uint8_t> packed, size_t out, size_t in) {
    const size_t rb = packed_bytes(in);
    if (packed.size() != out * rb)
        throw std::invalid_argument("ternary unpack size mismatch");
    std::vector<int8_t> q(out * in);
    for (size_t r = 0; r < out; ++r)
        for (size_t i = 0; i < in; ++i)
            q[r * in + i] = unpack_one(packed.data() + r * rb, i);
    return q;
}
// Per-token int8 absmax quantisation. Returns the scale (absmax / 127, eps-guarded).
inline float quantize_activation(std::span<const float> x, std::span<int8_t> q) {
    float m = 0;
    for (float v : x)
        m = std::fmax(m, std::fabs(v));
    const float scale = (m > 0 ? m : 1e-5f) / float(act_max);
    for (size_t i = 0; i < x.size(); ++i) {
        float r = std::nearbyint(x[i] / scale);
        q[i] = int8_t(r < -act_max ? -act_max : r > act_max ? act_max : r);
    }
    return scale;
}
// Exact int32 reference of one packed row against int8 activations.
inline int32_t dot_row(const uint8_t *row, const int8_t *x, size_t in) {
    int32_t acc = 0;
    for (size_t i = 0; i < in; ++i)
        acc += int32_t(unpack_one(row, i)) * int32_t(x[i]);
    return acc;
}
} // namespace behemoth::ternary
