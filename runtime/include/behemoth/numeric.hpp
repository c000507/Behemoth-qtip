#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace behemoth {
struct BFloat16 {
    uint16_t bits{};
    BFloat16() = default;
    static BFloat16 from_bits(uint16_t b) {
        BFloat16 r;
        r.bits = b;
        return r;
    }
    explicit BFloat16(float x) {
        auto u = std::bit_cast<uint32_t>(x);
        if ((u & 0x7fffffffU) > 0x7f800000U)
            bits = uint16_t((u >> 16) | 0x40U);
        else
            bits = uint16_t((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
    }
    explicit operator float() const {
        return std::bit_cast<float>(uint32_t(bits) << 16);
    }
};
static_assert(sizeof(BFloat16) == 2);

// File-format decoding only: never constructs or computes on an FP16 value.
inline float decode_checkpoint_f16(uint16_t bits) {
    unsigned e = (bits >> 10) & 31U, m = bits & 1023U;
    float x;
    if (e == 0)
        x = std::ldexp(float(m), -24);
    else if (e == 31)
        x = m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else
        x = std::ldexp(1.f + float(m) / 1024.f, int(e) - 15);
    return (bits & 0x8000U) ? -x : x;
}
inline float decode_e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float x = e == 0 ? std::ldexp(float(m), -9)
                     : (e == 15 && m == 7 ? std::numeric_limits<float>::quiet_NaN()
                                          : std::ldexp(1.f + float(m) / 8.f, e - 7));
    return (b & 128) ? -x : x;
}
inline unsigned round_even(float x) {
    const float f = std::floor(x), r = x - f;
    unsigned n = static_cast<unsigned>(f);
    return n + unsigned(r > .5f || (r == .5f && (n & 1U)));
}
inline uint8_t encode_e4m3(float x) {
    // Runtime policy: finite overflow and infinities saturate; NaNs retain sign.
    const uint8_t sign = std::signbit(x) ? 128 : 0;
    if (std::isnan(x))
        return uint8_t(sign | 127);
    x = std::min(std::abs(x), 448.f);
    if (x < std::ldexp(1.f, -6))
        return uint8_t(sign | round_even(x * 512.f));
    int exp;
    const float frac = std::frexp(x, &exp) * 2;
    unsigned mant = round_even((frac - 1) * 8);
    int encoded_exp = exp + 6;
    if (mant == 8) {
        mant = 0;
        ++encoded_exp;
    }
    return uint8_t(sign | std::min(126U, (unsigned(encoded_exp) << 3) | mant));
}
inline float quantization_scale(std::span<const float> x) {
    float m = 0;
    for (float v : x) {
        if (!std::isfinite(v))
            throw std::invalid_argument("Nonfinite quantization input");
        m = std::max(m, std::abs(v));
    }
    return m == 0 ? 1.f : std::max(m / 448.f, std::numeric_limits<float>::min());
}
struct QuantizedVector {
    std::vector<uint8_t> values;
    float scale;
};
inline QuantizedVector quantize(std::span<const float> x) {
    QuantizedVector q{{}, quantization_scale(x)};
    q.values.reserve(x.size());
    for (float v : x)
        q.values.push_back(encode_e4m3(v / q.scale));
    return q;
}
inline size_t checked_mul(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("Size overflow");
    return a * b;
}
inline size_t ceil_div(size_t n, size_t d) {
    return n / d + size_t(n % d != 0);
}
} // namespace behemoth
