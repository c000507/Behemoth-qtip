// NVFP4 weight-only quantization, format identical to the GLM converter (glm53/src/weights.cpp):
// two E2M1 values per byte (low nibble first), one E4M3 scale per 16 consecutive input channels of a
// row, one F32 global scale per matrix. Tensors are stored as <base>.weight_packed (U8 [m, n/2]),
// <base>.weight_scale (F8_E4M3 [m, n/16]) and <base>.weight_global_scale (F32 scalar).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace qtip {

// OCP FP8 E4M3 "fn" encode (no infinities, 0x7f/0xff NaN, saturates to +-448).
// Port of behemoth::encode_e4m3 (runtime/include/behemoth/numeric.hpp); keeps both in sync.
inline uint8_t f32_to_fp8e4m3(float x) {
  const uint8_t sign = std::signbit(x) ? 128 : 0;
  if (std::isnan(x)) return uint8_t(sign | 127);
  x = std::min(std::fabs(x), 448.f);
  auto round_even = [](float v) {
    const float f = std::floor(v), r = v - f;
    unsigned n = static_cast<unsigned>(f);
    return n + unsigned(r > .5f || (r == .5f && (n & 1U)));
  };
  if (x < std::ldexp(1.f, -6)) return uint8_t(sign | round_even(x * 512.f));
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

struct Nvfp4Tensor {
  int m = 0, n = 0;               // out x in, n must be a multiple of 16
  std::vector<uint8_t> packed;    // [m][n/2], low nibble = first of two values
  std::vector<uint8_t> scales;    // [m][n/16] E4M3 block scales
  float global = 1;               // F32 global scale
  int64_t size_bytes() const { return int64_t(packed.size()) + int64_t(scales.size()) + 4; }
};

// Quantize a row-major fp32 [m][n] matrix (nearest-value reconstruction, as glm53 convert_nvfp4).
Nvfp4Tensor nvfp4_quantize(const float* w, int m, int n);

// Dequantize to fp32 values already rounded to bf16 (what execution sees; matches glm53
// dequantize_nvfp4 / the GPU gemv that widens in registers with bf16 rounding).
std::vector<float> nvfp4_dequantize(const Nvfp4Tensor& t);

// Proxy error tr((W-What) H (W-What)^T) / (rows) of a dequantized matrix (host, for tests/reporting).
double nvfp4_proxy_error(const float* w, const float* what, int m, int n, const float* H);

// Decode one value (for tests / loaders).
inline float nvfp4_level(uint8_t code) {
  constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return (code & 8 ? -1.f : 1.f) * levels[code & 7];
}

}  // namespace qtip
