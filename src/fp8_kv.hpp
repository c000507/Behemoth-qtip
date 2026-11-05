// BF16 -> FP8-Dynamic converter for K/V projections of BF16 source checkpoints (--kv-format fp8).
// Output layout is exactly what FP8-Dynamic checkpoints store: `weight` F8_E4M3 [out][in] and `weight_scale`
// BF16 [out] (per output channel), with W ~= fp8(w) * scale. Scale = absmax(row)/448 rounded to bf16 (nudged up one
// bf16 step if the rounding would make the row absmax overflow 448); elements are rounded to nearest even E4M3
// and saturate at +-448 (never NaN).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "nvfp4.hpp"

namespace qtip {

struct Fp8Linear {
  int m = 0, n = 0;
  std::vector<uint8_t> w;       // [m][n] E4M3
  std::vector<uint16_t> scale;  // [m] bf16
};

inline uint16_t fp8_row_scale_bf16(float absmax) {
  if (!(absmax > 0.f) || !std::isfinite(absmax)) return f32_to_bf16(1.f);
  uint16_t s = f32_to_bf16(absmax / 448.f);
  if (bf16_to_f32(s) * 448.f < absmax) ++s;  // positive finite: next bf16 value is larger
  return s;
}

// w_bf16: raw bf16 bits [m][n]
inline Fp8Linear bf16_to_fp8_dynamic(const uint16_t* w_bf16, int m, int n) {
  Fp8Linear r;
  r.m = m;
  r.n = n;
  r.w.resize(size_t(m) * n);
  r.scale.resize(size_t(m));
#pragma omp parallel for schedule(static) if(size_t(m) * n >= (1u << 18))
  for (int i = 0; i < m; ++i) {
    const uint16_t* row = w_bf16 + size_t(i) * n;
    float mx = 0.f;
    for (int j = 0; j < n; ++j) mx = std::max(mx, std::fabs(bf16_to_f32(row[j])));
    const uint16_t sb = fp8_row_scale_bf16(mx);
    const float inv = 1.f / bf16_to_f32(sb);
    r.scale[size_t(i)] = sb;
    for (int j = 0; j < n; ++j) r.w[size_t(i) * n + j] = f32_to_fp8e4m3(bf16_to_f32(row[j]) * inv);
  }
  return r;
}

// dequantized weights as bf16 bits (what the runtime effectively multiplies with)
inline std::vector<uint16_t> fp8_dynamic_dequant_bf16(const Fp8Linear& r) {
  std::vector<uint16_t> out(size_t(r.m) * r.n);
  for (int i = 0; i < r.m; ++i) {
    const float sc = bf16_to_f32(r.scale[size_t(i)]);
    for (int j = 0; j < r.n; ++j) out[size_t(i) * r.n + j] = f32_to_bf16(fp8e4m3_to_f32(r.w[size_t(i) * r.n + j]) * sc);
  }
  return out;
}

}  // namespace qtip
