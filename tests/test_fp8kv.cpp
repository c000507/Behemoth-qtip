// bf16 -> FP8-Dynamic K/V converter (--kv-format fp8 for BF16 sources): layout, scale convention, rounding,
// saturation, exact reproduction of a hand-built FP8-Dynamic fixture. CPU only.
#include <cmath>
#include <cstdio>

#include "fp8_kv.hpp"

using namespace qtip;
static int g_fail = 0;
#define EXPECT(c, ...)                              \
  do {                                              \
    if (!(c)) {                                     \
      std::printf("  FAIL(line %d): ", __LINE__);   \
      std::printf(__VA_ARGS__);                     \
      std::printf("\n");                            \
      ++g_fail;                                     \
    }                                               \
  } while (0)

int main() {
  Rng r(3);
  const int m = 40, n = 192;
  // 1. FP8-Dynamic fixture: bytes b[i][j] (non-NaN, row max code 0x7e = 448) and bf16 scale s[i]; W = fp8(b)*s is
  // exactly representable in bf16 (4 significant bits).
  std::vector<uint8_t> b(size_t(m) * n);
  std::vector<uint16_t> s(m), wbf(b.size());
  for (int i = 0; i < m; ++i) {
    s[i] = uint16_t(f32_to_bf16(float(std::exp(0.7 * r.normal()) * 1e-3)) & 0xfff0);  // 4 significant bits: products stay exact in bf16
    for (int j = 0; j < n; ++j) {
      uint8_t c = uint8_t(r.next() % 127);  // 0..126 (0x7f is NaN)
      if (r.next() & 1) c |= 0x80;
      b[size_t(i) * n + j] = c;
    }
    b[size_t(i) * n + size_t(r.next() % n)] = (i & 1) ? 0xFE : 0x7E;  // +-448
    for (int j = 0; j < n; ++j) wbf[size_t(i) * n + j] = f32_to_bf16(fp8e4m3_to_f32(b[size_t(i) * n + j]) * bf16_to_f32(s[i]));
  }
  Fp8Linear f = bf16_to_fp8_dynamic(wbf.data(), m, n);
  EXPECT(f.m == m && f.n == n && f.w.size() == b.size() && f.scale.size() == size_t(m), "shapes");
  int bad_s = 0, bad_w = 0;
  for (int i = 0; i < m; ++i) {
    bad_s += f.scale[i] != s[i];
    for (int j = 0; j < n; ++j) bad_w += f.w[size_t(i) * n + j] != b[size_t(i) * n + j];
  }
  EXPECT(bad_s == 0 && bad_w == 0, "FP8-Dynamic fixture must be reproduced exactly (scale mismatches %d, byte mismatches %d)", bad_s, bad_w);
  auto dq = fp8_dynamic_dequant_bf16(f);
  EXPECT(dq == wbf, "dequantization of the fixture must be exact");

  // 2. random weights: error bound, no NaN, scale = absmax/448 (to bf16 + nudge)
  std::vector<uint16_t> w(size_t(m) * n);
  for (int i = 0; i < m; ++i) {
    const double rs = std::exp(1.0 * r.normal()) * 0.05;
    for (int j = 0; j < n; ++j) w[size_t(i) * n + j] = f32_to_bf16(float(rs * r.normal() * (r.next() % 50 == 0 ? 6.0 : 1.0)));
  }
  f = bf16_to_fp8_dynamic(w.data(), m, n);
  double worst = 0;
  int nan = 0, sat = 0;
  for (int i = 0; i < m; ++i) {
    float mx = 0;
    for (int j = 0; j < n; ++j) mx = std::max(mx, std::fabs(bf16_to_f32(w[size_t(i) * n + j])));
    const float sc = bf16_to_f32(f.scale[i]);
    EXPECT(sc * 448.f >= mx && sc <= mx / 448.f * 1.01f, "row %d scale %g vs absmax/448 %g", i, sc, mx / 448.f);
    for (int j = 0; j < n; ++j) {
      const uint8_t c = f.w[size_t(i) * n + j];
      nan += (c & 0x7f) == 0x7f;
      sat += (c & 0x7f) == 0x7e;
      const float x = bf16_to_f32(w[size_t(i) * n + j]), y = fp8e4m3_to_f32(c) * sc;
      // E4M3: 3 mantissa bits -> relative error <= 2^-4 for normals; subnormal region absolute <= 2^-10 * scale
      const double tol = std::max(std::fabs(x) / 16.0, 0.5 * std::ldexp(1.0, -9) * sc) * 1.001 + 1e-30;
      worst = std::max(worst, std::fabs(x - y) / tol);
    }
  }
  EXPECT(nan == 0, "no NaN codes (%d)", nan);
  EXPECT(sat >= m / 2, "row maxima map to the saturating code (%d)", sat);
  EXPECT(worst <= 1.0, "round-trip error exceeds the E4M3 bound (worst %.3g x bound)", worst);

  // 3. saturation / specials of the element encoder and degenerate rows
  EXPECT(f32_to_fp8e4m3(1e9f) == 0x7e && f32_to_fp8e4m3(-1e9f) == 0xfe && f32_to_fp8e4m3(449.f) == 0x7e, "saturate, not NaN");
  EXPECT(f32_to_fp8e4m3(1.0625f) == f32_to_fp8e4m3(1.0f) && f32_to_fp8e4m3(1.1875f) == f32_to_fp8e4m3(1.25f), "round-to-nearest-even ties");
  std::vector<uint16_t> z(2 * 16, 0);
  z[20] = f32_to_bf16(3.f);
  Fp8Linear fz = bf16_to_fp8_dynamic(z.data(), 2, 16);
  EXPECT(bf16_to_f32(fz.scale[0]) == 1.f && fz.w[0] == 0, "all-zero row: scale 1, zero codes");
  EXPECT(fp8_dynamic_dequant_bf16(fz)[20] == z[20], "single-value row exact");
  for (float am : {0.017f, 1.f, 3.3e-3f, 123.456f, 448.f}) {
    uint16_t sb = fp8_row_scale_bf16(am);
    EXPECT(bf16_to_f32(sb) * 448.f >= am, "scale covers absmax %g", am);
  }
  std::printf(g_fail ? "FP8 K/V TESTS FAILED (%d)\n" : "ALL FP8 K/V TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
