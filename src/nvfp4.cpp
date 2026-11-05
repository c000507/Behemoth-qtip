#include "nvfp4.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.hpp"

namespace qtip {

Nvfp4Tensor nvfp4_quantize(const float* w, int m, int n) {
  QCHECK(m > 0 && n > 0 && n % 16 == 0, "NVFP4 needs n % 16 == 0");
  float maximum = 0;
  const size_t numel = size_t(m) * n;
#pragma omp parallel for schedule(static) reduction(max:maximum) if(numel >= (1u << 20))
  for (size_t i = 0; i < numel; ++i) {
    QCHECK(std::isfinite(w[i]), "nonfinite NVFP4 input");
    maximum = std::max(maximum, std::fabs(w[i]));
  }
  Nvfp4Tensor t;
  t.m = m, t.n = n;
  t.global = maximum ? maximum / (448.f * 6) : 1.f;
  QCHECK(t.global > 0 && std::isfinite(t.global), "NVFP4 global scale underflow/overflow");
  t.packed.resize(numel / 2);
  t.scales.resize(numel / 16);
  auto fp4 = [](float x) {
    constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float a = std::fabs(x), best = INFINITY;
    int code = 0;
    for (int i = 0; i < 8; ++i) {
      float e = std::fabs(a - levels[i]);
      if (e < best || (e == best && (i % 2 == 0))) {
        best = e;
        code = i;
      }
    }
    return uint8_t(code | (std::signbit(x) ? 8 : 0));
  };
#pragma omp parallel for schedule(static) if(numel >= (1u << 20))
  for (size_t b = 0; b < t.scales.size(); ++b) {
    float mx = 0;
    for (int j = 0; j < 16; ++j) mx = std::max(mx, std::fabs(w[b * 16 + j]));
    uint8_t s = f32_to_fp8e4m3(mx / (6 * t.global));
    t.scales[b] = s;
    float scale = fp8e4m3_to_f32(s) * t.global;
    for (int j = 0; j < 16; j += 2)
      t.packed[b * 8 + j / 2] = fp4(scale ? w[b * 16 + j] / scale : 0) | (fp4(scale ? w[b * 16 + j + 1] / scale : 0) << 4);
  }
  return t;
}

std::vector<float> nvfp4_dequantize(const Nvfp4Tensor& t) {
  QCHECK(t.m > 0 && t.n > 0 && t.n % 16 == 0 && t.packed.size() == size_t(t.m) * t.n / 2 &&
             t.scales.size() == size_t(t.m) * t.n / 16 && t.global > 0 && std::isfinite(t.global),
         "invalid NVFP4 payload");
  std::vector<float> out(size_t(t.m) * t.n);
#pragma omp parallel for schedule(static) if(out.size() >= (1u << 20))
  for (size_t i = 0; i < out.size(); ++i) {
    uint8_t code = (t.packed[i / 2] >> (i % 2 * 4)) & 15;
    float s = fp8e4m3_to_f32(t.scales[i / 16]);
    QCHECK(s >= 0 && std::isfinite(s), "invalid NVFP4 block scale");
    out[i] = bf16_to_f32(f32_to_bf16(nvfp4_level(code) * s * t.global));
  }
  return out;
}

double nvfp4_proxy_error(const float* w, const float* what, int m, int n, const float* H) {
  // tr(E H E^T) / m with E = w - what; H null => identity
  double total = 0;
#pragma omp parallel for reduction(+ : total) schedule(static)
  for (int r = 0; r < m; ++r) {
    std::vector<double> e(size_t(n), 0.0);
    for (int j = 0; j < n; ++j) e[size_t(j)] = double(w[size_t(r) * n + j]) - what[size_t(r) * n + j];
    if (!H) {
      for (int j = 0; j < n; ++j) total += e[size_t(j)] * e[size_t(j)];
      continue;
    }
    for (int i = 0; i < n; ++i) {
      double acc = 0;
      for (int j = 0; j < n; ++j) acc += double(H[size_t(i) * n + j]) * e[size_t(j)];
      total += e[size_t(i)] * acc;
    }
  }
  return total / m;
}

}  // namespace qtip
