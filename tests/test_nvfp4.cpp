// NVFP4 quantizer upgrades: scale search and Hessian-calibrated (LDLQ) rounding vs the naive quantizer.
// Output format must stay byte-compatible: tensors decode through nvfp4_dequantize, GPU kernels with both
// upgrades off reproduce the naive CPU quantizer bit for bit.
#include <cmath>
#include <cstdio>

#include "common.hpp"
#include "gpu.hpp"
#include "nvfp4.hpp"
#include "nvfp4_gpu.hpp"

using namespace qtip;
static int g_fail = 0;
#define EXPECT(c, ...)                                  \
  do {                                                  \
    if (!(c)) {                                         \
      std::printf("  FAIL(line %d): ", __LINE__);      \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
      ++g_fail;                                         \
    }                                                   \
  } while (0)

// random PD Hessian with correlated, unevenly scaled input channels (X^T X / N)
static std::vector<float> make_hessian(int n, int samples, Rng& r) {
  std::vector<double> colscale(size_t(n), 0.0), mix(size_t(n) * 8, 0.0);
  for (auto& c : colscale) c = std::exp(0.8 * r.normal());
  for (auto& v : mix) v = r.normal();
  std::vector<double> X(size_t(samples) * n);
  for (int s = 0; s < samples; ++s)
    for (int j = 0; j < n; ++j) {
      double v = 0.5 * r.normal();
      for (int k = 0; k < 8; ++k) v += mix[size_t(j) * 8 + k] * r.normal() * 0.4;
      X[size_t(s) * n + j] = v * colscale[size_t(j)];
    }
  std::vector<float> H(size_t(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) {
      double a = 0;
      for (int s = 0; s < samples; ++s) a += X[size_t(s) * n + i] * X[size_t(s) * n + j];
      H[size_t(i) * n + j] = H[size_t(j) * n + i] = float(a / samples);
    }
  return H;
}

static void run_case(const char* name, bool heavy) {
  std::printf("[nvfp4 %s]\n", name);
  const int m = 96, n = 256;
  Rng r(heavy ? 11 : 5);
  std::vector<float> w(size_t(m) * n);
  for (auto& v : w) {
    double x = r.normal() * 0.02;
    if (heavy) x *= std::exp(1.2 * r.normal());  // log-normal scale mixture: heavy tails, outlier blocks
    v = float(x);
  }
  auto H = make_hessian(n, 600, r);
  hipStream_t s;
  HIP_CHECK(hipStreamCreate(&s));
  auto dq = [&](const Nvfp4Tensor& t) { return nvfp4_dequantize(t); };

  Nvfp4Tensor naive = nvfp4_quantize(w.data(), m, n);
  Nvfp4Opts off{false, false, 1e-2};
  auto gpu_naive = nvfp4_quantize_group(s, {w.data()}, {m}, n, nullptr, off)[0];
  EXPECT(gpu_naive.packed == naive.packed && gpu_naive.scales == naive.scales && gpu_naive.global == naive.global,
         "GPU kernel with search/calibration off must equal the naive quantizer byte for byte");

  Nvfp4Opts search{true, false, 1e-2};
  auto t_search_plain = nvfp4_quantize_group(s, {w.data()}, {m}, n, nullptr, search)[0];
  Nvfp4Opts both{true, true, 1e-2};
  bool used = false;
  auto t_cal = nvfp4_quantize_group(s, {w.data()}, {m}, n, H.data(), both, &used)[0];
  EXPECT(used, "Hessian must be used for calibration");
  auto t_search_h = nvfp4_quantize_group(s, {w.data()}, {m}, n, H.data(), search)[0];

  const double mse_naive = nvfp4_proxy_error(w.data(), dq(naive).data(), m, n, nullptr);
  const double mse_search = nvfp4_proxy_error(w.data(), dq(t_search_plain).data(), m, n, nullptr);
  const double p_naive = nvfp4_proxy_error(w.data(), dq(naive).data(), m, n, H.data());
  const double p_search = nvfp4_proxy_error(w.data(), dq(t_search_h).data(), m, n, H.data());
  const double p_cal = nvfp4_proxy_error(w.data(), dq(t_cal).data(), m, n, H.data());
  std::printf("  plain MSE: naive %.4g  scale-search %.4g\n", mse_naive, mse_search);
  std::printf("  Hessian proxy: naive %.4g  scale-search(diag H) %.4g  calibrated+search %.4g (%.1f%% of naive)\n",
              p_naive, p_search, p_cal, 100 * p_cal / p_naive);
  EXPECT(mse_search <= mse_naive * (1 + 1e-9), "scale search must not increase plain MSE");
  EXPECT(p_search <= p_naive * 1.02, "scale search (diag-H weighted) must not hurt the Hessian proxy");
  EXPECT(p_cal <= p_naive, "calibrated rounding must not be worse than naive in the Hessian proxy");
  EXPECT(p_cal < 0.9 * p_naive, "calibrated rounding should clearly beat naive");
  // calibrated-only (no scale search) must also beat naive
  auto t_cal_only = nvfp4_quantize_group(s, {w.data()}, {m}, n, H.data(), Nvfp4Opts{false, true, 1e-2})[0];
  const double p_cal_only = nvfp4_proxy_error(w.data(), dq(t_cal_only).data(), m, n, H.data());
  std::printf("  calibrated only (naive scales): %.4g\n", p_cal_only);
  EXPECT(p_cal_only <= p_naive, "calibrated rounding alone must not be worse than naive");
  // format validity: scale codes legal (<=126), tensors decode finite, geometry unchanged
  for (const Nvfp4Tensor* t : {&t_cal, &t_search_h, &t_search_plain}) {
    EXPECT(t->packed.size() == naive.packed.size() && t->scales.size() == naive.scales.size() && t->global == naive.global,
           "tensor geometry/global scale must match the naive format");
    bool ok = true;
    for (uint8_t c : t->scales) ok = ok && (c & 0x7f) <= 126;
    EXPECT(ok, "block scale code outside E4M3 range");
    for (float v : nvfp4_dequantize(*t)) ok = ok && std::isfinite(v);
    EXPECT(ok, "non-finite dequantized value");
  }
  // shared-Hessian group call equals separate calls
  auto grp = nvfp4_quantize_group(s, {w.data(), w.data()}, {m, m}, n, H.data(), both);
  EXPECT(grp[0].packed == t_cal.packed && grp[1].scales == t_cal.scales, "group call must equal single calls");
  HIP_CHECK(hipStreamDestroy(s));
}

int main() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::puts("no GPU device; skipped");
    return 77;
  }
  run_case("gaussian weights", false);
  run_case("heavy-tailed weights", true);
  std::printf(g_fail ? "NVFP4 TESTS FAILED (%d)\n" : "ALL NVFP4 TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
