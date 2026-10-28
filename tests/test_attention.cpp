// Quantizer attention numerical checks and optional real-shape timing.
// qtip_attention_tests [--device N] [--bench 32768]
#include "gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

using namespace qtip;
namespace {
void check(bool ok, const char* message) { QCHECK(ok, message); }
std::vector<float> reference(const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
                             const std::vector<uint16_t>& v, int T, int nq, int nkv, int hd) {
  std::vector<float> out(size_t(T) * nq * hd);
  std::vector<double> scores(T), sum(hd);
  const double scale = 1 / std::sqrt(double(hd));
  for (int h = 0; h < nq; ++h)
    for (int t = 0; t < T; ++t) {
      int kh = h / (nq / nkv);
      double top = -INFINITY;
      for (int key = 0; key <= t; ++key) {
        double value = 0;
        for (int d = 0; d < hd; ++d)
          value += double(bf16_to_f32(q[(size_t(h) * T + t) * hd + d])) *
                   bf16_to_f32(k[(size_t(kh) * T + key) * hd + d]);
        scores[key] = value * scale;
        top = std::max(top, scores[key]);
      }
      std::fill(sum.begin(), sum.end(), 0);
      double denominator = 0;
      for (int key = 0; key <= t; ++key) {
        double p = std::exp(scores[key] - top);
        denominator += p;
        for (int d = 0; d < hd; ++d) sum[d] += p * bf16_to_f32(v[(size_t(kh) * T + key) * hd + d]);
      }
      for (int d = 0; d < hd; ++d) out[(size_t(t) * nq + h) * hd + d] = float(sum[d] / denominator);
    }
  return out;
}
void compare(const std::vector<uint16_t>& got, const std::vector<float>& ref, const char* message) {
  check(got.size() == ref.size(), "attention result size");
  double ss = 0, rr = 0, max_error = 0, maximum = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double actual = bf16_to_f32(got[i]), error = actual - ref[i];
    check(std::isfinite(actual), "nonfinite attention output");
    ss += error * error;
    rr += double(ref[i]) * ref[i];
    max_error = std::max(max_error, std::abs(error));
    maximum = std::max(maximum, std::abs(double(ref[i])));
  }
  double relative = std::sqrt(ss / std::max(rr, 1e-30));
  std::printf("  %s: relative L2 %.4g, max error %.4g\n", message, relative, max_error);
  check(relative < .008 && max_error < .001 + .02 * maximum, message);
}
void legacy(Gemm& gemm, const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
            int T, int nq, int nkv, int hd, hipStream_t stream) {
  int group = nq / nkv;
  DBuf<float> scores(size_t(group) * T * T), values(size_t(group) * T * hd);
  DBuf<uint16_t> probabilities(size_t(group) * T * T);
  for (int kh = 0; kh < nkv; ++kh) {
    gemm.run(true, false, T, group * T, hd, k + size_t(kh) * T * hd, hd,
             q + size_t(kh * group) * T * hd, hd, scores, T, OutT::F32);
    k_causal_softmax(scores, probabilities, group, T, 1.f / std::sqrt(float(hd)), stream);
    gemm.run(false, false, hd, group * T, T, v + size_t(kh) * T * hd, hd,
             probabilities, T, values, hd, OutT::F32);
    k_heads_to_tokens(values, out, group, T, hd, kh * group, nq * hd, stream);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
}
void numerical(hipStream_t stream) {
  Gemm gemm(stream);
  struct Case { int T, nq, nkv, hd; float scale; };
  for (auto [T, nq, nkv, hd, scale] : std::vector<Case>{
      {1, 1, 1, 128, 1}, {15, 6, 2, 128, 1}, {16, 12, 1, 128, 1},
      {17, 24, 2, 128, 4}, {31, 3, 1, 96, 1}, {32, 24, 2, 128, 1},
      {33, 20, 1, 128, 1}, {65, 96, 8, 128, 1}, {129, 4, 2, 256, 1}, {17, 4, 2, 64, 0}}) {
    std::printf("T=%d Q=%d KV=%d D=%d scale=%g\n", T, nq, nkv, hd, scale);
    Rng rng(71 + T);
    std::vector<uint16_t> q(size_t(nq) * T * hd), k(size_t(nkv) * T * hd), v(k.size());
    for (auto& x : q) x = f32_to_bf16(float(rng.normal() * scale));
    for (auto& x : k) x = f32_to_bf16(float(rng.normal() * scale));
    for (auto& x : v) x = f32_to_bf16(float(rng.normal()));
    const size_t size = size_t(T) * nq * hd;
    const uint16_t guard = f32_to_bf16(-19);
    std::vector<uint16_t> guarded(size + 32, guard);
    DBuf<uint16_t> dq(q.size()), dk(k.size()), dv(v.size()), output(guarded.size());
    HIP_CHECK(hipMemcpyAsync(dq, q.data(), q.size() * 2, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dk, k.data(), k.size() * 2, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dv, v.data(), v.size() * 2, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(output, guarded.data(), guarded.size() * 2, hipMemcpyHostToDevice, stream));
    k_tiled_attention(dq, dk, dv, output + 16, T, nq, nkv, hd, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(guarded.data(), output, guarded.size() * 2, hipMemcpyDeviceToHost));
    for (int i = 0; i < 16; ++i)
      check(guarded[i] == guard && guarded[guarded.size() - 1 - i] == guard, "attention output guard changed");
    std::vector<uint16_t> got(guarded.begin() + 16, guarded.end() - 16);
    compare(got, reference(q, k, v, T, nq, nkv, hd), "tiled vs FP64 attention");
    for (int h = 0; h < nq; ++h)
      for (int d = 0; d < hd; ++d)
        check(got[size_t(h) * hd + d] == v[size_t(h / (nq / nkv)) * T * hd + d], "first token must equal V");
    if (T == 32) {
      legacy(gemm, dq, dk, dv, output + 16, T, nq, nkv, hd, stream);
      HIP_CHECK(hipMemcpy(guarded.data(), output, guarded.size() * 2, hipMemcpyDeviceToHost));
      std::vector<float> old(size);
      for (size_t i = 0; i < size; ++i) old[i] = bf16_to_f32(guarded[i + 16]);
      compare(got, old, "tiled vs previous GEMM/softmax path");
    }
    if (T > 1) {
      // Poison the last key/value: the entire preceding output prefix must be
      // bitwise unchanged, including query tiles overlapping the causal edge.
      for (int h = 0; h < nkv; ++h)
        for (int d = 0; d < hd; ++d) {
          k[(size_t(h) * T + T - 1) * hd + d] = f32_to_bf16(21.f);
          v[(size_t(h) * T + T - 1) * hd + d] = f32_to_bf16(-31.f);
        }
      HIP_CHECK(hipMemcpyAsync(dk, k.data(), k.size() * 2, hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipMemcpyAsync(dv, v.data(), v.size() * 2, hipMemcpyHostToDevice, stream));
      k_tiled_attention(dq, dk, dv, output + 16, T, nq, nkv, hd, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipMemcpy(guarded.data(), output, guarded.size() * 2, hipMemcpyDeviceToHost));
      check(std::equal(got.begin(), got.begin() + size_t(T - 1) * nq * hd, guarded.begin() + 16),
             "future key/value changed an earlier query");
    }
  }
}
void benchmark(int T, hipStream_t stream) {
  check(T > 0 && T <= 131072, "benchmark length outside [1,131072]");
  constexpr int nq = 96, nkv = 8, hd = 128, group = nq / nkv;
  const size_t size = size_t(T) * nq * hd, kvsize = size_t(T) * nkv * hd;
  std::vector<uint16_t> q(size, 0), k(kvsize, 0), v(kvsize), out(size);
  for (int h = 0; h < nkv; ++h)
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < hd; ++d)
        v[(size_t(h) * T + t) * hd + d] = f32_to_bf16(float(h + 1) / 16 + float((t * 3 + d) % 29 - 14) / 32);
  DBuf<uint16_t> dq(size), dk(kvsize), dv(kvsize), output(size);
  HIP_CHECK(hipMemcpyAsync(dq, q.data(), size * 2, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dk, k.data(), kvsize * 2, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dv, v.data(), kvsize * 2, hipMemcpyHostToDevice, stream));
  k_tiled_attention(dq, dk, dv, output, T, nq, nkv, hd, stream);
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  constexpr int repeats = 3;
  for (int i = 0; i < repeats; ++i) k_tiled_attention(dq, dk, dv, output, T, nq, nkv, hd, stream);
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  d2h_staged(out.data(), output, size * 2, stream);
  for (auto x : out) check(std::isfinite(bf16_to_f32(x)), "long-context output is nonfinite");
  std::set<int> queries{0, 15, 16, 31, 32, 511, 8191, T - 1};
  for (int h : {0, 11, 12, nq - 1}) {
    std::vector<double> sums(hd);
    for (int t = 0; t < T; ++t) {
      for (int d = 0; d < hd; ++d) sums[d] += bf16_to_f32(v[(size_t(h / group) * T + t) * hd + d]);
      if (!queries.contains(t)) continue;
      for (int d = 0; d < hd; ++d)
        check(std::abs(bf16_to_f32(out[(size_t(t) * nq + h) * hd + d]) - sums[d] / (t + 1)) < .003,
              "long-context uniform attention differs from prefix mean");
    }
  }
  double seconds = double(ms) / (1000 * repeats), flops = 2.0 * nq * hd * T * double(T + 1);
  std::printf("T=%d Q=%d KV=%d D=%d: %.3f ms, %.2f useful TFLOP/s, %.3f GiB Q/K/V/output, zero global attention workspace\n",
              T, nq, nkv, hd, ms / repeats, flops / seconds / 1e12, double(2 * (size + kvsize)) * 2 / (1ull << 30));
}
}  // namespace
int main(int argc, char** argv) {
  int device = 0, length = 0;
  for (int i = 1; i < argc; ++i) {
    check(i + 1 < argc, "missing option value");
    std::string key = argv[i], value = argv[++i];
    size_t used = 0;
    int n = std::stoi(value, &used);
    check(used == value.size(), "invalid integer");
    if (key == "--device") device = n;
    else if (key == "--bench") { check(n > 0, "benchmark length must be positive"); length = n; }
    else check(false, "unknown attention test option");
  }
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::puts("no HIP device; skipped");
    return 77;
  }
  HIP_CHECK(hipSetDevice(device));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  numerical(stream);
  if (length) benchmark(length, stream);
  HIP_CHECK(hipStreamDestroy(stream));
  std::puts("quantizer tiled attention passed");
  return 0;
}
