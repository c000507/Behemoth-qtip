// CUDA-backend spot checks: two-pass tail-biting Viterbi and Hadamard vs the CPU (qtip_core) references.
// Built only with -DQTIP_CUDA=ON.  usage: qtip_tests_cuda [--bench B]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "gpu.hpp"
#include "qtip_core.hpp"
#include "quantizer.hpp"

using namespace qtip;
static int g_fail = 0;
#define EXPECT(c, ...)                                                                             \
  do {                                                                                             \
    if (!(c)) {                                                                                    \
      std::printf("  FAIL: " __VA_ARGS__);                                                         \
      std::printf("\n");                                                                           \
      ++g_fail;                                                                                    \
    }                                                                                              \
  } while (0)

static hipStream_t S;

// The compressed CUDA LUT must preserve every original half2 bit pattern,
// including the sign of the first component, across all 65536 states.
static void test_compact_codebook() {
  const auto& lut = codebook().lut_h2;
  std::vector<uint32_t> compact(1024);
  std::vector<bool> seen(1024, false);
  for (uint32_t s = 0; s < 65536; ++s) {
    uint32_t idx = ((s * (s + 1)) >> 6) & 1023;
    if (seen[idx]) EXPECT(compact[idx] == lut[s], "compact LUT collision at state %u", s);
    else compact[idx] = lut[s], seen[idx] = true;
  }
  EXPECT(std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }), "compact LUT has uninitialized entries");
}

// ---- Hadamard: k_hadamard_rows vs hadamard_rows_ref (fp64), hadU/hadUt, strided rows ----------------
static void test_hadamard() {
  std::printf("[hadamard cuda] vs fp64 reference (hadU / hadUt, strided rows)\n");
  for (int n : {96, 1024, 8192, 12288, 28672}) {
    long rows = 6, ld = n + 32;
    Rng r(n);
    std::vector<float> x(size_t(rows) * ld);
    for (auto& v : x) v = float(r.normal());
    for (int tr = 0; tr < 2; ++tr) {
      std::vector<double> ref(size_t(rows) * n);
      for (long i = 0; i < rows; ++i)
        for (int j = 0; j < n; ++j) ref[size_t(i) * n + j] = x[size_t(i) * ld + j];
      hadamard_rows_ref(ref.data(), rows, n, n, tr == 1);
      DBuf<float> d(x.size());
      HIP_CHECK(hipMemcpy(d, x.data(), x.size() * 4, hipMemcpyHostToDevice));
      k_hadamard_rows(d, rows, n, ld, tr == 1, S);
      std::vector<float> y(x.size());
      HIP_CHECK(hipMemcpy(y.data(), d, y.size() * 4, hipMemcpyDeviceToHost));
      double md = 0;
      for (long i = 0; i < rows; ++i)
        for (int j = 0; j < n; ++j) md = std::max(md, std::fabs(y[size_t(i) * ld + j] - ref[size_t(i) * n + j]));
      std::printf("  n=%d %s max|diff|=%.3g\n", n, tr ? "hadUt" : "hadU", md);
      EXPECT(md < 2e-5, "hadamard cuda n=%d", n);
    }
  }
}

// ---- Viterbi: GPU two-pass tail-biting quantization vs quantize_seq_ref (fp64) ---------------------
static void gpu_quantize_seqs(const std::vector<float>& X, int B, std::vector<uint16_t>& st,
                              std::vector<float>& hat) {
  const uint32_t* lut = device_lut();
  DBuf<float> dX(X.size()), dHat(size_t(B) * 256);
  DBuf<uint16_t> A(size_t(B) * 127 * 1024), fs(static_cast<size_t>(B)), dst(size_t(B) * 128);
  DBuf<int> ov(static_cast<size_t>(B));
  HIP_CHECK(hipMemcpy(dX, X.data(), X.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  k_viterbi_fwd(dX, nullptr, 1, B, A, fs, lut, S);  // rolled pass (overlap-free)
  k_viterbi_overlap(A, fs, B, ov, S);
  k_viterbi_fwd(dX, ov, 0, B, A, fs, lut, S);  // constrained pass
  k_viterbi_finish(A, fs, B, 0, dst, 1, 0, nullptr, 0, 0, lut, S);
  HIP_CHECK(hipStreamSynchronize(S));
  k_decode_states(dst, B, dHat, lut, S);
  st.resize(size_t(B) * 128);
  hat.resize(size_t(B) * 256);
  HIP_CHECK(hipMemcpy(st.data(), dst, st.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hat.data(), dHat, hat.size() * 4, hipMemcpyDeviceToHost));
}

static void test_viterbi(int bench) {
  std::printf("[viterbi cuda] vs fp64 reference tail-biting Viterbi\n");
  int B = 32;
  Rng r(11);
  std::vector<float> X(size_t(B) * 256);
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < 256; ++i) {
      double v = r.normal() * (b % 3 == 0 ? 0.6 : b % 3 == 1 ? 1.0 : 1.6);
      X[size_t(b) * 256 + i] = round_fp16(float(v));
    }
  std::vector<uint16_t> st, st2;
  std::vector<float> hat;
  gpu_quantize_seqs(X, B, st, hat);
  gpu_quantize_seqs(X, B, st2, hat);  // determinism: identical runs must give identical states
  EXPECT(st == st2, "viterbi not deterministic across runs");
  int same = 0;
  double worst = -1e9, mean = 0;
  for (int b = 0; b < B; ++b) {
    uint16_t rs[kSteps];
    float rh[kSeq];
    double cr = quantize_seq_ref(&X[size_t(b) * 256], rs, rh);
    const uint16_t* gs = &st[size_t(b) * 128];
    double cg = path_cost(&X[size_t(b) * 256], gs);
    same += std::equal(rs, rs + kSteps, gs);
    double rel = (cg - cr) / cr;
    worst = std::max(worst, rel);
    mean += rel / B;
    EXPECT((gs[127] & 0xfff) == (gs[0] >> 4), "tail-biting b=%d", b);
    for (int t = 1; t < kSteps; ++t) EXPECT(((gs[t - 1] << 4) & 0xffff) == (gs[t] & 0xfff0), "trellis link b=%d t=%d",
                                            b, t);
    float dh[kSeq];
    decode_states(gs, dh);
    for (int i = 0; i < kSeq; ++i) EXPECT(dh[i] == hat[size_t(b) * 256 + i], "decode(encode) b=%d i=%d", b, i);
  }
  std::printf("  %d/%d identical paths to fp64 reference; cost excess vs fp64: mean %.2e, worst %.2e\n", same, B,
              mean, worst);
  EXPECT(worst < 2e-3, "cuda viterbi cost too far from optimum");
  // packing round trip of the GPU states
  std::vector<uint16_t> pk(size_t(B) * 32), un(st.size());
  pack_trellis(st.data(), B, pk.data());
  unpack_trellis(pk.data(), B, un.data());
  EXPECT(un == st, "pack/unpack round trip of GPU states");
  if (bench > 0) {
    std::vector<float> Xb(size_t(bench) * 256);
    for (auto& v : Xb) v = round_fp16(float(r.normal()));
    double t0 = now_s();
    gpu_quantize_seqs(Xb, bench, st, hat);
    gpu_quantize_seqs(Xb, bench, st, hat);
    double secs = now_s() - t0;
    std::printf("  bench: %d sequences x 2 quantizations in %.3f s -> %.1f us per sequence, %.2f Mweight/s\n", bench,
                secs, secs / (2.0 * bench) * 1e6, 2.0 * bench * 256.0 / secs / 1e6);
  }
}

int main(int argc, char** argv) {
  int dev = 0, bench = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc) dev = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bench") && i + 1 < argc) bench = std::atoi(argv[++i]);
  }
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::puts("no CUDA device; skipped");
    return 77;
  }
  HIP_CHECK(hipSetDevice(dev));
  HIP_CHECK(hipStreamCreate(&S));
  test_compact_codebook();
  test_hadamard();
  test_viterbi(bench);
  HIP_CHECK(hipStreamDestroy(S));
  std::printf(g_fail ? "CUDA TESTS FAILED (%d)\n" : "ALL CUDA TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
