// CPU unit tests vs reference vectors produced by tools/ref_vectors.py (unmodified QTIP python code).
// usage: qtip_tests_cpu <refvec_dir>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "qtip_core.hpp"

using namespace qtip;
static std::string D;
static int g_fail = 0;
#define EXPECT(c, ...)                         \
  do {                                         \
    if (!(c)) {                                \
      std::printf("  FAIL: " __VA_ARGS__);     \
      std::printf("\n");                       \
      ++g_fail;                                \
    }                                          \
  } while (0)

template <class T>
static std::vector<T> load(const std::string& name) {
  std::ifstream f(D + "/" + name, std::ios::binary | std::ios::ate);
  QCHECK(f, "missing " + name);
  size_t sz = size_t(f.tellg());
  std::vector<T> v(sz / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(sz));
  return v;
}

static void test_hadamard() {
  std::printf("[hadamard] matmul_hadU / matmul_hadUt vs reference\n");
  for (int n : {96, 1024, 12288, 28672}) {
    auto x = load<double>("had_" + std::to_string(n) + "_x.bin");
    for (int t = 0; t < 2; ++t) {
      auto ref = load<double>("had_" + std::to_string(n) + (t ? "_ut.bin" : "_u.bin"));
      std::vector<double> y = x;
      hadamard_rows_ref(y.data(), 4, n, n, t == 1);
      double md = 0;
      for (size_t i = 0; i < y.size(); ++i) md = std::max(md, std::fabs(y[i] - ref[i]));
      std::printf("  n=%d %s K=%d max|diff|=%.3g\n", n, t ? "hadUt" : "hadU", had_K(n), md);
      EXPECT(md < 1e-6, "hadamard n=%d", n);  // reference divides by fp32 sqrt(n)
    }
  }
}

static void test_lut_perm() {
  std::printf("[codebook] lut (quantlut_sym of tlut, fp16) and _PERMUTE vs reference\n");
  auto lut = load<float>("lut.bin");
  const Codebook& cb = codebook();
  int bad = 0;
  for (int s = 0; s < kStates; ++s) bad += (lut[s] != cb.lut0[s]) + (lut[kStates + s] != cb.lut1[s]);
  std::printf("  lut mismatches: %d, lut_rms=%.9f\n", bad, cb.lut_rms);
  EXPECT(bad == 0, "lut");
  auto perm = load<int64_t>("perm.bin");
  bad = 0;
  for (int i = 0; i < 256; ++i) bad += perm[i] != perm256()[i];
  EXPECT(bad == 0, "perm");
}

static void test_viterbi() {
  std::printf("[viterbi] reference fp64 tail-biting Viterbi vs python bitshift_codebook.quantize\n");
  auto X = load<float>("vit_x.bin");
  auto st = load<int32_t>("vit_states.bin");
  auto hat = load<float>("vit_hat.bin");
  int B = int(X.size() / 256), same = 0;
  double worst = 0;
  for (int b = 0; b < B; ++b) {
    uint16_t s[kSteps];
    float h[kSeq];
    double c = quantize_seq_ref(&X[b * 256], s, h);
    uint16_t rs[kSteps];
    for (int i = 0; i < kSteps; ++i) rs[i] = uint16_t(st[b * kSteps + i]);
    double cr = path_cost(&X[b * 256], rs);
    bool eq = true;
    for (int i = 0; i < kSteps; ++i) eq = eq && rs[i] == s[i];
    same += eq;
    worst = std::max(worst, (c - cr) / cr);
    // decode(encode(x)) must equal the reconstructed values exactly; python hat must decode from its states
    float hr[kSeq];
    decode_states(rs, hr);
    for (int i = 0; i < kSeq; ++i) EXPECT(hr[i] == hat[b * 256 + i], "python hat vs decode(states) b=%d", b);
    float h2[kSeq];
    decode_states(s, h2);
    for (int i = 0; i < kSeq; ++i) EXPECT(h2[i] == h[i], "decode(encode) b=%d", b);
    // tail biting: last state's low 12 bits == first state's high 12 bits
    EXPECT((s[kSteps - 1] & 0xfff) == (s[0] >> 4), "tail-biting b=%d", b);
  }
  std::printf("  %d/%d identical state paths; max relative cost excess vs python = %.3g (<=0 means ours <= ref)\n",
              same, B, worst);
  EXPECT(worst < 1e-5, "viterbi cost worse than reference");
}

static void test_packing() {
  std::printf("[packing] pack_trellis + kernel swizzle vs reference, round trips\n");
  int m = 64, n = 96;
  auto st32 = load<int32_t>("pack_states.bin");
  auto plain = load<uint16_t>("pack_plain.bin");
  auto kern = load<uint16_t>("pack_kernel.bin");
  long T = long(st32.size() / kSteps);
  std::vector<uint16_t> st(st32.begin(), st32.end()), pk(size_t(T) * 32), sw(pk.size()), back(pk.size()),
      un(st.size());
  pack_trellis(st.data(), T, pk.data());
  int bad = 0;
  for (size_t i = 0; i < pk.size(); ++i) bad += pk[i] != plain[i];
  EXPECT(bad == 0, "pack_trellis (%d words differ)", bad);
  swizzle_for_kernel(pk.data(), m, n, sw.data());
  bad = 0;
  for (size_t i = 0; i < sw.size(); ++i) bad += sw[i] != kern[i];
  EXPECT(bad == 0, "kernel swizzle (%d words differ)", bad);
  unswizzle_for_kernel(sw.data(), m, n, back.data());
  unpack_trellis(back.data(), T, un.data());
  bad = 0;
  for (size_t i = 0; i < un.size(); ++i) bad += un[i] != st[i];
  EXPECT(bad == 0, "unpack(unswizzle(swizzle(pack))) (%d differ)", bad);
  std::printf("  %ld trellises checked\n", T);
}

static void test_ldl() {
  std::printf("[ldl] fp64 blocked Cholesky + block LDL vs reference block_LDL (and vs naive for n=1000)\n");
  auto H = load<double>("ldl_H.bin");
  auto Lr = load<double>("ldl_L.bin");
  int n = int(std::lround(std::sqrt(double(H.size()))));
  std::vector<float> L(size_t(n) * n);
  std::vector<double> A = H;
  EXPECT(block_ldl_to_L(A.data(), n, L.data(), 4), "ldl failed");
  double md = 0;
  for (size_t i = 0; i < L.size(); ++i) md = std::max(md, std::fabs(double(L[i]) - Lr[i]));
  std::printf("  n=%d max|L - L_ref| = %.3g\n", n, md);
  EXPECT(md < 1e-5, "ldl vs reference");
  // larger: blocked cholesky vs naive fp64 (exercises the panel/microkernel path, n not multiple of 256)
  int N = 1000;
  Rng r(3);
  std::vector<double> G(size_t(N) * 300), S(size_t(N) * N);
  for (auto& g : G) g = r.normal();
  for (int i = 0; i < N; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = 0;
      for (int p = 0; p < 300; ++p) s += G[size_t(i) * 300 + p] * G[size_t(j) * 300 + p];
      S[size_t(i) * N + j] = S[size_t(j) * N + i] = s / 300 + (i == j ? 0.05 : 0);
    }
  std::vector<double> C = S, Nv = S;
  double t0 = now_s();
  EXPECT(cholesky_inplace(C.data(), N, 8), "chol");
  double t1 = now_s();
  for (int j = 0; j < N; ++j) {  // naive
    double d = Nv[size_t(j) * N + j];
    for (int p = 0; p < j; ++p) d -= Nv[size_t(j) * N + p] * Nv[size_t(j) * N + p];
    d = std::sqrt(d);
    Nv[size_t(j) * N + j] = d;
    for (int i = j + 1; i < N; ++i) {
      double s = Nv[size_t(i) * N + j];
      for (int p = 0; p < j; ++p) s -= Nv[size_t(i) * N + p] * Nv[size_t(j) * N + p];
      Nv[size_t(i) * N + j] = s / d;
    }
  }
  md = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j <= i; ++j) md = std::max(md, std::fabs(C[size_t(i) * N + j] - Nv[size_t(i) * N + j]));
  std::printf("  n=%d blocked vs naive max|diff| = %.3g (%.3fs)\n", N, md, t1 - t0);
  EXPECT(md < 1e-10, "blocked cholesky");
}

int main(int argc, char** argv) {
  QCHECK(argc >= 2, "usage: qtip_tests_cpu <refvec_dir>");
  D = argv[1];
  test_hadamard();
  test_lut_perm();
  test_viterbi();
  test_packing();
  test_ldl();
  std::printf(g_fail ? "CPU TESTS FAILED (%d)\n" : "ALL CPU TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
