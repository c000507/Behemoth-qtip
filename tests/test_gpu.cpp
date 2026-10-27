// GPU unit tests (small shapes; fits in ~1 GB of VRAM).  usage: qtip_tests_gpu [--device N] [--bench B]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "gpu.hpp"
#include "qtip_core.hpp"
#include "quantizer.hpp"
#include "qwen35.hpp"
#include "qwen_runner.hpp"

using namespace qtip;
static int g_fail = 0;
#define EXPECT(c, ...)                     \
  do {                                     \
    if (!(c)) {                            \
      std::printf("  FAIL: " __VA_ARGS__); \
      std::printf("\n");                   \
      ++g_fail;                            \
    }                                      \
  } while (0)

static hipStream_t S;

static void test_hadamard() {
  std::printf("[hadamard gpu] vs fp64 reference (hadU / hadUt, strided rows)\n");
  for (int n : {96, 1024, 12288, 28672}) {
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
      EXPECT(md < 2e-5, "hadamard gpu n=%d", n);
    }
  }
  // transpose
  long R = 100, C = 70;
  std::vector<float> a(size_t(R) * C);
  for (size_t i = 0; i < a.size(); ++i) a[i] = float(i);
  DBuf<float> da(a.size()), db(a.size());
  HIP_CHECK(hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice));
  k_transpose(da, R, C, C, db, R, S);
  std::vector<float> b(a.size());
  HIP_CHECK(hipMemcpy(b.data(), db, b.size() * 4, hipMemcpyDeviceToHost));
  int bad = 0;
  for (long i = 0; i < R; ++i)
    for (long j = 0; j < C; ++j) bad += b[size_t(j) * R + i] != a[size_t(i) * C + j];
  EXPECT(bad == 0, "transpose");
}

// GPU two-pass tail-biting quantization of B sequences
static void gpu_quantize_seqs(const std::vector<float>& X, int B, std::vector<uint16_t>& st, std::vector<float>& hat,
                              double* secs) {
  const uint32_t* lut = device_lut();
  DBuf<float> dX(X.size()), dHat(size_t(B) * 256);
  DBuf<uint16_t> A(size_t(B) * 127 * 1024), fs(static_cast<size_t>(B)), dst(size_t(B) * 128);
  DBuf<int> ov(static_cast<size_t>(B));
  HIP_CHECK(hipMemcpy(dX, X.data(), X.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  double t0 = now_s();
  k_viterbi_fwd(dX, nullptr, 1, B, A, fs, lut, S);
  k_viterbi_overlap(A, fs, B, ov, S);
  k_viterbi_fwd(dX, ov, 0, B, A, fs, lut, S);
  // states only (no hatWT scatter): nblk=1, blk=0
  k_viterbi_finish(A, fs, B, 0, dst, 1, 0, nullptr, 0, 0, lut, S);
  HIP_CHECK(hipStreamSynchronize(S));
  if (secs) *secs = now_s() - t0;
  k_decode_states(dst, B, dHat, lut, S);
  st.resize(size_t(B) * 128);
  hat.resize(size_t(B) * 256);
  HIP_CHECK(hipMemcpy(st.data(), dst, st.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hat.data(), dHat, hat.size() * 4, hipMemcpyDeviceToHost));
}

static void test_viterbi(int bench) {
  std::printf("[viterbi gpu] vs fp64 reference tail-biting Viterbi\n");
  int B = 48;
  Rng r(11);
  std::vector<float> X(size_t(B) * 256);
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < 256; ++i) {
      double v = r.normal() * (b % 3 == 0 ? 0.6 : b % 3 == 1 ? 1.0 : 1.6);
      X[size_t(b) * 256 + i] = round_fp16(float(v));
    }
  std::vector<uint16_t> st;
  std::vector<float> hat;
  gpu_quantize_seqs(X, B, st, hat, nullptr);
  int same = 0;
  double worst = -1e9, mean = 0;
  for (int b = 0; b < B; ++b) {
    uint16_t rs[kSteps];
    float rh[kSeq];
    double cr = quantize_seq_ref(&X[size_t(b) * 256], rs, rh);
    const uint16_t* gs = &st[size_t(b) * 128];
    double cg = path_cost(&X[size_t(b) * 256], gs);
    bool eq = std::equal(rs, rs + kSteps, gs);
    same += eq;
    double rel = (cg - cr) / cr;
    worst = std::max(worst, rel);
    mean += rel / B;
    EXPECT((gs[127] & 0xfff) == (gs[0] >> 4), "tail-biting b=%d", b);
    for (int t = 1; t < kSteps; ++t) EXPECT(((gs[t - 1] << 4) & 0xffff) == (gs[t] & 0xfff0), "trellis link b=%d", b);
    float dh[kSeq];
    decode_states(gs, dh);
    for (int i = 0; i < kSeq; ++i) EXPECT(dh[i] == hat[size_t(b) * 256 + i], "decode(encode) b=%d i=%d", b, i);
  }
  std::printf("  %d/%d identical paths to fp64 reference; cost excess vs fp64: mean %.2e, worst %.2e\n", same, B,
              mean, worst);
  EXPECT(worst < 2e-3, "gpu viterbi cost too far from optimum");
  // packing round trip of GPU states
  std::vector<uint16_t> pk(size_t(B) * 32), un(st.size());
  pack_trellis(st.data(), B, pk.data());
  unpack_trellis(pk.data(), B, un.data());
  EXPECT(un == st, "pack/unpack round trip of GPU states");
  if (bench > 0) {
    std::vector<float> Xb(size_t(bench) * 256);
    for (auto& v : Xb) v = round_fp16(float(r.normal()));
    double secs = 0;
    gpu_quantize_seqs(Xb, bench, st, hat, &secs);  // warm
    gpu_quantize_seqs(Xb, bench, st, hat, &secs);
    std::printf("  bench: %d sequences x 2 passes in %.3f s -> %.1f us per sequence-pass, %.2f Mweight/s\n", bench,
                secs, secs / (2.0 * bench) * 1e6, bench * 256.0 / secs / 1e6);
  }
}

static void test_gemm() {
  std::printf("[gemm] hipBLASLt bf16 layouts used by the code vs CPU\n");
  Gemm g(S);
  int M = 96, N = 80, K = 64;
  Rng r(5);
  auto rnd = [&](size_t n) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = f32_to_bf16(float(r.normal()));
    return v;
  };
  for (int ta = 0; ta < 2; ++ta)
    for (int tb = 0; tb < 2; ++tb) {
      long lda = ta ? K : M, ldb = tb ? N : K;
      auto A = rnd(size_t(lda) * (ta ? M : K)), Bm = rnd(size_t(ldb) * (tb ? K : N));
      DBuf<uint16_t> dA(A.size()), dB(Bm.size());
      DBuf<float> dC(size_t(M) * N);
      HIP_CHECK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dB, Bm.data(), Bm.size() * 2, hipMemcpyHostToDevice));
      std::vector<float> C0(size_t(M) * N, 0.5f);
      HIP_CHECK(hipMemcpy(dC, C0.data(), C0.size() * 4, hipMemcpyHostToDevice));
      g.run(ta, tb, M, N, K, dA, lda, dB, ldb, dC, M, OutT::F32, 1.f, 1.f);
      std::vector<float> C(C0.size());
      HIP_CHECK(hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));
      double md = 0;
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
          double s = 0.5;
          for (int k = 0; k < K; ++k) {
            float a = bf16_to_f32(ta ? A[size_t(i) * lda + k] : A[size_t(k) * lda + i]);
            float b = bf16_to_f32(tb ? Bm[size_t(k) * ldb + j] : Bm[size_t(j) * ldb + k]);
            s += double(a) * b;
          }
          md = std::max(md, std::fabs(s - C[size_t(j) * M + i]));
        }
      std::printf("  trans=%d%d max|diff|=%.3g\n", ta, tb, md);
      EXPECT(md < 1e-3, "gemm layout %d%d", ta, tb);
    }
}

// CPU port of lib/algo/ldlq.py (buf_cols=128, for_kernel=True) on top of quantize_seq_ref
static void ldlq_ref(const std::vector<float>& Wr /*m x n*/, const std::vector<float>& L /*n x n*/, int m, int n,
                     std::vector<float>& hatWr, std::vector<uint16_t>& Q /*[(m/16)*(n/16)][128]*/) {
  const int* perm = perm256();
  std::vector<double> WT(size_t(n) * m), hT(size_t(n) * m, 0.0), prod(size_t(n) * m, 0.0);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) WT[size_t(j) * m + i] = Wr[size_t(i) * n + j];
  Q.assign(size_t(m / 16) * (n / 16) * 128, 0);
  for (int cur = n / 16; cur > 0; cur -= 8) {
    int bufend = 16 * cur, c0 = bufend - 128;
    for (int bi = 7; bi >= 0; --bi) {
      int col0 = c0 + 16 * bi;
      std::vector<double> WX(size_t(16) * m);
      for (int c = 0; c < 16; ++c)
        for (int j = 0; j < m; ++j) {
          double v = WT[size_t(col0 + c) * m + j] + prod[size_t(col0 + c) * m + j];
          for (int k = col0 + 16; k < bufend; ++k)
            v += double(L[size_t(k) * n + col0 + c]) * (WT[size_t(k) * m + j] - hT[size_t(k) * m + j]);
          WX[size_t(c) * m + j] = v;
        }
      for (int b = 0; b < m / 16; ++b) {
        float x[256], h[256];
        uint16_t st[128];
        for (int p = 0; p < 256; ++p) {
          int q = perm[p], r = q >> 4, c = q & 15;
          x[p] = round_fp16(float(WX[size_t(c) * m + 16 * b + r]));  // reference quantizes in fp16
        }
        quantize_seq_ref(x, st, h);
        for (int p = 0; p < 256; ++p) {
          int q = perm[p], r = q >> 4, c = q & 15;
          hT[size_t(col0 + c) * m + 16 * b + r] = h[p];
        }
        std::copy(st, st + 128, &Q[(size_t(b) * (n / 16) + col0 / 16) * 128]);
      }
    }
    for (int r = 0; r < c0; ++r)
      for (int k = c0; k < bufend; ++k) {
        double l = L[size_t(k) * n + r];
        if (l == 0) continue;
        for (int j = 0; j < m; ++j) prod[size_t(r) * m + j] += l * (WT[size_t(k) * m + j] - hT[size_t(k) * m + j]);
      }
  }
  hatWr.assign(size_t(m) * n, 0.f);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) hatWr[size_t(i) * n + j] = float(hT[size_t(j) * m + i]);
}

static void test_ldlq() {
  std::printf("[ldlq] GPU LDLQ+Viterbi vs CPU port of ldlq.LDLQ (m=64, n=256)\n");
  int m = 64, n = 256;
  Rng r(21);
  std::vector<float> Wr(size_t(m) * n);
  for (auto& v : Wr) v = float(r.normal());
  // correlated SPD H -> L
  std::vector<double> G(size_t(n) * 2 * n), H(size_t(n) * n);
  for (auto& g : G) g = r.normal();
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = 0;
      for (int p = 0; p < 2 * n; ++p) s += G[size_t(i) * 2 * n + p] * G[size_t(j) * 2 * n + p] * (1.0 + (p % 7));
      H[size_t(i) * n + j] = H[size_t(j) * n + i] = s / (2 * n) + (i == j ? 0.05 : 0.0);
    }
  std::vector<double> A = H;
  std::vector<float> L(size_t(n) * n);
  EXPECT(block_ldl_to_L(A.data(), n, L.data(), 4), "ldl");
  std::vector<float> hr;
  std::vector<uint16_t> qr;
  double t0 = now_s();
  ldlq_ref(Wr, L, m, n, hr, qr);
  double tref = now_s() - t0;
  // GPU
  Gemm g(S);
  std::vector<float> WT(size_t(n) * m);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) WT[size_t(j) * m + i] = Wr[size_t(i) * n + j];
  DBuf<float> dWT(WT.size()), dL(L.size()), dH(WT.size());
  DBuf<uint16_t> dQ(qr.size());
  HIP_CHECK(hipMemcpy(dWT, WT.data(), WT.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dL, L.data(), L.size() * 4, hipMemcpyHostToDevice));
  ldlq_gpu(S, g, dWT, dL, n, m, dH, dQ, 1024, device_lut(), nullptr);
  std::vector<float> hT(WT.size());
  std::vector<uint16_t> qg(qr.size());
  HIP_CHECK(hipMemcpy(hT.data(), dH, hT.size() * 4, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qg.data(), dQ, qg.size() * 2, hipMemcpyDeviceToHost));
  auto proxy = [&](const std::vector<float>& hat_mn) {
    double num = 0, den = 0;
    for (int i = 0; i < m; ++i) {
      std::vector<double> d(static_cast<size_t>(n)), w(static_cast<size_t>(n));
      for (int j = 0; j < n; ++j) d[size_t(j)] = Wr[size_t(i) * n + j] - hat_mn[size_t(i) * n + j], w[size_t(j)] = Wr[size_t(i) * n + j];
      for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
          num += d[size_t(a)] * H[size_t(a) * n + b] * d[size_t(b)];
          den += w[size_t(a)] * H[size_t(a) * n + b] * w[size_t(b)];
        }
    }
    return num / den;
  };
  std::vector<float> hg(size_t(m) * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) hg[size_t(i) * n + j] = hT[size_t(j) * m + i];
  double pr = proxy(hr), pg = proxy(hg);
  // trellis consistency: decoded states == hatW, per trellis
  int bad = 0;
  const int* perm = perm256();
  for (int b = 0; b < m / 16; ++b)
    for (int blk = 0; blk < n / 16; ++blk) {
      float h[256];
      decode_states(&qg[(size_t(b) * (n / 16) + blk) * 128], h);
      for (int p = 0; p < 256; ++p) {
        int q = perm[p], rr = q >> 4, c = q & 15;
        bad += h[p] != hg[size_t(16 * b + rr) * n + blk * 16 + c];
      }
    }
  size_t same = 0;
  for (size_t i = 0; i < qr.size(); i += 128) same += std::equal(&qr[i], &qr[i] + 128, &qg[i]);
  std::printf("  proxy ref %.6f gpu %.6f (rel %.2e); identical trellises %zu/%zu; decode mismatches %d (ref %.1fs)\n",
              pr, pg, (pg - pr) / pr, same, qr.size() / 128, bad, tref);
  EXPECT(std::fabs(pg - pr) / pr < 0.01, "ldlq proxy differs >1%% from reference port");
  EXPECT(bad == 0, "decode(states) != hatW");
}

static void test_group() {
  std::printf("[quantize_group] end to end on a synthetic 2-linear group (n=256)\n");
  Gemm g(S);
  int n = 256;
  Rng r(33);
  std::vector<float> W1(size_t(64) * n), W2(size_t(32) * n), H(size_t(n) * n);
  for (auto& v : W1) v = float(r.normal() * 0.02);
  for (auto& v : W2) v = float(r.normal() * 0.05);
  std::vector<double> G(size_t(n) * 3 * n);
  for (auto& x : G) x = r.normal();
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = 0;
      for (int p = 0; p < 3 * n; ++p) s += G[size_t(i) * 3 * n + p] * G[size_t(j) * 3 * n + p];
      H[size_t(i) * n + j] = H[size_t(j) * n + i] = float(s / (3 * n));
    }
  std::vector<LinearIn> lins(2);
  lins[0].name = "a", lins[0].m = 64, lins[0].w_f32 = W1.data();
  lins[1].name = "b", lins[1].m = 32, lins[1].w_f32 = W2.data();
  QuantOpts o;
  GroupTiming tm;
  auto out = quantize_group(S, g, lins, n, H.data(), 7, o, &tm, "test");
  for (auto& lo : out) {
    std::printf("  %s: proxy %.5f proxy_rot_reg %.5f Wscale %.5f trellis words %zu\n", lo.name.c_str(), lo.proxy,
                lo.proxy_rot_reg, lo.wscale, lo.trellis.size());
    EXPECT(lo.proxy > 0 && lo.proxy < 0.2, "implausible proxy");
    EXPECT(lo.trellis.size() == size_t(lo.m / 16) * (n / 16) * 32, "trellis size");
  }
  // CPU check of the proxy: decode trellis -> hatWr -> original domain, proxy with unregularized H
  for (size_t li = 0; li < out.size(); ++li) {
    const LinearOut& lo = out[li];
    int m = lo.m;
    std::vector<uint16_t> pk(lo.trellis.size()), st(size_t(m / 16) * (n / 16) * 128);
    unswizzle_for_kernel(lo.trellis.data(), m, n, pk.data());
    unpack_trellis(pk.data(), long(st.size() / 128), st.data());
    std::vector<double> hw(size_t(m) * n);
    const int* perm = perm256();
    for (int b = 0; b < m / 16; ++b)
      for (int blk = 0; blk < n / 16; ++blk) {
        float h[256];
        decode_states(&st[(size_t(b) * (n / 16) + blk) * 128], h);
        for (int p = 0; p < 256; ++p) {
          int q = perm[p];
          hw[size_t(16 * b + (q >> 4)) * n + blk * 16 + (q & 15)] = double(h[p]) * lo.wscale;
        }
      }
    // hatW = diag(SV) T_m hatWr T_n^T diag(SU): rows hadU (n), then columns hadU (m)
    hadamard_rows_ref(hw.data(), m, n, n, false);
    std::vector<double> tcol(size_t(n) * m);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) tcol[size_t(j) * m + i] = hw[size_t(i) * n + j];
    hadamard_rows_ref(tcol.data(), n, m, m, false);
    const std::vector<float>& W = li == 0 ? W1 : W2;
    double num = 0, den = 0;
    for (int i = 0; i < m; ++i) {
      std::vector<double> d(static_cast<size_t>(n)), w(static_cast<size_t>(n));
      for (int j = 0; j < n; ++j) {
        double hv = tcol[size_t(j) * m + i] * lo.sv[size_t(i)] * bf16_to_f32(lo.su_bf16[size_t(j)]);
        d[size_t(j)] = W[size_t(i) * n + j] - hv;
        w[size_t(j)] = W[size_t(i) * n + j];
      }
      for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
          num += d[size_t(a)] * H[size_t(a) * n + b] * d[size_t(b)];
          den += w[size_t(a)] * H[size_t(a) * n + b] * w[size_t(b)];
        }
    }
    std::printf("  %s: CPU decode of packed output -> proxy %.5f (binary reported %.5f)\n", lo.name.c_str(),
                num / den, lo.proxy);
    EXPECT(std::fabs(num / den - lo.proxy) / lo.proxy < 1e-3, "decoded proxy mismatch");
  }
}

// timing at a real shape with a synthetic Hessian H = X^T X / N (X: N x n correlated bf16) and a random weight
static void shape_bench(int m, int n, int nsamp) {
  std::printf("[shape bench] m=%d n=%d (synthetic H from %d samples)\n", m, n, nsamp);
  Gemm g(S);
  Rng r(77);
  std::vector<uint16_t> X(size_t(nsamp) * n);
  std::vector<double> scale(static_cast<size_t>(n));
  for (auto& v : scale) v = std::exp(r.normal());
  for (long i = 0; i < nsamp; ++i) {
    double common = r.normal();
    for (int j = 0; j < n; ++j) X[size_t(i) * n + j] = f32_to_bf16(float((r.normal() + 0.5 * common) * scale[size_t(j)]));
  }
  DBuf<uint16_t> dX(X.size());
  DBuf<float> dH(size_t(n) * n);
  HIP_CHECK(hipMemcpy(dX, X.data(), X.size() * 2, hipMemcpyHostToDevice));
  g.run(false, true, n, n, nsamp, dX, n, dX, n, dH, n, OutT::F32, 1.f / nsamp, 0.f);
  std::vector<float> H(size_t(n) * n);
  HIP_CHECK(hipMemcpy(H.data(), dH, H.size() * 4, hipMemcpyDeviceToHost));
  dX.free();
  dH.free();
  std::vector<float> W(size_t(m) * n);
  for (auto& v : W) v = float(r.normal() * 0.01 * (1 + 0.5 * std::fabs(r.normal())));
  std::vector<LinearIn> lins(1);
  lins[0].name = "w", lins[0].m = m, lins[0].w_f32 = W.data();
  QuantOpts o;
  GroupTiming tm;
  double t0 = now_s();
  auto out = quantize_group(S, g, lins, n, H.data(), 3, o, &tm, "bench");
  double t = now_s() - t0;
  std::printf("  wall %.2fs: rht %.2f ldl_cpu %.2f ldlq %.2f (viterbi %.2f) proxy %.2f pack %.2f load %.2f\n", t, tm.rht,
              tm.ldl_cpu, tm.ldlq, tm.viterbi, tm.proxy, tm.pack, tm.load);
  std::printf("  proxy %.6f proxy_rot_reg %.6f; %.3f s/Mweight (ldlq+viterbi %.3f s/Mweight)\n", out[0].proxy,
              out[0].proxy_rot_reg, t / (double(m) * n / 1e6), tm.ldlq / (double(m) * n / 1e6));
}

// Qwen3.5 layer forward parity: QwenLayerRunner (GPU, bf16) vs qwen_cpu_layer (fp32). Runs when
// QWEN35_MODEL points at a dense Qwen3.5 checkpoint (e.g. Qwen3.5-0.8B); tests layers 0 (linear) and
// the first full-attention layer.
static void test_qwen_layer(const std::string& model_dir, int dev) {
  std::printf("[qwen35] layer forward parity vs CPU (%s)\n", model_dir.c_str());
  QwenConfig cfg = load_qwen_config(model_dir);
  QCHECK(!cfg.moe(), "MoE checkpoint not supported by the parity test");
  SafeTensorsReader st(model_dir);
  const int T = 64;
  std::vector<int> layers = {0};
  for (int L = 0; L < cfg.layers; ++L)
    if (cfg.is_full(L)) {
      layers.push_back(L);
      break;
    }
  for (int L : layers) {
    auto ql = read_qwen_layer(st, cfg, L);
    // input: bf16 activations around token 0's embedding + small noise
    std::vector<uint16_t> xb(size_t(T) * cfg.hidden);
    {
      Rng r(1234 + L);
      std::string embed_name;
      for (auto& kv : st.tensors())
        if (kv.first.size() > 19 && kv.first.substr(kv.first.size() - 19) == "embed_tokens.weight") embed_name = kv.first;
      const TensorInfo& e = st.info(embed_name);
      std::vector<uint16_t> emb_row(size_t(e.shape[1]));
      st.read_into(embed_name, emb_row.data(), emb_row.size() * 2);  // token 0 embedding
      for (int t = 0; t < T; ++t) {
        float noise = float(r.normal()) * 0.05f;
        for (int i = 0; i < cfg.hidden; ++i) {
          float v = bf16_to_f32(emb_row[size_t(i)]) + noise * float(r.normal());
          xb[size_t(t) * cfg.hidden + size_t(i)] = f32_to_bf16(v);
        }
      }
    }
    // CPU fp32 reference
    std::vector<float> xc(size_t(T) * cfg.hidden);
    for (size_t i = 0; i < xc.size(); ++i) xc[i] = bf16_to_f32(xb[i]);
    qwen_cpu_layer(cfg, *ql, xc.data(), T, 32);
    // GPU forward (bf16 in/out)
    std::vector<uint16_t> xg = xb;
    {
      QwenLayerRunner R(dev, cfg, T, T);
      R.upload(*ql);
      R.forward(xg.data(), 1, nullptr);
    }
    double max_abs = 0, denom = 1e-9;
    float gscale = 0;
    for (size_t i = 0; i < xc.size(); ++i) {
      double a = std::fabs(xc[i] - bf16_to_f32(xg[i]));
      max_abs = std::max(max_abs, a);
      denom = std::max(denom, double(std::fabs(xc[i])));
    }
    gscale = float(denom);
    std::printf("  layer %2d (%s): max|diff| %.5f  scale %.3f  rel %.2e  %s\n", L, ql->full ? "full" : "linear",
                max_abs, gscale, max_abs / gscale, max_abs / gscale < 3e-2 ? "ok" : "FAIL");
    EXPECT(max_abs / gscale < 3e-2, "qwen layer %d forward parity", L);
  }
}

int main(int argc, char** argv) {
  int dev = 0, bench = 0, sm = 0, sn = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc) dev = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bench") && i + 1 < argc) bench = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shape") && i + 2 < argc) sm = std::atoi(argv[++i]), sn = std::atoi(argv[++i]);
  }
  HIP_CHECK(hipSetDevice(dev));
  HIP_CHECK(hipStreamCreate(&S));
  if (sm > 0) {
    shape_bench(sm, sn, 4096);
    return 0;
  }
  test_hadamard();
  test_gemm();
  test_viterbi(bench);
  test_ldlq();
  test_group();
  if (const char* m = std::getenv("QWEN35_MODEL")) test_qwen_layer(m, dev);
  std::printf(g_fail ? "GPU TESTS FAILED (%d)\n" : "ALL GPU TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
