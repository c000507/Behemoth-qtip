#include "quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>

#include "qtip_core.hpp"

namespace qtip {

const uint32_t* device_lut() {
  static std::mutex mu;
  static std::map<int, uint32_t*> cache;
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;
  const Codebook& cb = codebook();
  uint32_t* d = nullptr;
  HIP_CHECK(hipMalloc(&d, cb.lut_h2.size() * 4));
  HIP_CHECK(hipMemcpy(d, cb.lut_h2.data(), cb.lut_h2.size() * 4, hipMemcpyHostToDevice));
  cache[dev] = d;
  return d;
}

void ldlq_gpu(hipStream_t s, Gemm& gemm, const float* WT, const float* L, long n, long m, float* hatWT, uint16_t* Q,
              int vit_batch, const uint32_t* lut, double* vit_s) {
  QCHECK(n % 128 == 0, "LDLQ needs n % 128 == 0");
  QCHECK(m % 16 == 0, "LDLQ needs m % 16 == 0");
  const long ntr = m / 16;  // trellises per column block
  const int nblk = int(n / 16);
  const int vb = int(std::min<long>(ntr, vit_batch));
  DBuf<float> prod(size_t(n) * m), X(size_t(ntr) * 256);
  DBuf<uint16_t> A(size_t(vb) * 127 * 1024), fs(static_cast<size_t>(vb));
  DBuf<int> ov(static_cast<size_t>(ntr));
  DBuf<uint16_t> Dcat(size_t(384) * m), Lcat(size_t(384) * n);
  HIP_CHECK(hipMemsetAsync(prod, 0, prod.bytes(), s));
  HIP_CHECK(hipMemsetAsync(hatWT, 0, size_t(n) * m * 4, s));
  std::vector<hipEvent_t> ev;
  auto mark = [&]() {
    hipEvent_t e;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(e, s));
    ev.push_back(e);
  };
  for (long cur = n / 16; cur > 0; cur -= 8) {
    const int bufend = int(16 * cur), c0 = bufend - 128;
    for (int i = 7; i >= 0; --i) {
      const int col0 = c0 + 16 * i;
      k_ldlq_prep(WT, hatWT, prod, L, n, m, col0, bufend, X, s);
      if (vit_s) mark();
      for (long b0 = 0; b0 < ntr; b0 += vb) {
        int nb = int(std::min<long>(vb, ntr - b0));
        const float* xb = X + b0 * 256;
        k_viterbi_fwd(xb, nullptr, 1, nb, A, fs, lut, s);
        k_viterbi_overlap(A, fs, nb, ov + b0, s);
        k_viterbi_fwd(xb, ov + b0, 0, nb, A, fs, lut, s);
        k_viterbi_finish(A, fs, nb, int(b0), Q, nblk, col0 / 16, hatWT, m, col0, lut, s);
      }
      if (vit_s) mark();
    }
    if (c0 > 0) {
      // prod[0:c0, :] += L[c0:c0+128, 0:c0]^T @ (WT - hatWT)[c0:c0+128, :]   (bf16 hi/lo split, K-concat = 384)
      const float* wt = WT + long(c0) * m;
      const float* ht = hatWT + long(c0) * m;
      k_diff_split_bf16(wt, ht, 128, m, m, Dcat, Dcat + 256L * m, m, s);
      HIP_CHECK(hipMemcpyAsync(Dcat + 128L * m, Dcat, size_t(128) * m * 2, hipMemcpyDeviceToDevice, s));
      k_split_bf16(L + long(c0) * n, 128, c0, n, Lcat, Lcat + 128L * c0, c0, s);
      HIP_CHECK(hipMemcpyAsync(Lcat + 256L * c0, Lcat, size_t(128) * c0 * 2, hipMemcpyDeviceToDevice, s));
      // col-major: prod_cm (m x c0) += Dcat_cm (m x 384) * Lcat_cm(c0 x 384)^T
      gemm.run(false, true, int(m), c0, 384, Dcat, m, Lcat, c0, prod, m, OutT::F32, 1.f, 1.f);
    }
  }
  if (vit_s) {
    HIP_CHECK(hipStreamSynchronize(s));
    double t = 0;
    for (size_t i = 0; i + 1 < ev.size(); i += 2) {
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, ev[i], ev[i + 1]));
      t += ms * 1e-3;
    }
    *vit_s = t;
    for (auto e : ev) (void)hipEventDestroy(e);
  }
}

namespace {
// tr(X^T H X) for X = cols [c0, c0+mc) of a row-major (n x ld) matrix: sum(X .* (H X)), split-bf16 GEMMs.
// Hcat: bf16 [Hh | Hl | Hh] each n x n (symmetric). Returns per-call value.
double quad_trace(hipStream_t s, Gemm& gemm, const float* Xr, long n, long ld, long c0, long mc,
                  const uint16_t* Hcat, DBuf<uint16_t>& Acat, DBuf<float>& Y, DBuf<double>& acc) {
  Acat.ensure(size_t(3) * n * mc);
  Y.ensure(size_t(n) * mc);
  // A_cm (mc x 3n) = [Xh | Xh | Xl], each block col-major (mc x n) = row-major (n x mc)
  k_split_bf16(Xr + c0, n, mc, ld, Acat, Acat + 2 * n * mc, mc, s);
  HIP_CHECK(hipMemcpyAsync(Acat + n * mc, Acat, size_t(n) * mc * 2, hipMemcpyDeviceToDevice, s));
  // Y_cm (mc x n) = A_cm (mc x 3n) * op_T(Hcat_cm (n x 3n))
  gemm.run(false, true, int(mc), int(n), int(3 * n), Acat, mc, Hcat, n, Y, mc, OutT::F32, 1.f, 0.f);
  HIP_CHECK(hipMemsetAsync(acc, 0, 8, s));
  k_dot(Xr + c0, ld, Y, mc, n, mc, acc, s);
  double h = 0;
  HIP_CHECK(hipMemcpyAsync(&h, acc, 8, hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));
  return h;
}
double dev_sumsq(hipStream_t s, const float* X, long rows, long cols, long ld, DBuf<double>& acc) {
  HIP_CHECK(hipMemsetAsync(acc, 0, 8, s));
  k_sumsq(X, rows, cols, ld, acc, s);
  double h = 0;
  HIP_CHECK(hipMemcpyAsync(&h, acc, 8, hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));
  return h;
}
void write_f32(const std::string& path, const float* p, size_t n) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(p), std::streamsize(n * 4));
}
}  // namespace

void rotate_hessian(hipStream_t s, const float* H, int n, uint64_t seed, const QuantOpts& opt, HessPrep& hp) {
  double t0 = now_s();
  hp.n = n;
  hp.seed = seed;
  Rng rng(seed);
  hp.su.resize(static_cast<size_t>(n));
  for (auto& v : hp.su) v = rng.sign();
  DBuf<float> d_su(static_cast<size_t>(n));
  HIP_CHECK(hipMemcpyAsync(d_su, hp.su.data(), hp.su.size() * 4, hipMemcpyHostToDevice, s));
  DBuf<double> acc(1);
  DBuf<float> Ht(size_t(n) * n), HR(size_t(n) * n);
  HIP_CHECK(hipMemcpyAsync(Ht, H, size_t(n) * n * 4, hipMemcpyHostToDevice, s));
  HIP_CHECK(hipMemsetAsync(acc, 0, 8, s));
  k_diag_sum(Ht, n, acc, s);
  double dsum = 0;
  HIP_CHECK(hipMemcpyAsync(&dsum, acc, 8, hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));
  // regularize_H: H += sigma_reg * mean(diag(H)) * I
  hp.creg = opt.sigma_reg * dsum / n;
  k_add_diag(Ht, n, float(hp.creg), s);
  // HRr = hadUt(hadUt(HR * SU)^T * SU)
  k_scale_rows_cols(Ht, n, n, n, d_su, d_su, s);
  k_hadamard_rows(Ht, n, n, n, true, s);
  k_transpose(Ht, n, n, n, HR, n, s);
  k_hadamard_rows(HR, n, n, n, true, s);
  hp.HRr.resize(size_t(n) * n);
  HIP_CHECK(hipMemcpyAsync(hp.HRr.data(), HR, hp.HRr.size() * 4, hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));
  hp.t_rot = now_s() - t0;
}

void ldl_hessian(HessPrep& hp, int threads, const std::string& tag) {
  double t0 = now_s();
  const long n = hp.n;
  std::vector<double> A(size_t(n) * n);
  const float* hr = hp.HRr.data();
#pragma omp parallel for schedule(static) num_threads(threads)
  for (long i = 0; i < n; ++i)
    for (long j = 0; j <= i; ++j) A[size_t(i) * n + j] = 0.5 * (double(hr[size_t(i) * n + j]) + hr[size_t(j) * n + i]);
  hp.L.resize(size_t(n) * n);
  QCHECK(block_ldl_to_L(A.data(), int(n), hp.L.data(), threads), "block LDL failed (H not PD) for " + tag);
  hp.t_ldl = now_s() - t0;
}

std::vector<LinearOut> quantize_group(hipStream_t s, Gemm& gemm, const std::vector<LinearIn>& lins, int n,
                                      const float* H, uint64_t seed, const QuantOpts& opt, GroupTiming* tm,
                                      const std::string& tag) {
  HessPrep hp;
  rotate_hessian(s, H, n, seed, opt, hp);
  ldl_hessian(hp, opt.ldl_threads, tag);
  return quantize_group(s, gemm, lins, hp, opt, tm, tag);
}

std::vector<LinearOut> quantize_group(hipStream_t s, Gemm& gemm, const std::vector<LinearIn>& lins, HessPrep& hp,
                                      const QuantOpts& opt, GroupTiming* tm, const std::string& tag) {
  const Codebook& cb = codebook();
  const uint32_t* lut = device_lut();
  const int n = hp.n;
  const uint64_t seed = hp.seed;
  GroupTiming T{};
  T.rht = hp.t_rot;
  T.ldl_cpu = hp.t_ldl;
  double t_start = now_s(), t0 = t_start;
  long mt = 0;
  std::vector<long> off;
  for (auto& l : lins) off.push_back(mt), mt += l.m;
  std::vector<LinearOut> out(lins.size());
  const std::vector<float>& su = hp.su;
  DBuf<float> d_su(static_cast<size_t>(n)), d_sv(static_cast<size_t>(mt));
  HIP_CHECK(hipMemcpyAsync(d_su, su.data(), su.size() * 4, hipMemcpyHostToDevice, s));
  DBuf<double> acc(1);

  // ---- weights -> Wr^T (n x mt), each linear's column block scaled by 1/Wscale_i
  DBuf<float> WT(size_t(n) * mt);
  {
    long maxm = 0;
    for (auto& l : lins) maxm = std::max<long>(maxm, l.m);
    DBuf<float> Wt(size_t(maxm) * n);
    DBuf<uint8_t> w8;
    DBuf<uint16_t> sc;
    for (size_t li = 0; li < lins.size(); ++li) {
      const LinearIn& L = lins[li];
      LinearOut& O = out[li];
      O.name = L.name, O.m = L.m, O.n = n;
      O.sv.resize(size_t(L.m));
      Rng r2(seed * 1000003ull + li + 1);
      if (!opt.share_su && li > 0) {
        QCHECK(false, "share_su=false not supported (one LDL per group)");
      }
      for (auto& v : O.sv) v = r2.sign();
      O.su_bf16.resize(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) O.su_bf16[size_t(i)] = f32_to_bf16(su[size_t(i)]);
      HIP_CHECK(hipMemcpyAsync(d_sv + off[li], O.sv.data(), size_t(L.m) * 4, hipMemcpyHostToDevice, s));
      if (L.w_fp8) {
        w8.ensure(size_t(L.m) * n);
        sc.ensure(size_t(L.m));
        HIP_CHECK(hipMemcpyAsync(w8, L.w_fp8, size_t(L.m) * n, hipMemcpyHostToDevice, s));
        HIP_CHECK(hipMemcpyAsync(sc, L.scale_bf16, size_t(L.m) * 2, hipMemcpyHostToDevice, s));
        k_fp8_to_f32(w8, sc, Wt, L.m, n, s);
      } else {
        HIP_CHECK(hipMemcpyAsync(Wt, L.w_f32, size_t(L.m) * n * 4, hipMemcpyHostToDevice, s));
      }
      if (!opt.dump_dir.empty()) {
        std::vector<float> h(size_t(L.m) * n);
        HIP_CHECK(hipMemcpyAsync(h.data(), Wt, h.size() * 4, hipMemcpyDeviceToHost, s));
        HIP_CHECK(hipStreamSynchronize(s));
        write_f32(opt.dump_dir + "/" + tag + "_" + L.name + ".W.f32", h.data(), h.size());
      }
      HIP_CHECK(hipStreamSynchronize(s));
      T.load += now_s() - t0;
      t0 = now_s();
      // Wr = hadUt(hadUt(W^T * SV)^T * SU)  ==  T_m^T diag(SV) W diag(SU) T_n ; we build Wr^T directly
      k_scale_rows_cols(Wt, L.m, n, n, d_sv + off[li], d_su, s);
      k_hadamard_rows(Wt, L.m, n, n, true, s);
      k_transpose(Wt, L.m, n, n, WT + off[li], mt, s);
      k_hadamard_rows(WT + off[li], n, L.m, mt, true, s);
      double ss = dev_sumsq(s, WT + off[li], n, L.m, mt, acc);
      double rms = std::sqrt(ss / (double(n) * L.m));
      O.wscale = float(rms / (double(float(cb.lut_rms)) * opt.scale_override));
      k_scale(WT + off[li], n, L.m, mt, 1.0f / O.wscale, s);
      HIP_CHECK(hipStreamSynchronize(s));
      T.rht += now_s() - t0;
      t0 = now_s();
    }
  }

  DBuf<float> Ld(size_t(n) * n);
  HIP_CHECK(hipMemcpyAsync(Ld, hp.L.data(), hp.L.size() * 4, hipMemcpyHostToDevice, s));
  HIP_CHECK(hipStreamSynchronize(s));
  hp.L.clear();
  hp.L.shrink_to_fit();
  const double creg = hp.creg;
  t0 = now_s();

  // ---- LDLQ + Viterbi
  DBuf<float> hatWT(size_t(n) * mt);
  DBuf<uint16_t> Q(size_t(mt / 16) * (n / 16) * 128);
  double vit = 0;
  ldlq_gpu(s, gemm, WT, Ld, n, mt, hatWT, Q, opt.vit_batch, lut, &vit);
  HIP_CHECK(hipStreamSynchronize(s));
  T.ldlq += now_s() - t0;
  T.viterbi += vit;
  t0 = now_s();
  Ld.free();

  // ---- proxy errors (rotated, scaled domain; Wscale cancels): tr(D HRr D^T) / tr(W HRr W^T)
  // The unregularized original-domain proxy follows from HRr = rot(H) + creg * I (rotation is orthogonal).
  {
    DBuf<uint16_t> Hcat(size_t(3) * n * n);
    {
      DBuf<float> HR(size_t(n) * n);
      HIP_CHECK(hipMemcpyAsync(HR, hp.HRr.data(), HR.bytes(), hipMemcpyHostToDevice, s));
      k_split_bf16(HR, n, n, n, Hcat, Hcat + long(n) * n, n, s);  // [Hh | Hl | Hh]
      HIP_CHECK(hipMemcpyAsync(Hcat + 2L * n * n, Hcat, size_t(n) * n * 2, hipMemcpyDeviceToDevice, s));
      HIP_CHECK(hipStreamSynchronize(s));
    }
    DBuf<float> DT(size_t(n) * mt);
    k_sub(WT, hatWT, DT, n, mt, mt, s);
    DBuf<uint16_t> Acat;
    DBuf<float> Y;
    for (size_t li = 0; li < lins.size(); ++li) {
      long m = lins[li].m;
      long chunk = std::max<long>(16, std::min<long>(m, (long(1) << 30) / (long(n) * 6) / 16 * 16));
      double num = 0, den = 0, dn2 = 0, wn2 = 0;
      for (long c = 0; c < m; c += chunk) {
        long mc = std::min(chunk, m - c);
        den += quad_trace(s, gemm, WT, n, mt, off[li] + c, mc, Hcat, Acat, Y, acc);
        num += quad_trace(s, gemm, DT, n, mt, off[li] + c, mc, Hcat, Acat, Y, acc);
      }
      wn2 = dev_sumsq(s, WT + off[li], n, m, mt, acc);
      dn2 = dev_sumsq(s, DT + off[li], n, m, mt, acc);
      out[li].proxy_rot_reg = num / den;
      out[li].proxy = (num - creg * dn2) / (den - creg * wn2);
    }
  }
  HIP_CHECK(hipStreamSynchronize(s));
  T.proxy += now_s() - t0;
  t0 = now_s();

  // ---- states -> packed trellis (reference pack_trellis + kernel swizzle)
  {
    const long nblk = n / 16;
    std::vector<uint16_t> Qh(size_t(mt / 16) * nblk * 128);
    HIP_CHECK(hipMemcpy(Qh.data(), Q, Qh.size() * 2, hipMemcpyDeviceToHost));
    for (size_t li = 0; li < lins.size(); ++li) {
      long m = lins[li].m;
      long ntr = (m / 16) * nblk;
      const uint16_t* q = Qh.data() + size_t(off[li] / 16) * nblk * 128;
      std::vector<uint16_t> packed(size_t(ntr) * kWordsPerTrellis);
      pack_trellis(q, ntr, packed.data());
      out[li].trellis.resize(packed.size());
      swizzle_for_kernel(packed.data(), int(m), n, out[li].trellis.data());
    }
  }
  if (!opt.dump_dir.empty()) {
    // reconstructed weight in the original domain: hatW = diag(SV) T_m (hatWr * Wscale) T_n^T diag(SU)
    for (size_t li = 0; li < lins.size(); ++li) {
      long m = lins[li].m;
      DBuf<float> a(size_t(n) * m), b(size_t(m) * n);
      HIP_CHECK(hipMemcpy2DAsync(a, size_t(m) * 4, hatWT + off[li], size_t(mt) * 4, size_t(m) * 4, size_t(n),
                                 hipMemcpyDeviceToDevice, s));
      k_hadamard_rows(a, n, int(m), m, false, s);  // rows of hatW_T: x <- T_m x
      k_transpose(a, n, m, m, b, n, s);              // -> T_m hatWr (m x n)
      k_hadamard_rows(b, m, n, n, false, s);         // -> T_m hatWr T_n^T
      k_scale(b, m, n, n, out[li].wscale, s);
      k_scale_rows_cols(b, m, n, n, d_sv + off[li], d_su, s);
      std::vector<float> h(size_t(m) * n);
      HIP_CHECK(hipMemcpyAsync(h.data(), b, h.size() * 4, hipMemcpyDeviceToHost, s));
      HIP_CHECK(hipStreamSynchronize(s));
      write_f32(opt.dump_dir + "/" + tag + "_" + lins[li].name + ".hatW.f32", h.data(), h.size());
    }
  }
  T.pack += now_s() - t0;
  T.total = now_s() - t_start + T.rht + T.ldl_cpu;
  double wsum = 0;
  for (auto& l : lins) wsum += double(l.m) * n;
  for (size_t li = 0; li < lins.size(); ++li) out[li].seconds = T.total * double(lins[li].m) * n / wsum;
  if (tm) *tm = T;
  return out;
}

}  // namespace qtip
