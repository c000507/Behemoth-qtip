#include "qtip_core.hpp"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

#include "common.hpp"
#include "qtip_constants.hpp"

namespace qtip {

// ------------------------------------------------------------------ codebook
static Codebook build_codebook() {
  Codebook cb;
  for (int i = 0; i < 512; ++i)
    for (int v = 0; v < 2; ++v) {
      uint32_t bits = kTlutF32Bits[i * 2 + v];
      float f;
      std::memcpy(&f, &bits, 4);
      cb.tlut[i][v] = f;
      cb.tlut_h[i][v] = f32_to_fp16(f);
    }
  cb.lut0.resize(kStates);
  cb.lut1.resize(kStates);
  cb.lut_h2.resize(kStates);
  double ss = 0;
  for (uint32_t s = 0; s < uint32_t(kStates); ++s) {
    // quantlut_sym: lut = (s+1)*s; sflp = 1 - ((lut >> 15) & 1) * 2; idx = (lut >> (16 - nbits - 1)) & (2^nbits - 1)
    uint64_t p = uint64_t(s + 1) * uint64_t(s);
    int flip = int((p >> 15) & 1);
    int idx = int((p >> (16 - kTlutBits - 1)) & ((1u << kTlutBits) - 1));
    uint16_t h0 = cb.tlut_h[idx][0], h1 = cb.tlut_h[idx][1];
    if (flip) h0 ^= 0x8000;  // exact negation in fp16
    cb.lut0[s] = fp16_to_f32(h0);
    cb.lut1[s] = fp16_to_f32(h1);
    cb.lut_h2[s] = uint32_t(h0) | (uint32_t(h1) << 16);
    ss += double(cb.lut0[s]) * cb.lut0[s] + double(cb.lut1[s]) * cb.lut1[s];
  }
  cb.lut_rms = std::sqrt(ss / (2.0 * kStates));
  return cb;
}
const Codebook& codebook() {
  static Codebook cb = build_codebook();
  return cb;
}

static std::vector<int> g_perm, g_inv;
static std::once_flag g_perm_once;
static void build_perm() {
  g_perm.resize(256);
  g_inv.resize(256);
  // arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
  int p = 0;
  for (int b = 0; b < 8; ++b)
    for (int d = 0; d < 4; ++d)
      for (int c = 0; c < 2; ++c)
        for (int a = 0; a < 2; ++a)
          for (int e = 0; e < 2; ++e) g_perm[p++] = a * 128 + b * 16 + c * 8 + d * 2 + e;
  for (int i = 0; i < 256; ++i) g_inv[g_perm[i]] = i;
}
const int* perm256() {
  std::call_once(g_perm_once, build_perm);
  return g_perm.data();
}
const int* inv_perm256() {
  std::call_once(g_perm_once, build_perm);
  return g_inv.data();
}

// ------------------------------------------------------------------ hadamard
static bool is_pow2(long n) { return n > 0 && (n & (n - 1)) == 0; }
int had_K(int n) {
  for (int K : kHadOrder)
    if (n % K == 0) return is_pow2(n / K) ? K : 0;
  return is_pow2(n) ? 1 : 0;
}

void hadamard_rows_ref(double* X, long rows, int n, long ld, bool transpose) {
  int K = had_K(n);
  QCHECK(K > 0, "unsupported hadamard size " + std::to_string(n));
  const int8_t* hk = K > 1 ? had_table(K) : nullptr;
  int P = n / K;
  double sc = 1.0 / std::sqrt(double(n));
#pragma omp parallel for schedule(static)
  for (long r = 0; r < rows; ++r) {
    double* x = X + r * ld;
    std::vector<double> y(x, x + n), z(n);
    // matmul_hadU loop: view (-1, n, 1) -> repeatedly pair adjacent along the shrinking axis.
    // Equivalent to a natural-order FWHT on each contiguous chunk of P = n / K entries.
    for (int c = 0; c < K; ++c) {
      double* v = y.data() + long(c) * P;
      for (int h = 1; h < P; h <<= 1)
        for (int i = 0; i < P; i += 2 * h)
          for (int j = i; j < i + h; ++j) {
            double a = v[j], b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
          }
    }
    if (K > 1) {
      for (int k2 = 0; k2 < K; ++k2)
        for (int i = 0; i < P; ++i) {
          double s = 0;
          for (int k = 0; k < K; ++k) {
            int8_t h = transpose ? hk[k * K + k2] : hk[k2 * K + k];
            s += h * y[long(k) * P + i];
          }
          z[long(k2) * P + i] = s;
        }
      for (int i = 0; i < n; ++i) x[i] = z[i] * sc;
    } else {
      for (int i = 0; i < n; ++i) x[i] = y[i] * sc;
    }
  }
}

// ------------------------------------------------------------------ viterbi (reference)
static inline double sq_err(uint32_t s, double x0, double x1) {
  const Codebook& cb = codebook();
  double a = cb.lut0[s] - x0, b = cb.lut1[s] - x1;
  return a * a + b * b;
}

double viterbi_ref(const float* X, int overlap, uint16_t* states) {
  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> cost(kStates), ncost(kStates), M(kGroups);
  std::vector<uint8_t> A(size_t(kSteps) * kGroups);
  for (uint32_t s = 0; s < uint32_t(kStates); ++s) {
    cost[s] = sq_err(s, X[0], X[1]);
    if (overlap >= 0 && int(s >> kKV) != overlap) cost[s] = INF;
  }
  for (int i = 1; i < kSteps; ++i) {
    for (int j = 0; j < kGroups; ++j) {
      double best = cost[j];
      int bd = 0;
      for (int d = 1; d < 16; ++d) {
        double c = cost[j | (d << 12)];
        if (c < best) best = c, bd = d;
      }
      M[j] = best;
      A[size_t(i) * kGroups + j] = uint8_t(bd);
    }
    double x0 = X[2 * i], x1 = X[2 * i + 1];
    for (uint32_t s = 0; s < uint32_t(kStates); ++s) ncost[s] = sq_err(s, x0, x1) + M[s >> kKV];
    std::swap(cost, ncost);
  }
  double best = INF;
  uint32_t bs = 0;
  for (uint32_t s = 0; s < uint32_t(kStates); ++s) {
    if (overlap >= 0 && int(s & (kGroups - 1)) != overlap) continue;
    if (cost[s] < best) best = cost[s], bs = s;
  }
  states[kSteps - 1] = uint16_t(bs);
  for (int i = kSteps - 1; i > 0; --i) {
    uint32_t j = states[i] >> kKV;
    states[i - 1] = uint16_t(j | (uint32_t(A[size_t(i) * kGroups + j]) << 12));
  }
  return best;
}

void decode_states(const uint16_t* states, float* hatX) {
  const Codebook& cb = codebook();
  for (int i = 0; i < kSteps; ++i) {
    hatX[2 * i] = cb.lut0[states[i]];
    hatX[2 * i + 1] = cb.lut1[states[i]];
  }
}

double path_cost(const float* X, const uint16_t* states) {
  double c = 0;
  for (int i = 0; i < kSteps; ++i) c += sq_err(states[i], X[2 * i], X[2 * i + 1]);
  return c;
}

double quantize_seq_ref(const float* X, uint16_t* states, float* hatX) {
  // roll_X = torch.roll(X, T // (2V) * V, 0) -> roll_X[t] = X[t - 128]
  float R[kSeq];
  const int sh = kSeq / (2 * kV) * kV;  // 128
  for (int t = 0; t < kSeq; ++t) R[t] = X[(t - sh + kSeq) % kSeq];
  uint16_t st1[kSteps];
  viterbi_ref(R, -1, st1);
  int overlap = st1[kSeq / (2 * kV)] >> kKV;
  double c = viterbi_ref(X, overlap, states);
  decode_states(states, hatX);
  return c;
}

// ------------------------------------------------------------------ packing
void pack_trellis(const uint16_t* states, long B, uint16_t* packed) {
  for (long b = 0; b < B; ++b) {
    const uint16_t* st = states + b * kSteps;
    uint16_t* out = packed + b * kWordsPerTrellis;
    std::memset(out, 0, kWordsPerTrellis * 2);
    // bitstream: state[0] (16 bits, MSB first), then low KV bits of each later state; keep the first 512 bits
    auto put = [&](int pos, int bit) {
      if (pos < kSeq * kK && bit) out[pos >> 4] |= uint16_t(1u << (15 - (pos & 15)));
    };
    int pos = 0;
    for (int i = kL - 1; i >= 0; --i) put(pos++, (st[0] >> i) & 1);
    for (int t = 1; t < kSteps; ++t)
      for (int i = kKV - 1; i >= 0; --i) put(pos++, (st[t] >> i) & 1);
  }
}

void unpack_trellis(const uint16_t* packed, long B, uint16_t* states) {
  for (long b = 0; b < B; ++b) {
    const uint16_t* in = packed + b * kWordsPerTrellis;
    uint16_t* st = states + b * kSteps;
    auto bit = [&](int pos) -> uint32_t {
      pos %= (kSeq * kK);  // wrap-around: bf = concat(bf, bf[:, :L - KV])
      return (in[pos >> 4] >> (15 - (pos & 15))) & 1u;
    };
    uint32_t s = 0;
    for (int i = 0; i < kL; ++i) s = (s << 1) | bit(i);
    st[0] = uint16_t(s);
    for (int t = 1; t < kSteps; ++t) {
      uint32_t e = 0;
      for (int i = 0; i < kKV; ++i) e = (e << 1) | bit(kL + (t - 1) * kKV + i);
      s = ((s << kKV) & (kStates - 1)) | e;
      st[t] = uint16_t(s);
    }
  }
}

// packed.view(uint8).view(-1, 2).flip(-1).reshape(m/32, 2, n/32, 2, 32, K).permute(0, 2, 4, 3, 1, 5).flip(-1)
// The two byte flips cancel, so at uint16 granularity: out[(((i0*(n/32)+j0)*32 + c)*2 + j1)*2 + i1] =
// packed[(2*i0+i1)*(n/16) + 2*j0+j1][c]   (verified against the python reference in tests/ref_check.py)
void swizzle_for_kernel(const uint16_t* packed, int m, int n, uint16_t* out) {
  QCHECK(m % 32 == 0 && n % 32 == 0, "kernel layout needs m, n multiples of 32");
  long nb = n / 16, M2 = m / 32, N2 = n / 32;
#pragma omp parallel for schedule(static)
  for (long i0 = 0; i0 < M2; ++i0)
    for (long j0 = 0; j0 < N2; ++j0)
      for (int c = 0; c < 32; ++c)
        for (int j1 = 0; j1 < 2; ++j1)
          for (int i1 = 0; i1 < 2; ++i1)
            out[(((i0 * N2 + j0) * 32 + c) * 2 + j1) * 2 + i1] = packed[((2 * i0 + i1) * nb + 2 * j0 + j1) * 32 + c];
}
void unswizzle_for_kernel(const uint16_t* in, int m, int n, uint16_t* packed) {
  long nb = n / 16, M2 = m / 32, N2 = n / 32;
#pragma omp parallel for schedule(static)
  for (long i0 = 0; i0 < M2; ++i0)
    for (long j0 = 0; j0 < N2; ++j0)
      for (int c = 0; c < 32; ++c)
        for (int j1 = 0; j1 < 2; ++j1)
          for (int i1 = 0; i1 < 2; ++i1)
            packed[((2 * i0 + i1) * nb + 2 * j0 + j1) * 32 + c] = in[(((i0 * N2 + j0) * 32 + c) * 2 + j1) * 2 + i1];
}

// ------------------------------------------------------------------ fp64 Cholesky
// Right-looking blocked Cholesky. Trailing update C -= P P^T with an AVX-512 6x16 microkernel on packed panels.
namespace {
constexpr int NB = 256;  // panel width
constexpr int MR = 6, NR = 16;

// C[MR x NR] (ldc) -= sum_p a[p*MR + i] * b[p*NR + j]
static inline void micro_6x16(int kc, const double* a, const double* b, double* C, long ldc, int mr, int nr) {
  __m512d c[MR][2];
  for (int i = 0; i < MR; ++i) c[i][0] = c[i][1] = _mm512_setzero_pd();
  for (int p = 0; p < kc; ++p) {
    __m512d b0 = _mm512_loadu_pd(b + p * NR), b1 = _mm512_loadu_pd(b + p * NR + 8);
    for (int i = 0; i < MR; ++i) {
      __m512d ai = _mm512_set1_pd(a[p * MR + i]);
      c[i][0] = _mm512_fmadd_pd(ai, b0, c[i][0]);
      c[i][1] = _mm512_fmadd_pd(ai, b1, c[i][1]);
    }
  }
  if (mr == MR && nr == NR) {
    for (int i = 0; i < MR; ++i) {
      double* ci = C + i * ldc;
      _mm512_storeu_pd(ci, _mm512_sub_pd(_mm512_loadu_pd(ci), c[i][0]));
      _mm512_storeu_pd(ci + 8, _mm512_sub_pd(_mm512_loadu_pd(ci + 8), c[i][1]));
    }
  } else {
    alignas(64) double t[MR][NR];
    for (int i = 0; i < MR; ++i) {
      _mm512_store_pd(t[i], c[i][0]);
      _mm512_store_pd(t[i] + 8, c[i][1]);
    }
    for (int i = 0; i < mr; ++i)
      for (int j = 0; j < nr; ++j) C[i * ldc + j] -= t[i][j];
  }
}
}  // namespace

bool cholesky_inplace(double* A, int n, int nthreads) {
  if (nthreads > 0) omp_set_num_threads(nthreads);
  std::vector<double> packA, packB;
  bool ok = true;
  for (int k = 0; k < n && ok; k += NB) {
    int kb = std::min(NB, n - k);
    // 1) unblocked Cholesky of the diagonal block
    for (int j = k; j < k + kb; ++j) {
      double* aj = A + long(j) * n;
      double d = aj[j];
      for (int p = k; p < j; ++p) d -= aj[p] * aj[p];
      if (!(d > 0)) { ok = false; break; }
      d = std::sqrt(d);
      aj[j] = d;
      for (int i = j + 1; i < k + kb; ++i) {
        double* ai = A + long(i) * n;
        double s = ai[j];
        for (int p = k; p < j; ++p) s -= ai[p] * aj[p];
        ai[j] = s / d;
      }
    }
    if (!ok) break;
    int r0 = k + kb, nr = n - r0;
    if (nr <= 0) break;
    // 2) panel solve: rows i >= r0: x L11^T = a
#pragma omp parallel for schedule(static)
    for (int i = r0; i < n; ++i) {
      double* ai = A + long(i) * n + k;
      for (int j = 0; j < kb; ++j) {
        const double* lj = A + long(k + j) * n + k;
        double s = ai[j];
        for (int p = 0; p < j; ++p) s -= ai[p] * lj[p];
        ai[j] = s / lj[j];
      }
    }
    // 3) pack panel rows r0..n-1 (cols k..k+kb) in MR- and NR-interleaved layouts
    int nA = (nr + MR - 1) / MR, nBk = (nr + NR - 1) / NR;
    packA.assign(size_t(nA) * MR * kb, 0.0);
    packB.assign(size_t(nBk) * NR * kb, 0.0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nr; ++i) {
      const double* src = A + long(r0 + i) * n + k;
      double* pa = packA.data() + size_t(i / MR) * MR * kb + (i % MR);
      double* pb = packB.data() + size_t(i / NR) * NR * kb + (i % NR);
      for (int p = 0; p < kb; ++p) {
        pa[size_t(p) * MR] = src[p];
        pb[size_t(p) * NR] = src[p];
      }
    }
    // 4) trailing update of the lower triangle: tiles of (MC rows) x (NR cols)
    constexpr int MC = 96;  // rows per task (multiple of MR)
    int nTi = (nr + MC - 1) / MC;
#pragma omp parallel for schedule(dynamic, 1)
    for (long task = 0; task < long(nTi) * nBk; ++task) {
      int ti = int(task / nBk), tj = int(task % nBk);
      int i0 = ti * MC, j0 = tj * NR;
      if (j0 > i0 + MC - 1) continue;  // tile fully above the diagonal
      for (int ii = i0; ii < std::min(nr, i0 + MC); ii += MR) {
        if (j0 > ii + MR - 1) continue;
        int mr = std::min(MR, nr - ii), nrr = std::min(NR, nr - j0);
        micro_6x16(kb, packA.data() + size_t(ii / MR) * MR * kb, packB.data() + size_t(tj) * NR * kb,
                   A + long(r0 + ii) * n + r0 + j0, n, mr, nrr);
      }
    }
  }
  return ok;
}

bool block_ldl_to_L(double* A, int n, float* L_out, int nthreads) {
  QCHECK(n % kTd == 0, "n must be a multiple of 16");
  if (!cholesky_inplace(A, n, nthreads)) return false;
  int nb = n / kTd;
  // inverse of each diagonal (lower-triangular) 16x16 block
  std::vector<double> inv(size_t(nb) * 256);
#pragma omp parallel for schedule(static)
  for (int bi = 0; bi < nb; ++bi) {
    double D[16][16], Iv[16][16];
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) D[i][j] = j <= i ? A[long(bi * 16 + i) * n + bi * 16 + j] : 0.0;
    for (int c = 0; c < 16; ++c)  // forward substitution for column c of the inverse
      for (int i = 0; i < 16; ++i) {
        double s = (i == c) ? 1.0 : 0.0;
        for (int p = 0; p < i; ++p) s -= D[i][p] * Iv[p][c];
        Iv[i][c] = s / D[i][i];
      }
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) inv[size_t(bi) * 256 + i * 16 + j] = Iv[i][j];
  }
  bool finite = true;
#pragma omp parallel for schedule(dynamic, 16) reduction(&& : finite)
  for (int r = 0; r < n; ++r) {
    const double* ar = A + long(r) * n;
    float* lr = L_out + long(r) * n;
    int rb = r / 16;
    for (int bj = 0; bj < nb; ++bj) {
      float* o = lr + bj * 16;
      if (bj >= rb) {  // diagonal block (identity, then diagonal zeroed -> all zero) and upper part
        std::memset(o, 0, 16 * sizeof(float));
        continue;
      }
      const double* a = ar + bj * 16;
      const double* iv = inv.data() + size_t(bj) * 256;
      for (int j = 0; j < 16; ++j) {
        double s = 0;
        for (int p = j; p < 16; ++p) s += a[p] * iv[p * 16 + j];
        o[j] = float(s);
        finite = finite && std::isfinite(s);
      }
    }
  }
  return finite;
}

}  // namespace qtip
