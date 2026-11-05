// Calibrated NVFP4 quantization kernels (see nvfp4_gpu.hpp). Single source for HIP and CUDA.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.hpp"
#include "gpu.hpp"
#include "nvfp4_gpu.hpp"
#include "qtip_core.hpp"

namespace qtip {

namespace {
inline unsigned nblocks(long n, int bs) { return unsigned((n + bs - 1) / bs); }

__device__ __forceinline__ float nv_fp8_dec(unsigned b) {
  unsigned e = (b >> 3) & 0xf, m = b & 7;
  return e == 0 ? ldexpf(float(m), -9) : ldexpf(float(m | 8), int(e) - 10);
}
// same as qtip::f32_to_fp8e4m3 for non-negative finite x (saturating at 448 = code 126)
__device__ __forceinline__ unsigned nv_fp8_enc(float x) {
  x = fminf(fabsf(x), 448.f);
  auto round_even = [](float v) {
    const float f = floorf(v), r = v - f;
    unsigned n = unsigned(f);
    return n + unsigned(r > .5f || (r == .5f && (n & 1U)));
  };
  if (x < ldexpf(1.f, -6)) return round_even(x * 512.f);
  int exp;
  const float frac = frexpf(x, &exp) * 2;
  unsigned mant = round_even((frac - 1) * 8);
  int encoded_exp = exp + 6;
  if (mant == 8) {
    mant = 0;
    ++encoded_exp;
  }
  unsigned c = (unsigned(encoded_exp) << 3) | mant;
  return c < 126u ? c : 126u;
}
__device__ __forceinline__ float nv_bf16_round(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xffff0000u);
}
// nearest E2M1 magnitude code of |a| (ties to the even code, as nvfp4_quantize)
__device__ __forceinline__ int nv_fp4_code(float a) {
  const float lv[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  float best = INFINITY;
  int code = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float e = fabsf(a - lv[i]);
    if (e < best || (e == best && (i % 2 == 0))) {
      best = e;
      code = i;
    }
  }
  return code;
}
__device__ __forceinline__ float nv_level(int c) {
  const float lv[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return lv[c];
}

constexpr int kCand = 11;

// One thread per matrix row, one 16-column block (columns c0..c0+15) per launch.
//   t_k   = W[k] + F[k] + sum_{j>k in block} d_j Lb[j][k]       (target after LDLQ feedback)
//   q_k   = bf16(level * dec(s) * global)                        (what execution sees)
//   d_k   = W[k] - q_k                                           (feedback error, original weight)
//   cost  = sum_k dw[k] (t_k - q_k)^2                            (proxy increment; dw = D, or diag(H) w/o feedback)
__global__ void nv_block_kernel(const float* __restrict__ W, const float* __restrict__ F, int m, int n, int c0,
                                const float* __restrict__ L, const float* __restrict__ dw, float global,
                                int calibrate, int search, uint8_t* __restrict__ packed,
                                uint8_t* __restrict__ scales, float* __restrict__ E) {
  __shared__ float Lb[16][16];
  __shared__ float Dw[16];
  for (int idx = threadIdx.x; idx < 256; idx += blockDim.x)
    Lb[idx / 16][idx % 16] = calibrate ? L[size_t(c0 + idx / 16) * n + c0 + idx % 16] : 0.f;
  if (threadIdx.x < 16) Dw[threadIdx.x] = dw ? dw[c0 + threadIdx.x] : 1.f;
  __syncthreads();
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= m) return;
  float w[16], base[16];
  float amax = 0;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    w[k] = W[size_t(r) * n + c0 + k];
    base[k] = w[k] + (calibrate ? F[size_t(r) * n + c0 + k] : 0.f);
    amax = fmaxf(amax, fabsf(calibrate ? base[k] : w[k]));
  }
  unsigned cand[kCand];
  int nc = 0;
  const unsigned c1 = nv_fp8_enc(amax / (6.f * global));
  cand[nc++] = c1;
  if (search) {
    const float g[8] = {0.70f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f, 1.00f, 1.05f};
    for (int i = 0; i < 8; ++i) cand[nc++] = nv_fp8_enc(amax * g[i] / (6.f * global));
    cand[nc++] = c1 > 0 ? c1 - 1 : 0;
    cand[nc++] = c1 < 126 ? c1 + 1 : 126;
  }
  float best_cost = INFINITY;
  unsigned best = c1;
  for (int ci = 0; ci < nc; ++ci) {
    const unsigned s = cand[ci];
    const float sd = nv_fp8_dec(s);
    const float scale = sd * global;
    float err[16];
    float cost = 0;
#pragma unroll
    for (int k = 15; k >= 0; --k) {
      float t = base[k];
      if (calibrate) {
#pragma unroll
        for (int j = k + 1; j < 16; ++j) t += err[j] * Lb[j][k];
      }
      const float x = scale != 0.f ? t / scale : 0.f;
      const int code = nv_fp4_code(fabsf(x));
      const float q = nv_bf16_round((x < 0 ? -1.f : 1.f) * nv_level(code) * sd * global);
      // calibrated: t is the LDLQ target (cost uses t - q); plain: t == w
      cost += Dw[k] * (t - q) * (t - q);
      err[k] = w[k] - q;
    }
    if (cost < best_cost) {
      best_cost = cost;
      best = s;
    }
  }
  // final pass with the winning scale
  const float sd = nv_fp8_dec(best);
  const float scale = sd * global;
  float err[16];
  uint8_t nib[16];
#pragma unroll
  for (int k = 15; k >= 0; --k) {
    float t = base[k];
    if (calibrate) {
#pragma unroll
      for (int j = k + 1; j < 16; ++j) t += err[j] * Lb[j][k];
    }
    const float x = scale != 0.f ? t / scale : 0.f;
    const int code = nv_fp4_code(fabsf(x));
    const float q = nv_bf16_round((x < 0 ? -1.f : 1.f) * nv_level(code) * sd * global);
    err[k] = w[k] - q;
    nib[k] = uint8_t(code | ((__float_as_uint(x) >> 31) ? 8 : 0));
  }
#pragma unroll
  for (int k = 0; k < 16; k += 2)
    packed[size_t(r) * (n / 2) + (c0 + k) / 2] = uint8_t(nib[k] | (nib[k + 1] << 4));
  scales[size_t(r) * (n / 16) + c0 / 16] = uint8_t(best);
#pragma unroll
  for (int k = 0; k < 16; ++k) E[size_t(r) * 16 + k] = err[k];
}

// F[r][col] += sum_{j<16} E[r][j] * L[c0+j][col]   for col < c0
__global__ void nv_feedback_kernel(float* __restrict__ F, int m, int n, int c0, const float* __restrict__ E,
                                   const float* __restrict__ L) {
  const long col = long(blockIdx.x) * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (col >= c0 || r >= m) return;
  float acc = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) acc += E[size_t(r) * 16 + j] * L[size_t(c0 + j) * n + col];
  F[size_t(r) * n + col] += acc;
}
}  // namespace

std::vector<Nvfp4Tensor> nvfp4_quantize_group(hipStream_t s, const std::vector<const float*>& ws,
                                              const std::vector<int>& ms, int n, const float* H,
                                              const Nvfp4Opts& opt, bool* calibrated) {
  QCHECK(n > 0 && n % 16 == 0 && ws.size() == ms.size(), "NVFP4 group geometry");
  const bool cal = opt.calibrate && H != nullptr;
  if (calibrated) *calibrated = cal;
  DBuf<float> d_L, d_dw;
  std::vector<float> dwh(size_t(n), 1.f);
  if (H) {
    // regularized Hessian; weights for the scale search: diag(H) (uncalibrated) or D = diag(chol)^2 (calibrated)
    double dsum = 0;
    for (int i = 0; i < n; ++i) dsum += H[size_t(i) * n + i];
    const double creg = opt.sigma_reg * dsum / n;
    if (cal) {
      std::vector<double> A(size_t(n) * n);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n; ++i)
        for (long j = 0; j <= i; ++j) A[size_t(i) * n + j] = 0.5 * (double(H[size_t(i) * n + j]) + H[size_t(j) * n + i]) + (i == j ? creg : 0.0);
      QCHECK(cholesky_inplace(A.data(), n, 16), "NVFP4 calibration: Hessian not positive definite");
      std::vector<float> L(size_t(n) * n, 0.f);
#pragma omp parallel for schedule(static)
      for (long j = 0; j < n; ++j) {
        const double d = A[size_t(j) * n + j];
        dwh[size_t(j)] = float(d * d);
        for (long i = j + 1; i < n; ++i) L[size_t(i) * n + j] = float(A[size_t(i) * n + j] / d);  // unit-lower, H = L D L^T
      }
      d_L.alloc(L.size());
      HIP_CHECK(hipMemcpyAsync(d_L, L.data(), L.size() * 4, hipMemcpyHostToDevice, s));
      HIP_CHECK(hipStreamSynchronize(s));
    } else {
      for (int i = 0; i < n; ++i) dwh[size_t(i)] = float(H[size_t(i) * n + i] + creg);
    }
    d_dw.alloc(size_t(n));
    HIP_CHECK(hipMemcpyAsync(d_dw, dwh.data(), size_t(n) * 4, hipMemcpyHostToDevice, s));
  }
  std::vector<Nvfp4Tensor> out(ws.size());
  for (size_t mi = 0; mi < ws.size(); ++mi) {
    const int m = ms[mi];
    const float* w = ws[mi];
    float maximum = 0;
    for (size_t i = 0; i < size_t(m) * n; ++i) {
      QCHECK(std::isfinite(w[i]), "nonfinite NVFP4 input");
      maximum = std::max(maximum, std::fabs(w[i]));
    }
    Nvfp4Tensor& t = out[mi];
    t.m = m, t.n = n;
    t.global = maximum ? maximum / (448.f * 6) : 1.f;
    QCHECK(t.global > 0 && std::isfinite(t.global), "NVFP4 global scale underflow/overflow");
    DBuf<float> d_W(size_t(m) * n), d_F, d_E(size_t(m) * 16);
    DBuf<uint8_t> d_packed(size_t(m) * n / 2), d_scales(size_t(m) * n / 16);
    HIP_CHECK(hipMemcpyAsync(d_W, w, size_t(m) * n * 4, hipMemcpyHostToDevice, s));
    if (cal) {
      d_F.alloc(size_t(m) * n);
      HIP_CHECK(hipMemsetAsync(d_F, 0, size_t(m) * n * 4, s));
    }
    for (int c0 = n - 16; c0 >= 0; c0 -= 16) {
      nv_block_kernel<<<nblocks(m, 128), 128, 0, s>>>(d_W, cal ? d_F.p : nullptr, m, n, c0, cal ? d_L.p : nullptr,
                                                     d_dw.p, t.global, cal ? 1 : 0, opt.scale_search ? 1 : 0,
                                                     d_packed, d_scales, d_E);
      HIP_CHECK(hipGetLastError());
      if (cal && c0 > 0) {
        dim3 grid(nblocks(c0, 128), unsigned(m));
        nv_feedback_kernel<<<grid, 128, 0, s>>>(d_F, m, n, c0, d_E, d_L);
        HIP_CHECK(hipGetLastError());
      }
    }
    t.packed.resize(size_t(m) * n / 2);
    t.scales.resize(size_t(m) * n / 16);
    HIP_CHECK(hipMemcpyAsync(t.packed.data(), d_packed, t.packed.size(), hipMemcpyDeviceToHost, s));
    HIP_CHECK(hipMemcpyAsync(t.scales.data(), d_scales, t.scales.size(), hipMemcpyDeviceToHost, s));
    HIP_CHECK(hipStreamSynchronize(s));
  }
  return out;
}

}  // namespace qtip
