// Elementwise / layout / Hadamard / model-forward kernels.
#include <hip/hip_runtime.h>

#include <map>
#include <mutex>

#include "gpu.hpp"
#include "qtip_constants.hpp"
#include "qtip_core.hpp"

namespace qtip {

namespace {
inline unsigned nblocks(long n, int bs) { return unsigned((n + bs - 1) / bs); }
__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
__device__ __forceinline__ float fp8_to_f(uint8_t b) {
  uint32_t s = b >> 7, e = (b >> 3) & 0xf, m = b & 7;
  float v;
  if (e == 0) v = ldexpf(float(m), -9);
  else if (e == 15 && m == 7) v = __builtin_nanf("");
  else v = ldexpf(float(m | 8), int(e) - 10);
  return s ? -v : v;
}
}  // namespace

// ---------------------------------------------------------------- conversions
__global__ void fp8_to_f32_kernel(const uint8_t* w, const uint16_t* sc, float* out, long rows, long cols) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  float v = fp8_to_f(w[i]);
  out[i] = sc ? v * bf2f(sc[i / cols]) : v;
}
void k_fp8_to_f32(const uint8_t* w, const uint16_t* scale_bf16, float* out, long rows, long cols, hipStream_t s) {
  fp8_to_f32_kernel<<<nblocks(rows * cols, 256), 256, 0, s>>>(w, scale_bf16, out, rows, cols);
  HIP_CHECK(hipGetLastError());
}
__global__ void fp8_to_bf16_kernel(const uint8_t* w, uint16_t* out, long n) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = f2bf(fp8_to_f(w[i]));  // exact
}
void k_fp8_to_bf16(const uint8_t* w, uint16_t* out, long n, hipStream_t s) {
  fp8_to_bf16_kernel<<<nblocks(n, 256), 256, 0, s>>>(w, out, n);
  HIP_CHECK(hipGetLastError());
}
__global__ void f32_to_bf16_kernel(const float* in, uint16_t* out, long n) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = f2bf(in[i]);
}
void k_f32_to_bf16(const float* in, uint16_t* out, long n, hipStream_t s) {
  f32_to_bf16_kernel<<<nblocks(n, 256), 256, 0, s>>>(in, out, n);
  HIP_CHECK(hipGetLastError());
}
__global__ void bf16_to_f32_kernel(const uint16_t* in, float* out, long n) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = bf2f(in[i]);
}
void k_bf16_to_f32(const uint16_t* in, float* out, long n, hipStream_t s) {
  bf16_to_f32_kernel<<<nblocks(n, 256), 256, 0, s>>>(in, out, n);
  HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------- scaling / reductions
__global__ void scale_rc_kernel(float* X, long rows, long cols, long ld, const float* rs, const float* cs) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long r = i / cols, c = i % cols;
  float v = X[r * ld + c];
  if (rs) v *= rs[r];
  if (cs) v *= cs[c];
  X[r * ld + c] = v;
}
void k_scale_rows_cols(float* X, long rows, long cols, long ld, const float* rs, const float* cs, hipStream_t s) {
  scale_rc_kernel<<<nblocks(rows * cols, 256), 256, 0, s>>>(X, rows, cols, ld, rs, cs);
  HIP_CHECK(hipGetLastError());
}
__global__ void scale_kernel(float* X, long rows, long cols, long ld, float a) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long r = i / cols, c = i % cols;
  X[r * ld + c] *= a;
}
void k_scale(float* X, long rows, long cols, long ld, float a, hipStream_t s) {
  scale_kernel<<<nblocks(rows * cols, 256), 256, 0, s>>>(X, rows, cols, ld, a);
  HIP_CHECK(hipGetLastError());
}

__global__ void sub_kernel(const float* A, const float* B, float* D, long rows, long cols, long ld) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long o = (i / cols) * ld + i % cols;
  D[o] = A[o] - B[o];
}
void k_sub(const float* A, const float* B, float* D, long rows, long cols, long ld, hipStream_t s) {
  sub_kernel<<<nblocks(rows * cols, 256), 256, 0, s>>>(A, B, D, rows, cols, ld);
  HIP_CHECK(hipGetLastError());
}

__device__ __forceinline__ double block_reduce_sum(double v) {
  __shared__ double red[32];
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) red[w] = v;
  __syncthreads();
  int nw = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0;
  if (w == 0)
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
  return v;
}
__global__ void sumsq_kernel(const float* X, long rows, long cols, long ld, double* out) {
  double acc = 0;
  for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * cols; i += long(gridDim.x) * blockDim.x) {
    long r = i / cols, c = i % cols;
    double v = X[r * ld + c];
    acc += v * v;
  }
  acc = block_reduce_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}
void k_sumsq(const float* X, long rows, long cols, long ld, double* out, hipStream_t s) {
  sumsq_kernel<<<1024, 256, 0, s>>>(X, rows, cols, ld, out);
  HIP_CHECK(hipGetLastError());
}
__global__ void dot_kernel(const float* A, long lda, const float* B, long ldb, long rows, long cols, double* out) {
  double acc = 0;
  for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * cols; i += long(gridDim.x) * blockDim.x) {
    long r = i / cols, c = i % cols;
    acc += double(A[r * lda + c]) * double(B[r * ldb + c]);
  }
  acc = block_reduce_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}
void k_dot(const float* A, long lda, const float* B, long ldb, long rows, long cols, double* out, hipStream_t s) {
  dot_kernel<<<1024, 256, 0, s>>>(A, lda, B, ldb, rows, cols, out);
  HIP_CHECK(hipGetLastError());
}
__global__ void add_diag_kernel(float* H, long n, float a) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) H[i * n + i] += a;
}
void k_add_diag(float* H, long n, float a, hipStream_t s) {
  add_diag_kernel<<<nblocks(n, 256), 256, 0, s>>>(H, n, a);
  HIP_CHECK(hipGetLastError());
}
__global__ void diag_sum_kernel(const float* H, long n, double* out) {
  double acc = 0;
  for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += long(gridDim.x) * blockDim.x)
    acc += H[i * n + i];
  acc = block_reduce_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}
void k_diag_sum(const float* H, long n, double* out, hipStream_t s) {
  diag_sum_kernel<<<64, 256, 0, s>>>(H, n, out);
  HIP_CHECK(hipGetLastError());
}
__global__ void split_bf16_kernel(const float* X, long rows, long cols, long ld_in, uint16_t* hi, uint16_t* lo,
                                  long ld_out) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long r = i / cols, c = i % cols;
  float v = X[r * ld_in + c];
  uint16_t h = f2bf(v);
  hi[r * ld_out + c] = h;
  if (lo) lo[r * ld_out + c] = f2bf(v - bf2f(h));
}
void k_split_bf16(const float* X, long rows, long cols, long ld_in, uint16_t* hi, uint16_t* lo, long ld_out,
                  hipStream_t s) {
  split_bf16_kernel<<<nblocks(rows * cols, 256), 256, 0, s>>>(X, rows, cols, ld_in, hi, lo, ld_out);
  HIP_CHECK(hipGetLastError());
}
__global__ void diff_split_kernel(const float* A, const float* B, long rows, long cols, long ld, uint16_t* hi,
                                  uint16_t* lo, long ld_out) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long r = i / cols, c = i % cols;
  float v = A[r * ld + c] - B[r * ld + c];
  uint16_t h = f2bf(v);
  hi[r * ld_out + c] = h;
  if (lo) lo[r * ld_out + c] = f2bf(v - bf2f(h));
}
void k_diff_split_bf16(const float* A, const float* B, long rows, long cols, long ld, uint16_t* hi, uint16_t* lo,
                       long ld_out, hipStream_t s) {
  diff_split_kernel<<<nblocks(rows * cols, 256), 256, 0, s>>>(A, B, rows, cols, ld, hi, lo, ld_out);
  HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------- transpose
__global__ void transpose_kernel(const float* in, long rows, long cols, long ld_in, float* out, long ld_out) {
  __shared__ float tile[32][33];
  long c0 = long(blockIdx.x) * 32, r0 = long(blockIdx.y) * 32;
  int tx = threadIdx.x, ty = threadIdx.y;  // 32 x 8
  for (int k = ty; k < 32; k += 8) {
    long r = r0 + k, c = c0 + tx;
    if (r < rows && c < cols) tile[k][tx] = in[r * ld_in + c];
  }
  __syncthreads();
  for (int k = ty; k < 32; k += 8) {
    long c = c0 + k, r = r0 + tx;
    if (r < rows && c < cols) out[c * ld_out + r] = tile[tx][k];
  }
}
void k_transpose(const float* in, long rows, long cols, long ld_in, float* out, long ld_out, hipStream_t s) {
  dim3 grid(nblocks(cols, 32), nblocks(rows, 32));
  transpose_kernel<<<grid, dim3(32, 8), 0, s>>>(in, rows, cols, ld_in, out, ld_out);
  HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------- Hadamard
// stage 1: natural-order FWHT on each contiguous chunk of P (power of two <= 4096) entries
__global__ __launch_bounds__(256) void fwht_chunks_kernel(float* X, long rows, int n, long ld, int P, int K,
                                                          float scale) {
  __shared__ float v[4096];
  long row = blockIdx.x / K;
  int chunk = blockIdx.x % K;
  float* x = X + row * ld + long(chunk) * P;
  for (int i = threadIdx.x; i < P; i += blockDim.x) v[i] = x[i];
  __syncthreads();
  for (int h = 1; h < P; h <<= 1) {
    for (int k = threadIdx.x; k < P / 2; k += blockDim.x) {
      int i = (k / h) * 2 * h + (k % h);
      float a = v[i], b = v[i + h];
      v[i] = a + b;
      v[i + h] = a - b;
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < P; i += blockDim.x) x[i] = v[i] * scale;
}
// stage 2: y[k2*P + i] = sum_k had[k2][k] y[k*P + i]  (had = hadK or hadK^T), times scale
template <int K>
__global__ void had_mix_kernel(float* X, long rows, int P, long ld, const int8_t* had, float scale) {
  __shared__ int8_t hs[K * K];
  for (int i = threadIdx.x; i < K * K; i += blockDim.x) hs[i] = had[i];
  __syncthreads();
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * P) return;
  long row = idx / P;
  int i = int(idx % P);
  float* x = X + row * ld;
  float y[K];
#pragma unroll
  for (int k = 0; k < K; ++k) y[k] = x[long(k) * P + i];
#pragma unroll
  for (int k2 = 0; k2 < K; ++k2) {
    float s = 0;
#pragma unroll
    for (int k = 0; k < K; ++k) s += hs[k2 * K + k] > 0 ? y[k] : -y[k];
    x[long(k2) * P + i] = s * scale;
  }
}
__global__ void had_mix_generic_kernel(float* X, long rows, int P, int K, long ld, const int8_t* had, float scale,
                                       float* tmp) {
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * P) return;
  long row = idx / P;
  int i = int(idx % P);
  float* x = X + row * ld;
  float* t = tmp + idx * K;
  for (int k = 0; k < K; ++k) t[k] = x[long(k) * P + i];
  for (int k2 = 0; k2 < K; ++k2) {
    float s = 0;
    for (int k = 0; k < K; ++k) s += had[k2 * K + k] * t[k];
    x[long(k2) * P + i] = s * scale;
  }
}

static const int8_t* device_had(int K, bool transpose) {
  static std::mutex mu;
  static std::map<std::tuple<int, int, bool>, int8_t*> cache;
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  std::lock_guard<std::mutex> lk(mu);
  auto key = std::make_tuple(dev, K, transpose);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  const int8_t* h = had_table(K);
  QCHECK(h, "no hadamard table for K=" + std::to_string(K));
  std::vector<int8_t> m(size_t(K) * K);
  // row k2 of the mixing matrix: hadK[k2][k] (hadU) or hadK^T[k2][k] = hadK[k][k2] (hadUt)
  for (int k2 = 0; k2 < K; ++k2)
    for (int k = 0; k < K; ++k) m[size_t(k2) * K + k] = transpose ? h[k * K + k2] : h[k2 * K + k];
  int8_t* d = nullptr;
  HIP_CHECK(hipMalloc(&d, m.size()));
  HIP_CHECK(hipMemcpy(d, m.data(), m.size(), hipMemcpyHostToDevice));
  cache[key] = d;
  return d;
}

void k_hadamard_rows(float* X, long rows, int n, long ld, bool transpose, hipStream_t s) {
  int K = had_K(n);
  QCHECK(K > 0, "unsupported hadamard dim " + std::to_string(n));
  int P = n / K;
  QCHECK(P <= 4096, "hadamard pow2 part > 4096 unsupported");
  float sc = 1.0f / std::sqrt(float(n));
  fwht_chunks_kernel<<<unsigned(rows * K), 256, 0, s>>>(X, rows, n, ld, P, K, K == 1 ? sc : 1.0f);
  HIP_CHECK(hipGetLastError());
  if (K == 1) return;
  const int8_t* had = device_had(K, transpose);
  unsigned nb = nblocks(rows * P, 256);
  if (K == 12) had_mix_kernel<12><<<nb, 256, 0, s>>>(X, rows, P, ld, had, sc);
  else if (K == 28) had_mix_kernel<28><<<nb, 256, 0, s>>>(X, rows, P, ld, had, sc);
  else if (K == 20) had_mix_kernel<20><<<nb, 256, 0, s>>>(X, rows, P, ld, had, sc);
  else {
    float* tmp = nullptr;
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&tmp), size_t(rows) * P * K * sizeof(float), s));
    had_mix_generic_kernel<<<nb, 256, 0, s>>>(X, rows, P, K, ld, had, sc, tmp);
    HIP_CHECK(hipFreeAsync(tmp, s));
  }
  HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------- model forward
__global__ __launch_bounds__(256) void rmsnorm_kernel(const uint16_t* x, const uint16_t* w, uint16_t* out, int n,
                                                      float eps) {
  long row = blockIdx.x;
  const uint16_t* xr = x + row * n;
  double ss = 0;  // fp32 is what HF uses; the extra precision is harmless
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = bf2f(xr[i]);
    ss += double(v) * v;
  }
  ss = block_reduce_sum(ss);
  __shared__ float inv;
  if (threadIdx.x == 0) inv = rsqrtf(float(ss / n) + eps);
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    // HF MistralRMSNorm: weight * (x * rsqrt(var + eps)).to(bf16)
    float h = bf2f(f2bf(bf2f(xr[i]) * inv));
    out[row * n + i] = f2bf(h * bf2f(w[i]));
  }
}
void k_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* out, long rows, int n, float eps, hipStream_t s) {
  rmsnorm_kernel<<<unsigned(rows), 256, 0, s>>>(x, w, out, n, eps);
  HIP_CHECK(hipGetLastError());
}

// one thread per (token, head, d < hd/2): rotate pairs (d, d + hd/2)
__global__ void qkv_rope_kernel(const float* qkv, const float* cs, uint16_t* Qp, uint16_t* Kp, uint16_t* Vp, int T,
                                int nq, int nkv, int hd, float theta) {
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  int half = hd / 2;
  int nh = nq + 2 * nkv;
  long total = long(T) * nh * half;
  if (idx >= total) return;
  int d = int(idx % half);
  long rest = idx / half;
  int h = int(rest % nh);
  int t = int(rest / nh);
  long width = long(nh) * hd;
  long col = long(h) * hd + d;
  const float* row = qkv + long(t) * width;
  float a = bf2f(f2bf(row[col] * cs[col]));                  // q_proj output in bf16
  float b = bf2f(f2bf(row[col + half] * cs[col + half]));
  if (h >= nq + nkv) {  // v: no rope
    int hv = h - nq - nkv;
    uint16_t* o = Vp + (long(hv) * T + t) * hd;
    o[d] = f2bf(a);
    o[d + half] = f2bf(b);
    return;
  }
  float inv_freq = 1.0f / powf(theta, float(2 * d) / float(hd));
  float ang = float(t) * inv_freq;
  float c = bf2f(f2bf(cosf(ang))), sn = bf2f(f2bf(sinf(ang)));
  float o1 = a * c - b * sn;  // x[d] cos - x[d+half] sin
  float o2 = b * c + a * sn;  // x[d+half] cos + x[d] sin
  uint16_t* o = (h < nq) ? Qp + (long(h) * T + t) * hd : Kp + (long(h - nq) * T + t) * hd;
  o[d] = f2bf(o1);
  o[d + half] = f2bf(o2);
}
void k_qkv_rope(const float* qkv, const float* col_scale, uint16_t* Qp, uint16_t* Kp, uint16_t* Vp, int T, int nq,
                int nkv, int hd, float theta, hipStream_t s) {
  long total = long(T) * (nq + 2 * nkv) * (hd / 2);
  qkv_rope_kernel<<<nblocks(total, 256), 256, 0, s>>>(qkv, col_scale, Qp, Kp, Vp, T, nq, nkv, hd, theta);
  HIP_CHECK(hipGetLastError());
}

__global__ __launch_bounds__(256) void causal_softmax_kernel(const float* S, uint16_t* P, int T, float scale) {
  long r = blockIdx.x;
  int t = int(r % T);
  const float* s = S + r * T;
  uint16_t* p = P + r * T;
  float mx = -INFINITY;
  for (int i = threadIdx.x; i <= t; i += blockDim.x) mx = fmaxf(mx, s[i] * scale);
  __shared__ float red[8];
  for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = red[0];
  for (int k = 1; k < 8; ++k) mx = fmaxf(mx, red[k]);
  __syncthreads();
  float sum = 0;
  for (int i = threadIdx.x; i <= t; i += blockDim.x) sum += __expf(s[i] * scale - mx);
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  sum = 0;
  for (int k = 0; k < 8; ++k) sum += red[k];
  float inv = 1.0f / sum;
  for (int i = threadIdx.x; i < T; i += blockDim.x) p[i] = i <= t ? f2bf(__expf(s[i] * scale - mx) * inv) : 0;
}
void k_causal_softmax(const float* S, uint16_t* P, int nh, int T, float scale, hipStream_t s) {
  causal_softmax_kernel<<<unsigned(long(nh) * T), 256, 0, s>>>(S, P, T, scale);
  HIP_CHECK(hipGetLastError());
}

__global__ void heads_to_tokens_kernel(const float* O, uint16_t* out, int nh, int T, int hd, int h0, long ld) {
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= long(nh) * T * hd) return;
  int d = int(idx % hd);
  long rest = idx / hd;
  int t = int(rest % T);
  int h = int(rest / T);
  out[long(t) * ld + long(h0 + h) * hd + d] = f2bf(O[idx]);
}
void k_heads_to_tokens(const float* O, uint16_t* out, int nh, int T, int hd, int h0, long ld, hipStream_t s) {
  heads_to_tokens_kernel<<<nblocks(long(nh) * T * hd, 256), 256, 0, s>>>(O, out, nh, T, hd, h0, ld);
  HIP_CHECK(hipGetLastError());
}

__global__ void residual_add_kernel(uint16_t* x, const float* y, const float* cs, long rows, int n) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * n) return;
  float yb = bf2f(f2bf(y[i] * cs[i % n]));  // projection output in bf16, then bf16 residual add
  x[i] = f2bf(bf2f(x[i]) + yb);
}
void k_residual_add(uint16_t* x, const float* y, const float* col_scale, long rows, int n, hipStream_t s) {
  residual_add_kernel<<<nblocks(rows * n, 256), 256, 0, s>>>(x, y, col_scale, rows, n);
  HIP_CHECK(hipGetLastError());
}

__global__ void silu_mul_kernel(const float* gu, const float* cs, uint16_t* act, long rows, int F) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * F) return;
  long r = i / F;
  int c = int(i % F);
  float g = bf2f(f2bf(gu[r * 2 * F + c] * cs[c]));
  float u = bf2f(f2bf(gu[r * 2 * F + F + c] * cs[F + c]));
  float sg = bf2f(f2bf(g / (1.0f + __expf(-g))));
  act[i] = f2bf(sg * u);
}
void k_silu_mul(const float* gu, const float* col_scale, uint16_t* act, long rows, int F, hipStream_t s) {
  silu_mul_kernel<<<nblocks(rows * F, 256), 256, 0, s>>>(gu, col_scale, act, rows, F);
  HIP_CHECK(hipGetLastError());
}

}  // namespace qtip
