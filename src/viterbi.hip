// QTIP bitshift-trellis Viterbi (L=16, K=2, V=2, HYB quantlut_sym) and LDLQ helper kernels for gfx1201 (wave32).
//
// Viterbi layout: one 1024-thread workgroup per 256-weight sequence (128 trellis steps, 2^16 states).
// Thread t owns the 64 states s = (d << 12) | (4t + q), d = 0..15, q = 0..3, i.e. the 16 predecessor candidates of
// the 4 successor groups j' = 4t + q (a group j collects the states with s & 0xfff == j; all states s' with
// s' >> 4 == j share the same best predecessor). The thread keeps its 64 fp16 LUT pairs and |lut|^2 in registers;
// per step it reads 16 predecessor-group minima M[(d << 8) | (t >> 2)] from LDS, forms the 64 costs with
// v_dot2_f32_f16, takes the 4 group minima with the predecessor index d tagged into the low 4 mantissa bits, writes
// them to LDS (double-buffered, one barrier per step) and stores the 4 x 4-bit argmins (one uint16) to global
// memory for the backtrace.
#include <hip/hip_runtime.h>

#include "gpu.hpp"

namespace qtip {

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
constexpr float kBig = 1e30f;

// ldlq._PERMUTE: p -> tile index q (= 16 r + c);  arange(256).reshape(2,8,2,4,2).permute(1,3,2,0,4)
__device__ __forceinline__ int perm_fwd(int p) {
  int e = p & 1, a = (p >> 1) & 1, c = (p >> 2) & 1, d = (p >> 3) & 3, b = (p >> 5) & 7;
  return a * 128 + b * 16 + c * 8 + d * 2 + e;
}
__device__ __forceinline__ int perm_inv(int q) {
  int a = q >> 7, b = (q >> 4) & 7, c = (q >> 3) & 1, d = (q >> 1) & 3, e = q & 1;
  return (((b * 4 + d) * 2 + c) * 2 + a) * 2 + e;
}

__global__ __launch_bounds__(1024) void viterbi_fwd_kernel(const float* __restrict__ X, const int* __restrict__ overlap,
                                                           int roll, uint16_t* __restrict__ A,
                                                           uint16_t* __restrict__ final_state,
                                                           const uint32_t* __restrict__ lut_h2) {
  __shared__ float M[2][4096];
  __shared__ float xs[256];
  __shared__ float rv[32];
  __shared__ uint32_t rs[32];
  const int b = blockIdx.x, t = threadIdx.x, g = t >> 2;
  const float* xb = X + size_t(b) * 256;
  if (t < 256) xs[t] = float(_Float16(roll ? xb[(t + 128) & 255] : xb[t]));  // reference quantizes in fp16

  h2_t l[16][4];
  float nrm[16][4];
#pragma unroll
  for (int d = 0; d < 16; ++d)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t u = lut_h2[(d << 12) | (t << 2) | q];
      h2_t v = __builtin_bit_cast(h2_t, u);
      l[d][q] = v;
      float a = float(v.x), c = float(v.y);
      nrm[d][q] = a * a + c * c;
    }
  const int ov = overlap ? overlap[b] : -1;
  for (int k = t; k < 4096; k += 1024) M[0][k] = (ov < 0 || k == ov) ? 0.f : kBig;
  __syncthreads();

  uint16_t* Ab = A + size_t(b) * 127 * 1024;
  int cur = 0;
  for (int i = 0; i < 127; ++i) {
    const float x0 = xs[2 * i], x1 = xs[2 * i + 1];
    const h2_t xm2 = {_Float16(-2.f * x0), _Float16(-2.f * x1)};
    const float xx = x0 * x0 + x1 * x1;
    float best[4] = {3e30f, 3e30f, 3e30f, 3e30f};
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      const float mp = M[cur][(d << 8) | g] + xx;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float c = __builtin_amdgcn_fdot2(l[d][q], xm2, nrm[d][q] + mp, false);
        uint32_t tg = (__float_as_uint(c) & ~15u) | uint32_t(d);
        best[q] = fminf(best[q], __uint_as_float(tg));
      }
    }
    uint32_t pk = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t bb = __float_as_uint(best[q]);
      pk |= (bb & 15u) << (4 * q);
      M[cur ^ 1][(t << 2) | q] = __uint_as_float(bb & ~15u);
    }
    Ab[i * 1024 + t] = uint16_t(pk);
    __syncthreads();
    cur ^= 1;
  }
  // last step: full argmin with the tail-biting end constraint (state & 0xfff == overlap)
  {
    const float x0 = xs[254], x1 = xs[255];
    const h2_t xm2 = {_Float16(-2.f * x0), _Float16(-2.f * x1)};
    const float xx = x0 * x0 + x1 * x1;
    float bv = 3e30f;
    uint32_t bs = 0;
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      const float mp = M[cur][(d << 8) | g] + xx;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        int j = (t << 2) | q;
        if (ov >= 0 && j != ov) continue;
        float c = __builtin_amdgcn_fdot2(l[d][q], xm2, nrm[d][q] + mp, false);
        uint32_t s = (uint32_t(d) << 12) | uint32_t(j);
        if (c < bv || (c == bv && s < bs)) bv = c, bs = s;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float ov2 = __shfl_xor(bv, off, 32);
      uint32_t os = __shfl_xor(bs, off, 32);
      if (ov2 < bv || (ov2 == bv && os < bs)) bv = ov2, bs = os;
    }
    if ((t & 31) == 0) rv[t >> 5] = bv, rs[t >> 5] = bs;
    __syncthreads();
    if (t < 32) {
      bv = rv[t], bs = rs[t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float ov2 = __shfl_xor(bv, off, 32);
        uint32_t os = __shfl_xor(bs, off, 32);
        if (ov2 < bv || (ov2 == bv && os < bs)) bv = ov2, bs = os;
      }
      if (t == 0) final_state[b] = uint16_t(bs);
    }
  }
}

__device__ __forceinline__ uint32_t prev_state(const uint16_t* Ab, int i, uint32_t s) {
  uint32_t j = s >> 4;
  uint32_t w = Ab[i * 1024 + (j >> 2)];
  uint32_t d = (w >> (4 * (j & 3))) & 15u;
  return j | (d << 12);
}

__global__ void viterbi_overlap_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ final_state, int B,
                                       int* __restrict__ overlap) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const uint16_t* Ab = A + size_t(b) * 127 * 1024;
  uint32_t s = final_state[b];
  for (int i = 126; i >= 64; --i) s = prev_state(Ab, i, s);  // s = state at step i
  overlap[b] = int(s >> 4);
}

__global__ void viterbi_finish_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ final_state, int B,
                                      int b0, uint16_t* __restrict__ Q, int nblk, int blk, float* __restrict__ hatWT,
                                      long m, int col0, const uint32_t* __restrict__ lut_h2) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const uint16_t* Ab = A + size_t(b) * 127 * 1024;
  const long row0 = long(b0 + b) * 16;  // first W row of this trellis
  uint16_t* q = Q ? Q + (size_t(b0 + b) * nblk + blk) * 128 : nullptr;
  uint32_t s = final_state[b];
  for (int i = 127; i >= 0; --i) {
    if (i < 127) s = prev_state(Ab, i, s);
    if (q) q[i] = uint16_t(s);
    if (hatWT) {
      h2_t v = __builtin_bit_cast(h2_t, lut_h2[s]);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        int tq = perm_fwd(2 * i + e);  // tile index 16 r + c
        int r = tq >> 4, c = tq & 15;
        hatWT[long(col0 + c) * m + row0 + r] = float(e ? v.y : v.x);
      }
    }
  }
}

__global__ void decode_states_kernel(const uint16_t* __restrict__ st, int B, float* __restrict__ hat,
                                     const uint32_t* __restrict__ lut_h2) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= long(B) * 128) return;
  h2_t v = __builtin_bit_cast(h2_t, lut_h2[st[i]]);
  hat[2 * i] = float(v.x);
  hat[2 * i + 1] = float(v.y);
}

// one thread per output column j of W (= column of W_T), 16 block columns, <= 112 feedback rows
__global__ __launch_bounds__(256) void ldlq_prep_kernel(const float* __restrict__ WT, const float* __restrict__ hatWT,
                                                        const float* __restrict__ prod, const float* __restrict__ L,
                                                        long n, long m, int col0, int bufend, float* __restrict__ X) {
  __shared__ float Ls[112][16];
  const int k0 = col0 + 16, nk = bufend - k0;
  for (int idx = threadIdx.x; idx < nk * 16; idx += blockDim.x) {
    int kk = idx >> 4, c = idx & 15;
    Ls[kk][c] = L[long(k0 + kk) * n + col0 + c];
  }
  __syncthreads();
  long j = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= m) return;
  float acc[16];
#pragma unroll
  for (int c = 0; c < 16; ++c) acc[c] = WT[long(col0 + c) * m + j] + prod[long(col0 + c) * m + j];
  for (int kk = 0; kk < nk; ++kk) {
    long o = long(k0 + kk) * m + j;
    float dk = WT[o] - hatWT[o];
#pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] = fmaf(Ls[kk][c], dk, acc[c]);
  }
  long bt = j >> 4;
  int r = int(j & 15);
  float* xb = X + bt * 256;
#pragma unroll
  for (int c = 0; c < 16; ++c) xb[perm_inv(r * 16 + c)] = acc[c];
}

void k_ldlq_prep(const float* WT, const float* hatWT, const float* prod, const float* L, long n, long m, int col0,
                 int bufend, float* X, hipStream_t s) {
  QCHECK(bufend - col0 - 16 <= 112 && bufend - col0 - 16 >= 0, "ldlq_prep buffer");
  ldlq_prep_kernel<<<dim3(unsigned((m + 255) / 256)), dim3(256), 0, s>>>(WT, hatWT, prod, L, n, m, col0, bufend, X);
  HIP_CHECK(hipGetLastError());
}
void k_viterbi_fwd(const float* X, const int* overlap, int roll, int B, uint16_t* A, uint16_t* final_state,
                   const uint32_t* lut_h2, hipStream_t s) {
  viterbi_fwd_kernel<<<dim3(unsigned(B)), dim3(1024), 0, s>>>(X, overlap, roll, A, final_state, lut_h2);
  HIP_CHECK(hipGetLastError());
}
void k_viterbi_overlap(const uint16_t* A, const uint16_t* final_state, int B, int* overlap, hipStream_t s) {
  viterbi_overlap_kernel<<<dim3(unsigned((B + 63) / 64)), dim3(64), 0, s>>>(A, final_state, B, overlap);
  HIP_CHECK(hipGetLastError());
}
void k_viterbi_finish(const uint16_t* A, const uint16_t* final_state, int B, int b0, uint16_t* Q, int nblk, int blk,
                      float* hatWT, long m, int col0, const uint32_t* lut_h2, hipStream_t s) {
  viterbi_finish_kernel<<<dim3(unsigned((B + 63) / 64)), dim3(64), 0, s>>>(A, final_state, B, b0, Q, nblk, blk, hatWT,
                                                                          m, col0, lut_h2);
  HIP_CHECK(hipGetLastError());
}
void k_decode_states(const uint16_t* states, int B, float* hat, const uint32_t* lut_h2, hipStream_t s) {
  long n = long(B) * 128;
  decode_states_kernel<<<dim3(unsigned((n + 255) / 256)), dim3(256), 0, s>>>(states, B, hat, lut_h2);
  HIP_CHECK(hipGetLastError());
}

}  // namespace qtip
