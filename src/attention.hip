// Quantizer attention: 16-query / 32-key online-softmax WMMA tiles.
// Adapted from training/src/flash.hip's native forward, for head-major Q/K/V.
// Only Q/K/V and the final output live in global memory: no T-by-T workspace.
// HIP path: gfx12 rocWMMA-style intrinsics; CUDA path: nvcuda::wmma (sm_80+), same schedule.
#include "gpu.hpp"

#ifndef QTIP_CUDA
#include "../runtime/src/device_numeric.hpp"
#include <algorithm>
#include <cmath>

namespace qtip {
namespace {
using Reg = __bf16 __attribute__((ext_vector_type(8)));
using Acc = float __attribute__((ext_vector_type(8)));
using behemoth::device::bf16;
__device__ __forceinline__ Acc mma(Reg a, Reg b, Acc c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ int kbase(int s, int h) { return 32 * (s >> 1) + 16 * h + 8 * (s & 1); }

template <int D>
__global__ void attention_kernel(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
                                 int T, int nq, int group, int hd, float scale) {
  constexpr int KROW = D * 2, VROW = 72, KBYTES = 32 * KROW;
  __shared__ __align__(16) unsigned char shared[KBYTES + D * VROW];
  int tid = threadIdx.x, wave = tid / 32, c = tid % 16, h = tid % 32 / 16;
  int first = blockIdx.x * 16, query = first + c, kh = blockIdx.y;
  int head0 = blockIdx.z * (blockDim.x / 32), head = kh * group + head0 + wave;
  int count = min(int(blockDim.x / 32), group - head0);
  bool live = query < T && wave < count;
  Reg qr[D / 16]{};
  // Reuse the K/V tile storage to load four query heads at a time, then keep Q
  // in registers. D pads small/non-power-of-two head dimensions with zero.
  for (int lo = 0; lo < count; lo += 4) {
    int heads = min(4, count - lo);
    for (int i = tid; i < heads * 16 * D; i += blockDim.x) {
      int w = i / (16 * D), row = i / D % 16, d = i % D;
      reinterpret_cast<uint16_t*>(shared)[i] = first + row < T && d < hd
          ? q[(size_t(kh * group + head0 + lo + w) * T + first + row) * hd + d] : 0;
    }
    __syncthreads();
    if (wave >= lo && wave < lo + heads) {
      const unsigned char* row = shared + ((wave - lo) * 16 + c) * D * 2;
#pragma unroll
      for (int s = 0; s < D / 16; ++s)
        qr[s] = __builtin_bit_cast(Reg, *reinterpret_cast<const uint4*>(row + kbase(s, h) * 2));
    }
    __syncthreads();
  }
  Acc acc[D / 16]{};
  float mx = -INFINITY, den = 0;
  int limit = min(first + 16, T);
  for (int base = 0; base < limit; base += 32) {
    for (int i = tid; i < 32 * D; i += blockDim.x) {
      int key = i / D, d = i % D, t = base + key;
      size_t index = (size_t(kh) * T + t) * hd + d;
      uint16_t key_value = t < limit && d < hd ? k[index] : 0;
      uint16_t value = t < limit && d < hd ? v[index] : 0;
      *reinterpret_cast<uint16_t*>(shared + key * KROW + ((d * 2) ^ ((key & 7) * 16))) = key_value;
      *reinterpret_cast<uint16_t*>(shared + KBYTES + d * VROW + key * 2) = value;
    }
    __syncthreads();
    Acc scores[2]{};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      int key = 16 * (c >> 3) + 8 * n + (c & 7);
      const unsigned char* row = shared + key * KROW;
#pragma unroll
      for (int s = 0; s < D / 16; ++s) {
        auto kr = __builtin_bit_cast(
            Reg, *reinterpret_cast<const uint4*>(row + ((kbase(s, h) * 2) ^ ((key & 7) * 16))));
        scores[n] = mma(kr, qr[s], scores[n]);
      }
    }
    float top = -INFINITY;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int key = base + 16 * h + 8 * n + i;
        float score = live && key <= query ? scores[n][i] * scale : -INFINITY;
        scores[n][i] = score;
        top = fmaxf(top, score);
      }
    top = fmaxf(top, __shfl_xor(top, 16));
    float next = fmaxf(mx, top);
    float alpha = mx == next ? 1.f : __builtin_amdgcn_exp2f(mx - next);
    float shift = isfinite(next) ? next : 0.f;
    Reg pr[2];
    float sum = 0;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      uint16_t bits[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float p = __builtin_amdgcn_exp2f(scores[n][i] - shift);
        sum += p;
        bits[i] = bf16(p);
      }
      pr[n] = __builtin_bit_cast(Reg, bits);
    }
    sum += __shfl_xor(sum, 16);
    den = den * alpha + sum;
    mx = next;
#pragma unroll
    for (int nd = 0; nd < D / 16; ++nd) {
      Acc value = acc[nd];
#pragma unroll
      for (int i = 0; i < 8; ++i) value[i] *= alpha;
      const unsigned char* row = shared + KBYTES + (16 * nd + c) * VROW;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const unsigned char* p = row + (16 * h + 8 * n) * 2;
        uint2 lo = *reinterpret_cast<const uint2*>(p), hi = *reinterpret_cast<const uint2*>(p + 8);
        value = mma(__builtin_bit_cast(Reg, make_uint4(lo.x, lo.y, hi.x, hi.y)), pr[n], value);
      }
      acc[nd] = value;
    }
    __syncthreads();
  }
  if (live) {
    float inv = 1.f / den;
#pragma unroll
    for (int nd = 0; nd < D / 16; ++nd)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int d = 16 * nd + 8 * h + i;
        if (d < hd) out[(size_t(query) * nq + head) * hd + d] = bf16(acc[nd][i] * inv);
      }
  }
}
}  // namespace

void k_tiled_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
                        int T, int nq, int nkv, int hd, hipStream_t s) {
  QCHECK(q && k && v && out && T > 0 && nq > 0 && nkv > 0 && nq % nkv == 0 && hd > 0 && hd <= 256,
         "invalid tiled attention shape (head_dim must be <=256)");
  int group = nq / nkv, waves = std::min(group, 16);
  dim3 grid(unsigned((T + 15) / 16), unsigned(nkv), unsigned((group + waves - 1) / waves));
  float scale = (1.f / std::sqrt(float(hd))) * 1.4426950408889634f;
  if (hd <= 128)
    attention_kernel<128><<<grid, waves * 32, 0, s>>>(q, k, v, out, T, nq, group, hd, scale);
  else
    attention_kernel<256><<<grid, waves * 32, 0, s>>>(q, k, v, out, T, nq, group, hd, scale);
  HIP_CHECK(hipGetLastError());
}
#else
// ------------------------------------------------------------------ CUDA (sm_80+) wmma port
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>

namespace qtip {
namespace {
namespace wmma = nvcuda::wmma;
using bf16_t = __nv_bfloat16;

__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

// One warp per query head, 16 queries x 32 keys per iteration, online softmax (fp32).
// WMMA fragment element mapping for m16n16k16 (f32 accumulator and row-major bf16 A alike;
// HMMA.16816 lane layout, stable since Volta): element i (0..7) of x belongs to
//   row = (lane >> 2) + 8 * ((i >> 1) & 1),  col = (lane & 3) * 2 + (i & 1) + 8 * (i >> 2).
// A row (query) is therefore spread over the 4 lanes of a lane>>2 group: the per-row max/sum
// reductions below run over __shfl_xor_sync with offsets 1 and 2.
template <int D>
__global__ __launch_bounds__(256) void attention_kernel(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                                         uint16_t* out, int T, int nq, int group, int hd,
                                                         float scale) {
  extern __shared__ __align__(16) uint16_t smem[];  // K[32][D] | V[32][D] | Q staging [4][16][D]
  uint16_t* Ks = smem;
  uint16_t* Vs = Ks + 32 * D;
  uint16_t* Qs = Vs + 32 * D;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int first = blockIdx.x * 16;
  const int kh = blockIdx.y;
  const int head0 = blockIdx.z * (int(blockDim.x) >> 5);
  const int head = kh * group + head0 + wave;
  const int count = min(int(blockDim.x) >> 5, group - head0);
  const bool live = wave < count;
  const int q0 = first + (lane >> 2);  // query row of fragment elements with ((i >> 1) & 1) == 0
  const int q1 = q0 + 8;               // ... == 1
  const bool l0 = live && q0 < T, l1 = live && q1 < T;

  // Stage the Q tile through shared memory (4 heads at a time), then keep it in registers.
  // D pads small / non-power-of-two head dimensions with zero.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> qf[D / 16];
  for (int lo = 0; lo < count; lo += 4) {
    const int heads = min(4, count - lo);
    for (int i = tid; i < heads * 16 * D; i += blockDim.x) {
      int w = i / (16 * D), row = i / D % 16, d = i % D;
      Qs[i] = first + row < T && d < hd ? q[(size_t(kh * group + head0 + lo + w) * T + first + row) * hd + d] : 0;
    }
    __syncthreads();
    if (wave >= lo && wave < lo + heads)
#pragma unroll
      for (int s = 0; s < D / 16; ++s)
        wmma::load_matrix_sync(qf[s], reinterpret_cast<const bf16_t*>(Qs + (wave - lo) * 16 * D + 16 * s), D);
    __syncthreads();
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[D / 16];
#pragma unroll
  for (int nd = 0; nd < D / 16; ++nd) wmma::fill_fragment(acc[nd], 0.f);
  float mx0 = -INFINITY, mx1 = -INFINITY, den0 = 0, den1 = 0;
  const int limit = min(first + 16, T);
  for (int base = 0; base < limit; base += 32) {
    for (int i = tid; i < 32 * D; i += blockDim.x) {
      int key = i / D, d = i % D, t = base + key;
      bool ok = t < limit && d < hd;
      size_t idx = (size_t(kh) * T + t) * hd + d;
      Ks[i] = ok ? k[idx] : 0;
      Vs[i] = ok ? v[idx] : 0;
    }
    __syncthreads();
    // scores[16 queries][32 keys] = Q K^T, two 16x16 accumulators (keys 0..15 / 16..31)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc[2];
    wmma::fill_fragment(sc[0], 0.f);
    wmma::fill_fragment(sc[1], 0.f);
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int s = 0; s < D / 16; ++s) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::col_major> bf;
        wmma::load_matrix_sync(bf, reinterpret_cast<const bf16_t*>(Ks + 16 * n * D + 16 * s), D);  // B[d][key] = K[key][d]
        wmma::mma_sync(sc[n], qf[s], bf, sc[n]);
      }
    // causal mask + log2 scale; per-thread row maxima, then merge over the 4 lanes holding the row
    float top0 = -INFINITY, top1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int key = base + 16 * n + ((lane & 3) * 2 + (i & 1) + 8 * (i >> 2));
        bool r1 = (i >> 1) & 1;
        float s = (r1 ? l1 && key <= q1 : l0 && key <= q0) ? sc[n].x[i] * scale : -INFINITY;
        sc[n].x[i] = s;
        if (r1) top1 = fmaxf(top1, s);
        else top0 = fmaxf(top0, s);
      }
    top0 = fmaxf(top0, __shfl_xor_sync(0xffffffffu, top0, 1));
    top0 = fmaxf(top0, __shfl_xor_sync(0xffffffffu, top0, 2));
    top1 = fmaxf(top1, __shfl_xor_sync(0xffffffffu, top1, 1));
    top1 = fmaxf(top1, __shfl_xor_sync(0xffffffffu, top1, 2));
    const float nx0 = fmaxf(mx0, top0), nx1 = fmaxf(mx1, top1);
    const float a0 = mx0 == nx0 ? 1.f : exp2f(mx0 - nx0);
    const float a1 = mx1 == nx1 ? 1.f : exp2f(mx1 - nx1);
    const float sh0 = isfinite(nx0) ? nx0 : 0.f, sh1 = isfinite(nx1) ? nx1 : 0.f;
    float p[2][8], sum0 = 0, sum1 = 0;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        bool r1 = (i >> 1) & 1;
        float s = sc[n].x[i];
        p[n][i] = isfinite(s) ? exp2f(s - (r1 ? sh1 : sh0)) : 0.f;
        if (r1) sum1 += p[n][i];
        else sum0 += p[n][i];
      }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
    den0 = den0 * a0 + sum0, mx0 = nx0;
    den1 = den1 * a1 + sum1, mx1 = nx1;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16_t, wmma::row_major> pf[2];
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int i = 0; i < 8; ++i) pf[n].x[i] = __float2bfloat16_rn(p[n][i]);
    // acc <- P V; rescale the old rows by their per-query alpha
#pragma unroll
    for (int nd = 0; nd < D / 16; ++nd)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[nd].x[i] *= ((i >> 1) & 1) ? a1 : a0;
#pragma unroll
    for (int nd = 0; nd < D / 16; ++nd)
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16_t, wmma::row_major> bf;
        wmma::load_matrix_sync(bf, reinterpret_cast<const bf16_t*>(Vs + 16 * n * D + 16 * nd), D);  // B[key][d] = V[key][d]
        wmma::mma_sync(acc[nd], pf[n], bf, acc[nd]);
      }
    __syncthreads();
  }
  if (live) {
    const float i0 = 1.f / den0, i1 = 1.f / den1;
#pragma unroll
    for (int nd = 0; nd < D / 16; ++nd)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int d = 16 * nd + (lane & 3) * 2 + (i & 1) + 8 * (i >> 2);
        bool r1 = (i >> 1) & 1;
        int query = r1 ? q1 : q0;
        if (d < hd && query < T)
          out[(size_t(query) * nq + head) * hd + d] = f2bf(acc[nd].x[i] * (r1 ? i1 : i0));
      }
  }
}

template <int D>
void launch_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, int T, int nq,
                      int group, int hd, float scale, dim3 grid, int waves, hipStream_t s) {
  constexpr int smem_bytes = (32 * D + 32 * D + 4 * 16 * D) * int(sizeof(uint16_t));
  static bool smem_attr = [] {
    return cudaFuncSetAttribute(attention_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes) == cudaSuccess;
  }();
  (void)smem_attr;
  attention_kernel<D><<<grid, waves * 32, smem_bytes, s>>>(q, k, v, out, T, nq, group, hd, scale);
}
}  // namespace

void k_tiled_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
                        int T, int nq, int nkv, int hd, hipStream_t s) {
  QCHECK(q && k && v && out && T > 0 && nq > 0 && nkv > 0 && nq % nkv == 0 && hd > 0 && hd <= 256,
         "invalid tiled attention shape (head_dim must be <=256)");
  // <= 8 warps (256 threads): the D=256 register footprint (qf + acc fragments) needs 255 regs/thread,
  // which fits an SM only up to 256 threads
  int group = nq / nkv, waves = std::min(group, 8);
  dim3 grid(unsigned((T + 15) / 16), unsigned(nkv), unsigned((group + waves - 1) / waves));
  float scale = (1.f / std::sqrt(float(hd))) * 1.4426950408889634f;
  if (hd <= 128)
    launch_attention<128>(q, k, v, out, T, nq, group, hd, scale, grid, waves, s);
  else
    launch_attention<256>(q, k, v, out, T, nq, group, hd, scale, grid, waves, s);
  HIP_CHECK(hipGetLastError());
}
#endif  // QTIP_CUDA
}  // namespace qtip
