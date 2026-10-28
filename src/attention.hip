// Quantizer attention: 16-query / 32-key online-softmax WMMA tiles.
// Adapted from training/src/flash.hip's native forward, for head-major Q/K/V.
// Only Q/K/V and the final output live in global memory: no T-by-T workspace.
#include "gpu.hpp"

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
}  // namespace qtip
