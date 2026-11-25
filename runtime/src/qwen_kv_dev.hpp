#pragma once
// Device helpers of the paged KV cache (QKvCache): bf16 or OCP e4m3 + one fp32 scale per (pool row, kv head).
#include <hip/hip_runtime.h>
#include <cstdint>

namespace behemoth {
namespace kvd {
typedef float f2v __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float bf(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
// 4 e4m3 bytes -> 4 floats (exact)
__device__ __forceinline__ void dec4(uint32_t w, float *f) {
    const f2v a = __builtin_amdgcn_cvt_pk_f32_fp8(int(w), false), b = __builtin_amdgcn_cvt_pk_f32_fp8(int(w), true);
    f[0] = a.x;
    f[1] = a.y;
    f[2] = b.x;
    f[3] = b.y;
}
// 8 consecutive elements starting at element offset `off` (multiple of 8) of a bf16 / e4m3 tensor
template <bool F8>
__device__ __forceinline__ void ld8(const void *base, long off, float *f) {
    if constexpr (F8) {
        const uint2 u = *reinterpret_cast<const uint2 *>(reinterpret_cast<const uint8_t *>(base) + off);
        dec4(u.x, f);
        dec4(u.y, f + 4);
    } else {
        const uint4 u = *reinterpret_cast<const uint4 *>(reinterpret_cast<const uint16_t *>(base) + off);
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            f[2 * e] = bf(uint16_t(w4[e]));
            f[2 * e + 1] = bf(uint16_t(w4[e] >> 16));
        }
    }
}
// one element (scalar fallback)
template <bool F8>
__device__ __forceinline__ float ld1(const void *base, long off) {
    if constexpr (F8) {
        const f2v a = __builtin_amdgcn_cvt_pk_f32_fp8(int(reinterpret_cast<const uint8_t *>(base)[off]), false);
        return a.x;
    } else {
        return bf(reinterpret_cast<const uint16_t *>(base)[off]);
    }
}
// float -> e4m3 byte (|x| <= 448: saturating RNE)
__device__ __forceinline__ uint8_t enc1(float x) {
    return uint8_t(__builtin_amdgcn_cvt_pk_fp8_f32(x, 0.f, 0, false) & 0xff);
}
// block-wide max of |x| (every thread of the block calls; result to all threads). red: >= 32 floats of LDS
__device__ __forceinline__ float block_amax(float x, float *red) {
    float v = fabsf(x);
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 32));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < int((blockDim.x + 31) >> 5); ++i) r = fmaxf(r, red[i]);
    __syncthreads();
    return r;
}
// store element d of the (row, head) vector x (one thread per element, the whole block = one vector)
// into a bf16 or e4m3 (+ scale) cache; idx = pool row * nkv + head, hd elements per vector
template <bool F8>
__device__ __forceinline__ void st_row(void *base, float *scales, long idx, int hd, int d, float x, float *red) {
    if constexpr (F8) {
        const float am = block_amax(x, red);
        const float inv = am > 0.f ? 448.f / am : 0.f;
        reinterpret_cast<uint8_t *>(base)[idx * hd + d] = enc1(x * inv);
        if (d == 0) scales[idx] = am * (1.f / 448.f);
    } else {
        reinterpret_cast<uint16_t *>(base)[idx * hd + d] = tobf(x);
    }
}
}  // namespace kvd
}  // namespace behemoth
