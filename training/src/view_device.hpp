#pragma once
// Device element access for FP32 or BF16 (View::half) tensors.
#include "../../runtime/src/device_numeric.hpp"
#include "hip_ops.hpp"
namespace qat::gpu {
__device__ __forceinline__ float rnd(float x) {
    return __uint_as_float(uint32_t(behemoth::device::bf16(x)) << 16);
}
__device__ __forceinline__ float load(View a, size_t i) {
    return a.half ? __uint_as_float(uint32_t(reinterpret_cast<const uint16_t *>(a.p)[i]) << 16) : a.p[i];
}
__device__ __forceinline__ float load(View a, int r, int c) { return load(a, size_t(r) * a.ld + c); }
// BF16 bits of the element (rounding FP32 storage).
__device__ __forceinline__ uint16_t load_bits(View a, size_t i) {
    return a.half ? reinterpret_cast<const uint16_t *>(a.p)[i] : behemoth::device::bf16(a.p[i]);
}
__device__ __forceinline__ void store(View a, size_t i, float v) {
    if (a.half)
        reinterpret_cast<uint16_t *>(a.p)[i] = behemoth::device::bf16(v);
    else
        a.p[i] = v;
}
__device__ __forceinline__ void store(View a, int r, int c, float v) { store(a, size_t(r) * a.ld + c, v); }
// BF16-rounded value in either storage.
__device__ __forceinline__ void store_rounded(View a, size_t i, float v) {
    if (a.half)
        reinterpret_cast<uint16_t *>(a.p)[i] = behemoth::device::bf16(v);
    else
        a.p[i] = rnd(v);
}
} // namespace qat::gpu
