#pragma once
#include <cstdint>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if !defined(__gfx1201__)
#error "Behemoth kernels are specialized for gfx1201"
#endif
// Some ROCm clang releases do not define __AMDGCN_WAVEFRONT_SIZE. CMake
// enforces -mno-wavefrontsize64, and the matrix kernels assert rocWMMA's mode.
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "Behemoth requires Wave32, not Wave64"
#endif
#endif

namespace behemoth::device {
inline constexpr int wave_size = 32;
// OCP E4M3 only: max finite 448, signed zero. Never use *_fnuz on RDNA4.
__device__ inline float bf16(uint16_t x) {
    return __uint_as_float(uint32_t(x) << 16);
}
__device__ inline uint16_t bf16(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        return uint16_t((u >> 16) | 0x40U);
    return uint16_t((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}
__device__ inline uint8_t fp8(float x) {
    // Make our saturation policy independent of the HIP conversion intrinsic:
    // gfx1201 maps infinity to NaN even though finite overflow saturates.
    // Preserve signed NaNs; saturate signed infinities to the finite endpoints.
    const uint32_t bits = __float_as_uint(x);
    const uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude >= 0x7f800000U)
        return uint8_t(((bits >> 24) & 0x80U) | (magnitude == 0x7f800000U ? 0x7eU : 0x7fU));
    return __hip_fp8_e4m3(x).__x;
}
__device__ inline float fp8(uint8_t x) {
    __hip_fp8_e4m3 f;
    f.__x = x;
    return float(f);
}
__device__ inline float scale(float m) {
    return m == 0 ? 1.f : fmaxf(m / 448.f, 0x1p-126f);
}
} // namespace behemoth::device
