#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>
namespace glm53 {
// format: 0 F32, 1 scaled E4M3, 2 NVFP4, 3 IEEE F16. BF16 needs no conversion.
void decode_weights(const void *input, const float *scales, const uint8_t *fp8_scales, uint16_t *output,
                    int rows, int cols, int format, bool block_fp8, float global_scale, hipStream_t stream);
} // namespace glm53
