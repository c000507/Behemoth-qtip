#include "../../runtime/src/device_numeric.hpp"
#include "gpu.hpp"
#include "gpu_decode.hpp"
#include <hip/hip_fp16.h>
namespace glm53 {
namespace {
__global__ void decode(const uint8_t *input, const float *scales, const uint8_t *fp8_scales, uint16_t *out,
                       size_t count, int cols, int format, bool block, float global) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    float value;
    if (format == 0)
        value = reinterpret_cast<const float *>(input)[i];
    else if (format == 3)
        value = __half2float(reinterpret_cast<const __half *>(input)[i]);
    else if (format == 1) {
        size_t row = i / cols, col = i % cols;
        size_t scale = block ? (row / 128) * ((cols + 127) / 128) + col / 128 : row;
        value = behemoth::device::fp8(input[i]) * scales[scale];
    } else {
        int code = (input[i / 2] >> (4 * (i % 2))) & 15;
        constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        float magnitude = levels[code & 7];
        value = (code & 8 ? -magnitude : magnitude) * behemoth::device::fp8(fp8_scales[i / 16]) * global;
    }
    out[i] = behemoth::device::bf16(value);
}
} // namespace
void decode_weights(const void *input, const float *scales, const uint8_t *fp8_scales, uint16_t *out,
                    int rows, int cols, int format, bool block, float global, hipStream_t stream) {
    size_t count = size_t(rows) * cols;
    decode<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<const uint8_t *>(input), scales, fp8_scales,
                                                    out, count, cols, format, block, global);
    HIP_CHECK(hipGetLastError());
}
} // namespace glm53
