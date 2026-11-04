#pragma once
#include "device_ops.hpp"
namespace behemoth::cuda_runtime {
using Stream = glm53::gpu::Stream;
using B16 = glm53::gpu::B16;
struct QtipMatrix {
    const uint16_t *trellis;
    const float *su, *sv;
    const uint32_t *lut;
    int in, out;
};
void initialize_tables();
// z requires out * splits floats; scratch and rotated keep their original sizes.
// Split partials are reduced before the optional BF16 rounding of the rotated output.
void qtip_forward(QtipMatrix, const B16 *, B16 *, float *scratch, B16 *rotated, float *z, bool round_z,
                  Stream, int splits = 1);
void fp8_forward(const uint8_t *, const float *, const B16 *, B16 *, int in, int out, Stream);
void embedding(const B16 *, B16 *, int token, int hidden, Stream);
void residual(B16 *, const B16 *, int n, Stream);
void rope(B16 *, B16 *, int position, int qheads, int kvheads, int dim, float theta, Stream);
void gqa(const B16 *, const B16 *, const B16 *, B16 *, float *scores, int tokens, int qheads, int kvheads,
         int dim, Stream);
} // namespace behemoth::cuda_runtime
