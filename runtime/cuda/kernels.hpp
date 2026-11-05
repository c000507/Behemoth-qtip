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
void qtip_prepare(QtipMatrix, const B16 *, float *scratch, B16 *rotated, Stream, int tokens = 1,
                  bool rank_major = false);
void qtip_multiply(QtipMatrix, const B16 *rotated, float *z, int stride, int splits, Stream, int tokens = 1);
void qtip_finish(QtipMatrix, const float *z, B16 *, float *scratch, bool round_z, int splits, Stream,
                 int tokens = 1, bool rank_major = false);
void qtip_finish_b16(QtipMatrix, const B16 *, B16 *, float *scratch, int tokens, Stream, bool rank_major);
// z requires out * splits floats; scratch and rotated keep their original sizes.
// Split partials are reduced before the optional BF16 rounding of the rotated output.
void qtip_forward(QtipMatrix, const B16 *, B16 *, float *scratch, B16 *rotated, float *z, bool round_z,
                  Stream, int splits = 1);
void fp8_forward(const uint8_t *, const float *, const B16 *, B16 *, int in, int out, Stream);
// NVFP4 weight-only projection (E2M1 pairs, E4M3 block scales / 16 input channels, F32 global scale).
void nvfp4_forward(const uint8_t *packed, const uint8_t *block_scales, float global, const B16 *, B16 *, int in,
                   int out, Stream);
void embedding(const B16 *, B16 *, int token, int hidden, Stream);
void residual(B16 *, const B16 *, int n, Stream);
void rope(B16 *, B16 *, int position, int qheads, int kvheads, int dim, float theta, Stream);
// llama3 rope_scaling: inv_freq [dim/2] precomputed on the host (llama3_inv_freq)
void rope_scaled(B16 *, B16 *, int position, int qheads, int kvheads, int dim, const float *inv_freq, Stream);
void gqa(const B16 *, const B16 *, const B16 *, B16 *, float *scores, int tokens, int qheads, int kvheads,
         int dim, Stream);
} // namespace behemoth::cuda_runtime
