#pragma once
// QTIP 2-bit HYB format (bitshift trellis L=16, K=2, V=2, quantlut_sym with a 2^9 LUT, 16x16 tiles,
// "kernel" storage layout) as written by qtip_quant (QTIP-hfized llama layout, see
// qtip_quant/README.md "Output format"). Host side: format constants, the reference decoder, the
// Hadamard factorisation of the reference (hadK (x) H_2^p), a double-precision reference linear and
// a reference transformer block. Every function here mirrors github.com/Cornell-RelaxML/qtip
// (lib/utils/kernel_decompress.py, lib/codebook/bitshift.py, lib/utils/matmul_had.py,
// lib/linear/quantized_linear.py) and is validated against committed fixtures by tests/qtip_tests.cpp.
#include "checkpoint.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace behemoth::qtip {
inline constexpr int tile = 16;             // td_x = td_y
inline constexpr int words_per_tile = 32;   // 256 weights x 2 bits / 16
inline constexpr int lut_entries = 1024;    // sign-folded LUT: entry = (p >> 6) & 1023 of p = s(s+1)
inline constexpr int chunk = 1024;          // power-of-two Hadamard factor for every model dimension

// get_hadK(n): 12 (n = 12 * 2^p), 28 (n = 28 * 2^p) or 1 (power of two). Throws for any other n
// (only the model's dimensions 1024, 12288, 28672 are needed and supported).
int had_k(int n);
// hadK, K x K row-major (+1/-1), nullptr for K = 1. matmul_hadUt uses its transpose.
const int8_t *had_matrix(int k);
// One vector of length n, in place: x <- hadU_n(x) = (hadK (x) H_{n/K}) x / sqrt(n)
// (transpose = false, matmul_hadU) or hadUt_n(x) with hadK^T (transpose = true).
void hadamard(double *x, int n, bool transpose);

// quantlut_sym codebook of a tlut (F16 [512][2] raw bits). Entry e = sign << 9 | idx holds
// (sign ? -tlut[idx][0] : tlut[idx][0], tlut[idx][1]) exactly (fp16 values as float) and the
// device format: bf16x2 (low = component 0, high = component 1), RNE from the fp16 values.
struct Codebook {
    std::array<float, lut_entries> c0{}, c1{};
    std::array<uint32_t, lut_entries> bf16x2{};
};
Codebook make_codebook(std::span<const uint16_t> tlut_f16);
// LUT entry of a 16-bit trellis state.
inline int lut_entry(uint32_t state) {
    const uint32_t s = state & 0xffffU;
    return int(((s * (s + 1)) >> 6) & 1023U);
}

struct HostLinear {
    int in = 0, out = 0;           // full (unsharded) dimensions
    std::vector<uint16_t> trellis; // kernel layout, (out/16)*(in/16)*32 words
    std::vector<float> su, sv;     // SU [in] (+-1), SV [out] (sign * Wscale)
    std::vector<uint16_t> tlut;    // F16 raw [512][2]
    // Original, unrotated E4M3 weights and per-output-channel scales. Mutually
    // exclusive with trellis/SU/SV/tlut; FP8 projections stay frozen in QAT.
    std::vector<uint8_t> fp8;
    std::vector<float> fp8_scale;
    bool is_fp8() const { return !fp8.empty(); }
    // Kernel-layout word c (0..31) of tile (tile_row, tile_col):
    // trellis[(((i0*(in/32) + j0)*32 + c)*2 + j1)*2 + i1], tile_row = 2*i0+i1, tile_col = 2*j0+j1.
    uint16_t word(int tile_row, int tile_col, int c) const {
        const size_t i0 = size_t(tile_row / 2), i1 = size_t(tile_row % 2), j0 = size_t(tile_col / 2),
                     j1 = size_t(tile_col % 2), n2 = size_t(in / 32);
        return trellis[(((i0 * n2 + j0) * 32 + size_t(c)) * 2 + j1) * 2 + i1];
    }
    // Words of block-rows [block_row, block_row + count) (32 output rows each): contiguous.
    std::span<const uint16_t> block_rows(int block_row, int count) const {
        const size_t per = size_t(in / 32) * 128;
        return std::span<const uint16_t>(trellis).subspan(size_t(block_row) * per,
                                                          size_t(count) * per);
    }
};
// Rotated, Wscale-free weight hatWr rows [row0, row0 + rows) (row-major rows x in), exactly
// decode_compressed(16, 9, 2, 1, m, n, trellis, lut) of the reference with fp16 LUT values.
void decode_rows(const HostLinear &, const Codebook &, int row0, int rows, float *w);
// The same rows with the device (bf16) LUT values.
void decode_rows_bf16(const HostLinear &, const Codebook &, int row0, int rows, float *w);
// y = SV (.) hadU_out( hadUt_in(x (.) SU) . hatWr^T ) for rows x in inputs, in double precision
// (QuantizedLinear.forward without its fp16 internals). device_rounding = true emulates the runtime:
// bf16 rotated input, bf16 LUT values, fp32-exact accumulation in double.
// round_z_bf16 additionally rounds the rotated output z to bf16 before hadU (the runtime's
// gate/up exchange format).
std::vector<float> reference_forward(const HostLinear &, std::span<const float> x,
                                     bool device_rounding = false, bool round_z_bf16 = false);
// Linear `name` (e.g. "self_attn.q_proj") of a layer of a QTIP checkpoint, full matrix.
HostLinear load_linear(const Checkpoint &, int layer, const std::string &name);
// The same from one layer file (model-layer-XXXXX.safetensors) without an assembled checkpoint.
HostLinear load_linear(const SafeTensorFile &, int layer, const std::string &name);
HostLinear load_fp8_linear(const SafeTensorFile &, int layer, const std::string &name);
HostLinear load_fp8_linear(const SafeTensorFile &weight, const SafeTensorFile &scale, int layer, const std::string &name);
// Reference transformer block of a QTIP checkpoint for one token at position zero (same
// semantics/rounding as reference_block for AQLM: bf16 norms, residual and projection outputs,
// FP8 attention reference), using decoded weights. Slow (decodes the full layer on the CPU).
std::vector<BFloat16> reference_block(const Checkpoint &, int layer,
                                      std::span<const BFloat16> input);
} // namespace behemoth::qtip
