// CPU side of the QTIP algorithm: codebook (HYB quantlut_sym), Hadamard factorisation, reference Viterbi,
// trellis packing in the reference's on-disk layout, fp64 Cholesky / block LDL.
// Everything here mirrors github.com/Cornell-RelaxML/qtip (lib/codebook/bitshift.py, lib/utils/matmul_had.py,
// lib/utils/math_utils.py, quantize_llama/quantize_finetune_llama.py / lib/algo/finetune.py).
#pragma once
#include <cstdint>
#include <vector>

namespace qtip {

// Fixed config: the paper's 2-bit HYB code.
constexpr int kL = 16;          // trellis window (state bits)
constexpr int kK = 2;           // bits per weight
constexpr int kV = 2;           // weights per trellis step
constexpr int kKV = kK * kV;    // new bits per step (4)
constexpr int kTlutBits = 9;
constexpr int kTd = 16;         // tile 16x16 = one trellis
constexpr int kSeq = kTd * kTd; // 256 weights per trellis
constexpr int kSteps = kSeq / kV;  // 128 trellis steps
constexpr int kStates = 1 << kL;
constexpr int kGroups = 1 << (kL - kKV);  // 4096
constexpr int kWordsPerTrellis = kSeq * kK / 16;  // 32 uint16

struct Codebook {
  float tlut[512][2];          // k-means result (fp32)
  uint16_t tlut_h[512][2];     // stored (fp16) tlut
  std::vector<float> lut0, lut1;  // 65536-entry decoded lut (fp16 values as fp32), component 0/1
  std::vector<uint32_t> lut_h2;   // packed half2 (lo = comp0, hi = comp1)
  double lut_rms = 0;          // sqrt(mean(lut^2)) over the fp16 lut, fp64 (reference Wscale denominator)
};
const Codebook& codebook();

// _PERMUTE of lib/algo/ldlq.py (for_kernel layout): seq[p] = tile[kPerm[p]], tile index = r*16 + c
const int* perm256();
const int* inv_perm256();

// get_hadK(n): returns K (1 if n is a power of two), or 0 if unsupported.
int had_K(int n);
// In-place row transform of `rows` rows of length n (row stride ld): x <- (hadK^(T if transpose) (x) H_{n/K}) x / sqrt(n)
// transpose=false: matmul_hadU, transpose=true: matmul_hadUt. fp64 reference implementation.
void hadamard_rows_ref(double* X, long rows, int n, long ld, bool transpose);

// Exact reference Viterbi for one sequence of kSeq values (lib/codebook/bitshift.py viterbi), fp64 costs.
// overlap < 0: unconstrained; else tail-biting constraint as in the reference (start state >> 4 == overlap,
// end state & 0xfff == overlap).  states: kSteps entries.  Returns the optimal cost.
double viterbi_ref(const float* X, int overlap, uint16_t* states);
// Full tail-biting quantization of one 256-sequence (bitshift_codebook.quantize): rolled pass, overlap, pass 2.
double quantize_seq_ref(const float* X, uint16_t* states, float* hatX);
// Reconstruct a sequence from states (decode)
void decode_states(const uint16_t* states, float* hatX);
// Cost of a given state path on X (sum of squared errors), fp64
double path_cost(const float* X, const uint16_t* states);

// pack_trellis (bitshift.py) for B trellises of kSteps states -> B x 32 uint16 words
void pack_trellis(const uint16_t* states, long B, uint16_t* packed);
// unpack_trellis -> states
void unpack_trellis(const uint16_t* packed, long B, uint16_t* states);
// Kernel-layout swizzle applied after packing when has_kernel (quantize_finetune_decoder_layer):
// in: packed[(b * (n/16) + blk)][32] for an m x n linear; out: same size int16 array as stored in 'trellis'.
void swizzle_for_kernel(const uint16_t* packed, int m, int n, uint16_t* out);
void unswizzle_for_kernel(const uint16_t* in, int m, int n, uint16_t* packed);

// fp64 lower Cholesky in place (row-major n x n, lower triangle used). Returns false if not PD.
bool cholesky_inplace(double* A, int n, int nthreads);
// block_LDL(H, 16) of math_utils.py followed by LRr[diag]=0: in: HRr (fp64 n x n), out: L (fp32 n x n) with
// strictly block-lower entries (diagonal 16x16 blocks zero, upper zero).  A is destroyed.
bool block_ldl_to_L(double* A, int n, float* L_out, int nthreads);

}  // namespace qtip
