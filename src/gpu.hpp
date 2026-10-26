// HIP helpers + kernel launch wrappers (implemented in kernels.hip / viterbi.hip) + hipBLASLt GEMM wrapper.
#pragma once
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "common.hpp"

#define HIP_CHECK(x)                                                                                 \
  do {                                                                                               \
    hipError_t e_ = (x);                                                                             \
    if (e_ != hipSuccess)                                                                            \
      ::qtip::die(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #x " -> " +          \
                  hipGetErrorString(e_));                                                            \
  } while (0)

namespace qtip {

// RAII device buffer
template <class T>
struct DBuf {
  T* p = nullptr;
  size_t n = 0;
  DBuf() = default;
  explicit DBuf(size_t count) { alloc(count); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  DBuf(DBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
  DBuf& operator=(DBuf&& o) noexcept {
    if (this != &o) { free(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
    return *this;
  }
  ~DBuf() { free(); }
  void alloc(size_t count) {
    free();
    if (count) HIP_CHECK(hipMalloc(&p, count * sizeof(T)));
    n = count;
  }
  void ensure(size_t count) { if (count > n) alloc(count); }
  void free() {
    if (p) (void)hipFree(p);
    p = nullptr, n = 0;
  }
  size_t bytes() const { return n * sizeof(T); }
  operator T*() const { return p; }
};

// pinned host buffer
template <class T>
struct HBuf {
  T* p = nullptr;
  size_t n = 0;
  HBuf() = default;
  explicit HBuf(size_t count) { alloc(count); }
  HBuf(const HBuf&) = delete;
  HBuf& operator=(const HBuf&) = delete;
  ~HBuf() { free(); }
  void alloc(size_t count) {
    free();
    if (count) HIP_CHECK(hipHostMalloc(&p, count * sizeof(T), hipHostMallocDefault));
    n = count;
  }
  void ensure(size_t count) { if (count > n) alloc(count); }
  void free() {
    if (p) (void)hipHostFree(p);
    p = nullptr, n = 0;
  }
  operator T*() const { return p; }
};

// ---------------------------------------------------------------- GEMM (hipBLASLt), column-major semantics
// D(MxN) = alpha * op(A)(MxK) * op(B)(KxN) + beta * C, A/B bf16, C/D fp32 or bf16 (D == C), fp32 accumulate.
enum class OutT { F32, BF16 };
class Gemm {
 public:
  explicit Gemm(hipStream_t s);
  ~Gemm();
  void run(bool transA, bool transB, int M, int N, int K, const void* A, long lda, const void* B, long ldb, void* C,
           long ldc, OutT out, float alpha = 1.f, float beta = 0.f, int batch = 1, long strideA = 0,
           long strideB = 0, long strideC = 0);

 private:
  struct Plan {
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t a, b, c;
    hipblasLtMatmulAlgo_t algo;
    size_t ws;
  };
  hipStream_t s_;
  hipblasLtHandle_t h_;
  void* ws_ = nullptr;
  size_t ws_size_ = 64ull << 20;
  std::map<std::string, Plan> plans_;
};

// ---------------------------------------------------------------- elementwise / layout kernels (kernels.hip)
// fp8 e4m3 [rows][cols] * scale[row] (bf16) -> fp32 or bf16 (exact fp8 value, scale separate when scale==nullptr)
void k_fp8_to_f32(const uint8_t* w, const uint16_t* scale_bf16, float* out, long rows, long cols, hipStream_t s);
void k_fp8_to_bf16(const uint8_t* w, uint16_t* out, long n, hipStream_t s);  // no scale: exact fp8 values
// X[r][c] *= rs[r] * cs[c] (either may be null)
void k_scale_rows_cols(float* X, long rows, long cols, long ld, const float* rs, const float* cs, hipStream_t s);
void k_scale(float* X, long rows, long cols, long ld, float a, hipStream_t s);
// D = A - B (rows x cols, common ld)
void k_sub(const float* A, const float* B, float* D, long rows, long cols, long ld, hipStream_t s);
// out[c][r] = in[r][c]; in: rows x cols (ld_in), out: cols x rows (ld_out)
void k_transpose(const float* in, long rows, long cols, long ld_in, float* out, long ld_out, hipStream_t s);
// In-place row Hadamard: x <- (hadK^(T) (x) H) x / sqrt(n) on `rows` rows of length n (row stride ld)
void k_hadamard_rows(float* X, long rows, int n, long ld, bool transpose, hipStream_t s);
// sum of squares of a rows x cols block (ld) -> double accumulated on device into *out (atomic)
void k_sumsq(const float* X, long rows, long cols, long ld, double* out, hipStream_t s);
// add a to the diagonal of n x n
void k_add_diag(float* H, long n, float a, hipStream_t s);
void k_diag_sum(const float* H, long n, double* out, hipStream_t s);
// split fp32 -> bf16 hi + bf16 lo (X = hi + lo + O(2^-16 X)); rows x cols (ld_in) -> dense rows x cols (ld_out)
void k_split_bf16(const float* X, long rows, long cols, long ld_in, uint16_t* hi, uint16_t* lo, long ld_out,
                  hipStream_t s);
// D = A - B (rows x cols, common ld) -> bf16 hi/lo (dense, ld_out)
void k_diff_split_bf16(const float* A, const float* B, long rows, long cols, long ld, uint16_t* hi, uint16_t* lo,
                       long ld_out, hipStream_t s);
// sum over all elements of A .* B (rows x cols, lda/ldb), fp64 atomic into *out
void k_dot(const float* A, long lda, const float* B, long ldb, long rows, long cols, double* out, hipStream_t s);
void k_f32_to_bf16(const float* in, uint16_t* out, long n, hipStream_t s);
void k_bf16_to_f32(const uint16_t* in, float* out, long n, hipStream_t s);

// ---------------------------------------------------------------- model forward kernels
// out_bf16 = rmsnorm(x_bf16) * w ; rows x n
void k_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* out, long rows, int n, float eps, hipStream_t s);
// qkv fp32 [T][14336] (unscaled GEMM output) * scale[col] -> RoPE (HF rotate_half) -> Qp[96][T][128], Kp[8][T][128],
// Vp[8][T][128] bf16 (per-sequence; T tokens; positions 0..T-1)
void k_qkv_rope(const float* qkv, const float* col_scale, uint16_t* Qp, uint16_t* Kp, uint16_t* Vp, int T,
                int nq, int nkv, int hd, float theta, hipStream_t s);
// causal softmax on S [rows = nh*T][T] fp32 (row r -> token r % T), scale; writes P bf16 (zeros above diagonal)
void k_causal_softmax(const float* S, uint16_t* P, int nh, int T, float scale, hipStream_t s);
// O [nh][T][hd] fp32 -> out [T][(h0+h)*hd] bf16 with row stride ld
void k_heads_to_tokens(const float* O, uint16_t* out, int nh, int T, int hd, int h0, long ld, hipStream_t s);
// x_bf16 += y_fp32 * col_scale[col] (residual add, bf16 result)
void k_residual_add(uint16_t* x, const float* y, const float* col_scale, long rows, int n, hipStream_t s);
// act[T][F] = silu(gu[:, :F] * sg) * (gu[:, F:] * su), gu fp32 [T][2F]
void k_silu_mul(const float* gu, const float* col_scale, uint16_t* act, long rows, int F, hipStream_t s);

// ---------------------------------------------------------------- QTIP kernels (viterbi.hip)
// Build the 256-sequences for column block (16 rows of W_T starting at col0) for all m/16 trellises:
//   WXWX[c][j] = W_T[col0+c][j] + sum_{k=col0+16}^{bufend-1} L[k][col0+c] (W_T[k][j]-hatW_T[k][j]) + prod[col0+c][j]
//   X[b][p] = WXWX[c][16b+r] with perm[p] = 16r+c
void k_ldlq_prep(const float* WT, const float* hatWT, const float* prod, const float* L, long n, long m, int col0,
                 int bufend, float* X, hipStream_t s);
// Viterbi forward over B sequences (X [B][256]); roll=1 reads X rolled by 128; overlap [B] or null.
// A: [B][127][1024] uint16 argmin table, final_state [B].
void k_viterbi_fwd(const float* X, const int* overlap, int roll, int B, uint16_t* A, uint16_t* final_state,
                   const uint32_t* lut_h2, hipStream_t s);
// pass 1 backtrace: overlap[b] = state(step 64) >> 4
void k_viterbi_overlap(const uint16_t* A, const uint16_t* final_state, int B, int* overlap, hipStream_t s);
// pass 2 backtrace: write states into Q[(b * nblk + blk)][128] and the decoded (inverse-permuted) tile into hatW_T
void k_viterbi_finish(const uint16_t* A, const uint16_t* final_state, int B, int b0, uint16_t* Q, int nblk, int blk,
                      float* hatWT, long m, int col0, const uint32_t* lut_h2, hipStream_t s);
// standalone decode of states (for tests): hat[b][256] = lut[states]
void k_decode_states(const uint16_t* states, int B, float* hat, const uint32_t* lut_h2, hipStream_t s);

}  // namespace qtip
