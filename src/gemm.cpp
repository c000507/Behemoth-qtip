#include <cstdio>
#include <sstream>

#include "gpu.hpp"

#define LT_CHECK(x)                                                                                        \
  do {                                                                                                     \
    hipblasStatus_t st_ = (x);                                                                             \
    if (st_ != HIPBLAS_STATUS_SUCCESS)                                                                     \
      ::qtip::die(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #x " -> status " +         \
                  std::to_string(int(st_)));                                                               \
  } while (0)

namespace qtip {

Gemm::Gemm(hipStream_t s) : s_(s) {
  LT_CHECK(hipblasLtCreate(&h_));
  HIP_CHECK(hipMalloc(&ws_, ws_size_));
}
Gemm::~Gemm() {
  for (auto& kv : plans_) {
    hipblasLtMatrixLayoutDestroy(kv.second.a);
    hipblasLtMatrixLayoutDestroy(kv.second.b);
    hipblasLtMatrixLayoutDestroy(kv.second.c);
    hipblasLtMatmulDescDestroy(kv.second.desc);
  }
  if (ws_) (void)hipFree(ws_);
  hipblasLtDestroy(h_);
}

void Gemm::run(bool transA, bool transB, int M, int N, int K, const void* A, long lda, const void* B, long ldb,
               void* C, long ldc, OutT out, float alpha, float beta, int batch, long strideA, long strideB,
               long strideC) {
  std::ostringstream key;
  key << transA << transB << ' ' << M << ' ' << N << ' ' << K << ' ' << lda << ' ' << ldb << ' ' << ldc << ' '
      << int(out) << ' ' << batch << ' ' << strideA << ' ' << strideB << ' ' << strideC << ' ' << (beta != 0.f);
  auto it = plans_.find(key.str());
  if (it == plans_.end()) {
    Plan p{};
    LT_CHECK(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t ta = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N, tb = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    LT_CHECK(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    hipDataType ct = out == OutT::F32 ? HIP_R_32F : HIP_R_16BF;
    LT_CHECK(hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16BF, transA ? K : M, transA ? M : K, lda));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16BF, transB ? N : K, transB ? K : N, ldb));
    LT_CHECK(hipblasLtMatrixLayoutCreate(&p.c, ct, M, N, ldc));
    if (batch > 1) {
      int32_t bc = batch;
      int64_t sa = strideA, sb = strideB, sc = strideC;
      LT_CHECK(hipblasLtMatrixLayoutSetAttribute(p.a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
      LT_CHECK(hipblasLtMatrixLayoutSetAttribute(p.b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
      LT_CHECK(hipblasLtMatrixLayoutSetAttribute(p.c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
      LT_CHECK(hipblasLtMatrixLayoutSetAttribute(p.a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa)));
      LT_CHECK(hipblasLtMatrixLayoutSetAttribute(p.b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb)));
      LT_CHECK(hipblasLtMatrixLayoutSetAttribute(p.c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sc, sizeof(sc)));
    }
    hipblasLtMatmulPreference_t pref;
    LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t wsb = ws_size_;
    LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
    hipblasLtMatmulHeuristicResult_t res[8];
    int nres = 0;
    hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(h_, p.desc, p.a, p.b, p.c, p.c, pref, 8, res, &nres);
    hipblasLtMatmulPreferenceDestroy(pref);
    if (st != HIPBLAS_STATUS_SUCCESS || nres == 0)
      die("hipBLASLt: no algorithm for GEMM " + key.str());
    p.algo = res[0].algo;
    p.ws = res[0].workspaceSize;
    it = plans_.emplace(key.str(), p).first;
  }
  const Plan& p = it->second;
  LT_CHECK(hipblasLtMatmul(h_, p.desc, &alpha, A, p.a, B, p.b, &beta, C, p.c, C, p.c, &p.algo, ws_, ws_size_, s_));
}

}  // namespace qtip
