// Calibrated NVFP4 quantization (output format identical to nvfp4_quantize: same tensors, packing, E4M3 scale
// encoding, F32 global scale). Two independent upgrades over the naive nearest-value / absmax-scale quantizer:
//  * scale search: per 16-block candidate E4M3 scale codes (absmax/6 * {0.70..1.05} and the neighbouring codes) are
//    scored and the best one kept (the global scale stays max|W| / (448*6)).
//  * calibrated rounding: LDLQ-style error feedback (OPTQ/GPTQ equivalent) with the layer's input Hessian, in the
//    original basis (NVFP4 has no rotation): columns are processed in 16-wide blocks from the last to the first, the
//    residual of finished blocks is propagated to the remaining columns through the unit-lower factor of
//    H = L D L^T, and candidate scales are scored by sum_k D_k (t_k - q_k)^2, the exact proxy-error increment.
// All kernels live in nvfp4_gpu.hip (single source for HIP and CUDA).
#pragma once
#include <vector>

#include "gpu.hpp"
#include "nvfp4.hpp"

namespace qtip {

struct Nvfp4Opts {
  bool scale_search = true;
  bool calibrate = true;
  double sigma_reg = 1e-2;  // regularize_H as in the QTIP path: H += sigma_reg * mean(diag(H)) * I
};

// Quantize several matrices that share one input Hessian (n x n, fp32, any positive scale; may be null: then no
// calibration, scale search weighted uniformly). ws[i] is row-major [ms[i]][n] fp32. Returns one tensor per matrix.
// `diag_only_weights` of the log: set *calibrated to whether the Hessian was actually used.
std::vector<Nvfp4Tensor> nvfp4_quantize_group(hipStream_t s, const std::vector<const float*>& ws,
                                              const std::vector<int>& ms, int n, const float* H,
                                              const Nvfp4Opts& opt, bool* calibrated = nullptr);

}  // namespace qtip
