// QTIP quantization of a group of linears that share one input Hessian (q/k/v, o, gate/up, down).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gpu.hpp"

namespace qtip {

struct QuantOpts {
  double sigma_reg = 1e-2;       // regularize_H
  double scale_override = 0.9;   // Wscale = rms(Wr) / (rms(lut) * scale_override)
  int vit_batch = 1024;          // max trellises per Viterbi launch (memory: 260 KB each)
  int ldl_threads = 16;          // CPU threads for the fp64 Cholesky
  bool share_su = true;          // one SU (and one LDL) per Hessian group
  std::string dump_dir;          // if set: write W (fp32) and the reconstructed hatW (fp32) per linear
};

struct LinearIn {
  std::string name;      // e.g. "self_attn.q_proj"
  int m = 0;             // out features
  const uint8_t* w_fp8 = nullptr;     // [m][n] host
  const uint16_t* scale_bf16 = nullptr;  // [m] host (per-output-channel FP8 scale)
  const float* w_f32 = nullptr;       // alternatively a dense fp32 weight [m][n] (tests)
};

struct LinearOut {
  std::string name;
  int m = 0, n = 0;
  std::vector<uint16_t> trellis;  // [(m/16)*(n/16)][32] int16 bits, kernel layout
  std::vector<uint16_t> su_bf16;  // [n]
  std::vector<float> sv;          // [m] sign vector (fp32 +-1)
  float wscale = 0;
  double proxy = 0;          // tr(dW H dW^T)/tr(W H W^T), unregularized H (original domain, = rotated)
  double proxy_rot_reg = 0;  // same with the regularized rotated Hessian (what the reference prints)
  double seconds = 0;        // share of the group time (by weight count)
};

struct GroupTiming {
  double load = 0, rht = 0, ldl_cpu = 0, ldlq = 0, viterbi = 0, proxy = 0, pack = 0, total = 0;
};

// Rotated / factorised Hessian of one group (shared SU).  rotate_hessian runs on the GPU, ldl_hessian on the CPU
// (fp64), so the LDL of the next group can overlap the LDLQ of the current one.
struct HessPrep {
  int n = 0;
  uint64_t seed = 0;
  std::vector<float> su;   // [n] +-1
  double creg = 0;         // sigma_reg * mean(diag(H))
  std::vector<float> HRr;  // rotated regularized Hessian [n][n]
  std::vector<float> L;    // block-LDL factor (strictly block-lower, fp32)
  double t_rot = 0, t_ldl = 0;
};
void rotate_hessian(hipStream_t s, const float* H, int n, uint64_t seed, const QuantOpts& opt, HessPrep& hp);
void ldl_hessian(HessPrep& hp, int threads, const std::string& tag);
std::vector<LinearOut> quantize_group(hipStream_t s, Gemm& gemm, const std::vector<LinearIn>& lins, HessPrep& hp,
                                      const QuantOpts& opt, GroupTiming* tm, const std::string& tag);

// Quantize one Hessian group on the current device. H: n x n fp32 host (any positive scale).
// seed: deterministic sign vectors. Returns outputs in the order of `lins`.
std::vector<LinearOut> quantize_group(hipStream_t s, Gemm& gemm, const std::vector<LinearIn>& lins, int n,
                                      const float* H, uint64_t seed, const QuantOpts& opt, GroupTiming* tm,
                                      const std::string& tag);

// GPU LDLQ (lib/algo/ldlq.py with buf_cols=128, for_kernel layout) on WT (n x m, fp32, already scaled),
// L (n x n fp32, strictly block-lower). Writes hatWT (n x m) and states Q[(m/16)*(n/16)][128].
// Returns seconds spent in the Viterbi kernels (synchronised measurement) via *vit_s.
void ldlq_gpu(hipStream_t s, Gemm& gemm, const float* WT, const float* L, long n, long m, float* hatWT, uint16_t* Q,
              int vit_batch, const uint32_t* lut_h2_dev, double* vit_s);

// Upload the packed half2 LUT (65536 entries) to the current device (cached per device).
const uint32_t* device_lut();

}  // namespace qtip
