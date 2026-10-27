// GPU forward of one Qwen3.5 layer (bf16 weights) for Hessian collection, mirroring LayerRunner.
#pragma once
#include <memory>

#include "gpu.hpp"
#include "model.hpp"
#include "qwen35.hpp"

namespace qtip {

class QwenLayerRunner {
 public:
  QwenLayerRunner(int device, const QwenConfig& cfg, int max_rows, int seq_len);
  ~QwenLayerRunner();
  void upload(const QwenLayer& L);  // bf16 weights -> device
  // In-place forward of nseq sequences (host bf16 [nseq][T][hidden]); if acc: accumulate Hessians.
  void forward(uint16_t* host_x, long nseq, HessAcc* acc);
  void alloc_hess(HessAcc& a, bool full_layer);
  void zero_hess(HessAcc& a);
  void symmetrize_hess(HessAcc& a);
  hipStream_t stream() const { return s_; }
  Gemm& gemm() { return *gemm_; }
  int device() const { return dev_; }
  double gemm_flops = 0;
  bool layer_is_full() const { return full_layer_; }

 private:
  void forward_batch(long rows, HessAcc* acc);
  int dev_;
  QwenConfig cfg_;
  int max_rows_, T_;
  hipStream_t s_;
  std::unique_ptr<Gemm> gemm_;
  bool full_layer_ = false, moe_ = false;
  // weights (bf16) + fp32 aux
  DBuf<uint16_t> wq_, wk_, wv_, wo_, qn_, kn_;                       // full attention
  DBuf<uint16_t> w_in_qkv_, w_in_z_, w_in_ba_, w_out_;               // linear attention
  DBuf<uint16_t> conv_w_, dn_norm_;                                  // linear attention
  DBuf<float> a_dt_;                                                 // [2*nv] = A_log | dt_bias
  DBuf<uint16_t> w_gate_, w_up_, w_down_;                            // dense MLP
  DBuf<uint16_t> ln_in_, ln_post_;
  DBuf<float> ones_;  // all-1.0 scale vectors for the fp8-oriented epilogue kernels
  // activations
  DBuf<uint16_t> x_, h_, act_;
  DBuf<float> qkv_, y_, gu_;
  DBuf<uint16_t> Qp_, Kp_, Vp_, G_, attn_, o_;  // full: Qp/G/attn; linear: o_ (out_proj input)
  DBuf<float> S_, P_, O_;                       // attention scratch
  DBuf<float> dn_S_, dn_ring_;                  // delta state [seqs][nv][kd*vd], [seqs][conv_dim][K]
  HBuf<uint16_t> pin_;
};

}  // namespace qtip
