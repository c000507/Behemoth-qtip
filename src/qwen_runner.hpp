// GPU forward of one Qwen3.5 layer (bf16 weights) for Hessian collection, mirroring LayerRunner.
#pragma once
#include <memory>

#include "gpu.hpp"
#include "model.hpp"
#include "qwen35.hpp"

namespace qtip {

// Hessian accumulators of a MoE layer: per-expert gateup/down matrices plus weight mass.
// HessAcc slots 0..3 stay valid; slots 2/3 hold the all-token ("pooled") Hessians used by the
// shared expert and as the fallback for starved experts.
struct MoeHessAcc {
  std::vector<DBuf<float>> gate, down;  // per expert [H][H] / [moe_inter][moe_inter]
  DBuf<float> wsum;                     // [E] sum of w^2
  DBuf<int> cnt;                        // [E] routed token count
  std::vector<int> blocks_gate, blocks_down;
};

class QwenLayerRunner {
 public:
  QwenLayerRunner(int device, const QwenConfig& cfg, int max_rows, int seq_len);
  ~QwenLayerRunner();
  void upload(const QwenLayer& L);  // bf16 weights -> device
  // In-place forward of nseq sequences (host bf16 [nseq][T][hidden]); if acc: accumulate Hessians.
  void forward(uint16_t* host_x, long nseq, HessAcc* acc, MoeHessAcc* moe = nullptr);
  void alloc_hess(HessAcc& a, bool full_layer, bool moe_layer = false);
  void alloc_moe_hess(MoeHessAcc& m);
  void zero_hess(HessAcc& a);
  void zero_moe_hess(MoeHessAcc& m);
  void symmetrize_hess(HessAcc& a);
  void symmetrize_moe_hess(MoeHessAcc& m);
  hipStream_t stream() const { return s_; }
  Gemm& gemm() { return *gemm_; }
  int device() const { return dev_; }
  double gemm_flops = 0;
  bool layer_is_full() const { return full_layer_; }
  bool layer_is_moe() const { return moe_; }

 private:
  void forward_batch(long rows, HessAcc* acc, MoeHessAcc* moe);
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
  DBuf<uint16_t> w_router_;                                          // MoE router [E][H]
  DBuf<uint16_t> w_exp_gu_, w_exp_down_;                             // MoE experts: [E][2F][H], [E][H][F]
  DBuf<uint16_t> w_sh_gu_, w_sh_down_, w_sg_;                        // shared expert + its scalar gate [H]
  DBuf<uint16_t> ln_in_, ln_post_;
  DBuf<float> ones_;  // all-1.0 scale vectors for the fp8-oriented epilogue kernels
  // activations
  DBuf<uint16_t> x_, h_, act_;
  DBuf<float> qkv_, y_, gu_;
  DBuf<uint16_t> Qp_, Kp_, Vp_, G_, attn_, o_;  // full: Qp/G/attn; linear: o_ (out_proj input)
  DBuf<float> S_, O_;                           // attention scratch
  DBuf<uint16_t> P_;
  DBuf<float> dn_S_, dn_ring_;                  // delta state [seqs][nv][kd*vd], [seqs][conv_dim][K]
  // MoE routing / expert scratch
  DBuf<float> rlog_, rwts_, rwsum_;              // [R][E], [R][k], [E]
  DBuf<int> rids_, rcnt_, rbase_, rrun_;         // [R][k], [E], [E], [E]
  DBuf<int> slot_tok_;                           // [R*k]
  DBuf<float> slot_w_;                           // [R*k]
  DBuf<uint16_t> eX_, eXs_, eact_s_;             // [R][H], [R][H], [R][F]
  DBuf<float> egu_, ey_, esg_;                   // [R][2F], [R][H], [R]
  HBuf<int> rcnt_h_;
  HBuf<uint16_t> pin_;
};

}  // namespace qtip
