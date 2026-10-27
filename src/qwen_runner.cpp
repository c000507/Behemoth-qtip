// Qwen3.5 GPU layer forward for Hessian collection (bf16 weights, fp32 GEMM outputs). Mirrors
// LayerRunner's structure but dispatches per layer type (gated full attention / Gated DeltaNet).
#include "qwen_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "qwen_gpu.hpp"

namespace qtip {

QwenLayerRunner::QwenLayerRunner(int device, const QwenConfig& cfg, int max_rows, int seq_len)
    : dev_(device), cfg_(cfg), max_rows_(max_rows), T_(seq_len) {
  HIP_CHECK(hipSetDevice(dev_));
  HIP_CHECK(hipStreamCreateWithFlags(&s_, hipStreamNonBlocking));
  gemm_ = std::make_unique<Gemm>(s_);
  const long H = cfg.hidden, F = cfg.inter, R = max_rows;
  const long seqs = R / T_;
  const long A = long(cfg.nq) * cfg.hd;                       // attention out (o_proj in)
  const long V = cfg.value_dim();                             // delta out (out_proj in)
  const long Wfull = long(cfg.nq) * 2 * cfg.hd + 2 * long(cfg.nkv) * cfg.hd;
  const long Wlin = cfg.conv_dim() + V + 2 * cfg.lin_nv;
  const int gq = cfg.nq / cfg.nkv;
  QCHECK(R % T_ == 0, "max_rows must be a multiple of the sequence length");
  const size_t maxw = size_t(std::max<long>({Wfull, Wlin}));
  wq_.alloc(size_t(A) * 2 * H);
  wk_.alloc(size_t(cfg.nkv) * cfg.hd * H);
  wv_.alloc(size_t(cfg.nkv) * cfg.hd * H);
  wo_.alloc(size_t(H) * A);
  qn_.alloc(size_t(cfg.hd));
  kn_.alloc(size_t(cfg.hd));
  w_in_qkv_.alloc(size_t(cfg.conv_dim()) * H);
  w_in_z_.alloc(size_t(V) * H);
  w_in_ba_.alloc(size_t(2) * cfg.lin_nv * H);
  w_out_.alloc(size_t(H) * V);
  conv_w_.alloc(size_t(cfg.conv_dim()) * cfg.conv_k);
  dn_norm_.alloc(size_t(cfg.lin_vd));
  a_dt_.alloc(size_t(2) * cfg.lin_nv);
  w_gate_.alloc(size_t(F) * H);
  w_up_.alloc(size_t(F) * H);
  w_down_.alloc(size_t(H) * F);
  ln_in_.alloc(size_t(H));
  ln_post_.alloc(size_t(H));
  ones_.alloc(size_t(std::max<long>({2 * F, H, A, V})) + 16);
  x_.alloc(size_t(R) * H);
  h_.alloc(size_t(R) * H);
  act_.alloc(size_t(R) * F);
  qkv_.alloc(size_t(R) * maxw);
  y_.alloc(size_t(R) * H);
  gu_.alloc(size_t(R) * 2 * F);
  Qp_.alloc(size_t(cfg.nq) * T_ * cfg.hd);
  Kp_.alloc(size_t(cfg.nkv) * T_ * cfg.hd);
  Vp_.alloc(size_t(cfg.nkv) * T_ * cfg.hd);
  G_.alloc(size_t(R) * A);
  attn_.alloc(size_t(R) * A);
  o_.alloc(size_t(R) * V);
  S_.alloc(size_t(gq) * T_ * T_);
  P_.alloc(size_t(gq) * T_ * T_);
  O_.alloc(size_t(gq) * T_ * cfg.hd);
  dn_S_.alloc(size_t(seqs) * cfg.lin_nv * cfg.lin_kd * cfg.lin_vd);
  dn_ring_.alloc(size_t(seqs) * cfg.conv_dim() * cfg.conv_k);
  pin_.alloc(size_t(R) * H);
  k_q_ones(ones_, long(ones_.n), s_);
}

QwenLayerRunner::~QwenLayerRunner() { (void)hipStreamDestroy(s_); }

void QwenLayerRunner::alloc_hess(HessAcc& a, bool full_layer) {
  HIP_CHECK(hipSetDevice(dev_));
  a.dim[0] = cfg_.hidden;
  a.dim[1] = full_layer ? cfg_.nq * cfg_.hd : cfg_.value_dim();
  a.dim[2] = cfg_.hidden;
  a.dim[3] = cfg_.inter;
  for (int i = 0; i < 4; ++i) a.h[i].alloc(size_t(a.dim[i]) * a.dim[i]);
  zero_hess(a);
}
void QwenLayerRunner::symmetrize_hess(HessAcc& a) {
  for (int i = 0; i < 4; ++i)
    if (a.blocks[i] > 1) k_symmetrize_blocks(a.h[i], a.dim[i], a.dim[i] / a.blocks[i], s_);
  HIP_CHECK(hipStreamSynchronize(s_));
}
void QwenLayerRunner::zero_hess(HessAcc& a) {
  for (int i = 0; i < 4; ++i) HIP_CHECK(hipMemsetAsync(a.h[i], 0, a.h[i].bytes(), s_));
  a.count = 0;
}

void QwenLayerRunner::upload(const QwenLayer& L) {
  HIP_CHECK(hipSetDevice(dev_));
  auto up16 = [&](const std::vector<uint16_t>& v, DBuf<uint16_t>& d) {
    HIP_CHECK(hipMemcpyAsync(d, v.data(), v.size() * 2, hipMemcpyHostToDevice, s_));
  };
  full_layer_ = L.full;
  moe_ = L.moe;
  QCHECK(!moe_, "MoE Hessian collection not implemented yet (Qwen3.5 dense only)");
  if (full_layer_) {
    up16(L.w_q, wq_);
    up16(L.w_k, wk_);
    up16(L.w_v, wv_);
    up16(L.w_o, wo_);
    up16(L.q_norm, qn_);
    up16(L.k_norm, kn_);
  } else {
    up16(L.w_in_qkv, w_in_qkv_);
    up16(L.w_in_z, w_in_z_);
    up16(L.w_in_ba, w_in_ba_);
    up16(L.w_out, w_out_);
    up16(L.conv_w, conv_w_);
    up16(L.dn_norm, dn_norm_);
    std::vector<float> aux(L.a_log);
    aux.insert(aux.end(), L.dt_bias.begin(), L.dt_bias.end());
    HIP_CHECK(hipMemcpyAsync(a_dt_, aux.data(), aux.size() * 4, hipMemcpyHostToDevice, s_));
  }
  up16(L.w_gate, w_gate_);
  up16(L.w_up, w_up_);
  up16(L.w_down, w_down_);
  up16(L.ln_in, ln_in_);
  up16(L.ln_post, ln_post_);
  HIP_CHECK(hipStreamSynchronize(s_));
}

void QwenLayerRunner::forward(uint16_t* host_x, long nseq, HessAcc* acc) {
  HIP_CHECK(hipSetDevice(dev_));
  const long H = cfg_.hidden;
  const long per = max_rows_ / T_;
  for (long s0 = 0; s0 < nseq; s0 += per) {
    long ns = std::min(per, nseq - s0);
    long rows = ns * T_;
    size_t bytes = size_t(rows) * H * 2;
    uint16_t* hx = host_x + size_t(s0) * T_ * H;
    std::memcpy(pin_, hx, bytes);
    HIP_CHECK(hipMemcpyAsync(x_, pin_, bytes, hipMemcpyHostToDevice, s_));
    forward_batch(rows, acc);
    HIP_CHECK(hipMemcpyAsync(pin_, x_, bytes, hipMemcpyDeviceToHost, s_));
    HIP_CHECK(hipStreamSynchronize(s_));
    std::memcpy(hx, pin_, bytes);
    if (acc) acc->count += rows;
  }
}

void QwenLayerRunner::forward_batch(long R, HessAcc* acc) {
  const int H = cfg_.hidden, F = cfg_.inter, hd = cfg_.hd, nq = cfg_.nq, nkv = cfg_.nkv, T = T_;
  const int gq = nq / nkv, A = nq * hd;
  const long seqs = R / T;
  Gemm& g = *gemm_;
  auto hess = [&](int which, const uint16_t* X, int n) {
    if (!acc) return;
    const int nb = (n % 4 == 0 && (n / 4) % 16 == 0) ? 4 : 1, bs = n / nb;
    for (int bi = 0; bi < nb; ++bi)
      for (int bj = bi; bj < nb; ++bj)
        g.run(false, true, bs, bs, int(R), X + long(bi) * bs, n, X + long(bj) * bs, n,
              acc->h[which] + long(bj) * bs * n + long(bi) * bs, n, OutT::F32, 1.f, 1.f);
    acc->blocks[which] = nb;
    gemm_flops += 2.0 * n * n * R * (nb * (nb + 1) / 2) / double(nb * nb);
  };
  // Y (R x out, row stride yld) = X (R x in) W^T
  auto proj = [&](const uint16_t* X, int in, const uint16_t* W, int out, float* Y, long yld) {
    g.run(true, false, out, int(R), in, W, in, X, in, Y, yld, OutT::F32, 1.f, 0.f);
    gemm_flops += 2.0 * out * in * R;
  };
  k_q_rmsnorm(x_, ln_in_, h_, R, H, cfg_.eps, s_);
  hess(0, h_, H);
  if (full_layer_) {
    const int W = nq * 2 * hd + 2 * nkv * hd;
    proj(h_, H, wq_, nq * 2 * hd, qkv_, W);
    proj(h_, H, wk_, nkv * hd, qkv_ + long(nq) * 2 * hd, W);
    proj(h_, H, wv_, nkv * hd, qkv_ + long(nq) * 2 * hd + nkv * hd, W);
    const float sm_scale = 1.0f / std::sqrt(float(hd));
    for (long sq = 0; sq < seqs; ++sq) {
      k_q_qkv_rope(qkv_ + sq * T * W, Qp_, G_ + sq * T * A, Kp_, Vp_, T, nq, nkv, hd, cfg_.rot_dim, cfg_.theta,
                   qn_, kn_, cfg_.eps, s_);
      for (int kv = 0; kv < nkv; ++kv) {
        const uint16_t* K = Kp_ + long(kv) * T * hd;
        const uint16_t* V = Vp_ + long(kv) * T * hd;
        const uint16_t* Q = Qp_ + long(kv) * gq * T * hd;
        g.run(true, false, T, gq * T, hd, K, hd, Q, hd, S_, T, OutT::F32, 1.f, 0.f);
        k_causal_softmax(S_, P_, gq, T, sm_scale, s_);
        g.run(false, false, hd, gq * T, T, V, hd, P_, T, O_, hd, OutT::F32, 1.f, 0.f);
        k_q_attn_out(O_, G_ + sq * T * A, attn_ + sq * T * A, gq, T, hd, kv * gq, A, s_);
        gemm_flops += 4.0 * gq * T * double(T) * hd;
      }
    }
    hess(1, attn_, A);
    proj(attn_, A, wo_, H, y_, H);
  } else {
    const int conv_dim = cfg_.conv_dim(), VD = cfg_.value_dim(), nv = cfg_.lin_nv;
    const int W = conv_dim + VD + 2 * nv;
    proj(h_, H, w_in_qkv_, conv_dim, qkv_, W);
    proj(h_, H, w_in_z_, VD, qkv_ + conv_dim, W);
    proj(h_, H, w_in_ba_, 2 * nv, qkv_ + conv_dim + VD, W);
    HIP_CHECK(hipMemsetAsync(dn_S_, 0, dn_S_.bytes(), s_));
    HIP_CHECK(hipMemsetAsync(dn_ring_, 0, dn_ring_.bytes(), s_));
    for (int t = 0; t < T; ++t)
      k_q_delta_step(qkv_, conv_w_, dn_norm_, a_dt_, dn_ring_, dn_S_, o_, int(seqs), T, t, conv_dim, VD,
                     cfg_.lin_kd, cfg_.lin_vd, cfg_.lin_nk, nv, cfg_.conv_k, cfg_.eps, s_);
    hess(1, o_, VD);
    proj(o_, VD, w_out_, H, y_, H);
  }
  k_residual_add(x_, y_, ones_, R, H, s_);
  k_q_rmsnorm(x_, ln_post_, h_, R, H, cfg_.eps, s_);
  hess(2, h_, H);
  proj(h_, H, w_gate_, F, gu_, 2 * F);
  proj(h_, H, w_up_, F, gu_ + F, 2 * F);
  k_silu_mul(gu_, ones_, act_, R, F, s_);
  hess(3, act_, F);
  proj(act_, F, w_down_, H, y_, H);
  k_residual_add(x_, y_, ones_, R, H, s_);
}

}  // namespace qtip
