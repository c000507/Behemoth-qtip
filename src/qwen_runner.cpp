// Qwen3.5 GPU layer forward for Hessian collection (bf16 weights, fp32 GEMM outputs). Mirrors
// LayerRunner's structure but dispatches per layer type (gated full attention / Gated DeltaNet) and
// supports MoE MLPs: per-expert routing-weighted Hessians (see docs/QWEN35.md for the weighting
// choice) plus the pooled (all-token) Hessians used by the shared expert.
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
  const long FS = std::max<long>(cfg.inter, cfg.shared_inter);  // MLP scratch width (dense or shared)
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
  w_router_.alloc(size_t(std::max(1, cfg_.n_experts)) * H);
  w_exp_gu_.alloc(size_t(std::max(1, cfg.n_experts)) * 2 * std::max(1, cfg.moe_inter) * H);
  w_exp_down_.alloc(size_t(std::max(1, cfg.n_experts)) * H * std::max(1, cfg.moe_inter));
  w_sh_gu_.alloc(size_t(2 * std::max(1, cfg.shared_inter)) * H);
  w_sh_down_.alloc(size_t(H) * std::max(1, cfg.shared_inter));
  w_sg_.alloc(size_t(H));
  ln_in_.alloc(size_t(H));
  ln_post_.alloc(size_t(H));
  ones_.alloc(size_t(std::max<long>({2 * FS, H, A, V})) + 16);
  x_.alloc(size_t(R) * H);
  h_.alloc(size_t(R) * H);
  act_.alloc(size_t(R) * FS);
  qkv_.alloc(size_t(R) * maxw);
  y_.alloc(size_t(R) * H);
  gu_.alloc(size_t(R) * 2 * FS);
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
  rlog_.alloc(size_t(R) * std::max(1, cfg.n_experts));
  rwts_.alloc(size_t(R) * std::max(1, cfg.topk));
  rwsum_.alloc(size_t(std::max(1, cfg.n_experts)));
  rids_.alloc(size_t(R) * std::max(1, cfg.topk));
  rcnt_.alloc(size_t(std::max(1, cfg.n_experts)));
  rbase_.alloc(size_t(std::max(1, cfg.n_experts)));
  rrun_.alloc(size_t(std::max(1, cfg.n_experts)));
  slot_tok_.alloc(size_t(R) * std::max(1, cfg.topk));
  slot_w_.alloc(size_t(R) * std::max(1, cfg.topk));
  eX_.alloc(size_t(R) * H);
  eXs_.alloc(size_t(R) * H);
  eact_s_.alloc(size_t(R) * cfg.moe_inter);
  egu_.alloc(size_t(R) * 2 * cfg.moe_inter);
  ey_.alloc(size_t(R) * H);
  esg_.alloc(size_t(R));
  rcnt_h_.alloc(size_t(2 * std::max(1, cfg.n_experts)));
  pin_.alloc(size_t(R) * H);
  k_q_ones(ones_, long(ones_.n), s_);
}

QwenLayerRunner::~QwenLayerRunner() { (void)hipStreamDestroy(s_); }

void QwenLayerRunner::alloc_hess(HessAcc& a, bool full_layer, bool moe_layer) {
  HIP_CHECK(hipSetDevice(dev_));
  a.dim[0] = cfg_.hidden;
  a.dim[1] = full_layer ? cfg_.nq * cfg_.hd : cfg_.value_dim();
  a.dim[2] = cfg_.hidden;
  a.dim[3] = moe_layer ? std::max(cfg_.shared_inter, 1) : cfg_.inter;  // MoE slot 3 = shared-expert down
  for (int i = 0; i < 4; ++i) a.h[i].alloc(size_t(std::max(1, a.dim[i])) * std::max(1, a.dim[i]));
  zero_hess(a);
}
void QwenLayerRunner::alloc_moe_hess(MoeHessAcc& m) {
  HIP_CHECK(hipSetDevice(dev_));
  const int E = cfg_.n_experts;
  m.gate.resize(size_t(E));
  m.down.resize(size_t(E));
  m.blocks_gate.assign(size_t(E), 1);
  m.blocks_down.assign(size_t(E), 1);
  for (int e = 0; e < E; ++e) {
    m.gate[size_t(e)].alloc(size_t(cfg_.hidden) * cfg_.hidden);
    m.down[size_t(e)].alloc(size_t(cfg_.moe_inter) * cfg_.moe_inter);
  }
  m.wsum.alloc(size_t(E));
  m.cnt.alloc(size_t(E));
  zero_moe_hess(m);
}
void QwenLayerRunner::symmetrize_hess(HessAcc& a) {
  for (int i = 0; i < 4; ++i)
    if (a.blocks[i] > 1) k_symmetrize_blocks(a.h[i], a.dim[i], a.dim[i] / a.blocks[i], s_);
  HIP_CHECK(hipStreamSynchronize(s_));
}
void QwenLayerRunner::symmetrize_moe_hess(MoeHessAcc& m) {
  for (int e = 0; e < cfg_.n_experts; ++e) {
    if (m.blocks_gate[size_t(e)] > 1)
      k_symmetrize_blocks(m.gate[size_t(e)], cfg_.hidden, cfg_.hidden / m.blocks_gate[size_t(e)], s_);
    if (m.blocks_down[size_t(e)] > 1)
      k_symmetrize_blocks(m.down[size_t(e)], cfg_.moe_inter, cfg_.moe_inter / m.blocks_down[size_t(e)], s_);
  }
  HIP_CHECK(hipStreamSynchronize(s_));
}
void QwenLayerRunner::zero_hess(HessAcc& a) {
  for (int i = 0; i < 4; ++i) HIP_CHECK(hipMemsetAsync(a.h[i], 0, a.h[i].bytes(), s_));
  a.count = 0;
}
void QwenLayerRunner::zero_moe_hess(MoeHessAcc& m) {
  for (int e = 0; e < cfg_.n_experts; ++e) {
    HIP_CHECK(hipMemsetAsync(m.gate[size_t(e)], 0, m.gate[size_t(e)].bytes(), s_));
    HIP_CHECK(hipMemsetAsync(m.down[size_t(e)], 0, m.down[size_t(e)].bytes(), s_));
  }
  HIP_CHECK(hipMemsetAsync(m.wsum, 0, m.wsum.bytes(), s_));
  HIP_CHECK(hipMemsetAsync(m.cnt, 0, m.cnt.bytes(), s_));
}

void QwenLayerRunner::upload(const QwenLayer& L) {
  HIP_CHECK(hipSetDevice(dev_));
  auto up16 = [&](const std::vector<uint16_t>& v, DBuf<uint16_t>& d) {
    HIP_CHECK(hipMemcpyAsync(d, v.data(), v.size() * 2, hipMemcpyHostToDevice, s_));
  };
  full_layer_ = L.full;
  moe_ = L.moe;
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
  if (moe_) {
    QCHECK(int(L.w_exp_gate.size()) == cfg_.n_experts, "expert weight count mismatch");
    up16(L.w_router, w_router_);
    {  // experts: gate|up concatenated per expert into [E][2F][H]
      std::vector<uint16_t> gu(w_exp_gu_.n);
      size_t rows2 = size_t(cfg_.moe_inter) * 2 * cfg_.hidden;
      for (int e = 0; e < cfg_.n_experts; ++e) {
        std::memcpy(gu.data() + size_t(e) * rows2, L.w_exp_gate[size_t(e)].data(),
                    L.w_exp_gate[size_t(e)].size() * 2);
        std::memcpy(gu.data() + size_t(e) * rows2 + size_t(cfg_.moe_inter) * cfg_.hidden,
                    L.w_exp_up[size_t(e)].data(), L.w_exp_up[size_t(e)].size() * 2);
      }
      HIP_CHECK(hipMemcpyAsync(w_exp_gu_, gu.data(), gu.size() * 2, hipMemcpyHostToDevice, s_));
      std::vector<uint16_t> dn(w_exp_down_.n);
      for (int e = 0; e < cfg_.n_experts; ++e)
        std::memcpy(dn.data() + size_t(e) * cfg_.hidden * cfg_.moe_inter, L.w_exp_down[size_t(e)].data(),
                    L.w_exp_down[size_t(e)].size() * 2);
      HIP_CHECK(hipMemcpyAsync(w_exp_down_, dn.data(), dn.size() * 2, hipMemcpyHostToDevice, s_));
    }
    {  // shared expert gate|up [2*S][H], down, scalar gate
      std::vector<uint16_t> shg(w_sh_gu_.n);
      std::memcpy(shg.data(), L.w_sh_gate.data(), L.w_sh_gate.size() * 2);
      std::memcpy(shg.data() + size_t(cfg_.shared_inter) * cfg_.hidden, L.w_sh_up.data(),
                  L.w_sh_up.size() * 2);
      HIP_CHECK(hipMemcpyAsync(w_sh_gu_, shg.data(), shg.size() * 2, hipMemcpyHostToDevice, s_));
      up16(L.w_sh_down, w_sh_down_);
      if (cfg_.shared_gate) up16(L.w_sh_gate_w, w_sg_);
    }
  } else {
    up16(L.w_gate, w_gate_);
    up16(L.w_up, w_up_);
    up16(L.w_down, w_down_);
  }
  up16(L.ln_in, ln_in_);
  up16(L.ln_post, ln_post_);
  HIP_CHECK(hipStreamSynchronize(s_));
}

void QwenLayerRunner::forward(uint16_t* host_x, long nseq, HessAcc* acc, MoeHessAcc* moe) {
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
    forward_batch(rows, acc, moe);
    HIP_CHECK(hipMemcpyAsync(pin_, x_, bytes, hipMemcpyDeviceToHost, s_));
    HIP_CHECK(hipStreamSynchronize(s_));
    std::memcpy(hx, pin_, bytes);
    if (acc) acc->count += rows;
  }
}

void QwenLayerRunner::forward_batch(long R, HessAcc* acc, MoeHessAcc* moe) {
  const int H = cfg_.hidden, hd = cfg_.hd, nq = cfg_.nq, nkv = cfg_.nkv, T = T_;
  const int gq = nq / nkv, A = nq * hd;
  const long seqs = R / T;
  Gemm& g = *gemm_;
  // Hessian X^T X over `nr` rows (triangular block accumulation when n is 64-divisible)
  auto hess_n = [&](float* hdst, int* blocks_out, const uint16_t* X, int n, long nr) {
    if (!acc || !hdst) return;
    const int nb = (n % 4 == 0 && (n / 4) % 16 == 0) ? 4 : 1, bs = n / nb;
    for (int bi = 0; bi < nb; ++bi)
      for (int bj = bi; bj < nb; ++bj)
        g.run(false, true, bs, bs, int(nr), X + long(bi) * bs, n, X + long(bj) * bs, n,
              hdst + long(bj) * bs * n + long(bi) * bs, n, OutT::F32, 1.f, 1.f);
    *blocks_out = nb;
    gemm_flops += 2.0 * n * n * nr * (nb * (nb + 1) / 2) / double(nb * nb);
  };
  auto hess = [&](int which, const uint16_t* X, int n) {
    if (acc) hess_n(acc->h[which], &acc->blocks[which], X, n, R);
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
  if (!moe_) {
    const int F = cfg_.inter;
    proj(h_, H, w_gate_, F, gu_, 2 * F);
    proj(h_, H, w_up_, F, gu_ + F, 2 * F);
    k_silu_mul(gu_, ones_, act_, R, F, s_);
    hess(3, act_, F);
    proj(act_, F, w_down_, H, y_, H);
  } else {
    const int E = cfg_.n_experts, K = cfg_.topk, F = cfg_.moe_inter, S = cfg_.shared_inter;
    const long gstride = size_t(2) * F * H;  // per-expert row stride inside w_exp_gu_
    // --- route (softmax -> top-k -> renorm), compact per-expert token lists
    proj(h_, H, w_router_, E, rlog_, E);
    HIP_CHECK(hipMemsetAsync(rcnt_, 0, rcnt_.bytes(), s_));
    HIP_CHECK(hipMemsetAsync(rwsum_, 0, rwsum_.bytes(), s_));
    k_q_route(rlog_, int(R), E, K, rwts_, rids_, rcnt_, rwsum_, s_);
    k_q_scan(rcnt_, rbase_, E, s_);
    HIP_CHECK(hipMemsetAsync(rrun_, 0, rrun_.bytes(), s_));
    k_q_route_fill(rlog_, int(R), E, K, rbase_, rrun_, slot_tok_, slot_w_, s_);
    HIP_CHECK(hipMemcpyAsync(rcnt_h_, rcnt_, rcnt_.bytes(), hipMemcpyDeviceToHost, s_));
    HIP_CHECK(hipMemcpyAsync(rcnt_h_ + E, rbase_, rbase_.bytes(), hipMemcpyDeviceToHost, s_));
    HIP_CHECK(hipStreamSynchronize(s_));
    int* rh = rcnt_h_;
    std::vector<int> cnt(rh, rh + E);
    std::vector<long> base_h(rh + E, rh + 2 * E);
    // --- routed experts: per-expert weighted Hessians + forward contribution
    HIP_CHECK(hipMemsetAsync(y_, 0, size_t(R) * H * 4, s_));
    for (int e = 0; e < E; ++e) {
      const int ne = cnt[size_t(e)];
      if (ne <= 0) continue;
      const int* tok_e = slot_tok_ + base_h[size_t(e)];
      const float* w_e = slot_w_ + base_h[size_t(e)];
      // gather unscaled rows (forward) and w-scaled rows (Hessian) for this expert
      k_q_moe_gather(h_, R, H, tok_e, nullptr, eX_, ne, s_);
      k_q_moe_gather(h_, R, H, tok_e, w_e, eXs_, ne, s_);
      // per-expert gateup Hessian (w^2-weighted)
      if (moe) hess_n(moe->gate[size_t(e)], &moe->blocks_gate[size_t(e)], eXs_, H, ne);
      // forward: gate|up (f32 [ne][2F]) -> silu*u
      g.run(true, false, F, int(ne), H, w_exp_gu_ + size_t(e) * gstride, H, eX_, H, egu_, 2 * F, OutT::F32,
            1.f, 0.f);
      g.run(true, false, F, int(ne), H, w_exp_gu_ + size_t(e) * gstride + size_t(F) * H, H, eX_, H, egu_ + F,
            2 * F, OutT::F32, 1.f, 0.f);
      gemm_flops += 4.0 * F * H * ne;
      k_silu_mul(egu_, ones_, act_, ne, F, s_);
      // per-expert down Hessian (w^2-weighted activations)
      k_q_moe_gather(act_, R, F, tok_e, w_e, eact_s_, ne, s_);
      if (moe) hess_n(moe->down[size_t(e)], &moe->blocks_down[size_t(e)], eact_s_, F, ne);
      // forward: down -> weighted scatter into y
      g.run(true, false, H, int(ne), F, w_exp_down_ + size_t(e) * H * F, F, act_, F, ey_, H, OutT::F32, 1.f,
            0.f);
      gemm_flops += 2.0 * H * F * ne;
      k_q_moe_scatter_add(y_, ey_, tok_e, w_e, ne, int(R), H, s_);
    }
    // --- shared expert (pooled Hessians = HessAcc slots 2/3) + sigmoid gate
    proj(h_, H, w_sh_gu_, S, gu_, 2 * S);
    proj(h_, H, w_sh_gu_ + size_t(S) * H, S, gu_ + S, 2 * S);
    k_silu_mul(gu_, ones_, act_, R, S, s_);
    hess(3, act_, S);
    proj(act_, S, w_sh_down_, H, ey_, H);
    if (cfg_.shared_gate) {
      proj(h_, H, w_sg_, 1, esg_, 1);
      k_q_sigmoid_scale_add(y_, ey_, esg_, int(R), H, s_);
    } else {
      k_q_sigmoid_scale_add(y_, ey_, nullptr, int(R), H, s_);
    }
  }
  k_residual_add(x_, y_, ones_, R, H, s_);
}

}  // namespace qtip
