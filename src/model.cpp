#include "model.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace qtip {

const char* kLinName[kNumLin] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
const char* kHessName[4] = {"qkv", "o", "gateup", "down"};

ModelConfig load_model_config(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  QCHECK(f, "no config.json in " + dir);
  std::stringstream ss;
  ss << f.rdbuf();
  ModelConfig c;
  c.raw_json = ss.str();
  auto j = nlohmann::json::parse(c.raw_json);
  c.hidden = j["hidden_size"];
  c.inter = j["intermediate_size"];
  c.nq = j["num_attention_heads"];
  c.nkv = j["num_key_value_heads"];
  c.hd = j.value("head_dim", c.hidden / c.nq);
  c.layers = j["num_hidden_layers"];
  c.vocab = j["vocab_size"];
  c.eps = j.value("rms_norm_eps", 1e-5f);
  c.theta = j.value("rope_theta", 1e6f);
  return c;
}

std::shared_ptr<LayerHost> read_layer(const SafeTensorsReader& st, const ModelConfig& cfg, int idx) {
  auto L = std::make_shared<LayerHost>();
  L->idx = idx;
  double t0 = now_s();
  std::string pre = "model.layers." + std::to_string(idx) + ".";
  for (int i = 0; i < kNumLin; ++i) {
    const TensorInfo& w = st.info(pre + kLinName[i] + ".weight");
    QCHECK(w.dtype == "F8_E4M3" && w.shape.size() == 2, std::string("unexpected weight ") + kLinName[i]);
    L->out_dim[i] = int(w.shape[0]);
    L->in_dim[i] = int(w.shape[1]);
    L->w[i].resize(w.bytes());
    st.read_into(pre + kLinName[i] + ".weight", L->w[i].data(), L->w[i].size());
    const TensorInfo& s = st.info(pre + kLinName[i] + ".weight_scale");
    QCHECK(s.dtype == "BF16" && s.numel() == w.shape[0], "unexpected weight_scale");
    L->sc[i].resize(size_t(s.numel()));
    st.read_into(pre + kLinName[i] + ".weight_scale", L->sc[i].data(), L->sc[i].size() * 2);
  }
  L->ln_in.resize(size_t(cfg.hidden));
  L->ln_post.resize(size_t(cfg.hidden));
  st.read_into(pre + "input_layernorm.weight", L->ln_in.data(), L->ln_in.size() * 2);
  st.read_into(pre + "post_attention_layernorm.weight", L->ln_post.data(), L->ln_post.size() * 2);
  L->read_s = now_s() - t0;
  return L;
}

LayerCache::LayerCache(const SafeTensorsReader& st, const ModelConfig& cfg, int capacity)
    : st_(st), cfg_(cfg), cap_(capacity), th_([this] { worker(); }) {}
LayerCache::~LayerCache() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  th_.join();
}
void LayerCache::prefetch(int idx) {
  if (idx < 0 || idx >= cfg_.layers) return;
  std::lock_guard<std::mutex> lk(mu_);
  if (have_.count(idx) || loading_.count(idx)) return;
  for (int q : queue_)
    if (q == idx) return;
  queue_.push_back(idx);
  cv_.notify_all();
}
void LayerCache::worker() {
  for (;;) {
    int idx;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (stop_) return;
      idx = queue_.front();
      queue_.erase(queue_.begin());
      if (have_.count(idx) || loading_.count(idx)) continue;
      loading_[idx] = true;
    }
    auto L = read_layer(st_, cfg_, idx);
    {
      std::lock_guard<std::mutex> lk(mu_);
      have_[idx] = L;
      last_use_[idx] = ++tick_;
      loading_.erase(idx);
      while (int(have_.size()) > cap_) {  // evict least recently used
        int victim = -1;
        long best = 1L << 62;
        for (auto& kv : last_use_)
          if (kv.second < best && kv.first != idx) best = kv.second, victim = kv.first;
        if (victim < 0) break;
        have_.erase(victim);
        last_use_.erase(victim);
      }
    }
    cv_.notify_all();
  }
}
std::shared_ptr<LayerHost> LayerCache::get(int idx) {
  prefetch(idx);
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return have_.count(idx) != 0; });
  last_use_[idx] = ++tick_;
  return have_[idx];
}

// ------------------------------------------------------------------ LayerRunner
LayerRunner::LayerRunner(int device, const ModelConfig& cfg, int max_rows, int seq_len)
    : dev_(device), cfg_(cfg), max_rows_(max_rows), T_(seq_len) {
  HIP_CHECK(hipSetDevice(dev_));
  HIP_CHECK(hipStreamCreateWithFlags(&s_, hipStreamNonBlocking));
  gemm_ = std::make_unique<Gemm>(s_);
  const long H = cfg.hidden, F = cfg.inter, R = max_rows;
  const long nqkv = long(cfg.nq + 2 * cfg.nkv) * cfg.hd;
  QCHECK(R % T_ == 0, "max_rows must be a multiple of the sequence length");
  wqkv_.alloc(size_t(nqkv) * H);
  wo_.alloc(size_t(H) * cfg.nq * cfg.hd);
  wgu_.alloc(size_t(2 * F) * H);
  wd_.alloc(size_t(H) * F);
  ln_in_.alloc(size_t(H));
  ln_post_.alloc(size_t(H));
  sqkv_.alloc(size_t(nqkv));
  so_.alloc(size_t(H));
  sgu_.alloc(size_t(2 * F));
  sd_.alloc(size_t(H));
  stage8_.alloc(size_t(F) * H);
  x_.alloc(size_t(R) * H);
  h_.alloc(size_t(R) * H);
  attn_.alloc(size_t(R) * cfg.nq * cfg.hd);
  act_.alloc(size_t(R) * F);
  qkv_.alloc(size_t(R) * nqkv);
  y_.alloc(size_t(R) * H);
  gu_.alloc(size_t(R) * 2 * F);
  const int gq = cfg.nq / cfg.nkv;
  Qp_.alloc(size_t(cfg.nq) * T_ * cfg.hd);
  Kp_.alloc(size_t(cfg.nkv) * T_ * cfg.hd);
  Vp_.alloc(size_t(cfg.nkv) * T_ * cfg.hd);
  S_.alloc(size_t(gq) * T_ * T_);
  P_.alloc(size_t(gq) * T_ * T_);
  O_.alloc(size_t(gq) * T_ * cfg.hd);
  pin_.alloc(size_t(R) * H);
}
LayerRunner::~LayerRunner() { (void)hipStreamDestroy(s_); }

void LayerRunner::alloc_hess(HessAcc& a) {
  HIP_CHECK(hipSetDevice(dev_));
  a.dim[0] = cfg_.hidden;
  a.dim[1] = cfg_.nq * cfg_.hd;
  a.dim[2] = cfg_.hidden;
  a.dim[3] = cfg_.inter;
  for (int i = 0; i < 4; ++i) a.h[i].alloc(size_t(a.dim[i]) * a.dim[i]);
  zero_hess(a);
}
void LayerRunner::zero_hess(HessAcc& a) {
  for (int i = 0; i < 4; ++i) HIP_CHECK(hipMemsetAsync(a.h[i], 0, a.h[i].bytes(), s_));
  a.count = 0;
}

void LayerRunner::upload(const LayerHost& L) {
  HIP_CHECK(hipSetDevice(dev_));
  auto up = [&](int li, uint16_t* dst, float* sdst) {
    size_t n = L.w[li].size();
    HIP_CHECK(hipMemcpyAsync(stage8_, L.w[li].data(), n, hipMemcpyHostToDevice, s_));
    k_fp8_to_bf16(stage8_, dst, long(n), s_);
    std::vector<float> sc(L.sc[li].size());
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = bf16_to_f32(L.sc[li][i]);
    HIP_CHECK(hipMemcpyAsync(sdst, sc.data(), sc.size() * 4, hipMemcpyHostToDevice, s_));
    HIP_CHECK(hipStreamSynchronize(s_));  // sc and stage8_ reuse
  };
  const long H = cfg_.hidden, F = cfg_.inter;
  long qo = long(cfg_.nq) * cfg_.hd, ko = long(cfg_.nkv) * cfg_.hd;
  up(LQ, wqkv_, sqkv_);
  up(LK, wqkv_ + qo * H, sqkv_ + qo);
  up(LV, wqkv_ + (qo + ko) * H, sqkv_ + qo + ko);
  up(LO, wo_, so_);
  up(LGate, wgu_, sgu_);
  up(LUp, wgu_ + F * H, sgu_ + F);
  up(LDown, wd_, sd_);
  HIP_CHECK(hipMemcpyAsync(ln_in_, L.ln_in.data(), size_t(H) * 2, hipMemcpyHostToDevice, s_));
  HIP_CHECK(hipMemcpyAsync(ln_post_, L.ln_post.data(), size_t(H) * 2, hipMemcpyHostToDevice, s_));
  HIP_CHECK(hipStreamSynchronize(s_));
}

void LayerRunner::forward(uint16_t* host_x, long nseq, HessAcc* acc) {
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

void LayerRunner::forward_batch(long R, HessAcc* acc) {
  const int H = cfg_.hidden, F = cfg_.inter, hd = cfg_.hd, nq = cfg_.nq, nkv = cfg_.nkv, T = T_;
  const int gq = nq / nkv, nqkv = (nq + 2 * nkv) * hd, A = nq * hd;
  Gemm& g = *gemm_;
  auto hess = [&](int which, const uint16_t* X, int n) {
    if (!acc) return;
    // H(n x n) += X^T X ; X row-major (R x n) == col-major (n x R)
    g.run(false, true, n, n, int(R), X, n, X, n, acc->h[which], n, OutT::F32, 1.f, 1.f);
    gemm_flops += 2.0 * n * n * R;
  };
  auto proj = [&](const uint16_t* X, int in, const uint16_t* W, int out, float* Y) {
    // Y (R x out) = X (R x in) W^T ; col-major: Y_cm(out x R) = W_cm^T (out x in) X_cm (in x R)
    g.run(true, false, out, int(R), in, W, in, X, in, Y, out, OutT::F32, 1.f, 0.f);
    gemm_flops += 2.0 * out * in * R;
  };
  k_rmsnorm(x_, ln_in_, h_, R, H, cfg_.eps, s_);
  hess(0, h_, H);
  proj(h_, H, wqkv_, nqkv, qkv_);
  const float sm_scale = 1.0f / std::sqrt(float(hd));
  for (long sq = 0; sq < R / T; ++sq) {
    k_qkv_rope(qkv_ + sq * T * long(nqkv), sqkv_, Qp_, Kp_, Vp_, T, nq, nkv, hd, cfg_.theta, s_);
    for (int kv = 0; kv < nkv; ++kv) {
      const uint16_t* K = Kp_ + long(kv) * T * hd;
      const uint16_t* V = Vp_ + long(kv) * T * hd;
      const uint16_t* Q = Qp_ + long(kv) * gq * T * hd;
      // S row-major (gq*T x T) = Q K^T ; col-major S_cm (T x gq*T) = K_cm^T (T x hd) Q_cm (hd x gq*T)
      g.run(true, false, T, gq * T, hd, K, hd, Q, hd, S_, T, OutT::F32, 1.f, 0.f);
      k_causal_softmax(S_, P_, gq, T, sm_scale, s_);
      // O row-major (gq*T x hd) = P V ; col-major O_cm (hd x gq*T) = V_cm (hd x T) P_cm (T x gq*T)
      g.run(false, false, hd, gq * T, T, V, hd, P_, T, O_, hd, OutT::F32, 1.f, 0.f);
      k_heads_to_tokens(O_, attn_ + sq * T * long(A), gq, T, hd, kv * gq, A, s_);
      gemm_flops += 4.0 * gq * T * double(T) * hd;
    }
  }
  hess(1, attn_, A);
  proj(attn_, A, wo_, H, y_);
  k_residual_add(x_, y_, so_, R, H, s_);
  k_rmsnorm(x_, ln_post_, h_, R, H, cfg_.eps, s_);
  hess(2, h_, H);
  proj(h_, H, wgu_, 2 * F, gu_);
  k_silu_mul(gu_, sgu_, act_, R, F, s_);
  hess(3, act_, F);
  proj(act_, F, wd_, H, y_);
  k_residual_add(x_, y_, sd_, R, H, s_);
}

}  // namespace qtip
