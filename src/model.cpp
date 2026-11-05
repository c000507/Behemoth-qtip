#include "model.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "qtip_core.hpp"

namespace qtip {

const char* kLinName[kNumLin] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
const char* kHessName[4] = {"qkv", "o", "gateup", "down"};

void llama3_inv_freq(int hd, double theta, const RopeScalingLlama3& r, std::vector<float>& out) {
  // HF transformers _compute_llama3_parameters: per-dim wavelength interpolation between the
  // unscaled frequency and the frequency divided by `factor`.
  const double low_wavelen = double(r.original_max_position_embeddings) / r.low_freq_factor;
  const double high_wavelen = double(r.original_max_position_embeddings) / r.high_freq_factor;
  out.resize(size_t(hd / 2));
  for (int d = 0; d < hd / 2; ++d) {
    double freq = std::pow(theta, -2.0 * d / hd);
    double wavelen = 2.0 * M_PI / freq;
    double scale;
    if (wavelen < high_wavelen)
      scale = 1.0;
    else if (wavelen > low_wavelen)
      scale = 1.0 / r.factor;
    else {
      double smooth = (double(r.original_max_position_embeddings) / wavelen - r.low_freq_factor) /
                      (r.high_freq_factor - r.low_freq_factor);
      scale = (1.0 - smooth) / r.factor + smooth;
    }
    out[size_t(d)] = float(freq * scale);
  }
}

ModelConfig load_model_config(const std::string& dir, const SafeTensorsReader* st) {
  std::ifstream f(dir + "/config.json");
  QCHECK(f, "no config.json in " + dir);
  std::stringstream ss;
  ss << f.rdbuf();
  ModelConfig c;
  c.raw_json = ss.str();
  auto j = nlohmann::json::parse(c.raw_json);
  c.hidden = j.at("hidden_size");
  c.inter = j.at("intermediate_size");
  c.nq = j.at("num_attention_heads");
  c.nkv = j.at("num_key_value_heads");
  c.hd = j.value("head_dim", int(c.hidden / c.nq));
  c.layers = j.at("num_hidden_layers");
  c.vocab = j.at("vocab_size");
  c.eps = j.value("rms_norm_eps", 1e-5f);
  c.theta = j.value("rope_theta", 1e6f);
  c.max_pos = j.value("max_position_embeddings", 131072L);
  // rope_scaling: only HF type "llama3" is supported (e.g. Llama 3.3 70B).
  if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
    const auto& rs = j["rope_scaling"];
    std::string type = rs.value("rope_type", rs.value("type", std::string()));
    QCHECK(type == "llama3", "unsupported rope_scaling type '" + type + "' (only llama3)");
    c.rope.factor = rs.at("factor");
    c.rope.low_freq_factor = rs.at("low_freq_factor");
    c.rope.high_freq_factor = rs.at("high_freq_factor");
    c.rope.original_max_position_embeddings = rs.at("original_max_position_embeddings");
    QCHECK(c.rope.factor > 0 && c.rope.low_freq_factor > 0 &&
               c.rope.high_freq_factor > c.rope.low_freq_factor &&
               c.rope.original_max_position_embeddings > 0,
           "invalid llama3 rope_scaling factors");
    c.rope_llama3 = true;
  }
  // shape validation: everything the QTIP pipeline (RHT factorisation, 16x16 trellis kernel layout,
  // LDLQ 128-column buffers, attention tiles) requires
  auto positive = [&](long v, const char* what) { QCHECK(v > 0, std::string(what) + " must be positive"); };
  positive(c.hidden, "hidden_size");
  positive(c.inter, "intermediate_size");
  positive(c.layers, "num_hidden_layers");
  positive(c.vocab, "vocab_size");
  positive(c.nq, "num_attention_heads");
  positive(c.nkv, "num_key_value_heads");
  QCHECK(c.hd > 0 && c.hd <= 256 && c.hd % 2 == 0 && c.nq % c.nkv == 0,
         "invalid attention geometry (head_dim must be even and <=256, heads must be a GQA multiple)");
  auto mult32 = [&](long v, const char* what) {
    QCHECK(v % 32 == 0, std::string(what) + " must be a multiple of 32");
  };
  mult32(c.hidden, "hidden_size");
  mult32(c.inter, "intermediate_size");
  mult32(long(c.nq) * c.hd, "num_attention_heads * head_dim");
  mult32(long(c.nkv) * c.hd, "num_key_value_heads * head_dim");
  QCHECK(c.hidden % 128 == 0 && c.inter % 128 == 0, "hidden_size and intermediate_size must be multiples of 128");
  for (int n : {c.hidden, c.inter, c.nq * c.hd, c.nkv * c.hd}) {
    int K = had_K(n);
    QCHECK(K > 0, "unsupported Hadamard dimension " + std::to_string(n) + " (need 2^p, 12*2^p or 28*2^p)");
    QCHECK(n / K <= 8192, "Hadamard power-of-two part of " + std::to_string(n) + " exceeds 8192");
  }
  // source weight format: FP8-Dynamic (F8_E4M3 + weight_scale) or plain BF16
  if (st) {
    const TensorInfo& w = st->info("model.layers.0.self_attn.q_proj.weight");
    if (w.dtype == "F8_E4M3") {
      QCHECK(st->has("model.layers.0.self_attn.q_proj.weight_scale"), "FP8 source needs weight_scale tensors");
      c.bf16_source = false;
    } else if (w.dtype == "BF16") {
      c.bf16_source = true;
    } else {
      die("unsupported source weight dtype " + w.dtype + " (need F8_E4M3 FP8-Dynamic or BF16)");
    }
  }
  return c;
}

std::shared_ptr<LayerHost> read_layer(const SafeTensorsReader& st, const ModelConfig& cfg, int idx) {
  auto L = std::make_shared<LayerHost>();
  L->idx = idx;
  L->bf16 = cfg.bf16_source;
  double t0 = now_s();
  std::string pre = "model.layers." + std::to_string(idx) + ".";
  for (int i = 0; i < kNumLin; ++i) {
    const TensorInfo& w = st.info(pre + kLinName[i] + ".weight");
    L->out_dim[i] = int(w.shape[0]);
    L->in_dim[i] = int(w.shape[1]);
    if (L->bf16) {
      QCHECK(w.dtype == "BF16" && w.shape.size() == 2, std::string("expected BF16 weight ") + kLinName[i]);
      L->w[i].resize(w.bytes());
      st.read_into(pre + kLinName[i] + ".weight", L->w[i].data(), L->w[i].size());
      continue;
    }
    QCHECK(w.dtype == "F8_E4M3" && w.shape.size() == 2, std::string("unexpected weight ") + kLinName[i]);
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
  QCHECK(T_ > 0 && R > 0 && R % T_ == 0, "max_rows must be a positive multiple of the sequence length");
  QCHECK(cfg.hd > 0 && cfg.hd <= 256 && cfg.hd % 2 == 0 && cfg.nq > 0 && cfg.nkv > 0 && cfg.nq % cfg.nkv == 0,
         "invalid attention geometry (head_dim must be even and <=256)");
  wqkv_.alloc(size_t(nqkv) * H);
  wo_.alloc(size_t(H) * cfg.nq * cfg.hd);
  wgu_.alloc(size_t(2 * F) * H);
  wd_.alloc(size_t(H) * F);
  ln_in_.alloc(static_cast<size_t>(H));
  ln_post_.alloc(static_cast<size_t>(H));
  sqkv_.alloc(static_cast<size_t>(nqkv));
  so_.alloc(static_cast<size_t>(H));
  sgu_.alloc(size_t(2 * F));
  sd_.alloc(static_cast<size_t>(H));
  stage8_.alloc(size_t(F) * H);
  x_.alloc(size_t(R) * H);
  h_.alloc(size_t(R) * H);
  attn_.alloc(size_t(R) * cfg.nq * cfg.hd);
  act_.alloc(size_t(R) * F);
  qkv_.alloc(size_t(R) * nqkv);
  y_.alloc(size_t(R) * H);
  gu_.alloc(size_t(R) * 2 * F);
  Qp_.alloc(size_t(cfg.nq) * T_ * cfg.hd);
  Kp_.alloc(size_t(cfg.nkv) * T_ * cfg.hd);
  Vp_.alloc(size_t(cfg.nkv) * T_ * cfg.hd);
  pin_.alloc(size_t(R) * H);
  if (cfg_.rope_llama3) {
    std::vector<float> f;
    llama3_inv_freq(cfg_.hd, cfg_.theta, cfg_.rope, f);
    inv_freq_.alloc(f.size());
    HIP_CHECK(hipMemcpy(inv_freq_, f.data(), f.size() * 4, hipMemcpyHostToDevice));
  }
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
void LayerRunner::symmetrize_hess(HessAcc& a) {
  for (int i = 0; i < 4; ++i)
    if (a.blocks[i] > 1) k_symmetrize_blocks(a.h[i], a.dim[i], a.dim[i] / a.blocks[i], s_);
  HIP_CHECK(hipStreamSynchronize(s_));
}
void LayerRunner::zero_hess(HessAcc& a) {
  for (int i = 0; i < 4; ++i) HIP_CHECK(hipMemsetAsync(a.h[i], 0, a.h[i].bytes(), s_));
  a.count = 0;
}

void LayerRunner::upload(const LayerHost& L) {
  HIP_CHECK(hipSetDevice(dev_));
  auto up = [&](int li, uint16_t* dst, float* sdst) {
    if (L.bf16) {
      HIP_CHECK(hipMemcpyAsync(dst, L.w[li].data(), L.w[li].size(), hipMemcpyHostToDevice, s_));
      std::vector<float> ones(L.out_dim[li], 1.f);
      HIP_CHECK(hipMemcpyAsync(sdst, ones.data(), ones.size() * 4, hipMemcpyHostToDevice, s_));
      return;
    }
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

void LayerRunner::upload_raw(int li, const uint16_t* w_bf16, const float* scale) {
  HIP_CHECK(hipSetDevice(dev_));
  const long H = cfg_.hidden, F = cfg_.inter;
  long qo = long(cfg_.nq) * cfg_.hd, ko = long(cfg_.nkv) * cfg_.hd;
  uint16_t* dst;
  float* sdst;
  long m;
  switch (li) {
    case LQ: dst = wqkv_, sdst = sqkv_, m = qo; break;
    case LK: dst = wqkv_ + qo * H, sdst = sqkv_ + qo, m = ko; break;
    case LV: dst = wqkv_ + (qo + ko) * H, sdst = sqkv_ + qo + ko, m = ko; break;
    case LO: dst = wo_, sdst = so_, m = H; break;
    case LGate: dst = wgu_, sdst = sgu_, m = F; break;
    case LUp: dst = wgu_ + F * H, sdst = sgu_ + F, m = F; break;
    case LDown: dst = wd_, sdst = sd_, m = H; break;
    default: QCHECK(false, "bad linear index"); return;
  }
  long in = (li == LDown) ? F : H;
  HIP_CHECK(hipMemcpyAsync(dst, w_bf16, size_t(m) * in * 2, hipMemcpyHostToDevice, s_));
  if (scale) {
    HIP_CHECK(hipMemcpyAsync(sdst, scale, size_t(m) * 4, hipMemcpyHostToDevice, s_));
  } else {
    std::vector<float> ones(size_t(m), 1.f);
    HIP_CHECK(hipMemcpyAsync(sdst, ones.data(), ones.size() * 4, hipMemcpyHostToDevice, s_));
  }
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
  const int nqkv = (nq + 2 * nkv) * hd, A = nq * hd;
  Gemm& g = *gemm_;
  auto hess = [&](int which, const uint16_t* X, int n) {
    if (!acc) return;
    // H(n x n) += X^T X ; X row-major (R x n) == col-major (n x R). Only the 10 blocks (bi <= bj) of a 4 x 4 block
    // partition are accumulated (symmetry); symmetrize_hess() fills the rest before the Hessians are read.
    const int nb = (n % 4 == 0 && (n / 4) % 16 == 0) ? 4 : 1, bs = n / nb;
    for (int bi = 0; bi < nb; ++bi)
      for (int bj = bi; bj < nb; ++bj)
        g.run(false, true, bs, bs, int(R), X + long(bi) * bs, n, X + long(bj) * bs, n,
              acc->h[which] + long(bj) * bs * n + long(bi) * bs, n, OutT::F32, 1.f, 1.f);
    acc->blocks[which] = nb;
    gemm_flops += 2.0 * n * n * R * (nb * (nb + 1) / 2) / double(nb * nb);
  };
  auto proj = [&](const uint16_t* X, int in, const uint16_t* W, int out, float* Y) {
    // Y (R x out) = X (R x in) W^T ; col-major: Y_cm(out x R) = W_cm^T (out x in) X_cm (in x R)
    g.run(true, false, out, int(R), in, W, in, X, in, Y, out, OutT::F32, 1.f, 0.f);
    gemm_flops += 2.0 * out * in * R;
  };
  k_rmsnorm(x_, ln_in_, h_, R, H, cfg_.eps, s_);
  hess(0, h_, H);
  proj(h_, H, wqkv_, nqkv, qkv_);
  for (long sq = 0; sq < R / T; ++sq) {
    if (cfg_.rope_llama3)
      k_qkv_rope_scaled(qkv_ + sq * T * long(nqkv), sqkv_, Qp_, Kp_, Vp_, T, nq, nkv, hd, inv_freq_, s_);
    else
      k_qkv_rope(qkv_ + sq * T * long(nqkv), sqkv_, Qp_, Kp_, Vp_, T, nq, nkv, hd, cfg_.theta, s_);
    k_tiled_attention(Qp_, Kp_, Vp_, attn_ + sq * T * long(A), T, nq, nkv, hd, s_);
    // Causal useful work; the final 16-query tile also computes masked entries.
    gemm_flops += 2.0 * nq * hd * T * double(T + 1);
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
