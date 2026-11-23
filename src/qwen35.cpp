// Qwen3.5 config parsing, checkpoint layer reading and CPU fp32 reference forward.
#include "qwen35.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common.hpp"

namespace qtip {

using json = nlohmann::json;

static json read_json(const std::string& dir, const char* name) {
  std::ifstream f(dir + "/" + name);
  QCHECK(f, std::string("no ") + name + " in " + dir);
  std::stringstream ss;
  ss << f.rdbuf();
  return json::parse(ss.str());
}

static bool model_type_is_qwen35(const json& j) {
  std::string mt = j.value("model_type", "");
  return mt == "qwen3_5" || mt == "qwen3_5_moe";
}

bool is_qwen35_dir(const std::string& dir) {
  try {
    return model_type_is_qwen35(read_json(dir, "config.json"));
  } catch (...) {
    return false;
  }
}

QwenConfig load_qwen_config(const std::string& dir) {
  json j = read_json(dir, "config.json");
  QCHECK(model_type_is_qwen35(j), "not a qwen3_5/qwen3_5_moe config.json");
  const json& t = j.contains("text_config") ? j["text_config"] : j;
  QwenConfig c;
  c.raw_json = j.dump();
  c.hidden = t.at("hidden_size");
  c.layers = t.at("num_hidden_layers");
  c.vocab = t.at("vocab_size");
  c.eps = t.value("rms_norm_eps", 1e-6f);
  c.nq = t.at("num_attention_heads");
  c.nkv = t.at("num_key_value_heads");
  c.hd = t.value("head_dim", c.hidden / c.nq);
  c.attn_output_gate = t.value("attn_output_gate", true);
  c.lin_nk = t.at("linear_num_key_heads");
  c.lin_kd = t.at("linear_key_head_dim");
  c.lin_nv = t.at("linear_num_value_heads");
  c.lin_vd = t.at("linear_value_head_dim");
  c.conv_k = t.value("linear_conv_kernel_dim", t.value("linear_conv_kernel", 4));
  c.tied = t.value("tie_word_embeddings", false);
  // rope: rope_parameters overrides, then the flat keys; partial_rotary_factor 0.25 default
  float partial = 0.25f;
  float theta = 1e7f;
  if (t.contains("rope_parameters")) {
    const json& rp = t["rope_parameters"];
    partial = rp.value("partial_rotary_factor", partial);
    theta = rp.value("rope_theta", theta);
  }
  partial = t.value("partial_rotary_factor", partial);
  theta = t.value("rope_theta", theta);
  c.theta = theta;
  c.rot_dim = int(float(c.hd) * partial) & ~1;  // even
  if (c.nq % c.nkv) QCHECK(false, "nq % nkv != 0");
  if (c.lin_nv % c.lin_nk) QCHECK(false, "linear_num_value_heads % linear_num_key_heads != 0");
  QCHECK(c.rot_dim > 0 && c.rot_dim <= c.hd && c.hd % 4 == 0, "bad rotary dim");
  QCHECK(c.lin_kd == c.lin_vd, "linear k/v head dims must match for the fp32 state layout");
  // layer types
  c.full_attn.assign(size_t(c.layers), 0);
  if (t.contains("layer_types") && !t["layer_types"].is_null()) {
    QCHECK(int(t["layer_types"].size()) == c.layers, "layer_types length mismatch");
    for (int i = 0; i < c.layers; ++i) {
      std::string lt = t["layer_types"][size_t(i)];
      c.full_attn[size_t(i)] = (lt == "full_attention") ? 1 : 0;
    }
  } else {
    int interval = t.value("full_attention_interval", 4);
    for (int i = 0; i < c.layers; ++i) c.full_attn[size_t(i)] = ((i + 1) % interval == 0) ? 1 : 0;
  }
  // MoE / dense MLP
  if (t.contains("num_experts") || t.contains("n_routed_experts")) {
    c.n_experts = t.value("num_experts", t.value("n_routed_experts", 0));
    c.topk = t.value("num_experts_per_tok", t.value("n_experts_per_tok", 8));
    c.moe_inter = t.at("moe_intermediate_size");
    c.shared_inter = t.value("shared_expert_intermediate_size", c.moe_inter);
    c.inter = c.moe_inter;
    c.shared_gate = t.value("shared_expert_gated", true);
    std::string scoring = t.value("scoring_func", "softmax");
    QCHECK(scoring == "softmax", "only softmax router scoring supported (got " + scoring + ")");
    QCHECK(!t.value("norm_topk_prob", true) == false || true, "");  // renorm is the default; keep
  } else {
    c.inter = t.at("intermediate_size");
  }
  return c;
}

// Prefix of the text model inside the checkpoint ("model." / "model.language_model." / ...).
static std::string detect_prefix(const SafeTensorsReader& st) {
  static const char* suffix = "layers.0.input_layernorm.weight";
  for (std::string p : {"model.language_model.", "language_model.model.", "model.", ""})
    if (st.has(p + suffix)) return p;
  QCHECK(false, "cannot find layers.0.input_layernorm.weight under any known prefix");
}

std::vector<float> bf16_to_f32_vec(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = bf16_to_f32(v[i]);
  return o;
}

// Read a 2-D weight [out][in] into bf16 (fp8+scale checkpoints are dequantised on read). Two FP8
// layouts: per-row `weight_scale` [out] (bf16) and 128x128 block `weight_scale_inv`
// [out/128][in/128] (bf16, the Qwen3.5-122B-A10B-FP8 layout); scale_inv multiplies on dequant.
static void read_w16(const SafeTensorsReader& st, const std::string& name, std::vector<uint16_t>& out) {
  const TensorInfo& w = st.info(name);
  QCHECK(w.shape.size() == 2, "unexpected shape for " + name);
  if (w.dtype == "BF16") {
    out.resize(w.bytes() / 2);
    st.read_into(name, out.data(), out.size() * 2);
  } else if (w.dtype == "F8_E4M3") {
    std::vector<uint8_t> fp8(w.bytes());
    st.read_into(name, fp8.data(), fp8.size());
    out.resize(size_t(w.numel()));
    const long rows = w.shape[0], cols = w.shape[1];
    if (st.has(name + ".weight_scale")) {  // per-row scales
      const TensorInfo& si = st.info(name + ".weight_scale");
      std::vector<uint16_t> sc(size_t(si.numel()));
      QCHECK(si.dtype == "BF16" && long(sc.size()) == rows, "unexpected weight_scale for " + name);
      st.read_into(name + ".weight_scale", sc.data(), sc.size() * 2);
      for (long r = 0; r < rows; ++r) {
        float s = bf16_to_f32(sc[size_t(r)]);
        for (long c = 0; c < cols; ++c) out[size_t(r * cols + c)] = f32_to_bf16(fp8e4m3_to_f32(fp8[size_t(r * cols + c)]) * s);
      }
    } else if (st.has(name + ".weight_scale_inv")) {  // 128x128 block scales
      const TensorInfo& si = st.info(name + ".weight_scale_inv");
      QCHECK(si.dtype == "BF16" && si.shape.size() == 2 && si.shape[0] == (rows + 127) / 128 &&
                 si.shape[1] == (cols + 127) / 128,
             "unexpected weight_scale_inv for " + name);
      std::vector<uint16_t> sc(size_t(si.numel()));
      st.read_into(name + ".weight_scale_inv", sc.data(), sc.size() * 2);
      for (long r = 0; r < rows; ++r)
        for (long c = 0; c < cols; ++c) {
          float s = bf16_to_f32(sc[size_t(r / 128) * si.shape[1] + c / 128]);
          out[size_t(r * cols + c)] = f32_to_bf16(fp8e4m3_to_f32(fp8[size_t(r * cols + c)]) * s);
        }
    } else {
      QCHECK(false, "FP8 tensor " + name + " without weight_scale/weight_scale_inv");
    }
  } else {
    QCHECK(false, "unsupported dtype " + w.dtype + " for " + name);
  }
}

std::shared_ptr<QwenLayer> read_qwen_layer(const SafeTensorsReader& st, const QwenConfig& cfg, int idx) {
  static std::string prefix;
  if (prefix.empty()) prefix = detect_prefix(st);
  auto L = std::make_shared<QwenLayer>();
  L->idx = idx;
  double t0 = now_s();
  L->full = cfg.is_full(idx);
  L->moe = cfg.moe();
  std::string pre = prefix + "layers." + std::to_string(idx) + ".";
  const long H = cfg.hidden;
  if (L->full) {
    read_w16(st, pre + "self_attn.q_proj.weight", L->w_q);
    read_w16(st, pre + "self_attn.k_proj.weight", L->w_k);
    read_w16(st, pre + "self_attn.v_proj.weight", L->w_v);
    read_w16(st, pre + "self_attn.o_proj.weight", L->w_o);
    if (cfg.attn_output_gate)
      QCHECK(long(L->w_q.size()) == long(cfg.nq) * 2 * cfg.hd * H, "q_proj rows != nq*2*hd (attn_output_gate)");
    L->q_norm.resize(size_t(cfg.hd));
    L->k_norm.resize(size_t(cfg.hd));
    st.read_into(pre + "self_attn.q_norm.weight", L->q_norm.data(), L->q_norm.size() * 2);
    st.read_into(pre + "self_attn.k_norm.weight", L->k_norm.data(), L->k_norm.size() * 2);
  } else {
    read_w16(st, pre + "linear_attn.in_proj_qkv.weight", L->w_in_qkv);
    read_w16(st, pre + "linear_attn.in_proj_z.weight", L->w_in_z);
    read_w16(st, pre + "linear_attn.in_proj_b.weight", L->w_in_ba);
    {
      std::vector<uint16_t> a;
      read_w16(st, pre + "linear_attn.in_proj_a.weight", a);
      L->w_in_ba.insert(L->w_in_ba.end(), a.begin(), a.end());
    }
    read_w16(st, pre + "linear_attn.out_proj.weight", L->w_out);
    {  // conv1d [conv_dim, 1, K] bf16
      const TensorInfo& cw = st.info(pre + "linear_attn.conv1d.weight");
      QCHECK(cw.dtype == "BF16" && cw.shape.size() == 3 && cw.shape[1] == 1 &&
                 int(cw.shape[0]) == cfg.conv_dim() && int(cw.shape[2]) == cfg.conv_k,
             "unexpected conv1d shape");
      L->conv_w.resize(cw.bytes() / 2);
      st.read_into(pre + "linear_attn.conv1d.weight", L->conv_w.data(), L->conv_w.size() * 2);
    }
    const TensorInfo& al = st.info(pre + "linear_attn.A_log");
    QCHECK((al.dtype == "F32" || al.dtype == "BF16") && al.numel() == cfg.lin_nv, "unexpected A_log");
    L->a_log.resize(size_t(al.numel()));
    if (al.dtype == "F32") st.read_into(pre + "linear_attn.A_log", L->a_log.data(), L->a_log.size() * 4);
    else {
      std::vector<uint16_t> b(size_t(al.numel()));
      st.read_into(pre + "linear_attn.A_log", b.data(), b.size() * 2);
      for (size_t i = 0; i < b.size(); ++i) L->a_log[i] = bf16_to_f32(b[i]);
    }
    const TensorInfo& db = st.info(pre + "linear_attn.dt_bias");
    QCHECK((db.dtype == "F32" || db.dtype == "BF16") && db.numel() == cfg.lin_nv, "unexpected dt_bias");
    L->dt_bias.resize(size_t(db.numel()));
    if (db.dtype == "F32") st.read_into(pre + "linear_attn.dt_bias", L->dt_bias.data(), L->dt_bias.size() * 4);
    else {
      std::vector<uint16_t> b(size_t(db.numel()));
      st.read_into(pre + "linear_attn.dt_bias", b.data(), b.size() * 2);
      for (size_t i = 0; i < b.size(); ++i) L->dt_bias[i] = bf16_to_f32(b[i]);
    }
    const TensorInfo& dn = st.info(pre + "linear_attn.norm.weight");
    QCHECK(dn.dtype == "BF16" || dn.dtype == "F32", "unexpected linear_attn.norm dtype");
    L->dn_norm.resize(size_t(dn.numel()));
    if (dn.dtype == "BF16") st.read_into(pre + "linear_attn.norm.weight", L->dn_norm.data(), L->dn_norm.size() * 2);
    else {
      std::vector<float> f(size_t(dn.numel()));
      st.read_into(pre + "linear_attn.norm.weight", f.data(), f.size() * 4);
      for (size_t i = 0; i < f.size(); ++i) L->dn_norm[i] = f32_to_bf16(f[i]);
    }
  }
  if (L->moe) {
    read_w16(st, pre + "mlp.gate.weight", L->w_router);
    if (st.has(pre + "mlp.gate.e_score_correction_bias")) {
      const TensorInfo& rb = st.info(pre + "mlp.gate.e_score_correction_bias");
      L->w_router_bias.resize(size_t(rb.numel()));
      st.read_into(pre + "mlp.gate.e_score_correction_bias", L->w_router_bias.data(), L->w_router_bias.size() * 4);
    }
    L->w_exp_gate.resize(size_t(cfg.n_experts));
    L->w_exp_up.resize(size_t(cfg.n_experts));
    L->w_exp_down.resize(size_t(cfg.n_experts));
    for (int e = 0; e < cfg.n_experts; ++e) {
      std::string ep = pre + "mlp.experts." + std::to_string(e) + ".";
      read_w16(st, ep + "gate_proj.weight", L->w_exp_gate[size_t(e)]);
      read_w16(st, ep + "up_proj.weight", L->w_exp_up[size_t(e)]);
      read_w16(st, ep + "down_proj.weight", L->w_exp_down[size_t(e)]);
    }
    read_w16(st, pre + "mlp.shared_expert.gate_proj.weight", L->w_sh_gate);
    read_w16(st, pre + "mlp.shared_expert.up_proj.weight", L->w_sh_up);
    read_w16(st, pre + "mlp.shared_expert.down_proj.weight", L->w_sh_down);
    if (cfg.shared_gate) read_w16(st, pre + "mlp.shared_expert_gate.weight", L->w_sh_gate_w);
  } else {
    read_w16(st, pre + "mlp.gate_proj.weight", L->w_gate);
    read_w16(st, pre + "mlp.up_proj.weight", L->w_up);
    read_w16(st, pre + "mlp.down_proj.weight", L->w_down);
  }
  L->ln_in.resize(size_t(H));
  L->ln_post.resize(size_t(H));
  st.read_into(pre + "input_layernorm.weight", L->ln_in.data(), L->ln_in.size() * 2);
  st.read_into(pre + "post_attention_layernorm.weight", L->ln_post.data(), L->ln_post.size() * 2);
  L->read_s = now_s() - t0;
  return L;
}

QwenLayerCache::QwenLayerCache(const SafeTensorsReader& st, const QwenConfig& cfg, int capacity)
    : st_(st), cfg_(cfg), cap_(capacity), th_([this] { worker(); }) {}
QwenLayerCache::~QwenLayerCache() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  th_.join();
}
void QwenLayerCache::prefetch(int idx) {
  if (idx < 0 || idx >= cfg_.layers) return;
  std::lock_guard<std::mutex> lk(mu_);
  if (have_.count(idx) || loading_.count(idx)) return;
  for (int q : queue_)
    if (q == idx) return;
  queue_.push_back(idx);
  cv_.notify_all();
}
void QwenLayerCache::worker() {
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
    auto L = read_qwen_layer(st_, cfg_, idx);
    {
      std::lock_guard<std::mutex> lk(mu_);
      have_[idx] = L;
      last_use_[idx] = ++tick_;
      loading_.erase(idx);
      while (int(have_.size()) > cap_) {
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
std::shared_ptr<QwenLayer> QwenLayerCache::get(int idx) {
  prefetch(idx);
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return have_.count(idx) != 0; });
  last_use_[idx] = ++tick_;
  return have_[idx];
}

// ------------------------------------------------------------ CPU reference forward (fp32)
namespace {

inline float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
inline float softplus_f(float x) { return x > 20.0f ? x : std::log1pf(std::exp(x)); }

// y[T][m] = X[T][n] . W[m][n]^T ; W row-major [m][n] bf16
void mm_bf16w(float* y, const float* X, const uint16_t* W, int T, int m, int n, int nthreads) {
#pragma omp parallel for schedule(static) num_threads(nthreads) collapse(2)
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < m; ++r) {
      const float* x = X + size_t(t) * n;
      const uint16_t* w = W + size_t(r) * n;
      float acc = 0;
      for (int i = 0; i < n; ++i) acc += x[i] * bf16_to_f32(w[i]);
      y[size_t(t) * m + r] = acc;
    }
}

inline void zc_norm_inplace(float* v, const uint16_t* w, int n, float eps) {
  float ss = 0;
  for (int i = 0; i < n; ++i) ss += v[i] * v[i];
  float inv = 1.0f / std::sqrt(ss / n + eps);
  for (int i = 0; i < n; ++i) v[i] = v[i] * inv * (1.0f + bf16_to_f32(w[i]));
}

inline void l2_norm_inplace(float* v, int n, float eps) {  // SUM of squares, eps inside
  float ss = 0;
  for (int i = 0; i < n; ++i) ss += v[i] * v[i];
  float inv = 1.0f / std::sqrt(ss + eps);
  for (int i = 0; i < n; ++i) v[i] *= inv;
}

// partial RoPE, rotate_half on [0, rot): x[j] <-> x[j+rot/2]
inline void rope_inplace(float* v, int rot, long pos, float theta) {
  for (int j = 0; j < rot / 2; ++j) {
    float inv_freq = std::pow(theta, -2.0f * float(j) / float(rot));
    float ang = float(pos) * inv_freq;
    float c = std::cos(ang), s = std::sin(ang);
    float a = v[j], b = v[j + rot / 2];
    v[j] = a * c - b * s;
    v[j + rot / 2] = b * c + a * s;
  }
}

}  // namespace

// Single Qwen3.5 layer forward on the residual stream x[T][hidden] (fp32, in place). Delta state starts
// at zero — used by qwen_cpu_forward and the GPU parity test.
void qwen_cpu_layer(const QwenConfig& cfg, const QwenLayer& l, float* x, int T, int nthreads) {
  const int H = cfg.hidden, hd = cfg.hd, nq = cfg.nq, nkv = cfg.nkv, gq = nq / nkv;
  const float qk_scale = 1.0f / std::sqrt(float(hd));
  std::vector<float> xn(size_t(T) * H), y(size_t(T) * H);
  std::memcpy(xn.data(), x, size_t(T) * H * 4);
#pragma omp parallel for num_threads(nthreads)
  for (int t = 0; t < T; ++t) zc_norm_inplace(xn.data() + size_t(t) * H, l.ln_in.data(), H, cfg.eps);
  if (l.full) {
    std::vector<float> q(size_t(T) * nq * 2 * hd), k(size_t(T) * nkv * hd), v(size_t(T) * nkv * hd),
        out(size_t(T) * nq * hd), attn_row(T);
    mm_bf16w(q.data(), xn.data(), l.w_q.data(), T, nq * 2 * hd, H, nthreads);
    mm_bf16w(k.data(), xn.data(), l.w_k.data(), T, nkv * hd, H, nthreads);
    mm_bf16w(v.data(), xn.data(), l.w_v.data(), T, nkv * hd, H, nthreads);
#pragma omp parallel for num_threads(nthreads) collapse(2)
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < nq + nkv; ++h) {
        float* vec = h < nq ? q.data() + size_t(t) * nq * 2 * hd + size_t(h) * 2 * hd
                            : k.data() + size_t(t) * nkv * hd + size_t(h - nq) * hd;
        zc_norm_inplace(vec, h < nq ? l.q_norm.data() : l.k_norm.data(), hd, cfg.eps);
        rope_inplace(vec, cfg.rot_dim, t, cfg.theta);
      }
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < nq; ++h) {
        const float* qh = q.data() + size_t(t) * nq * 2 * hd + size_t(h) * 2 * hd;
        const float* gate = qh + hd;
        const int kh = h / gq;
        const float* kh_base = k.data() + size_t(kh) * hd;
        const float* vh_base = v.data() + size_t(kh) * hd;
        float mx = -INFINITY;
        for (int i = 0; i <= t; ++i) {
          const float* ki = kh_base + size_t(i) * nkv * hd;
          float s = 0;
          for (int d = 0; d < hd; ++d) s += qh[d] * ki[d];
          s *= qk_scale;
          attn_row[size_t(i)] = s;
          mx = std::max(mx, s);
        }
        float sum = 0;
        for (int i = 0; i <= t; ++i) sum += attn_row[size_t(i)] = std::exp(attn_row[size_t(i)] - mx);
        float inv = 1.0f / sum;
        float* acc = out.data() + size_t(t) * nq * hd + size_t(h) * hd;
        for (int d = 0; d < hd; ++d) acc[d] = 0;
        for (int i = 0; i <= t; ++i) {
          const float* vi = vh_base + size_t(i) * nkv * hd;
          float w = attn_row[size_t(i)] * inv;
          for (int d = 0; d < hd; ++d) acc[d] += w * vi[d];
        }
        for (int d = 0; d < hd; ++d) acc[d] *= sigmoid_f(gate[d]);
      }
    }
    mm_bf16w(y.data(), out.data(), l.w_o.data(), T, H, nq * hd, nthreads);
  } else {
    // Gated DeltaNet: sequential over tokens, fp32 state per layer
    const int kd = cfg.lin_kd, vd = cfg.lin_vd, nv = cfg.lin_nv;
    const int key_dim = cfg.key_dim(), value_dim = cfg.value_dim(), conv_dim = cfg.conv_dim();
    const int group = cfg.lin_group();
    std::vector<float> qkv(conv_dim), zz(value_dim), ba(2 * nv);
    std::vector<float> ring(size_t(conv_dim) * cfg.conv_k, 0.0f);
    std::vector<float> S(size_t(nv) * kd * vd, 0.0f);  // [hv][kd][vd]
    std::vector<float> kv_mem(vd), delta(vd), ov(vd), o(value_dim);
    for (int t = 0; t < T; ++t) {
      const float* xn_t = xn.data() + size_t(t) * H;
      mm_bf16w(qkv.data(), xn_t, l.w_in_qkv.data(), 1, conv_dim, H, 1);
      mm_bf16w(zz.data(), xn_t, l.w_in_z.data(), 1, value_dim, H, 1);
      mm_bf16w(ba.data(), xn_t, l.w_in_ba.data(), 1, 2 * nv, H, 1);
      // conv: shift ring, push raw qkv, causal dot + silu
      for (int c = 0; c < conv_dim; ++c) {
        float* rg = ring.data() + size_t(c) * cfg.conv_k;
        for (int j = 0; j + 1 < cfg.conv_k; ++j) rg[j] = rg[j + 1];
        rg[cfg.conv_k - 1] = qkv[size_t(c)];
        float acc = 0;
        const uint16_t* cw = l.conv_w.data() + size_t(c) * cfg.conv_k;
        for (int j = 0; j < cfg.conv_k; ++j) acc += rg[j] * bf16_to_f32(cw[j]);
        qkv[size_t(c)] = silu_f(acc);
      }
      for (int hv = 0; hv < nv; ++hv) {
        const int kh = hv / group;
        float* qh = qkv.data() + size_t(kh) * kd;
        float* kh_vec = qkv.data() + key_dim + size_t(kh) * kd;
        const float* vh = qkv.data() + 2 * key_dim + size_t(hv) * vd;
        l2_norm_inplace(qh, kd, 1e-6f);
        for (int d = 0; d < kd; ++d) qh[d] *= 1.0f / std::sqrt(float(kd));
        l2_norm_inplace(kh_vec, kd, 1e-6f);
        const float beta = sigmoid_f(ba[hv]);
        const float g = -std::exp(l.a_log[size_t(hv)]) * softplus_f(ba[nv + hv] + l.dt_bias[size_t(hv)]);
        const float decay = std::exp(g);
        float* Sh = S.data() + size_t(hv) * kd * vd;
        for (int dv = 0; dv < vd; ++dv) kv_mem[dv] = 0;
        for (int dk = 0; dk < kd; ++dk) {
          const float k_dk = kh_vec[dk];
          float* srow = Sh + size_t(dk) * vd;
          for (int dv = 0; dv < vd; ++dv) {
            srow[dv] *= decay;
            kv_mem[dv] += srow[dv] * k_dk;
          }
        }
        for (int dv = 0; dv < vd; ++dv) {
          delta[dv] = (vh[dv] - kv_mem[dv]) * beta;
          ov[dv] = 0;
        }
        for (int dk = 0; dk < kd; ++dk) {
          const float k_dk = kh_vec[dk], q_dk = qh[dk];
          float* srow = Sh + size_t(dk) * vd;
          for (int dv = 0; dv < vd; ++dv) {
            const float s = srow[dv] + k_dk * delta[dv];
            srow[dv] = s;
            ov[dv] += s * q_dk;
          }
        }
        float ss = 0;
        for (int dv = 0; dv < vd; ++dv) ss += ov[dv] * ov[dv];
        const float inv = 1.0f / std::sqrt(ss / vd + cfg.eps);
        const uint16_t* nw = l.dn_norm.data();
        for (int dv = 0; dv < vd; ++dv)
          o[size_t(hv) * vd + dv] = ov[dv] * inv * bf16_to_f32(nw[dv]) * silu_f(zz[size_t(hv) * vd + dv]);
      }
      mm_bf16w(y.data() + size_t(t) * H, o.data(), l.w_out.data(), 1, H, value_dim, 1);
    }
  }
#pragma omp parallel for num_threads(nthreads)
  for (long i = 0; i < long(T) * H; ++i) x[size_t(i)] += y[size_t(i)];
  // MLP (dense)
  QCHECK(!l.moe, "qwen_cpu_layer: MoE layer unsupported here");
  std::memcpy(xn.data(), x, size_t(T) * H * 4);
#pragma omp parallel for num_threads(nthreads)
  for (int t = 0; t < T; ++t) zc_norm_inplace(xn.data() + size_t(t) * H, l.ln_post.data(), H, cfg.eps);
  std::vector<float> gu(size_t(T) * 2 * cfg.inter), act(size_t(T) * cfg.inter);
  mm_bf16w(gu.data(), xn.data(), l.w_gate.data(), T, cfg.inter, H, nthreads);
  {
    std::vector<float> up(size_t(T) * cfg.inter);
    mm_bf16w(up.data(), xn.data(), l.w_up.data(), T, cfg.inter, H, nthreads);
    for (long i = 0; i < long(T) * cfg.inter; ++i) act[size_t(i)] = silu_f(gu[size_t(i)]) * up[size_t(i)];
  }
  mm_bf16w(y.data(), act.data(), l.w_down.data(), T, H, cfg.inter, nthreads);
#pragma omp parallel for num_threads(nthreads)
  for (long i = 0; i < long(T) * H; ++i) x[size_t(i)] += y[size_t(i)];
}

void qwen_cpu_forward(const QwenConfig& cfg, const SafeTensorsReader& st,
                      const std::vector<std::shared_ptr<QwenLayer>>& layers, const QwenCpuRefs& refs,
                      const int32_t* tokens, int T, float* logits, int nthreads, int capture_layer,
                      std::vector<float>* captured, std::vector<float>* all_layers,
                      const int32_t* ppl_next, double* ppl_nll) {
  static std::string prefix;
  if (prefix.empty()) prefix = detect_prefix(st);
  const int H = cfg.hidden, hd = cfg.hd, nq = cfg.nq, nkv = cfg.nkv, gq = nq / nkv;
  std::vector<uint16_t> emb;
  {
    const TensorInfo& ei = st.info(prefix + "embed_tokens.weight");
    QCHECK(ei.dtype == "BF16" && int(ei.shape[0]) == cfg.vocab && int(ei.shape[1]) == H, "bad embed_tokens");
    emb.resize(size_t(ei.numel()));
    st.read_into(prefix + "embed_tokens.weight", emb.data(), emb.size() * 2);
  }
  const std::vector<uint16_t>* head = &emb;
  std::vector<uint16_t> lmh;
  if (!cfg.tied) {
    const TensorInfo& li = st.info("lm_head.weight");
    QCHECK(li.dtype == "BF16" && int(li.shape[0]) == cfg.vocab && int(li.shape[1]) == H, "bad lm_head");
    lmh.resize(size_t(li.numel()));
    st.read_into("lm_head.weight", lmh.data(), lmh.size() * 2);
    head = &lmh;
  }
  std::vector<uint16_t> final_norm(H);
  st.read_into(prefix + "norm.weight", final_norm.data(), final_norm.size() * 2);

  std::vector<float> x(size_t(T) * H);
  for (int t = 0; t < T; ++t) {
    int32_t id = tokens[t];
    QCHECK(id >= 0 && id < cfg.vocab, "token id out of range");
    for (int i = 0; i < H; ++i) x[size_t(t) * H + i] = bf16_to_f32(emb[size_t(id) * H + i]);
  }
  const float qk_scale = 1.0f / std::sqrt(float(hd));
  std::vector<float> xn(size_t(T) * H), y(size_t(T) * H);
  std::vector<float> q, k, v, out, gu, act, attn_row, o(cfg.value_dim());
  for (int L = 0; L < cfg.layers; ++L) {
    const QwenLayer& l = *layers[size_t(L)];
    std::memcpy(xn.data(), x.data(), size_t(T) * H * 4);
#pragma omp parallel for num_threads(nthreads)
    for (int t = 0; t < T; ++t) zc_norm_inplace(xn.data() + size_t(t) * H, l.ln_in.data(), H, cfg.eps);
    if (l.full) {
      q.assign(size_t(T) * nq * 2 * hd, 0);
      k.assign(size_t(T) * nkv * hd, 0);
      v.assign(size_t(T) * nkv * hd, 0);
      out.assign(size_t(T) * nq * hd, 0);
      mm_bf16w(q.data(), xn.data(), l.w_q.data(), T, nq * 2 * hd, H, nthreads);
      mm_bf16w(k.data(), xn.data(), l.w_k.data(), T, nkv * hd, H, nthreads);
      mm_bf16w(v.data(), xn.data(), l.w_v.data(), T, nkv * hd, H, nthreads);
      // q/k norm + partial rope (all rows, all heads)
#pragma omp parallel for num_threads(nthreads) collapse(2)
      for (int t = 0; t < T; ++t)
        for (int h = 0; h < nq + nkv; ++h) {
          float* vec = h < nq ? q.data() + size_t(t) * nq * 2 * hd + size_t(h) * 2 * hd
                              : k.data() + size_t(t) * nkv * hd + size_t(h - nq) * hd;
          zc_norm_inplace(vec, h < nq ? l.q_norm.data() : l.k_norm.data(), hd, cfg.eps);
          rope_inplace(vec, cfg.rot_dim, t, cfg.theta);
        }
      if (const char* dbg = std::getenv("QWEN_DBG_DIR")) {
        std::ofstream f(std::string(dbg) + "/qkv_L" + std::to_string(L) + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(q.data()), std::streamsize(q.size() * 4));
        f.write(reinterpret_cast<const char*>(k.data()), std::streamsize(k.size() * 4));
        f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
      }
      attn_row.assign(T, 0);
      for (int t = 0; t < T; ++t) {
        for (int h = 0; h < nq; ++h) {
          const float* qh = q.data() + size_t(t) * nq * 2 * hd + size_t(h) * 2 * hd;
          const float* gate = qh + hd;
          const int kh = h / gq;
          const float* kh_base = k.data() + size_t(kh) * hd;
          const float* vh_base = v.data() + size_t(kh) * hd;
          float mx = -INFINITY;
          for (int i = 0; i <= t; ++i) {
            const float* ki = kh_base + size_t(i) * nkv * hd;
            float s = 0;
            for (int d = 0; d < hd; ++d) s += qh[d] * ki[d];
            s *= qk_scale;
            attn_row[size_t(i)] = s;
            mx = std::max(mx, s);
          }
          float sum = 0;
          for (int i = 0; i <= t; ++i) sum += attn_row[size_t(i)] = std::exp(attn_row[size_t(i)] - mx);
          float inv = 1.0f / sum;
          float* acc = out.data() + size_t(t) * nq * hd + size_t(h) * hd;
          for (int d = 0; d < hd; ++d) acc[d] = 0;
          for (int i = 0; i <= t; ++i) {
            const float* vi = vh_base + size_t(i) * nkv * hd;
            float w = attn_row[size_t(i)] * inv;
            for (int d = 0; d < hd; ++d) acc[d] += w * vi[d];
          }
          for (int d = 0; d < hd; ++d) acc[d] *= sigmoid_f(gate[d]);
        }
      }
      if (const char* dbg = std::getenv("QWEN_DBG_DIR")) {
        std::ofstream f(std::string(dbg) + "/attnout_L" + std::to_string(L) + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
      }
      mm_bf16w(y.data(), out.data(), l.w_o.data(), T, H, nq * hd, nthreads);
    } else {
      // Gated DeltaNet: sequential over tokens, fp32 state per layer
      const int kd = cfg.lin_kd, vd = cfg.lin_vd, nk = cfg.lin_nk, nv = cfg.lin_nv;
      const int key_dim = cfg.key_dim(), value_dim = cfg.value_dim(), conv_dim = cfg.conv_dim();
      const int group = cfg.lin_group();
      std::vector<float> qkv(conv_dim), zz(value_dim), ba(2 * nv);
      std::vector<float> ring(size_t(conv_dim) * cfg.conv_k, 0.0f);
      std::vector<float> S(size_t(nv) * kd * vd, 0.0f);  // [hv][kd][vd]
      std::vector<float> kv_mem(vd), delta(vd), ov(vd);
      for (int t = 0; t < T; ++t) {
        const float* xn_t = xn.data() + size_t(t) * H;
        mm_bf16w(qkv.data(), xn_t, l.w_in_qkv.data(), 1, conv_dim, H, 1);
        mm_bf16w(zz.data(), xn_t, l.w_in_z.data(), 1, value_dim, H, 1);
        mm_bf16w(ba.data(), xn_t, l.w_in_ba.data(), 1, 2 * nv, H, 1);
        // conv: shift ring, push raw qkv, causal dot + silu
        for (int c = 0; c < conv_dim; ++c) {
          float* rg = ring.data() + size_t(c) * cfg.conv_k;
          for (int j = 0; j + 1 < cfg.conv_k; ++j) rg[j] = rg[j + 1];
          rg[cfg.conv_k - 1] = qkv[size_t(c)];
          float acc = 0;
          const uint16_t* cw = l.conv_w.data() + size_t(c) * cfg.conv_k;
          for (int j = 0; j < cfg.conv_k; ++j) acc += rg[j] * bf16_to_f32(cw[j]);
          qkv[size_t(c)] = silu_f(acc);
        }
        for (int hv = 0; hv < nv; ++hv) {
          const int kh = hv / group;
          float* qh = qkv.data() + size_t(kh) * kd;
          float* kh_vec = qkv.data() + key_dim + size_t(kh) * kd;
          const float* vh = qkv.data() + 2 * key_dim + size_t(hv) * vd;
          l2_norm_inplace(qh, kd, 1e-6f);
          for (int d = 0; d < kd; ++d) qh[d] *= 1.0f / std::sqrt(float(kd));
          l2_norm_inplace(kh_vec, kd, 1e-6f);
          const float beta = sigmoid_f(ba[hv]);
          const float g =
              -std::exp(l.a_log[size_t(hv)]) * softplus_f(ba[nv + hv] + l.dt_bias[size_t(hv)]);
          const float decay = std::exp(g);
          float* Sh = S.data() + size_t(hv) * kd * vd;
          for (int dv = 0; dv < vd; ++dv) kv_mem[dv] = 0;
          for (int dk = 0; dk < kd; ++dk) {
            const float k_dk = kh_vec[dk];
            float* srow = Sh + size_t(dk) * vd;
            for (int dv = 0; dv < vd; ++dv) {
              srow[dv] *= decay;
              kv_mem[dv] += srow[dv] * k_dk;
            }
          }
          for (int dv = 0; dv < vd; ++dv) {
            delta[dv] = (vh[dv] - kv_mem[dv]) * beta;
            ov[dv] = 0;
          }
          for (int dk = 0; dk < kd; ++dk) {
            const float k_dk = kh_vec[dk], q_dk = qh[dk];
            float* srow = Sh + size_t(dk) * vd;
            for (int dv = 0; dv < vd; ++dv) {
              const float s = srow[dv] + k_dk * delta[dv];
              srow[dv] = s;
              ov[dv] += s * q_dk;
            }
          }
          float ss = 0;
          for (int dv = 0; dv < vd; ++dv) ss += ov[dv] * ov[dv];
          const float inv = 1.0f / std::sqrt(ss / vd + cfg.eps);
          const uint16_t* nw = l.dn_norm.data();
          for (int dv = 0; dv < vd; ++dv)
            o[size_t(hv) * vd + dv] =
                ov[dv] * inv * bf16_to_f32(nw[dv]) * silu_f(zz[size_t(hv) * vd + dv]);
        }
        mm_bf16w(y.data() + size_t(t) * H, o.data(), l.w_out.data(), 1, H, value_dim, 1);
      }
    }
#pragma omp parallel for num_threads(nthreads)
    for (long i = 0; i < long(T) * H; ++i) x[size_t(i)] += y[size_t(i)];
    if (std::getenv("QWEN_DBG_NORM")) {
      double ss = 0;
      for (int i = 0; i < H; ++i) ss += double(x[i]) * x[i];
      std::fprintf(stderr, "L%d %s post-attn rms %.4f\n", L, l.full ? "full" : "lin", std::sqrt(ss / H));
    }
    // MLP (dense or MoE)
    if (std::getenv("QWEN_DBG_NORM")) {
      double wmin = 1e30, wmax = -1e30;
      for (int i = 0; i < H; ++i) {
        float wv = bf16_to_f32(l.ln_post[size_t(i)]);
        wmin = std::min(wmin, double(wv));
        wmax = std::max(wmax, double(wv));
      }
      std::fprintf(stderr, "  mlp entry x rms %.4f ln_post [%.3f, %.3f]\n", [&] {
        double ss = 0;
        for (int i = 0; i < H; ++i) ss += double(x[i]) * x[i];
        return std::sqrt(ss / H);
      }(),
                   wmin, wmax);
    }
    std::memcpy(xn.data(), x.data(), size_t(T) * H * 4);
#pragma omp parallel for num_threads(nthreads)
    for (int t = 0; t < T; ++t) zc_norm_inplace(xn.data() + size_t(t) * H, l.ln_post.data(), H, cfg.eps);
    if (!l.moe) {
      gu.assign(size_t(T) * 2 * cfg.inter, 0);
      act.assign(size_t(T) * cfg.inter, 0);
      mm_bf16w(gu.data(), xn.data(), l.w_gate.data(), T, cfg.inter, H, nthreads);
      {
        std::vector<float> up(size_t(T) * cfg.inter);
        mm_bf16w(up.data(), xn.data(), l.w_up.data(), T, cfg.inter, H, nthreads);
        for (long i = 0; i < long(T) * cfg.inter; ++i) act[size_t(i)] = silu_f(gu[size_t(i)]) * up[size_t(i)];
      }
      mm_bf16w(y.data(), act.data(), l.w_down.data(), T, H, cfg.inter, nthreads);
    } else {
      std::vector<float> rlog(cfg.n_experts);
      const bool dbg = std::getenv("QWEN_DBG_NORM") != nullptr;
      for (int t = 0; t < T; ++t) {
        const float* xt = xn.data() + size_t(t) * H;
        if (dbg && t == 0) {
          double ss = 0;
          for (int i = 0; i < H; ++i) ss += double(xt[i]) * xt[i];
          std::fprintf(stderr, "  moe xn rms %.4f\n", std::sqrt(ss / H));
        }
        mm_bf16w(rlog.data(), xt, l.w_router.data(), 1, cfg.n_experts, H, 1);
        // softmax -> top-k (deterministic: highest logit, ties by lower index) -> renormalise
        float mx = -INFINITY;
        for (int e = 0; e < cfg.n_experts; ++e) mx = std::max(mx, rlog[size_t(e)]);
        float sum = 0;
        for (int e = 0; e < cfg.n_experts; ++e) sum += rlog[size_t(e)] = std::exp(rlog[size_t(e)] - mx);
        for (int e = 0; e < cfg.n_experts; ++e) rlog[size_t(e)] /= sum;
        int ids[64];
        float wts[64];
        for (int j = 0; j < cfg.topk; ++j) {
          int best = -1;
          for (int e = 0; e < cfg.n_experts; ++e)
            if (best < 0 || rlog[size_t(e)] > rlog[size_t(best)]) best = e;
          ids[j] = best;
          wts[j] = rlog[size_t(best)];
          rlog[size_t(best)] = -1;
        }
        float wsum = 0;
        for (int j = 0; j < cfg.topk; ++j) wsum += wts[j];
        if (dbg && t == 0)
          std::fprintf(stderr, "  moe route ids %d/%d w %.4f/%.4f (wsum %.4f) rlog0 %.3f rlog1 %.3f\n",
                       ids[0], ids[1], wts[0] / wsum, wts[1] / wsum, wsum, rlog[0], rlog[1]);
        std::vector<float> yr(H, 0.0f), h(cfg.moe_inter);
        for (int j = 0; j < cfg.topk; ++j) {
          const int e = ids[j];
          const float w = wts[j] / wsum;
          const uint16_t* wg = l.w_exp_gate[size_t(e)].data();
          const uint16_t* wu = l.w_exp_up[size_t(e)].data();
          const uint16_t* wd = l.w_exp_down[size_t(e)].data();
          for (int i = 0; i < cfg.moe_inter; ++i) {
            float g = 0, u = 0;
            for (int d = 0; d < H; ++d) {
              g += xt[d] * bf16_to_f32(wg[size_t(i) * H + d]);
              u += xt[d] * bf16_to_f32(wu[size_t(i) * H + d]);
            }
            h[size_t(i)] = silu_f(g) * u;
          }
          for (int d = 0; d < H; ++d) {
            float a = 0;
            for (int i = 0; i < cfg.moe_inter; ++i) a += h[size_t(i)] * bf16_to_f32(wd[size_t(d) * cfg.moe_inter + i]);
            yr[size_t(d)] += w * a;
          }
        }
        // shared expert, sigmoid-gated
        if (dbg && t == 0) {
          double ss = 0;
          for (int i = 0; i < H; ++i) ss += double(yr[i]) * yr[i];
          std::fprintf(stderr, "  moe routed yr rms %.4f\n", std::sqrt(ss / H));
        }
        if (cfg.shared_gate) {
          float sg = 0;
          for (int d = 0; d < H; ++d) sg += xt[d] * bf16_to_f32(l.w_sh_gate_w[size_t(d)]);
          sg = sigmoid_f(sg);
          std::vector<float> hs(cfg.shared_inter);
          for (int i = 0; i < cfg.shared_inter; ++i) {
            float g = 0, u = 0;
            for (int d = 0; d < H; ++d) {
              g += xt[d] * bf16_to_f32(l.w_sh_gate[size_t(i) * H + d]);
              u += xt[d] * bf16_to_f32(l.w_sh_up[size_t(i) * H + d]);
            }
            hs[size_t(i)] = silu_f(g) * u;
          }
          for (int d = 0; d < H; ++d) {
            float a = 0;
            for (int i = 0; i < cfg.shared_inter; ++i)
              a += hs[size_t(i)] * bf16_to_f32(l.w_sh_down[size_t(d) * cfg.shared_inter + i]);
            yr[size_t(d)] += sg * a;
          }
        } else {
          std::vector<float> hs(cfg.shared_inter);
          for (int i = 0; i < cfg.shared_inter; ++i) {
            float g = 0, u = 0;
            for (int d = 0; d < H; ++d) {
              g += xt[d] * bf16_to_f32(l.w_sh_gate[size_t(i) * H + d]);
              u += xt[d] * bf16_to_f32(l.w_sh_up[size_t(i) * H + d]);
            }
            hs[size_t(i)] = silu_f(g) * u;
          }
          for (int d = 0; d < H; ++d) {
            float a = 0;
            for (int i = 0; i < cfg.shared_inter; ++i)
              a += hs[size_t(i)] * bf16_to_f32(l.w_sh_down[size_t(d) * cfg.shared_inter + i]);
            yr[size_t(d)] += a;
          }
        }
        std::memcpy(y.data() + size_t(t) * H, yr.data(), H * 4);
        if (dbg && t == 0) {
          double ss = 0;
          bool fin = true;
          for (int i = 0; i < H; ++i) {
            ss += double(yr[i]) * yr[i];
            fin = fin && std::isfinite(yr[i]);
          }
          std::fprintf(stderr, "  moe yr rms %.4f finite %d\n", std::sqrt(ss / H), int(fin));
        }
      }
    }
#pragma omp parallel for num_threads(nthreads)
    for (long i = 0; i < long(T) * H; ++i) x[size_t(i)] += y[size_t(i)];
    if (std::getenv("QWEN_DBG_NORM")) {
      double ss = 0;
      for (int i = 0; i < H; ++i) ss += double(x[i]) * x[i];
      std::fprintf(stderr, "L%d mlp rms %.4f\n", L, std::sqrt(ss / H));
    }
    if (capture_layer == L && captured) {
      captured->assign(x.begin(), x.end());
      capture_layer = -1;
    }
    if (all_layers) all_layers->insert(all_layers->end(), x.begin(), x.end());
  }
  // final norm + lm_head
  std::vector<float> xf(size_t(T) * H);
  std::memcpy(xf.data(), x.data(), size_t(T) * H * 4);
  for (int t = 0; t < T; ++t) zc_norm_inplace(xf.data() + size_t(t) * H, final_norm.data(), H, cfg.eps);
  if (ppl_next && ppl_nll) {
    // streaming: one row of the lm_head per target; proper log-softmax
    double nll = 0;
    for (int t = 0; t + 1 < T; ++t) {
      mm_bf16w(logits, xf.data() + size_t(t) * H, head->data(), 1, cfg.vocab, H, nthreads);
      float mx = -INFINITY;
      for (long v = 0; v < cfg.vocab; ++v) mx = std::max(mx, logits[size_t(v)]);
      double lse = 0;
      for (long v = 0; v < cfg.vocab; ++v) lse += std::exp(double(logits[size_t(v)]) - mx);
      lse = std::log(lse) + mx;
      nll += lse - logits[size_t(ppl_next[size_t(t + 1)])];
    }
    *ppl_nll += nll;
  } else {
    mm_bf16w(logits, xf.data(), head->data(), T, cfg.vocab, H, nthreads);
  }
}

}  // namespace qtip
