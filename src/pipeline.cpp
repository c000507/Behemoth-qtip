// Pipeline implementation, extracted from main.cpp; see pipeline.hpp. The adaptive mode adds per-layer
// QTIP-vs-NVFP4 selection (see docs/ADAPTIVE.md for the metric definitions).
#include "pipeline.hpp"
#include "nvfp4_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <tuple>

namespace qtip {
namespace fs = std::filesystem;

// ------------------------------------------------------------------ Hessian file I/O
// safetensors with F32 tensors qkv, o, gateup, down, metadata "count"
static void write_hess(const std::string& path, const std::vector<std::vector<float>>& H, const int dims[4],
                       long count) {
  SafeTensorsWriter w;
  for (int i = 0; i < 4; ++i)
    w.add(kHessName[i], "F32", {dims[i], dims[i]}, H[size_t(i)].data(), H[size_t(i)].size() * 4);
  w.write(path, {{"format", "pt"}, {"count", std::to_string(count)}, {"normalised", "1"}});
}

static std::vector<std::vector<float>> read_hess(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  QCHECK(f, "cannot open " + path);
  uint64_t hl = 0;
  f.read(reinterpret_cast<char*>(&hl), 8);
  std::string hs(hl, '\0');
  f.read(hs.data(), std::streamsize(hl));
  json h = json::parse(hs);
  std::vector<std::vector<float>> H(4);
  for (int i = 0; i < 4; ++i) {
    auto& t = h[kHessName[i]];
    auto off = t["data_offsets"].get<std::vector<uint64_t>>();
    H[size_t(i)].resize((off[1] - off[0]) / 4);
    f.seekg(std::streamoff(8 + hl + off[0]));
    f.read(reinterpret_cast<char*>(H[size_t(i)].data()), std::streamsize(off[1] - off[0]));
    QCHECK(f.good(), "short read " + path);
  }
  return H;
}

// --- MoE Hessian file: qkv/o/gateup/down (pooled, shared expert) + per-expert gateup/down + stats.
// Layout of the returned vector: [0,4) as dense, [4+2e] expert e gateup, [5+2e] expert e down.
// stats (out-param): [0,E) sum of routing-weight^2, [E,2E) routed token counts (both still summed).
static std::vector<std::vector<float>> read_hess_moe(const std::string& path, std::vector<float>& stats,
                                                     int& experts) {
  std::ifstream f(path, std::ios::binary);
  QCHECK(f, "cannot open " + path);
  uint64_t hl = 0;
  f.read(reinterpret_cast<char*>(&hl), 8);
  std::string hs(hl, '\0');
  f.read(hs.data(), std::streamsize(hl));
  json h = json::parse(hs);
  QCHECK(h.contains("__metadata__") && h["__metadata__"].value("format", "") == "moe",
         "not a MoE Hessian file: " + path);
  experts = std::stoi(h["__metadata__"].at("experts").get<std::string>());
  auto read_t = [&](const std::string& name) {
    auto& t = h.at(name);
    auto off = t["data_offsets"].get<std::vector<uint64_t>>();
    std::vector<float> v((off[1] - off[0]) / 4);
    f.seekg(std::streamoff(8 + hl + off[0]));
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(off[1] - off[0]));
    QCHECK(f.good(), "short read " + path + " tensor " + name);
    return v;
  };
  std::vector<std::vector<float>> H(4 + size_t(2 * experts));
  for (int i = 0; i < 4; ++i) H[size_t(i)] = read_t(kHessName[i]);
  stats = read_t("stats");
  for (int e = 0; e < experts; ++e) {
    H[size_t(4 + 2 * e)] = read_t("e" + std::to_string(e) + ".gateup");
    H[size_t(5 + 2 * e)] = read_t("e" + std::to_string(e) + ".down");
  }
  return H;
}

static void write_hess_moe(const std::string& path, const std::vector<std::vector<float>>& H,
                           const std::vector<float>& stats, const int dims[4], int moe_inter, int experts,
                           long count) {
  SafeTensorsWriter w;
  for (int i = 0; i < 4; ++i)
    w.add(kHessName[i], "F32", {dims[i], dims[i]}, H[size_t(i)].data(), H[size_t(i)].size() * 4);
  w.add("stats", "F32", {long(stats.size())}, stats.data(), stats.size() * 4);
  for (int e = 0; e < experts; ++e) {
    w.add("e" + std::to_string(e) + ".gateup", "F32", {dims[2], dims[2]}, H[size_t(4 + 2 * e)].data(),
          H[size_t(4 + 2 * e)].size() * 4);
    w.add("e" + std::to_string(e) + ".down", "F32", {moe_inter, moe_inter}, H[size_t(5 + 2 * e)].data(),
          H[size_t(5 + 2 * e)].size() * 4);
  }
  w.write(path, {{"format", "moe"},
                 {"count", std::to_string(count)},
                 {"normalised", "1"},
                 {"experts", std::to_string(experts)}});
}

// Qwen3.5 checkpoints have no Llama-style config: the qwen path uses qcfg_ instead of cfg_.
static ModelConfig load_model_config_qtip(const std::string& dir, const SafeTensorsReader* st) {
  if (is_qwen35_dir(dir)) return ModelConfig{};
  return load_model_config(dir, st);
}
static QwenConfig load_qwen_config_or_empty(const std::string& dir) {
  if (!is_qwen35_dir(dir)) return QwenConfig{};  // raw_json stays empty
  return load_qwen_config(dir);
}

Pipeline::Pipeline(const Options& o)
    : o_(o),
      st_(o.model),
      cfg_(load_model_config_qtip(o.model, &st_)),
      cache_(st_, cfg_, 4),
      qcfg_(load_qwen_config_or_empty(o.model)),
      qwen_(!qcfg_.raw_json.empty()),
      qcache_(qwen_ ? std::make_unique<QwenLayerCache>(st_, qcfg_, 4) : nullptr) {
  if (o_.layer_end < 0) o_.layer_end = layers_total();
  if (qwen_) {
    // BF16 source; every attention/DeltaNet/MLP linear (incl. K/V) is QTIP-quantized
    QCHECK(o_.mode != "adaptive", "--mode adaptive is not supported for Qwen3.5 yet");
    QCHECK(!(o_.kv_explicit && o_.fp8_kv), "--kv-format fp8 is not supported for Qwen3.5");
    o_.fp8_kv = false;
  }
  if (o_.hess_dir.empty()) o_.hess_dir = o_.work + "/hess";
  // BF16 sources (e.g. Llama 3.x) default to all-QTIP; an explicit --kv-format fp8 converts K/V to FP8-Dynamic
  // (per-output-channel bf16 scale) instead of quantizing them.
  if (cfg_.bf16_source) {
    if (!o_.kv_explicit) o_.fp8_kv = false;
    cfg_.convert_kv_fp8 = o_.fp8_kv;
  }
  adaptive_ = o_.mode == "adaptive";
  QCHECK(!adaptive_ || (o_.kl_metric == "lmkl" || o_.kl_metric == "mse"), "--kl-metric must be lmkl or mse");
  QCHECK(!adaptive_ || o_.kl_threshold >= 0, "--kl-threshold must be >= 0");
  QCHECK(!adaptive_ || (o_.adaptive_next_input == "original" || o_.adaptive_next_input == "quantized"),
         "--adaptive-next-input must be original or quantized");
  QCHECK(!adaptive_ || o_.kl_tokens > 0, "--kl-tokens must be positive");
  fs::create_directories(o_.out);
  fs::create_directories(o_.work);
  fs::create_directories(o_.hess_dir);
  if (!o_.q.dump_dir.empty()) fs::create_directories(o_.q.dump_dir);
  log_set_file(o_.work + "/qtip_quant.log");
  load_progress();
}

void Pipeline::load_progress() {
  std::string p = o_.work + "/progress.json";
  if (fs::exists(p)) {
    std::ifstream f(p);
    progress_ = json::parse(f);
  } else {
    progress_ = json::object();
    progress_["layers"] = json::object();
    progress_["hessians"] = json::array();
  }
  QCHECK(progress_.value("kv_format", std::string("qtip")) == (o_.fp8_kv ? "fp8" : "qtip") ||
              progress_["layers"].empty(), "work directory has a different K/V format; set --kv-format to match or use a new --work directory");
  progress_["kv_format"] = o_.fp8_kv ? "fp8" : "qtip";
  for (auto& kv : progress_["layers"].items()) {
    if (kv.value().value("status", "") == "done" && fs::exists(layer_path(std::stoi(kv.key())))) {
      done_.insert(std::stoi(kv.key()));
      if (kv.value().contains("format")) layer_format_[std::stoi(kv.key())] = kv.value()["format"];
    }
  }
  for (auto& v : progress_["hessians"]) {
    int L = v.get<int>();
    if (fs::exists(hess_path(L))) hess_on_disk_.insert(L);
  }
  // quantize mode: accept any Hessian file present in hess_dir
  if (o_.mode == "quantize")
    for (int L = o_.layer_begin; L < o_.layer_end; ++L)
      if (fs::exists(hess_path(L))) hess_on_disk_.insert(L);
  if (adaptive_ && o_.adaptive_next_input == "quantized" && !done_.empty())
    die("resuming --mode adaptive with --adaptive-next-input quantized is not supported: the chain state of "
        "already-quantized layers cannot be reconstructed; start from a fresh --work/--out or use the default "
        "original-input chaining");
}

void Pipeline::save_progress_locked() {
  json h = json::array();
  for (int L : hess_on_disk_) h.push_back(L);
  progress_["hessians"] = h;
  std::string p = o_.work + "/progress.json", tmp = p + ".tmp";
  {
    std::ofstream f(tmp);
    f << progress_.dump(1);
  }
  fs::rename(tmp, p);
}

std::string Pipeline::hess_path(int L) const { return o_.hess_dir + "/layer" + std::to_string(L) + ".safetensors"; }
std::string Pipeline::layer_path(int L) const {
  char b[64];
  std::snprintf(b, sizeof b, "/model-layer-%05d.safetensors", L);
  return o_.out + b;
}

void Pipeline::deposit_partial(int L, HessAcc& acc, LayerRunner& R) {
  R.symmetrize_hess(acc);
  // download partial sums
  std::vector<std::vector<float>> part(4);
  for (int i = 0; i < 4; ++i) {
    part[size_t(i)].resize(size_t(acc.dim[i]) * acc.dim[i]);
    d2h_staged(part[size_t(i)].data(), acc.h[i], part[size_t(i)].size() * 4, R.stream());
  }
  long cnt = acc.count;
  bool complete = false;
  std::vector<std::vector<float>> mine;
  long total = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    LayerState& S = layers_[L];
    if (S.partial.empty()) {
      S.partial = std::move(part);
      S.partial_count = cnt;
    } else {
      for (int i = 0; i < 4; ++i) {
        float* a = S.partial[size_t(i)].data();
        const float* b = part[size_t(i)].data();
        long n = long(S.partial[size_t(i)].size());
#pragma omp parallel for schedule(static) num_threads(16)
        for (long k = 0; k < n; ++k) a[k] += b[k];
      }
      S.partial_count += cnt;
    }
    S.partial_parts++;
    S.st = LayerState::kPartial;
    if (S.partial_parts == int(o_.gpus.size())) {
      complete = true;
      mine = std::move(S.partial);
      total = S.partial_count;
      S.partial.clear();
    }
  }
  if (!complete) return;
  double inv = 1.0 / double(total);
  for (auto& v : mine) {
    long n = long(v.size());
    float* a = v.data();
#pragma omp parallel for schedule(static) num_threads(16)
    for (long k = 0; k < n; ++k) a[k] = float(a[k] * inv);
  }
  double t0 = now_s();
  write_hess(hess_path(L), mine, acc.dim, total);
  logf("layer %d: Hessians complete (%ld tokens), saved in %.1fs", L, total, now_s() - t0);
  {
    std::lock_guard<std::mutex> lk(mu_);
    LayerState& S = layers_[L];
    if (o_.mode == "collect") {
      S.st = LayerState::kDone;
    } else {
      S.H = std::move(mine);
      S.st = LayerState::kReady;
    }
    hess_on_disk_.insert(L);
    save_progress_locked();
  }
  cv_.notify_all();
}

void Pipeline::chain_step(int wi, LayerRunner& R, int L, bool collect) {
  cache_.prefetch(L + 1);
  double t0 = now_s();
  auto LH = cache_.get(L);
  double t_read = now_s() - t0;
  t0 = now_s();
  R.upload(*LH);
  double t_up = now_s() - t0;
  HessAcc* acc = nullptr;
  static thread_local std::unique_ptr<HessAcc> tacc;
  if (collect) {
    if (!tacc) {
      tacc = std::make_unique<HessAcc>();
      R.alloc_hess(*tacc);
    } else {
      R.zero_hess(*tacc);
    }
    acc = tacc.get();
  }
  R.gemm_flops = 0;
  t0 = now_s();
  R.forward(act_[size_t(wi)].get(), act_nseq_[size_t(wi)], acc);
  double t_fwd = now_s() - t0;
  long ntok = act_nseq_[size_t(wi)] * o_.seq_len;
  logf("gpu%d layer %d %s: %ld tokens in %.1fs (%.0f tok/s, %.1f TFLOPS), weights read %.1fs upload %.1fs", R.device(),
       L, collect ? "collect" : "forward", ntok, t_fwd, ntok / t_fwd, R.gemm_flops / t_fwd / 1e12, t_read, t_up);
  if (!o_.dump_acts.empty() && wi == 0) {
    fs::create_directories(o_.dump_acts);
    std::ofstream f(o_.dump_acts + "/act_after_layer" + std::to_string(L) + ".bf16", std::ios::binary);
    f.write(reinterpret_cast<const char*>(act_[0].get()), std::streamsize(size_t(o_.seq_len) * cfg_.hidden * 2));
  }
  if (collect) deposit_partial(L, *acc, R);
}

void Pipeline::chain_step_qwen(int wi, QwenLayerRunner& R, int L, bool collect) {
  qcache_->prefetch(L + 1);
  double t0 = now_s();
  auto QL = qcache_->get(L);
  double t_read = now_s() - t0;
  t0 = now_s();
  R.upload(*QL);
  double t_up = now_s() - t0;
  HessAcc* acc = nullptr;
  MoeHessAcc* moe = nullptr;
  static thread_local std::unique_ptr<HessAcc> tacc;
  static thread_local std::unique_ptr<MoeHessAcc> tmoe;
  if (collect) {
    const bool moe_layer = QL->moe;
    const int dim1 = qcfg_.is_full(L) ? qcfg_.nq * qcfg_.hd : qcfg_.value_dim();
    const int dim3 = moe_layer ? qcfg_.shared_inter : qcfg_.inter;
    if (!tacc || tacc->dim[1] != dim1 || tacc->dim[3] != dim3 || (moe_layer && !tmoe) ||
        (!moe_layer && tmoe)) {
      tacc = std::make_unique<HessAcc>();
      R.alloc_hess(*tacc, qcfg_.is_full(L), moe_layer);
      tmoe.reset();
      if (moe_layer) {
        tmoe = std::make_unique<MoeHessAcc>();
        R.alloc_moe_hess(*tmoe);
      }
    } else {
      R.zero_hess(*tacc);
      if (tmoe) R.zero_moe_hess(*tmoe);
    }
    acc = tacc.get();
    moe = tmoe.get();
  }
  QCHECK(!QL->moe || moe, "MoE layer without a MoE Hessian accumulator");
  R.gemm_flops = 0;
  t0 = now_s();
  R.forward(act_[size_t(wi)].get(), act_nseq_[size_t(wi)], acc, moe);
  double t_fwd = now_s() - t0;
  long ntok = act_nseq_[size_t(wi)] * o_.seq_len;
  logf("gpu%d qwen layer %d %s: %ld tokens in %.1fs (%.0f tok/s, %.1f TFLOPS), weights read %.1fs upload %.1fs",
       R.device(), L, collect ? "collect" : "forward", ntok, t_fwd, ntok / t_fwd, R.gemm_flops / t_fwd / 1e12,
       t_read, t_up);
  if (!o_.dump_acts.empty() && wi == 0) {
    fs::create_directories(o_.dump_acts);
    std::ofstream f(o_.dump_acts + "/act_after_layer" + std::to_string(L) + ".bf16", std::ios::binary);
    f.write(reinterpret_cast<const char*>(act_[0].get()),
            std::streamsize(size_t(o_.seq_len) * qcfg_.hidden * 2));
  }
  if (collect) deposit_partial_qwen(L, *acc, R, moe);
}

void Pipeline::deposit_partial_qwen(int L, HessAcc& acc, QwenLayerRunner& R, MoeHessAcc* moe) {
  R.symmetrize_hess(acc);
  const int E = moe ? qcfg_.n_experts : 0;
  if (moe) R.symmetrize_moe_hess(*moe);
  std::vector<std::vector<float>> part(4 + size_t(2 * E));
  for (int i = 0; i < 4; ++i) {
    part[size_t(i)].resize(size_t(acc.dim[i]) * acc.dim[i]);
    d2h_staged(part[size_t(i)].data(), acc.h[i], part[size_t(i)].size() * 4, R.stream());
  }
  std::vector<float> stats;
  if (moe) {
    for (int e = 0; e < E; ++e) {
      auto& g = part[size_t(4 + 2 * e)];
      g.resize(size_t(qcfg_.hidden) * qcfg_.hidden);
      d2h_staged(g.data(), moe->gate[size_t(e)], g.size() * 4, R.stream());
      auto& d = part[size_t(5 + 2 * e)];
      d.resize(size_t(qcfg_.moe_inter) * qcfg_.moe_inter);
      d2h_staged(d.data(), moe->down[size_t(e)], d.size() * 4, R.stream());
    }
    stats.resize(size_t(2 * E));
    d2h_staged(stats.data(), moe->wsum, size_t(E) * 4, R.stream());
    HIP_CHECK(hipStreamSynchronize(R.stream()));
    std::vector<int> cnt_h{size_t(E)};  // tiny copy after the sync: plain pageable memcpy is safe
    HIP_CHECK(hipMemcpy(cnt_h.data(), moe->cnt, size_t(E) * 4, hipMemcpyDeviceToHost));
    for (int e = 0; e < E; ++e) stats[size_t(E + e)] = float(cnt_h[size_t(e)]);
  }
  long cnt = acc.count;
  bool complete = false;
  std::vector<std::vector<float>> mine;
  std::vector<float> my_stats;
  long total = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    LayerState& S = layers_[L];
    if (S.partial.empty()) {
      S.partial = std::move(part);
      S.moe_stats_partial = std::move(stats);
      S.partial_count = cnt;
    } else {
      QCHECK(S.partial.size() == part.size(), "partial Hessian count mismatch (dense vs MoE?)");
      for (size_t i = 0; i < S.partial.size(); ++i) {
        float* a = S.partial[i].data();
        const float* b = part[i].data();
        long n = long(S.partial[i].size());
        QCHECK(n == long(part[i].size()), "partial Hessian size mismatch");
#pragma omp parallel for schedule(static) num_threads(16)
        for (long k = 0; k < n; ++k) a[k] += b[k];
      }
      for (size_t i = 0; i < S.moe_stats_partial.size() && i < stats.size(); ++i)
        S.moe_stats_partial[i] += stats[i];
      S.partial_count += cnt;
    }
    S.partial_parts++;
    S.st = LayerState::kPartial;
    if (S.partial_parts == int(o_.gpus.size())) {
      complete = true;
      mine = std::move(S.partial);
      my_stats = std::move(S.moe_stats_partial);
      total = S.partial_count;
      S.partial.clear();
      S.moe_stats_partial.clear();
    }
  }
  if (!complete) return;
  // normalise: attn groups + pooled (shared expert) by the token count; each expert by its own
  // routing-weight mass (sum of w^2), so every Hessian is an average xx^T under its own distribution
  double inv = 1.0 / double(total);
  for (size_t i = 0; i < mine.size(); ++i) {
    double s = inv;
    if (E && i >= 4) {
      int e = int((i - 4) / 2);
      double ws = my_stats[size_t(e)];
      s = ws > 0 ? 1.0 / ws : 0.0;
    }
    long n = long(mine[i].size());
    float* a = mine[i].data();
    float f = float(s);
#pragma omp parallel for schedule(static) num_threads(16)
    for (long k = 0; k < n; ++k) a[k] *= f;
  }
  double t0 = now_s();
  if (E) {
    int moe_dims[4] = {acc.dim[0], acc.dim[1], acc.dim[2], acc.dim[3]};
    write_hess_moe(hess_path(L), mine, my_stats, moe_dims, qcfg_.moe_inter, E, total);
  } else {
    write_hess(hess_path(L), mine, acc.dim, total);
  }
  logf("qwen layer %d: Hessians complete (%ld tokens%s), saved in %.1fs", L, total, E ? ", MoE" : "",
       now_s() - t0);
  {
    std::lock_guard<std::mutex> lk(mu_);
    LayerState& S = layers_[L];
    if (o_.mode == "collect") {
      S.st = LayerState::kDone;
    } else {
      S.H = std::move(mine);
      S.moe_stats = std::move(my_stats);
      S.st = LayerState::kReady;
    }
    hess_on_disk_.insert(L);
    save_progress_locked();
  }
  cv_.notify_all();
}

std::vector<std::vector<float>> Pipeline::take_hessians(int L) {
  std::lock_guard<std::mutex> lk(mu_);
  auto H = std::move(layers_[L].H);
  layers_[L].H.clear();
  return H;
}

// QTIP-quantize all Hessian groups of one layer (shared by --mode run/quantize and the adaptive mode).
std::vector<LinearOut> Pipeline::run_qtip_groups(int L, std::vector<std::vector<float>>&& H,
                                                 const std::shared_ptr<LayerHost>& LH, const QuantOpts& qo,
                                                 json& jl) {
  if (H.empty()) H = read_hess(hess_path(L));
  hipStream_t s = nullptr;
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  // a private stream + Gemm: caller's qs/qg may not exist on this device context
  HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  Gemm g(s);
  struct Grp {
    int hess;
    std::vector<int> lins;
  };
  std::vector<Grp> groups = {{0, {LQ, LK, LV}}, {1, {LO}}, {2, {LGate, LUp}}, {3, {LDown}}};
  if (o_.fp8_kv) groups[0].lins = {LQ};
  std::vector<LinearOut> outs(kNumLin);
  // rotate all four Hessians on the GPU first, then factorise them on the CPU in a side thread (fp64 LDL of
  // group g+1 overlaps the GPU LDLQ of group g)
  std::vector<HessPrep> preps(groups.size());
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Grp& G = groups[gi];
    int n = LH->in_dim[G.lins[0]];
    QCHECK(long(H[size_t(G.hess)].size()) == long(n) * n, "Hessian size mismatch");
    uint64_t seed = (o_.seed * 1000003ull + uint64_t(L)) * 16 + uint64_t(G.hess);
    rotate_hessian(s, H[size_t(G.hess)].data(), n, seed, qo, preps[gi]);
    H[size_t(G.hess)].clear();
    H[size_t(G.hess)].shrink_to_fit();
  }
  std::mutex lmu;
  std::condition_variable lcv;
  int ldl_done = 0;
  std::thread ldl_th([&] {
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      ldl_hessian(preps[gi], qo.ldl_threads, "L" + std::to_string(L) + "_" + kHessName[groups[gi].hess]);
      std::lock_guard<std::mutex> lk(lmu);
      ldl_done = int(gi) + 1;
      lcv.notify_all();
    }
  });
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Grp& G = groups[gi];
    std::vector<LinearIn> ins;
    for (int li : G.lins) {
      LinearIn in;
      in.name = kLinName[li];
      in.m = LH->out_dim[li];
      if (LH->bf16) in.w_bf16 = reinterpret_cast<const uint16_t*>(LH->w[li].data());
      else {
        in.w_fp8 = LH->w[li].data();
        in.scale_bf16 = LH->sc[li].data();
      }
      ins.push_back(in);
    }
    double tw = now_s();
    {
      std::unique_lock<std::mutex> lk(lmu);
      lcv.wait(lk, [&] { return ldl_done > int(gi); });
    }
    double ldl_wait = now_s() - tw;
    GroupTiming tm;
    auto res = quantize_group(s, g, ins, preps[gi], qo, &tm, "L" + std::to_string(L));
    preps[gi].HRr.clear();
    preps[gi].HRr.shrink_to_fit();
    if (ldl_wait > 0.5) logf("gpu%d layer %d group %s waited %.1fs for the CPU LDL", dev, L, kHessName[G.hess], ldl_wait);
    logf("gpu%d layer %d group %s: total %.1fs (load %.1f rht %.1f ldl_cpu %.1f ldlq %.1f [viterbi %.1f] proxy %.1f "
         "pack %.1f)",
         dev, L, kHessName[G.hess], tm.total, tm.load, tm.rht, tm.ldl_cpu, tm.ldlq, tm.viterbi, tm.proxy, tm.pack);
    for (size_t k = 0; k < res.size(); ++k) {
      const LinearOut& r = res[k];
      logf("  layer %d %-17s %5d x %5d  %.1fs  proxy %.6f  proxy_rot_reg %.6f  Wscale %.6g", L, r.name.c_str(), r.m,
           r.n, r.seconds, r.proxy, r.proxy_rot_reg, r.wscale);
      jl[r.name] = {{"m", r.m}, {"n", r.n}, {"seconds", r.seconds}, {"proxy", r.proxy},
                    {"proxy_rot_reg", r.proxy_rot_reg}, {"wscale", r.wscale}};
      outs[size_t(G.lins[k])] = res[k];
    }
    jl[std::string("group_") + kHessName[G.hess]] = {{"total", tm.total}, {"load", tm.load},       {"rht", tm.rht},
                                                     {"ldl_cpu", tm.ldl_cpu}, {"ldlq", tm.ldlq}, {"viterbi", tm.viterbi},
                                                     {"proxy", tm.proxy}, {"pack", tm.pack}};
  }
  ldl_th.join();
  HIP_CHECK(hipStreamDestroy(s));
  return outs;
}

size_t Pipeline::write_layer_file(int L, const LayerHost& LH, const char* format, const std::vector<LinearOut>* outs,
                                  const std::vector<Nvfp4Tensor>* nvs) {
  SafeTensorsWriter w;
  std::string pre = "model.layers." + std::to_string(L) + ".";
  const Codebook& cb = codebook();
  for (int li = 0; li < kNumLin; ++li) {
    std::string b = pre + kLinName[li] + ".";
    if (o_.fp8_kv && (li == LK || li == LV)) {
      const auto& w8 = LH.bf16 ? LH.kv8_w[li] : LH.w[li];
      const auto& sc = LH.bf16 ? LH.kv8_sc[li] : LH.sc[li];
      QCHECK(!w8.empty() && sc.size() == size_t(LH.out_dim[li]), "missing FP8 K/V");
      w.add(b + "weight", "F8_E4M3", {LH.out_dim[li], LH.in_dim[li]}, w8.data(), w8.size());
      w.add(b + "weight_scale", "BF16", {LH.out_dim[li]}, sc.data(), sc.size() * 2);
      continue;
    }
    if (std::string(format) == "qtip") {
      const LinearOut& r = (*outs)[size_t(li)];
      w.add(b + "trellis", "I16", {long(r.m / 16) * (r.n / 16), kWordsPerTrellis}, r.trellis.data(),
            r.trellis.size() * 2);
      w.add(b + "SU", "BF16", {r.n}, r.su_bf16.data(), r.su_bf16.size() * 2);
      std::vector<float> sv(r.sv.size());
      for (size_t i = 0; i < sv.size(); ++i) sv[i] = r.sv[i] * r.wscale;  // unpack_quip: SV <- SV * Wscale (rcp=0)
      w.add(b + "SV", "F32", {r.m}, sv.data(), sv.size() * 4);
      w.add(b + "tlut", "F16", {512, 2}, &cb.tlut_h[0][0], 512 * 2 * 2);
      int64_t zero = 0, eight = 8;
      w.add(b + "rcp", "I64", {}, &zero, 8);
      w.add(b + "tp_rank", "I64", {}, &eight, 8);
    } else {
      const Nvfp4Tensor& t = (*nvs)[size_t(li)];
      QCHECK(t.m == LH.out_dim[li] && t.n == LH.in_dim[li], "NVFP4 tensor geometry mismatch");
      w.add(b + "weight_packed", "U8", {t.m, long(t.n) / 2}, t.packed.data(), t.packed.size());
      w.add(b + "weight_scale", "F8_E4M3", {t.m, long(t.n) / 16}, t.scales.data(), t.scales.size());
      w.add(b + "weight_global_scale", "F32", {}, &t.global, 4);
    }
  }
  w.add(pre + "input_layernorm.weight", "BF16", {cfg_.hidden}, LH.ln_in.data(), LH.ln_in.size() * 2);
  w.add(pre + "post_attention_layernorm.weight", "BF16", {cfg_.hidden}, LH.ln_post.data(), LH.ln_post.size() * 2);
  return w.write(layer_path(L));
}

void Pipeline::finish_layer(int L, json& jl, double secs, int dev, const char* format) {
  std::lock_guard<std::mutex> lk(mu_);
  jl["status"] = "done";
  jl["seconds"] = secs;
  jl["gpu"] = dev;
  jl["finished"] = long(std::time(nullptr));
  if (format) jl["format"] = format;
  progress_["layers"][std::to_string(L)] = jl;
  layers_[L].st = LayerState::kDone;
  done_.insert(L);
  if (format) layer_format_[L] = format;
  if (!o_.keep_hessians && (o_.mode == "run" || o_.mode == "adaptive")) {
    std::error_code ec;
    fs::remove(hess_path(L), ec);
    hess_on_disk_.erase(L);
  }
  save_progress_locked();
  std::ofstream jf(o_.work + "/quant_log.jsonl", std::ios::app);
  json rec = jl;
  rec["layer"] = L;
  jf << rec.dump() << "\n";
  if (adaptive_) adaptive_report_write_locked();
}

void Pipeline::quantize_layer(int wi, LayerRunner* R, hipStream_t s, Gemm& g, int L) {
  (void)wi;
  (void)R;
  (void)s;
  (void)g;
  double t_layer = now_s();
  auto LH = cache_.get(L);
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  json jl = json::object();
  auto outs = run_qtip_groups(L, take_hessians(L), LH, o_.q, jl);
  size_t bytes = write_layer_file(L, *LH, "qtip", &outs, nullptr);
  double secs = now_s() - t_layer;
  logf("gpu%d layer %d quantized in %.1fs -> %s (%.1f MB)", dev, L, secs, layer_path(L).c_str(), bytes / 1e6);
  finish_layer(L, jl, secs, dev, "qtip");
  cv_.notify_all();
}

// ------------------------------------------------------------------ adaptive mode

void Pipeline::eval_runners_create(int device) {
  HIP_CHECK(hipSetDevice(device));
  const int T = int(probe_tok_.size());
  QCHECK(T > 0, "adaptive probe tokens not loaded");
  a_orig_ = std::make_unique<LayerRunner>(device, cfg_, T, T);
  a_eval_ = std::make_unique<LayerRunner>(device, cfg_, T, T);
}

void Pipeline::eval_runners_destroy() {
  a_orig_.reset();
  a_eval_.reset();
  // free the head/probe device buffers too: the next layer's QTIP quantization needs the VRAM
  d_head_.free();
  d_norm_.free();
  d_x_.free();
  d_logits_.free();
  head_loaded_ = false;
  HIP_CHECK(hipSetDevice(o_.gpus[0]));
}

// host bf16 [V][H] from a checkpoint tensor (BF16/F16/F32 supported)
static std::vector<uint16_t> tensor_to_bf16(const SafeTensorsReader& st, const std::string& name) {
  const TensorInfo& t = st.info(name);
  const long H = long(t.shape.back());
  std::vector<uint16_t> out(size_t(t.numel()));
  if (t.dtype == "BF16") {
    st.read_into(name, out.data(), out.size() * 2);
  } else if (t.dtype == "F32") {
    std::vector<float> v(size_t(t.numel()));
    st.read_into(name, v.data(), v.size() * 4);
    for (size_t i = 0; i < v.size(); ++i) out[i] = f32_to_bf16(v[i]);
  } else if (t.dtype == "F16") {
    std::vector<uint16_t> v(size_t(t.numel()));
    st.read_into(name, v.data(), v.size() * 2);
    for (size_t i = 0; i < v.size(); ++i) out[i] = f32_to_bf16(fp16_to_f32(v[i]));
  } else {
    die("adaptive KL eval needs BF16/F16/F32 " + name + ", checkpoint has " + t.dtype);
  }
  (void)H;
  return out;
}

void Pipeline::adaptive_load_head() {
  if (head_loaded_) return;
  const int T = int(probe_tok_.size());
  head_w_ = tensor_to_bf16(st_, "lm_head.weight");
  QCHECK(head_w_.size() == size_t(cfg_.vocab) * cfg_.hidden, "lm_head geometry mismatch");
  norm_w_ = tensor_to_bf16(st_, "model.norm.weight");
  QCHECK(norm_w_.size() == size_t(cfg_.hidden), "model.norm geometry mismatch");
  HIP_CHECK(hipSetDevice(o_.gpus[0]));
  d_head_.alloc(head_w_.size());
  HIP_CHECK(hipMemcpy(d_head_, head_w_.data(), head_w_.size() * 2, hipMemcpyHostToDevice));
  d_norm_.alloc(norm_w_.size());
  HIP_CHECK(hipMemcpy(d_norm_, norm_w_.data(), norm_w_.size() * 2, hipMemcpyHostToDevice));
  d_x_.alloc(size_t(T) * cfg_.hidden);
  // probe document (built once): embeddings of the first T calibration tokens, one causal document
  if (probe_.empty()) {
    std::vector<uint16_t> emb = tensor_to_bf16(st_, "model.embed_tokens.weight");
    QCHECK(emb.size() == size_t(cfg_.vocab) * cfg_.hidden, "embedding geometry mismatch");
    probe_.resize(size_t(T) * cfg_.hidden);
    for (int t = 0; t < T; ++t)
      std::memcpy(probe_.data() + size_t(t) * cfg_.hidden, emb.data() + size_t(probe_tok_[size_t(t)]) * cfg_.hidden,
                  size_t(cfg_.hidden) * 2);
    probe_at_ = 0;
  }
  head_loaded_ = true;
}

void Pipeline::probe_advance_to(int L) {
  if (!head_loaded_) adaptive_load_head();
  while (probe_at_ < L) {
    int l = probe_at_;
    auto LHl = cache_.get(l);
    a_orig_->upload(*LHl);
    if (o_.adaptive_next_input == "quantized") {
      // eager mode: the decision of layer l is the most recent one when catching up is impossible
      QCHECK(chosen_.layer == l && chosen_.quantized, "adaptive probe fell behind in quantized-input mode");
      for (int li = 0; li < kNumLin; ++li)
        if (!chosen_.w[size_t(li)].empty()) a_orig_->upload_raw(li, chosen_.w[size_t(li)].data());
    }
    a_orig_->forward(probe_.data(), 1, nullptr);
    probe_at_ = l + 1;
  }
}

std::vector<uint16_t> Pipeline::adaptive_layer_forward(const std::shared_ptr<LayerHost>& LH,
                                                       const std::vector<LinearOut>* qtip_out,
                                                       const std::vector<Nvfp4Tensor>* nv) {
  a_eval_->upload(*LH);  // originals first (also covers FP8-preserved K/V exactly)
  for (int li = 0; li < kNumLin; ++li) {
    if (o_.fp8_kv && (li == LK || li == LV)) {
      // BF16 sources: the candidate layer carries the dequantized FP8 K/V, not the original bf16
      if (LH->bf16) a_eval_->upload_raw(li, LH->kv8_deq[size_t(li)].data());
      continue;
    }
    std::vector<uint16_t> w;
    if (qtip_out) {
      const std::vector<float>& h = (*qtip_out)[size_t(li)].hatW;
      QCHECK(!h.empty(), "QTIP hatW missing for adaptive eval");
      w.resize(h.size());
#pragma omp parallel for schedule(static) if(h.size() >= (1u << 20))
      for (size_t i = 0; i < h.size(); ++i) w[i] = f32_to_bf16(h[i]);
    } else {
      std::vector<float> h = nvfp4_dequantize((*nv)[size_t(li)]);
      w.resize(h.size());
#pragma omp parallel for schedule(static) if(h.size() >= (1u << 20))
      for (size_t i = 0; i < h.size(); ++i) w[i] = f32_to_bf16(h[i]);
    }
    a_eval_->upload_raw(li, w.data());
  }
  std::vector<uint16_t> y = probe_;  // same input activations as the original layer
  a_eval_->forward(y.data(), 1, nullptr);
  return y;
}

// final norm + lm_head logits of a hidden state h [T][hidden], remaining layers (L+1..) kept original.
// The lm_head GEMM runs in vocab-row tiles so the fp32 logit buffer on the device stays bounded
// (Llama 3.3: 128256 vocab; the full head itself is resident, ~2.1 GB bf16).
std::vector<float> Pipeline::adaptive_logits(int L, std::vector<uint16_t> h) {
  const int T = int(probe_tok_.size());
  const long H = cfg_.hidden;
  for (int l = L + 1; l < cfg_.layers; ++l) {
    auto LHl = cache_.get(l);
    cache_.prefetch(l + 1);
    a_orig_->upload(*LHl);
    a_orig_->forward(h.data(), 1, nullptr);
  }
  HIP_CHECK(hipSetDevice(o_.gpus[0]));
  hipStream_t s = a_orig_->stream();
  HIP_CHECK(hipMemcpyAsync(d_x_, h.data(), size_t(T) * H * 2, hipMemcpyHostToDevice, s));
  DBuf<uint16_t> normed(size_t(T) * H);
  k_rmsnorm(d_x_, d_norm_, normed, T, int(H), cfg_.eps, s);
  // logits (T x V) = normed @ lm_head^T ; col-major: (V x T) = W_cm^T (V x H) * X_cm (H x T)
  std::vector<float> out(size_t(T) * cfg_.vocab);
  const long tile = o_.logit_tile > 0 ? o_.logit_tile : std::max<long>(1024, (256ll << 20) / (long(T) * 4));
  d_logits_.ensure(size_t(std::min(tile, long(cfg_.vocab))) * T);
  for (long v0 = 0; v0 < cfg_.vocab; v0 += tile) {
    const int vc = int(std::min(tile, cfg_.vocab - v0));
    a_orig_->gemm().run(true, false, vc, T, int(H), d_head_ + v0 * H, H, normed, H, d_logits_, vc, OutT::F32,
                        1.f, 0.f);
    // Transfer the contiguous tile once, then scatter its rows on the host.
    // One staged transfer per token otherwise synchronizes thousands of times.
    std::vector<float> host_tile(size_t(vc) * T);
    d2h_staged(host_tile.data(), d_logits_, host_tile.size() * sizeof(float), s);
#pragma omp parallel for schedule(static) if(T > 32)
    for (int t = 0; t < T; ++t)
      std::memcpy(out.data() + size_t(t) * cfg_.vocab + size_t(v0), host_tile.data() + size_t(t) * vc, size_t(vc) * 4);
  }
  return out;
}

double Pipeline::kl_mean_token(const std::vector<float>& p_logits, const std::vector<float>& q_logits) const {
  const int T = int(probe_tok_.size());
  const long V = cfg_.vocab;
  std::vector<double> row_kl(static_cast<size_t>(T));
#pragma omp parallel for schedule(static) if(long(T) * V >= (1u << 20))
  for (int t = 0; t < T; ++t) {
    const float *p = p_logits.data() + size_t(t) * V, *q = q_logits.data() + size_t(t) * V;
    double pm = -1e30, qm = -1e30;
    for (long i = 0; i < V; ++i) {
      pm = std::max(pm, double(p[i]));
      qm = std::max(qm, double(q[i]));
    }
    double lse_p = 0, lse_q = 0, row = 0;
    for (long i = 0; i < V; ++i) {
      double lp = p[i] - pm, lq = q[i] - qm;
      lse_p += std::exp(lp);
      lse_q += std::exp(lq);
    }
    lse_p = std::log(lse_p) + pm;
    lse_q = std::log(lse_q) + qm;
    for (long i = 0; i < V; ++i) {
      double pi = std::exp(double(p[i]) - lse_p);
      if (pi > 0) row += pi * ((double(p[i]) - lse_p) - (double(q[i]) - lse_q));
    }
    row_kl[size_t(t)] = row;
  }
  // Preserve the original token summation order regardless of thread count.
  double kl = 0;
  for (double row : row_kl) kl += row;
  return kl / T;
}

double Pipeline::nmse_layer(const std::vector<uint16_t>& y_q, const std::vector<uint16_t>& y_o) const {
  double num = 0, den = 0;
  QCHECK(y_q.size() == y_o.size(), "nmse size mismatch");
  for (size_t i = 0; i < y_o.size(); ++i) {
    double q = bf16_to_f32(y_q[i]), o = bf16_to_f32(y_o[i]);
    num += (q - o) * (q - o);
    den += o * o;
  }
  return den > 0 ? num / den : 0.0;
}

void Pipeline::adaptive_report_write_locked() {
  json layers = json::array();
  for (int L = o_.layer_begin; L < o_.layer_end; ++L)
    if (layer_format_.count(L)) {
      json e = {{"layer", L}, {"format", layer_format_[L]}};
      if (progress_["layers"].contains(std::to_string(L)) &&
          progress_["layers"][std::to_string(L)].contains("adaptive"))
        e["adaptive"] = progress_["layers"][std::to_string(L)]["adaptive"];
      layers.push_back(e);
    }
  report_["mode"] = "adaptive";
  report_["metric"] = o_.kl_metric;
  report_["threshold"] = o_.kl_threshold;
  report_["kl_tokens"] = o_.kl_tokens;
  report_["next_input"] = o_.adaptive_next_input;
  if (!o_.nvfp4_layers.empty()) report_["forced_nvfp4_layers"] = std::vector<int>(o_.nvfp4_layers.begin(), o_.nvfp4_layers.end());
  report_["nvfp4"] = {{"calibrate", o_.nvfp4_calibrate}, {"scale_search", o_.nvfp4_scale_search}};
  report_["kv_format"] = o_.fp8_kv ? "fp8" : "qtip";
  report_["layers"] = layers;
  std::string p = o_.out + "/adaptive_report.json", tmp = p + ".tmp";
  {
    std::ofstream f(tmp);
    f << report_.dump(2);
  }
  fs::rename(tmp, p);
}

bool Pipeline::adaptive_prev_decided(int L) const { return L <= o_.layer_begin || done_.count(L - 1) != 0; }

static std::vector<uint16_t> bf16_copy(const std::vector<float>& h) {
  std::vector<uint16_t> w(h.size());
#pragma omp parallel for schedule(static) if(h.size() >= (1u << 20))
  for (size_t i = 0; i < h.size(); ++i) w[i] = f32_to_bf16(h[i]);
  return w;
}

void Pipeline::adaptive_decide(int wi, LayerRunner* chain, hipStream_t qs, Gemm& qg, int L) {
  (void)wi;
  (void)qs;
  (void)qg;
  double t_layer = now_s();
  auto LH = cache_.get(L);
  int dev = o_.gpus[0];
  // 1) QTIP candidate (existing pipeline: Hessian -> RHT/LDL/LDLQ/Viterbi), keeping hatW for evaluation
  QuantOpts qo = o_.q;
  qo.want_hatW = true;
  json jl = json::object();
  auto Hs = take_hessians(L);
  if (Hs.empty() && fs::exists(hess_path(L))) Hs = read_hess(hess_path(L));  // resumed / seeded work dir
  // the NVFP4 fallback needs the Hessians after the QTIP pass consumed them: keep a copy (~1 GB at Llama-8B size)
  std::vector<std::vector<float>> Hnv;
  // forced mode (--nvfp4-layers): listed layers are NVFP4 without a QTIP pass, the others QTIP without a probe/metric
  const bool forced = !o_.nvfp4_layers.empty();
  const bool force_nv = forced && o_.nvfp4_layers.count(L) != 0;
  if (o_.nvfp4_calibrate || o_.nvfp4_scale_search) Hnv = Hs;
  std::vector<LinearOut> outs;
  if (!force_nv) outs = run_qtip_groups(L, std::move(Hs), LH, qo, jl);
  double t_qtip = now_s() - t_layer;
  // 2) metric vs the original layer on the probe document
  eval_runners_create(dev);
  probe_advance_to(L);
  std::vector<uint16_t> y_orig = probe_;
  {
    a_orig_->upload(*LH);
    a_orig_->forward(y_orig.data(), 1, nullptr);
  }
  std::vector<uint16_t> y_qtip;
  if (!forced || (!force_nv && o_.adaptive_next_input == "quantized")) y_qtip = adaptive_layer_forward(LH, &outs, nullptr);
  std::vector<float> fresh_teacher;
  const std::vector<float>* logits_orig = nullptr;
  if (o_.kl_metric == "lmkl" && !forced) {
    if (o_.cache_teacher_logits && o_.adaptive_next_input == "original") {
      if (teacher_logits_.empty()) teacher_logits_ = adaptive_logits(L, y_orig);
      logits_orig = &teacher_logits_;
    } else {
      fresh_teacher = adaptive_logits(L, y_orig);
      logits_orig = &fresh_teacher;
    }
  }
  auto metric_of = [&](const std::vector<uint16_t>& y_q) {
    if (o_.kl_metric == "mse") return nmse_layer(y_q, y_orig);
    return kl_mean_token(*logits_orig, adaptive_logits(L, y_q));
  };
  double m_qtip = -1;
  if (!forced) {
    m_qtip = metric_of(y_qtip);
    logf("adaptive layer %d: metric %s qtip %.6g (threshold %.6g)", L, o_.kl_metric.c_str(), m_qtip, o_.kl_threshold);
  } else {
    logf("adaptive layer %d: forced %s (--nvfp4-layers), no metric", L, force_nv ? "nvfp4" : "qtip");
  }
  // 3) NVFP4 fallback when the QTIP layer exceeds the threshold (or the layer is forced)
  bool nv = forced ? force_nv : m_qtip > o_.kl_threshold;
  std::vector<Nvfp4Tensor> nvs(kNumLin);
  std::vector<uint16_t> y_nv;
  double m_nv = -1;
  json nv_info = json::object();
  if (nv) {
    // original weights as fp32 (all linears), then one NVFP4 pass per Hessian group
    std::vector<std::vector<float>> Wf(kNumLin);
    for (int li = 0; li < kNumLin; ++li) {
      if (o_.fp8_kv && (li == LK || li == LV)) continue;
      const int m = LH->out_dim[li], n = LH->in_dim[li];
      std::vector<float>& W = Wf[size_t(li)];
      W.resize(size_t(m) * n);
#pragma omp parallel for schedule(static) if(size_t(m) * n >= (1u << 20))
      for (long i = 0; i < m; ++i) {
        if (LH->bf16) {
          const uint16_t* row = reinterpret_cast<const uint16_t*>(LH->w[size_t(li)].data()) + size_t(i) * n;
          for (long j = 0; j < n; ++j) W[size_t(i) * n + j] = bf16_to_f32(row[j]);
        } else {
          float sc = bf16_to_f32(LH->sc[size_t(li)][size_t(i)]);
          const uint8_t* row = LH->w[size_t(li)].data() + size_t(i) * n;
          for (long j = 0; j < n; ++j) W[size_t(i) * n + j] = fp8e4m3_to_f32(row[j]) * sc;
        }
      }
    }
    struct Grp {
      int hess;
      std::vector<int> lins;
    };
    std::vector<Grp> groups = {{0, {LQ, LK, LV}}, {1, {LO}}, {2, {LGate, LUp}}, {3, {LDown}}};
    if (o_.fp8_kv) groups[0].lins = {LQ};
    hipStream_t ns = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&ns, hipStreamNonBlocking));
    auto quantize_nv = [&](bool calibrate, bool search, bool* used_h) {
      std::vector<Nvfp4Tensor> res(kNumLin);
      bool all_cal = true;
      for (const Grp& G : groups) {
        std::vector<const float*> ws;
        std::vector<int> ms;
        for (int li : G.lins) ws.push_back(Wf[size_t(li)].data()), ms.push_back(LH->out_dim[li]);
        const int n = LH->in_dim[G.lins[0]];
        const float* Hp = Hnv.size() == 4 && long(Hnv[size_t(G.hess)].size()) == long(n) * n ? Hnv[size_t(G.hess)].data() : nullptr;
        if (!Hp && (calibrate || search))
          logf("WARNING layer %d NVFP4 group %s: Hessian unavailable, no calibration (scale search uses uniform weights)", L,
               kHessName[G.hess]);
        Nvfp4Opts no;
        no.calibrate = calibrate;
        no.scale_search = search;
        no.sigma_reg = o_.q.sigma_reg;
        bool cal = false;
        auto t = nvfp4_quantize_group(ns, ws, ms, n, Hp, no, &cal);
        all_cal = all_cal && (cal || !calibrate);
        for (size_t k = 0; k < G.lins.size(); ++k) res[size_t(G.lins[k])] = std::move(t[k]);
      }
      if (used_h) *used_h = all_cal;
      return res;
    };
    double t_nv0 = now_s();
    if (o_.nvfp4_compare && !forced) {
      json variants = json::object();
      for (auto [name, cal, srch] : {std::tuple<const char*, bool, bool>{"naive", false, false}, {"scale_search", false, true},
                                     {"calibrated", true, false}, {"calibrated_scale_search", true, true}}) {
        double tv = now_s();
        auto vn = quantize_nv(cal, srch, nullptr);
        double qs = now_s() - tv;
        double mv = metric_of(adaptive_layer_forward(LH, nullptr, &vn));
        logf("adaptive layer %d: NVFP4 variant %-24s metric %.6g (quantize %.1fs)", L, name, mv, qs);
        variants[name] = {{o_.kl_metric == "mse" ? "mse" : "kl", mv}, {"seconds", qs}};
      }
      nv_info["variants"] = variants;
    }
    bool used_h = false;
    nvs = quantize_nv(o_.nvfp4_calibrate, o_.nvfp4_scale_search, &used_h);
    nv_info["calibrate"] = o_.nvfp4_calibrate && used_h;
    nv_info["scale_search"] = o_.nvfp4_scale_search;
    nv_info["quantize_seconds"] = now_s() - t_nv0;
    HIP_CHECK(hipStreamDestroy(ns));
    if (!forced) {
      y_nv = adaptive_layer_forward(LH, nullptr, &nvs);
      m_nv = metric_of(y_nv);
      logf("adaptive layer %d: metric %s nvfp4 %.6g (qtip %.6g exceeded threshold %.6g)", L, o_.kl_metric.c_str(), m_nv,
           m_qtip, o_.kl_threshold);
    } else if (o_.adaptive_next_input == "quantized") {
      y_nv = adaptive_layer_forward(LH, nullptr, &nvs);
    }
  }
  const char* format = nv ? "nvfp4" : "qtip";
  const std::vector<uint16_t>& y_chosen = nv ? y_nv : y_qtip;
  // 4) mixed layer file + report
  size_t bytes = write_layer_file(L, *LH, format, nv ? nullptr : &outs, nv ? &nvs : nullptr);
  double secs = now_s() - t_layer;
  logf("gpu%d layer %d adaptive %s in %.1fs -> %s (%.1f MB)", dev, L, format, secs, layer_path(L).c_str(),
       bytes / 1e6);
  jl["adaptive"] = {{"metric", o_.kl_metric},
                    {"threshold", o_.kl_threshold},
                    {o_.kl_metric == "mse" ? "mse_qtip" : "kl_qtip", forced ? json(nullptr) : json(m_qtip)},
                    {o_.kl_metric == "mse" ? "mse_nvfp4" : "kl_nvfp4", nv && !forced ? json(m_nv) : json(nullptr)},
                    {"forced", forced},
                    {"qtip_seconds", t_qtip},
                    {"eval_seconds", secs - t_qtip}};
  if (nv) jl["adaptive"]["nvfp4"] = nv_info;
  // chosen weights + next probe input must be visible before the layer is marked done (other workers
  // replay the layer on the calibration activations as soon as done_ contains it)
  {
    std::lock_guard<std::mutex> lk(mu_);
    chosen_ = ChosenLayer{};
    chosen_.layer = L;
    chosen_.quantized = o_.adaptive_next_input == "quantized";
    chosen_.w.resize(kNumLin);
    for (int li = 0; chosen_.quantized && li < kNumLin; ++li) {
      if (o_.fp8_kv && (li == LK || li == LV)) {  // preserved: original FP8, or dequantized FP8 of a BF16 source
        if (LH->bf16) chosen_.w[size_t(li)] = LH->kv8_deq[size_t(li)];
        continue;
      }
      if (nv)
        chosen_.w[size_t(li)] = bf16_copy(nvfp4_dequantize(nvs[size_t(li)]));
      else
        chosen_.w[size_t(li)] = bf16_copy(outs[size_t(li)].hatW);
    }
    // probe becomes the input of layer L+1 (quantized-input chaining) or the original output
    probe_ = chosen_.quantized ? y_chosen : y_orig;
    probe_at_ = L + 1;
  }
  finish_layer(L, jl, secs, dev, format);  // sets done_ and appends the report entry
  eval_runners_destroy();
  cv_.notify_all();
  // quantized-input chaining: replay this layer on the calibration activations with the chosen weights so
  // the next layer's Hessians/KL input comes from the quantized model (worker 0's own chain)
  if (chosen_.quantized && chain && act_in_[size_t(0)]) {
    chain->upload(*LH);
    for (int li = 0; li < kNumLin; ++li)
      if (!chosen_.w[size_t(li)].empty()) chain->upload_raw(li, chosen_.w[size_t(li)].data());
    chain->forward(act_in_[size_t(0)].get(), act_nseq_[size_t(0)], nullptr);
    std::swap(act_[size_t(0)], act_in_[size_t(0)]);
  }
}

void Pipeline::quantize_qwen_layer(int wi, hipStream_t s, Gemm& g, int L) {
  (void)wi;
  double t_layer = now_s();
  std::vector<std::vector<float>> H;
  {
    std::lock_guard<std::mutex> lk(mu_);
    H = std::move(layers_[L].H);
  }
  auto QL = qcache_->get(L);
  const bool is_moe = QL->moe;
  std::vector<float> moe_stats;
  if (H.empty()) {
    if (is_moe) {
      int E = 0;
      H = read_hess_moe(hess_path(L), moe_stats, E);
      QCHECK(E == qcfg_.n_experts, "MoE Hessian expert count mismatch");
      std::lock_guard<std::mutex> lk(mu_);
      layers_[L].moe_stats = moe_stats;
    } else {
      H = read_hess(hess_path(L));
    }
  } else if (is_moe) {
    std::lock_guard<std::mutex> lk(mu_);
    moe_stats = layers_[L].moe_stats;
    QCHECK(moe_stats.size() == size_t(2 * qcfg_.n_experts), "missing MoE Hessian stats");
  }
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  const bool full = qcfg_.is_full(L);
  const int Hid = qcfg_.hidden;
  // linear index space of this layer type -> output tensor name, host fp32 weight and dims
  struct QLin {
    std::string name;
    int m = 0, n = 0;
    std::vector<float> w;  // fp32 rows of this linear [m][n]
  };
  std::vector<QLin> lins;
  auto add_lin = [&](std::string name, int m, int n, std::vector<float> w) {
    QLin q;
    q.name = std::move(name), q.m = m, q.n = n, q.w = std::move(w);
    lins.push_back(std::move(q));
  };
  if (full) {
    const int A = qcfg_.nq * qcfg_.hd;
    add_lin("self_attn.q_proj", qcfg_.nq * 2 * qcfg_.hd, Hid, bf16_to_f32_vec(QL->w_q));
    add_lin("self_attn.k_proj", qcfg_.nkv * qcfg_.hd, Hid, bf16_to_f32_vec(QL->w_k));
    add_lin("self_attn.v_proj", qcfg_.nkv * qcfg_.hd, Hid, bf16_to_f32_vec(QL->w_v));
    add_lin("self_attn.o_proj", Hid, A, bf16_to_f32_vec(QL->w_o));
  } else {
    const long nv = qcfg_.lin_nv, V = qcfg_.value_dim();
    add_lin("linear_attn.in_proj_qkv", qcfg_.conv_dim(), Hid, bf16_to_f32_vec(QL->w_in_qkv));
    add_lin("linear_attn.in_proj_z", int(V), Hid, bf16_to_f32_vec(QL->w_in_z));
    add_lin("linear_attn.out_proj", Hid, int(V), bf16_to_f32_vec(QL->w_out));
  }
  if (is_moe) {
    const int E = qcfg_.n_experts, F = qcfg_.moe_inter, S = qcfg_.shared_inter;
    for (int e = 0; e < E; ++e) {
      std::string ep = "mlp.experts." + std::to_string(e) + ".";
      add_lin(ep + "gate_proj", F, Hid, bf16_to_f32_vec(QL->w_exp_gate[size_t(e)]));
      add_lin(ep + "up_proj", F, Hid, bf16_to_f32_vec(QL->w_exp_up[size_t(e)]));
      add_lin(ep + "down_proj", Hid, F, bf16_to_f32_vec(QL->w_exp_down[size_t(e)]));
    }
    add_lin("mlp.shared_expert.gate_proj", S, Hid, bf16_to_f32_vec(QL->w_sh_gate));
    add_lin("mlp.shared_expert.up_proj", S, Hid, bf16_to_f32_vec(QL->w_sh_up));
    add_lin("mlp.shared_expert.down_proj", Hid, S, bf16_to_f32_vec(QL->w_sh_down));
  } else {
    add_lin("mlp.gate_proj", qcfg_.inter, Hid, bf16_to_f32_vec(QL->w_gate));
    add_lin("mlp.up_proj", qcfg_.inter, Hid, bf16_to_f32_vec(QL->w_up));
    add_lin("mlp.down_proj", Hid, qcfg_.inter, bf16_to_f32_vec(QL->w_down));
  }
  // groups: (Hessian matrix index in H, n, lins). MoE Hessian layout: [0,4) dense slots
  // (2/3 = pooled, shared expert), [4+2e] expert gateup, [5+2e] expert down.
  struct Grp {
    int hess;
    int n;
    std::vector<int> lins;
    std::string tag;  // log/seed tag
  };
  std::vector<Grp> groups;
  if (full)
    groups.push_back({0, Hid, {0, 1, 2}, "qkv"});
  else
    groups.push_back({0, Hid, {0, 1}, "qkv"});
  groups.push_back({1, full ? qcfg_.nq * qcfg_.hd : qcfg_.value_dim(), {full ? 3 : 2}, "o"});
  if (is_moe) {
    const int E = qcfg_.n_experts, F = qcfg_.moe_inter, S = qcfg_.shared_inter;
    int base = full ? 4 : 3;
    for (int e = 0; e < E; ++e) {
      int g0 = base + 3 * e;
      // starved experts (< moe_min_tok routed calibration tokens) use the pooled Hessian
      bool starved = moe_stats[size_t(E + e)] < float(o_.moe_min_tok);
      groups.push_back({starved ? 2 : 4 + 2 * e, Hid, {g0, g0 + 1}, "e" + std::to_string(e) + (starved ? "g*" : "g")});
      groups.push_back({starved ? 3 : 5 + 2 * e, starved ? S : F, {g0 + 2}, "e" + std::to_string(e) + (starved ? "d*" : "d")});
    }
    groups.push_back({2, Hid, {base + 3 * E, base + 3 * E + 1}, "sg"});
    groups.push_back({3, S, {base + 3 * E + 2}, "sd"});
  } else {
    groups.push_back({2, Hid, {full ? 4 : 3, full ? 5 : 4}, "gateup"});
    groups.push_back({3, qcfg_.inter, {full ? 6 : 5}, "down"});
  }
  std::vector<LinearOut> outs(lins.size());
  json jl = json::object();
  int n_starved = 0;
  std::vector<HessPrep> preps(groups.size());
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Grp& G = groups[gi];
    int n = G.n;
    QCHECK(long(H[size_t(G.hess)].size()) == long(n) * n, "Hessian size mismatch (group " + G.tag + ")");
    uint64_t seed = (o_.seed * 1000003ull + uint64_t(L)) * 16 + uint64_t(G.hess) + 1009ull * uint64_t(gi);
    rotate_hessian(s, H[size_t(G.hess)].data(), n, seed, o_.q, preps[gi]);
  }
  std::mutex lmu;
  std::condition_variable lcv;
  int ldl_done = 0;
  std::thread ldl_th([&] {
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      ldl_hessian(preps[gi], o_.q.ldl_threads, "L" + std::to_string(L) + "_" + groups[gi].tag);
      std::lock_guard<std::mutex> lk(lmu);
      ldl_done = int(gi) + 1;
      lcv.notify_all();
    }
  });
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Grp& G = groups[gi];
    std::vector<LinearIn> ins;
    for (int li : G.lins) {
      LinearIn in;
      in.name = lins[size_t(li)].name;
      in.m = lins[size_t(li)].m;
      QCHECK(int(lins[size_t(li)].w.size()) == long(in.m) * lins[size_t(li)].n, "weight size mismatch");
      in.w_f32 = lins[size_t(li)].w.data();
      ins.push_back(in);
    }
    std::unique_lock<std::mutex> lk(lmu);
    lcv.wait(lk, [&] { return ldl_done > int(gi); });
    lk.unlock();
    GroupTiming tm;
    auto res = quantize_group(s, g, ins, preps[gi], o_.q, &tm, "L" + std::to_string(L));
    preps[gi].HRr.clear();
    preps[gi].HRr.shrink_to_fit();
    bool expert_grp = G.tag.size() > 1 && (G.tag[0] == 'e') && G.tag[1] >= '0' && G.tag[1] <= '9';
    if (expert_grp) {
      if (G.tag.find("g*") != std::string::npos) n_starved++;
      // keep the JSON log small: one summary line per expert group, not into progress json
      logf("  qwen layer %d %-6s %5d x %5d  %.1fs  proxy %.6f", L, G.tag.c_str(), res[0].m, res[0].n,
           res[0].seconds, res[0].proxy);
    } else {
      logf("gpu%d qwen layer %d group %s: total %.1fs (load %.1f rht %.1f ldl_cpu %.1f ldlq %.1f [viterbi %.1f] "
           "proxy %.1f pack %.1f)",
           dev, L, G.tag.c_str(), tm.total, tm.load, tm.rht, tm.ldl_cpu, tm.ldlq, tm.viterbi, tm.proxy, tm.pack);
    }
    for (size_t k = 0; k < res.size(); ++k) {
      const LinearOut& r = res[k];
      if (!expert_grp) {
        logf("  qwen layer %d %-24s %5d x %5d  %.1fs  proxy %.6f  proxy_rot_reg %.6f  Wscale %.6g", L,
             r.name.c_str(), r.m, r.n, r.seconds, r.proxy, r.proxy_rot_reg, r.wscale);
        jl[r.name] = {{"m", r.m}, {"n", r.n}, {"seconds", r.seconds}, {"proxy", r.proxy},
                      {"proxy_rot_reg", r.proxy_rot_reg}, {"wscale", r.wscale}};
      }
      outs[size_t(G.lins[k])] = res[k];
    }
  }
  ldl_th.join();
  if (is_moe) {
    double px = 0;
    int np = 0;
    for (auto& o : outs)
      if (o.proxy >= 0) px += o.proxy, np++;
    jl["experts"] = {{"n", qcfg_.n_experts}, {"starved_pooled", n_starved},
                     {"mean_proxy", np ? px / np : 0.0}};
    logf("qwen layer %d: %d experts (%d starved -> pooled Hessian), mean proxy %.6f", L, qcfg_.n_experts,
         n_starved, np ? px / np : 0.0);
  }
  // ---- write layer file: QTIP tensors with the original HF names + unquantised small tensors
  SafeTensorsWriter w;
  std::string pre = "model.layers." + std::to_string(L) + ".";
  const Codebook& cb = codebook();
  for (size_t li = 0; li < lins.size(); ++li) {
    const LinearOut& r = outs[li];
    std::string b = pre + lins[li].name + ".";
    w.add(b + "trellis", "I16", {long(r.m / 16) * (r.n / 16), kWordsPerTrellis}, r.trellis.data(),
          r.trellis.size() * 2);
    w.add(b + "SU", "BF16", {r.n}, r.su_bf16.data(), r.su_bf16.size() * 2);
    std::vector<float> sv(r.sv.size());
    for (size_t i = 0; i < sv.size(); ++i) sv[i] = r.sv[i] * r.wscale;
    w.add(b + "SV", "F32", {r.m}, sv.data(), sv.size() * 4);
    w.add(b + "tlut", "F16", {512, 2}, &cb.tlut_h[0][0], 512 * 2 * 2);
    int64_t zero = 0, eight = 8;
    w.add(b + "rcp", "I64", {}, &zero, 8);
    w.add(b + "tp_rank", "I64", {}, &eight, 8);
  }
  w.add(pre + "input_layernorm.weight", "BF16", {qcfg_.hidden}, QL->ln_in.data(), QL->ln_in.size() * 2);
  w.add(pre + "post_attention_layernorm.weight", "BF16", {qcfg_.hidden}, QL->ln_post.data(),
        QL->ln_post.size() * 2);
  if (full) {
    w.add(pre + "self_attn.q_norm.weight", "BF16", {qcfg_.hd}, QL->q_norm.data(), QL->q_norm.size() * 2);
    w.add(pre + "self_attn.k_norm.weight", "BF16", {qcfg_.hd}, QL->k_norm.data(), QL->k_norm.size() * 2);
  } else {
    w.add(pre + "linear_attn.conv1d.weight", "BF16", {qcfg_.conv_dim(), 1, qcfg_.conv_k}, QL->conv_w.data(),
          QL->conv_w.size() * 2);
    {  // in_proj_b / in_proj_a stay bf16 (16 rows: below the 32-row trellis minimum)
      std::vector<uint16_t> b(QL->w_in_ba.begin(), QL->w_in_ba.begin() + size_t(qcfg_.lin_nv) * Hid);
      std::vector<uint16_t> a(QL->w_in_ba.begin() + size_t(qcfg_.lin_nv) * Hid, QL->w_in_ba.end());
      w.add(pre + "linear_attn.in_proj_b.weight", "BF16", {qcfg_.lin_nv, Hid}, b.data(), b.size() * 2);
      w.add(pre + "linear_attn.in_proj_a.weight", "BF16", {qcfg_.lin_nv, Hid}, a.data(), a.size() * 2);
    }
    w.add(pre + "linear_attn.A_log", "F32", {qcfg_.lin_nv}, QL->a_log.data(), QL->a_log.size() * 4);
    w.add(pre + "linear_attn.dt_bias", "F32", {qcfg_.lin_nv}, QL->dt_bias.data(), QL->dt_bias.size() * 4);
    w.add(pre + "linear_attn.norm.weight", "BF16", {qcfg_.lin_vd}, QL->dn_norm.data(), QL->dn_norm.size() * 2);
  }
  if (is_moe) {  // router + shared-expert gate stay unquantised (bf16/f32 verbatim)
    w.add(pre + "mlp.gate.weight", "BF16", {qcfg_.n_experts, qcfg_.hidden}, QL->w_router.data(),
          QL->w_router.size() * 2);
    if (!QL->w_router_bias.empty())
      w.add(pre + "mlp.gate.e_score_correction_bias", "F32", {qcfg_.n_experts}, QL->w_router_bias.data(),
            QL->w_router_bias.size() * 4);
    if (qcfg_.shared_gate)
      w.add(pre + "mlp.shared_expert_gate.weight", "BF16", {1, qcfg_.hidden}, QL->w_sh_gate_w.data(),
            QL->w_sh_gate_w.size() * 2);
  }
  size_t bytes = w.write(layer_path(L));
  double secs = now_s() - t_layer;
  logf("gpu%d qwen layer %d quantized in %.1fs -> %s (%.1f MB)", dev, L, secs, layer_path(L).c_str(), bytes / 1e6);
  {
    std::lock_guard<std::mutex> lk(mu_);
    jl["status"] = "done";
    jl["seconds"] = secs;
    jl["gpu"] = dev;
    jl["finished"] = long(std::time(nullptr));
    progress_["layers"][std::to_string(L)] = jl;
    layers_[L].st = LayerState::kDone;
    done_.insert(L);
    if (!o_.keep_hessians && o_.mode == "run") {
      std::error_code ec;
      fs::remove(hess_path(L), ec);
      hess_on_disk_.erase(L);
    }
    save_progress_locked();
    std::ofstream jf(o_.work + "/quant_log.jsonl", std::ios::app);
    json rec = jl;
    rec["layer"] = L;
    jf << rec.dump() << "\n";
  }
  cv_.notify_all();
}

// ------------------------------------------------------------------ workers / run

void Pipeline::worker(int wi) {
  int dev = o_.gpus[size_t(wi)];
  HIP_CHECK(hipSetDevice(dev));
  std::unique_ptr<LayerRunner> R;
  std::unique_ptr<QwenLayerRunner> QR;
  hipStream_t qs;
  HIP_CHECK(hipStreamCreateWithFlags(&qs, hipStreamNonBlocking));
  Gemm qg(qs);
  bool need_chain = chain_end_ > 0 && o_.mode != "quantize";
  if (need_chain) {
    if (qwen_) QR = std::make_unique<QwenLayerRunner>(dev, qcfg_, o_.batch_rows, o_.seq_len);
    else R = std::make_unique<LayerRunner>(dev, cfg_, o_.batch_rows, o_.seq_len);
  }
  int refwd = -1;  // adaptive quantized-input: layer whose chain output must be recomputed with chosen weights
  for (;;) {
    int task = -1;
    bool is_quant = false, collect = false;
    {
      std::unique_lock<std::mutex> lk(mu_);
      for (;;) {
        // 1) a ready layer to quantize (adaptive mode: worker 0 only, one layer at a time)
        if (o_.mode != "collect" && (!adaptive_ || wi == 0))
          for (auto& kv : layers_)
            if (kv.second.st == LayerState::kReady && kv.first >= o_.layer_begin && kv.first < o_.layer_end) {
              task = kv.first;
              is_quant = true;
              kv.second.st = LayerState::kQuantizing;
              break;
            }
        if (task >= 0) break;
        // 2) next chain layer. Adaptive mode is strict lockstep: layer L only after L-1 was decided.
        if (need_chain && chain_pos_[size_t(wi)] < chain_end_) {
          int L = chain_pos_[size_t(wi)];
          if (!adaptive_ || adaptive_prev_decided(L)) {
            collect = need_collect_.count(L) != 0;
            if (collect) {
              int backlog = 0;
              for (auto& kv : layers_)
                if (kv.second.st == LayerState::kPartial || kv.second.st == LayerState::kReady ||
                    kv.second.st == LayerState::kQuantizing)
                  backlog++;
              bool started = layers_.count(L) && layers_[L].st == LayerState::kPartial;
              if (adaptive_ || o_.mode == "collect" || started || backlog < o_.backlog) {
                task = L;
                chain_pos_[size_t(wi)]++;
                break;
              }
            } else {
              task = L;
              chain_pos_[size_t(wi)]++;
              break;
            }
          }
        }
        // 3) finished?
        bool chains_done = !need_chain || chain_pos_[size_t(wi)] >= chain_end_;
        bool any_pending = false;
        for (auto& kv : layers_)
          if (kv.first >= o_.layer_begin && kv.first < o_.layer_end && kv.second.st != LayerState::kDone)
            any_pending = true;
        if (chains_done && !any_pending) {
          lk.unlock();
          cv_.notify_all();
          if (R) {
            R.reset();
          }
          QR.reset();
          (void)hipStreamDestroy(qs);
          return;
        }
        if (chains_done && o_.mode == "collect") {
          lk.unlock();
          cv_.notify_all();
          (void)hipStreamDestroy(qs);
          return;
        }
        cv_.wait_for(lk, std::chrono::seconds(5));
      }
    }
    if (is_quant) {
      if (adaptive_) adaptive_decide(wi, R.get(), qs, qg, task);
      else if (qwen_) quantize_qwen_layer(wi, qs, qg, task);
      else quantize_layer(wi, R.get(), qs, qg, task);
    } else if (qwen_) {
      chain_step_qwen(wi, *QR, task, collect);
    } else {
      // adaptive quantized-input chaining: snapshot the calibration activations entering this layer
      if (adaptive_ && o_.adaptive_next_input == "quantized" && collect && !act_in_[size_t(wi)]) {
        act_in_[size_t(wi)].reset(new uint16_t[size_t(act_nseq_[size_t(wi)]) * o_.seq_len * cfg_.hidden]);
      }
      if (adaptive_ && o_.adaptive_next_input == "quantized" && collect && act_in_[size_t(wi)])
        std::memcpy(act_in_[size_t(wi)].get(), act_[size_t(wi)].get(),
                    size_t(act_nseq_[size_t(wi)]) * o_.seq_len * cfg_.hidden * 2);
      chain_step(wi, *R, task, collect);
      if (adaptive_ && o_.adaptive_next_input == "quantized" && collect) refwd = task;
      if (refwd == task && wi != 0) {
        // non-deciding workers wait for the decision, then replay the layer with the chosen weights
        {
          std::unique_lock<std::mutex> lk(mu_);
          cv_.wait(lk, [&] { return done_.count(task) != 0; });
          bool quant = chosen_.layer == task && chosen_.quantized;
          if (quant) {
            auto LHl = cache_.get(task);
            R->upload(*LHl);
            for (int li = 0; li < kNumLin; ++li)
              if (!chosen_.w[size_t(li)].empty()) R->upload_raw(li, chosen_.w[size_t(li)].data());
            lk.unlock();
            R->forward(act_in_[size_t(wi)].get(), act_nseq_[size_t(wi)], nullptr);
            std::swap(act_[size_t(wi)], act_in_[size_t(wi)]);
          }
        }
      }
      // worker 0's replay happens at the end of adaptive_decide (it makes the decision itself)
      if (!(adaptive_ && o_.adaptive_next_input == "quantized" && collect)) refwd = -1;
    }
  }
}

void Pipeline::run() {
  const int ng = int(o_.gpus.size());
  // which layers still need work
  for (int L = o_.layer_begin; L < o_.layer_end; ++L) {
    if (done_.count(L)) {
      layers_[L].st = LayerState::kDone;
      continue;
    }
    if (hess_on_disk_.count(L)) {
      layers_[L].st = (o_.mode == "collect") ? LayerState::kDone : LayerState::kReady;
      continue;
    }
    if (o_.mode == "quantize") {
      logf("layer %d: no Hessian file in %s, skipping", L, o_.hess_dir.c_str());
      continue;
    }
    need_collect_.insert(L);
    layers_[L].st = LayerState::kNone;
  }
  if (o_.mode == "quantize") need_collect_.clear();
  chain_end_ = need_collect_.empty() ? 0 : (*need_collect_.rbegin() + 1);
  logf("layers [%d, %d): %zu done, %zu with Hessians, %zu to collect (chains run layers 0..%d), %d GPU(s)",
       o_.layer_begin, o_.layer_end, done_.size(), hess_on_disk_.size(), need_collect_.size(), chain_end_ - 1, ng);
  if (chain_end_ > 0) {
    QCHECK(!o_.tokens.empty(), "--tokens required for Hessian collection");
    std::ifstream tf(o_.tokens, std::ios::binary | std::ios::ate);
    QCHECK(tf, "cannot open tokens " + o_.tokens);
    long nseq_file = long(tf.tellg()) / 4 / o_.seq_len;
    long nseq = std::min(nseq_file, o_.ntok / o_.seq_len);
    QCHECK(nseq >= ng, "not enough sequences");
    std::vector<int32_t> tok(size_t(nseq) * o_.seq_len);
    tf.seekg(0);
    tf.read(reinterpret_cast<char*>(tok.data()), std::streamsize(tok.size() * 4));
    if (adaptive_) {
      // probe document: the first --kl-tokens tokens as one causal sequence
      probe_tok_.assign(tok.begin(), tok.begin() + std::min<long>(o_.kl_tokens, long(tok.size())));
      QCHECK(int(probe_tok_.size()) == o_.kl_tokens, "not enough tokens for --kl-tokens");
      logf("adaptive: metric %s threshold %.6g probe %d tokens, next input %s", o_.kl_metric.c_str(),
           o_.kl_threshold, o_.kl_tokens, o_.adaptive_next_input.c_str());
    }
    // embeddings
    std::string embed_name = "model.embed_tokens.weight";
    if (qwen_) {
      for (std::string nm : {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"})
        if (st_.has(nm)) embed_name = nm;
    }
    const TensorInfo& ei = st_.info(embed_name);
    std::vector<uint16_t> emb(size_t(ei.numel()));
    double t0 = now_s();
    st_.read_into(embed_name, emb.data(), emb.size() * 2);
    logf("calibration: %ld sequences x %d tokens (%ld tokens); embeddings read in %.1fs", nseq, o_.seq_len,
         nseq * o_.seq_len, now_s() - t0);
    const long H = qwen_ ? qcfg_.hidden : cfg_.hidden;
    const int vocab = qwen_ ? qcfg_.vocab : cfg_.vocab;
    act_.resize(static_cast<size_t>(ng));
    act_in_.resize(static_cast<size_t>(ng));
    act_nseq_.resize(static_cast<size_t>(ng));
    long per = (nseq + ng - 1) / ng;
    for (int g = 0; g < ng; ++g) {
      long s0 = g * per, s1 = std::min(nseq, s0 + per);
      act_nseq_[size_t(g)] = s1 - s0;
      act_[size_t(g)].reset(new uint16_t[size_t(s1 - s0) * o_.seq_len * H]);
      uint16_t* a = act_[size_t(g)].get();
#pragma omp parallel for schedule(static) num_threads(32)
      for (long t = 0; t < (s1 - s0) * o_.seq_len; ++t) {
        int id = tok[size_t(s0 * o_.seq_len + t)];
        QCHECK(id >= 0 && id < vocab, "token id out of range");
        std::memcpy(a + size_t(t) * H, emb.data() + size_t(id) * H, size_t(H) * 2);
      }
    }
    logf("activations: %.1f GB host RAM", double(nseq) * o_.seq_len * H * 2 / 1e9);
    if (adaptive_ && o_.adaptive_next_input == "quantized")
      logf("adaptive quantized-input chaining: one extra activation snapshot + layer replay per layer "
           "(~2x chain host RAM and compute)");
  }
  if (adaptive_ && probe_tok_.empty()) {
    // resume with all Hessians already on disk: no chain is needed, but the metric probe still is
    QCHECK(!o_.tokens.empty(), "--tokens required for --mode adaptive");
    std::ifstream tf(o_.tokens, std::ios::binary | std::ios::ate);
    QCHECK(tf, "cannot open tokens " + o_.tokens);
    long avail = long(tf.tellg()) / 4;
    QCHECK(avail >= o_.kl_tokens, "not enough tokens for --kl-tokens");
    probe_tok_.resize(size_t(o_.kl_tokens));
    tf.seekg(0);
    tf.read(reinterpret_cast<char*>(probe_tok_.data()), std::streamsize(probe_tok_.size() * 4));
    for (int32_t id : probe_tok_) QCHECK(id >= 0 && id < cfg_.vocab, "token id out of range");
  }
  chain_pos_.assign(size_t(ng), 0);
  std::vector<std::thread> th;
  double t0 = now_s();
  for (int g = 0; g < ng; ++g) th.emplace_back([this, g] { worker(g); });
  for (auto& t : th) t.join();
  logf("pipeline finished in %.1fs", now_s() - t0);
  bool all = true;
  for (int L = 0; L < layers_total(); ++L) all = all && fs::exists(layer_path(L));
  if (all) assemble();
  else logf("not all %d layers present in %s yet: run --mode assemble when complete", layers_total(),
            o_.out.c_str());
}

void Pipeline::assemble() {
  // config.json (QTIP hfized llama layout), embeddings/norm/lm_head, index, tokenizer files
  json c = json::parse(qwen_ ? qcfg_.raw_json : cfg_.raw_json);
  c.erase("quantization_config");
  if (!qwen_) {  // Qwen3.5 keeps its original qwen3_5 arch/layer_types; the runtime reads the qwen fields itself
    c["architectures"] = {"LlamaForCausalLM"};
    c["model_type"] = "llama";
    c["torch_dtype"] = "bfloat16";
    c["attention_bias"] = false;
    c["mlp_bias"] = false;
    c["pretraining_tp"] = 1;
    // Keep the source RoPE configuration: Behemoth (plain RoPE) gets an explicit null like before;
    // Llama 3.3 keeps its llama3 rope_scaling block verbatim (the runtime needs it).
    if (!c.contains("rope_scaling") || c["rope_scaling"].is_null()) c["rope_scaling"] = nullptr;
  }
  c["quip_params"] = {{"codebook", "bitshift"}, {"codebook_version", 0}, {"L", kL},      {"K", kK},
                      {"V", kV},                {"tlut_bits", kTlutBits}, {"decode_mode", "quantlut_sym"},
                      {"td_x", kTd},            {"td_y", kTd},           {"split_for_tp", false},
                      {"skip_list", nullptr}};
  c["qtip_quant"] = {{"sigma_reg", o_.q.sigma_reg}, {"scale_override", o_.q.scale_override},
                     {"ntok", o_.ntok},             {"seq_len", o_.seq_len},
                     {"shared_su_per_hessian_group", true}, {"kv_format", o_.fp8_kv ? "fp8" : "qtip"}};
  if (!layer_format_.empty()) {
    // mixed-format (adaptive) checkpoint: tell loaders which layers are NVFP4
    json fmts = json::array();
    for (int L = 0; L < cfg_.layers; ++L)
      fmts.push_back(layer_format_.count(L) ? layer_format_[L] : "qtip");
    c["qtip_quant"]["layer_formats"] = fmts;
    c["qtip_quant"]["adaptive"] = {{"mixed_format", true},
                                   {"metric", o_.kl_metric},
                                   {"threshold", o_.kl_threshold},
                                   {"kl_tokens", o_.kl_tokens},
                                   {"next_input", o_.adaptive_next_input},
                                   {"nvfp4", {{"calibrate", o_.nvfp4_calibrate}, {"scale_search", o_.nvfp4_scale_search}}}};
  }
  {
    std::ofstream f(o_.out + "/config.json");
    f << c.dump(2);
  }
  std::string ef = o_.out + "/model-embed.safetensors";
  if (!fs::exists(ef)) {
    SafeTensorsWriter w;
    if (qwen_) {
      // explicit candidates first: a plain suffix scan also matches mtp.norm.weight /
      // model.visual.merger.norm.weight and the last match would win with the WRONG tensor
      std::string norm_name, embed_name;
      for (std::string nm : {"model.language_model.norm.weight", "model.norm.weight"})
        if (st_.has(nm)) norm_name = nm;
      for (std::string nm : {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"})
        if (st_.has(nm)) embed_name = nm;
      if (norm_name.empty() || embed_name.empty()) {
        for (auto& kv : st_.tensors()) {
          if (kv.first.find("mtp.") == 0 || kv.first.find("model.visual.") == 0) continue;
          if (kv.first.size() >= 11 && kv.first.substr(kv.first.size() - 11) == "norm.weight" &&
              kv.first.find("layers.") == std::string::npos && norm_name.empty())
            norm_name = kv.first;
          if (kv.first.size() > 19 && kv.first.substr(kv.first.size() - 19) == "embed_tokens.weight" &&
              embed_name.empty())
            embed_name = kv.first;
        }
      }
      QCHECK(!norm_name.empty() && !embed_name.empty(), "cannot find final norm / embeddings");
      for (std::string nm : {embed_name, norm_name}) {
        const TensorInfo& t = st_.info(nm);
        std::vector<uint8_t> v = st_.read(nm);
        w.add(nm, t.dtype, t.shape, v.data(), v.size());
      }
      if (!qcfg_.tied && st_.has("lm_head.weight")) {
        const TensorInfo& t = st_.info("lm_head.weight");
        std::vector<uint8_t> v = st_.read("lm_head.weight");
        w.add("lm_head.weight", t.dtype, t.shape, v.data(), v.size());
      }
    } else {
      for (std::string nm : {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}) {
        const TensorInfo& t = st_.info(nm);
        std::vector<uint8_t> v = st_.read(nm);
        w.add(nm, t.dtype, t.shape, v.data(), v.size());
      }
    }
    w.write(ef);
  }
  json wm = json::object();
  uint64_t total = 0;
  for (auto& e : fs::directory_iterator(o_.out)) {
    std::string fn = e.path().filename().string();
    if (fn.rfind("model-", 0) != 0 || e.path().extension() != ".safetensors") continue;
    std::ifstream f(e.path(), std::ios::binary);
    uint64_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 8);
    std::string hs(hl, '\0');
    f.read(hs.data(), std::streamsize(hl));
    json h = json::parse(hs);
    for (auto& kv : h.items())
      if (kv.key() != "__metadata__") wm[kv.key()] = fn;
    total += fs::file_size(e.path());
  }
  json idx = {{"metadata", {{"total_size", total}}}, {"weight_map", wm}};
  {
    std::ofstream f(o_.out + "/model.safetensors.index.json");
    f << idx.dump(2);
  }
  for (std::string fn : {"tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                         "generation_config.json", "vocab.json", "merges.txt"}) {
    fs::path src = fs::path(o_.model) / fn;
    if (fs::exists(src)) fs::copy_file(src, fs::path(o_.out) / fn, fs::copy_options::overwrite_existing);
  }
  logf("assembled %s: %zu tensors, %.2f GB", o_.out.c_str(), wm.size(), total / 1e9);
}

}  // namespace qtip
