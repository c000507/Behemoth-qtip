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

Pipeline::Pipeline(const Options& o) : o_(o), st_(o.model), cfg_(load_model_config(o.model, &st_)), cache_(st_, cfg_, 4) {
  if (o_.layer_end < 0) o_.layer_end = cfg_.layers;
  if (o_.hess_dir.empty()) o_.hess_dir = o_.work + "/hess";
  // BF16 sources (e.g. Llama 3.3) have no FP8 K/V to preserve: default to all-QTIP, reject an
  // explicit --kv-format fp8.
  if (cfg_.bf16_source) {
    QCHECK(!o_.kv_explicit || !o_.fp8_kv, "--kv-format fp8 requires an FP8-Dynamic source; BF16 sources use qtip");
    o_.fp8_kv = false;
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
      w.add(b + "weight", "F8_E4M3", {LH.out_dim[li], LH.in_dim[li]}, LH.w[li].data(), LH.w[li].size());
      w.add(b + "weight_scale", "BF16", {LH.out_dim[li]}, LH.sc[li].data(), LH.sc[li].size() * 2);
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
    if (o_.fp8_kv && (li == LK || li == LV)) continue;
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
  if (o_.nvfp4_calibrate || o_.nvfp4_scale_search) Hnv = Hs;
  auto outs = run_qtip_groups(L, std::move(Hs), LH, qo, jl);
  double t_qtip = now_s() - t_layer;
  // 2) metric vs the original layer on the probe document
  eval_runners_create(dev);
  probe_advance_to(L);
  std::vector<uint16_t> y_orig = probe_;
  {
    a_orig_->upload(*LH);
    a_orig_->forward(y_orig.data(), 1, nullptr);
  }
  auto y_qtip = adaptive_layer_forward(LH, &outs, nullptr);
  std::vector<float> fresh_teacher;
  const std::vector<float>* logits_orig = nullptr;
  if (o_.kl_metric == "lmkl") {
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
  double m_qtip = metric_of(y_qtip);
  logf("adaptive layer %d: metric %s qtip %.6g (threshold %.6g)", L, o_.kl_metric.c_str(), m_qtip, o_.kl_threshold);
  // 3) NVFP4 fallback when the QTIP layer exceeds the threshold
  bool nv = m_qtip > o_.kl_threshold;
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
    if (o_.nvfp4_compare) {
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
    y_nv = adaptive_layer_forward(LH, nullptr, &nvs);
    m_nv = metric_of(y_nv);
    logf("adaptive layer %d: metric %s nvfp4 %.6g (qtip %.6g exceeded threshold %.6g)", L, o_.kl_metric.c_str(), m_nv,
         m_qtip, o_.kl_threshold);
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
                    {o_.kl_metric == "mse" ? "mse_qtip" : "kl_qtip", m_qtip},
                    {o_.kl_metric == "mse" ? "mse_nvfp4" : "kl_nvfp4", nv ? json(m_nv) : json(nullptr)},
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
      if (o_.fp8_kv && (li == LK || li == LV)) continue;  // preserved originals
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

// ------------------------------------------------------------------ workers / run

void Pipeline::worker(int wi) {
  int dev = o_.gpus[size_t(wi)];
  HIP_CHECK(hipSetDevice(dev));
  std::unique_ptr<LayerRunner> R;
  hipStream_t qs;
  HIP_CHECK(hipStreamCreateWithFlags(&qs, hipStreamNonBlocking));
  Gemm qg(qs);
  bool need_chain = chain_end_ > 0 && o_.mode != "quantize";
  if (need_chain) R = std::make_unique<LayerRunner>(dev, cfg_, o_.batch_rows, o_.seq_len);
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
      else quantize_layer(wi, R.get(), qs, qg, task);
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
    const TensorInfo& ei = st_.info("model.embed_tokens.weight");
    std::vector<uint16_t> emb(size_t(ei.numel()));
    double t0 = now_s();
    st_.read_into("model.embed_tokens.weight", emb.data(), emb.size() * 2);
    logf("calibration: %ld sequences x %d tokens (%ld tokens); embeddings read in %.1fs", nseq, o_.seq_len,
         nseq * o_.seq_len, now_s() - t0);
    const long H = cfg_.hidden;
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
        QCHECK(id >= 0 && id < cfg_.vocab, "token id out of range");
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
  for (int L = 0; L < cfg_.layers; ++L) all = all && fs::exists(layer_path(L));
  if (all) assemble();
  else logf("not all %d layers present in %s yet: run --mode assemble when complete", cfg_.layers, o_.out.c_str());
}

void Pipeline::assemble() {
  // config.json (QTIP hfized llama layout), embeddings/norm/lm_head, index, tokenizer files
  json c = json::parse(cfg_.raw_json);
  c.erase("quantization_config");
  c["architectures"] = {"LlamaForCausalLM"};
  c["model_type"] = "llama";
  c["torch_dtype"] = "bfloat16";
  c["attention_bias"] = false;
  c["mlp_bias"] = false;
  c["pretraining_tp"] = 1;
  // Keep the source RoPE configuration: Behemoth (plain RoPE) gets an explicit null like before;
  // Llama 3.3 keeps its llama3 rope_scaling block verbatim (the runtime needs it).
  if (!c.contains("rope_scaling") || c["rope_scaling"].is_null()) c["rope_scaling"] = nullptr;
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
    for (std::string nm : {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}) {
      const TensorInfo& t = st_.info(nm);
      std::vector<uint8_t> v = st_.read(nm);
      w.add(nm, t.dtype, t.shape, v.data(), v.size());
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
                         "generation_config.json"}) {
    fs::path src = fs::path(o_.model) / fn;
    if (fs::exists(src)) fs::copy_file(src, fs::path(o_.out) / fn, fs::copy_options::overwrite_existing);
  }
  logf("assembled %s: %zu tensors, %.2f GB", o_.out.c_str(), wm.size(), total / 1e9);
}

}  // namespace qtip
