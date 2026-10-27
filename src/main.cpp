// qtip_quant: QTIP (HYB 2-bit, L=16 K=2 V=2, quantlut_sym) quantization of Behemoth-R1-123B-v2 on ROCm.
// Pipeline: calibration tokens -> original-model forward per layer on each GPU (token halves) accumulating the input
// Hessians -> per-layer QTIP quantization (RHT, block LDL, LDLQ with GPU Viterbi) -> QTIP-format safetensors.
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <thread>

#include "common.hpp"
#include "gpu.hpp"
#include "model.hpp"
#include "qtip_core.hpp"
#include "quantizer.hpp"
#include "qwen35.hpp"
#include "qwen_runner.hpp"
#include "safetensors.hpp"

using namespace qtip;
namespace fs = std::filesystem;
using json = nlohmann::json;

struct Options {
  std::string mode = "run";  // run | collect | quantize | assemble
  std::string model = "/mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic";
  std::string tokens;        // int32 [nseq][seq_len]
  std::string out;           // output model dir
  std::string work;          // progress, logs, Hessians
  std::string hess_dir;      // default <work>/hess
  int layer_begin = 0, layer_end = -1;
  std::vector<int> gpus = {0, 1};
  long ntok = 4194304;
  int seq_len = 2048;
  int batch_rows = 4096;
  int backlog = 2;
  bool keep_hessians = false;
  uint64_t seed = 0;
  std::string dump_acts;  // write the first sequence of GPU 0 after every chain layer (bf16)
  QuantOpts q;
};

static void usage() {
  std::printf(
      "usage: qtip_quant --out DIR --work DIR [--tokens FILE] [options]\n"
      "  --mode run|collect|quantize|assemble  (default run: collect Hessians + quantize)\n"
      "  --model DIR          FP8-Dynamic checkpoint (default /mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic)\n"
      "  --tokens FILE        calibration tokens, raw int32 [nseq][seq_len]\n"
      "  --ntok N             calibration tokens to use (default 4194304)\n"
      "  --seq-len N          sequence length of the token file (default 2048)\n"
      "  --layers A:B         layer range [A, B) (default all)\n"
      "  --gpus 0,1           HIP devices (each GPU owns a token half and runs its own layer chain)\n"
      "  --batch-rows N       tokens per forward batch (default 4096)\n"
      "  --hess-dir DIR       Hessian files (default <work>/hess)\n"
      "  --backlog N          max layers with collected-but-unquantized Hessians (default 2, ~5.1 GB each)\n"
      "  --keep-hessians      do not delete Hessians after a layer is quantized\n"
      "  --vit-batch N        trellises per Viterbi launch (default 1024; 260 KB scratch each)\n"
      "  --sigma-reg X        Hessian damping (default 1e-2)   --scale-override X (default 0.9)\n"
      "  --ldl-threads N      CPU threads per fp64 LDL (default 16)\n"
      "  --dump-dir DIR       write W / reconstructed hatW (fp32) per linear for validation\n"
      "  --seed N             sign-vector seed (default 0)\n"
      "  --dump-acts DIR      debug: dump sequence 0 (GPU 0) activations after each chain layer\n");
}

static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      QCHECK(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--mode") o.mode = next();
    else if (a == "--model") o.model = next();
    else if (a == "--tokens") o.tokens = next();
    else if (a == "--out") o.out = next();
    else if (a == "--work") o.work = next();
    else if (a == "--hess-dir") o.hess_dir = next();
    else if (a == "--ntok") o.ntok = std::stol(next());
    else if (a == "--seq-len") o.seq_len = std::stoi(next());
    else if (a == "--batch-rows") o.batch_rows = std::stoi(next());
    else if (a == "--backlog") o.backlog = std::stoi(next());
    else if (a == "--keep-hessians") o.keep_hessians = true;
    else if (a == "--vit-batch") o.q.vit_batch = std::stoi(next());
    else if (a == "--sigma-reg") o.q.sigma_reg = std::stod(next());
    else if (a == "--scale-override") o.q.scale_override = std::stod(next());
    else if (a == "--ldl-threads") o.q.ldl_threads = std::stoi(next());
    else if (a == "--dump-dir") o.q.dump_dir = next();
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--dump-acts") o.dump_acts = next();
    else if (a == "--layers") {
      std::string v = next();
      auto c = v.find(':');
      QCHECK(c != std::string::npos, "--layers A:B");
      o.layer_begin = std::stoi(v.substr(0, c));
      o.layer_end = std::stoi(v.substr(c + 1));
    } else if (a == "--gpus") {
      o.gpus.clear();
      std::stringstream ss(next());
      std::string t;
      while (std::getline(ss, t, ',')) o.gpus.push_back(std::stoi(t));
    } else if (a == "-h" || a == "--help") {
      usage();
      std::exit(0);
    } else {
      usage();
      die("unknown option " + a);
    }
  }
  QCHECK(!o.out.empty() && !o.work.empty(), "--out and --work are required");
  if (o.hess_dir.empty()) o.hess_dir = o.work + "/hess";
  return o;
}

// ------------------------------------------------------------------ progress / state
static ModelConfig load_model_config_qtip(const std::string& dir) {
  if (is_qwen35_dir(dir)) return ModelConfig{};  // the qwen path uses qcfg_
  return load_model_config(dir);
}
static QwenConfig load_qwen_config_or_empty(const std::string& dir) {
  QwenConfig c;
  if (!is_qwen35_dir(dir)) return c;  // raw_json stays empty
  return load_qwen_config(dir);
}

struct LayerState {
  enum St { kNone, kPartial, kReady, kQuantizing, kDone } st = kNone;
  std::vector<std::vector<float>> H;  // combined (normalised) Hessians in RAM when kReady (may be empty: on disk)
  std::vector<std::vector<float>> partial;
  long partial_count = 0;
  int partial_parts = 0;
};

class Pipeline {
 public:
  Pipeline(const Options& o)
      : o_(o),
        st_(o.model),
        cfg_(load_model_config_qtip(o.model)),
        cache_(st_, cfg_, 4),
        qcfg_(load_qwen_config_or_empty(o.model)),
        qwen_(!qcfg_.raw_json.empty()),
        qcache_(qwen_ ? std::make_unique<QwenLayerCache>(st_, qcfg_, 4) : nullptr) {
    if (qwen_) {
      o_.layer_end = o_.layer_end < 0 ? qcfg_.layers : o_.layer_end;
    } else if (o_.layer_end < 0) {
      o_.layer_end = cfg_.layers;
    }
    fs::create_directories(o_.out);
    fs::create_directories(o_.work);
    fs::create_directories(o_.hess_dir);
    if (!o_.q.dump_dir.empty()) fs::create_directories(o_.q.dump_dir);
    log_set_file(o_.work + "/qtip_quant.log");
    load_progress();
  }
  void run();
  void assemble();

 private:
  // progress
  void load_progress();
  void save_progress_locked();
  std::string hess_path(int L) const { return o_.hess_dir + "/layer" + std::to_string(L) + ".safetensors"; }
  std::string layer_path(int L) const {
    char b[64];
    std::snprintf(b, sizeof b, "/model-layer-%05d.safetensors", L);
    return o_.out + b;
  }
  // tasks
  void worker(int wi);
  void chain_step(int wi, LayerRunner& R, int L, bool collect);
  void chain_step_qwen(int wi, QwenLayerRunner& R, int L, bool collect);
  void quantize_layer(int wi, LayerRunner* R, hipStream_t s, Gemm& g, int L);
  void quantize_qwen_layer(int wi, hipStream_t s, Gemm& g, int L);
  void deposit_partial(int L, HessAcc& acc, LayerRunner& R);
  void deposit_partial_qwen(int L, HessAcc& acc, QwenLayerRunner& R);
  int layers_total() const { return qwen_ ? qcfg_.layers : cfg_.layers; }

  Options o_;
  SafeTensorsReader st_;
  ModelConfig cfg_;
  LayerCache cache_;
  QwenConfig qcfg_;
  bool qwen_;
  std::unique_ptr<QwenLayerCache> qcache_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<int, LayerState> layers_;
  std::set<int> done_;          // quantized (from progress)
  std::set<int> hess_on_disk_;  // complete Hessians on disk
  json progress_;
  // chains
  std::vector<int> chain_pos_;
  int chain_end_ = 0;           // chains must reach this layer (exclusive)
  std::set<int> need_collect_;
  std::vector<std::unique_ptr<uint16_t[]>> act_;
  std::vector<long> act_nseq_;
  std::atomic<int> failures_{0};
};

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
  for (auto& kv : progress_["layers"].items())
    if (kv.value().value("status", "") == "done" && fs::exists(layer_path(std::stoi(kv.key()))))
      done_.insert(std::stoi(kv.key()));
  for (auto& v : progress_["hessians"]) {
    int L = v.get<int>();
    if (fs::exists(hess_path(L))) hess_on_disk_.insert(L);
  }
  // quantize mode: accept any Hessian file present in hess_dir
  if (o_.mode == "quantize")
    for (int L = o_.layer_begin; L < o_.layer_end; ++L)
      if (fs::exists(hess_path(L))) hess_on_disk_.insert(L);
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

// --- Hessian file I/O (safetensors with F32 tensors qkv, o, gateup, down, metadata "count")
static void write_hess(const std::string& path, const std::vector<std::vector<float>>& H, const int dims[4],
                       long count) {
  SafeTensorsWriter w;
  for (int i = 0; i < 4; ++i)
    w.add(kHessName[i], "F32", {dims[i], dims[i]}, H[size_t(i)].data(), H[size_t(i)].size() * 4);
  w.write(path, {{"format", "pt"}, {"count", std::to_string(count)}, {"normalised", "1"}});
}

static std::vector<std::vector<float>> read_hess(const std::string& path) {
  // single-file safetensors reader: open via the directory form with a symlink-free trick
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
  static thread_local std::unique_ptr<HessAcc> tacc;
  if (collect) {
    if (!tacc) {
      tacc = std::make_unique<HessAcc>();
      R.alloc_hess(*tacc, qcfg_.is_full(L));
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
  logf("gpu%d qwen layer %d %s: %ld tokens in %.1fs (%.0f tok/s, %.1f TFLOPS), weights read %.1fs upload %.1fs",
       R.device(), L, collect ? "collect" : "forward", ntok, t_fwd, ntok / t_fwd, R.gemm_flops / t_fwd / 1e12,
       t_read, t_up);
  if (!o_.dump_acts.empty() && wi == 0) {
    fs::create_directories(o_.dump_acts);
    std::ofstream f(o_.dump_acts + "/act_after_layer" + std::to_string(L) + ".bf16", std::ios::binary);
    f.write(reinterpret_cast<const char*>(act_[0].get()),
            std::streamsize(size_t(o_.seq_len) * qcfg_.hidden * 2));
  }
  if (collect) deposit_partial_qwen(L, *acc, R);
}

void Pipeline::deposit_partial_qwen(int L, HessAcc& acc, QwenLayerRunner& R) {
  R.symmetrize_hess(acc);
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
  logf("qwen layer %d: Hessians complete (%ld tokens), saved in %.1fs", L, total, now_s() - t0);
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

void Pipeline::quantize_layer(int wi, LayerRunner* R, hipStream_t s, Gemm& g, int L) {
  (void)wi;
  (void)R;
  double t_layer = now_s();
  std::vector<std::vector<float>> H;
  {
    std::lock_guard<std::mutex> lk(mu_);
    H = std::move(layers_[L].H);
  }
  if (H.empty()) H = read_hess(hess_path(L));
  auto LH = cache_.get(L);
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  struct Grp {
    int hess;
    std::vector<int> lins;
  };
  std::vector<Grp> groups = {{0, {LQ, LK, LV}}, {1, {LO}}, {2, {LGate, LUp}}, {3, {LDown}}};
  std::vector<LinearOut> outs(kNumLin);
  json jl = json::object();
  // rotate all four Hessians on the GPU first, then factorise them on the CPU in a side thread (fp64 LDL of
  // group g+1 overlaps the GPU LDLQ of group g)
  std::vector<HessPrep> preps(groups.size());
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Grp& G = groups[gi];
    int n = LH->in_dim[G.lins[0]];
    QCHECK(long(H[size_t(G.hess)].size()) == long(n) * n, "Hessian size mismatch");
    uint64_t seed = (o_.seed * 1000003ull + uint64_t(L)) * 16 + uint64_t(G.hess);
    rotate_hessian(s, H[size_t(G.hess)].data(), n, seed, o_.q, preps[gi]);
    H[size_t(G.hess)].clear();
    H[size_t(G.hess)].shrink_to_fit();
  }
  std::mutex lmu;
  std::condition_variable lcv;
  int ldl_done = 0;
  std::thread ldl_th([&] {
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      ldl_hessian(preps[gi], o_.q.ldl_threads, "L" + std::to_string(L) + "_" + kHessName[groups[gi].hess]);
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
      in.w_fp8 = LH->w[li].data();
      in.scale_bf16 = LH->sc[li].data();
      ins.push_back(in);
    }
    double tw = now_s();
    {
      std::unique_lock<std::mutex> lk(lmu);
      lcv.wait(lk, [&] { return ldl_done > int(gi); });
    }
    double ldl_wait = now_s() - tw;
    GroupTiming tm;
    auto res = quantize_group(s, g, ins, preps[gi], o_.q, &tm, "L" + std::to_string(L));
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
  // ---- write layer file (QTIP hfized layout)
  SafeTensorsWriter w;
  std::string pre = "model.layers." + std::to_string(L) + ".";
  const Codebook& cb = codebook();
  for (int li = 0; li < kNumLin; ++li) {
    const LinearOut& r = outs[size_t(li)];
    std::string b = pre + kLinName[li] + ".";
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
  }
  w.add(pre + "input_layernorm.weight", "BF16", {cfg_.hidden}, LH->ln_in.data(), LH->ln_in.size() * 2);
  w.add(pre + "post_attention_layernorm.weight", "BF16", {cfg_.hidden}, LH->ln_post.data(), LH->ln_post.size() * 2);
  size_t bytes = w.write(layer_path(L));
  double secs = now_s() - t_layer;
  logf("gpu%d layer %d quantized in %.1fs -> %s (%.1f MB)", dev, L, secs, layer_path(L).c_str(), bytes / 1e6);
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

void Pipeline::quantize_qwen_layer(int wi, hipStream_t s, Gemm& g, int L) {
  (void)wi;
  double t_layer = now_s();
  std::vector<std::vector<float>> H;
  {
    std::lock_guard<std::mutex> lk(mu_);
    H = std::move(layers_[L].H);
  }
  if (H.empty()) H = read_hess(hess_path(L));
  auto QL = qcache_->get(L);
  QCHECK(!QL->moe, "layer " + std::to_string(L) + ": MoE quantization not implemented yet (Qwen3.5 dense only)");
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  const bool full = qcfg_.is_full(L);
  const int Hid = qcfg_.hidden;
  // linear index space of this layer type -> output tensor name, host fp32 weight and dims
  struct QLin {
    const char* name;
    int m = 0, n = 0;
    std::vector<float> w;  // fp32 rows of this linear [m][n]
  };
  auto rows16 = [&](const std::vector<uint16_t>& w, long r0, long nr, long n) {
    std::vector<float> o(size_t(nr) * n);
    for (long i = 0; i < nr * n; ++i) o[size_t(i)] = bf16_to_f32(w[size_t(r0) * n + size_t(i)]);
    return o;
  };
  std::vector<QLin> lins;
  auto add_lin = [&](const char* name, int m, int n, std::vector<float> w) {
    QLin q;
    q.name = name, q.m = m, q.n = n, q.w = std::move(w);
    lins.push_back(std::move(q));
  };
  if (full) {
    const int A = qcfg_.nq * qcfg_.hd;
    add_lin("self_attn.q_proj", qcfg_.nq * 2 * qcfg_.hd, Hid, bf16_to_f32_vec(QL->w_q));
    add_lin("self_attn.k_proj", qcfg_.nkv * qcfg_.hd, Hid, bf16_to_f32_vec(QL->w_k));
    add_lin("self_attn.v_proj", qcfg_.nkv * qcfg_.hd, Hid, bf16_to_f32_vec(QL->w_v));
    add_lin("self_attn.o_proj", Hid, A, bf16_to_f32_vec(QL->w_o));
    add_lin("mlp.gate_proj", qcfg_.inter, Hid, bf16_to_f32_vec(QL->w_gate));
    add_lin("mlp.up_proj", qcfg_.inter, Hid, bf16_to_f32_vec(QL->w_up));
    add_lin("mlp.down_proj", Hid, qcfg_.inter, bf16_to_f32_vec(QL->w_down));
  } else {
    const long nv = qcfg_.lin_nv, V = qcfg_.value_dim();
    add_lin("linear_attn.in_proj_qkv", qcfg_.conv_dim(), Hid, bf16_to_f32_vec(QL->w_in_qkv));
    add_lin("linear_attn.in_proj_z", int(V), Hid, bf16_to_f32_vec(QL->w_in_z));
    add_lin("linear_attn.in_proj_b", int(nv), Hid, rows16(QL->w_in_ba, 0, nv, Hid));
    add_lin("linear_attn.in_proj_a", int(nv), Hid, rows16(QL->w_in_ba, nv, nv, Hid));
    add_lin("linear_attn.out_proj", Hid, int(V), bf16_to_f32_vec(QL->w_out));
    add_lin("mlp.gate_proj", qcfg_.inter, Hid, bf16_to_f32_vec(QL->w_gate));
    add_lin("mlp.up_proj", qcfg_.inter, Hid, bf16_to_f32_vec(QL->w_up));
    add_lin("mlp.down_proj", Hid, qcfg_.inter, bf16_to_f32_vec(QL->w_down));
  }
  struct Grp {
    int hess;
    std::vector<int> lins;
  };
  std::vector<Grp> groups = full ? std::vector<Grp>{{0, {0, 1, 2}}, {1, {3}}, {2, {4, 5}}, {3, {6}}}
                                 : std::vector<Grp>{{0, {0, 1, 2, 3}}, {1, {4}}, {2, {5, 6}}, {3, {7}}};
  std::vector<LinearOut> outs(lins.size());
  json jl = json::object();
  std::vector<HessPrep> preps(groups.size());
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Grp& G = groups[gi];
    int n = qcfg_.hidden;
    if (G.hess == 1) n = full ? qcfg_.nq * qcfg_.hd : qcfg_.value_dim();
    else if (G.hess == 3) n = qcfg_.inter;
    QCHECK(long(H[size_t(G.hess)].size()) == long(n) * n, "Hessian size mismatch");
    uint64_t seed = (o_.seed * 1000003ull + uint64_t(L)) * 16 + uint64_t(G.hess);
    rotate_hessian(s, H[size_t(G.hess)].data(), n, seed, o_.q, preps[gi]);
    H[size_t(G.hess)].clear();
    H[size_t(G.hess)].shrink_to_fit();
  }
  std::mutex lmu;
  std::condition_variable lcv;
  int ldl_done = 0;
  std::thread ldl_th([&] {
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      ldl_hessian(preps[gi], o_.q.ldl_threads, "L" + std::to_string(L) + "_" + kHessName[groups[gi].hess]);
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
    logf("gpu%d qwen layer %d group %s: total %.1fs (load %.1f rht %.1f ldl_cpu %.1f ldlq %.1f [viterbi %.1f] "
         "proxy %.1f pack %.1f)",
         dev, L, kHessName[G.hess], tm.total, tm.load, tm.rht, tm.ldl_cpu, tm.ldlq, tm.viterbi, tm.proxy, tm.pack);
    for (size_t k = 0; k < res.size(); ++k) {
      const LinearOut& r = res[k];
      logf("  qwen layer %d %-24s %5d x %5d  %.1fs  proxy %.6f  proxy_rot_reg %.6f  Wscale %.6g", L, r.name.c_str(),
           r.m, r.n, r.seconds, r.proxy, r.proxy_rot_reg, r.wscale);
      jl[r.name] = {{"m", r.m}, {"n", r.n}, {"seconds", r.seconds}, {"proxy", r.proxy},
                    {"proxy_rot_reg", r.proxy_rot_reg}, {"wscale", r.wscale}};
      outs[size_t(G.lins[k])] = res[k];
    }
  }
  ldl_th.join();
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
    w.add(pre + "linear_attn.A_log", "F32", {qcfg_.lin_nv}, QL->a_log.data(), QL->a_log.size() * 4);
    w.add(pre + "linear_attn.dt_bias", "F32", {qcfg_.lin_nv}, QL->dt_bias.data(), QL->dt_bias.size() * 4);
    w.add(pre + "linear_attn.norm.weight", "BF16", {qcfg_.lin_vd}, QL->dn_norm.data(), QL->dn_norm.size() * 2);
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
  for (;;) {
    int task = -1;
    bool is_quant = false, collect = false;
    {
      std::unique_lock<std::mutex> lk(mu_);
      for (;;) {
        // 1) a ready layer to quantize
        if (o_.mode != "collect")
          for (auto& kv : layers_)
            if (kv.second.st == LayerState::kReady && kv.first >= o_.layer_begin && kv.first < o_.layer_end) {
              task = kv.first;
              is_quant = true;
              kv.second.st = LayerState::kQuantizing;
              break;
            }
        if (task >= 0) break;
        // 2) next chain layer
        if (need_chain && chain_pos_[size_t(wi)] < chain_end_) {
          int L = chain_pos_[size_t(wi)];
          collect = need_collect_.count(L) != 0;
          if (collect) {
            int backlog = 0;
            for (auto& kv : layers_)
              if (kv.second.st == LayerState::kPartial || kv.second.st == LayerState::kReady ||
                  kv.second.st == LayerState::kQuantizing)
                backlog++;
            bool started = layers_.count(L) && layers_[L].st == LayerState::kPartial;
            if (o_.mode == "collect" || started || backlog < o_.backlog) {
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
          if (QR) {
            QR.reset();
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
      if (qwen_) quantize_qwen_layer(wi, qs, qg, task);
      else quantize_layer(wi, R.get(), qs, qg, task);
    } else {
      if (qwen_) chain_step_qwen(wi, *QR, task, collect);
      else chain_step(wi, *R, task, collect);
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
    // embeddings
    std::string embed_name = "model.embed_tokens.weight";
    if (qwen_) {
      for (auto& kv : st_.tensors())
        if (kv.first.size() > 19 && kv.first.substr(kv.first.size() - 19) == "embed_tokens.weight")
          embed_name = kv.first;
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
  json c = qwen_ ? json::parse(qcfg_.raw_json) : json::parse(cfg_.raw_json);
  c.erase("quantization_config");
  if (qwen_) {
    // keep the original qwen3_5 arch/layer_types; the runtime reads the qwen fields itself
    c["qtip_quant"] = {{"sigma_reg", o_.q.sigma_reg}, {"scale_override", o_.q.scale_override},
                       {"ntok", o_.ntok},             {"seq_len", o_.seq_len},
                       {"shared_su_per_hessian_group", true}};
  } else {
    c["architectures"] = {"LlamaForCausalLM"};
    c["model_type"] = "llama";
    c["torch_dtype"] = "bfloat16";
    c["attention_bias"] = false;
    c["mlp_bias"] = false;
    c["pretraining_tp"] = 1;
    c["rope_scaling"] = nullptr;
  }
  c["quip_params"] = {{"codebook", "bitshift"}, {"codebook_version", 0}, {"L", kL},      {"K", kK},
                      {"V", kV},                {"tlut_bits", kTlutBits}, {"decode_mode", "quantlut_sym"},
                      {"td_x", kTd},            {"td_y", kTd},           {"split_for_tp", false},
                      {"skip_list", nullptr}};
  if (!qwen_) {
    c["qtip_quant"] = {{"sigma_reg", o_.q.sigma_reg}, {"scale_override", o_.q.scale_override},
                       {"ntok", o_.ntok},             {"seq_len", o_.seq_len},
                       {"shared_su_per_hessian_group", true}};
  }
  {
    std::ofstream f(o_.out + "/config.json");
    f << c.dump(2);
  }
  std::string ef = o_.out + "/model-embed.safetensors";
  if (!fs::exists(ef)) {
    SafeTensorsWriter w;
    if (qwen_) {
      std::string norm_name = "model.norm.weight", embed_name = "model.embed_tokens.weight";
      for (auto& kv : st_.tensors()) {
        if (kv.first.size() >= 11 && kv.first.substr(kv.first.size() - 11) == "norm.weight" &&
            kv.first.find("layers.") == std::string::npos)
          norm_name = kv.first;
        if (kv.first.size() > 19 && kv.first.substr(kv.first.size() - 19) == "embed_tokens.weight")
          embed_name = kv.first;
      }
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
  for (std::string fn : {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                         "generation_config.json", "vocab.json", "merges.txt", "tokenizer.model"}) {
    fs::path src = fs::path(o_.model) / fn;
    if (fs::exists(src)) fs::copy_file(src, fs::path(o_.out) / fn, fs::copy_options::overwrite_existing);
  }
  logf("assembled %s: %zu tensors, %.2f GB", o_.out.c_str(), wm.size(), total / 1e9);
}

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  Pipeline p(o);
  if (o.mode == "assemble") p.assemble();
  else p.run();
  return 0;
}
