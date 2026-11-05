// Adaptive mixed-format mode tests: a synthetic 3-layer FP8 "Behemoth" checkpoint is quantized with
// --mode adaptive under thresholds that force each branch (all-QTIP, all-NVFP4, MSE metric,
// quantized-input chaining). GPU required (LayerRunner forward + QTIP kernels).  usage:
// qtip_adaptive_tests [--device N]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <omp.h>

#include "common.hpp"
#include "nvfp4.hpp"
#include "pipeline.hpp"
#include "safetensors.hpp"

using namespace qtip;
namespace fs = std::filesystem;
using json = nlohmann::json;

static int g_fail = 0;
#define EXPECT(c, ...)                                                          \
  do {                                                                          \
    if (!(c)) {                                                                 \
      std::printf("  FAIL(%s:%d): ", fs::path(__FILE__).filename().c_str(), __LINE__); \
      std::printf(__VA_ARGS__); \
      std::printf("\n");                                                        \
      ++g_fail;                                                                 \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------- tiny synthetic FP8 checkpoint
// hidden 128, inter 256, 8 q heads / 2 kv heads, head_dim 16, 3 layers, vocab 64.
static constexpr int kH = 128, kF = 256, kNQ = 8, kNKV = 2, kHD = 16, kLayers = 3, kVocab = 64, kTSeq = 32;

struct TinyModel {
  fs::path dir;
  std::vector<std::vector<std::vector<uint8_t>>> w;     // [layer][linear] fp8
  std::vector<std::vector<std::vector<uint16_t>>> sc;   // [layer][linear] bf16 scale bits
  int m[kNumLin], n[kNumLin];
};

static int out_dim(int li) {
  if (li == LQ) return kNQ * kHD;
  if (li == LK || li == LV) return kNKV * kHD;
  if (li == LO) return kH;
  if (li == LGate || li == LUp) return kF;
  return kH;
}
static int in_dim(int li) { return li == LDown ? kF : kH; }

static TinyModel build_model(const fs::path& dir) {
  fs::create_directories(dir);
  TinyModel M;
  M.dir = dir;
  {
    json c = {{"hidden_size", kH},        {"intermediate_size", kF},   {"num_attention_heads", kNQ},
              {"num_key_value_heads", kNKV}, {"head_dim", kHD},       {"num_hidden_layers", kLayers},
              {"vocab_size", kVocab},     {"rms_norm_eps", 1e-5},      {"rope_theta", 1000000.0}};
    std::ofstream f(dir / "config.json");
    f << c.dump(2);
  }
  SafeTensorsWriter w;
  Rng r(1234);
  for (int li = 0; li < kNumLin; ++li) {
    M.m[li] = out_dim(li);
    M.n[li] = in_dim(li);
  }
  M.w.resize(kLayers);
  M.sc.resize(kLayers);
  for (int L = 0; L < kLayers; ++L) {
    M.w[size_t(L)].resize(kNumLin);
    M.sc[size_t(L)].resize(kNumLin);
    for (int li = 0; li < kNumLin; ++li) {
      const long m = M.m[li], n = M.n[li];
      M.w[size_t(L)][size_t(li)].resize(size_t(m) * n);
      M.sc[size_t(L)][size_t(li)].resize(size_t(m));
      for (long i = 0; i < m; ++i) {
        float scale = float(std::exp(r.normal() * 0.3));
        M.sc[size_t(L)][size_t(li)][size_t(i)] = f32_to_bf16(scale);
        for (long j = 0; j < n; ++j) M.w[size_t(L)][size_t(li)][size_t(i) * n + j] =
            f32_to_fp8e4m3(float(r.normal() * 8.0f));  // values in E4M3 range
      }
      std::string b = "model.layers." + std::to_string(L) + "." + kLinName[li] + ".";
      w.add(b + "weight", "F8_E4M3", {m, n}, M.w[size_t(L)][size_t(li)].data(), size_t(m) * n);
      w.add(b + "weight_scale", "BF16", {m}, M.sc[size_t(L)][size_t(li)].data(), size_t(m) * 2);
    }
    std::string p = "model.layers." + std::to_string(L) + ".";
    std::vector<uint16_t> ln{kH};
    for (auto& v : ln) v = f32_to_bf16(float(1.0 + 0.05 * r.normal()));
    w.add(p + "input_layernorm.weight", "BF16", {kH}, ln.data(), ln.size() * 2);
    std::vector<uint16_t> ln2{kH};
    for (auto& v : ln2) v = f32_to_bf16(float(1.0 + 0.05 * r.normal()));
    w.add(p + "post_attention_layernorm.weight", "BF16", {kH}, ln2.data(), ln2.size() * 2);
  }
  std::vector<uint16_t> emb(size_t(kVocab) * kH);
  for (auto& v : emb) v = f32_to_bf16(float(r.normal() * 0.5));
  w.add("model.embed_tokens.weight", "BF16", {kVocab, kH}, emb.data(), emb.size() * 2);
  std::vector<uint16_t> nrm{kH};
  for (auto& v : nrm) v = f32_to_bf16(float(1.0 + 0.05 * r.normal()));
  w.add("model.norm.weight", "BF16", {kH}, nrm.data(), nrm.size() * 2);
  std::vector<uint16_t> head(size_t(kVocab) * kH);
  for (auto& v : head) v = f32_to_bf16(float(r.normal() * 0.2));
  w.add("lm_head.weight", "BF16", {kVocab, kH}, head.data(), head.size() * 2);
  w.write((dir / "model.safetensors").string());
  return M;
}

static void write_tokens(const fs::path& path, int nseq) {
  Rng r(77);
  std::vector<int32_t> tok(size_t(nseq) * kTSeq);
  for (auto& t : tok) t = int(r.next() % kVocab);
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(tok.data()), std::streamsize(tok.size() * 4));
}

// minimal standalone safetensors reader for one layer file
struct LayerFile {
  json header;
  std::ifstream f;
  bool has(const std::string& n) const { return header.contains(n); }
  std::string dtype(const std::string& n) const { return header[n]["dtype"]; }
  std::vector<int64_t> shape(const std::string& n) const { return header[n]["shape"].get<std::vector<int64_t>>(); }
  std::vector<uint8_t> bytes(const std::string& n) {
    auto off = header[n]["data_offsets"].get<std::vector<uint64_t>>();
    std::vector<uint8_t> b(size_t(off[1] - off[0]));
    f.seekg(std::streamoff(8 + hlen + off[0]));
    f.read(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
    return b;
  }
  uint64_t hlen = 0;
};
static LayerFile open_layer(const fs::path& out, int L) {
  char b[64];
  std::snprintf(b, sizeof b, "model-layer-%05d.safetensors", L);
  LayerFile lf;
  lf.f.open(out / b, std::ios::binary);
  EXPECT(bool(lf.f), "open layer %d", L);
  lf.f.read(reinterpret_cast<char*>(&lf.hlen), 8);
  std::string h(lf.hlen, '\0');
  lf.f.read(h.data(), std::streamsize(lf.hlen));
  lf.header = json::parse(h);
  return lf;
}

static Options base_options(const std::string& tag, double threshold, const char* metric,
                            const char* next_input) {
  Options o;
  o.mode = "adaptive";
  o.model = (fs::path("/tmp/opencode") / "adaptive-test" / tag / "model").string();
  o.tokens = (fs::path("/tmp/opencode") / "adaptive-test" / tag / "tokens.i32").string();
  o.out = (fs::path("/tmp/opencode") / "adaptive-test" / tag / "out").string();
  o.work = (fs::path("/tmp/opencode") / "adaptive-test" / tag / "work").string();
  o.gpus = {0};
  o.ntok = 8 * kTSeq;
  o.seq_len = kTSeq;
  o.batch_rows = kTSeq;
  o.kl_threshold = threshold;
  o.kl_metric = metric;
  o.kl_tokens = kTSeq;
  o.adaptive_next_input = next_input;
  return o;
}

static json run_case(const std::string& tag, double threshold, const char* metric, const char* next_input,
                     bool cache_teacher = true) {
  fs::path root = fs::path("/tmp/opencode") / "adaptive-test" / tag;
  fs::remove_all(root);
  fs::create_directories(root);
  auto M = build_model(root / "model");
  write_tokens(root / "tokens.i32", 8);
  Options o = base_options(tag, threshold, metric, next_input);
  o.cache_teacher_logits = cache_teacher;
  {
    Pipeline p(o);  // logs into the work dir
    p.run();
  }
  // every layer written + assembled
  for (int L = 0; L < kLayers; ++L) {
    char b[64];
    std::snprintf(b, sizeof b, "model-layer-%05d.safetensors", L);
    EXPECT(fs::exists(fs::path(o.out) / b), "%s layer %d file", tag.c_str(), L);
  }
  EXPECT(fs::exists(fs::path(o.out) / "adaptive_report.json"), "%s report", tag.c_str());
  EXPECT(fs::exists(fs::path(o.out) / "config.json"), "%s config", tag.c_str());
  EXPECT(fs::exists(fs::path(o.out) / "model.safetensors.index.json"), "%s index", tag.c_str());
  json report = json::parse(std::ifstream(fs::path(o.out) / "adaptive_report.json"));
  json config = json::parse(std::ifstream(fs::path(o.out) / "config.json"));
  const bool force_nv = threshold == 0.0;
  EXPECT(report["layers"].size() == kLayers, "%s report layer count", tag.c_str());
  EXPECT(config["qtip_quant"].contains("layer_formats"), "%s config layer_formats", tag.c_str());
  auto fmts = config["qtip_quant"]["layer_formats"];
  EXPECT(fmts.size() == kLayers, "%s layer_formats count", tag.c_str());
  for (int L = 0; L < kLayers; ++L) {
    auto lf = open_layer(o.out, L);
    std::string pre = "model.layers." + std::to_string(L) + ".";
    const char* want = force_nv ? "nvfp4" : "qtip";
    EXPECT(report["layers"][size_t(L)]["format"] == want, "%s L%d format %s (report)", tag.c_str(), L,
           report["layers"][size_t(L)]["format"].dump().c_str());
    EXPECT(fmts[size_t(L)] == want, "%s L%d format (config)", tag.c_str(), L);
    for (int li = 0; li < kNumLin; ++li) {
      std::string b = pre + kLinName[li] + ".";
      bool preserved = li == LK || li == LV;  // default --kv-format fp8
      if (preserved) {
        EXPECT(lf.dtype(b + "weight") == "F8_E4M3", "%s L%d %s fp8 weight kept", tag.c_str(), L, kLinName[li]);
        EXPECT(lf.dtype(b + "weight_scale") == "BF16", "%s L%d %s fp8 scale kept", tag.c_str(), L, kLinName[li]);
        EXPECT(!lf.has(b + "trellis") && !lf.has(b + "weight_packed"), "%s L%d %s no quant tensors", tag.c_str(), L,
               kLinName[li]);
        continue;
      }
      if (force_nv) {
        EXPECT(lf.has(b + "weight_packed") && lf.dtype(b + "weight_packed") == "U8", "%s L%d %s packed", tag.c_str(),
               L, kLinName[li]);
        EXPECT(lf.shape(b + "weight_packed")[0] == M.m[li] && lf.shape(b + "weight_packed")[1] == M.n[li] / 2,
               "%s L%d %s packed shape", tag.c_str(), L, kLinName[li]);
        EXPECT(lf.dtype(b + "weight_scale") == "F8_E4M3" && lf.shape(b + "weight_scale")[0] == M.m[li] &&
                   lf.shape(b + "weight_scale")[1] == M.n[li] / 16,
               "%s L%d %s block scales", tag.c_str(), L, kLinName[li]);
        EXPECT(lf.has(b + "weight_global_scale") && lf.dtype(b + "weight_global_scale") == "F32",
               "%s L%d %s global scale", tag.c_str(), L, kLinName[li]);
      } else {
        EXPECT(lf.has(b + "trellis") && lf.dtype(b + "trellis") == "I16", "%s L%d %s trellis", tag.c_str(), L,
               kLinName[li]);
        EXPECT(lf.has(b + "SU") && lf.has(b + "SV") && lf.has(b + "tlut"), "%s L%d %s qtip vectors", tag.c_str(), L,
               kLinName[li]);
      }
    }
    // metric values are logged for every layer; NVFP4 candidates additionally log the fallback metric
    const json& adv = report["layers"][size_t(L)]["adaptive"];
    std::string mkey = std::string(metric) == "mse" ? "mse_qtip" : "kl_qtip";
    EXPECT(adv.contains(mkey) && adv[mkey].is_number(), "%s L%d %s logged", tag.c_str(), L, mkey.c_str());
    double mval = adv[mkey].get<double>();
    EXPECT(mval >= 0 && std::isfinite(mval), "%s L%d metric finite >= 0 (got %g)", tag.c_str(), L, mval);
    if (force_nv) {
      std::string nkey = std::string(metric) == "mse" ? "mse_nvfp4" : "kl_nvfp4";
      EXPECT(adv.contains(nkey) && adv[nkey].is_number(), "%s L%d %s logged", tag.c_str(), L, nkey.c_str());
    }
  }
  // NVFP4 decode fidelity: layer 0 o_proj vs the original FP8-decoded weights
  if (force_nv) {
    auto lf = open_layer(o.out, 0);
    std::string b = "model.layers.0.self_attn.o_proj.";
    Nvfp4Tensor t;
    t.m = M.m[LO], t.n = M.n[LO];
    t.packed = lf.bytes(b + "weight_packed");
    t.scales = lf.bytes(b + "weight_scale");
    std::vector<uint8_t> gs = lf.bytes(b + "weight_global_scale");
    std::memcpy(&t.global, gs.data(), 4);
    EXPECT(t.global > 0, "global scale positive");
    std::vector<float> rec = nvfp4_dequantize(t);
    double num = 0, den = 0;
    for (long i = 0; i < t.m; ++i) {
      float sc = bf16_to_f32(M.sc[0][size_t(LO)][size_t(i)]);
      for (long j = 0; j < t.n; ++j) {
        double o = fp8e4m3_to_f32(M.w[0][size_t(LO)][size_t(i) * t.n + j]) * sc;
        num += (rec[size_t(i) * t.n + j] - o) * (rec[size_t(i) * t.n + j] - o);
        den += o * o;
      }
    }
    double rel = std::sqrt(num / den);
    std::printf("  %s: NVFP4(o_proj) relative reconstruction error %.4f\n", tag.c_str(), rel);
    EXPECT(rel < 0.3, "%s NVFP4 reconstruction too lossy (%g)", tag.c_str(), rel);
  }
  return report;
}

// pure host checks of the NVFP4 codec
static void test_nvfp4_codec() {
  std::printf("[nvfp4 codec] round trip and geometry\n");
  Rng r(5);
  const int m = 8, n = 48;
  std::vector<float> w(size_t(m) * n);
  for (auto& v : w) v = float(r.normal() * 0.05);
  auto q = nvfp4_quantize(w.data(), m, n);
  EXPECT(q.packed.size() == size_t(m) * n / 2 && q.scales.size() == size_t(m) * n / 16, "codec geometry");
  auto back = nvfp4_dequantize(q);
  double num = 0, den = 0;
  for (size_t i = 0; i < w.size(); ++i) {
    num += (back[i] - w[i]) * (back[i] - w[i]);
    den += w[i] * w[i];
  }
  double rel = std::sqrt(num / den);
  std::printf("  round-trip relative error %.4f (global scale %.4g)\n", rel, q.global);
  EXPECT(rel < 0.15, "nvfp4 round trip too lossy");
  EXPECT(nvfp4_level(0) == 0 && nvfp4_level(7) == 6 && nvfp4_level(8 | 3) == -1.5, "e2m1 levels");
  std::vector<float> large(1u << 20);
  for (size_t i = 0; i < large.size(); ++i) large[i] = i % 128 < 16 ? 0.f : float(r.normal() * 0.1);
  int saved_threads = omp_get_max_threads();
  omp_set_num_threads(1);
  auto serial = nvfp4_quantize(large.data(), 1024, 1024);
  auto serial_decoded = nvfp4_dequantize(serial);
  omp_set_num_threads(8);
  auto parallel = nvfp4_quantize(large.data(), 1024, 1024);
  auto parallel_decoded = nvfp4_dequantize(parallel);
  omp_set_num_threads(saved_threads);
  EXPECT(serial.global == parallel.global && serial.packed == parallel.packed && serial.scales == parallel.scales,
         "NVFP4 encoding changes with thread count");
  EXPECT(serial_decoded == parallel_decoded, "NVFP4 decoding changes with thread count");
}

int main(int argc, char** argv) {
  int dev = 0;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc) dev = std::atoi(argv[++i]);
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::puts("no GPU device; skipped");
    return 77;
  }
  HIP_CHECK(hipSetDevice(dev));
  fs::remove_all("/tmp/opencode/adaptive-test");
  test_nvfp4_codec();
  // all-QTIP: threshold so large the metric never exceeds it
  run_case("qtip", 1e30, "lmkl", "original");
  // all-NVFP4: threshold 0 and a strictly positive KL forces every layer to fall back
  auto cached = run_case("nvfp4", 0.0, "lmkl", "original");
  auto uncached = run_case("nvfp4-uncached", 0.0, "lmkl", "original", false);
  auto compare_metrics = [&](const json& a, const json& b) {
    for (int layer = 0; layer < kLayers; ++layer)
      for (const char* key : {"kl_qtip", "kl_nvfp4"}) {
        double x = a["layers"][size_t(layer)]["adaptive"][key].get<double>();
        double y = b["layers"][size_t(layer)]["adaptive"][key].get<double>();
        EXPECT(std::abs(x - y) < 1e-10, "teacher cache changed L%d %s", layer, key);
      }
  };
  compare_metrics(cached, uncached);
  auto progressive = run_case("lmkl-quantized", 0.0, "lmkl", "quantized");
  auto progressive_uncached = run_case("lmkl-quantized-uncached", 0.0, "lmkl", "quantized", false);
  compare_metrics(progressive, progressive_uncached);
  // MSE metric + quantized-input chaining (probe and chain flow through the chosen layers)
  run_case("mse-quantized", 0.0, "mse", "quantized");
  std::printf(g_fail ? "ADAPTIVE TESTS FAILED (%d)\n" : "ALL ADAPTIVE TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
