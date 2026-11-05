// Llama 3.3 70B support tests for the quantizer: a synthetic BF16 checkpoint with Llama-3.3-scaled
// geometry (GQA 8:1, FFN 3.5x -> 28*2^p Hadamard factor, untied 'vocab' embeddings, rope_theta 5e5 +
// llama3 rope_scaling, two safetensors shards + index) is quantized in QTIP mode and in adaptive mode
// (all-QTIP, all-NVFP4, tiled lm_head). GPU required.
// usage: qtip_llama_tests [--device N] [--keep DIR]   (--keep leaves the quantized checkpoints in DIR
// for behemoth_cuda_llama_tests --checkpoint DIR/qtip, DIR/nvfp4)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/wait.h>
#include <unistd.h>

#include "common.hpp"
#include "fp8_kv.hpp"
#include "gpu.hpp"
#include "model.hpp"
#include "pipeline.hpp"
#include "qtip_core.hpp"
#include "safetensors.hpp"

using namespace qtip;
namespace fs = std::filesystem;
using json = nlohmann::json;

static int g_fail = 0;
#define EXPECT(c, ...)                                                                    \
  do {                                                                                    \
    if (!(c)) {                                                                           \
      std::printf("  FAIL(line %d): " , __LINE__);                                          \
      std::printf(__VA_ARGS__);                                                           \
      std::printf("\n");                                                                  \
      ++g_fail;                                                                           \
    }                                                                                     \
  } while (0)

// Llama 3.3 70B ratios at toy size: hidden 256 (=nq*hd), ffn 896 = 28*32, 8 q / 1 kv heads (64/8).
static constexpr int kH = 256, kF = 896, kNQ = 8, kNKV = 1, kHD = 32, kLayers = 3, kVocab = 320, kTSeq = 32;
// original_max_position_embeddings is scaled with the toy head_dim so all three llama3 bands occur
static const RopeScalingLlama3 kRope{8.0, 1.0, 4.0, 16};

static int out_dim(int li) {
  if (li == LQ) return kNQ * kHD;
  if (li == LK || li == LV) return kNKV * kHD;
  if (li == LGate || li == LUp) return kF;
  return kH;
}
static int in_dim(int li) { return li == LDown ? kF : kH; }

static json source_config() {
  return {{"architectures", {"LlamaForCausalLM"}},
          {"model_type", "llama"},
          {"hidden_size", kH},
          {"intermediate_size", kF},
          {"num_attention_heads", kNQ},
          {"num_key_value_heads", kNKV},
          {"head_dim", kHD},
          {"num_hidden_layers", kLayers},
          {"vocab_size", kVocab},
          {"hidden_act", "silu"},
          {"tie_word_embeddings", false},
          {"rms_norm_eps", 1e-5},
          {"rope_theta", 500000.0},
          {"max_position_embeddings", 131072},
          {"bos_token_id", 1},
          {"eos_token_id", {2, 3}},
          {"rope_scaling",
           {{"factor", kRope.factor},
            {"low_freq_factor", kRope.low_freq_factor},
            {"high_freq_factor", kRope.high_freq_factor},
            {"original_max_position_embeddings", kRope.original_max_position_embeddings},
            {"rope_type", "llama3"}}}};
}

// BF16 source (no weight_scale tensors), two shards + index like the HF release
static void build_source(const fs::path& dir) {
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::ofstream(dir / "config.json") << source_config().dump(2);
  std::ofstream(dir / "generation_config.json") << json{{"bos_token_id", 1}, {"eos_token_id", {2, 3}}}.dump();
  Rng r(4321);
  SafeTensorsWriter w[2];
  std::map<std::string, std::string> where;
  auto add = [&](int shard, const std::string& name, std::vector<int64_t> shape, const std::vector<uint16_t>& v) {
    w[shard].add(name, "BF16", shape, v.data(), v.size() * 2);
    where[name] = shard ? "model-00002-of-00002.safetensors" : "model-00001-of-00002.safetensors";
  };
  for (int L = 0; L < kLayers; ++L) {
    const std::string p = "model.layers." + std::to_string(L) + ".";
    for (int li = 0; li < kNumLin; ++li) {
      const long m = out_dim(li), n = in_dim(li);
      std::vector<uint16_t> v(size_t(m) * n);
      for (auto& x : v) x = f32_to_bf16(float(r.normal() / std::sqrt(double(n)) * (li <= LK ? 3.0 : 1.0)));
      add(L % 2, p + kLinName[li] + ".weight", {m, n}, v);
    }
    for (const char* nm : {"input_layernorm.weight", "post_attention_layernorm.weight"}) {
      std::vector<uint16_t> v(kH);
      for (auto& x : v) x = f32_to_bf16(float(1.0 + 0.05 * r.normal()));
      add(L % 2, p + nm, {kH}, v);
    }
  }
  std::vector<uint16_t> emb(size_t(kVocab) * kH), head(emb.size()), norm(kH);
  for (auto& v : emb) v = f32_to_bf16(float(r.normal() * 0.5));
  for (auto& v : head) v = f32_to_bf16(float(r.normal() * 0.2));
  for (auto& v : norm) v = f32_to_bf16(float(1.0 + 0.05 * r.normal()));
  add(1, "model.embed_tokens.weight", {kVocab, kH}, emb);
  add(1, "lm_head.weight", {kVocab, kH}, head);
  add(1, "model.norm.weight", {kH}, norm);
  w[0].write((dir / "model-00001-of-00002.safetensors").string());
  w[1].write((dir / "model-00002-of-00002.safetensors").string());
  json wm = json::object();
  for (auto& [k, v] : where) wm[k] = v;
  std::ofstream(dir / "model.safetensors.index.json") << json{{"metadata", json::object()}, {"weight_map", wm}}.dump();
  std::vector<int32_t> tok(size_t(8) * kTSeq);
  for (auto& t : tok) t = int32_t(r.next() % kVocab);
  std::ofstream f(dir / "tokens.i32", std::ios::binary);
  f.write(reinterpret_cast<const char*>(tok.data()), std::streamsize(tok.size() * 4));
}

static Options base_options(const fs::path& root, const std::string& tag, const char* mode) {
  Options o;
  o.mode = mode;
  o.model = (root / "source").string();
  o.tokens = (root / "source" / "tokens.i32").string();
  o.out = (root / tag).string();
  o.work = (root / (tag + "-work")).string();
  o.gpus = {0};
  o.ntok = 8 * kTSeq;
  o.seq_len = kTSeq;
  o.batch_rows = kTSeq;
  o.kl_tokens = kTSeq;
  return o;
}

struct LayerFile {
  json header;
  std::ifstream f;
  uint64_t hlen = 0;
  bool has(const std::string& n) const { return header.contains(n); }
  std::string dtype(const std::string& n) const { return header[n]["dtype"]; }
  std::vector<int64_t> shape(const std::string& n) const { return header[n]["shape"].get<std::vector<int64_t>>(); }
  std::vector<uint8_t> bytes(const std::string& n) {
    auto off = header[n]["data_offsets"].get<std::vector<uint64_t>>();
    std::vector<uint8_t> v(off[1] - off[0]);
    f.clear();
    f.seekg(std::streamoff(8 + hlen + off[0]));
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size()));
    return v;
  }
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

static void check_output(const fs::path& out, const std::string& tag, bool nvfp4_all, const std::vector<bool>* per_layer_nvfp4 = nullptr,
                         bool fp8_kv = false, const fs::path& source = {}) {
  json config = json::parse(std::ifstream(out / "config.json"));
  EXPECT(config["model_type"] == "llama" && config["quip_params"]["K"] == 2, "%s config", tag.c_str());
  EXPECT(config["rope_scaling"].is_object() && config["rope_scaling"]["rope_type"] == "llama3" &&
             config["rope_scaling"]["factor"] == 8.0 &&
             config["rope_scaling"]["original_max_position_embeddings"] == kRope.original_max_position_embeddings,
         "%s rope_scaling must be preserved in the assembled config", tag.c_str());
  EXPECT(config["qtip_quant"]["kv_format"] == (fp8_kv ? "fp8" : "qtip"), "%s kv_format (BF16 sources default to qtip)", tag.c_str());
  EXPECT(config["rope_theta"] == 500000.0 && config["vocab_size"] == kVocab, "%s config geometry", tag.c_str());
  EXPECT(fs::exists(out / "generation_config.json") && fs::exists(out / "model.safetensors.index.json") &&
             fs::exists(out / "model-embed.safetensors"),
         "%s assembled files", tag.c_str());
  for (int L = 0; L < kLayers; ++L) {
    const bool nvfp4 = per_layer_nvfp4 ? (*per_layer_nvfp4)[size_t(L)] : nvfp4_all;
    auto lf = open_layer(out, L);
    const std::string pre = "model.layers." + std::to_string(L) + ".";
    for (int li = 0; li < kNumLin; ++li) {
      const std::string b = pre + kLinName[li] + ".";
      if (fp8_kv && (li == LK || li == LV)) {
        EXPECT(lf.has(b + "weight") && lf.dtype(b + "weight") == "F8_E4M3" && lf.shape(b + "weight")[0] == out_dim(li) &&
                   lf.shape(b + "weight")[1] == in_dim(li) && lf.has(b + "weight_scale") &&
                   lf.dtype(b + "weight_scale") == "BF16" && lf.shape(b + "weight_scale").size() == 1 &&
                   lf.shape(b + "weight_scale")[0] == out_dim(li) && !lf.has(b + "trellis") && !lf.has(b + "weight_packed"),
               "%s L%d %s: FP8 K/V layout", tag.c_str(), L, kLinName[li]);
        // bytes must equal the converter applied to the bf16 source tensor
        SafeTensorsReader st(source.string());
        auto src = st.read(b + "weight");
        Fp8Linear f = bf16_to_fp8_dynamic(reinterpret_cast<const uint16_t*>(src.data()), out_dim(li), in_dim(li));
        auto w8 = lf.bytes(b + "weight"), sc = lf.bytes(b + "weight_scale");
        EXPECT(w8 == f.w && sc.size() == f.scale.size() * 2 && !std::memcmp(sc.data(), f.scale.data(), sc.size()),
               "%s L%d %s: FP8 bytes differ from converter", tag.c_str(), L, kLinName[li]);
        continue;
      }
      EXPECT(!lf.has(b + "weight"), "%s L%d %s: K/V must not stay FP8/bf16 for BF16 sources", tag.c_str(), L, kLinName[li]);
      if (nvfp4) {
        EXPECT(lf.has(b + "weight_packed") && lf.shape(b + "weight_packed")[0] == out_dim(li) &&
                   lf.shape(b + "weight_packed")[1] == in_dim(li) / 2,
               "%s L%d %s packed", tag.c_str(), L, kLinName[li]);
      } else {
        EXPECT(lf.has(b + "trellis") && lf.dtype(b + "trellis") == "I16" &&
                   lf.shape(b + "trellis")[0] == int64_t(out_dim(li) / 16) * (in_dim(li) / 16) &&
                   lf.has(b + "SU") && lf.has(b + "SV") && lf.has(b + "tlut") &&
                   lf.shape(b + "SU")[0] == in_dim(li) && lf.shape(b + "SV")[0] == out_dim(li),
               "%s L%d %s trellis/SU/SV geometry", tag.c_str(), L, kLinName[li]);
      }
    }
  }
}

// ------------------------------------------------------------------ unit tests
static void test_config() {
  std::printf("[config] Llama 3.3 geometry from config.json, unsupported shapes rejected\n");
  fs::path dir = fs::temp_directory_path() / "qtip_llama_config";
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto load = [&](json c) {
    std::ofstream(dir / "config.json") << c.dump();
    return load_model_config(dir.string());
  };
  json real = {{"hidden_size", 8192}, {"intermediate_size", 28672}, {"num_attention_heads", 64},
               {"num_key_value_heads", 8}, {"num_hidden_layers", 80}, {"vocab_size", 128256},
               {"rms_norm_eps", 1e-5}, {"rope_theta", 500000.0}, {"max_position_embeddings", 131072},
               {"rope_scaling", {{"factor", 8.0}, {"low_freq_factor", 1.0}, {"high_freq_factor", 4.0},
                                 {"original_max_position_embeddings", 8192}, {"rope_type", "llama3"}}}};
  ModelConfig c = load(real);
  EXPECT(c.hidden == 8192 && c.inter == 28672 && c.nq == 64 && c.nkv == 8 && c.hd == 128 && c.layers == 80 &&
             c.vocab == 128256 && c.rope_llama3 && c.rope.original_max_position_embeddings == 8192 &&
             c.theta == 500000.f,
         "Llama 3.3 70B config parsed");
  for (int n : {8192, 28672, 1024}) EXPECT(had_K(n) > 0 && n / had_K(n) <= 8192, "hadamard factorisation of %d", n);
  EXPECT(had_K(8192) == 1 && had_K(28672) == 28 && had_K(1024) == 1, "8192=2^13, 28672=28*1024");
  ModelConfig b = load({{"hidden_size", 12288}, {"intermediate_size", 28672}, {"num_attention_heads", 96},
                        {"num_key_value_heads", 8}, {"head_dim", 128}, {"num_hidden_layers", 88},
                        {"vocab_size", 32768}, {"rope_theta", 1000000.0}});
  EXPECT(!b.rope_llama3 && b.theta == 1e6f && b.hd == 128, "Behemoth config unchanged (plain RoPE)");
  // QCHECK aborts, so the rejection paths are exercised in a forked child each
  auto rejects = [&](const char* what, json c) {
    std::ofstream(dir / "config.json") << c.dump();
    pid_t pid = fork();
    if (pid == 0) {
      if (!freopen("/dev/null", "w", stderr)) _exit(2);
      load_model_config(dir.string());
      _exit(0);  // accepted: wrong
    }
    int st = 0;
    waitpid(pid, &st, 0);
    EXPECT(!(WIFEXITED(st) && WEXITSTATUS(st) == 0), "must reject: %s", what);
  };
  json y = real, odd = real, big = real, kv = real, gqa = real;
  y["rope_scaling"] = {{"rope_type", "yarn"}, {"factor", 4.0}};
  odd["hidden_size"] = 8200;
  big["intermediate_size"] = 28 * 16384;
  kv["num_key_value_heads"] = 7;
  gqa["head_dim"] = 33;
  rejects("rope_scaling yarn", y);
  rejects("hidden 8200 (no Hadamard factor)", odd);
  rejects("Hadamard pow2 part > 8192", big);
  rejects("q heads not a multiple of kv heads", kv);
  rejects("odd head_dim", gqa);
}

static void test_hadamard_8192() {
  std::printf("[hadamard] pow2 part 8192 (Llama 3.3 hidden)\n");
  for (int n : {8192, 28672}) {
    const long rows = 3;
    Rng r(n);
    std::vector<float> x(size_t(rows) * n);
    for (auto& v : x) v = float(r.normal());
    for (int tr = 0; tr < 2; ++tr) {
      std::vector<double> ref(x.begin(), x.end());
      hadamard_rows_ref(ref.data(), rows, n, n, tr == 1);
      DBuf<float> d(x.size());
      HIP_CHECK(hipMemcpy(d, x.data(), x.size() * 4, hipMemcpyHostToDevice));
      k_hadamard_rows(d, rows, n, n, tr == 1, nullptr);
      std::vector<float> y(x.size());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(y.data(), d, y.size() * 4, hipMemcpyDeviceToHost));
      double md = 0;
      for (size_t i = 0; i < y.size(); ++i) md = std::max(md, std::fabs(y[i] - ref[i]));
      std::printf("  n=%d %s max|diff|=%.3g\n", n, tr ? "hadUt" : "hadU", md);
      EXPECT(md < 2e-5, "hadamard n=%d", n);
    }
  }
}

static void test_rope_scaled() {
  std::printf("[rope] llama3 scaled RoPE kernel vs fp64 reference\n");
  const int T = 48, nq = 4, nkv = 2, hd = 128;
  const RopeScalingLlama3 real{8.0, 1.0, 4.0, 8192};
  // reference table: literal HF _compute_llama3_parameters
  std::vector<double> hf(hd / 2);
  for (int i = 0; i < hd / 2; ++i) {
    const double inv = 1.0 / std::pow(500000.0, double(2 * i) / hd);
    const double wavelen = 2 * M_PI / inv, low = 8192.0 / 1.0, high = 8192.0 / 4.0;
    double v = wavelen > low ? inv / 8.0 : inv;
    const double smooth = (8192.0 / wavelen - 1.0) / (4.0 - 1.0);
    const double smoothed = (1 - smooth) * v / 8.0 + smooth * v;
    hf[size_t(i)] = (!(wavelen < high) && !(wavelen > low)) ? smoothed : v;
  }
  std::vector<float> table;
  llama3_inv_freq(hd, 500000.0, real, table);
  double worst_table = 0;
  for (int i = 0; i < hd / 2; ++i) worst_table = std::max(worst_table, std::fabs(double(table[size_t(i)]) - hf[size_t(i)]) / hf[size_t(i)]);
  EXPECT(worst_table < 1e-6, "inv_freq vs fp64 HF (rel %g)", worst_table);
  const int nh = nq + 2 * nkv;
  Rng r(3);
  std::vector<float> qkv(size_t(T) * nh * hd), cs(size_t(nh) * hd, 1.f);
  for (auto& v : qkv) v = float(r.normal());
  DBuf<float> d_qkv(qkv.size()), d_cs(cs.size()), d_inv(table.size());
  HIP_CHECK(hipMemcpy(d_qkv, qkv.data(), qkv.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_cs, cs.data(), cs.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_inv, table.data(), table.size() * 4, hipMemcpyHostToDevice));
  DBuf<uint16_t> Q(size_t(nq) * T * hd), K(size_t(nkv) * T * hd), V(size_t(nkv) * T * hd), Q2(Q.n), K2(K.n), V2(V.n);
  k_qkv_rope_scaled(d_qkv, d_cs, Q, K, V, T, nq, nkv, hd, d_inv, nullptr);
  k_qkv_rope(d_qkv, d_cs, Q2, K2, V2, T, nq, nkv, hd, 500000.f, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> q(Q.n), q2(Q2.n), k(K.n);
  HIP_CHECK(hipMemcpy(q.data(), Q, q.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(q2.data(), Q2, q2.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k.data(), K, k.size() * 2, hipMemcpyDeviceToHost));
  double e = 0, n = 0, e_plain = 0;
  auto rotate = [&](int h, int t, int d, const std::vector<uint16_t>& out, size_t at, double& err) {
    const double a = bf16_to_f32(f32_to_bf16(qkv[size_t(t) * nh * hd + size_t(h) * hd + d]));
    const double b = bf16_to_f32(f32_to_bf16(qkv[size_t(t) * nh * hd + size_t(h) * hd + d + hd / 2]));
    const double ang = double(float(t) * float(hf[size_t(d)]));
    const double o1 = a * std::cos(ang) - b * std::sin(ang), o2 = b * std::cos(ang) + a * std::sin(ang);
    err += std::pow(bf16_to_f32(out[at + size_t(d)]) - o1, 2) + std::pow(bf16_to_f32(out[at + size_t(d + hd / 2)]) - o2, 2);
    return o1 * o1 + o2 * o2;
  };
  for (int h = 0; h < nq; ++h)
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < hd / 2; ++d) {
        n += rotate(h, t, d, q, (size_t(h) * T + t) * hd, e);
        double dummy = 0;
        rotate(h, t, d, q2, (size_t(h) * T + t) * hd, dummy);
        e_plain += dummy;
      }
  const double rel = std::sqrt(e / n), rel_plain = std::sqrt(e_plain / n);
  std::printf("  scaled-kernel rel L2 vs HF fp64: %.3g; plain-kernel vs same reference: %.3g\n", rel, rel_plain);
  EXPECT(rel < 0.008, "scaled RoPE kernel");
  EXPECT(rel_plain > rel, "the plain-RoPE kernel must differ from the llama3 reference");
}

// ------------------------------------------------------------------ pipeline cases
static std::vector<double> metric_values(const fs::path& out, const char* key) {
  json report = json::parse(std::ifstream(out / "adaptive_report.json"));
  std::vector<double> v;
  for (auto& l : report["layers"]) v.push_back(l["adaptive"][key].get<double>());
  return v;
}

int main(int argc, char** argv) {
  int dev = 0;
  fs::path keep;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc) dev = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--keep") && i + 1 < argc) keep = argv[++i];
  }
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::puts("no GPU device; skipped");
    return 77;
  }
  HIP_CHECK(hipSetDevice(dev));
  test_config();
  test_hadamard_8192();
  test_rope_scaled();

  fs::path root = keep.empty() ? fs::temp_directory_path() / "qtip_llama_tests" : keep;
  fs::remove_all(root);
  fs::create_directories(root);
  build_source(root / "source");

  std::printf("[run] QTIP quantization of a BF16 Llama-geometry source (all 7 linears incl. K/V)\n");
  {
    Options o = base_options(root, "qtip", "run");
    Pipeline p(o);
    p.run();
    check_output(o.out, "qtip", false);
  }
  std::printf("[run] --kv-format fp8 with a BF16 source: K/V converted to FP8-Dynamic, rest QTIP\n");
  {
    Options o = base_options(root, "fp8kv", "run");
    o.fp8_kv = true;
    o.kv_explicit = true;
    Pipeline p(o);
    p.run();
    check_output(o.out, "fp8kv", false, nullptr, true, root / "source");
    // work dir written with another K/V format must be refused on resume
    pid_t pid = fork();
    if (pid == 0) {
      if (!freopen("/dev/null", "w", stderr)) _exit(2);
      Options q = base_options(root, "fp8kv", "run");
      q.fp8_kv = false;
      q.kv_explicit = true;
      Pipeline pq(q);
      _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    EXPECT(!(WIFEXITED(st) && WEXITSTATUS(st) == 0), "resuming an fp8-K/V work dir with --kv-format qtip must fail");
  }
  std::printf("[adaptive] all-QTIP, then all-NVFP4 with a tiled lm_head (3 tiles of 96 over vocab %d)\n", kVocab);
  {
    Options o = base_options(root, "adaptive-qtip", "adaptive");
    o.kl_threshold = 1e30;
    Pipeline p(o);
    p.run();
    check_output(o.out, "adaptive-qtip", false);
    json cfg = json::parse(std::ifstream(fs::path(o.out) / "config.json"));
    for (auto& f : cfg["qtip_quant"]["layer_formats"]) EXPECT(f == "qtip", "all layers qtip");
    auto untiled = metric_values(o.out, "kl_qtip");
    Options t = base_options(root, "adaptive-tiled", "adaptive");
    t.kl_threshold = 1e30;
    t.logit_tile = 96;
    Pipeline pt(t);
    pt.run();
    auto tiled = metric_values(t.out, "kl_qtip");
    EXPECT(tiled.size() == untiled.size() && !tiled.empty(), "metric count");
    for (size_t i = 0; i < tiled.size(); ++i) {
      std::printf("  layer %zu kl_qtip untiled %.6g tiled %.6g\n", i, untiled[i], tiled[i]);
      EXPECT(std::fabs(tiled[i] - untiled[i]) <= 1e-6 * std::max(1.0, std::fabs(untiled[i])), "tiled lm_head must equal untiled (layer %zu)", i);
    }
  }
  {
    Options o = base_options(root, "nvfp4", "adaptive");
    o.kl_threshold = 0.0;
    o.logit_tile = 96;
    Pipeline p(o);
    p.run();
    check_output(o.out, "nvfp4", true);
    json cfg = json::parse(std::ifstream(fs::path(o.out) / "config.json"));
    for (auto& f : cfg["qtip_quant"]["layer_formats"]) EXPECT(f == "nvfp4", "all layers nvfp4");
  }
  std::printf("[adaptive] --kv-format fp8: K/V FP8 in QTIP and (forced) NVFP4 layers; KL uses dequantized FP8 K/V\n");
  {
    Options o = base_options(root, "adaptive-fp8kv-qtip", "adaptive");
    o.kl_threshold = 1e30;
    o.fp8_kv = true;
    o.kv_explicit = true;
    Pipeline p(o);
    p.run();
    check_output(o.out, "adaptive-fp8kv-qtip", false, nullptr, true, root / "source");
    json rep = json::parse(std::ifstream(fs::path(o.out) / "adaptive_report.json"));
    EXPECT(rep["kv_format"] == "fp8", "adaptive_report kv_format");
    auto with_fp8 = metric_values(o.out, "kl_qtip");
    auto with_qkv = metric_values(root / "adaptive-qtip", "kl_qtip");
    bool differs = false;
    for (size_t i = 0; i < with_fp8.size(); ++i) {
      std::printf("  layer %zu kl_qtip fp8-kv %.6g qtip-kv %.6g\n", i, with_fp8[i], with_qkv[i]);
      differs |= with_fp8[i] != with_qkv[i];
    }
    EXPECT(differs, "KL must reflect the FP8 K/V candidate");
  }
  {
    Options o = base_options(root, "adaptive-fp8kv-nvfp4", "adaptive");
    o.kl_threshold = 0.0;
    o.fp8_kv = true;
    o.kv_explicit = true;
    Pipeline p(o);
    p.run();
    check_output(o.out, "adaptive-fp8kv-nvfp4", true, nullptr, true, root / "source");
    json cfg = json::parse(std::ifstream(fs::path(o.out) / "config.json"));
    for (auto& f : cfg["qtip_quant"]["layer_formats"]) EXPECT(f == "nvfp4", "all layers nvfp4");
  }
  std::printf("[adaptive] MIXED: threshold inside the observed per-layer KL range -> some layers QTIP, some NVFP4\n");
  {
    auto kls = metric_values(root / "adaptive-qtip", "kl_qtip");
    std::vector<double> sorted = kls;
    std::sort(sorted.begin(), sorted.end());
    EXPECT(sorted.size() == size_t(kLayers) && sorted.front() < sorted.back(), "per-layer KLs must differ to place a threshold");
    // layers with kl > threshold become NVFP4; the midpoint of the two smallest values is robust to run-to-run noise
    const double thr = 0.5 * (sorted[0] + sorted[1]);
    Options o = base_options(root, "adaptive-mixed", "adaptive");
    o.kl_threshold = thr;
    o.logit_tile = 96;
    Pipeline p(o);
    p.run();
    json cfg = json::parse(std::ifstream(fs::path(o.out) / "config.json"));
    json rep = json::parse(std::ifstream(fs::path(o.out) / "adaptive_report.json"));
    std::vector<bool> nv;
    int n_nv = 0;
    for (int L = 0; L < kLayers; ++L) {
      const bool expect_nv = kls[size_t(L)] > thr;
      const std::string cf = cfg["qtip_quant"]["layer_formats"][size_t(L)];
      const std::string rf = rep["layers"][size_t(L)]["format"];
      std::printf("  layer %d kl_qtip %.6g -> %s (report %s)\n", L, kls[size_t(L)], cf.c_str(), rf.c_str());
      EXPECT(cf == (expect_nv ? "nvfp4" : "qtip") && rf == cf, "layer %d format must follow kl_qtip > threshold", L);
      nv.push_back(cf == "nvfp4");
      n_nv += cf == "nvfp4";
    }
    EXPECT(n_nv > 0 && n_nv < kLayers, "mixed run must contain both formats (got %d NVFP4 of %d)", n_nv, kLayers);
    check_output(o.out, "adaptive-mixed", false, &nv);
  }
  if (keep.empty()) fs::remove_all(root);
  std::printf(g_fail ? "LLAMA TESTS FAILED (%d)\n" : "ALL LLAMA TESTS PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
