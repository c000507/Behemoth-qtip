// Full-stack parity of the quantizer's Qwen3.5 GPU layer runner (bf16 weights, bf16 residual stream)
// against the CPU fp32 reference forward (qwen_cpu_forward) on a short prompt.
// usage: qwen_gpu_parity <model_dir> <tokens.i32> [T] [device]
// Reports, per layer: chained (GPU stack end to end) and isolated (CPU residual of layer L-1 rounded to
// bf16 -> one GPU layer) relative L2 / max|diff| of the residual; then final logits (CPU final norm +
// lm_head applied to the GPU residual) max|diff| and argmax agreement. Exit 1 if a tolerance fails.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "common.hpp"
#include "qwen35.hpp"
#include "qwen_runner.hpp"

using namespace qtip;

static std::vector<uint16_t> read_bf16(const SafeTensorsReader& st, const std::string& name) {
  const TensorInfo& t = st.info(name);
  QCHECK(t.dtype == "BF16", name + ": expected BF16");
  std::vector<uint16_t> v(size_t(t.numel()));
  st.read_into(name, v.data(), v.size() * 2);
  return v;
}

static std::string pick(const SafeTensorsReader& st, std::initializer_list<const char*> names) {
  for (const char* n : names)
    if (st.has(n)) return n;
  die("tensor not found");
  return {};
}

struct Err {
  double rel = 0, maxd = 0;
};
static Err compare(const float* ref, const uint16_t* got, size_t n) {
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = double(bf16_to_f32(got[i])) - ref[i];
    num += d * d;
    den += double(ref[i]) * ref[i];
    mx = std::max(mx, std::fabs(d));
  }
  return {std::sqrt(num / std::max(den, 1e-30)), mx};
}

int main(int argc, char** argv) {
  QCHECK(argc >= 3, "usage: qwen_gpu_parity <model_dir> <tokens.i32> [T] [device]");
  const std::string dir = argv[1];
  QwenConfig cfg = load_qwen_config(dir);
  QCHECK(!cfg.moe(), "dense checkpoints only");
  SafeTensorsReader st(dir);
  std::vector<int32_t> tok;
  {
    std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
    QCHECK(f, "cannot open tokens");
    size_t n = size_t(f.tellg()) / 4;
    int T = argc >= 4 ? std::atoi(argv[3]) : int(n);
    QCHECK(T > 0 && size_t(T) <= n, "T out of range");
    tok.resize(size_t(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(tok.data()), std::streamsize(size_t(T) * 4));
  }
  const int dev = argc >= 5 ? std::atoi(argv[4]) : 0;
  const int T = int(tok.size()), H = cfg.hidden, V = cfg.vocab;
  std::vector<std::shared_ptr<QwenLayer>> layers;
  for (int L = 0; L < cfg.layers; ++L) layers.push_back(read_qwen_layer(st, cfg, L));

  // ---- CPU fp32 reference: logits + residual after every layer
  std::vector<float> ref_logits(size_t(T) * V), ref_layers;
  qwen_cpu_forward(cfg, st, layers, QwenCpuRefs{}, tok.data(), T, ref_logits.data(), 64, -1, nullptr, &ref_layers);
  QCHECK(ref_layers.size() == size_t(cfg.layers) * T * H, "reference residual dump size");

  // ---- GPU chain from the bf16 embeddings
  const auto embed = read_bf16(st, pick(st, {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"}));
  const auto fnorm = read_bf16(st, pick(st, {"model.language_model.norm.weight", "model.norm.weight"}));
  std::vector<uint16_t> head_own;
  if (!cfg.tied) head_own = read_bf16(st, "lm_head.weight");
  const std::vector<uint16_t>& head = cfg.tied ? embed : head_own;
  std::vector<uint16_t> x(size_t(T) * H), xi(size_t(T) * H);
  for (int t = 0; t < T; ++t)
    std::copy_n(embed.data() + size_t(tok[size_t(t)]) * H, H, x.data() + size_t(t) * H);
  QwenLayerRunner R(dev, cfg, T, T);
  double worst_iso = 0, worst_chain = 0;
  std::printf("layer type    chained relL2  max|d|   isolated relL2  max|d|\n");
  for (int L = 0; L < cfg.layers; ++L) {
    R.upload(*layers[size_t(L)]);
    // isolated: input = CPU residual entering layer L (embeddings for L = 0), bf16-rounded
    if (L == 0) xi = std::vector<uint16_t>(x);
    else
      for (size_t i = 0; i < xi.size(); ++i) xi[i] = f32_to_bf16(ref_layers[size_t(L - 1) * T * H + i]);
    R.forward(xi.data(), 1, nullptr);
    R.forward(x.data(), 1, nullptr);
    const float* ref = ref_layers.data() + size_t(L) * T * H;
    Err c = compare(ref, x.data(), x.size()), i = compare(ref, xi.data(), xi.size());
    worst_iso = std::max(worst_iso, i.rel);
    worst_chain = std::max(worst_chain, c.rel);
    std::printf("%5d %-7s %10.2e %9.4f %14.2e %9.4f\n", L, cfg.is_full(L) ? "full" : "linear", c.rel, c.maxd,
                i.rel, i.maxd);
  }
  // ---- final zero-centred norm + lm_head on the GPU residual (fp32, CPU)
  std::vector<float> xf(size_t(T) * H), lg(size_t(T) * V);
  for (int t = 0; t < T; ++t) {
    double ss = 0;
    for (int k = 0; k < H; ++k) {
      float v = bf16_to_f32(x[size_t(t) * H + k]);
      ss += double(v) * v;
    }
    float r = float(1.0 / std::sqrt(ss / H + cfg.eps));
    for (int k = 0; k < H; ++k)
      xf[size_t(t) * H + k] = bf16_to_f32(x[size_t(t) * H + k]) * r * (1.f + bf16_to_f32(fnorm[size_t(k)]));
  }
#pragma omp parallel for schedule(static)
  for (long v = 0; v < V; ++v) {
    std::vector<float> w(static_cast<size_t>(H));
    for (int k = 0; k < H; ++k) w[size_t(k)] = bf16_to_f32(head[size_t(v) * H + k]);
    for (int t = 0; t < T; ++t) {
      float s = 0;
      for (int k = 0; k < H; ++k) s += xf[size_t(t) * H + k] * w[size_t(k)];
      lg[size_t(t) * V + v] = s;
    }
  }
  double maxd = 0, scale = 0;
  int agree = 0;
  for (int t = 0; t < T; ++t) {
    const float* a = ref_logits.data() + size_t(t) * V;
    const float* b = lg.data() + size_t(t) * V;
    int am = int(std::max_element(a, a + V) - a), bm = int(std::max_element(b, b + V) - b);
    agree += am == bm;
    for (int v = 0; v < V; ++v) {
      maxd = std::max(maxd, double(std::fabs(a[v] - b[v])));
      scale = std::max(scale, double(std::fabs(a[v])));
    }
    std::printf("  t=%2d token %6d  argmax ref %6d gpu %6d %s\n", t, tok[size_t(t)], am, bm, am == bm ? "" : "<-");
  }
  // tolerances: bf16 weights + bf16 residual stream vs fp32 everywhere
  const bool ok = worst_iso < 2e-2 && worst_chain < 5e-2 && agree >= (T * 9 + 9) / 10;
  std::printf("worst isolated relL2 %.2e (tol 2e-2), worst chained relL2 %.2e (tol 5e-2)\n", worst_iso, worst_chain);
  std::printf("logits max|diff| %.4f (max|logit| %.2f, rel %.2e), argmax agreement %d/%d (tol >= 90%%)\n", maxd,
              scale, maxd / scale, agree, T);
  std::printf(ok ? "QWEN GPU PARITY OK\n" : "QWEN GPU PARITY FAILED\n");
  return ok ? 0 : 1;
}
