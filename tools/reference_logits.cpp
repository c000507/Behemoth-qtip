// Stream an original checkpoint one layer at a time to produce the log-probability
// files consumed by runtime/behemoth_cuda_model_kl. A 70B teacher need not fit in VRAM.
#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace qtip;
using json = nlohmann::json;
namespace fs = std::filesystem;
int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::printf("qtip_reference_logits --model ORIGINAL_DIR --tokens TOKENS.i32 --seq-len T "
                  "--out REF_DIR [--nseq 1]\n");
      return 0;
    }
    QCHECK(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "missing/duplicate argument");
  }
  for (const auto& [key, value] : args) {
    (void)value;
    QCHECK(key == "--model" || key == "--tokens" || key == "--seq-len" || key == "--out" || key == "--nseq",
           "unknown argument " + key);
  }
  for (const char* key : {"--model", "--tokens", "--seq-len", "--out"})
    QCHECK(args.count(key), std::string("missing ") + key);
  int length = std::stoi(args.at("--seq-len"));
  int count = args.count("--nseq") ? std::stoi(args.at("--nseq")) : 1;
  QCHECK(length > 1 && count > 0 && long(length) * count <= INT32_MAX, "invalid sequence geometry");
  const int rows = length * count;
  SafeTensorsReader st(args.at("--model"));
  auto config = load_model_config(args.at("--model"), &st);
  QCHECK(length <= config.max_pos, "sequence exceeds model context");
  for (const char* name : {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"})
    QCHECK(st.info(name).dtype == "BF16", std::string("BF16 required for ") + name);
  const fs::path directory(args.at("--out"));
  fs::create_directories(directory);
  for (int s = 0; s < count; ++s) {
    char name[64]; std::snprintf(name, sizeof name, "seq%03d.f32", s);
    QCHECK(!fs::exists(directory / name), "reference output already exists");
  }
  std::vector<int32_t> tokens(static_cast<size_t>(rows));
  std::ifstream tf(args.at("--tokens"), std::ios::binary);
  QCHECK(tf && bool(tf.read(reinterpret_cast<char*>(tokens.data()), long(rows) * 4)), "cannot read tokens");
  for (int token : tokens) QCHECK(token >= 0 && token < config.vocab, "token outside vocabulary");
  std::vector<uint16_t> activations(size_t(rows) * config.hidden);
  {
    auto embedding = st.read("model.embed_tokens.weight");
    for (int t = 0; t < rows; ++t)
      std::memcpy(activations.data() + size_t(t) * config.hidden,
                  embedding.data() + size_t(tokens[size_t(t)]) * config.hidden * 2, size_t(config.hidden) * 2);
  }
  HIP_CHECK(hipSetDevice(0));
  LayerRunner runner(0, config, rows, length);
  LayerCache cache(st, config, 3);
  double started = now_s(), read_seconds = 0, upload_seconds = 0, forward_seconds = 0;
  for (int layer = 0; layer < config.layers; ++layer) {
    double begin = now_s();
    auto weights = cache.get(layer);
    cache.prefetch(layer + 1);
    read_seconds += now_s() - begin;
    begin = now_s(); runner.upload(*weights); upload_seconds += now_s() - begin;
    begin = now_s(); runner.forward(activations.data(), count, nullptr); forward_seconds += now_s() - begin;
    if ((layer + 1) % 8 == 0 || layer + 1 == config.layers)
      logf("reference layers %d/%d: read %.1fs upload %.1fs forward %.1fs", layer + 1, config.layers,
           read_seconds, upload_seconds, forward_seconds);
  }
  auto head = st.read("lm_head.weight"), norm = st.read("model.norm.weight");
  const long h = config.hidden, v = config.vocab;
  DBuf<uint16_t> input(activations.size()), normalized(activations.size()), dnorm(h), dhead(size_t(h) * v);
  HIP_CHECK(hipMemcpy(input, activations.data(), input.bytes(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dnorm, norm.data(), norm.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dhead, head.data(), head.size(), hipMemcpyHostToDevice));
  head.clear(); head.shrink_to_fit();
  auto stream = runner.stream();
  k_rmsnorm(input, dnorm, normalized, rows, int(h), config.eps, stream);
  DBuf<float> logits(size_t(rows) * v);
  runner.gemm().run(true, false, int(v), rows, int(h), dhead, h, normalized, h, logits, v, OutT::F32);
  std::vector<float> probabilities(size_t(rows) * v);
  d2h_staged(probabilities.data(), logits, probabilities.size() * 4, stream);
#pragma omp parallel for schedule(static)
  for (int t = 0; t < rows; ++t) {
    float* p = probabilities.data() + size_t(t) * v;
    double maximum = -INFINITY, sum = 0;
    for (long j = 0; j < v; ++j) maximum = std::max(maximum, double(p[j]));
    for (long j = 0; j < v; ++j) sum += std::exp(double(p[j]) - maximum);
    QCHECK(std::isfinite(maximum) && std::isfinite(sum) && sum > 0, "nonfinite reference logits");
    const float normalizer = float(maximum + std::log(sum));
    for (long j = 0; j < v; ++j) p[j] -= normalizer;
  }
  for (int s = 0; s < count; ++s) {
    char name[64]; std::snprintf(name, sizeof name, "seq%03d.f32", s);
    fs::path final = directory / name, temporary = final.string() + ".tmp";
    {
      std::ofstream out(temporary, std::ios::binary);
      out.write(reinterpret_cast<const char*>(probabilities.data() + size_t(s) * length * v), long(length) * v * 4);
      QCHECK(out.good(), "reference file write failed");
    }
    fs::rename(temporary, final);
  }
  json manifest = {{"model", args.at("--model")}, {"tokens", args.at("--tokens")},
      {"seq_len", length}, {"nseq", count}, {"vocab", v}, {"format", "little-endian float32 log probabilities"},
      {"source_forward", "native quantizer BF16 prefill; one independent causal document per sequence"},
      {"read_seconds", read_seconds}, {"upload_seconds", upload_seconds}, {"forward_seconds", forward_seconds},
      {"elapsed_seconds", now_s() - started}};
  std::ofstream(directory / "manifest.json") << manifest.dump(2) << '\n';
  logf("reference log probabilities written: %d x %d tokens, vocabulary %ld", count, length, v);
}
