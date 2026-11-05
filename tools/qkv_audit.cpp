// Audit layer-zero Q/K/V quantization using direct projection errors, independently
// of the regularization-subtracted Hessian proxy. No full activation chain is kept.
#include "model.hpp"
#include "quantizer.hpp"
#include "qtip_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace qtip;
using json = nlohmann::json;

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::printf("qtip_qkv_audit --model BF16_DIR --tokens FILE --out REPORT.json "
                  "[--ntok 4194304] [--batch-rows 4096] [--sigma-reg 0.01] [--ldl-threads 16]\n"
                  "Audits layer 0 only; token order and batch size should match the quantizer.\n");
      return 0;
    }
    QCHECK(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "missing/duplicate argument");
  }
  for (const auto& [key, value] : args) {
    (void)value;
    QCHECK(key == "--model" || key == "--tokens" || key == "--out" || key == "--ntok" ||
           key == "--batch-rows" || key == "--sigma-reg" || key == "--ldl-threads", "unknown option " + key);
  }
  QCHECK(args.count("--model") && args.count("--tokens") && args.count("--out"), "missing required argument");
  long ntok = args.count("--ntok") ? std::stol(args.at("--ntok")) : 4194304;
  int batch = args.count("--batch-rows") ? std::stoi(args.at("--batch-rows")) : 4096;
  QCHECK(ntok > 0 && batch > 0, "token and batch counts must be positive");
  SafeTensorsReader st(args.at("--model"));
  auto cfg = load_model_config(args.at("--model"), &st);
  QCHECK(cfg.bf16_source, "this audit requires a BF16 source checkpoint");
  const int n = cfg.hidden;
  std::vector<int32_t> tokens(static_cast<size_t>(ntok));
  std::ifstream tf(args.at("--tokens"), std::ios::binary);
  QCHECK(tf && bool(tf.read(reinterpret_cast<char*>(tokens.data()), ntok * 4)), "cannot read calibration tokens");
  for (int id : tokens) QCHECK(id >= 0 && id < cfg.vocab, "token ID out of range");
  QCHECK(st.info("model.embed_tokens.weight").dtype == "BF16" &&
         st.info("model.layers.0.input_layernorm.weight").dtype == "BF16", "BF16 embeddings and input norm required");
  auto embeddings = st.read("model.embed_tokens.weight");
  auto norm = st.read("model.layers.0.input_layernorm.weight");
  std::vector<std::vector<uint8_t>> weights(3);
  std::vector<LinearIn> inputs(3);
  for (int li = 0; li < 3; ++li) {
    auto name = std::string("model.layers.0.") + kLinName[li] + ".weight";
    QCHECK(st.info(name).dtype == "BF16", "BF16 Q/K/V required");
    weights[li] = st.read(name);
    inputs[li].name = kLinName[li];
    inputs[li].m = int(st.info(name).shape[0]);
    inputs[li].w_bf16 = reinterpret_cast<const uint16_t*>(weights[li].data());
  }
  HIP_CHECK(hipSetDevice(0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  Gemm gemm(stream);
  DBuf<uint16_t> x(size_t(batch) * n), normalized(size_t(batch) * n), gamma(n);
  HIP_CHECK(hipMemcpy(gamma, norm.data(), norm.size(), hipMemcpyHostToDevice));
  HBuf<uint16_t> host(size_t(batch) * n);
  auto embed_batch = [&](long offset, int rows) {
    for (int r = 0; r < rows; ++r)
      std::memcpy(host + size_t(r) * n,
                  embeddings.data() + size_t(tokens[size_t(offset + r)]) * n * 2, size_t(n) * 2);
    HIP_CHECK(hipMemcpyAsync(x, host, size_t(rows) * n * 2, hipMemcpyHostToDevice, stream));
    k_rmsnorm(x, gamma, normalized, rows, n, cfg.eps, stream);
  };
  DBuf<float> dh(size_t(n) * n);
  HIP_CHECK(hipMemsetAsync(dh, 0, dh.bytes(), stream));
  const int blocks = (n % 4 == 0 && (n / 4) % 16 == 0) ? 4 : 1, bs = n / blocks;
  double start = now_s();
  for (long offset = 0; offset < ntok; offset += batch) {
    int rows = int(std::min<long>(batch, ntok - offset));
    embed_batch(offset, rows);
    for (int bi = 0; bi < blocks; ++bi)
      for (int bj = bi; bj < blocks; ++bj)
        gemm.run(false, true, bs, bs, rows, normalized + bi * bs, n, normalized + bj * bs, n,
                 dh + long(bj) * bs * n + bi * bs, n, OutT::F32, 1.f, 1.f);
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  if (blocks > 1) k_symmetrize_blocks(dh, n, bs, stream);
  std::vector<float> h(size_t(n) * n);
  d2h_staged(h.data(), dh, h.size() * 4, stream);
  for (auto& value : h) value = float(value * (1.0 / ntok));
  dh.free();
  logf("QKV audit: collected exact layer-zero input covariance over %ld tokens in %.1fs", ntok, now_s() - start);
  QuantOpts options;
  options.want_hatW = true;
  if (args.count("--sigma-reg")) options.sigma_reg = std::stod(args.at("--sigma-reg"));
  if (args.count("--ldl-threads")) options.ldl_threads = std::stoi(args.at("--ldl-threads"));
  QCHECK(std::isfinite(options.sigma_reg) && options.sigma_reg >= 0 && options.ldl_threads > 0,
         "invalid damping or thread count");
  auto outputs = quantize_group(stream, gemm, inputs, n, h.data(), 0, options, nullptr, "audit-L0");
  json report = {{"model", args.at("--model")}, {"tokens", args.at("--tokens")}, {"ntok", ntok},
                 {"batch_rows", batch}, {"layer", 0}, {"sigma_reg", options.sigma_reg}};
  // Measure real projection SSE rather than subtracting two damped quadratic forms.
  for (int li = 0; li < 3; ++li) {
    const auto& candidate = outputs[size_t(li)];
    int m = inputs[size_t(li)].m;
    std::vector<uint16_t> rounded(candidate.hatW.size());
    for (size_t i = 0; i < rounded.size(); ++i) rounded[i] = f32_to_bf16(candidate.hatW[i]);
    DBuf<uint16_t> original(size_t(m) * n), quantized(size_t(m) * n);
    HIP_CHECK(hipMemcpy(original, weights[li].data(), original.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(quantized, rounded.data(), quantized.bytes(), hipMemcpyHostToDevice));
    DBuf<float> yo(size_t(batch) * m), yq(size_t(batch) * m), error(size_t(batch) * m);
    DBuf<double> totals(2);
    HIP_CHECK(hipMemsetAsync(totals, 0, totals.bytes(), stream));
    for (long offset = 0; offset < ntok; offset += batch) {
      int rows = int(std::min<long>(batch, ntok - offset));
      embed_batch(offset, rows);
      gemm.run(true, false, m, rows, n, original, n, normalized, n, yo, m, OutT::F32);
      gemm.run(true, false, m, rows, n, quantized, n, normalized, n, yq, m, OutT::F32);
      k_sub(yo, yq, error, rows, m, m, stream);
      k_sumsq(error, rows, m, m, totals, stream);
      k_sumsq(yo, rows, m, m, totals + 1, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    double values[2];
    HIP_CHECK(hipMemcpy(values, totals, sizeof values, hipMemcpyDeviceToHost));
    double ratio = values[1] > 0 ? values[0] / values[1] : INFINITY;
    report[kLinName[li]] = {{"direct_relative_sse_bf16", ratio}, {"direct_relative_rmse_bf16", std::sqrt(ratio)},
                            {"proxy", candidate.proxy}, {"proxy_rot_reg", candidate.proxy_rot_reg},
                            {"signal_sumsq", values[1]}, {"error_sumsq", values[0]}};
    logf("QKV audit %s: direct relative SSE %.8g, proxy %.8g, regularized proxy %.8g",
         kLinName[li], ratio, candidate.proxy, candidate.proxy_rot_reg);
  }
  report["elapsed_seconds"] = now_s() - start;
  std::ofstream out(args.at("--out"));
  QCHECK(out && bool(out << report.dump(2) << '\n'), "cannot write audit report");
  HIP_CHECK(hipStreamDestroy(stream));
}
