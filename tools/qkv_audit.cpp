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
                  "[--measure-ntok N] limits direct projection checks, retaining all --ntok covariance tokens.\n"
                  "[--balance on|off] tests diagonal weight/Hessian equilibration (default off).\n"
                  "Audits layer 0 only; token order and batch size should match the quantizer.\n");
      return 0;
    }
    QCHECK(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "missing/duplicate argument");
  }
  for (const auto& [key, value] : args) {
    (void)value;
    QCHECK(key == "--model" || key == "--tokens" || key == "--out" || key == "--ntok" ||
           key == "--batch-rows" || key == "--sigma-reg" || key == "--ldl-threads" || key == "--measure-ntok" ||
           key == "--balance", "unknown option " + key);
  }
  QCHECK(args.count("--model") && args.count("--tokens") && args.count("--out"), "missing required argument");
  long ntok = args.count("--ntok") ? std::stol(args.at("--ntok")) : 4194304;
  int batch = args.count("--batch-rows") ? std::stoi(args.at("--batch-rows")) : 4096;
  QCHECK(ntok > 0 && batch > 0, "token and batch counts must be positive");
  long measure = args.count("--measure-ntok") ? std::stol(args.at("--measure-ntok")) : ntok;
  QCHECK(measure > 0 && measure <= ntok, "measurement count must be within calibration tokens");
  QCHECK(!args.count("--balance") || args.at("--balance") == "on" || args.at("--balance") == "off", "--balance expects on|off");
  bool balance = args.count("--balance") && args.at("--balance") == "on";
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
  std::vector<float> column_scale(size_t(n), 1.f), balanced_h;
  std::vector<std::vector<float>> balanced_w(3);
  if (balance) {
    double mean_diag = 0;
    for (int j = 0; j < n; ++j) mean_diag += h[size_t(j) * n + j] / n;
    long total_rows = 0;
    for (const auto& input : inputs) total_rows += input.m;
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
      double column_norm = 0;
      for (int li = 0; li < 3; ++li)
        for (int row = 0; row < inputs[size_t(li)].m; ++row) {
          double w = bf16_to_f32(inputs[size_t(li)].w_bf16[size_t(row) * n + j]);
          column_norm += w * w / total_rows;
        }
      double d = std::pow(std::max(double(h[size_t(j) * n + j]), mean_diag * 1e-12) /
                          std::max(column_norm, 1e-30), 0.25);
      // A future SU export can represent the inverse scale exactly in BF16.
      column_scale[size_t(j)] = 1.f / bf16_to_f32(f32_to_bf16(float(1.0 / d)));
    }
    balanced_h.resize(h.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        balanced_h[size_t(i) * n + j] = float(double(h[size_t(i) * n + j]) /
                (double(column_scale[size_t(i)]) * column_scale[size_t(j)]));
    for (int li = 0; li < 3; ++li) {
      auto& w = balanced_w[size_t(li)];
      w.resize(size_t(inputs[size_t(li)].m) * n);
#pragma omp parallel for schedule(static)
      for (size_t k = 0; k < w.size(); ++k)
        w[k] = bf16_to_f32(inputs[size_t(li)].w_bf16[k]) * column_scale[k % n];
      inputs[size_t(li)].w_bf16 = nullptr;
      inputs[size_t(li)].w_f32 = w.data();
    }
  }
  auto outputs = quantize_group(stream, gemm, inputs, n, balance ? balanced_h.data() : h.data(), 0, options, nullptr, "audit-L0");
  json report = {{"model", args.at("--model")}, {"tokens", args.at("--tokens")}, {"ntok", ntok},
                 {"projection_tokens", measure}, {"batch_rows", batch}, {"layer", 0}, {"sigma_reg", options.sigma_reg}};
  report["balance"] = {{"enabled", balance}, {"min_scale", *std::min_element(column_scale.begin(), column_scale.end())},
      {"max_scale", *std::max_element(column_scale.begin(), column_scale.end())},
      {"formula", "D=(diag(H)/mean_rows(W^2))^(1/4); W'=W D, H'=D^-1 H D^-1"}};
  double trace = 0, min_diag = INFINITY, max_diag = 0;
  for (int j = 0; j < n; ++j) {
    double d = h[size_t(j) * n + j];
    trace += d; min_diag = std::min(min_diag, d); max_diag = std::max(max_diag, d);
  }
  const double damping = options.sigma_reg * trace / n;
  double v_energy = 0, low_variance_energy = 0;
  int damped_columns = 0;
  for (int j = 0; j < n; ++j) {
    bool low = h[size_t(j) * n + j] < damping;
    damped_columns += low;
    for (int row = 0; row < inputs[LV].m; ++row) {
      double w = bf16_to_f32(reinterpret_cast<const uint16_t*>(weights[LV].data())[size_t(row) * n + j]);
      v_energy += w * w;
      if (low) low_variance_energy += w * w;
    }
  }
  report["conditioning"] = {{"hessian_diag_min", min_diag}, {"hessian_diag_max", max_diag},
      {"damping", damping}, {"columns_below_damping", damped_columns},
      {"v_weight_energy_fraction_below_damping", low_variance_energy / v_energy}};
  // Measure real projection SSE rather than subtracting two damped quadratic forms.
  for (int li = 0; li < 3; ++li) {
    const auto& candidate = outputs[size_t(li)];
    int m = inputs[size_t(li)].m;
    uint64_t fingerprint = 14695981039346656037ull;
    for (uint16_t word : candidate.trellis)
      for (int shift : {0, 8}) {
        fingerprint ^= (word >> shift) & 255;
        fingerprint *= 1099511628211ull;
      }
    std::vector<uint16_t> rounded(candidate.hatW.size());
    for (size_t i = 0; i < rounded.size(); ++i) rounded[i] = f32_to_bf16(candidate.hatW[i] / column_scale[i % n]);
    DBuf<uint16_t> original(size_t(m) * n), quantized(size_t(m) * n);
    HIP_CHECK(hipMemcpy(original, weights[li].data(), original.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(quantized, rounded.data(), quantized.bytes(), hipMemcpyHostToDevice));
    DBuf<float> yo(size_t(batch) * m), yq(size_t(batch) * m), error(size_t(batch) * m);
    DBuf<double> totals(2);
    HIP_CHECK(hipMemsetAsync(totals, 0, totals.bytes(), stream));
    for (long offset = 0; offset < measure; offset += batch) {
      int rows = int(std::min<long>(batch, measure - offset));
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
                            {"signal_sumsq", values[1]}, {"error_sumsq", values[0]}, {"trellis_fnv64", fingerprint},
                            {"wscale", candidate.wscale}};
    logf("QKV audit %s: direct relative SSE %.8g, proxy %.8g, regularized proxy %.8g",
         kLinName[li], ratio, candidate.proxy, candidate.proxy_rot_reg);
    if (li == LV) {
      // Decode the actual packed payload, then invert its rotations in FP64.
      // This checks the large production shape independently of GPU hatW.
      std::vector<uint16_t> packed(candidate.trellis.size()), states(size_t(m / 16) * (n / 16) * 128);
      unswizzle_for_kernel(candidate.trellis.data(), m, n, packed.data());
      unpack_trellis(packed.data(), long(states.size() / 128), states.data());
      std::vector<double> decoded(size_t(m) * n), transposed(size_t(m) * n);
      for (int block_row = 0; block_row < m / 16; ++block_row)
        for (int block_col = 0; block_col < n / 16; ++block_col) {
          float tile[256];
          decode_states(states.data() + (size_t(block_row) * (n / 16) + block_col) * 128, tile);
          for (int p = 0; p < 256; ++p) {
            int q = perm256()[p];
            decoded[size_t(block_row * 16 + q / 16) * n + block_col * 16 + q % 16] =
                double(tile[p]) * candidate.wscale;
          }
        }
      hadamard_rows_ref(decoded.data(), m, n, n, false);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) transposed[size_t(j) * m + i] = decoded[size_t(i) * n + j];
      hadamard_rows_ref(transposed.data(), n, m, m, false);
      double diff2 = 0, ref2 = 0;
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          double reference = transposed[size_t(j) * m + i] * candidate.sv[size_t(i)] * bf16_to_f32(candidate.su_bf16[size_t(j)]);
          double delta = reference - candidate.hatW[size_t(i) * n + j];
          diff2 += delta * delta; ref2 += reference * reference;
        }
      double relative = std::sqrt(diff2 / ref2);
      report[kLinName[li]]["packed_decode_relative_l2"] = relative;
      logf("QKV audit V: CPU packed decode vs GPU reconstruction relative L2 %.8g", relative);
      QCHECK(relative < 2e-5, "packed V weights disagree with GPU reconstruction");
    }
  }
  report["elapsed_seconds"] = now_s() - start;
  std::ofstream out(args.at("--out"));
  QCHECK(out && bool(out << report.dump(2) << '\n'), "cannot write audit report");
  HIP_CHECK(hipStreamDestroy(stream));
}
