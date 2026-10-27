// Analytical prefill accounting, not a hardware profiler. Counts FMA as two FLOPs.
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
using json = nlohmann::json;
namespace {
void check(bool ok, const std::string &what) {
    if (!ok)
        throw std::invalid_argument(what);
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << R"(behemoth_prefill_compute [options]
  --config FILE               HF config.json (default: Behemoth 123B)
  --rows N                    New query tokens (2048)
  --context N                 Prefix tokens already in KV (0)
  --sampled-rows N            Rows projected through lm_head (1)
  --tokens-per-second X       Measured whole-step prefill throughput
  --attention fp8|bf16        Matrix precision (fp8; KV remains FP8 in either mode)
  --bf16-peak-tflops X        Per-GPU dense matrix peak (191.4)
  --fp8-peak-tflops X         Per-GPU dense matrix peak (382.8)

Outputs JSON for TP=2. Peak-normalized fractions are analytical useful-matrix
throughput, not GPU busy percentages or hardware-counter measurements.
Context is an existing prefix; rows=32768, context=0 models a whole cold prompt.
)";
                return 0;
            }
            check(key == "--config" || key == "--rows" || key == "--context" ||
                      key == "--sampled-rows" || key == "--tokens-per-second" ||
                      key == "--attention" || key == "--bf16-peak-tflops" ||
                      key == "--fp8-peak-tflops",
                  "unknown option: " + key);
            check(i + 1 < argc, "missing value for " + key);
            check(args.emplace(key, argv[++i]).second, "duplicate option: " + key);
        }
        auto integer = [&](const char *key, uint64_t fallback) {
            if (!args.contains(key))
                return fallback;
            const auto &s = args.at(key);
            size_t end = 0;
            check(!s.empty() && s[0] != '-', "negative/empty integer");
            auto x = std::stoull(s, &end);
            check(end == s.size() && x <= uint64_t(INT32_MAX), "integer out of range");
            return uint64_t(x);
        };
        auto real = [&](const char *key, double fallback) {
            if (!args.contains(key))
                return fallback;
            size_t end = 0;
            double x = std::stod(args.at(key), &end);
            check(end == args.at(key).size() && std::isfinite(x) && x > 0, "invalid rate");
            return x;
        };
        json config = {{"hidden_size", 12288},     {"intermediate_size", 28672},
                       {"num_hidden_layers", 88},  {"num_attention_heads", 96},
                       {"num_key_value_heads", 8}, {"head_dim", 128},
                       {"vocab_size", 32768}};
        if (args.contains("--config")) {
            std::ifstream f(args.at("--config"));
            check(bool(f), "cannot read config");
            f >> config;
        }
        auto dimension = [&](const char *key) {
            int64_t x = config.at(key).get<int64_t>();
            check(x > 0 && x <= INT32_MAX, std::string("invalid ") + key);
            return double(x);
        };
        const double n = dimension("hidden_size"), inter = dimension("intermediate_size"),
                     layers = dimension("num_hidden_layers"),
                     heads = dimension("num_attention_heads"),
                     kv = dimension("num_key_value_heads"), vocab = dimension("vocab_size");
        const double dim = config.contains("head_dim") ? dimension("head_dim") : n / heads;
        check(dim > 0 && std::floor(dim) == dim && std::fmod(heads, kv) == 0 &&
                  std::fmod(heads, 2) == 0 && std::fmod(kv, 2) == 0,
              "invalid TP=2 head geometry");
        const double qd = heads * dim, kd = kv * dim;
        const uint64_t rows = integer("--rows", 2048), context = integer("--context", 0),
                       sampled = integer("--sampled-rows", 1);
        check(rows > 0 && sampled <= rows, "invalid query/sampled row count");
        const double bf16 = real("--bf16-peak-tflops", 191.4),
                     fp8 = real("--fp8-peak-tflops", 382.8);
        const std::string attention = args.contains("--attention") ? args.at("--attention") : "fp8";
        check(attention == "fp8" || attention == "bf16", "invalid attention precision");
        const double rate = real("--tokens-per-second", 0);
        json projections = json::array();
        double weights = 0;
        auto add = [&](const char *name, double in, double out) {
            weights += in * out;
            projections.push_back({{"name", name},
                                   {"input", in},
                                   {"output", out},
                                   {"gflop_per_token_all_layers", 2 * in * out * layers / 1e9},
                                   {"step_tflop_per_gpu", rows * in * out * layers / 1e12}});
        };
        add("q", n, qd);
        add("k", n, kd);
        add("v", n, kd);
        add("o", qd, n);
        add("gate", n, inter);
        add("up", n, inter);
        add("down", inter, n);
        const double b = double(rows), c = double(context);
        const double pairs = b * c + b * (b + 1) / 2;
        const double pf = 2 * weights * layers * b, af = 4 * layers * qd * pairs;
        const double hf = 2 * sampled * n * vocab;
        // Each of the five TP=2 all-gathers sends one local BF16 half in each
        // direction: qkv, attention, o, gate/up, down. No PCIe-P2P assumed.
        const double outgoing = (qd + 2 * kd + qd + n + 2 * inter + n) * b * layers;
        const double bf16_peak_s = pf / (2 * bf16 * 1e12);
        const double attention_peak_s = af / (2 * (attention == "fp8" ? fp8 : bf16) * 1e12);
        // Serving lm_head is FP8; only the sampled rows are evaluated.
        const double peak_s = bf16_peak_s + attention_peak_s + hf / (2 * fp8 * 1e12);
        json result = {{"kind", "analytical_useful_matrix_work_not_hardware_counters"},
                       {"gpus", 2},
                       {"query_rows", rows},
                       {"prefix_tokens", context},
                       {"attention_precision", attention},
                       {"projections", projections},
                       {"projection_weights_all_layers", weights * layers},
                       {"projection_gflop_per_token", pf / b / 1e9},
                       {"attention_gflop_per_token", af / b / 1e9},
                       {"step_projection_tflop_pair", pf / 1e12},
                       {"step_attention_tflop_pair", af / 1e12},
                       {"step_head_tflop_pair", hf / 1e12},
                       {"sampled_rows", sampled},
                       {"bf16_peak_tflops_per_gpu", bf16},
                       {"fp8_peak_tflops_per_gpu", fp8},
                       {"ideal_serial_matrix_time_seconds", peak_s},
                       {"ideal_matrix_only_tokens_per_second", b / peak_s},
                       {"bf16_exchange_gb_per_rank_per_direction", outgoing / 1e9},
                       {"exchange_floor_seconds_at_12_5_gb_s", outgoing / 12.5e9}};
        if (rate > 0) {
            const double elapsed = b / rate;
            result["measured_tokens_per_second"] = rate;
            result["measured_step_seconds"] = elapsed;
            result["useful_projection_tflops_per_gpu_over_wall"] = pf / 2 / elapsed / 1e12;
            result["useful_all_matrix_tflops_per_gpu_over_wall"] =
                (pf + af + hf) / 2 / elapsed / 1e12;
            result["precision_weighted_peak_fraction"] = peak_s / elapsed;
            result["projection_only_peak_fraction"] = bf16_peak_s / elapsed;
        }
        std::cout << result.dump(2) << '\n';
    } catch (const std::exception &e) {
        std::cerr << "prefill_compute: " << e.what() << '\n';
        return 1;
    }
}
