// Checks expand_qtip_to_fp8 against qtip::reference_forward on a real layer.
// build: g++ -std=c++20 -O2 -fopenmp -I runtime/include -I third_party neuron/cpp/qtip_expand_test.cpp neuron/cpp/qtip_expand.cpp \
//            runtime/src/qtip.cpp runtime/src/checkpoint.cpp runtime/src/attention.cpp -o qtip_expand_test
// usage: qtip_expand_test MODEL_DIR LAYER PROJECTION TP     (prints relative L2 error of the FP8 shards, weight-only)
#include "qtip_expand.hpp"
#include "behemoth/numeric.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

using namespace behemoth;

int main(int argc, char **argv) {
    if (argc != 5) { std::cerr << "usage: MODEL_DIR LAYER PROJECTION TP\n"; return 2; }
    auto root = std::filesystem::path(argv[1]);
    const int layer = std::stoi(argv[2]), tp = std::stoi(argv[4]);
    const std::string name = argv[3];
    auto config = ModelConfig::read(root / "config.json");
    char filename[64];
    std::snprintf(filename, sizeof filename, "model-layer-%05d.safetensors", layer);
    SafeTensorFile shard(root / filename);
    auto l = qtip::load_linear(shard, layer, name, &config);
    const bool column_parallel = name == "self_attn.q_proj" || name == "mlp.gate_proj" || name == "mlp.up_proj";

    std::vector<float> x(size_t(l.in));
    for (int i = 0; i < l.in; ++i) x[size_t(i)] = .1f * std::sin(float(i) * .071f);
    auto y_ref = qtip::reference_forward(l, x, true, false);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<neuron::Fp8Shard> shards;
    for (int r = 0; r < tp; ++r) shards.push_back(neuron::expand_qtip_to_fp8(l, tp, r, column_parallel));
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<double> y(size_t(l.out), 0.0);
    for (int r = 0; r < tp; ++r) {
        const auto &s = shards[size_t(r)];
        for (int row = 0; row < s.rows; ++row) {
            double acc = 0;
            for (int c = 0; c < s.cols; ++c) {
                const int xi = column_parallel ? c : r * (l.in / tp) + c;
                acc += double(decode_e4m3(s.w[size_t(row) * size_t(s.cols) + size_t(c)])) * double(float(BFloat16(x[size_t(xi)])));
            }
            const int oi = column_parallel ? r * (l.out / tp) + row : row;
            y[size_t(oi)] += acc * double(s.scale[size_t(row)]);
        }
    }
    double num = 0, den = 0;
    for (int o = 0; o < l.out; ++o) {
        const double d = y[size_t(o)] - double(y_ref[size_t(o)]);
        num += d * d;
        den += double(y_ref[size_t(o)]) * double(y_ref[size_t(o)]);
    }
    std::cout << name << " layer " << layer << " tp=" << tp << (column_parallel ? " col" : " row")
              << ": expand " << secs << " s, shard " << shards[0].rows << "x" << shards[0].cols
              << ", FP8-vs-QTIP reference relative_l2=" << std::sqrt(num / den) << '\n';
    return std::sqrt(num / den) < 0.08 ? 0 : 1;
}
