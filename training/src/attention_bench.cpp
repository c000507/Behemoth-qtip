#include "train.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
using namespace qat;

int main(int argc, char **argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "qtip_attention_bench SEQLEN [DEVICE=0|cpu] [DOC_LEN=SEQLEN] "
                         "[REPEATS=3] [HEADS=96] [KV_HEADS=8] [ARENA_GB=24]\n";
            return argc < 2 ? 1 : 0;
        }
        int T = std::stoi(argv[1]),
            device = argc > 2 ? (std::string(argv[2]) == "cpu" ? -1 : std::stoi(argv[2])) : 0,
            doc = argc > 3 ? std::stoi(argv[3]) : T, repeats = argc > 4 ? std::stoi(argv[4]) : 3,
            h = argc > 5 ? std::stoi(argv[5]) : 96, kh = argc > 6 ? std::stoi(argv[6]) : 8;
        double arena = argc > 7 ? std::stod(argv[7]) : 24;
        require(T > 0 && T <= 1048576 && doc > 0 && repeats > 0 && repeats <= 100,
                "invalid sequence, document length or repeat count");
        require(h > 0 && h <= 512 && kh > 0 && h % kh == 0, "invalid attention heads");
        require(arena > 0 && arena <= 1024, "arena GB out of range");
        require(device >= -1, "invalid device");
        require(device >= 0 || T <= 256, "CPU benchmark is limited to 256 tokens");
        Config c({{"hidden_size", h * 128},
                  {"intermediate_size", 128},
                  {"num_attention_heads", h},
                  {"num_key_value_heads", kh},
                  {"head_dim", 128},
                  {"num_hidden_layers", 1},
                  {"vocab_size", 2},
                  {"quip_params",
                   {{"L", 16},
                    {"K", 2},
                    {"V", 2},
                    {"tlut_bits", 9},
                    {"decode_mode", "quantlut_sym"},
                    {"td_x", 16},
                    {"td_y", 16}}}});
        auto o = device < 0 ? cpu_ops(true) : hip_ops(device, size_t(arena * 1e9));
        Batch b;
        b.tokens.assign(T, 0);
        b.segments.resize(T);
        b.positions.resize(T);
        double pairs = 0;
        for (int t = 0; t < T; ++t) {
            b.segments[t] = t / doc;
            b.positions[t] = t % doc;
            pairs += b.positions[t] + 1;
        }
        b.boundaries();
        std::vector<uint16_t> embedding(2 * c.n, 0);
        auto weight = o->frozen(embedding, 2, c.n);
        o->embedding(weight, b); // Upload starts/ends; no model weights required.
        auto input = [&](int cols, unsigned seed) {
            auto x = o->alloc(T, cols);
            std::vector<float> data(x.size());
            for (float &v : data) {
                seed = seed * 1664525u + 1013904223u;
                v = bf(float(seed >> 8) / 8388608.f - 1.f);
            }
            o->upload(x, data);
            return x;
        };
        auto q = input(c.n, 11), k = input(kh * 128, 19), v = input(kh * 128, 37), dy = input(c.n, 43);
        o->synchronize();
        std::vector<double> forwards, backwards;
        for (int i = 0; i <= repeats; ++i) {
            auto start = std::chrono::steady_clock::now();
            auto a = o->attention(q, k, v, c, b);
            o->synchronize();
            auto middle = std::chrono::steady_clock::now();
            auto gradients = o->attention_backward(a, dy, c, b);
            o->synchronize();
            auto end = std::chrono::steady_clock::now();
            double f = std::chrono::duration<double>(middle - start).count();
            double back = std::chrono::duration<double>(end - middle).count();
            std::cout << json({{"iteration", i},
                               {"warmup", i == 0},
                               {"forward_seconds", f},
                               {"backward_seconds", back}})
                             .dump()
                      << std::endl;
            if (i) {
                forwards.push_back(f);
                backwards.push_back(back);
            }
        }
        auto median = [](std::vector<double> x) {
            std::sort(x.begin(), x.end());
            return .5 * (x[(x.size() - 1) / 2] + x[x.size() / 2]);
        };
        double f = median(forwards), back = median(backwards);
        // Useful causal work excludes masked/tail tile overhead. The backward
        // GPU kernels evaluate seven products (three dQ, four dKV), forward two.
        // The CPU reference reuses scores and dP for all three gradients.
        double backward_products = device < 0 ? 5 : 7;
        std::cout << json({{"seqlen", T},
                           {"document_length", doc},
                           {"heads", h},
                           {"kv_heads", kh},
                           {"forward_median_seconds", f},
                           {"backward_median_seconds", back},
                           {"forward_useful_tflops", 4 * double(c.n) * pairs / f / 1e12},
                           {"backward_useful_tflops", 2 * backward_products * double(c.n) * pairs / back / 1e12},
                           {"device_peak_bytes", o->peak_bytes()}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "qtip_attention_bench: " << e.what() << '\n';
        return 1;
    }
}
