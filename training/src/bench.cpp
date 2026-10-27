#include "train.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
using namespace qat;
int main(int argc, char **argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "qtip_bench MODEL [DEVICE=cpu|0] [SEQLEN=32] [LAYER=0] [MLP_ROWS=1024]\n";
            return argc < 2 ? 1 : 0;
        }
        int device = argc > 2 && std::string(argv[2]) != "cpu" ? std::stoi(argv[2]) : -1,
            T = argc > 3 ? std::stoi(argv[3]) : 32, l = argc > 4 ? std::stoi(argv[4]) : 0,
            chunk = argc > 5 ? std::stoi(argv[5]) : 1024;
        require(T > 0 && chunk > 0, "invalid sequence/chunk");
        if (device < 0)
            require(Config(read_json(fs::path(argv[1]) / "config.json")).n <= 1024,
                    "CPU benchmark is for tiny models");
        Model m(argv[1]);
        require(l >= 0 && l < m.config.layers, "invalid layer");
        Replica r(m, device, T, size_t(24) * 1000000000, true);
        auto &o = *r.ops;
        Batch b;
        b.tokens.assign(T, 0);
        b.labels.assign(T, -100);
        b.segments.assign(T, 0);
        b.positions.resize(T);
        for (int i = 0; i < T; ++i)
            b.positions[i] = i;
        b.boundaries();
        auto x = o.embedding(r.embedding, b), dy = o.alloc(T, m.config.n);
        std::vector<float> g(dy.size());
        for (size_t i = 0; i < g.size(); ++i)
            g[i] = float(std::sin(double(i))) * .001f;
        o.upload(dy, g);
        o.prefetch(l, l % 2, false);
        o.acquire(l % 2);
        auto start = std::chrono::steady_clock::now();
        Layer layer(o, m, l, r.parameters, r.gradients);
        o.synchronize();
        auto decoded = std::chrono::steady_clock::now();
        auto y = layer.forward(x, b, chunk);
        o.synchronize();
        auto fwd = std::chrono::steady_clock::now();
        auto dx = layer.backward(x, dy, b, chunk);
        o.synchronize();
        auto end = std::chrono::steady_clock::now();
        auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        std::cout << json({{"layer", l},
                           {"seqlen", T},
                           {"decode_seconds", sec(start, decoded)},
                           {"forward_seconds", sec(decoded, fwd)},
                           {"backward_seconds", sec(fwd, end)},
                           {"device_peak_bytes", o.peak_bytes()}})
                         .dump()
                  << '\n';
        o.release(l % 2);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "qtip_bench: " << e.what() << '\n';
        return 1;
    }
}
