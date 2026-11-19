// Warm microbenchmark of the BitNet b1.58 ternary projection against the dense BF16 gemv at the same
// shapes: GPU event time per call (median of repeated batches) and the effective weight bandwidth.
// Shapes default to the SmolLM2-360M projections; pass --shapes OUTxIN[,OUTxIN...] for others.
#include "kernels.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cg = glm53::gpu;
namespace cu = behemoth::cuda_runtime;

namespace {
struct Event {
    cudaEvent_t event{};
    Event() { cg::checked(cudaEventCreate(&event), "create timing event"); }
    ~Event() { (void)cudaEventDestroy(event); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
};
template <class F> double median_ms(F &&run, int batches, int iterations, cg::Stream s) {
    std::vector<double> times;
    for (int b = 0; b < batches; ++b) {
        Event start, stop;
        cg::checked(cudaEventRecord(start.event, s), "record start");
        for (int i = 0; i < iterations; ++i)
            run();
        cg::checked(cudaEventRecord(stop.event, s), "record stop");
        cg::checked(cudaEventSynchronize(stop.event), "wait for timing");
        float ms = 0;
        cg::checked(cudaEventElapsedTime(&ms, start.event, stop.event), "elapsed");
        times.push_back(double(ms) / iterations);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::vector<std::pair<int, int>> shapes = {{2560, 960}, {960, 2560}, {960, 960}, {320, 960}};
        int iterations = 50, tokens_max = 32;
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            if (a == "--shapes") {
                shapes.clear();
                std::stringstream ss(argv[i + 1]);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    const auto x = item.find('x');
                    shapes.emplace_back(std::stoi(item.substr(0, x)), std::stoi(item.substr(x + 1)));
                }
            } else if (a == "--iterations")
                iterations = std::stoi(argv[i + 1]);
            else if (a == "--tokens")
                tokens_max = std::stoi(argv[i + 1]);
            else
                throw std::runtime_error("unknown argument " + a);
        }
        cg::Stream s{};
        cg::checked(cg::stream_create(&s), "stream");
        std::mt19937 gen(3);
        std::cout << "shape,format,tokens,ms_per_call,weight_GB_s,ns_per_token\n";
        for (auto [out, in] : shapes) {
            std::vector<uint8_t> packed(size_t(out) * in / 4);
            for (auto &b : packed) {
                // valid codes only (no 3 in any 2-bit field)
                uint8_t v = 0;
                for (int k = 0; k < 4; ++k)
                    v |= uint8_t(gen() % 3) << (2 * k);
                b = v;
            }
            std::vector<uint16_t> dense(size_t(out) * in), x(size_t(tokens_max) * in);
            for (auto &v : dense)
                v = uint16_t(0x3c00 + gen() % 0x800) ^ ((gen() & 1) << 15);
            for (auto &v : x)
                v = uint16_t(0x3f00 + gen() % 0x400) ^ ((gen() & 1) << 15);
            cg::Buffer<uint8_t> w, scratch;
            cg::Buffer<uint16_t> wd, input, y;
            w.ensure(packed.size(), s);
            cg::checked(cg::h2d(w, packed.data(), packed.size(), s), "upload packed");
            wd.ensure(dense.size(), s);
            cg::checked(cg::h2d(wd, dense.data(), dense.size() * 2, s), "upload dense");
            input.ensure(x.size(), s);
            cg::checked(cg::h2d(input, x.data(), x.size() * 2, s), "upload x");
            y.ensure(size_t(tokens_max) * out, s);
            scratch.ensure(cu::ternary_scratch_bytes(in, tokens_max), s);
            const double packed_bytes = double(packed.size()), dense_bytes = double(dense.size()) * 2;
            for (int tokens : {1, 4, 8, 16, 32}) {
                if (tokens > tokens_max)
                    break;
                const double t_ms = median_ms(
                    [&] { cu::ternary_forward(w, 0.01f, input, y, in, out, scratch, s, tokens); }, 5, iterations, s);
                std::cout << out << "x" << in << ",ternary," << tokens << "," << std::setprecision(5) << t_ms << ","
                          << packed_bytes / (t_ms * 1e6) << "," << t_ms * 1e6 / tokens << "\n";
                const double d_ms = median_ms(
                    [&] {
                        for (int t = 0; t < tokens; ++t)
                            cu::dense_forward(wd, input.p + size_t(t) * in, y.p + size_t(t) * out, in, out, s);
                    },
                    5, iterations, s);
                std::cout << out << "x" << in << ",bf16," << tokens << "," << d_ms << ","
                          << dense_bytes * tokens / (d_ms * 1e6) << "," << d_ms * 1e6 / tokens << "\n";
            }
        }
        cg::checked(cg::stream_destroy(s), "stream");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
