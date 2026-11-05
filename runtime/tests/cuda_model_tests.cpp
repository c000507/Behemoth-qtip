#include "behemoth/cuda.hpp"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
using namespace behemoth::cuda_runtime;
namespace {
void identical(const Vec &a, const Vec &b, const char *message) {
    check(!a.empty() && a.size() == b.size(), "Model logits size changed");
    for (float value : a)
        check(std::isfinite(value), "Nonfinite model logits");
    check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, message);
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "behemoth_cuda_model_tests --model DIR\n";
            return 0;
        }
        check(argc == 3 && std::string(argv[1]) == "--model", "Expected --model DIR");
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count < 2))
            return 77;
        check(status == cudaSuccess, cudaGetErrorString(status));
        Model model(argv[2], 0, 2048, 2, 1);
        std::vector<int> short_prompt(32), long_prompt(1024);
        for (int i = 0; i < 32; ++i)
            short_prompt[i] = i + 1;
        for (int i = 0; i < 1024; ++i)
            long_prompt[i] = i % 127 + 1;
        auto run_short = [&] {
            model.reset();
            Vec prefill = model.prefill(short_prompt);
            Vec decode = model.step(33);
            return std::pair{std::move(prefill), std::move(decode)};
        };
        // A large earlier request used to hide the under-allocation in a cold
        // 64-row prefill: the small matmul still writes sixteen split partials.
        auto cold = run_short();
        model.reset();
        (void)model.prefill(long_prompt);
        auto warm = run_short();
        identical(cold.first, warm.first, "Cold short prefill differs after arena growth/reset");
        identical(cold.second, warm.second, "Cold short decode/cache differs after arena growth/reset");
        std::cout << "PASS cold short prefill, split workspace, arena growth and sequence reset\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
