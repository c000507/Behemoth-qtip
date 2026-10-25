#include "behemoth/attention.hpp"
#include "behemoth/checkpoint.hpp"
#include "behemoth/model.hpp"
#include <iostream>
#include <random>

int main(int argc, char **argv) {
    using namespace behemoth;
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: behemoth_block_tests MODEL_DIRECTORY [LAYER=0]\n";
            return 2;
        }
        int layer = argc > 2 ? std::stoi(argv[2]) : 0;
        RuntimeOptions options;
        options.model = argv[1];
        options.token_budget = 1;
        options.prefill_chunk = 1;
        TensorParallelModel gpu(options);
        Checkpoint checkpoint(options.model);
        std::vector<BFloat16> input(12288);
        std::mt19937 rng(101);
        std::normal_distribution<float> normal(0, .1f);
        for (auto &value : input)
            value = BFloat16(normal(rng));
        auto actual = gpu.block_for_validation(layer, input);
        std::cerr << "Computing independent scalar block reference; this performs about 1.38 "
                     "billion MACs\n";
        auto expected = reference_block(checkpoint, layer, input);
        std::vector<float> a(actual.size()), e(expected.size());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = float(actual[i]);
            e[i] = float(expected[i]);
        }
        auto error = compare_attention(a, e);
        std::cout << "block=" << layer << " rms=" << error.rms_error
                  << " reference_rms=" << error.reference_rms << " max_abs=" << error.max_abs
                  << " pass=" << error.pass << '\n';
        return error.pass ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "FAIL (block acceptance NOT passed): " << e.what() << '\n';
        return 1;
    }
}
