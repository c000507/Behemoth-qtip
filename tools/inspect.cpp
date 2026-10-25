#include "behemoth/cache.hpp"
#include "behemoth/checkpoint.hpp"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: behemoth_inspect MODEL_DIRECTORY\n";
            return 2;
        }
        behemoth::Checkpoint c(argv[1]);
        for (int layer = 0; layer < 88; ++layer)
            for (const char *name :
                 {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
                auto key = "model.layers." + std::to_string(layer) + "." + name + ".codebooks";
                if (c.tensor(key).shape(key) != std::vector<size_t>{1, 65536, 1, 8})
                    throw std::runtime_error("Bad codebook: " + key);
            }
        std::cout << "Validated 88 blocks / 616 AQLM projections\nFP8 KV bytes per token per GPU "
                     "(including scales): "
                  << behemoth::PagePool::bytes_per_page() / 16 << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
