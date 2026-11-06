// Prints the input/output tensors of a NEFF (names, sizes, usage) as libnrt sees them.
// build: g++ -std=c++20 -O1 -I/opt/aws/neuron/include neff_info.cpp -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o neff_info
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> neff((std::istreambuf_iterator<char>(f)), {});
    if (nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", "") != NRT_SUCCESS) return 3;
    nrt_model_t *model = nullptr;
    if (nrt_load(neff.data(), neff.size(), 0, 1, &model) != NRT_SUCCESS) { std::puts("load failed"); return 4; }
    nrt_tensor_info_array_t *info = nullptr;
    nrt_get_model_tensor_info(model, &info);
    for (uint64_t i = 0; i < info->tensor_count; ++i) {
        const auto &t = info->tensor_array[i];
        std::printf("%3llu %-8s %-24s %12zu B\n", (unsigned long long)i, t.usage == NRT_TENSOR_USAGE_INPUT ? "input" : "output", t.name, t.size);
    }
    nrt_unload(model);
    nrt_close();
}
