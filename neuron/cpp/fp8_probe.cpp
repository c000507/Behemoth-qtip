// Known-answer probe of the FP8 matmul graph (tests/trace_fp8_probe.py): row n of the weight is filled with the E4M3
// byte listed in kBytes, the activation is all ones, scales are 1. The ideal result is y[n] = 256 * decode(byte[n]).
// build: g++ -std=c++20 -O2 -I runtime/include -I/opt/aws/neuron/include neuron/cpp/fp8_probe.cpp -L/opt/aws/neuron/lib -lnrt -o fp8_probe
// run:   fp8_probe graph.neff
#include "behemoth/numeric.hpp"
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>
using namespace behemoth;
static void check(NRT_STATUS s, const char *w) { if (s != NRT_SUCCESS) { std::fprintf(stderr, "%s failed: %d\n", w, int(s)); std::exit(2); } }
int main(int argc, char **argv) {
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> neff((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "init");
    nrt_model_t *m = nullptr;
    check(nrt_load(neff.data(), neff.size(), 0, -1, &m), "load");
    const uint8_t kBytes[16] = {0x00, 0x08, 0x10, 0x20, 0x28, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x50, 0x58, 0x60, 0x68, 0x6e};
    std::vector<float> a(256, 1.0f), s(16, 1.0f);
    std::vector<uint8_t> w(16 * 256);
    for (int n = 0; n < 16; ++n) for (int k = 0; k < 256; ++k) w[size_t(n) * 256 + size_t(k)] = kBytes[n];
    nrt_tensor_info_array_t *info = nullptr;
    check(nrt_get_model_tensor_info(m, &info), "info");
    nrt_tensor_set_t *ins = nullptr, *outs = nullptr;
    check(nrt_allocate_tensor_set(&ins), "set"); check(nrt_allocate_tensor_set(&outs), "set");
    nrt_tensor_t *out = nullptr;
    for (uint64_t i = 0; i < info->tensor_count; ++i) {
        const auto &t = info->tensor_array[i];
        nrt_tensor_t *dev = nullptr;
        check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, t.size, t.name, &dev), "alloc");
        if (t.usage == NRT_TENSOR_USAGE_INPUT) {
            const void *src = std::string(t.name) == "input0" ? (void *)a.data() : std::string(t.name) == "input1" ? (void *)w.data() : (void *)s.data();
            check(nrt_tensor_write(dev, src, 0, t.size), "write");
            check(nrt_add_tensor_to_tensor_set(ins, t.name, dev), "in");
            std::printf("%s: %zu bytes\n", t.name, t.size);
        } else { check(nrt_add_tensor_to_tensor_set(outs, t.name, dev), "out"); out = dev; }
    }
    check(nrt_execute(m, ins, outs), "execute");
    std::vector<float> y(16);
    check(nrt_tensor_read(out, y.data(), 0, 16 * 4), "read");
    std::printf("byte  ideal(256*decode_e4m3)   device    ratio\n");
    for (int n = 0; n < 16; ++n) {
        const double ideal = 256.0 * double(decode_e4m3(kBytes[n]));
        std::printf("0x%02x  %14.6f  %14.6f  %8.4f\n", kBytes[n], ideal, y[size_t(n)], ideal != 0 ? y[size_t(n)] / ideal : 0.0);
    }
    return 0;
}
