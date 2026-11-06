// Loads a NEFF with the Neuron runtime C API (libnrt), allocates every tensor once in device
// memory, and times steady-state nrt_execute calls with no Python in the loop.
//
// build: g++ -std=c++20 -O2 -I/opt/aws/neuron/include neuron/cpp/layer_bench.cpp \
//            -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o layer_bench
// run:   layer_bench graph.neff [steps] [vnc]
//
// Inputs are filled with small deterministic values (this is a timing harness; numerics are
// checked separately against the C++ oracle). Output tensors are allocated once and reused.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void check(NRT_STATUS status, const char* what) {
    if (status != NRT_SUCCESS) {
        std::fprintf(stderr, "%s failed: status %d\n", what, int(status));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s graph.neff [steps] [vnc]\n", argv[0]);
        return 1;
    }
    const int steps = argc > 2 ? std::atoi(argv[2]) : 64;
    const int vnc = argc > 3 ? std::atoi(argv[3]) : 0;
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> neff((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (neff.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }

    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init");
    nrt_model_t* model = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    check(nrt_load(neff.data(), neff.size(), vnc, 1, &model), "nrt_load");
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    nrt_tensor_info_array_t* info = nullptr;
    check(nrt_get_model_tensor_info(model, &info), "nrt_get_model_tensor_info");

    nrt_tensor_set_t* inputs = nullptr;
    nrt_tensor_set_t* outputs = nullptr;
    check(nrt_allocate_tensor_set(&inputs), "nrt_allocate_tensor_set(in)");
    check(nrt_allocate_tensor_set(&outputs), "nrt_allocate_tensor_set(out)");

    size_t in_bytes = 0, out_bytes = 0;
    int n_in = 0, n_out = 0;
    std::vector<nrt_tensor_t*> tensors;
    for (uint64_t i = 0; i < info->tensor_count; ++i) {
        const nrt_tensor_info_t& t = info->tensor_array[i];
        nrt_tensor_t* dev = nullptr;
        check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, vnc, t.size, t.name, &dev), "nrt_tensor_allocate");
        tensors.push_back(dev);
        if (t.usage == NRT_TENSOR_USAGE_INPUT) {
            std::vector<uint8_t> fill(t.size);
            // Tiny tensors are the int64 slot/position index: keep it a valid index (0). Everything
            // else gets small bytes (finite floats, valid packed codes).
            if (t.size > 8)
                for (size_t k = 0; k < fill.size(); ++k) fill[k] = uint8_t((k * 2654435761u) >> 24) & 0x3f;
            check(nrt_tensor_write(dev, fill.data(), 0, fill.size()), "nrt_tensor_write");
            check(nrt_add_tensor_to_tensor_set(inputs, t.name, dev), "add input");
            in_bytes += t.size;
            ++n_in;
        } else {
            check(nrt_add_tensor_to_tensor_set(outputs, t.name, dev), "add output");
            out_bytes += t.size;
            ++n_out;
        }
        std::printf("  %-8s %-40s %10zu B dtype=%d ndim=%u\n", t.usage == NRT_TENSOR_USAGE_INPUT ? "input" : "output",
                    t.name, t.size, int(t.dtype), t.ndim);
    }

    // warm-up, then steady state
    for (int i = 0; i < 4; ++i) check(nrt_execute(model, inputs, outputs), "nrt_execute(warmup)");
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) check(nrt_execute(model, inputs, outputs), "nrt_execute");
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    std::printf("{\"neff\":\"%s\",\"load_s\":%.2f,\"inputs\":%d,\"outputs\":%d,\"input_MB\":%.1f,\"output_MB\":%.1f,"
                "\"steps\":%d,\"ms_per_step\":%.3f,\"steps_per_s\":%.1f}\n",
                argv[1], load_s, n_in, n_out, in_bytes / 1e6, out_bytes / 1e6, steps, sec / steps * 1000.0,
                steps / sec);

    nrt_destroy_tensor_set(&inputs);
    nrt_destroy_tensor_set(&outputs);
    for (auto* t : tensors) nrt_tensor_free(&t);
    nrt_free_model_tensor_info(info);
    nrt_unload(model);
    nrt_close();
    return 0;
}
