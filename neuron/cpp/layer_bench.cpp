// Loads a tensor-parallel NEFF with the Neuron runtime C API (libnrt) on every rank, allocates each
// rank's tensors once in its NeuronCore's device memory, and times steady-state nrt_execute calls
// (all ranks run concurrently, collectives inside the NEFF) with no Python in the loop.
//
// build: g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include neuron/cpp/layer_bench.cpp \
//            -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o layer_bench
// run:   layer_bench graph.neff [steps] [ranks]      (ranks defaults to 2: tensor parallelism is always on)
//
// The same NEFF is loaded on every rank (SPMD). Inputs are filled with small deterministic values
// (a timing harness; numerics are checked separately against the C++ oracle). Tensors of 8 bytes or
// less are the position index and are zeroed.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

static void check(NRT_STATUS status, const char* what, int rank) {
    if (status != NRT_SUCCESS) {
        std::fprintf(stderr, "rank %d: %s failed: status %d\n", rank, what, int(status));
        std::exit(2);
    }
}

struct RankResult {
    double load_s = 0, sec = 0;
    size_t in_bytes = 0, out_bytes = 0;
    int n_in = 0, n_out = 0;
};

static void run_rank(int rank, int ranks, const std::vector<char>& neff, int steps, std::barrier<>& sync,
                     RankResult& result, bool print_tensors) {
    nrt_model_t* model = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if (ranks > 1)
        check(nrt_load_collectives(neff.data(), neff.size(), rank, -1, uint32_t(rank), uint32_t(ranks), &model),
              "nrt_load_collectives", rank);
    else
        check(nrt_load(neff.data(), neff.size(), rank, -1, &model), "nrt_load", rank);
    result.load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    nrt_tensor_info_array_t* info = nullptr;
    check(nrt_get_model_tensor_info(model, &info), "nrt_get_model_tensor_info", rank);
    nrt_tensor_set_t *inputs = nullptr, *outputs = nullptr;
    check(nrt_allocate_tensor_set(&inputs), "nrt_allocate_tensor_set(in)", rank);
    check(nrt_allocate_tensor_set(&outputs), "nrt_allocate_tensor_set(out)", rank);
    std::vector<nrt_tensor_t*> tensors;
    for (uint64_t i = 0; i < info->tensor_count; ++i) {
        const nrt_tensor_info_t& t = info->tensor_array[i];
        nrt_tensor_t* dev = nullptr;
        check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, rank, t.size, t.name, &dev), "nrt_tensor_allocate", rank);
        tensors.push_back(dev);
        const bool is_input = t.usage == NRT_TENSOR_USAGE_INPUT;
        if (is_input) {
            std::vector<uint8_t> fill(t.size);
            if (t.size > 8)
                for (size_t k = 0; k < fill.size(); ++k) fill[k] = uint8_t((k * 2654435761u + rank) >> 24) & 0x3f;
            check(nrt_tensor_write(dev, fill.data(), 0, fill.size()), "nrt_tensor_write", rank);
            check(nrt_add_tensor_to_tensor_set(inputs, t.name, dev), "add input", rank);
            result.in_bytes += t.size;
            ++result.n_in;
        } else {
            check(nrt_add_tensor_to_tensor_set(outputs, t.name, dev), "add output", rank);
            result.out_bytes += t.size;
            ++result.n_out;
        }
        if (print_tensors)
            std::printf("  %-6s %-14s %12zu B dtype=%d ndim=%u\n", is_input ? "input" : "output", t.name, t.size,
                        int(t.dtype), t.ndim);
    }

    // Collectives rendezvous across ranks, so every rank must execute the same number of times.
    for (int i = 0; i < 4; ++i) check(nrt_execute(model, inputs, outputs), "nrt_execute(warmup)", rank);
    sync.arrive_and_wait();
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) check(nrt_execute(model, inputs, outputs), "nrt_execute", rank);
    sync.arrive_and_wait();
    result.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    nrt_destroy_tensor_set(&inputs);
    nrt_destroy_tensor_set(&outputs);
    for (auto* t : tensors) nrt_tensor_free(&t);
    nrt_free_model_tensor_info(info);
    nrt_unload(model);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s graph.neff [steps] [ranks]\n", argv[0]);
        return 1;
    }
    const int steps = argc > 2 ? std::atoi(argv[2]) : 64;
    const int ranks = argc > 3 ? std::atoi(argv[3]) : 2;
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> neff((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (neff.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init", -1);

    std::barrier<> sync(ranks);
    std::vector<RankResult> results(ranks);
    std::vector<std::thread> threads;
    for (int r = 0; r < ranks; ++r)
        threads.emplace_back(run_rank, r, ranks, std::cref(neff), steps, std::ref(sync), std::ref(results[r]), r == 0);
    for (auto& t : threads) t.join();

    const RankResult& r0 = results[0];
    double sec = 0;
    for (auto& r : results) sec = std::max(sec, r.sec);
    std::printf("{\"neff\":\"%s\",\"ranks\":%d,\"load_s\":%.2f,\"inputs\":%d,\"outputs\":%d,\"input_MB_per_rank\":%.1f,"
                "\"steps\":%d,\"ms_per_step\":%.3f,\"steps_per_s\":%.1f}\n",
                argv[1], ranks, r0.load_s, r0.n_in, r0.n_out, r0.in_bytes / 1e6, steps, sec / steps * 1000.0, steps / sec);
    nrt_close();
    return 0;
}
