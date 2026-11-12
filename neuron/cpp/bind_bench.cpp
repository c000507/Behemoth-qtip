// Where does a launch's host-side time go when consecutive launches use different tensor sets / different models?
// (The engine runs every layer through the same NEFF with that layer's weights: a different tensor set per launch.)
//   bind_bench STEPS RANKS MODE K neffA [neffB ...]
//   MODE sets:   one loaded instance per NEFF, K tensor sets each (distinct device tensors), cycled
//   MODE models: K loaded instances per NEFF, one tensor set each, cycled
// A step runs every NEFF once (A then B ...), so "sets 1" with two NEFFs measures pure model alternation.
// build: g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include bind_bench.cpp -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o bind_bench
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

static void check(NRT_STATUS s, const char *what, int rank) {
    if (s != NRT_SUCCESS) { std::fprintf(stderr, "rank %d: %s failed: %d\n", rank, what, int(s)); std::exit(2); }
}
struct Unit { nrt_model_t *model; nrt_tensor_set_t *in, *out; nrt_tensor_t *small = nullptr; size_t small_bytes = SIZE_MAX; };

int main(int argc, char **argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: bind_bench STEPS RANKS sets|models K neff...\n"); return 2; }
    const int steps = std::atoi(argv[1]), ranks = std::atoi(argv[2]), K = std::atoi(argv[4]);
    const bool per_model = std::string(argv[3]) == "models";
    std::vector<std::vector<char>> neffs;
    for (int i = 5; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary);
        neffs.emplace_back((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init", -1);
    std::barrier<> sync(ranks);
    std::vector<double> sec(size_t(ranks), 0), load(size_t(ranks), 0);
    std::vector<std::thread> threads;
    for (int r = 0; r < ranks; ++r)
        threads.emplace_back([&, r] {
            std::vector<std::vector<Unit>> units(neffs.size()); // [neff][k]
            const auto tl = std::chrono::steady_clock::now();
            for (int k = 0; k < K; ++k)
                for (size_t n = 0; n < neffs.size(); ++n) {
                    Unit u{};
                    if (per_model || k == 0) {
                        if (ranks > 1)
                            check(nrt_load_collectives(neffs[n].data(), neffs[n].size(), r, -1, uint32_t(r), uint32_t(ranks), &u.model), "load", r);
                        else
                            check(nrt_load(neffs[n].data(), neffs[n].size(), r, -1, &u.model), "load", r);
                    } else
                        u.model = units[n][0].model;
                    nrt_tensor_info_array_t *info = nullptr;
                    check(nrt_get_model_tensor_info(u.model, &info), "info", r);
                    check(nrt_allocate_tensor_set(&u.in), "set", r);
                    check(nrt_allocate_tensor_set(&u.out), "set", r);
                    for (uint64_t i = 0; i < info->tensor_count; ++i) {
                        const auto &t = info->tensor_array[i];
                        nrt_tensor_t *dev = nullptr;
                        check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, r, t.size, t.name, &dev), "alloc", r);
                        if (t.usage == NRT_TENSOR_USAGE_INPUT) {
                            std::vector<uint8_t> fill(t.size);
                            if (t.size > 8 && int(t.dtype) != 8)
                                for (size_t j = 0; j < fill.size(); ++j) fill[j] = uint8_t((j * 2654435761u + r) >> 24) & 0x3f;
                            check(nrt_tensor_write(dev, fill.data(), 0, fill.size()), "write", r);
                            check(nrt_add_tensor_to_tensor_set(u.in, t.name, dev), "add", r);
                        } else {
                            check(nrt_add_tensor_to_tensor_set(u.out, t.name, dev), "add", r);
                            if (t.size < u.small_bytes) { u.small_bytes = t.size; u.small = dev; }
                        }
                    }
                    nrt_free_model_tensor_info(info);
                    units[n].push_back(u);
                }
            load[size_t(r)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();
            auto step = [&](int i) {
                for (size_t n = 0; n < neffs.size(); ++n) {
                    Unit &u = units[n][size_t(i % K)];
                    check(nrt_execute(u.model, u.in, u.out), "execute", r);
                }
            };
            for (int i = 0; i < 2 * K; ++i) step(i);
            sync.arrive_and_wait();
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < steps; ++i) step(i);
            std::vector<uint8_t> sink(units[0][0].small_bytes);
            check(nrt_tensor_read(units.back()[size_t((steps - 1) % K)].small, sink.data(), 0, 4), "drain", r);
            sync.arrive_and_wait();
            sec[size_t(r)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        });
    for (auto &t : threads) t.join();
    double s = 0;
    for (double v : sec) s = std::max(s, v);
    std::printf("{\"mode\":\"%s\",\"K\":%d,\"neffs\":%zu,\"ranks\":%d,\"load_s\":%.2f,\"ms_per_launch\":%.3f}\n", argv[3], K, neffs.size(), ranks, load[0],
                s / steps / double(neffs.size()) * 1000.0);
    std::fflush(stdout);
    std::_Exit(0);
}
