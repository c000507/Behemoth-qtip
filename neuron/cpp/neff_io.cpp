// Runs a (tensor-parallel) NEFF once with inputs read from files and writes its outputs to files: a generic harness for
// device checks driven from Python (neuron/tests/test_qtip_tp_device.py).
//   DIR/in<i>.r<rank>   raw bytes of input<i> for that rank (DIR/in<i> when the same on every rank)
//   DIR/out<j>.r<rank>  raw bytes of output<j> of that rank, written by this tool
// build: g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include neuron/cpp/neff_io.cpp -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o neff_io
// run:   NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 neff_io graph.neff DIR [ranks]
// NEFF_IO_STATES=n: the graph's last n inputs are aliased state (outputs 1..n), bound to the same device tensors as
// in the runtime (a graph that writes its state and then reads it, like a KV cache, needs that).
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

static void check(NRT_STATUS s, const char *what, int rank) {
    if (s != NRT_SUCCESS) { std::fprintf(stderr, "rank %d: %s failed: %d\n", rank, what, int(s)); std::exit(2); }
}
static std::vector<char> slurp(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: neff_io graph.neff DIR [ranks]\n"); return 2; }
    const auto neff = slurp(argv[1]);
    const std::filesystem::path dir(argv[2]);
    const int ranks = argc > 3 ? std::atoi(argv[3]) : 2;
    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init", -1);
    std::vector<std::thread> threads;
    for (int r = 0; r < ranks; ++r)
        threads.emplace_back([&, r] {
            nrt_model_t *model = nullptr;
            if (ranks > 1)
                check(nrt_load_collectives(neff.data(), neff.size(), r, -1, uint32_t(r), uint32_t(ranks), &model), "load", r);
            else
                check(nrt_load(neff.data(), neff.size(), r, -1, &model), "load", r);
            nrt_tensor_info_array_t *info = nullptr;
            check(nrt_get_model_tensor_info(model, &info), "info", r);
            nrt_tensor_set_t *in = nullptr, *out = nullptr;
            check(nrt_allocate_tensor_set(&in), "set", r);
            check(nrt_allocate_tensor_set(&out), "set", r);
            struct Out { nrt_tensor_t *dev; std::string path; size_t bytes; };
            std::vector<Out> outputs;
            const int states = std::getenv("NEFF_IO_STATES") ? std::atoi(std::getenv("NEFF_IO_STATES")) : 0;
            int inputs = 0;
            for (uint64_t i = 0; i < info->tensor_count; ++i)
                inputs += info->tensor_array[i].usage == NRT_TENSOR_USAGE_INPUT;
            std::vector<nrt_tensor_t *> by_input(size_t(inputs), nullptr);
            for (int pass = 0; pass < 2; ++pass) // inputs first, so outputs can alias them
            for (uint64_t i = 0; i < info->tensor_count; ++i) {
                const auto &t = info->tensor_array[i];
                const std::string name = t.name;
                if ((t.usage == NRT_TENSOR_USAGE_INPUT) != (pass == 0))
                    continue;
                nrt_tensor_t *dev = nullptr;
                const int out_index = t.usage == NRT_TENSOR_USAGE_INPUT ? -1 : std::atoi(name.substr(6).c_str());
                if (out_index >= 1 && out_index <= states)
                    dev = by_input[size_t(inputs - states + out_index - 1)];
                else
                    check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, r, t.size, t.name, &dev), "alloc", r);
                if (t.usage == NRT_TENSOR_USAGE_INPUT) {
                    by_input[size_t(std::atoi(name.substr(5).c_str()))] = dev;
                    auto path = dir / (std::string("in") + name.substr(5) + ".r" + std::to_string(r));
                    if (!std::filesystem::exists(path)) path = dir / (std::string("in") + name.substr(5));
                    const auto data = slurp(path);
                    if (data.size() != t.size) { std::fprintf(stderr, "rank %d: %s has %zu bytes, NEFF wants %zu\n", r, path.c_str(), data.size(), size_t(t.size)); std::exit(2); }
                    check(nrt_tensor_write(dev, data.data(), 0, t.size), "write", r);
                    check(nrt_add_tensor_to_tensor_set(in, t.name, dev), "add in", r);
                } else {
                    check(nrt_add_tensor_to_tensor_set(out, t.name, dev), "add out", r);
                    outputs.push_back({dev, (dir / (std::string("out") + name.substr(6) + ".r" + std::to_string(r))).string(), size_t(t.size)});
                }
            }
            check(nrt_execute(model, in, out), "execute", r);
            for (auto &o : outputs) {
                std::vector<char> host(o.bytes);
                check(nrt_tensor_read(o.dev, host.data(), 0, o.bytes), "read", r);
                std::ofstream(o.path, std::ios::binary).write(host.data(), std::streamsize(o.bytes));
            }
        });
    for (auto &t : threads) t.join();
    std::printf("done\n");
    return 0;
}
