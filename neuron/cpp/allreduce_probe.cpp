// Known-answer all-reduce on two NeuronCores: rank r supplies the value (r + 1) in every element; the NEFF
// (tests/trace_allreduce_probe.py) should return 3 on both ranks.
// build: g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include neuron/cpp/allreduce_probe.cpp -L/opt/aws/neuron/lib -lnrt -o allreduce_probe
// run:   NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 allreduce_probe graph.neff [f32|bf16]
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
static void check(NRT_STATUS s, const char *w) { if (s != NRT_SUCCESS) { std::fprintf(stderr, "%s failed: %d\n", w, int(s)); std::exit(2); } }
int main(int argc, char **argv) {
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> neff((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const bool bf16 = argc > 2 && std::string(argv[2]) == "bf16";
    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "init");
    std::barrier<> sync(2);
    auto worker = [&](int r) {
        nrt_model_t *m = nullptr;
        check(nrt_load_collectives(neff.data(), neff.size(), r, -1, uint32_t(r), 2, &m), "load");
        nrt_tensor_info_array_t *info = nullptr;
        check(nrt_get_model_tensor_info(m, &info), "info");
        nrt_tensor_set_t *ins = nullptr, *outs = nullptr;
        check(nrt_allocate_tensor_set(&ins), "set"); check(nrt_allocate_tensor_set(&outs), "set");
        nrt_tensor_t *out = nullptr; size_t out_bytes = 0;
        for (uint64_t i = 0; i < info->tensor_count; ++i) {
            const auto &t = info->tensor_array[i];
            nrt_tensor_t *dev = nullptr;
            check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, r, t.size, t.name, &dev), "alloc");
            if (t.usage == NRT_TENSOR_USAGE_INPUT) {
                std::vector<uint8_t> raw(t.size);
                const float v = float(r + 1);
                for (size_t k = 0; k < 8; ++k) {
                    if (bf16) { uint32_t b; std::memcpy(&b, &v, 4); uint16_t h = uint16_t(b >> 16); std::memcpy(&raw[k * 2], &h, 2); }
                    else std::memcpy(&raw[k * 4], &v, 4);
                }
                check(nrt_tensor_write(dev, raw.data(), 0, t.size), "write");
                check(nrt_add_tensor_to_tensor_set(ins, t.name, dev), "in");
            } else { check(nrt_add_tensor_to_tensor_set(outs, t.name, dev), "out"); out = dev; out_bytes = t.size; }
        }
        for (int rep = 0; rep < 3; ++rep) {
            check(nrt_execute(m, ins, outs), "execute");
            std::vector<uint8_t> raw(out_bytes);
            check(nrt_tensor_read(out, raw.data(), 0, out_bytes), "read");
            float y;
            if (bf16) { uint16_t h; std::memcpy(&h, raw.data(), 2); uint32_t b = uint32_t(h) << 16; std::memcpy(&y, &b, 4); }
            else std::memcpy(&y, raw.data(), 4);
            std::printf("rank %d run %d: element0 = %g (expected 3)\n", r, rep, y);
            sync.arrive_and_wait();
        }
    };
    std::thread a(worker, 0), b(worker, 1);
    a.join(); b.join();
    return 0;
}
