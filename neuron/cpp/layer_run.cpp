// Runs ONE real layer of a Behemoth checkpoint on a pair of NeuronCores through the tensor-parallel FP8 layer NEFF
// (neuron/trace_layer_real.py), from C++ with no Python, and compares the result with the CPU reference block.
//
// Weights: QTIP projections are expanded at load to FP8 shards (qtip_expand.cpp); FP8 and NVFP4 projections are
// re-encoded; everything stays device-resident across steps. One decode step at position 0 with a fixed input.
//
// build: g++ -std=c++20 -O2 -fopenmp -pthread -I runtime/include -I third_party -I/opt/aws/neuron/include \
//            neuron/cpp/layer_run.cpp neuron/cpp/qtip_expand.cpp runtime/src/qtip.cpp runtime/src/checkpoint.cpp \
//            runtime/src/attention.cpp -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o layer_run
// run:   NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 layer_run MODEL_DIR LAYER graph.neff [steps]
#include "qtip_expand.hpp"
#include "behemoth/numeric.hpp"
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>

using namespace behemoth;

namespace {
constexpr int TP = 2, CONTEXT = 128;

void check(NRT_STATUS status, const char *what) {
    if (status != NRT_SUCCESS) {
        std::fprintf(stderr, "%s failed: status %d\n", what, int(status));
        std::exit(2);
    }
}

struct Blob {
    const void *data;
    size_t bytes;
};

// per-rank input data in the graph's ORDER (trace_layer_real.py)
struct RankInputs {
    std::vector<std::vector<uint8_t>> store;
    std::vector<Blob> by_position;
};
} // namespace

int main(int argc, char **argv) {
    if (argc < 4) { std::cerr << "usage: layer_run MODEL_DIR LAYER graph.neff [steps]\n"; return 2; }
    const std::filesystem::path root(argv[1]);
    const int layer = std::stoi(argv[2]);
    const int steps = argc > 4 ? std::atoi(argv[4]) : 32;
    auto config = ModelConfig::read(root / "config.json");
    char filename[64];
    std::snprintf(filename, sizeof filename, "model-layer-%05d.safetensors", layer);
    SafeTensorFile file(root / filename);
    const int hidden = config.hidden, dim = config.dim, kvheads = config.kv_heads / TP;

    auto t0 = std::chrono::steady_clock::now();
    struct P { const char *key, *name; bool col; };
    const P projs[] = {{"q", "self_attn.q_proj", true}, {"o", "self_attn.o_proj", false}, {"k", "self_attn.k_proj", true},
                       {"v", "self_attn.v_proj", true}, {"g", "mlp.gate_proj", true}, {"u", "mlp.up_proj", true},
                       {"d", "mlp.down_proj", false}};
    std::map<std::string, std::vector<neuron::Fp8Shard>> shards;
    for (const auto &p : projs) shards[p.key] = neuron::load_projection_shards(file, layer, p.name, config, TP, p.col);
    std::cout << "weights loaded/expanded in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";

    auto norm = [&](const std::string &n) {
        std::vector<float> v(size_t(hidden));
        auto view = file.floats("model.layers." + std::to_string(layer) + "." + n + ".weight");
        for (int i = 0; i < hidden; ++i) v[size_t(i)] = view[size_t(i)];
        return v;
    };
    const auto norm1 = norm("input_layernorm"), norm2 = norm("post_attention_layernorm");

    std::vector<float> x(size_t(hidden));
    std::vector<BFloat16> xb(size_t(hidden));
    for (int i = 0; i < hidden; ++i) { xb[size_t(i)] = BFloat16(.5f * std::sin(float(i) * .071f)); x[size_t(i)] = float(xb[size_t(i)]); }

    const int half = dim / 2;
    std::vector<float> inv(size_t(half));
    if (config.rope_llama3) inv = llama3_inv_freq(dim, config.theta, config.rope);
    else for (int d = 0; d < half; ++d) inv[size_t(d)] = float(std::pow(double(config.theta), -2.0 * d / dim));
    std::vector<float> cos_sin(size_t(CONTEXT) * 2 * size_t(half));
    for (int p = 0; p < CONTEXT; ++p)
        for (int d = 0; d < half; ++d) {
            cos_sin[(size_t(p) * 2 + 0) * size_t(half) + size_t(d)] = std::cos(float(p) * inv[size_t(d)]);
            cos_sin[(size_t(p) * 2 + 1) * size_t(half) + size_t(d)] = std::sin(float(p) * inv[size_t(d)]);
        }
    std::vector<float> mask(size_t(CONTEXT), -1e9f);
    mask[0] = 0;
    const int64_t slot = 0;
    std::vector<float> cache(size_t(CONTEXT) * size_t(kvheads) * size_t(dim), 0.f);

    // CPU reference (QTIP-correct layer block at position 0)
    std::vector<float> reference;
    {
        Checkpoint cp(root);
        auto yb = qtip::reference_block(cp, layer, xb);
        for (auto v : yb) reference.push_back(float(v));
    }

    // inputs per rank, in trace_layer_real.ORDER
    std::vector<RankInputs> ranks(TP);
    for (int r = 0; r < TP; ++r) {
        auto &ri = ranks[size_t(r)];
        auto add = [&](const void *p, size_t bytes) { ri.by_position.push_back({p, bytes}); };
        add(x.data(), x.size() * 4); add(&slot, 8); add(cos_sin.data(), cos_sin.size() * 4); add(mask.data(), mask.size() * 4);
        add(cache.data(), cache.size() * 4); add(cache.data(), cache.size() * 4);
        add(norm1.data(), norm1.size() * 4); add(norm2.data(), norm2.size() * 4);
        for (const char *k : {"q", "o", "k", "v", "g", "u", "d"}) {
            const auto &s = shards[k][size_t(r)];
            add(s.w.data(), s.w.size()); add(s.scale.data(), s.scale.size() * 4);
        }
    }

    std::ifstream in(argv[3], std::ios::binary);
    std::vector<char> neff((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init");
    std::barrier<> sync(TP);
    std::vector<float> result(size_t(hidden));
    std::vector<double> seconds(TP, 0);
    auto worker = [&](int r) {
        nrt_model_t *model = nullptr;
        check(nrt_load_collectives(neff.data(), neff.size(), r, -1, uint32_t(r), TP, &model), "nrt_load_collectives");
        nrt_tensor_info_array_t *info = nullptr;
        check(nrt_get_model_tensor_info(model, &info), "tensor info");
        nrt_tensor_set_t *ins = nullptr, *outs = nullptr;
        check(nrt_allocate_tensor_set(&ins), "set"); check(nrt_allocate_tensor_set(&outs), "set");
        nrt_tensor_t *out0 = nullptr;
        std::map<std::string, nrt_tensor_t *> by_name;
        for (uint64_t i = 0; i < info->tensor_count; ++i) {
            const auto &t = info->tensor_array[i];
            nrt_tensor_t *dev = nullptr;
            check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, r, t.size, t.name, &dev), "alloc");
            by_name[t.name] = dev;
            if (t.usage == NRT_TENSOR_USAGE_INPUT) {
                const std::string n = t.name;
                const size_t pos = std::stoul(n.substr(5));
                const Blob &b = ranks[size_t(r)].by_position.at(pos);
                if (b.bytes != t.size) {
                    std::fprintf(stderr, "rank %d: %s expects %zu bytes, host has %zu (graph ORDER mismatch)\n", r, t.name, t.size, b.bytes);
                    std::exit(3);
                }
                check(nrt_tensor_write(dev, b.data, 0, b.bytes), "write");
                check(nrt_add_tensor_to_tensor_set(ins, t.name, dev), "add in");
            } else {
                check(nrt_add_tensor_to_tensor_set(outs, t.name, dev), "add out");
                if (std::string(t.name) == "output0") out0 = dev;
            }
        }
        check(nrt_execute(model, ins, outs), "nrt_execute");
        if (r == 0) check(nrt_tensor_read(out0, result.data(), 0, result.size() * 4), "read");
        sync.arrive_and_wait();
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) check(nrt_execute(model, ins, outs), "nrt_execute");
        sync.arrive_and_wait();
        seconds[size_t(r)] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        nrt_unload(model);
    };
    std::vector<std::thread> threads;
    for (int r = 0; r < TP; ++r) threads.emplace_back(worker, r);
    for (auto &t : threads) t.join();

    double num = 0, den = 0, maxabs = 0;
    for (int i = 0; i < hidden; ++i) {
        const double d = double(result[size_t(i)]) - double(reference[size_t(i)]);
        num += d * d; den += double(reference[size_t(i)]) * double(reference[size_t(i)]);
        maxabs = std::max(maxabs, std::fabs(d));
    }
    std::printf("{\"layer\":%d,\"device_vs_reference_block_relative_l2\":%.4f,\"max_abs\":%.4f,\"ms_per_step\":%.3f}\n",
                layer, std::sqrt(num / den), maxabs, std::max(seconds[0], seconds[1]) / steps * 1000.0);
    nrt_close();
    return 0;
}
