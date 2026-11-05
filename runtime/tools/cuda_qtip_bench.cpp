#include "behemoth/qtip.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

namespace cg = glm53::gpu;
namespace cu = behemoth::cuda_runtime;
using namespace behemoth;

namespace {
template <class T> void put(cg::Buffer<T> &d, const std::vector<T> &x, cg::Stream s) {
    d.ensure(x.size(), s);
    cg::checked(cg::h2d(d, x.data(), x.size() * sizeof(T), s), "benchmark upload");
}
struct Event {
    cudaEvent_t event{};
    Event() { cg::checked(cudaEventCreate(&event), "create timing event"); }
    ~Event() { (void)cudaEventDestroy(event); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
};
qtip::HostLinear synthetic(int out, int in) {
    std::mt19937 random(41);
    qtip::HostLinear h;
    h.in = in;
    h.out = out;
    h.trellis.resize(size_t(in) * out / 8);
    h.tlut.resize(1024);
    h.su.resize(in);
    h.sv.resize(out);
    for (auto &v : h.trellis)
        v = uint16_t(random());
    for (auto &v : h.tlut)
        v = uint16_t((random() & 0x8000) | (0x3000 + random() % 0x1000));
    for (auto &v : h.su)
        v = random() % 2 ? 1.f : -1.f;
    for (auto &v : h.sv)
        v = (random() % 2 ? 1.f : -1.f) * .003f;
    return h;
}
void bench(const qtip::HostLinear &h, const char *name, bool rz, int iterations, cg::Stream s) {
    if (h.is_fp8())
        throw std::runtime_error("Benchmark requires a QTIP projection");
    cg::Buffer<uint16_t> trellis, input, rotated, output;
    cg::Buffer<float> su, sv, scratch, z;
    cg::Buffer<uint32_t> lut;
    auto codebook = qtip::make_codebook(h.tlut);
    put(trellis, h.trellis, s);
    put(su, h.su, s);
    put(sv, h.sv, s);
    std::vector<uint32_t> table(codebook.bf16x2.begin(), codebook.bf16x2.end());
    put(lut, table, s);
    std::vector<uint16_t> x(h.in);
    std::mt19937 random(123);
    std::normal_distribution<float> normal(0, .1f);
    for (auto &v : x)
        v = BFloat16(normal(random)).bits;
    put(input, x, s);
    scratch.ensure(std::max(h.in, h.out), s);
    rotated.ensure(h.in, s);
    z.ensure(size_t(h.out) * 32, s);
    output.ensure(h.out, s);
    cg::Pinned<uint16_t> host;
    host.ensure(h.out);
    cg::checked(cg::sync(s), "finish benchmark upload");
    std::vector<uint16_t> reference;
    double baseline_ms = 0;
    Event start, end;
    for (int splits : {1, 2, 3, 4, 8, 16, 32}) {
        if (splits > h.in / 32)
            continue;
        auto run = [&] {
            cu::qtip_forward({trellis, su, sv, lut, h.in, h.out}, input, output, scratch, rotated, z, rz, s,
                             splits);
        };
        for (int i = 0; i < 5; ++i)
            run();
        cg::checked(cg::d2h(host.p, output, size_t(h.out) * 2, s), "read benchmark output");
        cg::checked(cg::sync(s), "finish benchmark warmup");
        if (splits == 1)
            reference.assign(host.p, host.p + h.out);
        double difference = 0, norm = 0;
        for (int i = 0; i < h.out; ++i) {
            float a = std::bit_cast<float>(uint32_t(host.p[i]) << 16);
            float b = std::bit_cast<float>(uint32_t(reference[i]) << 16);
            if (!std::isfinite(a) || !std::isfinite(b))
                throw std::runtime_error("Nonfinite benchmark output");
            difference += (double(a) - b) * (double(a) - b);
            norm += double(b) * b;
        }
        double relative = std::sqrt(difference / std::max(norm, 1e-30));
        if (relative >= .015)
            throw std::runtime_error("Split-K output differs excessively from the unsplit baseline");
        std::vector<float> samples;
        for (int sample = 0; sample < 5; ++sample) {
            cg::checked(cudaEventRecord(start.event, s), "record benchmark start");
            for (int i = 0; i < iterations; ++i)
                run();
            cg::checked(cudaEventRecord(end.event, s), "record benchmark end");
            cg::checked(cudaEventSynchronize(end.event), "wait for timing event");
            float ms = 0;
            cg::checked(cudaEventElapsedTime(&ms, start.event, end.event), "read timing event");
            samples.push_back(ms / iterations);
        }
        std::sort(samples.begin(), samples.end());
        double ms = samples[samples.size() / 2];
        if (splits == 1)
            baseline_ms = ms;
        // This includes both Hadamard transforms, not just the trellis decoder.
        std::cout << name << ',' << h.out << ',' << h.in << ',' << splits << ',' << ms << ','
                  << baseline_ms / ms << ',' << relative << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
    cg::Stream s{};
    try {
        int iterations = 20, device = 0;
        std::string model;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "behemoth_cuda_qtip_bench [--model PATH] [--device N] [--iterations N]\n";
                return 0;
            }
            if (i + 1 >= argc)
                throw std::runtime_error("Missing argument value; use --help");
            if (arg == "--model")
                model = argv[++i];
            else if (arg == "--device")
                device = std::stoi(argv[++i]);
            else if (arg == "--iterations")
                iterations = std::stoi(argv[++i]);
            else
                throw std::runtime_error("Unknown argument: " + arg);
        }
        if (iterations <= 0 || device < 0)
            throw std::runtime_error("Iterations must be positive; device must be nonnegative");
        int count = 0;
        auto status = cg::device_count(&count);
        if (status == cg::no_device || status == cg::insufficient_driver ||
            (status == cg::success && !count)) {
            std::cerr << "CUDA GPU unavailable; benchmark not run\n";
            return 77;
        }
        cg::checked(status, "GPU discovery");
        cg::checked(cg::set_device(device), "select benchmark GPU");
        cudaDeviceProp properties{};
        cg::checked(cudaGetDeviceProperties(&properties, device), "read benchmark GPU properties");
        std::cerr << properties.name << ", " << properties.multiProcessorCount << " SMs; "
                  << "median of 5 batches, " << iterations << " iterations/batch; "
                  << (model.empty() ? "synthetic weights" : "checkpoint layer 0 weights") << '\n';
        cg::checked(cg::stream_create(&s), "create benchmark stream");
        cu::initialize_tables();
        std::unique_ptr<Checkpoint> cp;
        if (!model.empty())
            cp = std::make_unique<Checkpoint>(model);
        std::cout << std::setprecision(8) << "projection,out,in,splits,median_ms,speedup,relative_l2\n";
        for (const char *name : {"self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj"}) {
            bool gate = std::string(name) == "mlp.gate_proj", down = std::string(name) == "mlp.down_proj";
            auto h =
                cp ? qtip::load_linear(*cp, 0, name) : synthetic(gate ? 28672 : 12288, down ? 28672 : 12288);
            bench(h, name, gate, iterations, s);
        }
        cg::checked(cg::stream_destroy(s), "destroy benchmark stream");
        s = nullptr;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (s)
            (void)cg::stream_destroy(s);
        return 1;
    }
}
