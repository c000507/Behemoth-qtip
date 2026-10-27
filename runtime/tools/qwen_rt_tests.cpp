// Qwen runtime linear-path check: trellis GEMV pipeline (input_transform -> matmul -> slice sum ->
// hadU -> SV) vs qtip::reference_forward with device rounding, on a real quantized linear.
// usage: qwen_rt_tests <model_dir> [layer] [name]
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "behemoth/qtip.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/qwen_kernels.hpp"
#include "behemoth/qwen_model.hpp"

using namespace behemoth;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qwen_rt_tests <model_dir> [layer] [name]\n");
        return 1;
    }
    std::filesystem::path dir = argv[1];
    int layer = argc >= 3 ? std::atoi(argv[2]) : 0;
    std::string name = argc >= 4 ? argv[3] : "mlp.gate_proj";
    QwenRuntimeConfig cfg = QwenRuntimeConfig::read(dir);
    // load the host linear straight from the layer file via behemoth::Checkpoint-like access
    qtip::HostLinear h;
    int m = 0;
    {
        std::string fn = "model-layer-" + std::string(layer < 10 ? "0000" : "000") + std::to_string(layer) +
                         ".safetensors";
        SafeTensorFile f(dir / fn);
        std::string base = "model.layers." + std::to_string(layer) + "." + name + ".";
        auto rd = [&](const std::string &sfx) { return f.bytes(base + sfx); };
        auto tre = rd("trellis");
        h.trellis.resize(tre.size() / 2);
        std::memcpy(h.trellis.data(), tre.data(), tre.size());
        auto su_v = f.floats(base + "SU");
        auto sv_v = f.floats(base + "SV");
        h.su.assign(su_v.begin(), su_v.end());
        h.sv.assign(sv_v.begin(), sv_v.end());
        auto tl = rd("tlut");
        h.tlut.resize(tl.size() / 2);
        std::memcpy(h.tlut.data(), tl.data(), tl.size());
        h.in = int(h.su.size());
        h.out = int(h.sv.size());
        m = h.out;
    }
    (void)cfg;
    const int rows = 4, n = h.in;
    qtip::Codebook cb = qtip::make_codebook(std::span<const uint16_t>(h.tlut));
    qtip::GpuCodebook lut;
    lut.upload(cb);
    qtip::GpuLinear gl;
    gl.upload({std::span<const uint16_t>(h.trellis)}, n);
    DeviceBuffer<float> su, sv;
    su.upload(h.su);
    sv.upload(h.sv);
    std::vector<float> xf(size_t(rows) * n);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    for (auto &v : xf) v = nd(rng) * 0.5f;
    std::vector<uint16_t> xb(size_t(rows) * n);
    for (size_t i = 0; i < xb.size(); ++i) {
        float f = xf[i];
        uint32_t u;
        std::memcpy(&u, &f, 4);
        xb[i] = uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        xf[i] = float(BFloat16(f));  // emulate the bf16 input the runtime sees
    }
    hipStream_t s;
    HIP_CHECK(hipStreamCreate(&s));
    DeviceBuffer<uint16_t> d_x, d_u;
    d_x.upload(xb);
    d_u.resize(size_t(rows) * n);
    DeviceBuffer<float> partial, z, z2;
    int slices = qtip::matmul_slices(gl.view(), rows);
    partial.resize(size_t(slices) * rows * m);
    z.resize(size_t(rows) * m);
    z2.resize(size_t(rows) * m);
    DeviceBuffer<uint16_t> d_y(size_t(rows) * m);
    // runtime pipeline
    q_input_transform(d_x.data(), rows, n, nullptr, 0, su.data(), n, d_u.data(), n, s);
    qtip::matmul(gl.view(), lut.lut.data(), reinterpret_cast<const BFloat16 *>(d_u.data()), n, rows,
                 partial.data(), slices, s);
    q_sum_slices(partial.data(), slices, size_t(rows) * m, z.data(), s);
    q_hadamard_rows(z.data(), rows, m, m, false, s);
    q_sv_cast(z.data(), sv.data(), d_y.data(), rows, m, m, s);
    auto gpu = d_y.download();
    // reference (device rounding: bf16 rotated input, bf16 LUT values)
    auto ref = qtip::reference_forward(h, xf, true);
    double maxd = 0, scale = 1e-9;
    for (size_t i = 0; i < ref.size(); ++i) {
        BFloat16 b;
        b.bits = gpu[i];
        double g = float(b);
        maxd = std::max(maxd, std::fabs(g - ref[i]));
        scale = std::max(scale, std::fabs(ref[i]));
    }
    std::printf("linear %s L%d: %dx%d slices %d max|diff| %.5f scale %.3f rel %.2e -> %s\n", name.c_str(),
                layer, m, n, slices, maxd, scale, maxd / scale, maxd / scale < 2e-2 ? "ok" : "FAIL");
    return maxd / scale < 2e-2 ? 0 : 1;
}
