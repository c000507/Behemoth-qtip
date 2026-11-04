#include "glm53.hpp"
#include "safetensors.hpp"
#include <cmath>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <unistd.h>
using namespace glm53;
int main() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0)
        return 77;
    auto root = fs::temp_directory_path() / ("glm53-gpu-" + std::to_string(getpid()));
    try {
        fs::create_directory(root);
        constexpr int rows = 128, cols = 256;
        qtip::SafeTensorsWriter writer;
        json map = json::object();
        auto add = [&](const std::string &name, const std::string &dtype, std::vector<int64_t> shape,
                       const void *data, size_t bytes) {
            writer.add(name, dtype, shape, data, bytes);
            map[name] = "model.safetensors";
        };
        Vec zero(cols);
        add("model.embed_tokens.weight", "F32", {1, cols}, zero.data(), zero.size() * 4);
        Vec dense(size_t(rows) * cols), row_scales(rows);
        std::vector<uint16_t> bf16(dense.size());
        std::vector<uint8_t> fp8(dense.size());
        for (int r = 0; r < rows; ++r) {
            size_t at = size_t(r) * cols + (r * 7) % cols;
            dense[at] = float((r % 13) + 1) / 16;
            bf16[at] = behemoth::BFloat16(dense[at]).bits;
            fp8[at] = behemoth::encode_e4m3(r % 2 ? -1.f : 1.f);
            row_scales[r] = float(1 + r % 7) / 8;
        }
        add("probe.f32.weight", "F32", {rows, cols}, dense.data(), dense.size() * 4);
        add("probe.bf16.weight", "BF16", {rows, cols}, bf16.data(), bf16.size() * 2);
        add("probe.channel.weight", "F8_E4M3", {rows, cols}, fp8.data(), fp8.size());
        add("probe.channel.weight_scale", "F32", {rows}, row_scales.data(), row_scales.size() * 4);
        Vec block_scales{.5f, 2.f};
        add("probe.block.weight", "F8_E4M3", {rows, cols}, fp8.data(), fp8.size());
        add("probe.block.weight_scale_inv", "F32", {1, 2}, block_scales.data(), block_scales.size() * 4);
        auto nv = quantize_nvfp4(dense, rows, cols);
        add("probe.nv.weight_packed", "U8", {rows, cols / 2}, nv.packed.data(), nv.packed.size());
        add("probe.nv.weight_scale", "F8_E4M3", {rows, cols / 16}, nv.scales.data(), nv.scales.size());
        add("probe.nv.weight_global_scale", "F32", {}, &nv.global, 4);
        writer.write((root / "model.safetensors").string());
        {
            std::ofstream f(root / "model.safetensors.index.json");
            f << json({{"weight_map", map}});
        }
        {
            std::ofstream f(root / "config.json");
            f << json(
                {{"glm53_quant", {{"mlp_format", "nvfp4"}, {"block_size", 16}, {"format_version", 1}}}});
        }
        {
            Weights w(root);
            auto project = gpu_projector(0);
            Vec x(cols);
            for (int i = 0; i < cols; ++i)
                x[i] = float(i % 31 - 15) / 16;
            for (auto base : {"probe.f32", "probe.bf16", "probe.channel", "probe.block", "probe.nv"}) {
                auto cpu = w.linear(base);
                auto y = project(w, base, x);
                for (int r = 0; r < rows; ++r) {
                    double expected = 0;
                    for (int c = 0; c < cols; ++c)
                        expected += double(float(behemoth::BFloat16(cpu.data[size_t(r) * cols + c]))) *
                                    float(behemoth::BFloat16(x[c]));
                    check(std::isfinite(y[r]) && std::abs(y[r] - expected) < 1e-6,
                          "GPU weight decode/projection mismatch: " + std::string(base));
                }
            }
        }
        fs::remove_all(root);
        std::cout << "PASS GPU BF16/F32/FP8-channel/FP8-block/NVFP4 streamed projections\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << " fixture " << root << '\n';
        return 1;
    }
}
