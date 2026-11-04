#include "device_ops.hpp"
#include "glm53.hpp"
#include "gpu_runtime.hpp"
#include "safetensors.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace glm53;
void compare_model(const fs::path &path) {
    Model cpu(path), device(path);
    device.set_backend(gpu_backend(path, 0));
    std::vector<Vec> replay;
    double total_error = 0, total_signal = 0;
    for (int t = 0; t < 33; ++t) {
        int token = 1 + t % 25;
        auto expected = cpu.step(token), actual = device.step(token);
        check(expected.size() == actual.size(), "GPU model vocabulary mismatch");
        double error = 0, signal = 0, maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            check(std::isfinite(actual[i]), "nonfinite GPU model output");
            error += std::pow(double(expected[i]) - actual[i], 2);
            signal += double(expected[i]) * expected[i];
            maximum = std::max(maximum, std::abs(double(expected[i]) - actual[i]));
        }
        double relative = std::sqrt(error / std::max(signal, 1e-30));
        if (relative >= .02 || maximum >= .03)
            std::cerr << path << " position " << t << " relative L2 " << relative << " max absolute "
                      << maximum << '\n';
        check(relative < .02 && maximum < .03, "GPU/CPU BF16 model logits differ");
        total_error += error;
        total_signal += signal;
        replay.push_back(actual);
    }
    device.reset();
    for (int t = 0; t < 33; ++t)
        check(device.step(1 + t % 25) == replay[t], "GPU state leaked across reset");
    std::cout << "PASS GPU model " << path << " relative L2 " << std::sqrt(total_error / total_signal)
              << '\n';
}
void select_test() {
    using namespace gpu;
    Stream stream{};
    checked(stream_create(&stream), "test stream");
    try {
        Buffer<float> scores;
        Buffer<int> ids, selected;
        Pinned<int> result;
        for (int pools : {0, 1, 2, 17, 1000, 1025, 5003}) {
            int padded = 1;
            while (padded < pools)
                padded *= 2;
            Vec values(std::max(1024, padded), -INFINITY);
            std::vector<int> order(pools);
            for (int i = 0; i < pools; ++i) {
                values[i] = float((i * 97) % 701 - 350);
                order[i] = i;
            }
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return values[a] == values[b] ? a < b : values[a] > values[b]; });
            scores.ensure(values.size(), stream);
            ids.ensure(values.size(), stream);
            selected.ensure(35, stream);
            result.ensure(35);
            checked(h2d(scores, values.data(), values.size() * 4, stream), "upload index scores");
            int count = std::min(pools, 8), position = pools * 4 + 2, n = count * 4 + 3;
            select_keys(scores, ids, selected, pools, padded, 4, 32, position, stream);
            checked(d2h(result.p, selected, size_t(n) * sizeof(int), stream), "read sparse selection");
            checked(sync(stream), "index selection sync");
            for (int i = 0; i < count * 4; ++i)
                check(result.p[i] == order[i / 4] * 4 + i % 4, "GPU top-k ordering/tie break");
            for (int i = 0; i < 3; ++i)
                check(result.p[count * 4 + i] == pools * 4 + i, "GPU sparse attention omitted causal tail");
        }
    } catch (...) {
        (void)stream_destroy(stream);
        throw;
    }
    checked(stream_destroy(stream), "destroy test stream");
    std::cout << "PASS GPU sparse top-k partial/global bitonic merges, ties and tail\n";
}
void kda_test(int dim = 2, int heads = 1) {
    using namespace gpu;
    Stream stream{};
    checked(stream_create(&stream), "KDA test stream");
    try {
        Buffer<B16> q, k, v, g, b, y;
        Buffer<float> state, dt, a, values;
        Pinned<float> output;
        int width = dim * heads, memory = width * dim;
        q.ensure(width, stream);
        k.ensure(width, stream);
        v.ensure(width, stream);
        g.ensure(width, stream);
        b.ensure(heads, stream);
        y.ensure(width, stream);
        values.ensure(width, stream);
        state.ensure(memory, stream);
        dt.ensure(width, stream);
        a.ensure(heads, stream);
        output.ensure(memory + width);
        Vec zero(width);
        checked(gpu::zero(state, size_t(memory) * 4, stream), "clear KDA test");
        checked(h2d(dt, zero.data(), size_t(width) * 4, stream), "KDA dt");
        checked(h2d(a, zero.data(), size_t(heads) * 4, stream), "KDA A");
        Vec cpu_state;
        for (int step = 0; step < 3; ++step) {
            Vec qv(width), vv(width);
            std::vector<B16> qb(width), vb(width), gb(width), bb(heads);
            for (int h = 0; h < heads; ++h)
                qv[h * dim + step % dim] = 1;
            for (int i = 0; i < width; ++i) {
                vv[i] = float((i + step) % 17) / 4;
                qb[i] = behemoth::BFloat16(qv[i]).bits;
                vb[i] = behemoth::BFloat16(vv[i]).bits;
            }
            checked(h2d(q, qb.data(), size_t(width) * 2, stream), "KDA q");
            checked(h2d(k, qb.data(), size_t(width) * 2, stream), "KDA k");
            checked(h2d(v, vb.data(), size_t(width) * 2, stream), "KDA v");
            checked(h2d(g, gb.data(), size_t(width) * 2, stream), "KDA g");
            checked(h2d(b, bb.data(), size_t(heads) * 2, stream), "KDA beta");
            gpu::kda(q, k, v, g, b, dt, a, state, y, heads, dim, -5, stream);
            checked(d2h(output.p, state, size_t(memory) * 4, stream), "read KDA state");
            gpu::to_float(y, values, width, stream);
            checked(d2h(output.p + memory, values, size_t(width) * 4, stream), "read KDA output");
            checked(sync(stream), "KDA sync");
            auto expected = kda_step(qv, qv, vv, Vec(width, -2.5f), Vec(heads, .5f), cpu_state, heads, dim);
            for (int i = 0; i < memory; ++i)
                check(std::abs(output.p[i] - cpu_state[i]) < 2e-5, "GPU KDA recurrence/decay mismatch");
            for (int i = 0; i < width; ++i)
                check(std::abs(output.p[memory + i] - float(behemoth::BFloat16(expected[i]))) < .002,
                      "GPU KDA query scaling/output mismatch");
        }
    } catch (...) {
        (void)stream_destroy(stream);
        throw;
    }
    checked(stream_destroy(stream), "destroy KDA test stream");
    std::cout << "PASS GPU KDA persistent delta state\n";
}
int main(int argc, char **argv) {
    int count = 0;
    auto status = gpu::device_count(&count);
    if (status == gpu::no_device || status == gpu::insufficient_driver ||
        (status == gpu::success && count == 0))
        return 77;
    if (status != gpu::success) {
        std::cerr << "GPU device discovery failed: " << gpu::error_string(status) << '\n';
        return 1;
    }
    auto root = fs::temp_directory_path() / ("glm53-gpu-" + std::to_string(getpid()));
    try {
        if (argc == 3 && std::string(argv[1]) == "--model") {
            compare_model(argv[2]);
            return 0;
        }
        check(argc == 1, "expected --model CHECKPOINT or no arguments");
        select_test();
        kda_test();
        kda_test(128, 2);
        fs::create_directory(root);
        constexpr int rows = 130, cols = 272;
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
        std::vector<uint16_t> bf16(dense.size()), half(dense.size());
        std::vector<uint8_t> fp8(dense.size());
        for (int r = 0; r < rows; ++r) {
            for (int col = 0; col < cols; ++col) {
                size_t at = size_t(r) * cols + col;
                dense[at] = float((r * 17 + col * 11) % 23 - 11) / 128;
                bf16[at] = behemoth::BFloat16(dense[at]).bits;
                fp8[at] = behemoth::encode_e4m3(float((r * 7 + col * 3) % 27 - 13) / 8);
            }
            half[size_t(r) * cols + (r * 7) % cols] = r % 2 ? 0xbc00 : 0x3800;
            row_scales[r] = float(1 + r % 7) / 8;
        }
        add("probe.f32.weight", "F32", {rows, cols}, dense.data(), dense.size() * 4);
        add("probe.bf16.weight", "BF16", {rows, cols}, bf16.data(), bf16.size() * 2);
        add("probe.f16.weight", "F16", {rows, cols}, half.data(), half.size() * 2);
        add("probe.channel.weight", "F8_E4M3", {rows, cols}, fp8.data(), fp8.size());
        add("probe.channel.weight_scale", "F32", {rows}, row_scales.data(), row_scales.size() * 4);
        Vec block_scales{.5f, 2.f, .25f, 1.f, .125f, 4.f};
        add("probe.block.weight", "F8_E4M3", {rows, cols}, fp8.data(), fp8.size());
        add("probe.block.weight_scale_inv", "F32", {2, 3}, block_scales.data(), block_scales.size() * 4);
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
            for (auto base :
                 {"probe.f32", "probe.bf16", "probe.f16", "probe.channel", "probe.block", "probe.nv"}) {
                auto cpu = w.linear(base);
                auto y = project(w, base, x);
                for (int r = 0; r < rows; ++r) {
                    double expected = 0;
                    for (int c = 0; c < cols; ++c)
                        expected += double(float(behemoth::BFloat16(cpu.data[size_t(r) * cols + c]))) *
                                    float(behemoth::BFloat16(x[c]));
                    check(std::isfinite(y[r]) &&
                              std::abs(y[r] - expected) < 2e-5 * std::max(1., std::abs(expected)),
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
