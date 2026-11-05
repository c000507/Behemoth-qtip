#include "behemoth/qtip.hpp"
#include "behemoth/numeric.hpp"
#include "kernels.hpp"
#include "safetensors.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
using namespace behemoth;
namespace cg = glm53::gpu;
namespace cu = behemoth::cuda_runtime;
template <class T> void put(cg::Buffer<T> &d, const std::vector<T> &x, cg::Stream s) {
    d.ensure(x.size(), s);
    cg::checked(cg::h2d(d, x.data(), x.size() * sizeof(T), s), "test upload");
}
void require(bool b, const char *text) {
    if (!b)
        throw std::runtime_error(text);
}
std::vector<uint16_t> bits(const std::vector<float> &v) {
    std::vector<uint16_t> x(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        x[i] = BFloat16(v[i]).bits;
    return x;
}
void error(const std::vector<float> &actual, const std::vector<float> &expected, const char *name) {
    double e = 0, r = 0, maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "Nonfinite CUDA result");
        e += std::pow(double(actual[i]) - expected[i], 2);
        r += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(double(actual[i]) - expected[i]));
    }
    double relative = std::sqrt(e / std::max(r, 1e-30));
    std::cout << name << " relative L2 " << relative << " max " << maximum << '\n';
    require(relative < .015 && maximum < .03, "CUDA/reference mismatch");
}
void linear_test(cg::Stream s, int out, int in, bool rz) {
    std::mt19937 gen(41);
    std::normal_distribution<float> normal(0, .1f);
    behemoth::qtip::HostLinear h;
    h.in = in;
    h.out = out;
    h.trellis.resize(size_t(in) * out / 8);
    h.tlut.resize(1024);
    h.su.resize(in);
    h.sv.resize(out);
    for (auto &v : h.trellis)
        v = uint16_t(gen());
    for (auto &v : h.tlut)
        v = uint16_t((gen() & 0x8000) | (0x3000 + gen() % 0x1000));
    for (auto &v : h.su)
        v = gen() % 2 ? 1.f : -1.f;
    for (auto &v : h.sv)
        v = (gen() % 2 ? 1.f : -1.f) * .003f;
    std::vector<float> x(in);
    for (auto &v : x)
        v = float(BFloat16(normal(gen)));
    auto expected = behemoth::qtip::reference_forward(h, x, true, rz);
    for (auto &v : expected)
        v = float(BFloat16(v));
    auto codebook = behemoth::qtip::make_codebook(h.tlut);
    cg::Buffer<uint16_t> trellis, input, rotated, output;
    cg::Buffer<float> su, sv, scratch, z, result;
    cg::Buffer<uint32_t> lut;
    put(trellis, h.trellis, s);
    put(input, bits(x), s);
    put(su, h.su, s);
    put(sv, h.sv, s);
    std::vector<uint32_t> table(codebook.bf16x2.begin(), codebook.bf16x2.end());
    put(lut, table, s);
    scratch.ensure(std::max(in, out), s);
    rotated.ensure(in, s);
    z.ensure(out, s);
    output.ensure(out, s);
    result.ensure(out, s);
    cu::qtip_forward({trellis, su, sv, lut, in, out}, input, output, scratch, rotated, z, rz, s);
    cg::to_float(output, result, out, s);
    cg::Pinned<float> host;
    host.ensure(out);
    cg::checked(cg::d2h(host.p, result, size_t(out) * 4, s), "read QTIP result");
    cg::checked(cg::sync(s), "finish QTIP test");
    error({host.p, host.p + out}, expected, "QTIP trellis/RHT");
}
void fp8_test(cg::Stream s) {
    behemoth::qtip::HostLinear h;
    h.in = 96;
    h.out = 64;
    h.fp8.resize(96 * 64);
    h.fp8_scale.resize(64);
    std::mt19937 gen(18);
    for (auto &v : h.fp8) {
        v = uint8_t(gen());
        if ((v & 127) == 127)
            v = 0;
    }
    for (auto &v : h.fp8_scale)
        v = float(1 + gen() % 7) / 4096;
    std::vector<float> x(96);
    for (auto &v : x)
        v = float(BFloat16(float(int(gen() % 31) - 15) / 32));
    auto expected = behemoth::qtip::reference_forward(h, x, true);
    for (auto &v : expected)
        v = float(BFloat16(v));
    cg::Buffer<uint8_t> w;
    cg::Buffer<float> scales, result;
    cg::Buffer<uint16_t> input, out;
    put(w, h.fp8, s);
    put(scales, h.fp8_scale, s);
    put(input, bits(x), s);
    out.ensure(64, s);
    result.ensure(64, s);
    cu::fp8_forward(w, scales, input, out, 96, 64, s);
    cg::to_float(out, result, 64, s);
    cg::Pinned<float> host;
    host.ensure(64);
    cg::checked(cg::d2h(host.p, result, 64 * 4, s), "read FP8");
    cg::checked(cg::sync(s), "finish FP8 test");
    error({host.p, host.p + 64}, expected, "FP8 output-channel scaling");
}
void nvfp4_test(cg::Stream s) {
    // Adaptive mixed-format checkpoints: NVFP4 projection (E2M1 pairs, E4M3 block scales over 16
    // input channels, F32 global scale) through the loader and the shared gemv kernel.
    const int out = 64, in = 96;
    std::mt19937 gen(29);
    std::normal_distribution<float> normal(0, .05f);
    std::vector<float> w(size_t(out) * in);
    for (auto &v : w)
        v = normal(gen);
    float maximum = 0;
    for (float v : w)
        maximum = std::max(maximum, std::abs(v));
    const float global = maximum / (448.f * 6);
    auto fp4 = [](float x) {
        constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        float a = std::abs(x), best = INFINITY;
        int code = 0;
        for (int i = 0; i < 8; ++i) {
            float e = std::abs(a - levels[i]);
            if (e < best || (e == best && (i % 2 == 0))) {
                best = e;
                code = i;
            }
        }
        return uint8_t(code | (std::signbit(x) ? 8 : 0));
    };
    std::vector<uint8_t> packed(w.size() / 2), scales(w.size() / 16);
    std::vector<float> wq(w.size());
    for (size_t b = 0; b < scales.size(); ++b) {
        float m = 0;
        for (int j = 0; j < 16; ++j)
            m = std::max(m, std::abs(w[b * 16 + j]));
        uint8_t code = encode_e4m3(m / (6 * global));
        scales[b] = code;
        float scale = decode_e4m3(code) * global;
        for (int j = 0; j < 16; j += 2)
            packed[b * 8 + j / 2] =
                fp4(scale ? w[b * 16 + j] / scale : 0) | (fp4(scale ? w[b * 16 + j + 1] / scale : 0) << 4);
    }
    // exact dequantized values (device rounding: level * scale * global rounded to bf16)
    constexpr float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (size_t b = 0; b < scales.size(); ++b) {
        float scale = decode_e4m3(scales[b]) * global;
        for (int j = 0; j < 16; ++j) {
            uint8_t c = (packed[b * 8 + j / 2] >> (4 * (j & 1))) & 15;
            wq[b * 16 + j] = float(BFloat16((c & 8 ? -1.f : 1.f) * levels[c & 7] * scale));
        }
    }
    std::vector<float> x(in);
    for (auto &v : x)
        v = float(BFloat16(normal(gen)));
    std::vector<float> expected(out);
    for (int o = 0; o < out; ++o) {
        double sum = 0;
        for (int i = 0; i < in; ++i)
            sum += double(wq[size_t(o) * in + i]) * double(x[size_t(i)]);
        expected[size_t(o)] = float(BFloat16(float(sum)));
    }
    // loader: write a layer shard exactly as qtip_quant --mode adaptive does and read it back
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "behemoth_nvfp4_test";
    std::filesystem::create_directories(dir);
    const std::string base = "model.layers.0.self_attn.q_proj.";
    {
        ::qtip::SafeTensorsWriter writer;
        writer.add(base + "weight_packed", "U8", {out, in / 2}, packed.data(), packed.size());
        writer.add(base + "weight_scale", "F8_E4M3", {out, in / 16}, scales.data(), scales.size());
        writer.add(base + "weight_global_scale", "F32", {}, &global, 4);
        writer.add(base + "input_layernorm.weight", "BF16", {16}, std::vector<uint16_t>(16).data(), 32);
        writer.write((dir / "model-layer-00000.safetensors").string());
    }
    SafeTensorFile file(dir / "model-layer-00000.safetensors");
    auto h = behemoth::qtip::load_nvfp4_linear(file, 0, "self_attn.q_proj");
    require(h.is_nvfp4() && h.in == in && h.out == out, "NVFP4 loader geometry");
    require(h.nvfp4_packed == packed && h.nvfp4_scales == scales, "NVFP4 loader payload");
    require(h.nvfp4_global == global, "NVFP4 loader global scale");
    require(behemoth::qtip::load_linear(file, 0, "self_attn.q_proj").is_nvfp4(), "NVFP4 dispatch through load_linear");
    // kernel: expected computed from the same dequantized bf16 weights
    cg::Buffer<uint8_t> d_packed, d_scales;
    cg::Buffer<uint16_t> input, out_b;
    cg::Buffer<float> result;
    put(d_packed, packed, s);
    put(d_scales, scales, s);
    put(input, bits(x), s);
    out_b.ensure(out, s);
    result.ensure(out, s);
    cu::nvfp4_forward(d_packed, d_scales, global, input, out_b, in, out, s);
    cg::to_float(out_b, result, out, s);
    cg::Pinned<float> host;
    host.ensure(out);
    cg::checked(cg::d2h(host.p, result, size_t(out) * 4, s), "read NVFP4");
    cg::checked(cg::sync(s), "finish NVFP4 test");
    error({host.p, host.p + out}, expected, "NVFP4 mixed-format projection");
}
void attention_test(cg::Stream s) {
    constexpr int heads = 6, kvheads = 2, dim = 128, max_tokens = 17;
    std::mt19937 gen(21);
    std::normal_distribution<float> normal(0, .2f);
    std::vector<float> q(heads * dim), k(max_tokens * kvheads * dim), v(k.size());
    for (auto *vec : {&q, &k, &v})
        for (auto &a : *vec)
            a = float(BFloat16(normal(gen)));
    cg::Buffer<uint16_t> dq, dk, dv, out;
    cg::Buffer<float> scores, result;
    put(dq, bits(q), s);
    put(dk, bits(k), s);
    put(dv, bits(v), s);
    out.ensure(heads * dim, s);
    scores.ensure(heads * max_tokens, s);
    result.ensure(heads * dim, s);
    cg::Pinned<float> host;
    host.ensure(heads * dim);
    for (int n : {1, 2, 7, 17}) {
        std::vector<float> expected(heads * dim);
        for (int h = 0; h < heads; ++h) {
            int kh = h / (heads / kvheads);
            std::vector<double> p(n);
            double maximum = -INFINITY, total = 0;
            for (int t = 0; t < n; ++t) {
                double sum = 0;
                for (int d = 0; d < dim; ++d)
                    sum += double(q[h * dim + d]) * k[(t * kvheads + kh) * dim + d];
                p[t] = sum / std::sqrt(double(dim));
                maximum = std::max(maximum, p[t]);
            }
            for (auto &a : p) {
                a = std::exp(a - maximum);
                total += a;
            }
            for (auto &a : p)
                a = float(BFloat16(float(a / total)));
            for (int d = 0; d < dim; ++d) {
                double sum = 0;
                for (int t = 0; t < n; ++t)
                    sum += p[t] * v[(t * kvheads + kh) * dim + d];
                expected[h * dim + d] = float(BFloat16(float(sum)));
            }
        }
        cu::gqa(dq, dk, dv, out, scores, n, heads, kvheads, dim, s);
        cg::to_float(out, result, heads * dim, s);
        cg::checked(cg::d2h(host.p, result, heads * dim * 4, s), "read GQA");
        cg::checked(cg::sync(s), "finish GQA test");
        error({host.p, host.p + heads * dim}, expected, "GQA causal prefix/group mapping");
    }
}
int main(int argc, char **argv) {
    int count = 0;
    auto status = cg::device_count(&count);
    if (status == cg::no_device || status == cg::insufficient_driver || (status == cg::success && !count))
        return 77;
    if (argc == 2 && std::string(argv[1]) == "--probe")
        return status == cg::success ? 0 : 1;
    cg::Stream s{};
    try {
        cg::checked(status, "GPU discovery");
        cg::checked(cg::stream_create(&s), "test stream");
        cu::initialize_tables();
        for (auto [out, in] :
             {std::pair{32, 32}, std::pair{128, 128}, std::pair{768, 1792}, std::pair{1792, 768},
              std::pair{1024, 12288}, std::pair{12288, 1024}, std::pair{28672, 32}})
            for (bool rz : {false, true})
                linear_test(s, out, in, rz);
        fp8_test(s);
        nvfp4_test(s);
        attention_test(s);
        cg::checked(cg::stream_destroy(s), "test stream shutdown");
        std::cout << "PASS native CUDA QTIP, Hadamard orientations, FP8 scales, NVFP4 and causal GQA\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (s)
            (void)cg::stream_destroy(s);
        return 1;
    }
}
