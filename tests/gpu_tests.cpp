#include "behemoth/gpu.hpp"
#include "behemoth/model.hpp"
#include "fixtures.hpp"
#include <iostream>
#include <sstream>

using namespace behemoth;
static void check(bool condition, const char *what) {
    if (!condition)
        throw std::runtime_error(what);
}
static void run_case(const test::Fixture &f, int partition, hipStream_t stream,
                     bool known = false) {
    const auto &l = f.layout;
    const auto &a = f.data;
    GpuMetadata metadata;
    metadata.upload(l);
    DeviceBuffer<uint8_t> q, k, v;
    DeviceBuffer<float> qs, ks, vs, mx, den;
    q.upload(a.q);
    k.upload(a.k);
    v.upload(a.v);
    qs.upload(a.q_scale);
    ks.upload(a.k_scale);
    vs.upload(a.v_scale);
    size_t rows = size_t(l.queries()) * size_t(l.query_heads);
    mx.resize(rows);
    den.resize(rows);
    std::vector<BFloat16> guarded(rows * 128 + 256, BFloat16(-19));
    DeviceBuffer<BFloat16> out;
    out.upload(guarded);
    DeviceBuffer<std::byte> workspace(attention_workspace_bytes(l, partition));
    AttentionDeviceView view{q.data(),  qs.data(),        k.data(),  v.data(),  ks.data(),
                             vs.data(), out.data() + 128, mx.data(), den.data()};
    paged_attention(metadata, view, workspace.data(), workspace.size(), partition, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    auto actual = out.download();
    auto reference = reference_attention(l, a, true, partition);
    std::vector<float> values(reference.output.size());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(actual[i + 128]);
        if (known)
            check(values[i] == 1, "Known-answer attention mismatch");
    }
    auto error = compare_attention(values, reference.output);
    if (!error.pass)
        throw std::runtime_error("GPU attention mismatch: rms=" + std::to_string(error.rms_error) +
                                 " max=" + std::to_string(error.max_abs));
    auto maxima = mx.download(), denominators = den.download();
    for (size_t i = 0; i < rows; ++i) {
        check(std::abs(maxima[i] - reference.maximum[i]) <
                  .01f + .001f * std::abs(reference.maximum[i]),
              "Softmax maximum mismatch");
        check(std::abs(denominators[i] - reference.denominator[i]) <
                  .01f + .001f * reference.denominator[i],
              "Softmax denominator mismatch");
    }
    for (size_t i = 0; i < 128; ++i)
        check(actual[i].bits == guarded[i].bits &&
                  actual[actual.size() - 1 - i].bits == guarded.back().bits,
              "Attention wrote outside output");
}
static void conversion(hipStream_t stream) {
    std::vector<float> values;
    for (int i = 0; i < 256; ++i)
        if ((i & 127) != 127)
            values.push_back(decode_e4m3(uint8_t(i)));
    values.insert(values.end(), {0.f, -0.f, 1.f + 1.f / 16, 1.f + 3.f / 16, 1000, -1000, INFINITY,
                                 -INFINITY, std::ldexp(1.f, -10)});
    values.insert(values.end(),
                  {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                   std::bit_cast<float>(0x7fc00000U), std::bit_cast<float>(0xffc00000U)});
    std::mt19937 rng(73);
    std::uniform_real_distribution<float> uniform(-600, 600);
    for (int i = 0; i < 10000; ++i)
        values.push_back(uniform(rng));
    DeviceBuffer<float> input;
    input.upload(values);
    DeviceBuffer<uint8_t> output(values.size());
    fp8_conversion_probe(input.data(), output.data(), int(values.size()), stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    auto encoded = output.download();
    for (size_t i = 0; i < values.size(); ++i) {
        const auto expected = encode_e4m3(values[i]);
        if (encoded[i] != expected) {
            std::ostringstream message;
            message << "Runtime OCP E4M3 conversion differs at index " << i
                    << " (value=" << values[i] << ", float_bits=0x" << std::hex
                    << std::bit_cast<uint32_t>(values[i]) << ", expected=0x" << unsigned(expected)
                    << ", actual=0x" << unsigned(encoded[i]) << ')';
            throw std::runtime_error(message.str());
        }
    }
    check(encode_e4m3(-0.f) == 128, "OCP negative zero changed");
}
static void append(hipStream_t stream) {
    auto f = test::fixture({33, 17}, {{15, 16, 32}, {0, 16}});
    auto expected = f;
    size_t count = size_t(f.layout.queries()) * 4 * 128;
    std::vector<BFloat16> k(count), v(count);
    for (size_t i = 0; i < count; ++i) {
        k[i] = BFloat16(std::sin(float(i)) * .5f);
        v[i] = BFloat16(std::cos(float(i)) * 7);
    }
    reference_append(expected.layout, expected.data, k, v);
    GpuMetadata metadata;
    metadata.upload(f.layout);
    DeviceBuffer<BFloat16> kin, vin;
    kin.upload(k);
    vin.upload(v);
    DeviceBuffer<uint8_t> kd, vd;
    kd.upload(f.data.k);
    vd.upload(f.data.v);
    DeviceBuffer<float> ks, vs;
    ks.upload(f.data.k_scale);
    vs.upload(f.data.v_scale);
    AttentionDeviceView view{};
    view.k = kd.data();
    view.v = vd.data();
    view.k_scale = ks.data();
    view.v_scale = vs.data();
    append_kv(metadata, view, kin.data(), vin.data(), stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    check(kd.download() == expected.data.k, "KV append K mismatch");
    check(vd.download() == expected.data.v, "KV append V mismatch");
    auto aks = ks.download(), avs = vs.download();
    for (size_t i = 0; i < aks.size(); ++i) {
        check((std::isnan(aks[i]) && std::isnan(expected.data.k_scale[i])) ||
                  aks[i] == expected.data.k_scale[i],
              "KV K scale mismatch");
        check((std::isnan(avs[i]) && std::isnan(expected.data.v_scale[i])) ||
                  avs[i] == expected.data.v_scale[i],
              "KV V scale mismatch");
    }
}
static void projection_selection(hipStream_t stream) {
    HostProjection p;
    p.in = 128;
    p.out = 32;
    p.codes.assign(size_t(p.out * p.in / 8), 0);
    p.codes.back() = 65535;
    p.codebook.assign(65536 * 8, encode_e4m3(1));
    p.row_scale.assign(size_t(p.out), 1);
    GpuProjection first, second;
    first.upload(p);
    second.upload(p);
    DeviceBuffer<BFloat16> input;
    input.upload(std::vector<BFloat16>(size_t(p.in), BFloat16(.25f)));
    DeviceBuffer<float> output(size_t(p.out));
    ProjectionWorkspace workspace;
    ProjectionRunner runner;
    auto run = [&](const GpuProjection &projection, bool scaled = true) {
        runner.run(projection, input.data(), output.data(), 1, workspace, stream, scaled);
        HIP_CHECK(hipStreamSynchronize(stream));
        return output.download();
    };
    auto automatic = run(first);
    auto compact_count = [&] {
        size_t count = 0;
        for (const auto &entry : runner.profile(false))
            count += entry.at("selection_count").get<size_t>();
        return count;
    };
    check(compact_count() == 1, "Initial compact selection count differs");
    auto selection = runner.selection(first, 1);
    check(selection.at("weight_identity").get<uint64_t>() == first.tuning_identity,
          "Auto did not key decode selection by weight identity");
    (void)run(first);
    check(runner.selection(first, 1) == selection, "Cache hit unexpectedly retuned the weight");
    auto selected = parse_projection_mode(selection.at("selected").get<std::string>());
    runner.set_mode(selected);
    check(run(first) == automatic, "Auto and its selected forced kernel differ");
    const auto forced =
        selected == ProjectionMode::Fused ? ProjectionMode::Pipelined64 : ProjectionMode::Fused;
    runner.set_mode(forced);
    for (const auto &entry : runner.profile(false))
        check(entry.at("selected") == projection_mode_name(forced),
              "Compact profile retained an old forced mode");
    runner.set_mode(ProjectionMode::Auto);
    check(runner.profile(false)[0].at("selected") == projection_mode_name(selected),
          "Compact profile did not restore the automatic choice");
    (void)run(second);
    check(runner.profile().size() == 2 && compact_count() == 2 &&
              first.tuning_identity != second.tuning_identity,
          "Equal shapes incorrectly shared a decode selection");
    const auto previous = second.tuning_identity;
    p.codebook.assign(65536 * 8, encode_e4m3(.5f));
    second.upload(p);
    auto replaced = run(second);
    check(second.tuning_identity != previous && runner.profile().size() == 3 &&
              compact_count() == 3,
          "Reupload reused a stale tuning decision");
    check(compare_attention(replaced, reference_projection(p, std::vector<float>(128, .25f))).pass,
          "Reuploaded weight output differs from its reference");
    (void)run(first, false);
    check(runner.profile().size() == 4 && compact_count() == 4 &&
              !runner.selection(first, 1, false).at("apply_row_scale").get<bool>(),
          "Scaled and unscaled calls incorrectly shared a selection");
    runner.set_search(ProjectionSearch::Exhaustive);
    check(runner.profile().empty() && runner.profile(false).empty(),
          "Changing search policy kept stale selections");
    (void)run(first);
    check(runner.selection(first, 1).at("search") == "exhaustive",
          "Exhaustive search policy was not applied");
    auto compact = runner.profile(false);
    check(compact.size() == 1 && compact[0].at("selection_count") == 1,
          "Compact selection profile has incorrect counts");
}
static void projections(hipStream_t stream) {
    struct Shape {
        int width, outputs;
        std::vector<int> rows;
    };
    std::vector<Shape> shapes;
    for (int width : {128, 1032, 2048})
        for (int outputs : {32, 37, 64, 65, 128})
            shapes.push_back({width, outputs, {1, 2, 7, 16, 33, 64}});
    // Cover multiple 2048-input partitions and 128-column output tiles,
    // plus simultaneous K/N tails, without expanding the whole Cartesian suite.
    shapes.push_back({4096, 256, {1, 7, 65}});
    shapes.push_back({4104, 129, {1, 7, 130}});
    for (const auto &shape : shapes) {
        const int width = shape.width, outputs = shape.outputs;
        HostProjection p;
        p.in = width;
        p.out = outputs;
        p.codes.resize(size_t(p.out) * size_t(p.in / 8));
        p.row_scale.resize(size_t(p.out));
        p.codebook.resize(65536 * 8);
        p.codebook_scale = .02f;
        std::mt19937 rng(11);
        std::normal_distribution<float> normal;
        for (auto &c : p.codes)
            c = uint16_t(rng());
        p.codes[0] = 0;
        p.codes[1] = 65535;
        for (auto &b : p.codebook)
            b = encode_e4m3(normal(rng) * 40);
        for (auto &s : p.row_scale)
            s = .5f + std::abs(normal(rng));
        ProjectionRunner runner;
        ProjectionWorkspace workspace;
        for (auto layout : {CodeLayout::RowMajor, CodeLayout::GroupMajor}) {
            for (bool dense : {false, true}) {
                GpuProjection gpu;
                gpu.upload(p, layout);
                // Also exercise the codebook-major layouts (tails in rows and
                // groups): the chunk-code "sorted" encoding with the row-major
                // codes, the lane-owns-row encoding with the group-major codes.
                if (!dense && layout == CodeLayout::RowMajor)
                    gpu.upload_sorted(p);
                if (!dense && layout == CodeLayout::GroupMajor)
                    gpu.upload_lane(p);
                if (dense) {
                    std::vector<uint8_t> weights(size_t(p.out) * size_t(p.in));
                    for (int o = 0; o < p.out; ++o)
                        for (int i = 0; i < p.in; ++i)
                            weights[size_t(o) * size_t(p.in) + size_t(i)] =
                                p.codebook[size_t(p.codes[size_t(o) * size_t(p.in / 8) +
                                                          size_t(i / 8)]) *
                                               8 +
                                           size_t(i % 8)];
                    gpu.dense.upload(weights);
                }
                for (int rows : shape.rows) {
                    std::vector<float> input(size_t(rows) * size_t(p.in));
                    std::vector<BFloat16> bf(input.size());
                    for (size_t i = 0; i < input.size(); ++i) {
                        bf[i] = BFloat16(normal(rng));
                        input[i] = float(bf[i]);
                    }
                    DeviceBuffer<BFloat16> in;
                    in.upload(bf);
                    const size_t count = size_t(rows) * size_t(p.out);
                    constexpr size_t guard = 16;
                    constexpr float sentinel = 123456.f;
                    DeviceBuffer<float> out;
                    const std::vector<float> poisoned(count + 2 * guard, sentinel);
                    auto read = [&] {
                        auto values = out.download();
                        for (size_t i = 0; i < guard; ++i)
                            check(values[i] == sentinel && values[guard + count + i] == sentinel,
                                  "Projection output guard overwritten");
                        return std::vector<float>(values.begin() + std::ptrdiff_t(guard),
                                                  values.end() - std::ptrdiff_t(guard));
                    };
                    auto expected = reference_projection(p, input),
                         unscaled = reference_projection_unscaled(p, input);
                    for (auto mode : projection_modes()) {
                        runner.set_mode(mode);
                        // A missing store must not inherit another path's correct output.
                        out.upload(poisoned);
                        try {
                            runner.run(gpu, in.data(), out.data() + guard, rows, workspace, stream);
                        } catch (const ProjectionUnavailable &) {
                            const int needs = projection_mode_layout(mode);
                            check(mode == ProjectionMode::Blas ||
                                      (needs && (!gpu.has_sorted() ||
                                                 (needs == 1 && gpu.sorted_lane) ||
                                                 (needs == 2 && !gpu.sorted_lane))),
                                  "Mandatory projection path unavailable");
                            continue;
                        }
                        HIP_CHECK(hipStreamSynchronize(stream));
                        auto error = compare_attention(read(), expected);
                        auto description =
                            std::string(projection_mode_name(mode)) +
                            " in=" + std::to_string(width) + " out=" + std::to_string(outputs) +
                            " rows=" + std::to_string(rows) + " dense=" + std::to_string(dense);
                        description += " layout=";
                        description += code_layout_name(layout);
                        if (!error.pass)
                            throw std::runtime_error("Projection differs: " + description +
                                                     " rms=" + std::to_string(error.rms_error) +
                                                     " max=" + std::to_string(error.max_abs));
                        out.upload(poisoned);
                        runner.run(gpu, in.data(), out.data() + guard, rows, workspace, stream,
                                   false);
                        HIP_CHECK(hipStreamSynchronize(stream));
                        if (!compare_attention(read(), unscaled).pass)
                            throw std::runtime_error("Row-parallel projection differs: " +
                                                     description);
                    }
                }
            }
        }
    }
}
int main(int argc, char **argv) {
    try {
        if (argc > 1 &&
            (std::string(argv[1]) == "--two-gpu" || std::string(argv[1]) == "--two-gpu-rccl")) {
            tensor_parallel_smoke({0, 1}, std::string(argv[1]) == "--two-gpu-rccl");
            std::cout << "PASS selected collective transport with P2P disabled\n";
            return 0;
        }
        HIP_CHECK(hipSetDevice(0));
        verify_device();
        hipStream_t stream{};
        HIP_CHECK(hipStreamCreate(&stream));
        try {
            conversion(stream);
            append(stream);
            projection_selection(stream);
            projections(stream);
            for (int seed : {7, 19, 103})
                for (int len : {1, 15, 16, 17, 31, 32, 33, 127, 129, 1025}) {
                    auto f = test::fixture({len}, {{len - 1}}, seed);
                    run_case(f, 0, stream);
                    run_case(f, 1024, stream);
                }
            for (int seed : {7, 19, 103}) {
                run_case(test::fixture({33, 17}, {{0, 15, 16, 32}, {0, 16}}, seed), 0, stream);
                run_case(test::fixture({129, 17}, {{0, 16, 128}, {0, 16}}, seed, 96), 32, stream);
            }
            auto known = test::fixture({33}, {{0, 16, 32}});
            std::fill(known.data.q.begin(), known.data.q.end(), 0);
            std::fill(known.data.v.begin(), known.data.v.end(), encode_e4m3(1));
            std::fill(known.data.v_scale.begin(), known.data.v_scale.end(), 1);
            run_case(known, 0, stream, true);
            run_case(known, 32, stream, true);
            // Distinct streams and caller-owned workspaces: neither call may depend on stream zero.
            hipStream_t other{};
            HIP_CHECK(hipStreamCreate(&other));
            run_case(test::fixture({17}, {{16}}, 53), 0, other);
            HIP_CHECK(hipStreamDestroy(other));
            HIP_CHECK(hipStreamDestroy(stream));
        } catch (...) {
            (void)hipStreamDestroy(stream);
            throw;
        }
        std::cout << "PASS GPU conversion, append, projections, prefill, decode, split/merge, "
                     "output guards and stream checks\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL (GPU acceptance NOT passed): " << e.what() << '\n';
        return 1;
    }
}
