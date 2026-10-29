#include "behemoth/gpu.hpp"
#include "behemoth/model.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "fixtures.hpp"
#include <iostream>
#include <sstream>

using namespace behemoth;
static void check(bool condition, const char *what) {
    if (!condition)
        throw std::runtime_error(what);
}
static void fp8_weights(hipStream_t stream) {
    for (int n : {32, 128, 256})
        for (int out : {32, 96})
            for (int rows : {1, 3, 5, 17, 33}) {
                qtip::HostLinear host;
                host.in = n; host.out = out;
                host.fp8.resize(size_t(n) * out);
                host.fp8_scale.resize(out);
                for (size_t i = 0; i < host.fp8.size(); ++i)
                    host.fp8[i] = encode_e4m3(float(std::sin(double(i) * .17) * 12));
                for (int i = 0; i < out; ++i) host.fp8_scale[i] = i == 0 ? 0 : .01f * (1 + i % 3);
                qtip::GpuFp8Linear weight; weight.upload(host, 0, out);
                const int ldx = n + 16, ldy = out + 16;
                std::vector<BFloat16> x(size_t(rows) * ldx), y(size_t(rows) * ldy + 32, BFloat16(-19));
                std::vector<float> dense(size_t(rows) * n);
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < n; ++c) {
                        x[size_t(r) * ldx + c] = BFloat16(float(std::cos(double(r * n + c) * .23)));
                        dense[size_t(r) * n + c] = float(x[size_t(r) * ldx + c]);
                    }
                DeviceBuffer<BFloat16> dx, dy; dx.upload(x); dy.upload(y);
                qtip::fp8_linear(weight, dx.data(), ldx, rows, dy.data() + 16, ldy, stream);
                HIP_CHECK(hipStreamSynchronize(stream));
                auto got = dy.download();
                auto ref = qtip::reference_forward(host, dense, true);
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < ldy; ++c) {
                        float value = float(got[16 + size_t(r) * ldy + c]);
                        if (c >= out) check(value == -19, "FP8 projection overwrote row padding");
                        else {
                            float expected = float(BFloat16(ref[size_t(r) * out + c]));
                            check(std::abs(value - expected) <= .001f + .01f * std::abs(expected), "W8A16 projection mismatch");
                        }
                    }
                for (int i = 0; i < 16; ++i)
                    check(float(got[i]) == -19 && float(got[got.size() - 1 - i]) == -19, "FP8 projection output guard");
            }
    // Production K/V geometry and both TP shards. Each output selects a
    // different input channel, giving an exact oracle independent of GEMM.
    {
        qtip::HostLinear host;
        host.in = 12288; host.out = 1024;
        host.fp8.assign(size_t(host.in) * host.out, 0);
        host.fp8_scale.resize(host.out);
        for (int o = 0; o < host.out; ++o) {
            host.fp8[size_t(o) * host.in + (o * 17) % host.in] = encode_e4m3(o % 2 ? -1.f : 1.f);
            host.fp8_scale[o] = float(1 + o % 7) / 8;
        }
        for (int rows : {1, 5, 17}) {
            std::vector<BFloat16> x(size_t(rows) * host.in);
            for (size_t i = 0; i < x.size(); ++i) x[i] = BFloat16(float(int(i % 61) - 30) / 16);
            DeviceBuffer<BFloat16> input, output(size_t(rows) * 512);
            input.upload(x);
            for (int rank = 0; rank < 2; ++rank) {
                qtip::GpuFp8Linear weights;
                weights.upload(host, rank * 512, 512);
                qtip::fp8_linear(weights, input.data(), host.in, rows, output.data(), 512, stream);
                HIP_CHECK(hipStreamSynchronize(stream));
                auto got = output.download();
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < 512; ++c) {
                        int o = rank * 512 + c;
                        BFloat16 expected(float(x[size_t(r) * host.in + (o * 17) % host.in]) *
                                          (o % 2 ? -1.f : 1.f) * host.fp8_scale[o]);
                        check(got[size_t(r) * 512 + c].bits == expected.bits, "production FP8 K/V shard projection mismatch");
                    }
            }
        }
    }
    const int saved = qtip::row_transform_threshold();
    for (int threshold : {0, 1}) {
        qtip::set_row_transform_threshold(threshold);
        for (int rows : {1, 17}) {
            std::vector<float> legacy(size_t(rows) * 2 * 7168), only_q(size_t(rows) * 2 * 6144);
            for (int rank = 0; rank < 2; ++rank)
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < 6144; ++c) {
                        float value = float(std::sin(double((rank * rows + r) * 6144 + c) * .013));
                        legacy[(size_t(rank) * rows + r) * 7168 + c] = value;
                        only_q[(size_t(rank) * rows + r) * 6144 + c] = value;
                    }
            DeviceBuffer<float> a, b, sq, sk;
            a.upload(legacy); b.upload(only_q);
            std::vector<BFloat16> legacy_bf16(legacy.size()), only_bf16(only_q.size());
            for (size_t i = 0; i < legacy.size(); ++i) legacy_bf16[i] = BFloat16(legacy[i]);
            for (size_t i = 0; i < only_q.size(); ++i) only_bf16[i] = BFloat16(only_q[i]);
            DeviceBuffer<BFloat16> ab, bb;
            ab.upload(legacy_bf16); bb.upload(only_bf16);
            sq.upload(std::vector<float>(12288, 1)); sk.upload(std::vector<float>(1024, 1));
            DeviceBuffer<BFloat16> qa(size_t(rows) * 6144), qb(size_t(rows) * 6144), k, v;
            k.upload(std::vector<BFloat16>(size_t(rows) * 512, BFloat16(-19)));
            v.upload(std::vector<BFloat16>(size_t(rows) * 512, BFloat16(-19)));
            for (int rank = 0; rank < 2; ++rank) {
                qtip::output_qkv(a.data(), rows, rank, sq.data(), sk.data(), sk.data(), qa.data(), k.data(), v.data(), stream);
                HIP_CHECK(hipStreamSynchronize(stream));
                auto expected = qa.download();
                k.upload(std::vector<BFloat16>(size_t(rows) * 512, BFloat16(-19)));
                v.upload(std::vector<BFloat16>(size_t(rows) * 512, BFloat16(-19)));
                qtip::output_qkv(b.data(), rows, rank, sq.data(), sk.data(), sk.data(), qb.data(), k.data(), v.data(), stream, true);
                HIP_CHECK(hipStreamSynchronize(stream));
                auto actual = qb.download();
                for (size_t i = 0; i < expected.size(); ++i) check(expected[i].bits == actual[i].bits, "Q-only TP epilogue changed Q");
                for (auto value : k.download()) check(float(value) == -19, "Q-only epilogue wrote K");
                for (auto value : v.download()) check(float(value) == -19, "Q-only epilogue wrote V");
                qtip::output_qkv(ab.data(), rows, rank, sq.data(), sk.data(), sk.data(), qa.data(), k.data(), v.data(), stream);
                HIP_CHECK(hipStreamSynchronize(stream));
                expected = qa.download();
                qtip::output_qkv(bb.data(), rows, rank, sq.data(), sk.data(), sk.data(), qb.data(), k.data(), v.data(), stream, true);
                HIP_CHECK(hipStreamSynchronize(stream));
                actual = qb.download();
                for (size_t i = 0; i < expected.size(); ++i) check(expected[i].bits == actual[i].bits, "BF16 Q-only TP epilogue changed Q");
            }
        }
    }
    qtip::set_row_transform_threshold(saved);
}
static std::vector<float> run_case(const test::Fixture &f, int partition, hipStream_t stream,
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
    DeviceBuffer<BFloat16> qb;
    const bool bf16 = !a.q_bf16.empty();
    if (bf16) {
        qb.upload(a.q_bf16);
        view.q_bf16 = qb.data();
    }
    paged_attention(metadata, view, workspace.data(), workspace.size(), partition, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    auto actual = out.download();
    auto reference = reference_attention(l, a, !bf16, partition);
    std::vector<float> values(reference.output.size());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(actual[i + 128]);
        if (known)
            check(values[i] == 1, "Known-answer attention mismatch");
    }
    auto error = compare_attention(values, reference.output);
    if (!error.pass)
        throw std::runtime_error("GPU attention mismatch: rms=" + std::to_string(error.rms_error) +
                                 " max=" + std::to_string(error.max_abs) +
                                 " partition=" + std::to_string(partition) +
                                 " max_len=" + std::to_string(*std::max_element(
                                                   l.lengths.begin(), l.lengths.end())) +
                                 (bf16 ? " bf16" : ""));
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
    return values;
}
// BF16 Q/P mode: unquantised queries alongside the FP8 copy (which stays valid).
static test::Fixture with_bf16_queries(test::Fixture f, int seed) {
    std::mt19937 rng{unsigned(seed) * 31 + 5};
    std::normal_distribution<float> normal(0, 1);
    f.data.q_bf16.resize(f.data.q.size());
    for (auto &x : f.data.q_bf16)
        x = BFloat16(normal(rng));
    return f;
}
// Independent invariants in addition to the scalar implementation comparison.
static void attention_audit(hipStream_t stream) {
    for (int heads : {48, 96}) {
        std::vector<int> positions(65);
        std::iota(positions.begin(), positions.end(), 0);
        auto routing = test::fixture({97, 81}, {positions, {0, 15, 16, 31, 32, 64}}, 91, heads);
        std::fill(routing.data.q.begin(), routing.data.q.end(), 0);
        // Distinct powers of two identify each request and KV head without
        // depending on reference_attention's GQA or page-index calculations.
        for (int r = 0; r < 2; ++r)
            for (int t = 0; t < routing.layout.lengths[r]; ++t)
                for (int h = 0; h < heads / 12; ++h) {
                    int page = routing.layout.page_table[r * routing.layout.table_stride + t / 16];
                    size_t loc = (size_t(page) * (heads / 12) + h) * 16 + t % 16;
                    routing.data.v_scale[loc] = std::ldexp(1.f, r * 8 + h - 8);
                    for (int d = 0; d < 128; ++d)
                        routing.data.v[loc * 128 + d] = encode_e4m3(d % 2 ? -1.f : 1.f);
                }
        for (bool bf16 : {false, true}) {
            if (bf16) routing.data.q_bf16.assign(routing.data.q.size(), BFloat16(0));
            for (int partition : {0, 32, 64}) {
                auto got = run_case(routing, partition, stream);
                for (int r = 0; r < 2; ++r)
                    for (int q = routing.layout.query_offsets[r]; q < routing.layout.query_offsets[r + 1]; ++q)
                        for (int h = 0; h < heads; ++h)
                            for (int d = 0; d < 128; ++d) {
                                float expected = std::ldexp(d % 2 ? -1.f : 1.f, r * 8 + h / 12 - 8);
                                check(std::abs(got[(size_t(q) * heads + h) * 128 + d] - expected) <= .008f * std::abs(expected),
                                      "GQA head/request/page routing invariant failed");
                            }
            }
        }
    }
    std::vector<int> positions(65);
    std::iota(positions.begin(), positions.end(), 0);
    for (bool bf16 : {false, true}) {
        auto f = test::fixture({97}, {positions}, 47);
        if (bf16) f = with_bf16_queries(std::move(f), 47);
        for (int partition : {0, 32}) {
            auto full = run_case(f, partition, stream);
            for (int t : {0, 15, 16, 31, 32, 64}) {
                auto one = f;
                one.layout.positions = {t};
                one.layout.query_offsets = {0, 1};
                const size_t row = size_t(t) * 48;
                one.data.q = {f.data.q.begin() + row * 128, f.data.q.begin() + (row + 48) * 128};
                one.data.q_scale = {f.data.q_scale.begin() + row, f.data.q_scale.begin() + row + 48};
                if (bf16) one.data.q_bf16 = {f.data.q_bf16.begin() + row * 128, f.data.q_bf16.begin() + (row + 48) * 128};
                auto decoded = run_case(one, partition, stream);
                std::vector<float> expected(full.begin() + row * 128, full.begin() + (row + 48) * 128);
                check(compare_attention(decoded, expected).pass, "prefill/decode attention disagreement");
            }
            auto poisoned = f;
            for (int t = 33; t < 97; ++t)
                for (int h = 0; h < 4; ++h) {
                    int page = f.layout.page_table[t / 16];
                    size_t loc = (size_t(page) * 4 + h) * 16 + t % 16;
                    poisoned.data.k_scale[loc] = 10;
                    poisoned.data.v_scale[loc] = 10;
                    for (int d = 0; d < 128; ++d) {
                        poisoned.data.k[loc * 128 + d] = encode_e4m3(7);
                        poisoned.data.v[loc * 128 + d] = encode_e4m3(-7);
                    }
                }
            auto changed = run_case(poisoned, partition, stream);
            check(std::equal(full.begin(), full.begin() + 33 * 48 * 128, changed.begin()),
                  "future K/V changed earlier attention outputs");
        }
    }
    std::cout << "PASS GQA routing, full prefill tiles, prefill/decode equivalence and causal isolation\n";
}
// Long single-query contexts (decode) through every partition the model can
// pick: unsplit, the forced 32/1024-key splits and the automatic block-filling
// choice at the production 32 MB workspace.
static void long_contexts(hipStream_t stream) {
    struct Batch {
        std::vector<int> lengths;
        int seed;
    };
    const std::vector<Batch> batches = {{{4096}, 7},
                                        {{13000}, 19},
                                        {{33000}, 103},
                                        {{13000, 517, 8191, 2049}, 7},
                                        {{33, 4097, 1, 2048}, 19}};
    for (const auto &b : batches) {
        std::vector<std::vector<int>> positions;
        for (int len : b.lengths)
            positions.push_back({len - 1});
        auto f = test::fixture(b.lengths, positions, b.seed);
        std::vector<int> partitions = {0, 32, 1024};
        for (int target : {512, 2048})
            partitions.push_back(decode_attention_partition(f.layout, 32u << 20, 0, target));
        partitions.push_back(decode_attention_partition(f.layout, 32u << 20, 32));
        for (int partition : partitions)
            run_case(f, partition, stream);
        auto g = with_bf16_queries(f, b.seed);
        run_case(g, 0, stream);
        run_case(g, partitions[3], stream);
    }
    // Decode queries that are not at the end of their context (masked tail keys).
    run_case(test::fixture({4100, 700}, {{2049}, {31}}, 11), 256, stream);
    // Partition sizing: short contexts stay unsplit; long ones split along 32-key tiles.
    {
        AttentionLayout l = test::fixture({90}, {{89}}).layout;
        check(decode_attention_partition(l, 32u << 20) == 32, "Short context: one tile per block");
        check(decode_attention_partition(test::fixture({32}, {{31}}).layout, 32u << 20) == 0,
              "Single-tile context should be unsplit");
        l = test::fixture({13000}, {{12999}}).layout;
        int p = decode_attention_partition(l, 32u << 20);
        check(p > 0 && p % 32 == 0 && ceil_div(size_t(13000), size_t(p)) * 4 >= 256 &&
                  ceil_div(size_t(13000), size_t(p)) * 4 <= 512,
              "Automatic partition should give about the target block count");
        check(decode_attention_partition(l, 32u << 20, 32) == 32, "Requested partition kept");
        check(decode_attention_partition(l, 1024, 32) % 32 == 0 &&
                  attention_workspace_bytes(l, decode_attention_partition(l, 1024, 32)) <= 1024,
              "Partition must grow until the partials fit the workspace");
    }
    // Mixed decode + small prefill piece over a long context (PREFILL_SCHED_REPORT.md): the
    // automatic multi-query key split against the reference.
    {
        std::vector<int> piece;
        for (int q = 12968; q < 13000; ++q)
            piece.push_back(q);
        auto f = test::fixture({13000, 517, 2049}, {piece, {516}, {2048}}, 29);
        const int p = prefill_attention_partition(f.layout, 32u << 20);
        check(p > 0 && p % 32 == 0 && p >= 256 && attention_workspace_bytes(f.layout, p) <= (32u << 20),
              "Mixed-step prefill partition splits the long context within the workspace");
        run_case(f, p, stream);
        run_case(with_bf16_queries(f, 29), p, stream);
        std::vector<int> chunk;
        for (int q = 0; q < 2048; ++q)
            chunk.push_back(q);
        check(prefill_attention_partition(test::fixture({2048}, {chunk}).layout, 32u << 20) == 0,
              "Large prefill chunks stay unsplit");
    }
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
            if (argc > 1 && std::string(argv[1]) == "--attention-audit") {
                fp8_weights(stream);
                append(stream);
                attention_audit(stream);
                long_contexts(stream);
                HIP_CHECK(hipStreamDestroy(stream));
                return 0;
            }
            conversion(stream);
            append(stream);
            projection_selection(stream);
            projections(stream);
            fp8_weights(stream);
            for (int seed : {7, 19, 103})
                for (int len : {1, 15, 16, 17, 31, 32, 33, 127, 129, 1025}) {
                    auto f = test::fixture({len}, {{len - 1}}, seed);
                    run_case(f, 0, stream);
                    run_case(f, 1024, stream);
                }
            for (int seed : {7, 19, 103}) {
                run_case(test::fixture({33, 17}, {{0, 15, 16, 32}, {0, 16}}, seed), 0, stream);
                run_case(test::fixture({129, 17}, {{0, 16, 128}, {0, 16}}, seed, 96), 32, stream);
                // Mixed decode batch with 96/8 heads and masked tails; BF16 Q/P on both kernels.
                run_case(test::fixture({129, 17, 1}, {{100}, {16}, {0}}, seed, 96), 32, stream);
                run_case(with_bf16_queries(test::fixture({33, 17}, {{0, 15, 16, 32}, {0, 16}}, seed),
                                           seed),
                         32, stream);
                run_case(with_bf16_queries(test::fixture({129, 17}, {{128}, {16}}, seed), seed), 0,
                         stream);
            }
            long_contexts(stream);
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
        std::cout << "PASS GPU conversion, append, projections, prefill, decode (GQA, long "
                     "contexts, BF16 Q/P), split/merge, output guards and stream checks\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL (GPU acceptance NOT passed): " << e.what() << '\n';
        return 1;
    }
}
