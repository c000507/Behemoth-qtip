#include "behemoth/cache.hpp"
#include "behemoth/checkpoint.hpp"
#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"
#include "behemoth/sorted_layout.hpp"
#include "behemoth/tuning.hpp"
#include "behemoth/worker.hpp"
#include "fixtures.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <tuple>
#include <iostream>
#include <unistd.h>

using namespace behemoth;
static int assertions = 0;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        ++assertions;                                                                              \
        if (!(x))                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +      \
                                     ": " #x);                                                     \
    } while (false)
template <class F> void rejects(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception &) {
        caught = true;
    }
    CHECK(caught);
}
static void numeric_tests() {
    for (int i = 0; i < 256; ++i) {
        auto b = uint8_t(i);
        float f = decode_e4m3(b);
        if ((b & 127) == 127)
            CHECK(std::isnan(f));
        else
            CHECK(encode_e4m3(f) == b);
    }
    CHECK(encode_e4m3(1000) == 126);
    CHECK(encode_e4m3(INFINITY) == 126);
    CHECK(encode_e4m3(-INFINITY) == 254);
    CHECK(encode_e4m3(std::numeric_limits<float>::max()) == 126);
    CHECK(encode_e4m3(-std::numeric_limits<float>::max()) == 254);
    CHECK(encode_e4m3(std::bit_cast<float>(0x7fc00000U)) == 127);
    CHECK(encode_e4m3(std::bit_cast<float>(0xffc00000U)) == 255);
    CHECK(decode_e4m3(126) == 448);
    CHECK(encode_e4m3(1 + 1.f / 16) == 56);
    CHECK(encode_e4m3(1 + 3.f / 16) == 58);
    CHECK(encode_e4m3(std::ldexp(1.f, -10)) == 0);
    CHECK(encode_e4m3(std::ldexp(3.f, -10)) == 2);
    // Brute-force nearest representable value is independent of exponent-based conversion.
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> uniform(-600, 600);
    for (int t = 0; t < 20000; ++t) {
        float x = uniform(rng), a = std::abs(x), best = INFINITY;
        int winner = 0;
        for (int i = 0; i < 127; ++i) {
            float e = std::abs(a - decode_e4m3(uint8_t(i)));
            if (e < best || (e == best && !(i & 1))) {
                best = e;
                winner = i;
            }
        }
        CHECK(encode_e4m3(x) == uint8_t(winner | (std::signbit(x) ? 128 : 0)));
    }
    CHECK(float(BFloat16(1.f)) == 1);
    CHECK(float(BFloat16(std::bit_cast<float>(0x3f808000U))) == 1);
    CHECK(float(BFloat16(std::bit_cast<float>(0x3f818000U))) == std::bit_cast<float>(0x3f820000U));
    CHECK(std::isnan(float(BFloat16(NAN))));
    CHECK(decode_checkpoint_f16(0x3c00) == 1);
    CHECK(decode_checkpoint_f16(1) == std::ldexp(1.f, -24));
    CHECK(decode_checkpoint_f16(0x7bff) == 65504);
    CHECK(std::isinf(decode_checkpoint_f16(0x7c00)));
    CHECK(std::isnan(decode_checkpoint_f16(0x7fff)));
    rejects([] {
        std::array<float, 1> x{NAN};
        quantize(x);
    });
    rejects([] { checked_mul(SIZE_MAX, 2); });
}
static void attention_tests() {
    for (int len : {1, 15, 16, 17, 31, 32, 33, 127, 129}) {
        auto f = test::fixture({len}, {{len - 1}});
        auto r = reference_attention(f.layout, f.data),
             split = reference_attention(f.layout, f.data, true, 32);
        auto unquant = reference_attention(f.layout, f.data, false),
             unsplit_unquant = reference_attention(f.layout, f.data, false, 32);
        CHECK(compare_attention(unquant.output, unsplit_unquant.output).max_abs < 1e-6);
        CHECK(compare_attention(r.output, r.output).pass);
        CHECK(compare_attention(split.output, split.output).pass);
        std::fill(f.data.q.begin(), f.data.q.end(), 0);
        for (auto &v : f.data.v)
            v = encode_e4m3(1);
        std::fill(f.data.v_scale.begin(), f.data.v_scale.end(), 1);
        auto constant = reference_attention(f.layout, f.data, true, 32);
        for (float x : constant.output)
            CHECK(x == 1.f);
        auto bad = f.layout;
        bad.positions[0] = len;
        rejects([&] { bad.validate(); });
    }
    auto f = test::fixture({33, 17}, {{0, 15, 16, 32}, {0, 16}}, 9);
    auto r = reference_attention(f.layout, f.data);
    auto changed = f;
    // Poison future values in request 0; its first query must be unchanged.
    for (int t = 1; t < 33; ++t)
        for (int h = 0; h < 4; ++h) {
            int page = f.layout.page_table[size_t(t / 16)];
            size_t loc = (size_t(page) * 4 + size_t(h)) * 16 + size_t(t % 16);
            std::fill_n(changed.data.v.begin() + std::ptrdiff_t(loc * 128), 128, encode_e4m3(448));
        }
    auto r2 = reference_attention(changed.layout, changed.data);
    CHECK(std::equal(r.output.begin(), r.output.begin() + 48 * 128, r2.output.begin()));
    auto full = test::fixture({17}, {{16}}, 17, 96);
    auto full_result = reference_attention(full.layout, full.data);
    for (int rank = 0; rank < 2; ++rank) {
        auto shard = full;
        auto &l = shard.layout;
        auto &d = shard.data;
        l.query_heads = 48;
        l.kv_heads = 4;
        d.q.assign(full.data.q.begin() + rank * 48 * 128,
                   full.data.q.begin() + (rank + 1) * 48 * 128);
        d.q_scale.assign(full.data.q_scale.begin() + rank * 48,
                         full.data.q_scale.begin() + (rank + 1) * 48);
        d.k.resize(size_t(l.physical_pages) * 4 * 16 * 128);
        d.v.resize(d.k.size());
        d.k_scale.resize(size_t(l.physical_pages) * 4 * 16);
        d.v_scale.resize(d.k_scale.size());
        for (int p = 0; p < l.physical_pages; ++p)
            for (int h = 0; h < 4; ++h)
                for (int t = 0; t < 16; ++t) {
                    size_t src = (size_t(p) * 8 + size_t(h + rank * 4)) * 16 + size_t(t),
                           dst = (size_t(p) * 4 + size_t(h)) * 16 + size_t(t);
                    std::copy_n(full.data.k.begin() + std::ptrdiff_t(src * 128), 128,
                                d.k.begin() + std::ptrdiff_t(dst * 128));
                    std::copy_n(full.data.v.begin() + std::ptrdiff_t(src * 128), 128,
                                d.v.begin() + std::ptrdiff_t(dst * 128));
                    d.k_scale[dst] = full.data.k_scale[src];
                    d.v_scale[dst] = full.data.v_scale[src];
                }
        auto got = reference_attention(l, d);
        CHECK(std::equal(got.output.begin(), got.output.end(),
                         full_result.output.begin() + rank * 48 * 128));
    }
    auto invalid = f.layout;
    invalid.page_table[1] = invalid.page_table[0];
    rejects([&] { invalid.validate(); });
    invalid = f.layout;
    invalid.query_offsets[1] = -1;
    rejects([&] { invalid.validate(); });
    invalid = f.layout;
    invalid.query_offsets[1] = 100;
    rejects([&] { invalid.validate(); });
    auto append = test::fixture({17}, {{15, 16}});
    std::vector<BFloat16> k(2 * 4 * 128, BFloat16(2)), v(k.size(), BFloat16(-3));
    reference_append(append.layout, append.data, k, v);
    for (int t : {15, 16})
        for (int h = 0; h < 4; ++h) {
            int p = append.layout.page_table[size_t(t / 16)];
            size_t i = (size_t(p) * 4 + size_t(h)) * 16 + size_t(t % 16);
            CHECK(std::abs(decode_e4m3(append.data.k[i * 128]) * append.data.k_scale[i] - 2) <
                  1e-5);
            CHECK(std::abs(decode_e4m3(append.data.v[i * 128]) * append.data.v_scale[i] + 3) <
                  1e-5);
        }
    auto corrupted = r.output;
    corrupted[0] += 1;
    CHECK(!compare_attention(corrupted, r.output).pass);
}
static void scheduler_tests() {
    PagePool pool(3);
    std::vector<int> a, b;
    CHECK(pool.grow(a, 17));
    CHECK(!pool.grow(b, 17));
    CHECK(b.empty());
    std::vector<int> duplicate{a[0], a[0]};
    rejects([&] { pool.release(duplicate); });
    CHECK(pool.free_pages() == 1);
    CHECK(pool.grow(b, 1));
    pool.release(a);
    pool.release(b);
    CHECK(pool.free_pages() == 3);
    CHECK(PagePool::bytes_per_page() == 1486848);
    rejects([] { PagePool::capacity_for_budget(1024, 1024, 1); });
    Scheduler s(4, 9, 5);
    std::vector<int> prompt(17, 7);
    std::vector<uint64_t> ids;
    for (int i = 0; i < 5; ++i)
        ids.push_back(s.submit(prompt, 19));
    int rounds = 0;
    while (s.pending() && ++rounds < 1000) {
        auto batch = s.schedule();
        CHECK(!batch.requests.empty());
        CHECK(batch.tokens <= 9);
        for (const auto &step : batch.requests)
            s.complete(step, step.sample ? 7 : -1);
    }
    CHECK(rounds < 1000);
    CHECK(s.free_pages() == 4);
    CHECK(s.preemptions() > 0);
    for (auto id : ids) {
        CHECK(s.request(id).generated == 19);
        s.erase(id);
    }
    auto id = s.submit(std::vector<int>{1}, 1);
    s.cancel(id);
    CHECK(s.request(id).reason == "cancelled");
    s.erase(id);
    rejects([&] { s.submit(std::vector<int>(65, 1), 1); });
    rejects([&] { s.submit(std::vector<int>{-1}, 1); });
    Scheduler r(4, 1, 1);
    auto r1 = r.submit(std::vector<int>{1}, 3), r2 = r.submit(std::vector<int>{1}, 3);
    auto step = r.schedule();
    CHECK(step.requests.size() == 1);
    rejects([&] { r.schedule(); });
    r.complete(step.requests[0], 2);
    for (int i = 0; i < 20 && r.pending(); ++i) {
        auto bch = r.schedule();
        for (auto x : bch.requests)
            r.complete(x, x.sample ? 2 : -1);
    }
    CHECK(r.request(r1).done && r.request(r2).done);
}
static void sampling_tests() {
    std::thread::id worker_id;
    {
        RankWorker worker;
        auto first = worker.submit([&] { worker_id = std::this_thread::get_id(); });
        first.get();
        for (int i = 0; i < 10; ++i)
            worker.submit([&] { CHECK(worker_id == std::this_thread::get_id()); }).get();
        auto failed = worker.submit([] { throw std::runtime_error("test task failure"); });
        rejects([&] { failed.get(); });
        worker.submit([&] { CHECK(worker_id == std::this_thread::get_id()); }).get();
    }
    int completed = 0;
    {
        RankWorker worker;
        for (int i = 0; i < 10; ++i)
            (void)worker.submit([&] { ++completed; });
    }
    CHECK(completed == 10);
    std::mt19937_64 r(7), r2(7);
    std::array<float, 4> x{1, 3, 2, 0};
    CHECK(sample_logits(x, {}, r) == 1);
    SamplingOptions p;
    p.temperature = 1;
    p.top_k = 2;
    p.top_p = .8f;
    for (int i = 0; i < 100; ++i)
        CHECK(sample_logits(x, p, r) == sample_logits(x, p, r2));
    rejects([&] {
        auto bad = p;
        bad.top_p = NAN;
        sample_logits(x, bad, r);
    });
    TextStream text({"STOP"});
    CHECK(text.append("hello ST") == "hello ");
    CHECK(text.append("OPmore").empty());
    CHECK(text.stopped());
    TextStream utf;
    CHECK(utf.append("\xe2").empty());
    CHECK(utf.append("\x82").empty());
    CHECK(utf.append("\xac!") == "\xe2\x82\xac!");
    TextStream broken;
    CHECK(broken.append("\xe2").empty());
    CHECK(broken.append("", true) == "\xef\xbf\xbd");
    TextStream invalid;
    CHECK(invalid.append("\x80!") == "\xef\xbf\xbd!");
    auto request = parse_chat({{"messages",
                                {{{"role", "system"}, {"content", "S"}},
                                 {{"role", "user"}, {"content", "U"}},
                                 {{"role", "assistant"}, {"content", "A"}}}},
                               {"max_tokens", 7}});
    CHECK(request.rendered == "<s>[SYSTEM_PROMPT] S[/SYSTEM_PROMPT][INST] U[/INST] A</s>");
    CHECK(request.max_new == 7);
    rejects([] { parse_chat({{"messages", nlohmann::json::array()}}); });
    rejects([] { parse_chat({{"messages", {{{"role", "tool"}, {"content", "x"}}}}}); });
    rejects([] {
        parse_chat({{"messages", {{{"role", "user"}, {"content", "x"}}}}, {"max_tokens", -1}});
    });
}
// Reference: the pre-PERF_ROUND3 sampler (fresh vectors, full-vocab
// std::sort per call). The rewritten sampler must draw IDENTICAL tokens
// for the same RNG stream, including temperature/top-k/top-p paths.
static int reference_sample_logits(std::span<const float> logits, const SamplingOptions &p,
                                   std::mt19937_64 &rng) {
    validate_sampling(p);
    if (logits.empty())
        throw std::invalid_argument("Empty logits");
    for (float v : logits)
        if (!std::isfinite(v))
            throw std::runtime_error("Nonfinite model logits");
    if (p.temperature == 0)
        return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<size_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    if (p.top_k > 0 && size_t(p.top_k) < order.size())
        order.resize(size_t(p.top_k));
    std::vector<double> probabilities;
    double total = 0;
    for (size_t i : order) {
        double v = std::exp((double(logits[i]) - double(logits[order[0]])) / p.temperature);
        probabilities.push_back(v);
        total += v;
    }
    double cumulative = 0;
    size_t keep = 0;
    do {
        cumulative += probabilities[keep++];
    } while (keep < order.size() && cumulative < p.top_p * total);
    probabilities.resize(keep);
    std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
    return int(order[dist(rng)]);
}
static void sampler_equivalence_tests() {
    std::mt19937_64 logits_rng(20261026);
    auto make = [&](size_t n, int flavor) {
        std::vector<float> x(n);
        std::uniform_real_distribution<float> u(-20.f, 20.f);
        for (float &v : x)
            switch (flavor) {
            case 1: // heavy ties: quantise to half-integers
                v = std::round(u(logits_rng) * 2) / 2;
                break;
            case 2: // sharp distribution with outliers
                v = u(logits_rng) * .01f;
                break;
            default:
                v = u(logits_rng);
            }
        if (flavor == 2 && n > 8)
            for (int i : {0, 3, 7})
                x[size_t(i)] += 25.f + float(i);
        return x;
    };
    size_t checked = 0;
    for (int flavor : {0, 1, 2})
        for (double temperature : {0., .7, 1.})
            for (int top_k : {0, 1, 2, 40, 1000, 32768})
                for (double top_p : {.05, .5, .9, 1.}) {
                    SamplingOptions p;
                    p.temperature = float(temperature);
                    p.top_k = top_k;
                    p.top_p = float(top_p);
                    for (size_t n : {size_t(513), size_t(32768)}) {
                        auto logits = make(n, flavor);
                        // Split-vocabulary form: rank halves in device order.
                        size_t half = n / 2;
                        std::span<const float> low(logits.data(), half),
                            high(logits.data() + half, n - half);
                        for (uint64_t seed : {1u, 42u}) {
                            std::mt19937_64 reference(seed), rewritten(seed), split(seed);
                            for (int draw = 0; draw < 3; ++draw) {
                                int a = reference_sample_logits(logits, p, reference);
                                int b = sample_logits(logits, p, rewritten);
                                int c = sample_logits(low, high, p, split);
                                CHECK(a == b);
                                CHECK(a == c);
                                ++checked;
                            }
                        }
                    }
                }
    CHECK(checked == 3 * 3 * 6 * 4 * 2 * 2 * 3);
}
static void tuning_tests() {
    // Identical selection reports must compare equal before serialization.
    nlohmann::json report = {{"blas_ms", timing_json(NAN)},
                             {"candidates",
                              {{{"mode", "unused"}, {"ms", timing_json(NAN)}},
                               {{"mode", "measured"}, {"ms", timing_json(.125)}}}}};
    const auto copy = report;
    CHECK(report["blas_ms"].is_null());
    CHECK(report["candidates"][0]["ms"].is_null());
    CHECK(report["candidates"][1]["ms"].is_number_float());
    CHECK(report["candidates"][1]["ms"] == .125);
    CHECK(report == report);
    CHECK(report == copy);
    CHECK(report == nlohmann::json::parse(report.dump()));
    CHECK(timing_json(INFINITY).is_null());
    CHECK(timing_json(-INFINITY).is_null());
    CHECK(timing_median({3, 1, 2}) == 2);
    CHECK(timing_median({4, 1, 2, 3}) == 2.5);
    rejects([] { timing_median({}); });
    rejects([] { timing_median({1, NAN}); });
    rejects([] { tune_candidates(0, [](size_t) { return 1.; }); });
    rejects([] { tune_candidates(2, [](size_t) { return 0.; }); });
    rejects([] { tune_candidates(2, [](size_t) { return INFINITY; }); });
    CHECK(tune_candidates(1, [](size_t) { return .4; }).selected == 0);
    // A one-percent nominal win must not churn the stable incumbent.
    CHECK(tune_candidates(2, [](size_t c) { return c ? .99 : 1.; }).selected == 0);
    CHECK(tune_candidates(2, [](size_t c) { return c ? .95 : 1.; }).selected == 1);
    for (int outlier_position = 0; outlier_position < 26; ++outlier_position) {
        int calls = 0;
        auto choice = tune_candidates(2, [&](size_t c) {
            const int index = calls++;
            // Simulate a warming GPU and one unrelated scheduling stall.
            const double clock = 1.6 - .008 * index;
            return (c ? .80 : 1.) * clock * (index == outlier_position ? 12. : 1.);
        });
        CHECK(choice.selected == 1);
    }
    int calls = 0;
    auto drift = tune_candidates(2, [&](size_t c) { return (c ? .75 : 1.) + .03 * calls++; });
    CHECK(drift.selected == 1);
    // A wider set must still reject isolated lucky/unlucky measurements.
    calls = 0;
    auto wide = tune_candidates(17, [&](size_t c) {
        const int index = calls++;
        const double nominal = c == 11 ? .4 : .8 + .017 * double(c);
        return nominal * (1.8 - .002 * index) * (index % 23 == 4 ? 8. : 1.);
    });
    CHECK(wide.selected == 11);
}
static void checkpoint_tests() {
    HostProjection matrix;
    matrix.in = 35 * 8;
    matrix.out = 67;
    matrix.codes.resize(size_t(35 * 67));
    for (size_t i = 0; i < matrix.codes.size(); ++i)
        matrix.codes[i] = uint16_t(i * 29);
    matrix.codes[0] = 65535;
    auto transposed = group_major_codes(matrix);
    CHECK(transposed.size() == matrix.codes.size());
    for (size_t o = 0; o < 67; ++o)
        for (size_t g = 0; g < 35; ++g)
            CHECK(transposed[g * 67 + o] == matrix.codes[o * 35 + g]);
    CHECK(parse_code_layout("row-major") == CodeLayout::RowMajor);
    CHECK(parse_code_layout("group-major") == CodeLayout::GroupMajor);
    CHECK(parse_code_layout("sorted") == CodeLayout::Sorted);
    CHECK(parse_code_layout("lane") == CodeLayout::Lane);
    {
        // Codebook-major layouts round-trip exactly (tails in rows and groups,
        // clustered codes forcing gap escapes, several group tiles, drift limiter).
        std::mt19937 rng(5);
        for (auto [in, out, max_groups, drift] : std::vector<std::tuple<int, int, int, int>>{
                 {1032, 37, 2047, 0}, {4104, 129, 300, 0}, {2048, 64, 2047, 1000}, {128, 16, 2047, 0}}) {
            HostProjection q;
            q.in = in;
            q.out = out;
            q.codes.resize(size_t(out) * size_t(in / 8));
            for (size_t i = 0; i < q.codes.size(); ++i)
                q.codes[i] = i % 7 ? uint16_t(rng()) : uint16_t(rng() % 300 + 60000);
            q.codes[0] = 0;
            q.codes[1] = 65535;
            auto lane = lane_layout(in, out, q.codes.data(), max_groups, drift);
            std::vector<uint16_t> back;
            CHECK(lane_layout_decode(lane, in, out, back));
            CHECK(back == q.codes);
            CHECK(lane.entries == q.codes.size());
            CHECK(lane.group_tiles == (in / 8 + max_groups - 1) / max_groups);
            auto sorted = sorted_layout(in, out, q.codes.data(), sorted_rows_per_tile,
                                        sorted_groups_per_tile, sorted_default_delta_bits);
            CHECK(sorted_layout_decode(sorted, in, out, back));
            CHECK(back == q.codes);
        }
        rejects([] {
            uint16_t c[16]{};
            lane_layout(128, 1, c, 4096);
        });
    }
    rejects([] { parse_code_layout("invalid"); });
    auto bad_matrix = matrix;
    bad_matrix.codes.pop_back();
    rejects([&] { group_major_codes(bad_matrix); });
    auto path = std::filesystem::temp_directory_path() /
                ("behemoth-test-" + std::to_string(getpid()) + ".safetensors");
    auto write = [&](nlohmann::json h, std::vector<uint16_t> data) {
        std::string header = h.dump();
        uint64_t n = header.size();
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(&n), 8);
        f.write(header.data(), std::streamsize(header.size()));
        f.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size() * 2));
    };
    write({{"weight", {{"dtype", "F16"}, {"shape", {2}}, {"data_offsets", {0, 4}}}},
           {"codes", {{"dtype", "I16"}, {"shape", {2}}, {"data_offsets", {4, 8}}}}},
          {0x3c00, 0xc000, 0xffff, 0x8000});
    {
        SafeTensorFile f(path);
        CHECK(f.value("weight", 0) == 1);
        CHECK(f.value("weight", 1) == -2);
        CHECK(f.code("codes", 0) == 65535);
        CHECK(f.code("codes", 1) == 32768);
        rejects([&] { f.value("weight", 2); });
    }
    write({{"w", {{"dtype", "F16"}, {"shape", {100}}, {"data_offsets", {0, 4}}}}}, {0, 0});
    rejects([&] { SafeTensorFile f(path); });
    std::filesystem::remove(path);
    HostProjection p;
    p.in = 8;
    p.out = 2;
    p.codes = {65535, 32768};
    p.codebook.assign(65536 * 8, encode_e4m3(1));
    p.row_scale = {2, 3};
    std::array<float, 8> input{1, 1, 1, 1, 1, 1, 1, 1};
    auto y = reference_projection(p, input);
    CHECK(std::abs(y[0] - 16) < 1e-4);
    CHECK(std::abs(y[1] - 24) < 1e-4);
}
int main() {
    try {
        for (auto [name, fn] : std::vector<std::pair<const char *, std::function<void()>>>{
                 {"numeric", numeric_tests},
                 {"attention", attention_tests},
                 {"scheduler", scheduler_tests},
                  {"sampling", sampling_tests},
                  {"sampler_equivalence", sampler_equivalence_tests},
                  {"tuning", tuning_tests},
                 {"checkpoint", checkpoint_tests}}) {
            fn();
            std::cout << "PASS " << name << '\n';
        }
        std::cout << "PASS " << assertions << " assertions\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
