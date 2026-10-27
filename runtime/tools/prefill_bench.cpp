// QTIP prefill step benchmark (PREFILL_SCHED_REPORT.md): loads the model once and times single
// forward steps of one request, `rows` new tokens at context offset `ctx`, under the prefill
// path settings:
//   behemoth_prefill_bench MODEL [rows,...] [ctx,...] [reps]
// Each configuration line: path (wmma / dense / dense+overlap), rows, context, median ms and
// tokens/s. Settings: dense threshold (qtip::set_dense_rows_threshold), micro-batch overlap
// (set_qtip_overlap_rows), row-batched transforms (qtip::set_row_transform_threshold).
#include "behemoth/model.hpp"
#include "behemoth/qtip_gpu.hpp"
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace behemoth;
namespace {
std::vector<int> list(const char *s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        v.push_back(std::stoi(item));
    return v;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: behemoth_prefill_bench MODEL [rows,...] [ctx,...] [reps]\n";
            return 2;
        }
        const auto rows_list = argc > 2 ? list(argv[2]) : std::vector<int>{256, 512, 1024, 2048};
        // rows "0": decode-only run (BEHEMOTH_BENCH_DECODE), no prefill cases
        const auto ctx_list = argc > 3 ? list(argv[3]) : std::vector<int>{0, 8192};
        const int reps = argc > 4 ? std::stoi(argv[4]) : 3;
        if (rows_list.empty() || ctx_list.empty() || reps < 1 ||
            *std::min_element(rows_list.begin(), rows_list.end()) < 0 ||
            *std::max_element(rows_list.begin(), rows_list.end()) > 4096 ||
            *std::min_element(ctx_list.begin(), ctx_list.end()) < 0)
            throw std::invalid_argument("need rows 1..4096, nonnegative contexts and positive reps");
        const int max_rows = std::max(1, *std::max_element(rows_list.begin(), rows_list.end()));
        RuntimeOptions o;
        o.model = argv[1];
        o.token_budget = o.prefill_chunk = size_t(max_rows);
        o.zero_copy_transport = true;
        o.memory_fraction = 0.9;
        if (const char *value = std::getenv("BEHEMOTH_QTIP_BF16_PREFILL_ROWS"))
            o.qtip_bf16_prefill_rows = std::stoi(value);
        // BEHEMOTH_BENCH_PROFILE=1: GPU event timings of the last timed step (adds overhead).
        if (const char *value = std::getenv("BEHEMOTH_BENCH_ZC_ROWS"))
            o.zero_copy_rows = size_t(std::stoi(value)); // zero-copy exchange buffer rows (default 8)
        o.profile = std::getenv("BEHEMOTH_BENCH_PROFILE") && std::atoi(std::getenv("BEHEMOTH_BENCH_PROFILE"));
        // BEHEMOTH_BENCH_DECODE="1,4,16": decode steps of c one-token requests at each context
        // (the prefix pages are shared; each request appends its token to its own page, so
        // contexts are rounded up to a multiple of 16).
        const auto decode_list = std::getenv("BEHEMOTH_BENCH_DECODE")
                                     ? list(std::getenv("BEHEMOTH_BENCH_DECODE")) : std::vector<int>{};
        if (!decode_list.empty())
            o.token_budget = o.prefill_chunk = std::max<size_t>(
                o.token_budget, size_t(*std::max_element(decode_list.begin(), decode_list.end())));
        TensorParallelModel model(o);
        auto report = [&](const char *name, int rows, int ctx, std::vector<double> ms, int rows_divisor) {
            ms.erase(ms.begin()); // first: plan creation / warm-up
            std::sort(ms.begin(), ms.end());
            const double med = ms[ms.size() / 2];
            std::cout << name << " rows=" << rows << " ctx=" << ctx << " " << med << " ms "
                      << rows_divisor / med * 1000 << " tok/s bf16-prefill-rows=" << o.qtip_bf16_prefill_rows << '\n';
            if (o.profile) {
                auto p = model.last_profile();
                std::cout << "  profile execution_wall_ms=" << p.value("execution_wall_ms", 0.0);
                for (const auto &r : p.at("ranks"))
                    std::cout << " rank" << r.at("rank") << "=" << r.at("gpu_event_totals").dump();
                std::cout << '\n';
            }
            std::cout << std::flush;
        };
        auto decode_cases = [&] {
          for (int ctx0 : ctx_list)
            for (int c : decode_list) {
                const int ctx = int(ceil_div(size_t(ctx0), 16) * 16);
                InferenceBatch b;
                auto &l = b.layout;
                l.physical_pages = model.cache_pages();
                const int prefix_pages = ctx / 16, per = prefix_pages + 1;
                if (size_t(per) * size_t(c) > size_t(l.physical_pages)) {
                    std::cout << "decode rows=" << c << " ctx=" << ctx << " skipped: exceeds the KV pool\n";
                    continue;
                }
                l.table_stride = per;
                l.query_offsets = {0};
                for (int r = 0; r < c; ++r) {
                    l.query_offsets.push_back(r + 1);
                    l.lengths.push_back(ctx + 1);
                    l.positions.push_back(ctx);
                    b.tokens.push_back(100 + (r * 7919) % 30000);
                    for (int p = 0; p < per; ++p)
                        l.page_table.push_back(r * per + p);
                    b.sample_rows.push_back(r);
                    b.sample_ids.push_back(uint64_t(r));
                }
                // every request gets its own prefix KV (no page aliasing): c prefills of ctx tokens
                for (int r = 0; r < c; ++r)
                    for (int start = 0; start < ctx; start += max_rows) {
                        const int end = std::min(ctx, start + max_rows);
                        InferenceBatch warm;
                        warm.layout.physical_pages = l.physical_pages;
                        warm.layout.table_stride = prefix_pages;
                        for (int p = 0; p < prefix_pages; ++p)
                            warm.layout.page_table.push_back(r * per + p);
                        warm.layout.query_offsets = {0, end - start};
                        warm.layout.lengths = {end};
                        for (int pos = start; pos < end; ++pos) {
                            warm.layout.positions.push_back(pos);
                            warm.tokens.push_back(100 + (pos * 7919) % 30000);
                        }
                        (void)model.forward(warm);
                    }
                std::vector<double> ms;
                for (int r = 0; r < reps + 1; ++r) {
                    const auto t0 = std::chrono::steady_clock::now();
                    (void)model.forward(b);
                    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                report("decode", c, ctx, ms, c);
            }
        };
        const int dense_default = qtip::dense_rows_threshold();
        struct Path {
            const char *name;
            int dense, overlap;
        };
        const Path paths[] = {{"wmma", 0, 0}, {"dense", dense_default, 0}, {"dense+overlap", dense_default, -1}};
        for (int ctx : ctx_list)
            for (int rows : rows_list)
                for (const auto &p : paths) {
                    if (rows < 1 || (p.dense > 0 && rows < p.dense))
                        continue;
                    // BEHEMOTH_BENCH_PATHS: subset of "wmma,dense,overlap" (default all)
                    if (const char *only = std::getenv("BEHEMOTH_BENCH_PATHS")) {
                        const std::string key = p.overlap < 0 ? "overlap" : p.dense ? "dense" : "wmma";
                        if (std::string(only).find(key) == std::string::npos)
                            continue;
                    }
                    if (p.overlap < 0 && rows < 2 * p.dense)
                        continue;
                    qtip::set_dense_rows_threshold(p.dense);
                    set_qtip_overlap_rows(p.overlap);
                    InferenceBatch b;
                    auto &l = b.layout;
                    l.physical_pages = model.cache_pages();
                    if (size_t(ctx) + size_t(rows) > size_t(l.physical_pages) * 16)
                        throw std::invalid_argument("benchmark context exceeds the KV pool");
                    const int total = ctx + rows;
                    l.table_stride = int(ceil_div(size_t(total), 16));
                    l.query_offsets = {0, rows};
                    l.lengths = {total};
                    for (int i = 0; i < rows; ++i) {
                        l.positions.push_back(ctx + i);
                        b.tokens.push_back(100 + (i * 7919) % 30000);
                    }
                    l.page_table.resize(size_t(l.table_stride));
                    std::iota(l.page_table.begin(), l.page_table.end(), 0);
                    b.sample_rows = {rows - 1};
                    b.sample_ids = {0};
                    // ctx>0 must contain real KV. The old benchmark read uninitialised
                    // cache pages beyond earlier cases' short prompts. Initialise the
                    // whole prefix outside timing for each path, on the same page table.
                    for (int start = 0; start < ctx; start += max_rows) {
                        const int end = std::min(ctx, start + max_rows);
                        InferenceBatch warm;
                        warm.layout.physical_pages = l.physical_pages;
                        warm.layout.table_stride = l.table_stride;
                        warm.layout.page_table = l.page_table;
                        warm.layout.query_offsets = {0, end - start};
                        warm.layout.lengths = {end};
                        for (int pos = start; pos < end; ++pos) {
                            warm.layout.positions.push_back(pos);
                            warm.tokens.push_back(100 + (pos * 7919) % 30000);
                        }
                        (void)model.forward(warm);
                    }
                    std::vector<double> ms;
                    for (int r = 0; r < reps + 1; ++r) {
                        const auto t0 = std::chrono::steady_clock::now();
                        (void)model.forward(b);
                        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    }
                    report(p.name, rows, ctx, ms, rows);
                }
        qtip::set_dense_rows_threshold(dense_default);
        set_qtip_overlap_rows(-1);
        decode_cases();
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
