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
        const auto ctx_list = argc > 3 ? list(argv[3]) : std::vector<int>{0, 8192};
        const int reps = argc > 4 ? std::stoi(argv[4]) : 3;
        const int max_rows = *std::max_element(rows_list.begin(), rows_list.end());
        RuntimeOptions o;
        o.model = argv[1];
        o.token_budget = o.prefill_chunk = size_t(max_rows);
        o.zero_copy_transport = true;
        o.memory_fraction = 0.9;
        TensorParallelModel model(o);
        const int dense_default = qtip::dense_rows_threshold();
        struct Path {
            const char *name;
            int dense, overlap;
        };
        const Path paths[] = {{"wmma", 0, 0}, {"dense", dense_default, 0}, {"dense+overlap", dense_default, -1}};
        for (int ctx : ctx_list)
            for (int rows : rows_list)
                for (const auto &p : paths) {
                    if (p.dense > 0 && rows < p.dense)
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
                    std::vector<double> ms;
                    for (int r = 0; r < reps + 1; ++r) {
                        const auto t0 = std::chrono::steady_clock::now();
                        (void)model.forward(b);
                        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    }
                    ms.erase(ms.begin()); // first: plan creation / warm-up
                    std::sort(ms.begin(), ms.end());
                    const double med = ms[ms.size() / 2];
                    std::cout << p.name << " rows=" << rows << " ctx=" << ctx << " " << med << " ms "
                              << rows / med * 1000 << " tok/s\n"
                              << std::flush;
                }
        qtip::set_dense_rows_threshold(dense_default);
        set_qtip_overlap_rows(-1);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
