#include "behemoth/model.hpp"
#include "behemoth/qtip_gpu.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

using namespace behemoth;
namespace {
// QTIP prefill paths (PREFILL_SCHED_REPORT.md): one 300-token prompt as a single 300-row step,
// (a) WMMA trellis kernel + per-chunk transforms (the pre-change path), (b) dense hipBLASLt
// prefill + row-batched/fused transforms. The transforms are bit-identical by construction;
// the GEMM summation order differs, so logits must agree within tolerance and the argmax must
// match. Then the same prompt in mixed 1-decode + 64-token steps through the scheduler.
int prefill_paths(const std::string &model_dir, int bf16_rows = 0) {
    RuntimeOptions o;
    o.model = model_dir;
    o.token_budget = 512;
    o.prefill_chunk = 512;
    o.zero_copy_transport = true;
    o.qtip_bf16_prefill_rows = bf16_rows;
    TensorParallelModel model(o);
    std::mt19937 rng(3);
    std::vector<int> prompt(300);
    for (auto &t : prompt)
        t = int(rng() % 32000) + 100;
    auto run = [&](int dense_rows, int row_transforms, int overlap = 0) {
        qtip::set_dense_rows_threshold(dense_rows);
        qtip::set_row_transform_threshold(row_transforms);
        set_qtip_overlap_rows(overlap);
        Scheduler scheduler(model.cache_pages(), 512, 512);
        scheduler.submit(prompt, 1);
        auto step = scheduler.schedule();
        auto logits = model.forward(make_inference_batch(scheduler, step));
        return std::vector<float>(logits.begin(), logits.end());
    };
    auto wmma = run(0, 0), dense = run(1, 1), dense_chunked = run(1, 0), wmma_rows = run(0, 1),
         overlapped = run(1, 1, 2);
    auto argmax = [](const std::vector<float> &x) {
        return size_t(std::max_element(x.begin(), x.end()) - x.begin());
    };
    auto rel = [](const std::vector<float> &a, const std::vector<float> &b) -> double {
        double se = 0, sr = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]))
                return INFINITY;
            se += double(a[i] - b[i]) * double(a[i] - b[i]);
            sr += double(b[i]) * double(b[i]);
        }
        return std::sqrt(se / sr);
    };
    int failures = 0;
    auto check = [&](bool ok, const std::string &what) {
        std::cout << (ok ? "PASS " : "FAIL ") << what << '\n';
        failures += !ok;
    };
    check(wmma == wmma_rows, "row-batched transforms bit-identical through the full model (WMMA path)");
    check(dense == dense_chunked, "row-batched transforms bit-identical through the full model (dense path)");
    const double e = rel(dense, wmma);
    // Random-token prompt through 88 layers of bf16 activations: any change of fp32 summation
    // order is amplified to a few percent of the logits (chunking the same WMMA prefill into
    // 64-token steps alone gives ~3e-2, checked below). Quality gate: behemoth_eval ppl
    // (PREFILL_SCHED_REPORT.md: 2.2612 WMMA vs 2.2605 dense, 4096 tokens).
    check(e < 1e-1 && argmax(dense) == argmax(wmma),
          "dense prefill vs WMMA prefill logits rel_rms=" + std::to_string(e) + " argmax " +
              std::to_string(argmax(dense)) + "/" + std::to_string(argmax(wmma)));
    const double eo = rel(overlapped, dense);
    check(overlapped == dense,
          "two overlapped micro-batches (2 streams, copy-engine channels) vs one dense step rel_rms=" +
              std::to_string(eo));
    qtip::set_dense_rows_threshold(64);
    qtip::set_row_transform_threshold(64);
    {
        // mixed steps: a decoding request plus the prompt in 64-token pieces
        SchedulerPolicy p;
        p.mixed_prefill_tokens = 64;
        Scheduler scheduler(model.cache_pages(), 512, 512, p);
        auto d = scheduler.submit(std::vector<int>{1, 2, 3}, 50);
        auto first = scheduler.schedule();
        auto l0 = model.forward(make_inference_batch(scheduler, first));
        scheduler.complete(first.requests[0], int(std::max_element(l0.begin(), l0.end()) - l0.begin()));
        auto id = scheduler.submit(prompt, 1);
        std::vector<float> last;
        int steps = 0;
        while (!scheduler.request(id).done && ++steps < 100) {
            auto step = scheduler.schedule();
            auto batch = make_inference_batch(scheduler, step);
            auto logits = model.forward(batch);
            size_t row = 0;
            for (const auto &r : step.requests) {
                if (!r.sample)
                    continue;
                const float *base = logits.data();
                const size_t n = batch.sample_rows.size();
                std::vector<float> full(32768);
                std::copy_n(base + row * 16384, 16384, full.begin());
                std::copy_n(base + n * 16384 + row * 16384, 16384, full.begin() + 16384);
                if (r.id == id)
                    last = full;
                scheduler.complete(r, int(std::max_element(full.begin(), full.end()) - full.begin()));
                ++row;
            }
            for (const auto &r : step.requests)
                if (!r.sample)
                    scheduler.complete(r, -1);
        }
        (void)d;
        check(steps == 5 && !last.empty(), "mixed decode + 64-token prefill steps: " + std::to_string(steps));
        // logits rank-blocked in `wmma`: [rank0 16384 | rank1 16384] for one sampled row
        const double em = last.empty() ? INFINITY : rel(last, wmma);
        check(em < 1e-1 && !last.empty() && argmax(last) == argmax(wmma),
              "chunked mixed-step prefill vs single-step WMMA prefill logits rel_rms=" + std::to_string(em));
    }
    std::cout << (failures ? "PREFILL PATH TESTS FAILED" : "PREFILL PATH TESTS PASSED") << '\n';
    return failures ? 1 : 0;
}

// Prefix cache (PREFIX_CACHE_REPORT.md): KV pages of a 1700-token prompt are saved to pinned host
// pages and restored into different (permuted) GPU pages, then a 60-token tail is prefilled.
//  - logits must equal those of a cold prefill with the same step split (1700 + 60) bit for bit;
//  - save -> restore -> save reproduces the host bytes exactly (both ranks);
//  - a restore ending inside a page (1690 tokens, tail recomputes the rest of that page) and
//    the restored prefix vs one 1760-token cold prefill are compared by rel_rms / argmax.
int prefix_cache_paths(const std::string &model_dir, int bf16_rows = 0) {
    RuntimeOptions o;
    o.model = model_dir;
    o.qtip_bf16_prefill_rows = bf16_rows;
    o.token_budget = 2048;
    o.prefill_chunk = 2048;
    o.zero_copy_transport = true;
    o.kv_transfer_pages = 32;
    TensorParallelModel model(o);
    std::mt19937 rng(11);
    const size_t prefix = 1700, tail = 60;
    std::vector<int> full(prefix + tail);
    for (auto &t : full)
        t = int(rng() % 32000) + 100;
    const std::vector<int> head(full.begin(), full.begin() + std::ptrdiff_t(prefix));
    auto argmax = [](const std::vector<float> &x) {
        return size_t(std::max_element(x.begin(), x.end()) - x.begin());
    };
    auto rel = [](const std::vector<float> &a, const std::vector<float> &b) -> double {
        double se = 0, sr = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]))
                return INFINITY;
            se += double(a[i] - b[i]) * double(a[i] - b[i]);
            sr += double(b[i]) * double(b[i]);
        }
        return std::sqrt(se / sr);
    };
    // Runs a request to completion (max_new 1); returns the sampled logits ([rank0 | rank1]).
    auto run = [&](Scheduler &s, uint64_t id) {
        std::vector<float> last;
        while (!s.request(id).done) {
            auto st = s.schedule();
            if (st.requests.empty())
                throw std::runtime_error("Nothing scheduled");
            auto lg = model.forward(make_inference_batch(s, st));
            for (const auto &r : st.requests)
                if (r.sample) {
                    last.assign(lg.begin(), lg.end());
                    s.complete(r, int(argmax(last)));
                } else
                    s.complete(r, -1);
        }
        return last;
    };
    int failures = 0;
    auto check = [&](bool ok, const std::string &what) {
        std::cout << (ok ? "PASS " : "FAIL ") << what << '\n';
        failures += !ok;
    };
    const int pages = model.cache_pages();
    // A: cold prefill, steps of 1700 + 60 tokens.
    std::vector<float> cold_split;
    {
        Scheduler s(pages, 2048, prefix);
        auto id = s.submit(full, 1);
        cold_split = run(s, id);
        s.erase(id);
    }
    // A': cold prefill, steps of 1690 + 70 tokens (the partial-page restore's split).
    std::vector<float> cold_split_partial;
    {
        Scheduler s(pages, 2048, prefix - 10);
        auto id = s.submit(full, 1);
        cold_split_partial = run(s, id);
        s.erase(id);
    }
    // D: cold prefill in one 1760-token step.
    std::vector<float> cold_one;
    {
        Scheduler s(pages, 2048, 2048);
        auto id = s.submit(full, 1);
        cold_one = run(s, id);
        s.erase(id);
    }
    Scheduler s(pages, 2048, 2048);
    s.set_retain_finished(true);
    auto first = s.submit(head, 1);
    run(s, first);
    auto saved_pages = s.take_pages(first);
    s.erase(first);
    const size_t n = saved_pages.size();
    check(n == (prefix + 15) / 16, "prompt pages " + std::to_string(n));
    model.kv_host_reserve(2 * n);
    std::vector<int> host(n), host2(n);
    std::iota(host.begin(), host.end(), 0);
    std::iota(host2.begin(), host2.end(), int(n));
    auto timed = [&](uint64_t t, const char *what) {
        model.kv_wait(t);
        std::array<float, 2> ms{};
        if (!model.kv_poll(t, &ms))
            throw std::runtime_error("KV transfer not complete after wait");
        const double gb = double(n) * double(model.kv_page_bytes()) / 1e9;
        std::cout << what << ": " << n << " pages, " << gb << " GB per GPU, rank0 " << ms[0]
                  << " ms (" << gb / (ms[0] / 1e3) << " GB/s), rank1 " << ms[1] << " ms ("
                  << gb / (ms[1] / 1e3) << " GB/s)\n";
    };
    timed(model.kv_save(saved_pages, host), "save");
    s.release_pages(saved_pages);
    std::vector<int> spacer;
    s.allocate_pages(spacer, 50 * 16); // shift the free list so the restore lands elsewhere
    auto restore_into = [&](uint64_t id, size_t tokens) {
        s.wait_restore(id);
        std::vector<int> table;
        if (!s.allocate_pages(table, tokens))
            throw std::runtime_error("No pages for restore");
        std::reverse(table.begin(), table.end()); // any page permutation must work
        std::vector<int> src(host.begin(), host.begin() + std::ptrdiff_t(table.size()));
        auto t = model.kv_restore(src, table);
        timed(t, "restore");
        return table;
    };
    // B: restored 1700-token prefix + 60-token tail.
    auto b = s.submit(full, 1);
    auto table = restore_into(b, prefix);
    bool moved = false;
    for (size_t k = 0; k < table.size(); ++k)
        moved |= table[k] != saved_pages[k];
    check(moved, "restore targets different physical pages");
    {
        model.kv_wait(model.kv_save(table, host2));
        size_t mismatched = 0;
        for (int rank = 0; rank < 2; ++rank)
            for (size_t k = 0; k < n; ++k) {
                auto x = model.kv_host_page(rank, host[k]), y = model.kv_host_page(rank, host2[k]);
                mismatched += std::memcmp(x.data(), y.data(), x.size()) != 0;
            }
        check(mismatched == 0, "save -> restore -> save reproduces every host page byte for byte "
                               "(both ranks), mismatched pages=" + std::to_string(mismatched));
    }
    s.begin_restore(b, table);
    s.finish_restore(b, prefix);
    auto restored = run(s, b);
    {
        auto p = s.take_pages(b);
        s.erase(b);
        s.release_pages(p);
    }
    check(restored == cold_split,
          "restored prefix + tail logits bit-identical to cold prefill with the same step split");
    // C: restore ending inside a page: 1690 tokens (106 pages); the tail step recomputes
    // positions 1690.. of the last restored page.
    auto c = s.submit(full, 1);
    auto table_c = restore_into(c, prefix - 10);
    s.begin_restore(c, table_c);
    s.finish_restore(c, prefix - 10);
    auto partial = run(s, c);
    {
        auto p = s.take_pages(c);
        s.erase(c);
        s.release_pages(p);
    }
    // The restored KV of positions < 1690 was computed in a 1700-row step, the cold
    // reference's in a 1690-row step (different GEMM/micro-batch row tiling, so not bit-equal:
    // rel_rms 0.03-0.05 measured, like any change of step split).
    auto top5 = [](const std::vector<float> &x, size_t id) {
        size_t above = 0;
        for (float v : x)
            above += v > x[id];
        return above < 5;
    };
    const double ec = rel(partial, cold_split_partial);
    std::cout << "partial-page restore vs cold 1690+70: bit-identical="
              << (partial == cold_split_partial) << " rel_rms=" << ec << " argmax "
              << argmax(partial) << "/" << argmax(cold_split_partial) << '\n';
    check(ec < 1e-1 && top5(cold_split_partial, argmax(partial)) &&
              top5(partial, argmax(cold_split_partial)),
          "restore of 1690 tokens (partial page) + 70-token tail vs cold prefill with the same "
          "split within tolerance (rel_rms " + std::to_string(ec) + ")");
    // Restored prefix + short tail vs one long cold prefill: differs like any change of step
    // split. Random-token prompts give nearly flat logits, so greedy agreement is reported and
    // only top-5 containment is required (dense GEMM autotune picks per process, so the cold
    // split-vs-single difference itself varies between runs).
    const double ed = rel(restored, cold_one), es = rel(cold_split, cold_one),
                 ep = rel(partial, cold_split);
    std::cout << "restored 1700 + 60 vs cold single 1760-row step: rel_rms=" << ed
              << " argmax " << argmax(restored) << "/" << argmax(cold_one)
              << "; reference cold 1700+60 vs cold single: rel_rms=" << es
              << "; partial restore vs cold 1700+60: rel_rms=" << ep << " argmax "
              << argmax(partial) << "/" << argmax(cold_split) << '\n';
    check(ed < 1e-1 && top5(cold_one, argmax(restored)) && top5(restored, argmax(cold_one)),
          "restored prefix + tail vs one cold prefill within tolerance (rel_rms " +
              std::to_string(ed) + ", greedy token " +
              (argmax(restored) == argmax(cold_one) ? "equal" : "in the other's top 5") + ")");
    s.release_pages(spacer);
    check(s.free_pages() == size_t(pages), "no pages leaked");
    std::cout << (failures ? "PREFIX CACHE TESTS FAILED" : "PREFIX CACHE TESTS PASSED") << '\n';
    return failures ? 1 : 0;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[2]) == "prefill")
            return prefill_paths(argv[1]);
        if (argc == 3 && std::string(argv[2]) == "prefill-bf16")
            return prefill_paths(argv[1], 64);
        if (argc == 3 && std::string(argv[2]) == "prefix")
            return prefix_cache_paths(argv[1]);
        if (argc == 3 && std::string(argv[2]) == "prefix-bf16")
            return prefix_cache_paths(argv[1], 64);
        if (argc != 2) {
            std::cerr << "Usage: behemoth_model_tests MODEL_DIRECTORY [prefill|prefix|prefill-bf16|prefix-bf16]\n";
            return 2;
        }
        RuntimeOptions o;
        o.model = argv[1];
        o.token_budget = 2;
        o.prefill_chunk = 2;
        // Exercise the production zero-copy transport incl. the fused
        // publish-in-merge reduce/residual path (PERF_ROUND3 item 4).
        o.zero_copy_transport = true;
        o.zero_copy_rows = 8;
        TensorParallelModel model(o);
        Scheduler scheduler(model.cache_pages(), 2, 2);
        auto id = scheduler.submit(std::vector<int>{1, 2}, 2);
        auto scheduled = scheduler.schedule();
        auto batch = make_inference_batch(scheduler, scheduled);
        auto first = model.forward(batch);
        std::vector<float> first_copy(first.begin(), first.end()); // staging is reused
        auto repeat = model.forward(batch);
        auto tuning = model.projection_tuning();
        if (tuning.at("last_step_query_rows") != 2 || tuning.at("last_step_sample_rows") != 1 ||
            tuning.at("ranks").size() != 2)
            throw std::runtime_error("Tuning snapshot has wrong prefill dimensions");
        if (first_copy.size() != 32768)
            throw std::runtime_error("Wrong logits shape");
        for (size_t i = 0; i < 32768; ++i)
            if (!std::isfinite(first_copy[i]) ||
                std::abs(first_copy[i] - repeat[i]) > 1e-5f)
                throw std::runtime_error("Nonfinite/nondeterministic full-model logits");
        scheduler.complete(scheduled.requests[0], 7);
        auto next = scheduler.schedule();
        auto decoded = model.forward(make_inference_batch(scheduler, next));
        tuning = model.projection_tuning();
        if (tuning.at("last_step_query_rows") != 1 || tuning.at("profiling_enabled") != false)
            throw std::runtime_error("Tuning snapshot did not switch to decode rows");
        for (const auto &rank : tuning.at("ranks")) {
            if (rank.at("selections").size() != 8)
                throw std::runtime_error("Tuning snapshot omitted projections");
            for (const auto &selection : rank.at("selections")) {
                if (selection.at("rows") != 1 || !selection.at("weight_name").is_string())
                    throw std::runtime_error("Tuning snapshot included dormant prefill choices");
                auto name = selection.at("weight_name").get<std::string>();
                if (name != "lm_head" && !name.starts_with("model.layers.0."))
                    throw std::runtime_error("Tuning snapshot has wrong weight labels");
            }
        }
        for (float v : decoded)
            if (!std::isfinite(v))
                throw std::runtime_error("Nonfinite decode logits");
        scheduler.complete(next.requests[0], 8);
        if (!scheduler.request(id).done || scheduler.free_pages() != size_t(scheduler.capacity()))
            throw std::runtime_error("Request leaked KV pages");
        std::cout << "PASS full-model prefill, repeatability, decode and cache reclamation (not a "
                     "quality evaluation)\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL (full-model acceptance NOT passed): " << e.what() << '\n';
        return 1;
    }
}
