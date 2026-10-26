#include "behemoth/model.hpp"
#include "behemoth/qtip_gpu.hpp"
#include <cmath>
#include <iostream>
#include <random>

using namespace behemoth;
namespace {
// QTIP prefill paths (PREFILL_SCHED_REPORT.md): one 300-token prompt as a single 300-row step,
// (a) WMMA trellis kernel + per-chunk transforms (the pre-change path), (b) dense hipBLASLt
// prefill + row-batched/fused transforms. The transforms are bit-identical by construction;
// the GEMM summation order differs, so logits must agree within tolerance and the argmax must
// match. Then the same prompt in mixed 1-decode + 64-token steps through the scheduler.
int prefill_paths(const std::string &model_dir) {
    RuntimeOptions o;
    o.model = model_dir;
    o.token_budget = 512;
    o.prefill_chunk = 512;
    o.zero_copy_transport = true;
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
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[2]) == "prefill")
            return prefill_paths(argv[1]);
        if (argc != 2) {
            std::cerr << "Usage: behemoth_model_tests MODEL_DIRECTORY [prefill]\n";
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
