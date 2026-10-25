#include "behemoth/model.hpp"
#include <cmath>
#include <iostream>

using namespace behemoth;
int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: behemoth_model_tests MODEL_DIRECTORY\n";
            return 2;
        }
        RuntimeOptions o;
        o.model = argv[1];
        o.token_budget = 2;
        o.prefill_chunk = 2;
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
