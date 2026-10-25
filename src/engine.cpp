#include "behemoth/engine.hpp"
#include "behemoth/sampling.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <set>
#include <thread>

namespace behemoth {
struct Engine::Impl {
    TensorParallelModel model;
    Scheduler scheduler;
    mutable std::mutex mutex;
    std::condition_variable cv;
    bool stop = false, failed = false;
    std::map<uint64_t, GenerationCallback> callbacks;
    std::map<uint64_t, std::mt19937_64> rng;
    std::set<uint64_t> cancellations;
    std::thread worker;
    uint64_t completed = 0, tokens = 0;
    double gpu_seconds = 0;
    nlohmann::json last_profile;
    using Clock = std::chrono::steady_clock;
    std::map<uint64_t, Clock::time_point> submitted, last_token;
    std::deque<double> ttft, itl;
    explicit Impl(const RuntimeOptions &o)
        : model(o), scheduler(model.cache_pages(), o.token_budget, o.prefill_chunk),
          worker([this] { loop(); }) {}
    void shutdown() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable())
            worker.join();
    }
    ~Impl() {
        shutdown();
    }
    void loop() {
        for (;;) {
            std::vector<std::pair<GenerationCallback, GenerationEvent>> events;
            ScheduledBatch step;
            InferenceBatch input;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] { return stop || !callbacks.empty() || !cancellations.empty(); });
                if (stop) {
                    for (const auto &[id, cb] : callbacks)
                        events.push_back({cb, {id, -1, true, "cancelled", "Engine shutting down"}});
                    lock.unlock();
                    for (auto &[cb, e] : events)
                        try {
                            cb(e);
                        } catch (...) {
                        }
                    return;
                }
                for (uint64_t id : cancellations)
                    if (callbacks.contains(id))
                        scheduler.cancel(id);
                cancellations.clear();
                for (auto it = callbacks.begin(); it != callbacks.end();) {
                    const auto &r = scheduler.request(it->first);
                    if (r.done) {
                        events.push_back(
                            {it->second,
                             {r.id, -1, true, r.reason, "", r.prompt_size, r.generated}});
                        scheduler.erase(r.id);
                        rng.erase(it->first);
                        submitted.erase(it->first);
                        last_token.erase(it->first);
                        it = callbacks.erase(it);
                    } else
                        ++it;
                }
                step = scheduler.schedule();
                if (!step.requests.empty())
                    input = make_inference_batch(scheduler, step);
            }
            for (auto &[cb, e] : events)
                try {
                    cb(e);
                } catch (...) {
                }
            events.clear();
            if (step.requests.empty())
                continue;
            try {
                auto start = std::chrono::steady_clock::now();
                const size_t sample_count = input.sample_rows.size();
                const auto logits = model.forward(input);
                double elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                {
                    std::lock_guard lock(mutex);
                    gpu_seconds += elapsed;
                    last_profile = model.last_profile();
                    size_t index = 0;
                    for (const auto &request : step.requests) {
                        const auto &r = scheduler.request(request.id);
                        // Logits are rank-blocked; sample without an interleave copy.
                        const size_t row = index++;
                        int token =
                            request.sample
                                ? sample_logits(
                                      std::span<const float>(logits.data() + row * 16384, 16384),
                                      std::span<const float>(
                                          logits.data() + sample_count * 16384 + row * 16384,
                                          16384),
                                      r.sampling, rng.at(request.id))
                                : -1;
                        scheduler.complete(request, token, token == 2);
                        const auto &after = scheduler.request(request.id);
                        if (token >= 0) {
                            ++tokens;
                            auto now = Clock::now();
                            auto &samples = last_token.contains(request.id) ? itl : ttft;
                            samples.push_back(std::chrono::duration<double, std::milli>(
                                                  now - (last_token.contains(request.id)
                                                             ? last_token.at(request.id)
                                                             : submitted.at(request.id)))
                                                  .count());
                            if (samples.size() > 1024)
                                samples.pop_front();
                            last_token[request.id] = now;
                            events.push_back({callbacks.at(request.id),
                                              {request.id, token, false, "", "", after.prompt_size,
                                               after.generated}});
                        }
                        if (after.done) {
                            ++completed;
                            events.push_back({callbacks.at(request.id),
                                              {request.id, -1, true, after.reason, "",
                                               after.prompt_size, after.generated}});
                        }
                    }
                    for (const auto &request : step.requests)
                        if (scheduler.request(request.id).done) {
                            scheduler.erase(request.id);
                            callbacks.erase(request.id);
                            rng.erase(request.id);
                            submitted.erase(request.id);
                            last_token.erase(request.id);
                        }
                }
                for (auto &[cb, e] : events)
                    try {
                        cb(e);
                    } catch (...) {
                    }
            } catch (const std::exception &error) {
                std::vector<std::pair<GenerationCallback, GenerationEvent>> errors;
                {
                    std::lock_guard lock(mutex);
                    failed = true;
                    stop = true;
                    for (auto &[id, cb] : callbacks)
                        errors.push_back({cb, {id, -1, true, "error", error.what()}});
                    callbacks.clear();
                }
                for (auto &[cb, e] : errors)
                    try {
                        cb(e);
                    } catch (...) {
                    }
                return;
            }
        }
    }
};
Engine::Engine(const RuntimeOptions &options) : impl_(std::make_unique<Impl>(options)) {}
Engine::~Engine() = default;
uint64_t Engine::submit(std::span<const int> prompt, size_t max_new, SamplingOptions options,
                        GenerationCallback callback) {
    validate_sampling(options);
    if (!callback)
        throw std::invalid_argument("Missing generation callback");
    if (!max_new)
        throw std::invalid_argument("max_tokens must be positive");
    std::lock_guard lock(impl_->mutex);
    if (impl_->failed || impl_->stop)
        throw std::runtime_error("Engine unavailable");
    uint64_t id = impl_->scheduler.submit(prompt, max_new, options);
    impl_->callbacks[id] = std::move(callback);
    impl_->rng.emplace(id, std::mt19937_64(options.seed));
    impl_->submitted[id] = Impl::Clock::now();
    impl_->cv.notify_one();
    return id;
}
void Engine::cancel(uint64_t id) {
    std::lock_guard lock(impl_->mutex);
    impl_->cancellations.insert(id);
    impl_->cv.notify_one();
}
void Engine::shutdown() {
    impl_->shutdown();
}
bool Engine::healthy() const {
    std::lock_guard lock(impl_->mutex);
    return !impl_->failed && !impl_->stop;
}
nlohmann::json Engine::projection_tuning(int layer) const {
    return impl_->model.projection_tuning(layer);
}
nlohmann::json Engine::metrics() const {
    auto percentile = [](const std::deque<double> &values, double p) {
        if (values.empty())
            return 0.;
        std::vector<double> x(values.begin(), values.end());
        std::sort(x.begin(), x.end());
        return x[size_t(p * double(x.size() - 1))];
    };
    std::lock_guard lock(impl_->mutex);
    return {{"active_requests", impl_->scheduler.pending()},
            {"kv_pages_total", impl_->scheduler.capacity()},
            {"kv_pages_free", impl_->scheduler.free_pages()},
            {"preemptions", impl_->scheduler.preemptions()},
            {"completed_requests", impl_->completed},
            {"completion_tokens", impl_->tokens},
            {"gpu_step_seconds", impl_->gpu_seconds},
            {"last_step_profile", impl_->last_profile},
            {"ttft_p50_ms", percentile(impl_->ttft, .5)},
            {"ttft_p95_ms", percentile(impl_->ttft, .95)},
            {"inter_token_p50_ms", percentile(impl_->itl, .5)},
            {"inter_token_p95_ms", percentile(impl_->itl, .95)},
            {"precision", "fp8_e4m3/bf16/fp32"},
            {"healthy", !impl_->failed && !impl_->stop}};
}
} // namespace behemoth
