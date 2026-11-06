// behemoth::Engine on NeuronCores: the server's scheduler, 16-token page tables, sampling, cancellation and
// metrics, with neuron::NeuronModel as the model backend. Same public contract as runtime/src/engine.cpp
// (engine.hpp), so runtime/src/server.cpp links against it unchanged.
//
// Not carried over from the GPU engine: the prefix cache (KV save/restore) and structured-output constraints;
// submit() rejects a constraint instead of ignoring it.
#include "behemoth/engine.hpp"
#include "behemoth/constraint.hpp"
#include "behemoth/sampling.hpp"
#include "neuron_model.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <thread>

namespace behemoth {
InferenceBatch make_inference_batch(const Scheduler &scheduler, const ScheduledBatch &batch) {
    InferenceBatch input;
    auto &l = input.layout;
    l.physical_pages = scheduler.capacity();
    l.query_offsets = {0};
    l.table_stride = 1;
    for (const auto &step : batch.requests)
        l.table_stride = std::max(l.table_stride, int(ceil_div(step.begin + step.count, 16)));
    for (const auto &step : batch.requests) {
        const auto &r = scheduler.request(step.id);
        l.lengths.push_back(int(step.begin + step.count));
        for (size_t i = 0; i < step.count; ++i) {
            input.tokens.push_back(r.history[step.begin + i]);
            l.positions.push_back(int(step.begin + i));
        }
        l.query_offsets.push_back(int(input.tokens.size()));
        size_t base = l.page_table.size();
        l.page_table.resize(base + size_t(l.table_stride), -1);
        std::copy_n(r.pages.begin(), ceil_div(step.begin + step.count, 16), l.page_table.begin() + std::ptrdiff_t(base));
        if (step.sample) {
            input.sample_rows.push_back(int(input.tokens.size() - 1));
            input.sample_ids.push_back(step.id);
        }
    }
    return input;
}

struct Engine::Impl {
    using Clock = std::chrono::steady_clock;
    neuron::NeuronModel model;
    Scheduler scheduler;
    std::set<int> eos;
    size_t max_context;
    mutable std::mutex mutex;
    std::condition_variable cv;
    bool stop = false, failed = false;
    std::map<uint64_t, GenerationCallback> callbacks;
    std::map<uint64_t, std::mt19937_64> rng;
    std::set<uint64_t> cancellations;
    std::map<uint64_t, Clock::time_point> submitted, last_token;
    std::deque<double> ttft, itl;
    uint64_t completed = 0, tokens = 0, steps = 0;
    double model_seconds = 0;
    nlohmann::json last_profile;
    std::thread thread;

    static neuron::ModelOptions neuron_options(const RuntimeOptions &o) {
        neuron::ModelOptions m;
        m.model = o.model;
        const char *dir = std::getenv("BEHEMOTH_NEURON_NEFF_DIR");
        if (!dir)
            throw std::invalid_argument("set BEHEMOTH_NEURON_NEFF_DIR to the directory of traced NEFF buckets");
        m.neff_dir = dir;
        return m;
    }
    explicit Impl(const RuntimeOptions &o)
        : model(neuron_options(o)), scheduler(model.cache_pages(), o.token_budget, o.prefill_chunk, o.scheduler),
          eos(model.eos().begin(), model.eos().end()), max_context(size_t(model.max_context())) {
        thread = std::thread([this] { loop(); });
    }
    ~Impl() { shutdown(); }

    void shutdown() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        cv.notify_all();
        if (thread.joinable() && std::this_thread::get_id() != thread.get_id())
            thread.join();
    }

    static void deliver(std::vector<std::pair<GenerationCallback, GenerationEvent>> &events) {
        for (auto &[cb, e] : events)
            try {
                cb(e);
            } catch (...) {
            }
        events.clear();
    }

    void forget(uint64_t id) { // under the lock
        scheduler.erase(id);
        callbacks.erase(id);
        rng.erase(id);
        submitted.erase(id);
        last_token.erase(id);
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
                    deliver(events);
                    return;
                }
                for (auto it = cancellations.begin(); it != cancellations.end(); it = cancellations.erase(it))
                    if (callbacks.contains(*it) && !scheduler.request(*it).done)
                        scheduler.cancel(*it);
                for (auto it = callbacks.begin(); it != callbacks.end();) {
                    const auto &r = scheduler.request(it->first);
                    if (!r.done) {
                        ++it;
                        continue;
                    }
                    events.push_back({it->second, {r.id, -1, true, r.reason, "", r.prompt_size, r.generated, r.restored}});
                    const uint64_t id = r.id;
                    ++it;
                    forget(id);
                }
                step = scheduler.schedule();
                if (!step.requests.empty())
                    input = make_inference_batch(scheduler, step);
            }
            deliver(events);
            if (step.requests.empty()) {
                std::unique_lock lock(mutex);
                if (!callbacks.empty())
                    cv.wait_for(lock, std::chrono::milliseconds(1));
                continue;
            }
            try {
                const auto start = Clock::now();
                const auto logits = model.forward(input);
                const size_t vocab = size_t(model.vocab());
                std::vector<int> sampled(step.requests.size(), -1);
                size_t row = 0;
                for (size_t i = 0; i < step.requests.size(); ++i) {
                    const auto &request = step.requests[i];
                    if (!request.sample)
                        continue;
                    sampled[i] = sample_logits(logits.subspan(row * vocab, vocab), scheduler.request(request.id).sampling, rng.at(request.id));
                    ++row;
                }
                std::lock_guard lock(mutex);
                model_seconds += std::chrono::duration<double>(Clock::now() - start).count();
                last_profile = model.last_profile();
                ++steps;
                for (size_t i = 0; i < step.requests.size(); ++i) {
                    const auto &request = step.requests[i];
                    const int token = sampled[i];
                    scheduler.complete(request, token, token >= 0 && eos.contains(token));
                    const auto &after = scheduler.request(request.id);
                    if (token >= 0) {
                        ++tokens;
                        const auto now = Clock::now();
                        const bool later = last_token.contains(request.id);
                        (later ? itl : ttft).push_back(std::chrono::duration<double, std::milli>(now - (later ? last_token.at(request.id) : submitted.at(request.id))).count());
                        for (auto *samples : {&ttft, &itl})
                            if (samples->size() > 1024)
                                samples->pop_front();
                        last_token[request.id] = now;
                        events.push_back({callbacks.at(request.id), {request.id, token, false, "", "", after.prompt_size, after.generated, after.restored}});
                    }
                    if (after.done) {
                        ++completed;
                        events.push_back({callbacks.at(request.id), {request.id, -1, true, after.reason, "", after.prompt_size, after.generated, after.restored}});
                    }
                }
                for (const auto &request : step.requests)
                    if (scheduler.request(request.id).done)
                        forget(request.id);
            } catch (const std::exception &error) {
                std::vector<std::pair<GenerationCallback, GenerationEvent>> errors;
                {
                    std::lock_guard lock(mutex);
                    failed = stop = true;
                    for (auto &[id, cb] : callbacks)
                        errors.push_back({cb, {id, -1, true, "error", error.what()}});
                    callbacks.clear();
                }
                deliver(errors);
                return;
            }
            deliver(events);
        }
    }
};

Engine::Engine(const RuntimeOptions &options) : impl_(std::make_unique<Impl>(options)) {}
Engine::~Engine() = default;
uint64_t Engine::submit(std::span<const int> prompt, size_t max_new, SamplingOptions options, GenerationCallback callback,
                        std::shared_ptr<TokenConstraint> constraint) {
    validate_sampling(options);
    if (!callback)
        throw std::invalid_argument("Missing generation callback");
    if (!max_new)
        throw std::invalid_argument("max_tokens must be positive");
    if (constraint)
        throw std::invalid_argument("structured outputs are not supported by the Neuron backend");
    if (prompt.empty() || prompt.size() + max_new > impl_->max_context)
        throw std::invalid_argument("prompt and completion exceed the Neuron context bucket (" + std::to_string(impl_->max_context) + " tokens)");
    std::lock_guard lock(impl_->mutex);
    if (impl_->failed || impl_->stop)
        throw std::runtime_error("Engine unavailable");
    const uint64_t id = impl_->scheduler.submit(prompt, max_new, options);
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
void Engine::shutdown() { impl_->shutdown(); }
bool Engine::healthy() const {
    std::lock_guard lock(impl_->mutex);
    return !impl_->failed && !impl_->stop;
}
nlohmann::json Engine::projection_tuning(int) const { return nlohmann::json::object(); }
nlohmann::json Engine::metrics() const {
    auto percentile = [](const std::deque<double> &values, double p) {
        if (values.empty())
            return 0.;
        std::vector<double> x(values.begin(), values.end());
        std::sort(x.begin(), x.end());
        return x[size_t(p * double(x.size() - 1))];
    };
    std::lock_guard lock(impl_->mutex);
    return {{"backend", "neuron"},
            {"active_requests", impl_->scheduler.pending()},
            {"kv_pages_total", impl_->scheduler.capacity()},
            {"kv_pages_free", impl_->scheduler.free_pages()},
            {"preemptions", impl_->scheduler.preemptions()},
            {"completed_requests", impl_->completed},
            {"generated_tokens", impl_->tokens},
            {"steps", impl_->steps},
            {"model_seconds", impl_->model_seconds},
            {"ttft_ms_p50", percentile(impl_->ttft, .5)},
            {"ttft_ms_p95", percentile(impl_->ttft, .95)},
            {"itl_ms_p50", percentile(impl_->itl, .5)},
            {"itl_ms_p95", percentile(impl_->itl, .95)},
            {"last_step", impl_->last_profile}};
}
} // namespace behemoth
