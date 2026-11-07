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
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <thread>

namespace behemoth {
// Speculative decoding by prompt lookup (no draft model): a decode step of a request may carry draft tokens after
// its scheduled token; the model computes logits for the scheduled token and every draft position in one launch
// (rows of a decode bucket). Draft i is verified when the token sampled at row i - 1 equals it; the first mismatch
// ends the run and the token sampled there is still exact, so every accepted token is distributed as in sequential
// decoding (greedy output is identical). Drafts come from the most recent earlier occurrence of the last `ngram`
// tokens of the history: the tokens that followed it. Cost: the step runs a wider bucket (R = rows + drafts).
using Drafts = std::map<uint64_t, std::vector<int>>;

std::vector<int> prompt_lookup(const std::vector<int> &history, int max_draft, int ngram = 3) {
    const size_t len = history.size();
    for (int n = std::min<int>(ngram, int(len) - 1); n >= 1 && max_draft > 0; --n) {
        size_t best_end = 0, best_avail = 0; // the latest match with a full-length draft, else the longest draft
        for (size_t end = len - 1; end >= size_t(n) && best_avail < size_t(max_draft); --end) { // end: index after a match (not the suffix itself)
            const size_t start = end - size_t(n);
            const size_t avail = std::min<size_t>(size_t(max_draft), len - end); // the tokens that followed the match
            if (avail > best_avail && std::equal(history.begin() + std::ptrdiff_t(start), history.begin() + std::ptrdiff_t(end), history.end() - n)) {
                best_end = end;
                best_avail = avail;
            }
        }
        if (best_avail)
            return std::vector<int>(history.begin() + std::ptrdiff_t(best_end), history.begin() + std::ptrdiff_t(best_end + best_avail));
    }
    return {};
}

InferenceBatch make_inference_batch(const Scheduler &scheduler, const ScheduledBatch &batch, const Drafts &drafts = {}) {
    InferenceBatch input;
    auto &l = input.layout;
    l.physical_pages = scheduler.capacity();
    l.query_offsets = {0};
    l.table_stride = 1;
    auto draft_of = [&](uint64_t id) -> const std::vector<int> * {
        auto it = drafts.find(id);
        return it == drafts.end() ? nullptr : &it->second;
    };
    for (const auto &step : batch.requests) {
        const auto *d = draft_of(step.id);
        l.table_stride = std::max(l.table_stride, int(ceil_div(step.begin + step.count + (d ? d->size() : 0), 16)));
    }
    for (const auto &step : batch.requests) {
        const auto &r = scheduler.request(step.id);
        const auto *d = draft_of(step.id);
        const size_t extra = d ? d->size() : 0, count = step.count + extra;
        l.lengths.push_back(int(step.begin + count));
        for (size_t i = 0; i < count; ++i) {
            input.tokens.push_back(i < step.count ? r.history[step.begin + i] : (*d)[i - step.count]);
            l.positions.push_back(int(step.begin + i));
        }
        l.query_offsets.push_back(int(input.tokens.size()));
        size_t base = l.page_table.size();
        l.page_table.resize(base + size_t(l.table_stride), -1);
        std::copy_n(r.pages.begin(), ceil_div(step.begin + count, 16), l.page_table.begin() + std::ptrdiff_t(base));
        if (step.sample) // the scheduled token's row and every draft row
            for (size_t i = step.count - 1; i < count; ++i) {
                input.sample_rows.push_back(int(input.tokens.size() - count + i));
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
    uint64_t completed = 0, tokens = 0, steps = 0, drafted_tokens = 0, accepted_drafts = 0;
    double model_seconds = 0;
    nlohmann::json last_profile;
    // prompt-lookup speculation: up to BEHEMOTH_NEURON_SPEC_TOKENS drafts per request (0 disables), only in steps of at
    // most BEHEMOTH_NEURON_SPEC_REQUESTS decode requests and BEHEMOTH_NEURON_SPEC_ROWS rows after drafting (the bucket a
    // verify step may grow to), matching the last BEHEMOTH_NEURON_SPEC_NGRAM tokens
    int spec_tokens = 0, spec_requests = 4, spec_rows = 8, spec_ngram = 3, max_rows = 1;
    // Adaptive: a verify step costs about spec_page_cost ms per draft row and table page of the bucket it lands in (the
    // paged K/V gather, measured 0.006-0.013 on the 8B), on top of a plain step (plain_step_ms, running mean). A request
    // drafts only while its running acceptance (accepted drafts per verify step, EMA, optimistic start) beats twice that
    // overhead measured in plain steps, and a request that stopped drafting retries as its estimate recovers (~10 steps).
    double spec_page_cost = 0.01, plain_step_ms = 40;
    std::map<uint64_t, double> acceptance;
    Drafts drafts;
    std::thread thread;

    static int env_int(const char *name, int fallback) {
        const char *v = std::getenv(name);
        return v ? std::atoi(v) : fallback;
    }

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
        : model(neuron_options(o)), scheduler(model.cache_pages(), o.token_budget, o.prefill_chunk, o.scheduler, model.vocab()),
          eos(model.eos().begin(), model.eos().end()), max_context(size_t(model.max_context())) {
        max_rows = model.max_decode_rows();
        spec_tokens = std::min(env_int("BEHEMOTH_NEURON_SPEC_TOKENS", 7), max_rows - 1);
        spec_requests = env_int("BEHEMOTH_NEURON_SPEC_REQUESTS", 4);
        spec_rows = std::min(env_int("BEHEMOTH_NEURON_SPEC_ROWS", 8), max_rows);
        spec_ngram = std::max(1, env_int("BEHEMOTH_NEURON_SPEC_NGRAM", 3));
        if (const char *v = std::getenv("BEHEMOTH_NEURON_SPEC_PAGE_COST"))
            spec_page_cost = std::atof(v);
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
        acceptance.erase(id);
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
                drafts.clear();
                if (!step.requests.empty()) {
                    // Draft only for small decode steps, and never past spec_rows rows in total: rows are the cost of a
                    // decode bucket, and a wide step already has rows to spare on throughput, not latency.
                    int rows = int(step.requests.size());
                    const int budget = rows <= spec_requests ? std::min(spec_tokens, (spec_rows - rows) / rows) : 0;
                    if (budget > 0)
                        for (const auto &s : step.requests) {
                            const auto &r = scheduler.request(s.id);
                            if (s.count != 1 || !s.sample || s.begin + 1 + size_t(budget) > max_context)
                                continue;
                            double &alpha = acceptance.try_emplace(s.id, 1.0).first->second;
                            const double overhead = spec_page_cost * budget * model.decode_bucket_pages(int(ceil_div(s.begin + 1 + size_t(budget), size_t(16))));
                            // a verify step within 10 % of a plain one is always worth trying; dearer ones need the acceptance to pay
                            if (overhead > 0.1 * plain_step_ms && alpha * plain_step_ms < 2 * overhead) { // recover slowly to retry
                                alpha = std::min(1.0, alpha + 0.02);
                                continue;
                            }
                            auto draft = prompt_lookup(r.history, budget, spec_ngram);
                            if (!draft.empty() && scheduler.reserve(s.id, s.begin + 1 + draft.size())) {
                                rows += int(draft.size());
                                drafts[s.id] = std::move(draft);
                            }
                        }
                    input = make_inference_batch(scheduler, step, drafts);
                }
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
                // sampled[i]: the tokens of request i's rows (scheduled token, then one per draft row), in row order
                std::vector<std::vector<int>> sampled(step.requests.size());
                std::vector<size_t> logits_row(step.requests.size(), 0);
                std::vector<size_t> sampling_steps;
                for (size_t i = 0, row = 0; i < step.requests.size(); ++i)
                    if (step.requests[i].sample) {
                        logits_row[i] = row;
                        auto d = drafts.find(step.requests[i].id);
                        sampled[i].assign(1 + (d == drafts.end() ? 0 : d->second.size()), -1);
                        row += sampled[i].size();
                        sampling_steps.push_back(i);
                    }
                // Each request is an independent pass over its rows of 128k logits with its own RNG: spread them over threads.
                std::exception_ptr sampling_error;
                std::mutex error_mutex;
#pragma omp parallel for schedule(dynamic, 1)
                for (long k = 0; k < long(sampling_steps.size()); ++k) {
                    const size_t i = sampling_steps[size_t(k)];
                    try {
                        for (size_t j = 0; j < sampled[i].size(); ++j)
                            sampled[i][j] = sample_logits(logits.subspan((logits_row[i] + j) * vocab, vocab),
                                                          scheduler.request(step.requests[i].id).sampling, rng.at(step.requests[i].id));
                    } catch (...) {
                        std::lock_guard guard(error_mutex);
                        sampling_error = std::current_exception();
                    }
                }
                if (sampling_error)
                    std::rethrow_exception(sampling_error);
                std::lock_guard lock(mutex);
                model_seconds += std::chrono::duration<double>(Clock::now() - start).count();
                last_profile = model.last_profile();
                ++steps;
                size_t step_tokens = 0, step_drafts = 0;
                for (size_t i = 0; i < step.requests.size(); ++i) {
                    const auto &request = step.requests[i];
                    auto emit = [&](int token) {
                        const auto &after = scheduler.request(request.id);
                        ++tokens;
                        ++step_tokens;
                        const auto now = Clock::now();
                        const bool later = last_token.contains(request.id);
                        (later ? itl : ttft).push_back(std::chrono::duration<double, std::milli>(now - (later ? last_token.at(request.id) : submitted.at(request.id))).count());
                        for (auto *samples : {&ttft, &itl})
                            if (samples->size() > 1024)
                                samples->pop_front();
                        last_token[request.id] = now;
                        events.push_back({callbacks.at(request.id), {request.id, token, false, "", "", after.prompt_size, after.generated, after.restored}});
                    };
                    const int token = sampled[i].empty() ? -1 : sampled[i][0];
                    scheduler.complete(request, token, token >= 0 && eos.contains(token));
                    if (token >= 0)
                        emit(token);
                    // verified drafts: row j's token was sampled after draft j - 1; it stands when that draft matched
                    auto d = drafts.find(request.id);
                    if (d != drafts.end()) {
                        step_drafts += d->second.size();
                        size_t accepted = 0;
                        for (size_t j = 1; j < sampled[i].size() && !scheduler.request(request.id).done && sampled[i][j - 1] == d->second[j - 1]; ++j) {
                            scheduler.accept(request.id, sampled[i][j], eos.contains(sampled[i][j]));
                            emit(sampled[i][j]);
                            ++accepted;
                        }
                        accepted_drafts += accepted;
                        if (auto a = acceptance.find(request.id); a != acceptance.end())
                            a->second = 0.5 * (a->second + double(accepted));
                    }
                    const auto &after = scheduler.request(request.id);
                    if (after.done) {
                        ++completed;
                        events.push_back({callbacks.at(request.id), {request.id, -1, true, after.reason, "", after.prompt_size, after.generated, after.restored}});
                    }
                }
                drafted_tokens += step_drafts;
                if (!step_drafts && step.requests.size() == 1 && step.requests[0].count == 1) // a plain single decode step
                    plain_step_ms = 0.9 * plain_step_ms + 0.1 * std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                if (std::getenv("BEHEMOTH_NEURON_STEP_LOG"))
                    std::cerr << "step rows=" << step.requests.size() << " drafts=" << step_drafts << " tokens=" << step_tokens << " wall_ms="
                              << std::chrono::duration<double, std::milli>(Clock::now() - start).count() << " " << last_profile.dump() << "\n";
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
            {"drafted_tokens", impl_->drafted_tokens},
            {"accepted_drafts", impl_->accepted_drafts},
            {"model_seconds", impl_->model_seconds},
            {"ttft_ms_p50", percentile(impl_->ttft, .5)},
            {"ttft_ms_p95", percentile(impl_->ttft, .95)},
            {"itl_ms_p50", percentile(impl_->itl, .5)},
            {"itl_ms_p95", percentile(impl_->itl, .95)},
            {"last_step", impl_->last_profile}};
}
} // namespace behemoth
