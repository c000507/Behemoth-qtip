// behemoth::Engine on NeuronCores: the server's scheduler, 16-token page tables, sampling, cancellation and
// metrics, with neuron::NeuronModel as the model backend. Same public contract as runtime/src/engine.cpp
// (engine.hpp), so runtime/src/server.cpp links against it unchanged.
//
// Prefix cache: the GPU engine's PrefixCache and scheduler hooks unchanged, with synchronous copies on the loop thread
// (forward() is blocking, so saves and restores run between steps): a finished request's pages stay resident as the
// cache's device tier (zero-copy hit for the next request with that prefix) and are copied to a host arena when the
// pool needs them back.
//
// Structured outputs / tool calls: the GPU engine's per-request TokenConstraint, applied host-side to each sampled row
// after the logit processing (unconstrained sample first when the constraint allows it, else the allowed-token mask),
// advanced with every sampled token. Draft rows are sampled in order and only while the drafts match, so a row's
// constraint state is that of the tokens before it, and the state equals the emitted tokens (rows after a mismatch
// are discarded unsampled). The state covers generated tokens only: a preempted request re-prefills its history and
// keeps its constraint, and prefix-cache restores (prompt K/V) do not touch it.
#include "behemoth/engine.hpp"
#include "behemoth/prefix_cache.hpp"
#include "behemoth/constraint.hpp"
#include "behemoth/sampling.hpp"
#include "neuron_model.hpp"
#include <algorithm>
#include <atomic>
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
    // structured outputs: constraints of the constrained requests; per-token cost of the constraint calls
    std::map<uint64_t, std::shared_ptr<TokenConstraint>> constraints;
    uint64_t constrained_requests = 0;
    std::atomic<uint64_t> constraint_ns{0}, constrained_tokens{0}, constraint_fallbacks{0};
    // prefix cache (null when disabled): leases of requests restoring a prefix, requests waiting for their restore
    std::unique_ptr<PrefixCache> cache;
    std::map<uint64_t, PrefixCache::Lease> leases;
    std::vector<uint64_t> awaiting;
    std::vector<std::byte> arena; // host tier: host page h at h * model.kv_page_bytes()
    uint64_t gpu_hits = 0, host_hits = 0, restored_tokens = 0, saves = 0, saved_pages = 0, failed_saves = 0;
    double save_ms = 0, restore_ms = 0;
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
        if (const char *v = std::getenv("BEHEMOTH_NEURON_QTIP")) // expanded (default) | compressed | auto
            m.qtip_mode = v;
        if (const char *v = std::getenv("BEHEMOTH_NEURON_HBM_GIB"))
            m.hbm_gib = std::atof(v);
        if (const char *v = std::getenv("BEHEMOTH_NEURON_LOAD_THREADS")) // layers prepared and uploaded concurrently (default 8)
            m.load_threads = std::max(1, std::atoi(v));
        if (const char *v = std::getenv("BEHEMOTH_NEURON_HBM_RESERVE_GIB"))
            m.hbm_reserve_gib = std::atof(v);
        return m;
    }
    explicit Impl(const RuntimeOptions &o)
        : model(neuron_options(o)), scheduler(model.cache_pages(), o.token_budget, o.prefill_chunk, o.scheduler, model.vocab()),
          eos(model.eos().begin(), model.eos().end()), max_context(size_t(model.max_context())) {
        max_rows = model.max_decode_rows();
        spec_tokens = std::min(env_int("BEHEMOTH_NEURON_SPEC_TOKENS", 7), max_rows - 1);
        if (model.recurrent()) // a rejected draft row would already have advanced the recurrent state
            spec_tokens = 0;
        spec_requests = env_int("BEHEMOTH_NEURON_SPEC_REQUESTS", 4);
        spec_rows = std::min(env_int("BEHEMOTH_NEURON_SPEC_ROWS", 8), max_rows);
        spec_ngram = std::max(1, env_int("BEHEMOTH_NEURON_SPEC_NGRAM", 3));
        if (const char *v = std::getenv("BEHEMOTH_NEURON_SPEC_PAGE_COST"))
            spec_page_cost = std::atof(v);
        if (o.prefix_cache.slots && model.recurrent())
            std::cerr << "Prefix cache: off (the model keeps recurrent state that restored K/V pages do not carry)\n";
        if (o.prefix_cache.slots && env_int("BEHEMOTH_NEURON_PREFIX_CACHE", 1) && !model.recurrent()) {
            auto co = o.prefix_cache;
            // host tier: --prefix-cache-gb, capped by BEHEMOTH_NEURON_PREFIX_HOST_GB (default 8: the instance's RAM also
            // holds the embeddings and the load-time staging)
            const char *cap_env = std::getenv("BEHEMOTH_NEURON_PREFIX_HOST_GB");
            const double cap = std::min(std::max(0.0, o.prefix_cache_gb), cap_env ? std::atof(cap_env) : 8.0) * 1e9;
            co.host_pages = size_t(cap / double(model.kv_page_bytes()));
            if (!co.host_pages)
                throw std::invalid_argument("Prefix cache memory cap below one page");
            cache = std::make_unique<PrefixCache>(co);
            arena.resize(co.host_pages * model.kv_page_bytes()); // untouched pages stay unbacked
            scheduler.set_retain_finished(true);
            scheduler.set_reclaim([this](size_t need) {
                std::vector<int> pages;
                cache->reclaim_gpu(need, pages);
                return pages;
            });
            std::cerr << "Prefix cache: " << co.slots << " slots, host tier " << double(arena.size()) / 1e9 << " GB ("
                      << co.host_pages << " pages of " << model.kv_page_bytes() / 1024 << " KiB), device tier "
                      << (co.gpu_tier ? "on" : "off") << "\n";
        }
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

    // Under the lock: a done request's sequence (prompt + generated tokens with KV) is offered to the prefix cache, the
    // pages it needs copied to the host tier now (synchronously: no step is running), then the request is erased.
    void retire(uint64_t id) {
        auto lease = leases.find(id);
        if (cache) {
            const auto &r = scheduler.request(id);
            auto pages = scheduler.take_pages(id);
            std::vector<int> release;
            PrefixCache::SavePlan plan;
            if (!failed && r.cached && pages.size() >= ceil_div(r.cached, 16)) {
                const PrefixCache::Lease *restored =
                    lease != leases.end() && r.restored && r.restored == lease->second.tokens ? &lease->second : nullptr;
                plan = cache->insert(std::span<const int>(r.history.data(), r.cached), r.prompt_size, restored, release);
            }
            if (!plan.slot)
                release.insert(release.end(), pages.begin(), pages.end());
            else {
                const size_t first = plan.first_page, n = plan.slot->host.size();
                bool ok = true;
                if (first < n)
                    try {
                        const auto t0 = Clock::now();
                        model.kv_save(std::span<const int>(pages.data() + first, n - first),
                                      std::span<const int>(plan.slot->host.data() + first, n - first), arena.data());
                        save_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                        ++saves;
                        saved_pages += n - first;
                    } catch (const std::exception &e) {
                        std::cerr << "Prefix cache: save failed: " << e.what() << "\n";
                        ok = false;
                        ++failed_saves;
                    }
                if (ok)
                    cache->saved(plan, std::move(pages), release);
                else {
                    cache->abandon(plan);
                    release.insert(release.end(), pages.begin(), pages.end());
                }
            }
            scheduler.release_pages(release);
        }
        if (lease != leases.end()) {
            cache->release(lease->second);
            leases.erase(lease);
        }
        scheduler.erase(id);
    }

    // Under the lock, before scheduling: restore the prefixes of waiting requests, from the device tier (the slot's
    // resident pages, no copy) or the host tier into freshly allocated pages (whole prefix, else whatever whole pages fit).
    void start_restores() {
        for (auto it = awaiting.begin(); it != awaiting.end();) {
            const uint64_t id = *it;
            auto &lease = leases.at(id);
            auto slot = lease.slot;
            if (slot->failed) {
                scheduler.abort_restore(id);
                cache->release(lease);
                leases.erase(id);
                it = awaiting.erase(it);
                continue;
            }
            if (!slot->ready) {
                ++it;
                continue;
            }
            size_t m = lease.tokens;
            std::vector<int> release;
            auto gpu = cache->take_gpu(slot, ceil_div(m, 16), release);
            scheduler.release_pages(release);
            if (!gpu.empty()) {
                scheduler.begin_restore(id, std::move(gpu));
                scheduler.finish_restore(id, m);
                ++gpu_hits;
                restored_tokens += m;
                it = awaiting.erase(it);
                continue;
            }
            std::vector<int> table;
            if (!scheduler.allocate_pages(table, m)) {
                const size_t fit = std::min(m, scheduler.free_pages() * 16);
                if (fit < cache->options().min_tokens || !scheduler.allocate_pages(table, fit)) {
                    ++it;
                    continue;
                }
                m = lease.tokens = fit;
            }
            scheduler.begin_restore(id, table);
            try {
                const auto t0 = Clock::now();
                model.kv_restore(std::span<const int>(lease.host.data(), ceil_div(m, 16)), std::span<const int>(table.data(), ceil_div(m, 16)), arena.data());
                restore_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                scheduler.finish_restore(id, m);
                ++host_hits;
                restored_tokens += m;
            } catch (const std::exception &e) {
                std::cerr << "Prefix cache: restore failed: " << e.what() << "\n";
                scheduler.abort_restore(id);
                cache->release(lease);
                leases.erase(id);
            }
            it = awaiting.erase(it);
        }
    }

    void forget(uint64_t id) { // under the lock
        retire(id);
        callbacks.erase(id);
        rng.erase(id);
        submitted.erase(id);
        last_token.erase(id);
        acceptance.erase(id);
        constraints.erase(id);
    }

    // One token of a constrained request from `row` (logits after processing), committed to the constraint.
    int sample_constrained(std::span<const float> row, TokenConstraint &constraint, const SamplingOptions &sampling,
                           std::mt19937_64 &generator) {
        using C = std::chrono::steady_clock;
        C::duration spent{};
        int token = -1;
        if (constraint.accept_first(sampling.temperature == 0)) {
            token = sample_logits(row, sampling, generator);
            const auto t0 = C::now();
            if (!constraint.accepts(token))
                token = -1;
            spent += C::now() - t0;
        }
        if (token < 0) {
            const auto t0 = C::now();
            const auto allowed = constraint.allowed();
            spent += C::now() - t0;
            constraint_fallbacks.fetch_add(1, std::memory_order_relaxed);
            if (allowed.count == 1)
                token = allowed.single;
            else {
                const uint64_t *mask = allowed.mask->data();
                thread_local std::vector<uint64_t> padded; // an LM head wider than the tokenizer: no ids past it
                if (allowed.mask->size() * 64 < row.size()) {
                    padded.assign((row.size() + 63) / 64, 0);
                    std::copy(allowed.mask->begin(), allowed.mask->end(), padded.begin());
                    mask = padded.data();
                }
                token = sample_logits(row, {}, sampling, generator, mask);
            }
        }
        const auto t0 = C::now();
        constraint.advance(token);
        spent += C::now() - t0;
        constraint_ns.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(spent).count()), std::memory_order_relaxed);
        constrained_tokens.fetch_add(1, std::memory_order_relaxed);
        return token;
    }

    void loop() {
        for (;;) {
            std::vector<std::pair<GenerationCallback, GenerationEvent>> events;
            ScheduledBatch step;
            InferenceBatch input;
            std::vector<TokenConstraint *> step_constraints; // per step request (null: unconstrained), taken under the lock
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
                    if (callbacks.contains(*it) && !scheduler.request(*it).done) {
                        if (scheduler.request(*it).restoring) { // restores are synchronous: only a waiting one can be cancelled
                            awaiting.erase(std::find(awaiting.begin(), awaiting.end(), *it));
                            scheduler.abort_restore(*it);
                        }
                        scheduler.cancel(*it);
                    }
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
                if (cache)
                    start_restores();
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
                    for (const auto &s : step.requests) {
                        auto c = constraints.find(s.id);
                        step_constraints.push_back(c == constraints.end() ? nullptr : c->second.get());
                    }
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
                        const auto &req = scheduler.request(step.requests[i].id);
                        const bool process = has_logit_processing(req.sampling);
                        thread_local std::vector<float> processed;
                        auto d = drafts.find(step.requests[i].id);
                        TokenConstraint *constraint = step_constraints[i];
                        for (size_t j = 0; j < sampled[i].size(); ++j) {
                            auto row = logits.subspan((logits_row[i] + j) * vocab, vocab);
                            if (process) { // penalties see the tokens before this row: history, then drafts 0 .. j - 1
                                processed.assign(row.begin(), row.end());
                                std::span<const int> extra;
                                if (d != drafts.end())
                                    extra = std::span<const int>(d->second.data(), j);
                                apply_logit_processing(processed, req.sampling, req.history, req.prompt_size, extra);
                                row = processed;
                            }
                            if (!constraint) {
                                sampled[i][j] = sample_logits(row, req.sampling, rng.at(step.requests[i].id));
                                continue;
                            }
                            sampled[i][j] = sample_constrained(row, *constraint, req.sampling, rng.at(step.requests[i].id));
                            // row j + 1 follows draft j: past a mismatch it is discarded, and the constraint must
                            // not see it (its state stays that of the emitted tokens)
                            if (d == drafts.end() || j >= d->second.size() || sampled[i][j] != d->second[j])
                                break;
                        }
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
    if (prompt.empty() || prompt.size() + max_new > impl_->max_context)
        throw std::invalid_argument("prompt and completion exceed the Neuron context bucket (" + std::to_string(impl_->max_context) + " tokens)");
    std::lock_guard lock(impl_->mutex);
    if (impl_->failed || impl_->stop)
        throw std::runtime_error("Engine unavailable");
    const uint64_t id = impl_->scheduler.submit(prompt, max_new, options);
    if (impl_->cache)
        if (auto lease = impl_->cache->lookup(prompt)) {
            impl_->scheduler.wait_restore(id);
            impl_->leases.emplace(id, std::move(*lease));
            impl_->awaiting.push_back(id);
        }
    impl_->callbacks[id] = std::move(callback);
    impl_->rng.emplace(id, std::mt19937_64(options.seed));
    if (constraint) {
        impl_->constraints.emplace(id, std::move(constraint));
        ++impl_->constrained_requests;
    }
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
            {"last_step", impl_->last_profile},
            {"structured_output", {{"requests", impl_->constrained_requests},
                                   {"tokens", impl_->constrained_tokens.load()},
                                   {"mask_fallbacks", impl_->constraint_fallbacks.load()},
                                   {"constraint_ms_total", double(impl_->constraint_ns.load()) / 1e6},
                                   {"constraint_us_per_token", impl_->constrained_tokens.load()
                                                                   ? double(impl_->constraint_ns.load()) / 1e3 / double(impl_->constrained_tokens.load())
                                                                   : 0.0}}},
            {"prefix_cache", impl_->cache ? nlohmann::json{{"slots", impl_->cache->slots()},
                                                           {"hits", impl_->cache->stats().hits},
                                                           {"misses", impl_->cache->stats().misses},
                                                           {"device_hits", impl_->gpu_hits},
                                                           {"host_hits", impl_->host_hits},
                                                           {"restored_tokens", impl_->restored_tokens},
                                                           {"cached_tokens", impl_->cache->cached_tokens()},
                                                           {"device_pages", impl_->cache->gpu_pages()},
                                                           {"host_pages_used", impl_->cache->host_pages_used()},
                                                           {"saves", impl_->saves},
                                                           {"saved_pages", impl_->saved_pages},
                                                           {"failed_saves", impl_->failed_saves},
                                                           {"save_ms", impl_->save_ms},
                                                           {"restore_ms", impl_->restore_ms},
                                                           {"awaiting_restore", impl_->awaiting.size()}}
                                           : nlohmann::json(nullptr)}};
}
} // namespace behemoth
