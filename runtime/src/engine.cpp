#include "behemoth/engine.hpp"
#include "behemoth/sampling.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

namespace behemoth {
// Small persistent pool for per-request host sampling (a full-vocabulary sort per request at
// temperature > 0): run() executes the jobs on the pool and the calling thread and returns when
// all are done. Exceptions are rethrown on the caller.
class SamplerPool {
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable work_, done_;
    const std::vector<std::function<void()>> *jobs_ = nullptr;
    size_t next_ = 0, finished_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
    void drain(std::unique_lock<std::mutex> &lock) {
        while (jobs_ && next_ < jobs_->size()) {
            const size_t i = next_++;
            const auto *jobs = jobs_;
            lock.unlock();
            std::exception_ptr e;
            try {
                (*jobs)[i]();
            } catch (...) {
                e = std::current_exception();
            }
            lock.lock();
            if (e && !error_)
                error_ = e;
            if (++finished_ == jobs->size())
                done_.notify_all();
        }
    }

  public:
    explicit SamplerPool(unsigned n) {
        for (unsigned t = 0; t < n; ++t)
            threads_.emplace_back([this] {
                std::unique_lock lock(mutex_);
                uint64_t seen = 0;
                for (;;) {
                    work_.wait(lock, [&] { return stop_ || generation_ != seen; });
                    if (stop_)
                        return;
                    seen = generation_;
                    drain(lock);
                }
            });
    }
    ~SamplerPool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        work_.notify_all();
        for (auto &t : threads_)
            t.join();
    }
    void run(const std::vector<std::function<void()>> &jobs) {
        if (jobs.size() <= 1 || threads_.empty()) {
            for (const auto &j : jobs)
                j();
            return;
        }
        std::unique_lock lock(mutex_);
        jobs_ = &jobs;
        next_ = finished_ = 0;
        error_ = nullptr;
        ++generation_;
        work_.notify_all();
        drain(lock);
        done_.wait(lock, [&] { return finished_ == jobs.size(); });
        jobs_ = nullptr;
        if (error_)
            std::rethrow_exception(error_);
    }
};
namespace {
const bool prefix_debug = [] {
    const char *v = std::getenv("BEHEMOTH_PREFIX_DEBUG");
    return v && std::atoi(v) != 0;
}();
RuntimeOptions with_transfers(RuntimeOptions o) {
    if (o.prefix_cache.slots && o.kv_transfer_pages <= 0)
        o.kv_transfer_pages = 32;
    return o;
}
} // namespace
struct Engine::Impl {
    SamplerPool samplers{std::min(7U, std::max(1U, std::thread::hardware_concurrency() / 4))};
    TensorParallelModel model;
    Scheduler scheduler;
    mutable std::mutex mutex;
    std::condition_variable cv;
    bool stop = false, failed = false;
    std::map<uint64_t, GenerationCallback> callbacks;
    std::map<uint64_t, std::mt19937_64> rng;
    std::set<uint64_t> cancellations;
    uint64_t completed = 0, tokens = 0;
    double gpu_seconds = 0;
    nlohmann::json last_profile;
    using Clock = std::chrono::steady_clock;
    std::map<uint64_t, Clock::time_point> submitted, last_token;
    std::deque<double> ttft, itl;

    // Prefix cache (PREFIX_CACHE_REPORT.md). All state below is guarded by `mutex`; HIP copies
    // are issued by the engine thread outside the lock.
    std::unique_ptr<PrefixCache> cache;
    size_t restore_pace = 0; // pages per step while decoding (0: whole restore at once)
    std::map<uint64_t, PrefixCache::Lease> leases; // request -> matched prefix, until erased
    std::deque<uint64_t> awaiting;                 // hits waiting for slot readiness / pages
    struct SaveWork {
        PrefixCache::SavePlan plan;
        std::vector<int> gpu;
    };
    // A restore may be issued in pieces (--prefix-cache-restore-pace while requests decode); the
    // pieces share one progress record and the request finishes with the last piece.
    struct RestoreProgress {
        Clock::time_point first_issue{};
        std::array<double, 2> gpu_ms{};
        bool started = false;
    };
    struct RestoreWork {
        uint64_t request;
        size_t tokens;
        std::vector<int> host, gpu;
        size_t offset = 0;
        std::shared_ptr<RestoreProgress> progress = std::make_shared<RestoreProgress>();
    };
    std::vector<SaveWork> to_save;
    std::vector<RestoreWork> to_restore;
    struct Inflight {
        uint64_t ticket = 0;
        bool save = false;
        uint64_t request = 0;
        size_t tokens = 0, pages = 0;
        PrefixCache::SavePlan plan;
        std::vector<int> gpu;
        Clock::time_point issued;
        std::shared_ptr<RestoreProgress> progress; // restores
        bool last_piece = true;
    };
    std::vector<Inflight> inflight;
    size_t issuing = 0; // copies taken out of to_save/to_restore, being issued (metrics)
    std::set<uint64_t> restoring_inflight;
    struct CacheCounters {
        uint64_t gpu_hits = 0, host_hits = 0, partial_restores = 0, restored_tokens = 0,
                 saves = 0, saved_pages = 0, shared_pages = 0, save_failures = 0,
                 cancelled_saves = 0;
        double restore_wall_ms = 0, save_wall_ms = 0, save_issue_ms = 0, issue_max_ms = 0,
               alloc_ms = 0;
        // *_ms: GPU time of the copy (slower rank); *_wall_ms: issue -> completion seen by the
        // engine loop (polled at step boundaries, so it includes waiting for a running step).
        std::deque<double> restore_ms, restore_gbps0, restore_gbps1, save_ms, save_gbps0,
            save_gbps1, restore_wall, save_wall;
        double last_restore_ms = 0, last_save_ms = 0, last_restore_wall = 0, last_save_wall = 0;
    } cc;

    explicit Impl(const RuntimeOptions &o)
        : model(with_transfers(o)),
          scheduler(model.cache_pages(), o.token_budget, o.prefill_chunk, o.scheduler) {
        if (o.prefix_cache.slots) {
            auto co = o.prefix_cache;
            const double cap = std::max(0.0, o.prefix_cache_gb) * 1e9;
            co.host_pages = size_t(cap / double(2 * model.kv_page_bytes()));
            if (!co.host_pages)
                throw std::invalid_argument("Prefix cache memory cap below one page");
            cache = std::make_unique<PrefixCache>(co);
            restore_pace = o.prefix_cache_restore_pace;
            scheduler.set_retain_finished(true);
            scheduler.set_reclaim([this](size_t need) {
                std::vector<int> pages;
                cache->reclaim_gpu(need, pages);
                return pages;
            });
            // The whole pinned arena is allocated now, while the GPUs are idle: mapping pinned
            // memory during serving triggered MES INVALIDATE_TLBS failures here.
            const auto t0 = Clock::now();
            model.kv_host_reserve(co.host_pages);
            cc.alloc_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::cerr << "Prefix cache: pinned " << double(model.kv_host_bytes()) / 1e9
                      << " GB host arena in " << cc.alloc_ms / 1e3 << " s\n";
            std::cerr << "Prefix cache: slots=" << co.slots << " cap=" << o.prefix_cache_gb
                      << " GB (" << co.host_pages << " host pages, "
                      << double(2 * model.kv_page_bytes()) / 1e6 << " MB each for both ranks)"
                      << " min_tokens=" << co.min_tokens << " gpu_tier=" << co.gpu_tier << '\n';
        }
        worker = std::thread([this] { loop(); });
    }
    std::thread worker;
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
    static void push_sample(std::deque<double> &d, double v) {
        d.push_back(v);
        if (d.size() > 1024)
            d.pop_front();
    }
    // Lock held. A done request leaves: its sequence (prompt + generated tokens with KV, also
    // for cancelled requests) is offered to the prefix cache, then it is erased.
    void retire(uint64_t id) {
        auto lease = leases.find(id);
        if (cache) {
            const auto &r = scheduler.request(id);
            auto pages = scheduler.take_pages(id);
            std::vector<int> release;
            PrefixCache::SavePlan plan;
            if (!failed && r.cached && pages.size() >= ceil_div(r.cached, 16)) {
                const PrefixCache::Lease *restored =
                    lease != leases.end() && r.restored && r.restored == lease->second.tokens
                        ? &lease->second
                        : nullptr;
                plan = cache->insert(std::span<const int>(r.history.data(), r.cached),
                                     r.prompt_size, restored, release);
                if (plan.slot && r.reason == "cancelled")
                    ++cc.cancelled_saves;
            }
            if (prefix_debug)
                std::cerr << "[prefix] retire " << id << " reason=" << r.reason
                          << " cached=" << r.cached << " prompt=" << r.prompt_size
                          << " restored=" << r.restored << " pages=" << pages.size()
                          << " plan=" << (plan.slot ? int64_t(plan.slot->id) : -1)
                          << " first=" << plan.first_page
                          << " slot_pages=" << (plan.slot ? plan.slot->host.size() : 0)
                          << " release_from_cache=" << release.size() << '\n';
            if (!plan.slot)
                release.insert(release.end(), pages.begin(), pages.end());
            else if (plan.first_page >= plan.slot->host.size()) {
                cc.shared_pages += plan.first_page;
                cache->saved(plan, std::move(pages), release);
            } else {
                cc.shared_pages += plan.first_page;
                to_save.push_back({std::move(plan), std::move(pages)});
            }
            scheduler.release_pages(release);
        }
        if (lease != leases.end()) {
            cache->release(lease->second);
            leases.erase(lease);
        }
        scheduler.erase(id);
    }
    // Lock held: completed copies.
    void poll_transfers() {
        for (auto it = inflight.begin(); it != inflight.end();) {
            std::array<float, 2> ms{};
            if (!model.kv_poll(it->ticket, &ms)) {
                ++it;
                continue;
            }
            const double wall =
                std::chrono::duration<double, std::milli>(Clock::now() - it->issued).count();
            const double bytes = double(it->pages) * double(model.kv_page_bytes());
            auto gbps = [&](float m) { return m > 0 ? bytes / (double(m) * 1e6) : 0.; };
            if (it->save) {
                std::vector<int> release;
                const size_t gpu_n = it->gpu.size();
                cache->saved(it->plan, std::move(it->gpu), release);
                if (prefix_debug)
                    std::cerr << "[prefix] saved slot " << it->plan.slot->id << " gpu=" << gpu_n
                              << " kept=" << it->plan.slot->gpu.size()
                              << " released=" << release.size()
                              << " indexed=" << it->plan.slot->indexed << '\n';
                scheduler.release_pages(release);
                ++cc.saves;
                cc.saved_pages += it->pages;
                cc.save_wall_ms += wall;
                cc.last_save_ms = std::max(ms[0], ms[1]);
                cc.last_save_wall = wall;
                push_sample(cc.save_ms, cc.last_save_ms);
                push_sample(cc.save_wall, wall);
                push_sample(cc.save_gbps0, gbps(ms[0]));
                push_sample(cc.save_gbps1, gbps(ms[1]));
            } else {
                auto &p = *it->progress;
                p.gpu_ms[0] += double(ms[0]);
                p.gpu_ms[1] += double(ms[1]);
                if (it->last_piece) {
                    const double total_wall = std::chrono::duration<double, std::milli>(
                                                  Clock::now() - p.first_issue)
                                                  .count();
                    const double total_bytes = double(ceil_div(it->tokens, 16)) *
                                               double(model.kv_page_bytes());
                    auto rate = [&](double m) { return m > 0 ? total_bytes / (m * 1e6) : 0.; };
                    scheduler.finish_restore(it->request, it->tokens);
                    restoring_inflight.erase(it->request);
                    cc.restored_tokens += it->tokens;
                    cc.restore_wall_ms += total_wall;
                    cc.last_restore_ms = std::max(p.gpu_ms[0], p.gpu_ms[1]);
                    cc.last_restore_wall = total_wall;
                    push_sample(cc.restore_ms, cc.last_restore_ms);
                    push_sample(cc.restore_wall, total_wall);
                    push_sample(cc.restore_gbps0, rate(p.gpu_ms[0]));
                    push_sample(cc.restore_gbps1, rate(p.gpu_ms[1]));
                }
            }
            it = inflight.erase(it);
        }
    }
    // Lock held: start restores whose slot is ready, preferring the GPU tier (no copy).
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
            if (prefix_debug)
                std::cerr << "[prefix] restore " << id << " slot=" << slot->id << " m=" << m
                          << " gpu_tier=" << gpu.size() << " released=" << release.size() << '\n';
            if (!gpu.empty()) {
                scheduler.begin_restore(id, std::move(gpu));
                scheduler.finish_restore(id, m);
                ++cc.gpu_hits;
                cc.restored_tokens += m;
                it = awaiting.erase(it);
                continue;
            }
            std::vector<int> table;
            if (!scheduler.allocate_pages(table, m)) {
                // Not enough free pages for the whole prefix: restore what fits (whole pages).
                const size_t fit = std::min(m, scheduler.free_pages() * 16);
                if (fit < cache->options().min_tokens || !scheduler.allocate_pages(table, fit)) {
                    ++it;
                    continue;
                }
                m = lease.tokens = fit;
                ++cc.partial_restores;
            }
            ++cc.host_hits;
            scheduler.begin_restore(id, table);
            to_restore.push_back({id, m,
                                  std::vector<int>(lease.host.begin(),
                                                   lease.host.begin() +
                                                       std::ptrdiff_t(ceil_div(m, 16))),
                                  std::move(table)});
            restoring_inflight.insert(id);
            it = awaiting.erase(it);
        }
    }
    // Engine thread, lock NOT held: issue the copies collected under the lock.
    void issue_transfers(std::vector<SaveWork> &saves, std::vector<RestoreWork> &restores,
                         size_t arena_pages, bool decoding, std::vector<Inflight> &issued,
                         std::vector<RestoreWork> &remaining, std::vector<SaveWork> &failed_saves,
                         std::vector<RestoreWork> &failed_restores) {
        if (saves.empty() && restores.empty())
            return;
        // the arena (reserved at startup) backs every host page the cache hands out
        for (auto it = saves.begin(); it != saves.end();) {
            const auto &h = it->plan.slot->host;
            const int top = *std::max_element(h.begin() + std::ptrdiff_t(it->plan.first_page), h.end());
            if (size_t(top) < arena_pages)
                ++it;
            else {
                failed_saves.push_back(std::move(*it));
                it = saves.erase(it);
            }
        }
        const auto started = Clock::now();
        for (auto &r : restores)
            try {
                // Host -> GPU copies compete with the decode steps' zero-copy reads of host memory
                // on the same link; with requests decoding, at most restore_pace pages per step.
                const size_t left = r.gpu.size() - r.offset;
                const size_t n = decoding && restore_pace ? std::min(left, restore_pace) : left;
                Inflight f;
                f.issued = Clock::now();
                if (!r.progress->started) {
                    r.progress->started = true;
                    r.progress->first_issue = f.issued;
                }
                f.ticket = model.kv_restore(
                    std::span<const int>(r.host.data() + r.offset, n),
                    std::span<const int>(r.gpu.data() + r.offset, n));
                cc_issue_max = std::max(cc_issue_max, std::chrono::duration<double, std::milli>(
                                                          Clock::now() - f.issued)
                                                          .count());
                f.request = r.request;
                f.tokens = r.tokens;
                f.pages = n;
                f.progress = r.progress;
                f.last_piece = n == left;
                issued.push_back(std::move(f));
                if (n < left) {
                    r.offset += n;
                    remaining.push_back(std::move(r));
                }
            } catch (const std::exception &e) {
                std::cerr << "Prefix cache: restore not issued: " << e.what() << '\n';
                failed_restores.push_back(std::move(r));
            }
        for (auto &s : saves)
            try {
                const size_t first = s.plan.first_page, n = s.plan.slot->host.size();
                Inflight f;
                f.save = true;
                f.issued = Clock::now();
                f.ticket = model.kv_save(
                    std::span<const int>(s.gpu.data() + first, n - first),
                    std::span<const int>(s.plan.slot->host.data() + first, n - first));
                cc_issue_max = std::max(cc_issue_max, std::chrono::duration<double, std::milli>(
                                                          Clock::now() - f.issued)
                                                          .count());
                f.pages = n - first;
                f.plan = std::move(s.plan);
                f.gpu = std::move(s.gpu);
                issued.push_back(std::move(f));
            } catch (const std::exception &e) {
                std::cerr << "Prefix cache: save not issued: " << e.what() << '\n';
                failed_saves.push_back(std::move(s));
            }
        cc_issue_ms += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    }
    double cc_issue_ms = 0, cc_issue_max = 0; // engine thread only; copied to cc under the lock
    void loop() {
        for (;;) {
            std::vector<std::pair<GenerationCallback, GenerationEvent>> events;
            ScheduledBatch step;
            InferenceBatch input;
            std::vector<SaveWork> saves;
            std::vector<RestoreWork> restores;
            size_t arena_pages = 0;
            bool decoding = false;
            bool waiting_transfers = false;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] {
                    return stop || !callbacks.empty() || !cancellations.empty() ||
                           !inflight.empty() || !to_save.empty() || !to_restore.empty();
                });
                if (stop) {
                    for (const auto &[id, cb] : callbacks)
                        events.push_back({cb, {id, -1, true, "cancelled", "Engine shutting down"}});
                    auto pending = inflight;
                    lock.unlock();
                    for (auto &[cb, e] : events)
                        try {
                            cb(e);
                        } catch (...) {
                        }
                    for (auto &f : pending)
                        try {
                            model.kv_wait(f.ticket);
                        } catch (...) {
                        }
                    return;
                }
                if (cache)
                    poll_transfers();
                for (auto it = cancellations.begin(); it != cancellations.end();) {
                    const uint64_t id = *it;
                    if (callbacks.contains(id) && !scheduler.request(id).done) {
                        if (scheduler.request(id).restoring) {
                            if (restoring_inflight.contains(id)) {
                                ++it; // copy in flight: cancel once it lands
                                continue;
                            }
                            awaiting.erase(std::find(awaiting.begin(), awaiting.end(), id));
                            scheduler.abort_restore(id);
                        }
                        scheduler.cancel(id);
                    }
                    it = cancellations.erase(it);
                }
                for (auto it = callbacks.begin(); it != callbacks.end();) {
                    const auto &r = scheduler.request(it->first);
                    if (r.done) {
                        events.push_back(
                            {it->second,
                             {r.id, -1, true, r.reason, "", r.prompt_size, r.generated,
                              r.restored}});
                        const uint64_t id = r.id;
                        retire(id);
                        rng.erase(id);
                        submitted.erase(id);
                        last_token.erase(id);
                        it = callbacks.erase(it);
                    } else
                        ++it;
                }
                if (cache)
                    start_restores();
                step = scheduler.schedule();
                if (!step.requests.empty())
                    input = make_inference_batch(scheduler, step);
                for (const auto &s : step.requests)
                    decoding |= s.count == 1 && s.sample && s.begin > 0;
                saves.swap(to_save);
                restores.swap(to_restore);
                issuing = saves.size() + restores.size();
                if (cache)
                    arena_pages = model.kv_host_pages();
                waiting_transfers = !inflight.empty() || !saves.empty() || !restores.empty();
            }
            for (auto &[cb, e] : events)
                try {
                    cb(e);
                } catch (...) {
                }
            events.clear();
            if (cache) {
                std::vector<Inflight> issued;
                std::vector<SaveWork> failed_saves;
                std::vector<RestoreWork> remaining, failed_restores;
                issue_transfers(saves, restores, arena_pages, decoding, issued, remaining,
                                failed_saves, failed_restores);
                std::lock_guard lock(mutex);
                for (auto &f : issued)
                    inflight.push_back(std::move(f));
                for (auto &r : remaining)
                    to_restore.push_back(std::move(r));
                issuing = 0;
                for (auto &s : failed_saves) {
                    cache->abandon(s.plan);
                    scheduler.release_pages(s.gpu);
                    ++cc.save_failures;
                }
                for (auto &r : failed_restores) {
                    // earlier pieces of a paced restore may still be copying into its pages
                    for (auto it = inflight.begin(); it != inflight.end();)
                        if (!it->save && it->request == r.request) {
                            model.kv_wait(it->ticket);
                            model.kv_poll(it->ticket);
                            it = inflight.erase(it);
                        } else
                            ++it;
                    restoring_inflight.erase(r.request);
                    scheduler.abort_restore(r.request);
                }
                cc.save_issue_ms = cc_issue_ms;
                cc.issue_max_ms = cc_issue_max;
            }
            if (step.requests.empty()) {
                // Nothing runnable: copies in flight (or requests waiting for pages). Sleep
                // briefly instead of spinning; submissions and cancellations wake the loop.
                std::unique_lock lock(mutex);
                if (waiting_transfers || !callbacks.empty())
                    cv.wait_for(lock, std::chrono::milliseconds(1));
                continue;
            }
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
                    // Logits are rank-blocked, one row per *sampled* request in step order
                    // (make_inference_batch); sample without an interleave copy. Requests own
                    // their RNGs, so sampling them in parallel gives the same tokens.
                    std::vector<int> sampled(step.requests.size(), -1);
                    {
                        std::vector<std::function<void()>> jobs;
                        size_t row = 0;
                        for (size_t i = 0; i < step.requests.size(); ++i) {
                            const auto &request = step.requests[i];
                            if (!request.sample)
                                continue;
                            const auto *sampling = &scheduler.request(request.id).sampling;
                            auto *generator = &rng.at(request.id);
                            const float *low = logits.data() + row * 16384,
                                        *high = logits.data() + sample_count * 16384 + row * 16384;
                            ++row;
                            jobs.push_back([&sampled, i, sampling, generator, low, high] {
                                sampled[i] = sample_logits(std::span<const float>(low, 16384),
                                                           std::span<const float>(high, 16384),
                                                           *sampling, *generator);
                            });
                        }
                        samplers.run(jobs);
                    }
                    for (size_t i = 0; i < step.requests.size(); ++i) {
                        const auto &request = step.requests[i];
                        const int token = sampled[i];
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
                                               after.generated, after.restored}});
                        }
                        if (after.done) {
                            ++completed;
                            events.push_back({callbacks.at(request.id),
                                              {request.id, -1, true, after.reason, "",
                                               after.prompt_size, after.generated,
                                               after.restored}});
                        }
                    }
                    for (const auto &request : step.requests)
                        if (scheduler.request(request.id).done) {
                            retire(request.id);
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
    if (impl_->cache)
        if (auto lease = impl_->cache->lookup(prompt)) {
            impl_->scheduler.wait_restore(id);
            impl_->leases.emplace(id, std::move(*lease));
            impl_->awaiting.push_back(id);
        }
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
    nlohmann::json m = {{"active_requests", impl_->scheduler.pending()},
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
    if (const auto &c = impl_->cache) {
        const auto &s = c->stats();
        const auto &cc = impl_->cc;
        const double page = double(impl_->model.kv_page_bytes()) * 2;
        m["prefix_cache"] = {
            {"enabled", true},
            {"slots_max", c->options().slots},
            {"slots_used", c->slots()},
            {"cached_tokens", c->cached_tokens()},
            {"host_bytes_used", double(c->host_pages_used()) * page},
            {"host_bytes_cap", double(c->options().host_pages) * page},
            {"host_bytes_pinned", impl_->model.kv_host_bytes()},
            {"gpu_tier_pages", c->gpu_pages()},
            {"hits", s.hits},
            {"misses", s.misses},
            {"gpu_tier_hits", cc.gpu_hits},
            {"host_restores", cc.host_hits},
            {"partial_restores", cc.partial_restores},
            {"tokens_reused", cc.restored_tokens},
            {"restore_wall_ms_total", cc.restore_wall_ms},
            {"restore_wall_ms_last", cc.last_restore_wall},
            {"restore_wall_ms_p50", percentile(cc.restore_wall, .5)},
            {"restore_ms_last", cc.last_restore_ms},
            {"restore_ms_p50", percentile(cc.restore_ms, .5)},
            {"restore_gbps_rank0_p50", percentile(cc.restore_gbps0, .5)},
            {"restore_gbps_rank1_p50", percentile(cc.restore_gbps1, .5)},
            {"restore_gbps_rank0_last", cc.restore_gbps0.empty() ? 0. : cc.restore_gbps0.back()},
            {"restore_gbps_rank1_last", cc.restore_gbps1.empty() ? 0. : cc.restore_gbps1.back()},
            {"restores_in_flight", impl_->restoring_inflight.size()},
            {"awaiting_restore", impl_->awaiting.size()},
            {"saves", cc.saves},
            {"saves_of_cancelled", cc.cancelled_saves},
            {"saved_pages", cc.saved_pages},
            {"shared_pages", cc.shared_pages},
            {"save_wall_ms_total", cc.save_wall_ms},
            {"save_wall_ms_last", cc.last_save_wall},
            {"save_wall_ms_p50", percentile(cc.save_wall, .5)},
            {"save_ms_last", cc.last_save_ms},
            {"save_ms_p50", percentile(cc.save_ms, .5)},
            {"save_gbps_rank0_p50", percentile(cc.save_gbps0, .5)},
            {"save_gbps_rank1_p50", percentile(cc.save_gbps1, .5)},
            {"save_gbps_rank0_last", cc.save_gbps0.empty() ? 0. : cc.save_gbps0.back()},
            {"save_gbps_rank1_last", cc.save_gbps1.empty() ? 0. : cc.save_gbps1.back()},
            {"issue_ms_total", cc.save_issue_ms},
            {"issue_ms_max", cc.issue_max_ms}, // engine-thread time to enqueue one transfer
            {"host_alloc_ms_startup", cc.alloc_ms},
            {"save_failures", cc.save_failures},
            {"saves_skipped", s.skipped},
            {"inserts", s.inserts},
            {"already_covered", s.covered},
            {"evictions", s.evictions},
            {"gpu_tier_reclaimed_pages", s.gpu_reclaimed_pages},
            {"transfers_in_flight", impl_->inflight.size()},
            // pool pages not free, not held by a request, a save or the GPU tier (must be 0)
            {"kv_pages_unaccounted",
             [&] {
                 size_t held = impl_->scheduler.free_pages() + impl_->scheduler.held_pages() +
                               c->gpu_pages();
                 for (const auto &f : impl_->inflight)
                     held += f.gpu.size();
                 for (const auto &s : impl_->to_save)
                     held += s.gpu.size();
                 return int64_t(impl_->scheduler.capacity()) - int64_t(held);
             }()},
            {"saves_pending", impl_->to_save.size() + impl_->to_restore.size() + impl_->issuing}};
    } else
        m["prefix_cache"] = {{"enabled", false}};
    return m;
}
} // namespace behemoth
