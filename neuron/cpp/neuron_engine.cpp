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
#include <array>
#include <initializer_list>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
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
    struct RequestStats { // per request, for GenerationEvent (the server's request_metrics log)
        double queue_wait_ms = -1;
        size_t steps = 0, drafted = 0, accepted = 0;
    };
    std::map<uint64_t, RequestStats> request_stats;
    std::deque<double> ttft, itl;
    uint64_t completed = 0, tokens = 0, steps = 0, drafted_tokens = 0, accepted_drafts = 0;
    double model_seconds = 0;
    nlohmann::json last_profile;
    // prompt-lookup speculation: up to BEHEMOTH_NEURON_SPEC_TOKENS drafts per request (0 disables), only in steps of at
    // most BEHEMOTH_NEURON_SPEC_REQUESTS decode requests and BEHEMOTH_NEURON_SPEC_ROWS rows after drafting (the bucket a
    // verify step may grow to), matching the last BEHEMOTH_NEURON_SPEC_NGRAM tokens
    int spec_tokens = 0, spec_requests = 4, spec_rows = 8, spec_ngram = 3, max_rows = 1;
    // Adaptive: a verify step costs about spec_page_cost ms per draft row and table page of the bucket it lands in (the
    // paged K/V gather, measured 0.006-0.013 on the 8B with the per-row kernel), on top of a plain step (plain_step_ms, running
    // mean). A request drafts only while its running acceptance (accepted drafts per verify step, EMA, optimistic start) beats
    // twice that overhead measured in plain steps, and a request that stopped drafting retries as its estimate recovers (~10 steps).
    // Default 0.01 for graph sets WITHOUT shared-table verify buckets; 0.002 when the graph set has dec8s_* buckets (see the constructor;
    // docs/neuron-verify-kernel.md section 7:
    // measured +10.9 / +12.7 ms per verify step at 128 / 512 pages on Devstral, nearly flat in pages): 128 pages always try, 512 pages
    // draft while alpha >= 0.24, 1024 pages while alpha >= 0.47. Set BEHEMOTH_NEURON_SPEC_PAGE_COST=0.01 for a graph set WITHOUT dec8s
    // buckets (the per-row dec8 verify costs +47 ms per step at 512 pages and would then draft at a loss).
    double spec_page_cost = 0.01, plain_step_ms = 40;
    std::map<uint64_t, double> acceptance;
    Drafts drafts;
    // MTP speculation (recurrent Qwen3.5 backends with the checkpoint's draft head loaded; docs/neuron-qwen397-mtp.md):
    // BEHEMOTH_NEURON_SPEC_MTP = drafts per step (0: off). After every step that was ONE decode request the head drafts the
    // tokens after the one just sampled; the next step verifies them as one chunk of that request (scheduled token + drafts,
    // processed sequentially through the recurrence), the backend commits the state of the accepted prefix. Steps with
    // several requests do not draft. BEHEMOTH_NEURON_SPEC_MTP_ADAPT=1: the draft count per step follows the running
    // per-position acceptance and the measured step times (see mtp_choose).
    int spec_mtp = 0;
    bool mtp_adapt = false;
    std::map<uint64_t, std::pair<size_t, std::vector<int>>> mtp_drafts; // request -> (the begin they follow, drafts)
    std::vector<uint64_t> mtp_drafted, mtp_accepted;                    // by draft position: verified, accepted
    std::vector<double> mtp_p, mtp_cost;                                // EMA: P(drafts 1..j accepted); step ms with k drafts (incl. drafting)
    std::vector<uint64_t> mtp_steps_k;
    std::vector<double> mtp_verify, mtp_draftc;                         // EMA by k: step ms up to the sampling with k drafts; mtp_draft ms for k drafts
    std::vector<uint64_t> mtp_verify_n, mtp_draft_n;
    uint64_t mtp_step_count = 0;
    std::vector<int> mtp_by_rows;                                       // drafts per request for 2, 3, ... decode requests in a step
    double mtp_ms_total = 0;
    // structured outputs: constraints of the constrained requests; per-token cost of the constraint calls
    std::map<uint64_t, std::shared_ptr<TokenConstraint>> constraints;
    uint64_t constrained_requests = 0;
    std::atomic<uint64_t> constraint_ns{0}, constrained_tokens{0}, constraint_fallbacks{0};
    // prefix cache (null when disabled): leases of requests restoring a prefix, requests waiting for their restore
    std::unique_ptr<PrefixCache> cache;
    std::map<uint64_t, PrefixCache::Lease> leases;
    std::vector<uint64_t> awaiting;
    // host tier: host page h at h * model.kv_page_bytes(). Default-initialised (new[] without ()), NOT a vector: a vector's
    // resize zero-fills every byte, which made the whole --prefix-cache-gb cap resident at start (measured: +7.8 GB RSS for
    // 8 GB on the Ornith 2xlarge, docs/neuron-reap-prefix-cache.md); this way a page is backed when a save first writes it.
    std::unique_ptr<std::byte[]> arena;
    size_t arena_bytes = 0;
    uint64_t gpu_hits = 0, host_hits = 0, restored_tokens = 0, saves = 0, saved_pages = 0, failed_saves = 0;
    double save_ms = 0, restore_ms = 0;
    // Recurrent models with state snapshots (Qwen3.5; BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE=1, docs/neuron-qwen35-prefix-cache.md):
    // K/V pages alone do not make a prefix, the recurrent state after its last token is needed too, and that exists only
    // where it was copied out between two steps. A request collects snapshots of its sequence (position -> state): every
    // snap_interval tokens of prefill (at the chunk boundary reaching it), at prompt_size - snap_tail (the scheduler
    // splits the last chunk there: the next turn of the conversation usually matches up to a few tokens before the
    // generation prompt's end) and at its end (all tokens with K/V; next turn when the client sends the answer back
    // token-exact). They move to the cache slot of the sequence when the request retires; a lookup restores only at a
    // snapshot position (PrefixCache::lookup's `usable`), and the new request inherits the slot's snapshots below it.
    // Blobs are shared (never copied) and bounded by BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB (least recently used slots
    // lose their snapshots first; a slot without snapshots cannot be hit).
    struct StateBlob {
        std::vector<std::byte> data;
        size_t *total;
        StateBlob(size_t n, size_t *t) : data(n), total(t) { *total += n; }
        ~StateBlob() { *total -= data.size(); }
    };
    using Snapshot = std::shared_ptr<StateBlob>;
    using Snapshots = std::map<size_t, Snapshot>; // by position = tokens the state has absorbed
    bool state_cache = false;
    size_t snap_interval = 1024, snap_tail = 8, state_cap = 0, state_total = 0;
    bool snap_end = true;
    // BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN=1: snapshots only at multiples of the prefill chunk, the tail one at the last such
    // boundary at or before prompt_size - snap_tail, no extra chunk split, no end snapshot by default. A cold run is then
    // chunked exactly like the cache-off engine and a restored run continues on the same chunk grid, so cached output ==
    // cache-off output bit for bit (the gdn prefill's arithmetic depends on the chunk partition; docs/neuron-reap-prefix-cache.md).
    // Cost: a hit recomputes up to one chunk more than with the split tail.
    bool snap_align = false;
    size_t snap_chunk = 256;
    std::map<uint64_t, Snapshots> request_snaps;                                // by request id
    std::map<uint64_t, std::pair<PrefixCache::SlotPtr, Snapshots>> slot_snaps; // by cache slot id
    std::map<uint64_t, Snapshot> pending_state;                                // requests waiting for their restore
    uint64_t snapshots_taken = 0, snapshots_dropped = 0, state_restores = 0, snapshot_failures = 0;
    double snapshot_ms = 0, state_restore_ms = 0;
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
        // the shared-table verify buckets make a verify step nearly flat in pages: only then is the lower default right (per-row dec8
        // verify at 0.002 would draft at a loss on long contexts)
        spec_page_cost = model.has_shared_verify() ? 0.002 : 0.01;
        if (const char *v = std::getenv("BEHEMOTH_NEURON_SPEC_PAGE_COST"))
            spec_page_cost = std::atof(v);
        spec_mtp = std::max(0, std::min(env_int("BEHEMOTH_NEURON_SPEC_MTP", 0), model.mtp_max_drafts()));
        mtp_adapt = env_int("BEHEMOTH_NEURON_SPEC_MTP_ADAPT", 0) != 0;
        {
            const char *v = std::getenv("BEHEMOTH_NEURON_SPEC_MTP_BY_ROWS");
            std::stringstream list(v ? v : "");
            for (std::string item; std::getline(list, item, ',');)
                mtp_by_rows.push_back(std::atoi(item.c_str()));
        }
        mtp_drafted.assign(size_t(spec_mtp), 0);
        mtp_accepted.assign(size_t(spec_mtp), 0);
        mtp_p.assign(size_t(spec_mtp), 0.3);
        for (size_t j = 0; j < mtp_p.size() && j < 4; ++j)
            mtp_p[j] = std::array<double, 4>{0.89, 0.69, 0.50, 0.36}[j];
        mtp_verify.assign(size_t(spec_mtp) + 1, 0.0);
        mtp_draftc.assign(size_t(spec_mtp) + 1, 0.0);
        mtp_verify_n.assign(size_t(spec_mtp) + 1, 0);
        mtp_draft_n.assign(size_t(spec_mtp) + 1, 0);
        mtp_cost.assign(size_t(spec_mtp) + 1, 0.0);
        mtp_steps_k.assign(size_t(spec_mtp) + 1, 0);
        if (env_int("BEHEMOTH_NEURON_SPEC_MTP", 0) > 0)
            std::cerr << "MTP speculation: " << (spec_mtp ? "up to " + std::to_string(spec_mtp) + " drafts per step" + (mtp_adapt ? " (adaptive)" : "")
                                                           : std::string("off (the model has no usable MTP head / verify buckets)")) << "\n";
        // recurrent family: only with state snapshots (backend support) and opted in
        state_cache = model.recurrent() && model.state_bytes() && model.kv_page_bytes() && env_int("BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE", 0);
        if (o.prefix_cache.slots && model.recurrent() && !state_cache)
            std::cerr << "Prefix cache: off (the model keeps recurrent state that restored K/V pages do not carry"
                      << (model.state_bytes() ? "; BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE=1 enables the cache with state snapshots)\n" : ")\n");
        if (o.prefix_cache.slots && env_int("BEHEMOTH_NEURON_PREFIX_CACHE", 1) && (!model.recurrent() || state_cache)) {
            auto co = o.prefix_cache;
            if (state_cache) {
                snap_interval = size_t(std::max(1, env_int("BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL", 1024)));
                snap_tail = size_t(std::max(0, env_int("BEHEMOTH_NEURON_QWEN35_PREFIX_TAIL", 8)));
                if (snap_tail == 1) // a one-token last piece would be scheduled as a decode step
                    snap_tail = 2;
                // the end-of-sequence snapshot needs the state of the last ACCEPTED token: with MTP speculation that is a
                // commit (rename) of the final verify step, a path not yet verified on a device, so it is opt-in there
                snap_align = env_int("BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN", 0) != 0;
                snap_chunk = std::max<size_t>(1, o.prefill_chunk);
                snap_end = env_int("BEHEMOTH_NEURON_QWEN35_PREFIX_END", spec_mtp > 0 || snap_align ? 0 : 1) != 0;
                const char *sg = std::getenv("BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB");
                state_cap = size_t((sg ? std::atof(sg) : 4.0) * 1e9);
                co.turn_slack = snap_tail + (snap_align ? snap_chunk : 0); // a restore this close to the slot's prompt end continues it
            }
            // host tier: --prefix-cache-gb, capped by BEHEMOTH_NEURON_PREFIX_HOST_GB (default 8: the instance's RAM also
            // holds the embeddings and the load-time staging)
            const char *cap_env = std::getenv("BEHEMOTH_NEURON_PREFIX_HOST_GB");
            const double cap = std::min(std::max(0.0, o.prefix_cache_gb), cap_env ? std::atof(cap_env) : 8.0) * 1e9;
            co.host_pages = size_t(cap / double(model.kv_page_bytes()));
            if (!co.host_pages)
                throw std::invalid_argument("Prefix cache memory cap below one page");
            cache = std::make_unique<PrefixCache>(co);
            arena_bytes = co.host_pages * model.kv_page_bytes();
            arena.reset(new std::byte[arena_bytes]); // untouched pages stay unbacked (large new[] = fresh anonymous mmap)
            scheduler.set_retain_finished(true);
            scheduler.set_reclaim([this](size_t need) {
                std::vector<int> pages;
                cache->reclaim_gpu(need, pages);
                return pages;
            });
            std::cerr << "Prefix cache: " << co.slots << " slots, host tier " << double(arena_bytes) / 1e9 << " GB ("
                      << co.host_pages << " pages of " << model.kv_page_bytes() / 1024 << " KiB), device tier "
                      << (co.gpu_tier ? "on" : "off") << "\n";
            if (state_cache)
                std::cerr << "Prefix cache: recurrent state snapshots of " << double(model.state_bytes()) / (1 << 20) << " MiB every " << snap_interval
                          << " prefill tokens, " << (snap_align ? "on the " + std::to_string(snap_chunk) + "-token chunk grid, the last boundary " : std::string())
                          << snap_tail << " tokens before the prompt's end" << (snap_end ? " and at the sequence's end" : "")
                          << ", cap " << double(state_cap) / 1e9 << " GB\n";
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
            // recurrent state: the request's snapshots (those its cached tokens cover) and the state it ends with
            Snapshots snaps;
            if (state_cache) {
                if (auto rs = request_snaps.find(id); rs != request_snaps.end()) {
                    snaps = std::move(rs->second);
                    request_snaps.erase(rs);
                }
                snaps.erase(snaps.upper_bound(r.cached), snaps.end());
            }
            // a recurrent sequence without any snapshot (and none to come at its end) can never be restored: caching it
            // would only evict slots that can (e.g. short prompts with long answers when PREFIX_ALIGN turns END off)
            const bool restorable_seq = !state_cache || snap_end || !snaps.empty();
            if (!failed && r.cached && pages.size() >= ceil_div(r.cached, 16) && restorable_seq) {
                const PrefixCache::Lease *restored =
                    lease != leases.end() && r.restored && r.restored == lease->second.tokens ? &lease->second : nullptr;
                plan = cache->insert(std::span<const int>(r.history.data(), r.cached), r.prompt_size, restored, release);
            }
            if (state_cache) {
                if (plan.slot && snap_end && !pages.empty() && !snaps.count(r.cached))
                    if (auto snap = take_snapshot(pages[0]))
                        snaps[r.cached] = std::move(snap);
                if (!pages.empty())
                    model.state_drop(pages[0]);
                pending_state.erase(id);
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
                                      std::span<const int>(plan.slot->host.data() + first, n - first), arena.get());
                        save_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                        ++saves;
                        saved_pages += n - first;
                    } catch (const std::exception &e) {
                        std::cerr << "Prefix cache: save failed: " << e.what() << "\n";
                        ok = false;
                        ++failed_saves;
                    }
                if (ok && state_cache && !snaps.empty())
                    slot_snaps[plan.slot->id] = {plan.slot, std::move(snaps)};
                if (ok)
                    cache->saved(plan, std::move(pages), release);
                else {
                    cache->abandon(plan);
                    release.insert(release.end(), pages.begin(), pages.end());
                }
            }
            scheduler.release_pages(release);
            if (state_cache) {
                for (auto it = slot_snaps.begin(); it != slot_snaps.end();) // evicted / superseded slots
                    it = it->second.first->indexed ? std::next(it) : slot_snaps.erase(it);
                trim_snapshots();
            }
        }
        if (lease != leases.end()) {
            cache->release(lease->second);
            leases.erase(lease);
        }
        scheduler.erase(id);
    }

    // Under the lock, between steps: a copy of the recurrent state the sequence on first page `seq` has now (null: over
    // the cap, or the copy failed: the position is then simply not restorable).
    Snapshot take_snapshot(int seq) {
        const size_t n = model.state_bytes();
        if (state_total + n > state_cap) {
            trim_snapshots(n);
            if (state_total + n > state_cap) {
                ++snapshots_dropped;
                return nullptr;
            }
        }
        try {
            const auto t0 = Clock::now();
            auto snap = std::make_shared<StateBlob>(n, &state_total);
            model.state_save(seq, snap->data.data());
            snapshot_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            ++snapshots_taken;
            return snap;
        } catch (const std::exception &e) {
            std::cerr << "Prefix cache: state snapshot failed: " << e.what() << "\n";
            ++snapshot_failures;
            return nullptr;
        }
    }
    // Keep the snapshot bytes (+ `room`) under the cap: the least recently used slots lose their snapshots (blobs shared
    // with a newer slot or a running request stay alive through them).
    void trim_snapshots(size_t room = 0) {
        while (state_total + room > state_cap) {
            auto victim = slot_snaps.end();
            for (auto it = slot_snaps.begin(); it != slot_snaps.end(); ++it)
                if (victim == slot_snaps.end() || it->second.first->last_use < victim->second.first->last_use)
                    victim = it;
            if (victim == slot_snaps.end())
                return;
            snapshots_dropped += victim->second.second.size();
            slot_snaps.erase(victim);
        }
    }
    // The longest prefix of `slot` not beyond `limit` tokens that has a state snapshot (0: none).
    size_t restorable(const PrefixCache::Slot &slot, size_t limit) const {
        auto it = slot_snaps.find(slot.id);
        if (it == slot_snaps.end())
            return 0;
        auto s = it->second.second.upper_bound(limit);
        return s == it->second.second.begin() ? 0 : std::prev(s)->first;
    }
    // After a step, under the lock: `request` ran a prefill chunk that ended at `end` tokens.
    void snapshot_prefill(const Request &r, size_t end) {
        if (r.pages.empty() || end < cache->options().min_tokens)
            return;
        auto &snaps = request_snaps[r.id];
        if (snaps.count(end))
            return;
        size_t last = r.restored;
        for (auto it = snaps.rbegin(); it != snaps.rend(); ++it)
            if (it->first <= end) {
                last = std::max(last, it->first);
                break;
            }
        if (snap_align && end % snap_chunk)
            return;
        const size_t target = r.prompt_size - std::min(r.prompt_size, snap_tail);
        const bool tail = snap_align ? end <= target && end + snap_chunk > target
                                     : snap_tail ? r.prefill_split && end == r.prefill_split : end == r.prompt_size;
        if (!tail && end - std::min(end, last) < snap_interval)
            return;
        if (auto snap = take_snapshot(r.pages[0]))
            snaps[end] = std::move(snap);
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
                pending_state.erase(id);
                request_snaps.erase(id);
                it = awaiting.erase(it);
                continue;
            }
            if (!slot->ready) {
                ++it;
                continue;
            }
            size_t m = lease.tokens;
            // recurrent state: written (to the sequence's first page) before the request can be scheduled; a failure
            // aborts the restore, the request then prefills from position 0
            auto restore_state = [&](int first_page) {
                if (!state_cache)
                    return;
                const auto t0 = Clock::now();
                model.state_restore(first_page, pending_state.at(id)->data.data());
                state_restore_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                ++state_restores;
                pending_state.erase(id);
            };
            std::vector<int> release;
            auto gpu = cache->take_gpu(slot, ceil_div(m, 16), release);
            scheduler.release_pages(release);
            if (!gpu.empty()) {
                const int first_page = gpu[0];
                scheduler.begin_restore(id, std::move(gpu));
                try {
                    restore_state(first_page);
                } catch (const std::exception &e) {
                    std::cerr << "Prefix cache: state restore failed: " << e.what() << "\n";
                    scheduler.abort_restore(id);
                    cache->release(lease);
                    leases.erase(id);
                    pending_state.erase(id);
                    request_snaps.erase(id);
                    it = awaiting.erase(it);
                    continue;
                }
                scheduler.finish_restore(id, m);
                ++gpu_hits;
                restored_tokens += m;
                it = awaiting.erase(it);
                continue;
            }
            std::vector<int> table;
            if (!scheduler.allocate_pages(table, m)) {
                size_t fit = std::min(m, scheduler.free_pages() * 16);
                if (state_cache) { // only a snapshot position can be restored: the deepest inherited one that fits
                    auto &snaps = request_snaps[id];
                    auto s = snaps.upper_bound(fit);
                    fit = s == snaps.begin() ? 0 : std::prev(s)->first;
                    if (fit)
                        pending_state[id] = snaps.at(fit);
                }
                if (fit < cache->options().min_tokens || !fit || !scheduler.allocate_pages(table, fit)) {
                    ++it;
                    continue;
                }
                m = lease.tokens = fit;
                if (state_cache)
                    request_snaps[id].erase(request_snaps[id].upper_bound(m), request_snaps[id].end());
            }
            scheduler.begin_restore(id, table);
            try {
                const auto t0 = Clock::now();
                model.kv_restore(std::span<const int>(lease.host.data(), ceil_div(m, 16)), std::span<const int>(table.data(), ceil_div(m, 16)), arena.get());
                restore_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                restore_state(table[0]);
                scheduler.finish_restore(id, m);
                ++host_hits;
                restored_tokens += m;
            } catch (const std::exception &e) {
                std::cerr << "Prefix cache: restore failed: " << e.what() << "\n";
                scheduler.abort_restore(id);
                cache->release(lease);
                leases.erase(id);
                pending_state.erase(id);
                request_snaps.erase(id);
            }
            it = awaiting.erase(it);
        }
    }

    // Under the lock: the event with the request's queue wait and speculation counters filled in.
    GenerationEvent timed(GenerationEvent e) const {
        if (auto it = request_stats.find(e.id); it != request_stats.end()) {
            e.queue_wait_ms = it->second.queue_wait_ms;
            e.spec_steps = it->second.steps;
            e.drafted = it->second.drafted;
            e.accepted = it->second.accepted;
        }
        return e;
    }

    void forget(uint64_t id) { // under the lock
        retire(id);
        request_snaps.erase(id);
        pending_state.erase(id);
        request_stats.erase(id);
        callbacks.erase(id);
        rng.erase(id);
        submitted.erase(id);
        last_token.erase(id);
        acceptance.erase(id);
        constraints.erase(id);
        mtp_drafts.erase(id);
    }
    // Drafts for the next step: fixed (spec_mtp), or (BEHEMOTH_NEURON_SPEC_MTP_ADAPT=1) the k maximising expected tokens per
    // millisecond in the steady state "every step verifies k drafts and drafts k more":
    //   (1 + sum_{j <= k} P(drafts 1..j accepted)) / (verify_ms(k) + draft_ms(k)).
    // verify_ms(k) = EMA of the step time up to the sampling with k drafts attached, draft_ms(k) = EMA of mtp_draft for k
    // drafts: the two are timed separately because the step that verifies k drafts drafts another count when k changes
    // (one EMA per k over the whole step, the first version, mixed them and settled on the widest k). A k never timed is
    // extrapolated from the nearest timed one (measured slopes: +6 ms per verify row, 1.65 ms per chained
    // draft). P starts from the offline study (0.89 0.69 0.50 0.36) and is only updated for positions that are drafted,
    // so every 32nd step tries one draft more than the best to keep the next position's estimate alive.
    // Cost estimate per k. The first launch of every bucket / (Q, D) chain graph carries one-time setup (100+ ms), and a k
    // that looks bad is only run by the exploration steps, so a seed outlier survives for hundreds of steps (plain EMA:
    // stuck at the widest k, 68.8 tok/s; EMA without the first sample: stuck at k = 1, 78.6; 70 % pull towards a faster
    // sample: 90 steps at k = 4 before settling, 80.6). Hence: the first sample of a k is dropped, a faster sample
    // replaces the estimate at once, a slower one moves it by 5 % (clamped to 1.5 x).
    static void mtp_ema(double &c, uint64_t &n, double v) {
        if (++n < 2)
            return;
        if (n == 2 || v < c)
            c = v;
        else
            c += 0.05 * (std::min(v, 1.5 * c) - c);
    }
    // a k not timed yet: the measured single-stream profile (section 9 / 11 of the doc: verify step ms and chain launch ms
    // by k), scaled by timed / profile of the nearest timed k (long contexts scale the verify side)
    static double mtp_profile(const std::vector<double> &timed, const std::vector<uint64_t> &n, int k, int first, std::initializer_list<double> base, double slope) {
        auto prof = [&](int j) { return j < int(base.size()) ? base.begin()[j] : base.begin()[base.size() - 1] + slope * (j - int(base.size()) + 1); };
        if (n[size_t(k)] >= 2)
            return timed[size_t(k)];
        int near = -1;
        for (int j = first; j < int(timed.size()); ++j)
            if (n[size_t(j)] >= 2 && (near < 0 || std::abs(j - k) < std::abs(near - k)))
                near = j;
        return near < 0 ? prof(k) : prof(k) * timed[size_t(near)] / prof(near);
    }
    double mtp_verify_ms(int k) const { return mtp_profile(mtp_verify, mtp_verify_n, k, 0, {15.8, 22.2, 25.0, 29.1, 39.8}, 11.0); }
    double mtp_draft_ms(int k) const { return k <= 0 ? 0.0 : mtp_profile(mtp_draftc, mtp_draft_n, k, 1, {1.0, 1.5, 2.1, 2.7, 6.3}, 1.65); }
    // Drafts per request in a step of n decode requests: spec_mtp alone; with company the measured table
    // BEHEMOTH_NEURON_SPEC_MTP_BY_ROWS ("k for 2 requests,k for 3,..."; beyond the list 0: a verify row costs what a batch
    // row costs, so speculation stops paying when the step is wide - docs/neuron-qwen397-mtp.md section 14)
    int mtp_k_rows(int n) const {
        if (n <= 1)
            return spec_mtp;
        return size_t(n - 2) < mtp_by_rows.size() ? std::min(spec_mtp, mtp_by_rows[size_t(n - 2)]) : 0;
    }
    int mtp_choose() {
        if (!mtp_adapt || spec_mtp < 1) // (spec_mtp = 1 adapts too: k in {0, 1}, i.e. drafting switches off while the
            return spec_mtp;            //  acceptance does not pay for the verify row; every 16th step probes k = 1)
        int best = 0;
        double best_rate = 0, tokens = 1;
        for (int k = 0; k <= spec_mtp; ++k) {
            if (k)
                tokens += mtp_p[size_t(k - 1)];
            const double rate = tokens / std::max(mtp_verify_ms(k) + mtp_draft_ms(k), 1e-3);
            if (rate > best_rate)
                best_rate = rate, best = k;
        }
        // every 16th step (8th at first) one draft more or fewer than the best, alternately: keeps the neighbours' timings and the next
        // position's acceptance current (a k that is never run is never re-timed)
        const uint64_t every = mtp_step_count < 96 ? 8 : 16; // (more often while the estimates are young)
        if (++mtp_step_count % every == 0)
            best = std::clamp(best + (mtp_step_count / every % 2 ? 1 : -1), 1, spec_mtp);
        return best;
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
                            pending_state.erase(*it);
                            request_snaps.erase(*it);
                        }
                        scheduler.cancel(*it);
                    }
                for (auto it = callbacks.begin(); it != callbacks.end();) {
                    const auto &r = scheduler.request(it->first);
                    if (!r.done) {
                        ++it;
                        continue;
                    }
                    events.push_back({it->second, timed({r.id, -1, true, r.reason, "", r.prompt_size, r.generated, r.restored})});
                    const uint64_t id = r.id;
                    ++it;
                    forget(id);
                }
                if (cache)
                    start_restores();
                step = scheduler.schedule();
                for (const auto &scheduled : step.requests)
                    if (auto sub = submitted.find(scheduled.id); sub != submitted.end())
                        if (request_stats.try_emplace(scheduled.id).second)
                            request_stats[scheduled.id].queue_wait_ms = std::chrono::duration<double, std::milli>(Clock::now() - sub->second).count();
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
                    if (spec_mtp > 0) { // the MTP head's drafts for exactly these positions; k by the step's request count (mtp_k_rows)
                        const size_t n = step.requests.size();
                        bool singles = true;
                        size_t top = 0;
                        for (const auto &s : step.requests) {
                            singles = singles && s.count == 1 && s.sample;
                            top = std::max(top, s.begin);
                        }
                        int kmax = singles ? mtp_k_rows(int(n)) : 0;
                        if (kmax > 0 && n > 1)
                            kmax = std::min(kmax, model.mtp_max_drafts(int(n), int((top + 1 + size_t(kmax) + 15) / 16)));
                        for (const auto &s : step.requests) {
                            auto it = mtp_drafts.find(s.id);
                            if (it == mtp_drafts.end())
                                continue;
                            auto d = std::move(it->second.second);
                            const bool follows = it->second.first == s.begin;
                            mtp_drafts.erase(it);
                            if (kmax <= 0 || !follows)
                                continue;
                            const auto &r = scheduler.request(s.id);
                            const size_t room = r.max_new > r.generated ? r.max_new - r.generated - 1 : 0; // tokens after this step's own
                            if (d.size() > std::min(room, size_t(kmax)))
                                d.resize(std::min(room, size_t(kmax)));
                            if (!d.empty() && s.begin + 1 + d.size() <= max_context && scheduler.reserve(s.id, s.begin + 1 + d.size())) {
                                // test hook: BEHEMOTH_NEURON_SPEC_MTP_CORRUPT=percent of the verify steps get one wrong draft
                                // (forces rejections at every draft position: the rollback must leave the stream unchanged)
                                static const int corrupt = env_int("BEHEMOTH_NEURON_SPEC_MTP_CORRUPT", 0);
                                static std::mt19937_64 crng(12345);
                                if (corrupt > 0 && int(crng() % 100) < corrupt) {
                                    const size_t at = crng() % d.size();
                                    d[at] = (d[at] + 1) % model.vocab();
                                }
                                drafts[s.id] = std::move(d);
                            }
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
                // Device argmax fast path: every sampled row must be unconstrained greedy (the drafts are verified by
                // greedy comparison, so they qualify); the head graph then returns per-row (max, argmax) and the host
                // only merges the two ranks' candidates.
                bool greedy_step = false; // default OFF (docs); BEHEMOTH_NEURON_DEVICE_ARGMAX=1 enables
                if (const char *da = std::getenv("BEHEMOTH_NEURON_DEVICE_ARGMAX"))
                    if (da[0] == '1')
                        greedy_step = true;
                for (size_t i = 0; i < step.requests.size(); ++i) {
                    if (!step.requests[i].sample)
                        continue;
                    if (step_constraints[i]) {
                        greedy_step = false;
                        break;
                    }
                    const auto &req0 = scheduler.request(step.requests[i].id);
                    if (has_logit_processing(req0.sampling) || req0.sampling.temperature != 0.f) {
                        greedy_step = false;
                        break;
                    }
                }
                const auto logits = model.forward(input, greedy_step);
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
                const auto t_sample0 = Clock::now(); // sample_ms = the sampling block only (excludes the forward)
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
                            if (greedy_step) {
                                const int dt = model.device_token(logits_row[i] + j);
                                if (dt >= 0) {
                                    sampled[i][j] = dt;
                                    continue;
                                }
                            }
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
                last_profile["sample_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - t_sample0).count();
                ++steps;
                size_t step_tokens = 0, step_drafts = 0, step_accepted = 0;
                std::vector<std::vector<int>> emitted_by(step.requests.size()); // per request: the tokens this step emitted, in order
                std::vector<size_t> req_drafts(step.requests.size(), 0), req_accepted(step.requests.size(), 0);
                for (size_t i = 0; i < step.requests.size(); ++i) {
                    const auto &request = step.requests[i];
                    auto emit = [&](int token) {
                        const auto &after = scheduler.request(request.id);
                        ++tokens;
                        ++step_tokens;
                        emitted_by[i].push_back(token);
                        const auto now = Clock::now();
                        const bool later = last_token.contains(request.id);
                        (later ? itl : ttft).push_back(std::chrono::duration<double, std::milli>(now - (later ? last_token.at(request.id) : submitted.at(request.id))).count());
                        for (auto *samples : {&ttft, &itl})
                            if (samples->size() > 1024)
                                samples->pop_front();
                        last_token[request.id] = now;
                        events.push_back({callbacks.at(request.id), timed({request.id, token, false, "", "", after.prompt_size, after.generated, after.restored})});
                    };
                    const int token = sampled[i].empty() ? -1 : sampled[i][0];
                    // debug: BEHEMOTH_NEURON_MARGIN_LOG=file gets, per emitted greedy token, "id index token runner-up gap" (the
                    // logit gap to the second choice: how close a near tie was when two configurations diverge)
                    auto margin = [&](size_t j) {
                        static const char *path = std::getenv("BEHEMOTH_NEURON_MARGIN_LOG");
                        if (!path || greedy_step)
                            return;
                        const auto row = logits.subspan((logits_row[i] + j) * vocab, vocab);
                        int a = 0, b = -1;
                        for (int v = 1; v < int(vocab); ++v) {
                            if (row[size_t(v)] > row[size_t(a)])
                                b = a, a = v;
                            else if (b < 0 || row[size_t(v)] > row[size_t(b)])
                                b = v;
                        }
                        static std::ofstream f(path, std::ios::app);
                        f << request.id << " " << scheduler.request(request.id).generated << " " << sampled[i][j] << " " << (sampled[i][j] == a ? b : a) << " "
                          << row[size_t(a)] - row[size_t(b)] << "\n";
                    };
                    if (token >= 0)
                        margin(0);
                    scheduler.complete(request, token, token >= 0 && eos.contains(token));
                    // a prefill chunk of a recurrent model: its state is now the state after begin + count tokens
                    if (state_cache && request.begin + request.count <= scheduler.request(request.id).prompt_size)
                        snapshot_prefill(scheduler.request(request.id), request.begin + request.count);
                    if (token >= 0)
                        emit(token);
                    // verified drafts: row j's token was sampled after draft j - 1; it stands when that draft matched
                    auto d = drafts.find(request.id);
                    if (d != drafts.end()) {
                        step_drafts += d->second.size();
                        size_t accepted = 0;
                        for (size_t j = 1; j < sampled[i].size() && !scheduler.request(request.id).done && sampled[i][j - 1] == d->second[j - 1]; ++j) {
                            margin(j);
                            scheduler.accept(request.id, sampled[i][j], eos.contains(sampled[i][j]));
                            emit(sampled[i][j]);
                            ++accepted;
                        }
                        accepted_drafts += accepted;
                        step_accepted += accepted;
                        req_drafts[i] = d->second.size();
                        req_accepted[i] = accepted;
                        if (auto a = acceptance.find(request.id); a != acceptance.end())
                            a->second = 0.5 * (a->second + double(accepted));
                    }
                    if (token >= 0)
                        if (auto stats = request_stats.find(request.id); stats != request_stats.end()) {
                            ++stats->second.steps;
                            stats->second.drafted += req_drafts[i];
                            stats->second.accepted += req_accepted[i];
                        }
                    const auto &after = scheduler.request(request.id);
                    if (after.done) {
                        ++completed;
                        events.push_back({callbacks.at(request.id), timed({request.id, -1, true, after.reason, "", after.prompt_size, after.generated, after.restored})});
                    }
                }
                drafted_tokens += step_drafts;
                bool mtp_step = spec_mtp > 0; // every request of the step is one decode token (+ its drafts): the head drafts for each
                for (const auto &request : step.requests)
                    mtp_step = mtp_step && request.count == 1 && request.sample;
                if (mtp_step) {
                    const size_t n = step.requests.size();
                    const double verify_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                    if (n == 1 && step_drafts < mtp_verify.size()) { // the verify side of the cost model: this step up to here
                        mtp_ema(mtp_verify[step_drafts], mtp_verify_n[step_drafts], verify_ms);
                    }
                    double draft_ms = 0;
                    for (size_t i = 0; i < n; ++i) {
                        const auto &request = step.requests[i];
                        const auto &emitted = emitted_by[i];
                        if (emitted.empty())
                            continue;
                        const auto &rq = scheduler.request(request.id);
                        for (size_t j = 0; j < req_drafts[i] && j < mtp_drafted.size(); ++j) {
                            ++mtp_drafted[j];
                            mtp_accepted[j] += j < req_accepted[i];
                            mtp_p[j] = 0.95 * mtp_p[j] + 0.05 * (j < req_accepted[i] ? 1.0 : 0.0);
                        }
                        if (rq.done) {
                            // the sequence's end state is snapshotted when it retires: commit the accepted prefix first
                            if (state_cache && snap_end && emitted.size() > 1 && !rq.pages.empty())
                                model.mtp_draft(rq.pages, int(request.begin), emitted, 0);
                            continue;
                        }
                        const size_t next_begin = request.begin + emitted.size();
                        const size_t room = rq.max_new > rq.generated ? rq.max_new - rq.generated - 1 : 0;
                        int want = int(std::min<size_t>(size_t(n == 1 ? mtp_choose() : mtp_k_rows(int(n))), room));
                        if (want > 0 && (next_begin + 1 + size_t(want) > max_context || !scheduler.reserve(request.id, next_begin + 1 + size_t(want))))
                            want = 0;
                        // always called: it also commits the recurrent state of the accepted prefix
                        const auto td = Clock::now();
                        auto d = model.mtp_draft(scheduler.request(request.id).pages, int(request.begin), emitted, want);
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                        draft_ms += ms;
                        if (n == 1 && !d.empty() && d.size() < mtp_draftc.size()) {
                            mtp_ema(mtp_draftc[d.size()], mtp_draft_n[d.size()], ms);
                        }
                        if (!d.empty())
                            mtp_drafts[request.id] = {next_begin, std::move(d)};
                    }
                    last_profile["mtp_ms"] = draft_ms;
                    mtp_ms_total += draft_ms;
                    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                    if (n == 1 && step_drafts < mtp_cost.size()) {
                        double &c = mtp_cost[step_drafts];
                        c = mtp_steps_k[step_drafts]++ ? 0.9 * c + 0.1 * wall : wall;
                    }
                }
                if (!step_drafts && step.requests.size() == 1 && step.requests[0].count == 1) // a plain single decode step
                    plain_step_ms = 0.9 * plain_step_ms + 0.1 * std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                if (std::getenv("BEHEMOTH_NEURON_STEP_LOG"))
                    std::cerr << "step rows=" << step.requests.size() << " drafts=" << step_drafts << " accepted=" << step_accepted << " tokens=" << step_tokens << " wall_ms="
                              << std::chrono::duration<double, std::milli>(Clock::now() - start).count() << " " << last_profile.dump() << "\n";
                for (const auto &request : step.requests)
                    if (scheduler.request(request.id).done)
                        forget(request.id);
            } catch (const std::exception &error) {
                // fatal: /health turns 503 and the orchestrator replaces the server, so say why
                std::cerr << "engine step failed, engine stopped: " << error.what() << std::endl;
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
    if (impl_->cache) {
        PrefixCache::Usable usable;
        if (impl_->state_cache) {
            usable = [&](const PrefixCache::Slot &slot, size_t limit) { return impl_->restorable(slot, limit); };
            if (impl_->snap_align)
                impl_->scheduler.set_prefill_align(id, impl_->snap_chunk);
            else if (impl_->snap_tail && prompt.size() >= impl_->snap_tail + impl_->cache->options().min_tokens)
                impl_->scheduler.set_prefill_split(id, prompt.size() - impl_->snap_tail);
        }
        if (auto lease = impl_->cache->lookup(prompt, usable)) {
            if (impl_->state_cache) { // the snapshot to restore, and the slot's snapshots below it (this sequence's too)
                const auto &snaps = impl_->slot_snaps.at(lease->slot->id).second;
                impl_->pending_state[id] = snaps.at(lease->tokens);
                impl_->request_snaps[id] = Impl::Snapshots(snaps.begin(), snaps.upper_bound(lease->tokens));
            }
            impl_->scheduler.wait_restore(id);
            impl_->leases.emplace(id, std::move(*lease));
            impl_->awaiting.push_back(id);
        }
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
            {"mtp", {{"max_drafts", impl_->spec_mtp}, {"adaptive", impl_->mtp_adapt}, {"verified_by_position", impl_->mtp_drafted},
                     {"accepted_by_position", impl_->mtp_accepted}, {"steps_by_drafts", impl_->mtp_steps_k}, {"step_ms_by_drafts", impl_->mtp_cost},
                     {"verify_ms_by_drafts", impl_->mtp_verify}, {"draft_ms_by_drafts", impl_->mtp_draftc}, {"p_by_position", impl_->mtp_p},
                     {"draft_ms_total", impl_->mtp_ms_total}}},
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
                                                           {"evictions", impl_->cache->stats().evictions},
                                                           {"state_snapshots", {{"enabled", impl_->state_cache}, {"taken", impl_->snapshots_taken}, {"dropped", impl_->snapshots_dropped},
                                                                                {"failed", impl_->snapshot_failures}, {"restores", impl_->state_restores},
                                                                                {"bytes", impl_->state_total}, {"slots", impl_->slot_snaps.size()},
                                                                                {"snapshot_ms", impl_->snapshot_ms}, {"restore_ms", impl_->state_restore_ms}}},
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
