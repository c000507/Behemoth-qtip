#include "behemoth/scheduler.hpp"
#include <algorithm>

namespace behemoth {
Scheduler::Scheduler(int pages, size_t budget, size_t chunk, SchedulerPolicy policy)
    : pool_(pages), token_budget_(budget), chunk_(chunk), policy_(policy) {
    if (!budget || !chunk || chunk > budget)
        throw std::invalid_argument("Invalid scheduler token budget/chunk");
    if (!policy_.prefill_interval || !policy_.max_requests)
        throw std::invalid_argument("Invalid scheduler policy");
}
uint64_t Scheduler::submit(std::span<const int> prompt, size_t max_new, SamplingOptions sampling) {
    if (prompt.empty() || prompt.size() > 131072 || max_new > 131072 ||
        prompt.size() + max_new > 131072 || ceil_div(prompt.size(), 16) > size_t(pool_.capacity()))
        throw std::invalid_argument("Request exceeds context or standalone KV capacity");
    for (int t : prompt)
        if (t < 0 || t >= 32768)
            throw std::invalid_argument("Token outside vocabulary");
    Request r;
    r.id = next_id_++;
    r.history.assign(prompt.begin(), prompt.end());
    r.prompt_size = prompt.size();
    r.max_new = max_new;
    r.sampling = sampling;
    if (!max_new) {
        r.done = true;
        r.reason = "length";
    }
    uint64_t id = r.id;
    requests_.emplace(id, std::move(r));
    return id;
}
bool Scheduler::grow(std::vector<int> &table, size_t tokens) {
    if (pool_.grow(table, tokens))
        return true;
    if (!reclaim_)
        return false;
    const size_t need = ceil_div(tokens, size_t(16)) - table.size();
    auto pages = reclaim_(need > pool_.free_pages() ? need - pool_.free_pages() : 1);
    if (pages.empty())
        return false;
    pool_.release(pages);
    return pool_.grow(table, tokens);
}
std::vector<int> Scheduler::take_pages(uint64_t id) {
    auto &r = requests_.at(id);
    if (!r.done || r.inflight || r.restoring)
        throw std::logic_error("Pages of an active request cannot be taken");
    return std::move(r.pages);
}
void Scheduler::release_pages(std::vector<int> &pages) {
    pool_.release(pages);
}
bool Scheduler::allocate_pages(std::vector<int> &table, size_t tokens) {
    return grow(table, tokens);
}
void Scheduler::wait_restore(uint64_t id) {
    auto &r = requests_.at(id);
    if (r.done || r.inflight || r.cached || !r.pages.empty())
        throw std::logic_error("Only a queued request can wait for a prefix restore");
    r.restoring = true;
}
void Scheduler::begin_restore(uint64_t id, std::vector<int> pages) {
    auto &r = requests_.at(id);
    if (!r.restoring || !r.pages.empty())
        throw std::logic_error("Restore without wait_restore");
    r.pages = std::move(pages);
}
void Scheduler::finish_restore(uint64_t id, size_t tokens) {
    auto &r = requests_.at(id);
    if (!r.restoring || tokens >= r.history.size() || ceil_div(tokens, size_t(16)) > r.pages.size())
        throw std::logic_error("Invalid prefix restore completion");
    r.restoring = false;
    r.cached = r.restored = tokens;
}
void Scheduler::abort_restore(uint64_t id) {
    auto &r = requests_.at(id);
    if (!r.restoring)
        throw std::logic_error("No restore to abort");
    pool_.release(r.pages);
    r.restoring = false;
    r.cached = r.restored = 0;
}
void Scheduler::cancel(uint64_t id) {
    auto &r = requests_.at(id);
    if (r.inflight || r.restoring)
        throw std::logic_error("Cancel must be processed between GPU steps");
    if (!retain_finished_)
        pool_.release(r.pages);
    r.done = true;
    r.reason = "cancelled";
}
void Scheduler::erase(uint64_t id) {
    auto &r = requests_.at(id);
    if (!r.done || r.inflight || r.restoring)
        throw std::logic_error("Cannot erase active request");
    pool_.release(r.pages);
    requests_.erase(id);
}
size_t Scheduler::held_pages() const {
    size_t n = 0;
    for (const auto &[id, r] : requests_) {
        (void)id;
        n += r.pages.size();
    }
    return n;
}
size_t Scheduler::pending() const {
    size_t n = 0;
    for (const auto &[id, r] : requests_) {
        (void)id;
        n += size_t(!r.done);
    }
    return n;
}
ScheduledBatch Scheduler::schedule() {
    ScheduledBatch batch;
    std::vector<uint64_t> decode, prefill;
    for (auto &[id, r] : requests_)
        if (!r.done && !r.restoring) {
            if (r.inflight)
                throw std::logic_error("Previous GPU step not completed");
            if (r.history.size() > 131072 ||
                ceil_div(r.history.size(), 16) > size_t(pool_.capacity())) {
                pool_.release(r.pages);
                r.done = true;
                r.reason = "length";
                continue;
            }
            (r.cached > 0 && r.cached + 1 == r.history.size() ? decode : prefill).push_back(id);
        }
    if (!decode.empty())
        std::rotate(decode.begin(), decode.begin() + std::ptrdiff_t(cursor_ % decode.size()),
                    decode.end());
    ++cursor_;
    // Prefill allowance of this step (decode-first policy).
    size_t prefill_cap = token_budget_, prefill_tokens = 0;
    if (!decode.empty() && !prefill.empty()) {
        if (++steps_since_prefill_ < policy_.prefill_interval)
            prefill_cap = 0;
        else {
            steps_since_prefill_ = 0;
            if (policy_.mixed_prefill_tokens)
                prefill_cap = policy_.mixed_prefill_tokens;
        }
    } else
        steps_since_prefill_ = policy_.prefill_interval; // next mixed step may prefill at once
    auto append = [&](uint64_t id, bool is_decode) {
        auto &r = requests_.at(id);
        if (r.done || r.inflight || batch.tokens == token_budget_ ||
            batch.requests.size() >= policy_.max_requests)
            return;
        size_t count = std::min({r.history.size() - r.cached, token_budget_ - batch.tokens,
                                 is_decode ? size_t(1) : std::min(chunk_, prefill_cap - prefill_tokens)});
        if (!count)
            return;
        // Queued prefills cannot evict older work. Decode growth may reclaim newest unpinned
        // requests.
        while (!grow(r.pages, r.cached + count)) {
            if (!is_decode)
                return;
            auto victim = requests_.rend();
            for (auto it = requests_.rbegin(); it != requests_.rend(); ++it)
                if (!it->second.done && !it->second.inflight && !it->second.restoring &&
                    !it->second.pages.empty()) {
                    victim = it;
                    break;
                }
            if (victim == requests_.rend())
                return;
            pool_.release(victim->second.pages);
            victim->second.cached = victim->second.restored = 0;
            ++preemptions_;
            if (victim->first == id)
                return;
        }
        batch.requests.push_back({id, r.cached, count, r.cached + count == r.history.size()});
        batch.tokens += count;
        if (!is_decode)
            prefill_tokens += count;
        r.inflight = true;
    };
    for (auto id : decode) {
        auto &r = requests_.at(id);
        if (r.cached > 0 && r.cached + 1 == r.history.size())
            append(id, true);
    }
    for (auto id : prefill)
        append(id, false);
    // If incomplete prefills fill the pool, retain the oldest and recompute newer ones to guarantee
    // progress.
    if (batch.requests.empty() && pending()) {
        auto first = std::find_if(requests_.begin(), requests_.end(), [](const auto &p) {
            return !p.second.done && !p.second.restoring;
        });
        if (first == requests_.end())
            return batch; // only prefix restores pending
        for (auto it = requests_.rbegin(); it != requests_.rend(); ++it)
            if (it->first != first->first && !it->second.done && !it->second.restoring &&
                !it->second.pages.empty()) {
                pool_.release(it->second.pages);
                it->second.cached = it->second.restored = 0;
                ++preemptions_;
            }
        prefill_cap = token_budget_;
        append(first->first, first->second.cached > 0 &&
                                 first->second.cached + 1 == first->second.history.size());
    }
    return batch;
}
void Scheduler::complete(const ScheduledRequest &step, int token, bool eos) {
    auto &r = requests_.at(step.id);
    if (!r.inflight || r.cached != step.begin || !step.count ||
        step.count > r.history.size() - r.cached)
        throw std::logic_error("Invalid step completion");
    if (step.sample && (token < 0 || token >= 32768))
        throw std::invalid_argument("Invalid sampled token");
    r.cached += step.count;
    r.inflight = false;
    if (step.sample) {
        r.history.push_back(token);
        ++r.generated;
        if (eos || r.generated >= r.max_new || r.history.size() >= 131072) {
            r.done = true;
            r.reason = eos ? "stop" : "length";
            if (!retain_finished_)
                pool_.release(r.pages);
        }
    }
}
} // namespace behemoth
