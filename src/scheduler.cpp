#include "behemoth/scheduler.hpp"
#include <algorithm>

namespace behemoth {
Scheduler::Scheduler(int pages, size_t budget, size_t chunk)
    : pool_(pages), token_budget_(budget), chunk_(chunk) {
    if (!budget || !chunk || chunk > budget)
        throw std::invalid_argument("Invalid scheduler token budget/chunk");
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
void Scheduler::cancel(uint64_t id) {
    auto &r = requests_.at(id);
    if (r.inflight)
        throw std::logic_error("Cancel must be processed between GPU steps");
    pool_.release(r.pages);
    r.done = true;
    r.reason = "cancelled";
}
void Scheduler::erase(uint64_t id) {
    auto &r = requests_.at(id);
    if (!r.done || r.inflight)
        throw std::logic_error("Cannot erase active request");
    pool_.release(r.pages);
    requests_.erase(id);
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
        if (!r.done) {
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
    auto append = [&](uint64_t id, bool is_decode) {
        auto &r = requests_.at(id);
        if (r.done || r.inflight || batch.tokens == token_budget_)
            return;
        size_t count = std::min({r.history.size() - r.cached, token_budget_ - batch.tokens,
                                 is_decode ? size_t(1) : chunk_});
        if (!count)
            return;
        // Queued prefills cannot evict older work. Decode growth may reclaim newest unpinned
        // requests.
        while (!pool_.grow(r.pages, r.cached + count)) {
            if (!is_decode)
                return;
            auto victim = requests_.rend();
            for (auto it = requests_.rbegin(); it != requests_.rend(); ++it)
                if (!it->second.done && !it->second.inflight && !it->second.pages.empty()) {
                    victim = it;
                    break;
                }
            if (victim == requests_.rend())
                return;
            pool_.release(victim->second.pages);
            victim->second.cached = 0;
            ++preemptions_;
            if (victim->first == id)
                return;
        }
        batch.requests.push_back({id, r.cached, count, r.cached + count == r.history.size()});
        batch.tokens += count;
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
        auto first = std::find_if(requests_.begin(), requests_.end(),
                                  [](const auto &p) { return !p.second.done; });
        for (auto it = requests_.rbegin(); it != requests_.rend(); ++it)
            if (it->first != first->first && !it->second.done && !it->second.pages.empty()) {
                pool_.release(it->second.pages);
                it->second.cached = 0;
                ++preemptions_;
            }
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
            pool_.release(r.pages);
        }
    }
}
} // namespace behemoth
