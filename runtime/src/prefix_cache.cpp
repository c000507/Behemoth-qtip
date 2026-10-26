#include "behemoth/prefix_cache.hpp"
#include "behemoth/attention.hpp"
#include <algorithm>
#include <stdexcept>

namespace behemoth {
namespace {
size_t pages_for(size_t tokens) {
    return (tokens + size_t(page_tokens) - 1) / size_t(page_tokens);
}
} // namespace
PrefixCache::PrefixCache(PrefixCacheOptions o) : options_(o) {}
void PrefixCache::ref(int page) {
    auto &r = refs_.at(size_t(page));
    if (r++ == 0)
        ++used_;
}
void PrefixCache::unref(int page) {
    auto &r = refs_.at(size_t(page));
    if (r <= 0)
        throw std::logic_error("Prefix cache host page released twice");
    if (--r == 0) {
        --used_;
        free_.push_back(page);
    }
}
int PrefixCache::allocate() {
    if (!free_.empty()) {
        int p = free_.back();
        free_.pop_back();
        return p;
    }
    if (high_water_ >= options_.host_pages)
        return -1;
    refs_.push_back(0);
    return int(high_water_++);
}
void PrefixCache::drop(const SlotPtr &s, std::vector<int> &release_gpu) {
    if (!s->indexed)
        return;
    s->indexed = false;
    index_.erase(std::find(index_.begin(), index_.end(), s));
    for (int p : s->host)
        unref(p);
    release_gpu.insert(release_gpu.end(), s->gpu.begin(), s->gpu.end());
    s->gpu.clear();
}
bool PrefixCache::evict_one(const Slot *keep, std::vector<int> &release_gpu) {
    SlotPtr victim;
    for (const auto &s : index_)
        if (s.get() != keep && (!victim || s->last_use < victim->last_use))
            victim = s;
    if (!victim)
        return false;
    drop(victim, release_gpu);
    ++stats_.evictions;
    return true;
}
std::optional<PrefixCache::Lease> PrefixCache::lookup(std::span<const int> prompt) {
    if (!options_.slots || prompt.size() < 2)
        return std::nullopt;
    SlotPtr best;
    size_t best_len = 0;
    for (const auto &s : index_) {
        if (s->failed)
            continue;
        const size_t n = std::min(s->tokens.size(), prompt.size());
        const size_t l = size_t(
            std::mismatch(prompt.begin(), prompt.begin() + std::ptrdiff_t(n), s->tokens.begin())
                .first -
            prompt.begin());
        if (l > best_len || (l == best_len && best && l &&
                             (s->ready > best->ready ||
                              (s->ready == best->ready && s->last_use > best->last_use)))) {
            best = s;
            best_len = l;
        }
    }
    const size_t m = std::min(best_len, prompt.size() - 1);
    if (!best || !m || m < options_.min_tokens) {
        ++stats_.misses;
        return std::nullopt;
    }
    Lease lease{best, m, {}};
    lease.host.assign(best->host.begin(), best->host.begin() + std::ptrdiff_t(pages_for(m)));
    for (int p : lease.host)
        ref(p);
    best->last_use = ++clock_;
    ++stats_.hits;
    return lease;
}
void PrefixCache::release(Lease &lease) {
    for (int p : lease.host)
        unref(p);
    lease.host.clear();
    lease.slot.reset();
    lease.tokens = 0;
}
PrefixCache::SavePlan PrefixCache::insert(std::span<const int> tokens, size_t prompt_tokens,
                                          const Lease *restored, std::vector<int> &release_gpu) {
    if (!options_.slots || tokens.empty() || tokens.size() < options_.min_tokens)
        return {};
    for (const auto &s : index_)
        if (!s->failed && s->tokens.size() >= tokens.size() &&
            std::equal(tokens.begin(), tokens.end(), s->tokens.begin())) {
            s->last_use = ++clock_;
            ++stats_.covered;
            return {};
        }
    const size_t pages = pages_for(tokens.size());
    size_t shared = 0;
    if (restored && restored->slot && restored->tokens <= tokens.size() &&
        restored->tokens <= restored->slot->tokens.size() &&
        std::equal(tokens.begin(), tokens.begin() + std::ptrdiff_t(restored->tokens),
                   restored->slot->tokens.begin()))
        shared = std::min({restored->tokens / size_t(page_tokens), restored->host.size(), pages});
    if (pages - shared > options_.host_pages) {
        ++stats_.skipped;
        return {};
    }
    auto slot = std::make_shared<Slot>();
    slot->id = next_id_++;
    slot->tokens.assign(tokens.begin(), tokens.end());
    slot->prompt = std::min(prompt_tokens, tokens.size());
    if (shared)
        slot->host.assign(restored->host.begin(), restored->host.begin() + std::ptrdiff_t(shared));
    for (int p : slot->host)
        ref(p);
    // Indexed slots that are a prefix of the new sequence are superseded by it (any lookup
    // matching them matches the new slot at least as far), and so is the restored-from slot
    // when the restore covered its whole prompt (next turn of the same conversation; a match
    // that ends inside its prompt, e.g. a shared system prompt or an edited earlier message,
    // keeps it).
    const Slot *previous_turn =
        restored && restored->slot && restored->tokens >= restored->slot->prompt &&
                restored->slot->prompt
            ? restored->slot.get()
            : nullptr;
    for (auto it = index_.begin(); it != index_.end();) {
        auto s = *it;
        if (s.get() == previous_turn ||
            (s->tokens.size() <= tokens.size() &&
             std::equal(s->tokens.begin(), s->tokens.end(), tokens.begin()))) {
            drop(s, release_gpu);
            it = index_.begin();
        } else
            ++it;
    }
    while (index_.size() >= options_.slots)
        evict_one(nullptr, release_gpu);
    while (slot->host.size() < pages) {
        int p = allocate();
        while (p < 0 && evict_one(nullptr, release_gpu))
            p = allocate();
        if (p < 0) {
            for (int q : slot->host)
                unref(q);
            ++stats_.skipped;
            return {};
        }
        ref(p);
        slot->host.push_back(p);
    }
    for (size_t k = shared; k < pages; ++k)
        ref(slot->host[k]); // the save's reference, dropped by saved()/abandon()
    slot->ready = shared == pages;
    slot->last_use = ++clock_;
    index_.push_back(slot);
    ++stats_.inserts;
    return {slot, shared};
}
void PrefixCache::saved(const SavePlan &plan, std::vector<int> gpu_pages,
                        std::vector<int> &release_gpu) {
    auto &s = *plan.slot;
    for (size_t k = plan.first_page; k < s.host.size(); ++k)
        unref(s.host[k]);
    s.ready = true;
    if (options_.gpu_tier && s.indexed && s.gpu.empty() && gpu_pages.size() >= s.host.size()) {
        s.gpu.assign(gpu_pages.begin(), gpu_pages.begin() + std::ptrdiff_t(s.host.size()));
        release_gpu.insert(release_gpu.end(), gpu_pages.begin() + std::ptrdiff_t(s.host.size()),
                           gpu_pages.end());
    } else
        release_gpu.insert(release_gpu.end(), gpu_pages.begin(), gpu_pages.end());
}
void PrefixCache::abandon(const SavePlan &plan) {
    auto &s = *plan.slot;
    for (size_t k = plan.first_page; k < s.host.size(); ++k)
        unref(s.host[k]);
    s.failed = true;
    std::vector<int> release;
    drop(plan.slot, release);
}
std::vector<int> PrefixCache::take_gpu(const SlotPtr &s, size_t pages,
                                       std::vector<int> &release_gpu) {
    if (!s->indexed || !s->ready || !pages || s->gpu.size() < pages)
        return {};
    std::vector<int> taken(s->gpu.begin(), s->gpu.begin() + std::ptrdiff_t(pages));
    release_gpu.insert(release_gpu.end(), s->gpu.begin() + std::ptrdiff_t(pages), s->gpu.end());
    s->gpu.clear();
    return taken;
}
size_t PrefixCache::reclaim_gpu(size_t need, std::vector<int> &release_gpu) {
    std::vector<SlotPtr> resident;
    for (const auto &s : index_)
        if (!s->gpu.empty())
            resident.push_back(s);
    std::sort(resident.begin(), resident.end(),
              [](const SlotPtr &a, const SlotPtr &b) { return a->last_use < b->last_use; });
    size_t released = 0;
    for (const auto &s : resident) {
        if (released >= need)
            break;
        released += s->gpu.size();
        release_gpu.insert(release_gpu.end(), s->gpu.begin(), s->gpu.end());
        s->gpu.clear();
    }
    stats_.gpu_reclaimed_pages += released;
    return released;
}
size_t PrefixCache::gpu_pages() const {
    size_t n = 0;
    for (const auto &s : index_)
        n += s->gpu.size();
    return n;
}
size_t PrefixCache::cached_tokens() const {
    size_t n = 0;
    for (const auto &s : index_)
        n += s->tokens.size();
    return n;
}
} // namespace behemoth
