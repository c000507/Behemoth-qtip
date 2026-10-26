#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <vector>

namespace behemoth {
// Prefix cache index (PREFIX_CACHE_REPORT.md). Pure bookkeeping, no GPU calls: the engine
// copies KV pages between the GPU page pool and host pages that this index hands out.
//
// A slot holds the token ids of one finished (or cancelled) sequence and one host page per
// 16-token KV page (a host page index stands for the page of *both* tensor-parallel ranks).
// Host pages are reference counted: slots, pending/active restores (leases) and in-flight
// saves each hold references, so evicting a slot never frees memory that a copy still reads
// or writes; a superseding slot (the next turn of the same conversation) shares the pages of
// the prefix it restored instead of copying them again.
//
// Optional GPU tier: after its save completes a slot may keep its GPU KV pages (scheduler pool
// page ids). A later hit on that slot takes the pages over (no copy); the scheduler reclaims
// GPU-tier pages, least recently used first, whenever it runs out of free pages.
//
// Eviction is LRU over slots (lookup hits and inserts refresh a slot), bounded by `slots`
// and by `host_pages` (the memory cap). Thread safety: callers serialise all calls (the
// engine holds its mutex).
struct PrefixCacheOptions {
    size_t slots = 10;      // 0 disables the cache
    size_t host_pages = 0;  // host page cap (one page = both ranks' 16-token page)
    size_t min_tokens = 64; // shortest sequence saved and shortest prefix restored
    bool gpu_tier = true;
};
class PrefixCache {
  public:
    struct Slot {
        uint64_t id = 0;
        std::vector<int> tokens; // the sequence whose KV the slot holds
        size_t prompt = 0;       // leading tokens that were the request's prompt (rest: generated)
        std::vector<int> host;   // host page per 16-token page: ceil(tokens / 16)
        std::vector<int> gpu;    // GPU tier: resident pool pages (empty or host.size())
        bool ready = false;      // all host pages hold their KV (save complete)
        bool failed = false;     // save abandoned: never becomes ready
        bool indexed = true;     // matchable (false once evicted or superseded)
        uint64_t last_use = 0;
    };
    using SlotPtr = std::shared_ptr<Slot>;
    // A matched prefix of `tokens` tokens; holds references on the slot's first
    // ceil(tokens / 16) host pages (`host`) until release().
    struct Lease {
        SlotPtr slot;
        size_t tokens = 0;
        std::vector<int> host;
    };
    // Result of insert(): pages [first_page, slot->host.size()) must be copied from the GPU
    // (pages below first_page are shared with the restored-from slot). Holds references on the
    // pages to copy until saved()/abandon().
    struct SavePlan {
        SlotPtr slot; // null: nothing to save
        size_t first_page = 0;
    };
    struct Stats {
        uint64_t hits = 0, misses = 0, inserts = 0, covered = 0, evictions = 0, skipped = 0,
                 gpu_reclaimed_pages = 0;
    };

    explicit PrefixCache(PrefixCacheOptions);
    const PrefixCacheOptions &options() const {
        return options_;
    }
    // Longest common prefix over all indexed slots (ready or still saving), capped at
    // prompt.size() - 1 so at least one prompt token is prefilled (its logits are sampled).
    // Prefixes shorter than min_tokens are misses.
    std::optional<Lease> lookup(std::span<const int> prompt);
    void release(Lease &);
    // A sequence finished; KV for all `tokens` is on the GPU, the first `prompt_tokens` of them
    // were its prompt. `restored` (may be null): the lease whose host pages were restored into
    // this sequence's first pages, so they are exact copies and can be shared. Does nothing when
    // an indexed slot already covers `tokens`. Supersedes (replaces) indexed slots that are a
    // prefix of `tokens`, and the restored-from slot when the match covered that slot's whole
    // prompt: the next turn of the same conversation (its previous reply is usually re-tokenized
    // differently or edited by the client, so only the slot's prompt part matched). Evicts LRU
    // slots to stay within the slot count and host page cap; may return an empty plan
    // (stats().skipped) when the pages still needed are all referenced by active copies.
    SavePlan insert(std::span<const int> tokens, size_t prompt_tokens, const Lease *restored,
                    std::vector<int> &release_gpu);
    // Save of plan.slot completed. gpu_pages: the sequence's GPU pages; kept as the slot's GPU
    // tier when enabled and the slot is still indexed, otherwise appended to release_gpu.
    void saved(const SavePlan &, std::vector<int> gpu_pages, std::vector<int> &release_gpu);
    // Save failed or cancelled: the slot is dropped (marked failed).
    void abandon(const SavePlan &);
    // GPU-tier hit: the slot's first `pages` resident pages, or empty when it has none. The
    // remaining resident pages go to release_gpu; the slot keeps only its host copy.
    std::vector<int> take_gpu(const SlotPtr &, size_t pages, std::vector<int> &release_gpu);
    // Drop GPU tiers (least recently used first) until at least `need` pages were released.
    size_t reclaim_gpu(size_t need, std::vector<int> &release_gpu);

    size_t slots() const {
        return index_.size();
    }
    size_t host_pages_used() const {
        return used_;
    }
    // Largest host page index handed out + 1 (the engine backs [0, high_water) with memory).
    size_t host_high_water() const {
        return high_water_;
    }
    size_t gpu_pages() const;
    size_t cached_tokens() const;
    const Stats &stats() const {
        return stats_;
    }
    const std::vector<SlotPtr> &index() const {
        return index_;
    }

  private:
    PrefixCacheOptions options_;
    std::vector<SlotPtr> index_;
    std::vector<int> refs_;
    std::set<int> free_; // lowest index first: consecutive allocations stay contiguous
    size_t used_ = 0, high_water_ = 0;
    uint64_t clock_ = 0, next_id_ = 1;
    Stats stats_;
    void ref(int page);
    void unref(int page);
    int allocate();
    void drop(const SlotPtr &, std::vector<int> &release_gpu); // unindex + unref
    bool evict_one(const Slot *keep, std::vector<int> &release_gpu);
};
} // namespace behemoth
