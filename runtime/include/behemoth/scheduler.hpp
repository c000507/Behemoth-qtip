#pragma once
#include "cache.hpp"
#include <deque>
#include <map>
#include <span>
#include <string>

namespace behemoth {
struct SamplingOptions {
    float temperature = 0;
    int top_k = 0;
    float top_p = 1;
    uint64_t seed = 0;
};
struct Request {
    uint64_t id{};
    std::vector<int> history, pages;
    size_t cached = 0, prompt_size = 0, generated = 0, max_new = 0;
    SamplingOptions sampling;
    bool done = false, inflight = false;
    std::string reason;
};
struct ScheduledRequest {
    uint64_t id;
    size_t begin, count;
    bool sample;
};
struct ScheduledBatch {
    std::vector<ScheduledRequest> requests;
    size_t tokens = 0;
};
class Scheduler {
    PagePool pool_;
    std::map<uint64_t, Request> requests_;
    uint64_t next_id_ = 1;
    size_t token_budget_, chunk_, cursor_ = 0;
    size_t preemptions_ = 0;

  public:
    Scheduler(int pages, size_t token_budget = 1024, size_t prefill_chunk = 256);
    uint64_t submit(std::span<const int> prompt, size_t max_new, SamplingOptions = {});
    void cancel(uint64_t id);
    ScheduledBatch schedule();
    void complete(const ScheduledRequest &, int sampled_token = -1, bool eos = false);
    const Request &request(uint64_t id) const {
        return requests_.at(id);
    }
    void erase(uint64_t id);
    size_t preemptions() const {
        return preemptions_;
    }
    size_t free_pages() const {
        return pool_.free_pages();
    }
    int capacity() const {
        return pool_.capacity();
    }
    size_t pending() const;
};
} // namespace behemoth
