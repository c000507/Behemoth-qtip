#pragma once
#include "model.hpp"
#include <functional>
#include <nlohmann/json.hpp>

namespace behemoth {
class TokenConstraint;
struct GenerationEvent {
    uint64_t id{};
    int token = -1;
    bool done = false;
    std::string reason, error;
    size_t prompt_tokens = 0, completion_tokens = 0;
    size_t cached_tokens = 0; // prompt tokens whose KV came from the prefix cache
    // Per-request metrics (optional, for the server's request_metrics log; an engine that does not fill them leaves
    // the defaults): submit -> first scheduled in ms (-1: unknown), and the request's decode steps that emitted a
    // token, the drafts verified for it and the drafts accepted (speculation).
    double queue_wait_ms = -1;
    size_t spec_steps = 0, drafted = 0, accepted = 0;
};
using GenerationCallback = std::function<void(const GenerationEvent &)>;
class Engine {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit Engine(const RuntimeOptions &);
    ~Engine();
    // `constraint` (optional): structured-output token constraint, applied before sampling.
    uint64_t submit(std::span<const int>, size_t max_new, SamplingOptions, GenerationCallback,
                    std::shared_ptr<TokenConstraint> constraint = nullptr);
    void cancel(uint64_t);
    void shutdown();
    nlohmann::json metrics() const;
    nlohmann::json projection_tuning(int layer = 0) const;
    bool healthy() const;
};
} // namespace behemoth
