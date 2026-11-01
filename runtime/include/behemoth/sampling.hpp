#pragma once
#include "scheduler.hpp"
#include <random>

namespace behemoth {
void validate_sampling(const SamplingOptions &);
int sample_logits(std::span<const float>, const SamplingOptions &, std::mt19937_64 &);
// Split-vocabulary overload: the token id space is the concatenation
// low ++ high (the two ranks' lm_head halves stay in device order; no host
// interleave copy is needed). Produces identical tokens to the contiguous
// form on the same logits.
// `allowed` (optional): bitset over the 32768 ids (bit i of word i/64); masked-out ids
// are excluded before temperature/top-k/top-p (structured outputs, constraint.hpp).
int sample_logits(std::span<const float> low, std::span<const float> high,
                  const SamplingOptions &, std::mt19937_64 &, const uint64_t *allowed = nullptr);
class TextStream {
    std::string pending_, emitted_;
    std::vector<std::string> stops_;
    bool stopped_ = false;

  public:
    explicit TextStream(std::vector<std::string> stops = {});
    std::string append(std::string_view text, bool finish = false);
    bool stopped() const {
        return stopped_;
    }
};
} // namespace behemoth
