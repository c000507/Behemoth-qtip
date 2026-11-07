#pragma once
#include "scheduler.hpp"
#include <random>

namespace behemoth {
void validate_sampling(const SamplingOptions &);
int sample_logits(std::span<const float>, const SamplingOptions &, std::mt19937_64 &);
// Whether the options ask for any logit processing (penalties, logit_bias, min_p with temperature > 0).
bool has_logit_processing(const SamplingOptions &);
// Applies repetition / frequency / presence penalties, logit_bias and min_p to `logits` in place. history: the request's
// tokens (prompt then generated); prompt_size: its prompt length; extra: tokens after history (speculative draft rows).
// Repetition penalty looks at prompt and output, frequency / presence at the output only (vLLM); penalty_range > 0
// limits all three to the most recent tokens. min_p masks with -1e30 (finite: sample_logits rejects non-finite logits).
void apply_logit_processing(std::span<float> logits, const SamplingOptions &, std::span<const int> history, size_t prompt_size,
                            std::span<const int> extra = {});
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
