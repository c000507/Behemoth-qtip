#pragma once
#include "scheduler.hpp"
#include <random>

namespace behemoth {
void validate_sampling(const SamplingOptions &);
int sample_logits(std::span<const float>, const SamplingOptions &, std::mt19937_64 &);
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
