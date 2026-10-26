#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace behemoth {
// JSON serializes a nonfinite float as null, but keeps NaN in memory, where it
// compares unequal to itself. Export unavailable timings as actual null values.
inline nlohmann::json timing_json(double value) {
    return std::isfinite(value) ? nlohmann::json(value) : nlohmann::json(nullptr);
}
inline uint64_t next_projection_identity() {
    static std::atomic<uint64_t> next{1};
    const auto id = next.fetch_add(1, std::memory_order_relaxed);
    if (!id)
        throw std::overflow_error("Projection identity exhausted");
    return id;
}
inline double timing_median(std::vector<double> values) {
    if (values.empty())
        throw std::invalid_argument("Empty timing sample");
    for (double value : values)
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("Invalid timing sample");
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
struct TuningMeasurement {
    std::vector<double> samples, confirmation;
    double median_ms() const {
        return timing_median(confirmation.empty() ? samples : confirmation);
    }
};
struct TuningDecision {
    size_t selected = 0, incumbent = 0, challenger = 0;
    std::vector<TuningMeasurement> measurements;
    std::vector<double> confirmation_gain;
};
// Measure independent candidates in rotated forward/reverse rounds, then
// compare the two finalists in A-B-B-A order. Symmetric pairs suppress linear
// clock/thermal drift; medians reject occasional event/host scheduling outliers.
// Close results prefer the earlier candidate rather than chasing noise.
template <class Measure>
TuningDecision tune_candidates(size_t count, Measure measure, double minimum_gain = .02) {
    if (!count || minimum_gain < 0 || minimum_gain >= 1 || !std::isfinite(minimum_gain))
        throw std::invalid_argument("Invalid tuning tournament");
    TuningDecision result;
    result.measurements.resize(count);
    auto sample = [&](size_t candidate, bool confirmation) {
        const double value = measure(candidate);
        if (!std::isfinite(value) || value <= 0)
            throw std::runtime_error("Nonpositive or nonfinite autotuning measurement");
        auto &m = result.measurements[candidate];
        (confirmation ? m.confirmation : m.samples).push_back(value);
        return value;
    };
    constexpr size_t rounds = 7, confirmations = 3;
    for (size_t round = 0; round < rounds; ++round) {
        const size_t offset = (round / 2) % count;
        for (size_t i = 0; i < count; ++i) {
            const size_t position = round % 2 ? count - 1 - i : i;
            sample((offset + position) % count, false);
        }
    }
    std::vector<size_t> order(count);
    for (size_t i = 0; i < count; ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return result.measurements[a].median_ms() < result.measurements[b].median_ms();
    });
    result.selected = order[0];
    if (count == 1)
        return result;
    result.incumbent = std::min(order[0], order[1]);
    result.challenger = std::max(order[0], order[1]);
    size_t wins = 0;
    for (size_t round = 0; round < confirmations; ++round) {
        const double a1 = sample(result.incumbent, true);
        const double b1 = sample(result.challenger, true);
        const double b2 = sample(result.challenger, true);
        const double a2 = sample(result.incumbent, true);
        const double gain = 1 - (b1 + b2) / (a1 + a2);
        result.confirmation_gain.push_back(gain);
        wins += gain > minimum_gain;
    }
    // Gains may be negative or zero, unlike elapsed-time samples.
    auto gains = result.confirmation_gain;
    std::sort(gains.begin(), gains.end());
    result.selected =
        wins >= 2 && gains[gains.size() / 2] > minimum_gain ? result.challenger : result.incumbent;
    return result;
}
} // namespace behemoth
