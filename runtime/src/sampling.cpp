#include "behemoth/sampling.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <cstdlib>
#include <cmath>

namespace behemoth {
namespace {
std::pair<std::string, size_t> utf8_prefix(std::string_view x, bool final) {
    std::string out;
    size_t i = 0;
    while (i < x.size()) {
        auto c = uint8_t(x[i]);
        if (c < 128) {
            out.push_back(char(c));
            ++i;
            continue;
        }
        size_t n = c >= 0xc2 && c <= 0xdf   ? 2
                   : c >= 0xe0 && c <= 0xef ? 3
                   : c >= 0xf0 && c <= 0xf4 ? 4
                                            : 0;
        if (!n) {
            out += "\xef\xbf\xbd";
            ++i;
            continue;
        }
        if (n > x.size() - i) {
            if (!final)
                break;
            out += "\xef\xbf\xbd";
            i = x.size();
            break;
        }
        bool valid = true;
        for (size_t j = 1; j < n; ++j)
            valid &= (uint8_t(x[i + j]) & 0xc0) == 0x80;
        if (n >= 3) {
            auto second = uint8_t(x[i + 1]);
            valid &= !(c == 0xe0 && second < 0xa0) && !(c == 0xed && second >= 0xa0) &&
                     !(c == 0xf0 && second < 0x90) && !(c == 0xf4 && second >= 0x90);
        }
        if (!valid) {
            out += "\xef\xbf\xbd";
            ++i;
            continue;
        }
        out.append(x.substr(i, n));
        i += n;
    }
    return {out, i};
}
// Reused per-thread buffers: the decode path calls this once per token and
// must not allocate (PERF_ROUND3 item 2).
thread_local std::vector<size_t> order_buffer;
thread_local std::vector<double> probability_buffer;
thread_local std::vector<uint64_t> key_buffer, key_scratch;
// The exact full order of the sampler (value descending, index ascending; -0 == +0 ties by
// index) as a radix sort of 64-bit keys: descending-orderable float bits << 32 | index. Same
// permutation as std::sort with `descending`, ~10x faster than the indirect comparator sort on
// the full 32768-entry vocabulary (PREFILL_SCHED_REPORT.md).
template <class Value> void full_order(size_t n, Value value, std::vector<size_t> &order) {
    auto &keys = key_buffer;
    auto &tmp = key_scratch;
    keys.resize(n);
    tmp.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float v = value(i);
        if (v == 0)
            v = 0; // -0 == +0 in the comparator: same key, tie broken by index
        uint32_t u;
        std::memcpy(&u, &v, 4);
        const uint32_t ascending = (u & 0x80000000U) ? ~u : (u | 0x80000000U);
        keys[i] = uint64_t(~ascending) << 32 | uint64_t(i);
    }
    for (int shift = 0; shift < 64; shift += 8) {
        std::array<uint32_t, 257> count{};
        for (auto k : keys)
            ++count[size_t((k >> shift) & 0xffU) + 1];
        if (count[size_t((keys[0] >> shift) & 0xffU) + 1] == n)
            continue; // every key shares this digit
        for (size_t d = 0; d < 256; ++d)
            count[d + 1] += count[d];
        for (auto k : keys)
            tmp[count[size_t((k >> shift) & 0xffU)]++] = k;
        keys.swap(tmp);
    }
    order.resize(n);
    for (size_t i = 0; i < n; ++i)
        order[i] = size_t(keys[i] & 0xffffffffU);
}
} // namespace
void validate_sampling(const SamplingOptions &p) {
    if (!std::isfinite(p.temperature) || p.temperature < 0 || p.top_k < 0 || p.top_k > (1 << 24) ||
        !std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1)
        throw std::invalid_argument("Invalid sampling parameters");
    if (!std::isfinite(p.repetition_penalty) || p.repetition_penalty <= 0 || p.repetition_penalty > 10)
        throw std::invalid_argument("repetition_penalty must be in (0, 10]");
    if (!std::isfinite(p.frequency_penalty) || std::abs(p.frequency_penalty) > 2 || !std::isfinite(p.presence_penalty) ||
        std::abs(p.presence_penalty) > 2)
        throw std::invalid_argument("frequency_penalty and presence_penalty must be in [-2, 2]");
    if (!std::isfinite(p.min_p) || p.min_p < 0 || p.min_p > 1)
        throw std::invalid_argument("min_p must be in [0, 1]");
    if (p.penalty_range < 0)
        throw std::invalid_argument("repetition_penalty_range must be >= 0");
    if (p.logit_bias.size() > 1024)
        throw std::invalid_argument("At most 1024 logit_bias entries are supported");
    for (const auto &[id, bias] : p.logit_bias)
        if (id < 0 || !std::isfinite(bias) || std::abs(bias) > 100)
            throw std::invalid_argument("logit_bias needs token ids >= 0 and biases in [-100, 100]");
}
bool has_logit_processing(const SamplingOptions &p) {
    return p.repetition_penalty != 1 || p.frequency_penalty != 0 || p.presence_penalty != 0 || !p.logit_bias.empty() ||
           (p.min_p > 0 && p.temperature > 0);
}
void apply_logit_processing(std::span<float> logits, const SamplingOptions &p, std::span<const int> history, size_t prompt_size,
                            std::span<const int> extra) {
    const size_t total = history.size() + extra.size(), n = logits.size();
    const size_t begin = p.penalty_range > 0 && size_t(p.penalty_range) < total ? total - size_t(p.penalty_range) : 0;
    auto token = [&](size_t i) { return i < history.size() ? history[i] : extra[i - history.size()]; };
    if (p.repetition_penalty != 1 || p.frequency_penalty != 0 || p.presence_penalty != 0) {
        thread_local std::vector<int> counts; // generated-token counts, -1: seen only in the prompt (or not at all: 0)
        thread_local std::vector<int> touched;
        counts.assign(n, 0);
        touched.clear();
        for (size_t i = begin; i < total; ++i) {
            const int t = token(i);
            if (t < 0 || size_t(t) >= n)
                continue;
            if (!counts[size_t(t)])
                touched.push_back(t);
            if (i >= prompt_size)
                counts[size_t(t)] = std::max(counts[size_t(t)], 0) + 1;
            else if (!counts[size_t(t)])
                counts[size_t(t)] = -1;
        }
        for (int t : touched) {
            float &l = logits[size_t(t)];
            if (p.repetition_penalty != 1)
                l = l > 0 ? l / p.repetition_penalty : l * p.repetition_penalty;
            if (const int c = counts[size_t(t)]; c > 0)
                l -= p.frequency_penalty * float(c) + p.presence_penalty;
        }
    }
    for (const auto &[id, bias] : p.logit_bias)
        if (size_t(id) < n)
            logits[size_t(id)] += bias;
    if (p.min_p > 0 && p.temperature > 0) { // p_i / p_max = exp((l_i - l_max) / T) >= min_p
        const float m = *std::max_element(logits.begin(), logits.end());
        const float threshold = m + p.temperature * std::log(p.min_p);
        for (float &l : logits)
            if (l < threshold)
                l = -1e30f;
    }
}
int sample_logits(std::span<const float> logits, const SamplingOptions &p, std::mt19937_64 &rng) {
    return sample_logits(logits, {}, p, rng);
}
namespace {
// Shared sampler body; `value(i)` is the logit of id i (-inf for ids a constraint masks out).
template <class Value> int sample_values(size_t n, Value value, const SamplingOptions &p, std::mt19937_64 &rng) {
    if (p.temperature == 0) {
        size_t best = 0;
        for (size_t i = 1; i < n; ++i)
            if (value(i) > value(best))
                best = i;
        return int(best);
    }
    if (p.top_k == 0 && p.top_p >= 1) {
        // No truncation: one pass of float exp over the unsorted logits and a CDF walk, O(n)
        // without the full-vocabulary order (perf/single-decode: the sorted path cost 2.3 ms per
        // token standalone, ~6 ms per token in the serving loop, at batch 1 more than 10 % of a
        // 45 ms decode step). Same distribution as the sorted path; a given seed draws
        // differently. BEHEMOTH_SAMPLER_SORTED=1 restores the sorted path.
        const char *sorted = std::getenv("BEHEMOTH_SAMPLER_SORTED"); // per call: tests toggle it
        if (!(sorted && std::atoi(sorted) != 0)) {
            float m = value(0);
            for (size_t i = 1; i < n; ++i)
                m = std::max(m, value(i));
            auto &probabilities = probability_buffer;
            probabilities.resize(n);
            const float inv_t = 1.f / p.temperature;
            double total = 0;
            for (size_t i = 0; i < n; ++i) {
                const float e = std::exp((value(i) - m) * inv_t);
                probabilities[i] = e;
                total += e;
            }
            const double u = std::uniform_real_distribution<double>(0.0, total)(rng);
            double cumulative = 0;
            for (size_t i = 0; i < n; ++i) {
                cumulative += probabilities[i];
                if (cumulative > u)
                    return int(i);
            }
            size_t last = n - 1; // rounding: the last id with non-zero probability
            while (last > 0 && probabilities[last] == 0)
                --last;
            return int(last);
        }
    }
    auto &order = order_buffer;
    order.resize(n);
    std::iota(order.begin(), order.end(), size_t(0));
    auto descending = [&](size_t a, size_t b) {
        return value(a) == value(b) ? a < b : value(a) > value(b);
    };
    if (p.top_k > 0 && size_t(p.top_k) < n) {
        // partial_sort yields exactly the first top_k entries of the full sort
        // (the comparator is a total order), at O(n log k) instead of a
        // full-vocab std::sort per token.
        std::partial_sort(order.begin(), order.begin() + p.top_k, order.end(), descending);
        order.resize(size_t(p.top_k));
    } else {
        // Exact path (no top-k cut): the full order, as a radix sort (identical permutation);
        // the probability math below is unchanged.
        full_order(n, value, order);
    }
    auto &probabilities = probability_buffer;
    probabilities.clear();
    double total = 0;
    for (size_t i : order) {
        double v = std::exp((double(value(i)) - double(value(order[0]))) / p.temperature);
        probabilities.push_back(v);
        total += v;
    }
    double cumulative = 0;
    size_t keep = 0;
    do {
        cumulative += probabilities[keep++];
    } while (keep < order.size() && cumulative < p.top_p * total);
    probabilities.resize(keep);
    std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
    return int(order[dist(rng)]);
}
} // namespace
int sample_logits(std::span<const float> low, std::span<const float> high, const SamplingOptions &p,
                  std::mt19937_64 &rng, const uint64_t *allowed) {
    validate_sampling(p);
    const size_t n = low.size() + high.size();
    if (!n)
        throw std::invalid_argument("Empty logits");
    if (high.empty() && !allowed && p.temperature == 0) {
        // Greedy over one contiguous row: one vectorised pass for the maximum and the finite check (x * 0 is NaN for a
        // non-finite x), then the first index holding the maximum (the scalar loop's tie rule). The scalar form below cost
        // ~0.9 ms per 248k-logit row, 5 % of a 19 ms decode step (docs/neuron-qwen397-batch.md).
        const float *x = low.data();
        float m = -std::numeric_limits<float>::infinity(), acc = 0;
#pragma omp simd reduction(max : m) reduction(+ : acc)
        for (size_t i = 0; i < n; ++i) {
            m = x[i] > m ? x[i] : m;
            acc += x[i] * 0.f;
        }
        if (!(acc == 0))
            throw std::runtime_error("Nonfinite model logits");
        const size_t block = 1024;
        for (size_t at = 0; at < n; at += block) {
            const size_t end = std::min(n, at + block);
            int hit = 0;
#pragma omp simd reduction(| : hit)
            for (size_t i = at; i < end; ++i)
                hit |= int(x[i] == m);
            if (hit)
                for (size_t i = at; i < end; ++i)
                    if (x[i] == m)
                        return int(i);
        }
        return 0;
    }
    auto value = [&](size_t i) -> float { return i < low.size() ? low[i] : high[i - low.size()]; };
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(value(i)))
            throw std::runtime_error("Nonfinite model logits");
    if (!allowed)
        return sample_values(n, value, p, rng);
    // Constrained: disallowed ids get -inf (zero probability, never the argmax).
    // Temperature, top-k and top-p then act on the allowed ids only.
    size_t first_allowed = n;
    for (size_t i = 0; i < n && first_allowed == n; ++i)
        if ((allowed[i >> 6] >> (i & 63)) & 1)
            first_allowed = i;
    if (first_allowed == n)
        throw std::invalid_argument("Constraint allows no token");
    auto masked = [&](size_t i) -> float {
        return ((allowed[i >> 6] >> (i & 63)) & 1) ? value(i) : -std::numeric_limits<float>::infinity();
    };
    if (p.temperature == 0) { // argmax over allowed ids, lowest id on ties
        size_t best = first_allowed;
        for (size_t i = first_allowed + 1; i < n; ++i)
            if (masked(i) > masked(best))
                best = i;
        return int(best);
    }
    // Masked ids contribute exp(-inf) = 0 and are never drawn.
    return sample_values(n, masked, p, rng);
}
TextStream::TextStream(std::vector<std::string> stops) : stops_(std::move(stops)) {
    for (const auto &s : stops_)
        if (s.empty())
            throw std::invalid_argument("Empty stop string");
}
std::string TextStream::append(std::string_view text, bool finish) {
    if (stopped_)
        return {};
    pending_ += text;
    size_t end = pending_.size();
    for (const auto &s : stops_) {
        size_t i = pending_.find(s);
        if (i != std::string::npos) {
            end = std::min(end, i);
            stopped_ = true;
        }
    }
    if (!stopped_ && !finish) {
        for (const auto &s : stops_)
            for (size_t n = 1; n < s.size() && n <= pending_.size(); ++n)
                if (pending_.compare(pending_.size() - n, n, s, 0, n) == 0)
                    end = std::min(end, pending_.size() - n);
    }
    auto [out, consumed] =
        utf8_prefix(std::string_view(pending_).substr(0, end), finish || stopped_);
    pending_.erase(0, consumed);
    if (stopped_ || finish)
        pending_.clear();
    return out;
}
} // namespace behemoth
