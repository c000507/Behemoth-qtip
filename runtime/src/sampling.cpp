#include "behemoth/sampling.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>

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
    if (!std::isfinite(p.temperature) || p.temperature < 0 || p.top_k < 0 || p.top_k > 32768 ||
        !std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1)
        throw std::invalid_argument("Invalid sampling parameters");
}
int sample_logits(std::span<const float> logits, const SamplingOptions &p, std::mt19937_64 &rng) {
    return sample_logits(logits, {}, p, rng);
}
int sample_logits(std::span<const float> low, std::span<const float> high, const SamplingOptions &p,
                  std::mt19937_64 &rng) {
    validate_sampling(p);
    const size_t n = low.size() + high.size();
    if (!n)
        throw std::invalid_argument("Empty logits");
    auto value = [&](size_t i) -> float { return i < low.size() ? low[i] : high[i - low.size()]; };
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(value(i)))
            throw std::runtime_error("Nonfinite model logits");
    if (p.temperature == 0) {
        size_t best = 0;
        for (size_t i = 1; i < n; ++i)
            if (value(i) > value(best))
                best = i;
        return int(best);
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
