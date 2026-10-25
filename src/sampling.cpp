#include "behemoth/sampling.hpp"
#include <algorithm>
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
} // namespace
void validate_sampling(const SamplingOptions &p) {
    if (!std::isfinite(p.temperature) || p.temperature < 0 || p.top_k < 0 || p.top_k > 32768 ||
        !std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1)
        throw std::invalid_argument("Invalid sampling parameters");
}
int sample_logits(std::span<const float> logits, const SamplingOptions &p, std::mt19937_64 &rng) {
    validate_sampling(p);
    if (logits.empty())
        throw std::invalid_argument("Empty logits");
    for (float v : logits)
        if (!std::isfinite(v))
            throw std::runtime_error("Nonfinite model logits");
    if (p.temperature == 0)
        return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<size_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    if (p.top_k > 0 && size_t(p.top_k) < order.size())
        order.resize(size_t(p.top_k));
    std::vector<double> probabilities;
    double total = 0;
    for (size_t i : order) {
        double v = std::exp((double(logits[i]) - double(logits[order[0]])) / p.temperature);
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
