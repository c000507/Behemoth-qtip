#include "behemoth/protocol.hpp"
#include <algorithm>

namespace behemoth {
std::vector<ReasoningPart> ReasoningStream::append(std::string_view text, bool finish) {
    if (finished_) return {};
    pending_.append(text);
    std::vector<ReasoningPart> out;
    auto emit = [&](size_t count) {
        if (!count) return;
        bool reasoning = depth_ != 0;
        if (!out.empty() && out.back().reasoning == reasoning) out.back().text.append(pending_, 0, count);
        else out.push_back({reasoning, pending_.substr(0, count)});
        pending_.erase(0, count);
    };
    constexpr std::string_view open = "<think>", close = "</think>";
    for (;;) {
        size_t opening = pending_.find(open);
        size_t closing = depth_ ? pending_.find(close) : std::string::npos;
        size_t marker = std::min(opening, closing);
        if (marker != std::string::npos) {
            bool entering = marker == opening;
            emit(marker);
            pending_.erase(0, entering ? open.size() : close.size());
            if (entering) ++depth_;
            else --depth_;
            continue;
        }
        size_t hold = 0;
        if (!finish) {
            for (auto tag : {open, close}) {
                if (tag == close && !depth_) continue;
                for (size_t n = 1; n < tag.size() && n <= pending_.size(); ++n)
                    if (std::string_view(pending_).substr(pending_.size() - n) == tag.substr(0, n))
                        hold = std::max(hold, n);
            }
        }
        emit(pending_.size() - hold);
        break;
    }
    finished_ = finish;
    return out;
}
} // namespace behemoth
