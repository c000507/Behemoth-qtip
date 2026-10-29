#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"

#include <random>
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
ChatRequest parse_chat(const nlohmann::json &j) {
    if (!j.is_object())
        throw std::invalid_argument("Request must be a JSON object");
    if (j.value("model", std::string(model_id)) != model_id)
        throw std::invalid_argument("Unknown model");
    if (j.value("n", 1) != 1)
        throw std::invalid_argument("Only n=1 is supported");
    for (const char *key : {"tools", "tool_choice", "functions", "function_call", "response_format",
                            "logprobs", "logit_bias"})
        if (j.contains(key))
            throw std::invalid_argument(std::string("Unsupported parameter: ") + key);
    for (const char *key : {"presence_penalty", "frequency_penalty"})
        if (j.value(key, 0.f) != 0)
            throw std::invalid_argument(std::string("Unsupported parameter: ") + key);
    auto messages = j.at("messages");
    if (!messages.is_array() || messages.empty())
        throw std::invalid_argument("messages must be a nonempty array");
    ChatRequest r;
    r.rendered = "<s>";
    for (const auto &m : messages) {
        std::string role = m.at("role").get<std::string>();
        if (!m.at("content").is_string())
            throw std::invalid_argument("Only text message content is supported");
        auto text = m.at("content").get<std::string>();
        if (role == "user")
            r.rendered += "[INST] " + text + "[/INST]";
        else if (role == "system")
            r.rendered += "[SYSTEM_PROMPT] " + text + "[/SYSTEM_PROMPT]";
        else if (role == "assistant")
            r.rendered += " " + text + "</s>";
        else
            throw std::invalid_argument("Only system, user and assistant roles are supported");
    }
    if (j.contains("max_tokens") && j.contains("max_completion_tokens"))
        throw std::invalid_argument("Specify one output-token limit");
    auto maximum = j.contains("max_completion_tokens") ? j.at("max_completion_tokens")
                                                       : j.value("max_tokens", nlohmann::json(256));
    if (!maximum.is_number_integer() || maximum.get<int64_t>() < 1 ||
        maximum.get<int64_t>() > 131072)
        throw std::invalid_argument("Output-token limit must be 1..131072");
    r.max_new = maximum.get<size_t>();
    r.stream = j.value("stream", false);
    if (j.contains("stream_options") && j.at("stream_options").is_object())
        r.include_usage = j.at("stream_options").value("include_usage", false);
    r.sampling.temperature = j.value("temperature", 0.f);
    r.sampling.top_k = j.value("top_k", 0);
    r.sampling.top_p = j.value("top_p", 1.f);
    // Without an explicit seed every request draws a fresh one; a fixed default made
    // regenerations of the same prompt return the identical sample.
    r.sampling.seed = j.contains("seed") ? j.at("seed").get<uint64_t>()
                                         : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    validate_sampling(r.sampling);
    if (j.contains("stop")) {
        if (j.at("stop").is_string())
            r.stops.push_back(j.at("stop").get<std::string>());
        else
            r.stops = j.at("stop").get<std::vector<std::string>>();
        if (r.stops.size() > 16)
            throw std::invalid_argument("At most 16 stop strings are supported");
        for (const auto &s : r.stops)
            if (s.empty())
                throw std::invalid_argument("Stop strings must be nonempty");
    }
    return r;
}
} // namespace behemoth
