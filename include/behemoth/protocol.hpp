#pragma once
#include "scheduler.hpp"
#include <nlohmann/json.hpp>

namespace behemoth {
inline constexpr const char *model_id = "cgg507/Behemoth-R1-123B-v2-aqlm";
struct ChatRequest {
    std::string rendered;
    SamplingOptions sampling;
    size_t max_new = 256;
    bool stream = false;
    std::vector<std::string> stops;
};
ChatRequest parse_chat(const nlohmann::json &);
} // namespace behemoth
