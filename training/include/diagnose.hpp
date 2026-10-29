#pragma once
#include "train.hpp"
namespace qat {
json diagnostic_request(const json &, int position = -1);
json diagnostic_trace(const fs::path &, const json &, int device, size_t arena, int rows);
json diagnostic_compare(const json &, const json &, int topk = 10);
}
