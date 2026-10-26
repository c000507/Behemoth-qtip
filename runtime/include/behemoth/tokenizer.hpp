#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace behemoth {
class Tokenizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit Tokenizer(const std::filesystem::path &model);
    ~Tokenizer();
    std::vector<int> encode(const std::string &rendered) const;
    std::string piece(int token, bool &first) const;
    std::string decode(const std::vector<int> &) const;
};
} // namespace behemoth
