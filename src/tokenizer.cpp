#include "behemoth/tokenizer.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>
#include <stdexcept>

namespace behemoth {
struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
    std::vector<std::pair<std::string, int>> added;
    std::map<int, bool> special;
};
Tokenizer::Tokenizer(const std::filesystem::path &root) : impl_(std::make_unique<Impl>()) {
    auto status = impl_->processor.Load((root / "tokenizer.model").string());
    if (!status.ok())
        throw std::runtime_error(status.ToString());
    std::ifstream f(root / "tokenizer_config.json");
    if (!f)
        throw std::runtime_error("Missing tokenizer_config.json");
    auto j = nlohmann::json::parse(f);
    if (j.at("legacy") != true || j.at("add_bos_token") != true || j.at("add_eos_token") != false ||
        impl_->processor.GetPieceSize() != 32768)
        throw std::invalid_argument("Unsupported tokenizer configuration");
    for (auto it = j.at("added_tokens_decoder").begin(); it != j.at("added_tokens_decoder").end();
         ++it) {
        int id = std::stoi(it.key());
        const auto &a = it.value();
        std::string text = a.at("content");
        if (id < 0 || id >= 32768 || text.empty() || a.at("lstrip") != false ||
            a.at("rstrip") != false || a.at("single_word") != false || a.at("normalized") != false)
            throw std::invalid_argument("Unsupported added-token behavior");
        impl_->added.emplace_back(text, id);
        impl_->special[id] = a.at("special").get<bool>();
    }
    std::sort(impl_->added.begin(), impl_->added.end(),
              [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
}
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string &text) const {
    std::vector<int> tokens;
    size_t segment = 0, i = 0;
    auto flush = [&](size_t end) {
        if (end > segment) {
            std::vector<int> ids;
            auto status = impl_->processor.Encode(text.substr(segment, end - segment), &ids);
            if (!status.ok())
                throw std::runtime_error(status.ToString());
            tokens.insert(tokens.end(), ids.begin(), ids.end());
        }
    };
    while (i < text.size()) {
        const std::pair<std::string, int> *found = nullptr;
        if (text[i] == '<' || text[i] == '[')
            for (const auto &token : impl_->added)
                if (text.compare(i, token.first.size(), token.first) == 0) {
                    found = &token;
                    break;
                }
        if (found) {
            flush(i);
            tokens.push_back(found->second);
            i += found->first.size();
            segment = i;
        } else
            ++i;
    }
    flush(text.size());
    return tokens;
}
std::string Tokenizer::piece(int id, bool &first) const {
    if (id < 0 || id >= 32768)
        throw std::invalid_argument("Token outside vocabulary");
    auto special = impl_->special.find(id);
    if (special != impl_->special.end() && special->second)
        return {};
    std::string p = impl_->processor.IdToPiece(id);
    if (p.size() == 6 && p.starts_with("<0x") && p.back() == '>') {
        char c = char(std::stoi(p.substr(3, 2), nullptr, 16));
        first = false;
        return std::string(1, c);
    }
    const std::string underline = "\xe2\x96\x81";
    size_t pos = 0;
    while ((pos = p.find(underline, pos)) != std::string::npos) {
        p.replace(pos, underline.size(), " ");
        ++pos;
    }
    if (first && p.starts_with(' '))
        p.erase(0, 1);
    if (!p.empty())
        first = false;
    return p;
}
std::string Tokenizer::decode(const std::vector<int> &ids) const {
    std::string out;
    bool first = true;
    for (int t : ids)
        out += piece(t, first);
    return out;
}
} // namespace behemoth
