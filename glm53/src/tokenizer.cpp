#include "glm53.hpp"
#include "minja/minja.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <algorithm>
#include <fstream>
#include <memory>
#include <pcre2.h>
namespace glm53 {
namespace {
std::string utf8(int c) {
    if (c < 128)
        return std::string(1, char(c));
    if (c < 2048)
        return {char(0xc0 | (c >> 6)), char(0x80 | (c & 63))};
    return {char(0xe0 | (c >> 12)), char(0x80 | ((c >> 6) & 63)), char(0x80 | (c & 63))};
}
std::vector<std::string> split(const std::string &text, const std::string &pattern) {
    int error;
    PCRE2_SIZE where;
    auto code = std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>(
        pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF | PCRE2_UCP,
                      &error, &where, nullptr),
        pcre2_code_free);
    check(bool(code), "unsupported tokenizer regex at byte " + std::to_string(where));
    auto match = std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>(
        pcre2_match_data_create_from_pattern(code.get(), nullptr), pcre2_match_data_free);
    std::vector<std::string> out;
    size_t cursor = 0;
    while (cursor < text.size()) {
        int rc = pcre2_match(code.get(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), cursor, 0,
                             match.get(), nullptr);
        if (rc == PCRE2_ERROR_NOMATCH) {
            out.push_back(text.substr(cursor));
            break;
        }
        check(rc > 0, "invalid UTF-8 or tokenizer regex match failure");
        auto v = pcre2_get_ovector_pointer(match.get());
        check(v[1] > v[0], "tokenizer regex produced empty match");
        if (v[0] > cursor)
            out.push_back(text.substr(cursor, v[0] - cursor));
        out.push_back(text.substr(v[0], v[1] - v[0]));
        cursor = v[1];
    }
    return out;
}
} // namespace
Tokenizer::Tokenizer(const fs::path &path) {
    auto j = read_json(path / "tokenizer.json");
    config_ = read_json(path / "tokenizer_config.json");
    if (!config_.contains("chat_template") && fs::exists(path / "chat_template.jinja")) {
        std::ifstream f(path / "chat_template.jinja");
        config_["chat_template"] = std::string(std::istreambuf_iterator<char>(f), {});
    }
    const auto &m = j.at("model");
    ignore_merges_ = m.value("ignore_merges", false);
    check(m.at("type") == "BPE" && !m.value("byte_fallback", false), "expected byte-level BPE tokenizer");
    check(!m.contains("dropout") || m["dropout"].is_null(), "tokenizer dropout unsupported");
    for (const auto &v : m.at("vocab").items()) {
        int id = v.value();
        check(id >= 0, "negative vocabulary ID");
        vocab_[v.key()] = id;
    }
    int rank = 0;
    for (const auto &item : m.at("merges")) {
        std::string a, b;
        if (item.is_array()) {
            check(item.size() == 2, "invalid BPE merge");
            a = item[0];
            b = item[1];
        } else {
            std::string s = item;
            auto p = s.find(' ');
            check(p != std::string::npos, "invalid BPE merge");
            a = s.substr(0, p);
            b = s.substr(p + 1);
        }
        check(merges_.emplace(a + std::string(1, '\0') + b, rank++).second, "duplicate BPE merge");
    }
    for (const auto &a : j.value("added_tokens", json::array())) {
        check(!a.value("lstrip", false) && !a.value("rstrip", false) && !a.value("single_word", false) &&
                  !a.value("normalized", false),
              "unsupported added-token matching flags");
        std::string text = a.at("content");
        int id = a.at("id");
        check(!text.empty() && id >= 0, "invalid added token");
        added_[text] = id;
        vocab_[text] = id;
    }
    int maxid = 0;
    for (const auto &[s, id] : vocab_) {
        (void)s;
        maxid = std::max(maxid, id);
    }
    decode_.resize(size_t(maxid) + 1);
    for (const auto &[s, id] : vocab_) {
        check(decode_[id].empty() || decode_[id] == s, "duplicate vocabulary ID");
        decode_[id] = s;
    }
    int next = 256;
    for (int b = 0; b < 256; ++b) {
        int code = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) ? b : next++;
        byte_map_[b] = utf8(code);
        inverse_[byte_map_[b]] = uint8_t(b);
        check(vocab_.contains(byte_map_[b]), "missing byte token in BPE vocabulary");
    }
    bool bytelevel = false;
    std::function<void(const json &)> pre = [&](const json &p) {
        std::string type = p.at("type");
        if (type == "Sequence") {
            for (const auto &v : p.at("pretokenizers"))
                pre(v);
        } else if (type == "Split") {
            check(p.at("behavior") == "Isolated" && !p.value("invert", false) &&
                      p.at("pattern").contains("Regex"),
                  "unsupported BPE split behavior");
            patterns_.push_back(p["pattern"]["Regex"]);
        } else if (type == "ByteLevel") {
            bytelevel = true;
            prefix_space_ = p.value("add_prefix_space", false);
            if (p.value("use_regex", true)) {
                check(patterns_.empty(), "ByteLevel regex after custom Split unsupported");
                patterns_.push_back(
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
            }
        } else
            throw std::runtime_error("unsupported pre-tokenizer " + type);
    };
    pre(j.at("pre_tokenizer"));
    check(bytelevel, "ByteLevel pre-tokenizer required");
    check(j.at("decoder").at("type") == "ByteLevel", "ByteLevel decoder required");
    check(!j.contains("normalizer") || j["normalizer"].is_null(), "tokenizer normalizer unsupported");
    // tokenizer.json's serialized truncation/padding is deliberately not applied.
}
void Tokenizer::bpe(std::string_view word, std::vector<int> &out) const {
    std::vector<std::string> symbols;
    for (unsigned char b : word)
        symbols.push_back(byte_map_[b]);
    if (ignore_merges_) {
        std::string whole;
        for (const auto &s : symbols)
            whole += s;
        if (auto it = vocab_.find(whole); it != vocab_.end()) {
            out.push_back(it->second);
            return;
        }
    }
    while (symbols.size() > 1) {
        int best = INT32_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merges_.find(symbols[i] + std::string(1, '\0') + symbols[i + 1]);
            if (it != merges_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == INT32_MAX)
            break;
        symbols[at] += symbols[at + 1];
        symbols.erase(symbols.begin() + at + 1);
    }
    for (const auto &s : symbols)
        out.push_back(vocab_.at(s));
}
std::vector<int> Tokenizer::encode(const std::string &text) const {
    std::vector<int> out;
    auto ordinary = [&](std::string s) {
        if (s.empty())
            return;
        if (prefix_space_ && s[0] != ' ')
            s = " " + s;
        std::vector<std::string> pieces{s};
        for (const auto &pattern : patterns_) {
            std::vector<std::string> next;
            for (const auto &p : pieces) {
                auto v = split(p, pattern);
                next.insert(next.end(), v.begin(), v.end());
            }
            pieces = std::move(next);
        }
        for (const auto &p : pieces)
            bpe(p, out);
    };
    size_t at = 0, plain = 0;
    while (at < text.size()) {
        const std::pair<const std::string, int> *match = nullptr;
        for (const auto &s : added_)
            if (text.compare(at, s.first.size(), s.first) == 0 &&
                (!match || s.first.size() > match->first.size()))
                match = &s;
        if (match) {
            ordinary(text.substr(plain, at - plain));
            out.push_back(match->second);
            at += match->first.size();
            plain = at;
        } else
            ++at;
    }
    ordinary(text.substr(plain));
    return out;
}
std::string Tokenizer::decode(const std::vector<int> &ids, bool special) const {
    std::string out;
    for (int id : ids) {
        check(id >= 0, "negative token ID");
        // Match HF decoding for padded LM-head rows with no tokenizer entry.
        if (size_t(id) >= decode_.size() || decode_[id].empty())
            continue;
        const auto &s = decode_[id];
        if (added_.contains(s)) {
            if (special)
                out += s;
            continue;
        }
        for (size_t i = 0; i < s.size();) {
            unsigned char b = s[i];
            size_t n = b < 128 ? 1 : (b < 224 ? 2 : 3);
            check(i + n <= s.size(), "invalid byte vocabulary encoding");
            auto it = inverse_.find(s.substr(i, n));
            check(it != inverse_.end(), "non-byte BPE vocabulary token");
            out += char(it->second);
            i += n;
        }
    }
    return out;
}
std::string Tokenizer::chat(const json &messages, bool generation) const {
    check(messages.is_array() && !messages.empty(), "messages must be nonempty array");
    for (const auto &message : messages) {
        std::string role = message.at("role");
        check(role == "system" || role == "user" || role == "assistant", "text-only chat roles required");
        check(message.at("content").is_string() && !message.contains("tool_calls"),
              "text-only content required; tool calls are not supported");
    }
    auto source = config_.at("chat_template");
    if (source.is_array()) {
        json chosen;
        for (const auto &entry : source)
            if (entry.at("name") == "default")
                chosen = entry.at("template");
        source = chosen;
    }
    check(source.is_string(), "a default chat_template is required");
    auto token = [&](const char *key) -> std::string {
        auto value = config_.value(key, json(nullptr));
        if (value.is_null())
            return "";
        if (value.is_string())
            return value.get<std::string>();
        return value.at("content").get<std::string>();
    };
    nlohmann::ordered_json inputs = {{"messages", nlohmann::ordered_json::parse(messages.dump())},
                                     {"bos_token", token("bos_token")},
                                     {"eos_token", token("eos_token")},
                                     {"add_generation_prompt", generation},
                                     {"tools", nullptr}};
    auto parsed =
        minja::Parser::parse(source.get<std::string>(),
                             {.trim_blocks = true, .lstrip_blocks = true, .keep_trailing_newline = false});
    return parsed->render(minja::Context::make(minja::Value(inputs)));
}
json Tokenizer::summary() const {
    return {{"model", "byte-level BPE"},
            {"vocabulary_entries", vocab_.size()},
            {"merges", merges_.size()},
            {"added_tokens", added_.size()},
            {"regex_patterns", patterns_},
            {"serialized_truncation_applied", false},
            {"bos_token", config_.value("bos_token", json(nullptr))},
            {"eos_token", config_.value("eos_token", json(nullptr))}};
}
} // namespace glm53
