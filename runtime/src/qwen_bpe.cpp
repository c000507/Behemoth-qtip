// GPT2 byte-level BPE (Qwen flavour): vocab.json + merges.txt, special tokens from
// tokenizer_config.json. Host-only, links into behemoth_core.
#include "behemoth/qwen_bpe.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace behemoth {
namespace {
// GPT2 bytes_to_unicode: printable bytes map to themselves, others to 0x100+n
std::array<std::string, 256> byte_maps() {
    std::array<std::string, 256> m;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff))
            m[size_t(b)] = std::string(1, char(b));
        else {
            int c = 256 + n++;
            std::string s;
            s.push_back(char(0xc0 | (c >> 6)));
            s.push_back(char(0x80 | (c & 0x3f)));
            m[size_t(b)] = s;
        }
    }
    return m;
}
const std::array<std::string, 256> &bytes_to_unicode() {
    static const std::array<std::string, 256> m = byte_maps();
    return m;
}
const std::map<std::string, unsigned char> &unicode_to_bytes() {
    static std::map<std::string, unsigned char> m = [] {
        std::map<std::string, unsigned char> r;
        const auto &bm = bytes_to_unicode();
        for (int b = 0; b < 256; ++b) r[bm[size_t(b)]] = unsigned char(b);
        return r;
    }();
    return m;
}
}  // namespace

QwenTokenizer::QwenTokenizer(const std::filesystem::path &dir) {
    using json = nlohmann::json;
    {
        std::ifstream f(dir / "vocab.json");
        if (!f) throw std::runtime_error("no vocab.json in " + dir.string());
        json v = json::parse(f);
        for (auto it = v.begin(); it != v.end(); ++it) vocab_[it.key()] = it.value();
    }
    {
        std::ifstream f(dir / "merges.txt");
        if (!f) throw std::runtime_error("no merges.txt in " + dir.string());
        std::string line;
        int rank = 0;
        bool header = true;
        while (std::getline(f, line)) {
            if (header && line.rfind("#version", 0) == 0) {
                header = false;
                continue;
            }
            header = false;
            auto sp = line.find(' ');
            if (sp == std::string::npos || line.empty()) continue;
            ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
        }
    }
    if (std::filesystem::exists(dir / "tokenizer_config.json")) {
        std::ifstream f(dir / "tokenizer_config.json");
        json c = json::parse(f);
        if (c.contains("added_tokens_decoder")) {
            for (auto it = c["added_tokens_decoder"].begin();
                 it != c["added_tokens_decoder"].end(); ++it) {
                std::string content = it.value().value("content", "");
                int id = std::stoi(it.key());
                if (!content.empty()) {
                    special_[content] = id;
                    vocab_[content] = id;
                }
            }
        }
    }
    int max_id = 0;
    for (auto &kv : vocab_) max_id = std::max(max_id, kv.second);
    id_to_token_.assign(size_t(max_id) + 1, "");
    for (auto &kv : vocab_) id_to_token_[size_t(kv.second)] = kv.first;
}

std::vector<std::string> QwenTokenizer::pre_split(const std::string &text) const {
    // hand-rolled GPT2 pattern; bytes >= 0x80 count as letters (Latin-diacritics exact, CJK may
    // differ from the HF pre-tokenizer)
    std::vector<std::string> out;
    size_t i = 0;
    auto is_letter = [](unsigned char c) {
        return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= 0x80;
    };
    auto is_digit = [](unsigned char c) { return c >= '0' && c <= '9'; };
    auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    static const char *cts[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    while (i < text.size()) {
        unsigned char c = unsigned(text[i]);
        if (is_space(c)) {
            size_t j = i;
            while (j < text.size() && is_space(unsigned(text[j]))) ++j;
            if (j < text.size() && j - i > 1) {  // \s+(?!\S): keep last space for the next word
                out.push_back(text.substr(i, j - i - 1));
                i = j - 1;
                ++i;
                continue;
            }
            out.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        // contractions: part of the "other" run boundary — emit ('s|'t|'re|'ve|'m|'ll|'d) alone
        if (c == '\'') {
            for (auto ct : cts) {
                size_t cl = std::strlen(ct);
                if (text.compare(i, cl, ct) == 0) {
                    out.push_back(text.substr(i, cl));
                    i += cl;
                    goto next;
                }
            }
        }
        {
            size_t j = i;
            if (c == ' ' && i + 1 < text.size() && !is_space(unsigned(text[i + 1]))) ++j;
            unsigned char c1 = j < text.size() ? unsigned(text[j]) : 0;
            if (is_letter(c1)) {
                while (j < text.size() && is_letter(unsigned(text[j]))) ++j;
            } else if (is_digit(c1)) {
                while (j < text.size() && is_digit(unsigned(text[j]))) ++j;
            } else {
                while (j < text.size() && !is_space(unsigned(text[j])) && !is_letter(unsigned(text[j])) &&
                       !is_digit(unsigned(text[j])) && text[j] != '\'')
                    ++j;
                if (j == i) ++j;  // lone apostrophe
            }
            out.push_back(text.substr(i, j - i));
            i = j;
        }
    next:;
    }
    return out;
}

std::vector<std::string> QwenTokenizer::bpe(const std::string &word) const {
    const auto &bm = bytes_to_unicode();
    std::vector<std::string> syms;
    for (unsigned char c : word) syms.push_back(bm[c]);
    while (syms.size() > 1) {
        int best = -1, best_rank = 1 << 30;
        for (size_t k = 0; k + 1 < syms.size(); ++k) {
            auto it = ranks_.find({syms[k], syms[k + 1]});
            if (it != ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = int(k);
            }
        }
        if (best < 0) break;
        syms[size_t(best)] = syms[size_t(best)] + syms[size_t(best) + 1];
        syms.erase(syms.begin() + best + 1);
    }
    return syms;
}

std::vector<int> QwenTokenizer::encode(const std::string &text, bool add_special) const {
    (void)add_special;  // chat template wrapping is the caller's job
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.size()) {
        size_t best = std::string::npos;
        size_t best_len = 0;
        for (auto &kv : special_) {
            if (kv.first.empty()) continue;
            size_t p = text.find(kv.first, i);
            if (p != std::string::npos &&
                (best == std::string::npos || p < best || (p == best && kv.first.size() > best_len))) {
                best = p;
                best_len = kv.first.size();
            }
        }
        size_t end = (best == std::string::npos) ? text.size() : best;
        if (end > i) {
            for (auto &w : pre_split(text.substr(i, end - i)))
                for (auto &s : bpe(w)) {
                    auto it = vocab_.find(s);
                    if (it != vocab_.end()) ids.push_back(it->second);
                }
        }
        if (best == std::string::npos) break;
        ids.push_back(special_.at(text.substr(best, best_len)));
        i = best + best_len;
    }
    return ids;
}

std::string QwenTokenizer::decode(const std::vector<int> &ids) const {
    const auto &u2b = unicode_to_bytes();
    std::string out;
    std::string cur;  // accumulate unicode symbols of one token, then convert
    auto flush = [&](const std::string &tok) {
        size_t k = 0;
        while (k < tok.size()) {
            unsigned char c = unsigned(tok[k]);
            if (c < 0x80) {
                out.push_back(char(c));
                ++k;
                continue;
            }
            if (k + 1 < tok.size()) {
                auto it = u2b.find(tok.substr(k, 2));
                if (it != u2b.end()) {
                    out.push_back(char(it->second));
                    k += 2;
                    continue;
                }
            }
            ++k;  // skip unmapped (multi-byte literal UTF-8 in the vocab entry)
            out.push_back(char(c));
        }
    };
    for (int id : ids) {
        if (id < 0 || size_t(id) >= id_to_token_.size() || id_to_token_[size_t(id)].empty()) continue;
        flush(id_to_token_[size_t(id)]);
    }
    return out;
}

}  // namespace behemoth
