#pragma once
// GPT2-style byte-level BPE tokenizer for Qwen3.5 (vocab.json + merges.txt + tokenizer_config.json
// added tokens). ASCII pre-tokenisation is exact; non-ASCII letters are grouped with letters by
// their UTF-8 bytes (matches HF for Latin diacritics; CJK spacing may differ — eval tools use raw
// token ids anyway).
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace behemoth {

class QwenTokenizer {
  public:
    explicit QwenTokenizer(const std::filesystem::path &dir);
    // splits on special tokens (e.g. <|im_end|>) unless allowed is given
    std::vector<int> encode(const std::string &text, bool add_special = false) const;
    std::string decode(const std::vector<int> &ids) const;
    int eos() const { return eos_; }
    int im_end() const { return im_end_; }
    int im_start() const { return im_start_; }
    int vocab_size() const { return int(vocab_.size()); }

  private:
    std::map<std::string, int> vocab_;
    std::map<std::pair<std::string, std::string>, int> ranks_;
    std::map<std::string, int> special_;
    int eos_ = 248044, im_start_ = 248045, im_end_ = 248046;
    std::vector<std::string> id_to_token_;

    std::vector<std::string> pre_split(const std::string &text) const;
    std::vector<std::string> bpe(const std::string &word) const;
};

}  // namespace behemoth
