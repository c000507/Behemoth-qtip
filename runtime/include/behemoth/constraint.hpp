// Structured outputs: per-request token constraints applied before sampling.
//
// Pieces: every token id has the exact bytes the server streams for it
// (Tokenizer::piece semantics: SentencePiece "▁" -> space, <0xNN> byte
// tokens -> one raw byte, special tokens -> nothing, the very first nonempty
// piece of a completion loses one leading space). A Vocabulary is built once
// per server from those pieces and holds a byte trie for mask computation.
//
// Gate: generated text is split by ThinkSplitter into an optional leading
// <think>...</think> block (unconstrained reasoning) and the answer. Up to
// `max_lead_whitespace` whitespace bytes before the answer (and before the
// <think> tag) are dropped. The answer is constrained by a GBNF grammar; in
// JSON mode the answer ends exactly when the root value is complete (EOS is
// then the only allowed token). Tags are not recognized inside the answer, so
// a JSON string value may contain "<think>" literally.
#pragma once

#include "behemoth/gbnf.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace behemoth {

inline constexpr int vocabulary_size = 32768;
inline constexpr int eos_token = 2;
inline constexpr size_t mask_words = vocabulary_size / 64;
using TokenMask = std::array<uint64_t, mask_words>;

// Engine-facing interface. All calls for one request come from one thread at a
// time (the engine's sampler job for that request); implementations need no locks.
class TokenConstraint {
  public:
    virtual ~TokenConstraint() = default;
    // Fast path: is `token` allowed now? (No state change.) Used for greedy
    // sampling and inside reasoning, where the unconstrained choice is usually allowed.
    virtual bool accepts(int token) = 0;
    // When true, the engine may sample unconstrained first and only fall back to
    // the mask when accepts() rejects the choice. Must give the same token as
    // masking first: true for greedy decoding; also used inside reasoning.
    virtual bool accept_first(bool greedy) const = 0;
    // Allowed set for the next token. `count` = number of allowed ids; `single` is
    // the allowed id when count == 1 (forced token, e.g. EOS when the JSON is done).
    struct Allowed {
        const TokenMask *mask = nullptr;
        int count = 0;
        int single = -1;
    };
    virtual Allowed allowed() = 0;
    // Commit the sampled token.
    virtual void advance(int token) = 0;
};

// Byte pieces of the vocabulary plus a byte trie over the non-first forms.
class Vocabulary {
  public:
    struct Token {
        std::string first, rest; // piece when it is / is not the first nonempty piece
        bool first_after = false; // value of `first` after emitting at first position
        bool special = false;     // control token: emits nothing, never part of an answer
    };
    explicit Vocabulary(std::vector<Token> tokens);
    const Token &token(int id) const {
        return tokens_[static_cast<size_t>(id)];
    }
    struct Node {
        int32_t first_child = -1, next_sibling = -1, token = -1; // token: first id ending here
        uint8_t byte = 0;
    };
    // Further ids with the same piece as `id` (byte-fallback <0x41> and "A", ...), -1 ends.
    int32_t same_piece(int32_t id) const {
        return same_[static_cast<size_t>(id)];
    }
    const std::vector<Node> &nodes() const {
        return nodes_;
    }
    // Ids whose rest piece contains '>' (only these can complete "</think>").
    const std::vector<int> &close_candidates() const {
        return close_candidates_;
    }
    size_t size() const {
        return tokens_.size();
    }

  private:
    std::vector<Token> tokens_;
    std::vector<Node> nodes_;
    std::vector<int32_t> same_;
    std::vector<int> close_candidates_;
};

// Splits a completion into one optional leading think block and the answer.
struct ThinkSplitter {
    enum Phase : uint8_t { Lead, Think, AnswerLead, Answer };
    static constexpr size_t max_lead_whitespace = 8;
    Phase phase = Lead;
    uint8_t depth = 0;      // think nesting depth (Think phase)
    uint8_t whitespace = 0; // whitespace bytes dropped in Lead/AnswerLead
    std::string held;       // Lead: partial "<think>"; Think: partial tag suffix
    explicit ThinkSplitter(bool allow_think = true) : phase(allow_think ? Lead : AnswerLead) {}
    // Feeds bytes; appends reasoning/answer bytes to the outputs. `finish`
    // flushes held bytes. Returns false if the text cannot be produced under the
    // gate (more than max_lead_whitespace leading whitespace bytes).
    bool feed(std::string_view text, std::string *reasoning, std::string *answer, bool finish = false);
};

// Compiled grammar plus memoized matcher and mask cache. Not thread-safe; a
// GrammarCache hands each request its own instance.
struct GrammarEngine {
    std::string text;
    grammar::Grammar grammar;
    grammar::Matcher matcher;
    bool json_mode;
    struct Key {
        int sid;
        uint32_t value;
        int remain, total;
        bool operator==(const Key &) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key &k) const {
            return (static_cast<size_t>(static_cast<uint32_t>(k.sid)) * 0x9E3779B97F4A7C15ull) ^
                   (static_cast<size_t>(k.value) << 3) ^ static_cast<size_t>(k.remain) ^
                   (static_cast<size_t>(k.total) << 8);
        }
    };
    struct CachedMask {
        TokenMask mask;
        int count, single;
    };
    std::unordered_map<Key, CachedMask, KeyHash> masks;
    GrammarEngine(std::string gbnf, bool json);
};

// Reuses compiled grammars (and their warm matcher/mask caches) across
// requests with the same grammar text. Thread-safe.
class GrammarCache {
    std::mutex mutex_;
    std::unordered_map<std::string, std::vector<std::unique_ptr<GrammarEngine>>> free_;
    std::vector<std::string> order_; // LRU of grammar texts
    size_t max_grammars_;

  public:
    explicit GrammarCache(size_t max_grammars = 16) : max_grammars_(max_grammars) {}
    std::unique_ptr<GrammarEngine> acquire(const std::string &gbnf, bool json);
    void release(std::unique_ptr<GrammarEngine>);
    // Validates and compiles; throws grammar::ParseError (-> HTTP 400).
    static void check(const std::string &gbnf);
};

struct ConstraintStats {
    uint64_t steps = 0, mask_builds = 0, mask_hits = 0, fallbacks = 0, scans = 0;
    double mask_build_ms = 0;
};

class StructuredConstraint final : public TokenConstraint {
  public:
    StructuredConstraint(const Vocabulary &, std::unique_ptr<GrammarEngine>, bool allow_think,
                         GrammarCache *cache = nullptr);
    ~StructuredConstraint() override;
    bool accepts(int token) override;
    bool accept_first(bool greedy) const override;
    Allowed allowed() override;
    void advance(int token) override;

    ThinkSplitter::Phase phase() const {
        return cur_.split.phase;
    }
    bool complete() const; // answer is a full match
    bool dead() const {
        return dead_;
    }
    const ConstraintStats &stats() const {
        return stats_;
    }

  private:
    struct State {
        ThinkSplitter split;
        int sid;
        grammar::PartialUtf8 partial;
        bool first;
    };
    // Applies one token to a state copy; false if it is not allowed.
    bool step(State &, int token);
    bool step_answer_bytes(State &, std::string_view bytes);
    bool accepting(int sid, const grammar::PartialUtf8 &) const;
    void scan_all(TokenMask &); // exact per-token check (Lead/AnswerLead/first piece/think)
    void walk(int32_t node, int sid, grammar::PartialUtf8, TokenMask &);
    const Vocabulary &vocab_;
    std::unique_ptr<GrammarEngine> engine_;
    GrammarCache *cache_;
    State cur_;
    bool dead_ = false;
    TokenMask scratch_{};
    ConstraintStats stats_;
};

} // namespace behemoth
