// GBNF grammars (llama.cpp-compatible syntax) and a pushdown matcher over
// Unicode codepoints. This is the constrained-decoding core: a grammar is
// compiled to flat rules, the matcher keeps the set of live parse stacks and
// advances it one codepoint at a time. Token-level masking lives in
// token_mask.h; JSON-schema compilation to GBNF in json_schema.h.
//
// Syntax (subset of llama.cpp grammars/README.md):
//   root ::= "{" ws pair ("," ws pair)* "}"      # comments
//   name ::= [a-z0-9_-]+ | "lit\n\x41A" | . | (group) | x? x* x+ x{2} x{2,} x{2,5}
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace behemoth::grammar {

enum class ElemType : uint8_t {
    End,           // end of a rule's alternate list
    Alt,           // separates alternates within a rule
    RuleRef,       // value = rule index
    Char,          // value = codepoint (start of a positive class)
    CharNot,       // value = codepoint (start of a negated class)
    CharRngUpper,  // value = upper bound; modifies the preceding Char* elem
    CharAlt,       // value = codepoint; another member of the same class
    CharAny,       // '.'
};

struct Elem {
    ElemType type;
    uint32_t value;
};

struct Grammar {
    std::vector<std::vector<Elem>> rules;  // each: alt elems ... Alt alt elems ... End
    std::vector<std::string> names;        // rule index → symbol name (for debugging)
    uint32_t root = 0;
};

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Parses GBNF text. Throws ParseError with a message suitable for a 400.
// Besides syntax, rejects left recursion (including through nullable
// prefixes) and rule-reference chains longer than kMaxRuleChain, so that
// the matcher's expansion depth is bounded for every grammar this accepts.
constexpr int kMaxRuleChain = 2048;
Grammar ParseGbnf(const std::string& text);

// Partial UTF-8 sequence carried across token boundaries (byte-level BPE can
// split a codepoint across tokens).
struct PartialUtf8 {
    uint32_t value = 0;
    int n_remain = 0;  // 0 = at a codepoint boundary
    int n_total = 0;   // sequence length (2..4) while n_remain > 0
    bool operator==(const PartialUtf8& o) const {
        return value == o.value && n_remain == o.n_remain && n_total == o.n_total;
    }
};

// One live parse stack: positions into Grammar::rules, innermost last. The
// top is always a terminal (Char/CharNot/CharAny) or the stack is empty
// (= the grammar has accepted the input so far).
using Stack = std::vector<const Elem*>;

// The matcher proper. States are interned so copying/hashing a state is one
// integer; transitions on a complete codepoint are memoized.
class Matcher {
public:
    explicit Matcher(const Grammar& g);

    int initial_state() const { return init_; }

    // Advance state `sid` over codepoint `cp`. Returns -1 when no stack
    // survives (dead end).
    int Step(int sid, uint32_t cp);

    // Advance over raw bytes, carrying a partial UTF-8 prefix. Returns false
    // if a complete codepoint fails to match or a trailing partial sequence
    // cannot be completed by any accepted codepoint.
    bool StepBytes(int& sid, PartialUtf8& partial, const char* bytes, size_t n);

    // True if some stack is empty (the input so far is a complete match).
    bool Accepting(int sid) const;

    // Could some codepoint in [lo, hi] be matched from this state?
    bool MayMatchRange(int sid, uint32_t lo, uint32_t hi) const;

    // Convenience: does the grammar accept exactly this UTF-8 string?
    bool Matches(const std::string& s);

    size_t num_states() const { return states_.size(); }

private:
    struct StateKey {
        std::vector<Stack> stacks;  // sorted, unique
        bool operator==(const StateKey& o) const { return stacks == o.stacks; }
    };
    struct StateKeyHash {
        size_t operator()(const StateKey& k) const;
    };
    struct TransKey {
        int sid;
        uint32_t cp;
        bool operator==(const TransKey& o) const { return sid == o.sid && cp == o.cp; }
    };
    struct TransKeyHash {
        size_t operator()(const TransKey& k) const {
            return static_cast<size_t>(k.sid) * 0x9E3779B97F4A7C15ull ^ k.cp;
        }
    };

    static constexpr int kMaxExpandDepth = 4096;
    int Intern(std::vector<Stack>&& stacks);
    void AdvanceStack(Stack stack, std::vector<Stack>& out, int depth = 0) const;

    const Grammar& g_;
    std::vector<StateKey> states_;
    std::unordered_map<StateKey, int, StateKeyHash> state_ids_;
    std::unordered_map<TransKey, int, TransKeyHash> trans_;
    int init_ = 0;
};

// Range of codepoints a partial UTF-8 sequence could still complete to. The
// low end is clamped to the sequence length's minimum (0x80/0x800/0x10000):
// a lead byte with zero value bits (C0, E0, F0) would otherwise cover ASCII.
void PartialRange(const PartialUtf8& p, uint32_t& lo, uint32_t& hi);

}  // namespace behemoth::grammar
