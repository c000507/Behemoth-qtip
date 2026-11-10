#include "behemoth/gbnf.hpp"

#include <algorithm>
#include <cstring>
#include <map>

namespace behemoth::grammar {

// ---------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------
static uint32_t DecodeCp(const char* s, size_t n, size_t& len) {
    const uint8_t b = static_cast<uint8_t>(s[0]);
    auto cont = [&](size_t k) { return k < n && (static_cast<uint8_t>(s[k]) & 0xC0) == 0x80; };
    if (b < 0x80) { len = 1; return b; }
    if ((b & 0xE0) == 0xC0 && cont(1)) {
        len = 2;
        return (static_cast<uint32_t>(b & 0x1F) << 6) | (static_cast<uint8_t>(s[1]) & 0x3F);
    }
    if ((b & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        len = 3;
        return (static_cast<uint32_t>(b & 0x0F) << 12) |
               ((static_cast<uint8_t>(s[1]) & 0x3Fu) << 6) | (static_cast<uint8_t>(s[2]) & 0x3Fu);
    }
    if ((b & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        len = 4;
        return (static_cast<uint32_t>(b & 0x07) << 18) |
               ((static_cast<uint8_t>(s[1]) & 0x3Fu) << 12) |
               ((static_cast<uint8_t>(s[2]) & 0x3Fu) << 6) | (static_cast<uint8_t>(s[3]) & 0x3Fu);
    }
    len = 1;
    return b;  // invalid: treat as Latin-1 so parsing never throws on bytes
}

void PartialRange(const PartialUtf8& p, uint32_t& lo, uint32_t& hi) {
    const int bits = 6 * p.n_remain;
    lo = p.value << bits;
    hi = lo | ((1u << bits) - 1);
    static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
    const uint32_t floor = p.n_total >= 2 && p.n_total <= 4 ? kMin[p.n_total] : 0;
    if (lo < floor) lo = floor;
    if (hi > 0x10FFFF) hi = 0x10FFFF;
    // UTF-16 surrogates are not scalar values. A 3-byte prefix narrowed to a
    // 64-codepoint block lies wholly inside or outside D800-DFFF.
    if (lo >= 0xD800 && hi <= 0xDFFF) { lo = 1; hi = 0; }
    // lo > hi here means nothing valid completes it (overlong C0/C1/E0 80-9F/
    // F0 80-8F forms, surrogates, > U+10FFFF): MayMatchRange() treats an empty
    // range as no match.
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
namespace {

struct Parser {
    // Bounds: every repetition `x{m,n}` expands to n rules and the matcher's
    // state is per rule, so unbounded counts are a memory DoS (`"x"{1,5e7}`).
    static constexpr int kMaxRepeat = 4096;
    static constexpr size_t kMaxRules = 1 << 17;
    // Parsing a "( ... )" group recurses, so nesting is bounded before the
    // descent: 50k '(' in a 100 KB grammar would overflow the stack.
    static constexpr int kMaxGroupDepth = 256;

    const std::string& src;
    size_t pos = 0;
    int group_depth = 0;  // inside "( ... )" a newline does not end the rule
    std::map<std::string, uint32_t> symbol_ids;
    std::vector<std::vector<Elem>> rules;
    std::vector<std::string> names;

    explicit Parser(const std::string& s) : src(s) {}

    [[noreturn]] void Fail(const std::string& msg) const {
        throw ParseError("grammar parse error at offset " + std::to_string(pos) + ": " + msg);
    }
    bool Eof() const { return pos >= src.size(); }
    char Peek(size_t k = 0) const { return pos + k < src.size() ? src[pos + k] : '\0'; }

    void SkipSpace(bool newline_ok) {
        while (!Eof()) {
            char c = src[pos];
            if (c == ' ' || c == '\t' || (newline_ok && (c == '\n' || c == '\r'))) {
                ++pos;
            } else if (c == '#') {
                while (!Eof() && src[pos] != '\n' && src[pos] != '\r') ++pos;
            } else {
                break;
            }
        }
    }

    static bool IsWordChar(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '-' || c == '_';
    }

    uint32_t GetSymbolId(const std::string& name) {
        auto it = symbol_ids.find(name);
        if (it != symbol_ids.end()) return it->second;
        if (rules.size() >= kMaxRules)
            Fail("grammar has more than " + std::to_string(kMaxRules) + " rules");
        uint32_t id = static_cast<uint32_t>(rules.size());
        symbol_ids[name] = id;
        rules.emplace_back();
        names.push_back(name);
        return id;
    }
    uint32_t GenSymbolId(const std::string& base) {
        return GetSymbolId(base + "_" + std::to_string(rules.size()));
    }
    void AddRule(uint32_t id, std::vector<Elem> rule) {
        if (!rules[id].empty()) Fail("duplicate definition of rule '" + names[id] + "'");
        rules[id] = std::move(rule);
    }

    uint32_t ParseChar() {
        if (Eof()) Fail("unexpected end of input in character");
        if (src[pos] == '\\') {
            ++pos;
            if (Eof()) Fail("unexpected end of escape");
            char e = src[pos++];
            auto hex = [&](int n) {
                uint32_t v = 0;
                for (int i = 0; i < n; ++i) {
                    if (Eof()) Fail("truncated hex escape");
                    char c = src[pos++];
                    int d = (c >= '0' && c <= '9')   ? c - '0'
                            : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                            : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                                                     : -1;
                    if (d < 0) Fail("bad hex digit in escape");
                    v = v * 16 + static_cast<uint32_t>(d);
                }
                return v;
            };
            switch (e) {
            case 'x': return hex(2);
            case 'u': return hex(4);
            case 'U': return hex(8);
            case 't': return '\t';
            case 'r': return '\r';
            case 'n': return '\n';
            case '\\': case '"': case '[': case ']': case '-': case '.': case '/':
                return static_cast<uint32_t>(static_cast<uint8_t>(e));
            default: Fail(std::string("unknown escape '\\") + e + "'");
            }
        }
        size_t len;
        uint32_t cp = DecodeCp(src.data() + pos, src.size() - pos, len);
        pos += len;
        return cp;
    }

    void ParseSequence(const std::string& rule_name, std::vector<Elem>& out);

    void ParseAlternates(const std::string& rule_name, uint32_t rule_id) {
        std::vector<Elem> rule;
        ParseSequence(rule_name, rule);
        while (Peek() == '|') {
            rule.push_back({ElemType::Alt, 0});
            ++pos;
            SkipSpace(true);
            ParseSequence(rule_name, rule);
        }
        rule.push_back({ElemType::End, 0});
        AddRule(rule_id, std::move(rule));
    }

    void Parse() {
        SkipSpace(true);
        while (!Eof()) {
            size_t start = pos;
            while (IsWordChar(Peek())) ++pos;
            if (pos == start) Fail("expected rule name");
            std::string name = src.substr(start, pos - start);
            SkipSpace(false);
            if (!(Peek() == ':' && Peek(1) == ':' && Peek(2) == '=')) Fail("expected ::=");
            pos += 3;
            SkipSpace(true);
            uint32_t id = GetSymbolId(name);
            ParseAlternates(name, id);
            SkipSpace(true);
        }
        for (size_t i = 0; i < rules.size(); ++i)
            if (rules[i].empty()) Fail("rule '" + names[i] + "' is referenced but never defined");
        if (!symbol_ids.count("root")) Fail("grammar has no 'root' rule");
    }
};

void Parser::ParseSequence(const std::string& rule_name, std::vector<Elem>& out) {
    size_t last_sym_start = out.size();
    auto handle_repetition = [&](int min_times, int max_times) {
        // Everything from last_sym_start to the end is the repeated item.
        if (last_sym_start == out.size()) Fail("repetition with nothing to repeat");
        std::vector<Elem> item(out.begin() + static_cast<long>(last_sym_start), out.end());
        out.resize(last_sym_start);
        // Fold the item into a single RuleRef unless it already is one.
        Elem item_ref;
        if (item.size() == 1 && item[0].type == ElemType::RuleRef) {
            item_ref = item[0];
        } else {
            uint32_t id = GenSymbolId(rule_name);
            std::vector<Elem> r = item;
            r.push_back({ElemType::End, 0});
            AddRule(id, std::move(r));
            item_ref = {ElemType::RuleRef, id};
        }
        for (int i = 0; i < min_times; ++i) out.push_back(item_ref);
        if (max_times < 0) {
            // x* → rep ::= x rep | ε ; x+ → min copies then rep*
            uint32_t rep = GenSymbolId(rule_name);
            AddRule(rep, {item_ref, {ElemType::RuleRef, rep}, {ElemType::Alt, 0}, {ElemType::End, 0}});
            out.push_back({ElemType::RuleRef, rep});
        } else if (max_times > min_times) {
            // nested optionals: opt_k ::= x opt_{k+1} | ε
            uint32_t prev = 0;
            bool have_prev = false;
            for (int k = 0; k < max_times - min_times; ++k) {
                uint32_t id = GenSymbolId(rule_name);
                std::vector<Elem> r = {item_ref};
                if (have_prev) r.push_back({ElemType::RuleRef, prev});
                r.push_back({ElemType::Alt, 0});
                r.push_back({ElemType::End, 0});
                AddRule(id, std::move(r));
                prev = id;
                have_prev = true;
            }
            out.push_back({ElemType::RuleRef, prev});
        }
    };

    while (!Eof()) {
        char c = Peek();
        if (c == '"') {
            ++pos;
            last_sym_start = out.size();
            while (Peek() != '"') {
                if (Eof()) Fail("unterminated string literal");
                out.push_back({ElemType::Char, ParseChar()});
            }
            ++pos;
            SkipSpace(group_depth > 0);
        } else if (c == '[') {
            ++pos;
            ElemType start_type = ElemType::Char;
            if (Peek() == '^') { ++pos; start_type = ElemType::CharNot; }
            last_sym_start = out.size();
            bool first = true;
            while (Peek() != ']') {
                if (Eof()) Fail("unterminated character class");
                uint32_t cp = ParseChar();
                out.push_back({first ? start_type : ElemType::CharAlt, cp});
                first = false;
                if (Peek() == '-' && Peek(1) != ']') {
                    ++pos;
                    out.push_back({ElemType::CharRngUpper, ParseChar()});
                }
            }
            if (first) Fail("empty character class");
            ++pos;
            SkipSpace(group_depth > 0);
        } else if (IsWordChar(c)) {
            size_t start = pos;
            while (IsWordChar(Peek())) ++pos;
            std::string name = src.substr(start, pos - start);
            SkipSpace(group_depth > 0);
            last_sym_start = out.size();
            out.push_back({ElemType::RuleRef, GetSymbolId(name)});
        } else if (c == '(') {
            ++pos;
            SkipSpace(true);
            if (group_depth >= kMaxGroupDepth)
                Fail("groups nested deeper than " + std::to_string(kMaxGroupDepth));
            uint32_t id = GenSymbolId(rule_name);
            ++group_depth;
            ParseAlternates(rule_name, id);
            --group_depth;
            if (Peek() != ')') Fail("expected ')'");
            ++pos;
            SkipSpace(group_depth > 0);
            last_sym_start = out.size();
            out.push_back({ElemType::RuleRef, id});
        } else if (c == '.') {
            ++pos;
            SkipSpace(group_depth > 0);
            last_sym_start = out.size();
            out.push_back({ElemType::CharAny, 0});
        } else if (c == '*') {
            ++pos; SkipSpace(group_depth > 0); handle_repetition(0, -1);
        } else if (c == '+') {
            ++pos; SkipSpace(group_depth > 0); handle_repetition(1, -1);
        } else if (c == '?') {
            ++pos; SkipSpace(group_depth > 0); handle_repetition(0, 1);
        } else if (c == '{') {
            ++pos;
            auto number = [&]() -> int {
                size_t s = pos;
                long v = 0;
                while (Peek() >= '0' && Peek() <= '9') {
                    v = v * 10 + (Peek() - '0');
                    if (v > kMaxRepeat)
                        Fail("repetition count exceeds " + std::to_string(kMaxRepeat));
                    ++pos;
                }
                if (s == pos) Fail("expected number in repetition");
                return static_cast<int>(v);
            };
            int mn = number(), mx = mn;
            if (Peek() == ',') {
                ++pos;
                mx = (Peek() >= '0' && Peek() <= '9') ? number() : -1;
            }
            if (Peek() != '}') Fail("expected '}'");
            ++pos;
            SkipSpace(group_depth > 0);
            if (mx >= 0 && mx < mn) Fail("repetition max < min");
            handle_repetition(mn, mx);
        } else {
            break;  // '|', ')', newline (end of rule), ...
        }
    }
}

}  // namespace

Grammar ParseGbnf(const std::string& text) {
    Parser p(text);
    p.Parse();
    Grammar g;
    g.rules = std::move(p.rules);
    g.names = std::move(p.names);
    g.root = p.symbol_ids.at("root");
    // Reject left recursion: AdvanceStack expands rule references at the
    // top of a stack until it reaches a terminal, so a rule that can reach
    // itself without consuming input recurses forever (stack overflow →
    // SIGSEGV). "Without consuming input" includes hidden left recursion
    // through nullable prefixes (`root ::= ("a")* root | "b"`) and stars over
    // nullable bodies (`("a"?)*` expands to `rep ::= opt rep | ε`), so first
    // compute the nullable set, then look for cycles in the graph whose
    // edges go from a rule to every reference preceded only by nullables.
    const size_t n = g.rules.size();
    std::vector<char> nullable(n, 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t r = 0; r < n; ++r) {
            if (nullable[r]) continue;
            bool alt_nullable = true;
            for (const Elem& e : g.rules[r]) {
                if (e.type == ElemType::Alt || e.type == ElemType::End) {
                    if (alt_nullable) { nullable[r] = 1; changed = true; break; }
                    alt_nullable = true;
                    continue;
                }
                if (e.type == ElemType::RuleRef) {
                    if (!nullable[e.value]) alt_nullable = false;
                } else if (e.type == ElemType::CharRngUpper || e.type == ElemType::CharAlt) {
                    // part of the preceding class
                } else {
                    alt_nullable = false;  // a terminal consumes a codepoint
                }
            }
        }
    }
    std::vector<std::vector<uint32_t>> edges(n);
    for (size_t r = 0; r < n; ++r) {
        bool prefix_nullable = true;
        for (const Elem& e : g.rules[r]) {
            if (e.type == ElemType::Alt) { prefix_nullable = true; continue; }
            if (e.type == ElemType::End) break;
            if (!prefix_nullable) continue;
            if (e.type == ElemType::RuleRef) {
                edges[r].push_back(e.value);
                if (!nullable[e.value]) prefix_nullable = false;
            } else if (e.type != ElemType::CharRngUpper && e.type != ElemType::CharAlt) {
                prefix_nullable = false;
            }
        }
    }
    // DFS: a back edge is left recursion; on the way out of each node record
    // the longest edge path below it (the graph is a DAG once no back edge
    // exists). AdvanceStack recurses once per edge it follows, so this
    // longest path bounds its depth: reject grammars whose chains of rule
    // references exceed kMaxRuleChain, so Matcher::kMaxExpandDepth can never
    // fire at generation time on a grammar that passed here.
    std::vector<int> color(n, 0);  // 0 unvisited, 1 on stack, 2 done
    std::vector<int> longest(n, 0);
    for (uint32_t r = 0; r < n; ++r) {
        if (color[r]) continue;
        std::vector<std::pair<uint32_t, size_t>> st;
        st.push_back({r, 0});
        color[r] = 1;
        while (!st.empty()) {
            auto [cur, i] = st.back();
            if (i < edges[cur].size()) {
                st.back().second = i + 1;
                uint32_t nx = edges[cur][i];
                if (color[nx] == 1)
                    throw ParseError("grammar is left-recursive at rule '" + g.names[nx] +
                                     "' (a rule may not reach itself without consuming input)");
                if (color[nx] == 0) { color[nx] = 1; st.push_back({nx, 0}); }
                else longest[cur] = std::max(longest[cur], longest[nx] + 1);
            } else {
                color[cur] = 2;
                st.pop_back();
                if (!st.empty()) {
                    uint32_t parent = st.back().first;
                    longest[parent] = std::max(longest[parent], longest[cur] + 1);
                }
                if (longest[cur] > kMaxRuleChain)
                    throw ParseError("grammar nests rule references deeper than " +
                                     std::to_string(kMaxRuleChain) + " at rule '" + g.names[cur] + "'");
            }
        }
    }
    return g;
}

// ---------------------------------------------------------------------------
// Matcher
// ---------------------------------------------------------------------------
static bool IsEndOfSeq(const Elem* e) { return e->type == ElemType::End || e->type == ElemType::Alt; }

// Match one codepoint against the class starting at `pos`; returns the
// element after the class.
static std::pair<bool, const Elem*> MatchChar(const Elem* pos, uint32_t cp) {
    if (pos->type == ElemType::CharAny) return {true, pos + 1};
    const bool positive = pos->type == ElemType::Char;
    bool found = false;
    do {
        if (pos[1].type == ElemType::CharRngUpper) {
            found = found || (pos->value <= cp && cp <= pos[1].value);
            pos += 2;
        } else {
            found = found || pos->value == cp;
            pos += 1;
        }
    } while (pos->type == ElemType::CharAlt);
    return {found == positive, pos};
}

// Could any codepoint in [lo, hi] match the class at `pos`? Exact for
// positive classes; for negated classes, true unless both ends are
// excluded (exact for the ASCII-only exclusions JSON grammars use).
static bool MayMatchRange(const Elem* pos, uint32_t lo, uint32_t hi) {
    if (pos->type == ElemType::CharAny) return true;
    if (pos->type == ElemType::CharNot)
        return MatchChar(pos, lo).first || MatchChar(pos, hi).first;
    do {
        if (pos[1].type == ElemType::CharRngUpper) {
            if (pos->value <= hi && lo <= pos[1].value) return true;
            pos += 2;
        } else {
            if (lo <= pos->value && pos->value <= hi) return true;
            pos += 1;
        }
    } while (pos->type == ElemType::CharAlt);
    return false;
}

size_t Matcher::StateKeyHash::operator()(const StateKey& k) const {
    size_t h = k.stacks.size();
    for (const Stack& s : k.stacks) {
        h ^= s.size() + 0x9E3779B9 + (h << 6) + (h >> 2);
        for (const Elem* e : s)
            h ^= reinterpret_cast<uintptr_t>(e) + 0x9E3779B9 + (h << 6) + (h >> 2);
    }
    return h;
}

Matcher::Matcher(const Grammar& g) : g_(g) {
    std::vector<Stack> stacks;
    const Elem* pos = g_.rules[g_.root].data();
    while (true) {
        Stack s;
        if (!IsEndOfSeq(pos)) s.push_back(pos);
        AdvanceStack(std::move(s), stacks);
        while (!IsEndOfSeq(pos)) ++pos;
        if (pos->type == ElemType::Alt) ++pos;
        else break;
    }
    init_ = Intern(std::move(stacks));
}

void Matcher::AdvanceStack(Stack stack, std::vector<Stack>& out, int depth) const {
    if (stack.empty()) {
        out.push_back(std::move(stack));
        return;
    }
    // ParseGbnf rejects every grammar that can recurse here without
    // consuming input; this is the belt to that suspender, so a missed
    // case surfaces as an error instead of a stack overflow.
    if (depth > kMaxExpandDepth)
        throw ParseError("grammar expansion too deep (left recursion or nesting beyond " +
                         std::to_string(kMaxExpandDepth) + ")");
    const Elem* pos = stack.back();
    switch (pos->type) {
    case ElemType::RuleRef: {
        const Elem* sub = g_.rules[pos->value].data();
        while (true) {
            Stack ns(stack.begin(), stack.end() - 1);
            if (!IsEndOfSeq(pos + 1)) ns.push_back(pos + 1);
            if (!IsEndOfSeq(sub)) ns.push_back(sub);
            AdvanceStack(std::move(ns), out, depth + 1);
            while (!IsEndOfSeq(sub)) ++sub;
            if (sub->type == ElemType::Alt) ++sub;
            else break;
        }
        break;
    }
    case ElemType::Char:
    case ElemType::CharNot:
    case ElemType::CharAny:
        out.push_back(std::move(stack));
        break;
    default:
        throw std::logic_error("grammar: unexpected element at stack top");
    }
}

int Matcher::Intern(std::vector<Stack>&& stacks) {
    std::sort(stacks.begin(), stacks.end());
    stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());
    StateKey key{std::move(stacks)};
    auto it = state_ids_.find(key);
    if (it != state_ids_.end()) return it->second;
    int id = static_cast<int>(states_.size());
    states_.push_back(key);
    state_ids_.emplace(std::move(key), id);
    return id;
}

int Matcher::Step(int sid, uint32_t cp) {
    if (sid < 0) return -1;
    TransKey tk{sid, cp};
    auto it = trans_.find(tk);
    if (it != trans_.end()) return it->second;
    std::vector<Stack> out;
    // states_ may reallocate inside Intern; copy the key out first.
    const std::vector<Stack> stacks = states_[static_cast<size_t>(sid)].stacks;
    for (const Stack& s : stacks) {
        if (s.empty()) continue;
        auto [ok, next] = MatchChar(s.back(), cp);
        if (!ok) continue;
        Stack ns(s.begin(), s.end() - 1);
        if (!IsEndOfSeq(next)) ns.push_back(next);
        AdvanceStack(std::move(ns), out);
    }
    int res = out.empty() ? -1 : Intern(std::move(out));
    trans_.emplace(tk, res);
    return res;
}

bool Matcher::Accepting(int sid) const {
    if (sid < 0) return false;
    for (const Stack& s : states_[static_cast<size_t>(sid)].stacks)
        if (s.empty()) return true;
    return false;
}

bool Matcher::MayMatchRange(int sid, uint32_t lo, uint32_t hi) const {
    if (sid < 0 || lo > hi) return false;
    for (const Stack& s : states_[static_cast<size_t>(sid)].stacks)
        if (!s.empty() && grammar::MayMatchRange(s.back(), lo, hi)) return true;
    return false;
}

bool Matcher::StepBytes(int& sid, PartialUtf8& partial, const char* bytes, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint8_t b = static_cast<uint8_t>(bytes[i]);
        if (partial.n_remain > 0) {
            if ((b & 0xC0) != 0x80) return false;  // broken sequence
            partial.value = (partial.value << 6) | (b & 0x3F);
            if (--partial.n_remain == 0) {
                sid = Step(sid, partial.value);
                partial = {};
                if (sid < 0) return false;
            } else {
                uint32_t lo, hi;
                PartialRange(partial, lo, hi);
                if (!MayMatchRange(sid, lo, hi)) return false;
            }
            continue;
        }
        if (b < 0x80) {
            sid = Step(sid, b);
            if (sid < 0) return false;
        } else if ((b & 0xE0) == 0xC0) {
            partial = {b & 0x1Fu, 1, 2};
        } else if ((b & 0xF0) == 0xE0) {
            partial = {b & 0x0Fu, 2, 3};
        } else if ((b & 0xF8) == 0xF0 && b <= 0xF4) {
            partial = {b & 0x07u, 3, 4};
        } else {
            return false;  // stray continuation / invalid lead byte (F5+)
        }
        if (partial.n_remain > 0) {
            uint32_t lo, hi;
            PartialRange(partial, lo, hi);
            if (!MayMatchRange(sid, lo, hi)) return false;
        }
    }
    return true;
}

bool Matcher::Matches(const std::string& s) {
    int sid = init_;
    PartialUtf8 p;
    if (!StepBytes(sid, p, s.data(), s.size())) return false;
    return p.n_remain == 0 && Accepting(sid);
}

}  // namespace behemoth::grammar
