// Structured outputs (CPU only): GBNF matcher, JSON-schema compiler, token masks,
// think gate, masked sampling and request parsing. Uses a synthetic 32768-token
// vocabulary with SentencePiece-like pieces (leading-space words, byte tokens,
// duplicates, multi-terminal and partial-UTF-8 pieces).
#include "behemoth/constraint.hpp"
#include "behemoth/gbnf.hpp"
#include "behemoth/json_schema.hpp"
#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"
#include "structured_support.hpp"
#include <chrono>
#include <functional>
#include <iostream>

using namespace behemoth;
using namespace structured_test;
static int assertions = 0;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        ++assertions;                                                                              \
        if (!(x))                                                                                  \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +      \
                                     ": " #x);                                                     \
    } while (false)
template <class F> static std::string rejects(F f) {
    try {
        f();
    } catch (const std::exception &e) {
        ++assertions;
        return e.what();
    }
    throw std::runtime_error("expected an exception");
}

// ---------------------------------------------------------------------------
// Synthetic vocabulary
// ---------------------------------------------------------------------------
static std::map<std::string, int> piece_ids;
static const Vocabulary &vocab() {
    static const Vocabulary v = [] {
        std::vector<Vocabulary::Token> t(vocabulary_size);
        size_t next = 0;
        auto special = [&](const char *) { t[next++].special = true; };
        special("<unk>");
        special("<s>");
        special("</s>");
        special("[INST]");
        special("[/INST]");
        for (int b = 0; b < 256; ++b) { // byte-fallback tokens
            auto &x = t[next++];
            x.first = x.rest = std::string(1, char(b));
            x.first_after = false;
        }
        auto add = [&](const std::string &rest) {
            if (next >= t.size())
                return;
            auto &x = t[next];
            x.rest = rest;
            x.first = !rest.empty() && rest[0] == ' ' ? rest.substr(1) : rest;
            x.first_after = x.first.empty();
            piece_ids.emplace(rest, int(next));
            ++next;
        };
        for (const char *p :
             {"{", "}", "[", "]", "\"", ":", ",", " {", " \"", "\":", "\",", "\":\"", "\"}", "}}", "}]", "]}",
              "\":{", "\": ", "true", "false", "null", " true", "<", "think", ">", "</", "<think>", "</think>",
              "</think>\n", "</think>\n\n{", "</think>x", "<th", "ink>", "\n", "\n\n", "  ", "    ", " ", "0", "1",
              "12", "123", "-", ".", "5", "e", "\xc3\xa9", "\xe2\x82\xac", "\xf0\x9f\x98\x80", "ab", " ab", "name",
              " name", "\"name", "age", "\"age\":", "x", "hello", " world", "\\", "\\n", "\\u00", "{\"", "\"]", ",\"",
              "2026-", "01", "-02", "T", ":00", "Z", "abc>", "> <", "-->", "e>", "}\n", "} ", "\":1}", "A", "\xe2\x82",
              "\xac\"", ":1}", "\xe2\x96\x81", "red", "green", "\"red\"", " null"})
            add(p);
        // Filler words (with and without a leading space), as a real vocabulary has.
        for (size_t i = 0; next < t.size(); ++i) {
            std::string w;
            for (size_t k = i; ; k /= 26) {
                w.push_back(char('a' + k % 26));
                if (k < 26)
                    break;
            }
            add(i % 2 ? " " + w : w + "s");
        }
        return Vocabulary(std::move(t));
    }();
    return v;
}
static int id(const std::string &piece) {
    if (piece.size() == 1 && !piece_ids.count(piece))
        return 5 + int(uint8_t(piece[0]));
    auto it = piece_ids.find(piece);
    if (it == piece_ids.end())
        throw std::runtime_error("test vocabulary lacks piece [" + piece + "]");
    return it->second;
}
static int byte_id(char c) {
    return 5 + int(uint8_t(c));
}
static std::unique_ptr<StructuredConstraint> make(const std::string &gbnf, bool json, bool think = true,
                                                  GrammarCache *cache = nullptr) {
    return std::make_unique<StructuredConstraint>(vocab(), std::make_unique<GrammarEngine>(gbnf, json), think,
                                                  cache);
}
// Feeds token ids; returns false at the first rejected one.
static bool feed(StructuredConstraint &c, const std::vector<int> &tokens) {
    for (int t : tokens) {
        if (!c.accepts(t))
            return false;
        c.advance(t);
    }
    return true;
}
static std::vector<int> bytes(const std::string &s) {
    std::vector<int> out;
    for (char c : s)
        out.push_back(byte_id(c));
    return out;
}
static bool matches(const std::string &gbnf, const std::string &text) {
    grammar::Grammar g = grammar::ParseGbnf(gbnf);
    grammar::Matcher m(g);
    return m.Matches(text);
}

// ---------------------------------------------------------------------------
static void gbnf_tests() {
    CHECK(matches("root ::= \"a\" [b-d]+ \"e\"?", "abcd"));
    CHECK(matches("root ::= \"a\" [b-d]+ \"e\"?", "abde"));
    CHECK(!matches("root ::= \"a\" [b-d]+ \"e\"?", "a"));
    CHECK(matches("root ::= x{2,3}\nx ::= \"ab\"", "ababab"));
    CHECK(!matches("root ::= x{2,3}\nx ::= \"ab\"", "abababab"));
    CHECK(matches("root ::= [^a]*", "\xe2\x82\xac\xf0\x9f\x98\x80"));
    CHECK(!matches("root ::= [^a]*", "\xe2\x82")); // incomplete UTF-8 never matches
    rejects([] { grammar::ParseGbnf("root ::= root \"a\" | \"b\""); }); // left recursion
    rejects([] { grammar::ParseGbnf("root ::= missing"); });
    rejects([] { grammar::ParseGbnf("start ::= \"a\""); });
    rejects([] { grammar::ParseGbnf("root ::= \"a\"{1,99999}"); });
    rejects([] { GrammarCache::check("root ::= [a-"); });
    GrammarCache::check("root ::= \"ok\"");
}

static void json_object_tests() {
    const std::string g = grammar::json_object_grammar();
    for (const char *ok : {"{}", "{\"a\":1}", "{\"a\": [1, 2.5e3, -0, true, null, \"x\"]}",
                           "{\n  \"a\": {\"b\": {}}\n}", "{\"\\u00e9\\n\":\"\xc3\xa9\xe2\x82\xac\"}"})
        CHECK(matches(g, ok));
    for (const char *bad : {"[]", "{\"a\":1,}", "{a:1}", "{\"a\":01}", "{\"a\":1}x", "\"x\"", "{\"a\":\"\x01\"}",
                            "{\"a\":1.}", " {}", "{\"a\":\"\\q\"}"})
        CHECK(!matches(g, bad));
}

static ojson schema(const char *text) {
    return ojson::parse(text);
}
static bool schema_matches(const ojson &s, const std::string &text) {
    return matches(grammar::json_schema_grammar(s), text);
}

static void json_schema_tests() {
    auto person = schema(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0,"maximum":150},
        "email":{"type":"string","format":"email"},"tags":{"type":"array","items":{"type":"string","enum":["a","b"]},"minItems":1,"maxItems":2}},
        "required":["name","age"],"additionalProperties":false})");
    CHECK(schema_matches(person, R"({"name":"Ann","age":30})"));
    CHECK(schema_matches(person, R"({"name":"Ann","age":0,"tags":["a","b"]})"));
    CHECK(schema_matches(person, R"({"name":"Ann","age":150,"email":"x@y"})"));
    CHECK(!schema_matches(person, R"({"name":"Ann","age":151})"));
    CHECK(!schema_matches(person, R"({"name":"Ann","age":-1})"));
    CHECK(!schema_matches(person, R"({"age":3,"name":"Ann"})")); // declared order
    CHECK(!schema_matches(person, R"({"name":"Ann"})"));
    CHECK(!schema_matches(person, R"({"name":"Ann","age":3,"x":1})"));
    CHECK(!schema_matches(person, R"({"name":"Ann","age":3,"tags":[]})"));
    CHECK(!schema_matches(person, R"({"name":"Ann","age":3,"tags":["a","b","a"]})"));
    CHECK(!schema_matches(person, R"({"name":"Ann","age":3,"tags":["c"]})"));
    CHECK(!schema_matches(person, R"({"name":"Ann","age":3.5})"));
    // Optional property before a required one; extras allowed at the end by default.
    auto open = schema(R"({"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["b"]})");
    CHECK(schema_matches(open, R"({"b":null})"));
    CHECK(schema_matches(open, R"({"a":true,"b":null})"));
    CHECK(schema_matches(open, R"({"b":null,"z":[1]})"));
    CHECK(!schema_matches(open, R"({"a":true})"));
    // anyOf, const, nullable, type lists, $ref/$defs (recursive), nested.
    auto tree = schema(R"({"$defs":{"node":{"type":"object","properties":{"v":{"anyOf":[{"type":"integer"},{"const":"leaf"}]},
        "kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"],"additionalProperties":false}},"$ref":"#/$defs/node"})");
    CHECK(schema_matches(tree, R"({"v":1,"kids":[{"v":"leaf"},{"v":-2,"kids":[]}]})"));
    CHECK(!schema_matches(tree, R"({"v":"other"})"));
    auto misc = schema(R"({"type":"object","properties":{"n":{"type":["number","null"]},"s":{"type":"string","nullable":true,"minLength":2,"maxLength":3},
        "d":{"type":"string","format":"date-time"},"u":{"type":"string","format":"uuid"}},"required":["n","s","d","u"],"additionalProperties":false})");
    CHECK(schema_matches(misc, R"({"n":null,"s":null,"d":"2026-11-01T12:30:00Z","u":"123e4567-e89b-12d3-a456-426614174000"})"));
    CHECK(schema_matches(misc, R"({"n":-1.5e3,"s":"\u00e9x","d":"2026-01-02T00:00:00.5+02:00","u":"123e4567-e89b-12d3-a456-426614174000"})"));
    CHECK(!schema_matches(misc, R"({"n":1,"s":"x","d":"2026-11-01T12:30:00Z","u":"123e4567-e89b-12d3-a456-426614174000"})"));
    CHECK(!schema_matches(misc, R"({"n":1,"s":"xy","d":"2026-13-01T12:30:00Z","u":"123e4567-e89b-12d3-a456-426614174000"})"));
    // Tuples and root scalars.
    auto tuple = schema(R"({"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}],"items":false})");
    CHECK(schema_matches(tuple, R"(["a",1])"));
    CHECK(!schema_matches(tuple, R"(["a",1,2])"));
    CHECK(schema_matches(schema(R"({"enum":["red","green",3,null]})"), "null"));
    CHECK(schema_matches(schema(R"({"type":"integer","exclusiveMinimum":0})"), "7"));
    CHECK(!schema_matches(schema(R"({"type":"integer","exclusiveMinimum":0})"), "0"));
    CHECK(schema_matches(schema("{}"), "[1,{\"a\":null}]"));

    // Integer ranges: exhaustive against the arithmetic definition on windows.
    std::mt19937_64 rng(5);
    std::vector<std::tuple<long long, bool, long long, bool>> ranges = {
        {0, true, 150, true}, {-7, true, 7, true}, {-120, true, -3, true}, {13, true, 0, false},
        {0, false, -45, true}, {99, true, 1000, true}, {-1000, true, -1000, true}, {101, true, 109, true}};
    for (int i = 0; i < 40; ++i) {
        long long a = std::uniform_int_distribution<long long>(-3000, 3000)(rng);
        long long b = a + std::uniform_int_distribution<long long>(0, 4000)(rng);
        ranges.push_back({a, true, b, true});
    }
    for (auto [lo, has_lo, hi, has_hi] : ranges) {
        std::string g = "root ::= " + grammar::integer_range_gbnf(lo, has_lo, hi, has_hi) + "\n";
        grammar::Grammar gr = grammar::ParseGbnf(g);
        grammar::Matcher m(gr);
        for (long long x = -5000; x <= 9000; x += (x > -1200 && x < 1200) ? 1 : 7) {
            bool want = (!has_lo || x >= lo) && (!has_hi || x <= hi);
            CHECK(m.Matches(std::to_string(x)) == want);
        }
        CHECK(!m.Matches("-0") && !m.Matches("007") && !m.Matches(""));
    }
    {
        std::string g = "root ::= " + grammar::integer_range_gbnf(5, true, 0, false) + "\n";
        CHECK(matches(g, "9999999999999999"));
        CHECK(!matches(g, "10000000000000000")); // 16-digit cap
    }
    // Number ranges with integer-valued limits.
    auto in_range = [](const char *s, const char *text) { return schema_matches(schema(s), text); };
    const char *unit = R"({"type":"number","minimum":0,"maximum":1})";
    for (const char *ok : {"0", "0.5", "1", "1.0", "0.999", "1.000"})
        CHECK(in_range(unit, ok));
    for (const char *bad : {"-0.1", "1.01", "2", "1e0", "-1"})
        CHECK(!in_range(unit, bad));
    const char *open_unit = R"({"type":"number","exclusiveMinimum":-2,"exclusiveMaximum":3})";
    for (const char *ok : {"-1.99", "-1", "0", "2.999", "-0.5"})
        CHECK(in_range(open_unit, ok));
    for (const char *bad : {"-2", "-2.0", "3", "3.0", "-2.5"})
        CHECK(!in_range(open_unit, bad));
    CHECK(in_range(R"({"type":"number","minimum":-5})", "123456.25"));
    CHECK(!in_range(R"({"type":"number","minimum":-5})", "-5.5"));
    CHECK(in_range(R"({"type":"number","maximum":-5})", "-5"));
    CHECK(!in_range(R"({"type":"number","maximum":-5})", "-4.5"));
    CHECK(in_range(R"({"type":"integer","minimum":1.5,"maximum":3.5})", "2"));
    CHECK(!in_range(R"({"type":"integer","minimum":1.5,"maximum":3.5})", "1"));
    CHECK(!in_range(R"({"type":"integer","minimum":1.5,"maximum":3.5})", "4"));

    // Rejected features name the keyword.
    for (const char *bad :
         {R"({"type":"string","pattern":"^a"})", R"({"type":"number","multipleOf":2})", R"({"not":{"type":"null"}})",
          R"({"type":"object","minProperties":1})", R"({"type":"array","uniqueItems":true})",
          R"({"type":"object","patternProperties":{}})", R"({"if":{},"then":{}})", R"({"type":"string","foo":1})",
          R"({"type":"object","properties":{"a":{"type":"string"}},"anyOf":[{"required":["a"]}]})",
          R"({"$ref":"#/$defs/x","type":"object","$defs":{"x":{}}})", R"({"$ref":"#/$defs/missing"})",
          R"({"$ref":"https://example.com/s.json"})", R"({"type":"number","minimum":0.5})", R"({"type":"date"})",
          R"({"allOf":[{"type":"string"},{"minLength":1}]})", R"({"type":"integer","minimum":5,"maximum":4})",
          R"({"type":"array","minItems":3,"maxItems":2})", R"({"type":"array","contains":{}})", "false",
          R"({"type":"string","format":"date","maxLength":3})", R"({"minimum":1})", R"({"enum":[]})",
          R"({"type":"string","maxLength":100000})"}) {
        auto msg = rejects([&] { grammar::json_schema_grammar(schema(bad)); });
        CHECK(!msg.empty());
    }
    CHECK(rejects([] { grammar::json_schema_grammar(schema(R"({"type":"string","pattern":"a"})")); })
              .find("pattern") != std::string::npos);
    // Annotations are accepted.
    grammar::json_schema_grammar(schema(R"({"title":"t","description":"d","$schema":"x","type":"object","examples":[{}],
        "properties":{"a":{"type":"string","default":"x","format":"uri","deprecated":false}}})"));
    // Deep nesting bound.
    std::string deep = "{\"type\":\"string\"}";
    for (int i = 0; i < 80; ++i)
        deep = "{\"type\":\"array\",\"items\":" + deep + "}";
    rejects([&] { grammar::json_schema_grammar(ojson::parse(deep)); });
}

static void splitter_tests() {
    // Every split point of each text gives the same reasoning/answer.
    struct Case {
        std::string text, reasoning, answer;
        bool think;
    };
    for (const auto &c : std::vector<Case>{
             {"<think>r</think>{}", "r", "{}", true},
             {"\n <think>\nr<x>\n</think>\n\n{\"a\":\"<think>\"}", "\nr<x>\n", "{\"a\":\"<think>\"}", true},
             {"<think>a<think>b</think>c</think> x", "abc", "x", true}, // nested: inner tags dropped
             {"{\"a\":1}", "", "{\"a\":1}", true},
             {"  <thx", "", "<thx", true},
             {"<think>unfinished</thi", "unfinished</thi", "", true},
             {"<think>{}", "", "<think>{}", false},
             {"\n\n[1]", "", "[1]", false}}) {
        for (size_t cut = 0; cut <= c.text.size(); ++cut) {
            ThinkSplitter s(c.think);
            std::string r, a;
            CHECK(s.feed(std::string_view(c.text).substr(0, cut), &r, &a));
            CHECK(s.feed(std::string_view(c.text).substr(cut), &r, &a));
            s.feed("", &r, &a, true);
            if (r != c.reasoning || a != c.answer)
                throw std::runtime_error("splitter mismatch on [" + c.text + "] cut " + std::to_string(cut) + ": [" +
                                         r + "] [" + a + "]");
        }
    }
    ThinkSplitter s;
    CHECK(!s.feed(std::string(ThinkSplitter::max_lead_whitespace + 1, ' '), nullptr, nullptr));
}

static void constraint_tests() {
    const std::string obj = grammar::json_object_grammar();
    // Direct answer, multi-terminal tokens, EOS forced when the root closes.
    {
        auto c = make(obj, true);
        CHECK(!c->accepts(eos_token));
        CHECK(!c->accepts(id("[")));
        CHECK(!c->accepts(3)); // control token outside reasoning
        CHECK(c->accepts(id("\n\n")));
        CHECK(feed(*c, {id("\n\n"), id("{\""), id("name"), id("\":\""), id("hello"), id(" world")}));
        CHECK(c->accepts(id("}")) && !c->accepts(eos_token)); // "}" is string text here
        CHECK(feed(*c, {id("\"}")}));
        CHECK(c->complete());
        auto a = c->allowed();
        CHECK(a.count == 1 && a.single == eos_token); // forced EOS
        CHECK(!c->accepts(id("\n")) && !c->accepts(id(" ")) && !c->accepts(id("}\n")));
    }
    // Leading whitespace bound.
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("\n\n"), id("\n\n"), id("\n\n"), id("\n\n")})); // 8 bytes
        CHECK(!c->accepts(id(" ")) && !c->accepts(id("\n")));
        CHECK(c->accepts(id("{")));
    }
    // Think block: tags split over tokens, unconstrained inside, EOS forbidden inside.
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("<"), id("think"), id(">")}));
        CHECK(c->phase() == ThinkSplitter::Think);
        CHECK(c->accept_first(false));
        CHECK(c->accepts(id("[")) && c->accepts(3) && c->accepts(id("}]")));
        CHECK(!c->accepts(eos_token));
        CHECK(!c->accepts(id("</think>x"))); // closes into text the grammar rejects
        CHECK(c->accepts(id("</think>\n\n{")));
        auto a = c->allowed();
        CHECK(a.count == vocabulary_size - 2);
        CHECK(!bit(*a.mask, eos_token) && !bit(*a.mask, id("</think>x")));
        CHECK(feed(*c, {id("hello"), id("</"), id("think"), id(">")}));
        CHECK(c->phase() == ThinkSplitter::AnswerLead);
        CHECK(!c->accepts(id("<think>"))); // no second block
        CHECK(feed(*c, {id("\n\n"), id("{"), id("}")}));
        CHECK(c->allowed().single == eos_token);
    }
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("<think>"), id("abc>"), id("</think>\n\n{"), id("}")}));
        CHECK(c->complete());
    }
    // Partial think tag that turns into an invalid answer is rejected.
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("<th")}));
        CHECK(!c->accepts(id("x")) && c->accepts(id("ink>")));
    }
    // Reasoning disabled: '<' is just invalid JSON.
    {
        auto c = make(obj, true, false);
        CHECK(!c->accepts(id("<")) && !c->accepts(id("<think>")));
        CHECK(c->accepts(id(" {"))); // first piece: leading space stripped -> "{"
    }
    // A lone space piece emits nothing at first position: rejected outside reasoning.
    {
        auto c = make(obj, true, false);
        CHECK(!c->accepts(id(" ")));
        CHECK(feed(*c, {id(" {")}));
        CHECK(c->accepts(id(" \""))); // later pieces keep their space
    }
    // Multi-byte UTF-8 split across tokens (byte tokens and partial pieces).
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("{\""), 5 + 0xE2, 5 + 0x82}));
        CHECK(!c->accepts(id("\"")));     // incomplete codepoint cannot close the string
        CHECK(!c->accepts(5 + 0x41));      // not a continuation byte
        CHECK(c->accepts(5 + 0xAC));
        CHECK(feed(*c, {5 + 0xAC, id("\xe2\x82"), id("\xac\""), id(":1}")}));
        CHECK(c->complete());
    }
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("{\"")}));
        CHECK(!c->accepts(5 + 0xC0) && !c->accepts(5 + 0xC1) && !c->accepts(5 + 0xF5) &&
              !c->accepts(5 + 0x80)); // overlong/invalid lead bytes, stray continuation
        CHECK(c->accepts(id("\xf0\x9f\x98\x80")));
        auto overlong = make(obj, true);
        CHECK(feed(*overlong, {id("{\""), 5 + 0xE0}));
        CHECK(!overlong->accepts(5 + 0x80) && overlong->accepts(5 + 0xA0)); // E0 80.. is overlong
        auto surrogate = make(obj, true);
        CHECK(feed(*surrogate, {id("{\""), 5 + 0xED}));
        CHECK(!surrogate->accepts(5 + 0xA0) && surrogate->accepts(5 + 0x9F)); // ED A0.. = surrogates
        auto beyond = make(obj, true);
        CHECK(feed(*beyond, {id("{\""), 5 + 0xF4}));
        CHECK(!beyond->accepts(5 + 0x90) && beyond->accepts(5 + 0x8F)); // > U+10FFFF
    }
    // Duplicate pieces: byte token 'A' and piece "A" are both allowed.
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("{\"")}));
        auto a = c->allowed();
        CHECK(bit(*a.mask, byte_id('A')) && bit(*a.mask, id("A")));
    }
    // Schema: enum forces exact tokens; integer bounds in masks.
    {
        auto s = schema(R"({"type":"object","properties":{"c":{"enum":["red","green"]},"n":{"type":"integer","minimum":1,"maximum":12}},
            "required":["c","n"],"additionalProperties":false})");
        auto c = make(grammar::json_schema_grammar(s), true);
        CHECK(feed(*c, bytes("{\"c\":")));
        CHECK(c->accepts(id("\"red\"")) && !c->accepts(id("\"name")));
        CHECK(feed(*c, {id("\"red\"")}));
        CHECK(feed(*c, bytes(",\"n\":")));
        CHECK(c->accepts(id("12")) && !c->accepts(id("123")) && !c->accepts(id("0")) && !c->accepts(id("-")));
        CHECK(feed(*c, {id("1")}));
        CHECK(!c->accepts(eos_token) && c->accepts(id("}")));
        CHECK(feed(*c, {id("}")}));
        CHECK(c->allowed().single == eos_token);
    }
    // Raw grammar (not JSON mode): trailing whitespace allowed, EOS optional while accepting.
    {
        auto c = make("root ::= \"ab\" [ ]*", false);
        CHECK(feed(*c, {id("ab")}));
        CHECK(c->accepts(eos_token) && c->accepts(id(" ")));
    }
    // Mask cache: a state's mask is computed once.
    {
        auto c = make(obj, true);
        CHECK(feed(*c, {id("{\"")}));
        c->allowed();
        CHECK(feed(*c, {id("ab")}));
        c->allowed();
        CHECK(feed(*c, {id("ab")}));
        c->allowed();
        CHECK(c->stats().mask_hits >= 1);
    }
}

// Random constrained decoding: every step's mask equals accepts() over the whole
// vocabulary; completed answers parse and validate against the schema.
static void random_walk_tests() {
    std::mt19937_64 rng(1234);
    std::vector<ojson> schemas = {
        ojson(nullptr), // json_object
        schema(R"({"type":"object","properties":{"name":{"type":"string","maxLength":12},"age":{"type":"integer","minimum":0,"maximum":150},
            "tags":{"type":"array","items":{"enum":["red","green"]},"maxItems":3},"ok":{"type":"boolean"}},"required":["name","age"],"additionalProperties":false})"),
        schema(R"({"$defs":{"n":{"type":"object","properties":{"v":{"type":"number","minimum":-2,"maximum":3}, "k":{"type":"array","items":{"$ref":"#/$defs/n"},"maxItems":2}},
            "required":["v"],"additionalProperties":false}},"$ref":"#/$defs/n"})"),
        schema(R"({"anyOf":[{"type":"integer","exclusiveMinimum":10},{"type":"null"},{"type":"string","minLength":1,"maxLength":2}]})"),
        schema(R"({"type":"array","prefixItems":[{"const":"x"},{"type":"string","format":"date"}],"items":{"type":"boolean"},"minItems":2,"maxItems":4})"),
    };
    int completed = 0, walks = 0;
    for (size_t si = 0; si < schemas.size(); ++si)
        for (int trial = 0; trial < 12; ++trial) {
            const bool think = trial % 3 != 2;
            const std::string g = schemas[si].is_null() ? grammar::json_object_grammar()
                                                        : grammar::json_schema_grammar(schemas[si]);
            auto c = make(g, true, think);
            // Optionally open a think block first.
            std::vector<int> prefix;
            if (think && trial % 3 == 1)
                prefix = {id("<think>"), id("hello"), id(" world"), id("</think>\n")};
            CHECK(feed(*c, prefix));
            auto chooser = biased_chooser(vocab(), rng);
            auto r = walk(vocab(), *c, think, chooser, 300, trial < 3, prefix.empty());
            ++walks;
            if (!r.ended)
                continue;
            ++completed;
            json parsed;
            try {
                parsed = json::parse(r.answer);
            } catch (const std::exception &e) {
                throw std::runtime_error("walk produced invalid JSON: [" + r.answer + "] " + e.what());
            }
            if (schemas[si].is_null())
                CHECK(parsed.is_object());
            else if (!validate(schemas[si], parsed, schemas[si]))
                throw std::runtime_error("walk output violates schema: " + r.answer);
        }
    std::cout << "  random walks: " << completed << "/" << walks << " completed and validated\n";
    CHECK(completed >= walks / 2);
}

static void sampling_tests() {
    std::vector<float> logits(vocabulary_size, 0.f);
    logits[7] = 10;
    logits[9] = 5;
    logits[100] = 4;
    logits[20000] = 4.5f;
    TokenMask mask{};
    auto allow = [&](int t) { mask[size_t(t) >> 6] |= uint64_t(1) << (t & 63); };
    allow(9);
    allow(100);
    allow(20000);
    std::mt19937_64 rng(1);
    std::span<const float> lo(logits.data(), 16384), hi(logits.data() + 16384, 16384);
    SamplingOptions greedy;
    CHECK(sample_logits(lo, hi, greedy, rng) == 7);
    CHECK(sample_logits(lo, hi, greedy, rng, mask.data()) == 9);
    SamplingOptions hot{1.0f, 0, 1.0f, 0};
    std::map<int, int> counts;
    for (int i = 0; i < 3000; ++i)
        counts[sample_logits(lo, hi, hot, rng, mask.data())]++;
    CHECK(counts.size() <= 3 && counts[9] > counts[20000] && counts[20000] > counts[100] && counts[100] > 0);
    SamplingOptions topk{1.0f, 1, 1.0f, 0};
    for (int i = 0; i < 50; ++i)
        CHECK(sample_logits(lo, hi, topk, rng, mask.data()) == 9);
    SamplingOptions topp{1.0f, 0, 0.5f, 0};
    for (int i = 0; i < 50; ++i)
        CHECK(sample_logits(lo, hi, topp, rng, mask.data()) == 9);
    TokenMask none{};
    rejects([&] { sample_logits(lo, hi, hot, rng, none.data()); });
    // Unconstrained results are unchanged by the refactor (same RNG stream).
    std::mt19937_64 a(42), b(42);
    for (int i = 0; i < 20; ++i)
        CHECK(sample_logits(lo, hi, hot, a) == sample_logits(std::span<const float>(logits), hot, b));
}

static void protocol_tests() {
    auto base = [] {
        return nlohmann::json{{"messages", {{{"role", "user"}, {"content", "Hi"}}}}};
    };
    auto j = base();
    CHECK(parse_chat(j).structured.kind == StructuredOutput::None);
    j["response_format"] = {{"type", "text"}};
    CHECK(parse_chat(j).structured.kind == StructuredOutput::None);
    j["response_format"] = {{"type", "json_object"}};
    auto r = parse_chat(j);
    CHECK(r.structured.kind == StructuredOutput::JsonObject && r.structured.json && r.structured.allow_think);
    j["reasoning_effort"] = "none";
    CHECK(!parse_chat(j).structured.allow_think);
    j["reasoning_effort"] = "high";
    CHECK(parse_chat(j).structured.allow_think);
    // Property order follows the request text when the ordered parse is passed.
    const std::string body = R"({"messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_schema",
        "json_schema":{"name":"p","strict":true,"schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}},
        "required":["zeta","alpha"],"additionalProperties":false}}}})";
    auto ordered = nlohmann::ordered_json::parse(body);
    auto rs = parse_chat(nlohmann::json::parse(body), &ordered);
    CHECK(rs.structured.kind == StructuredOutput::JsonSchema && rs.structured.name == "p" && rs.structured.strict);
    CHECK(matches(rs.structured.gbnf, R"({"zeta":"a","alpha":1})"));
    CHECK(!matches(rs.structured.gbnf, R"({"alpha":1,"zeta":"a"})"));
    auto g = base();
    g["grammar"] = "root ::= \"yes\" | \"no\"";
    CHECK(parse_chat(g).structured.kind == StructuredOutput::Grammar && !parse_chat(g).structured.json);
    auto param = [&](nlohmann::json req) {
        try {
            parse_chat(req);
        } catch (const RequestError &e) {
            return e.param;
        }
        throw std::runtime_error("expected RequestError");
    };
    auto bad = base();
    bad["response_format"] = {{"type", "xml"}};
    CHECK(param(bad) == "response_format.type");
    bad["response_format"] = {{"type", "json_schema"}};
    CHECK(param(bad) == "response_format.json_schema");
    bad["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema", {{"type", "string"}, {"pattern", "a"}}}}}};
    CHECK(param(bad) == "response_format.json_schema.schema");
    bad["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "bad name!"}, {"schema", nlohmann::json::object()}}}};
    CHECK(param(bad) == "response_format.json_schema.name");
    auto both = g;
    both["response_format"] = {{"type", "json_object"}};
    CHECK(param(both) == "grammar");
    auto effort = base();
    effort["reasoning_effort"] = "none";
    CHECK(param(effort) == "reasoning_effort");
    effort["reasoning_effort"] = "max";
    CHECK(param(effort) == "reasoning_effort");
}

static void cache_tests() {
    GrammarCache cache(2);
    const std::string g = grammar::json_object_grammar();
    const GrammarEngine *first;
    {
        auto c = std::make_unique<StructuredConstraint>(vocab(), cache.acquire(g, true), true, &cache);
        CHECK(feed(*c, {id("{\"")}));
        c->allowed();
        first = nullptr;
    }
    auto e = cache.acquire(g, true);
    first = e.get();
    CHECK(!e->masks.empty()); // warm engine came back from the cache
    cache.release(std::move(e));
    auto again = cache.acquire(g, true);
    CHECK(again.get() == first);
    auto other = cache.acquire(g, false); // json_mode differs: new engine
    CHECK(other.get() != first && other->masks.empty());
}

// Rough per-token cost on the synthetic vocabulary (the real vocabulary is measured by
// behemoth_structured_bench).
static void timing_report() {
    std::mt19937_64 rng(7);
    GrammarCache cache;
    auto s = schema(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0,"maximum":150},
        "tags":{"type":"array","items":{"type":"string"}}},"required":["name","age","tags"],"additionalProperties":false})");
    const std::string g = grammar::json_schema_grammar(s);
    double total_ms = 0;
    int tokens = 0;
    for (int i = 0; i < 6; ++i) {
        auto c = std::make_unique<StructuredConstraint>(vocab(), cache.acquire(g, true), false, &cache);
        auto chooser = biased_chooser(vocab(), rng);
        auto t0 = std::chrono::steady_clock::now();
        auto r = walk(vocab(), *c, false, chooser, 200, false);
        total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        tokens += r.tokens;
    }
    std::cout << "  synthetic vocab: " << tokens << " tokens, " << total_ms * 1000 / tokens
              << " us/token (allowed()+accepts()+advance(), mask always built)\n";
}

int main() {
    try {
        for (auto [name, fn] : std::vector<std::pair<const char *, std::function<void()>>>{
                 {"gbnf", gbnf_tests},
                 {"json_object", json_object_tests},
                 {"json_schema", json_schema_tests},
                 {"think_splitter", splitter_tests},
                 {"constraint", constraint_tests},
                 {"random_walk", random_walk_tests},
                 {"masked_sampling", sampling_tests},
                 {"protocol", protocol_tests},
                 {"grammar_cache", cache_tests},
                 {"timing", timing_report}}) {
            fn();
            std::cout << "PASS " << name << '\n';
        }
        std::cout << "PASS " << assertions << " assertions\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
