// Shared helpers for the structured-output tests and the vocabulary benchmark:
// a small JSON-Schema validator (independent of the grammar compiler) and a
// constrained random-walk decoder.
#pragma once
#include "behemoth/constraint.hpp"
#include <bit>
#include <functional>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>

namespace structured_test {
using behemoth::StructuredConstraint;
using behemoth::ThinkSplitter;
using behemoth::TokenMask;
using behemoth::Vocabulary;
using ojson = nlohmann::ordered_json;
using json = nlohmann::json;

inline size_t codepoints(const std::string &s) {
    size_t n = 0;
    for (unsigned char c : s)
        n += (c & 0xC0) != 0x80;
    return n;
}

// Validates `v` against the supported JSON-Schema subset (reference semantics).
inline bool validate(const ojson &schema, const json &v, const ojson &root) {
    if (schema.is_boolean())
        return schema.get<bool>();
    if (schema.contains("$ref")) {
        std::string ref = schema["$ref"];
        if (ref == "#")
            return validate(root, v, root);
        const ojson *cur = &root;
        size_t i = 2;
        while (i <= ref.size()) {
            size_t j = ref.find('/', i);
            cur = &(*cur)[ref.substr(i, j == std::string::npos ? std::string::npos : j - i)];
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
        return validate(*cur, v, root);
    }
    if (schema.value("nullable", false) && v.is_null())
        return true;
    for (const char *k : {"anyOf", "oneOf"})
        if (schema.contains(k)) {
            for (const auto &alt : schema[k])
                if (validate(alt, v, root))
                    return true;
            return false;
        }
    if (schema.contains("allOf")) {
        for (const auto &alt : schema["allOf"])
            if (!validate(alt, v, root))
                return false;
        return true;
    }
    if (schema.contains("const"))
        return json::parse(schema["const"].dump()) == v;
    if (schema.contains("enum")) {
        for (const auto &e : schema["enum"])
            if (json::parse(e.dump()) == v)
                return true;
        return false;
    }
    if (schema.contains("type")) {
        auto matches = [&](const std::string &t) {
            if (t == "object")
                return v.is_object();
            if (t == "array")
                return v.is_array();
            if (t == "string")
                return v.is_string();
            if (t == "integer")
                return v.is_number_integer() || (v.is_number_float() && v.get<double>() == std::floor(v.get<double>()));
            if (t == "number")
                return v.is_number();
            if (t == "boolean")
                return v.is_boolean();
            if (t == "null")
                return v.is_null();
            return false;
        };
        bool any = false;
        if (schema["type"].is_string())
            any = matches(schema["type"]);
        else
            for (const auto &t : schema["type"])
                any |= matches(t);
        if (!any)
            return false;
    }
    if (v.is_object()) {
        if (schema.contains("required"))
            for (const auto &r : schema["required"])
                if (!v.contains(r.get<std::string>()))
                    return false;
        const ojson props = schema.value("properties", ojson::object());
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (props.contains(it.key())) {
                if (!validate(props[it.key()], it.value(), root))
                    return false;
            } else if (schema.contains("additionalProperties")) {
                const auto &ap = schema["additionalProperties"];
                if (ap.is_boolean() ? !ap.get<bool>() : !validate(ap, it.value(), root))
                    return false;
            }
        }
    }
    if (v.is_array()) {
        size_t n = v.size(), prefix = 0;
        if (schema.contains("minItems") && n < schema["minItems"].get<size_t>())
            return false;
        if (schema.contains("maxItems") && n > schema["maxItems"].get<size_t>())
            return false;
        if (schema.contains("prefixItems")) {
            prefix = schema["prefixItems"].size();
            for (size_t i = 0; i < std::min(n, prefix); ++i)
                if (!validate(schema["prefixItems"][i], v[i], root))
                    return false;
        }
        if (schema.contains("items"))
            for (size_t i = prefix; i < n; ++i)
                if (!validate(schema["items"], v[i], root))
                    return false;
    }
    if (v.is_string()) {
        const size_t n = codepoints(v.get<std::string>());
        if (schema.contains("minLength") && n < schema["minLength"].get<size_t>())
            return false;
        if (schema.contains("maxLength") && n > schema["maxLength"].get<size_t>())
            return false;
    }
    if (v.is_number()) {
        const double d = v.get<double>();
        auto num = [&](const char *k) { return schema[k].get<double>(); };
        if (schema.contains("minimum") && d < num("minimum"))
            return false;
        if (schema.contains("maximum") && d > num("maximum"))
            return false;
        if (schema.contains("exclusiveMinimum") && schema["exclusiveMinimum"].is_number() && d <= num("exclusiveMinimum"))
            return false;
        if (schema.contains("exclusiveMaximum") && schema["exclusiveMaximum"].is_number() && d >= num("exclusiveMaximum"))
            return false;
    }
    return true;
}

inline bool bit(const TokenMask &m, int t) {
    return (m[size_t(t) >> 6] >> (t & 63)) & 1;
}
inline std::vector<int> ids(const TokenMask &m) {
    std::vector<int> out;
    for (size_t w = 0; w < m.size(); ++w)
        for (uint64_t x = m[w]; x; x &= x - 1)
            out.push_back(int(w * 64 + size_t(std::countr_zero(x))));
    return out;
}

struct WalkResult {
    std::string reasoning, answer;
    bool ended = false;
    int tokens = 0;
    std::vector<int> sequence;
};

// Decodes with the constraint: `choose` picks a token from the allowed ids.
// With `exact`, every step also checks the mask against accepts() for every id.
inline WalkResult walk(const Vocabulary &vocab, StructuredConstraint &c, bool allow_think,
                       const std::function<int(const std::vector<int> &)> &choose, int max_tokens,
                       bool exact, bool first = true) {
    // A walk that starts after already fed tokens begins in the Lead phase of its own
    // splitter: fine for prefixes that end with a closed think block.
    WalkResult r;
    ThinkSplitter split(allow_think);
    for (; r.tokens < max_tokens; ++r.tokens) {
        auto a = c.allowed();
        if (!a.mask || a.count <= 0)
            throw std::runtime_error("constraint allows nothing");
        auto allowed = ids(*a.mask);
        if (int(allowed.size()) != a.count)
            throw std::runtime_error("allowed count mismatch");
        if (exact)
            for (int id = 0; id < int(vocab.size()); ++id)
                if (bit(*a.mask, id) != c.accepts(id))
                    throw std::runtime_error("mask/accepts mismatch at token " + std::to_string(id) +
                                             " after " + std::to_string(r.tokens) + " tokens");
        int t = a.count == 1 ? a.single : choose(allowed);
        if (!c.accepts(t))
            throw std::runtime_error("chosen token not accepted");
        c.advance(t);
        r.sequence.push_back(t);
        if (t == vocab.eos()) {
            r.ended = true;
            ++r.tokens;
            break;
        }
        const auto &tok = vocab.token(t);
        std::string piece;
        if (!tok.special) {
            piece = first ? tok.first : tok.rest;
            if (first)
                first = tok.first_after;
        }
        split.feed(piece, &r.reasoning, &r.answer);
    }
    split.feed("", &r.reasoning, &r.answer, true);
    return r;
}

// Prefers EOS and short structural tokens so random walks terminate.
inline std::function<int(const std::vector<int> &)> biased_chooser(const Vocabulary &vocab, std::mt19937_64 &rng) {
    return [&vocab, &rng](const std::vector<int> &allowed) {
        std::vector<int> preferred;
        for (int id : allowed) {
            if (id == vocab.eos()) {
                preferred.assign(4, id);
                break;
            }
            const auto &p = vocab.token(id).rest;
            if (p.size() == 1 && std::string("\"}],:0123456789tfn.e").find(p[0]) != std::string::npos)
                preferred.push_back(id);
        }
        std::uniform_real_distribution<double> u(0, 1);
        const auto &pool = !preferred.empty() && u(rng) < 0.7 ? preferred : allowed;
        return pool[std::uniform_int_distribution<size_t>(0, pool.size() - 1)(rng)];
    };
}
} // namespace structured_test
