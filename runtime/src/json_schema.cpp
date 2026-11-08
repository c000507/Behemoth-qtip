#include "behemoth/json_schema.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace behemoth::grammar {

using json = nlohmann::ordered_json;

namespace {

// Exponents have at most 2 digits so every number fits a double (no overflow to inf).
// Shared terminal rules. `space` is deliberately bounded (one space, or a
// newline plus up to 20 columns of indentation) so the model cannot stall
// emitting whitespace forever. Digit runs are capped at 16 for the same reason.
// There is no leading/trailing whitespace rule: the structured-output gate
// strips (bounded) whitespace before the answer and, in JSON mode, stops the
// answer as soon as the root value is complete (constraint.hpp).
const char *primitives = R"GBNF(
space ::= ( " " | "\n" [ \t]{0,20} | "\r\n" [ \t]{0,20} )?
string ::= "\"" char* "\"" space
char ::= [^"\\\x00-\x1F\x7F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} )
integer ::= "-"? ( "0" | [1-9] [0-9]{0,15} ) space
number ::= "-"? ( "0" | [1-9] [0-9]{0,15} ) ( "." [0-9]{1,16} )? ( [eE] [-+]? [0-9]{1,2} )? space
boolean ::= ( "true" | "false" ) space
null ::= "null" space
value ::= object | array | string | number | boolean | null
object ::= "{" space ( kv ( "," space kv )* )? "}" space
kv ::= string ":" space value
array ::= "[" space ( value ( "," space value )* )? "]" space
)GBNF";

[[noreturn]] void unsupported(const std::string &what) {
    throw SchemaError("unsupported JSON Schema feature: " + what);
}

std::string literal(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20 || c == 0x7F) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\x%02X", c);
                out += buf;
            } else
                out += static_cast<char>(c);
        }
    }
    return out + "\"";
}

// ---------------------------------------------------------------------------
// Decimal ranges
// ---------------------------------------------------------------------------
constexpr long long max_magnitude = 9999999999999999LL; // 16 digits, as the primitives

bool all_of(const std::string &s, char c) {
    return s.find_first_not_of(c) == std::string::npos;
}

// Digit strings of a.size() == b.size() digits whose value lies in [a, b].
std::string same_length(const std::string &a, const std::string &b) {
    if (a == b)
        return literal(a);
    if (a.size() == 1)
        return std::string("[") + a[0] + "-" + b[0] + "]";
    if (a[0] == b[0])
        return literal(a.substr(0, 1)) + " " + same_length(a.substr(1), b.substr(1));
    const std::string at = a.substr(1), bt = b.substr(1);
    const size_t rest = at.size();
    const bool low_full = all_of(at, '0'), high_full = all_of(bt, '9');
    std::vector<std::string> alts;
    if (!low_full)
        alts.push_back(literal(a.substr(0, 1)) + " " +
                       same_length(at, std::string(rest, '9')));
    const char lo = char(a[0] + (low_full ? 0 : 1)), hi = char(b[0] - (high_full ? 0 : 1));
    if (lo <= hi) {
        std::string cls = lo == hi ? literal(std::string(1, lo)) : std::string("[") + lo + "-" + hi + "]";
        alts.push_back(cls + " [0-9]{" + std::to_string(rest) + "}");
    }
    if (!high_full)
        alts.push_back(literal(b.substr(0, 1)) + " " +
                       same_length(std::string(rest, '0'), bt));
    if (alts.size() == 1)
        return alts[0];
    std::string out = "( ";
    for (size_t i = 0; i < alts.size(); ++i)
        out += (i ? " | " : "") + alts[i];
    return out + " )";
}

long long pow10(size_t n) {
    long long v = 1;
    while (n--)
        v *= 10;
    return v;
}

// Canonical decimal integers in [a, b] (a >= 0); unbounded b means up to 16 digits.
std::string nonneg_range(long long a, long long b, bool has_b) {
    if (!has_b || b > max_magnitude)
        b = max_magnitude;
    if (a > b)
        return "";
    const std::string as = std::to_string(a), bs = std::to_string(b);
    std::vector<std::string> alts;
    for (size_t len = as.size(); len <= bs.size(); ++len) {
        long long lo = len == as.size() ? a : pow10(len - 1);
        long long hi = len == bs.size() ? b : pow10(len) - 1;
        if (len > as.size() && len < bs.size() && !(has_b && b < max_magnitude)) {
            // every remaining length is complete up to the 16-digit cap
            alts.push_back("[1-9] [0-9]{" + std::to_string(len - 1) + "," + std::to_string(bs.size() - 1) + "}");
            break;
        }
        alts.push_back(same_length(std::to_string(lo), std::to_string(hi)));
    }
    if (alts.size() == 1)
        return alts[0];
    std::string out = "( ";
    for (size_t i = 0; i < alts.size(); ++i)
        out += (i ? " | " : "") + alts[i];
    return out + " )";
}

std::string join_alternatives(const std::vector<std::string> &alts) {
    std::vector<std::string> nonempty;
    for (const auto &a : alts)
        if (!a.empty())
            nonempty.push_back(a);
    if (nonempty.empty())
        return "";
    if (nonempty.size() == 1)
        return nonempty[0];
    std::string out = "( ";
    for (size_t i = 0; i < nonempty.size(); ++i)
        out += (i ? " | " : "") + nonempty[i];
    return out + " )";
}

// Reals x >= 0 (no exponent) with integer bounds: x > / >= a, x < / <= b.
std::string real_nonneg(long long a, bool inc_a, long long b, bool inc_b, bool has_b) {
    const std::string fraction = "( \".\" [0-9]{1,16} )?";
    const std::string nonzero_fraction = "\".\" [0-9]{0,15} [1-9]";
    const std::string zero_fraction = "( \".\" \"0\"{1,16} )?";
    if (has_b && a > b)
        return "";
    if (has_b && a == b)
        return inc_a && inc_b ? literal(std::to_string(a)) + " " + zero_fraction : "";
    std::vector<std::string> alts;
    alts.push_back(literal(std::to_string(a)) + " " + (inc_a ? fraction : nonzero_fraction));
    if (!has_b) {
        std::string mid = nonneg_range(a + 1, 0, false);
        if (!mid.empty())
            alts.push_back(mid + " " + fraction);
    } else {
        if (a + 1 <= b - 1)
            alts.push_back(nonneg_range(a + 1, b - 1, true) + " " + fraction);
        if (inc_b)
            alts.push_back(literal(std::to_string(b)) + " " + zero_fraction);
    }
    return join_alternatives(alts);
}

struct Bound {
    bool has = false;
    long long value = 0;
    bool inclusive = true;
};

std::string number_range(Bound lo, Bound hi) {
    if (lo.has && hi.has && (lo.value > hi.value || (lo.value == hi.value && !(lo.inclusive && hi.inclusive))))
        throw SchemaError("numeric bounds admit no value");
    std::vector<std::string> alts;
    // x < 0: magnitude m = -x > 0 with m <= -lo, and m >= -hi when hi < 0.
    if (!lo.has || lo.value < 0) {
        long long m_lo = 0;
        bool m_lo_inc = false; // m > 0
        if (hi.has && hi.value < 0) {
            m_lo = -hi.value;
            m_lo_inc = hi.inclusive;
        }
        std::string mag = real_nonneg(m_lo, m_lo_inc, lo.has ? -lo.value : 0, lo.inclusive, lo.has);
        if (!mag.empty())
            alts.push_back("\"-\" " + mag);
    }
    // x >= 0
    if (!hi.has || hi.value > 0 || (hi.value == 0 && hi.inclusive)) {
        long long a = 0;
        bool inc_a = true;
        if (lo.has && lo.value >= 0) {
            a = lo.value;
            inc_a = lo.inclusive;
        }
        alts.push_back(real_nonneg(a, inc_a, hi.has ? hi.value : 0, hi.inclusive, hi.has));
    }
    std::string body = join_alternatives(alts);
    if (body.empty())
        throw SchemaError("numeric bounds admit no value");
    return body;
}

} // namespace

std::string integer_range_gbnf(long long lo, bool has_lo, long long hi, bool has_hi) {
    if (has_lo && lo < -max_magnitude)
        has_lo = false;
    if (has_hi && hi > max_magnitude)
        has_hi = false;
    if (has_lo && lo > max_magnitude)
        throw SchemaError("integer minimum exceeds the supported 16 digits");
    if (has_hi && hi < -max_magnitude)
        throw SchemaError("integer maximum exceeds the supported 16 digits");
    if (has_lo && has_hi && lo > hi)
        throw SchemaError("integer bounds admit no value");
    std::vector<std::string> alts;
    if (!has_lo || lo < 0) {
        long long m_lo = has_hi && hi < 0 ? -hi : 1;
        alts.push_back("\"-\" " + nonneg_range(m_lo, has_lo ? -lo : 0, has_lo));
    }
    if (!has_hi || hi >= 0)
        alts.push_back(nonneg_range(has_lo && lo > 0 ? lo : 0, has_hi ? hi : 0, has_hi));
    return join_alternatives(alts);
}

namespace {

struct Compiler {
    // Bounds, all -> SchemaError (a 400): schema nesting depth (visit is
    // recursive and runs on the HTTP handler thread), generated rule count
    // and grammar text size (each generated rule becomes matcher state),
    // per-repetition counts and properties per object.
    static constexpr int max_depth = 64;
    static constexpr size_t max_rules = 65536;
    static constexpr size_t max_grammar_bytes = 2u << 20;
    static constexpr int max_repeat = 4096; // = the GBNF parser's cap
    static constexpr size_t max_properties = 1024;

    const json *root; // the schema being compiled ($ref "#" and "#/..." resolve against it)
    int depth = 0;
    size_t grammar_bytes = 0;
    std::map<std::string, int> next_suffix;
    std::vector<std::pair<std::string, std::string>> rules; // (name, body) in order
    std::map<std::string, std::string> body_to_name;
    std::set<std::string> names;
    std::map<std::string, std::string> ref_rules; // "$ref" pointer -> rule name
    bool used_format[3] = {false, false, false};

    explicit Compiler(const json &r) : root(&r) {
        for (const char *n : {"space", "string", "char", "integer", "number", "boolean", "null", "value",
                              "object", "kv", "array", "root", "date-fmt", "time-fmt", "uuid-fmt"})
            names.insert(n);
    }

    static std::string sanitize(const std::string &s) {
        std::string out;
        for (char c : s) {
            if (out.size() >= 24)
                break;
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-')
                out += c;
            else
                out += '-';
        }
        return out.empty() ? "r" : out;
    }

    std::string add_rule(const std::string &hint, const std::string &body) {
        auto it = body_to_name.find(body);
        if (it != body_to_name.end())
            return it->second;
        std::string name = unique_name(sanitize(hint));
        rules.push_back({name, body});
        body_to_name[body] = name;
        account(name, body);
        return name;
    }
    void account(const std::string &name, const std::string &body) {
        grammar_bytes += name.size() + body.size() + 5;
        if (rules.size() > max_rules)
            throw SchemaError("schema compiles to more than " + std::to_string(max_rules) + " grammar rules");
        if (grammar_bytes > max_grammar_bytes)
            throw SchemaError("schema compiles to a grammar larger than " +
                              std::to_string(max_grammar_bytes >> 20) + " MB");
    }

    std::string unique_name(const std::string &hint) {
        std::string name = hint;
        while (names.count(name))
            name = hint + "-" + std::to_string(++next_suffix[hint]);
        names.insert(name);
        return name;
    }

    static std::string json_literal(const json &v) {
        return literal(v.dump());
    }

    static std::string repeat(const std::string &item, int min_n, int max_n) {
        if (min_n > max_repeat || max_n > max_repeat)
            throw SchemaError("minLength/maxLength/minItems/maxItems bound " +
                              std::to_string(std::max(min_n, max_n)) + " exceeds the supported " +
                              std::to_string(max_repeat));
        if (max_n < 0) {
            if (min_n == 0)
                return item + "*";
            if (min_n == 1)
                return item + "+";
            return item + "{" + std::to_string(min_n) + ",}";
        }
        if (min_n == max_n) {
            if (min_n == 0)
                return "";
            if (min_n == 1)
                return item;
            return item + "{" + std::to_string(min_n) + "}";
        }
        if (min_n == 0 && max_n == 1)
            return item + "?";
        return item + "{" + std::to_string(min_n) + "," + std::to_string(max_n) + "}";
    }

    const json *resolve_ref(const std::string &ref) {
        if (ref == "#")
            return root;
        if (ref.rfind("#/", 0) != 0)
            unsupported("$ref '" + ref + "' (only local #/ references)");
        const json *cur = root;
        size_t i = 2;
        while (i <= ref.size()) {
            size_t j = ref.find('/', i);
            std::string seg = ref.substr(i, j == std::string::npos ? std::string::npos : j - i);
            std::string key;
            for (size_t k = 0; k < seg.size(); ++k) {
                if (seg[k] == '~' && k + 1 < seg.size()) {
                    key += seg[k + 1] == '0' ? '~' : '/';
                    ++k;
                } else
                    key += seg[k];
            }
            if (cur->is_object() && cur->contains(key))
                cur = &(*cur)[key];
            else if (cur->is_array()) {
                size_t idx = 0;
                bool numeric = !key.empty() && key.size() <= 9;
                for (char c : key) {
                    if (c < '0' || c > '9') {
                        numeric = false;
                        break;
                    }
                    idx = idx * 10 + static_cast<size_t>(c - '0');
                }
                if (!numeric || idx >= cur->size())
                    throw SchemaError("$ref '" + ref + "' does not resolve");
                cur = &(*cur)[idx];
            } else
                throw SchemaError("$ref '" + ref + "' does not resolve");
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
        return cur;
    }

    static int count_or(const json &s, const char *key, int fallback) {
        if (!s.contains(key))
            return fallback;
        const json &v = s[key];
        if (!v.is_number_integer() || v.get<long long>() < 0 || v.get<long long>() > max_repeat)
            throw SchemaError(std::string("'") + key + "' must be an integer in 0.." + std::to_string(max_repeat));
        return v.get<int>();
    }

    // Keywords that only annotate (never change the accepted language).
    static bool annotation(const std::string &k) {
        static const std::set<std::string> a = {"title", "description", "default", "examples", "$schema",
                                                "$id", "$comment", "deprecated", "readOnly", "writeOnly",
                                                "contentEncoding", "contentMediaType", "$defs", "definitions"};
        return a.count(k) != 0;
    }
    static void check_keywords(const json &s) {
        static const std::set<std::string> known = {
            "type", "properties", "required", "additionalProperties", "items", "prefixItems",
            "additionalItems", "minItems", "maxItems", "enum", "const", "anyOf", "oneOf", "allOf", "$ref",
            "nullable", "minLength", "maxLength", "format", "minimum", "maximum", "exclusiveMinimum",
            "exclusiveMaximum"};
        for (auto it = s.begin(); it != s.end(); ++it)
            if (!known.count(it.key()) && !annotation(it.key()))
                unsupported("keyword '" + it.key() + "'");
    }
    // Combinators/refs/const/enum must stand alone: sibling keywords would be
    // silently dropped otherwise. `type` and `nullable` beside enum/const are
    // tolerated (the listed values are the constraint).
    static void check_alone(const json &s, const char *keyword, bool allow_type) {
        for (auto it = s.begin(); it != s.end(); ++it) {
            const std::string &k = it.key();
            if (k == keyword || annotation(k) || (allow_type && (k == "type" || k == "nullable")))
                continue;
            unsupported(std::string("'") + keyword + "' combined with sibling keyword '" + k + "'");
        }
    }

    std::string visit_object(const json &s, const std::string &name) {
        std::vector<std::string> prop_names;
        std::map<std::string, std::string> prop_rules;
        std::set<std::string> required;
        if (s.contains("required")) {
            if (!s["required"].is_array())
                throw SchemaError("'required' must be an array");
            for (const auto &r : s["required"]) {
                if (!r.is_string())
                    throw SchemaError("'required' entries must be strings");
                required.insert(r.get<std::string>());
            }
        }
        auto add_property = [&](const std::string &key, const json &schema) {
            prop_names.push_back(key);
            std::string vr = visit(schema, name + "-" + key);
            prop_rules[key] = add_rule(name + "-" + key + "-kv", json_literal(json(key)) + " space \":\" space " + vr);
        };
        if (s.contains("properties")) {
            if (!s["properties"].is_object())
                throw SchemaError("'properties' must be an object");
            if (s["properties"].size() > max_properties)
                throw SchemaError("an object may declare at most " + std::to_string(max_properties) +
                                  " properties (got " + std::to_string(s["properties"].size()) + ")");
            for (auto it = s["properties"].begin(); it != s["properties"].end(); ++it)
                add_property(it.key(), it.value());
        }
        // A required name without a property schema accepts any value.
        for (const auto &r : s.value("required", json::array()))
            if (!prop_rules.count(r.get<std::string>()))
                add_property(r.get<std::string>(), json(true));

        // additionalProperties: absent/true -> any value; schema -> that value; false -> none.
        std::string extra_rule;
        bool extras = true;
        if (s.contains("additionalProperties")) {
            const json &ap = s["additionalProperties"];
            if (ap.is_boolean())
                extras = ap.get<bool>();
            else if (ap.is_object())
                extra_rule = visit(ap, name + "-additional");
            else
                throw SchemaError("'additionalProperties' must be a boolean or schema");
        }
        if (extras && extra_rule.empty())
            extra_rule = "value";
        if (extras) // extra keys are any string (a repeated key is still valid JSON)
            extra_rule = add_rule(name + "-extra-kv", "string \":\" space " + extra_rule);

        // Members in declared order, each optional property optional in
        // place, every required one mandatory:
        //   rest_i  ::= what may follow once a member has been emitted, from property i
        //   first_i ::= the first emitted member, from property i
        const size_t np = prop_names.size();
        std::string tail_end = extras ? "( \",\" space " + extra_rule + " )*" : "";
        std::vector<std::string> rest(np + 1);
        rest[np] = tail_end;
        for (size_t i = np; i-- > 0;) {
            const std::string &kv = prop_rules[prop_names[i]];
            std::string body = required.count(prop_names[i]) ? "\",\" space " + kv : "( \",\" space " + kv + " )?";
            if (!rest[i + 1].empty())
                body += " " + rest[i + 1];
            rest[i] = add_rule(name + "-rest", body);
        }
        std::string first = extras ? extra_rule + (tail_end.empty() ? "" : " " + tail_end) : "";
        bool skippable = true; // no required property from i on
        for (size_t i = np; i-- > 0;) {
            const std::string &kv = prop_rules[prop_names[i]];
            std::string body = kv;
            if (!rest[i + 1].empty())
                body += " " + rest[i + 1];
            if (required.count(prop_names[i]))
                skippable = false;
            else if (!first.empty())
                body += " | " + first;
            first = add_rule(name + "-first", body);
        }
        std::string members = first;
        if (skippable && !members.empty())
            members = "( " + members + " )?";
        std::string body = "\"{\" space " + (members.empty() ? "" : members + " ") + "\"}\" space";
        return add_rule(name, body);
    }

    std::string visit_array(const json &s, const std::string &name) {
        int min_items = count_or(s, "minItems", 0);
        int max_items = count_or(s, "maxItems", -1);
        if (max_items >= 0 && max_items < min_items)
            throw SchemaError("maxItems is smaller than minItems");
        std::vector<std::string> prefix;
        if (s.contains("prefixItems")) {
            if (!s["prefixItems"].is_array())
                throw SchemaError("'prefixItems' must be an array");
            for (size_t i = 0; i < s["prefixItems"].size(); ++i)
                prefix.push_back(visit(s["prefixItems"][i], name + "-tuple-" + std::to_string(i)));
        }
        std::string item;
        bool more = true;
        if (s.contains("items")) {
            const json &it = s["items"];
            if (it.is_boolean())
                more = it.get<bool>();
            else if (it.is_object())
                item = visit(it, name + "-item");
            else if (it.is_array()) { // draft-7 tuple form
                for (size_t i = 0; i < it.size(); ++i)
                    prefix.push_back(visit(it[i], name + "-tuple-" + std::to_string(i)));
                if (s.contains("additionalItems")) {
                    const json &ai = s["additionalItems"];
                    if (ai.is_boolean())
                        more = ai.get<bool>();
                    else
                        item = visit(ai, name + "-item");
                }
            } else
                throw SchemaError("'items' must be a schema");
        } else if (s.contains("additionalItems"))
            unsupported("'additionalItems' without tuple 'items'");
        if (more && item.empty())
            item = "value";
        if (!more)
            item.clear();

        const int np = static_cast<int>(prefix.size());
        if (max_items >= 0 && max_items < np)
            unsupported("maxItems smaller than the number of prefixItems");
        std::string seq;
        for (int i = 0; i < np; ++i)
            seq += (i ? " \",\" space " : "") + prefix[static_cast<size_t>(i)];
        int rest_min = std::max(0, min_items - np);
        int rest_max = max_items < 0 ? -1 : std::max(0, max_items - np);
        if (!item.empty() && rest_max != 0) {
            std::string comma_item = add_rule(name + "-more", "\",\" space " + item);
            if (np > 0) {
                std::string r = repeat(comma_item, rest_min, rest_max);
                if (!r.empty())
                    seq += " " + r;
            } else if (rest_min > 0) {
                std::string r = repeat(comma_item, rest_min - 1, rest_max < 0 ? -1 : rest_max - 1);
                seq = item + (r.empty() ? "" : " " + r);
            } else {
                std::string r = repeat(comma_item, 0, rest_max < 0 ? -1 : rest_max - 1);
                seq = "( " + item + (r.empty() ? "" : " " + r) + " )?";
            }
        } else if (rest_min > 0)
            throw SchemaError("minItems exceeds the number of allowed items");
        std::string body = "\"[\" space " + (seq.empty() ? "" : seq + " ") + "\"]\" space";
        return add_rule(name, body);
    }

    std::string visit_string(const json &s, const std::string &name) {
        int min_len = count_or(s, "minLength", 0);
        int max_len = count_or(s, "maxLength", -1);
        if (max_len >= 0 && max_len < min_len)
            throw SchemaError("maxLength is smaller than minLength");
        if (s.contains("format")) {
            if (!s["format"].is_string())
                throw SchemaError("'format' must be a string");
            std::string f = s["format"], fmt;
            if (f == "date") {
                fmt = "date-fmt";
                used_format[0] = true;
            } else if (f == "time") {
                fmt = "time-fmt";
                used_format[1] = true;
            } else if (f == "date-time") {
                fmt = "date-fmt \"T\" time-fmt";
                used_format[0] = used_format[1] = true;
            } else if (f == "uuid") {
                fmt = "uuid-fmt";
                used_format[2] = true;
            }
            // other formats (email, uri, hostname, ...) are annotations only
            if (!fmt.empty()) {
                if (s.contains("minLength") || s.contains("maxLength"))
                    unsupported("minLength/maxLength combined with format '" + f + "'");
                return add_rule(name, "\"\\\"\" " + fmt + " \"\\\"\" space");
            }
        }
        if (min_len == 0 && max_len < 0)
            return "string";
        std::string chars = repeat("char", min_len, max_len);
        return add_rule(name, "\"\\\"\" " + (chars.empty() ? "" : chars + " ") + "\"\\\"\" space");
    }

    // Numeric bound as an exact integer: integer schemas round toward the
    // admitted side; number schemas need integer-valued limits.
    static Bound bound(const json &s, const char *inclusive_key, const char *exclusive_key, bool integer,
                       bool lower) {
        Bound b;
        auto take = [&](const json &v, bool inclusive, const char *key) {
            if (!v.is_number())
                throw SchemaError(std::string("'") + key + "' must be a number");
            double d = v.get<double>();
            if (!std::isfinite(d) || std::fabs(d) > 9.0e15)
                unsupported(std::string("'") + key + "' beyond 16 digits");
            long long value;
            if (integer) {
                // x integer: x >= d <=> x >= ceil(d); x > d <=> x >= floor(d) + 1
                double r = lower ? (inclusive ? std::ceil(d) : std::floor(d) + 1)
                                 : (inclusive ? std::floor(d) : std::ceil(d) - 1);
                value = static_cast<long long>(r);
                inclusive = true;
            } else {
                if (d != std::floor(d))
                    unsupported(std::string("'") + key + "' with a non-integer value on a number");
                value = static_cast<long long>(d);
            }
            if (b.has) { // keep the tighter bound
                bool tighter = lower ? (value > b.value || (value == b.value && !inclusive))
                                     : (value < b.value || (value == b.value && !inclusive));
                if (!tighter)
                    return;
            }
            b = {true, value, inclusive};
        };
        bool draft4_exclusive = s.contains(exclusive_key) && s[exclusive_key].is_boolean();
        if (s.contains(inclusive_key))
            take(s[inclusive_key], !(draft4_exclusive && s[exclusive_key].get<bool>()), inclusive_key);
        if (s.contains(exclusive_key) && !draft4_exclusive)
            take(s[exclusive_key], false, exclusive_key);
        return b;
    }

    std::string visit_number(const json &s, const std::string &name, bool integer) {
        Bound lo = bound(s, "minimum", "exclusiveMinimum", integer, true);
        Bound hi = bound(s, "maximum", "exclusiveMaximum", integer, false);
        if (!lo.has && !hi.has)
            return integer ? "integer" : "number";
        if (integer)
            return add_rule(name, integer_range_gbnf(lo.value, lo.has, hi.value, hi.has) + " space");
        return add_rule(name, number_range(lo, hi) + " space");
    }

    std::string visit_type(const json &s, const std::string &type, const std::string &name) {
        if (type == "object")
            return visit_object(s, name);
        if (type == "array")
            return visit_array(s, name);
        if (type == "string")
            return visit_string(s, name);
        if (type == "integer")
            return visit_number(s, name, true);
        if (type == "number")
            return visit_number(s, name, false);
        if (type == "boolean")
            return "boolean";
        if (type == "null")
            return "null";
        throw SchemaError("unknown type '" + type + "'");
    }

    struct DepthGuard {
        int &d;
        explicit DepthGuard(int &depth) : d(depth) {
            if (++d > max_depth)
                throw SchemaError("schema nesting deeper than " + std::to_string(max_depth) + " levels");
        }
        ~DepthGuard() {
            --d;
        }
    };

    std::string visit(const json &s, const std::string &name) {
        DepthGuard guard(depth);
        if (s.is_boolean()) {
            if (s.get<bool>())
                return "value";
            unsupported("schema 'false' (nothing can match)");
        }
        if (!s.is_object())
            throw SchemaError("schema must be an object or boolean");
        check_keywords(s);
        bool nullable = false;
        if (s.contains("nullable")) {
            if (!s["nullable"].is_boolean())
                throw SchemaError("'nullable' must be a boolean");
            nullable = s["nullable"].get<bool>();
        }
        std::string rule = visit_inner(s, name);
        if (nullable && rule != "value" && rule != "null")
            return add_rule(name + "-nullable", rule + " | null");
        return rule;
    }

    std::string visit_inner(const json &s, const std::string &name) {
        if (s.contains("$ref")) {
            check_alone(s, "$ref", false);
            if (!s["$ref"].is_string())
                throw SchemaError("'$ref' must be a string");
            std::string ref = s["$ref"];
            auto it = ref_rules.find(ref);
            if (it != ref_rules.end())
                return it->second;
            // Reserve the name first so recursive schemas terminate.
            std::string rname =
                unique_name(sanitize(ref == "#" ? "root-ref" : ref.substr(ref.find_last_of('/') + 1)));
            ref_rules[ref] = rname;
            const json *target = resolve_ref(ref);
            std::string inner = visit(*target, rname + "-def");
            rules.push_back({rname, inner});
            account(rname, inner);
            return rname;
        }
        for (const char *comb : {"anyOf", "oneOf"}) {
            if (!s.contains(comb))
                continue;
            check_alone(s, comb, false);
            const json &alts = s[comb];
            if (!alts.is_array() || alts.empty())
                throw SchemaError(std::string("'") + comb + "' must be a non-empty array");
            std::string body;
            for (size_t i = 0; i < alts.size(); ++i)
                body += (i ? " | " : "") + visit(alts[i], name + "-" + std::to_string(i));
            return add_rule(name, body);
        }
        if (s.contains("allOf")) {
            check_alone(s, "allOf", false);
            const json &alts = s["allOf"];
            if (!alts.is_array() || alts.size() != 1)
                unsupported("allOf with more than one schema");
            return visit(alts[0], name);
        }
        if (s.contains("const")) {
            check_alone(s, "const", true);
            return add_rule(name, json_literal(s["const"]) + " space");
        }
        if (s.contains("enum")) {
            check_alone(s, "enum", true);
            const json &en = s["enum"];
            if (!en.is_array() || en.empty())
                throw SchemaError("'enum' must be a non-empty array");
            std::string body;
            for (size_t i = 0; i < en.size(); ++i)
                body += (i ? " | " : "") + json_literal(en[i]);
            return add_rule(name, "( " + body + " ) space");
        }
        if (s.contains("type")) {
            const json &t = s["type"];
            if (t.is_string())
                return visit_type(s, t.get<std::string>(), name);
            if (t.is_array()) {
                if (t.empty())
                    throw SchemaError("'type' array is empty");
                std::string body;
                for (size_t i = 0; i < t.size(); ++i) {
                    if (!t[i].is_string())
                        throw SchemaError("'type' entries must be strings");
                    body += (i ? " | " : "") +
                            visit_type(s, t[i].get<std::string>(), name + "-" + t[i].get<std::string>());
                }
                return add_rule(name, body);
            }
            throw SchemaError("'type' must be a string or array");
        }
        if (s.contains("properties") || s.contains("required") || s.contains("additionalProperties"))
            return visit_object(s, name);
        if (s.contains("items") || s.contains("prefixItems"))
            return visit_array(s, name);
        for (const char *k : {"minLength", "maxLength", "format", "minimum", "maximum", "exclusiveMinimum",
                              "exclusiveMaximum", "minItems", "maxItems"})
            if (s.contains(k))
                unsupported(std::string("'") + k + "' without 'type'");
        return "value"; // {} -- anything
    }

    // Rule matching `schema` (a further root: its references resolve within it), added to this grammar.
    std::string compile(const json &schema, const std::string &hint) {
        root = &schema;
        ref_rules.clear(); // pointers are per root; rule names stay unique
        return visit(schema, hint);
    }
    std::string build() {
        return build_root(visit(*root, "root-value"));
    }
    std::string build_root(const std::string &root_body) {
        std::string out = "root ::= " + root_body + "\n";
        for (const auto &[n, b] : rules)
            out += n + " ::= " + b + "\n";
        out += primitives;
        if (used_format[0])
            out += "date-fmt ::= [0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [12] [0-9] | \"3\" [01] )\n";
        if (used_format[1])
            out += "time-fmt ::= ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{1,6} )? "
                   "( \"Z\" | [+-] ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )\n";
        if (used_format[2])
            out += "uuid-fmt ::= [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" "
                   "[0-9a-fA-F]{12}\n";
        return out;
    }
};

} // namespace

std::string json_object_grammar() {
    return std::string("root ::= object\n") + primitives;
}

std::string json_schema_grammar(const json &schema) {
    Compiler c(schema);
    return c.build();
}

std::string json_schemas_grammar(const std::vector<json> &schemas,
                                 const std::function<std::string(const std::vector<std::string> &)> &root) {
    const json empty = json::object();
    Compiler c(empty);
    std::vector<std::string> rules;
    for (size_t i = 0; i < schemas.size(); ++i)
        rules.push_back(c.compile(schemas[i], "schema-" + std::to_string(i)));
    return c.build_root(root(rules));
}

std::string gbnf_literal(const std::string &s) {
    return literal(s);
}

} // namespace behemoth::grammar
