#include "behemoth/tools.hpp"
#include "behemoth/gbnf.hpp"
#include "behemoth/json_schema.hpp"
#include "behemoth/protocol.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <random>
#include <set>

namespace behemoth {
namespace {
using ojson = nlohmann::ordered_json;

bool valid_name(const std::string &n) {
    return !n.empty() && n.size() <= 64 &&
           n.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}
bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
std::string trim(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b]))
        ++b;
    while (e > b && is_space(s[e - 1]))
        --e;
    return std::string(s.substr(b, e - b));
}
// Length of the longest suffix of `text` that is a proper prefix of `marker` (text that may still become it).
size_t partial_suffix(std::string_view text, std::string_view marker) {
    for (size_t n = std::min(marker.size() - 1, text.size()); n > 0; --n)
        if (text.substr(text.size() - n) == marker.substr(0, n))
            return n;
    return 0;
}
// Message content as text: a string, null (assistant turns that only call tools) or text blocks.
std::string text_content(const ojson &m, bool allow_null) {
    if (!m.contains("content") || m.at("content").is_null()) {
        if (allow_null)
            return "";
        throw std::invalid_argument("message content is required");
    }
    const auto &c = m.at("content");
    if (c.is_string())
        return c.get<std::string>();
    if (!c.is_array())
        throw std::invalid_argument("message content must be a string or an array of text blocks");
    std::string out;
    for (const auto &block : c) {
        if (!block.is_object() || block.value("type", "") != "text" || !block.contains("text") || !block.at("text").is_string())
            throw std::invalid_argument("Only text message content is supported (no images or other content parts)");
        out += block.at("text").get<std::string>();
    }
    return out;
}
// `v` with object keys in the order the schema declares its properties (the grammar's order), recursively.
ojson schema_order(const nlohmann::json &v, const ojson &s) {
    const ojson none = ojson::object();
    if (v.is_object()) {
        ojson o = ojson::object();
        const ojson &props = s.is_object() && s.contains("properties") && s.at("properties").is_object() ? s.at("properties") : none;
        for (auto it = props.begin(); it != props.end(); ++it)
            if (v.contains(it.key()))
                o[it.key()] = schema_order(v.at(it.key()), it.value());
        const ojson &extra = s.is_object() && s.contains("additionalProperties") && s.at("additionalProperties").is_object()
                                 ? s.at("additionalProperties")
                                 : none;
        for (auto it = v.begin(); it != v.end(); ++it)
            if (!props.contains(it.key()))
                o[it.key()] = schema_order(it.value(), extra);
        return o;
    }
    if (v.is_array()) {
        ojson a = ojson::array();
        const ojson &items = s.is_object() && s.contains("items") && s.at("items").is_object() ? s.at("items") : none;
        for (const auto &x : v)
            a.push_back(schema_order(x, items));
        return a;
    }
    return ojson(v);
}
} // namespace

bool schema_accepts(const ojson &schema, const nlohmann::json &value) {
    try {
        grammar::Grammar g = grammar::ParseGbnf(grammar::json_schema_grammar(schema));
        grammar::Matcher m(g);
        return m.Matches(schema_order(value, schema).dump());
    } catch (const std::exception &) {
        return false;
    }
}

std::optional<std::pair<size_t, nlohmann::json>> trailing_json(std::string_view text) {
    while (!text.empty() && is_space(text.back()))
        text.remove_suffix(1);
    if (text.ends_with("```")) {
        if (text.size() < 6)
            return std::nullopt;
        const size_t open = text.rfind("```", text.size() - 6);
        if (open == std::string_view::npos)
            return std::nullopt;
        std::string_view inner = text.substr(open + 3, text.size() - 3 - (open + 3));
        size_t tag = 0; // language tag (```json)
        while (tag < inner.size() && std::isalnum(static_cast<unsigned char>(inner[tag])))
            ++tag;
        const auto v = nlohmann::json::parse(inner.substr(tag), nullptr, false);
        if (v.is_discarded() || !v.is_structured())
            return std::nullopt;
        return std::pair{open, v};
    }
    int tries = 0;
    for (size_t i = text.find_first_of("{["); i != std::string_view::npos && tries < 64; i = text.find_first_of("{[", i + 1), ++tries) {
        const auto v = nlohmann::json::parse(text.substr(i), nullptr, false);
        if (!v.is_discarded() && v.is_structured())
            return std::pair{i, v};
    }
    return std::nullopt;
}

ToolRequest parse_tools(const ojson &j) {
    ToolRequest r;
    std::set<std::string> names;
    if (j.contains("tools") && !j.at("tools").is_null()) {
        if (!j.at("tools").is_array())
            throw RequestError("tools must be an array", "tools");
        if (j.at("tools").size() > 128)
            throw RequestError("At most 128 tools are supported", "tools");
        for (size_t i = 0; i < j.at("tools").size(); ++i) {
            const auto &t = j.at("tools")[i];
            const std::string at = "tools[" + std::to_string(i) + "]";
            if (!t.is_object() || t.value("type", "") != "function" || !t.contains("function") || !t.at("function").is_object())
                throw RequestError(at + " must be {\"type\": \"function\", \"function\": {...}}", at);
            const auto &f = t.at("function");
            if (!f.contains("name") || !f.at("name").is_string() || !valid_name(f.at("name").get<std::string>()))
                throw RequestError(at + ".function.name must match ^[a-zA-Z0-9_-]{1,64}$", at + ".function.name");
            const std::string name = f.at("name");
            if (!names.insert(name).second)
                throw RequestError("Duplicate tool name: " + name, at + ".function.name");
            ojson fn = {{"name", name}};
            if (f.contains("description") && !f.at("description").is_null()) {
                if (!f.at("description").is_string())
                    throw RequestError(at + ".function.description must be a string", at + ".function.description");
                fn["description"] = f.at("description");
            }
            if (f.contains("parameters") && !f.at("parameters").is_null()) {
                if (!f.at("parameters").is_object())
                    throw RequestError(at + ".function.parameters must be a JSON Schema object", at + ".function.parameters");
                fn["parameters"] = f.at("parameters");
            } else
                fn["parameters"] = {{"type", "object"}, {"properties", ojson::object()}};
            r.tools.push_back({{"type", "function"}, {"function", std::move(fn)}});
        }
    }
    if (j.contains("parallel_tool_calls") && !j.at("parallel_tool_calls").is_null()) {
        if (!j.at("parallel_tool_calls").is_boolean())
            throw RequestError("parallel_tool_calls must be a boolean", "parallel_tool_calls");
        r.parallel = j.at("parallel_tool_calls");
    }
    r.choice = r.tools.empty() ? ToolRequest::None : ToolRequest::Auto;
    if (j.contains("tool_choice") && !j.at("tool_choice").is_null()) {
        const auto &c = j.at("tool_choice");
        if (c.is_string()) {
            const std::string s = c;
            if (s == "none")
                r.choice = ToolRequest::None;
            else if (s == "auto" || s == "required")
                r.choice = s == "auto" ? ToolRequest::Auto : ToolRequest::Required;
            else
                throw RequestError("tool_choice must be none, auto, required or a function", "tool_choice");
        } else if (c.is_object() && c.value("type", "") == "function" && c.contains("function") &&
                   c.at("function").is_object() && c.at("function").contains("name") &&
                   c.at("function").at("name").is_string()) {
            r.choice = ToolRequest::Named;
            r.name = c.at("function").at("name");
            if (!names.contains(r.name))
                throw RequestError("tool_choice names a function that is not in tools: " + r.name, "tool_choice");
        } else
            throw RequestError("tool_choice must be none, auto, required or {\"type\": \"function\", \"function\": {\"name\": ...}}",
                               "tool_choice");
        if (r.choice != ToolRequest::None && r.tools.empty())
            throw RequestError("tool_choice requires tools", "tool_choice");
    }
    return r;
}

namespace {
// A property schema whose values GLM writes raw (a plain string; an enum of strings becomes literal alternatives).
bool glm_raw_string(const ojson &s) {
    return s.is_object() && s.value("type", ojson()) == "string";
}
std::string glm_tool_call_grammar(const ToolRequest &r) {
    struct Prop { std::string key; const ojson *schema; bool required; int rule; };
    struct Tool { std::string name; std::vector<Prop> props; };
    std::vector<Tool> tools;
    std::vector<ojson> schemas; // the non-string property schemas, compiled to JSON rules
    for (const auto &t : r.tools) {
        const auto &fn = t.at("function");
        if (r.choice == ToolRequest::Named && fn.at("name") != r.name)
            continue;
        Tool tool{fn.at("name"), {}};
        const auto &params = fn.at("parameters");
        std::set<std::string> required;
        if (params.contains("required") && params.at("required").is_array())
            for (const auto &k : params.at("required"))
                if (k.is_string())
                    required.insert(k.get<std::string>());
        if (params.contains("properties") && params.at("properties").is_object())
            for (auto it = params.at("properties").begin(); it != params.at("properties").end(); ++it) {
                Prop p{it.key(), &it.value(), required.count(it.key()) > 0, -1};
                if (!glm_raw_string(it.value())) {
                    p.rule = int(schemas.size());
                    schemas.push_back(it.value());
                }
                tool.props.push_back(p);
            }
        tools.push_back(std::move(tool));
    }
    if (tools.empty())
        throw std::invalid_argument("No tools to call");
    if (schemas.empty())
        schemas.push_back(ojson::object()); // json_schemas_grammar needs one schema; this rule stays unused
    return grammar::json_schemas_grammar(schemas, [&](const std::vector<std::string> &rules) {
        std::string call = "(";
        for (size_t i = 0; i < tools.size(); ++i) {
            call += (i ? " | " : " ") + grammar::gbnf_literal(std::string("<tool_call>") + tools[i].name);
            for (const auto &p : tools[i].props) {
                std::string value;
                if (p.rule >= 0)
                    value = rules[size_t(p.rule)];
                else if (p.schema->contains("enum") && p.schema->at("enum").is_array()) {
                    value = "(";
                    bool first = true;
                    for (const auto &e : p.schema->at("enum"))
                        if (e.is_string()) {
                            value += (first ? " " : " | ") + grammar::gbnf_literal(e.get<std::string>());
                            first = false;
                        }
                    value += " )";
                } else
                    value = "( [^<] | \"<\" [^/] )*"; // raw text up to </arg_value>
                const std::string kv = grammar::gbnf_literal(std::string(glm_key_open) + p.key + glm_key_close + glm_value_open) + " " +
                                       value + " " + grammar::gbnf_literal(glm_value_close);
                call += " " + (p.required ? kv : "( " + kv + " )?");
            }
            call += " " + grammar::gbnf_literal(glm_call_end);
        }
        call += " )";
        return r.parallel ? call + " ( " + call + " )*" : call; // back to back: the grammar runs in JSON mode (no whitespace after a complete answer)
    });
}
} // namespace

std::string tool_call_grammar(const ToolRequest &r, const ToolFormat &f) {
    if (f.glm)
        return glm_tool_call_grammar(r);
    std::vector<ojson> schemas;
    std::vector<std::string> names;
    for (const auto &t : r.tools) {
        const auto &fn = t.at("function");
        if (r.choice == ToolRequest::Named && fn.at("name") != r.name)
            continue;
        names.push_back(fn.at("name"));
        schemas.push_back(fn.at("parameters"));
    }
    if (names.empty())
        throw std::invalid_argument("No tools to call");
    return grammar::json_schemas_grammar(schemas, [&](const std::vector<std::string> &rules) {
        std::string call = grammar::gbnf_literal(f.calls) + " (";
        for (size_t i = 0; i < rules.size(); ++i)
            call += (i ? " | " : " ") + grammar::gbnf_literal(names[i] + f.args) + " " + rules[i];
        call += " )";
        return r.parallel ? call + " ( " + call + " )*" : call;
    });
}

std::string python_json(const ojson &v) {
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            out += (first ? "" : ", ") + ojson(it.key()).dump(-1, ' ', false, ojson::error_handler_t::replace) + ": " +
                   python_json(it.value());
            first = false;
        }
        return out + "}";
    }
    if (v.is_array()) {
        std::string out = "[";
        for (size_t i = 0; i < v.size(); ++i)
            out += (i ? ", " : "") + python_json(v[i]);
        return out + "]";
    }
    return v.dump(-1, ' ', false, ojson::error_handler_t::replace);
}

std::string render_mistral_tools(const ojson &messages, const ojson &tools, const ToolFormat &f, const std::string &bos,
                                 const std::string &eos, const std::string &system) {
    if (!messages.is_array() || messages.empty())
        throw std::invalid_argument("messages must be a nonempty array");
    std::string out = bos;
    size_t begin = 0, last_user = messages.size();
    std::string prompt = system;
    if (messages[0].is_object() && messages[0].value("role", "") == "system") {
        prompt = text_content(messages[0], false);
        begin = 1;
    }
    const bool offered = tools.is_array() && !tools.empty();
    if (offered && !f.tools_first) // fine-tunes of tool-trained bases may need telling (tool_instruction)
        prompt += (prompt.empty() ? "" : "\n\n") + tool_instruction(f);
    if (!prompt.empty())
        out += "[SYSTEM_PROMPT]" + prompt + "[/SYSTEM_PROMPT]";
    if (offered && f.tools_first)
        out += "[AVAILABLE_TOOLS]" + python_json(tools) + "[/AVAILABLE_TOOLS]";
    else
        for (size_t i = begin; i < messages.size(); ++i)
            if (messages[i].is_object() && messages[i].value("role", "") == "user")
                last_user = i;
    for (size_t i = begin; i < messages.size(); ++i) {
        const auto &m = messages[i];
        if (!m.is_object() || !m.contains("role") || !m.at("role").is_string())
            throw std::invalid_argument("every message needs a string role");
        const std::string role = m.at("role");
        if (role == "user") {
            if (i == last_user && tools.is_array() && !tools.empty())
                out += "[AVAILABLE_TOOLS]" + python_json(tools) + "[/AVAILABLE_TOOLS]";
            out += "[INST]" + text_content(m, false) + "[/INST]";
        } else if (role == "system")
            out += "[SYSTEM_PROMPT]" + text_content(m, false) + "[/SYSTEM_PROMPT]";
        else if (role == "assistant") {
            const bool calls = m.contains("tool_calls") && !m.at("tool_calls").is_null();
            out += text_content(m, calls);
            if (calls) {
                if (!m.at("tool_calls").is_array())
                    throw std::invalid_argument("tool_calls must be an array");
                for (const auto &c : m.at("tool_calls")) {
                    if (!c.is_object() || !c.contains("function") || !c.at("function").is_object() ||
                        !c.at("function").contains("name") || !c.at("function").at("name").is_string())
                        throw std::invalid_argument("tool_calls entries need function.name");
                    const auto &fn = c.at("function");
                    std::string arguments = "{}";
                    if (fn.contains("arguments") && fn.at("arguments").is_string()) {
                        const std::string raw = fn.at("arguments");
                        const auto parsed = ojson::parse(raw, nullptr, false);
                        arguments = parsed.is_discarded() ? raw : python_json(parsed);
                    } else if (fn.contains("arguments") && !fn.at("arguments").is_null())
                        arguments = python_json(fn.at("arguments"));
                    out += f.calls + fn.at("name").get<std::string>() + f.args + arguments;
                }
            }
            out += eos;
        } else if (role == "tool")
            out += "[TOOL_RESULTS]" + text_content(m, false) + "[/TOOL_RESULTS]";
        else
            throw std::invalid_argument("Only system, user, assistant and tool roles are supported");
    }
    return out;
}

std::string tool_instruction(const ToolFormat &f) {
    return "You can call the functions in the available tools. Call one whenever the request needs it: current or "
           "real-time information (weather, time, news, prices), lookups, calculations, or actions you cannot do "
           "yourself; never say you have no access to something a function provides. To call a function, reply with "
           "only " + f.calls + "function_name" + f.args + "{\"argument\": \"value\"} (the arguments as one JSON "
           "object; one " + f.calls + " per call; no other text, no Markdown, no code block). The results come back "
           "as tool results; then answer from them. If no function fits, answer normally.";
}

nlohmann::json ToolCall::openai() const {
    return {{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", arguments}}}};
}

ToolCallParser::ToolCallParser(ToolFormat format, const ToolRequest *fallback) : format_(std::move(format)) {
    if (fallback)
        tools_ = fallback->tools; // GLM values are typed by the tool's parameters schema
    fallback_ = fallback && fallback->choice == ToolRequest::Auto;
}

// GLM: a value is raw text when the property's schema is a string, else JSON (raw text when it does not parse).
nlohmann::json ToolCallParser::glm_value(const std::string &key, const std::string &text) const {
    for (const auto &t : tools_) {
        const auto &fn = t.at("function");
        if (calls_.empty() || fn.at("name") != calls_.back().name)
            continue;
        const auto &params = fn.at("parameters");
        if (params.contains("properties") && params.at("properties").contains(key) && glm_raw_string(params.at("properties").at(key)))
            return text;
    }
    auto parsed = nlohmann::json::parse(text, nullptr, false);
    return parsed.is_discarded() ? nlohmann::json(text) : parsed;
}

// One step of a GLM call: <tool_call> was consumed; name, then <arg_key>k</arg_key><arg_value>v</arg_value> pairs, then
// </tool_call>. The arguments go out as one JSON fragment when the call is complete. False: needs more text.
bool ToolCallParser::glm_step(Delta &d, bool finish) {
    auto close = [&]() {
        add_arguments(glm_args_.dump(), d);
        if (calls_.back().arguments.empty())
            calls_.back().arguments = "{}";
        glm_args_ = ojson::object();
        mode_ = Content;
    };
    auto skip_space = [&]() {
        size_t i = 0;
        while (i < buffer_.size() && is_space(buffer_[i]))
            ++i;
        buffer_.erase(0, i);
    };
    switch (mode_) {
    case GlmName: {
        const size_t key = buffer_.find(glm_key_open), end = buffer_.find(glm_call_end);
        const size_t at = std::min(key, end);
        if (at == std::string::npos) {
            if (finish && !trim(buffer_).empty()) {
                start_call(trim(buffer_), d);
                buffer_.clear();
                close();
            }
            return false;
        }
        start_call(trim(std::string_view(buffer_).substr(0, at)), d);
        glm_args_ = ojson::object();
        if (at == key) {
            buffer_.erase(0, at + std::string_view(glm_key_open).size());
            mode_ = GlmKey;
        } else {
            buffer_.erase(0, at + std::string_view(glm_call_end).size());
            close();
        }
        return true;
    }
    case GlmKey: {
        const size_t at = buffer_.find(glm_key_close);
        if (at == std::string::npos)
            break;
        glm_key_ = trim(std::string_view(buffer_).substr(0, at));
        buffer_.erase(0, at + std::string_view(glm_key_close).size());
        mode_ = GlmValue;
        return true;
    }
    case GlmValue: {
        skip_space();
        const std::string_view open = glm_value_open;
        if (buffer_.size() < open.size() && open.substr(0, buffer_.size()) == buffer_ && !finish)
            return false;
        if (buffer_.compare(0, open.size(), open) != 0) { // a key without a value: null
            glm_args_[glm_key_] = nullptr;
            mode_ = GlmArgs;
            return true;
        }
        const size_t at = buffer_.find(glm_value_close, open.size());
        if (at == std::string::npos)
            break;
        glm_args_[glm_key_] = glm_value(glm_key_, buffer_.substr(open.size(), at - open.size()));
        buffer_.erase(0, at + std::string_view(glm_value_close).size());
        mode_ = GlmArgs;
        return true;
    }
    case GlmArgs: {
        skip_space();
        for (const char *m : {glm_key_open, glm_call_end}) {
            const std::string_view marker = m;
            if (buffer_.compare(0, marker.size(), marker) == 0) {
                buffer_.erase(0, marker.size());
                if (m == glm_key_open)
                    mode_ = GlmKey;
                else
                    close();
                return true;
            }
            if (!finish && buffer_.size() < marker.size() && marker.substr(0, buffer_.size()) == buffer_)
                return false;
        }
        if (buffer_.empty())
            break;
        const size_t next = buffer_.find('<', 1); // stray text between pairs: dropped
        buffer_.erase(0, next == std::string::npos ? buffer_.size() : next);
        return !buffer_.empty();
    }
    default:
        break;
    }
    if (finish) { // the answer ended inside a call: keep what was parsed
        if (mode_ == GlmKey || mode_ == GlmValue)
            buffer_.clear();
        close();
    }
    return false;
}

// Fallback (tool_choice auto): the answer ends with a JSON tool call written as text, bare or in a ``` fence:
// {"name", "arguments" | "parameters"} (or {"function": {...}}) or a list of them, every name offered and every
// argument object matching that tool's parameters schema; or, with exactly one tool offered, an object matching its
// parameters. The text before it stays content.
bool ToolCallParser::convert_trailing(Delta &d) {
    const auto found = trailing_json(buffer_);
    if (!found)
        return false;
    const auto &[at, value] = *found;
    std::vector<std::pair<std::string, nlohmann::json>> calls;
    auto parameters = [&](const std::string &name) -> const ojson * {
        for (const auto &t : tools_)
            if (t.at("function").at("name") == name)
                return &t.at("function").at("parameters");
        return nullptr;
    };
    auto named = [&](const nlohmann::json &item) {
        const nlohmann::json *x = &item;
        if (x->is_object() && x->contains("function") && x->at("function").is_object())
            x = &x->at("function");
        if (!x->is_object() || !x->contains("name") || !x->at("name").is_string())
            return false;
        const ojson *schema = parameters(x->at("name"));
        if (!schema)
            return false;
        nlohmann::json args = nlohmann::json::object();
        for (const char *key : {"arguments", "parameters"})
            if (x->contains(key))
                args = x->at(key).is_string() ? nlohmann::json::parse(x->at(key).get<std::string>(), nullptr, false) : x->at(key);
        if (!args.is_object() || !schema_accepts(*schema, args))
            return false;
        calls.push_back({x->at("name").get<std::string>(), std::move(args)});
        return true;
    };
    bool ok = false;
    if (value.is_array()) {
        ok = !value.empty();
        for (const auto &item : value)
            ok = ok && named(item);
    } else if (!(ok = named(value)) && tools_.size() == 1 && value.is_object()) {
        calls.clear();
        const auto &fn = tools_[0].at("function");
        if ((ok = schema_accepts(fn.at("parameters"), value)))
            calls.push_back({fn.at("name").get<std::string>(), value});
    }
    if (!ok)
        return false;
    std::string_view prose = std::string_view(buffer_).substr(0, at);
    while (!prose.empty() && is_space(prose.back()))
        prose.remove_suffix(1);
    d.content += prose;
    for (auto &[name, args] : calls) {
        start_call(name, d);
        add_arguments(args.dump(), d);
    }
    buffer_.clear();
    return true;
}

std::string ToolCallParser::new_id() {
    static constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::string id = "call_";
    for (int i = 0; i < 24; ++i)
        id += alphabet[rng() % (sizeof alphabet - 1)];
    return id;
}

void ToolCallParser::start_call(std::string name, Delta &d) {
    calls_.push_back({new_id(), std::move(name), ""});
    d.tool_calls.push_back({{"index", calls_.size() - 1},
                            {"id", calls_.back().id},
                            {"type", "function"},
                            {"function", {{"name", calls_.back().name}, {"arguments", ""}}}});
}

void ToolCallParser::add_arguments(std::string_view text, Delta &d) {
    if (text.empty() || calls_.empty())
        return;
    // leading whitespace of the arguments is dropped (the parser trims the final arguments too)
    if (calls_.back().arguments.empty()) {
        while (!text.empty() && is_space(text.front()))
            text.remove_prefix(1);
        if (text.empty())
            return;
    }
    calls_.back().arguments += text;
    d.tool_calls.push_back({{"index", calls_.size() - 1}, {"function", {{"arguments", std::string(text)}}}});
}

// v7 form: [TOOL_CALLS][{"name": ..., "arguments": {...}}, ...] (or one object). Unparseable text stays content.
void ToolCallParser::finish_list(Delta &d) {
    const auto parsed = nlohmann::json::parse(list_, nullptr, false);
    std::vector<nlohmann::json> items;
    if (parsed.is_array())
        items.assign(parsed.begin(), parsed.end());
    else if (parsed.is_object())
        items.push_back(parsed);
    bool ok = !items.empty();
    for (const auto &it : items)
        ok = ok && it.is_object() && it.contains("name") && it.at("name").is_string();
    if (!ok) {
        d.content += format_.calls + list_;
        content_ += format_.calls + list_;
        list_.clear();
        return;
    }
    for (const auto &it : items) {
        start_call(it.at("name"), d);
        const auto args = it.contains("arguments") ? it.at("arguments") : nlohmann::json::object();
        add_arguments(args.is_string() ? args.get<std::string>() : args.dump(), d);
    }
    list_.clear();
}

ToolCallParser::Delta ToolCallParser::feed(std::string_view text, bool finish) {
    Delta d;
    buffer_ += text;
    for (bool progress = true; progress;) {
        progress = false;
        switch (mode_) {
        case Content: {
            // whitespace before anything else is held: dropped when the answer turns out to be calls only (and, GLM,
            // between consecutive calls)
            const bool leading = (content_.empty() && d.content.empty()) || (format_.glm && !calls_.empty());
            const size_t at = buffer_.find(format_.calls);
            if (at != std::string::npos) {
                if (!leading || !trim(std::string_view(buffer_).substr(0, at)).empty())
                    d.content += buffer_.substr(0, at);
                buffer_.erase(0, at + format_.calls.size());
                mode_ = format_.glm ? GlmName : Head;
                progress = true;
                break;
            }
            const size_t keep = finish ? 0 : partial_suffix(buffer_, format_.calls);
            if (!finish && leading && trim(std::string_view(buffer_).substr(0, buffer_.size() - keep)).empty())
                break;
            size_t emit = buffer_.size() - keep;
            if (fallback_) { // auto: from the first possible JSON / fence start on, the text is held to the end
                if (finish && convert_trailing(d))
                    break;
                if (!finish) { // and whitespace before it, which a converted call drops
                    emit = std::min(emit, buffer_.find_first_of("{[`"));
                    while (emit > 0 && is_space(buffer_[emit - 1]))
                        --emit;
                }
            }
            d.content += buffer_.substr(0, emit);
            buffer_.erase(0, emit);
            break;
        }
        case Head: {
            size_t i = 0;
            while (i < buffer_.size() && is_space(buffer_[i]))
                ++i;
            buffer_.erase(0, i);
            if (buffer_.empty())
                break;
            mode_ = buffer_[0] == '[' || buffer_[0] == '{' ? List : Name;
            progress = true;
            break;
        }
        case Name: {
            // name, then [ARGS] (or the arguments' "{" directly, as some fine-tunes write it)
            size_t at = format_.args.empty() ? std::string::npos : buffer_.find(format_.args);
            const size_t brace = buffer_.find('{');
            size_t skip = format_.args.size();
            if (brace != std::string::npos && (at == std::string::npos || brace < at)) {
                at = brace;
                skip = 0;
            }
            if (at == std::string::npos) {
                if (finish && !trim(buffer_).empty()) { // a name without arguments
                    start_call(trim(buffer_), d);
                    buffer_.clear();
                }
                break;
            }
            start_call(trim(std::string_view(buffer_).substr(0, at)), d);
            buffer_.erase(0, at + skip);
            mode_ = Args;
            progress = true;
            break;
        }
        case Args: {
            const size_t at = buffer_.find(format_.calls);
            if (at != std::string::npos) {
                add_arguments(std::string_view(buffer_).substr(0, at), d);
                buffer_.erase(0, at + format_.calls.size());
                mode_ = Head;
                progress = true;
                break;
            }
            const size_t keep = finish ? 0 : partial_suffix(buffer_, format_.calls);
            add_arguments(std::string_view(buffer_).substr(0, buffer_.size() - keep), d);
            buffer_.erase(0, buffer_.size() - keep);
            break;
        }
        case List:
            list_ += buffer_;
            buffer_.clear();
            if (finish)
                finish_list(d);
            break;
        case GlmName:
        case GlmArgs:
        case GlmKey:
        case GlmValue:
            progress = glm_step(d, finish);
            break;
        }
    }
    if (finish)
        for (auto &c : calls_) {
            c.arguments = trim(c.arguments);
            if (c.arguments.empty())
                c.arguments = "{}";
        }
    content_ += d.content;
    return d;
}

TriggeredConstraint::TriggeredConstraint(int trigger, std::shared_ptr<TokenConstraint> inner, size_t vocabulary)
    : trigger_(trigger), inner_(std::move(inner)), all_((vocabulary + 63) / 64, ~uint64_t(0)), count_(int(vocabulary)) {
    if (vocabulary & 63)
        all_.back() = (uint64_t(1) << (vocabulary & 63)) - 1;
}

bool TriggeredConstraint::accepts(int token) {
    return triggered_ ? inner_->accepts(token) : true;
}

bool TriggeredConstraint::accept_first(bool greedy) const {
    return triggered_ ? inner_->accept_first(greedy) : true;
}

TokenConstraint::Allowed TriggeredConstraint::allowed() {
    if (triggered_)
        return inner_->allowed();
    return {&all_, count_, -1};
}

void TriggeredConstraint::advance(int token) {
    if (!triggered_ && token == trigger_)
        triggered_ = true;
    if (triggered_)
        inner_->advance(token);
}

} // namespace behemoth
