#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"
#include "behemoth/json_schema.hpp"

#include <random>
#include <algorithm>

namespace behemoth {
StructuredOutput parse_structured(const nlohmann::ordered_json &j, bool tool_constraint) {
    using ojson = nlohmann::ordered_json;
    StructuredOutput out;
    if (j.contains("response_format") && !j.at("response_format").is_null()) {
        const ojson &rf = j.at("response_format");
        if (!rf.is_object() || !rf.contains("type") || !rf.at("type").is_string())
            throw RequestError("response_format must be an object with a string 'type'", "response_format");
        const std::string type = rf.at("type");
        if (type == "json_object") {
            out.kind = StructuredOutput::JsonObject;
            out.gbnf = grammar::json_object_grammar();
            out.json = true;
        } else if (type == "json_schema") {
            if (!rf.contains("json_schema") || !rf.at("json_schema").is_object())
                throw RequestError("response_format.json_schema must be an object", "response_format.json_schema");
            const ojson &js = rf.at("json_schema");
            if (js.contains("name")) {
                if (!js.at("name").is_string())
                    throw RequestError("response_format.json_schema.name must be a string",
                                       "response_format.json_schema.name");
                out.name = js.at("name");
                if (out.name.empty() || out.name.size() > 64 ||
                    out.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
                        std::string::npos)
                    throw RequestError("response_format.json_schema.name must match ^[a-zA-Z0-9_-]{1,64}$",
                                       "response_format.json_schema.name");
            }
            if (js.contains("strict") && !js.at("strict").is_null()) {
                if (!js.at("strict").is_boolean())
                    throw RequestError("response_format.json_schema.strict must be a boolean",
                                       "response_format.json_schema.strict");
                out.strict = js.at("strict");
            }
            if (!js.contains("schema") || !(js.at("schema").is_object() || js.at("schema").is_boolean()))
                throw RequestError("response_format.json_schema.schema must be a JSON Schema object",
                                   "response_format.json_schema.schema");
            try {
                out.gbnf = grammar::json_schema_grammar(js.at("schema"));
            } catch (const grammar::SchemaError &e) {
                throw RequestError(std::string("Invalid response_format.json_schema.schema: ") + e.what(),
                                   "response_format.json_schema.schema");
            }
            out.kind = StructuredOutput::JsonSchema;
            out.json = true;
        } else if (type != "text")
            throw RequestError("response_format.type must be text, json_object or json_schema",
                               "response_format.type");
    }
    if (j.contains("grammar") && !j.at("grammar").is_null()) {
        if (out.kind != StructuredOutput::None)
            throw RequestError("Specify either response_format or grammar", "grammar");
        if (!j.at("grammar").is_string() || j.at("grammar").get<std::string>().empty())
            throw RequestError("grammar must be a nonempty GBNF string", "grammar");
        if (j.at("grammar").get<std::string>().size() > (1u << 20))
            throw RequestError("grammar is larger than 1 MiB", "grammar");
        out.kind = StructuredOutput::Grammar;
        out.gbnf = j.at("grammar");
    }
    if (j.contains("reasoning_effort") && !j.at("reasoning_effort").is_null()) {
        const auto &e = j.at("reasoning_effort");
        static const std::vector<std::string> levels = {"none", "minimal", "low", "medium", "high"};
        if (!e.is_string() || std::find(levels.begin(), levels.end(), e.get<std::string>()) == levels.end())
            throw RequestError("reasoning_effort must be none, minimal, low, medium or high", "reasoning_effort");
        if (e == "none") {
            if (out.kind == StructuredOutput::None && !tool_constraint)
                throw RequestError("reasoning_effort=none is only enforced together with response_format "
                                   "json_object/json_schema or grammar",
                                   "reasoning_effort");
            out.allow_think = false;
        }
    }
    return out;
}
namespace {
ChatBackend &backend() {
    static ChatBackend b;
    return b;
}
} // namespace
void set_chat_backend(ChatBackend b) { backend() = std::move(b); }
const ChatBackend &chat_backend() { return backend(); }
const std::string &model_name() {
    static const std::string fallback = model_id;
    return backend().model_id.empty() ? fallback : backend().model_id;
}
ChatRequest parse_chat(const nlohmann::json &j, const nlohmann::ordered_json *ordered) {
    if (!j.is_object())
        throw std::invalid_argument("Request must be a JSON object");
    if (j.value("model", model_name()) != model_name())
        throw std::invalid_argument("Unknown model");
    if (j.value("n", 1) != 1)
        throw std::invalid_argument("Only n=1 is supported");
    const bool tools = backend().tools.supported() && backend().render_tools;
    for (const char *key : {"tools", "tool_choice", "functions", "function_call", "logprobs"})
        if (j.contains(key) && (!tools || (std::string_view(key) != "tools" && std::string_view(key) != "tool_choice")))
            throw std::invalid_argument(std::string("Unsupported parameter: ") + key);
    auto messages = j.at("messages");
    if (!messages.is_array() || messages.empty())
        throw std::invalid_argument("messages must be a nonempty array");
    ChatRequest r;
    bool tool_messages = false; // the conversation has tool results or assistant tool calls
    // tool definitions and call arguments keep their declared key order (ordered), as the model is shown them
    const nlohmann::ordered_json own = tools && !ordered ? nlohmann::ordered_json(j) : nlohmann::ordered_json();
    const auto &o = ordered ? *ordered : own;
    if (tools) {
        r.tools = parse_tools(o);
        for (const auto &m : messages)
            tool_messages |= m.is_object() && (m.value("role", "") == "tool" || m.contains("tool_calls"));
        if (r.tools.active())
            try {
                r.tool_grammar = tool_call_grammar(r.tools, backend().tools);
            } catch (const grammar::SchemaError &e) {
                throw RequestError(std::string("Invalid tool parameters schema: ") + e.what(), "tools");
            }
    }
    if (tools && (r.tools.active() || tool_messages)) {
        r.rendered = backend().render_tools(o.at("messages"), r.tools.active() ? r.tools.tools : nlohmann::ordered_json());
    } else if (backend().render) {
        r.rendered = backend().render(messages);
    } else {
        r.rendered = "<s>";
        for (const auto &m : messages) {
            std::string role = m.at("role").get<std::string>();
            if (!m.at("content").is_string())
                throw std::invalid_argument("Only text message content is supported");
            auto text = m.at("content").get<std::string>();
            if (role == "user")
                r.rendered += "[INST] " + text + "[/INST]";
            else if (role == "system")
                r.rendered += "[SYSTEM_PROMPT] " + text + "[/SYSTEM_PROMPT]";
            else if (role == "assistant")
                r.rendered += " " + text + "</s>";
            else
                throw std::invalid_argument("Only system, user and assistant roles are supported");
        }
    }
    if (j.contains("max_tokens") && j.contains("max_completion_tokens"))
        throw std::invalid_argument("Specify one output-token limit");
    auto maximum = j.contains("max_completion_tokens") ? j.at("max_completion_tokens")
                                                       : j.value("max_tokens", nlohmann::json(256));
    if (!maximum.is_number_integer() || maximum.get<int64_t>() < 1 ||
        maximum.get<int64_t>() > 131072)
        throw std::invalid_argument("Output-token limit must be 1..131072");
    r.max_new = maximum.get<size_t>();
    r.stream = j.value("stream", false);
    if (j.contains("stream_options") && j.at("stream_options").is_object())
        r.include_usage = j.at("stream_options").value("include_usage", false);
    r.sampling.temperature = j.value("temperature", 0.f);
    r.sampling.top_k = j.value("top_k", 0);
    r.sampling.top_p = j.value("top_p", 1.f);
    r.sampling.repetition_penalty = j.value("repetition_penalty", 1.f);
    r.sampling.frequency_penalty = j.value("frequency_penalty", 0.f);
    r.sampling.presence_penalty = j.value("presence_penalty", 0.f);
    r.sampling.min_p = j.value("min_p", 0.f);
    r.sampling.penalty_range = j.value("repetition_penalty_range", j.value("repeat_last_n", 0));
    if (j.contains("logit_bias") && !j.at("logit_bias").is_null()) {
        if (!j.at("logit_bias").is_object())
            throw std::invalid_argument("logit_bias must map token ids to biases");
        for (auto it = j.at("logit_bias").begin(); it != j.at("logit_bias").end(); ++it) {
            size_t used = 0;
            int id = -1;
            try {
                id = std::stoi(it.key(), &used);
            } catch (...) {
            }
            if (used != it.key().size() || !it.value().is_number())
                throw std::invalid_argument("logit_bias keys must be token ids and values numbers");
            r.sampling.logit_bias.push_back({id, it.value().get<float>()});
        }
    }
    // Without an explicit seed every request draws a fresh one; a fixed default made
    // regenerations of the same prompt return the identical sample.
    r.sampling.seed = j.contains("seed") ? j.at("seed").get<uint64_t>()
                                         : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    validate_sampling(r.sampling);
    if (j.contains("stop")) {
        if (j.at("stop").is_string())
            r.stops.push_back(j.at("stop").get<std::string>());
        else
            r.stops = j.at("stop").get<std::vector<std::string>>();
        if (r.stops.size() > 16)
            throw std::invalid_argument("At most 16 stop strings are supported");
        for (const auto &s : r.stops)
            if (s.empty())
                throw std::invalid_argument("Stop strings must be nonempty");
    }
    const bool tool_constraint = r.tools.choice == ToolRequest::Required || r.tools.choice == ToolRequest::Named;
    r.structured = ordered ? parse_structured(*ordered, tool_constraint)
                           : parse_structured(nlohmann::ordered_json(j), tool_constraint);
    if (!backend().think && (!j.contains("reasoning_effort") || j.at("reasoning_effort").is_null()))
        r.structured.allow_think = false;
    if (r.tools.active() && r.structured.kind != StructuredOutput::None)
        throw RequestError("response_format and grammar cannot be combined with tools (tool_choice must be none)",
                           "tool_choice");
    return r;
}
} // namespace behemoth
