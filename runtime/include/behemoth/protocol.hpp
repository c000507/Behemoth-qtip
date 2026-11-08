#pragma once
#include "scheduler.hpp"
#include "tools.hpp"
#include <functional>
#include <nlohmann/json.hpp>

namespace behemoth {
inline constexpr const char *model_id = "cgg507/Behemoth-R1-123B-v2-aqlm";
// Model-specific request handling for backends that do not serve the Behemoth/Mistral template: the id reported
// and accepted in "model", and the renderer turning the chat messages into the prompt text. Unset (default):
// model_id and the Mistral [INST] template above.
struct ChatBackend {
    std::string model_id;
    std::function<std::string(const nlohmann::json &messages)> render;
    // Tool calling (tools.hpp), offered when `tools` is supported: the model's call format, and the renderer of
    // conversations with tools (null or an array) or tool messages. Unset: tools / tool_choice are rejected.
    ToolFormat tools;
    std::function<std::string(const nlohmann::ordered_json &messages, const nlohmann::ordered_json &tools)> render_tools;
};
const ChatBackend &chat_backend();
void set_chat_backend(ChatBackend);
const std::string &model_name(); // the id the server reports

// 400 error that names the offending request parameter (OpenAI error "param").
struct RequestError : std::invalid_argument {
    std::string param;
    RequestError(const std::string &message, std::string p)
        : std::invalid_argument(message), param(std::move(p)) {}
};
// Structured output request (response_format / grammar), compiled to GBNF.
struct StructuredOutput {
    enum Kind { None, JsonObject, JsonSchema, Grammar } kind = None;
    std::string gbnf;        // grammar the answer must match
    bool json = false;       // answer is one JSON value: it ends when the value is complete
    bool allow_think = true; // a leading <think> block may precede the answer (unconstrained)
    std::string name;        // json_schema.name
    bool strict = false;     // json_schema.strict (always enforced; recorded only)
};
struct ChatRequest {
    std::string rendered;
    SamplingOptions sampling;
    size_t max_new = 256;
    bool stream = false;
    bool include_usage = false; // stream_options.include_usage: final chunk carries usage
    std::vector<std::string> stops;
    StructuredOutput structured;
    ToolRequest tools;
    std::string tool_grammar; // tools.active(): tool_call_grammar() of the request
};
// `ordered`: the same request parsed with nlohmann::ordered_json, so JSON-schema
// properties keep their declared order (the order the model is made to emit them in).
ChatRequest parse_chat(const nlohmann::json &, const nlohmann::ordered_json *ordered = nullptr);
// Parses response_format / grammar / reasoning_effort into a StructuredOutput. `tool_constraint`: the answer is
// constrained to tool calls (tool_choice required / a function), so reasoning_effort=none is enforced without a
// response_format; allow_think then applies to that constraint.
StructuredOutput parse_structured(const nlohmann::ordered_json &request, bool tool_constraint = false);
struct ReasoningPart {
    bool reasoning = false;
    std::string text;
};
// Consumes decoded/UTF-8-safe TextStream output. Explicit think tags route
// generated text; unmarked output remains answer content.
class ReasoningStream {
    std::string pending_;
    size_t depth_ = 0;
    bool finished_ = false;
  public:
    std::vector<ReasoningPart> append(std::string_view text, bool finish = false);
};
} // namespace behemoth
