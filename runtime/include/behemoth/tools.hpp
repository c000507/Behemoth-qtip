// OpenAI tool calling (tools / tool_choice in the request, tool_calls in the answer) for checkpoints that write
// Mistral's native tool format (tokenizer v11, Mistral Small 3.2 / Tekken):
//
//   prompt:  [AVAILABLE_TOOLS][{"type": "function", "function": {...}}][/AVAILABLE_TOOLS] before the last [INST],
//            assistant turns with calls as  content[TOOL_CALLS]name[ARGS]{...}</s>, results as
//            [TOOL_RESULTS]content[/TOOL_RESULTS]
//   output:  [TOOL_CALLS]name[ARGS]{"a": 1}[TOOL_CALLS]other[ARGS]{...}</s>
//
// [TOOL_CALLS] and [ARGS] are control tokens: the server streams them as their marker text into ToolCallParser (and
// the constraint vocabulary gives them the same text), so the parser and the grammars work on text alone. The parser
// also takes the older v7 form ([TOOL_CALLS][{"name": ..., "arguments": {...}}]).
//
// Enforcement: tool_choice "required" / a named function constrains the whole answer with tool_call_grammar();
// "auto" leaves the answer free until the model emits [TOOL_CALLS], then the same grammar takes over
// (TriggeredConstraint), so arguments always match the tool's parameters schema.
#pragma once

#include "behemoth/constraint.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace behemoth {

// How a backend's model writes tool calls. Empty `calls`: the backend offers no tool calling.
//   Mistral: [TOOL_CALLS]name[ARGS]{json}   (calls = "[TOOL_CALLS]", args = "[ARGS]")
//   GLM:     <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>, a string value raw, any other
//            value as JSON (GLM-4.5 .. 5.x chat templates render tools themselves); calls = "<tool_call>", glm = true
struct ToolFormat {
    std::string calls, args;               // marker texts ("[TOOL_CALLS]", "[ARGS]")
    int calls_token = -1, args_token = -1; // their control-token ids (streamed as the marker text)
    bool glm = false;
    // Mistral Tekken v13+ (Devstral 2507, Magistral): [AVAILABLE_TOOLS] once after the system prompt, not before the
    // last [INST] (v11), and no tool_instruction(): the checkpoint was trained on this format
    bool tools_first = false;
    // every control token of the format with its text (GLM: <tool_call> </tool_call> <arg_key> </arg_key> <arg_value>
    // </arg_value>), streamed as text into the parser and the constraints; Mistral: empty (calls / args tokens)
    std::vector<std::pair<int, std::string>> markers;
    bool supported() const {
        return !calls.empty();
    }
    // The text a control token stands for in tool calls, else null.
    const std::string *marker(int token) const {
        if (token < 0)
            return nullptr;
        if (token == calls_token)
            return &calls;
        if (token == args_token)
            return &args;
        for (const auto &[id, text] : markers)
            if (id == token)
                return &text;
        return nullptr;
    }
};
inline constexpr const char *glm_call_end = "</tool_call>", *glm_key_open = "<arg_key>", *glm_key_close = "</arg_key>",
                            *glm_value_open = "<arg_value>", *glm_value_close = "</arg_value>";

struct ToolRequest {
    enum Choice { None, Auto, Required, Named } choice = None; // None also when the request has no tools
    nlohmann::ordered_json tools = nlohmann::ordered_json::array(); // validated [{"type":"function","function":{...}}]
    std::string name;     // Named: the function the answer must call
    bool parallel = true; // parallel_tool_calls: more than one call per answer
    bool active() const {
        return choice != None;
    }
};

// Parses and validates tools / tool_choice / parallel_tool_calls (RequestError -> HTTP 400).
ToolRequest parse_tools(const nlohmann::ordered_json &request);

// GBNF of an answer made of tool calls: one call (several with `parallel`) of the offered functions (only `name`
// for Named), each [calls]name[args] then a JSON value matching that function's parameters schema. JSON mode.
// GLM: <tool_call>name, then the schema's properties in declared order (required ones mandatory) as
// <arg_key>key</arg_key><arg_value>value</arg_value> (strings raw, other types as JSON matching their schema), </tool_call>.
std::string tool_call_grammar(const ToolRequest &, const ToolFormat &);

// Mistral v11 prompt for a conversation with tools and/or tool messages. `system` is the system prompt used when
// the conversation has none (the checkpoint template's default; empty: no system block); tools null or empty: no
// [AVAILABLE_TOOLS] block. With tools, the system prompt ends with tool_instruction().

// How and when to call (fine-tunes of a tool-trained base, Cydonia, otherwise answer in prose or write bare JSON).
std::string tool_instruction(const ToolFormat &);
std::string render_mistral_tools(const nlohmann::ordered_json &messages, const nlohmann::ordered_json &tools,
                                 const ToolFormat &, const std::string &bos, const std::string &eos,
                                 const std::string &system);

// JSON text as Python's json.dumps(ensure_ascii=False) writes it (", " and ": " separators): the form Mistral's
// tokenizer gives the tool definitions and call arguments in.
std::string python_json(const nlohmann::ordered_json &);

// Whether `value` matches the JSON Schema (the grammar compiler's subset; object keys in any order).
bool schema_accepts(const nlohmann::ordered_json &schema, const nlohmann::json &value);
// The JSON value (object or array, bare or in a ``` fence) that ends `text` (trailing whitespace aside): its start
// offset in `text` (the fence's when fenced) and the value; the earliest start that parses as the whole rest.
std::optional<std::pair<size_t, nlohmann::json>> trailing_json(std::string_view text);

struct ToolCall {
    std::string id, name, arguments;
    nlohmann::json openai() const; // {"id", "type": "function", "function": {"name", "arguments"}}
};

// Incremental split of the answer text into content and tool calls. feed() returns what can be streamed now:
// content text (held back while it may be the start of a marker) and OpenAI stream items for "tool_calls"
// ({"index", "id", "type", "function": {"name", "arguments": ""}} when a call's name is complete, then
// {"index", "function": {"arguments": fragment}}).
// `fallback` (tool_choice auto): an answer that ends with a JSON tool call written as text (no [TOOL_CALLS]; see
// convert_trailing) becomes calls too. Streaming then holds the content from its first "{", "[" or "`" to the end.
class ToolCallParser {
  public:
    explicit ToolCallParser(ToolFormat format, const ToolRequest *fallback = nullptr);
    struct Delta {
        std::string content;
        nlohmann::json tool_calls = nlohmann::json::array();
    };
    Delta feed(std::string_view text, bool finish = false);
    const std::vector<ToolCall> &calls() const {
        return calls_;
    }
    const std::string &content() const {
        return content_;
    }

  private:
    enum Mode { Content, Head, Name, Args, List, GlmName, GlmArgs, GlmKey, GlmValue } mode_ = Content;
    ToolFormat format_;
    std::string buffer_, content_, list_;
    std::vector<ToolCall> calls_;
    bool fallback_ = false;
    nlohmann::ordered_json tools_;
    std::string new_id();
    bool convert_trailing(Delta &);
    void start_call(std::string name, Delta &);
    void add_arguments(std::string_view, Delta &);
    void finish_list(Delta &);
    // GLM calls: the call being parsed (name started, arguments collected until </tool_call>)
    nlohmann::ordered_json glm_args_;
    std::string glm_key_;
    bool glm_step(Delta &, bool finish);
    nlohmann::json glm_value(const std::string &key, const std::string &text) const;
};

// Lazy grammar (tool_choice auto): no constraint until `trigger` is sampled, then `inner` (a constraint whose answer
// starts with the trigger's piece, advanced with the trigger itself) for the rest of the answer.
class TriggeredConstraint final : public TokenConstraint {
  public:
    TriggeredConstraint(int trigger, std::shared_ptr<TokenConstraint> inner, size_t vocabulary);
    bool accepts(int token) override;
    bool accept_first(bool greedy) const override;
    Allowed allowed() override;
    void advance(int token) override;
    bool triggered() const {
        return triggered_;
    }

  private:
    int trigger_;
    std::shared_ptr<TokenConstraint> inner_;
    bool triggered_ = false;
    TokenMask all_;
    int count_;
};

} // namespace behemoth
