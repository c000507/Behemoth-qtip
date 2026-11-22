// Inkling 2 tool-call format through ToolCallParser (runtime/src/tools.cpp) and the call grammar: the text the server feeds
// the parser (control tokens as their marker texts, reasoning and the message ends already split off by ReasoningStream).
//   host_test.sh builds and runs it.
#include "behemoth/tools.hpp"
#include <iostream>

using namespace behemoth;
static int failures = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; } } while (0)

static ToolFormat format() {
    ToolFormat f;
    f.calls = "<|message_model|>";
    f.args = "<|content_invoke_tool_json|>";
    f.inkling = true;
    return f;
}
// feeds `text` in pieces of `step` characters
static ToolCallParser run(const std::string &text, size_t step) {
    ToolCallParser p(format());
    for (size_t i = 0; i < text.size(); i += step)
        p.feed(std::string_view(text).substr(i, step));
    p.feed("", true);
    return p;
}

int main() {
    const std::string call1 = "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":2}}";
    const std::string call2 = "<|message_model|>now<|content_invoke_tool_json|>{\"name\":\"now\",\"args\":{}}";
    for (size_t step : {size_t(1), size_t(3), size_t(7), size_t(1000)}) {
        { // thinking message (split off: only its message start is left), then one call
            auto p = run("<|message_model|>" + call1, step);
            CHECK(p.calls().size() == 1);
            CHECK(p.content().empty());
            if (p.calls().size() == 1) {
                CHECK(p.calls()[0].name == "get_weather");
                CHECK(nlohmann::json::parse(p.calls()[0].arguments) == nlohmann::json({{"city", "Paris"}, {"days", 2}}));
            }
        }
        { // a plain answer
            auto p = run("<|message_model|><|message_model|><|content_text|>The capital is {Paris}.", step);
            CHECK(p.calls().empty());
            CHECK(p.content() == "The capital is {Paris}.");
        }
        { // an answer, then two calls
            auto p = run("<|message_model|><|content_text|>Let me check." + call1 + call2, step);
            CHECK(p.content() == "Let me check.");
            CHECK(p.calls().size() == 2);
            if (p.calls().size() == 2) {
                CHECK(p.calls()[1].name == "now");
                CHECK(p.calls()[1].arguments == "{}");
                CHECK(p.calls()[0].id != p.calls()[1].id);
            }
        }
        { // a constrained answer: the message end arrives as </think> (not dropped by the reasoning splitter)
            auto p = run(call1 + "</think>", step);
            CHECK(p.calls().size() == 1 && p.content().empty());
            if (p.calls().size() == 1)
                CHECK(nlohmann::json::parse(p.calls()[0].arguments, nullptr, false) == nlohmann::json({{"city", "Paris"}, {"days", 2}}));
        }
        { // broken JSON is passed on as the arguments text
            auto p = run("<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":", step);
            CHECK(p.calls().size() == 1 && p.calls()[0].name == "f" && !p.calls()[0].arguments.empty());
        }
    }
    ToolRequest r;
    r.choice = ToolRequest::Required;
    r.tools = nlohmann::ordered_json::parse(R"([{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])");
    const std::string g = tool_call_grammar(r, format());
    CHECK(g.find("<|message_model|>") != std::string::npos);
    CHECK(g.find("<|content_invoke_tool_json|>") != std::string::npos);
    CHECK(g.find("args") != std::string::npos);
    std::cout << (failures ? "FAIL" : "PASS") << " inkling_tools: " << checks << " checks\n";
    return failures ? 1 : 0;
}
