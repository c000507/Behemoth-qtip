// behemoth::Tokenizer for a Llama-3-style byte-level BPE checkpoint (tokenizer.json + chat_template), so
// runtime/src/server.cpp serves it unchanged: encode/piece/decode through glm53::Tokenizer, and the checkpoint's
// chat template installed as the server's prompt renderer (protocol.hpp set_chat_backend).
//
// Structured outputs: vocabulary_pieces() gives every id its decoded bytes (the BPE vocabulary as is: no SentencePiece
// first-piece rule), control tokens marked special, EOS the tokenizer's eos_token.
//
// Tool calling (tools.hpp), for Mistral tokenizers (Tekken: [AVAILABLE_TOOLS], [TOOL_CALLS] and [ARGS] control
// tokens; Cydonia / Mistral Small 3.2): their chat templates take no tools, so conversations with tools or tool
// messages are rendered in Mistral's v11 form (render_mistral_tools) behind the template's own system prompt.
// BEHEMOTH_TOOLS=0 turns tool calling off. Constrained answers start with an optional <think> block only for templates
// that know <think> (ChatBackend::think). Message content may be text blocks (joined); other parts are rejected.
#include "behemoth/protocol.hpp"
#include "behemoth/tokenizer.hpp"
#include "glm53.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace behemoth {
struct Tokenizer::Impl {
    glm53::Tokenizer bpe;
    std::string bos, eos;
    int eos_id = -1;
    ToolFormat tools;
    explicit Impl(const std::filesystem::path &root) : bpe(root) {
        auto token = [&](const char *key) -> std::string {
            const auto v = bpe.summary().value(key, nlohmann::json(nullptr));
            return v.is_string() ? v.get<std::string>() : v.is_object() ? v.value("content", "") : "";
        };
        bos = token("bos_token");
        eos = token("eos_token");
        eos_id = control(eos);
        const char *enabled = std::getenv("BEHEMOTH_TOOLS");
        const int calls = control("[TOOL_CALLS]");
        if ((enabled && std::atoi(enabled) == 0) || calls < 0 || control("[AVAILABLE_TOOLS]") < 0 || control("[TOOL_RESULTS]") < 0)
            return;
        // [ARGS] is id 32 of the Tekken v11 vocabulary; HF exports of v11 checkpoints may still name it <SPECIAL_32>
        int args = control("[ARGS]");
        if (args < 0 && calls == 9 && control("<SPECIAL_32>") == 32)
            args = 32;
        if (args < 0)
            return;
        tools = {"[TOOL_CALLS]", bpe.decode({args}, true), calls, args};
    }
    // Id of the control token spelled `text`, else -1.
    int control(const std::string &text) const {
        if (text.empty())
            return -1;
        const auto ids = bpe.encode(text);
        return ids.size() == 1 && bpe.decode(ids, false).empty() && bpe.decode(ids, true) == text ? ids[0] : -1;
    }
    // The system prompt the template inserts when the conversation has none (its [SYSTEM_PROMPT] block).
    std::string default_system() const {
        const std::string probe = "\x01probe\x01";
        const std::string text = bpe.chat(nlohmann::json::array({{{"role", "user"}, {"content", probe}}}), true);
        const size_t open = text.find("[SYSTEM_PROMPT]"), close = text.find("[/SYSTEM_PROMPT]"), user = text.find(probe);
        if (open == std::string::npos || close == std::string::npos || close > user)
            return "";
        return text.substr(open + 15, close - open - 15);
    }
};
// Content given as text blocks joined into one string (the templates take strings); anything else is rejected.
static nlohmann::json text_only(nlohmann::json messages) {
    for (auto &m : messages)
        if (m.is_object() && m.contains("content") && m.at("content").is_array()) {
            std::string text;
            for (const auto &b : m.at("content")) {
                if (!b.is_object() || b.value("type", "") != "text" || !b.contains("text") || !b.at("text").is_string())
                    throw std::invalid_argument("Only text message content is supported (no images or other content parts)");
                text += b.at("text").get<std::string>();
            }
            m["content"] = text;
        }
    return messages;
}
// The checkpoint's chat template source (tokenizer_config.json, else chat_template.jinja), for feature checks.
static std::string template_source(const std::filesystem::path &root) {
    std::ifstream config(root / "tokenizer_config.json");
    const auto c = nlohmann::json::parse(config, nullptr, false);
    if (!c.is_discarded() && c.contains("chat_template"))
        return c.at("chat_template").dump();
    std::ifstream jinja(root / "chat_template.jinja");
    return std::string(std::istreambuf_iterator<char>(jinja), {});
}
Tokenizer::Tokenizer(const std::filesystem::path &root) : impl_(std::make_unique<Impl>(root)) {
    auto *impl = impl_.get();
    const char *name = std::getenv("BEHEMOTH_MODEL_NAME");
    ChatBackend backend{name ? name : root.filename().string(),
                        [impl](const nlohmann::json &messages) { return impl->bpe.chat(text_only(messages), true); }};
    // Constrained answers may open with <think> only for templates that know it (reasoning models); others (Cydonia)
    // are constrained from the first token unless the request sets reasoning_effort.
    backend.think = template_source(root).find("<think>") != std::string::npos;
    if (impl->tools.supported()) {
        backend.tools = impl->tools;
        backend.render_tools = [impl](const nlohmann::ordered_json &messages, const nlohmann::ordered_json &tools) {
            return render_mistral_tools(messages, tools, impl->tools, impl->bos, impl->eos, impl->default_system());
        };
        std::cerr << "Tool calling: Mistral v11 format ([TOOL_CALLS] " << impl->tools.calls_token << ", [ARGS] "
                  << impl->tools.args_token << " '" << impl->tools.args << "')\n";
    }
    set_chat_backend(std::move(backend));
}
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string &rendered) const { return impl_->bpe.encode(rendered); }
std::string Tokenizer::piece(int token, bool &) const { return impl_->bpe.decode({token}, false); }
std::string Tokenizer::decode(const std::vector<int> &ids) const { return impl_->bpe.decode(ids, false); }
bool Tokenizer::is_special(int token) const {
    return impl_->bpe.decode({token}, false).empty() && !impl_->bpe.decode({token}, true).empty();
}
std::vector<Vocabulary::Token> vocabulary_pieces(const Tokenizer &tokenizer) {
    const auto &impl = *tokenizer.impl_;
    std::vector<Vocabulary::Token> tokens(impl.bpe.size());
    for (size_t id = 0; id < tokens.size(); ++id) {
        auto &t = tokens[id];
        t.first = t.rest = impl.bpe.decode({int(id)}, false);
        t.special = t.rest.empty(); // control tokens and ids without a vocabulary entry
    }
    // the tool-call control tokens are text to the constraints: the server streams them as their marker text
    for (auto [id, marker] : {std::pair{impl.tools.calls_token, impl.tools.calls}, {impl.tools.args_token, impl.tools.args}})
        if (id >= 0 && size_t(id) < tokens.size()) {
            tokens[size_t(id)].first = tokens[size_t(id)].rest = marker;
            tokens[size_t(id)].special = false;
        }
    return tokens;
}
int vocabulary_eos(const Tokenizer &tokenizer) {
    if (tokenizer.impl_->eos_id >= 0)
        return tokenizer.impl_->eos_id;
    std::cerr << "Structured outputs: tokenizer eos_token is not a control token; using id " << eos_token << "\n";
    return eos_token;
}
} // namespace behemoth
