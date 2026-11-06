// behemoth::Tokenizer for a Llama-3-style byte-level BPE checkpoint (tokenizer.json + chat_template), so
// runtime/src/server.cpp serves it unchanged: encode/piece/decode through glm53::Tokenizer, and the checkpoint's
// chat template installed as the server's prompt renderer (protocol.hpp set_chat_backend).
#include "behemoth/protocol.hpp"
#include "behemoth/tokenizer.hpp"
#include "glm53.hpp"
#include <cstdlib>

namespace behemoth {
struct Tokenizer::Impl {
    glm53::Tokenizer bpe;
    explicit Impl(const std::filesystem::path &root) : bpe(root) {}
};
Tokenizer::Tokenizer(const std::filesystem::path &root) : impl_(std::make_unique<Impl>(root)) {
    auto *bpe = &impl_->bpe;
    const char *name = std::getenv("BEHEMOTH_MODEL_NAME");
    set_chat_backend({name ? name : root.filename().string(),
                      [bpe](const nlohmann::json &messages) { return bpe->chat(messages, true); }});
}
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string &rendered) const { return impl_->bpe.encode(rendered); }
std::string Tokenizer::piece(int token, bool &) const { return impl_->bpe.decode({token}, false); }
std::string Tokenizer::decode(const std::vector<int> &ids) const { return impl_->bpe.decode(ids, false); }
bool Tokenizer::is_special(int token) const {
    return impl_->bpe.decode({token}, false).empty() && !impl_->bpe.decode({token}, true).empty();
}
// Structured outputs are not offered on this backend (Engine::submit rejects constraints); the server still builds
// a vocabulary object at startup, which must have the Behemoth size, so every id is an empty control token.
std::vector<Vocabulary::Token> vocabulary_pieces(const Tokenizer &) {
    Vocabulary::Token empty;
    empty.special = true;
    return std::vector<Vocabulary::Token>(size_t(vocabulary_size), empty);
}
} // namespace behemoth
