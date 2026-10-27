#define main qtip_prepare_entry
#include "../src/prepare.cpp"
#undef main
#include <sentencepiece_trainer.h>
int main() {
    auto root = fs::temp_directory_path() / ("qtip-pack-test-" + std::to_string(getpid()));
    try {
        fs::create_directories(root);
        {
            std::ofstream f(root / "corpus.txt");
            for (int i = 0; i < 100; ++i)
                f << "hello user answer café 世界 reasoning result science and mathematics\n";
        }
        auto status = sentencepiece::SentencePieceTrainer::Train(
            "--input=" + (root / "corpus.txt").string() + " --model_prefix=" + (root / "tokenizer").string() +
            " --vocab_size=64 --hard_vocab_limit=false --model_type=bpe "
            "--user_defined_symbols=[INST],[/INST],[SYSTEM_PROMPT],[/SYSTEM_PROMPT],<think>,</think>");
        require(status.ok(), status.ToString());
        sentencepiece::SentencePieceProcessor sp;
        require(sp.Load((root / "tokenizer.model").string()).ok(), "load fixture tokenizer");
        json config = {{"legacy", true}, {"added_tokens_decoder", json::object()}};
        for (auto text : {"<s>", "</s>", "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]",
                          "<think>", "</think>"})
            config["added_tokens_decoder"][std::to_string(sp.PieceToId(text))] = {{"content", text},
                                                                                  {"lstrip", false},
                                                                                  {"rstrip", false},
                                                                                  {"single_word", false},
                                                                                  {"normalized", false}};
        durable_json(root / "tokenizer_config.json", config);
        Tokenizer tok(root);
        json messages = {{{"role", "user"}, {"content", "hello user"}},
                         {{"role", "assistant"},
                          {"content", "<think>old</think> answer café 世界"},
                          {"reasoning_content", "reasoning result"}}};
        auto r = tok.conversation(messages, 7);
        require(r.tokens.front() == sp.bos_id() && r.tokens.back() == sp.eos_id(), "chat BOS/EOS");
        require(r.loss.front() == 0 && r.loss.back() == 1, "assistant EOS target");
        auto empty = tok.conversation(json::array({{{"role", "assistant"}, {"content", ""}}}), 8);
        require(empty.tokens.size() == 2 && empty.loss == std::vector<uint8_t>({0, 0}),
                "empty assistant/whitespace segment");
        auto first = std::find(r.loss.begin(), r.loss.end(), 1);
        require(first != r.loss.end() && first - r.loss.begin() > 5, "user/prompt loss mask");
        for (auto it = first; it != r.loss.end(); ++it)
            require(*it == 1, "assistant span gap");
        Row a{{1, 2, 3}, {4, 4, 4}, {0, 1, 2}, {0, 1, 1}}, b{{6, 7}, {9, 9}, {0, 1}, {0, 1}};
        auto packed = pack({a, b}, 4, 2);
        require(packed.size() == 2, "packing count");
        require(packed[0].tokens == std::vector<int32_t>({1, 2, 3, 6}) &&
                    packed[1].tokens == std::vector<int32_t>({7, 2, 2, 2}),
                "packing duplicates or loses tokens");
        require(packed[1].segments == std::vector<int32_t>({9, -1, -1, -1}) && packed[1].positions[0] == 1 &&
                    packed[1].loss[1] == 0,
                "padding/positions");
        fs::remove_all(root);
        std::cout << "native packing tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
