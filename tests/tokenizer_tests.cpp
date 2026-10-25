#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"
#include "behemoth/tokenizer.hpp"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: behemoth_tokenizer_tests CHECKPOINT_TOKENIZER_DIRECTORY\n";
            return 2;
        }
        behemoth::Tokenizer t(argv[1]);
        if (t.encode("<s>[INST]</s>") != std::vector<int>{1, 3, 2})
            throw std::runtime_error("Special-token IDs mismatch");
        for (const std::string text : {"Hello world!", "C++ ROCm 123",
                                       "\xe2\x82\xac \xe4\xbd\xa0\xe5\xa5\xbd", "hello\nworld"}) {
            auto ids = t.encode(text);
            if (t.decode(ids) != text)
                throw std::runtime_error("Tokenizer roundtrip mismatch: " + text);
            bool first = true;
            std::string incremental;
            for (int id : ids)
                incremental += t.piece(id, first);
            if (incremental != t.decode(ids))
                throw std::runtime_error("Incremental decoder mismatch");
        }
        auto c = behemoth::parse_chat({{"messages", {{{"role", "user"}, {"content", "Hello"}}}}});
        auto encoded = t.encode(c.rendered);
        if (encoded.empty() || encoded[0] != 1 ||
            std::count(encoded.begin(), encoded.end(), 1) != 1)
            throw std::runtime_error("Chat inserted duplicate BOS");
        std::cout << "PASS checkpoint tokenizer special tokens, roundtrips, UTF-8, incremental "
                     "decode and chat BOS\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
