// Host check of a checkpoint's tokenizer + chat template through the server's path (glm53::Tokenizer: tokenizer.json
// ByteLevel BPE + minja chat template): prints the rendered prompt and the ids for a messages JSON (and optional tools),
// to be compared with HF AutoTokenizer.apply_chat_template / encode (neuron/tests/test_qwen35_tokenizer.py).
//   tok_check MODEL_DIR messages.json [tools.json]        (messages.json: OpenAI messages array)
//   tok_check MODEL_DIR --text file                        (encode raw text)
//   tok_check MODEL_DIR --decode ids.json                  (decode an id array)
// build: g++ -std=c++20 -O2 -I glm53/include -I third_party neuron/cpp/tok_check.cpp glm53/src/tokenizer.cpp glm53/src/config.cpp -lpcre2-8 -o tok_check
#include "glm53.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 3) { std::cerr << "usage: tok_check MODEL_DIR messages.json [tools.json] | --text file\n"; return 2; }
    glm53::Tokenizer tok(argv[1]);
    std::string rendered;
    if (std::string(argv[2]) == "--decode") {
        std::ifstream f(argv[3]);
        const auto ids = nlohmann::json::parse(f).get<std::vector<int>>();
        std::cout << nlohmann::json{{"text", tok.decode(ids, true)}}.dump() << std::endl;
        return 0;
    }
    if (std::string(argv[2]) == "--text") {
        std::ifstream f(argv[3]);
        rendered = std::string(std::istreambuf_iterator<char>(f), {});
    } else {
        std::ifstream f(argv[2]);
        auto messages = nlohmann::ordered_json::parse(f);
        if (argc > 3) {
            std::ifstream t(argv[3]);
            rendered = tok.chat_tools(messages, nlohmann::ordered_json::parse(t));
        } else
            rendered = tok.chat(nlohmann::json::parse(messages.dump()), true);
    }
    const auto ids = tok.encode(rendered);
    nlohmann::json out{{"rendered", rendered}, {"ids", ids}, {"decoded", tok.decode(ids, true)}};
    std::cout << out.dump() << std::endl;
}
