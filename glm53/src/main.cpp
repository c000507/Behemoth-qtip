#include "glm53.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
    using namespace glm53;
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << R"(glm53 inspect --model DIR
glm53 tokenize --model DIR --input TEXT_FILE --out TOKENS.json
glm53 render --model DIR --messages CHAT.json --out PROMPT.txt
glm53 quantize --model SOURCE --out NEW_DIR
glm53 generate --model DIR --tokens TOKENS.json [--steps 32] [--arithmetic bf16|fp32] [--device cpu|0]
glm53 generate --model DIR --input RENDERED_TEXT_FILE [--steps 32]
glm53 generate --model DIR --messages CHAT.json [--steps 32]
glm53 trace --model DIR --tokens REQUEST.json --out TRACE.json [--arithmetic bf16|fp32]
Native CPU text reference backend. Quantize stores NVFP4 MLP weights and preserves other tensors.
Tokenization applies tokenizer.json's BPE/regex and ignores serialized truncation.
Generate uses greedy decoding and preserves recurrent state; this is not the GPU HTTP server.
)";
            return argc < 2 ? 1 : 0;
        }
        std::string mode = argv[1];
        std::map<std::string, std::string> a;
        for (int i = 2; i < argc; i += 2) {
            check(i + 1 < argc && a.emplace(argv[i], argv[i + 1]).second, "missing/duplicate option");
        }
        for (const auto &[k, v] : a) {
            (void)v;
            check(k == "--model" || k == "--input" || k == "--out" || k == "--steps" || k == "--arithmetic" ||
                      k == "--tokens" || k == "--messages" || k == "--device",
                  "unknown option " + k);
        }
        fs::path path = a.at("--model");
        Config config(read_json(path / "config.json"));
        if (mode == "inspect") {
            auto j = config.summary();
            if (fs::exists(path / "tokenizer.json"))
                j["tokenizer"] = Tokenizer(path).summary();
            std::cout << j.dump(2) << '\n';
            return 0;
        }
        if (mode == "quantize") {
            convert_nvfp4(path, a.at("--out"));
            std::cout << "Saved " << a.at("--out") << '\n';
            return 0;
        }
        if (mode == "render") {
            auto messages = read_json(a.at("--messages"));
            if (messages.is_object())
                messages = messages.at("messages");
            auto rendered = Tokenizer(path).chat(messages);
            auto out = fs::path(a.at("--out"));
            check(!fs::exists(out), "output exists");
            std::ofstream f(out);
            f << rendered;
            check(bool(f), "render write failed");
            return 0;
        }
        if (mode == "tokenize") {
            std::ifstream f(a.at("--input"));
            check(bool(f), "cannot read input text");
            std::string text((std::istreambuf_iterator<char>(f)), {});
            auto out = fs::path(a.at("--out"));
            check(!fs::exists(out), "output exists");
            auto ids = Tokenizer(path).encode(text);
            std::ofstream o(out);
            o << json({{"tokens", ids}}).dump() << '\n';
            check(bool(o), "write failed");
            return 0;
        }
        check(mode == "generate" || mode == "trace", "unknown mode");
        int supplied =
            int(a.contains("--tokens")) + int(a.contains("--input")) + int(a.contains("--messages"));
        check(supplied == 1, "supply exactly one of --tokens, --input, --messages");
        std::unique_ptr<Tokenizer> tokenizer;
        if (fs::exists(path / "tokenizer.json"))
            tokenizer = std::make_unique<Tokenizer>(path);
        std::vector<int> ids;
        if (a.contains("--tokens")) {
            auto j = read_json(a.at("--tokens"));
            ids = (j.is_array() ? j : j.at("tokens")).get<std::vector<int>>();
        } else {
            check(bool(tokenizer), "tokenizer assets missing; provide exact token IDs");
            std::string text;
            if (a.contains("--messages"))
                text = tokenizer->chat(read_json(a.at("--messages")));
            else {
                std::ifstream f(a.at("--input"));
                check(bool(f), "cannot read text");
                text.assign(std::istreambuf_iterator<char>(f), {});
            }
            ids = tokenizer->encode(text);
        }
        check(!ids.empty(), "empty prompt");
        size_t used = 0;
        std::string s = a.contains("--steps") ? a.at("--steps") : "32";
        int steps = std::stoi(s, &used);
        check(used == s.size() && steps > 0 && steps <= config.max_context, "invalid steps");
        std::string arithmetic = a.contains("--arithmetic") ? a.at("--arithmetic") : "bf16";
        check(arithmetic == "bf16" || arithmetic == "fp32", "invalid arithmetic");
        check(ids.size() + size_t(mode == "generate" ? steps : 0) <= size_t(config.max_context),
              "prompt plus generation exceeds context");
        Model model(path, arithmetic == "bf16");
        Vec logits;
        if (a.contains("--device") && a.at("--device") != "cpu") {
#if defined(GLM53_HIP)
            check(arithmetic == "bf16", "GPU projections require bf16 arithmetic");
            size_t used = 0;
            int device = std::stoi(a.at("--device"), &used);
            check(used == a.at("--device").size() && device >= 0, "invalid GPU device");
            model.set_projector(gpu_projector(device));
#else
            throw std::runtime_error("rebuild with -DGLM53_HIP=ON for GPU projection offload");
#endif
        }
        std::vector<int> selected;
        if (mode == "trace") {
            auto request = a.contains("--tokens") ? read_json(a.at("--tokens")) : json::object();
            selected = request.is_object()
                           ? request.value("logit_positions", std::vector<int>{int(ids.size() - 1)})
                           : std::vector<int>{int(ids.size() - 1)};
            check(!selected.empty() && selected.size() <= 128, "trace needs 1..128 selected positions");
            std::set<int> unique;
            for (int t : selected)
                check(t >= 0 && size_t(t) < ids.size() && unique.insert(t).second,
                      "invalid/duplicate trace position");
            check(!fs::exists(a.at("--out")), "trace output exists");
        }
        std::map<int, Vec> captured;
        for (size_t t = 0; t < ids.size(); ++t) {
            logits = model.step(ids[t]);
            if (std::find(selected.begin(), selected.end(), int(t)) != selected.end())
                captured[int(t)] = logits;
            std::cerr << "prefill " << t + 1 << '/' << ids.size() << '\n';
        }
        if (mode == "trace") {
            json rows = json::array();
            for (int t : selected)
                rows.push_back(captured.at(t));
            std::ofstream out(a.at("--out"));
            out << json({{"model", path.string()},
                         {"tokens", ids},
                         {"logit_positions", selected},
                         {"logits", rows},
                         {"arithmetic", arithmetic}})
                       .dump()
                << '\n';
            check(bool(out), "trace write failed");
            return 0;
        }
        std::vector<int> generated;
        for (int i = 0; i < steps; ++i) {
            for (float v : logits)
                check(std::isfinite(v), "nonfinite model logits");
            int token = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (std::find(config.eos.begin(), config.eos.end(), token) != config.eos.end())
                break;
            generated.push_back(token);
            std::cerr << "generated " << i + 1 << '/' << steps << " token " << token << '\n';
            if (i + 1 < steps)
                logits = model.step(token);
        }
        json result = {{"tokens", generated},
                       {"arithmetic", arithmetic},
                       {"backend", a.contains("--device") && a.at("--device") != "cpu"
                                       ? "CPU attention/state with streamed HIP projections"
                                       : "native CPU reference"}};
        if (tokenizer)
            result["text"] = tokenizer->decode(generated);
        std::cout << result.dump() << '\n';
    } catch (const std::exception &e) {
        std::cerr << "glm53: " << e.what() << '\n';
        return 1;
    }
}
