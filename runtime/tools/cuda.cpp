#include "behemoth/cuda.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
int main(int argc, char **argv) {
    using namespace behemoth::cuda_runtime;
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i < argc; i += 2) {
            if (std::string(argv[i]) == "--help") {
                std::cout << "behemoth_cuda --model DIR [--prompt TEXT | --messages CHAT.json | --tokens "
                             "TOKENS.json] [--max-tokens 64] [--context 4096] [--device 0] [--tp N | --layer-split N] "
                             "[--trace TRACE.json]\n"
                             "GLM: --tp N shards every layer over N GPUs (tensor parallelism); --layer-split N "
                             "runs contiguous layer ranges on N GPUs instead.\n";
                return 0;
            }
            check(i + 1 < argc && a.emplace(argv[i], argv[i + 1]).second, "Missing/duplicate argument");
        }
        for (const auto &[k, v] : a) {
            (void)v;
            check(k == "--model" || k == "--prompt" || k == "--messages" || k == "--tokens" ||
                      k == "--max-tokens" || k == "--context" || k == "--device" || k == "--trace" ||
                      k == "--tp" || k == "--layer-split",
                  "Unknown argument " + k);
        }
        auto number = [&](const char *key, int fallback) {
            if (!a.contains(key))
                return fallback;
            size_t used = 0;
            int v = std::stoi(a.at(key), &used);
            check(used == a.at(key).size(), "Invalid integer argument");
            return v;
        };
        int steps = number("--max-tokens", 64), context = number("--context", 4096),
            device = number("--device", 0);
        check(steps >= 0 && device >= 0, "Invalid generation/device setting");
        std::unique_ptr<Tokenizer> tok;
        if (fs::exists(fs::path(a.at("--model")) / "tokenizer.json"))
            tok = std::make_unique<Tokenizer>(a.at("--model"));
        std::vector<int> ids, positions;
        auto read = [](const std::string &p) {
            std::ifstream f(p);
            check(bool(f), "Cannot read " + p);
            return json::parse(f);
        };
        check(int(a.contains("--prompt")) + int(a.contains("--messages")) + int(a.contains("--tokens")) <= 1,
              "Select one input format");
        if (a.contains("--tokens")) {
            auto j = read(a.at("--tokens"));
            ids = (j.is_array() ? j : j.at("tokens")).get<std::vector<int>>();
            if (j.is_object())
                positions = j.value("logit_positions", std::vector<int>{});
        } else {
            auto messages = a.contains("--messages")
                                ? read(a.at("--messages"))
                                : json::array({{{"role", "user"},
                                                {"content", a.contains("--prompt")
                                                                ? a.at("--prompt")
                                                                : "Say hello in one short sentence."}}});
            check(bool(tok), "Tokenizer assets missing; provide exact --tokens");
            ids = tok->encode(tok->chat(messages));
        }
        check(!ids.empty() && ids.size() + size_t(steps) <= size_t(context), "Prompt/output exceed context");
        if (positions.empty())
            positions = {int(ids.size() - 1)};
        check(positions.size() <= 128, "Too many trace positions");
        for (int p : positions)
            check(p >= 0 && size_t(p) < ids.size(), "Invalid trace position");
        if (a.contains("--trace"))
            check(!fs::exists(a.at("--trace")), "Trace output already exists");
        int tp = number("--tp", 1), split = number("--layer-split", 0);
        check(tp >= 1 && split >= 0 && (tp == 1 || split <= 1), "Use --tp N or --layer-split N, not both");
        Model model(a.at("--model"), device, context, split > 1 ? split : tp, 1, split > 1);
        Vec logits;
        std::map<int, Vec> captured;
        auto start = std::chrono::steady_clock::now();
        if (!a.contains("--trace")) {
            logits = model.prefill(ids);
            std::cerr << "prefill " << ids.size() << '/' << ids.size() << '\n';
        } else
            for (size_t t = 0; t < ids.size(); ++t) {
                logits = model.step(ids[t]);
                if (a.contains("--trace") &&
                    std::find(positions.begin(), positions.end(), int(t)) != positions.end())
                    captured[int(t)] = logits;
                std::cerr << "prefill " << t + 1 << '/' << ids.size() << '\n';
            }
        auto decode_start = std::chrono::steady_clock::now();
        bool first = true;
        int generated = 0;
        for (int t = 0; t < steps; ++t) {
            int id = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (std::find(model.config.eos.begin(), model.config.eos.end(), id) != model.config.eos.end())
                break;
            std::cout << (tok ? tok->piece(id, first) : std::to_string(id) + " ") << std::flush;
            ++generated;
            if (t + 1 < steps)
                logits = model.step(id);
        }
        std::cout << '\n';
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        std::cerr << "Generated " << generated << " tokens in " << seconds << " s; "
                  << (seconds > 0 ? generated / seconds : 0) << " tok/s; prefill "
                  << std::chrono::duration<double>(decode_start - start).count() << " s\n";
        if (a.contains("--trace")) {
            json rows = json::array();
            for (int p : positions)
                rows.push_back(captured.at(p));
            std::ofstream f(a.at("--trace"));
            f << json({{"tokens", ids},
                       {"logit_positions", positions},
                       {"logits", rows},
                       {"backend", "CUDA"},
                       {"model_type",
                        read((fs::path(a.at("--model")) / "config.json").string()).value("model_type", "")}});
            check(bool(f), "Cannot write trace");
        }
    } catch (const std::exception &e) {
        std::cerr << "behemoth_cuda: " << e.what() << '\n';
        return 1;
    }
}
