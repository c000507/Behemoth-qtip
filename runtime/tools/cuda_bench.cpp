#include "behemoth/cuda.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace behemoth::cuda_runtime;
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (std::string(argv[i]) == "--help") {
                std::cout << "behemoth_cuda_bench --model DIR [--tp 2] [--device 0] [--batch 1|8] "
                             "[--prompt-tokens 8192] [--decode-tokens 256] [--warmup 1] [--output FILE] "
                             "[--trace-logits FILE]\n";
                return 0;
            }
            check(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second,
                  "Missing/duplicate benchmark option");
        }
        for (const auto &[name, value] : args) {
            (void)value;
            check(name == "--model" || name == "--tp" || name == "--device" || name == "--batch" ||
                      name == "--prompt-tokens" || name == "--decode-tokens" || name == "--warmup" ||
                      name == "--output" || name == "--trace-logits",
                  "Unknown benchmark option: " + name);
        }
        auto number = [&](const char *name, int fallback) {
            if (!args.contains(name))
                return fallback;
            size_t used = 0;
            int n = std::stoi(args.at(name), &used);
            check(used == args.at(name).size(), "Invalid benchmark integer");
            return n;
        };
        int batch = number("--batch", 1), prompt = number("--prompt-tokens", 8192),
            decode = number("--decode-tokens", 256);
        int tp = number("--tp", 2), device = number("--device", 0), warmup = number("--warmup", 1);
        check(batch >= 1 && batch <= 8 && prompt > 0 && decode > 0 && prompt + decode <= 131072,
              "Invalid benchmark dimensions");
        Tokenizer tokenizer(args.at("--model"));
        auto seed = tokenizer.encode("The library was quiet as the researcher opened a notebook and began "
                                     "describing the experiment. The observations were precise, and the "
                                     "measurements would be checked carefully before drawing conclusions. ");
        check(!seed.empty(), "Benchmark tokenizer returned no tokens");
        std::vector<std::vector<int>> input(batch, std::vector<int>(prompt));
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < prompt; ++t)
                input[b][t] = seed[(t + b * 7) % seed.size()];
        Model model(args.at("--model"), device, prompt + decode, tp, batch);
        auto run = [&](bool record_trace) {
            model.reset();
            std::cerr << "Prefill " << batch << " x " << prompt << " tokens\n";
            auto start = std::chrono::steady_clock::now();
            auto logits = model.prefill_batch(input);
            auto prefill_end = std::chrono::steady_clock::now();
            std::cerr << "Prefill seconds " << std::chrono::duration<double>(prefill_end - start).count()
                      << "; decode " << decode << " steps\n";
            std::vector<int> ids(batch);
            std::vector<double> step_seconds;
            std::vector<Vec> prefill_logits;
            std::vector<std::vector<int>> generated;
            if (record_trace) {
                prefill_logits = logits;
                generated.reserve(decode);
            }
            auto decode_start = std::chrono::steady_clock::now();
            for (int step = 0; step < decode; ++step) {
                auto step_start = std::chrono::steady_clock::now();
                for (int b = 0; b < batch; ++b)
                    ids[b] = int(std::max_element(logits[b].begin(), logits[b].end()) - logits[b].begin());
                logits = model.step_batch(ids);
                step_seconds.push_back(
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count());
                if ((step + 1) % 32 == 0)
                    std::cerr << "decode " << step + 1 << '/' << decode << '\n';
                if (record_trace)
                    generated.push_back(ids);
            }
            auto end = std::chrono::steady_clock::now();
            double prefill_seconds = std::chrono::duration<double>(prefill_end - start).count();
            double decode_seconds = std::chrono::duration<double>(end - decode_start).count();
            if (record_trace) {
                std::ofstream trace(args.at("--trace-logits"));
                trace << json{{"prefill_logits", prefill_logits},
                              {"final_logits", logits},
                              {"generated_token_ids", generated}}
                             .dump()
                      << '\n';
                check(bool(trace), "Cannot write benchmark logits trace");
            }
            json result{{"tp", tp},
                        {"batch", batch},
                        {"prompt_tokens_per_sequence", prompt},
                        {"decode_tokens_per_sequence", decode},
                        {"prefill_seconds", prefill_seconds},
                        {"decode_seconds", decode_seconds},
                        {"prefill_tokens_per_second", double(batch) * prompt / prefill_seconds},
                        {"decode_tokens_per_second", double(batch) * decode / decode_seconds},
                        {"ignore_eos", true},
                        {"kv_dtype", "BF16"}};
            auto sorted = step_seconds;
            std::sort(sorted.begin(), sorted.end());
            result["decode_step_seconds_summary"] = {{"min", sorted.front()},
                                                     {"median", sorted[sorted.size() / 2]},
                                                     {"p95", sorted[(sorted.size() - 1) * 95 / 100]},
                                                     {"max", sorted.back()}};
            if (std::getenv("BEHEMOTH_CUDA_BENCH_STEP_TIMES"))
                result["decode_step_seconds"] = step_seconds;
            return result;
        };
        for (int i = 0; i < warmup; ++i) {
            std::cerr << "Warmup " << i + 1 << '/' << warmup << '\n';
            (void)run(false);
        }
        auto result = run(args.contains("--trace-logits"));
        result["warmup_runs"] = warmup;
        std::cout << result.dump(2) << '\n';
        if (args.contains("--output")) {
            std::ofstream f(args.at("--output"));
            f << result.dump(2) << '\n';
            check(bool(f), "Cannot write benchmark output");
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
