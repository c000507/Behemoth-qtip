// Prefill timing and parity against the stepwise path of the same engine.
// usage: behemoth_cuda_prefill_bench --model DIR --tokens TOKENS.json [--lengths 64,512,2048]
//        [--repeats 5] [--follow 8] [--context N] [--no-stepwise 1]
// For every length L: best/median wall time of Model::prefill over the first L tokens, then (unless
// disabled) the same tokens through Model::step, and the relative L2 / argmax / KL of the last-token
// logits. `--follow` more tokens are then decoded after both paths so that the KV cache written by
// the prefill is checked as well (worst relative L2 over those steps).
#include "behemoth/cuda.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace behemoth::cuda_runtime;
namespace {
double seconds(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}
double relative(const Vec &a, const Vec &b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        e += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        n += double(b[i]) * b[i];
    }
    return std::sqrt(e / std::max(n, 1e-30));
}
int argmax(const Vec &a) { return int(std::max_element(a.begin(), a.end()) - a.begin()); }
// KL(reference || other) in nats
double kl(const Vec &reference, const Vec &other) {
    auto logsoftmax = [](const Vec &z) {
        double top = *std::max_element(z.begin(), z.end()), sum = 0;
        for (float v : z)
            sum += std::exp(v - top);
        std::vector<double> out(z.size());
        for (size_t i = 0; i < z.size(); ++i)
            out[i] = z[i] - top - std::log(sum);
        return out;
    };
    auto p = logsoftmax(reference), q = logsoftmax(other);
    double out = 0;
    for (size_t i = 0; i < p.size(); ++i)
        out += std::exp(p[i]) * (p[i] - q[i]);
    return out;
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i + 1 < argc; i += 2)
            a[argv[i]] = argv[i + 1];
        check(a.contains("--model") && a.contains("--tokens"), "usage: --model DIR --tokens TOKENS.json");
        std::ifstream file(a.at("--tokens"));
        check(bool(file), "Cannot read the token file");
        auto j = json::parse(file);
        auto all = (j.is_array() ? j : j.at("tokens")).get<std::vector<int>>();
        std::vector<int> lengths;
        std::stringstream list(a.contains("--lengths") ? a.at("--lengths") : std::to_string(all.size()));
        for (std::string item; std::getline(list, item, ',');)
            lengths.push_back(std::stoi(item));
        const int repeats = a.contains("--repeats") ? std::stoi(a.at("--repeats")) : 5;
        const int follow = a.contains("--follow") ? std::stoi(a.at("--follow")) : 8;
        const bool stepwise = !a.contains("--no-stepwise");
        int longest = 0;
        for (int n : lengths) {
            check(n > 0 && size_t(n) <= all.size(), "Length exceeds the token file");
            longest = std::max(longest, n);
        }
        const int context = a.contains("--context") ? std::stoi(a.at("--context")) : longest + follow + 1;
        Model model(a.at("--model"), 0, context);
        for (int n : lengths) {
            std::vector<int> ids(all.begin(), all.begin() + n);
            std::vector<double> times;
            Vec fast;
            for (int r = 0; r < repeats; ++r) {
                auto start = std::chrono::steady_clock::now();
                fast = model.prefill(ids);
                times.push_back(seconds(start));
            }
            std::sort(times.begin(), times.end());
            json row = {{"tokens", n},
                        {"prefill_best_s", times.front()},
                        {"prefill_median_s", times[times.size() / 2]},
                        {"prefill_tok_s", n / times.front()}};
            if (stepwise) {
                // decode `follow` tokens after the batched prefill (greedy on the prefill logits)
                std::vector<int> next;
                std::vector<Vec> after;
                Vec logits = fast;
                for (int t = 0; t < follow; ++t) {
                    next.push_back(argmax(logits));
                    logits = model.step(next.back());
                    after.push_back(logits);
                }
                model.reset();
                auto start = std::chrono::steady_clock::now();
                Vec slow;
                for (int id : ids)
                    slow = model.step(id);
                row["stepwise_s"] = seconds(start);
                row["relative_l2"] = relative(fast, slow);
                row["argmax_equal"] = argmax(fast) == argmax(slow);
                row["kl_nats"] = kl(slow, fast);
                double worst = 0, worst_kl = 0;
                int agree = 0;
                start = std::chrono::steady_clock::now();
                for (int t = 0; t < follow; ++t) {
                    auto reference = model.step(next[size_t(t)]);
                    worst = std::max(worst, relative(after[size_t(t)], reference));
                    worst_kl = std::max(worst_kl, kl(reference, after[size_t(t)]));
                    agree += argmax(after[size_t(t)]) == argmax(reference);
                }
                if (follow) {
                    row["decode_tok_s"] = follow / seconds(start);
                    row["follow_worst_relative_l2"] = worst;
                    row["follow_worst_kl_nats"] = worst_kl;
                    row["follow_argmax_equal"] = std::to_string(agree) + "/" + std::to_string(follow);
                }
            }
            std::cout << row.dump() << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "behemoth_cuda_prefill_bench: " << e.what() << '\n';
        return 1;
    }
}
