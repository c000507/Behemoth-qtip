// Qwen3.5 generation worker for tools/qwen_openai_server.py: loads the model once (optionally split
// over several GPUs), then serves one request at a time over stdin/stdout JSON lines.
// request : {"tokens":[...], "max_new":N, "temperature":T, "top_p":P, "top_k":K, "seed":S, "stop":[ids]}
// replies : {"t":id} per generated token, then
//           {"done":reason, "prompt":n, "completion":n, "prefill_s":s, "decode_s":s}
// usage   : qwen_worker --model DIR [--gpus 0,1] [--chunk N] [--max-context N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "behemoth/qwen_model.hpp"

using json = nlohmann::json;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int sample(const std::vector<float> &logits, float temperature, float top_p, int top_k,
                  std::mt19937_64 &rng) {
    const int V = int(logits.size());
    if (temperature <= 0) return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    // candidates: top_k (default / cap 512) by logit
    int k = top_k > 0 ? std::min(top_k, V) : std::min(512, V);
    std::vector<int> idx(static_cast<size_t>(V));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[size_t(a)] > logits[size_t(b)]; });
    idx.resize(size_t(k));
    std::vector<double> p(static_cast<size_t>(k));
    const double mx = logits[size_t(idx[0])];
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += p[size_t(i)] = std::exp((logits[size_t(idx[size_t(i)])] - mx) / temperature);
    int keep = k;
    if (top_p > 0 && top_p < 1) {
        double c = 0;
        for (int i = 0; i < k; ++i) {
            c += p[size_t(i)] / sum;
            if (c >= top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double tot = 0;
    for (int i = 0; i < keep; ++i) tot += p[size_t(i)];
    double u = std::uniform_real_distribution<double>(0, tot)(rng);
    for (int i = 0; i < keep; ++i) {
        u -= p[size_t(i)];
        if (u <= 0) return idx[size_t(i)];
    }
    return idx[size_t(keep - 1)];
}

int main(int argc, char **argv) {
    std::string model, gpus = "0";
    int chunk = 512, max_context = 32768;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> std::string { return argv[++i]; };
        if (!std::strcmp(argv[i], "--model")) model = next();
        else if (!std::strcmp(argv[i], "--gpus")) gpus = next();
        else if (!std::strcmp(argv[i], "--chunk")) chunk = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--max-context")) max_context = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 1;
        }
    }
    std::vector<int> devs;
    for (size_t p = 0; p < gpus.size();) {
        size_t q = gpus.find(',', p);
        if (q == std::string::npos) q = gpus.size();
        devs.push_back(std::atoi(gpus.substr(p, q - p).c_str()));
        p = q + 1;
    }
    try {
        double t0 = now_s();
        behemoth::QwenModel m(model, devs, chunk, max_context);
        std::fprintf(stderr, "qwen_worker: loaded in %.1f s on %zu gpu(s), max_context %d\n", now_s() - t0,
                     devs.size(), max_context);
        std::cout << json{{"ready", true}, {"max_context", max_context}}.dump() << std::endl;
        std::vector<float> logits(size_t(m.config().vocab));
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            json rq;
            try {
                rq = json::parse(line);
            } catch (const std::exception &e) {
                std::cout << json{{"done", "error"}, {"error", e.what()}}.dump() << std::endl;
                continue;
            }
            std::vector<int> toks = rq.value("tokens", std::vector<int>{});
            int max_new = rq.value("max_new", 256);
            const float temperature = rq.value("temperature", 0.0f), top_p = rq.value("top_p", 1.0f);
            const int top_k = rq.value("top_k", 0);
            std::mt19937_64 rng(rq.value("seed", uint64_t(std::random_device{}())));
            std::vector<int> stop = rq.value("stop", std::vector<int>{});
            if (toks.empty() || int(toks.size()) >= max_context) {
                std::cout << json{{"done", "error"}, {"error", "prompt empty or longer than max_context"}}.dump()
                          << std::endl;
                continue;
            }
            max_new = std::min(max_new, max_context - int(toks.size()));
            m.reset();
            double a = now_s();
            m.prefill(toks.data(), toks.size(), logits.data());
            double b = now_s();
            std::string reason = "length";
            int n = 0;
            while (n < max_new) {
                int t = sample(logits, temperature, top_p, top_k, rng);
                ++n;
                std::cout << json{{"t", t}}.dump() << std::endl;
                if (std::find(stop.begin(), stop.end(), t) != stop.end()) {
                    reason = "stop";
                    break;
                }
                if (n >= max_new) break;
                m.decode(t, logits.data());
            }
            double c = now_s();
            std::cout << json{{"done", reason}, {"prompt", toks.size()}, {"completion", n},
                              {"prefill_s", b - a}, {"decode_s", c - b}}
                             .dump()
                      << std::endl;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
