// Drives behemoth::Engine (Neuron backend) with token-id requests from a JSON file, all submitted at once so the
// scheduler batches them, and prints every request's generated tokens as JSON.
//   requests.json: [{"tokens":[...], "max_new":16, "temperature":0, "top_k":0, "top_p":1, "seed":0, "cancel_after":N?}]
//   neuron_engine_cli MODEL_DIR requests.json [sequential]   (env BEHEMOTH_NEURON_NEFF_DIR=bucket dir)
// `sequential` runs one request at a time (batch-invariance baseline).
#include "behemoth/engine.hpp"
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>

using namespace behemoth;

int main(int argc, char **argv) {
    if (argc < 3) { std::cerr << "usage: neuron_engine_cli MODEL_DIR requests.json\n"; return 2; }
    RuntimeOptions options;
    options.model = argv[1];
    options.token_budget = 1024;
    options.prefill_chunk = 256;
    std::ifstream f(argv[2]);
    const auto requests = nlohmann::json::parse(f);
    Engine engine(options);
    std::mutex mutex;
    std::condition_variable cv;
    size_t open = requests.size();
    std::vector<nlohmann::json> out(requests.size());
    std::map<uint64_t, size_t> index;
    std::vector<uint64_t> ids(requests.size());
    std::vector<std::chrono::steady_clock::time_point> submitted(requests.size());
    const auto start = std::chrono::steady_clock::now();
    const bool sequential = argc > 3 && std::string(argv[3]) == "sequential";
    for (size_t i = 0; i < requests.size(); ++i) {
        if (sequential) { // wait for everything submitted so far
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return open == requests.size() - i; });
        }
        const auto &r = requests[i];
        SamplingOptions s;
        s.temperature = r.value("temperature", 0.f);
        s.top_k = r.value("top_k", 0);
        s.top_p = r.value("top_p", 1.f);
        s.seed = r.value("seed", uint64_t(0));
        out[i] = {{"tokens", nlohmann::json::array()}};
        std::lock_guard lock(mutex);
        submitted[i] = std::chrono::steady_clock::now();
        ids[i] = engine.submit(r.at("tokens").get<std::vector<int>>(), r.value("max_new", size_t(16)), s, [&, i](const GenerationEvent &e) {
            std::lock_guard lock(mutex);
            if (e.token >= 0) {
                if (out[i]["tokens"].empty())
                    out[i]["first_token_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submitted[i]).count();
                out[i]["tokens"].push_back(e.token);
                out[i]["last_token_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submitted[i]).count();
            }
            if (e.done) {
                out[i]["finish"] = e.reason;
                if (!e.error.empty()) out[i]["error"] = e.error;
                out[i]["prompt_tokens"] = e.prompt_tokens;
                --open;
                cv.notify_all();
            }
        });
    }
    {
        std::unique_lock lock(mutex);
        // cancel_after: cancel a request once it has produced that many tokens
        while (open) {
            cv.wait_for(lock, std::chrono::milliseconds(20));
            for (size_t i = 0; i < requests.size(); ++i)
                if (requests[i].contains("cancel_after") && out[i]["tokens"].size() >= requests[i]["cancel_after"].get<size_t>() && !out[i].contains("finish"))
                    engine.cancel(ids[i]);
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << nlohmann::json{{"requests", out}, {"seconds", seconds}, {"metrics", engine.metrics()}}.dump(1) << std::endl;
    engine.shutdown();
    return 0;
}
