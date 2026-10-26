#include "behemoth/protocol.hpp"
#include <atomic>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <utility>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using json = nlohmann::json;
using tcp = asio::ip::tcp;
json call(const std::string &host, const std::string &port, const std::string &target,
          const json *body) {
    asio::io_context io;
    tcp::resolver resolver(io);
    beast::tcp_stream stream(io);
    stream.expires_after(std::chrono::minutes(60));
    stream.connect(resolver.resolve(host, port));
    http::request<http::string_body> request(body ? http::verb::post : http::verb::get, target, 11);
    request.set(http::field::host, host);
    request.set(http::field::content_type, "application/json");
    if (body)
        request.body() = body->dump();
    request.prepare_payload();
    http::write(stream, request);
    beast::flat_buffer buffer;
    http::response_parser<http::string_body> parser;
    parser.body_limit(16 * 1024 * 1024);
    http::read(stream, buffer, parser);
    if (parser.get().result() != http::status::ok)
        throw std::runtime_error(parser.get().body());
    return json::parse(parser.get().body());
}
int main(int argc, char **argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Usage: behemoth_load_bench PROMPT_FILE CONCURRENCY [REQUESTS=4*concurrency] "
                   "[OUTPUT_TOKENS=128] [HOST=127.0.0.1] [PORT=8080]\n";
            return 2;
        }
        std::ifstream file(argv[1]);
        if (!file)
            throw std::invalid_argument("Cannot open prompt file");
        std::string prompt((std::istreambuf_iterator<char>(file)), {});
        int concurrency = std::stoi(argv[2]),
            requests = argc > 3 ? std::stoi(argv[3]) : concurrency * 4,
            output = argc > 4 ? std::stoi(argv[4]) : 128;
        std::string host = argc > 5 ? argv[5] : "127.0.0.1", port = argc > 6 ? argv[6] : "8080";
        if (concurrency < 1 || concurrency > 1024 || requests < 1 || output < 1)
            throw std::invalid_argument("Invalid benchmark dimensions");
        json request = {{"model", behemoth::model_id},
                        {"messages", {{{"role", "user"}, {"content", prompt}}}},
                        {"max_tokens", output},
                        {"temperature", 0}};
        std::atomic<int> next = 0;
        std::atomic<uint64_t> completion_tokens = 0, prompt_tokens = 0;
        std::mutex mutex;
        std::vector<double> latency;
        std::vector<std::future<void>> workers;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < concurrency; ++i)
            workers.push_back(std::async(std::launch::async, [&] {
                while (next.fetch_add(1) < requests) {
                    auto begin = std::chrono::steady_clock::now();
                    auto response = call(host, port, "/v1/chat/completions", &request);
                    completion_tokens +=
                        response.at("usage").at("completion_tokens").get<uint64_t>();
                    prompt_tokens += response.at("usage").at("prompt_tokens").get<uint64_t>();
                    double seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
                            .count();
                    std::lock_guard lock(mutex);
                    latency.push_back(seconds);
                }
            }));
        for (auto &job : workers)
            job.get();
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::sort(latency.begin(), latency.end());
        auto pct = [&](double p) { return latency.at(size_t(p * double(latency.size() - 1))); };
        json report = {{"requests", requests},
                       {"concurrency", concurrency},
                       {"elapsed_seconds", seconds},
                       {"completion_tokens", completion_tokens.load()},
                       {"mean_prompt_tokens", double(prompt_tokens.load()) / requests},
                       {"completion_tokens_per_second", double(completion_tokens.load()) / seconds},
                       {"request_latency_p50_seconds", pct(.5)},
                       {"request_latency_p95_seconds", pct(.95)},
                       {"request_latency_p99_seconds", pct(.99)},
                       {"server_rolling_metrics", call(host, port, "/metrics", nullptr)}};
        std::cout << report.dump(2) << '\n';
    } catch (const std::exception &e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
