#include "behemoth/protocol.hpp"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <utility>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;
using json = nlohmann::json;
struct Reply {
    unsigned status;
    std::string body;
};
static std::string host = "127.0.0.1", port = "8080";
static void check(bool value, const char *what) {
    if (!value)
        throw std::runtime_error(what);
}
static Reply call(const std::string &target, const json *body = nullptr, bool disconnect = false) {
    asio::io_context io;
    tcp::resolver resolver(io);
    beast::tcp_stream stream(io);
    stream.expires_after(std::chrono::minutes(5));
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
    parser.body_limit(8 * 1024 * 1024);
    if (disconnect) {
        http::read_header(stream, buffer, parser);
        beast::error_code ec;
        stream.socket().shutdown(tcp::socket::shutdown_both, ec);
        stream.socket().close(ec);
        return {parser.get().result_int(), ""};
    }
    http::read(stream, buffer, parser);
    return {parser.get().result_int(), parser.get().body()};
}
int main(int argc, char **argv) {
    try {
        if (argc > 1)
            host = argv[1];
        if (argc > 2)
            port = argv[2];
        check(call("/health").status == 200, "Server not healthy");
        auto models = call("/v1/models");
        check(models.status == 200 &&
                  json::parse(models.body).at("data")[0].at("id") == behemoth::model_id,
              "Wrong served model");
        json request = {{"model", behemoth::model_id},
                        {"messages", {{{"role", "user"}, {"content", "Say hello."}}}},
                        {"max_tokens", 2},
                        {"temperature", 0},
                        {"seed", 101}};
        auto invalid = request;
        invalid["tools"] = json::array();
        check(call("/v1/chat/completions", &invalid).status == 400, "Unsupported tools accepted");
        auto ordinary = call("/v1/chat/completions", &request);
        check(ordinary.status == 200, "Nonstreaming generation failed");
        auto body = json::parse(ordinary.body);
        auto text = body.at("choices")[0].at("message").at("content").get<std::string>();
        check(body.at("usage").at("completion_tokens").get<int>() <= 2,
              "Output-token limit violated");
        auto tuning_reply = call("/metrics/tuning");
        check(tuning_reply.status == 200, "Tuning snapshot endpoint failed");
        auto tuning = json::parse(tuning_reply.body);
        check(tuning.at("layer") == 0 && tuning.at("ranks").size() == 2,
              "Tuning snapshot has wrong layer/ranks");
        for (const auto &rank : tuning.at("ranks")) {
            check(rank.at("selections").size() == 8, "Tuning snapshot omitted projections");
            for (const auto &selection : rank.at("selections")) {
                const auto expected_rows = selection.at("weight_name") == "lm_head"
                                               ? tuning.at("last_step_sample_rows")
                                               : tuning.at("last_step_query_rows");
                check(selection.at("rows") == expected_rows,
                      "Tuning snapshot included an inactive batch size");
            }
        }
        request["stream"] = true;
        auto streamed = call("/v1/chat/completions", &request);
        check(streamed.status == 200, "Streaming generation failed");
        size_t begin = 0;
        bool done = false;
        std::string assembled;
        while (begin < streamed.body.size()) {
            size_t end = streamed.body.find("\n\n", begin);
            check(end != std::string::npos, "Incomplete SSE event");
            auto line = streamed.body.substr(begin, end - begin);
            begin = end + 2;
            check(line.starts_with("data: "), "Malformed SSE event");
            auto payload = line.substr(6);
            if (payload == "[DONE]") {
                check(!done && begin == streamed.body.size(), "Duplicate/premature DONE event");
                done = true;
                continue;
            }
            check(!done, "Output after DONE");
            auto event = json::parse(payload);
            check(!event.contains("error"), "Streaming error");
            auto delta = event.at("choices")[0].at("delta");
            if (delta.contains("content"))
                assembled += delta.at("content").get<std::string>();
        }
        check(done, "Missing DONE event");
        check(assembled == text, "Seeded streaming/nonstreaming output differs");
        request["stream"] = false;
        std::vector<std::future<Reply>> concurrent;
        for (int i = 0; i < 4; ++i)
            concurrent.push_back(std::async(
                std::launch::async, [&] { return call("/v1/chat/completions", &request); }));
        for (auto &job : concurrent) {
            auto r = job.get();
            check(r.status == 200, "Concurrent request failed");
            check(json::parse(r.body).contains("choices"), "Concurrent response lacks choices");
        }
        if (!text.empty() && !text.starts_with("\xef\xbf\xbd")) {
            uint8_t c = uint8_t(text[0]);
            size_t width = c < 128 ? 1 : (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : 4;
            request["stop"] = text.substr(0, width);
            auto stopped = call("/v1/chat/completions", &request);
            check(stopped.status == 200, "Stop-string request failed");
            check(json::parse(stopped.body).at("choices")[0].at("message").at("content") == "",
                  "Stop string leaked to output");
            request.erase("stop");
        }
        request["stream"] = true;
        request["max_tokens"] = 128;
        check(call("/v1/chat/completions", &request, true).status == 200,
              "Disconnect setup failed");
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        json metrics;
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto r = call("/metrics");
            check(r.status == 200, "Metrics request failed");
            metrics = json::parse(r.body);
            if (metrics.at("active_requests") == 0)
                break;
        } while (std::chrono::steady_clock::now() < deadline);
        check(metrics.at("active_requests") == 0, "Disconnected request did not cancel");
        check(metrics.at("kv_pages_free") == metrics.at("kv_pages_total"), "KV pages leaked");
        std::cout << "PASS live HTTP endpoints, validation, streaming, deterministic decoding, "
                     "concurrency, stop strings and disconnect reclamation\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL (live-server acceptance NOT passed): " << e.what() << '\n';
        return 1;
    }
}
