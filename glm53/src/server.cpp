// Serial text server for the CPU reference and CUDA/HIP backends. It intentionally does
// not reuse Behemoth's fixed-size GQA scheduler/cache or its 32768-token sampler.
#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"
#if defined(BEHEMOTH_QTIP_CUDA)
#include "behemoth/cuda.hpp"
#else
#include "glm53.hpp"
#endif
#include <algorithm>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

namespace asio = boost::asio;
namespace http = boost::beast::http;
using tcp = asio::ip::tcp;
#if defined(BEHEMOTH_QTIP_CUDA)
using namespace behemoth::cuda_runtime;
constexpr bool behemoth_cuda = true;
#else
using namespace glm53;
constexpr bool behemoth_cuda = false;
#endif
namespace {
int sample(const Vec &logits, float temperature, int topk, double topp, std::mt19937_64 &rng) {
    for (float x : logits)
        check(std::isfinite(x), "nonfinite logits");
    if (temperature == 0)
        return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; });
    if (topk > 0)
        order.resize(std::min(size_t(topk), order.size()));
    std::vector<double> p(order.size());
    double sum = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        p[i] = std::exp((double(logits[order[i]]) - logits[order[0]]) / temperature);
        sum += p[i];
    }
    double mass = 0;
    size_t keep = 0;
    do {
        mass += p[keep++];
    } while (keep < p.size() && mass < topp * sum);
    double choice = std::generate_canonical<double, 53>(rng) * mass;
    for (size_t i = 0; i < keep; ++i) {
        choice -= p[i];
        if (choice < 0)
            return order[i];
    }
    return order[keep - 1];
}
void reply(tcp::socket &socket, http::status status, const json &body) {
    http::response<http::string_body> response(status, 11);
    response.set(http::field::content_type, "application/json");
    response.body() = body.dump();
    response.prepare_payload();
    http::write(socket, response);
}
void event(tcp::socket &socket, const std::string &body) {
    std::ostringstream length;
    length << std::hex << body.size();
    auto wire = length.str() + "\r\n" + body + "\r\n";
    asio::write(socket, asio::buffer(wire));
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (std::string(argv[i]) == "--help") {
                std::cout << (behemoth_cuda ? "behemoth_cuda_server" : "glm53_server")
                          << " --model DIR [--host 127.0.0.1] [--port 8080] [--name MODEL] [--device 0]"
                          << (behemoth_cuda ? " [--context 4096] [--tp 1|2]\n" : "\n");
                return 0;
            }
            check(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "missing/duplicate argument");
        }
        for (const auto &[k, v] : args) {
            (void)v;
            check(k == "--model" || k == "--host" || k == "--port" || k == "--name" || k == "--device" ||
                      (behemoth_cuda && (k == "--context" || k == "--tp")),
                  "unknown option " + k);
        }
#if defined(BEHEMOTH_QTIP_CUDA)
        auto integer = [&](const char *key, int fallback) {
            if (!args.contains(key))
                return fallback;
            size_t used = 0;
            int value = std::stoi(args.at(key), &used);
            check(used == args.at(key).size(), "invalid integer option");
            return value;
        };
        Model model(args.at("--model"), integer("--device", 0), integer("--context", 4096),
                    integer("--tp", 1));
        Tokenizer tokenizer(args.at("--model"));
#else
        Model model(args.at("--model"));
        Tokenizer tokenizer(args.at("--model"));
        if (args.contains("--device") && args.at("--device") != "cpu") {
#if defined(GLM53_HIP) || defined(GLM53_CUDA)
            size_t used = 0;
            int device = std::stoi(args.at("--device"), &used);
            check(used == args.at("--device").size() && device >= 0, "invalid device");
            model.set_backend(gpu_backend(args.at("--model"), device));
#else
            throw std::runtime_error("rebuild with GLM53_HIP=ON or GLM53_CUDA=ON for GPU inference");
#endif
        }
#endif
        auto name = args.contains("--name") ? args.at("--name") : behemoth_cuda ? "behemoth-qtip" : "glm53";
        auto host = args.contains("--host") ? args.at("--host") : "127.0.0.1";
        size_t used = 0;
        std::string port_text = args.contains("--port") ? args.at("--port") : "8080";
        int port = std::stoi(port_text, &used);
        check(used == port_text.size() && port > 0 && port <= 65535, "invalid port");
        asio::io_context io;
        tcp::acceptor acceptor(io, {asio::ip::make_address(host), static_cast<unsigned short>(port)});
        std::cerr << (behemoth_cuda ? "Behemoth CUDA server " : "GLM53 server ") << host << ':' << port
                  << "; serialized requests; "
                  << (behemoth_cuda || (args.contains("--device") && args.at("--device") != "cpu")
                          ? "GPU inference"
                          : "CPU reference")
                  << '\n';
        uint64_t serial = 0;
        for (;;) {
            tcp::socket socket(io);
            acceptor.accept(socket);
            bool streaming = false, headers = false;
            try {
                http::request_parser<http::string_body> parser;
                parser.body_limit(4 * 1024 * 1024);
                boost::beast::flat_buffer buffer;
                http::read(socket, buffer, parser);
                const auto &request = parser.get();
                if (request.method() == http::verb::get && request.target() == "/health") {
                    reply(socket, http::status::ok, {{"healthy", true}});
                    continue;
                }
                if (request.method() == http::verb::get && request.target() == "/v1/models") {
                    reply(socket, http::status::ok,
                          {{"object", "list"}, {"data", json::array({{{"id", name}, {"object", "model"}}})}});
                    continue;
                }
                check(request.method() == http::verb::post && request.target() == "/v1/chat/completions",
                      "expected POST /v1/chat/completions");
                auto j = json::parse(request.body());
                check(j.value("model", name) == name, "unknown model");
                check(j.value("n", 1) == 1, "only one completion supported");
                for (auto key :
                     {"tools", "tool_choice", "response_format", "logprobs", "logit_bias", "functions"})
                    check(!j.contains(key), std::string("unsupported parameter ") + key);
                check(j.value("frequency_penalty", 0.f) == 0 && j.value("presence_penalty", 0.f) == 0,
                      "penalties are not supported");
                std::string prompt = tokenizer.chat(j.at("messages"));
                auto ids = tokenizer.encode(prompt);
                check(!ids.empty(), "empty prompt");
                for (int id : ids)
                    check(id >= 0 && id < model.config.vocab, "tokenizer/model vocabulary mismatch");
                check(!(j.contains("max_tokens") && j.contains("max_completion_tokens")),
                      "specify one token limit");
                auto limit = j.contains("max_completion_tokens") ? j.at("max_completion_tokens")
                                                                 : j.value("max_tokens", json(128));
                check(limit.is_number_integer(), "token limit must be an integer");
                int steps = limit.get<int>();
                check(steps > 0 && ids.size() + size_t(steps) <= size_t(model.config.max_context),
                      "prompt/output exceed context limit");
                float temperature = j.value("temperature", 0.f);
                double topp = j.value("top_p", 1.);
                int topk = j.value("top_k", 0);
                check(std::isfinite(temperature) && temperature >= 0 && std::isfinite(topp) && topp > 0 &&
                          topp <= 1 && topk >= 0 && topk <= model.config.vocab,
                      "invalid sampling settings");
                uint64_t seed = j.contains("seed")
                                    ? j.at("seed").get<uint64_t>()
                                    : uint64_t(std::random_device{}()) << 32 ^ std::random_device{}();
                std::mt19937_64 rng(seed);
                std::vector<std::string> stops;
                if (j.contains("stop")) {
                    if (j["stop"].is_string())
                        stops.push_back(j["stop"]);
                    else
                        stops = j["stop"].get<std::vector<std::string>>();
                }
                check(stops.size() <= 16, "too many stops");
                for (const auto &s : stops)
                    check(!s.empty(), "empty stop string");
                streaming = j.value("stream", false);
                std::string id = "chatcmpl-glm53-" + std::to_string(++serial);
                auto chunk = [&](json delta, json finish = nullptr) {
                    return json(
                        {{"id", id},
                         {"object", "chat.completion.chunk"},
                         {"model", name},
                         {"choices",
                          json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}});
                };
                if (streaming) {
                    http::response<http::empty_body> h(http::status::ok, 11);
                    h.set(http::field::content_type, "text/event-stream");
                    h.chunked(true);
                    http::response_serializer<http::empty_body> s(h);
                    http::write_header(socket, s);
                    headers = true;
                    event(socket, "data: " + chunk({{"role", "assistant"}}).dump() + "\n\n");
                }
                model.reset();
                Vec logits;
#if defined(BEHEMOTH_QTIP_CUDA)
                logits = model.prefill(ids);
#else
                for (int token : ids)
                    logits = model.step(token);
#endif
                behemoth::TextStream text(stops);
                behemoth::ReasoningStream reasoning;
                if (prompt.ends_with("<think>"))
                    (void)reasoning.append("<think>");
                std::string answer, thoughts;
                int count = 0;
                [[maybe_unused]] bool first_piece = true;
                std::string finish = "length";
                auto emit = [&](const std::string &s, bool end = false) {
                    for (const auto &p : reasoning.append(s, end)) {
                        (p.reasoning ? thoughts : answer) += p.text;
                        if (streaming)
                            event(
                                socket,
                                "data: " +
                                    chunk({{p.reasoning ? "reasoning_content" : "content", p.text}}).dump() +
                                    "\n\n");
                    }
                };
                for (int i = 0; i < steps; ++i) {
                    int token = sample(logits, temperature, topk, topp, rng);
                    ++count;
                    if (std::find(model.config.eos.begin(), model.config.eos.end(), token) !=
                        model.config.eos.end()) {
                        finish = "stop";
                        break;
                    }
#if defined(BEHEMOTH_QTIP_CUDA)
                    emit(text.append(tokenizer.piece(token, first_piece, true)));
#else
                    emit(text.append(tokenizer.decode({token}, true)));
#endif
                    if (text.stopped()) {
                        finish = "stop";
                        break;
                    }
                    if (i + 1 < steps)
                        logits = model.step(token);
                }
                emit(text.append("", true), true);
                json usage = {{"prompt_tokens", ids.size()},
                              {"completion_tokens", count},
                              {"total_tokens", ids.size() + count}};
                if (streaming) {
                    event(socket, "data: " + chunk(json::object(), finish).dump() + "\n\n");
                    if (j.value("stream_options", json::object()).value("include_usage", false))
                        event(socket, "data: " +
                                          json({{"id", id},
                                                {"object", "chat.completion.chunk"},
                                                {"model", name},
                                                {"choices", json::array()},
                                                {"usage", usage}})
                                              .dump() +
                                          "\n\n");
                    event(socket, "data: [DONE]\n\n");
                    asio::write(socket, asio::buffer(std::string("0\r\n\r\n")));
                } else
                    reply(socket, http::status::ok,
                          {{"id", id},
                           {"object", "chat.completion"},
                           {"model", name},
                           {"choices", json::array({{{"index", 0},
                                                     {"message",
                                                      {{"role", "assistant"},
                                                       {"content", answer},
                                                       {"reasoning_content", thoughts}}},
                                                     {"finish_reason", finish}}})},
                           {"usage", usage}});
            } catch (const std::exception &e) {
                try {
                    if (!headers)
                        reply(socket, http::status::bad_request, {{"error", {{"message", e.what()}}}});
                    else {
                        event(socket, "data: " + json({{"error", {{"message", e.what()}}}}).dump() + "\n\n");
                        event(socket, "data: [DONE]\n\n");
                        asio::write(socket, asio::buffer(std::string("0\r\n\r\n")));
                    }
                } catch (...) {
                }
                std::cerr << "request failed: " << e.what() << '\n';
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "glm53_server: " << e.what() << '\n';
        return 1;
    }
}
