#include "behemoth/engine.hpp"
#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"
#include "behemoth/tokenizer.hpp"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <algorithm>
#include <csignal>
#include <ctime>
#include <deque>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <utility>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;
using json = nlohmann::json;
using namespace behemoth;

class Session : public std::enable_shared_from_this<Session> {
    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    http::request_parser<http::string_body> parser_;
    Engine &engine_;
    const Tokenizer &tokenizer_;
    uint64_t id_ = 0;
    std::time_t created_ = std::time(nullptr);
    bool streaming_ = false, finished_ = false, header_done_ = false, writing_ = false,
         closed_ = false, first_piece_ = true, include_usage_ = false;
    std::string output_;
    std::string diagnostic_dir_;
    json diagnostic_;
    TextStream text_;
    std::deque<std::string> queue_;
    size_t queued_ = 0;
    std::array<char, 1> disconnect_{};
    http::response<http::empty_body> header_{http::status::ok, 11};
    std::unique_ptr<http::response_serializer<http::empty_body>> serializer_;
    std::string completion_id() const {
        return "chatcmpl-" + std::to_string(id_);
    }
    json chunk(json delta, json reason = nullptr) const {
        return {
            {"id", completion_id()},
            {"object", "chat.completion.chunk"},
            {"created", created_},
            {"model", model_id},
            {"choices",
             {{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(reason)}}}}};
    }
    void close() {
        if (closed_)
            return;
        closed_ = true;
        if (id_ && !finished_)
            engine_.cancel(id_);
        beast::error_code ec;
        stream_.socket().shutdown(tcp::socket::shutdown_both, ec);
        stream_.socket().close(ec);
    }
    void watch_disconnect() {
        stream_.socket().async_read_some(
            asio::buffer(disconnect_),
            [self = shared_from_this()](beast::error_code, size_t) { self->close(); });
    }
    void respond(http::status status, json body) {
        auto response = std::make_shared<http::response<http::string_body>>(status, 11);
        response->set(http::field::content_type, "application/json");
        response->keep_alive(false);
        response->body() = body.dump();
        response->prepare_payload();
        http::async_write(
            stream_, *response,
            [self = shared_from_this(), response](beast::error_code, size_t) { self->close(); });
    }
    void write_next() {
        if (closed_ || writing_ || !header_done_)
            return;
        if (queue_.empty()) {
            if (finished_)
                close();
            return;
        }
        writing_ = true;
        asio::async_write(stream_, asio::buffer(queue_.front()),
                          [self = shared_from_this()](beast::error_code ec, size_t) {
                              self->writing_ = false;
                              if (ec) {
                                  self->close();
                                  return;
                              }
                              self->queued_ -= self->queue_.front().size();
                              self->queue_.pop_front();
                              self->write_next();
                          });
    }
    void enqueue(std::string data, bool last = false) {
        if (closed_)
            return;
        std::string wire;
        if (last)
            wire = "0\r\n\r\n";
        else {
            std::ostringstream prefix;
            prefix << std::hex << data.size();
            wire = prefix.str() + "\r\n" + data + "\r\n";
        }
        queued_ += wire.size();
        if (queued_ > 1024 * 1024) {
            close();
            return;
        }
        queue_.push_back(std::move(wire));
        write_next();
    }
    void emit(const std::string &text) {
        output_ += text;
        if (streaming_ && !text.empty())
            enqueue("data: " + chunk({{"content", text}}).dump() + "\n\n");
    }
    static json usage(const GenerationEvent &event) {
        return {{"prompt_tokens", event.prompt_tokens},
                {"completion_tokens", event.completion_tokens},
                {"total_tokens", event.prompt_tokens + event.completion_tokens},
                {"prompt_tokens_details", {{"cached_tokens", event.cached_tokens}}}};
    }
    void finish(const GenerationEvent &event, const std::string &reason) {
        if (finished_ || closed_)
            return;
        if (!diagnostic_dir_.empty()) {
            try {
                namespace fs = std::filesystem;
                fs::create_directories(diagnostic_dir_);
                auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
                auto path = fs::path(diagnostic_dir_) / (completion_id() + "-" + std::to_string(stamp) + ".json");
                auto temp = fs::path(path.string() + ".tmp");
                diagnostic_["finish_reason"] = reason;
                diagnostic_["completion_tokens"] = event.completion_tokens;
                diagnostic_["attention_qp"] = std::getenv("BEHEMOTH_ATTN_QP") ? std::getenv("BEHEMOTH_ATTN_QP") : "fp8";
                diagnostic_["attention_legacy"] = std::getenv("BEHEMOTH_ATTN_LEGACY") ? std::getenv("BEHEMOTH_ATTN_LEGACY") : "0";
                std::ofstream out(temp);
                out << diagnostic_.dump() << '\n'; out.close();
                if (!out) throw std::runtime_error("capture write failed");
                fs::rename(temp, path);
                std::cerr << "Diagnostic capture: " << path << '\n';
            } catch (const std::exception &e) { std::cerr << "Diagnostic capture failed: " << e.what() << '\n'; }
        }
        emit(text_.append("", true));
        finished_ = true;
        if (streaming_) {
            enqueue("data: " + chunk(json::object(), reason).dump() + "\n\n");
            if (include_usage_)
                enqueue("data: " +
                        json{{"id", completion_id()},
                             {"object", "chat.completion.chunk"},
                             {"created", created_},
                             {"model", model_id},
                             {"choices", json::array()},
                             {"usage", usage(event)}}
                            .dump() +
                        "\n\n");
            enqueue("data: [DONE]\n\n");
            enqueue("", true);
        } else
            respond(http::status::ok,
                    {{"id", completion_id()},
                     {"object", "chat.completion"},
                     {"created", created_},
                     {"model", model_id},
                     {"choices",
                      {{{"index", 0},
                        {"message", {{"role", "assistant"}, {"content", output_}}},
                        {"finish_reason", reason}}}},
                     {"usage", usage(event)}});
    }
    void on_event(const GenerationEvent &event) {
        if (closed_ || finished_)
            return;
        if (!event.error.empty()) {
            finished_ = true;
            json error = {{"error", {{"message", event.error}, {"type", "server_error"}}}};
            if (streaming_) {
                enqueue("data: " + error.dump() + "\n\n");
                enqueue("data: [DONE]\n\n");
                enqueue("", true);
            } else
                respond(http::status::internal_server_error, error);
            return;
        }
        if (event.token >= 0) {
            if (!diagnostic_dir_.empty()) diagnostic_["tokens"].push_back(event.token);
            emit(text_.append(tokenizer_.piece(event.token, first_piece_)));
            if (text_.stopped()) {
                engine_.cancel(id_);
                finish(event, "stop");
                return;
            }
        }
        if (event.done)
            finish(event, event.reason == "length" ? "length" : "stop");
    }
    void route() {
        try {
            const auto &req = parser_.get();
            auto target = req.target();
            if (req.method() == http::verb::get) {
                if (target == "/health") {
                    bool ok = engine_.healthy();
                    respond(ok ? http::status::ok : http::status::service_unavailable,
                            {{"healthy", ok}});
                    return;
                }
                if (target == "/metrics") {
                    respond(http::status::ok, engine_.metrics());
                    return;
                }
                if (target == "/metrics/tuning" || target == "/metrics/tuning/all") {
                    respond(http::status::ok,
                            engine_.projection_tuning(target == "/metrics/tuning/all" ? -1 : 0));
                    return;
                }
                if (target == "/v1/models") {
                    respond(http::status::ok,
                            {{"object", "list"},
                             {"data",
                              {{{"id", model_id}, {"object", "model"}, {"owned_by", "cgg507"}}}}});
                    return;
                }
            }
            if (target != "/v1/chat/completions") {
                respond(http::status::not_found, {{"error", {{"message", "Unknown endpoint"}}}});
                return;
            }
            if (req.method() != http::verb::post) {
                respond(http::status::method_not_allowed, {{"error", {{"message", "Use POST"}}}});
                return;
            }
            auto request = parse_chat(json::parse(req.body()));
            auto prompt = tokenizer_.encode(request.rendered);
            if (const char *directory = std::getenv("BEHEMOTH_DIAGNOSTIC_DIR"); directory && *directory) {
                diagnostic_dir_ = directory;
                diagnostic_ = {{"tokens", prompt}, {"prompt_tokens", prompt.size()},
                    {"rendered_prompt", request.rendered}, {"request", json::parse(req.body())},
                    {"sampling", {{"seed", request.sampling.seed}, {"temperature", request.sampling.temperature},
                                  {"top_k", request.sampling.top_k}, {"top_p", request.sampling.top_p}}},
                    {"producer", "behemoth_server exact delivered token IDs"}};
            }
            streaming_ = request.stream;
            include_usage_ = request.include_usage;
            text_ = TextStream(request.stops);
            std::weak_ptr<Session> weak = shared_from_this();
            id_ = engine_.submit(prompt, request.max_new, request.sampling,
                                 [weak](const GenerationEvent &event) {
                                     if (auto self = weak.lock())
                                         asio::post(self->stream_.get_executor(),
                                                    [self, event] { self->on_event(event); });
                                 });
            stream_.expires_never();
            watch_disconnect();
            if (streaming_) {
                header_.set(http::field::content_type, "text/event-stream");
                header_.set(http::field::cache_control, "no-cache");
                header_.keep_alive(false);
                header_.chunked(true);
                serializer_ =
                    std::make_unique<http::response_serializer<http::empty_body>>(header_);
                http::async_write_header(stream_, *serializer_,
                                         [self = shared_from_this()](beast::error_code ec, size_t) {
                                             if (ec) {
                                                 self->close();
                                                 return;
                                             }
                                             self->header_done_ = true;
                                             self->write_next();
                                         });
                enqueue("data: " + chunk({{"role", "assistant"}}).dump() + "\n\n");
            }
        } catch (const std::exception &e) {
            respond(http::status::bad_request,
                    {{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}});
        }
    }

  public:
    Session(tcp::socket socket, Engine &engine, const Tokenizer &tokenizer)
        : stream_(std::move(socket)), engine_(engine), tokenizer_(tokenizer) {
        parser_.body_limit(4 * 1024 * 1024);
    }
    void run() {
        stream_.expires_after(std::chrono::seconds(30));
        http::async_read(stream_, buffer_, parser_,
                         [self = shared_from_this()](beast::error_code ec, size_t) {
                             if (ec) {
                                 self->close();
                                 return;
                             }
                             self->route();
                         });
    }
};

class Listener : public std::enable_shared_from_this<Listener> {
    asio::io_context &io_;
    tcp::acceptor acceptor_;
    Engine &engine_;
    const Tokenizer &tokenizer_;

  public:
    Listener(asio::io_context &io, tcp::endpoint endpoint, Engine &engine,
             const Tokenizer &tokenizer)
        : io_(io), acceptor_(io), engine_(engine), tokenizer_(tokenizer) {
        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(asio::socket_base::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen(asio::socket_base::max_listen_connections);
    }
    void accept() {
        acceptor_.async_accept(
            asio::make_strand(io_),
            [self = shared_from_this()](beast::error_code ec, tcp::socket socket) {
                if (!ec)
                    std::make_shared<Session>(std::move(socket), self->engine_, self->tokenizer_)
                        ->run();
                if (self->acceptor_.is_open())
                    self->accept();
            });
    }
    void close() {
        beast::error_code ec;
        acceptor_.close(ec);
    }
};
int main(int argc, char **argv) {
    try {
        RuntimeOptions options;
        options.prefix_cache.slots = 10; // PREFIX_CACHE_REPORT.md; --prefix-cache-slots 0 disables
        // BF16 prefill exchange for pure-prefill micro-batches of >= 512 rows (PREFILL_INVESTIGATION.md,
        // PERF_ROUND5_REPORT.md: quality-neutral, +5-13 % prefill); --qtip-bf16-prefill-rows 0 disables.
        options.qtip_bf16_prefill_rows = 512;
        std::string host = "127.0.0.1";
        unsigned short port = 8080;
        int threads = 4;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("Every option requires a value");
            std::string value = argv[++i];
            if (key == "--model")
                options.model = value;
            else if (key == "--host")
                host = value;
            else if (key == "--port") {
                int p = std::stoi(value);
                if (p < 1 || p > 65535)
                    throw std::invalid_argument("Invalid port");
                port = static_cast<unsigned short>(p);
            } else if (key == "--threads")
                threads = std::stoi(value);
            else if (key == "--memory-fraction")
                options.memory_fraction = std::stod(value);
            else if (key == "--token-budget")
                options.token_budget = std::stoul(value);
            else if (key == "--prefill-chunk")
                options.prefill_chunk = std::stoul(value);
            else if (key == "--mixed-prefill-tokens")
                options.scheduler.mixed_prefill_tokens = std::stoul(value);
            else if (key == "--prefill-interval")
                options.scheduler.prefill_interval = std::stoul(value);
            else if (key == "--max-batch-requests")
                options.scheduler.max_requests = std::stoul(value);
            else if (key == "--zero-copy-rows")
                options.zero_copy_rows = std::stoul(value);
            else if (key == "--qtip-dense-rows")
                options.qtip_dense_rows = std::stoi(value);
            else if (key == "--qtip-row-transforms")
                options.qtip_row_transforms = std::stoi(value);
            else if (key == "--qtip-bf16-prefill-rows")
                options.qtip_bf16_prefill_rows = std::stoi(value);
            else if (key == "--devices") {
                auto comma = value.find(',');
                if (comma == std::string::npos)
                    throw std::invalid_argument("Specify two devices: 0,1");
                options.devices = {std::stoi(value.substr(0, comma)),
                                   std::stoi(value.substr(comma + 1))};
            } else if (key == "--transport") {
                if (value != "host" && value != "rccl-shm" && value != "zerocopy")
                    throw std::invalid_argument("Transport must be host, zerocopy or rccl-shm");
                options.rccl_shared_memory = value == "rccl-shm";
                options.zero_copy_transport = value == "zerocopy";
            } else if (key == "--projection-mode") {
                options.projection_mode = value;
            } else if (key == "--projection-search") {
                options.projection_search = value;
            } else if (key == "--code-layout") {
                options.code_layout = value;
            } else if (key == "--prefix-cache-slots") {
                options.prefix_cache.slots = std::stoul(value);
            } else if (key == "--prefix-cache-gb") {
                options.prefix_cache_gb = std::stod(value);
                if (!(options.prefix_cache_gb > 0))
                    throw std::invalid_argument("--prefix-cache-gb must be positive");
            } else if (key == "--prefix-cache-min-tokens") {
                options.prefix_cache.min_tokens = std::max<size_t>(1, std::stoul(value));
            } else if (key == "--prefix-cache-gpu") {
                if (value != "true" && value != "false" && value != "1" && value != "0")
                    throw std::invalid_argument("--prefix-cache-gpu expects true or false");
                options.prefix_cache.gpu_tier = value == "true" || value == "1";
            } else if (key == "--prefix-cache-restore-pace") {
                options.prefix_cache_restore_pace = std::stoul(value);
            } else if (key == "--prefix-cache-staging-pages") {
                options.kv_transfer_pages = std::stoi(value);
                if (options.kv_transfer_pages < 1 || options.kv_transfer_pages > 64)
                    throw std::invalid_argument("--prefix-cache-staging-pages must be 1..64");
            } else if (key == "--profile") {
                if (value != "true" && value != "false" && value != "1" && value != "0")
                    throw std::invalid_argument("--profile expects true or false");
                options.profile = value == "true" || value == "1";
            } else
                throw std::invalid_argument("Unknown option: " + key);
        }
        if (options.model.empty() || threads < 1 || threads > 32)
            throw std::invalid_argument("Usage: behemoth_server --model DIRECTORY [--transport "
                                        "host|rccl-shm] [--port 8080] [--threads 4]");
        Tokenizer tokenizer(options.model);
        Engine engine(options);
        asio::io_context io(threads);
        auto listener = std::make_shared<Listener>(
            io, tcp::endpoint(asio::ip::make_address(host), port), engine, tokenizer);
        listener->accept();
        asio::signal_set signals(io, SIGINT, SIGTERM);
        signals.async_wait([&](beast::error_code, int) {
            listener->close();
            engine.shutdown();
            io.stop();
        });
        std::cerr << "Listening on http://" << host << ':' << port
                  << "; transport="
                  << (options.rccl_shared_memory  ? "rccl-shm"
                      : options.zero_copy_transport ? "zerocopy"
                                                    : "pinned-host")
                  << "; P2P disabled\n";
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i)
            pool.emplace_back([&] { io.run(); });
        io.run();
        for (auto &t : pool)
            t.join();
        engine.shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Server failed: " << e.what() << '\n';
        return 1;
    }
}
