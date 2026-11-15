#include "behemoth/constraint.hpp"
#include "behemoth/engine.hpp"
#include "behemoth/protocol.hpp"
#include "behemoth/sampling.hpp"
#include "behemoth/tokenizer.hpp"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
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
#include <unistd.h>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;
using json = nlohmann::json;
using namespace behemoth;

// Structured outputs (constraint.hpp): the vocabulary's byte pieces, built once, and
// compiled grammars reused across requests.
struct StructuredContext {
    Vocabulary vocabulary;
    GrammarCache grammars;
    explicit StructuredContext(const Tokenizer &tokenizer) : vocabulary(vocabulary_pieces(tokenizer), vocabulary_eos(tokenizer)) {}
};

// Requests between submit and their request_metrics line (concurrency in the log); BEHEMOTH_REQUEST_LOG=0 silences it.
static std::atomic<int> active_requests{0};
static bool request_log_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("BEHEMOTH_REQUEST_LOG");
        return !(v && (std::string(v) == "0" || std::string(v) == "false" || std::string(v) == "off"));
    }();
    return on;
}

class Session : public std::enable_shared_from_this<Session> {
    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    http::request_parser<http::string_body> parser_;
    Engine &engine_;
    const Tokenizer &tokenizer_;
    StructuredContext &structured_;
    bool constrained_ = false;
    ThinkSplitter splitter_;
    // Tool calling (tools.hpp): the answer's split into content and tool calls, and whether the format's control
    // tokens are streamed as their marker text (requests with tools or a constraint, which sees them as text).
    std::unique_ptr<ToolCallParser> tool_parser_;
    bool markers_ = false;
    bool answer_lead_ = false; // the whitespace between a prompt-opened think block's </think> and the answer is dropped
    uint64_t id_ = 0;
    std::time_t created_ = std::time(nullptr);
    bool streaming_ = false, finished_ = false, header_done_ = false, writing_ = false,
         closed_ = false, first_piece_ = true, include_usage_ = false;
    std::string output_, reasoning_output_;
    // Per-request metrics (one request_metrics line to stderr when the request ends; never any text)
    using Clock = std::chrono::steady_clock;
    Clock::time_point submitted_at_, first_token_at_, last_token_at_;
    bool counted_ = false, logged_ = false, got_token_ = false;
    int active_start_ = 0;
    GenerationEvent seen_; // the latest event: token counts, queue wait and speculation counters
    ReasoningStream reasoning_;
    std::string diagnostic_dir_;
    json diagnostic_;
    TextStream text_;
    std::deque<std::string> queue_;
    size_t queued_ = 0;
    asio::steady_timer heartbeat_;
    std::chrono::steady_clock::time_point last_sent_ = std::chrono::steady_clock::now();
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
            {"model", model_name()},
            {"choices",
             {{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(reason)}}}}};
    }
    void close() {
        if (closed_)
            return;
        closed_ = true;
        heartbeat_.cancel();
        if (id_ && !finished_) {
            engine_.cancel(id_);
            log_request("cancelled", "client_disconnect");
        }
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
        last_sent_ = std::chrono::steady_clock::now();
        if (queued_ > 1024 * 1024) {
            close();
            return;
        }
        queue_.push_back(std::move(wire));
        write_next();
    }
    // A stream silent for 2 s (a long prefill, a tool call held until it parses) gets an empty chunk: clients
    // with an idle timeout (OpenCode: 5 s) otherwise drop the request and retry it from scratch.
    void heartbeat() {
        heartbeat_.expires_after(std::chrono::seconds(1));
        heartbeat_.async_wait([self = shared_from_this()](beast::error_code ec) {
            if (ec || self->closed_ || self->finished_)
                return;
            if (std::chrono::steady_clock::now() - self->last_sent_ >= std::chrono::seconds(2))
                self->enqueue("data: " + self->chunk(json::object()).dump() + "\n\n");
            self->heartbeat();
        });
    }
    // Answer text: content, or with tools the parser's content and tool_calls deltas.
    void answer(const std::string &text, bool finish) {
        if (!tool_parser_) {
            if (text.empty())
                return;
            output_ += text;
            if (streaming_)
                enqueue("data: " + chunk({{"content", text}}).dump() + "\n\n");
            return;
        }
        auto delta = tool_parser_->feed(text, finish);
        output_ += delta.content;
        if (streaming_ && !delta.content.empty())
            enqueue("data: " + chunk({{"content", delta.content}}).dump() + "\n\n");
        if (streaming_ && !delta.tool_calls.empty())
            enqueue("data: " + chunk({{"tool_calls", delta.tool_calls}}).dump() + "\n\n");
    }
    void emit(const std::string &text, bool finish = false) {
        if (constrained_) {
            // Same split as the decoding gate: one leading think block, then the answer.
            std::string reasoning, answer_text;
            splitter_.feed(text, &reasoning, &answer_text, finish);
            if (!reasoning.empty()) {
                reasoning_output_ += reasoning;
                if (streaming_)
                    enqueue("data: " + chunk({{"reasoning_content", reasoning}}).dump() + "\n\n");
            }
            answer(answer_text, finish);
            return;
        }
        for (const auto &part : reasoning_.append(text, finish)) {
            if (!part.reasoning) {
                const size_t lead = answer_lead_ ? std::min(part.text.find_first_not_of(" \t\r\n"), part.text.size()) : 0;
                answer_lead_ = answer_lead_ && lead == part.text.size();
                answer(lead ? part.text.substr(lead) : part.text, false);
                continue;
            }
            reasoning_output_ += part.text;
            if (streaming_)
                enqueue("data: " + chunk({{"reasoning_content", part.text}}).dump() + "\n\n");
        }
        if (finish)
            answer("", true);
    }
    static json usage(const GenerationEvent &event) {
        return {{"prompt_tokens", event.prompt_tokens},
                {"completion_tokens", event.completion_tokens},
                {"total_tokens", event.prompt_tokens + event.completion_tokens},
                {"prompt_tokens_details", {{"cached_tokens", event.cached_tokens}}}};
    }
    static double ms(Clock::time_point from, Clock::time_point to) {
        return std::round(std::chrono::duration<double, std::milli>(to - from).count() * 100) / 100;
    }
    // The request's timings (ms; null when unknown), also returned as `timings` next to `usage`. TTFT runs from submit;
    // prefill is the part after the request was first scheduled; decode is first to last token.
    json timings(const GenerationEvent &event) const {
        const double queue = event.queue_wait_ms >= 0 ? std::round(event.queue_wait_ms * 100) / 100 : -1;
        json t = {{"queue_wait_ms", queue >= 0 ? json(queue) : json(nullptr)}, {"ttft_ms", nullptr}, {"prefill_ms", nullptr},
                  {"prefill_tokens", event.prompt_tokens - std::min(event.cached_tokens, event.prompt_tokens)},
                  {"prefill_tokens_per_s", nullptr}, {"decode_ms", nullptr}, {"decode_tokens_per_s", nullptr},
                  {"total_ms", ms(submitted_at_, Clock::now())}};
        if (got_token_) {
            const double ttft = ms(submitted_at_, first_token_at_);
            t["ttft_ms"] = ttft;
            if (queue >= 0 && ttft >= queue) {
                const double prefill = std::round((ttft - queue) * 100) / 100;
                t["prefill_ms"] = prefill;
                if (prefill > 0)
                    t["prefill_tokens_per_s"] = std::round(double(t["prefill_tokens"].get<size_t>()) / prefill * 1e5) / 100;
            }
            const double decode = ms(first_token_at_, last_token_at_);
            t["decode_ms"] = decode;
            if (event.completion_tokens > 1 && decode > 0)
                t["decode_tokens_per_s"] = std::round(double(event.completion_tokens - 1) / decode * 1e5) / 100;
        }
        return t;
    }
    // One line per request, once: `request_metrics {json}`. status: ok | error | cancelled (client disconnect).
    // active_start / active_end count the in-flight requests including this one at submit and at the end.
    json log_request(const std::string &status, const std::string &reason, const GenerationEvent *final_event = nullptr,
                     const std::string &error = "") {
        if (logged_ || !counted_)
            return nullptr;
        logged_ = true;
        const GenerationEvent &event = final_event ? *final_event : seen_;
        json t = timings(event);
        const int active_end = active_requests.fetch_sub(1);
        if (!request_log_enabled())
            return t;
        const auto now = std::chrono::system_clock::now();
        const std::time_t sec = std::chrono::system_clock::to_time_t(now);
        char stamp[32];
        std::tm utc{};
        gmtime_r(&sec, &utc);
        std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S", &utc);
        const int millis = int(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
        json line = {{"id", completion_id()}, {"ts", std::string(stamp) + "." + std::to_string(1000 + millis).substr(1) + "Z"},
                     {"status", status}, {"finish_reason", reason}, {"stream", streaming_},
                     {"prompt_tokens", event.prompt_tokens}, {"completion_tokens", event.completion_tokens},
                     {"cached_tokens", event.cached_tokens}};
        for (auto it = t.begin(); it != t.end(); ++it)
            line[it.key()] = it.value();
        line["active_start"] = active_start_;
        line["active_end"] = active_end;
        if (event.drafted > 0) // speculation (prompt lookup or MTP) drafted for this request
            line["spec"] = {{"steps", event.spec_steps}, {"drafted", event.drafted}, {"accepted", event.accepted},
                            {"tokens_per_step", event.spec_steps ? std::round(double(event.completion_tokens) / double(event.spec_steps) * 100) / 100 : 0.0}};
        if (!error.empty())
            line["error"] = error;
        // one write(2) per line: concurrent requests finishing together must not interleave their lines
        const std::string out = "request_metrics " + line.dump() + "\n";
        for (size_t off = 0; off < out.size();) {
            const ssize_t n = ::write(STDERR_FILENO, out.data() + off, out.size() - off);
            if (n <= 0)
                break;
            off += size_t(n);
        }
        return t;
    }
    void finish(const GenerationEvent &event, std::string reason) {
        if (finished_ || closed_)
            return;
        emit(text_.append("", true), true);
        const bool calls = tool_parser_ && !tool_parser_->calls().empty();
        if (calls && reason == "stop")
            reason = "tool_calls";
        if (!diagnostic_dir_.empty()) {
            try {
                namespace fs = std::filesystem;
                fs::create_directories(diagnostic_dir_);
                auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
                auto path = fs::path(diagnostic_dir_) / (completion_id() + "-" + std::to_string(stamp) + ".json");
                auto temp = fs::path(path.string() + ".tmp");
                diagnostic_["finish_reason"] = reason;
                diagnostic_["completion_tokens"] = event.completion_tokens;
                diagnostic_["content"] = output_;
                diagnostic_["reasoning_content"] = reasoning_output_;
                diagnostic_["attention_qp"] = std::getenv("BEHEMOTH_ATTN_QP") ? std::getenv("BEHEMOTH_ATTN_QP") : "fp8";
                diagnostic_["attention_legacy"] = std::getenv("BEHEMOTH_ATTN_LEGACY") ? std::getenv("BEHEMOTH_ATTN_LEGACY") : "0";
                std::ofstream out(temp);
                out << diagnostic_.dump() << '\n'; out.close();
                if (!out) throw std::runtime_error("capture write failed");
                fs::rename(temp, path);
                std::cerr << "Diagnostic capture: " << path << '\n';
            } catch (const std::exception &e) { std::cerr << "Diagnostic capture failed: " << e.what() << '\n'; }
        }
        finished_ = true;
        GenerationEvent last = event;
        last.queue_wait_ms = event.queue_wait_ms >= 0 ? event.queue_wait_ms : seen_.queue_wait_ms;
        json time = log_request("ok", reason, &last);
        if (streaming_) {
            enqueue("data: " + chunk(json::object(), reason).dump() + "\n\n");
            if (include_usage_)
                enqueue("data: " +
                        json{{"id", completion_id()},
                             {"object", "chat.completion.chunk"},
                             {"created", created_},
                             {"model", model_name()},
                             {"choices", json::array()},
                             {"usage", usage(event)},
                             {"timings", time}}
                            .dump() +
                        "\n\n");
            enqueue("data: [DONE]\n\n");
            enqueue("", true);
        } else {
            json message = {{"role", "assistant"}, {"content", output_}, {"reasoning_content", reasoning_output_}};
            if (calls) {
                message["tool_calls"] = json::array();
                for (const auto &c : tool_parser_->calls())
                    message["tool_calls"].push_back(c.openai());
                if (output_.empty())
                    message["content"] = nullptr;
            }
            respond(http::status::ok,
                    {{"id", completion_id()},
                     {"object", "chat.completion"},
                     {"created", created_},
                     {"model", model_name()},
                     {"choices", {{{"index", 0}, {"message", std::move(message)}, {"finish_reason", reason}}}},
                     {"usage", usage(event)},
                     {"timings", time}});
        }
    }
    void on_event(const GenerationEvent &event) {
        if (closed_ || finished_)
            return;
        if (event.queue_wait_ms >= 0 || event.spec_steps)
            seen_.queue_wait_ms = event.queue_wait_ms, seen_.spec_steps = event.spec_steps,
            seen_.drafted = event.drafted, seen_.accepted = event.accepted;
        if (!event.error.empty()) {
            finished_ = true;
            log_request("error", event.reason.empty() ? "error" : event.reason, nullptr, event.error);
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
            last_token_at_ = Clock::now();
            if (!got_token_)
                got_token_ = true, first_token_at_ = last_token_at_;
            seen_.prompt_tokens = event.prompt_tokens, seen_.completion_tokens = event.completion_tokens,
            seen_.cached_tokens = event.cached_tokens;
            if (!diagnostic_dir_.empty()) diagnostic_["tokens"].push_back(event.token);
            const auto &format = chat_backend().tools;
            if (const std::string *marker = markers_ ? format.marker(event.token) : nullptr)
                emit(text_.append(*marker));
            else
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
                              {{{"id", model_name()}, {"object", "model"}, {"owned_by", "cgg507"}}}}});
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
            const auto ordered = nlohmann::ordered_json::parse(req.body());
            auto request = parse_chat(json::parse(req.body()), &ordered);
            auto prompt = tokenizer_.encode(request.rendered);
            if (const char *directory = std::getenv("BEHEMOTH_DIAGNOSTIC_DIR"); directory && *directory) {
                diagnostic_dir_ = directory;
                diagnostic_ = {{"tokens", prompt}, {"prompt_tokens", prompt.size()},
                    {"rendered_prompt", request.rendered}, {"request", json::parse(req.body())},
                    {"sampling", {{"seed", request.sampling.seed}, {"temperature", request.sampling.temperature},
                                  {"top_k", request.sampling.top_k}, {"top_p", request.sampling.top_p},
                                  {"min_p", request.sampling.min_p}, {"repetition_penalty", request.sampling.repetition_penalty},
                                  {"frequency_penalty", request.sampling.frequency_penalty},
                                  {"presence_penalty", request.sampling.presence_penalty},
                                  {"repetition_penalty_range", request.sampling.penalty_range},
                                  {"logit_bias_entries", request.sampling.logit_bias.size()}}},
                    {"producer", "behemoth_server exact delivered token IDs"}};
            }
            streaming_ = request.stream;
            include_usage_ = request.include_usage;
            text_ = TextStream(request.stops);
            std::shared_ptr<TokenConstraint> constraint;
            // the prompt ends inside an open think block (ChatBackend::think_open): the answer starts as reasoning
            const bool think_open = chat_backend().think_open >= 0 && request.rendered.ends_with("<think>\n");
            if (request.structured.kind != StructuredOutput::None) {
                std::unique_ptr<GrammarEngine> grammar;
                try {
                    grammar = structured_.grammars.acquire(request.structured.gbnf, request.structured.json);
                } catch (const std::exception &e) {
                    throw RequestError(std::string("Invalid grammar: ") + e.what(),
                                       request.structured.kind == StructuredOutput::Grammar ? "grammar"
                                                                                            : "response_format");
                }
                constraint = std::make_shared<StructuredConstraint>(structured_.vocabulary, std::move(grammar),
                                                                    request.structured.allow_think,
                                                                    &structured_.grammars);
                constrained_ = true;
                splitter_ = ThinkSplitter(request.structured.allow_think);
                if (!diagnostic_dir_.empty())
                    diagnostic_["structured"] = {{"kind", int(request.structured.kind)},
                                                 {"allow_think", request.structured.allow_think},
                                                 {"grammar", request.structured.gbnf}};
            }
            if (request.tools.active()) {
                // required / a named function: the whole answer is tool calls; auto: free text until the model
                // starts a call ([TOOL_CALLS]), then the call grammar
                const auto &format = chat_backend().tools;
                const bool lazy = request.tools.choice == ToolRequest::Auto;
                std::unique_ptr<GrammarEngine> grammar;
                try {
                    grammar = structured_.grammars.acquire(request.tool_grammar, true);
                } catch (const std::exception &e) {
                    throw RequestError(std::string("Invalid tool grammar: ") + e.what(), "tools");
                }
                auto calls = std::make_shared<StructuredConstraint>(structured_.vocabulary, std::move(grammar),
                                                                    !lazy && request.structured.allow_think,
                                                                    &structured_.grammars);
                if (!lazy) {
                    constraint = std::move(calls);
                    constrained_ = true;
                    splitter_ = ThinkSplitter(request.structured.allow_think);
                } else if (format.calls_token >= 0) {
                    auto triggered = std::make_shared<TriggeredConstraint>(format.calls_token, std::move(calls),
                                                                           structured_.vocabulary.size());
                    triggered->set_think(chat_backend().think_open, chat_backend().think_close, think_open);
                    constraint = std::move(triggered);
                }
                tool_parser_ = std::make_unique<ToolCallParser>(format, &request.tools); // auto: JSON-as-text fallback
                if (!diagnostic_dir_.empty())
                    diagnostic_["tools"] = {{"choice", int(request.tools.choice)}, {"grammar", request.tool_grammar}};
            }
            markers_ = (constrained_ || tool_parser_) && chat_backend().tools.supported();
            if (think_open && !constrained_) {
                reasoning_.append("<think>");
                answer_lead_ = true;
            } else if (think_open && request.structured.allow_think) { // the gate and its mirror here enter the block
                constraint->advance(chat_backend().think_open);
                splitter_.feed("<think>", nullptr, nullptr);
            }
            std::weak_ptr<Session> weak = shared_from_this();
            seen_.prompt_tokens = prompt.size();
            submitted_at_ = Clock::now();
            counted_ = true;
            active_start_ = active_requests.fetch_add(1) + 1;
            try {
                id_ = engine_.submit(prompt, request.max_new, request.sampling,
                                     [weak](const GenerationEvent &event) {
                                         if (auto self = weak.lock())
                                             asio::post(self->stream_.get_executor(),
                                                        [self, event] { self->on_event(event); });
                                     },
                                     std::move(constraint));
            } catch (...) {
                counted_ = false;
                active_requests.fetch_sub(1);
                throw;
            }
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
                heartbeat();
            }
        } catch (const CapacityError &e) {
            respond(http::status::service_unavailable,
                    {{"error", {{"message", e.what()}, {"type", "server_overloaded"}, {"param", nullptr},
                                {"code", nullptr}}}});
        } catch (const RequestError &e) {
            respond(http::status::bad_request,
                    {{"error", {{"message", e.what()}, {"type", "invalid_request_error"}, {"param", e.param},
                                {"code", nullptr}}}});
        } catch (const std::exception &e) {
            respond(http::status::bad_request,
                    {{"error", {{"message", e.what()}, {"type", "invalid_request_error"}, {"param", nullptr},
                                {"code", nullptr}}}});
        }
    }

  public:
    Session(tcp::socket socket, Engine &engine, const Tokenizer &tokenizer, StructuredContext &structured)
        : stream_(std::move(socket)), engine_(engine), tokenizer_(tokenizer), structured_(structured),
          heartbeat_(stream_.get_executor()) {
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
    StructuredContext &structured_;

  public:
    Listener(asio::io_context &io, tcp::endpoint endpoint, Engine &engine,
             const Tokenizer &tokenizer, StructuredContext &structured)
        : io_(io), acceptor_(io), engine_(engine), tokenizer_(tokenizer), structured_(structured) {
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
                    std::make_shared<Session>(std::move(socket), self->engine_, self->tokenizer_,
                                              self->structured_)
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
            else if (key == "--spec-mtp") // Neuron Qwen3.5: MTP drafts per step (0: off); the engine reads the environment
                setenv("BEHEMOTH_NEURON_SPEC_MTP", value.c_str(), 1);
            else if (key == "--max-batch-requests")
                options.scheduler.max_requests = std::stoul(value);
            else if (key == "--max-queued-requests")
                options.scheduler.max_queued = std::stoul(value);
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
        StructuredContext structured(tokenizer);
        Engine engine(options);
        asio::io_context io(threads);
        auto listener = std::make_shared<Listener>(
            io, tcp::endpoint(asio::ip::make_address(host), port), engine, tokenizer, structured);
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
