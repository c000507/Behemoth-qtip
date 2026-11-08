// Host-side test of neuron_engine.cpp's structured outputs and tool calls on the CPU fake model (fake_model.cpp),
// a 131072-id Tekken-like vocabulary and the runtime's constraints (no NeuronCores): constrained answers are valid,
// identical with and without prompt-lookup speculation (greedy and sampled: draft rows advance the constraint only
// while the drafts match), and unchanged by preemption and prefix-cache restores.
//   build and run: neuron/cpp/host_test.sh
#include "behemoth/constraint.hpp"
#include "behemoth/engine.hpp"
#include "behemoth/json_schema.hpp"
#include "behemoth/tools.hpp"
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>

using namespace behemoth;
using ojson = nlohmann::ordered_json;

static int checks = 0;
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        ++checks;                                                                                                      \
        if (!(x))                                                                                                      \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #x);               \
    } while (false)

static const ToolFormat format{"[TOOL_CALLS]", "[ARGS]", 9, 32};
// ids 0..999 control (EOS 2; 9 / 32 the tool markers), bytes at 1000.., multi-byte pieces, then filler words
static const Vocabulary &vocab() {
    static const Vocabulary v = [] {
        std::vector<Vocabulary::Token> t(131072);
        for (size_t i = 0; i < 1000; ++i)
            t[i].special = true;
        auto set = [&](size_t i, const std::string &p) {
            t[i].special = false;
            t[i].first = t[i].rest = p;
        };
        set(9, format.calls);
        set(32, format.args);
        for (int b = 0; b < 256; ++b)
            set(1000 + size_t(b), std::string(1, char(b)));
        size_t next = 1256;
        for (const char *p : {"\"ab\"", "\"cd\"", ",", "[\"", "\"]", "get_weather", "{\"city\":", " \"Paris\"}", "true", "ping"})
            set(next++, p);
        for (size_t i = 0; next < t.size(); ++i, ++next)
            set(next, " w" + std::to_string(i));
        return Vocabulary(std::move(t), 2);
    }();
    return v;
}
static std::vector<int> bytes(const std::string &s) {
    std::vector<int> out;
    for (char c : s)
        out.push_back(1000 + int(uint8_t(c)));
    return out;
}
static std::string text(const std::vector<int> &tokens) {
    std::string out;
    for (int t : tokens)
        if (!vocab().token(t).special)
            out += vocab().token(t).rest;
    return out;
}

struct Job {
    std::vector<int> prompt;
    size_t max_new;
    SamplingOptions sampling;
    std::string grammar; // empty: unconstrained
    bool lazy = false;   // TriggeredConstraint on [TOOL_CALLS]
};
struct Result {
    std::vector<int> tokens;
    std::string reason;
    size_t cached = 0;
};

static std::shared_ptr<TokenConstraint> constraint(const Job &j, GrammarCache &cache) {
    if (j.grammar.empty())
        return nullptr;
    auto c = std::make_shared<StructuredConstraint>(vocab(), cache.acquire(j.grammar, true), false, &cache);
    if (!j.lazy)
        return c;
    return std::make_shared<TriggeredConstraint>(format.calls_token, c, vocab().size());
}

// Runs the jobs on a fresh engine (all at once, or one after the other), with `env` set for its construction.
static std::vector<Result> run(const std::vector<Job> &jobs, const std::map<std::string, std::string> &env, bool sequential,
                               nlohmann::json *metrics = nullptr, size_t prefix_slots = 0) {
    for (const auto &[k, v] : env)
        setenv(k.c_str(), v.c_str(), 1);
    RuntimeOptions o;
    o.model = "/nonexistent";
    o.token_budget = 256;
    o.prefill_chunk = 64;
    o.prefix_cache.slots = prefix_slots;
    o.prefix_cache.min_tokens = 16;
    o.prefix_cache_gb = 0.01;
    GrammarCache cache;
    std::vector<Result> out(jobs.size());
    {
        Engine engine(o);
        for (const auto &[k, v] : env)
            unsetenv(k.c_str());
        std::mutex mutex;
        std::condition_variable cv;
        size_t open = 0;
        auto submit = [&](size_t i) {
            std::lock_guard lock(mutex);
            ++open;
            engine.submit(jobs[i].prompt, jobs[i].max_new, jobs[i].sampling,
                          [&, i](const GenerationEvent &e) {
                              std::lock_guard l(mutex);
                              if (!e.error.empty())
                                  out[i].reason = "error: " + e.error;
                              if (e.token >= 0)
                                  out[i].tokens.push_back(e.token);
                              if (e.done) {
                                  if (out[i].reason.empty())
                                      out[i].reason = e.reason;
                                  out[i].cached = e.cached_tokens;
                                  --open;
                                  cv.notify_all();
                              }
                          },
                          constraint(jobs[i], cache));
        };
        auto wait = [&] {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return open == 0; });
        };
        for (size_t i = 0; i < jobs.size(); ++i) {
            submit(i);
            if (sequential)
                wait();
        }
        wait();
        if (metrics)
            *metrics = engine.metrics();
    }
    for (const auto &r : out)
        if (r.reason.rfind("error", 0) == 0)
            throw std::runtime_error(r.reason);
    return out;
}

int main() {
    try {
        setenv("BEHEMOTH_NEURON_NEFF_DIR", "/nonexistent", 1);
        setenv("BEHEMOTH_NEURON_PREFIX_HOST_GB", "0.01", 1);
        const auto array_schema = ojson::parse(R"({"type":"array","items":{"enum":["ab","cd"]},"minItems":10,"maxItems":10})");
        const std::string array_grammar = grammar::json_schema_grammar(array_schema);
        const auto tools = parse_tools(ojson::parse(R"({"tools":[{"type":"function","function":{"name":"get_weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string","maxLength":12}},"required":["city"],
            "additionalProperties":false}}},{"type":"function","function":{"name":"ping"}}],"tool_choice":"required",
            "parallel_tool_calls":false})")); // one call: the fake model would repeat calls up to the token limit
        const std::string tool_grammar = tool_call_grammar(tools, format);
        // prompts with the answer's patterns in them, so the induction head (and prompt lookup) copies them
        auto prompt = [](const std::string &s, int pad) {
            auto p = bytes(s);
            for (int i = 0; i < pad; ++i)
                p.insert(p.begin(), 1256 + 10 + i); // filler ids: distinct prompts, longer sequences
            return p;
        };
        SamplingOptions greedy, hot;
        hot.temperature = 0.8f;
        hot.top_p = 0.95f;
        std::vector<Job> jobs;
        for (int i = 0; i < 3; ++i) {
            jobs.push_back({prompt("x\n[\"ab\",\"cd\",\"ab\",\"ab\"]\nx\n", 20 * i), 240, greedy, array_grammar});
            hot.seed = uint64_t(i);
            jobs.push_back({prompt("x\n[\"ab\",\"ab\",\"cd\"]\nx\n", 7 + 20 * i), 240, hot, array_grammar});
            jobs.push_back({prompt("q\n[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}\nq\n", 3 + 20 * i), 80, greedy, tool_grammar});
            jobs.push_back({prompt("Ok:Yes [TOOL_CALLS]ping[ARGS]{}\nOk:", 11 + 20 * i), 60, greedy, tool_grammar, true});
            jobs.push_back({prompt("free text free text free text ", 5 + 20 * i), 40, greedy, ""});
        }
        // the [TOOL_CALLS] marker of the prompts above as its control token
        for (auto &j : jobs)
            for (size_t k = 0; k + 12 <= j.prompt.size(); ++k)
                if (text(std::vector<int>(j.prompt.begin() + std::ptrdiff_t(k), j.prompt.begin() + std::ptrdiff_t(k + 12))) == "[TOOL_CALLS]") {
                    j.prompt.erase(j.prompt.begin() + std::ptrdiff_t(k) + 1, j.prompt.begin() + std::ptrdiff_t(k) + 12);
                    j.prompt[k] = 9;
                }

        nlohmann::json m_ref, m_spec, m_pre, m_prefix;
        const auto ref = run(jobs, {{"BEHEMOTH_NEURON_SPEC_TOKENS", "0"}}, true, &m_ref);
        // 1. validity
        for (size_t i = 0; i < jobs.size(); ++i) {
            const auto &r = ref[i];
            const std::string answer = text(r.tokens);
            std::cout << "  " << i << " (" << r.reason << "): " << answer << "\n";
            if (jobs[i].grammar.empty())
                continue;
            CHECK(r.reason == "stop" && r.tokens.back() == 2);
            if (jobs[i].grammar == array_grammar) {
                const auto v = nlohmann::json::parse(answer);
                CHECK(v.is_array() && v.size() == 10);
                for (const auto &x : v)
                    CHECK(x == "ab" || x == "cd");
            } else {
                ToolCallParser p(format);
                p.feed(answer, true);
                CHECK(!p.calls().empty());
                CHECK(jobs[i].lazy ? p.content() == "Yes " : p.content().empty());
                for (const auto &c : p.calls()) {
                    const auto args = nlohmann::json::parse(c.arguments);
                    CHECK(c.name == "ping" || (c.name == "get_weather" && args.at("city").is_string() && args.size() == 1));
                }
            }
        }
        CHECK(m_ref["structured_output"]["requests"] == 12 && m_ref["structured_output"]["tokens"] > 0);
        // 2. speculation: the same tokens, greedy and sampled, with drafts verified under the constraints
        const auto spec = run(jobs, {{"BEHEMOTH_NEURON_SPEC_TOKENS", "6"}, {"BEHEMOTH_NEURON_SPEC_ROWS", "8"}}, true, &m_spec);
        for (size_t i = 0; i < jobs.size(); ++i)
            CHECK(spec[i].tokens == ref[i].tokens);
        std::cout << "  speculation: drafted " << m_spec["drafted_tokens"] << ", accepted " << m_spec["accepted_drafts"] << "\n";
        CHECK(m_spec["accepted_drafts"].get<uint64_t>() > 20);
        // 3. preemption: all at once in a small K/V pool (requests re-prefill their history and keep their constraint)
        const auto pre = run(jobs, {{"BEHEMOTH_NEURON_SPEC_TOKENS", "6"}, {"BEHEMOTH_FAKE_PAGES", "24"}}, false, &m_pre);
        for (size_t i = 0; i < jobs.size(); ++i)
            CHECK(pre[i].tokens == ref[i].tokens);
        std::cout << "  preemption: " << m_pre["preemptions"] << " preemptions\n";
        CHECK(m_pre["preemptions"].get<uint64_t>() > 0);
        // 4. prefix cache: every job twice, the second restoring the first's prompt (and answer) K/V
        std::vector<Job> twice;
        for (const auto &j : jobs) {
            twice.push_back(j);
            twice.push_back(j);
        }
        const auto prefix = run(twice, {{"BEHEMOTH_NEURON_SPEC_TOKENS", "6"}}, true, &m_prefix, 8);
        size_t restored = 0;
        for (size_t i = 0; i < twice.size(); ++i) {
            CHECK(prefix[i].tokens == ref[i / 2].tokens);
            restored += prefix[i].cached;
        }
        std::cout << "  prefix cache: " << restored << " prompt tokens restored, " << m_prefix["prefix_cache"]["hits"] << " hits\n";
        CHECK(restored > 0);
        std::cout << "  constraint cost: " << m_ref["structured_output"]["constraint_us_per_token"] << " us/token over "
                  << m_ref["structured_output"]["tokens"] << " tokens, " << m_ref["structured_output"]["mask_fallbacks"]
                  << " masked samples\n";
        std::cout << "PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << "\n";
        return 1;
    }
}
