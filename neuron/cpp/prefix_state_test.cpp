// Host test of the prefix cache for recurrent models (state snapshots; docs/neuron-qwen35-prefix-cache.md) through
// neuron_engine.cpp on the CPU fake model in recurrent mode (fake_model.cpp, BEHEMOTH_FAKE_RECURRENT=1): every sequence
// has a recurrent state that must be exactly the state after the tokens before the one being run, else the fake model
// throws and the engine fails. A cached run must give the tokens of a cold run (the fake logits are a function of the
// whole sequence), restore only at snapshot positions, and survive partial matches, eviction, shared prefixes,
// preemption and cancellation. Built and run by host_test.sh.
#include "behemoth/engine.hpp"
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <thread>

using namespace behemoth;

static int checks = 0;
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        ++checks;                                                                                                      \
        if (!(x))                                                                                                      \
            throw std::runtime_error(std::string("check failed: ") + #x + " (line " + std::to_string(__LINE__) + ")"); \
    } while (0)

struct Job {
    std::vector<int> prompt;
    size_t max_new = 24;
    int cancel_after_ms = -1; // >= 0: cancelled that long after the submit
};
struct Result {
    std::vector<int> tokens;
    std::string reason;
    size_t cached = 0;
};
using Env = std::map<std::string, std::string>;

// One engine for all the `rounds`: the jobs of a round are submitted together (concurrent) and awaited.
static std::vector<std::vector<Result>> run(const std::vector<std::vector<Job>> &rounds, Env env, nlohmann::json *metrics = nullptr,
                                            size_t slots = 8) {
    env.try_emplace("BEHEMOTH_FAKE_RECURRENT", "1");
    env.try_emplace("BEHEMOTH_FAKE_CONTEXT", "8192");
    env.try_emplace("BEHEMOTH_FAKE_PAGES", "1024");
    for (const auto &[k, v] : env)
        setenv(k.c_str(), v.c_str(), 1);
    RuntimeOptions o;
    o.model = "/nonexistent";
    o.token_budget = 256;
    o.prefill_chunk = 64;
    o.prefix_cache.slots = slots;
    o.prefix_cache.min_tokens = 16;
    o.prefix_cache_gb = 0.01;
    std::vector<std::vector<Result>> out;
    {
        Engine engine(o);
        for (const auto &[k, v] : env)
            unsetenv(k.c_str());
        std::mutex mutex;
        std::condition_variable cv;
        for (const auto &jobs : rounds) {
            std::vector<Result> res(jobs.size());
            size_t open = 0;
            std::vector<std::thread> cancels;
            for (size_t i = 0; i < jobs.size(); ++i) {
                uint64_t id;
                {
                    std::lock_guard lock(mutex);
                    ++open;
                }
                id = engine.submit(jobs[i].prompt, jobs[i].max_new, SamplingOptions{}, [&, i](const GenerationEvent &e) {
                    std::lock_guard l(mutex);
                    if (!e.error.empty())
                        res[i].reason = "error: " + e.error;
                    if (e.token >= 0)
                        res[i].tokens.push_back(e.token);
                    if (e.done) {
                        if (res[i].reason.empty())
                            res[i].reason = e.reason;
                        res[i].cached = e.cached_tokens;
                        --open;
                        cv.notify_all();
                    }
                });
                if (jobs[i].cancel_after_ms >= 0)
                    cancels.emplace_back([&engine, id, ms = jobs[i].cancel_after_ms] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                        engine.cancel(id);
                    });
            }
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] { return open == 0; });
            }
            for (auto &t : cancels)
                t.join();
            out.push_back(std::move(res));
        }
        if (metrics)
            *metrics = engine.metrics();
    }
    for (const auto &round : out)
        for (const auto &r : round)
            if (r.reason.rfind("error", 0) == 0)
                throw std::runtime_error(r.reason);
    return out;
}

int main() {
    try {
        setenv("BEHEMOTH_NEURON_NEFF_DIR", "/nonexistent", 1);
        setenv("BEHEMOTH_NEURON_PREFIX_HOST_GB", "0.01", 1);
        std::mt19937 gen(7);
        auto tokens = [&](size_t n) {
            std::vector<int> t(n);
            for (auto &x : t)
                x = 300 + int(gen() % 60000);
            return t;
        };
        auto cat = [](std::vector<int> a, const std::vector<int> &b) {
            a.insert(a.end(), b.begin(), b.end());
            return a;
        };
        const Env on{{"BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE", "1"}, {"BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL", "128"}};
        const auto X = tokens(700), A = tokens(150), B = tokens(90), C = tokens(333), S = tokens(40);
        const std::vector<int> XA = cat(X, A), XB = cat(X, B), XC = cat(X, C);
        nlohmann::json m;

        // cold references: the recurrent fake model without any cache (one engine per prompt set, sequential)
        const auto cold = run({{{XA}}, {{XB}}, {{XC}}, {{S}}, {{X}}}, {});
        CHECK(cold[0][0].cached == 0 && cold[0][0].tokens.size() == 24);

        // 0. opt-in: a recurrent model without the flag, and one without snapshot support, never restore
        auto off = run({{{XA}}, {{XA}}}, {}, &m);
        CHECK(off[1][0].cached == 0 && off[1][0].tokens == cold[0][0].tokens && m["prefix_cache"].is_null());
        off = run({{{XA}}, {{XA}}}, {{"BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE", "1"}, {"BEHEMOTH_FAKE_STATE_SNAPSHOTS", "0"}}, &m);
        CHECK(off[1][0].cached == 0 && m["prefix_cache"].is_null());
        // prompt-lookup speculation stays off for a recurrent model, cache or not (a repeated prompt would draft)
        Env spec = on;
        spec["BEHEMOTH_NEURON_SPEC_TOKENS"] = "6";
        run({{{cat(S, S), 40}}}, spec, &m);
        CHECK(m["drafted_tokens"].get<uint64_t>() == 0);

        // 1. the same prompt again: restored at the snapshot 8 tokens before the prompt's end; a short prompt likewise
        auto r = run({{{XA}}, {{XA}}, {{S}}, {{S}}}, on, &m);
        CHECK(r[0][0].cached == 0 && r[0][0].tokens == cold[0][0].tokens);
        CHECK(r[1][0].cached == XA.size() - 8 && r[1][0].tokens == cold[0][0].tokens);
        CHECK(r[3][0].cached == S.size() - 8 && r[3][0].tokens == cold[3][0].tokens);
        CHECK(m["prefix_cache"]["state_snapshots"]["restores"].get<uint64_t>() == 2);
        std::cout << "  same prompt: " << r[1][0].cached << " of " << XA.size() << " tokens restored; snapshots " << m["prefix_cache"]["state_snapshots"].dump() << "\n";

        // 2. partial match: only an interval snapshot at or below the common prefix can be restored (700 -> 640)
        r = run({{{XA}}, {{XB}}, {{XC}}, {{X}}}, on, &m);
        CHECK(r[1][0].cached == 640 && r[1][0].tokens == cold[1][0].tokens);
        CHECK(r[2][0].cached == 640 && r[2][0].tokens == cold[2][0].tokens);
        CHECK(r[3][0].cached == 640 && r[3][0].tokens == cold[4][0].tokens); // a prompt that IS the prefix: capped at size - 1
        // without interval snapshots a partial match below the tail snapshot is a miss
        Env sparse = on;
        sparse["BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL"] = "100000";
        r = run({{{XA}}, {{XB}}, {{XA}}}, sparse);
        CHECK(r[1][0].cached == 0 && r[1][0].tokens == cold[1][0].tokens && r[2][0].cached == XA.size() - 8);

        // 3. next turn of a conversation: prompt + the answer token-exact + a new message restores at the sequence's
        // end (prompt + generated - 1: the last sampled token was never run); with an edited answer at the tail snapshot
        std::vector<int> answer(cold[0][0].tokens.begin(), cold[0][0].tokens.end());
        const auto turn2 = cat(cat(XA, answer), B);
        auto edited = answer;
        edited[2] ^= 1;
        const auto turn2e = cat(cat(XA, edited), B);
        const auto cold2 = run({{{turn2}}, {{turn2e}}}, {});
        r = run({{{XA}}, {{turn2}}}, on);
        CHECK(r[1][0].cached == XA.size() + answer.size() - 1 && r[1][0].tokens == cold2[0][0].tokens);
        r = run({{{XA}}, {{turn2e}}, {{turn2}}}, on);
        CHECK(r[1][0].cached == XA.size() - 8 && r[1][0].tokens == cold2[1][0].tokens);
        CHECK(r[2][0].cached >= XA.size() - 8 && r[2][0].tokens == cold2[0][0].tokens);
        Env noend = on;
        noend["BEHEMOTH_NEURON_QWEN35_PREFIX_END"] = "0";
        noend["BEHEMOTH_NEURON_QWEN35_PREFIX_TAIL"] = "0"; // snapshot at the prompt's end instead of before it
        r = run({{{XA}}, {{turn2}}, {{XA}}}, noend);
        CHECK(r[1][0].cached == XA.size() && r[1][0].tokens == cold2[0][0].tokens);
        CHECK(r[2][0].cached == 768 && r[2][0].tokens == cold[0][0].tokens);

        // 4. concurrent requests sharing a cached prefix (one takes the device tier, the others host copies)
        r = run({{{XA}}, {{XB}, {XC}, {XA}, {XB}}}, on, &m);
        CHECK(r[1][0].tokens == cold[1][0].tokens && r[1][1].tokens == cold[2][0].tokens && r[1][2].tokens == cold[0][0].tokens && r[1][3].tokens == cold[1][0].tokens);
        for (const auto &x : r[1])
            CHECK(x.cached >= 640);
        std::cout << "  shared prefix: device hits " << m["prefix_cache"]["device_hits"] << ", host hits " << m["prefix_cache"]["host_hits"] << "\n";

        // 5. eviction under pressure: 2 cache slots, room for 3 state snapshots, 6 different conversations revisited
        std::vector<std::vector<int>> P;
        std::vector<std::vector<Job>> rounds;
        for (int i = 0; i < 6; ++i)
            P.push_back(cat(tokens(200 + 37 * size_t(i)), i % 2 ? A : B));
        for (int pass = 0; pass < 3; ++pass)
            for (int i = 0; i < 6; ++i)
                rounds.push_back({{P[size_t(pass == 1 ? 5 - i : i)]}});
        const auto coldp = run(rounds, {});
        Env tight = on;
        tight["BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB"] = "0.00000005"; // 50 bytes: three 16-byte fake states
        r = run(rounds, tight, &m, 2);
        size_t hits = 0;
        for (size_t i = 0; i < rounds.size(); ++i) {
            CHECK(r[i][0].tokens == coldp[i][0].tokens);
            hits += r[i][0].cached > 0;
        }
        CHECK(m["prefix_cache"]["state_snapshots"]["dropped"].get<uint64_t>() > 0 && m["prefix_cache"]["evictions"].get<uint64_t>() > 0);
        CHECK(m["prefix_cache"]["state_snapshots"]["bytes"].get<uint64_t>() <= 50);
        std::cout << "  eviction: " << hits << " of " << rounds.size() << " requests hit, " << m["prefix_cache"]["evictions"] << " slot evictions, "
                  << m["prefix_cache"]["state_snapshots"]["dropped"] << " snapshots dropped\n";

        // 6. preemption: a K/V pool too small for the concurrent requests (misses: preempted ones re-prefill from position
        // 0, their snapshots stay valid; hits wait for pages), then the same prompts again from the cache
        const auto D1 = tokens(800), D2 = tokens(820), D3 = tokens(790);
        const auto coldd = run({{{D1, 200}}, {{D2, 200}}, {{D3, 200}}}, {});
        Env small = on;
        small["BEHEMOTH_FAKE_PAGES"] = "150";
        r = run({{{XA}}, {{D1, 200}, {D2, 200}, {D3, 200}, {XB}}, {{D1, 200}, {D3, 200}, {XC}}}, small, &m);
        CHECK(r[1][0].tokens == coldd[0][0].tokens && r[1][1].tokens == coldd[1][0].tokens && r[1][2].tokens == coldd[2][0].tokens && r[1][3].tokens == cold[1][0].tokens);
        CHECK(r[2][0].tokens == coldd[0][0].tokens && r[2][1].tokens == coldd[2][0].tokens && r[2][2].tokens == cold[2][0].tokens);
        std::cout << "  preemption: " << m["preemptions"] << " preemptions, then " << r[2][0].cached << " / " << r[2][1].cached << " / " << r[2][2].cached << " tokens restored\n";
        CHECK(m["preemptions"].get<uint64_t>() > 0);

        // 7. cancel in mid-prefill / mid-decode (slow fake device), then the same prompt: whatever was cached is exact
        Env slow = on;
        slow["BEHEMOTH_FAKE_STEP_MS"] = "3";
        for (int ms : {0, 8, 20, 35, 60, 90}) {
            r = run({{{XA, 24, ms}}, {{XA}}, {{XB}}}, slow, &m);
            CHECK(r[1][0].tokens == cold[0][0].tokens && r[2][0].tokens == cold[1][0].tokens);
            std::cout << "  cancel after " << ms << " ms: " << r[0][0].reason << " with " << r[0][0].tokens.size() << " tokens; then " << r[1][0].cached << " and "
                      << r[2][0].cached << " tokens restored\n";
        }
        std::cout << "PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << "\n";
        return 1;
    }
}
