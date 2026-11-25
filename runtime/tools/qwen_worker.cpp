// Qwen3.5 generation worker for tools/qwen_openai_server.py: loads the model once (tensor parallel
// over 2 GPUs or one GPU), then serves many requests concurrently with continuous batching: new
// requests are admitted between decode steps, prompts are prefilled in chunks interleaved with the
// batched decode steps, finished sequences are retired immediately.
// stdin  (JSON lines): {"id":"r1","tokens":[...],"max_new":N,"temperature":T,"top_p":P,"top_k":K,"seed":S,
//                       "stop":[ids]}   |   {"cancel":"r1"}
// stdout (JSON lines): {"ready":true,...} once, then {"id":"r1","t":id} per token and
//                      {"id":"r1","done":"stop|length|cancelled|error","prompt":n,"completion":n,"ttft_s":s,
//                       "decode_s":s}
// usage  : qwen_worker --model DIR [--gpus 0,1] [--tp] [--chunk N] [--max-context N] [--max-seqs N]
//          [--kv-tokens N] [--prefill-tokens N (prompt tokens per decode step, default chunk)]
//          [--kv bf16|k8v16|fp8 (KV cache element type; sets QWEN_KV)]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "behemoth/qwen_model.hpp"

using json = nlohmann::json;
using behemoth::QwenCand;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Samples from the sorted top-K candidates (K = 64: top_k 0 or > 64 is truncated to 64).
static int sample(const QwenCand *c, int K, float temperature, float top_p, int top_k, std::mt19937_64 &rng) {
    if (temperature <= 0) return c[0].id;
    const int k = top_k > 0 ? std::min(top_k, K) : K;
    std::vector<double> p(static_cast<size_t>(k));
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += p[size_t(i)] = std::exp(double(c[i].logit - c[0].logit) / temperature);
    int keep = k;
    if (top_p > 0 && top_p < 1) {
        double acc = 0;
        for (int i = 0; i < k; ++i) {
            acc += p[size_t(i)] / sum;
            if (acc >= top_p) {
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
        if (u <= 0) return c[i].id;
    }
    return c[keep - 1].id;
}

struct Req {
    std::string id;
    std::vector<int> toks, stop;
    int max_new = 256, top_k = 0;
    float temperature = 0, top_p = 1;
    std::mt19937_64 rng;
    int slot = -1;
    size_t prefilled = 0;
    int ngen = 0, last = -1;
    bool decoding = false;
    double t_arrive = 0, t_first = 0;
};

struct Inbox {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<json> q;
    bool eof = false;
};

static std::mutex out_mu;
static void emit(const json &j) {
    std::lock_guard<std::mutex> g(out_mu);
    std::cout << j.dump() << '\n';
}
static void flush_out() {
    std::lock_guard<std::mutex> g(out_mu);
    std::cout.flush();
}

int main(int argc, char **argv) {
    bool tp = false;
    std::string model, gpus = "0";
    int chunk = 2048, max_context = 32768, max_seqs = 32;
    long prefill_tokens = -1;
    long kv_tokens = 0;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> std::string { return argv[++i]; };
        if (!std::strcmp(argv[i], "--model")) model = next();
        else if (!std::strcmp(argv[i], "--gpus")) gpus = next();
        else if (!std::strcmp(argv[i], "--tp")) tp = true;
        else if (!std::strcmp(argv[i], "--chunk")) chunk = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--max-context")) max_context = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--max-seqs")) max_seqs = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--kv-tokens")) kv_tokens = std::atol(next().c_str());
        else if (!std::strcmp(argv[i], "--prefill-tokens")) prefill_tokens = std::atol(next().c_str());
        else if (!std::strcmp(argv[i], "--kv")) setenv("QWEN_KV", next().c_str(), 1);
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
    max_seqs = std::max(1, std::min(64, max_seqs));
    if (prefill_tokens < 0) prefill_tokens = chunk;
    try {
        double t0 = now_s();
        behemoth::QwenServeOptions so;
        so.max_seqs = max_seqs;
        so.kv_tokens = kv_tokens;
        behemoth::QwenModel m(model, devs, chunk, max_context, tp, so);
        const int K = m.topk(), PT = m.page_tokens();
        const long pages = m.kv_pages_total();
        std::fprintf(stderr,
                     "qwen_worker: loaded in %.1f s on %zu gpu(s); %d slots, KV pool %ld tokens (%s), max_context %d, "
                     "chunk %d\n",
                     now_s() - t0, devs.size(), max_seqs, pages * PT, std::getenv("QWEN_KV") ? std::getenv("QWEN_KV") : "bf16",
                     max_context, chunk);
        emit(json{{"ready", true}, {"max_context", max_context}, {"max_seqs", max_seqs}, {"kv_tokens", pages * PT}});
        flush_out();

        Inbox inbox;
        std::thread reader([&]() {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                json j;
                try {
                    j = json::parse(line);
                } catch (const std::exception &e) {
                    emit(json{{"id", ""}, {"done", "error"}, {"error", e.what()}});
                    flush_out();
                    continue;
                }
                std::lock_guard<std::mutex> g(inbox.mu);
                inbox.q.push_back(std::move(j));
                inbox.cv.notify_one();
            }
            std::lock_guard<std::mutex> g(inbox.mu);
            inbox.eof = true;
            inbox.cv.notify_one();
        });
        reader.detach();

        std::deque<std::unique_ptr<Req>> waiting;
        std::map<int, std::unique_ptr<Req>> running;  // slot -> request
        std::vector<int> free_slots;
        for (int s = max_seqs - 1; s >= 0; --s) free_slots.push_back(s);
        std::vector<QwenCand> cand(size_t(max_seqs) * K);

        auto finish = [&](Req &r, const std::string &reason) {
            const double t = now_s();
            emit(json{{"id", r.id}, {"done", reason}, {"prompt", r.toks.size()}, {"completion", r.ngen},
                      {"ttft_s", r.t_first > 0 ? r.t_first - r.t_arrive : 0.0},
                      {"decode_s", r.t_first > 0 ? t - r.t_first : 0.0}});
        };
        auto retire = [&](int slot, const std::string &reason) {
            auto it = running.find(slot);
            finish(*it->second, reason);
            m.seq_free(slot);
            free_slots.push_back(slot);
            running.erase(it);
        };
        // accepts token t for r; returns the finish reason if r is done
        auto accept = [&](Req &r, int t) -> std::string {
            if (r.ngen == 0) r.t_first = now_s();
            ++r.ngen;
            r.last = t;
            emit(json{{"id", r.id}, {"t", t}});
            if (std::find(r.stop.begin(), r.stop.end(), t) != r.stop.end()) return "stop";
            if (r.ngen >= r.max_new) return "length";
            if (m.seq_len(r.slot) + 1 >= max_context) return "length";
            return "";
        };
        bool eof = false;
        while (true) {
            {  // inbox
                std::unique_lock<std::mutex> lk(inbox.mu);
                if (inbox.q.empty() && waiting.empty() && running.empty()) {
                    if (inbox.eof) break;
                    inbox.cv.wait(lk, [&] { return !inbox.q.empty() || inbox.eof; });
                }
                eof = inbox.eof;
                while (!inbox.q.empty()) {
                    json j = std::move(inbox.q.front());
                    inbox.q.pop_front();
                    if (j.contains("cancel")) {
                        const std::string id = j["cancel"].get<std::string>();
                        for (auto it = waiting.begin(); it != waiting.end(); ++it)
                            if ((*it)->id == id) {
                                finish(**it, "cancelled");
                                waiting.erase(it);
                                break;
                            }
                        for (auto &kv : running)
                            if (kv.second->id == id) {
                                retire(kv.first, "cancelled");
                                break;
                            }
                        continue;
                    }
                    auto r = std::make_unique<Req>();
                    r->id = j.value("id", std::string(""));
                    r->toks = j.value("tokens", std::vector<int>{});
                    r->max_new = j.value("max_new", 256);
                    r->temperature = j.value("temperature", 0.0f);
                    r->top_p = j.value("top_p", 1.0f);
                    r->top_k = j.value("top_k", 0);
                    r->stop = j.value("stop", std::vector<int>{});
                    r->rng.seed(j.value("seed", uint64_t(std::random_device{}())));
                    r->t_arrive = now_s();
                    if (r->toks.empty() || int(r->toks.size()) >= max_context || r->max_new < 1) {
                        emit(json{{"id", r->id}, {"done", "error"},
                                  {"error", "prompt empty or longer than max_context (" + std::to_string(max_context) + ")"}});
                        continue;
                    }
                    r->max_new = std::min(r->max_new, max_context - int(r->toks.size()));
                    waiting.push_back(std::move(r));
                }
            }
            (void)eof;
            // admit: a free slot and KV pages for the prompt + some headroom
            while (!waiting.empty() && !free_slots.empty()) {
                Req &r = *waiting.front();
                const long need = long((r.toks.size() + std::min(r.max_new, 256) + PT - 1) / PT) + long(running.size());
                if (m.kv_pages_free() < need && !running.empty()) break;
                r.slot = free_slots.back();
                free_slots.pop_back();
                m.seq_start(r.slot);
                running[r.slot] = std::move(waiting.front());
                waiting.pop_front();
            }
            // prefill: oldest prefilling sequences first; while sequences are decoding, at most
            // prefill_tokens prompt tokens (whole chunks, >= 1 chunk) between two decode steps, so
            // short prompts are admitted several per step and a long one cannot stall the batch
            bool any_decoding = false;
            for (auto &kv : running) any_decoding |= kv.second->decoding;
            long budget = any_decoding ? prefill_tokens : long(chunk);
            std::vector<int> pre;
            for (auto &kv : running)
                if (!kv.second->decoding) pre.push_back(kv.first);
            // alternate shortest-remaining-first (short prompts are not stuck behind a long one) and
            // oldest-first (a long prompt still progresses under a stream of short ones)
            static unsigned iter = 0;
            if (++iter & 1u)
                std::sort(pre.begin(), pre.end(), [&](int a, int b) {
                    const size_t ra = running[a]->toks.size() - running[a]->prefilled;
                    const size_t rb = running[b]->toks.size() - running[b]->prefilled;
                    return ra != rb ? ra < rb : running[a]->t_arrive < running[b]->t_arrive;
                });
            else
                std::sort(pre.begin(), pre.end(), [&](int a, int b) { return running[a]->t_arrive < running[b]->t_arrive; });
            // pack the next piece of several prompts into one forward pass (<= chunk rows); each
            // prompt is still split at the same chunk boundaries as when prefilled alone
            std::vector<behemoth::QwenModel::PrefillPart> parts;
            std::vector<int> part_slot;
            long rows = 0;
            for (int slot : pre) {
                Req &r = *running[slot];
                const size_t n = std::min<size_t>(size_t(chunk), r.toks.size() - r.prefilled);
                if (!parts.empty() && (rows + long(n) > chunk || long(n) > budget)) break;
                parts.push_back({slot, r.toks.data() + r.prefilled, n, r.prefilled + n == r.toks.size()});
                part_slot.push_back(slot);
                rows += long(n);
                budget -= long(n);
            }
            if (!parts.empty()) {
                if (!m.prefill_multi(parts, cand.data())) {
                    retire(part_slot.back(), "error");  // no KV pages for the prompt
                } else {
                    int ci = 0;
                    for (size_t i = 0; i < parts.size(); ++i) {
                        Req &r = *running[part_slot[i]];
                        r.prefilled += parts[i].n;
                        if (!parts[i].want) continue;
                        r.decoding = true;
                        const int t = sample(cand.data() + size_t(ci++) * K, K, r.temperature, r.top_p, r.top_k, r.rng);
                        const std::string why = accept(r, t);
                        if (!why.empty()) retire(part_slot[i], why);
                    }
                    flush_out();
                }
            }
            // batched decode of every decoding sequence
            std::vector<int> slots, toks;
            for (auto &kv : running)
                if (kv.second->decoding) {
                    slots.push_back(kv.first);
                    toks.push_back(kv.second->last);
                }
            if (!slots.empty()) {
                if (!m.decode_batch(slots.data(), toks.data(), int(slots.size()), cand.data())) {
                    // out of KV pages: finish the longest sequence and retry next iteration
                    int victim = slots[0];
                    for (int s : slots)
                        if (m.seq_len(s) > m.seq_len(victim)) victim = s;
                    retire(victim, "length");
                } else {
                    for (size_t i = 0; i < slots.size(); ++i) {
                        Req &r = *running[slots[i]];
                        const int t = sample(cand.data() + i * K, K, r.temperature, r.top_p, r.top_k, r.rng);
                        const std::string why = accept(r, t);
                        if (!why.empty()) retire(slots[i], why);
                    }
                }
            }
            flush_out();
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
