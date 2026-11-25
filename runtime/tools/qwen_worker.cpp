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
//          [--kv bf16|k8v16|fp8 (KV cache element type; sets QWEN_KV, default fp8)]
//          [--mtp (MTP speculative decoding, draft k = 1)] [--mtp-seqs N (max sequences per verify pass,
//           default 32)] [--mtp-max N (use MTP only while <= N sequences decode; 0 = adaptive, default)]
// MTP: per step either a verify pass ([last, draft] per sequence; greedy: exact-match acceptance,
// sampling: the speculative acceptance rule min(1, p/q) with residual resampling, so the output
// distribution is the plain sampler's) + an MTP draft pass, or a plain decode pass. Adaptive mode
// picks MTP while (1 + acceptance) beats the measured verify/plain step-time ratio of the batch size.
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

// The sampler's distribution over the sorted candidates (the same keep set / weights as sample()).
static std::vector<std::pair<int, double>> sampler_dist(const QwenCand *c, int K, float temperature, float top_p, int top_k) {
    std::vector<std::pair<int, double>> d;
    if (temperature <= 0) {
        d.push_back({c[0].id, 1.0});
        return d;
    }
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
    for (int i = 0; i < keep; ++i) d.push_back({c[i].id, p[size_t(i)] / tot});
    return d;
}
static double prob_of(const std::vector<std::pair<int, double>> &d, int id) {
    for (auto &e : d)
        if (e.first == id) return e.second;
    return 0.0;
}
static int draw(const std::vector<std::pair<int, double>> &d, std::mt19937_64 &rng) {
    double tot = 0;
    for (auto &e : d) tot += e.second;
    double u = std::uniform_real_distribution<double>(0, tot)(rng);
    for (auto &e : d) {
        u -= e.second;
        if (u <= 0) return e.first;
    }
    return d.back().first;
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
    int draft = -1;                           // MTP draft for the token after `last` (-1: none)
    std::vector<std::pair<int, double>> qd;   // its draft distribution
    long nver = 0, nacc = 0;                  // verified / accepted drafts
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
    bool use_mtp = false;
    int mtp_seqs = 32, mtp_max = 0;
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
        else if (!std::strcmp(argv[i], "--mtp")) use_mtp = true;
        else if (!std::strcmp(argv[i], "--mtp-seqs")) mtp_seqs = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--mtp-max")) mtp_max = std::atoi(next().c_str());
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
    setenv("QWEN_KV", "fp8", 0);  // serving default: e4m3 K+V (2x KV pool, ppl within noise of bf16)
    if (prefill_tokens < 0) prefill_tokens = chunk;
    try {
        double t0 = now_s();
        behemoth::QwenServeOptions so;
        so.max_seqs = max_seqs;
        so.kv_tokens = kv_tokens;
        so.mtp = use_mtp;
        so.mtp_seqs = mtp_seqs;
        behemoth::QwenModel m(model, devs, chunk, max_context, tp, so);
        const bool spec_on = use_mtp && m.has_mtp();
        if (use_mtp && !spec_on) std::fprintf(stderr, "qwen_worker: --mtp but the checkpoint has no MTP head\n");
        const int K = m.topk(), PT = m.page_tokens();
        const long pages = m.kv_pages_total();
        std::fprintf(stderr,
                     "qwen_worker: loaded in %.1f s on %zu gpu(s); %d slots, KV pool %ld tokens (%s), max_context %d, "
                     "chunk %d, MTP %s\n",
                     now_s() - t0, devs.size(), max_seqs, pages * PT, std::getenv("QWEN_KV") ? std::getenv("QWEN_KV") : "bf16",
                     max_context, chunk, spec_on ? "on" : "off");
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
            json j{{"id", r.id}, {"done", reason}, {"prompt", r.toks.size()}, {"completion", r.ngen},
                   {"ttft_s", r.t_first > 0 ? r.t_first - r.t_arrive : 0.0},
                   {"decode_s", r.t_first > 0 ? t - r.t_first : 0.0}};
            if (r.nver > 0) j["mtp"] = {r.nacc, r.nver};
            emit(j);
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
        // MTP bookkeeping: acceptance EMA, measured step times per batch size (plain / verify+draft)
        std::vector<QwenCand> vcand, dcand;
        double acc_ema = 0.7;
        long mtp_ver = 0, mtp_acc = 0, mtp_steps = 0;
        unsigned probe = 0;
        std::vector<double> t_plain(65, 0.0), t_spec(65, 0.0);
        auto note_step = [&](int n, bool sp, double dt) {
            if (n < 1 || n > 64) return;
            double &e = (sp ? t_spec : t_plain)[size_t(n)];
            e = e > 0 ? 0.9 * e + 0.1 * dt : dt;
        };
        // verify+draft / plain step-time ratio: measured (both modes seen at n) or the TP2 R9700 table
        auto step_ratio = [&](int n) -> double {
            if (t_plain[size_t(n)] > 0 && t_spec[size_t(n)] > 0) return t_spec[size_t(n)] / t_plain[size_t(n)];
            static const int bn[] = {1, 4, 8, 16, 24, 32};  // qwen_eval --bench-batch, ctx 1k / 8k mean
            static const double br[] = {1.10, 1.155, 1.21, 1.29, 1.49, 1.55};
            if (n <= 1) return br[0];
            for (int i = 1; i < 6; ++i)
                if (n <= bn[i]) return br[i - 1] + (br[i] - br[i - 1]) * double(n - bn[i - 1]) / double(bn[i] - bn[i - 1]);
            return 1e9;
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
                const bool fin = r.prefilled + n == r.toks.size();
                parts.push_back({slot, r.toks.data() + r.prefilled, n, fin, fin ? -1 : r.toks[r.prefilled + n]});
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
            const int nd = int(slots.size());
            bool spec = spec_on && nd > 0 && nd <= m.mtp_seqs() && (mtp_max <= 0 || nd <= mtp_max);
            for (int s : slots) spec = spec && m.seq_len(s) + 3 < max_context;
            if (spec && mtp_max <= 0) {  // adaptive: expected tokens per step vs the verify / plain time ratio
                const double ratio = step_ratio(nd);
                spec = (1.0 + acc_ema) > ratio * 1.03 || (++probe % 64) == 0;
            }
            if (spec) {
                const double ts0 = now_s();
                std::vector<int> vs, vt, vd;
                for (int s : slots) {
                    Req &r = *running[s];
                    if (r.draft >= 0) {
                        vs.push_back(s);
                        vt.push_back(r.last);
                        vd.push_back(r.draft);
                    }
                }
                std::vector<std::pair<int, std::string>> done_now;
                if (!vs.empty()) {
                    const int nv = int(vs.size());
                    vcand.resize(size_t(2 * nv) * K);
                    if (!m.decode_verify(vs.data(), vt.data(), vd.data(), nv, vcand.data())) {
                        spec = false;  // out of KV pages: plain step below retires a victim
                    } else {
                        std::vector<int> acc(static_cast<size_t>(nv));
                        std::vector<std::vector<int>> emitted(static_cast<size_t>(nv));
                        for (int i = 0; i < nv; ++i) {
                            Req &r = *running[vs[size_t(i)]];
                            const QwenCand *c0 = vcand.data() + size_t(2 * i) * K, *c1 = c0 + K;
                            auto p = sampler_dist(c0, K, r.temperature, r.top_p, r.top_k);
                            const double pd = prob_of(p, r.draft), qd = std::max(prob_of(r.qd, r.draft), 1e-300);
                            bool ok;
                            if (r.temperature <= 0) ok = p[0].first == r.draft;
                            else ok = std::uniform_real_distribution<double>(0, 1)(r.rng) < std::min(1.0, pd / qd);
                            acc[size_t(i)] = ok;
                            ++r.nver;
                            ++mtp_ver;
                            if (ok) {
                                ++r.nacc;
                                ++mtp_acc;
                                emitted[size_t(i)] = {r.draft, sample(c1, K, r.temperature, r.top_p, r.top_k, r.rng)};
                            } else {  // residual max(0, p - q)
                                std::vector<std::pair<int, double>> res;
                                if (r.temperature <= 0) {
                                    res = p;
                                } else {
                                    for (auto &e : p) {
                                        const double v = e.second - prob_of(r.qd, e.first);
                                        if (v > 0) res.push_back({e.first, v});
                                    }
                                    if (res.empty()) res = p;
                                }
                                emitted[size_t(i)] = {draw(res, r.rng)};
                            }
                            acc_ema = 0.98 * acc_ema + 0.02 * double(ok);
                        }
                        m.spec_accept(acc.data());
                        for (int i = 0; i < nv; ++i) {
                            Req &r = *running[vs[size_t(i)]];
                            r.draft = -1;
                            for (int t : emitted[size_t(i)]) {
                                const std::string why = accept(r, t);
                                if (!why.empty()) {
                                    done_now.push_back({vs[size_t(i)], why});
                                    break;
                                }
                            }
                        }
                    }
                }
                if (spec) {
                    for (auto &d : done_now) retire(d.first, d.second);
                    std::vector<int> ds, dl;  // next drafts of every sequence still decoding
                    for (int s : slots)
                        if (running.count(s)) {
                            ds.push_back(s);
                            dl.push_back(running[s]->last);
                        }
                    if (!ds.empty()) {
                        dcand.resize(ds.size() * size_t(K));
                        m.mtp_draft(ds.data(), dl.data(), int(ds.size()), dcand.data());
                        for (size_t i = 0; i < ds.size(); ++i) {
                            Req &r = *running[ds[i]];
                            r.qd = sampler_dist(dcand.data() + i * K, K, r.temperature, r.top_p, r.top_k);
                            r.draft = r.temperature <= 0 ? r.qd[0].first : draw(r.qd, r.rng);
                        }
                    }
                    if (!vs.empty()) note_step(int(vs.size()), true, now_s() - ts0);
                    if (++mtp_steps % 500 == 0)
                        std::fprintf(stderr, "qwen_worker: MTP %ld spec steps, acceptance %.3f (ema %.3f)\n", mtp_steps,
                                     double(mtp_acc) / double(std::max(1L, mtp_ver)), acc_ema);
                    flush_out();
                    continue;
                }
            }
            for (int s : slots)
                if (running.count(s)) running[s]->draft = -1;
            const double tp0 = now_s();
            if (!slots.empty()) {
                if (!m.decode_batch(slots.data(), toks.data(), int(slots.size()), cand.data())) {
                    // out of KV pages: finish the longest sequence and retry next iteration
                    int victim = slots[0];
                    for (int s : slots)
                        if (m.seq_len(s) > m.seq_len(victim)) victim = s;
                    retire(victim, "length");
                } else {
                    note_step(nd, false, now_s() - tp0);
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
