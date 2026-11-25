// Qwen3.5 eval: greedy generation + perplexity from a token file (int32 raw).
// usage: qwen_eval --model DIR [--tokens FILE] [--prompt FILE] [--greedy N] [--ppl FILE]
//          [--chunk N] [--device N] [--max-context N] [--window W [--stride S]] [--kv bf16|k8v16|fp8]
// --window: score W-token windows (state reset per window) every S tokens (default S = W - 64), the
// same windows as qwen35_ref / qwen_qt_ref --ppl (512 / 448), so the numbers are directly comparable.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "behemoth/qwen_model.hpp"

static double behemoth_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<int> load_tokens(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    size_t sz = size_t(f.tellg()) / 4;
    std::vector<int> t(sz);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(t.data()), std::streamsize(sz * 4));
    return t;
}

int main(int argc, char **argv) {
    bool tp = false;
    std::string model, prompt_file, tokens_file, ppl_file;
    std::string gpus;
    std::vector<int> bench;
    int bench_decode = 0;
    int serve_seqs = 0, bench_ctx = 1024, bench_steps = 32, btest_n = 0, btest_gen = 32;
    std::vector<int> bench_batch;
    std::string btest_file;
    int decode_ppl = 0;  // --window scoring through decode steps after a decode_ppl-token prefill
    int greedy = 0, chunk = 256, device = 0, max_context = 8192, window = 0, stride = 0;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> std::string { return argv[++i]; };
        if (!std::strcmp(argv[i], "--model")) model = next();
        else if (!std::strcmp(argv[i], "--prompt")) prompt_file = next();
        else if (!std::strcmp(argv[i], "--tokens")) tokens_file = next();
        else if (!std::strcmp(argv[i], "--ppl")) ppl_file = next();
        else if (!std::strcmp(argv[i], "--greedy")) greedy = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--chunk")) chunk = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--device")) device = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--gpus")) gpus = next();
        else if (!std::strcmp(argv[i], "--tp")) tp = true;
        else if (!std::strcmp(argv[i], "--bench-prefill")) bench.push_back(std::atoi(next().c_str()));
        else if (!std::strcmp(argv[i], "--bench-decode")) bench_decode = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--max-context")) max_context = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--serve-seqs")) serve_seqs = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--bench-batch")) {
            std::string v = next();
            for (size_t p = 0; p < v.size();) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                bench_batch.push_back(std::atoi(v.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
        else if (!std::strcmp(argv[i], "--bench-ctx")) bench_ctx = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--bench-steps")) bench_steps = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--batch-test")) { btest_n = std::atoi(next().c_str()); btest_file = next(); }
        else if (!std::strcmp(argv[i], "--batch-test-gen")) btest_gen = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--window")) window = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--stride")) stride = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--kv")) setenv("QWEN_KV", next().c_str(), 1);
        else if (!std::strcmp(argv[i], "--decode-ppl")) decode_ppl = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (model.empty()) {
        std::fprintf(stderr,
                     "usage: qwen_eval --model DIR [--prompt FILE] [--greedy N] [--ppl FILE] "
                     "[--chunk N] [--device N]\n");
        return 1;
    }
    try {
        std::vector<int> devs{device};
        if (!gpus.empty()) {
            devs.clear();
            for (size_t p = 0; p < gpus.size();) {
                size_t q = gpus.find(',', p);
                if (q == std::string::npos) q = gpus.size();
                devs.push_back(std::atoi(gpus.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
        double tl = behemoth_now();
        behemoth::QwenServeOptions so;
        so.max_seqs = serve_seqs;
        behemoth::QwenModel m(model, devs, chunk, max_context, tp, so);
        std::printf("load %.1f s on %zu gpu(s)\n", behemoth_now() - tl, devs.size());
        if (serve_seqs > 0) {
            const long pages = m.kv_pages_total();
            std::printf("serving: %d slots, KV pool %ld pages = %ld tokens; max concurrent seqs @8k %ld, @32k %ld\n",
                        m.max_seqs(), pages, pages * m.page_tokens(),
                        std::min<long>(m.max_seqs(), pages / ((8192 + 63) / 64)),
                        std::min<long>(m.max_seqs(), pages / ((32768 + 63) / 64)));
            const int K = m.topk();
            for (int B : bench_batch) {  // aggregate decode throughput: one prefilled context forked to B slots
                if (B > m.max_seqs()) continue;
                const long need = long(B) * ((bench_ctx + bench_steps + 2 + 63) / 64);
                if (need > m.kv_pages_free()) {
                    std::printf("batch %2d ctx %d: skipped (needs %ld KV pages, %ld free)\n", B, bench_ctx, need, m.kv_pages_free());
                    continue;
                }
                std::vector<int> ctx(static_cast<size_t>(bench_ctx));
                for (int i = 0; i < bench_ctx; ++i) ctx[size_t(i)] = 1000 + (i * 7919) % 100000;
                std::vector<behemoth::QwenCand> cand(size_t(B) * K);
                m.seq_start(0);
                double t0 = behemoth_now();
                m.seq_prefill(0, ctx.data(), ctx.size(), cand.data());
                double t1 = behemoth_now();
                for (int b = 1; b < B; ++b) m.seq_fork(0, b);
                std::vector<int> slots(static_cast<size_t>(B)), toks(static_cast<size_t>(B), 11);
                for (int b = 0; b < B; ++b) slots[size_t(b)] = b;
                m.decode_batch(slots.data(), toks.data(), B, cand.data());  // warm (graph capture)
                double t2 = behemoth_now();
                for (int i = 0; i < bench_steps; ++i) {
                    for (int b = 0; b < B; ++b) toks[size_t(b)] = cand[size_t(b) * K].id;
                    if (!m.decode_batch(slots.data(), toks.data(), B, cand.data())) throw std::runtime_error("out of pages");
                }
                double t3 = behemoth_now();
                const double step = (t3 - t2) / bench_steps;
                std::printf("batch %2d ctx %d: prefill %.2f s, step %.2f ms, %.1f tok/s/seq, aggregate %.1f tok/s\n", B,
                            bench_ctx, t1 - t0, step * 1e3, 1.0 / step, B / step);
                std::fflush(stdout);
                for (int b = 0; b < B; ++b) m.seq_free(b);
            }
            // greedy through the serving path (B = 1); --prompt a,b,c: one line per file
            for (size_t p0 = 0; greedy > 0 && p0 < prompt_file.size();) {
                size_t p1 = prompt_file.find(',', p0);
                if (p1 == std::string::npos) p1 = prompt_file.size();
                const std::string pf = prompt_file.substr(p0, p1 - p0);
                p0 = p1 + 1;
                auto toks = load_tokens(pf);
                std::vector<behemoth::QwenCand> cand(static_cast<size_t>(K));
                m.seq_start(0);
                m.seq_prefill(0, toks.data(), toks.size(), cand.data());
                int slot = 0;
                std::printf("serve greedy %d %s:", greedy, pf.c_str());
                for (int g = 0; g < greedy; ++g) {
                    int t = cand[0].id;
                    std::printf(" %d", t);
                    if (g + 1 < greedy) m.decode_batch(&slot, &t, 1, cand.data());
                }
                std::printf("\n");
                std::fflush(stdout);
                m.seq_free(0);
            }
            if (btest_n > 0) {  // batch invariance: greedy alone vs together
                auto src = load_tokens(btest_file);
                std::vector<std::vector<int>> prompts;
                size_t off = 0;
                for (int i = 0; i < btest_n; ++i) {
                    const size_t len = size_t(17 + (i * 97) % 900);
                    if (off + len > src.size()) off = 0;
                    prompts.emplace_back(src.begin() + long(off), src.begin() + long(off + len));
                    off += len;
                }
                const int G = btest_gen;
                std::vector<std::vector<int>> alone(prompts.size()), batch(prompts.size());
                std::vector<std::vector<float>> gaps(prompts.size());
                std::vector<behemoth::QwenCand> cand(size_t(btest_n) * K);
                for (size_t i = 0; i < prompts.size(); ++i) {
                    m.seq_start(0);
                    m.seq_prefill(0, prompts[i].data(), prompts[i].size(), cand.data());
                    int slot = 0;
                    for (int g = 0; g < G; ++g) {
                        int t = cand[0].id;
                        alone[i].push_back(t);
                        gaps[i].push_back(cand[0].logit - cand[1].logit);
                        if (g + 1 < G) m.decode_batch(&slot, &t, 1, cand.data());
                    }
                }
                double t0 = behemoth_now();
                std::vector<int> slots, toks(prompts.size());
                {  // prefill packed several prompts per pass (prefill_multi), as the worker does
                    size_t i = 0;
                    std::vector<size_t> done(prompts.size(), 0);
                    for (size_t p = 0; p < prompts.size(); ++p) {
                        m.seq_start(int(p));
                        slots.push_back(int(p));
                    }
                    std::vector<behemoth::QwenCand> pc(prompts.size() * K);
                    while (i < prompts.size()) {
                        std::vector<behemoth::QwenModel::PrefillPart> parts;
                        std::vector<size_t> who;
                        long rows = 0;
                        for (size_t p = i; p < prompts.size(); ++p) {
                            const size_t n = std::min<size_t>(size_t(m.chunk()), prompts[p].size() - done[p]);
                            if (!parts.empty() && rows + long(n) > m.chunk()) break;
                            parts.push_back({int(p), prompts[p].data() + done[p], n, done[p] + n == prompts[p].size()});
                            who.push_back(p);
                            rows += long(n);
                        }
                        m.prefill_multi(parts, pc.data());
                        int ci = 0;
                        for (size_t q = 0; q < parts.size(); ++q) {
                            done[who[q]] += parts[q].n;
                            if (parts[q].want) {
                                std::copy(pc.begin() + long(ci) * K, pc.begin() + long(ci + 1) * K, cand.begin() + long(who[q]) * K);
                                ++ci;
                            }
                        }
                        while (i < prompts.size() && done[i] == prompts[i].size()) ++i;
                    }
                }
                for (int g = 0; g < G; ++g) {
                    for (size_t i = 0; i < prompts.size(); ++i) {
                        toks[i] = cand[i * K].id;
                        batch[i].push_back(toks[i]);
                    }
                    if (g + 1 < G) m.decode_batch(slots.data(), toks.data(), int(prompts.size()), cand.data());
                }
                if (const char *dp = std::getenv("QWEN_BTEST_DUMP")) {
                    FILE *f = std::fopen(dp, "w");
                    for (auto &v : alone) {
                        for (int t : v) std::fprintf(f, "%d ", t);
                        std::fprintf(f, "\n");
                    }
                    std::fclose(f);
                }
                int same = 0;
                for (size_t i = 0; i < prompts.size(); ++i) {
                    int d = -1;
                    for (int g = 0; g < G; ++g)
                        if (alone[i][size_t(g)] != batch[i][size_t(g)]) {
                            d = g;
                            break;
                        }
                    if (d < 0) ++same;
                    else std::printf("  prompt %zu (len %zu): diverges at %d (alone top1-top2 gap %.4f)\n", i,
                                     prompts[i].size(), d, gaps[i][size_t(d)]);
                }
                std::printf("batch test: %d prompts x %d greedy tokens: %d identical alone vs batched (%.1f s batched)\n",
                            btest_n, G, same, behemoth_now() - t0);
            }
            return 0;
        }
        const auto &c = m.config();
        std::printf("model: %d layers (hidden %d, %d full-attn), vocab %d, inter %d, rot %d\n", c.layers,
                    c.hidden, int(std::count(c.full_attn.begin(), c.full_attn.end(), 1)), c.vocab, c.inter,
                    c.rot_dim);
        if (!ppl_file.empty()) {
            auto toks = load_tokens(ppl_file);
            double t0 = behemoth_now();
            if (window > 1 && decode_ppl > 0) {  // teacher-forced decode steps (decode attention path)
                const size_t W = size_t(window), S = size_t(stride > 0 ? stride : std::max(1, window - 64));
                const size_t P = size_t(decode_ppl);
                std::vector<float> lg(size_t(c.vocab));
                double nll = 0;
                long cnt = 0;
                for (size_t i = 0; i + W + 1 < toks.size(); i += S) {
                    m.reset();
                    m.prefill(toks.data() + i, P, lg.data());
                    for (size_t t = P; t < W; ++t) {
                        double mx = -1e30, se = 0;
                        for (float v : lg) mx = std::max(mx, double(v));
                        for (float v : lg) se += std::exp(double(v) - mx);
                        nll += mx + std::log(se) - double(lg[size_t(toks[i + t])]);
                        ++cnt;
                        if (t + 1 < W) m.decode(toks[i + t], lg.data());
                    }
                }
                std::printf("decode ppl over %ld predictions (%zu-token windows, %zu-token prefill, stride %zu): %.4f (%.1f tok/s)\n",
                            cnt, W, P, S, std::exp(nll / double(cnt)), double(cnt) / (behemoth_now() - t0));
            } else if (window > 1) {
                const size_t W = size_t(window), S = size_t(stride > 0 ? stride : std::max(1, window - 64));
                double nll = 0;
                long cnt = 0;
                for (size_t i = 0; i + W + 1 < toks.size(); i += S) {
                    double p = m.perplexity(toks.data() + i, W);  // W - 1 predictions, fresh state
                    nll += std::log(p) * double(W - 1);
                    cnt += long(W - 1);
                }
                std::printf("ppl over %ld predictions (%zu-token windows, stride %zu): %.4f (%.1f tok/s)\n", cnt,
                            W, S, std::exp(nll / double(cnt)), double(cnt) / (behemoth_now() - t0));
            } else {
                double ppl = m.perplexity(toks.data(), toks.size());
                std::printf("ppl %zu tokens: %.4f (%.1f tok/s)\n", toks.size(), ppl,
                            double(toks.size()) / (behemoth_now() - t0));
            }
        }
        for (int np : bench) {  // prefill throughput on synthetic tokens (+ optional decode steps)
            std::vector<int> toks(static_cast<size_t>(np));
            for (int i = 0; i < np; ++i) toks[size_t(i)] = 1000 + (i * 7919) % 100000;
            std::vector<float> lg(size_t(c.vocab));
            m.reset();
            double t0 = behemoth_now();
            m.prefill(toks.data(), toks.size(), lg.data());
            double t1 = behemoth_now();
            std::printf("prefill %d tokens: %.3f s (%.1f tok/s)\n", np, t1 - t0, np / (t1 - t0));
            if (bench_decode > 0) {
                int tok = 11;
                m.decode(tok, lg.data());  // warm
                double t2 = behemoth_now();
                for (int i = 0; i < bench_decode; ++i) m.decode(tok, lg.data());
                double t3 = behemoth_now();
                std::printf("decode after %d: %d steps %.3f s (%.2f tok/s)\n", np, bench_decode, t3 - t2,
                            bench_decode / (t3 - t2));
            }
            std::fflush(stdout);
        }
        if (greedy > 0 && !prompt_file.empty()) {
            auto toks = load_tokens(prompt_file);
            double t0 = behemoth_now();
            auto out = m.generate(toks.data(), toks.size(), size_t(greedy));
            std::fprintf(stderr, "generate: %.3f s\n", behemoth_now() - t0);
            std::printf("prompt: %zu tokens; greedy %d:", toks.size(), greedy);
            for (int t : out) std::printf(" %d", t);
            std::printf("\n");
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
