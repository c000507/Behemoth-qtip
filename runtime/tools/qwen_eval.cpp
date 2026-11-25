// Qwen3.5 eval: greedy generation + perplexity from a token file (int32 raw).
// usage: qwen_eval --model DIR [--tokens FILE] [--prompt FILE] [--greedy N] [--ppl FILE]
//          [--chunk N] [--device N] [--max-context N] [--window W [--stride S]]
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
        else if (!std::strcmp(argv[i], "--window")) window = std::atoi(next().c_str());
        else if (!std::strcmp(argv[i], "--stride")) stride = std::atoi(next().c_str());
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
        behemoth::QwenModel m(model, devs, chunk, max_context, tp);
        std::printf("load %.1f s on %zu gpu(s)\n", behemoth_now() - tl, devs.size());
        const auto &c = m.config();
        std::printf("model: %d layers (hidden %d, %d full-attn), vocab %d, inter %d, rot %d\n", c.layers,
                    c.hidden, int(std::count(c.full_attn.begin(), c.full_attn.end(), 1)), c.vocab, c.inter,
                    c.rot_dim);
        if (!ppl_file.empty()) {
            auto toks = load_tokens(ppl_file);
            double t0 = behemoth_now();
            if (window > 1) {
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
