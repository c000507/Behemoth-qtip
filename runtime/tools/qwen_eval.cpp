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
    std::string model, prompt_file, tokens_file, ppl_file;
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
        behemoth::QwenModel m(model, device, chunk, max_context);
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
        if (greedy > 0 && !prompt_file.empty()) {
            auto toks = load_tokens(prompt_file);
            auto out = m.generate(toks.data(), toks.size(), size_t(greedy));
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
