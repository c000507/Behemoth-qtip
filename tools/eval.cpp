// Model-level quality probes that the HTTP API does not expose (works for AQLM and QTIP):
//   behemoth_eval MODEL ppl TEXT_FILE [MAX_TOKENS=4096] [CHUNK=256]
//       teacher-forced perplexity of BOS + tokenize(text) (raw text, no chat template), chunked
//       prefill through the paged KV cache exactly like the server.
//   behemoth_eval MODEL topk PROMPTS_FILE [K=10] [GENERATE=32]
//       for each prompt (separated by lines "-----", raw text after BOS): top-K next-token logits
//       and a greedy continuation of GENERATE tokens. One JSON object per line on stdout.
#include "behemoth/model.hpp"
#include "behemoth/tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace behemoth;
namespace {
struct Session {
    TensorParallelModel &model;
    int length = 0; // tokens already in the cache
    // Runs `tokens` at positions [length, length + n) of one request with every row sampled;
    // returns full-vocab logits [n][32768].
    std::vector<float> run(const std::vector<int> &tokens) {
        const int n = int(tokens.size());
        InferenceBatch batch;
        auto &l = batch.layout;
        l.physical_pages = model.cache_pages();
        const int total = length + n;
        l.table_stride = int(ceil_div(size_t(total), 16));
        if (l.table_stride > l.physical_pages)
            throw std::runtime_error("Sequence does not fit the KV cache");
        l.query_offsets = {0, n};
        l.lengths = {total};
        for (int i = 0; i < n; ++i) {
            l.positions.push_back(length + i);
            batch.sample_rows.push_back(i);
            batch.sample_ids.push_back(0);
        }
        l.page_table.resize(size_t(l.table_stride));
        std::iota(l.page_table.begin(), l.page_table.end(), 0);
        batch.tokens = tokens;
        auto logits = model.forward(batch);
        std::vector<float> out(size_t(n) * 32768);
        for (int r = 0; r < n; ++r)
            for (int half = 0; half < 2; ++half)
                std::copy_n(logits.data() + size_t(half) * size_t(n) * 16384 + size_t(r) * 16384, 16384,
                            out.data() + size_t(r) * 32768 + size_t(half) * 16384);
        length = total;
        return out;
    }
};
double log_softmax_at(const float *row, int token) {
    float m = *std::max_element(row, row + 32768);
    double s = 0;
    for (int v = 0; v < 32768; ++v)
        s += std::exp(double(row[v]) - m);
    return double(row[token]) - m - std::log(s);
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: behemoth_eval MODEL ppl TEXT [MAX_TOKENS] [CHUNK] | MODEL topk PROMPTS [K] [GEN]\n";
            return 2;
        }
        const std::string mode = argv[2];
        std::ifstream f(argv[3]);
        if (!f)
            throw std::runtime_error("Cannot read " + std::string(argv[3]));
        std::string text((std::istreambuf_iterator<char>(f)), {});
        Tokenizer tokenizer(argv[1]);
        RuntimeOptions options;
        options.model = argv[1];
        options.zero_copy_transport = true;
        options.token_budget = 256;
        options.prefill_chunk = 256;
        options.memory_fraction = 0.85;
        if (mode == "ppl") {
            const size_t max_tokens = argc > 4 ? std::stoul(argv[4]) : 4096;
            const size_t chunk = argc > 5 ? std::stoul(argv[5]) : 256;
            options.token_budget = options.prefill_chunk = chunk;
            auto ids = tokenizer.encode(text);
            ids.insert(ids.begin(), 1);
            if (ids.size() > max_tokens + 1)
                ids.resize(max_tokens + 1);
            TensorParallelModel model(options);
            Session s{model};
            double nll = 0;
            size_t counted = 0;
            for (size_t b = 0; b + 1 < ids.size(); b += chunk) {
                const size_t e = std::min(ids.size() - 1, b + chunk);
                std::vector<int> in(ids.begin() + std::ptrdiff_t(b), ids.begin() + std::ptrdiff_t(e));
                auto logits = s.run(in);
                for (size_t i = 0; i < in.size(); ++i) {
                    nll -= log_softmax_at(logits.data() + i * 32768, ids[b + i + 1]);
                    ++counted;
                }
                std::cerr << "ppl: " << e << "/" << ids.size() - 1 << " tokens, running ppl "
                          << std::exp(nll / double(counted)) << '\n';
            }
            nlohmann::json r = {{"model", argv[1]},          {"tokens", counted},
                                {"nll_per_token", nll / double(counted)},
                                {"perplexity", std::exp(nll / double(counted))}};
            std::cout << r.dump() << std::endl;
            return 0;
        }
        if (mode != "topk")
            throw std::invalid_argument("mode must be ppl or topk");
        const int k = argc > 4 ? std::stoi(argv[4]) : 10, generate = argc > 5 ? std::stoi(argv[5]) : 32;
        std::vector<std::string> prompts;
        {
            std::istringstream in(text);
            std::string line, cur;
            while (std::getline(in, line)) {
                if (line == "-----") {
                    if (!cur.empty())
                        prompts.push_back(cur);
                    cur.clear();
                } else
                    cur += (cur.empty() ? "" : "\n") + line;
            }
            if (!cur.empty())
                prompts.push_back(cur);
        }
        TensorParallelModel model(options);
        for (const auto &p : prompts) {
            auto ids = tokenizer.encode(p);
            ids.insert(ids.begin(), 1);
            Session s{model};
            std::vector<float> last;
            for (size_t b = 0; b < ids.size(); b += 256) {
                std::vector<int> in(ids.begin() + std::ptrdiff_t(b),
                                    ids.begin() + std::ptrdiff_t(std::min(ids.size(), b + 256)));
                auto logits = s.run(in);
                last.assign(logits.end() - 32768, logits.end());
            }
            std::vector<int> order(32768);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int a, int b) { return last[size_t(a)] > last[size_t(b)]; });
            nlohmann::json top = nlohmann::json::array();
            for (int i = 0; i < k; ++i)
                top.push_back({{"id", order[size_t(i)]},
                               {"piece", tokenizer.decode({order[size_t(i)]})},
                               {"logit", last[size_t(order[size_t(i)])]},
                               {"logprob", log_softmax_at(last.data(), order[size_t(i)])}});
            std::vector<int> gen;
            int next = order[0];
            for (int g = 0; g < generate; ++g) {
                gen.push_back(next);
                if (next == 2)
                    break;
                auto logits = s.run({next});
                next = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            }
            nlohmann::json r = {{"prompt", p},
                                {"prompt_tokens", ids.size()},
                                {"top", top},
                                {"greedy_ids", gen},
                                {"greedy_text", tokenizer.decode(gen)}};
            std::cout << r.dump() << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
