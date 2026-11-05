// behemoth_cuda_model_kl: whole-model KL(p_ref || p_quant) (nats/token) and perplexity of a quantized checkpoint
// over a held-out token set, through the CUDA runtime (the very dequant/decode semantics used for serving: QTIP,
// NVFP4 and mixed checkpoints). The reference side is a cache of log-probabilities, one float32 [T][V] file
// per sequence (seqNNN.f32), written either by an external reference run (the original bf16 model, e.g.
// scratchpad ref.py on CPU) or by this tool itself with --write-ref (self-test: KL of a model vs itself is 0).
//
//   behemoth_cuda_model_kl --model DIR --tokens FILE.i32 --seq-len T --ref-dir DIR [--nseq N] [--device 0]
//                          [--write-ref] [--json OUT.json]
//
// KL is the mean over all T positions of every sequence (position t predicts token t+1; the last position's
// distribution is included for KL, excluded for perplexity, which uses targets 1..T-1). Per-sequence means give the
// standard error. Positions are evaluated with incremental decode (Model::step), the KV cache is reset per sequence.
#include "behemoth/cuda.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
using namespace behemoth::cuda_runtime;
namespace fs = std::filesystem;
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        bool write_ref = false;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k == "--write-ref") {
                write_ref = true;
                continue;
            }
            check(i + 1 < argc && a.emplace(k, argv[i + 1]).second, "missing/duplicate argument " + k);
            ++i;
        }
        for (const char *r : {"--model", "--tokens", "--seq-len", "--ref-dir"})
            check(a.contains(r), std::string("missing ") + r);
        const int T = std::atoi(a["--seq-len"].c_str());
        const int device = a.contains("--device") ? std::atoi(a["--device"].c_str()) : 0;
        std::ifstream tf(a["--tokens"], std::ios::binary | std::ios::ate);
        check(bool(tf) && T > 1, "cannot open tokens / bad --seq-len");
        int nseq = int(size_t(tf.tellg()) / 4 / size_t(T));
        if (a.contains("--nseq"))
            nseq = std::min(nseq, std::atoi(a["--nseq"].c_str()));
        std::vector<int32_t> tok(size_t(nseq) * T);
        tf.seekg(0);
        tf.read(reinterpret_cast<char *>(tok.data()), std::streamsize(tok.size() * 4));
        Model model(a["--model"], device, T, 1, 1);
        const int V = model.config.vocab;
        fs::create_directories(a["--ref-dir"]);
        std::vector<double> seq_kl, seq_nll_ref, seq_nll_q;
        std::vector<float> lq(size_t(T) * V), lr;
        for (int s = 0; s < nseq; ++s) {
            model.reset();
            for (int t = 0; t < T; ++t) {
                const int id = tok[size_t(s) * T + t];
                check(id >= 0 && id < V, "token id outside the model vocabulary");
                auto logits = model.step(id);
                check(logits.size() == size_t(V), "logit count");
                double m = -1e300;
                for (float x : logits)
                    m = std::max(m, double(x));
                double z = 0;
                for (float x : logits)
                    z += std::exp(double(x) - m);
                const float lse = float(m + std::log(z));
                float *row = lq.data() + size_t(t) * V;
                for (int v = 0; v < V; ++v)
                    row[v] = logits[size_t(v)] - lse;
            }
            char name[64];
            std::snprintf(name, sizeof name, "seq%03d.f32", s);
            const fs::path ref_path = fs::path(a["--ref-dir"]) / name;
            if (write_ref) {
                std::ofstream(ref_path, std::ios::binary).write(reinterpret_cast<const char *>(lq.data()),
                                                                 std::streamsize(lq.size() * 4));
                continue;
            }
            lr.resize(lq.size());
            std::ifstream rf(ref_path, std::ios::binary);
            check(bool(rf), "missing reference " + ref_path.string());
            rf.read(reinterpret_cast<char *>(lr.data()), std::streamsize(lr.size() * 4));
            check(size_t(rf.gcount()) == lr.size() * 4, "short reference file " + ref_path.string());
            double kl = 0, nll_r = 0, nll_q = 0;
#pragma omp parallel for reduction(+ : kl) schedule(static)
            for (int t = 0; t < T; ++t) {
                const float *p = lr.data() + size_t(t) * V, *q = lq.data() + size_t(t) * V;
                double row = 0;
                for (int v = 0; v < V; ++v)
                    row += std::exp(double(p[v])) * (double(p[v]) - double(q[v]));
                kl += row;
            }
            for (int t = 0; t + 1 < T; ++t) {
                const int next = tok[size_t(s) * T + t + 1];
                nll_r -= lr[size_t(t) * V + next];
                nll_q -= lq[size_t(t) * V + next];
            }
            seq_kl.push_back(kl / T);
            seq_nll_ref.push_back(nll_r / (T - 1));
            seq_nll_q.push_back(nll_q / (T - 1));
            std::fprintf(stderr, "seq %d: KL %.6f  nll ref %.4f quant %.4f\n", s, seq_kl.back(), seq_nll_ref.back(),
                         seq_nll_q.back());
        }
        if (write_ref) {
            std::cout << "wrote " << nseq << " reference sequences to " << a["--ref-dir"] << '\n';
            return 0;
        }
        auto stats = [](const std::vector<double> &v) {
            double mean = 0, var = 0;
            for (double x : v)
                mean += x;
            mean /= double(v.size());
            for (double x : v)
                var += (x - mean) * (x - mean);
            var = v.size() > 1 ? var / double(v.size() - 1) : 0;
            return std::pair<double, double>{mean, std::sqrt(var / double(v.size()))};
        };
        auto [kl, kl_se] = stats(seq_kl);
        auto [nr, nr_se] = stats(seq_nll_ref);
        auto [nq, nq_se] = stats(seq_nll_q);
        std::vector<double> dn(seq_kl.size());
        for (size_t i = 0; i < dn.size(); ++i)
            dn[i] = seq_nll_q[i] - seq_nll_ref[i];
        auto [dm, dse] = stats(dn);
        std::printf("sequences %d x %d tokens\nKL(ref||quant) = %.6f +- %.6f nats/token (stderr over sequences)\n"
                    "ppl ref %.4f  ppl quant %.4f  (nll %.5f / %.5f, delta nll %.5f +- %.5f)\n",
                    nseq, T, kl, kl_se, std::exp(nr), std::exp(nq), nr, nq, dm, dse);
        if (a.contains("--json")) {
            std::ofstream(a["--json"]) << "{\"model\":\"" << a["--model"] << "\",\"nseq\":" << nseq << ",\"seq_len\":" << T
                                       << ",\"kl\":" << kl << ",\"kl_stderr\":" << kl_se << ",\"nll_ref\":" << nr
                                       << ",\"nll_quant\":" << nq << ",\"delta_nll\":" << dm << ",\"delta_nll_stderr\":" << dse
                                       << ",\"ppl_ref\":" << std::exp(nr) << ",\"ppl_quant\":" << std::exp(nq) << "}\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
