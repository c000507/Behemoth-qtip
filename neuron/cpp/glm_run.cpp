// Teacher-forced logits of a GLM prepared model on NeuronCores through NeuronModel::forward (no scheduler), for oracle checks.
//   glm_run PREPARED_DIR NEFF_DIR TOKENS.json OUT.f32 [chunk|rows2]
// chunk 1 (default): every token is a one-row decode step and is sampled -> [n][vocab]
// chunk N > 1:       prefill chunks of N tokens, the last token of each chunk sampled -> [ceil(n/N)][vocab]
// rows2:             the sequence and its reverse as two rows of every decode step -> [n][2][vocab]
// Prints the per-step wall time (ms) of the decode steps.
#include "neuron_model.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "usage: glm_run PREPARED_DIR NEFF_DIR TOKENS.json OUT.f32 [chunk|rows2]\n";
        return 2;
    }
    try {
        neuron::ModelOptions o;
        o.model = argv[1];
        o.neff_dir = argv[2];
        neuron::NeuronModel model(o);
        std::ifstream f(argv[3]);
        const auto j = nlohmann::json::parse(f);
        const std::vector<int> tokens = (j.is_array() ? j : j.at("tokens")).get<std::vector<int>>();
        const std::string mode = argc > 5 ? argv[5] : "1";
        const bool rows2 = mode == "rows2";
        const int chunk = rows2 ? 1 : std::stoi(mode);
        const int n = int(tokens.size()), pages = (n + 15) / 16, second = pages + 1; // second sequence's first page
        std::ofstream out(argv[4], std::ios::binary);
        std::vector<double> step_ms;
        for (int at = 0; at < n; at += chunk) {
            const int take = std::min(chunk, n - at);
            behemoth::InferenceBatch b;
            auto &l = b.layout;
            l.physical_pages = model.cache_pages();
            l.table_stride = pages;
            l.query_offsets = {0};
            for (int s = 0; s < (rows2 ? 2 : 1); ++s) {
                for (int i = 0; i < take; ++i) {
                    const int p = at + i;
                    b.tokens.push_back(s ? tokens[size_t(n - 1 - p)] : tokens[size_t(p)]);
                    l.positions.push_back(p);
                }
                l.query_offsets.push_back(l.query_offsets.back() + take);
                l.lengths.push_back(at + take);
                for (int pg = 0; pg < pages; ++pg)
                    l.page_table.push_back(s ? second + pg : pg);
                b.sample_rows.push_back(l.query_offsets.back() - 1);
                b.sample_ids.push_back(uint64_t(s + 1));
            }
            const auto t0 = std::chrono::steady_clock::now();
            const auto logits = model.forward(b);
            step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            out.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * 4));
        }
        double total = 0;
        for (size_t i = 1; i < step_ms.size(); ++i) total += step_ms[i];
        std::cerr << "steps " << step_ms.size() << ", first " << step_ms[0] << " ms, mean of the rest "
                  << (step_ms.size() > 1 ? total / double(step_ms.size() - 1) : 0) << " ms; profile " << model.last_profile().dump() << "\n";
    } catch (const std::exception &e) {
        std::cerr << "glm_run: " << e.what() << "\n";
        return 1;
    }
}
