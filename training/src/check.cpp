#include "train.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
using namespace qat;
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        bool capture = false;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k == "--help") {
                std::cout << R"(qtip_check --model MODEL --trace TRACE.json [--device cpu|0]
  --capture --eval-binary PATH --prompt REQUEST.json [--chunk 256]
Capture invokes the native behemoth_eval and stamps a source-model digest.
Check reports exact logits and diagnostic error; any mismatch exits 1.
)";
                return 0;
            }
            if (k == "--capture") {
                capture = true;
                continue;
            }
            require(i + 1 < argc, "missing option value");
            require(a.emplace(k, argv[++i]).second, "duplicate option");
        }
        for (auto &[k, v] : a)
            require(k == "--model" || k == "--trace" || k == "--device" || k == "--eval-binary" ||
                        k == "--prompt" || k == "--chunk",
                    "unknown option " + k);
        require(a.contains("--model") && a.contains("--trace"), "--model --trace required");
        auto model = fs::canonical(a["--model"]);
        auto trace = fs::absolute(a["--trace"]);
        if (capture) {
            require(a.contains("--eval-binary") && a.contains("--prompt"),
                    "--eval-binary --prompt required for capture");
            require(!fs::exists(trace), "trace output exists");
            auto before = model_identity(model), binary = fs::canonical(a["--eval-binary"]).string(),
                 prompt = fs::canonical(a["--prompt"]).string(),
                 chunk = a.contains("--chunk") ? a["--chunk"] : "256";
            pid_t child = fork();
            require(child >= 0, "fork failed");
            if (child == 0) {
                execl(binary.c_str(), binary.c_str(), model.c_str(), "trace", prompt.c_str(), trace.c_str(),
                      chunk.c_str(), static_cast<char *>(nullptr));
                _exit(127);
            }
            int status = 0;
            require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                    "native runtime capture failed");
            auto j = read_json(trace), request = read_json(prompt);
            require(j["tokens"] == request["tokens"] && j["logit_positions"] == request["logit_positions"] &&
                        fs::canonical(j.at("model").get<std::string>()) == model,
                    "runtime returned a different request/model");
            require(before == model_identity(model), "model changed during capture");
            j["model_sha256"] = before;
            j["producer"] = "behemoth_eval trace";
            durable_json(trace, j);
            std::cout << "captured " << trace << '\n';
            return 0;
        }
        auto j = read_json(trace);
        Model m(model);
        require(j.at("model_sha256").get<std::string>() == m.identity, "trace source model differs");
        Batch b;
        b.tokens = j.at("tokens").get<std::vector<int32_t>>();
        require(!b.tokens.empty(), "empty trace prompt");
        b.labels.assign(b.tokens.size(), -100);
        b.segments.assign(b.tokens.size(), 0);
        b.positions.resize(b.tokens.size());
        for (size_t i = 0; i < b.tokens.size(); ++i)
            b.positions[i] = int(i);
        b.boundaries();
        auto selected = j.at("logit_positions").get<std::vector<int>>();
        require(!selected.empty() && selected.size() <= 128, "invalid selected positions");
        int device = a.contains("--device") && a["--device"] != "cpu" ? std::stoi(a["--device"]) : -1;
        require(device >= 0 || m.config.n <= 1024, "CPU trace check is for tiny models");
        Replica r(m, device, int(b.tokens.size()), size_t(24) * 1000000000, true);
        auto &o = *r.ops;
        auto x = o.embedding(r.embedding, b);
        o.prefetch(0, 0, false);
        if (m.config.layers > 1)
            o.prefetch(1, 1, false);
        for (int l = 0; l < m.config.layers; ++l) {
            int s = l % 2;
            o.acquire(s);
            {
                Layer layer(o, m, l, r.parameters, r.gradients);
                x = layer.forward(x, b, 1024);
            }
            o.release(s);
            if (l + 2 < m.config.layers)
                o.prefetch(l + 2, s, false);
        }
        auto &p = m.params[m.final_norm];
        auto w = o.alloc(1, int(p.size));
        o.copy(w, r.parameters.col(int(p.offset), int(p.offset + p.size)));
        o.round(w);
        auto norm = o.rms(x, w, m.config.eps);
        bool exact = true;
        double error = 0, sq = 0, ref = 0;
        size_t same = 0;
        require(j.at("logits").size() == selected.size(), "trace row count mismatch");
        for (size_t i = 0; i < selected.size(); ++i) {
            int t = selected[i];
            require(t >= 0 && t < x.rows, "selected position outside prompt");
            auto logits = o.mm(norm.y.row(t, t + 1), r.head);
            std::vector<float> v(logits.size()), target = j["logits"][i].get<std::vector<float>>();
            require(target.size() == v.size(), "trace vocabulary mismatch");
            o.download(logits, v);
            for (size_t k = 0; k < v.size(); ++k) {
                require(std::isfinite(v[k]) && std::isfinite(target[k]), "nonfinite logits");
                exact &= std::bit_cast<uint32_t>(v[k]) == std::bit_cast<uint32_t>(target[k]);
                double d = double(v[k]) - target[k];
                error = std::max(error, std::abs(d));
                sq += d * d;
                ref += double(target[k]) * target[k];
            }
            same += std::max_element(v.begin(), v.end()) - v.begin() ==
                    std::max_element(target.begin(), target.end()) - target.begin();
        }
        std::cout << json({{"bit_identical", exact},
                           {"max_abs_diff", error},
                           {"rel_rms", std::sqrt(sq / std::max(ref, 1e-30))},
                           {"argmax_agreement", double(same) / selected.size()},
                           {"positions", selected.size()}})
                         .dump()
                  << '\n';
        return exact ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "qtip_check: " << e.what() << '\n';
        return 1;
    }
}
