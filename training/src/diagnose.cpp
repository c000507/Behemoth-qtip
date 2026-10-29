#include "diagnose.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <iostream>

namespace qat {
namespace {
std::vector<int> selected(const json &j, const char *key, int length, int limit) {
    auto out = j.value(key, std::vector<int>{});
    require(int(out.size()) <= limit, "too many diagnostic positions");
    std::set<int> seen;
    for (int t : out) require(t >= 0 && t < length && seen.insert(t).second, "invalid/duplicate diagnostic position");
    return out;
}
std::vector<double> log_probs(const std::vector<float> &v) {
    require(v.size() > 1, "invalid diagnostic vocabulary");
    for (float x : v) require(std::isfinite(x), "nonfinite diagnostic logits");
    double maximum = *std::max_element(v.begin(), v.end()), sum = 0;
    for (float x : v) sum += std::exp(double(x) - maximum);
    double z = std::log(sum);
    std::vector<double> p(v.size());
    for (size_t i = 0; i < v.size(); ++i) p[i] = (double(v[i]) - maximum) - z;
    return p;
}
json token_score(const std::vector<double> &p, int token) {
    require(token >= 0 && size_t(token) < p.size(), "diagnostic token outside vocabulary");
    int rank = 1;
    for (double x : p) rank += x > p[token];
    return {{"token", token}, {"logprob", p[token]}, {"rank", rank}};
}
json top(const std::vector<double> &p, int k) {
    std::vector<int> order(p.size());
    std::iota(order.begin(), order.end(), 0);
    k = std::min(k, int(p.size()));
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
        return p[a] == p[b] ? a < b : p[a] > p[b];
    });
    json result = json::array();
    for (int i = 0; i < k; ++i) result.push_back(token_score(p, order[i]));
    return result;
}
}
json diagnostic_request(const json &capture, int position) {
    auto tokens = capture.at("tokens").get<std::vector<int32_t>>();
    int prompt = capture.at("prompt_tokens");
    require(prompt > 0 && prompt < int(tokens.size()), "capture needs prompt plus generated tokens");
    int original = -1;
    if (position < 0) {
        // Exact repeated 32-token spans in generated output. Verify matches to
        // avoid hash collisions. This is an onset hint, not a semantic detector.
        std::map<std::vector<int32_t>, int> spans;
        for (int t = prompt; t + 32 <= int(tokens.size()); ++t) {
            std::vector<int32_t> span(tokens.begin() + t, tokens.begin() + t + 32);
            auto [it, fresh] = spans.emplace(std::move(span), t);
            if (!fresh && t - it->second >= 32) { position = t; original = it->second; break; }
        }
        require(position >= 0, "no repeated 32-token span found; specify --position (zero-based repeated token index)");
    }
    require(position >= prompt && position < int(tokens.size()), "repeat position outside completion");
    int first = std::max(prompt - 1, position - 65), end = std::min(int(tokens.size()) - 1, first + 128);
    std::vector<int> positions;
    for (int t = first; t < end; ++t) positions.push_back(t);
    require(!positions.empty(), "capture needs a token preceding the repeated output");
    tokens.resize(size_t(end + 1));
    return {{"tokens", tokens}, {"logit_positions", positions}, {"prompt_tokens", prompt},
            {"repeat_start", position}, {"previous_match", original},
            {"activation_positions", {position - 1}},
            {"note", "position t predicts tokens[t+1]; exact replay, no template/BOS insertion"}};
}
json diagnostic_trace(const fs::path &path, const json &request, int device, size_t arena, int rows) {
    require(device >= -1 && rows > 0, "invalid trace device/chunk rows");
    auto tokens = request.at("tokens").get<std::vector<int32_t>>();
    require(!tokens.empty() && tokens.size() <= 131072, "trace length outside [1,131072]");
    auto positions = selected(request, "logit_positions", int(tokens.size()), 128);
    auto activation_positions = selected(request, "activation_positions", int(tokens.size()), 8);
    require(!positions.empty(), "trace needs logit_positions");
    if (device < 0) {
        Config c(read_json(path / "config.json"));
        require(c.n <= 1024 && tokens.size() <= 256, "CPU diagnostics are for tiny verification models");
    }
    Model m(path, true);
    for (int t : tokens) require(t >= 0 && t < m.config.vocab, "input token outside vocabulary");
    Batch b;
    b.tokens = tokens; b.labels.assign(tokens.size(), -100); b.segments.assign(tokens.size(), 0);
    b.positions.resize(tokens.size()); std::iota(b.positions.begin(), b.positions.end(), 0); b.boundaries();
    Replica replica(m, device, int(tokens.size()), arena, true, false);
    auto &o = *replica.ops;
    auto x = o.embedding(replica.embedding, b);
    json layers = json::array();
    for (int l = 0; l < m.config.layers; ++l) {
        o.prefetch(l, 0, false); o.acquire(0);
        { Layer block(o, m, l, replica.parameters, replica.gradients); x = block.forward(x, b, rows); }
        json activations = json::array();
        for (int t : activation_positions) {
            std::vector<float> v(m.config.n); o.download(x.row(t, t + 1), v); activations.push_back(v);
        }
        if (!activation_positions.empty()) layers.push_back({{"layer", l}, {"activations", activations}});
        o.release(0);
        std::cerr << "trace " << m.path.filename().string() << ": block " << l + 1 << '/' << m.config.layers << '\n';
    }
    auto &p = m.params[m.final_norm];
    auto w = o.alloc(1, int(p.size));
    o.copy(w, replica.parameters.col(int(p.offset), int(p.offset + p.size))); o.round(w);
    auto norm = o.rms(x, w, m.config.eps);
    json logits = json::array();
    for (int t : positions) {
        auto z = o.mm(norm.y.row(t, t + 1), replica.head);
        std::vector<float> v(z.size()); o.download(z, v);
        (void)log_probs(v); logits.push_back(v);
    }
    o.synchronize();
    return {{"producer", "qtip_diagnose native full-context replay"}, {"model", m.path.string()},
            {"tokenizer_sha256", fs::exists(m.path / "tokenizer.model") ? json(sha256(m.path / "tokenizer.model")) : json(nullptr)},
            {"model_sha256", m.identity}, {"fp8_source", m.fp8_source}, {"tokens", tokens},
            {"logit_positions", positions}, {"logits", logits}, {"activation_positions", activation_positions},
            {"layers", layers}, {"request", request},
            {"arithmetic", "native trainer BF16 activations; no inference FP8 cache/Q/P"}};
}
json diagnostic_compare(const json &reference, const json &candidate, int topk) {
    require(topk > 0 && topk <= 100, "topk outside [1,100]");
    auto tokens = reference.at("tokens").get<std::vector<int>>();
    require(candidate.at("tokens") == reference.at("tokens"), "trace token sequences differ");
    if (reference.contains("tokenizer_sha256") && candidate.contains("tokenizer_sha256") &&
        !reference["tokenizer_sha256"].is_null() && !candidate["tokenizer_sha256"].is_null())
        require(reference["tokenizer_sha256"] == candidate["tokenizer_sha256"], "trace tokenizer mismatch");
    auto positions = selected(reference, "logit_positions", int(tokens.size()), 128);
    require(!positions.empty() && candidate.at("logit_positions") == reference.at("logit_positions"), "trace positions differ");
    require(reference.at("logits").size() == positions.size() && candidate.at("logits").size() == positions.size(), "trace row count mismatch");
    json records = json::array(), layers = json::array();
    double total_kl = 0;
    int disagreements = 0;
    int first_disagreement = -1, peak_position = positions.front();
    double peak_kl = -1;
    size_t vocabulary = 0;
    auto focus = reference.value("request", json::object()).value("focus_tokens", std::vector<int>{});
    for (size_t i = 0; i < positions.size(); ++i) {
        auto a = log_probs(reference["logits"][i].get<std::vector<float>>());
        auto b = log_probs(candidate["logits"][i].get<std::vector<float>>());
        require(a.size() == b.size(), "trace vocabulary mismatch");
        if (!vocabulary) {
            vocabulary = a.size();
            for (int t : tokens) require(t >= 0 && size_t(t) < vocabulary, "trace input token outside vocabulary");
        }
        require(a.size() == vocabulary, "trace vocabulary varies across rows");
        double kl = 0, ha = 0, hb = 0;
        for (size_t k = 0; k < a.size(); ++k) {
            double pa = std::exp(a[k]), pb = std::exp(b[k]);
            kl += pa * (a[k] - b[k]); ha -= pa * a[k]; hb -= pb * b[k];
        }
        int aa = int(std::max_element(a.begin(), a.end()) - a.begin());
        int ab = int(std::max_element(b.begin(), b.end()) - b.begin());
        disagreements += aa != ab; total_kl += std::max(0., kl);
        if (aa != ab && (first_disagreement < 0 || positions[i] < first_disagreement)) first_disagreement = positions[i];
        if (kl > peak_kl) { peak_kl = kl; peak_position = positions[i]; }
        json r = {{"position", positions[i]}, {"kl_reference_to_candidate", std::max(0., kl)},
                  {"reference_entropy", ha}, {"candidate_entropy", hb},
                  {"reference_top", top(a, topk)}, {"candidate_top", top(b, topk)}};
        if (positions[i] + 1 < int(tokens.size())) {
            int target = tokens[positions[i] + 1];
            r["reference_actual_next"] = token_score(a, target);
            r["candidate_actual_next"] = token_score(b, target);
            r["actual_next_logprob_delta"] = b[target] - a[target];
        }
        r["focus"] = json::array();
        for (int t : focus) r["focus"].push_back({{"reference", token_score(a, t)}, {"candidate", token_score(b, t)}});
        records.push_back(r);
    }
    if (reference.contains("layers") && candidate.contains("layers") && !reference["layers"].empty() && !candidate["layers"].empty()) {
        require(reference.at("activation_positions") == candidate.at("activation_positions") && reference["layers"].size() == candidate["layers"].size(), "layer trace positions/count differ");
        for (size_t l = 0; l < reference["layers"].size(); ++l) {
            const auto &a = reference["layers"][l], &b = candidate["layers"][l];
            require(a.at("layer") == b.at("layer") && a.at("activations").size() == reference["activation_positions"].size() && b.at("activations").size() == a["activations"].size(), "layer trace shape mismatch");
            for (size_t i = 0; i < a["activations"].size(); ++i) {
                auto av = a["activations"][i].get<std::vector<float>>(), bv = b["activations"][i].get<std::vector<float>>();
                require(!av.empty() && av.size() == bv.size(), "activation width mismatch");
                double err = 0, ar = 0, br = 0, dot = 0;
                for (size_t k = 0; k < av.size(); ++k) {
                    require(std::isfinite(av[k]) && std::isfinite(bv[k]), "nonfinite activation");
                    err += std::pow(double(av[k]) - bv[k], 2); ar += double(av[k]) * av[k];
                    br += double(bv[k]) * bv[k]; dot += double(av[k]) * bv[k];
                }
                layers.push_back({{"layer", a["layer"]}, {"position", reference["activation_positions"][i]},
                    {"relative_rms", std::sqrt(err / std::max(ar, 1e-30))},
                    {"cosine", ar > 0 && br > 0 ? json(dot / std::sqrt(ar * br)) : json(nullptr)}});
            }
        }
    }
    return {{"reference_model", reference.value("model", "unspecified")}, {"candidate_model", candidate.value("model", "unspecified")},
            {"mean_kl_reference_to_candidate", total_kl / positions.size()},
            {"peak_kl_position", peak_position}, {"first_argmax_disagreement_position", first_disagreement},
            {"argmax_disagreements", disagreements}, {"positions", records}, {"layers", layers},
            {"note", "metrics use untempered logits; activation divergence is diagnostic, not causal proof"}};
}
}
