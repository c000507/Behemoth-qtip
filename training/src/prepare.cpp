#include "train.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sentencepiece_processor.h>
#include <set>
#include <unistd.h>
using namespace qat;
namespace {
struct Row {
    std::vector<int32_t> tokens, segments, positions;
    std::vector<uint8_t> loss;
};
std::string trim(std::string s, bool right = true) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    s.erase(0, first);
    if (right)
        s.erase(s.find_last_not_of(" \t\r\n") + 1);
    return s;
}
struct Tokenizer {
    sentencepiece::SentencePieceProcessor sp;
    std::vector<std::pair<std::string, int>> added;
    int pad;
    explicit Tokenizer(const fs::path &p) {
        auto status = sp.Load((p / "tokenizer.model").string());
        require(status.ok(), status.ToString());
        auto j = read_json(p / "tokenizer_config.json");
        require(j.value("legacy", false),
                "native preparation requires the runtime legacy SentencePiece tokenizer");
        for (auto &it : j.at("added_tokens_decoder").items()) {
            auto &a = it.value();
            require(!a.at("lstrip").get<bool>() && !a.at("rstrip").get<bool>() &&
                        !a.at("single_word").get<bool>() && !a.at("normalized").get<bool>(),
                    "unsupported added-token behavior");
            std::string text = a.at("content");
            int id = std::stoi(it.key());
            require(!text.empty() && id >= 0 && id < sp.GetPieceSize(), "invalid added token");
            added.emplace_back(text, id);
        }
        std::sort(added.begin(), added.end(),
                  [](auto &a, auto &b) { return a.first.size() > b.first.size(); });
        pad = sp.eos_id();
        if (j.contains("pad_token") && j["pad_token"].is_string()) {
            auto text = j["pad_token"].get<std::string>();
            for (auto &[s, id] : added)
                if (s == text)
                    pad = id;
        }
        require(pad >= 0, "tokenizer needs pad/eos");
    }
    Row conversation(const json &messages, int doc) {
        std::string text = "<s>";
        std::vector<std::pair<size_t, size_t>> spans;
        for (auto &m : messages) {
            std::string role = m.at("role"),
                        body = m.value("content", json()).is_null() ? "" : m.at("content").get<std::string>();
            if (role == "assistant") {
                size_t start = 0;
                while ((start = body.find("<think>", start)) != std::string::npos) {
                    auto end = body.find("</think>", start + 7);
                    if (end == std::string::npos)
                        break;
                    end += 8;
                    while (end < body.size() && std::isspace(static_cast<unsigned char>(body[end])))
                        ++end;
                    body.erase(start, end - start);
                }
                body = trim(body, false);
                auto reason = m.value("reasoning_content", json());
                std::string r = reason.is_null() ? "" : trim(reason.get<std::string>());
                if (!r.empty())
                    body = "<think>\n" + r + "\n</think>\n\n" + body;
                text += ' ';
                size_t lo = text.size();
                text += body;
                text += "</s>";
                if (!body.empty())
                    spans.emplace_back(lo, text.size());
            } else if (role == "user")
                text += "[INST] " + body + "[/INST]";
            else if (role == "system")
                text += "[SYSTEM_PROMPT] " + body + "[/SYSTEM_PROMPT]";
            else
                throw std::runtime_error("unsupported chat role " + role);
        }
        Row row;
        auto emit = [&](int id, size_t lo, size_t hi) {
            row.tokens.push_back(id);
            row.segments.push_back(doc);
            row.positions.push_back(int(row.positions.size()));
            bool loss = false;
            for (auto [a, b] : spans)
                loss |= hi > lo && hi > a && lo < b;
            row.loss.push_back(loss);
        };
        size_t segment = 0, i = 0;
        auto flush = [&](size_t end) {
            if (end == segment)
                return;
            sentencepiece::ImmutableSentencePieceText encoded;
            auto status = sp.Encode(text.substr(segment, end - segment), encoded.mutable_proto());
            require(status.ok(), status.ToString());
            for (auto &p : encoded.pieces())
                emit(p.id(), segment + p.begin(), segment + p.end());
        };
        while (i < text.size()) {
            const std::pair<std::string, int> *found = nullptr;
            if (text[i] == '<' || text[i] == '[')
                for (auto &t : added)
                    if (text.compare(i, t.first.size(), t.first) == 0) {
                        found = &t;
                        break;
                    }
            if (found) {
                flush(i);
                emit(found->second, i, i + found->first.size());
                i += found->first.size();
                segment = i;
            } else
                ++i;
        }
        flush(text.size());
        return row;
    }
};
std::vector<Row> pack(const std::vector<Row> &docs, int n, int pad) {
    std::vector<Row> out;
    Row r;
    for (auto &d : docs)
        for (size_t i = 0; i < d.tokens.size(); ++i) {
            r.tokens.push_back(d.tokens[i]);
            r.segments.push_back(d.segments[i]);
            r.positions.push_back(d.positions[i]);
            r.loss.push_back(d.loss[i]);
            if (int(r.tokens.size()) == n) {
                out.push_back(std::move(r));
                r = {};
            }
        }
    if (!r.tokens.empty()) {
        r.tokens.resize(n, pad);
        r.segments.resize(n, -1);
        r.positions.resize(n, 0);
        r.loss.resize(n, 0);
        out.push_back(std::move(r));
    }
    return out;
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k == "--help") {
                std::cout << R"(Native GM-CoT packing with assistant and document masks.
qtip_prepare --data conversations.jsonl --tokenizer MODEL --out DIRECTORY
  --split-metadata FILE  Reuse calibration holdout_conversations (recommended)
  --split train|heldout  Default train; heldout requires --gm-frac 1
  --general FILE        Raw int32 general-text sequences, no repetition
  --general-seqlen N    Default 2048
  --seqlen N            Default 24576
  --gm-frac X           Default .8 (requires --general); heldout use 1
  --seed N              Default 0; native mt19937_64 shuffle
  --max-tokens N        Optional limit, in whole packed sequences
)";
                return 0;
            }
            require(i + 1 < argc, "missing value for " + k);
            require(a.emplace(k, argv[++i]).second, "duplicate option " + k);
        }
        std::set<std::string> valid = {
            "--data",           "--tokenizer", "--out",     "--split-metadata", "--split",     "--general",
            "--general-seqlen", "--seqlen",    "--gm-frac", "--seed",           "--max-tokens"};
        for (auto &[k, v] : a)
            require(valid.contains(k), "unknown option " + k);
        auto s = [&](const char *k, std::string def = "") { return a.contains(k) ? a[k] : def; };
        int n = std::stoi(s("--seqlen", "24576")), gn = std::stoi(s("--general-seqlen", "2048"));
        double fraction = std::stod(s("--gm-frac", ".8"));
        uint64_t seed = std::stoull(s("--seed", "0")),
                 budget = std::stoull(s("--max-tokens", std::to_string(UINT64_MAX)));
        std::string split = s("--split", "train");
        require(n >= 2 && gn >= 2 && fraction > 0 && fraction <= 1 && std::isfinite(fraction) &&
                    budget >= uint64_t(n),
                "invalid lengths/mixture/budget");
        require(split == "train" || split == "heldout", "invalid split");
        require(split != "heldout" || fraction == 1, "heldout requires --gm-frac 1");
        require(fraction == 1 || a.contains("--general"), "--general required for general share");
        require(a.contains("--data") && a.contains("--tokenizer") && a.contains("--out"),
                "--data --tokenizer --out required");
        fs::path out = fs::absolute(a["--out"]);
        require(!fs::exists(out), "output exists");
        Tokenizer tok(a["--tokenizer"]);
        std::ifstream f(a["--data"]);
        require(bool(f), "cannot read conversations");
        std::vector<json> conversations;
        std::string line;
        while (std::getline(f, line))
            if (!trim(line).empty())
                conversations.push_back(json::parse(line).at("messages"));
        require(!conversations.empty(), "empty conversations");
        std::vector<int> order(conversations.size()), held, training;
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 rng(seed);
        std::shuffle(order.begin(), order.end(), rng);
        if (a.contains("--split-metadata"))
            held = read_json(a["--split-metadata"]).at("holdout_conversations").get<std::vector<int>>();
        else
            held.assign(order.begin(), order.begin() + order.size() / 10);
        std::set<int> hs;
        for (int i : held)
            require(i >= 0 && size_t(i) < conversations.size() && hs.insert(i).second, "invalid heldout ids");
        std::sort(held.begin(), held.end());
        for (int i : order)
            if (!hs.contains(i))
                training.push_back(i);
        std::vector<Row> docs;
        for (int i : split == "train" ? training : held)
            docs.push_back(tok.conversation(conversations[i], i));
        auto rows = pack(docs, n, tok.pad);
        std::shuffle(rows.begin(), rows.end(), rng);
        require(!rows.empty(), "empty selected split");
        if (budget != UINT64_MAX)
            rows.resize(std::min(rows.size(), std::max(size_t(1), size_t(budget / n * fraction))));
        size_t gm = rows.size(), general = size_t(std::llround(gm * (1 - fraction) / fraction));
        general = std::min(general, size_t(budget / n) - gm);
        if (general) {
            std::ifstream g(a["--general"], std::ios::binary | std::ios::ate);
            require(bool(g), "cannot read general data");
            auto bytes = size_t(g.tellg());
            require(bytes % (size_t(gn) * 4) == 0 && bytes / 4 >= general * size_t(n),
                    "insufficient or malformed general data; repetition forbidden");
            g.seekg(0);
            std::vector<int> ids(bytes / (size_t(gn) * 4));
            std::iota(ids.begin(), ids.end(), 0);
            std::shuffle(ids.begin(), ids.end(), rng);
            docs.clear();
            size_t have = 0;
            for (int id : ids) {
                Row r;
                r.tokens.resize(gn);
                g.seekg(std::streamoff(size_t(id) * gn * 4));
                g.read(reinterpret_cast<char *>(r.tokens.data()), gn * 4);
                require(bool(g), "short general read");
                r.segments.assign(gn, int(conversations.size() + docs.size()));
                r.positions.resize(gn);
                std::iota(r.positions.begin(), r.positions.end(), 0);
                r.loss.assign(gn, 1);
                docs.push_back(std::move(r));
                have += gn;
                if (have >= general * size_t(n))
                    break;
            }
            auto gr = pack(docs, n, tok.pad);
            rows.insert(rows.end(), std::make_move_iterator(gr.begin()),
                        std::make_move_iterator(gr.begin() + general));
        }
        std::shuffle(rows.begin(), rows.end(), rng);
        fs::create_directories(out.parent_path());
        auto tmp = out.string() + ".tmp-" + std::to_string(getpid());
        require(!fs::exists(tmp), "stale temporary output");
        fs::create_directory(tmp);
        try {
            json meta = {
                {"format_version", 1},
                {"split", split},
                {"seed", seed},
                {"shuffle", "std::mt19937_64"},
                {"nseq", rows.size()},
                {"seqlen", n},
                {"files", json::object()},
                {"data_sha256", sha256(a["--data"])},
                {"tokenizer", a["--tokenizer"]},
                {"tokenizer_model_sha256", sha256(fs::path(a["--tokenizer"]) / "tokenizer.model")},
                {"tokenizer_config_sha256", sha256(fs::path(a["--tokenizer"]) / "tokenizer_config.json")},
                {"training_conversations", split == "train" ? training : std::vector<int>{}},
                {"holdout_conversations", held},
                {"evaluated_conversations", split == "heldout" ? held : std::vector<int>{}},
                {"gm_sequences", gm},
                {"general_sequences", general},
                {"gm_frac", fraction},
                {"think_inlined", true},
                {"document_masking", true}};
            auto write = [&](const char *name, int which) {
                std::ofstream w(fs::path(tmp) / name, std::ios::binary);
                require(bool(w), "cannot create dataset");
                for (auto &r : rows) {
                    if (which == 0)
                        w.write(reinterpret_cast<char *>(r.tokens.data()), n * 4);
                    if (which == 1)
                        w.write(reinterpret_cast<char *>(r.loss.data()), n);
                    if (which == 2)
                        w.write(reinterpret_cast<char *>(r.segments.data()), n * 4);
                    if (which == 3)
                        w.write(reinterpret_cast<char *>(r.positions.data()), n * 4);
                }
                w.flush();
                require(bool(w), "dataset write");
                w.close();
                sync_file(fs::path(tmp) / name);
                meta["files"][name] = sha256(fs::path(tmp) / name);
            };
            write("tokens.i32", 0);
            write("loss.u8", 1);
            write("segments.i32", 2);
            write("positions.i32", 3);
            if (a.contains("--general"))
                meta["general_sha256"] = sha256(a["--general"]);
            durable_json(fs::path(tmp) / "metadata.json", meta);
            fs::rename(tmp, out);
            sync_directory(out.parent_path());
        } catch (...) {
            fs::remove_all(tmp);
            throw;
        }
        std::cout
            << json({{"nseq", rows.size()}, {"seqlen", n}, {"heldout", held.size()}, {"out", out.string()}})
                   .dump()
            << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "qtip_prepare: " << e.what() << '\n';
        return 1;
    }
}
