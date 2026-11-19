// Plain-text packing for BitNet distillation: JSONL {"text": ...} documents tokenised with the
// checkpoint's byte-level BPE tokenizer and packed into the native v1 dataset format
// (tokens.i32 / loss.u8 / segments.i32 / positions.i32 + checksummed metadata.json).
#include "glm53.hpp"
#include "train.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

int main(int argc, char **argv) {
    try {
        std::string data, tokenizer, out, split = "train";
        int seqlen = 1024, seed = 1;
        long max_tokens = 0, max_docs = 0;
        bool bos = true;
        std::set<std::string> seen;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            qat::require(seen.insert(a).second, "duplicate option " + a);
            if (a == "--help") {
                std::cout << R"(bitnet_prepare --data DOCS.jsonl --tokenizer MODEL_DIR --out DIR [options]
  --seqlen N        Packed row length (1024)
  --split NAME      metadata split name: train (default) or heldout
  --max-tokens N    Stop after N tokens (0 = all)
  --max-docs N      Stop after N documents (0 = all)
  --seed N          Document shuffle seed (1; 0 = keep file order)
  --no-bos          Do not prepend the BOS token to each document
)";
                return 0;
            }
            if (a == "--no-bos") {
                bos = false;
                continue;
            }
            qat::require(i + 1 < argc, "missing value for " + a);
            std::string v = argv[++i];
            if (a == "--data")
                data = v;
            else if (a == "--tokenizer")
                tokenizer = v;
            else if (a == "--out")
                out = v;
            else if (a == "--split")
                split = v;
            else if (a == "--seqlen")
                seqlen = std::stoi(v);
            else if (a == "--max-tokens")
                max_tokens = std::stol(v);
            else if (a == "--max-docs")
                max_docs = std::stol(v);
            else if (a == "--seed")
                seed = std::stoi(v);
            else
                qat::require(false, "unknown option " + a);
        }
        qat::require(!data.empty() && !tokenizer.empty() && !out.empty(), "--data, --tokenizer, --out required");
        qat::require(seqlen >= 2, "invalid --seqlen");
        qat::require(!std::filesystem::exists(out), "output exists: " + out);
        glm53::Tokenizer tok(tokenizer);
        const auto config = qat::read_json(std::filesystem::path(tokenizer) / "config.json");
        const int bos_id = config.value("bos_token_id", 0);
        std::vector<std::string> docs;
        {
            std::ifstream f(data);
            qat::require(bool(f), "cannot read " + data);
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty())
                    continue;
                auto j = nlohmann::json::parse(line);
                std::string text = j.at("text");
                if (!text.empty())
                    docs.push_back(std::move(text));
                if (max_docs > 0 && long(docs.size()) >= max_docs)
                    break;
            }
        }
        if (seed != 0) {
            std::mt19937_64 rng{uint64_t(seed)};
            std::shuffle(docs.begin(), docs.end(), rng);
        }
        std::vector<int32_t> tokens, segments, positions;
        std::vector<uint8_t> loss;
        long total = 0, used_docs = 0;
        int seg = 0;
        for (const auto &d : docs) {
            if (max_tokens > 0 && total >= max_tokens)
                break;
            auto ids = tok.encode(d);
            if (bos)
                ids.insert(ids.begin(), bos_id);
            for (size_t i = 0; i < ids.size(); ++i) {
                tokens.push_back(ids[i]);
                segments.push_back(seg);
                positions.push_back(int32_t(i));
                loss.push_back(1);
            }
            total += long(ids.size());
            ++seg;
            ++used_docs;
        }
        const size_t nseq = tokens.size() / size_t(seqlen);
        qat::require(nseq > 0, "not enough tokens for one row");
        const size_t keep = nseq * size_t(seqlen);
        tokens.resize(keep), segments.resize(keep), positions.resize(keep), loss.resize(keep);
        const std::filesystem::path dir = out;
        std::filesystem::create_directories(dir);
        auto write = [&](const char *name, const void *p, size_t bytes) {
            std::ofstream f(dir / name, std::ios::binary);
            f.write(static_cast<const char *>(p), std::streamsize(bytes));
            qat::require(bool(f), std::string("cannot write ") + name);
        };
        write("tokens.i32", tokens.data(), tokens.size() * 4);
        write("segments.i32", segments.data(), segments.size() * 4);
        write("positions.i32", positions.data(), positions.size() * 4);
        write("loss.u8", loss.data(), loss.size());
        nlohmann::json meta = {{"format_version", 1}, {"split", split}, {"nseq", nseq}, {"seqlen", seqlen},
                               {"documents", used_docs}, {"tokens", keep}, {"source", data},
                               {"tokenizer", tokenizer}, {"bos", bos}, {"seed", seed},
                               {"files", nlohmann::json::object()}};
        for (const char *name : {"tokens.i32", "loss.u8", "segments.i32", "positions.i32"})
            meta["files"][name] = qat::sha256(dir / name);
        qat::durable_json(dir / "metadata.json", meta);
        std::cout << nlohmann::json({{"nseq", nseq}, {"seqlen", seqlen}, {"documents", used_docs},
                                     {"tokens", keep}, {"out", out}})
                  << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
