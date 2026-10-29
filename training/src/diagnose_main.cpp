#include "diagnose.hpp"
#include <iostream>
#include <cmath>
#include <set>
int main(int argc, char **argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << R"(qtip_diagnose prepare --capture CAPTURE.json --out REQUEST.json [--position TOKEN_INDEX]
qtip_diagnose trace --model MODEL --request REQUEST.json --out TRACE.json
  [--device 0|cpu] [--arena-gb 24] [--rows 1024]
  MODEL accepts original FP8 source or QTIP/mixed checkpoints; no weights change.
qtip_diagnose compare --reference TEACHER.json --candidate STUDENT.json --out REPORT.json [--topk 10]
  Also accepts behemoth_eval trace output, for inference backend/precision comparisons.
Capture exact prompt/generated token IDs with BEHEMOTH_DIAGNOSTIC_DIR on the server.
All outputs must be new files. Position t predicts token t+1. CPU is for tiny models.
)";
            return argc < 2 ? 1 : 0;
        }
        std::string mode = argv[1];
        std::map<std::string, std::string> args;
        for (int i = 2; i < argc; i += 2) {
            qat::require(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "missing/duplicate argument");
        }
        std::set<std::string> valid = mode == "prepare" ? std::set<std::string>{"--capture", "--out", "--position"} :
            mode == "trace" ? std::set<std::string>{"--model", "--request", "--out", "--device", "--arena-gb", "--rows"} :
            std::set<std::string>{"--reference", "--candidate", "--out", "--topk"};
        for (const auto &[k,v] : args) { (void)v; qat::require(valid.contains(k), "unknown option " + k); }
        auto integer = [&](const std::string &key, int fallback) {
            if (!args.contains(key)) return fallback;
            size_t n = 0; int v = std::stoi(args.at(key), &n);
            qat::require(n == args.at(key).size(), "invalid integer"); return v;
        };
        qat::require(args.contains("--out") && !qat::fs::exists(args.at("--out")), "--out must be a new file");
        qat::json result;
        if (mode == "prepare") result = qat::diagnostic_request(qat::read_json(args.at("--capture")), integer("--position", -1));
        else if (mode == "trace") {
            size_t n = 0; std::string value = args.contains("--arena-gb") ? args.at("--arena-gb") : "24";
            double gb = std::stod(value, &n);
            qat::require(n == value.size() && std::isfinite(gb) && gb > .256 && gb <= 1024, "invalid arena budget");
            int device = args.contains("--device") && args.at("--device") == "cpu" ? -1 : integer("--device", 0);
            result = qat::diagnostic_trace(args.at("--model"), qat::read_json(args.at("--request")), device, size_t(gb * 1e9), integer("--rows", 1024));
        } else if (mode == "compare") result = qat::diagnostic_compare(qat::read_json(args.at("--reference")), qat::read_json(args.at("--candidate")), integer("--topk", 10));
        else qat::require(false, "mode must be prepare, trace or compare");
        auto out = qat::fs::absolute(args.at("--out")); qat::fs::create_directories(out.parent_path());
        qat::durable_json(out, result);
        std::cout << "Saved " << out << '\n';
    } catch (const std::exception &e) { std::cerr << "qtip_diagnose: " << e.what() << '\n'; return 1; }
}
