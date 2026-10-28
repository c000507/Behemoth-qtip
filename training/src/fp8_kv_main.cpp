#include "train.hpp"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "qtip_fp8_kv --model QTIP --source ORIGINAL_FP8 --out NEW_DIRECTORY\n";
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            qat::require(i + 1 < argc, "missing option value");
            std::string key = argv[i];
            qat::require(key == "--model" || key == "--source" || key == "--out", "unknown option " + key);
            qat::require(args.emplace(key, argv[i + 1]).second, "duplicate option " + key);
        }
        qat::require(args.size() == 3, "required: --model QTIP --source ORIGINAL_FP8 --out NEW_DIRECTORY");
        qat::replace_kv_fp8(args.at("--model"), args.at("--source"), args.at("--out"));
        std::cout << "Saved mixed QTIP/FP8 K/V checkpoint\n";
    } catch (const std::exception &e) {
        std::cerr << "qtip_fp8_kv: " << e.what() << '\n';
        return 1;
    }
}
