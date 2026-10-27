// Qwen BPE tokenizer tests against HF reference vectors (Qwen3.5-0.8B).
// usage: qwen_bpe_tests [model_dir]   (skips vectors if the dir is absent)
#include <cstdio>
#include <filesystem>
#include <vector>

#include "behemoth/qwen_bpe.hpp"

static int g_fail = 0;
#define EXPECT(c, ...)                                                          \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::printf("  FAIL: " __VA_ARGS__);                                \
            std::printf("\n");                                                  \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

int main(int argc, char **argv) {
    std::filesystem::path dir =
        argc >= 2 ? argv[1] : std::filesystem::path("/mnt/nvme/models/Qwen3.5-0.8B");
    if (!std::filesystem::exists(dir / "vocab.json")) {
        std::printf("[qwen_bpe] no vocab.json in %s; skipping\n", dir.string().c_str());
        return 0;
    }
    behemoth::QwenTokenizer tok(dir);
    struct Case {
        const char *text;
        std::vector<int> ids;
    };
    std::vector<Case> cases = {
        {"The quick brown fox jumps over the lazy dog.", {760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13}},
        {"hello world", {14556, 1814}},
        {"  spaces   and\ttabs\n", {220, 12258, 256, 321, 3139, 3325, 198}},
        {"Qwen3.5 test 123 45.67!", {48, 16451, 18, 13, 20, 1228, 220, 16, 17, 18, 220, 19, 20, 13, 21, 22, 0}},
        {"don't stop", {14572, 914, 2842}},
        {"a<|endoftext|>b", {64, 248044, 65}},
    };
    for (auto &c : cases) {
        auto got = tok.encode(c.text);
        bool ok = got == c.ids;
        std::printf("  %-44s %s\n", c.text, ok ? "ok" : "MISMATCH");
        if (!ok) {
            std::printf("    got:");
            for (int i : got) std::printf(" %d", i);
            std::printf("\n    ref:");
            for (int i : c.ids) std::printf(" %d", i);
            std::printf("\n");
        }
        EXPECT(ok, "encode %s", c.text);
        // round trip (special tokens decode literally)
        auto dec = tok.decode(got);
        EXPECT(dec == std::string(c.text), "decode round trip %s", c.text);
    }
    EXPECT(tok.eos() == 248044 && tok.im_end() == 248046, "special ids");
    std::printf(g_fail ? "QWEN BPE TESTS FAILED (%d)\n" : "ALL QWEN BPE TESTS PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
