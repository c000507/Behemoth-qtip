// test_neff_writer: builds a small NEFF container; python verifies it
// (tar roundtrip + digest A == md5(gz) word-swapped).
#include <cstdio>
#include <string>
#include <vector>

#include "neff_writer.hpp"

int main() {
    using namespace nasm;
    NeffBuild b;
    b.graph_name = "test.neff";
    b.add_text("neff.json", "{\"nodes\": []}\n");
    b.add_text("sg00/def.json", "{\"name\": \"definition\", \"version\": \"0.6\"}\n");
    b.add("sg00/PE0.bin", {0xB1, 16, 0, 0});
    std::vector<uint8_t> out = b.finish();
    for (uint8_t c : out) printf("%02x", c);
    printf("\n");
    return 0;
}
