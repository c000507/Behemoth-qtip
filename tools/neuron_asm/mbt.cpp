#include "neff_writer.hpp"
#include <cstdio>
int main() {
    using namespace nasm;
    std::vector<uint8_t> raw(200000, 0x55);
    auto gz = gzip_stored(raw);
    for (auto b : gz) printf("%02x", b);
    printf("\n");
    return 0;
}
