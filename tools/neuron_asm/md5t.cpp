#include "neff_writer.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
int main() {
    using namespace nasm;
    // md5 of "abc" = 900150983cd24fb0d6963f7d28e17f72
    MD5 m;
    std::string s = "abc";
    m.update((const uint8_t*)s.data(), s.size());
    auto d = m.digest();
    for (auto b : d) printf("%02x", b);
    printf("\n");
    // gzip of {0xb1,16,0,0} via python-parity
    std::vector<uint8_t> raw{0xb1, 16, 0, 0};
    auto gz = gzip_stored(raw);
    for (auto b : gz) printf("%02x", b);
    printf("\n");
    return 0;
}
