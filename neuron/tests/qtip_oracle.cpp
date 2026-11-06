#include "behemoth/qtip.hpp"
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using namespace behemoth;
template <typename T> void write(const std::string &path, const std::vector<T> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
    if (!f) throw std::runtime_error("Could not write fixture");
}
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    std::string dir = argv[1];
    qtip::HostLinear l; l.in = 64; l.out = 64;
    std::mt19937 rng(31657);
    l.trellis.resize(size_t(l.in * l.out / 8));
    l.tlut.resize(1024); l.su.resize(l.in); l.sv.resize(l.out);
    for (auto &v : l.trellis) v = uint16_t(rng());
    // Deliberately exercise zero/all-one words and the circular 31 -> 0 join.
    l.trellis[0] = 0; l.trellis[124] = 0xffff; l.trellis[127] = 0xa55a;
    for (auto &v : l.tlut) v = uint16_t((rng() & 0x8000U) | (0x2800U + rng() % 0x1000U));
    for (auto &v : l.su) v = rng() & 1 ? 1.f : -1.f;
    for (auto &v : l.sv) v = (rng() & 1 ? 1.f : -1.f) * .125f;
    auto cb = qtip::make_codebook(l.tlut);
    std::vector<float> dense(size_t(l.in * l.out));
    qtip::decode_rows_bf16(l, cb, 0, l.out, dense.data());
    std::vector<float> x(size_t(l.in * 3));
    for (auto &v : x) v = float(int(rng() % 201) - 100) / 100;
    auto y = qtip::reference_forward(l, x, true, false);
    auto yz = qtip::reference_forward(l, x, true, true);
    for (int k : {12, 28}) {
        int n = k * 32;
        std::vector<float> v(size_t(n), 0.f);
        for (auto &item : v) item = float(int(rng() % 201) - 100) / 100;
        write(dir + "/had_input_" + std::to_string(k) + ".bin", v);
        const int8_t *matrix = qtip::had_matrix(k);
        write(dir + "/had_matrix_" + std::to_string(k) + ".bin", std::vector<int8_t>(matrix, matrix + k * k));
        for (bool transpose : {false, true}) {
            std::vector<double> work(v.begin(), v.end());
            qtip::hadamard(work.data(), n, transpose);
            std::vector<float> result(work.begin(), work.end());
            write(dir + "/had_" + std::to_string(k) + (transpose ? "_t.bin" : "_n.bin"), result);
        }
    }
    write(dir + "/tlut.bin", l.tlut); write(dir + "/trellis.bin", l.trellis); write(dir + "/lut.bin", std::vector<uint32_t>(cb.bf16x2.begin(), cb.bf16x2.end()));
    write(dir + "/dense.bin", dense); write(dir + "/x.bin", x); write(dir + "/su.bin", l.su); write(dir + "/sv.bin", l.sv); write(dir + "/y.bin", y); write(dir + "/yz.bin", yz);
}
