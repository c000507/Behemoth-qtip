// Independent full QTIP projection oracle for a real model layer.
#include "behemoth/qtip.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace behemoth;
template <typename T> void save(const std::filesystem::path &path, const std::vector<T> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
    if (!f) throw std::runtime_error("Failed to write oracle output");
}
int main(int argc, char **argv) {
    if (argc != 5) { std::cerr << "usage: ORACLE MODEL_DIR LAYER PROJECTION OUT_DIR\n"; return 2; }
    auto root = std::filesystem::path(argv[1]);
    int layer = std::stoi(argv[2]);
    auto config = ModelConfig::read(root / "config.json");
    char filename[64]; std::snprintf(filename, sizeof filename, "model-layer-%05d.safetensors", layer);
    SafeTensorFile shard(root / filename);
    auto linear = qtip::load_linear(shard, layer, argv[3], &config);
    if (linear.is_fp8() || linear.is_nvfp4()) throw std::runtime_error("Expected QTIP projection");
    std::vector<float> x(size_t(linear.in));
    for (int i = 0; i < linear.in; ++i) x[size_t(i)] = .1f * std::sin(float(i) * .071f);
    auto y = qtip::reference_forward(linear, x, true, false);
    save(std::filesystem::path(argv[4]) / "x.bin", x);
    save(std::filesystem::path(argv[4]) / "y.bin", y);
    std::cout << "QTIP CPU reference: in=" << linear.in << " out=" << linear.out << '\n';
}
