// Dumps the CPU reference decode (qtip::decode_rows_bf16: rotated weights W~ with the device bf16 codebook values) of a
// QTIP projection to a raw f32 file, for the device kernel check neuron/tests/check_qtip_real.py.
// build: g++ -std=c++20 -O2 -fopenmp -I runtime/include -I third_party neuron/cpp/qtip_decode_dump.cpp neuron/cpp/qtip_expand.cpp \
//            runtime/src/qtip.cpp runtime/src/checkpoint.cpp runtime/src/attention.cpp -o qtip_decode_dump
// usage: qtip_decode_dump MODEL_DIR LAYER PROJECTION ROW0 ROWS OUT.f32 [TP RANK]   (rows [ROW0, ROW0 + ROWS) x in, row-major f32;
//        with TP RANK also OUT.words = pack_qtip_shard words of that rank, uint16, for a cross-check with qtip_nki.pack_shard)
#include "qtip_expand.hpp"
#include "behemoth/qtip.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

using namespace behemoth;

int main(int argc, char **argv) {
    if (argc != 7 && argc != 9) { std::cerr << "usage: MODEL_DIR LAYER PROJECTION ROW0 ROWS OUT.f32 [TP RANK]\n"; return 2; }
    const std::filesystem::path root(argv[1]);
    const int layer = std::stoi(argv[2]), row0 = std::stoi(argv[4]), rows = std::stoi(argv[5]);
    auto config = ModelConfig::read(root / "config.json");
    char filename[64];
    std::snprintf(filename, sizeof filename, "model-layer-%05d.safetensors", layer);
    SafeTensorFile shard(root / filename);
    const auto l = qtip::load_linear(shard, layer, argv[3], &config);
    const qtip::Codebook cb = qtip::make_codebook(l.tlut);
    std::vector<float> w(size_t(rows) * size_t(l.in));
    qtip::decode_rows_bf16(l, cb, row0, rows, w.data());
    std::ofstream(argv[6], std::ios::binary).write(reinterpret_cast<const char *>(w.data()), std::streamsize(w.size() * 4));
    std::cout << "wrote " << rows << " x " << l.in << " f32 of " << argv[3] << " (out " << l.out << ")\n";
    if (argc == 9) {
        const std::string name = argv[3];
        const bool col = name == "self_attn.q_proj" || name == "mlp.gate_proj" || name == "mlp.up_proj";
        const auto p = neuron::pack_qtip_shard(l, std::stoi(argv[7]), std::stoi(argv[8]), col);
        std::ofstream(std::string(argv[6]) + ".words", std::ios::binary).write(reinterpret_cast<const char *>(p.words.data()), std::streamsize(p.words.size() * 2));
        std::cout << "wrote words [" << p.groups << "][128][" << p.out_tiles * 32 << "]\n";
    }
    return 0;
}
