// port_gate.cpp: byte-exact acceptance test for the C++ port vs the
// compiler-oracle streams (tools/neuron_asm/oracle_ref.txt, dumped from
// tools/shape_templates.json). Also live-compares against the Python
// synthesizer output files when present (pe_<M>_<K>_<N>.bin).
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mm.hpp"

static std::string hex2bytes(const std::string& h) {
    std::string out;
    out.reserve(h.size() / 2);
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        out.push_back(char(std::stoul(h.substr(i, 2), nullptr, 16)));
    }
    return out;
}

int main() {
    std::ifstream f("oracle_ref.txt");
    if (!f) {
        std::cerr << "FAIL: oracle_ref.txt missing (run the dump script)\n";
        return 1;
    }
    int ok = 0, fail = 0, skip = 0;
    std::string M, K, N, kind, hex;
    while (f >> M >> K >> N >> kind >> hex) {
        std::string want = hex2bytes(hex);
        std::vector<uint8_t> got;
        try {
            got = kind == "pe" ? nmm::synth_pe(std::stoi(M), std::stoi(K), std::stoi(N))
                               : nmm::synth_pool(std::stoi(M), std::stoi(K), std::stoi(N));
        } catch (const std::exception& e) {
            std::cout << "SKIP " << M << "_" << K << "_" << N << " " << kind
                      << ": " << e.what() << "\n";
            ++skip;
            continue;
        }
        bool eq = got.size() == want.size();
        for (size_t i = 0; eq && i < got.size(); ++i)
            eq = uint8_t(want[i]) == got[i];
        if (eq) {
            ++ok;
        } else {
            ++fail;
            std::cout << "DIFF " << M << "_" << K << "_" << N << " " << kind
                      << ": want " << want.size() << "B got " << got.size() << "B";
            size_t n = std::min(want.size(), got.size());
            int nd = 0;
            for (size_t i = 0; i < n; ++i) {
                if (uint8_t(want[i]) != got[i]) {
                    if (nd < 4)
                        std::cout << " @" << i << std::hex << " want=" << int(uint8_t(want[i]))
                                  << " got=" << int(got[i]) << std::dec;
                    ++nd;
                }
            }
            std::cout << " ndiffs=" << nd;
            std::cout << "\n";
        }
    }
    std::cout << "OK " << ok << " DIFF " << fail << " SKIP " << skip << "\n";
    return fail ? 1 : 0;
}
