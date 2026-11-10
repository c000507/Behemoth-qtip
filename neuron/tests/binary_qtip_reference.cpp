// Identity-Hessian QTIP baseline, using the project's actual tail-biting
// Viterbi encoder. Input: normalized, already rotated row-major float32
// matrix; output: decoded float32 matrix + native packed trellis and tlut.
// Deliberately not advertised as the production Hessian-aware LDLQ pipeline.
#include "qtip_core.hpp"
#include "common.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T> void save(const std::string& p, const std::vector<T>& x) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(T));
    if (!f) throw std::runtime_error("write failed: " + p);
}
int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "INPUT OUT_PREFIX ROWS COLS SCALE_OVERRIDE\n"; return 1; }
    const int m = std::stoi(argv[3]), n = std::stoi(argv[4]);
    const double override = std::stod(argv[5]);
    if (m <= 0 || n <= 0 || m % 32 || n % 32 || !(override > 0)) return 2;
    std::vector<float> w(size_t(m)*n), hat(w.size());
    std::ifstream f(argv[1], std::ios::binary);
    f.read(reinterpret_cast<char*>(w.data()), w.size()*4);
    if (!f || f.peek() != EOF) throw std::runtime_error("input size mismatch");
    const auto& cb = qtip::codebook();
    // Input RMS is set by the training set, not chosen from the test set.
    const float scale = 1.f / (cb.lut_rms * override);
    const int count = m/16 * (n/16);
    std::vector<uint16_t> states(size_t(count)*128), packed(size_t(count)*32), disk(packed.size());
    const int* perm = qtip::perm256();
    #pragma omp parallel for
    for (int t=0; t<count; ++t) {
        const int r=t/(n/16)*16, c=t%(n/16)*16;
        float x[256], hx[256];
        for (int i=0;i<256;++i) x[i]=w[size_t(r+perm[i]/16)*n+c+perm[i]%16]/scale;
        qtip::quantize_seq_ref(x, states.data()+size_t(t)*128, hx);
        for (int i=0;i<256;++i) {
            // Match the deployed BF16 codebook representation.
            hat[size_t(r+perm[i]/16)*n+c+perm[i]%16] = qtip::bf16_to_f32(qtip::f32_to_bf16(hx[i])) * scale;
        }
    }
    qtip::pack_trellis(states.data(), count, packed.data());
    qtip::swizzle_for_kernel(packed.data(), m, n, disk.data());
    const std::string prefix=argv[2];
    save(prefix+".dense.bin",hat); save(prefix+".trellis.bin",disk);
    save(prefix+".tlut.bin",std::vector<uint16_t>(&cb.tlut_h[0][0], &cb.tlut_h[0][0]+1024));
    save(prefix+".scale.bin",std::vector<float>{scale});
    std::cout << "QTIP identity-Hessian Viterbi: " << count << " tiles, scale=" << scale << '\n';
}
