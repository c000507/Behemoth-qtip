// QTIP kernel benchmark on one real layer file (rank-0 TP shards of one layer):
//   behemoth_qtip_bench LAYER_FILE [rows,rows,...] [device]
// Times matmul for the stacked q/k/v, o, gate/up and down shards with every kernel variant, the
// reduce and the Hadamard epilogues, and prints the per-layer sum of the best variants
// (effective GB/s of trellis bytes). Rotates over 4 copies of each shard so the 64 MiB MALL does
// not serve repeated reads.
#include "behemoth/qtip_gpu.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>

using namespace behemoth;
namespace {
template <class F> float time_ms(F f, int iters, hipStream_t s) {
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    f(0); // warm
    HIP_CHECK(hipStreamSynchronize(s));
    HIP_CHECK(hipEventRecord(a, s));
    for (int i = 0; i < iters; ++i)
        f(i);
    HIP_CHECK(hipEventRecord(b, s));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return ms / float(iters);
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: behemoth_qtip_bench LAYER_FILE [rows,...] [device]\n";
            return 2;
        }
        std::vector<int> rows_list{1, 2, 3, 4, 8, 16, 32, 64, 256};
        if (argc > 2) {
            rows_list.clear();
            std::stringstream ss(argv[2]);
            std::string item;
            while (std::getline(ss, item, ','))
                rows_list.push_back(std::stoi(item));
        }
        const int device = argc > 3 ? std::stoi(argv[3]) : 0;
        HIP_CHECK(hipSetDevice(device));
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));
        std::cout << "kernels: " << qtip::kernel_info().dump() << '\n';
        SafeTensorFile file(argv[1]);
        const std::string stem = std::filesystem::path(argv[1]).stem().string();
        const int layer = std::stoi(stem.substr(stem.size() - 5));
        auto lin = [&](const char *n) { return qtip::load_linear(file, layer, n); };
        auto q = lin("self_attn.q_proj"), k = lin("self_attn.k_proj"), v = lin("self_attn.v_proj"),
             o = lin("self_attn.o_proj"), g = lin("mlp.gate_proj"), u = lin("mlp.up_proj"),
             d = lin("mlp.down_proj");
        qtip::GpuCodebook lut;
        lut.upload(qtip::make_codebook(q.tlut));
        constexpr int copies_n = 4;
        struct Shape {
            std::string name;
            std::array<qtip::GpuLinear, copies_n> w;
        };
        std::vector<Shape> shapes(4);
        shapes[0].name = "qkv";
        shapes[1].name = "o";
        shapes[2].name = "gate_up";
        shapes[3].name = "down";
        for (int c = 0; c < copies_n; ++c) {
            shapes[0].w[size_t(c)].upload({q.block_rows(0, 192), k.block_rows(0, 16), v.block_rows(0, 16)}, 12288);
            shapes[1].w[size_t(c)].upload({o.block_rows(0, 192)}, 12288);
            shapes[2].w[size_t(c)].upload({g.block_rows(0, 448), u.block_rows(0, 448)}, 12288);
            shapes[3].w[size_t(c)].upload({d.block_rows(0, 192)}, 28672);
        }
        const int max_rows = *std::max_element(rows_list.begin(), rows_list.end());
        DeviceBuffer<BFloat16> x(size_t(max_rows) * 28672);
        {
            std::vector<BFloat16> h(x.size());
            for (size_t i = 0; i < h.size(); ++i)
                h[i] = BFloat16(float(int(i * 2654435761U % 2001) - 1000) / 1000.f);
            x.upload(h);
        }
        DeviceBuffer<float> partial(size_t(64) * size_t(max_rows) * 28672), z(size_t(max_rows) * 28672);
        nlohmann::json report = nlohmann::json::array();
        for (int rows : rows_list) {
            double best_total = 0, bytes_total = 0;
            for (auto &s : shapes) {
                const auto &w0 = s.w[0];
                std::vector<qtip::MatmulOptions> variants;
                for (auto kernel : {qtip::MatmulKernel::Gemv, qtip::MatmulKernel::Wmma}) {
                    if (kernel == qtip::MatmulKernel::Gemv && rows > 4)
                        continue;
                    for (int copies : {4, 8, 16})
                        for (int slices : {0, 1, 2, 4, 8, 16, 32}) {
                            qtip::MatmulOptions mo{kernel, slices, copies};
                            if (slices && slices * rows * w0.rows > int(partial.size()))
                                continue;
                            variants.push_back(mo);
                        }
                }
                float best = 1e30f;
                std::string best_name;
                for (auto &mo : variants) {
                    const int slices = qtip::matmul_slices(w0, rows, mo);
                    if (size_t(slices) * size_t(rows) * size_t(w0.rows) > partial.size())
                        continue;
                    const float ms = time_ms(
                        [&](int i) {
                            qtip::matmul(s.w[size_t(i % copies_n)], lut.lut.data(), x.data(), w0.in, rows,
                                         partial.data(), slices, stream, mo);
                            qtip::reduce_publish(partial.data(), slices, size_t(rows) * size_t(w0.rows), false,
                                                 z.data(), nullptr, nullptr, nullptr, 0, stream);
                        },
                        20, stream);
                    std::ostringstream n;
                    n << (mo.kernel == qtip::MatmulKernel::Gemv ? "gemv" : "wmma") << "-c" << mo.copies
                      << "-s" << slices << (mo.slices == 0 ? "(auto)" : "");
                    report.push_back({{"rows", rows}, {"shape", s.name}, {"variant", n.str()}, {"ms", ms},
                                      {"GBps", double(w0.bytes()) / ms / 1e6}});
                    if (ms < best) {
                        best = ms;
                        best_name = n.str();
                    }
                }
                std::cout << "rows=" << rows << " " << s.name << " (" << w0.bytes() / 1e6 << " MB): best "
                          << best_name << " " << best << " ms = " << double(w0.bytes()) / best / 1e6
                          << " GB/s\n";
                best_total += best;
                bytes_total += double(w0.bytes());
            }
            // epilogues for one layer at this row count
            DeviceBuffer<float> all(size_t(2) * size_t(rows) * 14336), svf(28672);
            DeviceBuffer<BFloat16> st(size_t(rows) * 12288), qb(size_t(rows) * 6144), kb(size_t(rows) * 512),
                vb(size_t(rows) * 512), h(size_t(rows) * 28672), ub(size_t(rows) * 28672), nw(12288);
            HIP_CHECK(hipMemset(all.data(), 0, all.bytes()));
            HIP_CHECK(hipMemset(svf.data(), 0, svf.bytes()));
            HIP_CHECK(hipMemset(st.data(), 0, st.bytes()));
            HIP_CHECK(hipMemset(h.data(), 0, h.bytes()));
            HIP_CHECK(hipMemset(nw.data(), 0, nw.bytes()));
            const float t_in = time_ms(
                [&](int) {
                    qtip::input_transform(qtip::Source{st.data(), 12288, 0, 12288, 0}, nw.data(), 1e-5f,
                                          svf.data(), 12288, rows, ub.data(), 12288, stream);
                },
                50, stream);
            const float t_qkv = time_ms(
                [&](int) {
                    qtip::output_qkv(all.data(), rows, 0, svf.data(), svf.data(), svf.data(), qb.data(),
                                     kb.data(), vb.data(), stream);
                },
                50, stream);
            const float t_res = time_ms(
                [&](int) { qtip::output_residual(all.data(), rows, svf.data(), st.data(), stream); }, 50,
                stream);
            const float t_gu = time_ms(
                [&](int) {
                    qtip::output_gateup(reinterpret_cast<const BFloat16 *>(all.data()), rows, svf.data(),
                                        svf.data(), h.data(), stream);
                },
                50, stream);
            const float t_din = time_ms(
                [&](int) {
                    qtip::input_transform(qtip::Source{h.data(), 28672, 0, 28672, 0}, nullptr, 0.f,
                                          svf.data(), 28672, rows, ub.data(), 28672, stream);
                },
                50, stream);
            const double transforms = 3 * t_in + t_qkv + 2 * t_res + t_gu + t_din;
            std::cout << "rows=" << rows << " layer matmuls " << best_total << " ms ("
                      << bytes_total / best_total / 1e6 << " GB/s) x88 = " << best_total * 88
                      << " ms; transforms/layer " << transforms << " ms (in " << t_in << ", qkv " << t_qkv
                      << ", res " << t_res << ", gu " << t_gu << ", down-in " << t_din << ") x88 = "
                      << transforms * 88 << " ms\n";
        }
        std::cerr << report.dump() << '\n';
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
