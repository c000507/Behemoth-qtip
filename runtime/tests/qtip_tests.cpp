// QTIP runtime tests (CPU references + GPU kernels).
//   behemoth_qtip_tests                       synthetic tests only (CPU + one GPU, < 1 GB)
//   behemoth_qtip_tests LAYER_FILE REFVEC     + real-tensor tests on one qtip_quant layer file
//       (model-layer-XXXXX.safetensors) against the python reference vectors written by
//       tools/qtip_ref_vectors.py (decode_compressed rows, QuantizedLinear forward).
#include "behemoth/qtip_gpu.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>

using namespace behemoth;
namespace fs = std::filesystem;

namespace {
int failures = 0;
void check(bool ok, const std::string &what) {
    std::cout << (ok ? "PASS " : "FAIL ") << what << '\n';
    if (!ok)
        ++failures;
}
struct Err {
    double rel_rms = 0, max_abs = 0, ref_rms = 0;
    size_t identical = 0, n = 0;
};
template <class A, class B> Err compare(const A &actual, const B &expected) {
    Err e;
    double se = 0, sr = 0;
    e.n = expected.size();
    for (size_t i = 0; i < e.n; ++i) {
        const double a = double(actual[i]), b = double(expected[i]);
        if (!std::isfinite(a)) {
            e.max_abs = INFINITY;
            se = INFINITY;
            continue;
        }
        se += (a - b) * (a - b);
        sr += b * b;
        e.max_abs = std::max(e.max_abs, std::abs(a - b));
        e.identical += a == b;
    }
    e.ref_rms = std::sqrt(sr / double(std::max<size_t>(e.n, 1)));
    e.rel_rms = sr > 0 ? std::sqrt(se / sr) : std::sqrt(se);
    return e;
}
std::string describe(const Err &e) {
    std::ostringstream s;
    s << "rel_rms=" << e.rel_rms << " max_abs=" << e.max_abs << " ref_rms=" << e.ref_rms
      << " identical=" << e.identical << "/" << e.n;
    return s.str();
}
std::vector<float> read_f32(const fs::path &p, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read " + p.string());
    std::vector<float> v(n);
    f.read(reinterpret_cast<char *>(v.data()), std::streamsize(n * 4));
    if (size_t(f.gcount()) != n * 4)
        throw std::runtime_error("Short reference file " + p.string());
    return v;
}
std::vector<float> to_float(const std::vector<BFloat16> &v) {
    std::vector<float> r(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        r[i] = float(v[i]);
    return r;
}
std::vector<BFloat16> to_bf16(const std::vector<float> &v) {
    std::vector<BFloat16> r(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        r[i] = BFloat16(v[i]);
    return r;
}
uint16_t f16_bits(float x) {
    // test-only fp16 encoder (round to nearest even) for synthetic tluts
    uint32_t u = std::bit_cast<uint32_t>(x);
    uint32_t sign = (u >> 16) & 0x8000U;
    float a = std::abs(x);
    if (a < 6.1035156e-05f) {
        uint32_t m = uint32_t(std::nearbyint(a / 5.9604645e-08f));
        return uint16_t(sign | m);
    }
    int e;
    float f = std::frexp(a, &e); // a = f * 2^e, f in [0.5, 1)
    uint32_t mant = uint32_t(std::nearbyint((f * 2 - 1) * 1024));
    int ex = e - 1 + 15;
    if (mant == 1024) {
        mant = 0;
        ++ex;
    }
    return uint16_t(sign | uint32_t(ex) << 10 | mant);
}
qtip::HostLinear synthetic(int out, int in, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0, 1);
    qtip::HostLinear l;
    l.in = in;
    l.out = out;
    l.trellis.resize(size_t(out / 16) * size_t(in / 16) * 32);
    for (auto &w : l.trellis)
        w = uint16_t(rng() & 0xffffU);
    l.tlut.resize(1024);
    for (auto &t : l.tlut)
        t = f16_bits(normal(rng));
    l.su.resize(size_t(in));
    for (auto &s : l.su)
        s = (rng() & 1) ? 1.f : -1.f;
    l.sv.resize(size_t(out));
    for (auto &s : l.sv)
        s = ((rng() & 1) ? 1.f : -1.f) * 0.0017f * (1 + 0.1f * normal(rng));
    return l;
}
// x . W^T in double for row-major W [out][in] (float) and x [rows][in] (float).
std::vector<float> dense_matmul(const std::vector<float> &w, int out, int in,
                                const std::vector<float> &x) {
    const size_t rows = x.size() / size_t(in);
    std::vector<float> y(rows * size_t(out));
#pragma omp parallel for
    for (int o = 0; o < out; ++o)
        for (size_t r = 0; r < rows; ++r) {
            double s = 0;
            for (int i = 0; i < in; ++i)
                s += double(w[size_t(o) * size_t(in) + size_t(i)]) * x[r * size_t(in) + size_t(i)];
            y[r * size_t(out) + size_t(o)] = float(s);
        }
    return y;
}
std::vector<float> random_vector(size_t n, unsigned seed, float scale = 1) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0, scale);
    std::vector<float> v(n);
    for (auto &x : v)
        x = normal(rng);
    return v;
}
std::vector<float> bf16_round(std::vector<float> v) {
    for (auto &x : v)
        x = float(BFloat16(x));
    return v;
}

// ---------------------------------------------------------------- CPU
void test_hadamard_cpu() {
    for (int n : {1024, 12288, 28672}) {
        const int k = qtip::had_k(n), p = n / k;
        auto x = random_vector(size_t(n), unsigned(n));
        for (bool t : {false, true}) {
            std::vector<double> y(x.begin(), x.end());
            qtip::hadamard(y.data(), n, t);
            // explicit definition for 24 outputs: (hadK^(T) (x) H_p) x / sqrt(n)
            std::mt19937 rng(5);
            double worst = 0;
            for (int s = 0; s < 24; ++s) {
                const int o = int(rng() % unsigned(n)), k2 = o / p, i = o % p;
                double acc = 0;
                for (int c = 0; c < k; ++c) {
                    const int hk = k == 1 ? 1 : (t ? qtip::had_matrix(k)[c * k + k2] : qtip::had_matrix(k)[k2 * k + c]);
                    for (int j = 0; j < p; ++j) {
                        const int h = (__builtin_popcount(unsigned(i & j)) & 1) ? -1 : 1;
                        acc += hk * h * double(x[size_t(c * p + j)]);
                    }
                }
                acc /= std::sqrt(double(n));
                worst = std::max(worst, std::abs(acc - y[size_t(o)]));
            }
            // inverse: hadU(hadUt(x)) = x
            std::vector<double> z = y;
            qtip::hadamard(z.data(), n, !t);
            double inv = 0;
            for (int i = 0; i < n; ++i)
                inv = std::max(inv, std::abs(z[size_t(i)] - x[size_t(i)]));
            check(worst < 1e-9 && inv < 1e-9, "cpu hadamard n=" + std::to_string(n) +
                                                  (t ? " hadUt" : " hadU") + " explicit_err=" +
                                                  std::to_string(worst) + " inverse_err=" +
                                                  std::to_string(inv));
        }
    }
    bool threw = false;
    try {
        qtip::had_k(3000);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "cpu hadamard rejects unsupported sizes");
}
void test_codebook_cpu() {
    auto l = synthetic(32, 1024, 3);
    auto cb = qtip::make_codebook(l.tlut);
    bool ok = true;
    for (uint32_t s = 0; s < 65536; s += 7) {
        const uint64_t p = uint64_t(s + 1) * s;
        const int idx = int((p >> 6) & 511), flip = int((p >> 15) & 1);
        const int e = qtip::lut_entry(s);
        float c0 = decode_checkpoint_f16(l.tlut[size_t(idx) * 2]) * (flip ? -1.f : 1.f);
        float c1 = decode_checkpoint_f16(l.tlut[size_t(idx) * 2 + 1]);
        ok &= e == (flip << 9 | idx) && cb.c0[size_t(e)] == c0 && cb.c1[size_t(e)] == c1;
        // garbage above bit 15 must not change the entry (the kernels rely on it)
        ok &= qtip::lut_entry(s | 0xabc0000U) == e;
    }
    check(ok, "cpu quantlut_sym codebook (sign fold, 16-bit state)");
}

// ---------------------------------------------------------------- GPU
qtip::GpuLinear upload_shard(const qtip::HostLinear &l, int block_row, int count) {
    qtip::GpuLinear g;
    g.upload({l.block_rows(block_row, count)}, l.in);
    return g;
}
void test_hadamard_gpu() {
    for (int n : {1024, 12288, 28672}) {
        const int rows = 3;
        auto x = random_vector(size_t(n) * rows, unsigned(n + 1));
        DeviceBuffer<float> dx, dy(size_t(n) * rows);
        dx.upload(x);
        for (bool t : {false, true}) {
            if (n == 1024 && t)
                continue;
            qtip::hadamard_rows(dx.data(), n, rows, t, dy.data(), nullptr);
            HIP_CHECK(hipDeviceSynchronize());
            auto y = dy.download();
            std::vector<double> ref(x.begin(), x.end());
            for (int r = 0; r < rows; ++r)
                qtip::hadamard(ref.data() + size_t(r) * size_t(n), n, t);
            auto e = compare(y, ref);
            check(e.rel_rms < 1e-6, "gpu hadamard n=" + std::to_string(n) + (t ? " hadUt " : " hadU ") +
                                        describe(e));
        }
    }
}
void test_wmma_probe() {
    auto a = to_bf16(random_vector(256, 11)), b = to_bf16(random_vector(256, 12));
    DeviceBuffer<BFloat16> da, db;
    DeviceBuffer<float> dd(256);
    da.upload(a);
    db.upload(b);
    qtip::wmma_probe(da.data(), db.data(), dd.data(), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto d = dd.download();
    std::vector<double> ref(256);
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            double s = 0;
            for (int k = 0; k < 16; ++k)
                s += double(float(a[size_t(m * 16 + k)])) * float(b[size_t(k * 16 + n)]);
            ref[size_t(m * 16 + n)] = s;
        }
    auto e = compare(d, ref);
    check(e.rel_rms < 1e-6, "gpu wmma bf16 16x16x16 fragment layout " + describe(e));
}
void test_decode_gpu(const qtip::HostLinear &l, int block_row, int count, const std::string &name) {
    auto cb = qtip::make_codebook(l.tlut);
    qtip::GpuCodebook lut;
    lut.upload(cb);
    auto g = upload_shard(l, block_row, count);
    DeviceBuffer<BFloat16> w(size_t(g.rows) * size_t(l.in));
    qtip::decode_dense(g, lut.lut.data(), w.data(), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = to_float(w.download());
    std::vector<float> ref(got.size());
    qtip::decode_rows_bf16(l, cb, block_row * 32, g.rows, ref.data());
    auto e = compare(got, ref);
    check(e.identical == e.n, "gpu trellis decode (bf16 LUT) bit-exact vs CPU decoder: " + name + " " +
                                  describe(e));
}
void test_matmul_gpu(const qtip::HostLinear &l, int block_row, int count, const std::string &name,
                     const std::vector<int> &rows_list, bool all_variants) {
    auto cb = qtip::make_codebook(l.tlut);
    qtip::GpuCodebook lut;
    lut.upload(cb);
    auto g = upload_shard(l, block_row, count);
    std::vector<float> w(size_t(g.rows) * size_t(l.in));
    qtip::decode_rows_bf16(l, cb, block_row * 32, g.rows, w.data());
    for (int rows : rows_list) {
        auto x = bf16_round(random_vector(size_t(rows) * size_t(l.in), unsigned(rows * 31 + l.in)));
        auto ref = dense_matmul(w, g.rows, l.in, x);
        DeviceBuffer<BFloat16> dx;
        dx.upload(to_bf16(x));
        std::vector<qtip::MatmulOptions> variants{{}};
        if (all_variants)
            for (auto kernel : {qtip::MatmulKernel::Gemv, qtip::MatmulKernel::Wmma})
                for (int copies : {1, 4, 8, 16})
                    for (int slices : {1, 3}) {
                        if (kernel == qtip::MatmulKernel::Gemv && rows > 4)
                            continue;
                        variants.push_back({kernel, slices, copies});
                    }
        for (const auto &o : variants) {
            const int slices = qtip::matmul_slices(g, rows, o);
            DeviceBuffer<float> partial(size_t(slices) * size_t(rows) * size_t(g.rows));
            DeviceBuffer<float> z(size_t(rows) * size_t(g.rows));
            qtip::matmul(g, lut.lut.data(), dx.data(), l.in, rows, partial.data(), slices, nullptr, o);
            qtip::reduce_publish(partial.data(), slices, size_t(rows) * size_t(g.rows), false, z.data(),
                                 nullptr, nullptr, nullptr, 0, nullptr);
            HIP_CHECK(hipDeviceSynchronize());
            auto e = compare(z.download(), ref);
            const char *kn = o.kernel == qtip::MatmulKernel::Gemv   ? "gemv"
                             : o.kernel == qtip::MatmulKernel::Wmma ? "wmma"
                                                                    : "auto";
            check(e.rel_rms < 1e-5, "gpu matmul " + name + " rows=" + std::to_string(rows) + " " + kn +
                                        " copies=" + std::to_string(o.copies) + " slices=" +
                                        std::to_string(slices) + " " + describe(e));
        }
    }
}
// Full linear on the GPU: input_transform (SU, hadUt), matmul over both TP shards, reduce, hadU,
// SV — vs the python reference QuantizedLinear forward and the double reference.
void test_linear_gpu(const qtip::HostLinear &l, const std::vector<float> &x, int rows,
                     const std::vector<float> &y_ref, const std::string &name) {
    auto cb = qtip::make_codebook(l.tlut);
    qtip::GpuCodebook lut;
    lut.upload(cb);
    const int half = l.out / 64; // block-rows per rank
    std::array<qtip::GpuLinear, 2> shard{upload_shard(l, 0, half), upload_shard(l, half, half)};
    DeviceBuffer<BFloat16> dx, du(size_t(rows) * size_t(l.in));
    dx.upload(to_bf16(x));
    DeviceBuffer<float> su, z(size_t(rows) * size_t(l.out)), y(size_t(rows) * size_t(l.out));
    su.upload(l.su);
    qtip::input_transform(qtip::Source{dx.data(), l.in, 0, l.in, 0}, nullptr, 0.f, su.data(), l.in, rows,
                          du.data(), l.in, nullptr);
    // per-shard results, then scatter into [rows][out]
    std::vector<float> zz(size_t(rows) * size_t(l.out));
    for (int r = 0; r < 2; ++r) {
        const int slices = qtip::matmul_slices(shard[size_t(r)], rows);
        DeviceBuffer<float> partial(size_t(slices) * size_t(rows) * size_t(shard[size_t(r)].rows)),
            zr(size_t(rows) * size_t(shard[size_t(r)].rows));
        qtip::matmul(shard[size_t(r)], lut.lut.data(), du.data(), l.in, rows, partial.data(), slices,
                     nullptr);
        qtip::reduce_publish(partial.data(), slices, size_t(rows) * size_t(shard[size_t(r)].rows), false,
                             zr.data(), nullptr, nullptr, nullptr, 0, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        auto h = zr.download();
        for (int t = 0; t < rows; ++t)
            std::copy_n(h.begin() + std::ptrdiff_t(size_t(t) * size_t(l.out / 2)), l.out / 2,
                        zz.begin() + std::ptrdiff_t(size_t(t) * size_t(l.out) + size_t(r * l.out / 2)));
    }
    z.upload(zz);
    qtip::hadamard_rows(z.data(), l.out, rows, false, y.data(), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto out = y.download();
    for (size_t i = 0; i < out.size(); ++i)
        out[i] *= l.sv[i % size_t(l.out)];
    auto e = compare(out, y_ref);
    // The reference forward runs fp16 internals (x/32 in fp16, fp16 LUT, fp16 matmul output);
    // the runtime uses bf16 inputs/LUT with fp32 accumulation: expect ~1e-3 relative.
    check(e.rel_rms < 5e-3, "gpu full linear (TP shards 0+1) vs reference QuantizedLinear.forward: " +
                                name + " " + describe(e));
    // the runtime's activations are bf16: emulate with the bf16-rounded input
    auto dev_ref = qtip::reference_forward(l, bf16_round(x), true);
    auto e2 = compare(out, dev_ref);
    check(e2.rel_rms < 1e-4, "gpu full linear vs CPU emulation of the runtime rounding: " + name + " " +
                                 describe(e2));
}
void test_epilogues_gpu() {
    const int rows = 3;
    std::mt19937 rng(77);
    // input transform with rms norm (12288)
    {
        auto x = bf16_round(random_vector(size_t(rows) * 12288, 21, 2.f));
        auto w = bf16_round(random_vector(12288, 22, 1.f));
        std::vector<float> su(12288);
        for (auto &s : su)
            s = (rng() & 1) ? 1.f : -1.f;
        DeviceBuffer<BFloat16> dx, dw, du(size_t(rows) * 12288);
        DeviceBuffer<float> dsu;
        dx.upload(to_bf16(x));
        dw.upload(to_bf16(w));
        dsu.upload(su);
        qtip::input_transform(qtip::Source{dx.data(), 12288, 0, 12288, 0}, dw.data(), 1e-5f, dsu.data(),
                              12288, rows, du.data(), 12288, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        auto got = to_float(du.download());
        std::vector<double> ref(got.size());
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int i = 0; i < 12288; ++i)
                ss += double(x[size_t(r) * 12288 + size_t(i)]) * x[size_t(r) * 12288 + size_t(i)];
            const double f = 1 / std::sqrt(ss / 12288 + 1e-5);
            double *v = ref.data() + size_t(r) * 12288;
            for (int i = 0; i < 12288; ++i)
                v[i] = su[size_t(i)] *
                       double(float(BFloat16(float(x[size_t(r) * 12288 + size_t(i)] * f * w[size_t(i)]))));
            qtip::hadamard(v, 12288, true);
        }
        auto e = compare(got, ref);
        check(e.rel_rms < 3e-3, "gpu input transform (rms norm, SU, hadUt 12288) " + describe(e));
    }
    // qkv epilogue
    for (int rank = 0; rank < 2; ++rank) {
        auto all = random_vector(size_t(2) * rows * 7168, 30 + unsigned(rank), 20.f);
        auto svq = random_vector(12288, 31, 0.002f), svk = random_vector(1024, 32, 0.002f),
             svv = random_vector(1024, 33, 0.002f);
        DeviceBuffer<float> dall, dq_, dk_, dv_;
        dall.upload(all);
        dq_.upload(svq);
        dk_.upload(svk);
        dv_.upload(svv);
        DeviceBuffer<BFloat16> q(size_t(rows) * 6144), k(size_t(rows) * 512), v(size_t(rows) * 512);
        qtip::output_qkv(dall.data(), rows, rank, dq_.data(), dk_.data(), dv_.data(), q.data(), k.data(),
                         v.data(), nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        auto gq = to_float(q.download()), gk = to_float(k.download()), gv = to_float(v.download());
        std::vector<double> rq, rk, rv;
        for (int r = 0; r < rows; ++r) {
            std::vector<double> zq(12288), zk(1024), zv(1024);
            for (int h = 0; h < 2; ++h) {
                const float *src = all.data() + size_t(h) * rows * 7168 + size_t(r) * 7168;
                std::copy_n(src, 6144, zq.begin() + h * 6144);
                std::copy_n(src + 6144, 512, zk.begin() + h * 512);
                std::copy_n(src + 6656, 512, zv.begin() + h * 512);
            }
            qtip::hadamard(zq.data(), 12288, false);
            qtip::hadamard(zk.data(), 1024, false);
            qtip::hadamard(zv.data(), 1024, false);
            for (int i = 0; i < 6144; ++i)
                rq.push_back(zq[size_t(rank * 6144 + i)] * svq[size_t(rank * 6144 + i)]);
            for (int i = 0; i < 512; ++i) {
                rk.push_back(zk[size_t(rank * 512 + i)] * svk[size_t(rank * 512 + i)]);
                rv.push_back(zv[size_t(rank * 512 + i)] * svv[size_t(rank * 512 + i)]);
            }
        }
        auto eq = compare(gq, rq), ek = compare(gk, rk), ev = compare(gv, rv);
        check(eq.rel_rms < 3e-3 && ek.rel_rms < 3e-3 && ev.rel_rms < 3e-3,
              "gpu qkv epilogue rank " + std::to_string(rank) + " q " + describe(eq) + " | k " +
                  describe(ek) + " | v " + describe(ev));
    }
    // residual epilogue
    {
        auto all = random_vector(size_t(2) * rows * 6144, 40, 20.f);
        auto sv = random_vector(12288, 41, 0.002f);
        auto st = bf16_round(random_vector(size_t(rows) * 12288, 42, 0.5f));
        DeviceBuffer<float> dall, dsv;
        DeviceBuffer<BFloat16> dst;
        dall.upload(all);
        dsv.upload(sv);
        dst.upload(to_bf16(st));
        qtip::output_residual(dall.data(), rows, dsv.data(), dst.data(), nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        auto got = to_float(dst.download());
        std::vector<double> ref;
        for (int r = 0; r < rows; ++r) {
            std::vector<double> z(12288);
            for (int h = 0; h < 2; ++h)
                std::copy_n(all.data() + size_t(h) * rows * 6144 + size_t(r) * 6144, 6144, z.begin() + h * 6144);
            qtip::hadamard(z.data(), 12288, false);
            for (int i = 0; i < 12288; ++i)
                ref.push_back(float(BFloat16(st[size_t(r) * 12288 + size_t(i)] +
                                             float(BFloat16(float(z[size_t(i)] * sv[size_t(i)]))))));
        }
        auto e = compare(got, ref);
        check(e.rel_rms < 3e-3, "gpu o/down residual epilogue " + describe(e));
    }
    // gate/up epilogue + down input transform
    {
        auto allf = bf16_round(random_vector(size_t(2) * rows * 28672, 50, 20.f));
        auto svg = random_vector(28672, 51, 0.003f), svu = random_vector(28672, 52, 0.003f);
        std::vector<float> sud(28672);
        for (auto &s : sud)
            s = (rng() & 1) ? 1.f : -1.f;
        DeviceBuffer<BFloat16> dall, dh(size_t(rows) * 28672), du(size_t(rows) * 28672);
        DeviceBuffer<float> dsg, dsu_, dsud;
        dall.upload(to_bf16(allf));
        dsg.upload(svg);
        dsu_.upload(svu);
        dsud.upload(sud);
        qtip::output_gateup(dall.data(), rows, dsg.data(), dsu_.data(), dh.data(), nullptr);
        qtip::input_transform(qtip::Source{dh.data(), 28672, 0, 28672, 0}, nullptr, 0.f, dsud.data(), 28672,
                              rows, du.data(), 28672, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        auto gh = to_float(dh.download()), gu = to_float(du.download());
        std::vector<double> rh, ru;
        for (int r = 0; r < rows; ++r) {
            std::vector<double> zg(28672), zu(28672);
            for (int h = 0; h < 2; ++h) {
                const float *src = allf.data() + size_t(h) * rows * 28672 + size_t(r) * 28672;
                std::copy_n(src, 14336, zg.begin() + h * 14336);
                std::copy_n(src + 14336, 14336, zu.begin() + h * 14336);
            }
            qtip::hadamard(zg.data(), 28672, false);
            qtip::hadamard(zu.data(), 28672, false);
            std::vector<double> hv(28672);
            for (int i = 0; i < 28672; ++i) {
                const float g = float(BFloat16(float(zg[size_t(i)] * svg[size_t(i)])));
                const float u = float(BFloat16(float(zu[size_t(i)] * svu[size_t(i)])));
                hv[size_t(i)] = float(BFloat16(g / (1 + std::exp(-g)) * u));
                rh.push_back(hv[size_t(i)]);
                hv[size_t(i)] *= sud[size_t(i)];
            }
            qtip::hadamard(hv.data(), 28672, true);
            ru.insert(ru.end(), hv.begin(), hv.end());
        }
        auto eh = compare(gh, rh), eu = compare(gu, ru);
        check(eh.rel_rms < 3e-3 && eu.rel_rms < 3e-3,
              "gpu gate/up epilogue (hadU 28672, SV, silu) " + describe(eh) +
                  " | down input transform (SU, hadUt 28672) " + describe(eu));
    }
}
void test_real(const fs::path &layer_file, const fs::path &refvec) {
    SafeTensorFile file(layer_file);
    const std::string stem = layer_file.stem().string(); // model-layer-00040
    const int layer = std::stoi(stem.substr(stem.size() - 5));
    for (const std::string name : {"self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj"}) {
        auto l = qtip::load_linear(file, layer, name);
        auto cb = qtip::make_codebook(l.tlut);
        const auto base = refvec / ("L" + std::to_string(layer) + "_" + name);
        if (fs::exists(fs::path(base.string() + ".rows.json"))) {
            std::ifstream jf(base.string() + ".rows.json");
            auto j = nlohmann::json::parse(jf);
            size_t total = 0;
            for (auto &r : j.at("ranges"))
                total += size_t(r[1].get<int>() - r[0].get<int>()) * size_t(l.in);
            auto rows = read_f32(base.string() + ".rows.f32", total);
            std::vector<float> mine;
            for (auto &r : j.at("ranges")) {
                const int a = r[0], b = r[1];
                std::vector<float> w(size_t(b - a) * size_t(l.in));
                qtip::decode_rows(l, cb, a, b - a, w.data());
                mine.insert(mine.end(), w.begin(), w.end());
            }
            auto e = compare(mine, rows);
            check(e.identical == e.n, "cpu decoder vs python decode_compressed (layer " +
                                          std::to_string(layer) + " " + name + ") " + describe(e));
            const int n = l.in, m = l.out;
            auto x = read_f32(base.string() + ".x.f32", size_t(8) * size_t(n));
            auto y = read_f32(base.string() + ".y.f32", size_t(8) * size_t(m));
            auto yd = read_f32(base.string() + ".yd.f32", size_t(8) * size_t(m));
            auto mine_y = qtip::reference_forward(l, x);
            auto e1 = compare(mine_y, yd), e2 = compare(mine_y, y);
            check(e1.rel_rms < 1e-6, "cpu reference_forward vs python dense hatW forward: " + name + " " +
                                         describe(e1));
            check(e2.rel_rms < 5e-3, "cpu reference_forward vs python QuantizedLinear.forward: " + name +
                                         " " + describe(e2));
            test_linear_gpu(l, x, 8, y, "layer " + std::to_string(layer) + " " + name);
        } else
            std::cout << "SKIP python reference vectors for " << name << " (no " << base.string()
                      << ".rows.json)\n";
        const int half = l.out / 64;
        test_decode_gpu(l, 0, half, name + " rank0");
        test_decode_gpu(l, half, half, name + " rank1");
        test_matmul_gpu(l, half, half, name + " rank1", {1, 2, 4, 5, 33}, false);
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        test_hadamard_cpu();
        test_codebook_cpu();
        HIP_CHECK(hipSetDevice(0));
        test_hadamard_gpu();
        test_wmma_probe();
        {
            auto l = synthetic(64, 12288, 9);
            test_decode_gpu(l, 0, 2, "synthetic 64x12288");
            test_matmul_gpu(l, 0, 2, "synthetic 64x12288", {1, 2, 3, 4, 5, 16, 33, 64, 130}, true);
            auto t = synthetic(96, 224, 10); // 7 blocks along K: odd slices / tails
            test_matmul_gpu(t, 1, 2, "synthetic 64x224 (block-rows 1..2)", {1, 4, 7, 70}, true);
        }
        test_epilogues_gpu();
        if (argc == 3)
            test_real(argv[1], argv[2]);
        else if (argc != 1)
            throw std::invalid_argument("Usage: behemoth_qtip_tests [LAYER_FILE REFVEC_DIR]");
    } catch (const std::exception &e) {
        std::cout << "FAIL exception: " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "QTIP TESTS FAILED: " : "QTIP TESTS PASSED") << (failures ? std::to_string(failures) : "")
              << '\n';
    return failures ? 1 : 0;
}
