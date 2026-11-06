#include "qtip_expand.hpp"
#include "behemoth/numeric.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuron {
using namespace behemoth;

namespace {
constexpr float kFp8Max = 224.0f; // Trn1/Trn2 only support IEEE-style E4M3 (exponent 15 = inf/NaN): stay below 240
}

Fp8Shard expand_qtip_to_fp8(const qtip::HostLinear &l, int tp, int rank, bool column_parallel) {
    if (l.is_fp8() || l.is_nvfp4())
        throw std::invalid_argument("expand_qtip_to_fp8 expects a QTIP projection");
    const int out = l.out, in = l.in;
    if (tp < 1 || rank < 0 || rank >= tp || (column_parallel ? out : in) % tp)
        throw std::invalid_argument("invalid tensor-parallel shard");
    const qtip::Codebook cb = qtip::make_codebook(l.tlut);

    std::vector<float> w(size_t(out) * size_t(in));
    qtip::decode_rows_bf16(l, cb, 0, out, w.data());

    // W~ . M_in . diag(SU): row o of W~ M_in is M_in^T applied to the row, i.e. hadamard with the opposite flag.
#pragma omp parallel
    {
        std::vector<double> v(static_cast<size_t>(in));
#pragma omp for schedule(static)
        for (int o = 0; o < out; ++o) {
            float *row = w.data() + size_t(o) * size_t(in);
            for (int i = 0; i < in; ++i) v[size_t(i)] = double(row[i]);
            qtip::hadamard(v.data(), in, false);
            for (int i = 0; i < in; ++i) row[i] = float(v[size_t(i)] * double(l.su[size_t(i)]));
        }
    }
    // diag(SV) . M_out . (...): M_out acts on every column.
#pragma omp parallel
    {
        std::vector<double> v(static_cast<size_t>(out));
#pragma omp for schedule(static)
        for (int c = 0; c < in; ++c) {
            for (int o = 0; o < out; ++o) v[size_t(o)] = double(w[size_t(o) * size_t(in) + size_t(c)]);
            qtip::hadamard(v.data(), out, false);
            for (int o = 0; o < out; ++o) w[size_t(o) * size_t(in) + size_t(c)] = float(v[size_t(o)] * double(l.sv[size_t(o)]));
        }
    }

    Fp8Shard s;
    const int r0 = column_parallel ? rank * (out / tp) : 0, r1 = column_parallel ? r0 + out / tp : out;
    const int c0 = column_parallel ? 0 : rank * (in / tp), c1 = column_parallel ? in : c0 + in / tp;
    s.rows = r1 - r0;
    s.cols = c1 - c0;
    s.w.resize(size_t(s.rows) * size_t(s.cols));
    s.scale.resize(size_t(s.rows));
#pragma omp parallel for schedule(static)
    for (int r = 0; r < s.rows; ++r) {
        const float *src = w.data() + size_t(r0 + r) * size_t(in) + size_t(c0);
        float amax = 0;
        for (int c = 0; c < s.cols; ++c) amax = std::max(amax, std::fabs(src[c]));
        const float scale = amax > 0 ? amax / kFp8Max : 1.0f;
        s.scale[size_t(r)] = scale;
        for (int c = 0; c < s.cols; ++c) s.w[size_t(r) * size_t(s.cols) + size_t(c)] = encode_e4m3(src[c] / scale);
    }
    return s;
}
} // namespace neuron
