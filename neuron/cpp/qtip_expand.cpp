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

// Quantize rows [r0, r1) x columns [c0, c1) of a dense [out][in] matrix to row-scaled E4M3.
static Fp8Shard quantize_slice(const std::vector<float> &w, int in, int r0, int r1, int c0, int c1) {
    Fp8Shard s;
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

static std::vector<Fp8Shard> slices(const std::vector<float> &w, int out, int in, int tp, bool column_parallel) {
    if (tp < 1 || (column_parallel ? out : in) % tp)
        throw std::invalid_argument("invalid tensor-parallel shard");
    std::vector<Fp8Shard> shards;
    for (int rank = 0; rank < tp; ++rank) {
        const int r0 = column_parallel ? rank * (out / tp) : 0, r1 = column_parallel ? r0 + out / tp : out;
        const int c0 = column_parallel ? 0 : rank * (in / tp), c1 = column_parallel ? in : c0 + in / tp;
        shards.push_back(quantize_slice(w, in, r0, r1, c0, c1));
    }
    return shards;
}

static std::vector<float> effective_qtip(const qtip::HostLinear &l) {
    const int out = l.out, in = l.in;
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
    return w;
}

std::vector<Fp8Shard> expand_qtip_all(const qtip::HostLinear &l, int tp, bool column_parallel) {
    if (l.is_fp8() || l.is_nvfp4())
        throw std::invalid_argument("expand_qtip_all expects a QTIP projection");
    return slices(effective_qtip(l), l.out, l.in, tp, column_parallel);
}

Fp8Shard expand_qtip_to_fp8(const qtip::HostLinear &l, int tp, int rank, bool column_parallel) {
    if (rank < 0 || rank >= tp)
        throw std::invalid_argument("invalid tensor-parallel shard");
    return expand_qtip_all(l, tp, column_parallel)[size_t(rank)];
}

std::vector<Fp8Shard> load_projection_shards(const SafeTensorFile &file, int layer, const std::string &name,
                                             const ModelConfig &config, int tp, bool column_parallel) {
    const std::string base = "model.layers." + std::to_string(layer) + "." + name + ".";
    if (file.contains(base + "trellis"))
        return expand_qtip_all(qtip::load_linear(file, layer, name, &config), tp, column_parallel);
    if (file.contains(base + "weight_packed")) {
        const auto l = qtip::load_nvfp4_linear(file, layer, name);
        static const float e2m1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
        std::vector<float> w(size_t(l.out) * size_t(l.in));
#pragma omp parallel for schedule(static)
        for (int o = 0; o < l.out; ++o)
            for (int i = 0; i < l.in; ++i) {
                const uint8_t b = l.nvfp4_packed[size_t(o) * size_t(l.in / 2) + size_t(i / 2)];
                const float code = e2m1[(i & 1) ? (b >> 4) : (b & 15)];
                const float scale = decode_e4m3(l.nvfp4_scales[size_t(o) * size_t(l.in / 16) + size_t(i / 16)]);
                w[size_t(o) * size_t(l.in) + size_t(i)] = code * scale * l.nvfp4_global;
            }
        return slices(w, l.out, l.in, tp, column_parallel);
    }
    if (file.contains(base + "weight")) {
        const auto l = qtip::load_fp8_linear(file, layer, name);
        std::vector<float> w(size_t(l.out) * size_t(l.in));
#pragma omp parallel for schedule(static)
        for (int o = 0; o < l.out; ++o)
            for (int i = 0; i < l.in; ++i)
                w[size_t(o) * size_t(l.in) + size_t(i)] =
                    decode_e4m3(l.fp8[size_t(o) * size_t(l.in) + size_t(i)]) * l.fp8_scale[size_t(o)];
        return slices(w, l.out, l.in, tp, column_parallel);
    }
    throw std::invalid_argument("projection not found in layer file: " + base);
}
} // namespace neuron
