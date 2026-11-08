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
static Fp8Shard quantize_slice(const float *w, int in, int r0, int r1, int c0, int c1) {
    Fp8Shard s;
    s.rows = r1 - r0;
    s.cols = c1 - c0;
    s.w.resize(size_t(s.rows) * size_t(s.cols));
    s.scale.resize(size_t(s.rows));
#pragma omp parallel for schedule(static)
    for (int r = 0; r < s.rows; ++r) {
        const float *src = w + size_t(r0 + r) * size_t(in) + size_t(c0);
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
        shards.push_back(quantize_slice(w.data(), in, r0, r1, c0, c1));
    }
    return shards;
}

Fp8Shard quantize_rows(const float *w, int rows, int cols) { return quantize_slice(w, cols, 0, rows, 0, cols); }

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
    const std::string base = "model.layers." + std::to_string(layer + layer_offset()) + "." + name + ".";
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
        // Codes with exponent < 15 (|v| < 256) mean the same in e4m3fn and the devices' IEEE-style e4m3: slice the bytes
        // and row scales as they are (exact, no f32 pass over the matrix). Checkpoints using the e4m3fn range above 240
        // (stock FP8 exports, max 448) are requantised below.
        if (std::none_of(l.fp8.begin(), l.fp8.end(), [](uint8_t b) { return (b & 0x78) == 0x78; })) {
            if (tp < 1 || (column_parallel ? l.out : l.in) % tp)
                throw std::invalid_argument("invalid tensor-parallel shard");
            std::vector<Fp8Shard> shards(static_cast<size_t>(tp));
            for (int rank = 0; rank < tp; ++rank) {
                const int r0 = column_parallel ? rank * (l.out / tp) : 0, r1 = column_parallel ? r0 + l.out / tp : l.out;
                const int c0 = column_parallel ? 0 : rank * (l.in / tp), c1 = column_parallel ? l.in : c0 + l.in / tp;
                Fp8Shard &s = shards[size_t(rank)];
                s.rows = r1 - r0;
                s.cols = c1 - c0;
                s.w.resize(size_t(s.rows) * size_t(s.cols));
                s.scale.assign(l.fp8_scale.begin() + r0, l.fp8_scale.begin() + r1);
#pragma omp parallel for schedule(static)
                for (int r = 0; r < s.rows; ++r)
                    std::copy_n(l.fp8.data() + size_t(r0 + r) * size_t(l.in) + size_t(c0), size_t(s.cols), s.w.data() + size_t(r) * size_t(s.cols));
            }
            return shards;
        }
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

bool is_qtip_projection(const SafeTensorFile &file, int layer, const std::string &name) {
    return file.contains("model.layers." + std::to_string(layer + layer_offset()) + "." + name + ".trellis");
}

namespace {
constexpr int kInGroup = 2048;   // qtip_nki.IN_GROUP: input channels per 128-partition group (16 per partition)
constexpr int kChunkTiles = 32;  // qtip_nki.NT: output tiles per kernel chunk
}

size_t qtip_packed_bytes(int out, int in, int tp, bool column_parallel) {
    const int out_local = column_parallel ? out / tp : out, in_local = column_parallel ? in : in / tp;
    const int groups = (in_local / 16 + 127) / 128;
    return size_t(groups) * 128 * size_t(out_local / 16) * 32 * 2 + size_t(128) * 1024 * 4 + size_t(in + out) * 4;
}

QtipPacked pack_qtip_shard(const qtip::HostLinear &l, int tp, int rank, bool column_parallel) {
    if (l.is_fp8() || l.is_nvfp4())
        throw std::invalid_argument("pack_qtip_shard expects a QTIP projection");
    if (rank < 0 || rank >= tp || (column_parallel ? l.out / 16 : l.in / 16) % tp)
        throw std::invalid_argument("invalid tensor-parallel shard");
    QtipPacked p;
    p.out_local = column_parallel ? l.out / tp : l.out;
    p.in_local = column_parallel ? l.in : l.in / tp;
    p.out_tiles = p.out_local / 16;
    if (p.out_tiles % kChunkTiles)
        throw std::invalid_argument("QTIP output tiles must be a multiple of " + std::to_string(kChunkTiles));
    const int in_tiles = p.in_local / 16;
    p.groups = (in_tiles + 127) / 128;
    const int ob0 = column_parallel ? rank * p.out_tiles : 0, ib0 = column_parallel ? 0 : rank * in_tiles;
    p.words.assign(size_t(p.groups) * 128 * size_t(p.out_tiles) * 32, 0);
#pragma omp parallel for schedule(static)
    for (int ib = 0; ib < in_tiles; ++ib) {
        uint16_t *dst = p.words.data() + (size_t(ib / 128) * 128 + size_t(ib % 128)) * size_t(p.out_tiles) * 32;
        for (int ob = 0; ob < p.out_tiles; ++ob)
            for (int c = 0; c < 32; ++c)
                dst[size_t(ob) * 32 + size_t(c)] = l.word(ob0 + ob, ib0 + ib, c);
    }
    const qtip::Codebook cb = qtip::make_codebook(l.tlut);
    p.lut.resize(size_t(128) * qtip::lut_entries);
    for (int part = 0; part < 128; ++part)
        std::copy(cb.bf16x2.begin(), cb.bf16x2.end(), p.lut.begin() + size_t(part) * qtip::lut_entries);
    p.su = l.su;
    p.sv = l.sv;
    return p;
}
} // namespace neuron
