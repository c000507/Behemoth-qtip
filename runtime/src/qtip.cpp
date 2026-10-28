#include "behemoth/qtip.hpp"
#include "behemoth/attention.hpp"
#include "qtip_tables.hpp"
#include <cmath>
#include <cstring>

namespace behemoth::qtip {
int had_k(int n) {
    auto pow2 = [](long v) { return v > 0 && (v & (v - 1)) == 0; };
    // Order of the divisibility checks in get_hadK (only 12 and 28 occur for this model; the
    // larger factors 172..36 and 20 do not divide 1024, 12288 or 28672).
    for (int k : {172, 156, 140, 124, 116, 108, 60, 52, 36, 28, 20, 12})
        if (n % k == 0) {
            if ((k != 12 && k != 28) || !pow2(n / k))
                throw std::invalid_argument("Unsupported QTIP Hadamard size " + std::to_string(n));
            return k;
        }
    if (!pow2(n))
        throw std::invalid_argument("Unsupported QTIP Hadamard size " + std::to_string(n));
    return 1;
}
const int8_t *had_matrix(int k) {
    return k == 12 ? kHad12 : k == 28 ? kHad28 : nullptr;
}
void hadamard(double *x, int n, bool transpose) {
    const int k = had_k(n), p = n / k;
    const int8_t *h = had_matrix(k);
    // matmul_hadU_cuda: view (K, n/K), natural-order FWHT on each row, then hadK @ rows.
    for (int c = 0; c < k; ++c) {
        double *v = x + size_t(c) * size_t(p);
        for (int s = 1; s < p; s <<= 1)
            for (int i = 0; i < p; i += 2 * s)
                for (int j = i; j < i + s; ++j) {
                    const double a = v[j], b = v[j + s];
                    v[j] = a + b;
                    v[j + s] = a - b;
                }
    }
    const double scale = 1 / std::sqrt(double(n));
    if (k == 1) {
        for (int i = 0; i < n; ++i)
            x[i] *= scale;
        return;
    }
    std::vector<double> y(size_t(n), 0.0);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c) {
            const int sign = transpose ? h[c * k + r] : h[r * k + c];
            const double *src = x + size_t(c) * size_t(p);
            double *dst = y.data() + size_t(r) * size_t(p);
            for (int i = 0; i < p; ++i)
                dst[i] += sign * src[i];
        }
    for (int i = 0; i < n; ++i)
        x[i] = y[size_t(i)] * scale;
}
namespace {
uint16_t bf16_bits(float x) {
    return BFloat16(x).bits;
}
} // namespace
Codebook make_codebook(std::span<const uint16_t> tlut) {
    if (tlut.size() != 1024)
        throw std::invalid_argument("QTIP tlut must be F16 [512, 2]");
    Codebook cb;
    for (int e = 0; e < lut_entries; ++e) {
        const int idx = e & 511;
        const bool flip = e >> 9;
        uint16_t h0 = tlut[size_t(idx) * 2], h1 = tlut[size_t(idx) * 2 + 1];
        if (flip)
            h0 ^= 0x8000U; // exact fp16 negation (lut[:, 0] * sflp)
        cb.c0[size_t(e)] = decode_checkpoint_f16(h0);
        cb.c1[size_t(e)] = decode_checkpoint_f16(h1);
        if (!std::isfinite(cb.c0[size_t(e)]) || !std::isfinite(cb.c1[size_t(e)]))
            throw std::invalid_argument("Nonfinite QTIP tlut entry");
        cb.bf16x2[size_t(e)] = uint32_t(bf16_bits(cb.c0[size_t(e)])) |
                               uint32_t(bf16_bits(cb.c1[size_t(e)])) << 16;
    }
    return cb;
}
namespace {
// Decode one 16x16 tile into out[16][16] using component tables c0/c1.
void decode_tile(const HostLinear &l, int tr, int tc, const float *c0, const float *c1,
                 float (*out)[16]) {
    uint16_t w[32];
    for (int c = 0; c < 32; ++c)
        w[c] = l.word(tr, tc, c);
    for (int t = 0; t < 128; ++t) {
        // 16-bit window starting at bit 4t of the 512-bit tail-biting stream (MSB first).
        const int i = t >> 2, s = 4 * (t & 3);
        const uint32_t u = uint32_t(w[i]) << 16 | w[(i + 1) & 31];
        const int e = lut_entry(u >> (16 - s));
        // m16n8k16 A-fragment deswizzle of decode_compressed: t = A*16 + B*4 + C*2 + D,
        // row = D*8 + A, columns C*8 + B*2 + {0, 1}.
        const int A = t >> 4, B = (t >> 2) & 3, C = (t >> 1) & 1, D = t & 1;
        const int r = D * 8 + A, col = C * 8 + B * 2;
        out[r][col] = c0[e];
        out[r][col + 1] = c1[e];
    }
}
void decode_rows_impl(const HostLinear &l, const float *c0, const float *c1, int row0, int rows,
                      float *wout) {
    if (row0 % 16 || rows % 16 || row0 < 0 || row0 + rows > l.out || l.in % 32 || l.out % 32)
        throw std::invalid_argument("QTIP decode rows must be whole 16-row tiles");
    const int tiles_c = l.in / 16;
#pragma omp parallel for schedule(static)
    for (int tr = row0 / 16; tr < (row0 + rows) / 16; ++tr) {
        float t[16][16];
        for (int tc = 0; tc < tiles_c; ++tc) {
            decode_tile(l, tr, tc, c0, c1, t);
            for (int r = 0; r < 16; ++r)
                std::memcpy(wout + (size_t(tr * 16 - row0 + r) * size_t(l.in) + size_t(tc * 16)),
                            t[r], 16 * sizeof(float));
        }
    }
}
} // namespace
void decode_rows(const HostLinear &l, const Codebook &cb, int row0, int rows, float *w) {
    decode_rows_impl(l, cb.c0.data(), cb.c1.data(), row0, rows, w);
}
void decode_rows_bf16(const HostLinear &l, const Codebook &cb, int row0, int rows, float *w) {
    std::array<float, lut_entries> c0{}, c1{};
    for (int e = 0; e < lut_entries; ++e) {
        c0[size_t(e)] = std::bit_cast<float>((cb.bf16x2[size_t(e)] & 0xffffU) << 16);
        c1[size_t(e)] = std::bit_cast<float>(cb.bf16x2[size_t(e)] & 0xffff0000U);
    }
    decode_rows_impl(l, c0.data(), c1.data(), row0, rows, w);
}
std::vector<float> reference_forward(const HostLinear &l, std::span<const float> x,
                                     bool device_rounding, bool round_z_bf16) {
    if (x.size() % size_t(l.in))
        throw std::invalid_argument("QTIP reference input shape");
    const size_t rows = x.size() / size_t(l.in);
    if (l.is_fp8()) {
        if (l.fp8.size() != size_t(l.in) * size_t(l.out) || l.fp8_scale.size() != size_t(l.out))
            throw std::invalid_argument("Invalid FP8 reference weight");
        std::vector<float> y(rows * size_t(l.out));
        for (size_t r = 0; r < rows; ++r)
            for (int o = 0; o < l.out; ++o) {
                double sum = 0;
                for (int i = 0; i < l.in; ++i) {
                    float v = x[r * size_t(l.in) + size_t(i)];
                    if (device_rounding) v = float(BFloat16(v));
                    sum += double(v) * decode_e4m3(l.fp8[size_t(o) * size_t(l.in) + size_t(i)]);
                }
                y[r * size_t(l.out) + size_t(o)] = float(sum) * l.fp8_scale[size_t(o)];
            }
        return y;
    }
    const Codebook cb = make_codebook(l.tlut);
    std::vector<double> u(rows * size_t(l.in));
    for (size_t r = 0; r < rows; ++r) {
        double *v = u.data() + r * size_t(l.in);
        for (int i = 0; i < l.in; ++i)
            v[i] = double(x[r * size_t(l.in) + size_t(i)]) * l.su[size_t(i)];
        hadamard(v, l.in, true);
        if (device_rounding)
            for (int i = 0; i < l.in; ++i)
                v[i] = double(float(BFloat16(float(v[i]))));
    }
    std::vector<double> z(rows * size_t(l.out), 0.0);
    constexpr int block = 256;
    std::vector<float> w(size_t(block) * size_t(l.in));
    for (int r0 = 0; r0 < l.out; r0 += block) {
        const int nr = std::min(block, l.out - r0);
        if (device_rounding)
            decode_rows_bf16(l, cb, r0, nr, w.data());
        else
            decode_rows(l, cb, r0, nr, w.data());
#pragma omp parallel for schedule(static)
        for (int o = 0; o < nr; ++o)
            for (size_t r = 0; r < rows; ++r) {
                double s = 0;
                const float *wr = w.data() + size_t(o) * size_t(l.in);
                const double *ur = u.data() + r * size_t(l.in);
                for (int i = 0; i < l.in; ++i)
                    s += double(wr[i]) * ur[i];
                z[r * size_t(l.out) + size_t(r0 + o)] = s;
            }
    }
    std::vector<float> y(rows * size_t(l.out));
    for (size_t r = 0; r < rows; ++r) {
        double *v = z.data() + r * size_t(l.out);
        if (round_z_bf16)
            for (int o = 0; o < l.out; ++o)
                v[o] = double(float(BFloat16(float(v[o]))));
        hadamard(v, l.out, false);
        for (int o = 0; o < l.out; ++o)
            y[r * size_t(l.out) + size_t(o)] = float(v[o] * l.sv[size_t(o)]);
    }
    return y;
}
HostLinear load_linear(const Checkpoint &checkpoint, int layer, const std::string &name) {
    if (layer < 0 || layer >= checkpoint.config.layers)
        throw std::invalid_argument("Invalid QTIP layer");
    const auto base = "model.layers." + std::to_string(layer) + "." + name + ".";
    const bool fp8 = checkpoint.config.fp8_kv && (name == "self_attn.k_proj" || name == "self_attn.v_proj");
    if (fp8) {
        for (const auto *suffix : {"trellis", "SU", "SV", "tlut", "rcp", "tp_rank"})
            if (checkpoint.has_tensor(base + suffix))
                throw std::invalid_argument("Mixed checkpoint has both FP8 and QTIP K/V: " + base);
        auto w = load_fp8_linear(checkpoint.tensor(base + "weight"), checkpoint.tensor(base + "weight_scale"), layer, name);
        if (w.in != 12288 || w.out != 1024)
            throw std::invalid_argument("Invalid FP8 K/V geometry: " + base);
        return w;
    }
    if (checkpoint.has_tensor(base + "weight"))
        throw std::invalid_argument("FP8 projection requires qtip_quant.kv_format=fp8: " + base);
    return load_linear(
        checkpoint.tensor("model.layers." + std::to_string(layer) + "." + name + ".trellis"), layer,
        name);
}
HostLinear load_linear(const SafeTensorFile &file, int layer, const std::string &name) {
    const auto prefix = "model.layers." + std::to_string(layer) + "." + name + ".";
    if (file.contains(prefix + "weight"))
        return load_fp8_linear(file, layer, name);
    int in = 12288, out = 12288;
    if (name == "self_attn.k_proj" || name == "self_attn.v_proj")
        out = 1024;
    else if (name == "mlp.gate_proj" || name == "mlp.up_proj")
        out = 28672;
    else if (name == "mlp.down_proj")
        in = 28672;
    else if (name != "self_attn.q_proj" && name != "self_attn.o_proj")
        throw std::invalid_argument("Unknown projection " + name);
    const std::string base = "model.layers." + std::to_string(layer) + "." + name + ".";
    HostLinear l;
    l.in = in;
    l.out = out;
    {
        const auto &t = file;
        if (t.dtype(base + "trellis") != "I16" ||
            t.shape(base + "trellis") !=
                std::vector<size_t>{size_t(out / 16) * size_t(in / 16), size_t(words_per_tile)})
            throw std::invalid_argument("Invalid QTIP trellis shape for " + base);
        auto raw = t.bytes(base + "trellis");
        l.trellis.resize(raw.size() / 2);
        std::memcpy(l.trellis.data(), raw.data(), raw.size());
    }
    auto floats = [&](const std::string &n, size_t size) {
        auto v = file.floats(n);
        if (v.size() != size)
            throw std::invalid_argument("Invalid QTIP vector " + n);
        std::vector<float> r(size);
        for (size_t i = 0; i < size; ++i) {
            r[i] = v[i];
            if (!std::isfinite(r[i]))
                throw std::invalid_argument("Nonfinite QTIP vector " + n);
        }
        return r;
    };
    l.su = floats(base + "SU", size_t(in));
    l.sv = floats(base + "SV", size_t(out));
    {
        const auto &t = file;
        if (t.dtype(base + "tlut") != "F16" ||
            t.shape(base + "tlut") != std::vector<size_t>{512, 2})
            throw std::invalid_argument("Invalid QTIP tlut for " + base);
        auto raw = t.bytes(base + "tlut");
        l.tlut.resize(1024);
        std::memcpy(l.tlut.data(), raw.data(), raw.size());
    }
    for (const char *scalar : {"rcp", "tp_rank"}) {
        const std::string n = base + scalar;
        const auto &t = file;
        if (t.dtype(n) != "I64")
            throw std::invalid_argument("Invalid QTIP " + n);
        int64_t v = 0;
        std::memcpy(&v, t.bytes(n).data(), 8);
        if (std::string(scalar) == "rcp" && v != 0)
            throw std::invalid_argument("QTIP rcp != 0 (TP-split checkpoints are not supported)");
    }
    return l;
}
HostLinear load_fp8_linear(const SafeTensorFile &file, int layer, const std::string &name) {
    return load_fp8_linear(file, file, layer, name);
}
HostLinear load_fp8_linear(const SafeTensorFile &file, const SafeTensorFile &scale_file, int layer, const std::string &name) {
    const std::string base = "model.layers." + std::to_string(layer) + "." + name + ".";
    auto shape = file.shape(base + "weight");
    if (file.dtype(base + "weight") != "F8_E4M3" || shape.size() != 2 || !shape[0] || !shape[1] ||
        shape[0] > size_t(INT32_MAX) || shape[1] > size_t(INT32_MAX) ||
        file.contains(base + "trellis") || file.contains(base + "SU") || file.contains(base + "SV") ||
        file.contains(base + "tlut") || file.contains(base + "rcp") || file.contains(base + "tp_rank"))
        throw std::invalid_argument("Invalid or ambiguous FP8 projection: " + base);
    HostLinear w;
    w.out = int(shape[0]);
    w.in = int(shape[1]);
    auto raw = file.bytes(base + "weight");
    w.fp8.resize(raw.size());
    std::memcpy(w.fp8.data(), raw.data(), raw.size());
    const auto dtype = scale_file.dtype(base + "weight_scale");
    if (dtype != "BF16" && dtype != "F32")
        throw std::invalid_argument("FP8 scales must be BF16 or FP32: " + base);
    auto scales = scale_file.floats(base + "weight_scale");
    if (scales.size() != size_t(w.out))
        throw std::invalid_argument("Invalid FP8 per-output scale: " + base);
    w.fp8_scale.resize(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        w.fp8_scale[i] = scales[i];
        if (!std::isfinite(scales[i]) || scales[i] < 0)
            throw std::invalid_argument("Nonfinite or negative FP8 scale: " + base);
    }
    for (uint8_t x : w.fp8)
        if (!std::isfinite(decode_e4m3(x)))
            throw std::invalid_argument("Nonfinite FP8 weight: " + base);
    return w;
}
namespace {
std::vector<BFloat16> norm_ref(std::span<const BFloat16> x, const std::vector<BFloat16> &weights) {
    double squares = 0;
    for (auto v : x)
        squares += double(float(v)) * float(v);
    double factor = 1 / std::sqrt(squares / double(x.size()) + 1e-5);
    std::vector<BFloat16> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = BFloat16(float(double(float(x[i])) * factor * float(weights[i])));
    return out;
}
std::vector<float> to_floats(std::span<const BFloat16> x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = float(x[i]);
    return out;
}
std::vector<BFloat16> to_bf16(const std::vector<float> &x) {
    std::vector<BFloat16> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = BFloat16(x[i]);
    return out;
}
} // namespace
std::vector<BFloat16> reference_block(const Checkpoint &checkpoint, int layer,
                                      std::span<const BFloat16> input) {
    if (input.size() != 12288 || layer < 0 || layer >= checkpoint.config.layers)
        throw std::invalid_argument("QTIP block reference accepts one 12288-wide token");
    std::vector<BFloat16> residual(input.begin(), input.end());
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    auto x = to_floats(norm_ref(input, checkpoint.bf16(prefix + "input_layernorm.weight")));
    auto proj = [&](const char *name, std::span<const float> v, bool round_z = false) {
        return to_bf16(reference_forward(load_linear(checkpoint, layer, name), v, true, round_z));
    };
    auto q = proj("self_attn.q_proj", x), k = proj("self_attn.k_proj", x),
         v = proj("self_attn.v_proj", x);
    std::vector<BFloat16> attention(12288);
    for (int rank = 0; rank < 2; ++rank) {
        // Same per-rank FP8 attention reference as the AQLM block reference (position zero: RoPE
        // is the identity); rank r owns q heads 48r.. and kv heads 4r..
        AttentionLayout l;
        l.physical_pages = 1;
        l.table_stride = 1;
        l.query_offsets = {0, 1};
        l.positions = {0};
        l.lengths = {1};
        l.page_table = {0};
        AttentionData a;
        a.q.resize(48 * 128);
        a.q_scale.resize(48);
        a.k.assign(4 * 16 * 128, 0);
        a.v = a.k;
        a.k_scale.assign(4 * 16, 1);
        a.v_scale = a.k_scale;
        for (int h = 0; h < 48; ++h) {
            auto fq = to_floats(std::span(q).subspan(size_t(rank * 48 + h) * 128, 128));
            auto quantized = quantize(fq);
            a.q_scale[size_t(h)] = quantized.scale;
            std::copy(quantized.values.begin(), quantized.values.end(), a.q.begin() + h * 128);
        }
        std::vector<BFloat16> kr(k.begin() + rank * 512, k.begin() + rank * 512 + 512),
            vr(v.begin() + rank * 512, v.begin() + rank * 512 + 512);
        reference_append(l, a, kr, vr);
        auto out = to_bf16(reference_attention(l, a, true).output);
        std::copy(out.begin(), out.end(), attention.begin() + rank * 6144);
    }
    auto add = [&](const std::vector<BFloat16> &y) {
        for (size_t i = 0; i < residual.size(); ++i)
            residual[i] = BFloat16(float(residual[i]) + float(y[i]));
    };
    add(proj("self_attn.o_proj", to_floats(attention)));
    x = to_floats(norm_ref(residual, checkpoint.bf16(prefix + "post_attention_layernorm.weight")));
    auto gate = proj("mlp.gate_proj", x, true), up = proj("mlp.up_proj", x, true);
    std::vector<float> h(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        const float g = float(gate[i]);
        h[i] = float(BFloat16(g / (1 + std::exp(-g)) * float(up[i])));
    }
    add(proj("mlp.down_proj", h));
    return residual;
}
} // namespace behemoth::qtip
