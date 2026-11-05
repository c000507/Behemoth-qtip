#include "batch_kernels.hpp"
#include "behemoth/qtip.hpp"
#if defined(BEHEMOTH_CUDA_FAST)
#include "fast_prefill.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
using namespace behemoth;
namespace cg = glm53::gpu;
namespace cu = behemoth::cuda_runtime;
namespace {
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class T> void put(cg::Buffer<T> &d, const std::vector<T> &v, cg::Stream s) {
    d.ensure(v.size(), s);
    cg::checked(cg::h2d(d, v.data(), v.size() * sizeof(T), s), "batch test upload");
}
std::vector<uint16_t> bits(const std::vector<float> &x) {
    std::vector<uint16_t> result;
    for (float v : x)
        result.push_back(BFloat16(v).bits);
    return result;
}
std::vector<float> read(const float *x, size_t n, cg::Stream s) {
    cg::Pinned<float> host;
    host.ensure(n);
    cg::checked(cg::d2h(host.p, x, n * 4, s), "batch test read");
    cg::checked(cg::sync(s), "batch test completion");
    return {host.p, host.p + n};
}
std::vector<float> read(const uint16_t *x, size_t n, cg::Stream s) {
    cg::Buffer<float> converted;
    converted.ensure(n, s);
    cg::to_float(x, converted, n, s);
    return read(converted.p, n, s);
}
void compare(const std::vector<float> &a, const std::vector<float> &b, const char *name,
             double tolerance = .015, double absolute = .03) {
    require(a.size() == b.size(), "Batch output size mismatch");
    double e = 0, n = 0, maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]), "Nonfinite batch output");
        e += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        n += double(b[i]) * b[i];
        maximum = std::max(maximum, std::abs(double(a[i]) - b[i]));
    }
    double relative = std::sqrt(e / std::max(n, 1e-30));
    std::cout << name << " relative L2 " << relative << " max " << maximum << '\n';
    require(relative < tolerance && maximum < absolute, "Batch numerical reference mismatch");
}
qtip::HostLinear matrix(int out, int in) {
    qtip::HostLinear h;
    h.in = in;
    h.out = out;
    std::mt19937 gen(91);
    h.trellis.resize(size_t(in) * out / 8);
    h.tlut.resize(1024);
    h.su.resize(in);
    h.sv.resize(out);
    for (auto &v : h.trellis)
        v = uint16_t(gen());
    for (auto &v : h.tlut)
        v = uint16_t((gen() & 0x8000) | (0x3000 + gen() % 0x1000));
    for (auto &v : h.su)
        v = gen() % 2 ? 1.f : -1.f;
    for (auto &v : h.sv)
        v = (gen() % 2 ? 1.f : -1.f) * .003f;
    return h;
}
void qtip_test(cg::Stream s, int out, int in, int tokens) {
    auto h = matrix(out, in);
    auto cb = qtip::make_codebook(h.tlut);
    std::mt19937 gen(12);
    std::normal_distribution<float> normal(0, .1f);
    std::vector<float> x(size_t(in) * tokens);
    for (auto &v : x)
        v = float(BFloat16(normal(gen)));
    cg::Buffer<uint16_t> input, trellis, rotated, output, decoded;
    cg::Buffer<uint32_t> lut;
    cg::Buffer<float> su, sv, work, z;
    put(input, bits(x), s);
    put(trellis, h.trellis, s);
    put(su, h.su, s);
    put(sv, h.sv, s);
    std::vector<uint32_t> table(cb.bf16x2.begin(), cb.bf16x2.end());
    put(lut, table, s);
    rotated.ensure(size_t(in) * tokens, s);
    output.ensure(size_t(out) * tokens, s);
    work.ensure(size_t(std::max(in, out)) * tokens, s);
    z.ensure(size_t(out) * tokens * 8, s);
    cu::QtipMatrix w{trellis, su, sv, lut, in, out};
    if (tokens == 8) {
        std::vector<float> expected(size_t(in) * out);
        qtip::decode_rows_bf16(h, cb, 0, out, expected.data());
        decoded.ensure(expected.size(), s);
        cu::qtip_decode_matrix(w, decoded, s);
        auto actual = read(decoded.p, expected.size(), s);
        require(actual == expected, "Dense BF16 QTIP reconstruction must match the CPU decoder exactly");
    }
    for (bool rz : {false, true}) {
        std::vector<float> expected;
        for (int t = 0; t < tokens; ++t) {
            auto row =
                qtip::reference_forward(h, std::span<const float>(x.data() + size_t(t) * in, in), true, rz);
            for (float v : row)
                expected.push_back(float(BFloat16(v)));
        }
        for (int splits : {1, 3, 8}) {
            if (splits > in / 32)
                continue;
            cu::qtip_prepare(w, input, work, rotated, s, tokens);
            if (tokens == 65 && splits == 1) {
                auto batched = read(rotated.p, size_t(tokens) * in, s);
                for (int t = 0; t < tokens; ++t)
                    cu::qtip_prepare(w, input.p + size_t(t) * in, work, rotated.p + size_t(t) * in, s);
                require(batched == read(rotated.p, size_t(tokens) * in, s),
                        "Fused input Hadamard must match per-token transform exactly");
            }
            cu::qtip_multiply_batch(w, rotated, z, out, splits, tokens, s, 0);
            auto baseline_products = read(z.p, size_t(tokens) * out * splits, s);
            cu::qtip_multiply_batch(w, rotated, z, out, splits, tokens, s, 1);
            if (tokens > 1)
                require(baseline_products == read(z.p, size_t(tokens) * out * splits, s),
                        "Wide register QTIP must preserve tensor-core products exactly");
            cu::qtip_finish(w, z, output, work, rz, splits, s, tokens);
            if (tokens == 65 && splits == 1) {
                auto batched = read(output.p, size_t(tokens) * out, s);
                for (int t = 0; t < tokens; ++t)
                    cu::qtip_finish(w, z.p + size_t(t) * out, output.p + size_t(t) * out, work, rz, 1, s);
                require(batched == read(output.p, size_t(tokens) * out, s),
                        "Fused output Hadamard must match per-token transform exactly");
            }
            std::cout << "QTIP " << out << 'x' << in << " tokens=" << tokens << " splits=" << splits
                      << " round_z=" << rz << ' ';
            compare(read(output.p, expected.size(), s), expected, "batch RHT/tensor product");
            if (splits == 3) {
                auto original = bits(x);
                std::vector<uint16_t> packed_input(original.size());
                for (int rank = 0; rank < 2; ++rank)
                    for (int t = 0; t < tokens; ++t)
                        for (int c = 0; c < in / 2; ++c)
                            packed_input[(size_t(rank) * tokens + t) * (in / 2) + c] =
                                original[size_t(t) * in + rank * (in / 2) + c];
                cg::Buffer<uint16_t> ranked_input;
                put(ranked_input, packed_input, s);
                cu::qtip_prepare(w, ranked_input, work, rotated, s, tokens, true);
                cu::qtip_multiply_batch(w, rotated, z, out, splits, tokens, s);
                auto normal = read(z.p, size_t(tokens) * out * splits, s);
                std::vector<float> ranked(normal.size());
                for (int part = 0; part < splits; ++part)
                    for (int rank = 0; rank < 2; ++rank)
                        for (int t = 0; t < tokens; ++t)
                            for (int c = 0; c < out / 2; ++c)
                                ranked[((size_t(part) * 2 + rank) * tokens + t) * (out / 2) + c] =
                                    normal[(size_t(part) * tokens + t) * out + rank * (out / 2) + c];
                put(z, ranked, s);
                cu::qtip_finish(w, z, output, work, rz, splits, s, tokens, true);
                compare(read(output.p, expected.size(), s), expected, "Rank-major RHT input/output");
                if (rz) {
                    std::vector<uint16_t> compact(size_t(tokens) * out);
                    for (int rank = 0; rank < 2; ++rank)
                        for (int t = 0; t < tokens; ++t)
                            for (int c = 0; c < out / 2; ++c) {
                                size_t at = size_t(t) * out + rank * (out / 2) + c;
                                float sum = normal[at];
                                for (int part = 1; part < splits; ++part)
                                    sum += normal[size_t(part) * tokens * out + at];
                                compact[(size_t(rank) * tokens + t) * (out / 2) + c] = BFloat16(sum).bits;
                            }
                    put(output, compact, s);
                    cu::qtip_finish_b16(w, output, output, work, tokens, s, true);
                    compare(read(output.p, expected.size(), s), expected,
                            "Aliased BF16 rank-major output RHT");
                }
            }
        }
    }
}
void dense_test(cg::Stream s, int tokens) {
    constexpr int rows = 128, in = 96;
    std::mt19937 gen(17);
    std::normal_distribution<float> normal(0, .1f);
    std::vector<float> weight(rows * in), x(tokens * in), expected(rows * tokens);
    for (auto *v : {&weight, &x})
        for (auto &a : *v)
            a = float(BFloat16(normal(gen)));
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            double sum = 0;
            for (int c = 0; c < in; ++c)
                sum += double(weight[r * in + c]) * x[t * in + c];
            expected[t * rows + r] = float(sum);
        }
    cg::Buffer<uint16_t> w, input;
    cg::Buffer<float> output;
    put(w, bits(weight), s);
    put(input, bits(x), s);
    output.ensure(expected.size(), s);
    cu::bf16_multiply_batch(w, input, output, in, rows, tokens, s);
    compare(read(output.p, expected.size(), s), expected, "BF16 tensor product", 1e-4, 1e-3);
    qtip::HostLinear h;
    h.in = in;
    h.out = rows;
    h.fp8.resize(rows * in);
    h.fp8_scale.resize(rows);
    for (auto &v : h.fp8) {
        v = uint8_t(gen());
        if ((v & 127) == 127)
            v = 0;
    }
    for (auto &v : h.fp8_scale)
        v = float(1 + gen() % 7) / 4096;
    cg::Buffer<uint8_t> low;
    cg::Buffer<float> scale;
    cg::Buffer<uint16_t> y;
    put(low, h.fp8, s);
    put(scale, h.fp8_scale, s);
    y.ensure(expected.size(), s);
    expected.clear();
    for (int t = 0; t < tokens; ++t) {
        auto row = qtip::reference_forward(h, std::span<const float>(x.data() + t * in, in), true);
        for (float a : row)
            expected.push_back(float(BFloat16(a)));
    }
    cu::fp8_multiply_batch(low, scale, input, y, in, rows, tokens, s);
    compare(read(y.p, expected.size(), s), expected, "FP8 batched post-accumulation scale");
    cg::Buffer<float> partials;
    partials.ensure(size_t(rows) * tokens * std::min(32, in / 32), s);
    cu::fp8_multiply_batch(low, scale, input, y, in, rows, tokens, s, partials);
    compare(read(y.p, expected.size(), s), expected, "FP8 split-K post-accumulation scale");
}
void norm_test(cg::Stream s, int batch) {
    constexpr int width = 12288;
    constexpr float epsilon = 2e-5f;
    std::mt19937 gen(819);
    std::normal_distribution<float> normal;
    std::vector<float> x(size_t(batch) * width), weights(width), expected(x.size());
    for (auto &v : x)
        v = float(BFloat16(normal(gen)));
    std::fill(x.begin(), x.begin() + width, 0.f);
    for (auto &v : weights)
        v = 1.f + normal(gen) * .03f;
    for (int b = 0; b < batch; ++b) {
        double squares = 0;
        for (int i = 0; i < width; ++i)
            squares += double(x[b * width + i]) * x[b * width + i];
        double inv = 1 / std::sqrt(squares / width + epsilon);
        for (int i = 0; i < width; ++i)
            expected[b * width + i] = float(BFloat16(float(x[b * width + i] * inv * weights[i])));
    }
    cg::Buffer<uint16_t> input, output;
    cg::Buffer<float> scale;
    put(input, bits(x), s);
    put(scale, weights, s);
    output.ensure(x.size(), s);
    cu::norm_batch(input, scale, output, batch, width, epsilon, s, false);
    auto baseline = read(output.p, x.size(), s);
    cu::norm_batch(input, scale, output, batch, width, epsilon, s, true);
    require(baseline == read(output.p, x.size(), s),
            "Resident RMS must match the original reduction exactly");
    cu::norm_batch(input, scale, input, batch, width, epsilon, s, true);
    require(baseline == read(input.p, x.size(), s), "Resident RMS must support in-place output");
    compare(baseline, expected, "RMS normalization and epsilon", .003, .02);
}
void attention_test(cg::Stream s, int context = 17, int qh = 6, int kh = 2) {
    constexpr int batch = 3, dim = 128;
    std::vector<int> positions = context == 17 ? std::vector<int>{0, 3, 7} : std::vector<int>{128, 137, 256};
    if (context > 257)
        positions = {context / 2, context - 129, context - 1};
    std::mt19937 gen(64);
    std::normal_distribution<float> normal(0, .2f);
    std::vector<float> q(batch * qh * dim), k(batch * context * kh * dim), v(k.size());
    std::vector<float> next_k(batch * kh * dim), next_v(next_k.size());
    for (auto *vec : {&q, &k, &v, &next_k, &next_v})
        for (auto &a : *vec)
            a = float(BFloat16(normal(gen)));
    cg::Buffer<uint16_t> dq, dk, dv, nk, nv, out;
    cg::Buffer<int> pos;
    cg::Buffer<float> work;
    put(dq, bits(q), s);
    put(dk, bits(k), s);
    put(dv, bits(v), s);
    put(nk, bits(next_k), s);
    put(nv, bits(next_v), s);
    put(pos, positions, s);
    out.ensure(q.size(), s);
    work.ensure(size_t(batch) * qh * context, s);
    cu::cache_batch(nk, nv, dk, dv, pos, batch, 1, context, kh * dim, s);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < kh * dim; ++i) {
            k[(b * context + positions[b]) * kh * dim + i] = next_k[b * kh * dim + i];
            v[(b * context + positions[b]) * kh * dim + i] = next_v[b * kh * dim + i];
        }
    require(read(dk.p, k.size(), s) == k, "Batched key-cache append changed another sequence/position");
    require(read(dv.p, v.size(), s) == v, "Batched value-cache append changed another sequence/position");
    std::vector<float> expected(q.size());
    for (int b = 0; b < batch; ++b)
        for (int h = 0; h < qh; ++h) {
            int n = positions[b] + 1, head = h / (qh / kh);
            std::vector<double> p(n);
            double top = -INFINITY, total = 0;
            for (int t = 0; t < n; ++t) {
                double sum = 0;
                for (int d = 0; d < dim; ++d)
                    sum += double(q[(b * qh + h) * dim + d]) * k[((b * context + t) * kh + head) * dim + d];
                p[t] = sum / std::sqrt(double(dim));
                top = std::max(top, p[t]);
            }
            for (auto &a : p) {
                a = std::exp(a - top);
                total += a;
            }
            for (auto &a : p)
                a = float(BFloat16(float(a / total)));
            for (int d = 0; d < dim; ++d) {
                double sum = 0;
                for (int t = 0; t < n; ++t)
                    sum += p[t] * v[((b * context + t) * kh + head) * dim + d];
                expected[(b * qh + h) * dim + d] = float(BFloat16(float(sum)));
            }
        }
    int maximum = *std::max_element(positions.begin(), positions.end()) + 1;
    cu::gqa_batch(dq, dk, dv, out, work, pos, batch, context, maximum, qh, kh, dim, s);
    compare(read(out.p, q.size(), s), expected, "Batched causal GQA/cache isolation");
    cg::Buffer<float> parts;
    parts.ensure(size_t(batch) * kh * ((maximum + 127) / 128) * 16 * dim, s);
    cu::gqa_batch(dq, dk, dv, out, work, pos, batch, context, maximum, qh, kh, dim, s, 0, parts);
    auto registered = read(out.p, q.size(), s);
    compare(registered, expected, "Register tensor-core GQA, split context, padded tails");
    cu::gqa_batch(dq, dk, dv, out, work, pos, batch, context, maximum, qh, kh, dim, s, 0, parts, false);
    require(registered == read(out.p, q.size(), s), "Register GQA must match WMMA staging exactly");
    std::vector<uint16_t> wide(size_t(batch) * qh * dim * 2, 0x7fc0);
    auto encoded = bits(q);
    for (int b = 0; b < batch; ++b)
        std::copy(encoded.begin() + size_t(b) * qh * dim, encoded.begin() + size_t(b + 1) * qh * dim,
                  wide.begin() + size_t(2 * b + 1) * qh * dim);
    cg::Buffer<uint16_t> strided_query;
    put(strided_query, wide, s);
    cu::gqa_batch(strided_query.p + qh * dim, dk, dv, out, work, pos, batch, context, maximum, qh, kh, dim, s,
                  2 * qh * dim, parts);
    require(registered == read(out.p, q.size(), s),
            "Register GQA must respect rank offsets and query strides");
}
#if defined(BEHEMOTH_CUDA_FAST)
void prefill_attention_test(cg::Stream s) {
    constexpr int batch = 2, qh = 48, kh = 4, dim = 128, sequence = 64, context = 80, stride = 96 * 128;
    std::vector<int> lengths{41, 64};
    std::mt19937 gen(53);
    std::normal_distribution<float> normal(0, .2f);
    std::vector<float> q(size_t(batch) * sequence * stride), k(size_t(batch) * context * kh * dim),
        v(k.size());
    for (auto *vec : {&q, &k, &v})
        for (auto &a : *vec)
            a = float(BFloat16(normal(gen)));
    cg::Buffer<uint16_t> dq, dk, dv, out;
    cg::Buffer<int> len;
    put(dq, bits(q), s);
    put(dk, bits(k), s);
    put(dv, bits(v), s);
    put(len, lengths, s);
    out.ensure(size_t(batch) * sequence * qh * dim, s);
    cu::FastPrefill attention(qh, kh, dim, stride, s);
    attention.run(dq.p + qh * dim, dk, dv, out, len, batch, sequence, context);
    auto actual = read(out.p, size_t(batch) * sequence * qh * dim, s);
    std::vector<float> selected, expected;
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < lengths[b]; ++t)
            for (int h = 0; h < qh; ++h) {
                int head = h / (qh / kh);
                std::vector<double> p(t + 1);
                double top = -INFINITY, total = 0;
                for (int at = 0; at <= t; ++at) {
                    double sum = 0;
                    for (int d = 0; d < dim; ++d)
                        sum += double(q[(size_t(b) * sequence + t) * stride + qh * dim + h * dim + d]) *
                               k[((size_t(b) * context + at) * kh + head) * dim + d];
                    p[at] = sum / std::sqrt(double(dim));
                    top = std::max(top, p[at]);
                }
                for (auto &a : p) {
                    a = std::exp(a - top);
                    total += a;
                }
                for (int d = 0; d < dim; ++d) {
                    double sum = 0;
                    for (int at = 0; at <= t; ++at)
                        sum += (p[at] / total) * v[((size_t(b) * context + at) * kh + head) * dim + d];
                    expected.push_back(float(BFloat16(float(sum))));
                    selected.push_back(actual[((size_t(b) * sequence + t) * qh + h) * dim + d]);
                }
            }
    compare(selected, expected, "cuDNN causal GQA, strided heads, variable lengths");
}
#endif
} // namespace
int main() {
    int count = 0;
    auto status = cg::device_count(&count);
    if (status == cg::no_device || status == cg::insufficient_driver || (status == cg::success && !count))
        return 77;
    cg::Stream s{};
    try {
        cg::checked(status, "batch GPU discovery");
        cg::checked(cg::stream_create(&s), "batch test stream");
        cu::initialize_tables();
        for (auto [out, in] : {std::pair{128, 128}, std::pair{768, 1792}, std::pair{1792, 768}})
            for (int tokens : {8, 17, 65})
                qtip_test(s, out, in, tokens);
        // Exercise the register kernel's partial row tile and both token tiles.
        for (int tokens : {1, 2, 7, 16})
            qtip_test(s, 96, 128, tokens);
        qtip_test(s, 32, 12288, 1);
        qtip_test(s, 32, 28672, 1);
        for (int tokens : {8, 17, 65})
            dense_test(s, tokens);
        norm_test(s, 1);
        norm_test(s, 8);
        attention_test(s);
        attention_test(s, 257, 24, 2);
        attention_test(s, 8193, 24, 2);
#if defined(BEHEMOTH_CUDA_FAST)
        prefill_attention_test(s);
#endif
        cg::checked(cg::stream_destroy(s), "destroy batch test stream");
        std::cout << "PASS batched QTIP, BF16, FP8, tail tiles, and causal cache isolation\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (s)
            (void)cg::stream_destroy(s);
        return 1;
    }
}
