#include "behemoth/attention.hpp"
#include <algorithm>
#include <array>
#include <unordered_set>

namespace behemoth {
void AttentionLayout::validate() const {
    if (!((query_heads == 48 && kv_heads == 4) || (query_heads == 96 && kv_heads == 8)))
        throw std::invalid_argument("Expected GQA heads 48/4 or 96/8");
    if (physical_pages <= 0 || table_stride <= 0 || lengths.empty() ||
        query_offsets.size() != lengths.size() + 1)
        throw std::invalid_argument("Invalid attention dimensions");
    if (query_offsets[0] != 0 || query_offsets.back() < 0 ||
        size_t(query_offsets.back()) != positions.size() ||
        page_table.size() != checked_mul(lengths.size(), size_t(table_stride)))
        throw std::invalid_argument("Invalid query offsets or page-table shape");
    if (queries() > 1024 || lengths.size() > 1024)
        throw std::invalid_argument("Attention batch exceeds the 1024-token runtime budget");
    for (size_t r = 0; r < lengths.size(); ++r)
        if (query_offsets[r] < 0 || query_offsets[r] > query_offsets[r + 1] ||
            query_offsets[r + 1] > queries())
            throw std::invalid_argument("Query offsets outside packed storage");
    std::unordered_set<int> owned;
    for (size_t r = 0; r < lengths.size(); ++r) {
        if (lengths[r] < 1 || lengths[r] > 131072 || query_offsets[r] > query_offsets[r + 1])
            throw std::invalid_argument("Invalid request length/offset");
        const int pages = int(ceil_div(size_t(lengths[r]), page_tokens));
        if (pages > table_stride)
            throw std::invalid_argument("Page table too short");
        for (int p = 0; p < pages; ++p) {
            int id = page_table[r * size_t(table_stride) + size_t(p)];
            if (id < 0 || id >= physical_pages || !owned.insert(id).second)
                throw std::invalid_argument("Invalid or aliased physical page");
        }
        int previous = -1;
        for (int q = query_offsets[r]; q < query_offsets[r + 1]; ++q) {
            if (positions[size_t(q)] < 0 || positions[size_t(q)] >= lengths[r] ||
                positions[size_t(q)] <= previous)
                throw std::invalid_argument("Invalid/nonmonotonic query position");
            previous = positions[size_t(q)];
        }
    }
}
void AttentionData::validate(const AttentionLayout &l) const {
    l.validate();
    auto qs = checked_mul(size_t(l.queries()), size_t(l.query_heads));
    auto ks = checked_mul(checked_mul(size_t(l.physical_pages), size_t(l.kv_heads)), page_tokens);
    if (q.size() != checked_mul(qs, head_dim) || q_scale.size() != qs ||
        k.size() != checked_mul(ks, head_dim) || v.size() != k.size() || k_scale.size() != ks ||
        v_scale.size() != ks)
        throw std::invalid_argument("Attention storage shape mismatch");
    if (!q_bf16.empty() && q_bf16.size() != checked_mul(qs, head_dim))
        throw std::invalid_argument("BF16 query storage shape mismatch");
    // Unused pages/padding may be poisoned; validate only the entries the kernel can read.
    for (float s : q_scale)
        if (!(s > 0 && std::isfinite(s)))
            throw std::invalid_argument("Invalid Q scale");
    for (int r = 0; r < l.requests(); ++r)
        for (int t = 0; t < l.lengths[size_t(r)]; ++t)
            for (int h = 0; h < l.kv_heads; ++h) {
                int p = l.page_table[size_t(r * l.table_stride + t / page_tokens)];
                size_t i = (size_t(p) * size_t(l.kv_heads) + size_t(h)) * page_tokens +
                           size_t(t % page_tokens);
                if (!(k_scale[i] > 0 && std::isfinite(k_scale[i]) && v_scale[i] > 0 &&
                      std::isfinite(v_scale[i])))
                    throw std::invalid_argument("Invalid live KV scale");
            }
}
AttentionResult reference_attention(const AttentionLayout &l, const AttentionData &a, bool fp8_p,
                                    int partition) {
    a.validate(l);
    if (partition < 0 || partition % key_tile)
        throw std::invalid_argument("Partition must be a multiple of 32");
    AttentionResult result;
    size_t rows = size_t(l.queries()) * size_t(l.query_heads);
    result.output.resize(rows * head_dim);
    result.maximum.resize(rows);
    result.denominator.resize(rows);
    // Deliberately scalar indexing and double dot/exp/merge; no GPU address helpers are used.
    const bool bf16_q = !a.q_bf16.empty();
    for (int request = 0; request < l.requests(); ++request)
        for (int query = l.query_offsets[size_t(request)];
             query < l.query_offsets[size_t(request + 1)]; ++query)
            for (int head = 0; head < l.query_heads; ++head) {
                const size_t row = size_t(query) * size_t(l.query_heads) + size_t(head);
                const int kh = head / 12, end = l.positions[size_t(query)] + 1;
                const int part_size =
                    partition ? partition : int(ceil_div(size_t(end), key_tile) * key_tile);
                double global_m = -INFINITY, global_l = 0;
                std::array<double, head_dim> global_o{};
                for (int start = 0; start < end; start += part_size) {
                    double m = -INFINITY, sum = 0;
                    std::array<double, head_dim> out{};
                    for (int base = start; base < std::min(end, start + part_size);
                         base += key_tile) {
                        int count = std::min({key_tile, end - base, start + part_size - base});
                        std::array<double, key_tile> scores{}, p{};
                        std::array<size_t, key_tile> locations{};
                        double tile_m = -INFINITY;
                        for (int j = 0; j < count; ++j) {
                            int token = base + j;
                            int physical = l.page_table[size_t(request) * size_t(l.table_stride) +
                                                        size_t(token / page_tokens)];
                            size_t loc =
                                (size_t(physical) * size_t(l.kv_heads) + size_t(kh)) * page_tokens +
                                size_t(token % page_tokens);
                            locations[size_t(j)] = loc;
                            double dot = 0;
                            for (int d = 0; d < head_dim; ++d)
                                dot += (bf16_q ? double(float(a.q_bf16[row * head_dim + size_t(d)]))
                                               : double(decode_e4m3(a.q[row * head_dim + size_t(d)])) *
                                                     double(a.q_scale[row])) *
                                       double(decode_e4m3(a.k[loc * head_dim + size_t(d)]));
                            scores[size_t(j)] =
                                dot * double(a.k_scale[loc]) / std::sqrt(double(head_dim));
                            tile_m = std::max(tile_m, scores[size_t(j)]);
                        }
                        const double next_m = std::max(m, tile_m),
                                     alpha = std::isfinite(m) ? std::exp(m - next_m) : 0;
                        sum *= alpha;
                        for (double &v : out)
                            v *= alpha;
                        float weighted_max = 0;
                        for (int j = 0; j < count; ++j) {
                            double e = std::exp(scores[size_t(j)] - next_m);
                            sum += e;
                            p[size_t(j)] = e * double(a.v_scale[locations[size_t(j)]]);
                            weighted_max = std::max(weighted_max, float(p[size_t(j)]));
                        }
                        float ps = weighted_max == 0 ? 1.f
                                                     : std::max(weighted_max / 448.f,
                                                                std::numeric_limits<float>::min());
                        for (int j = 0; j < count; ++j) {
                            double w =
                                fp8_p ? double(decode_e4m3(encode_e4m3(float(p[size_t(j)]) / ps))) *
                                            double(ps)
                                      : p[size_t(j)];
                            for (int d = 0; d < head_dim; ++d)
                                out[size_t(d)] +=
                                    w * double(decode_e4m3(
                                            a.v[locations[size_t(j)] * head_dim + size_t(d)]));
                        }
                        m = next_m;
                    }
                    double next_m = std::max(global_m, m),
                           ga = std::isfinite(global_m) ? std::exp(global_m - next_m) : 0;
                    double pa = std::exp(m - next_m);
                    global_l = global_l * ga + sum * pa;
                    for (int d = 0; d < head_dim; ++d)
                        global_o[size_t(d)] = global_o[size_t(d)] * ga + out[size_t(d)] * pa;
                    global_m = next_m;
                }
                result.maximum[row] = float(global_m);
                result.denominator[row] = float(global_l);
                for (int d = 0; d < head_dim; ++d)
                    result.output[row * head_dim + size_t(d)] =
                        float(global_o[size_t(d)] / global_l);
            }
    return result;
}
void reference_append(const AttentionLayout &l, AttentionData &a, std::span<const BFloat16> k,
                      std::span<const BFloat16> v) {
    l.validate();
    size_t n = size_t(l.queries()) * size_t(l.kv_heads) * head_dim;
    if (k.size() != n || v.size() != n)
        throw std::invalid_argument("KV append shape mismatch");
    size_t entries = size_t(l.physical_pages) * size_t(l.kv_heads) * page_tokens;
    if (a.k.size() != entries * head_dim || a.v.size() != entries * head_dim ||
        a.k_scale.size() != entries || a.v_scale.size() != entries)
        throw std::invalid_argument("KV destination shape mismatch");
    for (int r = 0; r < l.requests(); ++r)
        for (int q = l.query_offsets[size_t(r)]; q < l.query_offsets[size_t(r + 1)]; ++q)
            for (int h = 0; h < l.kv_heads; ++h) {
                int t = l.positions[size_t(q)],
                    p = l.page_table[size_t(r * l.table_stride + t / page_tokens)];
                size_t dest = (size_t(p) * size_t(l.kv_heads) + size_t(h)) * page_tokens +
                              size_t(t % page_tokens);
                std::array<float, head_dim> fk{}, fv{};
                for (int d = 0; d < head_dim; ++d) {
                    size_t i = (size_t(q) * size_t(l.kv_heads) + size_t(h)) * head_dim + size_t(d);
                    fk[size_t(d)] = float(k[i]);
                    fv[size_t(d)] = float(v[i]);
                }
                auto qk = quantize(fk), qv = quantize(fv);
                a.k_scale[dest] = qk.scale;
                a.v_scale[dest] = qv.scale;
                std::copy(qk.values.begin(), qk.values.end(),
                          a.k.begin() + std::ptrdiff_t(dest * head_dim));
                std::copy(qv.values.begin(), qv.values.end(),
                          a.v.begin() + std::ptrdiff_t(dest * head_dim));
            }
}
ErrorMetrics compare_attention(std::span<const float> a, std::span<const float> b) {
    if (a.empty() || a.size() != b.size())
        throw std::invalid_argument("Comparison shape mismatch");
    ErrorMetrics m;
    double se = 0, sr = 0;
    bool elements = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return {INFINITY, INFINITY, INFINITY, false};
        double e = std::abs(double(a[i]) - double(b[i]));
        se += e * e;
        sr += double(b[i]) * double(b[i]);
        m.max_abs = std::max(m.max_abs, e);
        elements &= e <= .01 + .02 * std::abs(double(b[i]));
    }
    m.rms_error = std::sqrt(se / double(a.size()));
    m.reference_rms = std::sqrt(sr / double(a.size()));
    m.pass = elements && m.rms_error <= .001 + .01 * m.reference_rms;
    return m;
}
} // namespace behemoth
