#pragma once
#include "behemoth/attention.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace behemoth::test {
struct Fixture {
    AttentionLayout layout;
    AttentionData data;
};
inline Fixture fixture(std::vector<int> lengths, std::vector<std::vector<int>> positions,
                       int seed = 7, int qheads = 48) {
    Fixture f;
    auto &l = f.layout;
    auto &a = f.data;
    l.query_heads = qheads;
    l.kv_heads = qheads / 12;
    l.lengths = std::move(lengths);
    l.table_stride =
        int(ceil_div(size_t(*std::max_element(l.lengths.begin(), l.lengths.end())), page_tokens));
    l.physical_pages = 3;
    l.query_offsets = {0};
    for (size_t r = 0; r < l.lengths.size(); ++r) {
        l.physical_pages += int(ceil_div(size_t(l.lengths[r]), page_tokens));
        l.positions.insert(l.positions.end(), positions[r].begin(), positions[r].end());
        l.query_offsets.push_back(int(l.positions.size()));
    }
    std::vector<int> pages(size_t(l.physical_pages));
    std::iota(pages.begin(), pages.end(), 0);
    std::mt19937 rng{unsigned(seed)};
    std::shuffle(pages.begin(), pages.end(), rng);
    l.page_table.assign(l.lengths.size() * size_t(l.table_stride), -999);
    size_t next = 0;
    for (size_t r = 0; r < l.lengths.size(); ++r)
        for (int t = 0; t < l.lengths[r]; t += page_tokens)
            l.page_table[r * size_t(l.table_stride) + size_t(t / page_tokens)] = pages[next++];
    size_t qs = size_t(l.queries()) * size_t(l.query_heads),
           ks = size_t(l.physical_pages) * size_t(l.kv_heads) * page_tokens;
    a.q.resize(qs * head_dim);
    a.q_scale.resize(qs);
    a.k.assign(ks * head_dim, 127);
    a.v = a.k;
    a.k_scale.assign(ks, NAN);
    a.v_scale = a.k_scale;
    std::normal_distribution<float> normal(0, 1);
    std::array<float, head_dim> values{};
    for (size_t i = 0; i < qs; ++i) {
        for (float &v : values)
            v = normal(rng);
        auto q = quantize(values);
        a.q_scale[i] = q.scale;
        std::copy(q.values.begin(), q.values.end(), a.q.begin() + std::ptrdiff_t(i * head_dim));
    }
    for (size_t r = 0; r < l.lengths.size(); ++r)
        for (int t = 0; t < l.lengths[r]; ++t)
            for (int h = 0; h < l.kv_heads; ++h) {
                int p = l.page_table[r * size_t(l.table_stride) + size_t(t / page_tokens)];
                size_t loc = (size_t(p) * size_t(l.kv_heads) + size_t(h)) * page_tokens +
                             size_t(t % page_tokens);
                for (float &v : values)
                    v = normal(rng);
                auto k = quantize(values);
                a.k_scale[loc] = k.scale;
                std::copy(k.values.begin(), k.values.end(),
                          a.k.begin() + std::ptrdiff_t(loc * head_dim));
                for (float &v : values)
                    v = normal(rng) * (t % 3 == 0 ? .125f : 2.f);
                auto v = quantize(values);
                a.v_scale[loc] = v.scale;
                std::copy(v.values.begin(), v.values.end(),
                          a.v.begin() + std::ptrdiff_t(loc * head_dim));
            }
    return f;
}
} // namespace behemoth::test
