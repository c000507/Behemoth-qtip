// CPU simulation of the "lane-owns-row" sorted layout: L0 locality of the
// codebook gathers for W lock-step waves, dummy fraction of a drift limiter.
// Usage: lane_sim MODEL LAYER NAME GT D W [L0_LINES]
#include "behemoth/checkpoint.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

using namespace behemoth;
struct Lru {
    size_t cap;
    std::list<uint32_t> order;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> map;
    size_t hits = 0, misses = 0;
    explicit Lru(size_t c) : cap(c) {}
    void touch(uint32_t line) {
        auto it = map.find(line);
        if (it != map.end()) {
            order.erase(it->second);
            order.push_front(line);
            it->second = order.begin();
            ++hits;
            return;
        }
        ++misses;
        if (map.size() == cap) {
            map.erase(order.back());
            order.pop_back();
        }
        order.push_front(line);
        map[line] = order.begin();
    }
};
int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "lane_sim MODEL LAYER NAME GT D W [L0_LINES=256]\n");
        return 2;
    }
    Checkpoint ck(argv[1]);
    int layer = atoi(argv[2]);
    std::string name = argv[3];
    int GT = atoi(argv[4]), D = atoi(argv[5]), W = atoi(argv[6]);
    size_t l0 = argc > 7 ? size_t(atoi(argv[7])) : 256;
    bool column = name != "self_attn.o_proj" && name != "mlp.down_proj";
    auto p = ck.projection(layer, name, 0, column);
    const int groups = p.in / 8, R = 16;
    const int gtiles = (groups + GT - 1) / GT;
    const int gt_size = (groups + gtiles - 1) / gtiles;
    const int row_tiles = p.out / R;
    // Build lane streams for all tiles: tile (rt, gt) -> 32 streams (sorted codes).
    size_t total_steps = 0, total_entries = 0, total_dummies = 0;
    double max_span_sum = 0;
    size_t span_steps = 0;
    size_t lines_per_step_sum = 0;
    Lru cache(l0);
    size_t l2_lines_sorted = 0; // reference: fully sorted stream lines per 32 entries
    for (int gt = 0; gt < gtiles; ++gt) {
        const int g0 = gt * gt_size, g1 = std::min(groups, g0 + gt_size);
        for (int rt0 = 0; rt0 < row_tiles; rt0 += W) {
            // W waves in lock-step.
            std::vector<std::vector<std::vector<uint16_t>>> streams{size_t(W)};
            std::vector<std::vector<uint16_t>> full{size_t(W)};
            for (int w = 0; w < W; ++w) {
                int rt = rt0 + w;
                streams[size_t(w)].assign(32, {});
                if (rt >= row_tiles)
                    continue;
                for (int r = 0; r < R; ++r) {
                    std::vector<uint16_t> c;
                    for (int g = g0; g < g1; ++g)
                        c.push_back(p.codes[size_t(rt * R + r) * size_t(groups) + size_t(g)]);
                    std::sort(c.begin(), c.end());
                    full[size_t(w)].insert(full[size_t(w)].end(), c.begin(), c.end());
                    for (size_t i = 0; i < c.size(); ++i)
                        streams[size_t(w)][size_t(2 * r + (i & 1))].push_back(c[i]);
                }
                std::sort(full[size_t(w)].begin(), full[size_t(w)].end());
                // reference lines per chunk for the fully sorted stream
                for (size_t i = 0; i < full[size_t(w)].size(); i += 32) {
                    std::vector<uint32_t> ls;
                    for (size_t j = i; j < std::min(i + 32, full[size_t(w)].size()); ++j)
                        ls.push_back(full[size_t(w)][j] >> 4);
                    std::sort(ls.begin(), ls.end());
                    l2_lines_sorted += size_t(std::unique(ls.begin(), ls.end()) - ls.begin());
                }
            }
            std::vector<std::vector<size_t>> pos(size_t(W), std::vector<size_t>(32, 0));
            bool any = true;
            while (any) {
                any = false;
                std::vector<uint32_t> lines;
                uint32_t lo = 65536, hi = 0;
                for (int w = 0; w < W; ++w) {
                    auto &s = streams[size_t(w)];
                    auto &ps = pos[size_t(w)];
                    // frontier: min next code across this wave's lanes
                    uint32_t frontier = 65536;
                    for (int l = 0; l < 32; ++l)
                        if (ps[size_t(l)] < s[size_t(l)].size())
                            frontier = std::min<uint32_t>(frontier, s[size_t(l)][ps[size_t(l)]]);
                    if (frontier == 65536)
                        continue;
                    any = true;
                    ++total_steps;
                    for (int l = 0; l < 32; ++l) {
                        if (ps[size_t(l)] >= s[size_t(l)].size()) {
                            ++total_dummies; // tail padding
                            continue;
                        }
                        uint32_t c = s[size_t(l)][ps[size_t(l)]];
                        if (D > 0 && c > frontier + uint32_t(D)) {
                            ++total_dummies; // drift limiter wait
                            continue;
                        }
                        ++ps[size_t(l)];
                        ++total_entries;
                        lines.push_back(c >> 4);
                        lo = std::min(lo, c);
                        hi = std::max(hi, c);
                    }
                }
                if (lines.empty())
                    continue;
                for (uint32_t ln : lines)
                    cache.touch(ln);
                std::sort(lines.begin(), lines.end());
                lines_per_step_sum += size_t(std::unique(lines.begin(), lines.end()) - lines.begin());
                max_span_sum += double(hi - lo) * 8 / 1024;
                ++span_steps;
            }
        }
    }
    double slots = double(total_entries + total_dummies);
    printf("%s layer %d in=%d out=%d GT=%d(size %d, %d tiles) D=%d W=%d L0=%zu lines\n", name.c_str(),
           layer, p.in, p.out, GT, gt_size, gtiles, D, W, l0);
    printf("  entries %zu dummies %zu (%.2f%%) wave-steps %zu\n", total_entries, total_dummies,
           100.0 * double(total_dummies) / slots, total_steps);
    printf("  L0 hit rate %.3f  (L2 lines per real entry %.3f; fully-sorted reference %.3f)\n",
           double(cache.hits) / double(cache.hits + cache.misses),
           double(cache.misses) / double(total_entries),
           double(l2_lines_sorted) / double(total_entries));
    printf("  distinct lines per lockstep step (all %d waves) %.1f; mean code span %.1f KiB\n", W,
           double(lines_per_step_sum) / double(span_steps), max_span_sum / double(span_steps));
    return 0;
}
