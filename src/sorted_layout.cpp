#include "behemoth/sorted_layout.hpp"
#include <algorithm>
#include <bit>
#include <stdexcept>

namespace behemoth {
namespace {
struct Emitter {
    SortedLayout &l;
    const int S, DB, MAXD;
    uint16_t dummy_index = 0;
    uint32_t tile_chunk_begin = 0, tile_segment_begin = 0;
    uint32_t chunk = 0; // global chunk index of the chunk being filled
    int fill = 0;       // entries in the current chunk
    uint32_t running = 0;
    bool segment_open = false;
    Emitter(SortedLayout &layout) : l(layout), S(layout.chunks_per_segment), DB(layout.delta_bits),
                                    MAXD((1 << layout.delta_bits) - 1) {}
    void put(uint16_t index, int delta) {
        if (fill == 0) {
            l.idx.resize(size_t(chunk + 1) * 32, 0);
            l.planes.resize(size_t(chunk + 1) * size_t(DB), 0);
        }
        l.idx[size_t(chunk) * 32 + size_t(fill)] = index;
        for (int k = 0; k < DB; ++k)
            if ((delta >> k) & 1)
                l.planes[size_t(chunk) * size_t(DB) + size_t(k)] |= 1u << fill;
        if (++fill == 32) {
            fill = 0;
            ++chunk;
            if ((chunk - tile_chunk_begin) % uint32_t(S) == 0)
                segment_open = false;
        }
    }
    void begin_tile() {
        tile_chunk_begin = chunk;
        tile_segment_begin = uint32_t(l.segment_codes.size());
        segment_open = false;
    }
    void entry(uint16_t index, uint32_t code) {
        if (!segment_open) {
            l.segment_codes.push_back(uint16_t(code));
            running = code;
            segment_open = true;
            put(index, 0);
            ++l.entries;
            return;
        }
        uint32_t gap = code - running;
        while (gap > uint32_t(MAXD)) {
            put(dummy_index, MAXD);
            ++l.dummies;
            running += uint32_t(MAXD);
            gap -= uint32_t(MAXD);
            if (!segment_open) { // the dummy closed a segment: restart cleanly
                l.segment_codes.push_back(uint16_t(code));
                running = code;
                segment_open = true;
                put(index, 0);
                ++l.entries;
                return;
            }
        }
        put(index, int(gap));
        running = code;
        ++l.entries;
    }
    SortedTileDesc end_tile() {
        while (fill != 0) { // pad the last chunk with dummy entries (no code advance)
            put(dummy_index, 0);
            ++l.dummies;
        }
        SortedTileDesc d{};
        d.chunk_begin = tile_chunk_begin;
        d.chunk_count = chunk - tile_chunk_begin;
        d.segment_begin = tile_segment_begin;
        d.segment_count = uint32_t(l.segment_codes.size()) - tile_segment_begin;
        return d;
    }
};
} // namespace
SortedLayout sorted_layout(int in, int out, const uint16_t *codes, int R, int G, int DB, int S) {
    if (in <= 0 || in % 8 || out <= 0 || R <= 0 || G <= 0 || !std::has_single_bit(unsigned(R)) ||
        !std::has_single_bit(unsigned(G)) || size_t(R) * size_t(G) > 65536 || DB < 1 || DB > 4 ||
        S < 1 || in / 8 >= G)
        throw std::invalid_argument("Invalid sorted layout parameters (need in/8 < G)");
    SortedLayout l;
    l.rows_per_tile = R;
    l.groups_per_tile = G;
    l.delta_bits = DB;
    l.chunks_per_segment = S;
    const int groups = in / 8;
    l.row_tiles = (out + R - 1) / R;
    l.group_tiles = (groups + G - 1) / G;
    const int gbits = std::countr_zero(unsigned(G));
    Emitter e(l);
    e.dummy_index = uint16_t(G - 1);
    std::vector<uint32_t> count(65537);
    std::vector<uint16_t> order; // tile entries sorted by code (counting sort)
    std::vector<uint16_t> local_codes;
    l.idx.reserve(size_t(out) * size_t(groups) * 12 / 10);
    for (int rt = 0; rt < l.row_tiles; ++rt)
        for (int gt = 0; gt < l.group_tiles; ++gt) {
            const int r0 = rt * R, r1 = std::min(out, r0 + R), g0 = gt * G,
                      g1 = std::min(groups, g0 + G);
            std::fill(count.begin(), count.end(), 0u);
            local_codes.clear();
            for (int r = r0; r < r1; ++r)
                for (int g = g0; g < g1; ++g) {
                    uint16_t c = codes[size_t(r) * size_t(groups) + size_t(g)];
                    local_codes.push_back(c);
                    ++count[size_t(c) + 1];
                }
            for (size_t c = 0; c < 65536; ++c)
                count[c + 1] += count[c];
            order.resize(local_codes.size());
            {
                size_t k = 0;
                for (int r = r0; r < r1; ++r)
                    for (int g = g0; g < g1; ++g, ++k)
                        order[count[local_codes[k]]++] =
                            uint16_t(((r - r0) << gbits) | (g - g0));
            }
            // count[c] now holds the end of bucket c; walk buckets in order.
            e.begin_tile();
            size_t pos = 0;
            for (uint32_t c = 0; c < 65536 && pos < order.size(); ++c)
                for (; pos < count[c]; ++pos)
                    e.entry(order[pos], c);
            l.tiles.push_back(e.end_tile());
        }
    return l;
}
bool sorted_layout_decode(const SortedLayout &l, int in, int out, std::vector<uint16_t> &codes) {
    const int groups = in / 8, R = l.rows_per_tile, G = l.groups_per_tile, DB = l.delta_bits;
    const int gbits = std::countr_zero(unsigned(G));
    if (l.tiles.size() != size_t(l.row_tiles) * size_t(l.group_tiles))
        return false;
    codes.assign(size_t(out) * size_t(groups), 0);
    std::vector<uint8_t> seen(codes.size(), 0);
    size_t real = 0;
    for (int rt = 0; rt < l.row_tiles; ++rt)
        for (int gt = 0; gt < l.group_tiles; ++gt) {
            const auto &t = l.tiles[size_t(rt) * size_t(l.group_tiles) + size_t(gt)];
            for (uint32_t s = 0; s < t.segment_count; ++s) {
                uint32_t running = l.segment_codes[t.segment_begin + s];
                const uint32_t c0 = t.chunk_begin + s * uint32_t(l.chunks_per_segment),
                               c1 = std::min(t.chunk_begin + t.chunk_count,
                                             c0 + uint32_t(l.chunks_per_segment));
                for (uint32_t c = c0; c < c1; ++c)
                    for (int lane = 0; lane < 32; ++lane) {
                        int d = 0;
                        for (int k = 0; k < DB; ++k)
                            d |= int((l.planes[size_t(c) * size_t(DB) + size_t(k)] >> lane) & 1)
                                 << k;
                        running += uint32_t(d);
                        const uint16_t e = l.idx[size_t(c) * 32 + size_t(lane)];
                        if ((e & (G - 1)) == G - 1)
                            continue; // dummy
                        const int r = rt * R + (e >> gbits), g = gt * G + (e & (G - 1));
                        if (r >= out || g >= groups || running > 65535)
                            return false;
                        const size_t at = size_t(r) * size_t(groups) + size_t(g);
                        if (seen[at])
                            return false;
                        seen[at] = 1;
                        codes[at] = uint16_t(running);
                        ++real;
                    }
            }
        }
    return real == codes.size();
}
} // namespace behemoth
