#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace behemoth {
// Codebook-major ("sorted tile") code layout for AQLM 1x16 decode.
//
// The weight matrix is cut into tiles of R output rows x G eight-weight groups.
// Inside a tile every (row, group, code) entry is sorted by code and stored in
// chunks of 32 entries (one Wave32 step):
//   idx[chunk*32 + lane]  = (row_local << log2(G)) | group_local   (uint16)
//   planes[chunk*DB + k]  = bit k of the 32 code deltas (bit-plane layout):
//                           delta_lane = sum_k ((planes[k] >> lane) & 1) << k
// The code of an entry is the running code at the start of its segment plus
// the inclusive prefix sum of deltas (two popcounts per lane, no shuffles).
// Dummy entries carry group_local == G-1 (the layout requires in/8 < G, so
// that group never holds a weight and the kernel keeps a zero activation in
// that slot): they bridge code gaps larger than 2^DB-1 and pad the last chunk
// of a tile, contributing exactly zero without any masking. Consecutive lanes
// therefore read consecutive or nearly consecutive codebook entries, so the
// 8-byte codebook gather becomes a near-streaming read instead of a random one.
//
// Chunks are grouped into segments of S chunks; segment_codes holds the code
// of the first (real) entry of each segment so that independent waves/blocks
// can start anywhere without a serial scan. chunk_codes holds the running
// code at the start of every chunk (code of an entry = chunk code + inclusive
// delta prefix), so any chunk decodes independently: the batched kernels
// decode 8 chunks per wave step without a serial scan (+2 B per 32 entries).
// Runtime configuration: tiles of 16 output rows (a 16 x 1792-group code tile
// is 56 KiB, so one tile's codes fit in LDS for the batched kernels; code
// density 16*groups/65536 >= 0.375 keeps the gather near-streaming); 2048 >
// 1792 groups covers every projection of the model (group 2047 is the dummy
// slot).
constexpr int sorted_rows_per_tile = 16, sorted_groups_per_tile = 2048,
              sorted_default_chunks_per_segment = 16, sorted_default_delta_bits = 4;
struct SortedTileDesc {
    uint32_t chunk_begin, chunk_count, segment_begin, segment_count;
};
struct SortedLayout {
    int rows_per_tile = 256, groups_per_tile = 256, delta_bits = 3, chunks_per_segment = 64;
    int row_tiles = 0, group_tiles = 0;
    std::vector<uint16_t> idx;           // 32 per chunk
    std::vector<uint32_t> planes;        // delta_bits per chunk
    std::vector<uint16_t> segment_codes; // one per segment
    std::vector<uint16_t> chunk_codes;   // one per chunk (running code at chunk start)
    std::vector<SortedTileDesc> tiles;   // row_tiles * group_tiles, tile = rt * group_tiles + gt
    size_t entries = 0, dummies = 0;
    size_t bytes() const {
        return idx.size() * 2 + planes.size() * 4 + segment_codes.size() * 2 +
               chunk_codes.size() * 2 + tiles.size() * sizeof(SortedTileDesc);
    }
    size_t chunks() const {
        return idx.size() / 32;
    }
};
// codes: row-major [out][in/8] uint16 (the checkpoint layout).
// R and G must be powers of two with R * G <= 65536 and in/8 < G; delta_bits in 1..4.
SortedLayout sorted_layout(int in, int out, const uint16_t *row_major_codes, int rows_per_tile,
                           int groups_per_tile, int delta_bits,
                           int chunks_per_segment = sorted_default_chunks_per_segment);
// Decodes a layout back to row-major codes (verification). Returns false on
// malformed input or if an entry is missing/duplicated.
bool sorted_layout_decode(const SortedLayout &, int in, int out, std::vector<uint16_t> &codes);

// "Lane" layout (lane-owns-row, PERF_ROUND2_REPORT.md). Tiles of 16 output
// rows x up to 2047 groups; lane l of a Wave32 owns row l/2 of the tile: the
// row's codes (over the tile's groups) are sorted and alternate between its
// two lanes, so chunk j holds the j-th entry of every lane's own stream.
//   idx[(segment*32 + lane)*S + j]  = group_local | (delta & 31) << 11   (uint16)
//   hi[(segment*32 + lane)*S/8 + j/8] nibble j%8 = delta >> 5             (uint32)
//   (both lane-major within a segment of S = 32 chunks: a lane's whole
//   segment is four 16-byte idx loads plus one 16-byte nibble load, no
//   cross-lane or scalar loads in the hot loop)
//   code_j = (code_{j-1} + delta_j) & 0xffff, per lane; delta_bits = 9
//   segment_codes[(segment)*32 + lane] = each lane's running code before
//                                        the segment's first chunk
//   tiles hold whole segments (the last one padded with delta-0 dummies), so
//   chunk_begin = segment_begin * S and chunk_count = segment_count * S.
// Dummies have group_local == 2047 (tail padding, gap escapes with delta
// 2^delta_bits - 1, drift-limiter waits); the kernels map any group >=
// groups_in_tile to a zero activation slot. No LDS atomics are needed: every
// lane accumulates its own row in registers, and M batch rows cost M
// activation reads per entry instead of M atomics. Consecutive chunks of a lane
// walk the codebook in order; the 32 lanes of a chunk stay within a few tens
// of KiB of each other (order-statistic drift), so the gathers hit L0.
// tile = gt * row_tiles + rt (consecutive tiles share their group range).
constexpr int lane_group_bits = 11, lane_dummy_group = 2047, lane_max_groups_per_tile = 2047,
              lane_delta_low_bits = 5, lane_delta_bits = 9, lane_chunks_per_segment = 32;
struct LaneLayout {
    int groups_per_tile = 0; // groups covered by every tile but the last of a row
    int drift_limit = 0;
    int row_tiles = 0, group_tiles = 0;
    std::vector<uint16_t> idx;           // 32 x S per segment, lane-major
    std::vector<uint32_t> hi;            // 32 x S/8 per segment, lane-major nibbles
    std::vector<uint16_t> segment_codes; // 32 per segment
    std::vector<SortedTileDesc> tiles;   // tile = gt * row_tiles + rt
    size_t entries = 0, dummies = 0;
    size_t chunks() const {
        return idx.size() / 32;
    }
    size_t segments() const {
        return segment_codes.size() / 32;
    }
    size_t bytes() const {
        return idx.size() * 2 + hi.size() * 4 + segment_codes.size() * 2 +
               tiles.size() * sizeof(SortedTileDesc);
    }
};
// codes: row-major [out][in/8]. max_groups_per_tile <= 2047 (group tiles are
// balanced: ceil(groups / tiles) each); drift_limit > 0 makes a lane wait
// (dummy) while its next code exceeds the chunk's smallest next code by more
// than drift_limit (tighter gather window, more dummies).
LaneLayout lane_layout(int in, int out, const uint16_t *row_major_codes,
                       int max_groups_per_tile = lane_max_groups_per_tile, int drift_limit = 0);
bool lane_layout_decode(const LaneLayout &, int in, int out, std::vector<uint16_t> &codes);
} // namespace behemoth
