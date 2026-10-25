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
} // namespace behemoth
