#pragma once
// Load-time expansion of a QTIP projection into an FP8 (E4M3) row-scaled shard for the Neuron FP8 graph.
//
// The checkpoint stays QTIP (2 bits/weight). At model load the host decodes the trellis, applies the
// randomized Hadamard rotations and the SU/SV scales to form the effective weight
//     W_eff = diag(SV) . M_out . W~ . M_in . diag(SU)        (y = W_eff x, M_in = hadUt_in, M_out = hadU_out)
// and writes this rank's slice as one E4M3 byte per weight plus one fp32 scale per row. That slice is exactly
// what the tensor-parallel FP8 layer graph (trace_layer_tp.py) consumes, so QTIP layers need no extra
// collectives at run time. Column-parallel projections keep a block of output rows, row-parallel ones a
// block of input columns.
#include "behemoth/qtip.hpp"
#include <cstdint>
#include <vector>

namespace neuron {
struct Fp8Shard {
    int rows = 0, cols = 0;
    std::vector<uint8_t> w;     // [rows][cols] E4M3 bytes (|value| <= 224)
    std::vector<float> scale;   // [rows] row scale: weight = decode(byte) * scale
};

// tp = number of ranks, rank in [0, tp). column_parallel: shard of rows; otherwise shard of columns.
Fp8Shard expand_qtip_to_fp8(const behemoth::qtip::HostLinear &l, int tp, int rank, bool column_parallel);
// All tp shards of one projection, decoding the QTIP matrix only once.
std::vector<Fp8Shard> expand_qtip_all(const behemoth::qtip::HostLinear &l, int tp, bool column_parallel);

// A dense f32 [rows][cols] matrix (e.g. this rank's lm_head rows) as one row-scaled E4M3 shard.
Fp8Shard quantize_rows(const float *w, int rows, int cols);

// Any projection of a layer file (QTIP trellis, FP8 weight + scale, or NVFP4 packed) as tp FP8 shards,
// ready to bind to the tensor-parallel FP8 layer graph. FP8 and NVFP4 tensors are dequantized and
// re-encoded with a per-row scale (max 224).
std::vector<Fp8Shard> load_projection_shards(const behemoth::SafeTensorFile &file, int layer, const std::string &name,
                                             const behemoth::ModelConfig &config, int tp, bool column_parallel);

// A QTIP projection kept COMPRESSED on the device for the fused decode kernel (neuron/qtip_nki.py, WEIGHTS=qtip
// graphs): this rank's trellis words re-ordered so that the matmul's contraction axis is on the SBUF partitions,
// plus the device codebook and the rotation scales. Mirrors qtip_nki.pack_shard / codebook_u32 exactly.
struct QtipPacked {
    int out_local = 0, in_local = 0;  // this rank's matrix (column-parallel: rows sharded; row-parallel: columns)
    int groups = 0, out_tiles = 0;    // words layout [groups][128][out_tiles * 32]: partition p of group g holds the
    std::vector<uint16_t> words;      // 32 words of every output tile for input channels g*2048 + 16 p + c (zero padded)
    std::vector<uint32_t> lut;        // [128][1024] bf16x2 codebook (qtip::Codebook::bf16x2) replicated per partition
    std::vector<float> su, sv;        // [in] and [out] of the FULL matrix (the graph rotates full vectors)
    size_t device_bytes() const { return words.size() * 2 + lut.size() * 4 + (su.size() + sv.size()) * 4; }
};
QtipPacked pack_qtip_shard(const behemoth::qtip::HostLinear &l, int tp, int rank, bool column_parallel);
// Resident bytes of pack_qtip_shard for a (out, in) QTIP projection without decoding it (planning).
size_t qtip_packed_bytes(int out, int in, int tp, bool column_parallel);
// True when the layer file stores `name` as a QTIP trellis (rather than FP8 or NVFP4 tensors).
bool is_qtip_projection(const behemoth::SafeTensorFile &file, int layer, const std::string &name);
} // namespace neuron
