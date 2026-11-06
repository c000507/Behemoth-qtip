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

// Any projection of a layer file (QTIP trellis, FP8 weight + scale, or NVFP4 packed) as tp FP8 shards,
// ready to bind to the tensor-parallel FP8 layer graph. FP8 and NVFP4 tensors are dequantized and
// re-encoded with a per-row scale (max 224).
std::vector<Fp8Shard> load_projection_shards(const behemoth::SafeTensorFile &file, int layer, const std::string &name,
                                             const behemoth::ModelConfig &config, int tp, bool column_parallel);
} // namespace neuron
