#pragma once
// QTIP device kernels (src/qtip.hip): fused trellis-decode GEMV / WMMA matmuls on the rotated
// weights, fast Walsh-Hadamard transforms (hadK (x) H_1024) fused with norms, sign vectors,
// SV scaling and the block epilogues, and the reduce+publish / gather steps of the two-rank
// exchange. See QTIP_RUNTIME_REPORT.md for the design.
#include "gpu.hpp"
#include "qtip.hpp"

namespace behemoth::qtip {
// One TP shard of a linear (or several linears stacked along the output dimension): whole
// 32-row block-rows of the kernel-layout trellis, contiguous in device memory.
struct GpuLinear {
    DeviceBuffer<uint16_t> trellis;
    int in = 0, rows = 0; // rows: output rows of this shard (multiple of 32)
    // Concatenate block-row ranges (each a span from HostLinear::block_rows) along the output.
    void upload(const std::vector<std::span<const uint16_t>> &parts, int in);
    size_t bytes() const {
        return trellis.bytes();
    }
};
// Device codebook: 1024 sign-folded bf16x2 entries (Codebook::bf16x2).
struct GpuCodebook {
    DeviceBuffer<uint32_t> lut;
    void upload(const Codebook &cb) {
        lut.upload(std::span<const uint32_t>(cb.bf16x2.data(), cb.bf16x2.size()));
    }
};
// Kernel families of qtip_matmul: auto (by rows), GEMV (dot2, rows <= 4), WMMA (bf16 16x16x16).
enum class MatmulKernel { Auto, Gemv, Wmma };
struct MatmulOptions {
    MatmulKernel kernel = MatmulKernel::Auto;
    int slices = 0; // K splits (0: automatic)
    int copies = 0; // LDS LUT copies 1/4/8/16 (0: default)
};
// partial[s][rows][lin.rows] (fp32) = slice s of x . hatWr^T over the shard's rows, with x
// bf16 [rows][ldx] (the rotated input). Returns the number of slices written; the caller sums
// them (reduce_publish). partial must hold slices * rows * lin.rows floats: query with
// matmul_slices() first.
int matmul_slices(const GpuLinear &, int rows, const MatmulOptions & = {});
// Occupancy / resources of the matmul kernels on the current device (bench report).
nlohmann::json kernel_info();
void matmul(const GpuLinear &, const uint32_t *lut, const BFloat16 *x, int ldx, int rows,
            float *partial, int slices, hipStream_t, const MatmulOptions & = {});
// Sum `slices` partial slices of n elements (slice stride n): result fp32 or bf16 into `local`
// and, when host is non-null, also into the pinned host exchange buffer `host` followed by the
// release of `flag` = seq (zero-copy publish; counter: device scratch, zero between calls).
void reduce_publish(const float *partial, int slices, size_t n, bool bf16, void *local,
                    void *host, unsigned *flag, unsigned *counter, unsigned seq, hipStream_t);
// Zero-copy gather: waits for the peer's flag, then copies the peer's `words` 32-bit words from
// its host buffer to all + (1 - rank) * words (the local half is already at all + rank * words).
void gather_peer(const void *peer_host, void *all, size_t words, int rank,
                 const unsigned *peer_flag, unsigned seq, hipStream_t);

// Fused zero-copy all-gather step (one launch instead of reduce_publish + gather_peer): sums
// the slices (Fp32 / Bf16 payload; n = elements) or copies n local 32-bit words (Words) into
// `local` and the host exchange buffer, releases flag = seq, waits for the peer's flag and
// copies the peer's payload from peer_host to peer_dst.
enum class ExchangeMode { Fp32, Bf16, Words };
void exchange(const float *partial, int slices, size_t n, ExchangeMode, void *local, void *host,
              unsigned *flag, unsigned *counter, unsigned seq, const void *peer_host, void *peer_dst,
              const unsigned *peer_flag, hipStream_t);

// Element source of a transform: element e of row r is
// p[(e / half) * part + r * ld + offset + e % half]  (half = n, part = 0 for one plain matrix;
// half = n/2 for the rank-blocked result of a gather).
struct Source {
    const void *p = nullptr;
    int half = 0;
    size_t part = 0;
    int ld = 0, offset = 0;
};
// u = bf16( hadUt_n( SU (.) x' ) ), x' = bf16(x * rsqrt(mean(x^2) + eps) * w) when norm is
// non-null (the runtime's rms_norm rounding), else x' = x. x bf16, u bf16 [rows][ldu].
void input_transform(const Source &x, const BFloat16 *norm, float epsilon, const float *su, int n,
                     int rows, BFloat16 *u, int ldu, hipStream_t);
// q/k/v epilogue: all = gathered [2][rows][7168] fp32 rotated outputs (q 6144 | k 512 | v 512
// per rank), writes this rank's q heads [rows][6144], k/v heads [rows][512] as bf16 of
// SV (.) hadU(z).
void output_qkv(const float *all, int rows, int rank, const float *sv_q, const float *sv_k,
                const float *sv_v, BFloat16 *q, BFloat16 *k, BFloat16 *v, hipStream_t);
// o/down epilogue: all = gathered [2][rows][6144] fp32 rotated outputs; state[rows][12288] =
// bf16(state + bf16(SV (.) hadU(z))).
void output_residual(const float *all, int rows, const float *sv, BFloat16 *state, hipStream_t);
// gate/up epilogue: all = gathered [2][rows][28672] bf16 (gate 14336 | up 14336 per rank);
// h[rows][28672] = bf16(silu(g) * u), g = bf16(SVg (.) hadU(zg)), u = bf16(SVu (.) hadU(zu)).
void output_gateup(const BFloat16 *all, int rows, const float *sv_gate, const float *sv_up,
                   BFloat16 *h, hipStream_t);
// Standalone transforms for tests: y = bf16(hadU(x)) or hadUt (transpose), x fp32 [rows][n].
void hadamard_rows(const float *x, int n, int rows, bool transpose, float *y, hipStream_t);
// Dense decode for tests: w[rows][in] = bf16 LUT values of the shard (row-major, as bf16).
void decode_dense(const GpuLinear &, const uint32_t *lut, BFloat16 *w, hipStream_t);
// WMMA layout probe (tests): D = A * B for one 16x16x16 bf16 tile with A/B given row-major as
// bf16 [16][16] (A: M x K, B: K x N); returns D row-major fp32.
void wmma_probe(const BFloat16 *a, const BFloat16 *b, float *d, hipStream_t);
} // namespace behemoth::qtip
