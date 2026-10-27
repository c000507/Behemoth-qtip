// QTIP 2-bit HYB kernels for gfx1201 (Wave32). See include/behemoth/qtip_gpu.hpp and
// QTIP_RUNTIME_REPORT.md.
//
// Trellis decode. Every 16x16 tile is a 512-bit tail-biting bitstream of 32 words; state t
// (t = 0..127) is the 16-bit window starting at bit 4t, and its LUT entry is bits 6..15 of
// p = s(s+1) (bits 6..14 index the 512-entry tlut, bit 15 flips component 0). State t carries
// the two weights (row D*8 + A, columns C*8 + B*2 + {0,1}) with t = A*16 + B*4 + C*2 + D. The
// storage interleaves the 4 tiles of each 2x2 tile block at word granularity
// ([block][c][j1][i1]), so one 8-byte load at word c covers all four tiles.
//
// p mod 2^16 only needs the low 16 bits of the window, so windows are cut with a plain shift and
// may keep garbage above bit 15: v_mad_u32_u24(s, s, s) is still exact mod 2^16. The LUT lives in
// LDS as COPIES interleaved copies (entry e, copy c at word e*COPIES + c; lane l reads copy
// l % COPIES), which spreads the random lookups of a wave over all banks; with 16 copies the byte
// address is just (p & 0xffc0) | 4*(lane & 15).
#include "behemoth/qtip_gpu.hpp"
#include "device_numeric.hpp"
#include "qtip_tables.hpp"
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <tuple>

namespace behemoth::qtip {
namespace {
using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));
using bf16x8_t = __bf16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

__constant__ int8_t c_had12[144];
__constant__ int8_t c_had28[784];

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, a),
                                           __builtin_bit_cast(bf16x2_t, b), c, false);
}
template <int C> __device__ __forceinline__ uint32_t lut_address(uint32_t p, uint32_t lane_offset) {
    static_assert(C == 1 || C == 2 || C == 4 || C == 8 || C == 16, "LUT copies");
    constexpr int lg = C == 16 ? 4 : C == 8 ? 3 : C == 4 ? 2 : C == 2 ? 1 : 0;
    // entry e = (p >> 6) & 1023 at byte e * 4C + lane_offset (one v_and_or_b32 for C = 16; the
    // compiler otherwise splits it into and + add)
    uint32_t r;
    if constexpr (lg == 4)
        asm("v_and_or_b32 %0, %1, 0xffc0, %2" : "=v"(r) : "v"(p), "v"(lane_offset));
    else {
        const uint32_t t = p >> (4 - lg);
        asm("v_and_or_b32 %0, %1, %2, %3"
                     : "=v"(r)
                     : "v"(t), "i"(1023U << (2 + lg)), "v"(lane_offset));
    }
    return r;
}
__device__ __forceinline__ uint32_t lds_word(const uint32_t *base, uint32_t byte_offset) {
    return *reinterpret_cast<const uint32_t *>(reinterpret_cast<const char *>(base) + byte_offset);
}
__device__ __forceinline__ uint32_t lut_state(uint32_t s) {
    // s * s + s with 24-bit multiplicands: exact mod 2^16 even with garbage above bit 15
    uint32_t p;
    asm("v_mad_u32_u24 %0, %1, %1, %1" : "=v"(p) : "v"(s));
    return p;
}
template <int C> __device__ __forceinline__ void fill_table(uint32_t *table, const uint32_t *lut) {
    for (int i = int(threadIdx.x); i < 1024 * C; i += int(blockDim.x))
        table[i] = lut[i / C];
    __syncthreads();
}

// ---------------------------------------------------------------- GEMV (rows 1..4)
// One wave per (block-row of 32 outputs, K slice); lane w owns word w of each tile, i.e. states
// 4w..4w+3: rows A = w/4 and A + 8, column pairs 2B and 8 + 2B (B = w % 4) of every tile.
template <int R, int C, int WAVES>
__global__ __launch_bounds__(WAVES * 32) void gemv_kernel(
    const uint2 *__restrict__ trellis, int block_rows, int n2, int per, int slices,
    const uint32_t *__restrict__ lut, const uint32_t *__restrict__ x, int ldx2, int rows,
    float *__restrict__ partial, int out_rows) {
    __shared__ uint32_t table[1024 * C];
    fill_table<C>(table, lut);
    // wave-uniform work indices live in SGPRs, so the loads use scalar base + 32-bit lane offsets
    const int lane = int(threadIdx.x) & 31,
              wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x) >> 5);
    const int A = lane >> 2, B = lane & 3;
    const uint32_t offset = uint32_t(lane % C) * 4;
    const int items = block_rows * slices;
    for (int item = int(blockIdx.x) * WAVES + wave; item < items; item += int(gridDim.x) * WAVES) {
        const int br = item / slices, s = item - br * slices;
        const int j_begin = s * per, j_end = min(n2, j_begin + per);
        float acc[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[r][q] = 0.f;
        const uint2 *tp = trellis + size_t(br) * size_t(n2) * 32;
        // Two blocks in flight per wave (software prefetch): the loads of block j0 + 2 are issued
        // before block j0 is decoded. Out-of-range prefetches are clamped to the last block.
        struct Buf {
            uint2 d0, d1;
            uint32_t xv[R][2][2];
        };
        const uint32_t lane_next = uint32_t((lane + 1) & 31);
        auto load = [&](int j0, Buf &b) {
            j0 = min(j0, j_end - 1);
            const uint2 *blk = tp + size_t(j0) * 32;
            b.d0 = blk[uint32_t(lane)];
            b.d1 = blk[lane_next];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t *xr = x + size_t(r) * size_t(ldx2) + size_t(16 * j0);
#pragma unroll
                for (int j1 = 0; j1 < 2; ++j1)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc)
                        b.xv[r][j1][cc] = xr[uint32_t(B + 8 * j1 + 4 * cc)];
            }
        };
        auto compute = [&](const Buf &b) {
#pragma unroll
            for (int j1 = 0; j1 < 2; ++j1) {
                const uint32_t a = j1 ? b.d0.y : b.d0.x, c = j1 ? b.d1.y : b.d1.x;
#pragma unroll
                for (int i1 = 0; i1 < 2; ++i1) {
                    const uint32_t u = __builtin_amdgcn_perm(a, c, i1 ? 0x07060302U : 0x05040100U);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t st = u >> (16 - 4 * k);
                        const uint32_t w = lds_word(table, lut_address<C>(lut_state(st), offset));
                        const int cc = k >> 1, D = k & 1;
#pragma unroll
                        for (int r = 0; r < R; ++r)
                            acc[r][i1 * 2 + D] = dot2(w, b.xv[r][j1][cc], acc[r][i1 * 2 + D]);
                    }
                }
            }
        };
        if (j_begin < j_end) {
            Buf b0, b1;
            load(j_begin, b0);
            load(j_begin + 1, b1);
            for (int j0 = j_begin; j0 < j_end; j0 += 2) {
                const Buf c0 = b0;
                load(j0 + 2, b0);
                compute(c0);
                if (j0 + 1 < j_end) {
                    const Buf c1 = b1;
                    load(j0 + 3, b1);
                    compute(c1);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                acc[r][q] += __shfl_xor(acc[r][q], 1);
                acc[r][q] += __shfl_xor(acc[r][q], 2);
            }
        // lane B stores accumulator B: output row (B / 2) * 16 + (B % 2) * 8 + A of the block-row
        const int row = br * 32 + (B >> 1) * 16 + (B & 1) * 8 + A;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = B == 0 ? acc[r][0] : B == 1 ? acc[r][1] : B == 2 ? acc[r][2] : acc[r][3];
            partial[(size_t(s) * size_t(rows) + size_t(r)) * size_t(out_rows) + size_t(row)] = v;
        }
    }
}

// ---------------------------------------------------------------- WMMA (rows > 4)
// One wave per (block-row, K slice, pass of 16*MT batch rows). Lane l decodes the A fragment of
// row l % 16, K half l / 16 of each tile (4 states with words 4q + B, shift 8*(l/16) + 4*(r/8)),
// the batch rows come in as B fragments, v_wmma_f32_16x16x16_bf16 accumulates in fp32.
template <int MT, int C, int WAVES>
__global__ __launch_bounds__(WAVES * 32) void wmma_kernel(
    const uint2 *__restrict__ trellis, int block_rows, int n2, int per, int slices,
    const uint32_t *__restrict__ lut, const uint16_t *__restrict__ x, int ldx, int rows,
    float *__restrict__ partial, int out_rows) {
    __shared__ uint32_t table[1024 * C];
    fill_table<C>(table, lut);
    const int lane = int(threadIdx.x) & 31,
              wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x) >> 5);
    const int rr = lane & 15, half = lane >> 4;
    const int q = rr & 7;
    const uint32_t shift = uint32_t(16 - (8 * half + 4 * (rr >> 3)));
    const uint32_t offset = uint32_t(lane % C) * 4;
    const int passes = (rows + 16 * MT - 1) / (16 * MT);
    const int items = block_rows * slices * passes;
    for (int item = int(blockIdx.x) * WAVES + wave; item < items; item += int(gridDim.x) * WAVES) {
        const int pass = item % passes, rest = item / passes;
        const int s = rest % slices, br = rest / slices;
        const int j_begin = s * per, j_end = min(n2, j_begin + per);
        f32x8_t acc[MT][2];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int i1 = 0; i1 < 2; ++i1)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    acc[m][i1][i] = 0.f;
        const uint2 *tp = trellis + size_t(br) * size_t(n2) * 32;
        const int row_base = pass * 16 * MT;
        const uint16_t *xrow[MT];
        bool valid[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int n = row_base + m * 16 + rr;
            valid[m] = n < rows;
            xrow[m] = x + size_t(valid[m] ? n : 0) * size_t(ldx) + size_t(8 * half);
        }
        struct Buf {
            uint32_t d[5][2]; // words c = 4q .. 4q+4 of the j1 tiles (lo: i1 = 0, hi: i1 = 1)
            uint4 bx[MT][2];
        };
        auto load = [&](int j0, Buf &b) {
            j0 = min(j0, j_end - 1);
            const uint2 *blk = tp + size_t(j0) * 32;
            const uint4 *t4 = reinterpret_cast<const uint4 *>(blk + 4 * q);
            const uint4 l0 = t4[0], l1 = t4[1];
            const uint2 l2 = blk[(4 * q + 4) & 31];
            b.d[0][0] = l0.x;
            b.d[0][1] = l0.y;
            b.d[1][0] = l0.z;
            b.d[1][1] = l0.w;
            b.d[2][0] = l1.x;
            b.d[2][1] = l1.y;
            b.d[3][0] = l1.z;
            b.d[3][1] = l1.w;
            b.d[4][0] = l2.x;
            b.d[4][1] = l2.y;
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int j1 = 0; j1 < 2; ++j1) {
                    const uint4 v = *reinterpret_cast<const uint4 *>(xrow[m] + size_t(32 * j0 + 16 * j1));
                    b.bx[m][j1] = valid[m] ? v : make_uint4(0, 0, 0, 0);
                }
        };
        auto compute = [&](const Buf &b) {
#pragma unroll
            for (int j1 = 0; j1 < 2; ++j1)
#pragma unroll
                for (int i1 = 0; i1 < 2; ++i1) {
                    uint32_t f[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t u = __builtin_amdgcn_perm(b.d[k][j1], b.d[k + 1][j1],
                                                                 i1 ? 0x07060302U : 0x05040100U);
                        f[k] = lds_word(table, lut_address<C>(lut_state(u >> shift), offset));
                    }
                    const bf16x8_t af = __builtin_bit_cast(bf16x8_t, make_uint4(f[0], f[1], f[2], f[3]));
#pragma unroll
                    for (int m = 0; m < MT; ++m)
                        acc[m][i1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            af, __builtin_bit_cast(bf16x8_t, b.bx[m][j1]), acc[m][i1]);
                }
        };
        if (j_begin < j_end) {
            Buf nb;
            load(j_begin, nb);
            for (int j0 = j_begin; j0 < j_end; ++j0) {
                const Buf cb = nb;
                load(j0 + 1, nb);
                compute(cb);
            }
        }
        // D: lane l holds batch row l % 16, output rows 8 * (l / 16) + i of the tile-row.
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (!valid[m])
                continue;
            const int n = row_base + m * 16 + rr;
#pragma unroll
            for (int i1 = 0; i1 < 2; ++i1) {
                float *dst = partial + (size_t(s) * size_t(rows) + size_t(n)) * size_t(out_rows) +
                             size_t(br * 32 + i1 * 16 + 8 * half);
                *reinterpret_cast<float4 *>(dst) =
                    make_float4(acc[m][i1][0], acc[m][i1][1], acc[m][i1][2], acc[m][i1][3]);
                *reinterpret_cast<float4 *>(dst + 4) =
                    make_float4(acc[m][i1][4], acc[m][i1][5], acc[m][i1][6], acc[m][i1][7]);
            }
        }
    }
}

// ---------------------------------------------------------------- dense decode
// grid (ceil(n2 / 8), block_rows), 256 threads. Wave w decodes 2x2-tile block j0 = 8 * x + w of
// block-row br (lane l: word l of the 4 tiles) into an LDS tile of 32 rows x 256 columns; the
// block then writes the tile with coalesced 16-byte stores (prefill streams the whole model
// through this kernel, so it runs at write bandwidth).
constexpr int decode_tile_words = 128 + 4; // 256 bf16 columns as words, padded
__global__ __launch_bounds__(256) void decode_dense_kernel(const uint2 *__restrict__ trellis, int n2,
                                                           const uint32_t *__restrict__ lut,
                                                           uint16_t *__restrict__ w, int in) {
    __shared__ uint32_t table[1024];
    __shared__ uint32_t tile[32 * decode_tile_words];
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
    for (int i = tid; i < 1024; i += 256)
        table[i] = lut[i];
    __syncthreads();
    const int br = int(blockIdx.y), j0 = int(blockIdx.x) * 8 + wave;
    if (j0 < n2) {
        const uint2 *tp = trellis + (size_t(br) * size_t(n2) + size_t(j0)) * 32;
        const uint2 d0 = tp[lane], d1 = tp[(lane + 1) & 31];
        const int A = lane >> 2, B = lane & 3;
#pragma unroll
        for (int j1 = 0; j1 < 2; ++j1) {
            const uint32_t a = j1 ? d0.y : d0.x, b = j1 ? d1.y : d1.x;
#pragma unroll
            for (int i1 = 0; i1 < 2; ++i1) {
                const uint32_t u = __builtin_amdgcn_perm(a, b, i1 ? 0x07060302U : 0x05040100U);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t p = lut_state(u >> (16 - 4 * k));
                    const int cc = k >> 1, D = k & 1;
                    const int row = i1 * 16 + D * 8 + A;
                    // columns 16 j1 + 8 cc + 2 B (+1) of block j0: word 8 j1 + 4 cc + B
                    tile[row * decode_tile_words + wave * 16 + 8 * j1 + 4 * cc + B] =
                        table[(p >> 6) & 1023U];
                }
            }
        }
    }
    __syncthreads();
    const int words = min(8, n2 - int(blockIdx.x) * 8) * 16; // valid words per tile row
    for (int idx = tid; idx < 32 * 32; idx += 256) {
        const int r = idx >> 5, c4 = idx & 31;
        if (c4 * 4 >= words)
            continue;
        const uint32_t *t = tile + r * decode_tile_words + c4 * 4;
        *reinterpret_cast<uint4 *>(w + size_t(br * 32 + r) * size_t(in) + size_t(blockIdx.x) * 256 +
                                   size_t(c4) * 8) = make_uint4(t[0], t[1], t[2], t[3]);
    }
}
__global__ void wmma_probe_kernel(const uint16_t *a, const uint16_t *b, float *d) {
    const int lane = int(threadIdx.x), rr = lane & 15, half = lane >> 4;
    uint16_t av[8], bv[8];
    for (int e = 0; e < 8; ++e) {
        av[e] = a[rr * 16 + 8 * half + e];   // A row rr, K 8*half + e
        bv[e] = b[(8 * half + e) * 16 + rr]; // B column rr, K 8*half + e
    }
    bf16x8_t fa = __builtin_bit_cast(bf16x8_t, av), fb = __builtin_bit_cast(bf16x8_t, bv);
    f32x8_t c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(fa, fb, c);
    for (int i = 0; i < 8; ++i)
        d[(8 * half + i) * 16 + rr] = c[i];
}

// ---------------------------------------------------------------- reduce / publish / gather
template <bool BF16>
__global__ void reduce_publish_kernel(const float4 *__restrict__ partial, int slices, int n4,
                                      void *__restrict__ local, void *__restrict__ host,
                                      unsigned *flag, unsigned *counter, unsigned seq) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n4) {
        float4 s = partial[i];
        for (int k = 1; k < slices; ++k) {
            const float4 v = partial[size_t(k) * size_t(n4) + size_t(i)];
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
        }
        if constexpr (BF16) {
            const uint2 o = make_uint2(uint32_t(device::bf16(s.x)) | uint32_t(device::bf16(s.y)) << 16,
                                       uint32_t(device::bf16(s.z)) | uint32_t(device::bf16(s.w)) << 16);
            static_cast<uint2 *>(local)[i] = o;
            if (host)
                static_cast<uint2 *>(host)[i] = o;
        } else {
            static_cast<float4 *>(local)[i] = s;
            if (host)
                static_cast<float4 *>(host)[i] = s;
        }
    }
    if (!host)
        return;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = atomicAdd(counter, 1u) + 1;
        if (done == gridDim.x) {
            __threadfence_system();
            __hip_atomic_store(flag, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            *counter = 0;
        }
    }
}
__global__ void gather_peer_kernel(const uint4 *__restrict__ peer, uint4 *__restrict__ all, int n4,
                                   int rank, const unsigned *peer_flag, unsigned seq) {
    if (threadIdx.x == 0)
        while (__hip_atomic_load(peer_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq)
            __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n4)
        all[size_t(1 - rank) * size_t(n4) + size_t(i)] = peer[i];
}

// Fused zero-copy exchange: publish (sum of slices or a copy of local words) + flag release,
// then wait for the peer's flag and copy its half. Grid-stride with at most 64 blocks so every
// block is resident while it spins (no deadlock against blocks that still have to publish).
// MODE 0: partial fp32 -> fp32, 1: partial -> bf16, 2: local words (n4 = uint4 count).
template <int MODE>
__global__ __launch_bounds__(256) void exchange_kernel(const float4 *__restrict__ partial, int slices,
                                                       int n4, void *__restrict__ local,
                                                       void *__restrict__ host, unsigned *flag,
                                                       unsigned *counter, unsigned seq,
                                                       const void *__restrict__ peer_host,
                                                       void *__restrict__ peer_dst,
                                                       const unsigned *peer_flag) {
    const int stride = int(gridDim.x) * int(blockDim.x);
    for (int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x); i < n4; i += stride) {
        if constexpr (MODE == 2) {
            static_cast<uint4 *>(host)[i] = static_cast<const uint4 *>(local)[i];
        } else {
            float4 s = partial[i];
            for (int k = 1; k < slices; ++k) {
                const float4 v = partial[size_t(k) * size_t(n4) + size_t(i)];
                s.x += v.x;
                s.y += v.y;
                s.z += v.z;
                s.w += v.w;
            }
            if constexpr (MODE == 1) {
                const uint2 o = make_uint2(uint32_t(device::bf16(s.x)) | uint32_t(device::bf16(s.y)) << 16,
                                           uint32_t(device::bf16(s.z)) | uint32_t(device::bf16(s.w)) << 16);
                static_cast<uint2 *>(local)[i] = o;
                static_cast<uint2 *>(host)[i] = o;
            } else {
                static_cast<float4 *>(local)[i] = s;
                static_cast<float4 *>(host)[i] = s;
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned done = atomicAdd(counter, 1u) + 1;
        if (done == gridDim.x) {
            __threadfence_system();
            __hip_atomic_store(flag, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            *counter = 0;
        }
        while (__hip_atomic_load(peer_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq)
            __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    if constexpr (MODE == 1) {
        for (int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x); i < n4; i += stride)
            static_cast<uint2 *>(peer_dst)[i] = static_cast<const uint2 *>(peer_host)[i];
    } else {
        for (int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x); i < n4; i += stride)
            static_cast<uint4 *>(peer_dst)[i] = static_cast<const uint4 *>(peer_host)[i];
    }
}

// ---------------------------------------------------------------- Hadamard transforms
// 256 threads hold the 1024 values of one chunk, 4 consecutive positions each; natural-order
// (Sylvester) WHT: in-thread bits 0-1, lane shuffles for bits 2-6, LDS for bits 7-9.
__device__ __forceinline__ void fwht1024(float v[4], float *scratch) {
    {
        const float a = v[0], b = v[1], c = v[2], d = v[3];
        v[0] = a + b;
        v[1] = a - b;
        v[2] = c + d;
        v[3] = c - d;
    }
    {
        const float a = v[0], b = v[1], c = v[2], d = v[3];
        v[0] = a + c;
        v[1] = b + d;
        v[2] = a - c;
        v[3] = b - d;
    }
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor(v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
#pragma unroll
    for (int m = 1; m < 8; m <<= 1) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
            scratch[4 * tid + i] = v[i];
        __syncthreads();
        const int partner = tid ^ (32 * m);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = scratch[4 * partner + i];
            v[i] = (wave & m) ? o - v[i] : v[i] + o;
        }
    }
}
template <int K> __device__ __forceinline__ int had_sign(int r, int c) {
    if constexpr (K == 12)
        return c_had12[r * 12 + c];
    else
        return c_had28[r * 28 + c];
}
template <class T> __device__ __forceinline__ float load_elem(const T *p, size_t i);
template <> __device__ __forceinline__ float load_elem<float>(const float *p, size_t i) {
    return p[i];
}
template <> __device__ __forceinline__ float load_elem<uint16_t>(const uint16_t *p, size_t i) {
    return device::bf16(p[i]);
}
// Four consecutive elements e..e+3 (e % 4 == 0, never crossing `half`).
template <class T>
__device__ __forceinline__ void load4(const Source &s, int row, int e, float out[4]) {
    const T *p = static_cast<const T *>(s.p);
    const size_t i = size_t(e / s.half) * s.part + size_t(row) * size_t(s.ld) + size_t(s.offset) +
                     size_t(e % s.half);
    if constexpr (sizeof(T) == 4) {
        const float4 v = *reinterpret_cast<const float4 *>(p + i);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
    } else {
        const uint2 v = *reinterpret_cast<const uint2 *>(p + i);
        out[0] = device::bf16(uint16_t(v.x & 0xffffU));
        out[1] = device::bf16(uint16_t(v.x >> 16));
        out[2] = device::bf16(uint16_t(v.y & 0xffffU));
        out[3] = device::bf16(uint16_t(v.y >> 16));
    }
}
// v[i] = sum_k M[k2][k] * src(k * 1024 + 4 tid + i) with M = hadK (TRANSPOSE: hadK^T).
template <int K, bool TRANSPOSE, class T>
__device__ __forceinline__ void mix(const Source &s, int row, int k2, float v[4]) {
    const int j = 4 * int(threadIdx.x);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = 0.f;
    if constexpr (K == 1) {
        load4<T>(s, row, j, v);
    } else {
        // all K loads in flight at once (the transforms are L2-latency bound at decode sizes)
        float t[K][4];
#pragma unroll
        for (int k = 0; k < K; ++k)
            load4<T>(s, row, k * 1024 + j, t[k]);
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float sign = float(TRANSPOSE ? had_sign<K>(k, k2) : had_sign<K>(k2, k));
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] += sign * t[k][i];
        }
    }
}
__device__ __forceinline__ float block_sum(float v, float *red) {
    for (int m = 16; m; m >>= 1)
        v += __shfl_xor(v, m);
    __syncthreads();
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
    for (int w = 0; w < int(blockDim.x) / 32; ++w)
        s += red[w];
    return s;
}
// Output chunk k2 of row `row` of the input transform (one 256-thread block).
template <int K, bool NORM>
__device__ __forceinline__ void input_transform_chunk(const Source &src, const uint16_t *norm,
                                                      float epsilon, const float *su, uint16_t *u,
                                                      int ldu, int k2, int row, float *scratch,
                                                      float *red) {
    const int j = 4 * int(threadIdx.x);
    constexpr int n = K * 1024;
    // The block's mix inputs (4 positions of every chunk) are the whole row, so the rms of the
    // row comes from the same registers (no second pass over x).
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    float tk[K][4];
#pragma unroll
    for (int k = 0; k < K; ++k)
        load4<uint16_t>(src, row, k * 1024 + j, tk[k]);
    float factor = 1.f;
    if constexpr (NORM) {
        float ss = 0;
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ss += tk[k][i] * tk[k][i];
        factor = rsqrtf(block_sum(ss, red) / float(n) + epsilon);
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int e = k * 1024 + j;
        const float *t = tk[k];
        const float4 sg = *reinterpret_cast<const float4 *>(su + e);
        const float sgn[4] = {sg.x, sg.y, sg.z, sg.w};
        float w[4] = {1.f, 1.f, 1.f, 1.f};
        if constexpr (NORM) {
            const uint2 wv = *reinterpret_cast<const uint2 *>(norm + e);
            w[0] = device::bf16(uint16_t(wv.x & 0xffffU));
            w[1] = device::bf16(uint16_t(wv.x >> 16));
            w[2] = device::bf16(uint16_t(wv.y & 0xffffU));
            w[3] = device::bf16(uint16_t(wv.y >> 16));
        }
        const float sign = K == 1 ? 1.f : float(had_sign<K>(k, k2)); // hadK^T[k2][k]
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = NORM ? device::bf16(device::bf16(t[i] * factor * w[i])) : t[i];
            v[i] += sign * (sgn[i] * x);
        }
    }
    fwht1024(v, scratch);
    const float scale = rsqrtf(float(n));
    uint16_t *dst = u + size_t(row) * size_t(ldu) + size_t(k2 * 1024 + j);
    *reinterpret_cast<uint2 *>(dst) =
        make_uint2(uint32_t(device::bf16(v[0] * scale)) | uint32_t(device::bf16(v[1] * scale)) << 16,
                   uint32_t(device::bf16(v[2] * scale)) | uint32_t(device::bf16(v[3] * scale)) << 16);
}
// grid (K, rows), 256 threads.
template <int K, bool NORM>
__global__ __launch_bounds__(256) void input_transform_kernel(Source src, const uint16_t *norm,
                                                              float epsilon, const float *su,
                                                              uint16_t *u, int ldu) {
    __shared__ float scratch[1024];
    __shared__ float red[8];
    input_transform_chunk<K, NORM>(src, norm, epsilon, su, u, ldu, int(blockIdx.x), int(blockIdx.y),
                                   scratch, red);
}
// grid (8, rows): blocks 0..5 = this rank's q chunks, 6 = k, 7 = v.
template <class T>
__global__ __launch_bounds__(256) void output_qkv_kernel(const T *all, int rows, int rank,
                                                         const float *sv_q, const float *sv_k,
                                                         const float *sv_v, uint16_t *q, uint16_t *k,
                                                         uint16_t *v) {
    __shared__ float scratch[1024];
    const int b = int(blockIdx.x), row = int(blockIdx.y), j = 4 * int(threadIdx.x);
    const size_t part = size_t(rows) * 7168;
    float x[4];
    if (b < 6) {
        const int k2 = 6 * rank + b;
        Source s{all, 6144, part, 7168, 0};
        mix<12, false, T>(s, row, k2, x);
        fwht1024(x, scratch);
        const float scale = rsqrtf(12288.f);
        const int e = k2 * 1024 + j;
        uint16_t *dst = q + size_t(row) * 6144 + size_t(b * 1024 + j);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            dst[i] = device::bf16(x[i] * scale * sv_q[e + i]);
        return;
    }
    const bool is_k = b == 6;
    Source s{all, 512, part, 7168, is_k ? 6144 : 6656};
    mix<1, false, T>(s, row, 0, x);
    fwht1024(x, scratch);
    const float scale = 1.f / 32.f; // 1/sqrt(1024)
    const float *sv = is_k ? sv_k : sv_v;
    uint16_t *out = is_k ? k : v;
    if (j >= 512 * rank && j < 512 * rank + 512) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[size_t(row) * 512 + size_t(j - 512 * rank + i)] = device::bf16(x[i] * scale * sv[j + i]);
    }
}
// Residual epilogue of output chunk k2 of row `row` (one 256-thread block).
template <class T>
__device__ __forceinline__ void output_residual_chunk(const T *all, int rows, const float *sv,
                                                      uint16_t *state, int k2, int row, float *scratch) {
    const int j = 4 * int(threadIdx.x);
    Source s{all, 6144, size_t(rows) * 6144, 6144, 0};
    float x[4];
    mix<12, false, T>(s, row, k2, x);
    fwht1024(x, scratch);
    const float scale = rsqrtf(12288.f);
    const int e = k2 * 1024 + j;
    uint16_t *st = state + size_t(row) * 12288 + size_t(e);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float t = device::bf16(device::bf16(x[i] * scale * sv[e + i]));
        st[i] = device::bf16(device::bf16(st[i]) + t);
    }
}
// grid (12, rows)
template <class T>
__global__ __launch_bounds__(256) void output_residual_kernel(const T *all, int rows,
                                                              const float *sv, uint16_t *state) {
    __shared__ float scratch[1024];
    output_residual_chunk<T>(all, rows, sv, state, int(blockIdx.x), int(blockIdx.y), scratch);
}
// grid (12, rows): the residual epilogue of chunk k2, a rendezvous of the row's 12 blocks
// (sync[row] counts arrivals across calls, a multiple of 12 between launches; the blocks of a row are
// dispatched consecutively, and rows <= residual_sync_rows keeps them co-resident), then the
// input transform of the updated row for output chunk k2. Both halves are the per-chunk kernels'
// code: results are identical to output_residual_kernel + input_transform_kernel<12, true>.
template <class T>
__global__ __launch_bounds__(256) void output_residual_input_kernel(
    const T *all, int rows, const float *sv, uint16_t *state, const uint16_t *norm, float epsilon,
    const float *su, uint16_t *u, int ldu, unsigned long long *sync) {
    __shared__ float scratch[1024];
    __shared__ float red[8];
    const int k2 = int(blockIdx.x), row = int(blockIdx.y);
    output_residual_chunk<T>(all, rows, sv, state, k2, row, scratch);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        // The row's counter only advances in launches that include the row (the row count
        // varies between steps), so each block derives its target from the value it saw: the
        // 12 arrivals of one launch fill one multiple of 12 (launches on the stream are ordered).
        const unsigned long long seen =
            __hip_atomic_fetch_add(sync + row, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned long long target = (seen / 12 + 1) * 12;
        while (__hip_atomic_load(sync + row, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target)
            __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); // every wave: fresh view of `state`
    input_transform_chunk<12, true>(Source{state, 12288, 0, 12288, 0}, norm, epsilon, su, u, ldu, k2,
                                    row, scratch, red);
}
// grid (28, rows)
__global__ __launch_bounds__(256) void output_gateup_kernel(const uint16_t *all, int rows,
                                                            const float *sv_g, const float *sv_u,
                                                            uint16_t *h) {
    __shared__ float scratch[1024];
    const int k2 = int(blockIdx.x), row = int(blockIdx.y), j = 4 * int(threadIdx.x);
    const size_t part = size_t(rows) * 28672;
    float g[4], u[4];
    mix<28, false, uint16_t>(Source{all, 14336, part, 28672, 0}, row, k2, g);
    mix<28, false, uint16_t>(Source{all, 14336, part, 28672, 14336}, row, k2, u);
    fwht1024(g, scratch);
    fwht1024(u, scratch);
    const float scale = rsqrtf(28672.f);
    const int e = k2 * 1024 + j;
    uint16_t out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float gv = device::bf16(device::bf16(g[i] * scale * sv_g[e + i]));
        const float uv = device::bf16(device::bf16(u[i] * scale * sv_u[e + i]));
        out[i] = device::bf16(gv / (1 + expf(-gv)) * uv);
    }
    *reinterpret_cast<uint2 *>(h + size_t(row) * 28672 + size_t(e)) =
        make_uint2(uint32_t(out[0]) | uint32_t(out[1]) << 16, uint32_t(out[2]) | uint32_t(out[3]) << 16);
}
// Tests: plain hadU / hadUt of fp32 rows. grid (K, rows)
template <int K, bool TRANSPOSE>
__global__ __launch_bounds__(256) void hadamard_kernel(const float *x, int rows, float *y) {
    __shared__ float scratch[1024];
    const int k2 = int(blockIdx.x), row = int(blockIdx.y), j = 4 * int(threadIdx.x);
    constexpr int n = K * 1024;
    float v[4];
    mix<K, TRANSPOSE, float>(Source{x, n, 0, n, 0}, row, k2, v);
    fwht1024(v, scratch);
    const float scale = rsqrtf(float(n));
    for (int i = 0; i < 4; ++i)
        y[size_t(row) * n + size_t(k2 * 1024 + j + i)] = v[i] * scale;
}

// ---------------------------------------------------------------- row-batched transforms
// One 256-thread block per row over the whole vector (PREFILL_SCHED_REPORT.md, item 3): thread
// t holds positions 4t..4t+3 of every 1024-chunk in registers, so each input element is read
// once (the per-chunk kernels above read the whole row once per output chunk, K = 12 / 28 times)
// and the hadK mix of output chunk k2 is computed from registers. The arithmetic per output
// element (mix order, fwht1024, scaling and rounding) is exactly that of the per-chunk kernels:
// the results are bit-identical.
template <int K, bool TRANSPOSE>
__device__ __forceinline__ void mix_regs(const float (&t)[K][4], int k2, float v[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float sign =
            K == 1 ? 1.f : float(TRANSPOSE ? had_sign<K>(k, k2) : had_sign<K>(k2, k));
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] += sign * t[k][i];
    }
}
template <class T, int K>
__device__ __forceinline__ void load_row(const Source &s, int row, float (&t)[K][4]) {
    const int j = 4 * int(threadIdx.x);
#pragma unroll
    for (int k = 0; k < K; ++k)
        load4<T>(s, row, k * 1024 + j, t[k]);
}
// u = bf16(hadUt(SU (.) x')) from registers t (the row's values, see input_transform_kernel):
// rms norm (NORM), SU, then the K output chunks.
template <int K, bool NORM>
__device__ __forceinline__ void input_transform_regs(float (&t)[K][4], const uint16_t *norm,
                                                     float epsilon, const float *su, uint16_t *u,
                                                     float *scratch, float *red) {
    constexpr int n = K * 1024;
    const int j = 4 * int(threadIdx.x);
    float factor = 1.f;
    if constexpr (NORM) {
        float ss = 0;
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ss += t[k][i] * t[k][i];
        factor = rsqrtf(block_sum(ss, red) / float(n) + epsilon);
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int e = k * 1024 + j;
        const float4 sg = *reinterpret_cast<const float4 *>(su + e);
        const float sgn[4] = {sg.x, sg.y, sg.z, sg.w};
        float w[4] = {1.f, 1.f, 1.f, 1.f};
        if constexpr (NORM) {
            const uint2 wv = *reinterpret_cast<const uint2 *>(norm + e);
            w[0] = device::bf16(uint16_t(wv.x & 0xffffU));
            w[1] = device::bf16(uint16_t(wv.x >> 16));
            w[2] = device::bf16(uint16_t(wv.y & 0xffffU));
            w[3] = device::bf16(uint16_t(wv.y >> 16));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = NORM ? device::bf16(device::bf16(t[k][i] * factor * w[i])) : t[k][i];
            t[k][i] = sgn[i] * x;
        }
    }
    const float scale = rsqrtf(float(n));
    for (int k2 = 0; k2 < K; ++k2) {
        float v[4];
        mix_regs<K, true>(t, k2, v);
        fwht1024(v, scratch);
        *reinterpret_cast<uint2 *>(u + size_t(k2 * 1024 + j)) = make_uint2(
            uint32_t(device::bf16(v[0] * scale)) | uint32_t(device::bf16(v[1] * scale)) << 16,
            uint32_t(device::bf16(v[2] * scale)) | uint32_t(device::bf16(v[3] * scale)) << 16);
    }
}
// grid (rows)
template <int K, bool NORM>
__global__ __launch_bounds__(256) void input_transform_rows_kernel(Source src, const uint16_t *norm,
                                                                   float epsilon, const float *su,
                                                                   uint16_t *u, int ldu) {
    __shared__ float scratch[1024];
    __shared__ float red[8];
    const int row = int(blockIdx.x);
    float t[K][4];
    load_row<uint16_t, K>(src, row, t);
    input_transform_regs<K, NORM>(t, norm, epsilon, su, u + size_t(row) * size_t(ldu), scratch, red);
}
// grid (rows): this rank's 6 q chunks from one load of the row, then k and v.
template <class T>
__global__ __launch_bounds__(256) void output_qkv_rows_kernel(const T *all, int rows, int rank,
                                                              const float *sv_q, const float *sv_k,
                                                              const float *sv_v, uint16_t *q,
                                                              uint16_t *k, uint16_t *v) {
    __shared__ float scratch[1024];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    const size_t part = size_t(rows) * 7168;
    {
        float t[12][4];
        load_row<T, 12>(Source{all, 6144, part, 7168, 0}, row, t);
        const float scale = rsqrtf(12288.f);
        for (int b = 0; b < 6; ++b) {
            const int k2 = 6 * rank + b;
            float x[4];
            mix_regs<12, false>(t, k2, x);
            fwht1024(x, scratch);
            const int e = k2 * 1024 + j;
            uint16_t *dst = q + size_t(row) * 6144 + size_t(b * 1024 + j);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                dst[i] = device::bf16(x[i] * scale * sv_q[e + i]);
        }
    }
    for (int b = 6; b < 8; ++b) {
        const bool is_k = b == 6;
        float x[4];
        mix<1, false, T>(Source{all, 512, part, 7168, is_k ? 6144 : 6656}, row, 0, x);
        fwht1024(x, scratch);
        const float scale = 1.f / 32.f;
        const float *sv = is_k ? sv_k : sv_v;
        uint16_t *out = is_k ? k : v;
        if (j >= 512 * rank && j < 512 * rank + 512) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out[size_t(row) * 512 + size_t(j - 512 * rank + i)] =
                    device::bf16(x[i] * scale * sv[j + i]);
        }
    }
}
// grid (rows): residual epilogue; FUSE: then the next input transform (rms norm, SU, hadUt
// 12288) of the updated state, from registers.
template <bool FUSE, class T>
__global__ __launch_bounds__(256) void output_residual_rows_kernel(const T *all, int rows,
                                                                   const float *sv, uint16_t *state,
                                                                   const uint16_t *norm, float epsilon,
                                                                   const float *su, uint16_t *u,
                                                                   int ldu) {
    __shared__ float scratch[1024];
    __shared__ float red[8];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    float t[12][4], s[12][4];
    load_row<T, 12>(Source{all, 6144, size_t(rows) * 6144, 6144, 0}, row, t);
    const float scale = rsqrtf(12288.f);
    // fully unrolled: s[k2] must stay in registers
#pragma unroll
    for (int k2 = 0; k2 < 12; ++k2) {
        float x[4];
        mix_regs<12, false>(t, k2, x);
        fwht1024(x, scratch);
        const int e = k2 * 1024 + j;
        uint16_t *st = state + size_t(row) * 12288 + size_t(e);
        const uint2 old = *reinterpret_cast<const uint2 *>(st);
        const uint16_t o[4] = {uint16_t(old.x & 0xffffU), uint16_t(old.x >> 16),
                               uint16_t(old.y & 0xffffU), uint16_t(old.y >> 16)};
        uint16_t r[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float y = device::bf16(device::bf16(x[i] * scale * sv[e + i]));
            r[i] = device::bf16(device::bf16(o[i]) + y);
        }
        *reinterpret_cast<uint2 *>(st) =
            make_uint2(uint32_t(r[0]) | uint32_t(r[1]) << 16, uint32_t(r[2]) | uint32_t(r[3]) << 16);
        if constexpr (FUSE) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                s[k2][i] = device::bf16(r[i]);
        }
    }
    if constexpr (FUSE)
        input_transform_regs<12, true>(s, norm, epsilon, su, u + size_t(row) * size_t(ldu), scratch,
                                       red);
}
// grid (rows): gate pass (g -> h as bf16), then up pass reading g back from h.
__global__ __launch_bounds__(256) void output_gateup_rows_kernel(const uint16_t *all, int rows,
                                                                 const float *sv_g, const float *sv_u,
                                                                 uint16_t *h) {
    __shared__ float scratch[1024];
    const int row = int(blockIdx.x), j = 4 * int(threadIdx.x);
    const size_t part = size_t(rows) * 28672;
    const float scale = rsqrtf(28672.f);
    uint16_t *hr = h + size_t(row) * 28672;
    float t[28][4];
    load_row<uint16_t, 28>(Source{all, 14336, part, 28672, 0}, row, t);
    for (int k2 = 0; k2 < 28; ++k2) {
        float g[4];
        mix_regs<28, false>(t, k2, g);
        fwht1024(g, scratch);
        const int e = k2 * 1024 + j;
        uint16_t gb[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            gb[i] = device::bf16(g[i] * scale * sv_g[e + i]); // = bits of output_gateup's gv
        *reinterpret_cast<uint2 *>(hr + e) =
            make_uint2(uint32_t(gb[0]) | uint32_t(gb[1]) << 16, uint32_t(gb[2]) | uint32_t(gb[3]) << 16);
    }
    load_row<uint16_t, 28>(Source{all, 14336, part, 28672, 14336}, row, t);
    for (int k2 = 0; k2 < 28; ++k2) {
        float u[4];
        mix_regs<28, false>(t, k2, u);
        fwht1024(u, scratch);
        const int e = k2 * 1024 + j;
        const uint2 gv2 = *reinterpret_cast<const uint2 *>(hr + e); // this thread's own store
        const uint16_t gb[4] = {uint16_t(gv2.x & 0xffffU), uint16_t(gv2.x >> 16),
                                uint16_t(gv2.y & 0xffffU), uint16_t(gv2.y >> 16)};
        uint16_t out[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float gv = device::bf16(gb[i]);
            const float uv = device::bf16(device::bf16(u[i] * scale * sv_u[e + i]));
            out[i] = device::bf16(gv / (1 + expf(-gv)) * uv);
        }
        *reinterpret_cast<uint2 *>(hr + e) =
            make_uint2(uint32_t(out[0]) | uint32_t(out[1]) << 16, uint32_t(out[2]) | uint32_t(out[3]) << 16);
    }
}

std::once_flag tables_once;
void upload_tables() {
    std::call_once(tables_once, [] {
        int count = 0, current = 0;
        HIP_CHECK(hipGetDeviceCount(&count));
        HIP_CHECK(hipGetDevice(&current));
        for (int d = 0; d < count; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_had12), kHad12, sizeof(kHad12)));
            HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_had28), kHad28, sizeof(kHad28)));
        }
        HIP_CHECK(hipSetDevice(current));
    });
}
// LDS LUT copies: BEHEMOTH_QTIP_COPIES (GEMV) / BEHEMOTH_QTIP_WMMA_COPIES override.
int env_copies(const char *name, int fallback) {
    const char *v = std::getenv(name);
    const int x = v ? std::atoi(v) : fallback;
    return (x == 1 || x == 4 || x == 8 || x == 16) ? x : fallback;
}
int default_copies(bool wmma) {
    static const int gemv = env_copies("BEHEMOTH_QTIP_COPIES", 16);
    static const int mma = env_copies("BEHEMOTH_QTIP_WMMA_COPIES", 8);
    return wmma ? mma : gemv;
}
int multiprocessors() {
    static int mp = [] {
        int device = 0, value = 0;
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeMultiprocessorCount, device));
        return value;
    }();
    return mp;
}
template <class F> int occupancy(F kernel, int threads, size_t lds) {
    int blocks = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, lds));
    return std::max(blocks, 1);
}
bool use_wmma(int rows, const MatmulOptions &o) {
    if (o.kernel == MatmulKernel::Gemv)
        return false;
    if (o.kernel == MatmulKernel::Wmma)
        return true;
    static const int threshold = [] {
        const char *v = std::getenv("BEHEMOTH_QTIP_GEMV_ROWS");
        return v ? std::atoi(v) : 4;
    }();
    return rows > std::min(threshold, 4);
}
int env_rows(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}
// Defaults measured on 2x R9700 (PREFILL_SCHED_REPORT.md).
std::atomic<int> &dense_rows_setting() {
    static std::atomic<int> v{env_rows("BEHEMOTH_QTIP_DENSE_ROWS", 512)};
    return v;
}
std::atomic<int> &row_transform_setting() {
    static std::atomic<int> v{env_rows("BEHEMOTH_QTIP_ROW_TRANSFORMS", 64)};
    return v;
}
bool row_transforms(int rows) {
    const int t = row_transform_setting().load(std::memory_order_relaxed);
    return t > 0 && rows >= t;
}
constexpr int gemv_waves = 16, wmma_waves = 8;
// 16-row batch tiles per WMMA pass: 1 (rows <= 16), 2 (<= 32), else 4 (64 rows per pass).
int wmma_tiles(int rows) {
    return rows <= 16 ? 1 : rows <= 32 ? 2 : 4;
}
} // namespace

void GpuLinear::upload(const std::vector<std::span<const uint16_t>> &parts, int in_) {
    if (in_ % 32)
        throw std::invalid_argument("QTIP input dimension must be a multiple of 32");
    size_t words = 0;
    for (auto p : parts)
        words += p.size();
    const size_t per = size_t(in_ / 32) * 128;
    if (words % per)
        throw std::invalid_argument("QTIP shard must be whole block-rows");
    std::vector<uint16_t> all;
    all.reserve(words);
    for (auto p : parts)
        all.insert(all.end(), p.begin(), p.end());
    trellis.upload(all);
    in = in_;
    rows = int(words / per) * 32;
}
int matmul_slices(LinearView l, int rows, const MatmulOptions &o) {
    if (o.slices > 0)
        return o.slices;
    const int block_rows = l.rows / 32, n2 = l.in / 32;
    // Target ~2048 waves in flight for the bandwidth-bound decode kernels (64 CUs), at least 16
    // blocks (4 KB of trellis) per wave.
    int items = block_rows;
    if (use_wmma(rows, o))
        items *= (rows + 16 * wmma_tiles(rows) - 1) / (16 * wmma_tiles(rows));
    int slices = 1;
    while (items * slices < 2048 && n2 / (slices * 2) >= 16)
        slices *= 2;
    return slices;
}
int matmul_slices(const GpuLinear &l, int rows, const MatmulOptions &o) {
    return matmul_slices(l.view(), rows, o);
}
void matmul(LinearView l, const uint32_t *lut, const BFloat16 *x, int ldx, int rows,
            float *partial, int slices, hipStream_t stream, const MatmulOptions &o) {
    if (rows < 1 || l.rows % 32 || l.in % 32 || ldx % 8 || slices < 1)
        throw std::invalid_argument("QTIP matmul shape");
    const int block_rows = l.rows / 32, n2 = l.in / 32, per = (n2 + slices - 1) / slices;
    const int copies = o.copies ? o.copies : default_copies(use_wmma(rows, o));
    const uint2 *tr = reinterpret_cast<const uint2 *>(l.trellis);
    auto launch = [&](auto kernel, int waves, int items, auto... args) {
        const int threads = waves * 32;
        const int max_blocks = multiprocessors() * occupancy(kernel, threads, 0);
        const int blocks = std::max(1, std::min(max_blocks, (items + waves - 1) / waves));
        kernel<<<blocks, threads, 0, stream>>>(tr, block_rows, n2, per, slices, lut, args...);
        HIP_CHECK(hipGetLastError());
    };
#define BEHEMOTH_QTIP_COPIES(MACRO)                                                               \
    switch (copies) {                                                                             \
    case 1:                                                                                       \
        MACRO(1);                                                                                 \
        break;                                                                                    \
    case 4:                                                                                       \
        MACRO(4);                                                                                 \
        break;                                                                                    \
    case 8:                                                                                       \
        MACRO(8);                                                                                 \
        break;                                                                                    \
    default:                                                                                      \
        MACRO(16);                                                                                \
    }
    if (!use_wmma(rows, o)) {
        if (rows > 4)
            throw std::invalid_argument("QTIP GEMV supports at most 4 rows");
        const auto *x2 = reinterpret_cast<const uint32_t *>(x);
        const int items = block_rows * slices;
#define GEMV_R(RR, CC)                                                                             \
    launch(gemv_kernel<RR, CC, gemv_waves>, gemv_waves, items, x2, ldx / 2, rows, partial, l.rows)
#define GEMV_C(CC)                                                                                 \
    do {                                                                                          \
        if (rows == 1)                                                                            \
            GEMV_R(1, CC);                                                                        \
        else if (rows == 2)                                                                       \
            GEMV_R(2, CC);                                                                        \
        else if (rows == 3)                                                                       \
            GEMV_R(3, CC);                                                                        \
        else                                                                                      \
            GEMV_R(4, CC);                                                                        \
    } while (0)
        BEHEMOTH_QTIP_COPIES(GEMV_C)
#undef GEMV_C
#undef GEMV_R
        return;
    }
    const auto *xb = reinterpret_cast<const uint16_t *>(x);
    const int mt = wmma_tiles(rows);
    const int passes = (rows + 16 * mt - 1) / (16 * mt);
    const int items = block_rows * slices * passes;
#define WMMA_MT(MT, CC) launch(wmma_kernel<MT, CC, wmma_waves>, wmma_waves, items, xb, ldx, rows, partial, l.rows)
#define WMMA_C(CC)                                                                                 \
    do {                                                                                          \
        if (mt == 1)                                                                              \
            WMMA_MT(1, CC);                                                                       \
        else if (mt == 2)                                                                         \
            WMMA_MT(2, CC);                                                                       \
        else                                                                                      \
            WMMA_MT(4, CC);                                                                       \
    } while (0)
    BEHEMOTH_QTIP_COPIES(WMMA_C)
#undef WMMA_C
#undef WMMA_MT
#undef BEHEMOTH_QTIP_COPIES
}
nlohmann::json kernel_info() {
    nlohmann::json j;
    j["multiprocessors"] = multiprocessors();
    auto add = [&](const char *name, auto kernel, int threads) {
        hipFuncAttributes a{};
        HIP_CHECK(hipFuncGetAttributes(&a, reinterpret_cast<const void *>(kernel)));
        j[name] = {{"blocks_per_mp", occupancy(kernel, threads, 0)},
                   {"threads", threads},
                   {"lds_bytes", a.sharedSizeBytes},
                   {"vgprs", a.numRegs}};
    };
    add("gemv_r1_c16", gemv_kernel<1, 16, gemv_waves>, gemv_waves * 32);
    add("gemv_r4_c16", gemv_kernel<4, 16, gemv_waves>, gemv_waves * 32);
    add("gemv_r1_c8", gemv_kernel<1, 8, gemv_waves>, gemv_waves * 32);
    add("wmma_mt4_c8", wmma_kernel<4, 8, wmma_waves>, wmma_waves * 32);
    add("wmma_mt4_c16", wmma_kernel<4, 16, wmma_waves>, wmma_waves * 32);
    add("wmma_mt1_c8", wmma_kernel<1, 8, wmma_waves>, wmma_waves * 32);
    return j;
}
void reduce_publish(const float *partial, int slices, size_t n, bool bf16, void *local, void *host,
                    unsigned *flag, unsigned *counter, unsigned seq, hipStream_t stream) {
    if (n % 4)
        throw std::invalid_argument("QTIP reduce needs a multiple of four elements");
    const int n4 = int(n / 4), blocks = (n4 + 255) / 256;
    if (bf16)
        reduce_publish_kernel<true><<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const float4 *>(partial), slices, n4, local, host, flag, counter, seq);
    else
        reduce_publish_kernel<false><<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const float4 *>(partial), slices, n4, local, host, flag, counter, seq);
    HIP_CHECK(hipGetLastError());
}
void gather_peer(const void *peer_host, void *all, size_t words, int rank, const unsigned *peer_flag,
                 unsigned seq, hipStream_t stream) {
    if (words % 4)
        throw std::invalid_argument("QTIP gather needs a multiple of four words");
    const int n4 = int(words / 4);
    gather_peer_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint4 *>(peer_host), static_cast<uint4 *>(all), n4, rank, peer_flag, seq);
    HIP_CHECK(hipGetLastError());
}
void exchange(const float *partial, int slices, size_t n, ExchangeMode mode, void *local, void *host,
              unsigned *flag, unsigned *counter, unsigned seq, const void *peer_host, void *peer_dst,
              const unsigned *peer_flag, hipStream_t stream) {
    if (n % 4)
        throw std::invalid_argument("QTIP exchange needs a multiple of four");
    const int n4 = int(n / 4);
    const int blocks = std::max(1, std::min(64, (n4 + 255) / 256));
    auto *pf = reinterpret_cast<const float4 *>(partial);
    switch (mode) {
    case ExchangeMode::Fp32:
        exchange_kernel<0><<<blocks, 256, 0, stream>>>(pf, slices, n4, local, host, flag, counter, seq,
                                                       peer_host, peer_dst, peer_flag);
        break;
    case ExchangeMode::Bf16:
        exchange_kernel<1><<<blocks, 256, 0, stream>>>(pf, slices, n4, local, host, flag, counter, seq,
                                                       peer_host, peer_dst, peer_flag);
        break;
    case ExchangeMode::Words:
        exchange_kernel<2><<<blocks, 256, 0, stream>>>(pf, slices, n4, local, host, flag, counter, seq,
                                                       peer_host, peer_dst, peer_flag);
        break;
    }
    HIP_CHECK(hipGetLastError());
}
void input_transform(const Source &x, const BFloat16 *norm, float epsilon, const float *su, int n,
                     int rows, BFloat16 *u, int ldu, hipStream_t stream) {
    upload_tables();
    auto *out = reinterpret_cast<uint16_t *>(u);
    const auto *w = reinterpret_cast<const uint16_t *>(norm);
    if (row_transforms(rows)) {
        if (n == 12288) {
            if (norm)
                input_transform_rows_kernel<12, true><<<rows, 256, 0, stream>>>(x, w, epsilon, su, out, ldu);
            else
                input_transform_rows_kernel<12, false><<<rows, 256, 0, stream>>>(x, w, epsilon, su, out, ldu);
        } else if (n == 28672 && !norm) {
            input_transform_rows_kernel<28, false><<<rows, 256, 0, stream>>>(x, w, epsilon, su, out, ldu);
        } else
            throw std::invalid_argument("QTIP input transform supports 12288 (+norm) and 28672");
        HIP_CHECK(hipGetLastError());
        return;
    }
    const dim3 grid(unsigned(n / 1024), unsigned(rows));
    if (n == 12288) {
        if (norm)
            input_transform_kernel<12, true><<<grid, 256, 0, stream>>>(x, w, epsilon, su, out, ldu);
        else
            input_transform_kernel<12, false><<<grid, 256, 0, stream>>>(x, w, epsilon, su, out, ldu);
    } else if (n == 28672 && !norm) {
        input_transform_kernel<28, false><<<grid, 256, 0, stream>>>(x, w, epsilon, su, out, ldu);
    } else
        throw std::invalid_argument("QTIP input transform supports 12288 (+norm) and 28672");
    HIP_CHECK(hipGetLastError());
}
template <class T>
void output_qkv_impl(const T *all, int rows, int rank, const float *sv_q, const float *sv_k,
                const float *sv_v, BFloat16 *q, BFloat16 *k, BFloat16 *v, hipStream_t stream) {
    upload_tables();
    if (row_transforms(rows)) {
        output_qkv_rows_kernel<<<rows, 256, 0, stream>>>(
            all, rows, rank, sv_q, sv_k, sv_v, reinterpret_cast<uint16_t *>(q),
            reinterpret_cast<uint16_t *>(k), reinterpret_cast<uint16_t *>(v));
        HIP_CHECK(hipGetLastError());
        return;
    }
    output_qkv_kernel<<<dim3(8, unsigned(rows)), 256, 0, stream>>>(
        all, rows, rank, sv_q, sv_k, sv_v, reinterpret_cast<uint16_t *>(q),
        reinterpret_cast<uint16_t *>(k), reinterpret_cast<uint16_t *>(v));
    HIP_CHECK(hipGetLastError());
}
template <class T>
void output_residual_impl(const T *all, int rows, const float *sv, BFloat16 *state,
                     hipStream_t stream) {
    upload_tables();
    if (row_transforms(rows)) {
        output_residual_rows_kernel<false><<<rows, 256, 0, stream>>>(
            all, rows, sv, reinterpret_cast<uint16_t *>(state), nullptr, 0.f, nullptr, nullptr, 0);
        HIP_CHECK(hipGetLastError());
        return;
    }
    output_residual_kernel<<<dim3(12, unsigned(rows)), 256, 0, stream>>>(
        all, rows, sv, reinterpret_cast<uint16_t *>(state));
    HIP_CHECK(hipGetLastError());
}
template <class T>
void output_residual_input_impl(const T *all, int rows, const float *sv, BFloat16 *state,
                           const BFloat16 *norm, float epsilon, const float *su, BFloat16 *u,
                           int ldu, ResidualSync *sync, hipStream_t stream) {
    upload_tables();
    if (!row_transforms(rows)) {
        static const int sync_rows = env_rows("BEHEMOTH_QTIP_SYNC_ROWS", residual_sync_rows);
        if (sync && sync->counters && rows <= std::min(sync->capacity, sync_rows) && sync_rows > 0) {
            output_residual_input_kernel<<<dim3(12, unsigned(rows)), 256, 0, stream>>>(
                all, rows, sv, reinterpret_cast<uint16_t *>(state),
                reinterpret_cast<const uint16_t *>(norm), epsilon, su, reinterpret_cast<uint16_t *>(u),
                ldu, sync->counters);
            HIP_CHECK(hipGetLastError());
            return;
        }
        output_residual_impl(all, rows, sv, state, stream);
        input_transform(Source{state, 12288, 0, 12288, 0}, norm, epsilon, su, 12288, rows, u, ldu,
                        stream);
        return;
    }
    output_residual_rows_kernel<true><<<rows, 256, 0, stream>>>(
        all, rows, sv, reinterpret_cast<uint16_t *>(state),
        reinterpret_cast<const uint16_t *>(norm), epsilon, su, reinterpret_cast<uint16_t *>(u), ldu);
    HIP_CHECK(hipGetLastError());
}
void output_qkv(const float *all, int rows, int rank, const float *sq, const float *sk,
                const float *sv, BFloat16 *q, BFloat16 *k, BFloat16 *v, hipStream_t s) {
    output_qkv_impl(all, rows, rank, sq, sk, sv, q, k, v, s);
}
void output_qkv(const BFloat16 *all, int rows, int rank, const float *sq, const float *sk,
                const float *sv, BFloat16 *q, BFloat16 *k, BFloat16 *v, hipStream_t s) {
    output_qkv_impl(reinterpret_cast<const uint16_t *>(all), rows, rank, sq, sk, sv, q, k, v, s);
}
void output_residual(const float *all, int rows, const float *sv, BFloat16 *state, hipStream_t s) {
    output_residual_impl(all, rows, sv, state, s);
}
void output_residual(const BFloat16 *all, int rows, const float *sv, BFloat16 *state, hipStream_t s) {
    output_residual_impl(reinterpret_cast<const uint16_t *>(all), rows, sv, state, s);
}
void output_residual_input(const float *all, int rows, const float *sv, BFloat16 *state,
                           const BFloat16 *norm, float eps, const float *su, BFloat16 *u, int ldu,
                           hipStream_t s, ResidualSync *sync) {
    output_residual_input_impl(all, rows, sv, state, norm, eps, su, u, ldu, sync, s);
}
void output_residual_input(const BFloat16 *all, int rows, const float *sv, BFloat16 *state,
                           const BFloat16 *norm, float eps, const float *su, BFloat16 *u, int ldu,
                           hipStream_t s, ResidualSync *sync) {
    output_residual_input_impl(reinterpret_cast<const uint16_t *>(all), rows, sv, state, norm, eps,
                               su, u, ldu, sync, s);
}
void output_gateup(const BFloat16 *all, int rows, const float *sv_gate, const float *sv_up,
                   BFloat16 *h, hipStream_t stream) {
    upload_tables();
    if (row_transforms(rows)) {
        output_gateup_rows_kernel<<<rows, 256, 0, stream>>>(
            reinterpret_cast<const uint16_t *>(all), rows, sv_gate, sv_up,
            reinterpret_cast<uint16_t *>(h));
        HIP_CHECK(hipGetLastError());
        return;
    }
    output_gateup_kernel<<<dim3(28, unsigned(rows)), 256, 0, stream>>>(
        reinterpret_cast<const uint16_t *>(all), rows, sv_gate, sv_up,
        reinterpret_cast<uint16_t *>(h));
    HIP_CHECK(hipGetLastError());
}
void hadamard_rows(const float *x, int n, int rows, bool transpose, float *y, hipStream_t stream) {
    upload_tables();
    const dim3 grid(unsigned(n / 1024 > 0 ? n / 1024 : 1), unsigned(rows));
    if (n == 1024)
        hadamard_kernel<1, false><<<grid, 256, 0, stream>>>(x, rows, y);
    else if (n == 12288)
        transpose ? hadamard_kernel<12, true><<<grid, 256, 0, stream>>>(x, rows, y)
                  : hadamard_kernel<12, false><<<grid, 256, 0, stream>>>(x, rows, y);
    else if (n == 28672)
        transpose ? hadamard_kernel<28, true><<<grid, 256, 0, stream>>>(x, rows, y)
                  : hadamard_kernel<28, false><<<grid, 256, 0, stream>>>(x, rows, y);
    else
        throw std::invalid_argument("QTIP Hadamard size");
    HIP_CHECK(hipGetLastError());
}
void matmul(const GpuLinear &l, const uint32_t *lut, const BFloat16 *x, int ldx, int rows,
            float *partial, int slices, hipStream_t stream, const MatmulOptions &o) {
    matmul(l.view(), lut, x, ldx, rows, partial, slices, stream, o);
}
void decode_dense(const GpuLinear &l, const uint32_t *lut, BFloat16 *w, hipStream_t stream) {
    decode_dense(l.view(), lut, w, stream);
}
void decode_dense(LinearView l, const uint32_t *lut, BFloat16 *w, hipStream_t stream) {
    if (l.rows % 32 || l.in % 32 || l.rows < 32)
        throw std::invalid_argument("QTIP decode shape");
    const int n2 = l.in / 32;
    decode_dense_kernel<<<dim3(unsigned((n2 + 7) / 8), unsigned(l.rows / 32)), 256, 0, stream>>>(
        reinterpret_cast<const uint2 *>(l.trellis), n2, lut, reinterpret_cast<uint16_t *>(w), l.in);
    HIP_CHECK(hipGetLastError());
}
int dense_rows_threshold() {
    return dense_rows_setting().load(std::memory_order_relaxed);
}
void set_dense_rows_threshold(int rows) {
    dense_rows_setting().store(rows, std::memory_order_relaxed);
}
int row_transform_threshold() {
    return row_transform_setting().load(std::memory_order_relaxed);
}
void set_row_transform_threshold(int rows) {
    row_transform_setting().store(rows, std::memory_order_relaxed);
}

void wmma_probe(const BFloat16 *a, const BFloat16 *b, float *d, hipStream_t stream) {
    wmma_probe_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<const uint16_t *>(a),
                                            reinterpret_cast<const uint16_t *>(b), d);
    HIP_CHECK(hipGetLastError());
}
} // namespace behemoth::qtip
