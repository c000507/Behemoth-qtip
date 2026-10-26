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
#include <mutex>

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
    const int lane = int(threadIdx.x) & 31, wave = int(threadIdx.x) >> 5;
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
        for (int j0 = j_begin; j0 < j_end; ++j0) {
            const uint4 *t4 = reinterpret_cast<const uint4 *>(tp + size_t(j0) * 32 + size_t(4 * q));
            const uint4 l0 = t4[0], l1 = t4[1];
            const uint2 l2 = tp[size_t(j0) * 32 + size_t((4 * q + 4) & 31)];
            // d[c][j1]: words c = 4q .. 4q+4 of the j1 tiles (lo half: i1 = 0, hi: i1 = 1)
            const uint32_t d[5][2] = {{l0.x, l0.y}, {l0.z, l0.w}, {l1.x, l1.y}, {l1.z, l1.w}, {l2.x, l2.y}};
            uint4 bx[MT][2];
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int j1 = 0; j1 < 2; ++j1) {
                    const uint4 v = *reinterpret_cast<const uint4 *>(xrow[m] + size_t(32 * j0 + 16 * j1));
                    bx[m][j1] = valid[m] ? v : make_uint4(0, 0, 0, 0);
                }
#pragma unroll
            for (int j1 = 0; j1 < 2; ++j1)
#pragma unroll
                for (int i1 = 0; i1 < 2; ++i1) {
                    uint32_t f[4];
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const uint32_t u = __builtin_amdgcn_perm(d[b][j1], d[b + 1][j1],
                                                                 i1 ? 0x07060302U : 0x05040100U);
                        f[b] = lds_word(table, lut_address<C>(lut_state(u >> shift), offset));
                    }
                    const bf16x8_t af = __builtin_bit_cast(bf16x8_t, make_uint4(f[0], f[1], f[2], f[3]));
#pragma unroll
                    for (int m = 0; m < MT; ++m)
                        acc[m][i1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            af, __builtin_bit_cast(bf16x8_t, bx[m][j1]), acc[m][i1]);
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

// ---------------------------------------------------------------- decode (tests)
__global__ void decode_dense_kernel(const uint2 *__restrict__ trellis, int n2, const uint32_t *lut,
                                    uint16_t *w, int in) {
    // grid (n2, block_rows), 32 threads: lane w decodes word w of the 4 tiles of one block
    const int lane = int(threadIdx.x), j0 = int(blockIdx.x), br = int(blockIdx.y);
    const uint2 *tp = trellis + (size_t(br) * size_t(n2) + size_t(j0)) * 32;
    const uint2 d0 = tp[lane], d1 = tp[(lane + 1) & 31];
    const int A = lane >> 2, B = lane & 3;
    for (int j1 = 0; j1 < 2; ++j1) {
        const uint32_t a = j1 ? d0.y : d0.x, b = j1 ? d1.y : d1.x;
        for (int i1 = 0; i1 < 2; ++i1) {
            const uint32_t u = __builtin_amdgcn_perm(a, b, i1 ? 0x07060302U : 0x05040100U);
            for (int k = 0; k < 4; ++k) {
                const uint32_t p = lut_state(u >> (16 - 4 * k));
                const uint32_t e = (p >> 6) & 1023U, v = lut[e];
                const int cc = k >> 1, D = k & 1;
                const size_t row = size_t(br * 32 + i1 * 16 + D * 8 + A);
                const size_t col = size_t(32 * j0 + 16 * j1 + 8 * cc + 2 * B);
                w[row * size_t(in) + col] = uint16_t(v & 0xffffU);
                w[row * size_t(in) + col + 1] = uint16_t(v >> 16);
            }
        }
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
#pragma unroll 4
        for (int k = 0; k < K; ++k) {
            float t[4];
            load4<T>(s, row, k * 1024 + j, t);
            const float sign = float(TRANSPOSE ? had_sign<K>(k, k2) : had_sign<K>(k2, k));
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] += sign * t[i];
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
// grid (K, rows), 256 threads.
template <int K, bool NORM>
__global__ __launch_bounds__(256) void input_transform_kernel(Source src, const uint16_t *norm,
                                                              float epsilon, const float *su,
                                                              uint16_t *u, int ldu) {
    __shared__ float scratch[1024];
    __shared__ float red[8];
    const int k2 = int(blockIdx.x), row = int(blockIdx.y), j = 4 * int(threadIdx.x);
    constexpr int n = K * 1024;
    float factor = 1.f;
    if constexpr (NORM) {
        float ss = 0;
        for (int e = j; e < n; e += 1024) {
            float t[4];
            load4<uint16_t>(src, row, e, t);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ss += t[i] * t[i];
        }
        factor = rsqrtf(block_sum(ss, red) / float(n) + epsilon);
    }
    float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
    for (int k = 0; k < K; ++k) {
        const int e = k * 1024 + j;
        float t[4];
        load4<uint16_t>(src, row, e, t);
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
// grid (8, rows): blocks 0..5 = this rank's q chunks, 6 = k, 7 = v.
__global__ __launch_bounds__(256) void output_qkv_kernel(const float *all, int rows, int rank,
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
        mix<12, false, float>(s, row, k2, x);
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
    mix<1, false, float>(s, row, 0, x);
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
// grid (12, rows)
__global__ __launch_bounds__(256) void output_residual_kernel(const float *all, int rows,
                                                              const float *sv, uint16_t *state) {
    __shared__ float scratch[1024];
    const int k2 = int(blockIdx.x), row = int(blockIdx.y), j = 4 * int(threadIdx.x);
    Source s{all, 6144, size_t(rows) * 6144, 6144, 0};
    float x[4];
    mix<12, false, float>(s, row, k2, x);
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
constexpr int gemv_waves = 16, wmma_waves = 8;
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
int matmul_slices(const GpuLinear &l, int rows, const MatmulOptions &o) {
    if (o.slices > 0)
        return o.slices;
    const int block_rows = l.rows / 32, n2 = l.in / 32;
    // Target ~2048 waves in flight for the bandwidth-bound decode kernels (64 CUs), at least 16
    // blocks (4 KB of trellis) per wave.
    int items = block_rows;
    if (use_wmma(rows, o))
        items *= (rows + 63) / 64;
    int slices = 1;
    while (items * slices < 2048 && n2 / (slices * 2) >= 16)
        slices *= 2;
    return slices;
}
void matmul(const GpuLinear &l, const uint32_t *lut, const BFloat16 *x, int ldx, int rows,
            float *partial, int slices, hipStream_t stream, const MatmulOptions &o) {
    if (rows < 1 || l.rows % 32 || l.in % 32 || ldx % 8 || slices < 1)
        throw std::invalid_argument("QTIP matmul shape");
    const int block_rows = l.rows / 32, n2 = l.in / 32, per = (n2 + slices - 1) / slices;
    const int copies = o.copies ? o.copies : default_copies(use_wmma(rows, o));
    const uint2 *tr = reinterpret_cast<const uint2 *>(l.trellis.data());
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
    const int passes = (rows + 63) / 64;
    const int items = block_rows * slices * passes;
#define WMMA_C(CC) launch(wmma_kernel<4, CC, wmma_waves>, wmma_waves, items, xb, ldx, rows, partial, l.rows)
    BEHEMOTH_QTIP_COPIES(WMMA_C)
#undef WMMA_C
#undef BEHEMOTH_QTIP_COPIES
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
void input_transform(const Source &x, const BFloat16 *norm, float epsilon, const float *su, int n,
                     int rows, BFloat16 *u, int ldu, hipStream_t stream) {
    upload_tables();
    auto *out = reinterpret_cast<uint16_t *>(u);
    const auto *w = reinterpret_cast<const uint16_t *>(norm);
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
void output_qkv(const float *all, int rows, int rank, const float *sv_q, const float *sv_k,
                const float *sv_v, BFloat16 *q, BFloat16 *k, BFloat16 *v, hipStream_t stream) {
    upload_tables();
    output_qkv_kernel<<<dim3(8, unsigned(rows)), 256, 0, stream>>>(
        all, rows, rank, sv_q, sv_k, sv_v, reinterpret_cast<uint16_t *>(q),
        reinterpret_cast<uint16_t *>(k), reinterpret_cast<uint16_t *>(v));
    HIP_CHECK(hipGetLastError());
}
void output_residual(const float *all, int rows, const float *sv, BFloat16 *state,
                     hipStream_t stream) {
    upload_tables();
    output_residual_kernel<<<dim3(12, unsigned(rows)), 256, 0, stream>>>(
        all, rows, sv, reinterpret_cast<uint16_t *>(state));
    HIP_CHECK(hipGetLastError());
}
void output_gateup(const BFloat16 *all, int rows, const float *sv_gate, const float *sv_up,
                   BFloat16 *h, hipStream_t stream) {
    upload_tables();
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
void decode_dense(const GpuLinear &l, const uint32_t *lut, BFloat16 *w, hipStream_t stream) {
    decode_dense_kernel<<<dim3(unsigned(l.in / 32), unsigned(l.rows / 32)), 32, 0, stream>>>(
        reinterpret_cast<const uint2 *>(l.trellis.data()), l.in / 32, lut,
        reinterpret_cast<uint16_t *>(w), l.in);
    HIP_CHECK(hipGetLastError());
}
void wmma_probe(const BFloat16 *a, const BFloat16 *b, float *d, hipStream_t stream) {
    wmma_probe_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<const uint16_t *>(a),
                                            reinterpret_cast<const uint16_t *>(b), d);
    HIP_CHECK(hipGetLastError());
}
} // namespace behemoth::qtip
