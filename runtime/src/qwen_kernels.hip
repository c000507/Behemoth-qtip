// Qwen3.5 runtime kernels: generic Hadamard (K x P, K in {1,12,28}), fused input transform (zc-norm +
// SU + hadUt), epilogues (SV cast / residual add / gate-up), gated full attention (norm + partial rope +
// output gate, bf16 KV cache) and the Gated DeltaNet recurrent step.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <stdexcept>
#include <algorithm>

#include <map>
#include <mutex>
#include <shared_mutex>

#include "behemoth/gpu.hpp"
#include "behemoth/qwen_kernels.hpp"

namespace behemoth {

namespace {
inline unsigned nblocks(long n, int bs) { return unsigned((n + bs - 1) / bs); }
__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
// paged KV: position t of a sequence lives at row pt[t / 64] * 64 + t % 64 of the pool (pt null: row t)
__device__ __forceinline__ long kvrow(const int *pt, int t) { return pt ? long(pt[t >> 6]) * 64 + (t & 63) : long(t); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : __logf(1.0f + __expf(x)); }

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float red[32];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
    if (w == 0)
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    return v;
}

// ------------------------------------------------------------------ hadamard K x P
// stage 1: natural-order FWHT on each contiguous chunk of P entries (P power of two, <= 1024)
__global__ __launch_bounds__(256) void q_fwht_kernel(float *X, long rows, int n, long ld, int P, int K,
                                                     float scale) {
    extern __shared__ float v[];  // P floats (P <= 4096)
    long row = blockIdx.x / K;
    int chunk = blockIdx.x % K;
    float *x = X + row * ld + long(chunk) * P;
    for (int i = threadIdx.x; i < P; i += blockDim.x) v[i] = x[i];
    __syncthreads();
    for (int h = 1; h < P; h <<= 1) {
        for (int k = threadIdx.x; k < P / 2; k += blockDim.x) {
            int i = (k / h) * 2 * h + (k % h);
            float a = v[i], b = v[i + h];
            v[i] = a + b;
            v[i + h] = a - b;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < P; i += blockDim.x) x[i] = v[i] * scale;
}
// stage 2: y[k2*P + i] = sum_k had[k2][k] y[k*P + i]
__global__ void q_had_mix_kernel(float *X, long rows, int P, int K, long ld, const int8_t *had,
                                 float scale) {
    __shared__ int8_t hs[28 * 28];
    for (int i = threadIdx.x; i < K * K; i += blockDim.x) hs[i] = had[i];
    __syncthreads();
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * P) return;
    long row = idx / P;
    int i = int(idx % P);
    float *x = X + row * ld;
    float y[28];
    for (int k = 0; k < K; ++k) y[k] = x[long(k) * P + i];
    for (int k2 = 0; k2 < K; ++k2) {
        float s = 0;
        for (int k = 0; k < K; ++k) s += hs[k2 * K + k] > 0 ? y[k] : -y[k];
        x[long(k2) * P + i] = s * scale;
    }
}
const int8_t *q_device_had(int K, bool transpose) {
    static std::mutex mu;
    static std::map<std::pair<int, bool>, int8_t *> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto key = std::make_pair(K, transpose);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const int8_t *h = qtip::had_matrix(K);
    if (!h) throw std::runtime_error("no hadamard matrix for K=" + std::to_string(K));
    std::vector<int8_t> m(size_t(K) * K);
    for (int k2 = 0; k2 < K; ++k2)
        for (int k = 0; k < K; ++k) m[size_t(k2) * K + k] = transpose ? h[k * K + k2] : h[k2 * K + k];
    int8_t *d = nullptr;
    HIP_CHECK(hipMalloc(&d, m.size()));
    HIP_CHECK(hipMemcpy(d, m.data(), m.size(), hipMemcpyHostToDevice));
    cache[key] = d;
    return d;
}
void q_had_factor(int n, int &K, int &P) {
    // P = power of two <= 4096 with K = n/P in {1, 12, 28} (covers 32/512/1024/2048/4096 pow2,
    // 3584 = 28*128, 6144 = 12*512)
    for (P = 4096; P > 1; P >>= 1) {
        if (n % P) continue;
        K = n / P;
        if (K == 1 || K == 12 || K == 28) return;
    }
    P = 1;
    K = n;
    throw std::runtime_error("unsupported hadamard dim " + std::to_string(n));
}
}  // namespace

void q_hadamard_rows(float *X, long rows, int n, long ld, bool transpose, hipStream_t s) {
    int K, P;
    q_had_factor(n, K, P);
    float sc = 1.0f / std::sqrt(float(n));
    q_fwht_kernel<<<unsigned(rows * K), 256, size_t(P) * 4, s>>>(X, rows, n, ld, P, K, K == 1 ? sc : 1.0f);
    if (K != 1)
        q_had_mix_kernel<<<nblocks(rows * P, 256), 256, 0, s>>>(X, rows, P, K, ld, q_device_had(K, transpose), sc);
    HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ input transform
// u = bf16( hadUt_n( SU (.) x' ) ), x' = bf16(x * rsqrt(mean(x²)+eps) * (1+w)) when norm non-null.
// One block per row; n floats of shared memory (n <= 6144).
__global__ __launch_bounds__(256) void q_input_transform_kernel(const uint16_t *x, int ldx,
                                                                const uint16_t *norm, float eps,
                                                                const float *su, int n, int K, int P,
                                                                const int8_t *had, uint16_t *u,
                                                                int ldu) {
    const uint16_t *xr = x + long(blockIdx.x) * ldx;
    extern __shared__ float t[];
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = bf2f(xr[i]);
        ss += v * v;
    }
    ss = block_sum(ss);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = norm ? rsqrtf(ss / n + eps) : 1.0f;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = bf2f(xr[i]) * inv;
        if (norm) v = bf2f(f2bf(v)) * (1.0f + bf2f(norm[i]));
        t[i] = v * su[i];
    }
    __syncthreads();
    const float sc = 1.0f / sqrtf(float(n));
    for (int c = 0; c < K; ++c) {
        float *v = t + c * P;
        for (int h = 1; h < P; h <<= 1) {
            for (int k = threadIdx.x; k < P / 2; k += blockDim.x) {
                int i = (k / h) * 2 * h + (k % h);
                float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
            __syncthreads();
        }
        if (K == 1)
            for (int i = threadIdx.x; i < P; i += blockDim.x)
                u[long(blockIdx.x) * ldu + long(c) * P + i] = f2bf(v[i] * sc);
    }
    if (K != 1) {
        __shared__ int8_t hs[28 * 28];
        for (int i = threadIdx.x; i < K * K; i += blockDim.x) hs[i] = had[i];
        __syncthreads();
        for (int idx = threadIdx.x; idx < K * P; idx += blockDim.x) {
            int i = idx % P, k2 = idx / P;
            float sum = 0;
            for (int k = 0; k < K; ++k) sum += hs[k2 * K + k] > 0 ? t[k * P + i] : -t[k * P + i];
            u[long(blockIdx.x) * ldu + long(k2) * P + i] = f2bf(sum * sc);
        }
    }
}

void q_input_transform(const uint16_t *x, long rows, int ldx, const uint16_t *norm, float eps,
                       const float *su, int n, uint16_t *u, int ldu, hipStream_t s) {
    int K, P;
    q_had_factor(n, K, P);
    const int8_t *had = (K != 1) ? q_device_had(K, true) : nullptr;
    q_input_transform_kernel<<<unsigned(rows), 256, size_t(n) * 4, s>>>(x, ldx, norm, eps, su, n, K, P,
                                                                        had, u, ldu);
    HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ epilogues
__global__ void q_sv_cast_kernel(const float *z, const float *sv, uint16_t *out, long rows, int n,
                                 long ldz) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * n) return;
    long r = idx / n;
    int c = int(idx % n);
    out[idx] = f2bf(bf2f(f2bf(z[r * ldz + c] * sv[c])));
}

// f32 variant with a wider output row stride (scatter into the delta-layer z buffer)
__global__ void q_sv_cast_f32_kernel(const float *z, const float *sv, float *out, long rows, int n,
                                     long ldz, int ldo) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * n) return;
    long r = idx / n;
    int c = int(idx % n);
    out[long(r) * ldo + c] = z[r * ldz + c] * sv[c];
}

// zero-centred rmsnorm in place on bf16 rows: x = bf16(x*rsqrt(mean+eps)*(1+w))
__global__ __launch_bounds__(256) void q_zc_norm_kernel(uint16_t *x, int n, const uint16_t *w,
                                                        float eps) {
    uint16_t *xr = x + long(blockIdx.x) * n;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = bf2f(xr[i]);
        ss += v * v;
    }
    ss = block_sum(ss);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        xr[i] = f2bf(bf2f(xr[i]) * inv * (1.0f + bf2f(w[i])));
}

__global__ void q_sv_add_kernel(const float *z, const float *sv, uint16_t *state, long rows, int n,
                                long ldz, int lds) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * n) return;
    long r = idx / n;
    int c = int(idx % n);
    float y = bf2f(f2bf(z[r * ldz + c] * sv[c]));
    state[r * lds + c] = f2bf(bf2f(state[r * lds + c]) + y);
}

__global__ void q_gateup_kernel(const float *zg, const float *zu, const float *svg, const float *svu,
                                uint16_t *h, long rows, int F, long ld) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * F) return;
    long r = idx / F;
    int c = int(idx % F);
    float g = bf2f(f2bf(zg[r * ld + c] * svg[c]));
    float u = bf2f(f2bf(zu[r * ld + c] * svu[c]));
    h[idx] = f2bf(silu_f(g) * u);  // was bf2f(silu_f(g)): a float silently truncated to uint16_t
}

__global__ void q_embed_gather_kernel(const uint16_t *embed, const int *tokens, uint16_t *state,
                                      long rows, int H) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * H) return;
    long r = idx / H;
    state[idx] = embed[long(tokens[r]) * H + idx % H];
}

void q_sv_cast(const float *z, const float *sv, uint16_t *out, long rows, int n, long ldz,
               hipStream_t s) {
    q_sv_cast_kernel<<<nblocks(rows * n, 256), 256, 0, s>>>(z, sv, out, rows, n, ldz);
    HIP_CHECK(hipGetLastError());
}
void q_sv_cast_f32(const float *z, const float *sv, float *out, long rows, int n, long ldz, int ldo,
                   hipStream_t s) {
    q_sv_cast_f32_kernel<<<nblocks(rows * n, 256), 256, 0, s>>>(z, sv, out, rows, n, ldz, ldo);
    HIP_CHECK(hipGetLastError());
}
void q_zc_norm(uint16_t *x, long rows, int n, const uint16_t *w, float eps, hipStream_t s) {
    q_zc_norm_kernel<<<unsigned(rows), 256, 0, s>>>(x, n, w, eps);
    HIP_CHECK(hipGetLastError());
}
void q_sv_add(const float *z, const float *sv, uint16_t *state, long rows, int n, long ldz, int lds,
              hipStream_t s) {
    q_sv_add_kernel<<<nblocks(rows * n, 256), 256, 0, s>>>(z, sv, state, rows, n, ldz, lds);
    HIP_CHECK(hipGetLastError());
}
void q_gateup(const float *zg, const float *zu, const float *svg, const float *svu, uint16_t *h,
              long rows, int F, long ld, hipStream_t s) {
    q_gateup_kernel<<<nblocks(rows * F, 256), 256, 0, s>>>(zg, zu, svg, svu, h, rows, F, ld);
    HIP_CHECK(hipGetLastError());
}
void q_embed_gather(const uint16_t *embed, const int *tokens, uint16_t *state, long rows, int H,
                    hipStream_t s) {
    q_embed_gather_kernel<<<nblocks(rows * H, 256), 256, 0, s>>>(embed, tokens, state, rows, H);
    HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ sum of GEMV slices
__global__ void q_sum_slices_kernel(const float *partial, int slices, size_t n, float *out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0;
    for (int s = 0; s < slices; ++s) acc += partial[size_t(s) * n + i];
    out[i] = acc;
}
void q_sum_slices(const float *partial, int slices, size_t n, float *out, hipStream_t s) {
    q_sum_slices_kernel<<<nblocks(long(n), 256), 256, 0, s>>>(partial, slices, n, out);
    HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ gated full attention
// q: bf16 [rows][nq*2*hd] (dequantised; per head [q hd | gate hd]); k: bf16 [rows][nkv*hd]; v same.
// In place on q/k: zc-norm + partial rope (positions pos0+r). Appends k/v to the caches
// [maxctx][nkv][hd]; G: bf16 [rows][nq*hd] raw gate. One block per (r, head), blockDim = hd.
__global__ void q_qk_norm_rope_kernel(uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *G,
                                      uint16_t *kc, uint16_t *vc, int rows, int pos0, int nq, int nkv,
                                      int hd, int rot, float theta, const uint16_t *q_norm,
                                      const uint16_t *k_norm, float eps, const int *pt) {
    int r = blockIdx.x % rows;
    int h = blockIdx.x / rows;
    int d = threadIdx.x;
    if (h >= nq + nkv) {
        int hv = h - nq - nkv;
        uint16_t *vv = v + long(r) * nkv * hd + long(hv) * hd;
        uint16_t *vdst = vc + (kvrow(pt, pos0 + r) * nkv + hv) * hd;
        vdst[d] = vv[d];
        return;
    }
    bool is_q = h < nq;
    uint16_t *vec = is_q ? q + long(r) * nq * 2 * hd + long(h) * 2 * hd
                         : k + long(r) * nkv * hd + long(h - nq) * hd;
    if (is_q)  // raw gate copy (second half of the q head block; never touched by rope)
        G[long(r) * nq * hd + long(h) * hd + d] = vec[hd + d];
    float raw = bf2f(vec[d]);
    float ss = block_sum(raw * raw);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / hd + eps);
    __syncthreads();
    const uint16_t *w = is_q ? q_norm : k_norm;
    float sv = raw * inv * (1.0f + bf2f(w[d]));
    if (d < rot / 2) {
        float praw = bf2f(vec[d + rot / 2]);
        float psv = praw * inv * (1.0f + bf2f(w[d + rot / 2]));
        float inv_freq = __powf(theta, -2.0f * float(d) / float(rot));
        float ang = float(pos0 + r) * inv_freq;
        float c = cosf(ang), sn = sinf(ang);
        vec[d] = f2bf(sv * c - psv * sn);
        vec[d + rot / 2] = f2bf(psv * c + sv * sn);
    } else if (d >= rot) {
        vec[d] = f2bf(sv);
    }
    if (!is_q) {  // append k to cache (after ALL rope writes of this head: sync)
        __syncthreads();
        uint16_t *kdst = kc + (kvrow(pt, pos0 + r) * nkv + long(h - nq)) * hd;
        kdst[d] = vec[d];
    }
}
void q_qk_norm_rope(uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *G, uint16_t *kc, uint16_t *vc,
                    int rows, int pos0, int nq, int nkv, int hd, int rot, float theta,
                    const uint16_t *q_norm, const uint16_t *k_norm, float eps, hipStream_t s, const int *pt) {
    q_qk_norm_rope_kernel<<<unsigned((nq + 2 * nkv) * rows), hd, 0, s>>>(q, k, v, G, kc, vc, rows,
                                                                         pos0, nq, nkv, hd, rot,
                                                                         theta, q_norm, k_norm, eps, pt);
    HIP_CHECK(hipGetLastError());
}

// One block per (row, q head): online softmax over cache positions [0, pos0+r], out *= sigmoid(gate).
__global__ void q_attn_kernel(const uint16_t *q, const uint16_t *kc, const uint16_t *vc,
                              const uint16_t *G, uint16_t *out, int rows, int pos0, int nq, int nkv,
                              int hd, float scale) {
    int r = blockIdx.x % rows;
    int h = blockIdx.x / rows;
    int kh = h / (nq / nkv);
    int t_max = pos0 + r;  // inclusive
    const uint16_t *qh = q + long(r) * nq * 2 * hd + long(h) * 2 * hd;
    const uint16_t *gh = G + long(r) * nq * hd + long(h) * hd;
    float m = -INFINITY, l = 0;
    float acc[8];  // hd/32 per thread (hd 256, blockDim 32x8? use blockDim == hd and strided acc)
    const int nacc = hd / int(blockDim.x);
    for (int a = 0; a < nacc; ++a) acc[a] = 0;
    // pass 1: max
    for (int t = 0; t <= t_max; ++t) {
        const uint16_t *kh_vec = kc + (long(t) * nkv + kh) * hd;
        float dot = 0;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) dot += bf2f(qh[d]) * bf2f(kh_vec[d]);
        dot = block_sum(dot) * scale;
        m = fmaxf(m, dot);
    }
    // pass 2: softmax denominator + weighted v (online over t)
    for (int t = 0; t <= t_max; ++t) {
        const uint16_t *kh_vec = kc + (long(t) * nkv + kh) * hd;
        const uint16_t *vh_vec = vc + (long(t) * nkv + kh) * hd;
        float dot = 0;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) dot += bf2f(qh[d]) * bf2f(kh_vec[d]);
        dot = block_sum(dot) * scale;
        float p = __expf(dot - m);
        l += p;
        for (int a = 0; a < nacc; ++a) {
            int d = a * blockDim.x + threadIdx.x;
            acc[a] += p * bf2f(vh_vec[d]);
        }
    }
    float invl = 1.0f / l;
    uint16_t *o = out + long(r) * nq * hd + long(h) * hd;
    for (int a = 0; a < nacc; ++a) {
        int d = a * blockDim.x + threadIdx.x;
        o[d] = f2bf(acc[a] * invl / (1.0f + __expf(-bf2f(gh[d]))));
    }
}
void q_attn(const uint16_t *q, const uint16_t *kc, const uint16_t *vc, const uint16_t *G,
            uint16_t *out, int rows, int pos0, int nq, int nkv, int hd, float scale, hipStream_t s) {
    if (hd % 32) throw std::runtime_error("head dim must be a multiple of 32");
    q_attn_kernel<<<unsigned(rows * nq), 32, 0, s>>>(q, kc, vc, G, out, rows, pos0, nq, nkv, hd, scale);
    HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ Gated DeltaNet step
// Same math as the quantizer's k_q_delta_step (fp32 state), one block per v-head, rows = T tokens of
// one sequence processed sequentially by the host loop; qkvz rows [t][conv_dim + value_dim + 2nv].
__global__ void q_delta_step_kernel(const float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm,
                                    const float *a_dt, float *ring, float *S, uint16_t *out, int T,
                                    int t, int pos0, int conv_dim, int value_dim, int kd, int vd, int nk, int nv,
                                    int K, float eps) {
    int hv = blockIdx.x;
    const int group = nv / nk;
    const int kh = hv / group;
    const int key_dim = nk * kd;
    const int d = threadIdx.x;
    const long W = long(conv_dim) + value_dim + 2 * nv;
    const float *row = qkvz + long(t) * W;
    const int cq = kh * kd + d, ck = key_dim + kh * kd + d, cv = 2 * key_dim + hv * vd + d;
    float *rg_q = ring + long(cq) * K, *rg_k = ring + long(ck) * K, *rg_v = ring + long(cv) * K;
    const int tp = pos0 + t;  // absolute position: the ring persists across chunks / decode steps
    rg_q[tp % K] = row[cq];
    rg_k[tp % K] = row[ck];
    rg_v[tp % K] = row[cv];
    float qv = 0, kvv = 0, vv = 0;
    {
        const uint16_t *cw = conv_w + long(cq) * K;
        float a = 0;
        for (int i = 0; i < K; ++i) a += rg_q[(tp + 1 + i) % K] * bf2f(cw[i]);
        qv = silu_f(a);
        cw = conv_w + long(ck) * K;
        a = 0;
        for (int i = 0; i < K; ++i) a += rg_k[(tp + 1 + i) % K] * bf2f(cw[i]);
        kvv = silu_f(a);
        cw = conv_w + long(cv) * K;
        a = 0;
        for (int i = 0; i < K; ++i) a += rg_v[(tp + 1 + i) % K] * bf2f(cw[i]);
        vv = silu_f(a);
    }
    float qs = block_sum(qv * qv);
    float ks = block_sum(kvv * kvv);
    __shared__ float inv_q, inv_k;
    __shared__ float sh_q[256], sh_k[256];
    if (threadIdx.x == 0) {
        inv_q = rsqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(kd)));
        inv_k = rsqrtf(ks + 1e-6f);
    }
    __syncthreads();
    sh_q[d] = qv * inv_q;
    sh_k[d] = kvv * inv_k;
    __syncthreads();
    const float beta = 1.0f / (1.0f + __expf(-row[conv_dim + value_dim + hv]));
    const float g = -__expf(a_dt[hv]) * softplus_f(row[conv_dim + value_dim + nv + hv] + a_dt[nv + hv]);
    const float decay = __expf(g);
    float *Sh = S + long(hv) * kd * vd;
    float kv_mem = 0;
    for (int dk = 0; dk < kd; ++dk) {
        float sv = Sh[long(dk) * vd + d] * decay;
        Sh[long(dk) * vd + d] = sv;
        kv_mem += sv * sh_k[dk];
    }
    const float delta = (vv - kv_mem) * beta;
    float o = 0;
    for (int dk = 0; dk < kd; ++dk) {
        float sv = Sh[long(dk) * vd + d] + sh_k[dk] * delta;
        Sh[long(dk) * vd + d] = sv;
        o += sv * sh_q[dk];
    }
    float os = block_sum(o * o);
    __shared__ float oinv;
    if (threadIdx.x == 0) oinv = rsqrtf(os / vd + eps);
    __syncthreads();
    const float z = row[conv_dim + long(hv) * vd + d];
    out[long(t) * value_dim + long(hv) * vd + d] = f2bf(o * oinv * bf2f(dn_norm[d]) * silu_f(z));
}
void q_delta_step(const float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm, const float *a_dt,
                  float *ring, float *S, uint16_t *out, int T, int t, int pos0, int conv_dim, int value_dim,
                  int kd, int vd, int nk, int nv, int K, float eps, hipStream_t s) {
    if (kd != vd || vd > 256) throw std::runtime_error("delta step requires kd == vd <= 256");
    q_delta_step_kernel<<<unsigned(nv), vd, 0, s>>>(qkvz, conv_w, dn_norm, a_dt, ring, S, out, T, t,
                                                    pos0, conv_dim, value_dim, kd, vd, nk, nv, K, eps);
    HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ MoE routing (softmax top-k renorm)
// Deterministic selection matching the CPU reference: softmax -> top-k (ties to the lowest index)
// -> renormalise. One block per row; thread 0 selects.
__global__ void q_moe_route_kernel(const float *rlog, int rows, int E, int k, float *wts, int *ids,
                                   int *cnt) {
    extern __shared__ float p[];
    const float *lg = rlog + long(blockIdx.x) * E;
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int e = 0; e < E; ++e) mx = fmaxf(mx, lg[e]);
        float sum = 0;
        for (int e = 0; e < E; ++e) sum += p[e] = __expf(lg[e] - mx);
        float inv = 1.0f / sum;
        for (int j = 0; j < k; ++j) {
            float best = -1.0f;
            int bi = 0;
            for (int e = 0; e < E; ++e)
                if (p[e] > best) best = p[e], bi = e;
            wts[long(blockIdx.x) * k + j] = best * inv;
            ids[long(blockIdx.x) * k + j] = bi;
            p[bi] = -1.0f;
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < k; j += blockDim.x) atomicAdd(&cnt[ids[long(blockIdx.x) * k + j]], 1);
}
void q_moe_route(const float *rlog, int rows, int E, int k, float *wts, int *ids, int *cnt,
                 hipStream_t s) {
    if (E > 4096 || k > 64) throw std::runtime_error("moe route supports E <= 4096, k <= 64");
    q_moe_route_kernel<<<unsigned(rows), 256, size_t(E) * 4, s>>>(rlog, rows, E, k, wts, ids, cnt);
    HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_scan_kernel(const int *cnt, int *base, int E) {
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < E; ++e) {
            base[e] = acc;
            acc += cnt[e];
        }
    }
}
void q_moe_scan(const int *cnt, int *base, int E, hipStream_t s) {
    q_moe_scan_kernel<<<1, 32, 0, s>>>(cnt, base, E);
    HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_route_fill_kernel(const float *rlog, int rows, int E, int k, const int *base,
                                        int *run, int *slot_tok, float *slot_w) {
    extern __shared__ float p[];
    const float *lg = rlog + long(blockIdx.x) * E;
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int e = 0; e < E; ++e) mx = fmaxf(mx, lg[e]);
        float sum = 0;
        for (int e = 0; e < E; ++e) sum += p[e] = __expf(lg[e] - mx);
        float inv = 1.0f / sum;
        for (int j = 0; j < k; ++j) {
            float best = -1.0f;
            int bi = 0;
            for (int e = 0; e < E; ++e)
                if (p[e] > best) best = p[e], bi = e;
            int slot = base[bi] + atomicAdd(&run[bi], 1);
            slot_tok[slot] = int(blockIdx.x);
            slot_w[slot] = best * inv;
            p[bi] = -1.0f;
        }
    }
}
void q_moe_route_fill(const float *rlog, int rows, int E, int k, const int *base, int *run, int *slot_tok,
                      float *slot_w, hipStream_t s) {
    q_moe_route_fill_kernel<<<unsigned(rows), 256, size_t(E) * 4, s>>>(rlog, rows, E, k, base, run,
                                                                       slot_tok, slot_w);
    HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_gather_kernel(const uint16_t *src, int n, const int *slot_tok, uint16_t *dst,
                                    long total) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = int(idx % n);
    int i = int(idx / n);
    dst[idx] = src[long(slot_tok[i]) * n + c];
}
void q_moe_gather(const uint16_t *src, int n, const int *slot_tok, uint16_t *dst, int nslots,
                  hipStream_t s) {
    q_moe_gather_kernel<<<nblocks(long(nslots) * n, 256), 256, 0, s>>>(src, n, slot_tok, dst,
                                                                        long(nslots) * n);
    HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_scatter_sv_add_kernel(float *y, const float *z, const float *sv,
                                            const int *slot_tok, const float *slot_w, int nslots,
                                            int n) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= long(nslots) * n) return;
    int c = int(idx % n);
    int i = int(idx / n);
    atomicAdd(&y[long(slot_tok[i]) * n + c], slot_w[i] * z[idx] * sv[c]);
}
void q_moe_scatter_sv_add(float *y, const float *z, const float *sv, const int *slot_tok,
                          const float *slot_w, int nslots, int n, hipStream_t s) {
    q_moe_scatter_sv_add_kernel<<<nblocks(long(nslots) * n, 256), 256, 0, s>>>(
        y, z, sv, slot_tok, slot_w, nslots, n);
    HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_residual_add_kernel(uint16_t *state, const float *y, long total, int n) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    state[idx] = f2bf(bf2f(state[idx]) + y[idx]);
}
void q_moe_residual_add(uint16_t *state, const float *y, long rows, int n, hipStream_t s) {
    q_moe_residual_add_kernel<<<nblocks(rows * n, 256), 256, 0, s>>>(state, y, rows * n, n);
    HIP_CHECK(hipGetLastError());
}

__global__ void q_scatter_f32_kernel(float *dst, int ldd, int col0, const float *src, int lds, long rows,
                                     int cols) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int c = int(idx % cols);
    int r = int(idx / cols);
    dst[long(r) * ldd + col0 + c] = src[long(r) * lds + c];
}
void q_scatter_f32(float *dst, int ldd, int col0, const float *src, int lds, long rows, int cols,
                   hipStream_t s) {
    q_scatter_f32_kernel<<<nblocks(rows * cols, 256), 256, 0, s>>>(dst, ldd, col0, src, lds, rows, cols);
    HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_sv_add_gated_kernel(const float *z, const float *sv, const float *sg,
                                          uint16_t *state, long rows, int n) {
    long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * n) return;
    int r = int(idx / n);
    int c = int(idx % n);
    float g = sg ? (1.0f / (1.0f + __expf(-sg[r]))) : 1.0f;
    state[idx] = f2bf(bf2f(state[idx]) + g * z[idx] * sv[c]);
}
void q_moe_sv_add_gated(const float *z, const float *sv, const float *sg, uint16_t *state, long rows,
                        int n, hipStream_t s) {
    q_moe_sv_add_gated_kernel<<<nblocks(rows * n, 256), 256, 0, s>>>(z, sv, sg, state, rows, n);
    HIP_CHECK(hipGetLastError());
}

}  // namespace behemoth

// ------------------------------------------------------------------ FP8 (E4M3, 128x128 block scales)
namespace behemoth {
namespace {
__device__ __forceinline__ float e4m3(uint32_t b) {
    __hip_fp8_e4m3 f;
    f.__x = uint8_t(b);
    return float(f);
}
// One wave per output feature, 8 waves per block. Each lane streams 16 consecutive weight bytes per
// 512-wide step (one 128-bit load); the 16 bytes lie inside one 128-column scale block.
template <int R>
__global__ __launch_bounds__(256) void q_fp8_gemv_kernel(const uint8_t *__restrict__ w,
                                                         const float *__restrict__ ws, int m, int n,
                                                         const uint16_t *__restrict__ x, int ldx,
                                                         int rows, float *__restrict__ y, int ldy) {
    const int lane = threadIdx.x & 31;
    const int j = int(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int row0 = int(blockIdx.y) * R;
    if (j >= m) return;
    const int live = min(R, rows - row0);
    const int nb = (n + 127) / 128;
    const uint8_t *wr = w + size_t(j) * n;
    const float *sr = ws + size_t(j / 128) * nb;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int d = lane * 16; d < n; d += 512) {
        const uint4 pk = *reinterpret_cast<const uint4 *>(wr + d);
        const float s = sr[d >> 7];
        float wv[16];
        const uint32_t pw[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            wv[4 * q] = e4m3(pw[q]);
            wv[4 * q + 1] = e4m3(pw[q] >> 8);
            wv[4 * q + 2] = e4m3(pw[q] >> 16);
            wv[4 * q + 3] = e4m3(pw[q] >> 24);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < live) {
                const uint4 *xp = reinterpret_cast<const uint4 *>(x + size_t(row0 + r) * ldx + d);
                const uint4 xa = xp[0], xb = xp[1];
                const uint32_t px[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                float part = 0.f;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    part = fmaf(bf2f(uint16_t(px[q])), wv[2 * q], part);
                    part = fmaf(bf2f(uint16_t(px[q] >> 16)), wv[2 * q + 1], part);
                }
                acc[r] = fmaf(part, s, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (lane == 0 && r < live) y[size_t(row0 + r) * ldy + j] = v;
    }
}
__global__ void q_fp8_dequant_kernel(const uint8_t *__restrict__ w, const float *__restrict__ ws, int m,
                                     int n, uint16_t *__restrict__ out) {
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (i >= size_t(m) * n) return;
    const int j = int(i / n), d = int(i % n);
    const int nb = (n + 127) / 128;
    const float s = ws[size_t(j / 128) * nb + (d >> 7)];
    const uint4 pk = *reinterpret_cast<const uint4 *>(w + i);
    const uint32_t pw[4] = {pk.x, pk.y, pk.z, pk.w};
    uint32_t o[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint32_t b = pw[q];
        o[2 * q] = uint32_t(f2bf(e4m3(b) * s)) | (uint32_t(f2bf(e4m3(b >> 8) * s)) << 16);
        o[2 * q + 1] = uint32_t(f2bf(e4m3(b >> 16) * s)) | (uint32_t(f2bf(e4m3(b >> 24) * s)) << 16);
    }
    uint4 *op = reinterpret_cast<uint4 *>(out + i);
    op[0] = make_uint4(o[0], o[1], o[2], o[3]);
    op[1] = make_uint4(o[4], o[5], o[6], o[7]);
}
}  // namespace

void q_fp8_gemv(const uint8_t *w, const float *ws, int m, int n, const uint16_t *x, int ldx, int rows,
                float *y, int ldy, hipStream_t s) {
    if (n % 16 || ldx % 8) throw std::runtime_error("q_fp8_gemv: n must be a multiple of 16");
    const unsigned gx = unsigned((m + 7) / 8);
    if (rows == 1) {
        q_fp8_gemv_kernel<1><<<dim3(gx, 1), 256, 0, s>>>(w, ws, m, n, x, ldx, rows, y, ldy);
    } else if (rows == 2) {
        q_fp8_gemv_kernel<2><<<dim3(gx, 1), 256, 0, s>>>(w, ws, m, n, x, ldx, rows, y, ldy);
    } else if (rows <= 4) {
        q_fp8_gemv_kernel<4><<<dim3(gx, 1), 256, 0, s>>>(w, ws, m, n, x, ldx, rows, y, ldy);
    } else {
        q_fp8_gemv_kernel<8><<<dim3(gx, unsigned((rows + 7) / 8)), 256, 0, s>>>(w, ws, m, n, x, ldx, rows,
                                                                              y, ldy);
    }
    HIP_CHECK(hipGetLastError());
}
void q_fp8_dequant(const uint8_t *w, const float *ws, int m, int n, uint16_t *out, hipStream_t s) {
    if (n % 16) throw std::runtime_error("q_fp8_dequant: n must be a multiple of 16");
    const size_t total = size_t(m) * n / 16;
    q_fp8_dequant_kernel<<<unsigned((total + 255) / 256), 256, 0, s>>>(w, ws, m, n, out);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth

// ------------------------------------------------------------------ split-K gated attention
namespace behemoth {
namespace {
// Block = 8 waves for (row r, kv head kh, split). Each wave owns 32-position chunks: lane = one
// position (full hd-dim dots for all G = nq/nkv q heads of the group, q in LDS), then the 32
// probabilities are broadcast and every lane accumulates hd/32 output dims. One partial (m, l,
// acc[hd]) per (row, q head, split, wave); q_attn_combine merges them.
template <int G>
__global__ __launch_bounds__(256) void q_attn_split_kernel(const uint16_t *__restrict__ q,
                                                           const uint16_t *__restrict__ kc,
                                                           const uint16_t *__restrict__ vc, int rows,
                                                           int pos0, int nq, int nkv, int S, int nsplit,
                                                           float scale, float *__restrict__ pml,
                                                           float *__restrict__ pacc, const int *__restrict__ pt) {
    constexpr int HD = 256, DPL = HD / 32;
    const int split = int(blockIdx.x) % nsplit;
    const int kh = (int(blockIdx.x) / nsplit) % nkv;
    const int r = int(blockIdx.x) / nsplit / nkv;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ float qs[G][HD];
    for (int i = threadIdx.x; i < G * HD; i += 256) {
        const int g = i / HD, d = i % HD;
        qs[g][d] = bf2f(q[long(r) * nq * 2 * HD + long(kh * G + g) * 2 * HD + d]) * scale;
    }
    __syncthreads();
    const int len = pos0 + r + 1;
    const int t_beg = split * S, t_end = min(len, (split + 1) * S);
    float m[G], l[G], acc[G][DPL];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -1e30f;
        l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[g][i] = 0.f;
    }
    const long kstride = long(nkv) * HD;
    for (int base = t_beg + wave * 32; base < t_end; base += 8 * 32) {
        const int t = base + lane;
        const bool ok = t < t_end;
        float dot[G];
#pragma unroll
        for (int g = 0; g < G; ++g) dot[g] = 0.f;
        if (ok) {
            const uint4 *kp = reinterpret_cast<const uint4 *>(kc + kvrow(pt, t) * kstride + long(kh) * HD);
#pragma unroll 4
            for (int c = 0; c < HD / 8; ++c) {
                const uint4 kv = kp[c];
                const uint32_t w4[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float k0 = bf2f(uint16_t(w4[e])), k1 = bf2f(uint16_t(w4[e] >> 16));
#pragma unroll
                    for (int g = 0; g < G; ++g)
                        dot[g] = fmaf(qs[g][c * 8 + 2 * e + 1], k1, fmaf(qs[g][c * 8 + 2 * e], k0, dot[g]));
                }
            }
        }
        float p[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float cm = ok ? dot[g] : -1e30f;
            for (int off = 16; off > 0; off >>= 1) cm = fmaxf(cm, __shfl_xor(cm, off, 32));
            const float mn = fmaxf(m[g], cm);
            const float corr = __expf(m[g] - mn);
            m[g] = mn;
            p[g] = ok ? __expf(dot[g] - mn) : 0.f;
            l[g] = l[g] * corr + p[g];
#pragma unroll
            for (int i = 0; i < DPL; ++i) acc[g][i] *= corr;
        }
        const int nj = min(32, t_end - base);
        for (int j = 0; j < nj; ++j) {
            const uint4 vv = *reinterpret_cast<const uint4 *>(vc + kvrow(pt, base + j) * kstride + long(kh) * HD +
                                                              lane * DPL);
            const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
            float v[DPL];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[2 * e] = bf2f(uint16_t(w4[e]));
                v[2 * e + 1] = bf2f(uint16_t(w4[e] >> 16));
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float pg = __shfl(p[g], j, 32);
#pragma unroll
                for (int i = 0; i < DPL; ++i) acc[g][i] = fmaf(pg, v[i], acc[g][i]);
            }
        }
    }
    const int NS = nsplit * 8, s = split * 8 + wave;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float ls = l[g];
        for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor(ls, off, 32);
        const long e = (long(r) * nq + kh * G + g) * NS + s;
        if (lane == 0) {
            pml[2 * e] = m[g];
            pml[2 * e + 1] = ls;
        }
        float4 *ap = reinterpret_cast<float4 *>(pacc + e * HD + lane * DPL);
        ap[0] = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
        ap[1] = make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
    }
}
// one block (hd threads) per (row, q head): merge NS partials, * sigmoid(gate)
__global__ void q_attn_combine_kernel(const float *pml, const float *pacc, int NS, const uint16_t *G,
                                      uint16_t *out, int nq, int hd) {
    const long rh = blockIdx.x;
    const int d = threadIdx.x;
    float M = -1e30f;
    for (int s = 0; s < NS; ++s) M = fmaxf(M, pml[2 * (rh * NS + s)]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < NS; ++s) {
        const long e = rh * NS + s;
        const float l = pml[2 * e + 1];
        if (l <= 0.f) continue;
        const float w = __expf(pml[2 * e] - M);
        den += w * l;
        num += w * pacc[e * hd + d];
    }
    const long r = rh / nq, h = rh % nq;
    const float gt = bf2f(G[r * nq * hd + h * hd + d]);
    out[r * nq * hd + h * hd + d] = f2bf(num / den / (1.0f + __expf(-gt)));
}
}  // namespace

size_t q_attn_split_scratch(int rows, int nq, int nkv, int hd, int max_len) {
    int nsplit = 1;
    const int want = std::max(1, 1024 / std::max(1, rows * nkv));
    nsplit = std::max(1, std::min(want, (max_len + 255) / 256));
    return size_t(rows) * nq * nsplit * 8 * size_t(hd + 2);
}

void q_attn_split(const uint16_t *q, const uint16_t *kc, const uint16_t *vc, const uint16_t *G,
                  uint16_t *out, int rows, int pos0, int nq, int nkv, int hd, float scale, float *scratch,
                  size_t scratch_floats, hipStream_t s, const int *pt) {
    if (hd != 256) throw std::runtime_error("q_attn_split: head_dim 256 only");
    const int len = pos0 + rows;
    const int want = std::max(1, 1024 / std::max(1, rows * nkv));
    const int nsplit = std::max(1, std::min(want, (len + 255) / 256));
    const int S = ((len + nsplit - 1) / nsplit + 255) / 256 * 256;
    const int NS = nsplit * 8;
    const size_t ne = size_t(rows) * nq * NS;
    if (ne * size_t(hd + 2) > scratch_floats) throw std::runtime_error("q_attn_split: scratch too small");
    float *pml = scratch, *pacc = scratch + 2 * ne;
    const unsigned blocks = unsigned(rows * nkv * nsplit);
    switch (nq / nkv) {
    case 1: q_attn_split_kernel<1><<<blocks, 256, 0, s>>>(q, kc, vc, rows, pos0, nq, nkv, S, nsplit, scale, pml, pacc, pt); break;
    case 2: q_attn_split_kernel<2><<<blocks, 256, 0, s>>>(q, kc, vc, rows, pos0, nq, nkv, S, nsplit, scale, pml, pacc, pt); break;
    case 4: q_attn_split_kernel<4><<<blocks, 256, 0, s>>>(q, kc, vc, rows, pos0, nq, nkv, S, nsplit, scale, pml, pacc, pt); break;
    case 6: q_attn_split_kernel<6><<<blocks, 256, 0, s>>>(q, kc, vc, rows, pos0, nq, nkv, S, nsplit, scale, pml, pacc, pt); break;
    case 8: q_attn_split_kernel<8><<<blocks, 256, 0, s>>>(q, kc, vc, rows, pos0, nq, nkv, S, nsplit, scale, pml, pacc, pt); break;
    default: throw std::runtime_error("q_attn_split: unsupported GQA group");
    }
    HIP_CHECK(hipGetLastError());
    q_attn_combine_kernel<<<unsigned(rows * nq), unsigned(hd), 0, s>>>(pml, pacc, NS, G, out, nq, hd);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth

// ------------------------------------------------------------------ Gated DeltaNet, whole chunk
namespace behemoth {
namespace {
// conv[t][c] = silu(sum_i w[c][i] * x(pos0 + t - K + 1 + i)); x(p) = qkvz row p - pos0 (this chunk)
// or the ring (raw inputs of earlier positions, ring[c][p % K]).
__global__ void q_delta_conv_kernel(const float *qkvz, long W, const uint16_t *conv_w, const float *ring,
                                    int T, int pos0, int conv_dim, int K, float *conv) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= long(T) * conv_dim) return;
    const int t = int(i / conv_dim), c = int(i % conv_dim);
    const int tp = pos0 + t;
    float a = 0.f;
    for (int k = 0; k < K; ++k) {
        const int p = tp - K + 1 + k;
        float x = 0.f;
        if (p >= pos0) x = qkvz[long(p - pos0) * W + c];
        else if (p >= 0) x = ring[long(c) * K + (p % K)];
        a += x * bf2f(conv_w[long(c) * K + k]);
    }
    conv[i] = silu_f(a);
}
__global__ void q_delta_ring_kernel(const float *qkvz, long W, float *ring, int T, int pos0, int conv_dim,
                                    int K) {
    const int c = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    for (int p = max(pos0, pos0 + T - K); p < pos0 + T; ++p) ring[long(c) * K + (p % K)] = qkvz[long(p - pos0) * W + c];
}
// block-wide sum returned to every thread (KD/32 waves)
template <int NW>
__device__ __forceinline__ float bsum_all(float v, float *red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) r += red[i];
    __syncthreads();
    return r;
}
// One block per v-head (KD threads), all T tokens sequentially with the state column S[:, d] in
// registers (thread d = value dim d).
template <int KD>
__global__ __launch_bounds__(KD) void q_delta_seq_kernel(const float *__restrict__ qkvz, long W,
                                                         const float *__restrict__ conv,
                                                         const uint16_t *__restrict__ dn_norm,
                                                         const float *__restrict__ a_dt, float *__restrict__ S,
                                                         uint16_t *__restrict__ out, int T, int conv_dim,
                                                         int value_dim, int nk, int nv, float eps) {
    const int hv = blockIdx.x, d = threadIdx.x;
    const int kh = hv / (nv / nk), key_dim = nk * KD;
    float s[KD];
    float *Sh = S + long(hv) * KD * KD;
#pragma unroll
    for (int dk = 0; dk < KD; ++dk) s[dk] = Sh[long(dk) * KD + d];
    __shared__ float sh_q[KD], sh_k[KD];
    const float A = __expf(a_dt[hv]), dtb = a_dt[nv + hv], nw = bf2f(dn_norm[d]);
    for (int t = 0; t < T; ++t) {
        const float *cr = conv + long(t) * conv_dim;
        const float *row = qkvz + long(t) * W;
        const float qv = cr[kh * KD + d], kv = cr[key_dim + kh * KD + d], vv = cr[2 * key_dim + hv * KD + d];
        __shared__ float red[KD / 32];
        const float qs = bsum_all<KD / 32>(qv * qv, red);
        const float ks = bsum_all<KD / 32>(kv * kv, red);
        sh_q[d] = qv * rsqrtf(qs + 1e-6f) * rsqrtf(float(KD));
        sh_k[d] = kv * rsqrtf(ks + 1e-6f);
        __syncthreads();
        const float beta = 1.0f / (1.0f + __expf(-row[conv_dim + value_dim + hv]));
        const float decay = __expf(-A * softplus_f(row[conv_dim + value_dim + nv + hv] + dtb));
        float kv_mem = 0.f;
#pragma unroll
        for (int dk = 0; dk < KD; ++dk) {
            s[dk] *= decay;
            kv_mem = fmaf(s[dk], sh_k[dk], kv_mem);
        }
        const float delta = (vv - kv_mem) * beta;
        float o = 0.f;
#pragma unroll
        for (int dk = 0; dk < KD; ++dk) {
            s[dk] = fmaf(sh_k[dk], delta, s[dk]);
            o = fmaf(s[dk], sh_q[dk], o);
        }
        const float os = bsum_all<KD / 32>(o * o, red);
        const float z = row[conv_dim + long(hv) * KD + d];
        out[long(t) * value_dim + long(hv) * KD + d] = f2bf(o * rsqrtf(os / KD + eps) * nw * silu_f(z));
        __syncthreads();
    }
#pragma unroll
    for (int dk = 0; dk < KD; ++dk) Sh[long(dk) * KD + d] = s[dk];
}
}  // namespace

void q_delta_chunk(const float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm, const float *a_dt,
                   float *ring, float *S, uint16_t *out, float *conv_scratch, int T, int pos0, int conv_dim,
                   int value_dim, int kd, int vd, int nk, int nv, int K, float eps, hipStream_t s) {
    if (kd != 128 || vd != 128) throw std::runtime_error("q_delta_chunk: kd == vd == 128 only");
    const long W = long(conv_dim) + value_dim + 2 * nv;
    const long n = long(T) * conv_dim;
    q_delta_conv_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(qkvz, W, conv_w, ring, T, pos0, conv_dim, K,
                                                                  conv_scratch);
    HIP_CHECK(hipGetLastError());
    q_delta_ring_kernel<<<unsigned((conv_dim + 255) / 256), 256, 0, s>>>(qkvz, W, ring, T, pos0, conv_dim, K);
    HIP_CHECK(hipGetLastError());
    q_delta_seq_kernel<128><<<unsigned(nv), 128, 0, s>>>(qkvz, W, conv_scratch, dn_norm, a_dt, S, out, T,
                                                         conv_dim, value_dim, nk, nv, eps);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth

namespace behemoth {
namespace {
__global__ void q_residual_add2_kernel(uint16_t *state, const float *a, const float *b, long total) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) state[i] = f2bf(bf2f(state[i]) + (a[i] + b[i]));
}
}  // namespace
void q_residual_add2(uint16_t *state, const float *a, const float *b, long rows, int n, hipStream_t s) {
    const long total = rows * n;
    q_residual_add2_kernel<<<unsigned((total + 255) / 256), 256, 0, s>>>(state, a, b, total);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth

// ------------------------------------------------------------------ decode helpers (fewer host ops)
namespace behemoth {
namespace {
__global__ __launch_bounds__(256) void q_zc_norm_to_kernel(uint16_t *dst, const uint16_t *src, int n,
                                                           const uint16_t *w, float eps) {
    const uint16_t *xr = src + long(blockIdx.x) * n;
    uint16_t *yr = dst + long(blockIdx.x) * n;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = bf2f(xr[i]);
        ss += v * v;
    }
    ss = block_sum(ss);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = f2bf(bf2f(xr[i]) * inv * (1.0f + bf2f(w[i])));
}
// bf16 weights [m][n], same wave-per-feature layout as q_fp8_gemv (8 bf16 per 128-bit load).
template <int R>
__global__ __launch_bounds__(256) void q_bf16_gemv_kernel(const uint16_t *__restrict__ w, int m, int n,
                                                          const uint16_t *__restrict__ x, int ldx, int rows,
                                                          float *__restrict__ y, int ldy) {
    const int lane = threadIdx.x & 31;
    const int j = int(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (j >= m) return;
    const uint16_t *wr = w + size_t(j) * n;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int d = lane * 8; d < n; d += 256) {
        const uint4 pk = *reinterpret_cast<const uint4 *>(wr + d);
        const uint32_t pw[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < rows) {
                const uint4 xa = *reinterpret_cast<const uint4 *>(x + size_t(r) * ldx + d);
                const uint32_t px[4] = {xa.x, xa.y, xa.z, xa.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    acc[r] = fmaf(bf2f(uint16_t(px[q])), bf2f(uint16_t(pw[q])), acc[r]);
                    acc[r] = fmaf(bf2f(uint16_t(px[q] >> 16)), bf2f(uint16_t(pw[q] >> 16)), acc[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
        if (lane == 0 && r < rows) y[size_t(r) * ldy + j] = v;
    }
}
__global__ void q_copy_u16_kernel(uint16_t *dst, const uint16_t *src, long n) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
}  // namespace
void q_zc_norm_to(uint16_t *dst, const uint16_t *src, long rows, int n, const uint16_t *w, float eps,
                  hipStream_t s) {
    q_zc_norm_to_kernel<<<unsigned(rows), 256, 0, s>>>(dst, src, n, w, eps);
    HIP_CHECK(hipGetLastError());
}
void q_bf16_gemv(const uint16_t *w, int m, int n, const uint16_t *x, int ldx, int rows, float *y, int ldy,
                 hipStream_t s) {
    if (n % 8 || ldx % 8 || rows > 8) throw std::runtime_error("q_bf16_gemv: n % 8, rows <= 8");
    const unsigned gx = unsigned((m + 7) / 8);
    if (rows == 1) q_bf16_gemv_kernel<1><<<gx, 256, 0, s>>>(w, m, n, x, ldx, rows, y, ldy);
    else if (rows <= 4) q_bf16_gemv_kernel<4><<<gx, 256, 0, s>>>(w, m, n, x, ldx, rows, y, ldy);
    else q_bf16_gemv_kernel<8><<<gx, 256, 0, s>>>(w, m, n, x, ldx, rows, y, ldy);
    HIP_CHECK(hipGetLastError());
}
void q_copy_u16(uint16_t *dst, const uint16_t *src, long n, hipStream_t s) {
    q_copy_u16_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(dst, src, n);
    HIP_CHECK(hipGetLastError());
}
}  // namespace behemoth
