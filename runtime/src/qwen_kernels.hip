// Qwen3.5 runtime kernels: generic Hadamard (K x P, K in {1,12,28}), fused input transform (zc-norm +
// SU + hadUt), epilogues (SV cast / residual add / gate-up), gated full attention (norm + partial rope +
// output gate, bf16 KV cache) and the Gated DeltaNet recurrent step.
#include <hip/hip_runtime.h>

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
    h[idx] = f2bf(bf2f(silu_f(g)) * u);
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
                                      const uint16_t *k_norm, float eps) {
    int r = blockIdx.x % rows;
    int h = blockIdx.x / rows;
    int d = threadIdx.x;
    if (h >= nq + nkv) {
        int hv = h - nq - nkv;
        uint16_t *vv = v + long(r) * nkv * hd + long(hv) * hd;
        uint16_t *vdst = vc + (long(pos0 + r) * nkv + hv) * hd;
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
    if (!is_q) {  // append k to cache
        uint16_t *kdst = kc + (long(pos0 + r) * nkv + long(h - nq)) * hd;
        kdst[d] = vec[d];
    }
}
void q_qk_norm_rope(uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *G, uint16_t *kc, uint16_t *vc,
                    int rows, int pos0, int nq, int nkv, int hd, int rot, float theta,
                    const uint16_t *q_norm, const uint16_t *k_norm, float eps, hipStream_t s) {
    q_qk_norm_rope_kernel<<<unsigned((nq + 2 * nkv) * rows), hd, 0, s>>>(q, k, v, G, kc, vc, rows,
                                                                         pos0, nq, nkv, hd, rot,
                                                                         theta, q_norm, k_norm, eps);
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
                                    int t, int conv_dim, int value_dim, int kd, int vd, int nk, int nv,
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
    rg_q[t % K] = row[cq];
    rg_k[t % K] = row[ck];
    rg_v[t % K] = row[cv];
    float qv = 0, kvv = 0, vv = 0;
    {
        const uint16_t *cw = conv_w + long(cq) * K;
        float a = 0;
        for (int i = 0; i < K; ++i) a += rg_q[(t + 1 + i) % K] * bf2f(cw[i]);
        qv = silu_f(a);
        cw = conv_w + long(ck) * K;
        a = 0;
        for (int i = 0; i < K; ++i) a += rg_k[(t + 1 + i) % K] * bf2f(cw[i]);
        kvv = silu_f(a);
        cw = conv_w + long(cv) * K;
        a = 0;
        for (int i = 0; i < K; ++i) a += rg_v[(t + 1 + i) % K] * bf2f(cw[i]);
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
                  float *ring, float *S, uint16_t *out, int T, int t, int conv_dim, int value_dim,
                  int kd, int vd, int nk, int nv, int K, float eps, hipStream_t s) {
    if (kd != vd || vd > 256) throw std::runtime_error("delta step requires kd == vd <= 256");
    q_delta_step_kernel<<<unsigned(nv), vd, 0, s>>>(qkvz, conv_w, dn_norm, a_dt, ring, S, out, T, t,
                                                    conv_dim, value_dim, kd, vd, nk, nv, K, eps);
    HIP_CHECK(hipGetLastError());
}

}  // namespace behemoth
