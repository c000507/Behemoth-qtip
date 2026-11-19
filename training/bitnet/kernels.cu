#include "kernels.hpp"
#include <stdexcept>
#include <string>

namespace bitnet::gpu {
namespace {
constexpr int threads = 256;
__device__ __forceinline__ uint16_t bf(float x) {
    // round to nearest even (NaN preserved), identical to behemoth::BFloat16
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return uint16_t((u >> 16) | 0x40u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float fb(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Block reductions for blockDim.x == threads; `s` is shared scratch of 32 floats.
__device__ float block_sum(float v, float *s) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0)
        s[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? s[threadIdx.x] : 0.f;
    if (warp == 0)
        v = warp_sum(v);
    if (threadIdx.x == 0)
        s[0] = v;
    __syncthreads();
    return s[0];
}
__device__ float block_max(float v, float *s) {
    v = warp_max(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0)
        s[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? s[threadIdx.x] : -INFINITY;
    if (warp == 0)
        v = warp_max(v);
    if (threadIdx.x == 0)
        s[0] = v;
    __syncthreads();
    return s[0];
}
inline unsigned blocks(size_t n) { return unsigned((n + threads - 1) / threads); }

__global__ void rmsnorm_fwd_kernel(const float *x, const float *w, float *y, uint16_t *yb, float *rstd, int n,
                                   float eps) {
    __shared__ float s[32];
    const float *row = x + size_t(blockIdx.x) * n;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += row[i] * row[i];
    ss = block_sum(ss, s);
    const float r = rsqrtf(ss / float(n) + eps);
    if (threadIdx.x == 0 && rstd)
        rstd[blockIdx.x] = r;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i] * r * w[i];
        if (y)
            y[size_t(blockIdx.x) * n + i] = v;
        if (yb)
            yb[size_t(blockIdx.x) * n + i] = bf(v);
    }
}
__global__ void rmsnorm_bwd_kernel(const float *x, const float *w, const float *rstd, const float *dy, float *dx,
                                   float *dw, int n, bool accumulate) {
    __shared__ float s[32];
    const size_t off = size_t(blockIdx.x) * n;
    const float r = rstd[blockIdx.x];
    float dot = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        dot += dy[off + i] * w[i] * x[off + i] * r;
    dot = block_sum(dot, s) / float(n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xh = x[off + i] * r;
        const float g = r * (dy[off + i] * w[i] - xh * dot);
        if (accumulate)
            dx[off + i] += g;
        else
            dx[off + i] = g;
        atomicAdd(dw + i, dy[off + i] * xh);
    }
}
__global__ void act_quant_kernel(const float *x, uint16_t *xq, float *scale, int n, bool quant) {
    __shared__ float s[32];
    const size_t off = size_t(blockIdx.x) * n;
    if (!quant) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            xq[off + i] = bf(x[off + i]);
        if (threadIdx.x == 0)
            scale[blockIdx.x] = 1.f;
        return;
    }
    float m = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        m = fmaxf(m, fabsf(x[off + i]));
    m = block_max(m, s);
    const float sc = (m > 0 ? m : 1e-5f) / 127.f;
    if (threadIdx.x == 0)
        scale[blockIdx.x] = sc;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float q = rintf(x[off + i] / sc);
        q = fminf(fmaxf(q, -127.f), 127.f);
        xq[off + i] = bf(q);
    }
}
__global__ void abssum_kernel(const float *w, size_t n, float *out) {
    __shared__ float s[32];
    float a = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        a += fabsf(w[i]);
    a = block_sum(a, s);
    if (threadIdx.x == 0)
        atomicAdd(out, a);
}
__global__ void ternarize_kernel(const float *w, size_t n, uint16_t *wq, float *scale, const float *abssum) {
    const float sc = *abssum / float(n) + 1e-8f;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *scale = sc;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        float q = rintf(w[i] / sc);
        wq[i] = bf(fminf(fmaxf(q, -1.f), 1.f));
    }
}
__global__ void to_bf16_kernel(const float *x, uint16_t *y, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = bf(x[i]);
}
__global__ void to_f32_kernel(const uint16_t *x, float *y, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = fb(x[i]);
}
__global__ void scale_rows_kernel(const float *x, const float *s, float alpha, float *y, uint16_t *yb, int n) {
    const size_t off = size_t(blockIdx.x) * n;
    const float f = alpha * s[blockIdx.x];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[off + i] * f;
        if (y)
            y[off + i] = v;
        if (yb)
            yb[off + i] = bf(v);
    }
}
__global__ void scale_kernel(float *y, float s, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] *= s;
}
__global__ void add_kernel(float *y, const float *x, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += x[i];
}
__global__ void fill_kernel(float *y, float v, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = v;
}
// grid (rows, heads + kv); HF rotate_half pairing (d, d + dim/2); angles in fp32 like HF with fp32 cos/sin.
__global__ void rope_kernel(float *q, float *k, const int32_t *positions, int heads, int kv, int dim, float theta,
                            float sign) {
    const int t = blockIdx.x, h = blockIdx.y;
    float *v = h < heads ? q + (size_t(t) * heads + h) * dim : k + (size_t(t) * kv + (h - heads)) * dim;
    const float pos = float(positions[t]);
    const int half = dim / 2;
    for (int d = threadIdx.x; d < half; d += blockDim.x) {
        const float inv = powf(theta, -2.f * float(d) / float(dim));
        const float ang = pos * inv;
        float c, s;
        sincosf(ang, &s, &c);
        s *= sign;
        const float a = v[d], b = v[d + half];
        v[d] = a * c - b * s;
        v[d + half] = b * c + a * s;
    }
}
__global__ void softmax_fwd_kernel(const float *S, const int32_t *starts, float scale, float *P, uint16_t *Pb,
                                   int rows) {
    __shared__ float s[32];
    const int t = blockIdx.x, h = blockIdx.y;
    const size_t off = (size_t(h) * rows + t) * size_t(rows);
    const int lo = starts[t], hi = t;
    float m = -INFINITY;
    for (int j = lo + threadIdx.x; j <= hi; j += blockDim.x)
        m = fmaxf(m, S[off + j] * scale);
    m = block_max(m, s);
    float z = 0;
    for (int j = lo + threadIdx.x; j <= hi; j += blockDim.x)
        z += __expf(S[off + j] * scale - m);
    z = block_sum(z, s);
    const float inv = 1.f / z;
    for (int j = threadIdx.x; j < rows; j += blockDim.x) {
        const float p = (j >= lo && j <= hi) ? __expf(S[off + j] * scale - m) * inv : 0.f;
        P[off + j] = p;
        if (Pb)
            Pb[off + j] = bf(p);
    }
}
__global__ void softmax_bwd_kernel(const float *P, const float *dP, float scale, uint16_t *dSb, float *dS32,
                                   int rows) {
    __shared__ float s[32];
    const int t = blockIdx.x, h = blockIdx.y;
    const size_t off = (size_t(h) * rows + t) * size_t(rows);
    float dot = 0;
    for (int j = threadIdx.x; j <= t; j += blockDim.x)
        dot += P[off + j] * dP[off + j];
    dot = block_sum(dot, s);
    for (int j = threadIdx.x; j < rows; j += blockDim.x) {
        const float v = j <= t ? scale * P[off + j] * (dP[off + j] - dot) : 0.f;
        if (dSb)
            dSb[off + j] = bf(v);
        if (dS32)
            dS32[off + j] = v;
    }
}
__global__ void relation_kl_kernel(const float *Pt, const float *Ps, const int32_t *starts, float norm, float *loss,
                                   float *dS, int rows) {
    __shared__ float s[32];
    const int t = blockIdx.x, h = blockIdx.y;
    const size_t off = (size_t(h) * rows + t) * size_t(rows);
    const int lo = starts[t];
    float l = 0;
    for (int j = threadIdx.x; j < rows; j += blockDim.x) {
        const bool valid = j >= lo && j <= t;
        const float pt = valid ? Pt[off + j] : 0.f, ps = valid ? Ps[off + j] : 0.f;
        if (valid && pt > 0)
            l += pt * (__logf(fmaxf(pt, 1e-30f)) - __logf(fmaxf(ps, 1e-30f)));
        dS[off + j] = valid ? (ps - pt) * norm : 0.f;
    }
    l = block_sum(l, s);
    if (threadIdx.x == 0)
        atomicAdd(loss, l * norm);
}
__device__ __forceinline__ float silu(float g) { return g / (1.f + __expf(-g)); }
__global__ void swiglu_fwd_kernel(const float *g, const float *u, float *s, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        s[i] = silu(g[i]) * u[i];
}
__global__ void swiglu_bwd_kernel(const float *g, const float *u, const float *ds, float *dg, float *du, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float sg = 1.f / (1.f + __expf(-g[i]));
        const float si = g[i] * sg;
        du[i] = ds[i] * si;
        dg[i] = ds[i] * u[i] * (sg + si * (1.f - sg));
    }
}
__global__ void embedding_kernel(const uint16_t *E, const int32_t *tokens, float *h, int n) {
    const size_t off = size_t(blockIdx.x) * n;
    const uint16_t *row = E + size_t(tokens[blockIdx.x]) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        h[off + i] = fb(row[i]);
}
__global__ void lm_loss_kernel(const float *logits, const int32_t *labels, const float *teacher, int vocab, float T,
                               float ce_w, float kl_w, float inv_n_ce, float inv_n_kl, float *loss,
                               uint16_t *dlogits, float *row_loss) {
    __shared__ float s[32];
    const int t = blockIdx.x;
    const size_t off = size_t(t) * vocab;
    const int label = labels[t];
    if (label < 0) {
        if (dlogits)
            for (int i = threadIdx.x; i < vocab; i += blockDim.x)
                dlogits[off + i] = 0;
        if (threadIdx.x == 0 && row_loss)
            row_loss[t] = 0;
        return;
    }
    const float *z = logits + off;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        m = fmaxf(m, z[i]);
    m = block_max(m, s);
    float sum = 0, sumT = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        sum += __expf(z[i] - m);
        sumT += __expf((z[i] - m) / T);
    }
    sum = block_sum(sum, s);
    sumT = block_sum(sumT, s);
    const float lse = m + __logf(sum), lseT = m / T + __logf(sumT);
    const float ce = lse - z[label];
    float mt = -INFINITY, sumt = 0;
    const float *tz = teacher ? teacher + off : nullptr;
    if (tz) {
        for (int i = threadIdx.x; i < vocab; i += blockDim.x)
            mt = fmaxf(mt, tz[i]);
        mt = block_max(mt, s);
        for (int i = threadIdx.x; i < vocab; i += blockDim.x)
            sumt += __expf((tz[i] - mt) / T);
        sumt = block_sum(sumt, s);
    }
    const float lset = mt / T + __logf(sumt);
    float kl = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float lps = z[i] - lse;
        const float lpsT = z[i] / T - lseT;
        float g = ce_w * inv_n_ce * (__expf(lps) - (i == label ? 1.f : 0.f));
        if (tz) {
            const float lptT = tz[i] / T - lset;
            const float pt = __expf(lptT);
            kl += pt * (lptT - lpsT);
            g += kl_w * inv_n_kl * T * (__expf(lpsT) - pt);
        }
        if (dlogits)
            dlogits[off + i] = bf(g);
    }
    kl = block_sum(kl, s) * T * T;
    if (threadIdx.x == 0) {
        atomicAdd(loss, ce);
        atomicAdd(loss + 1, kl);
        atomicAdd(loss + 2, 1.f);
        if (row_loss)
            row_loss[t] = ce;
    }
}
__global__ void sumsq_kernel(const float *x, size_t n, double *out) {
    __shared__ double s[32];
    double a = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        a += double(x[i]) * double(x[i]);
    for (int o = 16; o; o >>= 1)
        a += __shfl_xor_sync(0xffffffffu, a, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0)
        s[warp] = a;
    __syncthreads();
    if (warp == 0) {
        a = threadIdx.x < (blockDim.x >> 5) ? s[threadIdx.x] : 0.0;
        for (int o = 16; o; o >>= 1)
            a += __shfl_xor_sync(0xffffffffu, a, o);
        if (threadIdx.x == 0)
            atomicAdd(out, a);
    }
}
__global__ void adamw_kernel(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2,
                             float eps, float decay, float c1, float c2, float gscale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float gi = g[i] * gscale;
    const float mi = b1 * m[i] + (1.f - b1) * gi;
    const float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    const float mh = mi * c1, vh = vi * c2;
    p[i] = p[i] * (1.f - lr * decay) - lr * mh / (sqrtf(vh) + eps);
}
} // namespace

void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void rmsnorm_fwd(const float *x, const float *w, float *y, uint16_t *yb, float *rstd, int rows, int n, float eps,
                 cudaStream_t s) {
    rmsnorm_fwd_kernel<<<rows, threads, 0, s>>>(x, w, y, yb, rstd, n, eps);
}
void rmsnorm_bwd(const float *x, const float *w, const float *rstd, const float *dy, float *dx, float *dw, int rows,
                 int n, bool accumulate, cudaStream_t s) {
    rmsnorm_bwd_kernel<<<rows, threads, 0, s>>>(x, w, rstd, dy, dx, dw, n, accumulate);
}
void act_quant(const float *x, uint16_t *xq, float *scale, int rows, int n, bool quant, cudaStream_t s) {
    act_quant_kernel<<<rows, threads, 0, s>>>(x, xq, scale, n, quant);
}
void ternarize(const float *w, size_t n, uint16_t *wq, float *scale, float *scratch, cudaStream_t s) {
    check(cudaMemsetAsync(scratch, 0, sizeof(float), s), "memset");
    const unsigned b = unsigned(std::min<size_t>(blocks(n), 1024));
    abssum_kernel<<<b, threads, 0, s>>>(w, n, scratch);
    ternarize_kernel<<<b, threads, 0, s>>>(w, n, wq, scale, scratch);
}
void to_bf16(const float *x, uint16_t *y, size_t n, cudaStream_t s) {
    to_bf16_kernel<<<blocks(n), threads, 0, s>>>(x, y, n);
}
void to_f32(const uint16_t *x, float *y, size_t n, cudaStream_t s) {
    to_f32_kernel<<<blocks(n), threads, 0, s>>>(x, y, n);
}
void scale_rows(const float *x, const float *sc, float alpha, float *y, uint16_t *yb, int rows, int n,
                cudaStream_t s) {
    scale_rows_kernel<<<rows, threads, 0, s>>>(x, sc, alpha, y, yb, n);
}
void scale_inplace(float *y, float v, size_t n, cudaStream_t s) { scale_kernel<<<blocks(n), threads, 0, s>>>(y, v, n); }
void add(float *y, const float *x, size_t n, cudaStream_t s) { add_kernel<<<blocks(n), threads, 0, s>>>(y, x, n); }
void fill(float *y, float v, size_t n, cudaStream_t s) { fill_kernel<<<blocks(n), threads, 0, s>>>(y, v, n); }
void rope(float *q, float *k, const int32_t *positions, int rows, int heads, int kv, int dim, float theta,
          bool inverse, cudaStream_t s) {
    const dim3 grid(rows, heads + kv);
    rope_kernel<<<grid, std::min(dim / 2, threads), 0, s>>>(q, k, positions, heads, kv, dim, theta,
                                                             inverse ? -1.f : 1.f);
}
void softmax_fwd(const float *S, const int32_t *starts, float scale, float *P, uint16_t *Pb, int heads, int rows,
                 cudaStream_t s) {
    const dim3 grid(rows, heads);
    softmax_fwd_kernel<<<grid, threads, 0, s>>>(S, starts, scale, P, Pb, rows);
}
void softmax_bwd(const float *P, const float *dP, float scale, uint16_t *dSb, float *dS32, int heads, int rows,
                 cudaStream_t s) {
    const dim3 grid(rows, heads);
    softmax_bwd_kernel<<<grid, threads, 0, s>>>(P, dP, scale, dSb, dS32, rows);
}
void relation_kl(const float *Pt, const float *Ps, const int32_t *starts, float norm, float *loss, float *dS,
                 int heads, int rows, cudaStream_t s) {
    const dim3 grid(rows, heads);
    relation_kl_kernel<<<grid, threads, 0, s>>>(Pt, Ps, starts, norm, loss, dS, rows);
}
void swiglu_fwd(const float *g, const float *u, float *sv, size_t n, cudaStream_t s) {
    swiglu_fwd_kernel<<<blocks(n), threads, 0, s>>>(g, u, sv, n);
}
void swiglu_bwd(const float *g, const float *u, const float *ds, float *dg, float *du, size_t n, cudaStream_t s) {
    swiglu_bwd_kernel<<<blocks(n), threads, 0, s>>>(g, u, ds, dg, du, n);
}
void embedding(const uint16_t *E, const int32_t *tokens, float *h, int rows, int n, cudaStream_t s) {
    embedding_kernel<<<rows, threads, 0, s>>>(E, tokens, h, n);
}
void lm_loss(const float *logits, const int32_t *labels, const float *teacher, int rows, int vocab, float T,
             float ce_w, float kl_w, float inv_n_ce, float inv_n_kl, float *loss, uint16_t *dlogits,
             float *row_loss, cudaStream_t s) {
    lm_loss_kernel<<<rows, threads, 0, s>>>(logits, labels, teacher, vocab, T, ce_w, kl_w, inv_n_ce, inv_n_kl, loss,
                                            dlogits, row_loss);
}
void sumsq(const float *x, size_t n, double *out, cudaStream_t s) {
    sumsq_kernel<<<unsigned(std::min<size_t>(blocks(n), 1024)), threads, 0, s>>>(x, n, out);
}
void adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float beta1, float beta2, float eps,
           float decay, int step, float gscale, cudaStream_t s) {
    const float c1 = 1.f / (1.f - powf(beta1, float(step))), c2 = 1.f / (1.f - powf(beta2, float(step)));
    adamw_kernel<<<blocks(n), threads, 0, s>>>(p, g, m, v, n, lr, beta1, beta2, eps, decay, c1, c2, gscale);
}
} // namespace bitnet::gpu
