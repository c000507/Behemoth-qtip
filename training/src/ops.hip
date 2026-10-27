#include "../../runtime/src/device_numeric.hpp"
#include "hip_ops.hpp"
#include <cmath>
namespace qat::gpu {
using behemoth::device::bf16;
__device__ float f16(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ float read(View a, int r, int c) { return a.p[size_t(r) * a.ld + c]; }
__device__ float &at(View a, int r, int c) { return a.p[size_t(r) * a.ld + c]; }
__device__ float sum(float x) {
    __shared__ float s[8];
    int l = threadIdx.x % 32, w = threadIdx.x / 32;
    for (int i = 16; i; i /= 2)
        x += __shfl_down(x, i);
    if (l == 0)
        s[w] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? s[l] : 0;
    if (w == 0)
        for (int i = 16; i; i /= 2)
            x += __shfl_down(x, i);
    if (threadIdx.x == 0)
        s[0] = x;
    __syncthreads();
    return s[0];
}
__global__ void bin(View d, View a, View b, int mode, bool round) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(d.rows) * d.cols)
        return;
    int r = i / d.cols, c = i % d.cols;
    float x = read(a, r, c), y = read(b, b.rows == 1 ? 0 : r, b.cols == 1 ? 0 : c);
    float z = mode == 0 ? x + y : mode == 1 ? x * y : x;
    if (round)
        z = f16(bf16(z));
    at(d, r, c) = z;
}
void binary(View d, View a, View b, int mode, bool round, hipStream_t s) {
    bin<<<(size_t(d.rows) * d.cols + 255) / 256, 256, 0, s>>>(d, a, b, mode, round);
}
void rounded(View d, hipStream_t s) { binary(d, d, d, 2, true, s); }
__global__ void convert(View a, uint16_t *b, bool to) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(a.rows) * a.cols)
        return;
    int r = i / a.cols, c = i % a.cols;
    if (to)
        b[i] = bf16(read(a, r, c));
    else
        at(a, r, c) = f16(b[i]);
}
void cast(View a, uint16_t *b, bool to, hipStream_t s) {
    convert<<<(size_t(a.rows) * a.cols + 255) / 256, 256, 0, s>>>(a, b, to);
}
__global__ void reduction(View a, View b, View d, bool scalar) {
    int c = blockIdx.x;
    if (scalar) {
        size_t n = size_t(a.rows) * a.cols;
        float x = 0;
        for (size_t i = size_t(c) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256)
            x += read(a, i / a.cols, i % a.cols) * read(b, i / b.cols, i % b.cols);
        x = sum(x);
        if (threadIdx.x == 0)
            atomicAdd(d.p, x);
    } else {
        float x = 0;
        for (int r = threadIdx.x; r < a.rows; r += 256)
            x += read(a, r, c) * read(b, r, c);
        x = sum(x);
        if (threadIdx.x == 0)
            d.p[c] += x;
    }
}
void reduce(View a, View b, View d, bool scalar, hipStream_t s) {
    reduction<<<scalar ? 256 : a.cols, 256, 0, s>>>(a, b, d, scalar);
}
__global__ void norm(View x, View w, View y, View rr, float eps) {
    int r = blockIdx.x;
    float ss = 0;
    for (int c = threadIdx.x; c < x.cols; c += 256) {
        float v = read(x, r, c);
        ss += v * v;
    }
    float inv = rsqrtf(sum(ss) / x.cols + eps);
    if (threadIdx.x == 0)
        rr.p[r] = inv;
    for (int c = threadIdx.x; c < x.cols; c += 256)
        at(y, r, c) = f16(bf16(read(x, r, c) * inv * w.p[c]));
}
void rms(View x, View w, View y, View rr, float eps, hipStream_t s) {
    norm<<<x.rows, 256, 0, s>>>(x, w, y, rr, eps);
}
__global__ void norm_back(View x, View rr, View w, View dy, View dx, View dw) {
    int r = blockIdx.x;
    float inv = rr.p[r], dot = 0;
    for (int c = threadIdx.x; c < x.cols; c += 256)
        dot += read(dy, r, c) * w.p[c] * read(x, r, c) * inv;
    dot = sum(dot) / x.cols;
    for (int c = threadIdx.x; c < x.cols; c += 256) {
        float xh = read(x, r, c) * inv, d = read(dy, r, c);
        at(dx, r, c) = inv * (d * w.p[c] - xh * dot);
        atomicAdd(dw.p + c, d * xh);
    }
}
void rms_back(View x, View rr, View w, View dy, View dx, View dw, hipStream_t s) {
    norm_back<<<x.rows, 256, 0, s>>>(x, rr, w, dy, dx, dw);
}
__global__ void rotary(View x, View y, int heads, int dim, float theta, const int32_t *pos, bool back) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(x.rows) * heads * (dim / 2))
        return;
    int j = i % (dim / 2), h = i / (dim / 2) % heads, r = i / (dim / 2) / heads;
    float angle = pos[r] * powf(theta, -2.f * j / dim), cs = cosf(angle), sn = sinf(angle),
          u = read(x, r, h * dim + j), v = read(x, r, h * dim + j + dim / 2);
    float a = u * cs + (back ? v : -v) * sn, b = v * cs + (back ? -u : u) * sn;
    at(y, r, h * dim + j) = back ? a : f16(bf16(a));
    at(y, r, h * dim + j + dim / 2) = back ? b : f16(bf16(b));
}
void rope(View x, View y, int h, int d, float theta, const int32_t *p, bool back, hipStream_t s) {
    rotary<<<(size_t(x.rows) * h * (d / 2) + 255) / 256, 256, 0, s>>>(x, y, h, d, theta, p, back);
}
__global__ void glu(View g, View u, View y) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(g.rows) * g.cols)
        return;
    int r = i / g.cols, c = i % g.cols;
    float x = read(g, r, c);
    at(y, r, c) = f16(bf16(x / (1 + expf(-x)) * read(u, r, c)));
}
void swiglu(View g, View u, View y, hipStream_t s) {
    glu<<<(size_t(g.rows) * g.cols + 255) / 256, 256, 0, s>>>(g, u, y);
}
__global__ void glu_back(View g, View u, View dy, View dg, View du) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(g.rows) * g.cols)
        return;
    int r = i / g.cols, c = i % g.cols;
    float x = read(g, r, c), sig = 1 / (1 + expf(-x)), d = read(dy, r, c);
    at(dg, r, c) = d * read(u, r, c) * sig * (1 + x * (1 - sig));
    at(du, r, c) = d * x * sig;
}
void swiglu_back(View g, View u, View dy, View dg, View du, hipStream_t s) {
    glu_back<<<(size_t(g.rows) * g.cols + 255) / 256, 256, 0, s>>>(g, u, dy, dg, du);
}
__global__ void embed(View x, const uint16_t *w, const int32_t *ids, int vocab) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(x.rows) * x.cols)
        return;
    int r = i / x.cols, c = i % x.cols;
    at(x, r, c) = ids[r] >= 0 && ids[r] < vocab ? f16(w[size_t(ids[r]) * x.cols + c]) : NAN;
}
void embedding(View x, const uint16_t *w, const int32_t *ids, int vocab, hipStream_t s) {
    embed<<<(size_t(x.rows) * x.cols + 255) / 256, 256, 0, s>>>(x, w, ids, vocab);
}
__global__ void ce(View x, const int32_t *labels, float *loss, float den, bool back) {
    int r = blockIdx.x, label = labels[r];
    if (label == -100) {
        if (threadIdx.x == 0)
            loss[r] = 0;
        if (back)
            for (int c = threadIdx.x; c < x.cols; c += 256)
                at(x, r, c) = 0;
        return;
    }
    if (label < 0 || label >= x.cols) {
        if (threadIdx.x == 0)
            loss[r] = NAN;
        return;
    }
    __shared__ float maxima[8];
    float top = -INFINITY;
    for (int c = threadIdx.x; c < x.cols; c += 256)
        top = fmaxf(top, read(x, r, c));
    for (int i = 16; i; i /= 2)
        top = fmaxf(top, __shfl_down(top, i));
    int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    if (lane == 0)
        maxima[wave] = top;
    __syncthreads();
    top = threadIdx.x < 8 ? maxima[lane] : -INFINITY;
    if (wave == 0)
        for (int i = 16; i; i /= 2)
            top = fmaxf(top, __shfl_down(top, i));
    if (threadIdx.x == 0)
        maxima[0] = top;
    __syncthreads();
    top = maxima[0];
    float ss = 0;
    for (int c = threadIdx.x; c < x.cols; c += 256)
        ss += expf(read(x, r, c) - top);
    float lse = top + logf(sum(ss));
    if (threadIdx.x == 0)
        loss[r] = lse - read(x, r, label);
    __syncthreads();
    if (back)
        for (int c = threadIdx.x; c < x.cols; c += 256)
            at(x, r, c) = (expf(read(x, r, c) - lse) - (c == label)) / den;
}
void loss(View x, const int32_t *l, float *out, float den, bool back, hipStream_t s) {
    ce<<<x.rows, 256, 0, s>>>(x, l, out, den, back);
}
__global__ void update(float *p, const float *g, float *m, float *v, size_t n, bool norm, uint64_t step,
                       float rate, float decay) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    float d = g[i], a = .9f * m[i] + .1f * d, b = .999f * v[i] + .001f * d * d;
    m[i] = a;
    v[i] = b;
    p[i] *= 1 - rate * (norm ? decay : 0);
    p[i] -= rate * (a / (1 - powf(.9f, float(step)))) / (sqrtf(b / (1 - powf(.999f, float(step)))) + 1e-8f);
}
void adam(float *p, const float *g, float *m, float *v, size_t n, bool norm, uint64_t step, float rate,
          float decay, hipStream_t s) {
    update<<<(n + 255) / 256, 256, 0, s>>>(p, g, m, v, n, norm, step, rate, decay);
}
} // namespace qat::gpu
