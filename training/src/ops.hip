#include "view_device.hpp"
#include <cmath>
namespace qat::gpu {
using behemoth::device::bf16;
__global__ void fp8_decode_kernel(const uint8_t *x, uint16_t *y, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16(behemoth::device::fp8(x[i]));
}
void fp8_decode(const uint8_t *x, uint16_t *y, size_t n, hipStream_t stream) {
    fp8_decode_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
}
__device__ float f16(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ float read(View a, int r, int c) { return load(a, r, c); }
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
__global__ void mse_kernel(View a, View b, const int32_t *segments, float *loss, View gradient, float denominator) {
    int row = blockIdx.x;
    float ss = 0;
    for (int c = threadIdx.x; c < a.cols; c += blockDim.x) {
        float d = segments[row] >= 0 ? load(a, row, c) - load(b, row, c) : 0;
        ss = fmaf(d, d, ss);
        if (gradient.p) store(gradient, size_t(row) * gradient.ld + c, 2 * d / denominator);
    }
    ss = sum(ss);
    if (threadIdx.x == 0) loss[row] = ss / denominator;
}
void mse(View a, View b, const int32_t *segments, float *loss, View gradient, float denominator, hipStream_t stream) {
    mse_kernel<<<a.rows, 256, 0, stream>>>(a, b, segments, loss, gradient, denominator);
}
__global__ void bin(View d, View a, View b, int mode, bool round) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(d.rows) * d.cols)
        return;
    int r = i / d.cols, c = i % d.cols;
    float x = read(a, r, c), y = read(b, b.rows == 1 ? 0 : r, b.cols == 1 ? 0 : c);
    float z = mode == 0 ? x + y : mode == 1 ? x * y : x;
    if (round)
        z = rnd(z);
    store(d, r, c, z);
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
        b[i] = load_bits(a, size_t(r) * a.ld + c);
    else
        store(a, r, c, f16(b[i]));
}
void cast(View a, uint16_t *b, bool to, hipStream_t s) {
    convert<<<(size_t(a.rows) * a.cols + 255) / 256, 256, 0, s>>>(a, b, to);
}
__global__ void reduction(View a, View b, View d) {
    size_t n = size_t(a.rows) * a.cols;
    float x = 0;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256)
        x += read(a, int(i / a.cols), int(i % a.cols)) * read(b, int(i / b.cols), int(i % b.cols));
    x = sum(x);
    if (threadIdx.x == 0)
        atomicAdd(d.p, x);
}
void scalar_reduce(View a, View b, View d, hipStream_t s) {
    size_t blocks = (size_t(a.rows) * a.cols + 255) / 256;
    blocks = blocks < 256 ? blocks : 256;
    if (blocks)
        reduction<<<blocks, 256, 0, s>>>(a, b, d);
}
// Adjacent lanes read adjacent columns. Eight waves reduce 256 rows into one
// partial per column; a second pass owns each destination, including accumulation
// into gradients shared by tied projections and successive MLP chunks.
__global__ void column_partials(View a, View b, View scale, View partial) {
    __shared__ float sums[8][32];
    int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    int c = blockIdx.x * 32 + lane, first = blockIdx.y * reduction_rows;
    float value = 0;
    if (c < a.cols)
        for (int r = first + wave; r < a.rows && r < first + reduction_rows; r += 8) {
            float v = read(b, r, c);
            if (scale.p)
                v *= read(scale, r, 0);
            value += read(a, r, c) * v;
        }
    sums[wave][lane] = value;
    __syncthreads();
    if (wave == 0 && c < a.cols) {
        float total = 0;
        for (int w = 0; w < 8; ++w)
            total += sums[w][lane];
        store(partial, int(blockIdx.y), c, total);
    }
}
__global__ void column_finish(View partial, View d) {
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= d.cols)
        return;
    float total = 0;
    for (int r = 0; r < partial.rows; ++r)
        total += read(partial, r, c);
    d.p[c] += total;
}
void column_reduce(View a, View b, View scale, View partial, View d, hipStream_t s) {
    if (a.rows == 0)
        return;
    column_partials<<<dim3((a.cols + 31) / 32, partial.rows), 256, 0, s>>>(a, b, scale, partial);
    column_finish<<<(d.cols + 255) / 256, 256, 0, s>>>(partial, d);
}
__global__ void column_finish_factor(View partial, View factor, View d) {
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= d.cols)
        return;
    float total = 0;
    for (int r = 0; r < partial.rows; ++r)
        total += read(partial, r, c);
    d.p[c] += total * factor.p[c];
}
void column_reduce_factor(View a, View b, View scale, View factor, View partial, View d, hipStream_t s) {
    if (a.rows == 0)
        return;
    column_partials<<<dim3((a.cols + 31) / 32, partial.rows), 256, 0, s>>>(a, b, scale, partial);
    column_finish_factor<<<(d.cols + 255) / 256, 256, 0, s>>>(partial, factor, d);
}
// Finishes SV/Wscale gradients: dc[c] = sum of partials; gsv[c] += dc[c] * scale;
// gscale += sum_c dc[c] * sv[c] (one atomic per block).
__global__ void sv_finish(View partial, View sv, View scale, View gsv, View gscale) {
    int c = blockIdx.x * 256 + threadIdx.x;
    float dc = 0;
    if (c < gsv.cols) {
        for (int r = 0; r < partial.rows; ++r)
            dc += read(partial, r, c);
        gsv.p[c] += dc * scale.p[0];
    }
    float w = sum(c < gsv.cols ? dc * sv.p[c] : 0.f);
    if (threadIdx.x == 0)
        atomicAdd(gscale.p, w);
}
void sv_reduce(View dy, View c, View sv, View scale, View partial, View gsv, View gscale, hipStream_t s) {
    if (dy.rows == 0)
        return;
    column_partials<<<dim3((dy.cols + 31) / 32, partial.rows), 256, 0, s>>>(dy, c, View{}, partial);
    sv_finish<<<(gsv.cols + 255) / 256, 256, 0, s>>>(partial, sv, scale, gsv, gscale);
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
        store_rounded(y, size_t(r) * y.ld + c, read(x, r, c) * inv * w.p[c]);
}
void rms(View x, View w, View y, View rr, float eps, hipStream_t s) {
    norm<<<x.rows, 256, 0, s>>>(x, w, y, rr, eps);
}
__global__ void norm_back(View x, View rr, View w, View dy, View dx) {
    int r = blockIdx.x;
    float inv = rr.p[r], dot = 0;
    for (int c = threadIdx.x; c < x.cols; c += 256)
        dot += read(dy, r, c) * w.p[c] * read(x, r, c) * inv;
    dot = sum(dot) / x.cols;
    for (int c = threadIdx.x; c < x.cols; c += 256) {
        float xh = read(x, r, c) * inv, d = read(dy, r, c);
        store(dx, r, c, inv * (d * w.p[c] - xh * dot));
    }
}
void rms_back(View x, View rr, View w, View dy, View dx, hipStream_t s) {
    norm_back<<<x.rows, 256, 0, s>>>(x, rr, w, dy, dx);
}
__global__ void angles(View table, int dim, float theta, const int32_t *pos) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(table.rows) * (dim / 2))
        return;
    int r = i / (dim / 2), j = i % (dim / 2);
    float angle = pos[r] * powf(theta, -2.f * j / dim);
    store(table, r, j, cosf(angle));
    store(table, r, j + dim / 2, sinf(angle));
}
void rope_angles(View table, int dim, float theta, const int32_t *pos, hipStream_t s) {
    angles<<<(size_t(table.rows) * (dim / 2) + 255) / 256, 256, 0, s>>>(table, dim, theta, pos);
}
__global__ void rotary(View x, View y, int heads, int dim, View table, bool back) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(x.rows) * heads * (dim / 2))
        return;
    int j = i % (dim / 2), h = i / (dim / 2) % heads, r = i / (dim / 2) / heads;
    float cs = read(table, r, j), sn = read(table, r, j + dim / 2), u = read(x, r, h * dim + j),
          v = read(x, r, h * dim + j + dim / 2);
    float a = u * cs + (back ? v : -v) * sn, b = v * cs + (back ? -u : u) * sn;
    store(y, r, h * dim + j, back ? a : rnd(a));
    store(y, r, h * dim + j + dim / 2, back ? b : rnd(b));
}
void rope(View x, View y, int h, int d, View table, bool back, hipStream_t s) {
    rotary<<<(size_t(x.rows) * h * (d / 2) + 255) / 256, 256, 0, s>>>(x, y, h, d, table, back);
}
__global__ void glu(View g, View u, View y) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(g.rows) * g.cols)
        return;
    int r = i / g.cols, c = i % g.cols;
    float x = read(g, r, c);
    store_rounded(y, size_t(r) * y.ld + c, x / (1 + expf(-x)) * read(u, r, c));
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
    store(dg, r, c, d * read(u, r, c) * sig * (1 + x * (1 - sig)));
    store(du, r, c, d * x * sig);
}
void swiglu_back(View g, View u, View dy, View dg, View du, hipStream_t s) {
    glu_back<<<(size_t(g.rows) * g.cols + 255) / 256, 256, 0, s>>>(g, u, dy, dg, du);
}
__global__ void embed(View x, const uint16_t *w, const int32_t *ids, int vocab) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(x.rows) * x.cols)
        return;
    int r = i / x.cols, c = i % x.cols;
    store(x, r, c, ids[r] >= 0 && ids[r] < vocab ? f16(w[size_t(ids[r]) * x.cols + c]) : NAN);
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
                store(x, r, c, 0.f);
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
            store(x, r, c, (expf(read(x, r, c) - lse) - (c == label)) / den);
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
