// Qwen3.5 forward kernels: zero-centered rmsnorm, gated full attention (q/k norm, partial RoPE, output
// gate) and the Gated DeltaNet recurrent step. See docs/QWEN35.md for the math reference.
#include <hip/hip_runtime.h>

#include "qwen_gpu.hpp"

namespace qtip {

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
}  // namespace

__global__ void ones_kernel(float* x, long n) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.0f;
}
void k_q_ones(float* x, long n, hipStream_t s) {
  ones_kernel<<<nblocks(n, 256), 256, 0, s>>>(x, n);
  HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ rmsnorm (zero-centred)
__global__ __launch_bounds__(256) void q_rmsnorm_kernel(const uint16_t* x, const uint16_t* w, uint16_t* out, int n,
                                                        float eps) {
  long row = blockIdx.x;
  const uint16_t* xr = x + row * n;
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
    out[row * n + i] = f2bf(bf2f(xr[i]) * inv * (1.0f + bf2f(w[i])));
}
void k_q_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* out, long rows, int n, float eps, hipStream_t s) {
  q_rmsnorm_kernel<<<unsigned(rows), 256, 0, s>>>(x, w, out, n, eps);
  HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ qkv split + norm + partial rope
// One block per (head-index, row); blockDim.x == hd. Head list: [0, nq) = q heads (rows [q|gate]),
// [nq, nq+nkv) = k heads, [nq+nkv, nq+2nkv) = v heads. q rows: q part normed + roped into Qp, gate part
// copied raw (sigmoid applied in k_q_attn_out). k rows normed + roped into Kp. v rows copied into Vp.
__global__ void q_qkv_rope_kernel(const float* qkv, uint16_t* Qp, uint16_t* G, uint16_t* Kp, uint16_t* Vp, int rows,
                                  int nq, int nkv, int hd, int rot, float theta, const uint16_t* q_norm,
                                  const uint16_t* k_norm, float eps) {
  int t = int(blockIdx.x % unsigned(rows));
  int h = int(blockIdx.x / unsigned(rows));
  long width = long(nq) * 2 * hd + long(2 * nkv) * hd;
  const float* row = qkv + long(t) * width;
  const int d = threadIdx.x;  // blockDim.x == hd
  if (h >= nq + nkv) {  // v
    int hv = h - nq - nkv;
    Vp[(long(hv) * rows + t) * hd + d] = f2bf(row[long(nq) * 2 * hd + long(nkv + hv) * hd + d]);
    return;
  }
  bool is_q = h < nq;
  const uint16_t* w = is_q ? q_norm : k_norm;
  long col0 = is_q ? long(h) * 2 * hd : long(nq) * 2 * hd + long(h - nq) * hd;
  float raw = row[col0 + d];
  float ss = block_sum(raw * raw);
  __shared__ float inv;
  if (threadIdx.x == 0) inv = rsqrtf(ss / hd + eps);
  __syncthreads();
  float sv = raw * inv * (1.0f + bf2f(w[d]));  // zero-centred norm over the full head
  uint16_t* o = is_q ? Qp + (long(h) * rows + t) * hd : Kp + (long(h - nq) * rows + t) * hd;
  if (d < rot / 2) {
    // rotate_half pair (d, d + rot/2); partner's normed value recomputed locally (uniform inv)
    float praw = row[col0 + d + rot / 2];
    float psv = praw * inv * (1.0f + bf2f(w[d + rot / 2]));
    float inv_freq = __powf(theta, -2.0f * float(d) / float(rot));
    float ang = float(t) * inv_freq;
    float c = cosf(ang), sn = sinf(ang);
    o[d] = f2bf(sv * c - psv * sn);
    o[d + rot / 2] = f2bf(psv * c + sv * sn);
  } else if (d >= rot) {
    o[d] = f2bf(sv);
  }
  if (is_q && d < hd) {  // gate part of the q row: raw copy (no norm, no rope)
    G[long(t) * long(nq) * hd + long(h) * hd + d] = f2bf(row[long(h) * 2 * hd + hd + d]);
  }
}
void k_q_qkv_rope(const float* qkv, uint16_t* Qp, uint16_t* G, uint16_t* Kp, uint16_t* Vp, int rows, int nq, int nkv,
                  int hd, int rot, float theta, const uint16_t* q_norm, const uint16_t* k_norm, float eps,
                  hipStream_t s) {
  QCHECK(rot > 0 && rot % 2 == 0 && rot < hd, "bad rotary dim");
  q_qkv_rope_kernel<<<unsigned((nq + 2 * nkv) * rows), hd, 0, s>>>(qkv, Qp, G, Kp, Vp, rows, nq, nkv, hd, rot,
                                                                   theta, q_norm, k_norm, eps);
  HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ attention output gate
__global__ void q_attn_out_kernel(const float* O, const uint16_t* G, uint16_t* out, int nh, int rows, int hd, int h0,
                                  long ld) {
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= long(nh) * rows * hd) return;
  int d = int(idx % hd);
  long rest = idx / hd;
  int t = int(rest % rows);
  int h = int(rest / rows);
  float g = bf2f(G[long(t) * ld + long(h0 + h) * hd + d]);
  out[long(t) * ld + long(h0 + h) * hd + d] = f2bf(O[idx] / (1.0f + __expf(-g)));
}
void k_q_attn_out(const float* O, const uint16_t* G, uint16_t* out, int nh, int rows, int hd, int h0, long ld,
                  hipStream_t s) {
  q_attn_out_kernel<<<nblocks(long(nh) * rows * hd, 256), 256, 0, s>>>(O, G, out, nh, rows, hd, h0, ld);
  HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ Gated DeltaNet step
// One block per (seq, v-head); blockDim.x == vd (== kd). fp32 math, fp32 state.
// qkvz row layout: [qkv(conv_dim) | z(value_dim) | b(nv) | a(nv)]; a_dt: fp32 [2*nv] = A_log | dt_bias.
__global__ void q_delta_step_kernel(const float* qkvz, const uint16_t* conv_w, const uint16_t* dn_norm,
                                    const float* a_dt, float* ring, float* S, uint16_t* out, int T, int t,
                                    int conv_dim, int value_dim, int kd, int vd, int nk, int nv, int K, float eps) {
  int s = blockIdx.x / nv;
  int hv = blockIdx.x % nv;
  const int group = nv / nk;
  const int kh = hv / group;
  const int key_dim = nk * kd;
  const int d = threadIdx.x;
  const long W = long(conv_dim) + value_dim + 2 * nv;
  const float* row = qkvz + (long(s) * T + t) * W;
  // conv (depthwise causal over the ring of raw qkv values, oldest..newest = ring[(t+1+i)%K])
  const int cq = kh * kd + d, ck = key_dim + kh * kd + d, cv = 2 * key_dim + hv * vd + d;
  float* ring_s = ring + long(s) * conv_dim * K;
  float qv = 0, kvv = 0, vv = 0;
  {
    float a = 0;
    const uint16_t* cw = conv_w + long(cq) * K;
    float* rg = ring_s + long(cq) * K;
    rg[t % K] = row[cq];
    for (int i = 0; i < K; ++i) a += rg[(t + 1 + i) % K] * bf2f(cw[i]);
    qv = silu_f(a);
    a = 0;
    cw = conv_w + long(ck) * K;
    rg = ring_s + long(ck) * K;
    rg[t % K] = row[ck];
    for (int i = 0; i < K; ++i) a += rg[(t + 1 + i) % K] * bf2f(cw[i]);
    kvv = silu_f(a);
    a = 0;
    cw = conv_w + long(cv) * K;
    rg = ring_s + long(cv) * K;
    rg[t % K] = row[cv];
    for (int i = 0; i < K; ++i) a += rg[(t + 1 + i) % K] * bf2f(cw[i]);
    vv = silu_f(a);
  }
  // l2norm (sum of squares + 1e-6); q additionally scaled by kd^-0.5
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
  // per-head gates
  const float beta = 1.0f / (1.0f + __expf(-row[conv_dim + value_dim + hv]));
  const float g = -__expf(a_dt[hv]) * softplus_f(row[conv_dim + value_dim + nv + hv] + a_dt[nv + hv]);
  const float decay = __expf(g);
  // S *= decay ; kv = S k ; delta = (v - kv) beta ; S += k (x) delta ; o = S q
  float* Sh = S + (long(s) * nv + hv) * kd * vd;
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
  // gated rmsnorm (plain weight) * silu(z_head)
  float os = block_sum(o * o);
  __shared__ float oinv;
  if (threadIdx.x == 0) oinv = rsqrtf(os / vd + eps);
  __syncthreads();
  const float z = row[conv_dim + long(hv) * vd + d];
  out[(long(s) * T + t) * value_dim + long(hv) * vd + d] =
      f2bf(o * oinv * bf2f(dn_norm[d]) * silu_f(z));
}
void k_q_delta_step(const float* qkvz, const uint16_t* conv_w, const uint16_t* dn_norm, const float* a_dt,
                    float* ring, float* S, uint16_t* out, int seqs, int T, int t, int conv_dim, int value_dim,
                    int kd, int vd, int nk, int nv, int K, float eps, hipStream_t s) {
  QCHECK(kd == vd && vd <= 256, "delta step kernel requires kd == vd <= 256");
  q_delta_step_kernel<<<unsigned(seqs * nv), vd, 0, s>>>(qkvz, conv_w, dn_norm, a_dt, ring, S, out, T, t, conv_dim,
                                                         value_dim, kd, vd, nk, nv, K, eps);
  HIP_CHECK(hipGetLastError());
}

// ------------------------------------------------------------------ MoE routing + gather/scatter
// Deterministic softmax -> top-k (ties to the lowest index) -> renormalise, one block per row.
__global__ void q_route_kernel(const float* rlog, int rows, int E, int k, float* wts, int* ids, int* cnt,
                               float* wsum) {
  extern __shared__ float p[];  // E probs
  const float* lg = rlog + long(blockIdx.x) * E;
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
      float w = best * inv;
      wts[long(blockIdx.x) * k + j] = w;
      ids[long(blockIdx.x) * k + j] = bi;
      atomicAdd(&wsum[bi], w * w);
      p[bi] = -1.0f;
    }
  }
  __syncthreads();
  for (int j = threadIdx.x; j < k; j += blockDim.x) atomicAdd(&cnt[ids[long(blockIdx.x) * k + j]], 1);
}
void k_q_route(const float* rlog, int rows, int E, int k, float* wts, int* ids, int* cnt, float* wsum,
               hipStream_t s) {
  QCHECK(E <= 4096 && k <= 64, "router kernel supports E <= 4096, k <= 64");
  q_route_kernel<<<unsigned(rows), 256, size_t(E) * 4, s>>>(rlog, rows, E, k, wts, ids, cnt, wsum);
  HIP_CHECK(hipGetLastError());
}

__global__ void q_scan_kernel(const int* cnt, int* base, int E) {
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int e = 0; e < E; ++e) {
      base[e] = acc;
      acc += cnt[e];
    }
  }
}
void k_q_scan(const int* cnt, int* base, int E, hipStream_t s) {
  q_scan_kernel<<<1, 32, 0, s>>>(cnt, base, E);
  HIP_CHECK(hipGetLastError());
}

__global__ void q_route_fill_kernel(const float* rlog, int rows, int E, int k, const int* base, int* run,
                                    int* slot_tok, float* slot_w) {
  extern __shared__ float p[];
  const float* lg = rlog + long(blockIdx.x) * E;
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
      float w = best * inv;
      int slot = base[bi] + atomicAdd(&run[bi], 1);
      slot_tok[slot] = int(blockIdx.x);
      slot_w[slot] = w;
      p[bi] = -1.0f;
    }
  }
}
void k_q_route_fill(const float* rlog, int rows, int E, int k, const int* base, int* run, int* slot_tok,
                    float* slot_w, hipStream_t s) {
  q_route_fill_kernel<<<unsigned(rows), 256, size_t(E) * 4, s>>>(rlog, rows, E, k, base, run, slot_tok, slot_w);
  HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_gather_kernel(const uint16_t* src, long rows, int n, const int* slot_tok,
                                    const float* scale, uint16_t* dst, int nslots) {
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= long(nslots) * n) return;
  int c = int(idx % n);
  int i = int(idx / n);
  float v = bf2f(src[long(slot_tok[i]) * n + c]);
  if (scale) v *= scale[i];
  dst[idx] = f2bf(v);
}
void k_q_moe_gather(const uint16_t* src, long rows, int n, const int* slot_tok, const float* scale,
                    uint16_t* dst, int nslots, hipStream_t s) {
  q_moe_gather_kernel<<<nblocks(long(nslots) * n, 256), 256, 0, s>>>(src, rows, n, slot_tok, scale, dst,
                                                                     nslots);
  HIP_CHECK(hipGetLastError());
}

__global__ void q_moe_scatter_add_kernel(float* y, const float* out, const int* slot_tok, const float* slot_w,
                                         int nslots, int n) {
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= long(nslots) * n) return;
  int c = int(idx % n);
  int i = int(idx / n);
  atomicAdd(&y[long(slot_tok[i]) * n + c], slot_w[i] * out[idx]);
}
void k_q_moe_scatter_add(float* y, const float* out, const int* slot_tok, const float* slot_w, int nslots,
                         int rows, int n, hipStream_t s) {
  q_moe_scatter_add_kernel<<<nblocks(long(nslots) * n, 256), 256, 0, s>>>(y, out, slot_tok, slot_w, nslots,
                                                                          n);
  HIP_CHECK(hipGetLastError());
}

__global__ void q_sigmoid_scale_add_kernel(float* y, const float* out, const float* sg, int rows, int n) {
  long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= long(rows) * n) return;
  int r = int(idx / n);
  float w = sg ? (1.0f / (1.0f + __expf(-sg[r]))) : 1.0f;
  y[idx] += w * out[idx];
}
void k_q_sigmoid_scale_add(float* y, const float* out, const float* sg, int rows, int n, hipStream_t s) {
  q_sigmoid_scale_add_kernel<<<nblocks(long(rows) * n, 256), 256, 0, s>>>(y, out, sg, rows, n);
  HIP_CHECK(hipGetLastError());
}

}  // namespace qtip
