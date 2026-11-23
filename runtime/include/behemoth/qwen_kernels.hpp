#pragma once
// Qwen3.5 runtime host-side kernel launchers (src/qwen_kernels.hip).
#include "gpu.hpp"
#include "qtip.hpp"
#include <cstddef>
#include <cstdint>

namespace behemoth {

// In-place row Hadamard on X (rows x n, row stride ld): hadU (transpose = false) / hadUt.
// Supports n = K * P with P a power of two <= 1024 and K in {1, 12, 20, 28}.
void q_hadamard_rows(float *X, long rows, int n, long ld, bool transpose, hipStream_t s);
// u = bf16(hadUt(SU (.) x')), x' = bf16(x * rsqrt(mean(x²)+eps) * (1+norm)) when norm non-null.
void q_input_transform(const uint16_t *x, long rows, int ldx, const uint16_t *norm, float eps,
                       const float *su, int n, uint16_t *u, int ldu, hipStream_t s);
// out = bf16(bf16(z * sv))
void q_sv_cast(const float *z, const float *sv, uint16_t *out, long rows, int n, long ldz,
               hipStream_t s);
// out[r][ldo + c] = z[r][ldz + c] * sv[c] (f32; for the delta-layer z assembly)
void q_sv_cast_f32(const float *z, const float *sv, float *out, long rows, int n, long ldz, int ldo,
                   hipStream_t s);
// zero-centred rmsnorm in place on bf16 rows: x = bf16(x*rsqrt(mean(x²)+eps)*(1+w))
void q_zc_norm(uint16_t *x, long rows, int n, const uint16_t *w, float eps, hipStream_t s);
// state = bf16(state + bf16(z * sv))
void q_sv_add(const float *z, const float *sv, uint16_t *state, long rows, int n, long ldz, int lds,
              hipStream_t s);
// h = bf16(silu(bf16(zg * svg)) * bf16(zu * svu)); z/zu row stride ld
void q_gateup(const float *zg, const float *zu, const float *svg, const float *svu, uint16_t *h,
              long rows, int F, long ld, hipStream_t s);
// state[r][:] = embed[tokens[r]]
void q_embed_gather(const uint16_t *embed, const int *tokens, uint16_t *state, long rows, int H,
                    hipStream_t s);
// out = sum_s partial[s] (slices x n -> n)
void q_sum_slices(const float *partial, int slices, size_t n, float *out, hipStream_t s);

// Gated full attention (single sequence): in-place zc-norm + partial rope on q/k (positions pos0+r),
// raw gate extraction, k/v append to the bf16 caches [maxctx][nkv][hd].
void q_qk_norm_rope(uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *G, uint16_t *kc, uint16_t *vc,
                    int rows, int pos0, int nq, int nkv, int hd, int rot, float theta,
                    const uint16_t *q_norm, const uint16_t *k_norm, float eps, hipStream_t s);
// out[r][h*hd+d] = softmax(q·K/sqrt(hd)) · V, gated by sigmoid(G); one block per (row, head).
void q_attn(const uint16_t *q, const uint16_t *kc, const uint16_t *vc, const uint16_t *G,
            uint16_t *out, int rows, int pos0, int nq, int nkv, int hd, float scale, hipStream_t s);
// One Gated DeltaNet step at token t of T (single sequence): qkvz rows [T][conv_dim+value_dim+2nv].
void q_delta_step(const float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm, const float *a_dt,
                  float *ring, float *S, uint16_t *out, int T, int t, int conv_dim, int value_dim,
                  int kd, int vd, int nk, int nv, int K, float eps, hipStream_t s);

// ------------------------------------------------------------------ MoE (softmax top-k renorm routing)
// Deterministic softmax -> top-k (ties to the lowest index) -> renormalise, one block per row,
// matching the CPU reference. rlog: f32 [rows][E]; wts: f32 [rows][k]; ids: i32 [rows][k];
// cnt: i32 [E] += 1 per routed pair (pre-zeroed).
void q_moe_route(const float *rlog, int rows, int E, int k, float *wts, int *ids, int *cnt, hipStream_t s);
// base: i32 [E] = exclusive scan of cnt (single block).
void q_moe_scan(const int *cnt, int *base, int E, hipStream_t s);
// Re-runs the selection and places each (row, w) into [base[e], base[e]+cnt[e]) via atomicAdd on
// run (pre-zeroed). slot_tok: i32 [rows*k]; slot_w: f32 [rows*k].
void q_moe_route_fill(const float *rlog, int rows, int E, int k, const int *base, int *run, int *slot_tok,
                      float *slot_w, hipStream_t s);
// dst[i][c] = bf16(src[slot_tok[i]][c]); src rows have stride n. i < nslots.
void q_moe_gather(const uint16_t *src, int n, const int *slot_tok, uint16_t *dst, int nslots,
                  hipStream_t s);
// y[tok[i]][c] += slot_w[i] * z[i][c] * sv[c]; y f32 [rows][n], z f32 [nslots][n] (row stride n).
void q_moe_scatter_sv_add(float *y, const float *z, const float *sv, const int *slot_tok,
                          const float *slot_w, int nslots, int n, hipStream_t s);
// state[r][:] = bf16(state[r][:] + y[r][:]); state bf16, y f32, rows x n.
void q_moe_residual_add(uint16_t *state, const float *y, long rows, int n, hipStream_t s);
// dst[r][col0 + c] = src[r][c]; f32 row-block scatter (assembling bf16-linear qkvz rows).
void q_scatter_f32(float *dst, int ldd, int col0, const float *src, int lds, long rows, int cols,
                   hipStream_t s);
// state[r][:] = bf16(state[r][:] + g[r] * z[r][c] * sv[c]); g = sigmoid(sg[r]) when sg non-null
// else 1. z f32 [rows][n] (row stride n); shared-expert epilogue.
void q_moe_sv_add_gated(const float *z, const float *sv, const float *sg, uint16_t *state, long rows,
                        int n, hipStream_t s);

}  // namespace behemoth
