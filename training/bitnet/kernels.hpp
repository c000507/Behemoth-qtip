// CUDA kernels of the native BitNet b1.58 distillation trainer.
// Activations are FP32; GEMM operands are BF16 (`uint16_t`). For a BitLinear
// the operands hold exact small integers: ternary weights in {-1, 0, 1} and
// int8 activations in [-127, 127], so BF16 GEMMs with FP32 accumulation give
// exact integer sums (|sum| <= in * 127 < 2^24).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace bitnet::gpu {
void check(cudaError_t, const char *what);
// y = x * rstd * w (fp32) and optionally bf16(y); rstd[t] = 1/sqrt(mean(x^2) + eps)
void rmsnorm_fwd(const float *x, const float *w, float *y, uint16_t *yb, float *rstd, int rows, int n, float eps,
                 cudaStream_t);
// dx (+)= rstd (dy w - xhat mean(dy w xhat)); dw += sum_t dy xhat. accumulate selects += for dx.
void rmsnorm_bwd(const float *x, const float *w, const float *rstd, const float *dy, float *dx, float *dw, int rows,
                 int n, bool accumulate, cudaStream_t);
// Per-token absmax int8 quantisation written as exact BF16 integers; scale[t] = absmax/127.
// quant=false: xq = bf16(x), scale = 1 (dense teacher operand).
void act_quant(const float *x, uint16_t *xq, float *scale, int rows, int n, bool quant, cudaStream_t);
// Absmean ternarisation of a flat weight: scale = mean|w| + eps, wq = bf16(clip(round(w/scale),-1,1)).
// `scratch` is one float zeroed by the call.
void ternarize(const float *w, size_t n, uint16_t *wq, float *scale, float *scratch, cudaStream_t);
// Dense bf16 operand copy.
void to_bf16(const float *x, uint16_t *y, size_t n, cudaStream_t);
void to_f32(const uint16_t *x, float *y, size_t n, cudaStream_t);
// y[t, :] = alpha * x[t, :] * s[t] (+ optional in-place) as fp32, and/or bf16(y) for GEMM operands.
void scale_rows(const float *x, const float *s, float alpha, float *y, uint16_t *yb, int rows, int n, cudaStream_t);
// y *= s (scalar) in place
void scale_inplace(float *y, float s, size_t n, cudaStream_t);
// y += x
void add(float *y, const float *x, size_t n, cudaStream_t);
// RoPE in place on q [T, heads, dim] and k [T, kv, dim]; inverse applies the opposite rotation.
void rope(float *q, float *k, const int32_t *positions, int rows, int heads, int kv, int dim, float theta,
          bool inverse, cudaStream_t);
// Masked softmax rows: for head h, query t, keys j in [starts[t], t] are valid.
// P = softmax(scale * S) written fp32 and bf16. Each row is one block.
void softmax_fwd(const float *S, const int32_t *starts, float scale, float *P, uint16_t *Pb, int heads, int rows,
                 cudaStream_t);
// dS = scale * P (dP - sum(P dP)) written as bf16 operand (and fp32 if dS32 != nullptr).
void softmax_bwd(const float *P, const float *dP, float scale, uint16_t *dSb, float *dS32, int heads, int rows,
                 cudaStream_t);
// KL(Pt || Ps) summed over valid rows and heads: loss partial added to *loss; dS = (Ps - Pt) / norm
// as fp32 (unmasked entries 0).
void relation_kl(const float *Pt, const float *Ps, const int32_t *starts, float norm, float *loss, float *dS,
                 int heads, int rows, cudaStream_t);
// s = silu(g) * u ; backward: dg = ds u silu'(g), du = ds silu(g) (fp32 and bf16 copies)
void swiglu_fwd(const float *g, const float *u, float *s, size_t n, cudaStream_t);
void swiglu_bwd(const float *g, const float *u, const float *ds, float *dg, float *du, size_t n, cudaStream_t);
// h[t,:] = bf16 embedding row tokens[t]
void embedding(const uint16_t *E, const int32_t *tokens, float *h, int rows, int n, cudaStream_t);
// Loss on a chunk of rows. logits fp32 [rows, vocab]; labels (-100 = ignore). teacher fp32 logits (may be null).
// Accumulates sum CE, sum KL (temperature T, times T^2) into loss[0], loss[1] and counts in loss[2];
// writes dlogits = bf16((p_s - onehot) * ce_w / n_ce + (p_sT - p_tT) * T * kl_w / n_kl) when dlogits != null.
// Per-row CE is written to row_loss when non-null (for perplexity).
void lm_loss(const float *logits, const int32_t *labels, const float *teacher, int rows, int vocab, float T,
             float ce_w, float kl_w, float inv_n_ce, float inv_n_kl, float *loss, uint16_t *dlogits,
             float *row_loss, cudaStream_t);
// Sum of squares (fp64 accumulation per block, atomic into *out which must be zeroed by the caller).
void sumsq(const float *x, size_t n, double *out, cudaStream_t);
// AdamW with decoupled weight decay; gradients scaled by gscale before use.
void adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float beta1, float beta2, float eps,
           float decay, int step, float gscale, cudaStream_t);
void fill(float *p, float v, size_t n, cudaStream_t);
} // namespace bitnet::gpu
