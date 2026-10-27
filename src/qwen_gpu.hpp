// Qwen3.5 forward kernels (Hessian collection). Declarations; implementations in qwen_kernels.hip.
#pragma once
#include "gpu.hpp"

namespace qtip {

// zero-centered rmsnorm: out = x*rsqrt(mean(x²)+eps)*(1+w); bf16 in/out, fp32 math
void k_q_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* out, long rows, int n, float eps, hipStream_t s);

// Qwen3.5 full-attention qkv split + q/k norm + partial RoPE + output-gate extraction.
// qkv: fp32 [rows][nq*2*hd + 2*nkv*hd], rows of ONE sequence (positions 0..rows-1).
// Layout: q rows per head [q(hd) | gate(hd)], then k heads, then v heads.
// Qp: bf16 [nq][rows][hd]; G: bf16 [rows][nq*hd] (gate, sigmoid applied later); Kp: [nkv][rows][hd];
// Vp: [nkv][rows][hd].
void k_q_qkv_rope(const float* qkv, uint16_t* Qp, uint16_t* G, uint16_t* Kp, uint16_t* Vp, int rows, int nq,
                  int nkv, int hd, int rot, float theta, const uint16_t* q_norm, const uint16_t* k_norm,
                  float eps, hipStream_t s);

// O: fp32 [nh][rows][hd] attention output; G: bf16 [rows][h0+nh ... ] gate rows (row stride ld);
// out: bf16 [rows][(h0+h)*hd + d] = O * sigmoid(gate)
void k_q_attn_out(const float* O, const uint16_t* G, uint16_t* out, int nh, int rows, int hd, int h0, long ld,
                  hipStream_t s);

// One Gated DeltaNet step for all sequences of the batch at absolute position t (local pos t%T per seq).
// qkvz: fp32 [seqs*T][conv_dim + value_dim + 2*nv] rows = [qkv | z | b(nv) | a(nv)] (unscaled GEMM output);
// conv_w: bf16 [conv_dim][K]; dn_norm: bf16 [vd]; a_dt: fp32 [2*nv] = A_log | dt_bias;
// ring: f32 [seqs][conv_dim][K]; S: f32 [seqs][nv][kd*vd]; out: bf16 [seqs*T][value_dim].
void k_q_delta_step(const float* qkvz, const uint16_t* conv_w, const uint16_t* dn_norm, const float* a_dt,
                    float* ring, float* S, uint16_t* out, int seqs, int T, int t, int conv_dim, int value_dim,
                    int kd, int vd, int nk, int nv, int K, float eps, hipStream_t s);

// fill with 1.0 (scale vectors for the fp8-oriented kernels when weights are unquantised)
void k_q_ones(float* x, long n, hipStream_t s);

}  // namespace qtip
