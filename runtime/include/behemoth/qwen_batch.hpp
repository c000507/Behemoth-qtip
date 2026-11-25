#pragma once
// Qwen3.5 batched-serving kernels (src/qwen_batch.hip): W8A8 FP8 WMMA GEMMs (gfx12) with dynamic
// per-token x 128-column activation scales, paged KV gated attention, per-slot Gated DeltaNet decode
// step, device-flag tensor-parallel all-reduce through pinned host memory, top-k candidates.
#include "gpu.hpp"
#include <cstddef>
#include <cstdint>

namespace behemoth {

// ------------------------------------------------------------------ FP8 activations
// q[r][c] = e4m3(x[r][c] / s[r][c/128]), s = amax(block) / 448 (1 when the block is zero).
// x bf16 rows (row stride ldx), n % 128 == 0; q row stride n, s row stride n/128.
void q_act_quant(const uint16_t *x, int ldx, int rows, int n, uint8_t *q, float *s, hipStream_t st);
// Fused zero-centred rmsnorm (h = bf16(x * rsqrt(mean(x^2)+eps) * (1+w))) + q_act_quant(h).
void q_norm_quant(const uint16_t *x, int rows, int n, const uint16_t *w, float eps, uint16_t *h, uint8_t *q, float *s,
                  hipStream_t st);
// Same from f32 rows.
void q_act_quant_f32(const float *x, int ldx, int rows, int n, uint8_t *q, float *s, hipStream_t st);

// ------------------------------------------------------------------ W8A8 GEMMs
// y[r][j] = sum_k (q[r][k] * xs[r][k/128]) * (w[j][k] * ws[j/128][k/128])   (fp32 out, row stride ldy)
// w: e4m3 [m][n], ws: f32 [m/128][n/128] (m, n multiples of 128).
// Decode-sized (rows <= 64): split-K with deterministic partials. The number of K splits depends on
// (m, n) only, so every row's result is independent of the batch size (batch invariance).
int q_w8a8_splits(int m, int n);
size_t q_w8a8_small_scratch(int m, int n, int rows);  // floats of partials (0 when one split)
// x8/xs must hold ceil16(rows) readable rows. partial: q_w8a8_small_scratch floats (or null if 0).
void q_w8a8_small(const uint8_t *w, const float *ws, int m, int n, const uint8_t *x8, const float *xs, int rows,
                  float *y, int ldy, float *partial, hipStream_t st);
// Same, but the partials are only written (no reduction): for q_splitk_reduce_push consumers.
// Prefill-sized GEMM (128x128 tiles); x8/xs: rows readable rows.
void q_w8a8_big(const uint8_t *w, const float *ws, int m, int n, const uint8_t *x8, const float *xs, int rows,
                float *y, int ldy, hipStream_t st);
// Picks small (rows <= 64) or big.
void q_w8a8(const uint8_t *w, const float *ws, int m, int n, const uint8_t *x8, const float *xs, int rows,
            float *y, int ldy, float *partial, hipStream_t st);

// bf16 weights [m][n] GEMV for any row count (weights shared through L2 across row groups);
// deterministic per row (same order for every batch size).
void q_bf16_gemv_any(const uint16_t *w, int m, int n, const uint16_t *x, int ldx, int rows, float *y, int ldy,
                     hipStream_t st);
// fp32 -> fp8 128x128 block quantisation of a bf16 [m][n] matrix (load time; m padded to 128 rows of
// zeros by the caller when needed). ws [ceil(m/128)][n/128].
void q_quant_weight_bf16(const uint16_t *w, int m, int n, uint8_t *w8, float *ws, hipStream_t st);

}  // namespace behemoth

namespace behemoth {
// ------------------------------------------------------------------ TP all-reduce via host memory
// Host slots: nslot slots of slot_bytes at hbase (pinned, coherent, mapped); flags: 16 unsigned per
// slot (one cache line). seq / ctr: device counters (one pair per rank, zero-initialised).
// n: float count (multiple of 4). Push: writes the rank's partial (bf16: rounded; the pull rounds its own copy).
void q_ar_push(float *part, long n, void *hbase, long slot_bytes, bool bf16, unsigned *hflag, const unsigned *seq,
               unsigned *ctr, int nslot, hipStream_t st);
// Pull: waits for the peer's flag, state = bf16(state + (p0 + p1)), advances seq.
void q_ar_pull_add(uint16_t *state, const float *own, const void *peer_hbase, long slot_bytes, bool bf16, long n,
                   int rank, const unsigned *peer_flag, unsigned *seq, unsigned *ctr, int nslot, hipStream_t st);

// DMA variant of the pull: wait kernel (spins on the peer flag of exchange *seq), the caller copies
// the peer slot to device memory (hipMemcpyAsync), then the add (advances seq).
void q_ar_round_bf16(float *part, void *dst, long n, hipStream_t st);  // DMA push: bf16 copy + round own
void q_ar_flag(unsigned *hflag, const unsigned *seq, int nslot, hipStream_t st);  // raise flag of exchange *seq
void q_ar_wait(const unsigned *peer_flag, const unsigned *seq, int nslot, hipStream_t st);
void q_ar_add_local(uint16_t *state, const float *own, const void *peer_dev, bool bf16, long n, int rank, unsigned *seq,
                    unsigned *ctr, hipStream_t st);

// Whole-chunk Gated DeltaNet v2 (kd == vd == 128), same math / contract as q_delta_chunk; modifies
// the qkvz b|a columns in place (beta / decay). conv scratch f32 [T][conv_dim].
void q_delta_chunk2(float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm, const float *a_dt, float *ring,
                    float *S, uint16_t *out, float *conv, int T, int pos0, int conv_dim, int value_dim, int nk, int nv,
                    int K, float eps, hipStream_t st);
}  // namespace behemoth

namespace behemoth {
// ------------------------------------------------------------------ batched decode
// meta = [tokens | slots | positions] (3 x bmax ints); ptab [slots][maxpages] (64-token pages).
// Gated attention input z f32 [B][Wf]: per q head [q hd | gate hd], then k (nkv*hd), v (nkv*hd).
void q_rope_b(const float *z, int Wf, uint16_t *qout, uint16_t *G, uint16_t *kc, uint16_t *vc, const int *meta, int bmax,
              int B, const int *ptab, int maxpages, int nq, int nkv, int hd, int rot, float theta, const uint16_t *q_norm,
              const uint16_t *k_norm, float eps, hipStream_t st);
size_t q_attn_b_scratch(int bmax, int nq, int max_context);
void q_attn_b(const uint16_t *q, const uint16_t *kc, const uint16_t *vc, const uint16_t *G, uint16_t *out,
              const int *meta, int bmax, int B, const int *ptab, int maxpages, int nq, int nkv, int max_context,
              float scale, float *scratch, hipStream_t st);
// One DeltaNet step per row; ring [slots][conv_dim][K], S [slots][nv][128][128] (f32).
void q_delta_b(const float *qkvz, const uint16_t *conv_w, const uint16_t *dn_norm, const float *a_dt, float *ring, float *S,
               uint16_t *out, const int *meta, int bmax, int B, int conv_dim, int value_dim, int nk, int nv, int K,
               float eps, hipStream_t st);
// Top-K (unsorted, K <= 64) of each row of logits [rows][V]; ids offset by voff.
void q_topk(const float *logits, int rows, int V, int K, int voff, float *oval, int *oid, int ostride, hipStream_t st);
}  // namespace behemoth

namespace behemoth {
// Causal gated attention of `rows` consecutive positions pos0.. of one sequence (prefill), bf16 WMMA
// flash attention. Same contract as q_attn_split: q bf16 [rows][nq][2*hd] (q | gate), paged bf16 KV
// (pt: page table, 64-token pages; null: contiguous), out bf16 [rows][nq*hd]. hd 256.
void q_flash_prefill(const uint16_t *q, const uint16_t *kc, const uint16_t *vc, const uint16_t *G, uint16_t *out,
                     int rows, int pos0, int nq, int nkv, int hd, float scale, const int *pt, hipStream_t st);
}  // namespace behemoth
