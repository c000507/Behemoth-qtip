// C ABI over the runtime kernels. PyTorch owns all storage and stream ordering.
#include "behemoth/qtip_gpu.hpp"
#include <string>

namespace qtip_training {
void had_rows(const void *in, int in_bf16, long ld_in, int rows, int n, int transpose, const float *pre,
              int round_in, float *out32, long ld32, uint16_t *out16, long ld16, const float *post,
              hipStream_t stream);
void rms_had(const uint16_t *x, long ld, int rows, int n, const float *w, float eps, const float *su, uint16_t *xn,
             long ldxn, float *r, uint16_t *u, long ldu, hipStream_t stream);
void gateup(const float *z, long ldz, int rows, int inter, const float *svs_g, const float *svs_u, uint16_t *gate,
            uint16_t *up, uint16_t *product, long ldo, float *cg, float *cu, long ldc, hipStream_t stream);
void colsum(const float *a, long lda, const void *b, int b16, long ldb, const float *rs, int rows, int n,
            float *acc, hipStream_t stream);
void swiglu_bwd(const float *dxsu, const float *su, const uint16_t *gate, const uint16_t *up, float *dg, float *du,
                int rows, int cols, hipStream_t stream);
void rms_bwd(const uint16_t *x, long ldx, const float *r, const float *w, const float *su, const float *dxsu, long ldd,
             float *dh, long ldh, int rows, int n, hipStream_t stream);
void rope(const uint16_t *in, long ld, int rows, int heads, int head_dim, const float *cs, const float *sn, void *out,
          int backward, hipStream_t stream);
}

namespace {
thread_local std::string error;
template <class F> int checked(F f) noexcept {
    try { f(); error.clear(); return 0; }
    catch (const std::exception &e) { error = e.what(); return -1; }
}
}
extern "C" {
const char *qtip_training_error() { return error.c_str(); }
int qtip_training_abi() { return 3; }
int qtip_train_slices(int rows, int m, int n) {
    int result = -1;
    if (checked([&] { result = behemoth::qtip::matmul_slices(behemoth::qtip::LinearView{nullptr, n, m}, rows); })) return -1;
    return result;
}
int qtip_train_matmul(const uint16_t *codes, const uint32_t *lut, const uint16_t *x,
                     int rows, int m, int n, int slices, float *partial, float *out, void *stream) {
    return checked([&] {
        auto s = static_cast<hipStream_t>(stream);
        behemoth::qtip::matmul(behemoth::qtip::LinearView{codes, n, m}, lut,
            reinterpret_cast<const behemoth::BFloat16 *>(x), n, rows, partial, slices, s);
        behemoth::qtip::reduce_publish(partial, slices, size_t(rows) * size_t(m), false,
                                      out, nullptr, nullptr, nullptr, 0, s);
    });
}
int qtip_train_hadamard(const float *x, int rows, int n, int transpose, float *out, void *stream) {
    return checked([&] {
        behemoth::qtip::hadamard_rows(x, n, rows, transpose, out, static_cast<hipStream_t>(stream));
    });
}
// Fast-engine row transform (train_kernels.hip): see qtip_training/runtime_backend.py had().
int qtip_train_had(const void *in, int in_bf16, long ld_in, int rows, int n, int transpose,
                   const float *pre, int round_in, float *out32, long ld32, uint16_t *out16,
                   long ld16, const float *post, void *stream) {
    return checked([&] {
        qtip_training::had_rows(in, in_bf16, ld_in, rows, n, transpose, pre, round_in, out32, ld32,
                                out16, ld16, post, static_cast<hipStream_t>(stream));
    });
}
int qtip_train_rms_had(const uint16_t *x, long ld, int rows, int n, const float *w, float eps, const float *su,
                       uint16_t *xn, long ldxn, float *r, uint16_t *u, long ldu, void *stream) {
    return checked([&] { qtip_training::rms_had(x, ld, rows, n, w, eps, su, xn, ldxn, r, u, ldu,
                                                static_cast<hipStream_t>(stream)); });
}
int qtip_train_gateup(const float *z, long ldz, int rows, int inter, const float *svs_g, const float *svs_u,
                      uint16_t *gate, uint16_t *up, uint16_t *product, long ldo, float *cg, float *cu, long ldc,
                      void *stream) {
    return checked([&] { qtip_training::gateup(z, ldz, rows, inter, svs_g, svs_u, gate, up, product, ldo, cg, cu,
                                               ldc, static_cast<hipStream_t>(stream)); });
}
int qtip_train_colsum(const float *a, long lda, const void *b, int b16, long ldb, const float *rs, int rows, int n,
                      float *acc, void *stream) {
    return checked([&] { qtip_training::colsum(a, lda, b, b16, ldb, rs, rows, n, acc,
                                               static_cast<hipStream_t>(stream)); });
}
int qtip_train_swiglu_bwd(const float *dxsu, const float *su, const uint16_t *gate, const uint16_t *up, float *dg,
                          float *du, int rows, int cols, void *stream) {
    return checked([&] { qtip_training::swiglu_bwd(dxsu, su, gate, up, dg, du, rows, cols,
                                                   static_cast<hipStream_t>(stream)); });
}
int qtip_train_rms_bwd(const uint16_t *x, long ldx, const float *r, const float *w, const float *su,
                       const float *dxsu, long ldd, float *dh, long ldh, int rows, int n, void *stream) {
    return checked([&] { qtip_training::rms_bwd(x, ldx, r, w, su, dxsu, ldd, dh, ldh, rows, n,
                                                static_cast<hipStream_t>(stream)); });
}
int qtip_train_rope(const uint16_t *in, long ld, int rows, int heads, int head_dim, const float *cs,
                    const float *sn, void *out, int backward, void *stream) {
    return checked([&] { qtip_training::rope(in, ld, rows, heads, head_dim, cs, sn, out, backward,
                                             static_cast<hipStream_t>(stream)); });
}
int qtip_train_decode(const uint16_t *codes, const uint32_t *lut, int m, int n,
                     uint16_t *out, void *stream) {
    return checked([&] {
        behemoth::qtip::decode_dense(behemoth::qtip::LinearView{codes, n, m}, lut,
            reinterpret_cast<behemoth::BFloat16 *>(out), static_cast<hipStream_t>(stream));
    });
}
}
