// C ABI over the runtime kernels. PyTorch owns all storage and stream ordering.
#include "behemoth/qtip_gpu.hpp"
#include <string>

namespace {
thread_local std::string error;
template <class F> int checked(F f) noexcept {
    try { f(); error.clear(); return 0; }
    catch (const std::exception &e) { error = e.what(); return -1; }
}
}
extern "C" {
const char *qtip_training_error() { return error.c_str(); }
int qtip_training_abi() { return 1; }
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
int qtip_train_decode(const uint16_t *codes, const uint32_t *lut, int m, int n,
                     uint16_t *out, void *stream) {
    return checked([&] {
        behemoth::qtip::decode_dense(behemoth::qtip::LinearView{codes, n, m}, lut,
            reinterpret_cast<behemoth::BFloat16 *>(out), static_cast<hipStream_t>(stream));
    });
}
}
