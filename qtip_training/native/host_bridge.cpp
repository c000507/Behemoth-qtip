// CPU oracle calls the actual runtime implementation, not a second decoder.
#include "behemoth/qtip.hpp"
#include <algorithm>
#include <string>

namespace {
thread_local std::string error;
template <class F> int checked(F f) noexcept {
    try { f(); error.clear(); return 0; }
    catch (const std::exception &e) { error = e.what(); return -1; }
}
behemoth::qtip::HostLinear linear(const uint16_t *codes, const uint16_t *tlut, int m, int n) {
    if (m <= 0 || n <= 0 || m % 32 || n % 32)
        throw std::invalid_argument("invalid host bridge shape");
    behemoth::qtip::HostLinear l;
    l.in = n; l.out = m;
    l.trellis.assign(codes, codes + size_t(m) * size_t(n) / 8);
    l.tlut.assign(tlut, tlut + 1024);
    return l;
}
}
extern "C" {
const char *qtip_training_error() { return error.c_str(); }
int qtip_host_decode(const uint16_t *codes, const uint16_t *tlut, int m, int n,
                     int start, int count, int bf16, float *out) {
    return checked([&] {
        auto l = linear(codes, tlut, m, n);
        auto cb = behemoth::qtip::make_codebook(l.tlut);
        if (bf16) behemoth::qtip::decode_rows_bf16(l, cb, start, count, out);
        else behemoth::qtip::decode_rows(l, cb, start, count, out);
    });
}
int qtip_host_forward(const uint16_t *codes, const uint16_t *tlut, const float *su,
                     const float *sv, const float *x, int rows, int m, int n,
                     int bf16, int round_z, float *out) {
    return checked([&] {
        if (rows < 1) throw std::invalid_argument("invalid row count");
        auto l = linear(codes, tlut, m, n);
        l.su.assign(su, su + n); l.sv.assign(sv, sv + m);
        auto y = behemoth::qtip::reference_forward(l, std::span(x, size_t(rows) * size_t(n)), bf16, round_z);
        std::copy(y.begin(), y.end(), out);
    });
}
}
