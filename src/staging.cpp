// Device -> pageable host copies through reused pinned staging buffers.
//
// Why: a large hipMemcpy(DeviceToHost) into pageable memory (e.g. a fresh std::vector of a 3.3 GB Hessian) makes
// the HIP runtime lock the user pages on the fly through KFD SVM-range ioctls. On this box (gfx1201 / R9700,
// amdgpu DKMS 7.1.9, MES firmware 0x93) the TLB invalidations issued for those ranges are intermittently rejected by
// MES: dmesg shows bursts of "MES(1) failed to respond to msg=INVALIDATE_TLBS" (the driver ignores the failure and
// has no fallback, so the flush is simply lost). Copying into persistent pinned memory never triggers it, so all
// large D2H transfers go through here. H2D from pageable memory did not trigger it and is left as is.
#include <algorithm>
#include <cstring>

#include "gpu.hpp"

namespace qtip {

namespace {
constexpr size_t kStageBytes = 64ull << 20;
struct Stage {
  void* buf[2] = {nullptr, nullptr};
  hipEvent_t ev[2] = {nullptr, nullptr};
  bool init = false;
};
}  // namespace

void d2h_staged(void* dst, const void* src, size_t bytes, hipStream_t s) {
  // One staging pair per thread (each GPU worker is one thread). Intentionally never freed: process lifetime.
  static thread_local Stage st;
  if (!st.init) {
    for (int i = 0; i < 2; ++i) {
      HIP_CHECK(hipHostMalloc(&st.buf[i], kStageBytes, hipHostMallocDefault));
      HIP_CHECK(hipEventCreateWithFlags(&st.ev[i], hipEventDisableTiming));
    }
    st.init = true;
  }
  char* d = static_cast<char*>(dst);
  const char* sp = static_cast<const char*>(src);
  size_t prev_off = 0, prev_len = 0;
  int k = 0;
  for (size_t off = 0; off < bytes; off += kStageBytes, k ^= 1) {
    size_t len = std::min(kStageBytes, bytes - off);
    HIP_CHECK(hipMemcpyAsync(st.buf[k], sp + off, len, hipMemcpyDeviceToHost, s));
    HIP_CHECK(hipEventRecord(st.ev[k], s));
    if (prev_len) {  // drain the previous chunk while this one is in flight
      HIP_CHECK(hipEventSynchronize(st.ev[k ^ 1]));
      std::memcpy(d + prev_off, st.buf[k ^ 1], prev_len);
    }
    prev_off = off, prev_len = len;
  }
  if (prev_len) {
    HIP_CHECK(hipEventSynchronize(st.ev[k ^ 1]));
    std::memcpy(d + prev_off, st.buf[k ^ 1], prev_len);
  }
}

}  // namespace qtip
