#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#if defined(GLM53_CUDA)
#include <cuda_runtime.h>
namespace glm53::gpu {
using Stream = cudaStream_t;
using Error = cudaError_t;
constexpr auto success = cudaSuccess;
constexpr auto no_device = cudaErrorNoDevice;
constexpr auto insufficient_driver = cudaErrorInsufficientDriver;
inline auto error_string(Error e) { return cudaGetErrorString(e); }
inline auto set_device(int d) { return cudaSetDevice(d); }
inline auto device_count(int *n) { return cudaGetDeviceCount(n); }
inline auto stream_create(Stream *s) { return cudaStreamCreate(s); }
inline auto stream_destroy(Stream s) { return cudaStreamDestroy(s); }
inline auto sync(Stream s) { return cudaStreamSynchronize(s); }
inline auto alloc(void **p, size_t n) { return cudaMalloc(p, n); }
inline auto release(void *p) { return cudaFree(p); }
inline auto host_alloc(void **p, size_t n) { return cudaMallocHost(p, n); }
inline auto host_release(void *p) { return cudaFreeHost(p); }
inline auto zero(void *p, size_t n, Stream s) { return cudaMemsetAsync(p, 0, n, s); }
inline auto h2d(void *d, const void *h, size_t n, Stream s) {
    return cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s);
}
inline auto d2h(void *h, const void *d, size_t n, Stream s) {
    return cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, s);
}
inline auto d2d(void *d, const void *a, size_t n, Stream s) {
    return cudaMemcpyAsync(d, a, n, cudaMemcpyDeviceToDevice, s);
}
inline auto last_error() { return cudaGetLastError(); }
using Event = cudaEvent_t;
inline auto event_create(Event *e) { return cudaEventCreateWithFlags(e, cudaEventDisableTiming); }
inline auto event_destroy(Event e) { return cudaEventDestroy(e); }
inline auto event_record(Event e, Stream s) { return cudaEventRecord(e, s); }
inline auto stream_wait(Stream s, Event e) { return cudaStreamWaitEvent(s, e, 0); }
inline auto copy_peer(void *d, int dd, const void *a, int ad, size_t n, Stream s) {
    return cudaMemcpyPeerAsync(d, dd, a, ad, n, s);
}
inline auto memory_info(size_t *free, size_t *total) { return cudaMemGetInfo(free, total); }
} // namespace glm53::gpu
#else
#include <hip/hip_runtime.h>
namespace glm53::gpu {
using Stream = hipStream_t;
using Error = hipError_t;
constexpr auto success = hipSuccess;
constexpr auto no_device = hipErrorNoDevice;
constexpr auto insufficient_driver = hipErrorInsufficientDriver;
inline auto error_string(Error e) { return hipGetErrorString(e); }
inline auto set_device(int d) { return hipSetDevice(d); }
inline auto device_count(int *n) { return hipGetDeviceCount(n); }
inline auto stream_create(Stream *s) { return hipStreamCreate(s); }
inline auto stream_destroy(Stream s) { return hipStreamDestroy(s); }
inline auto sync(Stream s) { return hipStreamSynchronize(s); }
inline auto alloc(void **p, size_t n) { return hipMalloc(p, n); }
inline auto release(void *p) { return hipFree(p); }
inline auto host_alloc(void **p, size_t n) { return hipHostMalloc(p, n, hipHostMallocDefault); }
inline auto host_release(void *p) { return hipHostFree(p); }
inline auto zero(void *p, size_t n, Stream s) { return hipMemsetAsync(p, 0, n, s); }
inline auto h2d(void *d, const void *h, size_t n, Stream s) {
    return hipMemcpyAsync(d, h, n, hipMemcpyHostToDevice, s);
}
inline auto d2h(void *h, const void *d, size_t n, Stream s) {
    return hipMemcpyAsync(h, d, n, hipMemcpyDeviceToHost, s);
}
inline auto d2d(void *d, const void *a, size_t n, Stream s) {
    return hipMemcpyAsync(d, a, n, hipMemcpyDeviceToDevice, s);
}
inline auto last_error() { return hipGetLastError(); }
using Event = hipEvent_t;
inline auto event_create(Event *e) { return hipEventCreateWithFlags(e, hipEventDisableTiming); }
inline auto event_destroy(Event e) { return hipEventDestroy(e); }
inline auto event_record(Event e, Stream s) { return hipEventRecord(e, s); }
inline auto stream_wait(Stream s, Event e) { return hipStreamWaitEvent(s, e, 0); }
inline auto copy_peer(void *d, int dd, const void *a, int ad, size_t n, Stream s) {
    return hipMemcpyPeerAsync(d, dd, a, ad, n, s);
}
inline auto memory_info(size_t *free, size_t *total) { return hipMemGetInfo(free, total); }
} // namespace glm53::gpu
#endif
namespace glm53::gpu {
inline void checked(Error e, const char *what) {
    if (e != success)
        throw std::runtime_error(std::string(what) + ": " + error_string(e));
}
template <class T> struct Buffer {
    T *p = nullptr;
    size_t capacity = 0;
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&b) noexcept : p(std::exchange(b.p, nullptr)), capacity(std::exchange(b.capacity, 0)) {}
    Buffer &operator=(Buffer &&b) noexcept {
        if (this != &b) {
            clear();
            p = std::exchange(b.p, nullptr);
            capacity = std::exchange(b.capacity, 0);
        }
        return *this;
    }
    ~Buffer() { clear(); }
    void clear() {
        if (p)
            (void)release(p);
        p = nullptr;
        capacity = 0;
    }
    void ensure(size_t n, Stream s, bool preserve = false) {
        if (n <= capacity)
            return;
        if (n > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("GPU allocation size overflow");
        T *q = nullptr;
        checked(alloc(reinterpret_cast<void **>(&q), n * sizeof(T)), "GPU allocation");
        try {
            if (preserve && p) {
                checked(d2d(q, p, capacity * sizeof(T), s), "GPU buffer growth");
                checked(sync(s), "GPU buffer growth synchronization");
            }
        } catch (...) {
            (void)release(q);
            throw;
        }
        clear();
        p = q;
        capacity = n;
    }
    operator T *() { return p; }
    operator const T *() const { return p; }
};
template <class T> struct Pinned {
    T *p = nullptr;
    size_t capacity = 0;
    Pinned() = default;
    Pinned(const Pinned &) = delete;
    Pinned &operator=(const Pinned &) = delete;
    ~Pinned() {
        if (p)
            (void)host_release(p);
    }
    void ensure(size_t n) {
        if (n <= capacity)
            return;
        if (p) {
            checked(host_release(p), "free pinned memory");
            p = nullptr;
            capacity = 0;
        }
        checked(host_alloc(reinterpret_cast<void **>(&p), n * sizeof(T)), "allocate pinned memory");
        capacity = n;
    }
};
} // namespace glm53::gpu
