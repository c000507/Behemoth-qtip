#include "behemoth/gpu.hpp"
#include "device_numeric.hpp"

// Stream-ordered two-rank exchange through fine-grained pinned host memory.
// No peer access, no host-thread rendezvous, no stream synchronisation: the
// publishing kernel copies this rank's values into its own host buffer and
// releases a sequence flag; the consuming kernel acquires the peer's flag and
// reads the peer's host buffer over PCIe. Both ranks compute a + b in the same
// operand order, so the reduced residual streams stay bitwise identical.
namespace behemoth {
namespace {
__global__ void publish_kernel(const float4 *__restrict__ values, float4 *__restrict__ mine,
                               int n4, unsigned *flag, unsigned *counter, unsigned seq) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n4)
        mine[i] = values[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        // Last block to arrive publishes the flag after every block's stores.
        const unsigned done = atomicAdd(counter, 1u) + 1;
        if (done == gridDim.x) {
            __threadfence_system();
            __hip_atomic_store(flag, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            *counter = 0;
        }
    }
}
__device__ __forceinline__ void wait_flag(const unsigned *peer_flag, unsigned seq) {
    if (threadIdx.x == 0)
        while (__hip_atomic_load(peer_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq)
            __builtin_amdgcn_s_sleep(1);
    __syncthreads();
}
__global__ void reduce_kernel(float4 *__restrict__ values, const float4 *__restrict__ peer, int n4,
                              const unsigned *peer_flag, unsigned seq) {
    wait_flag(peer_flag, seq);
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n4) {
        const float4 a = values[i], b = peer[i];
        values[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}
__global__ void gather_kernel(const float4 *__restrict__ local, const float4 *__restrict__ peer,
                              float4 *__restrict__ all, int n4, int rank, const unsigned *peer_flag,
                              unsigned seq) {
    wait_flag(peer_flag, seq);
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n4) {
        all[size_t(rank) * size_t(n4) + size_t(i)] = local[i];
        all[size_t(1 - rank) * size_t(n4) + size_t(i)] = peer[i];
    }
}
// Fused all-reduce + scaled residual (item 4): after a+b the update
// state = bf16(bf16(state) + bf16(sum * row_scale[row])) matches
// residual_add_scaled on the reduced buffer bit for bit (float addition of
// the two rank partials is commutative, so both ranks stay identical).
__device__ __forceinline__ void residual_update(uint16_t *state, float sum, float scale) {
    const float t = device::bf16(device::bf16(sum * scale));
    state[0] = device::bf16(device::bf16(state[0]) + t);
}
__global__ void
reduce_residual_dev_kernel(float4 *__restrict__ values, const float4 *__restrict__ peer,
                           uint16_t *__restrict__ state, const float *__restrict__ row_scale,
                           int n4, int width4, const unsigned *peer_flag, unsigned seq) {
    wait_flag(peer_flag, seq);
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n4) {
        const float4 a = values[i], b = peer[i];
        const float4 sum = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        values[i] = sum;
        const float s = row_scale[size_t(i) / size_t(width4)];
        const size_t base = size_t(i) * 4;
        residual_update(state + base, sum.x, s);
        residual_update(state + base + 1, sum.y, s);
        residual_update(state + base + 2, sum.z, s);
        residual_update(state + base + 3, sum.w, s);
    }
}
unsigned blocks_for(int n4) {
    return unsigned((n4 + 255) / 256);
}
} // namespace
void zero_copy_publish(const float *values, float *mine_host_device_pointer, size_t n,
                       unsigned *flag_device_pointer, unsigned *counter, unsigned seq,
                       hipStream_t stream) {
    if (n % 4)
        throw std::invalid_argument("Zero-copy exchange needs a multiple of four floats");
    const int n4 = int(n / 4);
    publish_kernel<<<blocks_for(n4), 256, 0, stream>>>(
        reinterpret_cast<const float4 *>(values),
        reinterpret_cast<float4 *>(mine_host_device_pointer), n4, flag_device_pointer, counter,
        seq);
    HIP_CHECK(hipGetLastError());
}
void zero_copy_reduce(float *values, const float *peer_host_device_pointer, size_t n,
                      const unsigned *peer_flag_device_pointer, unsigned seq,
                      hipStream_t stream) {
    const int n4 = int(n / 4);
    reduce_kernel<<<blocks_for(n4), 256, 0, stream>>>(
        reinterpret_cast<float4 *>(values),
        reinterpret_cast<const float4 *>(peer_host_device_pointer), n4,
        peer_flag_device_pointer, seq);
    HIP_CHECK(hipGetLastError());
}
void zero_copy_gather(const float *local, const float *peer_host_device_pointer, float *all,
                      size_t n, int rank, const unsigned *peer_flag_device_pointer, unsigned seq,
                      hipStream_t stream) {
    const int n4 = int(n / 4);
    gather_kernel<<<blocks_for(n4), 256, 0, stream>>>(
        reinterpret_cast<const float4 *>(local),
        reinterpret_cast<const float4 *>(peer_host_device_pointer),
        reinterpret_cast<float4 *>(all), n4, rank, peer_flag_device_pointer, seq);
    HIP_CHECK(hipGetLastError());
}
void zero_copy_reduce_residual_dev(float *values, const float *peer_host, BFloat16 *state,
                                   const float *row_scale, size_t n, int width,
                                   const unsigned *peer_flag_device_pointer, unsigned seq,
                                   hipStream_t stream) {
    if (n % 4 || width % 4)
        throw std::invalid_argument("Fused residual exchange needs multiples of four floats");
    const int n4 = int(n / 4);
    reduce_residual_dev_kernel<<<blocks_for(n4), 256, 0, stream>>>(
        reinterpret_cast<float4 *>(values),
        reinterpret_cast<const float4 *>(peer_host),
        reinterpret_cast<uint16_t *>(state), row_scale, n4, int(width / 4),
        peer_flag_device_pointer, seq);
    HIP_CHECK(hipGetLastError());
}
} // namespace behemoth
