// Two-GPU all-reduce latency microbenchmark (no P2P) for decode-sized messages.
// Build: hipcc --offload-arch=gfx1201 -O3 -mno-wavefrontsize64 micro_coll.hip -o micro_coll -lrccl
//   a) "host": replica of behemoth HostTransport::reduce (D2H, sync, barrier, CPU sum, barrier, H2D, sync, barrier)
//   b) "rccl": ncclAllReduce with NCCL_P2P_DISABLE=1 (SHM path), stream-ordered
//   c) "zerocopy": stream-ordered kernel-driven exchange through pinned host memory:
//      each rank's kernel writes its partial to its own host buffer, bumps a host
//      flag (system-scope release), spins on the peer's flag (acquire), then reads
//      the peer's buffer and sums.  No host thread involvement, no stream sync.
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_), __FILE__,        \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define NC(x)                                                                                      \
    do {                                                                                           \
        ncclResult_t r_ = (x);                                                                     \
        if (r_ != ncclSuccess) {                                                                   \
            fprintf(stderr, "%s failed: %s\n", #x, ncclGetErrorString(r_));                        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

struct Barrier {
    std::mutex m;
    std::condition_variable cv;
    unsigned arrived = 0, gen = 0;
    void wait() {
        std::unique_lock l(m);
        unsigned g = gen;
        if (++arrived == 2) {
            arrived = 0;
            ++gen;
            cv.notify_all();
        } else
            cv.wait(l, [&] { return gen != g; });
    }
};

// ---- zero-copy kernels -------------------------------------------------------
// Phase 1: write own values into host buffer `mine` (float4 granularity), then
// one thread publishes `seq` to the flag with system-scope release.
__global__ void zc_publish(const float4 *__restrict__ values, float4 *__restrict__ mine, int n4,
                           unsigned *flag, unsigned *counter, unsigned seq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4)
        mine[i] = values[i];
    // Make all blocks' writes visible before the flag: device-memory arrival counter.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned done = atomicAdd(counter, 1u) + 1;
        if (done == gridDim.x) {
            __threadfence_system();
            __hip_atomic_store(flag, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            *counter = 0;
        }
    }
}
// Phase 2: wait for the peer's flag, then y = mine + peer (read peer over PCIe).
__global__ void zc_reduce(float4 *__restrict__ values, const float4 *__restrict__ peer, int n4,
                          const unsigned *peer_flag, unsigned seq) {
    if (threadIdx.x == 0) {
        while (__hip_atomic_load(peer_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq)
            __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 a = values[i], b = peer[i];
        values[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "all";
    int n = argc > 2 ? atoi(argv[2]) : 12288; // floats per rank (decode row: 12288)
    int iters = argc > 3 ? atoi(argv[3]) : 200;
    printf("# all-reduce of %d floats (%zu bytes), %d iterations, 2 GPUs no P2P\n", n,
           size_t(n) * 4, iters);
    int devs[2] = {0, 1};
    Barrier bar;

    auto run_host = [&] {
        float *host[2];
        for (auto &p : host)
            CK(hipHostMalloc(&p, size_t(n) * 4, hipHostMallocPortable));
        std::vector<double> lat;
        std::mutex lm;
        auto worker = [&](int rank) {
            CK(hipSetDevice(devs[rank]));
            hipStream_t s;
            CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
            float *d;
            CK(hipMalloc(&d, size_t(n) * 4));
            std::vector<float> h(n, float(rank + 1));
            CK(hipMemcpy(d, h.data(), size_t(n) * 4, hipMemcpyHostToDevice));
            for (int it = 0; it < iters + 20; ++it) {
                bar.wait();
                auto t0 = std::chrono::steady_clock::now();
                CK(hipMemcpyAsync(host[rank], d, size_t(n) * 4, hipMemcpyDeviceToHost, s));
                CK(hipStreamSynchronize(s));
                bar.wait();
                if (rank == 0)
                    for (int i = 0; i < n; ++i)
                        host[0][i] += host[1][i];
                bar.wait();
                CK(hipMemcpyAsync(d, host[0], size_t(n) * 4, hipMemcpyHostToDevice, s));
                CK(hipStreamSynchronize(s));
                bar.wait();
                auto t1 = std::chrono::steady_clock::now();
                if (it >= 20 && rank == 0) {
                    std::lock_guard g(lm);
                    lat.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                // reset values so the sum stays bounded (not timed)
                CK(hipMemcpy(d, h.data(), size_t(n) * 4, hipMemcpyHostToDevice));
            }
            std::vector<float> out(n);
            CK(hipMemcpy(out.data(), d, size_t(n) * 4, hipMemcpyDeviceToHost));
        };
        std::thread a(worker, 0), b(worker, 1);
        a.join();
        b.join();
        std::sort(lat.begin(), lat.end());
        printf("host-staged (current):  median %.1f us  p10 %.1f  p90 %.1f  min %.1f\n",
               lat[lat.size() / 2], lat[lat.size() / 10], lat[lat.size() * 9 / 10], lat[0]);
    };

    auto run_rccl = [&] {
        setenv("NCCL_P2P_DISABLE", "1", 1);
        setenv("NCCL_SHM_DISABLE", "0", 1);
        ncclComm_t comm[2];
        NC(ncclCommInitAll(comm, 2, devs));
        std::vector<double> lat;
        std::mutex lm;
        auto worker = [&](int rank) {
            CK(hipSetDevice(devs[rank]));
            hipStream_t s;
            CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
            float *d;
            CK(hipMalloc(&d, size_t(n) * 4));
            std::vector<float> h(n, float(rank + 1));
            CK(hipMemcpy(d, h.data(), size_t(n) * 4, hipMemcpyHostToDevice));
            for (int it = 0; it < 20; ++it)
                NC(ncclAllReduce(d, d, n, ncclFloat, ncclSum, comm[rank], s));
            CK(hipStreamSynchronize(s));
            // timed: 10 back-to-back all-reduces per sample, stream-ordered
            hipEvent_t e0, e1;
            CK(hipEventCreate(&e0));
            CK(hipEventCreate(&e1));
            for (int it = 0; it < iters / 10; ++it) {
                bar.wait();
                CK(hipEventRecord(e0, s));
                for (int k = 0; k < 10; ++k)
                    NC(ncclAllReduce(d, d, n, ncclFloat, ncclSum, comm[rank], s));
                CK(hipEventRecord(e1, s));
                CK(hipEventSynchronize(e1));
                float ms;
                CK(hipEventElapsedTime(&ms, e0, e1));
                if (rank == 0) {
                    std::lock_guard g(lm);
                    lat.push_back(ms * 100.0);
                }
            }
            std::vector<float> out(n);
            CK(hipMemcpy(out.data(), d, size_t(n) * 4, hipMemcpyDeviceToHost));
        };
        std::thread a(worker, 0), b(worker, 1);
        a.join();
        b.join();
        std::sort(lat.begin(), lat.end());
        printf("rccl SHM (P2P off):     median %.1f us  p10 %.1f  p90 %.1f  min %.1f  (per all-reduce, 10 back-to-back)\n",
               lat[lat.size() / 2], lat[lat.size() / 10], lat[lat.size() * 9 / 10], lat[0]);
        for (auto c : comm)
            ncclCommDestroy(c);
    };

    auto run_zerocopy = [&] {
        // fine-grained, coherent pinned host memory visible to both devices
        float4 *hbuf[2][2]; // [rank][parity] double buffer
        unsigned *hflag[2];
        for (int r = 0; r < 2; ++r) {
            for (int p = 0; p < 2; ++p)
                CK(hipHostMalloc(&hbuf[r][p], size_t(n) * 4, hipHostMallocCoherent | hipHostMallocPortable));
            CK(hipHostMalloc(&hflag[r], 256, hipHostMallocCoherent | hipHostMallocPortable));
            std::memset(hflag[r], 0, 256);
        }
        std::vector<double> lat;
        std::mutex lm;
        std::atomic<bool> ok{true};
        auto worker = [&](int rank) {
            CK(hipSetDevice(devs[rank]));
            hipStream_t s;
            CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
            float4 *d;
            CK(hipMalloc(&d, size_t(n) * 4));
            float4 *dmine[2], *dpeer[2];
            unsigned *dflag_mine, *dflag_peer, *dcounter;
            CK(hipMalloc(&dcounter, 4));
            CK(hipMemset(dcounter, 0, 4));
            for (int p = 0; p < 2; ++p) {
                CK(hipHostGetDevicePointer((void **)&dmine[p], hbuf[rank][p], 0));
                CK(hipHostGetDevicePointer((void **)&dpeer[p], hbuf[1 - rank][p], 0));
            }
            CK(hipHostGetDevicePointer((void **)&dflag_mine, hflag[rank], 0));
            CK(hipHostGetDevicePointer((void **)&dflag_peer, hflag[1 - rank], 0));
            std::vector<float> h(n, float(rank + 1));
            CK(hipMemcpy(d, h.data(), size_t(n) * 4, hipMemcpyHostToDevice));
            int n4 = n / 4, threads = 256, blocks = (n4 + threads - 1) / threads;
            unsigned seq = 0;
            auto allreduce = [&] {
                ++seq;
                zc_publish<<<blocks, threads, 0, s>>>(d, dmine[seq & 1], n4, dflag_mine, dcounter, seq);
                zc_reduce<<<blocks, threads, 0, s>>>(d, dpeer[seq & 1], n4, dflag_peer, seq);
            };
            for (int it = 0; it < 20; ++it)
                allreduce();
            CK(hipStreamSynchronize(s));
            // correctness check once
            {
                CK(hipMemcpy(d, h.data(), size_t(n) * 4, hipMemcpyHostToDevice));
                bar.wait();
                allreduce();
                CK(hipStreamSynchronize(s));
                std::vector<float> out(n);
                CK(hipMemcpy(out.data(), d, size_t(n) * 4, hipMemcpyDeviceToHost));
                for (float v : out)
                    if (v != 3.f)
                        ok = false;
                bar.wait();
            }
            hipEvent_t e0, e1;
            CK(hipEventCreate(&e0));
            CK(hipEventCreate(&e1));
            for (int it = 0; it < iters / 10; ++it) {
                bar.wait();
                CK(hipEventRecord(e0, s));
                for (int k = 0; k < 10; ++k)
                    allreduce();
                CK(hipEventRecord(e1, s));
                CK(hipEventSynchronize(e1));
                float ms;
                CK(hipEventElapsedTime(&ms, e0, e1));
                if (rank == 0) {
                    std::lock_guard g(lm);
                    lat.push_back(ms * 100.0);
                }
            }
        };
        std::thread a(worker, 0), b(worker, 1);
        a.join();
        b.join();
        std::sort(lat.begin(), lat.end());
        printf("zero-copy kernel-driven: median %.1f us  p10 %.1f  p90 %.1f  min %.1f  (per all-reduce, 10 back-to-back) correctness=%s\n",
               lat[lat.size() / 2], lat[lat.size() / 10], lat[lat.size() * 9 / 10], lat[0],
               ok ? "PASS" : "FAIL");
    };

    if (mode == "all" || mode == "host")
        run_host();
    if (mode == "all" || mode == "zerocopy")
        run_zerocopy();
    if (mode == "all" || mode == "rccl")
        run_rccl();
    return 0;
}
