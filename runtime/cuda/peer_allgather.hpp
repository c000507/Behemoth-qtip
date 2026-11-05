#pragma once
#include "kernels.hpp"
#include <algorithm>
#include <array>
#include <vector>
namespace behemoth::cuda_runtime {
// Two-rank, in-place AllGather using GPU stores or copy engines. Receive buffers contain
// [rank0 bytes][rank1 bytes]; send buffers may alias their own receive region.
class PeerAllGather {
    struct Device {
        int old;
        explicit Device(int current) {
            glm53::gpu::checked(cudaGetDevice(&old), "get collective device");
            glm53::gpu::checked(cudaSetDevice(current), "select collective device");
        }
        ~Device() { (void)cudaSetDevice(old); }
    };
    std::array<int, 2> devices;
    std::array<Stream, 2> streams;
    std::array<cudaEvent_t, 2> ready{}, done{};
    bool kernel_small;
    void validate(size_t bytes) {
        using glm53::gpu::checked;
        std::array<void *, 2> buffers{};
        auto release = [&] {
            for (int r = 0; r < 2; ++r) {
                Device device(devices[r]);
                if (buffers[r])
                    (void)cudaFree(buffers[r]);
            }
        };
        try {
            std::array<std::vector<uint8_t>, 2> patterns;
            std::array<const void *, 2> send;
            for (int r = 0; r < 2; ++r) {
                Device device(devices[r]);
                checked(cudaMalloc(&buffers[r], bytes * 2), "allocate peer validation buffer");
                checked(cudaMemsetAsync(buffers[r], 0xee, bytes * 2, streams[r]), "clear peer validation");
                patterns[r].resize(bytes);
                for (size_t i = 0; i < bytes; ++i)
                    patterns[r][i] = uint8_t(i * 17 + r * 83 + 19);
                auto *own = static_cast<uint8_t *>(buffers[r]) + r * bytes;
                checked(cudaMemcpyAsync(own, patterns[r].data(), bytes, cudaMemcpyHostToDevice, streams[r]),
                        "upload peer validation pattern");
                send[r] = own;
            }
            run(send, buffers, bytes);
            std::vector<uint8_t> result(bytes * 2);
            for (int r = 0; r < 2; ++r) {
                Device device(devices[r]);
                checked(
                    cudaMemcpyAsync(result.data(), buffers[r], bytes * 2, cudaMemcpyDeviceToHost, streams[r]),
                    "read peer validation result");
                checked(cudaStreamSynchronize(streams[r]), "complete peer validation");
                for (int sender = 0; sender < 2; ++sender)
                    if (!std::equal(patterns[sender].begin(), patterns[sender].end(),
                                    result.begin() + sender * bytes))
                        throw std::runtime_error("CUDA peer transfer failed data integrity on this host; "
                                                 "use BEHEMOTH_CUDA_TP_TRANSPORT=nccl");
            }
        } catch (...) {
            release();
            throw;
        }
        release();
    }
    void clear() {
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            if (ready[r])
                (void)cudaEventDestroy(ready[r]);
            if (done[r])
                (void)cudaEventDestroy(done[r]);
            ready[r] = done[r] = nullptr;
        }
    }

  public:
    PeerAllGather(std::array<int, 2> d, std::array<Stream, 2> s, bool small = true)
        : devices(d), streams(s), kernel_small(small) {
        using glm53::gpu::checked;
        try {
            for (int r = 0; r < 2; ++r) {
                Device device(devices[r]);
                int supported = 0;
                checked(cudaDeviceCanAccessPeer(&supported, devices[r], devices[1 - r]),
                        "query TP peer access");
                if (!supported)
                    throw std::runtime_error("Peer TP requires bidirectional CUDA peer access");
                auto status = cudaDeviceEnablePeerAccess(devices[1 - r], 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled)
                    (void)cudaGetLastError();
                else
                    checked(status, "enable TP peer access");
                checked(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming),
                        "create peer producer event");
                checked(cudaEventCreateWithFlags(&done[r], cudaEventDisableTiming),
                        "create peer completion event");
            }
            // Driver-reported peer access is insufficient on some hosted PCIe
            // topologies. Check both small and bulk transfers before inference.
            validate(4096);
            validate(2 * 1024 * 1024);
        } catch (...) {
            clear();
            throw;
        }
    }
    ~PeerAllGather() { clear(); }
    PeerAllGather(const PeerAllGather &) = delete;
    PeerAllGather &operator=(const PeerAllGather &) = delete;
    void run(const std::array<const void *, 2> &send, const std::array<void *, 2> &receive, size_t bytes) {
        using glm53::gpu::checked;
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            checked(cudaEventRecord(ready[r], streams[r]), "record peer producer");
        }
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            int other = 1 - r;
            checked(cudaStreamWaitEvent(streams[r], ready[other], 0), "wait peer producer");
            if (kernel_small && bytes <= 1024 * 1024) {
                auto *destination = static_cast<uint8_t *>(receive[other]) + r * bytes;
                peer_store(send[r], destination, bytes, streams[r]);
            } else {
                auto *destination = static_cast<uint8_t *>(receive[r]) + other * bytes;
                checked(
                    cudaMemcpyAsync(destination, send[other], bytes, cudaMemcpyDeviceToDevice, streams[r]),
                    "gather peer TP output");
            }
            checked(cudaEventRecord(done[r], streams[r]), "record peer copy completion");
        }
        // Both readers must finish before either rank can reuse its send region.
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            checked(cudaStreamWaitEvent(streams[r], done[1 - r], 0), "finish peer collective");
        }
    }
};
} // namespace behemoth::cuda_runtime
