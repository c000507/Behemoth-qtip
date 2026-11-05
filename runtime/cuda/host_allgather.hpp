#pragma once
#include "kernels.hpp"
#include <algorithm>
#include <array>
namespace behemoth::cuda_runtime {
// In-place, two-rank AllGather. Pipeline D2H and H2D through two portable pinned slots per sender. Copy
// engines can overlap the two PCIe directions without requiring GPU peer access.
class HostAllGather {
    struct Device {
        int old;
        explicit Device(int current) {
            glm53::gpu::checked(cudaGetDevice(&old), "get host collective device");
            glm53::gpu::checked(cudaSetDevice(current), "select host collective device");
        }
        ~Device() { (void)cudaSetDevice(old); }
    };
    std::array<int, 2> devices;
    std::array<Stream, 2> streams, download{};
    std::array<cudaEvent_t, 2> producer{}, downloaded{}, done{};
    std::array<std::array<cudaEvent_t, 2>, 2> staged{}, consumed{};
    uint8_t *host = nullptr;
    size_t chunk;
    uint8_t *slot(int rank, int index) { return host + (rank * 2 + index) * chunk; }
    void clear() {
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            if (download[r]) {
                (void)cudaStreamSynchronize(download[r]);
                (void)cudaStreamDestroy(download[r]);
            }
            for (auto event : {producer[r], downloaded[r], done[r], staged[r][0], staged[r][1]})
                if (event)
                    (void)cudaEventDestroy(event);
            // A sender's slots are consumed on the other device.
            for (auto event : consumed[1 - r])
                if (event)
                    (void)cudaEventDestroy(event);
        }
        if (host)
            (void)cudaFreeHost(host);
    }

  public:
    HostAllGather(std::array<int, 2> d, std::array<Stream, 2> s, size_t bytes = 16 * 1024 * 1024)
        : devices(d), streams(s), chunk(bytes) {
        using glm53::gpu::checked;
        if (!chunk || chunk > (size_t(1) << 30))
            throw std::runtime_error("Invalid host AllGather chunk size");
        try {
            checked(cudaHostAlloc(reinterpret_cast<void **>(&host), chunk * 4, cudaHostAllocPortable),
                    "allocate host AllGather slots");
            for (int r = 0; r < 2; ++r) {
                Device device(devices[r]);
                checked(cudaStreamCreateWithFlags(&download[r], cudaStreamNonBlocking),
                        "create host download stream");
                auto create = [&](cudaEvent_t &event) {
                    checked(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                            "create host collective event");
                };
                create(producer[r]);
                create(downloaded[r]);
                create(done[r]);
                for (int i = 0; i < 2; ++i) {
                    create(staged[r][i]);
                    create(consumed[1 - r][i]);
                }
            }
        } catch (...) {
            clear();
            throw;
        }
    }
    ~HostAllGather() { clear(); }
    HostAllGather(const HostAllGather &) = delete;
    HostAllGather &operator=(const HostAllGather &) = delete;
    void run(const std::array<const void *, 2> &send, const std::array<void *, 2> &receive, size_t bytes) {
        using glm53::gpu::checked;
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            checked(cudaEventRecord(producer[r], streams[r]), "record host collective producer");
            checked(cudaStreamWaitEvent(download[r], producer[r], 0), "wait host collective producer");
        }
        for (size_t offset = 0, index = 0; offset < bytes; offset += chunk, ++index) {
            int current = index % 2;
            size_t count = std::min(chunk, bytes - offset);
            for (int r = 0; r < 2; ++r) {
                Device device(devices[r]);
                if (index >= 2)
                    checked(cudaStreamWaitEvent(download[r], consumed[r][current], 0),
                            "wait host staging slot reuse");
                checked(cudaMemcpyAsync(slot(r, current), static_cast<const uint8_t *>(send[r]) + offset,
                                        count, cudaMemcpyDeviceToHost, download[r]),
                        "stage host AllGather sender");
                checked(cudaEventRecord(staged[r][current], download[r]), "record staged host chunk");
            }
            for (int r = 0; r < 2; ++r) {
                Device device(devices[r]);
                int other = 1 - r;
                checked(cudaStreamWaitEvent(streams[r], staged[other][current], 0), "wait remote host chunk");
                checked(cudaMemcpyAsync(static_cast<uint8_t *>(receive[r]) + other * bytes + offset,
                                        slot(other, current), count, cudaMemcpyHostToDevice, streams[r]),
                        "receive host AllGather chunk");
                checked(cudaEventRecord(consumed[other][current], streams[r]), "release host staging slot");
            }
        }
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            checked(cudaEventRecord(downloaded[r], download[r]), "record host download completion");
            checked(cudaStreamWaitEvent(streams[r], downloaded[r], 0), "join host download stream");
            checked(cudaEventRecord(done[r], streams[r]), "record host collective completion");
        }
        // Every slot and GPU source must be free before either next producer.
        // This also joins all side streams into a captured graph's origin.
        for (int r = 0; r < 2; ++r) {
            Device device(devices[r]);
            checked(cudaStreamWaitEvent(streams[r], done[1 - r], 0), "finish host collective");
        }
    }
};
} // namespace behemoth::cuda_runtime
