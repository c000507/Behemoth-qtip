#include "peer_allgather.hpp"
#include <iostream>
#include <memory>
#if defined(BEHEMOTH_CUDA_FAST)
#include <cstdlib>
#include <nccl.h>
#endif
namespace cg = glm53::gpu;
namespace cu = behemoth::cuda_runtime;
namespace {
class Collective {
    std::unique_ptr<cu::PeerAllGather> peer;
    std::array<cg::Stream, 2> streams;
#if defined(BEHEMOTH_CUDA_FAST)
    std::array<ncclComm_t, 2> comms{};
    static void check(ncclResult_t result) {
        if (result != ncclSuccess)
            throw std::runtime_error(ncclGetErrorString(result));
    }
#endif
  public:
    Collective(std::array<int, 2> devices, std::array<cg::Stream, 2> s, bool small, bool nccl) : streams(s) {
#if defined(BEHEMOTH_CUDA_FAST)
        if (nccl) {
            setenv("NCCL_P2P_DISABLE", "1", 0);
            setenv("NCCL_MAX_NCHANNELS", "2", 0);
            check(ncclCommInitAll(comms.data(), 2, devices.data()));
            return;
        }
#else
        if (nccl)
            throw std::runtime_error("NCCL requires BEHEMOTH_CUDA_FAST");
#endif
        peer = std::make_unique<cu::PeerAllGather>(devices, streams, small);
    }
    ~Collective() {
#if defined(BEHEMOTH_CUDA_FAST)
        for (int r = 0; r < 2; ++r)
            if (comms[r]) {
                (void)cudaSetDevice(r);
                (void)ncclCommDestroy(comms[r]);
            }
#endif
    }
    void run(const std::array<const void *, 2> &send, const std::array<void *, 2> &receive, size_t bytes) {
        if (peer) {
            peer->run(send, receive, bytes);
            return;
        }
#if defined(BEHEMOTH_CUDA_FAST)
        check(ncclGroupStart());
        for (int r = 0; r < 2; ++r) {
            cg::checked(cudaSetDevice(r), "select NCCL rank");
            check(ncclAllGather(send[r], receive[r], bytes, ncclUint8, comms[r], streams[r]));
        }
        check(ncclGroupEnd());
#endif
    }
};
struct Rank {
    int device;
    cg::Stream stream{};
    cg::Buffer<uint8_t> buffer;
    cg::Pinned<uint8_t> source, result;
    Rank(int d, size_t bytes) : device(d) {
        cg::checked(cudaSetDevice(device), "select test rank");
        cg::checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create test stream");
        buffer.ensure(bytes * 2, stream);
        source.ensure(bytes);
        result.ensure(bytes * 2);
    }
    ~Rank() {
        (void)cudaSetDevice(device);
        (void)cudaStreamSynchronize(stream);
        buffer.clear();
        (void)cudaStreamDestroy(stream);
    }
};
void test(size_t bytes, bool kernel_small = true, bool nccl_mode = false) {
    std::cout << "Testing peer AllGather bytes=" << bytes << " kernel=" << kernel_small
              << " nccl=" << nccl_mode << std::endl;
    std::array<std::unique_ptr<Rank>, 2> ranks;
    for (int r = 0; r < 2; ++r)
        ranks[r] = std::make_unique<Rank>(r, bytes);
    Collective collective({0, 1}, {ranks[0]->stream, ranks[1]->stream}, kernel_small, nccl_mode);
    auto work = [&] {
        // Repeated buffer reuse exercises both producer and consumer dependencies.
        for (int repeat = 0; repeat < 3; ++repeat) {
            std::array<const void *, 2> send;
            std::array<void *, 2> receive;
            for (int r = 0; r < 2; ++r) {
                auto &rank = *ranks[r];
                cg::checked(cudaSetDevice(r), "select producer");
                cg::checked(cudaMemsetAsync(rank.buffer, 0xee, bytes * 2, rank.stream),
                            "overwrite collective buffer");
                send[r] = rank.buffer.p + r * bytes;
                receive[r] = rank.buffer.p;
                cg::checked(cg::h2d(rank.buffer.p + r * bytes, rank.source.p, bytes, rank.stream),
                            "upload collective pattern");
            }
            collective.run(send, receive, bytes);
        }
    };
    auto pattern = [&](int epoch) {
        for (int r = 0; r < 2; ++r)
            for (size_t i = 0; i < bytes; ++i)
                ranks[r]->source.p[i] = uint8_t(i * 17 + r * 83 + epoch * 19);
    };
    auto verify = [&] {
        for (int r = 0; r < 2; ++r) {
            auto &rank = *ranks[r];
            cg::checked(cudaSetDevice(r), "select reader");
            cg::checked(cg::d2h(rank.result.p, rank.buffer, bytes * 2, rank.stream),
                        "read collective result");
            cg::checked(cg::sync(rank.stream), "finish collective read");
            for (int sender = 0; sender < 2; ++sender)
                for (size_t i = 0; i < bytes; ++i)
                    if (rank.result.p[sender * bytes + i] != ranks[sender]->source.p[i])
                        throw std::runtime_error(
                            "Peer AllGather changed or mixed rank data: receiver=" + std::to_string(r) +
                            " sender=" + std::to_string(sender) + " byte=" + std::to_string(i) +
                            " actual=" + std::to_string(rank.result.p[sender * bytes + i]) +
                            " expected=" + std::to_string(ranks[sender]->source.p[i]));
        }
    };
    pattern(0);
    work();
    verify();
    cudaEvent_t fork{}, join{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cg::checked(cudaSetDevice(0), "select graph origin");
    cg::checked(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "create graph fork");
    cg::checked(cudaSetDevice(1), "select graph peer");
    cg::checked(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "create graph join");
    cg::checked(cudaSetDevice(0), "select capture origin");
    cg::checked(cudaStreamBeginCapture(ranks[0]->stream, cudaStreamCaptureModeGlobal), "begin peer graph");
    cg::checked(cudaEventRecord(fork, ranks[0]->stream), "fork peer graph");
    cg::checked(cudaSetDevice(1), "select capture peer");
    cg::checked(cudaStreamWaitEvent(ranks[1]->stream, fork, 0), "join peer capture");
    work();
    cg::checked(cudaSetDevice(1), "select capture completion");
    cg::checked(cudaEventRecord(join, ranks[1]->stream), "complete peer capture");
    cg::checked(cudaSetDevice(0), "select capture end");
    cg::checked(cudaStreamWaitEvent(ranks[0]->stream, join, 0), "close peer capture");
    cg::checked(cudaStreamEndCapture(ranks[0]->stream, &graph), "end peer graph");
    cg::checked(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate peer graph");
    for (int epoch = 1; epoch <= 3; ++epoch) {
        pattern(epoch);
        cg::checked(cudaSetDevice(0), "select graph replay");
        cg::checked(cudaGraphLaunch(executable, ranks[0]->stream), "replay peer graph");
        cg::checked(cg::sync(ranks[0]->stream), "finish peer graph");
        verify();
    }
    cg::checked(cudaSetDevice(0), "select graph cleanup");
    (void)cudaGraphExecDestroy(executable);
    (void)cudaGraphDestroy(graph);
    (void)cudaEventDestroy(fork);
    cg::checked(cudaSetDevice(1), "select peer event cleanup");
    (void)cudaEventDestroy(join);
    std::cout << "PASS peer AllGather and graph replay, " << bytes
              << " bytes per rank, small kernel=" << kernel_small << '\n';
}
} // namespace
int main(int argc, char **argv) {
    try {
        int count = 0;
        auto status = cg::device_count(&count);
        if (status == cg::no_device || status == cg::insufficient_driver ||
            (status == cg::success && count < 2))
            return 77;
        cg::checked(status, "peer test GPU discovery");
        if (argc == 2 && std::string(argv[1]) == "--nccl") {
            test(56 * 1024, false, true);
            test(224 * 1024, false, true);
            test(8 * 1024 * 1024, false, true);
            return 0;
        }
        for (int r = 0; r < 2; ++r) {
            int supported = 0;
            cg::checked(cudaDeviceCanAccessPeer(&supported, r, 1 - r), "peer test access query");
            if (!supported)
                return 77;
        }
        if (argc == 2 && std::string(argv[1]) == "--dma") {
            test(56 * 1024, false);
            test(8 * 1024 * 1024, false);
            return 0;
        }
        if (argc != 1)
            throw std::runtime_error("Usage: behemoth_cuda_peer_tests [--dma|--nccl]");
        test(56 * 1024);
        test(224 * 1024);
        test(4099);
        test(56 * 1024, false);
        test(8 * 1024 * 1024);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
