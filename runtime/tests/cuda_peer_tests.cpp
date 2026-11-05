#include "peer_allgather.hpp"
#include <iostream>
#include <memory>
namespace cg = glm53::gpu;
namespace cu = behemoth::cuda_runtime;
namespace {
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
void test(size_t bytes) {
    std::array<std::unique_ptr<Rank>, 2> ranks;
    for (int r = 0; r < 2; ++r)
        ranks[r] = std::make_unique<Rank>(r, bytes);
    cu::PeerAllGather collective({0, 1}, {ranks[0]->stream, ranks[1]->stream});
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
                        throw std::runtime_error("Peer AllGather changed or mixed rank data");
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
    std::cout << "PASS peer AllGather and graph replay, " << bytes << " bytes per rank\n";
}
} // namespace
int main() {
    try {
        int count = 0;
        auto status = cg::device_count(&count);
        if (status == cg::no_device || status == cg::insufficient_driver ||
            (status == cg::success && count < 2))
            return 77;
        cg::checked(status, "peer test GPU discovery");
        for (int r = 0; r < 2; ++r) {
            int supported = 0;
            cg::checked(cudaDeviceCanAccessPeer(&supported, r, 1 - r), "peer test access query");
            if (!supported)
                return 77;
        }
        test(56 * 1024);
        test(8 * 1024 * 1024);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
