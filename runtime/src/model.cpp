#include "behemoth/model.hpp"
#include "behemoth/gpu.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/prefill.hpp"
#include "behemoth/worker.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <rccl/rccl.h>
#include <sstream>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace behemoth {
namespace {
size_t sampled_capacity(const RuntimeOptions &o) {
    return std::min(o.token_budget, o.sampled_rows ? o.sampled_rows : max_sampled_rows);
}
std::atomic<int> &qtip_overlap_setting() {
    static std::atomic<int> v{[] {
        const char *e = std::getenv("BEHEMOTH_QTIP_OVERLAP_ROWS");
        return e ? std::atoi(e) : -1;
    }()};
    return v;
}
void collective_check(ncclResult_t s, const char *op) {
    if (s != ncclSuccess)
        throw std::runtime_error(std::string(op) + ": " + ncclGetErrorString(s));
}
#define RCCL_CHECK(x) collective_check((x), #x)
void topology(int device) {
    HIP_CHECK(hipSetDevice(device));
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, device));
    char bus[64]{};
    HIP_CHECK(hipDeviceGetPCIBusId(bus, sizeof(bus), device));
    auto root = std::filesystem::path("/sys/bus/pci/devices") / bus;
    auto line = [&](const char *file) {
        std::ifstream f(root / file);
        std::string value;
        std::getline(f, value);
        return value.empty() ? "unavailable" : value;
    };
    uint64_t largest = 0;
    std::ifstream resources(root / "resource");
    std::string entry;
    while (std::getline(resources, entry)) {
        std::istringstream s(entry);
        uint64_t a = 0, b = 0, flags = 0;
        s >> std::hex >> a >> b >> flags;
        if (a && b >= a && (flags & 0x200))
            largest = std::max(largest, b - a + 1);
    }
    std::ifstream cmd("/proc/cmdline");
    std::string flag;
    bool iommu_pt = false;
    while (cmd >> flag)
        iommu_pt |= flag == "iommu=pt" || flag == "iommu.passthrough=1";
    std::cerr << "GPU " << device << " " << p.name << " arch=" << p.gcnArchName
              << " wave=" << p.warpSize << " pci=" << bus << " link=" << line("current_link_speed")
              << " width=" << line("current_link_width") << " largest_memory_BAR=" << largest
              << " BAR_covers_VRAM=" << (largest >= p.totalGlobalMem) << " iommu_pt=" << iommu_pt
              << " P2P=disabled; BAR/IOMMU are diagnostic only\n";
}
// Stream-ordered copy-engine all-gather between the two ranks through pinned host memory
// (PREFILL_SCHED_REPORT.md). gather(): the rank's stream copies its half to its host buffer in
// pieces and records one event per piece; a second (copy) stream of the rank waits for the
// peer's piece events (cross-device hipStreamWaitEvent: a command-processor barrier, not a
// kernel) and copies the peer's pieces in, so outgoing and incoming PCIe traffic overlap (full
// duplex); the rank's stream then waits for the copy stream. The copy engines reach ~13 (GPU 0)
// / ~27 GB/s (GPU 1) here; kernel-issued PCIe stores (the publish kernels) only ~7 GB/s, and
// hipStreamWaitValue32 runs as a spinning kernel that slows concurrent GEMMs.
// Host handshake: a rank issues its waits for message e only after both ranks recorded message
// e's events (one host arrival per message per rank), so a wait never sees a stale record.
// Buffer/event reuse: message e uses parity e & 1. The peer's H2D of message e completes before
// its stream issues the copy-out of message e + 1 (its stream waits for its copy stream), and
// this rank's copy-in of message e + 1 waits for those events before its stream writes
// buffer e + 2; events of message e + 2 are recorded after this rank passed the handshake of
// e + 1, which the peer reaches only after issuing its waits for message e. Each channel is
// used by one stream per rank, with the same message order on both ranks.
class DmaChannel {
    static constexpr int max_pieces = 16;
    std::array<std::array<float *, 2>, 2> host_{};          // [owner][parity]
    std::array<std::array<std::array<hipEvent_t, max_pieces>, 2>, 2> events_{}; // [rank][parity]
    std::array<int, 2> event_device_{-1, -1};
    std::array<unsigned, 2> messages_{};
    std::atomic<uint64_t> arrivals_{0};
    const std::atomic<bool> *aborted_ = nullptr;
    size_t capacity_ = 0;
    void release() noexcept {
        for (auto &r : host_)
            for (auto p : r)
                if (p)
                    (void)hipHostFree(p);
        for (size_t r = 0; r < 2; ++r) {
            if (event_device_[r] < 0)
                continue;
            int current = 0;
            (void)hipGetDevice(&current);
            (void)hipSetDevice(event_device_[r]);
            for (auto &parity : events_[r])
                for (auto e : parity)
                    if (e)
                        (void)hipEventDestroy(e);
            (void)hipSetDevice(current);
        }
    }

  public:
    DmaChannel(size_t capacity, const std::atomic<bool> *aborted) : aborted_(aborted), capacity_(capacity) {
        try {
            for (size_t owner = 0; owner < 2; ++owner)
                for (auto &p : host_[owner])
                    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&p), capacity * sizeof(float),
                                            hipHostMallocPortable));
        } catch (...) {
            release();
            throw;
        }
    }
    ~DmaChannel() {
        release();
    }
    DmaChannel(const DmaChannel &) = delete;
    DmaChannel &operator=(const DmaChannel &) = delete;
    size_t capacity() const {
        return capacity_;
    }
    // local: this rank's n floats; all[(1 - rank) * n ..] receives the peer's n floats (the
    // caller's half at all[rank * n ..] is untouched). Runs on the rank's thread and device.
    void gather(int rank, const float *local, float *all, size_t n, hipStream_t stream,
                hipStream_t copy_stream, hipEvent_t start, hipEvent_t done) {
        if (n > capacity_)
            throw std::invalid_argument("DMA gather exceeds the channel buffers");
        const size_t r = size_t(rank), peer = size_t(1 - rank);
        if (event_device_[r] < 0) {
            HIP_CHECK(hipGetDevice(&event_device_[r]));
            for (auto &parity : events_[r])
                for (auto &e : parity)
                    HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
        }
        static const size_t piece_limit = [] {
            const char *v = std::getenv("BEHEMOTH_DMA_PIECES");
            const int x = v ? std::atoi(v) : 4;
            return size_t(std::clamp(x, 1, max_pieces));
        }();
        // pieces of at least 1 MiB; both ranks derive the same count from n
        const size_t pieces = std::clamp<size_t>(n * sizeof(float) >> 20, 1, piece_limit);
        const size_t per = (n + pieces - 1) / pieces;
        const unsigned message = messages_[r]++, parity = message & 1U;
        HIP_CHECK(hipEventRecord(start, stream));
        HIP_CHECK(hipStreamWaitEvent(copy_stream, start, 0));
        for (size_t p = 0; p < pieces; ++p) {
            const size_t a = std::min(n, p * per), b = std::min(n, a + per);
            if (b > a)
                HIP_CHECK(hipMemcpyAsync(host_[r][parity] + a, local + a, (b - a) * sizeof(float),
                                         hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipEventRecord(events_[r][parity][p], stream));
        }
        // handshake: both ranks recorded this message's events
        const uint64_t target = 2 * (uint64_t(message) + 1);
        arrivals_.fetch_add(1, std::memory_order_acq_rel);
        while (arrivals_.load(std::memory_order_acquire) < target) {
            if (aborted_ && aborted_->load(std::memory_order_relaxed))
                throw std::runtime_error("Host transport aborted");
            std::this_thread::yield();
        }
        for (size_t p = 0; p < pieces; ++p) {
            const size_t a = std::min(n, p * per), b = std::min(n, a + per);
            HIP_CHECK(hipStreamWaitEvent(copy_stream, events_[peer][parity][p], 0));
            if (b > a)
                HIP_CHECK(hipMemcpyAsync(all + peer * n + a, host_[peer][parity] + a,
                                         (b - a) * sizeof(float), hipMemcpyHostToDevice,
                                         copy_stream));
        }
        HIP_CHECK(hipEventRecord(done, copy_stream));
        HIP_CHECK(hipStreamWaitEvent(stream, done, 0));
    }
};
class HostTransport {
    std::array<float *, 2> host_{};
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
    unsigned arrived_ = 0, generation_ = 0;
    bool aborted_ = false;
    std::atomic<bool> abort_flag_{false}; // for the DMA channels' host handshake
    // Zero-copy path: per-rank double-buffered fine-grained host buffers plus a
    // sequence flag, exchanged by stream-ordered kernels (see transport.hip).
    // Decode-sized messages avoid the three host rendezvous and two stream
    // synchronisations of the staged path; large prefill messages keep it.
    bool zero_copy_ = false;
    size_t zero_copy_limit_ = 0;
    std::array<std::array<float *, 2>, 2> zc_host_{};
    std::array<unsigned *, 2> zc_flag_{};
    std::array<unsigned *, 2> zc_counter_{}; // device memory, owned by each rank
    std::array<unsigned, 2> zc_seq_{};
    // Buffer parity alternates per exchange (both paths); flag values advance per published
    // piece (the DMA path publishes a message in pieces).
    std::array<unsigned, 2> zc_exchange_{};
    struct Slot {
        unsigned parity, seq; // seq: flag value of the (last) piece
    };
    Slot next_slot(int rank, unsigned pieces = 1) {
        const unsigned parity = zc_exchange_[size_t(rank)]++ & 1U;
        zc_seq_[size_t(rank)] += pieces;
        return {parity, zc_seq_[size_t(rank)]};
    }
    struct DevicePointers {
        bool valid = false;
        std::array<std::array<float *, 2>, 2> host{}; // [owner][parity]
        std::array<unsigned *, 2> flag{};
    };
    std::array<DevicePointers, 2> zc_dev_{}; // indexed by the accessing rank
    const DevicePointers &device_pointers(int rank) {
        auto &d = zc_dev_[size_t(rank)];
        if (!d.valid) {
            for (int owner = 0; owner < 2; ++owner) {
                for (int parity = 0; parity < 2; ++parity)
                    HIP_CHECK(hipHostGetDevicePointer(
                        reinterpret_cast<void **>(&d.host[size_t(owner)][size_t(parity)]),
                        zc_host_[size_t(owner)][size_t(parity)], 0));
                HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&d.flag[size_t(owner)]),
                                                  zc_flag_[size_t(owner)], 0));
            }
            d.valid = true;
        }
        return d;
    }
    void barrier() {
        std::unique_lock lock(mutex_);
        if (aborted_)
            throw std::runtime_error("Host transport aborted");
        unsigned generation = generation_;
        if (++arrived_ == 2) {
            arrived_ = 0;
            ++generation_;
            cv_.notify_all();
        } else
            cv_.wait(lock, [&] { return aborted_ || generation_ != generation; });
        if (aborted_)
            throw std::runtime_error("Host transport aborted");
    }
    void release_zero_copy() noexcept {
        for (auto &r : zc_host_)
            for (auto &p : r)
                if (p)
                    (void)hipHostFree(p);
        for (auto p : zc_flag_)
            if (p)
                (void)hipHostFree(p);
        for (auto p : zc_counter_)
            if (p)
                (void)hipFree(p);
    }

  public:
    // zero_copy_limit: largest message (floats) exchanged through host memory
    // by kernels; 0 disables the path. devices: the two ranks' HIP devices.
    explicit HostTransport(size_t capacity, size_t zero_copy_limit = 0,
                           std::array<int, 2> devices = {0, 1})
        : capacity_(capacity) {
        try {
            for (auto &p : host_)
                HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&p), capacity * sizeof(float),
                                        hipHostMallocPortable));
            if (zero_copy_limit) {
                zero_copy_ = true;
                zero_copy_limit_ = std::min(zero_copy_limit, capacity);
                int current = 0;
                HIP_CHECK(hipGetDevice(&current));
                for (int rank = 0; rank < 2; ++rank) {
                    for (auto &p : zc_host_[size_t(rank)])
                        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&p),
                                                zero_copy_limit_ * sizeof(float),
                                                hipHostMallocCoherent | hipHostMallocPortable));
                    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&zc_flag_[size_t(rank)]), 256,
                                            hipHostMallocCoherent | hipHostMallocPortable));
                    std::memset(zc_flag_[size_t(rank)], 0, 256);
                    HIP_CHECK(hipSetDevice(devices[size_t(rank)]));
                    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&zc_counter_[size_t(rank)]), 4));
                    HIP_CHECK(hipMemset(zc_counter_[size_t(rank)], 0, 4));
                }
                HIP_CHECK(hipSetDevice(current));
            }
        } catch (...) {
            for (auto p : host_)
                if (p)
                    (void)hipHostFree(p);
            release_zero_copy();
            throw;
        }
    }
    ~HostTransport() {
        for (auto p : host_)
            if (p)
                (void)hipHostFree(p);
        release_zero_copy();
    }
    bool zero_copy() const {
        return zero_copy_;
    }
    // A prepared zero-copy exchange slot (PERF_ROUND3 item 4): the caller may
    // let a projection kernel publish its partials straight into mine (the
    // device pointer of a pinned host buffer) and then consume peer+mine in
    // one fused reduce+residual kernel.
    struct Exchange {
        float *mine{}, *peer{};       // device pointers of pinned host buffers
        unsigned *flag{}, *peer_flag{}, *counter{};
        unsigned seq{};
        bool valid = false;
    };
    bool zero_copy_eligible(size_t n) const {
        return zero_copy_ && n <= zero_copy_limit_ && n % 4 == 0;
    }
    Exchange begin_exchange(int rank, size_t n) {
        if (!zero_copy_eligible(n))
            return {};
        const auto &d = device_pointers(rank);
        const auto [parity, seq] = next_slot(rank);
        Exchange e;
        e.mine = d.host[size_t(rank)][parity];
        e.peer = d.host[size_t(1 - rank)][parity];
        e.flag = d.flag[size_t(rank)];
        e.peer_flag = d.flag[size_t(1 - rank)];
        e.counter = zc_counter_[size_t(rank)];
        e.seq = seq;
        e.valid = true;
        return e;
    }
    // Device partials: classic publish then one fused reduce+residual kernel.
    void reduce_residual_dev(int rank, const Exchange &e, float *values, BFloat16 *state,
                             const float *row_scale, size_t n, int width, hipStream_t stream) {
        zero_copy_publish(values, e.mine, n, e.flag, e.counter, e.seq, stream);
        zero_copy_reduce_residual_dev(values, e.peer, state, row_scale, n, width, e.peer_flag,
                                      e.seq, stream);
    }
    // Full fallback (staged transport or ineligible size): reduce, then the
    // separate scaled-residual pass with identical rounding.
    void reduce_residual(int rank, float *values, BFloat16 *state, const float *row_scale, size_t n,
                         int width, hipStream_t stream) {
        if (zero_copy_eligible(n)) {
            const auto e = begin_exchange(rank, n);
            reduce_residual_dev(rank, e, values, state, row_scale, n, width, stream);
            return;
        }
        reduce(rank, values, n, stream);
        residual_add_scaled(state, values, row_scale, int(n / size_t(width)), width, stream);
    }
    void abort() {
        std::lock_guard lock(mutex_);
        aborted_ = true;
        abort_flag_ = true;
        cv_.notify_all();
    }
    void reduce(int rank, float *values, size_t n, hipStream_t stream) {
        if (n > capacity_)
            throw std::invalid_argument("Host transport capacity exceeded");
        if (zero_copy_ && n <= zero_copy_limit_ && n % 4 == 0) {
            const auto &d = device_pointers(rank);
            const auto [parity, seq] = next_slot(rank);
            zero_copy_publish(values, d.host[size_t(rank)][parity], n, d.flag[size_t(rank)],
                              zc_counter_[size_t(rank)], seq, stream);
            zero_copy_reduce(values, d.host[size_t(1 - rank)][parity], n,
                             d.flag[size_t(1 - rank)], seq, stream);
            return;
        }
        HIP_CHECK(hipMemcpyAsync(host_[size_t(rank)], values, n * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        barrier();
        if (rank == 0)
            for (size_t i = 0; i < n; ++i)
                host_[0][i] += host_[1][i];
        barrier();
        HIP_CHECK(
            hipMemcpyAsync(values, host_[0], n * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        barrier();
    }
    // Copy-engine channels for prefill-sized all-gathers: channel i for micro-batch i (0: the
    // rank stream; see DmaChannel).
    std::array<std::unique_ptr<DmaChannel>, 4> dma_;
    // rows: token budget; channel 0 carries whole steps, channels 1.. the later micro-batches
    // (at most half of a step each). Largest message: the bf16 gate/up words, rows x 14336.
    void enable_dma(size_t rows, int channels) {
        for (int c = 0; c < channels && c < int(dma_.size()); ++c)
            dma_[size_t(c)] =
                std::make_unique<DmaChannel>((c ? (rows + 1) / 2 : rows) * 14336, &abort_flag_);
    }
    bool dma_eligible(size_t n) const {
        return dma_[0] && n <= dma_[0]->capacity();
    }
    void dma_gather(int channel, int rank, const float *local, float *all, size_t n,
                    hipStream_t stream, hipStream_t copy_stream, hipEvent_t start, hipEvent_t done) {
        dma_.at(size_t(channel))->gather(rank, local, all, n, stream, copy_stream, start, done);
    }
    void gather(int rank, const float *local, float *all, size_t n, hipStream_t stream) {
        if (n > capacity_)
            throw std::invalid_argument("Host transport capacity exceeded");
        if (zero_copy_ && n <= zero_copy_limit_ && n % 4 == 0) {
            const auto &d = device_pointers(rank);
            const auto [parity, seq] = next_slot(rank);
            zero_copy_publish(local, d.host[size_t(rank)][parity], n, d.flag[size_t(rank)],
                              zc_counter_[size_t(rank)], seq, stream);
            zero_copy_gather(local, d.host[size_t(1 - rank)][parity], all, n, rank,
                             d.flag[size_t(1 - rank)], seq, stream);
            return;
        }
        HIP_CHECK(hipMemcpyAsync(host_[size_t(rank)], local, n * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        barrier();
        for (size_t r = 0; r < 2; ++r)
            HIP_CHECK(hipMemcpyAsync(all + r * n, host_[r], n * sizeof(float),
                                     hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        barrier();
    }
};
// Reuse diagnostic events; disabled profiling introduces no stream synchronization.
class GpuTimes {
    struct Entry {
        hipEvent_t begin{}, end{};
        std::string name;
        Entry() {
            HIP_CHECK(hipEventCreate(&begin));
            try {
                HIP_CHECK(hipEventCreate(&end));
            } catch (...) {
                (void)hipEventDestroy(begin);
                throw;
            }
        }
        ~Entry() {
            (void)hipEventDestroy(begin);
            (void)hipEventDestroy(end);
        }
    };
    std::vector<std::unique_ptr<Entry>> entries_;
    size_t cursor_ = 0;

  public:
    bool enabled = false;
    void reset() {
        cursor_ = 0;
    }
    template <class F> void measure(const char *name, hipStream_t stream, F f) {
        if (!enabled) {
            f();
            return;
        }
        if (cursor_ == entries_.size())
            entries_.push_back(std::make_unique<Entry>());
        auto &e = *entries_[cursor_++];
        e.name = name;
        HIP_CHECK(hipEventRecord(e.begin, stream));
        f();
        HIP_CHECK(hipEventRecord(e.end, stream));
    }
    nlohmann::json summary() const {
        nlohmann::json result = nlohmann::json::object();
        for (size_t i = 0; i < cursor_; ++i) {
            const auto &e = *entries_[i];
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, e.begin, e.end));
            result[e.name] = result.value(e.name, 0.0) + ms;
        }
        return result;
    }
};
// QTIP weights of one layer on one rank (QTIP_RUNTIME_REPORT.md, "Tensor parallelism"): the
// rotated weights are split by output rows (q/k/v, gate/up stacked per rank), every collective is
// an all-gather of rotated outputs (or of the head-sharded attention output), and the Hadamard
// transforms run on full vectors on both ranks.
struct QtipLayer {
    qtip::GpuLinear qkv, o, gate_up, down;
    qtip::GpuCodebook lut;
    DeviceBuffer<float> su_qkv, su_o, su_gate_up, su_down;
    DeviceBuffer<float> sv_q, sv_k, sv_v, sv_o, sv_gate, sv_up, sv_down;
    size_t bytes() const {
        return qkv.bytes() + o.bytes() + gate_up.bytes() + down.bytes();
    }
};
struct Layer {
    GpuProjection q, k, v, o, gate, up, down;
    QtipLayer qtip;
    DeviceBuffer<BFloat16> input_norm, post_norm;
    DeviceBuffer<uint8_t> cache_k, cache_v;
    DeviceBuffer<float> k_scale, v_scale;
};
struct Shard {
    int device, rank;
    hipStream_t stream{};
    std::unique_ptr<ProjectionRunner> runner;
    std::vector<Layer> layers;
    DeviceBuffer<BFloat16> embeddings, final_norm, state, norm, tmp, q, k, v, attention, gate, up,
        selected;
    GpuProjection head;
    ProjectionWorkspace projection;
    DeviceBuffer<uint8_t> query;
    DeviceBuffer<float> query_scale, raw, gathered;
    DeviceBuffer<int> sample_rows;
    DeviceBuffer<std::byte> attention_workspace;
    GpuMetadata metadata;
    GpuTimes timings;
    // QTIP path: rotated bf16 inputs, matmul slices, rank-blocked exchange buffer, MLP product.
    bool qtip = false;
    int bf16_prefill_rows = 0;
    DeviceBuffer<BFloat16> qtip_u, qtip_h;
    DeviceBuffer<float> qtip_partial, qtip_all;
    // Fused decode-sized residual + input transform (qtip::ResidualSync): rank stream only.
    DeviceBuffer<unsigned long long> qtip_sync_counters;
    qtip::ResidualSync qtip_sync;
    // Dense prefill path (PREFILL_SCHED_REPORT.md): one decoded-weight scratch shared by every
    // linear (pieces of <= qtip::dense_scratch_elements), hipBLASLt plans per shape.
    std::unique_ptr<qtip::DenseMatmul> qtip_dense;
    DeviceBuffer<BFloat16> qtip_dense_scratch;
    hipStream_t copy_stream{};
    hipEvent_t copy_start{}, copy_done{};
    // Second micro-batch stream of large prefill steps (qtip_forward) and its events.
    // Micro-batch streams 1..max_micro_batches-1 of large prefill steps (qtip_forward; micro-
    // batch 0 runs on the rank stream) and their cross-stream events.
    static constexpr int max_micro_batches = 4;
    std::array<hipStream_t, max_micro_batches> mb_stream{}, mb_copy_stream{};
    std::array<hipEvent_t, max_micro_batches> mb_copy_start{}, mb_copy_done{}, ev_gemm_done{}, ev_kv{},
        ev_mb_end{};
    hipEvent_t ev_decoded{}, ev_mb_start{};
    GpuMetadata mb_metadata[max_micro_batches];
    bool scratch_busy = false;
    int active_micro_batches = 1, last_qtip_micro_batches = 1;
    size_t sample_capacity = 0;
    // Prefix-cache KV page transfers (PREFIX_CACHE_REPORT.md): pack/unpack kernels on kv_stream,
    // copy-engine transfers on kv_copy_stream, ping-pong between two device staging buffers.
    // kv_ready[b]: staging b holds data for its consumer; kv_free[b]: staging b may be rewritten.
    int kv_chunk = 0, kv_parity = 0;
    std::array<DeviceBuffer<uint8_t>, 2> kv_staging;
    std::array<hipEvent_t, 2> kv_ready{}, kv_free{};
    hipStream_t kv_stream{}, kv_copy_stream{};
    DeviceBuffer<KvLayerPointers> kv_layers;
    explicit Shard(int id, int r, Checkpoint &checkpoint, const RuntimeOptions &options)
        : device(id), rank(r), bf16_prefill_rows(options.qtip_bf16_prefill_rows) {
        HIP_CHECK(hipSetDevice(device));
        verify_device();
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        HIP_CHECK(hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking));
        HIP_CHECK(hipEventCreateWithFlags(&copy_start, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&copy_done, hipEventDisableTiming));
        for (int i = 1; i < max_micro_batches; ++i) {
            HIP_CHECK(hipStreamCreateWithFlags(&mb_stream[size_t(i)], hipStreamNonBlocking));
            HIP_CHECK(hipStreamCreateWithFlags(&mb_copy_stream[size_t(i)], hipStreamNonBlocking));
        }
        for (int i = 0; i < max_micro_batches; ++i)
            for (auto *e : {&mb_copy_start[size_t(i)], &mb_copy_done[size_t(i)], &ev_gemm_done[size_t(i)],
                            &ev_kv[size_t(i)], &ev_mb_end[size_t(i)]})
                HIP_CHECK(hipEventCreateWithFlags(e, hipEventDisableTiming));
        for (auto *e : {&ev_decoded, &ev_mb_start})
            HIP_CHECK(hipEventCreateWithFlags(e, hipEventDisableTiming));
        runner = std::make_unique<ProjectionRunner>();
        runner->set_mode(parse_projection_mode(options.projection_mode));
        runner->set_search(parse_projection_search(options.projection_search));
        timings.enabled = options.profile;
        auto emb = checkpoint.bf16("model.embed_tokens.weight");
        if (emb.size() != size_t(32768) * 12288)
            throw std::runtime_error("Invalid embedding shape");
        embeddings.upload(emb);
        final_norm.upload(checkpoint.bf16("model.norm.weight"));
        if (final_norm.size() != 12288)
            throw std::runtime_error("Invalid final norm shape");
        layers.resize(88);
        qtip = checkpoint.config.quantization == Quantization::Qtip;
        auto layout = parse_code_layout(options.code_layout);
        for (int i = 0; qtip && i < 88; ++i)
            load_qtip_layer(checkpoint, i);
        for (int i = 0; !qtip && i < 88; ++i) {
            auto &l = layers[size_t(i)];
            l.q.upload(checkpoint.projection(i, "self_attn.q_proj", rank, true), layout);
            l.k.upload(checkpoint.projection(i, "self_attn.k_proj", rank, true), layout);
            l.v.upload(checkpoint.projection(i, "self_attn.v_proj", rank, true), layout);
            l.o.upload(checkpoint.projection(i, "self_attn.o_proj", rank, false), layout);
            l.gate.upload(checkpoint.projection(i, "mlp.gate_proj", rank, true), layout);
            l.up.upload(checkpoint.projection(i, "mlp.up_proj", rank, true), layout);
            l.down.upload(checkpoint.projection(i, "mlp.down_proj", rank, false), layout);
            std::string base = "model.layers." + std::to_string(i) + ".";
            l.input_norm.upload(checkpoint.bf16(base + "input_layernorm.weight"));
            l.post_norm.upload(checkpoint.bf16(base + "post_attention_layernorm.weight"));
            if (l.input_norm.size() != 12288 || l.post_norm.size() != 12288)
                throw std::runtime_error("Invalid block norm shape");
            if (i % 8 == 0)
                std::cerr << "rank " << rank << ": loaded block " << i << "/88\n";
        }
        auto name = std::string("lm_head.weight");
        const auto &ht = checkpoint.tensor(name);
        if (ht.shape(name) != std::vector<size_t>{32768, 12288})
            throw std::runtime_error("Invalid vocabulary projection shape");
        auto values = ht.floats(name);
        head.in = 12288;
        head.out = 16384;
        std::vector<uint8_t> weights(size_t(head.in) * size_t(head.out));
        std::vector<float> scales(size_t(head.out)), row(size_t(head.in));
        for (int o = 0; o < head.out; ++o) {
            for (int d = 0; d < head.in; ++d)
                row[size_t(d)] = values[size_t(o + rank * head.out) * size_t(head.in) + size_t(d)];
            auto quantized = quantize(row);
            scales[size_t(o)] = quantized.scale;
            std::copy(quantized.values.begin(), quantized.values.end(),
                      weights.begin() + std::ptrdiff_t(size_t(o) * size_t(head.in)));
        }
        head.dense.upload(weights);
        head.row_scale.upload(scales);
        sample_capacity = sampled_capacity(options);
        reserve(options.token_budget);
        // Split-K scratch: decode (rows <= 64) needs <= 64*14336*16*4 B ~ 59 MB
        // and a 256-row prefill chunk needs ~205 MB; the launch path grows the
        // buffer on demand, so one fat prefill pays a single transient realloc
        // instead of a permanent reservation for the whole token budget.
        size_t split_rows = std::min(options.token_budget, size_t(64));
        projection.partial.resize(split_rows * 16384 * 28);
        if (qtip) {
            qtip_dense = std::make_unique<qtip::DenseMatmul>();
            const int dense = qtip::dense_rows_threshold();
            if (dense > 0 && options.token_budget >= size_t(dense))
                qtip_dense_scratch.resize(qtip::dense_scratch_elements);
        }
    }
    void load_qtip_layer(const Checkpoint &checkpoint, int i) {
        auto &l = layers[size_t(i)];
        auto &w = l.qtip;
        auto lin = [&](const char *name) { return qtip::load_linear(checkpoint, i, name); };
        const auto q = lin("self_attn.q_proj"), k = lin("self_attn.k_proj"),
                   v = lin("self_attn.v_proj"), o = lin("self_attn.o_proj"),
                   gate = lin("mlp.gate_proj"), up = lin("mlp.up_proj"), down = lin("mlp.down_proj");
        // The stacked q/k/v and gate/up matmuls share one rotated input: the checkpoint must use
        // one SU per Hessian group (qtip_quant does) and one tlut per layer.
        if (q.su != k.su || q.su != v.su || gate.su != up.su)
            throw std::invalid_argument("QTIP layer " + std::to_string(i) +
                                        ": q/k/v or gate/up do not share SU");
        for (const auto *x : {&k, &v, &o, &gate, &up, &down})
            if (x->tlut != q.tlut)
                throw std::invalid_argument("QTIP layer " + std::to_string(i) +
                                            ": linears use different tluts");
        w.lut.upload(qtip::make_codebook(q.tlut));
        // Rank r owns rotated output rows [r*m/2, (r+1)*m/2) of every linear (whole 32-row
        // block-rows: q/o/down 192, k/v 16, gate/up 448 per rank).
        w.qkv.upload({q.block_rows(rank * 192, 192), k.block_rows(rank * 16, 16),
                      v.block_rows(rank * 16, 16)},
                     12288);
        w.o.upload({o.block_rows(rank * 192, 192)}, 12288);
        w.gate_up.upload({gate.block_rows(rank * 448, 448), up.block_rows(rank * 448, 448)}, 12288);
        w.down.upload({down.block_rows(rank * 192, 192)}, 28672);
        w.su_qkv.upload(q.su);
        w.su_o.upload(o.su);
        w.su_gate_up.upload(gate.su);
        w.su_down.upload(down.su);
        w.sv_q.upload(q.sv);
        w.sv_k.upload(k.sv);
        w.sv_v.upload(v.sv);
        w.sv_o.upload(o.sv);
        w.sv_gate.upload(gate.sv);
        w.sv_up.upload(up.sv);
        w.sv_down.upload(down.sv);
        std::string base = "model.layers." + std::to_string(i) + ".";
        l.input_norm.upload(checkpoint.bf16(base + "input_layernorm.weight"));
        l.post_norm.upload(checkpoint.bf16(base + "post_attention_layernorm.weight"));
        if (l.input_norm.size() != 12288 || l.post_norm.size() != 12288)
            throw std::invalid_argument("Invalid block norm shape");
        if (i % 8 == 0)
            std::cerr << "rank " << rank << ": loaded QTIP block " << i << "/88 ("
                      << double(w.bytes()) / 1e6 << " MB trellis)\n";
    }
    ~Shard() {
        (void)hipSetDevice(device);
        if (stream)
            (void)hipStreamDestroy(stream);
        if (copy_stream)
            (void)hipStreamDestroy(copy_stream);
        if (copy_start)
            (void)hipEventDestroy(copy_start);
        if (copy_done)
            (void)hipEventDestroy(copy_done);
        for (int i = 0; i < max_micro_batches; ++i) {
            for (auto st : {mb_stream[size_t(i)], mb_copy_stream[size_t(i)]})
                if (st)
                    (void)hipStreamDestroy(st);
            for (auto e : {mb_copy_start[size_t(i)], mb_copy_done[size_t(i)], ev_gemm_done[size_t(i)],
                           ev_kv[size_t(i)], ev_mb_end[size_t(i)]})
                if (e)
                    (void)hipEventDestroy(e);
        }
        for (auto e : {ev_decoded, ev_mb_start})
            if (e)
                (void)hipEventDestroy(e);
        for (auto st : {kv_stream, kv_copy_stream})
            if (st) {
                (void)hipStreamSynchronize(st);
                (void)hipStreamDestroy(st);
            }
        for (int b = 0; b < 2; ++b)
            for (auto e : {kv_ready[size_t(b)], kv_free[size_t(b)]})
                if (e)
                    (void)hipEventDestroy(e);
    }
    static size_t kv_page_bytes() {
        return kv_layer_page_bytes * 88;
    }
    // Staging for prefix-cache transfers; called before the KV pool is sized.
    void kv_init(int pages) {
        HIP_CHECK(hipSetDevice(device));
        kv_chunk = std::min(pages, kv_transfer_max_pages);
        HIP_CHECK(hipStreamCreateWithFlags(&kv_stream, hipStreamNonBlocking));
        HIP_CHECK(hipStreamCreateWithFlags(&kv_copy_stream, hipStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            kv_staging[size_t(b)].resize(size_t(kv_chunk) * kv_page_bytes());
            HIP_CHECK(hipEventCreateWithFlags(&kv_ready[size_t(b)], hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&kv_free[size_t(b)], hipEventDisableTiming));
            HIP_CHECK(hipEventRecord(kv_free[size_t(b)], kv_stream));
        }
    }
    // Device table of the final cache allocation's per-layer pointers.
    void kv_bind() {
        if (!kv_chunk)
            return;
        HIP_CHECK(hipSetDevice(device));
        std::vector<KvLayerPointers> table;
        for (auto &l : layers)
            table.push_back({l.cache_k.data(), l.cache_v.data(), l.k_scale.data(), l.v_scale.data()});
        kv_layers.upload(table);
    }
    // Enqueues one transfer; `host_page(i)` is the pinned address of host page index i.
    template <class HostPage>
    void kv_issue(bool save, std::span<const int> gpu, std::span<const int> host,
                  HostPage host_page, hipEvent_t start, hipEvent_t done) {
        HIP_CHECK(hipSetDevice(device));
        const size_t pb = kv_page_bytes(), n = gpu.size();
        hipStream_t first = save ? kv_stream : kv_copy_stream;
        for (size_t c0 = 0; c0 < n; c0 += size_t(kv_chunk)) {
            const size_t count = std::min(n - c0, size_t(kv_chunk));
            const size_t b = size_t(kv_parity);
            kv_parity ^= 1;
            KvPageList list;
            list.count = int(count);
            for (size_t i = 0; i < count; ++i)
                list.ids[i] = gpu[c0 + i];
            uint8_t *staging = kv_staging[b].data();
            auto copies = [&](hipStream_t s) {
                // runs of consecutive host pages inside one arena block -> one copy each
                for (size_t i = 0; i < count;) {
                    size_t j = i + 1;
                    while (j < count && host[c0 + j] == host[c0 + j - 1] + 1 &&
                           host_page(host[c0 + j]) == host_page(host[c0 + i]) + (j - i) * pb)
                        ++j;
                    std::byte *h = host_page(host[c0 + i]);
                    if (save)
                        HIP_CHECK(hipMemcpyAsync(h, staging + i * pb, (j - i) * pb,
                                                 hipMemcpyDeviceToHost, s));
                    else
                        HIP_CHECK(hipMemcpyAsync(staging + i * pb, h, (j - i) * pb,
                                                 hipMemcpyHostToDevice, s));
                    i = j;
                }
            };
            if (save) {
                HIP_CHECK(hipStreamWaitEvent(kv_stream, kv_free[b], 0));
                if (!c0)
                    HIP_CHECK(hipEventRecord(start, first));
                kv_pack_pages(kv_layers.data(), 88, list, staging, false, kv_stream);
                HIP_CHECK(hipEventRecord(kv_ready[b], kv_stream));
                HIP_CHECK(hipStreamWaitEvent(kv_copy_stream, kv_ready[b], 0));
                copies(kv_copy_stream);
                HIP_CHECK(hipEventRecord(kv_free[b], kv_copy_stream));
            } else {
                HIP_CHECK(hipStreamWaitEvent(kv_copy_stream, kv_free[b], 0));
                if (!c0)
                    HIP_CHECK(hipEventRecord(start, first));
                copies(kv_copy_stream);
                HIP_CHECK(hipEventRecord(kv_ready[b], kv_copy_stream));
                HIP_CHECK(hipStreamWaitEvent(kv_stream, kv_ready[b], 0));
                kv_pack_pages(kv_layers.data(), 88, list, staging, true, kv_stream);
                HIP_CHECK(hipEventRecord(kv_free[b], kv_stream));
            }
        }
        if (!n)
            HIP_CHECK(hipEventRecord(start, first));
        HIP_CHECK(hipEventRecord(done, save ? kv_copy_stream : kv_stream));
    }
    void reserve(size_t rows) {
        if (qtip) {
            // QTIP never touches the AQLM projection buffers; the lm_head path only sees the
            // sampled rows (at most sample_capacity per step, Scheduler::max_requests).
            const size_t n = std::max<size_t>(sample_capacity, 1);
            state.resize(rows * 12288);
            q.resize(rows * 6144);
            k.resize(rows * 512);
            v.resize(rows * 512);
            query.resize(rows * 6144);
            query_scale.resize(rows * 48);
            selected.resize(n * 12288);
            raw.resize(n * 16384);
            gathered.resize(n * 32768);
            sample_rows.resize(n);
            projection.input.resize(n * 14336);
            projection.input_scale.resize(n);
            projection.blas.resize(64 * 1024 * 1024);
            attention_workspace.resize(32 * 1024 * 1024);
            qtip_u.resize(rows * 28672);
            qtip_h.resize(rows * 28672);
            qtip_all.resize(rows * 2 * 14336);
            qtip_partial.resize(std::max<size_t>(rows, 64) * 28672);
            qtip_sync_counters.resize(64);
            HIP_CHECK(hipMemset(qtip_sync_counters.data(), 0, qtip_sync_counters.bytes()));
            qtip_sync = {qtip_sync_counters.data(), 64};
            return;
        }
        state.resize(rows * 12288);
        norm.resize(rows * 12288);
        tmp.resize(rows * 16384);
        selected.resize(rows * 12288);
        q.resize(rows * 6144);
        k.resize(rows * 512);
        v.resize(rows * 512);
        attention.resize(rows * 6144);
        gate.resize(rows * 14336);
        up.resize(rows * 14336);
        query.resize(rows * 6144);
        query_scale.resize(rows * 48);
        raw.resize(rows * 16384);
        gathered.resize(rows * 32768);
        sample_rows.resize(rows);
        projection.input.resize(rows * 14336);
        projection.input_scale.resize(rows);
        projection.weight.resize(size_t(12288) * 14336);
        projection.blas.resize(64 * 1024 * 1024);
        // Decode split-K attention partials: partition-32 at c=4 fits contexts
        // up to ~10k tokens in 32 MB; longer contexts (and prefill, which runs
        // unpartitioned) need no workspace at all — the per-run candidate
        // check in forward() falls back to the 1024-key partition or to a
        // single partition. The old flat 256 MB bought nothing (item 1).
        attention_workspace.resize(32 * 1024 * 1024);
        if (qtip) {
            qtip_u.resize(rows * 28672);
            qtip_h.resize(rows * 28672);
            qtip_all.resize(rows * 2 * 14336);
            qtip_partial.resize(std::max<size_t>(rows, 64) * 28672);
        }
    }
    int page_capacity(double fraction) {
        HIP_CHECK(hipSetDevice(device));
        size_t free = 0, total = 0;
        HIP_CHECK(hipMemGetInfo(&free, &total));
        return PagePool::capacity_for_budget(total, total - free, 0, fraction);
    }
    void allocate_cache(int pages) {
        HIP_CHECK(hipSetDevice(device));
        size_t vectors = size_t(pages) * 4 * 16;
        for (auto &l : layers) {
            l.cache_k.resize(vectors * 128);
            l.cache_v.resize(vectors * 128);
            l.k_scale.resize(vectors);
            l.v_scale.resize(vectors);
        }
    }
    void forward(const InferenceBatch &batch, ncclComm_t comm, HostTransport *transport,
                 int only_layer = -1, const std::vector<BFloat16> *injected = nullptr) {
        HIP_CHECK(hipSetDevice(device));
        if (qtip && batch.sample_rows.size() > sample_capacity)
            throw std::invalid_argument("More sampled rows than the lm_head buffers hold");
        timings.reset();
        metadata.upload(batch.layout, batch.tokens, stream);
        int rows = int(batch.tokens.size());
        if (injected)
            state.upload(*injected);
        else
            embed(embeddings.data(), metadata.tokens, state.data(), rows, 12288, stream);
        // Fused elementwise path (PERF_ROUND2_REPORT.md): the normed input is
        // quantised once into the projection workspace and shared by q/k/v
        // (gate/up); column projections write bf16 from their merge kernel;
        // row projections end in one scaled residual add.
        auto normed_input = [&](const BFloat16 *x, const BFloat16 *weight) {
            auto [q8, scales] = ProjectionRunner::prepare_input(projection, rows, 12288, stream);
            rms_norm_quantized(x, weight, q8, scales, rows, 12288, 1e-5f, stream);
        };
        auto project = [&](const GpuProjection &p, BFloat16 *y) {
            timings.measure("projections_ms", stream, [&] {
                ProjectionRunner::Call call;
                call.output = raw.data();
                call.bf16_output = y;
                runner->run(p, call, rows, projection, stream);
            });
        };
        // Row projections (o/down) end in the collective + scaled residual.
        // Zero-copy path (PERF_ROUND3 item 4): the classic publish kernel
        // copies the device partials to the pinned host exchange buffer and
        // ONE fused kernel reduces both ranks and updates the residual —
        // a launch and a full device pass per layer saved, with identical
        // rounding. The prepared variant consumes an already-quantised
        // workspace input.
        auto row_project_common = [&](const GpuProjection &p, ProjectionRunner::Call &call) {
            const size_t n = size_t(rows) * 12288;
            HostTransport::Exchange exchange{};
            if (transport && transport->zero_copy_eligible(n))
                exchange = transport->begin_exchange(rank, n);
            timings.measure("projections_ms", stream, [&] {
                runner->run(p, call, rows, projection, stream);
            });
            timings.measure("collectives_ms", stream, [&] {
                if (transport) {
                    if (exchange.valid)
                        transport->reduce_residual_dev(rank, exchange, raw.data(), state.data(),
                                                       p.row_scale.data(), n, 12288, stream);
                    else
                        transport->reduce_residual(rank, raw.data(), state.data(),
                                                   p.row_scale.data(), n, 12288, stream);
                } else {
                    RCCL_CHECK(ncclAllReduce(raw.data(), raw.data(), n, ncclFloat, ncclSum, comm,
                                             stream));
                    residual_add_scaled(state.data(), raw.data(), p.row_scale.data(), rows, 12288,
                                        stream);
                }
            });
        };
        auto row_project = [&](const GpuProjection &p, const BFloat16 *x) {
            ProjectionRunner::Call call;
            call.input = x;
            call.output = raw.data();
            call.apply_row_scale = false;
            row_project_common(p, call);
        };
        auto row_project_prepared = [&](const GpuProjection &p) {
            ProjectionRunner::Call call;
            call.output = raw.data();
            call.apply_row_scale = false;
            row_project_common(p, call);
        };
        int partition = 0;
        bool decode = true;
        for (size_t r = 0; r < batch.layout.lengths.size(); ++r)
            decode &= batch.layout.query_offsets[r + 1] - batch.layout.query_offsets[r] == 1;
        if (decode) {
            // Decode attention: the GQA kernel runs one block per (request, kv
            // head, key partition); the partition is sized so ~512 blocks cover
            // the longest context (ATTENTION_ROUND4_REPORT.md). Overrides:
            // BEHEMOTH_ATTN_PARTITION (keys, multiple of 32), BEHEMOTH_ATTN_BLOCKS.
            static const int requested_partition = [] {
                const char *v = std::getenv("BEHEMOTH_ATTN_PARTITION");
                return v ? std::atoi(v) : 0;
            }();
            static const int target_blocks = [] {
                const char *v = std::getenv("BEHEMOTH_ATTN_BLOCKS");
                return v ? std::atoi(v) : 512;
            }();
            partition = decode_attention_partition(batch.layout, attention_workspace.size(),
                                                   requested_partition, target_blocks);
        } else if (qtip) {
            // QTIP multi-query steps: split the keys when there are few query tiles (mixed
            // decode + small prefill pieces over long contexts, PREFILL_SCHED_REPORT.md).
            // BEHEMOTH_ATTN_PREFILL_BLOCKS=0 disables.
            static const int prefill_blocks = [] {
                const char *v = std::getenv("BEHEMOTH_ATTN_PREFILL_BLOCKS");
                return v ? std::atoi(v) : 512;
            }();
            if (prefill_blocks > 0)
                partition = prefill_attention_partition(batch.layout, attention_workspace.size(),
                                                        prefill_blocks);
        }
        // BEHEMOTH_ATTN_QP=bf16: unquantised BF16 queries and probabilities (K/V FP8).
        static const bool bf16_qp = [] {
            const char *v = std::getenv("BEHEMOTH_ATTN_QP");
            return v && std::string(v) == "bf16";
        }();
        if (qtip)
            qtip_forward(batch, rows, decode, partition, bf16_qp, comm, transport, only_layer);
        for (int index = only_layer < 0 ? 0 : only_layer;
             !qtip && index < (only_layer < 0 ? 88 : only_layer + 1); ++index) {
            auto &l = layers[size_t(index)];
            timings.measure("elementwise_ms", stream,
                            [&] { normed_input(state.data(), l.input_norm.data()); });
            project(l.q, q.data());
            project(l.k, k.data());
            project(l.v, v.data());
            timings.measure("attention_and_cache_ms", stream, [&] {
                rope_quantize(q.data(), k.data(), metadata.positions, rows, 48, 4, 1000000,
                              query.data(), query_scale.data(), stream);
                AttentionDeviceView view{query.data(),     query_scale.data(), l.cache_k.data(),
                                         l.cache_v.data(), l.k_scale.data(),   l.v_scale.data(),
                                         attention.data()};
                if (bf16_qp)
                    view.q_bf16 = q.data(); // rotated BF16 queries written by rope_quantize
                append_kv(metadata, view, k.data(), v.data(), stream);
                paged_attention(metadata, view, attention_workspace.data(),
                                attention_workspace.size(), partition, stream);
            });
            row_project(l.o, attention.data());
            timings.measure("elementwise_ms", stream,
                            [&] { normed_input(state.data(), l.post_norm.data()); });
            project(l.gate, gate.data());
            project(l.up, up.data());
            timings.measure("elementwise_ms", stream, [&] {
                // Fused silu + per-row quantisation straight into the
                // projection workspace (item 4): no bf16 intermediate round
                // trip, bit-identical values (bf16 rounding before the row
                // maximum, exactly like the stored-and-reread path).
                auto [q8, scales] = ProjectionRunner::prepare_input(projection, rows, 14336,
                                                                    stream);
                silu_multiply_quantized(gate.data(), up.data(), q8, scales, rows, 14336, stream);
            });
            row_project_prepared(l.down);
        }
        if (only_layer >= 0) {
            HIP_CHECK(hipStreamSynchronize(stream));
            return;
        }
        int n = int(batch.sample_rows.size());
        if (n) {
            sample_rows.upload(batch.sample_rows);
            timings.measure("elementwise_ms", stream, [&] {
                select_rows(state.data(), sample_rows.data(), selected.data(), n, 12288, stream);
                auto [q8, scales] = ProjectionRunner::prepare_input(projection, n, 12288, stream);
                rms_norm_quantized(selected.data(), final_norm.data(), q8, scales, n, 12288, 1e-5f,
                                   stream);
            });
            timings.measure("lm_head_ms", stream, [&] {
                ProjectionRunner::Call call;
                call.output = raw.data();
                runner->run(head, call, n, projection, stream);
            });
            timings.measure("collectives_ms", stream, [&] {
                if (transport)
                    transport->gather(rank, raw.data(), gathered.data(), size_t(n) * 16384, stream);
                else
                    RCCL_CHECK(ncclAllGather(raw.data(), gathered.data(), size_t(n) * 16384,
                                             ncclFloat, comm, stream));
            });
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    // Startup autotuning of the dense-prefill GEMMs (hipBLASLt's first heuristic choice is
    // 20-35 % slower than its best candidate for these shapes): every linear shape at the
    // batch sizes the dense path sees (dense threshold .. token budget, and the overlapped
    // halves). Synchronous; runs before the KV cache is sized.
    void tune_dense(size_t budget, int candidates) {
        HIP_CHECK(hipSetDevice(device));
        const int dense = qtip::dense_rows_threshold();
        if (!qtip || !qtip_dense || dense <= 0 || budget < size_t(dense) ||
            qtip_dense_scratch.size() < qtip::dense_scratch_elements)
            return;
        auto &w0 = layers[0].qtip;
        qtip::decode_dense(w0.gate_up.view(), w0.lut.lut.data(), qtip_dense_scratch.data(), stream);
        HIP_CHECK(hipMemsetAsync(qtip_u.data(), 0x3c, qtip_u.bytes(), stream)); // bf16 ~0.0115
        std::vector<int> sizes;
        for (size_t n = size_t(dense); n <= budget; n *= 2)
            sizes.push_back(int(n));
        sizes.push_back(int(budget));
        sizes.push_back(int(budget / 2));
        std::sort(sizes.begin(), sizes.end());
        sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
        std::erase_if(sizes, [&](int n) { return n < dense; });
        const std::pair<const char *, const qtip::GpuLinear *> shapes[] = {
            {"qkv", &w0.qkv}, {"o", &w0.o}, {"gate_up", &w0.gate_up}, {"down", &w0.down}};
        // Output types the dense path uses: bf16 (direct exchange payload) for gate/up always
        // and for q/k/v, o, down in bf16-exchange prefill steps; fp32 for the rest.
        const bool bf16_direct = qtip_bf16_direct();
        std::ostringstream log;
        log << "rank " << rank << " dense GEMM autotune (TFLOP/s):";
        for (int n : sizes)
            for (const auto &[name, w] : shapes) {
                const bool gate_up = w == &w0.gate_up;
                if (!(gate_up && bf16_direct)) {
                    const double tflops = qtip_dense->tune(w->rows, w->in, qtip_u.data(), w->in, n,
                                                           qtip_partial.data(), w->rows,
                                                           qtip_dense_scratch.data(), stream, candidates);
                    log << ' ' << name << '@' << n << '=' << int(tflops);
                }
                if (bf16_direct && (gate_up || (bf16_prefill_rows > 0 && n >= bf16_prefill_rows))) {
                    const double tflops = qtip_dense->tune(
                        w->rows, w->in, qtip_u.data(), w->in, n,
                        reinterpret_cast<BFloat16 *>(qtip_partial.data()), w->rows,
                        qtip_dense_scratch.data(), stream, candidates);
                    log << ' ' << name << "/bf16@" << n << '=' << int(tflops);
                }
            }
        std::cerr << log.str() << '\n';
    }
    // ---------------------------------------------------------------- QTIP execution
    // A micro-batch: rows [r0, r0 + rows) of the step, its stream, copy stream, attention
    // metadata and copy-engine channel (PREFILL_SCHED_REPORT.md). The whole step is one
    // micro-batch on the rank stream unless forward() splits a large prefill step into two
    // that run on two streams, so one micro-batch's all-gathers overlap the other's compute.
    struct Mb {
        int index = 0, r0 = 0, rows = 0;
        hipStream_t s{}, copy{};
        hipEvent_t copy_start{}, copy_done{};
        const GpuMetadata *meta = nullptr;
        bool overlap = false; // two micro-batches in flight
        bool input_ready = false; // the previous down epilogue wrote this layer's q/k/v input
        bool bf16_projection = false; // q/k/v, o and down payloads for pure prefill only
    };
    template <class F> void measure(const Mb &mb, const char *name, F f) {
        // Include every micro-batch. Summed event intervals on concurrent streams
        // are not wall-clock busy time; profile() marks that distinction explicitly.
        timings.measure(name, mb.s, f);
    }
    // Rank-blocked all-gather of `words` 32-bit words per rank through the micro-batch's block
    // of qtip_all ([2][rows][W] at word offset 2 * r0 * 14336): this rank's half must already
    // be at block + rank * words, or is produced there from the matmul slices when partial is
    // non-null (fp32 or bf16 payload).
    float *qtip_block(const Mb &mb) {
        return qtip_all.data() + size_t(2) * size_t(mb.r0) * 14336;
    }
    void qtip_exchange(const Mb &mb, const float *partial, int slices, size_t elements, bool bf16,
                       size_t words, ncclComm_t comm, HostTransport *transport) {
        float *all = qtip_block(mb);
        float *local = all + size_t(rank) * words;
        if (qtip_use_dma(mb.rows, transport)) {
            // slice sum on the device, then the copy-engine all-gather
            if (partial)
                measure(mb, "elementwise_ms", [&] {
                    qtip::reduce_publish(partial, slices, elements, bf16, local, nullptr, nullptr,
                                         nullptr, 0, mb.s);
                });
            measure(mb, "collectives_ms", [&] {
                transport->dma_gather(mb.index, rank, local, all, words, mb.s, mb.copy,
                                      mb.copy_start, mb.copy_done);
            });
            return;
        }
        if (mb.overlap)
            throw std::logic_error("QTIP micro-batch overlap needs the copy-engine transport");
        hipStream_t stream = mb.s;
        HostTransport::Exchange e{};
        if (transport && transport->zero_copy_eligible(words))
            e = transport->begin_exchange(rank, words);
        static const bool fused = [] {
            const char *v = std::getenv("BEHEMOTH_QTIP_FUSED_EXCHANGE");
            return !v || std::atoi(v) != 0;
        }();
        // The fused kernel spins with <= 64 resident blocks: decode-sized messages only.
        if (e.valid && fused && words <= size_t(8) * 16384) {
            // one kernel: publish (slice sum) + flag, then the peer half
            timings.measure("collectives_ms", stream, [&] {
                qtip::exchange(partial, slices, partial ? elements : words,
                               !partial ? qtip::ExchangeMode::Words
                               : bf16   ? qtip::ExchangeMode::Bf16
                                        : qtip::ExchangeMode::Fp32,
                               local, e.mine, e.flag, e.counter, e.seq, e.peer,
                               all + size_t(1 - rank) * words, e.peer_flag, stream);
            });
            return;
        }
        if (partial)
            timings.measure("elementwise_ms", stream, [&] {
                qtip::reduce_publish(partial, slices, elements, bf16, local, e.valid ? e.mine : nullptr,
                                     e.flag, e.counter, e.seq, stream);
            });
        else if (e.valid)
            timings.measure("collectives_ms", stream, [&] {
                zero_copy_publish(local, e.mine, words, e.flag, e.counter, e.seq, stream);
            });
        timings.measure("collectives_ms", stream, [&] {
            if (e.valid)
                qtip::gather_peer(e.peer, all, words, rank, e.peer_flag, e.seq, stream);
            else if (transport)
                transport->gather(rank, local, all, words, stream);
            else
                RCCL_CHECK(ncclAllGather(local, all, words, ncclFloat, comm, stream));
        });
    }
    // Copy-engine all-gathers for every QTIP step of more than BEHEMOTH_QTIP_DMA_ROWS - 1 rows
    // (default 9; smaller steps keep the fused zero-copy kernels).
    // bf16-output dense GEMMs straight into the exchange block (env BEHEMOTH_QTIP_BF16_DIRECT,
    // default 1; 0 restores the fp32 GEMM + reduce_publish rounding kernel).
    static bool qtip_bf16_direct() {
        static const bool on = [] {
            const char *v = std::getenv("BEHEMOTH_QTIP_BF16_DIRECT");
            return !v || std::atoi(v) != 0;
        }();
        return on;
    }
    static bool qtip_use_dma(int rows, HostTransport *transport) {
        static const int min_rows = [] {
            const char *v = std::getenv("BEHEMOTH_QTIP_DMA_ROWS");
            return v ? std::atoi(v) : 9;
        }();
        return transport && min_rows > 0 && rows >= min_rows && transport->dma_eligible(1);
    }
    // x . hatWr^T for this rank's rows of `w`, slices summed and all-gathered into the
    // micro-batch's qtip_all block. Dense path: the shard is decoded once per step into the
    // shared scratch; with two micro-batches, micro-batch 0 decodes (after micro-batch 1's
    // previous GEMM released the scratch) and micro-batch 1 waits for the decode.
    void qtip_project(Mb &mb, const qtip::GpuLinear &w, const qtip::GpuCodebook &lut, int ldu,
                      bool bf16, ncclComm_t comm, HostTransport *transport) {
        const int rows = mb.rows;
        const int dense = qtip::dense_rows_threshold();
        const bool use_dense = dense > 0 && rows >= dense;
        const int slices = use_dense ? 1 : qtip::matmul_slices(w, rows);
        // micro-batch regions use the widest row strides (28672) so that no stage of one
        // micro-batch overlaps any stage of the other
        const BFloat16 *u = qtip_u.data() + size_t(mb.r0) * 28672;
        float *partial = qtip_partial.data() + size_t(mb.r0) * 28672;
        if (mb.overlap && !use_dense)
            throw std::logic_error("QTIP micro-batch overlap needs the dense path");
        if (size_t(mb.r0) * 28672 + size_t(slices) * size_t(rows) * size_t(w.rows) >
            qtip_partial.size()) {
            if (mb.overlap || mb.r0)
                throw std::logic_error("QTIP partial buffer too small for the micro-batch");
            qtip_partial.resize(size_t(slices) * size_t(rows) * size_t(w.rows));
            partial = qtip_partial.data();
        }
        const size_t elements = size_t(rows) * size_t(w.rows);
        if (use_dense)
            qtip_dense_scratch.resize(qtip::dense_scratch_elements); // no-op when reserved
        const bool whole = use_dense && size_t(w.rows) * size_t(w.in) <= qtip_dense_scratch.size();
        // Copy-engine path: the dense GEMM writes this rank's half of the exchange block directly
        // (one slice: the reduce kernel would be a plain copy / bf16 rounding). bf16 payloads use
        // a hipBLASLt bf16-output plan (fp32 accumulation) on whole-shard GEMMs.
        const bool direct =
            use_dense && qtip_use_dma(rows, transport) && (!bf16 || (whole && qtip_bf16_direct()));
        BFloat16 *direct_bf16 = nullptr;
        if (direct && bf16)
            direct_bf16 = reinterpret_cast<BFloat16 *>(qtip_block(mb)) + size_t(rank) * elements;
        else if (direct)
            partial = qtip_block(mb) + size_t(rank) * elements;
        auto gemm = [&] {
            if (direct_bf16)
                qtip_dense->gemm(w.rows, w.in, u, ldu, rows, direct_bf16, w.rows,
                                 qtip_dense_scratch.data(), mb.s);
            else
                qtip_dense->gemm(w.rows, w.in, u, ldu, rows, partial, w.rows,
                                 qtip_dense_scratch.data(), mb.s);
        };
        if (use_dense) {
            measure(mb, "dense_projections_ms", [&] {
                if (!mb.overlap) {
                    if (whole) {
                        measure(mb, "qtip_weight_decode_ms", [&] {
                            qtip::decode_dense(w.view(), lut.lut.data(), qtip_dense_scratch.data(), mb.s);
                        });
                        measure(mb, "qtip_gemm_ms", gemm);
                    } else {
                        // Bounded-scratch fallback still contains interleaved decode/GEMMs.
                        measure(mb, "qtip_tiled_decode_gemm_ms", [&] {
                            qtip_dense->run(w.view(), lut.lut.data(), u, ldu, rows, partial,
                                            qtip_dense_scratch.data(), qtip_dense_scratch.size(), mb.s);
                        });
                    }
                    return;
                }
                if (!whole)
                    throw std::logic_error("QTIP overlap needs a whole-shard dense scratch");
                if (mb.index == 0) {
                    // every other micro-batch's previous GEMM released the scratch
                    for (int i = 1; scratch_busy && i < active_micro_batches; ++i)
                        HIP_CHECK(hipStreamWaitEvent(mb.s, ev_gemm_done[size_t(i)], 0));
                    measure(mb, "qtip_weight_decode_ms", [&] {
                        qtip::decode_dense(w.view(), lut.lut.data(), qtip_dense_scratch.data(), mb.s);
                    });
                    HIP_CHECK(hipEventRecord(ev_decoded, mb.s));
                } else
                    HIP_CHECK(hipStreamWaitEvent(mb.s, ev_decoded, 0));
                measure(mb, "qtip_gemm_ms", gemm);
                if (mb.index > 0) {
                    HIP_CHECK(hipEventRecord(ev_gemm_done[size_t(mb.index)], mb.s));
                    scratch_busy = true;
                }
            });
        } else
            measure(mb, "projections_ms", [&] {
                qtip::matmul(w, lut.lut.data(), u, ldu, rows, partial, slices, mb.s);
            });
        if (direct)
            qtip_exchange(mb, nullptr, 0, 0, false, bf16 ? elements / 2 : elements, comm, transport);
        else
            qtip_exchange(mb, partial, slices, elements, bf16, bf16 ? elements / 2 : elements, comm,
                          transport);
    }
    // One layer in four steps; with two micro-batches the steps are issued alternately
    // (step A of micro-batch 0, step A of 1, step B of 0, ...) so every cross-stream event is
    // recorded before the matching wait is issued.
    BFloat16 *mb_state(const Mb &mb) {
        return state.data() + size_t(mb.r0) * 12288;
    }
    // The fused decode-sized residual+input kernel needs one stream per counter set: the rank
    // stream of a non-overlapped step, at or below residual_sync_rows.
    qtip::ResidualSync *qtip_sync_for(const Mb &mb) {
        if (mb.overlap || mb.index != 0 || mb.rows > qtip::residual_sync_rows ||
            (qtip::row_transform_threshold() > 0 && mb.rows >= qtip::row_transform_threshold()))
            return nullptr;
        return &qtip_sync;
    }
    // A: q/k/v input transform (norm + SU + hadUt), stacked matmul, all-gather of the rotated
    // outputs, hadU + SV for this rank's heads, rope, KV append, attention, all-gather of the
    // head-sharded attention output.
    void qtip_step_attention(Layer &l, Mb &mb, int partition, bool bf16_qp, ncclComm_t comm,
                             HostTransport *transport) {
        auto &w = l.qtip;
        const int rows = mb.rows;
        BFloat16 *st = mb_state(mb), *u = qtip_u.data() + size_t(mb.r0) * 28672;
        if (!mb.input_ready)
            measure(mb, "elementwise_ms", [&] {
                qtip::input_transform(qtip::Source{st, 12288, 0, 12288, 0}, l.input_norm.data(),
                                      1e-5f, w.su_qkv.data(), 12288, rows, u, 12288, mb.s);
            });
        mb.input_ready = false;
        qtip_project(mb, w.qkv, w.lut, 12288, mb.bf16_projection, comm, transport);
        BFloat16 *qm = q.data() + size_t(mb.r0) * 6144, *km = k.data() + size_t(mb.r0) * 512,
                 *vm = v.data() + size_t(mb.r0) * 512;
        float *all = qtip_block(mb);
        measure(mb, "elementwise_ms", [&] {
            if (mb.bf16_projection)
                qtip::output_qkv(reinterpret_cast<const BFloat16 *>(all), rows, rank,
                                 w.sv_q.data(), w.sv_k.data(), w.sv_v.data(), qm, km, vm, mb.s);
            else
                qtip::output_qkv(all, rows, rank, w.sv_q.data(), w.sv_k.data(), w.sv_v.data(), qm, km,
                                 vm, mb.s);
        });
        // Attention writes this rank's heads straight into its half of the exchange block.
        auto *attention_out = reinterpret_cast<BFloat16 *>(all) + size_t(rank) * size_t(rows) * 6144;
        measure(mb, "attention_and_cache_ms", [&] {
            uint8_t *qq = query.data() + size_t(mb.r0) * 6144;
            float *qs = query_scale.data() + size_t(mb.r0) * 48;
            rope_quantize(qm, km, mb.meta->positions, rows, 48, 4, 1000000, qq, qs, mb.s);
            AttentionDeviceView view{qq,               qs,
                                     l.cache_k.data(), l.cache_v.data(),
                                     l.k_scale.data(), l.v_scale.data(),
                                     attention_out};
            if (bf16_qp)
                view.q_bf16 = qm;
            append_kv(*mb.meta, view, km, vm, mb.s);
            // a micro-batch's queries attend to the earlier micro-batches' keys of the same
            // layer: chained append events
            if (mb.overlap && mb.index > 0)
                HIP_CHECK(hipStreamWaitEvent(mb.s, ev_kv[size_t(mb.index - 1)], 0));
            if (mb.overlap)
                HIP_CHECK(hipEventRecord(ev_kv[size_t(mb.index)], mb.s));
            paged_attention(*mb.meta, view, attention_workspace.data(), attention_workspace.size(),
                            partition, mb.s);
        });
        qtip_exchange(mb, nullptr, 0, 0, false, size_t(rows) * 3072, comm, transport);
    }
    // B: o input transform (hadUt over the gathered 12288), matmul, all-gather, residual epilogue
    // fused with the MLP input transform.
    void qtip_step_o(Layer &l, Mb &mb, ncclComm_t comm, HostTransport *transport) {
        auto &w = l.qtip;
        const int rows = mb.rows;
        BFloat16 *u = qtip_u.data() + size_t(mb.r0) * 28672;
        float *all = qtip_block(mb);
        measure(mb, "elementwise_ms", [&] {
            qtip::input_transform(qtip::Source{all, 6144, size_t(rows) * 3072 * 2, 6144, 0}, nullptr,
                                  0.f, w.su_o.data(), 12288, rows, u, 12288, mb.s);
        });
        qtip_project(mb, w.o, w.lut, 12288, mb.bf16_projection, comm, transport);
        measure(mb, "elementwise_ms", [&] {
            if (mb.bf16_projection)
                qtip::output_residual_input(reinterpret_cast<const BFloat16 *>(all), rows,
                    w.sv_o.data(), mb_state(mb), l.post_norm.data(), 1e-5f, w.su_gate_up.data(), u,
                    12288, mb.s, qtip_sync_for(mb));
            else
                qtip::output_residual_input(all, rows, w.sv_o.data(), mb_state(mb), l.post_norm.data(),
                                            1e-5f, w.su_gate_up.data(), u, 12288, mb.s,
                                            qtip_sync_for(mb));
        });
    }
    // C: gate/up stacked matmul, bf16 all-gather, silu product, down input transform.
    void qtip_step_gate_up(Layer &l, Mb &mb, ncclComm_t comm, HostTransport *transport) {
        auto &w = l.qtip;
        const int rows = mb.rows;
        qtip_project(mb, w.gate_up, w.lut, 12288, true, comm, transport);
        BFloat16 *h = qtip_h.data() + size_t(mb.r0) * 28672,
                 *u = qtip_u.data() + size_t(mb.r0) * 28672;
        measure(mb, "elementwise_ms", [&] {
            qtip::output_gateup(reinterpret_cast<const BFloat16 *>(qtip_block(mb)), rows,
                                w.sv_gate.data(), w.sv_up.data(), h, mb.s);
            qtip::input_transform(qtip::Source{h, 28672, 0, 28672, 0}, nullptr, 0.f, w.su_down.data(),
                                  28672, rows, u, 28672, mb.s);
        });
    }
    // D: down matmul, all-gather, residual epilogue (fused with the next layer's q/k/v input
    // transform for row-batched sizes).
    void qtip_step_down(Layer &l, Mb &mb, Layer *next, ncclComm_t comm, HostTransport *transport) {
        auto &w = l.qtip;
        const int rows = mb.rows;
        qtip_project(mb, w.down, w.lut, 28672, mb.bf16_projection, comm, transport);
        measure(mb, "elementwise_ms", [&] {
            const int t = qtip::row_transform_threshold();
            // row-batched sizes fuse from registers; decode sizes through the rendezvous kernel
            if (next && ((t > 0 && rows >= t) || qtip_sync_for(mb))) {
                if (mb.bf16_projection)
                    qtip::output_residual_input(reinterpret_cast<const BFloat16 *>(qtip_block(mb)),
                        rows, w.sv_down.data(), mb_state(mb), next->input_norm.data(), 1e-5f,
                        next->qtip.su_qkv.data(), qtip_u.data() + size_t(mb.r0) * 28672, 12288, mb.s,
                        qtip_sync_for(mb));
                else
                    qtip::output_residual_input(qtip_block(mb), rows, w.sv_down.data(), mb_state(mb),
                                                next->input_norm.data(), 1e-5f, next->qtip.su_qkv.data(),
                                                qtip_u.data() + size_t(mb.r0) * 28672, 12288, mb.s,
                                                qtip_sync_for(mb));
                mb.input_ready = true;
            } else if (mb.bf16_projection)
                qtip::output_residual(reinterpret_cast<const BFloat16 *>(qtip_block(mb)), rows,
                                     w.sv_down.data(), mb_state(mb), mb.s);
            else
                qtip::output_residual(qtip_block(mb), rows, w.sv_down.data(), mb_state(mb), mb.s);
        });
    }
    // Sub-layout of rows [begin, end) of a step: whole or partial requests, lengths cut at the
    // last included position (causal: earlier rows never see later ones).
    static AttentionLayout sub_layout(const AttentionLayout &l, int begin, int end) {
        AttentionLayout s;
        s.query_heads = l.query_heads;
        s.kv_heads = l.kv_heads;
        s.physical_pages = l.physical_pages;
        s.table_stride = l.table_stride;
        s.query_offsets = {0};
        for (int r = 0; r < l.requests(); ++r) {
            const int a = std::max(begin, l.query_offsets[size_t(r)]),
                      b = std::min(end, l.query_offsets[size_t(r + 1)]);
            if (a >= b)
                continue;
            for (int q = a; q < b; ++q)
                s.positions.push_back(l.positions[size_t(q)]);
            s.query_offsets.push_back(int(s.positions.size()));
            s.lengths.push_back(b == l.query_offsets[size_t(r + 1)] ? l.lengths[size_t(r)]
                                                                   : l.positions[size_t(b - 1)] + 1);
            s.page_table.insert(s.page_table.end(),
                                l.page_table.begin() + std::ptrdiff_t(size_t(r) * size_t(l.table_stride)),
                                l.page_table.begin() + std::ptrdiff_t(size_t(r + 1) * size_t(l.table_stride)));
        }
        s.validate();
        return s;
    }
    // Micro-batch count for a step: dense prefill steps of at least BEHEMOTH_QTIP_OVERLAP_ROWS rows
    // (default 2x the dense threshold; 0 disables) run as up to BEHEMOTH_QTIP_MICRO_BATCHES
    // (default 2, max 4) overlapped micro-batches of at least the dense threshold each.
    int qtip_micro_batches(int rows, bool decode, HostTransport *transport) const {
        const int setting = qtip_overlap_setting().load(std::memory_order_relaxed);
        const int dense = qtip::dense_rows_threshold();
        const int min_rows = setting >= 0 ? setting : 2 * dense;
        if (decode || dense <= 0 || min_rows <= 0 || rows < min_rows || !qtip_dense ||
            qtip_dense_scratch.size() < qtip::dense_scratch_elements || !qtip_use_dma(dense, transport))
            return 1;
        static const int requested = [] {
            const char *v = std::getenv("BEHEMOTH_QTIP_MICRO_BATCHES");
            return std::clamp(v ? std::atoi(v) : 2, 1, max_micro_batches);
        }();
        int n = requested;
        while (n > 1 && rows / n < dense)
            --n;
        return n;
    }
    void qtip_forward(const InferenceBatch &batch, int rows, bool decode, int partition, bool bf16_qp,
                      ncclComm_t comm, HostTransport *transport, int only_layer) {
        const int first = only_layer < 0 ? 0 : only_layer, last = only_layer < 0 ? 88 : only_layer + 1;
        auto next_layer = [&](int i) {
            return only_layer < 0 && i + 1 < 88 ? &layers[size_t(i + 1)] : nullptr;
        };
        const int count = only_layer >= 0 ? 1 : qtip_micro_batches(rows, decode, transport);
        active_micro_batches = count;
        last_qtip_micro_batches = count;
        std::array<Mb, max_micro_batches> mbs{};
        bool has_decode = false;
        for (int r = 0; r < batch.layout.requests(); ++r)
            has_decode |= batch.layout.query_offsets[size_t(r + 1)] - batch.layout.query_offsets[size_t(r)] == 1;
        for (int i = 0; i < count; ++i) {
            auto &m = mbs[size_t(i)];
            m.index = i;
            m.r0 = int(size_t(rows) * size_t(i) / size_t(count));
            m.rows = int(size_t(rows) * size_t(i + 1) / size_t(count)) - m.r0;
            m.s = i ? mb_stream[size_t(i)] : stream;
            m.copy = i ? mb_copy_stream[size_t(i)] : copy_stream;
            m.copy_start = i ? mb_copy_start[size_t(i)] : copy_start;
            m.copy_done = i ? mb_copy_done[size_t(i)] : copy_done;
            m.overlap = count > 1;
            m.bf16_projection = qtip_bf16_prefill_enabled(m.rows, bf16_prefill_rows, has_decode);
            m.meta = &metadata;
        }
        if (count > 1) {
            partition = 0; // the micro-batches share the attention workspace: unsplit keys
            // micro-batches start after the embedding (and everything before) on the rank stream
            HIP_CHECK(hipEventRecord(ev_mb_start, stream));
            for (int i = 0; i < count; ++i) {
                auto &m = mbs[size_t(i)];
                if (i)
                    HIP_CHECK(hipStreamWaitEvent(m.s, ev_mb_start, 0));
                mb_metadata[i].upload(sub_layout(batch.layout, m.r0, m.r0 + m.rows), {}, m.s);
                m.meta = &mb_metadata[i];
            }
        }
        scratch_busy = false;
        for (int l = first; l < last; ++l) {
            auto &layer = layers[size_t(l)];
            for (int i = 0; i < count; ++i)
                qtip_step_attention(layer, mbs[size_t(i)], partition, bf16_qp, comm, transport);
            for (int i = 0; i < count; ++i)
                qtip_step_o(layer, mbs[size_t(i)], comm, transport);
            for (int i = 0; i < count; ++i)
                qtip_step_gate_up(layer, mbs[size_t(i)], comm, transport);
            for (int i = 0; i < count; ++i)
                qtip_step_down(layer, mbs[size_t(i)], next_layer(l), comm, transport);
        }
        for (int i = 1; i < count; ++i) {
            HIP_CHECK(hipEventRecord(ev_mb_end[size_t(i)], mbs[size_t(i)].s));
            HIP_CHECK(hipStreamWaitEvent(stream, ev_mb_end[size_t(i)], 0));
        }
        active_micro_batches = 1;
    }
    nlohmann::json profile() const {
        return {{"rank", rank},
                {"device", device},
                {"qtip_micro_batches", last_qtip_micro_batches},
                {"gpu_event_totals_are_sums_across_streams", true},
                {"gpu_event_totals", timings.summary()},
                {"projection_dispatch", runner->profile(false)},
                {"projection_kernel_resources", runner->resources()}};
    }
};
} // namespace
void tensor_parallel_smoke(std::array<int, 2> devices, bool rccl) {
    if (devices[0] == devices[1])
        throw std::invalid_argument("Tensor parallelism needs two distinct GPUs");
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    for (int id : devices) {
        if (id < 0 || id >= count)
            throw std::invalid_argument("Requested GPU unavailable");
        HIP_CHECK(hipSetDevice(id));
        verify_device();
        topology(id);
    }
    std::array<ncclComm_t, 2> comm{};
    if (rccl) {
        setenv("NCCL_P2P_DISABLE", "1", 1);
        setenv("NCCL_SHM_DISABLE", "0", 1);
        RCCL_CHECK(ncclCommInitAll(comm.data(), 2, devices.data()));
    }
    constexpr size_t probe_count = 262144;
    std::unique_ptr<HostTransport> host;
    if (!rccl)
        host = std::make_unique<HostTransport>(probe_count);
    auto started = std::chrono::steady_clock::now();
    std::atomic<bool> failed = false;
    std::array<std::future<void>, 2> jobs;
    for (int rank = 0; rank < 2; ++rank)
        jobs[size_t(rank)] = std::async(std::launch::async, [&, rank] {
            try {
                HIP_CHECK(hipSetDevice(devices[size_t(rank)]));
                DeviceBuffer<float> x;
                std::vector<float> data(probe_count, float(rank + 1));
                x.upload(data);
                if (host)
                    host->reduce(rank, x.data(), probe_count, nullptr);
                else
                    RCCL_CHECK(ncclAllReduce(x.data(), x.data(), probe_count, ncclFloat, ncclSum,
                                             comm[size_t(rank)], nullptr));
                HIP_CHECK(hipDeviceSynchronize());
                for (float v : x.download())
                    if (v != 3)
                        throw std::runtime_error(
                            "Collective reduction produced an incorrect result");
                DeviceBuffer<float> all(probe_count * 2);
                x.upload(data);
                if (host)
                    host->gather(rank, x.data(), all.data(), probe_count, nullptr);
                else
                    RCCL_CHECK(ncclAllGather(x.data(), all.data(), probe_count, ncclFloat,
                                             comm[size_t(rank)], nullptr));
                HIP_CHECK(hipDeviceSynchronize());
                auto gathered = all.download();
                for (size_t i = 0; i < gathered.size(); ++i)
                    if (gathered[i] != float(i / probe_count + 1))
                        throw std::runtime_error("Collective gather produced an incorrect result");
            } catch (...) {
                if (!failed.exchange(true)) {
                    if (host)
                        host->abort();
                    for (auto c : comm)
                        if (c)
                            ncclCommAbort(c);
                }
                throw;
            }
        });
    std::exception_ptr error;
    for (auto &job : jobs)
        try {
            job.get();
        } catch (...) {
            error = std::current_exception();
        }
    if (!failed)
        for (auto c : comm)
            if (c)
                ncclCommDestroy(c);
    if (error)
        std::rethrow_exception(error);
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cerr
        << "Collective smoke: " << (rccl ? "RCCL SHM, NCCL_P2P_DISABLE=1" : "explicit pinned-host")
        << "; reduction/gather validated; "
        << (double(probe_count) * sizeof(float) * 10 / elapsed / 1e9)
        << " GB/s end-to-end smoke transfer rate (includes allocation, host sum and validation)\n";
}
struct TensorParallelModel::Impl {
    std::array<std::unique_ptr<Shard>, 2> shards;
    std::array<std::unique_ptr<RankWorker>, 2> workers;
    std::mutex execution_mutex;
    std::array<ncclComm_t, 2> comm{};
    std::unique_ptr<HostTransport> transport;
    std::atomic<bool> aborted = false;
    bool profiling = false;
    nlohmann::json last;
    int last_query_rows = 0, last_sample_rows = 0;
    int pages = 0;
    // Persistent pinned logits staging (PERF_ROUND3 item 2): one portable
    // pinned buffer sized for the token budget replaces the per-step pageable
    // vector, blocking pageable D2H and rank-interleave copy.
    float *logits_host = nullptr;
    size_t logits_capacity = 0;
    // Prefix-cache transfers: pinned host arena (per rank, 64-page blocks on the GPU's NUMA
    // node), in-flight transfer events (timing-enabled, recycled).
    static constexpr size_t kv_block_pages = 64;
    mutable std::mutex kv_mutex;
    std::mutex kv_alloc_mutex; // serialises arena growth (allocation runs without kv_mutex)
    std::array<std::vector<std::byte *>, 2> kv_blocks;
    size_t kv_arena_pages = 0;
    struct KvTransfer {
        std::array<hipEvent_t, 2> start{}, done{};
    };
    std::map<uint64_t, KvTransfer> kv_transfers;
    std::vector<KvTransfer> kv_event_pool;
    uint64_t kv_next = 1;
    std::byte *kv_host_page(int rank, int page) const {
        return kv_blocks[size_t(rank)][size_t(page) / kv_block_pages] +
               (size_t(page) % kv_block_pages) * Shard::kv_page_bytes();
    }
    ~Impl() {
        for (auto &worker : workers)
            worker.reset();
        for (auto &shard : shards)
            if (shard && shard->kv_chunk) {
                (void)hipSetDevice(shard->device);
                (void)hipStreamSynchronize(shard->kv_stream);
                (void)hipStreamSynchronize(shard->kv_copy_stream);
            }
        for (auto &blocks : kv_blocks)
            for (auto *b : blocks)
                (void)hipHostFree(b);
        for (auto &[id, t] : kv_transfers) {
            (void)id;
            kv_event_pool.push_back(t);
        }
        for (auto &t : kv_event_pool)
            for (auto e : {t.start[0], t.start[1], t.done[0], t.done[1]})
                if (e)
                    (void)hipEventDestroy(e);
        if (!aborted)
            for (auto c : comm)
                if (c)
                    ncclCommDestroy(c);
        if (logits_host)
            (void)hipHostFree(logits_host);
    }
    void abort() {
        if (!aborted.exchange(true)) {
            if (transport)
                transport->abort();
            for (auto c : comm)
                if (c)
                    ncclCommAbort(c);
        }
    }
};
TensorParallelModel::TensorParallelModel(const RuntimeOptions &options)
    : impl_(std::make_unique<Impl>()) {
    if (!std::isfinite(options.memory_fraction) || options.memory_fraction <= 0 ||
        options.memory_fraction > 1)
        throw std::invalid_argument("Memory fraction must be in (0,1]");
    (void)parse_projection_mode(options.projection_mode);
    (void)parse_projection_search(options.projection_search);
    (void)parse_code_layout(options.code_layout);
    impl_->profiling = options.profile;
    if (options.token_budget < 1 || options.token_budget > 4096 || options.prefill_chunk < 1 ||
        options.prefill_chunk > options.token_budget)
        throw std::invalid_argument("Token budget must be 1..4096 and cover the prefill chunk");
    if (options.qtip_bf16_prefill_rows != 0 &&
        (options.qtip_bf16_prefill_rows < 64 || options.qtip_bf16_prefill_rows > 4096))
        throw std::invalid_argument("qtip-bf16-prefill-rows must be 0 (off) or 64..4096");
    if (std::min(options.scheduler.max_requests, options.token_budget) > sampled_capacity(options))
        throw std::invalid_argument("Scheduler max_requests exceeds the sampled-row capacity");
    if (options.qtip_dense_rows >= 0)
        qtip::set_dense_rows_threshold(options.qtip_dense_rows);
    if (options.qtip_row_transforms >= 0)
        qtip::set_row_transform_threshold(options.qtip_row_transforms);
    tensor_parallel_smoke(options.devices, options.rccl_shared_memory);
    Checkpoint checkpoint(options.model);
    if (checkpoint.config.quantization != Quantization::Qtip && options.token_budget > 1024)
        throw std::invalid_argument("AQLM token budget must be 1..1024");
    for (int rank = 0; rank < 2; ++rank)
        impl_->shards[size_t(rank)] =
            std::make_unique<Shard>(options.devices[size_t(rank)], rank, checkpoint, options);
    if (options.rccl_shared_memory)
        RCCL_CHECK(ncclCommInitAll(impl_->comm.data(), 2, options.devices.data()));
    else {
        const bool is_qtip = checkpoint.config.quantization == Quantization::Qtip;
        const size_t zc_rows = options.zero_copy_rows ? options.zero_copy_rows : 8;
        impl_->transport = std::make_unique<HostTransport>(
            options.token_budget * 16384,
            options.zero_copy_transport ? std::min(zc_rows, options.token_budget) * 16384 : 0,
            options.devices);
        // QTIP all-gathers of more than 8 rows: copy-engine channels (largest message: the
        // bf16 gate/up words, rows x 14336).
        if (is_qtip && options.zero_copy_transport)
            impl_->transport->enable_dma(options.token_budget, 4);
    }
    for (auto &worker : impl_->workers)
        worker = std::make_unique<RankWorker>();
    {
        // Autotuner pre-warm (PERF_ROUND3 item 3): first use of a row count
        // re-tunes the whole 616-weight set (~1.3 s stall inside the first
        // forward). Walk the expected decode row counts plus the prefill
        // chunk through synthetic batches at startup, on a small temporary
        // cache, so the final cache sizing also sees the fully grown tuning
        // scratch (partial/weight buffers) before pages are computed.
        static const bool prewarm_enabled = [] {
            const char *v = std::getenv("BEHEMOTH_PREWARM");
            return !v || std::atoi(v) != 0;
        }();
        // Decode row counts the scheduler produces (rows = active requests,
        // bounded by the realistic concurrency). Prefill-sized row counts are
        // prompt-length dependent and their tuning candidates reconstruct
        // dense weights (~20 s per row count), so they stay lazy: a new
        // prompt length still pays one tuning pass inside its prefill.
        // BEHEMOTH_PREWARM_ROWS="1,2,..." overrides the set.
        std::vector<int> rows_list;
        if (const char *v = std::getenv("BEHEMOTH_PREWARM_ROWS")) {
            std::istringstream stream(v);
            std::string item;
            while (std::getline(stream, item, ','))
                if (int r = std::atoi(item.c_str()); r >= 1 && size_t(r) <= options.token_budget)
                    rows_list.push_back(r);
        } else
            for (int r = 1; r <= 4; ++r)
                if (size_t(r) <= options.token_budget)
                    rows_list.push_back(r);
        const size_t chunk = options.prefill_chunk;
        if (prewarm_enabled && !rows_list.empty()) {
            const auto started = std::chrono::steady_clock::now();
            const int warm_pages = int(std::max<size_t>(64, ceil_div(chunk, 16)));
            for (auto &shard : impl_->shards)
                shard->allocate_cache(warm_pages);
            auto run_batch = [&](InferenceBatch &batch) {
                batch.layout.physical_pages = warm_pages;
                batch.layout.validate();
                std::array<std::future<void>, 2> jobs;
                for (int rank = 0; rank < 2; ++rank)
                    jobs[size_t(rank)] = impl_->workers[size_t(rank)]->submit([&, rank] {
                        try {
                            impl_->shards[size_t(rank)]->forward(
                                batch, impl_->comm[size_t(rank)], impl_->transport.get());
                        } catch (...) {
                            impl_->abort();
                            throw;
                        }
                    });
                std::exception_ptr error;
                for (auto &job : jobs)
                    try {
                        job.get();
                    } catch (...) {
                        error = std::current_exception();
                    }
                if (error)
                    std::rethrow_exception(error);
            };
            // Decode-shaped: rows requests x one query, sampling every row
            // (tunes the layer set at `rows` and lm_head at `rows`).
            for (int rows : rows_list) {
                InferenceBatch batch;
                auto &l = batch.layout;
                l.table_stride = 1;
                l.query_offsets.push_back(0);
                for (int r = 0; r < rows; ++r) {
                    l.lengths.push_back(1);
                    l.positions.push_back(0);
                    l.page_table.push_back(r);
                    l.query_offsets.push_back(r + 1);
                    batch.tokens.push_back(1);
                    batch.sample_rows.push_back(r);
                    batch.sample_ids.push_back(0);
                }
                run_batch(batch);
            }
            double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cerr << "Autotuner pre-warm: " << rows_list.size() << " decode row counts in "
                      << seconds << " s\n";
        }
    }
    {
        // Dense-prefill GEMM autotune on both ranks (BEHEMOTH_QTIP_DENSE_TUNE = candidates,
        // default 64; 0 disables).
        static const int candidates = [] {
            const char *v = std::getenv("BEHEMOTH_QTIP_DENSE_TUNE");
            return v ? std::atoi(v) : 64;
        }();
        if (candidates > 0 && impl_->shards[0]->qtip) {
            const auto started = std::chrono::steady_clock::now();
            std::array<std::future<void>, 2> jobs;
            for (int rank = 0; rank < 2; ++rank)
                jobs[size_t(rank)] = impl_->workers[size_t(rank)]->submit([&, rank] {
                    impl_->shards[size_t(rank)]->tune_dense(options.token_budget, candidates);
                });
            std::exception_ptr error;
            for (auto &job : jobs)
                try {
                    job.get();
                } catch (...) {
                    error = std::current_exception();
                }
            if (error)
                std::rethrow_exception(error);
            std::cerr << "Dense GEMM autotune: "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                      << " s\n";
        }
    }
    if (options.kv_transfer_pages > 0)
        for (auto &shard : impl_->shards)
            shard->kv_init(options.kv_transfer_pages);
    impl_->pages = std::min(impl_->shards[0]->page_capacity(options.memory_fraction),
                            impl_->shards[1]->page_capacity(options.memory_fraction));
    for (auto &shard : impl_->shards) {
        shard->allocate_cache(impl_->pages);
        shard->kv_bind();
    }
    // Transfer events are created up front (no runtime allocations while serving).
    for (int i = 0; options.kv_transfer_pages > 0 && i < 32; ++i) {
        Impl::KvTransfer t;
        for (int rank = 0; rank < 2; ++rank) {
            HIP_CHECK(hipSetDevice(impl_->shards[size_t(rank)]->device));
            HIP_CHECK(hipEventCreate(&t.start[size_t(rank)]));
            HIP_CHECK(hipEventCreate(&t.done[size_t(rank)]));
        }
        impl_->kv_event_pool.push_back(t);
    }
    impl_->logits_capacity = sampled_capacity(options) * 32768;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&impl_->logits_host),
                            impl_->logits_capacity * sizeof(float), hipHostMallocPortable));
    std::cerr << "KV pages=" << impl_->pages << ", aggregate token capacity=" << impl_->pages * 16
              << ", bytes/token/GPU=" << PagePool::bytes_per_page() / 16 << '\n';
}
TensorParallelModel::~TensorParallelModel() = default;
void set_qtip_overlap_rows(int rows) {
    qtip_overlap_setting().store(rows, std::memory_order_relaxed);
}
int TensorParallelModel::cache_pages() const {
    return impl_->pages;
}
nlohmann::json TensorParallelModel::last_profile() const {
    std::lock_guard lock(impl_->execution_mutex);
    return impl_->last;
}
nlohmann::json TensorParallelModel::projection_tuning(int layer) const {
    std::lock_guard lock(impl_->execution_mutex);
    if (layer < -1 || layer >= 88)
        throw std::invalid_argument("Tuning report layer must be -1 or 0..87");
    auto ranks = nlohmann::json::array();
    for (const auto &shard : impl_->shards) {
        auto selections = nlohmann::json::array();
        auto add = [&](const GpuProjection &p, const std::string &name, int index, int rows,
                       bool scaled) {
            if (!rows)
                return;
            auto report = shard->runner->selection(p, rows, scaled);
            report["weight_name"] = name;
            report["tensor_identity"] = p.tuning_identity;
            report["layer"] = index < 0 ? nlohmann::json(nullptr) : nlohmann::json(index);
            selections.push_back(std::move(report));
        };
        const int first = layer < 0 ? 0 : layer, last = layer < 0 ? 88 : layer + 1;
        // QTIP layers have no autotuned projections (fixed GEMV/WMMA dispatch); only lm_head.
        for (int i = first; !shard->qtip && i < last; ++i) {
            const auto &l = shard->layers[size_t(i)];
            const auto prefix = "model.layers." + std::to_string(i) + ".";
            const int rows = impl_->last_query_rows;
            add(l.q, prefix + "self_attn.q_proj", i, rows, true);
            add(l.k, prefix + "self_attn.k_proj", i, rows, true);
            add(l.v, prefix + "self_attn.v_proj", i, rows, true);
            add(l.o, prefix + "self_attn.o_proj", i, rows, false);
            add(l.gate, prefix + "mlp.gate_proj", i, rows, true);
            add(l.up, prefix + "mlp.up_proj", i, rows, true);
            add(l.down, prefix + "mlp.down_proj", i, rows, false);
        }
        add(shard->head, "lm_head", -1, impl_->last_sample_rows, true);
        ranks.push_back({{"rank", shard->rank},
                         {"device", shard->device},
                         {"selections", std::move(selections)}});
    }
    auto timing = nlohmann::json::object();
    for (const auto *field : {"wall_ms", "execution_wall_ms", "profile_collection_ms"})
        if (impl_->last.contains(field))
            timing[field] = impl_->last.at(field);
    return {{"layer", layer < 0 ? nlohmann::json("all") : nlohmann::json(layer)},
            {"last_step_query_rows", impl_->last_query_rows},
            {"last_step_sample_rows", impl_->last_sample_rows},
            {"profiling_enabled", impl_->profiling},
            {"last_step_timing", timing.empty() ? nlohmann::json(nullptr) : std::move(timing)},
            {"ranks", std::move(ranks)},
            {"note", "Cached first-use calibration samples, filtered to the last completed "
                     "model step's row counts and actual row-scaling policy. Not per-step kernel "
                     "measurements. Fetch after generation completes; the snapshot holds the "
                     "model execution lock and does not retune."}};
}
std::vector<BFloat16> TensorParallelModel::block_for_validation(int layer,
                                                                std::span<const BFloat16> input) {
    std::lock_guard execution_lock(impl_->execution_mutex);
    if (layer < 0 || layer >= 88 || input.size() != 12288)
        throw std::invalid_argument("Invalid block validation input");
    std::vector<BFloat16> values(input.begin(), input.end());
    InferenceBatch batch;
    batch.layout.physical_pages = impl_->pages;
    batch.layout.table_stride = 1;
    batch.layout.query_offsets = {0, 1};
    batch.layout.positions = {0};
    batch.layout.lengths = {1};
    batch.layout.page_table = {0};
    batch.tokens = {1};
    std::array<std::future<void>, 2> jobs;
    for (int rank = 0; rank < 2; ++rank)
        jobs[size_t(rank)] = impl_->workers[size_t(rank)]->submit([&, rank] {
            try {
                impl_->shards[size_t(rank)]->forward(batch, impl_->comm[size_t(rank)],
                                                     impl_->transport.get(), layer, &values);
            } catch (...) {
                impl_->abort();
                throw;
            }
        });
    std::exception_ptr error;
    for (auto &job : jobs)
        try {
            job.get();
        } catch (...) {
            error = std::current_exception();
        }
    if (error)
        std::rethrow_exception(error);
    HIP_CHECK(hipSetDevice(impl_->shards[0]->device));
    auto first = impl_->shards[0]->state.download();
    HIP_CHECK(hipSetDevice(impl_->shards[1]->device));
    auto second = impl_->shards[1]->state.download();
    for (size_t i = 0; i < first.size(); ++i)
        if (first[i].bits != second[i].bits)
            throw std::runtime_error("Tensor-parallel residual streams diverged");
    return first;
}
std::span<const float> TensorParallelModel::forward(const InferenceBatch &batch) {
    std::lock_guard execution_lock(impl_->execution_mutex);
    const auto started = std::chrono::steady_clock::now();
    batch.layout.validate();
    if (batch.tokens.size() != size_t(batch.layout.queries()) ||
        batch.layout.physical_pages != impl_->pages ||
        batch.sample_rows.size() * 32768 > impl_->logits_capacity)
        throw std::invalid_argument("Inference batch shape mismatch");
    std::array<std::future<void>, 2> jobs;
    for (int rank = 0; rank < 2; ++rank)
        jobs[size_t(rank)] = impl_->workers[size_t(rank)]->submit([&, rank] {
            try {
                impl_->shards[size_t(rank)]->forward(batch, impl_->comm[size_t(rank)],
                                                     impl_->transport.get());
            } catch (...) {
                impl_->abort();
                throw;
            }
        });
    std::exception_ptr error;
    for (auto &job : jobs)
        try {
            job.get();
        } catch (...) {
            error = std::current_exception();
        }
    if (error)
        std::rethrow_exception(error);
    size_t n = batch.sample_rows.size();
    std::span<const float> result{};
    if (n) {
        // Single async D2H into pinned staging; the split sample_logits reads
        // the rank halves directly, so no interleave copy exists.
        HIP_CHECK(hipSetDevice(impl_->shards[0]->device));
        HIP_CHECK(hipMemcpyAsync(impl_->logits_host, impl_->shards[0]->gathered.data(),
                                 n * 32768 * sizeof(float), hipMemcpyDeviceToHost,
                                 impl_->shards[0]->stream));
        HIP_CHECK(hipStreamSynchronize(impl_->shards[0]->stream));
        result = {impl_->logits_host, n * 32768};
    }
    if (impl_->profiling) {
        const auto execution_finished = std::chrono::steady_clock::now();
        auto ranks = nlohmann::json::array();
        for (const auto &shard : impl_->shards) {
            HIP_CHECK(hipSetDevice(shard->device));
            ranks.push_back(shard->profile());
        }
        const auto profile_finished = std::chrono::steady_clock::now();
        impl_->last = {
            {"wall_ms",
             std::chrono::duration<double, std::milli>(profile_finished - started).count()},
            {"execution_wall_ms",
             std::chrono::duration<double, std::milli>(execution_finished - started).count()},
            {"profile_collection_ms",
             std::chrono::duration<double, std::milli>(profile_finished - execution_finished)
                 .count()},
            {"tokens", batch.tokens.size()},
            {"sampled_requests", n},
            {"context_lengths", batch.layout.lengths},
            {"ranks", std::move(ranks)},
            {"note", "Diagnostic events add overhead; discard first-use autotuning "
                     "steps. Collective event times include host rendezvous waits."}};
    }
    impl_->last_query_rows = int(batch.tokens.size());
    impl_->last_sample_rows = int(n);
    return result;
}
namespace {
int device_numa_node(int device) {
    char bus[64]{};
    if (hipDeviceGetPCIBusId(bus, sizeof(bus), device) != hipSuccess)
        return -1;
    std::string id(bus);
    for (auto &c : id)
        c = char(std::tolower(static_cast<unsigned char>(c)));
    std::ifstream f(std::filesystem::path("/sys/bus/pci/devices") / id / "numa_node");
    int node = -1;
    f >> node;
    return node;
}
} // namespace
size_t TensorParallelModel::kv_page_bytes() const {
    return Shard::kv_page_bytes();
}
size_t TensorParallelModel::kv_host_bytes() const {
    std::lock_guard lock(impl_->kv_mutex);
    return impl_->kv_arena_pages * Shard::kv_page_bytes() * 2;
}
std::span<const std::byte> TensorParallelModel::kv_host_page(int rank, int page) const {
    std::lock_guard lock(impl_->kv_mutex);
    if (rank < 0 || rank > 1 || page < 0 || size_t(page) >= impl_->kv_arena_pages)
        throw std::out_of_range("KV host page outside the arena");
    return {impl_->kv_host_page(rank, page), Shard::kv_page_bytes()};
}
size_t TensorParallelModel::kv_host_pages() const {
    std::lock_guard lock(impl_->kv_mutex);
    return impl_->kv_arena_pages;
}
void TensorParallelModel::kv_host_reserve(size_t pages) {
    std::lock_guard alloc_lock(impl_->kv_alloc_mutex);
    static const bool numa = [] {
        const char *v = std::getenv("BEHEMOTH_KV_NUMA");
        return !v || std::atoi(v) != 0;
    }();
    size_t have = 0;
    {
        std::lock_guard lock(impl_->kv_mutex);
        have = impl_->kv_arena_pages;
    }
    if (have >= pages)
        return;
    const size_t blocks = ceil_div(pages - have, Impl::kv_block_pages);
    const size_t bytes = Impl::kv_block_pages * Shard::kv_page_bytes();
    // One thread per rank (each bound to its GPU's NUMA node; hipHostMallocNumaUser follows the
    // calling thread's memory policy), allocating outside the transfer lock.
    std::array<std::vector<std::byte *>, 2> fresh;
    std::array<std::exception_ptr, 2> errors;
    auto allocate = [&](int rank) {
        try {
            const int device = impl_->shards[size_t(rank)]->device;
            HIP_CHECK(hipSetDevice(device));
            const int node = numa ? device_numa_node(device) : -1;
            unsigned long mask = node >= 0 && node < 64 ? 1UL << node : 0;
            const bool bound =
                mask && syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, &mask, 65UL) == 0;
            hipError_t e = hipSuccess;
            for (size_t b = 0; b < blocks && e == hipSuccess; ++b) {
                void *p = nullptr;
                e = hipHostMalloc(&p, bytes,
                                  hipHostMallocPortable | (bound ? hipHostMallocNumaUser : 0));
                if (e == hipSuccess)
                    fresh[size_t(rank)].push_back(static_cast<std::byte *>(p));
            }
            if (bound)
                (void)syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, 0UL);
            HIP_CHECK(e);
        } catch (...) {
            errors[size_t(rank)] = std::current_exception();
        }
    };
    std::thread second(allocate, 1);
    allocate(0);
    second.join();
    for (auto &e : errors)
        if (e) {
            for (auto &f : fresh)
                for (auto *p : f)
                    (void)hipHostFree(p);
            std::rethrow_exception(e);
        }
    std::lock_guard lock(impl_->kv_mutex);
    for (int rank = 0; rank < 2; ++rank)
        impl_->kv_blocks[size_t(rank)].insert(impl_->kv_blocks[size_t(rank)].end(),
                                              fresh[size_t(rank)].begin(), fresh[size_t(rank)].end());
    impl_->kv_arena_pages += blocks * Impl::kv_block_pages;
}
uint64_t TensorParallelModel::kv_start(bool save, std::span<const int> gpu,
                                       std::span<const int> host) {
    auto &impl = *impl_;
    const int pages = impl.pages;
    std::lock_guard lock(impl.kv_mutex);
    if (!impl.shards[0]->kv_chunk)
        throw std::logic_error("KV transfers need RuntimeOptions::kv_transfer_pages");
    if (gpu.size() != host.size())
        throw std::invalid_argument("KV transfer page lists differ in length");
    for (size_t i = 0; i < gpu.size(); ++i)
        if (gpu[i] < 0 || gpu[i] >= pages || host[i] < 0 || size_t(host[i]) >= impl.kv_arena_pages)
            throw std::out_of_range("KV transfer page outside the pool or arena");
    Impl::KvTransfer t;
    if (!impl.kv_event_pool.empty()) {
        t = impl.kv_event_pool.back();
        impl.kv_event_pool.pop_back();
    } else
        for (int rank = 0; rank < 2; ++rank) {
            HIP_CHECK(hipSetDevice(impl.shards[size_t(rank)]->device));
            HIP_CHECK(hipEventCreate(&t.start[size_t(rank)]));
            HIP_CHECK(hipEventCreate(&t.done[size_t(rank)]));
        }
    for (int rank = 0; rank < 2; ++rank)
        impl.shards[size_t(rank)]->kv_issue(
            save, gpu, host, [&](int page) { return impl.kv_host_page(rank, page); },
            t.start[size_t(rank)], t.done[size_t(rank)]);
    const uint64_t id = impl.kv_next++;
    impl.kv_transfers.emplace(id, t);
    return id;
}
uint64_t TensorParallelModel::kv_save(std::span<const int> gpu_pages,
                                      std::span<const int> host_pages) {
    return kv_start(true, gpu_pages, host_pages);
}
uint64_t TensorParallelModel::kv_restore(std::span<const int> host_pages,
                                         std::span<const int> gpu_pages) {
    return kv_start(false, gpu_pages, host_pages);
}
bool TensorParallelModel::kv_poll(uint64_t id, std::array<float, 2> *gpu_ms) {
    std::lock_guard lock(impl_->kv_mutex);
    auto it = impl_->kv_transfers.find(id);
    if (it == impl_->kv_transfers.end())
        throw std::invalid_argument("Unknown KV transfer");
    auto &t = it->second;
    for (int rank = 0; rank < 2; ++rank) {
        HIP_CHECK(hipSetDevice(impl_->shards[size_t(rank)]->device));
        hipError_t e = hipEventQuery(t.done[size_t(rank)]);
        if (e == hipErrorNotReady)
            return false;
        HIP_CHECK(e);
    }
    if (gpu_ms)
        for (int rank = 0; rank < 2; ++rank) {
            HIP_CHECK(hipSetDevice(impl_->shards[size_t(rank)]->device));
            HIP_CHECK(hipEventElapsedTime(&(*gpu_ms)[size_t(rank)], t.start[size_t(rank)],
                                          t.done[size_t(rank)]));
        }
    impl_->kv_event_pool.push_back(t);
    impl_->kv_transfers.erase(it);
    return true;
}
void TensorParallelModel::kv_wait(uint64_t id) {
    std::array<hipEvent_t, 2> done{};
    {
        std::lock_guard lock(impl_->kv_mutex);
        auto it = impl_->kv_transfers.find(id);
        if (it == impl_->kv_transfers.end())
            throw std::invalid_argument("Unknown KV transfer");
        done = it->second.done;
    }
    for (int rank = 0; rank < 2; ++rank) {
        HIP_CHECK(hipSetDevice(impl_->shards[size_t(rank)]->device));
        HIP_CHECK(hipEventSynchronize(done[size_t(rank)]));
    }
}
InferenceBatch make_inference_batch(const Scheduler &scheduler, const ScheduledBatch &batch) {
    InferenceBatch input;
    auto &l = input.layout;
    l.physical_pages = scheduler.capacity();
    l.query_offsets = {0};
    l.table_stride = 1;
    for (const auto &step : batch.requests)
        l.table_stride = std::max(l.table_stride, int(ceil_div(step.begin + step.count, 16)));
    for (const auto &step : batch.requests) {
        const auto &r = scheduler.request(step.id);
        l.lengths.push_back(int(step.begin + step.count));
        for (size_t i = 0; i < step.count; ++i) {
            input.tokens.push_back(r.history[step.begin + i]);
            l.positions.push_back(int(step.begin + i));
        }
        l.query_offsets.push_back(int(input.tokens.size()));
        size_t base = l.page_table.size();
        l.page_table.resize(base + size_t(l.table_stride), -1);
        std::copy_n(r.pages.begin(), ceil_div(step.begin + step.count, 16),
                    l.page_table.begin() + std::ptrdiff_t(base));
        if (step.sample) {
            input.sample_rows.push_back(int(input.tokens.size() - 1));
            input.sample_ids.push_back(step.id);
        }
    }
    l.validate();
    return input;
}
} // namespace behemoth
