#include "behemoth/model.hpp"
#include "behemoth/gpu.hpp"
#include "behemoth/worker.hpp"
#include <atomic>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <rccl/rccl.h>
#include <sstream>
#include <thread>

namespace behemoth {
namespace {
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
class HostTransport {
    std::array<float *, 2> host_{};
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
    unsigned arrived_ = 0, generation_ = 0;
    bool aborted_ = false;
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
    void abort() {
        std::lock_guard lock(mutex_);
        aborted_ = true;
        cv_.notify_all();
    }
    void reduce(int rank, float *values, size_t n, hipStream_t stream) {
        if (n > capacity_)
            throw std::invalid_argument("Host transport capacity exceeded");
        if (zero_copy_ && n <= zero_copy_limit_ && n % 4 == 0) {
            const auto &d = device_pointers(rank);
            const unsigned seq = ++zc_seq_[size_t(rank)];
            zero_copy_publish(values, d.host[size_t(rank)][seq & 1], n, d.flag[size_t(rank)],
                              zc_counter_[size_t(rank)], seq, stream);
            zero_copy_reduce(values, d.host[size_t(1 - rank)][seq & 1], n,
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
    void gather(int rank, const float *local, float *all, size_t n, hipStream_t stream) {
        if (n > capacity_)
            throw std::invalid_argument("Host transport capacity exceeded");
        if (zero_copy_ && n <= zero_copy_limit_ && n % 4 == 0) {
            const auto &d = device_pointers(rank);
            const unsigned seq = ++zc_seq_[size_t(rank)];
            zero_copy_publish(local, d.host[size_t(rank)][seq & 1], n, d.flag[size_t(rank)],
                              zc_counter_[size_t(rank)], seq, stream);
            zero_copy_gather(local, d.host[size_t(1 - rank)][seq & 1], all, n, rank,
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
struct Layer {
    GpuProjection q, k, v, o, gate, up, down;
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
        mlp, selected;
    GpuProjection head;
    ProjectionWorkspace projection;
    DeviceBuffer<uint8_t> query;
    DeviceBuffer<float> query_scale, raw, gathered;
    DeviceBuffer<int> tokens, sample_rows;
    DeviceBuffer<std::byte> attention_workspace;
    GpuMetadata metadata;
    GpuTimes timings;
    explicit Shard(int id, int r, Checkpoint &checkpoint, const RuntimeOptions &options)
        : device(id), rank(r) {
        HIP_CHECK(hipSetDevice(device));
        verify_device();
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
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
        auto layout = parse_code_layout(options.code_layout);
        for (int i = 0; i < 88; ++i) {
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
        reserve(options.token_budget);
        size_t split_rows = (options.projection_mode == "split-k" ||
                             options.projection_mode.starts_with("tiled-") ||
                             options.projection_mode.starts_with("pipelined-") ||
                             options.projection_mode.starts_with("sorted-") ||
                             options.projection_mode.starts_with("lane-"))
                                ? options.token_budget
                                : std::min(options.token_budget, size_t(64));
        projection.partial.resize(split_rows * 16384 * 28);
    }
    ~Shard() {
        (void)hipSetDevice(device);
        if (stream)
            (void)hipStreamDestroy(stream);
    }
    void reserve(size_t rows) {
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
        mlp.resize(rows * 14336);
        query.resize(rows * 6144);
        query_scale.resize(rows * 48);
        raw.resize(rows * 16384);
        gathered.resize(rows * 32768);
        tokens.resize(rows);
        sample_rows.resize(rows);
        projection.input.resize(rows * 14336);
        projection.input_scale.resize(rows);
        projection.weight.resize(size_t(12288) * 14336);
        projection.blas.resize(64 * 1024 * 1024);
        attention_workspace.resize(256 * 1024 * 1024);
        metadata.offsets.resize(rows + 1);
        metadata.positions.resize(rows);
        metadata.lengths.resize(rows);
        metadata.table.resize(rows * 8192);
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
        timings.reset();
        metadata.upload(batch.layout);
        tokens.upload(batch.tokens);
        int rows = int(batch.tokens.size());
        if (injected)
            state.upload(*injected);
        else
            embed(embeddings.data(), tokens.data(), state.data(), rows, 12288, stream);
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
        auto row_project = [&](const GpuProjection &p, const BFloat16 *x) {
            timings.measure("projections_ms", stream, [&] {
                runner->run(p, x, raw.data(), rows, projection, stream, false);
            });
            timings.measure("collectives_ms", stream, [&] {
                if (transport)
                    transport->reduce(rank, raw.data(), size_t(rows) * 12288, stream);
                else
                    RCCL_CHECK(ncclAllReduce(raw.data(), raw.data(), size_t(rows) * 12288,
                                             ncclFloat, ncclSum, comm, stream));
            });
            timings.measure("elementwise_ms", stream, [&] {
                residual_add_scaled(state.data(), raw.data(), p.row_scale.data(), rows, 12288,
                                    stream);
            });
        };
        int partition = 0;
        bool decode = true;
        for (size_t r = 0; r < batch.layout.lengths.size(); ++r)
            decode &= batch.layout.query_offsets[r + 1] - batch.layout.query_offsets[r] == 1;
        if (decode) {
            size_t needed = attention_workspace_bytes(batch.layout, 1024);
            if (needed <= attention_workspace.size())
                partition = 1024;
        }
        for (int index = only_layer < 0 ? 0 : only_layer;
             index < (only_layer < 0 ? 88 : only_layer + 1); ++index) {
            auto &l = layers[size_t(index)];
            timings.measure("elementwise_ms", stream,
                            [&] { normed_input(state.data(), l.input_norm.data()); });
            project(l.q, q.data());
            project(l.k, k.data());
            project(l.v, v.data());
            timings.measure("attention_and_cache_ms", stream, [&] {
                rope_quantize(q.data(), k.data(), metadata.positions.data(), rows, 48, 4, 1000000,
                              query.data(), query_scale.data(), stream);
                AttentionDeviceView view{query.data(),     query_scale.data(), l.cache_k.data(),
                                         l.cache_v.data(), l.k_scale.data(),   l.v_scale.data(),
                                         attention.data()};
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
                silu_multiply(gate.data(), up.data(), mlp.data(), size_t(rows) * 14336, stream);
            });
            row_project(l.down, mlp.data());
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
    nlohmann::json profile() const {
        return {{"rank", rank},
                {"device", device},
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
    ~Impl() {
        for (auto &worker : workers)
            worker.reset();
        if (!aborted)
            for (auto c : comm)
                if (c)
                    ncclCommDestroy(c);
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
    if (options.token_budget < 1 || options.token_budget > 1024 || options.prefill_chunk < 1 ||
        options.prefill_chunk > options.token_budget)
        throw std::invalid_argument("Token budget must be 1..1024 and cover the prefill chunk");
    tensor_parallel_smoke(options.devices, options.rccl_shared_memory);
    Checkpoint checkpoint(options.model);
    for (int rank = 0; rank < 2; ++rank)
        impl_->shards[size_t(rank)] =
            std::make_unique<Shard>(options.devices[size_t(rank)], rank, checkpoint, options);
    impl_->pages = std::min(impl_->shards[0]->page_capacity(options.memory_fraction),
                            impl_->shards[1]->page_capacity(options.memory_fraction));
    for (auto &shard : impl_->shards)
        shard->allocate_cache(impl_->pages);
    if (options.rccl_shared_memory)
        RCCL_CHECK(ncclCommInitAll(impl_->comm.data(), 2, options.devices.data()));
    else
        impl_->transport = std::make_unique<HostTransport>(
            options.token_budget * 16384,
            options.zero_copy_transport ? options.zero_copy_rows * 16384 : 0, options.devices);
    for (auto &worker : impl_->workers)
        worker = std::make_unique<RankWorker>();
    std::cerr << "KV pages=" << impl_->pages << ", aggregate token capacity=" << impl_->pages * 16
              << ", bytes/token/GPU=" << PagePool::bytes_per_page() / 16 << '\n';
}
TensorParallelModel::~TensorParallelModel() = default;
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
        for (int i = first; i < last; ++i) {
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
std::vector<std::vector<float>> TensorParallelModel::forward(const InferenceBatch &batch) {
    std::lock_guard execution_lock(impl_->execution_mutex);
    const auto started = std::chrono::steady_clock::now();
    batch.layout.validate();
    if (batch.tokens.size() != size_t(batch.layout.queries()) ||
        batch.layout.physical_pages != impl_->pages)
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
    std::vector<std::vector<float>> result(n, std::vector<float>(32768));
    if (n) {
        HIP_CHECK(hipSetDevice(impl_->shards[0]->device));
        std::vector<float> logits(n * 32768);
        HIP_CHECK(hipMemcpy(logits.data(), impl_->shards[0]->gathered.data(),
                            logits.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (size_t row = 0; row < n; ++row)
            for (size_t rank = 0; rank < 2; ++rank)
                std::copy_n(logits.begin() + std::ptrdiff_t((rank * n + row) * 16384), 16384,
                            result[row].begin() + std::ptrdiff_t(rank * 16384));
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
