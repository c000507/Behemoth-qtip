#include "behemoth/model.hpp"
#include "behemoth/gpu.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/worker.hpp"
#include <atomic>
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
        const unsigned seq = ++zc_seq_[size_t(rank)];
        Exchange e;
        e.mine = d.host[size_t(rank)][seq & 1];
        e.peer = d.host[size_t(1 - rank)][seq & 1];
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
    DeviceBuffer<BFloat16> qtip_u, qtip_h;
    DeviceBuffer<float> qtip_partial, qtip_all;
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
        reserve(options.token_budget);
        // Split-K scratch: decode (rows <= 64) needs <= 64*14336*16*4 B ~ 59 MB
        // and a 256-row prefill chunk needs ~205 MB; the launch path grows the
        // buffer on demand, so one fat prefill pays a single transient realloc
        // instead of a permanent reservation for the whole token budget.
        size_t split_rows = std::min(options.token_budget, size_t(64));
        projection.partial.resize(split_rows * 16384 * 28);
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
        }
        // BEHEMOTH_ATTN_QP=bf16: unquantised BF16 queries and probabilities (K/V FP8).
        static const bool bf16_qp = [] {
            const char *v = std::getenv("BEHEMOTH_ATTN_QP");
            return v && std::string(v) == "bf16";
        }();
        for (int index = only_layer < 0 ? 0 : only_layer;
             qtip && index < (only_layer < 0 ? 88 : only_layer + 1); ++index)
            qtip_layer(layers[size_t(index)], rows, partition, bf16_qp, comm, transport);
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
    // Rank-blocked all-gather of `words` 32-bit words per rank through qtip_all: this rank's
    // half must already be at qtip_all + rank * words (or is produced there by reduce_publish when
    // partial is non-null: the sum of the matmul slices, fp32 or bf16 payload).
    void qtip_exchange(const float *partial, int slices, size_t elements, bool bf16, size_t words,
                       ncclComm_t comm, HostTransport *transport) {
        float *all = qtip_all.data();
        float *local = all + size_t(rank) * words;
        HostTransport::Exchange e{};
        if (transport && transport->zero_copy_eligible(words))
            e = transport->begin_exchange(rank, words);
        static const bool fused = [] {
            const char *v = std::getenv("BEHEMOTH_QTIP_FUSED_EXCHANGE");
            return !v || std::atoi(v) != 0;
        }();
        if (e.valid && fused) {
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
    // x . hatWr^T for this rank's rows of `w`, slices summed and all-gathered into qtip_all.
    void qtip_project(const qtip::GpuLinear &w, const qtip::GpuCodebook &lut, int rows, int ldu,
                      bool bf16, ncclComm_t comm, HostTransport *transport) {
        const int slices = qtip::matmul_slices(w, rows);
        qtip_partial.resize(size_t(slices) * size_t(rows) * size_t(w.rows));
        timings.measure("projections_ms", stream, [&] {
            qtip::matmul(w, lut.lut.data(), qtip_u.data(), ldu, rows, qtip_partial.data(), slices,
                         stream);
        });
        const size_t elements = size_t(rows) * size_t(w.rows);
        qtip_exchange(qtip_partial.data(), slices, elements, bf16, bf16 ? elements / 2 : elements,
                      comm, transport);
    }
    void qtip_layer(Layer &l, int rows, int partition, bool bf16_qp, ncclComm_t comm,
                    HostTransport *transport) {
        auto &w = l.qtip;
        const qtip::Source state_source{state.data(), 12288, 0, 12288, 0};
        // q/k/v: norm + SU + hadUt, stacked matmul, all-gather of the rotated outputs, hadU + SV
        // for this rank's heads.
        timings.measure("elementwise_ms", stream, [&] {
            qtip::input_transform(state_source, l.input_norm.data(), 1e-5f, w.su_qkv.data(), 12288,
                                  rows, qtip_u.data(), 12288, stream);
        });
        qtip_project(w.qkv, w.lut, rows, 12288, false, comm, transport);
        timings.measure("elementwise_ms", stream, [&] {
            qtip::output_qkv(qtip_all.data(), rows, rank, w.sv_q.data(), w.sv_k.data(),
                             w.sv_v.data(), q.data(), k.data(), v.data(), stream);
        });
        // Attention writes this rank's heads straight into its half of the exchange buffer.
        auto *attention_out = reinterpret_cast<BFloat16 *>(qtip_all.data()) +
                              size_t(rank) * size_t(rows) * 6144;
        timings.measure("attention_and_cache_ms", stream, [&] {
            rope_quantize(q.data(), k.data(), metadata.positions, rows, 48, 4, 1000000,
                          query.data(), query_scale.data(), stream);
            AttentionDeviceView view{query.data(),     query_scale.data(), l.cache_k.data(),
                                     l.cache_v.data(), l.k_scale.data(),   l.v_scale.data(),
                                     attention_out};
            if (bf16_qp)
                view.q_bf16 = q.data();
            append_kv(metadata, view, k.data(), v.data(), stream);
            paged_attention(metadata, view, attention_workspace.data(), attention_workspace.size(),
                            partition, stream);
        });
        // o: all-gather the head-sharded attention output, hadUt over the full 12288.
        qtip_exchange(nullptr, 0, 0, false, size_t(rows) * 3072, comm, transport);
        timings.measure("elementwise_ms", stream, [&] {
            qtip::input_transform(qtip::Source{qtip_all.data(), 6144, size_t(rows) * 3072 * 2, 6144, 0},
                                  nullptr, 0.f, w.su_o.data(), 12288, rows, qtip_u.data(), 12288,
                                  stream);
        });
        qtip_project(w.o, w.lut, rows, 12288, false, comm, transport);
        timings.measure("elementwise_ms", stream, [&] {
            qtip::output_residual(qtip_all.data(), rows, w.sv_o.data(), state.data(), stream);
        });
        // MLP: gate/up stacked, bf16 exchange, silu product, down.
        timings.measure("elementwise_ms", stream, [&] {
            qtip::input_transform(state_source, l.post_norm.data(), 1e-5f, w.su_gate_up.data(), 12288,
                                  rows, qtip_u.data(), 12288, stream);
        });
        qtip_project(w.gate_up, w.lut, rows, 12288, true, comm, transport);
        timings.measure("elementwise_ms", stream, [&] {
            qtip::output_gateup(reinterpret_cast<const BFloat16 *>(qtip_all.data()), rows,
                                w.sv_gate.data(), w.sv_up.data(), qtip_h.data(), stream);
            qtip::input_transform(qtip::Source{qtip_h.data(), 28672, 0, 28672, 0}, nullptr, 0.f,
                                  w.su_down.data(), 28672, rows, qtip_u.data(), 28672, stream);
        });
        qtip_project(w.down, w.lut, rows, 28672, false, comm, transport);
        timings.measure("elementwise_ms", stream, [&] {
            qtip::output_residual(qtip_all.data(), rows, w.sv_down.data(), state.data(), stream);
        });
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
    // Persistent pinned logits staging (PERF_ROUND3 item 2): one portable
    // pinned buffer sized for the token budget replaces the per-step pageable
    // vector, blocking pageable D2H and rank-interleave copy.
    float *logits_host = nullptr;
    size_t logits_capacity = 0;
    ~Impl() {
        for (auto &worker : workers)
            worker.reset();
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
    if (options.token_budget < 1 || options.token_budget > 1024 || options.prefill_chunk < 1 ||
        options.prefill_chunk > options.token_budget)
        throw std::invalid_argument("Token budget must be 1..1024 and cover the prefill chunk");
    tensor_parallel_smoke(options.devices, options.rccl_shared_memory);
    Checkpoint checkpoint(options.model);
    for (int rank = 0; rank < 2; ++rank)
        impl_->shards[size_t(rank)] =
            std::make_unique<Shard>(options.devices[size_t(rank)], rank, checkpoint, options);
    if (options.rccl_shared_memory)
        RCCL_CHECK(ncclCommInitAll(impl_->comm.data(), 2, options.devices.data()));
    else
        impl_->transport = std::make_unique<HostTransport>(
            options.token_budget * 16384,
            options.zero_copy_transport ? options.zero_copy_rows * 16384 : 0, options.devices);
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
    impl_->pages = std::min(impl_->shards[0]->page_capacity(options.memory_fraction),
                            impl_->shards[1]->page_capacity(options.memory_fraction));
    for (auto &shard : impl_->shards)
        shard->allocate_cache(impl_->pages);
    impl_->logits_capacity = options.token_budget * 32768;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&impl_->logits_host),
                            impl_->logits_capacity * sizeof(float), hipHostMallocPortable));
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
