#include "behemoth/qtip_gpu.hpp"
#include "hip_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <map>
#include <mutex>
#include <tuple>

namespace qtip_training {
void had_rows(const void *, int, long, int, int, int, const float *, int, float *, long, uint16_t *, long,
              const float *, hipStream_t);
}
namespace qat {
namespace {
void blas(hipblasStatus_t s) {
    require(s == HIPBLAS_STATUS_SUCCESS, "hipBLASLt status " + std::to_string(int(s)));
}
struct Arena : std::enable_shared_from_this<Arena> {
    int device;
    void *base = nullptr;
    size_t bytes, used = 0, peak = 0;
    std::map<size_t, size_t> free;
    Arena(int d, size_t n) : device(d), bytes(n) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMalloc(&base, n));
        free[0] = n;
    }
    ~Arena() {
        (void)hipSetDevice(device);
        (void)hipFree(base);
    }
    std::shared_ptr<void> alloc(size_t bytes) {
        bytes = (bytes + 255) / 256 * 256;
        auto best = free.end();
        for (auto it = free.begin(); it != free.end(); ++it)
            if (it->second >= bytes && (best == free.end() || it->second < best->second))
                best = it;
        require(best != free.end(), "GPU arena exhausted: requested " + std::to_string(bytes) + ", active " +
                                        std::to_string(used) + ", budget " + std::to_string(this->bytes));
        size_t off = best->first, n = best->second;
        free.erase(best);
        if (n > bytes)
            free[off + bytes] = n - bytes;
        used += bytes;
        peak = std::max(peak, used);
        return {static_cast<char *>(base) + off, [self = shared_from_this(), off, bytes](void *) {
                    self->used -= bytes;
                    size_t start = off, n = bytes;
                    auto next = self->free.lower_bound(off);
                    if (next != self->free.end() && off + bytes == next->first) {
                        n += next->second;
                        self->free.erase(next);
                    }
                    next = self->free.lower_bound(off);
                    if (next != self->free.begin()) {
                        auto prev = std::prev(next);
                        if (prev->first + prev->second == off) {
                            start = prev->first;
                            n += prev->second;
                            self->free.erase(prev);
                        }
                    }
                    self->free[start] = n;
                }};
    }
};
struct Pinned {
    void *p = nullptr;
    size_t bytes = 0;
    explicit Pinned(size_t n) : bytes(n) { HIP_CHECK(hipHostMalloc(&p, n, hipHostMallocPortable)); }
    ~Pinned() { (void)hipHostFree(p); }
};
// Register shared immutable packed weights once, including across GPU replicas.
struct Registration {
    void *p;
    explicit Registration(const void *q, size_t n) : p(const_cast<void *>(q)) {
        HIP_CHECK(hipHostRegister(p, n, hipHostRegisterPortable));
    }
    ~Registration() { (void)hipHostUnregister(p); }
};
std::mutex registry_mutex;
std::map<const void *, std::weak_ptr<Registration>> registry;
std::shared_ptr<Registration> pin(const void *p, size_t n) {
    std::lock_guard lock(registry_mutex);
    auto r = registry[p].lock();
    if (!r) {
        r = std::make_shared<Registration>(p, n);
        registry[p] = r;
    }
    return r;
}
struct Plan {
    hipblasLtMatmulDesc_t op{};
    hipblasLtMatrixLayout_t a{}, b{}, c{};
    hipblasLtMatmulAlgo_t algo{};
    ~Plan() {
        if (a)
            hipblasLtMatrixLayoutDestroy(a);
        if (b)
            hipblasLtMatrixLayoutDestroy(b);
        if (c)
            hipblasLtMatrixLayoutDestroy(c);
        if (op)
            hipblasLtMatmulDescDestroy(op);
    }
};
struct Slot {
    std::shared_ptr<void> codes, cp;
    std::vector<size_t> offsets;
    hipEvent_t ready{}, done{}, offloaded{};
    int layer = -1;
};
struct Hip final : Ops {
    int device;
    std::shared_ptr<Arena> arena;
    hipStream_t stream{}, copy_stream{}, offload_stream{};
    hipblasLtHandle_t handle{};
    std::shared_ptr<void> workspace;
    size_t ws = size_t(64) << 20;
    std::map<std::tuple<int, int, int, int, bool>, std::unique_ptr<Plan>> plans;
    const Model *model = nullptr;
    int seqlen = 0, active_slot = -1;
    Slot slots[2];
    std::vector<std::unique_ptr<Pinned>> checkpoints;
    std::vector<std::shared_ptr<Registration>> registrations;
    std::unique_ptr<Pinned> stage;
    std::shared_ptr<void> ids, positions, starts, labels;
    size_t batch_capacity = 0;
    explicit Hip(int d, size_t bytes) : device(d) {
        rounding = true;
        HIP_CHECK(hipSetDevice(d));
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, d));
        require(std::string(prop.gcnArchName).starts_with("gfx1201"), "native trainer targets gfx1201");
        require(bytes > size_t(256) << 20, "arena too small");
        arena = std::make_shared<Arena>(d, bytes);
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        HIP_CHECK(hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking));
        HIP_CHECK(hipStreamCreateWithFlags(&offload_stream, hipStreamNonBlocking));
        blas(hipblasLtCreate(&handle));
        workspace = arena->alloc(ws);
        stage = std::make_unique<Pinned>(size_t(8) << 20);
        for (auto &s : slots) {
            HIP_CHECK(hipEventCreateWithFlags(&s.ready, hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&s.done, hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&s.offloaded, hipEventDisableTiming));
            HIP_CHECK(hipEventRecord(s.done, stream));
            HIP_CHECK(hipEventRecord(s.offloaded, offload_stream));
        }
    }
    ~Hip() {
        (void)hipSetDevice(device);
        (void)hipDeviceSynchronize();
        for (auto &s : slots) {
            (void)hipEventDestroy(s.ready);
            (void)hipEventDestroy(s.done);
            (void)hipEventDestroy(s.offloaded);
        }
        (void)hipblasLtDestroy(handle);
        (void)hipStreamDestroy(stream);
        (void)hipStreamDestroy(copy_stream);
        (void)hipStreamDestroy(offload_stream);
    }
    void enter() { HIP_CHECK(hipSetDevice(device)); }
    Mat alloc(int r, int c, bool clear = false) override {
        enter();
        require(r >= 0 && c > 0, "allocation shape");
        auto p = arena->alloc(std::max(size_t(1), size_t(r) * c) * 4);
        Mat m{p, static_cast<float *>(p.get()), r, c, c};
        if (clear)
            zero(m);
        return m;
    }
    void copy(Mat d, Mat s) override {
        require(d.rows == s.rows && d.cols == s.cols, "copy shape");
        HIP_CHECK(hipMemcpy2DAsync(d.p, d.ld * 4, s.p, s.ld * 4, d.cols * 4, d.rows, hipMemcpyDeviceToDevice,
                                   stream));
    }
    void upload(Mat d, std::span<const float> v) override {
        enter();
        require(d.size() == v.size(), "upload shape");
        for (int r = 0; r < d.rows; ++r) {
            size_t off = 0;
            while (off < size_t(d.cols)) {
                size_t take = std::min(size_t(d.cols) - off, stage->bytes / 4);
                HIP_CHECK(hipStreamSynchronize(stream));
                std::memcpy(stage->p, v.data() + size_t(r) * d.cols + off, take * 4);
                HIP_CHECK(hipMemcpyAsync(d.p + size_t(r) * d.ld + off, stage->p, take * 4,
                                         hipMemcpyHostToDevice, stream));
                off += take;
            }
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    void download(Mat s, std::span<float> v) override {
        enter();
        require(s.size() == v.size(), "download shape");
        for (int r = 0; r < s.rows; ++r) {
            size_t off = 0;
            while (off < size_t(s.cols)) {
                size_t take = std::min(size_t(s.cols) - off, stage->bytes / 4);
                HIP_CHECK(hipMemcpyAsync(stage->p, s.p + size_t(r) * s.ld + off, take * 4,
                                         hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                std::memcpy(v.data() + size_t(r) * s.cols + off, stage->p, take * 4);
                off += take;
            }
        }
    }
    void zero(Mat d) override { HIP_CHECK(hipMemset2DAsync(d.p, d.ld * 4, 0, d.cols * 4, d.rows, stream)); }
    void add(Mat d, Mat a, Mat b, bool round = false) override {
        gpu::binary(gpu::view(d), gpu::view(a), gpu::view(b), 0, round, stream);
    }
    void mul(Mat d, Mat a, Mat b) override {
        gpu::binary(gpu::view(d), gpu::view(a), gpu::view(b), 1, false, stream);
    }
    void round(Mat d) override { gpu::rounded(gpu::view(d), stream); }
    void colsum(Mat a, Mat b, Mat d) override {
        gpu::reduce(gpu::view(a), gpu::view(b), gpu::view(d), false, stream);
    }
    void scale_sum(Mat a, Mat b, Mat d) override {
        gpu::reduce(gpu::view(a), gpu::view(b), gpu::view(d), true, stream);
    }
    Mat had(Mat x, bool tr = false, Mat pre = {}) override {
        auto y = alloc(x.rows, x.cols);
        qtip_training::had_rows(x.p, 0, x.ld, x.rows, x.cols, tr, pre.p, 0, y.p, y.ld, nullptr, 0, nullptr,
                                stream);
        return y;
    }
    Norm rms(Mat x, Mat w, float eps) override {
        Norm n{alloc(x.rows, x.cols), alloc(x.rows, 1)};
        gpu::rms(gpu::view(x), gpu::view(w), gpu::view(n.y), gpu::view(n.r), eps, stream);
        return n;
    }
    Mat rms_backward(Mat x, Norm n, Mat w, Mat dy, Mat dw) override {
        auto dx = alloc(x.rows, x.cols);
        gpu::rms_back(gpu::view(x), gpu::view(n.r), gpu::view(w), gpu::view(dy), gpu::view(dx), gpu::view(dw),
                      stream);
        return dx;
    }
    Weight frozen(std::span<const uint16_t> v, int out, int in) override {
        auto p = arena->alloc(v.size_bytes());
        auto reg = pin(v.data(), v.size_bytes());
        HIP_CHECK(hipMemcpyAsync(p.get(), v.data(), v.size_bytes(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return {p, p.get(), out, in, true, false};
    }
    std::vector<Weight> decode(const LayerData &layer) override {
        require(active_slot >= 0, "decode outside acquired layer slot");
        auto &s = slots[active_slot];
        auto cb = behemoth::qtip::make_codebook(layer.linear[0].tlut);
        auto lut = arena->alloc(sizeof(cb.bf16x2));
        HIP_CHECK(hipStreamSynchronize(stream));
        std::memcpy(stage->p, cb.bf16x2.data(), sizeof(cb.bf16x2));
        HIP_CHECK(hipMemcpyAsync(lut.get(), stage->p, sizeof(cb.bf16x2), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<Weight> result;
        for (size_t i = 0; i < layer.linear.size(); ++i) {
            auto &w = layer.linear[i];
            auto p = arena->alloc(size_t(w.out) * w.in * 2);
            behemoth::qtip::decode_dense(
                {static_cast<const uint16_t *>(s.codes.get()) + s.offsets[i], w.in, w.out},
                static_cast<const uint32_t *>(lut.get()), static_cast<behemoth::BFloat16 *>(p.get()), stream);
            result.push_back({p, p.get(), w.out, w.in, true, false});
        }
        return result;
    }
    Plan &plan(int m, int n, int k, int win, bool transpose) {
        auto key = std::make_tuple(m, n, k, win, transpose);
        auto it = plans.find(key);
        if (it != plans.end())
            return *it->second;
        auto p = std::make_unique<Plan>();
        blas(hipblasLtMatmulDescCreate(&p->op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t tr = transpose ? HIPBLAS_OP_N : HIPBLAS_OP_T;
        blas(hipblasLtMatmulDescSetAttribute(p->op, HIPBLASLT_MATMUL_DESC_TRANSA, &tr, sizeof(tr)));
        blas(hipblasLtMatrixLayoutCreate(&p->a, HIP_R_16BF, transpose ? m : k, transpose ? k : m, win));
        blas(hipblasLtMatrixLayoutCreate(&p->b, HIP_R_16BF, k, n, k));
        blas(hipblasLtMatrixLayoutCreate(&p->c, HIP_R_32F, m, n, m));
        hipblasLtMatmulPreference_t pref{};
        blas(hipblasLtMatmulPreferenceCreate(&pref));
        blas(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws,
                                                   sizeof(ws)));
        hipblasLtMatmulHeuristicResult_t result{};
        int count = 0;
        auto status =
            hipblasLtMatmulAlgoGetHeuristic(handle, p->op, p->a, p->b, p->c, p->c, pref, 1, &result, &count);
        hipblasLtMatmulPreferenceDestroy(pref);
        blas(status);
        require(count > 0 && result.state == HIPBLAS_STATUS_SUCCESS, "no BF16 GEMM algorithm");
        p->algo = result.algo;
        auto &ref = *p;
        plans.emplace(key, std::move(p));
        return ref;
    }
    Mat mm(Mat x, const Weight &w, bool tr = false) override {
        int k = tr ? w.out : w.in, m = tr ? w.in : w.out;
        require(x.cols == k, "GEMM shape");
        auto y = alloc(x.rows, m);
        auto input = arena->alloc(x.size() * 2);
        gpu::cast(gpu::view(x), static_cast<uint16_t *>(input.get()), true, stream);
        auto &p = plan(m, x.rows, k, w.in, tr);
        float one = 1, zero = 0;
        blas(hipblasLtMatmul(handle, p.op, &one, w.p, p.a, input.get(), p.b, &zero, y.p, p.c, y.p, p.c,
                             &p.algo, workspace.get(), ws, stream));
        return y;
    }
    void upload_int(std::shared_ptr<void> &p, std::span<const int32_t> v) {
        require(v.size_bytes() <= stage->bytes, "integer staging buffer overflow");
        HIP_CHECK(hipStreamSynchronize(stream));
        std::memcpy(stage->p, v.data(), v.size_bytes());
        HIP_CHECK(hipMemcpyAsync(p.get(), stage->p, v.size_bytes(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    Mat embedding(const Weight &w, const Batch &b) override {
        for (int id : b.tokens)
            require(id >= 0 && id < w.out, "token outside vocabulary");
        size_t n = b.tokens.size();
        if (n > batch_capacity) {
            ids = arena->alloc(n * 4);
            positions = arena->alloc(n * 4);
            starts = arena->alloc(n * 4);
            labels = arena->alloc(n * 4);
            batch_capacity = n;
        }
        upload_int(ids, b.tokens);
        upload_int(positions, b.positions);
        upload_int(starts, b.starts);
        auto x = alloc(int(n), w.in);
        gpu::embedding(gpu::view(x), static_cast<const uint16_t *>(w.p),
                       static_cast<const int32_t *>(ids.get()), w.out, stream);
        return x;
    }
    Mat rope(Mat x, int h, const Config &c, const Batch &, bool back = false) override {
        auto y = alloc(x.rows, x.cols);
        gpu::rope(gpu::view(x), gpu::view(y), h, c.dim, c.theta,
                  static_cast<const int32_t *>(positions.get()), back, stream);
        return y;
    }
    Mat swiglu(Mat g, Mat u) override {
        auto y = alloc(g.rows, g.cols);
        gpu::swiglu(gpu::view(g), gpu::view(u), gpu::view(y), stream);
        return y;
    }
    std::pair<Mat, Mat> swiglu_backward(Mat g, Mat u, Mat dy) override {
        auto dg = alloc(g.rows, g.cols), du = alloc(g.rows, g.cols);
        gpu::swiglu_back(gpu::view(g), gpu::view(u), gpu::view(dy), gpu::view(dg), gpu::view(du), stream);
        return {dg, du};
    }
    Attention attention(Mat q, Mat k, Mat v, const Config &c, const Batch &) override {
        require(c.dim == 128, "HIP attention requires head_dim=128");
        Attention a{q, k, v, alloc(q.rows, q.cols), alloc(c.heads, q.rows)};
        gpu::flash(gpu::view(q), gpu::view(k), gpu::view(v), gpu::view(a.out), gpu::view(a.lse), c.heads,
                   c.kv, static_cast<const int32_t *>(starts.get()), stream);
        return a;
    }
    std::vector<Mat> attention_backward(const Attention &a, Mat dy, const Config &c, const Batch &) override {
        auto effective_dy = alloc(dy.rows, dy.cols);
        copy(effective_dy, dy);
        round(effective_dy);
        auto dq = alloc(a.q.rows, a.q.cols), dk = alloc(a.k.rows, a.k.cols, true),
             dv = alloc(a.v.rows, a.v.cols, true);
        gpu::flash_back(gpu::view(a.q), gpu::view(a.k), gpu::view(a.v), gpu::view(a.out), gpu::view(a.lse),
                        gpu::view(effective_dy), gpu::view(dq), gpu::view(dk), gpu::view(dv), c.heads, c.kv,
                        static_cast<const int32_t *>(starts.get()), stream);
        round(dq);
        round(dk);
        round(dv);
        return {dq, dk, dv};
    }
    double cross_entropy(Mat x, std::span<const int32_t> l, float den, bool back) override {
        for (int label : l)
            require(label == -100 || (label >= 0 && label < x.cols), "target outside vocabulary");
        upload_int(labels, l);
        auto loss = alloc(x.rows, 1);
        gpu::loss(gpu::view(x), static_cast<const int32_t *>(labels.get()), loss.p, den, back, stream);
        std::vector<float> v(x.rows);
        download(loss, v);
        double total = 0;
        for (float f : v) {
            require(std::isfinite(f), "nonfinite loss");
            total += f;
        }
        return total;
    }
    void adam(Mat p, Mat g, Mat m, Mat v, const std::vector<Param> &params, uint64_t step, float lr,
              float nlr, float decay, float clip) override {
        enter();
        require(clip == 0, "gradient clipping must precede native Adam");
        for (auto &a : params)
            gpu::adam(p.p + a.offset, g.p + a.offset, m.p + a.offset, v.p + a.offset, a.size, a.norm, step,
                      a.norm ? nlr : lr, decay, stream);
    }
    void synchronize() override {
        enter();
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipGetLastError());
    }
    void configure_streaming(const Model &m, int length) override {
        enter();
        require(m.config.dim == 128, "HIP backend requires head_dim=128");
        for (auto &layer : m.layers)
            for (auto &w : layer.linear)
                for (int n : {w.in, w.out})
                    require(n == 1024 || n == 12288 || n == 28672, "unsupported HIP Hadamard dimension");
        model = &m;
        seqlen = length;
        size_t packed = 0;
        for (auto &layer : m.layers) {
            size_t bytes = 0;
            for (auto &w : layer.linear) {
                registrations.push_back(pin(w.trellis.data(), w.trellis.size() * 2));
                bytes += w.trellis.size() * 2;
            }
            packed = std::max(packed, bytes);
        }
        for (auto &s : slots) {
            s.codes = arena->alloc(packed);
            s.cp = arena->alloc(size_t(length) * m.config.n * 2);
        }
        for (int l = 0; l < m.config.layers; ++l)
            checkpoints.push_back(std::make_unique<Pinned>(size_t(length) * m.config.n * 2));
        std::cerr << "GPU " << device << ": pinned " << size_t(length) * m.config.n * 2 * m.config.layers
                  << " activation checkpoint bytes\n";
    }
    void prefetch(int l, int slot, bool back) override {
        auto &s = slots[slot];
        HIP_CHECK(hipStreamWaitEvent(copy_stream, s.done, 0));
        s.offsets.clear();
        size_t offset = 0;
        for (auto &w : model->layers[l].linear) {
            s.offsets.push_back(offset);
            HIP_CHECK(hipMemcpyAsync(static_cast<uint16_t *>(s.codes.get()) + offset, w.trellis.data(),
                                     w.trellis.size() * 2, hipMemcpyHostToDevice, copy_stream));
            offset += w.trellis.size();
        }
        if (back) {
            HIP_CHECK(hipStreamWaitEvent(copy_stream, s.offloaded, 0));
            HIP_CHECK(hipMemcpyAsync(s.cp.get(), checkpoints[l]->p, checkpoints[l]->bytes,
                                     hipMemcpyHostToDevice, copy_stream));
        }
        HIP_CHECK(hipEventRecord(s.ready, copy_stream));
        s.layer = l;
    }
    void acquire(int slot) override {
        active_slot = slot;
        HIP_CHECK(hipStreamWaitEvent(stream, slots[slot].ready, 0));
    }
    void release(int slot) override {
        HIP_CHECK(hipEventRecord(slots[slot].done, stream));
        active_slot = -1;
    }
    Mat checkpoint(int l, Mat x, bool save) override {
        auto &s = slots[l % 2];
        if (save) {
            HIP_CHECK(hipStreamWaitEvent(stream, s.offloaded, 0));
            gpu::cast(gpu::view(x), static_cast<uint16_t *>(s.cp.get()), true, stream);
            HIP_CHECK(hipEventRecord(s.done, stream));
            HIP_CHECK(hipStreamWaitEvent(offload_stream, s.done, 0));
            HIP_CHECK(hipMemcpyAsync(checkpoints[l]->p, s.cp.get(), checkpoints[l]->bytes,
                                     hipMemcpyDeviceToHost, offload_stream));
            HIP_CHECK(hipEventRecord(s.offloaded, offload_stream));
            return {};
        }
        auto y = alloc(seqlen, model->config.n);
        gpu::cast(gpu::view(y), static_cast<uint16_t *>(s.cp.get()), false, stream);
        return y;
    }
    size_t peak_bytes() const override { return arena->peak; }
};
} // namespace
std::unique_ptr<Ops> hip_ops(int device, size_t arena) { return std::make_unique<Hip>(device, arena); }
} // namespace qat
