#include "device_ops.hpp"
#include "glm53.hpp"
#if defined(GLM53_NCCL)
#include <nccl.h>
#endif
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
namespace glm53 {
namespace {
using namespace gpu;
// Part of a linear weight as one tensor-parallel rank loads it: everything, a range of rows (output
// channels, column-parallel layers) or a range of columns (input channels, row-parallel layers).
struct Slice {
    int dim = -1, begin = 0, count = 0;
    static Slice rows(int b, int n) { return {0, b, n}; }
    static Slice cols(int b, int n) { return {1, b, n}; }
};
// A linear weight kept on its device for the model's lifetime (within the residency budget).
struct Held {
    Buffer<uint8_t> raw, block_scales;
    Buffer<float> scales;
    Matrix matrix{};
};
struct Resident {
    Vec host;
    Buffer<float> data;
};
struct Context {
    int device;
    int rank = 0, world = 1; // position in the tensor-parallel group (0 of 1 outside tensor parallelism)
    Stream stream{};
    Event done{};
    // Shared staging for streamed (non-resident) matrices.
    Buffer<uint8_t> raw, block_scales;
    Buffer<float> scales;
    Pinned<float> readback;
    Pinned<int> read_indices;
    std::map<std::string, Tensor> host_scales;
    std::map<std::string, Held> resident;
    std::map<std::string, Resident> constants;
    size_t resident_budget = 0, resident_used = 0, streamed_bytes = 0;
    int streamed_count = 0;
    explicit Context(int d) : device(d) {
        checked(set_device(d), "select GPU");
#if defined(GLM53_CUDA)
        cudaDeviceProp prop{};
        checked(cudaGetDeviceProperties(&prop, d), "CUDA properties");
        check(prop.major >= 7 && prop.warpSize == 32, "CUDA inference requires sm_70 or newer");
#else
        hipDeviceProp_t prop{};
        checked(hipGetDeviceProperties(&prop, d), "HIP properties");
        check(prop.warpSize == 32, "HIP inference requires wave32 (R9700/gfx1201)");
#endif
        checked(stream_create(&stream), "create GPU stream");
        if (event_create(&done) != success) {
            (void)stream_destroy(stream);
            throw std::runtime_error("create GPU event");
        }
    }
    ~Context() {
        (void)set_device(device);
        (void)sync(stream);
        resident.clear();
        constants.clear();
        raw.clear();
        block_scales.clear();
        scales.clear();
        (void)event_destroy(done);
        (void)stream_destroy(stream);
    }
    void activate() { checked(set_device(device), "select GPU"); }
    size_t free_memory() {
        activate();
        size_t free = 0, total = 0;
        checked(memory_info(&free, &total), "GPU memory info");
        return free;
    }
    // An FP32 constant, or the elements [begin, begin + count) of it when count > 0 (a rank's heads).
    const float *constant(const Weights &w, const std::string &name, size_t size, size_t begin = 0,
                          size_t count = 0) {
        if (!count)
            count = size;
        if (!constants.contains(name)) {
            auto t = w.tensor(name);
            check(t.data.size() == size, "GPU constant geometry mismatch: " + name);
            check(begin + count <= size, "GPU constant slice outside tensor: " + name);
            Resident r;
            r.host.assign(t.data.begin() + begin, t.data.begin() + begin + count);
            r.data.ensure(count, stream);
            auto it = constants.emplace(name, std::move(r)).first;
            const auto &a = it->second;
            checked(h2d(a.data.p, a.host.data(), count * 4, stream), "upload constant");
        }
        check(constants.at(name).host.size() == count, "GPU constant size changed");
        return constants.at(name).data;
    }
    // Uploads a linear weight or the slice of it this rank owns. `keep` requests residency: the matrix stays
    // on the device when the budget allows, otherwise it streams through the shared staging buffers.
    Matrix matrix(const Weights &w, const std::string &base, bool keep = true, Slice slice = {}) {
        if (keep)
            if (auto it = resident.find(base); it != resident.end())
                return it->second.matrix;
        bool nv = w.has(base + ".weight_packed");
        auto name = nv ? base + ".weight_packed" : w.has(base) ? base : base + ".weight";
        const auto &f = w.file(name);
        auto shape = f.shape(name);
        check(shape.size() == 2 && shape[0] > 0 && shape[0] <= INT32_MAX && shape[1] > 0 &&
                  shape[1] <= INT32_MAX / 2,
              "GPU matrix geometry outside supported range: " + name);
        int r = int(shape[0]), c = int(shape[1]) * (nv ? 2 : 1);
        int r0 = 0, rl = r, c0 = 0, cl = c;
        if (slice.dim == 0) {
            check(slice.begin >= 0 && slice.count > 0 && slice.begin + slice.count <= r,
                  "GPU row slice outside matrix: " + name);
            r0 = slice.begin;
            rl = slice.count;
        } else if (slice.dim == 1) {
            check(slice.begin >= 0 && slice.count > 0 && slice.begin + slice.count <= c,
                  "GPU column slice outside matrix: " + name);
            c0 = slice.begin;
            cl = slice.count;
        }
        auto bytes = f.bytes(name);
        auto dtype = f.dtype(name);
        Matrix m{nullptr, nullptr, nullptr, rl, cl, 0, false, 1};
        // Bytes per element as a fraction (NVFP4 packs two per byte).
        size_t numerator = 4, denominator = 1;
        std::span<const std::byte> side; // NVFP4 block scales, one byte per 16 columns
        Vec fp8_scales;                  // the slice's FP8 scales (per row, or per 128x128 block)
        if (nv) {
            check(dtype == "U8" && c % 16 == 0 && !w.has(base + ".weight"), "invalid NVFP4 matrix");
            check(c0 % 16 == 0 && cl % 16 == 0, "GPU NVFP4 column slices must align to 16-column blocks: " + base);
            m.format = 2;
            numerator = 1;
            denominator = 2;
            auto sn = base + ".weight_scale";
            const auto &sf = w.file(sn);
            check(sf.dtype(sn) == "F8_E4M3" && sf.shape(sn) == std::vector<size_t>{size_t(r), size_t(c / 16)},
                  "invalid NVFP4 block-scale geometry");
            side = sf.bytes(sn);
            auto gs = w.tensor(base + ".weight_global_scale");
            check(gs.data.size() == 1 && gs.data[0] > 0, "invalid NVFP4 global scale");
            m.global = gs.data[0];
        } else if (dtype == "F8_E4M3") {
            m.format = 1;
            numerator = 1;
            m.block = w.has(base + ".weight_scale_inv");
            auto sn = base + (m.block ? ".weight_scale_inv" : ".weight_scale");
            if (!host_scales.contains(sn))
                host_scales.emplace(sn, w.tensor(sn));
            const Tensor &full = host_scales.at(sn);
            if (m.block) {
                int grid_cols = (c + 127) / 128;
                check(full.shape == std::vector<size_t>{size_t((r + 127) / 128), size_t(grid_cols)},
                      "invalid FP8 scale geometry");
                check(r0 % 128 == 0 && c0 % 128 == 0 && (rl % 128 == 0 || r0 + rl == r) &&
                          (cl % 128 == 0 || c0 + cl == c),
                      "GPU FP8 block-scaled slices must align to 128-row/column blocks: " + base);
                int lr = (rl + 127) / 128, lc = (cl + 127) / 128;
                fp8_scales.resize(size_t(lr) * lc);
                for (int i = 0; i < lr; ++i)
                    for (int j = 0; j < lc; ++j)
                        fp8_scales[size_t(i) * lc + j] = full.data[size_t(r0 / 128 + i) * grid_cols + c0 / 128 + j];
            } else {
                check(full.data.size() == size_t(r), "invalid FP8 scale geometry");
                fp8_scales.assign(full.data.begin() + r0, full.data.begin() + r0 + rl);
            }
            for (float a : fp8_scales)
                check(a >= 0 && std::isfinite(a), "invalid FP8 scale");
        } else if (dtype == "BF16") {
            m.format = 4;
            numerator = 2;
        } else if (dtype == "F16") {
            m.format = 3;
            numerator = 2;
        } else
            check(dtype == "F32", "unsupported GPU linear dtype: " + dtype);
        auto row_bytes = [&](int cols) { return size_t(cols) * numerator / denominator; };
        size_t full_row = row_bytes(c), local_row = row_bytes(cl), offset = size_t(r0) * full_row + row_bytes(c0);
        check(bytes.size() == size_t(r) * full_row, "GPU matrix byte size mismatch: " + name);
        size_t side_full = nv ? size_t(c / 16) : 0, side_local = nv ? size_t(cl / 16) : 0;
        size_t total = size_t(rl) * local_row + size_t(rl) * side_local + fp8_scales.size() * 4;
        Held held;
        bool pin = keep && resident_used + total <= resident_budget;
        Buffer<uint8_t> &raw_target = pin ? held.raw : raw, &side_target = pin ? held.block_scales : block_scales;
        Buffer<float> &scale_target = pin ? held.scales : scales;
        // Host sources are pageable (mmap or temporaries): these copies complete before returning.
        raw_target.ensure(size_t(rl) * local_row, stream);
        if (slice.dim == 1)
            checked(h2d_2d(raw_target, local_row, bytes.data() + offset, full_row, local_row, rl, stream),
                    "upload matrix column slice");
        else
            checked(h2d(raw_target, bytes.data() + offset, size_t(rl) * local_row, stream), "upload matrix");
        m.data = raw_target;
        if (nv) {
            side_target.ensure(size_t(rl) * side_local, stream);
            const std::byte *src = side.data() + size_t(r0) * side_full + c0 / 16;
            if (slice.dim == 1)
                checked(h2d_2d(side_target, side_local, src, side_full, side_local, rl, stream),
                        "upload NVFP4 scale slice");
            else
                checked(h2d(side_target, src, size_t(rl) * side_local, stream), "upload NVFP4 scales");
            m.block_scales = side_target;
        }
        if (!fp8_scales.empty()) {
            scale_target.ensure(fp8_scales.size(), stream);
            checked(h2d(scale_target, fp8_scales.data(), fp8_scales.size() * 4, stream), "upload FP8 scales");
            m.scales = scale_target;
        }
        if (pin) {
            held.matrix = m;
            resident.emplace(base, std::move(held));
            resident_used += total;
        } else if (keep) {
            // One-shot uploads (keep == false, e.g. the decoded MLA expansion) are not per-token traffic.
            ++streamed_count;
            streamed_bytes += total;
        }
        return m;
    }
    int linear(const Weights &w, const std::string &base, const B16 *x, size_t width, Buffer<B16> &out,
               Slice slice = {}) {
        auto m = matrix(w, base, true, slice);
        check(size_t(m.cols) == width, "GPU linear input width mismatch: " + base);
        out.ensure(m.rows, stream);
        gemv(m, x, out, false, stream);
        return m.rows;
    }
    // FP32 output into caller-sized storage: the partial sums of row-parallel projections, reduced across
    // ranks before rounding.
    int linear32(const Weights &w, const std::string &base, const B16 *x, size_t width, float *out,
                 Slice slice = {}) {
        auto m = matrix(w, base, true, slice);
        check(size_t(m.cols) == width, "GPU linear input width mismatch: " + base);
        gemv(m, x, out, true, stream);
        return m.rows;
    }
};
// Sums the FP32 partial vectors of a tensor-parallel group into `out` (FP32, bitwise identical on every rank).
struct Collective {
    virtual ~Collective() = default;
    virtual void all_reduce(const std::vector<Context *> &ranks, const std::vector<const float *> &in,
                            const std::vector<float *> &out, int n) = 0;
    virtual const char *name() const = 0;
};
// Copy-engine collective: every rank gathers all partials into its own slots (peer copies, or host-staged
// copies where peer access is unavailable; plain device copies when ranks share one GPU) and sums them in a
// fixed order. Works for any rank count and needs no library.
class CopyCollective final : public Collective {
    std::vector<Buffer<float>> slots;
    std::vector<Event> ready, done;

  public:
    explicit CopyCollective(const std::vector<Context *> &ranks)
        : slots(ranks.size()), ready(ranks.size()), done(ranks.size()) {
        for (size_t r = 0; r < ranks.size(); ++r) {
            ranks[r]->activate();
            checked(event_create(&ready[r]), "create collective event");
            checked(event_create(&done[r]), "create collective event");
            for (size_t q = 0; q < ranks.size(); ++q)
                if (q != r)
                    (void)enable_peer(ranks[q]->device);
        }
    }
    ~CopyCollective() override {
        for (size_t r = 0; r < slots.size(); ++r) {
            slots[r].clear();
            if (ready[r])
                (void)event_destroy(ready[r]);
            if (done[r])
                (void)event_destroy(done[r]);
        }
    }
    const char *name() const override { return "copy"; }
    void all_reduce(const std::vector<Context *> &ranks, const std::vector<const float *> &in,
                    const std::vector<float *> &out, int n) override {
        size_t world = ranks.size();
        for (size_t r = 0; r < world; ++r) {
            ranks[r]->activate();
            checked(event_record(ready[r], ranks[r]->stream), "record partial sums ready");
        }
        for (size_t r = 0; r < world; ++r) {
            Context &x = *ranks[r];
            x.activate();
            slots[r].ensure(world * n, x.stream);
            for (size_t q = 0; q < world; ++q) {
                if (q != r)
                    checked(stream_wait(x.stream, ready[q]), "wait for peer partial sums");
                checked(copy_peer(slots[r].p + q * n, x.device, in[q], ranks[q]->device, size_t(n) * 4, x.stream),
                        "gather partial sums");
            }
            sum_slots(slots[r], int(world), n, out[r], x.stream);
            checked(event_record(done[r], x.stream), "record gathered partial sums");
        }
        // A rank may overwrite its partial sums only after every peer has copied them.
        for (size_t r = 0; r < world; ++r) {
            ranks[r]->activate();
            for (size_t q = 0; q < world; ++q)
                if (q != r)
                    checked(stream_wait(ranks[r]->stream, done[q]), "wait for peers' gathers");
        }
    }
};
#if defined(GLM53_NCCL)
void nccl_checked(ncclResult_t e, const char *what) {
    if (e != ncclSuccess)
        throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(e));
}
class NcclCollective final : public Collective {
    std::vector<ncclComm_t> comms;

  public:
    explicit NcclCollective(const std::vector<Context *> &ranks) : comms(ranks.size()) {
        std::vector<int> devices;
        for (auto *x : ranks)
            devices.push_back(x->device);
        auto sorted = devices;
        std::sort(sorted.begin(), sorted.end());
        check(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
              "NCCL needs one GPU per rank; use GLM53_TP_TRANSPORT=copy for ranks on one device");
        nccl_checked(ncclCommInitAll(comms.data(), int(comms.size()), devices.data()), "NCCL communicator");
    }
    ~NcclCollective() override {
        for (auto comm : comms)
            if (comm)
                (void)ncclCommDestroy(comm);
    }
    const char *name() const override { return "nccl"; }
    void all_reduce(const std::vector<Context *> &ranks, const std::vector<const float *> &in,
                    const std::vector<float *> &out, int n) override {
        nccl_checked(ncclGroupStart(), "NCCL group");
        for (size_t r = 0; r < ranks.size(); ++r) {
            ranks[r]->activate();
            nccl_checked(ncclAllReduce(in[r], out[r], size_t(n), ncclFloat, ncclSum, comms[r], ranks[r]->stream),
                         "NCCL all-reduce");
        }
        nccl_checked(ncclGroupEnd(), "NCCL group");
    }
};
#endif
std::unique_ptr<Collective> make_collective(const std::vector<Context *> &ranks) {
    const char *value = std::getenv("GLM53_TP_TRANSPORT");
    std::string transport = value ? value : "";
    check(transport.empty() || transport == "copy" || transport == "nccl",
          "GLM53_TP_TRANSPORT must be copy or nccl");
    bool shared_device = false;
    for (size_t r = 0; r < ranks.size(); ++r)
        for (size_t q = 0; q < r; ++q)
            shared_device |= ranks[r]->device == ranks[q]->device;
#if defined(GLM53_NCCL)
    if (transport == "nccl" || (transport.empty() && !shared_device))
        return std::make_unique<NcclCollective>(ranks);
#else
    check(transport != "nccl", "this build has no NCCL; rebuild with GLM53_NCCL or use GLM53_TP_TRANSPORT=copy");
#endif
    (void)shared_device;
    return std::make_unique<CopyCollective>(ranks);
}
int power_of_two(int n) {
    int p = 1;
    while (p < n) {
        check(p <= INT32_MAX / 2, "GPU context too large");
        p *= 2;
    }
    return p;
}
// Per-device activation scratch: every device owns one set.
struct Scratch {
    Buffer<B16> x, next, collapsed, normalized, branch, embed;
    Buffer<B16> q, k, v, low, forget, beta, gate, attended, qr, latent, iq, ik, iw, ig;
    Buffer<B16> mlp_gate, mlp_up, logits;
    Buffer<float> post, mix, hc_scratch, query, attention_scores, weighted, pool_scores;
    Buffer<float> partial, reduced; // this rank's FP32 partial sums and their all-reduced total
    Buffer<float> router_scores, router_weights, group_scores, host_output, upload_embedding;
    Buffer<int> pool_ids, expert_ids, allowed;
};
size_t env_bytes(const char *name, bool &set) {
    const char *value = std::getenv(name);
    set = value != nullptr;
    if (!value)
        return 0;
    double gib = 0;
    try {
        size_t used = 0;
        gib = std::stod(value, &used);
        check(used == std::char_traits<char>::length(value) && gib >= 0 && std::isfinite(gib),
              "invalid value");
    } catch (const std::exception &) {
        throw std::runtime_error(std::string(name) + " must be a non-negative number of GiB");
    }
    return size_t(gib * double(size_t(1) << 30));
}
bool env_flag(const char *name) {
    const char *value = std::getenv(name);
    return value && *value && std::string(value) != "0";
}
class DeviceModel final : public InferenceBackend {
    Weights weights;
    Config c;
    bool tensor_parallel;
    std::vector<std::unique_ptr<Context>> contexts;
    std::vector<Scratch> scratch;
    struct State {
        Buffer<float> recurrent;
        Buffer<B16> convolution, latent, keys, gates, pooled, kv;
        Buffer<int> selected;
        int count = 0;
    };
    std::vector<std::vector<State>> states;  // [context][layer]: KDA state and MLA caches of this rank's heads
    std::vector<std::vector<int>> layer_ranks; // contexts that execute each layer (all under tensor parallelism)
    std::unique_ptr<Collective> comm;
    std::vector<int> group; // contexts executing the current layer
    int position = 0;
    bool failed = false, reported = false;
    Vec embedding_row;
    const float *constant(Context &x, const std::string &name, size_t size, size_t begin = 0, size_t count = 0) {
        return x.constant(weights, name, size, begin, count);
    }
    int linear(Context &x, const std::string &name, const Buffer<B16> &in, size_t width, Buffer<B16> &out,
               Slice slice = {}) {
        return x.linear(weights, name, in, width, out, slice);
    }
    int linear32(Context &x, const std::string &name, const Buffer<B16> &in, size_t width, float *out,
                 Slice slice = {}) {
        return x.linear32(weights, name, in, width, out, slice);
    }
    void rms(Context &x, const Buffer<B16> &in, Buffer<B16> &out, const std::string &name, int width) {
        out.ensure(width, x.stream);
        norm(in, constant(x, name, width), nullptr, out, width, c.eps, false, x.stream);
    }
    // This rank's contiguous share of `units` equal units (heads, intermediate channels).
    static std::pair<int, int> shard(int units, const Context &x, const char *what) {
        check(units % x.world == 0, std::string("GPU tensor parallelism needs the ") + what +
                                        " divisible by the rank count (" + std::to_string(units) + " / " +
                                        std::to_string(x.world) + ")");
        int n = units / x.world;
        return {x.rank * n, n};
    }
    // Column-parallel gate/up on this rank's intermediate channels and the row-parallel down projection;
    // `out` (c.hidden floats) receives this rank's FP32 partial of the MLP output.
    void dense(Context &x, Scratch &s, const std::string &p, int intermediate, float *out) {
        auto [i0, il] = shard(intermediate, x, "MLP intermediate width");
        check(linear(x, p + "gate_proj", s.normalized, c.hidden, s.mlp_gate, Slice::rows(i0, il)) == il &&
                  linear(x, p + "up_proj", s.normalized, c.hidden, s.mlp_up, Slice::rows(i0, il)) == il,
              "GPU SwiGLU width mismatch");
        swiglu(s.mlp_gate, s.mlp_up, il, c.swiglu_limit, x.stream);
        check(linear32(x, p + "down_proj", s.mlp_gate, il, out, Slice::cols(i0, il)) == c.hidden,
              "GPU MLP output width mismatch");
    }
    // Sums the first `n` values of every rank's s.partial across the group; returns each rank's total
    // (its own partial when the group is one device).
    std::vector<const float *> reduce(int n) {
        std::vector<const float *> totals;
        if (group.size() == 1) {
            totals.push_back(scratch[group[0]].partial);
            return totals;
        }
        std::vector<Context *> ranks;
        std::vector<const float *> in;
        std::vector<float *> out;
        for (int i : group) {
            Context &x = *contexts[i];
            Scratch &s = scratch[i];
            x.activate();
            s.reduced.ensure(n, x.stream);
            ranks.push_back(&x);
            in.push_back(s.partial);
            out.push_back(s.reduced);
            totals.push_back(s.reduced);
        }
        comm->all_reduce(ranks, in, out, n);
        return totals;
    }
    // Rounds the reduced block output of every rank into its bf16 branch.
    void branch_from(const std::vector<const float *> &totals) {
        for (size_t k = 0; k < group.size(); ++k) {
            Context &x = *contexts[group[k]];
            Scratch &s = scratch[group[k]];
            x.activate();
            s.branch.ensure(c.hidden, x.stream);
            from_float(totals[k], s.branch, c.hidden, x.stream);
        }
    }
    void mlp(int layer, const std::string &p) {
        if (c.mlp[layer] == "dense") {
            for (int i : group) {
                contexts[i]->activate();
                scratch[i].partial.ensure(c.hidden, contexts[i]->stream);
                dense(*contexts[i], scratch[i], p, c.intermediate, scratch[i].partial);
            }
            branch_from(reduce(c.hidden));
            return;
        }
        // Routing intentionally stays FP32, including the router's source weights. It runs on the group's
        // first rank (its input is replicated) and the choice reaches the host once per layer.
        Context &x0 = *contexts[group[0]];
        Scratch &s0 = scratch[group[0]];
        x0.activate();
        s0.router_scores.ensure(c.experts, x0.stream);
        float_matvec(constant(x0, p + "gate.weight", size_t(c.experts) * c.hidden), s0.normalized, s0.router_scores,
                     c.experts, c.hidden, x0.stream);
        s0.expert_ids.ensure(c.expert_topk, x0.stream);
        s0.router_weights.ensure(c.expert_topk, x0.stream);
        s0.group_scores.ensure(c.groups, x0.stream);
        s0.allowed.ensure(c.groups, x0.stream);
        route(s0.router_scores, constant(x0, p + "gate.e_score_correction_bias", c.experts), s0.expert_ids,
              s0.router_weights, s0.group_scores, s0.allowed, c.experts, c.expert_topk, c.groups, c.top_groups,
              c.norm_topk, c.routed_scale, x0.stream);
        x0.read_indices.ensure(c.expert_topk);
        x0.readback.ensure(c.expert_topk);
        checked(d2h(x0.read_indices.p, s0.expert_ids, size_t(c.expert_topk) * sizeof(int), x0.stream),
                "read expert IDs");
        checked(d2h(x0.readback.p, s0.router_weights, size_t(c.expert_topk) * 4, x0.stream), "read router weights");
        checked(sync(x0.stream), "route experts");
        std::vector<int> ids(x0.read_indices.p, x0.read_indices.p + c.expert_topk);
        Vec factors(x0.readback.p, x0.readback.p + c.expert_topk);
        for (int i = 0; i < c.expert_topk; ++i)
            check(ids[i] >= 0 && ids[i] < c.experts && std::isfinite(factors[i]), "invalid GPU expert route");
        // Every expert is sharded like the dense MLP. The shared and routed partials sit side by side in
        // s.partial, so one all-reduce per MoE block sums them all; each expert output is then rounded to
        // bf16 before the weighted accumulation, as the reference does (the single-GPU path is unchanged).
        size_t outputs = size_t(c.expert_topk) + 1;
        for (int i : group) {
            Context &x = *contexts[i];
            Scratch &s = scratch[i];
            x.activate();
            s.partial.ensure(outputs * c.hidden, x.stream);
            dense(x, s, p + "shared_experts.", c.expert_intermediate, s.partial);
            for (int e = 0; e < c.expert_topk; ++e)
                dense(x, s, p + "experts." + std::to_string(ids[e]) + ".", c.expert_intermediate,
                      s.partial.p + size_t(e + 1) * c.hidden);
        }
        auto totals = reduce(int(outputs * c.hidden));
        for (size_t k = 0; k < group.size(); ++k) {
            Context &x = *contexts[group[k]];
            Scratch &s = scratch[group[k]];
            x.activate();
            s.router_weights.ensure(c.expert_topk, x.stream);
            // Pageable source: the copy completes before this returns, so `factors` may go out of scope.
            checked(h2d(s.router_weights, factors.data(), factors.size() * 4, x.stream), "upload expert weights");
            s.branch.ensure(c.hidden, x.stream);
            combine_experts(totals[k], s.router_weights, c.expert_topk, c.hidden, s.branch, x.stream);
        }
    }
    // This rank's heads of a KDA block: q/k/v, convolutions, gates, recurrent state and the gated norm follow
    // the heads; the row-parallel o_proj leaves an FP32 partial in s.partial.
    void kda_heads(int i, int layer, const std::string &p) {
        Context &x = *contexts[i];
        Scratch &s = scratch[i];
        State &st = states[i][layer];
        auto stream = x.stream;
        auto [h0, hl] = shard(c.linear_heads, x, "KDA head count");
        int dim = c.linear_dim, width = hl * dim, full = c.linear_heads * dim;
        Slice heads = Slice::rows(h0 * dim, width);
        check(linear(x, p + "q_proj", s.normalized, c.hidden, s.q, heads) == width &&
                  linear(x, p + "k_proj", s.normalized, c.hidden, s.k, heads) == width &&
                  linear(x, p + "v_proj", s.normalized, c.hidden, s.v, heads) == width,
              "GPU KDA projection geometry");
        size_t history = size_t(3) * width * (c.conv - 1), memory = size_t(width) * dim;
        if (!st.recurrent.p) {
            st.recurrent.ensure(memory, stream);
            checked(zero(st.recurrent, memory * 4, stream), "clear KDA memory");
            st.convolution.ensure(history, stream);
            if (history)
                checked(zero(st.convolution, history * 2, stream), "clear KDA convolution");
        }
        size_t taps = size_t(full) * c.conv, tap0 = size_t(h0) * dim * c.conv, tapn = size_t(width) * c.conv;
        convolution(s.q, s.k, s.v, st.convolution, constant(x, p + "q_conv1d.weight", taps, tap0, tapn),
                    constant(x, p + "k_conv1d.weight", taps, tap0, tapn),
                    constant(x, p + "v_conv1d.weight", taps, tap0, tapn), width, c.conv, stream);
        check(linear(x, p + "f_a_proj", s.normalized, c.hidden, s.low) == dim &&
                  linear(x, p + "f_b_proj", s.low, dim, s.forget, heads) == width &&
                  linear(x, p + "b_proj", s.normalized, c.hidden, s.beta, Slice::rows(h0, hl)) == hl,
              "GPU KDA gate geometry");
        s.attended.ensure(width, stream);
        kda(s.q, s.k, s.v, s.forget, s.beta, constant(x, p + "dt_bias", full, size_t(h0) * dim, width),
            constant(x, p + "A_log", c.linear_heads, h0, hl), st.recurrent, s.attended, hl, dim, c.gate_lower,
            stream);
        check(linear(x, p + "g_a_proj", s.normalized, c.hidden, s.low) == dim &&
                  linear(x, p + "g_b_proj", s.low, dim, s.gate, heads) == width,
              "GPU KDA output gate geometry");
        gated_norm(s.attended, s.gate, constant(x, p + "o_norm.weight", dim), hl, dim, c.eps, stream);
        s.partial.ensure(c.hidden, stream);
        check(linear32(x, p + "o_proj", s.attended, width, s.partial, Slice::cols(h0 * dim, width)) == c.hidden,
              "GPU KDA output geometry");
    }
    // This rank's heads of a sparse MLA block. The low-rank query/latent projections, the latent cache and the
    // indexer are replicated (identical inputs, deterministic kernels), so no exchange precedes the heads;
    // q_b, the decoded kv_b expansion and the row-parallel o_proj follow the heads.
    void mla_heads(int i, int layer, const std::string &p) {
        Context &x = *contexts[i];
        Scratch &s = scratch[i];
        State &st = states[i][layer];
        auto &layer_states = states[i];
        auto stream = x.stream;
        auto [h0, hl] = shard(c.heads, x, "MLA head count");
        check(linear(x, p + "q_a_proj", s.normalized, c.hidden, s.qr) == c.qrank, "GPU MLA low-rank query geometry");
        rms(x, s.qr, s.qr, p + "q_a_layernorm.weight", c.qrank);
        check(linear(x, p + "q_b_proj", s.qr, c.qrank, s.q, Slice::rows(h0 * c.qdim, hl * c.qdim)) == hl * c.qdim,
              "GPU MLA query geometry");
        check(linear(x, p + "kv_a_proj_with_mqa", s.normalized, c.hidden, s.latent) == c.kvrank,
              "GPU MLA latent geometry");
        rms(x, s.latent, s.latent, p + "kv_a_layernorm.weight", c.kvrank);
        int capacity = std::min(c.max_context, std::max(256, power_of_two(position + 1)));
        st.latent.ensure(size_t(capacity) * c.kvrank, stream, true);
        checked(d2d(st.latent.p + size_t(position) * c.kvrank, s.latent, size_t(c.kvrank) * 2, stream),
                "append MLA latent");
        st.selected.ensure(c.topk + c.pool, stream);
        if (c.indexers[layer] == "shared") {
            int previous = layer - 1;
            while (previous >= 0 && !layer_states[previous].count)
                --previous;
            check(previous >= 0, "GPU shared indexer has no preceding result");
            check(layer_ranks[previous] == layer_ranks[layer],
                  "GPU layer split must keep a shared indexer on the device of its full indexer");
            st.count = layer_states[previous].count;
            checked(d2d(st.selected, layer_states[previous].selected, size_t(st.count) * sizeof(int), stream),
                    "reuse sparse indices");
        } else {
            auto ix = p + "indexer.";
            check(linear(x, ix + "wq_b", s.qr, c.qrank, s.iq) == c.index_heads * c.index_dim &&
                      linear(x, ix + "wk", s.normalized, c.hidden, s.ik) == c.index_dim &&
                      linear(x, ix + "index_kpool_compress_gate", s.normalized, c.hidden, s.ig) == c.index_dim &&
                      linear(x, ix + "weights_proj", s.normalized, c.hidden, s.iw) == c.index_heads,
                  "GPU indexer geometry");
            norm(s.ik, constant(x, ix + "k_norm.weight", c.index_dim), constant(x, ix + "k_norm.bias", c.index_dim),
                 s.ik, c.index_dim, 1e-6f, true, stream);
            st.keys.ensure(size_t(c.pool) * c.index_dim, stream);
            st.gates.ensure(size_t(c.pool) * c.index_dim, stream);
            int slot = position % c.pool, pools = (position + 1) / c.pool;
            checked(d2d(st.keys.p + size_t(slot) * c.index_dim, s.ik, size_t(c.index_dim) * 2, stream),
                    "append index key");
            checked(d2d(st.gates.p + size_t(slot) * c.index_dim, s.ig, size_t(c.index_dim) * 2, stream),
                    "append index gate");
            st.pooled.ensure(size_t((capacity + c.pool - 1) / c.pool) * c.index_dim, stream, true);
            if (slot == c.pool - 1)
                pool_key(st.keys, st.gates, constant(x, ix + "index_kpool_compress_ape", size_t(c.pool) * c.index_dim),
                         st.pooled.p + size_t(pools - 1) * c.index_dim, c.pool, c.index_dim, stream);
            int padded = power_of_two(std::max(1, pools));
            s.pool_scores.ensure(std::max(1024, padded), stream);
            s.pool_ids.ensure(std::max(1024, padded), stream);
            index_scores(s.iq, st.pooled, s.iw, s.pool_scores, pools, c.index_heads, c.index_dim, stream);
            select_keys(s.pool_scores, s.pool_ids, st.selected, pools, padded, c.pool, c.topk, position, stream);
            st.count = std::min(pools, c.topk / c.pool) * c.pool + (position + 1) % c.pool;
        }
        int expansion = c.qdim + c.vdim;
        if (!st.kv.p) {
            // This rank's heads, decoded once into the layer's own buffer; the packed source need not stay.
            auto m = x.matrix(weights, p + "kv_b_proj", false, Slice::rows(h0 * expansion, hl * expansion));
            check(m.rows == hl * expansion && m.cols == c.kvrank, "GPU MLA expansion geometry");
            st.kv.ensure(size_t(m.rows) * m.cols, stream);
            decode_matrix(m, st.kv, stream);
        }
        s.query.ensure(size_t(hl) * c.kvrank, stream);
        s.weighted.ensure(size_t(hl) * c.kvrank, stream);
        s.attention_scores.ensure(size_t(hl) * st.count, stream);
        s.attended.ensure(size_t(hl) * c.vdim, stream);
        absorb_query(s.q, st.kv, s.query, hl, c.qdim, c.vdim, c.kvrank, stream);
        mla(s.query, st.latent, st.kv, st.selected, s.attended, s.attention_scores, s.weighted, st.count, hl, c.qdim,
            c.vdim, c.kvrank, stream);
        s.partial.ensure(c.hidden, stream);
        check(linear32(x, p + "o_proj", s.attended, size_t(hl) * c.vdim, s.partial,
                       Slice::cols(h0 * c.vdim, hl * c.vdim)) == c.hidden,
              "GPU MLA output geometry");
    }
    void attention(int layer, const std::string &p) {
        for (int i : group) {
            contexts[i]->activate();
            if (c.attention[layer] == "linear_attention")
                kda_heads(i, layer, p);
            else
                mla_heads(i, layer, p);
        }
        branch_from(reduce(c.hidden));
    }
    // Makes the contexts of layer `l` current, carrying the residual streams over from the previous group
    // (a layer-split boundary; tensor-parallel groups never change).
    void enter(int l) {
        const auto &next = layer_ranks[l];
        if (next == group)
            return;
        size_t n = size_t(c.streams) * c.hidden;
        if (!group.empty()) {
            Context &from = *contexts[group[0]];
            Scratch &source = scratch[group[0]];
            from.activate();
            checked(event_record(from.done, from.stream), "record GPU hand-off");
            for (int i : next) {
                Context &to = *contexts[i];
                Scratch &target = scratch[i];
                to.activate();
                target.x.ensure(n, to.stream);
                target.next.ensure(n, to.stream);
                checked(stream_wait(to.stream, from.done), "wait for previous GPU");
                checked(copy_peer(target.x, to.device, source.x, from.device, n * sizeof(B16), to.stream),
                        "transfer residual streams between GPUs");
            }
        }
        group = next;
    }
    // Layer split: contiguous layer ranges with roughly equal checkpoint bytes per device; a shared indexer
    // stays with the full indexer it reuses, and the head goes with the last layer.
    void assign_layers(int devices) {
        layer_ranks.assign(c.layers, {0});
        if (tensor_parallel) {
            std::vector<int> all(devices);
            for (int i = 0; i < devices; ++i)
                all[i] = i;
            layer_ranks.assign(c.layers, all);
            return;
        }
        if (devices <= 1)
            return;
        std::vector<double> bytes(c.layers, 0);
        auto prefix = weights.prefix + "layers.";
        for (const auto &it : weights.index().at("weight_map").items()) {
            const std::string &name = it.key();
            if (!name.starts_with(prefix))
                continue;
            size_t dot = name.find('.', prefix.size());
            if (dot == std::string::npos)
                continue;
            int layer = -1;
            auto digits = std::string_view(name).substr(prefix.size(), dot - prefix.size());
            auto result = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
            if (result.ec != std::errc{} || result.ptr != digits.data() + digits.size() || layer < 0 ||
                layer >= c.layers)
                continue;
            bytes[layer] += double(weights.file(name).bytes(name).size());
        }
        double total = 0;
        for (double b : bytes)
            total += b;
        double running = 0;
        int device = 0;
        for (int l = 0; l < c.layers; ++l) {
            // Advance when the running total has crossed this device's share, never splitting a shared
            // indexer from its source layer.
            if (device + 1 < devices && running >= total * (device + 1) / devices &&
                !(c.attention[l] == "deepseek_sparse_attention" && c.indexers[l] == "shared"))
                ++device;
            layer_ranks[l] = {device};
            running += bytes[l];
        }
    }
    void report() {
        if (reported)
            return;
        reported = true;
        for (size_t i = 0; i < contexts.size(); ++i) {
            const auto &x = *contexts[i];
            std::cerr << "GLM GPU " << x.device << ": ";
            if (tensor_parallel)
                std::cerr << "tensor-parallel rank " << x.rank << '/' << x.world << " (" << comm->name()
                          << " all-reduce), all layers";
            else {
                int first = -1, last = -1;
                for (int l = 0; l < c.layers; ++l)
                    if (layer_ranks[l][0] == int(i)) {
                        if (first < 0)
                            first = l;
                        last = l;
                    }
                std::cerr << "layers " << first << "-" << last;
            }
            std::cerr << ", resident " << double(x.resident_used) / double(size_t(1) << 30) << " GiB in "
                      << x.resident.size() << " matrices (budget " << double(x.resident_budget) / double(size_t(1) << 30)
                      << " GiB); streamed " << x.streamed_count << " matrices, "
                      << double(x.streamed_bytes) / double(size_t(1) << 30) << " GiB per token\n";
        }
    }

  public:
    DeviceModel(const fs::path &path, const GpuOptions &options)
        : weights(path), c(read_json(path / "config.json")), tensor_parallel(options.tensor_parallel && options.devices > 1) {
        check(c.linear_dim <= 2048, "GPU KDA head dimension exceeds shared-memory limit");
        check(c.expert_topk <= (c.experts / std::max(1, c.groups)) * c.top_groups ||
                  std::none_of(c.mlp.begin(), c.mlp.end(), [](const auto &t) { return t == "sparse"; }),
              "not enough experts in selected groups");
        check(options.device >= 0 && options.devices >= 1, "invalid GPU device selection");
        bool same_device = tensor_parallel && (options.same_device || env_flag("GLM53_TP_SAME_DEVICE"));
        int count = 0;
        checked(device_count(&count), "count GPUs");
        check(options.device + (same_device ? 1 : options.devices) <= count,
              tensor_parallel ? "GPU tensor parallelism needs consecutive devices from --device"
                              : "GPU layer split needs consecutive devices from --device");
        if (tensor_parallel) {
            bool sparse = std::any_of(c.mlp.begin(), c.mlp.end(), [](const auto &t) { return t == "sparse"; });
            check(c.linear_heads % options.devices == 0 && c.heads % options.devices == 0 &&
                      c.intermediate % options.devices == 0 && (!sparse || c.expert_intermediate % options.devices == 0),
                  "GPU tensor parallelism needs KDA heads, MLA heads and MLP widths divisible by the rank count");
        }
        int context = options.context > 0 ? std::min(options.context, c.max_context) : c.max_context;
        for (int i = 0; i < options.devices; ++i) {
            contexts.push_back(std::make_unique<Context>(same_device ? options.device : options.device + i));
            if (tensor_parallel) {
                contexts.back()->rank = i;
                contexts.back()->world = options.devices;
            }
        }
        scratch.resize(contexts.size());
        states.resize(contexts.size());
        for (auto &s : states)
            s.resize(c.layers);
        assign_layers(options.devices);
        if (tensor_parallel) {
            std::vector<Context *> ranks;
            for (auto &x : contexts)
                ranks.push_back(x.get());
            comm = make_collective(ranks);
        }
        bool set = false;
        size_t fixed = env_bytes("GLM53_RESIDENT_GIB", set);
        for (size_t i = 0; i < contexts.size(); ++i) {
            auto &x = *contexts[i];
            if (options.resident_bytes >= 0)
                x.resident_budget = size_t(options.resident_bytes);
            else if (set)
                x.resident_budget = fixed;
            else {
                // Leave room for the attention caches at `context` tokens (the latent cache is replicated,
                // the decoded expansions and scores follow this rank's heads), the FP32 constants and
                // activations, plus a fixed margin for the allocator and other users.
                int mla = 0;
                for (int l = 0; l < c.layers; ++l)
                    mla += std::find(layer_ranks[l].begin(), layer_ranks[l].end(), int(i)) != layer_ranks[l].end() &&
                           c.attention[l] == "deepseek_sparse_attention";
                size_t heads = size_t(c.heads) / x.world;
                size_t per_layer = size_t(context) * c.kvrank * 2 + size_t(context / c.pool + 1) * c.index_dim * 2 +
                                   heads * (c.qdim + c.vdim) * c.kvrank * 2 + heads * (c.topk + c.pool) * 4;
                size_t headroom = size_t(4) << 30;
                headroom += size_t(mla) * per_layer + size_t(c.vocab) * 8;
                size_t free = x.free_memory();
                if (same_device) // every rank shares this GPU's free memory
                    free /= contexts.size();
                x.resident_budget = free > headroom ? free - headroom : 0;
            }
        }
    }
    ~DeviceModel() override {
        for (auto &x : contexts) {
            (void)set_device(x->device);
            (void)sync(x->stream);
        }
    }
    void reset() override {
        for (size_t i = 0; i < contexts.size(); ++i) {
            auto &x = *contexts[i];
            x.activate();
            checked(sync(x.stream), "finish previous GPU sequence");
            for (auto &s : states[i]) {
                if (s.recurrent.p)
                    checked(zero(s.recurrent, s.recurrent.capacity * 4, x.stream), "reset KDA state");
                if (s.convolution.p)
                    checked(zero(s.convolution, s.convolution.capacity * 2, x.stream), "reset convolution");
                s.count = 0;
            }
        }
        position = 0;
        failed = false;
    }
    Vec step(int token) override {
        check(!failed, "GPU inference failed; reset before reusing this sequence");
        check(token >= 0 && token < c.vocab && position < c.max_context,
              "token/context outside GPU model limits");
        try {
            group.clear();
            enter(0);
            embedding_row = weights.row(weights.prefix + "embed_tokens.weight", token, c.hidden);
            for (int i : group) {
                Context &x = *contexts[i];
                Scratch &s = scratch[i];
                x.activate();
                s.upload_embedding.ensure(c.hidden, x.stream);
                s.embed.ensure(c.hidden, x.stream);
                checked(h2d(s.upload_embedding, embedding_row.data(), embedding_row.size() * 4, x.stream),
                        "upload embedding row");
                from_float(s.upload_embedding, s.embed, c.hidden, x.stream);
                s.x.ensure(size_t(c.streams) * c.hidden, x.stream);
                s.next.ensure(size_t(c.streams) * c.hidden, x.stream);
                replicate(s.embed, s.x, c.hidden, c.streams, x.stream);
            }
            for (int l = 0; l < c.layers; ++l) {
                enter(l);
                auto p = weights.prefix + "layers." + std::to_string(l) + ".";
                for (bool ffn : {false, true}) {
                    auto hc = p + (ffn ? "hc_ffn_" : "hc_attn_");
                    // mHC acts on the replicated residual streams: every rank mixes and normalizes alike.
                    for (int i : group) {
                        Context &x = *contexts[i];
                        Scratch &s = scratch[i];
                        x.activate();
                        s.collapsed.ensure(c.hidden, x.stream);
                        s.post.ensure(c.streams, x.stream);
                        s.mix.ensure(c.streams * c.streams, x.stream);
                        s.hc_scratch.ensure(c.streams * (c.streams + 3), x.stream);
                        hc_pre(s.x, constant(x, hc + "fn", size_t(c.streams) * (c.streams + 2) * c.streams * c.hidden),
                               constant(x, hc + "base", c.streams * (c.streams + 2)), constant(x, hc + "scale", 3),
                               s.collapsed, s.post, s.mix, s.hc_scratch, c.hidden, c.streams, c.sinkhorn, c.eps,
                               c.hc_eps, x.stream);
                        rms(x, s.collapsed, s.normalized,
                            p + (ffn ? "post_attention_layernorm.weight" : "input_layernorm.weight"), c.hidden);
                    }
                    if (ffn)
                        mlp(l, p + "mlp.");
                    else
                        attention(l, p + "self_attn.");
                    for (int i : group) {
                        Context &x = *contexts[i];
                        Scratch &s = scratch[i];
                        x.activate();
                        hc_post(s.x, s.branch, s.post, s.mix, s.next, c.hidden, c.streams, x.stream);
                        std::swap(s.x, s.next);
                    }
                }
            }
            // The head is vocabulary-parallel: each rank scores its rows and the host concatenates them.
            Vec out(c.vocab);
            std::vector<std::pair<int, int>> rows;
            for (int i : group) {
                Context &x = *contexts[i];
                Scratch &s = scratch[i];
                x.activate();
                int v0 = int(size_t(c.vocab) * x.rank / x.world), v1 = int(size_t(c.vocab) * (x.rank + 1) / x.world);
                rows.emplace_back(v0, v1 - v0);
                s.collapsed.ensure(c.hidden, x.stream);
                mean_streams(s.x, s.collapsed, c.hidden, c.streams, x.stream);
                rms(x, s.collapsed, s.normalized, weights.prefix + "norm.weight", c.hidden);
                s.host_output.ensure(size_t(v1 - v0), x.stream);
                check(linear32(x, "lm_head", s.normalized, c.hidden, s.host_output, Slice::rows(v0, v1 - v0)) == v1 - v0,
                      "GPU output vocabulary mismatch");
                x.readback.ensure(size_t(v1 - v0));
                checked(d2h(x.readback.p, s.host_output, size_t(v1 - v0) * 4, x.stream), "read logits");
            }
            for (size_t k = 0; k < group.size(); ++k) {
                Context &x = *contexts[group[k]];
                x.activate();
                checked(sync(x.stream), "GPU inference synchronization");
                std::copy(x.readback.p, x.readback.p + rows[k].second, out.begin() + rows[k].first);
            }
            for (float v : out)
                check(std::isfinite(v), "nonfinite GPU logits");
            ++position;
            report();
            return out;
        } catch (...) {
            failed = true;
            throw;
        }
    }
};
struct Projector {
    Context ctx;
    Buffer<float> source, result;
    Buffer<B16> input;
    explicit Projector(int d) : ctx(d) {}
    ~Projector() {
        (void)set_device(ctx.device);
        (void)sync(ctx.stream);
    }
    Vec operator()(const Weights &w, const std::string &base, const Vec &x, Slice slice = {}) {
        ctx.activate();
        source.ensure(x.size(), ctx.stream);
        input.ensure(x.size(), ctx.stream);
        checked(h2d(source, x.data(), x.size() * 4, ctx.stream), "upload projection input");
        from_float(source, input, int(x.size()), ctx.stream);
        auto m = ctx.matrix(w, base, false, slice);
        check(size_t(m.cols) == x.size(), "GPU projection width mismatch");
        result.ensure(m.rows, ctx.stream);
        gemv(m, input, result, true, ctx.stream);
        ctx.readback.ensure(m.rows);
        checked(d2h(ctx.readback.p, result, size_t(m.rows) * 4, ctx.stream), "read projection");
        checked(sync(ctx.stream), "GPU projection synchronization");
        return Vec(ctx.readback.p, ctx.readback.p + m.rows);
    }
};
} // namespace
std::function<Vec(const Weights &, const std::string &, const Vec &)> gpu_projector(int device) {
    auto p = std::make_shared<Projector>(device);
    return [p](const Weights &w, const std::string &name, const Vec &x) { return (*p)(w, name, x); };
}
std::function<Vec(const Weights &, const std::string &, const Vec &, int, int, int)> gpu_slice_projector(int device) {
    auto p = std::make_shared<Projector>(device);
    return [p](const Weights &w, const std::string &name, const Vec &x, int dim, int begin, int count) {
        check(dim == 0 || dim == 1, "slice dimension must be 0 (rows) or 1 (columns)");
        return (*p)(w, name, x, Slice{dim, begin, count});
    };
}
std::shared_ptr<InferenceBackend> gpu_backend(const fs::path &path, const GpuOptions &options) {
    return std::make_shared<DeviceModel>(path, options);
}
std::shared_ptr<InferenceBackend> gpu_backend(const fs::path &path, int device) {
    GpuOptions options;
    options.device = device;
    return gpu_backend(path, options);
}
} // namespace glm53
