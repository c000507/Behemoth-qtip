#include "tensor_model.hpp"
#include "batch_kernels.hpp"
#include "behemoth/qtip.hpp"
#include "fast_prefill.hpp"
#include "peer_allgather.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <map>
#include <nccl.h>
#include <set>
namespace behemoth::cuda_runtime {
namespace {
using namespace glm53::gpu;
constexpr int H = 12288, I = 28672, V = 32768, D = 128, QH = 96, KH = 8;
void nc(ncclResult_t error) {
    if (error != ncclSuccess)
        throw std::runtime_error(std::string("NCCL: ") + ncclGetErrorString(error));
}
void blas(cublasStatus_t error) {
    if (error != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(int(error)));
}
struct Scope {
    int old = 0;
    explicit Scope(int d) {
        checked(cudaGetDevice(&old), "get CUDA device");
        checked(set_device(d), "select TP rank");
    }
    ~Scope() { (void)set_device(old); }
};
struct Event {
    int device = 0;
    cudaEvent_t value{};
    void create(int d) {
        device = d;
        Scope scope(d);
        checked(cudaEventCreateWithFlags(&value, cudaEventDisableTiming), "create graph join event");
    }
    ~Event() {
        if (value) {
            Scope scope(device);
            (void)cudaEventDestroy(value);
        }
    }
};
struct Graph {
    int device;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    explicit Graph(int d) : device(d) {}
    ~Graph() {
        Scope scope(device);
        if (executable)
            (void)cudaGraphExecDestroy(executable);
        if (graph)
            (void)cudaGraphDestroy(graph);
    }
};
struct ProjectionProfile {
    int device, layer, index, rows;
    std::array<cudaEvent_t, 6> events{};
    ProjectionProfile(int d, int l, int i, int n) : device(d), layer(l), index(i), rows(n) {
        Scope scope(device);
        for (auto &e : events)
            checked(cudaEventCreate(&e), "create prefill profile event");
    }
    ~ProjectionProfile() {
        Scope scope(device);
        for (auto e : events)
            if (e)
                (void)cudaEventDestroy(e);
    }
};
struct DecodeMark {
    int device;
    std::string name;
    cudaEvent_t event{};
    DecodeMark(int d, const char *n, Stream stream) : device(d), name(n) {
        Scope scope(device);
        checked(cudaEventCreate(&event), "create decode profile event");
        checked(cudaEventRecord(event, stream), "record decode stage");
    }
    ~DecodeMark() {
        Scope scope(device);
        (void)cudaEventDestroy(event);
    }
};
template <class T> void upload(Buffer<T> &out, const T *data, size_t n, Stream s) {
    out.ensure(n, s);
    checked(h2d(out, data, n * sizeof(T), s), "upload TP weights");
}
struct Weight {
    int in = 0, out = 0, local = 0;
    Buffer<B16> trellis;
    Buffer<float> su, sv, scale;
    Buffer<uint32_t> lut;
    Buffer<uint8_t> fp8;
    void load(const qtip::HostLinear &h, int rank, int tp, Stream s) {
        in = h.in;
        out = h.out;
        local = out / tp;
        check(out % tp == 0 && local % 32 == 0, "TP projection rows must be divisible by 32");
        if (h.is_fp8()) {
            upload(fp8, h.fp8.data() + size_t(rank) * local * in, size_t(local) * in, s);
            upload(scale, h.fp8_scale.data() + rank * local, local, s);
        } else {
            check(!h.is_nvfp4(), "NVFP4 layers are not supported by the fast tensor-parallel profile");
            auto cb = qtip::make_codebook(h.tlut);
            size_t n = h.trellis.size() / tp;
            upload(trellis, h.trellis.data() + rank * n, n, s);
            upload(su, h.su.data(), h.su.size(), s);
            upload(sv, h.sv.data(), h.sv.size(), s);
            upload(lut, cb.bf16x2.data(), cb.bf16x2.size(), s);
            checked(sync(s), "finish codebook upload");
        }
        checked(sync(s), "finish TP projection upload");
    }
    QtipMatrix matrix(bool full = false) const { return {trellis, su, sv, lut, in, full ? out : local}; }
};
struct Layer {
    std::array<Weight, 7> weights;
    Buffer<float> norm1, norm2;
    Buffer<B16> keys, values;
    bool shared_su = false;
};
struct Rank {
    int device, rank, tp, context, max_batch;
    bool tensor = false;
    Stream stream{};
    ncclComm_t comm{};
    cublasHandle_t handle{};
    std::unique_ptr<FastPrefill> attention;
    std::vector<Layer> layers;
    Buffer<B16> embeddings, head, x, input, a, b, decoded;
    Buffer<float> final_norm, scratch, z, scores, logits, attention_parts;
    Buffer<int> ids, positions, lengths, selected;
    Rank(int d, int r, int t, int ctx, int mb)
        : device(d), rank(r), tp(t), context(ctx), max_batch(mb), layers(88) {
        Scope scope(device);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create TP stream");
        blas(cublasCreate(&handle));
        blas(cublasSetStream(handle, stream));
        cudaDeviceProp p{};
        checked(cudaGetDeviceProperties(&p, device), "query TP GPU");
        tensor = p.major >= 8;
        initialize_tables();
        ids.ensure(max_batch, stream);
        positions.ensure(max_batch, stream);
        lengths.ensure(max_batch, stream);
        selected.ensure(max_batch, stream);
        scores.ensure(size_t(max_batch) * (QH / tp) * context, stream);
        attention_parts.ensure(size_t(max_batch) * (KH / tp) * ((context + 127) / 128) * 16 * D, stream);
        logits.ensure(size_t(max_batch) * V, stream);
        if (tensor)
            attention = std::make_unique<FastPrefill>(QH / tp, KH / tp, D, H, stream);
    }
    ~Rank() {
        Scope scope(device);
        (void)sync(stream);
        attention.reset();
        layers.clear();
        embeddings.clear();
        head.clear();
        x.clear();
        input.clear();
        a.clear();
        b.clear();
        decoded.clear();
        final_norm.clear();
        scratch.clear();
        z.clear();
        scores.clear();
        attention_parts.clear();
        logits.clear();
        ids.clear();
        positions.clear();
        lengths.clear();
        selected.clear();
        if (comm)
            (void)ncclCommDestroy(comm);
        if (handle)
            (void)cublasDestroy(handle);
        if (stream)
            (void)stream_destroy(stream);
    }
    bool reserve(int rows, int splits, bool prefill) {
        Scope scope(device);
        size_t old = x.capacity + input.capacity + a.capacity + b.capacity + scratch.capacity + z.capacity +
                     ids.capacity;
        x.ensure(size_t(rows) * H, stream);
        input.ensure(size_t(rows) * I, stream);
        a.ensure(size_t(rows) * I, stream);
        b.ensure(size_t(rows) * I, stream);
        scratch.ensure(size_t(rows) * I, stream);
        z.ensure(std::max(size_t(rows) * I, size_t(rows) * (I / tp) * (splits + rank)), stream);
        ids.ensure(rows, stream);
        if (prefill && tensor)
            decoded.ensure(size_t(I / tp) * H, stream);
        size_t now = x.capacity + input.capacity + a.capacity + b.capacity + scratch.capacity + z.capacity +
                     ids.capacity;
        return now != old;
    }
    void multiply(const Weight &w, const B16 *rotated, float *out, int rows, int splits, bool large,
                  cudaEvent_t decoded_event = nullptr) {
        if (large && tensor) {
            qtip_decode_matrix(w.matrix(), decoded, stream);
            if (decoded_event)
                checked(cudaEventRecord(decoded_event, stream), "profile decoded weights");
            float alpha = 1, beta = 0;
            blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, w.local, rows, w.in, &alpha, decoded.p,
                              CUDA_R_16BF, w.in, rotated, CUDA_R_16BF, w.in, &beta, out, CUDA_R_32F, w.local,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else
            qtip_multiply_batch(w.matrix(), rotated, out, w.local, splits, rows, stream);
    }
    void frozen(const Weight &w, const B16 *value, B16 *out, int rows, bool large) {
        check(w.fp8.p != nullptr, "K/V projections must be FP8 in the fast Behemoth profile");
        if (large && tensor) {
            decode_fp8_matrix(w.fp8, decoded, size_t(w.local) * w.in, stream);
            float alpha = 1, beta = 0;
            blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, w.local, rows, w.in, &alpha, decoded.p,
                              CUDA_R_16BF, w.in, value, CUDA_R_16BF, w.in, &beta, z.p, CUDA_R_32F, w.local,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            scale_to_b16(z, w.scale, out, rows, w.local, stream);
        } else
            fp8_multiply_batch(w.fp8, w.scale, value, out, w.in, w.local, rows, stream, z);
    }
};
int split_count() {
    const char *env = std::getenv("BEHEMOTH_CUDA_QTIP_SPLITS");
    if (!env)
        return 16;
    int n = 0;
    auto end = env + std::char_traits<char>::length(env);
    auto p = std::from_chars(env, end, n);
    check(p.ec == std::errc{} && p.ptr == end && n >= 1 && n <= 32, "Invalid BEHEMOTH_CUDA_QTIP_SPLITS");
    return n;
}
} // namespace
struct TensorModel::Impl {
    Checkpoint checkpoint;
    int context, tp, max_batch, active = 0, splits;
    bool failed = false;
    std::vector<std::unique_ptr<Rank>> ranks;
    Event graph_join, graph_done;
    std::unique_ptr<PeerAllGather> peer_gather;
    bool peer_transport = false;
    bool peer_kernel = true;
    std::map<std::pair<int, int>, std::unique_ptr<Graph>> graphs;
    std::set<int> warmed_batches;
    bool graphs_enabled = true;
    bool register_attention = true;
    std::string profile_path;
    bool profile_reported = false;
    std::vector<std::unique_ptr<ProjectionProfile>> profile;
    std::string decode_profile_path;
    bool decode_profiled = false, record_decode = false;
    std::vector<std::unique_ptr<DecodeMark>> decode_marks;
    std::vector<int> position;
    Pinned<int> host_ids, host_positions, host_selected, host_lengths;
    Pinned<float> host_logits;
    Impl(const fs::path &path, int device, int ctx, int degree, int batch)
        : checkpoint(path), context(ctx), tp(degree), max_batch(batch), splits(split_count()) {
        check((tp == 1 || tp == 2) && batch >= 1 && batch <= 8, "Fast CUDA supports TP 1/2 and batch 1..8");
        check(ctx > 0 && ctx <= checkpoint.config.max_context, "Invalid fast CUDA context");
        int count = 0;
        checked(device_count(&count), "count TP GPUs");
        check(device >= 0 && device + tp <= count, "Insufficient CUDA devices for TP");
        setenv("NCCL_P2P_DISABLE", "1", 0);
        setenv("NCCL_MAX_NCHANNELS", "2", 0);
        setenv("NCCL_BUFFSIZE", "1048576", 0);
        if (const char *value = std::getenv("BEHEMOTH_CUDA_TP_TRANSPORT")) {
            check(std::string(value) == "peer" || std::string(value) == "peer-dma" ||
                      std::string(value) == "nccl",
                  "BEHEMOTH_CUDA_TP_TRANSPORT must be peer, peer-dma or nccl");
            peer_transport = std::string(value) != "nccl";
            peer_kernel = std::string(value) != "peer-dma";
        }
        for (int r = 0; r < tp; ++r)
            ranks.push_back(std::make_unique<Rank>(device + r, r, tp, context, max_batch));
        if (tp > 1) {
            if (peer_transport) {
                peer_gather = std::make_unique<PeerAllGather>(
                    std::array<int, 2>{device, device + 1},
                    std::array<Stream, 2>{ranks[0]->stream, ranks[1]->stream}, peer_kernel);
            } else {
                int devices[2] = {device, device + 1};
                ncclComm_t comms[2]{};
                nc(ncclCommInitAll(comms, tp, devices));
                for (int r = 0; r < tp; ++r)
                    ranks[r]->comm = comms[r];
            }
            graph_join.create(device);
            graph_done.create(device + 1);
        }
        if (const char *value = std::getenv("BEHEMOTH_CUDA_GRAPHS"))
            graphs_enabled = std::string(value) != "0";
        if (const char *value = std::getenv("BEHEMOTH_CUDA_ATTENTION_REGISTER"))
            register_attention = std::string(value) != "0";
        if (const char *value = std::getenv("BEHEMOTH_CUDA_PROFILE_PREFILL"))
            profile_path = value;
        if (const char *value = std::getenv("BEHEMOTH_CUDA_PROFILE_DECODE"))
            decode_profile_path = value;
        host_ids.ensure(max_batch);
        host_positions.ensure(max_batch);
        host_selected.ensure(max_batch);
        host_lengths.ensure(max_batch);
        host_logits.ensure(size_t(max_batch) * V);
        dense("model.embed_tokens.weight", false);
        dense("lm_head.weight", true);
        vector("model.norm.weight", -1, 0);
        const std::array<const char *, 7> names = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                                   "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
                                                   "mlp.down_proj"};
        for (int l = 0; l < 88; ++l) {
            std::string p = "model.layers." + std::to_string(l) + ".";
            vector(p + "input_layernorm.weight", l, 0);
            vector(p + "post_attention_layernorm.weight", l, 1);
            std::vector<float> gate_su;
            for (int j = 0; j < 7; ++j) {
                auto h = qtip::load_linear(checkpoint, l, names[j]);
                if (j == 4)
                    gate_su = h.su;
                for (auto &rank : ranks) {
                    Scope scope(rank->device);
                    rank->layers[l].weights[j].load(h, rank->rank, tp, rank->stream);
                    if (j == 5)
                        rank->layers[l].shared_su = !gate_su.empty() && gate_su == h.su;
                }
            }
            each([&](Rank &r) {
                auto &layer = r.layers[l];
                size_t n = size_t(max_batch) * context * (KH / tp) * D;
                layer.keys.ensure(n, r.stream);
                layer.values.ensure(n, r.stream);
            });
            std::cerr << "Loaded tensor-parallel layer " << l + 1 << "/88\n";
        }
        each([&](Rank &r) {
            r.reserve(max_batch, splits, false);
            size_t free = 0, total = 0;
            checked(cudaMemGetInfo(&free, &total), "TP memory");
            std::cerr << "TP rank " << r.rank << " GPU " << r.device << ": "
                      << double(total - free) / (1ull << 30) << " GiB allocated\n";
        });
        reset();
    }
    ~Impl() {
        for (auto &r : ranks) {
            Scope scope(r->device);
            (void)sync(r->stream);
        }
        graphs.clear();
    }
    template <class F> void each(F fn) {
        for (auto &r : ranks) {
            Scope scope(r->device);
            fn(*r);
        }
    }
    void vector(const std::string &name, int layer, int which) {
        auto view = checkpoint.tensor(name).floats(name);
        check(view.size() == H, "Invalid TP norm shape");
        std::vector<float> host(H);
        for (int i = 0; i < H; ++i) {
            host[i] = view[i];
            check(std::isfinite(host[i]), "Nonfinite TP norm");
        }
        each([&](Rank &r) {
            auto &dst = layer < 0 ? r.final_norm : which ? r.layers[layer].norm2 : r.layers[layer].norm1;
            upload(dst, host.data(), host.size(), r.stream);
            checked(sync(r.stream), "finish norm upload");
        });
    }
    void dense(const std::string &name, bool partition) {
        const auto &file = checkpoint.tensor(name);
        check(file.shape(name) == std::vector<size_t>{V, H}, "Invalid TP embedding/head shape");
        if (file.dtype(name) == "BF16") {
            auto bytes = file.bytes(name);
            const B16 *base = reinterpret_cast<const B16 *>(bytes.data());
            each([&](Rank &r) {
                int rows = partition ? V / tp : V;
                size_t offset = partition ? size_t(r.rank) * rows * H : 0;
                auto &dst = partition ? r.head : r.embeddings;
                upload(dst, base + offset, size_t(rows) * H, r.stream);
                checked(sync(r.stream), "finish dense upload");
            });
        } else {
            auto host = checkpoint.bf16(name);
            each([&](Rank &r) {
                int rows = partition ? V / tp : V;
                size_t offset = partition ? size_t(r.rank) * rows * H : 0;
                auto &dst = partition ? r.head : r.embeddings;
                upload(dst, reinterpret_cast<const B16 *>(host.data()) + offset, size_t(rows) * H, r.stream);
                checked(sync(r.stream), "finish converted dense upload");
            });
        }
    }
    template <class F> void gather(size_t n, ncclDataType_t dtype, F pointer) {
        if (tp == 1)
            return;
        if (peer_transport) {
            check(dtype == ncclUint8 || dtype == ncclFloat, "Unsupported peer collective datatype");
            size_t bytes = n * (dtype == ncclFloat ? sizeof(float) : 1);
            std::array<const void *, 2> send;
            std::array<void *, 2> receive;
            each([&](Rank &r) {
                auto buffers = pointer(r);
                send[r.rank] = buffers.first;
                receive[r.rank] = buffers.second;
            });
            peer_gather->run(send, receive, bytes);
            return;
        }
        nc(ncclGroupStart());
        for (auto &r : ranks) {
            Scope scope(r->device);
            auto [send, receive] = pointer(*r);
            nc(ncclAllGather(send, receive, n, dtype, r->comm, r->stream));
        }
        nc(ncclGroupEnd());
    }
    void project(int layer, int index, int rows, bool rz, bool input_rank_major, bool reuse_rotated,
                 bool large, int source, int target, bool rotate_into_b = false) {
        ProjectionProfile *timing = nullptr;
        if (large && !profile_path.empty() && !profile_reported) {
            profile.push_back(std::make_unique<ProjectionProfile>(ranks[0]->device, layer, index, rows));
            timing = profile.back().get();
        }
        auto mark = [&](int event) {
            if (timing) {
                Scope scope(ranks[0]->device);
                checked(cudaEventRecord(timing->events[event], ranks[0]->stream), "profile projection stage");
            }
        };
        mark(0);
        auto source_ptr = [&](Rank &r) -> B16 * {
            return source == 0 ? r.input.p : source == 1 ? r.a.p : r.b.p;
        };
        auto target_ptr = [&](Rank &r) -> B16 * {
            return target == 0 ? r.input.p : target == 1 ? r.a.p : r.b.p;
        };
        auto rotated_ptr = [&](Rank &r) -> B16 * { return rotate_into_b ? r.b.p : r.input.p; };
        int parts = large ? 1 : splits;
        if (!reuse_rotated)
            each([&](Rank &r) {
                auto &w = r.layers[layer].weights[index];
                check(!w.fp8.p, "Fast QTIP projection profile mismatch");
                qtip_prepare(w.matrix(true), source_ptr(r), r.scratch, rotated_ptr(r), r.stream, rows,
                             input_rank_major);
            });
        mark(1);
        each([&](Rank &r) {
            auto &w = r.layers[layer].weights[index];
            size_t n = size_t(rows) * w.local;
            float *out = r.z.p + r.rank * n;
            r.multiply(w, rotated_ptr(r), out, rows, parts, large,
                       timing && r.rank == 0 ? timing->events[2] : nullptr);
            if (rz)
                reduce_to_b16(out, nullptr, r.b.p + r.rank * n, n, w.local, parts, r.stream);
            else
                reduce_products(out, n, parts, r.stream);
        });
        mark(3);
        size_t n = size_t(rows) * ranks[0]->layers[layer].weights[index].local;
        if (rz)
            gather(n * sizeof(B16), ncclUint8,
                   [&](Rank &r) { return std::pair<void *, void *>{r.b.p + r.rank * n, r.b.p}; });
        else
            gather(n, ncclFloat,
                   [&](Rank &r) { return std::pair<void *, void *>{r.z.p + r.rank * n, r.z.p}; });
        mark(4);
        each([&](Rank &r) {
            auto &w = r.layers[layer].weights[index];
            if (rz)
                qtip_finish_b16(w.matrix(true), r.b, target_ptr(r), r.scratch, rows, r.stream, tp == 2);
            else
                qtip_finish(w.matrix(true), r.z, target_ptr(r), r.scratch, false, 1, r.stream, rows, tp == 2);
        });
        mark(5);
    }
    void report_profile() {
        if (profile.empty() || profile_reported)
            return;
        Scope scope(ranks[0]->device);
        const std::array<const char *, 5> stages = {"input_rht", "decode_weights", "gemm", "allgather",
                                                    "output_rht"};
        const std::array<const char *, 7> names = {"q", "k", "v", "o", "gate", "up", "down"};
        std::array<double, 5> totals{};
        std::map<std::string, std::array<double, 5>> details;
        for (const auto &record : profile)
            for (int i = 0; i < 5; ++i) {
                float ms = 0;
                checked(cudaEventElapsedTime(&ms, record->events[i], record->events[i + 1]),
                        "read prefill profile");
                totals[i] += ms / 1000.;
                details[names[record->index]][i] += ms / 1000.;
            }
        json result;
        result["rows"] = profile[0]->rows;
        result["rank"] = 0;
        result["tp"] = tp;
        for (int i = 0; i < 5; ++i) {
            result["seconds"][stages[i]] = totals[i];
            for (const auto &[name, times] : details)
                result["by_projection"][name][stages[i]] = times[i];
        }
        std::ofstream file(profile_path);
        file << result.dump(2) << '\n';
        check(bool(file), "Cannot write prefill profile");
        profile_reported = true;
        profile.clear();
    }
    void forward(int batch, int sequence, int base, bool prefill, int attention_bound = 0) {
        int rows = batch * sequence;
        bool large = prefill && rows >= 128;
        mark_decode("begin");
        each([&](Rank &r) { embedding_batch(r.embeddings, r.x, r.ids, rows, H, r.stream); });
        mark_decode("embedding");
        for (int l = 0; l < 88; ++l) {
            each([&](Rank &r) {
                auto &layer = r.layers[l];
                norm_batch(r.x, layer.norm1, r.input, rows, H, 1e-5f, r.stream);
            });
            mark_decode("norm1");
            each([&](Rank &r) {
                r.frozen(r.layers[l].weights[1], r.input, r.a.p + size_t(rows) * H, rows, large);
            });
            mark_decode("k");
            each([&](Rank &r) {
                r.frozen(r.layers[l].weights[2], r.input, r.a.p + size_t(rows) * (H + KH / tp * D), rows,
                         large);
            });
            mark_decode("v");
            project(l, 0, rows, false, false, false, large, 0, 1);
            mark_decode("q");
            each([&](Rank &r) {
                auto &layer = r.layers[l];
                B16 *k = r.a.p + size_t(rows) * H;
                B16 *v = r.a.p + size_t(rows) * (H + KH / tp * D);
                B16 *q = r.a.p + r.rank * (H / tp);
                rope_batch(q, k, r.positions.p + base, rows, sequence, QH / tp, KH / tp, D, 1000000.f,
                           r.stream, H);
                size_t offset = size_t(base) * context * (KH / tp) * D;
                cache_batch(k, v, layer.keys.p + offset, layer.values.p + offset, r.positions.p + base, batch,
                            sequence, context, KH / tp * D, r.stream);
                B16 *out = r.b.p + size_t(r.rank) * rows * (H / tp);
                if (prefill)
                    r.attention->run(q, layer.keys.p + offset, layer.values.p + offset, out,
                                     r.lengths.p + base, batch, sequence, context);
                else {
                    int maximum = 0;
                    for (int i = 0; i < batch; ++i)
                        maximum = std::max(maximum, position[i] + 1);
                    if (attention_bound)
                        maximum = attention_bound;
                    gqa_batch(q, layer.keys, layer.values, out, r.scores, r.positions, batch, context,
                              maximum, QH / tp, KH / tp, D, r.stream, H, r.attention_parts,
                              register_attention);
                }
            });
            mark_decode("attention");
            gather(size_t(rows) * (H / tp) * sizeof(B16), ncclUint8, [&](Rank &r) {
                return std::pair<void *, void *>{r.b.p + size_t(r.rank) * rows * (H / tp), r.b.p};
            });
            mark_decode("attention_gather");
            project(l, 3, rows, false, tp == 2, false, large, 2, 1);
            mark_decode("o");
            each([&](Rank &r) {
                residual(r.x, r.a, rows * H, r.stream);
                norm_batch(r.x, r.layers[l].norm2, r.input, rows, H, 1e-5f, r.stream);
            });
            mark_decode("norm2_residual");
            // Keep the rotated input in input; independent SU matrices are handled
            // below by preserving the unrotated normalization in a temporary view.
            bool shared = ranks[0]->layers[l].shared_su;
            project(l, 4, rows, true, false, false, large, 0, 1, !shared);
            mark_decode("gate");
            bool up_in_input = large && fused_prefill_hadamard();
            project(l, 5, rows, true, false, shared, large, 0, up_in_input ? 0 : 2);
            mark_decode("up");
            each(
                [&](Rank &r) { swiglu(r.a, up_in_input ? r.input.p : r.b.p, rows * I, INFINITY, r.stream); });
            mark_decode("swiglu");
            project(l, 6, rows, false, false, false, large, 1, 0);
            mark_decode("down");
            each([&](Rank &r) { residual(r.x, r.input, rows * H, r.stream); });
            mark_decode("residual");
        }
    }
    void mark_decode(const char *name) {
        if (record_decode)
            decode_marks.push_back(std::make_unique<DecodeMark>(ranks[0]->device, name, ranks[0]->stream));
    }
    void report_decode() {
        if (!record_decode)
            return;
        Scope scope(ranks[0]->device);
        json result{{"batch", active},
                    {"tp", tp},
                    {"captured", false},
                    {"position", *std::max_element(position.begin(), position.end())},
                    {"transport", peer_transport ? "peer" : "nccl"}};
        std::map<std::string, double> totals;
        for (size_t i = 1; i < decode_marks.size(); ++i) {
            float ms = 0;
            checked(cudaEventElapsedTime(&ms, decode_marks[i - 1]->event, decode_marks[i]->event),
                    "read decode stage timing");
            totals[decode_marks[i]->name] += ms / 1000.;
        }
        result["seconds"] = totals;
        std::ofstream file(decode_profile_path);
        file << result.dump(2) << '\n';
        check(bool(file), "Cannot write decode profile");
        decode_marks.clear();
        record_decode = false;
        decode_profiled = true;
    }
    void output(int batch, int sequence) {
        each([&](Rank &r) {
            if (sequence > 1)
                select_rows(r.x, r.input, r.selected, batch, H, r.stream);
            norm_batch(sequence > 1 ? r.input.p : r.x.p, r.final_norm, r.input, batch, H, 1e-5f, r.stream);
            bf16_multiply_batch(r.head, r.input, r.logits.p + size_t(r.rank) * batch * (V / tp), H, V / tp,
                                batch, r.stream);
        });
        gather(size_t(batch) * (V / tp), ncclFloat, [&](Rank &r) {
            return std::pair<void *, void *>{r.logits.p + size_t(r.rank) * batch * (V / tp), r.logits.p};
        });
    }
    std::vector<Vec> read_logits(int batch) {
        auto &r = *ranks[0];
        Scope scope(r.device);
        checked(d2h(host_logits.p, r.logits, size_t(batch) * V * 4, r.stream), "read TP logits");
        checked(sync(r.stream), "finish TP forward");
        std::vector<Vec> out(batch, Vec(V));
        for (int b = 0; b < batch; ++b)
            for (int rank = 0; rank < tp; ++rank) {
                auto *p = host_logits.p + (size_t(rank) * batch + b) * (V / tp);
                std::copy(p, p + V / tp, out[b].begin() + rank * (V / tp));
            }
        for (const auto &row : out)
            for (float x : row)
                check(std::isfinite(x), "Nonfinite TP logits");
        return out;
    }
    std::vector<Vec> finish(int batch, int sequence) {
        output(batch, sequence);
        return read_logits(batch);
    }
    void reserve(int rows, int parts, bool prefill) {
        bool changed = false;
        each([&](Rank &r) { changed = r.reserve(rows, parts, prefill) || changed; });
        if (changed)
            graphs.clear();
    }
    void compute_step(int bound) {
        each([&](Rank &r) {
            checked(h2d(r.ids, host_ids.p, size_t(active) * 4, r.stream), "upload decode IDs");
            checked(h2d(r.positions, host_positions.p, size_t(active) * 4, r.stream),
                    "upload decode positions");
        });
        forward(active, 1, 0, false, bound);
        output(active, 1);
        mark_decode("output");
    }
    void graph_step(int bound) {
        auto key = std::pair{active, bound};
        auto &primary = *ranks[0];
        if (!graphs.contains(key)) {
            auto graph = std::make_unique<Graph>(primary.device);
            {
                Scope scope(primary.device);
                checked(cudaStreamBeginCapture(primary.stream, cudaStreamCaptureModeGlobal),
                        "begin TP graph capture");
                if (tp == 2)
                    checked(cudaEventRecord(graph_join.value, primary.stream), "capture TP graph fork");
            }
            try {
                if (tp == 2) {
                    Scope scope(ranks[1]->device);
                    checked(cudaStreamWaitEvent(ranks[1]->stream, graph_join.value, 0),
                            "join secondary graph stream");
                }
                compute_step(bound);
                if (tp == 2) {
                    Scope scope(ranks[1]->device);
                    checked(cudaEventRecord(graph_done.value, ranks[1]->stream),
                            "record secondary graph completion");
                }
                Scope scope(primary.device);
                if (tp == 2)
                    checked(cudaStreamWaitEvent(primary.stream, graph_done.value, 0), "close TP graph fork");
                checked(cudaStreamEndCapture(primary.stream, &graph->graph), "end TP graph capture");
                checked(cudaGraphInstantiate(&graph->executable, graph->graph, nullptr, nullptr, 0),
                        "instantiate TP graph");
            } catch (...) {
                Scope scope(primary.device);
                cudaGraph_t aborted{};
                (void)cudaStreamEndCapture(primary.stream, &aborted);
                if (aborted)
                    (void)cudaGraphDestroy(aborted);
                throw;
            }
            graphs.emplace(key, std::move(graph));
        }
        Scope scope(primary.device);
        checked(cudaGraphLaunch(graphs.at(key)->executable, primary.stream), "launch TP decode graph");
    }
    void reset() {
        each([&](Rank &r) { checked(sync(r.stream), "reset TP stream"); });
        active = 0;
        position.clear();
        failed = false;
    }
    std::vector<Vec> step(const std::vector<int> &tokens) {
        check(!failed, "Reset after TP inference failure");
        check(!tokens.empty() && int(tokens.size()) <= max_batch, "Invalid TP decode batch");
        if (!active) {
            active = tokens.size();
            position.assign(active, 0);
        }
        check(int(tokens.size()) == active, "Decode batch changed without prefill/reset");
        for (int b = 0; b < active; ++b) {
            check(tokens[b] >= 0 && tokens[b] < V && position[b] < context,
                  "Token or context outside TP limits");
            host_ids.p[b] = tokens[b];
            host_positions.p[b] = position[b];
        }
        reserve(active, splits, false);
        try {
            record_decode = !decode_profile_path.empty() && !decode_profiled;
            int maximum = *std::max_element(position.begin(), position.end()) + 1;
            int bound = std::min(context, ((maximum + 127) / 128) * 128);
            if (graphs_enabled && warmed_batches.contains(active) && !record_decode)
                graph_step(bound);
            else {
                compute_step(bound);
                warmed_batches.insert(active);
            }
            auto result = read_logits(active);
            report_decode();
            for (auto &p : position)
                ++p;
            return result;
        } catch (...) {
            failed = true;
            throw;
        }
    }
    std::vector<Vec> prefill(const std::vector<std::vector<int>> &tokens) {
        reset();
        check(!tokens.empty() && int(tokens.size()) <= max_batch, "Invalid TP prefill batch");
        active = tokens.size();
        position.assign(active, 0);
        int maximum = 0;
        for (int b = 0; b < active; ++b) {
            check(!tokens[b].empty() && int(tokens[b].size()) <= context, "Invalid TP prompt length");
            maximum = std::max(maximum, int(tokens[b].size()));
            host_lengths.p[b] = tokens[b].size();
            host_positions.p[b] = 0;
        }
        int sequence = std::min(context, ((maximum + 63) / 64) * 64);
        each([&](Rank &r) {
            check(r.tensor, "Fast bulk prefill requires BF16 tensor cores (sm_80+)");
            checked(h2d(r.positions, host_positions.p, size_t(active) * 4, r.stream),
                    "upload prefill positions");
            checked(h2d(r.lengths, host_lengths.p, size_t(active) * 4, r.stream), "upload prefill lengths");
        });
        // A 16k-row arena keeps batch-8/8k prompts within A6000 memory while
        // processing multiple full sequences per cuBLAS/cuDNN invocation.
        int group = std::max(1, std::min(active, 16384 / sequence));
        Pinned<int> packed;
        packed.ensure(size_t(group) * sequence);
        std::vector<Vec> result;
        try {
            for (int base = 0; base < active; base += group) {
                int count = std::min(group, active - base);
                int rows = count * sequence;
                std::fill(packed.p, packed.p + rows, 2);
                for (int b = 0; b < count; ++b) {
                    std::copy(tokens[base + b].begin(), tokens[base + b].end(),
                              packed.p + size_t(b) * sequence);
                    host_selected.p[b] = b * sequence + tokens[base + b].size() - 1;
                }
                reserve(rows, 1, true);
                each([&](Rank &r) {
                    checked(h2d(r.ids, packed.p, size_t(rows) * 4, r.stream), "upload prefill IDs");
                    checked(h2d(r.selected, host_selected.p, size_t(count) * 4, r.stream),
                            "upload final row indices");
                });
                forward(count, sequence, base, true);
                auto rows_out = finish(count, sequence);
                report_profile();
                result.insert(result.end(), std::make_move_iterator(rows_out.begin()),
                              std::make_move_iterator(rows_out.end()));
            }
            for (int b = 0; b < active; ++b)
                position[b] = tokens[b].size();
            each([&](Rank &r) { r.decoded.clear(); });
            return result;
        } catch (...) {
            failed = true;
            throw;
        }
    }
};
TensorModel::TensorModel(const fs::path &p, int device, int context, int tp, int batch)
    : impl(std::make_unique<Impl>(p, device, context, tp, batch)) {}
TensorModel::~TensorModel() = default;
std::vector<Vec> TensorModel::prefill(const std::vector<std::vector<int>> &tokens) {
    return impl->prefill(tokens);
}
std::vector<Vec> TensorModel::step(const std::vector<int> &tokens) { return impl->step(tokens); }
void TensorModel::reset() { impl->reset(); }
} // namespace behemoth::cuda_runtime
