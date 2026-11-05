#include "fast_prefill.hpp"
#include "behemoth/cuda.hpp"
#include <cmath>
// The frontend's JSON-free mode stubs these macros. Preserve the application's
// vendored JSON definitions and ABI while building the private cuDNN graph.
#pragma push_macro("NLOHMANN_JSON_SERIALIZE_ENUM")
#pragma push_macro("NLOHMANN_DEFINE_TYPE_INTRUSIVE")
#undef NLOHMANN_JSON_SERIALIZE_ENUM
#undef NLOHMANN_DEFINE_TYPE_INTRUSIVE
#include <cudnn_frontend.h>
#pragma pop_macro("NLOHMANN_DEFINE_TYPE_INTRUSIVE")
#pragma pop_macro("NLOHMANN_JSON_SERIALIZE_ENUM")
#include <map>
#include <tuple>
#include <unordered_map>
namespace behemoth::cuda_runtime {
namespace fe = cudnn_frontend;
namespace {
void checked(cudnnStatus_t error, const char *stage) {
    if (error != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(stage) + ": " + cudnnGetErrorString(error));
}
void checked(fe::error_t error, const char *stage) {
    if (!error.is_good())
        throw std::runtime_error(std::string(stage) + ": " + error.get_message());
}
} // namespace
struct FastPrefill::Impl {
    int device, qh, kh, dim, query_stride;
    Stream stream;
    cudnnHandle_t handle{};
    glm53::gpu::Buffer<uint8_t> workspace;
    std::map<std::tuple<int, int, int>, std::shared_ptr<fe::graph::Graph>> plans;
    Impl(int q, int k, int d, int stride, Stream s) : qh(q), kh(k), dim(d), query_stride(stride), stream(s) {
        glm53::gpu::checked(cudaGetDevice(&device), "select cuDNN device");
        checked(cudnnCreate(&handle), "create cuDNN handle");
        checked(cudnnSetStream(handle, s), "set cuDNN stream");
    }
    ~Impl() {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        (void)cudaStreamSynchronize(stream);
        plans.clear();
        workspace.clear();
        (void)cudnnDestroy(handle);
        (void)cudaSetDevice(previous);
    }
    void run(const B16 *q, const B16 *k, const B16 *v, B16 *out, const int *lengths, int batch, int sequence,
             int context) {
        auto key = std::tuple{batch, sequence, context};
        if (!plans.contains(key)) {
            auto graph = std::make_shared<fe::graph::Graph>();
            graph->set_io_data_type(fe::DataType_t::BFLOAT16)
                .set_intermediate_data_type(fe::DataType_t::FLOAT)
                .set_compute_data_type(fe::DataType_t::FLOAT);
            int64_t b = batch, s = sequence, c = context;
            auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("Q")
                                       .set_uid(1)
                                       .set_dim({b, qh, s, dim})
                                       .set_stride({s * query_stride, dim, query_stride, 1}));
            auto K = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("K")
                                       .set_uid(2)
                                       .set_dim({b, kh, s, dim})
                                       .set_stride({c * kh * dim, dim, kh * dim, 1}));
            auto V = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("V")
                                       .set_uid(3)
                                       .set_dim({b, kh, s, dim})
                                       .set_stride({c * kh * dim, dim, kh * dim, 1}));
            auto L = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("lengths")
                                       .set_uid(5)
                                       .set_data_type(fe::DataType_t::INT32)
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1}));
            auto options = fe::graph::SDPA_attributes()
                               .set_name("causal_gqa")
                               .set_is_inference(true)
                               .set_causal_mask(true)
                               .set_attn_scale(1.f / std::sqrt(float(dim)))
                               .set_padding_mask(true)
                               .set_seq_len_q(L)
                               .set_seq_len_kv(L);
            auto [O, stats] = graph->sdpa(Q, K, V, options);
            O->set_output(true)
                .set_uid(4)
                .set_dim({b, qh, s, dim})
                .set_stride({s * qh * dim, dim, qh * dim, 1});
            checked(graph->build(handle, {fe::HeurMode_t::A}), "build cuDNN GQA prefill plan");
            plans.emplace(key, std::move(graph));
        }
        auto &graph = plans.at(key);
        workspace.ensure(graph->get_workspace_size(), stream);
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> pack{{1, const_cast<B16 *>(q)},
                                                                             {2, const_cast<B16 *>(k)},
                                                                             {3, const_cast<B16 *>(v)},
                                                                             {4, out},
                                                                             {5, const_cast<int *>(lengths)}};
        checked(graph->execute(handle, pack, workspace), "execute cuDNN GQA prefill");
    }
};
FastPrefill::FastPrefill(int q, int k, int d, int stride, Stream s)
    : impl(std::make_unique<Impl>(q, k, d, stride, s)) {}
FastPrefill::~FastPrefill() = default;
void FastPrefill::run(const B16 *q, const B16 *k, const B16 *v, B16 *out, const int *lengths, int b, int s,
                      int c) {
    impl->run(q, k, v, out, lengths, b, s, c);
}
} // namespace behemoth::cuda_runtime
