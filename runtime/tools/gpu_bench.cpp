#include "../tests/fixtures.hpp"
#include "behemoth/gpu.hpp"
#include "behemoth/model.hpp"
#include <iostream>
using namespace behemoth;
int main(int argc, char **argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--topology") {
            tensor_parallel_smoke();
            return 0;
        }
        int length = argc > 1 ? std::stoi(argv[1]) : 4096,
            requests = argc > 2 ? std::stoi(argv[2]) : 1,
            queries = argc > 3 ? std::stoi(argv[3]) : 1;
        if (length < 1 || length > 131072 || requests < 1 || queries < 1 || queries > length ||
            requests > 1024 / queries)
            throw std::invalid_argument(
                "Usage: behemoth_bench CONTEXT REQUESTS QUERY_TOKENS; total query tokens <=1024");
        HIP_CHECK(hipSetDevice(0));
        verify_device();
        std::vector<int> lengths(size_t(requests), length);
        std::vector<std::vector<int>> positions(size_t(requests),
                                                std::vector<int>(size_t(queries), 0));
        for (auto &p : positions)
            for (int i = 0; i < queries; ++i)
                p[size_t(i)] = length - queries + i;
        auto f = test::fixture(lengths, positions);
        GpuMetadata metadata;
        metadata.upload(f.layout);
        DeviceBuffer<uint8_t> q, k, v;
        q.upload(f.data.q);
        k.upload(f.data.k);
        v.upload(f.data.v);
        DeviceBuffer<float> qs, ks, vs;
        qs.upload(f.data.q_scale);
        ks.upload(f.data.k_scale);
        vs.upload(f.data.v_scale);
        DeviceBuffer<BFloat16> out(size_t(requests) * size_t(queries) * 48 * 128);
        int partition = queries == 1 && length > 1024 ? 1024 : 0;
        DeviceBuffer<std::byte> workspace(attention_workspace_bytes(f.layout, partition));
        AttentionDeviceView a{q.data(),  qs.data(), k.data(),  v.data(),
                              ks.data(), vs.data(), out.data()};
        hipStream_t stream{};
        HIP_CHECK(hipStreamCreate(&stream));
        hipEvent_t start{}, end{};
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&end));
        for (int i = 0; i < 5; ++i)
            paged_attention(metadata, a, workspace.data(), workspace.size(), partition, stream);
        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < 30; ++i)
            paged_attention(metadata, a, workspace.data(), workspace.size(), partition, stream);
        HIP_CHECK(hipEventRecord(end, stream));
        HIP_CHECK(hipEventSynchronize(end));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, end));
        ms /= 30;
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        int version = 0;
        HIP_CHECK(hipRuntimeGetVersion(&version));
        nlohmann::json result = {{"gpu", prop.name},
                                 {"arch", prop.gcnArchName},
                                 {"wave_size", prop.warpSize},
                                 {"hip_version", version},
                                 {"context", length},
                                 {"requests", requests},
                                 {"query_tokens", queries},
                                 {"milliseconds", ms},
                                 {"query_tokens_per_second", 1000.0 * requests * queries / ms},
                                 {"workspace_bytes", workspace.size()},
                                 {"precision", "OCP E4M3 / BF16 / FP32"}};
        auto got = out.download();
        std::vector<float> values(got.size());
        for (size_t i = 0; i < got.size(); ++i)
            values[i] = float(got[i]);
        auto reference = reference_attention(f.layout, f.data, true, partition);
        auto error = compare_attention(values, reference.output);
        result["correctness_passed"] = error.pass;
        result["rms_error"] = error.rms_error;
        result["max_abs_error"] = error.max_abs;
        std::cout << result.dump(2) << '\n';
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(end));
        HIP_CHECK(hipStreamDestroy(stream));
        return error.pass ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
