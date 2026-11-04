#include "gpu.hpp"
#include "glm53.hpp"
#include "gpu_decode.hpp"
#include <cmath>
#include <hip/hip_runtime.h>

namespace glm53 {
namespace {
struct Projector {
    int device;
    hipStream_t stream{};
    std::unique_ptr<qtip::Gemm> gemm;
    qtip::DBuf<uint16_t> weights, input;
    qtip::DBuf<uint8_t> packed, fp8_scales;
    qtip::DBuf<float> output, scales;
    qtip::HBuf<float> result;
    explicit Projector(int d) : device(d) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipStreamCreate(&stream));
        gemm = std::make_unique<qtip::Gemm>(stream);
    }
    ~Projector() {
        (void)hipSetDevice(device);
        gemm.reset();
        weights.free();
        input.free();
        output.free();
        packed.free();
        fp8_scales.free();
        scales.free();
        (void)hipStreamDestroy(stream);
    }
    Vec operator()(const Weights &w, const std::string &base, const Vec &x) {
        HIP_CHECK(hipSetDevice(device));
        const bool nvfp4 = w.has(base + ".weight_packed");
        const std::string name = nvfp4 ? base + ".weight_packed" : w.has(base) ? base : base + ".weight";
        const auto &file = w.file(name);
        auto shape = file.shape(name);
        auto raw = file.bytes(name);
        check(shape.size() == 2 && shape[0] > 0 && shape[0] <= INT32_MAX && shape[1] <= INT32_MAX / 2,
              "GPU linear geometry outside supported range");
        int rows = int(shape[0]), cols = int(shape[1]) * (nvfp4 ? 2 : 1);
        check(cols > 0 && size_t(cols) == x.size(), "GPU linear/input shape mismatch");
        std::string dtype = file.dtype(name);
        weights.ensure(size_t(rows) * cols);
        input.ensure(x.size());
        output.ensure(rows);
        std::vector<float> host_scales;
        if (dtype == "BF16" && !nvfp4) {
            HIP_CHECK(hipMemcpyAsync(weights, raw.data(), raw.size(), hipMemcpyHostToDevice, stream));
        } else {
            packed.ensure(raw.size());
            HIP_CHECK(hipMemcpyAsync(packed, raw.data(), raw.size(), hipMemcpyHostToDevice, stream));
            int format = 0;
            bool block = false;
            float global = 1;
            if (nvfp4) {
                check(dtype == "U8" && cols % 16 == 0, "invalid NVFP4 matrix");
                format = 2;
                const auto &sf = w.file(base + ".weight_scale");
                check(sf.dtype(base + ".weight_scale") == "F8_E4M3" &&
                          sf.shape(base + ".weight_scale") ==
                              std::vector<size_t>{size_t(rows), size_t(cols / 16)},
                      "invalid NVFP4 block-scale geometry");
                auto raw_scales = sf.bytes(base + ".weight_scale");
                fp8_scales.ensure(raw_scales.size());
                HIP_CHECK(hipMemcpyAsync(fp8_scales, raw_scales.data(), raw_scales.size(),
                                         hipMemcpyHostToDevice, stream));
                auto gs = w.tensor(base + ".weight_global_scale");
                check(gs.data.size() == 1 && gs.data[0] > 0, "invalid NVFP4 global scale");
                global = gs.data[0];
            } else if (dtype == "F8_E4M3") {
                format = 1;
                block = w.has(base + ".weight_scale_inv");
                auto s = w.tensor(base + (block ? ".weight_scale_inv" : ".weight_scale"));
                check(block ? s.shape ==
                                  std::vector<size_t>{size_t((rows + 127) / 128), size_t((cols + 127) / 128)}
                            : s.data.size() == size_t(rows),
                      "invalid FP8 scale geometry");
                host_scales = std::move(s.data);
                for (float a : host_scales)
                    check(a >= 0 && std::isfinite(a), "invalid FP8 scale");
                scales.ensure(host_scales.size());
                HIP_CHECK(hipMemcpyAsync(scales, host_scales.data(), host_scales.size() * 4,
                                         hipMemcpyHostToDevice, stream));
            } else if (dtype == "F16")
                format = 3;
            else
                check(dtype == "F32", "unsupported GPU linear dtype");
            decode_weights(packed, scales, fp8_scales, weights, rows, cols, format, block, global, stream);
        }
        std::vector<uint16_t> b(x.size());
        for (size_t i = 0; i < b.size(); ++i)
            b[i] = behemoth::BFloat16(x[i]).bits;
        HIP_CHECK(hipMemcpyAsync(input, b.data(), b.size() * 2, hipMemcpyHostToDevice, stream));
        gemm->run(true, false, rows, 1, cols, weights, cols, input, cols, output, rows, qtip::OutT::F32);
        result.ensure(rows);
        HIP_CHECK(hipMemcpyAsync(result, output, size_t(rows) * 4, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return Vec(result.p, result.p + rows);
    }
};
} // namespace
std::function<Vec(const Weights &, const std::string &, const Vec &)> gpu_projector(int device) {
    auto p = std::make_shared<Projector>(device);
    return [p](const Weights &w, const std::string &name, const Vec &x) { return (*p)(w, name, x); };
}
} // namespace glm53
