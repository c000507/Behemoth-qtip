// Micro bisect for the Qwen runtime kernel hang: stage-by-stage pipeline pieces.
// usage: qwen_micro <model_dir> <stage 1..6>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "behemoth/qtip.hpp"
#include "behemoth/qtip_gpu.hpp"
#include "behemoth/qwen_kernels.hpp"
#include "behemoth/qwen_model.hpp"

using namespace behemoth;
static float b16(uint16_t b) { return float(BFloat16::from_bits(b)); }

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::filesystem::path dir = argv[1];
    int stage = std::atoi(argv[2]);
    QwenRuntimeConfig c = QwenRuntimeConfig::read(dir);
    const int H = c.hidden, hd = c.hd, nq = c.nq, nkv = c.nkv;
    const int rows = 8;
    std::string fn = "model-layer-00003.safetensors";
    SafeTensorFile f(dir / fn);
    std::string b = "model.layers.3.self_attn.q_proj.";
    qtip::HostLinear h;
    {
        auto tre = f.bytes(b + "trellis");
        h.trellis.resize(tre.size() / 2);
        std::memcpy(h.trellis.data(), tre.data(), tre.size());
        auto su = f.floats(b + "SU"), sv = f.floats(b + "SV");
        h.su.resize(su.size());
        for (size_t i = 0; i < su.size(); ++i) h.su[i] = su[i];
        h.sv.resize(sv.size());
        for (size_t i = 0; i < sv.size(); ++i) h.sv[i] = sv[i];
        auto tl = f.bytes(b + "tlut");
        h.tlut.resize(tl.size() / 2);
        std::memcpy(h.tlut.data(), tl.data(), tl.size());
        h.in = int(h.su.size());
        h.out = int(h.sv.size());
    }
    qtip::Codebook cb = qtip::make_codebook(std::span<const uint16_t>(h.tlut));
    qtip::GpuCodebook lut;
    lut.upload(cb);
    qtip::GpuLinear gl;
    gl.upload({std::span<const uint16_t>(h.trellis)}, h.in);
    hipStream_t s;
    HIP_CHECK(hipStreamCreate(&s));
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    DeviceBuffer<uint16_t> x(size_t(rows) * H), u(size_t(rows) * H);
    DeviceBuffer<float> su, sv, z(size_t(rows) * h.out), partial;
    su.upload(h.su);
    sv.upload(h.sv);
    {
        auto v = x.download();
        (void)v;
    }
    std::vector<uint16_t> hx(size_t(rows) * H);
    for (auto &v : hx) v = uint16_t(0x3d80);  // ~1.0 bf16 pattern-ish
    HIP_CHECK(hipMemcpy(x.data(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    std::printf("stage 1: input_transform\n");
    q_input_transform(x.data(), rows, H, nullptr, c.eps, su.data(), H, u.data(), H, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("  ok\n");
    if (stage < 2) return 0;
    std::printf("stage 2: matmul\n");
    int slices = qtip::matmul_slices(gl.view(), rows);
    partial.resize(size_t(slices) * rows * h.out);
    qtip::matmul(gl.view(), lut.lut.data(), reinterpret_cast<const BFloat16 *>(u.data()), H, rows,
                 partial.data(), slices, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("  ok (slices %d)\n", slices);
    if (stage < 3) return 0;
    std::printf("stage 3: sum slices\n");
    q_sum_slices(partial.data(), slices, size_t(rows) * h.out, z.data(), s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("  ok\n");
    if (stage < 4) return 0;
    std::printf("stage 4: hadamard %d\n", h.out);
    q_hadamard_rows(z.data(), rows, h.out, h.out, false, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("  ok\n");
    if (stage < 5) return 0;
    std::printf("stage 5: sv_cast\n");
    DeviceBuffer<uint16_t> y(size_t(rows) * h.out);
    q_sv_cast(z.data(), sv.data(), y.data(), rows, h.out, h.out, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("  ok\n");
    if (stage < 6) return 0;
    std::printf("stage 6: attention kernels\n");
    if (std::getenv("ROPE_ONLY")) {
        DeviceBuffer<uint16_t> q3(size_t(rows) * nq * 2 * hd), k3(size_t(rows) * nkv * hd),
            v3(size_t(rows) * nkv * hd), g3(size_t(rows) * nq * hd), kc3(size_t(64) * nkv * hd),
            vc3(size_t(64) * nkv * hd);
        std::vector<uint16_t> qn(hd, 0x3f80), kn(hd, 0x3f80);
        std::printf("  rope only: q3=%p k3=%p v3=%p g3=%p kc3=%p vc3=%p qn=%p kn=%p\n",
                    (void *)q3.data(), (void *)k3.data(), (void *)v3.data(), (void *)g3.data(),
                    (void *)kc3.data(), (void *)vc3.data(), (void *)qn.data(), (void *)kn.data());
        q_qk_norm_rope(q3.data(), k3.data(), v3.data(), g3.data(), kc3.data(), vc3.data(), rows, 0,
                       nq, nkv, hd, c.rot_dim, c.theta, qn.data(), kn.data(), c.eps, s);
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("  rope ok\n");
        DeviceBuffer<uint16_t> ao3(size_t(rows) * nq * hd);
        std::printf("  attn only\n");
        q_attn(q3.data(), kc3.data(), vc3.data(), g3.data(), ao3.data(), rows, 0, nq, nkv, hd,
               1.0f / std::sqrt(float(hd)), s);
        HIP_CHECK(hipStreamSynchronize(s));
        std::printf("  attn ok\n");
        return 0;
    }
    DeviceBuffer<uint16_t> q(size_t(rows) * nq * 2 * hd), k(size_t(rows) * nkv * hd),
        v2(size_t(rows) * nkv * hd), g(size_t(rows) * nq * hd), ao(size_t(rows) * nq * hd);
    DeviceBuffer<uint16_t> kc(size_t(64) * nkv * hd), vc(size_t(64) * nkv * hd);
    {
        auto qv = q.download(), kv = k.download(), vv = v2.download();
        HIP_CHECK(hipMemcpy(q.data(), qv.data(), qv.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(k.data(), kv.data(), kv.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(v2.data(), vv.data(), vv.size() * 2, hipMemcpyHostToDevice));
    }
    std::vector<uint16_t> qn(hd), kn(hd);
    q_qk_norm_rope(q.data(), k.data(), v2.data(), g.data(), kc.data(), vc.data(), rows, 0, nq, nkv,
                   hd, c.rot_dim, c.theta, qn.data(), kn.data(), c.eps, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("  rope ok\n");
    q_attn(q.data(), kc.data(), vc.data(), g.data(), ao.data(), rows, 0, nq, nkv, hd,
           1.0f / std::sqrt(float(hd)), s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::printf("  attn ok\n");
    return 0;
}
