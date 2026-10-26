// Dense QTIP prefill matmul (PREFILL_SCHED_REPORT.md): hipBLASLt bf16 x bf16 -> fp32 GEMMs on
// shards decoded by qtip::decode_dense. Separate from qtip.hip so that the kernels there build
// without hipBLASLt (the fine-tuner's HIP bridge compiles qtip.hip alone).
#include "behemoth/qtip_gpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>

namespace behemoth::qtip {
namespace {
void blas_check(hipblasStatus_t s, const char *what) {
    if (s != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": hipBLASLt status " + std::to_string(int(s)));
}
#define QTIP_BLAS_CHECK(x) blas_check((x), #x)
struct DensePlan {
    hipblasLtMatmulDesc_t op{};
    hipblasLtMatrixLayout_t a{}, b{}, c{};
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace = 0;
    DensePlan() = default;
    DensePlan(const DensePlan &) = delete;
    DensePlan &operator=(const DensePlan &) = delete;
    ~DensePlan() {
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
} // namespace
struct DenseMatmul::State {
    hipblasLtHandle_t handle{};
    int device = 0;
    // (m, n, k, ldb, ldc) -> plan
    std::map<std::tuple<int, int, int, int, int>, std::unique_ptr<DensePlan>> plans;
    // autotuned algorithms: (m, k, ldb, ldc) -> [(n, algo)]
    std::map<std::tuple<int, int, int, int>, std::vector<std::pair<int, hipblasLtMatmulAlgo_t>>> tuned;
    DeviceBuffer<std::byte> workspace;
    ~State() {
        plans.clear();
        if (handle)
            hipblasLtDestroy(handle);
    }
    // Layouts of D[m][n] (column-major, ld ldc) = A^T B: A = decoded weight bf16 (k x m, ld k),
    // B = x bf16 (k x n, ld ldb), fp32 output and accumulation; up to `want` heuristic results.
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates(DensePlan &p, int m, int n, int k, int ldb,
                                                             int ldc, int want) {
        QTIP_BLAS_CHECK(hipblasLtMatmulDescCreate(&p.op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t trans = HIPBLAS_OP_T, none = HIPBLAS_OP_N;
        QTIP_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans,
                                                        sizeof(trans)));
        QTIP_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSB, &none,
                                                        sizeof(none)));
        QTIP_BLAS_CHECK(hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16BF, uint64_t(k), uint64_t(m), k));
        QTIP_BLAS_CHECK(hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16BF, uint64_t(k), uint64_t(n), ldb));
        QTIP_BLAS_CHECK(hipblasLtMatrixLayoutCreate(&p.c, HIP_R_32F, uint64_t(m), uint64_t(n), ldc));
        hipblasLtMatmulPreference_t pref{};
        QTIP_BLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        size_t max_workspace = workspace.size();
        QTIP_BLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));
        std::vector<hipblasLtMatmulHeuristicResult_t> results(size_t(std::max(want, 1)));
        int count = 0;
        const auto status = hipblasLtMatmulAlgoGetHeuristic(handle, p.op, p.a, p.b, p.c, p.c, pref, want,
                                                            results.data(), &count);
        hipblasLtMatmulPreferenceDestroy(pref);
        if (status != HIPBLAS_STATUS_SUCCESS)
            count = 0;
        results.resize(size_t(std::max(count, 0)));
        std::erase_if(results, [](const auto &r) { return r.state != HIPBLAS_STATUS_SUCCESS; });
        if (results.empty())
            throw std::runtime_error("QTIP dense prefill: no hipBLASLt bf16->fp32 algorithm for m=" +
                                     std::to_string(m) + " n=" + std::to_string(n) +
                                     " k=" + std::to_string(k));
        return results;
    }
    DensePlan &plan(int m, int n, int k, int ldb, int ldc) {
        auto key = std::make_tuple(m, n, k, ldb, ldc);
        auto it = plans.find(key);
        if (it != plans.end())
            return *it->second;
        auto p = std::make_unique<DensePlan>();
        auto t = tuned.find(std::make_tuple(m, k, ldb, ldc));
        const auto results = candidates(*p, m, n, k, ldb, ldc, t == tuned.end() ? 1 : 64);
        size_t pick = 0;
        if (t != tuned.end() && !t->second.empty()) {
            // the algorithm tuned at the nearest batch size >= n (else the largest), if the
            // heuristic offers it for this shape
            const std::pair<int, hipblasLtMatmulAlgo_t> *best = nullptr;
            for (const auto &e : t->second)
                if (e.first >= n && (!best || e.first < best->first))
                    best = &e;
            if (!best)
                for (const auto &e : t->second)
                    if (!best || e.first > best->first)
                        best = &e;
            for (size_t i = 0; i < results.size(); ++i)
                if (std::memcmp(results[i].algo.data, best->second.data, sizeof(best->second.data)) == 0) {
                    pick = i;
                    break;
                }
        }
        p->algo = results[pick].algo;
        p->workspace = results[pick].workspaceSize;
        auto &ref = *p;
        plans.emplace(key, std::move(p));
        return ref;
    }
};
DenseMatmul::DenseMatmul() : s_(std::make_unique<State>()) {
    HIP_CHECK(hipGetDevice(&s_->device));
    QTIP_BLAS_CHECK(hipblasLtCreate(&s_->handle));
    s_->workspace.resize(size_t(32) << 20);
}
DenseMatmul::~DenseMatmul() = default;
DenseMatmul::DenseMatmul(DenseMatmul &&) noexcept = default;
DenseMatmul &DenseMatmul::operator=(DenseMatmul &&) noexcept = default;
void DenseMatmul::run(LinearView l, const uint32_t *lut, const BFloat16 *x, int ldx, int rows,
                      float *out, BFloat16 *scratch, size_t scratch_elements, hipStream_t stream) {
    if (rows < 1 || l.rows % 32 || l.in % 32 || ldx < l.in)
        throw std::invalid_argument("QTIP dense matmul shape");
    const int piece = int(std::min<size_t>(size_t(l.rows), scratch_elements / size_t(l.in) / 32 * 32));
    if (piece < 32)
        throw std::invalid_argument("QTIP dense matmul scratch too small");
    const size_t words_per_block_row = size_t(l.in / 32) * 128;
    for (int r0 = 0; r0 < l.rows; r0 += piece) {
        const int m = std::min(piece, l.rows - r0);
        decode_dense(LinearView{l.trellis + size_t(r0 / 32) * words_per_block_row, l.in, m}, lut,
                     scratch, stream);
        gemm(m, l.in, x, ldx, rows, out + r0, l.rows, scratch, stream);
    }
}
double DenseMatmul::tune(int m, int in, const BFloat16 *x, int ldx, int rows, float *out, int ldo,
                         const BFloat16 *w, hipStream_t stream, int max_candidates) {
    DensePlan probe;
    const auto results = s_->candidates(probe, m, rows, in, ldx, ldo, std::clamp(max_candidates, 1, 64));
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    const float one = 1.f, zero = 0.f;
    float best_ms = INFINITY;
    size_t best = 0;
    try {
        for (size_t i = 0; i < results.size(); ++i) {
            auto call = [&] {
                QTIP_BLAS_CHECK(hipblasLtMatmul(s_->handle, probe.op, &one, w, probe.a, x, probe.b, &zero, out,
                                                probe.c, out, probe.c, &results[i].algo, s_->workspace.data(),
                                                s_->workspace.size(), stream));
            };
            call();
            HIP_CHECK(hipEventRecord(a, stream));
            call();
            call();
            HIP_CHECK(hipEventRecord(b, stream));
            HIP_CHECK(hipEventSynchronize(b));
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, a, b));
            if (ms / 2 < best_ms) {
                best_ms = ms / 2;
                best = i;
            }
        }
    } catch (...) {
        (void)hipEventDestroy(a);
        (void)hipEventDestroy(b);
        throw;
    }
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    auto &list = s_->tuned[std::make_tuple(m, in, ldx, ldo)];
    std::erase_if(list, [&](const auto &e) { return e.first == rows; });
    list.emplace_back(rows, results[best].algo);
    // plans created before this tuning keep their algorithm unless rebuilt
    std::erase_if(s_->plans, [&](const auto &e) {
        const auto &[pm, pn, pk, pb, pc] = e.first;
        return pm == m && pk == in && pb == ldx && pc == ldo;
    });
    return 2.0 * m * rows * double(in) / best_ms / 1e9;
}
void DenseMatmul::gemm(int m, int in, const BFloat16 *x, int ldx, int rows, float *out, int ldo,
                       const BFloat16 *w, hipStream_t stream) {
    if (rows < 1 || m < 1 || ldx < in || ldo < m)
        throw std::invalid_argument("QTIP dense gemm shape");
    const float one = 1.f, zero = 0.f;
    auto &p = s_->plan(m, rows, in, ldx, ldo);
    QTIP_BLAS_CHECK(hipblasLtMatmul(s_->handle, p.op, &one, w, p.a, x, p.b, &zero, out, p.c, out, p.c,
                                    &p.algo, s_->workspace.data(), s_->workspace.size(), stream));
}

} // namespace behemoth::qtip
