#include "behemoth/gpu.hpp"
#include "device_numeric.hpp"
#include <array>
#include <cstdlib>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <rocwmma/rocwmma.hpp>
#include <tuple>
#include <type_traits>
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
static_assert(rocwmma::Constants::AMDGCN_WAVE_SIZE == 32, "rocWMMA must use Wave32");
#endif

namespace behemoth {
std::span<const ProjectionMode> projection_modes() {
    static constexpr auto modes = [] {
        std::array<ProjectionMode, size_t(ProjectionMode::Count)> result{};
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = ProjectionMode(i);
        return result;
    }();
    return modes;
}
const char *projection_mode_name(ProjectionMode m) {
    switch (m) {
    case ProjectionMode::Auto:
        return "auto";
    case ProjectionMode::Fused:
        return "fused";
    case ProjectionMode::SplitK:
        return "split-k";
    case ProjectionMode::Blas:
        return "blas";
    case ProjectionMode::Tiled64:
        return "tiled-64";
    case ProjectionMode::Tiled128:
        return "tiled-128";
    case ProjectionMode::Pipelined64:
        return "pipelined-64";
    case ProjectionMode::Pipelined128:
        return "pipelined-128";
    case ProjectionMode::TiledLds64:
        return "tiled-lds-64";
    case ProjectionMode::TiledLds128:
        return "tiled-lds-128";
    case ProjectionMode::PipelinedN32K64:
        return "pipelined-n32-64";
    case ProjectionMode::PipelinedN32K128:
        return "pipelined-n32-128";
    case ProjectionMode::PipelinedN128K64:
        return "pipelined-n128-64";
    case ProjectionMode::PipelinedN128K128:
        return "pipelined-n128-128";
    case ProjectionMode::PipelinedSplit512K64:
        return "pipelined-s512-64";
    case ProjectionMode::PipelinedSplit512K128:
        return "pipelined-s512-128";
    case ProjectionMode::PipelinedSplit2048K64:
        return "pipelined-s2048-64";
    case ProjectionMode::PipelinedSplit2048K128:
        return "pipelined-s2048-128";
    case ProjectionMode::PipelinedR2K64:
        return "pipelined-r2-64";
    case ProjectionMode::PipelinedR2K128:
        return "pipelined-r2-128";
    case ProjectionMode::PipelinedR4K64:
        return "pipelined-r4-64";
    case ProjectionMode::PipelinedR4K128:
        return "pipelined-r4-128";
    case ProjectionMode::SortedS4:
        return "sorted-s4";
    case ProjectionMode::SortedS8:
        return "sorted-s8";
    case ProjectionMode::SortedS16:
        return "sorted-s16";
    case ProjectionMode::SortedDense:
        return "sorted-dense";
    case ProjectionMode::SortedLds:
        return "sorted-lds";
    case ProjectionMode::LaneS1:
        return "lane-s1";
    case ProjectionMode::LaneS2:
        return "lane-s2";
    case ProjectionMode::LaneS4:
        return "lane-s4";
    case ProjectionMode::LaneS8:
        return "lane-s8";
    case ProjectionMode::Count:
        break;
    }
    throw std::invalid_argument("Invalid projection mode");
}
int projection_mode_layout(ProjectionMode m) {
    switch (m) {
    case ProjectionMode::SortedS4:
    case ProjectionMode::SortedS8:
    case ProjectionMode::SortedS16:
        return 1;
    case ProjectionMode::SortedDense:
    case ProjectionMode::SortedLds:
        return 3;
    case ProjectionMode::LaneS1:
    case ProjectionMode::LaneS2:
    case ProjectionMode::LaneS4:
    case ProjectionMode::LaneS8:
        return 2;
    default:
        return 0;
    }
}
bool projection_mode_is_sorted(ProjectionMode m) {
    return projection_mode_layout(m) != 0;
}
ProjectionMode parse_projection_mode(std::string_view name) {
    for (auto m : projection_modes())
        if (name == projection_mode_name(m))
            return m;
    throw std::invalid_argument("Unknown projection mode: " + std::string(name));
}
const char *projection_search_name(ProjectionSearch search) {
    switch (search) {
    case ProjectionSearch::Conservative:
        return "conservative";
    case ProjectionSearch::Exhaustive:
        return "exhaustive";
    }
    throw std::invalid_argument("Invalid projection search");
}
ProjectionSearch parse_projection_search(std::string_view name) {
    if (name == "conservative")
        return ProjectionSearch::Conservative;
    if (name == "exhaustive")
        return ProjectionSearch::Exhaustive;
    throw std::invalid_argument("Projection search must be conservative or exhaustive");
}
namespace {
using F8 = rocwmma::float8_t;
constexpr int split_width = 512;
constexpr int tiled_split_width = 1024;
using Key = std::tuple<int, int, int, bool, CodeLayout>;
using SelectionKey = std::tuple<Key, uint64_t, bool>;
Key shape_key(const GpuProjection &p, int rows) {
    return {p.in, p.out, rows, !p.dense.size(),
            p.dense.size() ? CodeLayout::RowMajor : p.code_layout};
}
SelectionKey selection_key(const GpuProjection &p, int rows, bool apply_rows) {
    // Decode lookup behavior depends on the actual code tensor/codebook.
    // Large-prefill fused/BLAS choices retain shape caching to bound cold tuning.
    return {shape_key(p, rows), rows <= 16 ? p.tuning_identity : 0, apply_rows};
}
struct PipelineGeometry {
    ProjectionMode mode;
    int ktile, columns, split;
    int row_tiles = 1; // 16-row WMMA tiles sharing one weight-tile lookup
};
constexpr std::array pipeline_geometries{
    PipelineGeometry{ProjectionMode::Pipelined64, 64, 64, 1024},
    PipelineGeometry{ProjectionMode::Pipelined128, 128, 64, 1024},
    PipelineGeometry{ProjectionMode::PipelinedN32K64, 64, 32, 1024},
    PipelineGeometry{ProjectionMode::PipelinedN32K128, 128, 32, 1024},
    PipelineGeometry{ProjectionMode::PipelinedN128K64, 64, 128, 1024},
    PipelineGeometry{ProjectionMode::PipelinedN128K128, 128, 128, 1024},
    PipelineGeometry{ProjectionMode::PipelinedSplit512K64, 64, 64, 512},
    PipelineGeometry{ProjectionMode::PipelinedSplit512K128, 128, 64, 512},
    PipelineGeometry{ProjectionMode::PipelinedSplit2048K64, 64, 64, 2048},
    PipelineGeometry{ProjectionMode::PipelinedSplit2048K128, 128, 64, 2048},
    // Multi-row variants: one codebook lookup per weight tile feeds 2 or 4
    // row tiles (32 / 64 batch rows) so lookup cost is shared across rows.
    PipelineGeometry{ProjectionMode::PipelinedR2K64, 64, 64, 1024, 2},
    PipelineGeometry{ProjectionMode::PipelinedR2K128, 128, 64, 1024, 2},
    PipelineGeometry{ProjectionMode::PipelinedR4K64, 64, 64, 1024, 4},
    PipelineGeometry{ProjectionMode::PipelinedR4K128, 128, 64, 1024, 4}};
const PipelineGeometry *pipeline_geometry(ProjectionMode mode) {
    for (const auto &g : pipeline_geometries)
        if (g.mode == mode)
            return &g;
    return nullptr;
}
template <typename F> void dispatch_pipeline(ProjectionMode mode, F &&f) {
    switch (mode) {
    case ProjectionMode::Pipelined64:
        return f.template operator()<64, 64, 1024>();
    case ProjectionMode::Pipelined128:
        return f.template operator()<128, 64, 1024>();
    case ProjectionMode::PipelinedN32K64:
        return f.template operator()<64, 32, 1024>();
    case ProjectionMode::PipelinedN32K128:
        return f.template operator()<128, 32, 1024>();
    case ProjectionMode::PipelinedN128K64:
        return f.template operator()<64, 128, 1024>();
    case ProjectionMode::PipelinedN128K128:
        return f.template operator()<128, 128, 1024>();
    case ProjectionMode::PipelinedSplit512K64:
        return f.template operator()<64, 64, 512>();
    case ProjectionMode::PipelinedSplit512K128:
        return f.template operator()<128, 64, 512>();
    case ProjectionMode::PipelinedSplit2048K64:
        return f.template operator()<64, 64, 2048>();
    case ProjectionMode::PipelinedSplit2048K128:
        return f.template operator()<128, 64, 2048>();
    case ProjectionMode::PipelinedR2K64:
        return f.template operator()<64, 64, 1024, 2>();
    case ProjectionMode::PipelinedR2K128:
        return f.template operator()<128, 64, 1024, 2>();
    case ProjectionMode::PipelinedR4K64:
        return f.template operator()<64, 64, 1024, 4>();
    case ProjectionMode::PipelinedR4K128:
        return f.template operator()<128, 64, 1024, 4>();
    default:
        throw std::logic_error("Not a pipelined projection mode");
    }
}
void blas_check(hipblasStatus_t s, const char *what) {
    if (s != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": hipBLASLt status " +
                                 std::to_string(int(s)));
}
#define BLAS_CHECK(x) blas_check((x), #x)
struct Descriptors {
    hipblasLtMatmulDesc_t op{};
    hipblasLtMatrixLayout_t a{}, b{}, c{};
    hipblasLtMatmulPreference_t preference{};
    ~Descriptors() {
        if (preference)
            hipblasLtMatmulPreferenceDestroy(preference);
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
struct Plan {
    bool blas_checked = false, has_blas = false;
    std::unique_ptr<Descriptors> descriptors;
    hipblasLtMatmulHeuristicResult_t algorithm{};
};
struct Choice {
    bool tuned = false;
    ProjectionMode selected = ProjectionMode::Fused;
    std::array<float, size_t(ProjectionMode::Count) - 1> milliseconds;
    std::vector<ProjectionMode> candidates;
    std::vector<int> batch_runs;
    TuningDecision decision;
    Choice() {
        milliseconds.fill(NAN);
    }
};
struct RunnerState {
    hipblasLtHandle_t handle{};
    std::map<Key, Plan> plans;
    std::map<SelectionKey, Choice> choices;
    ProjectionMode override_mode = ProjectionMode::Auto;
    ProjectionSearch search = ProjectionSearch::Conservative;
    nlohmann::json resources;
    mutable nlohmann::json compact_profile;
};
void ensure_blas(RunnerState &s, Plan &p, int in, int out, int rows) {
    if (p.blas_checked)
        return;
    p.descriptors = std::make_unique<Descriptors>();
    auto &d = *p.descriptors;
    BLAS_CHECK(hipblasLtMatmulDescCreate(&d.op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t trans = HIPBLAS_OP_T;
    BLAS_CHECK(
        hipblasLtMatmulDescSetAttribute(d.op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    BLAS_CHECK(hipblasLtMatrixLayoutCreate(&d.a, HIP_R_8F_E4M3, uint64_t(in), uint64_t(out), in));
    BLAS_CHECK(hipblasLtMatrixLayoutCreate(&d.b, HIP_R_8F_E4M3, uint64_t(in), uint64_t(rows), in));
    BLAS_CHECK(hipblasLtMatrixLayoutCreate(&d.c, HIP_R_32F, uint64_t(out), uint64_t(rows), out));
    BLAS_CHECK(hipblasLtMatmulPreferenceCreate(&d.preference));
    size_t max_workspace = 64 * 1024 * 1024;
    BLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(d.preference,
                                                     HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &max_workspace, sizeof(max_workspace)));
    int count = 0;
    auto status = hipblasLtMatmulAlgoGetHeuristic(s.handle, d.op, d.a, d.b, d.c, d.c, d.preference,
                                                  1, &p.algorithm, &count);
    p.has_blas = status == HIPBLAS_STATUS_SUCCESS && count > 0 &&
                 p.algorithm.state == HIPBLAS_STATUS_SUCCESS;
    p.blas_checked = true;
}
nlohmann::json choice_report(const SelectionKey &selection, const Choice &choice, const Plan &plan,
                             const RunnerState &state) {
    const auto &[key, identity, apply_rows] = selection;
    const auto [in, out, rows, compressed, layout] = key;
    auto candidates = nlohmann::json::array();
    for (auto mode : projection_modes()) {
        if (mode == ProjectionMode::Auto)
            continue;
        nlohmann::json candidate = {
            {"mode", projection_mode_name(mode)},
            {"ms", timing_json(choice.milliseconds[size_t(int(mode) - 1)])}};
        auto it = std::find(choice.candidates.begin(), choice.candidates.end(), mode);
        if (choice.tuned && it != choice.candidates.end()) {
            const auto index = size_t(it - choice.candidates.begin());
            const auto &m = choice.decision.measurements[index];
            const auto &samples = m.confirmation.empty() ? m.samples : m.confirmation;
            candidate["batch_runs"] = choice.batch_runs[index];
            candidate["round_samples_ms"] = m.samples;
            candidate["confirmation_samples_ms"] = m.confirmation;
            candidate["min_ms"] = *std::min_element(samples.begin(), samples.end());
            candidate["max_ms"] = *std::max_element(samples.begin(), samples.end());
        }
        if (auto *g = pipeline_geometry(mode)) {
            candidate["k_tile"] = g->ktile;
            candidate["output_columns"] = g->columns;
            candidate["threads"] = g->columns * 2;
            candidate["k_partition"] = g->split;
            candidate["row_tiles"] = g->row_tiles;
            candidate["aligned"] = compressed && layout == CodeLayout::GroupMajor &&
                                   out % g->columns == 0 && in % g->ktile == 0 &&
                                   size_t(out) * size_t(in / 8) <= UINT32_MAX;
        }
        candidates.push_back(std::move(candidate));
    }
    auto report =
        nlohmann::json{{"in", in},
                       {"out", out},
                       {"rows", rows},
                       {"compressed", compressed},
                       {"code_layout", compressed ? code_layout_name(layout) : "dense"},
                       {"weight_identity", identity},
                       {"cache_scope", identity ? "weight" : "shape"},
                       {"apply_row_scale", apply_rows},
                       {"search", projection_search_name(state.search)},
                       {"override", projection_mode_name(state.override_mode)},
                       {"selected", state.override_mode != ProjectionMode::Auto
                                        ? projection_mode_name(state.override_mode)
                                    : choice.tuned ? projection_mode_name(choice.selected)
                                                   : "untuned"},
                       {"tuned", choice.tuned},
                       {"blas_checked", plan.blas_checked},
                       {"blas_available", plan.has_blas},
                       {"fused_ms", timing_json(choice.milliseconds[0])},
                       {"split_k_ms", timing_json(choice.milliseconds[1])},
                       {"blas_ms", timing_json(choice.milliseconds[2])},
                       {"tiled_64_ms", timing_json(choice.milliseconds[3])},
                       {"tiled_128_ms", timing_json(choice.milliseconds[4])},
                       {"pipelined_64_ms", timing_json(choice.milliseconds[5])},
                       {"pipelined_128_ms", timing_json(choice.milliseconds[6])},
                       {"tiled_lds_64_ms", timing_json(choice.milliseconds[7])},
                       {"tiled_lds_128_ms", timing_json(choice.milliseconds[8])},
                       {"tuning_method", "seven_interleaved_rounds_then_three_ABBA_pairs"},
                       {"timing_includes_quantization", true},
                       {"minimum_confirmation_gain", .02},
                       {"candidates", std::move(candidates)}};
    if (choice.tuned) {
        report["confirmation_incumbent"] =
            projection_mode_name(choice.candidates[choice.decision.incumbent]);
        report["confirmation_challenger"] =
            projection_mode_name(choice.candidates[choice.decision.challenger]);
        report["confirmation_relative_gain"] = choice.decision.confirmation_gain;
    }
    return report;
}
__device__ size_t code_index(int group, int output, int groups, int outputs, bool group_major) {
    return group_major ? size_t(group) * size_t(outputs) + size_t(output)
                       : size_t(output) * size_t(groups) + size_t(group);
}
__global__ void reconstruct_kernel(const uint16_t *codes, const uint8_t *book, uint8_t *weight,
                                   int in, int out, bool group_major) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, n = size_t(in) * size_t(out);
    if (i < n) {
        size_t row = i / size_t(in), col = i % size_t(in);
        weight[i] =
            book[size_t(codes[code_index(int(col / 8), int(row), in / 8, out, group_major)]) * 8 +
                 col % 8];
    }
}
__global__ void scale_output(float *y, const float *xs, const float *ws, float bs, int rows,
                             int out, bool apply_rows) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(rows) * size_t(out))
        y[i] *= xs[i / size_t(out)] * bs * (apply_rows ? ws[i % size_t(out)] : 1.f);
}
template <bool split>
__global__ void fused_projection(const uint8_t *x, const float *xs, const uint16_t *codes,
                                 const uint8_t *book, const uint8_t *dense, const float *ws,
                                 float bs, float *y, int rows, int in, int out, bool apply_rows,
                                 int parts, bool group_major) {
    const int lane = int(threadIdx.x), row0 = int(blockIdx.y) * 16, col0 = int(blockIdx.x) * 16,
              part = split ? int(blockIdx.z) : 0;
    const int begin = split ? part * split_width : 0,
              end = split ? min(in, begin + split_width) : in;
    __shared__ __attribute__((aligned(16))) F8 a[16 * 32], b[32 * 16];
    __shared__ float c[16 * 16];
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> acc;
    rocwmma::fill_fragment(acc, 0.f);
    for (int k0 = begin; k0 < end; k0 += 32) {
        for (int i = lane; i < 16 * 32; i += device::wave_size) {
            int row = row0 + i / 32, k = k0 + i % 32;
            uint8_t value = row < rows && k < end ? x[size_t(row) * size_t(in) + size_t(k)] : 0;
            a[i] = __builtin_bit_cast(F8, value);
        }
        // Load one contiguous eight-weight vector per codebook index.
        for (int group = lane; group < 4 * 16; group += device::wave_size) {
            int k = k0 + (group / 16) * 8, col = col0 + group % 16;
            uint64_t packed = 0;
            if (k < end && col < out) {
                if (dense) // dense layout follows the group_major flag (see run())
                    packed = reinterpret_cast<const uint64_t *>(
                        dense)[code_index(k / 8, col, in / 8, out, group_major)];
                else
                    packed = reinterpret_cast<const uint64_t *>(
                        book)[codes[code_index(k / 8, col, in / 8, out, group_major)]];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint8_t value = uint8_t(packed >> (j * 8));
                b[((group / 16) * 8 + j) * 16 + group % 16] = __builtin_bit_cast(F8, value);
            }
        }
        __syncthreads();
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major> ma;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::row_major> mb;
        rocwmma::load_matrix_sync(ma, a, 32);
        rocwmma::load_matrix_sync(mb, b, 16);
        rocwmma::mma_sync(acc, ma, mb, acc);
        __syncthreads();
    }
    rocwmma::store_matrix_sync(c, acc, 16, rocwmma::mem_row_major);
    __syncthreads();
    for (int i = lane; i < 256; i += device::wave_size) {
        int row = row0 + i / 16, col = col0 + i % 16;
        if (row >= rows || col >= out)
            continue;
        size_t index = size_t(row) * size_t(out) + size_t(col);
        if constexpr (split)
            y[size_t(part) * size_t(rows) * size_t(out) + index] = c[i];
        else
            y[index] = c[i] * xs[row] * bs * (apply_rows ? ws[col] : 1.f);
    }
}
__device__ __forceinline__ void projection_lds_barrier() {
    // Only LDS is exchanged between waves. A full __syncthreads() also fences
    // global memory, waits for register prefetches, and invalidates read caches.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
template <bool lds_only> __device__ __forceinline__ void projection_tile_barrier() {
    if constexpr (lds_only)
        projection_lds_barrier();
    else
        __syncthreads();
}
// Four waves own disjoint 16-column accumulators, share the activation tile,
// and issue independent lookup vectors before the next matrix products.
template <int ktile, bool lds_only = false>
__global__ void tiled_projection(const uint8_t *x, const uint16_t *codes, const uint8_t *book,
                                 const uint8_t *dense, float *partial, int rows, int in, int out,
                                 int parts, bool group_major) {
    constexpr int columns = 64, threads = 128;
    // A power-of-two column stride would send a whole wave's vector stores
    // to the same LDS banks. Keep vector alignment, but stagger columns.
    constexpr int b_stride = ktile + 16;
    constexpr int operand_bytes = 16 * ktile + columns * b_stride;
    constexpr int result_bytes = 16 * columns * sizeof(float);
    constexpr int shared_bytes = operand_bytes > result_bytes ? operand_bytes : result_bytes;
    __shared__ __attribute__((aligned(16))) uint8_t storage[shared_bytes];
    F8 *a = reinterpret_cast<F8 *>(storage);
    F8 *b = reinterpret_cast<F8 *>(storage + 16 * ktile);
    float *c = reinterpret_cast<float *>(storage); // reused only after all operand reads finish
    const int tid = int(threadIdx.x), wave = tid / device::wave_size;
    const int row0 = int(blockIdx.y) * 16, col0 = int(blockIdx.x) * columns, part = int(blockIdx.z);
    const int begin = part * tiled_split_width, end = min(in, begin + tiled_split_width);
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> acc;
    rocwmma::fill_fragment(acc, 0.f);
    for (int base = begin; base < end; base += ktile) {
#pragma unroll
        for (int pack = tid; pack < 16 * ktile / 8; pack += threads) {
            int row = pack / (ktile / 8), k = (pack % (ktile / 8)) * 8;
            uint64_t value = 0;
            if (row0 + row < rows && base + k < end)
                value = reinterpret_cast<const uint64_t *>(
                    x)[size_t(row0 + row) * size_t(in / 8) + size_t((base + k) / 8)];
            reinterpret_cast<uint64_t *>(a)[pack] = value;
        }
#pragma unroll
        for (int pack = tid; pack < columns * ktile / 8; pack += threads) {
            int col = pack % columns, k = (pack / columns) * 8;
            uint64_t value = 0;
            if (col0 + col < out && base + k < end) {
                if (dense)
                    value = reinterpret_cast<const uint64_t *>(
                        dense)[code_index((base + k) / 8, col0 + col, in / 8, out, group_major)];
                else
                    value = reinterpret_cast<const uint64_t *>(book)[codes[code_index(
                        (base + k) / 8, col0 + col, in / 8, out, group_major)]];
            }
            reinterpret_cast<uint64_t *>(b)[size_t(col) * size_t(b_stride / 8) + size_t(k / 8)] =
                value;
        }
        projection_tile_barrier<lds_only>();
        for (int k = 0; k < ktile && base + k < end; k += 32) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major> ma;
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::col_major> mb;
            rocwmma::load_matrix_sync(ma, a + k, ktile);
            rocwmma::load_matrix_sync(mb, b + wave * 16 * b_stride + k, b_stride);
            rocwmma::mma_sync(acc, ma, mb, acc);
        }
        projection_tile_barrier<lds_only>(); // also protects operand-to-output storage reuse
    }
    rocwmma::store_matrix_sync(c + wave * 16, acc, columns, rocwmma::mem_row_major);
    projection_tile_barrier<lds_only>();
    for (int i = tid; i < 16 * columns; i += threads) {
        int row = row0 + i / columns, col = col0 + i % columns;
        if (row < rows && col < out)
            partial[(size_t(part) * size_t(rows) + size_t(row)) * size_t(out) + size_t(col)] = c[i];
    }
}
// Hold the next tile in registers. Separate index loads from dependent book
// loads so each wave can have several independent lookup chains in flight.
template <int ktile, bool aligned, int columns = 64, int row_tiles = 1> struct ProjectionPrefetch {
    static constexpr int threads = columns * 2, activation_count = 16 * row_tiles * ktile / 8;
    static constexpr bool activation_tail = activation_count % threads != 0;
    static constexpr int weight_packs = ktile / 16,
                         activation_packs = (activation_count + threads - 1) / threads;
    uint64_t weight[weight_packs], activation[activation_packs];
    __device__ __forceinline__ void
    load(const uint8_t *__restrict__ x, const uint16_t *__restrict__ codes,
         const uint8_t *__restrict__ book, const uint8_t *__restrict__ dense, int tid, int row0,
         int col0, int base, int end, int rows, int in, int out, bool group_major) {
        uint32_t index[weight_packs];
#pragma unroll
        for (int i = 0; i < weight_packs; ++i) {
            const int pack = tid + i * threads, col = col0 + pack % columns,
                      k = base + (pack / columns) * 8;
            if constexpr (aligned)
                // Dispatch bounds the entire code tensor to uint32 addressing.
                index[i] = codes[uint32_t(k / 8) * uint32_t(out) + uint32_t(col)];
            else
                index[i] = !dense && col < out && k < end
                               ? codes[code_index(k / 8, col, in / 8, out, group_major)]
                               : 0;
        }
        // Compiler ordering only, without a device fence. LLVM still emits
        // dependency waits. Issue every index before the dependent book loads,
        // and issue the next tile's loads before current-tile LDS reads/WMMA.
        asm volatile("" ::: "memory");
#pragma unroll
        for (int i = 0; i < weight_packs; ++i) {
            if constexpr (aligned)
                weight[i] = reinterpret_cast<const uint64_t *>(book)[index[i]];
            else {
                const int pack = tid + i * threads, col = col0 + pack % columns,
                          k = base + (pack / columns) * 8;
                uint64_t value = 0;
                if (col < out && k < end)
                    value = dense ? reinterpret_cast<const uint64_t *>(
                                        dense)[code_index(k / 8, col, in / 8, out, group_major)]
                                  : reinterpret_cast<const uint64_t *>(book)[index[i]];
                weight[i] = value;
            }
        }
#pragma unroll
        for (int i = 0; i < activation_packs; ++i) {
            const int pack = tid + i * threads, row = row0 + pack / (ktile / 8),
                      k = base + (pack % (ktile / 8)) * 8;
            uint64_t value = 0;
            if ((!activation_tail || pack < activation_count) && row < rows && (aligned || k < end))
                value = reinterpret_cast<const uint64_t *>(
                    x)[size_t(row) * size_t(in / 8) + size_t(k / 8)];
            activation[i] = value;
        }
        asm volatile("" ::: "memory");
    }
    __device__ __forceinline__ void stage(F8 *a, F8 *b, int tid) const {
#pragma unroll
        for (int i = 0; i < activation_packs; ++i) {
            const int pack = tid + i * threads;
            if (!activation_tail || pack < activation_count)
                reinterpret_cast<uint64_t *>(a)[pack] = activation[i];
        }
#pragma unroll
        for (int i = 0; i < weight_packs; ++i) {
            const int pack = tid + i * threads;
            reinterpret_cast<uint64_t *>(
                b)[size_t(pack % columns) * size_t((ktile + 16) / 8) + size_t(pack / columns)] =
                weight[i];
        }
    }
};
// aligned is restricted at dispatch to compressed, group-major weights whose
// K and N are whole tiles. Generic variants retain dense/layout/tail support.
// row_tiles > 1: the block owns 16*row_tiles batch rows. The weight tile is
// looked up and staged once, then multiplied against each 16-row activation
// tile, so the codebook gather cost is shared by up to 64 rows.
template <int ktile, bool aligned, int columns = 64, int split = tiled_split_width,
          int row_tiles = 1>
__global__ void
pipelined_projection(const uint8_t *__restrict__ x, const uint16_t *__restrict__ codes,
                     const uint8_t *__restrict__ book, const uint8_t *__restrict__ dense,
                     float *__restrict__ partial, int rows, int in, int out, bool group_major) {
    static_assert(columns == 32 || columns == 64 || columns == 128);
    static_assert(split >= split_width && split % ktile == 0);
    static_assert(row_tiles == 1 || row_tiles == 2 || row_tiles == 4);
    constexpr int threads = columns * 2, b_stride = ktile + 16, block_rows = 16 * row_tiles;
    constexpr int operand_bytes = block_rows * ktile + columns * b_stride;
    constexpr int result_bytes = block_rows * columns * sizeof(float);
    constexpr int shared_bytes = operand_bytes > result_bytes ? operand_bytes : result_bytes;
    __shared__ __attribute__((aligned(16))) uint8_t storage[shared_bytes];
    F8 *a = reinterpret_cast<F8 *>(storage);
    F8 *b = reinterpret_cast<F8 *>(storage + block_rows * ktile);
    float *c = reinterpret_cast<float *>(storage);
    const int tid = int(threadIdx.x), wave = tid / device::wave_size;
    const int row0 = int(blockIdx.y) * block_rows, col0 = int(blockIdx.x) * columns,
              part = int(blockIdx.z), begin = part * split, end = min(in, begin + split);
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> acc[row_tiles];
#pragma unroll
    for (int r = 0; r < row_tiles; ++r)
        rocwmma::fill_fragment(acc[r], 0.f);
    ProjectionPrefetch<ktile, aligned, columns, row_tiles> next;
    next.load(x, codes, book, dense, tid, row0, col0, begin, end, rows, in, out, group_major);
    for (int base = begin; base < end; base += ktile) {
        next.stage(a, b, tid);
        projection_lds_barrier();
        if (base + ktile < end)
            next.load(x, codes, book, dense, tid, row0, col0, base + ktile, end, rows, in, out,
                      group_major);
#pragma unroll 1
        for (int k = 0; k < ktile && (aligned || base + k < end); k += 32) {
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::col_major> mb;
            rocwmma::load_matrix_sync(mb, b + wave * 16 * b_stride + k, b_stride);
#pragma unroll
            for (int r = 0; r < row_tiles; ++r) {
                rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major> ma;
                rocwmma::load_matrix_sync(ma, a + r * 16 * ktile + k, ktile);
                rocwmma::mma_sync(acc[r], ma, mb, acc[r]);
            }
        }
        projection_lds_barrier();
    }
#pragma unroll
    for (int r = 0; r < row_tiles; ++r)
        rocwmma::store_matrix_sync(c + r * 16 * columns + wave * 16, acc[r], columns,
                                   rocwmma::mem_row_major);
    projection_lds_barrier();
    for (int i = tid; i < block_rows * columns; i += threads) {
        int row = row0 + i / columns, col = col0 + i % columns;
        if (row < rows && (aligned || col < out))
            partial[(size_t(part) * size_t(rows) + size_t(row)) * size_t(out) + size_t(col)] = c[i];
    }
}
// ------------------------------------------------------------------------
// Codebook-major ("sorted") layout kernels, see include/behemoth/sorted_layout.hpp,
// CODEBOOK_GATHER_REPORT.md and BATCHED_SORTED_REPORT.md. Tiles of 16 output
// rows; every chunk of 32 entries carries its own base code, so any chunk
// decodes independently (code = base + inclusive prefix of the delta planes).
//
// sorted_projection (rows = 1 per pass): a wave streams a tile in code order,
// so consecutive lanes read consecutive codebook entries (L0 hits); each lane
// reads x[group] from LDS and adds the exact fixed-point (2^-18) dot product
// into one of C int64 LDS accumulators per row with ds_add_u64 (fp32 LDS
// atomics are serialised on gfx1201). Dummies address group 2047 (zero slot).
//
// sorted_lds_gemm (batched): one block per tile decodes the stream into an
// LDS code tile [groups][16] (2-byte LDS writes), then the waves gather the
// 8-byte codebook entries straight into v_wmma_f32_16x16x16_fp8_fp8 B
// operands (lane = column l%16, k-slice l/16 = one group) against up to 64
// activation rows. The stream is read with system-scope buffer loads: a stream
// that allocates in the GPU caches halves the random-gather rate of the
// concurrently running tiles (measured, perf/micro_batched.hip).
//
// sorted_lds_reconstruct: the same decode, then coalesced dense FP8 rows
// (row-major for hipBLASLt, group-major for the dense pipelined kernels).
__device__ __forceinline__ uint64_t sorted_exact_fixed18(float v) {
    // v is a sum of products of e4m3 values: an integer multiple of 2^-18 with
    // |v| < 2^21, so n = v*2^18 is an integer-valued float below 2^39. Split
    // n = a*2^16 + b with 0 <= b < 2^16: a = floor(v*4) (|a| < 2^23, exact),
    // b = n - a*2^16 (< 2^16, exact in fp32). No rounding anywhere.
    const float a_f = floorf(v * 4.f);
    const float b_f = fmaf(a_f, -65536.f, v * 262144.f);
    const int32_t a = int32_t(a_f);
    const uint32_t b = uint32_t(b_f);
    const uint32_t lo = (uint32_t(a) << 16) | b, hi = uint32_t(a >> 16);
    return (uint64_t(hi) << 32) | lo;
}
constexpr int sorted_waves = 8, sorted_unroll = 4, sorted_copies = 2;
constexpr int sorted_group_bits = 11; // log2(sorted_groups_per_tile)
constexpr int TR = sorted_rows_per_tile;
static_assert(sorted_groups_per_tile == 1 << sorted_group_bits && TR == 16);
// x: fp8 [rows][in]; partial: [splits][rows][out]. grid = (block_groups * splits, rows).
template <int W, int UNROLL, int C, int DB>
__global__ void __launch_bounds__(W * 32)
sorted_projection(const uint8_t *__restrict__ x, const uint16_t *__restrict__ idx,
                  const uint32_t *__restrict__ planes, const uint16_t *__restrict__ chunk_codes,
                  const SortedTileDesc *__restrict__ tiles, const uint8_t *__restrict__ book,
                  float *__restrict__ partial, int rows, int in, int out, int tile_count,
                  int splits) {
    constexpr int G = sorted_groups_per_tile, GB = sorted_group_bits, THREADS = W * 32;
    static_assert((C & (C - 1)) == 0 && C <= 32);
    __shared__ uint64_t xs[G];
    __shared__ uint64_t ys[W * 32 * C];
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
    const int tg = int(blockIdx.x) / splits, split = int(blockIdx.x) % splits, groups = in / 8;
    const int row = int(blockIdx.y), tile = tg * W + wave;
    const uint64_t *xr = reinterpret_cast<const uint64_t *>(x) + size_t(row) * size_t(groups);
    for (int i = tid; i < G; i += THREADS)
        xs[i] = i < groups ? xr[i] : 0;
    for (int i = tid; i < W * 32 * C; i += THREADS)
        ys[i] = 0;
    __syncthreads();
    if (tile < tile_count) {
        const SortedTileDesc t = tiles[tile];
        const uint32_t le = (2u << lane) - 1; // lanes <= this one
        const uint32_t c0 = __builtin_amdgcn_readfirstlane(
                           t.chunk_begin + (t.chunk_count * uint32_t(split)) / uint32_t(splits)),
                       c1 = __builtin_amdgcn_readfirstlane(
                           t.chunk_begin + (t.chunk_count * uint32_t(split + 1)) / uint32_t(splits));
        uint64_t *yi = ys + wave * (32 * C) + (lane & (C - 1)) * 32;
        auto step = [&](uint32_t cc, int n) {
            uint32_t e[UNROLL], off[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if (u >= n)
                    break;
                e[u] = idx[size_t(cc + u) * 32 + lane];
                uint32_t code = chunk_codes[cc + u];
#pragma unroll
                for (int k = 0; k < DB; ++k)
                    code += uint32_t(__builtin_popcount(planes[size_t(cc + u) * DB + k] & le)) << k;
                off[u] = code << 3; // byte offset into the codebook
            }
            uint2 w[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                if (u < n)
                    w[u] = *reinterpret_cast<const uint2 *>(book + off[u]);
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if (u >= n)
                    break;
                const int r = int(e[u] >> GB), grp = int(e[u] & (G - 1));
                const uint64_t xv = xs[grp];
                float v = __builtin_amdgcn_dot4_f32_fp8_fp8(w[u].x, uint32_t(xv), 0.f);
                v = __builtin_amdgcn_dot4_f32_fp8_fp8(w[u].y, uint32_t(xv >> 32), v);
                __hip_atomic_fetch_add(&yi[r], sorted_exact_fixed18(v), __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_WORKGROUP);
            }
        };
        uint32_t c = c0;
        for (; c + UNROLL <= c1; c += UNROLL)
            step(c, UNROLL);
        for (; c < c1; ++c)
            step(c, 1);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        int64_t acc = 0;
        const uint64_t *base = ys + wave * (32 * C);
#pragma unroll
        for (int k = 0; k < C; ++k)
            acc += int64_t(base[k * 32 + lane]);
        const int r = tile * TR + lane;
        if (lane < TR && r < out)
            partial[(size_t(split) * size_t(rows) + size_t(row)) * size_t(out) + size_t(r)] =
                float(double(acc) * (1.0 / 262144.0));
    }
}
// Decodes one tile's stream into ct[g * 16 + row] = code. Eight chunks per wave
// step (four lanes per chunk, eight entries per lane from one 16-byte load), U
// steps in flight. The loads carry the gfx12 SCOPE_SYS policy (cachepolicy 24):
// the stream is read once and must not displace the codebook from the caches
// that serve the random gathers of the other tiles.
typedef unsigned sorted_u32x4 __attribute__((ext_vector_type(4)));
constexpr int sorted_stream_policy = 24, sorted_buffer_flags = 0x31004000;
// Only groups in [gbeg, gbeg + groups) are written (K window of the block).
template <int DB, int U>
__device__ __forceinline__ void
sorted_decode_tile(uint16_t *ct, const SortedTileDesc &t, const uint16_t *__restrict__ idx,
                   const uint32_t *__restrict__ planes, const uint16_t *__restrict__ chunk_codes,
                   int gbeg, int groups, int wave, int waves, int lane) {
    constexpr int G = sorted_groups_per_tile, GB = sorted_group_bits;
    const __amdgpu_buffer_rsrc_t ridx = __builtin_amdgcn_make_buffer_rsrc(
                                     const_cast<uint16_t *>(idx), 0, 0x7fffffff, sorted_buffer_flags),
                                 rpl = __builtin_amdgcn_make_buffer_rsrc(
                                     const_cast<uint32_t *>(planes), 0, 0x7fffffff, sorted_buffer_flags),
                                 rcc = __builtin_amdgcn_make_buffer_rsrc(
                                     const_cast<uint16_t *>(chunk_codes), 0, 0x7fffffff,
                                     sorted_buffer_flags);
    const uint32_t octets = (t.chunk_count + 7) >> 3;
    const uint32_t sub = uint32_t(lane) >> 2, part = uint32_t(lane) & 3;
    for (uint32_t o = uint32_t(wave); o < octets; o += uint32_t(waves) * U) {
        sorted_u32x4 e[U];
        uint32_t pl[U][DB], base[U];
        bool valid[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint32_t rel = (o + uint32_t(u * waves)) * 8 + sub;
            valid[u] = (o + uint32_t(u * waves)) < octets && rel < t.chunk_count;
            const uint32_t cc = t.chunk_begin + (valid[u] ? rel : 0);
            e[u] = __builtin_amdgcn_raw_buffer_load_b128(ridx, int((size_t(cc) * 32 + part * 8) * 2),
                                                        0, sorted_stream_policy);
#pragma unroll
            for (int k = 0; k < DB; ++k)
                pl[u][k] = __builtin_amdgcn_raw_buffer_load_b32(rpl, int((size_t(cc) * DB + k) * 4),
                                                                0, sorted_stream_policy);
            base[u] = __builtin_amdgcn_raw_buffer_load_b16(rcc, int(size_t(cc) * 2), 0,
                                                           sorted_stream_policy);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!valid[u])
                continue;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t p = part * 8 + uint32_t(j), mask = (2u << p) - 1;
                uint32_t code = base[u];
#pragma unroll
                for (int k = 0; k < DB; ++k)
                    code += uint32_t(__builtin_popcount(pl[u][k] & mask)) << k;
                const uint32_t ent = (e[u][j >> 1] >> (16 * (j & 1))) & 0xffffu;
                const int r = int(ent >> GB), g = int(ent & (G - 1)) - gbeg;
                if (g >= 0 && g < groups)
                    ct[g * TR + r] = uint16_t(code);
            }
        }
    }
}
// Device view of either codebook-major encoding (sorted_layout.hpp).
// sorted: codes = per-chunk base codes, one tile per 16 output rows.
// lane:   codes = per-segment per-lane running codes, tiles = group_tiles x
//         row_tiles (tile = gt * row_tiles + rt); planes holds the delta nibbles.
struct SortedArgs {
    const uint16_t *idx;
    const uint32_t *planes;
    const uint16_t *codes;
    const SortedTileDesc *tiles;
    int row_tiles, gt_size, group_tiles;
};
constexpr int lane_G = 1 << lane_group_bits, lane_S = lane_chunks_per_segment;
static_assert(lane_S == 32, "lane kernels assume 32-chunk segments (4 idx vectors, 1 nibble vector)");
__device__ __forceinline__ SortedTileDesc uniform_tile(const SortedTileDesc *tiles, int index) {
    // Tile descriptors are wave-uniform; keep them (and everything derived) in SGPRs.
    const SortedTileDesc t = tiles[__builtin_amdgcn_readfirstlane(index)];
    SortedTileDesc u;
    u.chunk_begin = __builtin_amdgcn_readfirstlane(t.chunk_begin);
    u.chunk_count = __builtin_amdgcn_readfirstlane(t.chunk_count);
    u.segment_begin = __builtin_amdgcn_readfirstlane(t.segment_begin);
    u.segment_count = __builtin_amdgcn_readfirstlane(t.segment_count);
    return u;
}
// One lane's segment: 32 idx words (four 16-byte loads) and 32 delta nibbles
// (one 16-byte load), all lane-private; no scalar or cross-lane loads.
// gfx12 cache policy of the lane stream loads (immediate): bits 0-2 temporal
// hint (0 regular, 1 non-temporal, 3 last-use), bits 3-4 scope (0 CU, 1 SE,
// 2 DEV, 3 SYS). Measured end to end (PERF_ROUND2_REPORT.md): regular loads
// (0) and DEV scope (16) are equal and ~8 % faster than SYS scope (24), which
// the batched sorted-lds decode still uses; non-temporal hints are 25 %
// slower. BEHEMOTH_LANE_POLICY overrides the default (measurement).
constexpr int lane_stream_policy = 0;
__device__ __forceinline__ sorted_u32x4 lane_load(__amdgpu_buffer_rsrc_t r, int off, int policy) {
    switch (policy) {
    case 0:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 0);
    case 1:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 1);
    case 3:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 3);
    case 8:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 8);
    case 16:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 16);
    case 17:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 17);
    case 25:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 25);
    default:
        return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 24);
    }
}
struct LaneSeg {
    sorted_u32x4 v[4], h;
    __device__ __forceinline__ void load(__amdgpu_buffer_rsrc_t ridx, __amdgpu_buffer_rsrc_t rhi,
                                         uint32_t seg, int lane, int policy) {
        const size_t base = size_t(seg) * 32 + size_t(lane);
#pragma unroll
        for (int b = 0; b < 4; ++b)
            v[b] = lane_load(ridx, int((base * lane_S + size_t(b) * 8) * 2), policy);
        h = lane_load(rhi, int(base * 16), policy);
    }
    // Batch b (uniform) = chunks 8b..8b+7: idx words and the nibble word.
    __device__ __forceinline__ void batch(int b, sorted_u32x4 &words, uint32_t &nib) const {
        switch (b) {
        case 0:
            words = v[0];
            nib = h[0];
            return;
        case 1:
            words = v[1];
            nib = h[1];
            return;
        case 2:
            words = v[2];
            nib = h[2];
            return;
        default:
            words = v[3];
            nib = h[3];
            return;
        }
    }
    // Entry jj of a batch: idx word and full 9-bit delta.
    static __device__ __forceinline__ void entry(sorted_u32x4 words, uint32_t nib, int jj,
                                                 uint32_t &e, uint32_t &d) {
        e = (words[(jj >> 1) & 3] >> (16 * (jj & 1))) & 0xffffu;
        d = (e >> lane_group_bits) | (((nib >> (4 * jj)) & 15u) << lane_delta_low_bits);
    }
};
__device__ __forceinline__ __amdgpu_buffer_rsrc_t lane_rsrc(const void *p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), 0, 0x7fffffff,
                                             sorted_buffer_flags);
}
// Lane layout: ct[(g + gshift) * 16 + lane/2] = code for the tile's entries
// whose shifted group falls in [0, groups). Lane l walks its own stream of
// the segments assigned to this wave (U segments in flight).
template <int U>
__device__ __forceinline__ void
lane_decode_tile(uint16_t *ct, const SortedTileDesc &t, const uint16_t *__restrict__ idx,
                 const uint32_t *__restrict__ hi, const uint16_t *__restrict__ seg_codes, int gshift,
                 int groups, int wave, int waves, int lane) {
    const __amdgpu_buffer_rsrc_t ridx = lane_rsrc(idx), rhi = lane_rsrc(hi);
    const int r = lane >> 1;
    for (uint32_t s = uint32_t(wave); s < t.segment_count; s += uint32_t(waves) * U) {
        uint32_t code[U];
        bool valid[U];
        LaneSeg sv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint32_t rel = s + uint32_t(u * waves);
            valid[u] = rel < t.segment_count;
            const uint32_t seg = t.segment_begin + (valid[u] ? rel : 0);
            code[u] = seg_codes[size_t(seg) * 32 + size_t(lane)];
            sv[u].load(ridx, rhi, seg, lane, sorted_stream_policy);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!valid[u])
                continue;
#pragma unroll 1
            for (int b = 0; b < 4; ++b) {
                sorted_u32x4 words;
                uint32_t nib;
                sv[u].batch(b, words, nib);
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    uint32_t e, d;
                    LaneSeg::entry(words, nib, jj, e, d);
                    code[u] = (code[u] + d) & 0xffffu;
                    const int g = int(e & (lane_G - 1)) + gshift;
                    if ((e & (lane_G - 1)) != lane_dummy_group && g >= 0 && g < groups)
                        ct[g * TR + r] = uint16_t(code[u]);
                }
            }
        }
    }
}
// Decodes every group tile of row tile `rt` that intersects the K window
// [gbeg, gbeg + groups) into ct (window-relative groups).
__device__ __forceinline__ void lane_decode_window(uint16_t *ct, const SortedArgs &a, int rt,
                                                   int gbeg, int groups, int wave, int waves,
                                                   int lane) {
    const int gt0 = gbeg / a.gt_size, gt1 = min(a.group_tiles - 1, (gbeg + groups - 1) / a.gt_size);
    for (int gt = gt0; gt <= gt1; ++gt)
        lane_decode_tile<2>(ct, uniform_tile(a.tiles, gt * a.row_tiles + rt), a.idx, a.planes,
                            a.codes, gt * a.gt_size - gbeg, groups, wave, waves, lane);
}
// gfx12 Wave32 v_wmma_f32_16x16x16_fp8_fp8 operand layout: lane l holds
// A[row l%16][k 8*(l/16)..+8] (8 activations of one batch row: one 8-byte
// load) and B[k 8*(l/16)..+8][col l%16] (one codebook entry of output column
// l%16); D: lane l holds rows 8*(l/16)+i of column l%16.
typedef int sorted_v2i __attribute__((ext_vector_type(2)));
typedef float sorted_v8f __attribute__((ext_vector_type(8)));
__device__ __forceinline__ sorted_v8f sorted_wmma(uint64_t a, uint64_t b, sorted_v8f c) {
    const sorted_v2i av = {int(uint32_t(a)), int(uint32_t(a >> 32))},
                     bv = {int(uint32_t(b)), int(uint32_t(b >> 32))};
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, c);
}
constexpr int sorted_lds_threads = 512, sorted_lds_group_pad = 16, sorted_lds_max_window = 1024;
constexpr int sorted_lds_bytes(int window, int row_tiles) {
    const int codes = window * TR * 2, reduction = (sorted_lds_threads / 32) * row_tiles * 1024;
    return codes > reduction ? codes : reduction;
}
// K windows: a block decodes and multiplies only `window` groups of its tile
// (window <= 1024 -> code tile <= 32 KiB -> two blocks per WGP, so one block's
// stream decode overlaps the other's gathers; the stream is scanned once per
// window). With several windows every block writes raw sums to
// partial[window][rows][out] for merge_projection; a single window writes
// y = sum * xs[row] * bs * ws[col] directly.
struct SortedLdsGeometry {
    int window = 0, windows = 0;
};
SortedLdsGeometry sorted_lds_geometry(int groups, int unit) {
    SortedLdsGeometry g;
    g.windows = int(ceil_div(size_t(groups), size_t(sorted_lds_max_window)));
    g.window = int(ceil_div(ceil_div(size_t(groups), size_t(g.windows)), size_t(unit))) * unit;
    return g;
}
// grid = (tile_count, ceil(rows / (16*RT)), windows), block = 512 threads, dynamic
// LDS = sorted_lds_bytes(window, RT). window is a multiple of 2*U; x must stay
// readable for 16*RT*gridDim.y rows and windows*window*8 columns (the runner
// pads its quantised activation buffer).
template <int RT, int U, int DB, bool LANE>
__global__ void __launch_bounds__(sorted_lds_threads)
sorted_lds_gemm(const uint8_t *__restrict__ x, const SortedArgs a, const uint8_t *__restrict__ book,
                float *__restrict__ y, const float *__restrict__ xs, const float *__restrict__ ws,
                float bs, bool apply_rows, int rows, int in, int out, int window) {
    static_assert(RT == 1 || RT == 2 || RT == 4);
    static_assert(U % 2 == 0 && U <= 8);
    extern __shared__ __attribute__((aligned(16))) uint8_t sorted_lds[];
    uint16_t *ct = reinterpret_cast<uint16_t *>(sorted_lds);
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5, waves = int(blockDim.x) >> 5;
    const int tile = int(blockIdx.x), row0 = int(blockIdx.y) * (TR * RT);
    const int gbeg = int(blockIdx.z) * window, groups = min(in / 8 - gbeg, window);
    const bool direct = gridDim.z == 1;
    if constexpr (LANE)
        lane_decode_window(ct, a, tile, gbeg, groups, wave, waves, lane);
    else
        sorted_decode_tile<DB, 4>(ct, a.tiles[tile], a.idx, a.planes, a.codes, gbeg, groups, wave,
                                  waves, lane);
    __syncthreads();
    const int col = lane & 15, gh = lane >> 4;
    const uint64_t *b64 = reinterpret_cast<const uint64_t *>(book);
    const uint64_t *x64 = reinterpret_cast<const uint64_t *>(x);
    const int units = window / (2 * U);
    // Unit = 2U groups; lane half h owns groups g0 + h*U + i (i < U), so its U
    // activation slices are contiguous (8U bytes); substep i multiplies slice i
    // of both halves (K order within a unit is arbitrary as long as A and B agree).
    struct Unit {
        uint64_t w[U], a[U][RT];
    };
    auto load = [&](int unit, Unit &un) {
        const int g0 = unit * (2 * U) + gh * U; // window-relative group
        uint32_t code[U];
#pragma unroll
        for (int i = 0; i < U; ++i)
            code[i] = ct[(g0 + i) * TR + col];
        asm volatile("" ::: "memory");
#pragma unroll
        for (int i = 0; i < U; ++i)
            un.w[i] = (g0 + i) < groups ? b64[code[i]] : 0;
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            const uint64_t *xr =
                x64 + (size_t(row0 + r * 16 + col) * size_t(in) >> 3) + size_t(gbeg + g0);
#pragma unroll
            for (int i = 0; i < U; i += 2) {
                const uint4 v = *reinterpret_cast<const uint4 *>(xr + i);
                un.a[i][r] = uint64_t(v.x) | (uint64_t(v.y) << 32);
                un.a[i + 1][r] = uint64_t(v.z) | (uint64_t(v.w) << 32);
            }
        }
        asm volatile("" ::: "memory");
    };
    sorted_v8f acc[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r)
        acc[r] = sorted_v8f{};
    Unit cur, nxt;
    int u = wave;
    if (u < units)
        load(u, cur);
    for (; u < units; u += waves) {
        if (u + waves < units)
            load(u + waves, nxt);
#pragma unroll
        for (int i = 0; i < U; ++i)
#pragma unroll
            for (int r = 0; r < RT; ++r)
                acc[r] = sorted_wmma(cur.a[i][r], cur.w[i], acc[r]);
        cur = nxt;
    }
    __syncthreads(); // every wave is done with the code tile: reuse it for the reduction
    float *red = reinterpret_cast<float *>(sorted_lds);
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            red[((wave * RT + r) * 8 + i) * 32 + lane] = acc[r][i];
    __syncthreads();
    for (int e = tid; e < RT * 256; e += int(blockDim.x)) {
        const int r = e >> 8, i = (e >> 5) & 7, l = e & 31;
        float sum = 0;
        for (int wv = 0; wv < waves; ++wv)
            sum += red[((wv * RT + r) * 8 + i) * 32 + l];
        const int row = row0 + r * 16 + 8 * (l >> 4) + i, c = tile * TR + (l & 15);
        if (row < rows && c < out) {
            if (direct)
                y[size_t(row) * size_t(out) + size_t(c)] =
                    sum * xs[row] * bs * (apply_rows ? ws[c] : 1.f);
            else
                y[(size_t(blockIdx.z) * size_t(rows) + size_t(row)) * size_t(out) + size_t(c)] =
                    sum;
        }
    }
}
// Dense FP8 weight from the sorted layout: row-major [out][in] (hipBLASLt) or
// group-major [in/8][out] x 8 (coalesced for the dense pipelined kernels).
// grid = tile_count, block = 512, dynamic LDS = gpad * 32.
template <int DB, bool LANE>
__global__ void __launch_bounds__(sorted_lds_threads)
sorted_lds_reconstruct(const SortedArgs a, const uint8_t *__restrict__ book,
                       uint8_t *__restrict__ weight, int in, int out, bool group_major) {
    extern __shared__ __attribute__((aligned(16))) uint8_t sorted_lds[];
    uint16_t *ct = reinterpret_cast<uint16_t *>(sorted_lds);
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5, waves = int(blockDim.x) >> 5;
    const int tile = int(blockIdx.x), groups = in / 8;
    if constexpr (LANE)
        lane_decode_window(ct, a, tile, 0, groups, wave, waves, lane);
    else
        sorted_decode_tile<DB, 4>(ct, a.tiles[tile], a.idx, a.planes, a.codes, 0, groups, wave,
                                  waves, lane);
    __syncthreads();
    const uint64_t *b64 = reinterpret_cast<const uint64_t *>(book);
    uint64_t *w64 = reinterpret_cast<uint64_t *>(weight);
    const int rows_here = min(TR, out - tile * TR);
    if (group_major) {
        for (int i = tid; i < groups * TR; i += int(blockDim.x)) {
            const int r = i & 15, g = i >> 4;
            if (r < rows_here)
                w64[size_t(g) * size_t(out) + size_t(tile * TR + r)] = b64[ct[g * TR + r]];
        }
    } else {
        for (int r = 0; r < rows_here; ++r)
            for (int g = tid; g < groups; g += int(blockDim.x))
                w64[size_t(tile * TR + r) * size_t(groups) + size_t(g)] = b64[ct[g * TR + r]];
    }
}
// DB: delta bits of the resident encoding (sorted: 3/4; lane: 7/9).
template <typename F> void dispatch_sorted_db(const GpuProjection &p, F &&f) {
    const int delta_bits = p.sorted_delta_bits;
    if (p.sorted_lane) {
        if (delta_bits == lane_delta_bits)
            return f.template operator()<lane_delta_bits, true>();
        throw std::invalid_argument("Lane layout delta bits must be 9");
    }
    if (delta_bits == 3)
        return f.template operator()<3, false>();
    if (delta_bits == 4)
        return f.template operator()<4, false>();
    throw std::invalid_argument("Sorted layout delta bits must be 3 or 4");
}
template <typename F> void dispatch_sorted_rows(int rows, const GpuProjection &p, F &&f) {
    dispatch_sorted_db(p, [&]<int DB, bool LANE>() {
        if (rows <= 16)
            f.template operator()<1, 8, DB, LANE>();
        else if (rows <= 32)
            f.template operator()<2, 4, DB, LANE>();
        else
            f.template operator()<4, 4, DB, LANE>();
    });
}
SortedArgs sorted_args(const GpuProjection &p) {
    SortedArgs a{};
    a.idx = p.sorted_idx.data();
    a.planes = p.sorted_planes.data();
    a.codes = p.sorted_lane ? p.sorted_segment_codes.data() : p.sorted_chunk_codes.data();
    a.tiles = p.sorted_tiles.data();
    a.row_tiles = int(ceil_div(size_t(p.out), size_t(TR)));
    a.gt_size = p.sorted_lane ? p.lane_groups_per_tile : lane_G;
    a.group_tiles = p.sorted_lane ? p.lane_group_tiles : 1;
    return a;
}
// ---------------------------------------------------------------------------
// Lane layout GEMV (rows <= R per pass): lane l owns output row l/2 of its
// 16-row tile and accumulates R batch rows in registers (fp32, via the dot4
// accumulator operand). Per chunk: the lane-private idx word and delta nibble
// (loaded a whole segment ahead: 80 bytes per lane per segment in flight), the
// per-lane code chain (VALU only), one 8-byte codebook gather (L0: the 32
// lanes of a chunk stay within a few tens of KiB and the W tiles of a block
// run in lock-step at the same code position), R x 8 bytes of activations
// from LDS (xs[group][R], slot `groups` is zero for dummies), 2R dot4. No LDS
// atomics, no fixed-point split, no scalar loads in the hot loop.
// Persistent blocks: grid.x blocks loop over (group tile, tile group, split)
// items in stride, reloading the activation tile only when the group tile
// changes and prefetching the next item's first segment; grid.y = passes of
// R rows. partial[(gt*splits+split)][rows][out].
template <int R, int W>
__global__ void __launch_bounds__(W * 32)
lane_projection(const uint8_t *__restrict__ x, const SortedArgs a, const uint8_t *__restrict__ book,
                float *__restrict__ partial, int rows, int in, int out, int splits, int items,
                int diag, int policy) {
    // diag (BEHEMOTH_LANE_DIAG, measurement only): 2 no codebook gather,
    // 4 no activation read, 8 single-line gather.
    static_assert(R == 1 || R == 2 || R == 4);
    extern __shared__ __attribute__((aligned(16))) uint64_t lane_xs[];
    constexpr int THREADS = W * 32;
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
    const int groups = in / 8, r0 = int(blockIdx.y) * R;
    const int block_groups = (a.row_tiles + W - 1) / W;
    const uint64_t *x64 = reinterpret_cast<const uint64_t *>(x);
    const __amdgpu_buffer_rsrc_t ridx = lane_rsrc(a.idx), rhi = lane_rsrc(a.planes);
    // This wave's segment range for an item (empty when the tile does not exist).
    struct Range {
        uint32_t seg_begin, s0, s1;
    };
    auto range_of = [&](int item) {
        Range rg{0, 0, 0};
        const int gt = item / (block_groups * splits), rem = item % (block_groups * splits);
        const int bg = rem / splits, split = rem % splits, rt = bg * W + wave;
        if (rt < a.row_tiles) {
            const SortedTileDesc t = uniform_tile(a.tiles, gt * a.row_tiles + rt);
            rg.seg_begin = t.segment_begin;
            rg.s0 = (t.segment_count * uint32_t(split)) / uint32_t(splits);
            rg.s1 = (t.segment_count * uint32_t(split + 1)) / uint32_t(splits);
        }
        return rg;
    };
    int loaded_gt = -1;
    LaneSeg nxt;
    bool have_next = false;
    for (int item = int(blockIdx.x); item < items; item += int(gridDim.x)) {
        const int gt = item / (block_groups * splits), rem = item % (block_groups * splits);
        const int bg = rem / splits, split = rem % splits;
        const int g0 = gt * a.gt_size, gn = min(groups - g0, a.gt_size);
        if (gt != loaded_gt) {
            if (loaded_gt >= 0)
                __syncthreads(); // every wave finished reading the previous tile
            // xs[(r/2) * (gn+1) + g] holds rows 2(r/2), 2(r/2)+1 of group g (16
            // bytes): the two 16-byte reads of an R=4 entry then spread over all
            // eight 4-bank LDS groups instead of four.
            for (int i = tid; i < (gn + 1) * R; i += THREADS) {
                const int g = i / R, r = i % R;
                const size_t slot = R == 1 ? size_t(g)
                                           : size_t(r >> 1) * size_t(gn + 1) * 2 + size_t(g) * 2 +
                                                 size_t(r & 1);
                lane_xs[slot] = (g < gn && r0 + r < rows)
                                    ? x64[size_t(r0 + r) * size_t(groups) + size_t(g0 + g)]
                                    : 0;
            }
            __syncthreads();
            loaded_gt = gt;
        }
        const int rt = bg * W + wave;
        const Range rg = range_of(item);
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r)
            acc[r] = 0.f;
        if (rg.s1 > rg.s0) {
            uint32_t code = a.codes[(size_t(rg.seg_begin) + rg.s0) * 32 + size_t(lane)];
            LaneSeg cur;
            if (have_next)
                cur = nxt;
            else
                cur.load(ridx, rhi, rg.seg_begin + rg.s0, lane, policy);
            have_next = false;
            for (uint32_t sg = rg.s0; sg < rg.s1; ++sg) {
                // Prefetch: the next segment of this item, or the first one of
                // this wave's next item (so the stream never drains).
                if (sg + 1 < rg.s1)
                    nxt.load(ridx, rhi, rg.seg_begin + sg + 1, lane, policy);
                else {
                    const int next_item = item + int(gridDim.x);
                    if (next_item < items) {
                        const Range nr = range_of(next_item);
                        if (nr.s1 > nr.s0) {
                            nxt.load(ridx, rhi, nr.seg_begin + nr.s0, lane, policy);
                            have_next = true;
                        }
                    }
                }
#pragma unroll 1
                for (int b = 0; b < 4; ++b) {
                    sorted_u32x4 words;
                    uint32_t nib;
                    cur.batch(b, words, nib);
                    uint32_t off[8], g[8];
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        uint32_t e, d;
                        LaneSeg::entry(words, nib, jj, e, d);
                        code = (code + d) & 0xffffu;
                        off[jj] = (diag & 8 ? (code & 63u) : code) << 3;
                        g[jj] = min(e & uint32_t(lane_G - 1), uint32_t(gn)) * (R == 1 ? 1 : 2);
                    }
                    uint2 w[8];
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj)
                        w[jj] = diag & 2 ? make_uint2(off[jj], g[jj])
                                         : *reinterpret_cast<const uint2 *>(book + off[jj]);
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        // R x 8 activation bytes per entry: 16-byte LDS reads.
                        uint32_t xv[2 * R];
                        if (diag & 4) {
#pragma unroll
                            for (int i = 0; i < 2 * R; ++i)
                                xv[i] = g[jj] + uint32_t(i);
                        } else if constexpr (R == 1) {
                            const uint2 v = *reinterpret_cast<const uint2 *>(lane_xs + g[jj]);
                            xv[0] = v.x;
                            xv[1] = v.y;
                        } else {
#pragma unroll
                            for (int q = 0; q < R / 2; ++q) {
                                const uint4 v = *reinterpret_cast<const uint4 *>(
                                    lane_xs + size_t(q) * size_t(gn + 1) * 2 + g[jj]);
                                xv[4 * q] = v.x;
                                xv[4 * q + 1] = v.y;
                                xv[4 * q + 2] = v.z;
                                xv[4 * q + 3] = v.w;
                            }
                        }
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            acc[r] = __builtin_amdgcn_dot4_f32_fp8_fp8(w[jj].x, xv[2 * r], acc[r]);
                            acc[r] = __builtin_amdgcn_dot4_f32_fp8_fp8(w[jj].y, xv[2 * r + 1], acc[r]);
                        }
                    }
                }
                if (sg + 1 < rg.s1)
                    cur = nxt;
            }
        }
        if (rt >= a.row_tiles)
            continue;
        // The two lanes of a row hold its two half-streams.
#pragma unroll
        for (int r = 0; r < R; ++r)
            acc[r] += __shfl_xor(acc[r], 1);
        const int row_out = rt * TR + (lane >> 1);
        if ((lane & 1) == 0 && row_out < out) {
            const size_t part = size_t(gt * splits + split);
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (r0 + r < rows)
                    partial[(part * size_t(rows) + size_t(r0 + r)) * size_t(out) + size_t(row_out)] =
                        acc[r];
        }
    }
}
template <typename F> void dispatch_lane_rows(int rows, F &&f) {
    if (rows <= 1)
        f.template operator()<1, 8>();
    else if (rows <= 2)
        f.template operator()<2, 8>();
    else
        f.template operator()<4, 16>();
}
// Persistent-block count: at most `cap`, and a divisor of `items` when one
// exists in [cap/2, cap] so every block does the same number of items.
int lane_blocks(int items, int cap) {
    if (items <= cap)
        return items;
    for (int g = cap; g >= cap / 2; --g)
        if (items % g == 0)
            return g;
    return cap;
}
// Sums the split/window/tile partials and applies the scales; writes fp32 or
// bf16 (the fused cast of the projection's consumer).
template <typename T>
__global__ void merge_projection(const float *partial, const float *xs, const float *ws, float bs,
                                 T *y, int rows, int out, int parts, bool apply_rows) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * size_t(out))
        return;
    float sum = 0;
    for (int part = 0; part < parts; ++part)
        sum += partial[size_t(part) * size_t(rows) * size_t(out) + i];
    const float v = sum * xs[i / size_t(out)] * bs * (apply_rows ? ws[i % size_t(out)] : 1.f);
    if constexpr (std::is_same_v<T, float>)
        y[i] = v;
    else
        y[i] = device::bf16(v);
}
} // namespace
ProjectionRunner::ProjectionRunner() {
    auto s = std::make_unique<RunnerState>();
    BLAS_CHECK(hipblasLtCreate(&s->handle));
    handle_ = s.release();
}
ProjectionRunner::~ProjectionRunner() {
    auto *s = static_cast<RunnerState *>(handle_);
    if (s) {
        s->plans.clear();
        hipblasLtDestroy(s->handle);
        delete s;
    }
}
void ProjectionRunner::set_mode(ProjectionMode mode) {
    (void)projection_mode_name(mode);
    auto &state = *static_cast<RunnerState *>(handle_);
    if (state.override_mode != mode) {
        state.override_mode = mode;
        state.compact_profile = nullptr;
    }
}
nlohmann::json ProjectionRunner::resources() const {
    auto &s = *static_cast<RunnerState *>(handle_);
    if (!s.resources.is_null())
        return s.resources;
    s.resources = nlohmann::json::array();
    auto add = [&](const std::string &name, const void *kernel, int threads) {
        hipFuncAttributes a{};
        auto status = hipFuncGetAttributes(&a, kernel);
        if (status != hipSuccess) {
            s.resources.push_back({{"mode", name}, {"error", hipGetErrorString(status)}});
            (void)hipGetLastError();
            return;
        }
        s.resources.push_back({{"mode", name},
                               {"threads_per_block", threads},
                               {"static_shared_bytes", a.sharedSizeBytes},
                               {"registers_per_thread", a.numRegs},
                               {"local_bytes", a.localSizeBytes}});
        int blocks = 0;
        status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0);
        auto &entry = s.resources.back();
        if (status == hipSuccess) {
            entry["occupancy_max_blocks_per_multiprocessor"] = blocks;
            entry["occupancy_max_waves_per_multiprocessor"] =
                blocks * (threads / device::wave_size);
        } else {
            entry["occupancy_error"] = hipGetErrorString(status);
            (void)hipGetLastError();
        }
    };
    add("fused", reinterpret_cast<const void *>(fused_projection<false>), 32);
    add("split-k", reinterpret_cast<const void *>(fused_projection<true>), 32);
    add("tiled-64", reinterpret_cast<const void *>(tiled_projection<64>), 128);
    add("tiled-128", reinterpret_cast<const void *>(tiled_projection<128>), 128);
    add("tiled-lds-64", reinterpret_cast<const void *>(tiled_projection<64, true>), 128);
    add("tiled-lds-128", reinterpret_cast<const void *>(tiled_projection<128, true>), 128);
    add("sorted-s4/s8/s16",
        reinterpret_cast<const void *>(
            sorted_projection<sorted_waves, sorted_unroll, sorted_copies, 4>),
        sorted_waves * 32);
    add("sorted-dense (reconstruct)",
        reinterpret_cast<const void *>(sorted_lds_reconstruct<4, false>), sorted_lds_threads);
    add("sorted-dense (reconstruct, lane layout)",
        reinterpret_cast<const void *>(sorted_lds_reconstruct<9, true>), sorted_lds_threads);
    add("sorted-lds (rows<=16)", reinterpret_cast<const void *>(sorted_lds_gemm<1, 8, 4, false>),
        sorted_lds_threads);
    add("sorted-lds (rows<=32)", reinterpret_cast<const void *>(sorted_lds_gemm<2, 4, 4, false>),
        sorted_lds_threads);
    add("sorted-lds (rows<=64 per pass)",
        reinterpret_cast<const void *>(sorted_lds_gemm<4, 4, 4, false>), sorted_lds_threads);
    add("sorted-lds (rows<=16, lane layout)",
        reinterpret_cast<const void *>(sorted_lds_gemm<1, 8, 9, true>), sorted_lds_threads);
    add("lane-sN (rows=1)", reinterpret_cast<const void *>(lane_projection<1, 8>), 256);
    add("lane-sN (rows=2)", reinterpret_cast<const void *>(lane_projection<2, 8>), 256);
    add("lane-sN (rows<=4 per pass)", reinterpret_cast<const void *>(lane_projection<4, 16>), 512);
    for (const auto &g : pipeline_geometries)
        dispatch_pipeline(g.mode, [&]<int k, int n, int split, int rt = 1>() {
            const auto name = std::string(projection_mode_name(g.mode));
            add(name + "-aligned",
                reinterpret_cast<const void *>(pipelined_projection<k, true, n, split, rt>), n * 2);
            add(name + "-generic",
                reinterpret_cast<const void *>(pipelined_projection<k, false, n, split, rt>),
                n * 2);
        });
    return s.resources;
}
void ProjectionRunner::set_search(ProjectionSearch search) {
    (void)projection_search_name(search);
    auto &state = *static_cast<RunnerState *>(handle_);
    if (state.search != search) {
        state.search = search;
        state.choices.clear();
        state.compact_profile = nullptr;
    }
}
nlohmann::json ProjectionRunner::selection(const GpuProjection &p, int rows,
                                           bool apply_rows) const {
    const auto &state = *static_cast<const RunnerState *>(handle_);
    auto key = selection_key(p, rows, apply_rows);
    auto it = state.choices.find(key);
    if (it == state.choices.end())
        throw std::logic_error("Projection has not run with this selection key");
    return choice_report(key, it->second, state.plans.at(std::get<0>(key)), state);
}
nlohmann::json ProjectionRunner::profile(bool detailed) const {
    const auto &state = *static_cast<const RunnerState *>(handle_);
    auto result = nlohmann::json::array();
    if (detailed) {
        for (const auto &[key, choice] : state.choices)
            result.push_back(choice_report(key, choice, state.plans.at(std::get<0>(key)), state));
        return result;
    }
    if (!state.compact_profile.is_null())
        return state.compact_profile;
    // The server can have one choice per weight and batch size. Group its
    // diagnostic output instead of rebuilding every sample array each token.
    std::map<std::tuple<Key, bool, ProjectionMode>, nlohmann::json> groups;
    for (const auto &[selection, choice] : state.choices) {
        const auto &[key, identity, apply_rows] = selection;
        const auto [in, out, rows, compressed, layout] = key;
        auto mode =
            state.override_mode == ProjectionMode::Auto ? choice.selected : state.override_mode;
        auto &group = groups[{key, apply_rows, mode}];
        if (group.is_null())
            group = {{"in", in},
                     {"out", out},
                     {"rows", rows},
                     {"compressed", compressed},
                     {"code_layout", compressed ? code_layout_name(layout) : "dense"},
                     {"cache_scope", identity ? "weight" : "shape"},
                     {"apply_row_scale", apply_rows},
                     {"selected", projection_mode_name(mode)},
                     {"search", projection_search_name(state.search)},
                     {"selection_count", 0},
                     {"selected_ms_min", nullptr},
                     {"selected_ms_max", nullptr}};
        group["selection_count"] = group["selection_count"].get<size_t>() + 1;
        const float ms = choice.milliseconds[size_t(int(mode) - 1)];
        if (std::isfinite(ms)) {
            if (group["selected_ms_min"].is_null() || ms < group["selected_ms_min"].get<float>())
                group["selected_ms_min"] = ms;
            if (group["selected_ms_max"].is_null() || ms > group["selected_ms_max"].get<float>())
                group["selected_ms_max"] = ms;
        }
    }
    for (auto &[key, group] : groups)
        result.push_back(std::move(group));
    state.compact_profile = result;
    return result;
}
std::pair<uint8_t *, float *> ProjectionRunner::prepare_input(ProjectionWorkspace &w, int rows,
                                                               int in, hipStream_t stream) {
    if (rows < 1 || in < 8 || in % 8)
        throw std::invalid_argument("Invalid projection input shape");
    // The quantised activations are padded to whole 64-row groups plus one
    // K step: sorted-lds reads whole 16-row WMMA tiles and whole group units
    // past the real rows/columns (their weights are zero, so any finite
    // padding contributes nothing). Fresh allocations are zeroed once; stale
    // rows hold earlier finite activations.
    const size_t padded = size_t(ceil_div(size_t(rows), size_t(64))) * 64 * size_t(in) + 4096;
    const uint8_t *before = w.input.data();
    w.input.resize(padded);
    if (w.input.data() != before)
        HIP_CHECK(hipMemsetAsync(w.input.data(), 0, w.input.bytes(), stream));
    w.input_scale.resize(size_t(rows));
    return {w.input.data(), w.input_scale.data()};
}
void ProjectionRunner::run(const GpuProjection &p, const BFloat16 *input, float *output, int rows,
                           ProjectionWorkspace &w, hipStream_t stream, bool apply_rows) {
    if (!input)
        throw std::invalid_argument("Projection input missing");
    Call call;
    call.input = input;
    call.output = output;
    call.apply_row_scale = apply_rows;
    run(p, call, rows, w, stream);
}
void ProjectionRunner::run(const GpuProjection &p, const Call &call, int rows,
                           ProjectionWorkspace &w, hipStream_t stream) {
    const BFloat16 *input = call.input;
    float *output = call.output;
    BFloat16 *bf16_output = call.bf16_output;
    const bool apply_rows = call.apply_row_scale;
    const bool has_codes = p.codes.size() == size_t(p.out) * size_t(p.in / 8);
    if (rows < 1 || p.in < 1 || p.out < 1 || p.in % 8 || !output ||
        p.row_scale.size() != size_t(p.out) ||
        (p.dense.size() != size_t(p.in) * size_t(p.out) &&
         ((!has_codes && !p.has_sorted()) || p.codebook.size() != 65536 * 8)))
        throw std::invalid_argument("Invalid FP8 projection");
    auto &state = *static_cast<RunnerState *>(handle_);
    auto &plan = state.plans[shape_key(p, rows)];
    auto [choice_it, inserted] = state.choices.try_emplace(selection_key(p, rows, apply_rows));
    if (inserted)
        state.compact_profile = nullptr;
    auto &choice = choice_it->second;
    const bool group_major = p.code_layout == CodeLayout::GroupMajor;
    if (input)
        prepare_input(w, rows, p.in, stream);
    else if (w.input.size() < size_t(rows) * size_t(p.in) || w.input_scale.size() < size_t(rows))
        throw std::invalid_argument("Projection workspace holds no quantised input");
    // Merge-based modes write the bf16 result directly; the others cast after.
    bool merged_bf16 = false;
    auto merge = [&](const float *partial, int parts) {
        const unsigned blocks = unsigned(ceil_div(size_t(rows) * size_t(p.out), 256));
        if (bf16_output) {
            merge_projection<uint16_t><<<blocks, 256, 0, stream>>>(
                partial, w.input_scale.data(), p.row_scale.data(), p.codebook_scale,
                reinterpret_cast<uint16_t *>(bf16_output), rows, p.out, parts, apply_rows);
            merged_bf16 = true;
        } else
            merge_projection<float><<<blocks, 256, 0, stream>>>(
                partial, w.input_scale.data(), p.row_scale.data(), p.codebook_scale, output, rows,
                p.out, parts, apply_rows);
        HIP_CHECK(hipGetLastError());
    };
    // Dense FP8 weights for the BLAS / sorted-dense paths: stored dense, or
    // reconstructed from the group-major codes (random gather) or the sorted
    // layout (streaming) into the shared scratch.
    // `dense_gm` selects the scratch layout: row-major [out][in] for hipBLASLt,
    // group-major for the pipelined dense kernels (coalesced column loads).
    auto dense_weight = [&](bool dense_gm) -> const uint8_t * {
        if (p.dense.size())
            return p.dense.data();
        w.weight.resize(size_t(p.out) * size_t(p.in));
        if (has_codes && !dense_gm)
            reconstruct_kernel<<<unsigned(ceil_div(w.weight.size(), 256)), 256, 0, stream>>>(
                p.codes.data(), p.codebook.data(), w.weight.data(), p.in, p.out, group_major);
        else if (p.has_sorted())
            dispatch_sorted_db(p, [&]<int DB, bool LANE>() {
                const int gpad = int(ceil_div(size_t(p.in / 8), size_t(sorted_lds_group_pad))) *
                                 sorted_lds_group_pad;
                static bool attribute_set = false;
                if (!attribute_set) {
                    HIP_CHECK(hipFuncSetAttribute(
                        reinterpret_cast<const void *>(sorted_lds_reconstruct<DB, LANE>),
                        hipFuncAttributeMaxDynamicSharedMemorySize, 65536));
                    attribute_set = true;
                }
                const auto args = sorted_args(p);
                sorted_lds_reconstruct<DB, LANE>
                    <<<unsigned(args.row_tiles), sorted_lds_threads, size_t(gpad) * TR * 2,
                       stream>>>(args, p.codebook.data(), w.weight.data(), p.in, p.out, dense_gm);
            });
        else
            throw ProjectionUnavailable("No source for a dense weight");
        HIP_CHECK(hipGetLastError());
        return w.weight.data();
    };
    auto launch = [&](ProjectionMode mode) {
        const uint8_t *dense = p.dense.data();
        bool kernel_gm = group_major; // code layout, or the dense scratch layout when dense
        if (dense)
            kernel_gm = false; // stored dense weights (lm_head) are row-major
        if (projection_mode_is_sorted(mode)) {
            if (!p.has_sorted())
                throw ProjectionUnavailable("Projection has no codebook-major (sorted) layout");
            const int needs = projection_mode_layout(mode);
            if (needs == 1 && p.sorted_lane)
                throw ProjectionUnavailable("sorted-sN modes need the sorted (chunk-code) layout");
            if (needs == 2 && !p.sorted_lane)
                throw ProjectionUnavailable("lane-sN modes need the lane layout");
            if (needs == 2) {
                const int splits = mode == ProjectionMode::LaneS1   ? 1
                                   : mode == ProjectionMode::LaneS2 ? 2
                                   : mode == ProjectionMode::LaneS4 ? 4
                                                                    : 8;
                const auto args = sorted_args(p);
                const int parts = args.group_tiles * splits;
                w.partial.resize(size_t(rows) * size_t(p.out) * size_t(parts));
                dispatch_lane_rows(rows, [&]<int R, int W>() {
                    static bool attribute_set = false; // one per instantiation
                    if (!attribute_set) {
                        HIP_CHECK(hipFuncSetAttribute(
                            reinterpret_cast<const void *>(lane_projection<R, W>),
                            hipFuncAttributeMaxDynamicSharedMemorySize, 65536));
                        attribute_set = true;
                    }
                    const int block_groups = int(ceil_div(size_t(args.row_tiles), size_t(W)));
                    const int items = args.group_tiles * block_groups * splits;
                    dim3 grid(unsigned(lane_blocks(items, R == 4 ? 64 : 128)),
                              unsigned(ceil_div(size_t(rows), size_t(R))));
                    const size_t lds = size_t(args.gt_size + 1) * R * sizeof(uint64_t);
                    static const int diag = [] {
                        const char *v = std::getenv("BEHEMOTH_LANE_DIAG");
                        return v ? std::atoi(v) : 0;
                    }();
                    static const int policy = [] {
                        const char *v = std::getenv("BEHEMOTH_LANE_POLICY");
                        return v ? std::atoi(v) : lane_stream_policy;
                    }();
                    lane_projection<R, W><<<grid, W * 32, lds, stream>>>(
                        w.input.data(), args, p.codebook.data(), w.partial.data(), rows, p.in,
                        p.out, splits, items, diag, policy);
                });
                HIP_CHECK(hipGetLastError());
                merge(w.partial.data(), parts);
                return;
            }
            if (mode == ProjectionMode::SortedDense) {
                dense = dense_weight(true);
                kernel_gm = true;
                mode = rows <= 16   ? ProjectionMode::Pipelined64
                       : rows <= 32 ? ProjectionMode::PipelinedR2K64
                                    : ProjectionMode::PipelinedR4K64;
            } else if (mode == ProjectionMode::SortedLds) {
                dispatch_sorted_rows(rows, p, [&]<int RT, int U, int DB, bool LANE>() {
                    const auto g = sorted_lds_geometry(p.in / 8, 2 * U);
                    const size_t lds = size_t(sorted_lds_bytes(g.window, RT));
                    static bool attribute_set = false; // one per instantiation
                    if (!attribute_set) {
                        HIP_CHECK(hipFuncSetAttribute(
                            reinterpret_cast<const void *>(sorted_lds_gemm<RT, U, DB, LANE>),
                            hipFuncAttributeMaxDynamicSharedMemorySize, 65536));
                        attribute_set = true;
                    }
                    const auto args = sorted_args(p);
                    dim3 grid(unsigned(args.row_tiles),
                              unsigned(ceil_div(size_t(rows), size_t(TR * RT))),
                              unsigned(g.windows));
                    float *target = output;
                    if (g.windows > 1) {
                        w.partial.resize(size_t(rows) * size_t(p.out) * size_t(g.windows));
                        target = w.partial.data();
                    }
                    sorted_lds_gemm<RT, U, DB, LANE><<<grid, sorted_lds_threads, lds, stream>>>(
                        w.input.data(), args, p.codebook.data(), target, w.input_scale.data(),
                        p.row_scale.data(), p.codebook_scale, apply_rows, rows, p.in, p.out,
                        g.window);
                    HIP_CHECK(hipGetLastError());
                    if (g.windows > 1)
                        merge(w.partial.data(), g.windows);
                });
                return;
            } else {
                const int splits = mode == ProjectionMode::SortedS4    ? 4
                                   : mode == ProjectionMode::SortedS8  ? 8
                                                                       : 16;
                const int block_groups = int(ceil_div(size_t(p.sorted_tile_count), sorted_waves));
                dim3 grid(unsigned(block_groups * splits), unsigned(rows));
                w.partial.resize(size_t(rows) * size_t(p.out) * size_t(splits));
                dispatch_sorted_db(p, [&]<int DB, bool LANE>() {
                    if constexpr (LANE)
                        throw std::logic_error("sorted-sN dispatched on the lane layout");
                    else
                    sorted_projection<sorted_waves, sorted_unroll, sorted_copies, DB>
                        <<<grid, sorted_waves * 32, 0, stream>>>(
                            w.input.data(), p.sorted_idx.data(), p.sorted_planes.data(),
                            p.sorted_chunk_codes.data(), p.sorted_tiles.data(), p.codebook.data(),
                            w.partial.data(), rows, p.in, p.out, p.sorted_tile_count, splits);
                });
                HIP_CHECK(hipGetLastError());
                merge(w.partial.data(), splits);
                return;
            }
        } else if (!dense && !has_codes && mode != ProjectionMode::Blas) {
            throw ProjectionUnavailable("Projection has no codes for this mode (sorted layout only)");
        }
        const auto *geometry = pipeline_geometry(mode);
        dim3 grid(unsigned(ceil_div(size_t(p.out), 16)), unsigned(ceil_div(size_t(rows), 16)));
        if (mode == ProjectionMode::Fused) {
            fused_projection<false><<<grid, device::wave_size, 0, stream>>>(
                w.input.data(), w.input_scale.data(), p.codes.data(), p.codebook.data(),
                dense, p.row_scale.data(), p.codebook_scale, output, rows, p.in, p.out,
                apply_rows, 1, kernel_gm);
            HIP_CHECK(hipGetLastError());
        } else if (mode == ProjectionMode::SplitK) {
            int parts = int(ceil_div(size_t(p.in), split_width));
            grid.z = unsigned(parts);
            w.partial.resize(size_t(rows) * size_t(p.out) * size_t(parts));
            fused_projection<true><<<grid, device::wave_size, 0, stream>>>(
                w.input.data(), w.input_scale.data(), p.codes.data(), p.codebook.data(),
                dense, p.row_scale.data(), p.codebook_scale, w.partial.data(), rows, p.in,
                p.out, apply_rows, parts, kernel_gm);
            HIP_CHECK(hipGetLastError());
            merge(w.partial.data(), parts);
        } else if (mode == ProjectionMode::Tiled64 || mode == ProjectionMode::Tiled128 ||
                   mode == ProjectionMode::TiledLds64 || mode == ProjectionMode::TiledLds128 ||
                   geometry) {
            const int partition = geometry ? geometry->split : tiled_split_width;
            const int columns = geometry ? geometry->columns : 64;
            const int block_rows = 16 * (geometry ? geometry->row_tiles : 1);
            int parts = int(ceil_div(size_t(p.in), size_t(partition)));
            grid.x = unsigned(ceil_div(size_t(p.out), size_t(columns)));
            grid.y = unsigned(ceil_div(size_t(rows), size_t(block_rows)));
            grid.z = unsigned(parts);
            w.partial.resize(size_t(rows) * size_t(p.out) * size_t(parts));
            if (mode == ProjectionMode::Tiled64)
                tiled_projection<64><<<grid, 128, 0, stream>>>(
                    w.input.data(), p.codes.data(), p.codebook.data(), dense,
                    w.partial.data(), rows, p.in, p.out, parts, kernel_gm);
            else if (mode == ProjectionMode::Tiled128)
                tiled_projection<128><<<grid, 128, 0, stream>>>(
                    w.input.data(), p.codes.data(), p.codebook.data(), dense,
                    w.partial.data(), rows, p.in, p.out, parts, kernel_gm);
            else if (mode == ProjectionMode::TiledLds64)
                tiled_projection<64, true><<<grid, 128, 0, stream>>>(
                    w.input.data(), p.codes.data(), p.codebook.data(), dense,
                    w.partial.data(), rows, p.in, p.out, parts, kernel_gm);
            else if (mode == ProjectionMode::TiledLds128)
                tiled_projection<128, true><<<grid, 128, 0, stream>>>(
                    w.input.data(), p.codes.data(), p.codebook.data(), dense,
                    w.partial.data(), rows, p.in, p.out, parts, kernel_gm);
            else {
                dispatch_pipeline(mode, [&]<int k, int n, int split, int rt = 1>() {
                    const bool aligned = !dense && kernel_gm && p.out % n == 0 &&
                                         p.in % k == 0 && p.codes.size() <= UINT32_MAX;
                    if (aligned)
                        pipelined_projection<k, true, n, split, rt><<<grid, n * 2, 0, stream>>>(
                            w.input.data(), p.codes.data(), p.codebook.data(), dense,
                            w.partial.data(), rows, p.in, p.out, kernel_gm);
                    else
                        pipelined_projection<k, false, n, split, rt><<<grid, n * 2, 0, stream>>>(
                            w.input.data(), p.codes.data(), p.codebook.data(), dense,
                            w.partial.data(), rows, p.in, p.out, kernel_gm);
                });
            }
            HIP_CHECK(hipGetLastError());
            merge(w.partial.data(), parts);
        } else if (mode == ProjectionMode::Blas) {
            ensure_blas(state, plan, p.in, p.out, rows);
            if (!plan.has_blas)
                throw ProjectionUnavailable("No native FP8 hipBLASLt algorithm for this shape");
            const uint8_t *weight = dense_weight(false);
            w.blas.resize(plan.algorithm.workspaceSize);
            auto &d = *plan.descriptors;
            float one = 1, zero = 0;
            BLAS_CHECK(hipblasLtMatmul(state.handle, d.op, &one, weight, d.a, w.input.data(), d.b,
                                       &zero, output, d.c, output, d.c, &plan.algorithm.algo,
                                       w.blas.data(), w.blas.size(), stream));
            scale_output<<<unsigned(ceil_div(size_t(rows) * size_t(p.out), 256)), 256, 0, stream>>>(
                output, w.input_scale.data(), p.row_scale.data(), p.codebook_scale, rows, p.out,
                apply_rows);
            HIP_CHECK(hipGetLastError());
        }
    };
    auto execute = [&](ProjectionMode mode) {
        if (input)
            quantize_rows(input, w.input.data(), w.input_scale.data(), rows, p.in, stream);
        merged_bf16 = false;
        launch(mode);
        if (bf16_output && !merged_bf16)
            cast_scaled(output, nullptr, bf16_output, rows, p.out, stream);
    };
    if (state.override_mode != ProjectionMode::Auto) {
        execute(state.override_mode);
        return;
    }
    if (!choice.tuned) {
        ensure_blas(state, plan, p.in, p.out, rows);
        std::vector<ProjectionMode> candidates;
        if (!p.dense.size() && rows <= 16) {
            // sorted-sN streams the layout once per row (exact int64): the
            // batch-1/2 choice; sorted-lds decodes once and shares the gather
            // across the batch rows.
            if (p.has_lane()) {
                // Register-accumulating lane kernel: one pass per 4 rows.
                candidates.push_back(ProjectionMode::LaneS2);
                candidates.push_back(ProjectionMode::LaneS4);
                candidates.push_back(ProjectionMode::LaneS8);
                if (rows <= 4)
                    candidates.push_back(ProjectionMode::LaneS1);
                if (rows >= 5) // two or more 4-row passes: compare the batched kernel
                    candidates.push_back(ProjectionMode::SortedLds);
            } else if (p.has_sorted()) {
                candidates.push_back(ProjectionMode::SortedLds);
                if (rows <= 8) {
                    candidates.push_back(ProjectionMode::SortedS8);
                    candidates.push_back(ProjectionMode::SortedS16);
                }
                if (rows <= 2)
                    candidates.push_back(ProjectionMode::SortedS4);
            }
            if (has_codes) {
                candidates.push_back(ProjectionMode::Pipelined64);
                candidates.push_back(ProjectionMode::Pipelined128);
                if (p.in < 1024 || p.out < 512)
                    candidates.push_back(ProjectionMode::Fused);
            }
            if (rows >= 4 && plan.has_blas)
                candidates.push_back(ProjectionMode::Blas);
        } else if (!p.dense.size() && rows <= 64) {
            // Lookup cost is shared across 32/64 rows; still compare BLAS.
            if (has_codes)
                candidates = rows <= 32 ? std::vector{ProjectionMode::PipelinedR2K128,
                                                      ProjectionMode::PipelinedR2K64,
                                                      ProjectionMode::PipelinedR4K128}
                                        : std::vector{ProjectionMode::PipelinedR4K128,
                                                      ProjectionMode::PipelinedR4K64};
            if (p.has_sorted()) {
                candidates.push_back(ProjectionMode::SortedLds);
                candidates.push_back(ProjectionMode::SortedDense);
                if (p.has_lane() && rows <= 32)
                    candidates.push_back(ProjectionMode::LaneS4); // ceil(rows/4) passes
            }
            if (plan.has_blas)
                candidates.push_back(ProjectionMode::Blas);
            if (has_codes)
                candidates.push_back(ProjectionMode::Fused);
        } else {
            if (plan.has_blas)
                candidates.push_back(ProjectionMode::Blas);
            if (has_codes || p.dense.size())
                candidates.push_back(ProjectionMode::Fused);
            // Prefill-sized batches re-gather once per 64 rows: cheaper than a
            // dense reconstruct + GEMM up to a few hundred rows.
            if (has_codes && rows <= 512)
                candidates.push_back(ProjectionMode::PipelinedR4K64);
            if (p.has_sorted()) {
                // Sorted layout: decode + coalesced dense scratch, then hipBLASLt
                // or the dense WMMA kernels; or the batched kernel per 64 rows.
                candidates.push_back(ProjectionMode::SortedDense);
                if (rows <= 512)
                    candidates.push_back(ProjectionMode::SortedLds);
            }
        }
        if (state.search == ProjectionSearch::Exhaustive && rows <= 64)
            for (auto mode : projection_modes()) {
                if (mode == ProjectionMode::Auto ||
                    (mode == ProjectionMode::Blas && !plan.has_blas) ||
                    (mode == ProjectionMode::SplitK && p.in <= split_width) ||
                    (projection_mode_is_sorted(mode) && !p.has_sorted()) ||
                    (projection_mode_layout(mode) == 1 && p.sorted_lane) ||
                    (projection_mode_layout(mode) == 2 && !p.sorted_lane) ||
                    (projection_mode_is_sorted(mode) && mode != ProjectionMode::SortedDense &&
                     mode != ProjectionMode::SortedLds && rows > 16) ||
                    (!projection_mode_is_sorted(mode) && mode != ProjectionMode::Blas &&
                     !has_codes && !p.dense.size()))
                    continue;
                if (std::find(candidates.begin(), candidates.end(), mode) == candidates.end())
                    candidates.push_back(mode);
            }
        choice.candidates = candidates;
        choice.batch_runs.assign(candidates.size(), 1);
        // Allocate every candidate's scratch and warm both candidate orders
        // before starting the timing events. Actual calls include quantization.
        for (size_t pass = 0; pass < 2; ++pass)
            for (size_t i = 0; i < candidates.size(); ++i)
                execute(candidates[pass ? candidates.size() - 1 - i : i]);
        HIP_CHECK(hipStreamSynchronize(stream));
        hipEvent_t start{}, end{};
        HIP_CHECK(hipEventCreate(&start));
        try {
            HIP_CHECK(hipEventCreate(&end));
            auto measure = [&](size_t index, int runs) {
                execute(candidates[index]); // self-warm after the preceding candidate
                HIP_CHECK(hipEventRecord(start, stream));
                for (int i = 0; i < runs; ++i)
                    execute(candidates[index]);
                HIP_CHECK(hipEventRecord(end, stream));
                HIP_CHECK(hipEventSynchronize(end));
                float elapsed = 0;
                HIP_CHECK(hipEventElapsedTime(&elapsed, start, end));
                const double per_call = double(elapsed) / runs;
                if (!std::isfinite(per_call) || per_call <= 0)
                    throw std::runtime_error("Invalid projection autotuning event duration");
                return per_call;
            };
            for (size_t i = 0; i < candidates.size(); ++i) {
                const double pilot = measure(i, 1);
                choice.batch_runs[i] = int(std::clamp(std::ceil(.5 / pilot), 1., 32.));
            }
            choice.decision = tune_candidates(candidates.size(), [&](size_t index) {
                return measure(index, choice.batch_runs[index]);
            });
            for (size_t i = 0; i < candidates.size(); ++i)
                choice.milliseconds[size_t(int(candidates[i]) - 1)] =
                    float(choice.decision.measurements[i].median_ms());
            choice.selected = candidates[choice.decision.selected];
            choice.tuned = true;
            state.compact_profile = nullptr;
            (void)hipEventDestroy(end);
            (void)hipEventDestroy(start);
        } catch (...) {
            if (end)
                (void)hipEventDestroy(end);
            (void)hipEventDestroy(start);
            throw;
        }
    }
    execute(choice.selected);
}
} // namespace behemoth
