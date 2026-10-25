#include "behemoth/gpu.hpp"
#include "device_numeric.hpp"
#include <array>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <rocwmma/rocwmma.hpp>
#include <tuple>
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
    case ProjectionMode::Count:
        break;
    }
    throw std::invalid_argument("Invalid projection mode");
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
                if (dense)
                    packed = reinterpret_cast<const uint64_t *>(
                        dense)[size_t(col) * size_t(in / 8) + size_t(k / 8)];
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
                        dense)[size_t(col0 + col) * size_t(in / 8) + size_t((base + k) / 8)];
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
                                        dense)[size_t(col) * size_t(in / 8) + size_t(k / 8)]
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
__global__ void merge_projection(const float *partial, const float *xs, const float *ws, float bs,
                                 float *y, int rows, int out, int parts, bool apply_rows) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * size_t(out))
        return;
    float sum = 0;
    for (int part = 0; part < parts; ++part)
        sum += partial[size_t(part) * size_t(rows) * size_t(out) + i];
    y[i] = sum * xs[i / size_t(out)] * bs * (apply_rows ? ws[i % size_t(out)] : 1.f);
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
void ProjectionRunner::run(const GpuProjection &p, const BFloat16 *input, float *output, int rows,
                           ProjectionWorkspace &w, hipStream_t stream, bool apply_rows) {
    if (rows < 1 || p.in < 1 || p.out < 1 || p.in % 8 || !input || !output ||
        p.row_scale.size() != size_t(p.out) ||
        (p.dense.size() != size_t(p.in) * size_t(p.out) &&
         (p.codes.size() != size_t(p.out) * size_t(p.in / 8) || p.codebook.size() != 65536 * 8)))
        throw std::invalid_argument("Invalid FP8 projection");
    auto &state = *static_cast<RunnerState *>(handle_);
    auto &plan = state.plans[shape_key(p, rows)];
    auto [choice_it, inserted] = state.choices.try_emplace(selection_key(p, rows, apply_rows));
    if (inserted)
        state.compact_profile = nullptr;
    auto &choice = choice_it->second;
    const bool group_major = p.code_layout == CodeLayout::GroupMajor;
    w.input.resize(size_t(rows) * size_t(p.in));
    w.input_scale.resize(size_t(rows));
    auto launch = [&](ProjectionMode mode) {
        const auto *geometry = pipeline_geometry(mode);
        dim3 grid(unsigned(ceil_div(size_t(p.out), 16)), unsigned(ceil_div(size_t(rows), 16)));
        if (mode == ProjectionMode::Fused) {
            fused_projection<false><<<grid, device::wave_size, 0, stream>>>(
                w.input.data(), w.input_scale.data(), p.codes.data(), p.codebook.data(),
                p.dense.data(), p.row_scale.data(), p.codebook_scale, output, rows, p.in, p.out,
                apply_rows, 1, group_major);
            HIP_CHECK(hipGetLastError());
        } else if (mode == ProjectionMode::SplitK) {
            int parts = int(ceil_div(size_t(p.in), split_width));
            grid.z = unsigned(parts);
            w.partial.resize(size_t(rows) * size_t(p.out) * size_t(parts));
            fused_projection<true><<<grid, device::wave_size, 0, stream>>>(
                w.input.data(), w.input_scale.data(), p.codes.data(), p.codebook.data(),
                p.dense.data(), p.row_scale.data(), p.codebook_scale, w.partial.data(), rows, p.in,
                p.out, apply_rows, parts, group_major);
            HIP_CHECK(hipGetLastError());
            merge_projection<<<unsigned(ceil_div(size_t(rows) * size_t(p.out), 256)), 256, 0,
                               stream>>>(w.partial.data(), w.input_scale.data(), p.row_scale.data(),
                                         p.codebook_scale, output, rows, p.out, parts, apply_rows);
            HIP_CHECK(hipGetLastError());
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
                    w.input.data(), p.codes.data(), p.codebook.data(), p.dense.data(),
                    w.partial.data(), rows, p.in, p.out, parts, group_major);
            else if (mode == ProjectionMode::Tiled128)
                tiled_projection<128><<<grid, 128, 0, stream>>>(
                    w.input.data(), p.codes.data(), p.codebook.data(), p.dense.data(),
                    w.partial.data(), rows, p.in, p.out, parts, group_major);
            else if (mode == ProjectionMode::TiledLds64)
                tiled_projection<64, true><<<grid, 128, 0, stream>>>(
                    w.input.data(), p.codes.data(), p.codebook.data(), p.dense.data(),
                    w.partial.data(), rows, p.in, p.out, parts, group_major);
            else if (mode == ProjectionMode::TiledLds128)
                tiled_projection<128, true><<<grid, 128, 0, stream>>>(
                    w.input.data(), p.codes.data(), p.codebook.data(), p.dense.data(),
                    w.partial.data(), rows, p.in, p.out, parts, group_major);
            else {
                dispatch_pipeline(mode, [&]<int k, int n, int split, int rt = 1>() {
                    const bool aligned = !p.dense.size() && group_major && p.out % n == 0 &&
                                         p.in % k == 0 && p.codes.size() <= UINT32_MAX;
                    if (aligned)
                        pipelined_projection<k, true, n, split, rt><<<grid, n * 2, 0, stream>>>(
                            w.input.data(), p.codes.data(), p.codebook.data(), p.dense.data(),
                            w.partial.data(), rows, p.in, p.out, group_major);
                    else
                        pipelined_projection<k, false, n, split, rt><<<grid, n * 2, 0, stream>>>(
                            w.input.data(), p.codes.data(), p.codebook.data(), p.dense.data(),
                            w.partial.data(), rows, p.in, p.out, group_major);
                });
            }
            HIP_CHECK(hipGetLastError());
            merge_projection<<<unsigned(ceil_div(size_t(rows) * size_t(p.out), 256)), 256, 0,
                               stream>>>(w.partial.data(), w.input_scale.data(), p.row_scale.data(),
                                         p.codebook_scale, output, rows, p.out, parts, apply_rows);
            HIP_CHECK(hipGetLastError());
        } else if (mode == ProjectionMode::Blas) {
            ensure_blas(state, plan, p.in, p.out, rows);
            if (!plan.has_blas)
                throw ProjectionUnavailable("No native FP8 hipBLASLt algorithm for this shape");
            const uint8_t *weight = p.dense.data();
            if (!weight) {
                w.weight.resize(size_t(p.out) * size_t(p.in));
                reconstruct_kernel<<<unsigned(ceil_div(w.weight.size(), 256)), 256, 0, stream>>>(
                    p.codes.data(), p.codebook.data(), w.weight.data(), p.in, p.out, group_major);
                HIP_CHECK(hipGetLastError());
                weight = w.weight.data();
            }
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
        quantize_rows(input, w.input.data(), w.input_scale.data(), rows, p.in, stream);
        launch(mode);
    };
    if (state.override_mode != ProjectionMode::Auto) {
        execute(state.override_mode);
        return;
    }
    if (!choice.tuned) {
        ensure_blas(state, plan, p.in, p.out, rows);
        std::vector<ProjectionMode> candidates;
        if (!p.dense.size() && rows <= 16) {
            candidates = {ProjectionMode::Pipelined64, ProjectionMode::Pipelined128};
            if (p.in < 1024 || p.out < 512)
                candidates.push_back(ProjectionMode::Fused);
            if (rows >= 4 && plan.has_blas)
                candidates.push_back(ProjectionMode::Blas);
        } else if (!p.dense.size() && rows <= 64) {
            // Lookup cost is shared across 32/64 rows; still compare BLAS.
            candidates = rows <= 32 ? std::vector{ProjectionMode::PipelinedR2K128,
                                                  ProjectionMode::PipelinedR2K64,
                                                  ProjectionMode::PipelinedR4K128}
                                    : std::vector{ProjectionMode::PipelinedR4K128,
                                                  ProjectionMode::PipelinedR4K64};
            if (plan.has_blas)
                candidates.push_back(ProjectionMode::Blas);
            candidates.push_back(ProjectionMode::Fused);
        } else {
            if (plan.has_blas)
                candidates.push_back(ProjectionMode::Blas);
            candidates.push_back(ProjectionMode::Fused);
        }
        if (state.search == ProjectionSearch::Exhaustive && rows <= 64)
            for (auto mode : projection_modes()) {
                if (mode == ProjectionMode::Auto ||
                    (mode == ProjectionMode::Blas && !plan.has_blas) ||
                    (mode == ProjectionMode::SplitK && p.in <= split_width))
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
