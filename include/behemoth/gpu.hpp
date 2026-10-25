#pragma once
#include "attention.hpp"
#include "checkpoint.hpp"
#include "sorted_layout.hpp"
#include "tuning.hpp"
#include <cstring>
#include <hip/hip_runtime.h>
#include <utility>

namespace behemoth {
inline void hip_check(hipError_t e, const char *operation) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(e));
}
#define HIP_CHECK(x) ::behemoth::hip_check((x), #x)
template <class T> class DeviceBuffer {
    T *p_ = nullptr;
    size_t size_ = 0, capacity_ = 0;
    int device_ = 0;
    void release() noexcept {
        if (p_) {
            int current = 0;
            (void)hipGetDevice(&current);
            (void)hipSetDevice(device_);
            (void)hipFree(p_);
            (void)hipSetDevice(current);
        }
        p_ = nullptr;
        size_ = capacity_ = 0;
    }

  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) {
        resize(n);
    }
    ~DeviceBuffer() {
        release();
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept {
        swap(other);
    }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }
    void swap(DeviceBuffer &other) noexcept {
        std::swap(p_, other.p_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(device_, other.device_);
    }
    void resize(size_t n) {
        if (n <= capacity_) {
            size_ = n;
            return;
        }
        release();
        HIP_CHECK(hipGetDevice(&device_));
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&p_), checked_mul(n, sizeof(T))));
        size_ = capacity_ = n;
    }
    void upload(std::span<const T> x) {
        resize(x.size());
        if (!x.empty())
            HIP_CHECK(hipMemcpy(p_, x.data(), x.size_bytes(), hipMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> x(size_);
        if (size_)
            HIP_CHECK(hipMemcpy(x.data(), p_, size_ * sizeof(T), hipMemcpyDeviceToHost));
        return x;
    }
    T *data() {
        return p_;
    }
    const T *data() const {
        return p_;
    }
    size_t size() const {
        return size_;
    }
    size_t bytes() const {
        return size_ * sizeof(T);
    }
};
struct GpuMetadata {
  private:
    AttentionLayout host_;
    int max_queries_ = 0, max_length_ = 0;
    bool valid_ = false;
    // One packed device buffer plus a pinned host mirror: a step uploads
    // offsets/positions/lengths/page-table/tokens with a single async copy
    // instead of five blocking pageable hipMemcpy calls (PERF_ROUND3 item 3).
    DeviceBuffer<int> device_;
    int *staging_ = nullptr;
    size_t staging_capacity_ = 0; // ints
    void ensure_staging(size_t ints, hipStream_t stream);

  public:
    GpuMetadata() = default;
    ~GpuMetadata();
    GpuMetadata(const GpuMetadata &) = delete;
    GpuMetadata &operator=(const GpuMetadata &) = delete;
    // Device pointers of the packed upload (offsets: requests()+1 ints,
    // positions/lengths: requests() ints, table: requests()*table_stride ints,
    // tokens: queries() ints).
    const int *offsets = nullptr, *positions = nullptr, *lengths = nullptr, *table = nullptr,
              *tokens = nullptr;
    const AttentionLayout &layout() const {
        if (!valid_)
            throw std::logic_error("Attention metadata has not been uploaded");
        return host_;
    }
    int max_queries() const {
        return max_queries_;
    }
    int max_length() const {
        return max_length_;
    }
    // Tests / host-side use: blocking upload on the null stream, no tokens.
    void upload(const AttentionLayout &l) {
        upload(l, {}, nullptr);
    }
    // Production: single pinned-staging async copy on the rank stream. The
    // caller must not overwrite the staging (upload again) before the copy
    // completes; stream ordering handles it inside one forward pass.
    void upload(const AttentionLayout &l, std::span<const int> tokens, hipStream_t stream);
};
struct AttentionDeviceView {
    const uint8_t *q{};
    const float *q_scale{};
    uint8_t *k{};
    uint8_t *v{};
    float *k_scale{};
    float *v_scale{};
    BFloat16 *output{};
    float *maximum{};
    float *denominator{}; // optional final statistics for correctness checks
};
size_t attention_workspace_bytes(const AttentionLayout &, int partition_keys = 0);
void paged_attention(const GpuMetadata &, AttentionDeviceView, void *workspace,
                     size_t workspace_bytes, int partition_keys, hipStream_t);
void append_kv(const GpuMetadata &, AttentionDeviceView, const BFloat16 *k, const BFloat16 *v,
               hipStream_t);
void quantize_rows(const BFloat16 *, uint8_t *, float *scales, int rows, int width, hipStream_t);
void fp8_conversion_probe(const float *, uint8_t *, int count, hipStream_t);
void verify_device();

struct GpuProjection {
    uint64_t tuning_identity = next_projection_identity();
    int in = 0, out = 0;
    DeviceBuffer<uint16_t> codes;
    DeviceBuffer<uint8_t> codebook, dense;
    DeviceBuffer<float> row_scale;
    float codebook_scale = 1;
    CodeLayout code_layout = CodeLayout::GroupMajor;
    // Codebook-major ("sorted") layout, see sorted_layout.hpp. Present when
    // sorted_tile_count > 0; with CodeLayout::Sorted it replaces `codes`.
    // 16-row tiles, 4-bit deltas, per-chunk base codes (the kernels never
    // use the segment table).
    // The "lane" (lane-owns-row) variant shares idx/tiles, stores its delta
    // nibbles in sorted_planes and adds per-segment per-lane codes;
    // sorted_lane tells the kernels which encoding the buffers hold
    // (CodeLayout::Lane replaces `codes` too).
    DeviceBuffer<uint16_t> sorted_idx, sorted_chunk_codes, sorted_segment_codes;
    DeviceBuffer<uint32_t> sorted_planes;
    DeviceBuffer<SortedTileDesc> sorted_tiles;
    int sorted_tile_count = 0, sorted_delta_bits = 0;
    bool sorted_lane = false;
    int lane_groups_per_tile = 0, lane_group_tiles = 0;
    size_t sorted_entries = 0, sorted_dummies = 0;
    bool has_sorted() const {
        return sorted_tile_count > 0;
    }
    bool has_lane() const {
        return has_sorted() && sorted_lane;
    }
    size_t sorted_bytes() const {
        return sorted_idx.bytes() + sorted_planes.bytes() + sorted_chunk_codes.bytes() +
               sorted_segment_codes.bytes() + sorted_tiles.bytes();
    }
    void clear_sorted() {
        sorted_idx = {};
        sorted_planes = {};
        sorted_chunk_codes = {};
        sorted_segment_codes = {};
        sorted_tiles = {};
        sorted_tile_count = 0;
        sorted_lane = false;
        lane_groups_per_tile = lane_group_tiles = 0;
        sorted_entries = sorted_dummies = 0;
    }
    void upload_sorted(const HostProjection &p, int delta_bits = sorted_default_delta_bits) {
        clear_sorted();
        auto l = sorted_layout(p.in, p.out, p.codes.data(), sorted_rows_per_tile,
                               sorted_groups_per_tile, delta_bits, sorted_default_chunks_per_segment);
        sorted_idx.upload(l.idx);
        sorted_planes.upload(l.planes);
        sorted_chunk_codes.upload(l.chunk_codes);
        sorted_tiles.upload(l.tiles);
        sorted_tile_count = int(l.tiles.size());
        sorted_delta_bits = l.delta_bits;
        sorted_entries = l.entries;
        sorted_dummies = l.dummies;
    }
    void upload_lane(const HostProjection &p, int max_groups_per_tile = lane_max_groups_per_tile,
                     int drift_limit = 0) {
        clear_sorted();
        auto l = lane_layout(p.in, p.out, p.codes.data(), max_groups_per_tile, drift_limit);
        sorted_idx.upload(l.idx);
        sorted_planes.upload(l.hi);
        sorted_segment_codes.upload(l.segment_codes);
        sorted_tiles.upload(l.tiles);
        sorted_tile_count = int(l.tiles.size());
        sorted_delta_bits = lane_delta_bits;
        sorted_lane = true;
        lane_groups_per_tile = l.groups_per_tile;
        lane_group_tiles = l.group_tiles;

        sorted_entries = l.entries;
        sorted_dummies = l.dummies;
    }
    void upload(const HostProjection &p, CodeLayout layout = CodeLayout::GroupMajor) {
        tuning_identity = next_projection_identity();
        in = p.in;
        out = p.out;
        code_layout = layout;
        codes = {};
        clear_sorted();
        if (layout == CodeLayout::GroupMajor)
            codes.upload(group_major_codes(p));
        else if (layout == CodeLayout::RowMajor)
            codes.upload(p.codes);
        else if (layout == CodeLayout::Sorted)
            upload_sorted(p);
        else
            upload_lane(p);
        codebook.upload(p.codebook);
        row_scale.upload(p.row_scale);
        codebook_scale = p.codebook_scale;
    }
    size_t bytes() const {
        return codes.bytes() + codebook.bytes() + dense.bytes() + row_scale.bytes() +
               sorted_bytes();
    }
};
struct ProjectionWorkspace {
    DeviceBuffer<uint8_t> input, weight;
    DeviceBuffer<float> input_scale, partial;
    DeviceBuffer<std::byte> blas;
};
enum class ProjectionMode {
    Auto,
    Fused,
    SplitK,
    Blas,
    Tiled64,
    Tiled128,
    Pipelined64,
    Pipelined128,
    TiledLds64,
    TiledLds128,
    PipelinedN32K64,
    PipelinedN32K128,
    PipelinedN128K64,
    PipelinedN128K128,
    PipelinedSplit512K64,
    PipelinedSplit512K128,
    PipelinedSplit2048K64,
    PipelinedSplit2048K128,
    PipelinedR2K64,
    PipelinedR2K128,
    PipelinedR4K64,
    PipelinedR4K128,
    // Codebook-major layout (GpuProjection::has_sorted()). sorted-sN: one pass
    // per batch row (exact int64 accumulation), each 16-row tile stream cut
    // into N chunk splits. sorted-lds: per tile, decode the stream into an LDS
    // code tile, then gather + WMMA over up to 64 batch rows per pass (the
    // batched decode kernel). sorted-dense: decode + coalesced dense FP8
    // scratch, then the dense pipelined WMMA kernels (prefill-sized batches).
    SortedS4,
    SortedS8,
    SortedS16,
    SortedDense,
    SortedLds,
    // Lane layout (GpuProjection::has_lane()): register accumulation, up to 4
    // batch rows per pass (passes = ceil(rows / 4)), lane streams cut into N
    // segment splits. sorted-lds / sorted-dense decode either encoding.
    LaneS1,
    LaneS2,
    LaneS4,
    LaneS8,
    Count
};
// Layout the mode needs: 0 none (codes/dense), 1 sorted, 2 lane, 3 either.
int projection_mode_layout(ProjectionMode);
bool projection_mode_is_sorted(ProjectionMode);
std::span<const ProjectionMode> projection_modes();
ProjectionMode parse_projection_mode(std::string_view);
const char *projection_mode_name(ProjectionMode);
enum class ProjectionSearch { Conservative, Exhaustive };
ProjectionSearch parse_projection_search(std::string_view);
const char *projection_search_name(ProjectionSearch);
struct ProjectionUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};
class ProjectionRunner {
    void *handle_ = nullptr;

  public:
    ProjectionRunner();
    ~ProjectionRunner();
    ProjectionRunner(const ProjectionRunner &) = delete;
    void set_mode(ProjectionMode);
    void set_search(ProjectionSearch);
    nlohmann::json profile(bool detailed = true) const;
    nlohmann::json selection(const GpuProjection &, int rows, bool apply_row_scale = true) const;
    nlohmann::json resources() const;
    void run(const GpuProjection &, const BFloat16 *, float *output, int rows,
             ProjectionWorkspace &, hipStream_t, bool apply_row_scale = true);
    // Fused variants (PERF_ROUND2_REPORT.md): the quantised activations may be
    // produced directly into the workspace (prepare_input + rms_norm_quantized),
    // shared by several projections of the same input, and the result may be
    // written as bf16 by the merge kernel instead of a separate cast.
    struct Call {
        const BFloat16 *input = nullptr; // nullptr: workspace already holds rows x in fp8 + scales
        float *output = nullptr;         // fp32 result; also scratch when bf16_output is set
        BFloat16 *bf16_output = nullptr; // optional bf16 result
        bool apply_row_scale = true;
    };
    void run(const GpuProjection &, const Call &, int rows, ProjectionWorkspace &, hipStream_t);
    // Sizes/pads the quantised-input buffers for rows x in and returns them.
    static std::pair<uint8_t *, float *> prepare_input(ProjectionWorkspace &, int rows, int in,
                                                      hipStream_t);
};
void cast_scaled(const float *, const float *row_scale, BFloat16 *, int rows, int width,
                 hipStream_t);
// Stream-ordered two-rank exchange through fine-grained pinned host memory (transport.hip).
void zero_copy_publish(const float *values, float *mine_host_device_pointer, size_t n,
                       unsigned *flag_device_pointer, unsigned *counter, unsigned seq, hipStream_t);
void zero_copy_reduce(float *values, const float *peer_host_device_pointer, size_t n,
                      const unsigned *peer_flag_device_pointer, unsigned seq, hipStream_t);
void zero_copy_gather(const float *local, const float *peer_host_device_pointer, float *all,
                      size_t n, int rank, const unsigned *peer_flag_device_pointer, unsigned seq,
                      hipStream_t);
void rms_norm(const BFloat16 *, const BFloat16 *, BFloat16 *, int rows, int width, float epsilon,
              hipStream_t);
// rms_norm followed by per-row fp8 quantisation (identical rounding to
// rms_norm + quantize_rows), written straight into projection input buffers.
void rms_norm_quantized(const BFloat16 *, const BFloat16 *w, uint8_t *q, float *scales, int rows,
                        int width, float epsilon, hipStream_t);
void residual_add(BFloat16 *, const BFloat16 *, size_t n, hipStream_t);
// state = bf16(state + bf16(x * row_scale[col])): cast_scaled + residual_add in one pass.
void residual_add_scaled(BFloat16 *state, const float *x, const float *row_scale, int rows,
                         int width, hipStream_t);
// rope on q (heads x 128) and k (kv_heads x 128) plus per-head fp8 quantisation of
// the rotated q: replaces rope(q), rope(k), quantize_rows(q).
void rope_quantize(BFloat16 *q, BFloat16 *k, const int *positions, int rows, int heads,
                   int kv_heads, float theta, uint8_t *q8, float *q_scale, hipStream_t);
void silu_multiply(const BFloat16 *, const BFloat16 *, BFloat16 *, size_t n, hipStream_t);
void rope(BFloat16 *, const int *positions, int rows, int heads, float theta, hipStream_t);
void embed(const BFloat16 *, const int *tokens, BFloat16 *, int rows, int width, hipStream_t);
void select_rows(const BFloat16 *, const int *, BFloat16 *, int rows, int width, hipStream_t);
} // namespace behemoth
