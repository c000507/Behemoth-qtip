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

  public:
    DeviceBuffer<int> offsets, positions, lengths, table;
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
    void upload(const AttentionLayout &l) {
        valid_ = false;
        l.validate();
        host_ = l;
        max_queries_ = max_length_ = 0;
        for (int r = 0; r < l.requests(); ++r) {
            max_queries_ =
                std::max(max_queries_, l.query_offsets[size_t(r + 1)] - l.query_offsets[size_t(r)]);
            max_length_ = std::max(max_length_, l.lengths[size_t(r)]);
        }
        offsets.upload(l.query_offsets);
        positions.upload(l.positions);
        lengths.upload(l.lengths);
        table.upload(l.page_table);
        valid_ = true;
    }
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
    DeviceBuffer<uint16_t> sorted_idx, sorted_segment_codes;
    DeviceBuffer<uint32_t> sorted_planes;
    DeviceBuffer<SortedTileDesc> sorted_tiles;
    int sorted_tile_count = 0, sorted_chunks_per_segment = 0, sorted_delta_bits = 0;
    size_t sorted_entries = 0, sorted_dummies = 0;
    bool has_sorted() const {
        return sorted_tile_count > 0;
    }
    void upload_sorted(const HostProjection &p) {
        auto l = sorted_layout(p.in, p.out, p.codes.data(), sorted_rows_per_tile,
                               sorted_groups_per_tile, 3, sorted_default_chunks_per_segment);
        sorted_idx.upload(l.idx);
        sorted_planes.upload(l.planes);
        sorted_segment_codes.upload(l.segment_codes);
        sorted_tiles.upload(l.tiles);
        sorted_tile_count = int(l.tiles.size());
        sorted_chunks_per_segment = l.chunks_per_segment;
        sorted_delta_bits = l.delta_bits;
        sorted_entries = l.entries;
        sorted_dummies = l.dummies;
    }
    void upload(const HostProjection &p, CodeLayout layout = CodeLayout::GroupMajor) {
        tuning_identity = next_projection_identity();
        in = p.in;
        out = p.out;
        code_layout = layout;
        codes = {};
        sorted_idx = {};
        sorted_planes = {};
        sorted_segment_codes = {};
        sorted_tiles = {};
        sorted_tile_count = 0;
        if (layout == CodeLayout::GroupMajor)
            codes.upload(group_major_codes(p));
        else if (layout == CodeLayout::RowMajor)
            codes.upload(p.codes);
        else
            upload_sorted(p);
        codebook.upload(p.codebook);
        row_scale.upload(p.row_scale);
        codebook_scale = p.codebook_scale;
    }
    size_t bytes() const {
        return codes.bytes() + codebook.bytes() + dense.bytes() + row_scale.bytes() +
               sorted_idx.bytes() + sorted_planes.bytes() + sorted_segment_codes.bytes() +
               sorted_tiles.bytes();
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
    // per batch row, each 32-row stream cut into N segment splits. sorted-dense:
    // stream the sorted layout once into a dense FP8 scratch, then the dense
    // pipelined WMMA kernels (for rows > 2).
    SortedS4,
    SortedS8,
    SortedS16,
    SortedDense,
    Count
};
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
void residual_add(BFloat16 *, const BFloat16 *, size_t n, hipStream_t);
void silu_multiply(const BFloat16 *, const BFloat16 *, BFloat16 *, size_t n, hipStream_t);
void rope(BFloat16 *, const int *positions, int rows, int heads, float theta, hipStream_t);
void embed(const BFloat16 *, const int *tokens, BFloat16 *, int rows, int width, hipStream_t);
void select_rows(const BFloat16 *, const int *, BFloat16 *, int rows, int width, hipStream_t);
} // namespace behemoth
