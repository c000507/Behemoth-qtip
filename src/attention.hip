#include "behemoth/gpu.hpp"
#include "device_numeric.hpp"
#include <algorithm>
#include <cstring>
#include <rocwmma/rocwmma.hpp>
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
static_assert(rocwmma::Constants::AMDGCN_WAVE_SIZE == 32, "rocWMMA must use Wave32");
#endif

namespace behemoth {
namespace {
using F8 = rocwmma::float8_t;
static_assert(sizeof(F8) == 1);
__device__ F8 f8(uint8_t x) {
    return __builtin_bit_cast(F8, x);
}
struct Meta {
    int qh, kh, pages, stride, requests, queries, max_q;
    const int *offsets, *pos, *len, *table;
};
Meta meta(const GpuMetadata &m) {
    const auto &l = m.layout();
    return {l.query_heads,      l.kv_heads,       l.physical_pages, l.table_stride,
            l.requests(),       l.queries(),      m.max_queries(),  m.offsets,
            m.positions,        m.lengths,        m.table};
}
__global__ void attention_kernel(Meta m, AttentionDeviceView a, float *partial, int parts,
                                  int partition_keys) {
    const int lane = int(threadIdx.x), req = int(blockIdx.x), head = int(blockIdx.y),
              part = int(blockIdx.z) % parts;
    const int tile = int(blockIdx.z) / parts, first = m.offsets[req] + tile * 16,
              stop = m.offsets[req + 1], kh = head / 12;
    if (first >= stop)
        return;
    __shared__ __attribute__((aligned(16))) F8 q[16 * 128], k[32 * 128], v[32 * 128], p[16 * 32];
    __shared__ float scores[16 * 32], product[16 * 128], out[16 * 128], mx[16], den[16], alpha[16],
        pscale[16], ks[32], vs[32];
    for (int i = lane; i < 16 * 128; i += 32) {
        int row = i / 128, d = i % 128, g = first + row;
        q[i] = f8(g < stop ? a.q[(size_t(g) * size_t(m.qh) + size_t(head)) * 128 + size_t(d)] : 0);
        out[i] = 0;
    }
    if (lane < 16) {
        mx[lane] = -INFINITY;
        den[lane] = 0;
    }
    __syncthreads();
    int begin = partition_keys ? part * partition_keys : 0;
    int end = partition_keys ? min(m.len[req], begin + partition_keys) : m.len[req];
    for (int base = begin; base < end; base += 32) {
        for (int i = lane; i < 32 * 128; i += 32) {
            int t = base + i / 128, d = i % 128;
            bool valid = t < end;
            int page = valid ? m.table[req * m.stride + t / 16] : 0;
            size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
            k[i] = f8(valid ? a.k[loc * 128 + size_t(d)] : 0);
            v[i] = f8(valid ? a.v[loc * 128 + size_t(d)] : 0);
        }
        int t = base + lane;
        bool valid = t < end;
        int page = valid ? m.table[req * m.stride + t / 16] : 0;
        size_t loc = (size_t(page) * size_t(m.kh) + size_t(kh)) * 16 + size_t(t % 16);
        ks[lane] = valid ? a.k_scale[loc] : 1;
        vs[lane] = valid ? a.v_scale[loc] : 1;
        __syncthreads();
        // Two 16-column score fragments, four K=32 products over head_dim=128.
        for (int n = 0; n < 32; n += 16) {
            rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> c;
            rocwmma::fill_fragment(c, 0.f);
            for (int d = 0; d < 128; d += 32) {
                rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major> qa;
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::col_major> kb;
                rocwmma::load_matrix_sync(qa, q + d, 128);
                rocwmma::load_matrix_sync(kb, k + n * 128 + d, 128);
                rocwmma::mma_sync(c, qa, kb, c);
            }
            rocwmma::store_matrix_sync(scores + n, c, 32, rocwmma::mem_row_major);
        }
        __syncthreads();
        if (lane < 16) {
            int g = first + lane;
            float maximum = -INFINITY;
            for (int j = 0; j < 32; ++j) {
                bool keep = g < stop && base + j < end && base + j <= m.pos[g];
                float score = keep ? scores[lane * 32 + j] *
                                         a.q_scale[size_t(g) * size_t(m.qh) + size_t(head)] *
                                         ks[j] * 0.08838834764831845f
                                   : -INFINITY;
                scores[lane * 32 + j] = score;
                maximum = fmaxf(maximum, score);
            }
            float next = fmaxf(mx[lane], maximum);
            alpha[lane] = isfinite(mx[lane]) ? expf(mx[lane] - next) : 0;
            float sum = den[lane] * alpha[lane], weighted_max = 0;
            for (int j = 0; j < 32; ++j) {
                float probability =
                    isfinite(scores[lane * 32 + j]) ? expf(scores[lane * 32 + j] - next) : 0;
                sum += probability;
                scores[lane * 32 + j] = probability * vs[j];
                weighted_max = fmaxf(weighted_max, scores[lane * 32 + j]);
            }
            pscale[lane] = device::scale(weighted_max);
            for (int j = 0; j < 32; ++j)
                p[lane * 32 + j] = f8(device::fp8(scores[lane * 32 + j] / pscale[lane]));
            mx[lane] = next;
            den[lane] = sum;
        }
        __syncthreads();
        for (int n = 0; n < 128; n += 16) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major> pa;
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::row_major> vb;
            rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> c;
            rocwmma::fill_fragment(c, 0.f);
            rocwmma::load_matrix_sync(pa, p, 32);
            rocwmma::load_matrix_sync(vb, v + n, 128);
            rocwmma::mma_sync(c, pa, vb, c);
            rocwmma::store_matrix_sync(product + n, c, 128, rocwmma::mem_row_major);
        }
        __syncthreads();
        for (int i = lane; i < 16 * 128; i += 32)
            out[i] = out[i] * alpha[i / 128] + product[i] * pscale[i / 128];
        __syncthreads();
    }
    for (int i = lane; i < 16 * 128; i += 32) {
        int g = first + i / 128;
        if (g >= stop)
            continue;
        size_t row = size_t(g) * size_t(m.qh) + size_t(head);
        if (parts == 1)
            reinterpret_cast<uint16_t *>(a.output)[row * 128 + size_t(i % 128)] =
                device::bf16(den[i / 128] > 0 ? out[i] / den[i / 128] : 0);
        else
            partial[(row * size_t(parts) + size_t(part)) * 130 + 2 + size_t(i % 128)] = out[i];
    }
    if (lane < 16 && first + lane < stop) {
        size_t row = size_t(first + lane) * size_t(m.qh) + size_t(head);
        if (parts == 1) {
            if (a.maximum)
                a.maximum[row] = mx[lane];
            if (a.denominator)
                a.denominator[row] = den[lane];
        } else {
            size_t i = (row * size_t(parts) + size_t(part)) * 130;
            partial[i] = mx[lane];
            partial[i + 1] = den[lane];
        }
    }
}
__global__ void merge_kernel(AttentionDeviceView a, const float *partial, int rows, int parts) {
    int row = int(blockIdx.x), d = int(threadIdx.x);
    if (row >= rows || d >= 128)
        return;
    float maximum = -INFINITY;
    for (int p = 0; p < parts; ++p)
        maximum = fmaxf(maximum, partial[(size_t(row) * size_t(parts) + size_t(p)) * 130]);
    float sum = 0, out = 0;
    for (int p = 0; p < parts; ++p) {
        size_t i = (size_t(row) * size_t(parts) + size_t(p)) * 130;
        float weight = isfinite(partial[i]) ? expf(partial[i] - maximum) : 0;
        sum += partial[i + 1] * weight;
        out += partial[i + 2 + size_t(d)] * weight;
    }
    reinterpret_cast<uint16_t *>(a.output)[size_t(row) * 128 + size_t(d)] =
        device::bf16(sum > 0 ? out / sum : 0);
    if (d == 0) {
        if (a.maximum)
            a.maximum[row] = maximum;
        if (a.denominator)
            a.denominator[row] = sum;
    }
}
__global__ void append_kernel(Meta m, AttentionDeviceView a, const uint16_t *kin,
                              const uint16_t *vin) {
    int q = int(blockIdx.x), h = int(blockIdx.y), lane = int(threadIdx.x), req = 0;
    while (req + 1 < m.requests && q >= m.offsets[req + 1])
        ++req;
    int t = m.pos[q], page = m.table[req * m.stride + t / 16];
    size_t dest = (size_t(page) * size_t(m.kh) + size_t(h)) * 16 + size_t(t % 16),
           src = (size_t(q) * size_t(m.kh) + size_t(h)) * 128;
    float k = lane < 128 ? device::bf16(kin[src + size_t(lane)]) : 0,
          v = lane < 128 ? device::bf16(vin[src + size_t(lane)]) : 0;
    __shared__ float km[128], vm[128];
    km[lane] = fabsf(k);
    vm[lane] = fabsf(v);
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (lane < s) {
            km[lane] = fmaxf(km[lane], km[lane + s]);
            vm[lane] = fmaxf(vm[lane], vm[lane + s]);
        }
        __syncthreads();
    }
    float sk = device::scale(km[0]), sv = device::scale(vm[0]);
    a.k[dest * 128 + size_t(lane)] = device::fp8(k / sk);
    a.v[dest * 128 + size_t(lane)] = device::fp8(v / sv);
    if (lane == 0) {
        a.k_scale[dest] = sk;
        a.v_scale[dest] = sv;
    }
}
__global__ void quantize_kernel(const uint16_t *input, uint8_t *out, float *scales, int width) {
    int row = int(blockIdx.x), lane = int(threadIdx.x);
    __shared__ float maximum[256];
    float m = 0;
    for (int i = lane; i < width; i += 256)
        m = fmaxf(m, fabsf(device::bf16(input[size_t(row) * size_t(width) + size_t(i)])));
    maximum[lane] = m;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s)
            maximum[lane] = fmaxf(maximum[lane], maximum[lane + s]);
        __syncthreads();
    }
    float scale = device::scale(maximum[0]);
    if (lane == 0)
        scales[row] = scale;
    for (int i = lane; i < width; i += 256)
        out[size_t(row) * size_t(width) + size_t(i)] =
            device::fp8(device::bf16(input[size_t(row) * size_t(width) + size_t(i)]) / scale);
}
__global__ void conversion_kernel(const float *x, uint8_t *y, int n) {
    int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        y[i] = device::fp8(x[i]);
}
__global__ void matrix_probe(float *result) {
    __shared__ __attribute__((aligned(16))) F8 a[16 * 32], b[32 * 16];
    int lane = int(threadIdx.x);
    for (int i = lane; i < 512; i += 32) {
        a[i] = f8(device::fp8(1.f));
        b[i] = f8(device::fp8(1.f));
    }
    __syncthreads();
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, F8, rocwmma::row_major> ma;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, F8, rocwmma::row_major> mb;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, float> c;
    rocwmma::fill_fragment(c, 0.f);
    rocwmma::load_matrix_sync(ma, a, 32);
    rocwmma::load_matrix_sync(mb, b, 16);
    rocwmma::mma_sync(c, ma, mb, c);
    rocwmma::store_matrix_sync(result, c, 16, rocwmma::mem_row_major);
}
} // namespace
void verify_device() {
    hipDeviceProp_t p{};
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&p, device));
    if (std::string(p.gcnArchName).find("gfx1201") != 0 || p.warpSize != 32)
        throw std::runtime_error(
            "Native FP8 runtime requires gfx1201 / wave32; no precision fallback");
    DeviceBuffer<float> result(256);
    matrix_probe<<<1, 32>>>(result.data());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (float v : result.download())
        if (v != 32.f)
            throw std::runtime_error("Native FP8 matrix-instruction smoke test failed");
    std::array<float, 10> source{448,
                                 1000,
                                 -0.f,
                                 1.0625f,
                                 INFINITY,
                                 -INFINITY,
                                 std::numeric_limits<float>::max(),
                                 -std::numeric_limits<float>::max(),
                                 std::bit_cast<float>(0x7fc00000U),
                                 std::bit_cast<float>(0xffc00000U)};
    DeviceBuffer<float> input;
    input.upload(source);
    DeviceBuffer<uint8_t> encoded(source.size());
    fp8_conversion_probe(input.data(), encoded.data(), int(source.size()), nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto values = encoded.download();
    for (size_t i = 0; i < source.size(); ++i)
        if (values[i] != encode_e4m3(source[i]))
            throw std::runtime_error("OCP E4M3 smoke test failed (saturation/signed zero/NaN/RNE)");
}
GpuMetadata::~GpuMetadata() {
    if (staging_)
        (void)hipHostFree(staging_);
}
void GpuMetadata::ensure_staging(size_t ints, hipStream_t) {
    if (ints <= staging_capacity_)
        return;
    if (staging_)
        (void)hipHostFree(staging_);
    staging_capacity_ = ints * 2; // one-time headroom for typical batch growth
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&staging_),
                            staging_capacity_ * sizeof(int), hipHostMallocPortable));
}
void GpuMetadata::upload(const AttentionLayout &l, std::span<const int> tokens_in,
                         hipStream_t stream) {
    valid_ = false;
    l.validate();
    host_ = l;
    max_queries_ = max_length_ = 0;
    for (int r = 0; r < l.requests(); ++r) {
        max_queries_ =
            std::max(max_queries_, l.query_offsets[size_t(r + 1)] - l.query_offsets[size_t(r)]);
        max_length_ = std::max(max_length_, l.lengths[size_t(r)]);
    }
    const size_t r = size_t(l.requests()), queries = size_t(l.queries()),
                 stride = size_t(l.table_stride);
    const size_t ints = (r + 1) + queries + r + r * stride + queries;
    ensure_staging(ints, stream);
    device_.resize(ints);
    size_t o = 0;
    auto put = [&](const int *src, size_t n) {
        if (n)
            std::memcpy(staging_ + o, src, n * sizeof(int));
        o += n;
    };
    put(l.query_offsets.data(), r + 1);
    put(l.positions.data(), queries);
    put(l.lengths.data(), r);
    put(l.page_table.data(), r * stride);
    put(tokens_in.data(), tokens_in.size());
    offsets = device_.data();
    positions = offsets + (r + 1);
    lengths = positions + queries;
    table = lengths + r;
    tokens = table + r * stride;
    if (ints) {
        if (stream)
            HIP_CHECK(hipMemcpyAsync(device_.data(), staging_, ints * sizeof(int),
                                     hipMemcpyHostToDevice, stream));
        else
            HIP_CHECK(hipMemcpy(device_.data(), staging_, ints * sizeof(int),
                                hipMemcpyHostToDevice));
    }
    valid_ = true;
}
size_t attention_workspace_bytes(const AttentionLayout &l, int partition) {
    l.validate();
    if (partition < 0 || partition % 32)
        throw std::invalid_argument("Partition must be a multiple of 32");
    int parts = partition
                    ? int(ceil_div(size_t(*std::max_element(l.lengths.begin(), l.lengths.end())),
                                   size_t(partition)))
                    : 1;
    return parts == 1 ? 0
                      : checked_mul(
                            checked_mul(size_t(l.queries()) * size_t(l.query_heads), size_t(parts)),
                            130 * sizeof(float));
}
void paged_attention(const GpuMetadata &metadata, AttentionDeviceView a, void *workspace,
                     size_t bytes, int partition, hipStream_t stream) {
    const auto &l = metadata.layout();
    if (partition < 0 || partition % 32)
        throw std::invalid_argument("Partition must be a multiple of 32");
    int parts = partition ? int(ceil_div(size_t(metadata.max_length()), size_t(partition))) : 1;
    size_t needed =
        parts == 1
            ? 0
            : checked_mul(checked_mul(size_t(l.queries()) * size_t(l.query_heads), size_t(parts)),
                          130 * sizeof(float));
    if (!a.q || !a.q_scale || !a.k || !a.v || !a.k_scale || !a.v_scale || !a.output ||
        (needed && (!workspace || bytes < needed)))
        throw std::invalid_argument("Missing/undersized attention buffers");
    if (!l.queries())
        return;
    auto m = meta(metadata);
    dim3 grid(unsigned(m.requests), unsigned(m.qh),
              unsigned(ceil_div(size_t(m.max_q), 16)) * unsigned(parts));
    attention_kernel<<<grid, 32, 0, stream>>>(m, a, static_cast<float *>(workspace), parts,
                                              partition);
    HIP_CHECK(hipGetLastError());
    if (parts > 1) {
        merge_kernel<<<unsigned(m.queries * m.qh), 128, 0, stream>>>(
            a, static_cast<float *>(workspace), m.queries * m.qh, parts);
        HIP_CHECK(hipGetLastError());
    }
}
void append_kv(const GpuMetadata &metadata, AttentionDeviceView a, const BFloat16 *k,
               const BFloat16 *v, hipStream_t stream) {
    (void)metadata.layout(); // validated once per batch, rather than 88 times per token
    if (!k || !v || !a.k || !a.v || !a.k_scale || !a.v_scale)
        throw std::invalid_argument("Missing KV append buffers");
    auto m = meta(metadata);
    if (!m.queries)
        return;
    append_kernel<<<dim3(unsigned(m.queries), unsigned(m.kh)), 128, 0, stream>>>(
        m, a, reinterpret_cast<const uint16_t *>(k), reinterpret_cast<const uint16_t *>(v));
    HIP_CHECK(hipGetLastError());
}
void quantize_rows(const BFloat16 *x, uint8_t *y, float *s, int rows, int width,
                   hipStream_t stream) {
    if (rows < 0 || width < 1 || !x || !y || !s)
        throw std::invalid_argument("Invalid quantization buffers/dimensions");
    if (rows) {
        quantize_kernel<<<unsigned(rows), 256, 0, stream>>>(reinterpret_cast<const uint16_t *>(x),
                                                            y, s, width);
        HIP_CHECK(hipGetLastError());
    }
}
void fp8_conversion_probe(const float *x, uint8_t *y, int n, hipStream_t stream) {
    if (n > 0) {
        conversion_kernel<<<unsigned(ceil_div(size_t(n), 256)), 256, 0, stream>>>(x, y, n);
        HIP_CHECK(hipGetLastError());
    }
}
} // namespace behemoth
