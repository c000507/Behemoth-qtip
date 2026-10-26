#include "behemoth/gpu.hpp"

namespace behemoth {
namespace {
// One block per (page, layer): 16896 bytes = 1056 16-byte words (K 512, V 512, K/V scales 16
// each). Page offsets are 8192 / 256 bytes, so every word is 16-byte aligned.
__global__ void kv_pack_kernel(const KvLayerPointers *layers, KvPageList list, uint8_t *staging,
                               int layer_count, bool unpack) {
    const int i = int(blockIdx.x), l = int(blockIdx.y);
    const size_t page = size_t(list.ids[i]);
    const KvLayerPointers p = layers[l];
    uint4 *packed = reinterpret_cast<uint4 *>(
        staging + (size_t(i) * size_t(layer_count) + size_t(l)) * kv_layer_page_bytes);
    uint4 *k = reinterpret_cast<uint4 *>(p.k + page * 8192),
          *v = reinterpret_cast<uint4 *>(p.v + page * 8192),
          *ks = reinterpret_cast<uint4 *>(p.k_scale + page * 64),
          *vs = reinterpret_cast<uint4 *>(p.v_scale + page * 64);
    for (int w = int(threadIdx.x); w < 1056; w += int(blockDim.x)) {
        uint4 *src = w < 512    ? k + w
                     : w < 1024 ? v + (w - 512)
                     : w < 1040 ? ks + (w - 1024)
                                : vs + (w - 1040);
        if (unpack)
            *src = packed[w];
        else
            packed[w] = *src;
    }
}
} // namespace
void kv_pack_pages(const KvLayerPointers *layers, int layer_count, const KvPageList &pages,
                   uint8_t *staging, bool unpack, hipStream_t stream) {
    if (pages.count < 1 || pages.count > kv_transfer_max_pages || layer_count < 1)
        throw std::invalid_argument("Invalid KV page transfer");
    hipLaunchKernelGGL(kv_pack_kernel, dim3(unsigned(pages.count), unsigned(layer_count)),
                       dim3(256), 0, stream, layers, pages, staging, layer_count, unpack);
    HIP_CHECK(hipGetLastError());
}
} // namespace behemoth
