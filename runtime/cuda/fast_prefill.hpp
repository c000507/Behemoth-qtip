#pragma once
#include "kernels.hpp"
#include <memory>
namespace behemoth::cuda_runtime {
class FastPrefill {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    FastPrefill(int qheads, int kvheads, int dimension, int query_stride, Stream);
    ~FastPrefill();
    void run(const B16 *q, const B16 *k, const B16 *v, B16 *out, const int *lengths, int batch, int sequence,
             int context);
};
} // namespace behemoth::cuda_runtime
