#pragma once
#include "behemoth/cuda.hpp"
namespace behemoth::cuda_runtime {
class TensorModel {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    TensorModel(const fs::path &, int device, int context, int tp, int max_batch);
    ~TensorModel();
    std::vector<Vec> prefill(const std::vector<std::vector<int>> &);
    std::vector<Vec> step(const std::vector<int> &);
    void reset();
};
} // namespace behemoth::cuda_runtime
